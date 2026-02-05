"""Tools service for MCP utilities and data access.

This service module provides utility functions for MCP (Model Context Protocol) 
tools including project listings and document type information access.

The service provides:
1. Simple project listing without processing statistics 
2. Document type lookups and metadata
"""

import json
import logging
import psycopg
import uuid
from typing import List, Dict, Any, Optional
from flask import current_app
from utils.document_types import (
    get_all_document_types, 
    get_document_type, 
    get_all_document_type_aliases,
    get_document_type_aliases
)


class ToolsService:
    """Tools service for MCP utilities and data access.
    
    This service class provides utility functions for external tools and systems
    that need access to project listings and document type information without
    the overhead of full processing statistics.
    """

    @staticmethod
    def _classify_document_type_act(type_id: str) -> str:
        """Classify a document type ID as 2002 Act or 2018 Act terms.
        
        Args:
            type_id (str): The document type ID
            
        Returns:
            str: "2002_act_terms" or "2018_act_terms"
        """
        if type_id.startswith("5cf00c03") or type_id.startswith("5d0d212c"):
            return "2002_act_terms"
        elif type_id.startswith("5df79dd7") or type_id.startswith("5dfc209b"):
            return "2018_act_terms"
        else:
            # Default fallback, though this shouldn't happen with current data
            return "unknown_act"

    @classmethod
    def get_projects_list(cls) -> Dict[str, Any]:
        """Retrieve a simple list of all projects.
        
        This method queries the projects table to return a lightweight list
        of all projects with their basic information (ID and name only).
        
        Returns:
            dict: A structured response containing project list:
                {
                    "projects": [
                        {
                            "project_id": "uuid-string",
                            "project_name": "Project Name"
                        },
                        ...
                    ],
                    "total_projects": 5
                }
        """
        
        try:
            # Simple SQL query to get all projects
            projects_query = """
            SELECT 
                project_id,
                project_name,
                project_metadata
            FROM projects
            ORDER BY project_name;
            """
            
            logging.info("Executing projects list query")
            
            # Execute the query
            conn_params = current_app.vector_settings.database_url
            with psycopg.connect(conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(projects_query)
                    results = cur.fetchall()
            
            # Process the results
            projects_list = []

            for row in results:
                project_id, project_name, project_metadata = row
                # Log sample metadata structure for debugging
                if project_metadata and project_name and "brucejack" in project_name.lower():
                    meta_type = type(project_metadata).__name__
                    meta_keys = list(project_metadata.keys()) if isinstance(project_metadata, dict) else "NOT_A_DICT"
                    has_desc = bool(project_metadata.get("description")) if isinstance(project_metadata, dict) else False
                    has_status = bool(project_metadata.get("status")) if isinstance(project_metadata, dict) else False
                    has_phase = bool(project_metadata.get("currentPhaseName")) if isinstance(project_metadata, dict) else False
                    logging.info(f"PROJECT METADATA DEBUG [{project_name}]: type={meta_type}, keys_sample={str(meta_keys)[:200]}, has_description={has_desc}, has_status={has_status}, has_currentPhaseName={has_phase}")
                projects_list.append({
                    "project_id": project_id,
                    "project_name": project_name,
                    "project_metadata": project_metadata
                })
            
            response = {
                "projects": projects_list,
                "total_projects": len(projects_list)
            }
            
            logging.info(f"Retrieved {len(projects_list)} projects")
            return response
            
        except Exception as e:
            logging.error(f"Error retrieving projects list: {e}")
            return {
                "projects": [],
                "total_projects": 0,
                "error": str(e)
            }

    @classmethod
    def get_document_types(cls) -> Dict[str, Any]:
        """Retrieve all document types with their metadata.
        
        This method returns the complete document type mappings including
        both 2002 Act and 2018 Act terms, along with their aliases for
        inference and lookup purposes.
        
        Returns:
            dict: A structured response containing document types:
                {
                    "document_types": {
                        "type_id": {
                            "name": "Human Readable Name",
                            "aliases": ["alias1", "alias2", ...],
                            "act": "2002_act_terms" | "2018_act_terms"
                        },
                        ...
                    },
                    "lookup_only": {
                        "type_id": "Human Readable Name",
                        ...
                    },
                    "total_types": 42,
                    "act_breakdown": {
                        "2002_act_terms": 20,
                        "2018_act_terms": 22
                    }
                }
        """
        
        try:
            # Get all document types and aliases
            all_types = get_all_document_types()
            all_aliases = get_all_document_type_aliases()
            
            # Build comprehensive response
            document_types = {}
            
            for type_id, type_name in all_types.items():
                aliases = get_document_type_aliases(type_id)
                act_classification = cls._classify_document_type_act(type_id)
                document_types[type_id] = {
                    "name": type_name,
                    "aliases": aliases,
                    "act": act_classification
                }
            
            # Count by Act using the classification function
            act_2002_count = 0
            act_2018_count = 0
            
            for type_id in all_types.keys():
                act_classification = cls._classify_document_type_act(type_id)
                if act_classification == "2002_act_terms":
                    act_2002_count += 1
                elif act_classification == "2018_act_terms":
                    act_2018_count += 1
            
            response = {
                "document_types": document_types,
                "lookup_only": all_types,  # Simple ID -> name mapping
                "total_types": len(all_types),
                "act_breakdown": {
                    "2002_act_terms": act_2002_count,
                    "2018_act_terms": act_2018_count
                }
            }
            
            logging.info(f"Retrieved {len(all_types)} document types")
            return response
            
        except Exception as e:
            logging.error(f"Error retrieving document types: {e}")
            return {
                "document_types": {},
                "lookup_only": {},
                "total_types": 0,
                "act_breakdown": {
                    "2002_act_terms": 0,
                    "2018_act_terms": 0
                },
                "error": str(e)
            }

    @classmethod
    def get_document_type_by_id(cls, type_id: str) -> Dict[str, Any]:
        """Retrieve a specific document type by ID.
        
        Args:
            type_id (str): The document type ID to look up
            
        Returns:
            dict: Document type information or error:
                {
                    "document_type": {
                        "id": "type_id",
                        "name": "Human Readable Name",
                        "aliases": ["alias1", "alias2", ...],
                        "act": "2002_act_terms" | "2018_act_terms"
                    }
                }
        """
        
        try:
            type_name = get_document_type(type_id)
            
            if type_name == "Unknown":
                return {
                    "document_type": None,
                    "error": f"Document type ID '{type_id}' not found"
                }
            
            aliases = get_document_type_aliases(type_id)
            act_classification = cls._classify_document_type_act(type_id)
            
            response = {
                "document_type": {
                    "id": type_id,
                    "name": type_name,
                    "aliases": aliases,
                    "act": act_classification
                }
            }
            
            logging.info(f"Retrieved document type for ID: {type_id}")
            return response
            
        except Exception as e:
            logging.error(f"Error retrieving document type {type_id}: {e}")
            return {
                "document_type": None,
                "error": str(e)
            }
    @classmethod
    def create_feedback_session(
        cls,
        user_id: Optional[str],
        query_text: str,
        project_ids: Optional[List[str]] = None,
        document_type_ids: Optional[List[str]] = None,
        search_result: Optional[dict] = None
    ) -> str:
        try:
            session_id = str(uuid.uuid4())

            insert_query = """
                INSERT INTO search_feedback
                (session_id, user_id, query_text, project_ids, document_type_ids, search_result)
                VALUES (%s, %s, %s, %s, %s, %s);
            """

            conn_params = current_app.vector_settings.database_url
            with psycopg.connect(conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        insert_query,
                        (
                            session_id,
                            user_id,
                            query_text,
                            project_ids,
                            document_type_ids,
                            json.dumps(search_result) if search_result else None
                        )
                    )

            logging.info(f"Created feedback session {session_id}")
            return session_id

        except Exception as e:
            logging.error(f"Error creating feedback session: {e}")
            return None

    @classmethod
    def update_feedback(
        cls,
        session_id: str,
        feedback: Optional[str] = None,
        comments: Optional[str] = None,
        summary_helpful: Optional[int] = None,
        summary_accurate: Optional[int] = None,
        doc_helpful: Optional[int] = None,
        doc_accurate: Optional[int] = None,
        summary_improvement: Optional[str] = None,
        doc_improvement: Optional[str] = None,
    ) -> bool:
        """
        Update an existing feedback record based on session_id using raw SQL.

        Returns:
            bool: True if update succeeded, False otherwise
        """
        try:
            fields = []
            values = []

            def add(field_name, value):
                if value is not None:
                    fields.append(f"{field_name} = %s")
                    values.append(value)

            add("feedback", feedback)
            add("comments", comments)
            add("summary_helpful", summary_helpful)
            add("summary_accurate", summary_accurate)
            add("doc_helpful", doc_helpful)
            add("doc_accurate", doc_accurate)
            add("summary_improvement", summary_improvement)
            add("doc_improvement", doc_improvement)

            if not fields:
                logging.warning("No feedback fields provided for update")
                return False

            update_query = f"""
                UPDATE search_feedback
                SET {", ".join(fields)}
                WHERE session_id = %s;
            """

            values.append(session_id)

            conn_params = current_app.vector_settings.database_url
            with psycopg.connect(conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(update_query, values)
                    if cur.rowcount == 0:
                        logging.warning(f"No feedback record found for session {session_id}")
                        return False

            logging.info(f"✅ Updated feedback for session {session_id}")
            return True

        except Exception as e:
            logging.error(f"Error updating feedback for session {session_id}: {e}")
            return False

    @classmethod
    def get_search_strategies(cls) -> Dict[str, Any]:
        """Retrieve all available search strategies with their descriptions.
        
        Returns:
            dict: A structured response containing search strategies:
                {
                    "search_strategies": {
                        "strategy_name": {
                            "name": "HYBRID_SEMANTIC_FALLBACK",
                            "description": "Description...",
                            "use_cases": ["General-purpose queries", ...],
                            "steps": [...]
                        }
                    },
                    "default_strategy": "HYBRID_PARALLEL",
                    "total_strategies": 6
                }
        """
        
        try:
            strategies = {
                "HYBRID_SEMANTIC_FALLBACK": {
                    "name": "HYBRID_SEMANTIC_FALLBACK",
                    "description": "Default strategy implementing document-level filtering followed by semantic search",
                    "use_cases": [
                        "General-purpose queries",
                        "Balanced efficiency and accuracy",
                        "Mixed conceptual and keyword queries"
                    ],
                    "steps": [
                        "Document-Level Keyword Filtering",
                        "Chunk-Level Semantic Search", 
                        "Semantic Fallback",
                        "Keyword Fallback"
                    ],
                    "performance": "Medium",
                    "accuracy": "High"
                },
                "HYBRID_KEYWORD_FALLBACK": {
                    "name": "HYBRID_KEYWORD_FALLBACK",
                    "description": "Similar to default but prioritizes keyword matching",
                    "use_cases": [
                        "Queries with specific technical terms",
                        "Exact phrase matching",
                        "Known terminology searches"
                    ],
                    "steps": [
                        "Document-Level Keyword Filtering",
                        "Chunk-Level Keyword Search",
                        "Keyword Fallback",
                        "Semantic Fallback"
                    ],
                    "performance": "Fast",
                    "accuracy": "High for exact matches"
                },
                "SEMANTIC_ONLY": {
                    "name": "SEMANTIC_ONLY",
                    "description": "Pure semantic search without document-level filtering or keyword fallbacks",
                    "use_cases": [
                        "Conceptual queries",
                        "When exact keyword matches aren't important",
                        "Exploratory searches"
                    ],
                    "steps": [
                        "Direct Semantic Search",
                        "Cross-Encoder Re-ranking"
                    ],
                    "performance": "Medium",
                    "accuracy": "High for concepts"
                },
                "KEYWORD_ONLY": {
                    "name": "KEYWORD_ONLY",
                    "description": "Pure keyword search without semantic components",
                    "use_cases": [
                        "Exact term matching",
                        "Fastest performance",
                        "Queries with specific terminology"
                    ],
                    "steps": [
                        "Direct Keyword Search",
                        "Cross-Encoder Re-ranking"
                    ],
                    "performance": "Fastest",
                    "accuracy": "High for exact terms"
                },
                "HYBRID_PARALLEL": {
                    "name": "HYBRID_PARALLEL",
                    "description": "Comprehensive search running both semantic and keyword approaches simultaneously",
                    "use_cases": [
                        "Maximum recall",
                        "When computational cost is not a concern",
                        "Comprehensive document discovery"
                    ],
                    "steps": [
                        "Parallel Execution (Semantic + Keyword)",
                        "Result Merging",
                        "Cross-Encoder Re-ranking"
                    ],
                    "performance": "Slowest",
                    "accuracy": "Highest"
                },
                "DOCUMENT_ONLY": {
                    "name": "DOCUMENT_ONLY",
                    "description": "Metadata-based document retrieval without semantic or keyword search",
                    "use_cases": [
                        "Generic document browsing requests",
                        "When you need all documents of specific types/projects",
                        "Document listing and discovery",
                        "Fastest retrieval with date-ordered results"
                    ],
                    "steps": [
                        "Direct Metadata Filtering",
                        "Date-Based Ordering (newest first)",
                        "No Re-ranking Required"
                    ],
                    "performance": "Fastest",
                    "accuracy": "Perfect for metadata-based queries"
                }
            }
            
            # Convert dictionary of strategies to an array
            strategies_array = list(strategies.values())
            
            response = {
                "search_strategies": strategies_array,
                "default_strategy": "HYBRID_PARALLEL",
                "total_strategies": len(strategies_array)
            }
            
            logging.info(f"Retrieved {len(strategies_array)} search strategies")
            return response
            
        except Exception as e:
            logging.error(f"Error retrieving search strategies: {e}")
            return {
                "search_strategies": {},
                "default_strategy": None,
                "total_strategies": 0,
                "error": str(e)
            }

    @classmethod
    def get_inference_options(cls) -> Dict[str, Any]:
        """Retrieve all available inference options and configurations.
        
        Returns:
            dict: A structured response containing inference options:
                {
                    "inference_types": [...],
                    "inference_configurations": {...},
                    "environment_settings": {...}
                }
        """
        
        try:
            inference_types = [
                {
                    "type": "PROJECT",
                    "description": "Automatically infer relevant project IDs from query content",
                    "purpose": "Filters search to relevant projects based on query analysis"
                },
                {
                    "type": "DOCUMENTTYPE", 
                    "description": "Automatically infer relevant document type IDs from query content",
                    "purpose": "Filters search to relevant document types based on query analysis"
                }
            ]
            
            configurations = {
                "parameter_values": [
                    {
                        "value": ["PROJECT"],
                        "description": "Only run project inference"
                    },
                    {
                        "value": ["DOCUMENTTYPE"],
                        "description": "Only run document type inference"
                    },
                    {
                        "value": ["PROJECT", "DOCUMENTTYPE"],
                        "description": "Run both inference pipelines"
                    },
                    {
                        "value": [],
                        "description": "Disable all inference pipelines"
                    },
                    {
                        "value": None,
                        "description": "Use USE_DEFAULT_INFERENCE environment setting"
                    }
                ],
                "automatic_skipping": {
                    "description": "Inference is automatically skipped when explicit IDs are provided",
                    "rules": [
                        "If projectIds are provided, PROJECT inference is skipped",
                        "If documentTypeIds are provided, DOCUMENTTYPE inference is skipped"
                    ]
                }
            }
            
            environment_settings = {
                "USE_DEFAULT_INFERENCE": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable default inference pipelines when inference parameter is not provided"
                }
            }
            
            response = {
                "inference_types": inference_types,
                "inference_configurations": configurations,
                "environment_settings": environment_settings,
                "behavior_logic": [
                    "If inference parameter is explicitly provided: Use it exactly as specified",
                    "If inference parameter is null/not provided AND USE_DEFAULT_INFERENCE=true: Run all pipelines",
                    "If inference parameter is null/not provided AND USE_DEFAULT_INFERENCE=false: Run no pipelines"
                ]
            }
            
            logging.info("Retrieved inference options configuration")
            return response
            
        except Exception as e:
            logging.error(f"Error retrieving inference options: {e}")
            return {
                "inference_types": [],
                "inference_configurations": {},
                "environment_settings": {},
                "error": str(e)
            }

    @classmethod
    def get_api_capabilities(cls) -> Dict[str, Any]:
        """Retrieve complete API capabilities and configuration options.
        
        Returns:
            dict: A structured response containing API capabilities:
                {
                    "endpoints": {...},
                    "search_parameters": {...},
                    "ranking_options": {...},
                    "response_formats": {...}
                }
        """
        
        try:
            endpoints = {
                "vector_search": {
                    "endpoint": "POST /api/vector-search",
                    "description": "Primary search functionality with two-stage pipeline",
                    "required_parameters": ["query"],
                    "optional_parameters": ["projectIds", "documentTypeIds", "inference", "searchStrategy", "ranking"]
                },
                "similar_documents": {
                    "endpoint": "POST /api/similar-documents", 
                    "description": "Find documents similar to a reference document",
                    "required_parameters": ["referenceDocumentId"],
                    "optional_parameters": ["projectIds", "documentTypeIds", "topN"]
                },
                "tools_projects": {
                    "endpoint": "GET /api/tools/projects",
                    "description": "Simple project listing for external tools",
                    "parameters": []
                },
                "tools_document_types": {
                    "endpoint": "GET /api/tools/document-types",
                    "description": "Document type information with Act classifications",
                    "parameters": []
                }
            }
            
            search_parameters = {
                "query": {
                    "type": "string",
                    "required": True,
                    "description": "Search query text"
                },
                "projectIds": {
                    "type": "array[string]",
                    "required": False,
                    "description": "Filter to specific project IDs"
                },
                "documentTypeIds": {
                    "type": "array[string]", 
                    "required": False,
                    "description": "Filter to specific document type IDs"
                },
                "searchStrategy": {
                    "type": "string",
                    "required": False,
                    "enum": ["HYBRID_SEMANTIC_FALLBACK", "HYBRID_KEYWORD_FALLBACK", "SEMANTIC_ONLY", "KEYWORD_ONLY", "HYBRID_PARALLEL", "DOCUMENT_ONLY"],
                    "description": "Search strategy to use"
                },
                "inference": {
                    "type": "array[string]",
                    "required": False,
                    "enum": [["PROJECT"], ["DOCUMENTTYPE"], ["PROJECT", "DOCUMENTTYPE"], []],
                    "description": "Inference pipelines to run"
                }
            }
            
            ranking_options = {
                "minScore": {
                    "type": "float",
                    "description": "Minimum relevance score threshold",
                    "notes": "Can be negative; lower values are more inclusive"
                },
                "topN": {
                    "type": "integer",
                    "range": "1-100",
                    "description": "Maximum number of results to return"
                }
            }
            
            response_formats = {
                "document_level": {
                    "key": "documents",
                    "description": "Document-level results ordered by date",
                    "when": "Direct metadata search mode"
                },
                "chunk_level": {
                    "key": "document_chunks", 
                    "description": "Document chunk results ranked by semantic relevance",
                    "when": "Semantic search mode"
                }
            }
            
            response = {
                "endpoints": endpoints,
                "search_parameters": search_parameters,
                "ranking_options": ranking_options,
                "response_formats": response_formats,
                "api_version": "1.0",
                "features": [
                    "Two-stage search pipeline",
                    "Multiple search strategies",
                    "Automatic inference",
                    "Cross-encoder re-ranking",
                    "Project and document type filtering"
                ]
            }
            
            logging.info("Retrieved API capabilities configuration")
            return response
            
        except Exception as e:
            logging.error(f"Error retrieving API capabilities: {e}")
            return {
                "endpoints": {},
                "search_parameters": {},
                "ranking_options": {},
                "response_formats": {},
                "error": str(e)
            }
