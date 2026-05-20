"""Tools service for MCP utilities and data access.

This service module provides utility functions for MCP (Model Context Protocol) 
tools including project listings and document type information access.

The service provides:
1. Simple project listing without processing statistics 
2. Document type lookups and metadata
"""

import json
import logging
import os
import tempfile
import threading
import time
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from flask import current_app
from utils.document_types import (
    get_all_document_types,
    get_document_type,
    get_all_document_type_aliases,
    get_document_type_aliases
)

# ---------------------------------------------------------------------------
# Project embedding cache.
#
# Per-process in-memory cache (fast lookup after first build).
# Disk cache at EMBED_CACHE_DIR (default /tmp) lets all Gunicorn workers share
# the computation — the first worker that finishes saves to disk, the rest load
# it in <1s instead of recomputing.
# ---------------------------------------------------------------------------
_project_emb_cache: Dict[str, Any] = {
    "embeddings": None,
    "project_ids": None,
    "project_names": None,
    "cache_key": None,
}
_emb_cache_lock = threading.Lock()
_EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR", "/tmp")


def _disk_cache_path(cache_key: str) -> str:
    return os.path.join(_EMBED_CACHE_DIR, f"proj_embeddings_{cache_key}.npz")


def _build_project_description(project_name: str, metadata: dict) -> str:
    """Short, distinctive description for embedding — intentionally excludes long free-text.

    Keeping descriptions short (≈30-50 tokens) is critical: all-mpnet-base-v2 on CPU
    encodes ~358 short strings in ~10s vs ~3 minutes for 300-char descriptions.
    """
    meta = metadata or {}
    parts = [project_name]
    if meta.get("type"):
        parts.append(meta["type"])
    if meta.get("region"):
        parts.append(meta["region"])
    proponent = meta.get("proponent", "")
    if isinstance(proponent, dict):
        proponent = proponent.get("name", "")
    if proponent:
        parts.append(proponent)
    if meta.get("sector"):
        parts.append(meta["sector"])
    # Deliberately omit meta["description"] — it doubles encoding time per project
    return " ".join(filter(None, parts))


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

            with current_app.db_pool.connection() as conn:
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
    def _get_project_embeddings(cls):
        """Return pre-normalized project embeddings.

        Cache hierarchy (fastest → slowest):
          1. In-process memory  — zero cost after first build in this worker
          2. Disk file          — ~0.2s; lets other Gunicorn workers skip computation
          3. Compute + save     — ~10-20s for 358 short descriptions on CPU

        Returns:
            (embeddings, project_ids, project_names) — embeddings shape (n, dim), pre-normalized.
            Returns (None, [], []) on error.
        """
        global _project_emb_cache

        projects_data = cls.get_projects_list()
        projects = projects_data.get("projects", [])
        if not projects:
            return None, [], []

        current_ids = tuple(p["project_id"] for p in projects)
        # Use abs() so the key string is always positive (hash() can be negative)
        cache_key = str(abs(hash(current_ids)))

        # --- 1. In-process memory check ---
        with _emb_cache_lock:
            if _project_emb_cache["cache_key"] == cache_key and _project_emb_cache["embeddings"] is not None:
                return _project_emb_cache["embeddings"], _project_emb_cache["project_ids"], _project_emb_cache["project_names"]

        # --- 2. Disk cache check ---
        disk_path = _disk_cache_path(cache_key)
        if os.path.exists(disk_path):
            try:
                data = np.load(disk_path, allow_pickle=True)
                embeddings = data["embeddings"]
                project_ids = list(data["project_ids"])
                project_names = list(data["project_names"])
                logging.info(f"Loaded project embeddings from disk cache ({embeddings.shape})")
                with _emb_cache_lock:
                    _project_emb_cache.update({
                        "embeddings": embeddings,
                        "project_ids": project_ids,
                        "project_names": project_names,
                        "cache_key": cache_key,
                    })
                return embeddings, project_ids, project_names
            except Exception as exc:
                logging.warning(f"Disk cache load failed, recomputing: {exc}")

        # --- 3. Compute embeddings ---
        descriptions, project_ids, project_names = [], [], []
        for p in projects:
            meta = p.get("project_metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            descriptions.append(_build_project_description(p["project_name"], meta))
            project_ids.append(p["project_id"])
            project_names.append(p["project_name"])

        logging.info(f"Computing embeddings for {len(descriptions)} projects (short descriptions)…")
        t0 = time.time()

        from services.embedding import get_embedding
        batch_size = 64
        batches = [get_embedding(descriptions[i:i + batch_size]) for i in range(0, len(descriptions), batch_size)]
        embeddings = np.vstack(batches)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)

        logging.info(f"Project embeddings computed in {time.time() - t0:.1f}s ({embeddings.shape})")

        # Save to disk (atomic write so other workers see a complete file)
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(dir=_EMBED_CACHE_DIR, suffix=".npz")
            os.close(tmp_fd)
            np.savez(tmp_path, embeddings=embeddings,
                     project_ids=np.array(project_ids),
                     project_names=np.array(project_names))
            os.replace(tmp_path, disk_path)  # atomic on POSIX
            logging.info(f"Project embeddings saved to disk: {disk_path}")
        except Exception as exc:
            logging.warning(f"Could not save embeddings to disk: {exc}")

        with _emb_cache_lock:
            _project_emb_cache.update({
                "embeddings": embeddings,
                "project_ids": project_ids,
                "project_names": project_names,
                "cache_key": cache_key,
            })
        return embeddings, project_ids, project_names

    @classmethod
    def match_projects_by_embedding(cls, query: str, top_k: int = 3, threshold: float = 0.55) -> List[Dict]:
        """Find the projects most semantically similar to the query.

        Uses the SentenceTransformer already loaded in memory (Phase 1b preloading),
        so embedding a short query costs ~30ms instead of ~800ms for an LLM call.

        Args:
            query: User search query.
            top_k: Maximum number of results.
            threshold: Minimum cosine similarity to include a result.

        Returns:
            List of {"project_id", "project_name", "score"} sorted by score descending.
        """
        t0 = time.time()
        try:
            embeddings, project_ids, project_names = cls._get_project_embeddings()
            if embeddings is None:
                return []

            from services.embedding import get_embedding
            q_emb = get_embedding([query])[0]
            q_emb = q_emb / max(float(np.linalg.norm(q_emb)), 1e-10)

            scores = embeddings @ q_emb  # cosine similarity, shape (n,)

            results = []
            for idx in np.argsort(scores)[::-1]:
                score = float(scores[idx])
                if score < threshold or len(results) >= top_k:
                    break
                results.append({
                    "project_id": project_ids[idx],
                    "project_name": project_names[idx],
                    "score": round(score, 4),
                })

            top = results[0]["score"] if results else 0.0
            logging.info(f"Embedding project match: {len(results)} results in {(time.time()-t0)*1000:.0f}ms (top={top:.3f})")
            return results

        except Exception as exc:
            logging.error(f"match_projects_by_embedding failed: {exc}")
            return []

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

            with current_app.db_pool.connection() as conn:
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

            with current_app.db_pool.connection() as conn:
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
