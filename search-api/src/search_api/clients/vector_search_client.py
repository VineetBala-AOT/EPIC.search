"""Client for interacting with the external vector search API.

This module provides comprehensive functionality to call the vector search service.

The client is organized into logical groups:
- Search Operations: Core search functionality with various strategies
- Discovery Operations: Metadata and capability discovery  
- Statistics Operations: Processing metrics and health monitoring

LOCATION PARAMETER ARCHITECTURE
================================

This client handles TWO distinct location-related parameters:

1. location (Geographic Search Filter)
   - Type: str (e.g., "Vancouver, BC", "Peace River region") 
   - Source: INFERRED from query text by LLM (AI/Agent modes only)
   - Purpose: Filter search results to specific geographic areas
   - Example: User asks "projects in Vancouver" → location="Vancouver"
   - When to use: Only when the query explicitly mentions a geographic search area
   - Modes: AI mode and Agent mode (via parameter extraction)
   - Never user-provided directly via API

2. user_location (User's Physical Location)
   - Type: dict with {latitude, longitude, city, region, country, timestamp}
   - Source: User's browser/device via geolocation API
   - Purpose: Proximity-based relevance, "near me" queries
   - Example: User in Victoria searches "nearby projects" → user_location={Victoria coords}
   - When to use: Always pass through when provided by user's browser
   - Modes: All modes (RAG, RAG+Summary, AI, Agent)
   - Always user-provided via API request body

Example Scenarios:
- User in Victoria searches "projects near me"
  → location=None, user_location={Victoria coordinates}
  
- User in Victoria searches "projects in Vancouver"  
  → location="Vancouver" (inferred), user_location={Victoria coordinates}
  
- User location unknown, searches "Peace River projects"
  → location="Peace River region" (inferred), user_location=None

The vector API uses both parameters:
- location: Filter results to the specified geographic area
- user_location: Boost relevance of geographically closer results
"""

import os
import requests
from typing import Optional
from flask import current_app
from ..utils.cache import cache_with_ttl
from ..utils.token_info import get_user_id

class VectorSearchClient:
    """Client for communicating with the external vector search API."""

    # =============================================================================
    # SEARCH OPERATIONS - Core search functionality (5 methods)
    # =============================================================================

    @staticmethod
    def search(query, project_ids=None, document_type_ids=None, project_names=None, document_type_names=None, inference=None, ranking=None, search_strategy=None, semantic_query=None, location=None, user_location=None, project_status=None, years=None):
        """Advanced two-stage hybrid search with comprehensive parameters.
        
        Endpoint: POST /vector-search
        
        Args:
            query (str): The search query to send to the vector search service
            project_ids (list, optional): Optional list of project IDs to filter search results by
            document_type_ids (list, optional): Optional list of document type IDs to filter search results by
            project_names (list, optional): Optional list of project names for fuzzy matching (handled by vector API)
            document_type_names (list, optional): Optional list of document type names for fuzzy matching (handled by vector API)
            inference (list, optional): Optional list of inference types to enable (e.g., ["PROJECT", "DOCUMENTTYPE"])
            ranking (dict, optional): Optional ranking configuration with keys like 'minScore' and 'topN'
            search_strategy (str, optional): Optional search strategy:
                - "HYBRID_SEMANTIC_FALLBACK" (default)
                - "HYBRID_KEYWORD_FALLBACK" 
                - "SEMANTIC_ONLY"
                - "KEYWORD_ONLY"
                - "HYBRID_PARALLEL"
            semantic_query (str, optional): Cleaned semantic query with project/document type info removed
                for more focused vector search (used by agentic mode)
            location (str or dict, optional): Geographic search filter INFERRED from query text.
                This represents WHERE TO SEARCH for projects (e.g., "Vancouver", "Peace River region").
                ⚠️ IMPORTANT: This is NOT user-provided. It's extracted by LLM in AI/Agent modes only.
                Example: User asks "projects in Vancouver" while physically in Victoria
                → location="Vancouver" (what they're searching for)
                → user_location={Victoria coordinates} (where they are)
                If dict format is provided, it will be converted to string: "city, region, country"
            user_location (dict, optional): User's physical location FROM BROWSER/DEVICE.
                This represents WHERE THE USER IS physically located. Always passed through when provided.
                Used for resolving "near me" queries and providing proximity-based relevance.
                Contains: latitude, longitude, city, region, country, timestamp.
                Example: {"latitude": 48.4284, "longitude": -123.3656, "city": "Victoria", 
                "region": "British Columbia", "country": "Canada", "timestamp": 1696291200000}
            project_status (str, optional): Project status parameter for status filtering  
            years (list, optional): Years parameter for temporal filtering
            
        Returns:
            dict: Complete search results with metadata for better agentic integration
        """
        try:
            base_url = os.getenv("VECTOR_SEARCH_API_URL", "http://localhost:8080/api")
            vector_search_url = f"{base_url}/vector-search"
            
            # Use semantic_query as the primary query if provided, otherwise use the original query
            primary_query = semantic_query if semantic_query else query
            payload = {"query": primary_query}
            
            # Add optional parameters if provided - maintains backward compatibility
            if project_ids:
                payload["projectIds"] = project_ids
            if document_type_ids:
                payload["documentTypeIds"] = document_type_ids
            if project_names:
                payload["projectNames"] = project_names  # For fuzzy matching by vector API
            if document_type_names:
                payload["documentTypeNames"] = document_type_names  # For fuzzy matching by vector API
            if inference:
                payload["inference"] = inference
            if ranking:
                payload["ranking"] = ranking
            if search_strategy:
                payload["searchStrategy"] = search_strategy
            if location:
                # Handle different location formats  
                if isinstance(location, dict):
                    # Convert location object to string format that vector API expects
                    current_app.logger.info(f"Converting location object to string format for vector API")
                    
                    # Build location string from available fields
                    location_parts = []
                    if location.get('city'):
                        location_parts.append(location['city'])
                    if location.get('region'):
                        location_parts.append(location['region'])
                    if location.get('country'):
                        location_parts.append(location['country'])
                    
                    if location_parts:
                        location_string = ', '.join(location_parts)
                        current_app.logger.info(f"Converted location object to string: '{location_string}'")
                        payload["location"] = location_string
                    else:
                        # Fallback - use coordinates as string
                        lat = location.get('latitude')
                        lng = location.get('longitude') 
                        if lat is not None and lng is not None:
                            location_string = f"{lat},{lng}"
                            current_app.logger.info(f"Using coordinates as location string: '{location_string}'")
                            payload["location"] = location_string
                        else:
                            current_app.logger.warning(f"Location object has no usable fields: {location}")
                elif isinstance(location, str):
                    # If it's already a string, use as-is
                    payload["location"] = location
                else:
                    current_app.logger.warning(f"Unexpected location format: {type(location)} - {location}")
                    payload["location"] = str(location)
            if user_location:
                payload["userLocation"] = user_location
                current_app.logger.info(f"Added userLocation to payload: {user_location}")
            if project_status:
                payload["projectStatus"] = project_status
            if years:
                payload["years"] = years
            # Note: We don't send semanticQuery separately since we're using it as the primary query
                
            current_app.logger.info(f"Calling vector search API at address: {vector_search_url}")
            current_app.logger.info(f"Search payload: {payload}")
            if semantic_query:
                current_app.logger.info(f"Using semantic query as primary query: '{semantic_query}' (original: '{query}')")
            response = requests.post(vector_search_url, json=payload, timeout=300)
            response.raise_for_status()

            api_response = response.json()
            
            # Extract both documents and document_chunks from the response separately
            vector_search_data = api_response.get("vector_search", {})
            documents = vector_search_data.get("documents", [])
            document_chunks = vector_search_data.get("document_chunks", [])
            
            current_app.logger.info(f"Vector API returned {len(documents)} documents and {len(document_chunks)} document chunks")
            
            # Return tuple format (documents, document_chunks, api_response) for proper separation
            return documents, document_chunks, api_response
        except requests.exceptions.ConnectionError as e:
            current_app.logger.error(f"Vector search API connection failed: {str(e)}")
            current_app.logger.error(f"Check if vector search service is running on: {vector_search_url}")
            # Return tuple format with empty documents, chunks and error response
            return [], [], {
                "vector_search": {
                    "documents": [],
                    "document_chunks": []
                },
                "status": "error",
                "error": f"Vector API connection failed: {str(e)}",
                "error_type": "connection_error"
            }
        except requests.exceptions.HTTPError as e:
            # Log specific HTTP errors like 500
            current_app.logger.error(f"Vector search API HTTP error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                current_app.logger.error(f"HTTP Status: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    current_app.logger.error(f"API Error Details: {error_details}")
                except:
                    current_app.logger.error(f"API Error Text: {e.response.text}")
            # Return tuple format with empty documents, chunks and error response
            return [], [], {
                "vector_search": {
                    "documents": [],
                    "document_chunks": []
                },
                "status": "error", 
                "error": f"Vector API HTTP error: {str(e)}",
                "error_type": "http_error"
            }
        except Exception as e:
            current_app.logger.error(f"Error calling vector search API: {str(e)}")
            # Return tuple format with empty documents, chunks and error response
            return [], [], {
                "vector_search": {
                    "documents": [],
                    "document_chunks": []
                },
                "status": "error",
                "error": str(e),
                "error_type": "unknown_error"
            }

    @staticmethod
    def document_similarity_search(document_id, project_ids=None, limit=10):
        """Document-level embedding similarity search.
        
        
        Endpoint: POST /document-similarity
        
        Args:
            document_id (str): The document ID to find similar documents for
            project_ids (list, optional): Optional list of project IDs to filter similar documents by
            limit (int): Maximum number of similar documents to return
            
        Returns:
            dict: Similar documents with document-level embeddings
        """
        try:
            base_url = os.getenv("VECTOR_SEARCH_API_URL", "http://localhost:8080/api")
            vector_search_url = f"{base_url}/document-similarity"
            
            payload = {
                "documentId": document_id,
                "limit": limit
            }
            if project_ids:
                payload["projectIds"] = project_ids
                
            current_app.logger.info(f"Calling vector search document similarity API at: {vector_search_url}")
            response = requests.post(vector_search_url, json=payload, timeout=300)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            current_app.logger.error(f"Error calling vector search document similarity API: {str(e)}")
            return {}

    # =============================================================================
    # DISCOVERY OPERATIONS - Metadata and capability discovery (6 methods)
    # =============================================================================

    @staticmethod
    @cache_with_ttl(ttl_seconds=86400)  # Cache for 24 hours (86400 seconds)
    def get_projects_list(include_metadata: bool = False):
        """Get list of available projects for filtering (optionally with metadata).
        
        
        Endpoint: GET /tools/projects
        
        Returns:
            list: Array of project objects with project_id and project_name (optionally with metadata)
        """
        try:
            base_url = os.getenv("VECTOR_SEARCH_API_URL", "http://localhost:8080/api")
            vector_search_url = f"{base_url}/tools/projects"
            
            current_app.logger.info(f"Calling vector search projects list API at: {vector_search_url}")
            response = requests.get(vector_search_url, timeout=300)
            response.raise_for_status()
            
            data = response.json()

            projects = data.get('projects', [])

            if include_metadata:
                # Log metadata presence for debugging
                projects_with_meta = sum(1 for p in projects if p.get("project_metadata"))
                current_app.logger.info(f"Projects list: {len(projects)} total, {projects_with_meta} have project_metadata")
                # Log a sample project's metadata structure
                for p in projects:
                    if p.get("project_metadata") and "brucejack" in p.get("project_name", "").lower():
                        meta = p["project_metadata"]
                        meta_type = type(meta).__name__
                        if isinstance(meta, dict):
                            current_app.logger.info(f"METADATA RECEIVED [{p['project_name']}]: type={meta_type}, description_present={bool(meta.get('description'))}, status={meta.get('status', 'N/A')}, currentPhaseName={meta.get('currentPhaseName', 'N/A')}")
                        else:
                            current_app.logger.info(f"METADATA RECEIVED [{p['project_name']}]: type={meta_type}, value_preview={str(meta)[:200]}")
                return projects  # full list including project_metadata
            else:
                # Strip metadata if caller doesn't need it
                return [{"project_id": p["project_id"], "project_name": p["project_name"]} for p in projects]

        except Exception as e:
            current_app.logger.error(f"Error calling vector search projects list API: {str(e)}")
            return []

    @staticmethod
    @cache_with_ttl(ttl_seconds=86400)  # Cache for 24 hours (86400 seconds)
    def get_document_types():
        """Get document types with aliases and descriptions.
        
        
        Endpoint: GET /tools/document-types
        
        Returns:
            list: Array of document type objects with document_type_id and document_type_name for consistency with get_projects_list
        """
        try:
            base_url = os.getenv("VECTOR_SEARCH_API_URL", "http://localhost:8080/api")
            vector_search_url = f"{base_url}/tools/document-types"
            
            current_app.logger.info(f"Calling vector search document types API at: {vector_search_url}")
            response = requests.get(vector_search_url, timeout=300)
            response.raise_for_status()
            
            data = response.json()
            
            # Normalize the response format to be consistent with get_projects_list
            # Convert from {id: {name: "Letter", aliases: [...]}} to [{document_type_id: "id", document_type_name: "Letter", aliases: [...]}]
            document_types_dict = data.get('document_types', {})
            normalized_document_types = []
            
            for doc_type_id, doc_type_data in document_types_dict.items():
                if isinstance(doc_type_data, dict) and 'name' in doc_type_data:
                    normalized_doc_type = {
                        'document_type_id': doc_type_id,
                        'document_type_name': doc_type_data['name'],
                        'aliases': doc_type_data.get('aliases', []),
                        'act': doc_type_data.get('act', '')
                    }
                    normalized_document_types.append(normalized_doc_type)
            
            current_app.logger.info(f"Normalized {len(normalized_document_types)} document types from API response")
            return normalized_document_types
        except Exception as e:
            current_app.logger.error(f"Error calling vector search document types API: {str(e)}")
            return []

    @staticmethod
    @cache_with_ttl(ttl_seconds=86400)  # Cache for 24 hours (86400 seconds)
    def get_document_type_details(type_id):
        """Get detailed information for a specific document type.
        
        
        Endpoint: GET /tools/document-types/{type_id}
        
        Args:
            type_id (str): The document type ID to get details for
            
        Returns:
            dict: Document type details with id, name, and aliases
        """
        try:
            base_url = os.getenv("VECTOR_SEARCH_API_URL", "http://localhost:8080/api")
            vector_search_url = f"{base_url}/tools/document-types/{type_id}"
            
            current_app.logger.info(f"Calling vector search document type details API at: {vector_search_url}")
            response = requests.get(vector_search_url, timeout=300)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            current_app.logger.error(f"Error calling vector search document type details API: {str(e)}")
            return {}

    @staticmethod
    @cache_with_ttl(ttl_seconds=86400)  # Cache for 24 hours (86400 seconds)
    def get_search_strategies():
        """Get available search strategies and capabilities.
        
        
        Endpoint: GET /tools/search-strategies
        
        Returns:
            dict: Available search strategies with descriptions and capabilities
        """
        try:
            base_url = os.getenv("VECTOR_SEARCH_API_URL", "http://localhost:8080/api")
            vector_search_url = f"{base_url}/tools/search-strategies"
            
            current_app.logger.info(f"Calling vector search strategies API at: {vector_search_url}")
            response = requests.get(vector_search_url, timeout=300)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            current_app.logger.error(f"Error calling vector search strategies API: {str(e)}")
            return {}

    @staticmethod
    def get_inference_options():
        """Get available ML inference services and options.
        
        
        Endpoint: GET /tools/inference-options
        
        Returns:
            dict: Available inference services with capabilities
        """
        try:
            base_url = os.getenv("VECTOR_SEARCH_API_URL", "http://localhost:8080/api")
            vector_search_url = f"{base_url}/tools/inference-options"
            
            current_app.logger.info(f"Calling vector search inference options API at: {vector_search_url}")
            response = requests.get(vector_search_url, timeout=300)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            current_app.logger.error(f"Error calling vector search inference options API: {str(e)}")
            return {}

    @staticmethod
    def get_api_capabilities():
        """Complete API metadata discovery for adaptive clients.
        
        
        Endpoint: GET /tools/api-capabilities
        
        Returns:
            dict: Complete API metadata and endpoint discovery
        """
        try:
            base_url = os.getenv("VECTOR_SEARCH_API_URL", "http://localhost:8080/api")
            vector_search_url = f"{base_url}/tools/api-capabilities"
            
            current_app.logger.info(f"Calling vector search capabilities API at: {vector_search_url}")
            response = requests.get(vector_search_url, timeout=300)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            current_app.logger.error(f"Error calling vector search capabilities API: {str(e)}")
            return {}

    # =============================================================================
    # STATISTICS OPERATIONS - Processing metrics and health monitoring (4 methods)
    # =============================================================================

    @staticmethod
    def get_processing_stats(project_ids=None):
        """Get processing statistics for projects.
        
        
        Endpoint: GET /stats/processing
        
        Args:
            project_ids (list, optional): List of project IDs to filter by (Note: filtering now handled by vector API internally)
            
        Returns:
            dict: Processing statistics with per-project and aggregate metrics
        """
        try:
            base_url = os.getenv("VECTOR_SEARCH_API_URL", "http://localhost:8080/api")
            vector_search_url = f"{base_url}/stats/processing"
            
            current_app.logger.info(f"Calling vector search processing stats API (GET) at: {vector_search_url}")
            if project_ids:
                current_app.logger.info(f"Note: project_ids filter ({project_ids}) ignored - vector API handles filtering internally")
            
            response = requests.get(vector_search_url, timeout=300)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            current_app.logger.error(f"Error calling vector search processing stats API: {str(e)}")
            return {}

    @staticmethod
    def get_project_details(project_id):
        """Get detailed processing information for a specific project.
        
        
        Endpoint: GET /stats/processing/{project_id}
        
        Args:
            project_id (str): The project ID to get details for
            
        Returns:
            dict: Detailed project processing logs and metrics
        """
        try:
            base_url = os.getenv("VECTOR_SEARCH_API_URL", "http://localhost:8080/api")
            vector_search_url = f"{base_url}/stats/processing/{project_id}"
            
            current_app.logger.info(f"Calling vector search project details API at: {vector_search_url}")
            response = requests.get(vector_search_url, timeout=300)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            current_app.logger.error(f"Error calling vector search project details API: {str(e)}")
            return {}

    @staticmethod
    def get_system_summary():
        """Get high-level system overview and health status.
        
        
        Endpoint: GET /stats/summary
        
        Returns:
            dict: High-level system metrics and health status
        """
        try:
            base_url = os.getenv("VECTOR_SEARCH_API_URL", "http://localhost:8080/api")
            vector_search_url = f"{base_url}/stats/summary"
            
            current_app.logger.info(f"Calling vector search system summary API at: {vector_search_url}")
            response = requests.get(vector_search_url, timeout=300)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            current_app.logger.error(f"Error calling vector search system summary API: {str(e)}")
            return {}

    @staticmethod
    def analyze_project_health(project_ids=None, health_threshold=90):
        """Intelligent analysis of project processing health.
        
        
        Endpoint: GET /stats/health or GET /stats/health/{project_id}
        
        Args:
            project_ids (list, optional): List of project IDs to analyze
            health_threshold (float): Success rate threshold below which a project is considered unhealthy
            
        Returns:
            dict: Project health analysis with recommendations
        """
        try:
            base_url = os.getenv("VECTOR_SEARCH_API_URL", "http://localhost:8080/api")
            
            if project_ids and len(project_ids) == 1:
                # Single project health check
                vector_search_url = f"{base_url}/stats/health/{project_ids[0]}"
                params = {"health_threshold": health_threshold}
            else:
                # Multiple projects or all projects health check
                vector_search_url = f"{base_url}/stats/health"
                params = {"health_threshold": health_threshold}
                if project_ids:
                    params["project_ids"] = ",".join(project_ids)
            
            current_app.logger.info(f"Calling vector search project health API at: {vector_search_url}")
            response = requests.get(vector_search_url, params=params, timeout=300)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            current_app.logger.error(f"Error calling vector search project health API: {str(e)}")
            return {}

    # =============================================================================
    # FEEDBACK OPERATIONS - Capture user feedback and session
    # =============================================================================

    @staticmethod
    def create_feedback_session(query_text: str = None,
                                project_ids: list = None, document_type_ids: list = None,
                                search_result: dict = None) -> str:
        """
        Create a new feedback session in the vector DB.

        Args:
            user_id (str, optional): ID of the user submitting the query
            query_text (str): User's search query
            project_ids (list, optional): List of project IDs related to query
            document_type_ids (list, optional): List of document type IDs related to query

        Returns:
            str: The generated session ID for feedback tracking
        """
        try:
            base_url = os.getenv("VECTOR_SEARCH_API_URL", "http://localhost:8080/api")
            url = f"{base_url}/tools/feedback"

            payload = {
                "userId": get_user_id(),
                "queryText": query_text,
                "projectIds": project_ids,
                "documentTypeIds": document_type_ids
            }

            if search_result is not None:
                payload["searchResult"] = search_result

            current_app.logger.info(f"Creating feedback session via POST {url} with payload: {payload}")
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data.get("sessionId")

        except Exception as e:
            current_app.logger.error(f"Error creating feedback session: {e}")
            return None

    @staticmethod
    def update_feedback(        
        session_id: str,
        feedback: Optional[str] = None,
        comments: Optional[str] = None,
        summary_helpful: Optional[int] = None,
        summary_accurate: Optional[int] = None,
        doc_helpful: Optional[int] = None,
        doc_accurate: Optional[int] = None,
        summary_improvement: Optional[str] = None,
        doc_improvement: Optional[str] = None
    ) -> bool:
        """
        Update an existing feedback session in the vector DB.

        Args:
            session_id (str): The feedback session ID to update
            feedback (str): Feedback value (e.g., "useful", "not_useful")
            comments (str, optional): Optional comments from the user

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            base_url = os.getenv("VECTOR_SEARCH_API_URL", "http://localhost:8080/api")
            url = f"{base_url}/tools/feedback"

            payload = {
                "sessionId": session_id,
                "feedback": feedback,
                "comments": comments,
                "summary_helpful": summary_helpful,
                "summary_accurate": summary_accurate,
                "doc_helpful": doc_helpful,
                "doc_accurate": doc_accurate,
                "summary_improvement": summary_improvement,
                "doc_improvement": doc_improvement,
            }

            current_app.logger.info(f"Updating feedback via PATCH {url} with payload: {payload}")
            response = requests.patch(url, json=payload, timeout=300)
            response.raise_for_status()
            return True

        except Exception as e:
            current_app.logger.error(f"Error updating feedback for session {session_id}: {e}")
            return False
