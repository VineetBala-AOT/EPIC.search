"""Search service facade for accessing vector and keyword search functionality.

This service module provides a high-level interface for search operations,
abstracting the complexity of the underlying search implementation. It acts as a
facade between the API layer and the core search engine components, providing:

1. A clean, simplified interface for the REST API endpoints
2. Consistent result formatting and structure
3. A single entry point for search operations
4. Abstraction of the multi-stage search pipeline details

The service delegates the actual search operations to specialized components
while maintaining a consistent API for clients.
"""

import logging
from typing import Dict, List, Any

from .vector_search import search, document_similarity_search
from .inference import InferencePipeline

class SearchService:
    """Search management service for document retrieval and ranking.
    
    This service class provides a simplified interface for searching documents
    using a combination of semantic vector similarity and keyword-based search.
    It encapsulates the complete search pipeline and formats results consistently.
    
    The class follows a facade design pattern, providing a unified interface
    to the complex subsystem of search operations without exposing the
    underlying implementation details.
    """

    @classmethod
    def get_documents_by_query(cls, query: str, project_ids: List[str] = None, document_type_ids: List[str] = None, inference: List[str] = None, min_relevance_score: float = None, top_n: int = None, search_strategy: str = None, semantic_query: str = None) -> Dict[str, Any]:
        """Retrieve relevant documents using an advanced two-stage search strategy with intelligent project inference.
        
        This method implements a modern search approach that leverages document-level
        metadata for improved efficiency and accuracy:
        
        Stage 0: Intelligent Project Inference (when no project_ids provided)
        - Analyzes the query for project names and project-related entities
        - Matches extracted entities against known projects using fuzzy matching
        - Automatically applies project filtering when confidence exceeds 80%
        - Removes identified project names from search terms to focus on actual topics
        - Only activates when no explicit project IDs are provided in the request
        
        Stage 1: Document Discovery
        - Searches the documents table using pre-computed keywords, tags, and headings
        - Applies project filtering (explicit or inferred) to narrow search scope
        - Identifies the most relevant documents based on metadata matching
        - Much faster than searching all chunks directly
        
        Stage 2: Chunk Retrieval
        - Performs semantic vector search within the chunks of relevant documents
        - Uses embeddings to find the most semantically similar content
        - Returns the best matching chunks from the promising documents
        
        Fallback Strategy:
        - If no relevant documents are found in Stage 1, falls back to searching all chunks
        - Ensures comprehensive coverage even for edge cases
        
        The pipeline also includes:
        - Cross-encoder re-ranking for optimal relevance ordering
        - Intelligent query-document mismatch detection
        - Comprehensive stage-specific performance metrics for each pipeline component
        - Consistent result formatting for API consumption
        
        Performance Metrics:
        The response includes detailed timing metrics for each stage:
        - inference_ms: Time spent on project/document type inference and query cleaning
        - search_ms: Time spent on the actual search operations (document + chunk search)
        - document_search_ms: Time spent finding relevant documents (Stage 1)
        - chunk_search_ms: Time spent searching chunks within documents (Stage 2)
        - semantic_search_ms: Time spent on semantic search across all chunks (fallback)
        - reranking_ms: Time spent re-ranking results with cross-encoder
        - formatting_ms: Time spent formatting results for API response
        - total_pipeline_ms: Total time for inference + search stages
        - total_search_ms: Total time for search operations only
        
        Document Type Population:
        - All search results include a 'document_type' field with human-readable type names
        - For chunk-based results, the system prioritizes document type from chunk metadata:
          1. First checks chunk_metadata.document_metadata.document_type (preferred nested structure)
          2. Falls back to chunk_metadata.document_type (legacy direct field)
          3. If not found, looks up document_type_id from document-level metadata
          4. Uses the DOCUMENT_TYPE_LOOKUP to convert IDs to display names
        
        Inference Control:
        - The inference parameter controls which inference pipelines to run:
          * ['PROJECT'] - Only run project inference
          * ['DOCUMENTTYPE'] - Only run document type inference  
          * ['PROJECT', 'DOCUMENTTYPE'] - Run both inference pipelines
          * [] or None - Behavior depends on USE_DEFAULT_INFERENCE environment variable
        - When USE_DEFAULT_INFERENCE is True (default) and inference is not provided,
          both PROJECT and DOCUMENTTYPE inference will run
        - When USE_DEFAULT_INFERENCE is False and inference is not provided or empty,
          no inference pipelines will run
        - IMPORTANT: Inference is automatically skipped when explicit IDs are provided:
          * If project_ids are provided, PROJECT inference is skipped regardless of settings
          * If document_type_ids are provided, DOCUMENTTYPE inference is skipped regardless of settings
          * This prevents unnecessary processing when IDs are already known
        
        Args:
            query (str): The natural language search query
            project_ids (List[str], optional): Explicit project IDs to filter results.
                                             If None/empty and PROJECT inference is enabled,
                                             will attempt to infer project IDs from query.
            document_type_ids (List[str], optional): Explicit document type IDs to filter results.
                                                   If None/empty and DOCUMENTTYPE inference is enabled,
                                                   will attempt to infer document type IDs from query.
            inference (List[str], optional): List of inference types to run.
                                           Valid values: 'PROJECT', 'DOCUMENTTYPE'.
                                           If not provided, uses USE_DEFAULT_INFERENCE setting.
            min_relevance_score (float, optional): Minimum relevance score threshold for filtering results.
                                                  If None, uses the MIN_RELEVANCE_SCORE config value (default: -8.0).
                                                  Cross-encoder models can produce negative scores for relevant documents.
            top_n (int, optional): Maximum number of results to return after ranking.
                                  If None, uses the TOP_RECORD_COUNT config value (default: 10).
            search_strategy (str, optional): The search strategy to use for this query.
                                           Valid values: 'HYBRID_SEMANTIC_FALLBACK', 'HYBRID_KEYWORD_FALLBACK',
                                           'SEMANTIC_ONLY', 'KEYWORD_ONLY', 'HYBRID_PARALLEL'.
                                           If None, uses the DEFAULT_SEARCH_STRATEGY config value.
        
        Returns:
            Dict[str, Any]: Search results including documents and detailed metrics
        
        Examples:
            Search with explicit project and document type filtering:
            >>> SearchService.get_documents_by_query("water quality", ["proj-001"], ["5df79dd77b5abbf7da6f51be"])
            
            Search with automatic project and document type inference:
            >>> SearchService.get_documents_by_query("I am looking for the Inspection Record for the BC Hydro project")
            # System automatically infers BC Hydro project and Inspection Record document type
            
            Search with only project inference enabled:
            >>> SearchService.get_documents_by_query("correspondence", inference=["PROJECT"])
            # Only project inference will run, document type inference is skipped
            
            Search with inference disabled:
            >>> SearchService.get_documents_by_query("correspondence", inference=[])
            # No inference pipelines will run regardless of USE_DEFAULT_INFERENCE setting
            
            Search with explicit IDs (inference automatically skipped):
            >>> SearchService.get_documents_by_query("documents", project_ids=["proj-123"], document_type_ids=["type-456"], inference=["PROJECT", "DOCUMENTTYPE"])
            # Even though inference is enabled, it will be skipped because explicit IDs are provided
        """
        
        # Store original inputs for metadata
        original_project_ids = project_ids
        original_document_type_ids = document_type_ids
        
        # Debug logging for project_ids parameter tracking
        logging.info(f"SearchService.get_documents_by_query - Received project_ids: {project_ids} (type: {type(project_ids)})")
        if project_ids:
            logging.info(f"SearchService.get_documents_by_query - project_ids length: {len(project_ids)}, values: {project_ids}")
        
        # Determine which inference pipelines to run based on the inference parameter and environment setting
        from flask import current_app
        use_default_inference = current_app.search_settings.use_default_inference
        
        # Determine search strategy to use
        if search_strategy is None:
            search_strategy = current_app.search_settings.default_search_strategy
        
        # Validate search strategy
        valid_strategies = {
            "HYBRID_SEMANTIC_FALLBACK", 
            "HYBRID_KEYWORD_FALLBACK", 
            "SEMANTIC_ONLY", 
            "KEYWORD_ONLY", 
            "HYBRID_PARALLEL",
            "DOCUMENT_ONLY"
        }
        
        if search_strategy not in valid_strategies:
            logging.warning(f"Invalid search strategy '{search_strategy}'. Using default 'HYBRID_SEMANTIC_FALLBACK'")
            search_strategy = "HYBRID_SEMANTIC_FALLBACK"
        
        # Logic for determining inference behavior:
        # 1. If inference parameter is explicitly provided (even if empty), use it
        # 2. If inference parameter is None and USE_DEFAULT_INFERENCE is True, run all inference
        # 3. If inference parameter is None and USE_DEFAULT_INFERENCE is False, run no inference
        if inference is not None:
            # Explicit inference parameter provided - use it exactly as specified
            run_project_inference = 'PROJECT' in inference
            run_document_type_inference = 'DOCUMENTTYPE' in inference
            logging.info(f"Using explicit inference parameter: {inference}")
        elif use_default_inference:
            # No inference parameter provided and default inference is enabled - run all
            run_project_inference = True
            run_document_type_inference = True
            logging.info("Using default inference: PROJECT and DOCUMENTTYPE enabled")
        else:
            # No inference parameter provided and default inference is disabled - run none
            run_project_inference = False
            run_document_type_inference = False
            logging.info("Default inference disabled: no inference will run")
        
        # Use the inference pipeline for intelligent context detection
        pipeline = InferencePipeline()
        
        # Check if this is a generic document request for cleaning decisions
        from src.services.vector_search import is_generic_document_request
        is_generic_request = is_generic_document_request(query)
        
        # Track inference stage timing
        import time
        inference_start_time = time.time()
        
        # Process query through inference pipeline with controlled inference options
        inference_results = pipeline.process_query(
            query=query,
            project_ids=project_ids,
            document_type_ids=document_type_ids,
            skip_generic_cleaning=is_generic_request,
            run_project_inference=run_project_inference,
            run_document_type_inference=run_document_type_inference
        )
        
        # Calculate inference timing
        inference_time_ms = round((time.time() - inference_start_time) * 1000, 2)
        
        # Extract results from pipeline
        search_query = inference_results["query_processing"]["final_query"]

        # Use inferred IDs ONLY if inference was successful AND no explicit IDs were provided by user
        # This ensures user-provided IDs are NEVER overwritten by inference results
        if inference_results["project_inference"]["applied"] and not original_project_ids:
            project_ids = inference_results["project_inference"]["inferred_project_ids"]
            logging.info(f"Using inferred project IDs (no explicit IDs provided): {project_ids}")
        elif original_project_ids:
            logging.info(f"Preserving user-provided project IDs (ignoring inference): {original_project_ids}")

        if inference_results["document_type_inference"]["applied"] and not original_document_type_ids:
            document_type_ids = inference_results["document_type_inference"]["inferred_document_type_ids"]
            logging.info(f"Using inferred document type IDs (no explicit IDs provided): {document_type_ids}")
        elif original_document_type_ids:
            logging.info(f"Preserving user-provided document type IDs (ignoring inference): {original_document_type_ids}")
        
        # Debug logging to see what IDs are being passed
        logging.info(f"About to call search with: project_ids={project_ids}, document_type_ids={document_type_ids}, is_generic={is_generic_request}")
        logging.info(f"Document type inference results: applied={inference_results.get('document_type_inference', {}).get('applied', False)}, ids={inference_results.get('document_type_inference', {}).get('inferred_document_type_ids', [])}")
        
        # Determine the final semantic query to use
        if semantic_query:
            # User provided a semantic query - use it directly
            final_search_query = semantic_query
            logging.info(f"Using user-provided semantic query: '{semantic_query}'")
        elif is_generic_request and not inference_results.get("document_type_inference", {}).get("applied", False):
            # Pure generic request without document type inference - use original query
            final_search_query = query
        else:
            # Content search or document type was inferred - use processed query
            final_search_query = search_query
            
        # For cases where explicit project/document type IDs are provided but no user semantic query,
        # we still want to apply semantic cleaning to improve search quality
        if not semantic_query and not is_generic_request and (project_ids or document_type_ids):
            # Apply semantic cleaning even when explicit IDs are provided
            cleaned_query = pipeline._perform_semantic_cleaning(
                query=inference_results["query_processing"]["final_query"],
                inference_results=inference_results
            )
            if cleaned_query != query:
                final_search_query = cleaned_query
                logging.info(f"Applied semantic cleaning for explicit ID search: '{cleaned_query}'")
        logging.info(f"Using search query: '{final_search_query}' (is_generic: {is_generic_request}, doc_type_applied: {inference_results.get('document_type_inference', {}).get('applied', False)})")
        
        # Track whether we applied additional semantic cleaning for explicit ID cases
        additional_semantic_cleaning_applied = False
        if not semantic_query and not is_generic_request and (project_ids or document_type_ids) and final_search_query != query:
            additional_semantic_cleaning_applied = True
        
        # Track search stage timing
        search_start_time = time.time()
        documents, search_metrics = search(final_search_query, project_ids, document_type_ids, min_relevance_score, top_n, search_strategy, semantic_query)
        search_time_ms = round((time.time() - search_start_time) * 1000, 2)
        
        # Create comprehensive stage-specific metrics
        stage_metrics = {
            "inference_ms": inference_time_ms,
            "search_ms": search_time_ms,
            "total_pipeline_ms": round(inference_time_ms + search_time_ms, 2)
        }
        
        # Add strategy metrics
        strategy_metrics = {
            "search_strategy": search_strategy,
            "strategy_applied": search_strategy,
            "strategy_source": "parameter" if search_strategy != current_app.search_settings.default_search_strategy else "default"
        }
        
        # Add breakdown of inference stage
        inference_breakdown = {
            "project_inference_enabled": run_project_inference,
            "document_type_inference_enabled": run_document_type_inference,
            "query_cleaning_applied": inference_results["query_processing"]["query_changed"],
            "generic_request_detected": is_generic_request
        }
        
        # Merge with existing search metrics to create comprehensive metrics
        comprehensive_metrics = {**search_metrics, **stage_metrics}
        comprehensive_metrics["inference_breakdown"] = inference_breakdown
        comprehensive_metrics["strategy_metrics"] = strategy_metrics

        # Check if results have low confidence (indicating possible query-document mismatch)
        search_quality = "normal"
        search_note = None
        
        if documents:
            # Check if any document has low confidence flag
            has_low_confidence = any(doc.get("search_quality") == "low_confidence" for doc in documents)
            if has_low_confidence:
                search_quality = "low_confidence"
                search_note = "The query may not be well-matched to the available documents. Consider refining your search terms or using more specific keywords related to the document content."

        # Determine the appropriate key name based on search mode
        search_mode = search_metrics.get("search_mode", "semantic")
        if search_mode == "direct_metadata":
            # Document-level results from direct metadata search
            results_key = "documents"
        else:
            # Chunk-level results from semantic search
            results_key = "document_chunks"

        response = {
            "vector_search": {
                results_key: documents,                
                "search_metrics": comprehensive_metrics,
                "search_quality": search_quality,
                "original_query": query,
                "final_semantic_query": final_search_query,  # Show the actual query sent to vector search
                "user_semantic_query_provided": semantic_query is not None,
                "semantic_cleaning_applied": "semantic_cleaning" in inference_results["query_processing"]["steps_applied"] or additional_semantic_cleaning_applied,
                "additional_semantic_cleaning_applied": additional_semantic_cleaning_applied,
                "search_mode": search_mode,
                "query_processed": inference_results["query_processing"]["query_changed"],  # Indicates if query was modified through cleaning
                "inference_settings": {
                    "use_default_inference": use_default_inference,
                    "inference_parameter": inference,
                    "project_inference_enabled": run_project_inference,
                    "document_type_inference_enabled": run_document_type_inference,
                    "project_inference_skipped": original_project_ids is not None and len(original_project_ids) > 0,
                    "document_type_inference_skipped": original_document_type_ids is not None and len(original_document_type_ids) > 0,
                    "skip_reason": "explicit_ids_provided" if (original_project_ids or original_document_type_ids) else None
                }
            }
        }
        
        if search_note:
            response["vector_search"]["search_note"] = search_note
        
        # Add project inference metadata if inference was attempted
        if not original_project_ids and inference_results["project_inference"]["attempted"]:
            response["vector_search"]["project_inference"] = {
                "attempted": True,
                "confidence": inference_results["project_inference"]["confidence"],
                "inferred_project_ids": inference_results["project_inference"]["inferred_project_ids"],
                "applied": inference_results["project_inference"]["applied"],
                "original_query": query,
                "cleaned_query": inference_results["project_inference"]["cleaned_query"],
                "metadata": inference_results["project_inference"]["metadata"]
            }
        
        # Add document type inference metadata if inference was attempted
        logging.info(f"Document type response check: original_document_type_ids={original_document_type_ids}, attempted={inference_results.get('document_type_inference', {}).get('attempted', False)}")
        logging.info(f"Document type inference full results: {inference_results.get('document_type_inference', {})}")
        if (not original_document_type_ids and 
            inference_results.get("document_type_inference", {}).get("attempted", False)):
            logging.info("Adding document type inference to response")
            response["vector_search"]["document_type_inference"] = {
                "attempted": True,
                "confidence": inference_results["document_type_inference"]["confidence"],
                "inferred_document_type_ids": inference_results["document_type_inference"]["inferred_document_type_ids"],
                "applied": inference_results["document_type_inference"]["applied"],
                "original_query": query,  # The original user query
                "cleaned_query": inference_results["document_type_inference"]["cleaned_query"],
                "metadata": inference_results["document_type_inference"]["metadata"]
            }
        else:
            logging.info(f"Not adding document type inference to response: original_ids_check={not original_document_type_ids}, attempted_check={inference_results.get('document_type_inference', {}).get('attempted', False)}")
            logging.info(f"Full document_type_inference result: {inference_results.get('document_type_inference', 'NOT_PRESENT')}")
            
        return response

    @classmethod
    def get_similar_documents(cls, document_id: str, project_ids: List[str] = None, limit: int = 10) -> Dict[str, Any]:
        """Find documents similar to the specified document using document-level embeddings.
        
        This method retrieves the embedding for the specified document and performs
        cosine similarity search against other documents in the database. The search
        leverages document-level embeddings which represent the semantic content of
        the document's keywords, tags, and headings.
        
        The similarity search process:
        1. Retrieves the embedding vector for the source document
        2. Performs cosine similarity search against all other document embeddings
        3. Optionally filters results by project IDs
        4. Returns the most similar documents ranked by similarity score
        
        Args:
            document_id (str): The ID of the document to find similar documents for
            project_ids (List[str], optional): List of project IDs to filter results.
                                             If None or empty, searches across all projects.
            limit (int): Maximum number of similar documents to return (default: 10)
            
        Returns:
            dict: A structured response containing similar documents and search metrics:
                {
                    "document_similarity": {
                        "source_document_id": "doc-123",
                        "documents": [
                            {
                                "document_id": "similar-doc-456",
                                "document_keywords": [...],
                                "document_tags": [...],
                                "document_headings": [...],
                                "project_id": "project-789",
                                "similarity_score": 0.87,
                                "created_at": "2024-01-15T10:30:00Z"
                            },
                            ...
                        ],
                        "search_metrics": {
                            "embedding_retrieval_ms": 5.2,
                            "similarity_search_ms": 45.8,
                            "formatting_ms": 1.1,
                            "total_search_ms": 52.1
                        }
                    }
                }
        """
        
        similar_documents, search_metrics = document_similarity_search(document_id, project_ids, limit)
        
        response = {
            "document_similarity": {
                "source_document_id": document_id,
                "documents": similar_documents,
                "search_metrics": search_metrics,
                "request_parameters": {
                    "limit": limit,
                    "documents_found": len(similar_documents)
                }
            }
        }
        
        # Add project_ids to response for debugging if they were provided
        if project_ids is not None:
            response["document_similarity"]["project_ids"] = project_ids
            response["document_similarity"]["request_parameters"]["project_filter_applied"] = True
        else:
            response["document_similarity"]["request_parameters"]["project_filter_applied"] = False
        
        return response
