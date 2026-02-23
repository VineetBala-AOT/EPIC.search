"""
AI Handler

Handles AI mode processing - LLM parameter extraction plus AI summarization.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
from flask import current_app

from .base_handler import BaseSearchHandler


class AIHandler(BaseSearchHandler):
    """Handler for AI mode processing - LLM parameter extraction plus AI summarization."""
    
    @classmethod
    def handle(cls, query: str, project_ids: Optional[List[str]] = None, 
               document_type_ids: Optional[List[str]] = None, 
               search_strategy: Optional[str] = None, 
               inference: Optional[List] = None, 
               ranking: Optional[Dict] = None, 
               metrics: Optional[Dict] = None,
               user_location: Optional[Dict] = None,
               project_status: Optional[str] = None, 
               years: Optional[List] = None) -> Dict[str, Any]:
        """Handle AI mode processing - LLM parameter extraction plus AI summarization.
        
        AI mode performs:
        - Query relevance check up front
        - LLM-based parameter extraction (projects, document types, strategy, location)
        - Vector search with optimized parameters
        - AI summarization of search results
        
        Args:
            query: The user query
            project_ids: Optional user-provided project IDs
            document_type_ids: Optional user-provided document type IDs  
            search_strategy: Optional user-provided search strategy
            inference: Inference settings
            ranking: Optional ranking configuration
            metrics: Metrics dictionary to update
            user_location: Optional user location data (from browser)
            project_status: Optional project status parameter (user-provided takes precedence)
            years: Optional years parameter (user-provided takes precedence)
            
        Returns:
            Complete response dictionary with AI results
        """
        start_time = time.time()
        current_app.logger.info("=== AI MODE: Starting LLM parameter extraction + AI summarization processing ===")

        # Track if this is an aggregation query (initialized here, set after relevance check)
        is_aggregation_query = False

        # =====================================================================
        # PARALLEL SETUP: Run relevance check + metadata fetches concurrently
        # These 4 operations are independent — running them in parallel saves ~2-4s
        # =====================================================================
        from search_api.services.generation.factories import QueryValidatorFactory, ParameterExtractorFactory
        from search_api.clients.vector_search_client import VectorSearchClient

        app = current_app._get_current_object()

        current_app.logger.info("🚀 AI MODE: Starting parallel setup (relevance check + 3 metadata fetches)...")
        parallel_start = time.time()

        # Define tasks that each run in their own thread with Flask app context
        def _check_relevance():
            with app.app_context():
                _start = time.time()
                checker = QueryValidatorFactory.create_validator()
                result = checker.validate_query_relevance(query)
                return result, round((time.time() - _start) * 1000, 2)

        def _fetch_projects():
            with app.app_context():
                return VectorSearchClient.get_projects_list(include_metadata=True)

        def _fetch_doc_types():
            with app.app_context():
                return VectorSearchClient.get_document_types()

        def _fetch_strategies():
            with app.app_context():
                return VectorSearchClient.get_search_strategies()

        # Launch all 4 tasks in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            relevance_future = executor.submit(_check_relevance)
            projects_future = executor.submit(_fetch_projects)
            doc_types_future = executor.submit(_fetch_doc_types)
            strategies_future = executor.submit(_fetch_strategies)

        # --- Collect results with individual error handling ---

        # Relevance check result
        relevance_result = None
        relevance_time = 0
        try:
            relevance_result, relevance_time = relevance_future.result(timeout=30)
            metrics["relevance_check_time_ms"] = relevance_time
            metrics["query_relevance"] = relevance_result
            current_app.logger.info(f"🔍 AI MODE: Relevance check completed in {relevance_time}ms: {relevance_result}")
        except Exception as e:
            current_app.logger.error(f"🔍 AI MODE: Relevance check failed: {e}")
            metrics["relevance_check_time_ms"] = round((time.time() - parallel_start) * 1000, 2)
            metrics["query_relevance"] = {"checked": False, "error": str(e)}

        # Projects list
        available_projects = []
        try:
            available_projects = projects_future.result(timeout=30) or []
            current_app.logger.info(f"🤖 LLM: Found {len(available_projects)} available projects")
        except Exception as e:
            current_app.logger.warning(f"🤖 LLM: Could not fetch projects: {e}")

        # Document types
        available_document_types = []
        try:
            available_document_types = doc_types_future.result(timeout=30) or []
            current_app.logger.info(f"🤖 LLM: Found {len(available_document_types)} document types")
        except Exception as e:
            current_app.logger.warning(f"🤖 LLM: Could not fetch document types: {e}")

        # Search strategies
        available_strategies = {}
        try:
            strategies_data = strategies_future.result(timeout=30)
            if isinstance(strategies_data, dict):
                search_strategies = strategies_data.get('search_strategies', {})
                for strategy_key, strategy_data in search_strategies.items():
                    if isinstance(strategy_data, dict) and 'name' in strategy_data:
                        strategy_name = strategy_data['name']
                        description = strategy_data.get('description', f"Search strategy: {strategy_name}")
                        available_strategies[strategy_name] = description
                current_app.logger.info(f"🤖 LLM: Found {len(available_strategies)} search strategies")
        except Exception as e:
            current_app.logger.warning(f"🤖 LLM: Could not fetch search strategies: {e}")

        parallel_time = round((time.time() - parallel_start) * 1000, 2)
        current_app.logger.info(f"🚀 AI MODE: Parallel setup completed in {parallel_time}ms (relevance: {relevance_time}ms)")
        metrics["parallel_setup_time_ms"] = parallel_time

        # =====================================================================
        # PROCESS RELEVANCE RESULT — handle early exits
        # =====================================================================
        if relevance_result:
            # Handle non-EAO queries
            if not relevance_result.get("is_relevant", True):
                current_app.logger.info("🔍 AI MODE: Query not relevant to EAO - returning early")
                metrics["total_time_ms"] = round((time.time() - start_time) * 1000, 2)

                return {
                    "result": {
                        "response": relevance_result.get("suggested_response", "This query appears to be outside the scope of EAO's mandate."),
                        "documents": [],
                        "document_chunks": [],
                        "metrics": metrics,
                        "search_quality": "not_applicable",
                        "project_inference": {},
                        "document_type_inference": {},
                        "early_exit": True,
                        "exit_reason": "query_not_relevant"
                    }
                }

            # Handle generic informational queries (e.g., "What is EAO?")
            query_type = relevance_result.get("query_type", "specific_search")
            current_app.logger.info(f"🔍 AI MODE: Query type from validator: '{query_type}' for query: '{query}'")

            # LOCAL FALLBACK: If LLM didn't classify as generic but query is clearly
            # a generic EAO question, override the classification.
            # Pass available_projects so we can check if query mentions a specific project name.
            if query_type != "generic_informational" and cls._is_generic_eao_query(query, available_projects):
                current_app.logger.info(f"🔍 AI MODE: Local detector overriding query_type from '{query_type}' to 'generic_informational'")
                query_type = "generic_informational"

            if query_type == "generic_informational":
                generic_response = relevance_result.get("generic_response") or cls._get_default_eao_response()
                current_app.logger.info("🔍 AI MODE: Generic informational query detected - returning response without documents")
                metrics["total_time_ms"] = round((time.time() - start_time) * 1000, 2)
                metrics["query_type"] = "generic_informational"

                return {
                    "result": {
                        "response": generic_response,
                        "documents": [],
                        "document_chunks": [],
                        "metrics": metrics,
                        "search_quality": "informational_response",
                        "project_inference": {},
                        "document_type_inference": {},
                        "early_exit": True,
                        "exit_reason": "generic_informational_query",
                        "query_type": "generic_informational"
                    }
                }

            # Handle broad category search queries (e.g., "Mining projects in BC")
            if query_type == "broad_category_search":
                current_app.logger.info("🔍 AI MODE: Broad category search detected - fetching project list")
                category_filter = relevance_result.get("category_filter")

                return cls._handle_broad_category_search(
                    query=query,
                    category_filter=category_filter,
                    metrics=metrics,
                    start_time=start_time
                )

            # Track if this is an aggregation query (we'll handle it later after search)
            is_aggregation_query = (query_type == "aggregation_summary")
            if is_aggregation_query:
                current_app.logger.info("🔍 AI MODE: Aggregation/summary query detected - will hide chunks from response")
                metrics["query_type"] = "aggregation_summary"

            # Handle project count/list queries directly from metadata
            # e.g., "how many projects does eao have", "list all eao projects"
            if cls._is_project_count_query(query) and available_projects:
                current_app.logger.info(f"🔍 AI MODE: Project count query detected - answering from metadata ({len(available_projects)} projects)")
                metrics["total_time_ms"] = round((time.time() - start_time) * 1000, 2)
                metrics["query_type"] = "project_count"

                return {
                    "result": {
                        "response": cls._generate_project_count_response(available_projects),
                        "documents": [],
                        "document_chunks": [],
                        "metrics": metrics,
                        "search_quality": "metadata_response",
                        "project_inference": {},
                        "document_type_inference": {},
                        "early_exit": True,
                        "exit_reason": "project_count_query",
                        "query_type": "project_count"
                    }
                }

        # Also catch generic EAO queries when relevance check failed entirely
        elif cls._is_generic_eao_query(query, available_projects):
            current_app.logger.info("🔍 AI MODE: Relevance check failed but query is generic EAO - returning default response")
            metrics["total_time_ms"] = round((time.time() - start_time) * 1000, 2)
            metrics["query_type"] = "generic_informational"

            return {
                "result": {
                    "response": cls._get_default_eao_response(),
                    "documents": [],
                    "document_chunks": [],
                    "metrics": metrics,
                    "search_quality": "informational_response",
                    "project_inference": {},
                    "document_type_inference": {},
                    "early_exit": True,
                    "exit_reason": "generic_informational_fallback",
                    "query_type": "generic_informational"
                }
            }

        # =====================================================================
        # LLM PARAMETER EXTRACTION (metadata already fetched in parallel above)
        # =====================================================================
        current_app.logger.info("🤖 AI MODE: Starting parameter extraction...")
        try:
            agentic_start = time.time()
            current_app.logger.info("🤖 LLM: Starting parameter extraction from generation package...")

            # EARLY PROJECT DETECTION: Try to match projects BEFORE LLM extraction.
            # Runs both name matching and metadata matching, then picks the best.
            # Name match wins only if it's a strong match (score >= 0.7).
            # Otherwise metadata match wins (handles "mines in lower mainland" type queries).
            early_matched_project_ids = None
            if not project_ids and available_projects:
                # Try both approaches
                name_match_id, name_match_score = cls._match_project_by_query_text(query, available_projects, return_score=True)
                metadata_matches = cls._match_projects_by_metadata(query, available_projects)

                if name_match_id and name_match_score >= 0.7:
                    # Strong name match (e.g., "brucejack" → Brucejack Gold Mine)
                    early_matched_project_ids = [name_match_id]
                    current_app.logger.info(f"🚀 EARLY DETECT: Strong name match (score={name_match_score:.2f}) - using single project")
                elif metadata_matches:
                    # Metadata match (e.g., "mines in lower mainland" → all mine-type projects in LM)
                    early_matched_project_ids = metadata_matches
                    current_app.logger.info(f"🚀 EARLY DETECT: Metadata match found {len(metadata_matches)} projects by type/region")
                elif name_match_id:
                    # Weak name match, but no metadata match - still use it
                    early_matched_project_ids = [name_match_id]
                    current_app.logger.info(f"🚀 EARLY DETECT: Weak name match (score={name_match_score:.2f}), no metadata match - using name match")

            # Use LLM parameter extractor from generation package
            parameter_extractor = ParameterExtractorFactory.create_extractor()

            extraction_result = parameter_extractor.extract_parameters(
                query=query,
                available_projects=available_projects,
                available_document_types=available_document_types,
                available_strategies=available_strategies,
                # If we found projects early, supply them to skip LLM project extraction
                supplied_project_ids=project_ids if project_ids else (early_matched_project_ids if early_matched_project_ids else None),
                supplied_document_type_ids=document_type_ids if document_type_ids else None,
                supplied_search_strategy=search_strategy if search_strategy else None,
                user_location=user_location,
                supplied_project_status=project_status if project_status else None,
                supplied_years=years if years else None
            )

            # Apply extracted parameters if not already provided
            if not project_ids and extraction_result.get('project_ids'):
                project_ids = extraction_result['project_ids']
                current_app.logger.info(f"🤖 LLM: Extracted project IDs: {project_ids}")
                # Validate project IDs are valid
                if not isinstance(project_ids, list) or not all(isinstance(pid, str) for pid in project_ids):
                    current_app.logger.warning(f"🤖 LLM: Invalid project IDs format, clearing: {project_ids}")
                    project_ids = None

                # POST-LLM VALIDATION: Filter out projects that don't match query type constraints.
                # E.g., if query asks about "mines" but LLM returned a transmission project, reject it.
                if project_ids and available_projects:
                    project_ids = cls._validate_projects_against_query_type(query, project_ids, available_projects)
                    if not project_ids:
                        current_app.logger.info(f"🤖 LLM: All LLM-extracted projects rejected by type validation - will try fallback")

            # Fallback: if still no project_ids, try direct name matching, then metadata matching
            if not project_ids and available_projects:
                current_app.logger.info(f"🤖 LLM: No project_ids from extraction, trying fallback matching...")
                matched_by_name = cls._match_project_by_query_text(query, available_projects)
                if matched_by_name:
                    project_ids = [matched_by_name]
                    current_app.logger.info(f"🤖 LLM: Direct name match found project_id: {matched_by_name}")
                else:
                    # Try metadata-based matching (type + region)
                    metadata_matches = cls._match_projects_by_metadata(query, available_projects)
                    if metadata_matches:
                        project_ids = metadata_matches
                        current_app.logger.info(f"🤖 LLM: Metadata match found {len(metadata_matches)} project_ids")

            if not document_type_ids and extraction_result.get('document_type_ids'):
                document_type_ids = extraction_result['document_type_ids']
                current_app.logger.info(f"🤖 LLM: Extracted document type IDs: {document_type_ids}")
                # Validate document type IDs are valid  
                if not isinstance(document_type_ids, list) or not all(isinstance(dtid, str) for dtid in document_type_ids):
                    current_app.logger.warning(f"🤖 LLM: Invalid document type IDs format, clearing: {document_type_ids}")
                    document_type_ids = None
            
            # Apply extracted search strategy if not already provided
            if not search_strategy and extraction_result.get('search_strategy'):
                search_strategy = extraction_result['search_strategy']
                current_app.logger.info(f"🤖 LLM: Extracted search strategy: {search_strategy}")
                # Validate search strategy is valid string
                if not isinstance(search_strategy, str) or not search_strategy.strip():
                    current_app.logger.warning(f"🤖 LLM: Invalid search strategy format, clearing: {search_strategy}")
                    search_strategy = None
            
            # Use semantic query if available
            semantic_query = extraction_result.get('semantic_query', query)
            if semantic_query != query:
                current_app.logger.info(f"🤖 LLM: Generated semantic query: '{semantic_query}'")
                
            # Extract location from LLM (never user-provided)
            location = extraction_result.get('location')
            if location:
                current_app.logger.info(f"🤖 LLM: Extracted location filter from query: {location}")
                
            # Apply extracted temporal parameters if not already provided
            if not project_status and extraction_result.get('project_status'):
                project_status = extraction_result['project_status']
                current_app.logger.info(f"🤖 LLM: Extracted project status: {project_status}")
                
            if not years and extraction_result.get('years'):
                years = extraction_result['years']
                current_app.logger.info(f"🤖 LLM: Extracted years: {years}")
            
            # Record metrics - use extraction_sources to determine what was actually extracted by AI
            metrics["ai_processing_time_ms"] = round((time.time() - agentic_start) * 1000, 2)
            metrics["ai_extraction"] = extraction_result
            
            # Check if AI actually extracted these parameters (vs supplied or fallback)
            extraction_sources = extraction_result.get('extraction_sources', {})
            metrics["ai_project_extraction"] = extraction_sources.get('project_ids') in ['llm_extracted', 'llm_sequential', 'llm_parallel']
            metrics["ai_document_type_extraction"] = extraction_sources.get('document_type_ids') in ['llm_extracted', 'llm_sequential', 'llm_parallel']
            metrics["ai_location_extraction"] = extraction_sources.get('location') in ['llm_extracted', 'llm_sequential', 'llm_parallel']
            metrics["ai_project_status_extraction"] = extraction_sources.get('project_status') in ['llm_extracted', 'llm_sequential', 'llm_parallel']
            metrics["ai_years_extraction"] = extraction_sources.get('years') in ['llm_extracted', 'llm_sequential', 'llm_parallel']
            metrics["ai_semantic_query_generated"] = semantic_query != query
            metrics["ai_extraction_confidence"] = extraction_result.get('confidence', 0.0)
            metrics["ai_extraction_provider"] = ParameterExtractorFactory.get_provider()
            
            # Add extraction summary for clarity
            extraction_sources = extraction_result.get('extraction_sources', {})
            metrics["agentic_extraction_summary"] = {
                "llm_calls_made": sum(1 for source in extraction_sources.values() if source in ["llm_extracted", "llm_sequential", "llm_parallel"]),
                "parameters_supplied": sum(1 for source in extraction_sources.values() if source == "supplied"),
                "parameters_extracted": sum(1 for source in extraction_sources.values() if source in ["llm_extracted", "llm_sequential", "llm_parallel"]),
                "parameters_fallback": sum(1 for source in extraction_sources.values() if source == "fallback")
            }
            
            current_app.logger.info(f"🤖 LLM: Parameter extraction completed in {metrics['ai_processing_time_ms']}ms using {ParameterExtractorFactory.get_provider()} (confidence: {extraction_result.get('confidence', 0.0)})")
            
        except Exception as e:
            current_app.logger.error(f"🤖 LLM: Error during parameter extraction: {e}")
            metrics["ai_error"] = str(e)
            metrics["ai_processing_time_ms"] = round((time.time() - agentic_start) * 1000, 2) if 'agentic_start' in locals() else 0
            semantic_query = query  # Fallback to original query
        
        # Handle parameter stuffing - user-provided parameters take precedence (except location which is always inferred)
        final_location = location  # Always from LLM extraction, never user-provided
        final_project_status = project_status  # User-provided takes precedence  
        final_years = years  # User-provided takes precedence
        
        # AI mode uses only LLM-extracted parameters (no pattern-based fallback)
        current_app.logger.info("🎯 AI MODE: Using LLM-extracted parameters (no pattern-based fallback)")
        current_app.logger.info(f"🎯 AI MODE: Final project_ids to search: {project_ids}")
        current_app.logger.info(f"🎯 AI MODE: Final document_type_ids to search: {document_type_ids}")
        current_app.logger.info(f"🎯 AI MODE: Final parameters - location (inferred): {final_location}, status: {final_project_status}, years: {final_years}")

        # For specific content queries (asking about content WITHIN documents), avoid restrictive document type filtering
        # The semantic search is better at finding relevant content than hard filtering
        query_lower = query.lower()
        specific_content_indicators = [
            "does the", "what does", "is there", "are there", "mentions", "mention",
            "contain", "contains", "refer to", "refers to", "say about", "says about",
            "condition about", "condition that", "conditions for", "schedule b", "schedule a"
        ]
        is_specific_content_query = any(indicator in query_lower for indicator in specific_content_indicators)

        if is_specific_content_query and project_ids and document_type_ids:
            current_app.logger.info("🎯 AI MODE: Specific content query detected with both project and doc type filters")
            current_app.logger.info("🎯 AI MODE: Relaxing document type filter to let semantic search find relevant content")
            document_type_ids = None  # Let semantic search find content across all document types
        
        # Execute vector search with optimized parameters
        current_app.logger.info("🔍 AI MODE: Executing vector search...")
        search_result = cls._execute_vector_search(
            query, project_ids, document_type_ids, inference, ranking, 
            search_strategy, semantic_query, metrics, 
            location=final_location,
            user_location=user_location,
            project_status=final_project_status, 
            years=final_years
        )
        
        # Check if search returned no results
        if not search_result["documents_or_chunks"]:
            current_app.logger.warning("🔍 AI MODE: No documents found")
            metrics["total_time_ms"] = round((time.time() - start_time) * 1000, 2)
            return {
                "result": {
                    "response": "No relevant information found.",
                    "documents": [],
                    "document_chunks": [],
                    "metrics": metrics,
                    "search_quality": search_result["search_quality"],
                    "project_inference": search_result["project_inference"],
                    "document_type_inference": search_result["document_type_inference"]
                }
            }
        
        # Build project metadata context for the summary
        # This provides project description, status, and other info to the LLM
        matched_project_metadata = None
        try:
            # Determine which projects to look up metadata for
            metadata_project_ids = project_ids

            # Fallback: if no project_ids were extracted by LLM but query mentions a project,
            # try direct text matching against available project names
            if not metadata_project_ids and available_projects:
                current_app.logger.info(f"🔍 AI MODE: No project_ids from LLM, trying direct name matching against query...")
                matched_by_name = cls._match_project_by_query_text(query, available_projects)
                if matched_by_name:
                    metadata_project_ids = [matched_by_name]
                    current_app.logger.info(f"🔍 AI MODE: Direct name match found project_id: {matched_by_name}")

            if metadata_project_ids and available_projects:
                current_app.logger.info(f"🔍 AI MODE: Looking for project_ids {metadata_project_ids} in {len(available_projects)} available projects")
                matched_projects_meta = []
                for proj in available_projects:
                    if proj.get("project_id") in metadata_project_ids:
                        proj_meta = proj.get("project_metadata", {})
                        proj_meta_type = type(proj_meta).__name__
                        current_app.logger.info(f"🔍 AI MODE: Matched project '{proj.get('project_name')}' (id={proj.get('project_id')}), project_metadata type={proj_meta_type}, truthy={bool(proj_meta)}")
                        if isinstance(proj_meta, str):
                            # Handle case where metadata is a JSON string instead of dict
                            import json as json_module
                            try:
                                proj_meta = json_module.loads(proj_meta)
                                current_app.logger.info(f"🔍 AI MODE: Parsed project_metadata from string to dict")
                            except Exception as parse_err:
                                current_app.logger.error(f"🔍 AI MODE: Failed to parse project_metadata string: {parse_err}")
                                proj_meta = {}
                        if proj_meta:
                            extracted = cls._extract_metadata_fields(proj, proj_meta)
                            if extracted:
                                matched_projects_meta.append(extracted)
                if matched_projects_meta:
                    matched_project_metadata = matched_projects_meta[0] if len(matched_projects_meta) == 1 else {"projects": matched_projects_meta}
                    current_app.logger.info(f"🔍 AI MODE: Found project metadata for summary: {matched_project_metadata.get('project_name', 'multiple')}")
                    current_app.logger.info(f"🔍 AI MODE: Project metadata details - description: '{str(matched_project_metadata.get('description', ''))[:100]}...', status: '{matched_project_metadata.get('status', '')}', ea_decision: '{matched_project_metadata.get('ea_decision', '')}'")
                else:
                    current_app.logger.info(f"🔍 AI MODE: No matching project metadata found. project_ids={metadata_project_ids}, available_projects count={len(available_projects)}")
            else:
                current_app.logger.info(f"🔍 AI MODE: Skipping project metadata - project_ids={metadata_project_ids}, available_projects={'present' if available_projects else 'empty'}")
        except Exception as e:
            current_app.logger.warning(f"🔍 AI MODE: Could not extract project metadata for summary: {e}")
            import traceback
            current_app.logger.warning(f"🔍 AI MODE: Metadata extraction traceback: {traceback.format_exc()}")

        # Generate AI summary of search results
        current_app.logger.info("🔍 AI MODE: Generating AI summary...")
        summary_result = cls._generate_agentic_summary(search_result["documents_or_chunks"], query, metrics, project_metadata=matched_project_metadata)
        
        # Handle summary generation errors
        if isinstance(summary_result, dict) and "error" in summary_result:
            current_app.logger.error("🔍 AI MODE: AI summary generation failed")
            metrics["total_time_ms"] = round((time.time() - start_time) * 1000, 2)
            return {
                "result": {
                    "response": summary_result.get("fallback_response", "An error occurred while processing your request."),
                    search_result["documents_key"]: search_result["documents_or_chunks"],
                    "metrics": metrics,
                    "search_quality": search_result["search_quality"],
                    "project_inference": search_result["project_inference"],
                    "document_type_inference": search_result["document_type_inference"]
                }
            }
        
        # Calculate final metrics
        total_time = round((time.time() - start_time) * 1000, 2)
        metrics["total_time_ms"] = total_time
        
        # Log summary
        cls._log_agentic_summary(metrics, search_result["search_duration"], search_result["search_quality"], search_result["documents_or_chunks"], query)
        
        current_app.logger.info("=== AI MODE: Processing completed ===")
        
        # Separate documents and document_chunks for consistent API response
        response_documents = []
        response_document_chunks = []
        
        # Categorize the search results based on their content
        for item in search_result["documents_or_chunks"]:
            if isinstance(item, dict):
                # Check if this looks like a document chunk (has chunk-specific fields)
                if any(field in item for field in ['chunk_text', 'chunk_content', 'content', 'chunk_id']):
                    response_document_chunks.append(item)
                else:
                    # Treat as document metadata
                    response_documents.append(item)
            else:
                response_documents.append(item)
        
        current_app.logger.info(f"📊 AI MODE: Categorized {len(response_documents)} documents and {len(response_document_chunks)} document chunks")
        current_app.logger.info(f"📊 AI MODE: is_aggregation_query={is_aggregation_query} - will {'HIDE' if is_aggregation_query else 'SHOW'} chunks")

        # For aggregation queries, hide the document chunks from the response
        # User only wants the AI summary (count, statistics, etc.), not the raw chunks
        if is_aggregation_query:
            current_app.logger.info("📊 AI MODE: Aggregation query - hiding document chunks from response (AI summary only)")
            response_document_chunks = []  # Clear chunks for aggregation queries
            metrics["chunks_hidden"] = True
            metrics["chunks_hidden_reason"] = "aggregation_summary_query"

        return {
            "result": {
                "response": summary_result.get("response", "No response generated"),
                "documents": response_documents,
                "document_chunks": response_document_chunks,
                "metrics": metrics,
                "search_quality": search_result["search_quality"],
                "project_inference": search_result["project_inference"],
                "document_type_inference": search_result["document_type_inference"],
                "query_type": "aggregation_summary" if is_aggregation_query else "specific_search"
            }
        }

    @classmethod
    def _handle_broad_category_search(cls, query: str, category_filter: Optional[str],
                                       metrics: Dict, start_time: float) -> Dict[str, Any]:
        """Handle broad category search queries like 'Mining projects in BC'.

        This method:
        1. Fetches the list of projects from the vector search API
        2. Filters projects by category based on metadata
        3. Generates an AI summary of the matching projects
        4. Adds a follow-up prompt asking user to narrow down their search

        Args:
            query: The user's query
            category_filter: The category to filter by (mining, lng, pipeline, etc.)
            metrics: Metrics dictionary to update
            start_time: Start time for timing calculations

        Returns:
            Complete response dictionary with project list and follow-up prompt
        """
        from search_api.clients.vector_search_client import VectorSearchClient

        current_app.logger.info(f"🔍 AI MODE: Handling broad category search for category: {category_filter}")

        try:
            # Fetch all projects with metadata
            fetch_start = time.time()
            all_projects = VectorSearchClient.get_projects_list(include_metadata=True)
            fetch_time = round((time.time() - fetch_start) * 1000, 2)
            metrics["project_fetch_time_ms"] = fetch_time

            current_app.logger.info(f"🔍 AI MODE: Fetched {len(all_projects)} total projects in {fetch_time}ms")

            # Filter projects by category if specified
            filtered_projects = cls._filter_projects_by_category(all_projects, category_filter)

            current_app.logger.info(f"🔍 AI MODE: {len(filtered_projects)} projects match category '{category_filter}'")

            # Generate AI summary of the filtered projects with follow-up prompt
            summary_response = cls._generate_category_summary(
                query=query,
                projects=filtered_projects,
                category_filter=category_filter
            )

            metrics["total_time_ms"] = round((time.time() - start_time) * 1000, 2)
            metrics["query_type"] = "broad_category_search"
            metrics["category_filter"] = category_filter
            metrics["projects_found"] = len(filtered_projects)

            return {
                "result": {
                    "response": summary_response,
                    "documents": [],
                    "document_chunks": [],
                    "metrics": metrics,
                    "search_quality": "category_list",
                    "project_inference": {},
                    "document_type_inference": {},
                    "early_exit": True,
                    "exit_reason": "broad_category_search",
                    "query_type": "broad_category_search",
                    "category_filter": category_filter,
                    "projects_in_category": [
                        {"project_id": p.get("project_id"), "project_name": p.get("project_name")}
                        for p in filtered_projects[:20]  # Limit to top 20 for response
                    ]
                }
            }

        except Exception as e:
            current_app.logger.error(f"🔍 AI MODE: Error handling broad category search: {e}")
            metrics["total_time_ms"] = round((time.time() - start_time) * 1000, 2)
            metrics["category_search_error"] = str(e)

            return {
                "result": {
                    "response": f"I found an error while searching for {category_filter or 'projects'} in BC. Please try again or search for a specific project name.",
                    "documents": [],
                    "document_chunks": [],
                    "metrics": metrics,
                    "search_quality": "error",
                    "project_inference": {},
                    "document_type_inference": {},
                    "early_exit": True,
                    "exit_reason": "category_search_error"
                }
            }

    @classmethod
    def _filter_projects_by_category(cls, projects: List[Dict], category_filter: Optional[str]) -> List[Dict]:
        """Filter projects by category based on project name and metadata.

        Args:
            projects: List of project dictionaries with metadata
            category_filter: Category to filter by

        Returns:
            Filtered list of projects matching the category
        """
        if not category_filter:
            return projects

        # Category keywords mapping
        category_keywords = {
            "mining": ["mine", "mining", "gold", "copper", "silver", "coal", "mineral", "quarry", "aggregate", "metal"],
            "lng": ["lng", "liquefied natural gas", "natural gas", "gas export"],
            "pipeline": ["pipeline", "transmission line", "gas line"],
            "energy": ["power", "energy", "hydro", "hydroelectric", "wind", "solar", "electricity", "transmission", "substation"],
            "infrastructure": ["infrastructure", "highway", "bridge", "tunnel"],
            "water": ["dam", "reservoir", "dyke", "water diversion", "flood control", "water management"],
            "industrial": ["refinery", "chemical", "industrial", "plant", "facility", "processing"],
            "transportation": ["port", "terminal", "railway", "railroad", "marine", "airport", "ferry"],
            "waste": ["landfill", "waste", "hazardous", "disposal"],
            "resort": ["resort", "ski", "tourism", "recreation", "hotel"]
        }

        keywords = category_keywords.get(category_filter, [])
        if not keywords:
            return projects

        filtered = []
        for project in projects:
            project_name = project.get("project_name", "").lower()
            project_type = project.get("project_metadata", {}).get("type", "").lower() if project.get("project_metadata") else ""
            project_description = project.get("project_metadata", {}).get("description", "").lower() if project.get("project_metadata") else ""

            # Check if any keyword matches project name, type, or description
            combined_text = f"{project_name} {project_type} {project_description}"
            if any(keyword in combined_text for keyword in keywords):
                filtered.append(project)

        return filtered

    @classmethod
    def _generate_category_summary(cls, query: str, projects: List[Dict], category_filter: Optional[str]) -> str:
        """Generate an AI summary of projects in a category with follow-up prompt.

        Args:
            query: The user's original query
            projects: List of filtered projects
            category_filter: The category that was searched

        Returns:
            Formatted response string with project list and follow-up prompt
        """
        category_display = category_filter.upper() if category_filter else "matching"

        if not projects:
            return f"""I couldn't find any {category_filter or ''} projects in the Environmental Assessment Office database.

You can try:
- Searching for a specific project name
- Using a different category (mining, LNG, pipeline, energy, infrastructure, etc.)
- Asking about a specific document type (certificates, reports, letters)"""

        # Build the project list (limit to prevent overly long responses)
        max_display = 15
        project_names = [p.get("project_name", "Unknown") for p in projects[:max_display]]

        # Format project list with bullets
        project_list = "\n".join([f"• {name}" for name in project_names])

        # Add ellipsis if there are more
        more_text = ""
        if len(projects) > max_display:
            more_text = f"\n• ... and {len(projects) - max_display} more projects"

        # Generate the response with follow-up prompt
        response = f"""I found **{len(projects)} {category_filter or ''} projects** in the Environmental Assessment Office database:

{project_list}{more_text}

---

**Would you like more details?**

To get specific information, please tell me:
1. **Which project** interests you? (e.g., "Tell me about Cariboo Gold Project")
2. **What type of document** are you looking for? (e.g., certificates, letters, reports, Schedule B conditions)

For example, you could ask:
- "What are the Schedule B conditions for [project name]?"
- "Show me the certificates for [project name]"
- "What letters are there about [project name]?"""

        return response

    @classmethod
    def _match_project_by_query_text(cls, query: str, available_projects: List[Dict], return_score: bool = False):
        """Match a project by directly comparing query text against project names.

        This is a fallback when LLM parameter extraction doesn't return project_ids.
        Uses word-based and substring matching to find the most likely project.

        Args:
            query: The user's query text
            available_projects: List of project dicts with project_id and project_name
            return_score: If True, returns tuple (project_id, score) instead of just project_id

        Returns:
            If return_score=False: The project_id of the best matching project, or None
            If return_score=True: Tuple of (project_id, score) or (None, 0.0)
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Common words to ignore when matching - extended list
        # These prevent false matches like "mines in lower mainland" matching project names
        generic_words = {
            # Stop words (ALL short common English words that should NEVER drive matching)
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'can', 'do', 'does',
            'for', 'from', 'get', 'has', 'have', 'how', 'if', 'in', 'is', 'it', 'its',
            'may', 'me', 'my', 'no', 'not', 'of', 'on', 'or', 'our', 'so', 'than',
            'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this',
            'those', 'to', 'up', 'us', 'was', 'we', 'were', 'what', 'when', 'where',
            'which', 'who', 'will', 'with', 'would', 'you',
            # Query action words
            'tell', 'about', 'current', 'status', 'describe', 'overview',
            'show', 'find', 'search', 'all', 'documents', 'document', 'type',
            'schedule', 'certificate', 'letter', 'report', 'many',
            'phase', 'decision', 'proponent', 'region', 'location', 'information',
            'due', 'near', 'any',
            # Project/industry terms
            'project', 'mine', 'mines', 'mining', 'gold',
            'copper', 'silver', 'coal', 'gas', 'oil', 'lng', 'power', 'energy',
            'terminal', 'port', 'pipeline', 'transmission', 'line', 'facility',
            'plant', 'station', 'expansion', 'upgrade', 'development',
            # Geographic/directional terms
            'mountain', 'river', 'creek', 'lake', 'valley', 'bc', 'british', 'columbia',
            'lower', 'upper', 'north', 'south', 'east', 'west', 'northern', 'southern',
            'eastern', 'western', 'central', 'mainland', 'island', 'coast', 'coastal',
            'bay', 'inlet', 'sound', 'strait', 'interior', 'pacific',
            # Environmental/topic terms
            'impact', 'impacts', 'effect', 'effects', 'fish', 'salmon', 'habitat',
            'water', 'air', 'noise', 'wildlife', 'environment', 'environmental',
            'assessment', 'condition', 'conditions', 'mitigation', 'monitoring',
            'contamination', 'pollution', 'emission', 'emissions', 'quality',
        }

        best_match_id = None
        best_score = 0.0
        best_match_name = ""

        current_app.logger.info(f"🔍 NAME MATCH: Searching {len(available_projects)} projects for query: '{query}'")

        for proj in available_projects:
            proj_name = proj.get("project_name", "")
            if not proj_name:
                continue
            proj_name_lower = proj_name.lower()

            # Split project name into words
            proj_words = set(proj_name_lower.split())
            distinctive_words = proj_words - generic_words

            # Filter out very short words (1-2 chars) that can cause false positives
            distinctive_words = {w for w in distinctive_words if len(w) >= 3}

            # If no distinctive words remain, skip this project (all words are generic)
            if not distinctive_words:
                continue

            # Method 1: Count how many distinctive project words appear in the query (substring match)
            matches = sum(1 for w in distinctive_words if w in query_lower)

            # Method 2: Also check if query words appear as substrings of project name
            # This helps with queries like "brucejack" matching "Brucejack Gold Mine"
            query_distinctive = query_words - generic_words
            reverse_matches = sum(1 for w in query_distinctive if len(w) >= 4 and w in proj_name_lower)

            # Take the best of both methods
            total_matches = max(matches, reverse_matches)

            if total_matches == 0:
                continue

            # Calculate score based on matches
            score = total_matches / len(distinctive_words)

            # Bonus for exact word boundary matches (not just substrings)
            exact_word_matches = len(distinctive_words & query_words)
            if exact_word_matches > 0:
                score += 0.1 * exact_word_matches

            # Lower threshold to 0.33 to handle projects with multiple distinctive words
            # e.g., "Pretium Brucejack" where only "brucejack" matches
            if total_matches >= 1 and score > best_score and score >= 0.33:
                best_score = score
                best_match_id = proj.get("project_id")
                best_match_name = proj_name
                current_app.logger.info(f"🔍 NAME MATCH: Candidate '{proj_name}' score={score:.2f} (matches={total_matches}, distinctive={len(distinctive_words)})")

        if best_match_id:
            current_app.logger.info(f"🔍 NAME MATCH: Best match is '{best_match_name}' with score {best_score:.2f}")
        else:
            current_app.logger.info(f"🔍 NAME MATCH: No matching project found for query")

        if return_score:
            return best_match_id, best_score
        return best_match_id

    @classmethod
    def _match_projects_by_metadata(cls, query: str, available_projects: List[Dict]) -> List[str]:
        """Match and rank projects by metadata attributes when no project name is in the query.

        Uses ALL available metadata fields with a scoring system:
        - type, sector, commodity (project type matching)
        - region, location (geographic matching)
        - proponent (company matching)
        - currentPhaseName, eaStatus (status/phase matching)
        - eacDecision (decision outcome matching)
        - legislation (regulatory framework matching)
        - description (fallback text matching)

        Projects are scored based on how many criteria match and how strongly.
        Results are sorted by score (descending) and capped at top 15.

        Handles queries like:
        - "mines in lower mainland" → type=mine + region=lower mainland
        - "schedule b for marine port near vancouver" → type=port + region=lower mainland
        - "projects by vitreo minerals" → proponent match
        - "silica sand projects" → sector/commodity/description match
        - "active mines in peace region" → type + region + status match
        - "approved projects in kootenay" → decision + region match
        - "projects under EAA 2018" → legislation match

        Args:
            query: The user's query text
            available_projects: List of project dicts with project_id, project_name, project_metadata

        Returns:
            List of matching project_ids sorted by relevance score (best first), capped at 15
        """
        query_lower = query.lower()

        # =================================================================
        # STEP 1: Detect ALL query dimensions
        # =================================================================

        # --- Type mappings ---
        type_mappings = {
            "mine": ["mine", "mines", "mining", "quarry", "quarries", "silica", "sand mine"],
            "energy": ["energy", "electricity", "power", "hydro", "hydroelectric", "electric", "generating"],
            "petroleum": ["pipeline", "gas", "petroleum", "oil", "lng", "natural gas"],
            "transportation": ["transportation", "road", "highway", "rail", "railway"],
            "industrial": ["industrial", "factory", "plant", "smelter", "refinery", "mill"],
            "water": ["water", "dam", "reservoir", "flood", "irrigation"],
            "waste": ["waste", "landfill", "disposal"],
            "tourist": ["resort", "tourism", "tourist", "ski"],
            "port": ["port", "terminal", "jetty", "marine", "wharf", "dock"],
        }

        # --- Region mappings ---
        region_mappings = {
            "lower mainland": [
                "lower mainland", "vancouver", "burnaby", "surrey", "richmond",
                "coquitlam", "langley", "delta", "new westminster", "abbotsford",
                "maple ridge", "port moody", "pitt meadows", "white rock",
                "north vancouver", "west vancouver", "squamish", "whistler",
                "roberts bank", "tsawwassen",
            ],
            "cariboo": [
                "cariboo", "williams lake", "quesnel", "100 mile house",
            ],
            "kootenay": [
                "kootenay", "kootenays", "nelson", "cranbrook", "trail",
                "rossland", "castlegar", "kimberley", "fernie", "golden",
                "revelstoke", "invermere",
            ],
            "omineca": [
                "omineca", "prince george", "bear lake", "fort st. james",
                "vanderhoof", "burns lake", "fraser lake", "mackenzie",
            ],
            "peace": [
                "peace", "peace river", "fort st. john", "dawson creek",
                "fort nelson", "hudson's hope", "chetwynd", "tumbler ridge",
                "northeast bc", "north east bc",
            ],
            "skeena": [
                "skeena", "terrace", "kitimat", "prince rupert",
                "smithers", "houston", "stewart",
            ],
            "thompson okanagan": [
                "thompson", "okanagan", "kamloops", "kelowna", "vernon",
                "penticton", "merritt", "salmon arm", "enderby",
            ],
            "vancouver island": [
                "vancouver island", "nanaimo", "victoria", "comox",
                "campbell river", "courtenay", "duncan", "port alberni",
                "parksville", "qualicum", "tofino", "ucluelet", "sooke",
            ],
        }

        # --- Status/phase mappings ---
        # Maps query terms to status/phase values in currentPhaseName or eaStatus
        status_mappings = {
            "active": ["active", "in progress", "under review", "ongoing", "current"],
            "pre-application": ["pre-application", "early engagement", "pre-ea"],
            "application_review": ["application review", "application development", "review"],
            "completed": ["complete", "completed", "closed", "post-certificate", "decision"],
            "suspended": ["suspended", "on hold", "paused"],
            "withdrawn": ["withdrawn", "terminated", "cancelled"],
            "effects_assessment": ["effects assessment", "assessment"],
        }

        # --- Decision mappings ---
        # Maps query terms to eacDecision values
        decision_mappings = {
            "approved": ["approved", "certified", "certificate issued", "ea certificate"],
            "rejected": ["rejected", "not approved", "denied", "refused"],
            "withdrawn": ["withdrawn", "revoked"],
            "exempted": ["exempted", "exemption order"],
        }

        # --- Legislation mappings ---
        legislation_mappings = {
            "eaa 2018": ["eaa 2018", "environmental assessment act 2018", "new act"],
            "eaa 2002": ["eaa 2002", "environmental assessment act 2002"],
            "bceaa": ["bceaa", "bc environmental assessment"],
        }

        # Detect type terms in query
        matched_types = set()
        for meta_type, query_terms in type_mappings.items():
            for term in query_terms:
                if term in query_lower:
                    matched_types.add(meta_type)
                    break

        # Detect region terms in query
        matched_regions = set()
        for meta_region, query_terms in region_mappings.items():
            for term in query_terms:
                if term in query_lower:
                    matched_regions.add(meta_region)
                    break

        # Detect proponent terms in query
        proponent_patterns = []
        proponent_indicators = ["by ", "proponent ", "company ", "proposed by ", "developed by "]
        for indicator in proponent_indicators:
            idx = query_lower.find(indicator)
            if idx >= 0:
                after = query_lower[idx + len(indicator):].strip()
                words = after.split()[:5]
                if words:
                    proponent_patterns.append(" ".join(words))

        # Detect status/phase terms in query
        matched_statuses = set()
        for status_key, query_terms in status_mappings.items():
            for term in query_terms:
                if term in query_lower:
                    matched_statuses.add(status_key)
                    break

        # Detect decision terms in query
        matched_decisions = set()
        for decision_key, query_terms in decision_mappings.items():
            for term in query_terms:
                if term in query_lower:
                    matched_decisions.add(decision_key)
                    break

        # Detect legislation terms in query
        matched_legislation = set()
        for leg_key, query_terms in legislation_mappings.items():
            for term in query_terms:
                if term in query_lower:
                    matched_legislation.add(leg_key)
                    break

        # Count how many query dimensions are active
        active_dimensions = sum([
            bool(matched_types), bool(matched_regions), bool(proponent_patterns),
            bool(matched_statuses), bool(matched_decisions), bool(matched_legislation),
        ])

        if active_dimensions == 0:
            return []

        current_app.logger.info(
            f"🔍 METADATA MATCH: Detected {active_dimensions} dimensions - "
            f"types={matched_types}, regions={matched_regions}, proponent_patterns={proponent_patterns}, "
            f"statuses={matched_statuses}, decisions={matched_decisions}, legislation={matched_legislation}"
        )

        # =================================================================
        # STEP 2: Keywords for matching against project text fields
        # =================================================================
        type_name_keywords = {
            "mine": ["mine", "mines", "mining", "mineral", "ore", "gold mine", "coal mine",
                     "copper mine", "quarry", "quarries", "aggregate", "sand mine", "silica"],
            "energy": ["energy", "electricity", "power", "hydro", "generating station",
                       "powerplant", "wind farm", "solar", "substation", "transmission"],
            "petroleum": ["pipeline", "gas plant", "lng", "petroleum", "oil refinery",
                          "natural gas", "compressor station"],
            "transportation": ["highway", "road", "rail", "railway", "bridge", "tunnel"],
            "industrial": ["industrial", "smelter", "refinery", "mill", "processing",
                           "manufacturing", "chemical"],
            "water": ["dam", "reservoir", "flood control", "dredg", "sediment removal",
                      "water treatment", "diversion", "dyke"],
            "waste": ["landfill", "waste", "disposal", "hazardous"],
            "tourist": ["resort", "ski", "tourism", "recreation", "hotel"],
            "port": ["port", "terminal", "jetty", "marine", "wharf", "dock", "berth",
                     "shipment", "cargo"],
        }

        # Status keywords matched against currentPhaseName.name and eaStatus
        status_keywords = {
            "active": ["application development", "application review", "effects assessment",
                       "early engagement", "pre-application", "in progress"],
            "pre-application": ["pre-application", "early engagement"],
            "application_review": ["application review", "application development"],
            "completed": ["complete", "post-certificate", "decision"],
            "suspended": ["suspended", "on hold"],
            "withdrawn": ["withdrawn", "terminated"],
            "effects_assessment": ["effects assessment"],
        }

        # Decision keywords matched against eacDecision
        decision_keywords = {
            "approved": ["approved", "certificate issued", "ea certificate issued"],
            "rejected": ["not approved", "rejected", "refused"],
            "withdrawn": ["withdrawn", "revoked"],
            "exempted": ["exemption order", "exempted"],
        }

        # =================================================================
        # STEP 3: Score each project across ALL metadata fields
        # =================================================================
        # Scoring: Higher = more relevant
        #   Type match on metadata type field:   +4
        #   Type match on sector/commodity:      +3
        #   Type match on name/description:      +2
        #   Region match on metadata region:     +4
        #   Region match on location field:      +3
        #   Region match on name/description:    +2
        #   Proponent match:                     +4
        #   Status/phase match:                  +3
        #   Decision match:                      +3
        #   Legislation match:                   +3
        #   Multi-dimension bonus:               +2 per extra dimension matched

        scored_projects = []  # List of (project_id, score, project_name)

        for proj in available_projects:
            proj_meta = proj.get("project_metadata", {}) or {}
            if isinstance(proj_meta, str):
                try:
                    import json as json_module
                    proj_meta = json_module.loads(proj_meta)
                except Exception:
                    proj_meta = {}
            if not isinstance(proj_meta, dict):
                proj_meta = {}

            # Extract ALL searchable fields from metadata
            proj_type = str(proj_meta.get("type", "")).lower()
            proj_region = str(proj_meta.get("region", "")).lower()
            proj_sector = str(proj_meta.get("sector", "")).lower()
            proj_location = str(proj_meta.get("location", "")).lower()
            proj_commodity = str(proj_meta.get("commodity", "")).lower()
            proj_name_lower = proj.get("project_name", "").lower()
            proj_description = str(proj_meta.get("description", "")).lower()[:500]
            proj_legislation = str(proj_meta.get("legislation", "")).lower()

            # Extract proponent name (nested dict or string)
            proponent_data = proj_meta.get("proponent")
            proj_proponent = ""
            if isinstance(proponent_data, dict):
                proj_proponent = str(proponent_data.get("name", proponent_data.get("company", ""))).lower()
            elif proponent_data:
                proj_proponent = str(proponent_data).lower()

            # Extract status/phase (nested dict or string)
            phase_data = proj_meta.get("currentPhaseName") or proj_meta.get("currentPhase")
            proj_phase = ""
            if isinstance(phase_data, dict):
                proj_phase = str(phase_data.get("name", "")).lower()
            elif phase_data:
                proj_phase = str(phase_data).lower()
            proj_ea_status = str(proj_meta.get("eaStatus", "")).lower() if not isinstance(proj_meta.get("eaStatus"), dict) else ""
            if isinstance(proj_meta.get("eaStatus"), dict):
                proj_ea_status = str(proj_meta["eaStatus"].get("name", "")).lower()

            # Extract decision (nested dict or string)
            decision_data = proj_meta.get("eacDecision") or proj_meta.get("eaDecision")
            proj_decision = ""
            if isinstance(decision_data, dict):
                proj_decision = str(decision_data.get("name", "")).lower()
            elif decision_data:
                proj_decision = str(decision_data).lower()

            score = 0
            dimensions_matched = 0

            # --- TYPE SCORING ---
            type_score = 0
            if matched_types:
                for mt in matched_types:
                    # Best: metadata type field (e.g., "Mines", "Energy-Electricity")
                    if mt in proj_type:
                        type_score = max(type_score, 4)
                        break
                    name_keywords = type_name_keywords.get(mt, [])
                    # Good: sector or commodity field
                    if any(kw in proj_sector for kw in name_keywords):
                        type_score = max(type_score, 3)
                    elif proj_commodity and any(kw in proj_commodity for kw in name_keywords):
                        type_score = max(type_score, 3)
                    # Acceptable: project name or description
                    elif any(kw in proj_name_lower for kw in name_keywords):
                        type_score = max(type_score, 2)
                    elif any(kw in proj_description for kw in name_keywords):
                        type_score = max(type_score, 2)
                if type_score > 0:
                    dimensions_matched += 1
                else:
                    # Required dimension not matched — skip this project
                    continue
            score += type_score

            # --- REGION SCORING ---
            region_score = 0
            if matched_regions:
                for mr in matched_regions:
                    # Best: metadata region field
                    if mr in proj_region:
                        region_score = max(region_score, 4)
                        break
                    region_terms = region_mappings.get(mr, [])
                    # Good: location field
                    if any(term in proj_location for term in region_terms):
                        region_score = max(region_score, 3)
                    # Acceptable: project name or description
                    elif mr in proj_name_lower:
                        region_score = max(region_score, 2)
                    elif any(term in proj_description for term in region_terms):
                        region_score = max(region_score, 2)
                if region_score > 0:
                    dimensions_matched += 1
                else:
                    # Required dimension not matched — skip this project
                    continue
            score += region_score

            # --- PROPONENT SCORING ---
            proponent_score = 0
            if proponent_patterns:
                for pattern in proponent_patterns:
                    if pattern in proj_proponent:
                        proponent_score = 4
                        break
                    elif pattern in proj_name_lower:
                        proponent_score = 3
                        break
                if proponent_score > 0:
                    dimensions_matched += 1
                else:
                    continue
            score += proponent_score

            # --- STATUS/PHASE SCORING (optional — boosts but doesn't exclude) ---
            status_score = 0
            if matched_statuses:
                combined_status = f"{proj_phase} {proj_ea_status}".strip()
                for sk in matched_statuses:
                    kws = status_keywords.get(sk, [])
                    if any(kw in combined_status for kw in kws):
                        status_score = 3
                        break
                if status_score > 0:
                    dimensions_matched += 1
            score += status_score

            # --- DECISION SCORING (optional — boosts but doesn't exclude) ---
            decision_score = 0
            if matched_decisions:
                for dk in matched_decisions:
                    kws = decision_keywords.get(dk, [])
                    if any(kw in proj_decision for kw in kws):
                        decision_score = 3
                        break
                if decision_score > 0:
                    dimensions_matched += 1
            score += decision_score

            # --- LEGISLATION SCORING (optional — boosts but doesn't exclude) ---
            legislation_score = 0
            if matched_legislation:
                for lk in matched_legislation:
                    if lk in proj_legislation:
                        legislation_score = 3
                        break
                if legislation_score > 0:
                    dimensions_matched += 1
            score += legislation_score

            # Multi-dimension bonus: reward projects matching more criteria
            if dimensions_matched >= 2:
                score += (dimensions_matched - 1) * 2

            if score > 0:
                scored_projects.append((proj.get("project_id"), score, proj.get("project_name", "")))

        # =================================================================
        # STEP 4: Sort by score, cap results, and return
        # =================================================================
        scored_projects.sort(key=lambda x: x[1], reverse=True)

        # Cap at top 15 to avoid overwhelming vector search with too many project IDs
        max_results = 15
        top_projects = scored_projects[:max_results]

        matched_project_ids = [pid for pid, _score, _name in top_projects]

        if matched_project_ids:
            current_app.logger.info(
                f"🔍 METADATA MATCH: Found {len(scored_projects)} projects, returning top {len(matched_project_ids)} "
                f"(types={matched_types}, regions={matched_regions}, statuses={matched_statuses}, "
                f"decisions={matched_decisions}, legislation={matched_legislation})"
            )
            for pid, s, pname in top_projects[:8]:
                current_app.logger.info(f"  - score={s}: '{pname}'")
            if len(top_projects) > 8:
                current_app.logger.info(f"  ... and {len(top_projects) - 8} more")
        else:
            current_app.logger.info(
                f"🔍 METADATA MATCH: No projects found for types={matched_types}, regions={matched_regions}, "
                f"statuses={matched_statuses}, decisions={matched_decisions}"
            )
            sample_count = 0
            for proj in available_projects[:10]:
                meta = proj.get('project_metadata', {})
                if isinstance(meta, dict) and (meta.get('type') or meta.get('region')):
                    current_app.logger.info(
                        f"  SAMPLE: '{proj.get('project_name')}' type='{meta.get('type', '')}' "
                        f"region='{meta.get('region', '')}' sector='{meta.get('sector', '')}'"
                    )
                    sample_count += 1
                    if sample_count >= 5:
                        break

        return matched_project_ids

    @classmethod
    def _validate_projects_against_query_type(cls, query: str, project_ids: List[str],
                                                available_projects: List[Dict]) -> List[str]:
        """Validate LLM-extracted project_ids against type constraints detected in the query.

        If the query mentions a specific project type (e.g., "mines"), filter out any
        LLM-extracted projects that clearly DON'T match that type. This prevents the LLM
        from returning projects like "Interior to Lower Mainland Transmission Project"
        when the user asked about "mines in lower mainland".

        Args:
            query: The user's query text
            project_ids: List of project_ids returned by LLM extraction
            available_projects: List of project dicts with metadata

        Returns:
            Filtered list of project_ids that match the query's type constraints.
            Returns original list if no type constraint detected or no filtering needed.
        """
        query_lower = query.lower()

        # Detect type constraints in the query
        type_constraint_mappings = {
            "mine": ["mine", "mines", "mining"],
            "energy": ["energy", "electricity", "power plant", "generating station"],
            "petroleum": ["pipeline", "gas plant", "lng", "petroleum", "oil refinery"],
            "transportation": ["highway", "road", "rail", "railway"],
            "industrial": ["industrial", "smelter", "refinery", "mill"],
            "water": ["dam", "reservoir", "flood control"],
            "waste": ["landfill", "waste disposal"],
            "tourist": ["resort", "ski resort", "tourism"],
            "port": ["port", "terminal", "marine", "jetty", "wharf", "dock"],
        }

        # Keywords that indicate a project IS of a given type (checked against name, metadata type, description)
        type_validation_keywords = {
            "mine": ["mine", "mines", "mining", "mineral", "ore", "gold", "copper", "silver", "coal", "quarry"],
            "energy": ["energy", "power", "hydro", "hydroelectric", "wind farm", "solar", "generating"],
            "petroleum": ["pipeline", "gas", "lng", "petroleum", "oil"],
            "transportation": ["highway", "road", "rail", "railway", "bridge"],
            "industrial": ["industrial", "smelter", "refinery", "mill", "processing"],
            "water": ["dam", "reservoir", "flood", "dredg", "water treatment"],
            "waste": ["landfill", "waste", "disposal"],
            "tourist": ["resort", "ski", "tourism"],
            "port": ["port", "terminal", "marine", "jetty", "wharf", "dock"],
        }

        # Find which type the query is asking about
        detected_type = None
        for type_key, query_terms in type_constraint_mappings.items():
            for term in query_terms:
                if term in query_lower:
                    detected_type = type_key
                    break
            if detected_type:
                break

        if not detected_type:
            # No type constraint in query, no filtering needed
            return project_ids

        current_app.logger.info(f"🔍 TYPE VALIDATE: Query has type constraint '{detected_type}', validating {len(project_ids)} LLM-extracted projects")

        validation_keywords = type_validation_keywords.get(detected_type, [])
        if not validation_keywords:
            return project_ids

        # Build a lookup of project_id → project data
        proj_lookup = {p.get("project_id"): p for p in available_projects}

        valid_ids = []
        rejected_ids = []
        for pid in project_ids:
            proj = proj_lookup.get(pid)
            if not proj:
                # Can't validate, keep it
                valid_ids.append(pid)
                continue

            proj_name_lower = proj.get("project_name", "").lower()
            proj_meta = proj.get("project_metadata", {}) or {}
            if isinstance(proj_meta, str):
                try:
                    import json as json_module
                    proj_meta = json_module.loads(proj_meta)
                except Exception:
                    proj_meta = {}

            proj_type = str(proj_meta.get("type", "")).lower() if isinstance(proj_meta, dict) else ""
            proj_sector = str(proj_meta.get("sector", "")).lower() if isinstance(proj_meta, dict) else ""
            proj_commodity = str(proj_meta.get("commodity", "")).lower() if isinstance(proj_meta, dict) else ""
            proj_desc = str(proj_meta.get("description", "")).lower()[:500] if isinstance(proj_meta, dict) else ""

            # Check if this project matches the type constraint
            combined_text = f"{proj_name_lower} {proj_type} {proj_sector} {proj_commodity} {proj_desc}"
            if any(kw in combined_text for kw in validation_keywords):
                valid_ids.append(pid)
                current_app.logger.info(f"🔍 TYPE VALIDATE: KEPT '{proj.get('project_name')}' - matches type '{detected_type}'")
            else:
                rejected_ids.append(pid)
                current_app.logger.info(f"🔍 TYPE VALIDATE: REJECTED '{proj.get('project_name')}' - does NOT match type '{detected_type}'")

        if rejected_ids and not valid_ids:
            current_app.logger.info(f"🔍 TYPE VALIDATE: All {len(rejected_ids)} LLM projects rejected for type '{detected_type}' - clearing project_ids for fallback")
        elif rejected_ids:
            current_app.logger.info(f"🔍 TYPE VALIDATE: Kept {len(valid_ids)}, rejected {len(rejected_ids)} projects")

        return valid_ids

    @classmethod
    def _is_generic_eao_query(cls, query: str, available_projects: Optional[List[Dict]] = None) -> bool:
        """Detect if a query is a generic EAO informational question.

        Catches queries like:
        - "what is eao all about"
        - "what does people do at eao"
        - "tell me about environmental assessment office"
        - "how does the eao process work"
        - "eao mandate and responsibilities"

        Does NOT match project-specific queries like:
        - "what is brucejack project all about" (mentions a project name)
        - "what is cariboo gold" (mentions a project name)
        - "eao certificate for ajax mine" (mentions document + project)

        Args:
            query: The user's query text
            available_projects: Optional list of project dicts to dynamically check
                               project names against the query

        Returns:
            True if the query is a generic EAO question
        """
        query_lower = query.lower().strip()

        # EAO-related terms that must be present
        eao_terms = ["eao", "environmental assessment office", "environmental assessment"]
        has_eao_term = any(term in query_lower for term in eao_terms)

        if not has_eao_term:
            return False

        # Generic question patterns (query must match one of these)
        generic_patterns = [
            # Direct questions
            "what is", "what are", "what does", "what do",
            "what is the", "what are the",
            # Informational requests
            "tell me about", "explain", "describe", "overview",
            "give me an overview", "provide an overview",
            # Process questions
            "how does", "how do", "how is", "how are",
            # Topical
            "all about", "purpose of", "role of", "mandate of",
            "who is", "who are", "responsibilities",
            # People/work questions
            "what do people do", "what does people do", "people do at",
            "work at", "working at", "jobs at", "careers at",
            "who works at", "people at",
            # General knowledge
            "about eao", "about the eao",
            "learn about", "know about",
            "introduction to", "intro to",
            "summary of", "summarize",
        ]
        has_generic_pattern = any(pattern in query_lower for pattern in generic_patterns)

        if not has_generic_pattern:
            return False

        # --- EXCLUSION: Check if query mentions a specific project name ---
        # Dynamic check against all known project names
        if available_projects:
            for proj in available_projects:
                proj_name = proj.get("project_name", "")
                if not proj_name or len(proj_name) < 4:
                    continue
                proj_name_lower = proj_name.lower()
                # Check if any distinctive word (4+ chars) from the project name appears in the query
                proj_words = proj_name_lower.split()
                generic_words = {
                    'the', 'and', 'for', 'project', 'mine', 'river', 'lake', 'creek',
                    'mountain', 'island', 'north', 'south', 'east', 'west', 'british',
                    'columbia', 'lower', 'upper', 'new', 'port', 'fort',
                }
                distinctive_words = [w for w in proj_words if len(w) >= 4 and w not in generic_words]
                # If 2+ distinctive words match, or 1 distinctive word (6+ chars) matches, it's project-specific
                matches = [w for w in distinctive_words if w in query_lower]
                if len(matches) >= 2 or any(len(w) >= 6 and w in query_lower for w in distinctive_words):
                    current_app.logger.info(
                        f"🔍 GENERIC CHECK: Query mentions project '{proj_name}' (matched: {matches}) — NOT generic"
                    )
                    return False

        # Static exclusion for specific document/project terms
        specific_indicators = [
            "schedule b", "schedule a", "certificate for", "permit for",
            "condition", "document for", "report for", "letter for",
            "compliance", "amendment",
        ]
        has_specific = any(term in query_lower for term in specific_indicators)

        return not has_specific

    @classmethod
    def _get_default_eao_response(cls) -> str:
        """Return a default informational response about EAO.

        Used as fallback when the LLM doesn't generate a generic_response.

        Returns:
            Formatted string with EAO overview
        """
        return (
            "The **Environmental Assessment Office (EAO)** is an independent office within the "
            "Government of British Columbia responsible for conducting environmental assessments "
            "of major proposed projects in the province.\n\n"
            "**What does the EAO do?**\n\n"
            "The EAO reviews proposed major projects to assess their potential environmental, "
            "economic, social, heritage, and health effects. Staff at the EAO work across a range "
            "of disciplines including environmental science, Indigenous relations, public engagement, "
            "project management, policy development, and compliance monitoring.\n\n"
            "**Types of projects assessed:**\n"
            "- Mining and mineral extraction (gold, copper, coal, quarries)\n"
            "- Energy projects (hydroelectric, wind, solar, electricity transmission)\n"
            "- Oil and gas (pipelines, LNG facilities, refineries)\n"
            "- Transportation infrastructure (highways, railways, bridges)\n"
            "- Water management (dams, reservoirs, flood control, dredging)\n"
            "- Waste management (landfills, hazardous waste disposal)\n"
            "- Resort and tourism developments\n"
            "- Industrial projects (smelters, mills, processing plants)\n"
            "- Marine terminals and port facilities\n\n"
            "**Environmental Assessment Process:**\n"
            "1. **Early Engagement** — Initial discussions with proponents, Indigenous nations, "
            "and the public to identify key issues\n"
            "2. **Application Development & Review** — Proponent prepares a detailed application "
            "addressing identified issues\n"
            "3. **Effects Assessment** — Technical review of potential effects and proposed "
            "mitigation measures\n"
            "4. **Recommendation & Decision** — EAO makes a recommendation to Ministers, who "
            "decide whether to issue an Environmental Assessment Certificate\n"
            "5. **Post-Certificate Management** — Ongoing compliance monitoring and enforcement "
            "of certificate conditions\n\n"
            "**Key documents:**\n"
            "- **Environmental Assessment Certificates** — Approval documents with legally binding conditions\n"
            "- **Schedule A** — Certified project description\n"
            "- **Schedule B** — Conditions that the proponent must follow\n"
            "- **Assessment reports** — Technical analysis of project effects\n"
            "- **Consultation records** — Documentation of Indigenous and public engagement\n"
            "- **Compliance monitoring reports** — Tracking of condition adherence\n\n"
            "**Governing legislation:** The EAO operates under the *Environmental Assessment Act* (2018), "
            "which replaced the earlier BC Environmental Assessment Act (2002).\n\n"
            "Feel free to ask about specific projects, documents, or any aspect of the "
            "environmental assessment process for more detailed information."
        )

    @classmethod
    def _is_project_count_query(cls, query: str) -> bool:
        """Detect if a query is asking about the total number/list of EAO projects.

        Catches queries like "how many projects does eao have", "list all projects",
        "total number of projects" that should be answered from metadata, not document search.

        Args:
            query: The user's query text

        Returns:
            True if the query is asking about project count/totals
        """
        query_lower = query.lower().strip()

        count_patterns = [
            "how many project",
            "how many total project",
            "number of project",
            "total project",
            "count of project",
            "list all project",
            "list of project",
            "all project",
            "how many assessment",
        ]
        has_count_pattern = any(pattern in query_lower for pattern in count_patterns)

        if not has_count_pattern:
            return False

        # Exclude queries about specific project types (those should go through search)
        # e.g., "how many mining projects" should still search, not just count
        specific_type_terms = [
            "mining", "mine", "energy", "pipeline", "lng", "port",
            "resort", "dam", "highway", "industrial",
        ]
        has_specific_type = any(term in query_lower for term in specific_type_terms)

        return not has_specific_type

    @classmethod
    def _generate_project_count_response(cls, available_projects: List[Dict]) -> str:
        """Generate a response about total project count from the available projects list.

        Args:
            available_projects: List of project dicts with metadata

        Returns:
            Formatted response string with project count and summary
        """
        total = len(available_projects)

        # Count projects by type (from metadata)
        type_counts = {}
        for proj in available_projects:
            proj_meta = proj.get("project_metadata", {}) or {}
            if isinstance(proj_meta, str):
                try:
                    import json as json_module
                    proj_meta = json_module.loads(proj_meta)
                except Exception:
                    proj_meta = {}
            proj_type = proj_meta.get("type", "Unknown") if isinstance(proj_meta, dict) else "Unknown"
            if not proj_type:
                proj_type = "Unknown"
            type_counts[proj_type] = type_counts.get(proj_type, 0) + 1

        # Sort by count descending
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

        # Build type breakdown
        type_lines = []
        for ptype, count in sorted_types:
            if ptype != "Unknown":
                type_lines.append(f"- **{ptype}**: {count} projects")
        if type_counts.get("Unknown", 0) > 0:
            type_lines.append(f"- **Other/Unclassified**: {type_counts['Unknown']} projects")

        type_breakdown = "\n".join(type_lines) if type_lines else "- Type information not available for all projects"

        return (
            f"The Environmental Assessment Office (EAO) currently has **{total} projects** "
            f"in its database.\n\n"
            f"**Breakdown by project type:**\n"
            f"{type_breakdown}\n\n"
            f"You can ask about specific project types (e.g., \"mining projects in BC\") "
            f"or a specific project by name for more details."
        )

    @classmethod
    def _extract_metadata_fields(cls, proj: Dict, proj_meta: Dict) -> Optional[Dict]:
        """Extract structured metadata fields from raw project metadata.

        Handles nested dicts, type variations, and date formatting for
        fields like currentPhaseName, proponent, eacDecision, etc.

        Args:
            proj: The project dict with project_id and project_name
            proj_meta: The raw project_metadata dict from the database

        Returns:
            Dict with extracted fields, or None if extraction fails
        """
        try:
            # Log raw metadata keys for debugging
            if isinstance(proj_meta, dict):
                current_app.logger.info(f"🔍 AI MODE: Raw metadata keys: {list(proj_meta.keys())[:15]}")
                current_app.logger.info(f"🔍 AI MODE: Raw description: '{str(proj_meta.get('description', ''))[:100]}'")
                current_app.logger.info(f"🔍 AI MODE: Raw status: '{proj_meta.get('status', '')}'")
                current_app.logger.info(f"🔍 AI MODE: Raw currentPhaseName: '{proj_meta.get('currentPhaseName', '')}'")

            # Extract status from currentPhaseName (dict with 'name') or fallback to 'status' field
            status = ""
            current_phase = proj_meta.get("currentPhaseName") or proj_meta.get("currentPhase")
            if isinstance(current_phase, dict):
                status = current_phase.get("name", "")
            elif current_phase:
                status = str(current_phase)
            if not status:
                status = proj_meta.get("status", "")

            # Extract proponent name from nested dict or string
            proponent = ""
            proponent_data = proj_meta.get("proponent")
            if isinstance(proponent_data, dict):
                proponent = proponent_data.get("name", proponent_data.get("company", ""))
            elif proponent_data:
                proponent = str(proponent_data)
            if not proponent:
                proponent = proj_meta.get("proponentName", "")

            # Extract EA decision from nested dict or string
            ea_decision = ""
            ea_decision_data = proj_meta.get("eacDecision") or proj_meta.get("eaDecision")
            if isinstance(ea_decision_data, dict):
                ea_decision = ea_decision_data.get("name", "")
            elif ea_decision_data:
                ea_decision = str(ea_decision_data)

            # Format decision date (strip time portion if present)
            decision_date = proj_meta.get("decisionDate", "")
            if decision_date and "T" in str(decision_date):
                decision_date = str(decision_date).split("T")[0]

            # Extract eaStatus from nested dict or string
            ea_status = ""
            ea_status_data = proj_meta.get("eaStatus")
            if isinstance(ea_status_data, dict):
                ea_status = ea_status_data.get("name", "")
            elif ea_status_data:
                ea_status = str(ea_status_data)

            result = {
                "project_name": proj.get("project_name", "") or proj_meta.get("name", ""),
                "project_id": proj.get("project_id", ""),
                "description": proj_meta.get("description", ""),
                "status": status,
                "region": proj_meta.get("region", ""),
                "type": proj_meta.get("type", ""),
                "sector": proj_meta.get("sector", ""),
                "proponent": proponent,
                "location": proj_meta.get("location", ""),
                "commodity": proj_meta.get("commodity", ""),
                "ea_status": ea_status,
                "ea_decision": ea_decision,
                "decision_date": decision_date,
                "legislation": proj_meta.get("legislation", ""),
            }

            # Verify we have at least some useful data
            has_useful_data = any([result["description"], result["status"], result["proponent"], result["ea_decision"], result["sector"], result["commodity"]])
            if has_useful_data:
                current_app.logger.info(f"🔍 AI MODE: Extracted metadata - description present: {bool(result['description'])}, status: '{result['status']}', proponent: '{result['proponent']}', sector: '{result['sector']}', commodity: '{result['commodity']}'")
            else:
                current_app.logger.warning(f"🔍 AI MODE: Metadata extracted but all key fields are empty for project '{result['project_name']}'")

            return result

        except Exception as e:
            current_app.logger.warning(f"🔍 AI MODE: Failed to extract metadata fields: {e}")
            return None