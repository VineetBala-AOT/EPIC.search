"""
AI Handler

Handles AI mode processing - LLM parameter extraction plus AI summarization.
"""
import time
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

        # Check query relevance up front
        current_app.logger.info("🔍 AI MODE: Checking query relevance...")
        relevance_start = time.time()
        
        try:
            from search_api.services.generation.factories import QueryValidatorFactory
            relevance_checker = QueryValidatorFactory.create_validator()
            relevance_result = relevance_checker.validate_query_relevance(query)
            
            relevance_time = round((time.time() - relevance_start) * 1000, 2)
            metrics["relevance_check_time_ms"] = relevance_time
            metrics["query_relevance"] = relevance_result
            
            current_app.logger.info(f"🔍 AI MODE: Relevance check completed in {relevance_time}ms: {relevance_result}")

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
            if query_type == "generic_informational" and relevance_result.get("generic_response"):
                current_app.logger.info("🔍 AI MODE: Generic informational query detected - returning AI-generated response")
                metrics["total_time_ms"] = round((time.time() - start_time) * 1000, 2)
                metrics["query_type"] = "generic_informational"

                return {
                    "result": {
                        "response": relevance_result.get("generic_response"),
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

        except Exception as e:
            current_app.logger.error(f"🔍 AI MODE: Relevance check failed: {e}")
            metrics["relevance_check_time_ms"] = round((time.time() - relevance_start) * 1000, 2)
            metrics["query_relevance"] = {"checked": False, "error": str(e)}
        
        # LLM parameter extraction
        current_app.logger.info("🤖 AI MODE: Starting parameter extraction...")
        available_projects = []  # Initialize early so it's accessible for summary generation
        try:
            from search_api.services.generation.factories import ParameterExtractorFactory
            from search_api.clients.vector_search_client import VectorSearchClient

            agentic_start = time.time()
            current_app.logger.info("🤖 LLM: Starting parameter extraction from generation package...")
            
            # Fetch available options to provide context to the LLM
            current_app.logger.info("🤖 LLM: Fetching available options for context...")
            
            try:
                # Get available projects from vector search API (pass array directly)
                available_projects = VectorSearchClient.get_projects_list(include_metadata=True)
                
                current_app.logger.info(f"🤖 LLM: Found {len(available_projects) if available_projects else 0} available projects")
                
            except Exception as e:
                current_app.logger.warning(f"🤖 LLM: Could not fetch projects: {e}")
                available_projects = []
            
            try:
                # Get available document types from vector search API (pass array directly)
                available_document_types = VectorSearchClient.get_document_types()
                
                current_app.logger.info(f"🤖 LLM: Found {len(available_document_types) if available_document_types else 0} document types")
                
            except Exception as e:
                current_app.logger.warning(f"🤖 LLM: Could not fetch document types: {e}")
                available_document_types = []
            
            try:
                # Get available search strategies from vector search API
                strategies_data = VectorSearchClient.get_search_strategies()
                available_strategies = {}
                
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
                available_strategies = {}
            
            # Use LLM parameter extractor from generation package
            parameter_extractor = ParameterExtractorFactory.create_extractor()
            
            extraction_result = parameter_extractor.extract_parameters(
                query=query,
                available_projects=available_projects,  # Now passing arrays directly
                available_document_types=available_document_types,  # Now passing arrays directly
                available_strategies=available_strategies,
                supplied_project_ids=project_ids if project_ids else None,
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
    def _match_project_by_query_text(cls, query: str, available_projects: List[Dict]) -> Optional[str]:
        """Match a project by directly comparing query text against project names.

        This is a fallback when LLM parameter extraction doesn't return project_ids.
        Uses simple word-based matching to find the most likely project.

        Args:
            query: The user's query text
            available_projects: List of project dicts with project_id and project_name

        Returns:
            The project_id of the best matching project, or None
        """
        query_lower = query.lower()
        # Common words to ignore when matching
        generic_words = {
            'project', 'mine', 'gold', 'the', 'of', 'and', 'a', 'in', 'is', 'what',
            'tell', 'me', 'about', 'for', 'current', 'status', 'describe', 'overview',
            'show', 'get', 'find', 'search', 'all', 'documents', 'document', 'type',
            'schedule', 'certificate', 'letter', 'report', 'how', 'many', 'where',
            'who', 'when', 'which', 'does', 'do', 'are', 'there', 'can', 'you',
            'phase', 'decision', 'proponent', 'region', 'location'
        }

        best_match_id = None
        best_score = 0.0

        for proj in available_projects:
            proj_name = proj.get("project_name", "")
            if not proj_name:
                continue
            proj_name_lower = proj_name.lower()

            # Split project name into words
            proj_words = set(proj_name_lower.split())
            distinctive_words = proj_words - generic_words

            if not distinctive_words:
                distinctive_words = proj_words

            # Count how many distinctive project words appear in the query
            matches = sum(1 for w in distinctive_words if w in query_lower)

            if not distinctive_words:
                continue

            score = matches / len(distinctive_words)

            # Require at least one distinctive word match with minimum 50% coverage
            if matches >= 1 and score > best_score and score >= 0.5:
                best_score = score
                best_match_id = proj.get("project_id")
                current_app.logger.info(f"🔍 AI MODE: Name match candidate: '{proj_name}' score={score:.2f} (matched {matches}/{len(distinctive_words)} distinctive words)")

        return best_match_id

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

            result = {
                "project_name": proj.get("project_name", "") or proj_meta.get("name", ""),
                "project_id": proj.get("project_id", ""),
                "description": proj_meta.get("description", ""),
                "status": status,
                "region": proj_meta.get("region", ""),
                "type": proj_meta.get("type", ""),
                "proponent": proponent,
                "location": proj_meta.get("location", ""),
                "ea_decision": ea_decision,
                "decision_date": decision_date,
            }

            # Verify we have at least some useful data
            has_useful_data = any([result["description"], result["status"], result["proponent"], result["ea_decision"]])
            if has_useful_data:
                current_app.logger.info(f"🔍 AI MODE: Extracted metadata - description present: {bool(result['description'])}, status: '{result['status']}', proponent: '{result['proponent']}'")
            else:
                current_app.logger.warning(f"🔍 AI MODE: Metadata extracted but all key fields are empty for project '{result['project_name']}'")

            return result

        except Exception as e:
            current_app.logger.warning(f"🔍 AI MODE: Failed to extract metadata fields: {e}")
            return None