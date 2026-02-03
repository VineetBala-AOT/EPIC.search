"""Main inference pipeline for processing search queries.

This module provides the main InferencePipeline class that orchestrates
intelligent inference of project IDs and document type IDs from natural
language search queries. The pipeline processes queries sequentially,
applying each inference step and maintaining detailed metadata about
the processing workflow.

The pipeline architecture:
1. Input validation and preprocessing
2. Project inference (when no explicit project IDs provided)
3. Document type inference (when no explicit document type IDs provided)
4. Query cleaning and normalization
5. Metadata collection and response formatting

Each step is designed to be transparent, providing detailed reasoning
and confidence scores for the inference decisions made.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

class InferencePipeline:
    """Main inference pipeline for processing search queries with intelligent context detection.
    
    This class orchestrates the inference workflow, applying project and document type
    inference sequentially while maintaining detailed metadata about each step.
    """
    
    def __init__(self):
        """Initialize the inference pipeline with default configuration."""
        self.project_confidence_threshold = 0.8
        self.document_type_confidence_threshold = 0.7
        
    def process_query(
        self,
        query: str,
        project_ids: Optional[List[str]] = None,
        document_type_ids: Optional[List[str]] = None,
        skip_generic_cleaning: bool = False,
        run_project_inference: bool = True,
        run_document_type_inference: bool = True,
        use_project_metadata: bool = False,
        use_database_doc_type_inference: bool = True
    ) -> Dict[str, Any]:
        """Process a search query through the complete inference pipeline.

        Args:
            query (str): The natural language search query
            project_ids (Optional[List[str]]): Explicit project IDs (skips project inference if provided)
            document_type_ids (Optional[List[str]]): Explicit document type IDs (skips doc type inference if provided)
            skip_generic_cleaning (bool): Whether to skip query cleaning for generic requests
            run_project_inference (bool): Whether to run project inference (default: True)
            run_document_type_inference (bool): Whether to run document type inference (default: True)
            use_database_doc_type_inference (bool): Whether to use database metadata for document type inference (default: True)

        Returns:
            Dict[str, Any]: Complete inference results including:
                - final_query: The processed query after all inference steps
                - inferred_project_ids: List of inferred project IDs (if any)
                - inferred_document_type_ids: List of inferred document type IDs (if any)
                - project_inference_metadata: Detailed project inference metadata
                - document_type_inference_metadata: Detailed document type inference metadata
                - processing_summary: High-level summary of what was inferred
        """
        # Import services here to avoid circular imports
        from .project_inference import project_inference_service
        from .document_type_inference import document_type_inference_service
        from ..vector_search import is_generic_document_request
        
        # Initialize processing state
        processing_state = {
            "original_query": query,
            "current_query": query,
            "project_cleaned_query": query,
            "doc_type_cleaned_query": query,
            "final_query": query,
            "inferred_project_ids": [],
            "inferred_document_type_ids": [],
            "project_inference_metadata": None,
            "document_type_inference_metadata": None,
            "project_confidence": 0.0,
            "document_type_confidence": 0.0,
            "steps_applied": []
        }
        
        # Check if this is a generic request that should skip cleaning
        is_generic_request = skip_generic_cleaning or is_generic_document_request(query)
        
        # Step 1: Project inference (if no explicit project IDs provided and inference is enabled)
        if (not project_ids or (isinstance(project_ids, list) and len(project_ids) == 0)) and run_project_inference:
            logging.info(f"Starting project inference for query: '{query}'")
            project_results = self._process_project_inference(
                processing_state, is_generic_request
            )
            # Always save the inference metadata, regardless of whether it's applied
            processing_state["inferred_project_ids"] = project_results["inferred_project_ids"]
            processing_state["project_confidence"] = project_results["project_confidence"]
            processing_state["project_inference_metadata"] = project_results["project_inference_metadata"]
            
            if project_results["applied"]:
                project_ids = project_results["inferred_project_ids"]
                # Update other processing state fields only if applied
                if "project_cleaned_query" in project_results:
                    processing_state["project_cleaned_query"] = project_results["project_cleaned_query"]
        elif not run_project_inference:
            logging.info("Project inference disabled by inference parameter")
        else:
            logging.info(f"Skipping project inference: explicit project IDs provided: {project_ids}")
            
        # Step 2: Document type inference (if no explicit document type IDs provided and inference is enabled)
        if (not document_type_ids or (isinstance(document_type_ids, list) and len(document_type_ids) == 0)) and run_document_type_inference:
            logging.info(f"Starting document type inference for query: '{processing_state['current_query']}'")
            doc_type_results = self._process_document_type_inference(
                processing_state, is_generic_request, use_database_inference=use_database_doc_type_inference
            )
            # Always save the inference metadata, regardless of whether it's applied
            processing_state["inferred_document_type_ids"] = doc_type_results["inferred_document_type_ids"]
            processing_state["document_type_confidence"] = doc_type_results["document_type_confidence"]
            processing_state["document_type_inference_metadata"] = doc_type_results["document_type_inference_metadata"]
            
            if doc_type_results["applied"]:
                document_type_ids = doc_type_results["inferred_document_type_ids"]
                # Update other processing state fields only if applied
                if "doc_type_cleaned_query" in doc_type_results:
                    processing_state["doc_type_cleaned_query"] = doc_type_results["doc_type_cleaned_query"]
        elif not run_document_type_inference:
            logging.info("Document type inference disabled by inference parameter")
        else:
            logging.info(f"Skipping document type inference: explicit document type IDs provided: {document_type_ids}")
 
        # Append project name to query if metadata is enabled
        appended_project_names = []
        if use_project_metadata and processing_state.get("project_inference_metadata"):
            for m in processing_state["project_inference_metadata"]:
                name = m.get("project_name")
                if name and name not in processing_state["current_query"]:
                    appended_project_names.append(name)
            if appended_project_names:
                processing_state["current_query"] += " " + " ".join(appended_project_names)
                logging.info(f"Appended project names to query: {appended_project_names}")
            
        # Step 3: Second-pass semantic cleaning (if not a generic request)
        if not is_generic_request and processing_state["current_query"] != processing_state["original_query"]:
            semantic_cleaned = self._perform_semantic_cleaning(processing_state["current_query"])
            if semantic_cleaned != processing_state["current_query"]:
                processing_state["semantic_cleaned_query"] = semantic_cleaned
                processing_state["current_query"] = semantic_cleaned
                processing_state["steps_applied"].append("semantic_cleaning")
                logging.info(f"Semantic cleaning applied: '{semantic_cleaned}'")
            else:
                processing_state["semantic_cleaned_query"] = processing_state["current_query"]
        else:
            processing_state["semantic_cleaned_query"] = processing_state["current_query"]
            
        # Finalize processing state
        processing_state["final_query"] = processing_state["current_query"]
        
        # Build comprehensive response
        return self._build_inference_response(
            processing_state, project_ids, document_type_ids
        )
        
    def _process_project_inference(
        self, 
        processing_state: Dict[str, Any], 
        is_generic_request: bool
    ) -> Dict[str, Any]:
        """Process project inference step.
        
        Args:
            processing_state (Dict[str, Any]): Current processing state
            is_generic_request (bool): Whether this is a generic request
            
        Returns:
            Dict[str, Any]: Project inference results
        """
        from .project_inference import project_inference_service
        
        # Perform project inference
        inferred_project_ids, project_confidence, project_metadata = project_inference_service.infer_projects_from_query(
            processing_state["original_query"]
        )
        
        result = {
            "inferred_project_ids": inferred_project_ids or [],
            "project_confidence": project_confidence,
            "project_inference_metadata": project_metadata,
            "applied": False
        }
        
        # Apply project inference if confidence meets threshold
        if inferred_project_ids and project_confidence > self.project_confidence_threshold:
            result["applied"] = True
            
            # Clean query unless it's a generic request
            if not is_generic_request:
                project_cleaned = project_inference_service.clean_query_after_inference(
                    processing_state["current_query"], project_metadata
                )
                processing_state["project_cleaned_query"] = project_cleaned
                processing_state["current_query"] = project_cleaned
                processing_state["steps_applied"].append("project_cleaning")
                
                logging.info(f"Project inference applied: {inferred_project_ids} with confidence {project_confidence:.3f}")
                logging.info(f"Query after project cleaning: '{project_cleaned}'")
            else:
                processing_state["project_cleaned_query"] = processing_state["current_query"]
                logging.info(f"Project inference applied but skipped cleaning for generic request")
                
        else:
            logging.info(f"Project inference not applied: confidence {project_confidence:.3f} below threshold {self.project_confidence_threshold}")
            
        return result
        
    def _process_document_type_inference(
        self,
        processing_state: Dict[str, Any],
        is_generic_request: bool,
        use_database_inference: bool = True
    ) -> Dict[str, Any]:
        """Process document type inference step.

        Args:
            processing_state (Dict[str, Any]): Current processing state
            is_generic_request (bool): Whether this is a generic request
            use_database_inference (bool): Whether to use database metadata for enhanced inference

        Returns:
            Dict[str, Any]: Document type inference results
        """
        from .document_type_inference import document_type_inference_service

        # Use the original query for document type inference (document type terms are in the original query)
        query_for_inference = processing_state["original_query"]

        # Get inferred project IDs to scope database search (if available)
        project_ids = processing_state.get("inferred_project_ids")

        # Perform document type inference - use combined method if database inference is enabled
        if use_database_inference:
            logging.info("Using combined alias + database document type inference")
            inferred_doc_type_ids, doc_type_confidence, doc_type_metadata = document_type_inference_service.infer_with_combined_methods(
                query_for_inference,
                project_ids=project_ids,
                confidence_threshold=self.document_type_confidence_threshold
            )
        else:
            # Use alias-only inference
            inferred_doc_type_ids, doc_type_confidence, doc_type_metadata = document_type_inference_service.infer_document_types_from_query(
                query_for_inference
            )
        
        result = {
            "inferred_document_type_ids": inferred_doc_type_ids or [],
            "document_type_confidence": doc_type_confidence,
            "document_type_inference_metadata": doc_type_metadata,
            "applied": False
        }
        
        # Apply document type inference if confidence meets threshold
        if inferred_doc_type_ids and doc_type_confidence > self.document_type_confidence_threshold:
            result["applied"] = True
            
            # Clean query unless it's a generic request
            if not is_generic_request:
                doc_type_cleaned = document_type_inference_service.clean_query_after_inference(
                    processing_state["current_query"], doc_type_metadata
                )
                processing_state["doc_type_cleaned_query"] = doc_type_cleaned
                processing_state["current_query"] = doc_type_cleaned
                processing_state["steps_applied"].append("document_type_cleaning")
                
                logging.info(f"Document type inference applied: {inferred_doc_type_ids} with confidence {doc_type_confidence:.3f}")
                logging.info(f"Query after document type cleaning: '{doc_type_cleaned}'")
            else:
                processing_state["doc_type_cleaned_query"] = processing_state["current_query"]
                logging.info(f"Document type inference applied but skipped cleaning for generic request")
                
        else:
            logging.info(f"Document type inference not applied: confidence {doc_type_confidence:.3f} below threshold {self.document_type_confidence_threshold}")
            
        return result
        
    def _build_inference_response(
        self, 
        processing_state: Dict[str, Any], 
        final_project_ids: Optional[List[str]], 
        final_document_type_ids: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Build the final inference response with all metadata.
        
        Args:
            processing_state (Dict[str, Any]): Complete processing state
            final_project_ids (Optional[List[str]]): Final project IDs to use
            final_document_type_ids (Optional[List[str]]): Final document type IDs to use
            
        Returns:
            Dict[str, Any]: Complete inference response
        """
        response = {
            "query_processing": {
                "original_query": processing_state["original_query"],
                "final_query": processing_state["final_query"],
                "steps_applied": processing_state["steps_applied"],
                "query_changed": processing_state["final_query"] != processing_state["original_query"],
                "semantic_cleaned_query": processing_state.get("semantic_cleaned_query") if processing_state.get("semantic_cleaned_query") != processing_state.get("original_query") else None
            },
            "project_inference": {
                "attempted": processing_state["project_inference_metadata"] is not None,
                "applied": bool(processing_state.get("inferred_project_ids")),
                "confidence": processing_state.get("project_confidence", 0.0),
                "inferred_project_ids": processing_state.get("inferred_project_ids", []),
                "final_project_ids": final_project_ids or [],
                "cleaned_query": processing_state["project_cleaned_query"] if processing_state["project_cleaned_query"] != processing_state["original_query"] else None,
                "metadata": processing_state.get("project_inference_metadata")
            },
            "document_type_inference": {
                "attempted": processing_state["document_type_inference_metadata"] is not None,
                "applied": bool(processing_state.get("inferred_document_type_ids")),
                "confidence": processing_state.get("document_type_confidence", 0.0),
                "inferred_document_type_ids": processing_state.get("inferred_document_type_ids", []),
                "final_document_type_ids": final_document_type_ids or [],
                "cleaned_query": processing_state.get("doc_type_cleaned_query") if processing_state.get("doc_type_cleaned_query") and processing_state.get("doc_type_cleaned_query") != processing_state.get("project_cleaned_query", processing_state["original_query"]) else None,
                "metadata": processing_state.get("document_type_inference_metadata")
            },
            "summary": {
                "inference_applied": len(processing_state["steps_applied"]) > 0,
                "project_inference_successful": bool(processing_state.get("inferred_project_ids")),
                "document_type_inference_successful": bool(processing_state.get("inferred_document_type_ids")),
                "query_modified": processing_state["final_query"] != processing_state["original_query"]
            }
        }
        
        return response

    def _perform_semantic_cleaning(self, query: str, inference_results: dict = None) -> str:
        """Perform second-pass semantic cleaning to remove grammatical artifacts and extract key terms.
        
        This method cleans queries that have already been processed by project and document type
        inference to remove leftover grammatical artifacts and focus on the core semantic content.
        
        Args:
            query (str): The query after project and document type cleaning
            
        Returns:
            str: Semantically cleaned query focused on key search terms
        """
        import re
        import logging
        
        cleaned_query = query.strip()
        original_length = len(cleaned_query)
        
        # Priority 1: Extract quoted content (highest confidence)
        quote_patterns = [
            r"['\"]([^'\"]+)['\"]",  # Content within single or double quotes
            r"'([^']+)'",           # Content within single quotes
            r'"([^"]+)"'            # Content within double quotes
        ]
        
        for pattern in quote_patterns:
            matches = re.findall(pattern, cleaned_query, re.IGNORECASE)
            if matches:
                # Use the longest quoted content as the main search term
                longest_match = max(matches, key=len)
                if len(longest_match.strip()) >= 3:  # Ensure meaningful content
                    logging.info(f"Semantic cleaning: extracted quoted content '{longest_match}'")
                    return longest_match.strip()
        
        # Priority 2: Remove common search action phrases
        action_phrases = [
            r'\blook\s+for\s+(reference\s+of\s*)?',  # "look for reference of"
            r'\bsearch\s+for\s*',                    # "search for"
            r'\bfind\s+(reference\s+to\s*)?',        # "find reference to"
            r'\bshow\s+me\s*',                       # "show me"
            r'\bget\s+me\s*',                        # "get me"
            r'\breference\s+of\s*',                  # "reference of"
            r'\breferences?\s+to\s*',                # "reference to" / "references to"
            r'\binformation\s+about\s*',             # "information about"
            r'\bdetails\s+about\s*',                 # "details about"
            r'\bany\s+information\s+on\s*',          # "any information on"
        ]
        
        for pattern in action_phrases:
            cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
        
        # Priority 3: Remove orphaned prepositions, articles, and conjunctions
        orphaned_words = [
            r'^\s*(the|a|an|in|on|at|for|of|to|with|by|from)\s+',  # At start
            r'\s+(the|a|an|in|on|at|for|of|to|with|by|from)\s*$',  # At end
        ]
        
        for pattern in orphaned_words:
            cleaned_query = re.sub(pattern, ' ', cleaned_query, flags=re.IGNORECASE)
        
        # Priority 4: Clean up extra whitespace and punctuation
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query)  # Multiple spaces to single
        cleaned_query = re.sub(r'^\s*[,;:.]\s*', '', cleaned_query)  # Leading punctuation
        cleaned_query = re.sub(r'\s*[,;:]\s*$', '', cleaned_query)   # Trailing punctuation
        cleaned_query = cleaned_query.strip()
        
        # Priority 5: Remove very short orphaned words that don't add meaning
        words = cleaned_query.split()
        meaningful_words = []
        
        for word in words:
            # Keep words that are:
            # - 3+ characters long, OR
            # - 2 characters and capitalized (like "BC"), OR
            # - Common meaningful short words
            if (len(word) >= 3 or 
                (len(word) == 2 and word[0].isupper()) or
                word.lower() in ['ai', 'ml', 'it', 'bc', 'ab', 'on', 'qc', 'pe', 'ns', 'nb', 'nt', 'nu', 'yt']):
                meaningful_words.append(word)
        
        if meaningful_words:
            cleaned_query = ' '.join(meaningful_words)
        
        # Safety checks
        # If we removed more than 85% of the original, it's too aggressive
        if len(cleaned_query) < original_length * 0.15:
            logging.warning(f"Semantic cleaning too aggressive (removed {100*(1-len(cleaned_query)/original_length):.0f}%), keeping previous: '{query}'")
            cleaned_query = query
        
        # If the cleaned query is too short, keep the previous version
        if len(cleaned_query.strip()) < 3:
            logging.info(f"Semantic cleaned query too short ('{cleaned_query}'), keeping previous: '{query}'")
            cleaned_query = query
        
        # If no meaningful change was made, return the input
        if cleaned_query.lower() == query.lower():
            cleaned_query = query

        # --------------------------
        # Remove original project/location mentions before appending project name
        # --------------------------
        project_name_to_append = None
        proj_metadata = {}
        if inference_results:
            proj_metadata = inference_results.get("project_inference", {}).get("metadata") or {}
            matched_projects = proj_metadata.get("matched_projects", [])

            # Prefer metadata matches first
            metadata_matches = [p for p in matched_projects if p.get("match_type") == "metadata"]
            if metadata_matches:
                best_match = max(metadata_matches, key=lambda x: x.get("similarity", 0))
                if best_match.get("similarity", 0) >= 0.8:
                    project_name_to_append = best_match.get("project_name")
            else:
                # Fallback to fuzzy matches
                fuzzy_matches = [p for p in matched_projects if p.get("match_type") == "fuzzy"]
                if fuzzy_matches:
                    best_fuzzy = max(fuzzy_matches, key=lambda x: x.get("similarity", 0))
                    if best_fuzzy.get("similarity", 0) >= 0.8:
                        project_name_to_append = best_fuzzy.get("project_name")

            # Remove project/location mentions from query
            for match in matched_projects:
                entity = match.get("entity")
                if entity and entity.lower() in cleaned_query.lower():
                    # Escape for regex
                    pattern = re.escape(entity)
                    cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)

            # Clean extra spaces after removal
            cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

        # --------------------------
        # Append canonical project name
        # --------------------------
        if project_name_to_append:
            cleaned_query = f"{cleaned_query.strip()} {project_name_to_append}"
        
        logging.info(f"Semantic cleaning successful: '{query}' -> '{cleaned_query}'")
        return cleaned_query


# Global pipeline instance
inference_pipeline = InferencePipeline()
