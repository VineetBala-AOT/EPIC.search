"""
Base Search Handler

Contains common functionality shared across all search mode handlers.
"""
import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Generator, List, Optional, Any
from flask import current_app

from search_api.clients.vector_search_client import VectorSearchClient


class BaseSearchHandler(ABC):
    """Base class for all search mode handlers with common functionality."""
    
    @classmethod
    @abstractmethod
    def handle(cls, query: str, project_ids: Optional[List[str]] = None, 
               document_type_ids: Optional[List[str]] = None, 
               search_strategy: Optional[str] = None, 
               inference: Optional[List] = None, 
               ranking: Optional[Dict] = None, 
               metrics: Optional[Dict] = None,
               user_location: Optional[Dict] = None,
               location: Optional[Dict] = None, 
               project_status: Optional[str] = None, 
               years: Optional[List] = None) -> Dict[str, Any]:
        """
        Handle search request for this mode.
        
        Args:
            query: The user query
            project_ids: Optional user-provided project IDs
            document_type_ids: Optional user-provided document type IDs  
            search_strategy: Optional user-provided search strategy
            inference: Inference settings
            ranking: Optional ranking configuration
            metrics: Metrics dictionary to update
            user_location: Optional user location data
            location: Optional location parameter (user-provided takes precedence)
            project_status: Optional project status parameter (user-provided takes precedence)
            years: Optional years parameter (user-provided takes precedence)
            
        Returns:
            Complete response dictionary with search results
        """
        pass
    
    @classmethod
    def _execute_vector_search(cls, query: str, project_ids: Optional[List[str]],
                              document_type_ids: Optional[List[str]],
                              inference: Optional[List], ranking: Optional[Dict],
                              search_strategy: Optional[str], semantic_query: Optional[str],
                              metrics: Dict, location: Optional[str] = None,
                              user_location: Optional[Dict] = None,
                              project_status: Optional[str] = None,
                              years: Optional[List] = None,
                              _override_result: Optional[tuple] = None,
                              semantic_fetch_count: Optional[int] = None,
                              keyword_fetch_count: Optional[int] = None) -> Dict[str, Any]:
        """Execute vector search and process the response.

        Args:
            _override_result: Optional (documents, chunks, api_response) tuple.
                When provided the HTTP call to the vector API is skipped entirely
                and the supplied data is used directly (speculative search path).
        """
        current_app.logger.info("=== VECTOR SEARCH: Starting search execution ===")

        # Add LLM provider and model information
        metrics["llm_provider"] = os.getenv("LLM_PROVIDER", "ollama")
        if metrics["llm_provider"] == "openai":
            metrics["llm_model"] = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        else:
            metrics["llm_model"] = os.getenv("LLM_MODEL", "")

        current_app.logger.info(f"LLM Configuration - Provider: {metrics['llm_provider']}, Model: {metrics['llm_model']}")

        # Log the search parameters being passed
        current_app.logger.info(f"🔎 SEARCH PARAMS: project_ids={project_ids}")
        current_app.logger.info(f"🔎 SEARCH PARAMS: document_type_ids={document_type_ids}")
        current_app.logger.info(f"🔎 SEARCH PARAMS: search_strategy={search_strategy}")
        current_app.logger.info(f"🔎 SEARCH PARAMS: semantic_query={semantic_query}")
        current_app.logger.info(f"🔎 SEARCH PARAMS: location={location}, project_status={project_status}, years={years}")

        # Apply a default result cap if the caller didn't specify one.
        # Without a topN limit the vector API scans the entire dataset which is
        # very slow on large corpora — 15 focused results are enough for a summary.
        if not ranking:
            ranking = {"topN": 15}
        elif isinstance(ranking, dict) and "topN" not in ranking:
            ranking = {**ranking, "topN": 15}

        # Perform the vector DB search — use pre-fetched speculative data when available
        current_app.logger.info(f"🔎 SEARCH PARAMS: ranking={ranking}")
        search_start = time.time()
        if _override_result is not None:
            documents, document_chunks, vector_api_response = _override_result
            current_app.logger.info("⚡ VECTOR SEARCH: using speculative pre-fetched results (HTTP call skipped)")
        else:
            current_app.logger.info("Starting vector search...")
            documents, document_chunks, vector_api_response = VectorSearchClient.search(
                query=query,
                project_ids=project_ids,
                document_type_ids=document_type_ids,
                inference=inference,
                ranking=ranking,
                search_strategy=search_strategy,
                semantic_query=semantic_query,
                location=location,
                project_status=project_status,
                years=years,
                user_location=user_location,
                semantic_fetch_count=semantic_fetch_count,
                keyword_fetch_count=keyword_fetch_count,
            )
        search_duration = round((time.time() - search_start) * 1000, 2)
        current_app.logger.info(f"Vector search completed in {search_duration}ms")
        current_app.logger.info(f"Documents returned: {len(documents) if documents else 0}")
        current_app.logger.info(f"Document chunks returned: {len(document_chunks) if document_chunks else 0}")
        
        # Combine documents and chunks for backwards compatibility where needed
        documents_or_chunks = []
        if documents:
            documents_or_chunks.extend(documents)
        if document_chunks:
            documents_or_chunks.extend(document_chunks)
        current_app.logger.info(f"Total documents/chunks: {len(documents_or_chunks)}")
        current_app.logger.info(f"Type of documents_or_chunks: {type(documents_or_chunks)}")
        
        # Extract data from the complete vector API response
        if isinstance(vector_api_response, dict):
            vector_search_data = vector_api_response.get("vector_search", {})
            search_metrics = vector_search_data.get("search_metrics", {})
            search_quality = vector_search_data.get("search_quality", "unknown")
            project_inference = vector_search_data.get("project_inference", {})
            document_type_inference = vector_search_data.get("document_type_inference", {})
            
            # Extract additional vector search metadata
            original_query = vector_search_data.get("original_query", "")
            final_semantic_query = vector_search_data.get("final_semantic_query", "")
            semantic_cleaning_applied = vector_search_data.get("semantic_cleaning_applied", False)
            search_mode = vector_search_data.get("search_mode", "unknown")
            query_processed = vector_search_data.get("query_processed", False)
            inference_settings = vector_search_data.get("inference_settings", {})
            
            # Extract search_breakdown from metrics if available
            api_metrics = vector_api_response.get("metrics", {})
            search_breakdown = api_metrics.get("search_breakdown", {})
        else:
            # Fallback if vector_api_response is not a dict (handle string responses)
            current_app.logger.warning(f"Vector API response is not a dict: {type(vector_api_response)}")
            vector_search_data = {}
            search_metrics = {}
            search_quality = "unknown"
            project_inference = {}
            document_type_inference = {}
            original_query = ""
            final_semantic_query = ""
            semantic_cleaning_applied = False
            search_mode = "unknown"
            query_processed = False
            inference_settings = {}
            search_breakdown = {}
        
        # Determine the response type based on what's actually in the vector search data
        if isinstance(vector_api_response, dict) and "vector_search" in vector_api_response:
            vector_search_data = vector_api_response.get("vector_search", {})
            # Check what's actually in the response
            if "documents" in vector_search_data and vector_search_data["documents"]:
                response_type = "documents"
                documents_key = "documents"
            elif "document_chunks" in vector_search_data and vector_search_data["document_chunks"]:
                response_type = "document_chunks"
                documents_key = "document_chunks"
            else:
                response_type = "document_chunks"
                documents_key = "document_chunks"
        else:
            # Default fallback
            response_type = "document_chunks"
            documents_key = "document_chunks"
            
        current_app.logger.info(f"Determined response type: {response_type}, documents_key: {documents_key}")
        
        # Add all search metrics regardless of whether documents were found
        metrics["search_time_ms"] = search_duration
        metrics["search_breakdown"] = search_breakdown if search_breakdown else search_metrics
        metrics["search_quality"] = search_quality
        metrics["original_query"] = original_query
        metrics["final_semantic_query"] = final_semantic_query
        metrics["semantic_cleaning_applied"] = semantic_cleaning_applied
        metrics["search_mode"] = search_mode
        metrics["query_processed"] = query_processed
        metrics["inference_settings"] = inference_settings
        
        current_app.logger.info("=== VECTOR SEARCH: Search execution complete ===")
        
        return {
            "documents_or_chunks": documents_or_chunks,
            "documents_key": documents_key,
            "search_duration": search_duration,
            "search_quality": search_quality,
            "project_inference": project_inference,
            "document_type_inference": document_type_inference
        }
    
    @classmethod
    def _generate_agentic_summary(cls, documents_or_chunks: List, query: str,
                                 metrics: Dict, project_metadata: Dict = None) -> Dict[str, Any]:
        """Generate summary using LLM summarizer from generation package.

        Args:
            documents_or_chunks (list): Retrieved documents or document chunks
            query (str): The original search query
            metrics (dict): Metrics dictionary to update
            project_metadata (dict): Optional project metadata including description, status, etc.

        Returns:
            dict: Summary result containing response text or error info
        """
        current_app.logger.info("=== AGENTIC SUMMARY: Starting LLM summarizer from generation package ===")

        llm_start = time.time()
        current_app.logger.info(f"Using LLM summarizer for summary: {query}")
        current_app.logger.info(f"Number of documents/chunks for summary: {len(documents_or_chunks) if documents_or_chunks else 0}")
        if project_metadata:
            current_app.logger.info(f"🤖 LLM: Project metadata provided for summary context: {list(project_metadata.keys())}")

        try:
            from search_api.services.generation.factories import SummarizerFactory

            summarizer = SummarizerFactory.create_summarizer()

            # Build search context with project metadata if available
            search_context = {
                "context": "Agentic search summary",
                "search_strategy": "agentic",
                "total_documents": len(documents_or_chunks)
            }
            if project_metadata:
                search_context["project_metadata"] = project_metadata

            current_app.logger.info("🤖 LLM: Generating summary using LLM summarizer...")
            summary_result = summarizer.summarize_search_results(
                query=query,
                documents_or_chunks=documents_or_chunks,
                search_context=search_context
            )
            
            summary_text = summary_result['summary']
            method = summary_result['method']
            confidence = summary_result['confidence']
            provider = summary_result['provider']
            model = summary_result['model']
            
            current_app.logger.info(f"🤖 LLM: Summary generated using method: {method}, provider: {provider}, model: {model}, confidence: {confidence}")
            
            metrics["llm_time_ms"] = round((time.time() - llm_start) * 1000, 2)
            metrics["agentic_summary_method"] = method
            metrics["agentic_summary_confidence"] = confidence
            metrics["agentic_summary_provider"] = provider
            metrics["agentic_summary_model"] = model
            
            current_app.logger.info("=== AGENTIC SUMMARY: LLM summarizer generation complete ===")
            return {"response": summary_text}
                
        except Exception as e:
            # Log the error and return error info
            current_app.logger.error(f"🤖 LLM: Summary generation error: {str(e)}")
            current_app.logger.error(f"🤖 LLM: Summary error type: {type(e).__name__}")
            current_app.logger.error(f"🤖 LLM: Summary error traceback: {traceback.format_exc()}")
            
            metrics["agentic_summary_error"] = str(e)
            metrics["llm_time_ms"] = round((time.time() - llm_start) * 1000, 2)
            
            current_app.logger.info("=== AGENTIC SUMMARY: Error occurred, returning fallback ===")
            return {
                "error": str(e),
                "fallback_response": "An error occurred while generating the summary. Please try again later."
            }
    
    @classmethod
    def _generate_agentic_summary_stream(
        cls,
        documents_or_chunks: List,
        query: str,
        project_metadata: Optional[Dict] = None
    ) -> Generator[str, None, None]:
        """Stream the AI summary token-by-token using the configured summarizer.

        Falls back to a single-chunk yield if the summarizer does not support
        streaming (e.g. Ollama).

        Yields:
            str: Text token chunks.
        """
        from search_api.services.generation.factories import SummarizerFactory

        summarizer = SummarizerFactory.create_summarizer()

        if hasattr(summarizer, "summarize_documents_stream"):
            yield from summarizer.summarize_documents_stream(
                documents=documents_or_chunks,
                query=query,
                project_metadata=project_metadata,
            )
        else:
            # Non-streaming fallback: return entire summary as one chunk
            result = summarizer.summarize_search_results(
                query=query,
                documents_or_chunks=documents_or_chunks,
                search_context={"project_metadata": project_metadata} if project_metadata else {},
            )
            yield result.get("summary", "")

    @classmethod
    def _generate_rag_summary(cls, documents_or_chunks: List, query: str,
                             metrics: Dict) -> Dict[str, Any]:
        """Generate basic summary for RAG mode (non-agentic).
        
        Args:
            documents_or_chunks (list): Retrieved documents or document chunks
            query (str): The original search query
            metrics (dict): Metrics dictionary to update
            
        Returns:
            dict: Summary result containing response text
        """
        current_app.logger.info("=== RAG SUMMARY: Generating basic summary ===")
        
        llm_start = time.time()
        current_app.logger.info(f"Using basic response generation for query: {query}")
        current_app.logger.info(f"Number of documents/chunks for basic processing: {len(documents_or_chunks) if documents_or_chunks else 0}")
        
        try:
            # Use the basic response processing method
            response = cls._process_basic_response(documents_or_chunks, query)
            
            metrics["llm_time_ms"] = round((time.time() - llm_start) * 1000, 2)
            
            current_app.logger.info("=== RAG SUMMARY: Basic summary generation complete ===")
            return {"response": response}
            
        except Exception as e:
            current_app.logger.error(f"Basic summary error: {str(e)}")
            metrics["llm_time_ms"] = round((time.time() - llm_start) * 1000, 2)
            
            return {
                "error": str(e),
                "fallback_response": "An error occurred while processing the documents."
            }
    
    @classmethod
    def _process_basic_response(cls, documents_or_chunks: List, query: str) -> str:
        """Process documents for basic mode - just return a simple summary.
        
        Args:
            documents_or_chunks (list): Retrieved documents or document chunks
            query (str): The original search query
            
        Returns:
            str: Basic response string with document count summary
        """
        doc_count = len(documents_or_chunks) if documents_or_chunks else 0
                
        current_app.logger.info(f"Basic mode: returning {doc_count} documents / sections without LLM processing")
        
        return f"Found {doc_count} documents / sections matching your query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
    
    @classmethod
    def _log_agentic_summary(cls, metrics: Dict, search_duration: float, 
                            search_quality: str, documents_or_chunks: List, 
                            query: str) -> None:
        """Log detailed summary for agentic mode with comprehensive metrics and analysis.
        
        Args:
            metrics (dict): Performance and processing metrics
            search_duration (float): Vector search duration in milliseconds
            search_quality (str): Assessed quality of the search results
            documents_or_chunks (list): Retrieved documents or document chunks
            query (str): The original search query
        """
        doc_count = len(documents_or_chunks) if documents_or_chunks else 0
        total_ms = metrics.get("total_time_ms", 0)
        timing = metrics.get("timing", {})

        current_app.logger.info(
            f"🤖 DONE | query='{query[:80]}{'...' if len(query) > 80 else ''}' | "
            f"docs={doc_count} | quality={search_quality} | total={total_ms}ms"
        )

        if timing:
            # Pretty breakdown table in logs
            rows = [
                ("parallel_setup",    timing.get("parallel_setup_ms", 0)),
                ("  relevance_check", timing.get("relevance_check_ms", 0)),
                ("early_detection",   timing.get("early_detection_ms", 0)),
                ("llm_extraction",    timing.get("llm_extraction_ms", 0)),
                ("post_llm_process",  timing.get("post_llm_processing_ms", 0)),
                ("vector_search",     timing.get("vector_search_ms", 0)),
                ("metadata_lookup",   timing.get("project_metadata_ms", 0)),
                ("summary_gen",       timing.get("summary_generation_ms", 0)),
            ]
            current_app.logger.info("⏱️ Stage breakdown:")
            for label, ms in rows:
                pct = f"{ms/total_ms*100:.0f}%" if total_ms else "?"
                bar = "█" * min(int(ms / max(total_ms, 1) * 30), 30)
                current_app.logger.info(f"   {label:<22} {ms:>7.0f}ms  {pct:>4}  {bar}")
            current_app.logger.info(f"   {'TOTAL':<22} {total_ms:>7.0f}ms  100%")
            if timing.get("speculative_used"):
                current_app.logger.info("   ⚡ speculative search result was reused (vector round-trip skipped)")
        else:
            current_app.logger.info(f"🤖 Vector search: {search_duration}ms | LLM: {metrics.get('llm_time_ms', 'N/A')}ms")

    @classmethod
    def _log_basic_summary(cls, documents_or_chunks: List, query: str) -> None:
        """Log basic summary with document count for non-agentic mode.
        
        Args:
            documents_or_chunks (list): Retrieved documents or document chunks
            query (str): The original search query
        """
        doc_count = len(documents_or_chunks) if documents_or_chunks else 0
        doc_type = "documents" if documents_or_chunks and hasattr(documents_or_chunks[0], 'document_id') else "document sections"
        
        current_app.logger.info(f"Search completed: {doc_count} {doc_type} returned for query")
        current_app.logger.info(f"Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")