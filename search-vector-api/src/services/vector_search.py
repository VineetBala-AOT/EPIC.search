"""Vector search implementation with modular strategy architecture.

This module serves as the main entry point for search functionality and contains shared utility
functions used across multiple search strategies. The core search functionality has been 
modularized into strategy-specific implementations located in the search_strategies package.

Architecture:
- Main search() function: Routes requests to appropriate strategy implementations via factory pattern
- Shared utility functions: Common operations used by multiple search strategies
- Direct metadata search: Special case handling for generic document browsing requests

The search strategies implement various approaches including:
1. Hybrid semantic fallback (default): Document filtering → semantic search → keyword fallback
2. Hybrid keyword fallback: Document filtering → keyword search → semantic fallback  
3. Semantic only: Pure semantic vector search across all chunks
4. Keyword only: Pure keyword-based search
5. Hybrid parallel: Parallel semantic and keyword search with result merging
6. Document only: Metadata-based document retrieval for generic requests

Shared utilities include:
- Document-level search using pre-computed metadata
- Semantic search across all document chunks
- Keyword search with configurable extraction methods
- Cross-encoder re-ranking for relevance optimization
- Result formatting and metadata extraction
- Post-search filtering and similarity search functions
"""

import time
import logging
import re

from flask import current_app
from .keywords.query_keyword_extractor import get_keywords
from .re_ranker import rerank_results_with_metrics
from .vector_store import VectorStore


def get_document_type_name(document_metadata, chunk_metadata=None):
    """Map document type ID to document type name using available metadata.
    
    This function supports multiple metadata structures to ensure robust document type
    population across different data sources:
    
    Priority Order:
    1. chunk_metadata.document_metadata.document_type (preferred nested structure)
    2. chunk_metadata.document_type (legacy direct field for backward compatibility)
    3. document_metadata.documentType (document-level direct field)
    4. document_metadata.document_type_id (lookup via document type mapping)
    
    Args:
        document_metadata (dict): The document metadata containing document_type_id (from documents table)
        chunk_metadata (dict, optional): The chunk metadata that should contain a document_metadata object
                                        with document_type field, or a direct document_type field for legacy support
        
    Returns:
        str: The document type name/display name, or None if not found
    """
    # First try to get document type from chunk metadata's document_metadata object
    if chunk_metadata:
        # Check if chunk metadata contains a document_metadata object
        chunk_doc_metadata = chunk_metadata.get('document_metadata')
        if chunk_doc_metadata and isinstance(chunk_doc_metadata, dict):
            document_type = chunk_doc_metadata.get('document_type')
            if document_type:
                return document_type
        
        # Fallback: try direct document_type field (legacy support)
        direct_document_type = chunk_metadata.get('document_type')
        if direct_document_type:
            return direct_document_type
    
    # Then try to get document type from document_metadata (document-level results)
    if document_metadata:
        # First try the direct documentType field (most reliable)
        document_type = document_metadata.get('documentType')
        if document_type:
            return document_type
            
        # Fallback to lookup by document_type_id
        document_type_id = document_metadata.get('document_type_id')
        if document_type_id:
            try:
                # Import the document types lookup dictionary using absolute import
                from src.utils.document_types import DOCUMENT_TYPE_LOOKUP
                
                # Look up the document type name by ID
                document_type_name = DOCUMENT_TYPE_LOOKUP.get(document_type_id)
                
                if document_type_name:
                    return document_type_name
                    
            except ImportError:
                # Try alternative import paths
                try:
                    from utils.document_types import DOCUMENT_TYPE_LOOKUP
                    document_type_name = DOCUMENT_TYPE_LOOKUP.get(document_type_id)
                    if document_type_name:
                        return document_type_name
                except ImportError:
                    pass
            except Exception:
                pass
            
            # If lookup fails, return the ID as fallback
            return document_type_id
    
    # No document type found
    return None


def get_document_display_name(document_metadata, chunk_metadata=None):
    """Extract display_name from document metadata or chunk metadata.
    
    This function supports multiple metadata structures to ensure robust display_name
    population across different data sources:
    
    Priority Order:
    1. chunk_metadata.document_metadata.display_name (preferred nested structure)
    2. chunk_metadata.display_name (legacy direct field for backward compatibility)
    3. document_metadata.display_name (document-level direct field)
    
    Args:
        document_metadata (dict): The document metadata containing display_name (from documents table)
        chunk_metadata (dict, optional): The chunk metadata that should contain a document_metadata object
                                        with display_name field, or a direct display_name field for legacy support
        
    Returns:
        str: The document display name, or None if not found
    """
    # First try to get display_name from chunk metadata's document_metadata object
    if chunk_metadata:
        # Check if chunk metadata contains a document_metadata object
        chunk_doc_metadata = chunk_metadata.get('document_metadata')
        if chunk_doc_metadata and isinstance(chunk_doc_metadata, dict):
            display_name = chunk_doc_metadata.get('display_name')
            if display_name:
                return display_name
        
        # Fallback: try direct display_name field (legacy support)
        direct_display_name = chunk_metadata.get('display_name')
        if direct_display_name:
            return direct_display_name
    
    # Then try to get display_name from document_metadata (document-level results)
    if document_metadata:
        display_name = document_metadata.get('display_name')
        if display_name:
            return display_name
    
    # No display_name found
    return None


def search(question, project_ids=None, document_type_ids=None, min_relevance_score=None, top_n=None, search_strategy=None, semantic_query=None):
    """Main search entry point that routes requests to appropriate search strategies.
    
    This function serves as the primary interface for search functionality. It handles:
    1. Parameter validation and configuration setup
    2. Strategy selection and execution via the search strategy factory
    3. Fallback error handling
    
    The actual search implementation is delegated to modular strategy classes located in
    the search_strategies package. Each strategy implements its own multi-stage pipeline
    optimized for different use cases (semantic-first, keyword-first, parallel, etc.).
    
    Strategy Options:
    - HYBRID_SEMANTIC_FALLBACK (default): Document filtering → semantic → keyword fallback
    - HYBRID_KEYWORD_FALLBACK: Document filtering → keyword → semantic fallback
    - SEMANTIC_ONLY: Pure semantic vector search
    - KEYWORD_ONLY: Pure keyword-based search  
    - HYBRID_PARALLEL: Parallel semantic and keyword search
    - DOCUMENT_ONLY: Metadata-based document browsing (explicit user choice only)
    
    Args:
        question (str): The search query text
        project_ids (list, optional): List of project IDs to filter results.
                                    If None or empty, searches across all projects.
        document_type_ids (list, optional): List of document type IDs to filter results.
                                          If None or empty, searches across all document types.
        min_relevance_score (float, optional): Minimum relevance score threshold for filtering results.
                                             If None, uses the MIN_RELEVANCE_SCORE config value.
        top_n (int, optional): Maximum number of results to return after ranking.
                              If None, uses the TOP_RECORD_COUNT config value.
        search_strategy (str, optional): The search strategy to use.
                                       Valid values: 'HYBRID_SEMANTIC_FALLBACK', 'HYBRID_KEYWORD_FALLBACK',
                                       'SEMANTIC_ONLY', 'KEYWORD_ONLY', 'HYBRID_PARALLEL', 'DOCUMENT_ONLY'.
                                       If None, uses 'HYBRID_SEMANTIC_FALLBACK'.
        semantic_query (str, optional): Override the automatic semantic query cleaning with a user-provided
                                      optimized query for vector search. If None, the system will automatically
                                      clean and optimize the question for semantic search.
        
    Returns:
        tuple: A tuple containing:
            - list: Formatted search results as a list of dictionaries
            - dict: Search performance metrics in milliseconds for each search stage
    """
    metrics = {}
    start_time = time.time()
    
    # Use strongly typed configuration properties
    doc_limit = current_app.search_settings.keyword_fetch_count  # Number of documents to find
    chunk_limit = current_app.search_settings.semantic_fetch_count  # Number of chunks to return
    
    # Use provided top_n parameter or fall back to config value
    original_top_n = top_n
    if top_n is None:
        top_n = current_app.search_settings.top_record_count
    
    # Ensure top_n is valid - provide a safe default if config is invalid
    if top_n is None or not isinstance(top_n, int) or top_n <= 0:
        from src.utils.config import _Config
        top_n = _Config.TOP_RECORD_COUNT  # Use config default
    
    # Use provided min_relevance_score parameter or fall back to config value
    original_min_relevance_score = min_relevance_score
    if min_relevance_score is None:
        min_relevance_score = current_app.search_settings.min_relevance_score
    
    # Ensure min_relevance_score is valid - just check it's a number
    if min_relevance_score is None or not isinstance(min_relevance_score, (int, float)):
        from src.utils.config import _Config
        min_relevance_score = _Config.MIN_RELEVANCE_SCORE  # Use config default
    
    # Set default search strategy if none provided
    if search_strategy is None:
        search_strategy = current_app.search_settings.default_search_strategy
    
    # Add search strategy to metrics
    metrics["search_strategy"] = search_strategy
    
    # Add ranking configuration to metrics
    ranking_config = {
        "minScore": {
            "value": min_relevance_score,
            "source": "parameter" if original_min_relevance_score is not None else "environment"
        },
        "topN": {
            "value": top_n,
            "source": "parameter" if original_top_n is not None else "environment"
        }
    }
    metrics["ranking_config"] = ranking_config
    
    # Instantiate VectorStore
    vec_store = VectorStore()
    
    # Check if this should be a direct metadata search (DOCUMENT_ONLY strategy)
    # This only happens when the user explicitly requests DOCUMENT_ONLY strategy
    explicit_document_only = (search_strategy == "DOCUMENT_ONLY")
    
    if explicit_document_only:
        
        import logging
        logging.info(f"Direct metadata search mode: User explicitly requested DOCUMENT_ONLY strategy for query: '{question}'")
        logging.info(f"Project IDs: {project_ids}, Document Type IDs: {document_type_ids}")
              
        # Perform direct metadata search
        documents, metadata_search_time = perform_direct_metadata_search(
            vec_store, project_ids, document_type_ids, doc_limit
        )
        metrics["metadata_search_ms"] = metadata_search_time
        metrics["search_mode"] = "direct_metadata"
        metrics["strategy_used"] = "DOCUMENT_ONLY (user requested)"
        
        # Format the document results directly (no re-ranking needed for date-ordered results)
        format_start = time.time()
        formatted_data = format_document_data(documents)
        metrics["formatting_ms"] = round((time.time() - format_start) * 1000, 2)
        
        # Total time
        metrics["total_search_ms"] = round((time.time() - start_time) * 1000, 2)
        
        logging.info(f"Direct metadata search completed: {len(formatted_data)} documents found")
        
        return formatted_data, metrics
    
    # Route to appropriate search strategy
    import logging
    logging.info(f"vector_search.search - Executing search strategy: {search_strategy}")
    logging.info(f"vector_search.search - Search parameters: project_ids={project_ids} (type: {type(project_ids)}), document_type_ids={document_type_ids}")
    if project_ids:
        logging.info(f"vector_search.search - project_ids length: {len(project_ids)}, values: {project_ids}")
    logging.info(f"vector_search.search - Search query: '{question}'")
    
    # Add query information to metrics for debugging
    metrics["search_query"] = question
    metrics["search_strategy"] = search_strategy
    metrics["project_filter_applied"] = project_ids is not None and len(project_ids) > 0
    metrics["document_type_filter_applied"] = document_type_ids is not None and len(document_type_ids) > 0
    metrics["keyword_extraction_method"] = current_app.model_settings.document_keyword_extraction_method
    
    # Use the strategy factory to get and execute the appropriate strategy
    from .search_strategies import get_search_strategy
    
    try:
        strategy = get_search_strategy(search_strategy)
        return strategy.execute(
            question=question,
            vec_store=vec_store,
            project_ids=project_ids,
            document_type_ids=document_type_ids,
            doc_limit=doc_limit,
            chunk_limit=chunk_limit,
            top_n=top_n,
            min_relevance_score=min_relevance_score,
            metrics=metrics,
            start_time=start_time,
            semantic_query=semantic_query
        )
    except Exception as e:
        # Fallback to default strategy if something goes wrong
        logging.error(f"Error executing search strategy '{search_strategy}': {e}")
        logging.warning("Falling back to default strategy execution")
        metrics["strategy_error"] = str(e)
        metrics["strategy_fallback"] = True
        
        # Try to get the default strategy and execute it
        try:
            default_strategy = get_search_strategy("HYBRID_PARALLEL")
            return default_strategy.execute(
                question=question,
                vec_store=vec_store,
                project_ids=project_ids,
                document_type_ids=document_type_ids,
                doc_limit=doc_limit,
                chunk_limit=chunk_limit,
                top_n=top_n,
                min_relevance_score=min_relevance_score,
                metrics=metrics,
                start_time=start_time,
                semantic_query=semantic_query
            )
        except Exception as fallback_error:
            logging.error(f"Critical error: Default strategy fallback also failed: {fallback_error}")
            # Return empty results rather than crashing
            return [], {"error": "All search strategies failed", "details": str(fallback_error)}


def perform_keyword_search(vec_store, table_name, query, limit, project_ids=None, document_type_ids=None):
    """Perform keyword search using vector store with optional project and document type filtering.
    
    Executes a keyword-based search using PostgreSQL's full-text search capabilities
    through the VectorStore interface. The keyword extraction method used will match
    the method used for document keywords to ensure optimal matching.
    
    Args:
        vec_store (VectorStore): The vector store instance
        table_name (str): The database table to search in
        query (str): The search query text
        limit (int): Maximum number of results to return
        project_ids (list, optional): List of project IDs to filter results
        document_type_ids (list, optional): List of document type IDs to filter results
        
    Returns:
        tuple: A tuple containing:
            - DataFrame: Search results with id, content, search_type, and metadata columns
            - float: Time taken in milliseconds
    """
    # Debug logging for project_ids parameter tracking
    logging.info(f"perform_keyword_search - Called with project_ids: {project_ids} (type: {type(project_ids)})")
    if project_ids:
        logging.info(f"perform_keyword_search - project_ids length: {len(project_ids)}, values: {project_ids}")
    
    # Use the appropriate keyword extraction method based on document configuration
    extraction_method = current_app.model_settings.document_keyword_extraction_method
    
    # Use the get_keywords function which now handles method selection internally
    weighted_keywords = get_keywords(query)
    logging.info(f"Using {extraction_method} keyword extraction for query: {query}")
    logging.info(f"Extracted keywords: {weighted_keywords} (method: {extraction_method})")

    # Extract just the keywords from the (keyword, score) tuples
    keywords_only = [keyword for keyword, score in weighted_keywords] if weighted_keywords else []

    start_time = time.time()
    
    # Build predicates for project and document type filtering (same as semantic search)
    predicates = {}
    if project_ids:
        predicates['project_ids'] = project_ids
        logging.info(f"perform_keyword_search - Added project_ids to predicates: {predicates['project_ids']}")
    if document_type_ids:
        predicates['document_type_ids'] = document_type_ids
        logging.info(f"perform_keyword_search - Added document_type_ids to predicates: {predicates['document_type_ids']}")
    
    logging.info(f"perform_keyword_search - Final predicates dict: {predicates}")
    
    # Use the enhanced keyword search method that supports predicates
    keyword_results = vec_store.keyword_search_with_predicates(
        table_name, query, limit=limit, predicates=predicates, return_dataframe=True, 
        weighted_keywords=keywords_only
    )
    
    if not keyword_results.empty:
        keyword_results["search_type"] = "keyword"
        
        # Ensure we have the expected columns - handle both old and new result formats
        if "metadata" not in keyword_results.columns:
            # Add empty metadata column for compatibility
            keyword_results["metadata"] = [{}] * len(keyword_results)
        
        expected_columns = ["id", "content", "search_type", "metadata"]
        available_columns = [col for col in expected_columns if col in keyword_results.columns]
        keyword_results = keyword_results[available_columns]
    
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    logging.info(f"perform_keyword_search - Returned {len(keyword_results) if not keyword_results.empty else 0} results")
    return keyword_results, elapsed_ms


def perform_semantic_search(vec_store, table_name, query, limit):
    """Perform semantic search using vector store.
    
    Executes a semantic vector-based search using pgvector for similarity matching
    through the VectorStore interface.
    
    Args:
        vec_store (VectorStore): The vector store instance
        table_name (str): The database table to search in
        query (str): The search query text
        limit (int): Maximum number of results to return
        
    Returns:
        tuple: A tuple containing:
            - DataFrame: Search results with id, content, search_type, and metadata columns
            - float: Time taken in milliseconds
    """
    start_time = time.time()
    
    # Perform semantic search
    semantic_results = vec_store.semantic_search(
        table_name, query, limit=limit, return_dataframe=True
    )
    
    semantic_results["search_type"] = "semantic"
    semantic_results = semantic_results[["id", "content", "search_type", "metadata"]]
    
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    return semantic_results, elapsed_ms


def limit_chunks_per_document(results, max_chunks_per_doc):
    """Limit the number of chunks returned per document to prevent semantic bias.
    
    This function prevents a single document from dominating search results
    by limiting how many chunks can be returned from each document.
    
    Args:
        results (DataFrame): Combined search results containing document chunks
        max_chunks_per_doc (int): Maximum number of chunks allowed per document
        
    Returns:
        DataFrame: Filtered results with chunk limits applied per document
    """
    if results.empty or max_chunks_per_doc <= 0:
        return results
    
    # Track chunks per document
    document_counts = {}
    filtered_rows = []
    
    for idx, row in results.iterrows():
        # Extract document_id from metadata
        metadata = row.get('metadata', {})
        doc_id = metadata.get('document_id')
        
        if doc_id is None:
            # If no document_id, include the chunk
            filtered_rows.append(row)
        else:
            # Check if we've reached the limit for this document
            current_count = document_counts.get(doc_id, 0)
            if current_count < max_chunks_per_doc:
                filtered_rows.append(row)
                document_counts[doc_id] = current_count + 1
    
    # Return filtered results as a new DataFrame
    if filtered_rows:
        import pandas as pd
        return pd.DataFrame(filtered_rows).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=results.columns)


def perform_reranking(query, combined_results, top_n, min_relevance_score=None):
    """Re-rank the results using the cross-encoder re-ranker model.
    
    Takes the combined search results and re-ranks them using a more powerful
    cross-encoder model to improve the relevance ordering.
    
    Args:
        query (str): The original search query
        combined_results (DataFrame): Combined search results 
        top_n (int): Number of top results to return after re-ranking
        min_relevance_score (float, optional): Minimum relevance score threshold for filtering results.
                                             If None, uses the MIN_RELEVANCE_SCORE config value.
        
    Returns:
        tuple: A tuple containing:
            - DataFrame: Re-ranked search results limited to top_n
            - float: Time taken in milliseconds
            - dict: Filtering metrics including exclusion counts
    """
    start_time = time.time()
    
    # Check if input is empty
    if combined_results.empty:
        import logging
        logging.warning("perform_reranking: received empty DataFrame")
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        empty_metrics = {
            "total_chunks_before_filtering": 0,
            "excluded_chunks_count": 0,
            "exclusion_percentage": 0.0,
            "final_chunk_count": 0,
            "score_range_excluded": None,
            "score_range_included": None
        }
        return combined_results, elapsed_ms, empty_metrics
    
    # Apply document chunk limit to prevent semantic bias
    max_chunks_per_doc = current_app.search_settings.max_chunks_per_document
    combined_results = limit_chunks_per_document(combined_results, max_chunks_per_doc)
    
    # Re-rank the results using the batch size from config and get metrics
    batch_size = current_app.search_settings.reranker_batch_size
    reranked_results, filtering_metrics = rerank_results_with_metrics(query, combined_results, top_n, batch_size=batch_size, min_relevance_score=min_relevance_score)
    
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    return reranked_results, elapsed_ms, filtering_metrics


def format_data(data):
    """Format the search results into a list of dictionaries.
    
    Args:
        data (DataFrame): Search results data
        
    Returns:
        list: List of dictionaries containing formatted document information
    """
    result = []
    
    # Check if data is empty
    if len(data) == 0:
        return result
    
    # Check if results have low confidence (all scores were very low)
    has_low_confidence = False
    if hasattr(data, 'iterrows') and not data.empty:
        # Check if any row has the low_confidence flag
        has_low_confidence = any(row.get('low_confidence', False) for _, row in data.iterrows())
    
    # Process DataFrame directly instead of converting to numpy array
    if hasattr(data, 'iterrows'):
        for idx, row in data.iterrows():
            # Get metadata - it should be in the 'metadata' column
            metadata = row.get('metadata', {})
            document_metadata = row.get('document_metadata', {})

            project_id = metadata.get("project_id")
            document_id = metadata.get("document_id")
            project_name = metadata.get("project_name")
            document_type = get_document_type_name(document_metadata, metadata)  # Pass both document_metadata and chunk metadata
            document_name = metadata.get("document_name")  # Human-readable filename
            document_saved_name = metadata.get("doc_internal_name")  # Technical/hash filename
            document_display_name = get_document_display_name(document_metadata, metadata)  # Extract display_name
            page_number = metadata.get("page_number")
            proponent_name = metadata.get("proponent_name")
            s3_key = metadata.get("s3_key")        

            # Append to a list or process as needed
            formatted_result = {
                "document_id": document_id,
                "document_type": document_type,
                "document_name": document_name,
                "document_saved_name": document_saved_name,
                "document_display_name": document_display_name,
                "page_number": page_number,
                "project_id": project_id,
                "project_name": project_name,
                "proponent_name": proponent_name,
                "s3_key": s3_key,
                "content": row.get('content', ''),
                "relevance_score": float(row.get('relevance_score', 0.0)),
                "search_mode": "semantic"  # Indicate this was a semantic search
            }
            
            # Add low confidence warning if applicable
            if has_low_confidence:
                formatted_result["search_quality"] = "low_confidence"
                formatted_result["search_note"] = "Results may not be highly relevant to your query. Consider refining your search terms."
            
            result.append(formatted_result)
    else:
        # Fallback to numpy array processing for backward compatibility
        import numpy as np
        documents = np.array(data)
        
        for i, row in enumerate(documents):
            # Try to intelligently determine the structure based on row length and content
            if len(row) >= 6:
                # Semantic search format with document_metadata: id, metadata, content, embedding, document_metadata, similarity
                metadata = row[1] if len(row) > 1 else {}
                content = row[2] if len(row) > 2 else ''
                document_metadata = row[4] if len(row) > 4 else {}
                relevance_score = float(row[5]) if len(row) > 5 else 0.0
            elif len(row) >= 5:
                # Check if this is a 5-column format with document_metadata or without
                # If row[3] looks like a dict/JSON, it might be document_metadata
                # If row[4] is a float, it's likely the score
                try:
                    score_candidate = float(row[4])
                    # This looks like: id, content, metadata, document_metadata, rank
                    content = row[1] if len(row) > 1 else ''
                    metadata = row[2] if len(row) > 2 else {}
                    document_metadata = row[3] if len(row) > 3 else {}
                    relevance_score = score_candidate
                except (ValueError, TypeError):
                    # This looks like: id, metadata, content, embedding, similarity (no document_metadata)
                    metadata = row[1] if len(row) > 1 else {}
                    content = row[2] if len(row) > 2 else ''
                    document_metadata = {}
                    relevance_score = float(row[4]) if len(row) > 4 else 0.0
            else:
                # Fallback for unknown format
                metadata = row[1] if len(row) > 1 else {}
                content = row[2] if len(row) > 2 else ''
                document_metadata = {}
                relevance_score = 0.0

            project_id = metadata.get("project_id")
            document_id = metadata.get("document_id")
            project_name = metadata.get("project_name")
            document_type = get_document_type_name(document_metadata, metadata)  # Pass both document_metadata and chunk metadata
            document_name = metadata.get("document_name")  # Human-readable filename
            document_saved_name = metadata.get("doc_internal_name")  # Technical/hash filename
            document_display_name = get_document_display_name(document_metadata, metadata)  # Extract display_name
            page_number = metadata.get("page_number")
            proponent_name = metadata.get("proponent_name")
            s3_key = metadata.get("s3_key")        

            # Append to a list or process as needed
            result.append(
                {
                    "document_id": document_id,
                    "document_type": document_type,
                    "document_name": document_name,
                    "document_saved_name": document_saved_name,
                    "document_display_name": document_display_name,
                    "page_number": page_number,
                    "project_id": project_id,
                    "project_name": project_name,
                    "proponent_name": proponent_name,
                    "s3_key": s3_key,
                    "content": content,
                    "relevance_score": relevance_score,
                    "search_mode": "semantic"  # Indicate this was a semantic search
                }
            )
    
    return result


def perform_document_level_search(vec_store, query, limit, project_ids=None, document_type_ids=None):
    """Perform document-level search using keywords, tags, and headings.
    
    Searches the documents table using pre-computed document-level metadata
    to identify the most relevant documents before searching their chunks.
    
    Args:
        vec_store (VectorStore): The vector store instance
        query (str): The search query text
        limit (int): Maximum number of documents to return
        project_ids (list, optional): List of project IDs to filter results
        document_type_ids (list, optional): List of document type IDs to filter results
        
    Returns:
        tuple: A tuple containing:
            - DataFrame: Document search results
            - float: Time taken in milliseconds
    """
    start_time = time.time()
    
    # Debug logging for project_ids parameter tracking
    logging.info(f"perform_document_level_search - Called with project_ids: {project_ids} (type: {type(project_ids)})")
    if project_ids:
        logging.info(f"perform_document_level_search - project_ids length: {len(project_ids)}, values: {project_ids}")
    
    # Build predicates for project and document type filtering
    predicates = {}
    if project_ids:
        predicates['project_ids'] = project_ids  # Pass as a special key
        logging.info(f"perform_document_level_search - Added project_ids to predicates: {predicates['project_ids']}")
    if document_type_ids:
        predicates['document_type_ids'] = document_type_ids  # Pass as a special key
        logging.info(f"perform_document_level_search - Added document_type_ids to predicates: {predicates['document_type_ids']}")
    
    logging.info(f"perform_document_level_search - Final predicates dict: {predicates}")
    
    # Perform document-level search
    document_results = vec_store.document_level_search(
        query, limit=limit, predicates=predicates, return_dataframe=True
    )
    
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    logging.info(f"perform_document_level_search - Returned {len(document_results) if not document_results.empty else 0} documents")
    return document_results, elapsed_ms


def perform_semantic_search_all_chunks(vec_store, query, limit, project_ids=None, document_type_ids=None):
    """Perform semantic search across all document chunks.
    
    Performs semantic vector search across all document chunks in the database
    when document-level filtering doesn't find relevant documents.
    This ensures comprehensive coverage even when the document-level
    keywords/tags don't match the query terms.
    
    Args:
        vec_store (VectorStore): The vector store instance
        query (str): The search query text
        limit (int): Maximum number of results to return
        project_ids (list, optional): List of project IDs to filter results
        document_type_ids (list, optional): List of document type IDs to filter results
        
    Returns:
        tuple: A tuple containing:
            - DataFrame: Search results with id, content, search_type, and metadata columns
            - float: Time taken in milliseconds
    """
    start_time = time.time()
    
    # Use the configured table name for chunks
    table_name = current_app.vector_settings.vector_table_name
    
    # Build predicates for project and document type filtering
    predicates = {}
    if project_ids:
        predicates['project_ids'] = project_ids  # Pass as a special key
    if document_type_ids:
        predicates['document_type_ids'] = document_type_ids  # Pass as a special key
    
    # Debug logging
    import logging
    logging.info(f"Semantic search all chunks - table: {table_name}, query: '{query}', limit: {limit}, predicates: {predicates}")
    
    # Perform semantic search on all chunks
    semantic_results = vec_store.semantic_search(
        table_name, query, limit=limit, predicates=predicates, return_dataframe=True
    )
    
    if not semantic_results.empty:
        semantic_results["search_type"] = "semantic"
        
        # Add empty document_metadata column for compatibility with format_data function  
        semantic_results["document_metadata"] = [{}] * len(semantic_results)
        
        # For semantic search on all chunks, semantic_search returns: [id, metadata, content, embedding, similarity]
        # We need to reorder to: [id, content, search_type, similarity, metadata, document_metadata]
        semantic_results = semantic_results[["id", "content", "search_type", "similarity", "metadata", "document_metadata"]]
    
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    return semantic_results, elapsed_ms


def document_similarity_search(document_id, project_ids=None, limit=10):
    """Find documents similar to the specified document using document-level embeddings.
    
    This function performs cosine similarity search on document-level embeddings
    to find documents that are semantically similar to the source document.
    
    Args:
        document_id (str): The ID of the document to find similar documents for
        project_ids (list, optional): List of project IDs to filter results
        limit (int): Maximum number of similar documents to return
        
    Returns:
        tuple: A tuple containing:
            - list: Formatted similar documents as a list of dictionaries
            - dict: Search performance metrics in milliseconds
    """
    import logging
    
    metrics = {}
    start_time = time.time()
    
    # Debug logging
    logging.info(f"Document similarity search started for document_id: {document_id}, project_ids: {project_ids}, limit: {limit}")
    
    # Instantiate VectorStore
    vec_store = VectorStore()
    
    # Step 1: Get the embedding for the source document
    source_embedding, embedding_time = get_document_embedding(vec_store, document_id)
    metrics["embedding_retrieval_ms"] = embedding_time
    
    if source_embedding is None:
        # Document not found
        logging.warning(f"Document embedding not found for document_id: {document_id}")
        metrics["total_search_ms"] = round((time.time() - start_time) * 1000, 2)
        return [], metrics
    
    # Step 2: Perform similarity search against other documents
    similar_docs, similarity_time = perform_document_similarity_search(
        vec_store, source_embedding, document_id, project_ids, limit
    )
    metrics["similarity_search_ms"] = similarity_time
    
    # Debug logging for similarity search results
    logging.info(f"Document similarity search found {len(similar_docs)} similar documents")
    
    # Step 3: Format the results
    format_start = time.time()
    formatted_docs = format_similar_documents(similar_docs)
    metrics["formatting_ms"] = round((time.time() - format_start) * 1000, 2)
    
    # Total time
    metrics["total_search_ms"] = round((time.time() - start_time) * 1000, 2)
    
    logging.info(f"Document similarity search completed. Total results: {len(formatted_docs)}, Total time: {metrics['total_search_ms']}ms")
    
    return formatted_docs, metrics


def get_document_embedding(vec_store, document_id):
    """Retrieve the embedding vector for a specific document.
    
    Args:
        vec_store (VectorStore): The vector store instance
        document_id (str): The document ID to get embedding for
        
    Returns:
        tuple: A tuple containing:
            - numpy.array or None: The document embedding vector, or None if not found
            - float: Time taken in milliseconds
    """
    start_time = time.time()
    
    embedding = vec_store.get_document_embedding(document_id)
    
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    return embedding, elapsed_ms


def perform_document_similarity_search(vec_store, source_embedding, exclude_document_id, project_ids, limit):
    """Perform cosine similarity search against document embeddings.
    
    Args:
        vec_store (VectorStore): The vector store instance
        source_embedding (numpy.array): The embedding vector to search for similar documents
        exclude_document_id (str): Document ID to exclude from results (the source document)
        project_ids (list, optional): List of project IDs to filter results
        limit (int): Maximum number of similar documents to return
        
    Returns:
        tuple: A tuple containing:
            - DataFrame: Similar documents with similarity scores
            - float: Time taken in milliseconds
    """
    start_time = time.time()
    
    # Build predicates for project filtering
    predicates = {}
    if project_ids:
        predicates['project_ids'] = project_ids
    
    # Perform similarity search
    similar_documents = vec_store.document_similarity_search(
        source_embedding, exclude_document_id, predicates, limit, return_dataframe=True
    )
    
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    return similar_documents, elapsed_ms


def format_similar_documents(similar_docs_df):
    """Format similar documents results for API response.
    
    Args:
        similar_docs_df (DataFrame): DataFrame containing similar documents
        
    Returns:
        list: List of formatted document dictionaries
    """
    if similar_docs_df.empty:
        return []
    
    formatted_docs = []
    for _, row in similar_docs_df.iterrows():
        # Get document metadata
        document_metadata = row.get('document_metadata', {})
        if isinstance(document_metadata, str):
            import json
            try:
                document_metadata = json.loads(document_metadata)
            except:
                document_metadata = {}
        elif document_metadata is None:
            document_metadata = {}
        
        # Extract fields from document metadata with correct mapping based on user example
        document_type = document_metadata.get("document_type") or get_document_type_name(document_metadata, None)
        
        # Based on user example: document_name should be human-readable filename
        document_name = document_metadata.get("document_name")  # Human-readable filename
        
        # Extract display_name field
        document_display_name = get_document_display_name(document_metadata, None)
        
        # Based on user example: document_saved_name should be hash/technical filename
        # Try the direct field first, then fallbacks
        document_saved_name = (document_metadata.get("document_saved_name") or 
                              document_metadata.get("doc_internal_name") or 
                              document_metadata.get("document_internal_name") or
                              document_metadata.get("internal_name") or
                              document_metadata.get("file_name") or
                              document_metadata.get("filename"))
                              
        project_name = document_metadata.get("project_name")
        proponent_name = document_metadata.get("proponent_name")
        s3_key = document_metadata.get("s3_key")
        
        doc = {
            "document_id": row["document_id"],
            "document_type": document_type,
            "document_name": document_name,
            "document_saved_name": document_saved_name,
            "document_display_name": document_display_name,
            "project_id": row.get("project_id"),
            "project_name": project_name,
            "proponent_name": proponent_name,
            "s3_key": s3_key,
            "similarity_score": round(float(row["similarity"]), 4)
        }
        formatted_docs.append(doc)
    
    return formatted_docs


def is_generic_document_request(query: str) -> bool:
    """Check if the query is a generic document request (not seeking specific content).
    
    This function is used by the inference pipeline to determine appropriate query cleaning
    strategies. Generic queries are those that ask for documents of a certain type without
    seeking specific information within those documents.
    
    Generic queries include:
    1. Explicit generic patterns like "show me all letters"
    2. Very short queries after inference cleaning (e.g., "Packages", "Reports")
    3. Simple document type words without context
    
    Args:
        query (str): The search query text
        
    Returns:
        bool: True if the query is generic, False otherwise
    """
    if not query or len(query.strip()) < 2:
        return True
    
    query_lower = query.lower().strip()
    
    # Very short queries (1-2 words) are likely generic after inference cleaning
    words = query_lower.split()
    if len(words) <= 2:
        # Check if it's a simple document type word or generic term
        simple_generic_terms = [
            'packages', 'package', 'documents', 'document', 'files', 'file',
            'letters', 'letter', 'correspondence', 'reports', 'report',
            'studies', 'study', 'assessments', 'assessment', 'certificates',
            'certificate', 'permits', 'permit', 'licenses', 'license',
            'orders', 'order', 'agreements', 'agreement', 'contracts', 'contract',
            'submissions', 'submission', 'comments', 'comment', 'responses', 'response'
        ]
        
        # If the entire query is just document type words, it's generic
        if all(word in simple_generic_terms for word in words):
            return True
    
    # First check for content-specific terms that indicate semantic search is needed
    content_specific_terms = [
        r'\b(complaints?|concerns?|issues?|problems?|violations?|incidents?)\b',
        r'\b(about|regarding|related to|concerning)\b',
        r'\b(contain|containing|with|have|having|include|including)\b',
        r'\b(mention|mentioning|discuss|discussing|address|addressing)\b',
        r'\b(refer|refers|referring|reference|references)\s+(to|the|a|an)\b',
        r'\b(refer|refers|referring|reference|references)\s+to\b',  # More flexible refer pattern
        r'\bthat\s+(refer|refers|referring|reference|references)\b',  # "that refers"
        r'\b(talk|talking|speak|speaking)\s+(about|of|on)\b',
        r'\b(environmental|safety|health|regulatory|compliance)\b',
        r'\b(impact|effect|consequence|result|outcome)\b',
        r'\bthat\s+(talk|talks|speak|speaks|discuss|discusses|mention|mentions|refer|refers|address|addresses|cover|covers)\b',
        r'\b(nation|first nation|indigenous|aboriginal|métis|inuit)\b',
        r'\b(band|tribe|tribal|nation)\b',  # Additional indigenous/band references
        r'\bnamed?\s+["\']?[A-Z][^"\']*["\']?\b',  # Named entities (proper nouns in quotes)
        r'\bcalled\s+["\']?[A-Z][^"\']*["\']?\b'  # "called X" patterns
    ]
    
    # If query contains content-specific terms, it's NOT generic
    for pattern in content_specific_terms:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return False
    
    # Check for truly generic patterns (asking for documents without specific content)
    generic_patterns = [
        # Patterns with explicit project references (for project inference)
        r'\b(any|all)\s+(correspondence|letters?|documents?|files?)\s+(for|from)\b',
        r'\bshow\s+me\s+(all|any)?\s*(correspondence|letters?|documents?|files?)\s+(for|from)\b',
        r'\bfind\s+(all|any)?\s*(correspondence|letters?|documents?|files?)\s+(for|from)\b',
        r'\bget\s+(all|any)?\s*(correspondence|letters?|documents?|files?)\s+(for|from)\b',
        r'\blist\s+(all|any)?\s*(correspondence|letters?|documents?|files?)\s+(for|from)\b',
        r'\blooking\s+for\s+(any|all)\s+(correspondence|letters?|documents?|files?)\s+(for|from)\b',
        r'\bneed\s+(any|all)\s+(correspondence|letters?|documents?|files?)\s+(for|from)\b',
        r'\bprovide\s+(any|all)\s+(correspondence|letters?|documents?|files?)\s+(for|from)\b',
        
        # Patterns for when project/type filtering is provided via API (no "for/from" needed)
        r'\b(any|all)\s+(?:the\s+)?(correspondence|letters?|documents?|files?|reports?|studies|assessments?|analyses|plans?|orders?|agreements?|contracts?|certificates?|permits?|licenses?)\b',
        r'\bshow\s+me\s+(all|any)?\s*(?:the\s+)?(correspondence|letters?|documents?|files?|reports?|studies|assessments?|analyses|plans?|orders?|agreements?|contracts?|certificates?|permits?|licenses?)\b',
        r'\bfind\s+(all|any)?\s*(?:the\s+)?(correspondence|letters?|documents?|files?|reports?|studies|assessments?|analyses|plans?|orders?|agreements?|contracts?|certificates?|permits?|licenses?)\b',
        r'\bget\s+(all|any)?\s*(?:the\s+)?(correspondence|letters?|documents?|files?|reports?|studies|assessments?|analyses|plans?|orders?|agreements?|contracts?|certificates?|permits?|licenses?)\b',
        r'\blist\s+(all|any)?\s*(?:the\s+)?(correspondence|letters?|documents?|files?|reports?|studies|assessments?|analyses|plans?|orders?|agreements?|contracts?|certificates?|permits?|licenses?)\b',
        r'\blooking\s+for\s+(any|all)\s+(?:the\s+)?(correspondence|letters?|documents?|files?|reports?|studies|assessments?|analyses|plans?|orders?|agreements?|contracts?|certificates?|permits?|licenses?)\b',
        r'\bneed\s+(any|all)\s+(?:the\s+)?(correspondence|letters?|documents?|files?|reports?|studies|assessments?|analyses|plans?|orders?|agreements?|contracts?|certificates?|permits?|licenses?)\b',
        r'\bprovide\s+(any|all)\s+(?:the\s+)?(correspondence|letters?|documents?|files?|reports?|studies|assessments?|analyses|plans?|orders?|agreements?|contracts?|certificates?|permits?|licenses?)\b',
        
        # Simpler patterns for direct document type requests
        r'^\s*(?:show me |give me |find |get |list )?(?:any |all )?(?:the\s+)?(correspondence|letters?|documents?|files?|reports?|studies|assessments?|analyses|plans?|orders?|agreements?|contracts?|certificates?|permits?|licenses?)\s*\??\s*$',
        
        # More specific patterns for pure document requests
        r'^\s*(?:show me |give me |find |get |list )?(?:any |all )?(?:the )?(?:correspondence|letters?|documents?|files?)\s+(?:for|from|related to)\s+[^?]*\??\s*$'
    ]
    
    for pattern in generic_patterns:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return True
    
    return False


def perform_direct_metadata_search(vec_store, project_ids, document_type_ids, limit):
    """Perform direct metadata-based document search without semantic analysis.
    
    This function is used when both project and document type are confidently inferred
    and the query is generic (e.g., "any correspondence for Project X"). Instead of
    semantic search, it returns all documents matching the metadata criteria,
    ordered by document date.
    
    Args:
        vec_store (VectorStore): The vector store instance
        project_ids (list): List of project IDs to filter results
        document_type_ids (list): List of document type IDs to filter results
        limit (int): Maximum number of documents to return
        
    Returns:
        tuple: A tuple containing:
            - DataFrame: Document results ordered by date
            - float: Time taken in milliseconds
    """
    start_time = time.time()
    
    # Get documents directly by metadata filtering
    documents = vec_store.get_documents_by_metadata(
        project_ids=project_ids,
        document_type_ids=document_type_ids,
        order_by="document_date DESC",  # Order by date, newest first
        limit=limit,
        return_dataframe=True
    )
    
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    return documents, elapsed_ms


def format_document_data(documents_df):
    """Format document-level search results into a list of dictionaries.
    
    Args:
        documents_df (DataFrame): Document-level search results data
        
    Returns:
        list: List of dictionaries containing formatted document information
    """
    result = []
    
    # Check if data is empty
    if len(documents_df) == 0:
        return result
    
    # Process DataFrame directly
    if hasattr(documents_df, 'iterrows'):
        for idx, row in documents_df.iterrows():
            # Extract document information directly from the DataFrame columns
            # The get_documents_by_metadata function already extracts JSON fields into columns
            document_id = str(row.get("document_id", ""))
            project_id = str(row.get("project_id", ""))
            
            # These fields are extracted from document_metadata JSON in the SQL query
            document_name = row.get("document_name")
            document_saved_name = row.get("document_saved_name") 
            project_name = row.get("project_name")
            proponent_name = row.get("proponent_name")
            s3_key = row.get("s3_key")
            document_date = row.get("document_date")
            
            # Get document type - try from metadata JSON first, then fallback to helper function
            document_metadata = row.get('document_metadata', {})
            if isinstance(document_metadata, str):
                import json
                try:
                    document_metadata = json.loads(document_metadata)
                except:
                    document_metadata = {}
            elif document_metadata is None:
                document_metadata = {}
            
            document_type = document_metadata.get("document_type") or get_document_type_name(document_metadata, None)
            
            # Extract display_name field
            document_display_name = get_document_display_name(document_metadata, None)
            
            # For document-level results, content is typically a summary or first chunk
            content = row.get('content', '') or row.get('document_summary', '') or "Full document available"

            # Append to result list
            formatted_result = {
                "document_id": document_id,
                "document_type": document_type,
                "document_name": document_name,
                "document_saved_name": document_saved_name,
                "document_display_name": document_display_name,
                "document_date": document_date,
                "page_number": None,  # Not applicable for document-level results
                "project_id": project_id,
                "project_name": project_name,
                "proponent_name": proponent_name,
                "s3_key": s3_key,
                "content": content,
                "relevance_score": 1.0,  # Perfect relevance for metadata matches
                "search_mode": "document_metadata"  # Indicate this was a metadata search
            }
            
            result.append(formatted_result)
    
    return result

def apply_post_search_filtering(results_df, project_ids=None, document_type_ids=None):
    """Apply project and document type filtering to search results after the main search.
    
    This function filters search results to only include results matching the specified
    project IDs and document type IDs by examining the metadata column in the results DataFrame.
    
    Args:
        results_df (DataFrame): Search results with metadata column
        project_ids (list, optional): List of project IDs to filter by
        document_type_ids (list, optional): List of document type IDs to filter by
        
    Returns:
        DataFrame: Filtered search results
    """
    
    logging.info(f"apply_post_search_filtering called with project_ids={project_ids}, document_type_ids={document_type_ids}")
    
    if results_df.empty:
        return results_df
    
    filtered_df = results_df.copy()
    
    # Filter by project IDs if provided
    if project_ids:
        def matches_project(metadata):
            if not isinstance(metadata, dict):
                return False
            
            # Check if project_id is directly in metadata
            metadata_project_id = metadata.get('project_id')
            if metadata_project_id and metadata_project_id in project_ids:
                return True
            
            # Check if project_id is in document_metadata
            document_metadata = metadata.get('document_metadata', {})
            if isinstance(document_metadata, dict):
                doc_project_id = document_metadata.get('project_id')
                if doc_project_id and doc_project_id in project_ids:
                    return True
            
            return False
        
        # Apply project filtering
        project_mask = filtered_df['metadata'].apply(matches_project)
        filtered_df = filtered_df[project_mask]
        logging.info(f"Post-search project filtering: {len(results_df)} -> {len(filtered_df)} results")
    
    # Filter by document type IDs if provided
    if document_type_ids:
        logging.info(f"Applying document type filtering with IDs: {document_type_ids}")
        def matches_document_type(metadata):
            if not isinstance(metadata, dict):
                return False
            
            # Check if document_type_id is directly in metadata
            metadata_doc_type_id = metadata.get('document_type_id')
            if metadata_doc_type_id and metadata_doc_type_id in document_type_ids:
                logging.debug(f"Document type match found: {metadata_doc_type_id} in {document_type_ids}")
                return True
            
            # Check if document_type_id is in document_metadata
            document_metadata = metadata.get('document_metadata', {})
            if isinstance(document_metadata, dict):
                doc_type_id = document_metadata.get('document_type_id')
                if doc_type_id and doc_type_id in document_type_ids:
                    logging.debug(f"Document type match found in document_metadata: {doc_type_id} in {document_type_ids}")
                    return True
            
            return False
        
        # Apply document type filtering
        doc_type_mask = filtered_df['metadata'].apply(matches_document_type)
        filtered_df = filtered_df[doc_type_mask]
        logging.info(f"Post-search document type filtering: {len(results_df) if not project_ids else len(filtered_df)} -> {len(filtered_df)} results")
    
    return filtered_df
