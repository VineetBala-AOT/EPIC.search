"""PostgreSQL vector database interface for semantic search operations.

This module provides a direct interface to PostgreSQL with pgvector extension,
enabling both semantic vector similarity search and keyword-based full-text search.
It handles the low-level database operations including:

1. Vector similarity matching using pgvector's cosine similarity operators (<=>)
2. Keyword-based search using PostgreSQL's full-text search capabilities
3. Result filtering by tags, metadata, and time ranges
4. Performance tracking of search operations

The VectorStore class is designed to be agnostic of the specific embedding models
or document structures, focusing purely on efficient database interaction.
"""

import ast
import logging
import time
import pandas as pd
import psycopg

from typing import Any, List, Optional, Tuple, Union
from datetime import datetime
from flask import current_app
from .embedding import get_embedding
from .tags.tag_extractor import get_tags

class VectorStore:
    """
    A service for vector-based and keyword-based document search using pgvector.
    
    This class provides methods for semantic vector search and keyword-based search
    on documents stored in a PostgreSQL database with the pgvector extension.
    """

    def __init__(self):
        """Initialize a VectorStore instance."""
        pass

    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert database query results into a pandas DataFrame.
        
        Args:
            results: A list of tuples returned from a database query.
            columns: Optional list of column names. If not provided, uses default.
            
        Returns:
            A pandas DataFrame with the search results.
        """
        if columns is None:
            # Default columns for backward compatibility
            columns = ["id", "metadata", "content", "embedding", "distance"]
            
        df = pd.DataFrame(results, columns=columns)
        df["id"] = df["id"].astype(str)
        return df

    def semantic_search(
        self,
        table_name: str,
        query: str,
        limit: int = 5,
        predicates: Optional[dict] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Search for documents similar to the query vector using pgvector directly.
        
        This method performs a semantic search using vector similarity with pgvector,
        converting the query text into embeddings and finding the most similar documents.
        
        Args:
            table_name: The table to search in.
            query: The search query text.
            limit: Maximum number of results to return (default: 5).
            predicates: Optional dictionary of field-value pairs to filter results.
            time_range: Optional tuple of (start_date, end_date) to filter by creation time.
            return_dataframe: If True, returns results as a DataFrame; otherwise as a list of tuples.
            
        Returns:
            Either a pandas DataFrame or a list of tuples containing search results.
        """
        query_embedding = get_embedding([query])
        start_time = time.time()

        # Build the WHERE clause based on filters
        where_conditions = ["TRUE"]
        params = []
        
        # Handle tags filter - disable automatic tag filtering to avoid overly restrictive results
        # Tags filtering should be explicit, not automatic based on query text
        tags = get_tags(query)
        if tags:
            # Log the tags that would be used, but don't apply the filter automatically
            logging.info(f"Semantic search - Detected tags (not filtering): {tags}")
        else:
            logging.info(f"Semantic search - No tags found for query: '{query}'")
        
        # Handle time range filter
        if time_range:
            start_date, end_date = time_range
            where_conditions.append("(metadata->>'created_at')::timestamp BETWEEN %s AND %s")
            params.append(start_date)
            params.append(end_date)
        
        # Handle predicates for additional filtering
        if predicates:
            logging.info(f"VectorStore.semantic_search - Processing predicates: {predicates}")
            for key, value in predicates.items():
                logging.info(f"VectorStore.semantic_search - Processing predicate: {key} = {value} (type: {type(value)})")
                if key == 'project_ids':
                    # Handle multiple project IDs for chunk search
                    if value and len(value) > 0:
                        placeholders = ','.join(['%s'] * len(value))
                        # For document_chunks table, project_id is a direct column
                        if table_name == current_app.vector_settings.vector_table_name:
                            where_conditions.append(f"project_id IN ({placeholders})")
                            params.extend(value)
                            logging.info(f"VectorStore.semantic_search - Added project_ids IN condition for chunks table")
                            logging.info(f"VectorStore.semantic_search - Extended params with: {value}")
                        else:
                            # For other tables, check metadata
                            where_conditions.append(f"metadata->>'project_id' IN ({placeholders})")
                            params.extend(value)
                            logging.info(f"VectorStore.semantic_search - Added project_ids IN condition for metadata")
                    else:
                        logging.info(f"VectorStore.semantic_search - Skipped empty project_ids: {value}")
                elif key == 'document_type_ids':
                    # Handle multiple document type IDs
                    if value and len(value) > 0:
                        placeholders = ','.join(['%s'] * len(value))
                        # Document type filtering depends on the table
                        if table_name == "documents":
                            # For documents table, use document_metadata column directly
                            where_conditions.append(f"document_metadata->>'document_type_id' IN ({placeholders})")
                            params.extend(value)
                        else:
                            # For document_chunks table, join with documents table to filter by document type
                            where_conditions.append(f"""
                                document_id IN (
                                    SELECT document_id FROM documents 
                                    WHERE document_metadata->>'document_type_id' IN ({placeholders})
                                )
                            """)
                            params.extend(value)
                elif key == 'project_id':
                    # Handle single project ID
                    if table_name == current_app.vector_settings.vector_table_name:
                        where_conditions.append("project_id = %s")
                        params.append(value)
                    else:
                        where_conditions.append("metadata->>'project_id' = %s")
                        params.append(value)
                else:
                    where_conditions.append(f"metadata->>{key} = %s")
                    params.append(value)
        
        # Build the final WHERE clause
        where_clause = " AND ".join(where_conditions)
        
        logging.info(f"Document search WHERE clause: {where_clause}")
        logging.info(f"Document search parameters: {params}")
        
        # Convert numpy array to Python list for database
        embedding_list = query_embedding[0].tolist()
        
        # Prepare parameters in the correct order for the SQL query
        # Order: embedding (for similarity), WHERE clause params, embedding (for ordering), limit
        sql_params = [embedding_list] + params + [embedding_list, limit]
        
        # Construct the SQL query using cosine distance with pgvector
        # Note: document_metadata only exists on documents table, not on document_chunks
        if table_name == "documents":
            search_sql = f"""
            SELECT id, metadata, content, embedding, document_metadata, 1 - (embedding <=> %s::vector) as similarity
            FROM {table_name}
            WHERE {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
        else:
            # For document_chunks table, don't select document_metadata
            search_sql = f"""
            SELECT id, metadata, content, embedding, 1 - (embedding <=> %s::vector) as similarity
            FROM {table_name}
            WHERE {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
        
        # Execute the query using psycopg
        conn_params = current_app.vector_settings.database_url
        with psycopg.connect(conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(search_sql, sql_params)
                results = cur.fetchall()
        
        elapsed_time = time.time() - start_time
        self._log_search_time("Vector", elapsed_time)
        
        if return_dataframe:
            # Specify the correct column order for the semantic search results
            # Note: document_metadata only included when querying documents table
            if table_name == "documents":
                columns = ["id", "metadata", "content", "embedding", "document_metadata", "similarity"]
            else:
                columns = ["id", "metadata", "content", "embedding", "similarity"]
            return self._create_dataframe_from_results(results, columns)
        else:
            return results

    def keyword_search(
        self, table_name: str, query: str, limit: int = 5, return_dataframe: bool = True, weighted_keywords=None
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Search for documents using only pre-computed metadata columns (document_keywords, document_tags, document_headings).
        This avoids scanning the raw content field and leverages GIN indexes for fast keyword search.
        
        Search logic:
        - Keywords are searched in: document_keywords, document_headings
        - Tags are searched in: document_tags, document_headings
        - Both keywords and tags can match headings for maximum coverage
        """
        if weighted_keywords is None:
            raise ValueError("weighted_keywords must be provided by the caller.")
        tags = get_tags(query)
        keywords = [keyword for keyword in weighted_keywords]
        start_time = time.time()

        # Always search documents first for matching keywords/tags/headings
        doc_where_conditions = ["TRUE"]
        doc_params = []
        doc_search_conditions = []
        if keywords:
            doc_search_conditions.append("document_keywords ?| %s")
            doc_params.append(keywords)
        if tags:
            doc_search_conditions.append("document_tags ?| %s")
            doc_params.append(tags)
        if keywords:
            doc_search_conditions.append("document_headings ?| %s")
            doc_params.append(keywords)
        if tags:
            doc_search_conditions.append("document_headings ?| %s")
            doc_params.append(tags)
        if doc_search_conditions:
            doc_where_conditions.append("(" + " OR ".join(doc_search_conditions) + ")")
        doc_where_clause = " AND ".join(doc_where_conditions)
        logging.info(f"Keyword search (documents) WHERE clause: {doc_where_clause}")
        logging.info(f"Keyword search (documents) parameters: {doc_params}")
        documents_table = current_app.vector_settings.documents_table_name
        doc_search_sql = f"""
        SELECT document_id
        FROM {documents_table}
        WHERE {doc_where_clause}
        ORDER BY document_id DESC
        LIMIT %s
        """
        doc_params.append(limit)
        conn_params = current_app.vector_settings.database_url
        with psycopg.connect(conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(doc_search_sql, doc_params)
                doc_id_results = cur.fetchall()
        document_ids = [row[0] for row in doc_id_results]

        # If searching documents, return those results
        if table_name == "documents":
            # Optionally, fetch full document info for those IDs
            if not document_ids:
                results = []
            else:
                placeholders = ','.join(['%s'] * len(document_ids))
                fetch_sql = f"""
                SELECT document_id, content, metadata, document_metadata
                FROM {documents_table}
                WHERE document_id IN ({placeholders})
                ORDER BY document_id DESC
                """
                with psycopg.connect(conn_params) as conn:
                    with conn.cursor() as cur:
                        cur.execute(fetch_sql, document_ids)
                        results = cur.fetchall()
            elapsed_time = time.time() - start_time
            self._log_search_time("Keyword", elapsed_time)
            if return_dataframe:
                columns = ["id", "content", "metadata", "document_metadata"]
                df = pd.DataFrame(results, columns=columns)
                df["id"] = df["id"].astype(str)
                return df
            else:
                return results

        # If searching document_chunks, filter by document_ids from above
        else:
            if not document_ids:
                results = []
            else:
                chunks_table = current_app.vector_settings.vector_table_name
                placeholders = ','.join(['%s'] * len(document_ids))
                chunk_where_conditions = [f"document_id IN ({placeholders})"]
                chunk_params = document_ids.copy()
                chunk_search_conditions = []
                # Filter by chunk metadata: keywords, tags, headings
                if keywords:
                    chunk_search_conditions.append("(metadata->'keywords') ?| %s")
                    chunk_params.append(keywords)
                if tags:
                    chunk_search_conditions.append("(metadata->'tags') ?| %s")
                    chunk_params.append(tags)
                if keywords:
                    chunk_search_conditions.append("(metadata->'headings') ?| %s")
                    chunk_params.append(keywords)
                if tags:
                    chunk_search_conditions.append("(metadata->'headings') ?| %s")
                    chunk_params.append(tags)
                if chunk_search_conditions:
                    chunk_where_conditions.append("(" + " OR ".join(chunk_search_conditions) + ")")
                chunk_where_clause = " AND ".join(chunk_where_conditions)
                chunk_sql = f"""
                SELECT id, content, metadata
                FROM {chunks_table}
                WHERE {chunk_where_clause}
                ORDER BY id DESC
                LIMIT %s
                """
                chunk_params.append(limit)
                with psycopg.connect(conn_params) as conn:
                    with conn.cursor() as cur:
                        cur.execute(chunk_sql, chunk_params)
                        results = cur.fetchall()
            elapsed_time = time.time() - start_time
            self._log_search_time("Keyword", elapsed_time)
            if return_dataframe:
                columns = ["id", "content", "metadata"]
                df = pd.DataFrame(results, columns=columns)
                df["id"] = df["id"].astype(str)
                return df
            else:
                return results

    def keyword_search_with_predicates(
        self, 
        table_name: str, 
        query: str, 
        limit: int = 5, 
        predicates: Optional[dict] = None,
        return_dataframe: bool = True, 
        weighted_keywords=None
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Enhanced keyword search with support for project and document type filtering.
        
        This method combines the keyword search functionality with predicate-based filtering
        similar to semantic_search, providing consistent filtering behavior across all search methods.
        
        Args:
            table_name: The table to search in.
            query: The search query text.
            limit: Maximum number of results to return (default: 5).
            predicates: Optional dictionary of field-value pairs to filter results.
            return_dataframe: If True, returns results as a DataFrame; otherwise as a list of tuples.
            weighted_keywords: Pre-extracted keywords to use for search.
            
        Returns:
            Either a pandas DataFrame or a list of tuples containing search results.
        """
        if weighted_keywords is None:
            raise ValueError("weighted_keywords must be provided by the caller.")
        
        tags = get_tags(query)
        keywords = [keyword for keyword in weighted_keywords]
        start_time = time.time()

        # Debug logging
        logging.info(f"VectorStore.keyword_search_with_predicates - Processing predicates: {predicates}")

        # Step 1: Find documents that match keywords/tags/headings with optional project/document type filtering
        doc_where_conditions = ["TRUE"]
        doc_params = []
        doc_search_conditions = []
        
        # Add keyword and tag search conditions
        if keywords:
            doc_search_conditions.append("document_keywords ?| %s")
            doc_params.append(keywords)
        if tags:
            doc_search_conditions.append("document_tags ?| %s")
            doc_params.append(tags)
        if keywords:
            doc_search_conditions.append("document_headings ?| %s")
            doc_params.append(keywords)
        if tags:
            doc_search_conditions.append("document_headings ?| %s")
            doc_params.append(tags)
        
        if doc_search_conditions:
            doc_where_conditions.append("(" + " OR ".join(doc_search_conditions) + ")")

        # Add predicate-based filtering (project_ids, document_type_ids)
        if predicates:
            for key, value in predicates.items():
                logging.info(f"VectorStore.keyword_search_with_predicates - Processing predicate: {key} = {value} (type: {type(value)})")
                if key == 'project_ids':
                    # Handle multiple project IDs
                    if value and len(value) > 0:
                        placeholders = ','.join(['%s'] * len(value))
                        doc_where_conditions.append(f"project_id IN ({placeholders})")
                        doc_params.extend(value)
                        logging.info(f"VectorStore.keyword_search_with_predicates - Added project_ids IN condition with {len(value)} IDs")
                    else:
                        logging.info(f"VectorStore.keyword_search_with_predicates - Skipped empty project_ids: {value}")
                elif key == 'document_type_ids':
                    # Handle multiple document type IDs
                    if value and len(value) > 0:
                        placeholders = ','.join(['%s'] * len(value))
                        doc_where_conditions.append(f"document_metadata->>'document_type_id' IN ({placeholders})")
                        doc_params.extend(value)
                        logging.info(f"VectorStore.keyword_search_with_predicates - Added document_type_ids IN condition with {len(value)} IDs")

        doc_where_clause = " AND ".join(doc_where_conditions)
        logging.info(f"VectorStore.keyword_search_with_predicates - Document search WHERE clause: {doc_where_clause}")
        logging.info(f"VectorStore.keyword_search_with_predicates - Document search parameters: {doc_params}")
        
        documents_table = current_app.vector_settings.documents_table_name
        doc_search_sql = f"""
        SELECT document_id
        FROM {documents_table}
        WHERE {doc_where_clause}
        ORDER BY document_id DESC
        LIMIT %s
        """
        doc_params.append(limit)
        
        conn_params = current_app.vector_settings.database_url
        with psycopg.connect(conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(doc_search_sql, doc_params)
                doc_id_results = cur.fetchall()
                
        document_ids = [row[0] for row in doc_id_results]
        logging.info(f"VectorStore.keyword_search_with_predicates - Found {len(document_ids)} matching documents")

        # Step 2: Return results based on target table
        if table_name == "documents":
            # Return document-level results
            if not document_ids:
                results = []
            else:
                placeholders = ','.join(['%s'] * len(document_ids))
                fetch_sql = f"""
                SELECT document_id, content, metadata, document_metadata
                FROM {documents_table}
                WHERE document_id IN ({placeholders})
                ORDER BY document_id DESC
                """
                with psycopg.connect(conn_params) as conn:
                    with conn.cursor() as cur:
                        cur.execute(fetch_sql, document_ids)
                        results = cur.fetchall()
            
            elapsed_time = time.time() - start_time
            self._log_search_time("Keyword (with predicates)", elapsed_time)
            logging.info(f"VectorStore.keyword_search_with_predicates - Returning {len(results)} document results")
            
            if return_dataframe:
                columns = ["id", "content", "metadata", "document_metadata"]
                df = pd.DataFrame(results, columns=columns)
                if not df.empty:
                    df["id"] = df["id"].astype(str)
                return df
            else:
                return results
        else:
            # Return chunk-level results from document_chunks table
            if not document_ids:
                results = []
            else:
                chunks_table = current_app.vector_settings.vector_table_name
                placeholders = ','.join(['%s'] * len(document_ids))
                chunk_where_conditions = [f"document_id IN ({placeholders})"]
                chunk_params = document_ids.copy()
                
                # Add chunk-level keyword/tag filtering
                chunk_search_conditions = []
                if keywords:
                    chunk_search_conditions.append("(metadata->'keywords') ?| %s")
                    chunk_params.append(keywords)
                if tags:
                    chunk_search_conditions.append("(metadata->'tags') ?| %s")
                    chunk_params.append(tags)
                if keywords:
                    chunk_search_conditions.append("(metadata->'headings') ?| %s")
                    chunk_params.append(keywords)
                if tags:
                    chunk_search_conditions.append("(metadata->'headings') ?| %s")
                    chunk_params.append(tags)
                
                if chunk_search_conditions:
                    chunk_where_conditions.append("(" + " OR ".join(chunk_search_conditions) + ")")
                
                chunk_where_clause = " AND ".join(chunk_where_conditions)
                chunk_sql = f"""
                SELECT id, content, metadata
                FROM {chunks_table}
                WHERE {chunk_where_clause}
                ORDER BY id DESC
                LIMIT %s
                """
                chunk_params.append(limit)
                
                logging.info(f"VectorStore.keyword_search_with_predicates - Chunk search WHERE clause: {chunk_where_clause}")
                logging.info(f"VectorStore.keyword_search_with_predicates - Chunk search parameters: {chunk_params}")
                
                with psycopg.connect(conn_params) as conn:
                    with conn.cursor() as cur:
                        cur.execute(chunk_sql, chunk_params)
                        results = cur.fetchall()
            
            elapsed_time = time.time() - start_time
            self._log_search_time("Keyword (with predicates)", elapsed_time)
            logging.info(f"VectorStore.keyword_search_with_predicates - Returning {len(results)} chunk results")
            
            if return_dataframe:
                columns = ["id", "content", "metadata"]
                df = pd.DataFrame(results, columns=columns)
                if not df.empty:
                    df["id"] = df["id"].astype(str)
                return df
            else:
                return results

    def document_level_search(
        self,
        query: str,
        limit: int = 10,
        predicates: Optional[dict] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Search for relevant documents using document-level keywords, tags, and headings.
        
        This method searches the documents table using the pre-computed keywords, tags,
        and headings stored at the document level, which should be much faster than
        searching through all chunks.
        
        Args:
            query: The search query text.
            limit: Maximum number of documents to return (default: 10).
            predicates: Optional dictionary of field-value pairs to filter results.
            return_dataframe: If True, returns results as a DataFrame; otherwise as a list of tuples.
            
        Returns:
            Either a pandas DataFrame or a list of tuples containing document search results.
        """
        from .keywords.query_keyword_extractor import get_keywords
        from .tags.tag_extractor import get_tags
        
        start_time = time.time()
        
        # Extract keywords and tags from query
        query_keywords = get_keywords(query)
        query_tags = get_tags(query)
        
        # Check extraction method to adjust search strategy
        extraction_method = current_app.model_settings.document_keyword_extraction_method
        
        # Debug logging
        import logging
        logging.info(f"Document-level search for query: '{query}' (extraction method: {extraction_method})")
        logging.info(f"Extracted keywords: {query_keywords}")
        logging.info(f"Extracted tags: {query_tags}")
        
        # Build the WHERE clause based on filters using OR logic for search terms
        where_conditions = ["TRUE"]
        params = []
        
        # Build search conditions using OR logic between different search criteria
        search_conditions = []
        
        # Adjust search strategy based on keyword extraction method
        if extraction_method == "fast":
            # For fast (TF-IDF-based) keywords, prioritize tags and headings over document keywords
            # since TF-IDF keywords are statistical/frequency-based rather than semantic
            
            # Search for tag matches first (highest priority for TF-IDF)
            if query_tags:
                search_conditions.append("document_tags ?| %s")
                params.append(query_tags)
            
            # Search for keyword matches in document_headings (medium priority)
            if query_keywords:
                keyword_list = [keyword for keyword, score in query_keywords]
                search_conditions.append("document_headings ?| %s")
                params.append(keyword_list)
            
            # Search for keyword matches in document_keywords (lower priority for TF-IDF)
            if query_keywords:
                keyword_list = [keyword for keyword, score in query_keywords]
                search_conditions.append("document_keywords ?| %s")
                params.append(keyword_list)
                
            logging.info("Fast mode: Prioritizing tags and headings over document keywords")
            
        else:
            # For semantic keywords, use the original strategy (keywords have high semantic relevance)
            
            # Search for keyword matches in document_keywords JSONB field (high priority)
            if query_keywords:
                keyword_list = [keyword for keyword, score in query_keywords]
                # Use JSONB operators to check if any query keywords exist in document keywords
                search_conditions.append("document_keywords ?| %s")
                params.append(keyword_list)
            
            # Search for tag matches in document_tags JSONB field
            if query_tags:
                search_conditions.append("document_tags ?| %s")
                params.append(query_tags)
            
            # Search for keyword matches in document_headings JSONB field (using same keyword list)
            if query_keywords:
                keyword_list = [keyword for keyword, score in query_keywords]
                search_conditions.append("document_headings ?| %s")
                params.append(keyword_list)  # Add the same keyword list again for headings
                
            logging.info("Semantic mode: Standard keyword-first search strategy")
        
        # Combine search conditions with OR logic
        if search_conditions:
            where_conditions.append("(" + " OR ".join(search_conditions) + ")")
        else:
            # If no search terms found, don't filter by search criteria
            # This allows the fallback search to handle it
            pass
        
        # Handle predicates for additional filtering
        if predicates:
            logging.info(f"VectorStore.document_level_search - Processing predicates: {predicates}")
            for key, value in predicates.items():
                logging.info(f"VectorStore.document_level_search - Processing predicate: {key} = {value} (type: {type(value)})")
                if key == 'project_id':
                    where_conditions.append("project_id = %s")
                    params.append(value)
                    logging.info(f"VectorStore.document_level_search - Added single project_id condition, params now: {params}")
                elif key == 'project_ids':
                    # Handle multiple project IDs
                    if value and len(value) > 0:
                        placeholders = ','.join(['%s'] * len(value))
                        where_conditions.append(f"project_id IN ({placeholders})")
                        params.extend(value)
                        logging.info(f"VectorStore.document_level_search - Added project_ids IN condition with {len(value)} IDs")
                        logging.info(f"VectorStore.document_level_search - Placeholders: {placeholders}")
                        logging.info(f"VectorStore.document_level_search - Extended params with: {value}")
                        logging.info(f"VectorStore.document_level_search - Params now: {params}")
                    else:
                        logging.info(f"VectorStore.document_level_search - Skipped empty project_ids: {value}")
                elif key == 'document_type_ids':
                    # Handle multiple document type IDs
                    if value and len(value) > 0:
                        placeholders = ','.join(['%s'] * len(value))
                        where_conditions.append(f"document_metadata->>'document_type_id' IN ({placeholders})")
                        params.extend(value)
                        logging.info(f"VectorStore.document_level_search - Added document_type_ids IN condition with {len(value)} IDs")
                # Add other predicate handling as needed
        else:
            logging.info("VectorStore.document_level_search - No predicates to process")
        
        # Build the final WHERE clause
        where_clause = " AND ".join(where_conditions)
        
        logging.info(f"VectorStore.document_level_search - Final WHERE clause: {where_clause}")
        logging.info(f"VectorStore.document_level_search - Final parameters: {params}")
        
        # Construct the SQL query for document-level search
        documents_table = current_app.vector_settings.documents_table_name
        search_sql = f"""
        SELECT document_id, document_keywords, document_tags, document_headings, 
               project_id, embedding, created_at
        FROM {documents_table}
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT %s
        """
        
        params.append(limit)
        
        logging.info(f"VectorStore.document_level_search - Complete SQL query: {search_sql}")
        logging.info(f"VectorStore.document_level_search - Complete parameters list: {params}")
        
        # Execute the query using psycopg
        conn_params = current_app.vector_settings.database_url
        with psycopg.connect(conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(search_sql, params)
                results = cur.fetchall()
        
        logging.info(f"VectorStore.document_level_search - Query returned {len(results)} rows")
        
        elapsed_time = time.time() - start_time
        self._log_search_time("Document-level", elapsed_time)
        
        if return_dataframe:
            df = pd.DataFrame(
                results, 
                columns=["document_id", "document_keywords", "document_tags", 
                        "document_headings", "project_id", "embedding", "created_at"]
            )
            df["document_id"] = df["document_id"].astype(str)
            return df
        else:
            return results

    def get_documents_by_metadata(
        self,
        project_ids: Optional[List[str]] = None,
        document_type_ids: Optional[List[str]] = None,
        order_by: str = "created_at DESC",
        limit: int = 50,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Get documents by direct metadata filtering without search terms.
        
        This method retrieves documents based purely on metadata criteria
        (project ID, document type ID) without any search term matching.
        Useful for queries like "any correspondence for project X".
        
        Args:
            project_ids: List of project IDs to filter by.
            document_type_ids: List of document type IDs to filter by.
            order_by: SQL ORDER BY clause (default: "created_at DESC").
            limit: Maximum number of documents to return (default: 50).
            return_dataframe: If True, returns results as a DataFrame; otherwise as a list of tuples.
            
        Returns:
            Either a pandas DataFrame or a list of tuples containing document results.
        """
        start_time = time.time()
        
        # Build WHERE conditions
        where_conditions = ["TRUE"]
        params = []
        
        # Filter by project IDs
        if project_ids:
            placeholders = ",".join(["%s"] * len(project_ids))
            where_conditions.append(f"project_id IN ({placeholders})")
            params.extend(project_ids)
        
        # Filter by document type IDs
        # Note: This assumes document metadata contains document_type_id (with underscore)
        if document_type_ids:
            placeholders = ",".join(["%s"] * len(document_type_ids))
            where_conditions.append(f"document_metadata->>'document_type_id' IN ({placeholders})")
            params.extend(document_type_ids)
        
        # Build the final WHERE clause
        where_clause = " AND ".join(where_conditions)
        
        import logging
        logging.info(f"Direct metadata search WHERE clause: {where_clause}")
        logging.info(f"Direct metadata search parameters: {params}")
        
        # Construct the SQL query for direct metadata search
        documents_table = current_app.vector_settings.documents_table_name
        metadata_sql = f"""
        SELECT document_id, document_keywords, document_tags, document_headings, 
               project_id, document_metadata, created_at,
               document_metadata->>'document_date' as document_date,
               document_metadata->>'document_name' as document_name,
               document_metadata->>'document_saved_name' as document_saved_name,
               document_metadata->>'project_name' as project_name,
               document_metadata->>'proponent_name' as proponent_name,
               document_metadata->>'s3_key' as s3_key
        FROM {documents_table}
        WHERE {where_clause}
        ORDER BY {order_by}
        LIMIT %s
        """
        
        params.append(limit)
        
        # Execute the query using psycopg
        conn_params = current_app.vector_settings.database_url
        with psycopg.connect(conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(metadata_sql, params)
                results = cur.fetchall()
                
                logging.info(f"Direct metadata search returned {len(results)} results")
        
        elapsed_time = time.time() - start_time
        self._log_search_time("Direct metadata", elapsed_time)
        
        if return_dataframe:
            df = pd.DataFrame(
                results, 
                columns=["document_id", "document_keywords", "document_tags", 
                        "document_headings", "project_id", "document_metadata", 
                        "created_at", "document_date", "document_name", "document_saved_name",
                        "project_name", "proponent_name", "s3_key"]
            )
            df["document_id"] = df["document_id"].astype(str)
            return df
        else:
            return results

    def search_chunks_by_documents(
        self,
        document_ids: List[str],
        query: str,
        limit: int = 20,
        return_dataframe: bool = True,
        use_metadata_boost: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Search for chunks within specific documents using semantic search with metadata boosting.

        This method performs semantic search on the document_chunks table,
        but only within the chunks belonging to the specified documents.
        Optionally boosts relevance scores based on chunk metadata (keywords, tags, headings).

        Args:
            document_ids: List of document IDs to search within.
            query: The search query text.
            limit: Maximum number of chunks to return (default: 20).
            return_dataframe: If True, returns results as a DataFrame; otherwise as a list of tuples.
            use_metadata_boost: If True, boost scores based on metadata matches (default: True).

        Returns:
            Either a pandas DataFrame or a list of tuples containing chunk search results.
        """
        if not document_ids:
            # Return empty result if no documents provided
            if return_dataframe:
                return pd.DataFrame(columns=["id", "metadata", "content", "document_id", "project_id", "similarity"])
            else:
                return []

        from .embedding import get_embedding
        from .keywords.query_keyword_extractor import get_keywords
        from .tags.tag_extractor import get_tags

        start_time = time.time()

        # Get query embedding
        query_embedding = get_embedding([query])
        embedding_list = query_embedding[0].tolist()

        # Extract keywords and tags from query for metadata boosting
        query_keywords = []
        query_tags = []
        if use_metadata_boost:
            try:
                raw_keywords = get_keywords(query)
                query_keywords = [kw for kw, score in raw_keywords] if raw_keywords else []
                query_tags = get_tags(query) or []
                logging.info(f"Chunk search - extracted keywords: {query_keywords[:5]}, tags: {query_tags[:5]}")
            except Exception as e:
                logging.warning(f"Failed to extract keywords/tags for metadata boost: {e}")

        # Create placeholders for document IDs
        placeholders = ','.join(['%s'] * len(document_ids))

        # Construct the SQL query for chunk search within specific documents
        # Uses metadata boost: adds score for chunks whose metadata matches query keywords/tags
        chunks_table = current_app.vector_settings.vector_table_name

        if use_metadata_boost and (query_keywords or query_tags):
            # Enhanced query with metadata boosting
            # Boost formula: semantic_similarity + metadata_boost (capped at 0.15 total boost)
            metadata_boost_parts = []
            boost_params_before = []

            if query_keywords:
                # Boost for keyword matches in metadata
                metadata_boost_parts.append("CASE WHEN (metadata->'keywords') ?| %s THEN 0.05 ELSE 0 END")
                boost_params_before.append(query_keywords)
                # Boost for heading matches
                metadata_boost_parts.append("CASE WHEN (metadata->'headings') ?| %s THEN 0.05 ELSE 0 END")
                boost_params_before.append(query_keywords)

            if query_tags:
                # Boost for tag matches
                metadata_boost_parts.append("CASE WHEN (metadata->'tags') ?| %s THEN 0.05 ELSE 0 END")
                boost_params_before.append(query_tags)

            metadata_boost_sql = " + ".join(metadata_boost_parts) if metadata_boost_parts else "0"

            search_sql = f"""
            SELECT id, metadata, content, document_id, project_id,
                   LEAST(1.0, (1 - (embedding <=> %s::vector)) + ({metadata_boost_sql})) as similarity
            FROM {chunks_table}
            WHERE document_id IN ({placeholders})
            ORDER BY similarity DESC
            LIMIT %s
            """

            # Prepare parameters: embedding, boost params, document_ids, limit
            params = [embedding_list] + boost_params_before + document_ids + [limit]
        else:
            # Standard semantic-only search
            search_sql = f"""
            SELECT id, metadata, content, document_id, project_id,
                   1 - (embedding <=> %s::vector) as similarity
            FROM {chunks_table}
            WHERE document_id IN ({placeholders})
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """

            # Prepare parameters: embedding, document_ids, embedding again, limit
            params = [embedding_list] + document_ids + [embedding_list, limit]

        # Execute the query using psycopg
        conn_params = current_app.vector_settings.database_url
        with psycopg.connect(conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(search_sql, params)
                results = cur.fetchall()

        elapsed_time = time.time() - start_time
        self._log_search_time("Chunk-within-documents", elapsed_time)
        logging.info(f"Chunk search found {len(results)} results (metadata_boost={use_metadata_boost})")

        if return_dataframe:
            df = pd.DataFrame(
                results,
                columns=["id", "metadata", "content", "document_id", "project_id", "similarity"]
            )
            df["id"] = df["id"].astype(str)
            df["document_id"] = df["document_id"].astype(str)
            return df
        else:
            return results

    def _log_search_time(self, search_type: str, elapsed_time: float) -> None:
        """
        Log the time taken for a search operation.
        
        Args:
            search_type: The type of search (e.g., "Vector", "Keyword").
            elapsed_time: The time taken for the search operation in seconds.
        """
        logging.info(f"{search_type} search completed in {elapsed_time:.3f} seconds")

    def get_document_embedding(self, document_id: str) -> Optional[List[float]]:
        """
        Retrieve the embedding vector for a specific document.
        
        Args:
            document_id: The ID of the document to get embedding for.
            
        Returns:
            The document embedding vector as a list, or None if document not found.
        """
        start_time = time.time()
        
        documents_table = current_app.vector_settings.documents_table_name
        search_sql = f"""
        SELECT embedding
        FROM {documents_table}
        WHERE document_id = %s
        """
        
        # Execute the query using psycopg
        conn_params = current_app.vector_settings.database_url
        with psycopg.connect(conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(search_sql, (document_id,))
                result = cur.fetchone()
        
        elapsed_time = time.time() - start_time
        self._log_search_time("Document embedding retrieval", elapsed_time)
        
        if result and result[0]:
            # Convert the embedding to a list - handle different return types from pgvector
            embedding = result[0]
            if isinstance(embedding, str):
                # If it's a string representation, parse it
                try:
                    return ast.literal_eval(embedding)
                except (ValueError, SyntaxError):
                    # If parsing fails, try removing brackets and splitting
                    clean_str = embedding.strip('[]')
                    return [float(x.strip()) for x in clean_str.split(',')]
            elif hasattr(embedding, 'tolist'):
                # If it's a numpy array or similar, convert to list
                return embedding.tolist()
            elif isinstance(embedding, list):
                # If it's already a list, return as-is
                return embedding
            else:
                # Try to convert to list
                return list(embedding)
        else:
            return None

    def search_documents_by_metadata(
        self,
        query: str,
        limit: int = 20,
        project_ids: Optional[List[str]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Search for documents using metadata fields (keywords, tags, headings).

        This method searches the documents table using the pre-computed metadata
        fields without requiring vector similarity search. Useful for quick
        metadata-based filtering and document type inference.

        Args:
            query: The search query text.
            limit: Maximum number of documents to return (default: 20).
            project_ids: Optional list of project IDs to filter by.
            return_dataframe: If True, returns results as a DataFrame.

        Returns:
            Either a pandas DataFrame or a list of tuples containing document results.
        """
        from .keywords.query_keyword_extractor import get_keywords
        from .tags.tag_extractor import get_tags

        start_time = time.time()

        # Extract keywords and tags from query
        try:
            raw_keywords = get_keywords(query)
            query_keywords = [kw for kw, score in raw_keywords] if raw_keywords else []
        except Exception as e:
            logging.warning(f"Failed to extract keywords: {e}")
            query_keywords = []

        try:
            query_tags = get_tags(query) or []
        except Exception as e:
            logging.warning(f"Failed to extract tags: {e}")
            query_tags = []

        logging.info(f"Metadata search - query: '{query}', keywords: {query_keywords[:5]}, tags: {query_tags[:5]}")

        # Build WHERE clause
        where_conditions = ["TRUE"]
        params = []
        search_conditions = []

        # Add keyword/tag/heading search conditions
        if query_keywords:
            search_conditions.append("document_keywords ?| %s")
            params.append(query_keywords)
            search_conditions.append("document_headings ?| %s")
            params.append(query_keywords)

        if query_tags:
            search_conditions.append("document_tags ?| %s")
            params.append(query_tags)

        if search_conditions:
            where_conditions.append("(" + " OR ".join(search_conditions) + ")")

        # Add project filter if provided
        if project_ids and len(project_ids) > 0:
            placeholders = ','.join(['%s'] * len(project_ids))
            where_conditions.append(f"project_id IN ({placeholders})")
            params.extend(project_ids)

        where_clause = " AND ".join(where_conditions)

        # Build SQL query
        documents_table = current_app.vector_settings.documents_table_name
        search_sql = f"""
        SELECT document_id, document_keywords, document_tags, document_headings,
               project_id, document_metadata, created_at
        FROM {documents_table}
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT %s
        """
        params.append(limit)

        # Execute query
        conn_params = current_app.vector_settings.database_url
        with psycopg.connect(conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(search_sql, params)
                results = cur.fetchall()

        elapsed_time = time.time() - start_time
        self._log_search_time("Metadata-based document search", elapsed_time)
        logging.info(f"Metadata search found {len(results)} documents")

        if return_dataframe:
            df = pd.DataFrame(
                results,
                columns=["document_id", "document_keywords", "document_tags",
                         "document_headings", "project_id", "document_metadata", "created_at"]
            )
            df["document_id"] = df["document_id"].astype(str)
            return df
        else:
            return results

    def search_chunks_with_metadata_filter(
        self,
        query: str,
        limit: int = 20,
        project_ids: Optional[List[str]] = None,
        document_type_ids: Optional[List[str]] = None,
        use_semantic: bool = True,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Search for chunks with combined semantic search and metadata filtering.

        This method performs a two-stage search:
        1. Find documents matching project/document_type filters and metadata
        2. Search chunks within those documents using semantic + metadata boost

        Args:
            query: The search query text.
            limit: Maximum number of chunks to return (default: 20).
            project_ids: Optional list of project IDs to filter by.
            document_type_ids: Optional list of document type IDs to filter by.
            use_semantic: If True, use semantic search; otherwise keyword-only.
            return_dataframe: If True, returns results as a DataFrame.

        Returns:
            Either a pandas DataFrame or a list of tuples containing chunk results.
        """
        from .keywords.query_keyword_extractor import get_keywords
        from .tags.tag_extractor import get_tags

        start_time = time.time()

        # Extract keywords and tags
        try:
            raw_keywords = get_keywords(query)
            query_keywords = [kw for kw, score in raw_keywords] if raw_keywords else []
        except Exception:
            query_keywords = []

        try:
            query_tags = get_tags(query) or []
        except Exception:
            query_tags = []

        logging.info(f"Chunk metadata search - keywords: {query_keywords[:5]}, tags: {query_tags[:5]}")

        # Step 1: Find matching documents
        doc_where_conditions = ["TRUE"]
        doc_params = []
        doc_search_conditions = []

        if query_keywords:
            doc_search_conditions.append("document_keywords ?| %s")
            doc_params.append(query_keywords)
            doc_search_conditions.append("document_headings ?| %s")
            doc_params.append(query_keywords)

        if query_tags:
            doc_search_conditions.append("document_tags ?| %s")
            doc_params.append(query_tags)

        if doc_search_conditions:
            doc_where_conditions.append("(" + " OR ".join(doc_search_conditions) + ")")

        if project_ids and len(project_ids) > 0:
            placeholders = ','.join(['%s'] * len(project_ids))
            doc_where_conditions.append(f"project_id IN ({placeholders})")
            doc_params.extend(project_ids)

        if document_type_ids and len(document_type_ids) > 0:
            placeholders = ','.join(['%s'] * len(document_type_ids))
            doc_where_conditions.append(f"document_metadata->>'document_type_id' IN ({placeholders})")
            doc_params.extend(document_type_ids)

        doc_where_clause = " AND ".join(doc_where_conditions)

        # Query for document IDs
        documents_table = current_app.vector_settings.documents_table_name
        doc_sql = f"""
        SELECT document_id
        FROM {documents_table}
        WHERE {doc_where_clause}
        ORDER BY created_at DESC
        LIMIT 100
        """

        conn_params = current_app.vector_settings.database_url
        with psycopg.connect(conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(doc_sql, doc_params)
                doc_results = cur.fetchall()

        document_ids = [str(r[0]) for r in doc_results]
        logging.info(f"Chunk metadata search - found {len(document_ids)} matching documents")

        if not document_ids:
            elapsed_time = time.time() - start_time
            self._log_search_time("Chunk metadata search (no docs)", elapsed_time)
            if return_dataframe:
                return pd.DataFrame(columns=["id", "metadata", "content", "document_id", "project_id", "similarity"])
            else:
                return []

        # Step 2: Search chunks within matching documents
        if use_semantic:
            # Use the enhanced search with metadata boost
            return self.search_chunks_by_documents(
                document_ids=document_ids,
                query=query,
                limit=limit,
                return_dataframe=return_dataframe,
                use_metadata_boost=True
            )
        else:
            # Keyword-only chunk search
            chunks_table = current_app.vector_settings.vector_table_name
            placeholders = ','.join(['%s'] * len(document_ids))
            chunk_where_conditions = [f"document_id IN ({placeholders})"]
            chunk_params = document_ids.copy()
            chunk_search_conditions = []

            if query_keywords:
                chunk_search_conditions.append("(metadata->'keywords') ?| %s")
                chunk_params.append(query_keywords)
                chunk_search_conditions.append("(metadata->'headings') ?| %s")
                chunk_params.append(query_keywords)

            if query_tags:
                chunk_search_conditions.append("(metadata->'tags') ?| %s")
                chunk_params.append(query_tags)

            if chunk_search_conditions:
                chunk_where_conditions.append("(" + " OR ".join(chunk_search_conditions) + ")")

            chunk_where_clause = " AND ".join(chunk_where_conditions)
            chunk_sql = f"""
            SELECT id, metadata, content, document_id, project_id, 0.5 as similarity
            FROM {chunks_table}
            WHERE {chunk_where_clause}
            ORDER BY id DESC
            LIMIT %s
            """
            chunk_params.append(limit)

            with psycopg.connect(conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(chunk_sql, chunk_params)
                    results = cur.fetchall()

            elapsed_time = time.time() - start_time
            self._log_search_time("Chunk metadata search", elapsed_time)

            if return_dataframe:
                df = pd.DataFrame(
                    results,
                    columns=["id", "metadata", "content", "document_id", "project_id", "similarity"]
                )
                df["id"] = df["id"].astype(str)
                df["document_id"] = df["document_id"].astype(str)
                return df
            else:
                return results

    def document_similarity_search(
        self,
        source_embedding: List[float],
        exclude_document_id: str,
        predicates: Optional[dict] = None,
        limit: int = 10,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Search for documents similar to the source embedding using cosine similarity.
        
        This method performs similarity search on document-level embeddings to find
        documents that are semantically similar to the source document.
        
        Args:
            source_embedding: The embedding vector to search for similar documents.
            exclude_document_id: Document ID to exclude from results (the source document).
            predicates: Optional dictionary of field-value pairs to filter results.
            limit: Maximum number of similar documents to return (default: 10).
            return_dataframe: If True, returns results as a DataFrame; otherwise as a list of tuples.
            
        Returns:
            Either a pandas DataFrame or a list of tuples containing similar documents.
        """
        start_time = time.time()
        
        # Build the WHERE clause based on filters
        where_conditions = ["document_id != %s"]  # Exclude the source document
        params = [exclude_document_id]
        
        # Handle project filtering
        if predicates:
            for key, value in predicates.items():
                if key == 'project_ids':
                    # Handle multiple project IDs
                    if value and len(value) > 0:
                        placeholders = ','.join(['%s'] * len(value))
                        where_conditions.append(f"project_id IN ({placeholders})")
                        params.extend(value)
                elif key == 'project_id':
                    where_conditions.append("project_id = %s")
                    params.append(value)
                elif key == 'document_type_ids':
                    # Handle multiple document type IDs
                    if value and len(value) > 0:
                        placeholders = ','.join(['%s'] * len(value))
                        where_conditions.append(f"document_metadata->>'document_type_id' IN ({placeholders})")
                        params.extend(value)
                # Add other predicate handling as needed
        
        # Build the final WHERE clause
        where_clause = " AND ".join(where_conditions)
        
        # Convert embedding to the format expected by the database
        if isinstance(source_embedding, list):
            embedding_list = source_embedding
        elif hasattr(source_embedding, 'tolist'):
            embedding_list = source_embedding.tolist()
        elif isinstance(source_embedding, str):
            # Handle string representation of embedding
            try:
                embedding_list = ast.literal_eval(source_embedding)
            except (ValueError, SyntaxError):
                # If parsing fails, try removing brackets and splitting
                clean_str = source_embedding.strip('[]')
                embedding_list = [float(x.strip()) for x in clean_str.split(',')]
        else:
            # Try to convert to list
            embedding_list = list(source_embedding)
        
        # Construct the SQL query for document similarity search
        documents_table = current_app.vector_settings.documents_table_name
        search_sql = f"""
        SELECT document_id, document_keywords, document_tags, document_headings, 
               project_id, embedding, created_at, document_metadata,
               1 - (embedding <=> %s::vector) as similarity
        FROM {documents_table}
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        
        # Convert embedding list to a format PostgreSQL can understand
        embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'
        
        # Build final params list in the correct order for the SQL query:
        # 1. embedding (for similarity calculation)
        # 2. document_id and other WHERE clause params
        # 3. embedding (for ORDER BY)
        # 4. limit
        final_params = [embedding_str] + params + [embedding_str, limit]
        
        # Execute the query using psycopg
        conn_params = current_app.vector_settings.database_url
        with psycopg.connect(conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(search_sql, final_params)
                results = cur.fetchall()
        
        elapsed_time = time.time() - start_time
        self._log_search_time("Document similarity", elapsed_time)
        
        if return_dataframe:
            df = pd.DataFrame(
                results, 
                columns=["document_id", "document_keywords", "document_tags", 
                        "document_headings", "project_id", "embedding", "created_at", 
                        "document_metadata", "similarity"]
            )
            df["document_id"] = df["document_id"].astype(str)
            return df
        else:
            return results
