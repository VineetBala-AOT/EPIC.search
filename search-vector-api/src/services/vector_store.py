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
from .tag_extractor import get_tags

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
            for key, value in predicates.items():
                if key == 'project_ids':
                    # Handle multiple project IDs for chunk search
                    if value and len(value) > 0:
                        placeholders = ','.join(['%s'] * len(value))
                        # For document_chunks table, project_id is a direct column
                        if table_name == current_app.vector_settings.vector_table_name:
                            where_conditions.append(f"project_id IN ({placeholders})")
                            params.extend(value)
                        else:
                            # For other tables, check metadata
                            where_conditions.append(f"metadata->>'project_id' IN ({placeholders})")
                            params.extend(value)
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
    ) -> Union[List[Tuple[str, str, float]], pd.DataFrame]:
        """
        Search for documents using keyword-based full-text search.
        
        This method uses PostgreSQL's full-text search capabilities to find documents
        that match the keywords extracted from the query. The keywords should be provided
        by the caller (extracted and timed at a higher level).
        
        Args:
            table_name: The table to search in.
            query: The search query text.
            limit: Maximum number of results to return (default: 5).
            return_dataframe: If True, returns results as a DataFrame; otherwise as a list of tuples.
            weighted_keywords: List of (keyword, weight) tuples extracted from the query.
        
        Returns:
            Either a pandas DataFrame or a list of tuples containing search results.
        """
        if weighted_keywords is None:
            raise ValueError("weighted_keywords must be provided by the caller.")
        tags = get_tags(query)
        keywords = [keyword for keyword in weighted_keywords]
        tsquery_str = " OR ".join(keywords)  
        # Disable automatic tag filtering to avoid overly restrictive results
        tags_condition = "TRUE"  # Always true - no tag filtering
        
        if tags:
            logging.info(f"Keyword search - Detected tags (not filtering): {tags}")
        else:
            logging.info(f"Keyword search - No tags found for query: '{query}'")
        # Note: document_metadata only exists on documents table, not on document_chunks
        if table_name == "documents":
            search_sql = f"""
            SELECT id, content, metadata, document_metadata, ts_rank_cd(to_tsvector('simple', content), query) as rank
            FROM {table_name}, websearch_to_tsquery('simple', %s) query
            WHERE to_tsvector('simple', content) @@ query  AND {tags_condition}
            ORDER BY rank DESC
            LIMIT %s
            """
        else:
            # For document_chunks table, don't select document_metadata
            search_sql = f"""
            SELECT id, content, metadata, ts_rank_cd(to_tsvector('simple', content), query) as rank
            FROM {table_name}, websearch_to_tsquery('simple', %s) query
            WHERE to_tsvector('simple', content) @@ query  AND {tags_condition}
            ORDER BY rank DESC
            LIMIT %s
            """

        start_time = time.time()
        # Attempt to reconstruct the raw SQL for debugging (not for production use)
        param_tuple = (tsquery_str, limit)
        try:
            reconstructed_sql = search_sql.replace('%s', '{}').format(*[repr(p) for p in param_tuple])
            logging.info(f"Keyword search reconstructed SQL: {reconstructed_sql}")
        except Exception as e:
            logging.info(f"Could not reconstruct SQL: {e}")

        # Create a new connection using psycopg
        conn_params = current_app.vector_settings.database_url
        with psycopg.connect(conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(search_sql, (tsquery_str, limit))
                results = cur.fetchall()

        elapsed_time = time.time() - start_time
        self._log_search_time("Keyword", elapsed_time)

        if return_dataframe:
            # Note: document_metadata only included when querying documents table
            if table_name == "documents":
                columns = ["id", "content", "metadata", "document_metadata", "rank"]
            else:
                columns = ["id", "content", "metadata", "rank"]
            df = pd.DataFrame(results, columns=columns)
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
        from .bert_keyword_extractor import extract_keywords_for_document_search
        from .tag_extractor import get_tags
        
        start_time = time.time()
        
        # Extract keywords and tags from query
        query_keywords = extract_keywords_for_document_search(query)
        query_tags = get_tags(query)
        
        # Debug logging
        import logging
        logging.info(f"Document-level search for query: '{query}'")
        logging.info(f"Extracted keywords: {query_keywords}")
        logging.info(f"Extracted tags: {query_tags}")
        
        # Build the WHERE clause based on filters using OR logic for search terms
        where_conditions = ["TRUE"]
        params = []
        
        # Build search conditions using OR logic between different search criteria
        search_conditions = []
        
        # Search for keyword matches in document_keywords JSONB field
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
        
        # Combine search conditions with OR logic
        if search_conditions:
            where_conditions.append("(" + " OR ".join(search_conditions) + ")")
        else:
            # If no search terms found, don't filter by search criteria
            # This allows the fallback search to handle it
            pass
        
        # Handle predicates for additional filtering
        if predicates:
            for key, value in predicates.items():
                if key == 'project_id':
                    where_conditions.append("project_id = %s")
                    params.append(value)
                elif key == 'project_ids':
                    # Handle multiple project IDs
                    if value and len(value) > 0:
                        placeholders = ','.join(['%s'] * len(value))
                        where_conditions.append(f"project_id IN ({placeholders})")
                        params.extend(value)
                elif key == 'document_type_ids':
                    # Handle multiple document type IDs
                    if value and len(value) > 0:
                        placeholders = ','.join(['%s'] * len(value))
                        where_conditions.append(f"document_metadata->>'document_type_id' IN ({placeholders})")
                        params.extend(value)
                # Add other predicate handling as needed
        
        # Build the final WHERE clause
        where_clause = " AND ".join(where_conditions)
        
        logging.info(f"Document search WHERE clause: {where_clause}")
        logging.info(f"Document search parameters: {params}")
        
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
        
        # Execute the query using psycopg
        conn_params = current_app.vector_settings.database_url
        with psycopg.connect(conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(search_sql, params)
                results = cur.fetchall()
        
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
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Search for chunks within specific documents using semantic search.
        
        This method performs semantic search on the document_chunks table,
        but only within the chunks belonging to the specified documents.
        
        Args:
            document_ids: List of document IDs to search within.
            query: The search query text.
            limit: Maximum number of chunks to return (default: 20).
            return_dataframe: If True, returns results as a DataFrame; otherwise as a list of tuples.
            
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
        
        start_time = time.time()
        
        # Get query embedding
        query_embedding = get_embedding([query])
        embedding_list = query_embedding[0].tolist()
        
        # Create placeholders for document IDs
        placeholders = ','.join(['%s'] * len(document_ids))
        
        # Construct the SQL query for chunk search within specific documents
        chunks_table = current_app.vector_settings.vector_table_name
        # Note: document_metadata does not exist on document_chunks table
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
