# Copyright © 2024 Province of British Columbia
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""API endpoints for vector and keyword-based document search operations.

This module provides REST API endpoints for searching documents using multiple search strategies:
1. Semantic vector search using document embeddings for meaning-based matches
2. Keyword-based search for traditional text matching  
3. Document similarity search using document-level embeddings
4. Two-stage search combining document-level filtering with chunk retrieval

The implementation uses Flask-RESTx for API definition with Swagger documentation,
Marshmallow for request validation, and delegates search logic to the SearchService.
Results include both matched documents and detailed performance metrics for each
search stage in the pipeline.

Database Structure:
- documents table: Contains document-level metadata (keywords, tags, headings, embeddings)
- document_chunks table: Contains text chunks with embeddings for semantic search
"""

from http import HTTPStatus

from flask_restx import Namespace, Resource
from flask import Response
from marshmallow import EXCLUDE, Schema, fields

import json

from services.search_service import SearchService
from .apihelper import Api as ApiHelper
from .query_enhancement import is_query_location_relevant, format_user_location_for_query


class UserLocationSchema(Schema):
    """Schema for validating user location data.
    
    Defines the structure for user location information including
    geographic coordinates and location metadata.
    
    Attributes:
        latitude: Geographic latitude coordinate
        longitude: Geographic longitude coordinate
        city: City name
        region: Region/province/state name
        country: Country name
        timestamp: Unix timestamp (milliseconds) when location was captured
    """

    class Meta:  # pylint: disable=too-few-public-methods
        """Exclude unknown fields in the deserialized output."""

        unknown = EXCLUDE

    latitude = fields.Float(data_key="latitude", required=False,
                           metadata={"description": "Geographic latitude coordinate (-90 to 90)"})
    longitude = fields.Float(data_key="longitude", required=False,
                            metadata={"description": "Geographic longitude coordinate (-180 to 180)"})
    city = fields.Str(data_key="city", required=False,
                     metadata={"description": "City name"})
    region = fields.Str(data_key="region", required=False,
                       metadata={"description": "Region, province, or state name"})
    country = fields.Str(data_key="country", required=False,
                        metadata={"description": "Country name"})
    timestamp = fields.Int(data_key="timestamp", required=False,
                          metadata={"description": "Unix timestamp in milliseconds when location was captured"})


class RankingConfigSchema(Schema):
    """Schema for validating ranking configuration.
    
    Defines the structure for ranking-related parameters including
    minimum relevance score and top N results count.
    
    Attributes:
        minScore: Optional minimum relevance score threshold for filtering results
        topN: Optional maximum number of results to return after ranking
    """

    class Meta:  # pylint: disable=too-few-public-methods
        """Exclude unknown fields in the deserialized output."""

        unknown = EXCLUDE

    minScore = fields.Float(data_key="minScore", required=False, 
                           metadata={"description": "Minimum relevance score threshold for filtering results. If not provided, uses MIN_RELEVANCE_SCORE environment variable setting (default: -8.0). Cross-encoder models can produce negative scores for relevant documents."})
    topN = fields.Int(data_key="topN", required=False, validate=lambda x: 1 <= x <= 100,
                     metadata={"description": "Maximum number of results to return after ranking (1-100). If not provided, uses TOP_RECORD_COUNT environment variable setting (default: 10)."})



class DocumentSimilarityRequestSchema(Schema):
    """Schema for validating document similarity search requests.
    
    Defines the structure and validation rules for document similarity requests,
    which find documents similar to a given document using document-level embeddings.
    
    Attributes:
        documentId: The required document ID to find similar documents for
        projectIds: Optional list of project IDs to filter search results
        limit: Optional limit for number of similar documents to return
    """

    class Meta:  # pylint: disable=too-few-public-methods
        """Exclude unknown fields in the deserialized output."""

        unknown = EXCLUDE

    documentId = fields.Str(data_key="documentId", required=True, 
                          metadata={"description": "Document ID to find similar documents for"})
    projectIds = fields.List(fields.Str(), data_key="projectIds", required=False, 
                           metadata={"description": "Optional list of project IDs to filter search results. If not provided, searches across all projects."})
    limit = fields.Int(data_key="limit", required=False, load_default=10, validate=lambda x: 1 <= x <= 50,
                      metadata={"description": "Maximum number of similar documents to return (1-50, default: 10)"})


class SearchRequestSchema(Schema):
    """Schema for validating and deserializing search API requests.
    
    Defines the structure and validation rules for incoming search requests,
    ensuring that required fields are present and properly formatted.
    
    Attributes:
        query: The required search query string provided by the user
        semanticQuery: Optional pre-optimized semantic query for vector search
        projectIds: Optional list of project IDs to filter search results
        documentTypeIds: Optional list of document type IDs to filter search results
        inference: Optional list of inference types to run ('PROJECT', 'DOCUMENTTYPE')
    """

    class Meta:  # pylint: disable=too-few-public-methods
        """Exclude unknown fields in the deserialized output."""

        unknown = EXCLUDE

    query = fields.Str(data_key="query", required=True, 
                      metadata={"description": "Search query text to find relevant documents"})
    semanticQuery = fields.Str(data_key="semanticQuery", required=False, 
                              metadata={"description": "Optional pre-optimized semantic query for vector search. If provided, bypasses automatic query cleaning and uses this query directly for semantic/vector operations. Useful for advanced users who want full control over the semantic search query."})
    projectIds = fields.List(fields.Str(), data_key="projectIds", required=False, 
                           metadata={"description": "Optional list of project IDs to filter search results. If not provided, searches across all projects."})
    documentTypeIds = fields.List(fields.Str(), data_key="documentTypeIds", required=False, 
                                metadata={"description": "Optional list of document type IDs to filter search results. If not provided, may be automatically inferred from the query."})
    inference = fields.List(fields.Str(validate=lambda x: x in ['PROJECT', 'DOCUMENTTYPE']), 
                          data_key="inference", required=False, load_default=None,
                          metadata={"description": "Optional list of inference types to run. Valid values: 'PROJECT', 'DOCUMENTTYPE'. If not provided, uses USE_DEFAULT_INFERENCE environment variable setting."})
    searchStrategy = fields.Str(data_key="searchStrategy", required=False, 
                               validate=lambda x: x in ['HYBRID_SEMANTIC_FALLBACK', 'HYBRID_KEYWORD_FALLBACK', 'SEMANTIC_ONLY', 'KEYWORD_ONLY', 'HYBRID_PARALLEL', 'DOCUMENT_ONLY'],
                               metadata={"description": "Optional search strategy to use. Valid values: 'HYBRID_SEMANTIC_FALLBACK' (default), 'HYBRID_KEYWORD_FALLBACK', 'SEMANTIC_ONLY', 'KEYWORD_ONLY', 'HYBRID_PARALLEL', 'DOCUMENT_ONLY'. If not provided, uses DEFAULT_SEARCH_STRATEGY environment variable setting."})
    ranking = fields.Nested(RankingConfigSchema, data_key="ranking", required=False,
                           metadata={"description": "Optional ranking configuration including minimum score threshold and maximum result count. If not provided, uses environment variable settings."})
    userLocation = fields.Nested(UserLocationSchema, data_key="userLocation", required=False,
                                metadata={"description": "Optional structured user location data including coordinates (latitude, longitude), city, region, country, and timestamp. Represents the user's current geographic position."})
    location = fields.Str(data_key="location", required=False,
                         metadata={"description": "Optional location context to enhance search relevance (e.g., 'Langford British Columbia'). Currently appended to search query for improved semantic matching."})
    projectStatus = fields.Str(data_key="projectStatus", required=False,
                              metadata={"description": "Optional project status context to enhance search relevance (e.g., 'recent', 'active', 'completed'). Currently appended to search query for improved semantic matching."})
    years = fields.List(fields.Int(), data_key="years", required=False,
                       metadata={"description": "Optional list of years to focus search on (e.g., [2023, 2024, 2025]). Currently appended to search query for improved semantic matching."})
    semanticFetchCount = fields.Int(data_key="semanticFetchCount", required=False,
                                    validate=lambda x: 5 <= x <= 200,
                                    metadata={"description": "Override the number of semantic candidates fetched before re-ranking (5-200). Defaults to SEMANTIC_FETCH_COUNT env var."})
    keywordFetchCount = fields.Int(data_key="keywordFetchCount", required=False,
                                   validate=lambda x: 5 <= x <= 200,
                                   metadata={"description": "Override the number of keyword candidates fetched before re-ranking (5-200). Defaults to KEYWORD_FETCH_COUNT env var."})


API = Namespace("vector-search", description="Endpoints for semantic and keyword vector search operations")
SIMILARITY_API = Namespace("document-similarity", description="Endpoints for document similarity search operations")

search_request_model = ApiHelper.convert_ma_schema_to_restx_model(
    API, SearchRequestSchema(), "Vector Search Request"
)

document_similarity_request_model = ApiHelper.convert_ma_schema_to_restx_model(
    SIMILARITY_API, DocumentSimilarityRequestSchema(), "Document Similarity Request"
)


@API.route("", methods=["POST", "OPTIONS"])
class Search(Resource):
    """REST resource for advanced two-stage document search operations.
    
    This resource exposes endpoints for searching documents using a modern two-stage
    approach that first filters at the document level using pre-computed metadata,
    then performs semantic search within relevant document chunks.
    
    The implementation prioritizes both search quality and performance through:
    - Intelligent project inference with automatic query cleaning
    - Document-level filtering using keywords, tags, and headings
    - Semantic vector search within relevant chunks
    - Cross-encoder re-ranking for optimal relevance
    - Optional project-based filtering
    """

    @staticmethod
    @ApiHelper.swagger_decorators(API, endpoint_description="Query the vector database for semantically similar documents")
    @API.expect(search_request_model)
    @API.response(400, "Bad Request")
    @API.response(200, "Search successful")
    def post():
        """Perform an advanced two-stage search operation against the document database.
        
        This endpoint implements a modern search strategy that leverages document-level
        metadata for improved efficiency and accuracy:
        
        Stage 0: Project Inference (when no project IDs provided)
        - Automatically detects project references in natural language queries
        - Applies project filtering when highly confident (>80% threshold)
        - Removes identified project names from search terms to focus on actual topics
        
        Stage 1: Document Discovery
        - Searches the documents table using pre-computed keywords, tags, and headings
        - Identifies the most relevant documents based on metadata matching
        - Much faster than searching all chunks directly
        
        Stage 2: Chunk Retrieval  
        - Performs semantic vector search within chunks of relevant documents
        - Uses embeddings to find the most semantically similar content
        - Returns the best matching chunks from promising documents
        
        Alternative Path:
        - If no relevant documents found in Stage 1, performs semantic search across all chunks
        - Ensures comprehensive coverage even for edge cases
        
        The pipeline also includes cross-encoder re-ranking for optimal relevance.
        
        Inference Control:
        The optional 'inference' parameter controls which inference pipelines to run:
        - ['PROJECT'] - Only run project inference
        - ['DOCUMENTTYPE'] - Only run document type inference
        - ['PROJECT', 'DOCUMENTTYPE'] - Run both inference pipelines (default behavior)
        - [] - Disable all inference pipelines
        - Not provided - Uses USE_DEFAULT_INFERENCE environment variable setting
        
        Search Strategy Control:
        The optional 'searchStrategy' parameter controls the search approach used:
        - HYBRID_SEMANTIC_FALLBACK: Document keyword filter → Semantic search → Keyword fallback (default)
        - HYBRID_KEYWORD_FALLBACK: Document keyword filter → Keyword search → Semantic fallback  
        - SEMANTIC_ONLY: Pure semantic search without keyword filtering or fallbacks
        - KEYWORD_ONLY: Pure keyword search without semantic components
        - HYBRID_PARALLEL: Run both semantic and keyword searches in parallel and merge results
        - DOCUMENT_ONLY: Direct document-level search without chunk analysis (auto-used for generic requests)
        - Not provided - Uses DEFAULT_SEARCH_STRATEGY environment variable setting (HYBRID_SEMANTIC_FALLBACK)
        
        Ranking Configuration:
        The optional 'ranking' object controls result filtering and limiting:
        - minScore: Minimum relevance score threshold (default: MIN_RELEVANCE_SCORE env var, currently -8.0)
        - topN: Maximum number of results to return (default: TOP_RECORD_COUNT env var, currently 10)
        - Cross-encoder models can produce negative scores for relevant documents
        - Lower minScore values are more inclusive, higher values are more restrictive
        
        Query Enhancement Parameters:
        The following optional parameters enhance the search query for improved semantic matching:
        - userLocation: Structured user location data with coordinates (latitude, longitude), city, region, country, and timestamp
        - location: Geographic context string (e.g., "Langford British Columbia") - appended to query
        - projectStatus: Project status context (e.g., "recent", "active", "completed") - appended to query  
        - years: List of relevant years (e.g., [2023, 2024, 2025]) - appended to query
        These parameters are currently integrated into the search query text for semantic processing.
        
        Returns:
            Response: JSON containing matched documents and detailed search metrics
                     for each stage of the search pipeline, including project inference
                     metadata when applicable
        """
        request_data = SearchRequestSchema().load(API.payload)
        query = request_data["query"]
        semantic_query = request_data.get("semanticQuery", None)  # Optional parameter
        project_ids = request_data.get("projectIds", None)  # Optional parameter
        document_type_ids = request_data.get("documentTypeIds", None)  # Optional parameter
        inference = request_data.get("inference", None)  # Optional parameter
        search_strategy = request_data.get("searchStrategy", None)  # Optional parameter
        ranking_config = request_data.get("ranking", {})  # Optional parameter
        user_location = request_data.get("userLocation", None)  # Optional parameter
        location = request_data.get("location", None)  # Optional parameter
        project_status = request_data.get("projectStatus", None)  # Optional parameter
        years = request_data.get("years", None)  # Optional parameter
        semantic_fetch_count = request_data.get("semanticFetchCount", None)  # Optional override
        keyword_fetch_count = request_data.get("keywordFetchCount", None)   # Optional override

        # Extract ranking parameters with fallback to None (will use env defaults)
        min_relevance_score = ranking_config.get("minScore") if ranking_config else None
        top_n = ranking_config.get("topN") if ranking_config else None
        
        # Enhance the query with additional context parameters
        enhanced_query = query
        query_enhancements = []
        
        # Smart user location enhancement - only add if query is location-relevant
        if user_location:
            is_location_relevant, relevance_score, relevance_metadata = is_query_location_relevant(query)
            
            if is_location_relevant:
                # Query is location-relevant (e.g., "projects near me"), enhance with user location
                user_loc_str = format_user_location_for_query(user_location)
                if user_loc_str:
                    query_enhancements.append(f"user location: {user_loc_str}")
                    # Log the decision for debugging/analytics
                    import logging
                    logging.info(f"Query location-relevant (score: {relevance_score:.2f}). "
                                f"Enhanced with user location: {user_loc_str}")
            else:
                # Query is not location-relevant, skip user location enhancement
                import logging
                logging.debug(f"Query not location-relevant (score: {relevance_score:.2f}). "
                             f"Skipping user location enhancement. "
                             f"Reason: {relevance_metadata.get('reasoning', 'N/A')}")
        
        # Always add explicit location string if provided (user manually specified)
        if location:
            query_enhancements.append(f"location: {location}")
        
        if project_status:
            query_enhancements.append(f"project status: {project_status}")
        
        if years:
            years_str = ", ".join(str(year) for year in years)
            query_enhancements.append(f"years: {years_str}")
        
        # Append enhancements to the query if any exist
        if query_enhancements:
            enhanced_query = f"{query} ({' | '.join(query_enhancements)})"
        
        documents = SearchService.get_documents_by_query(enhanced_query, project_ids, document_type_ids, inference, min_relevance_score, top_n, search_strategy, semantic_query, semantic_fetch_count=semantic_fetch_count, keyword_fetch_count=keyword_fetch_count)
        return Response(
            json.dumps(documents), status=HTTPStatus.OK, mimetype="application/json"
        )


@SIMILARITY_API.route("", methods=["POST", "OPTIONS"])
class DocumentSimilarity(Resource):
    """REST resource for document similarity search operations.
    
    This resource exposes endpoints for finding documents similar to a given document
    using document-level embeddings. The similarity is computed using cosine similarity
    on the embeddings of document keywords, tags, and headings.
    
    This is useful for:
    - Finding related documents within the same project
    - Discovering similar documents across different projects
    - Content recommendation and document clustering
    """

    @staticmethod
    @ApiHelper.swagger_decorators(SIMILARITY_API, endpoint_description="Find documents similar to a given document using document-level embeddings")
    @SIMILARITY_API.expect(document_similarity_request_model)
    @API.response(400, "Bad Request")
    @API.response(404, "Document Not Found")
    @API.response(200, "Similarity search successful")
    def post():
        """Find documents similar to the specified document.
        
        This endpoint takes a document ID and returns other documents that are
        semantically similar based on their document-level embeddings. The process:
        
        1. Retrieves the embedding vector for the specified document
        2. Performs cosine similarity search against other document embeddings
        3. Optionally filters results by project IDs
        4. Returns the most similar documents ranked by similarity score
        
        The document embeddings represent the semantic content of the document's
        keywords, tags, and headings, making this ideal for finding thematically
        related documents.
        
        Returns:
            Response: JSON containing similar documents and search metrics
        """
        request_data = DocumentSimilarityRequestSchema().load(API.payload)
        document_id = request_data["documentId"]
        project_ids = request_data.get("projectIds", None)
        limit = request_data.get("limit", 10)
        
        similar_documents = SearchService.get_similar_documents(document_id, project_ids, limit)
        return Response(
            json.dumps(similar_documents), status=HTTPStatus.OK, mimetype="application/json"
        )
