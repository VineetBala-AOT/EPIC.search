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
        projectIds: Optional list of project IDs to filter search results
    """

    class Meta:  # pylint: disable=too-few-public-methods
        """Exclude unknown fields in the deserialized output."""

        unknown = EXCLUDE

    query = fields.Str(data_key="query", required=True, 
                      metadata={"description": "Search query text to find relevant documents"})
    projectIds = fields.List(fields.Str(), data_key="projectIds", required=False, 
                           metadata={"description": "Optional list of project IDs to filter search results. If not provided, searches across all projects."})


API = Namespace("vector-search", description="Endpoints for semantic and keyword vector search operations")

search_request_model = ApiHelper.convert_ma_schema_to_restx_model(
    API, SearchRequestSchema(), "Vector Search Request"
)

document_similarity_request_model = ApiHelper.convert_ma_schema_to_restx_model(
    API, DocumentSimilarityRequestSchema(), "Document Similarity Request"
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
        
        Fallback Strategy:
        - If no relevant documents found in Stage 1, searches all chunks
        - Ensures comprehensive coverage even for edge cases
        
        The pipeline also includes cross-encoder re-ranking for optimal relevance.
        
        Returns:
            Response: JSON containing matched documents and detailed search metrics
                     for each stage of the search pipeline, including project inference
                     metadata when applicable
        """
        request_data = SearchRequestSchema().load(API.payload)
        query = request_data["query"]
        project_ids = request_data.get("projectIds", None)  # Optional parameter
        
        documents = SearchService.get_documents_by_query(query, project_ids)
        return Response(
            json.dumps(documents), status=HTTPStatus.OK, mimetype="application/json"
        )


@API.route("/similar", methods=["POST", "OPTIONS"])
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
    @ApiHelper.swagger_decorators(API, endpoint_description="Find documents similar to a given document using document-level embeddings")
    @API.expect(document_similarity_request_model)
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
