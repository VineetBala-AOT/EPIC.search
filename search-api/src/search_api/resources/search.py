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
"""API endpoints for managing an search resource."""

from http import HTTPStatus
from flask_restx import Namespace, Resource
from flask import Response, current_app, request, stream_with_context
import time
import json
import threading

from search_api.clients.vector_search_client import VectorSearchClient
from search_api.services.search_service import SearchService
from search_api.utils.util import cors_preflight
from search_api.schemas.search import SearchRequestSchema
from search_api.schemas.search import SimilaritySearchRequestSchema
from search_api.schemas.search import FeedbackSchema
from search_api.auth import auth
from .apihelper import Api as ApiHelper
from flask import Response, current_app

import json

API = Namespace("search", description="Endpoints for Search")
"""Custom exception messages
"""

search_request_model = ApiHelper.convert_ma_schema_to_restx_model(
    API, SearchRequestSchema(), "Search"
)

similar_request_model = ApiHelper.convert_ma_schema_to_restx_model(
    API, SimilaritySearchRequestSchema(), "Search"
)

feedback_model = ApiHelper.convert_ma_schema_to_restx_model(
    API, FeedbackSchema(), "Feedback"
)

@cors_preflight("POST, OPTIONS")
@API.route("/query", methods=["POST", "OPTIONS"])
class Search(Resource):
    """Resource for search."""    
    @staticmethod
    @auth.requires_epic_search_role(["viewer", "admin"])
    @ApiHelper.swagger_decorators(API, endpoint_description="Search Query")
    @API.expect(search_request_model)
    @API.response(400, "Bad Request")
    @API.response(500, "Internal Server Error")
    def post():
        """Search"""
        current_app.logger.info("=== Search query request started ===")
        current_app.logger.info(f"Request URL: {request.url}")
        current_app.logger.info(f"Request headers: {dict(request.headers)}")
        current_app.logger.info(f"Request payload: {API.payload}")
        
        try:
            request_data = SearchRequestSchema().load(API.payload)
            current_app.logger.info("Schema validation successful")
                        
            query = request_data.get("query", None)
            project_ids = request_data.get("projectIds", None)
            document_type_ids = request_data.get("documentTypeIds", None)
            inference = request_data.get("inference", None)
            ranking = request_data.get("ranking", None)
            search_strategy = request_data.get("searchStrategy", None)
            mode = request_data.get("mode", "rag")
            user_location = request_data.get("userLocation", None)
            
            # Extract new filtering parameters
            project_status = request_data.get("projectStatus", None)
            years = request_data.get("years", None)
            
            current_app.logger.info(f"Search parameters - Query: {query[:100] if query else None}{'...' if query and len(query) > 100 else ''}")
            current_app.logger.info(f"Search parameters - Project IDs: {project_ids}")
            current_app.logger.info(f"Search parameters - Document Type IDs: {document_type_ids}")
            current_app.logger.info(f"Search parameters - Project Status: {project_status}")
            current_app.logger.info(f"Search parameters - Years: {years}")
            current_app.logger.info(f"Search parameters - Inference: {inference}")
            current_app.logger.info(f"Search parameters - Ranking: {ranking}")
            current_app.logger.info(f"Search parameters - Search Strategy: {search_strategy}")
            current_app.logger.info(f"Search parameters - Processing Mode: {mode}")
            current_app.logger.info(f"Search parameters - User Location (from browser): {user_location}")

            current_app.logger.info("Calling SearchService.get_documents_by_query")
            start_time = time.time()
            documents = SearchService.get_documents_by_query(query, project_ids, document_type_ids, inference, ranking, search_strategy, mode, user_location, project_status, years)
            end_time = time.time()
            
            current_app.logger.info(f"SearchService completed in {(end_time - start_time):.2f} seconds")
            current_app.logger.info(f"Search results: {len(documents.get('documents', [])) if isinstance(documents, dict) else 'Unknown count'} documents returned")

            # ----------------------------
            # Create feedback session in a background thread so it never blocks the response.
            # We do NOT send search_result — that's a multi-MB payload across Azure that was
            # causing 20-25s delays. Query metadata is enough for feedback tracking.
            # ----------------------------
            import uuid
            session_id = str(uuid.uuid4())
            documents["feedback_session_id"] = session_id

            app = current_app._get_current_object()
            def _create_session():
                with app.app_context():
                    try:
                        VectorSearchClient.create_feedback_session(
                            query_text=query,
                            project_ids=project_ids,
                            document_type_ids=document_type_ids,
                        )
                    except Exception as ex:
                        app.logger.warning(f"Background feedback session creation failed: {ex}")

            threading.Thread(target=_create_session, daemon=True).start()

            current_app.logger.info("=== Search query request completed successfully ===")

            return Response(json.dumps(documents), status=HTTPStatus.OK, mimetype='application/json')
        except Exception as e:
            # Log the error internally
            current_app.logger.error(f"Search error occurred: {str(e)}")
            current_app.logger.error(f"Error type: {type(e).__name__}")
            import traceback
            current_app.logger.error(f"Full traceback: {traceback.format_exc()}")
            current_app.logger.error("=== Search query request ended with error ===")
            # Return a generic error message
            error_response = {"error": "Internal server error occurred"}
            return Response(json.dumps(error_response), status=HTTPStatus.INTERNAL_SERVER_ERROR, mimetype='application/json')


@cors_preflight("POST, OPTIONS")
@API.route("/query/stream", methods=["POST", "OPTIONS"])
class StreamSearch(Resource):
    """Streaming search endpoint — returns Server-Sent Events so the AI
    summary appears word-by-word instead of after a long wait.

    SSE event types sent to the client:
      {"type": "status",  "message": "..."}   — pipeline progress update
      {"type": "token",   "content": "..."}   — one text token from the LLM
      {"type": "done",    ...full payload...} — final event (documents, metrics)
      {"type": "error",   "message": "..."}   — on failure
    """

    @staticmethod
    @auth.requires_epic_search_role(["viewer", "admin"])
    def post():
        """Streaming search with Server-Sent Events."""
        from search_api.services.search_handlers.ai_handler import AIHandler

        try:
            request_data = SearchRequestSchema().load(request.get_json(force=True) or {})
        except Exception as e:
            return Response(
                json.dumps({"error": f"Bad request: {e}"}),
                status=HTTPStatus.BAD_REQUEST,
                mimetype="application/json",
            )

        query = request_data.get("query")
        if not query:
            return Response(
                json.dumps({"error": "Missing required field: query"}),
                status=HTTPStatus.BAD_REQUEST,
                mimetype="application/json",
            )

        project_ids        = request_data.get("projectIds")
        document_type_ids  = request_data.get("documentTypeIds")
        inference          = request_data.get("inference")
        ranking            = request_data.get("ranking")
        search_strategy    = request_data.get("searchStrategy")
        user_location      = request_data.get("userLocation")
        project_status     = request_data.get("projectStatus")
        years              = request_data.get("years")

        def generate():
            try:
                for event in AIHandler.handle_stream(
                    query=query,
                    project_ids=project_ids,
                    document_type_ids=document_type_ids,
                    search_strategy=search_strategy,
                    inference=inference,
                    ranking=ranking,
                    metrics={},
                    user_location=user_location,
                    project_status=project_status,
                    years=years,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                current_app.logger.error(f"Stream generation error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # disable nginx buffering
                "Connection": "keep-alive",
            },
        )


@cors_preflight("POST, OPTIONS")
@API.route("/document-similarity", methods=["POST", "OPTIONS"])
class DocumentSimilaritySearch(Resource):
    @staticmethod
    @auth.requires_epic_search_role(["viewer", "admin"])
    @ApiHelper.swagger_decorators(API, endpoint_description="Find documents similar to a specific document using document-level embeddings (new endpoint).")
    @API.expect(similar_request_model)
    def post():
        """Document-level embedding similarity search"""
        current_app.logger.info("=== Document similarity search request started ===")
        current_app.logger.info(f"Request URL: {request.url}")
        current_app.logger.info(f"Request headers: {dict(request.headers)}")
        
        try:
            request_data = SimilaritySearchRequestSchema().load(API.payload)
            current_app.logger.info(f"Request payload: {request_data}")
            
            document_id = request_data["document_id"]
            project_ids = request_data.get("project_ids", None)
            limit = request_data.get("limit", 10)
            
            current_app.logger.info(f"Document similarity parameters - Document ID: {document_id}")
            current_app.logger.info(f"Document similarity parameters - Project IDs: {project_ids}")
            current_app.logger.info(f"Document similarity parameters - Limit: {limit}")
            
            current_app.logger.info("Calling SearchService.get_document_similarity")
            start_time = time.time()
            result = SearchService.get_document_similarity(document_id, project_ids, limit)
            end_time = time.time()
            
            current_app.logger.info(f"SearchService.get_document_similarity completed in {(end_time - start_time):.2f} seconds")
            current_app.logger.info(f"Document similarity results: {len(result.get('documents', [])) if isinstance(result, dict) else 'Unknown count'} documents returned")
            current_app.logger.info("=== Document similarity search request completed successfully ===")
            
            return Response(json.dumps(result), status=HTTPStatus.OK, mimetype='application/json')
        except Exception as e:
            current_app.logger.error(f"Document similarity POST error: {str(e)}")
            current_app.logger.error(f"Error type: {type(e).__name__}")
            import traceback
            current_app.logger.error(f"Full traceback: {traceback.format_exc()}")
            current_app.logger.error("=== Document similarity search request ended with error ===")
            return Response(json.dumps({"error": "Failed to get document similarity"}), status=HTTPStatus.INTERNAL_SERVER_ERROR, mimetype='application/json')


@cors_preflight("POST, PATCH, OPTIONS")
@API.route("/feedback", methods=["POST", "PATCH", "OPTIONS"])
class Feedback(Resource):
    """Endpoint to submit or update user feedback for search sessions."""

    @staticmethod
    @API.expect(feedback_model)
    @ApiHelper.swagger_decorators(API, endpoint_description="Submit or update search feedback")
    @API.response(400, "Bad Request")
    @API.response(200, "Feedback updated successfully")
    def post():
        """Create or update feedback for a search session."""
        try:
            data = request.get_json()
            if not data:
                return Response(
                    json.dumps({"error": "Invalid request payload"}),
                    status=HTTPStatus.BAD_REQUEST,
                    mimetype="application/json"
                )

            session_id = data.get("sessionId")
            feedback = data.get("feedback")
            comments = data.get("comments")

            if not session_id or not feedback:
                return Response(
                    json.dumps({"error": "Missing required parameters: sessionId and feedback"}),
                    status=HTTPStatus.BAD_REQUEST,
                    mimetype="application/json"
                )

            success = VectorSearchClient.update_feedback(
                session_id=session_id,
                feedback=feedback,
                comments=comments
            )

            if success:
                return Response(
                    json.dumps({"message": "Feedback updated successfully", "sessionId": session_id}),
                    status=HTTPStatus.OK,
                    mimetype="application/json"
                )
            else:
                return Response(
                    json.dumps({"error": "Failed to update feedback"}),
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    mimetype="application/json"
                )

        except Exception as e:
            current_app.logger.error(f"Error in feedback submission: {e}")
            return Response(
                json.dumps({"error": "Internal server error"}),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                mimetype="application/json"
            )

    @staticmethod
    def patch():
        """
        Explicit PATCH method to update feedback for an existing session.
        """
        try:
            data = request.get_json()
            if not data:
                return Response(
                    json.dumps({"error": "Invalid request payload"}),
                    status=HTTPStatus.BAD_REQUEST,
                    mimetype="application/json"
                )

            session_id = data.get("sessionId")
            feedback = data.get("feedback")
            summary_helpful = data.get("summaryHelpful")
            summary_accurate = data.get("summaryAccurate")
            doc_helpful = data.get("docHelpful")
            doc_accurate = data.get("docAccurate")
            summary_improvement = data.get("summaryImprovement")
            doc_improvement = data.get("docImprovement")

            if not session_id or not feedback:
                return Response(
                    json.dumps({"error": "Missing required parameters: sessionId and feedback"}),
                    status=HTTPStatus.BAD_REQUEST,
                    mimetype="application/json"
                )

            success = VectorSearchClient.update_feedback(
                session_id=session_id,
                feedback=feedback,
                summary_helpful=summary_helpful,
                summary_accurate=summary_accurate,
                doc_helpful=doc_helpful,
                doc_accurate=doc_accurate,
                summary_improvement=summary_improvement,
                doc_improvement=doc_improvement
            )

            if success:
                return Response(
                    json.dumps({"message": "Feedback updated successfully", "sessionId": session_id}),
                    status=HTTPStatus.OK,
                    mimetype="application/json"
                )
            else:
                return Response(
                    json.dumps({"error": "Failed to update feedback"}),
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    mimetype="application/json"
                )

        except Exception as e:
            current_app.logger.error(f"Error in feedback PATCH: {e}")
            return Response(
                json.dumps({"error": "Internal server error"}),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                mimetype="application/json"
            )
