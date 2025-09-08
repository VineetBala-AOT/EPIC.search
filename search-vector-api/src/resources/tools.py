"""Tools API endpoints for MCP utilities and data access.

This module provides REST API endpoints for external tools and systems
that need access to basic project and document type information without
the overhead of full processing statistics.

The endpoints include:
1. Project listings for external tool integration
2. Document type lookups and metadata access
3. Simple data access utilities for MCP systems
"""

from http import HTTPStatus
from flask_restx import Namespace, Resource
from flask import Response
from marshmallow import EXCLUDE, Schema, fields
import json

from services.tools_service import ToolsService
from .apihelper import Api as ApiHelper


class DocumentTypeRequestSchema(Schema):
    """Schema for validating document type lookup requests.
    
    Defines the structure and validation rules for document type requests,
    which retrieve specific document type information by ID.
    
    Attributes:
        typeId: The required document type ID to look up
    """

    class Meta:  # pylint: disable=too-few-public-methods
        """Exclude unknown fields in the deserialized output."""
        unknown = EXCLUDE

    typeId = fields.Str(data_key="typeId", required=True, 
                       metadata={"description": "Document type ID to look up"})


API = Namespace("tools", description="Endpoints for MCP tools and utilities")

document_type_request_model = ApiHelper.convert_ma_schema_to_restx_model(
    API, DocumentTypeRequestSchema(), "Document Type Request"
)


@API.route("/projects", methods=["GET", "OPTIONS"])
class ProjectsList(Resource):
    """Projects list endpoint for external tools.
    
    Provides a simple list of all projects with basic information (ID and name)
    without processing statistics overhead. Designed for external tools that
    need project listings for integration purposes.
    """

    @staticmethod
    @ApiHelper.swagger_decorators(API, endpoint_description="Get simple list of all projects")
    @API.response(400, "Bad Request")
    @API.response(200, "Projects list retrieved successfully")
    def get():
        """Get a simple list of all projects.
        
        Retrieves a lightweight list of all projects containing only
        basic information (project ID and name). This endpoint is optimized
        for external tools that need project listings without the overhead
        of processing statistics.
        
        Returns:
            Response: JSON containing projects list:
                {
                    "projects": [
                        {
                            "project_id": "uuid-string",
                            "project_name": "Project Name"
                        },
                        ...
                    ],
                    "total_projects": 5
                }
        """
        try:
            result = ToolsService.get_projects_list()
            return Response(
                response=json.dumps(result),
                status=HTTPStatus.OK,
                mimetype="application/json"
            )
        except Exception as e:
            error_response = {
                "error": "Failed to retrieve projects list",
                "details": str(e)
            }
            return Response(
                response=json.dumps(error_response),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                mimetype="application/json"
            )


@API.route("/document-types", methods=["GET", "OPTIONS"])
class DocumentTypesList(Resource):
    """Document types list endpoint.
    
    Provides comprehensive document type information including names, IDs,
    and aliases for both 2002 Act and 2018 Act terms. Designed for external
    tools that need document type lookups and inference capabilities.
    """

    @staticmethod
    @ApiHelper.swagger_decorators(API, endpoint_description="Get all document types with metadata")
    @API.response(400, "Bad Request")
    @API.response(200, "Document types retrieved successfully")
    def get():
        """Get all document types with their metadata.
        
        Retrieves comprehensive document type information including:
        - Complete ID to name mappings
        - Alias lists for inference and lookup
        - Act classification (2002 vs 2018 terms) for each document type
        - Breakdown by Act (2002 vs 2018 terms)
        - Total counts and statistics
        
        Returns:
            Response: JSON containing document types:
                {
                    "document_types": {
                        "type_id": {
                            "name": "Human Readable Name",
                            "aliases": ["alias1", "alias2", ...],
                            "act": "2002_act_terms" | "2018_act_terms"
                        },
                        ...
                    },
                    "lookup_only": {
                        "type_id": "Human Readable Name",
                        ...
                    },
                    "total_types": 42,
                    "act_breakdown": {
                        "2002_act_terms": 20,
                        "2018_act_terms": 22
                    }
                }
        """
        try:
            result = ToolsService.get_document_types()
            return Response(
                response=json.dumps(result),
                status=HTTPStatus.OK,
                mimetype="application/json"
            )
        except Exception as e:
            error_response = {
                "error": "Failed to retrieve document types",
                "details": str(e)
            }
            return Response(
                response=json.dumps(error_response),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                mimetype="application/json"
            )


@API.route("/document-types/<type_id>", methods=["GET", "OPTIONS"])
class DocumentTypeDetails(Resource):
    """Document type details endpoint.
    
    Provides detailed information for a specific document type including
    name, aliases, and metadata. Designed for targeted lookups by external
    tools that need specific document type information.
    """

    @staticmethod
    @ApiHelper.swagger_decorators(API, endpoint_description="Get detailed information for a specific document type")
    @API.response(400, "Bad Request")
    @API.response(404, "Document type not found")
    @API.response(200, "Document type details retrieved successfully")
    def get(type_id):
        """Get detailed information for a specific document type.
        
        Retrieves comprehensive information for a single document type
        including its human-readable name, aliases for inference, Act
        classification, and metadata for integration purposes.
        
        Args:
            type_id (str): The document type ID to look up
        
        Returns:
            Response: JSON containing document type details:
                {
                    "document_type": {
                        "id": "type_id",
                        "name": "Human Readable Name",
                        "aliases": ["alias1", "alias2", ...],
                        "act": "2002_act_terms" | "2018_act_terms"
                    }
                }
        """
        try:
            result = ToolsService.get_document_type_by_id(type_id)
            
            # Check if document type was found
            if result.get("document_type") is None:
                return Response(
                    response=json.dumps(result),
                    status=HTTPStatus.NOT_FOUND,
                    mimetype="application/json"
                )
            
            return Response(
                response=json.dumps(result),
                status=HTTPStatus.OK,
                mimetype="application/json"
            )
        except Exception as e:
            error_response = {
                "error": f"Failed to retrieve document type {type_id}",
                "details": str(e)
            }
            return Response(
                response=json.dumps(error_response),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                mimetype="application/json"
            )


@API.route("/search-strategies")
class SearchStrategiesResource(Resource):
    @API.doc("Get available search strategies")
    @API.response(200, "Success", model=None)
    @API.response(500, "Internal Server Error")
    def get(self):
        """
        Get all available search strategies supported by the API.
        
        Returns the different search approaches available, including semantic,
        keyword, hybrid, and metadata search options with their descriptions
        and capabilities.
        """
        try:
            strategies = ToolsService.get_search_strategies()
            return Response(
                response=json.dumps(strategies),
                status=HTTPStatus.OK,
                mimetype="application/json"
            )
        except Exception as e:
            error_response = {
                "error": "Failed to retrieve search strategies",
                "details": str(e)
            }
            return Response(
                response=json.dumps(error_response),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                mimetype="application/json"
            )


@API.route("/inference-options")
class InferenceOptionsResource(Resource):
    @API.doc("Get available inference options")
    @API.response(200, "Success", model=None)
    @API.response(500, "Internal Server Error")
    def get(self):
        """
        Get all available inference options for document classification.
        
        Returns the different inference services available for automatic
        document type classification and project inference, including
        their capabilities and output formats.
        """
        try:
            options = ToolsService.get_inference_options()
            return Response(
                response=json.dumps(options),
                status=HTTPStatus.OK,
                mimetype="application/json"
            )
        except Exception as e:
            error_response = {
                "error": "Failed to retrieve inference options",
                "details": str(e)
            }
            return Response(
                response=json.dumps(error_response),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                mimetype="application/json"
            )


@API.route("/api-capabilities")
class ApiCapabilitiesResource(Resource):
    @API.doc("Get API capabilities and endpoints")
    @API.response(200, "Success", model=None)
    @API.response(500, "Internal Server Error")
    def get(self):
        """
        Get comprehensive information about API capabilities.
        
        Returns details about all available endpoints, their methods,
        parameters, and capabilities. Useful for MCP tools and external
        integrations to discover API functionality.
        """
        try:
            capabilities = ToolsService.get_api_capabilities()
            return Response(
                response=json.dumps(capabilities),
                status=HTTPStatus.OK,
                mimetype="application/json"
            )
        except Exception as e:
            error_response = {
                "error": "Failed to retrieve API capabilities",
                "details": str(e)
            }
            return Response(
                response=json.dumps(error_response),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                mimetype="application/json"
            )
