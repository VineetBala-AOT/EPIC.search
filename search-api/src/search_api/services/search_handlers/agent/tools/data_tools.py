"""Data Tool Implementations for Agentic Search.

This module provides data retrieval tools that wrap the VectorSearchClient
discovery operations with the tool framework's structured schema and validation.

Tools:
    - get_projects_list: Get list of all available projects
    - get_project_details: Get detailed information about a specific project
    - get_document_types: Get available document types
    - get_document_type_details: Get details for a specific document type
    - get_search_strategies: Get available search strategies

Usage:
    >>> from search_api.services.search_handlers.agent.tools.data_tools import (
    ...     register_data_tools,
    ...     create_get_projects_list_tool,
    ... )
    >>>
    >>> # Register all data tools
    >>> from search_api.services.search_handlers.agent.tools import ToolRegistry
    >>> registry = ToolRegistry()
    >>> registered = register_data_tools(registry)
    >>> print(f"Registered: {registered}")
"""

import logging
from typing import Dict, Any, List, Optional

from .tool_registry import (
    create_tool,
    Tool,
    ToolRegistry,
    ToolCategory,
    CostEstimate,
    ParameterType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# GET PROJECTS LIST TOOL
# =============================================================================

def get_projects_list(
    filter_by: Optional[str] = None,
    include_metadata: bool = False
) -> Dict[str, Any]:
    """Get list of all available projects.

    Retrieves all projects from the system with optional filtering.
    Results are cached for 24 hours by the VectorSearchClient.

    Args:
        filter_by: Optional text filter to search in project names
        include_metadata: Whether to include full project metadata

    Returns:
        Dictionary with projects list and total count
    """
    from search_api.clients.vector_search_client import VectorSearchClient

    # Get projects from client (cached)
    projects = VectorSearchClient.get_projects_list(include_metadata=include_metadata)

    # Apply text filter if provided
    if filter_by and projects:
        filter_lower = filter_by.lower()
        projects = [
            p for p in projects
            if filter_lower in p.get("project_name", "").lower() or
               filter_lower in p.get("project_id", "").lower()
        ]

    return {
        "projects": projects,
        "total": len(projects),
        "filtered": filter_by is not None,
        "filter_text": filter_by
    }


def create_get_projects_list_tool() -> Tool:
    """Create the get_projects_list tool.

    Returns:
        Configured Tool instance for getting projects list
    """
    return (
        create_tool("get_projects_list")

        .description(
            "Get list of all available projects in the system. "
            "Returns project names and IDs. Use this to discover what projects exist, "
            "find project IDs for filtering searches, or understand the scope of data. "
            "Best for: discovering projects, finding project IDs, understanding available data."
        )
        .category(ToolCategory.DATA)
        .cost(CostEstimate.LOW)  # Cached data
        .returns(
            "Dictionary with 'projects' (list of {project_id, project_name}), "
            "'total' count, and filter info"
        )

        .parameter(
            "filter_by",
            ParameterType.STRING,
            "Optional text filter to search in project names and IDs. "
            "Use to narrow down results when looking for specific projects.",
            required=False,
            examples=["mining", "LNG", "pipeline", "Air Liquide"]
        )

        .parameter(
            "include_metadata",
            ParameterType.BOOLEAN,
            "Whether to include full project metadata beyond name and ID.",
            required=False,
            default=False
        )

        .implementation(get_projects_list)

        .example(
            input={},
            output={
                "projects": [
                    {"project_id": "proj_123", "project_name": "Red Mountain Mining"},
                    {"project_id": "proj_456", "project_name": "Copper Creek Mine"},
                    {"project_id": "proj_789", "project_name": "Air Liquide Facility"}
                ],
                "total": 3,
                "filtered": False
            }
        )

        .example(
            input={"filter_by": "mining"},
            output={
                "projects": [
                    {"project_id": "proj_123", "project_name": "Red Mountain Mining"},
                    {"project_id": "proj_456", "project_name": "Copper Creek Mine"}
                ],
                "total": 2,
                "filtered": True,
                "filter_text": "mining"
            }
        )

        .metadata(
            version="1.0.0",
            tags=["data", "projects", "discovery", "cached"],
            idempotent=True
        )

        .build()
    )


# =============================================================================
# GET PROJECT DETAILS TOOL
# =============================================================================

def get_project_details(
    project_id: str
) -> Dict[str, Any]:
    """Get detailed information about a specific project.

    Args:
        project_id: The project ID to get details for

    Returns:
        Dictionary with project details or error info
    """
    from search_api.clients.vector_search_client import VectorSearchClient

    # Get all projects with metadata
    projects = VectorSearchClient.get_projects_list(include_metadata=True)

    # Find the specific project
    project = None
    for p in projects:
        if p.get("project_id") == project_id:
            project = p
            break

    if project:
        return {
            "found": True,
            "project": project,
            "project_id": project_id
        }
    else:
        return {
            "found": False,
            "project": None,
            "project_id": project_id,
            "error": f"Project not found: {project_id}"
        }


def create_get_project_details_tool() -> Tool:
    """Create the get_project_details tool.

    Returns:
        Configured Tool instance for getting project details
    """
    return (
        create_tool("get_project_details")

        .description(
            "Get detailed information about a specific project by ID. "
            "Returns full project metadata including name, status, and other attributes. "
            "Use after get_projects_list to get more info about a specific project."
        )
        .category(ToolCategory.DATA)
        .cost(CostEstimate.LOW)  # Uses cached data
        .returns(
            "Dictionary with 'found' boolean, 'project' details if found, "
            "or 'error' message if not found"
        )

        .parameter(
            "project_id",
            ParameterType.STRING,
            "The project ID to get details for. "
            "Get valid IDs from get_projects_list first.",
            required=True,
            examples=["proj_123", "air_liquide_project"]
        )

        .implementation(get_project_details)

        .example(
            input={"project_id": "proj_123"},
            output={
                "found": True,
                "project": {
                    "project_id": "proj_123",
                    "project_name": "Red Mountain Mining",
                    "status": "active",
                    "region": "Northern BC"
                },
                "project_id": "proj_123"
            }
        )

        .example(
            input={"project_id": "invalid_id"},
            output={
                "found": False,
                "project": None,
                "project_id": "invalid_id",
                "error": "Project not found: invalid_id"
            }
        )

        .metadata(
            version="1.0.0",
            tags=["data", "projects", "details", "cached"],
            idempotent=True
        )

        .build()
    )


# =============================================================================
# GET DOCUMENT TYPES TOOL
# =============================================================================

def get_document_types(
    filter_by: Optional[str] = None
) -> Dict[str, Any]:
    """Get available document types.

    Retrieves all document types from the system with their names and aliases.
    Results are cached for 24 hours.

    Args:
        filter_by: Optional text filter to search in type names and aliases

    Returns:
        Dictionary with document types list and total count
    """
    from search_api.clients.vector_search_client import VectorSearchClient

    # Get document types from client (cached)
    doc_types = VectorSearchClient.get_document_types()

    # Apply text filter if provided
    if filter_by and doc_types:
        filter_lower = filter_by.lower()
        filtered = []
        for dt in doc_types:
            name = dt.get("document_type_name", "").lower()
            type_id = dt.get("document_type_id", "").lower()
            aliases = [a.lower() for a in dt.get("aliases", [])]

            if (filter_lower in name or
                filter_lower in type_id or
                any(filter_lower in alias for alias in aliases)):
                filtered.append(dt)
        doc_types = filtered

    return {
        "document_types": doc_types,
        "total": len(doc_types),
        "filtered": filter_by is not None,
        "filter_text": filter_by
    }


def create_get_document_types_tool() -> Tool:
    """Create the get_document_types tool.

    Returns:
        Configured Tool instance for getting document types
    """
    return (
        create_tool("get_document_types")

        .description(
            "Get all available document types in the system. "
            "Returns type IDs, names, and aliases. Use to understand what document types "
            "exist and find the correct type IDs for filtering searches. "
            "Best for: understanding document categories, finding type IDs for search filters."
        )
        .category(ToolCategory.DATA)
        .cost(CostEstimate.LOW)  # Cached data
        .returns(
            "Dictionary with 'document_types' (list of {document_type_id, document_type_name, aliases}), "
            "'total' count, and filter info"
        )

        .parameter(
            "filter_by",
            ParameterType.STRING,
            "Optional text filter to search in type names and aliases. "
            "Use to find specific document types.",
            required=False,
            examples=["report", "certificate", "letter", "assessment"]
        )

        .implementation(get_document_types)

        .example(
            input={},
            output={
                "document_types": [
                    {
                        "document_type_id": "assessment_report",
                        "document_type_name": "Assessment Report",
                        "aliases": ["EA report", "environmental assessment"]
                    },
                    {
                        "document_type_id": "certificate",
                        "document_type_name": "Certificate",
                        "aliases": ["EAC", "environmental certificate"]
                    }
                ],
                "total": 2,
                "filtered": False
            }
        )

        .example(
            input={"filter_by": "report"},
            output={
                "document_types": [
                    {
                        "document_type_id": "assessment_report",
                        "document_type_name": "Assessment Report",
                        "aliases": ["EA report"]
                    }
                ],
                "total": 1,
                "filtered": True,
                "filter_text": "report"
            }
        )

        .metadata(
            version="1.0.0",
            tags=["data", "document_types", "discovery", "cached"],
            idempotent=True
        )

        .build()
    )


# =============================================================================
# GET DOCUMENT TYPE DETAILS TOOL
# =============================================================================

def get_document_type_details(
    type_id: str
) -> Dict[str, Any]:
    """Get detailed information about a specific document type.

    Args:
        type_id: The document type ID to get details for

    Returns:
        Dictionary with document type details
    """
    from search_api.clients.vector_search_client import VectorSearchClient

    # Get detailed info for the specific type
    details = VectorSearchClient.get_document_type_details(type_id)

    if details:
        return {
            "found": True,
            "document_type": details,
            "type_id": type_id
        }
    else:
        return {
            "found": False,
            "document_type": None,
            "type_id": type_id,
            "error": f"Document type not found: {type_id}"
        }


def create_get_document_type_details_tool() -> Tool:
    """Create the get_document_type_details tool.

    Returns:
        Configured Tool instance for getting document type details
    """
    return (
        create_tool("get_document_type_details")

        .description(
            "Get detailed information about a specific document type by ID. "
            "Returns full details including name, aliases, and associated act. "
            "Use after get_document_types to get more info about a specific type."
        )
        .category(ToolCategory.DATA)
        .cost(CostEstimate.LOW)  # Cached data
        .returns(
            "Dictionary with 'found' boolean, 'document_type' details if found, "
            "or 'error' message if not found"
        )

        .parameter(
            "type_id",
            ParameterType.STRING,
            "The document type ID to get details for. "
            "Get valid IDs from get_document_types first.",
            required=True,
            examples=["assessment_report", "certificate", "correspondence"]
        )

        .implementation(get_document_type_details)

        .example(
            input={"type_id": "assessment_report"},
            output={
                "found": True,
                "document_type": {
                    "document_type_id": "assessment_report",
                    "document_type_name": "Assessment Report",
                    "aliases": ["EA report", "environmental assessment"],
                    "act": "Environmental Assessment Act"
                },
                "type_id": "assessment_report"
            }
        )

        .metadata(
            version="1.0.0",
            tags=["data", "document_types", "details", "cached"],
            idempotent=True
        )

        .build()
    )


# =============================================================================
# GET SEARCH STRATEGIES TOOL
# =============================================================================

def get_search_strategies() -> Dict[str, Any]:
    """Get available search strategies.

    Retrieves all search strategies and their descriptions.
    Useful for understanding which strategy to use for different queries.

    Returns:
        Dictionary with search strategies and descriptions
    """
    from search_api.clients.vector_search_client import VectorSearchClient

    # Get strategies from client
    strategies = VectorSearchClient.get_search_strategies()

    # Format the response
    if strategies:
        return {
            "strategies": strategies.get("strategies", strategies),
            "total": len(strategies.get("strategies", strategies)),
            "available": True
        }
    else:
        # Return default known strategies if API call fails
        default_strategies = {
            "HYBRID_SEMANTIC_FALLBACK": {
                "description": "Try semantic search first, fall back to keyword if needed",
                "recommended_for": "Most queries, conceptual searches"
            },
            "HYBRID_KEYWORD_FALLBACK": {
                "description": "Try keyword search first, fall back to semantic if needed",
                "recommended_for": "Specific term searches"
            },
            "SEMANTIC_ONLY": {
                "description": "Pure semantic/vector search",
                "recommended_for": "Conceptual queries, finding related content"
            },
            "KEYWORD_ONLY": {
                "description": "Pure keyword matching",
                "recommended_for": "Exact matches, IDs, specific terms"
            },
            "HYBRID_PARALLEL": {
                "description": "Run both semantic and keyword in parallel, merge results",
                "recommended_for": "Comprehensive searches"
            }
        }
        return {
            "strategies": default_strategies,
            "total": len(default_strategies),
            "available": True,
            "source": "default"
        }


def create_get_search_strategies_tool() -> Tool:
    """Create the get_search_strategies tool.

    Returns:
        Configured Tool instance for getting search strategies
    """
    return (
        create_tool("get_search_strategies")

        .description(
            "Get available search strategies and their descriptions. "
            "Helps understand which strategy to use for different types of queries. "
            "Best for: understanding search options, optimizing search approach."
        )
        .category(ToolCategory.DATA)
        .cost(CostEstimate.VERY_LOW)  # May return cached defaults
        .returns(
            "Dictionary with 'strategies' (dict of strategy names to descriptions), "
            "'total' count, and availability info"
        )

        # No parameters required
        .implementation(get_search_strategies)

        .example(
            input={},
            output={
                "strategies": {
                    "HYBRID_SEMANTIC_FALLBACK": {
                        "description": "Try semantic search first, fall back to keyword",
                        "recommended_for": "Most queries"
                    },
                    "KEYWORD_ONLY": {
                        "description": "Pure keyword matching",
                        "recommended_for": "Exact matches"
                    }
                },
                "total": 2,
                "available": True
            }
        )

        .metadata(
            version="1.0.0",
            tags=["data", "strategies", "discovery"],
            idempotent=True
        )

        .build()
    )


# =============================================================================
# REGISTRATION FUNCTIONS
# =============================================================================

def register_data_tools(registry: ToolRegistry) -> List[str]:
    """Register all data-related tools with the registry.

    Args:
        registry: ToolRegistry instance to register tools with

    Returns:
        List of registered tool names
    """
    tools = [
        create_get_projects_list_tool(),
        create_get_project_details_tool(),
        create_get_document_types_tool(),
        create_get_document_type_details_tool(),
        create_get_search_strategies_tool(),
    ]

    registered = []
    for tool in tools:
        try:
            registry.register(tool)
            registered.append(tool.name)
            logger.info(f"Registered data tool: {tool.name}")
        except ValueError as e:
            logger.warning(f"Failed to register tool {tool.name}: {e}")

    logger.info(f"Registered {len(registered)} data tools: {', '.join(registered)}")
    return registered


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Tool creation functions
    "create_get_projects_list_tool",
    "create_get_project_details_tool",
    "create_get_document_types_tool",
    "create_get_document_type_details_tool",
    "create_get_search_strategies_tool",

    # Implementation functions
    "get_projects_list",
    "get_project_details",
    "get_document_types",
    "get_document_type_details",
    "get_search_strategies",

    # Registration
    "register_data_tools",
]


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    """Test data tools when running module directly."""
    import sys

    logging.basicConfig(level=logging.INFO)

    print("Data Tools Module")
    print("=" * 50)

    # Create tools and show info
    tools = [
        create_get_projects_list_tool(),
        create_get_project_details_tool(),
        create_get_document_types_tool(),
        create_get_document_type_details_tool(),
        create_get_search_strategies_tool(),
    ]

    for tool in tools:
        print(f"\nTool: {tool.name}")
        print(f"  Category: {tool.category.value}")
        print(f"  Cost: {tool.cost_estimate.value}")
        print(f"  Parameters: {len(tool.parameters)}")
        print(f"  Required: {[p.name for p in tool.parameters.values() if p.required]}")

    # Test registry
    print("\n" + "=" * 50)
    print("Testing Registry Registration")

    from .tool_registry import ToolRegistry

    registry = ToolRegistry()
    registered = register_data_tools(registry)

    print(f"\nRegistered tools: {registered}")
    print(f"Registry statistics: {registry.get_statistics()}")

    # Show LLM descriptions
    print("\n" + "=" * 50)
    print("LLM Tool Descriptions")
    print(registry.get_tool_descriptions(
        category=ToolCategory.DATA,
        include_examples=True
    ))

    sys.exit(0)
