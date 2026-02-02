"""Search Tool Implementations for Agentic Search.

This module provides search-related tools that wrap the VectorSearchClient
functionality with the tool framework's structured schema, validation, and metadata.

Tools:
    - vector_search: Semantic vector search for conceptual queries
    - keyword_search: Exact keyword matching for specific terms
    - document_similarity: Find documents similar to a given document

Usage:
    >>> from search_api.services.search_handlers.agent.tools.search_tools import (
    ...     register_search_tools,
    ...     create_vector_search_tool,
    ... )
    >>>
    >>> # Register all search tools
    >>> from search_api.services.search_handlers.agent.tools import ToolRegistry
    >>> registry = ToolRegistry()
    >>> registered = register_search_tools(registry)
    >>> print(f"Registered: {registered}")

    >>> # Or use individual tool
    >>> tool = create_vector_search_tool()
    >>> result = tool.execute(query="water quality impacts", limit=5)
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
# VECTOR SEARCH TOOL
# =============================================================================

def vector_search(
    query: str,
    project_ids: Optional[List[str]] = None,
    document_type_ids: Optional[List[str]] = None,
    location: Optional[str] = None,
    project_status: Optional[str] = None,
    years: Optional[List[int]] = None,
    search_strategy: str = "HYBRID_SEMANTIC_FALLBACK",
    min_score: float = -8.0,
    limit: int = 10
) -> Dict[str, Any]:
    """Perform semantic vector search using VectorSearchClient.

    This is the implementation function that performs the actual search.
    It wraps the VectorSearchClient.search() method with proper error handling.

    Args:
        query: Natural language search query
        project_ids: List of specific project IDs to search within
        document_type_ids: Filter to specific document types
        location: Geographic location to filter by
        project_status: Filter by project status
        years: List of years to filter documents by
        search_strategy: Search strategy to use
        min_score: Minimum similarity score threshold
        limit: Maximum number of results to return

    Returns:
        Dictionary with documents, document_chunks, and metadata

    Raises:
        Exception: If search fails (caught by Tool.execute wrapper)
    """
    # Import inside function to avoid circular imports and Flask context issues
    from search_api.clients.vector_search_client import VectorSearchClient

    # Build ranking configuration
    ranking = {
        "minScore": min_score,
        "topN": limit
    }

    # Execute search via VectorSearchClient
    # Returns tuple: (documents, document_chunks, api_response)
    documents, document_chunks, api_response = VectorSearchClient.search(
        query=query,
        project_ids=project_ids,
        document_type_ids=document_type_ids,
        location=location,
        project_status=project_status,
        years=years,
        search_strategy=search_strategy,
        ranking=ranking
    )

    # Check for API errors
    if api_response.get("status") == "error":
        raise RuntimeError(
            f"Vector search API error: {api_response.get('error', 'Unknown error')}"
        )

    # Build response
    return {
        "documents": documents,
        "document_chunks": document_chunks,
        "total_documents": len(documents),
        "total_chunks": len(document_chunks),
        "total_results": len(documents) + len(document_chunks),
        "search_strategy_used": search_strategy,
        "parameters": {
            "query": query,
            "project_ids": project_ids,
            "document_type_ids": document_type_ids,
            "location": location,
            "project_status": project_status,
            "years": years,
            "min_score": min_score,
            "limit": limit
        }
    }


def create_vector_search_tool() -> Tool:
    """Create the vector search tool with complete schema and validation.

    Returns:
        Configured Tool instance for vector search
    """
    return (
        create_tool("vector_search")

        # Basic info
        .description(
            "Perform semantic vector search to find documents by meaning and context. "
            "Uses embeddings to find conceptually similar content even if exact keywords differ. "
            "Best for: conceptual queries, understanding relationships, finding relevant context, "
            "broad topic exploration."
        )
        .category(ToolCategory.SEARCH)
        .cost(CostEstimate.MEDIUM)
        .returns(
            "Dictionary with 'documents' (list), 'document_chunks' (list), "
            "'total_results' (int), and search metadata"
        )

        # Required parameter
        .parameter(
            "query",
            ParameterType.STRING,
            "Natural language search query describing what you're looking for. "
            "Can be a question, phrase, or description of the topic.",
            required=True,
            examples=[
                "water quality impacts from mining",
                "First Nations consultation process",
                "environmental assessment timeline requirements",
                "wildlife habitat protection measures"
            ],
            constraints={"min_length": 3, "max_length": 500}
        )

        # Optional filter parameters
        .parameter(
            "project_ids",
            ParameterType.LIST,
            "List of specific project IDs to search within. "
            "Use when you know which projects are relevant to narrow scope.",
            required=False,
            examples=[["proj_123", "proj_456"], ["air_liquide_project"]],
            constraints={"item_type": str, "max_items": 20}
        )

        .parameter(
            "document_type_ids",
            ParameterType.LIST,
            "Filter to specific document types. "
            "Use to focus on particular types of documents.",
            required=False,
            examples=[
                ["assessment_report", "certificate"],
                ["correspondence", "meeting_notes"],
                ["public_comment"]
            ],
            constraints={"item_type": str}
        )

        .parameter(
            "location",
            ParameterType.STRING,
            "Geographic location to filter by (city, region, or area name). "
            "Use when searching for projects in specific areas.",
            required=False,
            examples=["Vancouver", "Peace River", "Northern BC", "Lower Mainland"]
        )

        .parameter(
            "project_status",
            ParameterType.STRING,
            "Filter by project status to focus on specific project phases.",
            required=False,
            default=None,
            constraints={
                "enum": ["active", "completed", "recent", "historical"]
            }
        )

        .parameter(
            "years",
            ParameterType.LIST,
            "List of years to filter documents by. "
            "Use for finding recent or historical documents.",
            required=False,
            examples=[[2023, 2024], [2020, 2021, 2022]],
            constraints={"item_type": int, "min_items": 1, "max_items": 10}
        )

        .parameter(
            "search_strategy",
            ParameterType.STRING,
            "Search strategy to use. HYBRID_SEMANTIC_FALLBACK is recommended for most queries.",
            required=False,
            default="HYBRID_SEMANTIC_FALLBACK",
            constraints={
                "enum": [
                    "HYBRID_SEMANTIC_FALLBACK",
                    "HYBRID_KEYWORD_FALLBACK",
                    "SEMANTIC_ONLY",
                    "KEYWORD_ONLY",
                    "HYBRID_PARALLEL"
                ]
            }
        )

        .parameter(
            "min_score",
            ParameterType.FLOAT,
            "Minimum similarity score threshold. Lower values return more results.",
            required=False,
            default=-8.0,
            constraints={"min": -20.0, "max": 1.0}
        )

        .parameter(
            "limit",
            ParameterType.INTEGER,
            "Maximum number of results to return.",
            required=False,
            default=10,
            constraints={"min": 1, "max": 100}
        )

        # Implementation
        .implementation(vector_search)

        # Usage examples
        .example(
            input={
                "query": "water quality impacts from mining",
                "location": "Northern BC",
                "years": [2023, 2024],
                "limit": 5
            },
            output={
                "documents": [
                    {"document_name": "Mining Impact Report 2023", "similarity_score": 0.89},
                    {"document_name": "Water Quality Assessment", "similarity_score": 0.85}
                ],
                "document_chunks": [],
                "total_results": 2,
                "search_strategy_used": "HYBRID_SEMANTIC_FALLBACK"
            }
        )

        .example(
            input={
                "query": "First Nations consultation requirements",
                "document_type_ids": ["correspondence", "meeting_notes"]
            },
            output={
                "documents": [],
                "document_chunks": [
                    {"content": "Consultation meeting with First Nations...", "score": 0.92}
                ],
                "total_results": 1
            }
        )

        # Metadata
        .metadata(
            version="1.0.0",
            tags=["search", "semantic", "vector", "primary"],
            idempotent=True
        )

        .build()
    )


# =============================================================================
# KEYWORD SEARCH TOOL
# =============================================================================

def keyword_search(
    query: str,
    project_ids: Optional[List[str]] = None,
    document_type_ids: Optional[List[str]] = None,
    exact_match: bool = False,
    case_sensitive: bool = False,
    limit: int = 10
) -> Dict[str, Any]:
    """Perform keyword-based exact text search.

    Uses KEYWORD_ONLY search strategy for fast, exact matching.
    Best for finding specific names, IDs, or quoted phrases.

    Args:
        query: Keywords or phrase to search for
        project_ids: Optional list of project IDs to filter
        document_type_ids: Optional list of document type IDs to filter
        exact_match: Whether to require exact phrase match
        case_sensitive: Whether search should be case-sensitive
        limit: Maximum number of results

    Returns:
        Dictionary with documents and search metadata
    """
    from search_api.clients.vector_search_client import VectorSearchClient

    # Build ranking configuration
    ranking = {
        "topN": limit
    }

    # Execute search with KEYWORD_ONLY strategy
    documents, document_chunks, api_response = VectorSearchClient.search(
        query=query,
        project_ids=project_ids,
        document_type_ids=document_type_ids,
        search_strategy="KEYWORD_ONLY",
        ranking=ranking
    )

    # Check for API errors
    if api_response.get("status") == "error":
        raise RuntimeError(
            f"Keyword search API error: {api_response.get('error', 'Unknown error')}"
        )

    # Apply exact match filter if requested
    if exact_match:
        query_lower = query.lower()

        # Filter documents
        filtered_documents = []
        for doc in documents:
            content = doc.get("content", "") or doc.get("document_name", "")
            if case_sensitive:
                if query in content:
                    filtered_documents.append(doc)
            else:
                if query_lower in content.lower():
                    filtered_documents.append(doc)
        documents = filtered_documents

        # Filter chunks
        filtered_chunks = []
        for chunk in document_chunks:
            content = chunk.get("content", "") or ""
            if case_sensitive:
                if query in content:
                    filtered_chunks.append(chunk)
            else:
                if query_lower in content.lower():
                    filtered_chunks.append(chunk)
        document_chunks = filtered_chunks

    return {
        "documents": documents,
        "document_chunks": document_chunks,
        "total_documents": len(documents),
        "total_chunks": len(document_chunks),
        "total_results": len(documents) + len(document_chunks),
        "exact_match_used": exact_match,
        "case_sensitive": case_sensitive,
        "parameters": {
            "query": query,
            "project_ids": project_ids,
            "document_type_ids": document_type_ids,
            "limit": limit
        }
    }


def create_keyword_search_tool() -> Tool:
    """Create the keyword search tool.

    Returns:
        Configured Tool instance for keyword search
    """
    return (
        create_tool("keyword_search")

        .description(
            "Exact keyword search for specific terms, names, identifiers, or phrases. "
            "Faster than vector search but requires exact matches. "
            "Best for: finding specific project names, document IDs, company names, "
            "exact quotes, technical terms, or reference numbers."
        )
        .category(ToolCategory.SEARCH)
        .cost(CostEstimate.LOW)
        .returns(
            "Dictionary with 'documents' list, 'document_chunks' list, "
            "'total_results' count, and filter metadata"
        )

        # Parameters
        .parameter(
            "query",
            ParameterType.STRING,
            "Exact keywords or phrase to search for. "
            "This will be matched literally against document content.",
            required=True,
            examples=[
                "Air Liquide",
                "EAC-2023-0142",
                "Trans Mountain",
                "environmental impact statement"
            ],
            constraints={"min_length": 1, "max_length": 200}
        )

        .parameter(
            "project_ids",
            ParameterType.LIST,
            "List of project IDs to search within.",
            required=False,
            constraints={"item_type": str, "max_items": 20}
        )

        .parameter(
            "document_type_ids",
            ParameterType.LIST,
            "List of document type IDs to filter by.",
            required=False,
            constraints={"item_type": str}
        )

        .parameter(
            "exact_match",
            ParameterType.BOOLEAN,
            "Whether to require exact phrase match vs any keyword match. "
            "When True, the entire query must appear as-is in the document.",
            required=False,
            default=False
        )

        .parameter(
            "case_sensitive",
            ParameterType.BOOLEAN,
            "Whether search should be case-sensitive. "
            "Usually False for better recall.",
            required=False,
            default=False
        )

        .parameter(
            "limit",
            ParameterType.INTEGER,
            "Maximum number of results to return.",
            required=False,
            default=10,
            constraints={"min": 1, "max": 100}
        )

        .implementation(keyword_search)

        .example(
            input={"query": "Air Liquide", "exact_match": True},
            output={
                "documents": [
                    {"document_name": "Air Liquide Project Assessment"},
                    {"document_name": "Air Liquide Certificate of Compliance"}
                ],
                "total_results": 2,
                "exact_match_used": True
            }
        )

        .example(
            input={"query": "EAC-2023", "limit": 5},
            output={
                "documents": [
                    {"document_name": "EAC-2023-0142 Decision"},
                    {"document_name": "EAC-2023-0089 Report"}
                ],
                "total_results": 2
            }
        )

        .metadata(
            version="1.0.0",
            tags=["search", "keyword", "exact", "fast"],
            idempotent=True
        )

        .build()
    )


# =============================================================================
# DOCUMENT SIMILARITY TOOL
# =============================================================================

def document_similarity(
    document_id: str,
    project_ids: Optional[List[str]] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """Find documents similar to a given document.

    Uses document-level embeddings to find semantically similar documents.
    Useful for finding related content or expanding search results.

    Args:
        document_id: The document ID to find similar documents for
        project_ids: Optional list of project IDs to filter results
        limit: Maximum number of similar documents to return

    Returns:
        Dictionary with similar documents and similarity scores
    """
    from search_api.clients.vector_search_client import VectorSearchClient

    # Execute similarity search
    result = VectorSearchClient.document_similarity_search(
        document_id=document_id,
        project_ids=project_ids,
        limit=limit
    )

    # Handle empty or error results
    if not result:
        return {
            "similar_documents": [],
            "total_results": 0,
            "source_document_id": document_id,
            "parameters": {
                "document_id": document_id,
                "project_ids": project_ids,
                "limit": limit
            }
        }

    # Extract similar documents from response
    similar_docs = result.get("similar_documents", result.get("documents", []))

    return {
        "similar_documents": similar_docs,
        "total_results": len(similar_docs),
        "source_document_id": document_id,
        "parameters": {
            "document_id": document_id,
            "project_ids": project_ids,
            "limit": limit
        }
    }


def create_document_similarity_tool() -> Tool:
    """Create the document similarity search tool.

    Returns:
        Configured Tool instance for document similarity search
    """
    return (
        create_tool("document_similarity")

        .description(
            "Find documents similar to a given document using document-level embeddings. "
            "Useful for: expanding search results with related content, finding similar reports, "
            "discovering related projects, following document chains."
        )
        .category(ToolCategory.SEARCH)
        .cost(CostEstimate.MEDIUM)
        .returns(
            "Dictionary with 'similar_documents' list containing documents "
            "ordered by similarity, and 'total_results' count"
        )

        # Parameters
        .parameter(
            "document_id",
            ParameterType.STRING,
            "The document ID to find similar documents for. "
            "This ID should come from a previous search result.",
            required=True,
            examples=["doc_12345", "assessment_report_2023_001"]
        )

        .parameter(
            "project_ids",
            ParameterType.LIST,
            "Optional list of project IDs to filter similar documents. "
            "Use to find similar documents within specific projects.",
            required=False,
            constraints={"item_type": str, "max_items": 20}
        )

        .parameter(
            "limit",
            ParameterType.INTEGER,
            "Maximum number of similar documents to return.",
            required=False,
            default=10,
            constraints={"min": 1, "max": 50}
        )

        .implementation(document_similarity)

        .example(
            input={
                "document_id": "doc_assessment_12345",
                "limit": 5
            },
            output={
                "similar_documents": [
                    {"document_id": "doc_67890", "document_name": "Related Assessment", "similarity": 0.92},
                    {"document_id": "doc_11111", "document_name": "Similar Report", "similarity": 0.88}
                ],
                "total_results": 2,
                "source_document_id": "doc_assessment_12345"
            }
        )

        .example(
            input={
                "document_id": "doc_mining_report",
                "project_ids": ["proj_copper_creek"],
                "limit": 3
            },
            output={
                "similar_documents": [
                    {"document_id": "doc_related", "similarity": 0.85}
                ],
                "total_results": 1
            }
        )

        .metadata(
            version="1.0.0",
            tags=["search", "similarity", "documents", "related"],
            idempotent=True
        )

        .build()
    )


# =============================================================================
# REGISTRATION FUNCTIONS
# =============================================================================

def register_search_tools(registry: ToolRegistry) -> List[str]:
    """Register all search-related tools with the registry.

    Args:
        registry: ToolRegistry instance to register tools with

    Returns:
        List of registered tool names
    """
    tools = [
        create_vector_search_tool(),
        create_keyword_search_tool(),
        create_document_similarity_tool(),
    ]

    registered = []
    for tool in tools:
        try:
            registry.register(tool)
            registered.append(tool.name)
            logger.info(f"Registered search tool: {tool.name}")
        except ValueError as e:
            logger.warning(f"Failed to register tool {tool.name}: {e}")

    logger.info(f"Registered {len(registered)} search tools: {', '.join(registered)}")
    return registered


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Tool creation functions
    "create_vector_search_tool",
    "create_keyword_search_tool",
    "create_document_similarity_tool",

    # Implementation functions (for direct use if needed)
    "vector_search",
    "keyword_search",
    "document_similarity",

    # Registration
    "register_search_tools",
]


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    """Test search tools when running module directly."""
    import sys

    logging.basicConfig(level=logging.INFO)

    print("Search Tools Module")
    print("=" * 50)

    # Create tools and show info
    tools = [
        create_vector_search_tool(),
        create_keyword_search_tool(),
        create_document_similarity_tool(),
    ]

    for tool in tools:
        print(f"\nTool: {tool.name}")
        print(f"  Category: {tool.category.value}")
        print(f"  Cost: {tool.cost_estimate.value}")
        print(f"  Parameters: {len(tool.parameters)}")
        print(f"  Required params: {[p.name for p in tool.parameters.values() if p.required]}")
        print(f"  Optional params: {[p.name for p in tool.parameters.values() if not p.required]}")

    # Test registry
    print("\n" + "=" * 50)
    print("Testing Registry Registration")

    from .tool_registry import ToolRegistry

    registry = ToolRegistry()
    registered = register_search_tools(registry)

    print(f"\nRegistered tools: {registered}")
    print(f"Registry statistics: {registry.get_statistics()}")

    # Show LLM descriptions
    print("\n" + "=" * 50)
    print("LLM Tool Descriptions")
    print(registry.get_tool_descriptions(
        category=ToolCategory.SEARCH,
        include_examples=True
    ))

    sys.exit(0)
