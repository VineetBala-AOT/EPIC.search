"""
Tool Framework for Agentic Search

This module provides a structured framework for defining, registering, and executing
tools that agents can use. Tools are first-class objects with schemas, validation,
and metadata.

The framework consists of:
- Core types and enums (ParameterType, ToolCategory, CostEstimate)
- Schema definitions (ToolParameter, Tool, ToolMetadata)
- Registry for managing tools (ToolRegistry)
- Builder pattern for easy tool creation (ToolBuilder, create_tool)
- Tool implementations (SearchTools, DataTools, UtilityTools)

Quick Start:
    >>> from search_api.services.search_handlers.agent.tools import (
    ...     create_tool, ToolRegistry, ParameterType, ToolCategory, CostEstimate
    ... )
    >>>
    >>> # Create a tool using builder pattern
    >>> tool = (create_tool("my_tool")
    ...         .description("My custom tool")
    ...         .category(ToolCategory.UTILITY)
    ...         .cost(CostEstimate.LOW)
    ...         .parameter("input", ParameterType.STRING, "Input text", required=True)
    ...         .returns("Processed result")
    ...         .implementation(lambda input: {"result": input.upper()})
    ...         .build())
    >>>
    >>> # Register and use
    >>> registry = ToolRegistry()
    >>> registry.register(tool)
    >>> result = registry.execute_tool("my_tool", {"input": "hello"})
    >>> print(result)
    {'success': True, 'result': {'result': 'HELLO'}, ...}

See Also:
    - TOOL_FRAMEWORK_API.md: Complete API documentation
    - TOOL_EXAMPLES.md: Working examples of tool implementations
"""

# Core enums and types
from .tool_registry import (
    ParameterType,
    ToolCategory,
    CostEstimate,
)

# Schema classes
from .tool_registry import (
    ToolParameter,
    ToolMetadata,
    Tool,
    ToolRegistry,
)

# Builder pattern
from .tool_registry import (
    ToolBuilder,
    create_tool,
)

# Re-export all public symbols
__all__ = [
    # Enums
    "ParameterType",
    "ToolCategory",
    "CostEstimate",

    # Schema classes
    "ToolParameter",
    "ToolMetadata",
    "Tool",
    "ToolRegistry",

    # Builder
    "ToolBuilder",
    "create_tool",

    # Factory functions
    "initialize_tool_registry",
    "get_default_registry",
    "reset_default_registry",
    "execute_tool",

    # Search tools
    "register_search_tools",
    "create_vector_search_tool",
    "create_keyword_search_tool",
    "create_document_similarity_tool",
    "vector_search",
    "keyword_search",
    "document_similarity",

    # Data tools
    "register_data_tools",
    "create_get_projects_list_tool",
    "create_get_project_details_tool",
    "create_get_document_types_tool",
    "create_get_document_type_details_tool",
    "create_get_search_strategies_tool",
    "get_projects_list",
    "get_project_details",
    "get_document_types",
    "get_document_type_details",
    "get_search_strategies",

    # Utility tools
    "register_utility_tools",
    "create_extract_entities_tool",
    "create_filter_by_relevance_tool",
    "create_deduplicate_results_tool",
    "create_merge_results_tool",
    "create_summarize_results_tool",
    "extract_entities",
    "filter_by_relevance",
    "deduplicate_results",
    "merge_results",
    "summarize_results",
]

# Tool implementations
# Search tools (implemented)
from .search_tools import (
    register_search_tools,
    create_vector_search_tool,
    create_keyword_search_tool,
    create_document_similarity_tool,
    vector_search,
    keyword_search,
    document_similarity,
)

# Data tools (implemented)
from .data_tools import (
    register_data_tools,
    create_get_projects_list_tool,
    create_get_project_details_tool,
    create_get_document_types_tool,
    create_get_document_type_details_tool,
    create_get_search_strategies_tool,
    get_projects_list,
    get_project_details,
    get_document_types,
    get_document_type_details,
    get_search_strategies,
)

# Utility tools (implemented)
from .utility_tools import (
    register_utility_tools,
    create_extract_entities_tool,
    create_filter_by_relevance_tool,
    create_deduplicate_results_tool,
    create_merge_results_tool,
    create_summarize_results_tool,
    extract_entities,
    filter_by_relevance,
    deduplicate_results,
    merge_results,
    summarize_results,
)

# Version info
__version__ = "2.0.0"  # Phase 2: Tool Framework
__author__ = "EPIC Search Team"


# Module-level convenience functions

def initialize_tool_registry() -> ToolRegistry:
    """Initialize and populate the default tool registry.

    This function creates a ToolRegistry and registers all available tools
    from all categories (search, data, utility).

    Returns:
        Fully initialized ToolRegistry with all tools registered

    Example:
        >>> registry = initialize_tool_registry()
        >>> print(f"Loaded {len(registry.tools)} tools")
        >>> tools = registry.list_tools(category=ToolCategory.SEARCH)
        >>> print(f"Search tools: {[t.name for t in tools]}")

    Note:
        This function will be fully implemented once tool modules are created.
        For now, it returns an empty registry.
    """
    import logging
    logger = logging.getLogger(__name__)

    registry = ToolRegistry()

    # Register all tool categories

    # Search tools (implemented)
    try:
        from .search_tools import register_search_tools
        search_tools = register_search_tools(registry)
        logger.info(f"Registered {len(search_tools)} search tools")
    except ImportError as e:
        logger.warning(f"Could not register search tools: {e}")

    # Data tools (implemented)
    try:
        from .data_tools import register_data_tools
        data_tools = register_data_tools(registry)
        logger.info(f"Registered {len(data_tools)} data tools")
    except ImportError as e:
        logger.warning(f"Could not register data tools: {e}")

    # Utility tools (implemented)
    try:
        from .utility_tools import register_utility_tools
        utility_tools = register_utility_tools(registry)
        logger.info(f"Registered {len(utility_tools)} utility tools")
    except ImportError as e:
        logger.warning(f"Could not register utility tools: {e}")

    stats = registry.get_statistics()
    logger.info(f"Tool registry initialized: {stats}")

    return registry


# Singleton pattern for default registry (optional)
_default_registry = None


def get_default_registry() -> ToolRegistry:
    """Get or create the default tool registry singleton.

    This provides a module-level shared registry that can be used
    across the application without explicitly passing it around.

    Returns:
        The default ToolRegistry singleton instance

    Example:
        >>> from search_api.services.search_handlers.agent.tools import get_default_registry
        >>> registry = get_default_registry()
        >>> tool = registry.get("vector_search")
        >>> result = registry.execute_tool("vector_search", {"query": "test"})

    Note:
        The registry is initialized on first access. Subsequent calls
        return the same instance.
    """
    global _default_registry

    if _default_registry is None:
        _default_registry = initialize_tool_registry()

    return _default_registry


def reset_default_registry():
    """Reset the default registry singleton.

    Useful for testing or when you want to reload tools.

    Example:
        >>> from search_api.services.search_handlers.agent.tools import (
        ...     get_default_registry, reset_default_registry
        ... )
        >>> registry = get_default_registry()
        >>> # ... use registry ...
        >>> reset_default_registry()  # Clear and reload
        >>> registry = get_default_registry()  # Fresh instance
    """
    global _default_registry
    _default_registry = None


# Helper function for common use case
def execute_tool(tool_name: str, parameters: dict) -> dict:
    """Execute a tool using the default registry.

    Convenience function that uses the default registry to execute a tool.

    Args:
        tool_name: Name of the tool to execute
        parameters: Dictionary of parameters for the tool

    Returns:
        Execution result dictionary with 'success', 'result', etc.

    Example:
        >>> from search_api.services.search_handlers.agent.tools import execute_tool
        >>> result = execute_tool("vector_search", {
        ...     "query": "water quality impacts",
        ...     "limit": 5
        ... })
        >>> if result["success"]:
        ...     print(f"Found {len(result['result']['documents'])} documents")
    """
    registry = get_default_registry()
    return registry.execute_tool(tool_name, parameters)


# Note: __all__ is defined above with all exports


# Module-level docstring examples for documentation tools
if __name__ == "__main__":
    # Example usage when running module directly
    import sys
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Tool Framework Module")
    logger.info(f"Version: {__version__}")

    # Create a simple example tool
    example_tool = (
        create_tool("uppercase")
        .description("Convert text to uppercase")
        .category(ToolCategory.UTILITY)
        .cost(CostEstimate.VERY_LOW)
        .parameter(
            "text",
            ParameterType.STRING,
            "Text to convert",
            required=True,
            examples=["hello world"]
        )
        .returns("Uppercase text")
        .implementation(lambda text: {"result": text.upper()})
        .build()
    )

    logger.info(f"\nExample tool created: {example_tool.name}")
    logger.info(f"Category: {example_tool.category.value}")
    logger.info(f"Cost: {example_tool.cost_estimate.value}")

    # Test execution
    result = example_tool.execute(text="hello world")
    logger.info(f"\nExecution result: {result}")

    if result["success"]:
        logger.info(f"Output: {result['result']['result']}")

    # Show LLM description
    logger.info("\nLLM Description:")
    logger.info(example_tool.to_llm_description(include_examples=True))

    # Test registry
    logger.info("\nTesting registry:")
    registry = ToolRegistry()
    registry.register(example_tool)

    stats = registry.get_statistics()
    logger.info(f"Registry stats: {stats}")

    # Test via registry
    result2 = registry.execute_tool("uppercase", {"text": "test"})
    logger.info(f"Via registry: {result2}")

    sys.exit(0)
