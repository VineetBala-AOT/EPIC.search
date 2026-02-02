"""
Tool Registry - Core framework for defining, registering, and executing agent tools.

This module provides the foundational classes for the tool framework:
- ParameterType: Enum for supported parameter types
- ToolParameter: Schema for a tool parameter with validation
- ToolCategory: Enum for tool categories
- CostEstimate: Enum for execution cost estimates
- ToolMetadata: Additional metadata about a tool
- Tool: Complete tool definition with schema and implementation
- ToolRegistry: Central registry for managing tools
- ToolBuilder: Fluent interface for building tools
- create_tool(): Factory function for ToolBuilder

Example:
    >>> from tool_registry import create_tool, ToolRegistry, ParameterType, ToolCategory, CostEstimate
    >>>
    >>> tool = (create_tool("my_tool")
    ...         .description("My custom tool")
    ...         .category(ToolCategory.UTILITY)
    ...         .cost(CostEstimate.LOW)
    ...         .parameter("input", ParameterType.STRING, "Input text", required=True)
    ...         .returns("Processed result")
    ...         .implementation(lambda input: {"result": input.upper()})
    ...         .build())
    >>>
    >>> registry = ToolRegistry()
    >>> registry.register(tool)
    >>> result = registry.execute_tool("my_tool", {"input": "hello"})
    >>> print(result["result"])
    {'result': 'HELLO'}
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ParameterType(Enum):
    """Supported parameter types for tool parameters."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    ANY = "any"

    def __str__(self) -> str:
        return self.value


class ToolCategory(Enum):
    """Categories for organizing tools."""

    SEARCH = "search"
    DATA = "data"
    UTILITY = "utility"
    ANALYSIS = "analysis"
    TRANSFORMATION = "transformation"

    def __str__(self) -> str:
        return self.value


class CostEstimate(Enum):
    """Estimated cost/latency for tool execution.

    Use these to help the agent make efficient tool choices:
    - VERY_LOW: < 50ms, pure computation, no I/O
    - LOW: 50-200ms, cached data, simple queries
    - MEDIUM: 200-1000ms, database queries, vector operations
    - HIGH: 1-5s, complex vector search, multiple queries
    - VERY_HIGH: > 5s, LLM calls, heavy processing
    """

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def get_order(cls) -> List['CostEstimate']:
        """Get cost estimates in order from lowest to highest."""
        return [cls.VERY_LOW, cls.LOW, cls.MEDIUM, cls.HIGH, cls.VERY_HIGH]

    def __lt__(self, other: 'CostEstimate') -> bool:
        if not isinstance(other, CostEstimate):
            return NotImplemented
        order = self.get_order()
        return order.index(self) < order.index(other)

    def __le__(self, other: 'CostEstimate') -> bool:
        if not isinstance(other, CostEstimate):
            return NotImplemented
        order = self.get_order()
        return order.index(self) <= order.index(other)


# =============================================================================
# ToolParameter
# =============================================================================

@dataclass
class ToolParameter:
    """Schema for a tool parameter with validation.

    Attributes:
        name: Parameter name (used in function signature)
        type: Parameter type (STRING, INTEGER, LIST, etc)
        description: Human-readable description for LLM understanding
        required: Whether this parameter is required
        default: Default value if not provided (only for optional params)
        validation: Optional validation function. Returns True if valid.
        examples: Example values to help LLM understand usage
        constraints: Additional constraints (min, max, pattern, enum, etc)

    Example:
        >>> param = ToolParameter(
        ...     name="query",
        ...     type=ParameterType.STRING,
        ...     description="Search query text",
        ...     required=True,
        ...     examples=["water quality", "mining impacts"],
        ...     constraints={"min_length": 3, "max_length": 500}
        ... )
        >>> is_valid, error = param.validate("hello")
        >>> print(is_valid)
        True
    """

    name: str
    type: ParameterType
    description: str
    required: bool = False
    default: Any = None
    validation: Optional[Callable[[Any], bool]] = None
    examples: Optional[List[Any]] = None
    constraints: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate parameter definition."""
        if not self.name:
            raise ValueError("Parameter name cannot be empty")

        if not isinstance(self.type, ParameterType):
            raise ValueError(f"Invalid parameter type: {self.type}")

        if self.required and self.default is not None:
            raise ValueError(
                f"Required parameter '{self.name}' cannot have a default value"
            )

        if self.examples is None:
            self.examples = []

        if self.constraints is None:
            self.constraints = {}

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a parameter value.

        Args:
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is None
        """
        # Check if None is provided for required parameter
        if value is None:
            if self.required:
                return False, f"Required parameter '{self.name}' is missing"
            # Optional with None - use default or skip
            return True, None

        # Type validation
        type_valid, type_error = self._validate_type(value)
        if not type_valid:
            return False, type_error

        # Constraints validation
        if self.constraints:
            constraint_valid, constraint_error = self._validate_constraints(value)
            if not constraint_valid:
                return False, constraint_error

        # Custom validation function
        if self.validation is not None:
            try:
                if not self.validation(value):
                    return False, f"Parameter '{self.name}' failed custom validation"
            except Exception as e:
                return False, f"Validation error for '{self.name}': {str(e)}"

        return True, None

    def _validate_type(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Check if value matches expected type."""
        if self.type == ParameterType.ANY:
            return True, None

        type_checks = {
            ParameterType.STRING: lambda v: isinstance(v, str),
            ParameterType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
            ParameterType.FLOAT: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            ParameterType.BOOLEAN: lambda v: isinstance(v, bool),
            ParameterType.LIST: lambda v: isinstance(v, list),
            ParameterType.DICT: lambda v: isinstance(v, dict),
        }

        checker = type_checks.get(self.type)
        if checker and not checker(value):
            return False, f"Parameter '{self.name}' must be of type {self.type.value}, got {type(value).__name__}"

        return True, None

    def _validate_constraints(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate value against constraints."""
        constraints = self.constraints

        # String constraints
        if self.type == ParameterType.STRING and isinstance(value, str):
            if "min_length" in constraints and len(value) < constraints["min_length"]:
                return False, f"'{self.name}' must be at least {constraints['min_length']} characters"

            if "max_length" in constraints and len(value) > constraints["max_length"]:
                return False, f"'{self.name}' must be at most {constraints['max_length']} characters"

            if "pattern" in constraints:
                if not re.match(constraints["pattern"], value):
                    return False, f"'{self.name}' does not match required pattern"

            if "enum" in constraints and value not in constraints["enum"]:
                allowed = ", ".join(map(str, constraints["enum"]))
                return False, f"'{self.name}' must be one of: {allowed}"

        # Numeric constraints
        if self.type in (ParameterType.INTEGER, ParameterType.FLOAT):
            if "min" in constraints and value < constraints["min"]:
                return False, f"'{self.name}' must be >= {constraints['min']}"

            if "max" in constraints and value > constraints["max"]:
                return False, f"'{self.name}' must be <= {constraints['max']}"

        # List constraints
        if self.type == ParameterType.LIST and isinstance(value, list):
            if "min_items" in constraints and len(value) < constraints["min_items"]:
                return False, f"'{self.name}' must have at least {constraints['min_items']} items"

            if "max_items" in constraints and len(value) > constraints["max_items"]:
                return False, f"'{self.name}' must have at most {constraints['max_items']} items"

            if "item_type" in constraints:
                item_type = constraints["item_type"]
                # Handle string type names
                if isinstance(item_type, str):
                    type_map = {"str": str, "int": int, "float": float, "bool": bool}
                    item_type = type_map.get(item_type, str)

                for i, item in enumerate(value):
                    if not isinstance(item, item_type):
                        return False, f"All items in '{self.name}' must be of type {item_type.__name__}"

        return True, None

    def get_value_with_default(self, value: Any) -> Any:
        """Get the value, using default if value is None.

        Args:
            value: Provided value

        Returns:
            Value or default
        """
        if value is None:
            return self.default
        return value

    def to_llm_description(self) -> str:
        """Generate description for LLM prompt.

        Returns:
            Formatted string describing this parameter
        """
        # Parameter name and type
        desc = f"{self.name} ({self.type.value})"

        # Required/optional indicator
        if self.required:
            desc += " [REQUIRED]"
        else:
            desc += " [optional]"
            if self.default is not None:
                desc += f" (default: {self.default})"

        # Description
        desc += f": {self.description}"

        # Add constraints info
        if self.constraints:
            constraint_parts = []

            if "enum" in self.constraints:
                enum_values = ", ".join(map(str, self.constraints["enum"]))
                constraint_parts.append(f"Options: {enum_values}")
            else:
                if "min" in self.constraints:
                    constraint_parts.append(f"min={self.constraints['min']}")
                if "max" in self.constraints:
                    constraint_parts.append(f"max={self.constraints['max']}")
                if "min_length" in self.constraints:
                    constraint_parts.append(f"min_length={self.constraints['min_length']}")
                if "max_length" in self.constraints:
                    constraint_parts.append(f"max_length={self.constraints['max_length']}")

            if constraint_parts:
                desc += f" | {', '.join(constraint_parts)}"

        # Add examples
        if self.examples:
            example_strs = [str(e) for e in self.examples[:2]]
            desc += f" | Examples: {', '.join(example_strs)}"

        return desc

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert parameter to JSON schema format.

        Returns:
            JSON schema dictionary
        """
        type_mapping = {
            ParameterType.STRING: "string",
            ParameterType.INTEGER: "integer",
            ParameterType.FLOAT: "number",
            ParameterType.BOOLEAN: "boolean",
            ParameterType.LIST: "array",
            ParameterType.DICT: "object",
            ParameterType.ANY: "string",
        }

        schema = {
            "type": type_mapping.get(self.type, "string"),
            "description": self.description
        }

        # Add constraints
        if self.constraints:
            if "enum" in self.constraints:
                schema["enum"] = self.constraints["enum"]
            if "min" in self.constraints:
                schema["minimum"] = self.constraints["min"]
            if "max" in self.constraints:
                schema["maximum"] = self.constraints["max"]
            if "min_length" in self.constraints:
                schema["minLength"] = self.constraints["min_length"]
            if "max_length" in self.constraints:
                schema["maxLength"] = self.constraints["max_length"]
            if "pattern" in self.constraints:
                schema["pattern"] = self.constraints["pattern"]

        # Add examples
        if self.examples:
            schema["examples"] = self.examples

        # Add default
        if self.default is not None:
            schema["default"] = self.default

        return schema


# =============================================================================
# ToolMetadata
# =============================================================================

@dataclass
class ToolMetadata:
    """Additional metadata about a tool.

    Attributes:
        version: Tool version for compatibility tracking
        author: Tool author/maintainer
        tags: Tags for discovery and filtering
        deprecation_warning: Warning if tool is deprecated
        requires_auth: Whether tool requires authentication
        rate_limited: Whether tool has rate limits
        idempotent: Whether multiple calls with same params produce same result
    """

    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    deprecation_warning: Optional[str] = None
    requires_auth: bool = False
    rate_limited: bool = False
    idempotent: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "deprecation_warning": self.deprecation_warning,
            "requires_auth": self.requires_auth,
            "rate_limited": self.rate_limited,
            "idempotent": self.idempotent
        }


# =============================================================================
# Tool
# =============================================================================

@dataclass
class Tool:
    """Complete tool definition with schema and implementation.

    Attributes:
        name: Unique tool identifier (snake_case)
        description: Human-readable description (2-3 sentences for LLM)
        parameters: Parameter definitions keyed by parameter name
        function: Python function that implements the tool
        returns: Description of what the tool returns
        category: Tool category for organization
        cost_estimate: Estimated execution cost/latency
        metadata: Additional tool metadata
        examples: Example invocations with input/output

    Example:
        >>> def my_func(query: str, limit: int = 10) -> Dict:
        ...     return {"results": [query] * limit}
        >>>
        >>> tool = Tool(
        ...     name="my_search",
        ...     description="Search for things",
        ...     parameters={
        ...         "query": ToolParameter("query", ParameterType.STRING, "Query", required=True),
        ...         "limit": ToolParameter("limit", ParameterType.INTEGER, "Limit", default=10)
        ...     },
        ...     function=my_func,
        ...     returns="Search results",
        ...     category=ToolCategory.SEARCH,
        ...     cost_estimate=CostEstimate.LOW
        ... )
        >>> result = tool.execute(query="test")
        >>> print(result["success"])
        True
    """

    name: str
    description: str
    parameters: Dict[str, ToolParameter]
    function: Callable
    returns: str
    category: ToolCategory
    cost_estimate: CostEstimate
    metadata: ToolMetadata = field(default_factory=ToolMetadata)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate tool definition."""
        if not self.name:
            raise ValueError("Tool name cannot be empty")

        # Validate name format (alphanumeric with underscores)
        if not re.match(r'^[a-z][a-z0-9_]*$', self.name):
            raise ValueError(
                f"Tool name '{self.name}' must be lowercase alphanumeric with underscores, "
                f"starting with a letter"
            )

        if not self.description:
            raise ValueError(f"Tool '{self.name}' must have a description")

        if not callable(self.function):
            raise ValueError(f"Tool '{self.name}' function must be callable")

        if not self.returns:
            raise ValueError(f"Tool '{self.name}' must specify return value description")

    def validate_parameters(
        self,
        params: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Validate parameters for tool execution.

        Args:
            params: Parameter values to validate

        Returns:
            Tuple of (is_valid, validated_params, error_messages)
            validated_params includes defaults for missing optional params
        """
        errors = []
        validated = {}

        # Validate each defined parameter
        for param_name, param_def in self.parameters.items():
            value = params.get(param_name)

            # Validate parameter
            is_valid, error = param_def.validate(value)
            if not is_valid:
                errors.append(error)
                continue

            # Get value with default applied
            final_value = param_def.get_value_with_default(value)

            # Only include if we have a value
            if final_value is not None:
                validated[param_name] = final_value

        # Check for unexpected parameters
        expected_params = set(self.parameters.keys())
        provided_params = set(params.keys())
        unexpected = provided_params - expected_params

        if unexpected:
            errors.append(f"Unexpected parameters: {', '.join(sorted(unexpected))}")

        is_valid = len(errors) == 0
        return is_valid, validated, errors

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with validated parameters.

        Args:
            **kwargs: Tool parameters

        Returns:
            Execution result dictionary with:
            - success: bool
            - result: Any (if successful)
            - error: str (if failed)
            - validation_errors: List[str] (if validation failed)
            - tool: str (tool name)
            - execution_time_ms: float
            - cost_estimate: str
        """
        start_time = time.time()

        # Validate parameters
        is_valid, validated_params, errors = self.validate_parameters(kwargs)

        if not is_valid:
            execution_time = (time.time() - start_time) * 1000
            logger.warning(f"🔧 Tool '{self.name}' validation failed: {errors}")
            return {
                "success": False,
                "error": "Parameter validation failed",
                "validation_errors": errors,
                "tool": self.name,
                "execution_time_ms": execution_time,
                "cost_estimate": self.cost_estimate.value
            }

        # Execute tool
        try:
            logger.debug(f"🔧 Executing tool '{self.name}' with params: {validated_params}")
            result = self.function(**validated_params)
            execution_time = (time.time() - start_time) * 1000

            logger.debug(f"🔧 Tool '{self.name}' completed in {execution_time:.1f}ms")
            return {
                "success": True,
                "result": result,
                "tool": self.name,
                "execution_time_ms": execution_time,
                "cost_estimate": self.cost_estimate.value
            }

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"🔧 Tool '{self.name}' execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "tool": self.name,
                "execution_time_ms": execution_time,
                "cost_estimate": self.cost_estimate.value
            }

    def to_llm_description(self, include_examples: bool = False) -> str:
        """Generate description for LLM prompt.

        Args:
            include_examples: Whether to include usage examples

        Returns:
            Formatted tool description for LLM
        """
        lines = []

        # Tool signature
        param_names = ", ".join(self.parameters.keys())
        lines.append(f"**{self.name}({param_names})**")
        lines.append("")

        # Description
        lines.append(self.description)
        lines.append("")

        # Category and cost
        lines.append(f"- Category: {self.category.value}")
        lines.append(f"- Cost: {self.cost_estimate.value}")
        lines.append(f"- Returns: {self.returns}")
        lines.append("")

        # Parameters
        if self.parameters:
            lines.append("Parameters:")
            for param in self.parameters.values():
                lines.append(f"  - {param.to_llm_description()}")
            lines.append("")

        # Examples
        if include_examples and self.examples:
            lines.append("Examples:")
            for i, example in enumerate(self.examples[:2], 1):
                input_str = json.dumps(example.get("input", {}), indent=None)
                output_str = str(example.get("output", "N/A"))[:100]
                lines.append(f"  {i}. Input: {input_str}")
                lines.append(f"     Output: {output_str}")
            lines.append("")

        # Deprecation warning
        if self.metadata.deprecation_warning:
            lines.append(f"⚠️ DEPRECATED: {self.metadata.deprecation_warning}")
            lines.append("")

        return "\n".join(lines)

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert tool to JSON schema format (OpenAI function calling format).

        Returns:
            JSON schema dictionary compatible with OpenAI function calling
        """
        properties = {}
        required = []

        for param_name, param_def in self.parameters.items():
            properties[param_name] = param_def.to_json_schema()

            if param_def.required:
                required.append(param_name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                name: {
                    "name": p.name,
                    "type": p.type.value,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "examples": p.examples,
                    "constraints": p.constraints
                }
                for name, p in self.parameters.items()
            },
            "returns": self.returns,
            "category": self.category.value,
            "cost_estimate": self.cost_estimate.value,
            "metadata": self.metadata.to_dict(),
            "examples": self.examples
        }


# =============================================================================
# ToolRegistry
# =============================================================================

class ToolRegistry:
    """Central registry for managing agent tools.

    The registry provides:
    - Tool registration and retrieval
    - Tool listing with filtering
    - Tool description generation for LLM prompts
    - Tool execution with validation

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(my_tool)
        >>> tool = registry.get("my_tool")
        >>> result = registry.execute_tool("my_tool", {"query": "test"})
    """

    def __init__(self):
        """Initialize empty registry."""
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {
            category: [] for category in ToolCategory
        }

    @property
    def tools(self) -> Dict[str, Tool]:
        """Get all registered tools."""
        return self._tools.copy()

    def register(self, tool: Tool) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool to register

        Raises:
            ValueError: If tool name already exists or tool is invalid
        """
        if not isinstance(tool, Tool):
            raise ValueError(f"Expected Tool instance, got {type(tool).__name__}")

        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")

        # Log deprecation warning if applicable
        if tool.metadata.deprecation_warning:
            logger.warning(
                f"Registering deprecated tool '{tool.name}': {tool.metadata.deprecation_warning}"
            )

        # Register tool
        self._tools[tool.name] = tool
        self._categories[tool.category].append(tool.name)

        logger.info(
            f"🔧 Registered tool: {tool.name} "
            f"(category: {tool.category.value}, cost: {tool.cost_estimate.value})"
        )

    def unregister(self, name: str) -> bool:
        """Unregister a tool from the registry.

        Args:
            name: Tool name to unregister

        Returns:
            True if tool was removed, False if not found
        """
        if name not in self._tools:
            logger.warning(f"🔧 Tool '{name}' not found in registry")
            return False

        tool = self._tools[name]
        del self._tools[name]
        self._categories[tool.category].remove(name)

        logger.info(f"🔧 Unregistered tool: {name}")
        return True

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool exists in the registry.

        Args:
            name: Tool name

        Returns:
            True if tool exists
        """
        return name in self._tools

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        tags: Optional[List[str]] = None,
        max_cost: Optional[CostEstimate] = None
    ) -> List[Tool]:
        """List tools with optional filtering.

        Args:
            category: Filter by category
            tags: Filter by tags (any match)
            max_cost: Maximum cost estimate

        Returns:
            List of matching tools
        """
        tools = list(self._tools.values())

        # Filter by category
        if category is not None:
            tools = [t for t in tools if t.category == category]

        # Filter by tags (any match)
        if tags:
            tools = [
                t for t in tools
                if any(tag in t.metadata.tags for tag in tags)
            ]

        # Filter by cost
        if max_cost is not None:
            tools = [t for t in tools if t.cost_estimate <= max_cost]

        return tools

    def get_tool_names(self, category: Optional[ToolCategory] = None) -> List[str]:
        """Get list of tool names.

        Args:
            category: Optional category filter

        Returns:
            List of tool names
        """
        if category is not None:
            return self._categories.get(category, []).copy()
        return list(self._tools.keys())

    def get_tool_descriptions(
        self,
        category: Optional[ToolCategory] = None,
        include_examples: bool = False,
        format: str = "text"
    ) -> str:
        """Get formatted tool descriptions for LLM prompt.

        Args:
            category: Optional category filter
            include_examples: Include usage examples
            format: Output format ("text" or "json")

        Returns:
            Formatted tool descriptions
        """
        tools = self.list_tools(category=category)

        if format == "json":
            schemas = [tool.to_json_schema() for tool in tools]
            return json.dumps(schemas, indent=2)

        # Text format - group by category
        descriptions = []
        by_category: Dict[ToolCategory, List[Tool]] = {}

        for tool in tools:
            by_category.setdefault(tool.category, []).append(tool)

        for cat in sorted(by_category.keys(), key=lambda c: c.value):
            cat_tools = by_category[cat]
            descriptions.append(f"\n## {cat.value.upper()} TOOLS\n")

            # Sort by cost (cheapest first)
            cat_tools.sort(key=lambda t: CostEstimate.get_order().index(t.cost_estimate))

            for tool in cat_tools:
                descriptions.append(tool.to_llm_description(include_examples=include_examples))

        return "\n".join(descriptions)

    def validate_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
        """Validate a tool call before execution.

        Args:
            tool_name: Name of tool to call
            parameters: Parameters for the call

        Returns:
            Tuple of (is_valid, validated_params, errors)
        """
        # Check tool exists
        tool = self.get(tool_name)
        if tool is None:
            return False, None, [f"Tool '{tool_name}' not found"]

        # Log deprecation warning
        if tool.metadata.deprecation_warning:
            logger.warning(
                f"Using deprecated tool '{tool_name}': {tool.metadata.deprecation_warning}"
            )

        # Validate parameters
        return tool.validate_parameters(parameters)

    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool with parameters.

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters

        Returns:
            Execution result dictionary
        """
        tool = self.get(tool_name)

        if tool is None:
            logger.error(f"🔧 Tool '{tool_name}' not found in registry")
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "tool": tool_name
            }

        return tool.execute(**parameters)

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Statistics dictionary with counts by category and cost
        """
        return {
            "total_tools": len(self._tools),
            "by_category": {
                cat.value: len(tools)
                for cat, tools in self._categories.items()
            },
            "by_cost": {
                cost.value: len([
                    t for t in self._tools.values()
                    if t.cost_estimate == cost
                ])
                for cost in CostEstimate
            },
            "deprecated": len([
                t for t in self._tools.values()
                if t.metadata.deprecation_warning
            ])
        }

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        for category in self._categories:
            self._categories[category].clear()
        logger.info("🔧 Tool registry cleared")


# =============================================================================
# ToolBuilder
# =============================================================================

class ToolBuilder:
    """Fluent interface for building tools.

    Example:
        >>> tool = (ToolBuilder("my_tool")
        ...         .description("My tool description")
        ...         .category(ToolCategory.UTILITY)
        ...         .cost(CostEstimate.LOW)
        ...         .parameter("input", ParameterType.STRING, "Input text", required=True)
        ...         .returns("Processed result")
        ...         .implementation(lambda input: {"result": input.upper()})
        ...         .build())
    """

    def __init__(self, name: str):
        """Initialize builder with tool name.

        Args:
            name: Tool name (snake_case)
        """
        self._name = name
        self._description: str = ""
        self._parameters: Dict[str, ToolParameter] = {}
        self._function: Optional[Callable] = None
        self._returns: str = ""
        self._category: ToolCategory = ToolCategory.UTILITY
        self._cost: CostEstimate = CostEstimate.LOW
        self._metadata: ToolMetadata = ToolMetadata()
        self._examples: List[Dict[str, Any]] = []

    def description(self, desc: str) -> 'ToolBuilder':
        """Set tool description.

        Args:
            desc: Human-readable description (2-3 sentences)

        Returns:
            Self for chaining
        """
        self._description = desc
        return self

    def category(self, cat: ToolCategory) -> 'ToolBuilder':
        """Set tool category.

        Args:
            cat: Tool category

        Returns:
            Self for chaining
        """
        self._category = cat
        return self

    def cost(self, cost: CostEstimate) -> 'ToolBuilder':
        """Set cost estimate.

        Args:
            cost: Estimated execution cost

        Returns:
            Self for chaining
        """
        self._cost = cost
        return self

    def returns(self, ret: str) -> 'ToolBuilder':
        """Set return description.

        Args:
            ret: Description of return value

        Returns:
            Self for chaining
        """
        self._returns = ret
        return self

    def parameter(
        self,
        name: str,
        type: ParameterType,
        description: str,
        required: bool = False,
        default: Any = None,
        validation: Optional[Callable[[Any], bool]] = None,
        examples: Optional[List[Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> 'ToolBuilder':
        """Add a parameter to the tool.

        Args:
            name: Parameter name
            type: Parameter type
            description: Parameter description
            required: Whether parameter is required
            default: Default value (for optional params)
            validation: Custom validation function
            examples: Example values
            constraints: Value constraints

        Returns:
            Self for chaining
        """
        self._parameters[name] = ToolParameter(
            name=name,
            type=type,
            description=description,
            required=required,
            default=default,
            validation=validation,
            examples=examples,
            constraints=constraints
        )
        return self

    def implementation(self, func: Callable) -> 'ToolBuilder':
        """Set implementation function.

        Args:
            func: Python function implementing the tool

        Returns:
            Self for chaining
        """
        self._function = func
        return self

    def example(
        self,
        input: Dict[str, Any],
        output: Any
    ) -> 'ToolBuilder':
        """Add usage example.

        Args:
            input: Example input parameters
            output: Example output

        Returns:
            Self for chaining
        """
        self._examples.append({"input": input, "output": output})
        return self

    def metadata(
        self,
        version: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
        deprecation_warning: Optional[str] = None,
        requires_auth: Optional[bool] = None,
        rate_limited: Optional[bool] = None,
        idempotent: Optional[bool] = None
    ) -> 'ToolBuilder':
        """Set metadata fields.

        Args:
            version: Tool version
            author: Tool author
            tags: Tool tags
            deprecation_warning: Deprecation message
            requires_auth: Whether auth is required
            rate_limited: Whether tool is rate limited
            idempotent: Whether tool is idempotent

        Returns:
            Self for chaining
        """
        if version is not None:
            self._metadata.version = version
        if author is not None:
            self._metadata.author = author
        if tags is not None:
            self._metadata.tags = tags
        if deprecation_warning is not None:
            self._metadata.deprecation_warning = deprecation_warning
        if requires_auth is not None:
            self._metadata.requires_auth = requires_auth
        if rate_limited is not None:
            self._metadata.rate_limited = rate_limited
        if idempotent is not None:
            self._metadata.idempotent = idempotent
        return self

    def build(self) -> Tool:
        """Build the tool.

        Returns:
            Tool instance

        Raises:
            ValueError: If tool definition is incomplete
        """
        if not self._description:
            raise ValueError(f"Tool '{self._name}' must have a description")

        if self._function is None:
            raise ValueError(f"Tool '{self._name}' must have an implementation function")

        if not self._returns:
            raise ValueError(f"Tool '{self._name}' must specify return value")

        return Tool(
            name=self._name,
            description=self._description,
            parameters=self._parameters,
            function=self._function,
            returns=self._returns,
            category=self._category,
            cost_estimate=self._cost,
            metadata=self._metadata,
            examples=self._examples
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_tool(name: str) -> ToolBuilder:
    """Create a new tool using builder pattern.

    This is the recommended way to create tools as it provides
    a clean, readable, fluent interface.

    Args:
        name: Tool name (snake_case, e.g., "vector_search")

    Returns:
        ToolBuilder instance for chaining

    Example:
        >>> tool = (create_tool("vector_search")
        ...         .description("Perform semantic vector search")
        ...         .category(ToolCategory.SEARCH)
        ...         .cost(CostEstimate.MEDIUM)
        ...         .parameter("query", ParameterType.STRING, "Search query", required=True)
        ...         .parameter("limit", ParameterType.INTEGER, "Max results", default=10)
        ...         .returns("Search results with documents and scores")
        ...         .implementation(my_search_function)
        ...         .example({"query": "water quality", "limit": 5}, {"documents": [...]})
        ...         .build())
    """
    return ToolBuilder(name)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "ParameterType",
    "ToolCategory",
    "CostEstimate",

    # Classes
    "ToolParameter",
    "ToolMetadata",
    "Tool",
    "ToolRegistry",
    "ToolBuilder",

    # Factory
    "create_tool",
]
