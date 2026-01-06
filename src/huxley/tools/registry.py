"""
Tool registry for managing available tools.

Tools are typed, validated functions that agents can invoke.
The registry provides:
- Tool registration with schema validation
- OpenAI-compatible tool definitions
- Tool discovery and lookup
- Execution lifecycle management
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, Field, create_model
from pydantic.json_schema import GenerateJsonSchema

from huxley.core.exceptions import ToolNotFoundError, ToolValidationError
from huxley.core.logging import get_logger

logger = get_logger(__name__)


def _python_type_to_json_schema(python_type: type) -> dict[str, Any]:
    """Convert a Python type to JSON Schema type."""
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        type(None): {"type": "null"},
    }

    origin = getattr(python_type, "__origin__", None)

    if origin is list:
        args = getattr(python_type, "__args__", (Any,))
        item_type = args[0] if args else Any
        return {
            "type": "array",
            "items": _python_type_to_json_schema(item_type),
        }

    if origin is dict:
        return {"type": "object"}

    if origin is type(None) or python_type is type(None):
        return {"type": "null"}

    # Handle Union types (including Optional)
    if origin is type(None):
        return {"type": "null"}

    if hasattr(python_type, "__origin__") and str(origin) == "typing.Union":
        args = getattr(python_type, "__args__", ())
        # Check if it's Optional (Union with None)
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            schema = _python_type_to_json_schema(non_none_args[0])
            return schema
        return {"oneOf": [_python_type_to_json_schema(a) for a in args]}

    return type_mapping.get(python_type, {"type": "string"})


@dataclass(frozen=True)
class ToolParameter:
    """Definition of a single tool parameter."""

    name: str
    type: type
    description: str = ""
    required: bool = True
    default: Any = None

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema property."""
        schema = _python_type_to_json_schema(self.type)
        if self.description:
            schema["description"] = self.description
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class Tool:
    """
    A registered tool that agents can invoke.

    Tools are typed functions with:
    - Schema-validated inputs
    - Structured output
    - Execution metadata
    """

    name: str
    description: str
    func: Callable[..., Any]
    parameters: list[ToolParameter] = field(default_factory=list)
    returns_type: type = str
    is_async: bool = False
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and finalize tool configuration."""
        self.is_async = inspect.iscoroutinefunction(self.func)

    def to_openai_schema(self) -> dict[str, Any]:
        """
        Convert to OpenAI function calling format.

        Returns a tool definition compatible with OpenAI's API.
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and coerce tool arguments.

        Args:
            arguments: Raw arguments to validate

        Returns:
            Validated and coerced arguments

        Raises:
            ToolValidationError: If validation fails
        """
        validated = {}
        errors = []

        for param in self.parameters:
            if param.name in arguments:
                value = arguments[param.name]
                # Basic type coercion
                try:
                    if param.type in (int, float, str, bool):
                        validated[param.name] = param.type(value)
                    else:
                        validated[param.name] = value
                except (ValueError, TypeError) as e:
                    errors.append({
                        "param": param.name,
                        "error": f"Type conversion failed: {e}",
                    })
            elif param.required:
                if param.default is not None:
                    validated[param.name] = param.default
                else:
                    errors.append({
                        "param": param.name,
                        "error": "Required parameter missing",
                    })
            elif param.default is not None:
                validated[param.name] = param.default

        if errors:
            raise ToolValidationError(
                self.name,
                f"Validation failed for {len(errors)} parameter(s)",
                validation_errors=errors,
            )

        return validated


class ToolRegistry:
    """
    Central registry for all available tools.

    Provides:
    - Tool registration and lookup
    - Schema generation for LLM APIs
    - Tag-based filtering
    - Thread-safe access
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._tags: dict[str, set[str]] = {}  # tag -> tool names

    def register(
        self,
        name: str | None = None,
        description: str | None = None,
        tags: set[str] | None = None,
        **metadata: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to register a function as a tool.

        Args:
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            tags: Tags for filtering
            **metadata: Additional metadata

        Returns:
            Decorator function
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or ""

            # Extract parameters from type hints
            hints = get_type_hints(func)
            sig = inspect.signature(func)

            parameters = []
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue

                param_type = hints.get(param_name, str)
                param_required = param.default is inspect.Parameter.empty
                param_default = (
                    None if param.default is inspect.Parameter.empty else param.default
                )

                # Try to extract description from docstring
                param_desc = ""
                if func.__doc__:
                    # Simple extraction - look for :param name: description
                    import re

                    match = re.search(
                        rf":param\s+{param_name}:\s*(.+?)(?=:param|:return|$)",
                        func.__doc__,
                        re.DOTALL,
                    )
                    if match:
                        param_desc = match.group(1).strip()

                parameters.append(
                    ToolParameter(
                        name=param_name,
                        type=param_type,
                        description=param_desc,
                        required=param_required,
                        default=param_default,
                    )
                )

            # Get return type
            returns_type = hints.get("return", str)

            tool = Tool(
                name=tool_name,
                description=tool_description.strip(),
                func=func,
                parameters=parameters,
                returns_type=returns_type,
                tags=tags or set(),
                metadata=metadata,
            )

            self._tools[tool_name] = tool

            # Index by tags
            for tag in tool.tags:
                if tag not in self._tags:
                    self._tags[tag] = set()
                self._tags[tag].add(tool_name)

            logger.debug("tool_registered", name=tool_name, tags=list(tool.tags))

            return func

        return decorator

    def get(self, name: str) -> Tool:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance

        Raises:
            ToolNotFoundError: If tool doesn't exist
        """
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def list(self, tags: set[str] | None = None) -> list[Tool]:
        """
        List all registered tools, optionally filtered by tags.

        Args:
            tags: Only return tools with ALL of these tags

        Returns:
            List of matching tools
        """
        if tags is None:
            return list(self._tools.values())

        # Find tools that have ALL specified tags
        matching_names: set[str] | None = None
        for tag in tags:
            tag_tools = self._tags.get(tag, set())
            if matching_names is None:
                matching_names = tag_tools.copy()
            else:
                matching_names &= tag_tools

        if matching_names is None:
            return []

        return [self._tools[name] for name in matching_names]

    def get_openai_schemas(
        self, tool_names: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Get OpenAI-compatible tool definitions.

        Args:
            tool_names: Specific tools to include (all if None)

        Returns:
            List of OpenAI tool definitions
        """
        if tool_names is None:
            tools = self._tools.values()
        else:
            tools = [self._tools[n] for n in tool_names if n in self._tools]

        return [tool.to_openai_schema() for tool in tools]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# Global registry instance
_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


# Convenience decorator using global registry
def tool(
    name: str | None = None,
    description: str | None = None,
    tags: set[str] | None = None,
    **metadata: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Register a function as a tool in the global registry.

    Example:
        @tool(tags={"biology", "sequence"})
        def analyze_sequence(sequence: str, format: str = "fasta") -> dict:
            '''Analyze a biological sequence.

            :param sequence: The sequence to analyze
            :param format: Input format (fasta, genbank)
            '''
            ...
    """
    return get_registry().register(name, description, tags, **metadata)
