"""Tests for tool registry."""

import pytest

from huxley.tools.registry import (
    Tool,
    ToolParameter,
    ToolRegistry,
    tool,
    get_registry,
)
from huxley.core.exceptions import ToolNotFoundError, ToolValidationError


class TestToolParameter:
    def test_to_json_schema_string(self):
        param = ToolParameter(name="test", type=str, description="A test param")
        schema = param.to_json_schema()
        assert schema["type"] == "string"
        assert schema["description"] == "A test param"

    def test_to_json_schema_int(self):
        param = ToolParameter(name="count", type=int)
        schema = param.to_json_schema()
        assert schema["type"] == "integer"


class TestTool:
    def test_to_openai_schema(self):
        def dummy_func(arg1: str, arg2: int = 5) -> str:
            return "result"

        tool = Tool(
            name="dummy",
            description="A dummy tool",
            func=dummy_func,
            parameters=[
                ToolParameter(name="arg1", type=str, required=True),
                ToolParameter(name="arg2", type=int, required=False, default=5),
            ],
        )

        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "dummy"
        assert "arg1" in schema["function"]["parameters"]["properties"]
        assert "arg1" in schema["function"]["parameters"]["required"]
        assert "arg2" not in schema["function"]["parameters"]["required"]

    def test_validate_arguments_valid(self):
        tool = Tool(
            name="test",
            description="Test",
            func=lambda x: x,
            parameters=[
                ToolParameter(name="x", type=str, required=True),
            ],
        )
        result = tool.validate_arguments({"x": "hello"})
        assert result == {"x": "hello"}

    def test_validate_arguments_missing_required(self):
        tool = Tool(
            name="test",
            description="Test",
            func=lambda x: x,
            parameters=[
                ToolParameter(name="x", type=str, required=True),
            ],
        )
        with pytest.raises(ToolValidationError):
            tool.validate_arguments({})


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()

        @registry.register(name="my_tool", description="My tool")
        def my_tool(arg: str) -> str:
            return arg

        tool = registry.get("my_tool")
        assert tool.name == "my_tool"
        assert tool.description == "My tool"

    def test_get_nonexistent(self):
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.get("nonexistent")

    def test_list_by_tag(self):
        registry = ToolRegistry()

        @registry.register(tags={"bio"})
        def bio_tool() -> str:
            return "bio"

        @registry.register(tags={"math"})
        def math_tool() -> str:
            return "math"

        bio_tools = registry.list(tags={"bio"})
        assert len(bio_tools) == 1
        assert bio_tools[0].name == "bio_tool"

    def test_get_openai_schemas(self):
        registry = ToolRegistry()

        @registry.register()
        def tool1() -> str:
            return "1"

        @registry.register()
        def tool2() -> str:
            return "2"

        schemas = registry.get_openai_schemas()
        assert len(schemas) == 2
        assert all(s["type"] == "function" for s in schemas)


class TestToolDecorator:
    def test_decorator_extracts_params(self):
        # Use a fresh registry
        registry = ToolRegistry()

        @registry.register()
        def analyze(sequence: str, format: str = "fasta") -> dict:
            """Analyze a sequence.

            :param sequence: The sequence
            :param format: The format
            """
            return {}

        tool = registry.get("analyze")
        assert len(tool.parameters) == 2
        assert tool.parameters[0].name == "sequence"
        assert tool.parameters[0].required is True
        assert tool.parameters[1].name == "format"
        assert tool.parameters[1].required is False
