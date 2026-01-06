"""Tests for core types."""

import pytest
from datetime import datetime

from huxley.core.types import (
    AgentConfig,
    AgentState,
    ExecutionContext,
    Message,
    MessageRole,
    ToolCall,
    ToolCallFunction,
    ToolResult,
)


class TestMessage:
    def test_create_user_message(self):
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.id is not None
        assert isinstance(msg.timestamp, datetime)

    def test_create_assistant_message_with_tool_calls(self):
        tool_call = ToolCall(
            function=ToolCallFunction(
                name="test_tool",
                arguments='{"arg": "value"}',
            )
        )
        msg = Message(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[tool_call],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "test_tool"


class TestExecutionContext:
    def test_create_context(self):
        ctx = ExecutionContext()
        assert ctx.execution_id is not None
        assert ctx.state == AgentState.IDLE
        assert ctx.iteration == 0
        assert len(ctx.messages) == 0

    def test_add_message(self):
        ctx = ExecutionContext()
        msg = Message(role=MessageRole.USER, content="Test")
        ctx.add_message(msg)
        assert len(ctx.messages) == 1
        assert ctx.messages[0].content == "Test"

    def test_add_tool_result(self):
        ctx = ExecutionContext()
        result = ToolResult(
            tool_call_id="test-123",
            output="Success",
        )
        ctx.add_tool_result(result)
        assert len(ctx.tool_results) == 1


class TestAgentConfig:
    def test_create_config(self):
        config = AgentConfig(
            name="test-agent",
            model="gpt-4",
        )
        assert config.name == "test-agent"
        assert config.model == "gpt-4"
        assert config.provider == "openai"
        assert config.max_iterations == 10
        assert config.temperature == 0.0

    def test_config_immutability(self):
        config = AgentConfig(name="test", model="gpt-4")
        with pytest.raises(Exception):
            config.name = "new-name"  # type: ignore
