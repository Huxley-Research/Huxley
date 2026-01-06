"""
Core type definitions for Huxley.

All fundamental data structures used across the framework are defined here.
These types are designed to be:
- Immutable where possible
- Serializable (JSON-compatible)
- Compatible with OpenAI API schemas
- Extensible without breaking changes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field
from ulid import ULID


def generate_id() -> str:
    """Generate a unique, sortable identifier."""
    return str(ULID())


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """
    A tool invocation requested by the model.

    Compatible with OpenAI's tool_call schema.
    """

    id: str = Field(default_factory=generate_id)
    type: Literal["function"] = "function"
    function: ToolCallFunction

    class Config:
        frozen = True


class ToolCallFunction(BaseModel):
    """Function details within a tool call."""

    name: str
    arguments: str  # JSON-encoded arguments

    class Config:
        frozen = True


class ToolResult(BaseModel):
    """
    Result of executing a tool.

    Contains the output, any errors, and execution metadata.
    """

    tool_call_id: str
    output: Any
    error: str | None = None
    execution_time_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True


class Message(BaseModel):
    """
    A message in a conversation.

    Compatible with OpenAI's chat completion message schema,
    extended with Huxley-specific metadata.
    """

    id: str = Field(default_factory=generate_id)
    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool response messages
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True


class AgentState(str, Enum):
    """Current state of an agent execution."""

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_FOR_TOOL = "waiting_for_tool"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class AgentConfig:
    """
    Configuration for an agent instance.

    Defines the model, tools, and behavioral parameters.
    """

    name: str
    model: str  # Model identifier (e.g., "gpt-4", "claude-3-opus")
    provider: str = "openai"  # LLM provider
    system_prompt: str | None = None
    tools: list[str] = field(default_factory=list)  # Tool names to enable
    max_iterations: int = 10
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout_seconds: float = 300.0
    retry_attempts: int = 3
    verify_outputs: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """
    Runtime context for an agent execution.

    Contains all state needed for a single execution run,
    including conversation history, tool results, and metrics.
    """

    execution_id: str = field(default_factory=generate_id)
    agent_config: AgentConfig | None = None
    messages: list[Message] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    state: AgentState = AgentState.IDLE
    iteration: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    parent_execution_id: str | None = None  # For sub-agent spawning
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    def add_tool_result(self, result: ToolResult) -> None:
        """Record a tool execution result."""
        self.tool_results.append(result)
        self.updated_at = datetime.utcnow()


class CompletionChoice(BaseModel):
    """A single completion choice from an LLM response."""

    index: int
    message: Message
    finish_reason: str | None = None

    class Config:
        frozen = True


class CompletionUsage(BaseModel):
    """Token usage statistics for a completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    class Config:
        frozen = True


class CompletionResponse(BaseModel):
    """
    Response from an LLM completion request.

    Compatible with OpenAI's chat completion response schema.
    """

    id: str = Field(default_factory=generate_id)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage | None = None

    class Config:
        frozen = True


class StreamDelta(BaseModel):
    """Delta content in a streaming response."""

    role: MessageRole | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

    class Config:
        frozen = True


class StreamChoice(BaseModel):
    """A single choice in a streaming response chunk."""

    index: int
    delta: StreamDelta
    finish_reason: str | None = None

    class Config:
        frozen = True


class StreamChunk(BaseModel):
    """
    A chunk in a streaming completion response.

    Compatible with OpenAI's streaming response schema.
    """

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]

    class Config:
        frozen = True
