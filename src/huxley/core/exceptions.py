"""
Custom exceptions for Huxley.

All exceptions inherit from HuxleyError for consistent error handling.
"""

from __future__ import annotations

from typing import Any


class HuxleyError(Exception):
    """Base exception for all Huxley errors."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message


# Configuration errors
class ConfigurationError(HuxleyError):
    """Invalid or missing configuration."""

    pass


# LLM errors
class LLMError(HuxleyError):
    """Base class for LLM-related errors."""

    pass


class LLMConnectionError(LLMError):
    """Failed to connect to LLM provider."""

    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class LLMContextLengthError(LLMError):
    """Context length exceeded."""

    def __init__(
        self,
        message: str = "Context length exceeded",
        *,
        max_tokens: int | None = None,
        requested_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.max_tokens = max_tokens
        self.requested_tokens = requested_tokens


class LLMInvalidResponseError(LLMError):
    """LLM returned an invalid or unparseable response."""

    pass


# Tool errors
class ToolError(HuxleyError):
    """Base class for tool-related errors."""

    pass


class ToolNotFoundError(ToolError):
    """Requested tool does not exist."""

    def __init__(self, tool_name: str, **kwargs: Any) -> None:
        super().__init__(f"Tool not found: {tool_name}", **kwargs)
        self.tool_name = tool_name


class ToolExecutionError(ToolError):
    """Tool execution failed."""

    def __init__(
        self,
        tool_name: str,
        message: str,
        *,
        execution_time_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(f"Tool '{tool_name}' failed: {message}", **kwargs)
        self.tool_name = tool_name
        self.execution_time_ms = execution_time_ms


class ToolValidationError(ToolError):
    """Tool input validation failed."""

    def __init__(
        self,
        tool_name: str,
        message: str,
        *,
        validation_errors: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(f"Tool '{tool_name}' validation failed: {message}", **kwargs)
        self.tool_name = tool_name
        self.validation_errors = validation_errors or []


class ToolTimeoutError(ToolError):
    """Tool execution timed out."""

    def __init__(
        self,
        tool_name: str,
        timeout_seconds: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Tool '{tool_name}' timed out after {timeout_seconds}s",
            **kwargs,
        )
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds


# Agent errors
class AgentError(HuxleyError):
    """Base class for agent-related errors."""

    pass


class AgentMaxIterationsError(AgentError):
    """Agent exceeded maximum iterations."""

    def __init__(
        self,
        max_iterations: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Agent exceeded maximum iterations ({max_iterations})",
            **kwargs,
        )
        self.max_iterations = max_iterations


class AgentTimeoutError(AgentError):
    """Agent execution timed out."""

    def __init__(
        self,
        timeout_seconds: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Agent execution timed out after {timeout_seconds}s",
            **kwargs,
        )
        self.timeout_seconds = timeout_seconds


class AgentCancelledError(AgentError):
    """Agent execution was cancelled."""

    pass


# Verification errors
class VerificationError(HuxleyError):
    """Base class for verification errors."""

    pass


class OutputVerificationError(VerificationError):
    """Output failed verification checks."""

    def __init__(
        self,
        message: str,
        *,
        failures: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.failures = failures or []


# Compute errors
class ComputeError(HuxleyError):
    """Base class for compute/worker errors."""

    pass


class WorkerUnavailableError(ComputeError):
    """No workers available to handle task."""

    pass


class TaskQueueError(ComputeError):
    """Error interacting with task queue."""

    pass
