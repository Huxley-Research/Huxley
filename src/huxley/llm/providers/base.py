"""
Base LLM provider interface.

All LLM providers must implement this interface to ensure
consistent behavior across different backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from huxley.core.types import (
    CompletionResponse,
    Message,
    StreamChunk,
)


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implementations must support:
    - Synchronous and asynchronous completion
    - Streaming responses
    - Tool/function calling
    - Token counting (optional but recommended)
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique identifier for this provider."""
        ...

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """List of model identifiers this provider supports."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Generate a completion for the given messages.

        Args:
            messages: Conversation history
            model: Model identifier
            tools: Tool definitions (OpenAI function calling format)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Provider-specific parameters

        Returns:
            CompletionResponse with generated content

        Raises:
            LLMError: On any provider error
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        model: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a completion for the given messages.

        Args:
            messages: Conversation history
            model: Model identifier
            tools: Tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Provider-specific parameters

        Yields:
            StreamChunk objects as they arrive

        Raises:
            LLMError: On any provider error
        """
        ...

    async def count_tokens(
        self,
        messages: list[Message],
        model: str,
    ) -> int:
        """
        Count tokens in messages for the given model.

        Default implementation returns -1 (unknown).
        Providers should override for accurate counting.

        Args:
            messages: Messages to count
            model: Model to use for tokenization

        Returns:
            Token count, or -1 if unknown
        """
        return -1

    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and reachable.

        Returns:
            True if healthy, False otherwise
        """
        return True

    def supports_tool_calling(self, model: str) -> bool:
        """
        Check if the given model supports tool/function calling.

        Args:
            model: Model identifier

        Returns:
            True if tool calling is supported
        """
        return True  # Assume support by default

    def get_context_window(self, model: str) -> int:
        """
        Get the context window size for the given model.

        Args:
            model: Model identifier

        Returns:
            Maximum context length in tokens, or -1 if unknown
        """
        return -1
