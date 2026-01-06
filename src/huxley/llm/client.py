"""
LLM client with provider abstraction.

Provides a unified interface for interacting with any supported LLM provider.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, AsyncIterator

from huxley.core.config import get_config
from huxley.core.exceptions import ConfigurationError
from huxley.core.logging import get_logger
from huxley.core.types import (
    CompletionResponse,
    Message,
    StreamChunk,
)
from huxley.llm.providers.base import BaseLLMProvider
from huxley.llm.providers.openai import OpenAIProvider

logger = get_logger(__name__)

# Provider registry
_PROVIDERS: dict[str, type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
}

# Lazy import for optional providers
def _get_anthropic_provider() -> type[BaseLLMProvider]:
    from huxley.llm.providers.anthropic import AnthropicProvider
    return AnthropicProvider


def register_provider(name: str, provider_class: type[BaseLLMProvider]) -> None:
    """
    Register a custom LLM provider.

    Args:
        name: Unique provider identifier
        provider_class: Provider class implementing BaseLLMProvider
    """
    _PROVIDERS[name] = provider_class
    logger.info("provider_registered", provider=name)


class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    Handles provider selection, configuration, and provides a consistent
    interface for all LLM operations.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        **provider_kwargs: Any,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            provider: Provider name (defaults to config)
            model: Default model (defaults to config)
            **provider_kwargs: Additional provider-specific configuration
        """
        config = get_config()
        self._provider_name = provider or config.default_provider
        self._default_model = model or config.default_model

        # Get provider configuration
        provider_config = config.get_provider_config(self._provider_name)

        # Instantiate provider
        self._provider = self._create_provider(
            self._provider_name,
            provider_config,
            **provider_kwargs,
        )

        logger.info(
            "llm_client_initialized",
            provider=self._provider_name,
            model=self._default_model,
        )

    def _create_provider(
        self,
        provider_name: str,
        config: Any,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """Create a provider instance."""
        # Handle lazy-loaded providers
        if provider_name == "anthropic":
            provider_class = _get_anthropic_provider()
        elif provider_name in _PROVIDERS:
            provider_class = _PROVIDERS[provider_name]
        else:
            raise ConfigurationError(
                f"Unknown LLM provider: {provider_name}",
                details={"available": list(_PROVIDERS.keys()) + ["anthropic"]},
            )

        return provider_class(config=config, **kwargs)

    @property
    def provider(self) -> BaseLLMProvider:
        """Get the underlying provider."""
        return self._provider

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def default_model(self) -> str:
        """Get the default model."""
        return self._default_model

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Generate a completion.

        Args:
            messages: Conversation history
            model: Model to use (defaults to client default)
            tools: Tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Provider-specific parameters

        Returns:
            CompletionResponse with generated content
        """
        return await self._provider.complete(
            messages=messages,
            model=model or self._default_model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs,
        )

    async def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a completion.

        Args:
            messages: Conversation history
            model: Model to use
            tools: Tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Provider-specific parameters

        Yields:
            StreamChunk objects as they arrive
        """
        async for chunk in self._provider.stream(
            messages=messages,
            model=model or self._default_model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs,
        ):
            yield chunk

    async def count_tokens(
        self,
        messages: list[Message],
        model: str | None = None,
    ) -> int:
        """
        Count tokens in messages.

        Args:
            messages: Messages to count
            model: Model to use for tokenization

        Returns:
            Token count, or -1 if unknown
        """
        return await self._provider.count_tokens(
            messages=messages,
            model=model or self._default_model,
        )

    def get_context_window(self, model: str | None = None) -> int:
        """
        Get context window size for a model.

        Args:
            model: Model to check

        Returns:
            Context window size, or -1 if unknown
        """
        return self._provider.get_context_window(model or self._default_model)

    async def health_check(self) -> bool:
        """Check provider health."""
        return await self._provider.health_check()


@lru_cache
def get_client(
    provider: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """
    Get a cached LLM client instance.

    Args:
        provider: Provider name
        model: Default model

    Returns:
        LLMClient instance
    """
    return LLMClient(provider=provider, model=model)
