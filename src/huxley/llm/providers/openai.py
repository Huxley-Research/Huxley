"""
OpenAI-compatible LLM provider.

This provider works with:
- OpenAI API
- Azure OpenAI
- Any OpenAI-compatible endpoint (vLLM, Ollama, etc.)
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from huxley.core.config import LLMProviderConfig
from huxley.core.exceptions import (
    LLMConnectionError,
    LLMContextLengthError,
    LLMInvalidResponseError,
    LLMRateLimitError,
)
from huxley.core.logging import get_logger
from huxley.core.types import (
    CompletionChoice,
    CompletionResponse,
    CompletionUsage,
    Message,
    MessageRole,
    StreamChunk,
    StreamChoice,
    StreamDelta,
    ToolCall,
    ToolCallFunction,
)
from huxley.llm.providers.base import BaseLLMProvider

logger = get_logger(__name__)

# Model context windows (approximate)
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
}


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI-compatible LLM provider.

    Supports the OpenAI chat completions API and any compatible endpoint.
    """

    def __init__(
        self,
        config: LLMProviderConfig | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the OpenAI provider.

        Args:
            config: Provider configuration (takes precedence)
            api_key: OpenAI API key
            base_url: Custom API base URL
            organization: OpenAI organization ID
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        if config:
            api_key = config.api_key.get_secret_value() if config.api_key else api_key
            base_url = config.base_url or base_url
            organization = config.organization or organization
            timeout = config.timeout
            max_retries = config.max_retries

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
            max_retries=0,  # We handle retries ourselves
        )
        self._max_retries = max_retries
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def supported_models(self) -> list[str]:
        return list(MODEL_CONTEXT_WINDOWS.keys())

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
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
        """Generate a completion using the OpenAI API."""
        try:
            request_messages = self._convert_messages(messages)
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": request_messages,
                "temperature": temperature,
            }

            if max_tokens:
                request_kwargs["max_tokens"] = max_tokens
            if stop:
                request_kwargs["stop"] = stop
            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = "auto"

            # Merge any additional kwargs
            request_kwargs.update(kwargs)

            logger.debug(
                "openai_request",
                model=model,
                message_count=len(messages),
                has_tools=bool(tools),
            )

            response: ChatCompletion = await self._client.chat.completions.create(
                **request_kwargs
            )

            return self._convert_response(response)

        except Exception as e:
            self._handle_error(e)

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
        """Stream a completion using the OpenAI API."""
        try:
            request_messages = self._convert_messages(messages)
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": request_messages,
                "temperature": temperature,
                "stream": True,
            }

            if max_tokens:
                request_kwargs["max_tokens"] = max_tokens
            if stop:
                request_kwargs["stop"] = stop
            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = "auto"

            request_kwargs.update(kwargs)

            logger.debug(
                "openai_stream_request",
                model=model,
                message_count=len(messages),
            )

            stream = await self._client.chat.completions.create(**request_kwargs)

            async for chunk in stream:  # type: ignore
                yield self._convert_stream_chunk(chunk)

        except Exception as e:
            self._handle_error(e)

    async def count_tokens(
        self,
        messages: list[Message],
        model: str,
    ) -> int:
        """
        Count tokens using tiktoken.

        Falls back to -1 if tiktoken is not available or model not supported.
        """
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(model)
            num_tokens = 0

            for message in messages:
                num_tokens += 4  # Message overhead
                if message.content:
                    num_tokens += len(encoding.encode(message.content))
                if message.name:
                    num_tokens += len(encoding.encode(message.name))

            num_tokens += 2  # Reply priming
            return num_tokens

        except Exception:
            return -1

    def get_context_window(self, model: str) -> int:
        """Get context window for OpenAI models."""
        # Check exact match
        if model in MODEL_CONTEXT_WINDOWS:
            return MODEL_CONTEXT_WINDOWS[model]

        # Check prefix match
        for known_model, window in MODEL_CONTEXT_WINDOWS.items():
            if model.startswith(known_model):
                return window

        return -1

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Huxley messages to OpenAI format."""
        result = []
        for msg in messages:
            converted: dict[str, Any] = {
                "role": msg.role.value,
            }

            if msg.content is not None:
                converted["content"] = msg.content

            if msg.name:
                converted["name"] = msg.name

            if msg.tool_calls:
                converted["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

            if msg.tool_call_id:
                converted["tool_call_id"] = msg.tool_call_id

            result.append(converted)

        return result

    def _convert_response(self, response: ChatCompletion) -> CompletionResponse:
        """Convert OpenAI response to Huxley format."""
        choices = []
        for choice in response.choices:
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        type="function",
                        function=ToolCallFunction(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
                    for tc in choice.message.tool_calls
                ]

            message = Message(
                role=MessageRole(choice.message.role),
                content=choice.message.content,
                tool_calls=tool_calls,
            )

            choices.append(
                CompletionChoice(
                    index=choice.index,
                    message=message,
                    finish_reason=choice.finish_reason,
                )
            )

        usage = None
        if response.usage:
            usage = CompletionUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return CompletionResponse(
            id=response.id,
            model=response.model,
            choices=choices,
            usage=usage,
        )

    def _convert_stream_chunk(self, chunk: ChatCompletionChunk) -> StreamChunk:
        """Convert OpenAI stream chunk to Huxley format."""
        choices = []
        for choice in chunk.choices:
            delta = choice.delta

            tool_calls = None
            if delta.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id or "",
                        type="function",
                        function=ToolCallFunction(
                            name=tc.function.name if tc.function else "",
                            arguments=tc.function.arguments if tc.function else "",
                        ),
                    )
                    for tc in delta.tool_calls
                ]

            stream_delta = StreamDelta(
                role=MessageRole(delta.role) if delta.role else None,
                content=delta.content,
                tool_calls=tool_calls,
            )

            choices.append(
                StreamChoice(
                    index=choice.index,
                    delta=stream_delta,
                    finish_reason=choice.finish_reason,
                )
            )

        return StreamChunk(
            id=chunk.id,
            model=chunk.model,
            created=chunk.created,
            choices=choices,
        )

    def _handle_error(self, error: Exception) -> None:
        """Convert provider errors to Huxley exceptions."""
        from openai import (
            APIConnectionError,
            APIStatusError,
            RateLimitError,
        )

        if isinstance(error, RateLimitError):
            retry_after = None
            if hasattr(error, "response") and error.response:
                retry_after_header = error.response.headers.get("retry-after")
                if retry_after_header:
                    retry_after = float(retry_after_header)
            raise LLMRateLimitError(
                str(error),
                retry_after=retry_after,
                cause=error,
            )

        if isinstance(error, APIConnectionError):
            raise LLMConnectionError(str(error), cause=error)

        if isinstance(error, APIStatusError):
            # Check for context length error
            if "context_length" in str(error).lower() or error.status_code == 400:
                raise LLMContextLengthError(str(error), cause=error)
            raise LLMInvalidResponseError(str(error), cause=error)

        if isinstance(error, (httpx.TimeoutException, httpx.ConnectError)):
            raise LLMConnectionError(str(error), cause=error)

        # Re-raise unknown errors
        raise error
