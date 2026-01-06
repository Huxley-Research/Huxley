"""
Anthropic Claude LLM provider.

Supports Claude models via the Anthropic API.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
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

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3-5-sonnet": 200000,
    "claude-3-5-haiku": 200000,
    "claude-2.1": 200000,
    "claude-2.0": 100000,
}


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude LLM provider.

    Converts between Huxley's OpenAI-compatible format and Anthropic's API.
    """

    def __init__(
        self,
        config: LLMProviderConfig | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url or "https://api.anthropic.com"
        self._timeout = timeout
        self._max_retries = max_retries

        if config:
            if config.api_key:
                self._api_key = config.api_key.get_secret_value()
            if config.base_url:
                self._base_url = config.base_url
            self._timeout = config.timeout
            self._max_retries = config.max_retries

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "x-api-key": self._api_key or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=self._timeout,
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

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
        """Generate a completion using the Anthropic API."""
        system_prompt, converted_messages = self._convert_messages(messages)

        request_body: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }

        if system_prompt:
            request_body["system"] = system_prompt

        if stop:
            request_body["stop_sequences"] = stop

        if tools:
            request_body["tools"] = self._convert_tools(tools)

        try:
            response = await self._client.post("/v1/messages", json=request_body)
            response.raise_for_status()
            data = response.json()
            return self._convert_response(data, model)

        except httpx.HTTPStatusError as e:
            self._handle_error(e)
        except Exception as e:
            raise LLMConnectionError(str(e), cause=e)

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
        """Stream a completion using the Anthropic API."""
        system_prompt, converted_messages = self._convert_messages(messages)

        request_body: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
            "stream": True,
        }

        if system_prompt:
            request_body["system"] = system_prompt

        if stop:
            request_body["stop_sequences"] = stop

        if tools:
            request_body["tools"] = self._convert_tools(tools)

        try:
            async with self._client.stream(
                "POST", "/v1/messages", json=request_body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        chunk = self._convert_stream_event(data, model)
                        if chunk:
                            yield chunk

        except httpx.HTTPStatusError as e:
            self._handle_error(e)
        except Exception as e:
            raise LLMConnectionError(str(e), cause=e)

    def get_context_window(self, model: str) -> int:
        for known_model, window in MODEL_CONTEXT_WINDOWS.items():
            if model.startswith(known_model) or known_model in model:
                return window
        return -1

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert Huxley messages to Anthropic format."""
        system_prompt = None
        converted = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
                continue

            role = "user" if msg.role == MessageRole.USER else "assistant"

            if msg.role == MessageRole.TOOL:
                # Tool results go as user messages with tool_result content
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content or "",
                    }],
                })
            elif msg.tool_calls:
                # Assistant with tool calls
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments),
                    })
                converted.append({"role": role, "content": content})
            else:
                converted.append({
                    "role": role,
                    "content": msg.content or "",
                })

        return system_prompt, converted

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                converted.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object"}),
                })
        return converted

    def _convert_response(
        self, data: dict[str, Any], model: str
    ) -> CompletionResponse:
        """Convert Anthropic response to Huxley format."""
        content_parts = data.get("content", [])
        text_content = ""
        tool_calls = []

        for part in content_parts:
            if part["type"] == "text":
                text_content += part["text"]
            elif part["type"] == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=part["id"],
                        type="function",
                        function=ToolCallFunction(
                            name=part["name"],
                            arguments=json.dumps(part["input"]),
                        ),
                    )
                )

        message = Message(
            role=MessageRole.ASSISTANT,
            content=text_content if text_content else None,
            tool_calls=tool_calls if tool_calls else None,
        )

        usage = None
        if "usage" in data:
            usage = CompletionUsage(
                prompt_tokens=data["usage"].get("input_tokens", 0),
                completion_tokens=data["usage"].get("output_tokens", 0),
                total_tokens=(
                    data["usage"].get("input_tokens", 0)
                    + data["usage"].get("output_tokens", 0)
                ),
            )

        return CompletionResponse(
            id=data.get("id", ""),
            model=model,
            choices=[
                CompletionChoice(
                    index=0,
                    message=message,
                    finish_reason=data.get("stop_reason"),
                )
            ],
            usage=usage,
        )

    def _convert_stream_event(
        self, data: dict[str, Any], model: str
    ) -> StreamChunk | None:
        """Convert Anthropic stream event to Huxley format."""
        event_type = data.get("type")

        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "text_delta":
                return StreamChunk(
                    id=data.get("index", 0),
                    model=model,
                    created=0,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=StreamDelta(content=delta.get("text", "")),
                            finish_reason=None,
                        )
                    ],
                )

        if event_type == "message_stop":
            return StreamChunk(
                id="",
                model=model,
                created=0,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=StreamDelta(),
                        finish_reason="stop",
                    )
                ],
            )

        return None

    def _handle_error(self, error: httpx.HTTPStatusError) -> None:
        """Convert HTTP errors to Huxley exceptions."""
        status = error.response.status_code

        if status == 429:
            retry_after = error.response.headers.get("retry-after")
            raise LLMRateLimitError(
                str(error),
                retry_after=float(retry_after) if retry_after else None,
                cause=error,
            )

        if status == 400:
            raise LLMContextLengthError(str(error), cause=error)

        raise LLMInvalidResponseError(str(error), cause=error)

    async def __aenter__(self) -> "AnthropicProvider":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.aclose()
