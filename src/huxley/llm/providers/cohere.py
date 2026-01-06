"""
Cohere LLM provider.

Supports Command A family models via the Cohere API v2.
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

# Command A models context windows
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "command-a-03-2025": 256000,
    "command-a": 256000,
    "command-a-reasoning": 256000,
    "command-a-vision": 256000,
    "command-r-plus": 128000,
    "command-r": 128000,
    "command-r7b": 128000,
}


class CohereProvider(BaseLLMProvider):
    """
    Cohere LLM provider.

    Converts between Huxley's OpenAI-compatible format and Cohere's Chat API v2.
    Supports Command A, Command A Reasoning, and Command A Vision models.
    """

    def __init__(
        self,
        config: LLMProviderConfig | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url or "https://api.cohere.com"
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
                "Authorization": f"Bearer {self._api_key or ''}",
                "Content-Type": "application/json",
                "X-Client-Name": "huxley",
            },
            timeout=self._timeout,
        )

    @property
    def provider_name(self) -> str:
        return "cohere"

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
        temperature: float = 0.3,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion using the Cohere Chat API v2."""
        converted_messages = self._convert_messages(messages)

        request_body: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "temperature": temperature,
            "stream": False,
        }

        if max_tokens:
            request_body["max_tokens"] = max_tokens

        if stop:
            request_body["stop_sequences"] = stop

        if tools:
            request_body["tools"] = self._convert_tools(tools)

        # Enable reasoning for reasoning models
        if "reasoning" in model.lower():
            request_body["thinking"] = {"enabled": True}

        try:
            response = await self._client.post("/v2/chat", json=request_body)
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
        temperature: float = 0.3,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion using the Cohere Chat API v2."""
        converted_messages = self._convert_messages(messages)

        request_body: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            request_body["max_tokens"] = max_tokens

        if stop:
            request_body["stop_sequences"] = stop

        if tools:
            request_body["tools"] = self._convert_tools(tools)

        if "reasoning" in model.lower():
            request_body["thinking"] = {"enabled": True}

        try:
            async with self._client.stream(
                "POST", "/v2/chat", json=request_body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            chunk = self._convert_stream_event(data, model)
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            self._handle_error(e)
        except Exception as e:
            raise LLMConnectionError(str(e), cause=e)

    def get_context_window(self, model: str) -> int:
        """Get the context window size for a model."""
        for known_model, window in MODEL_CONTEXT_WINDOWS.items():
            if model.startswith(known_model) or known_model in model:
                return window
        return 128000  # Default for Command models

    def _convert_messages(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        """Convert Huxley messages to Cohere v2 format."""
        converted = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # System messages go first with role "system"
                converted.append({
                    "role": "system",
                    "content": msg.content or "",
                })
            elif msg.role == MessageRole.USER:
                converted.append({
                    "role": "user",
                    "content": msg.content or "",
                })
            elif msg.role == MessageRole.ASSISTANT:
                if msg.tool_calls:
                    # Assistant with tool calls
                    content = []
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_call",
                            "id": tc.id,
                            "name": tc.function.name,
                            "parameters": json.loads(tc.function.arguments),
                        })
                    converted.append({"role": "assistant", "content": content})
                else:
                    converted.append({
                        "role": "assistant",
                        "content": msg.content or "",
                    })
            elif msg.role == MessageRole.TOOL:
                # Tool results
                converted.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content or "",
                })

        return converted

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Cohere format."""
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                converted.append({
                    "type": "function",
                    "function": {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {"type": "object"}),
                    },
                })
        return converted

    def _convert_response(
        self, data: dict[str, Any], model: str
    ) -> CompletionResponse:
        """Convert Cohere response to Huxley format."""
        message_data = data.get("message", {})
        content_parts = message_data.get("content", [])
        
        text_content = ""
        tool_calls = []

        for part in content_parts:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_content += part.get("text", "")
                elif part.get("type") == "tool_call":
                    tool_calls.append(
                        ToolCall(
                            id=part.get("id", ""),
                            type="function",
                            function=ToolCallFunction(
                                name=part.get("name", ""),
                                arguments=json.dumps(part.get("parameters", {})),
                            ),
                        )
                    )
            elif isinstance(part, str):
                text_content += part

        message = Message(
            role=MessageRole.ASSISTANT,
            content=text_content if text_content else None,
            tool_calls=tool_calls if tool_calls else None,
        )

        usage = None
        if "usage" in data:
            usage_data = data["usage"]
            tokens = usage_data.get("tokens", {})
            billed = usage_data.get("billed_units", {})
            prompt_tokens = tokens.get("input_tokens", billed.get("input_tokens", 0))
            completion_tokens = tokens.get("output_tokens", billed.get("output_tokens", 0))
            usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        finish_reason_map = {
            "COMPLETE": "stop",
            "MAX_TOKENS": "length",
            "STOP_SEQUENCE": "stop",
            "TOOL_CALL": "tool_calls",
            "ERROR": "error",
        }
        finish_reason = finish_reason_map.get(
            data.get("finish_reason", "COMPLETE"), "stop"
        )

        return CompletionResponse(
            id=data.get("id", ""),
            model=model,
            choices=[
                CompletionChoice(
                    index=0,
                    message=message,
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    def _convert_stream_event(
        self, data: dict[str, Any], model: str
    ) -> StreamChunk | None:
        """Convert Cohere stream event to Huxley format."""
        event_type = data.get("type")

        if event_type == "content-delta":
            delta = data.get("delta", {})
            if delta.get("type") == "text":
                return StreamChunk(
                    id=data.get("index", ""),
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

        if event_type == "message-end":
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
            # Could be context length or invalid request
            try:
                body = error.response.json()
                if "context" in body.get("message", "").lower():
                    raise LLMContextLengthError(str(error), cause=error)
            except json.JSONDecodeError:
                pass
            raise LLMInvalidResponseError(str(error), cause=error)

        if status == 401:
            raise LLMConnectionError("Invalid API key", cause=error)

        if status == 403:
            raise LLMConnectionError("Access forbidden", cause=error)

        raise LLMInvalidResponseError(str(error), cause=error)

    async def __aenter__(self) -> "CohereProvider":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.aclose()
