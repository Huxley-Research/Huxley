"""
OpenAI-compatible chat completions endpoint.

Provides /v1/chat/completions that matches OpenAI's API schema,
enabling drop-in replacement for OpenAI SDK users.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from huxley.core.logging import get_logger
from huxley.core.types import Message, MessageRole, generate_id
from huxley.llm.client import LLMClient

logger = get_logger(__name__)

router = APIRouter()


# Request/Response models matching OpenAI schema
class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""

    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=1.0, ge=0, le=2)
    max_tokens: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stop: list[str] | str | None = None
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    n: int = Field(default=1, ge=1, le=10)
    user: str | None = None


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str | None


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage | None = None


@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """
    Create a chat completion.

    This endpoint is compatible with the OpenAI chat completions API,
    allowing Huxley to serve as a drop-in replacement.
    """
    logger.info(
        "chat_completion_request",
        model=request.model,
        message_count=len(request.messages),
        stream=request.stream,
    )

    # Convert messages to internal format
    messages = []
    for msg in request.messages:
        messages.append(
            Message(
                role=MessageRole(msg.role),
                content=msg.content,
                name=msg.name,
                tool_call_id=msg.tool_call_id,
            )
        )

    # Get LLM client
    client = LLMClient()

    # Handle streaming
    if request.stream:
        return StreamingResponse(
            _stream_completion(client, messages, request),
            media_type="text/event-stream",
        )

    # Non-streaming completion
    try:
        response = await client.complete(
            messages=messages,
            model=request.model,
            tools=request.tools,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop=request.stop if isinstance(request.stop, list) else (
                [request.stop] if request.stop else None
            ),
        )

        # Convert to OpenAI format
        choices = []
        for choice in response.choices:
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]

            choices.append(
                ChatCompletionChoice(
                    index=choice.index,
                    message=ChatMessage(
                        role=choice.message.role.value,
                        content=choice.message.content,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=choice.finish_reason,
                )
            )

        usage = None
        if response.usage:
            usage = ChatCompletionUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return ChatCompletionResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=choices,
            usage=usage,
        )

    except Exception as e:
        logger.error("chat_completion_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_completion(
    client: LLMClient,
    messages: list[Message],
    request: ChatCompletionRequest,
):
    """Generate SSE stream for chat completion."""
    try:
        async for chunk in client.stream(
            messages=messages,
            model=request.model,
            tools=request.tools,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop=request.stop if isinstance(request.stop, list) else (
                [request.stop] if request.stop else None
            ),
        ):
            # Convert to OpenAI streaming format
            choices = []
            for choice in chunk.choices:
                delta: dict[str, Any] = {}
                if choice.delta.role:
                    delta["role"] = choice.delta.role.value
                if choice.delta.content:
                    delta["content"] = choice.delta.content
                if choice.delta.tool_calls:
                    delta["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.delta.tool_calls
                    ]

                choices.append({
                    "index": choice.index,
                    "delta": delta,
                    "finish_reason": choice.finish_reason,
                })

            data = {
                "id": chunk.id,
                "object": "chat.completion.chunk",
                "created": chunk.created,
                "model": chunk.model,
                "choices": choices,
            }

            yield f"data: {json.dumps(data)}\n\n"

        # Send done marker
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error("stream_error", error=str(e))
        error_data = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"
