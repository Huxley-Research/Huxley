"""
Tool management endpoints.

Provides APIs for listing, describing, and invoking tools.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from huxley.core.logging import get_logger
from huxley.core.types import ToolCall, ToolCallFunction
from huxley.tools.executor import get_executor
from huxley.tools.registry import get_registry

logger = get_logger(__name__)

router = APIRouter()


class ToolInfo(BaseModel):
    """Information about a registered tool."""

    name: str
    description: str
    parameters: list[dict[str, Any]]
    tags: list[str]
    is_async: bool


class ToolInvokeRequest(BaseModel):
    """Request to invoke a tool directly."""

    name: str = Field(..., description="Tool name")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    timeout: float = Field(default=30.0, description="Timeout in seconds")


class ToolInvokeResponse(BaseModel):
    """Response from tool invocation."""

    tool_name: str
    output: Any
    error: str | None
    execution_time_ms: float | None


@router.get("/tools")
async def list_tools(
    tag: str | None = None,
) -> list[ToolInfo]:
    """
    List all registered tools.

    Optionally filter by tag.
    """
    registry = get_registry()

    tags_filter = {tag} if tag else None
    tools = registry.list(tags=tags_filter)

    return [
        ToolInfo(
            name=tool.name,
            description=tool.description,
            parameters=[
                {
                    "name": p.name,
                    "type": p.type.__name__ if hasattr(p.type, "__name__") else str(p.type),
                    "description": p.description,
                    "required": p.required,
                }
                for p in tool.parameters
            ],
            tags=list(tool.tags),
            is_async=tool.is_async,
        )
        for tool in tools
    ]


@router.get("/tools/{tool_name}")
async def get_tool(tool_name: str) -> ToolInfo:
    """
    Get information about a specific tool.
    """
    registry = get_registry()

    if tool_name not in registry:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

    tool = registry.get(tool_name)

    return ToolInfo(
        name=tool.name,
        description=tool.description,
        parameters=[
            {
                "name": p.name,
                "type": p.type.__name__ if hasattr(p.type, "__name__") else str(p.type),
                "description": p.description,
                "required": p.required,
            }
            for p in tool.parameters
        ],
        tags=list(tool.tags),
        is_async=tool.is_async,
    )


@router.get("/tools/{tool_name}/schema")
async def get_tool_schema(tool_name: str) -> dict[str, Any]:
    """
    Get the OpenAI-compatible schema for a tool.

    This can be used directly in chat completion requests.
    """
    registry = get_registry()

    if tool_name not in registry:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

    tool = registry.get(tool_name)
    return tool.to_openai_schema()


@router.post("/tools/invoke")
async def invoke_tool(request: ToolInvokeRequest) -> ToolInvokeResponse:
    """
    Invoke a tool directly.

    This allows testing tools outside of agent execution.
    """
    registry = get_registry()

    if request.name not in registry:
        raise HTTPException(status_code=404, detail=f"Tool not found: {request.name}")

    executor = get_executor()

    # Create a tool call
    tool_call = ToolCall(
        function=ToolCallFunction(
            name=request.name,
            arguments=json.dumps(request.arguments),
        )
    )

    logger.info(
        "tool_invoke_request",
        tool=request.name,
        timeout=request.timeout,
    )

    result = await executor.execute(tool_call, timeout=request.timeout)

    return ToolInvokeResponse(
        tool_name=request.name,
        output=result.output,
        error=result.error,
        execution_time_ms=result.execution_time_ms,
    )


@router.get("/tools/schemas")
async def get_all_schemas(
    tools: str | None = None,
) -> list[dict[str, Any]]:
    """
    Get OpenAI-compatible schemas for multiple tools.

    Args:
        tools: Comma-separated list of tool names (all if not specified)
    """
    registry = get_registry()

    tool_names = None
    if tools:
        tool_names = [t.strip() for t in tools.split(",")]

    return registry.get_openai_schemas(tool_names)
