"""
Agent execution endpoints.

Provides APIs for running agents, managing executions,
and retrieving results.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from huxley.agents.base import Agent
from huxley.core.logging import get_logger
from huxley.core.types import AgentConfig, AgentState, ExecutionContext

logger = get_logger(__name__)

router = APIRouter()

# In-memory execution store (replace with proper store in production)
_executions: dict[str, ExecutionContext] = {}


class AgentRunRequest(BaseModel):
    """Request to run an agent."""

    name: str = Field(default="default", description="Agent name")
    query: str = Field(..., description="Query or task for the agent")
    model: str = Field(default="gpt-4", description="Model to use")
    provider: str = Field(default="openai", description="LLM provider")
    system_prompt: str | None = Field(default=None, description="System prompt")
    tools: list[str] = Field(default_factory=list, description="Tools to enable")
    max_iterations: int = Field(default=10, ge=1, le=50)
    temperature: float = Field(default=0.0, ge=0, le=2)
    timeout: float = Field(default=300.0, description="Timeout in seconds")


class AgentRunResponse(BaseModel):
    """Response from running an agent."""

    execution_id: str
    state: str
    response: str | None
    iterations: int
    tool_calls: int
    messages: list[dict[str, Any]]


class ExecutionStatusResponse(BaseModel):
    """Status of an execution."""

    execution_id: str
    state: str
    iteration: int
    message_count: int
    tool_result_count: int
    created_at: str
    updated_at: str


@router.post("/agents/run")
async def run_agent(request: AgentRunRequest) -> AgentRunResponse:
    """
    Run an agent on a query.

    Creates an agent with the specified configuration and executes
    it on the provided query. Returns the final response and execution
    metadata.
    """
    logger.info(
        "agent_run_request",
        name=request.name,
        model=request.model,
        tools=request.tools,
    )

    # Create agent config
    config = AgentConfig(
        name=request.name,
        model=request.model,
        provider=request.provider,
        system_prompt=request.system_prompt,
        tools=request.tools,
        max_iterations=request.max_iterations,
        temperature=request.temperature,
        timeout_seconds=request.timeout,
    )

    # Create and run agent
    agent = Agent(config)

    try:
        ctx = await agent.run(request.query, timeout=request.timeout)

        # Store execution
        _executions[ctx.execution_id] = ctx

        # Get final response
        response = agent.get_final_response(ctx)

        # Count tool calls
        tool_call_count = sum(
            len(m.tool_calls) if m.tool_calls else 0 for m in ctx.messages
        )

        return AgentRunResponse(
            execution_id=ctx.execution_id,
            state=ctx.state.value,
            response=response,
            iterations=ctx.iteration,
            tool_calls=tool_call_count,
            messages=[
                {
                    "role": m.role.value,
                    "content": m.content,
                    "tool_calls": (
                        [{"name": tc.function.name} for tc in m.tool_calls]
                        if m.tool_calls
                        else None
                    ),
                }
                for m in ctx.messages
            ],
        )

    except Exception as e:
        logger.error("agent_run_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/executions/{execution_id}")
async def get_execution(execution_id: str) -> ExecutionStatusResponse:
    """
    Get the status of an execution.

    Returns execution metadata and current state.
    """
    if execution_id not in _executions:
        raise HTTPException(status_code=404, detail="Execution not found")

    ctx = _executions[execution_id]

    return ExecutionStatusResponse(
        execution_id=ctx.execution_id,
        state=ctx.state.value,
        iteration=ctx.iteration,
        message_count=len(ctx.messages),
        tool_result_count=len(ctx.tool_results),
        created_at=ctx.created_at.isoformat(),
        updated_at=ctx.updated_at.isoformat(),
    )


@router.get("/agents/executions/{execution_id}/messages")
async def get_execution_messages(execution_id: str) -> list[dict[str, Any]]:
    """
    Get all messages from an execution.

    Returns the full conversation history.
    """
    if execution_id not in _executions:
        raise HTTPException(status_code=404, detail="Execution not found")

    ctx = _executions[execution_id]

    return [
        {
            "id": m.id,
            "role": m.role.value,
            "content": m.content,
            "timestamp": m.timestamp.isoformat(),
            "tool_call_id": m.tool_call_id,
            "tool_calls": (
                [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in m.tool_calls
                ]
                if m.tool_calls
                else None
            ),
        }
        for m in ctx.messages
    ]


@router.delete("/agents/executions/{execution_id}")
async def delete_execution(execution_id: str) -> dict[str, bool]:
    """
    Delete an execution from memory.
    """
    if execution_id not in _executions:
        raise HTTPException(status_code=404, detail="Execution not found")

    del _executions[execution_id]
    return {"deleted": True}


@router.get("/agents/executions")
async def list_executions(
    limit: int = 100,
    state: str | None = None,
) -> list[ExecutionStatusResponse]:
    """
    List recent executions.

    Optionally filter by state.
    """
    executions = list(_executions.values())

    if state:
        try:
            state_enum = AgentState(state)
            executions = [e for e in executions if e.state == state_enum]
        except ValueError:
            pass

    # Sort by updated_at descending
    executions.sort(key=lambda e: e.updated_at, reverse=True)
    executions = executions[:limit]

    return [
        ExecutionStatusResponse(
            execution_id=ctx.execution_id,
            state=ctx.state.value,
            iteration=ctx.iteration,
            message_count=len(ctx.messages),
            tool_result_count=len(ctx.tool_results),
            created_at=ctx.created_at.isoformat(),
            updated_at=ctx.updated_at.isoformat(),
        )
        for ctx in executions
    ]
