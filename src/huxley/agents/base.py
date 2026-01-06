"""
Base agent implementation.

Agents are the core execution units that:
- Receive tasks/queries
- Plan and execute tool calls
- Iterate until completion or failure
- Produce verified outputs
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from huxley.core.exceptions import (
    AgentCancelledError,
    AgentMaxIterationsError,
    AgentTimeoutError,
)
from huxley.core.logging import get_logger
from huxley.core.types import (
    AgentConfig,
    AgentState,
    CompletionResponse,
    ExecutionContext,
    Message,
    MessageRole,
    ToolResult,
)
from huxley.llm.client import LLMClient
from huxley.tools.executor import ToolExecutor, get_executor
from huxley.tools.registry import ToolRegistry, get_registry

logger = get_logger(__name__)


class Agent:
    """
    A reasoning agent that can plan and execute multi-step tasks.

    The agent follows a loop:
    1. Send conversation to LLM
    2. If LLM requests tool calls, execute them
    3. Add results to conversation
    4. Repeat until LLM produces final response or limits reached

    Agents are stateless between runs. All state is in ExecutionContext.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        llm_client: LLMClient | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        """
        Initialize an agent.

        Args:
            config: Agent configuration
            llm_client: LLM client (created from config if not provided)
            tool_registry: Tool registry (global if not provided)
            tool_executor: Tool executor (global if not provided)
        """
        self._config = config
        self._llm = llm_client or LLMClient(
            provider=config.provider,
            model=config.model,
        )
        self._registry = tool_registry or get_registry()
        self._executor = tool_executor or get_executor()
        self._cancelled = False

    @property
    def config(self) -> AgentConfig:
        """Get the agent configuration."""
        return self._config

    def cancel(self) -> None:
        """Request cancellation of the current execution."""
        self._cancelled = True

    async def run(
        self,
        query: str,
        *,
        context: ExecutionContext | None = None,
        timeout: float | None = None,
    ) -> ExecutionContext:
        """
        Execute the agent on a query.

        Args:
            query: User query or task
            context: Existing context (new one created if not provided)
            timeout: Overall execution timeout

        Returns:
            ExecutionContext with results

        Raises:
            AgentMaxIterationsError: If max iterations exceeded
            AgentTimeoutError: If timeout exceeded
            AgentCancelledError: If cancelled
        """
        self._cancelled = False

        # Initialize context
        ctx = context or ExecutionContext(agent_config=self._config)
        ctx.state = AgentState.PLANNING

        # Add system prompt if configured
        if self._config.system_prompt and not ctx.messages:
            ctx.add_message(
                Message(role=MessageRole.SYSTEM, content=self._config.system_prompt)
            )

        # Add user query
        ctx.add_message(Message(role=MessageRole.USER, content=query))

        # Get tool schemas
        tool_schemas = self._registry.get_openai_schemas(self._config.tools or None)

        effective_timeout = timeout or self._config.timeout_seconds

        try:
            result = await asyncio.wait_for(
                self._run_loop(ctx, tool_schemas),
                timeout=effective_timeout,
            )
            return result
        except asyncio.TimeoutError:
            ctx.state = AgentState.FAILED
            raise AgentTimeoutError(effective_timeout)

    async def _run_loop(
        self,
        ctx: ExecutionContext,
        tool_schemas: list[dict[str, Any]],
    ) -> ExecutionContext:
        """Main agent execution loop."""
        while ctx.iteration < self._config.max_iterations:
            if self._cancelled:
                ctx.state = AgentState.CANCELLED
                raise AgentCancelledError("Agent execution was cancelled")

            ctx.iteration += 1
            ctx.state = AgentState.EXECUTING

            logger.debug(
                "agent_iteration",
                execution_id=ctx.execution_id,
                iteration=ctx.iteration,
            )

            # Call LLM
            response = await self._llm.complete(
                messages=ctx.messages,
                model=self._config.model,
                tools=tool_schemas if tool_schemas else None,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )

            # Get assistant message
            if not response.choices:
                ctx.state = AgentState.FAILED
                raise RuntimeError("LLM returned no choices")

            assistant_message = response.choices[0].message
            ctx.add_message(assistant_message)

            # Check if we need to execute tools
            if assistant_message.tool_calls:
                ctx.state = AgentState.WAITING_FOR_TOOL

                # Execute all tool calls
                results = await self._executor.execute_many(
                    assistant_message.tool_calls,
                    timeout=self._config.timeout_seconds / 2,  # Allow time for multiple
                )

                # Add results to context
                for result in results:
                    ctx.add_tool_result(result)
                    ctx.add_message(
                        Message(
                            role=MessageRole.TOOL,
                            content=self._format_tool_result(result),
                            tool_call_id=result.tool_call_id,
                        )
                    )

                # Continue loop to process results
                continue

            # No tool calls - check if we're done
            finish_reason = response.choices[0].finish_reason

            if finish_reason in ("stop", "end_turn", None):
                ctx.state = AgentState.COMPLETED
                logger.info(
                    "agent_completed",
                    execution_id=ctx.execution_id,
                    iterations=ctx.iteration,
                )
                return ctx

        # Max iterations reached
        ctx.state = AgentState.FAILED
        raise AgentMaxIterationsError(self._config.max_iterations)

    async def stream(
        self,
        query: str,
        *,
        context: ExecutionContext | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream agent execution, yielding content as it's generated.

        This is a simplified streaming mode that yields assistant
        content but still executes tools internally.

        Args:
            query: User query
            context: Existing context

        Yields:
            Content chunks from the assistant
        """
        self._cancelled = False
        ctx = context or ExecutionContext(agent_config=self._config)

        if self._config.system_prompt and not ctx.messages:
            ctx.add_message(
                Message(role=MessageRole.SYSTEM, content=self._config.system_prompt)
            )

        ctx.add_message(Message(role=MessageRole.USER, content=query))
        tool_schemas = self._registry.get_openai_schemas(self._config.tools or None)

        while ctx.iteration < self._config.max_iterations:
            if self._cancelled:
                break

            ctx.iteration += 1

            # Collect streamed content
            content_buffer = ""
            tool_calls = []

            async for chunk in self._llm.stream(
                messages=ctx.messages,
                model=self._config.model,
                tools=tool_schemas if tool_schemas else None,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            ):
                for choice in chunk.choices:
                    if choice.delta.content:
                        content_buffer += choice.delta.content
                        yield choice.delta.content
                    if choice.delta.tool_calls:
                        tool_calls.extend(choice.delta.tool_calls)
                    if choice.finish_reason:
                        break

            # Add message to context
            ctx.add_message(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=content_buffer if content_buffer else None,
                    tool_calls=tool_calls if tool_calls else None,
                )
            )

            # Execute tools if needed
            if tool_calls:
                results = await self._executor.execute_many(tool_calls)
                for result in results:
                    ctx.add_tool_result(result)
                    ctx.add_message(
                        Message(
                            role=MessageRole.TOOL,
                            content=self._format_tool_result(result),
                            tool_call_id=result.tool_call_id,
                        )
                    )
                continue

            # No tools - done
            break

    def _format_tool_result(self, result: ToolResult) -> str:
        """Format a tool result for the conversation."""
        if result.error:
            return f"Error: {result.error}"

        output = result.output
        if isinstance(output, (dict, list)):
            import json

            return json.dumps(output, default=str, indent=2)
        return str(output)

    def get_final_response(self, ctx: ExecutionContext) -> str | None:
        """Extract the final assistant response from context."""
        for message in reversed(ctx.messages):
            if message.role == MessageRole.ASSISTANT and message.content:
                return message.content
        return None
