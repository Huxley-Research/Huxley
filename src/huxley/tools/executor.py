"""
Tool execution engine.

Handles the invocation of tools with:
- Argument parsing and validation
- Async/sync execution
- Timeout management
- Error handling and reporting
- Execution metrics
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from huxley.core.exceptions import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolValidationError,
)
from huxley.core.logging import get_logger
from huxley.core.types import ToolCall, ToolResult
from huxley.tools.registry import Tool, ToolRegistry, get_registry

logger = get_logger(__name__)


class ToolExecutor:
    """
    Executes tools with validation, timeout, and error handling.

    Provides a consistent execution environment for all tools,
    regardless of their implementation details.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        default_timeout: float = 30.0,
    ) -> None:
        """
        Initialize the tool executor.

        Args:
            registry: Tool registry to use (defaults to global)
            default_timeout: Default timeout for tool execution
        """
        self._registry = registry or get_registry()
        self._default_timeout = default_timeout

    async def execute(
        self,
        tool_call: ToolCall,
        *,
        timeout: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: The tool call to execute
            timeout: Execution timeout (overrides default)
            context: Additional context to pass to the tool

        Returns:
            ToolResult with output or error
        """
        tool_name = tool_call.function.name
        start_time = time.perf_counter()

        try:
            # Get tool
            tool = self._registry.get(tool_name)

            # Parse arguments
            try:
                raw_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                raise ToolValidationError(
                    tool_name,
                    f"Invalid JSON arguments: {e}",
                )

            # Validate arguments
            validated_args = tool.validate_arguments(raw_args)

            # Inject context if tool accepts it
            if context and "context" in [p.name for p in tool.parameters]:
                validated_args["context"] = context

            # Execute with timeout
            effective_timeout = timeout or self._default_timeout
            output = await self._execute_with_timeout(
                tool,
                validated_args,
                effective_timeout,
            )

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            logger.debug(
                "tool_executed",
                tool=tool_name,
                execution_time_ms=execution_time_ms,
            )

            return ToolResult(
                tool_call_id=tool_call.id,
                output=output,
                execution_time_ms=execution_time_ms,
            )

        except ToolNotFoundError:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.warning("tool_not_found", tool=tool_name)
            return ToolResult(
                tool_call_id=tool_call.id,
                output=None,
                error=f"Tool not found: {tool_name}",
                execution_time_ms=execution_time_ms,
            )

        except ToolValidationError as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "tool_validation_error",
                tool=tool_name,
                errors=e.validation_errors,
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                output=None,
                error=str(e),
                execution_time_ms=execution_time_ms,
                metadata={"validation_errors": e.validation_errors},
            )

        except ToolTimeoutError as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "tool_timeout",
                tool=tool_name,
                timeout=e.timeout_seconds,
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                output=None,
                error=str(e),
                execution_time_ms=execution_time_ms,
            )

        except ToolExecutionError as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "tool_execution_error",
                tool=tool_name,
                error=str(e),
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                output=None,
                error=str(e),
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "tool_unexpected_error",
                tool=tool_name,
                error=str(e),
                exc_info=True,
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                output=None,
                error=f"Unexpected error: {e}",
                execution_time_ms=execution_time_ms,
            )

    async def execute_many(
        self,
        tool_calls: list[ToolCall],
        *,
        timeout: float | None = None,
        context: dict[str, Any] | None = None,
        parallel: bool = True,
    ) -> list[ToolResult]:
        """
        Execute multiple tool calls.

        Args:
            tool_calls: Tool calls to execute
            timeout: Per-tool timeout
            context: Shared context
            parallel: Execute in parallel (default) or sequentially

        Returns:
            List of results in same order as calls
        """
        if parallel:
            tasks = [
                self.execute(tc, timeout=timeout, context=context)
                for tc in tool_calls
            ]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for tc in tool_calls:
                result = await self.execute(tc, timeout=timeout, context=context)
                results.append(result)
            return results

    async def _execute_with_timeout(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        timeout: float,
    ) -> Any:
        """Execute a tool with timeout handling."""
        try:
            if tool.is_async:
                result = await asyncio.wait_for(
                    tool.func(**arguments),
                    timeout=timeout,
                )
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: tool.func(**arguments)),
                    timeout=timeout,
                )
            return result

        except asyncio.TimeoutError:
            raise ToolTimeoutError(tool.name, timeout)

        except Exception as e:
            raise ToolExecutionError(
                tool.name,
                str(e),
                cause=e,
            )


# Default executor instance
_executor: ToolExecutor | None = None


def get_executor() -> ToolExecutor:
    """Get the global tool executor."""
    global _executor
    if _executor is None:
        _executor = ToolExecutor()
    return _executor
