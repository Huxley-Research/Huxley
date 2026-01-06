"""
Agent orchestrator for managing multi-agent workflows.

The orchestrator handles:
- Agent lifecycle management
- Parallel and sequential execution
- Inter-agent communication
- Workflow coordination
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from huxley.agents.base import Agent
from huxley.core.logging import get_logger
from huxley.core.types import AgentConfig, AgentState, ExecutionContext, generate_id

logger = get_logger(__name__)


@dataclass
class WorkflowStep:
    """A single step in an agent workflow."""

    agent_config: AgentConfig
    query_template: str  # Can include {variables} from previous steps
    name: str | None = None
    depends_on: list[str] = field(default_factory=list)
    condition: Callable[[dict[str, Any]], bool] | None = None


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    workflow_id: str
    success: bool
    step_results: dict[str, ExecutionContext]
    final_output: str | None
    errors: list[str] = field(default_factory=list)


class AgentOrchestrator:
    """
    Orchestrates multi-agent workflows.

    Supports:
    - Sequential execution (pipeline)
    - Parallel execution (fan-out)
    - Conditional branching
    - Result aggregation
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._running_workflows: dict[str, asyncio.Task[Any]] = {}

    def register_agent(self, name: str, agent: Agent) -> None:
        """
        Register an agent with the orchestrator.

        Args:
            name: Unique identifier for the agent
            agent: Agent instance
        """
        self._agents[name] = agent
        logger.debug("agent_registered", name=name)

    def create_agent(self, name: str, config: AgentConfig) -> Agent:
        """
        Create and register an agent.

        Args:
            name: Unique identifier
            config: Agent configuration

        Returns:
            Created agent instance
        """
        agent = Agent(config)
        self.register_agent(name, agent)
        return agent

    async def run_single(
        self,
        agent_name: str,
        query: str,
        *,
        context: ExecutionContext | None = None,
        timeout: float | None = None,
    ) -> ExecutionContext:
        """
        Run a single registered agent.

        Args:
            agent_name: Name of registered agent
            query: Query to execute
            context: Existing context
            timeout: Execution timeout

        Returns:
            ExecutionContext with results

        Raises:
            KeyError: If agent not found
        """
        if agent_name not in self._agents:
            raise KeyError(f"Agent not found: {agent_name}")

        agent = self._agents[agent_name]
        return await agent.run(query, context=context, timeout=timeout)

    async def run_pipeline(
        self,
        steps: list[tuple[str, str]],
        *,
        initial_context: dict[str, Any] | None = None,
    ) -> list[ExecutionContext]:
        """
        Run agents in sequence, passing results forward.

        Args:
            steps: List of (agent_name, query_template) tuples
            initial_context: Variables available to first step

        Returns:
            List of ExecutionContexts from each step
        """
        results: list[ExecutionContext] = []
        context_vars = initial_context or {}

        for agent_name, query_template in steps:
            # Format query with available context
            query = query_template.format(**context_vars)

            # Run agent
            result = await self.run_single(agent_name, query)
            results.append(result)

            # Extract result for next step
            agent = self._agents[agent_name]
            final_response = agent.get_final_response(result)
            context_vars[f"{agent_name}_result"] = final_response
            context_vars["last_result"] = final_response

        return results

    async def run_parallel(
        self,
        tasks: list[tuple[str, str]],
        *,
        timeout: float | None = None,
    ) -> dict[str, ExecutionContext]:
        """
        Run multiple agents in parallel.

        Args:
            tasks: List of (agent_name, query) tuples
            timeout: Overall timeout for all tasks

        Returns:
            Dict mapping agent names to their results
        """
        async def run_task(name: str, query: str) -> tuple[str, ExecutionContext]:
            result = await self.run_single(name, query)
            return name, result

        coroutines = [run_task(name, query) for name, query in tasks]

        if timeout:
            results = await asyncio.wait_for(
                asyncio.gather(*coroutines, return_exceptions=True),
                timeout=timeout,
            )
        else:
            results = await asyncio.gather(*coroutines, return_exceptions=True)

        output = {}
        for item in results:
            if isinstance(item, Exception):
                logger.error("parallel_task_failed", error=str(item))
            else:
                name, ctx = item
                output[name] = ctx

        return output

    async def run_workflow(
        self,
        steps: list[WorkflowStep],
        *,
        initial_vars: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """
        Execute a complex workflow with dependencies and conditions.

        Args:
            steps: Workflow step definitions
            initial_vars: Initial variables

        Returns:
            WorkflowResult with all step results
        """
        workflow_id = generate_id()
        logger.info("workflow_started", workflow_id=workflow_id, steps=len(steps))

        variables = initial_vars or {}
        step_results: dict[str, ExecutionContext] = {}
        errors: list[str] = []

        # Build dependency graph
        step_map = {(s.name or str(i)): s for i, s in enumerate(steps)}
        completed: set[str] = set()

        # Simple topological execution
        remaining = list(step_map.keys())

        while remaining:
            # Find steps whose dependencies are satisfied
            ready = [
                name
                for name in remaining
                if all(dep in completed for dep in step_map[name].depends_on)
            ]

            if not ready:
                errors.append("Circular dependency or unsatisfied dependencies")
                break

            # Execute ready steps (could be parallelized)
            for step_name in ready:
                step = step_map[step_name]

                # Check condition
                if step.condition and not step.condition(variables):
                    logger.debug("step_skipped", step=step_name, reason="condition")
                    completed.add(step_name)
                    remaining.remove(step_name)
                    continue

                try:
                    # Create agent for this step
                    agent = Agent(step.agent_config)
                    query = step.query_template.format(**variables)

                    result = await agent.run(query)
                    step_results[step_name] = result

                    # Update variables
                    final_response = agent.get_final_response(result)
                    variables[f"{step_name}_result"] = final_response

                    if result.state == AgentState.COMPLETED:
                        completed.add(step_name)
                    else:
                        errors.append(f"Step {step_name} failed: {result.state}")

                except Exception as e:
                    errors.append(f"Step {step_name} error: {e}")
                    logger.error("step_failed", step=step_name, error=str(e))

                remaining.remove(step_name)

        # Determine final output
        final_output = None
        if step_results:
            last_step = list(step_results.keys())[-1]
            last_ctx = step_results[last_step]
            for msg in reversed(last_ctx.messages):
                if msg.role.value == "assistant" and msg.content:
                    final_output = msg.content
                    break

        success = len(errors) == 0 and len(completed) == len(step_map)

        logger.info(
            "workflow_completed",
            workflow_id=workflow_id,
            success=success,
            completed_steps=len(completed),
            errors=len(errors),
        )

        return WorkflowResult(
            workflow_id=workflow_id,
            success=success,
            step_results=step_results,
            final_output=final_output,
            errors=errors,
        )

    def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a running workflow.

        Args:
            workflow_id: ID of workflow to cancel

        Returns:
            True if cancelled, False if not found
        """
        if workflow_id in self._running_workflows:
            self._running_workflows[workflow_id].cancel()
            return True
        return False

    def get_registered_agents(self) -> list[str]:
        """Get names of all registered agents."""
        return list(self._agents.keys())
