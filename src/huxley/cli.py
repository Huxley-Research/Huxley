"""
Huxley CLI.

Provides command-line interface for:
- Starting the API server
- Running agents
- Managing tools
- Configuration
"""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="huxley",
    help="Agentic, model-agnostic biological intelligence framework",
    no_args_is_help=True,
)

console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
) -> None:
    """Start the Huxley API server."""
    import uvicorn

    console.print(f"[bold green]Starting Huxley server on {host}:{port}[/bold green]")

    uvicorn.run(
        "huxley.api.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


@app.command()
def run(
    query: str = typer.Argument(..., help="Query or task for the agent"),
    model: str = typer.Option("gpt-4", "--model", "-m", help="Model to use"),
    provider: str = typer.Option("openai", "--provider", help="LLM provider"),
    system_prompt: Optional[str] = typer.Option(
        None, "--system", "-s", help="System prompt"
    ),
    tools: Optional[str] = typer.Option(
        None, "--tools", "-t", help="Comma-separated tool names"
    ),
    max_iterations: int = typer.Option(10, "--max-iter", help="Max iterations"),
    temperature: float = typer.Option(0.0, "--temp", help="Temperature"),
    stream: bool = typer.Option(False, "--stream", help="Stream output"),
) -> None:
    """Run an agent on a query."""
    from huxley.agents.base import Agent
    from huxley.core.types import AgentConfig

    # Parse tools
    tool_list = []
    if tools:
        tool_list = [t.strip() for t in tools.split(",")]

    config = AgentConfig(
        name="cli-agent",
        model=model,
        provider=provider,
        system_prompt=system_prompt,
        tools=tool_list,
        max_iterations=max_iterations,
        temperature=temperature,
    )

    agent = Agent(config)

    async def execute() -> None:
        if stream:
            async for chunk in agent.stream(query):
                console.print(chunk, end="")
            console.print()
        else:
            ctx = await agent.run(query)
            response = agent.get_final_response(ctx)
            if response:
                console.print(response)
            console.print(
                f"\n[dim]Completed in {ctx.iteration} iteration(s)[/dim]"
            )

    asyncio.run(execute())


@app.command()
def tools(
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by tag"),
) -> None:
    """List available tools."""
    from huxley.tools.registry import get_registry

    # Import builtin tools to register them
    import huxley.tools.builtin  # noqa: F401

    registry = get_registry()
    tags_filter = {tag} if tag else None
    tool_list = registry.list(tags=tags_filter)

    if not tool_list:
        console.print("[yellow]No tools found[/yellow]")
        return

    table = Table(title="Available Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Tags", style="green")
    table.add_column("Params")

    for tool in tool_list:
        param_names = [p.name for p in tool.parameters]
        table.add_row(
            tool.name,
            tool.description[:50] + "..." if len(tool.description) > 50 else tool.description,
            ", ".join(tool.tags),
            ", ".join(param_names),
        )

    console.print(table)


@app.command()
def config() -> None:
    """Show current configuration."""
    from huxley.core.config import get_config

    cfg = get_config()

    table = Table(title="Huxley Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Environment", cfg.env)
    table.add_row("Debug", str(cfg.debug))
    table.add_row("Data Directory", str(cfg.data_dir))
    table.add_row("Default Provider", cfg.default_provider)
    table.add_row("Default Model", cfg.default_model)
    table.add_row("Server Host", cfg.server.host)
    table.add_row("Server Port", str(cfg.server.port))
    table.add_row("Log Level", cfg.logging.level)
    table.add_row("Log Format", cfg.logging.format)

    console.print(table)


@app.command()
def version() -> None:
    """Show version information."""
    from huxley import __version__

    console.print(f"Huxley v{__version__}")


@app.command()
def invoke(
    tool_name: str = typer.Argument(..., help="Name of the tool to invoke"),
    args: Optional[str] = typer.Option(
        None, "--args", "-a", help="JSON arguments"
    ),
) -> None:
    """Invoke a tool directly."""
    import json as json_lib

    from huxley.core.types import ToolCall, ToolCallFunction
    from huxley.tools.executor import get_executor

    # Import builtin tools
    import huxley.tools.builtin  # noqa: F401

    arguments = {}
    if args:
        arguments = json_lib.loads(args)

    tool_call = ToolCall(
        function=ToolCallFunction(
            name=tool_name,
            arguments=json_lib.dumps(arguments),
        )
    )

    async def execute() -> None:
        executor = get_executor()
        result = await executor.execute(tool_call)

        if result.error:
            console.print(f"[red]Error: {result.error}[/red]")
        else:
            if isinstance(result.output, (dict, list)):
                console.print_json(data=result.output)
            else:
                console.print(result.output)

        console.print(
            f"\n[dim]Execution time: {result.execution_time_ms:.2f}ms[/dim]"
        )

    asyncio.run(execute())


if __name__ == "__main__":
    app()
