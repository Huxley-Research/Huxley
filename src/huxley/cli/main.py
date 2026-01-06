"""
Huxley CLI - Main entry point.

Academic-grade command-line interface for biological AI.
"""

import click
import asyncio
from pathlib import Path

from huxley.cli.ui import console, print_banner, rule
from huxley.cli.config import ConfigManager


# Create the main CLI group
@click.group(invoke_without_command=True)
@click.option('--version', '-v', is_flag=True, help='Show version')
@click.pass_context
def cli(ctx, version):
    """
    HUXLEY - Biological Computational Engine
    
    Generate proteins, design binders, and explore biological AI.
    
    \b
    Commands:
      huxley setup      Configure API keys and models
      huxley generate   Generate a protein structure
      huxley chat       Interactive AI assistant
      huxley research   Autonomous research mode

      huxley automate   Autonomous knowledge acquisition
      huxley check      System status
    
    \b
    For advanced usage, use the Python API:
      from huxley import generate_protein_structure
    """
    if version:
        console.print("huxley 0.6.0")
        return
    
    if ctx.invoked_subcommand is None:
        print_banner()
        console.print("Run 'huxley --help' to see all commands")
        console.print("Run 'huxley setup' to get started")
        console.print()


# =============================================================================
# SETUP COMMAND
# =============================================================================

@cli.command()
@click.option('--skip-weights', is_flag=True, help='Skip downloading model weights')
@click.option('--skip-keys', is_flag=True, help='Skip API key configuration')
def setup(skip_weights, skip_keys):
    """
    Configure API keys and download model weights.
    
    Interactive wizard for first-time setup.
    """
    from huxley.cli.commands.setup import run_setup
    asyncio.run(run_setup(skip_weights=skip_weights, skip_keys=skip_keys))


# =============================================================================
# GENERATE COMMAND
# =============================================================================

@cli.command()
@click.option('--length', '-l', default=100, help='Protein length (residues)')
@click.option('--description', '-d', default=None, help='Natural language description')
@click.option('--output', '-o', default=None, help='Output PDB file path')
@click.option('--samples', '-n', default=1, help='Number of structures to generate')
@click.option('--seed', '-s', default=None, type=int, help='Random seed for reproducibility')
def generate(length, description, output, samples, seed):
    """
    Generate a protein structure using SE(3) diffusion.
    
    \b
    Examples:
      huxley generate
      huxley generate -l 80 -d "alpha helical bundle"
      huxley generate -l 150 -o myprotein.pdb
    """
    from huxley.cli.commands.generate import run_generate
    asyncio.run(run_generate(
        length=length,
        description=description,
        output=output,
        samples=samples,
        seed=seed,
    ))


# =============================================================================
# CHECK COMMAND
# =============================================================================

@cli.command()
def check():
    """
    Display system status and configuration.
    
    Shows API keys, model weights, and dependencies.
    """
    from huxley.cli.commands.check import run_check
    asyncio.run(run_check())


# =============================================================================
# CHAT COMMAND
# =============================================================================

@cli.command()
@click.option('--model', '-m', default=None, help='LLM model to use (or "auto" for automatic selection)')
@click.option('--cost-tier', '-c', type=click.Choice(['economy', 'balanced', 'performance']), 
              default='balanced', help='Cost tier for auto model selection')
def chat(model, cost_tier):
    """
    Interactive AI assistant for biology.
    
    Chat with Huxley about proteins, structures, and biology.
    
    \b
    Examples:
      huxley chat
      huxley chat -m claude-sonnet-4-20250514
      huxley chat -m auto
      huxley chat -m auto -c performance
    """
    from huxley.cli.commands.chat import run_chat
    asyncio.run(run_chat(model=model, cost_tier=cost_tier))


# =============================================================================
# BINDER COMMAND
# =============================================================================

@cli.command()
@click.argument('target', required=True)
@click.option('--length', '-l', default=80, help='Binder length (residues)')
@click.option('--output', '-o', default=None, help='Output PDB file path')
@click.option('--designs', '-n', default=3, help='Number of designs')
def binder(target, length, output, designs):
    """
    Design a protein binder for a target structure.
    
    TARGET can be a PDB ID (e.g., 1ABC) or a PDB file path.
    
    \b
    Examples:
      huxley binder 1ABC
      huxley binder target.pdb -l 100 -n 5
    """
    from huxley.cli.commands.binder import run_binder
    asyncio.run(run_binder(
        target=target,
        length=length,
        output=output,
        designs=designs,
    ))


# =============================================================================
# RESEARCH COMMAND
# =============================================================================

@cli.command()
@click.argument('goal', required=True)
@click.option('--iterations', '-i', default=10, help='Maximum iterations')
@click.option('--model', '-m', default=None, help='LLM model (or "auto" for automatic selection)')
@click.option('--output', '-o', default=None, help='Output directory for results')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
@click.option('--cost-tier', '-c', type=click.Choice(['economy', 'balanced', 'performance']), 
              default='balanced', help='Cost tier for auto model selection')
def research(goal, iterations, model, output, verbose, cost_tier):
    """
    Autonomous research mode.
    
    The agent works independently to explore a biological question,
    run experiments, and synthesize findings.
    
    \b
    Examples:
      huxley research "Find proteins that bind to insulin"
      huxley research "Analyze EGFR inhibitor mechanisms" -i 20
      huxley research "Design a binder for spike protein" -v
      huxley research "Explore kinase mechanisms" -m auto -c performance
    """
    from huxley.cli.commands.research import run_research
    asyncio.run(run_research(
        goal=goal,
        max_iterations=iterations,
        model=model,
        output_dir=output,
        verbose=verbose,
        cost_tier=cost_tier,
    ))


# =============================================================================
# AUTOMATE COMMAND
# =============================================================================

@cli.command()
@click.option('--time', '-t', default=1.0, help='Duration in hours (e.g., 1.5)')
@click.option('--model', '-m', default=None, help='LLM model (or "auto" for automatic selection)')
@click.option('--domain', '-d', default=None, help='Research domain to explore')
@click.option('--objective', help='Specific research objective')
@click.option('--curiosity-policy', type=click.Choice(['uncertainty', 'gaps', 'contradictions']), default='uncertainty', help='How to choose research directions')
@click.option('--cost-tier', '-c', type=click.Choice(['economy', 'balanced', 'performance']), 
              default='balanced', help='Cost tier for auto model selection')
def automate(time, model, domain, objective, curiosity_policy, cost_tier):
    """
    Autonomous knowledge acquisition under epistemic constraints.
    
    Huxley explores biological knowledge, identifies gaps, generates
    speculative hypotheses, and tracks uncertainty - all with strict
    safety boundaries and full provenance.
    
    \b
    ALLOWED:
      • Literature review & synthesis
      • Structure database queries
      • Hypothesis generation (tagged speculative)
      • Knowledge gap identification
      • Contradiction detection
      • Memory & reasoning pattern improvement
    
    \b
    FORBIDDEN:
      ❌ Wet-lab protocols
      ❌ Real-world experiments
      ❌ Tool permission modification
      ❌ Claiming discoveries as facts
    
    \b
    Examples:
      huxley automate -t 0.5
      huxley automate -t 1.5 -d drug_discovery
      huxley automate -t 2 --objective "map kinase inhibitor failure modes"
      huxley automate -t 4 --curiosity-policy contradictions
      huxley automate -t 2 -m auto -c performance
    
    \b
    Output: All stored with provenance in configured database
    """
    from huxley.cli.commands.automate import run_automate
    asyncio.run(run_automate(
        hours=time,
        model=model,
        domain=domain,
        objective=objective,
        curiosity_policy=curiosity_policy,
        cost_tier=cost_tier,
    ))


# =============================================================================
# CONFIG COMMAND
# =============================================================================

@cli.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
    """
    Manage Huxley configuration.
    
    \b
    Run without subcommand to see current config:
      huxley config
    
    \b
    Or use subcommands:
      huxley config set <key> <value>
      huxley config get <key>
      huxley config delete <key>
    """
    if ctx.invoked_subcommand is None:
        # Show config when run without subcommand
        from huxley.cli.commands.config_cmd import show_config
        show_config()


@config.command('show')
def config_show():
    """Show current configuration and examples."""
    from huxley.cli.commands.config_cmd import show_config
    show_config()


@config.command('set')
@click.argument('key')
@click.argument('value')
def config_set(key, value):
    """
    Set a configuration value.
    
    \b
    Examples:
      huxley config set default_provider openai
      huxley config set api_keys.openai sk-xxx
      huxley config set database.url postgresql://...
      huxley config set redis.url redis://...
      huxley config set vector.url postgresql://...
    """
    from huxley.cli.commands.config_cmd import set_config
    set_config(key, value)


@config.command('get')
@click.argument('key')
def config_get(key):
    """
    Get a specific configuration value.
    
    \b
    Examples:
      huxley config get default_provider
      huxley config get database.type
    """
    from huxley.cli.commands.config_cmd import get_config
    get_config(key)


@config.command('delete')
@click.argument('key')
def config_delete(key):
    """Delete a configuration value."""
    from huxley.cli.commands.config_cmd import delete_config
    delete_config(key)


@config.command('path')
def config_path():
    """Show configuration file path."""
    manager = ConfigManager()
    console.print(f"Config: {manager.config_path}")


@config.command('init-db')
def config_init_db():
    """
    Initialize database tables.
    
    Creates all Huxley tables and enables vector memory
    if supported by your database provider.
    
    \b
    Supported providers with vector memory:
      - Supabase (pgvector)
      - Neon (pgvector)
      - PostgreSQL with pgvector extension
    
    \b
    Example:
      huxley config set database.url postgresql://user:pass@host/db
      huxley config init-db
    """
    from huxley.cli.commands.config_cmd import init_database
    init_database()


@config.command('test-db')
def config_test_db():
    """
    Test database connection.
    
    Verifies connectivity and shows capabilities.
    """
    from huxley.cli.commands.config_cmd import test_database
    test_database()


@config.command('check-db')
def config_check_db():
    """
    Check if database schema is up to date.
    
    Verifies all expected tables and columns exist.
    Reports missing tables or columns that need to be created.
    
    \b
    Example:
      huxley config check-db
    """
    from huxley.cli.commands.config_cmd import check_schema
    check_schema()


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
