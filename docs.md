# Huxley Documentation

**Biological Computational Engine** — v0.6.0

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Commands](#cli-commands)
  - [huxley setup](#huxley-setup)
  - [huxley generate](#huxley-generate)
  - [huxley chat](#huxley-chat)
  - [huxley research](#huxley-research)
  - [huxley automate](#huxley-automate)
  - [huxley binder](#huxley-binder)
  - [huxley check](#huxley-check)
  - [huxley config](#huxley-config)
- [Chat Slash Commands](#chat-slash-commands)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [Python API](#python-api)
- [Architecture](#architecture)

---

## Overview

Huxley is foundational infrastructure for long-horizon biological AI systems. It provides:

- **OpenAI-compatible LLM abstraction** — Swap providers without code changes
- **Typed tool registry** — Schema-validated tool definitions
- **Multi-step agent orchestration** — Planning, execution, memory, verification
- **Distributed compute** — Worker pools for heavy biological computation
- **SE(3) diffusion** — Protein structure generation via FrameDiff
- **RCSB PDB integration** — Direct access to the Protein Data Bank
- **Chemistry tools** — RDKit-powered molecular analysis

---

## Installation

```bash
# From source
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# With GPU support
pip install -e ".[gpu]"
```

---

## Quick Start

```bash
# First-time setup
huxley setup

# Generate a protein structure
huxley generate -l 100

# Interactive chat
huxley chat

# Autonomous research
huxley research "Find proteins that bind to insulin"

# Extended autonomous exploration
huxley automate -t 2 -d drug_discovery
```

---

## CLI Commands

### huxley setup

Configure API keys and download model weights.

```bash
huxley setup
huxley setup --skip-weights    # Skip model weight downloads
huxley setup --skip-keys       # Skip API key configuration
```

| Option | Description |
|--------|-------------|
| `--skip-weights` | Skip downloading model weights |
| `--skip-keys` | Skip API key configuration |

---

### huxley generate

Generate a protein structure using SE(3) diffusion.

```bash
huxley generate
huxley generate -l 80 -d "alpha helical bundle"
huxley generate -l 150 -o myprotein.pdb
huxley generate -l 100 -n 5 -s 42
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--length` | `-l` | `100` | Protein length in residues |
| `--description` | `-d` | `None` | Natural language description |
| `--output` | `-o` | `None` | Output PDB file path |
| `--samples` | `-n` | `1` | Number of structures to generate |
| `--seed` | `-s` | `None` | Random seed for reproducibility |

---

### huxley chat

Interactive AI assistant for biology with integrated tools.

```bash
huxley chat
huxley chat -m claude-4.5-sonnet
huxley chat -m auto                      # Automatic model selection
huxley chat -m auto -c performance       # Auto with performance tier
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | `None` | LLM model to use (or "auto" for automatic selection) |
| `--cost-tier` | `-c` | `balanced` | Cost tier for auto selection: economy, balanced, performance |

**Automatic Model Selection:**

When using `-m auto`, Huxley analyses each prompt and selects the optimal model based on:
- **Task type** — Reasoning, coding, creative, research, tool use
- **Complexity** — Simple queries use faster models, complex ones use stronger models
- **Cost tier** — Economy (cheapest), Balanced (default), Performance (best)

**In-chat commands:**
- `quit` / `exit` / `q` — End session
- `clear` — Reset chat history
- `save` — Save conversation to database
- `help` — Show help
- `/help` — Show slash commands

---

### huxley research

Autonomous research mode for biological questions.

```bash
huxley research "Find proteins that bind to insulin"
huxley research "Analyze EGFR inhibitor mechanisms" -i 20
huxley research "Design a binder for spike protein" -v
huxley research "Map kinase pathways" -o ./results
huxley research "Explore kinase mechanisms" -m auto -c performance
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--iterations` | `-i` | `10` | Maximum iterations |
| `--model` | `-m` | `None` | LLM model (or "auto" for automatic selection) |
| `--output` | `-o` | `None` | Output directory for results |
| `--verbose` | `-v` | `False` | Show detailed output |
| `--cost-tier` | `-c` | `balanced` | Cost tier for auto selection |

---

### huxley automate

Autonomous knowledge acquisition under epistemic constraints.

Huxley explores biological knowledge, identifies gaps, generates speculative hypotheses, and tracks uncertainty — all with strict safety boundaries and full provenance.

```bash
huxley automate -t 0.5
huxley automate -t 1.5 -d drug_discovery
huxley automate -t 2 --objective "map kinase inhibitor failure modes"
huxley automate -t 4 --curiosity-policy contradictions
huxley automate -t 2 -m auto -c performance
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--time` | `-t` | `1.0` | Duration in hours |
| `--model` | `-m` | `None` | LLM model (or "auto" for automatic selection) |
| `--domain` | `-d` | `None` | Research domain to explore |
| `--objective` | — | `None` | Specific research objective |
| `--curiosity-policy` | — | `uncertainty` | Direction selection strategy |
| `--cost-tier` | `-c` | `balanced` | Cost tier for auto selection |

**Curiosity Policies:**

| Policy | Description |
|--------|-------------|
| `uncertainty` | Prioritize areas with high epistemic uncertainty |
| `gaps` | Focus on unexplored knowledge gaps |
| `contradictions` | Investigate conflicting findings in literature |

**Safety Boundaries:**

✅ **Allowed:**
- Literature review & synthesis
- Structure database queries
- Hypothesis generation (tagged speculative)
- Knowledge gap identification
- Contradiction detection
- Memory & reasoning pattern improvement

❌ **Forbidden:**
- Wet-lab protocols
- Real-world experiments
- Tool permission modification
- Claiming discoveries as facts

---

### huxley binder

Design a protein binder for a target structure.

```bash
huxley binder 1ABC
huxley binder target.pdb -l 100 -n 5
huxley binder 6VXX -o spike_binder.pdb
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--length` | `-l` | `80` | Binder length in residues |
| `--output` | `-o` | `None` | Output PDB file path |
| `--designs` | `-n` | `3` | Number of designs to generate |

**TARGET** can be:
- A PDB ID (e.g., `1ABC`, `6VXX`)
- A local PDB file path (e.g., `target.pdb`)

---

### huxley check

Display system status and configuration.

```bash
huxley check
```

Shows:
- Configured API keys
- Downloaded model weights
- Available dependencies
- Database connection status

---

### huxley config

Manage Huxley configuration.

```bash
# Show current config
huxley config
huxley config show

# Set values
huxley config set default_provider openai
huxley config set api_keys.anthropic sk-ant-xxx
huxley config set database.url postgresql://user:pass@host/db
huxley config set redis.url redis://localhost:6379

# Get values
huxley config get default_provider
huxley config get database.type

# Delete values
huxley config delete api_keys.openai

# Show config file path
huxley config path

# Database commands
huxley config init-db     # Initialize database tables
huxley config test-db     # Test database connection
huxley config check-db    # Check schema and auto-update
```

**Config Subcommands:**

| Command | Description |
|---------|-------------|
| `show` | Show current configuration |
| `set <key> <value>` | Set a configuration value |
| `get <key>` | Get a configuration value |
| `delete <key>` | Delete a configuration value |
| `path` | Show config file path |
| `init-db` | Initialize database tables |
| `test-db` | Test database connection |
| `check-db` | Check and auto-update schema |

---

## Chat Slash Commands

Available in `huxley chat` mode:

| Command | Description | Example |
|---------|-------------|---------|
| `/search-pdb <query>` | Search the Protein Data Bank | `/search-pdb insulin receptor` |
| `/pdb <id>` | Get details about a PDB structure | `/pdb 1ABC` |
| `/construct <description>` | Design a molecule from description | `/construct aspirin-like analgesic` |
| `/properties <smiles>` | Calculate molecular properties | `/properties CC(=O)OC1=CC=CC=C1C(=O)O` |
| `/validate <smiles>` | Validate a SMILES string | `/validate CCO` |
| `/druglike <smiles>` | Check drug-likeness (Lipinski) | `/druglike CC(=O)OC1=CC=CC=C1C(=O)O` |
| `/generate` | Generate a protein structure | `/generate` |
| `/literature <query>` | Search scientific literature | `/literature CRISPR delivery` |
| `/help` | Show available commands | `/help` |

---

## Configuration

Configuration is stored in `~/.huxley/config.yaml`.

### Key Configuration Options

```yaml
# LLM Provider
default_provider: anthropic  # anthropic, openai, google, xai, cohere

# API Keys
api_keys:
  anthropic: sk-ant-xxx
  openai: sk-xxx
  google: xxx
  xai: xxx
  cohere: xxx

# Database (PostgreSQL/Supabase/Neon)
database:
  url: postgresql://user:pass@host:5432/dbname
  type: supabase  # supabase, neon, postgres, sqlite

# Redis (optional, for caching)
redis:
  url: redis://localhost:6379

# Vector Store (for embeddings)
vector:
  url: postgresql://user:pass@host:5432/dbname
  enabled: true
```

### Supported Providers

| Provider | Models |
|----------|--------|
| Anthropic | claude-4.5-opus, claude-4.5-sonnet, claude-4.5-haiku |
| OpenAI | gpt-5.2-pro, gpt-5.2, gpt-4 |
| Google | gemini-3-pro, gemini-3-flash |
| xAI | grok-4 |
| Cohere | command-a-03-2025, command-a-reasoning, command-a-vision |

---

## Database Setup

Huxley supports multiple database backends for conversation persistence, memory, and exploration sessions.

### Supported Databases

| Database | Vector Support | Best For |
|----------|----------------|----------|
| Supabase | ✅ pgvector | Production, hosted |
| Neon | ✅ pgvector | Serverless PostgreSQL |
| PostgreSQL | ✅ pgvector* | Self-hosted |
| SQLite | ❌ | Local development |

*Requires pgvector extension

### Setup Steps

```bash
# 1. Configure database URL
huxley config set database.url postgresql://user:pass@host:5432/db

# 2. Initialize tables
huxley config init-db

# 3. Verify connection
huxley config test-db

# 4. Check schema (auto-updates if needed)
huxley config check-db
```

### Tables Created

- `huxley_conversations` — Chat history
- `huxley_messages` — Individual messages
- `huxley_exploration_sessions` — Automate sessions
- `huxley_hypothesis_ledger` — Speculative hypotheses
- `huxley_skill_registry` — Learned reasoning patterns
- `huxley_embeddings` — Vector embeddings (if supported)

---

## Python API

### Basic Usage

```python
import asyncio
from huxley.agents import Agent
from huxley.core.types import AgentConfig

async def main():
    config = AgentConfig(
        name="bio-agent",
        model="claude-4.5-sonnet",
        provider="anthropic",
        system_prompt="You are a computational biology assistant.",
        tools=["pdb_search", "pdb_get_entry"],
        max_iterations=10,
    )
    
    agent = Agent(config)
    ctx = await agent.run("Find proteins similar to insulin")
    print(agent.get_final_response(ctx))

asyncio.run(main())
```

### Using Tools Directly

```python
from huxley.tools.biology.rcsb import pdb_search, pdb_get_entry

# Search PDB
results = await pdb_search("kinase inhibitor", max_results=5)

# Get structure details
entry = await pdb_get_entry("1ABC")
```

### Chemistry Tools

```python
from huxley.tools.chemistry.molecules import (
    calculate_properties,
    validate_smiles,
    check_drug_likeness,
)

# Calculate molecular properties
props = calculate_properties("CC(=O)OC1=CC=CC=C1C(=O)O")

# Validate SMILES
is_valid = validate_smiles("CCO")

# Check Lipinski's Rule of Five
druglike = check_drug_likeness("CC(=O)OC1=CC=CC=C1C(=O)O")
```

### Memory Store

```python
from huxley.memory.factory import create_memory_store
from huxley.cli.config import ConfigManager

manager = ConfigManager()
store = create_memory_store(manager)

# Save conversation
await store.save_conversation(session_id, messages)

# Load conversation
messages = await store.load_conversation(session_id)
```

---

## Automatic Model Selection

Huxley includes an intelligent model selection system that automatically chooses the optimal model for each task based on complexity, type, and cost constraints.

### Usage

```bash
# Enable auto mode in chat
huxley chat -m auto

# With cost tier selection
huxley chat -m auto -c performance     # Best models, higher cost
huxley chat -m auto -c balanced        # Default, good balance
huxley chat -m auto -c economy         # Cheapest options
```

### How It Works

The auto selector analyses each prompt to determine:

1. **Task Type** — Simple, reasoning, coding, creative, research, vision, tool_use, fast
2. **Complexity** — Based on length, technical terms, structural indicators
3. **Cost Tier** — User preference for cost vs capability tradeoff

### Model Capabilities

Each model is scored on multiple dimensions:

| Model | Reasoning | Coding | Creative | Tool Use | Speed | Cost |
|-------|-----------|--------|----------|----------|-------|------|
| claude-4.5-opus | 0.98 | 0.96 | 0.95 | 0.94 | 0.3 | 1.0 |
| claude-4.5-sonnet | 0.93 | 0.92 | 0.88 | 0.92 | 0.7 | 0.5 |
| gpt-5.2-pro | 0.95 | 0.94 | 0.90 | 0.92 | 0.5 | 0.8 |
| command-a-reasoning | 0.92 | 0.85 | 0.82 | 0.88 | 0.4 | 0.45 |
| gemini-3-flash | 0.80 | 0.78 | 0.75 | 0.82 | 0.95 | 0.15 |

### Cost Tiers

| Tier | Description | Use Case |
|------|-------------|----------|
| `economy` | Prioritises cheapest models | High-volume, simple tasks |
| `balanced` | Balance of capability and cost | General use (default) |
| `performance` | Best models regardless of cost | Complex research, critical tasks |

### Python API

```python
from huxley.llm.auto_selector import AutoModelSelector, CostTier

# Initialise with available models
selector = AutoModelSelector([
    "claude-4.5-sonnet",
    "gpt-5.2",
    "gemini-3-flash"
])

# Select optimal model
provider, model = selector.select_model(
    prompt="Analyse the binding affinity of EGFR inhibitors",
    cost_tier=CostTier.BALANCED,
    has_tools=True
)
```

---

## Architecture

```
huxley/
├── agents/         # Agent implementation and orchestration
│   ├── base.py           # Base agent class
│   └── orchestrator.py   # Multi-agent workflows
├── api/            # FastAPI server
│   ├── app.py            # Application setup
│   └── routes/           # API endpoints
├── cli/            # Command-line interface
│   ├── main.py           # CLI entry point
│   ├── ui.py             # Terminal UI components
│   ├── config.py         # Configuration manager
│   └── commands/         # Individual commands
├── compute/        # Distributed worker infrastructure
│   ├── tasks.py          # Task definitions
│   └── worker.py         # Worker pool
├── core/           # Core types and utilities
│   ├── config.py         # Configuration
│   ├── types.py          # Type definitions
│   └── exceptions.py     # Custom exceptions
├── llm/            # LLM abstraction layer
│   ├── client.py         # Unified client
│   ├── auto_selector.py  # Automatic model selection
│   └── providers/        # Provider implementations
├── memory/         # Persistence layer
│   ├── factory.py        # Store factory
│   ├── postgres_store.py # PostgreSQL backend
│   ├── sqlite_store.py   # SQLite backend
│   └── migrations/       # Schema definitions
├── prompts/        # System prompt templates
│   ├── registry.py       # Prompt registry
│   └── templates.py      # Model-specific prompts
├── tools/          # Tool implementations
│   ├── registry.py       # Tool registry
│   ├── executor.py       # Tool execution
│   ├── biology/          # Biology tools (PDB, diffusion)
│   └── chemistry/        # Chemistry tools (RDKit)
├── verification/   # Output validation
│   └── validator.py      # Response verification
└── visualization/  # Molecular visualization
    └── molecule_viewer.py # 3Dmol.js viewer
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `LLMClient` | Provider-agnostic LLM access |
| `ToolRegistry` | Typed tool management |
| `Agent` | Multi-step reasoning loop |
| `AgentOrchestrator` | Multi-agent workflows |
| `PromptRegistry` | Model-specific system prompts |
| `MemoryStore` | Persistent conversation/session storage |
| `OutputValidator` | Configurable output verification |
| `WorkerPool` | Distributed task execution |

---

## Recommended Models

| Model | Provider | Best For |
|-------|----------|----------|
| **Claude 4.5 Opus** | Anthropic | Complex reasoning, analysis |
| **Claude 4.5 Sonnet** | Anthropic | Balanced capability/cost |
| **Claude 4.5 Haiku** | Anthropic | Fast, high-volume tasks |
| **Gemini 3 Pro** | Google | Advanced planning, multimodal |
| **GPT-5.2 Pro** | OpenAI | Complex multi-step tasks |
| **Grok 4** | xAI | Agentic research, real-time data |
| **Command A** | Cohere | General purpose, RAG, structured outputs |
| **Command A Reasoning** | Cohere | Chain-of-thought, complex reasoning |
| **Command A Vision** | Cohere | Image analysis, multimodal tasks |

---

## License

MIT License

---

*Huxley — Biological Computational Engine*
