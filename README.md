# Huxley

**Biological Computational Engine**

A model-agnostic framework for long-horizon biological AI systems.

---

## Overview

Huxley provides foundational infrastructure for computational biology and AI-driven research. The framework is designed to be modular, extensible, and provider-agnostic.

### Core Capabilities

- **LLM Abstraction Layer** — Seamlessly swap between providers (Anthropic, OpenAI, Google, Cohere, xAI) without code modifications
- **Typed Tool Registry** — Schema-validated tool definitions with automatic documentation
- **Agent Orchestration** — Multi-step reasoning with planning, execution, memory, and verification
- **Distributed Compute** — Worker pools for computationally intensive biological tasks
- **SE(3) Diffusion** — De novo protein structure generation via FrameDiff
- **RCSB PDB Integration** — Direct programmatic access to the Protein Data Bank
- **Chemistry Tooling** — RDKit-powered molecular analysis and property calculation

Huxley is **not** a model, benchmark, or knowledge base. It is infrastructure that connects external intelligence to biological reality through tools, data, and verification layers.

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

### Interactive Setup

```bash
huxley setup
```

The setup wizard guides you through API key configuration, model selection, and optional database setup.

### Command-Line Interface

```bash
# Generate a protein structure
huxley generate -l 100 -d "alpha helical bundle"

# Interactive chat with tools
huxley chat

# Autonomous research mode
huxley research "Identify proteins that bind to insulin"

# Extended autonomous exploration
huxley automate -t 2 -d drug_discovery
```

### Python API

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
        tools=["pdb_search", "pdb_get_entry", "calculate_properties"],
        max_iterations=10,
    )
    
    agent = Agent(config)
    ctx = await agent.run("Analyse the binding site of PDB entry 1ABC.")
    print(agent.get_final_response(ctx))

asyncio.run(main())
```

### OpenAI-Compatible API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="unused",  # Authentication optional by default
)

response = client.chat.completions.create(
    model="claude-4.5-sonnet",
    messages=[{"role": "user", "content": "What is the structure of haemoglobin?"}],
)
print(response.choices[0].message.content)
```

---

## Architecture

```
huxley/
├── agents/         # Agent implementation and orchestration
├── api/            # FastAPI server with OpenAI-compatible endpoints
├── cli/            # Command-line interface
├── compute/        # Distributed worker infrastructure
├── core/           # Types, configuration, logging, exceptions
├── llm/            # LLM abstraction layer
│   └── providers/  # Anthropic, OpenAI, Google, Cohere, xAI
├── memory/         # Conversation and state persistence
├── prompts/        # Model-specific optimised system prompts
├── tools/          # Tool registry and execution
│   ├── biology/    # PDB, diffusion, structure tools
│   └── chemistry/  # RDKit molecular tools
├── verification/   # Output validation layer
└── visualisation/  # 3Dmol.js molecular viewer
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `LLMClient` | Provider-agnostic LLM access |
| `ToolRegistry` | Typed tool management with schema validation |
| `Agent` | Multi-step reasoning loop with tool execution |
| `AgentOrchestrator` | Multi-agent workflow coordination |
| `PromptRegistry` | Model-specific system prompt templates |
| `MemoryStore` | Persistent conversation and session storage |
| `OutputValidator` | Configurable response verification |
| `WorkerPool` | Distributed task execution |

---

## Supported Providers

| Provider | Models | Specialisation |
|----------|--------|----------------|
| **Anthropic** | Claude 4.5 Opus, Sonnet, Haiku | Complex reasoning, analysis |
| **OpenAI** | GPT-5.2 Pro, GPT-5.2 | Multi-step tasks, structured outputs |
| **Google** | Gemini 3 Pro, Flash | Advanced planning, multimodal |
| **Cohere** | Command A, Reasoning, Vision | RAG, chain-of-thought, vision |
| **xAI** | Grok 4 | Agentic research, real-time data |

### Model-Specific Prompts

Huxley provides optimised system prompts tailored to each model family's characteristics:

```python
from huxley.prompts import get_system_prompt, PromptContext

context = PromptContext(
    agent_name="BioHuxley",
    domain="structural biology and protein analysis",
    available_tools=["pdb_search", "pdb_get_entry", "calculate_properties"],
    constraints=["Only provide information backed by PDB data"],
    output_format="JSON with PDB IDs and structured analysis",
)

prompt = get_system_prompt("claude-4.5-sonnet", context=context)
```

**Prompt Characteristics by Family:**

| Family | Structure | Key Features |
|--------|-----------|--------------|
| Claude | XML tags | Extended thinking, explicit persona |
| Gemini | Structured sections | Time-sensitive clauses, temperature 1.0 |
| GPT | Markdown headers | Developer/user roles, high-level goals |
| Command | Structured sections | Chain-of-thought, few-shot examples |
| Grok | Iterative loops | Autonomous workflows, parallel tools |

---

## Protein Structure Generation

Huxley includes SE(3) diffusion-based tools for de novo protein structure generation using denoising diffusion probabilistic models (DDPMs).

### Available Tools

| Tool | Purpose |
|------|---------|
| `generate_protein_structure` | Unconditional or text-guided generation |
| `scaffold_protein_motif` | Generate scaffolds around fixed motifs |
| `design_protein_binder` | Design proteins that bind to targets |
| `generate_symmetric_assembly` | Generate symmetric assemblies (cages, etc.) |
| `validate_protein_structure` | Validate physical plausibility |

### Basic Generation

```python
from huxley import generate_protein_structure

result = await generate_protein_structure(
    target_length=100,
    num_samples=3,
    conditioning_text="alpha helical bundle",
    diffusion_steps=200,
    guidance_scale=3.0,
)

for struct in result["structures"]:
    print(f"Generated: {struct['id']}")
    print(f"  Sequence: {struct['sequence'][:30]}...")
    print(f"  Confidence: {struct['confidence_score']}")
```

### Binder Design

```python
from huxley import design_protein_binder

result = await design_protein_binder(
    target_pdb=target_structure,
    target_chain="A",
    hotspot_residues=[25, 30, 35],
    binder_length=80,
    num_designs=10,
)

for design in result["binder_designs"]:
    affinity = design["binding_prediction"]["predicted_affinity_nm"]
    print(f"{design['id']}: predicted Kd = {affinity} nM")
```

---

## Chemistry Tools

RDKit-powered molecular analysis integrated into the tool registry:

```python
from huxley.tools.chemistry.molecules import (
    calculate_properties,
    validate_smiles,
    check_drug_likeness,
)

# Calculate molecular properties
props = calculate_properties("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin

# Validate SMILES notation
is_valid = validate_smiles("CCO")

# Check Lipinski's Rule of Five
druglike = check_drug_likeness("CC(=O)OC1=CC=CC=C1C(=O)O")
```

---

## Configuration

Configuration is stored in `~/.huxley/config.json` and managed via the CLI:

```bash
# View current configuration
huxley config show

# Set values
huxley config set default_provider anthropic
huxley config set api_keys.cohere your-api-key

# Database setup
huxley config set database.url postgresql://user:pass@host:5432/db
huxley config init-db
huxley config check-db
```

### Environment Variables

```bash
# LLM Providers
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export COHERE_API_KEY=...
export GOOGLE_API_KEY=...

# Server
export HUXLEY_SERVER__HOST=0.0.0.0
export HUXLEY_SERVER__PORT=8000

# Logging
export HUXLEY_LOG__LEVEL=INFO
export HUXLEY_LOG__FORMAT=json
```

---

## Registering Custom Tools

```python
from huxley.tools import tool

@tool(tags={"biology", "sequence"})
async def analyse_sequence(
    sequence: str,
    format: str = "fasta",
) -> dict:
    """
    Analyse a biological sequence.
    
    :param sequence: The sequence to analyse
    :param format: Input format (fasta, genbank)
    """
    return {"length": len(sequence), "gc_content": 0.5}
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | OpenAI-compatible completions |
| `/v1/agents/run` | POST | Execute an agent |
| `/v1/agents/executions` | GET | List agent executions |
| `/v1/tools` | GET | List available tools |
| `/v1/tools/invoke` | POST | Invoke a tool directly |

---

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/huxley

# Linting
ruff check src/huxley
```

---

## Design Principles

1. **Model Agnosticism** — No hardcoded model assumptions; swap providers freely
2. **Explicit Interfaces** — Clear contracts between components
3. **Stateless Services** — Designed for horizontal scalability
4. **Verification First** — All outputs must be validated
5. **Audit Everything** — Structured logging and tracing throughout

---

## Licence

MIT

---

*Huxley — Biological Computational Engine*
