"""
Huxley: Agentic, model-agnostic biological intelligence framework.

Huxley provides foundational infrastructure for long-horizon biological
AI systems, including:

- OpenAI-compatible LLM abstraction
- Typed tool registry with schema validation
- Multi-step agent orchestration
- Memory and experiment tracking
- Distributed compute coordination
- Verification and validation layers
- Model-specific optimized system prompts
- Diffusion-based protein structure generation

Huxley is model-agnostic, vendor-neutral, and designed for
production-grade scientific computing at scale.
"""

from huxley.core.types import (
    AgentConfig,
    ExecutionContext,
    Message,
    ToolCall,
    ToolResult,
)
from huxley.prompts import get_system_prompt, PromptRegistry

# Diffusion structure generation
from huxley.tools.biology.diffusion import (
    generate_protein_structure,
    scaffold_protein_motif,
    design_protein_binder,
    generate_symmetric_assembly,
    validate_protein_structure,
    download_framediff_weights,
    check_framediff_setup,
    DiffusionConfig,
    DiffusionBackend,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "AgentConfig",
    "ExecutionContext",
    "Message",
    "ToolCall",
    "ToolResult",
    "get_system_prompt",
    "PromptRegistry",
    # Diffusion tools
    "generate_protein_structure",
    "scaffold_protein_motif",
    "design_protein_binder",
    "generate_symmetric_assembly",
    "validate_protein_structure",
    "download_framediff_weights",
    "check_framediff_setup",
    "DiffusionConfig",
    "DiffusionBackend",
]
