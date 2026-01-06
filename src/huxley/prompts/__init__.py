"""
Model-specific system prompts for Huxley.

While Huxley is model-agnostic, we provide optimized system prompts
tailored to each recommended model's unique characteristics and best practices.
"""

from huxley.prompts.registry import (
    PromptRegistry,
    get_prompt_registry,
    get_system_prompt,
)
from huxley.prompts.templates import (
    BaseSystemPrompt,
    ClaudeOpusPrompt,
    ClaudeSonnetPrompt,
    ClaudeHaikuPrompt,
    GeminiProPrompt,
    GeminiFlashPrompt,
    GPT52ProPrompt,
    GPT52Prompt,
    Grok4Prompt,
)

__all__ = [
    # Registry
    "PromptRegistry",
    "get_prompt_registry",
    "get_system_prompt",
    # Templates
    "BaseSystemPrompt",
    "ClaudeOpusPrompt",
    "ClaudeSonnetPrompt", 
    "ClaudeHaikuPrompt",
    "GeminiProPrompt",
    "GeminiFlashPrompt",
    "GPT52ProPrompt",
    "GPT52Prompt",
    "Grok4Prompt",
]
