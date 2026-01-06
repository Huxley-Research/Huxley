"""
Prompt registry for model-specific system prompts.

The registry manages prompt templates and provides a simple API
for generating optimized prompts for any supported model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from huxley.core.logging import get_logger
from huxley.prompts.templates import (
    BaseSystemPrompt,
    PromptContext,
    get_prompt_class,
    MODEL_PROMPT_MAP,
)

logger = get_logger(__name__)


class PromptRegistry:
    """
    Registry for model-specific prompt templates.
    
    Provides:
    - Access to all registered prompt templates
    - Custom prompt registration
    - System prompt generation for any model
    """
    
    def __init__(self) -> None:
        """Initialize the prompt registry."""
        self._custom_prompts: dict[str, type[BaseSystemPrompt]] = {}
        self._default_context = PromptContext()
    
    def register(
        self,
        model_name: str,
        prompt_class: type[BaseSystemPrompt],
    ) -> None:
        """
        Register a custom prompt template for a model.
        
        Args:
            model_name: Model identifier
            prompt_class: Prompt class implementing BaseSystemPrompt
        """
        normalized = model_name.lower().strip()
        self._custom_prompts[normalized] = prompt_class
        logger.info("prompt_registered", model=model_name)
    
    def get_prompt_class(self, model_name: str) -> type[BaseSystemPrompt]:
        """
        Get the prompt class for a model.
        
        Checks custom prompts first, then falls back to built-in prompts.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Prompt class for the model
        """
        normalized = model_name.lower().strip()
        
        if normalized in self._custom_prompts:
            return self._custom_prompts[normalized]
        
        return get_prompt_class(model_name)
    
    def generate(
        self,
        model_name: str,
        context: PromptContext | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a system prompt for a model.
        
        Args:
            model_name: Target model identifier
            context: Prompt context
            **kwargs: Context parameters if context not provided
            
        Returns:
            Formatted system prompt
        """
        prompt_class = self.get_prompt_class(model_name)
        prompt_instance = prompt_class()
        
        ctx = context or PromptContext(**kwargs)
        prompt = prompt_instance.generate(ctx)
        
        logger.debug(
            "prompt_generated",
            model=model_name,
            prompt_class=prompt_class.__name__,
            length=len(prompt),
        )
        
        return prompt
    
    def list_models(self) -> list[str]:
        """
        List all models with registered prompts.
        
        Returns:
            List of model identifiers
        """
        built_in = set(MODEL_PROMPT_MAP.keys())
        custom = set(self._custom_prompts.keys())
        return sorted(built_in | custom)
    
    def list_model_families(self) -> dict[str, list[str]]:
        """
        List models grouped by family.
        
        Returns:
            Dict mapping family name to list of models
        """
        families: dict[str, list[str]] = {
            "claude": [],
            "gemini": [],
            "openai": [],
            "xai": [],
            "cohere": [],
            "custom": [],
        }
        
        for model in self.list_models():
            if model in self._custom_prompts:
                families["custom"].append(model)
            elif "claude" in model:
                families["claude"].append(model)
            elif "gemini" in model:
                families["gemini"].append(model)
            elif "gpt" in model:
                families["openai"].append(model)
            elif "grok" in model:
                families["xai"].append(model)
            elif "command" in model:
                families["cohere"].append(model)
        
        return {k: v for k, v in families.items() if v}
    
    def set_default_context(self, context: PromptContext) -> None:
        """
        Set default context used when no context is provided.
        
        Args:
            context: Default prompt context
        """
        self._default_context = context
    
    def get_default_context(self) -> PromptContext:
        """Get the default prompt context."""
        return self._default_context


# Global registry instance
_registry: PromptRegistry | None = None


@lru_cache(maxsize=1)
def get_prompt_registry() -> PromptRegistry:
    """
    Get the global prompt registry.
    
    Returns:
        Singleton PromptRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry


def get_system_prompt(
    model_name: str,
    *,
    agent_name: str = "Huxley",
    domain: str = "general",
    available_tools: list[str] | None = None,
    constraints: list[str] | None = None,
    output_format: str | None = None,
    custom_instructions: str = "",
    current_date: str = "",
    **kwargs: Any,
) -> str:
    """
    Convenience function to generate a system prompt.
    
    Args:
        model_name: Target model identifier
        agent_name: Name for the agent
        domain: Domain/specialization description
        available_tools: List of available tool names
        constraints: List of constraints
        output_format: Expected output format
        custom_instructions: Additional custom instructions
        current_date: Current date string
        **kwargs: Additional context parameters
        
    Returns:
        Formatted system prompt for the model
        
    Example:
        >>> prompt = get_system_prompt(
        ...     "claude-4.5-sonnet",
        ...     agent_name="BioBot",
        ...     domain="molecular biology",
        ...     available_tools=["pdb_search", "pdb_get_entry"],
        ... )
    """
    registry = get_prompt_registry()
    
    context = PromptContext(
        agent_name=agent_name,
        domain=domain,
        available_tools=available_tools or [],
        constraints=constraints or [],
        output_format=output_format,
        custom_instructions=custom_instructions,
        current_date=current_date,
        **kwargs,
    )
    
    return registry.generate(model_name, context)
