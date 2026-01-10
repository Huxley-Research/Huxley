"""
Automatic model selection for Huxley.

Intelligently selects the optimal model from configured providers
based on task characteristics, complexity, and cost considerations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from huxley.core.logging import get_logger

logger = get_logger(__name__)


class TaskType(Enum):
    """Categories of tasks for model selection."""
    SIMPLE = "simple"           # Quick questions, simple lookups
    REASONING = "reasoning"     # Complex analysis, multi-step reasoning
    CODING = "coding"           # Code generation, debugging
    CREATIVE = "creative"       # Creative writing, brainstorming
    RESEARCH = "research"       # Literature review, synthesis
    VISION = "vision"           # Image analysis, multimodal
    TOOL_USE = "tool_use"       # Heavy tool calling, agentic tasks
    FAST = "fast"               # Speed-critical, high-volume


class CostTier(Enum):
    """Cost preference tiers."""
    ECONOMY = "economy"         # Minimise cost
    BALANCED = "balanced"       # Balance cost and capability
    PERFORMANCE = "performance" # Maximise capability


@dataclass
class ModelCapabilities:
    """Capabilities and characteristics of a model."""
    provider: str
    model: str
    
    # Capability scores (0-100)
    reasoning: int = 50
    coding: int = 50
    creative: int = 50
    tool_use: int = 50
    speed: int = 50
    
    # Features
    supports_vision: bool = False
    supports_reasoning_mode: bool = False
    context_window: int = 128000
    
    # Cost (relative, 1-100)
    cost: int = 50
    
    # Best for
    best_for: list[TaskType] = field(default_factory=list)


# Model capability definitions
MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    # Anthropic
    "claude-4.5-opus": ModelCapabilities(
        provider="anthropic",
        model="claude-4.5-opus",
        reasoning=95,
        coding=90,
        creative=95,
        tool_use=90,
        speed=40,
        supports_vision=True,
        context_window=200000,
        cost=100,
        best_for=[TaskType.REASONING, TaskType.CREATIVE, TaskType.RESEARCH],
    ),
    "claude-4.5-sonnet": ModelCapabilities(
        provider="anthropic",
        model="claude-4.5-sonnet",
        reasoning=85,
        coding=90,
        creative=85,
        tool_use=90,
        speed=70,
        supports_vision=True,
        context_window=200000,
        cost=50,
        best_for=[TaskType.CODING, TaskType.TOOL_USE, TaskType.REASONING],
    ),
    "claude-4.5-haiku": ModelCapabilities(
        provider="anthropic",
        model="claude-4.5-haiku",
        reasoning=70,
        coding=75,
        creative=70,
        tool_use=80,
        speed=95,
        supports_vision=True,
        context_window=200000,
        cost=15,
        best_for=[TaskType.FAST, TaskType.SIMPLE, TaskType.TOOL_USE],
    ),
    
    # OpenAI
    "gpt-5.2-pro": ModelCapabilities(
        provider="openai",
        model="gpt-5.2-pro",
        reasoning=95,
        coding=95,
        creative=85,
        tool_use=90,
        speed=50,
        supports_reasoning_mode=True,
        context_window=256000,
        cost=90,
        best_for=[TaskType.REASONING, TaskType.CODING],
    ),
    "gpt-5.2": ModelCapabilities(
        provider="openai",
        model="gpt-5.2",
        reasoning=80,
        coding=85,
        creative=80,
        tool_use=85,
        speed=80,
        context_window=128000,
        cost=40,
        best_for=[TaskType.CODING, TaskType.TOOL_USE, TaskType.FAST],
    ),
    
    # Google
    "gemini-3-pro": ModelCapabilities(
        provider="google",
        model="gemini-3-pro",
        reasoning=90,
        coding=85,
        creative=85,
        tool_use=85,
        speed=60,
        supports_vision=True,
        context_window=1000000,
        cost=60,
        best_for=[TaskType.REASONING, TaskType.RESEARCH, TaskType.VISION],
    ),
    "gemini-3-flash": ModelCapabilities(
        provider="google",
        model="gemini-3-flash",
        reasoning=75,
        coding=80,
        creative=75,
        tool_use=80,
        speed=95,
        supports_vision=True,
        context_window=1000000,
        cost=20,
        best_for=[TaskType.FAST, TaskType.SIMPLE, TaskType.VISION],
    ),
    
    # Cohere
    "command-a-03-2025": ModelCapabilities(
        provider="cohere",
        model="command-a-03-2025",
        reasoning=80,
        coding=75,
        creative=80,
        tool_use=85,
        speed=70,
        context_window=256000,
        cost=35,
        best_for=[TaskType.RESEARCH, TaskType.TOOL_USE],
    ),
    "command-a-reasoning-08-2025": ModelCapabilities(
        provider="cohere",
        model="command-a-reasoning-08-2025",
        reasoning=90,
        coding=80,
        creative=75,
        tool_use=80,
        speed=50,
        supports_reasoning_mode=True,
        context_window=256000,
        cost=45,
        best_for=[TaskType.REASONING, TaskType.RESEARCH],
    ),
    "command-a-vision-07-2025": ModelCapabilities(
        provider="cohere",
        model="command-a-vision-07-2025",
        reasoning=80,
        coding=70,
        creative=80,
        tool_use=80,
        speed=65,
        supports_vision=True,
        context_window=256000,
        cost=40,
        best_for=[TaskType.VISION],
    ),
    
    # xAI
    "grok-4": ModelCapabilities(
        provider="xai",
        model="grok-4",
        reasoning=90,
        coding=85,
        creative=90,
        tool_use=95,
        speed=70,
        supports_vision=True,
        context_window=2000000,
        cost=55,
        best_for=[TaskType.RESEARCH, TaskType.TOOL_USE, TaskType.CREATIVE],
    ),
    
    # Prime Intellect (via OpenRouter)
    "intellect-3": ModelCapabilities(
        provider="openrouter",
        model="prime-intellect/intellect-3",
        reasoning=92,
        coding=88,
        creative=85,
        tool_use=90,
        speed=60,
        supports_vision=False,
        supports_reasoning_mode=True,
        context_window=256000,
        cost=42,
        best_for=[TaskType.REASONING, TaskType.CODING, TaskType.RESEARCH],
    ),
}


@dataclass
class TaskAnalysis:
    """Analysis of a task's characteristics."""
    task_type: TaskType
    complexity: int  # 1-100
    estimated_tokens: int
    requires_vision: bool = False
    requires_reasoning: bool = False
    tool_heavy: bool = False


class AutoModelSelector:
    """
    Automatically selects the optimal model based on task and available providers.
    
    Selection criteria:
    1. Filter to available (configured) providers
    2. Filter by required capabilities (vision, reasoning mode)
    3. Score remaining models by task fit
    4. Apply cost preference
    5. Return best match
    """
    
    def __init__(self, configured_providers: list[str]):
        """
        Initialise the auto selector.
        
        Args:
            configured_providers: List of provider names with API keys configured
        """
        self.configured_providers = set(configured_providers)
        self._available_models = self._filter_available_models()
        
        logger.info(
            "auto_selector_initialised",
            providers=list(self.configured_providers),
            models=list(self._available_models.keys()),
        )
    
    def _filter_available_models(self) -> dict[str, ModelCapabilities]:
        """Filter to models from configured providers."""
        return {
            name: caps
            for name, caps in MODEL_CAPABILITIES.items()
            if caps.provider in self.configured_providers
        }
    
    def analyse_task(
        self,
        prompt: str,
        *,
        has_images: bool = False,
        has_tools: bool = False,
        conversation_length: int = 0,
    ) -> TaskAnalysis:
        """
        Analyse a task to determine its characteristics.
        
        Args:
            prompt: The user prompt or task description
            has_images: Whether the task includes images
            has_tools: Whether tools are available
            conversation_length: Number of prior messages
            
        Returns:
            TaskAnalysis with inferred characteristics
        """
        prompt_lower = prompt.lower()
        
        # Detect task type from keywords
        task_type = TaskType.SIMPLE
        complexity = 30
        
        # Reasoning indicators
        reasoning_keywords = [
            "analyse", "analyze", "explain", "why", "how does",
            "compare", "evaluate", "reason", "think", "consider",
            "implications", "consequences", "derive", "prove",
        ]
        if any(kw in prompt_lower for kw in reasoning_keywords):
            task_type = TaskType.REASONING
            complexity = 70
        
        # Coding indicators
        coding_keywords = [
            "code", "function", "class", "implement", "debug",
            "python", "javascript", "typescript", "fix", "refactor",
            "algorithm", "data structure",
        ]
        if any(kw in prompt_lower for kw in coding_keywords):
            task_type = TaskType.CODING
            complexity = 60
        
        # Creative indicators
        creative_keywords = [
            "write", "create", "story", "poem", "creative",
            "brainstorm", "ideas", "imagine", "design concept",
        ]
        if any(kw in prompt_lower for kw in creative_keywords):
            task_type = TaskType.CREATIVE
            complexity = 50
        
        # Research indicators
        research_keywords = [
            "research", "literature", "papers", "studies",
            "review", "synthesise", "synthesize", "survey",
            "state of the art", "findings",
        ]
        if any(kw in prompt_lower for kw in research_keywords):
            task_type = TaskType.RESEARCH
            complexity = 75
        
        # Vision required
        if has_images:
            task_type = TaskType.VISION
            complexity = max(complexity, 50)
        
        # Tool-heavy indicators
        tool_heavy = has_tools and any(
            kw in prompt_lower for kw in [
                "search", "fetch", "get", "find", "lookup",
                "calculate", "run", "execute", "query",
            ]
        )
        if tool_heavy:
            task_type = TaskType.TOOL_USE
            complexity = max(complexity, 55)
        
        # Simple/fast indicators
        simple_keywords = ["quick", "simple", "briefly", "short", "fast"]
        if any(kw in prompt_lower for kw in simple_keywords):
            if complexity < 50:
                task_type = TaskType.FAST
                complexity = min(complexity, 25)
        
        # Estimate tokens
        estimated_tokens = len(prompt.split()) * 2 + conversation_length * 500
        
        # Adjust complexity based on length
        if len(prompt) > 2000:
            complexity = min(complexity + 20, 100)
        
        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            estimated_tokens=estimated_tokens,
            requires_vision=has_images,
            requires_reasoning=complexity > 70,
            tool_heavy=tool_heavy,
        )
    
    def select_model(
        self,
        task: TaskAnalysis | None = None,
        *,
        prompt: str | None = None,
        cost_tier: CostTier = CostTier.BALANCED,
        min_context: int = 0,
        require_vision: bool = False,
        require_reasoning_mode: bool = False,
        has_images: bool = False,
        has_tools: bool = False,
    ) -> tuple[str, str]:
        """
        Select the optimal model for a task.
        
        Args:
            task: Pre-analysed task (or will analyse from prompt)
            prompt: Task prompt (used if task not provided)
            cost_tier: Cost preference
            min_context: Minimum context window required
            require_vision: Must support vision
            require_reasoning_mode: Must support reasoning mode
            has_images: Task includes images
            has_tools: Tools are available
            
        Returns:
            Tuple of (provider, model)
            
        Raises:
            ValueError: If no suitable model is available
        """
        if not self._available_models:
            raise ValueError("No models available from configured providers")
        
        # Analyse task if not provided
        if task is None:
            if prompt:
                task = self.analyse_task(
                    prompt,
                    has_images=has_images or require_vision,
                    has_tools=has_tools,
                )
            else:
                # Default to balanced general task
                task = TaskAnalysis(
                    task_type=TaskType.SIMPLE,
                    complexity=50,
                    estimated_tokens=1000,
                )
        
        # Filter candidates
        candidates: list[tuple[str, ModelCapabilities, float]] = []
        
        for name, caps in self._available_models.items():
            # Hard requirements
            if require_vision and not caps.supports_vision:
                continue
            if task.requires_vision and not caps.supports_vision:
                continue
            if require_reasoning_mode and not caps.supports_reasoning_mode:
                continue
            if min_context > 0 and caps.context_window < min_context:
                continue
            
            # Score the model
            score = self._score_model(caps, task, cost_tier)
            candidates.append((name, caps, score))
        
        if not candidates:
            # Fallback: return any available model
            name = next(iter(self._available_models))
            caps = self._available_models[name]
            logger.warning(
                "no_optimal_model",
                fallback=name,
                task_type=task.task_type.value,
            )
            return (caps.provider, caps.model)
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        best_name, best_caps, best_score = candidates[0]
        
        logger.info(
            "model_selected",
            provider=best_caps.provider,
            model=best_caps.model,
            task_type=task.task_type.value,
            complexity=task.complexity,
            score=round(best_score, 2),
            cost_tier=cost_tier.value,
        )
        
        return (best_caps.provider, best_caps.model)
    
    def _score_model(
        self,
        caps: ModelCapabilities,
        task: TaskAnalysis,
        cost_tier: CostTier,
    ) -> float:
        """
        Score a model for a specific task.
        
        Returns a score from 0-100.
        """
        score = 0.0
        
        # Task type matching (40% weight)
        if task.task_type in caps.best_for:
            score += 40
        else:
            # Partial credit based on relevant capability
            if task.task_type == TaskType.REASONING:
                score += caps.reasoning * 0.4
            elif task.task_type == TaskType.CODING:
                score += caps.coding * 0.4
            elif task.task_type == TaskType.CREATIVE:
                score += caps.creative * 0.4
            elif task.task_type == TaskType.TOOL_USE:
                score += caps.tool_use * 0.4
            elif task.task_type == TaskType.FAST:
                score += caps.speed * 0.4
            elif task.task_type == TaskType.VISION:
                score += 40 if caps.supports_vision else 0
            else:
                score += 20  # Base score for simple tasks
        
        # Complexity matching (20% weight)
        # High complexity tasks benefit from high-capability models
        if task.complexity > 70:
            score += caps.reasoning * 0.2
        elif task.complexity < 30:
            score += caps.speed * 0.2
        else:
            score += 15  # Moderate complexity, moderate credit
        
        # Cost preference (30% weight)
        cost_score = 100 - caps.cost  # Invert: lower cost = higher score
        if cost_tier == CostTier.ECONOMY:
            score += cost_score * 0.4
        elif cost_tier == CostTier.BALANCED:
            # Balance capability and cost
            capability_avg = (caps.reasoning + caps.coding + caps.tool_use) / 3
            score += (cost_score * 0.15) + (capability_avg * 0.15)
        else:  # PERFORMANCE
            capability_avg = (caps.reasoning + caps.coding + caps.tool_use) / 3
            score += capability_avg * 0.3
        
        # Tool-heavy bonus (10% weight)
        if task.tool_heavy:
            score += caps.tool_use * 0.1
        
        return score
    
    def get_available_models(self) -> list[dict[str, Any]]:
        """
        Get list of available models with their capabilities.
        
        Returns:
            List of model info dicts
        """
        return [
            {
                "provider": caps.provider,
                "model": caps.model,
                "best_for": [t.value for t in caps.best_for],
                "supports_vision": caps.supports_vision,
                "supports_reasoning": caps.supports_reasoning_mode,
                "context_window": caps.context_window,
                "cost_tier": (
                    "economy" if caps.cost < 30
                    else "balanced" if caps.cost < 70
                    else "performance"
                ),
            }
            for caps in self._available_models.values()
        ]


def create_auto_selector() -> AutoModelSelector:
    """
    Create an auto selector with the user's configured providers.
    
    Returns:
        Configured AutoModelSelector
    """
    from huxley.cli.config import ConfigManager
    
    manager = ConfigManager()
    configured = manager.get_configured_providers()
    
    return AutoModelSelector(configured)


def auto_select_model(
    prompt: str,
    *,
    cost_tier: str = "balanced",
    has_images: bool = False,
    has_tools: bool = False,
) -> tuple[str, str]:
    """
    Convenience function to auto-select a model.
    
    Args:
        prompt: The task prompt
        cost_tier: "economy", "balanced", or "performance"
        has_images: Whether task includes images
        has_tools: Whether tools are available
        
    Returns:
        Tuple of (provider, model)
    """
    selector = create_auto_selector()
    
    tier = CostTier(cost_tier)
    
    return selector.select_model(
        prompt=prompt,
        cost_tier=tier,
        has_images=has_images,
        has_tools=has_tools,
    )
