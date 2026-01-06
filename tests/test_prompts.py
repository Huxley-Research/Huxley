"""Tests for model-specific prompt templates."""

import pytest
from huxley.prompts.templates import (
    BaseSystemPrompt,
    PromptContext,
    ClaudeOpusPrompt,
    ClaudeSonnetPrompt,
    ClaudeHaikuPrompt,
    GeminiProPrompt,
    GeminiFlashPrompt,
    GPT52ProPrompt,
    GPT52Prompt,
    Grok4Prompt,
    get_prompt_class,
    create_system_prompt,
    MODEL_PROMPT_MAP,
)
from huxley.prompts.registry import (
    PromptRegistry,
    get_prompt_registry,
    get_system_prompt,
)


class TestPromptContext:
    """Tests for PromptContext dataclass."""
    
    def test_default_values(self):
        """Test default context values."""
        ctx = PromptContext()
        assert ctx.agent_name == "Huxley"
        assert ctx.domain == "general"
        assert ctx.available_tools == []
        assert ctx.constraints == []
        assert ctx.output_format is None
        assert ctx.knowledge_cutoff == "January 2025"
    
    def test_custom_values(self):
        """Test custom context values."""
        ctx = PromptContext(
            agent_name="BioBot",
            domain="molecular biology",
            available_tools=["pdb_search", "pdb_get_entry"],
            constraints=["Only use verified data sources"],
            output_format="JSON",
        )
        assert ctx.agent_name == "BioBot"
        assert ctx.domain == "molecular biology"
        assert len(ctx.available_tools) == 2
        assert len(ctx.constraints) == 1


class TestClaudePrompts:
    """Tests for Claude model prompts."""
    
    @pytest.fixture
    def context(self):
        return PromptContext(
            agent_name="TestAgent",
            domain="testing",
            available_tools=["tool1", "tool2"],
            current_date="January 4, 2026",
        )
    
    def test_claude_opus_structure(self, context):
        """Test Claude Opus prompt has XML structure."""
        prompt = ClaudeOpusPrompt()
        result = prompt.generate(context)
        
        # Should use XML tags
        assert "<identity>" in result
        assert "</identity>" in result
        assert "<capabilities>" in result
        assert "<behavioral_guidelines>" in result
        assert "<execution_protocol>" in result
        
        # Should include context
        assert "TestAgent" in result
        assert "testing" in result
        assert "tool1" in result
    
    def test_claude_sonnet_concise(self, context):
        """Test Claude Sonnet prompt is more concise."""
        opus = ClaudeOpusPrompt().generate(context)
        sonnet = ClaudeSonnetPrompt().generate(context)
        
        # Sonnet should be shorter than Opus
        assert len(sonnet) < len(opus)
        
        # But still have XML structure
        assert "<role>" in sonnet
        assert "<tools>" in sonnet
    
    def test_claude_haiku_minimal(self, context):
        """Test Claude Haiku prompt is minimal."""
        sonnet = ClaudeSonnetPrompt().generate(context)
        haiku = ClaudeHaikuPrompt().generate(context)
        
        # Haiku should be shortest
        assert len(haiku) < len(sonnet)
        
        # Still functional
        assert "TestAgent" in haiku
        assert "tool1" in haiku


class TestGeminiPrompts:
    """Tests for Gemini model prompts."""
    
    @pytest.fixture
    def context(self):
        return PromptContext(
            agent_name="GeminiAgent",
            domain="research",
            available_tools=["search", "analyze"],
            current_date="January 4, 2026",
        )
    
    def test_gemini_pro_structure(self, context):
        """Test Gemini Pro prompt has proper structure."""
        prompt = GeminiProPrompt()
        result = prompt.generate(context)
        
        # Uses XML tags per Gemini guidelines
        assert "<role>" in result
        assert "<instructions>" in result
        assert "<tools>" in result
        
        # Includes planning instructions
        assert "Plan" in result
        assert "Execute" in result
        assert "Validate" in result
    
    def test_gemini_flash_concise(self, context):
        """Test Gemini Flash is more concise."""
        pro = GeminiProPrompt().generate(context)
        flash = GeminiFlashPrompt().generate(context)
        
        assert len(flash) < len(pro)
        assert "GeminiAgent" in flash
    
    def test_gemini_time_sensitive_clause(self, context):
        """Test Gemini Flash includes time-sensitive instructions."""
        flash = GeminiFlashPrompt().generate(context)
        
        # Should have current date context
        assert "January 4, 2026" in flash
        assert "2026" in flash


class TestGPTPrompts:
    """Tests for GPT model prompts."""
    
    @pytest.fixture
    def context(self):
        return PromptContext(
            agent_name="GPTAgent",
            domain="coding",
            available_tools=["run_code", "analyze_code"],
        )
    
    def test_gpt52_pro_markdown(self, context):
        """Test GPT-5.2 Pro uses Markdown structure."""
        prompt = GPT52ProPrompt()
        result = prompt.generate(context)
        
        # Uses Markdown headers
        assert "# Identity" in result
        assert "# Instructions" in result
        assert "## Core Behaviors" in result
        assert "## Tool Usage" in result
    
    def test_gpt52_concise(self, context):
        """Test GPT-5.2 standard is more concise."""
        pro = GPT52ProPrompt().generate(context)
        standard = GPT52Prompt().generate(context)
        
        assert len(standard) < len(pro)
        assert "GPTAgent" in standard


class TestGrokPrompts:
    """Tests for Grok model prompts."""
    
    @pytest.fixture
    def context(self):
        return PromptContext(
            agent_name="GrokAgent",
            domain="research",
            available_tools=["web_search", "x_search", "code_execution"],
        )
    
    def test_grok4_structure(self, context):
        """Test Grok 4 prompt has proper agentic structure."""
        prompt = Grok4Prompt()
        result = prompt.generate(context)
        
        # Uses Markdown headers
        assert "# Identity" in result
        assert "# Agentic Capabilities" in result
        assert "# Tools" in result
        
        # Has agentic reasoning instructions
        assert "iterative" in result.lower() or "loop" in result.lower()
        assert "GrokAgent" in result
    
    def test_grok4_tool_section(self, context):
        """Test Grok 4 includes tools."""
        prompt = Grok4Prompt().generate(context)
        
        assert "web_search" in prompt
        assert "x_search" in prompt
        assert "code_execution" in prompt
    
    def test_grok4_agentic_workflow(self, context):
        """Test Grok 4 describes agentic workflow."""
        prompt = Grok4Prompt().generate(context)
        
        # Should describe the agentic loop
        assert "Analyze" in prompt or "analyze" in prompt
        assert "Execute" in prompt or "execute" in prompt


class TestPromptClassLookup:
    """Tests for prompt class lookup functions."""
    
    def test_exact_match(self):
        """Test exact model name lookup."""
        assert get_prompt_class("claude-4.5-opus") == ClaudeOpusPrompt
        assert get_prompt_class("gemini-3-pro") == GeminiProPrompt
        assert get_prompt_class("gpt-5.2-pro") == GPT52ProPrompt
        assert get_prompt_class("grok-4") == Grok4Prompt
    
    def test_case_insensitive(self):
        """Test case insensitive lookup."""
        assert get_prompt_class("Claude-4.5-Opus") == ClaudeOpusPrompt
        assert get_prompt_class("GEMINI-3-PRO") == GeminiProPrompt
        assert get_prompt_class("GROK-4") == Grok4Prompt
    
    def test_family_fallback(self):
        """Test fallback to family defaults."""
        # Unknown Claude variant -> Sonnet default
        assert get_prompt_class("claude-unknown") == ClaudeSonnetPrompt
        
        # Unknown Gemini variant -> Flash default  
        assert get_prompt_class("gemini-unknown") == GeminiFlashPrompt
        
        # Unknown GPT variant -> Standard default
        assert get_prompt_class("gpt-unknown") == GPT52Prompt
        
        # Unknown Grok variant -> Grok4 default
        assert get_prompt_class("grok-unknown") == Grok4Prompt
    
    def test_unknown_model_error(self):
        """Test error for completely unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_prompt_class("unknown-model-xyz")
    
    def test_alternate_naming(self):
        """Test alternate naming conventions."""
        # Dashes vs dots
        assert get_prompt_class("claude-4-5-opus") == ClaudeOpusPrompt
        assert get_prompt_class("gpt-5-2-pro") == GPT52ProPrompt


class TestCreateSystemPrompt:
    """Tests for the create_system_prompt function."""
    
    def test_with_context(self):
        """Test creating prompt with context object."""
        ctx = PromptContext(
            agent_name="CustomAgent",
            domain="science",
        )
        prompt = create_system_prompt("claude-4.5-sonnet", context=ctx)
        
        assert "CustomAgent" in prompt
        assert "science" in prompt
    
    def test_with_kwargs(self):
        """Test creating prompt with keyword arguments."""
        prompt = create_system_prompt(
            "gemini-3-flash",
            agent_name="KwargsAgent",
            domain="math",
        )
        
        assert "KwargsAgent" in prompt
        assert "math" in prompt


class TestPromptRegistry:
    """Tests for the PromptRegistry class."""
    
    def test_list_models(self):
        """Test listing all registered models."""
        registry = PromptRegistry()
        models = registry.list_models()
        
        assert len(models) > 0
        assert "claude-4.5-opus" in models
        assert "gemini-3-pro" in models
        assert "gpt-5.2" in models
    
    def test_list_model_families(self):
        """Test listing models by family."""
        registry = PromptRegistry()
        families = registry.list_model_families()
        
        assert "claude" in families
        assert "gemini" in families
        assert "openai" in families
        assert "xai" in families
        
        assert len(families["claude"]) >= 3
        assert len(families["gemini"]) >= 2
        assert len(families["xai"]) >= 1
    
    def test_custom_prompt_registration(self):
        """Test registering custom prompts."""
        registry = PromptRegistry()
        
        class CustomPrompt(BaseSystemPrompt):
            model_family = "custom"
            model_name = "my-model"
            
            def generate(self, context):
                return f"Custom prompt for {context.agent_name}"
        
        registry.register("my-custom-model", CustomPrompt)
        
        prompt = registry.generate("my-custom-model")
        assert "Custom prompt" in prompt
    
    def test_custom_overrides_builtin(self):
        """Test custom prompt overrides built-in."""
        registry = PromptRegistry()
        
        class OverridePrompt(BaseSystemPrompt):
            def generate(self, context):
                return "OVERRIDE"
        
        # Override Claude Opus
        registry.register("claude-4.5-opus", OverridePrompt)
        
        prompt = registry.generate("claude-4.5-opus")
        assert prompt == "OVERRIDE"


class TestConvenienceFunction:
    """Tests for the get_system_prompt convenience function."""
    
    def test_basic_usage(self):
        """Test basic function usage."""
        prompt = get_system_prompt("claude-4.5-sonnet")
        
        assert len(prompt) > 100
        assert "Huxley" in prompt  # Default agent name
    
    def test_with_tools(self):
        """Test with tool list."""
        prompt = get_system_prompt(
            "gemini-3-pro",
            available_tools=["pdb_search", "pdb_get_entry", "sequence_align"],
        )
        
        assert "pdb_search" in prompt
        assert "pdb_get_entry" in prompt
        assert "sequence_align" in prompt
    
    def test_with_constraints(self):
        """Test with constraints."""
        prompt = get_system_prompt(
            "gpt-5.2-pro",
            constraints=[
                "Only use peer-reviewed sources",
                "Cite all references",
            ],
        )
        
        assert "peer-reviewed" in prompt
        assert "Cite all references" in prompt
    
    def test_biology_agent_example(self):
        """Test creating a biology-focused agent prompt."""
        prompt = get_system_prompt(
            "claude-4.5-opus",
            agent_name="BioHuxley",
            domain="structural biology and protein analysis",
            available_tools=[
                "pdb_search",
                "pdb_get_entry",
                "pdb_sequence_search",
                "pdb_structure_summary",
            ],
            constraints=[
                "Only provide information backed by PDB data",
                "Include PDB IDs when referencing structures",
                "Acknowledge uncertainty in structural interpretations",
            ],
            output_format="Include relevant PDB IDs and provide structured analysis",
            current_date="January 4, 2026",
        )
        
        # Should be comprehensive
        assert "BioHuxley" in prompt
        assert "structural biology" in prompt
        assert "pdb_search" in prompt
        assert "PDB IDs" in prompt


class TestModelFamilyCharacteristics:
    """Tests ensuring prompts match model family characteristics."""
    
    def test_claude_uses_xml(self):
        """Test all Claude prompts use XML tags."""
        ctx = PromptContext()
        
        for cls in [ClaudeOpusPrompt, ClaudeSonnetPrompt, ClaudeHaikuPrompt]:
            prompt = cls().generate(ctx)
            # Should have at least some XML-style tags
            assert "<" in prompt and ">" in prompt
    
    def test_gpt_uses_markdown(self):
        """Test GPT prompts use Markdown headers."""
        ctx = PromptContext()
        
        for cls in [GPT52ProPrompt, GPT52Prompt]:
            prompt = cls().generate(ctx)
            assert "# " in prompt  # Markdown header
    
    def test_prompt_size_hierarchy(self):
        """Test prompt sizes follow capability hierarchy."""
        ctx = PromptContext(
            available_tools=["tool1", "tool2"],
            constraints=["constraint1"],
        )
        
        # Within Claude family: Opus > Sonnet > Haiku
        opus = ClaudeOpusPrompt().generate(ctx)
        sonnet = ClaudeSonnetPrompt().generate(ctx)
        haiku = ClaudeHaikuPrompt().generate(ctx)
        
        assert len(opus) > len(sonnet) > len(haiku)
        
        # Within GPT family: Pro > Standard
        pro = GPT52ProPrompt().generate(ctx)
        standard = GPT52Prompt().generate(ctx)
        
        assert len(pro) > len(standard)
