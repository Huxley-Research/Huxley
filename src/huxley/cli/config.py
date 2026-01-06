"""
Huxley CLI - Configuration management.

Handles API keys, settings, and persistent configuration.
"""

import json
import os
from pathlib import Path
from typing import Any


class ConfigManager:
    """Manage Huxley configuration."""
    
    # Default config location
    DEFAULT_CONFIG_DIR = Path.home() / ".huxley"
    DEFAULT_CONFIG_FILE = "config.json"
    
    # Supported API providers
    PROVIDERS = {
        "openai": {
            "name": "OpenAI",
            "env_var": "OPENAI_API_KEY",
            "base_url_env": "OPENAI_BASE_URL",
            "models": [
                "gpt-5.2-pro",
                "gpt-5.2",
            ],
            "url": "https://platform.openai.com/api-keys",
            "compatible": False,
        },
        "anthropic": {
            "name": "Anthropic",
            "env_var": "ANTHROPIC_API_KEY",
            "models": [
                "claude-4.5-opus",
                "claude-4.5-sonnet",
                "claude-4.5-haiku",
            ],
            "url": "https://console.anthropic.com/",
            "compatible": False,
        },
        "google": {
            "name": "Google AI (Gemini)",
            "env_var": "GOOGLE_API_KEY",
            "models": [
                "gemini-3-pro",
                "gemini-3-flash",
            ],
            "url": "https://makersuite.google.com/app/apikey",
            "compatible": False,
        },
        "xai": {
            "name": "xAI (Grok)",
            "env_var": "XAI_API_KEY",
            "base_url": "https://api.x.ai/v1",
            "models": [
                "grok-4",
            ],
            "url": "https://x.ai/",
            "compatible": True,
        },
        "cohere": {
            "name": "Cohere",
            "env_var": "COHERE_API_KEY",
            "models": [
                "command-a-03-2025",
                "command-a-reasoning",
                "command-a-vision",
            ],
            "url": "https://dashboard.cohere.com/api-keys",
            "compatible": False,
        },
        "openrouter": {
            "name": "OpenRouter",
            "env_var": "OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
            "models": [
                "anthropic/claude-4.5-sonnet",
                "openai/gpt-5.2",
                "google/gemini-3-flash",
                "xai/grok-4",
            ],
            "url": "https://openrouter.ai/keys",
            "compatible": True,
        },
        "together": {
            "name": "Together.ai",
            "env_var": "TOGETHER_API_KEY",
            "base_url": "https://api.together.xyz/v1",
            "models": [],
            "url": "https://api.together.xyz/",
            "compatible": True,
        },
        "ollama": {
            "name": "Ollama (Local)",
            "env_var": None,
            "base_url": "http://localhost:11434/v1",
            "models": [],
            "url": "https://ollama.ai/",
            "compatible": True,
            "no_key": True,
        },
        "lmstudio": {
            "name": "LM Studio (Local)",
            "env_var": None,
            "base_url": "http://localhost:1234/v1",
            "models": [],
            "url": "https://lmstudio.ai/",
            "compatible": True,
            "no_key": True,
        },
        "custom": {
            "name": "Custom OpenAI-Compatible",
            "env_var": "CUSTOM_API_KEY",
            "base_url": None,  # User must provide
            "models": [],
            "url": None,
            "compatible": True,
        },
    }
    
    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self.config_path = self.config_dir / self.DEFAULT_CONFIG_FILE
        self._config: dict[str, Any] = {}
        self._load()
    
    def _load(self) -> None:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._config = {}
        else:
            self._config = {}
    
    def _save(self) -> None:
        """Save configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value
        self._save()
    
    def delete(self, key: str) -> None:
        """Delete a configuration value."""
        if key in self._config:
            del self._config[key]
            self._save()
    
    def get_all(self) -> dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()
    
    # API Key Management
    def set_api_key(self, provider: str, key: str) -> None:
        """Set an API key for a provider."""
        api_keys = self._config.get("api_keys", {})
        api_keys[provider] = key
        self._config["api_keys"] = api_keys
        self._save()
        
        # Also set environment variable for current session
        if provider in self.PROVIDERS:
            env_var = self.PROVIDERS[provider].get("env_var")
            if env_var:
                os.environ[env_var] = key
    
    def get_api_key(self, provider: str) -> str | None:
        """Get an API key for a provider."""
        # First check config
        api_keys = self._config.get("api_keys", {})
        if provider in api_keys:
            return api_keys[provider]
        
        # Then check environment
        if provider in self.PROVIDERS:
            env_var = self.PROVIDERS[provider].get("env_var")
            if env_var:
                return os.environ.get(env_var)
        
        return None
    
    def get_configured_providers(self) -> list[str]:
        """Get list of providers with API keys configured."""
        configured = []
        for provider, info in self.PROVIDERS.items():
            # Local providers don't need keys
            if info.get("no_key"):
                configured.append(provider)
            elif self.get_api_key(provider):
                configured.append(provider)
        return configured
    
    def load_api_keys_to_env(self) -> None:
        """Load all stored API keys into environment variables."""
        api_keys = self._config.get("api_keys", {})
        for provider, key in api_keys.items():
            if provider in self.PROVIDERS:
                env_var = self.PROVIDERS[provider].get("env_var")
                if env_var and key and not os.environ.get(env_var):
                    os.environ[env_var] = key
        
        # Also load base URLs
        base_urls = self._config.get("base_urls", {})
        for provider, url in base_urls.items():
            if url:
                os.environ[f"{provider.upper()}_BASE_URL"] = url
    
    # Base URL Management
    def set_base_url(self, provider: str, url: str) -> None:
        """Set a custom base URL for a provider."""
        base_urls = self._config.get("base_urls", {})
        base_urls[provider] = url
        self._config["base_urls"] = base_urls
        self._save()
    
    def get_base_url(self, provider: str) -> str | None:
        """Get the base URL for a provider."""
        # First check config
        base_urls = self._config.get("base_urls", {})
        if provider in base_urls:
            return base_urls[provider]
        
        # Then use default from provider definition
        if provider in self.PROVIDERS:
            return self.PROVIDERS[provider].get("base_url")
        
        return None
    
    def get_provider_config(self, provider: str) -> dict[str, Any]:
        """Get full configuration for a provider including key and base URL."""
        return {
            "name": self.PROVIDERS.get(provider, {}).get("name", provider),
            "api_key": self.get_api_key(provider),
            "base_url": self.get_base_url(provider),
            "compatible": self.PROVIDERS.get(provider, {}).get("compatible", False),
            "no_key": self.PROVIDERS.get(provider, {}).get("no_key", False),
        }
    
    # Model Weights
    def get_weights_status(self) -> dict[str, bool]:
        """Check which model weights are downloaded."""
        weights_dir = Path.home() / ".huxley" / "models" / "framediff"
        return {
            "denovo": (weights_dir / "denovo.pth").exists(),
            "inpainting": (weights_dir / "inpainting.pth").exists(),
        }
    
    # Default Model
    def set_default_model(self, provider: str, model: str) -> None:
        """Set the default LLM model."""
        self._config["default_provider"] = provider
        self._config["default_model"] = model
        self._save()
    
    def get_default_model(self) -> tuple[str | None, str | None]:
        """Get the default LLM model."""
        return (
            self._config.get("default_provider"),
            self._config.get("default_model"),
        )
    
    def get_available_models(self) -> list[str]:
        """
        Get list of all models available from configured providers.
        
        Returns a flat list of model names that can be used for auto-selection.
        """
        available = []
        for provider in self.get_configured_providers():
            if provider in self.PROVIDERS:
                models = self.PROVIDERS[provider].get("models", [])
                available.extend(models)
        return available
