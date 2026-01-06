"""
Test script for Huxley setup wizard.

Simulates the setup process without requiring real API keys or database connections.
Run with: python -m tests.test_setup
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from io import StringIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class MockConsole:
    """Mock Rich console for testing."""
    
    def __init__(self):
        self.output = []
        self.status_context = MagicMock()
        self.status_context.__enter__ = MagicMock(return_value=None)
        self.status_context.__exit__ = MagicMock(return_value=None)
    
    def print(self, *args, **kwargs):
        text = " ".join(str(a) for a in args)
        self.output.append(text)
        print(text)  # Also print to real console
    
    def status(self, *args, **kwargs):
        return self.status_context


class MockConfigManager:
    """Mock config manager for testing."""
    
    PROVIDERS = {
        "openai": {
            "name": "OpenAI",
            "url": "https://platform.openai.com/api-keys",
            "models": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        },
        "anthropic": {
            "name": "Anthropic",
            "url": "https://console.anthropic.com/",
            "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
        },
        "google": {
            "name": "Google AI",
            "url": "https://aistudio.google.com/apikey",
            "models": ["gemini-pro", "gemini-pro-vision"],
        },
        "openrouter": {
            "name": "OpenRouter",
            "url": "https://openrouter.ai/keys",
            "compatible": True,
            "base_url": "https://openrouter.ai/api/v1",
            "models": ["anthropic/claude-3-opus", "openai/gpt-4-turbo"],
        },
        "together": {
            "name": "Together AI",
            "url": "https://api.together.xyz/settings/api-keys",
            "compatible": True,
            "base_url": "https://api.together.xyz/v1",
            "models": ["meta-llama/Llama-3-70b-chat-hf"],
        },
        "ollama": {
            "name": "Ollama (Local)",
            "no_key": True,
            "base_url": "http://localhost:11434/v1",
            "models": ["llama3", "mistral", "codellama"],
        },
        "custom": {
            "name": "Custom Endpoint",
            "compatible": True,
            "models": [],
        },
    }
    
    def __init__(self):
        self._config = {
            "api_keys": {},
            "base_urls": {},
            "database": {},
            "redis": {},
            "vector": {},
        }
        self._default_provider = None
        self._default_model = None
    
    def get_api_key(self, provider):
        return self._config["api_keys"].get(provider)
    
    def set_api_key(self, provider, key):
        self._config["api_keys"][provider] = key
        print(f"  [MOCK] Set API key for {provider}: {key[:8]}...")
    
    def get_base_url(self, provider):
        return self._config["base_urls"].get(provider)
    
    def set_base_url(self, provider, url):
        self._config["base_urls"][provider] = url
        print(f"  [MOCK] Set base URL for {provider}: {url}")
    
    def get_default_model(self):
        return self._default_provider, self._default_model
    
    def set_default_model(self, provider, model):
        self._default_provider = provider
        self._default_model = model
        print(f"  [MOCK] Set default model: {model} ({provider})")
    
    def get_configured_providers(self):
        return list(self._config["api_keys"].keys())
    
    def get_weights_status(self):
        return {"denovo": False, "inpainting": False}
    
    def get(self, key):
        return self._config.get(key, {})
    
    def set(self, key, value):
        self._config[key] = value
        print(f"  [MOCK] Set {key}: {value}")


# Simulated user inputs for automated testing
SIMULATED_INPUTS = [
    # API key setup
    "1",                    # Choose OpenAI
    "sk-test-fake-key-12345678901234567890",  # Fake API key
    "",                     # Default base URL
    "1",                    # Choose gpt-4o
    "done",                 # Done with API keys
    
    # Skip weights download
    "n",                    # Don't download weights
    
    # Default model selection
    "yes",                  # Keep current model
    
    # Database setup
    "1",                    # Choose Supabase
    "postgresql://test:test@db.test.supabase.co:5432/postgres",  # Fake Supabase URL
    "n",                    # Don't initialize DB (would fail without real connection)
    
    # Redis setup
    "5",                    # Skip Redis
]

input_index = 0


def mock_ask(prompt, default=None, password=False):
    """Mock ask function that returns simulated inputs."""
    global input_index
    
    if input_index < len(SIMULATED_INPUTS):
        response = SIMULATED_INPUTS[input_index]
        input_index += 1
    else:
        response = default or ""
    
    # Display the simulated interaction
    mask = "****" if password and response else response
    print(f"  ? {prompt} [{default or ''}]: {mask}")
    
    return response if response else default


def mock_confirm(prompt, default=True):
    """Mock confirm function."""
    global input_index
    
    if input_index < len(SIMULATED_INPUTS):
        response = SIMULATED_INPUTS[input_index]
        input_index += 1
        result = response.lower() in ("y", "yes", "true", "1")
    else:
        result = default
    
    print(f"  ? {prompt} [{'Y/n' if default else 'y/N'}]: {'yes' if result else 'no'}")
    return result


async def test_setup_flow():
    """Test the setup wizard flow with mocked inputs."""
    global input_index
    input_index = 0
    
    print("\n" + "=" * 60)
    print("HUXLEY SETUP WIZARD TEST")
    print("=" * 60)
    print("\nThis test simulates the setup wizard with mock inputs.")
    print("No real API calls or database connections are made.\n")
    print("-" * 60)
    
    # Create mocks
    mock_manager = MockConfigManager()
    mock_console = MockConsole()
    
    # Mock the UI functions
    with patch.multiple(
        "huxley.cli.ui",
        console=mock_console,
        ask=mock_ask,
        confirm=mock_confirm,
        print_banner=lambda: print("\n🧬 HUXLEY - Biological Intelligence Framework\n"),
        print_success=lambda msg: print(f"  ✓ {msg}"),
        print_error=lambda msg: print(f"  ✗ {msg}"),
        print_info=lambda msg: print(f"  ℹ {msg}"),
        print_warning=lambda msg: print(f"  ⚠ {msg}"),
        print_step=lambda c, t, s: print(f"\n[Step {c}/{t}] {s}"),
        print_setup_complete=lambda: print("\n✓ Setup complete!\n"),
        rule=lambda: print("-" * 40),
        S_MUTED="dim",
    ):
        # Mock ConfigManager in huxley.cli.config module
        with patch("huxley.cli.config.ConfigManager", return_value=mock_manager):
            # Import the setup functions - they will use the mocked ConfigManager
            from huxley.cli.commands.setup import (
                setup_api_keys,
                setup_default_model,
                setup_database,
                setup_redis,
            )
            
            # Test Step 1: API Keys
            print("\n[Step 1/5] Configure AI Model Access")
            print("-" * 40)
            await setup_api_keys(mock_manager)
            
            # Test Step 2: Skip weights (mocked)
            print("\n[Step 2/5] Download Model Weights")
            print("-" * 40)
            print("  (Skipped in test)")
            
            # Test Step 3: Default Model
            print("\n[Step 3/5] Choose Default Model")
            print("-" * 40)
            await setup_default_model(mock_manager)
            
            # Test Step 4: Database
            print("\n[Step 4/5] Database & Memory")
            print("-" * 40)
            await setup_database(mock_manager)
            
            # Test Step 5: Redis
            print("\n[Step 5/5] Redis Cache")
            print("-" * 40)
            await setup_redis(mock_manager)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print("\nFinal Configuration:")
    print(f"  API Keys:     {list(mock_manager._config['api_keys'].keys())}")
    print(f"  Default Model: {mock_manager._default_model} ({mock_manager._default_provider})")
    print(f"  Database:     {mock_manager._config.get('database', {}).get('type', 'not set')}")
    print(f"  Redis:        {'enabled' if mock_manager._config.get('redis', {}).get('enabled') else 'disabled'}")
    print(f"  Vector Store: {'enabled' if mock_manager._config.get('vector', {}).get('enabled') else 'disabled'}")
    print()
    
    return True


async def test_database_detection():
    """Test database URL auto-detection."""
    print("\n" + "=" * 60)
    print("DATABASE URL DETECTION TEST")
    print("=" * 60)
    
    test_urls = [
        ("postgresql://user:pass@db.supabase.co:5432/postgres", "supabase"),
        ("postgresql://user:pass@ep-cool-name-123.us-east-2.aws.neon.tech/neondb", "neon"),
        ("postgresql://user:pass@localhost:5432/mydb", "postgresql"),
        ("postgresql://user:pass@free-tier.gcp-us-central1.cockroachlabs.cloud:26257/mydb", "cockroachdb"),
        ("sqlite:///~/.huxley/huxley.db", "sqlite"),
    ]
    
    from huxley.cli.commands.config_cmd import _auto_detect_database
    
    print("\nTesting URL detection:\n")
    
    all_passed = True
    for url, expected_provider in test_urls:
        config = {}
        result = _auto_detect_database(url, config)
        detected = result.get("provider", "unknown")
        passed = detected == expected_provider
        
        status = "✓" if passed else "✗"
        print(f"  {status} {url[:50]}...")
        print(f"      Expected: {expected_provider}, Got: {detected}")
        
        if not passed:
            all_passed = False
    
    print()
    return all_passed


async def test_config_commands():
    """Test config command functions."""
    print("\n" + "=" * 60)
    print("CONFIG COMMANDS TEST")
    print("=" * 60)
    
    mock_manager = MockConfigManager()
    
    with patch("huxley.cli.commands.config_cmd.ConfigManager", return_value=mock_manager):
        with patch("huxley.cli.commands.config_cmd.console") as mock_console:
            mock_console.print = print
            
            from huxley.cli.commands.config_cmd import set_config
            
            print("\nTesting config set commands:\n")
            
            # Test setting various config values
            print("  Setting api_keys.openai...")
            set_config("api_keys.openai", "sk-test-key")
            
            print("  Setting database.url...")
            set_config("database.url", "postgresql://user:pass@db.supabase.co:5432/db")
            
            print("  Setting redis.url...")
            set_config("redis.url", "redis://default:xxx@us1-xxx.upstash.io:6379")
            
            print("  Setting vector.url...")
            set_config("vector.url", "postgresql://user:pass@db.supabase.co:5432/db")
    
    print("\n  All config commands executed successfully!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HUXLEY SETUP TEST SUITE")
    print("=" * 60)
    print("\nRunning simulated setup tests...\n")
    
    results = []
    
    # Run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Test 1: Database URL detection
        results.append(("Database Detection", loop.run_until_complete(test_database_detection())))
        
        # Test 2: Config commands
        results.append(("Config Commands", loop.run_until_complete(test_config_commands())))
        
        # Test 3: Full setup flow
        results.append(("Setup Flow", loop.run_until_complete(test_setup_flow())))
        
    finally:
        loop.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed. ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
