"""
Huxley CLI - Check command.

Check setup status and system information.
"""

import asyncio
import shutil
from pathlib import Path


async def run_check():
    """Check Huxley setup status."""
    # Lazy imports
    from huxley.cli.ui import (
        console, print_mini_banner, print_success, print_error,
        print_info, print_warning, print_status_table, rule, S_MUTED
    )
    from huxley.cli.config import ConfigManager
    
    print_mini_banner()
    
    manager = ConfigManager()
    
    # API Keys Status - organized by provider type
    api_items = []
    for provider, info in manager.PROVIDERS.items():
        is_local = info.get("no_key", False)
        is_compatible = info.get("compatible", False)
        
        if is_local:
            # Local providers: show base URL status
            base_url = manager.get_base_url(provider)
            if base_url:
                api_items.append((info["name"], "ready", base_url))
            else:
                api_items.append((info["name"], "not configured", None))
        else:
            # Key-based providers
            key = manager.get_api_key(provider)
            if key:
                # Mask the key
                masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
                # Show base URL for compatible providers
                if is_compatible:
                    base_url = manager.get_base_url(provider)
                    if base_url and base_url != info.get("base_url"):
                        api_items.append((info["name"], "OK", f"{masked} @ custom URL"))
                    else:
                        api_items.append((info["name"], "OK", masked))
                else:
                    api_items.append((info["name"], "OK", masked))
            else:
                api_items.append((info["name"], "not set", None))
    
    print_status_table("API Keys", api_items)
    
    # Model Weights Status
    weights_status = manager.get_weights_status()
    weights_dir = Path.home() / ".huxley" / "models" / "framediff"
    
    weights_items = []
    for model, downloaded in weights_status.items():
        if downloaded:
            path = weights_dir / f"{model}.pth"
            size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
            weights_items.append((f"{model}.pth", "OK", f"{size_mb:.1f} MB"))
        else:
            weights_items.append((f"{model}.pth", "MISSING", None))
    
    print_status_table("Model Weights", weights_items)
    
    # System Status
    import sys
    sys_items = []
    
    # Python
    sys_items.append(("Python", "OK", sys.version.split()[0]))
    
    # PyTorch
    try:
        import torch
        cuda_status = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "CPU"
        sys_items.append(("PyTorch", "OK", f"{torch.__version__} ({cuda_status})"))
    except ImportError:
        sys_items.append(("PyTorch", "MISSING", None))
    
    # git-lfs
    if shutil.which("git-lfs"):
        sys_items.append(("git-lfs", "OK", "installed"))
    else:
        sys_items.append(("git-lfs", "WARNING", "not installed"))
    
    print_status_table("System", sys_items)
    
    # Default Model
    provider, model = manager.get_default_model()
    console.print(f"Default Model: {model or 'not set'}")
    if provider:
        console.print(f"Provider:      {provider}")
    console.print()
    
    # Config Path
    console.print(f"Config: {manager.config_path}", style=S_MUTED)
    console.print(f"Weights: {weights_dir}", style=S_MUTED)
    
    # Summary
    console.print()
    configured = manager.get_configured_providers()
    weights_ready = all(weights_status.values())
    
    if configured and weights_ready:
        print_success("Huxley is fully configured and ready")
    elif configured:
        print_warning("API keys configured, but weights not downloaded")
        print_info("Run 'huxley setup' to download model weights")
    elif weights_ready:
        print_warning("Weights downloaded, but no API keys configured")
        print_info("Run 'huxley setup' to configure API keys")
    else:
        print_warning("Huxley needs to be set up")
        print_info("Run 'huxley setup' to get started")
