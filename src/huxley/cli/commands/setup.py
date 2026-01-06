"""
Huxley CLI - Setup command.

Interactive wizard for first-time setup.
"""

import asyncio


async def run_setup(skip_weights: bool = False, skip_keys: bool = False):
    """Run the interactive setup wizard."""
    # Lazy imports to avoid triggering tool registration at CLI startup
    from huxley.cli.ui import (
        console, print_banner, print_success, print_error, 
        print_info, print_warning, print_step, print_setup_complete,
        ask, confirm, rule
    )
    from huxley.cli.config import ConfigManager
    
    print_banner()
    
    console.print("Setup Wizard")
    rule()
    console.print()
    
    manager = ConfigManager()
    total_steps = 5
    current_step = 0
    
    # Step 1: API Keys
    if not skip_keys:
        current_step += 1
        print_step(current_step, total_steps, "Configure AI Model Access")
        console.print()
        
        await setup_api_keys(manager)
        console.print()
    
    # Step 2: Download Weights
    if not skip_weights:
        current_step += 1
        print_step(current_step, total_steps, "Download Model Weights")
        console.print()
        
        await setup_weights(manager)
        console.print()
    
    # Step 3: Choose Default Model
    current_step += 1
    print_step(current_step, total_steps, "Choose Default Model")
    console.print()
    
    await setup_default_model(manager)
    console.print()
    
    # Step 4: Database Configuration
    current_step += 1
    print_step(current_step, total_steps, "Database & Memory (Optional)")
    console.print()
    
    await setup_database(manager)
    console.print()
    
    # Step 5: Redis Configuration
    current_step += 1
    print_step(current_step, total_steps, "Redis Cache (Optional)")
    console.print()
    
    await setup_redis(manager)
    console.print()
    
    # Done!
    print_setup_complete()


async def setup_api_keys(manager):
    """Configure API keys."""
    from huxley.cli.ui import console, print_success, print_warning, print_info, ask, rule, S_MUTED
    from rich.text import Text
    
    console.print("  Huxley supports multiple AI providers.")
    console.print("  You only need to configure one.")
    console.print()
    
    # Group providers by type
    cloud_providers = []
    compatible_providers = []
    local_providers = []
    
    for key, info in manager.PROVIDERS.items():
        if info.get("no_key"):
            local_providers.append((key, info))
        elif info.get("compatible"):
            compatible_providers.append((key, info))
        else:
            cloud_providers.append((key, info))
    
    # Show provider options by category
    console.print("  NATIVE API PROVIDERS")
    idx = 1
    provider_list = []
    for key, info in cloud_providers:
        existing = manager.get_api_key(key)
        status = Text("configured", style=S_MUTED) if existing else Text("not set", style="dim")
        console.print(f"    {idx}. {info['name']:<26}", end="")
        console.print(status)
        provider_list.append((key, info))
        idx += 1
    
    console.print()
    console.print("  OPENAI-COMPATIBLE PROVIDERS")
    for key, info in compatible_providers:
        existing = manager.get_api_key(key)
        status = Text("configured", style=S_MUTED) if existing else Text("not set", style="dim")
        console.print(f"    {idx}. {info['name']:<26}", end="")
        console.print(status)
        provider_list.append((key, info))
        idx += 1
    
    console.print()
    console.print("  LOCAL MODELS (NO API KEY)")
    for key, info in local_providers:
        base_url = manager.get_base_url(key)
        status = Text(f"ready ({base_url})", style=S_MUTED) if base_url else Text(info.get("base_url", ""), style="dim")
        console.print(f"    {idx}. {info['name']:<24}", end="")
        console.print(status)
        provider_list.append((key, info))
        idx += 1
    
    console.print()
    
    max_choice = len(provider_list)
    
    # Ask which to configure
    while True:
        choice = ask(
            f"Which provider to configure? (1-{max_choice}, or 'done')",
            default="done"
        )
        
        if choice.lower() == "done":
            break
        
        try:
            choice_idx = int(choice) - 1
            provider_key, provider_info = provider_list[choice_idx]
            
            console.print()
            console.print(f"  {provider_info['name']}")
            
            # Handle local providers (no API key needed)
            if provider_info.get("no_key"):
                console.print(f"  No API key required.")
                default_url = provider_info.get("base_url", "")
                current_url = manager.get_base_url(provider_key) or default_url
                
                new_url = ask(f"Base URL", default=current_url)
                if new_url:
                    manager.set_base_url(provider_key, new_url)
                    print_success(f"{provider_info['name']} configured at {new_url}")
                    
                    # Prompt for model ID
                    console.print()
                    _prompt_for_model(console, manager, provider_key, provider_info, ask, print_success, print_info, S_MUTED)
                
            # Handle custom provider
            elif provider_key == "custom":
                console.print("  Custom OpenAI-compatible endpoint")
                
                base_url = ask("Base URL (e.g., http://localhost:8080/v1)")
                if base_url:
                    manager.set_base_url(provider_key, base_url)
                    
                    api_key = ask("API key (optional, press Enter to skip)", password=True)
                    if api_key:
                        manager.set_api_key(provider_key, api_key)
                    
                    print_success(f"Custom endpoint configured: {base_url}")
                    
                    # Prompt for model ID (required for custom)
                    console.print()
                    console.print("  Model ID is required for custom endpoints.")
                    model_id = ask("Model ID")
                    if model_id:
                        manager.set_default_model(provider_key, model_id)
                        print_success(f"Model: {model_id}")
                else:
                    print_warning("No URL entered, skipping")
                    
            # Handle OpenAI-compatible cloud providers
            elif provider_info.get("compatible"):
                if provider_info.get("url"):
                    console.print(f"  Get API key: {provider_info['url']}")
                
                api_key = ask("API key", password=True)
                
                if api_key:
                    manager.set_api_key(provider_key, api_key)
                    print_success(f"{provider_info['name']} API key saved")
                    
                    # Allow custom base URL override
                    default_url = provider_info.get("base_url", "")
                    override = ask(f"Base URL", default=default_url)
                    if override and override != default_url:
                        manager.set_base_url(provider_key, override)
                    
                    # Prompt for model ID
                    console.print()
                    _prompt_for_model(console, manager, provider_key, provider_info, ask, print_success, print_info, S_MUTED)
                else:
                    print_warning("No key entered, skipping")
                    
            # Handle native API providers
            else:
                console.print(f"  Get API key: {provider_info['url']}")
                
                api_key = ask("API key", password=True)
                
                if api_key:
                    manager.set_api_key(provider_key, api_key)
                    print_success(f"{provider_info['name']} API key saved")
                    
                    # Prompt for model ID
                    console.print()
                    _prompt_for_model(console, manager, provider_key, provider_info, ask, print_success, print_info, S_MUTED)
                else:
                    print_warning("No key entered, skipping")
            
            console.print()
            
        except (ValueError, IndexError):
            from huxley.cli.ui import print_error
            print_error(f"Invalid choice. Enter 1-{max_choice} or 'done'")


def _prompt_for_model(console, manager, provider_key, provider_info, ask, print_success, print_info, S_MUTED):
    """Prompt user to select or enter a model ID."""
    from rich.text import Text
    from huxley.cli.ui import print_warning
    
    recommended = provider_info.get("models", [])
    
    console.print("  Select a model:")
    console.print()
    
    if recommended:
        console.print("  Recommended:")
        for i, model in enumerate(recommended, 1):
            console.print(f"    {i}. {model}")
        console.print(f"    {len(recommended) + 1}. [enter custom model ID]")
        console.print()
        
        choice = ask(f"Model (1-{len(recommended) + 1})", default="1")
        
        try:
            idx = int(choice) - 1
            if idx < len(recommended):
                model_id = recommended[idx]
            else:
                model_id = ask("Model ID")
        except ValueError:
            # User entered a model name directly
            model_id = choice
    else:
        console.print("  No recommended models for this provider.")
        model_id = ask("Model ID")
    
    if model_id:
        manager.set_default_model(provider_key, model_id)
        print_success(f"Default model: {model_id}")
    
    # Check if any providers configured
    configured = manager.get_configured_providers()
    if configured:
        print_success(f"Configured: {', '.join(configured)}")
    else:
        print_warning("No providers configured. Add later with 'huxley config set'")


async def setup_weights(manager):
    """Download FrameDiff model weights."""
    from huxley.cli.ui import console, print_success, print_info, print_warning, confirm
    
    weights_status = manager.get_weights_status()
    
    if all(weights_status.values()):
        print_success("All model weights already downloaded")
        return
    
    console.print("  FrameDiff models enable protein structure generation.")
    console.print("  Download size: ~270MB")
    console.print()
    
    if not confirm("Download model weights?", default=True):
        print_info("Skipping. Download later with 'huxley setup'")
        return
    
    # Import here to avoid slow startup - this triggers tool registration
    from huxley import download_framediff_weights
    
    console.print()
    
    # Download denovo model
    if not weights_status["denovo"]:
        with console.status("Downloading denovo model..."):
            result = await download_framediff_weights("denovo")
        
        if result.get("success"):
            print_success(f"denovo.pth ({result.get('size_mb', '?')} MB)")
        else:
            from huxley.cli.ui import print_error
            print_error(f"Download failed: {result.get('error')}")
    else:
        print_success("denovo.pth (already downloaded)")
    
    # Download inpainting model
    if not weights_status["inpainting"]:
        with console.status("Downloading inpainting model..."):
            result = await download_framediff_weights("inpainting")
        
        if result.get("success"):
            print_success(f"inpainting.pth ({result.get('size_mb', '?')} MB)")
        else:
            from huxley.cli.ui import print_error
            print_error(f"Download failed: {result.get('error')}")
    else:
        print_success("inpainting.pth (already downloaded)")


async def setup_default_model(manager):
    """Choose default LLM model."""
    from huxley.cli.ui import console, print_success, print_warning, print_info, ask, S_MUTED
    from rich.text import Text
    
    # Check if a default model was already set during provider configuration
    existing_provider, existing_model = manager.get_default_model()
    if existing_model:
        print_success(f"Default model already configured: {existing_model}")
        
        change = ask("Keep this model? (yes/no)", default="yes")
        if change.lower() in ("yes", "y"):
            return
        console.print()
    
    configured = manager.get_configured_providers()
    
    # Filter out local providers without a configured model
    # (local providers are always "configured" but may not have a model set)
    providers_with_models = []
    for provider in configured:
        info = manager.PROVIDERS[provider]
        if info.get("no_key"):
            # Local provider - only include if user has set a model for it
            # Check if there's a stored model for this provider
            stored_provider, stored_model = manager.get_default_model()
            if stored_provider == provider and stored_model:
                providers_with_models.append(provider)
        else:
            # API provider with key configured
            providers_with_models.append(provider)
    
    if not providers_with_models:
        print_warning("No AI providers with models configured.")
        print_info("Protein generation works without an AI model.")
        return
    
    console.print("  Select default AI model for chat and analysis.")
    console.print()
    
    # Show options
    options = []
    for provider in providers_with_models:
        info = manager.PROVIDERS[provider]
        models = info.get("models", [])
        
        if models:
            for model in models[:3]:  # Show top 3 models per provider
                options.append((provider, model, info["name"]))
    
    if not options:
        print_warning("No recommended models available.")
        model_id = ask("Enter model ID")
        if model_id and providers_with_models:
            manager.set_default_model(providers_with_models[0], model_id)
            print_success(f"Default: {model_id}")
        return
    
    console.print("  Available Models:")
    for i, (provider, model, name) in enumerate(options, 1):
        console.print(f"    {i}. {model:<36}", end="")
        console.print(Text(name, style=S_MUTED))
    
    # Option for custom model name
    console.print(f"    {len(options) + 1}. [enter custom model ID]")
    
    console.print()
    
    choice = ask(f"Choose default model (1-{len(options) + 1})", default="1")
    
    try:
        idx = int(choice) - 1
        
        if idx == len(options):
            # Custom model name
            model_name = ask("Enter model name")
            if model_name:
                # Find which provider to use
                if len(configured) == 1:
                    provider = configured[0]
                else:
                    console.print()
                    console.print("  Which provider?")
                    for i, p in enumerate(configured, 1):
                        console.print(f"    {i}. {p}")
                    p_choice = ask(f"Provider (1-{len(configured)})", default="1")
                    provider = configured[int(p_choice) - 1]
                
                manager.set_default_model(provider, model_name)
                print_success(f"Default: {model_name} ({provider})")
            else:
                print_warning("No model name entered")
        else:
            provider, model, name = options[idx]
            manager.set_default_model(provider, model)
            print_success(f"Default: {model}")
            
    except (ValueError, IndexError):
        print_warning("Invalid choice, using first available model")
        if options:
            provider, model, _ = options[0]
            manager.set_default_model(provider, model)


async def setup_database(manager):
    """Configure database for persistent memory."""
    from huxley.cli.ui import console, print_success, print_warning, print_info, ask, confirm, S_MUTED
    from rich.text import Text
    
    console.print("  Huxley can use a database for persistent AI memory.")
    console.print("  This enables conversation history and learned context.")
    console.print()
    
    # Check existing config
    db_config = manager.get("database") or {}
    if db_config.get("type") and db_config.get("type") != "memory":
        console.print(f"  Current: {db_config.get('type')} ", end="")
        console.print(Text("configured", style=S_MUTED))
        if not confirm("Reconfigure database?", default=False):
            return
        console.print()
    
    # Database options
    console.print("  SUPPORTED DATABASES")
    console.print()
    console.print("    1. [bold green]Supabase[/bold green]      PostgreSQL with pgvector (recommended)")
    console.print("    2. [bold cyan]Neon[/bold cyan]           Serverless PostgreSQL with pgvector")
    console.print("    3. PostgreSQL    Standard PostgreSQL")
    console.print("    4. SQLite        Local file database")
    console.print("    5. Skip          Use in-memory (no persistence)")
    console.print()
    
    choice = ask("Choose database (1-5)", default="5")
    
    try:
        idx = int(choice)
        
        if idx == 1:
            # Supabase
            console.print()
            console.print("  [bold green]Supabase Setup[/bold green]")
            console.print("  Get your connection string from:")
            console.print("  Project Settings → Database → Connection string → URI")
            console.print()
            
            url = ask("Database URL (postgresql://...)")
            if url:
                db_config = _configure_database(manager, url, "supabase")
                await _setup_vector_extension(manager, db_config)
            else:
                print_warning("Skipped - no URL provided")
                
        elif idx == 2:
            # Neon
            console.print()
            console.print("  [bold cyan]Neon Setup[/bold cyan]")
            console.print("  Get your connection string from:")
            console.print("  Dashboard → Your Project → Connection Details")
            console.print()
            
            url = ask("Database URL (postgresql://...@...neon.tech/...)")
            if url:
                db_config = _configure_database(manager, url, "neon")
                await _setup_vector_extension(manager, db_config)
            else:
                print_warning("Skipped - no URL provided")
                
        elif idx == 3:
            # PostgreSQL
            console.print()
            console.print("  [bold]PostgreSQL Setup[/bold]")
            console.print()
            
            url = ask("Database URL (postgresql://user:pass@host:5432/db)")
            if url:
                db_config = _configure_database(manager, url, "postgresql")
                await _setup_vector_extension(manager, db_config)
            else:
                print_warning("Skipped - no URL provided")
                
        elif idx == 4:
            # SQLite
            console.print()
            console.print("  [bold]SQLite Setup[/bold]")
            console.print()
            
            default_path = "~/.huxley/huxley.db"
            path = ask(f"Database path", default=default_path)
            
            db_config = {
                "type": "sqlite",
                "provider": "sqlite",
                "url": f"sqlite:///{path}",
            }
            manager.set("database", db_config)
            print_success(f"SQLite configured: {path}")
            
        else:
            # Skip / Memory
            print_info("Using in-memory storage (no persistence)")
            db_config = {"type": "memory"}
            manager.set("database", db_config)
            
    except (ValueError, IndexError):
        print_warning("Invalid choice, using in-memory storage")


def _configure_database(manager, url: str, provider: str) -> dict:
    """Configure database and return config dict."""
    from huxley.cli.ui import print_success, print_info
    
    db_config = {
        "type": provider,
        "provider": provider,
        "url": url,
    }
    
    # Auto-detect from URL if provider not explicit
    url_lower = url.lower()
    if "supabase" in url_lower:
        db_config["type"] = "supabase"
        db_config["provider"] = "supabase"
        print_info("Detected Supabase database")
    elif "neon.tech" in url_lower:
        db_config["type"] = "neon"
        db_config["provider"] = "neon"
        print_info("Detected Neon database")
    elif "cockroach" in url_lower:
        db_config["type"] = "cockroachdb"
        db_config["provider"] = "cockroachdb"
        print_info("Detected CockroachDB")
    
    manager.set("database", db_config)
    print_success(f"Database configured: {provider}")
    
    return db_config


async def _setup_vector_extension(manager, db_config: dict):
    """Set up pgvector for AI memory."""
    from huxley.cli.ui import console, print_success, print_info, print_warning, print_error, confirm
    
    if db_config.get("type") not in ("supabase", "neon", "postgresql"):
        return
    
    console.print()
    console.print("  [bold]Initialize Database Tables[/bold]")
    console.print("  This will create Huxley's tables for:")
    console.print("    • Conversation history")
    console.print("    • Research sessions")
    console.print("    • Designed molecules")
    console.print("    • Tool execution logs")
    console.print()
    
    supports_vectors = db_config.get("type") in ("supabase", "neon")
    if supports_vectors:
        console.print("  [bold cyan]+ Vector Store (pgvector)[/bold cyan]")
        console.print("    AI memory with semantic search")
        console.print()
    
    if not confirm("Initialize database now?", default=True):
        print_info("Skipped. Run 'huxley config init-db' later to set up tables.")
        return
    
    # Initialize database
    try:
        from huxley.memory.migrations import test_connection, setup_database
        
        db_url = db_config.get("url")
        
        # Test connection
        console.print()
        console.print("  Testing connection...", end=" ")
        conn_result = await test_connection(db_url)
        
        if not conn_result["connected"]:
            console.print("[red]FAILED[/red]")
            print_error(f"Connection failed: {conn_result.get('error', 'Unknown error')}")
            print_info("Check your database URL and try again with 'huxley config init-db'")
            return
        
        console.print("[green]OK[/green]")
        console.print(f"  Provider detected: {conn_result['provider']}")
        console.print(f"  Vector support: {'Yes ✓' if conn_result['supports_vectors'] else 'No'}")
        
        # Set up tables
        console.print()
        console.print("  Creating tables...", end=" ")
        setup_result = await setup_database(db_url, include_vectors=conn_result["supports_vectors"])
        
        if not setup_result["success"]:
            console.print("[red]FAILED[/red]")
            print_error(f"Setup failed: {setup_result.get('error', 'Unknown error')}")
            return
        
        console.print("[green]OK[/green]")
        console.print()
        console.print("  [bold green]Tables created:[/bold green]")
        for table in setup_result["tables_created"]:
            console.print(f"    ✓ {table}")
        
        if setup_result["vectors_enabled"]:
            console.print()
            print_success("Vector memory enabled!")
            console.print("    AI memory with semantic search is now available.")
            
            # Update vector config
            vector_config = {
                "enabled": True,
                "provider": conn_result["provider"],
                "url": db_url,
                "table": "huxley_embeddings",
                "dimensions": 1536,
            }
            manager.set("vector", vector_config)
        
        console.print()
        print_success("Database initialized successfully!")
        
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_info("Install with: pip install asyncpg aiosqlite")
    except Exception as e:
        print_error(f"Error: {e}")
        print_info("Try again with 'huxley config init-db'")


async def setup_redis(manager):
    """Configure Redis for caching."""
    from huxley.cli.ui import console, print_success, print_warning, print_info, ask, confirm, S_MUTED
    from rich.text import Text
    
    console.print("  Redis provides fast caching for API responses and sessions.")
    console.print("  Optional but recommended for production use.")
    console.print()
    
    # Check existing config
    redis_config = manager.get("redis") or {}
    if redis_config.get("enabled"):
        console.print(f"  Current: {redis_config.get('provider', 'redis')} ", end="")
        console.print(Text("configured", style=S_MUTED))
        if not confirm("Reconfigure Redis?", default=False):
            return
        console.print()
    
    # Redis options
    console.print("  REDIS OPTIONS")
    console.print()
    console.print("    1. [bold red]Upstash[/bold red]        Serverless Redis (recommended)")
    console.print("    2. Redis Cloud   Managed Redis")
    console.print("    3. Local Redis   Self-hosted (localhost:6379)")
    console.print("    4. Custom URL    Other Redis provider")
    console.print("    5. Skip          No Redis caching")
    console.print()
    
    choice = ask("Choose Redis option (1-5)", default="5")
    
    try:
        idx = int(choice)
        
        if idx == 1:
            # Upstash
            console.print()
            console.print("  [bold red]Upstash Setup[/bold red]")
            console.print("  Get your credentials from:")
            console.print("  console.upstash.com → Your Database → REST API")
            console.print()
            
            url = ask("Redis URL (redis://default:xxx@xxx.upstash.io:6379)")
            if url:
                _configure_redis(manager, url, "upstash")
            else:
                print_warning("Skipped - no URL provided")
                
        elif idx == 2:
            # Redis Cloud
            console.print()
            url = ask("Redis Cloud URL")
            if url:
                _configure_redis(manager, url, "redis-cloud")
            else:
                print_warning("Skipped - no URL provided")
                
        elif idx == 3:
            # Local
            url = "redis://localhost:6379"
            _configure_redis(manager, url, "local")
            
        elif idx == 4:
            # Custom
            console.print()
            url = ask("Redis URL (redis://...)")
            if url:
                _configure_redis(manager, url, "custom")
            else:
                print_warning("Skipped - no URL provided")
                
        else:
            # Skip
            print_info("Redis caching disabled")
            redis_config = {"enabled": False}
            manager.set("redis", redis_config)
            
    except (ValueError, IndexError):
        print_warning("Invalid choice, Redis disabled")


def _configure_redis(manager, url: str, provider: str):
    """Configure Redis and save config."""
    from huxley.cli.ui import print_success, print_info
    
    redis_config = {
        "enabled": True,
        "provider": provider,
        "url": url,
    }
    
    # Auto-detect from URL
    url_lower = url.lower()
    if "upstash" in url_lower:
        redis_config["provider"] = "upstash"
        print_info("Detected Upstash Redis")
    
    manager.set("redis", redis_config)
    print_success(f"Redis configured: {provider}")
