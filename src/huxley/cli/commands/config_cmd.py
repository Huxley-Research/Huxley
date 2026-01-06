"""
Huxley CLI - Config command.

Manage configuration settings.
"""

from huxley.cli.ui import console, print_success, print_error, print_info, rule
from huxley.cli.config import ConfigManager


def show_config():
    """Show current configuration with examples."""
    manager = ConfigManager()
    config = manager.get_all()
    
    console.print()
    console.print("[bold]HUXLEY CONFIGURATION[/bold]")
    rule()
    
    # Provider & Model
    console.print()
    console.print("[bold cyan]LLM PROVIDER[/bold cyan]")
    provider = config.get("default_provider", "[dim]not set[/dim]")
    model = config.get("default_model", "[dim]not set[/dim]")
    console.print(f"  Provider:        {provider}")
    console.print(f"  Model:           {model}")
    
    # API Keys
    console.print()
    console.print("[bold cyan]API KEYS[/bold cyan]")
    api_keys = config.get("api_keys", {})
    if api_keys:
        for provider_name, api_key in api_keys.items():
            if api_key:
                masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "****"
                console.print(f"  {provider_name:<16} {masked}")
    else:
        console.print("  [dim]No API keys configured[/dim]")
    
    # Base URLs
    console.print()
    console.print("[bold cyan]BASE URLS[/bold cyan]")
    base_urls = config.get("base_urls", {})
    if base_urls:
        for provider_name, url in base_urls.items():
            if url:
                console.print(f"  {provider_name:<16} {url}")
    else:
        console.print("  [dim]Using default URLs[/dim]")
    
    # Database
    console.print()
    console.print("[bold cyan]DATABASE[/bold cyan]")
    db_config = config.get("database", {})
    db_type = db_config.get("type", "memory")
    console.print(f"  Type:            {db_type}")
    if db_type != "memory":
        console.print(f"  URL:             {_mask_db_url(db_config.get('url', ''))}")
    
    # Redis
    console.print()
    console.print("[bold cyan]REDIS[/bold cyan]")
    redis_config = config.get("redis", {})
    if redis_config.get("enabled"):
        console.print(f"  Enabled:         Yes")
        console.print(f"  URL:             {_mask_redis_url(redis_config.get('url', ''))}")
    else:
        console.print("  [dim]Not configured[/dim]")
    
    # Vector Store
    console.print()
    console.print("[bold cyan]VECTOR STORE (AI MEMORY)[/bold cyan]")
    vector_config = config.get("vector", {})
    if vector_config.get("enabled"):
        console.print(f"  Provider:        {vector_config.get('provider', 'unknown')}")
        console.print(f"  URL:             {_mask_db_url(vector_config.get('url', ''))}")
    else:
        console.print("  [dim]Not configured[/dim]")
    
    # Show examples
    console.print()
    rule()
    console.print()
    console.print("[bold]CONFIGURATION COMMANDS[/bold]")
    console.print()
    console.print("  [cyan]Set LLM Provider:[/cyan]")
    console.print("    huxley config set default_provider openai")
    console.print("    huxley config set default_model gpt-4")
    console.print()
    console.print("  [cyan]Set API Key:[/cyan]")
    console.print("    huxley config set api_keys.openai sk-your-key-here")
    console.print("    huxley config set api_keys.anthropic sk-ant-your-key")
    console.print()
    console.print("  [cyan]Set Database (Supabase):[/cyan]")
    console.print("    huxley config set database.type supabase")
    console.print("    huxley config set database.url postgresql://user:pass@host:5432/db")
    console.print()
    console.print("  [cyan]Set Database (Neon):[/cyan]")
    console.print("    huxley config set database.type neon")
    console.print("    huxley config set database.url postgresql://user:pass@ep-xxx.neon.tech/db")
    console.print()
    console.print("  [cyan]Set Redis:[/cyan]")
    console.print("    huxley config set redis.url redis://localhost:6379")
    console.print()
    console.print("  [cyan]Set Vector Store:[/cyan]")
    console.print("    huxley config set vector.provider supabase")
    console.print("    huxley config set vector.url postgresql://...")
    console.print()
    console.print("  [cyan]Run Setup Wizard:[/cyan]")
    console.print("    huxley setup")
    console.print()


def _mask_db_url(url: str) -> str:
    """Mask sensitive parts of database URL."""
    if not url:
        return "[dim]not set[/dim]"
    
    import re
    # Mask password in postgresql://user:password@host
    masked = re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', url)
    return masked


def _mask_redis_url(url: str) -> str:
    """Mask sensitive parts of Redis URL."""
    if not url:
        return "[dim]not set[/dim]"
    
    import re
    # Mask password in redis://:password@host or redis://user:password@host
    masked = re.sub(r'://(:?)([^@]+)@', r'://\1***@', url)
    return masked


def set_config(key: str, value: str):
    """Set a configuration value."""
    manager = ConfigManager()
    
    # Handle special keys
    if key.startswith("api_keys."):
        provider = key.split(".", 1)[1]
        if provider in manager.PROVIDERS:
            manager.set_api_key(provider, value)
            print_success(f"Set {provider} API key")
        else:
            print_error(f"Unknown provider: {provider}")
            print_info(f"Available: {', '.join(manager.PROVIDERS.keys())}")
    elif key.startswith("database."):
        db_key = key.split(".", 1)[1]
        db_config = manager.get("database") or {}
        db_config[db_key] = value
        
        # Auto-detect provider from URL
        if db_key == "url":
            db_config = _auto_detect_database(value, db_config)
        
        manager.set("database", db_config)
        print_success(f"Set database.{db_key}")
        
        if db_config.get("type"):
            print_info(f"Database type: {db_config['type']}")
            
    elif key.startswith("redis."):
        redis_key = key.split(".", 1)[1]
        redis_config = manager.get("redis") or {}
        
        if redis_key == "url":
            redis_config["url"] = value
            redis_config["enabled"] = True
            # Auto-detect Upstash
            if "upstash" in value.lower():
                redis_config["provider"] = "upstash"
                print_info("Detected Upstash Redis")
        else:
            redis_config[redis_key] = value
            
        manager.set("redis", redis_config)
        print_success(f"Set redis.{redis_key}")
        
    elif key.startswith("vector."):
        vector_key = key.split(".", 1)[1]
        vector_config = manager.get("vector") or {}
        
        if vector_key == "url":
            vector_config["url"] = value
            vector_config["enabled"] = True
            # Auto-detect provider
            if "supabase" in value.lower():
                vector_config["provider"] = "supabase"
                print_info("Detected Supabase pgvector")
            elif "neon" in value.lower():
                vector_config["provider"] = "neon"
                print_info("Detected Neon pgvector")
            elif "pinecone" in value.lower():
                vector_config["provider"] = "pinecone"
                print_info("Detected Pinecone")
        else:
            vector_config[vector_key] = value
            
        manager.set("vector", vector_config)
        print_success(f"Set vector.{vector_key}")
    else:
        manager.set(key, value)
        print_success(f"Set {key} = {value}")


def _auto_detect_database(url: str, config: dict) -> dict:
    """Auto-detect database provider from URL."""
    url_lower = url.lower()
    
    if "supabase" in url_lower or ".supabase.co" in url_lower:
        config["type"] = "supabase"
        config["provider"] = "supabase"
        config["supports_vectors"] = True
    elif "neon.tech" in url_lower or "neon" in url_lower:
        config["type"] = "neon"
        config["provider"] = "neon"
        config["supports_vectors"] = True
    elif "cockroachlabs" in url_lower or "cockroachdb" in url_lower:
        config["type"] = "cockroachdb"
        config["provider"] = "cockroachdb"
        config["supports_vectors"] = False
    elif "planetscale" in url_lower:
        config["type"] = "planetscale"
        config["provider"] = "planetscale"
        config["supports_vectors"] = False
    elif "postgresql" in url_lower or "postgres" in url_lower:
        config["type"] = "postgresql"
        config["provider"] = "postgresql"
        config["supports_vectors"] = True  # May support pgvector
    elif "sqlite" in url_lower:
        config["type"] = "sqlite"
        config["provider"] = "sqlite"
        config["supports_vectors"] = False
    
    return config


def init_database():
    """Initialize database tables automatically."""
    manager = ConfigManager()
    db_config = manager.get("database") or {}
    
    db_url = db_config.get("url")
    if not db_url:
        print_error("No database URL configured")
        print_info("Set one with: huxley config set database.url <url>")
        return
    
    provider = db_config.get("provider", "unknown")
    console.print()
    console.print(f"[bold]Initializing {provider.upper()} Database[/bold]")
    rule()
    
    try:
        from huxley.memory.migrations import test_connection_sync, setup_database_sync
        
        # Test connection
        console.print("  Testing connection...", end=" ")
        conn_result = test_connection_sync(db_url)
        
        if not conn_result["connected"]:
            console.print("[red]FAILED[/red]")
            print_error(f"Connection failed: {conn_result.get('error', 'Unknown error')}")
            return
        
        console.print("[green]OK[/green]")
        console.print(f"  Provider: {conn_result['provider']}")
        console.print(f"  Vectors:  {'Supported' if conn_result['supports_vectors'] else 'Not supported'}")
        
        # Set up tables
        console.print()
        console.print("  Creating tables...", end=" ")
        setup_result = setup_database_sync(db_url, include_vectors=conn_result["supports_vectors"])
        
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
            console.print("  [bold cyan]Vector memory enabled![/bold cyan]")
            console.print("    AI memory with semantic search is now available.")
            
            # Update config
            vector_config = manager.get("vector") or {}
            vector_config["enabled"] = True
            vector_config["provider"] = conn_result["provider"]
            vector_config["url"] = db_url
            manager.set("vector", vector_config)
        
        console.print()
        print_success("Database initialized successfully!")
        
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_info("Install with: pip install asyncpg aiosqlite")
    except Exception as e:
        print_error(f"Error: {e}")


def test_database():
    """Test database connection."""
    manager = ConfigManager()
    db_config = manager.get("database") or {}
    
    db_url = db_config.get("url")
    if not db_url:
        print_error("No database URL configured")
        return
    
    try:
        from huxley.memory.migrations import test_connection_sync
        
        console.print()
        console.print("[bold]Testing Database Connection[/bold]")
        rule()
        
        result = test_connection_sync(db_url)
        
        if result["connected"]:
            print_success(f"Connected to {result['provider']}")
            console.print(f"  Vector support: {'Yes' if result['supports_vectors'] else 'No'}")
        else:
            print_error(f"Connection failed: {result.get('error', 'Unknown error')}")
            
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
    except Exception as e:
        print_error(f"Error: {e}")


def get_config(key: str):
    """Get a specific configuration value."""
    manager = ConfigManager()
    
    # Handle nested keys
    if "." in key:
        parts = key.split(".")
        value = manager.get(parts[0])
        if isinstance(value, dict):
            for part in parts[1:]:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
        else:
            value = None
    else:
        value = manager.get(key)
    
    if value is not None:
        # Mask sensitive values
        if "key" in key.lower() or "password" in key.lower():
            if isinstance(value, str) and len(value) > 8:
                value = value[:8] + "..." + value[-4:]
        console.print(f"{key}: {value}")
    else:
        console.print(f"{key}: [dim]not set[/dim]")


def delete_config(key: str):
    """Delete a configuration value."""
    manager = ConfigManager()
    manager.delete(key)
    print_success(f"Deleted {key}")


def check_schema():
    """Check if database schema is up to date."""
    manager = ConfigManager()
    db_config = manager.get("database") or {}
    
    db_url = db_config.get("url")
    if not db_url:
        print_error("No database URL configured")
        print_info("Set one with: huxley config set database.url <url>")
        return
    
    console.print()
    console.print("[bold]Database Schema Check[/bold]")
    rule()
    
    # Define expected tables and their required columns
    EXPECTED_TABLES = {
        "huxley_conversations": ["id", "session_id", "metadata", "created_at"],
        "huxley_messages": ["id", "conversation_id", "role", "content", "created_at"],
        "huxley_memory": ["key", "value", "metadata"],
        "huxley_research_sessions": ["id", "session_id", "objective", "status", "iterations"],
        "huxley_tool_executions": ["id", "session_id", "tool_name", "input_params"],
        "huxley_molecules": ["id", "molecule_id", "smiles", "properties"],
        "huxley_ai_memory": ["key", "value", "metadata"],
        "huxley_exploration_sessions": ["id", "session_id", "domain", "objective", "curiosity_policy"],
        "huxley_hypothesis_ledger": ["id", "hypothesis_id", "session_id", "statement", "confidence"],
        "huxley_skill_registry": ["id", "skill_name", "task_pattern", "success_rate"],
        "huxley_risk_annotations": ["id", "entity_type", "entity_id"],
    }
    
    VECTOR_TABLES = {
        "huxley_embeddings": ["id", "doc_id", "content", "embedding"],
        "huxley_documents": ["id", "content", "metadata"],
    }
    
    import asyncio
    
    async def check_tables():
        try:
            from huxley.memory.factory import get_database_connection
            
            conn = await get_database_connection()
            if conn is None:
                return {"error": "Could not connect to database"}
            
            results = {
                "connected": True,
                "tables_found": [],
                "tables_missing": [],
                "columns_ok": [],
                "columns_missing": [],
                "vectors_available": False,
            }
            
            try:
                if hasattr(conn, 'fetch'):
                    # PostgreSQL
                    # Get all tables
                    rows = await conn.fetch("""
                        SELECT table_name FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name LIKE 'huxley_%'
                    """)
                    existing_tables = {row['table_name'] for row in rows}
                    
                    # Check each expected table
                    for table, columns in EXPECTED_TABLES.items():
                        if table in existing_tables:
                            results["tables_found"].append(table)
                            
                            # Check columns
                            col_rows = await conn.fetch("""
                                SELECT column_name FROM information_schema.columns 
                                WHERE table_name = $1
                            """, table)
                            existing_cols = {row['column_name'] for row in col_rows}
                            
                            for col in columns:
                                if col in existing_cols:
                                    results["columns_ok"].append(f"{table}.{col}")
                                else:
                                    results["columns_missing"].append(f"{table}.{col}")
                        else:
                            results["tables_missing"].append(table)
                    
                    # Check vector tables
                    for table, columns in VECTOR_TABLES.items():
                        if table in existing_tables:
                            results["vectors_available"] = True
                            results["tables_found"].append(table)
                    
                    # Check pgvector extension
                    try:
                        ext_rows = await conn.fetch("""
                            SELECT extname FROM pg_extension WHERE extname = 'vector'
                        """)
                        if ext_rows:
                            results["pgvector_enabled"] = True
                    except:
                        results["pgvector_enabled"] = False
                        
                else:
                    # SQLite
                    cursor = await conn.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name LIKE 'huxley_%'
                    """)
                    rows = await cursor.fetchall()
                    existing_tables = {row[0] for row in rows}
                    
                    for table, columns in EXPECTED_TABLES.items():
                        if table in existing_tables:
                            results["tables_found"].append(table)
                            
                            cursor = await conn.execute(f"PRAGMA table_info({table})")
                            col_rows = await cursor.fetchall()
                            existing_cols = {row[1] for row in col_rows}
                            
                            for col in columns:
                                if col in existing_cols:
                                    results["columns_ok"].append(f"{table}.{col}")
                                else:
                                    results["columns_missing"].append(f"{table}.{col}")
                        else:
                            results["tables_missing"].append(table)
                
                return results
                
            finally:
                await conn.close()
                
        except Exception as e:
            return {"error": str(e)}
    
    result = asyncio.run(check_tables())
    
    if "error" in result:
        print_error(f"Check failed: {result['error']}")
        return
    
    # Display results
    console.print()
    
    # Tables found
    console.print(f"[bold green]Tables Found ({len(result['tables_found'])})[/bold green]")
    for table in sorted(result["tables_found"]):
        console.print(f"  ✓ {table}")
    
    # Tables missing
    if result["tables_missing"]:
        console.print()
        console.print(f"[bold yellow]Tables Missing ({len(result['tables_missing'])})[/bold yellow]")
        for table in sorted(result["tables_missing"]):
            console.print(f"  ✗ {table}")
    
    # Columns missing
    if result["columns_missing"]:
        console.print()
        console.print(f"[bold red]Columns Missing ({len(result['columns_missing'])})[/bold red]")
        for col in sorted(result["columns_missing"]):
            console.print(f"  ✗ {col}")
    
    # Vector status
    console.print()
    if result.get("pgvector_enabled"):
        console.print("[bold cyan]pgvector:[/bold cyan] ✓ Enabled")
    elif result.get("vectors_available"):
        console.print("[bold cyan]Vector tables:[/bold cyan] ✓ Found")
    else:
        console.print("[dim]Vector tables: Not found[/dim]")
    
    # Summary
    console.print()
    rule()
    
    total_expected = len(EXPECTED_TABLES)
    total_found = len([t for t in result["tables_found"] if t in EXPECTED_TABLES])
    
    if result["tables_missing"] or result["columns_missing"]:
        console.print()
        console.print(f"[yellow]Schema is incomplete ({total_found}/{total_expected} tables)[/yellow]")
        console.print()
        console.print("[cyan]Auto-updating schema...[/cyan]")
        
        # Auto-update the schema
        update_result = asyncio.run(auto_update_schema(
            result["tables_missing"], 
            result["columns_missing"]
        ))
        
        if update_result.get("success"):
            console.print()
            if update_result.get("tables_created"):
                console.print(f"[green]✓ Created {len(update_result['tables_created'])} tables[/green]")
                for t in update_result["tables_created"]:
                    console.print(f"    + {t}")
            if update_result.get("columns_added"):
                console.print(f"[green]✓ Added {len(update_result['columns_added'])} columns[/green]")
                for c in update_result["columns_added"]:
                    console.print(f"    + {c}")
            console.print()
            print_success("Schema updated successfully!")
        else:
            print_error(f"Auto-update failed: {update_result.get('error', 'Unknown error')}")
            console.print("Run [cyan]huxley config init-db[/cyan] manually")
    else:
        console.print()
        print_success(f"Schema is up to date ({total_found}/{total_expected} tables)")


async def auto_update_schema(missing_tables: list, missing_columns: list):
    """
    Automatically create missing tables and add missing columns.
    """
    from huxley.memory.factory import get_database_connection
    
    # Column type definitions for each table/column
    COLUMN_TYPES_POSTGRES = {
        # huxley_conversations
        "huxley_conversations.id": "UUID PRIMARY KEY DEFAULT uuid_generate_v4()",
        "huxley_conversations.session_id": "VARCHAR(64) NOT NULL",
        "huxley_conversations.metadata": "JSONB DEFAULT '{}'::jsonb",
        "huxley_conversations.created_at": "TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
        "huxley_conversations.updated_at": "TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
        
        # huxley_messages
        "huxley_messages.id": "UUID PRIMARY KEY DEFAULT uuid_generate_v4()",
        "huxley_messages.conversation_id": "UUID",
        "huxley_messages.role": "VARCHAR(32)",
        "huxley_messages.content": "TEXT",
        "huxley_messages.tool_calls": "JSONB",
        "huxley_messages.tool_results": "JSONB",
        "huxley_messages.created_at": "TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
        "huxley_messages.metadata": "JSONB DEFAULT '{}'::jsonb",
        
        # huxley_memory
        "huxley_memory.id": "UUID PRIMARY KEY DEFAULT uuid_generate_v4()",
        "huxley_memory.key": "VARCHAR(512) NOT NULL",
        "huxley_memory.value": "JSONB NOT NULL DEFAULT '{}'::jsonb",
        "huxley_memory.metadata": "JSONB DEFAULT '{}'::jsonb",
        "huxley_memory.expires_at": "TIMESTAMP WITH TIME ZONE",
        "huxley_memory.created_at": "TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
        
        # huxley_research_sessions
        "huxley_research_sessions.id": "UUID PRIMARY KEY DEFAULT uuid_generate_v4()",
        "huxley_research_sessions.session_id": "VARCHAR(64) NOT NULL",
        "huxley_research_sessions.objective": "TEXT",
        "huxley_research_sessions.status": "VARCHAR(32) DEFAULT 'running'",
        "huxley_research_sessions.iterations": "INTEGER DEFAULT 0",
        "huxley_research_sessions.findings": "JSONB DEFAULT '[]'::jsonb",
        "huxley_research_sessions.hypotheses": "JSONB DEFAULT '[]'::jsonb",
        "huxley_research_sessions.created_at": "TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
        "huxley_research_sessions.metadata": "JSONB DEFAULT '{}'::jsonb",
        
        # huxley_tool_executions
        "huxley_tool_executions.id": "UUID PRIMARY KEY DEFAULT uuid_generate_v4()",
        "huxley_tool_executions.session_id": "VARCHAR(64)",
        "huxley_tool_executions.tool_name": "VARCHAR(128)",
        "huxley_tool_executions.input_params": "JSONB",
        "huxley_tool_executions.parameters": "JSONB",
        "huxley_tool_executions.result": "JSONB",
        "huxley_tool_executions.success": "BOOLEAN DEFAULT true",
        "huxley_tool_executions.duration_ms": "INTEGER",
        "huxley_tool_executions.created_at": "TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
        
        # huxley_molecules
        "huxley_molecules.id": "UUID PRIMARY KEY DEFAULT uuid_generate_v4()",
        "huxley_molecules.molecule_id": "VARCHAR(64)",
        "huxley_molecules.smiles": "TEXT",
        "huxley_molecules.name": "VARCHAR(256)",
        "huxley_molecules.target": "VARCHAR(64)",
        "huxley_molecules.properties": "JSONB DEFAULT '{}'::jsonb",
        "huxley_molecules.druglikeness": "JSONB DEFAULT '{}'::jsonb",
        "huxley_molecules.docking_results": "JSONB DEFAULT '[]'::jsonb",
        "huxley_molecules.created_at": "TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
        "huxley_molecules.session_id": "VARCHAR(64)",
        "huxley_molecules.metadata": "JSONB DEFAULT '{}'::jsonb",
        
        # huxley_ai_memory
        "huxley_ai_memory.id": "UUID PRIMARY KEY DEFAULT uuid_generate_v4()",
        "huxley_ai_memory.key": "VARCHAR(512) NOT NULL",
        "huxley_ai_memory.value": "JSONB NOT NULL DEFAULT '{}'::jsonb",
        "huxley_ai_memory.metadata": "JSONB DEFAULT '{}'::jsonb",
        "huxley_ai_memory.created_at": "TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
        
        # huxley_exploration_sessions
        "huxley_exploration_sessions.id": "UUID PRIMARY KEY DEFAULT gen_random_uuid()",
        "huxley_exploration_sessions.session_id": "TEXT NOT NULL",
        "huxley_exploration_sessions.domain": "TEXT NOT NULL",
        "huxley_exploration_sessions.objective": "TEXT NOT NULL",
        "huxley_exploration_sessions.curiosity_policy": "TEXT NOT NULL",
        "huxley_exploration_sessions.start_time": "TIMESTAMP",
        "huxley_exploration_sessions.end_time": "TIMESTAMP",
        "huxley_exploration_sessions.iterations": "INTEGER DEFAULT 0",
        "huxley_exploration_sessions.confidence_delta": "JSONB DEFAULT '{}'",
        "huxley_exploration_sessions.metadata": "JSONB DEFAULT '{}'",
        "huxley_exploration_sessions.created_at": "TIMESTAMP DEFAULT NOW()",
        
        # huxley_hypothesis_ledger
        "huxley_hypothesis_ledger.id": "UUID PRIMARY KEY DEFAULT gen_random_uuid()",
        "huxley_hypothesis_ledger.hypothesis_id": "TEXT NOT NULL",
        "huxley_hypothesis_ledger.session_id": "TEXT NOT NULL",
        "huxley_hypothesis_ledger.statement": "TEXT NOT NULL",
        "huxley_hypothesis_ledger.confidence": "REAL NOT NULL",
        "huxley_hypothesis_ledger.evidence_links": "JSONB DEFAULT '[]'",
        "huxley_hypothesis_ledger.speculative_flag": "BOOLEAN DEFAULT TRUE",
        "huxley_hypothesis_ledger.revision_history": "JSONB DEFAULT '[]'",
        "huxley_hypothesis_ledger.metadata": "JSONB DEFAULT '{}'",
        "huxley_hypothesis_ledger.created_at": "TIMESTAMP DEFAULT NOW()",
        
        # huxley_skill_registry
        "huxley_skill_registry.id": "UUID PRIMARY KEY DEFAULT gen_random_uuid()",
        "huxley_skill_registry.skill_name": "TEXT NOT NULL",
        "huxley_skill_registry.task_pattern": "TEXT NOT NULL",
        "huxley_skill_registry.success_rate": "REAL DEFAULT 0.0",
        "huxley_skill_registry.applicability_domain": "TEXT[]",
        "huxley_skill_registry.usage_count": "INTEGER DEFAULT 0",
        "huxley_skill_registry.metadata": "JSONB DEFAULT '{}'",
        "huxley_skill_registry.created_at": "TIMESTAMP DEFAULT NOW()",
        
        # huxley_risk_annotations
        "huxley_risk_annotations.id": "UUID PRIMARY KEY DEFAULT gen_random_uuid()",
        "huxley_risk_annotations.entity_type": "TEXT NOT NULL",
        "huxley_risk_annotations.entity_id": "TEXT NOT NULL",
        "huxley_risk_annotations.safety_relevance": "TEXT",
        "huxley_risk_annotations.uncertainty_level": "REAL",
        "huxley_risk_annotations.ethical_sensitivity": "TEXT",
        "huxley_risk_annotations.metadata": "JSONB DEFAULT '{}'",
        "huxley_risk_annotations.created_at": "TIMESTAMP DEFAULT NOW()",
    }
    
    COLUMN_TYPES_SQLITE = {
        # huxley_conversations
        "huxley_conversations.id": "TEXT PRIMARY KEY",
        "huxley_conversations.session_id": "TEXT NOT NULL",
        "huxley_conversations.metadata": "TEXT DEFAULT '{}'",
        "huxley_conversations.created_at": "TEXT DEFAULT (datetime('now'))",
        
        # huxley_messages
        "huxley_messages.id": "TEXT PRIMARY KEY",
        "huxley_messages.conversation_id": "TEXT",
        "huxley_messages.role": "TEXT",
        "huxley_messages.content": "TEXT",
        "huxley_messages.created_at": "TEXT DEFAULT (datetime('now'))",
        
        # huxley_memory
        "huxley_memory.key": "TEXT NOT NULL",
        "huxley_memory.value": "TEXT DEFAULT '{}'",
        "huxley_memory.metadata": "TEXT DEFAULT '{}'",
        
        # huxley_research_sessions
        "huxley_research_sessions.id": "TEXT PRIMARY KEY",
        "huxley_research_sessions.session_id": "TEXT NOT NULL",
        "huxley_research_sessions.objective": "TEXT",
        "huxley_research_sessions.status": "TEXT DEFAULT 'running'",
        "huxley_research_sessions.iterations": "INTEGER DEFAULT 0",
        
        # huxley_tool_executions
        "huxley_tool_executions.id": "TEXT PRIMARY KEY",
        "huxley_tool_executions.session_id": "TEXT",
        "huxley_tool_executions.tool_name": "TEXT",
        "huxley_tool_executions.input_params": "TEXT",
        
        # huxley_molecules
        "huxley_molecules.id": "TEXT PRIMARY KEY",
        "huxley_molecules.molecule_id": "TEXT",
        "huxley_molecules.smiles": "TEXT",
        "huxley_molecules.properties": "TEXT DEFAULT '{}'",
        
        # huxley_ai_memory
        "huxley_ai_memory.key": "TEXT NOT NULL",
        "huxley_ai_memory.value": "TEXT DEFAULT '{}'",
        "huxley_ai_memory.metadata": "TEXT DEFAULT '{}'",
        
        # huxley_exploration_sessions
        "huxley_exploration_sessions.id": "TEXT PRIMARY KEY",
        "huxley_exploration_sessions.session_id": "TEXT NOT NULL",
        "huxley_exploration_sessions.domain": "TEXT NOT NULL",
        "huxley_exploration_sessions.objective": "TEXT NOT NULL",
        "huxley_exploration_sessions.curiosity_policy": "TEXT NOT NULL",
        
        # huxley_hypothesis_ledger
        "huxley_hypothesis_ledger.id": "TEXT PRIMARY KEY",
        "huxley_hypothesis_ledger.hypothesis_id": "TEXT NOT NULL",
        "huxley_hypothesis_ledger.session_id": "TEXT NOT NULL",
        "huxley_hypothesis_ledger.statement": "TEXT NOT NULL",
        "huxley_hypothesis_ledger.confidence": "REAL NOT NULL",
        
        # huxley_skill_registry
        "huxley_skill_registry.id": "TEXT PRIMARY KEY",
        "huxley_skill_registry.skill_name": "TEXT NOT NULL",
        "huxley_skill_registry.task_pattern": "TEXT NOT NULL",
        "huxley_skill_registry.success_rate": "REAL DEFAULT 0.0",
        
        # huxley_risk_annotations
        "huxley_risk_annotations.id": "TEXT PRIMARY KEY",
        "huxley_risk_annotations.entity_type": "TEXT NOT NULL",
        "huxley_risk_annotations.entity_id": "TEXT NOT NULL",
    }
    
    # Table creation SQL for missing tables
    TABLE_CREATE_POSTGRES = {
        "huxley_conversations": """
            CREATE TABLE IF NOT EXISTS huxley_conversations (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                session_id VARCHAR(64) NOT NULL UNIQUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'::jsonb
            )
        """,
        "huxley_messages": """
            CREATE TABLE IF NOT EXISTS huxley_messages (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                conversation_id UUID,
                role VARCHAR(32) NOT NULL,
                content TEXT NOT NULL,
                tool_calls JSONB,
                tool_results JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'::jsonb
            )
        """,
        "huxley_memory": """
            CREATE TABLE IF NOT EXISTS huxley_memory (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                key VARCHAR(512) NOT NULL UNIQUE,
                value JSONB NOT NULL,
                expires_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'::jsonb
            )
        """,
        "huxley_research_sessions": """
            CREATE TABLE IF NOT EXISTS huxley_research_sessions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                session_id VARCHAR(64) NOT NULL UNIQUE,
                objective TEXT NOT NULL,
                status VARCHAR(32) DEFAULT 'running',
                iterations INTEGER DEFAULT 0,
                findings JSONB DEFAULT '[]'::jsonb,
                hypotheses JSONB DEFAULT '[]'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'::jsonb
            )
        """,
        "huxley_tool_executions": """
            CREATE TABLE IF NOT EXISTS huxley_tool_executions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                session_id VARCHAR(64),
                tool_name VARCHAR(128) NOT NULL,
                input_params JSONB,
                parameters JSONB,
                result JSONB,
                success BOOLEAN DEFAULT true,
                duration_ms INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """,
        "huxley_molecules": """
            CREATE TABLE IF NOT EXISTS huxley_molecules (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                molecule_id VARCHAR(64) NOT NULL UNIQUE,
                smiles TEXT NOT NULL,
                name VARCHAR(256),
                target VARCHAR(64),
                properties JSONB DEFAULT '{}'::jsonb,
                druglikeness JSONB DEFAULT '{}'::jsonb,
                docking_results JSONB DEFAULT '[]'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                session_id VARCHAR(64),
                metadata JSONB DEFAULT '{}'::jsonb
            )
        """,
        "huxley_ai_memory": """
            CREATE TABLE IF NOT EXISTS huxley_ai_memory (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                key VARCHAR(512) NOT NULL UNIQUE,
                value JSONB NOT NULL DEFAULT '{}'::jsonb,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """,
        "huxley_exploration_sessions": """
            CREATE TABLE IF NOT EXISTS huxley_exploration_sessions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id TEXT UNIQUE NOT NULL,
                domain TEXT NOT NULL,
                objective TEXT NOT NULL,
                curiosity_policy TEXT NOT NULL,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                iterations INTEGER DEFAULT 0,
                confidence_delta JSONB DEFAULT '{}',
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """,
        "huxley_hypothesis_ledger": """
            CREATE TABLE IF NOT EXISTS huxley_hypothesis_ledger (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                hypothesis_id TEXT UNIQUE NOT NULL,
                session_id TEXT NOT NULL,
                statement TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_links JSONB DEFAULT '[]',
                speculative_flag BOOLEAN DEFAULT TRUE,
                revision_history JSONB DEFAULT '[]',
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """,
        "huxley_skill_registry": """
            CREATE TABLE IF NOT EXISTS huxley_skill_registry (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                skill_name TEXT UNIQUE NOT NULL,
                task_pattern TEXT NOT NULL,
                success_rate REAL DEFAULT 0.0,
                applicability_domain TEXT[],
                usage_count INTEGER DEFAULT 0,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """,
        "huxley_risk_annotations": """
            CREATE TABLE IF NOT EXISTS huxley_risk_annotations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                safety_relevance TEXT,
                uncertainty_level REAL,
                ethical_sensitivity TEXT,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """,
    }
    
    TABLE_CREATE_SQLITE = {
        "huxley_conversations": """
            CREATE TABLE IF NOT EXISTS huxley_conversations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL UNIQUE,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                metadata TEXT DEFAULT '{}'
            )
        """,
        "huxley_messages": """
            CREATE TABLE IF NOT EXISTS huxley_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """,
        "huxley_memory": """
            CREATE TABLE IF NOT EXISTS huxley_memory (
                id TEXT PRIMARY KEY,
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """,
        "huxley_research_sessions": """
            CREATE TABLE IF NOT EXISTS huxley_research_sessions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL UNIQUE,
                objective TEXT NOT NULL,
                status TEXT DEFAULT 'running',
                iterations INTEGER DEFAULT 0
            )
        """,
        "huxley_tool_executions": """
            CREATE TABLE IF NOT EXISTS huxley_tool_executions (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                tool_name TEXT NOT NULL,
                input_params TEXT
            )
        """,
        "huxley_molecules": """
            CREATE TABLE IF NOT EXISTS huxley_molecules (
                id TEXT PRIMARY KEY,
                molecule_id TEXT NOT NULL UNIQUE,
                smiles TEXT NOT NULL,
                properties TEXT DEFAULT '{}'
            )
        """,
        "huxley_ai_memory": """
            CREATE TABLE IF NOT EXISTS huxley_ai_memory (
                id TEXT PRIMARY KEY,
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL DEFAULT '{}',
                metadata TEXT DEFAULT '{}'
            )
        """,
        "huxley_exploration_sessions": """
            CREATE TABLE IF NOT EXISTS huxley_exploration_sessions (
                id TEXT PRIMARY KEY,
                session_id TEXT UNIQUE NOT NULL,
                domain TEXT NOT NULL,
                objective TEXT NOT NULL,
                curiosity_policy TEXT NOT NULL,
                start_time TEXT,
                end_time TEXT,
                iterations INTEGER DEFAULT 0,
                confidence_delta TEXT DEFAULT '{}',
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "huxley_hypothesis_ledger": """
            CREATE TABLE IF NOT EXISTS huxley_hypothesis_ledger (
                id TEXT PRIMARY KEY,
                hypothesis_id TEXT UNIQUE NOT NULL,
                session_id TEXT NOT NULL,
                statement TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_links TEXT DEFAULT '[]',
                speculative_flag INTEGER DEFAULT 1,
                revision_history TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "huxley_skill_registry": """
            CREATE TABLE IF NOT EXISTS huxley_skill_registry (
                id TEXT PRIMARY KEY,
                skill_name TEXT UNIQUE NOT NULL,
                task_pattern TEXT NOT NULL,
                success_rate REAL DEFAULT 0.0,
                applicability_domain TEXT,
                usage_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "huxley_risk_annotations": """
            CREATE TABLE IF NOT EXISTS huxley_risk_annotations (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                safety_relevance TEXT,
                uncertainty_level REAL,
                ethical_sensitivity TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
    }
    
    result = {
        "success": False,
        "tables_created": [],
        "columns_added": [],
        "errors": []
    }
    
    try:
        conn = await get_database_connection()
        if conn is None:
            return {"success": False, "error": "Could not connect to database"}
        
        is_postgres = hasattr(conn, 'fetch')
        
        try:
            # Create missing tables
            for table in missing_tables:
                try:
                    if is_postgres:
                        if table in TABLE_CREATE_POSTGRES:
                            await conn.execute(TABLE_CREATE_POSTGRES[table])
                            result["tables_created"].append(table)
                    else:
                        if table in TABLE_CREATE_SQLITE:
                            await conn.execute(TABLE_CREATE_SQLITE[table])
                            await conn.commit()
                            result["tables_created"].append(table)
                except Exception as e:
                    result["errors"].append(f"Failed to create {table}: {str(e)}")
            
            # Add missing columns
            for col_spec in missing_columns:
                table, column = col_spec.split(".", 1)
                try:
                    if is_postgres:
                        col_type = COLUMN_TYPES_POSTGRES.get(col_spec, "TEXT")
                        # Remove PRIMARY KEY and constraints for ALTER TABLE
                        col_type_clean = col_type.replace("PRIMARY KEY", "").replace("NOT NULL", "").strip()
                        if "DEFAULT" in col_type_clean:
                            # Keep DEFAULT clause
                            pass
                        else:
                            col_type_clean = col_type_clean.split()[0] if col_type_clean else "TEXT"
                        
                        await conn.execute(f'ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "{column}" {col_type_clean}')
                        result["columns_added"].append(col_spec)
                    else:
                        col_type = COLUMN_TYPES_SQLITE.get(col_spec, "TEXT")
                        col_type_clean = col_type.replace("PRIMARY KEY", "").replace("NOT NULL", "").strip()
                        if not col_type_clean:
                            col_type_clean = "TEXT"
                        
                        # SQLite doesn't support IF NOT EXISTS for columns, need to check first
                        cursor = await conn.execute(f"PRAGMA table_info({table})")
                        cols = await cursor.fetchall()
                        existing = {c[1] for c in cols}
                        
                        if column not in existing:
                            await conn.execute(f'ALTER TABLE {table} ADD COLUMN "{column}" {col_type_clean}')
                            await conn.commit()
                            result["columns_added"].append(col_spec)
                        else:
                            result["columns_added"].append(f"{col_spec} (already exists)")
                except Exception as e:
                    result["errors"].append(f"Failed to add {col_spec}: {str(e)}")
            
            result["success"] = len(result["errors"]) == 0
            
        finally:
            await conn.close()
            
    except Exception as e:
        result["error"] = str(e)
    
    return result


