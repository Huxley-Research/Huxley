"""
Memory store factory.

Creates the appropriate memory store based on configuration.
Supports PostgreSQL (Supabase, Neon), Redis, and SQLite.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from huxley.memory.store import MemoryStore


def get_memory_store() -> "MemoryStore":
    """
    Get the configured memory store.
    
    Reads from Huxley config and returns the appropriate store:
    - PostgreSQL/Supabase/Neon: PostgresStore
    - Redis: RedisStore
    - SQLite: SQLiteStore
    - Default: InMemoryStore
    
    Returns:
        Configured MemoryStore instance
    """
    from huxley.memory.store import InMemoryStore
    
    try:
        from huxley.cli.config import ConfigManager
        manager = ConfigManager()
        
        # Check for Redis first (used as cache layer)
        redis_config = manager.get("redis") or {}
        if redis_config.get("enabled") and redis_config.get("url"):
            try:
                from huxley.memory.store import RedisStore
                return RedisStore(url=redis_config["url"])
            except ImportError:
                pass  # redis not installed, fall through
        
        # Check for database
        db_config = manager.get("database") or {}
        db_type = db_config.get("type", "memory")
        db_url = db_config.get("url")
        
        if db_type == "memory" or not db_url:
            return InMemoryStore()
        
        if db_type in ("supabase", "neon", "postgresql"):
            try:
                from huxley.memory.postgres_store import PostgresStore
                return PostgresStore(url=db_url)
            except ImportError:
                # asyncpg not installed, fall back
                pass
        
        if db_type == "sqlite":
            try:
                from huxley.memory.sqlite_store import SQLiteStore
                db_path = db_url.replace("sqlite:///", "").replace("sqlite://", "")
                return SQLiteStore(path=db_path)
            except ImportError:
                pass
        
        # Fall back to in-memory
        return InMemoryStore()
        
    except Exception:
        # If anything fails, use in-memory
        return InMemoryStore()


def get_conversation_memory():
    """
    Get a ConversationMemory instance with the configured store.
    
    Returns:
        ConversationMemory with appropriate backend
    """
    from huxley.memory.conversation import ConversationMemory
    store = get_memory_store()
    return ConversationMemory(store=store)


async def get_database_connection():
    """
    Get a raw database connection for direct queries.
    
    Returns:
        Database connection or None if not configured
    """
    try:
        from huxley.cli.config import ConfigManager
        manager = ConfigManager()
        
        db_config = manager.get("database") or {}
        db_url = db_config.get("url")
        
        if not db_url:
            return None
        
        if db_url.startswith("postgresql") or db_url.startswith("postgres"):
            import asyncpg
            return await asyncpg.connect(db_url)
        
        if db_url.startswith("sqlite"):
            import aiosqlite
            db_path = db_url.replace("sqlite:///", "").replace("sqlite://", "")
            return await aiosqlite.connect(db_path)
        
        return None
        
    except Exception:
        return None


async def save_research_session(
    session_id: str,
    objective: str,
    status: str = "running",
    iterations: int = 0,
    findings: list = None,
    hypotheses: list = None,
    viable_solutions: list = None,
    metadata: dict = None,
) -> bool:
    """
    Save a research session to the database.
    
    Returns:
        True if saved successfully, False otherwise
    """
    import json
    
    try:
        conn = await get_database_connection()
        if conn is None:
            return False
        
        try:
            # Check if this is asyncpg or aiosqlite
            if hasattr(conn, 'execute'):
                # asyncpg (PostgreSQL)
                await conn.execute("""
                    INSERT INTO huxley_research_sessions 
                    (session_id, objective, status, iterations, findings, hypotheses, viable_solutions, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (session_id) DO UPDATE SET
                        status = $3,
                        iterations = $4,
                        findings = $5,
                        hypotheses = $6,
                        viable_solutions = $7,
                        metadata = $8
                """, 
                    session_id, 
                    objective, 
                    status, 
                    iterations,
                    json.dumps(findings or []),
                    json.dumps(hypotheses or []),
                    json.dumps(viable_solutions or []),
                    json.dumps(metadata or {}),
                )
            else:
                # aiosqlite (SQLite)
                await conn.execute("""
                    INSERT OR REPLACE INTO huxley_research_sessions 
                    (id, session_id, objective, status, iterations, findings, hypotheses, viable_solutions, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,  # Use session_id as id for SQLite
                    session_id,
                    objective,
                    status,
                    iterations,
                    json.dumps(findings or []),
                    json.dumps(hypotheses or []),
                    json.dumps(viable_solutions or []),
                    json.dumps(metadata or {}),
                ))
                await conn.commit()
            
            return True
            
        finally:
            await conn.close()
            
    except Exception as e:
        print(f"[Huxley] Failed to save research session: {e}")
        return False


async def save_molecule(
    molecule_id: str,
    smiles: str,
    name: str = None,
    target: str = None,
    properties: dict = None,
    druglikeness: dict = None,
    docking_results: list = None,
    session_id: str = None,
    metadata: dict = None,
) -> bool:
    """
    Save a designed molecule to the database.
    
    Returns:
        True if saved successfully, False otherwise
    """
    import json
    
    try:
        conn = await get_database_connection()
        if conn is None:
            return False
        
        try:
            if hasattr(conn, 'execute') and hasattr(conn, 'fetchrow'):
                # asyncpg (PostgreSQL)
                await conn.execute("""
                    INSERT INTO huxley_molecules 
                    (molecule_id, smiles, name, target, properties, druglikeness, docking_results, session_id, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (molecule_id) DO UPDATE SET
                        smiles = $2,
                        name = $3,
                        target = $4,
                        properties = $5,
                        druglikeness = $6,
                        docking_results = $7,
                        metadata = $9
                """,
                    molecule_id,
                    smiles,
                    name,
                    target,
                    json.dumps(properties or {}),
                    json.dumps(druglikeness or {}),
                    json.dumps(docking_results or []),
                    session_id,
                    json.dumps(metadata or {}),
                )
            else:
                # aiosqlite (SQLite)
                await conn.execute("""
                    INSERT OR REPLACE INTO huxley_molecules 
                    (id, molecule_id, smiles, name, target, properties, druglikeness, docking_results, session_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    molecule_id,
                    molecule_id,
                    smiles,
                    name,
                    target,
                    json.dumps(properties or {}),
                    json.dumps(druglikeness or {}),
                    json.dumps(docking_results or []),
                    session_id,
                    json.dumps(metadata or {}),
                ))
                await conn.commit()
            
            return True
            
        finally:
            await conn.close()
            
    except Exception as e:
        print(f"[Huxley] Failed to save molecule: {e}")
        return False


async def log_tool_execution(
    tool_name: str,
    parameters: dict = None,
    result: dict = None,
    success: bool = True,
    duration_ms: int = None,
    session_id: str = None,
) -> bool:
    """
    Log a tool execution to the database.
    
    Returns:
        True if logged successfully, False otherwise
    """
    import json
    
    try:
        conn = await get_database_connection()
        if conn is None:
            return False
        
        try:
            if hasattr(conn, 'execute') and hasattr(conn, 'fetchrow'):
                # asyncpg
                await conn.execute("""
                    INSERT INTO huxley_tool_executions 
                    (session_id, tool_name, parameters, result, success, duration_ms)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    session_id,
                    tool_name,
                    json.dumps(parameters or {}),
                    json.dumps(result or {}),
                    success,
                    duration_ms,
                )
            else:
                # aiosqlite
                import uuid
                await conn.execute("""
                    INSERT INTO huxley_tool_executions 
                    (id, session_id, tool_name, parameters, result, success, duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    session_id,
                    tool_name,
                    json.dumps(parameters or {}),
                    json.dumps(result or {}),
                    1 if success else 0,
                    duration_ms,
                ))
                await conn.commit()
            
            return True
            
        finally:
            await conn.close()
            
    except Exception:
        return False
