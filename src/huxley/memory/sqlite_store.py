"""
SQLite memory store implementation.

Lightweight local storage for development and single-user deployments.
"""

from __future__ import annotations

from typing import Any, Optional
from pathlib import Path
import json


from huxley.memory.store import MemoryStore


class SQLiteStore(MemoryStore):
    """Memory store backed by SQLite."""
    
    def __init__(self, path: str = "~/.huxley/data.db"):
        """
        Initialize SQLite store.
        
        Args:
            path: Path to SQLite database file
        """
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
    
    async def _get_conn(self):
        """Get or create database connection."""
        if self._conn is None:
            import aiosqlite
            self._conn = await aiosqlite.connect(str(self.path))
            self._conn.row_factory = aiosqlite.Row
            await self._ensure_tables()
        return self._conn
    
    async def _ensure_tables(self):
        """Ensure required tables exist."""
        conn = self._conn
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS huxley_memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                ttl_seconds INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.commit()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT value FROM huxley_memory WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except:
                    return row[0]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with optional TTL (in seconds)."""
        conn = await self._get_conn()
        value_json = json.dumps(value) if not isinstance(value, str) else value
        
        await conn.execute("""
            INSERT OR REPLACE INTO huxley_memory (key, value, ttl_seconds, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (key, value_json, ttl))
        await conn.commit()
    
    async def delete(self, key: str) -> None:
        """Delete a key."""
        conn = await self._get_conn()
        await conn.execute("DELETE FROM huxley_memory WHERE key = ?", (key,))
        await conn.commit()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT 1 FROM huxley_memory WHERE key = ?", (key,)
        ) as cursor:
            return await cursor.fetchone() is not None
    
    async def list_keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern."""
        conn = await self._get_conn()
        if pattern == "*":
            query = "SELECT key FROM huxley_memory"
            params = ()
        else:
            # Convert glob to SQL LIKE
            like_pattern = pattern.replace("*", "%").replace("?", "_")
            query = "SELECT key FROM huxley_memory WHERE key LIKE ?"
            params = (like_pattern,)
        
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows]
    
    async def clear(self) -> None:
        """Clear all keys."""
        conn = await self._get_conn()
        await conn.execute("DELETE FROM huxley_memory")
        await conn.commit()
    
    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
    
    # Conversation methods
    async def save_conversation(self, session_id: str, messages: list[dict]) -> None:
        """Save conversation messages."""
        import uuid
        conn = await self._get_conn()
        
        # Ensure conversation tables exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS huxley_conversations (
                id TEXT PRIMARY KEY,
                session_id TEXT UNIQUE NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS huxley_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create conversation
        conv_id = str(uuid.uuid4())
        await conn.execute("""
            INSERT OR REPLACE INTO huxley_conversations (id, session_id, metadata)
            VALUES (?, ?, ?)
        """, (conv_id, session_id, json.dumps({})))
        
        # Save messages
        for msg in messages:
            await conn.execute("""
                INSERT INTO huxley_messages (id, conversation_id, role, content)
                VALUES (?, ?, ?, ?)
            """, (str(uuid.uuid4()), conv_id, msg.get('role', 'user'), msg.get('content', '')))
        
        await conn.commit()
    
    async def load_conversation(self, session_id: str) -> list[dict]:
        """Load conversation messages."""
        conn = await self._get_conn()
        
        # Get conversation ID
        async with conn.execute(
            "SELECT id FROM huxley_conversations WHERE session_id = ?",
            (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return []
            conv_id = row[0]
        
        # Get messages
        async with conn.execute("""
            SELECT role, content FROM huxley_messages 
            WHERE conversation_id = ? 
            ORDER BY created_at
        """, (conv_id,)) as cursor:
            rows = await cursor.fetchall()
            return [{'role': row[0], 'content': row[1]} for row in rows]
