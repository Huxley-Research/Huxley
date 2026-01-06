"""
PostgreSQL memory store implementation.

Supports Supabase, Neon, and standard PostgreSQL.
"""

from __future__ import annotations

from typing import Any, Optional
import json

from huxley.memory.store import MemoryStore


class PostgresStore(MemoryStore):
    """Memory store backed by PostgreSQL."""
    
    def __init__(self, url: str):
        """
        Initialize PostgreSQL store.
        
        Args:
            url: PostgreSQL connection URL
        """
        self.url = url
        self._pool = None
    
    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(self.url, min_size=1, max_size=5)
        return self._pool
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM huxley_memory WHERE key = $1",
                key
            )
            if row:
                try:
                    return json.loads(row['value'])
                except:
                    return row['value']
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with optional TTL (in seconds)."""
        pool = await self._get_pool()
        value_json = json.dumps(value) if not isinstance(value, str) else value
        
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO huxley_memory (key, value, ttl_seconds)
                VALUES ($1, $2, $3)
                ON CONFLICT (key) DO UPDATE SET 
                    value = $2, 
                    ttl_seconds = $3,
                    updated_at = NOW()
            """, key, value_json, ttl)
    
    async def delete(self, key: str) -> None:
        """Delete a key."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM huxley_memory WHERE key = $1", key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM huxley_memory WHERE key = $1",
                key
            )
            return row is not None
    
    async def list_keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if pattern == "*":
                rows = await conn.fetch("SELECT key FROM huxley_memory")
            else:
                # Convert glob to SQL LIKE
                like_pattern = pattern.replace("*", "%").replace("?", "_")
                rows = await conn.fetch(
                    "SELECT key FROM huxley_memory WHERE key LIKE $1",
                    like_pattern
                )
            return [row['key'] for row in rows]
    
    async def clear(self) -> None:
        """Clear all keys."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM huxley_memory")
    
    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    # Conversation methods
    async def save_conversation(self, session_id: str, messages: list[dict]) -> None:
        """Save conversation messages."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Create/update conversation
            await conn.execute("""
                INSERT INTO huxley_conversations (session_id, metadata)
                VALUES ($1, $2)
                ON CONFLICT (session_id) DO UPDATE SET updated_at = NOW()
            """, session_id, json.dumps({}))
            
            # Get conversation ID
            row = await conn.fetchrow(
                "SELECT id FROM huxley_conversations WHERE session_id = $1",
                session_id
            )
            if row:
                conv_id = row['id']
                for msg in messages:
                    await conn.execute("""
                        INSERT INTO huxley_messages (conversation_id, role, content)
                        VALUES ($1, $2, $3)
                    """, conv_id, msg.get('role', 'user'), msg.get('content', ''))
    
    async def load_conversation(self, session_id: str) -> list[dict]:
        """Load conversation messages."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM huxley_conversations WHERE session_id = $1",
                session_id
            )
            if not row:
                return []
            
            conv_id = row['id']
            rows = await conn.fetch("""
                SELECT role, content FROM huxley_messages 
                WHERE conversation_id = $1 
                ORDER BY created_at
            """, conv_id)
            
            return [{'role': r['role'], 'content': r['content']} for r in rows]
    
    # Vector search (if pgvector is available)
    async def store_embedding(
        self, 
        doc_id: str, 
        content: str, 
        embedding: list[float],
        metadata: dict = None
    ) -> None:
        """Store document with embedding."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO huxley_embeddings (doc_id, content, embedding, metadata)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (doc_id) DO UPDATE SET
                    content = $2, embedding = $3, metadata = $4
            """, doc_id, content, embedding, json.dumps(metadata or {}))
    
    async def semantic_search(
        self, 
        query_embedding: list[float], 
        limit: int = 5,
        threshold: float = 0.7
    ) -> list[dict]:
        """Search by embedding similarity."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT doc_id, content, metadata,
                       1 - (embedding <=> $1::vector) as similarity
                FROM huxley_embeddings
                WHERE 1 - (embedding <=> $1::vector) > $2
                ORDER BY similarity DESC
                LIMIT $3
            """, query_embedding, threshold, limit)
            
            return [
                {
                    'doc_id': r['doc_id'],
                    'content': r['content'],
                    'metadata': json.loads(r['metadata']) if r['metadata'] else {},
                    'similarity': r['similarity']
                }
                for r in rows
            ]
