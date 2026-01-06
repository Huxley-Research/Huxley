"""
Memory store abstraction.

Provides a consistent interface for storing and retrieving
agent state, conversation history, and experiment data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar

from huxley.core.types import ExecutionContext, Message, generate_id

T = TypeVar("T")


class MemoryStore(ABC, Generic[T]):
    """
    Abstract base class for memory storage.

    Implementations can use different backends:
    - In-memory (for testing)
    - Redis (for distributed)
    - PostgreSQL (for persistence)
    """

    @abstractmethod
    async def get(self, key: str) -> T | None:
        """
        Retrieve a value by key.

        Args:
            key: Unique identifier

        Returns:
            Stored value or None if not found
        """
        ...

    @abstractmethod
    async def set(
        self,
        key: str,
        value: T,
        *,
        ttl: int | None = None,
    ) -> None:
        """
        Store a value.

        Args:
            key: Unique identifier
            value: Value to store
            ttl: Time-to-live in seconds (None for no expiry)
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a value.

        Args:
            key: Unique identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists.

        Args:
            key: Unique identifier

        Returns:
            True if exists
        """
        ...

    @abstractmethod
    async def list_keys(self, pattern: str = "*") -> list[str]:
        """
        List keys matching a pattern.

        Args:
            pattern: Glob-style pattern

        Returns:
            List of matching keys
        """
        ...


class InMemoryStore(MemoryStore[T]):
    """
    Simple in-memory store for development and testing.

    Not suitable for production multi-process deployments.
    """

    def __init__(self) -> None:
        self._data: dict[str, tuple[T, datetime | None]] = {}

    async def get(self, key: str) -> T | None:
        if key not in self._data:
            return None

        value, expires_at = self._data[key]

        # Check expiry
        if expires_at and datetime.utcnow() > expires_at:
            del self._data[key]
            return None

        return value

    async def set(
        self,
        key: str,
        value: T,
        *,
        ttl: int | None = None,
    ) -> None:
        expires_at = None
        if ttl:
            from datetime import timedelta

            expires_at = datetime.utcnow() + timedelta(seconds=ttl)

        self._data[key] = (value, expires_at)

    async def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        if key not in self._data:
            return False

        _, expires_at = self._data[key]
        if expires_at and datetime.utcnow() > expires_at:
            del self._data[key]
            return False

        return True

    async def list_keys(self, pattern: str = "*") -> list[str]:
        import fnmatch

        # Clean up expired keys
        now = datetime.utcnow()
        expired = [
            k for k, (_, exp) in self._data.items() if exp and now > exp
        ]
        for k in expired:
            del self._data[k]

        if pattern == "*":
            return list(self._data.keys())

        return [k for k in self._data.keys() if fnmatch.fnmatch(k, pattern)]

    async def clear(self) -> None:
        """Clear all stored data."""
        self._data.clear()


class RedisStore(MemoryStore[T]):
    """
    Redis-backed memory store for distributed deployments.

    Requires redis-py async client.
    """

    def __init__(self, url: str | None = None) -> None:
        """
        Initialize Redis store.

        Args:
            url: Redis connection URL (uses config if not provided)
        """
        self._url = url
        self._client: Any = None

    async def _get_client(self) -> Any:
        """Lazy initialization of Redis client."""
        if self._client is None:
            import redis.asyncio as redis

            if self._url:
                self._client = redis.from_url(self._url)
            else:
                from huxley.core.config import get_config

                self._client = redis.from_url(get_config().redis.url)

        return self._client

    async def get(self, key: str) -> T | None:
        import json

        client = await self._get_client()
        data = await client.get(key)

        if data is None:
            return None

        return json.loads(data)

    async def set(
        self,
        key: str,
        value: T,
        *,
        ttl: int | None = None,
    ) -> None:
        import json

        client = await self._get_client()
        data = json.dumps(value, default=str)

        if ttl:
            await client.setex(key, ttl, data)
        else:
            await client.set(key, data)

    async def delete(self, key: str) -> bool:
        client = await self._get_client()
        result = await client.delete(key)
        return result > 0

    async def exists(self, key: str) -> bool:
        client = await self._get_client()
        return await client.exists(key) > 0

    async def list_keys(self, pattern: str = "*") -> list[str]:
        client = await self._get_client()
        keys = await client.keys(pattern)
        return [k.decode() if isinstance(k, bytes) else k for k in keys]

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
