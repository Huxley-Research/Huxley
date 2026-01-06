"""
Task queue abstraction.

Provides a consistent interface for task queuing with
support for multiple backends (local, Redis, etc.).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine

from huxley.core.types import generate_id


class TaskStatus(str, Enum):
    """Status of a task in the queue."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """
    A unit of work to be executed.

    Tasks are serializable and can be distributed across workers.
    """

    id: str = field(default_factory=generate_id)
    name: str = ""
    func_name: str = ""
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    priority: int = 0  # Higher = more urgent
    timeout: float = 3600.0  # 1 hour default
    retries: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    worker_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "func_name": self.func_name,
            "args": self.args,
            "kwargs": self.kwargs,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "priority": self.priority,
            "timeout": self.timeout,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "worker_id": self.worker_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Deserialize task from dictionary."""
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            func_name=data["func_name"],
            args=tuple(data.get("args", ())),
            kwargs=data.get("kwargs", {}),
            status=TaskStatus(data["status"]),
            result=data.get("result"),
            error=data.get("error"),
            priority=data.get("priority", 0),
            timeout=data.get("timeout", 3600.0),
            retries=data.get("retries", 0),
            max_retries=data.get("max_retries", 3),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            worker_id=data.get("worker_id"),
            metadata=data.get("metadata", {}),
        )


class TaskQueue(ABC):
    """
    Abstract base class for task queues.

    Implementations handle the actual queuing mechanism.
    """

    @abstractmethod
    async def enqueue(self, task: Task) -> str:
        """
        Add a task to the queue.

        Args:
            task: Task to enqueue

        Returns:
            Task ID
        """
        ...

    @abstractmethod
    async def dequeue(self, timeout: float | None = None) -> Task | None:
        """
        Get the next task from the queue.

        Args:
            timeout: How long to wait for a task

        Returns:
            Next task or None if timeout/empty
        """
        ...

    @abstractmethod
    async def get_task(self, task_id: str) -> Task | None:
        """
        Get a task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task or None if not found
        """
        ...

    @abstractmethod
    async def update_task(self, task: Task) -> None:
        """
        Update a task's state.

        Args:
            task: Task with updated state
        """
        ...

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task.

        Args:
            task_id: Task identifier

        Returns:
            True if cancelled
        """
        ...

    @abstractmethod
    async def get_queue_length(self) -> int:
        """Get number of pending tasks."""
        ...


class InMemoryTaskQueue(TaskQueue):
    """
    Simple in-memory task queue.

    Suitable for development and single-process deployments.
    """

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, str, Task]] = asyncio.PriorityQueue()
        self._tasks: dict[str, Task] = {}
        self._counter = 0  # For stable sorting

    async def enqueue(self, task: Task) -> str:
        task.status = TaskStatus.QUEUED
        self._tasks[task.id] = task
        self._counter += 1
        # Negate priority so higher priority = lower number = dequeued first
        await self._queue.put((-task.priority, self._counter, task))
        return task.id

    async def dequeue(self, timeout: float | None = None) -> Task | None:
        try:
            if timeout:
                _, _, task = await asyncio.wait_for(
                    self._queue.get(), timeout=timeout
                )
            else:
                _, _, task = await self._queue.get()

            # Refresh from storage in case it was updated
            return self._tasks.get(task.id)

        except asyncio.TimeoutError:
            return None

    async def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    async def update_task(self, task: Task) -> None:
        self._tasks[task.id] = task

    async def cancel_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task and task.status in (TaskStatus.PENDING, TaskStatus.QUEUED):
            task.status = TaskStatus.CANCELLED
            return True
        return False

    async def get_queue_length(self) -> int:
        return self._queue.qsize()


class RedisTaskQueue(TaskQueue):
    """
    Redis-backed task queue for distributed deployments.

    Uses Redis sorted sets for priority queuing.
    """

    def __init__(
        self,
        url: str | None = None,
        queue_name: str = "huxley:tasks",
    ) -> None:
        self._url = url
        self._queue_name = queue_name
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as redis

            if self._url:
                self._client = redis.from_url(self._url)
            else:
                from huxley.core.config import get_config

                self._client = redis.from_url(get_config().redis.url)

        return self._client

    async def enqueue(self, task: Task) -> str:
        import json

        client = await self._get_client()
        task.status = TaskStatus.QUEUED

        # Store task data
        await client.set(f"task:{task.id}", json.dumps(task.to_dict()))

        # Add to priority queue (score = -priority for proper ordering)
        await client.zadd(self._queue_name, {task.id: -task.priority})

        return task.id

    async def dequeue(self, timeout: float | None = None) -> Task | None:
        import json

        client = await self._get_client()

        # Pop highest priority task
        result = await client.bzpopmin(self._queue_name, timeout=timeout or 0)

        if not result:
            return None

        _, task_id, _ = result
        task_id = task_id.decode() if isinstance(task_id, bytes) else task_id

        # Get task data
        data = await client.get(f"task:{task_id}")
        if not data:
            return None

        return Task.from_dict(json.loads(data))

    async def get_task(self, task_id: str) -> Task | None:
        import json

        client = await self._get_client()
        data = await client.get(f"task:{task_id}")

        if not data:
            return None

        return Task.from_dict(json.loads(data))

    async def update_task(self, task: Task) -> None:
        import json

        client = await self._get_client()
        await client.set(f"task:{task.id}", json.dumps(task.to_dict()))

    async def cancel_task(self, task_id: str) -> bool:
        client = await self._get_client()
        task = await self.get_task(task_id)

        if task and task.status in (TaskStatus.PENDING, TaskStatus.QUEUED):
            task.status = TaskStatus.CANCELLED
            await self.update_task(task)
            await client.zrem(self._queue_name, task_id)
            return True

        return False

    async def get_queue_length(self) -> int:
        client = await self._get_client()
        return await client.zcard(self._queue_name)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
