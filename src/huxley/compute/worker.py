"""
Worker abstraction for distributed compute.

Workers execute tasks from a queue and report results.
Designed for stateless, horizontally-scalable deployment.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine

from huxley.compute.tasks import InMemoryTaskQueue, Task, TaskQueue, TaskStatus
from huxley.core.logging import get_logger
from huxley.core.types import generate_id

logger = get_logger(__name__)


@dataclass
class WorkerInfo:
    """Information about a worker."""

    id: str = field(default_factory=generate_id)
    name: str = ""
    status: str = "idle"
    current_task_id: str | None = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class Worker:
    """
    A worker that processes tasks from a queue.

    Workers are stateless and designed to be run in
    separate processes or containers.
    """

    def __init__(
        self,
        queue: TaskQueue,
        *,
        name: str = "",
        concurrency: int = 1,
    ) -> None:
        """
        Initialize a worker.

        Args:
            queue: Task queue to pull from
            name: Worker name for identification
            concurrency: Number of concurrent tasks
        """
        self._queue = queue
        self._info = WorkerInfo(name=name or f"worker-{generate_id()[:8]}")
        self._concurrency = concurrency
        self._running = False
        self._task_handlers: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._current_tasks: set[str] = set()

    @property
    def info(self) -> WorkerInfo:
        """Get worker information."""
        return self._info

    def register_handler(
        self,
        func_name: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """
        Register a task handler.

        Args:
            func_name: Function name that tasks will reference
            handler: Async function to handle the task
        """
        self._task_handlers[func_name] = handler
        logger.debug("handler_registered", func_name=func_name, worker=self._info.id)

    async def start(self) -> None:
        """Start the worker."""
        self._running = True
        self._info.status = "running"

        logger.info(
            "worker_started",
            worker_id=self._info.id,
            name=self._info.name,
            concurrency=self._concurrency,
        )

        # Start worker tasks
        tasks = [
            asyncio.create_task(self._process_loop())
            for _ in range(self._concurrency)
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("worker_cancelled", worker_id=self._info.id)
        finally:
            self._running = False
            self._info.status = "stopped"

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info("worker_stopping", worker_id=self._info.id)
        self._running = False

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                # Get next task
                task = await self._queue.dequeue(timeout=5.0)

                if task is None:
                    continue

                if task.status == TaskStatus.CANCELLED:
                    continue

                await self._process_task(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("worker_error", error=str(e), worker=self._info.id)
                await asyncio.sleep(1)

    async def _process_task(self, task: Task) -> None:
        """Process a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        task.worker_id = self._info.id
        await self._queue.update_task(task)

        self._info.current_task_id = task.id
        self._current_tasks.add(task.id)

        logger.info(
            "task_started",
            task_id=task.id,
            func=task.func_name,
            worker=self._info.id,
        )

        try:
            # Get handler
            handler = self._task_handlers.get(task.func_name)
            if not handler:
                raise ValueError(f"No handler for: {task.func_name}")

            # Execute with timeout
            result = await asyncio.wait_for(
                handler(*task.args, **task.kwargs),
                timeout=task.timeout,
            )

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            self._info.tasks_completed += 1

            logger.info(
                "task_completed",
                task_id=task.id,
                worker=self._info.id,
            )

        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.error = f"Task timed out after {task.timeout}s"
            task.completed_at = datetime.utcnow()
            self._info.tasks_failed += 1

            logger.warning(
                "task_timeout",
                task_id=task.id,
                timeout=task.timeout,
            )

        except Exception as e:
            task.error = str(e)
            task.retries += 1

            if task.retries < task.max_retries:
                # Requeue for retry
                task.status = TaskStatus.QUEUED
                task.started_at = None
                task.worker_id = None
                await self._queue.enqueue(task)

                logger.warning(
                    "task_retry",
                    task_id=task.id,
                    retry=task.retries,
                    max_retries=task.max_retries,
                )
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.utcnow()
                self._info.tasks_failed += 1

                logger.error(
                    "task_failed",
                    task_id=task.id,
                    error=str(e),
                )

        finally:
            await self._queue.update_task(task)
            self._current_tasks.discard(task.id)
            self._info.current_task_id = None
            self._info.last_heartbeat = datetime.utcnow()


class WorkerPool:
    """
    Manages a pool of workers.

    Provides convenient methods for creating and managing
    multiple workers with shared configuration.
    """

    def __init__(
        self,
        queue: TaskQueue | None = None,
        *,
        num_workers: int = 4,
        concurrency_per_worker: int = 1,
    ) -> None:
        """
        Initialize worker pool.

        Args:
            queue: Task queue (in-memory if not provided)
            num_workers: Number of workers to create
            concurrency_per_worker: Tasks per worker
        """
        self._queue = queue or InMemoryTaskQueue()
        self._num_workers = num_workers
        self._concurrency = concurrency_per_worker
        self._workers: list[Worker] = []
        self._worker_tasks: list[asyncio.Task[Any]] = []
        self._task_handlers: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}

    @property
    def queue(self) -> TaskQueue:
        """Get the task queue."""
        return self._queue

    def register_handler(
        self,
        func_name: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a handler for all workers."""
        self._task_handlers[func_name] = handler

    async def submit(
        self,
        func_name: str,
        *args: Any,
        priority: int = 0,
        timeout: float = 3600.0,
        **kwargs: Any,
    ) -> str:
        """
        Submit a task to the pool.

        Args:
            func_name: Handler function name
            *args: Positional arguments
            priority: Task priority
            timeout: Task timeout
            **kwargs: Keyword arguments

        Returns:
            Task ID
        """
        task = Task(
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
        )
        return await self._queue.enqueue(task)

    async def get_result(
        self,
        task_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 0.1,
    ) -> Any:
        """
        Wait for and return task result.

        Args:
            task_id: Task identifier
            timeout: How long to wait
            poll_interval: How often to check

        Returns:
            Task result

        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If task failed
        """
        start = asyncio.get_event_loop().time()

        while True:
            task = await self._queue.get_task(task_id)

            if task is None:
                raise ValueError(f"Task not found: {task_id}")

            if task.status == TaskStatus.COMPLETED:
                return task.result

            if task.status in (TaskStatus.FAILED, TaskStatus.TIMEOUT):
                raise RuntimeError(f"Task failed: {task.error}")

            if task.status == TaskStatus.CANCELLED:
                raise RuntimeError("Task was cancelled")

            if timeout:
                elapsed = asyncio.get_event_loop().time() - start
                if elapsed > timeout:
                    raise TimeoutError(f"Timeout waiting for task {task_id}")

            await asyncio.sleep(poll_interval)

    async def start(self) -> None:
        """Start all workers."""
        for i in range(self._num_workers):
            worker = Worker(
                self._queue,
                name=f"pool-worker-{i}",
                concurrency=self._concurrency,
            )

            # Register all handlers
            for func_name, handler in self._task_handlers.items():
                worker.register_handler(func_name, handler)

            self._workers.append(worker)
            self._worker_tasks.append(asyncio.create_task(worker.start()))

        logger.info(
            "worker_pool_started",
            num_workers=self._num_workers,
        )

    async def stop(self) -> None:
        """Stop all workers."""
        for worker in self._workers:
            await worker.stop()

        for task in self._worker_tasks:
            task.cancel()

        await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        self._workers.clear()
        self._worker_tasks.clear()

        logger.info("worker_pool_stopped")

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "num_workers": len(self._workers),
            "workers": [
                {
                    "id": w.info.id,
                    "name": w.info.name,
                    "status": w.info.status,
                    "tasks_completed": w.info.tasks_completed,
                    "tasks_failed": w.info.tasks_failed,
                }
                for w in self._workers
            ],
        }
