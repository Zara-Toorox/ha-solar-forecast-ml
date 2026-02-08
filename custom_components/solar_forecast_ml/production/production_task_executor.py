# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Task Executor for Solar Forecast ML.

Executes scheduled tasks for production tracking with
result tracking and background execution support.

@author zara
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, TYPE_CHECKING

from ..core.core_helpers import SafeDateTimeUtil as dt_util

if TYPE_CHECKING:
    from ..data.db_manager import DatabaseManager

_LOGGER = logging.getLogger(__name__)


class TaskExecutor:
    """Executes scheduled tasks for production tracking. @zara"""

    def __init__(self, db_manager: Optional["DatabaseManager"] = None):
        """Initialize task executor. @zara"""
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._task_results: Dict[str, Dict[str, Any]] = {}
        self.db = db_manager

    async def execute_task(
        self,
        task_id: str,
        task_func: Callable[[], Awaitable[Any]],
        description: str = ""
    ) -> Optional[Any]:
        """Execute a task and store result. @zara"""
        try:
            _LOGGER.debug("Executing task: %s - %s", task_id, description)

            if task_id in self._running_tasks:
                await self.cancel_task(task_id)

            start_time = dt_util.now()
            result = await task_func()
            end_time = dt_util.now()
            duration = (end_time - start_time).total_seconds()

            self._task_results[task_id] = {
                "success": True,
                "result": result,
                "timestamp": end_time.isoformat(),
                "description": description,
                "duration_seconds": duration,
            }

            if self.db:
                await self._store_task_result(
                    task_id, True, description, duration, None
                )

            _LOGGER.debug("Task completed: %s (%.2fs)", task_id, duration)
            return result

        except Exception as e:
            _LOGGER.error("Task failed: %s - %s", task_id, e, exc_info=True)

            self._task_results[task_id] = {
                "success": False,
                "error": str(e),
                "timestamp": dt_util.now().isoformat(),
                "description": description,
            }

            if self.db:
                await self._store_task_result(
                    task_id, False, description, 0, str(e)
                )

            return None

    async def execute_task_background(
        self,
        task_id: str,
        task_func: Callable[[], Awaitable[Any]],
        description: str = ""
    ) -> asyncio.Task:
        """Execute task in background. @zara"""
        if task_id in self._running_tasks:
            await self.cancel_task(task_id)

        task = asyncio.create_task(
            self.execute_task(task_id, task_func, description)
        )

        self._running_tasks[task_id] = task

        task.add_done_callback(lambda t: self._cleanup_running_task(task_id))

        _LOGGER.debug("Started background task: %s", task_id)
        return task

    def _cleanup_running_task(self, task_id: str) -> None:
        """Clean up task from running tasks dict when done. @zara"""
        if task_id in self._running_tasks:
            del self._running_tasks[task_id]

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task. @zara"""
        if task_id not in self._running_tasks:
            return False

        task = self._running_tasks[task_id]

        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        del self._running_tasks[task_id]
        _LOGGER.debug("Cancelled task: %s", task_id)

        return True

    async def cancel_all_tasks(self) -> None:
        """Cancel all running tasks. @zara"""
        task_ids = list(self._running_tasks.keys())

        for task_id in task_ids:
            await self.cancel_task(task_id)

        _LOGGER.info("All tasks cancelled")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task. @zara"""
        if task_id in self._running_tasks:
            task = self._running_tasks[task_id]
            return {
                "status": "running" if not task.done() else "completed",
                "done": task.done(),
            }

        if task_id in self._task_results:
            result = self._task_results[task_id]
            return {
                "status": "completed" if result.get("success") else "failed",
                **result,
            }

        return None

    def get_all_task_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tasks. @zara"""
        statuses = {}

        for task_id in self._running_tasks:
            statuses[task_id] = self.get_task_status(task_id)

        for task_id in self._task_results:
            if task_id not in statuses:
                statuses[task_id] = self._task_results[task_id]

        return statuses

    def is_task_running(self, task_id: str) -> bool:
        """Check if a task is currently running. @zara"""
        if task_id not in self._running_tasks:
            return False
        return not self._running_tasks[task_id].done()

    def get_running_tasks(self) -> list:
        """Get list of currently running task IDs. @zara"""
        return [
            task_id
            for task_id, task in self._running_tasks.items()
            if not task.done()
        ]

    def clear_task_results(self) -> None:
        """Clear stored task results. @zara"""
        self._task_results.clear()
        _LOGGER.debug("Task results cleared")

    async def _store_task_result(
        self,
        task_id: str,
        success: bool,
        description: str,
        duration: float,
        error: Optional[str]
    ) -> None:
        """Store task result in database (if available). @zara"""
        if not self.db:
            return

        try:
            pass

        except Exception as e:
            _LOGGER.debug("Could not store task result: %s", e)

    async def wait_for_task(
        self,
        task_id: str,
        timeout: float = 60.0
    ) -> Optional[Any]:
        """Wait for a specific task to complete. @zara"""
        if task_id not in self._running_tasks:
            if task_id in self._task_results:
                return self._task_results[task_id].get("result")
            return None

        task = self._running_tasks[task_id]

        try:
            await asyncio.wait_for(task, timeout=timeout)
            return self._task_results.get(task_id, {}).get("result")
        except asyncio.TimeoutError:
            _LOGGER.warning("Task %s timed out after %.1fs", task_id, timeout)
            return None
        except asyncio.CancelledError:
            return None

    async def wait_for_all_tasks(self, timeout: float = 120.0) -> None:
        """Wait for all running tasks to complete. @zara"""
        if not self._running_tasks:
            return

        tasks = list(self._running_tasks.values())

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            _LOGGER.warning(
                "Not all tasks completed within timeout (%.1fs)",
                timeout
            )


class TaskQueue:
    """Simple task queue for sequential execution. @zara"""

    def __init__(self, executor: TaskExecutor):
        """Initialize task queue. @zara"""
        self._executor = executor
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the queue worker. @zara"""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        _LOGGER.debug("Task queue worker started")

    async def stop(self) -> None:
        """Stop the queue worker. @zara"""
        self._running = False

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        _LOGGER.debug("Task queue worker stopped")

    async def enqueue(
        self,
        task_id: str,
        task_func: Callable[[], Awaitable[Any]],
        description: str = ""
    ) -> None:
        """Add a task to the queue. @zara"""
        await self._queue.put((task_id, task_func, description))
        _LOGGER.debug("Task queued: %s", task_id)

    async def _worker(self) -> None:
        """Worker that processes tasks from the queue. @zara"""
        while self._running:
            try:
                task_id, task_func, description = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )

                await self._executor.execute_task(task_id, task_func, description)
                self._queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                _LOGGER.error("Task queue worker error: %s", e)

    @property
    def pending_count(self) -> int:
        """Get number of pending tasks in queue. @zara"""
        return self._queue.qsize()
