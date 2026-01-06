# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)

class TaskExecutor:
    """Executes scheduled tasks for production tracking"""

    def __init__(self):
        """Initialize task executor @zara"""
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._task_results: Dict[str, Any] = {}

    async def execute_task(
        self, task_id: str, task_func: Callable[[], Awaitable[Any]], description: str = ""
    ) -> Optional[Any]:
        """Execute a task and store result"""
        try:
            _LOGGER.debug(f"Executing task: {task_id} - {description}")

            if task_id in self._running_tasks:
                await self.cancel_task(task_id)

            result = await task_func()

            self._task_results[task_id] = {
                "success": True,
                "result": result,
                "timestamp": dt_util.now().isoformat(),
                "description": description,
            }

            _LOGGER.debug(f"Task completed: {task_id}")
            return result

        except Exception as e:
            _LOGGER.error(f"Task failed: {task_id} - {e}", exc_info=True)

            self._task_results[task_id] = {
                "success": False,
                "error": str(e),
                "timestamp": dt_util.now().isoformat(),
                "description": description,
            }

            return None

    async def execute_task_background(
        self, task_id: str, task_func: Callable[[], Awaitable[Any]], description: str = ""
    ) -> asyncio.Task:
        """Execute task in background"""

        if task_id in self._running_tasks:
            await self.cancel_task(task_id)

        task = asyncio.create_task(self.execute_task(task_id, task_func, description))

        self._running_tasks[task_id] = task

        _LOGGER.debug(f"Started background task: {task_id}")
        return task

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task @zara"""
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
        _LOGGER.debug(f"Cancelled task: {task_id}")

        return True

    async def cancel_all_tasks(self) -> None:
        """Cancel all running tasks @zara"""
        task_ids = list(self._running_tasks.keys())

        for task_id in task_ids:
            await self.cancel_task(task_id)

        _LOGGER.info("All tasks cancelled")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task @zara"""
        if task_id in self._running_tasks:
            task = self._running_tasks[task_id]
            return {
                "status": "running" if not task.done() else "completed",
                "done": task.done(),
            }

        if task_id in self._task_results:
            return self._task_results[task_id]

        return None

    def get_all_task_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tasks @zara"""
        statuses = {}

        for task_id in self._running_tasks:
            statuses[task_id] = self.get_task_status(task_id)

        for task_id in self._task_results:
            if task_id not in statuses:
                statuses[task_id] = self._task_results[task_id]

        return statuses
