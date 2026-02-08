# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Task Scheduler for Solar Forecast ML.

Schedules and manages recurring tasks using Home Assistant's
event tracking system with database-backed task tracking.

@author zara
"""

import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional, TYPE_CHECKING

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_change

from ..core.core_helpers import SafeDateTimeUtil as dt_util

if TYPE_CHECKING:
    from ..data.db_manager import DatabaseManager

_LOGGER = logging.getLogger(__name__)


class TaskScheduler:
    """Schedules and manages recurring tasks. @zara"""

    def __init__(
        self,
        hass: HomeAssistant,
        db_manager: Optional["DatabaseManager"] = None
    ):
        """Initialize task scheduler. @zara"""
        self.hass = hass
        self.db = db_manager
        self._scheduled_tasks: Dict[str, Dict[str, Any]] = {}
        self._listeners: Dict[str, Callable] = {}

    def schedule_daily_task(
        self,
        task_id: str,
        hour: int,
        minute: int,
        task_func: Callable[[], Awaitable[None]],
        description: str = "",
    ) -> None:
        """Schedule a daily recurring task. @zara"""
        if task_id in self._listeners:
            self.cancel_task(task_id)

        listener_remove = async_track_time_change(
            self.hass,
            lambda now: self.hass.async_create_task(self._execute_task(task_id, task_func)),
            hour=hour,
            minute=minute,
            second=0,
        )

        self._listeners[task_id] = listener_remove
        self._scheduled_tasks[task_id] = {
            "type": "daily",
            "hour": hour,
            "minute": minute,
            "description": description,
            "scheduled_at": dt_util.now().isoformat(),
            "last_run": None,
            "run_count": 0,
        }

        _LOGGER.info(
            "Scheduled daily task: %s at %02d:%02d - %s",
            task_id, hour, minute, description
        )

    def schedule_hourly_task(
        self,
        task_id: str,
        minute: int,
        task_func: Callable[[], Awaitable[None]],
        description: str = "",
    ) -> None:
        """Schedule an hourly recurring task. @zara"""
        if task_id in self._listeners:
            self.cancel_task(task_id)

        listener_remove = async_track_time_change(
            self.hass,
            lambda now: self.hass.async_create_task(self._execute_task(task_id, task_func)),
            minute=minute,
            second=0,
        )

        self._listeners[task_id] = listener_remove
        self._scheduled_tasks[task_id] = {
            "type": "hourly",
            "minute": minute,
            "description": description,
            "scheduled_at": dt_util.now().isoformat(),
            "last_run": None,
            "run_count": 0,
        }

        _LOGGER.info(
            "Scheduled hourly task: %s at minute %d - %s",
            task_id, minute, description
        )

    def schedule_multi_hourly_task(
        self,
        task_id: str,
        minutes: list,
        task_func: Callable[[], Awaitable[None]],
        description: str = "",
    ) -> None:
        """Schedule a task to run at multiple minutes each hour. @zara"""
        if task_id in self._listeners:
            self.cancel_task(task_id)

        listener_remove = async_track_time_change(
            self.hass,
            lambda now: self.hass.async_create_task(self._execute_task(task_id, task_func)),
            minute=minutes,
            second=0,
        )

        self._listeners[task_id] = listener_remove
        self._scheduled_tasks[task_id] = {
            "type": "multi_hourly",
            "minutes": minutes,
            "description": description,
            "scheduled_at": dt_util.now().isoformat(),
            "last_run": None,
            "run_count": 0,
        }

        _LOGGER.info(
            "Scheduled multi-hourly task: %s at minutes %s - %s",
            task_id, minutes, description
        )

    async def _execute_task(
        self,
        task_id: str,
        task_func: Callable[[], Awaitable[None]]
    ) -> None:
        """Execute a scheduled task and update tracking. @zara"""
        if task_id in self._scheduled_tasks:
            self._scheduled_tasks[task_id]["last_run"] = dt_util.now().isoformat()
            self._scheduled_tasks[task_id]["run_count"] += 1

        try:
            await task_func()
            _LOGGER.debug("Scheduled task executed: %s", task_id)
        except Exception as e:
            _LOGGER.error("Scheduled task failed: %s - %s", task_id, e, exc_info=True)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task. @zara"""
        if task_id not in self._listeners:
            return False

        self._listeners[task_id]()

        del self._listeners[task_id]
        if task_id in self._scheduled_tasks:
            del self._scheduled_tasks[task_id]

        _LOGGER.debug("Cancelled scheduled task: %s", task_id)
        return True

    def cancel_all_tasks(self) -> None:
        """Cancel all scheduled tasks. @zara"""
        task_ids = list(self._listeners.keys())

        for task_id in task_ids:
            self.cancel_task(task_id)

        _LOGGER.info("All scheduled tasks cancelled")

    def get_scheduled_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all scheduled tasks. @zara"""
        return self._scheduled_tasks.copy()

    def is_task_scheduled(self, task_id: str) -> bool:
        """Check if a task is currently scheduled. @zara"""
        return task_id in self._scheduled_tasks

    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific task. @zara"""
        return self._scheduled_tasks.get(task_id)

    def get_next_run_time(self, task_id: str) -> Optional[datetime]:
        """Calculate next run time for a scheduled task. @zara"""
        task_info = self._scheduled_tasks.get(task_id)
        if not task_info:
            return None

        now = dt_util.now()
        task_type = task_info.get("type")

        if task_type == "daily":
            hour = task_info.get("hour", 0)
            minute = task_info.get("minute", 0)

            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            if next_run <= now:
                next_run = next_run.replace(day=now.day + 1)

            return next_run

        elif task_type == "hourly":
            minute = task_info.get("minute", 0)

            next_run = now.replace(minute=minute, second=0, microsecond=0)

            if next_run <= now:
                next_run = next_run.replace(hour=now.hour + 1)

            return next_run

        elif task_type == "multi_hourly":
            minutes = task_info.get("minutes", [0])

            for m in sorted(minutes):
                next_run = now.replace(minute=m, second=0, microsecond=0)
                if next_run > now:
                    return next_run

            next_run = now.replace(hour=now.hour + 1, minute=min(minutes), second=0, microsecond=0)
            return next_run

        return None

    def reschedule_task(
        self,
        task_id: str,
        hour: Optional[int] = None,
        minute: Optional[int] = None
    ) -> bool:
        """Reschedule an existing task with new timing. @zara"""
        task_info = self._scheduled_tasks.get(task_id)
        if not task_info:
            _LOGGER.warning("Cannot reschedule unknown task: %s", task_id)
            return False

        _LOGGER.warning(
            "Rescheduling requires re-creating the task. "
            "Call the appropriate schedule method with new timing."
        )
        return False


class ScheduledTaskTracker:
    """Tracks scheduled task execution history in database. @zara"""

    def __init__(self, db_manager: "DatabaseManager"):
        """Initialize task tracker. @zara"""
        self.db = db_manager

    async def record_execution(
        self,
        task_id: str,
        success: bool,
        duration_seconds: float = 0,
        error_message: Optional[str] = None
    ) -> None:
        """Record task execution in database. @zara"""
        pass

    async def get_execution_history(
        self,
        task_id: str,
        days: int = 7
    ) -> list:
        """Get execution history for a task. @zara"""
        return []

    async def get_task_statistics(self, task_id: str) -> Dict[str, Any]:
        """Get statistics for a task. @zara"""
        return {
            "task_id": task_id,
            "total_runs": 0,
            "success_count": 0,
            "failure_count": 0,
            "success_rate": 0.0,
            "avg_duration": 0.0,
        }
