"""Task Scheduler for Solar Forecast ML Integration V12.2.0 @zara

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from datetime import datetime, time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_change

from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)

class TaskScheduler:
    """Schedules and manages recurring tasks"""

    def __init__(self, hass: HomeAssistant):
        """Initialize task scheduler @zara"""
        self.hass = hass
        self._scheduled_tasks: Dict[str, Any] = {}
        self._listeners: Dict[str, Callable] = {}

    def schedule_daily_task(
        self,
        task_id: str,
        hour: int,
        minute: int,
        task_func: Callable[[], Awaitable[None]],
        description: str = "",
    ) -> None:
        """Schedule a daily recurring task"""

        if task_id in self._listeners:
            self.cancel_task(task_id)

        listener_remove = async_track_time_change(
            self.hass,
            lambda now: self.hass.async_create_task(task_func()),
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
        }

        _LOGGER.info(f"Scheduled daily task: {task_id} at {hour:02d}:{minute:02d} - {description}")

    def schedule_hourly_task(
        self,
        task_id: str,
        minute: int,
        task_func: Callable[[], Awaitable[None]],
        description: str = "",
    ) -> None:
        """Schedule an hourly recurring task"""

        if task_id in self._listeners:
            self.cancel_task(task_id)

        listener_remove = async_track_time_change(
            self.hass, lambda now: self.hass.async_create_task(task_func()), minute=minute, second=0
        )

        self._listeners[task_id] = listener_remove
        self._scheduled_tasks[task_id] = {
            "type": "hourly",
            "minute": minute,
            "description": description,
            "scheduled_at": dt_util.now().isoformat(),
        }

        _LOGGER.info(f"Scheduled hourly task: {task_id} at minute {minute} - {description}")

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task @zara"""
        if task_id not in self._listeners:
            return False

        self._listeners[task_id]()

        del self._listeners[task_id]
        del self._scheduled_tasks[task_id]

        _LOGGER.debug(f"Cancelled scheduled task: {task_id}")
        return True

    def cancel_all_tasks(self) -> None:
        """Cancel all scheduled tasks @zara"""
        task_ids = list(self._listeners.keys())

        for task_id in task_ids:
            self.cancel_task(task_id)

        _LOGGER.info("All scheduled tasks cancelled")

    def get_scheduled_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all scheduled tasks @zara"""
        return self._scheduled_tasks.copy()

    def is_task_scheduled(self, task_id: str) -> bool:
        """Check if a task is currently scheduled @zara"""
        return task_id in self._scheduled_tasks
