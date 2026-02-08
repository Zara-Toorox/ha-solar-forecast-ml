# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Debug Task Tracker for Solar Forecast ML
Helps identify hanging or slow tasks during startup
@zara
"""

import asyncio
import logging
from datetime import datetime

_LOGGER = logging.getLogger(__name__)


async def log_pending_tasks():
    """Log all pending asyncio tasks with details."""
    all_tasks = asyncio.all_tasks()
    pending = [t for t in all_tasks if not t.done()]

    _LOGGER.warning(f"=== PENDING TASKS DEBUG ({len(pending)} pending) ===")

    for i, task in enumerate(pending):
        try:
            # Get task name
            name = task.get_name() if hasattr(task, 'get_name') else f"Task-{i}"

            # Get coroutine info
            coro = task.get_coro() if hasattr(task, 'get_coro') else None
            if coro:
                coro_name = coro.__name__ if hasattr(coro, '__name__') else str(coro)
                coro_file = coro.cr_code.co_filename if hasattr(coro, 'cr_code') else "unknown"
                coro_line = coro.cr_frame.f_lineno if hasattr(coro, 'cr_frame') and coro.cr_frame else 0

                _LOGGER.warning(
                    f"  [{i}] {name}: {coro_name} "
                    f"at {coro_file}:{coro_line}"
                )
            else:
                _LOGGER.warning(f"  [{i}] {name}: No coro info")

        except Exception as e:
            _LOGGER.warning(f"  [{i}] Error getting task info: {e}")

    _LOGGER.warning("=== END PENDING TASKS ===")


def enable_task_debugging():
    """Enable task debugging for Solar Forecast ML."""
    _LOGGER.warning("Task debugging enabled for Solar Forecast ML")

    # Schedule periodic task logging
    async def _periodic_log():
        while True:
            await asyncio.sleep(30)  # Every 30 seconds
            await log_pending_tasks()

    # Don't return this - let it run in background
    asyncio.create_task(_periodic_log(), name="solar_ml_task_debugger")
