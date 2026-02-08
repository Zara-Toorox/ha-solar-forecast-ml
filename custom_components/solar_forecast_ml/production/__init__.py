# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Production module for Solar Forecast ML.

Contains production tracking, scheduling, and forecast management components.

@author zara
"""

from .production_adaptive_forecast import AdaptiveForecastEngine
from .production_external_helpers import (
    BaseExternalSensor,
    SensorValueExtractor,
    format_time_ago,
)
from .production_history import ProductionCalculator
from .production_morning_routine import MorningRoutineHandler
from .production_rule_based_strategy import RuleBasedForecastStrategy
from .production_scheduled_tasks import ScheduledTasksManager
from .production_task_executor import TaskExecutor, TaskQueue
from .production_task_scheduler import TaskScheduler, ScheduledTaskTracker
from .production_tracker import ProductionTimeCalculator

__all__ = [
    "AdaptiveForecastEngine",
    "BaseExternalSensor",
    "SensorValueExtractor",
    "format_time_ago",
    "ProductionCalculator",
    "MorningRoutineHandler",
    "RuleBasedForecastStrategy",
    "ScheduledTasksManager",
    "TaskExecutor",
    "TaskQueue",
    "TaskScheduler",
    "ScheduledTaskTracker",
    "ProductionTimeCalculator",
]
