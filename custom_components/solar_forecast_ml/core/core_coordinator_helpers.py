# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Coordinator Helper Functions for Solar Forecast ML V16.2.0.
Provides utility functions for the data update coordinator.
Pure utility functions with minimal database dependencies.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from .core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class CoordinatorHelpers:
    """Helper functions for the data update coordinator. @zara"""

    @staticmethod
    def calculate_next_update_time(
        last_update: Optional[datetime], interval_minutes: int = 15
    ) -> datetime:
        """Calculate the next scheduled update time. @zara"""
        if last_update is None:
            return dt_util.now()

        next_update = last_update + timedelta(minutes=interval_minutes)

        if next_update < dt_util.now():
            return dt_util.now()

        return next_update

    @staticmethod
    def should_force_update(last_update: Optional[datetime], max_age_hours: int = 24) -> bool:
        """Check if data should be force-updated due to age. @zara"""
        if last_update is None:
            return True

        age = dt_util.now() - last_update
        return age.total_seconds() > max_age_hours * 3600

    @staticmethod
    def validate_coordinator_data(data: Dict[str, Any]) -> bool:
        """Validate coordinator data structure. @zara"""
        required_keys = ["last_update", "forecasts"]

        for key in required_keys:
            if key not in data:
                _LOGGER.error(f"Missing required key in coordinator data: {key}")
                return False

        return True

    @staticmethod
    def merge_forecast_data(old_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge new forecast data with existing data. @zara"""
        merged = old_data.copy()

        for key, value in new_data.items():
            if key == "forecasts" and key in merged:
                merged[key].update(value)
            else:
                merged[key] = value

        merged["last_update"] = dt_util.now().isoformat()

        return merged

    @staticmethod
    def calculate_data_staleness(last_update: Optional[datetime]) -> Dict[str, Any]:
        """Calculate data staleness metrics. @zara"""
        if last_update is None:
            return {
                "stale": True,
                "age_seconds": None,
                "age_human": "Never updated",
                "status": "no_data",
            }

        age = dt_util.now() - last_update
        age_seconds = age.total_seconds()

        if age_seconds < 900:
            status = "fresh"
            stale = False
        elif age_seconds < 3600:
            status = "acceptable"
            stale = False
        elif age_seconds < 21600:
            status = "stale"
            stale = True
        else:
            status = "very_stale"
            stale = True

        if age_seconds < 60:
            age_human = f"{int(age_seconds)} seconds ago"
        elif age_seconds < 3600:
            age_human = f"{int(age_seconds / 60)} minutes ago"
        elif age_seconds < 86400:
            age_human = f"{int(age_seconds / 3600)} hours ago"
        else:
            age_human = f"{int(age_seconds / 86400)} days ago"

        return {
            "stale": stale,
            "age_seconds": age_seconds,
            "age_human": age_human,
            "status": status,
        }

    @staticmethod
    def format_update_summary(update_results: Dict[str, bool]) -> str:
        """Format update results into a readable summary. @zara"""
        total = len(update_results)
        successful = sum(1 for v in update_results.values() if v)
        failed = total - successful

        if failed == 0:
            return f"All {total} components updated successfully"
        elif successful == 0:
            return f"All {total} components failed to update"
        else:
            return f"{successful}/{total} components updated successfully"
