"""Production History Tracking V12.0.0 @zara

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

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

from homeassistant.core import HomeAssistant

from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)

class ProductionCalculator:
    """Historical Production Calculator - Simplified Version"""

    def __init__(self, hass: HomeAssistant, data_manager):
        """Initialize the Production Calculator @zara"""
        self.hass = hass
        self.data_manager = data_manager
        _LOGGER.info("ProductionCalculator initialized (Recorder-free mode)")

    async def async_get_peak_production_time(self, power_entity: Optional[str] = None) -> str:
        """Calculate peak production time using ML data @zara"""
        _LOGGER.debug("Peak time calculation: Using ML-based approach (no Recorder)")

        try:
            today = dt_util.now().date()
            today_str = today.isoformat()

            today_samples = await self.data_manager.hourly_predictions.get_predictions_for_date(today_str)

            if not today_samples:
                return "12:00"

            max_production = -1
            peak_hour = 12

            for sample in today_samples:
                production = sample.get("actual_kwh")
                if production is not None and production > max_production:
                    max_production = production

                    peak_hour = sample.get("hour", 12)

            return f"{peak_hour:02d}:00"

        except Exception as e:
            _LOGGER.error(f"Error calculating peak production time: {e}", exc_info=True)
            return "12:00"

    async def calculate_yesterday_total_yield(self, yield_entity: str) -> float:
        """Calculate yesterdays total yield from prediction history @zara"""
        try:
            yesterday_str = (dt_util.now() - timedelta(days=1)).date().isoformat()
            prediction = await self.data_manager.get_prediction_for_date(yesterday_str)

            if prediction and "actual_kwh" in prediction:
                return prediction["actual_kwh"]

            _LOGGER.debug("No actual kWh found for yesterday in prediction history.")
            return 0.0

        except Exception as e:
            _LOGGER.error(f"Error calculating yesterday's yield from history: {e}")
            return 0.0

    async def get_last_7_days_average_yield(self, yield_entity: str) -> float:
        """Get average yield for the last 7 days from prediction history @zara"""
        try:
            start_date = (dt_util.now() - timedelta(days=7)).date().isoformat()
            predictions = await self.data_manager.get_predictions(start_date=start_date)

            total_yield = 0
            count = 0

            for pred in predictions:
                if "actual_kwh" in pred:
                    total_yield += pred["actual_kwh"]
                    count += 1

            return total_yield / count if count > 0 else 0.0

        except Exception as e:
            _LOGGER.error(f"Error calculating 7-day average yield: {e}")
            return 0.0

    async def get_monthly_production_statistics(self, yield_entity: str) -> dict:
        """Get monthly production statistics from prediction history @zara"""
        try:
            start_of_month = dt_util.now().replace(day=1).date().isoformat()
            predictions = await self.data_manager.get_predictions(start_date=start_of_month)

            actuals = [p["actual_kwh"] for p in predictions if "actual_kwh" in p]

            if not actuals:
                return {
                    "total_kwh": 0.0,
                    "average_daily_kwh": 0.0,
                    "best_day_kwh": 0.0,
                    "worst_day_kwh": 0.0,
                    "days_with_data": 0,
                }

            total_kwh = sum(actuals)
            days_with_data = len(actuals)
            average_daily_kwh = total_kwh / days_with_data
            best_day_kwh = max(actuals)
            worst_day_kwh = min(actuals)

            return {
                "total_kwh": round(total_kwh, 2),
                "average_daily_kwh": round(average_daily_kwh, 2),
                "best_day_kwh": round(best_day_kwh, 2),
                "worst_day_kwh": round(worst_day_kwh, 2),
                "days_with_data": days_with_data,
            }

        except Exception as e:
            _LOGGER.error(f"Error calculating monthly production statistics: {e}")
            return {
                "total_kwh": 0.0,
                "average_daily_kwh": 0.0,
                "best_day_kwh": 0.0,
                "worst_day_kwh": 0.0,
                "days_with_data": 0,
            }

    async def get_historical_average(self) -> Optional[float]:
        """Get historical average production @zara"""
        _LOGGER.debug("Historical average: Not available without Recorder")
        return None
