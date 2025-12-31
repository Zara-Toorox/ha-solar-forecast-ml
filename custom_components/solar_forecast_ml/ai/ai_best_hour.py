# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from datetime import datetime
from typing import Any, Optional, Tuple

_LOGGER = logging.getLogger(__name__)


class BestHourCalculator:
    """Calculate best production hour based on forecasts @zara"""

    def __init__(self, data_manager: Any):
        """Initialize calculator @zara"""
        self.data_manager = data_manager
        self.ai_predictor = None

    async def calculate_best_hour_today(self) -> Tuple[Optional[int], Optional[float]]:
        """Find hour with highest predicted production @zara"""
        try:
            hourly_data = await self.data_manager.hourly_predictions._read_json_async()
            if not hourly_data:
                return self._get_solar_noon_fallback()

            predictions = hourly_data.get("predictions", [])
            if not predictions:
                return self._get_solar_noon_fallback()

            today = datetime.now().date().isoformat()
            today_predictions = [
                p for p in predictions
                if p.get("date") == today and p.get("predicted_kwh", 0) > 0
            ]

            if not today_predictions:
                return self._get_solar_noon_fallback()

            best = max(today_predictions, key=lambda x: x.get("predicted_kwh", 0))
            return best.get("hour"), best.get("predicted_kwh")

        except Exception as e:
            _LOGGER.debug(f"Best hour calculation failed: {e}")
            return self._get_solar_noon_fallback()

    def _get_solar_noon_fallback(self) -> Tuple[int, float]:
        """Return solar noon as fallback @zara"""
        return 12, 0.0
