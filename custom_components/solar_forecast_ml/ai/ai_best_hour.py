# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Best Hour Calculator for Solar Forecast ML V16.2.0.

Calculates the hour with highest predicted production.
Uses database instead of JSON files.

@zara
"""

import logging
from datetime import datetime
from typing import Any, Optional, Tuple

_LOGGER = logging.getLogger(__name__)


class BestHourCalculator:
    """Calculate best production hour based on forecasts @zara"""

    def __init__(self, db_manager: Any):
        """Initialize calculator @zara"""
        self.db_manager = db_manager
        self.ai_predictor = None

    async def calculate_best_hour_today(self) -> Tuple[Optional[int], Optional[float]]:
        """Find hour with highest predicted production @zara"""
        try:
            today = datetime.now().date().isoformat()

            rows = await self.db_manager.fetchall(
                """SELECT target_hour, prediction_kwh
                   FROM hourly_predictions
                   WHERE target_date = ? AND prediction_kwh > 0
                   ORDER BY prediction_kwh DESC
                   LIMIT 1""",
                (today,)
            )

            if rows:
                return rows[0][0], rows[0][1]
            else:
                return self._get_solar_noon_fallback()

        except Exception as e:
            _LOGGER.debug(f"Best hour calculation failed: {e}")
            return self._get_solar_noon_fallback()

    def _get_solar_noon_fallback(self) -> Tuple[int, float]:
        """Return solar noon as fallback @zara"""
        return 12, 0.0
