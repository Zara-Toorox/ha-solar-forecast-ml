"""
Production History Calculator for Solar Forecast ML integration.
Simplified version WITHOUT Home Assistant Recorder dependency.

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

from ..core.helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class ProductionCalculator:
    """
    Historical Production Calculator - Simplified Version.
    
    REMOVED: Recorder-based peak time calculation (caused blocking).
    USES: ML-based predictions and fallback values.
    """

    def __init__(self, hass: HomeAssistant):
        """Initialize the Production Calculator."""
        self.hass = hass
        _LOGGER.info("ProductionCalculator initialized (Recorder-free mode)")

    async def calculate_peak_production_time(
        self,
        power_entity: Optional[str] = None
    ) -> str:
        """
        Calculate peak production time using ML data.
        Returns default if no ML data available.
        
        RECORDER REMOVED: No longer queries Home Assistant database.
        
        Args:
            power_entity: Not used anymore (kept for compatibility)
            
        Returns:
            Peak hour as "HH:00" or "12:00" as fallback
        """
        _LOGGER.debug("Peak time calculation: Using ML-based approach (no Recorder)")
        
        # TODO: In future, analyze ML hourly samples to find peak
        # For now, return noon as sensible default
        
        return "12:00"

    async def calculate_yesterday_total_yield(self, yield_entity: str) -> float:
        """
        Calculate yesterday's total yield.
        
        RECORDER REMOVED: Returns 0.0 (no historical data access).
        Use ML prediction records instead.
        
        Args:
            yield_entity: Solar yield entity (not used)
            
        Returns:
            0.0 (no historical access)
        """
        _LOGGER.debug("Yesterday yield calculation: Not available without Recorder")
        return 0.0

    async def get_last_7_days_average_yield(self, yield_entity: str) -> float:
        """
        Get average yield for last 7 days.
        
        RECORDER REMOVED: Returns 0.0 (no historical data access).
        Use ML prediction records instead.
        
        Args:
            yield_entity: Solar yield entity (not used)
            
        Returns:
            0.0 (no historical access)
        """
        _LOGGER.debug("7-day average: Not available without Recorder")
        return 0.0

    async def get_monthly_production_statistics(
        self,
        yield_entity: str
    ) -> dict:
        """
        Get monthly production statistics.
        
        RECORDER REMOVED: Returns empty statistics.
        Use DataManager monthly yield tracking instead.
        
        Args:
            yield_entity: Solar yield entity (not used)
            
        Returns:
            Empty statistics dictionary
        """
        _LOGGER.debug("Monthly stats: Not available without Recorder")
        return {
            "total_kwh": 0.0,
            "average_daily_kwh": 0.0,
            "best_day_kwh": 0.0,
            "worst_day_kwh": 0.0,
            "days_with_data": 0
        }

    async def get_historical_average(self) -> Optional[float]:
        """
        Get historical average production.
        
        STUB: Not implemented - orchestrator calculates forecasts directly.
        
        Returns:
            None (use orchestrator.create_forecast() instead)
        """
        _LOGGER.debug("Historical average: Stub called, returning None (use orchestrator)")
        return None

    async def get_historical_average(self) -> Optional[float]:
        """
        Get historical average production.
        
        RECORDER REMOVED: Returns None (no historical data access).
        Use ML prediction records or DataManager instead.
        
        Returns:
            None (no historical access)
        """
        _LOGGER.debug("Historical average: Not available without Recorder")
        return None


# ============================================================================
# MIGRATION NOTES:
# ============================================================================
#
# WHAT WAS REMOVED:
# - _get_state_history() - Blocking Recorder DB access
# - calculate_peak_production_time() - 14-day history analysis
# - All HomeAssistant Recorder imports and dependencies
# - Executor jobs for synchronous history queries
#
# WHAT TO USE INSTEAD:
# - Peak Time: Use ML hourly samples or default 12:00
# - Historical Data: Use DataManager prediction_history.json
# - Monthly Stats: Use DataManager get_average_monthly_yield()
#
# BENEFITS:
# - Zero startup blocking
# - No database load
# - Faster, more reliable
# - Independent of Recorder configuration
#
# FUTURE ENHANCEMENTS:
# - Analyze hourly_samples.json for peak time patterns
# - Use weather_forecast_history.json for correlations
# - ML-based peak time prediction from training data
