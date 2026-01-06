# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from homeassistant.core import HomeAssistant

from ..const import DATA_VERSION
from ..core.core_exceptions import DataIntegrityException
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from .data_io import DataManagerIO
from .data_manager import DataManagerIO

_LOGGER = logging.getLogger(__name__)

class DataStateHandler(DataManagerIO):
    """Handles coordinator state and production time state"""

    def __init__(self, hass: HomeAssistant, data_dir: Path):
        super().__init__(hass, data_dir)

        self.coordinator_state_file = self.data_dir / "data" / "coordinator_state.json"
        self.production_time_state_file = self.data_dir / "data" / "production_time_state.json"

        self._coordinator_state_default = {
            "version": DATA_VERSION,
            "expected_daily_production": None,
            "last_set_date": None,
            "last_updated": None,
            "last_collected_hour": None,
        }

    async def save_expected_daily_production(self, value: float) -> bool:
        """Save expected daily production value persistently @zara"""
        try:
            now_local = dt_util.now()
            state = {
                "version": DATA_VERSION,
                "expected_daily_production": value,
                "last_set_date": now_local.date().isoformat(),
                "last_updated": now_local.isoformat(),
            }

            await self._atomic_write_json(self.coordinator_state_file, state)
            _LOGGER.debug(f"Expected daily production saved: {value:.2f} kWh")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to save expected daily production: {e}")
            return False

    async def load_expected_daily_production(
        self,
        check_daily_forecasts: bool = True,
        daily_forecasts_data: Optional[dict[str, Any]] = None,
    ) -> Optional[float]:
        """Load expected daily production from daily_forecasts.json @zara"""
        try:
            if check_daily_forecasts and daily_forecasts_data:
                if "today" in daily_forecasts_data and daily_forecasts_data["today"].get(
                    "forecast_day", {}
                ).get("locked"):
                    forecast_day = daily_forecasts_data["today"]["forecast_day"]
                    value = forecast_day.get("prediction_kwh")
                    if value is not None:
                        source_info = forecast_day.get('source')
                        _LOGGER.debug(
                            f"Loaded expected daily production from daily_forecasts.json: "
                            f"{value:.2f} kWh (source: {source_info})"
                        )
                        return float(value)

            return None

        except Exception as e:
            _LOGGER.error(f"Failed to load expected daily production: {e}")
            return None

    async def clear_expected_daily_production(self) -> bool:
        """Clear expected daily production from persistent storage @zara"""
        try:
            state = {
                "version": DATA_VERSION,
                "expected_daily_production": None,
                "last_set_date": None,
                "last_updated": dt_util.now().isoformat(),
            }

            await self._atomic_write_json(self.coordinator_state_file, state)
            _LOGGER.debug("Expected daily production cleared from persistent storage")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to clear expected daily production: {e}")
            return False

    async def get_last_collected_hour(self) -> Optional[datetime]:
        """Get the timestamp of the last collected hourly sample @zara"""
        try:
            state = await self._read_json_file(
                self.coordinator_state_file, self._coordinator_state_default
            )
            last_collected_hour_str = state.get("last_collected_hour")
            if last_collected_hour_str:
                return dt_util.parse_datetime(last_collected_hour_str)
            return None
        except Exception as e:
            _LOGGER.error(f"Failed to get last collected hour: {e}")
            return None

    async def set_last_collected_hour(self, timestamp: datetime) -> bool:
        """Set the timestamp of the last collected hourly sample @zara"""
        try:
            state = await self._read_json_file(
                self.coordinator_state_file, self._coordinator_state_default
            )
            state["last_collected_hour"] = timestamp.isoformat()
            state["last_updated"] = dt_util.now().isoformat()
            await self._atomic_write_json(self.coordinator_state_file, state)
            _LOGGER.debug(f"Last collected hour set to: {timestamp.isoformat()}")
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to set last collected hour: {e}")
            return False

