"""Data State Handler for Solar Forecast ML Integration V12.2.0 @zara

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
    """Handles coordinator state and weather cache"""

    def __init__(self, hass: HomeAssistant, data_dir: Path):
        super().__init__(hass, data_dir)

        self.coordinator_state_file = self.data_dir / "data" / "coordinator_state.json"
        self.weather_cache_file = self.data_dir / "data" / "weather_cache.json"
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
        """Load expected daily production from persistent storage

        Priority:
        1. prediction_kwh from daily_forecasts.json (locked forecast)
        2. coordinator_state.json (legacy fallback)
        """
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

            state = await self._read_json_file(
                self.coordinator_state_file, self._coordinator_state_default
            )

            if not state:
                return None

            now_local = dt_util.now()
            today_str = now_local.date().isoformat()
            last_set_date = state.get("last_set_date")

            if last_set_date != today_str:
                _LOGGER.debug(
                    f"Expected daily production expired "
                    f"(from {last_set_date}, today is {today_str})"
                )
                return None

            value = state.get("expected_daily_production")
            if value is not None:
                _LOGGER.debug(
                    f"Loaded expected daily production from coordinator_state.json (OLD): "
                    f"{value:.2f} kWh"
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

    async def save_weather_cache(self, weather_data: dict[str, Any]) -> bool:
        """Legacy method - Open-Meteo cache is the single data source. @zara"""
        return True

    async def load_weather_cache(self) -> Optional[dict[str, Any]]:
        """Legacy method - Open-Meteo cache is the single data source. @zara"""
        return None

    async def clear_weather_cache(self) -> bool:
        """Legacy method - Open-Meteo cache is the single data source. @zara"""
        return True

    async def get_weather_cache_age(self) -> Optional[int]:
        """Legacy method - Open-Meteo cache is the single data source. @zara"""
        return None

    async def is_weather_cache_valid(self, max_age_minutes: int = 180) -> bool:
        """Legacy method - Open-Meteo cache is the single data source. @zara"""
        return False
