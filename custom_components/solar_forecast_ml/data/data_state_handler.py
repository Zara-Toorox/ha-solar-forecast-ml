"""
Data State Handler for Solar Forecast ML Integration

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
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from homeassistant.core import HomeAssistant
from ..core.core_exceptions import DataIntegrityException
from .data_manager import DataManagerIO

from .data_io import DataManagerIO
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..const import DATA_VERSION

_LOGGER = logging.getLogger(__name__)


class DataStateHandler(DataManagerIO):
    """Handles coordinator state and weather cache by Zara"""

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
            "last_collected_hour": None # New field
        }

    async def ensure_state_files(self) -> None:
        """Ensure state files exist by Zara"""
        if not self.coordinator_state_file.exists():
            await self._atomic_write_json(
                self.coordinator_state_file,
                self._coordinator_state_default
            )

    # =============================================================
    # COORDINATOR STATE Methods (Expected Daily Production)
    # =============================================================

    async def save_expected_daily_production(self, value: float) -> bool:
        """Save expected daily production value persistently by Zara"""
        try:
            now_local = dt_util.now()
            state = {
                "version": DATA_VERSION,
                "expected_daily_production": value,
                "last_set_date": now_local.date().isoformat(),
                "last_updated": now_local.isoformat()
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
        daily_forecasts_data: Optional[dict[str, Any]] = None
    ) -> Optional[float]:
        """Load expected daily production from persistent storage by Zara"""
        try:
            # TRY NEW SYSTEM FIRST (daily_forecasts.json)
            if check_daily_forecasts and daily_forecasts_data:
                if "today" in daily_forecasts_data and daily_forecasts_data["today"].get("forecast_day", {}).get("locked"):
                    value = daily_forecasts_data["today"]["forecast_day"].get("prediction_kwh")
                    if value is not None:
                        _LOGGER.debug(
                            f"Loaded expected daily production from daily_forecasts.json: "
                            f"{value:.2f} kWh (source: {daily_forecasts_data['today']['forecast_day'].get('source')})"
                        )
                        return float(value)
            
            # FALLBACK TO OLD SYSTEM (coordinator_state.json)
            state = await self._read_json_file(
                self.coordinator_state_file,
                self._coordinator_state_default
            )
            
            if not state:
                return None
            
            # Check if value is from today
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

            return False

    async def get_last_collected_hour(self) -> Optional[datetime]:
        """Get the timestamp of the last collected hourly sample by Zara"""
        try:
            state = await self._read_json_file(
                self.coordinator_state_file,
                self._coordinator_state_default
            )
            last_collected_hour_str = state.get("last_collected_hour")
            if last_collected_hour_str:
                return dt_util.parse_datetime(last_collected_hour_str)
            return None
        except Exception as e:
            _LOGGER.error(f"Failed to get last collected hour: {e}")
            return None

    async def set_last_collected_hour(self, timestamp: datetime) -> bool:
        """Set the timestamp of the last collected hourly sample by Zara"""
        try:
            state = await self._read_json_file(
                self.coordinator_state_file,
                self._coordinator_state_default
            )
            state["last_collected_hour"] = timestamp.isoformat()
            state["last_updated"] = dt_util.now().isoformat()
            await self._atomic_write_json(self.coordinator_state_file, state)
            _LOGGER.debug(f"Last collected hour set to: {timestamp.isoformat()}")
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to set last collected hour: {e}")
            return False    # =============================================================
    # WEATHER CACHE Methods
    # =============================================================

    async def save_weather_cache(self, weather_data: dict[str, Any]) -> bool:
        """Save weather forecast data to cache by Zara"""
        try:
            now_local = dt_util.now()
            
            # Ensure new format with metadata
            if not isinstance(weather_data, dict) or "forecast_hours" not in weather_data:
                _LOGGER.warning("Weather data not in expected format, converting...")
                weather_data = {
                    "forecast_hours": weather_data if isinstance(weather_data, list) else [],
                    "cached_at": now_local.isoformat(),
                    "data_quality": {
                        "today_hours": 0,
                        "tomorrow_hours": 0,
                        "total_hours": len(weather_data) if isinstance(weather_data, list) else 0
                    }
                }
            
            # Update cached_at timestamp
            weather_data["cached_at"] = now_local.isoformat()
            
            await self._ensure_directory_exists(self.weather_cache_file.parent)
            await self._atomic_write_json(self.weather_cache_file, weather_data)
            
            total_hours = len(weather_data.get("forecast_hours", []))
            _LOGGER.debug(f"Weather cache saved ({total_hours} hours)")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Failed to save weather cache: {e}")
            return False

    async def load_weather_cache(self) -> Optional[dict[str, Any]]:
        """Load weather forecast data from cache by Zara"""
        try:
            if self.weather_cache_file.exists():
                data = await self._read_json_file(self.weather_cache_file, None)
                
                if data is None:
                    return None
                
                # Backward compatibility: If data is a list (old format), wrap it
                if isinstance(data, list):
                    _LOGGER.info("Converting old weather cache format (list) to new format (dict)")
                    return {
                        "forecast_hours": data,
                        "cached_at": dt_util.now().isoformat(),
                        "data_quality": {
                            "today_hours": 0,
                            "tomorrow_hours": 0,
                            "total_hours": len(data)
                        },
                        "converted_from_old_format": True
                    }
                
                # New format: Already a dict
                return data
            return None
        except Exception as e:
            _LOGGER.error(f"Failed to load weather cache: {e}")
            return None

    async def clear_weather_cache(self) -> bool:
        """Clear weather cache by Zara"""
        try:
            if self.weather_cache_file.exists():
                self.weather_cache_file.unlink()
                _LOGGER.debug("Weather cache cleared")
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to clear weather cache: {e}")
            return False

    async def get_weather_cache_age(self) -> Optional[int]:
        """Get age of weather cache in minutes by Zara"""
        try:
            data = await self.load_weather_cache()
            if not data or "cached_at" not in data:
                return None
            
            cached_at = dt_util.parse_datetime(data["cached_at"])
            if cached_at:
                now_local = dt_util.now()
                age_seconds = (now_local - cached_at).total_seconds()
                return int(age_seconds / 60)
            
            return None
            
        except Exception as e:
            _LOGGER.error(f"Failed to get weather cache age: {e}")
            return None

    async def is_weather_cache_valid(self, max_age_minutes: int = 60) -> bool:
        """Check if weather cache is still valid not too old by Zara"""
        try:
            age = await self.get_weather_cache_age()
            if age is None:
                return False
            
            return age <= max_age_minutes
            
        except Exception:
            return False
