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
from datetime import datetime, timedelta
from homeassistant.core import HomeAssistant
from ..core.core_exceptions import DataIntegrityException
from .data_manager import DataManagerIO

from .data_io import DataManagerIO
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..const import DATA_VERSION

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
            "last_collected_hour": None # New field
        }

    async def ensure_state_files(self) -> None:
        """Ensure state files exist"""
        if not self.coordinator_state_file.exists():
            await self._atomic_write_json(
                self.coordinator_state_file,
                self._coordinator_state_default
            )

    # =============================================================
    # COORDINATOR STATE Methods (Expected Daily Production)
    # =============================================================

    async def save_expected_daily_production(self, value: float) -> bool:
        """Save expected daily production value persistently"""
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
        """Load expected daily production from persistent storage"""
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

    async def clear_expected_daily_production(self) -> bool:
        """Clear expected daily production from persistent storage"""
        try:
            state = {
                "version": DATA_VERSION,
                "expected_daily_production": None,
                "last_set_date": None,
                "last_updated": dt_util.now().isoformat()
            }

            await self._atomic_write_json(self.coordinator_state_file, state)
            _LOGGER.debug("Expected daily production cleared from persistent storage")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to clear expected daily production: {e}")
            return False

    async def get_last_collected_hour(self) -> Optional[datetime]:
        """Get the timestamp of the last collected hourly sample"""
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
        """Set the timestamp of the last collected hourly sample"""
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
        """Save weather forecast data to cache with 7-day retention (Rolling Window)

        Strategy:
        1. Load existing cache (if any)
        2. Merge new forecast_hours with existing data
        3. Remove duplicates (by datetime)
        4. Remove entries older than 7 days
        5. Save merged cache
        """
        try:
            now_local = dt_util.now()
            retention_days = 7
            cutoff_dt = now_local - timedelta(days=retention_days)

            # Ensure new format with metadata
            if not isinstance(weather_data, dict) or "forecast_hours" not in weather_data:
                _LOGGER.warning("Weather data not in expected format, converting...")
                weather_data = {
                    "forecast_hours": weather_data if isinstance(weather_data, list) else [],
                    "cached_at": now_local.isoformat(),
                }

            new_forecast_hours = weather_data.get("forecast_hours", [])

            # Load existing cache to merge
            existing_cache = await self.load_weather_cache()
            existing_hours = []
            if existing_cache and isinstance(existing_cache, dict):
                existing_hours = existing_cache.get("forecast_hours", [])

            # Merge: existing + new, deduplicate by datetime
            all_hours = existing_hours + new_forecast_hours

            # Deduplicate and filter old entries
            unique_hours = {}
            for hour_entry in all_hours:
                dt_str = hour_entry.get("datetime")
                if not dt_str:
                    continue

                # Parse datetime to check age
                try:
                    dt_obj = dt_util.parse_datetime(dt_str)
                    if not dt_obj:
                        continue

                    # Skip entries older than retention period
                    if dt_obj < cutoff_dt:
                        continue

                    # Keep newest entry for each datetime (new data overwrites old)
                    unique_hours[dt_str] = hour_entry

                except Exception as e:
                    _LOGGER.debug(f"Failed to parse datetime '{dt_str}': {e}")
                    continue

            # Sort by datetime (chronological order)
            sorted_hours = sorted(unique_hours.values(), key=lambda x: x.get("datetime", ""))

            # Calculate statistics
            today_str = now_local.date().isoformat()
            tomorrow_str = (now_local + timedelta(days=1)).date().isoformat()

            today_count = sum(1 for h in sorted_hours if h.get("local_datetime", "").startswith(today_str))
            tomorrow_count = sum(1 for h in sorted_hours if h.get("local_datetime", "").startswith(tomorrow_str))

            # Build final cache structure
            merged_cache = {
                "version": "2.0",
                "cached_at": now_local.isoformat(),
                "retention_days": retention_days,
                "forecast_hours": sorted_hours,
                "metadata": {
                    "total_hours": len(sorted_hours),
                    "today_hours": today_count,
                    "tomorrow_hours": tomorrow_count,
                    "oldest_entry": sorted_hours[0].get("datetime") if sorted_hours else None,
                    "newest_entry": sorted_hours[-1].get("datetime") if sorted_hours else None,
                }
            }

            await self._ensure_directory_exists(self.weather_cache_file.parent)
            await self._atomic_write_json(self.weather_cache_file, merged_cache)

            _LOGGER.debug(
                f"Weather cache saved: {len(sorted_hours)} hours total "
                f"(today: {today_count}, tomorrow: {tomorrow_count}, retention: {retention_days}d)"
            )
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to save weather cache: {e}", exc_info=True)
            return False

    async def load_weather_cache(self) -> Optional[dict[str, Any]]:
        """Load weather forecast data from cache"""
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
                
                return data
            return None
        except Exception as e:
            _LOGGER.error(f"Failed to load weather cache: {e}")
            return None

    async def clear_weather_cache(self) -> bool:
        """Clear weather cache"""
        try:
            if self.weather_cache_file.exists():
                self.weather_cache_file.unlink()
                _LOGGER.debug("Weather cache cleared")
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to clear weather cache: {e}")
            return False

    async def get_weather_cache_age(self) -> Optional[int]:
        """Get age of weather cache in minutes"""
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

    async def is_weather_cache_valid(self, max_age_minutes: int = 180) -> bool:
        """Check if weather cache is still valid not too old Default: 180 minutes (3 hours) - Weather forecasts don't change significantly in this timeframe, and this provides better resilience against API failures."""
        try:
            age = await self.get_weather_cache_age()
            if age is None:
                return False

            return age <= max_age_minutes

        except Exception:
            return False
