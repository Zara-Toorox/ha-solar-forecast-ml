"""Weather Data Provider - Open-Meteo ONLY V12.2.0 @zara

IMPORTANT: This module now uses Open-Meteo as the ONLY data source.
- NO HA Weather Entity dependency for forecasts
- All weather data comes from Open-Meteo API
- GHI, cloud_cover, temperature etc. are used DIRECTLY

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

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..data.data_open_meteo_client import OpenMeteoClient

_LOGGER = logging.getLogger(__name__)

DEFAULT_WEATHER_DATA = {
    "temperature": 15.0,
    "humidity": 60.0,
    "cloud_cover": 50.0,
    "wind_speed": 3.0,
    "precipitation": 0.0,
    "pressure": 1013.25,
    "ghi": 0.0,
    "solar_radiation_wm2": 0.0,  # Alias for ghi - consistent naming
    "direct_radiation": 0.0,
    "diffuse_radiation": 0.0,
}


class WeatherService:
    """Weather Service using Open-Meteo as SINGLE data source

    IMPORTANT: All cache updates should go through the MultiWeatherBlender
    when available to ensure blend_info is preserved for weight learning.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        latitude: float,
        longitude: float,
        data_dir: Path,
        data_manager=None,
        error_handler=None
    ):
        """Initialize weather service with Open-Meteo"""
        self.hass = hass
        self.latitude = latitude
        self.longitude = longitude
        self.data_dir = data_dir
        self.data_manager = data_manager
        self.error_handler = error_handler

        cache_file = data_dir / "data" / "open_meteo_cache.json"
        self._open_meteo = OpenMeteoClient(latitude, longitude, cache_file)

        self._cached_forecast: List[Dict[str, Any]] = []
        self._background_update_task: Optional[asyncio.Task] = None

        # Reference to MultiWeatherBlender for proper cache updates
        # Set by WeatherDataPipelineManager after initialization
        self._multi_weather_blender = None

        _LOGGER.info(
            f"WeatherService initialized - Open-Meteo ONLY "
            f"(lat={latitude:.4f}, lon={longitude:.4f})"
        )

    async def initialize(self) -> bool:
        """Async initialization - loads Open-Meteo cache"""
        try:
            await self._open_meteo.async_init()

            forecast = await self._open_meteo.get_hourly_forecast(hours=72)
            if forecast:
                self._cached_forecast = self._transform_open_meteo_forecast(forecast)
                _LOGGER.info(
                    f"Loaded {len(self._cached_forecast)} hours from Open-Meteo"
                )
            else:
                _LOGGER.warning("No Open-Meteo data available yet")

            self._background_update_task = asyncio.create_task(
                self._background_forecast_update()
            )

            _LOGGER.info("Weather Service initialized (Open-Meteo ONLY)")
            return True

        except Exception as e:
            _LOGGER.error(f"Weather Service initialization failed: {e}", exc_info=True)
            return False

    def _transform_open_meteo_forecast(
        self, open_meteo_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Transform Open-Meteo data to internal format"""
        transformed = []

        for entry in open_meteo_data:
            dt_obj = entry.get("datetime")
            if isinstance(dt_obj, datetime):
                dt_str = dt_obj.isoformat()
                date_str = dt_obj.date().isoformat()
                hour = dt_obj.hour
            else:
                date_str = entry.get("date", "")
                hour = entry.get("hour", 0)
                dt_str = f"{date_str}T{hour:02d}:00:00"

            transformed_entry = {
                "datetime": dt_str,
                "local_datetime": dt_str,
                "date": date_str,
                "hour": hour,
                "local_hour": hour,
                "temperature": entry.get("temperature", DEFAULT_WEATHER_DATA["temperature"]),
                "humidity": entry.get("humidity", DEFAULT_WEATHER_DATA["humidity"]),
                "cloud_cover": entry.get("cloud_cover", DEFAULT_WEATHER_DATA["cloud_cover"]),
                "wind_speed": entry.get("wind_speed", DEFAULT_WEATHER_DATA["wind_speed"]),
                "precipitation": entry.get("precipitation", DEFAULT_WEATHER_DATA["precipitation"]),
                "pressure": entry.get("pressure", DEFAULT_WEATHER_DATA["pressure"]),
                "ghi": entry.get("ghi", 0.0),
                "solar_radiation_wm2": entry.get("ghi", 0.0),  # Alias for ghi - consistent naming
                "direct_radiation": entry.get("direct_radiation", 0.0),
                "diffuse_radiation": entry.get("diffuse_radiation", 0.0),
                "global_tilted_irradiance": entry.get("global_tilted_irradiance"),
                "_source": "open-meteo",
            }
            transformed.append(transformed_entry)

        return transformed

    async def get_hourly_forecast(
        self, force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """Get hourly forecast from Open-Meteo

        Args:
            force_refresh: If True, fetch fresh data from API

        Returns:
            List of hourly forecast entries
        """
        if force_refresh:
            _LOGGER.info("Force refresh requested - fetching from Open-Meteo API")
            forecast = await self._open_meteo.get_hourly_forecast(hours=72)
            if forecast:
                self._cached_forecast = self._transform_open_meteo_forecast(forecast)
                _LOGGER.info(f"Fetched {len(self._cached_forecast)} hours from Open-Meteo")
                return self._cached_forecast

        if self._cached_forecast:
            _LOGGER.debug(f"Using cached forecast: {len(self._cached_forecast)} hours")
            return self._cached_forecast

        forecast = await self._open_meteo.get_hourly_forecast(hours=72)
        if forecast:
            self._cached_forecast = self._transform_open_meteo_forecast(forecast)
            return self._cached_forecast

        _LOGGER.warning("No forecast data available from Open-Meteo")
        return []

    async def get_processed_hourly_forecast(
        self, force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """Get processed hourly forecast (alias for get_hourly_forecast)"""
        return await self.get_hourly_forecast(force_refresh)

    async def get_corrected_hourly_forecast(
        self, strict: bool = False
    ) -> List[Dict[str, Any]]:
        """Get forecast data - Open-Meteo data is already high quality

        Note: With Open-Meteo as single source, no "corrections" are needed.
        This method exists for backwards compatibility.

        Args:
            strict: If True, raise error if no data available

        Returns:
            List of hourly forecast entries
        """
        forecast = await self.get_hourly_forecast()

        if not forecast and strict:
            raise FileNotFoundError("No Open-Meteo forecast data available")

        return forecast

    async def get_current_weather(self) -> Dict[str, Any]:
        """Get current weather from Open-Meteo cache"""
        now = dt_util.now()
        current_hour = now.hour
        current_date = now.date().isoformat()

        entry = self._open_meteo.get_weather_for_hour(current_date, current_hour)

        if entry:
            return {
                "temperature": entry.get("temperature", DEFAULT_WEATHER_DATA["temperature"]),
                "humidity": entry.get("humidity", DEFAULT_WEATHER_DATA["humidity"]),
                "cloud_cover": entry.get("cloud_cover", DEFAULT_WEATHER_DATA["cloud_cover"]),
                "wind_speed": entry.get("wind_speed", DEFAULT_WEATHER_DATA["wind_speed"]),
                "precipitation": entry.get("precipitation", DEFAULT_WEATHER_DATA["precipitation"]),
                "pressure": entry.get("pressure", DEFAULT_WEATHER_DATA["pressure"]),
                "ghi": entry.get("ghi", 0.0),
                "solar_radiation_wm2": entry.get("ghi", 0.0),  # Alias for ghi - consistent naming
                "direct_radiation": entry.get("direct_radiation", 0.0),
                "diffuse_radiation": entry.get("diffuse_radiation", 0.0),
                "_source": "open-meteo",
            }

        _LOGGER.debug("No current hour data, using defaults")
        return DEFAULT_WEATHER_DATA.copy()

    def get_weather_for_hour(self, date: str, hour: int) -> Optional[Dict[str, Any]]:
        """Get weather data for a specific hour"""
        return self._open_meteo.get_weather_for_hour(date, hour)

    def get_radiation_for_hour(self, date: str, hour: int) -> tuple:
        """Get radiation values for a specific hour

        Returns:
            Tuple of (direct_radiation, diffuse_radiation, ghi)
        """
        return self._open_meteo.get_radiation_for_hour(date, hour)

    def get_forecast_for_date(self, date: str) -> List[Dict[str, Any]]:
        """Get all forecast entries for a specific date"""
        return self._open_meteo.get_forecast_for_date(date)

    def set_multi_weather_blender(self, blender) -> None:
        """Set reference to MultiWeatherBlender for proper cache updates.

        Args:
            blender: MultiWeatherBlender instance
        """
        self._multi_weather_blender = blender
        _LOGGER.debug("MultiWeatherBlender reference set in WeatherService")

    async def force_update(self) -> bool:
        """Force immediate forecast update - uses MultiWeatherBlender if available.

        IMPORTANT: When MultiWeatherBlender is available, all updates go through it
        to ensure blend_info is preserved for weight learning. This prevents the
        cache from being overwritten with data lacking blend_info.
        """
        try:
            # Prefer MultiWeatherBlender for cache updates (preserves blend_info)
            if self._multi_weather_blender:
                _LOGGER.info("Force update - using MultiWeatherBlender for proper blend_info...")
                success = await self._multi_weather_blender.update_and_save_cache()

                if success:
                    # Refresh internal cache from the blended data
                    forecast = await self._open_meteo.get_hourly_forecast(hours=72)
                    if forecast:
                        self._cached_forecast = self._transform_open_meteo_forecast(forecast)
                    _LOGGER.info(
                        f"Force update via Blender successful: {len(self._cached_forecast)} hours"
                    )
                    return True
                else:
                    _LOGGER.warning("Blender update failed - falling back to direct Open-Meteo")
                    # Fall through to direct update

            # Fallback: Direct Open-Meteo update
            # Note: Even without blending, data will have blend_info from _parse_hourly_response
            _LOGGER.info("Force update - fetching directly from Open-Meteo API...")
            forecast = await self._open_meteo.get_hourly_forecast(hours=72)

            if forecast:
                # Ensure cache is saved (auto_save might be disabled by Blender)
                # This ensures blend_info is persisted even in fallback mode
                await self._open_meteo._save_file_cache(forecast)

                self._cached_forecast = self._transform_open_meteo_forecast(forecast)
                _LOGGER.info(f"Force update successful: {len(self._cached_forecast)} hours")
                return True
            else:
                _LOGGER.warning("Force update returned no data")
                return False

        except Exception as e:
            _LOGGER.error(f"Force update failed: {e}", exc_info=True)
            return False

    async def _background_forecast_update(self):
        """Background task to periodically update forecast.

        Uses MultiWeatherBlender when available to preserve blend_info.
        """
        update_interval = 3600

        while True:
            try:
                await asyncio.sleep(update_interval)

                _LOGGER.debug("Background forecast update starting...")

                # Use force_update which routes through Blender when available
                success = await self.force_update()

                if success:
                    _LOGGER.debug(
                        f"Background update complete: {len(self._cached_forecast)} hours"
                    )
                else:
                    _LOGGER.debug("Background update: No new data")

            except asyncio.CancelledError:
                _LOGGER.info("Background forecast update task cancelled")
                break
            except Exception as e:
                _LOGGER.error(f"Background forecast update error: {e}", exc_info=True)

    def get_health_status(self) -> Dict[str, Any]:
        """Check health status of the weather service"""
        cache_info = self._open_meteo._get_cache_source_info()

        has_data = bool(self._cached_forecast)

        return {
            "healthy": has_data,
            "status": "ok" if has_data else "no_data",
            "message": f"Open-Meteo: {cache_info['source']}",
            "source": "open-meteo",
            "cached_hours": len(self._cached_forecast),
            "cache_confidence": cache_info.get("confidence", 0),
            "cache_age_hours": cache_info.get("age_hours"),
        }

    async def cleanup(self):
        """Cleanup resources"""
        if self._background_update_task and not self._background_update_task.done():
            self._background_update_task.cancel()
            try:
                await self._background_update_task
            except asyncio.CancelledError:
                pass

        _LOGGER.debug("Weather Service cleanup complete")
