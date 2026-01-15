# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

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
    """Weather Service using Open-Meteo as data source

    V12.3: All blending now goes through WeatherExpertBlender in the pipeline.
    This service only fetches raw Open-Meteo data for the ExpertBlender to use.
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

        _LOGGER.info(
            f"WeatherService initialized - Open-Meteo "
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
        """Get CORRECTED forecast data from weather_forecast_corrected.json

        This is the SINGLE SOURCE OF TRUTH for all forecast calculations.
        The corrected file contains precision-adjusted weather data that
        should be used for Today, Tomorrow, and Day After Tomorrow forecasts.

        Args:
            strict: If True, raise error if no data available

        Returns:
            List of hourly forecast entries from weather_forecast_corrected.json
        """
        import json

        corrected_file = self.data_dir / "stats" / "weather_forecast_corrected.json"

        try:
            if corrected_file.exists():
                def _load_corrected():
                    with open(corrected_file, "r") as f:
                        return json.load(f)

                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(None, _load_corrected)

                if data and "forecast" in data:
                    forecast_by_date = data.get("forecast", {})
                    result = []

                    # Convert dict structure to list format expected by callers
                    for date_str in sorted(forecast_by_date.keys()):
                        day_data = forecast_by_date[date_str]
                        for hour_str in sorted(day_data.keys(), key=int):
                            hour_data = day_data[hour_str]
                            hour = int(hour_str)

                            entry = {
                                "datetime": f"{date_str}T{hour:02d}:00:00",
                                "local_datetime": f"{date_str}T{hour:02d}:00:00",
                                "date": date_str,
                                "hour": hour,
                                "local_hour": hour,
                                "temperature": hour_data.get("temperature", DEFAULT_WEATHER_DATA["temperature"]),
                                "humidity": hour_data.get("humidity", DEFAULT_WEATHER_DATA["humidity"]),
                                "cloud_cover": hour_data.get("clouds", DEFAULT_WEATHER_DATA["cloud_cover"]),
                                "clouds": hour_data.get("clouds", DEFAULT_WEATHER_DATA["cloud_cover"]),
                                "wind_speed": hour_data.get("wind", DEFAULT_WEATHER_DATA["wind_speed"]),
                                "precipitation": hour_data.get("rain", DEFAULT_WEATHER_DATA["precipitation"]),
                                "rain": hour_data.get("rain", DEFAULT_WEATHER_DATA["precipitation"]),
                                "pressure": hour_data.get("pressure", DEFAULT_WEATHER_DATA["pressure"]),
                                "ghi": hour_data.get("solar_radiation_wm2", 0.0),
                                "solar_radiation": hour_data.get("solar_radiation_wm2", 0.0),
                                "solar_radiation_wm2": hour_data.get("solar_radiation_wm2", 0.0),
                                "direct_radiation": hour_data.get("direct_radiation", 0.0),
                                "diffuse_radiation": hour_data.get("diffuse_radiation", 0.0),
                                # V12.8.5: FOG detection parameters
                                "visibility_m": hour_data.get("visibility_m"),
                                "fog_detected": hour_data.get("fog_detected", False),
                                "fog_type": hour_data.get("fog_type"),
                                "_source": "weather_forecast_corrected",
                            }
                            result.append(entry)

                    if result:
                        # Count days for logging
                        dates = set(entry["date"] for entry in result)
                        _LOGGER.info(
                            f"✓ Forecast using weather_forecast_corrected.json: "
                            f"{len(result)} hours across {len(dates)} days "
                            f"({', '.join(sorted(dates)[-3:])})"
                        )
                        return result

            _LOGGER.warning(
                "⚠ weather_forecast_corrected.json not available, falling back to Open-Meteo"
            )

        except Exception as e:
            _LOGGER.warning(
                f"Failed to load weather_forecast_corrected.json: {e}, falling back to Open-Meteo"
            )

        # Fallback to Open-Meteo if corrected file not available
        forecast = await self.get_hourly_forecast()

        if not forecast and strict:
            raise FileNotFoundError("No forecast data available")

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

    async def force_update(self) -> bool:
        """Force immediate forecast update from Open-Meteo API.

        V12.3: Direct Open-Meteo fetch only. All blending happens in the pipeline
        through WeatherExpertBlender.
        """
        try:
            _LOGGER.info("Force update - fetching from Open-Meteo API...")
            forecast = await self._open_meteo.get_hourly_forecast(hours=72)

            if forecast:
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
        """Background task to periodically update forecast."""
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
