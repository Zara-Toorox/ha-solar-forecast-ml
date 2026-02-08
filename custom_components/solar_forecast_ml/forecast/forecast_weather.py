# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Weather Service Module.
Provides weather data access via Open-Meteo with database storage.
@zara
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..data.db_manager import DatabaseManager
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
    "solar_radiation_wm2": 0.0,
    "direct_radiation": 0.0,
    "diffuse_radiation": 0.0,
}


class WeatherService:
    """Weather Service using Open-Meteo as data source with database storage. @zara"""

    def __init__(
        self,
        hass: HomeAssistant,
        latitude: float,
        longitude: float,
        data_dir: Path,
        db_manager: DatabaseManager,
        data_manager: Optional[Any] = None,
        error_handler: Optional[Any] = None
    ):
        """Initialize weather service with Open-Meteo. @zara

        Args:
            hass: Home Assistant instance
            latitude: Geographic latitude
            longitude: Geographic longitude
            data_dir: Data directory path
            db_manager: Database manager instance
            data_manager: Optional data manager
            error_handler: Optional error handler
        """
        self.hass = hass
        self.latitude = latitude
        self.longitude = longitude
        self.data_dir = data_dir
        self._db = db_manager
        self.data_manager = data_manager
        self.error_handler = error_handler

        self._open_meteo = OpenMeteoClient(hass, db_manager, latitude, longitude)

        self._cached_forecast: List[Dict[str, Any]] = []
        self._background_update_task: Optional[asyncio.Task] = None

        _LOGGER.info(
            f"WeatherService initialized - Open-Meteo "
            f"(lat={latitude:.4f}, lon={longitude:.4f})"
        )

    async def initialize(self) -> bool:
        """Async initialization - loads Open-Meteo cache. @zara"""
        try:
            await self._open_meteo.async_init()

            forecast = await self._open_meteo.get_hourly_forecast(hours=72)
            if forecast:
                self._cached_forecast = self._transform_open_meteo_forecast(forecast)
                await self._store_forecast_in_db(self._cached_forecast)
                _LOGGER.info(
                    f"Loaded {len(self._cached_forecast)} hours from Open-Meteo"
                )
            else:
                _LOGGER.warning("No Open-Meteo data available yet")

            self._background_update_task = self.hass.async_create_background_task(
                self._background_forecast_update(),
                name="solar_forecast_ml_weather_update",
            )

            _LOGGER.info("Weather Service initialized (Open-Meteo with DB storage)")
            return True

        except Exception as e:
            _LOGGER.error(f"Weather Service initialization failed: {e}", exc_info=True)
            return False

    async def _store_forecast_in_db(self, forecast: List[Dict[str, Any]]) -> None:
        """Store forecast data in database. @zara"""
        try:
            for entry in forecast:
                date_str = entry.get("date")
                hour = entry.get("hour", 0)

                if not date_str:
                    continue

                await self._db.execute(
                    """INSERT INTO weather_forecast
                       (forecast_date, hour, temperature, solar_radiation_wm2, wind, humidity,
                        rain, clouds, pressure, direct_radiation, diffuse_radiation,
                        visibility_m, fog_detected, fog_type, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(forecast_date, hour) DO UPDATE SET
                           temperature = excluded.temperature,
                           solar_radiation_wm2 = excluded.solar_radiation_wm2,
                           wind = excluded.wind,
                           humidity = excluded.humidity,
                           rain = excluded.rain,
                           clouds = excluded.clouds,
                           pressure = excluded.pressure,
                           direct_radiation = excluded.direct_radiation,
                           diffuse_radiation = excluded.diffuse_radiation,
                           visibility_m = excluded.visibility_m,
                           fog_detected = excluded.fog_detected,
                           fog_type = excluded.fog_type,
                           updated_at = excluded.updated_at""",
                    (
                        date_str,
                        hour,
                        entry.get("temperature"),
                        entry.get("solar_radiation_wm2", entry.get("ghi")),
                        entry.get("wind_speed"),
                        entry.get("humidity"),
                        entry.get("precipitation"),
                        entry.get("cloud_cover"),
                        entry.get("pressure"),
                        entry.get("direct_radiation"),
                        entry.get("diffuse_radiation"),
                        entry.get("visibility_m"),
                        entry.get("fog_detected"),
                        entry.get("fog_type"),
                        datetime.now(),
                    )
                )
        except Exception as e:
            _LOGGER.debug(f"Could not store forecast in DB: {e}")

    def _transform_open_meteo_forecast(
        self, open_meteo_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Transform Open-Meteo data to internal format. @zara"""
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
                "solar_radiation_wm2": entry.get("ghi", 0.0),
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
        """Get hourly forecast from Open-Meteo. @zara

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
                await self._store_forecast_in_db(self._cached_forecast)
                _LOGGER.info(f"Fetched {len(self._cached_forecast)} hours from Open-Meteo")
                return self._cached_forecast

        if self._cached_forecast:
            _LOGGER.debug(f"Using cached forecast: {len(self._cached_forecast)} hours")
            return self._cached_forecast

        forecast = await self._open_meteo.get_hourly_forecast(hours=72)
        if forecast:
            self._cached_forecast = self._transform_open_meteo_forecast(forecast)
            await self._store_forecast_in_db(self._cached_forecast)
            return self._cached_forecast

        _LOGGER.warning("No forecast data available from Open-Meteo")
        return []

    async def get_processed_hourly_forecast(
        self, force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """Get processed hourly forecast (alias for get_hourly_forecast). @zara"""
        return await self.get_hourly_forecast(force_refresh)

    async def get_corrected_hourly_forecast(
        self, strict: bool = False
    ) -> List[Dict[str, Any]]:
        """Get corrected forecast data from database. @zara

        Args:
            strict: If True, raise error if no data available

        Returns:
            List of hourly forecast entries
        """
        try:
            # Try to get corrected forecast from DB
            rows = await self._db.fetchall(
                """SELECT forecast_date, hour, temperature, solar_radiation_wm2,
                          wind, humidity, rain, clouds, pressure, direct_radiation,
                          diffuse_radiation, visibility_m, fog_detected, fog_type
                   FROM weather_forecast
                   WHERE forecast_date >= DATE('now')
                   ORDER BY forecast_date, hour"""
            )

            if rows:
                result = []
                for row in rows:
                    date_str = str(row[0])
                    hour = row[1]
                    entry = {
                        "datetime": f"{date_str}T{hour:02d}:00:00",
                        "local_datetime": f"{date_str}T{hour:02d}:00:00",
                        "date": date_str,
                        "hour": hour,
                        "local_hour": hour,
                        "temperature": row[2] or DEFAULT_WEATHER_DATA["temperature"],
                        "humidity": row[5] or DEFAULT_WEATHER_DATA["humidity"],
                        "cloud_cover": row[7] or DEFAULT_WEATHER_DATA["cloud_cover"],
                        "clouds": row[7] or DEFAULT_WEATHER_DATA["cloud_cover"],
                        "wind_speed": row[4] or DEFAULT_WEATHER_DATA["wind_speed"],
                        "precipitation": row[6] or DEFAULT_WEATHER_DATA["precipitation"],
                        "rain": row[6] or DEFAULT_WEATHER_DATA["precipitation"],
                        "pressure": row[8] or DEFAULT_WEATHER_DATA["pressure"],
                        "ghi": row[3] or 0.0,
                        "solar_radiation": row[3] or 0.0,
                        "solar_radiation_wm2": row[3] or 0.0,
                        "direct_radiation": row[9] or 0.0,
                        "diffuse_radiation": row[10] or 0.0,
                        "visibility_m": row[11],
                        "fog_detected": row[12],
                        "fog_type": row[13],
                        "_source": "database",
                    }
                    result.append(entry)

                if result:
                    dates = set(entry["date"] for entry in result)
                    _LOGGER.info(
                        f"Forecast using database: "
                        f"{len(result)} hours across {len(dates)} days"
                    )
                    return result

            _LOGGER.warning("No weather forecast data in DB, falling back to Open-Meteo")

        except Exception as e:
            _LOGGER.warning(f"Failed to load forecast from DB: {e}, falling back to Open-Meteo")

        forecast = await self.get_hourly_forecast()

        if not forecast and strict:
            raise FileNotFoundError("No forecast data available")

        return forecast

    async def get_current_weather(self) -> Dict[str, Any]:
        """Get current weather from Open-Meteo cache. @zara"""
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
                "solar_radiation_wm2": entry.get("ghi", 0.0),
                "direct_radiation": entry.get("direct_radiation", 0.0),
                "diffuse_radiation": entry.get("diffuse_radiation", 0.0),
                "_source": "open-meteo",
            }

        _LOGGER.debug("No current hour data, using defaults")
        return DEFAULT_WEATHER_DATA.copy()

    def get_weather_for_hour(self, date: str, hour: int) -> Optional[Dict[str, Any]]:
        """Get weather data for a specific hour. @zara"""
        return self._open_meteo.get_weather_for_hour(date, hour)

    def get_radiation_for_hour(self, date: str, hour: int) -> tuple:
        """Get radiation values for a specific hour. @zara

        Returns:
            Tuple of (direct_radiation, diffuse_radiation, ghi)
        """
        return self._open_meteo.get_radiation_for_hour(date, hour)

    def get_forecast_for_date(self, date: str) -> List[Dict[str, Any]]:
        """Get all forecast entries for a specific date. @zara"""
        return self._open_meteo.get_forecast_for_date(date)

    async def force_update(self) -> bool:
        """Force immediate forecast update from Open-Meteo API. @zara"""
        try:
            _LOGGER.info("Force update - fetching from Open-Meteo API...")
            forecast = await self._open_meteo.get_hourly_forecast(hours=72)

            if forecast:
                # File cache removed - now using database only
                self._cached_forecast = self._transform_open_meteo_forecast(forecast)
                await self._store_forecast_in_db(self._cached_forecast)
                _LOGGER.info(f"Force update successful: {len(self._cached_forecast)} hours")
                return True
            else:
                _LOGGER.warning("Force update returned no data")
                return False

        except Exception as e:
            _LOGGER.error(f"Force update failed: {e}", exc_info=True)
            return False

    async def _background_forecast_update(self) -> None:
        """Background task to periodically update forecast. @zara"""
        update_interval = 3600

        while True:
            try:
                await asyncio.sleep(update_interval)

                _LOGGER.debug("Background forecast update starting...")
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
        """Check health status of the weather service. @zara"""
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

    async def cleanup(self) -> None:
        """Cleanup resources. @zara"""
        if self._background_update_task and not self._background_update_task.done():
            self._background_update_task.cancel()
            try:
                await self._background_update_task
            except asyncio.CancelledError:
                pass

        _LOGGER.debug("Weather Service cleanup complete")
