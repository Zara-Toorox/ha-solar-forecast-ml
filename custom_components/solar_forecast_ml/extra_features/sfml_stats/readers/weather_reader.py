# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Weather data reader for SFML Stats. @zara"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

import aiosqlite

from ..const import SOLAR_FORECAST_DB

if TYPE_CHECKING:
    from ..storage.db_connection_manager import DatabaseConnectionManager

_LOGGER = logging.getLogger(__name__)


@dataclass
class HourlyWeather:
    """Hourly weather data. @zara"""

    date: date
    hour: int
    temperature_c: float | None
    humidity_percent: float | None
    wind_speed_ms: float | None
    precipitation_mm: float | None
    pressure_hpa: float | None
    solar_radiation_wm2: float | None
    cloud_cover_percent: float | None


class WeatherDataReader:
    """Reads weather data from Solar Forecast ML SQLite database. @zara"""

    _db_manager: DatabaseConnectionManager | None = None

    def __init__(self, config_path: Path, db_manager: DatabaseConnectionManager | None = None) -> None:
        """Initialize the weather data reader. @zara"""
        self._config_path = config_path
        self._db_path = config_path / SOLAR_FORECAST_DB
        if db_manager is not None:
            WeatherDataReader._db_manager = db_manager

    @property
    def is_available(self) -> bool:
        """Check if Solar Forecast ML database is available. @zara"""
        if self._db_manager is not None:
            return self._db_manager.is_available
        return self._db_path.exists()

    @asynccontextmanager
    async def _get_db_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Get a database connection from the manager. @zara"""
        from ..storage.db_connection_manager import get_manager

        manager = get_manager()
        if manager is not None and manager.is_connected:
            _LOGGER.debug("WeatherDataReader: Using database connection manager")
            yield await manager.get_connection()
        else:
            _LOGGER.warning("WeatherDataReader: Database manager not available, using direct connection (THIS CAUSES THREADING ERRORS)")
            conn = await aiosqlite.connect(str(self._db_path))
            conn.row_factory = aiosqlite.Row
            try:
                yield conn
            finally:
                await conn.close()

    async def async_get_hourly_weather(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[HourlyWeather]:
        """Read hourly weather data from database. @zara"""
        if not self.is_available:
            _LOGGER.debug("Database not found: %s", self._db_path)
            return []

        try:
            async with self._get_db_connection() as conn:
                query = """
                    SELECT
                        date, hour, temperature_c, humidity_percent,
                        wind_speed_ms, precipitation_mm, pressure_hpa,
                        solar_radiation_wm2, cloud_cover_percent
                    FROM hourly_weather_actual
                """
                params = []
                conditions = []

                if start_date:
                    conditions.append("date >= ?")
                    params.append(start_date.isoformat())
                if end_date:
                    conditions.append("date <= ?")
                    params.append(end_date.isoformat())

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY date, hour"

                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                weather_data: list[HourlyWeather] = []

                for row in rows:
                    try:
                        weather_date = date.fromisoformat(row["date"]) if isinstance(row["date"], str) else row["date"]

                        weather = HourlyWeather(
                            date=weather_date,
                            hour=row["hour"] or 0,
                            temperature_c=row["temperature_c"],
                            humidity_percent=row["humidity_percent"],
                            wind_speed_ms=row["wind_speed_ms"],
                            precipitation_mm=row["precipitation_mm"],
                            pressure_hpa=row["pressure_hpa"],
                            solar_radiation_wm2=row["solar_radiation_wm2"],
                            cloud_cover_percent=row["cloud_cover_percent"],
                        )
                        weather_data.append(weather)

                    except Exception as err:
                        _LOGGER.warning("Error parsing weather row: %s", err)
                        continue

                return weather_data

        except Exception as err:
            _LOGGER.error("Error reading weather data from database: %s", err)
            return []

    async def async_get_forecast_weather(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[HourlyWeather]:
        """Read hourly weather forecast from database. @zara"""
        if not self.is_available:
            _LOGGER.debug("Database not found: %s", self._db_path)
            return []

        try:
            async with self._get_db_connection() as conn:
                query = """
                    SELECT
                        forecast_date as date, hour, temperature,
                        humidity, wind, rain, solar_radiation_wm2, clouds
                    FROM weather_forecast
                """
                params = []
                conditions = []

                if start_date:
                    conditions.append("forecast_date >= ?")
                    params.append(start_date.isoformat())
                if end_date:
                    conditions.append("forecast_date <= ?")
                    params.append(end_date.isoformat())

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY forecast_date, hour"

                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                weather_data: list[HourlyWeather] = []

                for row in rows:
                    try:
                        weather_date = date.fromisoformat(row["date"]) if isinstance(row["date"], str) else row["date"]

                        weather = HourlyWeather(
                            date=weather_date,
                            hour=row["hour"] or 0,
                            temperature_c=row["temperature"],
                            humidity_percent=row["humidity"],
                            wind_speed_ms=row["wind"],
                            precipitation_mm=row["rain"],
                            pressure_hpa=None,
                            solar_radiation_wm2=row["solar_radiation_wm2"],
                            cloud_cover_percent=row["clouds"],
                        )
                        weather_data.append(weather)

                    except Exception as err:
                        _LOGGER.warning("Error parsing forecast weather row: %s", err)
                        continue

                return weather_data

        except Exception as err:
            _LOGGER.error("Error reading forecast weather from database: %s", err)
            return []
