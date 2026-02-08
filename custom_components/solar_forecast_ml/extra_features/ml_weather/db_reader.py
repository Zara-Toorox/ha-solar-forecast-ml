# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML - ML Weather
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************
"""Database reader for ML Weather - reads from Solar Forecast ML SQLite database."""

import asyncio
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

MAX_RECONNECT_ATTEMPTS = 3


class DatabaseReader:
    """Reads weather and PV forecast data from the Solar Forecast ML database."""

    def __init__(self, hass: HomeAssistant, db_path: str) -> None:
        """Initialize the database reader."""
        self._hass = hass
        self._db_path = Path(hass.config.path(db_path))
        self._connection: aiosqlite.Connection | None = None
        self._is_connected: bool = False
        self._lock = asyncio.Lock()

    @property
    def is_available(self) -> bool:
        """Check if database file exists."""
        return self._db_path.exists()

    @property
    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._is_connected and self._connection is not None

    @property
    def db_path(self) -> str:
        """Return the resolved database path."""
        return str(self._db_path)

    async def async_connect(self) -> bool:
        """Establish database connection."""
        async with self._lock:
            if self._is_connected and self._connection is not None:
                return True

            if not self.is_available:
                _LOGGER.warning("Database not found: %s", self._db_path)
                return False

            try:
                self._connection = await aiosqlite.connect(str(self._db_path))
                self._connection.row_factory = aiosqlite.Row
                self._is_connected = True
                _LOGGER.info("ML Weather database connection established: %s", self._db_path)
                return True
            except Exception as err:
                _LOGGER.error("Failed to connect to database: %s", err)
                self._connection = None
                self._is_connected = False
                return False

    async def async_close(self) -> None:
        """Close database connection."""
        async with self._lock:
            if self._connection is not None:
                try:
                    await self._connection.close()
                    _LOGGER.debug("ML Weather database connection closed")
                except Exception as err:
                    _LOGGER.error("Error closing database connection: %s", err)
                finally:
                    self._connection = None
                    self._is_connected = False

    async def _ensure_connected(self) -> bool:
        """Ensure we have an active database connection, reconnect if needed."""
        if self.is_connected:
            return True

        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            if await self.async_connect():
                return True
            _LOGGER.debug(
                "Reconnect attempt %d/%d failed", attempt + 1, MAX_RECONNECT_ATTEMPTS
            )

        return False

    async def async_get_weather_forecast(
        self,
        start_date: str,
        end_date: str,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Read weather forecast from database.

        Returns data in nested dict format matching the old JSON structure:
        {date_str: {hour_str: {field: value, ...}, ...}, ...}
        """
        if not await self._ensure_connected():
            return {}

        try:
            query = """
                SELECT
                    forecast_date, hour,
                    temperature, humidity, pressure, wind, rain, clouds,
                    cloud_cover_low, cloud_cover_mid, cloud_cover_high,
                    solar_radiation_wm2, direct_radiation, diffuse_radiation,
                    visibility_m, fog_detected, fog_type
                FROM weather_forecast
                WHERE forecast_date >= ? AND forecast_date <= ?
                ORDER BY forecast_date, hour
            """

            async with self._connection.execute(query, [start_date, end_date]) as cursor:
                rows = await cursor.fetchall()

            result: dict[str, dict[str, dict[str, Any]]] = {}

            for row in rows:
                date_str = str(row["forecast_date"])
                hour_str = str(row["hour"])

                if date_str not in result:
                    result[date_str] = {}

                result[date_str][hour_str] = {
                    "temperature": row["temperature"],
                    "humidity": row["humidity"],
                    "pressure": row["pressure"],
                    "wind": row["wind"],
                    "rain": row["rain"],
                    "clouds": row["clouds"],
                    "cloud_cover_low": row["cloud_cover_low"],
                    "cloud_cover_mid": row["cloud_cover_mid"],
                    "cloud_cover_high": row["cloud_cover_high"],
                    "solar_radiation_wm2": row["solar_radiation_wm2"],
                    "direct_radiation": row["direct_radiation"],
                    "diffuse_radiation": row["diffuse_radiation"],
                    "visibility_m": row["visibility_m"],
                    "fog_detected": bool(row["fog_detected"]) if row["fog_detected"] is not None else None,
                    "fog_type": row["fog_type"],
                }

            return result

        except Exception as err:
            _LOGGER.error("Error reading weather forecast from database: %s", err)
            self._is_connected = False
            self._connection = None
            return {}

    async def async_get_weather_version(self) -> str:
        """Get the version string from the most recent weather_forecast row."""
        if not await self._ensure_connected():
            return "unknown"

        try:
            query = """
                SELECT version FROM weather_forecast
                ORDER BY updated_at DESC LIMIT 1
            """
            async with self._connection.execute(query) as cursor:
                row = await cursor.fetchone()
                return row["version"] if row and row["version"] else "unknown"
        except Exception:
            return "unknown"

    async def async_get_pv_forecast(self) -> dict[str, Any]:
        """Read PV forecast data from the daily_forecasts table.

        Returns flat dict with today/tomorrow/day_after_tomorrow values.
        """
        if not await self._ensure_connected():
            return {}

        try:
            today_str = date.today().isoformat()

            query = """
                SELECT
                    forecast_type, forecast_date, prediction_kwh,
                    locked, source
                FROM daily_forecasts
                WHERE forecast_date >= ?
                ORDER BY forecast_type, created_at DESC
            """

            async with self._connection.execute(query, [today_str]) as cursor:
                rows = await cursor.fetchall()

            result: dict[str, Any] = {}
            seen_types: set[str] = set()

            for row in rows:
                ftype = row["forecast_type"]
                if ftype in seen_types:
                    continue
                seen_types.add(ftype)

                if ftype == "today":
                    result["today"] = row["prediction_kwh"]
                    result["today_source"] = row["source"]
                    result["today_locked"] = bool(row["locked"]) if row["locked"] is not None else False
                elif ftype == "tomorrow":
                    result["tomorrow"] = row["prediction_kwh"]
                    result["tomorrow_date"] = str(row["forecast_date"])
                    result["tomorrow_source"] = row["source"]
                elif ftype == "day_after_tomorrow":
                    result["day_after_tomorrow"] = row["prediction_kwh"]
                    result["day_after_tomorrow_date"] = str(row["forecast_date"])
                    result["day_after_tomorrow_source"] = row["source"]

            return result

        except Exception as err:
            _LOGGER.error("Error reading PV forecast from database: %s", err)
            self._is_connected = False
            self._connection = None
            return {}

    async def async_get_db_info(self) -> dict[str, Any]:
        """Get information about the database status."""
        info = {
            "db_path": str(self._db_path),
            "db_exists": self.is_available,
            "db_connected": self.is_connected,
        }

        if self.is_available:
            try:
                stat = self._db_path.stat()
                info["db_size_kb"] = round(stat.st_size / 1024, 2)
                info["db_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            except OSError:
                pass

        if self.is_connected:
            try:
                async with self._connection.execute(
                    "SELECT COUNT(*) as cnt FROM weather_forecast"
                ) as cursor:
                    row = await cursor.fetchone()
                    info["weather_forecast_rows"] = row["cnt"] if row else 0

                async with self._connection.execute(
                    "SELECT COUNT(*) as cnt FROM daily_forecasts"
                ) as cursor:
                    row = await cursor.fetchone()
                    info["daily_forecasts_rows"] = row["cnt"] if row else 0

                async with self._connection.execute(
                    "SELECT MAX(updated_at) as latest FROM weather_forecast"
                ) as cursor:
                    row = await cursor.fetchone()
                    info["latest_update"] = row["latest"] if row else None

            except Exception as err:
                _LOGGER.debug("Error getting DB info: %s", err)

        return info
