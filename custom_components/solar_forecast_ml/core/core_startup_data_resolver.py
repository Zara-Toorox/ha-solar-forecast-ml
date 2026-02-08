# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Startup Data Resolver for Solar Forecast ML V16.2.0.
Non-blocking startup data resolution with DB-first approach and background sensor retry.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Callable

from ..data.db_manager import DatabaseManager
from .core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


@dataclass
class StartupData:
    weather_available: bool = False
    astronomy_available: bool = False
    sensors_available: bool = False
    weather_data: Dict[str, Any] = field(default_factory=dict)
    astronomy_data: Dict[str, Any] = field(default_factory=dict)
    sensor_data: Dict[str, Optional[float]] = field(default_factory=dict)
    source: str = "unknown"
    warnings: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.weather_available or self.astronomy_available

    @property
    def is_optimal(self) -> bool:
        return self.weather_available and self.astronomy_available and self.sensors_available


class StartupDataResolver:
    """Resolves startup data from DB first, with non-blocking sensor retry."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        sensor_collector: Any,
        on_sensors_available: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.db = db_manager
        self.sensor_collector = sensor_collector
        self.on_sensors_available = on_sensors_available

        self._retry_task: Optional[asyncio.Task] = None
        self._retry_count: int = 0
        self._max_retries: int = 5
        self._retry_interval: int = 30
        self._sensors_resolved: bool = False
        self._shutdown: bool = False

    async def resolve_startup_data(self) -> StartupData:
        """Resolve startup data: DB first, sensors non-blocking."""
        today = dt_util.now().date()
        result = StartupData()

        weather_data, weather_ok = await self._load_weather_from_db(today)
        if weather_ok:
            result.weather_available = True
            result.weather_data = weather_data
            _LOGGER.info("Startup: Weather data loaded from DB (%d hours)", len(weather_data))

        astro_data, astro_ok = await self._load_astronomy_from_db(today)
        if astro_ok:
            result.astronomy_available = True
            result.astronomy_data = astro_data
            _LOGGER.info("Startup: Astronomy data loaded from DB")

        sensor_data = self._try_collect_sensors()
        if self._has_valid_sensors(sensor_data):
            result.sensors_available = True
            result.sensor_data = sensor_data
            self._sensors_resolved = True
            _LOGGER.info("Startup: External sensors available immediately")
        else:
            _LOGGER.info("Startup: External sensors not yet available, starting background retry")
            self._start_sensor_retry_task()

        result.source = self._determine_source(result)

        if not result.is_usable:
            result.warnings.append("No weather or astronomy data available in DB")
            if not result.sensors_available:
                result.warnings.append("External sensors also unavailable")
            _LOGGER.warning(
                "Startup: Operating in degraded mode - no DB data and sensors offline"
            )

        return result

    async def _load_weather_from_db(self, target_date: date) -> tuple[Dict[str, Any], bool]:
        """Load weather forecast from database."""
        try:
            rows = await self.db.fetchall(
                """SELECT hour, temperature, solar_radiation_wm2, wind, humidity,
                          rain, clouds, pressure, direct_radiation, diffuse_radiation
                   FROM weather_forecast
                   WHERE forecast_date = ?
                   ORDER BY hour""",
                (target_date.isoformat(),)
            )

            if not rows or len(rows) < 6:
                return {}, False

            hourly = {}
            for row in rows:
                hour = row[0]
                hourly[str(hour)] = {
                    "temperature": row[1],
                    "solar_radiation_wm2": row[2],
                    "wind": row[3],
                    "humidity": row[4],
                    "rain": row[5],
                    "clouds": row[6],
                    "pressure": row[7],
                    "direct_radiation": row[8],
                    "diffuse_radiation": row[9],
                }

            return {"hourly": hourly, "date": target_date.isoformat()}, True

        except Exception as e:
            _LOGGER.debug("Failed to load weather from DB: %s", e)
            return {}, False

    async def _load_astronomy_from_db(self, target_date: date) -> tuple[Dict[str, Any], bool]:
        """Load astronomy cache from database."""
        try:
            rows = await self.db.fetchall(
                """SELECT hour, sun_elevation_deg, sun_azimuth_deg,
                          clear_sky_radiation_wm2, theoretical_max_kwh,
                          sunrise, sunset, solar_noon, daylight_hours
                   FROM astronomy_cache
                   WHERE cache_date = ?
                   ORDER BY hour""",
                (target_date.isoformat(),)
            )

            if not rows or len(rows) < 12:
                return {}, False

            hourly = {}
            sunrise = sunset = solar_noon = daylight_hours = None

            for row in rows:
                hour = row[0]
                hourly[str(hour)] = {
                    "elevation_deg": row[1],
                    "azimuth_deg": row[2],
                    "clear_sky_radiation_wm2": row[3],
                    "theoretical_max_kwh": row[4],
                }
                if sunrise is None:
                    sunrise = row[5]
                    sunset = row[6]
                    solar_noon = row[7]
                    daylight_hours = row[8]

            return {
                "hourly": hourly,
                "sunrise": sunrise,
                "sunset": sunset,
                "solar_noon": solar_noon,
                "daylight_hours": daylight_hours,
                "date": target_date.isoformat(),
            }, True

        except Exception as e:
            _LOGGER.debug("Failed to load astronomy from DB: %s", e)
            return {}, False

    def _try_collect_sensors(self) -> Dict[str, Optional[float]]:
        """Try to collect sensor data (non-blocking)."""
        try:
            return self.sensor_collector.collect_all_sensor_data_dict()
        except Exception as e:
            _LOGGER.debug("Sensor collection failed: %s", e)
            return {}

    def _has_valid_sensors(self, sensor_data: Dict[str, Optional[float]]) -> bool:
        """Check if sensor data contains at least one valid value."""
        if not sensor_data:
            return False
        return any(v is not None for v in sensor_data.values())

    def _determine_source(self, data: StartupData) -> str:
        """Determine data source description."""
        sources = []
        if data.weather_available:
            sources.append("DB-Weather")
        if data.astronomy_available:
            sources.append("DB-Astro")
        if data.sensors_available:
            sources.append("Sensors")
        return "+".join(sources) if sources else "None"

    def _start_sensor_retry_task(self) -> None:
        """Start background sensor retry task."""
        if self._retry_task is not None and not self._retry_task.done():
            return
        self._retry_task = asyncio.create_task(
            self._sensor_retry_loop(),
            name="startup_sensor_retry"
        )

    async def _sensor_retry_loop(self) -> None:
        """Background loop to retry sensor collection."""
        while not self._shutdown and self._retry_count < self._max_retries:
            await asyncio.sleep(self._retry_interval)

            if self._shutdown:
                break

            self._retry_count += 1
            _LOGGER.debug(
                "Sensor retry attempt %d/%d",
                self._retry_count,
                self._max_retries
            )

            sensor_data = self._try_collect_sensors()

            if self._has_valid_sensors(sensor_data):
                self._sensors_resolved = True
                _LOGGER.info(
                    "External sensors became available after %d retries",
                    self._retry_count
                )

                if self.on_sensors_available:
                    try:
                        self.on_sensors_available(sensor_data)
                    except Exception as e:
                        _LOGGER.error("Callback on_sensors_available failed: %s", e)

                await self._update_db_with_sensor_data(sensor_data)
                return

        if not self._sensors_resolved:
            _LOGGER.warning(
                "External sensors remain offline after %d retries - using DB data only",
                self._max_retries
            )

    async def _update_db_with_sensor_data(self, sensor_data: Dict[str, Optional[float]]) -> None:
        """Update DB with fresh sensor data when available."""
        try:
            now = dt_util.now()
            today = now.date().isoformat()
            current_hour = now.hour

            temp = sensor_data.get("temperature")
            humidity = sensor_data.get("humidity")
            solar_rad = sensor_data.get("solar_radiation")
            wind = sensor_data.get("wind_speed")
            pressure = sensor_data.get("pressure")
            rain = sensor_data.get("rain")

            if any(v is not None for v in [temp, humidity, solar_rad, wind, pressure]):
                await self.db.execute(
                    """INSERT INTO hourly_weather_actual
                       (date, hour, temperature_c, humidity_percent, solar_radiation_wm2,
                        wind_speed_ms, pressure_hpa, precipitation_mm, timestamp, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'startup_sensors')
                       ON CONFLICT(date, hour) DO UPDATE SET
                           temperature_c = COALESCE(excluded.temperature_c, temperature_c),
                           humidity_percent = COALESCE(excluded.humidity_percent, humidity_percent),
                           solar_radiation_wm2 = COALESCE(excluded.solar_radiation_wm2, solar_radiation_wm2),
                           wind_speed_ms = COALESCE(excluded.wind_speed_ms, wind_speed_ms),
                           pressure_hpa = COALESCE(excluded.pressure_hpa, pressure_hpa),
                           precipitation_mm = COALESCE(excluded.precipitation_mm, precipitation_mm),
                           timestamp = excluded.timestamp""",
                    (today, current_hour, temp, humidity, solar_rad, wind, pressure, rain, now.isoformat())
                )
                _LOGGER.debug("Updated DB with sensor data for hour %d", current_hour)

        except Exception as e:
            _LOGGER.warning("Failed to update DB with sensor data: %s", e)

    async def shutdown(self) -> None:
        """Shutdown the resolver and cancel pending tasks."""
        self._shutdown = True

        if self._retry_task is not None and not self._retry_task.done():
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass

        _LOGGER.debug("StartupDataResolver shutdown complete")

    @property
    def sensors_resolved(self) -> bool:
        return self._sensors_resolved

    @property
    def retry_count(self) -> int:
        return self._retry_count
