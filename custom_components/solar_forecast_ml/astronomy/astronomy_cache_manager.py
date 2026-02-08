# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
In-Memory Astronomy Cache Manager for Solar Forecast ML V16.2.0.
Provides fast, thread-safe synchronous access to astronomy data from database.
"""

import logging
from datetime import date
from threading import Lock
from typing import Any, Dict, Optional

from ..data.db_manager import DatabaseManager

_LOGGER = logging.getLogger(__name__)


class AstronomyCacheManager:
    """In-memory cache manager for astronomy data with thread-safe access. @zara"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self._cache: Dict[str, Any] = {}
        self._lock = Lock()
        self._loaded = False

    async def initialize(self) -> bool:
        """Initialize cache from database. @zara"""
        try:
            system_info = await self.db.fetchone(
                """SELECT latitude, longitude, elevation_m, timezone,
                          installed_capacity_kwp, max_peak_record_kwh
                   FROM astronomy_system_info WHERE id = 1"""
            )

            if not system_info:
                _LOGGER.debug("No astronomy system info found in database")
                return False

            with self._lock:
                self._cache["system_info"] = {
                    "latitude": system_info[0],
                    "longitude": system_info[1],
                    "elevation_m": system_info[2],
                    "timezone": system_info[3],
                    "installed_capacity_kwp": system_info[4],
                    "max_peak_record_kwh": system_info[5],
                }

            # Load daily astronomy data for today and tomorrow @zara
            await self._load_day_data()

            with self._lock:
                self._loaded = True

            _LOGGER.debug("Astronomy cache manager initialized from database")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to initialize astronomy cache manager: {e}")
            return False

    async def _load_day_data(self) -> None:
        """Load daily + hourly astronomy data from database. @zara V16.0.0"""
        try:
            # Load daily summaries @zara
            rows = await self.db.fetchall(
                """SELECT cache_date,
                          MIN(sunrise) as sunrise,
                          MAX(sunset) as sunset,
                          AVG(solar_noon) as solar_noon,
                          MAX(daylight_hours) as daylight_hours,
                          MAX(sun_elevation_deg) as max_elevation_deg
                   FROM astronomy_cache
                   WHERE cache_date >= DATE('now', '-1 day')
                   GROUP BY cache_date
                   ORDER BY cache_date"""
            )

            with self._lock:
                if "days" not in self._cache:
                    self._cache["days"] = {}

                for row in rows:
                    date_str = str(row[0])
                    sunrise = row[1]
                    sunset = row[2]

                    self._cache["days"][date_str] = {
                        "sunrise_time": sunrise,
                        "sunset_time": sunset,
                        "solar_noon": row[3],
                        # Aliase for compatibility @zara V16.0.0
                        "sunrise_local": sunrise,
                        "sunset_local": sunset,
                        "solar_noon_local": row[3],
                        "daylight_hours": row[4],
                        # Use sunrise/sunset as production window @zara
                        "production_window_start": sunrise,
                        "production_window_end": sunset,
                        "day_length_hours": row[4],
                        "max_elevation_deg": row[5],
                        "hourly": {},
                    }

            # Load hourly astronomy data @zara V16.0.0
            hourly_rows = await self.db.fetchall(
                """SELECT cache_date, hour, sun_elevation_deg, sun_azimuth_deg,
                          clear_sky_radiation_wm2, theoretical_max_kwh,
                          daylight_hours, sunrise, sunset, solar_noon
                   FROM astronomy_cache
                   WHERE cache_date >= DATE('now', '-1 day')
                   ORDER BY cache_date, hour"""
            )

            with self._lock:
                for row in hourly_rows:
                    date_str = str(row[0])
                    hour_str = str(row[1])

                    if date_str not in self._cache.get("days", {}):
                        continue

                    self._cache["days"][date_str]["hourly"][hour_str] = {
                        "elevation_deg": row[2],
                        "azimuth_deg": row[3],
                        "clear_sky_solar_radiation_wm2": row[4],
                        "theoretical_max_pv_kwh": row[5],
                        "daylight_hours": row[6],
                        "sunrise": row[7],
                        "sunset": row[8],
                        "solar_noon": row[9],
                    }

            _LOGGER.debug(
                "Loaded %d days + %d hourly entries of astronomy data into cache",
                len(rows), len(hourly_rows)
            )

        except Exception as e:
            _LOGGER.warning(f"Failed to load day data: {e}")

    def get_day_data(self, target_date: date) -> Optional[Dict[str, Any]]:
        """Get astronomy data for a specific date (synchronous, thread-safe). @zara"""
        if not self._loaded:
            return None

        date_str = target_date.isoformat()
        with self._lock:
            return self._cache.get("days", {}).get(date_str)

    def get_production_window(self, target_date: date) -> Optional[tuple]:
        """Get production window for a date (synchronous). @zara"""
        day_data = self.get_day_data(target_date)
        if not day_data:
            return None

        start = day_data.get("production_window_start")
        end = day_data.get("production_window_end")

        if start and end:
            return (start, end)
        return None

    def get_hourly_data(self, target_date: date, hour: int) -> Optional[Dict[str, Any]]:
        """Get hourly astronomy data for specific date and hour. @zara"""
        day_data = self.get_day_data(target_date)
        if not day_data:
            return None

        return day_data.get("hourly", {}).get(str(hour))

    def get_system_info(self) -> Optional[Dict[str, Any]]:
        """Get PV system information from cache. @zara"""
        if not self._loaded:
            return None

        with self._lock:
            return self._cache.get("system_info")

    def is_loaded(self) -> bool:
        """Check if cache is loaded. @zara"""
        return self._loaded

    def clear(self):
        """Clear the cache. @zara"""
        with self._lock:
            self._cache = {}
            self._loaded = False


_cache_manager: Optional[AstronomyCacheManager] = None


def get_cache_manager(db_manager: Optional[DatabaseManager] = None) -> Optional[AstronomyCacheManager]:
    """Get the global cache manager instance. @zara"""
    global _cache_manager
    if _cache_manager is None:
        if db_manager is None:
            _LOGGER.debug("AstronomyCacheManager not initialized and no db_manager provided")
            return None
        _cache_manager = AstronomyCacheManager(db_manager)
    elif db_manager is not None and _cache_manager.db is not db_manager:
        _cache_manager.db = db_manager
        _cache_manager._loaded = False
    return _cache_manager


def reset_cache_manager() -> None:
    """Reset the global cache manager instance on unload. @zara"""
    global _cache_manager
    if _cache_manager is not None:
        _cache_manager.clear()
        _cache_manager = None
        _LOGGER.debug("AstronomyCacheManager singleton reset")
