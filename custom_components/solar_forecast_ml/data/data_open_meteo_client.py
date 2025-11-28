"""Open-Meteo API Client - SINGLE DATA SOURCE (Direct Values, No Calculations) @zara

IMPORTANT: Open-Meteo is now the ONLY weather data source.
- GHI (global_horizontal_irradiance) is used DIRECTLY as solar_radiation_wm2
- cloud_cover is used DIRECTLY (no physics-based calculations)
- All weather parameters come from Open-Meteo API
- NO blending with other weather sources
- NO cloud layer transmission calculations

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

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import time

import aiofiles
import aiohttp

_LOGGER = logging.getLogger(__name__)

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEOUT = 10
OPEN_METEO_CACHE_DURATION = 3600
OPEN_METEO_FILE_CACHE_MAX_AGE = 43200
OPEN_METEO_STALE_CACHE_MAX_AGE = 86400

OPEN_METEO_MAX_API_CALLS_PER_HOUR = 5
OPEN_METEO_RATE_LIMIT_WINDOW = 3600

DEFAULT_WEATHER = {
    "temperature": 15.0,
    "humidity": 70.0,
    "cloud_cover": 50.0,
    "precipitation": 0.0,
    "wind_speed": 3.0,
    "pressure": 1013.0,
    "direct_radiation": 100.0,
    "diffuse_radiation": 50.0,
    "ghi": 150.0,
    "source": "default_fallback",
    "confidence": 0.1,
}


class OpenMeteoClient:
    """Client for fetching weather data from Open-Meteo API - SINGLE DATA SOURCE. @zara"""

    def __init__(self, latitude: float, longitude: float, cache_file: Optional[Path] = None):
        self.latitude = latitude
        self.longitude = longitude
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_file = cache_file
        self._file_cache_loaded = False
        self._api_call_timestamps: List[float] = []
        self._last_api_error: Optional[str] = None
        self._consecutive_failures: int = 0

        _LOGGER.info(
            f"OpenMeteoClient initialized - SINGLE DATA SOURCE MODE "
            f"(lat={latitude:.4f}, lon={longitude:.4f})"
            f"{' with persistent cache' if cache_file else ''}"
            f" - Using direct GHI and cloud_cover values, NO calculations"
        )

    async def async_init(self) -> bool:
        if self._file_cache_loaded:
            return True

        if self._cache_file and self._cache_file.exists():
            success = await self._load_file_cache()
            self._file_cache_loaded = True
            if success:
                _LOGGER.info("Open-Meteo file cache loaded successfully")
            return success

        self._file_cache_loaded = True
        return False

    async def _load_file_cache(self) -> bool:
        if not self._cache_file or not self._cache_file.exists():
            return False

        try:
            async with aiofiles.open(self._cache_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)

            return self._process_loaded_cache(data)

        except Exception as e:
            _LOGGER.warning(f"Error loading Open-Meteo cache file: {e}")
            return False

    def _process_loaded_cache(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict) or "forecast" not in data:
            _LOGGER.warning("Invalid Open-Meteo cache file structure")
            return False

        cache_time_str = data.get("metadata", {}).get("fetched_at")
        if cache_time_str:
            cache_time = datetime.fromisoformat(cache_time_str)
            age_seconds = (datetime.now() - cache_time).total_seconds()

            if age_seconds > OPEN_METEO_FILE_CACHE_MAX_AGE:
                _LOGGER.info(
                    f"Open-Meteo file cache is stale ({age_seconds/3600:.1f}h old), "
                    "will fetch fresh data"
                )
            else:
                self._cache_time = cache_time

        self._cache["hourly_forecast"] = []
        forecast_data = data.get("forecast", {})

        for date_str, hours in forecast_data.items():
            for hour_str, hour_data in hours.items():
                entry = {
                    "date": date_str,
                    "hour": int(hour_str),
                    **hour_data
                }
                self._cache["hourly_forecast"].append(entry)

        _LOGGER.info(
            f"Loaded Open-Meteo cache: {len(self._cache['hourly_forecast'])} hours"
        )
        return True

    async def _save_file_cache(self, hourly_data: List[Dict[str, Any]]) -> bool:
        if not self._cache_file:
            return False

        try:
            forecast_by_date: Dict[str, Dict[str, Dict[str, Any]]] = {}

            for entry in hourly_data:
                date_str = entry.get("date")
                hour = entry.get("hour")
                if date_str is None or hour is None:
                    continue

                if date_str not in forecast_by_date:
                    forecast_by_date[date_str] = {}

                direct_rad = entry.get("direct_radiation") or 0
                diffuse_rad = entry.get("diffuse_radiation") or 0
                ghi = direct_rad + diffuse_rad

                forecast_by_date[date_str][str(hour)] = {
                    "temperature": entry.get("temperature"),
                    "humidity": entry.get("humidity"),
                    "cloud_cover": entry.get("cloud_cover"),  # DIRECT VALUE - no calculations!
                    "precipitation": entry.get("precipitation"),
                    "wind_speed": entry.get("wind_speed"),
                    "pressure": entry.get("pressure"),
                    "direct_radiation": direct_rad,
                    "diffuse_radiation": diffuse_rad,
                    "ghi": ghi,  # DIRECT VALUE used as solar_radiation_wm2
                    "global_tilted_irradiance": entry.get("global_tilted_irradiance"),
                }

            cache_data = {
                "version": "2.0",
                "metadata": {
                    "fetched_at": datetime.now().isoformat(),
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "hours_cached": len(hourly_data),
                    "days_cached": len(forecast_by_date),
                    "mode": "direct_radiation",
                },
                "forecast": forecast_by_date
            }

            self._cache_file.parent.mkdir(parents=True, exist_ok=True)

            temp_file = self._cache_file.with_suffix('.tmp')
            async with aiofiles.open(temp_file, 'w') as f:
                await f.write(json.dumps(cache_data, indent=2))

            temp_file.replace(self._cache_file)

            _LOGGER.info(
                f"Saved Open-Meteo cache: {len(hourly_data)} hours, "
                f"{len(forecast_by_date)} days"
            )
            return True

        except Exception as e:
            _LOGGER.error(f"Error saving Open-Meteo cache file: {e}")
            return False

    async def get_hourly_forecast(self, hours: int = 72) -> Optional[List[Dict[str, Any]]]:
        if self._is_cache_valid():
            _LOGGER.debug("Using fresh cached Open-Meteo data")
            return self._cache.get("hourly_forecast")

        if self._check_rate_limit_budget():
            api_result = await self._fetch_from_api(hours)
            if api_result:
                self._consecutive_failures = 0
                self._last_api_error = None
                return api_result

            self._consecutive_failures += 1

        if self._is_cache_usable_as_fallback():
            cache_info = self._get_cache_source_info()
            _LOGGER.warning(
                f"Open-Meteo API unavailable, using stale cache "
                f"(age: {cache_info['age_hours']:.1f}h, confidence: {cache_info['confidence']:.0%})"
            )
            return self._cache.get("hourly_forecast")

        _LOGGER.warning(
            f"Open-Meteo: No cache available and API failed "
            f"({self._consecutive_failures} consecutive failures)."
        )
        return None

    async def _fetch_from_api(self, hours: int = 72) -> Optional[List[Dict[str, Any]]]:
        try:
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "hourly": ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "cloud_cover",
                    "precipitation",
                    "wind_speed_10m",
                    "pressure_msl",
                    "direct_radiation",
                    "diffuse_radiation",
                    "global_tilted_irradiance",
                    "shortwave_radiation",
                ]),
                "daily": ",".join([
                    "sunrise",
                    "sunset",
                    "daylight_duration",
                ]),
                "timezone": "auto",
                "forecast_days": 3,
            }

            self._record_api_call()

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    OPEN_METEO_BASE_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=OPEN_METEO_TIMEOUT)
                ) as response:
                    if response.status != 200:
                        self._last_api_error = f"HTTP {response.status}"
                        _LOGGER.warning(f"Open-Meteo API returned status {response.status}")
                        return None

                    data = await response.json()

            hourly_data = self._parse_hourly_response(data)

            if hourly_data:
                self._cache["hourly_forecast"] = hourly_data
                self._cache["daily"] = data.get("daily", {})
                self._cache_time = datetime.now()

                await self._save_file_cache(hourly_data)

                ghi_values = [h.get("ghi", 0) or 0 for h in hourly_data]
                _LOGGER.info(
                    f"Fetched {len(hourly_data)} hours from Open-Meteo "
                    f"(GHI range: {min(ghi_values):.0f} - {max(ghi_values):.0f} W/m²)"
                )

            return hourly_data

        except asyncio.TimeoutError:
            self._last_api_error = "Timeout"
            _LOGGER.warning("Open-Meteo API request timed out")
            return None
        except aiohttp.ClientError as e:
            self._last_api_error = f"Connection: {e}"
            _LOGGER.warning(f"Open-Meteo API connection error: {e}")
            return None
        except Exception as e:
            self._last_api_error = f"Error: {e}"
            _LOGGER.error(f"Open-Meteo API error: {e}", exc_info=True)
            return None

    def _parse_hourly_response(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        try:
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            if not times:
                _LOGGER.warning("Open-Meteo response has no time data")
                return None

            result = []
            for i, time_str in enumerate(times):
                dt = datetime.fromisoformat(time_str)

                direct_rad = self._safe_get(hourly, "direct_radiation", i) or 0
                diffuse_rad = self._safe_get(hourly, "diffuse_radiation", i) or 0
                shortwave = self._safe_get(hourly, "shortwave_radiation", i)
                ghi = shortwave if shortwave is not None else (direct_rad + diffuse_rad)

                hour_data = {
                    "datetime": dt,
                    "date": dt.date().isoformat(),
                    "hour": dt.hour,
                    "temperature": self._safe_get(hourly, "temperature_2m", i),
                    "humidity": self._safe_get(hourly, "relative_humidity_2m", i),
                    "cloud_cover": self._safe_get(hourly, "cloud_cover", i),
                    "precipitation": self._safe_get(hourly, "precipitation", i),
                    "wind_speed": self._safe_get(hourly, "wind_speed_10m", i),
                    "pressure": self._safe_get(hourly, "pressure_msl", i),
                    "direct_radiation": direct_rad,
                    "diffuse_radiation": diffuse_rad,
                    "ghi": ghi,
                    "global_tilted_irradiance": self._safe_get(hourly, "global_tilted_irradiance", i),
                    "source": "open-meteo",
                }
                result.append(hour_data)

            return result

        except Exception as e:
            _LOGGER.error(f"Error parsing Open-Meteo response: {e}", exc_info=True)
            return None

    @staticmethod
    def _safe_get(data: Dict, key: str, index: int) -> Optional[float]:
        try:
            values = data.get(key, [])
            if index < len(values):
                return values[index]
        except (IndexError, TypeError):
            pass
        return None

    def _is_cache_valid(self) -> bool:
        if not self._cache_time or not self._cache.get("hourly_forecast"):
            return False
        age = (datetime.now() - self._cache_time).total_seconds()
        return age < OPEN_METEO_CACHE_DURATION

    def _is_cache_usable_as_fallback(self) -> bool:
        if not self._cache.get("hourly_forecast"):
            return False
        if not self._cache_time:
            return True
        age = (datetime.now() - self._cache_time).total_seconds()
        return age < OPEN_METEO_STALE_CACHE_MAX_AGE

    def _check_rate_limit_budget(self) -> bool:
        now = time.time()
        cutoff = now - OPEN_METEO_RATE_LIMIT_WINDOW

        self._api_call_timestamps = [ts for ts in self._api_call_timestamps if ts > cutoff]

        if len(self._api_call_timestamps) >= OPEN_METEO_MAX_API_CALLS_PER_HOUR:
            remaining_seconds = self._api_call_timestamps[0] + OPEN_METEO_RATE_LIMIT_WINDOW - now
            _LOGGER.warning(
                f"Open-Meteo rate limit budget exhausted ({OPEN_METEO_MAX_API_CALLS_PER_HOUR}/h). "
                f"Budget resets in {remaining_seconds/60:.1f} minutes"
            )
            return False

        return True

    def _record_api_call(self) -> None:
        self._api_call_timestamps.append(time.time())

    def _get_cache_source_info(self) -> Dict[str, Any]:
        if not self._cache.get("hourly_forecast"):
            return {"source": "none", "confidence": 0.0, "age_hours": None}

        if not self._cache_time:
            return {"source": "file_cache_no_timestamp", "confidence": 0.5, "age_hours": None}

        age_hours = (datetime.now() - self._cache_time).total_seconds() / 3600

        if age_hours < 1:
            return {"source": "fresh_cache", "confidence": 0.95, "age_hours": age_hours}
        elif age_hours < 6:
            return {"source": "recent_cache", "confidence": 0.85, "age_hours": age_hours}
        elif age_hours < 12:
            return {"source": "stale_cache", "confidence": 0.7, "age_hours": age_hours}
        else:
            return {"source": "old_cache", "confidence": 0.5, "age_hours": age_hours}

    def get_weather_for_hour(self, date: str, hour: int) -> Optional[Dict[str, Any]]:
        cached = self._cache.get("hourly_forecast", [])
        for entry in cached:
            if entry.get("date") == date and entry.get("hour") == hour:
                return entry
        return None

    def get_radiation_for_hour(self, date: str, hour: int) -> Tuple[float, float, float]:
        entry = self.get_weather_for_hour(date, hour)
        if entry:
            direct = entry.get("direct_radiation") or 0
            diffuse = entry.get("diffuse_radiation") or 0
            ghi = entry.get("ghi") or (direct + diffuse)
            return direct, diffuse, ghi
        return 0.0, 0.0, 0.0

    def get_forecast_for_date(self, date: str) -> List[Dict[str, Any]]:
        cached = self._cache.get("hourly_forecast", [])
        return [entry for entry in cached if entry.get("date") == date]

    def get_daily_data(self) -> Dict[str, Any]:
        return self._cache.get("daily", {})


def get_default_weather() -> Dict[str, Any]:
    return DEFAULT_WEATHER.copy()


# Archive API for historical data
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class OpenMeteoArchiveClient:
    """Client for fetching HISTORICAL weather data from Open-Meteo Archive API.

    This is separate from the forecast client. The Archive API provides
    historical weather data for dates in the past (typically up to 2-3 years back).

    Used by:
    - bootstrap_physics_from_history service
    - Historical training data collection
    - Retroactive weather data for GeometryLearner
    """

    def __init__(self, latitude: float, longitude: float):
        """Initialize archive client.

        Args:
            latitude: Location latitude
            longitude: Location longitude
        """
        self.latitude = latitude
        self.longitude = longitude
        self._last_error: Optional[str] = None

        _LOGGER.info(
            f"OpenMeteoArchiveClient initialized for historical data "
            f"(lat={latitude:.4f}, lon={longitude:.4f})"
        )

    async def get_historical_weather(
        self,
        start_date: str,
        end_date: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch historical hourly weather data.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of hourly weather data or None if failed

        Example:
            data = await client.get_historical_weather("2025-06-01", "2025-11-28")
        """
        try:
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "cloud_cover",
                    "precipitation",
                    "wind_speed_10m",
                    "pressure_msl",
                    "direct_radiation",
                    "diffuse_radiation",
                    "shortwave_radiation",  # GHI
                ]),
                "timezone": "auto",
            }

            _LOGGER.info(
                f"Fetching historical weather from {start_date} to {end_date}..."
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    OPEN_METEO_ARCHIVE_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=60),  # Longer timeout for large requests
                ) as response:
                    if response.status != 200:
                        self._last_error = f"HTTP {response.status}"
                        _LOGGER.warning(
                            f"Open-Meteo Archive API returned status {response.status}"
                        )
                        return None

                    data = await response.json()

            hourly_data = self._parse_archive_response(data)

            if hourly_data:
                _LOGGER.info(
                    f"Fetched {len(hourly_data)} hours of historical weather data "
                    f"({start_date} to {end_date})"
                )
            return hourly_data

        except asyncio.TimeoutError:
            self._last_error = "Timeout"
            _LOGGER.warning("Open-Meteo Archive API request timed out")
            return None
        except aiohttp.ClientError as e:
            self._last_error = f"Connection: {e}"
            _LOGGER.warning(f"Open-Meteo Archive API connection error: {e}")
            return None
        except Exception as e:
            self._last_error = f"Error: {e}"
            _LOGGER.error(f"Open-Meteo Archive API error: {e}", exc_info=True)
            return None

    def _parse_archive_response(
        self, data: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Parse hourly response from archive API."""
        try:
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            if not times:
                _LOGGER.warning("Open-Meteo Archive response has no time data")
                return None

            result = []
            for i, time_str in enumerate(times):
                dt = datetime.fromisoformat(time_str)

                # Get radiation values
                direct_rad = self._safe_get(hourly, "direct_radiation", i) or 0
                diffuse_rad = self._safe_get(hourly, "diffuse_radiation", i) or 0
                shortwave = self._safe_get(hourly, "shortwave_radiation", i)
                ghi = shortwave if shortwave is not None else (direct_rad + diffuse_rad)

                hour_data = {
                    "datetime": dt,
                    "date": dt.date().isoformat(),
                    "hour": dt.hour,
                    "temperature": self._safe_get(hourly, "temperature_2m", i),
                    "humidity": self._safe_get(hourly, "relative_humidity_2m", i),
                    "cloud_cover": self._safe_get(hourly, "cloud_cover", i),
                    "precipitation": self._safe_get(hourly, "precipitation", i),
                    "wind_speed": self._safe_get(hourly, "wind_speed_10m", i),
                    "pressure": self._safe_get(hourly, "pressure_msl", i),
                    "direct_radiation": direct_rad,
                    "diffuse_radiation": diffuse_rad,
                    "ghi": ghi,
                    "source": "open-meteo-archive",
                }
                result.append(hour_data)

            return result

        except Exception as e:
            _LOGGER.error(f"Error parsing Open-Meteo Archive response: {e}", exc_info=True)
            return None

    @staticmethod
    def _safe_get(data: Dict, key: str, index: int) -> Optional[float]:
        """Safely get a value from array in dict."""
        try:
            values = data.get(key, [])
            if index < len(values):
                return values[index]
        except (IndexError, TypeError):
            pass
        return None

    def get_last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error
