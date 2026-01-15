# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

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
OPEN_METEO_HISTORY_RETENTION_DAYS = 730  # 2 years

# V12.8.5: Visibility API settings
# ECMWF model doesn't provide visibility, so we fetch it separately from default model
OPEN_METEO_VISIBILITY_CACHE_DURATION = 3600  # 1 hour cache for visibility data

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
    """Client for fetching weather data from Open-Meteo API. @zara

    V12.3: This client fetches raw Open-Meteo data. All blending happens
    through WeatherExpertBlender in the pipeline.
    """

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

        # Auto-save to file cache after API fetch
        self.auto_save_cache = True

        # V12.8.5: Visibility cache (separate from ECMWF data)
        self._visibility_cache: Dict[str, Dict[int, float]] = {}  # date -> hour -> visibility_m
        self._visibility_cache_time: Optional[datetime] = None

        _LOGGER.info(
            f"OpenMeteoClient initialized "
            f"(lat={latitude:.4f}, lon={longitude:.4f})"
            f"{' with persistent cache' if cache_file else ''}"
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
            # Load existing cache to preserve historical data
            existing_forecast: Dict[str, Dict[str, Dict[str, Any]]] = {}
            if self._cache_file.exists():
                try:
                    async with aiofiles.open(self._cache_file, 'r') as f:
                        content = await f.read()
                        existing_data = json.loads(content)
                        existing_forecast = existing_data.get("forecast", {})
                        _LOGGER.debug(f"Loaded {len(existing_forecast)} existing days from cache")
                except Exception as e:
                    _LOGGER.debug(f"Could not load existing cache: {e}")

            forecast_by_date: Dict[str, Dict[str, Dict[str, Any]]] = existing_forecast.copy()

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

                hour_entry = {
                    "temperature": entry.get("temperature"),
                    "humidity": entry.get("humidity"),
                    "cloud_cover": entry.get("cloud_cover"),
                    "cloud_cover_low": entry.get("cloud_cover_low"),
                    "cloud_cover_mid": entry.get("cloud_cover_mid"),
                    "cloud_cover_high": entry.get("cloud_cover_high"),
                    "precipitation": entry.get("precipitation"),
                    "wind_speed": entry.get("wind_speed"),
                    "pressure": entry.get("pressure"),
                    "direct_radiation": direct_rad,
                    "diffuse_radiation": diffuse_rad,
                    "ghi": ghi,
                    "global_tilted_irradiance": entry.get("global_tilted_irradiance"),
                    "visibility_m": entry.get("visibility_m"),  # V12.8: For fog detection
                }

                # Include source tracking for diagnostics
                hour_entry["source"] = entry.get("source", "open-meteo")

                forecast_by_date[date_str][str(hour)] = hour_entry

            # Remove data older than retention period
            cutoff_date = (datetime.now() - timedelta(days=OPEN_METEO_HISTORY_RETENTION_DAYS)).strftime("%Y-%m-%d")
            dates_to_remove = [d for d in forecast_by_date.keys() if d < cutoff_date]
            for old_date in dates_to_remove:
                del forecast_by_date[old_date]
                _LOGGER.debug(f"Removed old forecast data for {old_date}")

            if dates_to_remove:
                _LOGGER.info(f"Cleaned up {len(dates_to_remove)} old days from Open-Meteo cache")

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

            # CRITICAL FIX: Also update in-memory cache and timestamp!
            # This ensures get_hourly_forecast() returns the blended data
            # instead of fetching fresh (unblended) data from the API.
            self._cache["hourly_forecast"] = hourly_data
            self._cache_time = datetime.now()

            _LOGGER.info(
                f"Saved Open-Meteo cache: {len(hourly_data)} hours, "
                f"{len(forecast_by_date)} days (in-memory cache also updated)"
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

        # Last resort: Try to load from file cache (e.g., after HA restart when in-memory is empty)
        if self._cache_file and self._cache_file.exists():
            try:
                await self._load_file_cache()
                cached = self._cache.get("hourly_forecast", [])
                if cached:
                    cache_info = self._get_cache_source_info()
                    _LOGGER.info(
                        f"Open-Meteo API unavailable, restored from file cache "
                        f"({len(cached)} hours, age: {cache_info.get('age_hours', 0):.1f}h)"
                    )
                    return cached
            except Exception as e:
                _LOGGER.warning(f"Could not restore Open-Meteo file cache: {e}")

        _LOGGER.warning(
            f"Open-Meteo: No cache available and API failed "
            f"({self._consecutive_failures} consecutive failures)."
        )
        return None

    async def get_raw_forecast_for_blending(self, hours: int = 72) -> Optional[List[Dict[str, Any]]]:
        """Get RAW Open-Meteo data for blending - bypasses in-memory cache.

        V12.3: Used by WeatherExpertBlender to get fresh Open-Meteo data.
        Either fetches from API or falls back to file cache.

        Returns:
            Raw Open-Meteo forecast data
        """
        # First try: Fresh API data (always raw)
        if self._check_rate_limit_budget():
            api_result = await self._fetch_from_api(hours)
            if api_result:
                self._consecutive_failures = 0
                self._last_api_error = None
                _LOGGER.debug("Blender: Got fresh raw data from Open-Meteo API")
                return api_result
            self._consecutive_failures += 1

        # Fallback: Load from FILE cache and extract raw values from blend_info
        # The file cache contains blend_info with original open_meteo_cloud values
        if self._cache_file and self._cache_file.exists():
            try:
                await self._load_file_cache()
                cached = self._cache.get("hourly_forecast", [])
                if cached:
                    # Extract raw Open-Meteo values from blend_info if available
                    raw_data = []
                    for entry in cached:
                        raw_entry = entry.copy()
                        blend_info = entry.get("blend_info", {})
                        # If blend_info has original open_meteo_cloud, use that
                        if "open_meteo_cloud" in blend_info:
                            raw_entry["cloud_cover"] = blend_info["open_meteo_cloud"]
                            raw_entry["_restored_from_blend_info"] = True
                        raw_data.append(raw_entry)

                    _LOGGER.debug(
                        f"Blender: Using file cache with restored raw values "
                        f"({len(raw_data)} hours)"
                    )
                    return raw_data
            except Exception as e:
                _LOGGER.warning(f"Could not load file cache for blending: {e}")

        _LOGGER.warning("Blender: No raw Open-Meteo data available")
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
                    "cloud_cover_low",
                    "cloud_cover_mid",
                    "cloud_cover_high",
                    "precipitation",
                    "wind_speed_10m",
                    "pressure_msl",
                    "direct_radiation",
                    "diffuse_radiation",
                    "global_tilted_irradiance",
                    "shortwave_radiation",
                    "visibility",  # V12.8: For fog detection (in meters)
                ]),
                "daily": ",".join([
                    "sunrise",
                    "sunset",
                    "daylight_duration",
                ]),
                "timezone": "auto",
                "forecast_days": 3,
                "models": "ecmwf_ifs025",
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

                if self.auto_save_cache:
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

    async def _fetch_visibility_from_default_model(self) -> bool:
        """V12.8.5: Fetch visibility data from Open-Meteo default model.

        ECMWF model doesn't provide visibility data, so we fetch it separately
        from the default model (which uses best_match for the location).

        This is a lightweight request that only fetches visibility.
        """
        try:
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "hourly": "visibility",
                "timezone": "auto",
                "forecast_days": 3,
                # No "models" parameter = use default best_match model
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    OPEN_METEO_BASE_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=OPEN_METEO_TIMEOUT)
                ) as response:
                    if response.status != 200:
                        _LOGGER.debug(f"Open-Meteo visibility API returned status {response.status}")
                        return False

                    data = await response.json()

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            visibility_values = hourly.get("visibility", [])

            if not times or not visibility_values:
                _LOGGER.debug("Open-Meteo visibility response has no data")
                return False

            self._visibility_cache.clear()
            valid_count = 0

            for i, time_str in enumerate(times):
                if i >= len(visibility_values):
                    break

                visibility = visibility_values[i]
                if visibility is None:
                    continue

                dt = datetime.fromisoformat(time_str)
                date_str = dt.date().isoformat()
                hour = dt.hour

                if date_str not in self._visibility_cache:
                    self._visibility_cache[date_str] = {}
                self._visibility_cache[date_str][hour] = float(visibility)
                valid_count += 1

            self._visibility_cache_time = datetime.now()

            _LOGGER.debug(
                f"Open-Meteo: Fetched {valid_count} visibility values from default model "
                f"(separate from ECMWF)"
            )
            return valid_count > 0

        except Exception as e:
            _LOGGER.debug(f"Open-Meteo visibility fetch error: {e}")
            return False

    def _is_visibility_cache_valid(self) -> bool:
        """Check if visibility cache is still valid."""
        if not self._visibility_cache_time or not self._visibility_cache:
            return False
        age = (datetime.now() - self._visibility_cache_time).total_seconds()
        return age < OPEN_METEO_VISIBILITY_CACHE_DURATION

    async def get_visibility(self, date: str, hour: int) -> Optional[float]:
        """V12.8.5: Get visibility in meters for fog detection.

        Visibility is fetched separately from the default Open-Meteo model
        because ECMWF doesn't provide visibility data.

        Args:
            date: Date string (YYYY-MM-DD)
            hour: Hour (0-23)

        Returns:
            Visibility in meters, or None if not available
        """
        # Check cache first
        if self._is_visibility_cache_valid():
            if date in self._visibility_cache and hour in self._visibility_cache[date]:
                return self._visibility_cache[date][hour]

        # Fetch fresh visibility data if cache is stale/empty
        if not self._is_visibility_cache_valid():
            await self._fetch_visibility_from_default_model()

        # Return from cache (may be freshly fetched)
        if date in self._visibility_cache and hour in self._visibility_cache[date]:
            return self._visibility_cache[date][hour]

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

                cloud_low = self._safe_get(hourly, "cloud_cover_low", i)
                cloud_mid = self._safe_get(hourly, "cloud_cover_mid", i)
                cloud_high = self._safe_get(hourly, "cloud_cover_high", i)

                # V12.8: Get visibility for fog detection (in meters)
                visibility_m = self._safe_get(hourly, "visibility", i)

                hour_data = {
                    "datetime": dt,
                    "date": dt.date().isoformat(),
                    "hour": dt.hour,
                    "temperature": self._safe_get(hourly, "temperature_2m", i),
                    "humidity": self._safe_get(hourly, "relative_humidity_2m", i),
                    "cloud_cover": self._safe_get(hourly, "cloud_cover", i),
                    "cloud_cover_low": cloud_low,
                    "cloud_cover_mid": cloud_mid,
                    "cloud_cover_high": cloud_high,
                    "precipitation": self._safe_get(hourly, "precipitation", i),
                    "wind_speed": self._safe_get(hourly, "wind_speed_10m", i),
                    "pressure": self._safe_get(hourly, "pressure_msl", i),
                    "direct_radiation": direct_rad,
                    "diffuse_radiation": diffuse_rad,
                    "ghi": ghi,
                    "global_tilted_irradiance": self._safe_get(hourly, "global_tilted_irradiance", i),
                    "visibility_m": visibility_m,  # V12.8: For fog detection
                    "source": "open-meteo",
                    "blend_info": {
                        "sources": ["open-meteo"],
                        "trigger": "api_fetch",
                        "open_meteo_cloud": self._safe_get(hourly, "cloud_cover", i),
                        "cloud_layers": {
                            "low": cloud_low,
                            "mid": cloud_mid,
                            "high": cloud_high,
                        },
                        "visibility_m": visibility_m,  # V12.8: For fog detection
                    },
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
