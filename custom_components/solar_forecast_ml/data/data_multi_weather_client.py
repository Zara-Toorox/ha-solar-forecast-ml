"""Multi-Weather Client - Combines Open-Meteo with wttr.in (WWO) for improved cloud forecasts V12.0.0 @zara

Trigger: Only fetches additional sources when Open-Meteo cloud_cover > 50%
Learning: Tracks which source was more accurate and adjusts weights based on IST values

Architecture:
    [Open-Meteo API]    [wttr.in API (WWO)]
           |                    |
           v                    v
    [OpenMeteoClient]    [WttrInClient]
           |                    |
           +--------+-----------+
                    |
                    v
          [MultiWeatherBlender]  <- Weighted blending
                    |
                    v
         [open_meteo_cache.json]  <- Existing format (backwards compatible)
                    |
                    v
         [WeatherForecastCorrector] (unchanged!)

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

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from .data_io import DataManagerIO
from .data_open_meteo_client import OpenMeteoClient
from .data_validator import DataValidator

_LOGGER = logging.getLogger(__name__)

# Trigger threshold: Only fetch alternative sources if Open-Meteo cloud_cover > this
CLOUD_TRIGGER_THRESHOLD = 50.0

# wttr.in API settings
WTTR_IN_TIMEOUT = 20
WTTR_IN_MAX_RETRIES = 2

# wttr.in cache settings
WTTR_CACHE_MAX_AGE = 6 * 3600  # 6 hours - normal cache validity
WTTR_CACHE_FALLBACK_AGE = 12 * 3600  # 12 hours - use stale cache if API fails

# Default weights (before learning) - wttr.in slightly preferred as it's often more accurate
DEFAULT_WEIGHTS = {
    "open_meteo": 0.35,
    "wwo": 0.65,
}

# Extreme cloud threshold: When Open-Meteo reports >= this, boost wttr.in weight
EXTREME_CLOUD_THRESHOLD = 95.0
# Minimum difference for boost: Only boost if wttr.in is at least this much lower
EXTREME_CLOUD_DIFF_MIN = 30.0
# Boosted weight for wttr.in when extreme cloud discrepancy detected
EXTREME_CLOUD_WWO_WEIGHT = 0.80

# Learning parameters
MIN_HOURS_FOR_LEARNING = 4  # Minimum hours with actual data needed
SMOOTHING_FACTOR = 0.3  # How fast to adapt (0.3 = 30% new, 70% old)
MIN_ERROR = 5.0  # Minimum error to prevent division issues


class WttrInClient:
    """Client for fetching weather data from wttr.in (WWO backend) - No API key needed @zara

    Features:
    - 6-hour cache to reduce API calls (wttr.in has 3h resolution anyway)
    - 12-hour fallback cache when API is unavailable
    - Automatic cache creation and validation
    """

    def __init__(self, latitude: float, longitude: float, cache_file: Optional[Path] = None):
        """Initialize wttr.in client.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            cache_file: Path to cache file (optional)
        """
        self.latitude = latitude
        self.longitude = longitude
        self.cache_file = cache_file
        self._last_error: Optional[str] = None
        self._consecutive_failures: int = 0
        self._cache_data: Optional[Dict[str, Any]] = None
        self._cache_loaded: bool = False

        _LOGGER.info(
            f"WttrInClient initialized (lat={latitude:.4f}, lon={longitude:.4f}) "
            f"- Uses World Weather Online data via wttr.in"
            f"{' with cache' if cache_file else ''}"
        )

    async def get_forecast(self, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Fetch 3-day forecast from wttr.in in JSON format.

        Uses a 6-hour cache to reduce API calls. Falls back to 12-hour stale
        cache if API is unavailable.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            Dict with hourly forecasts by date or None if failed

        Format:
            {
                "2025-12-06": {
                    0: {"cloud_cover": 35, "temperature": 5, ...},
                    3: {"cloud_cover": 40, ...},
                    ...
                },
                ...
            }
        """
        # Step 1: Check cache (unless force refresh)
        if not force_refresh:
            cached = await self._get_from_cache()
            if cached is not None:
                return cached

        # Step 2: Fetch from API
        api_result = await self._fetch_from_api()

        if api_result:
            # Save to cache
            await self._save_to_cache(api_result)
            return api_result

        # Step 3: API failed - try stale cache (up to 12 hours)
        stale_cached = await self._get_from_cache(allow_stale=True)
        if stale_cached is not None:
            _LOGGER.info("Using stale wttr.in cache (API unavailable)")
            return stale_cached

        return None

    async def _fetch_from_api(self) -> Optional[Dict[str, Any]]:
        """Fetch data from wttr.in API."""
        for attempt in range(WTTR_IN_MAX_RETRIES):
            try:
                url = f"https://wttr.in/{self.latitude},{self.longitude}?format=j1"

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=WTTR_IN_TIMEOUT),
                        headers={"User-Agent": "SolarForecastML/1.0"}
                    ) as response:
                        if response.status != 200:
                            self._last_error = f"HTTP {response.status}"
                            _LOGGER.warning(f"wttr.in API returned status {response.status}")
                            continue

                        data = await response.json()

                result = self._parse_response(data)

                if result:
                    self._consecutive_failures = 0
                    self._last_error = None
                    return result

            except asyncio.TimeoutError:
                self._last_error = "Timeout"
                _LOGGER.debug(f"wttr.in API request timed out (attempt {attempt + 1})")
            except aiohttp.ClientError as e:
                self._last_error = f"Connection: {e}"
                _LOGGER.debug(f"wttr.in API connection error (attempt {attempt + 1}): {e}")
            except Exception as e:
                self._last_error = f"Error: {e}"
                _LOGGER.debug(f"wttr.in API error (attempt {attempt + 1}): {e}")

        self._consecutive_failures += 1
        # Only log warning after all retries failed - single retry failures are normal
        _LOGGER.debug(
            f"wttr.in fetch failed after {WTTR_IN_MAX_RETRIES} attempts "
            f"({self._consecutive_failures} consecutive failures) - using cache fallback"
        )
        return None

    async def _get_from_cache(self, allow_stale: bool = False) -> Optional[Dict[str, Any]]:
        """Get forecast from cache if valid.

        Args:
            allow_stale: If True, accept cache up to 12 hours old

        Returns:
            Cached forecast data or None
        """
        if not self.cache_file:
            return None

        try:
            # Check file existence in thread to avoid blocking
            exists = await asyncio.to_thread(self.cache_file.exists)
            if not exists:
                return None

            # Load cache if not already loaded
            if not self._cache_loaded:
                self._cache_data = await asyncio.to_thread(self._read_cache_file)
                self._cache_loaded = True

            if not self._cache_data:
                return None

            # Check cache age
            fetched_at_str = self._cache_data.get("metadata", {}).get("fetched_at")
            if not fetched_at_str:
                return None

            fetched_at = datetime.fromisoformat(fetched_at_str)
            age_seconds = (datetime.now() - fetched_at).total_seconds()

            max_age = WTTR_CACHE_FALLBACK_AGE if allow_stale else WTTR_CACHE_MAX_AGE

            if age_seconds > max_age:
                if not allow_stale:
                    _LOGGER.debug(
                        f"wttr.in cache expired (age={age_seconds/3600:.1f}h, max={max_age/3600:.0f}h)"
                    )
                return None

            forecast = self._cache_data.get("forecast")
            if forecast:
                cache_type = "stale " if allow_stale and age_seconds > WTTR_CACHE_MAX_AGE else ""
                _LOGGER.debug(
                    f"Using {cache_type}wttr.in cache (age={age_seconds/3600:.1f}h)"
                )
                return forecast

        except json.JSONDecodeError as e:
            _LOGGER.warning(f"wttr.in cache corrupted: {e}")
            self._cache_data = None
            self._cache_loaded = False
        except Exception as e:
            _LOGGER.warning(f"Error reading wttr.in cache: {e}")

        return None

    def _read_cache_file(self) -> Optional[Dict[str, Any]]:
        """Read cache file synchronously (called via asyncio.to_thread)."""
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    async def _save_to_cache(self, forecast: Dict[str, Any]) -> bool:
        """Save forecast to cache file.

        Args:
            forecast: Forecast data to cache

        Returns:
            True if saved successfully
        """
        if not self.cache_file:
            return False

        try:
            cache_data = {
                "version": "1.0",
                "metadata": {
                    "fetched_at": datetime.now().isoformat(),
                    "source": "wttr.in",
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "cache_max_age_hours": WTTR_CACHE_MAX_AGE / 3600,
                },
                "forecast": forecast,
            }

            # Write in thread to avoid blocking
            await asyncio.to_thread(self._write_cache_file, cache_data)

            # Update in-memory cache
            self._cache_data = cache_data
            self._cache_loaded = True

            _LOGGER.info(f"Saved wttr.in forecast to cache ({len(forecast)} days)")
            return True

        except Exception as e:
            _LOGGER.warning(f"Error saving wttr.in cache: {e}")
            return False

    def _write_cache_file(self, cache_data: Dict[str, Any]) -> None:
        """Write cache file synchronously (called via asyncio.to_thread)."""
        # Ensure directory exists
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write
        temp_file = self.cache_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        temp_file.replace(self.cache_file)

    def invalidate_cache(self) -> None:
        """Invalidate the in-memory cache."""
        self._cache_data = None
        self._cache_loaded = False

    def _parse_response(self, data: Dict[str, Any]) -> Optional[Dict[str, Dict[int, Dict[str, Any]]]]:
        """Parse wttr.in JSON response into hourly format by date.

        wttr.in returns data in 3-hour intervals (0, 3, 6, 9, 12, 15, 18, 21).
        We store these directly and interpolate when blending.

        Structure input:
        {
            "weather": [
                {
                    "date": "2025-12-06",
                    "hourly": [
                        {"time": "0", "cloudcover": "35", "tempC": "5", ...},
                        {"time": "300", "cloudcover": "40", ...},  # 3:00
                        ...
                    ]
                },
                ...
            ]
        }
        """
        try:
            result: Dict[str, Dict[int, Dict[str, Any]]] = {}

            weather_days = data.get("weather", [])
            if not weather_days:
                _LOGGER.warning("wttr.in response has no weather data")
                return None

            total_hours = 0

            for day in weather_days:
                date_str = day.get("date")
                if not date_str:
                    continue

                hourly_data = day.get("hourly", [])

                if date_str not in result:
                    result[date_str] = {}

                for hour_data in hourly_data:
                    # wttr.in time is in format "0", "300", "600" etc (HHMM without colon)
                    time_val = int(hour_data.get("time", "0"))
                    hour = time_val // 100

                    cloud_cover = self._safe_float(hour_data.get("cloudcover"), 50)
                    temp = self._safe_float(hour_data.get("tempC"), 15)
                    humidity = self._safe_float(hour_data.get("humidity"), 70)
                    wind_speed_kmh = self._safe_float(hour_data.get("windspeedKmph"), 10)
                    precip = self._safe_float(hour_data.get("precipMM"), 0)
                    pressure = self._safe_float(hour_data.get("pressure"), 1013)

                    result[date_str][hour] = {
                        "cloud_cover": cloud_cover,
                        "temperature": temp,
                        "humidity": humidity,
                        "wind_speed": wind_speed_kmh / 3.6,  # km/h to m/s
                        "precipitation": precip,
                        "pressure": pressure,
                        "source": "wttr.in-wwo",
                    }
                    total_hours += 1

            _LOGGER.info(
                f"Parsed wttr.in data: {len(result)} days, {total_hours} hour entries"
            )
            return result

        except Exception as e:
            _LOGGER.error(f"Error parsing wttr.in response: {e}", exc_info=True)
            return None

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        """Safely convert value to float."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error


class MultiWeatherBlender(DataManagerIO):
    """Blends multiple weather sources with learned weights @zara

    Strategy:
    1. Always fetch Open-Meteo (primary source)
    2. If Open-Meteo cloud_cover > 50% for any hour, fetch wttr.in
    3. Blend cloud_cover using learned weights
    4. Write to existing open_meteo_cache.json format (backwards compatible)
    """

    def __init__(
        self,
        hass,
        latitude: float,
        longitude: float,
        data_dir: Path,
        open_meteo_client: OpenMeteoClient,
    ):
        """Initialize MultiWeatherBlender.

        Args:
            hass: Home Assistant instance
            latitude: Location latitude
            longitude: Location longitude
            data_dir: Data directory path
            open_meteo_client: Existing OpenMeteoClient instance
        """
        super().__init__(hass, data_dir)

        self.latitude = latitude
        self.longitude = longitude

        # Weights file for learned blending
        self.weights_file = data_dir / "data" / "weather_source_weights.json"

        # wttr.in cache file
        self.wttr_cache_file = data_dir / "data" / "wttr_in_cache.json"

        # Use existing Open-Meteo client
        self.open_meteo_client = open_meteo_client

        # CRITICAL: Disable auto-save in OpenMeteoClient
        # The Blender manages the file cache to ensure blend_info is preserved
        self.open_meteo_client.auto_save_cache = False

        # Create wttr.in client with cache
        self.wttr_client = WttrInClient(latitude, longitude, cache_file=self.wttr_cache_file)

        # Cached weights
        self._weights: Dict[str, float] = DEFAULT_WEIGHTS.copy()
        self._weights_loaded = False

        # Stats for current session
        self._blend_stats = {
            "total_fetches": 0,
            "blended_fetches": 0,
            "wttr_failures": 0,
        }

        _LOGGER.info(
            f"MultiWeatherBlender initialized (lat={latitude:.4f}, lon={longitude:.4f}) "
            f"- Blends Open-Meteo + wttr.in (WWO) for cloud_cover > {CLOUD_TRIGGER_THRESHOLD}%"
        )
        _LOGGER.debug("OpenMeteoClient auto_save_cache disabled - Blender manages file cache")

    async def async_init(self) -> bool:
        """Initialize blender, validate cache files, and load weights."""
        # Validate/create wttr.in cache file (run in executor to avoid blocking)
        cache_result = await self.hass.async_add_executor_job(
            DataValidator.validate_and_create_wttr_cache,
            self.wttr_cache_file,
            self.latitude,
            self.longitude,
        )
        if cache_result.get("created"):
            _LOGGER.info("wttr.in cache file created")
        elif cache_result.get("repaired"):
            _LOGGER.info("wttr.in cache file repaired")

        await self._load_weights()

        # Migrate existing cache entries without blend_info
        await self._migrate_cache_blend_info()

        return True

    async def _migrate_cache_blend_info(self) -> None:
        """Migrate existing cache entries to include blend_info.

        This ensures weight learning can work even with old cache data
        that was saved before the blending feature was active.
        """
        try:
            cache_file = self.open_meteo_client._cache_file
            if not cache_file or not cache_file.exists():
                return

            # Read existing cache
            import json

            def _read_cache():
                with open(cache_file, 'r') as f:
                    return json.load(f)

            cache_data = await self.hass.async_add_executor_job(_read_cache)

            if not cache_data or "forecast" not in cache_data:
                return

            forecast = cache_data.get("forecast", {})
            migrated_count = 0
            total_count = 0

            for date_str, hours in forecast.items():
                for hour_str, hour_data in hours.items():
                    total_count += 1

                    # Add blend_info if missing
                    if "blend_info" not in hour_data:
                        hour_data["blend_info"] = {
                            "sources": ["open-meteo"],
                            "trigger": "migrated_legacy",
                        }
                        migrated_count += 1

                    # Add source if missing
                    if "source" not in hour_data:
                        hour_data["source"] = "open-meteo-only"

            if migrated_count > 0:
                # Write back migrated cache
                def _write_cache():
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    temp_file = cache_file.with_suffix('.tmp')
                    with open(temp_file, 'w') as f:
                        json.dump(cache_data, f, indent=2)
                    temp_file.replace(cache_file)

                await self.hass.async_add_executor_job(_write_cache)

                _LOGGER.info(
                    f"Migrated {migrated_count}/{total_count} cache entries with blend_info - "
                    "weight learning can now track accuracy"
                )
            else:
                _LOGGER.debug(
                    f"Cache migration not needed - all {total_count} entries already have blend_info"
                )

        except Exception as e:
            _LOGGER.warning(f"Cache migration failed (non-critical): {e}")

    async def _load_weights(self) -> bool:
        """Load learned weights from file."""
        try:
            if await self._file_exists(self.weights_file):
                data = await self._read_json_file(self.weights_file, None)
                if data and "weights" in data:
                    self._weights = data["weights"]
                    self._weights_loaded = True

                    learning_info = data.get("learning_metadata", {})
                    last_date = learning_info.get("last_learning_date", "never")

                    _LOGGER.info(
                        f"Loaded weather source weights: "
                        f"open_meteo={self._weights.get('open_meteo', 0.5):.1%}, "
                        f"wwo={self._weights.get('wwo', 0.5):.1%} "
                        f"(learned from {last_date})"
                    )
                    return True

            # No weights file - create with defaults
            _LOGGER.info(
                f"No learned weights found - creating with defaults: "
                f"open_meteo={DEFAULT_WEIGHTS['open_meteo']:.0%}, "
                f"wwo={DEFAULT_WEIGHTS['wwo']:.0%}"
            )

            # Save default weights to file so it exists
            await self._save_default_weights()
            return False

        except Exception as e:
            _LOGGER.warning(f"Error loading weights: {e}")
            return False

    async def _save_default_weights(self) -> None:
        """Save default weights to file.

        This ensures the weights file exists and can be updated by learning.
        """
        try:
            weights_data = {
                "version": "1.0",
                "weights": DEFAULT_WEIGHTS.copy(),
                "learning_metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "last_learning_date": None,
                    "note": "Default weights - will be updated by weight learning at 23:30",
                },
            }

            await self._atomic_write_json(self.weights_file, weights_data)
            _LOGGER.info(f"Created weather_source_weights.json with defaults")

        except Exception as e:
            _LOGGER.warning(f"Failed to save default weights: {e}")

    async def get_blended_forecast(
        self,
        hours: int = 72
    ) -> Optional[List[Dict[str, Any]]]:
        """Get blended forecast from multiple sources.

        Args:
            hours: Number of hours to forecast

        Returns:
            List of hourly forecasts with blended cloud_cover
        """
        self._blend_stats["total_fetches"] += 1

        # Step 1: Get RAW Open-Meteo data (not already-blended!)
        # CRITICAL: Use get_raw_forecast_for_blending() to prevent double-blending.
        # The regular get_hourly_forecast() may return already-blended data from
        # the in-memory cache, which would cause cloud values to be blended twice.
        open_meteo_data = await self.open_meteo_client.get_raw_forecast_for_blending(hours)

        if not open_meteo_data:
            _LOGGER.warning("Open-Meteo unavailable - cannot create blended forecast")
            return None

        # Step 2: Check if any hour has cloud_cover > threshold
        high_cloud_hours = [
            h for h in open_meteo_data
            if (h.get("cloud_cover") or 0) > CLOUD_TRIGGER_THRESHOLD
        ]

        if not high_cloud_hours:
            _LOGGER.debug(
                f"All Open-Meteo cloud_cover <= {CLOUD_TRIGGER_THRESHOLD}% - "
                "using Open-Meteo only (no blending needed)"
            )
            # Add source info and return as-is
            for entry in open_meteo_data:
                entry["source"] = "open-meteo-only"
                entry["blend_info"] = {"sources": ["open-meteo"], "trigger": "low_clouds"}
            return open_meteo_data

        _LOGGER.info(
            f"Open-Meteo shows {len(high_cloud_hours)}/{len(open_meteo_data)} hours "
            f"with cloud_cover > {CLOUD_TRIGGER_THRESHOLD}% - fetching wttr.in"
        )

        # Step 3: Fetch wttr.in (WWO) data
        wttr_data = await self.wttr_client.get_forecast()

        if not wttr_data:
            self._blend_stats["wttr_failures"] += 1
            _LOGGER.warning("wttr.in unavailable - using Open-Meteo only")
            for entry in open_meteo_data:
                entry["source"] = "open-meteo-only"
                entry["blend_info"] = {"sources": ["open-meteo"], "trigger": "wttr_failed"}
            return open_meteo_data

        # Step 4: Blend cloud_cover using weights
        self._blend_stats["blended_fetches"] += 1
        blended = self._blend_forecasts(open_meteo_data, wttr_data)

        return blended

    def _blend_forecasts(
        self,
        open_meteo: List[Dict[str, Any]],
        wttr: Dict[str, Dict[int, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Blend cloud_cover from multiple sources.

        Only blends cloud_cover - all other fields come from Open-Meteo.
        Uses interpolation for wttr.in's 3-hour data.

        Args:
            open_meteo: List of hourly forecasts from Open-Meteo
            wttr: Dict of forecasts from wttr.in {date: {hour: data}}

        Returns:
            List of blended forecasts
        """
        w_meteo = self._weights.get("open_meteo", 0.5)
        w_wwo = self._weights.get("wwo", 0.5)

        blended = []
        blended_count = 0

        for om_entry in open_meteo:
            result = om_entry.copy()
            date_str = om_entry.get("date")
            hour = om_entry.get("hour")
            om_cloud = om_entry.get("cloud_cover") or 0

            # Only blend if cloud_cover > threshold
            if om_cloud > CLOUD_TRIGGER_THRESHOLD and date_str in wttr:
                wwo_cloud = self._get_interpolated_cloud(wttr[date_str], hour)

                if wwo_cloud is not None:
                    # Check for extreme discrepancy: Open-Meteo says near 100% but wttr.in much lower
                    # This typically indicates Open-Meteo is wrong (common issue)
                    effective_w_meteo = w_meteo
                    effective_w_wwo = w_wwo
                    boost_applied = False

                    if (om_cloud >= EXTREME_CLOUD_THRESHOLD and
                        (om_cloud - wwo_cloud) >= EXTREME_CLOUD_DIFF_MIN):
                        # Boost wttr.in weight when Open-Meteo shows extreme clouds
                        # but wttr.in disagrees significantly
                        effective_w_wwo = EXTREME_CLOUD_WWO_WEIGHT
                        effective_w_meteo = 1.0 - EXTREME_CLOUD_WWO_WEIGHT
                        boost_applied = True

                    # Weighted blend
                    blended_cloud = (effective_w_meteo * om_cloud) + (effective_w_wwo * wwo_cloud)
                    blended_cloud = max(0.0, min(100.0, blended_cloud))

                    result["cloud_cover"] = round(blended_cloud, 1)
                    result["source"] = "blended"
                    result["blend_info"] = {
                        "sources": ["open-meteo", "wwo"],
                        "open_meteo_cloud": om_cloud,
                        "wwo_cloud": wwo_cloud,
                        "weights": {"open_meteo": effective_w_meteo, "wwo": effective_w_wwo},
                        "blended_cloud": blended_cloud,
                        "extreme_boost": boost_applied,
                    }
                    blended_count += 1

                    # Blending details logging removed to reduce log spam
                else:
                    result["source"] = "open-meteo-only"
                    result["blend_info"] = {"sources": ["open-meteo"], "trigger": "no_wwo_data"}
            else:
                result["source"] = "open-meteo-only"
                result["blend_info"] = {"sources": ["open-meteo"], "trigger": "low_clouds"}

            blended.append(result)

        _LOGGER.info(
            f"Blended {blended_count}/{len(blended)} hours using weights: "
            f"open_meteo={w_meteo:.1%}, wwo={w_wwo:.1%}"
        )

        return blended

    def _get_interpolated_cloud(
        self,
        day_data: Dict[int, Dict[str, Any]],
        target_hour: int
    ) -> Optional[float]:
        """Get cloud cover for a specific hour, interpolating if necessary.

        wttr.in provides data at 3-hour intervals (0, 3, 6, 9, 12, 15, 18, 21).
        For hours in between, we interpolate linearly.

        Args:
            day_data: Dict of {hour: data} for the day
            target_hour: Hour to get cloud cover for (0-23)

        Returns:
            Interpolated cloud cover or None if not available
        """
        if not day_data:
            return None

        # Convert keys to integers for comparison (keys may be strings like "0", "3", etc.)
        hour_mapping = {}
        for k, v in day_data.items():
            try:
                hour_mapping[int(k)] = v
            except (ValueError, TypeError):
                continue

        # Check if we have exact match
        if target_hour in hour_mapping:
            return hour_mapping[target_hour].get("cloud_cover")

        # Find surrounding hours (wttr.in uses 0, 3, 6, 9, 12, 15, 18, 21)
        available_hours = sorted(hour_mapping.keys())

        if not available_hours:
            return None

        # Find lower and upper bounds
        lower_hour = None
        upper_hour = None

        for h in available_hours:
            if h <= target_hour:
                lower_hour = h
            if h >= target_hour and upper_hour is None:
                upper_hour = h
                break

        # Handle edge cases
        if lower_hour is None:
            lower_hour = available_hours[0]
        if upper_hour is None:
            upper_hour = available_hours[-1]

        # If same, return directly
        if lower_hour == upper_hour:
            return hour_mapping[lower_hour].get("cloud_cover")

        # Linear interpolation
        lower_cloud = hour_mapping[lower_hour].get("cloud_cover")
        upper_cloud = hour_mapping[upper_hour].get("cloud_cover")

        if lower_cloud is None or upper_cloud is None:
            return lower_cloud or upper_cloud

        # Calculate interpolation factor
        factor = (target_hour - lower_hour) / (upper_hour - lower_hour)
        interpolated = lower_cloud + factor * (upper_cloud - lower_cloud)

        return round(interpolated, 1)

    async def update_and_save_cache(self) -> bool:
        """Fetch blended forecast and save to open_meteo_cache.json.

        This populates the existing cache format so downstream components
        (WeatherForecastCorrector, etc.) work without modification.

        IMPORTANT: This method ensures:
        1. All cache entries have blend_info (for WeatherSourceLearner)
        2. weather_source_weights.json exists (for weight learning)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure weights file exists (critical for weight learning)
            if not await self._file_exists(self.weights_file):
                _LOGGER.info("Creating weather_source_weights.json (first run)")
                await self._save_default_weights()

            blended = await self.get_blended_forecast(hours=72)

            if not blended:
                _LOGGER.warning("No blended forecast available to save")
                return False

            # Ensure all entries have blend_info before saving
            for entry in blended:
                if "blend_info" not in entry:
                    entry["blend_info"] = {
                        "sources": ["open-meteo"],
                        "trigger": "blender_default",
                        "open_meteo_cloud": entry.get("cloud_cover"),
                    }
                if "source" not in entry:
                    entry["source"] = "open-meteo"

            # Save to file cache with blend_info preserved
            await self.open_meteo_client._save_file_cache(blended)

            blended_count = len([b for b in blended if b.get("source") == "blended"])
            blend_info_count = len([b for b in blended if "blend_info" in b])

            _LOGGER.info(
                f"Saved blended forecast to cache: {len(blended)} hours, "
                f"{blended_count} blended, {blend_info_count} with blend_info"
            )
            return True

        except Exception as e:
            _LOGGER.error(f"Error updating blended cache: {e}", exc_info=True)
            return False

    def get_current_weights(self) -> Dict[str, float]:
        """Get current blending weights."""
        return self._weights.copy()

    def get_blend_stats(self) -> Dict[str, Any]:
        """Get blending statistics for diagnostics."""
        return {
            **self._blend_stats,
            "weights": self._weights.copy(),
            "weights_loaded": self._weights_loaded,
        }


class WeatherSourceLearner(DataManagerIO):
    """Learns optimal weights for weather sources based on actual observations @zara

    Learning Strategy:
    1. At end of day (23:30), compare forecasted cloud_cover with actual (from sensors)
    2. Calculate MAE (Mean Absolute Error) for each source
    3. Update weights inversely proportional to error

    Weight Update Formula:
        new_weight_i = (1 / error_i) / sum(1 / error_j for all j)

    With smoothing to prevent sudden jumps:
        smoothed_weight = 0.7 * old_weight + 0.3 * new_weight
    """

    def __init__(self, hass, data_dir: Path):
        """Initialize WeatherSourceLearner.

        Args:
            hass: Home Assistant instance
            data_dir: Data directory path
        """
        super().__init__(hass, data_dir)

        self.weights_file = data_dir / "data" / "weather_source_weights.json"
        self.learning_history_file = data_dir / "stats" / "weather_source_learning.json"
        self.cache_file = data_dir / "data" / "open_meteo_cache.json"
        self.actual_weather_file = data_dir / "stats" / "hourly_weather_actual.json"

        _LOGGER.info("WeatherSourceLearner initialized")

    async def learn_from_day(self, date_str: str) -> Dict[str, Any]:
        """Learn weights from a completed day's data.

        Args:
            date_str: Date to learn from (YYYY-MM-DD)

        Returns:
            Dict with learning results including new weights, MAE values, etc.
        """
        result = {
            "success": False,
            "date": date_str,
            "reason": None,
            "mae_open_meteo": None,
            "mae_wwo": None,
            "old_weights": None,
            "new_weights": None,
            "comparison_hours": 0,
        }

        try:
            # Step 1: Load forecasted values with blend_info from cache
            forecasts = await self._load_forecast_data(date_str)

            if not forecasts:
                result["reason"] = "No forecast data with blend_info found"
                _LOGGER.debug(f"No blended forecast data for {date_str}")
                return result

            # Step 2: Load actual cloud_cover from hourly_weather_actual.json
            actuals = await self._load_actual_weather(date_str)

            if len(actuals) < MIN_HOURS_FOR_LEARNING:
                result["reason"] = f"Only {len(actuals)} hours of actual data (need {MIN_HOURS_FOR_LEARNING})"
                _LOGGER.debug(result["reason"])
                return result

            # Step 3: Calculate errors for each source
            errors_by_source = {
                "open_meteo": [],
                "wwo": [],
            }

            for hour_str, actual_data in actuals.items():
                actual_cloud = actual_data.get("cloud_cover_percent")

                if actual_cloud is None:
                    continue

                hour = int(hour_str)
                forecast = forecasts.get(hour)

                if not forecast or "blend_info" not in forecast:
                    continue

                blend_info = forecast["blend_info"]

                # Get individual source predictions
                if "open_meteo_cloud" in blend_info:
                    om_error = abs(blend_info["open_meteo_cloud"] - actual_cloud)
                    errors_by_source["open_meteo"].append(om_error)

                if "wwo_cloud" in blend_info:
                    wwo_error = abs(blend_info["wwo_cloud"] - actual_cloud)
                    errors_by_source["wwo"].append(wwo_error)

            # Step 4: Calculate MAE for each source
            if not errors_by_source["open_meteo"] or not errors_by_source["wwo"]:
                result["reason"] = "Not enough comparison data (need both sources)"
                _LOGGER.debug(f"Not enough comparison data for {date_str}")
                return result

            mae_open_meteo = sum(errors_by_source["open_meteo"]) / len(errors_by_source["open_meteo"])
            mae_wwo = sum(errors_by_source["wwo"]) / len(errors_by_source["wwo"])

            result["mae_open_meteo"] = round(mae_open_meteo, 2)
            result["mae_wwo"] = round(mae_wwo, 2)
            result["comparison_hours"] = len(errors_by_source["open_meteo"])

            _LOGGER.info(
                f"Weather source accuracy for {date_str}: "
                f"Open-Meteo MAE={mae_open_meteo:.1f}%, WWO MAE={mae_wwo:.1f}% "
                f"({result['comparison_hours']} hours compared)"
            )

            # Step 5: Calculate new weights (inverse of error)
            # Lower error = higher weight
            mae_open_meteo_safe = max(mae_open_meteo, MIN_ERROR)
            mae_wwo_safe = max(mae_wwo, MIN_ERROR)

            inv_om = 1.0 / mae_open_meteo_safe
            inv_wwo = 1.0 / mae_wwo_safe
            total_inv = inv_om + inv_wwo

            new_weight_om = inv_om / total_inv
            new_weight_wwo = inv_wwo / total_inv

            # Step 6: Load old weights and apply smoothing
            old_weights = await self._load_weights()
            result["old_weights"] = old_weights.copy()

            smoothed_om = (
                (1 - SMOOTHING_FACTOR) * old_weights.get("open_meteo", 0.5) +
                SMOOTHING_FACTOR * new_weight_om
            )
            smoothed_wwo = (
                (1 - SMOOTHING_FACTOR) * old_weights.get("wwo", 0.5) +
                SMOOTHING_FACTOR * new_weight_wwo
            )

            # Normalize to ensure they sum to 1
            total = smoothed_om + smoothed_wwo
            smoothed_om /= total
            smoothed_wwo /= total

            # Step 7: Save new weights
            new_weights = {
                "open_meteo": round(smoothed_om, 4),
                "wwo": round(smoothed_wwo, 4),
            }
            result["new_weights"] = new_weights

            weights_data = {
                "version": "1.0",
                "weights": new_weights,
                "learning_metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "last_learning_date": date_str,
                    "last_mae": {
                        "open_meteo": round(mae_open_meteo, 2),
                        "wwo": round(mae_wwo, 2),
                    },
                    "comparison_hours": result["comparison_hours"],
                    "smoothing_factor": SMOOTHING_FACTOR,
                },
            }

            await self._atomic_write_json(self.weights_file, weights_data)

            # Save to learning history
            await self._save_learning_history(date_str, mae_open_meteo, mae_wwo, new_weights)

            result["success"] = True
            result["reason"] = "Weights updated successfully"

            _LOGGER.info(
                f"Updated weather source weights: "
                f"open_meteo={smoothed_om:.1%} (was {old_weights.get('open_meteo', 0.5):.1%}), "
                f"wwo={smoothed_wwo:.1%} (was {old_weights.get('wwo', 0.5):.1%})"
            )

            return result

        except Exception as e:
            result["reason"] = f"Error: {str(e)}"
            _LOGGER.error(f"Error learning weights for {date_str}: {e}", exc_info=True)
            return result

    async def _load_forecast_data(self, date_str: str) -> Dict[int, Dict[str, Any]]:
        """Load forecast data with blend_info for the given date."""
        try:
            if not await self._file_exists(self.cache_file):
                return {}

            data = await self._read_json_file(self.cache_file, None)
            if not data:
                return {}

            day_forecast = data.get("forecast", {}).get(date_str, {})

            # Convert to {hour: data} format
            result = {}
            for hour_str, hour_data in day_forecast.items():
                try:
                    hour = int(hour_str)
                    if "blend_info" in hour_data:
                        result[hour] = hour_data
                except ValueError:
                    continue

            return result

        except Exception as e:
            _LOGGER.warning(f"Error loading forecast data: {e}")
            return {}

    async def _load_actual_weather(self, date_str: str) -> Dict[str, Dict[str, Any]]:
        """Load actual weather data for the given date."""
        try:
            if not await self._file_exists(self.actual_weather_file):
                return {}

            data = await self._read_json_file(self.actual_weather_file, None)
            if not data:
                return {}

            return data.get("hourly_data", {}).get(date_str, {})

        except Exception as e:
            _LOGGER.warning(f"Error loading actual weather: {e}")
            return {}

    async def _load_weights(self) -> Dict[str, float]:
        """Load current weights."""
        try:
            if await self._file_exists(self.weights_file):
                data = await self._read_json_file(self.weights_file, None)
                if data and "weights" in data:
                    return data["weights"]
        except Exception:
            pass

        return DEFAULT_WEIGHTS.copy()

    async def _save_learning_history(
        self,
        date_str: str,
        mae_om: float,
        mae_wwo: float,
        new_weights: Dict[str, float],
    ) -> None:
        """Save learning history for analysis."""
        try:
            history = await self._read_json_file(
                self.learning_history_file,
                {"version": "1.0", "daily_history": {}, "metadata": {}}
            )

            # Ensure daily_history exists (migration from old format)
            if "daily_history" not in history:
                history["daily_history"] = history.pop("history", {})

            history["daily_history"][date_str] = {
                "mae_open_meteo": round(mae_om, 2),
                "mae_wwo": round(mae_wwo, 2),
                "weights_after": new_weights,
                "learned_at": datetime.now().isoformat(),
            }

            # Update metadata
            history.setdefault("metadata", {})
            history["metadata"]["last_learning_run"] = datetime.now().isoformat()
            history["metadata"]["total_learning_days"] = len(history["daily_history"])

            # Keep only last 30 days
            dates = sorted(history["daily_history"].keys())
            if len(dates) > 30:
                for old_date in dates[:-30]:
                    del history["daily_history"][old_date]

            await self._atomic_write_json(self.learning_history_file, history)

        except Exception as e:
            _LOGGER.warning(f"Error saving learning history: {e}")

    async def get_learning_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent learning history.

        Args:
            days: Number of days to return

        Returns:
            List of learning entries sorted by date (newest first)
        """
        try:
            if not await self._file_exists(self.learning_history_file):
                return []

            data = await self._read_json_file(self.learning_history_file, None)
            if not data:
                return []

            # Support both old "history" and new "daily_history" format
            history = data.get("daily_history") or data.get("history", {})
            sorted_dates = sorted(history.keys(), reverse=True)[:days]

            return [
                {"date": d, **history[d]}
                for d in sorted_dates
            ]

        except Exception as e:
            _LOGGER.warning(f"Error getting learning history: {e}")
            return []
