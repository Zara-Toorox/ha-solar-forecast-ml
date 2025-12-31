# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************
"""Weather Expert Blender - Multi-Source Weather Blending with Learned Weights V12.4.0

This module implements a multi-expert weather blending system that:
1. Treats each weather source as an independent "expert"
2. Learns accuracy weights per cloud type from actual observations
3. Blends forecasts using cloud-type-specific weights

Architecture:
    [Open-Meteo API] ──► OpenMeteoExpert ───┐
                                            │
    [wttr.in API]    ──► WttrInExpert     ──┤
                                            │
    [ECMWF Layers]   ──► ECMWFLayerExpert ──┼──► WeatherExpertBlender
                                            │         │
    [Bright Sky]     ──► BrightSkyExpert  ──┤         ▼
                                            │   Blended Forecast
    [Pirate Weather] ──► PirateWeatherExpert┘         │
                                              WeatherExpertLearner
                                              (learns from IST data)
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from homeassistant.core import HomeAssistant

from .data_io import DataManagerIO

_LOGGER = logging.getLogger(__name__)


class CloudType(Enum):
    """Cloud type classification based on layer distribution."""
    CLEAR = "clear"
    CIRRUS = "cirrus"
    FAIR = "fair"
    MIXED = "mixed"
    STRATUS = "stratus"
    OVERCAST = "overcast"
    SNOW = "snow"  # Snowy conditions - for learning snow prediction accuracy


# Cloud type classification thresholds
LAYER_THRESHOLD_DOMINANT = 50.0
LAYER_THRESHOLD_LOW = 20.0
LAYER_THRESHOLD_CLEAR = 25.0

# Default weights per cloud type (before learning)
# 5 experts: open_meteo, wttr_in, ecmwf_layers, bright_sky (DWD ICON), pirate_weather (NOAA)
# NOTE: For Germany, Bright Sky (DWD ICON 2.2km) and Pirate Weather (NOAA GFS/HRRR) are
# typically more accurate than Open-Meteo and wttr.in for cloud cover forecasts.
DEFAULT_EXPERT_WEIGHTS: Dict[str, Dict[str, float]] = {
    CloudType.CLEAR.value: {
        "open_meteo": 0.15,
        "wttr_in": 0.10,
        "ecmwf_layers": 0.15,
        "bright_sky": 0.35,      # DWD ICON - high resolution for Germany
        "pirate_weather": 0.25,  # NOAA GFS/HRRR
    },
    CloudType.CIRRUS.value: {
        "open_meteo": 0.10,
        "wttr_in": 0.10,
        "ecmwf_layers": 0.30,    # ECMWF layers best for high clouds
        "bright_sky": 0.30,
        "pirate_weather": 0.20,
    },
    CloudType.FAIR.value: {
        "open_meteo": 0.15,
        "wttr_in": 0.10,
        "ecmwf_layers": 0.15,
        "bright_sky": 0.35,
        "pirate_weather": 0.25,
    },
    CloudType.MIXED.value: {
        "open_meteo": 0.15,
        "wttr_in": 0.10,
        "ecmwf_layers": 0.15,
        "bright_sky": 0.30,
        "pirate_weather": 0.30,
    },
    CloudType.STRATUS.value: {
        "open_meteo": 0.10,
        "wttr_in": 0.10,
        "ecmwf_layers": 0.10,
        "bright_sky": 0.40,      # DWD best for low clouds in Germany
        "pirate_weather": 0.30,
    },
    CloudType.OVERCAST.value: {
        "open_meteo": 0.10,
        "wttr_in": 0.10,
        "ecmwf_layers": 0.10,
        "bright_sky": 0.40,
        "pirate_weather": 0.30,
    },
    CloudType.SNOW.value: {
        # Initial equal weights - will be learned from actual snow events
        "open_meteo": 0.20,
        "wttr_in": 0.20,
        "ecmwf_layers": 0.20,
        "bright_sky": 0.20,
        "pirate_weather": 0.20,
    },
}

# V12.4: Transmission coefficients for cloud-to-transmission conversion
# Used when source doesn't provide layer data (only total cloud %)
# These are conservative estimates based on typical cloud optical depths
TRANSMISSION_TAU_BY_CLOUD_TYPE: Dict[str, float] = {
    CloudType.CLEAR.value: 1.00,      # No blocking
    CloudType.CIRRUS.value: 0.90,     # High thin clouds - minimal blocking
    CloudType.FAIR.value: 0.70,       # Some blocking
    CloudType.MIXED.value: 0.55,      # Moderate blocking
    CloudType.STRATUS.value: 0.35,    # Low thick clouds - significant blocking
    CloudType.OVERCAST.value: 0.20,   # Heavy blocking
    CloudType.SNOW.value: 0.05,       # Snow on panels - near total blocking
}


def cloud_to_transmission(
    cloud_percent: float,
    cloud_type: CloudType = CloudType.MIXED
) -> float:
    """Convert cloud cover percentage to solar transmission percentage.

    V12.4: Used for sources that don't provide layer data.
    Uses cloud-type-specific transmission coefficients.

    Args:
        cloud_percent: Cloud cover (0-100%)
        cloud_type: Type of clouds for coefficient lookup

    Returns:
        Solar transmission as percentage (0-100%)

    Example:
        50% Cirrus → 95% transmission (high clouds don't block much)
        50% Stratus → 67.5% transmission (low clouds block more)
    """
    tau = TRANSMISSION_TAU_BY_CLOUD_TYPE.get(cloud_type.value, 0.55)
    # Formula: transmission = 100 * (1 - cloud_fraction * (1 - tau))
    transmission = 100.0 * (1.0 - (cloud_percent / 100.0) * (1.0 - tau))
    return round(max(0.0, min(100.0, transmission)), 1)


# API settings for external weather sources
BRIGHT_SKY_BASE_URL = "https://api.brightsky.dev/weather"
BRIGHT_SKY_TIMEOUT = 15
BRIGHT_SKY_CACHE_HOURS = 3

PIRATE_WEATHER_BASE_URL = "https://api.pirateweather.net/forecast"
PIRATE_WEATHER_TIMEOUT = 15
PIRATE_WEATHER_CACHE_HOURS = 3

# Learning parameters
LEARNING_SMOOTHING_FACTOR = 0.3
LEARNING_MIN_ERROR = 5.0
LEARNING_MIN_HOURS = 4
LEARNING_ACCELERATED_THRESHOLD = 35.0
LEARNING_ACCELERATED_FACTOR = 0.5


@dataclass
class ExpertForecast:
    """Forecast data from a single expert."""
    expert_name: str
    cloud_cover: float
    confidence: float = 1.0
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BlendedHourForecast:
    """Blended forecast for a single hour."""
    date: str
    hour: int
    cloud_cover: float
    cloud_type: CloudType
    expert_forecasts: Dict[str, float]
    blend_weights: Dict[str, float]
    cloud_cover_low: Optional[float] = None
    cloud_cover_mid: Optional[float] = None
    cloud_cover_high: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    precipitation: Optional[float] = None
    wind_speed: Optional[float] = None
    pressure: Optional[float] = None
    ghi: Optional[float] = None
    direct_radiation: Optional[float] = None
    diffuse_radiation: Optional[float] = None


class WeatherExpert(ABC):
    """Abstract base class for weather data experts."""

    def __init__(self, name: str):
        self.name = name
        self._last_error: Optional[str] = None

    @abstractmethod
    async def get_cloud_cover(self, date: str, hour: int) -> Optional[float]:
        """Get cloud cover prediction for a specific hour."""
        pass

    @abstractmethod
    async def get_forecast_data(self, date: str, hour: int) -> Optional[Dict[str, Any]]:
        """Get full forecast data for a specific hour."""
        pass

    def get_last_error(self) -> Optional[str]:
        """Get last error message."""
        return self._last_error


class OpenMeteoExpert(WeatherExpert):
    """Expert using Open-Meteo total cloud cover."""

    def __init__(
        self,
        cache_data: Dict[str, Dict[int, Dict[str, Any]]],
        cache_file: Optional[Path] = None,
    ):
        super().__init__("open_meteo")
        self._cache = cache_data
        self._cache_file = cache_file
        self._file_cache_loaded = False

    async def _ensure_cache_loaded(self) -> None:
        """Load cache from file if memory cache is empty."""
        if self._cache and len(self._cache) > 0:
            return
        if self._file_cache_loaded or not self._cache_file:
            return

        try:
            if self._cache_file.exists():
                def _read():
                    with open(self._cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                data = await asyncio.to_thread(_read)
                self._cache = data.get("forecast", {})
                _LOGGER.debug(f"OpenMeteoExpert: Loaded {len(self._cache)} days from file cache")
        except Exception as e:
            _LOGGER.warning(f"OpenMeteoExpert: Error loading file cache: {e}")
        self._file_cache_loaded = True

    async def get_cloud_cover(self, date: str, hour: int) -> Optional[float]:
        await self._ensure_cache_loaded()
        # Support both string and int keys (JSON uses strings)
        hour_key = str(hour) if str(hour) in self._cache.get(date, {}) else hour
        if date in self._cache and hour_key in self._cache[date]:
            return self._cache[date][hour_key].get("cloud_cover")
        return None

    async def get_forecast_data(self, date: str, hour: int) -> Optional[Dict[str, Any]]:
        await self._ensure_cache_loaded()
        hour_key = str(hour) if str(hour) in self._cache.get(date, {}) else hour
        if date in self._cache and hour_key in self._cache[date]:
            return self._cache[date][hour_key].copy()
        return None


class WttrInExpert(WeatherExpert):
    """Expert using wttr.in (World Weather Online) cloud cover."""

    def __init__(
        self,
        cache_data: Dict[str, Dict[int, Dict[str, Any]]],
        cache_file: Optional[Path] = None,
    ):
        super().__init__("wttr_in")
        self._cache = cache_data
        self._cache_file = cache_file
        self._file_cache_loaded = False

    async def _ensure_cache_loaded(self) -> None:
        """Load cache from file if memory cache is empty."""
        if self._cache and len(self._cache) > 0:
            return
        if self._file_cache_loaded or not self._cache_file:
            return

        try:
            if self._cache_file.exists():
                def _read():
                    with open(self._cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                data = await asyncio.to_thread(_read)
                self._cache = data.get("forecast", {})
                _LOGGER.debug(f"WttrInExpert: Loaded {len(self._cache)} days from file cache")
        except Exception as e:
            _LOGGER.warning(f"WttrInExpert: Error loading file cache: {e}")
        self._file_cache_loaded = True

    async def get_cloud_cover(self, date: str, hour: int) -> Optional[float]:
        await self._ensure_cache_loaded()
        if date not in self._cache:
            return None

        day_data = self._cache[date]
        # Support both string and int keys (JSON uses strings)
        hour_key = str(hour) if str(hour) in day_data else hour
        if hour_key in day_data:
            return day_data[hour_key].get("cloud_cover")

        return self._get_interpolated_cloud(day_data, hour)

    def _get_interpolated_cloud(
        self,
        day_data: Dict[int, Dict[str, Any]],
        target_hour: int
    ) -> Optional[float]:
        """Interpolate cloud cover for hours between wttr.in 3-hour intervals."""
        if not day_data:
            return None

        # Convert keys to int for sorting and comparison (JSON uses string keys)
        available_hours = sorted([int(h) for h in day_data.keys()])
        if not available_hours:
            return None

        lower_hour = None
        upper_hour = None

        for h in available_hours:
            if h <= target_hour:
                lower_hour = h
            if h >= target_hour and upper_hour is None:
                upper_hour = h
                break

        if lower_hour is None:
            lower_hour = available_hours[0]
        if upper_hour is None:
            upper_hour = available_hours[-1]

        if lower_hour == upper_hour:
            # Use string key to access data
            return day_data[str(lower_hour)].get("cloud_cover")

        # Use string keys to access data
        lower_cloud = day_data[str(lower_hour)].get("cloud_cover")
        upper_cloud = day_data[str(upper_hour)].get("cloud_cover")

        if lower_cloud is None or upper_cloud is None:
            return lower_cloud or upper_cloud

        factor = (target_hour - lower_hour) / (upper_hour - lower_hour)
        return round(lower_cloud + factor * (upper_cloud - lower_cloud), 1)

    async def get_forecast_data(self, date: str, hour: int) -> Optional[Dict[str, Any]]:
        await self._ensure_cache_loaded()
        hour_key = str(hour) if str(hour) in self._cache.get(date, {}) else hour
        if date in self._cache and hour_key in self._cache[date]:
            return self._cache[date][hour_key].copy()
        return None


class ECMWFLayerExpert(WeatherExpert):
    """Expert using ECMWF cloud layer data for physics-based cloud cover."""

    # Layer transmission coefficients (empirical values for solar radiation)
    TAU_LOW = 0.30   # Low clouds block ~70%
    TAU_MID = 0.60   # Mid clouds block ~40%
    TAU_HIGH = 0.90  # High clouds block ~10%

    def __init__(
        self,
        cache_data: Dict[str, Dict[int, Dict[str, Any]]],
        cache_file: Optional[Path] = None,
    ):
        super().__init__("ecmwf_layers")
        self._cache = cache_data
        self._cache_file = cache_file
        self._file_cache_loaded = False

    async def _ensure_cache_loaded(self) -> None:
        """Load cache from file if memory cache is empty."""
        if self._cache and len(self._cache) > 0:
            return
        if self._file_cache_loaded or not self._cache_file:
            return

        try:
            if self._cache_file.exists():
                def _read():
                    with open(self._cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                data = await asyncio.to_thread(_read)
                self._cache = data.get("forecast", {})
                _LOGGER.debug(f"ECMWFLayerExpert: Loaded {len(self._cache)} days from file cache")
        except Exception as e:
            _LOGGER.warning(f"ECMWFLayerExpert: Error loading file cache: {e}")
        self._file_cache_loaded = True

    async def get_cloud_cover(self, date: str, hour: int) -> Optional[float]:
        await self._ensure_cache_loaded()
        # Support both string and int keys (JSON uses strings)
        hour_key = str(hour) if str(hour) in self._cache.get(date, {}) else hour
        if date not in self._cache or hour_key not in self._cache[date]:
            return None

        hour_data = self._cache[date][hour_key]
        cloud_low = hour_data.get("cloud_cover_low")
        cloud_mid = hour_data.get("cloud_cover_mid")
        cloud_high = hour_data.get("cloud_cover_high")

        if cloud_low is None or cloud_mid is None or cloud_high is None:
            return None

        return self._calculate_effective_cloud_cover(cloud_low, cloud_mid, cloud_high)

    def _calculate_effective_cloud_cover(
        self,
        cloud_low: float,
        cloud_mid: float,
        cloud_high: float
    ) -> float:
        """Calculate effective cloud cover based on layer transmission physics.

        Uses Beer-Lambert-like transmission model where each layer
        reduces radiation independently based on its optical depth.
        """
        trans_low = 1.0 - (cloud_low / 100.0) * (1.0 - self.TAU_LOW)
        trans_mid = 1.0 - (cloud_mid / 100.0) * (1.0 - self.TAU_MID)
        trans_high = 1.0 - (cloud_high / 100.0) * (1.0 - self.TAU_HIGH)

        total_transmission = trans_low * trans_mid * trans_high
        effective_cloud = (1.0 - total_transmission) * 100.0

        return round(max(0.0, min(100.0, effective_cloud)), 1)

    async def get_solar_transmission(self, date: str, hour: int) -> Optional[float]:
        """Get solar transmission directly (0-100%) without converting to cloud %.

        V12.4: Returns the actual transmission value for physics-based blending.
        This preserves the layer information instead of losing it in conversion.

        Returns:
            Solar transmission as percentage (0-100%), or None if no data
        """
        await self._ensure_cache_loaded()
        hour_key = str(hour) if str(hour) in self._cache.get(date, {}) else hour
        if date not in self._cache or hour_key not in self._cache[date]:
            return None

        hour_data = self._cache[date][hour_key]
        cloud_low = hour_data.get("cloud_cover_low")
        cloud_mid = hour_data.get("cloud_cover_mid")
        cloud_high = hour_data.get("cloud_cover_high")

        if cloud_low is None or cloud_mid is None or cloud_high is None:
            return None

        # Calculate transmission directly using Beer-Lambert model
        trans_low = 1.0 - (cloud_low / 100.0) * (1.0 - self.TAU_LOW)
        trans_mid = 1.0 - (cloud_mid / 100.0) * (1.0 - self.TAU_MID)
        trans_high = 1.0 - (cloud_high / 100.0) * (1.0 - self.TAU_HIGH)

        total_transmission = trans_low * trans_mid * trans_high
        return round(total_transmission * 100.0, 1)

    async def get_forecast_data(self, date: str, hour: int) -> Optional[Dict[str, Any]]:
        await self._ensure_cache_loaded()
        hour_key = str(hour) if str(hour) in self._cache.get(date, {}) else hour
        if date in self._cache and hour_key in self._cache[date]:
            data = self._cache[date][hour_key].copy()
            data["effective_cloud_cover"] = await self.get_cloud_cover(date, hour)
            data["solar_transmission"] = await self.get_solar_transmission(date, hour)
            return data
        return None


class BrightSkyExpert(WeatherExpert):
    """Expert using DWD ICON model via Bright Sky API (high resolution for Germany).

    Bright Sky provides free access to DWD ICON-D2 (2.2km resolution) for
    Germany, Austria, Switzerland, and neighboring countries.
    No API key required.

    Features:
    - File-based cache with 2-year retention
    - 3-hour memory cache for fast access
    - Automatic fallback to file cache when API unavailable
    """

    MEMORY_CACHE_HOURS = 3
    FILE_CACHE_RETENTION_DAYS = 730  # 2 years

    def __init__(
        self,
        latitude: float,
        longitude: float,
        hass: Optional[Any] = None,
        cache_file: Optional[Path] = None,
    ):
        super().__init__("bright_sky")
        self.latitude = latitude
        self.longitude = longitude
        self.hass = hass
        self.cache_file = cache_file
        self._memory_cache: Dict[str, Dict[int, float]] = {}
        self._memory_cache_time: Optional[datetime] = None
        self._file_cache: Dict[str, Dict[str, float]] = {}
        self._file_cache_loaded = False
        self._consecutive_failures = 0

    async def get_cloud_cover(self, date: str, hour: int) -> Optional[float]:
        """Get cloud cover, using memory cache, then file cache, then API."""
        # 1. Check memory cache
        if self._is_memory_cache_valid() and date in self._memory_cache:
            if hour in self._memory_cache[date]:
                return self._memory_cache[date][hour]

        # 2. Load file cache if not loaded
        if not self._file_cache_loaded:
            await self._load_file_cache()

        # 3. Check file cache for this date/hour
        if date in self._file_cache and str(hour) in self._file_cache[date]:
            return float(self._file_cache[date][str(hour)])

        # 4. Fetch from API
        await self._fetch_forecast()

        # 5. Return from memory cache if available
        if date in self._memory_cache and hour in self._memory_cache[date]:
            return self._memory_cache[date][hour]

        return None

    def _is_memory_cache_valid(self) -> bool:
        if not self._memory_cache_time:
            return False
        age = (datetime.now() - self._memory_cache_time).total_seconds()
        return age < self.MEMORY_CACHE_HOURS * 3600

    async def _load_file_cache(self) -> bool:
        """Load cache from file."""
        if not self.cache_file:
            self._file_cache_loaded = True
            return False

        try:
            if not self.cache_file.exists():
                self._file_cache_loaded = True
                return False

            def _read():
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)

            data = await asyncio.to_thread(_read)
            self._file_cache = data.get("forecast", {})
            self._file_cache_loaded = True

            _LOGGER.debug(f"Bright Sky: Loaded {len(self._file_cache)} days from file cache")
            return True

        except Exception as e:
            _LOGGER.warning(f"Bright Sky: Error loading file cache: {e}")
            self._file_cache_loaded = True
            return False

    async def _save_file_cache(self) -> bool:
        """Save cache to file with 2-year retention."""
        if not self.cache_file:
            return False

        try:
            # CRITICAL: Load existing file cache first to preserve historical data
            # Without this, morning hours would be lost when API is called later in the day
            # because the API only returns future hours, not past ones
            if not self._file_cache_loaded:
                await self._load_file_cache()

            # Merge memory cache into file cache
            for date_str, hours in self._memory_cache.items():
                if date_str not in self._file_cache:
                    self._file_cache[date_str] = {}
                for hour, cloud in hours.items():
                    self._file_cache[date_str][str(hour)] = cloud

            # Apply retention (remove entries older than 2 years)
            cutoff_date = (datetime.now() - timedelta(days=self.FILE_CACHE_RETENTION_DAYS)).strftime("%Y-%m-%d")
            dates_to_remove = [d for d in self._file_cache.keys() if d < cutoff_date]
            for d in dates_to_remove:
                del self._file_cache[d]

            cache_data = {
                "version": "1.0",
                "source": "bright_sky",
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "retention_days": self.FILE_CACHE_RETENTION_DAYS,
                },
                "forecast": self._file_cache,
            }

            def _write():
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                temp_file = self.cache_file.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, indent=2)
                temp_file.replace(self.cache_file)

            await asyncio.to_thread(_write)
            _LOGGER.debug(f"Bright Sky: Saved {len(self._file_cache)} days to file cache")
            return True

        except Exception as e:
            _LOGGER.warning(f"Bright Sky: Error saving file cache: {e}")
            return False

    async def _fetch_forecast(self) -> bool:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")

            url = (
                f"{BRIGHT_SKY_BASE_URL}"
                f"?lat={self.latitude}&lon={self.longitude}"
                f"&date={today}&last_date={end_date}"
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=BRIGHT_SKY_TIMEOUT),
                    headers={"User-Agent": "SolarForecastML/1.0"}
                ) as response:
                    if response.status != 200:
                        self._last_error = f"HTTP {response.status}"
                        self._consecutive_failures += 1
                        _LOGGER.debug(f"Bright Sky API returned status {response.status}")
                        return False

                    data = await response.json()

            weather_list = data.get("weather", [])
            if not weather_list:
                self._last_error = "No weather data"
                return False

            self._memory_cache.clear()
            for entry in weather_list:
                timestamp = entry.get("timestamp")
                cloud_cover = entry.get("cloud_cover")

                if timestamp and cloud_cover is not None:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d")
                    hour = dt.hour

                    if date_str not in self._memory_cache:
                        self._memory_cache[date_str] = {}
                    self._memory_cache[date_str][hour] = float(cloud_cover)

            self._memory_cache_time = datetime.now()
            self._consecutive_failures = 0

            # Save to file cache
            await self._save_file_cache()

            _LOGGER.debug(
                f"Bright Sky: Fetched {len(weather_list)} hours "
                f"for {len(self._memory_cache)} days"
            )
            return True

        except asyncio.TimeoutError:
            self._last_error = "Timeout"
            self._consecutive_failures += 1
            _LOGGER.debug("Bright Sky API timeout")
            return False
        except aiohttp.ClientError as e:
            self._last_error = f"Connection: {e}"
            self._consecutive_failures += 1
            _LOGGER.debug(f"Bright Sky API connection error: {e}")
            return False
        except Exception as e:
            self._last_error = f"Error: {e}"
            self._consecutive_failures += 1
            _LOGGER.debug(f"Bright Sky API error: {e}")
            return False

    async def get_forecast_data(self, date: str, hour: int) -> Optional[Dict[str, Any]]:
        cloud = await self.get_cloud_cover(date, hour)
        if cloud is not None:
            return {"cloud_cover": cloud, "source": "bright_sky"}
        return None

    def get_file_cache(self) -> Dict[str, Dict[str, float]]:
        """Get file cache for blending."""
        return self._file_cache


class PirateWeatherExpert(WeatherExpert):
    """Expert using NOAA GFS/HRRR models via Pirate Weather API.

    Pirate Weather provides free access to NOAA weather models.
    Free tier: 20,000 API calls/month.
    Requires API key from https://pirate-weather.apiable.io/

    Features:
    - File-based cache with 2-year retention
    - 3-hour memory cache for fast access
    - Automatic fallback to file cache when API unavailable
    """

    MEMORY_CACHE_HOURS = 3
    FILE_CACHE_RETENTION_DAYS = 730  # 2 years

    def __init__(
        self,
        latitude: float,
        longitude: float,
        api_key: Optional[str] = None,
        hass: Optional[Any] = None,
        cache_file: Optional[Path] = None,
    ):
        super().__init__("pirate_weather")
        self.latitude = latitude
        self.longitude = longitude
        self.api_key = api_key
        self.hass = hass
        self.cache_file = cache_file
        self._memory_cache: Dict[str, Dict[int, float]] = {}
        self._memory_cache_time: Optional[datetime] = None
        self._file_cache: Dict[str, Dict[str, float]] = {}
        self._file_cache_loaded = False
        self._consecutive_failures = 0
        self._enabled = api_key is not None

        if not self._enabled:
            _LOGGER.info(
                "PirateWeatherExpert disabled - no API key configured. "
                "Get free key at https://pirate-weather.apiable.io/"
            )

    async def get_cloud_cover(self, date: str, hour: int) -> Optional[float]:
        """Get cloud cover, using memory cache, then file cache, then API."""
        # Even without API key, we can use file cache from previous runs
        # 1. Check memory cache
        if self._is_memory_cache_valid() and date in self._memory_cache:
            if hour in self._memory_cache[date]:
                return self._memory_cache[date][hour]

        # 2. Load file cache if not loaded
        if not self._file_cache_loaded:
            await self._load_file_cache()

        # 3. Check file cache for this date/hour
        if date in self._file_cache and str(hour) in self._file_cache[date]:
            return float(self._file_cache[date][str(hour)])

        # 4. If enabled and memory cache is stale, fetch from API (max once per cache period)
        if self._enabled and not self._is_memory_cache_valid():
            await self._fetch_forecast()

            # 5. Return from memory cache if available
            if date in self._memory_cache and hour in self._memory_cache[date]:
                return self._memory_cache[date][hour]

        return None

    def _is_memory_cache_valid(self) -> bool:
        if not self._memory_cache_time:
            return False
        age = (datetime.now() - self._memory_cache_time).total_seconds()
        return age < self.MEMORY_CACHE_HOURS * 3600

    async def _load_file_cache(self) -> bool:
        """Load cache from file."""
        if not self.cache_file:
            self._file_cache_loaded = True
            return False

        try:
            if not self.cache_file.exists():
                self._file_cache_loaded = True
                return False

            def _read():
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)

            data = await asyncio.to_thread(_read)
            self._file_cache = data.get("forecast", {})
            self._file_cache_loaded = True

            _LOGGER.debug(f"Pirate Weather: Loaded {len(self._file_cache)} days from file cache")
            return True

        except Exception as e:
            _LOGGER.warning(f"Pirate Weather: Error loading file cache: {e}")
            self._file_cache_loaded = True
            return False

    async def _save_file_cache(self) -> bool:
        """Save cache to file with 2-year retention."""
        if not self.cache_file:
            return False

        try:
            # CRITICAL: Load existing file cache first to preserve historical data
            # Without this, morning hours would be lost when API is called later in the day
            # because the API only returns future hours, not past ones
            if not self._file_cache_loaded:
                await self._load_file_cache()

            # Merge memory cache into file cache
            for date_str, hours in self._memory_cache.items():
                if date_str not in self._file_cache:
                    self._file_cache[date_str] = {}
                for hour, cloud in hours.items():
                    self._file_cache[date_str][str(hour)] = cloud

            # Apply retention (remove entries older than 2 years)
            cutoff_date = (datetime.now() - timedelta(days=self.FILE_CACHE_RETENTION_DAYS)).strftime("%Y-%m-%d")
            dates_to_remove = [d for d in self._file_cache.keys() if d < cutoff_date]
            for d in dates_to_remove:
                del self._file_cache[d]

            cache_data = {
                "version": "1.0",
                "source": "pirate_weather",
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "retention_days": self.FILE_CACHE_RETENTION_DAYS,
                },
                "forecast": self._file_cache,
            }

            def _write():
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                temp_file = self.cache_file.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, indent=2)
                temp_file.replace(self.cache_file)

            await asyncio.to_thread(_write)
            _LOGGER.debug(f"Pirate Weather: Saved {len(self._file_cache)} days to file cache")
            return True

        except Exception as e:
            _LOGGER.warning(f"Pirate Weather: Error saving file cache: {e}")
            return False

    async def _fetch_forecast(self) -> bool:
        if not self._enabled or not self.api_key:
            return False

        try:
            url = (
                f"{PIRATE_WEATHER_BASE_URL}/{self.api_key}"
                f"/{self.latitude},{self.longitude}"
                f"?units=si&extend=hourly"
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=PIRATE_WEATHER_TIMEOUT),
                    headers={"User-Agent": "SolarForecastML/1.0"}
                ) as response:
                    if response.status == 401:
                        self._last_error = "Invalid API key"
                        self._enabled = False
                        _LOGGER.warning("Pirate Weather API key invalid - disabling")
                        return False
                    if response.status != 200:
                        self._last_error = f"HTTP {response.status}"
                        self._consecutive_failures += 1
                        _LOGGER.debug(f"Pirate Weather API returned status {response.status}")
                        return False

                    data = await response.json()

            hourly = data.get("hourly", {})
            hourly_data = hourly.get("data", [])

            if not hourly_data:
                self._last_error = "No hourly data"
                return False

            self._memory_cache.clear()
            for entry in hourly_data:
                timestamp = entry.get("time")
                cloud_cover = entry.get("cloudCover")

                if timestamp and cloud_cover is not None:
                    dt = datetime.fromtimestamp(timestamp)
                    date_str = dt.strftime("%Y-%m-%d")
                    hour = dt.hour

                    if date_str not in self._memory_cache:
                        self._memory_cache[date_str] = {}
                    # Pirate Weather returns 0-1, convert to 0-100
                    self._memory_cache[date_str][hour] = round(float(cloud_cover) * 100.0, 1)

            self._memory_cache_time = datetime.now()
            self._consecutive_failures = 0

            # Save to file cache
            await self._save_file_cache()

            _LOGGER.debug(
                f"Pirate Weather: Fetched {len(hourly_data)} hours "
                f"for {len(self._memory_cache)} days"
            )
            return True

        except asyncio.TimeoutError:
            self._last_error = "Timeout"
            self._consecutive_failures += 1
            _LOGGER.debug("Pirate Weather API timeout")
            return False
        except aiohttp.ClientError as e:
            self._last_error = f"Connection: {e}"
            self._consecutive_failures += 1
            _LOGGER.debug(f"Pirate Weather API connection error: {e}")
            return False
        except Exception as e:
            self._last_error = f"Error: {e}"
            self._consecutive_failures += 1
            _LOGGER.debug(f"Pirate Weather API error: {e}")
            return False

    async def get_forecast_data(self, date: str, hour: int) -> Optional[Dict[str, Any]]:
        cloud = await self.get_cloud_cover(date, hour)
        if cloud is not None:
            return {"cloud_cover": cloud, "source": "pirate_weather"}
        return None

    def get_file_cache(self) -> Dict[str, Dict[str, float]]:
        """Get file cache for blending."""
        return self._file_cache


def classify_cloud_type(
    cloud_low: Optional[float],
    cloud_mid: Optional[float],
    cloud_high: Optional[float],
    cloud_total: Optional[float] = None
) -> CloudType:
    """Classify cloud type based on layer distribution.

    Args:
        cloud_low: Low cloud cover (0-100%)
        cloud_mid: Mid cloud cover (0-100%)
        cloud_high: High cloud cover (0-100%)
        cloud_total: Total cloud cover as fallback

    Returns:
        CloudType enum value
    """
    if cloud_low is None or cloud_mid is None or cloud_high is None:
        if cloud_total is not None:
            if cloud_total <= LAYER_THRESHOLD_CLEAR:
                return CloudType.CLEAR
            elif cloud_total <= 50:
                return CloudType.FAIR
            elif cloud_total <= 75:
                return CloudType.MIXED
            else:
                return CloudType.OVERCAST
        return CloudType.MIXED

    max_layer = max(cloud_low, cloud_mid, cloud_high)

    if max_layer < LAYER_THRESHOLD_CLEAR:
        return CloudType.CLEAR

    if cloud_high > LAYER_THRESHOLD_DOMINANT and cloud_low < LAYER_THRESHOLD_LOW:
        return CloudType.CIRRUS

    if cloud_low > LAYER_THRESHOLD_DOMINANT:
        return CloudType.STRATUS

    if cloud_low > 60 and cloud_mid > 40 and cloud_high > 40:
        return CloudType.OVERCAST

    layer_spread = max_layer - min(cloud_low, cloud_mid, cloud_high)
    if layer_spread < 25:
        return CloudType.MIXED

    return CloudType.FAIR


class WeatherExpertBlender(DataManagerIO):
    """Blends multiple weather experts with learned cloud-type-specific weights."""

    # Cache file names
    BRIGHT_SKY_CACHE_FILE = "bright_sky_cache.json"
    PIRATE_WEATHER_CACHE_FILE = "pirate_weather_cache.json"
    OPEN_METEO_CACHE_FILE = "open_meteo_cache.json"
    WTTR_IN_CACHE_FILE = "wttr_in_cache.json"

    def __init__(
        self,
        hass: HomeAssistant,
        data_dir: Path,
        open_meteo_cache: Dict[str, Dict[int, Dict[str, Any]]],
        wttr_cache: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        pirate_weather_api_key: Optional[str] = None,
    ):
        super().__init__(hass, data_dir)

        self.weights_file = data_dir / "data" / "weather_expert_weights.json"
        self.bright_sky_cache_file = data_dir / "data" / self.BRIGHT_SKY_CACHE_FILE
        self.pirate_weather_cache_file = data_dir / "data" / self.PIRATE_WEATHER_CACHE_FILE
        self.open_meteo_cache_file = data_dir / "data" / self.OPEN_METEO_CACHE_FILE
        self.wttr_in_cache_file = data_dir / "data" / self.WTTR_IN_CACHE_FILE
        self.latitude = latitude or hass.config.latitude
        self.longitude = longitude or hass.config.longitude

        # Cache-based experts with file fallback (use memory data or load from file)
        self._open_meteo_expert = OpenMeteoExpert(
            open_meteo_cache,
            cache_file=self.open_meteo_cache_file,
        )
        self._wttr_expert = WttrInExpert(
            wttr_cache or {},
            cache_file=self.wttr_in_cache_file,
        )
        self._ecmwf_expert = ECMWFLayerExpert(
            open_meteo_cache,
            cache_file=self.open_meteo_cache_file,  # ECMWF layers use Open-Meteo data
        )

        # API-based experts with file caches (fetch their own data)
        self._bright_sky_expert = BrightSkyExpert(
            latitude=self.latitude,
            longitude=self.longitude,
            hass=hass,
            cache_file=self.bright_sky_cache_file,
        )
        self._pirate_weather_expert = PirateWeatherExpert(
            latitude=self.latitude,
            longitude=self.longitude,
            api_key=pirate_weather_api_key,
            hass=hass,
            cache_file=self.pirate_weather_cache_file,
        )

        self._experts: List[WeatherExpert] = [
            self._open_meteo_expert,
            self._wttr_expert,
            self._ecmwf_expert,
            self._bright_sky_expert,
            self._pirate_weather_expert,
        ]

        self._weights: Dict[str, Dict[str, float]] = {}
        self._weights_loaded = False

        self._blend_stats = {
            "total_hours_blended": 0,
            "cloud_type_counts": {ct.value: 0 for ct in CloudType},
            "expert_contributions": {e.name: 0 for e in self._experts},
        }

        active_experts = [e.name for e in self._experts if self._is_expert_enabled(e)]
        _LOGGER.info(
            "WeatherExpertBlender initialized with %d experts (%d active): %s",
            len(self._experts),
            len(active_experts),
            ", ".join(active_experts)
        )

    def _is_expert_enabled(self, expert: WeatherExpert) -> bool:
        """Check if an expert is enabled and available."""
        if isinstance(expert, PirateWeatherExpert):
            return expert._enabled
        return True

    async def async_init(self) -> bool:
        """Initialize the blender and load weights."""
        await self._load_weights()

        # Pre-fetch data from API-based experts
        try:
            await self._bright_sky_expert._fetch_forecast()
        except Exception as e:
            _LOGGER.debug(f"Bright Sky initial fetch failed: {e}")

        if self._pirate_weather_expert._enabled:
            try:
                await self._pirate_weather_expert._fetch_forecast()
            except Exception as e:
                _LOGGER.debug(f"Pirate Weather initial fetch failed: {e}")

        return True

    def update_caches(
        self,
        open_meteo_cache: Dict[str, Dict[int, Dict[str, Any]]],
        wttr_cache: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
    ) -> None:
        """Update expert caches with fresh data."""
        self._open_meteo_expert._cache = open_meteo_cache
        self._ecmwf_expert._cache = open_meteo_cache
        if wttr_cache:
            self._wttr_expert._cache = wttr_cache

    async def _load_weights(self) -> bool:
        """Load learned weights from file."""
        try:
            if await self._file_exists(self.weights_file):
                data = await self._read_json_file(self.weights_file, None)
                if data and "weights" in data:
                    self._weights = data["weights"]
                    self._weights_loaded = True

                    metadata = data.get("metadata", {})
                    last_update = metadata.get("last_updated", "never")

                    _LOGGER.info(
                        "Loaded expert weights from %s (last updated: %s)",
                        self.weights_file.name,
                        last_update
                    )
                    return True

            _LOGGER.info("No learned weights found - using defaults")
            self._weights = {k: v.copy() for k, v in DEFAULT_EXPERT_WEIGHTS.items()}
            await self._save_weights()
            return False

        except Exception as e:
            _LOGGER.warning("Error loading expert weights: %s", e)
            self._weights = {k: v.copy() for k, v in DEFAULT_EXPERT_WEIGHTS.items()}
            return False

    async def _save_weights(self) -> bool:
        """Save learned weights to file."""
        try:
            data = {
                "version": "1.0",
                "weights": self._weights,
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "experts": [e.name for e in self._experts],
                    "cloud_types": [ct.value for ct in CloudType],
                },
            }

            await self._atomic_write_json(self.weights_file, data)
            _LOGGER.debug("Saved expert weights to %s", self.weights_file.name)
            return True

        except Exception as e:
            _LOGGER.warning("Error saving expert weights: %s", e)
            return False

    def get_weights_for_cloud_type(self, cloud_type: CloudType) -> Dict[str, float]:
        """Get blending weights for a specific cloud type."""
        type_key = cloud_type.value
        if type_key in self._weights:
            return self._weights[type_key].copy()
        return DEFAULT_EXPERT_WEIGHTS.get(type_key, {}).copy()

    async def blend_hour(
        self,
        date: str,
        hour: int,
        base_data: Optional[Dict[str, Any]] = None
    ) -> Optional[BlendedHourForecast]:
        """Blend forecasts from all experts for a single hour."""
        expert_clouds: Dict[str, float] = {}

        for expert in self._experts:
            cloud = await expert.get_cloud_cover(date, hour)
            if cloud is not None:
                expert_clouds[expert.name] = cloud

        if not expert_clouds:
            _LOGGER.debug("No expert data available for %s %02d:00", date, hour)
            return None

        cloud_low = None
        cloud_mid = None
        cloud_high = None

        if base_data:
            cloud_low = base_data.get("cloud_cover_low")
            cloud_mid = base_data.get("cloud_cover_mid")
            cloud_high = base_data.get("cloud_cover_high")

        cloud_type = classify_cloud_type(
            cloud_low, cloud_mid, cloud_high,
            expert_clouds.get("open_meteo")
        )

        weights = self.get_weights_for_cloud_type(cloud_type)

        total_weight = 0.0
        blended_cloud = 0.0

        for expert_name, cloud_value in expert_clouds.items():
            weight = weights.get(expert_name, 0.0)
            blended_cloud += weight * cloud_value
            total_weight += weight

        if total_weight > 0:
            blended_cloud /= total_weight
        else:
            blended_cloud = sum(expert_clouds.values()) / len(expert_clouds)

        blended_cloud = round(max(0.0, min(100.0, blended_cloud)), 1)

        self._blend_stats["total_hours_blended"] += 1
        self._blend_stats["cloud_type_counts"][cloud_type.value] += 1

        return BlendedHourForecast(
            date=date,
            hour=hour,
            cloud_cover=blended_cloud,
            cloud_type=cloud_type,
            expert_forecasts=expert_clouds,
            blend_weights=weights,
            cloud_cover_low=cloud_low,
            cloud_cover_mid=cloud_mid,
            cloud_cover_high=cloud_high,
            temperature=base_data.get("temperature") if base_data else None,
            humidity=base_data.get("humidity") if base_data else None,
            precipitation=base_data.get("precipitation") if base_data else None,
            wind_speed=base_data.get("wind_speed") if base_data else None,
            pressure=base_data.get("pressure") if base_data else None,
            ghi=base_data.get("ghi") if base_data else None,
            direct_radiation=base_data.get("direct_radiation") if base_data else None,
            diffuse_radiation=base_data.get("diffuse_radiation") if base_data else None,
        )

    async def blend_forecast(
        self,
        dates: List[str],
        base_cache: Dict[str, Dict[int, Dict[str, Any]]]
    ) -> List[BlendedHourForecast]:
        """Blend forecasts for multiple dates."""
        blended_results: List[BlendedHourForecast] = []

        for date in dates:
            if date not in base_cache:
                continue

            for hour in range(24):
                if hour not in base_cache[date]:
                    continue

                base_data = base_cache[date][hour]
                blended = await self.blend_hour(date, hour, base_data)

                if blended:
                    blended_results.append(blended)

        if blended_results:
            type_summary = ", ".join(
                f"{ct}={count}"
                for ct, count in self._blend_stats["cloud_type_counts"].items()
                if count > 0
            )
            _LOGGER.info(
                "Blended %d hours across %d days. Cloud types: %s",
                len(blended_results),
                len(dates),
                type_summary
            )

        return blended_results

    async def update_weights(
        self,
        cloud_type: CloudType,
        new_weights: Dict[str, float]
    ) -> bool:
        """Update weights for a specific cloud type."""
        type_key = cloud_type.value

        total = sum(new_weights.values())
        if total > 0:
            normalized = {k: v / total for k, v in new_weights.items()}
        else:
            return False

        self._weights[type_key] = normalized
        return await self._save_weights()

    async def get_blended_cloud_cover(
        self,
        date: str,
        hour: int,
        cloud_low: Optional[float] = None,
        cloud_mid: Optional[float] = None,
        cloud_high: Optional[float] = None,
        fallback_cloud: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Get blended cloud cover for a specific hour with wttr.in fallback.

        This is the main entry point for WeatherForecastCorrector.

        Args:
            date: Date string (YYYY-MM-DD)
            hour: Hour (0-23)
            cloud_low: ECMWF low cloud layer (for classification)
            cloud_mid: ECMWF mid cloud layer (for classification)
            cloud_high: ECMWF high cloud layer (for classification)
            fallback_cloud: Fallback cloud cover (from wttr.in) if blending fails

        Returns:
            Tuple of (blended_cloud_cover, blend_info_dict)
        """
        expert_clouds: Dict[str, float] = {}

        # Collect cloud cover from all experts
        for expert in self._experts:
            try:
                cloud = await expert.get_cloud_cover(date, hour)
                if cloud is not None:
                    expert_clouds[expert.name] = cloud
            except Exception as e:
                _LOGGER.debug(f"Expert {expert.name} failed for {date} {hour:02d}:00: {e}")

        # If no expert data available, use fallback (wttr.in)
        if not expert_clouds:
            if fallback_cloud is not None:
                return fallback_cloud, {
                    "source": "fallback_wttr",
                    "experts_available": 0,
                    "reason": "no_expert_data",
                }
            return 50.0, {
                "source": "default",
                "experts_available": 0,
                "reason": "no_data_available",
            }

        # Classify cloud type for weight selection
        cloud_type = classify_cloud_type(
            cloud_low, cloud_mid, cloud_high,
            expert_clouds.get("open_meteo")
        )

        # Get learned weights for this cloud type
        weights = self.get_weights_for_cloud_type(cloud_type)

        # Calculate weighted blend
        total_weight = 0.0
        blended_cloud = 0.0

        for expert_name, cloud_value in expert_clouds.items():
            weight = weights.get(expert_name, 0.0)
            if weight > 0:
                blended_cloud += weight * cloud_value
                total_weight += weight

        if total_weight > 0:
            blended_cloud /= total_weight
        else:
            # Equal weight fallback
            blended_cloud = sum(expert_clouds.values()) / len(expert_clouds)

        blended_cloud = round(max(0.0, min(100.0, blended_cloud)), 1)

        # Update stats
        self._blend_stats["total_hours_blended"] += 1
        self._blend_stats["cloud_type_counts"][cloud_type.value] += 1
        for expert_name in expert_clouds:
            self._blend_stats["expert_contributions"][expert_name] = \
                self._blend_stats["expert_contributions"].get(expert_name, 0) + 1

        blend_info = {
            "source": "expert_blend",
            "cloud_type": cloud_type.value,
            "experts_available": len(expert_clouds),
            "expert_values": expert_clouds,
            "weights_used": {k: round(v, 3) for k, v in weights.items() if k in expert_clouds},
            "blended_cloud": blended_cloud,
        }

        return blended_cloud, blend_info

    async def get_blended_transmission(
        self,
        date: str,
        hour: int,
        cloud_low: Optional[float] = None,
        cloud_mid: Optional[float] = None,
        cloud_high: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Get blended solar transmission using physics-based conversion.

        V12.4: PATH B - Transmission-based blending.
        - ECMWF: Uses direct transmission from layer physics
        - Others: Converts cloud% to transmission using cloud-type coefficients

        Args:
            date: Date string (YYYY-MM-DD)
            hour: Hour (0-23)
            cloud_low: ECMWF low cloud layer (for classification)
            cloud_mid: ECMWF mid cloud layer (for classification)
            cloud_high: ECMWF high cloud layer (for classification)

        Returns:
            Tuple of (blended_transmission, blend_info_dict)
        """
        expert_transmissions: Dict[str, float] = {}

        # Classify cloud type for transmission conversion
        cloud_type = classify_cloud_type(cloud_low, cloud_mid, cloud_high, None)

        # ECMWF: Get direct transmission (physics-based)
        if self._ecmwf_expert:
            ecmwf_trans = await self._ecmwf_expert.get_solar_transmission(date, hour)
            if ecmwf_trans is not None:
                expert_transmissions["ecmwf_layers"] = ecmwf_trans

        # Other experts: Convert cloud% to transmission
        for expert in self._experts:
            if expert.name == "ecmwf_layers":
                continue  # Already handled above

            try:
                cloud = await expert.get_cloud_cover(date, hour)
                if cloud is not None:
                    trans = cloud_to_transmission(cloud, cloud_type)
                    expert_transmissions[expert.name] = trans
            except Exception as e:
                _LOGGER.debug(f"Expert {expert.name} transmission failed: {e}")

        # Fallback if no data
        if not expert_transmissions:
            return 50.0, {
                "source": "default",
                "path": "transmission",
                "experts_available": 0,
                "reason": "no_data_available",
            }

        # Get learned weights for this cloud type
        weights = self.get_weights_for_cloud_type(cloud_type)

        # Calculate weighted blend
        total_weight = 0.0
        blended_trans = 0.0

        for expert_name, trans_value in expert_transmissions.items():
            weight = weights.get(expert_name, 0.0)
            if weight > 0:
                blended_trans += weight * trans_value
                total_weight += weight

        if total_weight > 0:
            blended_trans /= total_weight
        else:
            blended_trans = sum(expert_transmissions.values()) / len(expert_transmissions)

        blended_trans = round(max(0.0, min(100.0, blended_trans)), 1)

        blend_info = {
            "source": "expert_blend",
            "path": "transmission",
            "cloud_type": cloud_type.value,
            "experts_available": len(expert_transmissions),
            "expert_values": expert_transmissions,
            "weights_used": {k: round(v, 3) for k, v in weights.items() if k in expert_transmissions},
            "blended_transmission": blended_trans,
        }

        return blended_trans, blend_info

    async def get_dual_path_blend(
        self,
        date: str,
        hour: int,
        cloud_low: Optional[float] = None,
        cloud_mid: Optional[float] = None,
        cloud_high: Optional[float] = None,
        fallback_cloud: Optional[float] = None,
        path_a_weight: float = 0.5,
    ) -> Tuple[float, Dict[str, Any]]:
        """Get dual-path blended transmission combining both methods.

        V12.4: META-BLEND combining:
        - PATH A: Traditional cloud% blending (V12.3, proven)
        - PATH B: Transmission-based blending (physics-based)

        Args:
            date: Date string (YYYY-MM-DD)
            hour: Hour (0-23)
            cloud_low: ECMWF low cloud layer
            cloud_mid: ECMWF mid cloud layer
            cloud_high: ECMWF high cloud layer
            fallback_cloud: Fallback cloud cover for path A
            path_a_weight: Weight for path A (0-1), path B gets (1 - path_a_weight)

        Returns:
            Tuple of (final_transmission, dual_blend_info_dict)
        """
        # PATH A: Traditional cloud% blending → convert to transmission
        cloud_blend, cloud_info = await self.get_blended_cloud_cover(
            date, hour, cloud_low, cloud_mid, cloud_high, fallback_cloud
        )
        # Convert blended cloud% to transmission (simple: 100 - cloud%)
        path_a_transmission = 100.0 - cloud_blend

        # PATH B: Direct transmission blending
        path_b_transmission, trans_info = await self.get_blended_transmission(
            date, hour, cloud_low, cloud_mid, cloud_high
        )

        # META-BLEND: Weighted combination
        path_b_weight = 1.0 - path_a_weight
        final_transmission = (
            path_a_weight * path_a_transmission +
            path_b_weight * path_b_transmission
        )
        final_transmission = round(max(0.0, min(100.0, final_transmission)), 1)

        # Convert back to cloud% for compatibility with existing system
        final_cloud = round(100.0 - final_transmission, 1)

        dual_info = {
            "source": "dual_path_blend",
            "path_a": {
                "weight": path_a_weight,
                "cloud_blend": cloud_blend,
                "transmission": path_a_transmission,
                "cloud_type": cloud_info.get("cloud_type"),
                "experts": cloud_info.get("experts_available", 0),
            },
            "path_b": {
                "weight": path_b_weight,
                "transmission": path_b_transmission,
                "cloud_type": trans_info.get("cloud_type"),
                "experts": trans_info.get("experts_available", 0),
            },
            "final_transmission": final_transmission,
            "final_cloud": final_cloud,
            "improvement": round(path_b_transmission - path_a_transmission, 1),
        }

        # Update stats
        self._blend_stats["dual_path_blends"] = self._blend_stats.get("dual_path_blends", 0) + 1

        return final_transmission, dual_info

    def get_blend_stats(self) -> Dict[str, Any]:
        """Get blending statistics."""
        return {
            **self._blend_stats,
            "weights_loaded": self._weights_loaded,
            "current_weights": self._weights.copy(),
        }


class WeatherExpertLearner(DataManagerIO):
    """Learns optimal expert weights from actual weather observations."""

    def __init__(
        self,
        hass: HomeAssistant,
        data_dir: Path,
        blender: WeatherExpertBlender,
    ):
        super().__init__(hass, data_dir)

        self.blender = blender
        self.learning_history_file = data_dir / "stats" / "weather_expert_learning.json"
        self.actual_weather_file = data_dir / "stats" / "hourly_weather_actual.json"
        self.forecast_cache_file = data_dir / "data" / "open_meteo_cache.json"

        _LOGGER.info("WeatherExpertLearner initialized")

    async def learn_from_day(self, date_str: str) -> Dict[str, Any]:
        """Learn weights from a day's actual observations.

        Compares each expert's predictions with actual cloud cover
        and updates weights inversely proportional to error.
        """
        result = {
            "success": False,
            "date": date_str,
            "reason": None,
            "mae_by_expert": {},
            "mae_by_cloud_type": {},
            "weights_updated": {},
            "comparison_hours": 0,
            "skipped_night_hours": 0,
        }

        try:
            actual_data = await self._load_actual_weather(date_str)
            if len(actual_data) < LEARNING_MIN_HOURS:
                result["reason"] = f"Only {len(actual_data)} hours of actual data"
                return result

            forecast_data = await self._load_forecast_data(date_str)
            if not forecast_data:
                result["reason"] = "No forecast data found"
                return result

            errors_by_expert: Dict[str, List[float]] = {
                e.name: [] for e in self.blender._experts
            }
            errors_by_type: Dict[str, Dict[str, List[float]]] = {
                ct.value: {e.name: [] for e in self.blender._experts}
                for ct in CloudType
            }

            # Debug: Log expert cache status
            for expert in self.blender._experts:
                if hasattr(expert, '_cache'):
                    cache_dates = list(expert._cache.keys()) if expert._cache else []
                    _LOGGER.debug(
                        "Expert %s cache: %d dates, today=%s",
                        expert.name, len(cache_dates), date_str in cache_dates
                    )

            # Production time filter constants
            # Learn only during extended production window: 1.5h before sunrise to 1h after sunset
            HOURS_BEFORE_SUNRISE_LIMIT = -1.5  # Start learning 1.5h before sunrise
            HOURS_AFTER_SUNSET_LIMIT = -1.0    # Stop learning 1h after sunset

            skipped_hours = 0
            for hour_str, actual_hour in actual_data.items():
                actual_cloud = actual_hour.get("cloud_cover_percent")
                if actual_cloud is None:
                    continue

                hour = int(hour_str)
                if hour not in forecast_data:
                    continue

                # Filter to extended production window only
                # hours_after_sunrise < -1.5 means more than 1.5h BEFORE sunrise
                # hours_before_sunset < -1.0 means more than 1h AFTER sunset
                hours_after_sunrise = actual_hour.get("hours_after_sunrise")
                hours_before_sunset = actual_hour.get("hours_before_sunset")

                if hours_after_sunrise is not None and hours_before_sunset is not None:
                    if hours_after_sunrise < HOURS_BEFORE_SUNRISE_LIMIT or hours_before_sunset < HOURS_AFTER_SUNSET_LIMIT:
                        skipped_hours += 1
                        continue

                hour_forecast = forecast_data[hour]

                cloud_type = classify_cloud_type(
                    hour_forecast.get("cloud_cover_low"),
                    hour_forecast.get("cloud_cover_mid"),
                    hour_forecast.get("cloud_cover_high"),
                    hour_forecast.get("cloud_cover")
                )

                for expert in self.blender._experts:
                    predicted = await expert.get_cloud_cover(date_str, hour)
                    if predicted is None:
                        continue

                    error = abs(predicted - actual_cloud)
                    errors_by_expert[expert.name].append(error)
                    errors_by_type[cloud_type.value][expert.name].append(error)

                result["comparison_hours"] += 1

            if result["comparison_hours"] < LEARNING_MIN_HOURS:
                result["reason"] = f"Only {result['comparison_hours']} comparison hours"
                return result

            for expert_name, errors in errors_by_expert.items():
                if errors:
                    result["mae_by_expert"][expert_name] = round(
                        sum(errors) / len(errors), 2
                    )

            for cloud_type in CloudType:
                type_key = cloud_type.value
                type_errors = errors_by_type[type_key]

                expert_maes = {}
                for expert_name, errors in type_errors.items():
                    if errors:
                        expert_maes[expert_name] = sum(errors) / len(errors)

                if len(expert_maes) < 2:
                    continue

                result["mae_by_cloud_type"][type_key] = {
                    k: round(v, 2) for k, v in expert_maes.items()
                }

                new_weights = self._calculate_weights_from_mae(expert_maes)

                old_weights = self.blender.get_weights_for_cloud_type(cloud_type)
                max_mae = max(expert_maes.values())

                if max_mae > LEARNING_ACCELERATED_THRESHOLD:
                    smoothing = LEARNING_ACCELERATED_FACTOR
                else:
                    smoothing = LEARNING_SMOOTHING_FACTOR

                smoothed_weights = {}
                for expert_name in new_weights:
                    old_w = old_weights.get(expert_name, 0.33)
                    new_w = new_weights[expert_name]
                    smoothed_weights[expert_name] = (
                        (1 - smoothing) * old_w + smoothing * new_w
                    )

                total = sum(smoothed_weights.values())
                if total > 0:
                    smoothed_weights = {k: v / total for k, v in smoothed_weights.items()}

                await self.blender.update_weights(cloud_type, smoothed_weights)
                result["weights_updated"][type_key] = {
                    k: round(v, 4) for k, v in smoothed_weights.items()
                }

            await self._save_learning_history(date_str, result)

            result["success"] = True
            result["reason"] = "Weights updated successfully"
            result["skipped_night_hours"] = skipped_hours

            _LOGGER.info(
                "Expert learning for %s: %d hours compared (skipped %d night hours), MAE: %s",
                date_str,
                result["comparison_hours"],
                skipped_hours,
                ", ".join(f"{k}={v:.1f}" for k, v in result["mae_by_expert"].items())
            )

            return result

        except Exception as e:
            result["reason"] = f"Error: {str(e)}"
            _LOGGER.error("Error learning expert weights: %s", e, exc_info=True)
            return result

    def _calculate_weights_from_mae(
        self,
        mae_by_expert: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate new weights inversely proportional to MAE."""
        inv_maes = {}
        for expert, mae in mae_by_expert.items():
            safe_mae = max(mae, LEARNING_MIN_ERROR)
            inv_maes[expert] = 1.0 / safe_mae

        total_inv = sum(inv_maes.values())
        if total_inv <= 0:
            return {k: 1.0 / len(mae_by_expert) for k in mae_by_expert}

        return {k: v / total_inv for k, v in inv_maes.items()}

    async def _load_actual_weather(self, date_str: str) -> Dict[str, Dict[str, Any]]:
        """Load actual weather observations for a date."""
        try:
            if not await self._file_exists(self.actual_weather_file):
                return {}

            data = await self._read_json_file(self.actual_weather_file, None)
            if not data:
                return {}

            return data.get("hourly_data", {}).get(date_str, {})

        except Exception as e:
            _LOGGER.warning("Error loading actual weather: %s", e)
            return {}

    async def _load_forecast_data(
        self,
        date_str: str
    ) -> Dict[int, Dict[str, Any]]:
        """Load forecast data for a date."""
        try:
            if not await self._file_exists(self.forecast_cache_file):
                return {}

            data = await self._read_json_file(self.forecast_cache_file, None)
            if not data:
                return {}

            day_forecast = data.get("forecast", {}).get(date_str, {})

            result = {}
            for hour_str, hour_data in day_forecast.items():
                try:
                    result[int(hour_str)] = hour_data
                except ValueError:
                    continue

            return result

        except Exception as e:
            _LOGGER.warning("Error loading forecast data: %s", e)
            return {}

    async def _save_learning_history(
        self,
        date_str: str,
        result: Dict[str, Any]
    ) -> None:
        """Save learning result to history file."""
        try:
            history = await self._read_json_file(
                self.learning_history_file,
                {"version": "1.0", "daily_history": {}, "metadata": {}}
            )

            history["daily_history"][date_str] = {
                "mae_by_expert": result.get("mae_by_expert", {}),
                "mae_by_cloud_type": result.get("mae_by_cloud_type", {}),
                "weights_updated": result.get("weights_updated", {}),
                "comparison_hours": result.get("comparison_hours", 0),
                "learned_at": datetime.now().isoformat(),
            }

            history["metadata"]["last_learning_run"] = datetime.now().isoformat()
            history["metadata"]["total_learning_days"] = len(history["daily_history"])

            dates = sorted(history["daily_history"].keys())
            if len(dates) > 30:
                for old_date in dates[:-30]:
                    del history["daily_history"][old_date]

            await self._atomic_write_json(self.learning_history_file, history)

        except Exception as e:
            _LOGGER.warning("Error saving learning history: %s", e)

    async def get_learning_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of recent learning."""
        try:
            if not await self._file_exists(self.learning_history_file):
                return {"days": 0, "avg_mae": {}}

            data = await self._read_json_file(self.learning_history_file, None)
            if not data:
                return {"days": 0, "avg_mae": {}}

            history = data.get("daily_history", {})
            recent_dates = sorted(history.keys(), reverse=True)[:days]

            if not recent_dates:
                return {"days": 0, "avg_mae": {}}

            expert_errors: Dict[str, List[float]] = {}

            for date in recent_dates:
                day_data = history[date]
                for expert, mae in day_data.get("mae_by_expert", {}).items():
                    if expert not in expert_errors:
                        expert_errors[expert] = []
                    expert_errors[expert].append(mae)

            avg_mae = {
                expert: round(sum(errors) / len(errors), 2)
                for expert, errors in expert_errors.items()
                if errors
            }

            return {
                "days": len(recent_dates),
                "avg_mae": avg_mae,
                "current_weights": self.blender._weights.copy(),
            }

        except Exception as e:
            _LOGGER.warning("Error getting learning summary: %s", e)
            return {"days": 0, "avg_mae": {}}

    async def learn_snow_prediction_accuracy(
        self,
        date_str: str,
        hourly_predictions: List[Dict[str, Any]],
        hourly_actuals: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Learn how accurate snow predictions are locally.

        For users WITHOUT rain sensors, we learn by comparing:
        - API predicted snow (condition contains "snow" OR precip > 0 + temp < 2°C)
        - Actual production = 0 or near 0 despite expected radiation > 100 W/m²

        This gives us confidence in snow predictions for the SNOWY bucket.

        Args:
            date_str: Date to analyze (YYYY-MM-DD)
            hourly_predictions: List of hourly predictions with weather data
            hourly_actuals: Dict of hourly actual weather data (from weather_actual_tracker)

        Returns:
            Dict with snow prediction analysis results
        """
        result = {
            "success": False,
            "date": date_str,
            "snow_predictions": [],
            "total_snow_predicted": 0,
            "snow_confirmed": 0,
            "accuracy": 0.0,
            "experts_accuracy": {},
        }

        try:
            # Minimum solar radiation to detect snow (otherwise it's just dark/cloudy)
            MIN_RADIATION_FOR_SNOW_DETECTION = 100  # W/m²
            # Production threshold to consider "no production" (near zero)
            MAX_PRODUCTION_FOR_SNOW = 0.02  # kWh - essentially nothing

            snow_predictions = []

            for pred in hourly_predictions:
                hour = pred.get("target_hour")
                if hour is None:
                    continue

                weather_forecast = pred.get("weather_forecast") or pred.get("weather") or {}

                # Check if snow was predicted by the forecast
                condition = str(weather_forecast.get("condition", "")).lower()
                temp = weather_forecast.get("temperature", 10)
                precip = weather_forecast.get("precipitation_mm") or weather_forecast.get("precipitation", 0) or 0
                expected_radiation = weather_forecast.get("solar_radiation_wm2", 0)

                # Snow predicted if:
                # 1. Condition contains "snow" OR
                # 2. Precipitation > 0.5mm AND temp < 2°C (snow conditions)
                snow_predicted = (
                    "snow" in condition or
                    (precip > 0.5 and temp < 2.0)
                )

                if not snow_predicted:
                    continue

                # Can only verify snow if there should be radiation
                if expected_radiation < MIN_RADIATION_FOR_SNOW_DETECTION:
                    continue

                # Check actual outcome - did production drop to near zero?
                actual_production = pred.get("actual_kwh", 0) or 0
                actual_weather = hourly_actuals.get(str(hour), {})

                # Snow confirmed if: very low/no production despite expected radiation
                # OR if snow_covered_panels flag is set in actuals
                snow_confirmed = (
                    actual_production < MAX_PRODUCTION_FOR_SNOW or
                    actual_weather.get("snow_covered_panels", False)
                )

                snow_predictions.append({
                    "hour": hour,
                    "predicted": True,
                    "confirmed": snow_confirmed,
                    "expected_radiation": expected_radiation,
                    "actual_production": actual_production,
                    "condition": condition,
                    "temp": temp,
                    "precip": precip,
                })

            result["snow_predictions"] = snow_predictions
            result["total_snow_predicted"] = len(snow_predictions)

            if snow_predictions:
                correct = sum(1 for p in snow_predictions if p["confirmed"])
                total = len(snow_predictions)
                accuracy = correct / total if total > 0 else 0.0

                result["snow_confirmed"] = correct
                result["accuracy"] = round(accuracy, 3)
                result["success"] = True

                # Update weights for SNOW cloud type based on accuracy
                if accuracy > 0:
                    await self._update_snow_expert_weights(date_str, snow_predictions)

                _LOGGER.info(
                    f"Snow prediction learning for {date_str}: "
                    f"{correct}/{total} confirmed ({accuracy:.0%} accuracy)"
                )
            else:
                result["success"] = True
                _LOGGER.debug(f"No snow predictions to learn from on {date_str}")

            return result

        except Exception as e:
            _LOGGER.error(f"Error learning snow prediction accuracy: {e}", exc_info=True)
            result["reason"] = str(e)
            return result

    async def _update_snow_expert_weights(
        self,
        date_str: str,
        snow_predictions: List[Dict[str, Any]],
    ) -> None:
        """Update expert weights for SNOW cloud type based on prediction accuracy.

        Currently uses a simplified approach - boost weights for all experts
        when snow predictions are accurate (since we don't track per-expert predictions).

        Future enhancement: Track which expert predicted snow and update individually.
        """
        try:
            # For now, just track overall snow accuracy in the learning history
            # The weights update is handled by the main learn_from_day method
            # when it encounters SNOW cloud type hours

            accuracy = sum(1 for p in snow_predictions if p["confirmed"]) / len(snow_predictions)

            # Log the learning result
            _LOGGER.debug(
                f"Snow expert weights update: {len(snow_predictions)} predictions, "
                f"{accuracy:.0%} accuracy"
            )

            # Save to learning history
            history = await self._read_json_file(
                self.learning_history_file,
                {"version": "1.0", "daily_history": {}, "metadata": {}, "snow_learning": {}}
            )

            if "snow_learning" not in history:
                history["snow_learning"] = {}

            history["snow_learning"][date_str] = {
                "predictions": len(snow_predictions),
                "confirmed": sum(1 for p in snow_predictions if p["confirmed"]),
                "accuracy": accuracy,
                "learned_at": datetime.now().isoformat(),
            }

            # Update metadata with snow stats
            snow_history = history.get("snow_learning", {})
            total_predictions = sum(d.get("predictions", 0) for d in snow_history.values())
            total_confirmed = sum(d.get("confirmed", 0) for d in snow_history.values())
            overall_accuracy = total_confirmed / total_predictions if total_predictions > 0 else 0.0

            history["metadata"]["snow_total_predictions"] = total_predictions
            history["metadata"]["snow_total_confirmed"] = total_confirmed
            history["metadata"]["snow_overall_accuracy"] = round(overall_accuracy, 3)
            history["metadata"]["last_snow_learning"] = datetime.now().isoformat()

            await self._atomic_write_json(self.learning_history_file, history)

        except Exception as e:
            _LOGGER.warning(f"Error updating snow expert weights: {e}")
