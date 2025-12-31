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
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from ..astronomy.astronomy_cache_manager import get_cache_manager
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from .data_io import DataManagerIO
from .data_open_meteo_client import OpenMeteoClient, get_default_weather
from .data_weather_expert_blender import CloudType, classify_cloud_type, cloud_to_transmission

_LOGGER = logging.getLogger(__name__)

DEFAULT_WEATHER = {
    "temperature": 15.0,
    "solar_radiation_wm2": 150.0,
    "wind": 3.0,
    "humidity": 70.0,
    "rain": 0.0,
    "clouds": 50.0,
    "pressure": 1013.0,
    "source": "default_fallback",
    "confidence": 0.1,
}


class WeatherForecastCorrector(DataManagerIO):
    """Creates corrected weather forecasts by applying precision factors to Open-Meteo data.

    Architecture:
    - Open-Meteo provides raw regional forecast data
    - Local sensors provide actual measurements (hourly_weather_actual.json)
    - Precision tracking calculates correction factors (weather_precision_daily.json)
    - WeatherExpertBlender provides multi-source blended cloud cover
    - This class APPLIES those factors to create weather_forecast_corrected.json
    """

    def __init__(self, hass: HomeAssistant, data_manager, expert_blender=None):
        super().__init__(hass, data_manager.data_dir)

        self.data_manager = data_manager
        self.stats_dir = data_manager.data_dir / "stats"
        self.corrected_file = data_manager.weather_corrected_file
        self.precision_file = self.stats_dir / "weather_precision_daily.json"

        # Expert blender for multi-source cloud cover
        self._expert_blender = expert_blender

        self._open_meteo_client: Optional[OpenMeteoClient] = None
        self._open_meteo_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._wttr_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._consecutive_failures: int = 0
        self._last_error: Optional[str] = None

        try:
            latitude = hass.config.latitude
            longitude = hass.config.longitude
            if latitude and longitude:
                cache_file = data_manager.data_dir / "data" / "open_meteo_cache.json"
                self._open_meteo_client = OpenMeteoClient(
                    latitude, longitude,
                    cache_file=cache_file
                )
                # V12.3: Corrector reads from Open-Meteo cache, blending via ExpertBlender
                self._open_meteo_client.auto_save_cache = False
                _LOGGER.info(
                    f"WeatherForecastCorrector initialized "
                    f"(lat={latitude:.4f}, lon={longitude:.4f})"
                    f"{' with ExpertBlender' if expert_blender else ''}"
                )
            else:
                _LOGGER.warning("No coordinates in HA config - Open-Meteo disabled")
        except Exception as e:
            _LOGGER.warning(f"Could not initialize Open-Meteo client: {e}")

    def set_expert_blender(self, expert_blender) -> None:
        """Set the expert blender for multi-source cloud cover."""
        self._expert_blender = expert_blender
        _LOGGER.info("WeatherForecastCorrector: Expert blender attached")

    def _get_weighted_poa_radiation(self, date_str: str, hour: int) -> float:
        """Get capacity-weighted POA (tilted) radiation from astronomy cache.

        V12.4.2: Uses poa_wm2 per panel group weighted by kWp capacity.
        This is MUCH more accurate than horizontal clear_sky_solar_radiation_wm2
        because it accounts for panel tilt and azimuth.

        Returns:
            Weighted average POA radiation in W/m², or 0.0 if not available.
        """
        try:
            cache_manager = get_cache_manager()
            if not cache_manager.is_loaded():
                return 0.0

            hourly_astro = cache_manager.get_hourly_data(date_str, hour)
            if not hourly_astro:
                return 0.0

            # Get per-group POA data
            groups = hourly_astro.get("theoretical_max_per_group", [])
            if not groups:
                # Fallback to horizontal if no group data (shouldn't happen)
                return hourly_astro.get("clear_sky_solar_radiation_wm2", 0.0) or 0.0

            # Calculate capacity-weighted POA radiation
            total_poa_weighted = 0.0
            total_capacity = 0.0

            for group in groups:
                poa_wm2 = group.get("poa_wm2", 0.0) or 0.0
                capacity_kwp = group.get("power_kwp", 0.0) or 0.0

                if capacity_kwp > 0:
                    total_poa_weighted += poa_wm2 * capacity_kwp
                    total_capacity += capacity_kwp

            if total_capacity <= 0:
                return hourly_astro.get("clear_sky_solar_radiation_wm2", 0.0) or 0.0

            weighted_poa = total_poa_weighted / total_capacity

            # Log for debugging (only for production hours)
            horizontal = hourly_astro.get("clear_sky_solar_radiation_wm2", 0.0) or 0.0
            if horizontal > 0 and weighted_poa > 0:
                ratio = weighted_poa / horizontal
                _LOGGER.debug(
                    f"POA radiation {date_str} {hour:02d}:00: "
                    f"horizontal={horizontal:.1f} W/m², weighted_poa={weighted_poa:.1f} W/m² "
                    f"(ratio={ratio:.2f}x)"
                )

            return weighted_poa

        except Exception as e:
            _LOGGER.debug(f"Could not get weighted POA radiation for {date_str} {hour:02d}:00: {e}")
            return 0.0

    def set_wttr_cache(self, wttr_cache: Dict[str, Dict[int, Dict[str, Any]]]) -> None:
        """Set wttr.in cache for fallback."""
        self._wttr_cache = wttr_cache

    async def async_init(self) -> bool:
        if self._open_meteo_client:
            await self._open_meteo_client.async_init()
            await self._load_open_meteo_cache()
        return True

    async def _load_open_meteo_cache(self) -> bool:
        if not self._open_meteo_client:
            return False

        try:
            cached_forecast = self._open_meteo_client._cache.get("hourly_forecast", [])
            if not cached_forecast:
                _LOGGER.debug("No Open-Meteo data in client cache")
                return False

            self._open_meteo_cache.clear()
            for entry in cached_forecast:
                date = entry.get("date")
                hour = entry.get("hour")

                if date and hour is not None:
                    if date not in self._open_meteo_cache:
                        self._open_meteo_cache[date] = {}

                    direct = entry.get("direct_radiation") or 0
                    diffuse = entry.get("diffuse_radiation") or 0

                    self._open_meteo_cache[date][hour] = {
                        "temperature": entry.get("temperature"),
                        "humidity": entry.get("humidity"),
                        "cloud_cover": entry.get("cloud_cover"),
                        "cloud_cover_low": entry.get("cloud_cover_low"),
                        "cloud_cover_mid": entry.get("cloud_cover_mid"),
                        "cloud_cover_high": entry.get("cloud_cover_high"),
                        "precipitation": entry.get("precipitation"),
                        "wind_speed": entry.get("wind_speed"),
                        "pressure": entry.get("pressure"),
                        "direct_radiation": direct,
                        "diffuse_radiation": diffuse,
                        "ghi": entry.get("ghi") or (direct + diffuse),
                    }

            if self._open_meteo_cache:
                _LOGGER.info(
                    f"Loaded Open-Meteo cache: {len(cached_forecast)} hours, "
                    f"{len(self._open_meteo_cache)} days"
                )
                return True

            return False

        except Exception as e:
            _LOGGER.warning(f"Error loading Open-Meteo cache: {e}")
            return False

    async def _fetch_open_meteo_forecast(self) -> bool:
        """Load weather data from Open-Meteo cache file. @zara

        V12.3: Reads from Open-Meteo cache. ExpertBlender handles all blending
        separately and writes to weather_forecast_corrected.json.
        """
        if not self._open_meteo_client:
            _LOGGER.debug("Open-Meteo client not available")
            return False

        try:
            # Reload the file cache to get latest data
            if self._open_meteo_client._cache_file:
                await self._open_meteo_client._load_file_cache()
                _LOGGER.debug("Reloaded Open-Meteo cache file")

            cached_forecast = self._open_meteo_client._cache.get("hourly_forecast", [])
            if not cached_forecast:
                _LOGGER.warning("No Open-Meteo data in cache file")
                return False

            self._open_meteo_cache.clear()
            for entry in cached_forecast:
                date = entry.get("date")
                hour = entry.get("hour")

                if date and hour is not None:
                    if date not in self._open_meteo_cache:
                        self._open_meteo_cache[date] = {}

                    direct = entry.get("direct_radiation") or 0
                    diffuse = entry.get("diffuse_radiation") or 0

                    self._open_meteo_cache[date][hour] = {
                        "temperature": entry.get("temperature"),
                        "humidity": entry.get("humidity"),
                        "cloud_cover": entry.get("cloud_cover"),
                        "cloud_cover_low": entry.get("cloud_cover_low"),
                        "cloud_cover_mid": entry.get("cloud_cover_mid"),
                        "cloud_cover_high": entry.get("cloud_cover_high"),
                        "precipitation": entry.get("precipitation"),
                        "wind_speed": entry.get("wind_speed"),
                        "pressure": entry.get("pressure"),
                        "direct_radiation": direct,
                        "diffuse_radiation": diffuse,
                        "ghi": entry.get("ghi") or (direct + diffuse),
                    }

            _LOGGER.info(
                f"Loaded weather cache: {len(cached_forecast)} hours, "
                f"{len(self._open_meteo_cache)} days"
            )
            return True

        except Exception as e:
            _LOGGER.warning(f"Error loading weather cache: {e}")
            return False

    def _get_weather_for_hour(self, date_str: str, hour: int) -> Dict[str, Any]:
        """Get weather for hour with calculated radiation from POA + clouds.

        V12.4.2: Radiation is calculated from TILTED (POA) radiation + cloud cover.
        Uses capacity-weighted poa_wm2 per panel group, NOT horizontal radiation!
        """
        if date_str not in self._open_meteo_cache:
            _LOGGER.debug(f"No Open-Meteo data for {date_str} {hour:02d}:00, using defaults")
            return self._get_default_weather_for_hour(hour, date_str)

        hour_data = self._open_meteo_cache[date_str].get(hour)
        if not hour_data:
            _LOGGER.debug(f"No Open-Meteo data for {date_str} {hour:02d}:00, using defaults")
            return self._get_default_weather_for_hour(hour, date_str)

        # Use Open-Meteo cloud cover as base (sync version - no blending)
        cloud_cover = hour_data.get("cloud_cover")
        cloud_source = "open-meteo-raw"
        blend_info = None

        # V12.4.2: Use TILTED (POA) radiation, not horizontal!
        poa_radiation = self._get_weighted_poa_radiation(date_str, hour)

        # Calculate actual radiation based on cloud cover
        cloud_factor = (100.0 - (cloud_cover or 0.0)) / 100.0
        calculated_radiation = poa_radiation * cloud_factor

        return {
            "temperature": hour_data.get("temperature"),
            "solar_radiation_wm2": round(calculated_radiation, 1),
            "wind": hour_data.get("wind_speed"),
            "humidity": hour_data.get("humidity"),
            "rain": hour_data.get("precipitation") or 0,
            "clouds": cloud_cover,
            "cloud_cover_low": hour_data.get("cloud_cover_low"),
            "cloud_cover_mid": hour_data.get("cloud_cover_mid"),
            "cloud_cover_high": hour_data.get("cloud_cover_high"),
            "pressure": hour_data.get("pressure"),
            # V12.4.2: Calculated radiation split from POA
            "direct_radiation": round(calculated_radiation * 0.7, 1),
            "diffuse_radiation": round(calculated_radiation * 0.3, 1),
            "clear_sky_poa_radiation": round(poa_radiation, 1),
            "source": cloud_source,
            "blend_info": blend_info,
            "confidence": 0.95,
        }

    async def _get_weather_for_hour_async(self, date_str: str, hour: int) -> Dict[str, Any]:
        """Get weather for hour with dual-path blended cloud cover (async version).

        V12.4.2: Radiation is now CALCULATED from TILTED (POA) radiation + blended clouds.
        Uses capacity-weighted poa_wm2 per panel group, NOT horizontal radiation!

        Formula: solar_radiation = poa_radiation * (100 - clouds) / 100
        The physics engine's bucket learning (per cloud% range, per panel group)
        handles the fine-tuning automatically.
        """
        if date_str not in self._open_meteo_cache:
            _LOGGER.debug(f"No Open-Meteo data for {date_str} {hour:02d}:00, using defaults")
            return self._get_default_weather_for_hour(hour, date_str)

        hour_data = self._open_meteo_cache[date_str].get(hour)
        if not hour_data:
            _LOGGER.debug(f"No Open-Meteo data for {date_str} {hour:02d}:00, using defaults")
            return self._get_default_weather_for_hour(hour, date_str)

        cloud_low = hour_data.get("cloud_cover_low")
        cloud_mid = hour_data.get("cloud_cover_mid")
        cloud_high = hour_data.get("cloud_cover_high")

        # Get wttr.in fallback cloud cover
        wttr_cloud = None
        if date_str in self._wttr_cache and hour in self._wttr_cache[date_str]:
            wttr_cloud = self._wttr_cache[date_str][hour].get("cloud_cover")

        # Use expert blender if available
        if self._expert_blender:
            try:
                # V12.4: Dual-path blending - combines cloud% and transmission approaches
                transmission, dual_info = await self._expert_blender.get_dual_path_blend(
                    date=date_str,
                    hour=hour,
                    cloud_low=cloud_low,
                    cloud_mid=cloud_mid,
                    cloud_high=cloud_high,
                    fallback_cloud=wttr_cloud,
                    path_a_weight=0.5,  # Equal weighting for both paths
                )
                # Use the final cloud value from dual-path blend
                cloud_cover = dual_info.get("final_cloud", 100.0 - transmission)
                cloud_source = "dual_path_blend"
                blend_info = dual_info
            except Exception as e:
                _LOGGER.debug(f"Dual-path blending failed for {date_str} {hour:02d}:00: {e}")
                # Fallback to simple cloud% blending
                try:
                    cloud_cover, blend_info = await self._expert_blender.get_blended_cloud_cover(
                        date=date_str,
                        hour=hour,
                        cloud_low=cloud_low,
                        cloud_mid=cloud_mid,
                        cloud_high=cloud_high,
                        fallback_cloud=wttr_cloud,
                    )
                    cloud_source = "expert_blend_fallback"
                except Exception as e2:
                    _LOGGER.debug(f"Expert blending also failed: {e2}")
                    cloud_cover = wttr_cloud if wttr_cloud is not None else hour_data.get("cloud_cover")
                    cloud_source = "wttr_fallback" if wttr_cloud is not None else "open-meteo-raw"
                    blend_info = {"error": str(e), "fallback_error": str(e2)}
        else:
            # No expert blender - use Open-Meteo cloud cover
            cloud_cover = hour_data.get("cloud_cover")
            cloud_source = "open-meteo-raw"
            blend_info = None

        # V12.4.2: Use TILTED (POA) radiation, not horizontal!
        # This is the key fix - uses poa_wm2 per panel group weighted by kWp
        poa_radiation = self._get_weighted_poa_radiation(date_str, hour)

        # Calculate actual radiation based on cloud cover
        # Formula: radiation = poa * (100 - clouds) / 100
        cloud_factor = (100.0 - (cloud_cover or 0.0)) / 100.0
        calculated_radiation = poa_radiation * cloud_factor

        return {
            "temperature": hour_data.get("temperature"),
            "solar_radiation_wm2": round(calculated_radiation, 1),
            "wind": hour_data.get("wind_speed"),
            "humidity": hour_data.get("humidity"),
            "rain": hour_data.get("precipitation") or 0,
            "clouds": cloud_cover,
            "cloud_cover_low": cloud_low,
            "cloud_cover_mid": cloud_mid,
            "cloud_cover_high": cloud_high,
            "pressure": hour_data.get("pressure"),
            # V12.4.2: Calculated radiation split from POA
            "direct_radiation": round(calculated_radiation * 0.7, 1),  # ~70% direct
            "diffuse_radiation": round(calculated_radiation * 0.3, 1),  # ~30% diffuse
            "clear_sky_poa_radiation": round(poa_radiation, 1),
            "source": cloud_source,
            "blend_info": blend_info,
            "confidence": 0.98 if cloud_source == "dual_path_blend" else 0.95 if "expert" in cloud_source else 0.85,
        }

    def _get_default_weather_for_hour(self, hour: int, date_str: str) -> Dict[str, Any]:
        import math
        hour_offset = (hour - 14) / 24.0 * 2 * math.pi
        temp_amplitude = 5.0
        temp = DEFAULT_WEATHER["temperature"] + temp_amplitude * math.cos(hour_offset)

        default = DEFAULT_WEATHER.copy()
        default["temperature"] = round(temp, 1)
        default["source"] = "default_fallback"
        default["confidence"] = 0.1

        return default

    async def _load_correction_factors(self) -> Dict[str, Any]:
        """Load precision correction factors from weather_precision_daily.json"""
        default_factors = {
            "temperature": 0.0,
            "solar_radiation_wm2": 1.0,
            "clouds": 1.0,
            "wind": 1.0,
            "humidity": 1.0,
            "rain": 1.0,
            "pressure": 0.0,
        }
        default_confidence = {k: 0.0 for k in default_factors}
        default_hourly = {"solar_radiation_wm2": {}, "clouds": {}}

        try:
            if not self.precision_file.exists():
                _LOGGER.debug("No precision file found - using default factors (no correction)")
                return {
                    "factors": default_factors,
                    "factors_clear": default_factors.copy(),
                    "factors_cloudy": default_factors.copy(),
                    "confidence": default_confidence,
                    "sample_days": 0,
                    "sample_days_clear": 0,
                    "sample_days_cloudy": 0,
                    "hourly_factors": default_hourly,
                }

            precision_data = await self._read_json_file(self.precision_file, None)
            if not precision_data:
                return {
                    "factors": default_factors,
                    "factors_clear": default_factors.copy(),
                    "factors_cloudy": default_factors.copy(),
                    "confidence": default_confidence,
                    "sample_days": 0,
                    "sample_days_clear": 0,
                    "sample_days_cloudy": 0,
                    "hourly_factors": default_hourly,
                }

            rolling = precision_data.get("rolling_averages", {})
            factors = rolling.get("correction_factors", default_factors)
            factors_clear = rolling.get("correction_factors_clear", factors.copy())
            factors_cloudy = rolling.get("correction_factors_cloudy", factors.copy())
            confidence = rolling.get("confidence", default_confidence)
            sample_days = rolling.get("sample_days", 0)
            sample_days_clear = rolling.get("sample_days_clear", 0)
            sample_days_cloudy = rolling.get("sample_days_cloudy", 0)
            hourly_factors = rolling.get("hourly_factors", default_hourly)

            for key in default_factors:
                if key not in factors:
                    factors[key] = default_factors[key]
                if key not in factors_clear:
                    factors_clear[key] = factors[key]
                if key not in factors_cloudy:
                    factors_cloudy[key] = factors[key]
                if key not in confidence:
                    confidence[key] = 0.0

            return {
                "factors": factors,
                "factors_clear": factors_clear,
                "factors_cloudy": factors_cloudy,
                "confidence": confidence,
                "sample_days": sample_days,
                "sample_days_clear": sample_days_clear,
                "sample_days_cloudy": sample_days_cloudy,
                "hourly_factors": hourly_factors,
            }

        except Exception as e:
            _LOGGER.warning(f"Error loading correction factors: {e}")
            return {
                "factors": default_factors,
                "factors_clear": default_factors.copy(),
                "factors_cloudy": default_factors.copy(),
                "confidence": default_confidence,
                "sample_days": 0,
                "sample_days_clear": 0,
                "sample_days_cloudy": 0,
                "hourly_factors": default_hourly,
            }

    def _apply_correction(self, raw_value: float, factor: float, field_type: str) -> float:
        """Apply correction factor to a raw value based on field type"""
        if raw_value is None:
            return None

        if field_type in ("temperature", "pressure"):
            return round(raw_value + factor, 1)
        else:
            corrected = raw_value * factor
            if field_type == "clouds":
                corrected = max(0.0, min(100.0, corrected))
            elif field_type in ("solar_radiation_wm2", "wind", "humidity", "rain"):
                corrected = max(0.0, corrected)
            return round(corrected, 1)

    def _get_hourly_factor(
        self,
        hourly_factors: Dict[str, Dict[str, Any]],
        field: str,
        hour: int,
        daily_fallback: float
    ) -> float:
        """Get correction factor for a specific field and hour."""
        field_hourly = hourly_factors.get(field, {})
        hour_data = field_hourly.get(str(hour))

        if hour_data and "factor" in hour_data:
            return hour_data["factor"]

        return daily_fallback

    def _get_weather_specific_ghi_factor(
        self,
        cloud_cover: float,
        factors: Dict[str, float],
        factors_clear: Dict[str, float],
        factors_cloudy: Dict[str, float],
        sample_days_clear: int,
        sample_days_cloudy: int,
        min_samples: int = 2
    ) -> float:
        """Get GHI correction factor based on cloud cover."""
        WEATHER_CLEAR_MAX_CLOUDS = 30
        WEATHER_CLOUDY_MIN_CLOUDS = 50

        daily_factor = factors.get("solar_radiation_wm2", 1.0)
        clear_factor = factors_clear.get("solar_radiation_wm2", daily_factor)
        cloudy_factor = factors_cloudy.get("solar_radiation_wm2", daily_factor)

        has_clear = sample_days_clear >= min_samples
        has_cloudy = sample_days_cloudy >= min_samples

        if cloud_cover is None:
            return daily_factor

        if cloud_cover <= WEATHER_CLEAR_MAX_CLOUDS:
            if has_clear:
                return clear_factor
            return daily_factor

        if cloud_cover >= WEATHER_CLOUDY_MIN_CLOUDS:
            if has_cloudy:
                return cloudy_factor
            return daily_factor

        if has_clear and has_cloudy:
            t = (cloud_cover - WEATHER_CLEAR_MAX_CLOUDS) / (WEATHER_CLOUDY_MIN_CLOUDS - WEATHER_CLEAR_MAX_CLOUDS)
            return clear_factor + t * (cloudy_factor - clear_factor)

        return daily_factor

    async def create_corrected_forecast(self, min_confidence: float = 0.0) -> bool:
        """Create corrected weather forecast with precision factors applied."""
        try:
            correction_data = await self._load_correction_factors()
            factors = correction_data["factors"]
            factors_clear = correction_data.get("factors_clear", factors)
            factors_cloudy = correction_data.get("factors_cloudy", factors)
            confidence = correction_data["confidence"]
            sample_days = correction_data["sample_days"]
            sample_days_clear = correction_data.get("sample_days_clear", 0)
            sample_days_cloudy = correction_data.get("sample_days_cloudy", 0)
            hourly_factors = correction_data.get("hourly_factors", {})

            # Calculate average confidence across all fields
            confidence_values = [v for v in confidence.values() if isinstance(v, (int, float))]
            avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0

            # Check if confidence meets minimum threshold
            apply_corrections = sample_days > 0 and avg_confidence >= min_confidence

            # Check if hourly factors are available
            has_hourly_solar = bool(hourly_factors.get("solar_radiation_wm2", {}))
            has_hourly_clouds = bool(hourly_factors.get("clouds", {}))

            has_weather_specific = sample_days_clear >= 2 or sample_days_cloudy >= 2

            if apply_corrections:
                clear_solar = factors_clear.get("solar_radiation_wm2", 1.0)
                cloudy_solar = factors_cloudy.get("solar_radiation_wm2", 1.0)
                if has_weather_specific:
                    _LOGGER.info(
                        f"Creating corrected forecast - {sample_days} days "
                        f"(clear={sample_days_clear}, cloudy={sample_days_cloudy}), "
                        f"solar_rad: daily={factors.get('solar_radiation_wm2', 1.0):.3f}, "
                        f"clear={clear_solar:.3f}, cloudy={cloudy_solar:.3f}"
                    )
                elif has_hourly_solar:
                    solar_hourly = hourly_factors["solar_radiation_wm2"]
                    hour_summary = ", ".join(
                        f"{h}h:{d['factor']:.2f}"
                        for h, d in sorted(solar_hourly.items(), key=lambda x: int(x[0]))
                    )
                    _LOGGER.info(
                        f"Creating corrected forecast - applying {sample_days}-day precision factors "
                        f"with HOURLY solar_rad=[{hour_summary}], "
                        f"daily clouds={factors.get('clouds', 1.0):.3f}"
                    )
                else:
                    _LOGGER.info(
                        f"Creating corrected forecast - applying {sample_days}-day precision factors "
                        f"(daily solar_rad={factors.get('solar_radiation_wm2', 1.0):.3f}, "
                        f"clouds={factors.get('clouds', 1.0):.3f}, avg_confidence={avg_confidence:.2f})"
                    )
            elif sample_days > 0 and avg_confidence < min_confidence:
                _LOGGER.info(
                    f"Creating forecast - confidence too low ({avg_confidence:.2f} < {min_confidence:.2f}), "
                    "using raw Open-Meteo values"
                )
                factors = {k: (0.0 if k in ("temperature", "pressure") else 1.0) for k in factors}
                factors_clear = factors.copy()
                factors_cloudy = factors.copy()
                hourly_factors = {}
                sample_days_clear = 0
                sample_days_cloudy = 0
            else:
                _LOGGER.info("Creating forecast - no precision data yet (using raw Open-Meteo values)")

            open_meteo_available = await self._fetch_open_meteo_forecast()
            if not open_meteo_available:
                _LOGGER.warning("Open-Meteo unavailable - using cached data or defaults")

            astronomy_cache_file = self.stats_dir / "astronomy_cache.json"
            astronomy_data = {}
            if astronomy_cache_file.exists():
                try:
                    astro_cache = await self._read_json_file(astronomy_cache_file, None)
                    if astro_cache:
                        astronomy_data = astro_cache.get("days", {})
                except Exception as e:
                    _LOGGER.warning(f"Could not load astronomy cache: {e}")

            forecast_by_date: Dict[str, Dict[str, Dict[str, Any]]] = {}

            # Track which hourly factors were used for metadata
            hourly_factors_used: Dict[str, int] = {"solar_radiation_wm2": 0, "clouds": 0}

            for date_str in sorted(self._open_meteo_cache.keys()):
                day_data = self._open_meteo_cache[date_str]

                for hour in range(24):
                    # Use async version for expert blending
                    weather = await self._get_weather_for_hour_async(date_str, hour)

                    astronomy = {}
                    if date_str in astronomy_data:
                        hour_astro = astronomy_data[date_str].get("hourly", {}).get(str(hour), {})
                        astronomy = {
                            "sun_elevation": hour_astro.get("elevation_deg"),
                            "azimuth": hour_astro.get("azimuth_deg"),
                            "clear_sky_radiation": hour_astro.get("clear_sky_solar_radiation_wm2"),
                            "theoretical_max_kwh": hour_astro.get("theoretical_max_pv_kwh"),
                        }

                    if date_str not in forecast_by_date:
                        forecast_by_date[date_str] = {}

                    raw_solar = weather.get("solar_radiation_wm2") or 0
                    raw_clouds = weather.get("clouds")
                    raw_temp = weather.get("temperature")
                    raw_wind = weather.get("wind")
                    raw_humidity = weather.get("humidity")
                    raw_rain = weather.get("rain") or 0
                    raw_pressure = weather.get("pressure")

                    if has_weather_specific:
                        solar_factor = self._get_weather_specific_ghi_factor(
                            raw_clouds,
                            factors,
                            factors_clear,
                            factors_cloudy,
                            sample_days_clear,
                            sample_days_cloudy
                        )
                    else:
                        solar_factor = self._get_hourly_factor(
                            hourly_factors,
                            "solar_radiation_wm2",
                            hour,
                            factors.get("solar_radiation_wm2", 1.0)
                        )

                    cloud_factor = self._get_hourly_factor(
                        hourly_factors,
                        "clouds",
                        hour,
                        factors.get("clouds", 1.0)
                    )

                    if str(hour) in hourly_factors.get("solar_radiation_wm2", {}):
                        hourly_factors_used["solar_radiation_wm2"] += 1
                    if str(hour) in hourly_factors.get("clouds", {}):
                        hourly_factors_used["clouds"] += 1

                    corrected_clouds = raw_clouds if has_weather_specific else self._apply_correction(raw_clouds, cloud_factor, "clouds")
                    forecast_by_date[date_str][str(hour)] = {
                        "temperature": self._apply_correction(raw_temp, factors.get("temperature", 0.0), "temperature"),
                        "solar_radiation_wm2": self._apply_correction(raw_solar, solar_factor, "solar_radiation_wm2"),
                        "wind": self._apply_correction(raw_wind, factors.get("wind", 1.0), "wind"),
                        "humidity": self._apply_correction(raw_humidity, factors.get("humidity", 1.0), "humidity"),
                        "rain": self._apply_correction(raw_rain, factors.get("rain", 1.0), "rain"),
                        "clouds": corrected_clouds,
                        "cloud_cover_low": weather.get("cloud_cover_low"),
                        "cloud_cover_mid": weather.get("cloud_cover_mid"),
                        "cloud_cover_high": weather.get("cloud_cover_high"),
                        "pressure": self._apply_correction(raw_pressure, factors.get("pressure", 0.0), "pressure"),
                        "direct_radiation": weather.get("direct_radiation"),
                        "diffuse_radiation": weather.get("diffuse_radiation"),
                    }

            rb_correction = await self._get_preserved_rb_correction()

            mode = "precision_corrected"
            if has_weather_specific:
                mode = "precision_corrected_weather_specific"
            elif has_hourly_solar:
                mode = "precision_corrected_hourly"

            corrected_data = {
                "version": "4.3",
                "forecast": forecast_by_date,
                "metadata": {
                    "created": dt_util.now().isoformat(),
                    "source": "open-meteo-corrected",
                    "mode": mode,
                    "hours_forecast": sum(len(h) for h in forecast_by_date.values()),
                    "days_forecast": len(forecast_by_date),
                    "corrections_applied": {
                        "solar_radiation_wm2": factors.get("solar_radiation_wm2", 1.0),
                        "solar_radiation_wm2_clear": factors_clear.get("solar_radiation_wm2", 1.0),
                        "solar_radiation_wm2_cloudy": factors_cloudy.get("solar_radiation_wm2", 1.0),
                        "clouds": factors.get("clouds", 1.0),
                        "temperature": factors.get("temperature", 0.0),
                        "wind": factors.get("wind", 1.0),
                        "humidity": factors.get("humidity", 1.0),
                        "rain": factors.get("rain", 1.0),
                    },
                    "weather_specific_samples": {
                        "clear_days": sample_days_clear,
                        "cloudy_days": sample_days_cloudy,
                    },
                    "hourly_corrections": {
                        "solar_radiation_wm2": hourly_factors.get("solar_radiation_wm2", {}),
                        "clouds": hourly_factors.get("clouds", {}),
                        "hours_using_hourly_factors": hourly_factors_used,
                    },
                    "confidence_scores": confidence,
                    "precision_sample_days": sample_days,
                    "rb_overall_correction": rb_correction,
                    "self_healing": {
                        "open_meteo_status": "ok" if open_meteo_available else "using_cache",
                        "consecutive_failures": self._consecutive_failures,
                        "last_error": self._last_error,
                    },
                }
            }

            total_hours = sum(len(h) for h in forecast_by_date.values())

            # CRITICAL: Only mark as success if we actually have forecast data!
            # An empty forecast is not valid and downstream components shouldn't
            # think they have a working corrected forecast.
            if total_hours == 0:
                _LOGGER.error(
                    "Cannot create corrected forecast: No weather data available. "
                    "Check Open-Meteo cache and Multi-Weather Blender status."
                )
                return False

            await self._atomic_write_json(self.corrected_file, corrected_data)

            _LOGGER.info(
                f"Created corrected forecast: {total_hours} hours, {sample_days} days precision learning"
                + (f", {hourly_factors_used['solar_radiation_wm2']} hours using hourly solar factors"
                   if hourly_factors_used["solar_radiation_wm2"] > 0 else "")
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Error creating forecast: {e}", exc_info=True)
            return False

    async def _get_preserved_rb_correction(self) -> Dict[str, Any]:
        default_rb = {
            "factor": 1.0,
            "confidence": 0.0,
            "sample_days": 0,
            "last_updated": dt_util.now().isoformat(),
            "note": "Installation-specific correction factor"
        }

        if self.corrected_file.exists():
            try:
                existing = await self._read_json_file(self.corrected_file, None)
                if existing and "metadata" in existing:
                    existing_rb = existing["metadata"].get("rb_overall_correction", {})
                    if existing_rb and existing_rb.get("factor", 1.0) != 1.0:
                        return {
                            "factor": existing_rb.get("factor", 1.0),
                            "confidence": existing_rb.get("confidence", 0.0),
                            "sample_days": existing_rb.get("sample_days", 0),
                            "last_updated": existing_rb.get("last_updated", dt_util.now().isoformat()),
                            "note": existing_rb.get("note", "Installation-specific correction factor")
                        }
            except Exception:
                pass

        return default_rb

    async def get_corrected_weather_for_hour(self, target_datetime: datetime) -> Optional[Dict[str, Any]]:
        date_str = target_datetime.date().isoformat()
        hour = target_datetime.hour

        try:
            if self.corrected_file.exists():
                data = await self._read_json_file(self.corrected_file, None)
                if data:
                    forecast = data.get("forecast", {})
                    if date_str in forecast and str(hour) in forecast[date_str]:
                        result = forecast[date_str][str(hour)]
                        result["_source"] = "corrected_forecast"
                        result["_confidence"] = 0.95
                        return result

            weather = await self._get_weather_for_hour_async(date_str, hour)
            weather["_source"] = "open_meteo_direct"
            return weather

        except Exception as e:
            _LOGGER.error(f"Error getting weather for hour: {e}")
            return self._get_default_weather_for_hour(hour, date_str)

    async def calculate_rolling_accuracy_factor(self, lookback_days: int = 5) -> float:
        """Berechne gewichteten Faktor aus den letzten N Tagen (neuere = mehr Gewicht).

        Dies ermöglicht schnellere Anpassung an aktuelle Bedingungen.
        """
        try:
            predictions_file = self.stats_dir / "hourly_predictions.json"
            if not predictions_file.exists():
                return 1.0

            predictions_data = await self._read_json_file(predictions_file, None)
            if not predictions_data:
                return 1.0

            predictions = predictions_data.get("predictions", [])

            # Gruppiere nach Datum (nur letzte lookback_days Tage)
            from datetime import datetime, timedelta
            cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

            daily_totals = {}
            for pred in predictions:
                date = pred.get("target_date")
                if not date or date < cutoff:
                    continue

                actual = pred.get("actual_kwh")
                predicted = pred.get("prediction_kwh")
                ml_contrib = pred.get("ml_contribution_percent", 0)

                # Nur RB-dominierte Vorhersagen
                if ml_contrib > 75 or actual is None or predicted is None or predicted <= 0:
                    continue

                if date not in daily_totals:
                    daily_totals[date] = {"actual": 0.0, "predicted": 0.0}
                daily_totals[date]["actual"] += actual
                daily_totals[date]["predicted"] += predicted

            if not daily_totals:
                return 1.0

            # Gewichtete Berechnung (neuere Tage mehr Gewicht)
            sorted_dates = sorted(daily_totals.keys())[-lookback_days:]
            weighted_factor = 0.0
            total_weight = 0.0

            for i, date in enumerate(sorted_dates):
                weight = (i + 1) / len(sorted_dates)  # 0.2, 0.4, 0.6, 0.8, 1.0
                data = daily_totals[date]
                if data["predicted"] > 0:
                    daily_factor = data["actual"] / data["predicted"]
                    weighted_factor += daily_factor * weight
                    total_weight += weight

            if total_weight <= 0:
                return 1.0

            rolling_factor = weighted_factor / total_weight
            rolling_factor = max(0.35, min(1.25, rolling_factor))  # Clamping

            _LOGGER.debug(
                f"Rolling {lookback_days}-day accuracy factor: {rolling_factor:.3f} "
                f"(from {len(sorted_dates)} days)"
            )

            return rolling_factor

        except Exception as e:
            _LOGGER.warning(f"Error calculating rolling accuracy factor: {e}")
            return 1.0

    async def update_rb_correction(self, date_str: str) -> bool:
        try:
            _LOGGER.debug(f"Updating RB correction for {date_str}")

            predictions_file = self.stats_dir / "hourly_predictions.json"
            if not predictions_file.exists():
                return False

            predictions_data = await self._read_json_file(predictions_file, None)
            if not predictions_data:
                return False

            predictions = predictions_data.get("predictions", [])

            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            cutoff_date = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")

            daily_data = {}
            for pred in predictions:
                pred_date = pred.get("target_date")
                if not pred_date or pred_date < cutoff_date or pred_date > date_str:
                    continue

                actual_kwh = pred.get("actual_kwh")
                prediction_kwh = pred.get("prediction_kwh")
                ml_contrib = pred.get("ml_contribution_percent", 0)

                if actual_kwh is None or prediction_kwh is None or prediction_kwh <= 0:
                    continue

                rb_weight = 1.0 - (ml_contrib / 100.0)
                if rb_weight < 0.25:
                    continue

                if pred_date not in daily_data:
                    daily_data[pred_date] = {"rb_total": 0.0, "actual_total": 0.0, "hours": 0}

                # Time-weight: Peak hours (10-15) count more
                hour = pred.get("target_hour", 12)
                time_weight = 1.2 if 10 <= hour <= 15 else 0.8

                daily_data[pred_date]["rb_total"] += prediction_kwh * time_weight
                daily_data[pred_date]["actual_total"] += actual_kwh * time_weight
                daily_data[pred_date]["hours"] += time_weight

            # CUMULATIVE calculation instead of averaging daily factors
            # This fixes the issue where old days with factor > 1.0 would skew the average
            total_actual_all = sum(d["actual_total"] for d in daily_data.values() if d["hours"] > 0)
            total_predicted_all = sum(d["rb_total"] for d in daily_data.values() if d["hours"] > 0 and d["rb_total"] > 0)
            sample_days = len([d for d in daily_data.values() if d["hours"] > 0 and d["rb_total"] > 0])

            if total_predicted_all <= 0 or sample_days < 1:
                return False

            # Single cumulative factor across ALL days (more accurate than daily average)
            avg_factor = total_actual_all / total_predicted_all

            # CLAMPING: Prevent unrealistic correction factors
            avg_factor = max(0.35, min(1.25, avg_factor))

            if avg_factor > 1.05:
                _LOGGER.info(
                    f"RB Correction: factor={avg_factor:.3f} - RBS underpredicted by {(avg_factor-1)*100:.1f}%. "
                    f"Cumulative over {sample_days} days: actual={total_actual_all:.2f} kWh, predicted={total_predicted_all:.2f} kWh"
                )
            elif avg_factor < 0.95:
                _LOGGER.info(
                    f"RB Correction: factor={avg_factor:.3f} - RBS overpredicted by {(1-avg_factor)*100:.1f}%. "
                    f"Cumulative over {sample_days} days: actual={total_actual_all:.2f} kWh, predicted={total_predicted_all:.2f} kWh"
                )
            else:
                _LOGGER.debug(
                    f"RB Correction: factor={avg_factor:.3f} (within ±5%). "
                    f"Cumulative over {sample_days} days: actual={total_actual_all:.2f} kWh, predicted={total_predicted_all:.2f} kWh"
                )

            if sample_days == 1:
                confidence = 0.2
            elif sample_days < 3:
                confidence = 0.3
            elif sample_days < 7:
                confidence = 0.4
            elif sample_days < 30:
                confidence = 0.7
            else:
                confidence = 0.9

            # Combine with rolling factor for faster adaptation
            rolling_factor = await self.calculate_rolling_accuracy_factor(lookback_days=5)

            # Blend: 60% rolling (recent), 40% cumulative (stable)
            if sample_days >= 3:
                blended_factor = 0.6 * rolling_factor + 0.4 * avg_factor
                _LOGGER.info(
                    f"Blending factors: rolling={rolling_factor:.3f}, cumulative={avg_factor:.3f} → blended={blended_factor:.3f}"
                )
                avg_factor = blended_factor
                avg_factor = max(0.35, min(1.25, avg_factor))  # Re-apply clamping

            if not self.corrected_file.exists():
                return False

            corrected_data = await self._read_json_file(self.corrected_file, None)
            if not corrected_data:
                return False

            if "metadata" not in corrected_data:
                corrected_data["metadata"] = {}

            corrected_data["metadata"]["rb_overall_correction"] = {
                "factor": round(avg_factor, 3),
                "confidence": round(confidence, 2),
                "sample_days": sample_days,
                "rolling_factor": round(rolling_factor, 3),
                "last_updated": dt_util.now().isoformat(),
                "note": "Installation-specific correction factor"
            }

            await self._atomic_write_json(self.corrected_file, corrected_data)

            _LOGGER.info(
                f"Updated RB correction: factor={avg_factor:.3f}, "
                f"confidence={confidence:.1f}, samples={sample_days}"
            )
            return True

        except Exception as e:
            _LOGGER.error(f"Error updating RB correction: {e}", exc_info=True)
            return False

    async def get_correction_summary(self) -> Dict[str, Any]:
        try:
            if self.corrected_file.exists():
                data = await self._read_json_file(self.corrected_file, None)
                if data and "metadata" in data:
                    return data["metadata"]
            return {}
        except Exception as e:
            _LOGGER.error(f"Error getting correction summary: {e}")
            return {}
