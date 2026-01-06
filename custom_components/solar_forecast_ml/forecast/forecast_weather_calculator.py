# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)

class WeatherCalculator:
    """Calculates various rule-based weather adjustment factors based on simple heur..."""

    def __init__(self):
        """Initializes the WeatherCalculator with predefined factor mappings @zara"""

        self.SEASONAL_FACTORS = {
            "winter": 0.35,
            "spring": 0.75,
            "summer": 1.0,
            "autumn": 0.65,
        }

        self.SEASONAL_MONTH_MAPPING = {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }

        self.OPTIMAL_TEMPERATURE_C = 25.0
        self.TEMP_EFFICIENCY_LOSS_PER_C = (
            0.004
        )

        self.CONDITION_FACTORS = {

            "rainy": 0.40,
            "pouring": 0.20,
            "snowy": 0.30,
            "snowy-rainy": 0.25,
            "hail": 0.20,
            "lightning": 0.50,
            "lightning-rainy": 0.35,
            "fog": 0.45,
            "windy": 0.95,
            "windy-variant": 0.95,
            "exceptional": 0.50,

        }

    def get_temperature_factor(self, temperature_c: Optional[float]) -> float:
        """Calculates a simple efficiency factor based on ambient temperature @zara"""
        if temperature_c is None:
            return 0.9

        try:
            if temperature_c < 0:

                return 0.85
            elif temperature_c <= self.OPTIMAL_TEMPERATURE_C:

                return 0.85 + (temperature_c / self.OPTIMAL_TEMPERATURE_C) * 0.15
            else:

                loss = (
                    temperature_c - self.OPTIMAL_TEMPERATURE_C
                ) * self.TEMP_EFFICIENCY_LOSS_PER_C
                factor = 1.0 - loss

                return max(0.70, factor)
        except (ValueError, TypeError) as e:
            _LOGGER.warning(
                f"Temperature factor calculation failed for value '{temperature_c}': {e}. Using default 0.9."
            )
            return 0.9

    def get_cloud_factor(self, cloud_coverage_percent: Optional[float], condition: Optional[str] = None) -> float:
        """Calculates a simple factor based on cloud coverage percentage @zara"""
        if cloud_coverage_percent is None:

            if condition:
                condition_lower = condition.lower()

                if condition_lower in ["rainy", "pouring", "snowy", "snowy-rainy", "hail"]:
                    return 0.15

                elif condition_lower in ["lightning", "lightning-rainy", "fog"]:
                    return 0.25

                elif condition_lower in ["cloudy"]:
                    return 0.35
                elif condition_lower in ["partlycloudy"]:
                    return 0.65

                elif condition_lower in ["sunny", "clear", "clear-night"]:
                    return 0.95

            return 0.6

        try:

            coverage = max(0.0, min(100.0, float(cloud_coverage_percent)))

            import math
            CLOUD_DECAY_RATE = 0.002
            MIN_CLOUD_FACTOR = 0.35

            cloud_factor = max(MIN_CLOUD_FACTOR, math.exp(-CLOUD_DECAY_RATE * coverage))

            return cloud_factor
        except (ValueError, TypeError) as e:
            _LOGGER.warning(
                f"Cloud factor calculation failed for value '{cloud_coverage_percent}': {e}. Using condition-aware default."
            )

            return self.get_cloud_factor(None, condition)

    def calculate_effective_radiation(
        self,
        clear_sky_radiation_wm2: float,
        cloud_coverage_percent: Optional[float],
        condition: Optional[str] = None
    ) -> float:
        """
        Calculate effective solar radiation based on clear-sky radiation and cloud coverage.

        This replaces the unreliable lux values from weather APIs with a physics-based calculation.

        Formula:
            cloud_factor = e^(-0.008 × cloud_percent)
            effective_radiation = clear_sky × cloud_factor

        Args:
            clear_sky_radiation_wm2: Theoretical maximum radiation (W/m²) from astronomy
            cloud_coverage_percent: Cloud coverage 0-100% (from weather forecast)
            condition: Optional weather condition string for fallback

        Returns:
            Effective solar radiation in W/m² (always <= clear_sky)

        Example:
            clear_sky = 147.8 W/m², clouds = 50%
            → cloud_factor = e^(-0.4) = 0.670
            → effective = 147.8 × 0.670 = 99.0 W/m²

            At 100% clouds: factor = 0.449 (44.9% transmission, 55.1% reduction)
        """
        if clear_sky_radiation_wm2 <= 0:
            return 0.0

        cloud_factor = self.get_cloud_factor(cloud_coverage_percent, condition)

        effective_radiation = clear_sky_radiation_wm2 * cloud_factor

        return min(effective_radiation, clear_sky_radiation_wm2)

    def get_condition_factor(self, condition: Optional[str]) -> float:
        """Gets a reduction factor based on specific adverse weather conditions @zara"""
        if not condition or not isinstance(condition, str):
            return 1.0

        try:
            condition_lower = condition.lower()

            return self.CONDITION_FACTORS.get(condition_lower, 1.0)

        except Exception as e:

            _LOGGER.warning(
                f"Condition factor calculation failed for condition '{condition}': {e}. Using default 1.0."
            )
            return 1.0

    def get_seasonal_adjustment(self, now: Optional[datetime] = None) -> float:
        """Calculates a seasonal adjustment factor based on the month @zara"""
        try:

            if now is None:
                now_local = dt_util.now()

            elif now.tzinfo is None:
                now_local = dt_util.ensure_local(now)
            else:
                now_local = dt_util.as_local(now)

            month = now_local.month

            season = self.SEASONAL_MONTH_MAPPING.get(
                month, "autumn"
            )

            factor = self.SEASONAL_FACTORS.get(season, 0.65)

            if month in [12, 1]:
                factor *= 0.85
            elif month in [6, 7]:
                factor *= 1.05

            return max(0.2, min(1.2, factor))

        except Exception as e:
            _LOGGER.warning(
                f"Seasonal adjustment calculation failed: {e}. Using default 0.65 (Autumn).",
                exc_info=True,
            )
            return 0.65

    def get_current_season(self, now: Optional[datetime] = None) -> str:
        """Returns the current season name winter spring summer autumn @zara"""
        try:
            if now is None:
                now = dt_util.now()
            elif now.tzinfo is None:
                now = dt_util.ensure_local(now)
            else:
                now = dt_util.as_local(now)

            month = now.month
            return self.SEASONAL_MONTH_MAPPING.get(month, "autumn")
        except Exception as e:
            _LOGGER.warning(f"Failed to determine current season: {e}. Defaulting to 'autumn'.")
            return "autumn"

    def estimate_cloud_factor_from_sensors(
        self,
        actual_radiation_wm2: Optional[float] = None,
        theoretical_max_radiation_wm2: Optional[float] = None,
        actual_lux: Optional[float] = None,
        theoretical_max_lux: Optional[float] = None,
    ) -> Optional[float]:
        """Estimate cloud factor from real sensor data (LUX or W/m²)

        Uses the ratio of actual vs theoretical maximum radiation to estimate cloud cover.
        Only valid during daylight hours when sensors have reliable data.

        Returns:
            Cloud factor (0.0-1.0) or None if insufficient data
        """
        try:
            cloud_factor = None

            if (
                actual_radiation_wm2 is not None
                and theoretical_max_radiation_wm2 is not None
                and theoretical_max_radiation_wm2 > 0
            ):

                ratio = actual_radiation_wm2 / theoretical_max_radiation_wm2
                cloud_factor = max(0.0, min(1.0, ratio))
                _LOGGER.debug(
                    f"Cloud factor from W/m²: {cloud_factor:.2f} "
                    f"(actual={actual_radiation_wm2:.1f}, max={theoretical_max_radiation_wm2:.1f})"
                )

            elif (
                actual_lux is not None
                and theoretical_max_lux is not None
                and theoretical_max_lux > 0
            ):
                ratio = actual_lux / theoretical_max_lux
                cloud_factor = max(0.0, min(1.0, ratio))
                _LOGGER.debug(
                    f"Cloud factor from LUX: {cloud_factor:.2f} "
                    f"(actual={actual_lux:.1f}, max={theoretical_max_lux:.1f})"
                )

            return cloud_factor

        except Exception as e:
            _LOGGER.warning(f"Cloud factor estimation from sensors failed: {e}")
            return None

    def blend_cloud_factors(
        self,
        weather_cloud_factor: Optional[float],
        sensor_cloud_factor: Optional[float],
        condition: Optional[str] = None,
        sensor_weight: float = 0.7,
    ) -> float:
        """Blend cloud factors from multiple sources with smart weighting

        Priority:
        1. Sensor data (most accurate, but only available during daylight)
        2. Weather forecast cloud_cover percentage
        3. Condition-aware defaults (fallback when numerical data missing)

        Args:
            weather_cloud_factor: Cloud factor from weather forecast (0.0-1.0)
            sensor_cloud_factor: Cloud factor from real sensors (0.0-1.0)
            condition: Weather condition string for fallback defaults
            sensor_weight: Weight for sensor data (0.0-1.0), default 0.7

        Returns:
            Blended cloud factor (0.0-1.0)
        """
        try:

            if sensor_cloud_factor is not None and weather_cloud_factor is not None:
                blended = (
                    sensor_weight * sensor_cloud_factor
                    + (1 - sensor_weight) * weather_cloud_factor
                )
                _LOGGER.debug(
                    f"Blended cloud factor: {blended:.2f} "
                    f"(sensor={sensor_cloud_factor:.2f} @ {sensor_weight:.0%}, "
                    f"weather={weather_cloud_factor:.2f} @ {1-sensor_weight:.0%})"
                )
                return blended

            elif sensor_cloud_factor is not None:
                _LOGGER.debug(f"Using sensor cloud factor: {sensor_cloud_factor:.2f}")
                return sensor_cloud_factor

            elif weather_cloud_factor is not None:
                _LOGGER.debug(f"Using weather cloud factor: {weather_cloud_factor:.2f}")
                return weather_cloud_factor

            else:
                default_factor = self.get_cloud_factor(None, condition)
                _LOGGER.debug(
                    f"Using condition-aware default cloud factor: {default_factor:.2f} (condition={condition})"
                )
                return default_factor

        except Exception as e:
            _LOGGER.error(f"Cloud factor blending failed: {e}", exc_info=True)

            return self.get_cloud_factor(None, condition)

    def calculate_combined_weather_factor(
        self, weather_data: Dict[str, Any], include_seasonal: bool = True
    ) -> float:
        """Combines temperature cloud condition and optionally seasonal factors"""
        try:

            temp_c = weather_data.get("temperature")

            cloud_perc = weather_data.get("cloud_cover", weather_data.get("clouds"))
            condition_str = weather_data.get("condition")

            temp_factor = self.get_temperature_factor(temp_c)
            cloud_factor = self.get_cloud_factor(cloud_perc, condition_str)
            condition_factor = self.get_condition_factor(condition_str)

            combined_factor = temp_factor * cloud_factor * condition_factor

            if include_seasonal:
                seasonal_factor = self.get_seasonal_adjustment()
                combined_factor *= seasonal_factor

            return max(0.0, combined_factor)

        except Exception as e:
            _LOGGER.error(f"Combined weather factor calculation failed: {e}", exc_info=True)
            return 0.5
