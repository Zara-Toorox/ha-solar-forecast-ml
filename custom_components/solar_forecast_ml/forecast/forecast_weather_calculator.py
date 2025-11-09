"""
Weather Calculation Utilities

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

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional # Added Optional

# Use SafeDateTimeUtil for consistent timezone handling
from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class WeatherCalculator:
    """Calculates various rule-based weather adjustment factors based on simple heur..."""

    def __init__(self):
        """Initializes the WeatherCalculator with predefined factor mappings"""
        # Seasonal base factors (adjust based on general solar intensity per season)
        self.SEASONAL_FACTORS = {
            "winter": 0.35, # Slightly higher base for winter
            "spring": 0.75, # Higher base for spring
            "summer": 1.0,  # Peak season
            "autumn": 0.65  # Lower base for autumn
        }

        # Mapping month number to season name
        self.SEASONAL_MONTH_MAPPING = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn"
        }

        # Temperature related parameters
        self.OPTIMAL_TEMPERATURE_C = 25.0  # Assumed optimal panel operating temperature (Celsius)
        self.TEMP_EFFICIENCY_LOSS_PER_C = 0.004 # Approx. 0.4% efficiency loss per degree C above optimal

        # Factors for specific adverse weather conditions (multipliers < 1.0)
        self.CONDITION_FACTORS = {
            # Standard HA conditions (lowercase)
            "rainy": 0.40,          # Rain significantly reduces output
            "pouring": 0.20,        # Heavy rain even more so
            "snowy": 0.30,          # Snow can cover panels or reflect light differently
            "snowy-rainy": 0.25,    # Mix of snow and rain
            "hail": 0.20,           # Hail implies heavy clouds and potential impact
            "lightning": 0.50,      # Thunderstorms often have dark clouds, but can be brief
            "lightning-rainy": 0.35,# Thunderstorm with rain
            "fog": 0.45,            # Fog diffuses light
            "windy": 0.95,          # High wind might slightly cool panels, minor negative impact assumed otherwise
            "windy-variant": 0.95,
            "exceptional": 0.50,    # Unknown severe condition, assume significant reduction
            # Consider adding others if relevant: 'cloudy', 'partlycloudy' handled by cloud factor
        }

    def get_temperature_factor(self, temperature_c: Optional[float]) -> float:
        """Calculates a simple efficiency factor based on ambient temperature"""
        if temperature_c is None:
             return 0.9 # Default factor if temp is unavailable

        try:
            if temperature_c < 0:
                # Reduced efficiency at very low temperatures
                return 0.85 # Slightly higher than original guess
            elif temperature_c <= self.OPTIMAL_TEMPERATURE_C:
                # Linear increase in efficiency up to the optimal temperature
                # Starts at 0°C reaches 1.0 at OPTIMAL_TEMPERATURE_C
                return 0.85 + (temperature_c / self.OPTIMAL_TEMPERATURE_C) * 0.15
            else:
                # Linear decrease in efficiency above the optimal temperature
                loss = (temperature_c - self.OPTIMAL_TEMPERATURE_C) * self.TEMP_EFFICIENCY_LOSS_PER_C
                factor = 1.0 - loss
                # Ensure factor doesn't drop too low (e.g., minimum 70% efficiency assumed)
                return max(0.70, factor)
        except (ValueError, TypeError) as e:
            _LOGGER.warning(f"Temperature factor calculation failed for value '{temperature_c}': {e}. Using default 0.9.")
            return 0.9  # Fallback on calculation error

    def get_cloud_factor(self, cloud_coverage_percent: Optional[float]) -> float:
        """Calculates a simple factor based on cloud coverage percentage"""
        if cloud_coverage_percent is None:
            return 0.6 # Default factor for unknown cloud cover (partly cloudy guess)

        try:
            # Ensure coverage is within 0-100 range
            coverage = max(0.0, min(100.0, float(cloud_coverage_percent)))

            # Non-linear mapping: more impact from initial clouds
            if coverage < 10:
                return 1.0  # Clear sky
            elif coverage < 30:
                return 0.9  # Mostly sunny / Few clouds
            elif coverage < 60:
                return 0.65 # Partly cloudy / Scattered clouds
            elif coverage < 90:
                return 0.35 # Mostly cloudy / Broken clouds
            else:
                return 0.15 # Overcast
        except (ValueError, TypeError) as e:
            _LOGGER.warning(f"Cloud factor calculation failed for value '{cloud_coverage_percent}': {e}. Using default 0.6.")
            return 0.6  # Fallback on calculation error

    def get_condition_factor(self, condition: Optional[str]) -> float:
        """Gets a reduction factor based on specific adverse weather conditions"""
        if not condition or not isinstance(condition, str):
            return 1.0 # No condition specified or invalid type

        try:
            condition_lower = condition.lower()
            # Return specific factor if condition is adverse, otherwise default to 1.0 (no reduction)
            return self.CONDITION_FACTORS.get(condition_lower, 1.0)

        except Exception as e:
            # Catch potential errors if condition is unexpected type after check
            _LOGGER.warning(f"Condition factor calculation failed for condition '{condition}': {e}. Using default 1.0.")
            return 1.0  # Fallback on error

    def get_seasonal_adjustment(self, now: Optional[datetime] = None) -> float:
        """Calculates a seasonal adjustment factor based on the month"""
        try:
            # Use current local time if no specific time is provided
            if now is None:
                now_local = dt_util.now()
            # Ensure provided datetime is timezone-aware (use local timezone if naive)
            elif now.tzinfo is None:
                 now_local = dt_util.ensure_local(now)
            else:
                 now_local = dt_util.as_local(now)


            month = now_local.month
            # Determine season based on month
            season = self.SEASONAL_MONTH_MAPPING.get(month, "autumn") # Default to autumn if month mapping fails
            # Get base factor for the season
            factor = self.SEASONAL_FACTORS.get(season, 0.65) # Default to autumn factor

            # Apply additional modifiers for extreme months
            if month in [12, 1]:  # Deep winter modifier
                factor *= 0.85 # Slightly less harsh than 0.8
            elif month in [6, 7]:  # Peak summer modifier
                factor *= 1.05 # Slightly less boost than 1.1

            # Clamp the final factor to a reasonable range [0.2, 1.2]
            return max(0.2, min(1.2, factor))

        except Exception as e:
            _LOGGER.warning(f"Seasonal adjustment calculation failed: {e}. Using default 0.65 (Autumn).", exc_info=True)
            return 0.65  # Fallback to a neutral season factor

    def get_current_season(self, now: Optional[datetime] = None) -> str:
        """Returns the current season name winter spring summer autumn"""
        try:
            if now is None: now = dt_util.now()
            elif now.tzinfo is None: now = dt_util.ensure_local(now)
            else: now = dt_util.as_local(now)

            month = now.month
            return self.SEASONAL_MONTH_MAPPING.get(month, "autumn") # Default to autumn
        except Exception as e:
            _LOGGER.warning(f"Failed to determine current season: {e}. Defaulting to 'autumn'.")
            return "autumn"

    def calculate_combined_weather_factor(
        self,
        weather_data: Dict[str, Any],
        include_seasonal: bool = True
    ) -> float:
        """Combines temperature cloud condition and optionally seasonal factors"""
        try:
            # Extract values safely from weather_data dict
            temp_c = weather_data.get("temperature")
            # Allow 'clouds' or 'cloud_cover' for flexibility
            cloud_perc = weather_data.get("cloud_cover", weather_data.get("clouds"))
            condition_str = weather_data.get("condition")

            # Calculate individual factors
            temp_factor = self.get_temperature_factor(temp_c)
            cloud_factor = self.get_cloud_factor(cloud_perc)
            condition_factor = self.get_condition_factor(condition_str)

            # Combine core weather factors
            combined_factor = temp_factor * cloud_factor * condition_factor

            # Include seasonal factor if requested
            if include_seasonal:
                seasonal_factor = self.get_seasonal_adjustment() # Uses current time
                combined_factor *= seasonal_factor


            # Ensure final factor is non-negative
            return max(0.0, combined_factor)

        except Exception as e:
            _LOGGER.error(f"Combined weather factor calculation failed: {e}", exc_info=True)
            return 0.5  # Return a neutral fallback value on error