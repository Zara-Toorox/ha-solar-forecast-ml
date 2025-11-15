"""
Best Hour Calculator - Weather-based calculation of peak production hour

Pure weather-based approach with graceful degradation to historical profile.
Completely independent from ML model - does not affect training or predictions.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

# Configuration for solar production hours
SOLAR_PRODUCTION_START_HOUR = 7  # Start of potential solar production
SOLAR_PRODUCTION_END_HOUR = 19  # End of potential solar production (exclusive)


class BestHourCalculator:
    """Calculate best production hour using weather forecast. Strategy: 1. PRIMARY: Pure weather-based calculation (best weather conditions today) 2. FALLBACK: Historical profile if no weather data available This calculator is completely independent from the ML model and does not affect solar production forecasting, training, or predictions."""

    def __init__(self, data_manager):
        """Initialize calculator with data manager. Args: data_manager: DataManager instance for accessing weather and profile data"""
        self.data_manager = data_manager

    async def calculate_best_hour_today(self) -> Tuple[Optional[int], Optional[float]]:
        """Calculate the hour with the best expected solar production TODAY. Uses a pure weather-based approach: 1. BEST CASE: Analyzes today's weather forecast to find hour with best conditions 2. FALLBACK: Uses historical profile if weather data is unavailable Note: This returns a weather score (0-100), NOT a kWh prediction. The score represents expected production quality based on weather conditions. Returns: Tuple of (best_hour, weather_score) or (None, None) if no data available"""
        # PRIMARY METHOD: Weather-based calculation
        best_hour_weather, weather_score = await self._calculate_best_hour_from_weather()

        if best_hour_weather is not None:
            _LOGGER.info(
                f"✓ Best hour today: {best_hour_weather:02d}:00 (weather score: {weather_score:.1f}/100) "
                f"- weather-based calculation"
            )
            return best_hour_weather, weather_score

        # FALLBACK: Historical profile (when NO weather data available)
        best_hour_profile, profile_kwh = await self._get_best_hour_from_profile()

        if best_hour_profile is not None:
            _LOGGER.info(
                f"✓ Best hour today: {best_hour_profile:02d}:00 ({profile_kwh:.3f} kWh) "
                f"- profile-based (fallback - no weather data)"
            )
            # To maintain a consistent return type (hour, score), we return profile_kwh as score here.
            # The source is logged, so the distinction is clear.
            return best_hour_profile, profile_kwh

        # NO DATA AVAILABLE
        _LOGGER.warning("⚠ No data available for best hour calculation (no weather, no profile)")
        return None, None

    async def _get_best_hour_from_profile(self) -> Tuple[Optional[int], Optional[float]]:
        """Get best hour from historical hourly profile as a fallback. Returns: Tuple of (hour, avg_kwh) or (None, None)"""
        try:
            profile_data = await self.data_manager.load_hourly_profile()

            if not profile_data or not hasattr(profile_data, "hourly_averages"):
                _LOGGER.debug("No hourly profile data available for fallback.")
                return None, None

            hourly_averages = profile_data.hourly_averages
            if not hourly_averages:
                _LOGGER.debug("Hourly averages empty in profile.")
                return None, None

            # Find hour with maximum production in the solar window
            best_hour_str = max(
                (
                    hour_str
                    for hour_str in hourly_averages
                    if SOLAR_PRODUCTION_START_HOUR <= int(hour_str) < SOLAR_PRODUCTION_END_HOUR
                ),
                key=lambda h: hourly_averages.get(h, 0),
                default=None,
            )

            if best_hour_str is None:
                _LOGGER.debug("No production hours found in profile data within the solar window.")
                return None, None

            best_hour = int(best_hour_str)
            best_kwh = hourly_averages[best_hour_str]

            _LOGGER.debug(
                f"Profile fallback analysis: Best hour {best_hour}:00 with {best_kwh:.3f} kWh"
            )
            return best_hour, best_kwh

        except Exception as e:
            _LOGGER.error(f"Error calculating best hour from profile fallback: {e}", exc_info=True)
            return None, None

    async def _calculate_best_hour_from_weather(self) -> Tuple[Optional[int], Optional[float]]:
        """Calculate best hour using ONLY weather forecast data. Analyzes hours with available weather data within the solar production window, calculates a weather quality score (0-100), and returns the hour with the best score. Returns: Tuple of (best_hour, weather_score) or (None, None) if insufficient data."""
        try:
            weather_cache = await self.data_manager.load_weather_cache()
            if not weather_cache or "forecast_hours" not in weather_cache:
                _LOGGER.debug("No weather cache available - will use profile fallback.")
                return None, None

            forecast_hours = weather_cache.get("forecast_hours", [])
            if not forecast_hours:
                _LOGGER.debug("Weather cache empty - will use profile fallback.")
                return None, None

            today = datetime.now().date()
            valid_hours = []
            for hour_data in forecast_hours:
                try:
                    hour_dt_str = hour_data.get("local_datetime")
                    if not hour_dt_str:
                        continue

                    hour_dt = datetime.fromisoformat(hour_dt_str.replace("Z", "+00:00"))

                    if (
                        hour_dt.date() == today
                        and SOLAR_PRODUCTION_START_HOUR <= hour_dt.hour < SOLAR_PRODUCTION_END_HOUR
                    ):
                        if self._is_weather_data_valid(hour_data):
                            valid_hours.append({"hour": hour_dt.hour, "data": hour_data})
                        else:
                            _LOGGER.debug(
                                f"Skipping hour {hour_dt.hour}:00 - invalid weather data."
                            )
                except Exception as e:
                    _LOGGER.debug(f"Could not parse hour datetime: {e}")
                    continue

            if not valid_hours:
                _LOGGER.debug(
                    "No valid weather data for solar hours today - using profile fallback."
                )
                return None, None

            hour_scores = {
                hour_info["hour"]: self._calculate_weather_score(hour_info["data"])
                for hour_info in valid_hours
            }

            if not hour_scores:
                return None, None

            best_hour = max(hour_scores, key=hour_scores.get)
            best_score = hour_scores[best_hour]

            _LOGGER.debug(
                f"Weather analysis complete: Best hour is {best_hour:02d}:00 with score {best_score:.1f}/100."
            )
            return best_hour, best_score

        except Exception as e:
            _LOGGER.error(f"Error in weather-based best hour calculation: {e}", exc_info=True)
            return None, None

    def _is_weather_data_valid(self, weather_data: Dict[str, Any]) -> bool:
        """
        Validate weather data plausibility.
        """
        try:
            cloud_cover = weather_data.get("cloud_cover")
            if not isinstance(cloud_cover, (int, float)) or not (0 <= cloud_cover <= 100):
                return False

            ghi = weather_data.get("ghi")
            if ghi is not None and (not isinstance(ghi, (int, float)) or not (0 <= ghi <= 1500)):
                return False

            precip = weather_data.get("precipitation_probability")
            if precip is not None and (
                not isinstance(precip, (int, float)) or not (0 <= precip <= 100)
            ):
                return False

            return True
        except Exception:
            return False

    def _calculate_weather_score(self, weather_data: Dict[str, Any]) -> float:
        """
        Calculate weather quality score (0-100) for solar production.
        """
        cloud_cover = weather_data.get("cloud_cover", 50)
        base_score = 100 - cloud_cover

        ghi = weather_data.get("ghi")
        ghi_bonus = 0.0
        if ghi is not None and ghi > 0:
            ghi_bonus = min(20.0, (ghi / 800.0) * 20.0)

        precip = weather_data.get("precipitation_probability", 0)
        precip_penalty = (precip / 100.0) * 20.0

        final_score = max(0.0, min(100.0, base_score + ghi_bonus - precip_penalty))
        return final_score

    def get_best_hour_display(self, hour: Optional[int]) -> str:
        """
        Format best hour for display.
        """
        if hour is None:
            return "N/A"
        return f"{hour:02d}:00"
