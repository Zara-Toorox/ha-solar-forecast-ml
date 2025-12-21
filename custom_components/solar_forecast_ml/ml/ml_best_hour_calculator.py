"""Best Hour Calculator - Weather-based calculation of peak production hour V12.2.0 @zara

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
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

SOLAR_PRODUCTION_START_HOUR = 7
SOLAR_PRODUCTION_END_HOUR = 19

class BestHourCalculator:
    """Calculate best production hour using weather forecast. Strategy: 1. PRIMARY: Pure weather-based calculation (best weather conditions today) 2. FALLBACK: Historical profile if no weather data available This calculator is completely independent from the ML model and does not affect solar production forecasting, training, or predictions."""

    def __init__(self, data_manager):
        """Initialize calculator with data manager. Args: data_manager: DataManager instance for accessing weather and profile data @zara"""
        self.data_manager = data_manager

    async def calculate_best_hour_today(self) -> Tuple[Optional[int], Optional[float]]:
        """Calculate the hour with the best expected solar production TODAY. Uses a pure weather-based approach: 1. BEST CASE: Analyzes today's weather forecast to find hour with best conditions 2. FALLBACK: Uses historical profile if weather data is unavailable Note: This returns a weather score (0-100), NOT a kWh prediction. The score represents expected production quality based on weather conditions. Returns: Tuple of (best_hour, weather_score) or (None, None) if no data available @zara"""

        best_hour_weather, weather_score = await self._calculate_best_hour_from_weather()

        if best_hour_weather is not None:
            _LOGGER.info(
                f"✓ Best hour today: {best_hour_weather:02d}:00 (weather score: {weather_score:.1f}/100) "
                f"- weather-based calculation"
            )
            return best_hour_weather, weather_score

        best_hour_profile, profile_kwh = await self._get_best_hour_from_profile()

        if best_hour_profile is not None:
            _LOGGER.info(
                f"✓ Best hour today: {best_hour_profile:02d}:00 ({profile_kwh:.3f} kWh) "
                f"- profile-based (fallback - no weather data)"
            )

            return best_hour_profile, profile_kwh

        _LOGGER.warning("⚠ No data available for best hour calculation (no weather, no profile)")
        return None, None

    async def _get_best_hour_from_profile(self) -> Tuple[Optional[int], Optional[float]]:
        """Get best hour from historical hourly profile as a fallback. Returns: Tuple of (hour, avg_kwh) or (None, None) @zara"""
        try:
            profile_data = await self.data_manager.load_hourly_profile()

            if not profile_data or not hasattr(profile_data, "hourly_averages"):
                _LOGGER.debug("No hourly profile data available for fallback.")
                return None, None

            hourly_averages = profile_data.hourly_averages
            if not hourly_averages:
                _LOGGER.debug("Hourly averages empty in profile.")
                return None, None

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
        """Calculate best hour using Open-Meteo forecast data. @zara"""
        try:
            import json
            open_meteo_file = self.data_manager.data_dir / "data" / "open_meteo_cache.json"

            if not open_meteo_file.exists():
                _LOGGER.debug("No Open-Meteo cache available - will use profile fallback.")
                return None, None

            def _read_cache():
                with open(open_meteo_file, 'r') as f:
                    return json.load(f)

            cache = await self.data_manager.hass.async_add_executor_job(_read_cache)
            forecast = cache.get("forecast", {})

            if not forecast:
                _LOGGER.debug("Open-Meteo cache empty - will use profile fallback.")
                return None, None

            today = datetime.now().date().isoformat()
            valid_hours = []

            if today not in forecast:
                _LOGGER.debug(f"No Open-Meteo data for today ({today}) - will use profile fallback.")
                return None, None

            for hour_str, hour_data in forecast[today].items():
                try:
                    hour = int(hour_str)
                    if SOLAR_PRODUCTION_START_HOUR <= hour < SOLAR_PRODUCTION_END_HOUR:
                        if self._is_weather_data_valid(hour_data):
                            valid_hours.append({"hour": hour, "data": hour_data})
                        else:
                            _LOGGER.debug(f"Skipping hour {hour}:00 - invalid weather data.")
                except (ValueError, TypeError) as e:
                    _LOGGER.debug(f"Could not parse hour: {e}")
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
        """Validate weather data plausibility @zara"""
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
        """Calculate weather quality score (0-100) for solar production @zara"""
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
        """Format best hour for display @zara"""
        if hour is None:
            return "N/A"
        return f"{hour:02d}:00"
