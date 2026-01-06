# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

class WeatherDataProcessor:
    """Processes and transforms weather data for ML features"""

    @staticmethod
    def process_forecast_data(forecast_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw forecast data into standardized format @zara"""
        processed = []

        for entry in forecast_list:
            try:
                processed_entry = {
                    "datetime": entry.get("datetime"),
                    "temperature": WeatherDataProcessor._safe_float(entry.get("temperature"), 15.0),
                    "humidity": WeatherDataProcessor._safe_float(entry.get("humidity"), 60.0),
                    "cloud_coverage": WeatherDataProcessor._safe_float(
                        entry.get("cloud_coverage"), 50.0
                    ),
                    "wind_speed": WeatherDataProcessor._safe_float(entry.get("wind_speed"), 5.0),
                    "pressure": WeatherDataProcessor._safe_float(entry.get("pressure"), 1013.0),
                    "condition": entry.get("condition", "unknown"),
                    "precipitation": WeatherDataProcessor._safe_float(
                        entry.get("precipitation"), 0.0
                    ),
                }
                processed.append(processed_entry)
            except Exception as e:
                _LOGGER.warning(f"Error processing forecast entry: {e}")
                continue

        return processed

    @staticmethod
    def calculate_solar_radiation(cloud_cover: float, hour: int, latitude: float = 50.0) -> float:
        """Estimate solar radiation based on cloud cover and time @zara"""

        if hour < 6 or hour > 20:
            return 0.0

        hour_angle = abs(12 - hour) * 15
        elevation = 90 - abs(latitude) - hour_angle

        if elevation < 0:
            return 0.0

        max_radiation = 1000.0

        elevation_factor = elevation / 90.0

        cloud_factor = 1.0 - (cloud_cover / 100.0) * 0.75

        radiation = max_radiation * elevation_factor * cloud_factor

        return max(0.0, radiation)

    @staticmethod
    def normalize_weather_features(weather_data: Dict[str, Any]) -> Dict[str, float]:
        """Normalize weather features for ML input @zara"""
        return {
            "temperature_norm": WeatherDataProcessor._normalize(
                weather_data.get("temperature", 15.0), min_val=-20.0, max_val=40.0
            ),
            "humidity_norm": WeatherDataProcessor._normalize(
                weather_data.get("humidity", 60.0), min_val=0.0, max_val=100.0
            ),
            "cloud_coverage_norm": WeatherDataProcessor._normalize(
                weather_data.get("cloud_coverage", 50.0), min_val=0.0, max_val=100.0
            ),
            "wind_speed_norm": WeatherDataProcessor._normalize(
                weather_data.get("wind_speed", 5.0), min_val=0.0, max_val=30.0
            ),
            "pressure_norm": WeatherDataProcessor._normalize(
                weather_data.get("pressure", 1013.0), min_val=950.0, max_val=1050.0
            ),
        }

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        """Safely convert value to float with default @zara"""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to range 0 1 @zara"""
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
