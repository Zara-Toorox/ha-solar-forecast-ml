# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Weather Data Processor Module.
Processes and transforms weather data for ML features.
@zara
"""

import logging
from typing import Any, Dict, List

_LOGGER = logging.getLogger(__name__)


class WeatherDataProcessor:
    """Processes and transforms weather data for ML features. @zara"""

    @staticmethod
    def process_forecast_data(forecast_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw forecast data into standardized format. @zara

        Args:
            forecast_list: List of raw forecast entries

        Returns:
            List of processed forecast entries
        """
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
        """Estimate solar radiation based on cloud cover and time. @zara

        Args:
            cloud_cover: Cloud coverage percentage (0-100)
            hour: Hour of day (0-23)
            latitude: Geographic latitude

        Returns:
            Estimated solar radiation in W/m2
        """
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
        """Normalize weather features for ML input. @zara

        Args:
            weather_data: Raw weather data dictionary

        Returns:
            Dictionary of normalized features (0-1 range)
        """
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
    def extract_radiation_components(weather_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract radiation components from weather data. @zara

        Args:
            weather_data: Weather data dictionary

        Returns:
            Dictionary with ghi, dni, dhi values
        """
        ghi = WeatherDataProcessor._safe_float(
            weather_data.get("ghi", weather_data.get("solar_radiation", 0)), 0.0
        )
        dni = WeatherDataProcessor._safe_float(
            weather_data.get("dni", weather_data.get("direct_radiation", 0)), 0.0
        )
        dhi = WeatherDataProcessor._safe_float(
            weather_data.get("dhi", weather_data.get("diffuse_radiation", 0)), 0.0
        )

        # Estimate DHI from GHI if not available
        if dhi <= 0 and ghi > 0:
            cloud_cover = weather_data.get("cloud_coverage", weather_data.get("clouds", 50))
            if cloud_cover is not None:
                # Higher cloud cover = more diffuse radiation
                diffuse_ratio = 0.3 + (cloud_cover / 100.0) * 0.4
                dhi = ghi * diffuse_ratio
            else:
                dhi = ghi * 0.3

        # Estimate DNI from GHI if not available
        if dni <= 0 and ghi > 0:
            dni = ghi - dhi if dhi > 0 else ghi * 0.7

        return {
            "ghi": ghi,
            "dni": dni,
            "dhi": dhi,
        }

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        """Safely convert value to float with default. @zara

        Args:
            value: Value to convert
            default: Default if conversion fails

        Returns:
            Float value
        """
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to range 0-1. @zara

        Args:
            value: Value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value

        Returns:
            Normalized value (0-1)
        """
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    @staticmethod
    def validate_weather_data(weather_data: Dict[str, Any]) -> bool:
        """Validate weather data has required fields. @zara

        Args:
            weather_data: Weather data dictionary

        Returns:
            True if data is valid
        """
        required_fields = ["temperature"]
        for field in required_fields:
            if weather_data.get(field) is None:
                return False
        return True

    @staticmethod
    def fill_missing_values(weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing values with sensible defaults. @zara

        Args:
            weather_data: Weather data dictionary

        Returns:
            Dictionary with filled values
        """
        defaults = {
            "temperature": 15.0,
            "humidity": 60.0,
            "cloud_coverage": 50.0,
            "clouds": 50.0,
            "wind_speed": 5.0,
            "pressure": 1013.0,
            "precipitation": 0.0,
            "ghi": 0.0,
            "dni": 0.0,
            "dhi": 0.0,
        }

        result = weather_data.copy()
        for key, default in defaults.items():
            if result.get(key) is None:
                result[key] = default

        return result
