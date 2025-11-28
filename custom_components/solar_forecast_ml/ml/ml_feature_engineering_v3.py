"""Feature Engineering V3 - Simplified with Corrected Weather Data V10.0.0 @zara

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
import math
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

class FeatureEngineerV3:
    """
    Feature engineering V3 - Simplified architecture

    Features (14 total):
    - Time (3): hour, day_of_year, season
    - Weather Corrected (6): temp, lux, wind, humidity, rain, clouds
    - Astronomy (3): elevation, theoretical_max, clear_sky_radiation
    - LAG (2): production_yesterday, production_same_hour_yesterday
    """

    def __init__(self):
        """Initialize feature engineer @zara"""
        self.feature_names = self._build_feature_names()
        _LOGGER.info(f"FeatureEngineerV3 initialized with {len(self.feature_names)} features")

    def _build_feature_names(self) -> List[str]:
        """Build complete feature list (14 features) @zara"""
        return [

            "hour_of_day",
            "day_of_year",
            "season_encoded",

            "weather_temp",
            "weather_solar_radiation_wm2",
            "weather_wind",
            "weather_humidity",
            "weather_rain",
            "weather_clouds",

            "sun_elevation_deg",
            "theoretical_max_kwh",
            "clear_sky_radiation_wm2",

            "production_yesterday",
            "production_same_hour_yesterday"
        ]

    def extract_features(self, record: Dict[str, Any]) -> Optional[List[float]]:
        """Extract feature vector from enriched record @zara"""
        try:

            target_hour = float(record.get("target_hour", 12))
            target_day_of_year = float(record.get("target_day_of_year", 180))
            target_month = int(record.get("target_month", 6))

            season_str = record.get("target_season", "")
            if not season_str:

                if target_month in [12, 1, 2]:
                    season_encoded = 0.0
                elif target_month in [3, 4, 5]:
                    season_encoded = 1.0
                elif target_month in [6, 7, 8]:
                    season_encoded = 2.0
                else:
                    season_encoded = 3.0
            else:
                season_map = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
                season_encoded = float(season_map.get(season_str, 2))

            weather = record.get("weather_corrected", {})

            try:
                weather_temp = float(weather.get("temperature", 15.0))
            except (TypeError, ValueError):
                weather_temp = 15.0

            try:
                weather_solar_wm2 = float(weather.get("solar_radiation_wm2", 0.0))
            except (TypeError, ValueError):
                weather_solar_wm2 = 0.0

            try:
                weather_wind = float(weather.get("wind", 3.0))
            except (TypeError, ValueError):
                weather_wind = 3.0

            try:
                weather_humidity = float(weather.get("humidity", 70.0))
            except (TypeError, ValueError):
                weather_humidity = 70.0

            try:
                weather_rain = float(weather.get("rain", 0.0))
            except (TypeError, ValueError):
                weather_rain = 0.0

            try:
                weather_clouds = float(weather.get("clouds", 50.0))
            except (TypeError, ValueError):
                weather_clouds = 50.0

            astronomy = record.get("astronomy", {})

            sun_elevation = float(astronomy.get("sun_elevation_deg", -30.0))
            theoretical_max_kwh = float(astronomy.get("theoretical_max_kwh", 0.0))
            clear_sky_radiation = float(astronomy.get("clear_sky_radiation_wm2", 0.0))

            production_yesterday = float(record.get("production_yesterday", 0.0))
            production_same_hour_yesterday = float(record.get("production_same_hour_yesterday", 0.0))

            features = [

                target_hour,
                target_day_of_year,
                season_encoded,

                weather_temp,
                weather_solar_wm2,
                weather_wind,
                weather_humidity,
                weather_rain,
                weather_clouds,

                sun_elevation,
                theoretical_max_kwh,
                clear_sky_radiation,

                production_yesterday,
                production_same_hour_yesterday
            ]

            if len(features) != len(self.feature_names):
                _LOGGER.error(
                    f"Feature count mismatch! Expected {len(self.feature_names)}, got {len(features)}"
                )
                return None

            for i, val in enumerate(features):
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    _LOGGER.warning(
                        f"Invalid value for feature {self.feature_names[i]}: {val}, setting to 0.0"
                    )
                    features[i] = 0.0

            return features

        except Exception as e:
            _LOGGER.error(f"Error extracting features: {e}", exc_info=True)
            return None

    def get_default_features(self) -> List[float]:
        """Get default feature vector (fallback when no data available) @zara"""
        return [

            12.0,
            180.0,
            2.0,

            15.0,
            0.0,
            3.0,
            70.0,
            0.0,
            50.0,

            -30.0,
            0.0,
            0.0,

            0.0,
            0.0
        ]

    def get_feature_importance_categories(self) -> Dict[str, List[str]]:
        """Get features grouped by category for analysis @zara"""
        return {
            "time": [
                "hour_of_day",
                "day_of_year",
                "season_encoded"
            ],
            "weather": [
                "weather_temp",
                "weather_solar_radiation_wm2",
                "weather_wind",
                "weather_humidity",
                "weather_rain",
                "weather_clouds"
            ],
            "astronomy": [
                "sun_elevation",
                "theoretical_max_kwh",
                "clear_sky_radiation"
            ],
            "lag": [
                "production_yesterday",
                "production_same_hour_yesterday"
            ]
        }

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate that record has all required fields @zara"""
        required_fields = [
            "target_hour",
            "target_day_of_year",
            "target_month",
            "weather_corrected",
            "astronomy"
        ]

        for field in required_fields:
            if field not in record:
                _LOGGER.warning(f"Missing required field in record: {field}")
                return False

        weather = record.get("weather_corrected", {})
        required_weather = ["temperature", "lux", "wind", "humidity", "rain", "clouds"]

        for field in required_weather:
            if field not in weather:
                _LOGGER.warning(f"Missing weather field: {field}")
                return False

        astronomy = record.get("astronomy", {})
        required_astro = ["sun_elevation_deg", "theoretical_max_kwh", "clear_sky_radiation_wm2"]

        for field in required_astro:
            if field not in astronomy:
                _LOGGER.warning(f"Missing astronomy field: {field}")
                return False

        return True
