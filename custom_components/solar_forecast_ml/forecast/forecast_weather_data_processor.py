"""
Weather Data Processor for Solar Forecast ML Integration

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
from datetime import datetime
from typing import Dict, Any, Optional, List

_LOGGER = logging.getLogger(__name__)


class WeatherDataProcessor:
    """Processes and transforms weather data for ML features by Zara"""
    
    @staticmethod
    def process_forecast_data(forecast_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw forecast data into standardized format by Zara"""
        processed = []
        
        for entry in forecast_list:
            try:
                processed_entry = {
                    "datetime": entry.get("datetime"),
                    "temperature": WeatherDataProcessor._safe_float(entry.get("temperature"), 15.0),
                    "humidity": WeatherDataProcessor._safe_float(entry.get("humidity"), 60.0),
                    "cloud_coverage": WeatherDataProcessor._safe_float(entry.get("cloud_coverage"), 50.0),
                    "wind_speed": WeatherDataProcessor._safe_float(entry.get("wind_speed"), 5.0),
                    "pressure": WeatherDataProcessor._safe_float(entry.get("pressure"), 1013.0),
                    "condition": entry.get("condition", "unknown"),
                    "precipitation": WeatherDataProcessor._safe_float(entry.get("precipitation"), 0.0),
                }
                processed.append(processed_entry)
            except Exception as e:
                _LOGGER.warning(f"Error processing forecast entry: {e}")
                continue
        
        return processed
    
    @staticmethod
    def calculate_solar_radiation(
        cloud_cover: float,
        hour: int,
        latitude: float = 50.0
    ) -> float:
        """Estimate solar radiation based on cloud cover and time by Zara"""
        # Simple solar radiation model
        # Peak radiation around noon, reduced by cloud cover
        
        if hour < 6 or hour > 20:
            return 0.0
        
        # Solar elevation angle (simplified)
        hour_angle = abs(12 - hour) * 15  # degrees from solar noon
        elevation = 90 - abs(latitude) - hour_angle
        
        if elevation < 0:
            return 0.0
        
        # Max radiation at elevation 90 degrees
        max_radiation = 1000.0  # W/m²
        
        # Reduce by elevation angle
        elevation_factor = elevation / 90.0
        
        # Reduce by cloud cover
        cloud_factor = 1.0 - (cloud_cover / 100.0) * 0.75
        
        radiation = max_radiation * elevation_factor * cloud_factor
        
        return max(0.0, radiation)
    
    @staticmethod
    def normalize_weather_features(weather_data: Dict[str, Any]) -> Dict[str, float]:
        """Normalize weather features for ML input by Zara"""
        return {
            "temperature_norm": WeatherDataProcessor._normalize(
                weather_data.get("temperature", 15.0),
                min_val=-20.0,
                max_val=40.0
            ),
            "humidity_norm": WeatherDataProcessor._normalize(
                weather_data.get("humidity", 60.0),
                min_val=0.0,
                max_val=100.0
            ),
            "cloud_coverage_norm": WeatherDataProcessor._normalize(
                weather_data.get("cloud_coverage", 50.0),
                min_val=0.0,
                max_val=100.0
            ),
            "wind_speed_norm": WeatherDataProcessor._normalize(
                weather_data.get("wind_speed", 5.0),
                min_val=0.0,
                max_val=30.0
            ),
            "pressure_norm": WeatherDataProcessor._normalize(
                weather_data.get("pressure", 1013.0),
                min_val=950.0,
                max_val=1050.0
            ),
        }
    
    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        """Safely convert value to float with default by Zara"""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to range 0 1 by Zara"""
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
