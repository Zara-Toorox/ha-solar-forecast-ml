"""
Feature Engineering fÃ¼r ML Predictor.

Copyright (C) 2025 Zara-Toorox

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""
import math
from datetime import datetime
from typing import Dict, Any, List
from homeassistant.util import dt as dt_util


class FeatureEngineer:
    
    def __init__(self):
        self.base_features = [
            "temperature", "humidity", "cloudiness", "wind_speed", 
            "hour_of_day", "seasonal_factor", "weather_trend"
        ]
        
        self.polynomial_features = [
            "temperature_sq", "cloudiness_sq", "hour_of_day_sq",
            "seasonal_factor_sq"
        ]
        
        self.interaction_features = [
            "cloudiness_x_hour", "temperature_x_seasonal", 
            "humidity_x_cloudiness", "wind_x_hour",
            "weather_trend_x_seasonal"
        ]
        
        self.feature_names = (
            self.base_features + 
            self.polynomial_features + 
            self.interaction_features
        )
    
    async def extract_features(
        self, 
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        hour: int
    ) -> Dict[str, float]:
        try:
            temp = self._safe_extract(weather_data, 'temperature', 15.0)
            humidity = self._safe_extract(weather_data, 'humidity', 60.0)
            cloudiness = self._safe_extract(weather_data, 'cloudiness', 50.0)
            wind_speed = self._safe_extract(weather_data, 'wind_speed', 5.0)
            
            now = dt_util.utcnow()
            day_of_year = now.timetuple().tm_yday
            seasonal_factor = 0.5 + 0.5 * math.cos((day_of_year - 172) * 2 * math.pi / 365)
            
            weather_trend = self._calculate_weather_trend(cloudiness, wind_speed)
            
            features = {
                "temperature": temp,
                "humidity": humidity,
                "cloudiness": cloudiness,
                "wind_speed": wind_speed,
                "hour_of_day": float(hour),
                "seasonal_factor": seasonal_factor,
                "weather_trend": weather_trend,
                "temperature_sq": temp ** 2,
                "cloudiness_sq": cloudiness ** 2,
                "hour_of_day_sq": hour ** 2,
                "seasonal_factor_sq": seasonal_factor ** 2,
                "cloudiness_x_hour": cloudiness * hour,
                "temperature_x_seasonal": temp * seasonal_factor,
                "humidity_x_cloudiness": humidity * cloudiness,
                "wind_x_hour": wind_speed * hour,
                "weather_trend_x_seasonal": weather_trend * seasonal_factor
            }
            
            return features
            
        except Exception:
            return self.get_default_features(hour)
    
    def extract_features_sync(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        record: Dict[str, Any]
    ) -> Dict[str, float]:
        try:
            timestamp_str = record.get('timestamp', dt_util.utcnow().isoformat())
            timestamp = dt_util.parse_datetime(timestamp_str)
            hour = timestamp.hour
        except:
            hour = 12
        
        temp = self._safe_extract(weather_data, 'temperature', 15.0)
        humidity = self._safe_extract(weather_data, 'humidity', 60.0)
        cloudiness = self._safe_extract(weather_data, 'cloudiness', 50.0)
        wind_speed = self._safe_extract(weather_data, 'wind_speed', 5.0)
        
        try:
            day_of_year = timestamp.timetuple().tm_yday
            seasonal_factor = 0.5 + 0.5 * math.cos((day_of_year - 172) * 2 * math.pi / 365)
        except:
            seasonal_factor = 0.75
        
        weather_trend = self._calculate_weather_trend(cloudiness, wind_speed)
        
        features = {
            "temperature": temp,
            "humidity": humidity,
            "cloudiness": cloudiness,
            "wind_speed": wind_speed,
            "hour_of_day": float(hour),
            "seasonal_factor": seasonal_factor,
            "weather_trend": weather_trend,
            "temperature_sq": temp ** 2,
            "cloudiness_sq": cloudiness ** 2,
            "hour_of_day_sq": hour ** 2,
            "seasonal_factor_sq": seasonal_factor ** 2,
            "cloudiness_x_hour": cloudiness * hour,
            "temperature_x_seasonal": temp * seasonal_factor,
            "humidity_x_cloudiness": humidity * cloudiness,
            "wind_x_hour": wind_speed * hour,
            "weather_trend_x_seasonal": weather_trend * seasonal_factor
        }
        
        return features
    
    def get_default_features(self, hour: int) -> Dict[str, float]:
        seasonal_factor = 0.75
        return {
            "temperature": 15.0,
            "humidity": 60.0,
            "cloudiness": 50.0,
            "wind_speed": 5.0,
            "hour_of_day": float(hour),
            "seasonal_factor": seasonal_factor,
            "weather_trend": 0.5,
            "temperature_sq": 225.0,
            "cloudiness_sq": 2500.0,
            "hour_of_day_sq": hour ** 2,
            "seasonal_factor_sq": seasonal_factor ** 2,
            "cloudiness_x_hour": 50.0 * hour,
            "temperature_x_seasonal": 15.0 * seasonal_factor,
            "humidity_x_cloudiness": 3000.0,
            "wind_x_hour": 5.0 * hour,
            "weather_trend_x_seasonal": 0.5 * seasonal_factor
        }
    
    def _safe_extract(
        self, 
        data: Dict[str, Any], 
        key: str, 
        default: float
    ) -> float:
        try:
            value = data.get(key, default)
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _calculate_weather_trend(self, cloudiness: float, wind_speed: float) -> float:
        cloud_score = (100 - cloudiness) / 100.0
        wind_factor = 1.0 - min(wind_speed / 30.0, 1.0)
        return (cloud_score * 0.7 + wind_factor * 0.3)
