"""
Weather Calculator fÃƒÂ¼r Solar Forecast ML.
Berechnet Temperatur-, Cloud- und Seasonal-Faktoren.
Version 4.8.1

Copyright (C) 2025 Zara-Toorox

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
from typing import Dict, Any

from .helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class WeatherCalculator:
    
    def __init__(self):
        self.SEASONAL_FACTORS = {
            "winter": 0.3,
            "spring": 0.7,
            "summer": 1.0,
            "autumn": 0.6
        }
        
        self.SEASONAL_MONTH_MAPPING = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn"
        }
        
        self.OPTIMAL_TEMPERATURE = 25.0
        self.TEMP_EFFICIENCY_LOSS = 0.005
        
        self.CONDITION_FACTORS = {
            "rainy": 0.25,
            "pouring": 0.15,
            "snowy": 0.20,
            "snowy-rainy": 0.18,
            "lightning": 0.22,
            "lightning-rainy": 0.20,
            "hail": 0.18,
            "fog": 0.30,
            "windy": 0.85,
            "exceptional": 0.40
        }
        
        _LOGGER.debug("Ã¢Å“â€œ WeatherCalculator initialisiert")
    
    def get_temperature_factor(self, temperature: float) -> float:
        try:
            if temperature < 0:
                return 0.7
            elif temperature <= self.OPTIMAL_TEMPERATURE:
                return 0.8 + (temperature / self.OPTIMAL_TEMPERATURE) * 0.2
            else:
                factor = 1.0 - (temperature - self.OPTIMAL_TEMPERATURE) * self.TEMP_EFFICIENCY_LOSS
                return max(0.7, factor)
        except Exception as e:
            _LOGGER.warning(f"Ã¢Å¡Â Ã¯Â¸Â Temperatur-Faktor Berechnung fehlgeschlagen: {e}")
            return 0.9
    
    def get_cloud_factor(self, cloud_coverage: float) -> float:
        try:
            if cloud_coverage < 20:
                return 1.0
            elif cloud_coverage < 50:
                return 0.8
            elif cloud_coverage < 80:
                return 0.4
            else:
                return 0.15
        except Exception as e:
            _LOGGER.warning(f"Ã¢Å¡Â Ã¯Â¸Â Cloud-Faktor Berechnung fehlgeschlagen: {e}")
            return 0.6
    
    def get_condition_factor(self, condition: str) -> float:
        try:
            if not condition:
                return 1.0
            
            condition_lower = condition.lower()
            
            return self.CONDITION_FACTORS.get(condition_lower, 1.0)
            
        except Exception as e:
            _LOGGER.warning(f"Ã¢Å¡Â Ã¯Â¸Â Condition-Faktor Berechnung fehlgeschlagen: {e}")
            return 1.0
    
    def get_seasonal_adjustment(self, now: datetime = None) -> float:
        try:
            if now is None:
                now = dt_util.utcnow()
            
            month = now.month
            season = self.SEASONAL_MONTH_MAPPING.get(month, "autumn")
            factor = self.SEASONAL_FACTORS.get(season, 0.6)
            
            if month in [12, 1]:
                factor *= 0.8
            elif month in [6, 7]:
                factor *= 1.1
            
            return max(0.2, min(1.2, factor))
            
        except Exception as e:
            _LOGGER.warning(f"Ã¢Å¡Â Ã¯Â¸Â Seasonal adjustment Berechnung fehlgeschlagen: {e}")
            return 0.6
    
    def get_current_season(self) -> str:
        try:
            now = dt_util.utcnow()
            month = now.month
            return self.SEASONAL_MONTH_MAPPING.get(month, "autumn")
        except Exception:
            return "autumn"
    
    def calculate_combined_weather_factor(
        self,
        weather_data: Dict[str, Any],
        include_seasonal: bool = True
    ) -> float:
        try:
            temp_factor = self.get_temperature_factor(
                weather_data.get("temperature", 15.0)
            )
            cloud_factor = self.get_cloud_factor(
                weather_data.get("clouds", 50.0)
            )
            condition_factor = self.get_condition_factor(
                weather_data.get("condition", "")
            )
            
            combined = temp_factor * cloud_factor * condition_factor
            
            if include_seasonal:
                seasonal = self.get_seasonal_adjustment()
                combined *= seasonal
            
            return combined
            
        except Exception as e:
            _LOGGER.warning(f"Ã¢Å¡Â Ã¯Â¸Â Combined weather factor Berechnung fehlgeschlagen: {e}")
            return 0.5
