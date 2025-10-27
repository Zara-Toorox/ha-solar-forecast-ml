"""
Rule-basierte Forecast-Strategie (Fallback).
Verwendet einfache Regeln fÃ¼r Vorhersagen wenn ML nicht verfÃ¼gbar.
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
from typing import Any, Dict

from .forecast_strategy import ForecastStrategy, ForecastResult

_LOGGER = logging.getLogger(__name__)


class RuleBasedForecastStrategy(ForecastStrategy):
    
    def __init__(self, weather_calculator, solar_capacity: float):
        super().__init__("rule_based")
        self.weather_calculator = weather_calculator
        self.solar_capacity = solar_capacity
        
        self.BASE_DAILY_FACTOR = 4
        self.TOMORROW_DISCOUNT_FACTOR = 0.92
        self.MAX_REALISTIC_FACTOR = 8
        
    def is_available(self) -> bool:
        return True
    
    def get_priority(self) -> int:
        return 50
    
    async def calculate_forecast(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        correction_factor: float
    ) -> ForecastResult:
        try:
            _LOGGER.debug("ðŸ“Š Starte Rule-based Forecast-Berechnung...")
            
            base_capacity = sensor_data.get("solar_capacity", self.solar_capacity)
            
            temp_factor = self.weather_calculator.get_temperature_factor(
                weather_data.get("temperature", 15.0)
            )
            cloud_factor = self.weather_calculator.get_cloud_factor(
                weather_data.get("clouds", 50.0)
            )
            condition_factor = self.weather_calculator.get_condition_factor(
                weather_data.get("condition", "")
            )
            
            base_production = base_capacity * self.BASE_DAILY_FACTOR
            
            today_forecast = (
                base_production 
                * temp_factor 
                * cloud_factor
                * condition_factor
                * correction_factor
            )
            
            tomorrow_forecast = today_forecast * self.TOMORROW_DISCOUNT_FACTOR
            
            max_realistic = base_capacity * self.MAX_REALISTIC_FACTOR
            today_forecast = min(today_forecast, max_realistic)
            tomorrow_forecast = min(tomorrow_forecast, max_realistic)
            
            today_forecast = max(0.0, today_forecast)
            tomorrow_forecast = max(0.0, tomorrow_forecast)
            
            confidence_today = 80 if correction_factor > 0.5 else 65
            confidence_tomorrow = 75 if correction_factor > 0.5 else 60
            
            result = ForecastResult(
                forecast_today=today_forecast,
                forecast_tomorrow=tomorrow_forecast,
                confidence_today=float(confidence_today),
                confidence_tomorrow=float(confidence_tomorrow),
                method="rule_based",
                calibrated=True,
                base_capacity=base_capacity,
                correction_factor=correction_factor
            )
            
            self._log_calculation(
                result,
                f"(temp={temp_factor:.2f}, cloud={cloud_factor:.2f}, cond={condition_factor:.2f}, corr={correction_factor:.2f})"
            )
            
            return result
            
        except Exception as e:
            _LOGGER.error(f"âŒ Rule-based Forecast Berechnung fehlgeschlagen: {e}", exc_info=True)
            
            _LOGGER.warning("âš¡ï¸ Verwende Emergency Fallback fÃ¼r Rule-based Forecast")
            
            fallback_capacity = sensor_data.get("solar_capacity", self.solar_capacity)
            if fallback_capacity <= 0:
                fallback_capacity = 2.0
            
            return ForecastResult(
                forecast_today=fallback_capacity * 2,
                forecast_tomorrow=fallback_capacity * 2,
                confidence_today=50.0,
                confidence_tomorrow=50.0,
                method="emergency_fallback",
                calibrated=False
            )
