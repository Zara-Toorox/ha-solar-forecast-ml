"""
Rule-basierte Forecast-Strategie (Fallback).
Verwendet einfache Regeln für Vorhersagen wenn ML nicht verfügbar.
Version 4.8.0

Copyright (C) 2025 Zara-Toorox
# von Zara
"""
import logging
from typing import Any, Dict

from .forecast_strategy import ForecastStrategy, ForecastResult

_LOGGER = logging.getLogger(__name__)


class RuleBasedForecastStrategy(ForecastStrategy):
    """
    Rule-basierte Forecast-Strategie (Fallback).
    Verwendet bewährte Regeln und Faktoren für Vorhersagen.
    # von Zara
    """
    
    def __init__(self, weather_calculator, solar_capacity: float):
        """
        Initialisiere Rule-based Forecast Strategy.
        
        Args:
            weather_calculator: WeatherCalculator für Faktoren
            solar_capacity: Solar-Anlagen Kapazität
        # von Zara
        """
        super().__init__("rule_based")
        self.weather_calculator = weather_calculator
        self.solar_capacity = solar_capacity
        
        # Konstanten für Rule-based Forecast # von Zara
        self.BASE_DAILY_FACTOR = 4  # Base production factor # von Zara
        self.TOMORROW_DISCOUNT_FACTOR = 0.92  # Tomorrow etwas niedriger # von Zara
        self.MAX_REALISTIC_FACTOR = 8  # Maximum realistischer Produktionsfaktor # von Zara
        
    def is_available(self) -> bool:
        """
        Rule-based ist immer verfügbar (Fallback).
        # von Zara
        """
        return True
    
    def get_priority(self) -> int:
        """
        Rule-based hat niedrigere Priorität (Fallback).
        # von Zara
        """
        return 50  # Mittlere Priorität (unter ML) # von Zara
    
    async def calculate_forecast(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        correction_factor: float
    ) -> ForecastResult:
        """
        Berechnet rule-basierten Forecast.
        
        Args:
            weather_data: Wetter-Daten (temperature, clouds)
            sensor_data: Sensor-Daten (solar_capacity)
            correction_factor: Gelernter Korrekturfaktor
            
        Returns:
            ForecastResult mit regel-basierter Vorhersage
        # von Zara
        """
        try:
            _LOGGER.debug("📊 Starte Rule-based Forecast-Berechnung...")
            
            # Hole Base Capacity aus sensor_data oder verwende Instanz-Wert # von Zara
            base_capacity = sensor_data.get("solar_capacity", self.solar_capacity)
            
            # Hole Weather-Faktoren # von Zara
            temp_factor = self.weather_calculator.get_temperature_factor(
                weather_data.get("temperature", 15.0)
            )
            cloud_factor = self.weather_calculator.get_cloud_factor(
                weather_data.get("clouds", 50.0)
            )
            
            # Berechne Base Production # von Zara
            base_production = base_capacity * self.BASE_DAILY_FACTOR
            
            # Apply alle Faktoren # von Zara
            today_forecast = (
                base_production 
                * temp_factor 
                * cloud_factor 
                * correction_factor
            )
            
            # Berechne Tomorrow mit Discount # von Zara
            tomorrow_forecast = today_forecast * self.TOMORROW_DISCOUNT_FACTOR
            
            # Apply Maximum Realistic Bound # von Zara
            max_realistic = base_capacity * self.MAX_REALISTIC_FACTOR
            today_forecast = min(today_forecast, max_realistic)
            tomorrow_forecast = min(tomorrow_forecast, max_realistic)
            
            # Apply absolute Bounds (nie negativ) # von Zara
            today_forecast = max(0.0, today_forecast)
            tomorrow_forecast = max(0.0, tomorrow_forecast)
            
            # Berechne Confidence basierend auf correction_factor # von Zara
            # Höherer correction_factor = mehr Vertrauen in die Daten # von Zara
            confidence_today = 80 if correction_factor > 0.5 else 65
            confidence_tomorrow = 75 if correction_factor > 0.5 else 60
            
            # Erstelle Result # von Zara
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
            
            # Log erfolgreiche Berechnung # von Zara
            self._log_calculation(
                result,
                f"(temp={temp_factor:.2f}, cloud={cloud_factor:.2f}, corr={correction_factor:.2f})"
            )
            
            return result
            
        except Exception as e:
            _LOGGER.error(f"❌ Rule-based Forecast Berechnung fehlgeschlagen: {e}", exc_info=True)
            
            # Emergency Fallback mit minimalen Werten # von Zara
            _LOGGER.warning("⚠️ Verwende Emergency Fallback für Rule-based Forecast")
            
            fallback_capacity = sensor_data.get("solar_capacity", self.solar_capacity)
            if fallback_capacity <= 0:
                fallback_capacity = 2.0  # Ultra-Safe Fallback # von Zara
            
            return ForecastResult(
                forecast_today=fallback_capacity * 2,
                forecast_tomorrow=fallback_capacity * 2,
                confidence_today=50.0,
                confidence_tomorrow=50.0,
                method="emergency_fallback",
                calibrated=False
            )
