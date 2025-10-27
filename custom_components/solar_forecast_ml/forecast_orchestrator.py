"""
Forecast Orchestrator Module
Verwaltet Strategy-Auswahl und Forecast-Erstellung

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""
import logging
from typing import Any, Dict, Optional

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .ml_forecast_strategy import MLForecastStrategy
from .rule_based_forecast_strategy import RuleBasedForecastStrategy
from .weather_calculator import WeatherCalculator

_LOGGER = logging.getLogger(__name__)


class ForecastOrchestrator:
    
    def __init__(
        self,
        hass: HomeAssistant,
        solar_capacity: float,
        weather_calculator: WeatherCalculator
    ):
        self.hass = hass
        self.solar_capacity = solar_capacity
        self.weather_calculator = weather_calculator
        
        self.ml_strategy: Optional[MLForecastStrategy] = None
        self.rule_based_strategy: Optional[RuleBasedForecastStrategy] = None
        self.active_strategy: Optional[str] = None
        self.ml_predictor = None
    
    def initialize_strategies(self, ml_predictor=None, error_handler=None) -> None:
        self.rule_based_strategy = RuleBasedForecastStrategy(
            solar_capacity=self.solar_capacity,
            weather_calculator=self.weather_calculator
        )
        _LOGGER.info("Rule-Based Strategy initialisiert")
        
        if ml_predictor:
            self.ml_predictor = ml_predictor
            self.ml_strategy = MLForecastStrategy(
                ml_predictor=ml_predictor,
                error_handler=error_handler
            )
            _LOGGER.info("ML Strategy initialisiert")
        else:
            _LOGGER.info("ML Strategy nicht verfügbar (Predictor fehlt)")
    
    async def create_forecast(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        correction_factor: float = 1.0
    ) -> Dict[str, Any]:
        if self.ml_strategy and self.ml_strategy.is_available():
            try:
                _LOGGER.debug("⚡️ Verwende ML Strategy für Forecast")
                result = await self.ml_strategy.calculate_forecast(
                    weather_data=weather_data,
                    sensor_data=sensor_data,
                    correction_factor=correction_factor
                )
                
                self.active_strategy = "ml"
                
                return {
                    "today": result.forecast_today,
                    "tomorrow": result.forecast_tomorrow,
                    "peak_time": "12:00",
                    "confidence": result.confidence_today,
                    "method": result.method,
                    "model_accuracy": result.model_accuracy
                }
                
            except Exception as e:
                _LOGGER.warning(f"⚠️ ML Strategy fehlgeschlagen: {e}, Fallback zu Rule-Based")
        
        if self.rule_based_strategy:
            _LOGGER.debug("🌤️ Verwende Rule-Based Strategy für Forecast")
            result = await self.rule_based_strategy.calculate_forecast(
                weather_data=weather_data,
                sensor_data=sensor_data,
                correction_factor=correction_factor
            )
            
            self.active_strategy = "rule_based"
            
            return {
                "today": result.forecast_today,
                "tomorrow": result.forecast_tomorrow,
                "peak_time": "12:00",
                "confidence": result.confidence_today,
                "method": result.method,
                "model_accuracy": None
            }
        
        _LOGGER.warning("⚠️ Keine Strategy verfügbar, verwende einfache Berechnung")
        return await self._simple_forecast(weather_data)
    
    async def _simple_forecast(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        cloud_factor = 1.0 - (weather_data["cloud_cover"] / 100.0)
        temp_factor = self._calculate_temperature_factor(weather_data["temperature"])
        
        base_production = self.solar_capacity * 4.0
        today_forecast = base_production * cloud_factor * temp_factor
        
        return {
            "today": round(today_forecast, 2),
            "tomorrow": round(today_forecast, 2),
            "peak_time": "12:00",
            "confidence": 60.0,
            "method": "simple_fallback",
            "model_accuracy": None
        }
    
    @staticmethod
    def _calculate_temperature_factor(temp: float) -> float:
        if temp < 0:
            return 0.85
        elif temp < 15:
            return 0.90 + (temp / 150)
        elif temp < 25:
            return 1.0
        elif temp < 35:
            return 1.0 - ((temp - 25) * 0.015)
        else:
            return 0.85
    
    def calculate_next_hour_prediction(
        self, 
        forecast_today: float,
        weather_data: Optional[Dict[str, Any]] = None,
        sensor_data: Optional[Dict[str, Any]] = None
    ) -> float:
        try:
            now = dt_util.utcnow()
            current_hour = now.hour
            
            sun_state = self.hass.states.get("sun.sun")
            
            if sun_state and sun_state.state not in ['unavailable', 'unknown']:
                elevation = sun_state.attributes.get("elevation", 0)
                
                if elevation <= 0:
                    return 0.0
            else:
                if 21 <= current_hour or current_hour <= 5:
                    return 0.0
            
            ml_hourly_base = self._get_ml_hourly_base(forecast_today, current_hour)
            
            if ml_hourly_base is None:
                _LOGGER.debug("ML hourly profile nicht verfügbar, Fallback zu einfacher Berechnung")
                ml_hourly_base = forecast_today / 15.0
            
            sun_factor = 1.0
            if sun_state and sun_state.state not in ['unavailable', 'unknown']:
                elevation = sun_state.attributes.get("elevation", 0)
                if elevation > 0:
                    sun_factor = min(elevation / 60.0, 1.0)
            
            realtime_factors = self._get_realtime_factors(weather_data, sensor_data)
            
            cloud_factor = realtime_factors.get('cloud_factor', 1.0)
            temp_factor = realtime_factors.get('temp_factor', 1.0)
            lux_factor = realtime_factors.get('lux_factor', 1.0)
            rain_factor = realtime_factors.get('rain_factor', 1.0)
            
            hourly_prediction = ml_hourly_base * sun_factor * cloud_factor * temp_factor * lux_factor * rain_factor
            
            _LOGGER.debug(
                f"Stündliche Prognose: base={ml_hourly_base:.2f}, sun={sun_factor:.2f}, "
                f"cloud={cloud_factor:.2f}, temp={temp_factor:.2f}, lux={lux_factor:.2f}, "
                f"rain={rain_factor:.2f}, result={hourly_prediction:.2f}"
            )
            
            return round(hourly_prediction, 2)
                    
        except Exception as e:
            _LOGGER.debug(f"⚠️ Next Hour Berechnung fehlgeschlagen: {e}")
            return 0.0
    
    def _get_ml_hourly_base(self, forecast_today: float, current_hour: int) -> Optional[float]:
        try:
            if not self.ml_predictor:
                return None
            
            if not hasattr(self.ml_predictor, 'current_profile'):
                return None
            
            profile = self.ml_predictor.current_profile
            if not profile or not hasattr(profile, 'hourly_averages'):
                return None
            
            hourly_averages = profile.hourly_averages
            if not hourly_averages or not isinstance(hourly_averages, dict):
                return None
            
            hour_key = str(current_hour)
            if hour_key not in hourly_averages:
                return None
            
            hour_value = hourly_averages.get(hour_key, 0.0)
            if hour_value <= 0:
                return None
            
            total_day = sum(float(v) for v in hourly_averages.values() if v is not None and v > 0)
            if total_day <= 0:
                return None
            
            hour_fraction = hour_value / total_day
            ml_hourly_base = forecast_today * hour_fraction
            
            _LOGGER.debug(
                f"ML hourly base: hour={current_hour}, value={hour_value:.2f}, "
                f"total={total_day:.2f}, fraction={hour_fraction:.3f}, base={ml_hourly_base:.2f}"
            )
            
            return ml_hourly_base
            
        except Exception as e:
            _LOGGER.debug(f"Fehler beim Abrufen des ML hourly base: {e}")
            return None
    
    def _get_realtime_factors(
        self, 
        weather_data: Optional[Dict[str, Any]], 
        sensor_data: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        factors = {
            'cloud_factor': 1.0,
            'temp_factor': 1.0,
            'lux_factor': 1.0,
            'rain_factor': 1.0
        }
        
        try:
            cloudiness = None
            temperature = None
            lux = None
            rain = None
            
            if sensor_data:
                lux = sensor_data.get('lux_sensor')
                temperature = sensor_data.get('temp_sensor')
                rain = sensor_data.get('rain_sensor')
            
            if lux is not None:
                if lux < 1000:
                    factors['lux_factor'] = 0.1
                elif lux < 10000:
                    factors['lux_factor'] = 0.3 + (lux / 10000.0) * 0.4
                elif lux < 50000:
                    factors['lux_factor'] = 0.7 + (lux / 50000.0) * 0.3
                else:
                    factors['lux_factor'] = 1.0
            elif weather_data:
                cloudiness = weather_data.get('cloud_cover')
                if cloudiness is not None:
                    factors['cloud_factor'] = (100.0 - cloudiness) / 100.0
                    factors['cloud_factor'] = max(0.1, min(1.0, factors['cloud_factor']))
            
            if temperature is not None:
                factors['temp_factor'] = self._calculate_temperature_factor(temperature)
            elif weather_data and weather_data.get('temperature') is not None:
                factors['temp_factor'] = self._calculate_temperature_factor(weather_data['temperature'])
            
            if rain is not None:
                if rain > 10:
                    factors['rain_factor'] = 0.1
                elif rain > 5:
                    factors['rain_factor'] = 0.2
                elif rain > 2:
                    factors['rain_factor'] = 0.4
                elif rain > 0.5:
                    factors['rain_factor'] = 0.7
                else:
                    factors['rain_factor'] = 1.0
            elif weather_data and weather_data.get('precipitation') is not None:
                precip = weather_data.get('precipitation', 0)
                if precip > 10:
                    factors['rain_factor'] = 0.2
                elif precip > 5:
                    factors['rain_factor'] = 0.3
                elif precip > 2:
                    factors['rain_factor'] = 0.5
                elif precip > 0.5:
                    factors['rain_factor'] = 0.8
            
            _LOGGER.debug(
                f"Realtime-Faktoren berechnet: cloudiness={cloudiness}, "
                f"temp={temperature}, lux={lux}, rain={rain}"
            )
            
        except Exception as e:
            _LOGGER.debug(f"Fehler bei Realtime-Faktoren: {e}")
        
        return factors
