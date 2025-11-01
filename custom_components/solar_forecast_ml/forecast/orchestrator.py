"""
Forecast Orchestrator for Solar Forecast ML.
Manages the selection and execution of different forecast strategies
(ML, Rule-based, Fallback) and calculates short-term predictions.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""
import logging
from typing import Any, Dict, Optional, List
import asyncio

from datetime import timedelta, datetime, timezone

from homeassistant.core import HomeAssistant

from ..core.helpers import SafeDateTimeUtil as dt_util
from ..ml.forecast_strategy import MLForecastStrategy
from ..forecast.rule_based_strategy import RuleBasedForecastStrategy
from ..forecast.weather_calculator import WeatherCalculator
from ..ml.predictor import MLPredictor
from ..services.error_handler import ErrorHandlingService
from ..forecast.strategy import ForecastResult


_LOGGER = logging.getLogger(__name__)


class ForecastOrchestrator:
    """
    Selects and executes the most appropriate forecast strategy based on
    availability (e.g., ML model health) and calculates next-hour predictions.
    """

    FALLBACK_PRODUCTION_START_HOUR = 6
    FALLBACK_PRODUCTION_END_HOUR = 21

    def __init__(
        self,
        hass: HomeAssistant,
        solar_capacity: float,
        weather_calculator: WeatherCalculator
    ):
        """
        Initialize the ForecastOrchestrator.

        Args:
            hass: HomeAssistant instance.
            solar_capacity: Configured solar capacity (kWp).
            weather_calculator: Instance of WeatherCalculator for rule-based factors.
        """
        self.hass = hass
        self.solar_capacity = solar_capacity
        self.weather_calculator = weather_calculator

        self.ml_strategy: Optional[MLForecastStrategy] = None
        self.rule_based_strategy: Optional[RuleBasedForecastStrategy] = None
        
        self._ml_predictor: Optional[MLPredictor] = None
        self._historical_cache: Dict = {}

        self.active_strategy_name: Optional[str] = None

        _LOGGER.debug("ForecastOrchestrator initialized.")

    def is_production_hour(self, target_dt: datetime) -> bool:
        """
        Checks if a given datetime is within realistic solar production hours.
        Uses sun.sun entity with 90-minute safety margins, falls back to seasonal times.
        
        Args:
            target_dt: Timezone-aware datetime to check
            
        Returns:
            True if target_dt is within production hours, False otherwise
        """
        # Method 1: sun.sun with safety margins (most accurate)
        sun_state = self.hass.states.get("sun.sun")
        if sun_state and sun_state.attributes:
            try:
                next_rising_str = sun_state.attributes.get("next_rising")
                next_setting_str = sun_state.attributes.get("next_setting")
                
                if next_rising_str and next_setting_str:
                    next_rising = dt_util.parse_datetime(next_rising_str)
                    next_setting = dt_util.parse_datetime(next_setting_str)
                    
                    if next_rising and next_setting:
                        # 90-minute safety margins for realistic PV production
                        # PV needs strong sunlight, not just above horizon
                        production_start = next_rising + timedelta(minutes=90)
                        production_end = next_setting - timedelta(minutes=90)
                        
                        # Check if target is within production window
                        if production_start <= target_dt <= production_end:
                            _LOGGER.debug(
                                f"Production check (sun.sun): {target_dt.strftime('%H:%M')} is within "
                                f"{production_start.strftime('%H:%M')}-{production_end.strftime('%H:%M')}"
                            )
                            return True
                        else:
                            _LOGGER.debug(
                                f"Production check (sun.sun): {target_dt.strftime('%H:%M')} is outside "
                                f"{production_start.strftime('%H:%M')}-{production_end.strftime('%H:%M')}"
                            )
                            return False
            except Exception as e:
                _LOGGER.debug(f"sun.sun parsing failed, using fallback: {e}")
        
        # Method 2: Seasonal conservative estimates (fallback)
        hour = target_dt.hour
        month = target_dt.month
        
        # Conservative production hours by season
        if month in [11, 12, 1]:  # Winter: 7-16 Uhr
            is_production = 7 <= hour <= 16
        elif month in [5, 6, 7, 8]:  # Summer: 5-20 Uhr
            is_production = 5 <= hour <= 20
        else:  # Spring/Fall: 6-18 Uhr
            is_production = 6 <= hour <= 18
        
        _LOGGER.debug(
            f"Production check (fallback): {target_dt.strftime('%H:%M')} month={month} "
            f"hour={hour} -> {is_production}"
        )
        return is_production

    def initialize_strategies(
        self,
        ml_predictor: Optional[MLPredictor] = None,
        error_handler: Optional[ErrorHandlingService] = None
    ) -> None:
        """
        Initializes the available forecast strategy instances.

        Args:
            ml_predictor: The initialized MLPredictor instance, if available.
            error_handler: The initialized ErrorHandlingService instance, if available.
        """
        _LOGGER.info("Initializing forecast strategies...")
        
        self._ml_predictor = ml_predictor
        
        if self._ml_predictor:
            self._historical_cache = self._ml_predictor._historical_cache
        
        try:
            self.rule_based_strategy = RuleBasedForecastStrategy(
                weather_calculator=self.weather_calculator,
                solar_capacity=self.solar_capacity
            )
            _LOGGER.info("Rule-Based forecast strategy (iterative) initialized.")
                
        except Exception as e:
            _LOGGER.error(f"Failed to initialize RuleBasedForecastStrategy: {e}", exc_info=True)
            self.rule_based_strategy = None

        if ml_predictor:
            try:
                self.ml_strategy = MLForecastStrategy(
                    ml_predictor=ml_predictor,
                    error_handler=error_handler
                )
                _LOGGER.info("ML forecast strategy (iterative) initialized.")
            except Exception as e:
                _LOGGER.error(f"Failed to initialize MLForecastStrategy: {e}", exc_info=True)
                self.ml_strategy = None
        else:
            _LOGGER.info("ML forecast strategy not available (ML Predictor instance missing).")

    async def orchestrate_forecast(
        self,
        current_weather: Optional[Dict[str, Any]] = None,
        hourly_forecast: Optional[List[Dict[str, Any]]] = None,
        external_sensors: Optional[Dict[str, Any]] = None,
        historical_avg: Optional[float] = None,
        ml_prediction_today: Optional[float] = None,
        ml_prediction_tomorrow: Optional[float] = None,
        correction_factor: float = 1.0
    ) -> Dict[str, Any]:
        """
        Orchestriert die Forecast-Erstellung mit allen verfÃƒÆ’Ã‚Â¼gbaren Daten.
        
        Args:
            current_weather: Aktuelle Wetterdaten
            hourly_forecast: StÃƒÆ’Ã‚Â¼ndliche Wettervorhersage
            external_sensors: Externe Sensordaten
            historical_avg: Historischer Durchschnitt (7-Tage)
            ml_prediction_today: ML Vorhersage fÃƒÆ’Ã‚Â¼r heute
            ml_prediction_tomorrow: ML Vorhersage fÃƒÆ’Ã‚Â¼r morgen
            correction_factor: Gelernter Korrekturfaktor
            
        Returns:
            Dictionary mit Prognoseergebnissen
        """
        hourly_weather_forecast = hourly_forecast if hourly_forecast else []
        sensor_data = external_sensors if external_sensors else {}
        
        if 'current_yield' not in sensor_data and current_weather:
            sensor_data['current_yield'] = 0.0
            
        return await self.create_forecast(
            hourly_weather_forecast=hourly_weather_forecast,
            sensor_data=sensor_data,
            correction_factor=correction_factor
        )

    async def create_forecast(
        self,
        hourly_weather_forecast: List[Dict[str, Any]],
        sensor_data: Dict[str, Any],
        correction_factor: float = 1.0
    ) -> Dict[str, Any]:
        """
        Erstellt die tÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¤gliche Solarprognose (heute und morgen) durch
        iterative Berechnung und Blending der verfÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼gbaren Strategien.

        Args:
            hourly_weather_forecast: Verarbeitete stÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼ndliche Wettervorhersage vom WeatherService.
            sensor_data: Dictionary mit 'current_yield'.
            correction_factor: Gelernter Faktor fÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼r die Regel-Strategie.

        Returns:
            Ein Dictionary mit den finalen, geblendeten Prognoseergebnissen.
        """
        _LOGGER.debug("Creating blended daily forecast (Iterative Pipeline)...")

        ml_result: Optional[ForecastResult] = None
        rule_result: Optional[ForecastResult] = None
        model_accuracy = 0.0

        lag_features = {}
        try:
            now_local = dt_util.as_local(dt_util.utcnow())
            yesterday_dt = now_local - timedelta(days=1)
            yesterday_key = yesterday_dt.date().isoformat()
            yesterday_total_kwh = self._historical_cache.get('daily_productions', {}).get(yesterday_key, 0.0)
            lag_features['production_yesterday'] = float(yesterday_total_kwh)
            _LOGGER.debug(f"Lag-Feature 'production_yesterday' = {yesterday_total_kwh:.2f} kWh")
        except Exception as e:
            _LOGGER.warning(f"Could not retrieve lag feature 'production_yesterday': {e}")
            lag_features['production_yesterday'] = 0.0
            

        if self.ml_strategy and self.ml_strategy.is_available():
            try:
                _LOGGER.debug("Attempting forecast using ML (Iterative) strategy...")
                ml_result = await self.ml_strategy.calculate_forecast(
                    hourly_weather_forecast=hourly_weather_forecast,
                    sensor_data=sensor_data,
                    lag_features=lag_features,
                    correction_factor=correction_factor
                )
                if ml_result.model_accuracy is not None:
                    model_accuracy = ml_result.model_accuracy
                _LOGGER.debug(f"ML Strategy Success. Today={ml_result.forecast_today:.2f} kWh. Accuracy={model_accuracy:.3f}")
                
            except Exception as ml_err:
                _LOGGER.warning(f"ML (Iterative) forecast strategy failed: {ml_err}. ML result set to 0.")
                ml_result = None
        else:
            _LOGGER.debug("ML strategy not available or unhealthy.")


        if self.rule_based_strategy and self.rule_based_strategy.is_available():
            try:
                _LOGGER.debug("Attempting forecast using Rule-Based (Iterative) strategy...")
                rule_result = await self.rule_based_strategy.calculate_forecast(
                    hourly_weather_forecast=hourly_weather_forecast,
                    sensor_data=sensor_data,
                    correction_factor=correction_factor,
                    lag_features=lag_features
                )
                _LOGGER.debug(f"Rule-Based Strategy Success. Today={rule_result.forecast_today:.2f} kWh.")
                
            except Exception as rb_err:
                _LOGGER.error(f"Rule-Based (Iterative) forecast strategy also failed: {rb_err}.", exc_info=True)
                rule_result = None
        else:
            _LOGGER.error("Rule-Based strategy not available. Cannot calculate fallback.")

        
        today_ml = ml_result.forecast_today if ml_result else 0.0
        tomorrow_ml = ml_result.forecast_tomorrow if ml_result else 0.0
        
        if rule_result:
            today_rule = rule_result.forecast_today
            tomorrow_rule = rule_result.forecast_tomorrow
        else:
            _LOGGER.critical("Emergency Fallback: ML and Rule-Based strategies failed. Returning 0.0")
            today_rule = 0.0
            tomorrow_rule = 0.0
            
            if ml_result:
                today_rule, tomorrow_rule = today_ml, tomorrow_ml
        
        if not ml_result:
            today_ml, tomorrow_ml = today_rule, tomorrow_rule
            model_accuracy = 0.0
            _LOGGER.debug("Blending: ML failed, using 100% Rule-Based result.")
        
        accuracy_weight = max(0.0, min(1.0, model_accuracy))
        
        # CRITICAL: When ML accuracy is low (<60%), strongly favor Rule-Based strategy
        # This prevents unrealistic predictions from poorly-trained ML models
        ACCURACY_THRESHOLD = 0.60
        if accuracy_weight < ACCURACY_THRESHOLD and ml_result:
            _LOGGER.info(
                f"ML accuracy ({accuracy_weight:.1%}) below threshold ({ACCURACY_THRESHOLD:.0%}). "
                f"Applying conservative ML weighting (30% ML / 70% Rule-Based)."
            )
            accuracy_weight = 0.30
        
        rule_weight = 1.0 - accuracy_weight
        
        final_today = (today_ml * accuracy_weight) + (today_rule * rule_weight)
        final_tomorrow = (tomorrow_ml * accuracy_weight) + (tomorrow_rule * rule_weight)
        
        method_str = f"blended (ML: {accuracy_weight*100:.0f}% | Rule: {rule_weight*100:.0f}%)"
        if accuracy_weight == 0.0:
            method_str = "rule_based_iterative"
        elif accuracy_weight == 1.0:
            method_str = "ml_iterative"

        conf_ml = ml_result.confidence_today if ml_result else 30.0
        conf_rule = rule_result.confidence_today if rule_result else 30.0
        final_confidence = (conf_ml * accuracy_weight) + (conf_rule * rule_weight)


        _LOGGER.info(
            f"Blending complete (Accuracy={accuracy_weight:.3f}): "
            f"ML=({today_ml:.2f}, {tomorrow_ml:.2f}), "
            f"Rule=({today_rule:.2f}, {tomorrow_rule:.2f}) -> "
            f"Final=({final_today:.2f}, {final_tomorrow:.2f}) kWh"
        )

        return {
            "today": round(final_today, 2),
            "tomorrow": round(final_tomorrow, 2),
            "peak_time": "12:00",
            "confidence": round(final_confidence, 1),
            "method": method_str,
            "model_accuracy": model_accuracy if self._ml_predictor else None
        }

    def calculate_next_hour_prediction(
        self,
        forecast_today_kwh: float,
        weather_data: Optional[Dict[str, Any]] = None,
        sensor_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Estimates the solar production for the *next full hour*.
        Uses the ML hourly profile if available, adjusted by current conditions,
        otherwise falls back to simpler estimation.

        Args:
            forecast_today_kwh: The total predicted energy production for today (in kWh).
            weather_data: Current weather conditions dictionary.
            sensor_data: Current external sensor readings dictionary.

        Returns:
            Estimated production for the next hour (in kWh), rounded to 3 decimal places.
        """
        _LOGGER.debug("Calculating next hour prediction...")
        try:
            now_local = dt_util.now()
            target_dt_local = now_local + timedelta(hours=1)
            target_hour = target_dt_local.hour

            # Use new production hour check (sun.sun + safety margins)
            if not self.is_production_hour(target_dt_local):
                _LOGGER.debug(f"Target hour {target_hour} (local) is outside production hours. Next hour prediction is 0.0 kWh.")
                return 0.0

            ml_hourly_base_kwh: Optional[float] = self._get_ml_hourly_profile_base(forecast_today_kwh, target_hour)

            if ml_hourly_base_kwh is not None:
                base_kwh = ml_hourly_base_kwh
                _LOGGER.debug(f"Using ML profile base for hour {target_hour} (local): {base_kwh:.3f} kWh.")
            else:
                base_kwh = forecast_today_kwh / 10.0
                _LOGGER.warning(f"ML hourly profile base unavailable for hour {target_hour}. "
                                f"Using simple fallback base: {base_kwh:.3f} kWh.")

            adjustment_factors = self._get_realtime_adjustment_factors(weather_data, sensor_data)

            adjusted_kwh = base_kwh
            factors_log = []
            for factor_name, factor_value in adjustment_factors.items():
                adjusted_kwh *= factor_value
                factors_log.append(f"{factor_name}={factor_value:.2f}")

            final_prediction_kwh = max(0.0, adjusted_kwh)

            _LOGGER.info(
                f"Next Hour ({target_hour:02d}:00 local) Prediction: Base={base_kwh:.3f} kWh * "
                f"Adjustments [{', '.join(factors_log)}] -> Final={final_prediction_kwh:.3f} kWh"
            )

            return round(final_prediction_kwh, 3)

        except Exception as e:
            _LOGGER.error(f"Next hour prediction calculation failed: {e}", exc_info=True)
            return 0.0

    def _get_ml_hourly_profile_base(self, forecast_today_kwh: float, target_hour: int) -> Optional[float]:
        """
        Calculates the base production for a specific hour using the ML hourly profile,
        scaled by the total forecast for today.
        """
        if not self._ml_predictor or not self._ml_predictor.current_profile:
            _LOGGER.debug("ML Predictor or its current_profile not available for hourly base.")
            return None

        profile = self._ml_predictor.current_profile
        try:
            hourly_averages = profile.hourly_averages
            hour_key = str(target_hour)
            hour_avg_value = hourly_averages.get(hour_key)

            if hour_avg_value is None or hour_avg_value <= 0:
                _LOGGER.debug(f"No positive profile average found for hour {target_hour} (local).")
                return 0.0

            total_profile_sum = sum(float(v) for v in hourly_averages.values() if v is not None and float(v) > 0)

            if total_profile_sum <= 0:
                _LOGGER.warning("Sum of hourly profile averages is zero. Cannot calculate hourly fraction.")
                return None

            hour_fraction = hour_avg_value / total_profile_sum
            ml_hourly_base_kwh = forecast_today_kwh * hour_fraction

            _LOGGER.debug(
                f"ML hourly base calculation: Hour={target_hour} (local), ProfileAvg={hour_avg_value:.3f}, "
                f"ProfileTotal={total_profile_sum:.3f}, Fraction={hour_fraction:.4f} -> Base={ml_hourly_base_kwh:.3f} kWh"
            )
            return ml_hourly_base_kwh

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            _LOGGER.warning(f"Error accessing or using ML hourly profile data: {e}")
            return None

    def _get_realtime_adjustment_factors(
        self,
        current_weather_data: Optional[Dict[str, Any]],
        current_sensor_data: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculates adjustment multipliers based on current weather and sensor readings.
        """
        factors = {
            'cloud/lux': 1.0,
            'temperature': 1.0,
            'rain': 1.0,
        }
        _LOGGER.debug("Calculating real-time adjustment factors...")

        lux_value = current_sensor_data.get('lux') if current_sensor_data else None
        cloud_value = current_weather_data.get('cloud_cover', current_weather_data.get('clouds')) if current_weather_data else None

        if lux_value is not None and lux_value >= 0:
            typical_bright_lux = 60000.0
            if lux_value < 1000: factors['cloud/lux'] = 0.1
            elif lux_value < 20000: factors['cloud/lux'] = 0.1 + (lux_value / 20000.0) * 0.6
            else: factors['cloud/lux'] = 0.7 + min((lux_value / typical_bright_lux) * 0.5, 0.5)
            _LOGGER.debug(f"Using Lux value ({lux_value} lx) for adjustment factor: {factors['cloud/lux']:.2f}")

        elif cloud_value is not None:
            cloud_factor = self.weather_calculator.get_cloud_factor(cloud_value)
            reference_cloud_factor = 0.65
            factors['cloud/lux'] = cloud_factor / reference_cloud_factor if reference_cloud_factor > 0 else 1.0
            factors['cloud/lux'] = max(0.1, min(1.2, factors['cloud/lux']))
            _LOGGER.debug(f"Using Cloud Cover ({cloud_value}%) for adjustment. BaseFactor={cloud_factor:.2f} -> AdjFactor={factors['cloud/lux']:.2f}")
        else:
            _LOGGER.debug("No Lux or Cloud data available for real-time adjustment.")

        temp_value = current_sensor_data.get('temperature') if current_sensor_data else None
        if temp_value is None and current_weather_data:
            temp_value = current_weather_data.get('temperature')

        if temp_value is not None:
            temp_factor = self.weather_calculator.get_temperature_factor(temp_value)
            factors['temperature'] = max(0.7, min(1.1, temp_factor))
            _LOGGER.debug(f"Using Temperature ({temp_value}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°C) for adjustment factor: {factors['temperature']:.2f}")
        else:
            _LOGGER.debug("No Temperature data available for real-time adjustment.")

        rain_value = current_sensor_data.get('rain') if current_sensor_data else None

        if rain_value is not None and rain_value > 0:
            if rain_value > 5.0: factors['rain'] = 0.2
            elif rain_value > 1.0: factors['rain'] = 0.5
            else: factors['rain'] = 0.8
            _LOGGER.debug(f"Using Rain value ({rain_value}) for adjustment factor: {factors['rain']:.2f}")

        for key in factors:
            factors[key] = max(0.0, min(1.5, factors[key]))

        _LOGGER.debug(f"Final real-time adjustment factors: {factors}")
        return factors
