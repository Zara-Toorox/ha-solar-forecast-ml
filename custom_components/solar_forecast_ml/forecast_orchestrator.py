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
from typing import Any, Dict, Optional

from datetime import timedelta

from homeassistant.core import HomeAssistant

# Import helper for time functions
from .helpers import SafeDateTimeUtil as dt_util
# Import specific components used
from .ml_forecast_strategy import MLForecastStrategy
from .rule_based_forecast_strategy import RuleBasedForecastStrategy
from .weather_calculator import WeatherCalculator
# Import MLPredictor type hint if needed
from .ml_predictor import MLPredictor
# Import ErrorHandlingService type hint if needed
from .error_handling_service import ErrorHandlingService

_LOGGER = logging.getLogger(__name__)


class ForecastOrchestrator:
    """
    Selects and executes the most appropriate forecast strategy based on
    availability (e.g., ML model health) and calculates next-hour predictions.
    """

    # Fallback constants for daytime check
    FALLBACK_PRODUCTION_START_HOUR = 6  # Standard: ca. Sonnenaufgang
    FALLBACK_PRODUCTION_END_HOUR = 18   # Standard: ca. Sonnenuntergang

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
        self.weather_calculator = weather_calculator  # Used by rule-based and simple fallback

        # Strategy instances (initialized later)
        self.ml_strategy: Optional[MLForecastStrategy] = None
        self.rule_based_strategy: Optional[RuleBasedForecastStrategy] = None
        # Store reference to ML predictor if needed by other methods here
        self._ml_predictor_instance: Optional[MLPredictor] = None  # Renamed for clarity

        self.active_strategy_name: Optional[str] = None  # Track which strategy was used last

        _LOGGER.debug("ForecastOrchestrator initialized.")

    def initialize_strategies(
        self,
        ml_predictor: Optional[MLPredictor] = None,  # Accept Optional MLPredictor instance
        error_handler: Optional[ErrorHandlingService] = None  # Accept Optional ErrorHandler
    ) -> None:
        """
        Initializes the available forecast strategy instances.

        Args:
            ml_predictor: The initialized MLPredictor instance, if available.
            error_handler: The initialized ErrorHandlingService instance, if available.
        """
        _LOGGER.info("Initializing forecast strategies...")
        # Always initialize the rule-based strategy (fallback)
        try:
            self.rule_based_strategy = RuleBasedForecastStrategy(
                weather_calculator=self.weather_calculator,
                solar_capacity=self.solar_capacity
            )
            _LOGGER.info("Rule-Based forecast strategy initialized.")
        except Exception as e:
            _LOGGER.error(f"Failed to initialize RuleBasedForecastStrategy: {e}", exc_info=True)
            self.rule_based_strategy = None  # Mark as unavailable

        # Initialize ML strategy only if predictor instance is provided
        if ml_predictor:
            try:
                self._ml_predictor_instance = ml_predictor  # Store reference if needed later
                self.ml_strategy = MLForecastStrategy(
                    ml_predictor=ml_predictor,
                    error_handler=error_handler  # Pass error handler to ML strategy
                )
                _LOGGER.info("ML forecast strategy initialized.")
            except Exception as e:
                _LOGGER.error(f"Failed to initialize MLForecastStrategy: {e}", exc_info=True)
                self.ml_strategy = None  # Mark as unavailable
        else:
            _LOGGER.info("ML forecast strategy not available (ML Predictor instance missing).")

    async def create_forecast(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],  # Includes solar_capacity, external sensors etc.
        correction_factor: float = 1.0  # Learned fallback correction factor
    ) -> Dict[str, Any]:
        """
        Creates the main daily solar forecast (today and tomorrow) by selecting
        and executing the best available strategy.

        Args:
            weather_data: Current weather data dictionary.
            sensor_data: Dictionary with other relevant sensor/config data.
            correction_factor: Learned factor for adjusting rule-based predictions.

        Returns:
            A dictionary containing forecast results:
            {'today': float, 'tomorrow': float, 'peak_time': str,
             'confidence': float, 'method': str, 'model_accuracy': Optional[float]}
        """
        _LOGGER.debug("Creating daily forecast...")

        # --- Strategy Selection ---
        # 1. Try ML Strategy if available and healthy
        if self.ml_strategy and self.ml_strategy.is_available():
            try:
                _LOGGER.debug("Attempting forecast using ML strategy.")
                result = await self.ml_strategy.calculate_forecast(
                    weather_data=weather_data,
                    sensor_data=sensor_data,
                    correction_factor=correction_factor  # Passed, though ML might ignore it
                )
                self.active_strategy_name = "ml"
                # Format result into the dictionary expected by the coordinator
                return {
                    "today": result.forecast_today,
                    "tomorrow": result.forecast_tomorrow,
                    "peak_time": "12:00",  # ML strategy doesn't calculate peak time currently
                    "confidence": result.confidence_today,
                    "method": result.method,
                    "model_accuracy": result.model_accuracy  # Pass through model accuracy
                }
            except Exception as ml_err:
                _LOGGER.warning(f"ML forecast strategy failed: {ml_err}. Falling back...")
                # Continue to try rule-based strategy

        # 2. Try Rule-Based Strategy if ML failed or unavailable
        if self.rule_based_strategy and self.rule_based_strategy.is_available():
            try:
                _LOGGER.debug("Attempting forecast using Rule-Based strategy.")
                result = await self.rule_based_strategy.calculate_forecast(
                    weather_data=weather_data,
                    sensor_data=sensor_data,
                    correction_factor=correction_factor  # Rule-based uses this factor
                )
                self.active_strategy_name = "rule_based"
                return {
                    "today": result.forecast_today,
                    "tomorrow": result.forecast_tomorrow,
                    "peak_time": "12:00",  # Rule-based doesn't calculate peak time
                    "confidence": result.confidence_today,
                    "method": result.method,
                    "model_accuracy": None  # Rule-based has no ML model accuracy
                }
            except Exception as rb_err:
                _LOGGER.error(f"Rule-Based forecast strategy also failed: {rb_err}. Falling back to simple calculation.", exc_info=True)
                # Continue to simple fallback

        # 3. Simple Fallback (if both ML and Rule-Based fail)
        _LOGGER.warning("All primary forecast strategies failed. Using simple fallback calculation.")
        self.active_strategy_name = "simple_fallback"
        return self._simple_fallback_forecast(weather_data)

    def _simple_fallback_forecast(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        A very basic fallback forecast calculation used only if primary strategies fail.
        Provides a rough estimate based on capacity, cloud cover, and temperature.
        """
        try:
            # Safely get weather values with defaults
            cloud_cover = float(weather_data.get("cloud_cover", weather_data.get("clouds", 50.0)))
            temp = float(weather_data.get("temperature", 15.0))

            # Use weather calculator for factors
            cloud_factor = self.weather_calculator.get_cloud_factor(cloud_cover)
            temp_factor = self.weather_calculator.get_temperature_factor(temp)

            # Base production estimate (e.g., 3 peak sun hours equivalent)
            base_production_factor = 3.0
            today_forecast = self.solar_capacity * base_production_factor * cloud_factor * temp_factor

            # Ensure non-negative
            today_forecast = max(0.0, today_forecast)
            # Estimate tomorrow slightly lower
            tomorrow_forecast = max(0.0, today_forecast * 0.9)

            _LOGGER.debug(f"Simple fallback calculated: Today={today_forecast:.2f}, Tomorrow={tomorrow_forecast:.2f}")

            return {
                "today": round(today_forecast, 2),
                "tomorrow": round(tomorrow_forecast, 2),
                "peak_time": "12:00",  # Default peak time
                "confidence": 30.0,  # Low confidence for fallback
                "method": "simple_fallback",
                "model_accuracy": None
            }
        except Exception as e:
            _LOGGER.exception(f"Error during simple fallback forecast calculation: {e}")
            # Absolute last resort: return zeros
            return {
                "today": 0.0, "tomorrow": 0.0, "peak_time": "12:00",
                "confidence": 0.0, "method": "fallback_error", "model_accuracy": None
            }

    @staticmethod
    def _calculate_temperature_factor_simple(temp: float) -> float:
        """Simple temperature factor calculation (alternative to WeatherCalculator)."""
        # Kept original static method from coordinator for reference if needed,
        # but using WeatherCalculator instance is preferred for consistency.
        if temp < 0: return 0.85
        elif temp < 15: return 0.90 + (temp / 150.0)  # Adjusted divisor
        elif temp < 25: return 1.0
        elif temp < 35: return 1.0 - ((temp - 25.0) * 0.015)
        else: return 0.85

    def calculate_next_hour_prediction(
        self,
        forecast_today_kwh: float,  # Total estimated energy for the day
        weather_data: Optional[Dict[str, Any]] = None,
        sensor_data: Optional[Dict[str, Any]] = None  # External sensors (lux, temp etc.)
    ) -> float:
        """
        Estimates the solar production for the *next full hour*.
        Uses the ML hourly profile if available, adjusted by current conditions,
        otherwise falls back to simpler estimation.

        OPTION A: Uses LOCAL TIME for hour calculation (not UTC).

        Args:
            forecast_today_kwh: The total predicted energy production for today (in kWh).
            weather_data: Current weather conditions dictionary.
            sensor_data: Current external sensor readings dictionary.

        Returns:
            Estimated production for the next hour (in kWh), rounded to 3 decimal places.
        """
        _LOGGER.debug("Calculating next hour prediction...")
        try:
            # OPTION A: Use local time for "next hour" calculation
            now_local = dt_util.now()
            # Target hour is the *next* hour in LOCAL TIME
            target_dt_local = now_local + timedelta(hours=1)
            target_hour = target_dt_local.hour  # Hour index (0-23) in LOCAL TIME

            # --- Check Sun Position ---
            # Basic check: return 0 if it's clearly night time based on hour
            # More accurate check using sun elevation if available
            is_daytime = False
            sun_state = self.hass.states.get("sun.sun")
            if sun_state and sun_state.state not in ['unavailable', 'unknown']:
                elevation = sun_state.attributes.get("elevation")
                # Consider it daytime if elevation is positive or slightly below horizon (dawn/dusk)
                if elevation is not None and float(elevation) > -5.0:  # Threshold for usable light
                    is_daytime = True
                    _LOGGER.debug(f"Sun elevation {elevation}° > -5°. Considering daytime.")
                else:
                    _LOGGER.debug(f"Sun elevation {elevation}° <= -5°. Considering night time.")

            else:
                # Fallback to hour-based check if sun entity unavailable
                _LOGGER.debug("Sun entity unavailable, using hour-based check for daytime.")
                # Use fallback hours (local time)
                if self.FALLBACK_PRODUCTION_START_HOUR <= target_hour < self.FALLBACK_PRODUCTION_END_HOUR:
                    is_daytime = True

            if not is_daytime:
                _LOGGER.debug(f"Target hour {target_hour} (local) is considered night time. Next hour prediction is 0.0 kWh.")
                return 0.0  # Return 0 kWh if definitely night

            # --- Get Base Hourly Production ---
            # Try using the ML hourly profile for the target hour (in local time)
            ml_hourly_base_kwh: Optional[float] = self._get_ml_hourly_profile_base(forecast_today_kwh, target_hour)

            if ml_hourly_base_kwh is not None:
                base_kwh = ml_hourly_base_kwh
                _LOGGER.debug(f"Using ML profile base for hour {target_hour} (local): {base_kwh:.3f} kWh.")
            else:
                # Fallback: Simple distribution (e.g., assume 1/10th of daily total during peak hour)
                # This needs a better fallback - maybe sine curve based on target_hour?
                # Example: Distribute remaining daily forecast over remaining daylight hours?
                # Simple fallback: Assume average distribution over ~10 daylight hours
                base_kwh = forecast_today_kwh / 10.0  # Very rough estimate
                _LOGGER.warning(f"ML hourly profile base unavailable for hour {target_hour}. "
                                f"Using simple fallback base: {base_kwh:.3f} kWh.")

            # --- Apply Real-time Adjustments ---
            # Get adjustment factors based on current weather/sensor readings
            adjustment_factors = self._get_realtime_adjustment_factors(weather_data, sensor_data)

            # Apply factors to the base prediction
            adjusted_kwh = base_kwh
            factors_log = []
            for factor_name, factor_value in adjustment_factors.items():
                adjusted_kwh *= factor_value
                factors_log.append(f"{factor_name}={factor_value:.2f}")

            # Ensure prediction is not negative
            final_prediction_kwh = max(0.0, adjusted_kwh)

            _LOGGER.info(
                f"Next Hour ({target_hour:02d}:00 local) Prediction: Base={base_kwh:.3f} kWh * "
                f"Adjustments [{', '.join(factors_log)}] -> Final={final_prediction_kwh:.3f} kWh"
            )

            return round(final_prediction_kwh, 3)  # Return rounded value (more precision for hourly?)

        except Exception as e:
            _LOGGER.error(f"Next hour prediction calculation failed: {e}", exc_info=True)
            return 0.0  # Return 0.0 on any error

    def _get_ml_hourly_profile_base(self, forecast_today_kwh: float, target_hour: int) -> Optional[float]:
        """
        Calculates the base production for a specific hour using the ML hourly profile,
        scaled by the total forecast for today.

        Args:
            forecast_today_kwh: Total predicted energy for the day (kWh).
            target_hour: The hour (0-23) in LOCAL TIME to get the base value for.

        Returns:
            The estimated base production (kWh) for the target hour, or None if profile unavailable.
        """
        if not self._ml_predictor_instance or not self._ml_predictor_instance.current_profile:
            _LOGGER.debug("ML Predictor or its current_profile not available for hourly base.")
            return None

        profile = self._ml_predictor_instance.current_profile
        try:
            hourly_averages = profile.hourly_averages  # Dict keys are strings '0'-'23'
            hour_key = str(target_hour)

            # Get the average/median value for the target hour from the profile
            hour_avg_value = hourly_averages.get(hour_key)

            if hour_avg_value is None or hour_avg_value <= 0:
                _LOGGER.debug(f"No positive profile average found for hour {target_hour} (local).")
                return 0.0  # Return 0 if profile has no production for this hour

            # Calculate the sum of all positive hourly averages in the profile
            total_profile_sum = sum(float(v) for v in hourly_averages.values() if v is not None and float(v) > 0)

            if total_profile_sum <= 0:
                _LOGGER.warning("Sum of hourly profile averages is zero. Cannot calculate hourly fraction.")
                return None  # Avoid division by zero

            # Calculate the fraction of total daily production expected in this hour
            hour_fraction = hour_avg_value / total_profile_sum

            # Scale today's total forecast by this fraction
            ml_hourly_base_kwh = forecast_today_kwh * hour_fraction

            _LOGGER.debug(
                f"ML hourly base calculation: Hour={target_hour} (local), ProfileAvg={hour_avg_value:.3f}, "
                f"ProfileTotal={total_profile_sum:.3f}, Fraction={hour_fraction:.4f} -> Base={ml_hourly_base_kwh:.3f} kWh"
            )
            return ml_hourly_base_kwh

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            _LOGGER.warning(f"Error accessing or using ML hourly profile data: {e}")
            return None  # Return None if profile structure is invalid or calculation fails

    def _get_realtime_adjustment_factors(
        self,
        current_weather_data: Optional[Dict[str, Any]],
        current_sensor_data: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculates adjustment multipliers based on current weather and sensor readings
        compared to typical conditions. Factors are centered around 1.0.

        Args:
            current_weather_data: Dictionary of current weather conditions.
            current_sensor_data: Dictionary of current external sensor readings (lux, temp etc.).

        Returns:
            Dictionary of adjustment factors (e.g., {'cloud_factor': 0.8, 'temp_factor': 1.05}).
        """
        factors = {
            'cloud/lux': 1.0,  # Combined factor for cloudiness or illuminance
            'temperature': 1.0,
            'rain': 1.0,
            # Add other factors like wind if relevant
        }
        _LOGGER.debug("Calculating real-time adjustment factors...")

        # Prioritize Lux sensor if available and valid
        lux_value = current_sensor_data.get('lux') if current_sensor_data else None
        cloud_value = current_weather_data.get('cloud_cover', current_weather_data.get('clouds')) if current_weather_data else None

        # --- Cloud/Lux Factor ---
        if lux_value is not None and lux_value >= 0:
            # Adjust based on Lux deviation from a 'typical bright' value (e.g., 60000 Lux)
            # This mapping needs tuning based on expected values and impact.
            typical_bright_lux = 60000.0
            if lux_value < 1000:  # Very dark
                factors['cloud/lux'] = 0.1
            elif lux_value < 20000:  # Cloudy/Dim
                # Scale between 0.1 and 0.7 for 1k-20k Lux
                factors['cloud/lux'] = 0.1 + (lux_value / 20000.0) * 0.6
            else:  # Bright or very bright
                # Scale between 0.7 and 1.2 for 20k-100k+ Lux (allow slight boost for extremely bright)
                factors['cloud/lux'] = 0.7 + min((lux_value / typical_bright_lux) * 0.5, 0.5)  # Max factor 1.2
            _LOGGER.debug(f"Using Lux value ({lux_value} lx) for adjustment factor: {factors['cloud/lux']:.2f}")

        elif cloud_value is not None:
            # Fallback to cloud cover if Lux unavailable
            # Use the weather calculator's factor which maps percentage to a factor (0-1)
            cloud_factor = self.weather_calculator.get_cloud_factor(cloud_value)
            # Adjust slightly: maybe boost clear sky more? Or penalize overcast more?
            # Simple approach: use the factor directly. Assume profile base is for 'average' clouds.
            # Need reference point. Let's assume profile is for ~40% clouds (factor ~0.65).
            reference_cloud_factor = 0.65
            factors['cloud/lux'] = cloud_factor / reference_cloud_factor if reference_cloud_factor > 0 else 1.0
            factors['cloud/lux'] = max(0.1, min(1.2, factors['cloud/lux']))  # Clamp to reasonable range
            _LOGGER.debug(f"Using Cloud Cover ({cloud_value}%) for adjustment. BaseFactor={cloud_factor:.2f} -> AdjFactor={factors['cloud/lux']:.2f}")
        else:
            _LOGGER.debug("No Lux or Cloud data available for real-time adjustment.")

        # --- Temperature Factor ---
        temp_value = current_sensor_data.get('temperature') if current_sensor_data else None
        if temp_value is None and current_weather_data:
            temp_value = current_weather_data.get('temperature')  # Fallback to weather entity temp

        if temp_value is not None:
            temp_factor = self.weather_calculator.get_temperature_factor(temp_value)
            # Assume profile base is for optimal temp (factor 1.0). Adjustment is the factor itself.
            factors['temperature'] = max(0.7, min(1.1, temp_factor))  # Clamp adjustment
            _LOGGER.debug(f"Using Temperature ({temp_value}°C) for adjustment factor: {factors['temperature']:.2f}")
        else:
            _LOGGER.debug("No Temperature data available for real-time adjustment.")

        # --- Rain Factor ---
        rain_value = current_sensor_data.get('rain') if current_sensor_data else None
        # Could also use weather_data.get('precipitation') as fallback if rain sensor missing

        if rain_value is not None and rain_value > 0:
            # Apply reduction based on rain intensity (simple thresholds)
            if rain_value > 5.0:  # Heavy rain (e.g., > 5 mm/hr)
                factors['rain'] = 0.2
            elif rain_value > 1.0:  # Moderate rain
                factors['rain'] = 0.5
            else:  # Light rain
                factors['rain'] = 0.8
            _LOGGER.debug(f"Using Rain value ({rain_value}) for adjustment factor: {factors['rain']:.2f}")
        # Else: No rain detected or sensor unavailable, factor remains 1.0

        # --- Clamp all factors ---
        for key in factors:
            factors[key] = max(0.0, min(1.5, factors[key]))  # Conservative max 1.5

        _LOGGER.debug(f"Final real-time adjustment factors: {factors}")
        return factors
