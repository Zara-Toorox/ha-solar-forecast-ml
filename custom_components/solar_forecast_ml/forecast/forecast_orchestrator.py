"""
Forecast Orchestrator for Solar Forecast ML Integration

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
from typing import Any, Dict, Optional, List
import asyncio

from datetime import timedelta, datetime, timezone

from homeassistant.core import HomeAssistant

from ..core.core_helpers import SafeDateTimeUtil as dt_util
from .forecast_strategy import MLForecastStrategy
from .forecast_rule_based_strategy import RuleBasedForecastStrategy
from .forecast_weather_calculator import WeatherCalculator
from ..ml.ml_predictor import MLPredictor
from ..services.service_error_handler import ErrorHandlingService
from .forecast_strategy_base import ForecastResult
from ..const import ML_MODEL_VERSION
from ..astronomy.astronomy_cache_manager import get_cache_manager


_LOGGER = logging.getLogger(__name__)


class ForecastOrchestrator:
    """Selects and executes the most appropriate forecast strategy based on"""

    FALLBACK_PRODUCTION_START_HOUR = 6
    FALLBACK_PRODUCTION_END_HOUR = 21

    def __init__(
        self,
        hass: HomeAssistant,
        data_manager: Any, # Replace with DataManager once type hints are cleaner
        solar_capacity: float,
        weather_calculator: WeatherCalculator
    ):
        """Initialize the ForecastOrchestrator"""
        self.hass = hass
        self.data_manager = data_manager
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
        Checks if a given datetime is within realistic solar production hours

        Uses in-memory astronomy cache with fallback to seasonal estimates
        """
        # Ensure target_dt is in local timezone for comparison
        target_dt_local = dt_util.as_local(target_dt)

        # Method 1: Try in-memory astronomy cache first (no I/O blocking!)
        try:
            cache_manager = get_cache_manager()
            if cache_manager.is_loaded():
                # Get production window for target date
                date_str = target_dt_local.date().isoformat()
                window = cache_manager.get_production_window(date_str)

                if window:
                    window_start_str, window_end_str = window

                    # Parse ISO timestamps and convert to naive local time
                    window_start = datetime.fromisoformat(window_start_str).replace(tzinfo=None)
                    window_end = datetime.fromisoformat(window_end_str).replace(tzinfo=None)

                    # Convert target_dt_local to naive for comparison
                    target_dt_naive = target_dt_local.replace(tzinfo=None)

                    # Check if target_dt is within production window
                    if window_start <= target_dt_naive <= window_end:
                        # Removed verbose production check logging
                        return True
                    else:
                        # Removed verbose production check logging
                        return False

        except Exception as e:
            _LOGGER.debug(f"Astronomy cache access failed, using fallback: {e}")

        # Method 2: Seasonal conservative estimates (fallback)
        hour = target_dt_local.hour
        month = target_dt_local.month

        # Extended production times with twilight
        if month in [11, 12, 1, 2]:  # Winter: 6-17 (incl. twilight)
            is_production = 6 <= hour <= 17
        elif month in [5, 6, 7, 8]:  # Summer: 4-21 (long days)
            is_production = 4 <= hour <= 21
        else:  # Spring/Fall: 5-19
            is_production = 5 <= hour <= 19

        _LOGGER.debug(
            f"Production check (fallback): {target_dt_local.strftime('%Y-%m-%d %H:%M')} month={month} "
            f"hour={hour} -> {is_production}"
        )
        return is_production

    def initialize_strategies(
        self,
        ml_predictor: Optional[MLPredictor] = None,
        error_handler: Optional[ErrorHandlingService] = None
    ) -> None:
        """Initializes the available forecast strategy instances"""
        _LOGGER.info("Initializing forecast strategies...")
        
        self._ml_predictor = ml_predictor
        
        if self._ml_predictor:
            self._historical_cache = self._ml_predictor._historical_cache
        
        try:
            self.rule_based_strategy = RuleBasedForecastStrategy(
                weather_calculator=self.weather_calculator,
                solar_capacity=self.solar_capacity,
                orchestrator=self
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
        """Orchestrates forecast creation with all available data"""
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
        """Creates daily solar forecast today tomorrow day after tomorrow through"""
        _LOGGER.debug("Creating blended daily forecast (Iterative Pipeline)...")

        ml_result: Optional[ForecastResult] = None
        rule_result: Optional[ForecastResult] = None
        model_accuracy = 0.0

        lag_features = {}
        try:
            now_local = dt_util.now()
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
                _LOGGER.debug(
                    f"ML Strategy Success. Today={ml_result.forecast_today:.2f} kWh, "
                    f"Tomorrow={ml_result.forecast_tomorrow:.2f} kWh, "
                    f"Day After={ml_result.forecast_day_after_tomorrow:.2f} kWh. "
                    f"Accuracy={model_accuracy:.3f}"
                )
                
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
                _LOGGER.debug(
                    f"Rule-Based Strategy Success. Today={rule_result.forecast_today:.2f} kWh, "
                    f"Tomorrow={rule_result.forecast_tomorrow:.2f} kWh, "
                    f"Day After={rule_result.forecast_day_after_tomorrow:.2f} kWh."
                )
                
            except Exception as rb_err:
                _LOGGER.error(f"Rule-Based (Iterative) forecast strategy also failed: {rb_err}.", exc_info=True)
                rule_result = None
        else:
            _LOGGER.error("Rule-Based strategy not available. Cannot calculate fallback.")

        
        today_ml = ml_result.forecast_today if ml_result else 0.0
        tomorrow_ml = ml_result.forecast_tomorrow if ml_result else 0.0
        day_after_ml = ml_result.forecast_day_after_tomorrow if ml_result else 0.0
        
        if rule_result:
            today_rule = rule_result.forecast_today
            tomorrow_rule = rule_result.forecast_tomorrow
            day_after_rule = rule_result.forecast_day_after_tomorrow
        else:
            _LOGGER.critical("Emergency Fallback: ML and Rule-Based strategies failed. Returning 0.0")
            today_rule = 0.0
            tomorrow_rule = 0.0
            day_after_rule = 0.0
            
            if ml_result:
                today_rule, tomorrow_rule, day_after_rule = today_ml, tomorrow_ml, day_after_ml
        
        if not ml_result:
            today_ml, tomorrow_ml, day_after_ml = today_rule, tomorrow_rule, day_after_rule
            model_accuracy = 0.0
            _LOGGER.debug("Blending: ML failed, using 100% Rule-Based result.")
        
        if ml_result:
            max_realistic_daily = self.solar_capacity * 6.0  # 6 hours full capacity max
            if today_ml > max_realistic_daily:
                _LOGGER.warning(
                    f"ML prediction unrealistic: {today_ml:.2f} kWh exceeds theoretical maximum "
                    f"{max_realistic_daily:.2f} kWh (solar_capacity={self.solar_capacity} kW * 6h). "
                    f"Rejecting ML, using Rule-Based only."
                )
                today_ml = today_rule
                tomorrow_ml = tomorrow_rule
                day_after_ml = day_after_rule
                model_accuracy = 0.0
        
        accuracy_weight = max(0.0, min(1.0, model_accuracy))

        ACCURACY_THRESHOLD = 0.75
        MIN_ML_WEIGHT = 0.15
        MIN_SAMPLES_FOR_CONFIDENT_ML = 50  # NEW: Minimum samples for confident ML usage

        # NEW: Check training sample count for adaptive blending
        training_sample_count = 0
        if self._ml_predictor and hasattr(self._ml_predictor, 'last_training_samples'):
            training_sample_count = self._ml_predictor.last_training_samples

        # ADAPTIVE BLENDING: Reduce ML weight if insufficient training data
        if training_sample_count < MIN_SAMPLES_FOR_CONFIDENT_ML and ml_result:
            # Calculate sample-based weight reduction
            sample_ratio = training_sample_count / MIN_SAMPLES_FOR_CONFIDENT_ML
            # Apply additional damping beyond accuracy-based damping
            sample_damping = sample_ratio ** 0.5  # Square root for gentler damping

            original_weight = accuracy_weight
            accuracy_weight = accuracy_weight * sample_damping

            _LOGGER.warning(
                f"ADAPTIVE BLENDING: Only {training_sample_count} training samples "
                f"(< {MIN_SAMPLES_FOR_CONFIDENT_ML} required). "
                f"Reducing ML weight: {original_weight:.1%} → {accuracy_weight:.1%} "
                f"(sample_damping={sample_damping:.3f})"
            )

        if accuracy_weight < ACCURACY_THRESHOLD and ml_result:
            # Quadratic damping for poor models: heavily penalize low accuracy
            raw_weight = (accuracy_weight / ACCURACY_THRESHOLD) ** 2
            accuracy_weight = MIN_ML_WEIGHT * raw_weight
            _LOGGER.info(
                f"ML accuracy ({model_accuracy:.1%}) below threshold ({ACCURACY_THRESHOLD:.0%}). "
                f"Applying aggressive ML damping: {accuracy_weight:.1%} ML / {(1-accuracy_weight):.1%} Rule-Based "
                f"(raw_weight={raw_weight:.3f})"
            )

        # ========================================================================
        # SAFEGUARD OPTION 1: DIVERSITY CHECK
        # ========================================================================
        safeguard_diversity_applied = False
        if ml_result and self._ml_predictor and training_sample_count > 0:
            try:
                # Get feature statistics from loaded weights
                feature_stds = {}
                if hasattr(self._ml_predictor, 'current_weights') and self._ml_predictor.current_weights:
                    if hasattr(self._ml_predictor.current_weights, 'feature_stds'):
                        feature_stds = self._ml_predictor.current_weights.feature_stds

                # Check cloud cover variance (critical weather feature)
                cloud_std = feature_stds.get('weather_cloud_percent', 0.0)
                MIN_CLOUD_VARIANCE = 10.0  # Require at least 10% variance in training data

                if cloud_std < MIN_CLOUD_VARIANCE:
                    original_weight = accuracy_weight
                    accuracy_weight = min(accuracy_weight, 0.05)  # Cap at 5% ML
                    safeguard_diversity_applied = True

                    _LOGGER.warning(
                        f"SAFEGUARD OPTION 1 (Diversity Check) TRIGGERED: "
                        f"Low cloud variance in training data ({cloud_std:.2f}% < {MIN_CLOUD_VARIANCE}%). "
                        f"Training data appears homogeneous. "
                        f"Limiting ML weight: {original_weight:.1%} → {accuracy_weight:.1%}"
                    )
            except Exception as e:
                _LOGGER.debug(f"Could not apply diversity check safeguard: {e}")

        # ========================================================================
        # SAFEGUARD OPTION 2: SANITY CHECK (ML vs Rule-Based)
        # ========================================================================
        safeguard_sanity_applied = False
        if ml_result and rule_result and accuracy_weight > 0:
            try:
                # Compare ML vs Rule-Based predictions
                if today_rule > 0:
                    ratio = today_ml / today_rule
                    MIN_RATIO = 0.30  # ML should not be less than 30% of Rule-based
                    MAX_RATIO = 3.0   # ML should not be more than 300% of Rule-based

                    if ratio < MIN_RATIO:
                        original_weight = accuracy_weight
                        accuracy_weight = accuracy_weight * 0.10  # Reduce ML by 90%
                        safeguard_sanity_applied = True

                        _LOGGER.warning(
                            f"SAFEGUARD OPTION 2 (Sanity Check) TRIGGERED: "
                            f"ML prediction suspiciously low ({ratio:.1%} of Rule-based). "
                            f"ML={today_ml:.2f} kWh, Rule={today_rule:.2f} kWh. "
                            f"Reducing ML weight: {original_weight:.1%} → {accuracy_weight:.1%}"
                        )
                    elif ratio > MAX_RATIO:
                        original_weight = accuracy_weight
                        accuracy_weight = accuracy_weight * 0.50  # Reduce ML by 50%
                        safeguard_sanity_applied = True

                        _LOGGER.warning(
                            f"SAFEGUARD OPTION 2 (Sanity Check) TRIGGERED: "
                            f"ML prediction suspiciously high ({ratio:.1%} of Rule-based). "
                            f"ML={today_ml:.2f} kWh, Rule={today_rule:.2f} kWh. "
                            f"Reducing ML weight: {original_weight:.1%} → {accuracy_weight:.1%}"
                        )
            except Exception as e:
                _LOGGER.debug(f"Could not apply sanity check safeguard: {e}")

        # Log safeguard summary
        if safeguard_diversity_applied or safeguard_sanity_applied:
            safeguards = []
            if safeguard_diversity_applied:
                safeguards.append("Diversity")
            if safeguard_sanity_applied:
                safeguards.append("Sanity")
            _LOGGER.info(
                f"🛡️  SAFEGUARDS ACTIVE: {', '.join(safeguards)} | "
                f"Final ML weight: {accuracy_weight:.1%} (protecting against poor predictions)"
            )

        rule_weight = 1.0 - accuracy_weight

        # Blend today and tomorrow with ML + Rule-Based
        final_today = (today_ml * accuracy_weight) + (today_rule * rule_weight)
        final_tomorrow = (tomorrow_ml * accuracy_weight) + (tomorrow_rule * rule_weight)

        # STRATEGIC FIX: Day after tomorrow uses ONLY Rule-Based (no ML blending)
        # ML model struggles with day+2 predictions due to lag feature limitations
        final_day_after = day_after_rule

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
            f"ML=({today_ml:.2f}, {tomorrow_ml:.2f}, {day_after_ml:.2f}), "
            f"Rule=({today_rule:.2f}, {tomorrow_rule:.2f}, {day_after_rule:.2f}) -> "
            f"Final=({final_today:.2f}, {final_tomorrow:.2f}, {final_day_after:.2f} [Rule-Only]) kWh"
        )

        _LOGGER.debug(
            f"Final values before returning - today: {final_today}, tomorrow: {final_tomorrow}, day_after: {final_day_after}"
        )

        # Extract best hour from whichever strategy provided it (prefer ML if available)
        best_hour = None
        best_hour_kwh = None
        if ml_result and ml_result.best_hour_today is not None:
            best_hour = ml_result.best_hour_today
            best_hour_kwh = ml_result.best_hour_production_kwh
            _LOGGER.debug(f"Using ML best hour: {best_hour}:00 with {best_hour_kwh:.3f} kWh")
        elif rule_result and rule_result.best_hour_today is not None:
            best_hour = rule_result.best_hour_today
            best_hour_kwh = rule_result.best_hour_production_kwh
            _LOGGER.debug(f"Using Rule-based best hour: {best_hour}:00 with {best_hour_kwh:.3f} kWh")

        # Blend hourly values from both strategies
        hourly_values = []
        if ml_result and ml_result.hourly_values and rule_result and rule_result.hourly_values:
            # Both strategies available: blend hourly values
            _LOGGER.debug(f"Blending hourly values: ML={len(ml_result.hourly_values)} hours, Rule={len(rule_result.hourly_values)} hours")

            # Create lookup dict for rule-based values by hour
            rule_by_hour = {h['hour']: h for h in rule_result.hourly_values}

            for ml_hour in ml_result.hourly_values:
                hour = ml_hour['hour']
                rule_hour = rule_by_hour.get(hour)

                if rule_hour:
                    # Blend this hour
                    ml_kwh = ml_hour['production_kwh']
                    rule_kwh = rule_hour['production_kwh']
                    blended_kwh = (ml_kwh * accuracy_weight) + (rule_kwh * rule_weight)

                    hourly_values.append({
                        "hour": hour,
                        "datetime": ml_hour['datetime'],
                        "production_kwh": round(blended_kwh, 3),
                        "date": ml_hour['date']
                    })
                else:
                    # Only ML has this hour
                    hourly_values.append(ml_hour)

            _LOGGER.debug(f"Blended {len(hourly_values)} hourly values")
        elif ml_result and ml_result.hourly_values:
            # Only ML available
            hourly_values = ml_result.hourly_values
            _LOGGER.debug(f"Using ML hourly values: {len(hourly_values)} hours")
        elif rule_result and rule_result.hourly_values:
            # Only Rule-based available
            hourly_values = rule_result.hourly_values
            _LOGGER.debug(f"Using Rule-based hourly values: {len(hourly_values)} hours")

        # Note: prediction_history.json removed - hourly_predictions.json is now the single source of truth

        return {
            "today": round(final_today, 2),
            "tomorrow": round(final_tomorrow, 2),
            "day_after_tomorrow": round(final_day_after, 2),
            "peak_time": "12:00",
            "confidence": round(final_confidence, 1),
            "method": method_str,
            "model_accuracy": model_accuracy if self._ml_predictor else None,
            "best_hour": best_hour,
            "best_hour_kwh": round(best_hour_kwh, 3) if best_hour_kwh is not None else None,
            "hourly": hourly_values  # Include hourly breakdown
        }

    def calculate_next_hour_prediction(
        self,
        forecast_today_kwh: float,
        weather_data: Optional[Dict[str, Any]] = None,
        sensor_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Estimates the solar production for the next full hour"""
        _LOGGER.debug("Calculating next hour prediction...")
        try:
            now_local = dt_util.now()
            target_dt_local = now_local + timedelta(hours=1)
            target_hour = target_dt_local.hour

            # Use production hour check (astronomy cache with safety margins)
            if not self.is_production_hour(target_dt_local):
                if sensor_data and sensor_data.get('lux') is not None:
                    if sensor_data['lux'] < 500:  # Too dark for meaningful production
                        _LOGGER.debug(
                            f"Target hour {target_hour} has insufficient light (lux={sensor_data['lux']}). "
                            f"Next hour prediction is 0.0 kWh."
                        )
                        return 0.0
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

            max_hourly_kwh = self.solar_capacity * 1.2  # 20% safety margin
            adjusted_kwh = min(adjusted_kwh, max_hourly_kwh)

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
        """Calculates the base production for a specific hour using the ML hourly profile"""
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
        """Calculates adjustment multipliers based on current weather and sensor readings"""
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
            _LOGGER.debug(f"Using Temperature ({temp_value}C) for adjustment factor: {factors['temperature']:.2f}")
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