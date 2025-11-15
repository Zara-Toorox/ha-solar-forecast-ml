"""
Forecast Strategy Implementation

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
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from homeassistant.util import dt as ha_dt_util

from ..core.core_exceptions import MLModelException
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..services.service_error_handler import ErrorHandlingService
from .forecast_strategy_base import ForecastResult, ForecastStrategy

_LOGGER = logging.getLogger(__name__)


class MLForecastStrategy(ForecastStrategy):
    """Implements the forecast strategy using a trained Machine Learning model"""

    def __init__(self, ml_predictor: Any, error_handler: Optional[ErrorHandlingService] = None):
        """Initialize the ML Forecast Strategy"""
        super().__init__("ml_forecast")
        self.ml_predictor = ml_predictor
        self.error_handler = error_handler

        self.PREDICTION_MIN_VALUE = 0.0

        # Set HOURLY_PREDICTION_MAX_VALUE dynamically based on system's peak power
        # Get peak power from ML predictor (in kW)
        peak_power_kw = getattr(ml_predictor, "peak_power_kw", 0.0)

        if peak_power_kw > 0:
            # Max hourly production ≈ peak_power_kw * 1.2 (20% safety margin)
            from ..const import DEFAULT_MAX_HOURLY_KWH, HOURLY_PRODUCTION_SAFETY_MARGIN

            self.HOURLY_PREDICTION_MAX_VALUE = peak_power_kw * HOURLY_PRODUCTION_SAFETY_MARGIN
            _LOGGER.debug(
                f"MLForecastStrategy: Max hourly prediction set to {self.HOURLY_PREDICTION_MAX_VALUE:.2f} kWh "
                f"(based on {peak_power_kw} kWp)"
            )
        else:
            # Fallback if peak power not configured
            from ..const import DEFAULT_MAX_HOURLY_KWH

            self.HOURLY_PREDICTION_MAX_VALUE = DEFAULT_MAX_HOURLY_KWH
            _LOGGER.warning(
                f"MLForecastStrategy: Peak power not configured, using fallback max hourly: "
                f"{self.HOURLY_PREDICTION_MAX_VALUE} kWh"
            )

        self.TOMORROW_DISCOUNT_FACTOR = 0.92

    def is_available(self) -> bool:
        """Checks if the ML Predictor instance is available and reports itself as healthy"""
        if not self.ml_predictor:
            _LOGGER.debug("ML strategy unavailable: MLPredictor instance is missing.")
            return False

        try:
            if hasattr(self.ml_predictor, "is_healthy") and callable(self.ml_predictor.is_healthy):
                is_healthy = self.ml_predictor.is_healthy()
                if not is_healthy:
                    _LOGGER.debug("ML strategy unavailable: MLPredictor reports unhealthy.")
                return is_healthy
            else:
                _LOGGER.debug("MLPredictor has no 'is_healthy' method, assuming available.")
                return True
        except Exception as e:
            _LOGGER.warning(f"ML Predictor health check failed with an error: {e}")
            return False

    def get_priority(self) -> int:
        """Returns the priority of this strategy ML has the highest priority"""
        return 100

    async def calculate_forecast(
        self,
        hourly_weather_forecast: List[Dict[str, Any]],
        sensor_data: Dict[str, Any],
        lag_features: Dict[str, Any],
        correction_factor: float,
    ) -> ForecastResult:
        """Calculates the solar forecast using the ML model via the MLPredictor instance"""
        _LOGGER.debug("Attempting forecast calculation using ML (Iterative) strategy...")

        if not self.is_available():
            _LOGGER.warning("ML Predictor is unavailable or unhealthy. Cannot use ML strategy.")
            raise MLModelException("ML Predictor not available or unhealthy")

        try:
            total_today_kwh = 0.0
            total_tomorrow_kwh = 0.0
            total_day_after_kwh = 0.0

            # Track best production hour for today
            best_hour_today = None
            best_hour_production = 0.0

            # Collect hourly values for detailed breakdown
            hourly_values = []

            now_local = dt_util.now()  # now() already returns LOCAL time
            today_date = now_local.date()
            tomorrow_date = today_date + timedelta(days=1)
            day_after_tomorrow_date = today_date + timedelta(days=2)

            for hour_data in hourly_weather_forecast:
                try:
                    hour_dt_local = hour_data.get("local_datetime")
                    if not hour_dt_local:
                        _LOGGER.warning("Skipping hour, missing 'local_datetime'")
                        continue

                    # Parse if string (ISO format from JSON serialization)
                    if isinstance(hour_dt_local, str):
                        hour_dt_local = dt_util.parse_datetime(hour_dt_local)
                        if not hour_dt_local:
                            _LOGGER.warning("Skipping hour, invalid 'local_datetime' format")
                            continue

                    # Ensure timezone-aware for comparison
                    hour_dt_local = dt_util.as_local(hour_dt_local)

                    # CRITICAL: Skip past hours - only forecast current hour and future
                    if hour_dt_local < now_local:
                        continue

                    # CRITICAL: Skip hours outside realistic production time
                    # This prevents ML from predicting production during darkness
                    hour_local = hour_dt_local.hour
                    hour_date = hour_dt_local.date()

                    # Try to get dynamic production window from astronomy cache or sun.sun
                    is_production_hour = False
                    try:
                        # Method 1: Try astronomy cache first
                        if hasattr(self.ml_predictor, "_historical_cache"):
                            astro_cache = self.ml_predictor._historical_cache.get(
                                "astronomy_cache", {}
                            )
                            date_key = hour_date.isoformat()

                            if date_key in astro_cache:
                                sunrise_str = astro_cache[date_key].get("sunrise")
                                sunset_str = astro_cache[date_key].get("sunset")

                                if sunrise_str and sunset_str:
                                    sunrise = ha_dt_util.parse_datetime(sunrise_str)
                                    sunset = ha_dt_util.parse_datetime(sunset_str)

                                    if sunrise and sunset:
                                        # Production window: sunrise - 1 hour to sunset + 1 hour
                                        start_hour = max(0, sunrise.hour - 1)
                                        end_hour = min(23, sunset.hour + 1)
                                        is_production_hour = start_hour <= hour_local <= end_hour

                        # Method 2: Fallback to sun.sun entity (always available)
                        if not is_production_hour and hasattr(self.ml_predictor, "hass"):
                            sun_state = self.ml_predictor.hass.states.get("sun.sun")
                            if sun_state and sun_state.attributes:
                                next_rising = sun_state.attributes.get("next_rising")
                                next_setting = sun_state.attributes.get("next_setting")

                                if next_rising and next_setting:
                                    sunrise = ha_dt_util.parse_datetime(next_rising)
                                    sunset = ha_dt_util.parse_datetime(next_setting)

                                    if sunrise and sunset:
                                        # Production window: sunrise - 1 hour to sunset + 1 hour
                                        start_hour = max(0, sunrise.hour - 1)
                                        end_hour = min(23, sunset.hour + 1)
                                        is_production_hour = start_hour <= hour_local <= end_hour

                    except Exception as e:
                        _LOGGER.debug(
                            f"Failed to get dynamic production window: {e}, using fallback"
                        )

                    # Method 3: Fallback to conservative seasonal hours if dynamic methods fail
                    if not is_production_hour:
                        month = hour_dt_local.month
                        _LOGGER.debug(
                            f"Using seasonal fallback for hour {hour_local} (month {month}) - "
                            f"astronomy cache unavailable"
                        )
                        if month in [11, 12, 1]:  # Winter
                            is_production_hour = 6 <= hour_local <= 16
                        elif month in [5, 6, 7, 8]:  # Summer
                            is_production_hour = 5 <= hour_local <= 20
                        else:  # Spring/Fall
                            is_production_hour = 6 <= hour_local <= 18

                    if not is_production_hour:
                        # Skip hours outside production time (no log to reduce noise)
                        continue

                    ml_sensor_data_input = lag_features.copy()

                    # Load astronomy data for this hour from cache
                    # Previously astronomy_basic and astronomy_advanced were empty dicts causing ML to use defaults
                    # This resulted in 36% of features being incorrect (6 astronomy + 10 derived features)
                    astronomy_basic = {}
                    astronomy_advanced = {}

                    try:
                        if hasattr(self.ml_predictor, "_historical_cache"):
                            astro_cache = self.ml_predictor._historical_cache.get(
                                "astronomy_cache", {}
                            )
                            date_key = hour_date.isoformat()

                            if date_key in astro_cache:
                                day_data = astro_cache[date_key]

                                # Extract basic astronomy (sunrise, sunset, etc.)
                                astronomy_basic = {
                                    "sunrise": day_data.get("sunrise_local"),
                                    "sunset": day_data.get("sunset_local"),
                                    "solar_noon": day_data.get("solar_noon_local"),
                                    "daylight_hours": day_data.get("daylight_hours"),
                                }

                                # Extract advanced hourly astronomy for this specific hour
                                hourly_data = day_data.get("hourly", {})
                                hour_str = str(hour_local)

                                if hour_str in hourly_data:
                                    hour_astro = hourly_data[hour_str]
                                    astronomy_advanced = {
                                        "elevation_deg": hour_astro.get("elevation_deg"),
                                        "azimuth_deg": hour_astro.get("azimuth_deg"),
                                        "clear_sky_solar_radiation_wm2": hour_astro.get(
                                            "clear_sky_solar_radiation_wm2"
                                        ),
                                        "theoretical_max_pv_kwh": hour_astro.get(
                                            "theoretical_max_pv_kwh"
                                        ),
                                        "hours_since_solar_noon": hour_astro.get(
                                            "hours_since_solar_noon"
                                        ),
                                        "day_progress_ratio": hour_astro.get("day_progress_ratio"),
                                    }
                                    _LOGGER.debug(
                                        f"Loaded astronomy for {date_key} {hour_local:02d}:00 - "
                                        f"elevation={astronomy_advanced.get('elevation_deg', 'N/A')}°"
                                    )
                                else:
                                    _LOGGER.debug(
                                        f"No hourly astronomy data for {date_key} hour {hour_local}"
                                    )
                            else:
                                _LOGGER.debug(f"No astronomy cache for date {date_key}")
                    except Exception as e:
                        _LOGGER.warning(f"Failed to load astronomy data for prediction: {e}")

                    # Check if using V2 FeatureEngineer (has different signature)
                    from ..ml.ml_feature_engineering_v2 import FeatureEngineerV2

                    if isinstance(self.ml_predictor.feature_engineer, FeatureEngineerV2):
                        # V2: Pass a single record dict with REAL astronomy data
                        # BUG #13 FIX: Add all time-based fields required by Feature Engineering
                        current_hour = dt_util.now().hour
                        target_day_of_year = hour_date.timetuple().tm_yday
                        target_month = hour_date.month
                        target_day_of_week = hour_date.weekday()

                        # Calculate season from month
                        if target_month in [12, 1, 2]:
                            target_season = "winter"
                        elif target_month in [3, 4, 5]:
                            target_season = "spring"
                        elif target_month in [6, 7, 8]:
                            target_season = "summer"
                        else:
                            target_season = "autumn"

                        record = {
                            "weather_data": hour_data,
                            "sensor_data": ml_sensor_data_input,
                            "target_hour": hour_local,
                            "target_date": hour_date.isoformat(),
                            "target_datetime": hour_dt_local.isoformat(),
                            # Time-based fields (BUG #13 FIX)
                            "target_day_of_year": target_day_of_year,
                            "target_month": target_month,
                            "target_day_of_week": target_day_of_week,
                            "target_season": target_season,
                            "prediction_created_hour": current_hour,
                            # Use real astronomy data from cache instead of empty dicts!
                            "astronomy_basic": astronomy_basic,
                            "astronomy_advanced": astronomy_advanced,
                        }
                        features = self.ml_predictor.feature_engineer.extract_features(record)
                    else:
                        # V1: Pass separate parameters
                        features = await self.ml_predictor.feature_engineer.extract_features(
                            weather_data=hour_data,
                            sensor_data=ml_sensor_data_input,
                            prediction_hour=hour_local,
                            prediction_date=hour_dt_local,
                        )

                    # DEBUG BUG #13: Log RAW features BEFORE scaler
                    if features and len(features) > 1:
                        _LOGGER.debug(f"🔍 BUG#13 - RAW features[1] (target_day_of_year): {features[1]:.4f}")

                    if self.ml_predictor.scaler.is_fitted:
                        features_scaled = self.ml_predictor.scaler.transform_single(features)
                        # DEBUG BUG #13: Log SCALED features AFTER scaler
                        if features_scaled and len(features_scaled) > 1:
                            _LOGGER.debug(f"🔍 BUG#13 - SCALED features[1] (target_day_of_year): {features_scaled[1]:.4f}")
                    else:
                        features_scaled = features

                    prediction_result = await self.ml_predictor.prediction_orchestrator.predict(
                        features_scaled
                    )

                    hourly_kwh = prediction_result.prediction
                    hourly_kwh = max(
                        self.PREDICTION_MIN_VALUE, min(hourly_kwh, self.HOURLY_PREDICTION_MAX_VALUE)
                    )

                    # Store hourly value
                    hourly_values.append(
                        {
                            "hour": hour_local,
                            "datetime": hour_dt_local.isoformat(),
                            "production_kwh": round(hourly_kwh, 3),
                            "date": hour_date.isoformat(),
                        }
                    )

                    if hour_date == today_date:
                        total_today_kwh += hourly_kwh
                        # Track best production hour for today
                        if hourly_kwh > best_hour_production:
                            best_hour_today = hour_local
                            best_hour_production = hourly_kwh
                    elif hour_date == tomorrow_date:
                        total_tomorrow_kwh += hourly_kwh
                    elif hour_date == day_after_tomorrow_date:
                        total_day_after_kwh += hourly_kwh

                except Exception as e_inner:
                    _LOGGER.warning(
                        f"ML strategy failed to process hour {hour_data.get('local_hour')}: {e_inner}"
                    )
                    continue

            total_tomorrow_kwh *= self.TOMORROW_DISCOUNT_FACTOR
            total_day_after_kwh *= (
                self.TOMORROW_DISCOUNT_FACTOR * 0.95
            )  # Additional discount for day after

            _LOGGER.debug(
                f"ML (Iterative) iteration complete. "
                f"Today (raw): {total_today_kwh:.2f} kWh, "
                f"Tomorrow (raw): {total_tomorrow_kwh:.2f} kWh, "
                f"Day After (raw): {total_day_after_kwh:.2f} kWh"
            )

            try:
                current_yield = sensor_data.get("current_yield")
                if current_yield is not None and current_yield > 0:
                    current_yield_float = float(current_yield)

                    if current_yield_float > total_today_kwh:
                        end_hour_local = 21
                        remaining_hours = max(0, end_hour_local - now_local.hour)

                        additional_forecast = total_today_kwh * 0.1 * remaining_hours

                        adjusted_today_forecast = current_yield_float + additional_forecast

                        _LOGGER.info(
                            f"Minimum forecast adjustment (ML): Current yield {current_yield_float:.2f} kWh > "
                            f" Forecast {total_today_kwh:.2f} kWh. "
                            f"Adjusted to {adjusted_today_forecast:.2f} kWh."
                        )

                        original_today_forecast = total_today_kwh
                        total_today_kwh = adjusted_today_forecast

                        if original_today_forecast > 0:
                            adjustment_ratio = total_today_kwh / original_today_forecast
                            total_tomorrow_kwh = total_tomorrow_kwh * adjustment_ratio
                            total_day_after_kwh = total_day_after_kwh * adjustment_ratio

            except Exception as e:
                _LOGGER.debug(f"Minimum forecast check (ML) could not be performed: {e}")

            model_accuracy = getattr(self.ml_predictor, "current_accuracy", 0.0)

            confidence_today = max(0.0, min(100.0, (model_accuracy or 0.0) * 100.0))
            confidence_tomorrow = max(0.0, min(100.0, confidence_today * 0.95))
            confidence_day_after = max(0.0, min(100.0, confidence_tomorrow * 0.90))

            feature_count = (
                len(self.ml_predictor.feature_engineer.feature_names)
                if self.ml_predictor.feature_engineer
                else 0
            )

            result = ForecastResult(
                forecast_today=total_today_kwh,
                forecast_tomorrow=total_tomorrow_kwh,
                forecast_day_after_tomorrow=total_day_after_kwh,
                confidence_today=confidence_today,
                confidence_tomorrow=confidence_tomorrow,
                confidence_day_after=confidence_day_after,
                method="ml_iterative",
                calibrated=True,
                features_used=feature_count,
                model_accuracy=model_accuracy,
                best_hour_today=best_hour_today,
                best_hour_production_kwh=(
                    best_hour_production if best_hour_today is not None else None
                ),
                hourly_values=hourly_values,
            )

            accuracy_str = (
                f", model_acc={result.model_accuracy:.3f}"
                if result.model_accuracy is not None
                else ""
            )
            _LOGGER.info(
                f"ML (Iterative) Forecast successful: "
                f"Today={result.forecast_today:.2f} kWh, "
                f"Tomorrow={result.forecast_tomorrow:.2f} kWh, "
                f"Day After={result.forecast_day_after_tomorrow:.2f} kWh, "
                f"Confidence={result.confidence_today:.1f}%, Method='{result.method}'{accuracy_str}"
            )

            return result

        except MLModelException as me:
            _LOGGER.error(f"ML (Iterative) Forecast failed: {me}")
            if self.error_handler:
                await self.error_handler.handle_error(me, "ml_prediction_strategy")
            raise
        except Exception as e:
            _LOGGER.error(
                f"Unexpected error during ML (Iterative) Forecast calculation: {e}", exc_info=True
            )
            err = MLModelException(f"ML forecast calculation (iterative) failed: {e}")
            if self.error_handler:
                await self.error_handler.handle_error(err, "ml_prediction_strategy")
            raise err from e
