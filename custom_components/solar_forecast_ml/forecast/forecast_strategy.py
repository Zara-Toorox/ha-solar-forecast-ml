"""Forecast Strategy Implementation V10.0.0 @zara

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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from homeassistant.util import dt as ha_dt_util

from ..astronomy.astronomy_cache_manager import get_cache_manager
from ..core.core_exceptions import MLModelException
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..services.service_error_handler import ErrorHandlingService
from .forecast_strategy_base import ForecastResult, ForecastStrategy

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

class MLForecastStrategy(ForecastStrategy):
    """Implements the forecast strategy using a trained Machine Learning model

    ASYNC I/O: Uses async_add_executor_job for non-blocking file operations.
    """

    def __init__(
        self,
        ml_predictor: Any,
        error_handler: Optional[ErrorHandlingService] = None,
        hass: Optional["HomeAssistant"] = None
    ):
        """Initialize the ML Forecast Strategy

        Args:
            ml_predictor: ML predictor instance
            error_handler: Error handling service
            hass: Home Assistant instance for async file I/O
        """
        super().__init__("ml_forecast")
        self.ml_predictor = ml_predictor
        self.error_handler = error_handler
        self.hass = hass

        self.PREDICTION_MIN_VALUE = 0.0

        peak_power_kw = getattr(ml_predictor, "peak_power_kw", 0.0)

        if peak_power_kw > 0:

            from ..const import DEFAULT_MAX_HOURLY_KWH, HOURLY_PRODUCTION_SAFETY_MARGIN

            self.HOURLY_PREDICTION_MAX_VALUE = peak_power_kw * HOURLY_PRODUCTION_SAFETY_MARGIN
            _LOGGER.debug(
                f"MLForecastStrategy: Max hourly prediction set to {self.HOURLY_PREDICTION_MAX_VALUE:.2f} kWh "
                f"(based on {peak_power_kw} kWp)"
            )
        else:

            from ..const import DEFAULT_MAX_HOURLY_KWH

            self.HOURLY_PREDICTION_MAX_VALUE = DEFAULT_MAX_HOURLY_KWH
            _LOGGER.warning(
                f"MLForecastStrategy: Peak power not configured, using fallback max hourly: "
                f"{self.HOURLY_PREDICTION_MAX_VALUE} kWh"
            )

        self.TOMORROW_DISCOUNT_FACTOR = 0.92

    def is_available(self) -> bool:
        """Checks if the ML Predictor instance is available and reports itself as healthy @zara"""
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
        """Returns the priority of this strategy ML has the highest priority @zara"""
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

            best_hour_today = None
            best_hour_production = 0.0

            hourly_values = []

            now_local = dt_util.now()
            today_date = now_local.date()
            tomorrow_date = today_date + timedelta(days=1)
            day_after_tomorrow_date = today_date + timedelta(days=2)

            for hour_data in hourly_weather_forecast:
                try:
                    hour_dt_local = hour_data.get("local_datetime")
                    if not hour_dt_local:
                        _LOGGER.warning("Skipping hour, missing 'local_datetime'")
                        continue

                    if isinstance(hour_dt_local, str):
                        hour_dt_local = dt_util.parse_datetime(hour_dt_local)
                        if not hour_dt_local:
                            _LOGGER.warning("Skipping hour, invalid 'local_datetime' format")
                            continue

                    hour_dt_local = dt_util.as_local(hour_dt_local)

                    if hour_dt_local < now_local:
                        continue

                    hour_local = hour_dt_local.hour
                    hour_date = hour_dt_local.date()

                    is_production_hour = False
                    cache_data_found = False

                    try:

                        cache_manager = get_cache_manager()
                        date_key = hour_date.isoformat()

                        if cache_manager.is_loaded():
                            day_data = cache_manager.get_day_data(date_key)

                            if day_data:
                                cache_data_found = True
                                sunrise_str = day_data.get("sunrise_local")
                                sunset_str = day_data.get("sunset_local")

                                if sunrise_str and sunset_str:
                                    sunrise = ha_dt_util.parse_datetime(sunrise_str)
                                    sunset = ha_dt_util.parse_datetime(sunset_str)

                                    if sunrise and sunset:

                                        hour_start_dt = hour_dt_local.replace(minute=0, second=0, microsecond=0)
                                        hour_end_dt = hour_start_dt + timedelta(hours=1)

                                        production_start = sunrise - timedelta(minutes=30)
                                        production_end = sunset + timedelta(minutes=30)

                                        is_production_hour = hour_end_dt > production_start and hour_start_dt < production_end
                            else:
                                _LOGGER.debug(f"No astronomy cache data for {date_key} - cannot determine production window")

                    except Exception as e:
                        _LOGGER.debug(
                            f"Failed to get dynamic production window: {e}"
                        )

                    if not cache_data_found:

                        month = hour_dt_local.month

                        now_date = dt_util.now().date()
                        days_ahead = (hour_date - now_date).days

                        if days_ahead <= 2:

                            _LOGGER.error(
                                f"Astronomy cache missing for {date_key} (day +{days_ahead}) - "
                                f"this should exist! Using seasonal fallback for hour {hour_local} (month {month})."
                            )
                        else:

                            _LOGGER.debug(
                                f"Using seasonal fallback for hour {hour_local} (day +{days_ahead}, month {month}) - "
                                f"astronomy cache unavailable"
                            )

                        if month in [11, 12, 1]:
                            is_production_hour = 7 <= hour_local <= 16
                        elif month in [5, 6, 7, 8]:
                            is_production_hour = 5 <= hour_local <= 20
                        else:
                            is_production_hour = 6 <= hour_local <= 18

                    if not is_production_hour:

                        continue

                    ml_sensor_data_input = lag_features.copy()

                    astronomy_basic = {}
                    astronomy_advanced = {}

                    try:
                        cache_manager = get_cache_manager()
                        date_key = hour_date.isoformat()

                        if cache_manager.is_loaded():
                            day_data = cache_manager.get_day_data(date_key)

                            if day_data:

                                astronomy_basic = {
                                    "sunrise": day_data.get("sunrise_local"),
                                    "sunset": day_data.get("sunset_local"),
                                    "solar_noon": day_data.get("solar_noon_local"),
                                    "daylight_hours": day_data.get("daylight_hours"),
                                }

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

                    from ..ml.ml_feature_engineering_v3 import FeatureEngineerV3

                    if isinstance(self.ml_predictor.feature_engineer, FeatureEngineerV3):

                        current_hour = dt_util.now().hour
                        target_day_of_year = hour_date.timetuple().tm_yday
                        target_month = hour_date.month
                        target_day_of_week = hour_date.weekday()

                        if target_month in [12, 1, 2]:
                            target_season = "winter"
                        elif target_month in [3, 4, 5]:
                            target_season = "spring"
                        elif target_month in [6, 7, 8]:
                            target_season = "summer"
                        else:
                            target_season = "autumn"

                        corrected_weather = {
                            "temperature": hour_data.get("temperature"),
                            "solar_radiation_wm2": hour_data.get("solar_radiation_wm2"),
                            "wind": hour_data.get("wind_speed"),
                            "humidity": hour_data.get("humidity"),
                            "rain": hour_data.get("precipitation", 0),
                            "clouds": hour_data.get("cloud_cover")
                        }

                        astronomy_merged = {}
                        if astronomy_advanced:
                            astronomy_merged = {
                                "sun_elevation_deg": astronomy_advanced.get("elevation_deg", -30.0),
                                "theoretical_max_kwh": astronomy_advanced.get("theoretical_max_pv_kwh", 0.0),
                                "clear_sky_radiation_wm2": astronomy_advanced.get("clear_sky_solar_radiation_wm2", 0.0),
                            }

                        record = {
                            "weather_data": hour_data,
                            "weather_corrected": corrected_weather,
                            "sensor_data": ml_sensor_data_input,
                            "target_hour": hour_local,
                            "target_date": hour_date.isoformat(),
                            "target_datetime": hour_dt_local.isoformat(),

                            "target_day_of_year": target_day_of_year,
                            "target_month": target_month,
                            "target_day_of_week": target_day_of_week,
                            "target_season": target_season,
                            "prediction_created_hour": current_hour,

                            "astronomy": astronomy_merged,

                            "production_yesterday": ml_sensor_data_input.get("production_yesterday", 0.0),
                            "production_same_hour_yesterday": ml_sensor_data_input.get(
                                "production_same_hour_yesterday", 0.0
                            ),

                            "astronomy_basic": astronomy_basic,
                            "astronomy_advanced": astronomy_advanced,
                        }
                        features = self.ml_predictor.feature_engineer.extract_features(record)
                    else:

                        _LOGGER.error("Feature engineer is not V3 - this should not happen!")
                        continue

                    if self.ml_predictor.scaler.is_fitted:
                        features_scaled = self.ml_predictor.scaler.transform_single(features)
                    else:
                        features_scaled = features

                    prediction_result = await self.ml_predictor.prediction_orchestrator.predict(
                        features_scaled
                    )

                    hourly_kwh = prediction_result.prediction
                    hourly_kwh = max(
                        self.PREDICTION_MIN_VALUE, min(hourly_kwh, self.HOURLY_PREDICTION_MAX_VALUE)
                    )

                    hourly_kwh *= correction_factor

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
            )

            correction_info = f", correction_factor={correction_factor:.2f}" if correction_factor != 1.0 else ""
            _LOGGER.debug(
                f"ML (Iterative) iteration complete. "
                f"Today (raw): {total_today_kwh:.2f} kWh, "
                f"Tomorrow (raw): {total_tomorrow_kwh:.2f} kWh, "
                f"Day After (raw): {total_day_after_kwh:.2f} kWh{correction_info}"
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
