"""
ML-based Forecast Strategy.
Uses an MLPredictor instance for Machine Learning predictions.

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
from datetime import datetime, timedelta

from ..forecast.strategy import ForecastStrategy, ForecastResult
from ..services.error_handler import ErrorHandlingService
from ..exceptions import MLModelException
from ..core.helpers import SafeDateTimeUtil as dt_util


_LOGGER = logging.getLogger(__name__)


class MLForecastStrategy(ForecastStrategy):
    """
    Implements the forecast strategy using a trained Machine Learning model
    provided by an MLPredictor instance.
    
    Berechnet eine iterative stÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ndliche Prognose.
    """

    def __init__(self, ml_predictor: Any, error_handler: Optional[ErrorHandlingService] = None):
        """
        Initialize the ML Forecast Strategy.

        Args:
            ml_predictor: An instance of the MLPredictor class (or compatible).
            error_handler: Optional ErrorHandlingService instance for logging errors.
        """
        super().__init__("ml_forecast")
        self.ml_predictor = ml_predictor
        self.error_handler = error_handler

        self.PREDICTION_MIN_VALUE = 0.0
        
        # Set HOURLY_PREDICTION_MAX_VALUE dynamically based on system's peak power
        # Get peak power from ML predictor (in kW)
        peak_power_kw = getattr(ml_predictor, 'peak_power_kw', 0.0)
        
        if peak_power_kw > 0:
            # Max hourly production â‰ˆ peak_power_kw * 1.2 (20% safety margin)
            from ..const import HOURLY_PRODUCTION_SAFETY_MARGIN, DEFAULT_MAX_HOURLY_KWH
            self.HOURLY_PREDICTION_MAX_VALUE = peak_power_kw * HOURLY_PRODUCTION_SAFETY_MARGIN
            _LOGGER.debug(f"MLForecastStrategy: Max hourly prediction set to {self.HOURLY_PREDICTION_MAX_VALUE:.2f} kWh "
                         f"(based on {peak_power_kw} kWp)")
        else:
            # Fallback if peak power not configured
            from ..const import DEFAULT_MAX_HOURLY_KWH
            self.HOURLY_PREDICTION_MAX_VALUE = DEFAULT_MAX_HOURLY_KWH
            _LOGGER.warning(f"MLForecastStrategy: Peak power not configured, using fallback max hourly: "
                           f"{self.HOURLY_PREDICTION_MAX_VALUE} kWh")
        
        self.TOMORROW_DISCOUNT_FACTOR = 0.92

    def is_available(self) -> bool:
        """
        Checks if the ML Predictor instance is available and reports itself as healthy.
        """
        if not self.ml_predictor:
            _LOGGER.debug("ML strategy unavailable: MLPredictor instance is missing.")
            return False

        try:
            if hasattr(self.ml_predictor, 'is_healthy') and callable(self.ml_predictor.is_healthy):
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
        """
        Returns the priority of this strategy. ML has the highest priority.
        """
        return 100

    async def calculate_forecast(
        self,
        hourly_weather_forecast: List[Dict[str, Any]],
        sensor_data: Dict[str, Any],
        lag_features: Dict[str, Any],
        correction_factor: float
    ) -> ForecastResult:
        """
        Calculates the solar forecast using the ML model via the MLPredictor instance,
        iteriert ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ber die stÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ndliche Wettervorhersage.

        Args:
            hourly_weather_forecast: Liste der verarbeiteten stÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ndlichen Wettervorhersagen.
            sensor_data: Dictionary (wird fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r 'current_yield' Mindest-Check benÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¶tigt).
            lag_features: Dictionary mit Lag-Features (z.B. 'production_yesterday').
            correction_factor: Wird von dieser Strategie ignoriert.

        Returns:
            A ForecastResult object containing the prediction details.

        Raises:
            MLModelException: If the ML prediction fails or the predictor is unavailable/unhealthy.
            Exception: For other unexpected errors.
        """
        _LOGGER.debug("Attempting forecast calculation using ML (Iterative) strategy...")

        if not self.is_available():
            _LOGGER.warning("ML Predictor is unavailable or unhealthy. Cannot use ML strategy.")
            raise MLModelException("ML Predictor not available or unhealthy")

        try:
            total_today_kwh = 0.0
            total_tomorrow_kwh = 0.0
            
            now_local = dt_util.as_local(dt_util.utcnow())
            today_date = now_local.date()
            tomorrow_date = today_date + timedelta(days=1)

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
                    
                    # CRITICAL: Skip hours outside realistic production time
                    # This prevents ML from predicting production during darkness
                    from ..forecast.orchestrator import ForecastOrchestrator
                    # We need access to hass for sun.sun check - use simplified check here
                    hour_local = hour_dt_local.hour
                    month = hour_dt_local.month
                    
                    # Conservative production hours by season
                    if month in [11, 12, 1]:  # Winter
                        is_production_hour = 7 <= hour_local <= 16
                    elif month in [5, 6, 7, 8]:  # Summer
                        is_production_hour = 5 <= hour_local <= 20
                    else:  # Spring/Fall
                        is_production_hour = 6 <= hour_local <= 18
                    
                    if not is_production_hour:
                        _LOGGER.debug(f"Skipping hour {hour_local} (outside production time)")
                        continue
                        
                    hour_date = hour_dt_local.date()
                    hour_local = hour_dt_local.hour

                    ml_sensor_data_input = lag_features.copy()
                    
                    features = await self.ml_predictor.feature_engineer.extract_features(
                        weather_data=hour_data,
                        sensor_data=ml_sensor_data_input,
                        prediction_hour=hour_local,
                        prediction_date=hour_dt_local
                    )

                    if self.ml_predictor.scaler.is_fitted:
                        features_scaled = self.ml_predictor.scaler.transform_single(features)
                    else:
                        features_scaled = features
            
                    prediction_result = await self.ml_predictor.prediction_orchestrator.predict(features_scaled)

                    hourly_kwh = prediction_result.prediction
                    hourly_kwh = max(self.PREDICTION_MIN_VALUE, min(hourly_kwh, self.HOURLY_PREDICTION_MAX_VALUE))
                    
                    if hour_date == today_date:
                        total_today_kwh += hourly_kwh
                    elif hour_date == tomorrow_date:
                        total_tomorrow_kwh += hourly_kwh
                        
                except Exception as e_inner:
                    _LOGGER.warning(f"ML strategy failed to process hour {hour_data.get('local_hour')}: {e_inner}")
                    continue

            total_tomorrow_kwh *= self.TOMORROW_DISCOUNT_FACTOR
            
            _LOGGER.debug(f"ML (Iterative) iteration complete. Today (raw): {total_today_kwh:.2f} kWh, Tomorrow (raw): {total_tomorrow_kwh:.2f} kWh")

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
                            f"Mindest-Prognose Anpassung (ML): Aktueller Ertrag {current_yield_float:.2f} kWh > "
                            f"UrsprÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ngliche Prognose {total_today_kwh:.2f} kWh. "
                            f"Angepasst auf {adjusted_today_forecast:.2f} kWh."
                        )
                        
                        original_today_forecast = total_today_kwh
                        total_today_kwh = adjusted_today_forecast
                        
                        if original_today_forecast > 0:
                            adjustment_ratio = total_today_kwh / original_today_forecast
                            total_tomorrow_kwh = total_tomorrow_kwh * adjustment_ratio
                            
            except Exception as e:
                _LOGGER.debug(f"Mindest-Prognose Check (ML) konnte nicht durchgefÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼hrt werden: {e}")


            model_accuracy = getattr(self.ml_predictor, 'current_accuracy', 0.0)
            
            confidence_today = max(0.0, min(100.0, (model_accuracy or 0.0) * 100.0))
            confidence_tomorrow = max(0.0, min(100.0, confidence_today * 0.95))
            
            feature_count = len(self.ml_predictor.feature_engineer.feature_names) if self.ml_predictor.feature_engineer else 0

            result = ForecastResult(
                forecast_today=total_today_kwh,
                forecast_tomorrow=total_tomorrow_kwh,
                confidence_today=confidence_today,
                confidence_tomorrow=confidence_tomorrow,
                method="ml_iterative",
                calibrated=True,
                features_used=feature_count,
                model_accuracy=model_accuracy
            )

            accuracy_str = f", model_acc={result.model_accuracy:.3f}" if result.model_accuracy is not None else ""
            _LOGGER.info(
                f"ML (Iterative) Forecast successful: Today={result.forecast_today:.2f} kWh, "
                f"Tomorrow={result.forecast_tomorrow:.2f} kWh, "
                f"Confidence={result.confidence_today:.1f}%, Method='{result.method}'{accuracy_str}"
            )

            return result

        except MLModelException as me: 
             _LOGGER.error(f"ML (Iterative) Forecast failed: {me}")
             if self.error_handler:
                  await self.error_handler.handle_error(me, "ml_prediction_strategy")
             raise
        except Exception as e:
            _LOGGER.error(f"Unexpected error during ML (Iterative) Forecast calculation: {e}", exc_info=True)
            err = MLModelException(f"ML forecast calculation (iterative) failed: {e}")
            if self.error_handler:
                await self.error_handler.handle_error(err, "ml_prediction_strategy")
            raise err from e
