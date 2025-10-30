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
from typing import Any, Dict, Optional

# Import base strategy class and result dataclass
from .forecast_strategy import ForecastStrategy, ForecastResult
# Import error handling components if needed
from .error_handling_service import ErrorHandlingService
# --- HIER DIE KORREKTUR (Import) ---
from .exceptions import MLModelException # Use MLModelException instead
# --- ENDE KORREKTUR ---

# Note: MLPredictor class itself is not imported here, only an instance is passed in.

_LOGGER = logging.getLogger(__name__)


class MLForecastStrategy(ForecastStrategy):
    """
    Implements the forecast strategy using a trained Machine Learning model
    provided by an MLPredictor instance.
    """

    def __init__(self, ml_predictor: Any, error_handler: Optional[ErrorHandlingService] = None):
        """
        Initialize the ML Forecast Strategy.

        Args:
            ml_predictor: An instance of the MLPredictor class (or compatible).
            error_handler: Optional ErrorHandlingService instance for logging errors.
        """
        super().__init__("ml_forecast") # Call base class constructor with strategy name
        self.ml_predictor = ml_predictor
        self.error_handler = error_handler

        # Constants specific to ML-Forecast interpretation (adjust as needed)
        self.PREDICTION_MIN_VALUE = 0.0 # Minimum possible forecast value (kWh)
        self.PREDICTION_MAX_VALUE = 100.0 # Example: Max expected daily kWh
        self.TOMORROW_DISCOUNT_FACTOR = 0.92 # Apply slight discount for tomorrow's uncertainty

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
            return False # Unavailability on error

    def get_priority(self) -> int:
        """
        Returns the priority of this strategy. ML has the highest priority.
        """
        return 100 # Highest priority, preferred over rule-based

    async def calculate_forecast(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        correction_factor: float
    ) -> ForecastResult:
        """
        Calculates the solar forecast using the ML model via the MLPredictor instance.

        Args:
            weather_data: Dictionary containing current weather data.
            sensor_data: Dictionary containing current sensor data (like capacity, external sensors).
            correction_factor: Learned correction factor (might be used internally by predictor or ignored).

        Returns:
            A ForecastResult object containing the prediction details.

        Raises:
            MLModelException: If the ML prediction fails or the predictor is unavailable/unhealthy.
            Exception: For other unexpected errors.
        """
        _LOGGER.debug("Attempting forecast calculation using ML strategy...")

        if not self.is_available():
            _LOGGER.warning("ML Predictor is unavailable or unhealthy. Cannot use ML strategy.")
            # --- HIER DIE KORREKTUR (Raise) ---
            raise MLModelException("ML Predictor not available or unhealthy") # Use correct class
            # --- ENDE KORREKTUR ---

        try:
            prediction_result = await self.ml_predictor.predict(
                weather_data,
                sensor_data
            )

            raw_today_forecast = prediction_result.prediction
            raw_tomorrow_forecast = raw_today_forecast * self.TOMORROW_DISCOUNT_FACTOR

            today_forecast_kwh = max(self.PREDICTION_MIN_VALUE, min(raw_today_forecast, self.PREDICTION_MAX_VALUE))
            tomorrow_forecast_kwh = max(self.PREDICTION_MIN_VALUE, min(raw_tomorrow_forecast, self.PREDICTION_MAX_VALUE))

            confidence_percent = prediction_result.confidence * 100
            confidence_today = max(0.0, min(100.0, confidence_percent))
            confidence_tomorrow = max(0.0, min(100.0, confidence_today * 0.95))

            result = ForecastResult(
                forecast_today=today_forecast_kwh,
                forecast_tomorrow=tomorrow_forecast_kwh,
                confidence_today=confidence_today,
                confidence_tomorrow=confidence_tomorrow,
                method=prediction_result.method,
                calibrated=True,
                features_used=len(prediction_result.features_used) if prediction_result.features_used else None,
                model_accuracy=prediction_result.model_accuracy
            )

            accuracy_str = f", model_acc={result.model_accuracy:.3f}" if result.model_accuracy is not None else ""
            _LOGGER.info(
                f"ML Forecast successful: Today={result.forecast_today:.2f} kWh, "
                f"Tomorrow={result.forecast_tomorrow:.2f} kWh, "
                f"Confidence={result.confidence_today:.1f}%, Method='{result.method}'{accuracy_str}"
            )

            return result

        # --- HIER DIE KORREKTUR (Catch) ---
        except MLModelException as me: # Catch the correct class
        # --- ENDE KORREKTUR ---
             # Re-raise MLModelExceptions directly
             _LOGGER.error(f"ML Forecast failed: {me}")
             if self.error_handler:
                  # Ensure handle_error expects MLModelException
                  await self.error_handler.handle_error(me, "ml_prediction_strategy")
             raise
        except Exception as e:
            # Wrap unexpected errors in MLModelException
            _LOGGER.error(f"Unexpected error during ML Forecast calculation: {e}", exc_info=True)
            # --- HIER DIE KORREKTUR (Wrap) ---
            err = MLModelException(f"ML forecast calculation failed: {e}") # Use correct class
            # --- ENDE KORREKTUR ---
            if self.error_handler:
                await self.error_handler.handle_error(err, "ml_prediction_strategy")
            raise err from e