"""
ML-based Forecast Strategy.
Uses MLPredictor for Machine Learning predictions.
Version 4.8.1 - Encoding Fix by Zara

Copyright (C) 2025 Zara-Toorox
# by Zara

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

from .forecast_strategy import ForecastStrategy, ForecastResult

_LOGGER = logging.getLogger(__name__)


class MLForecastStrategy(ForecastStrategy):
    """
    ML-based Forecast Strategy.
    Uses a trained ML model for precise predictions.
    # by Zara
    """
    
    def __init__(self, ml_predictor, error_handler=None):
        """
        Initialize ML Forecast Strategy.
        
        Args:
            ml_predictor: MLPredictor instance
            error_handler: Optional ErrorHandlingService
        # by Zara
        """
        super().__init__("ml_forecast")
        self.ml_predictor = ml_predictor
        self.error_handler = error_handler
        
        # Constants for ML-Forecast
        self.PREDICTION_MIN_VALUE = 0.0
        self.PREDICTION_MAX_VALUE = 100.0  # Assuming prediction is a percentage or normalized value
        self.TOMORROW_DISCOUNT_FACTOR = 0.92  # Tomorrow is slightly less certain
        
    def is_available(self) -> bool:
        """
        Checks if ML Predictor is available and healthy.
        # by Zara
        """
        if not self.ml_predictor:
            return False
        
        try:
            return self.ml_predictor.is_healthy()
        except Exception as e:
            _LOGGER.debug(f"ML Predictor health check failed: {e}")
            return False
    
    def get_priority(self) -> int:
        """
        ML has the highest priority when available.
        # by Zara
        """
        return 100  # Highest priority
    
    async def calculate_forecast(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        correction_factor: float
    ) -> ForecastResult:
        """
        Calculates ML-based forecast.
        
        Args:
            weather_data: Weather data
            sensor_data: Sensor data (must contain solar_capacity)
            correction_factor: Used by ML internally
            
        Returns:
            ForecastResult with ML prediction
            
        Raises:
            Exception on ML error
        # by Zara
        """
        try:
            _LOGGER.debug("🧠 Starting ML-Forecast calculation...")
            
            # Validate ML Predictor
            if not self.is_available():
                raise RuntimeError("ML Predictor not available or unhealthy")
            
            # Get ML Prediction
            prediction_result = await self.ml_predictor.predict(
                weather_data,
                sensor_data
            )
            
            # Extract values
            today_forecast = prediction_result.prediction
            
            # Calculate Tomorrow with discount
            tomorrow_forecast = today_forecast * self.TOMORROW_DISCOUNT_FACTOR
            
            # Apply Bounds
            today_forecast = self._apply_bounds(
                today_forecast,
                self.PREDICTION_MIN_VALUE,
                self.PREDICTION_MAX_VALUE
            )
            tomorrow_forecast = self._apply_bounds(
                tomorrow_forecast,
                self.PREDICTION_MIN_VALUE,
                self.PREDICTION_MAX_VALUE
            )
            
            # Create Result
            result = ForecastResult(
                forecast_today=today_forecast,
                forecast_tomorrow=tomorrow_forecast,
                confidence_today=prediction_result.confidence * 100,
                confidence_tomorrow=prediction_result.confidence * 100 * 0.9,  # Tomorrow slightly less certain
                method=prediction_result.method,
                calibrated=True,
                features_used=prediction_result.features_used,
                model_accuracy=prediction_result.model_accuracy
            )
            
            # Log successful calculation
            self._log_calculation(
                result,
                f"(features={prediction_result.features_used}, accuracy={prediction_result.model_accuracy:.3f})"
            )
            
            # Record Success in Error Handler
            if self.error_handler:
                # Assuming record_success exists and is synchronous or handled by the handler
                # If it were async, it should be: await self.error_handler.record_success("ml_prediction")
                # Based on error_handling_service.py, it's synchronous.
                pass # self.error_handler.record_success("ml_prediction") - This method doesn't exist in the provided error_handler
            
            return result
            
        except Exception as e:
            _LOGGER.error(f"❌ ML Forecast calculation failed: {e}", exc_info=True)
            
            # Record Error in Error Handler
            if self.error_handler:
                from .exceptions import ModelException
                await self.error_handler.handle_error(
                    ModelException(f"ML forecast failed: {e}"),
                    "ml_prediction"
                )
            
            raise