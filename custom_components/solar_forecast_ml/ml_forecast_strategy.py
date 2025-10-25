"""
ML-basierte Forecast-Strategie.
Verwendet MLPredictor für Machine Learning Vorhersagen.
Version 4.8.1 - Encoding Fix von Zara

Copyright (C) 2025 Zara-Toorox
# von Zara
"""
import logging
from typing import Any, Dict, Optional

from .forecast_strategy import ForecastStrategy, ForecastResult

_LOGGER = logging.getLogger(__name__)


class MLForecastStrategy(ForecastStrategy):
    """
    ML-basierte Forecast-Strategie.
    Verwendet trainiertes ML-Model für präzise Vorhersagen.
    # von Zara
    """
    
    def __init__(self, ml_predictor, error_handler=None):
        """
        Initialisiere ML Forecast Strategy.
        
        Args:
            ml_predictor: MLPredictor Instanz
            error_handler: Optional ErrorHandlingService
        # von Zara
        """
        super().__init__("ml_forecast")
        self.ml_predictor = ml_predictor
        self.error_handler = error_handler
        
        # Konstanten für ML-Forecast # von Zara
        self.PREDICTION_MIN_VALUE = 0.0
        self.PREDICTION_MAX_VALUE = 100.0
        self.TOMORROW_DISCOUNT_FACTOR = 0.92  # Tomorrow ist etwas unsicherer # von Zara
        
    def is_available(self) -> bool:
        """
        Prüft ob ML Predictor verfügbar und gesund ist.
        # von Zara
        """
        if not self.ml_predictor:
            return False
        
        try:
            return self.ml_predictor.is_healthy()
        except Exception as e:
            _LOGGER.debug(f"ML Predictor Gesundheits-Check fehlgeschlagen: {e}")
            return False
    
    def get_priority(self) -> int:
        """
        ML hat höchste Priorität wenn verfügbar.
        # von Zara
        """
        return 100  # Höchste Priorität # von Zara
    
    async def calculate_forecast(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        correction_factor: float
    ) -> ForecastResult:
        """
        Berechnet ML-basierten Forecast.
        
        Args:
            weather_data: Wetter-Daten
            sensor_data: Sensor-Daten (muss solar_capacity enthalten)
            correction_factor: Wird von ML intern verwendet
            
        Returns:
            ForecastResult mit ML-Vorhersage
            
        Raises:
            Exception bei ML-Fehler
        # von Zara
        """
        try:
            _LOGGER.debug("🧠 Starte ML-Forecast-Berechnung...")
            
            # Validiere ML Predictor # von Zara
            if not self.is_available():
                raise RuntimeError("ML Predictor nicht verfügbar oder unhealthy")
            
            # Hole ML Prediction # von Zara
            prediction_result = await self.ml_predictor.predict(
                weather_data,
                sensor_data
            )
            
            # Extrahiere Werte # von Zara
            today_forecast = prediction_result.prediction
            
            # Berechne Tomorrow mit Discount # von Zara
            tomorrow_forecast = today_forecast * self.TOMORROW_DISCOUNT_FACTOR
            
            # Apply Bounds # von Zara
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
            
            # Erstelle Result # von Zara
            result = ForecastResult(
                forecast_today=today_forecast,
                forecast_tomorrow=tomorrow_forecast,
                confidence_today=prediction_result.confidence * 100,
                confidence_tomorrow=prediction_result.confidence * 100 * 0.9,  # Tomorrow etwas unsicherer # von Zara
                method=prediction_result.method,
                calibrated=True,
                features_used=prediction_result.features_used,
                model_accuracy=prediction_result.model_accuracy
            )
            
            # Log erfolgreiche Berechnung # von Zara
            self._log_calculation(
                result,
                f"(features={prediction_result.features_used}, accuracy={prediction_result.model_accuracy:.3f})"
            )
            
            # Record Success im Error Handler # von Zara
            if self.error_handler:
                self.error_handler.record_success("ml_prediction")
            
            return result
            
        except Exception as e:
            _LOGGER.error(f"❌ ML Forecast Berechnung fehlgeschlagen: {e}", exc_info=True)
            
            # Record Error im Error Handler # von Zara
            if self.error_handler:
                from .exceptions import ModelException
                await self.error_handler.handle_error(
                    ModelException(f"ML forecast failed: {e}"),
                    "ml_prediction"
                )
            
            raise
