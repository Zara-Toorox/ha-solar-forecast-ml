"""
ML Predictor für die Solar Forecast ML Integration.
- BALANCED UPGRADE v5.0.0: Advanced Features mit Pure NumPy // von Zara
- Polynomial Features & Interaktionen von Zara
- Zeitreihen-Features (Rolling Windows, Lags) von Zara
- Ridge Regression mit optimaler Regularisierung von Zara
- Pattern Recognition & Trend Detection von Basti
- ~30 Features statt 7 more Accuracy von Zara

Machine Learning Herzstück mit Training, Prediction Logic und robustem Error Handling.
Version 4.8.1 - Lazy NumPy Import Fix // von Zara

Copyright (C) 2025 Zara-Toorox

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
import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

# Lazy Import von numpy um Event Loop blockierung zu vermeiden // von Zara
if TYPE_CHECKING:
    import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_call_later

from .const import (
    MIN_TRAINING_DATA_POINTS, MODEL_ACCURACY_THRESHOLD, DATA_VERSION
)
from .data_manager import DataManager
from .ml_types import (
    PredictionRecord, LearnedWeights, HourlyProfile,
    create_default_learned_weights, create_default_hourly_profile,
    validate_prediction_record
)
from .typed_data_adapter import TypedDataAdapter
from .error_handling_service import ErrorHandlingService
from .exceptions import (
    MLModelException, DataIntegrityException, SolarForecastMLException,
    ErrorSeverity, create_context
)

_LOGGER = logging.getLogger(__name__)


class ModelState(Enum):
    """ML Model States."""
    UNINITIALIZED = "uninitialized"
    TRAINING = "training"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass
class TrainingResult:
    """Ergebnis eines Training Runs."""
    success: bool
    accuracy: float
    samples_used: int
    weights: Optional[LearnedWeights]
    error_message: Optional[str] = None
    training_time_seconds: Optional[float] = None
    feature_count: Optional[int] = None  # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ NEU: Anzahl verwendeter Features // von Zara


@dataclass
class PredictionResult:
    """Ergebnis einer ML Prediction."""
    prediction: float
    confidence: float
    method: str  # 'ml_model', 'hourly_profile', 'simple_fallback' // von Zara
    features_used: Dict[str, float]
    model_accuracy: Optional[float] = None


@dataclass
class ModelHealth:
    """Health Status des ML Models."""
    state: ModelState
    model_loaded: bool
    last_training: Optional[datetime]
    current_accuracy: float
    training_samples: int
    features_available: List[str]
    performance_metrics: Dict[str, float]


# Globale numpy Referenz fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Lazy Loading // von Zara
_np = None


def _ensure_numpy():
    """
    Lazy Import von numpy beim ersten Aufruf.
    Verhindert Event Loop Blockierung beim Modul-Load // von Zara
    """
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
        except ImportError as e:
            _LOGGER.error("NumPy konnte nicht importiert werden: %s", e)
            raise
    return _np


class MLPredictor:
    """
    Machine Learning Predictor mit Advanced Feature Engineering (Pure NumPy).
    ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ BALANCED UPGRADE v4.8.0: Polynomial + Zeitreihen + Ridge Regression // von Zara
    """
    
    def __init__(
        self, 
        hass: HomeAssistant, 
        data_manager: DataManager, 
        error_handler: ErrorHandlingService
    ):
        self.hass = hass
        self.data_manager = data_manager
        self.error_handler = error_handler
        
        # TypedDataAdapter fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Datenkonvertierung // von Zara
        self.data_adapter = TypedDataAdapter()
        
        # Model State // von Zara
        self.model_state = ModelState.UNINITIALIZED
        self.model_loaded = False
        self.current_weights: Optional[LearnedWeights] = None
        self.current_profile: Optional[HourlyProfile] = None
        
        # Performance Tracking // von Zara
        self.current_accuracy = 0.0
        self.training_samples = 0
        self.last_training_time: Optional[datetime] = None
        self.prediction_count = 0
        self.successful_predictions = 0
        
        # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ ERWEITERT: Feature Engineering mit ~30 Features // von Zara
        self.base_features = [
            "temperature", "humidity", "cloudiness", "wind_speed", 
            "hour_of_day", "seasonal_factor", "weather_trend"
        ]
        
        # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ NEU: Polynomial Features // von Zara
        self.polynomial_features = [
            "temperature_sq", "cloudiness_sq", "hour_of_day_sq",
            "seasonal_factor_sq"
        ]
        
        # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ NEU: Interaktions-Features // von Zara
        self.interaction_features = [
            "cloudiness_x_hour", "temperature_x_seasonal", 
            "humidity_x_cloudiness", "wind_x_hour",
            "weather_trend_x_seasonal"
        ]
        
        # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ NEU: Zeitreihen-Features (werden bei VerfÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼gbarkeit hinzugefÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼gt) // von Zara
        self.timeseries_features = [
            "production_yesterday", "production_last_week",
            "rolling_mean_3d", "rolling_std_3d",
            "rolling_mean_7d", "rolling_std_7d",
            "production_trend", "weather_stability",
            "hour_efficiency", "seasonal_deviation"
        ]
        
        # Kombinierte Feature-Liste // von Zara
        self.feature_names = (
            self.base_features + 
            self.polynomial_features + 
            self.interaction_features
        )
        
        # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ NEU: Historical Data Cache fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Zeitreihen-Features // von Zara
        self.historical_cache = {
            'production_history': [],  # Letzte 30 Tage
            'weather_history': [],     # Letzte 30 Tage
            'last_update': None
        }
        
        # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ NEU: Ridge Regularisierung Parameter // von Zara
        self.best_lambda = 0.1  # Wird durch Cross-Validation optimiert
        
        # Health Monitoring // von Zara
        self.performance_metrics = {
            "avg_prediction_time": 0.0,
            "memory_usage_mb": 0.0,
            "error_rate": 0.0,
            "feature_count": len(self.feature_names)  # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ NEU // von Zara
        }
    
    async def initialize(self) -> bool:
        """Initialisiert ML Predictor mit Advanced Features."""
        try:
            _LOGGER.info("ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™...ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  Initialisiere ML Predictor v4.8.0 (Balanced Upgrade)...")
            
            # Lade Historical Cache // von Zara
            await self._load_historical_cache()
            
            # Lade Model State // von Zara
            model_loaded = await self.load_model()
            
            if model_loaded:
                _LOGGER.info(f"ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ ML Model geladen ({len(self.feature_names)} Features)")
                self.model_state = ModelState.READY
            else:
                _LOGGER.info("ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â Kein trained Model, Training erforderlich")
                self.model_state = ModelState.UNINITIALIZED
                
                # Initial Training verzögern um Startup nicht zu blockieren - von Zara
                async_call_later(self.hass, 60, self._delayed_initial_training)
            # VerzÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¶gerter Start des Periodic Training Tasks (60s nach Startup) // von Zara
            async_call_later(self.hass, 60, self._start_periodic_training)
            
            # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ LÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“SUNG 2: Starte Hourly Learning Task (5 Min nach Startup) // von Zara
            async_call_later(self.hass, 300, self._start_hourly_learning_task)
            
            _LOGGER.info("ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ ML Predictor initialisiert mit Advanced Feature Engineering & Hourly Learning")
            return True
            
        except Exception as e:
            _LOGGER.error(f"ML Predictor Initialisierung fehlgeschlagen: {e}")
            await self.error_handler.handle_error(
                MLModelException(f"Initialization failed: {e}"),
                "ml_predictor"
            )
            return False
    
    def set_entities(self, power_entity: Optional[str] = None, solar_yield_today: Optional[str] = None, weather_entity: Optional[str] = None, solar_capacity: float = 5.0, forecast_cache: Optional[Dict] = None) -> None:
        """
        Setzt Entities fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Datensammlung.
        Wird vom Coordinator aufgerufen // von Zara
        """
        self.power_entity = power_entity
        self.solar_yield_today = solar_yield_today  # Tagesertrag in kWh - von Zara
        self.weather_entity = weather_entity
        self.solar_capacity = solar_capacity
        self._forecast_cache = forecast_cache if forecast_cache else {}
        _LOGGER.debug(f"ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ Entities gesetzt: power={power_entity}, yield_today={solar_yield_today}, weather={weather_entity}")
    
    async def _load_historical_cache(self) -> None:
        """
        NEU: historische Daten Zeitreihen-Features // von Zara
        die letzten 30 Tage Produktions- und Wetterdaten
        """
        try:
            # Lade Prediction History // von Zara
            history_data = await self.data_manager.get_prediction_history()
            records = history_data.get('predictions', [])
            
            # Filter letzte 30 Tage // von Zara
            cutoff_date = datetime.now() - timedelta(days=30)
            recent_records = [
                r for r in records 
                if datetime.fromisoformat(r['timestamp']) > cutoff_date
            ]
            
            # Sortiere nach Timestamp // von Zara
            recent_records.sort(key=lambda x: x['timestamp'])
            
            # Speichere in Cache // von Zara
            self.historical_cache['production_history'] = [
                {
                    'timestamp': r['timestamp'],
                    'actual': r.get('actual_value', 0.0),
                    'hour': datetime.fromisoformat(r['timestamp']).hour
                }
                for r in recent_records if r.get('actual_value') is not None
            ]
            
            self.historical_cache['weather_history'] = [
                {
                    'timestamp': r['timestamp'],
                    'cloudiness': r.get('weather_data', {}).get('cloudiness', 50.0),
                    'temperature': r.get('weather_data', {}).get('temperature', 15.0)
                }
                for r in recent_records
            ]
            
            self.historical_cache['last_update'] = datetime.now()
            
            _LOGGER.info(
                f"ÃƒHistorical Cache geladen: "
                f"{len(self.historical_cache['production_history'])} Produktions-Records"
            )
            
        except Exception as e:
            _LOGGER.warning(f"Historical Cache Loading fehlgeschlagen: {e}")
            # Nicht kritisch, System funktioniert ohne Zeitreihen-Features // von Zara

    async def _delayed_initial_training(self, _now=None) -> None:
        """
        Verzögertes Initial Training nach Startup - von Zara
        Verhindert Blocking während Home Assistant Startup
        """
        try:
            _LOGGER.info("Starte verzögertes Initial Training...")
            training_data_available = await self._check_training_data_availability()
            if training_data_available:
                result = await self.train_model()
                if result.success:
                    _LOGGER.info("Initial Training erfolgreich abgeschlossen")
                else:
                    _LOGGER.warning("Initial Training fehlgeschlagen")
            else:
                _LOGGER.info("Nicht genug Daten für Initial Training")
        except Exception as e:
            _LOGGER.error(f"Delayed Initial Training fehlgeschlagen: {e}")
    
    async def _start_periodic_training(self, _now=None) -> None:
        """
        Startet den periodischen Training Task verzögert’nach Startup // von Zara
        Callback  async_call_later - verhindert Startup-Blockierung // von Zara
        """
        _LOGGER.info("Starte periodischen Training Task...")
        self.hass.async_create_task(self._periodic_training_task())
    
    async def load_model(self) -> bool:
        """
        trained Model aus Data Manager.
        REFACTORED: Verwendet TypedDataAdapter Dict->Dataclass Konvertierung // von Zara
        """
        try:
            # Lade Weights als Dict // von Zara
            weights_dict = await self.data_manager.get_learned_weights()
            
            # KRITISCH: Konvertiere Dict zu LearnedWeights Dataclass // von Zara
            self.current_weights = self.data_adapter.dict_to_learned_weights(weights_dict)
            
            # Lade Hourly Profile als Dict // von Zara
            profile_dict = await self.data_manager.get_hourly_profile()
            
            # KRITISCH: Konvertiere Dict zu HourlyProfile Dataclass // von Zara
            self.current_profile = self.data_adapter.dict_to_hourly_profile(profile_dict)
            
            if self.current_weights and self.current_weights.training_samples > 0:
                self.current_accuracy = self.current_weights.accuracy
                self.training_samples = self.current_weights.training_samples
                self.last_training_time = datetime.fromisoformat(self.current_weights.last_trained)
                self.model_loaded = True
                
                _LOGGER.info(
                    f"Model geladen: Accuracy {self.current_accuracy:.2f}, "
                    f"Samples: {self.training_samples}"
                )
                return True
            else:
                _LOGGER.info("Kein trained Model verfügbar")
                self.model_loaded = False
                return False
                
        except Exception as e:
            _LOGGER.error("Model Loading Error: %s", str(e))
            await self.error_handler.handle_error(
                MLModelException(f"Model loading failed: {e}"),
                "ml_predictor"
            )
            self.model_loaded = False
            return False

    async def predict(
        self, 
        weather_data: Dict[str, Any], 
        sensor_data: Dict[str, Any]
    ) -> PredictionResult:
        """
        Generiert ML-basierte Solar Production Prediction mit Advanced Features.
        UPGRADED: Nutzt erweiterte Feature-Extraktion mit Zeitreihen // von Zara
        """
        prediction_start_time = asyncio.get_event_loop().time()
        self.prediction_count += 1
        
        try:
            # Extract features with advanced engineering // von Zara
            features = await self._extract_features_advanced(weather_data, sensor_data)
            
            # Attempt ML Model Prediction // von Zara
            if self.model_loaded and self.current_weights:
                try:
                    prediction = self._ml_predict(features)
                    confidence = self._calculate_confidence(features)
                    
                    self.successful_predictions += 1
                    self._update_performance_metrics(
                        asyncio.get_event_loop().time() - prediction_start_time
                    )
                    
                    return PredictionResult(
                        prediction=prediction,
                        confidence=confidence,
                        method="ml_model_advanced",  #  // von Zara
                        features_used=features,
                        model_accuracy=self.current_accuracy
                    )
                    
                except Exception as ml_error:
                    _LOGGER.warning("ML prediction failed, falling back: %s", str(ml_error))
                    self.model_state = ModelState.DEGRADED
            
            # Fallback 1: Hourly Profile-based Prediction // von Zara
            if self.current_profile:
                try:
                    prediction = self._profile_predict(features)
                    confidence = 0.6  # Lower confidence for profile-based // von Zara
                    
                    return PredictionResult(
                        prediction=prediction,
                        confidence=confidence,
                        method="hourly_profile",
                        features_used=features,
                        model_accuracy=None
                    )
                    
                except Exception as profile_error:
                    _LOGGER.warning("Profile prediction failed: %s", str(profile_error))
            
            # Fallback 2: Simple Heuristic // von Zara
            prediction = self._simple_fallback_predict(features)
            confidence = 0.3  # Lowest confidence for simple fallback // von Zara
            
            return PredictionResult(
                prediction=prediction,
                confidence=confidence,
                method="simple_fallback",
                features_used=features,
                model_accuracy=None
            )
            
        except Exception as e:
            _LOGGER.error("All prediction methods failed: %s", str(e))
            await self.error_handler.handle_error(
                MLModelException(f"Prediction completely failed: {e}"),
                "ml_predictor"
            )
            
            # Ultimate fallback: zero prediction // von Zara
            return PredictionResult(
                prediction=0.0,
                confidence=0.0,
                method="error_fallback",
                features_used={},
                model_accuracy=None
            )
    
    async def _extract_features_advanced(
        self, 
        weather_data: Dict[str, Any], 
        sensor_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        UPGRADED: Erweiterte Feature-Extraktion mit Polynomial, Interaktionen und Zeitreihen // von Zara
        Generiert ~30 Features aus 7 Basis-Features
        """
        now = datetime.now()
        
        # === BASIS-FEATURES === // von Zara
        temperature = self._safe_extract(weather_data, 'temperature', 15.0)
        humidity = self._safe_extract(weather_data, 'humidity', 50.0)
        cloudiness = self._safe_extract(weather_data, 'cloudiness', 50.0)
        wind_speed = self._safe_extract(weather_data, 'wind_speed', 5.0)
        
        # Temporal features // von Zara
        hour_of_day = now.hour + now.minute / 60.0
        day_of_year = now.timetuple().tm_yday
        
        # Seasonal factor (sinusoidal) // von Zara
        seasonal_factor = 0.5 + 0.5 * math.sin((day_of_year - 80) * 2 * math.pi / 365)
        
        # Weather trend calculation // von Zara
        weather_trend = self._calculate_weather_trend(cloudiness, wind_speed)
        
        features = {
            "temperature": temperature,
            "humidity": humidity,
            "cloudiness": cloudiness,
            "wind_speed": wind_speed,
            "hour_of_day": hour_of_day,
            "seasonal_factor": seasonal_factor,
            "weather_trend": weather_trend
        }
        
        # === POLYNOMIAL FEATURES === // von Zara
        features["temperature_sq"] = temperature ** 2
        features["cloudiness_sq"] = cloudiness ** 2
        features["hour_of_day_sq"] = hour_of_day ** 2
        features["seasonal_factor_sq"] = seasonal_factor ** 2
        
        # === INTERAKTIONS-FEATURES === // von Zara
        features["cloudiness_x_hour"] = cloudiness * hour_of_day
        features["temperature_x_seasonal"] = temperature * seasonal_factor
        features["humidity_x_cloudiness"] = humidity * cloudiness
        features["wind_x_hour"] = wind_speed * hour_of_day
        features["weather_trend_x_seasonal"] = weather_trend * seasonal_factor
        
        # === ZEITREIHEN-FEATURES (wenn verfügbar=== // von Zara
        timeseries_features = await self._extract_timeseries_features(now)
        features.update(timeseries_features)
        
        return features
    
    async def _extract_timeseries_features(self, current_time: datetime) -> Dict[str, float]:
        """
        NEU: Extrahiert Zeitreihen-basierte Features aus Historical Cache // von Zara
        Erzeugt Rolling Windows, Lags, Trends und Pattern Recognition Features
        """
        # Lazy Import von numpy // von Zara
        np = _ensure_numpy()
        
        features = {}
        
        try:
            prod_history = self.historical_cache.get('production_history', [])
            
            if len(prod_history) < 7:
                # Nicht genug Daten für Zeitreihen-Features // von Zara
                return self._get_default_timeseries_features()
            
            # Current hour gleiche-Stunde Vergleiche // von Zara
            current_hour = current_time.hour
            
            # === LAG FEATURES === // von Zara
            # Gestern gleiche Stunde // von Zara
            yesterday_same_hour = self._get_production_at(
                prod_history, current_time - timedelta(days=1), current_hour
            )
            features["production_yesterday"] = yesterday_same_hour
            
            # Letzte Woche gleiche Stunde // von Zara
            lastweek_same_hour = self._get_production_at(
                prod_history, current_time - timedelta(days=7), current_hour
            )
            features["production_last_week"] = lastweek_same_hour
            
            # === ROLLING WINDOW FEATURES === // von Zara
            recent_values = [r['actual'] for r in prod_history[-72:]]  # Letzte 3 Tage
            
            if len(recent_values) >= 24:
                features["rolling_mean_3d"] = np.mean(recent_values)
                features["rolling_std_3d"] = np.std(recent_values)
            else:
                features["rolling_mean_3d"] = yesterday_same_hour
                features["rolling_std_3d"] = 0.0
            
            if len(prod_history) >= 168:  # 7 Tage * 24h
                week_values = [r['actual'] for r in prod_history[-168:]]
                features["rolling_mean_7d"] = np.mean(week_values)
                features["rolling_std_7d"] = np.std(week_values)
            else:
                features["rolling_mean_7d"] = features["rolling_mean_3d"]
                features["rolling_std_7d"] = features["rolling_std_3d"]
            
            # === TREND DETECTION === // von Zara
            if len(recent_values) >= 48:
                # Linear Trend letzte 48h // von Zara
                x = np.arange(len(recent_values))
                trend_coef = np.polyfit(x, recent_values, 1)[0]
                features["production_trend"] = trend_coef
            else:
                features["production_trend"] = 0.0
            
            # === WEATHER STABILITY === // von Zara
            weather_history = self.historical_cache.get('weather_history', [])
            if len(weather_history) >= 24:
                recent_cloudiness = [w['cloudiness'] for w in weather_history[-24:]]
                cloudiness_std = np.std(recent_cloudiness)
                # Niedrige Std = stabil = gut vorhersagbar // von Zara
                features["weather_stability"] = max(0, 1.0 - (cloudiness_std / 50.0))
            else:
                features["weather_stability"] = 0.5
            
            # === HOUR EFFICIENCY === // von Zara
            # Durchschnittliche Produktion diese Stunde letzte 7 Tage // von Zara
            hour_productions = [
                r['actual'] for r in prod_history 
                if r['hour'] == current_hour
            ][-7:]  # Max 7 Tage
            
            if hour_productions:
                features["hour_efficiency"] = np.mean(hour_productions)
            else:
                features["hour_efficiency"] = yesterday_same_hour
            
            # === SEASONAL DEVIATION === // von Zara
            # Abweichung von 7-Tage-Durchschnitt // von Zara
            if len(prod_history) >= 168:
                week_mean = features["rolling_mean_7d"]
                if yesterday_same_hour > 0:
                    features["seasonal_deviation"] = (yesterday_same_hour - week_mean) / max(week_mean, 1.0)
                else:
                    features["seasonal_deviation"] = 0.0
            else:
                features["seasonal_deviation"] = 0.0
            
        except Exception as e:
            _LOGGER.warning(f"Zeitreihen-Feature-Extraktion fehlgeschlagen: {e}")
            return self._get_default_timeseries_features()
        
        return features
    
    def _get_production_at(
        self, 
        history: List[Dict], 
        target_time: datetime, 
        target_hour: int
    ) -> float:
        # Lazy Import von numpy // von Zara
        np = _ensure_numpy()
        """
        NEU: Holt Produktionswert bestimmte Zeit aus History // von Zara
        """
        # Suche Record der nahe an target_time und target_hour ist // von Zara
        for record in reversed(history):
            rec_time = datetime.fromisoformat(record['timestamp'])
            if (rec_time.date() == target_time.date() and 
                abs(rec_time.hour - target_hour) <= 1):
                return record['actual']
        
        # Fallback: Durchschnitt der Stunde aus allen  Daten // von Zara
        hour_values = [r['actual'] for r in history if r['hour'] == target_hour]
        return np.mean(hour_values) if hour_values else 0.0
    
    def _get_default_timeseries_features(self) -> Dict[str, float]:
        """
        NEU: Default-Werte wenn nicht genug historische Daten // von Zara
        """
        return {
            "production_yesterday": 0.0,
            "production_last_week": 0.0,
            "rolling_mean_3d": 0.0,
            "rolling_std_3d": 0.0,
            "rolling_mean_7d": 0.0,
            "rolling_std_7d": 0.0,
            "production_trend": 0.0,
            "weather_stability": 0.5,
            "hour_efficiency": 0.0,
            "seasonal_deviation": 0.0
        }
    
    def _safe_extract(
        self, 
        data: Dict[str, Any], 
        key: str, 
        default: float
    ) -> float:
        """Sichere Wert-Extraktion mit Fallback."""
        try:
            value = data.get(key, default)
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _calculate_weather_trend(self, cloudiness: float, wind_speed: float) -> float:
        """
        Berechnet Weather Trend Score.
        Werte = bessere Solar Bedingungen // von Zara
        """
        cloud_score = (100 - cloudiness) / 100.0
        wind_factor = 1.0 - min(wind_speed / 30.0, 1.0)
        return (cloud_score * 0.7 + wind_factor * 0.3)
    
    def _ml_predict(self, features: Dict[str, float]) -> float:
        """
        ML Model-basierte Prediction mit erweiterten Features.
        UPGRADED: Unterstützt jetzt ~30 Features // von Zara
        """
        if not self.current_weights:
            raise MLModelException("No trained weights available")
        
        # Calculate weighted sum // von Zara
        prediction = self.current_weights.bias
        
        for feature_name, feature_value in features.items():
            weight = self.current_weights.weights.get(feature_name, 0.0)
            prediction += weight * feature_value
        
        # Clamp to reasonable range [0, 15000] Wh // von Zara
        return max(0.0, min(prediction, 15000.0))
    
    def _profile_predict(self, features: Dict[str, float]) -> float:
        """
        Hourly Profile-basierte Prediction als Fallback.
        Verwendet historische Hourly Patterns // von Zara
        """
        if not self.current_profile:
            raise MLModelException("No hourly profile available")
        
        hour = int(features.get("hour_of_day", 12))
        base_prediction = self.current_profile.hourly_averages.get(str(hour), 0.0)
        
        # Apply weather adjustments // von Zara
        cloudiness = features.get("cloudiness", 50.0)
        cloud_factor = (100 - cloudiness) / 100.0
        
        seasonal_factor = features.get("seasonal_factor", 0.5)
        
        adjusted_prediction = base_prediction * cloud_factor * (0.5 + seasonal_factor)
        
        return max(0.0, min(adjusted_prediction, 15000.0))
    
    def _simple_fallback_predict(self, features: Dict[str, float]) -> float:
        """
        Simple Heuristic als Ultimate Fallback.
        Basiert auf Tageszeit und Wetter // von Zara
        """
        hour_of_day = features.get("hour_of_day", 12.0)
        cloudiness = features.get("cloudiness", 50.0)
        seasonal_factor = features.get("seasonal_factor", 0.5)
        
        # Simple bell curve for daily production // von Zara
        if hour_of_day < 6 or hour_of_day > 20:
            return 0.0
        
        # Peak at noon // von Zara
        hour_factor = math.sin((hour_of_day - 6) * math.pi / 14)
        
        # Weather adjustment // von Zara
        cloud_factor = (100 - cloudiness) / 100.0
        
        # Base prediction: ~5000W peak under ideal conditions // von Zara
        base_peak = 5000.0
        
        prediction = base_peak * hour_factor * cloud_factor * seasonal_factor
        
        return max(0.0, prediction)
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """
        Berechnet Confidence Score basierend auf Model Accuracy und Feature Quality.
        UPGRADED: jetzt auch Zeitreihen-Features // von Zara
        """
        if not self.current_accuracy:
            return 0.5
        
        base_confidence = self.current_accuracy
        
        # Boost confidence wenn Zeitreihen-Features // von Zara
        if features.get("production_yesterday", 0.0) > 0:
            base_confidence *= 1.1  # +10% Boost
        
        # Penalize bei hoher Wetterinstabilität// von Zara
        weather_stability = features.get("weather_stability", 0.5)
        base_confidence *= (0.8 + 0.2 * weather_stability)
        
        return min(1.0, max(0.0, base_confidence))

    async def train_model(self) -> TrainingResult:
        """
        Trainiert ML Model mit historischen Daten und Advanced Features.
        UPGRADED: Ridge Regression mit optimaler Lambda-Auswahl // von Zara
        """
        training_start_time = asyncio.get_event_loop().time()
        
        try:
            _LOGGER.info("Starte Advanced Model Training (Ridge Regression)...")
            self.model_state = ModelState.TRAINING
            
            # Lade und aktualisiere Historical Cache // von Zara
            await self._load_historical_cache()
            
            # Lade Training Data // von Zara
            history_data = await self.data_manager.get_prediction_history()
            records = history_data.get('predictions', [])
            
            # Filter valid records with actual values // von Zara
            valid_records = [
                r for r in records 
                if r.get('actual_value') is not None and r.get('predicted_value') is not None
            ]
            
            if len(valid_records) < MIN_TRAINING_DATA_POINTS:
                error_msg = (
                    f"Insufficient training data: {len(valid_records)} samples "
                    f"(minimum: {MIN_TRAINING_DATA_POINTS})"
                )
                _LOGGER.warning(error_msg)
                self.model_state = ModelState.UNINITIALIZED
                
                return TrainingResult(
                    success=False,
                    accuracy=0.0,
                    samples_used=len(valid_records),
                    weights=None,
                    error_message=error_msg
                )
            
            # Prepare training data mit Advanced Features // von Zara
            X_train = []
            y_train = []
            
            for record in valid_records:
                weather_data = record.get('weather_data', {})
                sensor_data = record.get('sensor_data', {})
                
                # Nutze vereinfachte Feature-Extraktion (ohne async) // von Zara
                features = self._extract_features_sync(weather_data, sensor_data, record)
                feature_vector = [features.get(name, 0.0) for name in self.feature_names]
                
                X_train.append(feature_vector)
                y_train.append(record['actual_value'])
            
            # NEU: Train Ridge Regression Model mit Lambda-Optimierung // von Zara
            weights_dict, bias, accuracy, best_lambda = await self._train_ridge_model(
                X_train, y_train
            )
            
            self.best_lambda = best_lambda  # Speichere optimales Lambda // von Zara
            
            # Erstelle LearnedWeights Dataclass // von Zara
            learned_weights = LearnedWeights(
                weights=weights_dict,
                bias=bias,
                accuracy=accuracy,
                training_samples=len(valid_records),
                last_trained=datetime.now().isoformat(),
                version=DATA_VERSION
            )
            
            # Konvertiere zu Dict Storage // von Zara
            weights_dict_for_storage = self.data_adapter.learned_weights_to_dict(learned_weights)
            
            # Speichere trained weights // von Zara
            await self.data_manager.save_learned_weights(weights_dict_for_storage)
            
            # Update hourly profile // von Zara
            await self._update_hourly_profile(valid_records)
            
            # Update internal state // von Zara
            self.current_weights = learned_weights
            self.current_accuracy = accuracy
            self.training_samples = len(valid_records)
            self.last_training_time = datetime.now()
            self.model_loaded = True
            self.model_state = ModelState.READY
            
            # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ LÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“SUNG 2: Update model_state.json // von Zara
            await self._update_model_state_after_training(
                accuracy=accuracy,
                samples=len(valid_records),
                training_time=training_time
            )
            
            training_time = asyncio.get_event_loop().time() - training_start_time
            
            _LOGGER.info(
                f"Training erfolgreich: Accuracy={accuracy:.2f}, "
                f"Samples={len(valid_records)}, Features={len(self.feature_names)}, "
                f"Lambda={best_lambda:.4f}, Time={training_time:.2f}s"
            )
            
            return TrainingResult(
                success=True,
                accuracy=accuracy,
                samples_used=len(valid_records),
                weights=learned_weights,
                training_time_seconds=training_time,
                feature_count=len(self.feature_names)
            )
            
        except Exception as e:
            _LOGGER.error("Model training failed: %s", str(e))
            self.model_state = ModelState.ERROR
            
            await self.error_handler.handle_error(
                MLModelException(f"Model training failed: {e}"),
                "ml_predictor"
            )
            
            return TrainingResult(
                success=False,
                accuracy=0.0,
                samples_used=0,
                weights=None,
                error_message=str(e)
            )
    
    def _extract_features_sync(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        record: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        NEU: Synchrone Feature-Extraktion  Training (ohne await) // von Zara
        Vereinfachte Version Batch-Processing Training
        """
        # Extrahiere Timestamp aus Record // von Zara
        timestamp = datetime.fromisoformat(record['timestamp'])
        
        # Basis-Features extrahieren // von Zara
        temperature = self._safe_extract(weather_data, 'temperature', 15.0)
        humidity = self._safe_extract(weather_data, 'humidity', 50.0)
        cloudiness = self._safe_extract(weather_data, 'cloudiness', 50.0)
        wind_speed = self._safe_extract(weather_data, 'wind_speed', 5.0)
        
        hour_of_day = timestamp.hour + timestamp.minute / 60.0
        day_of_year = timestamp.timetuple().tm_yday
        seasonal_factor = 0.5 + 0.5 * math.sin((day_of_year - 80) * 2 * math.pi / 365)
        weather_trend = self._calculate_weather_trend(cloudiness, wind_speed)
        
        features = {
            "temperature": temperature,
            "humidity": humidity,
            "cloudiness": cloudiness,
            "wind_speed": wind_speed,
            "hour_of_day": hour_of_day,
            "seasonal_factor": seasonal_factor,
            "weather_trend": weather_trend
        }
        
        # Polynomial Features // von Zara
        features["temperature_sq"] = temperature ** 2
        features["cloudiness_sq"] = cloudiness ** 2
        features["hour_of_day_sq"] = hour_of_day ** 2
        features["seasonal_factor_sq"] = seasonal_factor ** 2
        
        # Interaktions-Features // von Zara
        features["cloudiness_x_hour"] = cloudiness * hour_of_day
        features["temperature_x_seasonal"] = temperature * seasonal_factor
        features["humidity_x_cloudiness"] = humidity * cloudiness
        features["wind_x_hour"] = wind_speed * hour_of_day
        features["weather_trend_x_seasonal"] = weather_trend * seasonal_factor
        
        # Zeitreihen-Features nicht Training // von Zara
        # Werden bei Prediction mit echtem Historical Cache // von Zara
        
        return features
    
    async def _train_ridge_model(
        self, 
        X: List[List[float]], 
        y: List[float]
    ) -> Tuple[Dict[str, float], float, float, float]:
        """
        NEU: Trainiert Ridge Regression Model mit optimaler Lambda-Auswahl // von Zara
        Verwendet NumPy effiziente Matrix-Operationen und Cross-Validation
        
        Returns: (weights_dict, bias, accuracy, best_lambda)
        """
        try:
            # Lazy Import von numpy // von Zara
            np = _ensure_numpy()
            # Convert to numpy arrays // von Zara
            X_array = np.array(X)
            y_array = np.array(y)
            
            # Add bias term (column of ones) // von Zara
            X_with_bias = np.column_stack([X_array, np.ones(len(X_array))])
            
            # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ NEU: Lambda-Auswahl via Simple Hold-Out Validation // von Zara
            # Teste verschiedene Lambda-Werte // von Zara
            lambda_candidates = [0.001, 0.01, 0.1, 1.0, 10.0]
            best_lambda = 0.1
            best_score = -np.inf
            
            # 80-20 Train-Test Split fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Validation // von Zara
            split_idx = int(0.8 * len(X_array))
            X_train = X_with_bias[:split_idx]
            y_train = y_array[:split_idx]
            X_test = X_with_bias[split_idx:]
            y_test = y_array[split_idx:]
            
            if len(X_test) > 5:  # Nur wenn genug Test-Daten // von Zara
                for lambda_val in lambda_candidates:
                    # Train mit diesem Lambda // von Zara
                    XtX = np.dot(X_train.T, X_train)
                    Xty = np.dot(X_train.T, y_train)
                    
                    # Ridge Regularisierung: (X^T X + ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’...Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»I)^-1 X^T y // von Zara
                    regularization = lambda_val * np.eye(XtX.shape[0])
                    XtX_reg = XtX + regularization
                    
                    try:
                        weights_with_bias = np.linalg.solve(XtX_reg, Xty)
                        
                        # Test Score (RÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²) // von Zara
                        predictions = np.dot(X_test, weights_with_bias)
                        ss_res = np.sum((y_test - predictions) ** 2)
                        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                        
                        if ss_tot > 0:
                            r_squared = 1 - (ss_res / ss_tot)
                            if r_squared > best_score:
                                best_score = r_squared
                                best_lambda = lambda_val
                    except np.linalg.LinAlgError:
                        continue
            
            # Finales Training mit bestem Lambda auf allen Daten // von Zara
            XtX = np.dot(X_with_bias.T, X_with_bias)
            Xty = np.dot(X_with_bias.T, y_array)
            
            # Ridge Regularisierung mit optimal lambda // von Zara
            regularization = best_lambda * np.eye(XtX.shape[0])
            XtX_reg = XtX + regularization
            
            # Solve for weights // von Zara
            weights_with_bias = np.linalg.solve(XtX_reg, Xty)
            
            # Separate feature weights and bias // von Zara
            feature_weights = weights_with_bias[:-1]
            bias = weights_with_bias[-1]
            
            # Create weights dictionary // von Zara
            weights_dict = {}
            feature_names_subset = self.feature_names[:len(feature_weights)]
            for i, feature_name in enumerate(feature_names_subset):
                weights_dict[feature_name] = float(feature_weights[i])
            
            # Calculate final accuracy (R-squared) auf allen Daten // von Zara
            predictions = np.dot(X_with_bias, weights_with_bias)
            ss_res = np.sum((y_array - predictions) ** 2)
            ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
            
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
                accuracy = max(0.0, min(1.0, r_squared))  # Clamp to [0, 1] // von Zara
            else:
                accuracy = 0.0
            
            _LOGGER.info(f"Ridge Training: Lambda={best_lambda}, ‚Â²={accuracy:.4f}")
            
            return weights_dict, float(bias), accuracy, best_lambda
            
        except Exception as e:
            _LOGGER.error(f"Ridge model training failed: {e}")
            raise MLModelException(f"Ridge regression training failed: {e}")
    
    async def _update_hourly_profile(self, records: List[Dict[str, Any]]) -> None:
        """
        Update Hourly Production Profile.
        Berechnet durchschnittliche Produktion pro Stunde // von Zara
        """
        try:
            # Lazy Import von numpy // von Zara
            np = _ensure_numpy()
            # Group records by hour // von Zara
            hourly_data = {}
            for hour in range(24):
                hourly_data[hour] = []
            
            for record in records:
                if record.get('actual_value') is None:
                    continue
                    
                timestamp = datetime.fromisoformat(record['timestamp'])
                hour = timestamp.hour
                hourly_data[hour].append(record['actual_value'])
            
            # Calculate averages // von Zara
            hourly_averages = {}
            for hour, values in hourly_data.items():
                if values:
                    hourly_averages[str(hour)] = float(np.mean(values))
                else:
                    hourly_averages[str(hour)] = 0.0
            
            # Create HourlyProfile // von Zara
            hourly_profile = HourlyProfile(
                hourly_averages=hourly_averages,
                last_updated=datetime.now().isoformat(),
                sample_count=len(records),
                version=DATA_VERSION
            )
            
            # Convert to dict and save // von Zara
            profile_dict = self.data_adapter.hourly_profile_to_dict(hourly_profile)
            await self.data_manager.save_hourly_profile(profile_dict)
            
            self.current_profile = hourly_profile
            
        except Exception as e:
            _LOGGER.error(f"Hourly profile update failed: {e}")
    
    async def _update_model_state_after_training(
        self, 
        accuracy: float, 
        samples: int, 
        training_time: float
    ) -> None:
        """
        ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ LÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“SUNG 2: Aktualisiert model_state.json nach Training.
        Wird nach jedem erfolgreichen Training aufgerufen // von Zara
        """
        try:
            # Lade aktuellen State // von Zara
            current_state = await self.data_manager.get_model_state()
            
            # ZÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤hle Training-Events // von Zara
            training_count = current_state.get('training_count', 0) + 1
            
            # Berechne MAE und RMSE aus letzten Predictions // von Zara
            history_data = await self.data_manager.get_prediction_history()
            records = history_data.get('predictions', [])[-100:]  # Letzte 100
            
            mae = None
            rmse = None
            if len(records) >= 10:
                errors = []
                for r in records:
                    pred = r.get('predicted_value', 0.0)
                    actual = r.get('actual_value', 0.0)
                    if actual is not None:
                        errors.append(abs(pred - actual))
                
                if errors:
                    np = _ensure_numpy()
                    mae = float(np.mean(errors))
                    rmse = float(np.sqrt(np.mean([e**2 for e in errors])))
            
            # Update State // von Zara
            updated_state = {
                "version": DATA_VERSION,
                "created": current_state.get('created', dt_util.utcnow().isoformat()),
                "last_training": dt_util.utcnow().isoformat(),
                "training_count": training_count,
                "performance_metrics": {
                    "mae": mae,
                    "rmse": rmse,
                    "accuracy": accuracy
                },
                "status": "ready",
                "training_samples": samples,
                "training_time_seconds": training_time
            }
            
            # Speichere State // von Zara
            await self.data_manager.save_model_state(updated_state)
            
            _LOGGER.info(
                f"¦ model_state aktualisiert: Training #{training_count}, "
                f"MAE={mae:.2f if mae else 'N/A'}, RMSE={rmse:.2f if rmse else 'N/A'}"
            )
            
        except Exception as e:
            _LOGGER.error(f"model_state update failed: {e}")
    
    async def _check_training_data_availability(self) -> bool:
        """ob genug Daten  Training  sind."""
        try:
            history_data = await self.data_manager.get_prediction_history()
            records = history_data.get('predictions', [])
            
            valid_records = [
                r for r in records 
                if r.get('actual_value') is not None
            ]
            
            return len(valid_records) >= MIN_TRAINING_DATA_POINTS
            
        except Exception as e:
            _LOGGER.error(f"Training data check failed: {e}")
            return False
    
    async def _periodic_training_task(self) -> None:
        """Periodic Model Training Task."""
        while True:
            try:
                # Train daily at 23:00 // von Zara
                now = datetime.now()
                if now.hour == 23 and now.minute < 5:
                    
                    # Check if enough new data is available // von Zara
                    if await self._should_retrain():
                        _LOGGER.info("° Starte Periodic Training...")
                        training_result = await self.train_model()
                        
                        if training_result.success:
                            _LOGGER.info("Periodic Training erfolgreich")
                        else:
                            _LOGGER.warning("Periodic Training fehlgeschlagen")
                
                # Sleep for 5 minutes // von Zara
                await asyncio.sleep(300)
                
            except Exception as e:
                _LOGGER.error("Periodic training error: %s", str(e))
                await asyncio.sleep(300)
    
    async def _should_retrain(self) -> bool:
        """Bestimmt ob Re-Training notwendig ist."""
        try:
            # Retrain if no model loaded // von Zara
            if not self.model_loaded:
                return True
            
            # Retrain if accuracy too low // von Zara
            if self.current_accuracy < MODEL_ACCURACY_THRESHOLD:
                return True
            
            # Retrain if last training is old // von Zara
            if (self.last_training_time and 
                (datetime.now() - self.last_training_time) > timedelta(days=7)):
                return True
            
            # Retrain if significant new data available // von Zara
            history_data = await self.data_manager.get_prediction_history()
            records = history_data.get('predictions', [])
            
            valid_records = [
                r for r in records 
                if r.get('actual_value') is not None
            ]
            
            # Wenn 20% mehr Daten als beim letzten Training // von Zara
            if len(valid_records) > self.training_samples * 1.2:
                return True
            
            return False
            
        except Exception as e:
            _LOGGER.error(f"Should retrain check failed: {e}")
            return False
    
    def _update_performance_metrics(self, prediction_time: float) -> None:
        """Update Performance Metrics."""
        # Update average prediction time // von Zara
        if self.prediction_count > 0:
            alpha = 0.1  # Exponential moving average factor
            self.performance_metrics["avg_prediction_time"] = (
                alpha * prediction_time + 
                (1 - alpha) * self.performance_metrics["avg_prediction_time"]
            )
        else:
            self.performance_metrics["avg_prediction_time"] = prediction_time
        
        # Update error rate // von Zara
        if self.prediction_count > 0:
            self.performance_metrics["error_rate"] = (
                1.0 - (self.successful_predictions / self.prediction_count)
            )

    async def add_training_sample(
        self,
        prediction: float,
        actual: float,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any]
    ) -> None:
        """
        neues Training Sample zur History hinzu.
        """
        try:
            timestamp = datetime.now().isoformat()
            
            sample = {
                "timestamp": timestamp,
                "predicted_value": prediction,
                "actual_value": actual,
                "weather_data": weather_data,
                "sensor_data": sensor_data,
                "accuracy": self._calculate_sample_accuracy(prediction, actual),  # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ FIX // von Zara
                "model_version": ML_MODEL_VERSION  # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™...ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ FIX // von Zara
            }
            
            await self.data_manager.add_prediction_record(sample)
            
            # Update Historical Cache // von Zara
            if len(self.historical_cache.get('production_history', [])) > 0:
                self.historical_cache['production_history'].append({
                    'timestamp': timestamp,
                    'actual': actual,
                    'hour': datetime.now().hour
                })
                
                # Behalte nur letzte 30 Tage // von Zara
                cutoff = (datetime.now() - timedelta(days=30)).isoformat()
                self.historical_cache['production_history'] = [
                    r for r in self.historical_cache['production_history']
                    if r['timestamp'] > cutoff
                ]
            
        except Exception as e:
            _LOGGER.error(f"Add training sample failed: {e}")
    
    async def force_retrain(self) -> TrainingResult:
        """Erzwingt sofortiges Model Retraining."""
        _LOGGER.info("Force Retrain triggered by user")
        return await self.train_model()
    
    async def reset_model(self) -> bool:
        """Reset Model to uninitialized state."""
        try:
            self.model_loaded = False
            self.current_weights = None
            self.current_accuracy = 0.0
            self.training_samples = 0
            self.last_training_time = None
            self.model_state = ModelState.UNINITIALIZED
            
            _LOGGER.info(" Model reset erfolgreich")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Model reset failed: {e}")
            return False
    
    def get_model_health(self) -> ModelHealth:
        """Gibt aktuellen Health Status des Models zurÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ck."""
        return ModelHealth(
            state=self.model_state,
            model_loaded=self.model_loaded,
            last_training=self.last_training_time,
            current_accuracy=self.current_accuracy,
            training_samples=self.training_samples,
            features_available=self.feature_names,
            performance_metrics=self.performance_metrics
        )

    # ========================================================================
    # ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’...ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ LÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œSUNG 2: STÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’...ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œNDLICHE DATENSAMMLUNG UND AUTO-LEARNING // von Zara
    # ========================================================================
    
    async def _start_hourly_learning_task(self, *args) -> None:
        """
        Startet stÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ndliche Learning Task.
        Wird 5 Minuten nach Startup aufgerufen // von Zara
        FIX: Async Methode für korrekte Task-Erstellung // von Zara
        """
        try:
            _LOGGER.info(" Starte Hourly Learning Task...")
            self.hass.async_create_task(self._hourly_learning_task())
            _LOGGER.debug("ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ Hourly Learning Task erfolgreich gestartet")
        except Exception as e:
            _LOGGER.error(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Fehler beim Starten der Hourly Learning Task: {e}", exc_info=True)
    
    async def _hourly_learning_task(self) -> None:
        """
        Task: Sammelt Daten und triggert Auto-Training.
        jede Stunde zur Minute :05 // von Zara
        """
        _LOGGER.info("Hourly Learning Task gestartet")
        
        while True:
            try:
                now = datetime.now()
                
                # FÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼hre nur zur Minute :05 aus (5 Min nach jeder Stunde) // von Zara
                if now.minute == 5:
                    _LOGGER.info(f"Hourly Learning: Sammle Daten fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r {now.hour}:00 Uhr")
                    
                    # Sammle stÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ndliche Daten // von Zara
                    await self._collect_hourly_sample()
                    
                    # PrÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼fe ob Auto-Training nÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¶tig // von Zara
                    if await self._should_auto_train():
                        _LOGGER.info("Triggere Auto-Training...")
                        training_result = await self.train_model()
                        
                        if training_result.success:
                            _LOGGER.info(f"Auto-Training erfolgreich: Accuracy={training_result.accuracy:.2%}")
                        else:
                            _LOGGER.warning(f"Auto-Training fehlgeschlagen: {training_result.error_message}")
                    
                    # Warte 2 Minuten damit wir nicht mehrfach in Minute :05 laufen // von Zara
                    await asyncio.sleep(120)
                
                # PrÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼fe jede Minute // von Zara
                await asyncio.sleep(60)
                
            except Exception as e:
                _LOGGER.error(f"Hourly learning task error: {e}")
                await asyncio.sleep(300)  # 5 Min bei Fehler // von Zara
    
    async def _collect_hourly_sample(self) -> None:
        """
        Sammelt stÃƒÂ¼ndliche Sample-Daten.
        ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“... KORRIGIERT: Vergleicht Vorhersage (Wh) mit Tagesertrag (kWh) - von Zara
        """
        try:
            now = datetime.now()
            current_hour = now.hour
            
            # Hole Tagesertrag-Sensor (kWh) - von Zara
            solar_yield_entity = getattr(self, 'solar_yield_today', None)
            if not solar_yield_entity:
                _LOGGER.debug("Kein solar_yield_today konfiguriert, ÃƒÂ¼berspringe Sample-Collection")
                return
            
            # Lese aktuellen Tagesertrag in kWh - von Zara
            yield_state = self.hass.states.get(solar_yield_entity)
            if not yield_state or yield_state.state in ['unavailable', 'unknown']:
                _LOGGER.debug(f"Solar yield sensor {solar_yield_entity} nicht verfÃƒÂ¼gbar")
                return
            
            try:
                actual_value_kwh = float(yield_state.state)  # kWh - von Zara
            except (ValueError, TypeError):
                _LOGGER.warning(f"Ungültiger Wert von {solar_yield_entity}: {yield_state.state}")
                return
            
            # Hole die Vorhersage die wir gemacht haben - von Zara
            predicted_value_wh = self._get_stored_prediction_for_hour(current_hour)
            
            if predicted_value_wh is None:
                # Fallback: Nutze aktuelles Modell für SchÃƒÂ¤tzung - von Zara
                _LOGGER.debug("Keine gespeicherte Vorhersage, nutze Modell-SchÃƒÂ¤tzung")
                
                # Sammle aktuelle Weather-Daten - von Zara
                weather_data = await self._get_current_weather_data()
                sensor_data = {
                    'solar_capacity': getattr(self, 'solar_capacity', 5.0),
                    'power_entity': getattr(self, 'power_entity', None)
                }
                
                # Extrahiere Features - von Zara
                features = await self.extract_features(weather_data, sensor_data)
                
                # Mache Vorhersage - von Zara
                prediction_result = await self.predict(features)
                predicted_value_wh = prediction_result.prediction  # Wh - von Zara
            
            # ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“... KRITISCH: Konvertiere Wh ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ kWh für Vergleich - von Zara
            predicted_value_kwh = predicted_value_wh / 1000.0  # Wh ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ kWh - von Zara
            
            # Erstelle Training Sample - von Zara
            timestamp = now.replace(minute=0, second=0, microsecond=0).isoformat()
            
            # Hole aktuelle Weather-Daten für Sample - von Zara
            weather_data = await self._get_current_weather_data()
            sensor_data = {
                'solar_capacity': getattr(self, 'solar_capacity', 5.0),
                'solar_yield_today': solar_yield_entity  # ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“... KORRIGIERT - von Zara
            }
            
            # ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“... KORRIGIERT: Beide Werte in kWh speichern - von Zara
            sample = {
                "timestamp": timestamp,
                "predicted_value": predicted_value_kwh,  # kWh - von Zara
                "actual_value": actual_value_kwh,        # kWh - von Zara
                "weather_data": weather_data,
                "sensor_data": sensor_data,
                "accuracy": self._calculate_sample_accuracy(predicted_value_kwh, actual_value_kwh),
                "model_version": ML_MODEL_VERSION
            }
            
            # Speichere in prediction_history - von Zara
            await self.data_manager.add_prediction_record(sample)
            
            # ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“... KORRIGIERT: Log mit kWh - von Zara
            _LOGGER.info(
                f"... Sample gespeichert: {current_hour}:00 Uhr | "
                f"Predicted={predicted_value_kwh:.2f}kWh, Actual={actual_value_kwh:.2f}kWh, "
                f"Accuracy={sample['accuracy']:.1%}"
            )
            
        except Exception as e:
            _LOGGER.error(f"Fehler beim Sammeln des hourly samples: {e}")
    def _get_stored_prediction_for_hour(self, hour: int) -> Optional[float]:
        """
        Holt gespeicherte Vorhersage fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r bestimmte Stunde.
        Nutzt _forecast_cache falls vorhanden // von Zara
        """
        try:
            # PrÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼fe ob wir forecast_cache haben (vom Coordinator) // von Zara
            if hasattr(self, '_forecast_cache') and self._forecast_cache:
                # Suche nach Vorhersage fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r diese Stunde // von Zara
                hourly_forecast = self._forecast_cache.get('hourly_forecast', [])
                for forecast_hour in hourly_forecast:
                    if forecast_hour.get('hour') == hour:
                        return forecast_hour.get('production', 0.0)
            
            return None
            
        except Exception as e:
            _LOGGER.debug(f"Fehler beim Abrufen gespeicherter Vorhersage: {e}")
            return None
    
    async def _get_current_weather_data(self) -> Dict[str, Any]:
        """
        Holt aktuelle Wetterdaten.
        Nutzt weather_entity vom System // von Zara
        """
        try:
            weather_entity = getattr(self, 'weather_entity', None)
            if not weather_entity:
                # Fallback zu Default-Werten // von Zara
                return {
                    'temperature': 15.0,
                    'humidity': 60.0,
                    'cloudiness': 50.0,
                    'wind_speed': 5.0,
                    'pressure': 1013.0
                }
            
            weather_state = self.hass.states.get(weather_entity)
            if not weather_state:
                return {
                    'temperature': 15.0,
                    'humidity': 60.0,
                    'cloudiness': 50.0,
                    'wind_speed': 5.0,
                    'pressure': 1013.0
                }
            
            # Extrahiere Wetter-Attribute // von Zara
            attrs = weather_state.attributes
            
            return {
                'temperature': float(attrs.get('temperature', 15.0)),
                'humidity': float(attrs.get('humidity', 60.0)),
                'cloudiness': float(attrs.get('cloud_coverage', 50.0)),
                'wind_speed': float(attrs.get('wind_speed', 5.0)),
                'pressure': float(attrs.get('pressure', 1013.0))
            }
            
        except Exception as e:
            _LOGGER.debug(f"Fehler beim Abrufen von Weather-Daten: {e}")
            # Fallback zu Default-Werten // von Zara
            return {
                'temperature': 15.0,
                'humidity': 60.0,
                'cloudiness': 50.0,
                'wind_speed': 5.0,
                'pressure': 1013.0
            }
    
    def _calculate_sample_accuracy(self, predicted: float, actual: float) -> float:
        """
        Berechnet Accuracy fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r ein einzelnes Sample.
        Returns: 0.0 - 1.0 // von Zara
        """
        try:
            if predicted == 0.0 and actual == 0.0:
                return 1.0  # Perfekt wenn beide 0 // von Zara
            
            if predicted == 0.0 or actual == 0.0:
                return 0.0  # Einer ist 0, der andere nicht // von Zara
            
            # Berechne relative Abweichung // von Zara
            error = abs(predicted - actual) / max(predicted, actual)
            accuracy = max(0.0, 1.0 - error)
            
            return accuracy
            
        except Exception:
            return 0.5  # Fallback // von Zara
    
    async def _should_auto_train(self) -> bool:
        """
        PrÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ft ob Auto-Training ausgelÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¶st werden soll.
        Training bei 50+ neuen Samples oder 24h seit letztem Training // von Zara
        """
        try:
            # Lade aktuelle History // von Zara
            history_data = await self.data_manager.get_prediction_history()
            records = history_data.get('predictions', [])
            
            # ZÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤hle Samples seit letztem Training // von Zara
            if self.last_training_time:
                new_samples = [
                    r for r in records
                    if r.get('timestamp', '') > self.last_training_time.isoformat()
                ]
                new_sample_count = len(new_samples)
            else:
                new_sample_count = len(records)
            
            # Training bei 50+ neuen Samples // von Zara
            if new_sample_count >= 50:
                _LOGGER.info(f" Auto-Training: {new_sample_count} neue Samples verfügbar")
                return True
            
            # Training wenn 24h seit letztem Training // von Zara
            if self.last_training_time:
                hours_since_training = (datetime.now() - self.last_training_time).total_seconds() / 3600
                if hours_since_training >= 24:
                    _LOGGER.info(f"Auto-Training: {hours_since_training:.1f}h seit letztem Training")
                    return True
            else:
                # Noch nie trainiert, aber mindestens MIN_TRAINING_DATA_POINTS vorhanden // von Zara
                if len(records) >= MIN_TRAINING_DATA_POINTS:
                    _LOGGER.info(f"Auto-Training: {len(records)} Samples vorhanden, noch nie trainiert")
                    return True
            
            return False
            
        except Exception as e:
            _LOGGER.error(f"Fehler bei _should_auto_train: {e}")
            return False

    def is_healthy(self) -> bool:
        """
        Prueft ob ML Predictor funktionsfaehig ist.
        Verwendet fuer Coordinator Health Checks.

        """
        try:
            # Pruefe ob Model initialisiert ist
            if not self.model_loaded:
                return False
            
            # Pruefe ob Training-Daten vorhanden sind
            if self.training_samples < 3:
                return False
            
            # Pruefe ob Model State gueltig ist
            if self.model_state not in [ModelState.READY, ModelState.TRAINED]:
                return False
            
            # Pruefe ob aktuelle Accuracy akzeptabel ist
            if self.current_accuracy < 0.5:
                return False
            
            return True
            
        except Exception:
            return False
