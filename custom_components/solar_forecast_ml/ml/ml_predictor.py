"""
Machine Learning Predictor

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
from datetime import datetime, timedelta, timezone 
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field 
from enum import Enum

# Lazy import NumPy
# if TYPE_CHECKING:
#     import numpy as np
# _np = None

from homeassistant.core import HomeAssistant, callback 
from homeassistant.helpers.event import async_track_time_change, async_call_later

from ..const import (
    MIN_TRAINING_DATA_POINTS, MODEL_ACCURACY_THRESHOLD, DATA_VERSION,
    ML_MODEL_VERSION,
    CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX
)
from ..data.data_manager import DataManager
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..ml.ml_types import (
    LearnedWeights, HourlyProfile,
    create_default_learned_weights, create_default_hourly_profile
)
from ..data.data_adapter import TypedDataAdapter
from ..services.service_error_handler import ErrorHandlingService
from ..core.core_exceptions import MLModelException, ErrorSeverity, DataIntegrityException

# Lazy import NumPy
_np = None

# Import ML components
from ..ml.ml_scaler import StandardScaler
from ..ml.ml_feature_engineering import FeatureEngineer
from ..ml.ml_feature_engineering_v2 import FeatureEngineerV2
from ..ml.ml_trainer import RidgeTrainer
from ..ml.ml_prediction_strategies import (
    PredictionOrchestrator, PredictionResult,
    MLModelStrategy, ProfileStrategy, FallbackStrategy 
)
from ..ml.ml_sample_collector import SampleCollector

_LOGGER = logging.getLogger(__name__)


# --- Helper for Lazy NumPy Import ---
def _ensure_numpy():
    """Lazily imports and returns the NumPy module raising ImportError if unavailable"""
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
            _LOGGER.info("NumPy library loaded successfully.")
        except ImportError as e:
            _LOGGER.error("NumPy library is required but could not be imported. "
                          "Please ensure it is installed. Error: %s", e)
            raise # Re-raise to signal dependency issue
    return _np


# --- Enums and Dataclasses ---
class ModelState(Enum):
    """Represents the operational state of the ML model"""
    UNINITIALIZED = "uninitialized"
    TRAINING = "training"
    READY = "ready"
    DEGRADED = "degraded" # Model exists but accuracy is low or data is old
    ERROR = "error"


@dataclass
class TrainingResult:
    """Stores the outcome of a model training attempt"""
    success: bool
    accuracy: float | None = None # Accuracy might be None if training failed early
    samples_used: int = 0
    weights: Optional[LearnedWeights] = None
    error_message: Optional[str] = None
    training_time_seconds: Optional[float] = None
    feature_count: Optional[int] = None


@dataclass
class ModelHealth:
    """Represents the health status of the ML model"""
    state: ModelState
    model_loaded: bool
    last_training: Optional[datetime]
    current_accuracy: float | None # Can be None if never trained
    training_samples: int
    features_available: List[str] # List of feature names the model expects
    performance_metrics: Dict[str, float] = field(default_factory=dict) 


# --- Main MLPredictor Class ---
class MLPredictor:
    """Manages the machine learning model lifecycle including training"""

    def __init__(
        self,
        hass: HomeAssistant,
        data_manager: DataManager,
        error_handler: ErrorHandlingService,
        notification_service = None  # NEW: Optional for backward compatibility
    ):
        """Initialize the MLPredictor"""
        self.hass = hass
        self.data_manager = data_manager
        self.error_handler = error_handler
        self.notification_service = notification_service  # NEW
        self.data_adapter = TypedDataAdapter()
        
        # Historical cache for lag features
        self._historical_cache: Dict[str, Dict[str, float]] = {
            'daily_productions': {},
            'hourly_productions': {},
        }
        self._recent_weather_samples: List[Dict[str, Any]] = []  # Last 24h weather samples for trend
        self._weather_samples_last_loaded: Optional[datetime] = None  # Cache timestamp

        # ML Components
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.trainer = RidgeTrainer()
        self.prediction_orchestrator = PredictionOrchestrator()
        self.sample_collector = SampleCollector(hass, data_manager) # Pass SunGuard later if needed

        # Model State
        self.model_state = ModelState.UNINITIALIZED
        self.model_loaded = False
        self.current_weights: Optional[LearnedWeights] = None
        self.current_profile: Optional[HourlyProfile] = None # Hourly profile for fallback/comparison
        self.peak_power_kw: float = 0.0  # Peak power in kW (explicit unit) for physical limit clipping

        # Metrics and Timestamps
        self.current_accuracy: float | None = None
        self.training_samples: int = 0
        self.last_training_time: Optional[datetime] = None
        self.last_training_samples: int = 0  # NEW: Track training sample count for adaptive blending

        # Training mode flag (V2 uses hourly_predictions.json)
        self.use_v2_training = True  # NEW: Enable V2 training by default

        # Performance Tracking (optional)
        self.performance_metrics = {
            "avg_prediction_time_ms": 0.0,
            "total_predictions": 0,
            "successful_predictions": 0,
            "error_rate": 0.0
        }
        self._prediction_times: List[float] = [] # Store recent prediction times for averaging

        # Background Task Management
        self._hourly_sample_listener_remove = None
        self._daily_training_listener_remove = None
        self._daily_training_task: Optional[asyncio.TimerHandle] = None # Store timer handle
        self._stop_event = asyncio.Event() # For graceful shutdown

        # WARNING FIX 8: Training lock to prevent concurrent training
        self._training_lock = asyncio.Lock()

        _LOGGER.info("MLPredictor initialized.")

    @property
    def can_predict(self) -> bool:
        """Check if ML model can make predictions"""
        return (
            self.model_loaded and
            self.model_state == ModelState.READY and
            self.current_weights is not None and
            self.scaler is not None and
            self.scaler.is_fitted
        )

    async def initialize(self) -> bool:
        """Initializes the ML Predictor by loading existing model data"""
        _LOGGER.info("Initializing ML Predictor...")
        init_success = False
        try:
            _ensure_numpy() # Check if numpy is available early

            # 1. Load Learned Weights
            loaded_weights = await self.data_manager.get_learned_weights()
            if loaded_weights:
                # Validate feature count compatibility
                loaded_feature_count = len(loaded_weights.feature_names) if hasattr(loaded_weights, 'feature_names') and loaded_weights.feature_names else 0

                # Check if loaded model is V2 (44 features) and switch feature engineer if needed
                if loaded_feature_count == 44:
                    _LOGGER.info(f"Loaded model has {loaded_feature_count} features (V2). Switching to FeatureEngineerV2 for predictions.")
                    self.feature_engineer = FeatureEngineerV2()
                    expected_feature_count = len(self.feature_engineer.feature_names)
                elif loaded_feature_count == 27:
                    _LOGGER.info(f"Loaded model has {loaded_feature_count} features (V1). Using FeatureEngineer for predictions.")
                    expected_feature_count = len(self.feature_engineer.feature_names)
                else:
                    expected_feature_count = len(self.feature_engineer.feature_names)

                if loaded_feature_count > 0 and loaded_feature_count != expected_feature_count:
                    _LOGGER.warning(
                        f"Feature count mismatch: loaded model has {loaded_feature_count} features, "
                        f"current version expects {expected_feature_count}. Model needs retraining."
                    )

                    # Don't load incompatible weights
                    self.model_state = ModelState.UNINITIALIZED

                    # Send notification to user about retraining
                    if self.notification_service:
                        try:
                            await self.notification_service.show_model_retraining_required(
                                reason="feature_mismatch",
                                old_features=loaded_feature_count,
                                new_features=expected_feature_count
                            )
                        except Exception as notif_err:
                            _LOGGER.debug(f"Could not send retraining notification: {notif_err}")

                    # Trigger synchronous retrain to ensure model is available
                    _LOGGER.info("Starting automatic model retraining due to feature mismatch...")
                    try:
                        training_result = await self.train_model()
                        if training_result and training_result.success:
                            _LOGGER.info(
                                f"✓ Model successfully retrained with {expected_feature_count} features. "
                                f"Accuracy: {training_result.accuracy:.2%}"
                            )
                            init_success = True
                        else:
                            _LOGGER.error(
                                "✗ Model retraining failed. ML predictions will be unavailable. "
                                "Please check logs and consider manual retraining."
                            )
                            init_success = False
                    except Exception as train_err:
                        _LOGGER.error(f"Exception during automatic retraining: {train_err}", exc_info=True)
                        init_success = False

                    return init_success

                self.current_weights = loaded_weights
                self.current_accuracy = loaded_weights.accuracy
                self.training_samples = loaded_weights.training_samples
                try:
                     parsed_time = dt_util.parse_datetime(loaded_weights.last_trained)
                     if parsed_time:
                         # Ensure timezone-aware (HA requires this for TIMESTAMP sensors)
                         self.last_training_time = dt_util.as_local(parsed_time) if parsed_time.tzinfo is None else parsed_time
                     else:
                         self.last_training_time = None
                except (ValueError, TypeError):
                     _LOGGER.warning("Could not parse last_trained timestamp from weights file.")
                     self.last_training_time = None

                self.model_loaded = True
                
                if self.training_samples >= MIN_TRAINING_DATA_POINTS:
                    self.model_state = ModelState.READY
                else:
                    self.model_state = ModelState.UNINITIALIZED # Ready, but not trained

                if hasattr(loaded_weights, 'feature_means') and loaded_weights.feature_means:
                    try:
                        self.scaler.set_state({
                            'means': loaded_weights.feature_means,
                            'stds': loaded_weights.feature_stds,
                            'is_fitted': True,
                            'feature_names_order': loaded_weights.feature_names 
                        })
                        _LOGGER.info("Scaler state loaded from learned weights (%d features).", len(self.scaler.means))
                    except Exception as scaler_err:
                         _LOGGER.error("Failed to load scaler state from weights: %s", scaler_err)
                else:
                    _LOGGER.info("No scaler state found in learned weights. Scaler needs fitting.")


                _LOGGER.info(
                    "Learned weights loaded: Accuracy=%.2f%%, Samples=%d, Last Trained=%s, State=%s",
                    (self.current_accuracy * 100) if self.current_accuracy is not None else 0.0,
                    self.training_samples,
                    self.last_training_time.strftime('%Y-%m-%d %H:%M') if self.last_training_time else "Never",
                    self.model_state.value
                )
            else:
                _LOGGER.info("No existing learned weights found. Model needs training.")
                self.model_state = ModelState.UNINITIALIZED

            # 2. Load Hourly Profile (used as a fallback strategy)
            loaded_profile = await self.data_manager.get_hourly_profile()
            if loaded_profile:
                self.current_profile = loaded_profile
                _LOGGER.info("Hourly profile loaded (Samples=%d).", self.current_profile.samples_count)
            else:
                _LOGGER.info("No existing hourly profile found. Using default.")
                self.current_profile = create_default_hourly_profile()

            # 2b. Load Model State (for performance metrics and additional status info)
            loaded_model_state = await self.data_manager.load_model_state()
            if loaded_model_state and loaded_model_state.get("version") == DATA_VERSION:
                # Restore performance metrics if available
                if "performance_metrics" in loaded_model_state:
                    perf_metrics = loaded_model_state["performance_metrics"]
                    if perf_metrics:
                        self.performance_metrics.update({
                            "avg_prediction_time_ms": perf_metrics.get("avg_prediction_time_ms", 0.0),
                            "error_rate": perf_metrics.get("error_rate", 0.0)
                        })
                        _LOGGER.debug("Performance metrics restored from model_state.json")
                
                # Restore peak_power if stored and not already set
                if self.peak_power_kw == 0.0 and "peak_power_kw" in loaded_model_state:
                    self.peak_power_kw = loaded_model_state.get("peak_power_kw", 0.0)
                    _LOGGER.debug("Peak power restored: %.2f kW", self.peak_power_kw)
                
                # If no weights were loaded, try to restore basic state from model_state
                if not loaded_weights and loaded_model_state.get("model_loaded"):
                    _LOGGER.info("No weights file found, but model_state indicates previous training existed.")
                    self.training_samples = loaded_model_state.get("training_samples", 0)
                    self.current_accuracy = loaded_model_state.get("current_accuracy", 0.0)
                    
                    # Parse last_training if available
                    if loaded_model_state.get("last_training"):
                        try:
                            parsed_time = dt_util.parse_datetime(loaded_model_state["last_training"])
                            if parsed_time:
                                self.last_training_time = dt_util.as_local(parsed_time) if parsed_time.tzinfo is None else parsed_time
                        except (ValueError, TypeError):
                            pass
                
                _LOGGER.info("Model state loaded successfully from model_state.json")
            else:
                _LOGGER.debug("No valid model_state.json found or version mismatch.")

            # 3. Update Prediction Strategies
            self.prediction_orchestrator.update_strategies(
                weights=self.current_weights,
                profile=self.current_profile,
                accuracy=self.current_accuracy if self.current_accuracy is not None else 0.0,
                peak_power_kw=self.peak_power_kw
            )
            _LOGGER.debug("Prediction strategies updated.")

            # 4. Load Historical Cache
            asyncio.create_task(self._load_historical_cache())

            # 5. Schedule Background Tasks
            _LOGGER.info("=== Scheduling background tasks ===")
            
            self._schedule_hourly_sampling()
            _LOGGER.info("[OK] Hourly sampling scheduled (triggers at XX:02)")

            self._schedule_daily_training_check()
            _LOGGER.info("[OK] Daily training check scheduled")
            
            # Log current status
            samples_count = await self._check_training_data_availability()
            _LOGGER.info(f"Current hourly samples available: {samples_count}")

            _LOGGER.info("ML Predictor initialized successfully.")
            init_success = True

        except ImportError:
            _LOGGER.critical("ML Predictor initialization failed: NumPy dependency is missing!")
            self.model_state = ModelState.ERROR
        except Exception as e:
            _LOGGER.error("ML Predictor initialization failed: %s", str(e), exc_info=True)
            self.model_state = ModelState.ERROR
        finally:
             await self._update_model_state_file(status_override=self.model_state if not init_success else None)

        return init_success


    async def _load_historical_cache(self) -> None:
        """Loads historical production data into memory for lag feature calculation"""
        _LOGGER.debug("Loading historical production cache...")
        try:
            # Load hourly samples from last 60 days, as these contain the most accurate data
            start_date = (dt_util.now() - timedelta(days=60)).date().isoformat()
            samples_data = await self.data_manager.get_hourly_samples(start_date=start_date)
            records = samples_data if (samples_data and isinstance(samples_data, list)) else []

            daily_productions_cache: Dict[str, float] = {}
            hourly_productions_cache: Dict[str, float] = {}

            processed_records = 0
            if not records:
                _LOGGER.info("No hourly samples found to build historical cache.")
                return

            for record in records:
                try:
                    actual_kwh = record.get('actual_kwh')
                    timestamp_str = record.get('timestamp')

                    if actual_kwh is None or actual_kwh < 0 or not timestamp_str: continue

                    timestamp_dt = dt_util.parse_datetime(timestamp_str)
                    if not timestamp_dt:
                        _LOGGER.debug(f"Skipping cache record, invalid timestamp: {timestamp_str}")
                        continue

                    # CRITICAL FIX: Normalize timezone - handle naive, UTC, and local timestamps
                    # Assume naive timestamps are UTC (common after migration)
                    if timestamp_dt.tzinfo is None:
                        timestamp_dt = timestamp_dt.replace(tzinfo=timezone.utc)

                    # ENSURE local timezone (handles mixed UTC/local from migration)
                    timestamp_local = dt_util.as_local(timestamp_dt)
                    
                    date_key = timestamp_local.date().isoformat()
                    hour_key = f"{date_key}_{timestamp_local.hour:02d}"

                    # Store hourly value
                    hourly_productions_cache[hour_key] = actual_kwh
                    # Add to daily totals cache
                    daily_productions_cache[date_key] = daily_productions_cache.get(date_key, 0.0) + actual_kwh
                    
                    processed_records += 1

                except (ValueError, KeyError, TypeError) as parse_error:
                    _LOGGER.debug(f"Skipping record for cache due to parsing error: {parse_error}")
                    continue

            self._historical_cache = {
                'daily_productions': daily_productions_cache,
                'hourly_productions': hourly_productions_cache,
            }

            _LOGGER.info(
                "Historical cache loaded from hourly_samples: %d daily totals, %d hourly values processed from %d records.",
                len(self._historical_cache['daily_productions']),
                len(self._historical_cache['hourly_productions']),
                processed_records
            )

        except Exception as e:
            _LOGGER.error("Failed to load historical cache: %s", e, exc_info=True)
            self._historical_cache = {'daily_productions': {}, 'hourly_productions': {}}

    async def _load_recent_weather_samples(self, hours_back: int = 24, force_reload: bool = False) -> None:
        """IMPROVEMENT 7 Load recent weather samples for cloudiness trend calculation PERFORMANCE FIX: Only reload if cache is stale (>15 minutes old) or force_reload=True"""
        try:
            now = dt_util.now()

            # Check if cache is still fresh (< 15 minutes old)
            if not force_reload and self._weather_samples_last_loaded:
                cache_age = (now - self._weather_samples_last_loaded).total_seconds() / 60.0
                if cache_age < 15 and len(self._recent_weather_samples) > 0:
                    _LOGGER.debug(f"Using cached weather samples (age: {cache_age:.1f} min)")
                    return

            # Cache stale or empty - reload samples
            cutoff_time = now - timedelta(hours=hours_back)
            cutoff_str = cutoff_time.isoformat()

            samples = await self.data_manager.get_hourly_samples(
                limit=hours_back * 2,  # Get extra to ensure coverage
                start_date=cutoff_str
            )

            # HIGH PRIORITY FIX: Limit to last 100 samples to prevent memory leak
            # Sort in reverse chronological order (newest first) and limit
            sorted_samples = sorted(
                samples,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )
            self._recent_weather_samples = sorted_samples[:100]  # Keep only 100 most recent
            self._weather_samples_last_loaded = now

            _LOGGER.debug(f"Loaded {len(self._recent_weather_samples)} recent weather samples for trend calculation (limited to 100)")

        except Exception as e:
            _LOGGER.warning(f"Failed to load recent weather samples: {e}")
            self._recent_weather_samples = []
            self._weather_samples_last_loaded = None

    def _calculate_cloudiness_trends(self) -> Dict[str, float]:
        """IMPROVEMENT 7 Calculate cloudiness trends from recent weather samples"""
        trends = {
            'cloudiness_trend_1h': 0.0,
            'cloudiness_trend_3h': 0.0,
            'cloudiness_volatility': 0.0
        }

        try:
            if not self._recent_weather_samples:
                return trends

            # Extract cloudiness values with timestamps
            cloudiness_data = []
            now = dt_util.now()

            for sample in self._recent_weather_samples:
                timestamp_str = sample.get('timestamp')
                if not timestamp_str:
                    continue

                timestamp = dt_util.parse_datetime(timestamp_str)
                if not timestamp:
                    continue

                # Calculate age in hours
                age_hours = (now - timestamp).total_seconds() / 3600.0

                # Get cloudiness from weather_data
                weather_data = sample.get('weather_data', {})
                cloudiness = weather_data.get('cloudiness', weather_data.get('cloud_coverage', None))

                if cloudiness is not None and 0 <= cloudiness <= 100:
                    cloudiness_data.append((age_hours, float(cloudiness)))

            if not cloudiness_data:
                return trends

            # Sort by age (oldest first)
            cloudiness_data.sort()

            # Calculate 1-hour trend
            recent_1h = [c for age, c in cloudiness_data if age <= 1.0]
            if len(recent_1h) >= 2:
                trends['cloudiness_trend_1h'] = recent_1h[-1] - recent_1h[0]

            # Calculate 3-hour trend and volatility
            recent_3h = [c for age, c in cloudiness_data if age <= 3.0]
            if len(recent_3h) >= 2:
                trends['cloudiness_trend_3h'] = recent_3h[-1] - recent_3h[0]

                # Calculate volatility (standard deviation)
                if len(recent_3h) >= 3:
                    mean_cloudiness = sum(recent_3h) / len(recent_3h)
                    variance = sum((c - mean_cloudiness) ** 2 for c in recent_3h) / len(recent_3h)
                    trends['cloudiness_volatility'] = variance ** 0.5

            _LOGGER.debug(f"Cloudiness trends: 1h={trends['cloudiness_trend_1h']:.1f}, "
                         f"3h={trends['cloudiness_trend_3h']:.1f}, vol={trends['cloudiness_volatility']:.1f}")

        except Exception as e:
            _LOGGER.debug(f"Failed to calculate cloudiness trends: {e}")

        return trends


    # --- CORRECTION: Parameter order ---
    async def predict(
        self,
        weather_data: Dict[str, Any],
        # --- (IMPROVEMENT 1) Parameters moved (due to SyntaxError) ---
        prediction_hour: int,
        prediction_date: datetime,
        # --- END ---
        sensor_data: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
    # --- END CORRECTION ---
        """Generates a solar production prediction for a specific hour"""
        prediction_start_time = dt_util.now()
        result: PredictionResult | None = None

        try:
            _ensure_numpy()
            if sensor_data is None: sensor_data = {}
            
            # (IMPROVEMENT 1) Time parameters are now passed directly
            # prediction_time_local = dt_util.as_local(prediction_start_time)
            # prediction_hour = prediction_time_local.hour 

            try:
                # Fetch production from YESTERDAY (local time)
                yesterday_dt = prediction_date - timedelta(days=1)
                yesterday_key = yesterday_dt.date().isoformat()
                yesterday_total_kwh = self._historical_cache['daily_productions'].get(yesterday_key, 0.0)
                sensor_data['production_yesterday'] = float(yesterday_total_kwh)

                # Fetch production from SAME HOUR YESTERDAY (local time)
                same_hour_yesterday_key = f"{yesterday_key}_{prediction_hour:02d}"
                same_hour_yesterday_kwh = self._historical_cache['hourly_productions'].get(same_hour_yesterday_key, 0.0)
                sensor_data['production_same_hour_yesterday'] = float(same_hour_yesterday_kwh)

            except Exception as e:
                _LOGGER.warning(f"Could not retrieve lag features for prediction: {e}")
                sensor_data['production_yesterday'] = 0.0
                sensor_data['production_same_hour_yesterday'] = 0.0

            # IMPROVEMENT 7: Calculate cloudiness trends from recent samples
            try:
                # Force reload for prediction to get freshest data
                await self._load_recent_weather_samples(hours_back=6, force_reload=True)
                trend_features = self._calculate_cloudiness_trends()
                sensor_data.update(trend_features)  # Add trend features to sensor_data
            except Exception as e:
                _LOGGER.debug(f"Could not calculate cloudiness trends: {e}")
                sensor_data['cloudiness_trend_1h'] = 0.0
                sensor_data['cloudiness_trend_3h'] = 0.0
                sensor_data['cloudiness_volatility'] = 0.0

            features = await self.feature_engineer.extract_features(
                weather_data, 
                sensor_data, 
                prediction_hour,  # (IMPROVEMENT 1) Pass hour
                prediction_date  # (IMPROVEMENT 1) Pass date
            )

            if self.scaler.is_fitted:
                features_scaled = self.scaler.transform_single(features)
                _LOGGER.debug("Prediction features scaled.")
            else:
                features_scaled = features
                _LOGGER.debug("Scaler not fitted, using raw features for prediction.")

            result = await self.prediction_orchestrator.predict(features_scaled)

            prediction_end_time = dt_util.now()
            duration_ms = (prediction_end_time - prediction_start_time).total_seconds() * 1000
            self._update_performance_metrics(duration_ms, success=True)

            _LOGGER.debug(f"Hourly prediction successful (Method: {result.method}): {result.prediction:.2f} kWh, "
                          f"Confidence: {result.confidence:.2f}, Duration: {duration_ms:.1f}ms")
            return result

        except ImportError:
             _LOGGER.error("Prediction failed: NumPy dependency is missing.")
             await self.error_handler.handle_error(
                 error=MLModelException("Prediction failed due to missing NumPy"),
                 source="ml_predictor", context={}, pipeline_position="predict"
             )
             fallback_result = await self._get_fallback_prediction(prediction_hour, prediction_date)
             self._update_performance_metrics(0, success=False)
             return fallback_result

        except Exception as e:
            _LOGGER.error(f"Prediction failed: {e}", exc_info=True)
            await self.error_handler.handle_error(
                error=MLModelException(f"Prediction failed: {e}"),
                source="ml_predictor",
                context={
                    "weather_data_keys": list(weather_data.keys()) if weather_data else None,
                    "sensor_data_keys": list(sensor_data.keys()) if sensor_data else None,
                    "model_state": self.model_state.value
                },
                pipeline_position="predict"
            )
            fallback_result = await self._get_fallback_prediction(prediction_hour, prediction_date)
            prediction_end_time = dt_util.now()
            duration_ms = (prediction_end_time - prediction_start_time).total_seconds() * 1000
            self._update_performance_metrics(duration_ms, success=False)
            return fallback_result

    async def _get_fallback_prediction(self, hour: int, date: datetime) -> PredictionResult:
         """Generates a prediction using the fallback strategy"""
         _LOGGER.warning("Using fallback prediction strategy.")
         # WARNING FIX 5: Pass peak_power_kw to FallbackStrategy
         fallback_strategy = FallbackStrategy(self.peak_power_kw)
         default_features = self.feature_engineer.get_default_features(hour, date)
         return await fallback_strategy.predict(default_features)


    async def train_model(self) -> TrainingResult:
        """Trains the Ridge Regression model using historical hourly data - refactored"""
        # Route to V2 if enabled
        if self.use_v2_training:
            _LOGGER.info("Using V2 training pipeline (hourly_predictions.json + astronomy_cache.json)")
            return await self.train_model_v2()

        # Legacy training using hourly_samples.json
        _LOGGER.info("Using legacy training pipeline (hourly_samples.json)")
        from .ml_training_helpers import MLTrainingHelpers
        helpers = MLTrainingHelpers(self)

        # Prevent race condition with non-blocking lock
        try:
            async with asyncio.timeout(5.0):
                async with self._training_lock:
                    training_start_time = dt_util.now()
                    _LOGGER.info(f"Starting ML model training at {training_start_time}...")
                    self.model_state = ModelState.TRAINING
                    await self._update_model_state_file(status_override=ModelState.TRAINING)

                training_records = []
                result: TrainingResult | None = None

                try:
                    _ensure_numpy()
                    await self._load_historical_cache()
                    if not self._historical_cache['daily_productions']:
                        _LOGGER.warning("Historical cache empty. 'production_yesterday' will be 0.")

                    # Load and validate training data
                    training_records = await self.data_manager.get_all_training_records(days=60)
                    valid, error_msg, count = await helpers.validate_training_data(training_records)

                    if not valid:
                        self.model_state = ModelState.UNINITIALIZED if not self.model_loaded else ModelState.READY
                        return TrainingResult(success=False, samples_used=count, error_message=error_msg)

                    _LOGGER.info(f"Preparing {count} records for training...")

                    # Remove outliers
                    training_records = await self._remove_outliers(training_records)
                    _LOGGER.info(f"After outlier removal: {len(training_records)} records remaining")

                    # Prepare features
                    X_train_raw, y_train = await helpers.prepare_training_features(training_records)
                    _LOGGER.debug("Starting scaling and training...")

                    # Scale and train
                    weights_dict_raw, bias, accuracy, best_lambda = await helpers.scale_and_train(
                        X_train_raw, y_train
                    )

                    # Create learned weights
                    new_learned_weights = await helpers.create_learned_weights(
                        weights_dict_raw, bias, accuracy, len(training_records), training_start_time
                    )

                    # Finalize training
                    await helpers.finalize_training(
                        new_learned_weights, accuracy, len(training_records),
                        training_records, training_start_time
                    )

                    training_end_time = dt_util.now()
                    duration_seconds = (training_end_time - training_start_time).total_seconds()

                    result = TrainingResult(
                        success=True,
                        accuracy=accuracy,
                        samples_used=len(training_records),
                        weights=new_learned_weights,
                        training_time_seconds=duration_seconds,
                        feature_count=len(self.feature_engineer.feature_names)
                    )

                    _LOGGER.info(
                        f"ML Training successful in {duration_seconds:.2f}s. "
                        f"Accuracy={accuracy*100:.1f}%, Samples={len(training_records)}."
                    )
                    self.error_handler.log_ml_operation(
                        operation="model_training",
                        success=True,
                        metrics={
                            "accuracy": accuracy,
                            "samples": len(training_records),
                            "features": result.feature_count
                        },
                        duration_seconds=duration_seconds
                    )

                    await helpers.send_training_notifications(True, accuracy, len(training_records))
                    return result

                except ImportError:
                     _LOGGER.critical("ML Training failed: NumPy dependency is missing!")
                     self.model_state = ModelState.ERROR
                     error_msg = "NumPy is missing, cannot train model."
                     result = TrainingResult(success=False, error_message=error_msg)
                     await self.error_handler.handle_error(error=MLModelException(error_msg), source="ml_predictor", pipeline_position="train_model")
                     return result
                except DataIntegrityException as data_err:
                     _LOGGER.error("ML Training failed due to data integrity issue: %s", data_err)
                     self.model_state = ModelState.ERROR
                     result = TrainingResult(success=False, samples_used=len(training_records), error_message=str(data_err))
                     await self.error_handler.handle_error(error=data_err, source="ml_predictor", pipeline_position="train_model (data io)")
                     return result
                except Exception as e:
                    _LOGGER.error("ML Training failed unexpectedly: %s", str(e), exc_info=True)
                    self.model_state = ModelState.ERROR
                    error_msg = f"Unexpected training error: {e}"
                    result = TrainingResult(success=False, samples_used=len(training_records), error_message=error_msg)
                    await self.error_handler.handle_error(error=MLModelException(error_msg), source="ml_predictor", context={"samples_attempted": len(training_records)}, pipeline_position="train_model")
                    duration_seconds = (dt_util.now() - training_start_time).total_seconds()
                    self.error_handler.log_ml_operation(operation="model_training", success=False, metrics={"samples": len(training_records)}, context={"error": str(e)}, duration_seconds=duration_seconds)
                    return result
                finally:
                    if result and result.success:
                        self.model_state = ModelState.READY
                    elif result and not result.success:
                        if not self.model_loaded:
                            self.model_state = ModelState.UNINITIALIZED
                        else:
                            self.model_state = ModelState.READY # Keep old status
                    else:
                        self.model_state = ModelState.ERROR # Fallback

                    await self._update_model_state_file(
                         status_override=self.model_state.value,
                         accuracy_override=result.accuracy if result and result.success else self.current_accuracy,
                         samples_override=result.samples_used if result and result.success else self.training_samples,
                         training_time_override=result.training_time_seconds if result else None
                    )

                    if self.notification_service and result and not result.success:
                        try:
                            await self.notification_service.show_training_complete(
                                success=False,
                                sample_count=result.samples_used if result.samples_used else 0
                            )
                        except Exception as notify_err:
                            _LOGGER.debug(f"Training failure notification failed: {notify_err}")

        except asyncio.TimeoutError:
            # Lock is already held by another training process
            _LOGGER.warning("Training already in progress, skipping concurrent training request")
            return TrainingResult(
                success=False,
                error_message="Training already in progress",
                samples_used=0
            )

    async def train_model_v2(self) -> TrainingResult:
        """Train model using V2 data structure (hourly_predictions.json + astronomy_cache.json)"""
        from .ml_training_helpers_v2 import MLTrainingHelpersV2

        helpers = MLTrainingHelpersV2(self)

        # Prevent race condition with non-blocking lock
        try:
            async with asyncio.timeout(5.0):
                async with self._training_lock:
                    training_start_time = dt_util.now()
                    _LOGGER.info(f"Starting ML model training (V2) at {training_start_time}...")
                    self.model_state = ModelState.TRAINING
                    await self._update_model_state_file(status_override=ModelState.TRAINING)

                training_records = []
                result: TrainingResult | None = None

                try:
                    _ensure_numpy()

                    # Load training data from V2 structure
                    _LOGGER.info("Loading training data from V2 structure (hourly_predictions.json + astronomy_cache.json)...")
                    training_records, count = await helpers.load_training_data_v2()

                    # Validate
                    valid, error_msg, count = await helpers.validate_training_data(training_records)
                    if not valid:
                        self.model_state = ModelState.UNINITIALIZED if not self.model_loaded else ModelState.READY
                        return TrainingResult(success=False, samples_used=count, error_message=error_msg)

                    _LOGGER.info(f"Preparing {count} records for training (V2)...")

                    # Prepare features using V2 feature engineering
                    X_train_raw, y_train = await helpers.prepare_training_features_v2(training_records)
                    _LOGGER.debug(f"V2 Feature extraction complete: {len(X_train_raw)} samples")

                    # Scale and train (V2 returns feature_engineer_v2)
                    weights_dict_raw, bias, accuracy, best_lambda, feature_eng_v2 = await helpers.scale_and_train(
                        X_train_raw, y_train
                    )

                    # Create learned weights (pass V2 feature engineer for correct feature names)
                    new_learned_weights = await helpers.create_learned_weights(
                        weights_dict_raw, bias, accuracy, len(training_records), training_start_time, feature_eng_v2
                    )

                    # Finalize training
                    await helpers.finalize_training(
                        new_learned_weights, accuracy, len(training_records),
                        training_records, training_start_time
                    )

                    training_end_time = dt_util.now()
                    duration_seconds = (training_end_time - training_start_time).total_seconds()

                    result = TrainingResult(
                        success=True,
                        accuracy=accuracy,
                        samples_used=len(training_records),
                        weights=new_learned_weights,
                        training_time_seconds=duration_seconds,
                        feature_count=len(weights_dict_raw)
                    )

                    # Store training sample count for adaptive blending (used by forecast_orchestrator)
                    self.last_training_samples = len(training_records)

                    _LOGGER.info(
                        f"ML Training V2 successful in {duration_seconds:.2f}s. "
                        f"Accuracy={accuracy*100:.1f}%, Samples={len(training_records)}, Features={len(weights_dict_raw)}."
                    )

                    self.error_handler.log_ml_operation(
                        operation="model_training_v2",
                        success=True,
                        metrics={
                            "accuracy": accuracy,
                            "samples": len(training_records),
                            "features": result.feature_count
                        },
                        duration_seconds=duration_seconds
                    )

                    await helpers.send_training_notifications(True, accuracy, len(training_records))
                    return result

                except Exception as e:
                    _LOGGER.error(f"ML Training V2 failed: {e}", exc_info=True)
                    self.model_state = ModelState.ERROR
                    result = TrainingResult(success=False, samples_used=len(training_records), error_message=str(e))
                    await self.error_handler.handle_error(error=Exception(str(e)), source="ml_predictor_v2", pipeline_position="train_model_v2")
                    return result

        except asyncio.TimeoutError:
            _LOGGER.warning("Training already in progress, skipping concurrent training request")
            return TrainingResult(
                success=False,
                error_message="Training already in progress",
                samples_used=0
            )

    async def _update_hourly_profile(self, training_records: List[Dict[str, Any]]) -> None:
        """Updates the hourly production profile based on provided training records"""
        _LOGGER.debug(f"Updating hourly profile using {len(training_records)} records...")
        if not training_records:
            _LOGGER.warning("No records provided for hourly profile update.")
            if self.current_profile is None: self.current_profile = create_default_hourly_profile()
            return

        try:
            np = _ensure_numpy()
            hourly_data: Dict[int, List[float]] = {hour: [] for hour in range(24)}
            valid_sample_count = 0

            for record in training_records:
                try:
                    actual_kwh = record.get('actual_kwh')
                    timestamp_str = record.get('timestamp')
                    if actual_kwh is None or actual_kwh <= 0 or not timestamp_str: continue
                    
                    # Read LOCAL hour from timestamp
                    timestamp = dt_util.parse_datetime(timestamp_str) 
                    if not timestamp: continue
                    hour = timestamp.hour 
                    
                    hourly_data[hour].append(actual_kwh)
                    valid_sample_count += 1
                except (ValueError, KeyError, TypeError) as e:
                    _LOGGER.debug(f"Skipping invalid record during profile update: {e}")
                    continue

            if valid_sample_count == 0:
                _LOGGER.warning("No valid hourly values found in provided records. Hourly profile not updated.")
                if self.current_profile is None: self.current_profile = create_default_hourly_profile()
                return

            hourly_averages_median: Dict[str, float] = {}
            for hour, values in hourly_data.items():
                if values:
                    median_val = await self.hass.async_add_executor_job(np.median, values)
                    hourly_averages_median[str(hour)] = float(median_val)
                else:
                    hourly_averages_median[str(hour)] = 0.0

            confidence = min(1.0, max(0.1, valid_sample_count / 300.0))
            new_profile = HourlyProfile(
                hourly_averages=hourly_averages_median, samples_count=valid_sample_count,
                last_updated=dt_util.now().isoformat(), confidence=confidence,
            )

            await self.data_manager.save_hourly_profile(new_profile)
            self.current_profile = new_profile

            _LOGGER.info(f"Hourly profile updated successfully using {valid_sample_count} valid samples. Confidence: {confidence:.2f}")

        except ImportError:
             _LOGGER.error("Hourly profile update failed: NumPy dependency is missing!")
        except Exception as e:
            _LOGGER.error(f"Hourly profile update failed: {e}", exc_info=True)
            if self.current_profile is None: self.current_profile = create_default_hourly_profile()


    async def _remove_outliers(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """CRITICAL FIX 4 Removes outliers from training data using IQR method"""
        if not records:
            return records

        try:
            np = _ensure_numpy()

            # Group records by hour for hour-specific outlier detection
            hourly_groups: Dict[int, List[Dict[str, Any]]] = {hour: [] for hour in range(24)}

            for record in records:
                try:
                    timestamp_str = record.get('timestamp')
                    if not timestamp_str:
                        continue

                    timestamp = dt_util.parse_datetime(timestamp_str)
                    if not timestamp:
                        continue

                    hour = dt_util.ensure_local(timestamp).hour
                    hourly_groups[hour].append(record)
                except Exception:
                    continue

            # Apply IQR outlier removal per hour
            filtered_records = []
            outliers_removed = 0

            for hour, hour_records in hourly_groups.items():
                # HIGH PRIORITY FIX: Only apply IQR when we have enough samples
                # Require at least 10 samples per hour for robust statistics
                if len(hour_records) < 10:
                    filtered_records.extend(hour_records)
                    continue

                # Extract actual_kwh values
                values = []
                for rec in hour_records:
                    actual = rec.get('actual_kwh')
                    if actual is None:
                        actual = rec.get('actual_value')
                    if actual is not None and actual >= 0:
                        values.append(actual)

                if len(values) < 10:
                    filtered_records.extend(hour_records)
                    continue

                # Calculate IQR
                values_array = np.array(values)
                q1 = np.percentile(values_array, 25)
                q3 = np.percentile(values_array, 75)
                iqr = q3 - q1

                # HIGH PRIORITY FIX: Less aggressive outlier bounds (3.0×IQR instead of 2.0)
                # 3.0×IQR removes only extreme outliers, preserves best-case scenarios
                lower_bound = q1 - 3.0 * iqr
                upper_bound = q3 + 3.0 * iqr

                # Filter records within bounds
                for rec in hour_records:
                    actual = rec.get('actual_kwh')
                    if actual is None:
                        actual = rec.get('actual_value')

                    if actual is None:
                        filtered_records.append(rec)  # Keep records without actual values
                    elif lower_bound <= actual <= upper_bound:
                        filtered_records.append(rec)
                    else:
                        outliers_removed += 1
                        _LOGGER.debug(
                            f"Outlier removed: hour={hour}, value={actual:.2f} kWh "
                            f"(bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
                        )

            _LOGGER.info(
                f"Outlier removal complete: {outliers_removed} outliers removed "
                f"({len(records)} → {len(filtered_records)} records)"
            )
            return filtered_records

        except ImportError:
            _LOGGER.warning("NumPy not available for outlier removal, skipping")
            return records
        except Exception as e:
            _LOGGER.error(f"Outlier removal failed: {e}", exc_info=True)
            return records  # Return original records on error


    async def _update_model_state_file(
        self,
        status_override: Optional[ModelState | str] = None,
        accuracy_override: Optional[float] = None,
        samples_override: Optional[int] = None,
        training_time_override: Optional[float] = None
    ) -> None:
        """Safely updates the model_statejson file with current predictor status"""
        _LOGGER.debug("Updating model_state.json...")
        try:
            status_to_save = status_override if status_override else self.model_state
            status_str = status_to_save.value if isinstance(status_to_save, ModelState) else str(status_to_save)
            accuracy = accuracy_override if accuracy_override is not None else self.current_accuracy
            samples = samples_override if samples_override is not None else self.training_samples
            last_training_iso = self.last_training_time.isoformat() if self.last_training_time else None

            current_state = {
                "version": DATA_VERSION, "model_loaded": self.model_loaded,
                "last_training": last_training_iso, "training_samples": samples,
                "current_accuracy": float(accuracy) if accuracy is not None else None,
                "status": status_str,
                "peak_power_kw": self.peak_power_kw,  # Store peak power (in kW) for reload
                "model_info": {"version": ML_MODEL_VERSION, "type": "ridge_regression_normalized"},
                "performance_metrics": {
                     "avg_prediction_time_ms": self.performance_metrics.get("avg_prediction_time_ms"),
                     "error_rate": self.performance_metrics.get("error_rate")
                },
                **({"training_time_seconds": round(training_time_override, 2)} if training_time_override is not None else {}),
                "last_updated": dt_util.now().isoformat()
            }

            await self.data_manager.save_model_state(current_state)
            _LOGGER.debug(f"model_state.json updated successfully (Status: {status_str}).")

        except Exception as e:
            _LOGGER.error(f"Failed to update model_state.json: {e}", exc_info=True)


    async def _check_training_data_availability(self) -> int:
        """Checks how many valid hourly samples are available for training"""
        try:
            # Load samples from last 60 days
            from ..core.core_helpers import SafeDateTimeUtil as dt_util
            start_date = (dt_util.now() - timedelta(days=60)).date().isoformat()
            records_data = await self.data_manager.get_hourly_samples(start_date=start_date)
            records = records_data if isinstance(records_data, list) else []
            # Count all samples with valid actual_kwh (including zeros - night hours are valid!)
            valid_count = sum(1 for r in records 
                            if r.get('actual_kwh') is not None and r['actual_kwh'] >= 0)
            _LOGGER.debug(f"Training data check: Found {valid_count} valid hourly samples (including zero-yield).")
            return valid_count
        except Exception as e:
            _LOGGER.error(f"Training data availability check failed: {e}", exc_info=True)
            return 0


    def _schedule_hourly_sampling(self) -> None:
        """Schedules the hourly callback for data sampling"""
        if self._hourly_sample_listener_remove: self._hourly_sample_listener_remove()
        self._hourly_sample_listener_remove = async_track_time_change(
            self.hass, self._hourly_learning_callback, minute=2, second=0
        )
        _LOGGER.info("Hourly data sampling scheduled to run at minute 2 of every hour.")

    @callback 
    async def _hourly_learning_callback(self, now_local: datetime) -> None:
        """Callback triggered every hour to collect the sample for the previous hour"""
        hour_to_collect_dt = now_local - timedelta(hours=1)
        
        _LOGGER.info(
            f"=== Hourly sampling triggered at {now_local.strftime('%H:%M:%S')} for hour {hour_to_collect_dt.hour} ==="
        )
        
        # Check if sample collector is available
        if not hasattr(self, 'sample_collector') or self.sample_collector is None:
            _LOGGER.error("[X] Sample collector not initialized!")
            return
            
        _LOGGER.debug(f"Sample collector ready: {type(self.sample_collector).__name__}")
        
        task = asyncio.create_task(self.sample_collector.collect_sample(hour_to_collect_dt))
        
        def _handle_task_error(task):
            try:
                result = task.result()
                _LOGGER.info(f"[OK] Sample collection completed for hour {hour_to_collect_dt.hour}")
            except Exception as e:
                _LOGGER.error(
                    f"[X] Error during hourly sample collection for hour {hour_to_collect_dt.hour}: {e}",
                    exc_info=True
                )
        
        task.add_done_callback(_handle_task_error)


    def _schedule_daily_training_check(self, reschedule_delay_sec: Optional[float] = None) -> None:
        """Schedules the next daily check to see if model retraining is needed"""
        try:
            # Cancel existing task if present
            if self._daily_training_task:
                self._daily_training_task.cancel()
                self._daily_training_task = None

            # Calculate delay
            if reschedule_delay_sec is not None:
                delay = reschedule_delay_sec
                _LOGGER.info(f"Scheduling next training check in {delay:.0f} seconds.")
            else:
                now_local = dt_util.now()  # now() already returns local time
                target_time_local = now_local.replace(hour=23, minute=5, second=0, microsecond=0)
                if now_local >= target_time_local:
                    target_time_local += timedelta(days=1)
                delay = (target_time_local - now_local).total_seconds()
                _LOGGER.info(f"Scheduling next daily training check at {target_time_local.strftime('%Y-%m-%d %H:%M:%S')} (in {delay:.0f} seconds).")

            # Create a proper wrapper function that handles the async task correctly
            def _trigger_training_check():
                """Wrapper function to properly create and handle the async task"""
                try:
                    _LOGGER.debug("Daily training check trigger fired - creating async task...")
                    task = self.hass.async_create_task(self._daily_training_check_callback())
                    # Add error callback to catch any exceptions
                    task.add_done_callback(lambda t: self._handle_training_task_error(t))
                except Exception as e:
                    _LOGGER.error(f"Failed to create daily training check task: {e}", exc_info=True)

            # Schedule the task
            self._daily_training_task = self.hass.loop.call_later(delay, _trigger_training_check)
            _LOGGER.debug(f"Daily training check task scheduled successfully (delay={delay:.0f}s)")

        except Exception as e:
            _LOGGER.error(f"Failed to schedule daily training check: {e}", exc_info=True)
            # Try to reschedule with a short delay as fallback
            if reschedule_delay_sec is None:
                _LOGGER.info("Attempting fallback rescheduling in 1 hour...")
                try:
                    self._schedule_daily_training_check(reschedule_delay_sec=3600.0)
                except Exception as fallback_error:
                    _LOGGER.error(f"Fallback rescheduling also failed: {fallback_error}", exc_info=True)

    async def _daily_training_check_callback(self) -> None:
        """Callback executed daily to check if retraining is needed and trigger it"""
        _LOGGER.info("Performing daily check: Should ML model be retrained?")
        try:
            if await self._should_retrain():
                _LOGGER.info("Retraining criteria met. Starting scheduled model training...")
                training_result = await self.train_model()
                if training_result.success: _LOGGER.info("Scheduled model training completed successfully.")
                else: _LOGGER.warning("Scheduled model training failed: %s", training_result.error_message)
            else:
                _LOGGER.info("No scheduled retraining needed at this time.")
        except Exception as e:
            _LOGGER.error("Error during daily training check/trigger: %s", e, exc_info=True)
        finally:
            # CRITICAL FIX: Wrap rescheduling in try-except to prevent silent failures
            try:
                if not self._stop_event.is_set():
                    _LOGGER.debug("Attempting to reschedule next daily training check...")
                    self._schedule_daily_training_check(reschedule_delay_sec=None)
                    _LOGGER.debug("Daily training check rescheduled successfully")
                else:
                    _LOGGER.info("Stop event set - skipping daily training check rescheduling")
            except Exception as reschedule_error:
                _LOGGER.error(
                    f"CRITICAL: Failed to reschedule daily training check: {reschedule_error}",
                    exc_info=True
                )


    async def _should_retrain(self) -> bool:
        """Determines if the model should be automatically retrained based on criteria"""
        _LOGGER.debug("Checking retraining criteria...")
        try:
            available_samples = await self._check_training_data_availability()
            
            if not self.model_loaded or not self.last_training_time or self.current_accuracy is None or self.training_samples < MIN_TRAINING_DATA_POINTS:
                if available_samples >= MIN_TRAINING_DATA_POINTS:
                     _LOGGER.info(f"Retraining needed: Model not loaded/trained, but sufficient data ({available_samples}) available.")
                     return True
                else:
                     _LOGGER.info(f"Skipping retraining: Model not loaded/trained and insufficient data ({available_samples} < {MIN_TRAINING_DATA_POINTS}).")
                     return False

            accuracy_threshold = MODEL_ACCURACY_THRESHOLD * 0.9
            if self.current_accuracy < accuracy_threshold:
                _LOGGER.info(f"Retraining needed: Current accuracy ({self.current_accuracy:.3f}) is below threshold ({accuracy_threshold:.3f}).")
                return True

            max_model_age = timedelta(days=7)
            time_since_training = dt_util.now() - self.last_training_time
            if time_since_training > max_model_age:
                _LOGGER.info(f"Retraining needed: Last training was {time_since_training.days} days ago (max allowed: {max_model_age.days}).")
                return True

            if available_samples > self.training_samples * 1.5 or (available_samples > self.training_samples + 100):
                _LOGGER.info(f"Retraining needed: Significant new data available ({available_samples} current vs {self.training_samples} last trained).")
                return True

            _LOGGER.debug("Retraining criteria not met.")
            return False
        except Exception as e:
            _LOGGER.error(f"Error during _should_retrain check: {e}", exc_info=True)
            return False

    def _handle_training_task_error(self, task: asyncio.Task) -> None:
        """Handle errors from the daily training check task"""
        try:
            # This will raise if the task had an exception
            task.result()
        except asyncio.CancelledError:
            _LOGGER.debug("Daily training check task was cancelled")
        except Exception as e:
            _LOGGER.error(f"Daily training check task failed with error: {e}", exc_info=True)

    def _update_performance_metrics(self, prediction_time_ms: float, success: bool) -> None:
        """Updates rolling average prediction time and error rate"""
        try:
            self.performance_metrics["total_predictions"] += 1
            if success:
                self.performance_metrics["successful_predictions"] += 1
                self._prediction_times.append(prediction_time_ms)
                if len(self._prediction_times) > 100: self._prediction_times.pop(0)
            if self._prediction_times:
                 np = _ensure_numpy()
                 avg_time = np.mean(self._prediction_times)
                 self.performance_metrics["avg_prediction_time_ms"] = avg_time
            else: self.performance_metrics["avg_prediction_time_ms"] = 0.0
            total = self.performance_metrics["total_predictions"]
            successful = self.performance_metrics["successful_predictions"]
            if total > 0: self.performance_metrics["error_rate"] = 1.0 - (successful / total)
            else: self.performance_metrics["error_rate"] = 0.0
            _LOGGER.debug(f"Performance metrics updated: AvgTime={self.performance_metrics['avg_prediction_time_ms']:.1f}ms, ErrorRate={self.performance_metrics['error_rate']:.3f}")
        except ImportError: self.performance_metrics["avg_prediction_time_ms"] = -1.0
        except Exception as e: _LOGGER.warning(f"Failed to update performance metrics: {e}")


    def set_entities(
        self, power_entity: Optional[str]=None, weather_entity: Optional[str]=None, solar_capacity: Optional[float]=None,
        temp_sensor: Optional[str]=None, wind_sensor: Optional[str]=None, rain_sensor: Optional[str]=None,
        uv_sensor: Optional[str]=None, lux_sensor: Optional[str]=None, 
        humidity_sensor: Optional[str]=None
    ) -> None:
        """Configures the entity IDs used by the ML predictor and its components"""
        _LOGGER.info("Configuring entities for MLPredictor and SampleCollector...")
        _LOGGER.debug(f"set_entities called with solar_capacity={solar_capacity}")
        
        # Store peak power for physical limit clipping (in kW)
        if solar_capacity and solar_capacity > 0:
            self.peak_power_kw = float(solar_capacity)  # Ensure it's a float in kW
            _LOGGER.info(f"Peak-Power configured: {self.peak_power_kw} kW (from config: {solar_capacity})")
        else:
            _LOGGER.warning(f"Peak-Power not configured or invalid (received: {solar_capacity}) - predictions may be unrealistic")
        
        self.sample_collector.configure_entities(
            weather_entity=weather_entity, power_entity=power_entity, temp_sensor=temp_sensor,
            wind_sensor=wind_sensor, rain_sensor=rain_sensor, uv_sensor=uv_sensor, lux_sensor=lux_sensor,
            humidity_sensor=humidity_sensor 
        )
        
        _LOGGER.info(
            "Entities configured in SampleCollector: power=%s, weather=%s, temp=%s, wind=%s, rain=%s, uv=%s, lux=%s, humidity=%s",
            power_entity, weather_entity, temp_sensor, wind_sensor, rain_sensor, uv_sensor, lux_sensor, humidity_sensor
        )


    def is_healthy(self) -> bool:
        """Performs a health check on the ML predictor state"""
        _LOGGER.debug("Performing ML predictor health check...")
        try:
            _ensure_numpy() 
            
            if not self.model_loaded: 
                _LOGGER.debug("Health check failed: Model not loaded.")
                return False

            if self.model_state != ModelState.READY: 
                _LOGGER.debug(f"Health check failed: Model state is {self.model_state.value}, expected READY.")
                return False

            if self.training_samples < MIN_TRAINING_DATA_POINTS:
                _LOGGER.info(f"Health check: Model is loaded but not sufficiently trained (Samples: {self.training_samples} < {MIN_TRAINING_DATA_POINTS}). "
                             "ML strategy will be skipped (Fallback active).")
                return False 

            if self.current_accuracy is not None and self.current_accuracy < 0.3: 
                _LOGGER.warning(f"Health check warning: Accuracy is very low ({self.current_accuracy:.3f}). Model might be unreliable.")
            
            max_age = timedelta(days=14)
            if self.last_training_time and (dt_util.now() - self.last_training_time) > max_age: 
                _LOGGER.warning(f"Health check warning: Last training was more than {max_age.days} days ago. Model might be outdated.")
            
            _LOGGER.debug("ML predictor health check passed (Loaded, Ready, Sufficient Samples).")
            return True
            
        except ImportError: 
            _LOGGER.error("Health check failed: NumPy dependency missing.")
            return False
        except Exception as e: 
            _LOGGER.error(f"Error during ML predictor health check: {e}", exc_info=True)
            return False

    async def get_today_prediction(self) -> Optional[float]:
        """Get total daily prediction for today"""
        _LOGGER.debug("get_today_prediction stub called, returning None (use orchestrator)")
        return None

    async def get_tomorrow_prediction(self) -> Optional[float]:
        """Get total daily prediction for tomorrow"""
        _LOGGER.debug("get_tomorrow_prediction stub called, returning None (use orchestrator)")
        return None

    async def force_training(self) -> bool:
        """Force model training regardless of conditions"""
        _LOGGER.info("Force training requested via service...")
        try:
            result = await self.train_model()
            return result.success if result else False
        except Exception as e:
            _LOGGER.error(f"Force training failed: {e}", exc_info=True)
            return False

    async def reset_model(self) -> bool:
        """Reset the ML model by clearing learned weights and reverting to rule-based pr..."""
        _LOGGER.info("Resetting ML model...")
        try:
            # Clear current weights and model state
            self.current_weights = None
            self.current_accuracy = None
            self.training_samples = 0
            self.last_training_time = None
            self.model_loaded = False
            self.model_state = ModelState.UNINITIALIZED

            # Delete learned weights file
            success = await self.data_manager.delete_learned_weights()
            if not success:
                _LOGGER.warning("Could not delete learned weights file (may not exist)")

            # Reset scaler
            self.scaler = StandardScaler()

            # Update prediction orchestrator to use fallback strategies only
            self.prediction_orchestrator.update_strategies(
                weights=None,
                profile=self.current_profile,
                accuracy=None,
                peak_power_kw=self.peak_power_kw
            )

            # Update model state file
            await self._update_model_state_file(
                status_override=ModelState.UNINITIALIZED.value,
                accuracy_override=None,
                samples_override=0,
                training_time_override=None
            )

            _LOGGER.info("ML model reset successfully. System will use rule-based predictions.")

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to reset ML model: {e}", exc_info=True)
            await self.error_handler.handle_error(
                error=MLModelException(f"Model reset failed: {e}"),
                source="ml_predictor",
                pipeline_position="reset_model"
            )
            return False

    async def async_will_remove_from_hass(self) -> None:
        """Clean up resources when the integration is unloaded or HA stops"""
        _LOGGER.info("Cleaning up MLPredictor background tasks and listeners...")
        self._stop_event.set() # Signal background tasks to stop first

        if self._hourly_sample_listener_remove:
            try:
                self._hourly_sample_listener_remove()
                _LOGGER.debug("Removed hourly sample listener.")
            except Exception as e:
                _LOGGER.warning(f"Error removing hourly sample listener: {e}")
            finally:
                 self._hourly_sample_listener_remove = None # Ensure it's None

        if self._daily_training_task:
            self._daily_training_task.cancel()
            _LOGGER.debug("Cancelled scheduled daily training check timer.")
            self._daily_training_task = None

        _LOGGER.info("MLPredictor cleanup finished.")