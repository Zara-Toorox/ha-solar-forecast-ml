"""
ML Predictor for the Solar Forecast ML Integration.
Handles model training, prediction, and hourly data sampling.

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
from datetime import datetime, timedelta, timezone # Added timezone
from typing import Dict, Any, Optional, List, TYPE_CHECKING
# --- HIER IST DIE KORREKTUR INTEGRIERT ---
from dataclasses import dataclass, field # Added 'field' import
# --- ENDE KORREKTUR ---
from enum import Enum

# Lazy import NumPy
if TYPE_CHECKING:
    import numpy as np
_np = None

from homeassistant.core import HomeAssistant, callback # Added 'callback' import
from homeassistant.helpers.event import async_track_time_change, async_call_later

from .const import (
    MIN_TRAINING_DATA_POINTS, MODEL_ACCURACY_THRESHOLD, DATA_VERSION,
    ML_MODEL_VERSION,
    # KORREKTUR (Block 4): Import fÃƒÂ¼r Faktor-Begrenzung
    CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX
)
from .data_manager import DataManager
from .helpers import SafeDateTimeUtil as dt_util
from .ml_types import (
    LearnedWeights, HourlyProfile,
    create_default_learned_weights, create_default_hourly_profile
)
from .typed_data_adapter import TypedDataAdapter
from .error_handling_service import ErrorHandlingService
from .exceptions import MLModelException, ErrorSeverity, DataIntegrityException

# Import ML components
from .ml_scaler import StandardScaler
from .ml_feature_engineering import FeatureEngineer
from .ml_training import RidgeTrainer
from .ml_prediction_strategies import (
    PredictionOrchestrator, PredictionResult,
    MLModelStrategy, ProfileStrategy, FallbackStrategy # Added Fallback for clarity
)
from .ml_sample_collector import SampleCollector

_LOGGER = logging.getLogger(__name__)


# --- Helper for Lazy NumPy Import ---
def _ensure_numpy():
    """Lazily imports and returns the NumPy module, raising ImportError if unavailable."""
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
    """Represents the operational state of the ML model."""
    UNINITIALIZED = "uninitialized"
    TRAINING = "training"
    READY = "ready"
    DEGRADED = "degraded" # Model exists but accuracy is low or data is old
    ERROR = "error"


@dataclass
class TrainingResult:
    """Stores the outcome of a model training attempt."""
    success: bool
    accuracy: float | None = None # Accuracy might be None if training failed early
    samples_used: int = 0
    weights: Optional[LearnedWeights] = None
    error_message: Optional[str] = None
    training_time_seconds: Optional[float] = None
    feature_count: Optional[int] = None


@dataclass
class ModelHealth:
    """Represents the health status of the ML model."""
    state: ModelState
    model_loaded: bool
    last_training: Optional[datetime]
    current_accuracy: float | None # Can be None if never trained
    training_samples: int
    features_available: List[str] # List of feature names the model expects
    # Add performance metrics if tracked (e.g., prediction time)
    performance_metrics: Dict[str, float] = field(default_factory=dict) # 'field' is now defined


# --- Main MLPredictor Class ---
class MLPredictor:
    """
    Manages the machine learning model lifecycle including training,
    prediction, data sampling, and state management.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        data_manager: DataManager,
        error_handler: ErrorHandlingService
    ):
        """Initialize the MLPredictor."""
        self.hass = hass
        self.data_manager = data_manager
        self.error_handler = error_handler
        self.data_adapter = TypedDataAdapter()
        
        # Historisches Cache fÃƒÂ¼r Lag-Features
        self._historical_cache: Dict[str, Dict[str, float]] = {
            'daily_productions': {},
            'hourly_productions': {},
        }

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

        # Metrics and Timestamps
        self.current_accuracy: float | None = None
        self.training_samples: int = 0
        self.last_training_time: Optional[datetime] = None

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
        self._daily_training_listener_remove = None # Changed from time_change to call_later approach
        self._daily_training_task: Optional[asyncio.TimerHandle] = None # Store timer handle
        self._stop_event = asyncio.Event() # For graceful shutdown

        # Removed _register_shutdown_handler() - cleanup is handled by service_manager.cleanup()

        _LOGGER.info("MLPredictor initialized.")

    async def initialize(self) -> bool:
        """
        Initializes the ML Predictor by loading existing model data
        and setting up background tasks.
        """
        _LOGGER.info("Initializing ML Predictor...")
        init_success = False
        try:
            _ensure_numpy() # Check if numpy is available early

            # 1. Load Learned Weights
            loaded_weights = await self.data_manager.get_learned_weights()
            if loaded_weights:
                self.current_weights = loaded_weights
                self.current_accuracy = loaded_weights.accuracy
                self.training_samples = loaded_weights.training_samples
                try:
                     self.last_training_time = dt_util.parse_datetime(loaded_weights.last_trained)
                except (ValueError, TypeError):
                     _LOGGER.warning("Could not parse last_trained timestamp from weights file.")
                     self.last_training_time = None

                self.model_loaded = True
                
                # [KORREKTUR Block 10] Setze Status basierend auf Trainingsdaten
                if self.training_samples >= MIN_TRAINING_DATA_POINTS:
                    self.model_state = ModelState.READY
                else:
                    self.model_state = ModelState.UNINITIALIZED # Bereit, aber nicht trainiert

                if hasattr(loaded_weights, 'feature_means') and loaded_weights.feature_means:
                    try:
                        self.scaler.set_state({
                            'means': loaded_weights.feature_means,
                            'stds': loaded_weights.feature_stds,
                            'is_fitted': True,
                            # KORREKTUR: Feature-Reihenfolge auch laden
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

            # 3. Update Prediction Strategies
            self.prediction_orchestrator.update_strategies(
                weights=self.current_weights,
                profile=self.current_profile,
                accuracy=self.current_accuracy if self.current_accuracy is not None else 0.0
            )
            _LOGGER.debug("Prediction strategies updated.")

            # 4. Load Historical Cache
            asyncio.create_task(self._load_historical_cache())

            # 5. Schedule Background Tasks
            self._schedule_hourly_sampling()
            self._schedule_daily_training_check()

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
        """Loads historical production data into memory for lag feature calculation."""
        _LOGGER.debug("Loading historical production cache...")
        try:
            # Lade stÃƒÂ¼ndliche Samples, da diese die genauesten Daten enthalten
            samples_data = await self.data_manager.get_hourly_samples(days=60) # Lade letzte 60 Tage
            records = samples_data.get('samples', [])

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

                    timestamp = dt_util.parse_datetime(timestamp_str)
                    if not timestamp:
                        _LOGGER.debug(f"Skipping cache record, invalid timestamp: {timestamp_str}")
                        continue
                        
                    # WICHTIG: Zeitstempel sind UTC, konvertiere in LOKALE Zeit fÃƒÂ¼r die Tageszuordnung
                    timestamp_local = dt_util.as_local(timestamp)
                    
                    date_key = timestamp_local.date().isoformat()
                    hour_key = f"{date_key}_{timestamp_local.hour:02d}"

                    # Speichere stÃƒÂ¼ndlichen Wert
                    hourly_productions_cache[hour_key] = actual_kwh
                    # Addiere zum Tagessummen-Cache
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


    async def predict(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        """Generates a solar production prediction for the current context."""
        prediction_start_time = dt_util.utcnow()
        result: PredictionResult | None = None

        try:
            _ensure_numpy()
            if sensor_data is None: sensor_data = {}
            
            # Verwende LOKALE Zeit fÃƒÂ¼r die Feature-Erstellung
            prediction_time_local = dt_util.as_local(prediction_start_time)
            prediction_hour = prediction_time_local.hour # Lokale Stunde

            try:
                # Hole Produktion von GESTERN (Lokalzeit)
                yesterday_dt = prediction_time_local - timedelta(days=1)
                yesterday_key = yesterday_dt.date().isoformat()
                yesterday_total_kwh = self._historical_cache['daily_productions'].get(yesterday_key, 0.0)
                sensor_data['production_yesterday'] = float(yesterday_total_kwh)
                
                # Hole Produktion der LETZTEN STUNDE (Lokalzeit)
                last_hour_dt = prediction_time_local - timedelta(hours=1)
                last_hour_key = f"{last_hour_dt.date().isoformat()}_{last_hour_dt.hour:02d}"
                last_hour_kwh = self._historical_cache['hourly_productions'].get(last_hour_key, 0.0)
                sensor_data['production_last_hour'] = float(last_hour_kwh)
                
            except Exception as e:
                _LOGGER.warning(f"Could not retrieve lag features (yesterday/last_hour) for prediction: {e}")
                sensor_data['production_yesterday'] = 0.0
                sensor_data['production_last_hour'] = 0.0

            features = await self.feature_engineer.extract_features(
                weather_data, sensor_data, prediction_hour
            )

            if self.scaler.is_fitted:
                features_scaled = self.scaler.transform_single(features)
                _LOGGER.debug("Prediction features scaled.")
            else:
                features_scaled = features
                _LOGGER.debug("Scaler not fitted, using raw features for prediction.")

            result = await self.prediction_orchestrator.predict(features_scaled)

            prediction_end_time = dt_util.utcnow()
            duration_ms = (prediction_end_time - prediction_start_time).total_seconds() * 1000
            self._update_performance_metrics(duration_ms, success=True)

            _LOGGER.debug(f"Prediction successful (Method: {result.method}): {result.prediction:.2f} kWh, "
                          f"Confidence: {result.confidence:.2f}, Duration: {duration_ms:.1f}ms")
            return result

        except ImportError:
             _LOGGER.error("Prediction failed: NumPy dependency is missing.")
             await self.error_handler.handle_error(
                 error=MLModelException("Prediction failed due to missing NumPy"),
                 source="ml_predictor", context={}, pipeline_position="predict"
             )
             fallback_result = await self._get_fallback_prediction(dt_util.as_local(prediction_start_time).hour)
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
            fallback_result = await self._get_fallback_prediction(dt_util.as_local(prediction_start_time).hour)
            prediction_end_time = dt_util.utcnow()
            duration_ms = (prediction_end_time - prediction_start_time).total_seconds() * 1000
            self._update_performance_metrics(duration_ms, success=False)
            return fallback_result

    async def _get_fallback_prediction(self, hour: int) -> PredictionResult:
         """Generates a prediction using the fallback strategy."""
         _LOGGER.warning("Using fallback prediction strategy.")
         fallback_strategy = FallbackStrategy()
         default_features = self.feature_engineer.get_default_features(hour)
         return await fallback_strategy.predict(default_features)


    async def train_model(self) -> TrainingResult:
        """Trains the Ridge Regression model using historical hourly data."""
        training_start_time = dt_util.utcnow()
        _LOGGER.info(f"Starting ML model training at {training_start_time}...")
        self.model_state = ModelState.TRAINING
        await self._update_model_state_file(status_override=ModelState.TRAINING)

        training_records = []
        result: TrainingResult | None = None

        try:
            np = _ensure_numpy()
            await self._load_historical_cache() # Lade den Cache, um Lag-Features zu erstellen
            if not self._historical_cache['daily_productions']:
                 _LOGGER.warning("Historical cache for daily productions is empty. 'production_yesterday' feature will be 0.")

            training_records = await self.data_manager.get_all_training_records(days=60)
            valid_records_count = len(training_records)

            if valid_records_count < MIN_TRAINING_DATA_POINTS:
                error_msg = (f"Insufficient training data: {valid_records_count} valid hourly samples found (minimum required: {MIN_TRAINING_DATA_POINTS}). Training aborted.")
                _LOGGER.warning(error_msg)
                self.model_state = ModelState.UNINITIALIZED if not self.model_loaded else ModelState.READY
                result = TrainingResult(success=False, samples_used=valid_records_count, error_message=error_msg)
                return result

            _LOGGER.info(f"Preparing {valid_records_count} records for training...")

            X_train_raw = []
            y_train = []
            for record in training_records:
                weather_data = record.get('weather_data', {})
                sensor_data = record.get('sensor_data', {})
                actual_kwh = record['actual_value']
                try:
                    # KORREKTUR: Verwende LOKALE Zeit fÃƒÂ¼r Lag-Features
                    record_time_utc = dt_util.parse_datetime(record['timestamp'])
                    record_time_local = dt_util.as_local(record_time_utc)
                    
                    # Hole Gestern (Lokal)
                    yesterday_dt = record_time_local - timedelta(days=1)
                    yesterday_key = yesterday_dt.date().isoformat()
                    yesterday_total = self._historical_cache['daily_productions'].get(yesterday_key, 0.0)
                    sensor_data['production_yesterday'] = float(yesterday_total)
                    
                    # Hole Letzte Stunde (Lokal)
                    last_hour_dt = record_time_local - timedelta(hours=1)
                    last_hour_key = f"{last_hour_dt.date().isoformat()}_{last_hour_dt.hour:02d}"
                    last_hour_total = self._historical_cache['hourly_productions'].get(last_hour_key, 0.0)
                    sensor_data['production_last_hour'] = float(last_hour_total)
                    
                except Exception as e:
                    _LOGGER.debug(f"Could not find lag features for record {record['timestamp']}: {e}")
                    sensor_data['production_yesterday'] = 0.0
                    sensor_data['production_last_hour'] = 0.0

                features_dict = self.feature_engineer.extract_features_sync(weather_data, sensor_data, record)
                feature_vector = [features_dict.get(name, 0.0) for name in self.feature_engineer.feature_names]
                X_train_raw.append(feature_vector)
                y_train.append(actual_kwh)

            _LOGGER.debug("Feature extraction complete. Starting scaling and training...")

            X_train_scaled_list = await self.hass.async_add_executor_job(
                 self.scaler.fit_transform, X_train_raw, self.feature_engineer.feature_names
            )
            _LOGGER.info("Scaler fitted and training data transformed (%d features).", len(self.feature_engineer.feature_names))

            weights_dict_raw, bias, accuracy, best_lambda = await self.hass.async_add_executor_job(
                self.trainer.train, X_train_scaled_list, y_train
            )
            _LOGGER.info(f"Ridge training complete. Accuracy (R-squared): {accuracy:.4f}, Best Lambda: {best_lambda:.4f}")

            mapped_weights = self.trainer.map_weights_to_features(weights_dict_raw, self.feature_engineer.feature_names)

            old_weights = await self.data_manager.get_learned_weights()
            current_correction_factor = getattr(old_weights, 'correction_factor', 1.0)
            current_correction_factor = max(CORRECTION_FACTOR_MIN, min(CORRECTION_FACTOR_MAX, current_correction_factor))
            _LOGGER.debug(f"Preserving fallback correction factor: {current_correction_factor:.3f}")

            new_learned_weights = LearnedWeights(
                weights=mapped_weights, bias=bias, accuracy=accuracy,
                training_samples=valid_records_count, last_trained=training_start_time.isoformat(),
                model_version=ML_MODEL_VERSION, feature_names=self.feature_engineer.feature_names,
                feature_means=self.scaler.means, feature_stds=self.scaler.stds,
                correction_factor=current_correction_factor
            )

            await self.data_manager.save_learned_weights(new_learned_weights)
            _LOGGER.info("Updated learned weights and scaler state saved.")

            await self._update_hourly_profile(training_records)

            self.current_weights = new_learned_weights
            self.current_accuracy = accuracy
            self.training_samples = valid_records_count
            self.last_training_time = training_start_time
            self.model_loaded = True
            self.model_state = ModelState.READY

            self.prediction_orchestrator.update_strategies(
                weights=self.current_weights, profile=self.current_profile, accuracy=self.current_accuracy
            )
            _LOGGER.debug("Prediction orchestrator updated.")

            training_end_time = dt_util.utcnow()
            duration_seconds = (training_end_time - training_start_time).total_seconds()

            result = TrainingResult(
                success=True, accuracy=accuracy, samples_used=valid_records_count,
                weights=new_learned_weights, training_time_seconds=duration_seconds,
                feature_count=len(self.feature_engineer.feature_names)
            )

            _LOGGER.info(f"ML Training successful in {duration_seconds:.2f}s. Accuracy={accuracy*100:.1f}%, Samples={valid_records_count}.")
            self.error_handler.log_ml_operation(
                operation="model_training", success=True,
                metrics={"accuracy": accuracy, "samples": valid_records_count, "features": result.feature_count},
                duration_seconds=duration_seconds
            )
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
            duration_seconds = (dt_util.utcnow() - training_start_time).total_seconds()
            self.error_handler.log_ml_operation(operation="model_training", success=False, metrics={"samples": len(training_records)}, context={"error": str(e)}, duration_seconds=duration_seconds)
            return result
        finally:
            # Setze den Status korrekt basierend auf dem Ergebnis
            if result and result.success:
                self.model_state = ModelState.READY
            elif result and not result.success:
                # Wenn das Training fehlschlÃƒÂ¤gt, aber das Modell vorher geladen war, bleibe READY (oder DEGRADED)
                # Setze nur auf UNINITIALIZED, wenn es vorher auch so war
                if not self.model_loaded:
                    self.model_state = ModelState.UNINITIALIZED
                else:
                    self.model_state = ModelState.READY # Behalte alten Status bei
            else:
                self.model_state = ModelState.ERROR # Fallback bei unerwartetem Fehler

            await self._update_model_state_file(
                 status_override=self.model_state.value, # Verwende den finalen Status
                 accuracy_override=result.accuracy if result and result.success else self.current_accuracy,
                 samples_override=result.samples_used if result and result.success else self.training_samples,
                 training_time_override=result.training_time_seconds if result else None
            )


    async def _update_hourly_profile(self, training_records: List[Dict[str, Any]]) -> None:
        """Updates the hourly production profile based on provided training records."""
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
                    
                    # KORREKTUR: Verwende LOKALE Stunde
                    timestamp = dt_util.parse_datetime(timestamp_str) # Ist UTC
                    if not timestamp: continue
                    hour = dt_util.as_local(timestamp).hour # Konvertiere zu lokaler Stunde
                    
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
                last_updated=dt_util.utcnow().isoformat(), confidence=confidence,
            )

            await self.data_manager.save_hourly_profile(new_profile)
            self.current_profile = new_profile

            _LOGGER.info(f"Hourly profile updated successfully using {valid_sample_count} valid samples. Confidence: {confidence:.2f}")

        except ImportError:
             _LOGGER.error("Hourly profile update failed: NumPy dependency is missing!")
        except Exception as e:
            _LOGGER.error(f"Hourly profile update failed: {e}", exc_info=True)
            if self.current_profile is None: self.current_profile = create_default_hourly_profile()


    async def _update_model_state_file(
        self,
        status_override: Optional[ModelState | str] = None,
        accuracy_override: Optional[float] = None,
        samples_override: Optional[int] = None,
        training_time_override: Optional[float] = None
    ) -> None:
        """Safely updates the model_state.json file with current predictor status."""
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
                "model_info": {"version": ML_MODEL_VERSION, "type": "ridge_regression_normalized"},
                "performance_metrics": {
                     "avg_prediction_time_ms": self.performance_metrics.get("avg_prediction_time_ms"),
                     "error_rate": self.performance_metrics.get("error_rate")
                },
                **({"training_time_seconds": round(training_time_override, 2)} if training_time_override is not None else {}),
                "last_updated": dt_util.utcnow().isoformat()
            }

            await self.data_manager.save_model_state(current_state)
            _LOGGER.debug(f"model_state.json updated successfully (Status: {status_str}).")

        except Exception as e:
            _LOGGER.error(f"Failed to update model_state.json: {e}", exc_info=True)


    async def _check_training_data_availability(self) -> int:
        """Checks how many valid hourly samples are available for training."""
        try:
            records_data = await self.data_manager.get_hourly_samples(days=60)
            records = records_data.get('samples', [])
            valid_count = sum(1 for r in records if r.get('actual_kwh') is not None and r['actual_kwh'] > 0)
            _LOGGER.debug(f"Training data check: Found {valid_count} valid hourly samples.")
            return valid_count
        except Exception as e:
            _LOGGER.error(f"Training data availability check failed: {e}", exc_info=True)
            return 0


    def _schedule_hourly_sampling(self) -> None:
        """Schedules the hourly callback for data sampling."""
        if self._hourly_sample_listener_remove: self._hourly_sample_listener_remove()
        self._hourly_sample_listener_remove = async_track_time_change(
            self.hass, self._hourly_learning_callback, minute=2, second=0
        )
        _LOGGER.info("Hourly data sampling scheduled to run at minute 2 of every hour.")

    @callback # Decorator is now defined and imported
    async def _hourly_learning_callback(self, now_local: datetime) -> None:
        """Callback triggered every hour to collect the sample for the previous hour."""
        # 'now_local' ist hier die LOKALE Zeit des Triggers (z.B. 14:02)
        previous_hour_dt = now_local - timedelta(hours=1)
        hour_to_collect = previous_hour_dt.hour # Die Stunde, die geendet hat (z.B. 13)
        _LOGGER.debug(f"Hourly callback triggered at {now_local.strftime('%H:%M:%S')}. Requesting sample collection for hour {hour_to_collect} (Lokal).")
        
        # KRITISCHE KORREKTUR: Verwende sample_collector statt self fÃ¼r collect_sample
        task = asyncio.create_task(self.sample_collector.collect_sample(hour_to_collect))
        
        # FÃ¼ge Error Handler hinzu um zukÃ¼nftige Fehler sichtbar zu machen
        def _handle_task_error(task):
            try:
                task.result()
            except Exception as e:
                _LOGGER.error(f"Fehler bei stÃ¼ndlicher Datensammlung fÃ¼r Stunde {hour_to_collect}: {e}", exc_info=True)
        
        task.add_done_callback(_handle_task_error)


    def _schedule_daily_training_check(self, reschedule_delay_sec: Optional[float] = None) -> None:
        """Schedules the next daily check to see if model retraining is needed."""
        if self._daily_training_task: self._daily_training_task.cancel(); self._daily_training_task = None
        if reschedule_delay_sec is not None:
             delay = reschedule_delay_sec
             _LOGGER.info(f"Scheduling next training check in {delay:.0f} seconds.")
        else:
             now_local = dt_util.as_local(dt_util.utcnow())
             target_time_local = now_local.replace(hour=23, minute=5, second=0, microsecond=0)
             if now_local >= target_time_local: target_time_local += timedelta(days=1)
             delay = (target_time_local - now_local).total_seconds()
             _LOGGER.info(f"Scheduling next daily training check at {target_time_local} (in {delay:.0f} seconds).")

        self._daily_training_task = self.hass.loop.call_later(
            delay, lambda: asyncio.create_task(self._daily_training_check_callback())
        )

    async def _daily_training_check_callback(self) -> None:
        """Callback executed daily to check if retraining is needed and trigger it."""
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
             if not self._stop_event.is_set(): self._schedule_daily_training_check(reschedule_delay_sec=None)


    async def _should_retrain(self) -> bool:
        """Determines if the model should be automatically retrained based on criteria."""
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
            time_since_training = dt_util.utcnow() - self.last_training_time
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


    def _update_performance_metrics(self, prediction_time_ms: float, success: bool) -> None:
        """Updates rolling average prediction time and error rate."""
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


    # --- KORREKTUR (Block 8): set_entities aktualisiert ---
    def set_entities(
        self, power_entity: Optional[str]=None, weather_entity: Optional[str]=None, solar_capacity: Optional[float]=None,
        temp_sensor: Optional[str]=None, wind_sensor: Optional[str]=None, rain_sensor: Optional[str]=None,
        uv_sensor: Optional[str]=None, lux_sensor: Optional[str]=None, 
        humidity_sensor: Optional[str]=None, # <-- NEU
        sun_guard=None
    ) -> None:
        """Configures the entity IDs used by the ML predictor and its components."""
        _LOGGER.info("Configuring entities for MLPredictor and SampleCollector...")
        self.sample_collector.configure_entities(
            weather_entity=weather_entity, power_entity=power_entity, temp_sensor=temp_sensor,
            wind_sensor=wind_sensor, rain_sensor=rain_sensor, uv_sensor=uv_sensor, lux_sensor=lux_sensor,
            humidity_sensor=humidity_sensor # <-- NEU
        )
        if sun_guard: self.sample_collector.sun_guard = sun_guard; _LOGGER.debug("SunGuard instance passed to SampleCollector.")
        
        # Log-Nachricht aktualisiert
        _LOGGER.info(
            "Entities configured in SampleCollector: power=%s, weather=%s, temp=%s, wind=%s, rain=%s, uv=%s, lux=%s, humidity=%s",
            power_entity, weather_entity, temp_sensor, wind_sensor, rain_sensor, uv_sensor, lux_sensor, humidity_sensor
        )
    # --- ENDE KORREKTUR ---


    # --- KORREKTUR (Block 10): is_healthy aktualisiert ---
    def is_healthy(self) -> bool:
        """
        Performs a health check on the ML predictor state.
        The model is only "healthy" if it has been successfully trained 
        with a minimum number of data samples.
        """
        _LOGGER.debug("Performing ML predictor health check...")
        try:
            _ensure_numpy() # PrÃƒÂ¼fe, ob AbhÃƒÂ¤ngigkeit vorhanden ist
            
            # 1. Muss geladen sein
            if not self.model_loaded: 
                _LOGGER.debug("Health check failed: Model not loaded.")
                return False

            # 2. Muss (korrekt) auf 'Ready' stehen
            if self.model_state != ModelState.READY: 
                _LOGGER.debug(f"Health check failed: Model state is {self.model_state.value}, expected READY.")
                return False

            # 3. (NEU) Muss mit genÃƒÂ¼gend Daten trainiert sein
            if self.training_samples < MIN_TRAINING_DATA_POINTS:
                _LOGGER.info(f"Health check: Model is loaded but not sufficiently trained (Samples: {self.training_samples} < {MIN_TRAINING_DATA_POINTS}). "
                             "ML strategy will be skipped (Fallback active).")
                return False # ML ist NICHT gesund/verfÃƒÂ¼gbar

            # 4. (Optional) PrÃƒÂ¼fe Genauigkeit und Alter, aber gebe nur Warnungen aus
            if self.current_accuracy is not None and self.current_accuracy < 0.3: 
                _LOGGER.warning(f"Health check warning: Accuracy is very low ({self.current_accuracy:.3f}). Model might be unreliable.")
            
            max_age = timedelta(days=14)
            if self.last_training_time and (dt_util.utcnow() - self.last_training_time) > max_age: 
                _LOGGER.warning(f"Health check warning: Last training was more than {max_age.days} days ago. Model might be outdated.")
            
            _LOGGER.debug("ML predictor health check passed (Loaded, Ready, Sufficient Samples).")
            return True
            
        except ImportError: 
            _LOGGER.error("Health check failed: NumPy dependency missing.")
            return False
        except Exception as e: 
            _LOGGER.error(f"Error during ML predictor health check: {e}", exc_info=True)
            return False
    # --- ENDE KORREKTUR (Block 10) ---




    async def async_will_remove_from_hass(self) -> None:
        """Clean up resources when the integration is unloaded or HA stops."""
        _LOGGER.info("Cleaning up MLPredictor background tasks and listeners...")
        self._stop_event.set() # Signal background tasks to stop first

        # Cancel listeners and tasks gracefully
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