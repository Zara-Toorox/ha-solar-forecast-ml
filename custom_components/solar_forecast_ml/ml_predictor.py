"""
ML Predictor fÃƒÆ’Ã‚Â¼r die Solar Forecast ML Integration.
Refactored v5.2.0 - Modular Architecture

Copyright (C) 2025 Zara-Toorox

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_change
from homeassistant.util import dt as dt_util

from .const import (
    MIN_TRAINING_DATA_POINTS, MODEL_ACCURACY_THRESHOLD, DATA_VERSION,
    ML_MODEL_VERSION
)
from .data_manager import DataManager
from .ml_types import (
    LearnedWeights, HourlyProfile,
    create_default_learned_weights, create_default_hourly_profile
)
from .typed_data_adapter import TypedDataAdapter
from .error_handling_service import ErrorHandlingService
from .exceptions import MLModelException, ErrorSeverity

from .ml_scaler import StandardScaler
from .ml_feature_engineering import FeatureEngineer
from .ml_training import RidgeTrainer
from .ml_prediction_strategies import (
    PredictionOrchestrator, PredictionResult,
    MLModelStrategy, ProfileStrategy
)
from .ml_sample_collector import SampleCollector

_LOGGER = logging.getLogger(__name__)


def _ensure_numpy():
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
        except ImportError as e:
            _LOGGER.error("NumPy konnte nicht importiert werden: %s", e)
            raise
    return _np


_np = None


class ModelState(Enum):
    UNINITIALIZED = "uninitialized"
    TRAINING = "training"
    READY = "ready"
    TRAINED = "ready"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass
class TrainingResult:
    success: bool
    accuracy: float
    samples_used: int
    weights: Optional[LearnedWeights]
    error_message: Optional[str] = None
    training_time_seconds: Optional[float] = None
    feature_count: Optional[int] = None


@dataclass
class ModelHealth:
    state: ModelState
    model_loaded: bool
    last_training: Optional[datetime]
    current_accuracy: float
    training_samples: int
    features_available: List[str]
    performance_metrics: Dict[str, float]


class MLPredictor:
    
    def __init__(
        self, 
        hass: HomeAssistant, 
        data_manager: DataManager, 
        error_handler: ErrorHandlingService
    ):
        self.hass = hass
        self.data_manager = data_manager
        self.error_handler = error_handler
        
        self.data_adapter = TypedDataAdapter()
        
        self.model_state = ModelState.UNINITIALIZED
        self.model_loaded = False
        self.current_weights: Optional[LearnedWeights] = None
        self.current_profile: Optional[HourlyProfile] = None
        
        self.current_accuracy = 0.0
        self.training_samples = 0
        self.last_training_time: Optional[datetime] = None
        self.prediction_count = 0
        self.successful_predictions = 0
        
        self._time_trigger_unsub = None
        self._stop_event = asyncio.Event()
        self._periodic_training_task_handle: Optional[asyncio.Task] = None
        self._shutdown_registered = False
        
        self._external_sensors = {
            'temp_sensor': None,
            'wind_sensor': None,
            'rain_sensor': None,
            'uv_sensor': None,
            'lux_sensor': None
        }
        
        self._register_shutdown_handler()
        
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.trainer = RidgeTrainer()
        self.prediction_orchestrator = PredictionOrchestrator()
        self.sample_collector = SampleCollector(hass, data_manager)
        
        self.performance_metrics = {
            "avg_prediction_time": 0.0,
            "total_predictions": 0,
            "successful_predictions": 0,
            "error_rate": 0.0
        }
        
        self._historical_cache = {
            'daily_productions': {},
            'hourly_productions': {},
            'weather_history': []
        }
        
        _LOGGER.info("MLPredictor v5.2.0 - Modular Architecture initialisiert")

    async def initialize(self) -> bool:
        try:
            _LOGGER.info("Initialisiere ML Predictor...")
            
            learned_weights = await self.data_manager.get_learned_weights()
            if learned_weights:
                self.current_weights = learned_weights
                self.current_accuracy = learned_weights.accuracy
                self.training_samples = learned_weights.training_samples
                self.model_loaded = True
                self.model_state = ModelState.READY
                
                if hasattr(learned_weights, 'feature_means') and learned_weights.feature_means:
                    self.scaler.set_state({
                        'means': learned_weights.feature_means,
                        'stds': learned_weights.feature_stds,
                        'is_fitted': True
                    })
                    _LOGGER.info("Scaler State geladen: %d Features", len(self.scaler.means))
                
                _LOGGER.info(
                    "Model geladen: Accuracy=%.2f, Samples=%d", 
                    self.current_accuracy, self.training_samples
                )
            
            hourly_profile_dict = await self.data_manager.get_hourly_profile()
            if hourly_profile_dict:
                self.current_profile = self.data_adapter.dict_to_hourly_profile(
                    hourly_profile_dict
                )
            
            self.prediction_orchestrator.update_strategies(
                weights=self.current_weights,
                profile=self.current_profile,
                accuracy=self.current_accuracy
            )
            
            await self._load_historical_cache()
            
            self._periodic_training_task_handle = asyncio.create_task(
                self._periodic_training_task()
            )
            
            self._time_trigger_unsub = async_track_time_change(
                self.hass,
                self._hourly_learning_callback,
                minute=2,
                second=0
            )
            
            _LOGGER.info("ML Predictor erfolgreich initialisiert")
            return True
            
        except Exception as e:
            _LOGGER.error("ML Predictor initialization failed: %s", str(e))
            self.model_state = ModelState.ERROR
            return False

    async def _load_historical_cache(self) -> None:
        try:
            history_data = await self.data_manager.get_prediction_history()
            records = history_data.get('predictions', [])
            
            for record in records[-168:]:
                if record.get('actual_value') is None:
                    continue
                
                timestamp = dt_util.parse_datetime(record['timestamp'])
                date_key = timestamp.date().isoformat()
                hour_key = f"{date_key}_{timestamp.hour}"
                
                if date_key not in self._historical_cache['daily_productions']:
                    self._historical_cache['daily_productions'][date_key] = []
                self._historical_cache['daily_productions'][date_key].append(
                    record['actual_value']
                )
                
                self._historical_cache['hourly_productions'][hour_key] = record['actual_value']
                
                if record.get('weather_data'):
                    self._historical_cache['weather_history'].append({
                        'timestamp': timestamp,
                        'data': record['weather_data']
                    })
            
            _LOGGER.debug(
                "Historical cache geladen: %d Tage, %d Stunden", 
                len(self._historical_cache['daily_productions']),
                len(self._historical_cache['hourly_productions'])
            )
            
        except Exception as e:
            _LOGGER.error("Failed to load historical cache: %s", e)

    async def predict(
        self, 
        weather_data: Dict[str, Any],
        sensor_data: Optional[Dict[str, Any]] = None,
        hour: Optional[int] = None
    ) -> PredictionResult:
        prediction_start = asyncio.get_event_loop().time()
        
        try:
            self.prediction_count += 1
            
            if sensor_data is None:
                sensor_data = {}
            
            if hour is None:
                hour = dt_util.utcnow().hour
            
            features = await self.feature_engineer.extract_features(
                weather_data, sensor_data, hour
            )
            
            if self.scaler.is_fitted:
                features_scaled = self.scaler.transform_single(features)
            else:
                features_scaled = features
            
            result = await self.prediction_orchestrator.predict(features_scaled)
            
            self.successful_predictions += 1
            
            prediction_time = asyncio.get_event_loop().time() - prediction_start
            self._update_performance_metrics(prediction_time)
            
            return result
            
        except Exception as e:
            _LOGGER.error(f"Prediction failed: {e}")
            
            await self.error_handler.handle_error(
                error=MLModelException(f"Prediction failed: {e}"),
                source="ml_predictor",
                context={
                    "weather_data": weather_data,
                    "sensor_data": sensor_data,
                    "hour": hour,
                    "model_state": self.model_state.value
                },
                pipeline_position="predict"
            )
            
            features = self.feature_engineer.get_default_features(hour or 12)
            return await self.prediction_orchestrator.predict(features)

    async def train_model(self) -> TrainingResult:
        training_start_time = asyncio.get_event_loop().time()
        
        try:
            _LOGGER.info("Starte Advanced Model Training mit Feature Normalisierung...")
            self.model_state = ModelState.TRAINING
            
            await self._load_historical_cache()
            
            records = await self.data_manager.get_all_training_records(days=60)
            
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
            
            X_train = []
            y_train = []
            
            for record in valid_records:
                weather_data = record.get('weather_data', {})
                sensor_data = record.get('sensor_data', {})
                
                features = self.feature_engineer.extract_features_sync(
                    weather_data, sensor_data, record
                )
                feature_vector = [
                    features.get(name, 0.0) 
                    for name in self.feature_engineer.feature_names
                ]
                
                X_train.append(feature_vector)
                y_train.append(record['actual_value'])
            
            _LOGGER.info("Training mit %d Samples, Features vor Normalisierung", len(X_train))
            
            X_train_scaled = self.scaler.fit_transform(
                X_train, 
                self.feature_engineer.feature_names
            )
            
            _LOGGER.info("Features normalisiert: means=%s", 
                        {k: f"{v:.2f}" for k, v in list(self.scaler.means.items())[:3]})
            
            weights_dict, bias, accuracy, best_lambda = await self.trainer.train(
                X_train_scaled, y_train
            )
            
            mapped_weights = self.trainer.map_weights_to_features(
                weights_dict,
                self.feature_engineer.feature_names
            )
            
            learned_weights = LearnedWeights(
                weather_weights={
                    k: mapped_weights.get(k, 0.0) 
                    for k in ["temperature", "humidity", "cloudiness", "wind_speed", "pressure"]
                },
                seasonal_factors={
                    "spring": 1.0,
                    "summer": 1.0,
                    "autumn": 1.0,
                    "winter": 1.0
                },
                correction_factor=1.0,
                accuracy=accuracy,
                training_samples=len(valid_records),
                last_trained=dt_util.utcnow().isoformat(),
                model_version=DATA_VERSION,
                bias=bias,
                weights=mapped_weights,
                feature_names=self.feature_engineer.feature_names,
                feature_means=self.scaler.means,
                feature_stds=self.scaler.stds
            )
            
            await self.data_manager.save_learned_weights(learned_weights)
            
            await self._update_hourly_profile(valid_records)
            
            self.current_weights = learned_weights
            self.current_accuracy = accuracy
            self.training_samples = len(valid_records)
            self.last_training_time = dt_util.utcnow()
            self.model_loaded = True
            self.model_state = ModelState.READY
            
            self.prediction_orchestrator.update_strategies(
                weights=self.current_weights,
                profile=self.current_profile,
                accuracy=self.current_accuracy
            )
            
            training_time = asyncio.get_event_loop().time() - training_start_time
            
            await self._update_model_state_after_training(
                accuracy=accuracy,
                samples=len(valid_records),
                training_time=training_time
            )
            
            _LOGGER.info(
                f"Training erfolgreich: Accuracy={accuracy:.4f}, "
                f"Samples={len(valid_records)}, Features={len(self.feature_engineer.feature_names)}, "
                f"Lambda={best_lambda:.4f}, Time={training_time:.2f}s"
            )
            
            self.error_handler.log_ml_operation(
                operation="model_training",
                success=True,
                metrics={
                    "accuracy": accuracy,
                    "samples_used": len(valid_records),
                    "feature_count": len(self.feature_engineer.feature_names),
                    "best_lambda": best_lambda,
                    "normalized": True
                },
                context={
                    "model_state": self.model_state.value,
                    "model_version": DATA_VERSION
                },
                duration_seconds=training_time
            )
            
            return TrainingResult(
                success=True,
                accuracy=accuracy,
                samples_used=len(valid_records),
                weights=learned_weights,
                training_time_seconds=training_time,
                feature_count=len(self.feature_engineer.feature_names)
            )
            
        except Exception as e:
            _LOGGER.error("Model training failed: %s", str(e))
            self.model_state = ModelState.ERROR
            
            await self.error_handler.handle_error(
                error=MLModelException(f"Model training failed: {e}"),
                source="ml_predictor",
                context={
                    "training_samples": len(valid_records) if 'valid_records' in locals() else 0,
                    "feature_count": len(self.feature_engineer.feature_names),
                    "model_state": self.model_state.value
                },
                pipeline_position="train_model"
            )
            
            return TrainingResult(
                success=False,
                accuracy=0.0,
                samples_used=0,
                weights=None,
                error_message=str(e)
            )

    async def _update_hourly_profile(self, records: List[Dict[str, Any]]) -> None:
        try:
            np = _ensure_numpy()
            
            samples_data = await self.data_manager.get_hourly_samples(days=60)
            samples = samples_data.get('samples', [])
            
            if not samples:
                _LOGGER.warning("No hourly samples available for profile update")
                return
            
            hourly_data = {}
            for hour in range(24):
                hourly_data[hour] = []
            
            for sample in samples:
                try:
                    actual_kwh = sample.get('actual_kwh')
                    if actual_kwh is None or actual_kwh <= 0:
                        continue
                    
                    timestamp = dt_util.parse_datetime(sample['timestamp'])
                    hour = timestamp.hour
                    hourly_data[hour].append(actual_kwh)
                    
                except (ValueError, KeyError, TypeError) as e:
                    _LOGGER.debug(f"Skipping invalid sample: {e}")
                    continue
            
            hourly_averages = {}
            total_samples = 0
            for hour, values in hourly_data.items():
                if values:
                    hourly_averages[str(hour)] = float(np.mean(values))
                    total_samples += len(values)
                else:
                    hourly_averages[str(hour)] = 0.0
            
            hourly_profile = HourlyProfile(
                hourly_averages=hourly_averages,
                last_updated=dt_util.utcnow().isoformat(),
                samples_count=total_samples,
                version=DATA_VERSION
            )
            
            profile_dict = self.data_adapter.hourly_profile_to_dict(hourly_profile)
            await self.data_manager.save_hourly_profile(profile_dict)
            
            self.current_profile = hourly_profile
            
            _LOGGER.info(
                f"Hourly profile updated from {len(samples)} samples, "
                f"{total_samples} valid hourly values"
            )
            
        except Exception as e:
            _LOGGER.error(f"Hourly profile update failed: {e}")

    async def _update_model_state_after_training(
        self, 
        accuracy: float, 
        samples: int, 
        training_time: float
    ) -> None:
        try:
            current_state = await self.data_manager.get_model_state()
            
            training_count = current_state.get('training_count', 0) + 1
            
            history_data = await self.data_manager.get_prediction_history()
            records = history_data.get('predictions', [])[-100:]
            
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
            
            if training_time > 3600:
                _LOGGER.warning(
                    f"Unrealistic training_time detected: {training_time:.2f}s (>1h). "
                    "Capping to 3600s"
                )
                training_time = min(training_time, 3600.0)
            
            if accuracy < 0.0:
                _LOGGER.warning(f"Negative accuracy detected: {accuracy:.4f}. Setting to 0.0")
                accuracy = 0.0
            
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
                "training_time_seconds": training_time,
                "last_updated": dt_util.utcnow().isoformat()
            }
            
            await self.data_manager.save_model_state(updated_state)
            
            _LOGGER.info(
                f"model_state aktualisiert: Training #{training_count}, "
                f"MAE={mae:.2f if mae is not None else 'N/A'}, "
                f"RMSE={rmse:.2f if rmse is not None else 'N/A'}, "
                f"Time={training_time:.2f}s"
            )
            
        except Exception as e:
            _LOGGER.error(f"model_state update failed: {e}")

    async def _check_training_data_availability(self) -> bool:
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
        while not self._stop_event.is_set():
            try:
                now = dt_util.utcnow()
                if now.hour == 23 and now.minute < 5:
                    
                    if await self._should_retrain():
                        _LOGGER.info("Starte Periodic Training...")
                        training_result = await self.train_model()
                        
                        if training_result.success:
                            _LOGGER.info("Periodic Training erfolgreich")
                        else:
                            _LOGGER.warning("Periodic Training fehlgeschlagen")
                
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=300
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                    
            except asyncio.CancelledError:
                _LOGGER.info("Periodic training task cancelled")
                break
            except Exception as e:
                _LOGGER.error("Periodic training error: %s", str(e))
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=300
                    )
                    break
                except asyncio.TimeoutError:
                    pass

    async def _hourly_learning_callback(self, now: datetime) -> None:
        await self.sample_collector.collect_sample(now.hour)

    async def _should_auto_train(self) -> bool:
        try:
            history_data = await self.data_manager.get_prediction_history()
            records = history_data.get('predictions', [])
            
            if self.last_training_time:
                new_samples = [
                    r for r in records
                    if r.get('timestamp', '') > self.last_training_time.isoformat()
                ]
                new_sample_count = len(new_samples)
            else:
                new_sample_count = len(records)
            
            if new_sample_count >= 50:
                _LOGGER.info(f"Auto-Training: {new_sample_count} neue Samples verfÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼gbar")
                return True
            
            if self.last_training_time:
                hours_since_training = (dt_util.utcnow() - self.last_training_time).total_seconds() / 3600
                if hours_since_training >= 24:
                    _LOGGER.info(f"Auto-Training: {hours_since_training:.1f}h seit letztem Training")
                    return True
            else:
                if len(records) >= MIN_TRAINING_DATA_POINTS:
                    _LOGGER.info(f"Auto-Training: {len(records)} Samples vorhanden, noch nie trainiert")
                    return True
            
            return False
            
        except Exception as e:
            _LOGGER.error(f"Fehler bei _should_auto_train: {e}")
            return False

    async def _should_retrain(self) -> bool:
        try:
            if not self.model_loaded:
                return True
            
            if self.current_accuracy < MODEL_ACCURACY_THRESHOLD:
                return True
            
            if (self.last_training_time and 
                (dt_util.utcnow() - self.last_training_time) > timedelta(days=7)):
                return True
            
            history_data = await self.data_manager.get_prediction_history()
            records = history_data.get('predictions', [])
            
            valid_records = [
                r for r in records 
                if r.get('actual_value') is not None
            ]
            
            if len(valid_records) > self.training_samples * 1.2:
                return True
            
            return False
            
        except Exception as e:
            _LOGGER.error(f"Should retrain check failed: {e}")
            return False

    def _update_performance_metrics(self, prediction_time: float) -> None:
        if self.prediction_count > 0:
            alpha = 0.1
            self.performance_metrics["avg_prediction_time"] = (
                alpha * prediction_time + 
                (1 - alpha) * self.performance_metrics["avg_prediction_time"]
            )
        else:
            self.performance_metrics["avg_prediction_time"] = prediction_time
        
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
        try:
            timestamp = dt_util.utcnow().isoformat()
            
            sample = {
                "timestamp": timestamp,
                "predicted_value": prediction,
                "actual_value": actual,
                "weather_data": weather_data,
                "sensor_data": sensor_data,
                "accuracy": self.sample_collector._calculate_sample_accuracy(prediction, actual),
                "model_version": ML_MODEL_VERSION
            }
            
            await self.data_manager.add_prediction_record(sample)
            
            if await self._should_auto_train():
                _LOGGER.info("Auto-Training triggered nach neuem Sample")
                await self.train_model()
            
        except Exception as e:
            _LOGGER.error(f"Failed to add training sample: {e}")

    def set_external_sensors(
        self,
        temp_sensor: Optional[str] = None,
        wind_sensor: Optional[str] = None,
        rain_sensor: Optional[str] = None,
        uv_sensor: Optional[str] = None,
        lux_sensor: Optional[str] = None
    ) -> None:
        if temp_sensor:
            self._external_sensors['temp_sensor'] = temp_sensor
            self.sample_collector.temp_sensor = temp_sensor
        if wind_sensor:
            self._external_sensors['wind_sensor'] = wind_sensor
            self.sample_collector.wind_sensor = wind_sensor
        if rain_sensor:
            self._external_sensors['rain_sensor'] = rain_sensor
            self.sample_collector.rain_sensor = rain_sensor
        if uv_sensor:
            self._external_sensors['uv_sensor'] = uv_sensor
            self.sample_collector.uv_sensor = uv_sensor
        if lux_sensor:
            self._external_sensors['lux_sensor'] = lux_sensor
            self.sample_collector.lux_sensor = lux_sensor
        
        _LOGGER.info("Externe Sensoren konfiguriert: %s", self._external_sensors)

    def set_entities(
        self,
        power_entity: Optional[str] = None,
        solar_yield_today: Optional[str] = None,
        weather_entity: Optional[str] = None,
        solar_capacity: Optional[float] = None,
        forecast_cache: Optional[Dict[str, Any]] = None,
        temp_sensor: Optional[str] = None,
        wind_sensor: Optional[str] = None,
        rain_sensor: Optional[str] = None,
        uv_sensor: Optional[str] = None,
        lux_sensor: Optional[str] = None,
        sun_guard = None
    ) -> None:
        self.power_entity = power_entity
        self.solar_yield_today = solar_yield_today
        self.weather_entity = weather_entity
        self.solar_capacity = solar_capacity
        self._forecast_cache = forecast_cache if forecast_cache is not None else {}
        
        if sun_guard is not None:
            self.sample_collector.sun_guard = sun_guard
        
        self.sample_collector.configure_entities(
            weather_entity=weather_entity,
            power_entity=power_entity,
            temp_sensor=temp_sensor,
            wind_sensor=wind_sensor,
            rain_sensor=rain_sensor,
            uv_sensor=uv_sensor,
            lux_sensor=lux_sensor
        )
        self.sample_collector.set_forecast_cache(self._forecast_cache)
        
        self.set_external_sensors(
            temp_sensor=temp_sensor,
            wind_sensor=wind_sensor,
            rain_sensor=rain_sensor,
            uv_sensor=uv_sensor,
            lux_sensor=lux_sensor
        )
        
        _LOGGER.info("Entities konfiguriert: power=%s, weather=%s, capacity=%s", 
                     power_entity, weather_entity, solar_capacity)

    def is_healthy(self) -> bool:
        try:
            if not self.model_loaded:
                return False
            
            if self.training_samples < 3:
                return False
            
            if self.model_state not in [ModelState.READY, ModelState.TRAINED]:
                return False
            
            if self.current_accuracy < 0.5:
                return False
            
            return True
            
        except Exception:
            return False

    def _register_shutdown_handler(self) -> None:
        if not self._shutdown_registered:
            from homeassistant.const import EVENT_HOMEASSISTANT_STOP
            self.hass.bus.async_listen_once(
                EVENT_HOMEASSISTANT_STOP,
                self._handle_shutdown
            )
            self._shutdown_registered = True
            _LOGGER.debug("Shutdown handler registered")

    async def _handle_shutdown(self, event) -> None:
        await self.async_will_remove_from_hass()

    async def async_will_remove_from_hass(self) -> None:
        try:
            _LOGGER.info("MLPredictor Cleanup gestartet...")
            
            self._stop_event.set()
            
            if self._time_trigger_unsub:
                self._time_trigger_unsub()
                self._time_trigger_unsub = None
                _LOGGER.debug("Time trigger unsubscribed")
            
            if self._periodic_training_task_handle and not self._periodic_training_task_handle.done():
                self._periodic_training_task_handle.cancel()
                try:
                    await asyncio.wait_for(self._periodic_training_task_handle, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                _LOGGER.debug("Periodic training task stopped")
            
            _LOGGER.info("MLPredictor Cleanup abgeschlossen")
            
        except Exception as e:
            _LOGGER.error(f"Fehler beim MLPredictor Cleanup: {e}")
