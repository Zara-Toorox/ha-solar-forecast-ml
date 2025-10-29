"""
ML Predictor for the Solar Forecast ML Integration.
REFACTORED VERSION: Modular structure with separate manager classes
Version 6.0.0 - Modular Architecture

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
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_change, async_call_later
from homeassistant.components.recorder import get_instance

from .const import (
    MIN_TRAINING_DATA_POINTS, MODEL_ACCURACY_THRESHOLD, DATA_VERSION,
    ML_MODEL_VERSION
)
from .data_manager import DataManager
from .helpers import SafeDateTimeUtil as dt_util
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
            _LOGGER.error("NumPy could not be imported: %s", e)
            raise
    return _np


_np = None


class ModelState(Enum):
    UNINITIALIZED = "uninitialized"
    TRAINING = "training"
    READY = "ready"
    TRAINED = "ready"  # Alias for READY
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

        _LOGGER.info("MLPredictor v5.2.7 - Modular Architecture initialized")

    # === HELFER-FUNKTIONEN FÜR BENACHRICHTIGUNGEN ===
    async def _send_notification(self, notification_id: str, title: str, message: str):
        """Sendet eine persistente Benachrichtigung über den HA Service Call."""
        try:
            await self.hass.services.async_call(
                domain="persistent_notification",
                service="create",
                service_data={
                    "message": message,
                    "title": title,
                    "notification_id": notification_id,
                },
                blocking=False,
            )
        except Exception as e:
            _LOGGER.error(f"Konnte Benachrichtigung '{notification_id}' nicht senden: {e}")

    async def _dismiss_notification(self, notification_id: str):
        """Entfernt eine persistente Benachrichtigung."""
        try:
            await self.hass.services.async_call(
                domain="persistent_notification",
                service="dismiss",
                service_data={
                    "notification_id": notification_id,
                },
                blocking=False,
            )
        except Exception as e:
            _LOGGER.warning(f"Konnte Benachrichtigung '{notification_id}' nicht entfernen: {e}")
    # === ENDE HELFER-FUNKTIONEN ===

    async def initialize(self) -> bool:
        try:
            _LOGGER.info("Initializing ML Predictor...")

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
                    _LOGGER.info("Scaler State loaded: %d Features", len(self.scaler.means))

                _LOGGER.info(
                    "Model loaded: Accuracy=%.2f, Samples=%d",
                    self.current_accuracy, self.training_samples
                )

            hourly_profile_dict = await self.data_manager.get_hourly_profile()
            if hourly_profile_dict:
                self.current_profile = self.data_adapter.dict_to_hourly_profile(
                    hourly_profile_dict
                )
            else:
                self.current_profile = create_default_hourly_profile()

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

            _LOGGER.info("ML Predictor successfully initialized")
            return True

        except Exception as e:
            _LOGGER.error("ML Predictor initialization failed: %s", str(e), exc_info=True)
            self.model_state = ModelState.ERROR
            return False

    # === EINMALIGER BACKFILL (30 TAGE) ===
    async def async_run_backfill_process(self) -> bool:
        """
        EINMALIGER BACKFILL (Version 5 - 30 TAGE)
        """
        # === HIER DIE ÄNDERUNG ===
        BACKFILL_DAYS = 30
        # === ENDE ÄNDERUNG ===
        
        START_HOUR = 7
        END_HOUR = 19
        HOURS_TO_COLLECT = list(range(START_HOUR, END_HOUR))
        
        NOTIFICATION_ID = "solar_forecast_ml_backfill"

        if self.model_state == ModelState.TRAINING:
            _LOGGER.warning("Backfill abgebrochen: Training läuft bereits.")
            return False
            
        if await self.data_manager.has_backfill_run():
            _LOGGER.info("Backfill bereits durchgeführt – wird übersprungen.")
            await self._dismiss_notification(NOTIFICATION_ID)
            await self._send_notification(
                NOTIFICATION_ID,
                "Solar Forecast ML - Backfill Übersprungen",
                "Der Backfill-Prozess wurde bereits in der Vergangenheit ausgeführt. "
                "Um ihn zu erzwingen, muss die Datei `model_state.json` manuell bearbeitet werden (`backfill_run: false`)."
            )
            return True

        _LOGGER.info(f"START EINMALIGER BACKFILL: {BACKFILL_DAYS} Tage, {START_HOUR}:00 – {END_HOUR-1}:59 Uhr")
        
        await self._dismiss_notification(NOTIFICATION_ID)
        await self._send_notification(
            NOTIFICATION_ID,
            "Solar Forecast ML - Backfill Läuft",
            f"Datensammlung für {BACKFILL_DAYS} Tage ({BACKFILL_DAYS * len(HOURS_TO_COLLECT)} Stunden) wird vorbereitet...\n\n"
            "Dieser Vorgang kann einige Minuten dauern. Bitte habe Geduld."
        )

        try:
            today_utc = dt_util.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            total_collected = 0
            total_failed = 0

            for day_offset in range(1, BACKFILL_DAYS + 1):
                
                target_day_timestamp = today_utc - timedelta(days=day_offset)
                day_str = target_day_timestamp.strftime("%Y-%m-%d")

                _LOGGER.info(f"Backfill Tag: {day_str} ({len(HOURS_TO_COLLECT)} Stunden)")

                await self._send_notification(
                    NOTIFICATION_ID,
                    "Solar Forecast ML - Backfill Läuft",
                    f"Sammle Daten für Tag {day_offset}/{BACKFILL_DAYS} ({day_str})...\n\n"
                    f"Bisher gesammelt: {total_collected} Stunden."
                )

                for hour in HOURS_TO_COLLECT:
                    try:
                        _LOGGER.info(f"=== Backfill: Sammle {day_str} Stunde {hour} ===")
                        
                        await self.sample_collector.collect_sample(
                            current_hour=hour,
                            base_timestamp=target_day_timestamp
                        )
                        
                        total_collected += 1
                        await asyncio.sleep(0.5) 

                    except Exception as e:
                        _LOGGER.error(f"Fehler bei {day_str} {hour:02d}:00 → {e}", exc_info=True)
                        total_failed += 1
            
            _LOGGER.info(f"BACKFILL DATENSAMMLUNG ABGESCHLOSSEN: {total_collected} Stunden-Samples angefragt, {total_failed} fehlgeschlagen.")

            await self.data_manager.set_backfill_run(True)

            await self._dismiss_notification(NOTIFICATION_ID)
            await self._send_notification(
                NOTIFICATION_ID,
                "Solar Forecast ML - Backfill Abgeschlossen",
                f"Der Backfill ist abgeschlossen.\n\n"
                f"**Erfolgreich gesammelt:** {total_collected} Stunden\n"
                f"**Fehlgeschlagen:** {total_failed} Stunden\n\n"
                "Du kannst jetzt das 'Manual Learning' (Training) starten, um diese Daten zu verwenden."
            )
            
            return True

        except Exception as e:
            _LOGGER.error(f"Kritischer Backfill-Fehler: {e}", exc_info=True)
            
            await self._dismiss_notification(NOTIFICATION_ID)
            await self._send_notification(
                NOTIFICATION_ID,
                "Solar Forecast ML - Backfill FEHLER",
                f"Der Backfill-Prozess ist fehlgeschlagen: {e}"
            )
            
            return False
    # === ENDE BACKFILL ===

    async def _load_historical_cache(self) -> None:
        try:
            history_data = await self.data_manager.get_prediction_history()
            records = history_data.get('predictions', [])

            daily_prod_temp = {}

            for record in records: # Iteriere über alle, nicht nur die letzten 168
                if record.get('actual_value') is None:
                    continue

                try:
                    timestamp = dt_util.parse_datetime(record['timestamp'])
                    date_key = timestamp.date().isoformat()
                    hour_key = f"{date_key}_{timestamp.hour}"

                    # === START PATCH 4: Cache-Logik für daily_productions ===
                    # Wir summieren die stündlichen 'actual_value' (Ertrag der Stunde)
                    # zu einem Tagesgesamtwert.
                    if date_key not in daily_prod_temp:
                        daily_prod_temp[date_key] = 0.0
                    
                    # Addiere den stündlichen Ertrag zum Tag
                    daily_prod_temp[date_key] += record['actual_value']
                    # === ENDE PATCH 4 ===

                    self._historical_cache['hourly_productions'][hour_key] = record['actual_value']

                    if record.get('weather_data'):
                        self._historical_cache['weather_history'].append({
                            'timestamp': timestamp,
                            'data': record['weather_data']
                        })
                except (ValueError, KeyError, TypeError):
                    _LOGGER.debug(f"Skipping invalid record for cache: {record.get('timestamp')}")
                    continue

            # === START PATCH 4: Speichere die summierten Tageswerte ===
            self._historical_cache['daily_productions'] = daily_prod_temp
            # === ENDE PATCH 4 ===

            _LOGGER.debug(
                "Historical cache loaded: %d days, %d hours",
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

            # === START PATCH 4: ML-Kern-Upgrade (Lag Features) ===
            # Füge 'production_yesterday' zu sensor_data hinzu
            try:
                np = _ensure_numpy()
                yesterday_dt = dt_util.utcnow() - timedelta(days=1)
                yesterday_key = yesterday_dt.date().isoformat()
                
                yesterday_total = self._historical_cache['daily_productions'].get(yesterday_key, 0.0)
                
                sensor_data['production_yesterday'] = float(yesterday_total)
                
                # 'production_last_hour' wird für die Echtzeit-Prognose (noch) nicht verwendet,
                # da wir den Ertrag der letzten Stunde (z.B. 13:00-14:00) erst um 14:02 Uhr kennen,
                # die Prognose aber um 14:00 Uhr erstellen.
                # Wir setzen es auf 0.0, um die Signatur zu erfüllen.
                # Zukünftige Versionen könnten dies aus dem Cache (hourly_productions) holen.
                sensor_data['production_last_hour'] = 0.0 
                
            except Exception as e:
                _LOGGER.warning(f"Could not load lag features for prediction: {e}")
                sensor_data['production_yesterday'] = 0.0
                sensor_data['production_last_hour'] = 0.0
            # === ENDE PATCH 4 ===

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
            _LOGGER.error(f"Prediction failed: {e}", exc_info=True)

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
        valid_records = []
        
        NOTIFICATION_ID = "solar_forecast_ml_training"
        await self._dismiss_notification(NOTIFICATION_ID)
        await self._send_notification(
            NOTIFICATION_ID,
            "Solar Forecast ML - Training Läuft",
            "Das manuelle Training des ML-Modells wurde gestartet. "
            "Dies kann je nach Datenmenge einige Minuten dauern."
        )

        try:
            _LOGGER.info("Starting Advanced Model Training with Feature Normalization...")
            self.model_state = ModelState.TRAINING
            
            # Lade den Cache neu, um sicherzustellen, dass 'daily_productions' aktuell ist
            await self._load_historical_cache()

            # Lade stündliche Samples (die 'production_last_hour' enthalten)
            samples_data = await self.data_manager.get_hourly_samples(days=60)
            records = samples_data.get('samples', [])

            valid_records = [
                r for r in records
                if r.get('actual_kwh') is not None and r.get('actual_kwh') > 0
            ]

            if len(valid_records) < MIN_TRAINING_DATA_POINTS:
                error_msg = (
                    f"Insufficient training data: {len(valid_records)} valid samples found "
                    f"(minimum required: {MIN_TRAINING_DATA_POINTS})"
                )
                _LOGGER.warning(error_msg)
                self.model_state = ModelState.UNINITIALIZED if not self.model_loaded else ModelState.READY
                
                await self._dismiss_notification(NOTIFICATION_ID)
                await self._send_notification(
                    NOTIFICATION_ID,
                    "Solar Forecast ML - Training Fehlgeschlagen",
                    f"Training fehlgeschlagen: {error_msg}. "
                    "Bitte führe zuerst den Backfill aus oder warte, bis mehr Daten gesammelt wurden."
                )
                
                return TrainingResult(
                    success=False,
                    accuracy=self.current_accuracy,
                    samples_used=len(valid_records),
                    weights=self.current_weights,
                    error_message=error_msg
                )

            X_train = []
            y_train = []
            
            # === START PATCH 4: ML-Kern-Upgrade (Lag Features) ===
            # Stelle sicher, dass numpy für die Summierung geladen ist
            np = _ensure_numpy()
            _LOGGER.info(f"Training uses 'daily_productions' cache with {len(self._historical_cache['daily_productions'])} entries.")
            # === ENDE PATCH 4 ===

            for record in valid_records:
                weather_data = record.get('weather_data', {})
                sensor_data = record.get('sensor_data', {}) # Enthält 'production_last_hour'

                # === START PATCH 4: ML-Kern-Upgrade (Lag Features) ===
                # Füge 'production_yesterday' zu sensor_data hinzu, bevor es an den Feature Engineer geht
                try:
                    record_time = dt_util.parse_datetime(record['timestamp'])
                    yesterday_dt = record_time - timedelta(days=1)
                    yesterday_key = yesterday_dt.date().isoformat()
                    
                    # Hole den summierten Wert aus dem Cache
                    yesterday_total = self._historical_cache['daily_productions'].get(yesterday_key, 0.0)
                    
                    sensor_data['production_yesterday'] = float(yesterday_total)
                    
                except Exception as e:
                    _LOGGER.debug(f"Could not find yesterday's production for {record['timestamp']}: {e}")
                    sensor_data['production_yesterday'] = 0.0
                # === ENDE PATCH 4 ===

                features = self.feature_engineer.extract_features_sync(
                    weather_data, sensor_data, record
                )
                feature_vector = [
                    features.get(name, 0.0)
                    for name in self.feature_engineer.feature_names
                ]

                X_train.append(feature_vector)
                y_train.append(record['actual_kwh']) # Ziel ist der stündliche Ertrag

            _LOGGER.info("Training with %d Samples, Features before normalization: %s",
                         len(X_train), self.feature_engineer.feature_names)

            X_train_scaled = self.scaler.fit_transform(
                X_train,
                self.feature_engineer.feature_names
            )

            _LOGGER.info("Features normalized: means=%s...",
                         {k: f"{v:.2f}" for k, v in list(self.scaler.means.items())[:3]})

            weights_dict, bias, accuracy, best_lambda = await self.hass.async_add_executor_job(
                self.trainer.train,
                X_train_scaled,
                y_train
            )

            mapped_weights = self.trainer.map_weights_to_features(
                weights_dict,
                self.feature_engineer.feature_names
            )
            
            # Lade die alten Gewichte, um den correction_factor zu erhalten
            old_weights = await self.data_manager.get_learned_weights()
            current_correction_factor = 1.0
            if old_weights:
                current_correction_factor = old_weights.correction_factor
                _LOGGER.debug(f"Preserving existing correction_factor: {current_correction_factor}")

            learned_weights = LearnedWeights(
                weather_weights={
                    k: mapped_weights.get(k, 0.0)
                    for k in ["temperature", "humidity", "cloudiness", "wind_speed", "pressure"]
                },
                seasonal_factors={"spring": 1.0, "summer": 1.0, "autumn": 1.0, "winter": 1.0},
                correction_factor=current_correction_factor, # Behalte den gelernten Fallback-Faktor
                accuracy=accuracy,
                training_samples=len(valid_records),
                last_trained=dt_util.utcnow().isoformat(),
                model_version=ML_MODEL_VERSION,
                bias=bias,
                weights=mapped_weights,
                feature_names=self.feature_engineer.feature_names,
                feature_means=self.scaler.means,
                feature_stds=self.scaler.stds
            )

            await self.data_manager.save_learned_weights(learned_weights)

            # HINWEIS: Das Training basiert jetzt auf 'hourly_samples'.
            # Der Aufruf von _update_hourly_profile ist redundant, da es dieselben Daten verwendet.
            # Wir behalten es bei, falls die Logik zukünftig divergiert.
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
                f"Training successful: Accuracy={accuracy:.4f}, "
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
                    "model_version": ML_MODEL_VERSION
                },
                duration_seconds=training_time
            )
            
            await self._dismiss_notification(NOTIFICATION_ID)
            await self._send_notification(
                NOTIFICATION_ID,
                "Solar Forecast ML - Training Erfolgreich",
                f"Das ML-Training wurde erfolgreich abgeschlossen.\n\n"
                f"**Neue Genauigkeit:** {accuracy*100:.1f}%\n"
                f"**Verwendete Daten:** {len(valid_records)} Samples\n"
                f"**Dauer:** {training_time:.2f} Sekunden"
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
            _LOGGER.error("Model training failed: %s", str(e), exc_info=True)
            self.model_state = ModelState.ERROR
            
            await self._dismiss_notification(NOTIFICATION_ID)
            await self._send_notification(
                NOTIFICATION_ID,
                "Solar Forecast ML - Training FEHLER",
                f"Ein unerwarteter Fehler ist beim Training aufgetreten:\n\n"
                f"{e}"
            )

            await self.error_handler.handle_error(
                error=MLModelException(f"Model training failed: {e}"),
                source="ml_predictor",
                context={
                    "training_samples": len(valid_records),
                    "feature_count": len(self.feature_engineer.feature_names),
                    "model_state": self.model_state.value
                },
                pipeline_position="train_model"
            )

            training_time_failed = asyncio.get_event_loop().time() - training_start_time
            self.error_handler.log_ml_operation(
                operation="model_training",
                success=False,
                metrics={"accuracy": 0.0, "samples_used": len(valid_records)},
                context={"error": str(e)},
                duration_seconds=training_time_failed
            )

            return TrainingResult(
                success=False,
                accuracy=0.0,
                samples_used=len(valid_records),
                weights=None,
                error_message=str(e),
                training_time_seconds=training_time_failed
            )

    async def _update_hourly_profile(self, records: List[Dict[str, Any]]) -> None:
        try:
            np = _ensure_numpy()

            # Wir verwenden die bereits geladenen 'records' statt 'get_hourly_samples' erneut aufzurufen
            samples = records

            if not samples:
                _LOGGER.warning("No hourly samples available for profile update")
                return

            hourly_data = {hour: [] for hour in range(24)}
            valid_sample_count = 0

            for sample in samples:
                try:
                    actual_kwh = sample.get('actual_kwh')
                    if actual_kwh is None or actual_kwh <= 0:
                        continue

                    timestamp = dt_util.parse_datetime(sample['timestamp'])
                    hour = timestamp.hour
                    hourly_data[hour].append(actual_kwh)
                    valid_sample_count += 1

                except (ValueError, KeyError, TypeError) as e:
                    _LOGGER.debug(f"Skipping invalid hourly sample: {e}")
                    continue

            if valid_sample_count == 0:
                _LOGGER.warning("No valid hourly samples found. Skipping save.")
                if self.current_profile is None:
                    self.current_profile = create_default_hourly_profile()
                return

            hourly_averages = {}
            for hour, values in hourly_data.items():
                if values:
                    hourly_averages[str(hour)] = float(np.median(values))
                else:
                    hourly_averages[str(hour)] = 0.0

            new_profile = HourlyProfile(
                hourly_factors={},
                hourly_averages=hourly_averages,
                last_updated=dt_util.utcnow().isoformat(),
                samples_count=valid_sample_count,
                confidence=min(1.0, max(0.1, valid_sample_count / 100.0)),
            )

            await self.data_manager.save_hourly_profile(new_profile)
            self.current_profile = new_profile

            _LOGGER.info(
                f"Hourly profile updated from {len(samples)} total samples, "
                f"{valid_sample_count} valid hourly values used."
            )

        except Exception as e:
            _LOGGER.error(f"Hourly profile update failed: {e}", exc_info=True)

    async def _update_model_state_after_training(
        self,
        accuracy: float,
        samples: int,
        training_time: float
    ) -> None:
        try:
            current_state = await self.data_manager.get_model_state()
            training_count = current_state.get('training_count', 0) + 1

            mae = None
            rmse = None
            try:
                # === START PATCH 4: MSE/RMSE-Berechnung ===
                # Da das Training jetzt stündlich ist, müssen wir die täglichen 
                # Vorhersagen aus der 'prediction_history' laden, nicht die 'hourly_samples'.
                history_data = await self.data_manager.get_prediction_history()
                records = history_data.get('predictions', [])[-100:]
                # === ENDE PATCH 4 ===

                errors = []
                for r in records:
                    pred = r.get('predicted_value', 0.0)
                    actual = r.get('actual_value')
                    if actual is not None and actual > 0:
                        errors.append(abs(pred - actual))

                if len(errors) >= 10:
                    np = _ensure_numpy()
                    mae = float(np.mean(errors))
                    rmse = float(np.sqrt(np.mean([e**2 for e in errors])))
            except Exception as metric_e:
                _LOGGER.warning(f"Could not calculate MAE/RMSE: {metric_e}")

            if training_time > 3600:
                _LOGGER.warning(f"Unrealistic training_time: {training_time:.2f}s. Capping to 3600s")
                training_time = 3600.0
            if accuracy < 0.0:
                _LOGGER.warning(f"Negative accuracy: {accuracy:.4f}. Setting to 0.0")
                accuracy = 0.0
            if accuracy > 1.0:
                _LOGGER.warning(f"Accuracy > 1.0: {accuracy:.4f}. Setting to 1.0")
                accuracy = 1.0

            updated_state = {
                "version": DATA_VERSION,
                "created": current_state.get('created', dt_util.utcnow().isoformat()),
                "last_training": dt_util.utcnow().isoformat(),
                "training_count": training_count,
                "current_accuracy": accuracy,
                "training_samples": samples,
                "training_time_seconds": round(training_time, 2),
                "performance_metrics": {
                    "mae": round(mae, 4) if mae is not None else None,
                    "rmse": round(rmse, 4) if rmse is not None else None,
                },
                "status": "ready",
                # === HIER IST DER FIX ===
                "backfill_run": current_state.get("backfill_run", False),
                # === ENDE FIX ===
                "model_info": {
                    "version": ML_MODEL_VERSION,
                    "type": "ridge_regression"
                },
                "last_updated": dt_util.utcnow().isoformat()
            }

            await self.data_manager.save_model_state(updated_state)

            _LOGGER.info(
                f"model_state.json updated: Training #{training_count}, Accuracy={accuracy:.4f}, "
                f"MAE={mae:.2f if mae is not None else 'N/A'}, "
                f"RMSE={rmse:.2f if rmse is not None else 'N/A'}, "
                f"Samples={samples}, Time={training_time:.2f}s"
            )

        except Exception as e:
            _LOGGER.error(f"model_state.json update failed: {e}", exc_info=True)

    async def _check_training_data_availability(self) -> bool:
        try:
            # === START PATCH 4: Prüfung auf stündliche Samples ===
            records_data = await self.data_manager.get_hourly_samples(days=60)
            records = records_data.get('samples', [])
            valid_records = [r for r in records if r.get('actual_kwh') is not None and r.get('actual_kwh') > 0]
            # === ENDE PATCH 4 ===
            
            count = len(valid_records)
            _LOGGER.debug(f"Training data check: Found {count} valid samples (min required: {MIN_TRAINING_DATA_POINTS})")
            return count >= MIN_TRAINING_DATA_POINTS
        except Exception as e:
            _LOGGER.error(f"Training data availability check failed: {e}", exc_info=True)
            return False

    async def _periodic_training_task(self) -> None:
        _LOGGER.info("Starting periodic training check task (runs daily around 23:00)")
        while not self._stop_event.is_set():
            try:
                now = dt_util.now()
                if now.hour == 23 and now.minute < 5:
                    _LOGGER.info("Performing daily check: Should retrain model?")
                    if await self._should_retrain():
                        _LOGGER.info(">>> Starting Scheduled Periodic Training <<<")
                        training_result = await self.train_model()
                        if training_result.success:
                            _LOGGER.info("<<< Scheduled Periodic Training successful >>>")
                        else:
                            _LOGGER.warning("<<< Scheduled Periodic Training failed: %s >>>", training_result.error_message)
                    else:
                        _LOGGER.info("No periodic retraining needed at this time.")

                next_check = now.replace(hour=23, minute=0, second=0, microsecond=0)
                if now >= next_check:
                    next_check += timedelta(days=1)
                wait_seconds = (next_check - now).total_seconds()

                _LOGGER.debug(f"Periodic training task sleeping for {wait_seconds:.0f} seconds (until {next_check})")

                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=wait_seconds)
                    break
                except asyncio.TimeoutError:
                    pass

            except asyncio.CancelledError:
                _LOGGER.info("Periodic training task cancelled")
                break
            except Exception as e:
                _LOGGER.error("Error in periodic training task: %s", str(e), exc_info=True)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=300)
                    break
                except asyncio.TimeoutError:
                    pass

    async def _hourly_learning_callback(self, now: datetime) -> None:
        previous_hour_dt = now - timedelta(hours=1)
        hour_to_collect = previous_hour_dt.hour

        _LOGGER.debug(f"Hourly callback triggered at {now.strftime('%H:%M:%S')}. Requesting sample collection for hour {hour_to_collect}.")
        asyncio.create_task(self.sample_collector.collect_sample(hour_to_collect))

    async def _should_auto_train(self) -> bool:
        return False

    async def _should_retrain(self) -> bool:
        try:
            if not self.model_loaded or not self.last_training_time:
                data_available = await self._check_training_data_availability()
                _LOGGER.debug(f"Retrain check: Model not loaded/trained. Data available: {data_available}")
                return data_available

            if self.current_accuracy < MODEL_ACCURACY_THRESHOLD * 0.8:
                _LOGGER.info(f"Retrain check: Accuracy low ({self.current_accuracy:.2f} < {MODEL_ACCURACY_THRESHOLD * 0.8:.2f})")
                return True

            time_since_training = dt_util.utcnow() - self.last_training_time
            if time_since_training > timedelta(days=7):
                _LOGGER.info(f"Retrain check: Last training > 7 days ago ({time_since_training})")
                return True

            # === START PATCH 4: Prüfung auf stündliche Samples ===
            records_data = await self.data_manager.get_hourly_samples(days=60)
            records = records_data.get('samples', [])
            valid_records = [r for r in records if r.get('actual_kwh') is not None and r.get('actual_kwh') > 0]
            # === ENDE PATCH 4 ===
            current_total_samples = len(valid_records)

            if current_total_samples > self.training_samples * 1.5:
                _LOGGER.info(f"Retrain check: Significant new data ({current_total_samples} > {self.training_samples} * 1.5)")
                return True

            _LOGGER.debug("Retrain check: No retraining criteria met.")
            return False

        except Exception as e:
            _LOGGER.error(f"Error during _should_retrain check: {e}", exc_info=True)
            return False

    def _update_performance_metrics(self, prediction_time: float) -> None:
        current_avg = self.performance_metrics.get("avg_prediction_time", 0.0)
        alpha = 0.1
        new_avg = alpha * prediction_time + (1 - alpha) * current_avg
        self.performance_metrics["avg_prediction_time"] = new_avg

        self.performance_metrics["total_predictions"] = self.prediction_count
        self.performance_metrics["successful_predictions"] = self.successful_predictions

        if self.prediction_count > 0:
            error_rate = 1.0 - (self.successful_predictions / self.prediction_count)
            self.performance_metrics["error_rate"] = error_rate
        else:
            self.performance_metrics["error_rate"] = 0.0

    async def add_training_sample(
        self,
        prediction: float,
        actual: float,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any]
    ) -> None:
        try:
            timestamp = dt_util.utcnow().isoformat()

            accuracy = 0.0
            if actual > 0.1:
                error = abs(prediction - actual)
                accuracy = max(0.0, 1.0 - (error / actual))

            sample = {
                "timestamp": timestamp,
                "predicted_value": prediction,
                "actual_value": actual,
                "weather_data": weather_data,
                "sensor_data": sensor_data,
                "accuracy": accuracy,
                "model_version": ML_MODEL_VERSION
            }

            # === START PATCH 4: Speichern in hourly_samples ===
            # Manuelle Samples werden jetzt als stündliche Samples gespeichert,
            # damit sie vom neuen Trainingsprozess erfasst werden.
            # Wir benennen 'predicted_value' in 'actual_kwh' um, da dies der wahre Wert ist.
            hourly_sample = {
                "timestamp": timestamp,
                "actual_kwh": actual,
                "daily_total": 0.0, # Nicht bekannt im manuellen Modus
                "percentage_of_day": 0.0,
                "weather_data": weather_data,
                "sensor_data": sensor_data,
                "model_version": ML_MODEL_VERSION
            }
            await self.data_manager.add_hourly_sample(hourly_sample)
            _LOGGER.info(f"Manually added training sample as hourly_sample: Actual={actual:.2f}")
            # === ENDE PATCH 4 ===

        except Exception as e:
            _LOGGER.error(f"Failed to add manual training sample: {e}", exc_info=True)

    def set_external_sensors(
        self,
        temp_sensor: Optional[str] = None,
        wind_sensor: Optional[str] = None,
        rain_sensor: Optional[str] = None,
        uv_sensor: Optional[str] = None,
        lux_sensor: Optional[str] = None
    ) -> None:
        updated_sensors = {}
        if temp_sensor:
            self._external_sensors['temp_sensor'] = temp_sensor
            self.sample_collector.temp_sensor = temp_sensor
            updated_sensors['temp'] = temp_sensor
        if wind_sensor:
            self._external_sensors['wind_sensor'] = wind_sensor
            self.sample_collector.wind_sensor = wind_sensor
            updated_sensors['wind'] = wind_sensor
        if rain_sensor:
            self._external_sensors['rain_sensor'] = rain_sensor
            self.sample_collector.rain_sensor = rain_sensor
            updated_sensors['rain'] = rain_sensor
        if uv_sensor:
            self._external_sensors['uv_sensor'] = uv_sensor
            self.sample_collector.uv_sensor = uv_sensor
            updated_sensors['uv'] = uv_sensor
        if lux_sensor:
            self._external_sensors['lux_sensor'] = lux_sensor
            self.sample_collector.lux_sensor = lux_sensor
            updated_sensors['lux'] = lux_sensor

        _LOGGER.info("External sensors configured in MLPredictor/SampleCollector: %s", updated_sensors)

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
        sun_guard=None
    ) -> None:
        self.power_entity = power_entity
        self.solar_yield_today = solar_yield_today
        self.weather_entity = weather_entity
        self.solar_capacity = solar_capacity
        self._forecast_cache = forecast_cache if forecast_cache is not None else {}

        self.sample_collector.configure_entities(
            weather_entity=weather_entity,
            power_entity=power_entity,
            solar_yield_today=solar_yield_today,
            temp_sensor=temp_sensor,
            wind_sensor=wind_sensor,
            rain_sensor=rain_sensor,
            uv_sensor=uv_sensor,
            lux_sensor=lux_sensor
        )
        self.sample_collector.set_forecast_cache(self._forecast_cache)

        if sun_guard is not None:
            self.sample_collector.sun_guard = sun_guard

        _LOGGER.info("Entities configured in MLPredictor: power=%s, yield=%s, weather=%s, capacity=%s",
                     power_entity, solar_yield_today, weather_entity, solar_capacity)
        _LOGGER.info("External sensors passed to SampleCollector: temp=%s, wind=%s, rain=%s, uv=%s, lux=%s",
                     temp_sensor, wind_sensor, rain_sensor, uv_sensor, lux_sensor)

    def is_healthy(self) -> bool:
        try:
            if not self.model_loaded:
                _LOGGER.debug("Health check failed: Model not loaded")
                return False
            if self.model_state not in [ModelState.READY, ModelState.TRAINED]:
                _LOGGER.debug(f"Health check failed: Model state is {self.model_state.value}")
                return False
            if self.training_samples < MIN_TRAINING_DATA_POINTS:
                _LOGGER.debug(f"Health check failed: Insufficient samples ({self.training_samples} < {MIN_TRAINING_DATA_POINTS})")
                return False
            if self.current_accuracy < MODEL_ACCURACY_THRESHOLD * 0.7:
                _LOGGER.debug(f"Health check failed: Accuracy too low ({self.current_accuracy:.2f} < {MODEL_ACCURACY_THRESHOLD * 0.7:.2f})")
                return False
            if self.last_training_time and (dt_util.utcnow() - self.last_training_time) > timedelta(days=14):
                _LOGGER.debug("Health check failed: Last training too old")
                return False
            _LOGGER.debug("Health check passed.")
            return True
        except Exception as e:
            _LOGGER.error(f"Error during health check: {e}", exc_info=True)
            return False

    def _register_shutdown_handler(self) -> None:
        if not self._shutdown_registered:
            from homeassistant.const import EVENT_HOMEASSISTANT_STOP
            self.hass.bus.async_listen_once(
                EVENT_HOMEASSISTANT_STOP,
                self._handle_shutdown
            )
            self._shutdown_registered = True
            _LOGGER.debug("Async shutdown handler registered")

    async def _handle_shutdown(self, event) -> None:
        await self.async_will_remove_from_hass()

    async def async_will_remove_from_hass(self) -> None:
        try:
            _LOGGER.info("MLPredictor Cleanup started...")

            self._stop_event.set()

            if self._time_trigger_unsub:
                self._time_trigger_unsub()
                self._time_trigger_unsub = None
                _LOGGER.debug("Hourly time trigger unsubscribed")

            if self._periodic_training_task_handle and not self._periodic_training_task_handle.done():
                self._periodic_training_task_handle.cancel()
                try:
                    await asyncio.wait_for(self._periodic_training_task_handle, timeout=5.0)
                    _LOGGER.debug("Periodic training task successfully cancelled")
                except asyncio.CancelledError:
                    _LOGGER.debug("Periodic training task already cancelled")
                except asyncio.TimeoutError:
                    _LOGGER.warning("Periodic training task did not cancel within timeout")
                except Exception as task_e:
                    _LOGGER.error(f"Error cancelling periodic task: {task_e}")

            _LOGGER.info("MLPredictor Cleanup completed")

        except Exception as e:
            _LOGGER.error(f"Error during MLPredictor Cleanup: {e}", exc_info=True)