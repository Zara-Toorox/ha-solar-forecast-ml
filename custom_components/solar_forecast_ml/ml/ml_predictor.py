"""Machine Learning Predictor V12.0.0 @zara

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
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_call_later, async_track_time_change

from ..astronomy.astronomy_cache_manager import get_cache_manager
from ..const import (
    CORRECTION_FACTOR_MAX,
    CORRECTION_FACTOR_MIN,
    DATA_VERSION,
    MIN_TRAINING_DATA_POINTS,
    ML_MODEL_VERSION,
    MODEL_ACCURACY_THRESHOLD,
)
from ..core.core_exceptions import DataIntegrityException, ErrorSeverity, MLModelException
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..data.data_adapter import TypedDataAdapter
from ..data.data_manager import DataManager
from ..ml.ml_types import (
    HourlyProfile,
    LearnedWeights,
    create_default_hourly_profile,
    create_default_learned_weights,
)
from ..services.service_error_handler import ErrorHandlingService

_np = None

from ..ml.ml_feature_engineering_v3 import FeatureEngineerV3
from ..ml.ml_data_loader_v3 import MLDataLoaderV3

from ..ml.ml_prediction_strategies import (
    FallbackStrategy,
    MLModelStrategy,
    PredictionOrchestrator,
    PredictionResult,
    ProfileStrategy,
)
from ..ml.ml_sample_collector import SampleCollector

from ..ml.ml_scaler import StandardScaler
from ..ml.ml_trainer import RidgeTrainer
from ..ml.ml_adaptive_trainer import AdaptiveTrainer, AlgorithmType

_LOGGER = logging.getLogger(__name__)

def _ensure_numpy():
    """Lazily imports and returns the NumPy module raising ImportError if unavailable @zara"""
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
        except ImportError as e:
            _LOGGER.error("NumPy library required but not installed: %s", e)
            raise
    return _np

class ModelState(Enum):
    """Represents the operational state of the ML model"""

    UNINITIALIZED = "uninitialized"
    TRAINING = "training"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"

@dataclass
class TrainingResult:
    """Stores the outcome of a model training attempt"""

    success: bool
    accuracy: float | None = None
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
    current_accuracy: float | None
    training_samples: int
    features_available: List[str]
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class MLPredictor:
    """Manages the machine learning model lifecycle including training"""

    def __init__(
        self,
        hass: HomeAssistant,
        data_manager: DataManager,
        error_handler: ErrorHandlingService,
        notification_service=None,
        config_entry=None,
    ):
        """Initialize the MLPredictor"""
        self.hass = hass
        self.data_manager = data_manager
        self.error_handler = error_handler
        self.notification_service = notification_service
        self.config_entry = config_entry
        self.data_adapter = TypedDataAdapter()

        self._historical_cache: Dict[str, Dict[str, float]] = {
            "daily_productions": {},
            "hourly_productions": {},
        }
        self._recent_weather_samples: List[Dict[str, Any]] = (
            []
        )
        self._weather_samples_last_loaded: Optional[datetime] = None

        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineerV3()
        self.data_loader = MLDataLoaderV3(
            data_manager.data_dir / "stats",
            hass=hass
        )

        ml_algorithm = "auto"
        enable_tiny_lstm = True
        if self.config_entry is not None:
            ml_algorithm = self.config_entry.options.get(
                "ml_algorithm",
                self.config_entry.data.get("ml_algorithm", "auto")
            )
            enable_tiny_lstm = self.config_entry.options.get(
                "enable_tiny_lstm",
                self.config_entry.data.get("enable_tiny_lstm", True)
            )

        self.trainer = AdaptiveTrainer(
            algorithm=ml_algorithm,
            enable_lstm=enable_tiny_lstm,
            min_samples_for_lstm=100,
            min_memory_mb=200
        )
        self.algorithm_used = "ridge"
        self.prediction_orchestrator = PredictionOrchestrator()
        self.sample_collector = SampleCollector(hass, data_manager)

        self.model_state = ModelState.UNINITIALIZED
        self.model_loaded = False
        self.current_weights: Optional[LearnedWeights] = None
        self.current_profile: Optional[HourlyProfile] = (
            None
        )
        self.peak_power_kw: float = (
            0.0
        )

        self.current_accuracy: float | None = None
        self.training_samples: int = 0
        self.last_training_time: Optional[datetime] = None
        self.last_training_samples: int = (
            0
        )

        self.performance_metrics = {
            "avg_prediction_time_ms": 0.0,
            "total_predictions": 0,
            "successful_predictions": 0,
            "error_rate": 0.0,
        }
        self._prediction_times: List[float] = []

        self._hourly_sample_listener_remove = None
        self._daily_training_listener_remove = None
        self._daily_training_task: Optional[asyncio.TimerHandle] = None
        self._stop_event = asyncio.Event()

        self._training_lock = asyncio.Lock()

        _LOGGER.debug("MLPredictor initialized")

    @property
    def can_predict(self) -> bool:
        """Check if ML model can make predictions @zara"""
        return (
            self.model_loaded
            and self.model_state == ModelState.READY
            and self.current_weights is not None
            and self.scaler is not None
            and self.scaler.is_fitted
        )

    async def initialize(self) -> bool:
        """Initializes the ML Predictor by loading existing model data @zara"""
        _LOGGER.debug("Initializing ML Predictor")
        init_success = False
        try:
            _ensure_numpy()

            loaded_weights = await self.data_manager.get_learned_weights()
            if loaded_weights:

                loaded_feature_count = (
                    len(loaded_weights.feature_names)
                    if hasattr(loaded_weights, "feature_names") and loaded_weights.feature_names
                    else 0
                )

                if loaded_feature_count == 14:
                    loaded_algorithm = getattr(loaded_weights, 'algorithm_used', None)
                    if loaded_algorithm:
                        self.algorithm_used = loaded_algorithm
                        _LOGGER.info(
                            f"Loaded model has {loaded_feature_count} features (V3) "
                            f"using {loaded_algorithm.upper()}. Using FeatureEngineerV3 for predictions."
                        )
                    else:
                        _LOGGER.info(
                            f"Loaded model has {loaded_feature_count} features (V3). Using FeatureEngineerV3 for predictions."
                        )
                    self.feature_engineer = FeatureEngineerV3()
                    expected_feature_count = 14
                elif loaded_feature_count == 44:

                    _LOGGER.warning(
                        f"Loaded V2 model with {loaded_feature_count} features. Model needs retraining with V3 (14 features)."
                    )

                    self.feature_engineer = FeatureEngineerV3()
                    expected_feature_count = 14
                else:

                    expected_feature_count = 14
                    self.feature_engineer = FeatureEngineerV3()

                if loaded_feature_count > 0 and loaded_feature_count != expected_feature_count:
                    _LOGGER.warning(
                        f"Feature count mismatch: loaded model has {loaded_feature_count} features, "
                        f"V3 expects {expected_feature_count}. Model needs retraining."
                    )

                    self.model_state = ModelState.UNINITIALIZED

                    if self.notification_service:
                        try:
                            await self.notification_service.show_model_retraining_required(
                                reason="feature_mismatch",
                                old_features=loaded_feature_count,
                                new_features=expected_feature_count,
                            )
                        except Exception as notif_err:
                            _LOGGER.debug(f"Could not send retraining notification: {notif_err}")

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
                        _LOGGER.error(
                            f"Exception during automatic retraining: {train_err}", exc_info=True
                        )
                        init_success = False

                    return init_success

                self.current_weights = loaded_weights
                self.current_accuracy = loaded_weights.accuracy
                self.training_samples = loaded_weights.training_samples
                self.last_training_samples = loaded_weights.training_samples
                try:
                    parsed_time = dt_util.parse_datetime(loaded_weights.last_trained)
                    if parsed_time:

                        self.last_training_time = (
                            dt_util.as_local(parsed_time)
                            if parsed_time.tzinfo is None
                            else parsed_time
                        )
                    else:
                        self.last_training_time = None
                except (ValueError, TypeError):
                    _LOGGER.warning("Could not parse last_trained timestamp from weights file.")
                    self.last_training_time = None

                self.model_loaded = True

                if self.training_samples >= MIN_TRAINING_DATA_POINTS:
                    self.model_state = ModelState.READY
                else:
                    self.model_state = ModelState.UNINITIALIZED

                if hasattr(loaded_weights, "feature_means") and loaded_weights.feature_means:
                    try:
                        self.scaler.set_state(
                            {
                                "means": loaded_weights.feature_means,
                                "stds": loaded_weights.feature_stds,
                                "is_fitted": True,
                                "feature_names_order": loaded_weights.feature_names,
                            }
                        )
                        _LOGGER.info(
                            "Scaler state loaded from learned weights (%d features).",
                            len(self.scaler.means),
                        )
                    except Exception as scaler_err:
                        _LOGGER.error("Failed to load scaler state from weights: %s", scaler_err)
                else:
                    _LOGGER.info("No scaler state found in learned weights. Scaler needs fitting.")

                _LOGGER.info(
                    "Learned weights loaded: Accuracy=%.2f%%, Samples=%d, Last Trained=%s, State=%s",
                    (self.current_accuracy * 100) if self.current_accuracy is not None else 0.0,
                    self.training_samples,
                    (
                        self.last_training_time.strftime("%Y-%m-%d %H:%M")
                        if self.last_training_time
                        else "Never"
                    ),
                    self.model_state.value,
                )
            else:
                _LOGGER.info("No existing learned weights found. Model needs training.")
                self.model_state = ModelState.UNINITIALIZED

                _LOGGER.info("Creating neutral default learned_weights.json")
                from ..ml.ml_types import create_default_learned_weights
                default_weights = create_default_learned_weights()
                await self.data_manager.save_learned_weights(default_weights)
                _LOGGER.debug("✓ Neutral learned_weights.json created")

            loaded_profile = await self.data_manager.get_hourly_profile()
            if loaded_profile:
                self.current_profile = loaded_profile
                _LOGGER.info(
                    "Hourly profile loaded (Samples=%d).", self.current_profile.samples_count
                )
            else:
                _LOGGER.info("No existing hourly profile found. Creating default.")
                self.current_profile = create_default_hourly_profile()

                await self.data_manager.save_hourly_profile(self.current_profile)
                _LOGGER.debug("✓ Default hourly_profile.json created")

            loaded_model_state = await self.data_manager.load_model_state()
            if loaded_model_state and loaded_model_state.get("version") == DATA_VERSION:

                if "performance_metrics" in loaded_model_state:
                    perf_metrics = loaded_model_state["performance_metrics"]
                    if perf_metrics:
                        self.performance_metrics.update(
                            {
                                "avg_prediction_time_ms": perf_metrics.get(
                                    "avg_prediction_time_ms", 0.0
                                ),
                                "error_rate": perf_metrics.get("error_rate", 0.0),
                            }
                        )
                        _LOGGER.debug("Performance metrics restored from model_state.json")

                if self.peak_power_kw == 0.0 and "peak_power_kw" in loaded_model_state:
                    self.peak_power_kw = loaded_model_state.get("peak_power_kw", 0.0)
                    _LOGGER.debug("Peak power restored: %.2f kW", self.peak_power_kw)

                if not loaded_weights and loaded_model_state.get("model_loaded"):
                    _LOGGER.info(
                        "No weights file found, but model_state indicates previous training existed."
                    )
                    self.training_samples = loaded_model_state.get("training_samples", 0)
                    self.current_accuracy = loaded_model_state.get("current_accuracy", 0.0)

                    if loaded_model_state.get("last_training"):
                        try:
                            parsed_time = dt_util.parse_datetime(
                                loaded_model_state["last_training"]
                            )
                            if parsed_time:
                                self.last_training_time = (
                                    dt_util.as_local(parsed_time)
                                    if parsed_time.tzinfo is None
                                    else parsed_time
                                )
                        except (ValueError, TypeError):
                            pass

                _LOGGER.info("Model state loaded successfully from model_state.json")
            else:
                _LOGGER.debug("No valid model_state.json found or version mismatch.")

            self.prediction_orchestrator.update_strategies(
                weights=self.current_weights,
                profile=self.current_profile,
                accuracy=self.current_accuracy if self.current_accuracy is not None else 0.0,
                peak_power_kw=self.peak_power_kw,
            )
            _LOGGER.debug("Prediction strategies updated.")

            await self._load_historical_cache()

            _LOGGER.info("=== Scheduling background tasks ===")

            self._schedule_hourly_sampling()
            _LOGGER.info("[OK] Hourly sampling scheduled (triggers at XX:02)")

            self._schedule_daily_training_check()
            _LOGGER.info("[OK] Daily training check scheduled")

            samples_count = await self._check_training_data_availability()
            _LOGGER.info(f"Current hourly samples available: {samples_count}")

            _LOGGER.debug("ML Predictor ready")
            init_success = True

        except ImportError:
            _LOGGER.critical("ML Predictor initialization failed: NumPy dependency is missing!")
            self.model_state = ModelState.ERROR
        except Exception as e:
            _LOGGER.error("ML Predictor initialization failed: %s", str(e), exc_info=True)
            self.model_state = ModelState.ERROR
        finally:
            await self._update_model_state_file(
                status_override=self.model_state if not init_success else None
            )

        return init_success

    async def _load_historical_cache(self) -> None:
        """Loads historical production data into memory for lag feature calculation @zara"""
        _LOGGER.debug("Loading historical production cache...")
        try:

            start_date = (dt_util.now() - timedelta(days=60)).date()

            data = await self.data_manager.hourly_predictions._read_json_async()
            all_predictions = data.get("predictions", [])

            records = [
                {
                    "actual_kwh": p.get("actual_kwh"),
                    "timestamp": p.get("target_datetime"),
                    "date": p.get("id", "").split("_")[0] if p.get("id") else None,
                    "hour": p.get("target_hour"),
                }
                for p in all_predictions
                if p.get("actual_kwh") is not None and p.get("id")
            ]

            daily_productions_cache: Dict[str, float] = {}
            hourly_productions_cache: Dict[str, float] = {}

            processed_records = 0
            if not records:
                _LOGGER.info("No hourly samples found to build historical cache.")
                return

            for record in records:
                try:
                    actual_kwh = record.get("actual_kwh")
                    timestamp_str = record.get("timestamp")

                    if actual_kwh is None or actual_kwh < 0 or not timestamp_str:
                        continue

                    timestamp_dt = dt_util.parse_datetime(timestamp_str)
                    if not timestamp_dt:
                        _LOGGER.debug(f"Skipping cache record, invalid timestamp: {timestamp_str}")
                        continue

                    if timestamp_dt.tzinfo is None:
                        timestamp_dt = timestamp_dt.replace(tzinfo=timezone.utc)

                    timestamp_local = dt_util.as_local(timestamp_dt)

                    date_key = timestamp_local.date().isoformat()
                    hour_key = f"{date_key}_{timestamp_local.hour:02d}"

                    hourly_productions_cache[hour_key] = actual_kwh

                    daily_productions_cache[date_key] = (
                        daily_productions_cache.get(date_key, 0.0) + actual_kwh
                    )

                    processed_records += 1

                except (ValueError, KeyError, TypeError) as parse_error:
                    _LOGGER.debug(f"Skipping record for cache due to parsing error: {parse_error}")
                    continue

            self._historical_cache = {
                "daily_productions": daily_productions_cache,
                "hourly_productions": hourly_productions_cache,
            }

            _LOGGER.info(
                "Historical cache loaded from hourly_samples: %d daily totals, %d hourly values processed from %d records.",
                len(self._historical_cache["daily_productions"]),
                len(self._historical_cache["hourly_productions"]),
                processed_records,
            )

        except Exception as e:
            _LOGGER.error("Failed to load historical cache: %s", e, exc_info=True)
            self._historical_cache = {
                "daily_productions": {},
                "hourly_productions": {},
            }

    async def _load_recent_weather_samples(
        self, hours_back: int = 24, force_reload: bool = False
    ) -> None:
        """IMPROVEMENT 7 Load recent weather samples for cloudiness trend calculation PERFORMANCE FIX: Only reload if cache is stale (>15 minutes old) or force_reload=True"""
        try:
            now = dt_util.now()

            if not force_reload and self._weather_samples_last_loaded:
                cache_age = (now - self._weather_samples_last_loaded).total_seconds() / 60.0
                if cache_age < 15 and len(self._recent_weather_samples) > 0:
                    _LOGGER.debug(f"Using cached weather samples (age: {cache_age:.1f} min)")
                    return

            cutoff_time = now - timedelta(hours=hours_back)

            data = await self.data_manager.hourly_predictions._read_json_async()
            all_predictions = data.get("predictions", [])

            samples = [
                {
                    "timestamp": p.get("target_datetime"),
                    "weather_data": p.get("weather_forecast", {}),
                }
                for p in all_predictions
                if p.get("target_datetime") and p.get("target_datetime") >= cutoff_time.isoformat()
            ]

            sorted_samples = sorted(samples, key=lambda x: x.get("timestamp", ""), reverse=True)
            self._recent_weather_samples = sorted_samples[:100]
            self._weather_samples_last_loaded = now

            _LOGGER.debug(
                f"Loaded {len(self._recent_weather_samples)} recent weather samples for trend calculation (limited to 100)"
            )

        except Exception as e:
            _LOGGER.warning(f"Failed to load recent weather samples: {e}")
            self._recent_weather_samples = []
            self._weather_samples_last_loaded = None

    def _calculate_cloudiness_trends(self) -> Dict[str, float]:
        """IMPROVEMENT 7 Calculate cloudiness trends from recent weather samples @zara"""
        trends = {
            "cloudiness_trend_1h": 0.0,
            "cloudiness_trend_3h": 0.0,
            "cloudiness_volatility": 0.0,
        }

        try:
            if not self._recent_weather_samples:
                return trends

            cloudiness_data = []
            now = dt_util.now()

            for sample in self._recent_weather_samples:
                timestamp_str = sample.get("timestamp")
                if not timestamp_str:
                    continue

                timestamp = dt_util.parse_datetime(timestamp_str)
                if not timestamp:
                    continue

                age_hours = (now - timestamp).total_seconds() / 3600.0

                weather_data = sample.get("weather_data", {})
                cloudiness = weather_data.get("clouds")

                if cloudiness is not None and 0 <= cloudiness <= 100:
                    cloudiness_data.append((age_hours, float(cloudiness)))

            if not cloudiness_data:
                return trends

            cloudiness_data.sort()

            recent_1h = [c for age, c in cloudiness_data if age <= 1.0]
            if len(recent_1h) >= 2:
                trends["cloudiness_trend_1h"] = recent_1h[-1] - recent_1h[0]

            recent_3h = [c for age, c in cloudiness_data if age <= 3.0]
            if len(recent_3h) >= 2:
                trends["cloudiness_trend_3h"] = recent_3h[-1] - recent_3h[0]

                if len(recent_3h) >= 3:
                    mean_cloudiness = sum(recent_3h) / len(recent_3h)
                    variance = sum((c - mean_cloudiness) ** 2 for c in recent_3h) / len(recent_3h)
                    trends["cloudiness_volatility"] = variance**0.5

            _LOGGER.debug(
                f"Cloudiness trends: 1h={trends['cloudiness_trend_1h']:.1f}, "
                f"3h={trends['cloudiness_trend_3h']:.1f}, vol={trends['cloudiness_volatility']:.1f}"
            )

        except Exception as e:
            _LOGGER.debug(f"Failed to calculate cloudiness trends: {e}")

        return trends

    async def predict(
        self,
        weather_data: Dict[str, Any],
        prediction_hour: int,
        prediction_date: datetime,
        sensor_data: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        """Generates a solar production prediction for a specific hour using V3 pipeline"""
        prediction_start_time = dt_util.now()
        result: PredictionResult | None = None

        try:
            _ensure_numpy()
            if sensor_data is None:
                sensor_data = {}

            try:

                yesterday_dt = prediction_date - timedelta(days=1)
                yesterday_key = yesterday_dt.date().isoformat()
                yesterday_total_kwh = self._historical_cache["daily_productions"].get(
                    yesterday_key, 0.0
                )
                same_hour_yesterday_key = f"{yesterday_key}_{prediction_hour:02d}"
                same_hour_yesterday_kwh = self._historical_cache["hourly_productions"].get(
                    same_hour_yesterday_key, 0.0
                )

                sensor_data["production_yesterday"] = float(yesterday_total_kwh)
                sensor_data["production_same_hour_yesterday"] = float(same_hour_yesterday_kwh)

            except Exception as e:
                _LOGGER.warning(f"Could not retrieve lag features for prediction: {e}")
                sensor_data["production_yesterday"] = 0.0
                sensor_data["production_same_hour_yesterday"] = 0.0

            astronomy_basic = {}
            astronomy_advanced = {}

            try:
                cache_manager = get_cache_manager()
                date_key = prediction_date.date().isoformat()

                if cache_manager and hasattr(cache_manager, 'is_loaded') and cache_manager.is_loaded():
                    day_data = cache_manager.get_day_data(date_key) if hasattr(cache_manager, 'get_day_data') else None

                    if day_data and isinstance(day_data, dict):

                        astronomy_basic = {
                            "sunrise": day_data.get("sunrise_local"),
                            "sunset": day_data.get("sunset_local"),
                            "solar_noon": day_data.get("solar_noon_local"),
                            "daylight_hours": day_data.get("daylight_hours"),
                        }

                        hourly_data = day_data.get("hourly", {})
                        hour_str = str(prediction_hour)

                        if isinstance(hourly_data, dict) and hour_str in hourly_data:
                            hour_astro = hourly_data[hour_str]
                            if isinstance(hour_astro, dict):
                                astronomy_advanced = {
                                    "elevation_deg": hour_astro.get("elevation_deg"),
                                    "azimuth_deg": hour_astro.get("azimuth_deg"),
                                    "clear_sky_solar_radiation_wm2": hour_astro.get(
                                        "clear_sky_solar_radiation_wm2"
                                    ),
                                    "theoretical_max_pv_kwh": hour_astro.get("theoretical_max_pv_kwh"),
                                    "hours_since_solar_noon": hour_astro.get("hours_since_solar_noon"),
                                    "day_progress_ratio": hour_astro.get("day_progress_ratio"),
                                }
            except Exception as e:
                _LOGGER.debug(f"Could not load astronomy data for prediction: {e}")

            weather_corrected = {
                "temperature": weather_data.get("temperature", 15.0),
                "solar_radiation_wm2": weather_data.get("solar_radiation_wm2", 0.0),
                "wind": weather_data.get("wind_speed", weather_data.get("wind", 3.0)),
                "humidity": weather_data.get("humidity", 70.0),
                "rain": weather_data.get("precipitation", weather_data.get("rain", 0.0)),
                "clouds": weather_data.get("cloud_cover", weather_data.get("clouds", 50.0)),
            }

            astronomy_merged = {
                "sun_elevation_deg": astronomy_advanced.get("elevation_deg", -30.0),
                "theoretical_max_kwh": astronomy_advanced.get("theoretical_max_pv_kwh", 0.0),
                "clear_sky_radiation_wm2": astronomy_advanced.get("clear_sky_solar_radiation_wm2", 0.0),
            }

            record = {

                "target_hour": prediction_hour,
                "target_day_of_year": prediction_date.timetuple().tm_yday,
                "target_month": prediction_date.month,
                "target_day_of_week": prediction_date.weekday(),
                "target_season": self._get_season(prediction_date.month),
                "prediction_created_hour": dt_util.now().hour,
                "weather_data": weather_data,
                "weather_corrected": weather_corrected,
                "sensor_data": sensor_data,
                "astronomy": astronomy_merged,

                "astronomy_basic": astronomy_basic,
                "astronomy_advanced": astronomy_advanced,

                "production_yesterday": sensor_data.get("production_yesterday", 0.0),
                "production_same_hour_yesterday": sensor_data.get("production_same_hour_yesterday", 0.0),

                "is_production_hour": True,
            }

            features = self.feature_engineer.extract_features(record)

            if features and len(features) > 1:
                _LOGGER.debug(f"Raw features[1] (target_day_of_year): {features[1]:.4f}")

            if self.algorithm_used == "tiny_lstm":
                features_scaled = features
                _LOGGER.debug("Using raw features for TinyLSTM prediction.")

                await self._load_historical_data_for_lstm(prediction_date)

            elif self.scaler.is_fitted:
                features_scaled = self.scaler.transform_single(features)
                if features_scaled and len(features_scaled) > 1:
                    _LOGGER.debug(f"Scaled features[1] (target_day_of_year): {features_scaled[1]:.4f}")
                _LOGGER.debug("Prediction features scaled.")
            else:
                features_scaled = features
                _LOGGER.debug("Scaler not fitted, using raw features for prediction.")

            result = await self.prediction_orchestrator.predict(features_scaled)

            prediction_end_time = dt_util.now()
            duration_ms = (prediction_end_time - prediction_start_time).total_seconds() * 1000
            self._update_performance_metrics(duration_ms, success=True)

            _LOGGER.debug(
                f"Hourly prediction successful (Method: {result.method}): {result.prediction:.2f} kWh, "
                f"Confidence: {result.confidence:.2f}, Duration: {duration_ms:.1f}ms"
            )
            return result

        except ImportError:
            _LOGGER.error("Prediction failed: NumPy dependency is missing.")
            await self.error_handler.handle_error(
                error=MLModelException("Prediction failed due to missing NumPy"),
                source="ml_predictor",
                context={},
                pipeline_position="predict",
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
                    "model_state": self.model_state.value,
                },
                pipeline_position="predict",
            )
            fallback_result = await self._get_fallback_prediction(prediction_hour, prediction_date)
            prediction_end_time = dt_util.now()
            duration_ms = (prediction_end_time - prediction_start_time).total_seconds() * 1000
            self._update_performance_metrics(duration_ms, success=False)
            return fallback_result

    async def _get_fallback_prediction(self, hour: int, date: datetime) -> PredictionResult:
        """Generates a prediction using the fallback strategy @zara"""
        _LOGGER.warning("Using fallback prediction strategy.")

        fallback_strategy = FallbackStrategy(self.peak_power_kw)
        default_features = self.feature_engineer.get_default_features(hour, date)
        return await fallback_strategy.predict(default_features)

    async def _load_historical_data_for_lstm(self, prediction_date: datetime) -> None:
        """Load last 24+ hours from hourly_predictions for LSTM sequence building @zara"""
        try:

            recent_hours = None

            if hasattr(self, '_hourly_predictions_cache') and self._hourly_predictions_cache:
                predictions = self._hourly_predictions_cache.get("predictions", [])
                if predictions:
                    recent_hours = self._filter_recent_hours(predictions, prediction_date)

            if not recent_hours:
                hourly_file = Path(self.data_manager.data_dir) / "stats" / "hourly_predictions.json"
                if hourly_file.exists():
                    def _read_hourly():
                        with open(hourly_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        return data.get("predictions", [])

                    predictions = await self.hass.async_add_executor_job(_read_hourly)
                    if predictions:
                        recent_hours = self._filter_recent_hours(predictions, prediction_date)

            if recent_hours and len(recent_hours) >= 24:
                self.prediction_orchestrator.set_historical_data(recent_hours)
                _LOGGER.debug(f"LSTM: Loaded {len(recent_hours)} historical hours for sequence building")
            else:
                _LOGGER.debug(
                    f"LSTM: Insufficient historical data ({len(recent_hours) if recent_hours else 0} hours) - "
                    f"will use pseudo-sequence fallback"
                )
                self.prediction_orchestrator.set_historical_data(None)

        except Exception as e:
            _LOGGER.warning(f"Failed to load historical data for LSTM: {e}")
            self.prediction_orchestrator.set_historical_data(None)

    def _filter_recent_hours(
        self,
        predictions: list,
        prediction_date: datetime
    ) -> list:
        """
        Filter predictions to get the last 24-48 hours before prediction_date.

        Returns hourly predictions sorted chronologically, covering the 24-48 hours
        before the prediction date. This provides the lookback window needed for
        LSTM sequence building.
        """
        from datetime import timedelta

        cutoff_date = (prediction_date - timedelta(hours=48)).date().isoformat()
        target_date_str = prediction_date.date().isoformat()

        filtered = [
            p for p in predictions
            if p.get("target_date", "") >= cutoff_date
            and p.get("target_date", "") <= target_date_str
        ]

        filtered.sort(key=lambda x: (x.get("target_date", ""), x.get("target_hour", 0)))

        return filtered[-48:] if len(filtered) > 48 else filtered

    async def train_model(self) -> TrainingResult:
        """Trains the Ridge Regression model using V3 pipeline - 14 features with weather corrections @zara"""
        _LOGGER.info("Using V3 training pipeline (weather-corrected, 14 features)")
        return await self.train_model_v3()

    async def train_model_v3(self) -> TrainingResult:
        """Train model using V3 data structure (weather-corrected forecasts) @zara"""

        try:
            async with asyncio.timeout(30.0):
                async with self._training_lock:
                    training_start_time = dt_util.now()
                    _LOGGER.info(f"Starting ML model training (V3) at {training_start_time}...")
                    self.model_state = ModelState.TRAINING
                    await self._update_model_state_file(status_override=ModelState.TRAINING)

                training_records = []
                result: TrainingResult | None = None

                try:
                    _ensure_numpy()

                    _LOGGER.info(
                        "Loading training data using V3 structure (weather-corrected forecasts)..."
                    )

                    training_records, count = await self.data_loader.load_training_data()

                    absolute_min_samples = 10
                    recommended_min_samples = 30

                    if count < absolute_min_samples:
                        error_msg = (
                            f"Insufficient training data: {count} valid records found "
                            f"(absolute minimum: {absolute_min_samples}). Training aborted."
                        )
                        _LOGGER.warning(error_msg)
                        self.model_state = (
                            ModelState.UNINITIALIZED if not self.model_loaded else ModelState.READY
                        )
                        return TrainingResult(
                            success=False, samples_used=count, error_message=error_msg
                        )

                    if count < recommended_min_samples:
                        _LOGGER.warning(
                            f"Training with {count} samples (below recommended {recommended_min_samples}). "
                            f"Model accuracy may be lower than optimal."
                        )

                    _LOGGER.info(f"Preparing {count} records for training (V3)...")

                    X_train_raw = []
                    y_train = []
                    for record in training_records:
                        features = self.feature_engineer.extract_features(record)
                        X_train_raw.append(features)
                        y_train.append(record.get("actual_kwh", 0))

                    _LOGGER.debug(f"Feature extraction complete: {len(X_train_raw)} samples with 14 features")

                    feature_names = self.feature_engineer.feature_names

                    X_train_scaled_list = await self.hass.async_add_executor_job(
                        self.scaler.fit_transform, X_train_raw, feature_names
                    )
                    _LOGGER.info(f"Scaler fitted and training data transformed ({len(feature_names)} features)")

                    hourly_predictions_list = []
                    try:
                        hourly_file = Path(self.data_manager.data_dir) / "stats" / "hourly_predictions.json"
                        if hourly_file.exists():
                            def _read_hourly():
                                with open(hourly_file, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                return data.get("predictions", [])

                            hourly_predictions_list = await self.hass.async_add_executor_job(_read_hourly)
                            _LOGGER.debug(f"Loaded {len(hourly_predictions_list)} hourly predictions for LSTM sequence building")
                        else:
                            _LOGGER.warning("hourly_predictions.json not found - LSTM training will use fallback")
                    except Exception as e:
                        _LOGGER.warning(f"Could not load hourly_predictions.json: {e}")

                    weights_result, accuracy, algorithm_used = await self.trainer.train(
                        X_train=X_train_scaled_list,
                        y_train=y_train,
                        feature_names=feature_names,
                        hourly_predictions=hourly_predictions_list
                    )

                    self.algorithm_used = algorithm_used

                    _LOGGER.info(
                        f"{algorithm_used.upper()} training complete. "
                        f"Accuracy (R-squared): {accuracy:.4f}"
                    )

                    if algorithm_used == "tiny_lstm":

                        weights_dict_raw = weights_result
                        bias = 0.0
                        best_lambda = 0.0
                    else:

                        if isinstance(weights_result, tuple):
                            weights_dict_raw, bias, _, best_lambda = weights_result
                        else:
                            weights_dict_raw = weights_result
                            bias = 0.0
                            best_lambda = 0.0

                    if algorithm_used == "tiny_lstm":

                        weights_dict = weights_dict_raw
                    else:

                        weights_dict = {feature_names[i]: weights_dict_raw[f"feature_{i}"] for i in range(len(feature_names))}

                    feature_means = {}
                    feature_stds = {}
                    if hasattr(self.scaler, "means") and hasattr(self.scaler, "stds"):
                        feature_means = self.scaler.means.copy()
                        feature_stds = self.scaler.stds.copy()

                    new_learned_weights = LearnedWeights(
                        weights=weights_dict,
                        bias=bias,
                        feature_names=feature_names,
                        feature_means=feature_means,
                        feature_stds=feature_stds,
                        accuracy=accuracy,
                        training_samples=len(training_records),
                        last_trained=dt_util.now().isoformat(),
                        model_version=ML_MODEL_VERSION,
                        algorithm_used=algorithm_used,
                    )

                    from ..const import MODEL_ACCURACY_THRESHOLD

                    await self.data_manager.save_learned_weights(new_learned_weights)
                    _LOGGER.info("Learned weights saved to disk")

                    self.learned_weights = new_learned_weights
                    self.model_loaded = True
                    self.current_accuracy = accuracy
                    self.training_samples = len(y_train)
                    self.model_state = (
                        ModelState.READY if accuracy >= MODEL_ACCURACY_THRESHOLD else ModelState.DEGRADED
                    )
                    self.last_training = training_start_time

                    await self._update_model_state_file()

                    training_end_time = dt_util.now()
                    duration_seconds = (training_end_time - training_start_time).total_seconds()

                    result = TrainingResult(
                        success=True,
                        accuracy=accuracy,
                        samples_used=len(training_records),
                        weights=new_learned_weights,
                        training_time_seconds=duration_seconds,
                        feature_count=len(weights_dict_raw),
                    )

                    self.last_training_samples = len(training_records)

                    _LOGGER.info(
                        f"ML Training V3 successful in {duration_seconds:.2f}s. "
                        f"Accuracy={accuracy*100:.1f}%, Samples={len(training_records)}, Features={len(weights_dict_raw)}."
                    )

                    self.error_handler.log_ml_operation(
                        operation="model_training_v3",
                        success=True,
                        metrics={
                            "accuracy": accuracy,
                            "samples": len(training_records),
                            "features": result.feature_count,
                        },
                        duration_seconds=duration_seconds,
                    )

                    # Update hourly profile after successful training
                    try:
                        await self._update_hourly_profile(training_records)
                    except Exception as profile_err:
                        _LOGGER.warning(f"Hourly profile update failed (non-critical): {profile_err}")

                    return result

                except Exception as e:
                    _LOGGER.error(f"ML Training V3 failed: {e}", exc_info=True)
                    self.model_state = ModelState.ERROR
                    result = TrainingResult(
                        success=False, samples_used=len(training_records), error_message=str(e)
                    )
                    await self.error_handler.handle_error(
                        error=Exception(str(e)),
                        source="ml_predictor_v3",
                        pipeline_position="train_model_v3",
                    )
                    return result

        except asyncio.TimeoutError:
            _LOGGER.warning("Training already in progress, skipping concurrent training request")
            return TrainingResult(
                success=False, error_message="Training already in progress", samples_used=0
            )

    async def _update_hourly_profile(self, training_records: List[Dict[str, Any]]) -> None:
        """Updates the hourly production profile based on provided training records @zara"""
        _LOGGER.debug(f"Updating hourly profile using {len(training_records)} records...")
        if not training_records:
            _LOGGER.warning("No records provided for hourly profile update.")
            if self.current_profile is None:
                self.current_profile = create_default_hourly_profile()
            return

        try:
            np = _ensure_numpy()
            hourly_data: Dict[int, List[float]] = {hour: [] for hour in range(24)}
            valid_sample_count = 0

            for record in training_records:
                try:
                    actual_kwh = record.get("actual_kwh")
                    if actual_kwh is None or actual_kwh <= 0:
                        continue

                    # Try target_hour first (from MLDataLoaderV3), then parse from target_datetime
                    hour = record.get("target_hour")
                    if hour is None:
                        # Fallback: try parsing from target_datetime or timestamp
                        datetime_str = record.get("target_datetime") or record.get("timestamp")
                        if not datetime_str:
                            continue
                        parsed_dt = dt_util.parse_datetime(datetime_str)
                        if not parsed_dt:
                            continue
                        hour = parsed_dt.hour

                    hourly_data[hour].append(actual_kwh)
                    valid_sample_count += 1
                except (ValueError, KeyError, TypeError) as e:
                    _LOGGER.debug(f"Skipping invalid record during profile update: {e}")
                    continue

            if valid_sample_count == 0:
                _LOGGER.warning(
                    "No valid hourly values found in provided records. Hourly profile not updated."
                )
                if self.current_profile is None:
                    self.current_profile = create_default_hourly_profile()
                return

            hourly_averages_median: Dict[str, float] = {}
            for hour, values in hourly_data.items():
                if values:
                    median_val = await self.hass.async_add_executor_job(np.median, values)
                    hourly_averages_median[str(hour)] = float(median_val)
                else:
                    hourly_averages_median[str(hour)] = 0.0

            # Calculate hourly_factors as relative factors (hour average / daily average)
            # These factors show how each hour compares to the daily average
            hourly_factors: Dict[str, float] = {}
            non_zero_averages = [v for v in hourly_averages_median.values() if v > 0]
            if non_zero_averages:
                daily_mean = sum(non_zero_averages) / len(non_zero_averages)
                if daily_mean > 0:
                    for hour_str, avg in hourly_averages_median.items():
                        if avg > 0:
                            hourly_factors[hour_str] = round(avg / daily_mean, 3)
                        else:
                            hourly_factors[hour_str] = 0.0

            confidence = min(1.0, max(0.1, valid_sample_count / 300.0))
            new_profile = HourlyProfile(
                hourly_averages=hourly_averages_median,
                samples_count=valid_sample_count,
                last_updated=dt_util.now().isoformat(),
                confidence=confidence,
                hourly_factors=hourly_factors,
            )

            await self.data_manager.save_hourly_profile(new_profile)
            self.current_profile = new_profile

            _LOGGER.info(
                f"Hourly profile updated successfully using {valid_sample_count} valid samples. Confidence: {confidence:.2f}"
            )

        except ImportError:
            _LOGGER.error("Hourly profile update failed: NumPy dependency is missing!")
        except Exception as e:
            _LOGGER.error(f"Hourly profile update failed: {e}", exc_info=True)
            if self.current_profile is None:
                self.current_profile = create_default_hourly_profile()

    async def _remove_outliers(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """CRITICAL FIX 4 Removes outliers from training data using IQR method @zara"""
        if not records:
            return records

        try:
            np = _ensure_numpy()

            hourly_groups: Dict[int, List[Dict[str, Any]]] = {hour: [] for hour in range(24)}

            for record in records:
                try:
                    timestamp_str = record.get("timestamp")
                    if not timestamp_str:
                        continue

                    timestamp = dt_util.parse_datetime(timestamp_str)
                    if not timestamp:
                        continue

                    hour = dt_util.ensure_local(timestamp).hour
                    hourly_groups[hour].append(record)
                except Exception:
                    continue

            filtered_records = []
            outliers_removed = 0

            for hour, hour_records in hourly_groups.items():

                if len(hour_records) < 10:
                    filtered_records.extend(hour_records)
                    continue

                values = []
                for rec in hour_records:
                    actual = rec.get("actual_kwh")
                    if actual is None:
                        actual = rec.get("actual_value")
                    if actual is not None and actual >= 0:
                        values.append(actual)

                if len(values) < 10:
                    filtered_records.extend(hour_records)
                    continue

                values_array = np.array(values)
                q1 = np.percentile(values_array, 25)
                q3 = np.percentile(values_array, 75)
                iqr = q3 - q1

                lower_bound = q1 - 3.0 * iqr
                upper_bound = q3 + 3.0 * iqr

                for rec in hour_records:
                    actual = rec.get("actual_kwh")
                    if actual is None:
                        actual = rec.get("actual_value")

                    if actual is None:
                        filtered_records.append(rec)
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
            return records

    async def _update_model_state_file(
        self,
        status_override: Optional[ModelState | str] = None,
        accuracy_override: Optional[float] = None,
        samples_override: Optional[int] = None,
        training_time_override: Optional[float] = None,
    ) -> None:
        """Safely updates the model_statejson file with current predictor status"""
        _LOGGER.debug("Updating model_state.json...")
        try:
            status_to_save = status_override if status_override else self.model_state
            status_str = (
                status_to_save.value
                if isinstance(status_to_save, ModelState)
                else str(status_to_save)
            )
            accuracy = accuracy_override if accuracy_override is not None else self.current_accuracy
            samples = samples_override if samples_override is not None else self.training_samples
            last_training_iso = (
                self.last_training_time.isoformat() if self.last_training_time else None
            )

            current_state = {
                "version": DATA_VERSION,
                "model_loaded": self.model_loaded,
                "last_training": last_training_iso,
                "training_samples": samples,
                "current_accuracy": float(accuracy) if accuracy is not None else None,
                "status": status_str,
                "peak_power_kw": self.peak_power_kw,
                "model_info": {"version": ML_MODEL_VERSION, "type": "ridge_regression_normalized"},
                "performance_metrics": {
                    "avg_prediction_time_ms": self.performance_metrics.get(
                        "avg_prediction_time_ms"
                    ),
                    "error_rate": self.performance_metrics.get("error_rate"),
                },
                **(
                    {"training_time_seconds": round(training_time_override, 2)}
                    if training_time_override is not None
                    else {}
                ),
                "last_updated": dt_util.now().isoformat(),
            }

            await self.data_manager.save_model_state(current_state)
            _LOGGER.debug(f"model_state.json updated successfully (Status: {status_str}).")

        except Exception as e:
            _LOGGER.error(f"Failed to update model_state.json: {e}", exc_info=True)

    async def _check_training_data_availability(self) -> int:
        """Checks how many valid hourly samples are available for training @zara"""
        try:

            from ..core.core_helpers import SafeDateTimeUtil as dt_util

            data = await self.data_manager.hourly_predictions._read_json_async()
            all_predictions = data.get("predictions", [])

            valid_count = sum(
                1
                for p in all_predictions
                if p.get("actual_kwh") is not None and p.get("actual_kwh") >= 0
            )
            _LOGGER.debug(
                f"Training data check: Found {valid_count} valid hourly samples (including zero-yield)."
            )
            return valid_count
        except Exception as e:
            _LOGGER.error(f"Training data availability check failed: {e}", exc_info=True)
            return 0

    def _schedule_hourly_sampling(self) -> None:
        """Legacy method - sample collection handled by Coordinator. @zara"""
        _LOGGER.debug("Sample collection handled by Coordinator, skipping ML-based scheduling")

    def _schedule_daily_training_check(self, reschedule_delay_sec: Optional[float] = None) -> None:
        """Schedules the next daily check to see if model retraining is needed @zara"""
        try:

            if self._daily_training_task:
                self._daily_training_task.cancel()
                self._daily_training_task = None

            if reschedule_delay_sec is not None:
                delay = reschedule_delay_sec
                _LOGGER.info(f"Scheduling next training check in {delay:.0f} seconds.")
            else:
                now_local = dt_util.now()
                target_time_local = now_local.replace(hour=23, minute=5, second=0, microsecond=0)
                if now_local >= target_time_local:
                    target_time_local += timedelta(days=1)
                delay = (target_time_local - now_local).total_seconds()
                _LOGGER.info(
                    f"Scheduling next daily training check at {target_time_local.strftime('%Y-%m-%d %H:%M:%S')} (in {delay:.0f} seconds)."
                )

            def _trigger_training_check():
                """Wrapper function to properly create and handle the async task @zara"""
                try:
                    _LOGGER.debug("Daily training check trigger fired - creating async task...")
                    task = self.hass.async_create_task(self._daily_training_check_callback())

                    task.add_done_callback(lambda t: self._handle_training_task_error(t))
                except Exception as e:
                    _LOGGER.error(f"Failed to create daily training check task: {e}", exc_info=True)

            self._daily_training_task = self.hass.loop.call_later(delay, _trigger_training_check)
            _LOGGER.debug(f"Daily training check task scheduled successfully (delay={delay:.0f}s)")

        except Exception as e:
            _LOGGER.error(f"Failed to schedule daily training check: {e}", exc_info=True)

            if reschedule_delay_sec is None:
                _LOGGER.info("Attempting fallback rescheduling in 1 hour...")
                try:
                    self._schedule_daily_training_check(reschedule_delay_sec=3600.0)
                except Exception as fallback_error:
                    _LOGGER.error(
                        f"Fallback rescheduling also failed: {fallback_error}", exc_info=True
                    )

    async def _daily_training_check_callback(self) -> None:
        """Callback executed daily to check if retraining is needed and trigger it @zara"""
        _LOGGER.info("Performing daily check: Should ML model be retrained?")
        try:
            if await self._should_retrain():
                _LOGGER.info("Retraining criteria met. Starting scheduled model training...")
                training_result = await self.train_model()
                if training_result.success:
                    _LOGGER.info("Scheduled model training completed successfully.")
                else:
                    _LOGGER.warning(
                        "Scheduled model training failed: %s", training_result.error_message
                    )
            else:
                _LOGGER.info("No scheduled retraining needed at this time.")
        except Exception as e:
            _LOGGER.error("Error during daily training check/trigger: %s", e, exc_info=True)
        finally:

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
                    exc_info=True,
                )

    async def _should_retrain(self) -> bool:
        """Determines if the model should be automatically retrained based on criteria @zara"""
        _LOGGER.debug("Checking retraining criteria...")
        try:
            available_samples = await self._check_training_data_availability()

            if (
                not self.model_loaded
                or not self.last_training_time
                or self.current_accuracy is None
                or self.training_samples < MIN_TRAINING_DATA_POINTS
            ):
                if available_samples >= MIN_TRAINING_DATA_POINTS:
                    _LOGGER.info(
                        f"Retraining needed: Model not loaded/trained, but sufficient data ({available_samples}) available."
                    )
                    return True
                else:
                    _LOGGER.info(
                        f"Skipping retraining: Model not loaded/trained and insufficient data ({available_samples} < {MIN_TRAINING_DATA_POINTS})."
                    )
                    return False

            accuracy_threshold = MODEL_ACCURACY_THRESHOLD * 0.9
            if self.current_accuracy < accuracy_threshold:
                _LOGGER.info(
                    f"Retraining needed: Current accuracy ({self.current_accuracy:.3f}) is below threshold ({accuracy_threshold:.3f})."
                )
                return True

            max_model_age = timedelta(days=7)
            time_since_training = dt_util.now() - self.last_training_time
            if time_since_training > max_model_age:
                _LOGGER.info(
                    f"Retraining needed: Last training was {time_since_training.days} days ago (max allowed: {max_model_age.days})."
                )
                return True

            if available_samples > self.training_samples * 1.5 or (
                available_samples > self.training_samples + 100
            ):
                _LOGGER.info(
                    f"Retraining needed: Significant new data available ({available_samples} current vs {self.training_samples} last trained)."
                )
                return True

            _LOGGER.debug("Retraining criteria not met.")
            return False
        except Exception as e:
            _LOGGER.error(f"Error during _should_retrain check: {e}", exc_info=True)
            return False

    def _handle_training_task_error(self, task: asyncio.Task) -> None:
        """Handle errors from the daily training check task @zara"""
        try:

            task.result()
        except asyncio.CancelledError:
            _LOGGER.debug("Daily training check task was cancelled")
        except Exception as e:
            _LOGGER.error(f"Daily training check task failed with error: {e}", exc_info=True)

    def _update_performance_metrics(self, prediction_time_ms: float, success: bool) -> None:
        """Updates rolling average prediction time and error rate @zara"""
        try:
            self.performance_metrics["total_predictions"] += 1
            if success:
                self.performance_metrics["successful_predictions"] += 1
                self._prediction_times.append(prediction_time_ms)
                if len(self._prediction_times) > 100:
                    self._prediction_times.pop(0)
            if self._prediction_times:
                np = _ensure_numpy()
                avg_time = np.mean(self._prediction_times)
                self.performance_metrics["avg_prediction_time_ms"] = avg_time
            else:
                self.performance_metrics["avg_prediction_time_ms"] = 0.0
            total = self.performance_metrics["total_predictions"]
            successful = self.performance_metrics["successful_predictions"]
            if total > 0:
                self.performance_metrics["error_rate"] = 1.0 - (successful / total)
            else:
                self.performance_metrics["error_rate"] = 0.0
            _LOGGER.debug(
                f"Performance metrics updated: AvgTime={self.performance_metrics['avg_prediction_time_ms']:.1f}ms, ErrorRate={self.performance_metrics['error_rate']:.3f}"
            )
        except ImportError:
            self.performance_metrics["avg_prediction_time_ms"] = -1.0
        except Exception as e:
            _LOGGER.warning(f"Failed to update performance metrics: {e}")

    def set_entities(
        self,
        power_entity: Optional[str] = None,
        weather_entity: Optional[str] = None,
        solar_capacity: Optional[float] = None,
        temp_sensor: Optional[str] = None,
        wind_sensor: Optional[str] = None,
        rain_sensor: Optional[str] = None,
        lux_sensor: Optional[str] = None,
        humidity_sensor: Optional[str] = None,
        pressure_sensor: Optional[str] = None,
        solar_radiation_sensor: Optional[str] = None,
    ) -> None:
        """Configures the entity IDs used by the ML predictor and its components"""
        _LOGGER.info("Configuring entities for MLPredictor and SampleCollector...")
        _LOGGER.debug(f"set_entities called with solar_capacity={solar_capacity}")

        if solar_capacity and solar_capacity > 0:
            self.peak_power_kw = float(solar_capacity)
            _LOGGER.info(
                f"Peak-Power configured: {self.peak_power_kw} kW (from config: {solar_capacity})"
            )
        else:
            _LOGGER.warning(
                f"Peak-Power not configured or invalid (received: {solar_capacity}) - predictions may be unrealistic"
            )

        self.sample_collector.configure_entities(
            weather_entity=weather_entity,
            power_entity=power_entity,
            temp_sensor=temp_sensor,
            wind_sensor=wind_sensor,
            rain_sensor=rain_sensor,
            lux_sensor=lux_sensor,
            humidity_sensor=humidity_sensor,
            pressure_sensor=pressure_sensor,
            solar_radiation_sensor=solar_radiation_sensor,
        )

        _LOGGER.info(
            "Entities configured in SampleCollector: power=%s, weather=%s, temp=%s, wind=%s, rain=%s, lux=%s, humidity=%s, pressure=%s, solar_radiation=%s",
            power_entity,
            weather_entity,
            temp_sensor,
            wind_sensor,
            rain_sensor,
            lux_sensor,
            humidity_sensor,
            pressure_sensor,
            solar_radiation_sensor,
        )

    def is_healthy(self) -> bool:
        """Performs a health check on the ML predictor state @zara"""
        _LOGGER.debug("Performing ML predictor health check...")
        try:
            _ensure_numpy()

            if not self.model_loaded:
                _LOGGER.debug("Health check failed: Model not loaded.")
                return False

            if self.model_state != ModelState.READY:
                _LOGGER.debug(
                    f"Health check failed: Model state is {self.model_state.value}, expected READY."
                )
                return False

            if self.training_samples < MIN_TRAINING_DATA_POINTS:
                _LOGGER.info(
                    f"Health check: Model is loaded but not sufficiently trained (Samples: {self.training_samples} < {MIN_TRAINING_DATA_POINTS}). "
                    "ML strategy will be skipped (Fallback active)."
                )
                return False

            max_age = timedelta(days=30)
            if self.last_training_time and (dt_util.now() - self.last_training_time) > max_age:
                _LOGGER.info(
                    f"Model is {(dt_util.now() - self.last_training_time).days} days old. "
                    "Consider retraining for latest patterns."
                )

            _LOGGER.debug("ML predictor health check passed (Loaded, Ready, Sufficient Samples).")
            return True

        except ImportError:
            _LOGGER.error("Health check failed: NumPy dependency missing.")
            return False
        except Exception as e:
            _LOGGER.error(f"Error during ML predictor health check: {e}", exc_info=True)
            return False

    async def get_today_prediction(self) -> Optional[float]:
        """Get total daily prediction for today @zara"""
        _LOGGER.debug("get_today_prediction stub called, returning None (use orchestrator)")
        return None

    async def get_tomorrow_prediction(self) -> Optional[float]:
        """Get total daily prediction for tomorrow @zara"""
        _LOGGER.debug("get_tomorrow_prediction stub called, returning None (use orchestrator)")
        return None

    async def force_training(self) -> bool:
        """Force model training regardless of conditions @zara"""
        _LOGGER.info("Force training requested via service...")
        try:
            result = await self.train_model()
            return result.success if result else False
        except Exception as e:
            _LOGGER.error(f"Force training failed: {e}", exc_info=True)
            return False

    async def reset_model(self) -> bool:
        """Reset the ML model by clearing learned weights and reverting to rule-based pr... @zara"""
        _LOGGER.info("Resetting ML model...")
        try:

            self.current_weights = None
            self.current_accuracy = None
            self.training_samples = 0
            self.last_training_time = None
            self.model_loaded = False
            self.model_state = ModelState.UNINITIALIZED

            success = await self.data_manager.delete_learned_weights()
            if not success:
                _LOGGER.warning("Could not delete learned weights file (may not exist)")

            self.scaler = StandardScaler()

            self.prediction_orchestrator.update_strategies(
                weights=None,
                profile=self.current_profile,
                accuracy=None,
                peak_power_kw=self.peak_power_kw,
            )

            await self._update_model_state_file(
                status_override=ModelState.UNINITIALIZED.value,
                accuracy_override=None,
                samples_override=0,
                training_time_override=None,
            )

            _LOGGER.info("ML model reset successfully. System will use rule-based predictions.")

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to reset ML model: {e}", exc_info=True)
            await self.error_handler.handle_error(
                error=MLModelException(f"Model reset failed: {e}"),
                source="ml_predictor",
                pipeline_position="reset_model",
            )
            return False

    async def async_will_remove_from_hass(self) -> None:
        """Clean up resources when the integration is unloaded or HA stops @zara"""
        _LOGGER.info("Cleaning up MLPredictor background tasks and listeners...")
        self._stop_event.set()

        if self._hourly_sample_listener_remove:
            try:
                self._hourly_sample_listener_remove()
                _LOGGER.debug("Removed hourly sample listener.")
            except Exception as e:
                _LOGGER.warning(f"Error removing hourly sample listener: {e}")
            finally:
                self._hourly_sample_listener_remove = None

        if self._daily_training_task:
            self._daily_training_task.cancel()
            _LOGGER.debug("Cancelled scheduled daily training check timer.")
            self._daily_training_task = None

        _LOGGER.info("MLPredictor cleanup finished.")
