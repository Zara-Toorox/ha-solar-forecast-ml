# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from homeassistant.util import dt as dt_util

from .ai_tiny_lstm import TinyLSTM
from .ai_feature_engineering import FeatureEngineer
from .ai_seasonal import SeasonalAdjuster
from .ai_dni_tracker import DniTracker

_LOGGER = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 50
TRAINING_DAYS = 30

# Feature counts
BASE_FEATURE_COUNT = 17
GROUP_FEATURE_COUNT = 3  # Features per panel group


def calculate_feature_count(num_groups: int) -> int:
    """Calculate total feature count based on number of groups @zara

    - 0 groups: 17 base features only (single-output total prediction)
    - n groups: 17 base + 3*n group features (multi-output per-group prediction)
    """
    if num_groups <= 0:
        return BASE_FEATURE_COUNT
    return BASE_FEATURE_COUNT + GROUP_FEATURE_COUNT * num_groups

# Target normalization - max kWh per hour for scaling targets to 0-1 range
# This should be larger than any expected hourly production
MAX_KWH_PER_HOUR = 2.0


class ModelState(Enum):
    """State of AI model @zara"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    DEGRADED = "degraded"
    ERROR = "error"
    UNTRAINED = "untrained"


@dataclass
class TrainingResult:
    """Result of AI training @zara"""
    success: bool
    accuracy: float = 0.0
    samples_used: int = 0
    feature_count: int = BASE_FEATURE_COUNT
    num_outputs: int = 1
    error_message: Optional[str] = None
    epochs_trained: int = 0
    has_attention: bool = False
    rmse: float = 0.0


@dataclass
class GroupPrediction:
    """Prediction result for a single panel group @zara"""
    group_name: str
    power_kwh: float
    azimuth: float
    tilt: float
    capacity_kwp: float


@dataclass
class HourlyProfile:
    """Hourly production profile @zara"""
    hourly_averages: Dict[str, float] = field(default_factory=dict)


@dataclass
class LearnedWeights:
    """Learned weights container @zara"""
    feature_stds: Dict[str, float] = field(default_factory=dict)


class AIPredictor:
    """Multi-Output AI predictor with per-group predictions @zara"""

    def __init__(
        self,
        hass: Any,
        data_manager: Any,
        error_handler: Optional[Any] = None,
        notification_service: Optional[Any] = None,
        config_entry: Optional[Any] = None,
        panel_groups: Optional[List[Dict[str, Any]]] = None,
        solar_capacity: float = 5.0,
    ):
        """Initialize AI predictor @zara

        Args:
            hass: Home Assistant instance
            data_manager: Data manager for file storage
            error_handler: Optional error handler
            notification_service: Optional notification service
            config_entry: Optional config entry
            panel_groups: List of panel group configs with azimuth, tilt, capacity_kwp
            solar_capacity: Total solar capacity in kWp (used if no panel_groups)
        """
        self.hass = hass
        self.data_manager = data_manager
        self.error_handler = error_handler
        self.notification_service = notification_service
        self.config_entry = config_entry

        self._data_dir = Path(data_manager.data_dir) if data_manager else Path(".")
        self._ai_dir = self._data_dir / "ai"

        # Panel groups configuration
        self.panel_groups: List[Dict[str, Any]] = panel_groups or []
        self.num_groups: int = len(self.panel_groups) if self.panel_groups else 1
        # Use sum of group capacities if groups exist, otherwise use provided solar_capacity
        # Config stores capacity as power_wp (Watts), capacity_kwp (kWp), or kwp (kWp)
        self.total_capacity: float = self._calculate_total_capacity(
            self.panel_groups, solar_capacity
        )

        self.lstm: Optional[TinyLSTM] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.seasonal: Optional[SeasonalAdjuster] = None
        self.dni_tracker: Optional[DniTracker] = None

        self.state = ModelState.INITIALIZING
        self.current_accuracy: Optional[float] = None
        self.current_rmse: Optional[float] = None  # RMSE in kWh
        self.last_training_time: Optional[datetime] = None
        self.last_training_samples: int = 0
        self.training_samples: int = 0
        self.use_attention: bool = True  # Attention mechanism flag (default: ON since v2.0.0)

        self.current_profile: Optional[HourlyProfile] = None
        self.current_weights: Optional[LearnedWeights] = None

        self._historical_cache: Dict[str, Any] = {"daily_productions": {}, "hourly_data": {}}

        self.solar_capacity: float = self.total_capacity
        self.power_entity: Optional[str] = None
        self.weather_entity: Optional[str] = None

        _LOGGER.info(
            f"AIPredictor initialized: {self.num_groups} groups, "
            f"{self.total_capacity:.2f} kWp total"
        )

    def _calculate_total_capacity(
        self,
        panel_groups: List[Dict[str, Any]],
        fallback_capacity: float,
    ) -> float:
        """Calculate total capacity from panel groups @zara

        Config may store capacity as:
        - power_wp: Watts (needs /1000 conversion)
        - capacity_kwp: kWp (direct)
        - kwp: kWp (direct)
        """
        if not panel_groups:
            return fallback_capacity

        total = 0.0
        for group in panel_groups:
            # Try capacity_kwp first, then kwp, then power_wp (convert from Wp)
            if "capacity_kwp" in group:
                total += float(group["capacity_kwp"])
            elif "kwp" in group:
                total += float(group["kwp"])
            elif "power_wp" in group:
                total += float(group["power_wp"]) / 1000.0
            else:
                # Default 1 kWp per group if nothing specified
                total += 1.0

        return total if total > 0 else fallback_capacity

    def set_entities(
        self,
        solar_capacity: float = 5.0,
        power_entity: Optional[str] = None,
        weather_entity: Optional[str] = None,
        **kwargs
    ):
        """Set entity configuration @zara"""
        self.solar_capacity = solar_capacity
        self.power_entity = power_entity
        self.weather_entity = weather_entity

    async def initialize(self) -> bool:
        """Initialize AI components with Multi-Output LSTM @zara

        Architecture:
        - 0 groups: 17 base features, 1 output (total prediction)
        - n groups: 17 + 3*n features, n outputs (per-group predictions)
        """
        try:
            # Create directory in executor to avoid blocking
            import asyncio
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: self._ai_dir.mkdir(parents=True, exist_ok=True))

            # Initialize components without auto-loading (non-blocking)
            self.dni_tracker = DniTracker(self._ai_dir / "dni_tracker.json", auto_load=False)
            self.seasonal = SeasonalAdjuster(self._ai_dir / "seasonal.json", auto_load=False)

            # Load data asynchronously
            await self.dni_tracker.async_load()
            await self.seasonal.async_load()

            self.feature_engineer = FeatureEngineer(self.dni_tracker)

            # Calculate dynamic feature count and output count
            # - No groups: 17 features, 1 output (total production)
            # - n groups: 17 + 3*n features, n outputs (per-group production)
            num_outputs = self.num_groups if self.num_groups > 0 else 1
            feature_count = calculate_feature_count(self.num_groups)

            # Attention is now enabled by default (v2.0.0+)
            # Existing models will be upgraded to use attention on next training
            # New installations start with attention=True immediately
            weights_file = self._data_dir / "ai" / "learned_weights.json"
            use_attention = True  # Default: ON for all new installations
            weights_exists = await loop.run_in_executor(None, weights_file.exists)
            if weights_exists:
                try:
                    def _read_weights_check():
                        with open(weights_file, "r") as f:
                            return json.load(f)
                    saved_weights = await loop.run_in_executor(None, _read_weights_check)
                    # Use saved value if model already has attention weights
                    # Otherwise keep attention=True for upgrade
                    use_attention = saved_weights.get("has_attention", True)
                except Exception:
                    pass

            self.use_attention = use_attention

            self.lstm = TinyLSTM(
                input_size=feature_count,
                hidden_size=32,
                sequence_length=24,
                num_outputs=num_outputs,
                use_attention=self.use_attention,
            )

            _LOGGER.info(
                f"LSTM initialized: {feature_count} features, {num_outputs} outputs, "
                f"{self.num_groups} groups, attention={self.use_attention}"
            )

            # Load weights asynchronously
            weights_file = self._ai_dir / "learned_weights.json"
            weights_exists = await loop.run_in_executor(None, weights_file.exists)

            if weights_exists:
                try:
                    def _read_weights():
                        with open(weights_file, "r") as f:
                            return json.load(f)

                    weights = await loop.run_in_executor(None, _read_weights)

                    # Check if saved model has same architecture (input_size AND num_outputs)
                    saved_input_size = weights.get("input_size", 17)
                    saved_outputs = weights.get("num_outputs", 1)

                    if saved_input_size == feature_count and saved_outputs == num_outputs:
                        self.lstm.set_weights(weights)
                        self.training_samples = weights.get("training_samples", 0)
                        self.current_accuracy = weights.get("accuracy")
                        self.current_rmse = weights.get("rmse")  # RMSE in kWh
                        self.use_attention = weights.get("has_attention", False)
                        self.state = ModelState.READY
                        rmse_str = f", RMSE={self.current_rmse:.3f}kWh" if self.current_rmse else ""
                        attn_str = ", attention=ON" if self.use_attention else ""
                        _LOGGER.info(
                            f"AI model loaded: {self.training_samples} samples, "
                            f"{feature_count} features, {num_outputs} outputs, "
                            f"R²={self.current_accuracy:.3f}{rmse_str}{attn_str}" if self.current_accuracy else
                            f"AI model loaded: {self.training_samples} samples, "
                            f"{feature_count} features, {num_outputs} outputs{attn_str}"
                        )
                    else:
                        _LOGGER.warning(
                            f"Model architecture mismatch: saved input={saved_input_size}/outputs={saved_outputs}, "
                            f"needed input={feature_count}/outputs={num_outputs}. Retraining required."
                        )
                        self.state = ModelState.UNTRAINED
                except Exception as e:
                    _LOGGER.warning(f"Could not load weights: {e}")
                    self.state = ModelState.UNTRAINED
            else:
                self.state = ModelState.UNTRAINED
                _LOGGER.info(f"No weights found - model untrained ({num_outputs} outputs)")

            self.current_profile = HourlyProfile()
            self.current_weights = LearnedWeights()

            return True

        except Exception as e:
            _LOGGER.error(f"AI initialization failed: {e}")
            self.state = ModelState.ERROR
            return False

    def is_ready(self) -> bool:
        """Check if model is ready for predictions @zara"""
        return self.state == ModelState.READY and self.lstm is not None

    def is_healthy(self) -> bool:
        """Check if predictor is in healthy state @zara"""
        return self.state in (ModelState.READY, ModelState.UNTRAINED) and self.lstm is not None

    def get_base_ai_confidence(self) -> float:
        """Get base confidence level for AI predictions (0.0 - 1.0) @zara

        Based on R² score:
        - R² <= 0: 0% (AI worse than average)
        - R² 0-0.5: 0-50% (linear)
        - R² 0.5-0.8: 50-90% (linear)
        - R² > 0.8: 90-100%

        Note: This is the BASE confidence. Actual per-hour confidence
        is calculated in RuleBasedForecastStrategy with additional factors.
        """
        if not self.is_ready():
            return 0.0

        r2 = self.current_accuracy or 0.0

        if r2 <= 0:
            return 0.0
        elif r2 < 0.5:
            return r2  # 0.3 → 30%
        elif r2 < 0.8:
            return 0.5 + (r2 - 0.5) * 1.333  # Linear 50% → 90%
        else:
            return 0.9 + min((r2 - 0.8), 0.2) * 0.5  # Cap at 100%

    @property
    def base_ai_confidence(self) -> float:
        """Property accessor for base AI confidence @zara"""
        return self.get_base_ai_confidence()

    async def predict_hour(
        self,
        hour: int,
        weather_data: Dict[str, Any],
        astronomy_data: Dict[str, Any],
    ) -> float:
        """Predict total production for single hour (sum of all groups) @zara

        For per-group predictions, use predict_hour_per_group() instead.
        """
        if not self.is_ready():
            return 0.0

        try:
            group_predictions = await self.predict_hour_per_group(
                hour, weather_data, astronomy_data
            )
            total = sum(gp.power_kwh for gp in group_predictions)
            return max(0.0, total)

        except Exception as e:
            _LOGGER.error(f"Prediction failed: {e}")
            return 0.0

    async def predict_hour_per_group(
        self,
        hour: int,
        weather_data: Dict[str, Any],
        astronomy_data: Dict[str, Any],
    ) -> List[GroupPrediction]:
        """Predict production per panel group for single hour @zara

        Multi-Output mode: One forward pass with combined features → n outputs
        Single-Output mode: One forward pass with base features → 1 output (total)

        FIX #1: Now uses real 24-hour temporal sequences instead of repeating features.
        FIX #2: Now uses real historical lag-features instead of 0.0.

        Returns:
            List of GroupPrediction, one per panel group (or one "Total" if no groups)
        """
        if not self.is_ready():
            return []

        try:
            # Get seasonal factor
            month = datetime.now().month
            seasonal_factor = self.seasonal.get_factor(month) if self.seasonal else 1.0

            # Load historical predictions for building temporal sequence
            all_predictions = await self._load_predictions_for_sequence()
            target_date = datetime.now().strftime("%Y-%m-%d")

            # Build real 24-hour temporal sequence (FIX #1 & #2)
            sequence, prod_yesterday, prod_same_hour = await self._build_temporal_sequence(
                target_hour=hour,
                target_date=target_date,
                weather_data=weather_data,
                astronomy_data=astronomy_data,
                all_predictions=all_predictions,
            )

            results = []

            if self.panel_groups:
                # MULTI-OUTPUT MODE: Combined features for all groups
                # Single forward pass → multiple outputs (one per group)
                predictions = self.lstm.predict(sequence)  # Returns list of n values

                # Handle empty predictions array
                if not predictions or len(predictions) == 0:
                    predictions = []

                # Map each output to its corresponding group
                for i, group in enumerate(self.panel_groups):
                    if predictions and i < len(predictions):
                        # DENORMALIZE: multiply by MAX_KWH_PER_HOUR to get actual kWh
                        pred_kwh = predictions[i] * MAX_KWH_PER_HOUR * seasonal_factor
                    else:
                        pred_kwh = 0.0

                    results.append(GroupPrediction(
                        group_name=group.get("name", f"Group_{i+1}"),
                        power_kwh=max(0.0, pred_kwh),
                        azimuth=group.get("azimuth", 180.0),
                        tilt=group.get("tilt", 30.0),
                        capacity_kwp=group.get("capacity_kwp", group.get("kwp", 1.0)),
                    ))
            else:
                # SINGLE-OUTPUT MODE: Only base features (17)
                # Predict total production
                predictions = self.lstm.predict(sequence)
                # DENORMALIZE: multiply by MAX_KWH_PER_HOUR to get actual kWh
                pred_kwh = predictions[0] * MAX_KWH_PER_HOUR * seasonal_factor if predictions else 0.0

                results.append(GroupPrediction(
                    group_name="Total",
                    power_kwh=max(0.0, pred_kwh),
                    azimuth=180.0,
                    tilt=30.0,
                    capacity_kwp=self.solar_capacity,
                ))

            return results

        except Exception as e:
            _LOGGER.error(f"Per-group prediction failed: {e}")
            return []

    async def train_model(self) -> TrainingResult:
        """Train Multi-Output LSTM model with historical data @zara

        Training uses per-group production data when available.
        Falls back to total production if group data is missing.
        """
        if not self.lstm or not self.feature_engineer:
            return TrainingResult(
                success=False,
                error_message="LSTM or FeatureEngineer not initialized"
            )

        self.state = ModelState.TRAINING
        _LOGGER.info(f"AI training started: {self.num_groups} outputs")

        try:
            # Load training data from hourly_predictions
            X_sequences, y_targets, daily_productions = await self._prepare_training_data()

            if len(X_sequences) < MIN_TRAINING_SAMPLES:
                self.state = ModelState.UNTRAINED
                return TrainingResult(
                    success=False,
                    samples_used=len(X_sequences),
                    error_message=f"Need {MIN_TRAINING_SAMPLES} samples, got {len(X_sequences)}"
                )

            feature_count = calculate_feature_count(self.num_groups)
            num_outputs = self.num_groups if self.num_groups > 0 else 1
            _LOGGER.info(
                f"Training with {len(X_sequences)} samples, "
                f"{feature_count} features, {num_outputs} outputs"
            )

            # Train the LSTM
            result = await self.lstm.train(
                X_sequences=X_sequences,
                y_targets=y_targets,
                epochs=200,
                batch_size=16,
                validation_split=0.2,
                early_stopping_patience=20
            )

            if result.get("success"):
                # Save weights
                self.training_samples = len(X_sequences)
                self.current_accuracy = result.get("accuracy", 0.0)
                self.current_rmse = result.get("rmse", 0.0)  # RMSE in kWh
                self.last_training_time = dt_util.now()
                self.last_training_samples = len(X_sequences)

                await self._save_weights_async()

                # Update seasonal factors based on daily productions
                await self._update_seasonal_factors(daily_productions)

                # Update DNI tracker
                await self._update_dni_tracker()

                self.state = ModelState.READY
                rmse_str = f", RMSE={self.current_rmse:.3f}kWh" if self.current_rmse else ""
                attn_str = ", attention=ON" if self.use_attention else ""
                _LOGGER.info(
                    f"Training complete: R2={self.current_accuracy:.3f}{rmse_str}, "
                    f"samples={len(X_sequences)}, outputs={self.num_groups}{attn_str}"
                )

                return TrainingResult(
                    success=True,
                    accuracy=self.current_accuracy,
                    samples_used=len(X_sequences),
                    feature_count=calculate_feature_count(self.num_groups),
                    num_outputs=self.num_groups if self.num_groups > 0 else 1,
                    epochs_trained=result.get("epochs_trained", 0),
                    has_attention=self.use_attention,
                    rmse=self.current_rmse or 0.0,
                )
            else:
                self.state = ModelState.ERROR
                return TrainingResult(
                    success=False,
                    error_message="LSTM training failed"
                )

        except Exception as e:
            _LOGGER.error(f"Training failed: {e}")
            self.state = ModelState.ERROR
            return TrainingResult(
                success=False,
                error_message=str(e)
            )

    async def _prepare_training_data(self) -> Tuple[List[List[List[float]]], List[Any], Dict[str, float]]:
        """Load and prepare Multi-Output training data from hourly_predictions @zara

        FIX #1: Now creates real 24-hour temporal sequences instead of repeating features.
        FIX #2: Now uses real historical lag-features instead of 0.0.

        Returns:
            Tuple of:
            - X_sequences: List of feature sequences (each is 24 x 20 features)
            - y_targets: List of targets (List[float] per sample for multi-output)
            - daily_productions: Dict of date -> total production
        """
        X_sequences = []
        y_targets = []
        daily_productions: Dict[str, float] = {}

        try:
            # Load hourly predictions file
            predictions_file = self._data_dir / "stats" / "hourly_predictions.json"
            if not predictions_file.exists():
                _LOGGER.warning("No hourly_predictions.json found")
                return X_sequences, y_targets, daily_productions

            def _read_file():
                with open(predictions_file, "r") as f:
                    return json.load(f)

            data = await self.hass.async_add_executor_job(_read_file)
            predictions = data.get("predictions", [])

            # Build lookup dict for all predictions (needed for sequence building)
            predictions_by_id: Dict[str, Dict[str, Any]] = {}
            for p in predictions:
                pid = p.get("id")
                if pid:
                    predictions_by_id[pid] = p

            # Get dates from last TRAINING_DAYS days
            today = datetime.now().date()
            cutoff = today - timedelta(days=TRAINING_DAYS)

            for hour_data in predictions:
                self._process_hour_data_for_training(
                    hour_data, cutoff, today,
                    X_sequences, y_targets, daily_productions,
                    predictions_by_id  # Pass for sequence building
                )

            _LOGGER.info(
                f"Prepared {len(X_sequences)} training samples from "
                f"{len(daily_productions)} days, {self.num_groups} outputs"
            )

        except Exception as e:
            _LOGGER.error(f"Error preparing training data: {e}")

        return X_sequences, y_targets, daily_productions

    def _process_hour_data_for_training(
        self,
        hour_data: Dict[str, Any],
        cutoff: Any,
        today: Any,
        X_sequences: List,
        y_targets: List,
        daily_productions: Dict[str, float],
        predictions_by_id: Dict[str, Dict[str, Any]],
    ) -> None:
        """Process single hour data for Multi-Output training @zara

        FIX #1: Now creates real 24-hour temporal sequences using historical data.
        FIX #2: Now uses real historical lag-features instead of 0.0.

        CORRECT Multi-Output approach:
        - Features: 17 base + 3 per group = 17 + 3*n total
        - Targets: n values (one per group)

        The network sees ALL group information (azimuth, tilt, capacity for each)
        and predicts ALL group outputs simultaneously.
        """
        date_str = hour_data.get("target_date")
        if not date_str:
            return

        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            if date_obj < cutoff or date_obj >= today:
                return
        except ValueError:
            return

        target_hour = hour_data.get("target_hour")
        if target_hour is None:
            return

        # Get actual production - either total or per-group
        actual_kwh = hour_data.get("actual_kwh")

        # Per-group values: try panel_group_actuals (dict) first
        panel_group_actuals = hour_data.get("panel_group_actuals")  # Dict: {"Gruppe 1": 0.35, "Gruppe 2": 0.30}

        if actual_kwh is None or actual_kwh <= 0:
            return

        # Skip excluded hours (frost, clipping, alerts)
        flags = hour_data.get("flags", {}) or {}
        if flags.get("exclude_from_learning") or flags.get("inverter_clipped"):
            return

        # FIX #2: Calculate real lag-features from historical data
        target_dt = datetime.strptime(date_str, "%Y-%m-%d")
        yesterday_date = (target_dt - timedelta(days=1)).strftime("%Y-%m-%d")

        # Production yesterday (sum of all hours)
        production_yesterday = 0.0
        for h in range(24):
            pred_id = f"{yesterday_date}_{h}"
            pred = predictions_by_id.get(pred_id)
            if pred and pred.get("actual_kwh") is not None:
                production_yesterday += pred.get("actual_kwh", 0.0)

        # Production same hour yesterday
        same_hour_yesterday_id = f"{yesterday_date}_{target_hour}"
        same_hour_pred = predictions_by_id.get(same_hour_yesterday_id)
        production_same_hour_yesterday = (
            same_hour_pred.get("actual_kwh", 0.0)
            if same_hour_pred and same_hour_pred.get("actual_kwh") is not None
            else 0.0
        )

        # FIX #1: Build real 24-hour temporal sequence
        sequence = []
        for offset in range(-23, 1):  # -23 to 0 (inclusive), giving 24 timesteps
            seq_dt = target_dt.replace(hour=target_hour) + timedelta(hours=offset)
            seq_date = seq_dt.strftime("%Y-%m-%d")
            seq_hour = seq_dt.hour

            pred_id = f"{seq_date}_{seq_hour}"
            hist_pred = predictions_by_id.get(pred_id)

            if hist_pred and offset < 0:
                # Use historical data for past hours
                hist_weather = hist_pred.get("weather_corrected") or hist_pred.get("weather_forecast") or {}
                hist_astronomy = hist_pred.get("astronomy") or {}

                # Calculate lag features relative to this sequence position
                seq_yesterday_date = (seq_dt - timedelta(days=1)).strftime("%Y-%m-%d")
                seq_prod_yesterday = 0.0
                for h in range(24):
                    pid = f"{seq_yesterday_date}_{h}"
                    p = predictions_by_id.get(pid)
                    if p and p.get("actual_kwh") is not None:
                        seq_prod_yesterday += p.get("actual_kwh", 0.0)

                seq_same_hour_id = f"{seq_yesterday_date}_{seq_hour}"
                seq_same_hour_pred = predictions_by_id.get(seq_same_hour_id)
                seq_prod_same_hour = (
                    seq_same_hour_pred.get("actual_kwh", 0.0)
                    if seq_same_hour_pred and seq_same_hour_pred.get("actual_kwh") is not None
                    else 0.0
                )

                record = {
                    "target_hour": seq_hour,
                    "target_day_of_year": seq_dt.timetuple().tm_yday,
                    "target_month": seq_dt.month,
                    "weather_corrected": {
                        "temperature": hist_weather.get("temperature", hist_weather.get("temperature_c", 15)),
                        "solar_radiation_wm2": hist_weather.get("solar_radiation_wm2", 0),
                        "wind": hist_weather.get("wind", hist_weather.get("wind_speed", 3)),
                        "humidity": hist_weather.get("humidity", hist_weather.get("humidity_percent", 70)),
                        "rain": hist_weather.get("rain", hist_weather.get("precipitation", 0)),
                        "clouds": hist_weather.get("clouds", hist_weather.get("cloud_cover", 50)),
                        "dni": hist_weather.get("direct_radiation", 0),
                    },
                    "astronomy": {
                        "sun_elevation_deg": hist_astronomy.get("sun_elevation_deg", 0),
                        "theoretical_max_kwh": hist_astronomy.get("theoretical_max_kwh", 0),
                        "clear_sky_radiation_wm2": hist_astronomy.get("clear_sky_radiation_wm2", 0),
                        "max_elevation_today": hist_astronomy.get("max_elevation_today", 60),
                    },
                    "production_yesterday": seq_prod_yesterday,
                    "production_same_hour_yesterday": seq_prod_same_hour,
                }
            else:
                # Use current hour's data for target position (offset == 0)
                weather = hour_data.get("weather_corrected", {}) or {}
                astronomy = hour_data.get("astronomy", {}) or {}

                record = {
                    "target_hour": target_hour,
                    "target_day_of_year": target_dt.timetuple().tm_yday,
                    "target_month": target_dt.month,
                    "weather_corrected": {
                        "temperature": weather.get("temperature", weather.get("temperature_c", 15)),
                        "solar_radiation_wm2": weather.get("solar_radiation_wm2", 0),
                        "wind": weather.get("wind", weather.get("wind_speed", 3)),
                        "humidity": weather.get("humidity", weather.get("humidity_percent", 70)),
                        "rain": weather.get("rain", weather.get("precipitation", 0)),
                        "clouds": weather.get("clouds", weather.get("cloud_cover", 50)),
                        "dni": weather.get("direct_radiation", 0),
                    },
                    "astronomy": {
                        "sun_elevation_deg": astronomy.get("sun_elevation_deg", 0),
                        "theoretical_max_kwh": astronomy.get("theoretical_max_kwh", 0),
                        "clear_sky_radiation_wm2": astronomy.get("clear_sky_radiation_wm2", 0),
                        "max_elevation_today": astronomy.get("max_elevation_today", 60),
                    },
                    "production_yesterday": production_yesterday,
                    "production_same_hour_yesterday": production_same_hour_yesterday,
                }

            # Extract features based on mode
            if self.panel_groups:
                features = self.feature_engineer.extract_combined_for_multi_output(
                    record, self.panel_groups
                )
            else:
                features = self.feature_engineer.extract_base_features(record)

            if features is None:
                # Fallback: create zero features if extraction fails
                feature_count = calculate_feature_count(self.num_groups)
                features = [0.0] * feature_count

            sequence.append(features)

        # Build targets
        if self.panel_groups:
            # Build per-group targets (NORMALIZED to 0-1 range!)
            if panel_group_actuals and isinstance(panel_group_actuals, dict):
                # We have per-group actual values - extract in group order
                targets = []
                for group in self.panel_groups:
                    group_name = group.get("name", "")
                    group_kwh = panel_group_actuals.get(group_name, 0.0)
                    targets.append(float(group_kwh) / MAX_KWH_PER_HOUR)
            else:
                # Only total available - distribute proportionally by capacity
                targets = []
                for group in self.panel_groups:
                    capacity = group.get("capacity_kwp", group.get("kwp", 1.0))
                    ratio = capacity / self.total_capacity if self.total_capacity > 0 else 1.0 / self.num_groups
                    targets.append((actual_kwh * ratio) / MAX_KWH_PER_HOUR)
        else:
            # SINGLE-OUTPUT MODE
            targets = [actual_kwh / MAX_KWH_PER_HOUR]

        X_sequences.append(sequence)
        y_targets.append(targets)

        # Track daily productions (total)
        if date_str not in daily_productions:
            daily_productions[date_str] = 0.0
        daily_productions[date_str] += actual_kwh

    def _build_training_record(self, hour_data: Dict[str, Any], date_str: str) -> Optional[Dict[str, Any]]:
        """Build a record for feature extraction from hourly data @zara"""
        try:
            target_hour = hour_data.get("target_hour")
            if target_hour is None:
                return None

            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            day_of_year = date_obj.timetuple().tm_yday

            weather = hour_data.get("weather_corrected", {}) or {}
            astronomy = hour_data.get("astronomy", {}) or {}

            return {
                "target_hour": target_hour,
                "target_day_of_year": day_of_year,
                "target_month": date_obj.month,
                "weather_corrected": {
                    "temperature": weather.get("temperature", 15),
                    "solar_radiation_wm2": weather.get("solar_radiation_wm2", 0),
                    "wind": weather.get("wind_speed", 3),
                    "humidity": weather.get("humidity", 70),
                    "rain": weather.get("precipitation", 0),
                    "clouds": weather.get("clouds", weather.get("cloud_cover", 50)),
                    "dni": weather.get("direct_radiation", 0),
                },
                "astronomy": {
                    "sun_elevation_deg": astronomy.get("sun_elevation_deg", 30),
                    "theoretical_max_kwh": astronomy.get("theoretical_max_kwh", 0.5),
                    "clear_sky_radiation_wm2": astronomy.get("clear_sky_radiation_wm2", 500),
                    "max_elevation_today": astronomy.get("max_elevation_today", 60),
                },
                "production_yesterday": 0.0,
                "production_same_hour_yesterday": 0.0,
            }
        except Exception as e:
            _LOGGER.debug(f"Error building training record: {e}")
            return None

    async def _update_seasonal_factors(self, daily_productions: Dict[str, float]):
        """Update seasonal adjustment factors based on production data @zara"""
        if not self.seasonal or not daily_productions:
            return

        try:
            # Group by month
            monthly_totals: Dict[int, List[float]] = {}
            for date_str, production in daily_productions.items():
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    month = date_obj.month
                    if month not in monthly_totals:
                        monthly_totals[month] = []
                    monthly_totals[month].append(production)
                except ValueError:
                    continue

            # Calculate averages and update seasonal using update(month, actual, predicted)
            for month, values in monthly_totals.items():
                if len(values) >= 3:
                    avg_actual = sum(values) / len(values)
                    # Use current factor as "predicted" baseline
                    current_factor = self.seasonal.get_factor(month)
                    baseline = avg_actual / current_factor if current_factor > 0 else avg_actual
                    self.seasonal.update(month, avg_actual, baseline)

            await self.seasonal.async_save(self.hass)
            _LOGGER.debug(f"Updated seasonal factors for {len(monthly_totals)} months")

        except Exception as e:
            _LOGGER.warning(f"Error updating seasonal factors: {e}")

    async def _update_dni_tracker(self):
        """Update DNI tracker with today's data @zara"""
        if not self.dni_tracker:
            return

        try:
            # Load today's hourly predictions
            today = datetime.now().date().isoformat()
            predictions_file = self._data_dir / "stats" / "hourly_predictions.json"

            if not predictions_file.exists():
                return

            def _read_file():
                with open(predictions_file, "r") as f:
                    return json.load(f)

            data = await self.hass.async_add_executor_job(_read_file)
            predictions = data.get("predictions", [])

            for hour_data in predictions:
                if hour_data.get("target_date") != today:
                    continue
                weather = hour_data.get("weather_corrected") or {}
                dni = weather.get("direct_radiation", 0)
                if dni > 0:
                    hour = hour_data.get("target_hour", 0)
                    self.dni_tracker.record_dni(hour, dni)

            await self.dni_tracker.async_save(self.hass)

        except Exception as e:
            _LOGGER.warning(f"Error updating DNI tracker: {e}")

    async def _build_temporal_sequence(
        self,
        target_hour: int,
        target_date: str,
        weather_data: Dict[str, Any],
        astronomy_data: Dict[str, Any],
        all_predictions: List[Dict[str, Any]],
    ) -> Tuple[List[List[float]], float, float]:
        """Build a real 24-hour temporal sequence for LSTM prediction @zara

        FIX #1: Creates a proper temporal sequence using the last 24 hours of data
        instead of repeating the same features 24 times.

        FIX #2: Fills lag-features (production_yesterday, production_same_hour_yesterday)
        with real historical values instead of 0.0.

        Args:
            target_hour: The hour we're predicting for (0-23)
            target_date: Date string YYYY-MM-DD
            weather_data: Weather data for target hour
            astronomy_data: Astronomy data for target hour
            all_predictions: List of all hourly predictions for lookback

        Returns:
            Tuple of (24-step sequence, production_yesterday, production_same_hour_yesterday)
        """
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            target_dt = datetime.now()
            target_date = target_dt.strftime("%Y-%m-%d")

        # Build lookup dict for fast access to historical predictions
        predictions_by_id = {}
        for p in all_predictions:
            pid = p.get("id")
            if pid:
                predictions_by_id[pid] = p

        # Calculate lag-features from real historical data (FIX #2)
        yesterday_date = (target_dt - timedelta(days=1)).strftime("%Y-%m-%d")

        # Production yesterday (sum of all hours)
        production_yesterday = 0.0
        for h in range(24):
            pred_id = f"{yesterday_date}_{h}"
            pred = predictions_by_id.get(pred_id)
            if pred and pred.get("actual_kwh") is not None:
                production_yesterday += pred.get("actual_kwh", 0.0)

        # Production same hour yesterday
        same_hour_yesterday_id = f"{yesterday_date}_{target_hour}"
        same_hour_pred = predictions_by_id.get(same_hour_yesterday_id)
        production_same_hour_yesterday = (
            same_hour_pred.get("actual_kwh", 0.0)
            if same_hour_pred and same_hour_pred.get("actual_kwh") is not None
            else 0.0
        )

        # Build 24-step sequence (FIX #1)
        # We use the 23 hours before target_hour + target_hour itself
        sequence = []

        for offset in range(-23, 1):  # -23 to 0 (inclusive), giving 24 timesteps
            # Calculate the datetime for this sequence step
            seq_dt = target_dt.replace(hour=target_hour) + timedelta(hours=offset)
            seq_date = seq_dt.strftime("%Y-%m-%d")
            seq_hour = seq_dt.hour

            pred_id = f"{seq_date}_{seq_hour}"
            hist_pred = predictions_by_id.get(pred_id)

            if hist_pred and offset < 0:
                # Use historical data for past hours
                hist_weather = hist_pred.get("weather_corrected") or hist_pred.get("weather_forecast") or {}
                hist_astronomy = hist_pred.get("astronomy") or {}

                # Calculate lag features relative to this sequence position
                seq_yesterday_date = (seq_dt - timedelta(days=1)).strftime("%Y-%m-%d")
                seq_prod_yesterday = 0.0
                for h in range(24):
                    pid = f"{seq_yesterday_date}_{h}"
                    p = predictions_by_id.get(pid)
                    if p and p.get("actual_kwh") is not None:
                        seq_prod_yesterday += p.get("actual_kwh", 0.0)

                seq_same_hour_id = f"{seq_yesterday_date}_{seq_hour}"
                seq_same_hour_pred = predictions_by_id.get(seq_same_hour_id)
                seq_prod_same_hour = (
                    seq_same_hour_pred.get("actual_kwh", 0.0)
                    if seq_same_hour_pred and seq_same_hour_pred.get("actual_kwh") is not None
                    else 0.0
                )

                record = {
                    "target_hour": seq_hour,
                    "target_day_of_year": seq_dt.timetuple().tm_yday,
                    "target_month": seq_dt.month,
                    "weather_corrected": {
                        "temperature": hist_weather.get("temperature", hist_weather.get("temperature_c", 15)),
                        "solar_radiation_wm2": hist_weather.get("solar_radiation_wm2", 0),
                        "wind": hist_weather.get("wind", hist_weather.get("wind_speed", 3)),
                        "humidity": hist_weather.get("humidity", hist_weather.get("humidity_percent", 70)),
                        "rain": hist_weather.get("rain", hist_weather.get("precipitation", 0)),
                        "clouds": hist_weather.get("clouds", hist_weather.get("cloud_cover", 50)),
                        "dni": hist_weather.get("direct_radiation", 0),
                    },
                    "astronomy": {
                        "sun_elevation_deg": hist_astronomy.get("sun_elevation_deg", 0),
                        "theoretical_max_kwh": hist_astronomy.get("theoretical_max_kwh", 0),
                        "clear_sky_radiation_wm2": hist_astronomy.get("clear_sky_radiation_wm2", 0),
                        "max_elevation_today": hist_astronomy.get("max_elevation_today", 60),
                    },
                    "production_yesterday": seq_prod_yesterday,
                    "production_same_hour_yesterday": seq_prod_same_hour,
                }
            else:
                # Use provided data for target hour (offset == 0) or if no historical data
                record = {
                    "target_hour": seq_hour if offset < 0 else target_hour,
                    "target_day_of_year": seq_dt.timetuple().tm_yday if offset < 0 else target_dt.timetuple().tm_yday,
                    "target_month": seq_dt.month if offset < 0 else target_dt.month,
                    "weather_corrected": weather_data,
                    "astronomy": astronomy_data,
                    "production_yesterday": production_yesterday,
                    "production_same_hour_yesterday": production_same_hour_yesterday,
                }

            # Extract features based on mode
            if self.panel_groups:
                features = self.feature_engineer.extract_combined_for_multi_output(
                    record, self.panel_groups
                )
            else:
                features = self.feature_engineer.extract_base_features(record)

            if features is None:
                # Fallback: create zero features if extraction fails
                feature_count = calculate_feature_count(self.num_groups)
                features = [0.0] * feature_count

            sequence.append(features)

        return sequence, production_yesterday, production_same_hour_yesterday

    async def _load_predictions_for_sequence(self) -> List[Dict[str, Any]]:
        """Load recent predictions for building temporal sequences @zara"""
        try:
            predictions_file = self._data_dir / "stats" / "hourly_predictions.json"
            if not predictions_file.exists():
                return []

            def _read():
                with open(predictions_file, "r") as f:
                    return json.load(f)

            data = await self.hass.async_add_executor_job(_read)
            return data.get("predictions", [])
        except Exception as e:
            _LOGGER.debug(f"Could not load predictions for sequence: {e}")
            return []

    async def _save_weights_async(self):
        """Save LSTM weights asynchronously with backup @zara"""
        if not self.lstm:
            return

        try:
            weights = self.lstm.get_weights()
            weights["training_samples"] = self.training_samples
            weights["last_trained"] = datetime.now().isoformat()
            weights["accuracy"] = self.current_accuracy
            weights["rmse"] = self.current_rmse  # RMSE in kWh

            weights_file = self._ai_dir / "learned_weights.json"
            backup_file = self._ai_dir / "learned_weights.backup.json"

            def _write_with_backup():
                import shutil
                # Create backup of existing weights before overwriting
                if weights_file.exists():
                    shutil.copy2(weights_file, backup_file)

                # Write new weights atomically
                temp_file = weights_file.with_suffix(".tmp")
                with open(temp_file, "w") as f:
                    json.dump(weights, f, indent=2)
                temp_file.replace(weights_file)

            await self.hass.async_add_executor_job(_write_with_backup)

            attn_str = " (attention)" if weights.get("has_attention") else ""
            _LOGGER.info(f"Weights saved{attn_str}")

        except Exception as e:
            _LOGGER.error(f"Failed to save weights: {e}")

    def save_weights(self):
        """Save LSTM weights to file (sync version) @zara"""
        if not self.lstm:
            return

        try:
            weights = self.lstm.get_weights()
            weights["training_samples"] = self.training_samples
            weights_file = self._ai_dir / "learned_weights.json"
            with open(weights_file, "w") as f:
                json.dump(weights, f, indent=2)
            _LOGGER.info("Weights saved")
        except Exception as e:
            _LOGGER.error(f"Failed to save weights: {e}")

    async def enable_attention(self, enable: bool = True) -> bool:
        """Enable or disable attention mechanism @zara

        This will reinitialize the LSTM with the new attention setting.
        The model will need to be retrained after enabling attention.

        Args:
            enable: True to enable attention, False to disable

        Returns:
            True if successful, False otherwise
        """
        if not self.lstm:
            _LOGGER.warning("Cannot enable attention: LSTM not initialized")
            return False

        if self.use_attention == enable:
            _LOGGER.info(f"Attention already {'enabled' if enable else 'disabled'}")
            return True

        try:
            # Get current architecture params
            feature_count = calculate_feature_count(self.num_groups)
            num_outputs = self.num_groups if self.num_groups > 0 else 1

            # Create new LSTM with attention setting
            self.lstm = TinyLSTM(
                input_size=feature_count,
                hidden_size=32,
                sequence_length=24,
                num_outputs=num_outputs,
                use_attention=enable,
            )

            self.use_attention = enable
            self.state = ModelState.UNTRAINED  # Needs retraining

            _LOGGER.info(
                f"Attention {'enabled' if enable else 'disabled'} - "
                f"model needs retraining"
            )
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to enable attention: {e}")
            return False

    def get_attention_weights(
        self,
        sequence: List[List[float]]
    ) -> Optional[List[float]]:
        """Get attention weights for interpretability @zara

        Returns the attention weights showing which of the 24 input hours
        were most important for the prediction.

        Args:
            sequence: 24-step input sequence (24 x features)

        Returns:
            List of 24 attention weights (sum to 1.0), or None if attention disabled
        """
        if not self.lstm or not self.use_attention:
            return None

        try:
            import numpy as np

            # Run forward pass with training=True to get attention cache
            X = np.array(sequence)
            _, cache = self.lstm.forward(X, training=True)

            if cache and 'attn_weights' in cache:
                attn_weights = cache['attn_weights']
                return attn_weights.tolist() if hasattr(attn_weights, 'tolist') else list(attn_weights)

            return None

        except Exception as e:
            _LOGGER.debug(f"Could not get attention weights: {e}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for diagnostics @zara"""
        info = {
            "state": self.state.value if self.state else "unknown",
            "training_samples": self.training_samples,
            "accuracy": self.current_accuracy,
            "rmse": self.current_rmse,
            "num_groups": self.num_groups,
            "total_capacity_kwp": self.total_capacity,
            "use_attention": self.use_attention,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
        }

        if self.lstm:
            info["model_size_kb"] = self.lstm.get_model_size_kb()
            info["feature_count"] = self.lstm.input_size
            info["hidden_size"] = self.lstm.hidden_size
            info["num_outputs"] = self.lstm.num_outputs

        return info

    async def rollback_to_backup(self) -> bool:
        """Rollback to previous model weights from backup @zara

        Use this when the current model is performing poorly or
        when a training run produced worse results.

        Returns:
            True if rollback successful, False otherwise
        """
        try:
            weights_file = self._ai_dir / "learned_weights.json"
            backup_file = self._ai_dir / "learned_weights.backup.json"

            if not backup_file.exists():
                _LOGGER.warning("No backup weights available for rollback")
                return False

            def _do_rollback():
                import shutil
                # Save current (failing) weights for debugging
                if weights_file.exists():
                    failed_file = self._ai_dir / "learned_weights.failed.json"
                    shutil.copy2(weights_file, failed_file)

                # Restore from backup
                shutil.copy2(backup_file, weights_file)

            await self.hass.async_add_executor_job(_do_rollback)

            # Reload the restored weights
            def _read_weights():
                with open(weights_file, "r") as f:
                    return json.load(f)

            weights = await self.hass.async_add_executor_job(_read_weights)

            # Update LSTM with restored weights
            if self.lstm:
                self.lstm.set_weights(weights)
                self.training_samples = weights.get("training_samples", 0)
                self.current_accuracy = weights.get("accuracy")
                self.current_rmse = weights.get("rmse")
                self.use_attention = weights.get("has_attention", False)
                self.state = ModelState.READY

            attn_str = " (attention)" if self.use_attention else ""
            _LOGGER.info(
                f"Model rollback successful{attn_str}: "
                f"R²={self.current_accuracy:.3f}, samples={self.training_samples}"
            )
            return True

        except Exception as e:
            _LOGGER.error(f"Rollback failed: {e}")
            return False

    def has_backup_available(self) -> bool:
        """Check if a backup is available for rollback @zara"""
        backup_file = self._ai_dir / "learned_weights.backup.json"
        return backup_file.exists()

    async def get_backup_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the backup model @zara

        Returns:
            Dict with backup model info, or None if no backup exists
        """
        backup_file = self._ai_dir / "learned_weights.backup.json"

        if not backup_file.exists():
            return None

        try:
            def _read():
                with open(backup_file, "r") as f:
                    return json.load(f)

            weights = await self.hass.async_add_executor_job(_read)

            return {
                "accuracy": weights.get("accuracy"),
                "rmse": weights.get("rmse"),
                "training_samples": weights.get("training_samples", 0),
                "last_trained": weights.get("last_trained"),
                "has_attention": weights.get("has_attention", False),
                "input_size": weights.get("input_size"),
                "num_outputs": weights.get("num_outputs", 1),
            }

        except Exception as e:
            _LOGGER.debug(f"Could not read backup info: {e}")
            return None
