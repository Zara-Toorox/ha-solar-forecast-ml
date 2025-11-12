"""
ML Training Helpers V2 - For new data structure

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path

try:
    from ..const import MIN_TRAINING_DATA_POINTS
    from ..core.core_helpers import SafeDateTimeUtil as dt_util
except ImportError:
    # Fallback for standalone testing
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from const import MIN_TRAINING_DATA_POINTS
    from core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class MLTrainingHelpersV2:
    """Helper methods for ML training process using V2 data structure"""

    def __init__(self, predictor: "MLPredictor"):
        """Initialize training helpers"""
        self.predictor = predictor

    async def load_training_data_v2(self) -> Tuple[List[Dict[str, Any]], int]:
        """Load training data from hourly_predictions.json + astronomy_cache.json"""
        from .ml_data_loader_v2 import MLDataLoaderV2

        try:
            data_dir = Path(self.predictor.data_manager.data_dir)
            loader = MLDataLoaderV2(data_dir)

            # Run in executor to avoid blocking I/O
            records, count = await self.predictor.hass.async_add_executor_job(
                loader.load_training_data,
                10,  # min_samples
                True  # exclude_outliers
            )

            _LOGGER.info(f"Loaded {count} training records from V2 data structure")
            return records, count

        except Exception as e:
            _LOGGER.error(f"Failed to load V2 training data: {e}", exc_info=True)
            return [], 0

    async def validate_training_data(self, training_records: List[Dict]) -> Tuple[bool, str, int]:
        """Validate training data meets minimum requirements"""
        valid_records_count = len(training_records)
        absolute_min_samples = 10  # Lowered for V2 initial testing
        recommended_min_samples = 30  # Recommended for good accuracy

        if valid_records_count < absolute_min_samples:
            error_msg = (
                f"Insufficient training data: {valid_records_count} valid records found "
                f"(absolute minimum: {absolute_min_samples}). Training aborted."
            )
            _LOGGER.warning(error_msg)
            return False, error_msg, valid_records_count

        if valid_records_count < recommended_min_samples:
            _LOGGER.warning(
                f"Training with {valid_records_count} samples (below recommended {recommended_min_samples}). "
                f"Model accuracy may be lower than optimal."
            )

        return True, "", valid_records_count

    async def prepare_training_features_v2(
        self,
        training_records: List[Dict]
    ) -> Tuple[List[List[float]], List[float]]:
        """Extract features and labels from training records using V2 feature engineering"""
        from .ml_feature_engineering_v2 import FeatureEngineerV2

        feature_eng = FeatureEngineerV2()
        X_train_raw = []
        y_train = []

        for record in training_records:
            # Extract features
            features = feature_eng.extract_features(record)
            if features is None:
                _LOGGER.debug(f"Skipping record, feature extraction failed: {record.get('timestamp')}")
                continue

            # Get target
            actual_kwh = record.get('actual_kwh')
            if actual_kwh is None:
                _LOGGER.debug(f"Skipping record without actual_kwh: {record.get('timestamp')}")
                continue

            X_train_raw.append(features)
            y_train.append(actual_kwh)

        _LOGGER.info(f"Feature extraction complete: {len(X_train_raw)} samples with {len(features)} features each")
        return X_train_raw, y_train

    async def scale_and_train(
        self,
        X_train_raw: List[List[float]],
        y_train: List[float]
    ) -> Tuple[Dict, float, float, float, "FeatureEngineerV2"]:
        """Scale features and train model

        Returns:
            Tuple of (weights_dict_raw, bias, accuracy, best_lambda, feature_engineer_v2)
        """
        from .ml_feature_engineering_v2 import FeatureEngineerV2

        feature_eng = FeatureEngineerV2()
        feature_names = feature_eng.feature_names

        X_train_scaled_list = await self.predictor.hass.async_add_executor_job(
            self.predictor.scaler.fit_transform,
            X_train_raw,
            feature_names
        )
        _LOGGER.info(
            f"Scaler fitted and training data transformed ({len(feature_names)} features)"
        )

        weights_dict_raw, bias, accuracy, best_lambda = await self.predictor.hass.async_add_executor_job(
            self.predictor.trainer.train, X_train_scaled_list, y_train
        )
        _LOGGER.info(
            f"Ridge training complete. Accuracy (R-squared): {accuracy:.4f}, Best Lambda: {best_lambda:.4f}"
        )

        return weights_dict_raw, bias, accuracy, best_lambda, feature_eng

    async def create_learned_weights(
        self,
        weights_dict_raw: Dict,
        bias: float,
        accuracy: float,
        samples_used: int,
        training_start_time: datetime,
        feature_engineer_v2: "FeatureEngineerV2"
    ) -> Any:
        """Create learned weights structure

        Args:
            weights_dict_raw: Raw weights from training (feature_0, feature_1, ...)
            bias: Model bias
            accuracy: Model R² accuracy
            samples_used: Number of training samples
            training_start_time: Training start timestamp
            feature_engineer_v2: V2 Feature Engineer instance with V2 feature names

        Returns:
            LearnedWeights object
        """
        from ..ml.ml_types import LearnedWeights
        from ..const import ML_MODEL_VERSION, CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX

        # Map raw weights to V2 feature names
        mapped_weights = self.predictor.trainer.map_weights_to_features(
            weights_dict_raw,
            feature_engineer_v2.feature_names  # Use V2 feature names (44 features)
        )

        # Preserve correction factor from old weights
        old_weights = await self.predictor.data_manager.get_learned_weights()
        current_correction_factor = getattr(old_weights, 'correction_factor', 1.0)
        current_correction_factor = max(
            CORRECTION_FACTOR_MIN,
            min(CORRECTION_FACTOR_MAX, current_correction_factor)
        )

        return LearnedWeights(
            weights=mapped_weights,
            bias=bias,
            accuracy=accuracy,
            training_samples=samples_used,
            last_trained=training_start_time.isoformat(),
            model_version=ML_MODEL_VERSION,
            feature_names=feature_engineer_v2.feature_names,  # V2 feature names (44)
            feature_means=self.predictor.scaler.means,  # Fitted with V2 features
            feature_stds=self.predictor.scaler.stds,  # Fitted with V2 features
            correction_factor=current_correction_factor
        )

    async def finalize_training(
        self,
        new_learned_weights: Any,
        accuracy: float,
        samples_used: int,
        training_records: List[Dict],
        training_start_time: datetime
    ) -> None:
        """Finalize training - save weights, update state"""
        from ..const import MODEL_ACCURACY_THRESHOLD

        # Save learned weights
        await self.predictor.data_manager.save_learned_weights(new_learned_weights)
        _LOGGER.info("Learned weights saved to disk")

        # Update predictor state
        self.predictor.learned_weights = new_learned_weights
        self.predictor.model_loaded = True
        self.predictor.current_accuracy = accuracy  # CRITICAL: Set accuracy for model_state.json
        self.predictor.training_samples = samples_used  # CRITICAL: Set samples for model_state.json
        self.predictor.model_state = (
            ModelState.READY if accuracy >= MODEL_ACCURACY_THRESHOLD
            else ModelState.DEGRADED
        )
        self.predictor.last_training = training_start_time

        await self.predictor._update_model_state_file()

    async def send_training_notifications(
        self,
        success: bool,
        accuracy: float = None,
        samples_used: int = 0
    ) -> None:
        """Send training completion notifications"""
        if not self.predictor.notification_service:
            return

        try:
            if success and accuracy:
                await self.predictor.notification_service.send_ml_training_notification(
                    accuracy=accuracy,
                    samples_used=samples_used
                )
        except Exception as e:
            _LOGGER.debug(f"Failed to send training notification: {e}")


# Import ModelState from parent module
try:
    from ..ml.ml_predictor import ModelState
except ImportError:
    # Fallback definition
    from enum import Enum

    class ModelState(Enum):
        UNINITIALIZED = "uninitialized"
        TRAINING = "training"
        READY = "ready"
        DEGRADED = "degraded"
        ERROR = "error"
