"""
ML Training Helpers - Extract train_model logic

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
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone

from ..const import MIN_TRAINING_DATA_POINTS, CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX, ML_MODEL_VERSION
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from .ml_types import LearnedWeights

_LOGGER = logging.getLogger(__name__)


class MLTrainingHelpers:
    """Helper methods for ML training process"""

    def __init__(self, predictor: "MLPredictor"):
        """Initialize training helpers"""
        self.predictor = predictor

    async def validate_training_data(self, training_records: List[Dict]) -> Tuple[bool, str, int]:
        """Validate training data meets minimum requirements"""
        valid_records_count = len(training_records)
        absolute_min_samples = 20
        recommended_min_samples = MIN_TRAINING_DATA_POINTS

        if valid_records_count < absolute_min_samples:
            error_msg = (
                f"Insufficient training data: {valid_records_count} valid hourly samples found "
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

    async def prepare_training_features(
        self,
        training_records: List[Dict]
    ) -> Tuple[List[List[float]], List[float]]:
        """Extract features and labels from training records"""
        X_train_raw = []
        y_train = []

        for record in training_records:
            weather_data = record.get('weather_data', {})
            sensor_data = record.get('sensor_data', {})

            # Get actual value
            actual_kwh = record.get('actual_kwh')
            if actual_kwh is None:
                actual_kwh = record.get('actual_value')
            if actual_kwh is None:
                _LOGGER.debug(f"Skipping record without actual value: {record.get('timestamp')}")
                continue

            try:
                # Parse and normalize timestamp
                record_time_dt = dt_util.parse_datetime(record['timestamp'])
                if not record_time_dt:
                    _LOGGER.warning(f"Skipping training record, invalid timestamp: {record.get('timestamp')}")
                    continue

                if record_time_dt.tzinfo is None:
                    record_time_dt = record_time_dt.replace(tzinfo=timezone.utc)

                record_time_local = dt_util.as_local(record_time_dt)

                # Add lag features
                sensor_data = self._add_lag_features(sensor_data, record_time_local)

            except Exception as e:
                _LOGGER.debug(f"Could not find lag features for record {record['timestamp']}: {e}")
                sensor_data['production_yesterday'] = 0.0
                sensor_data['production_same_hour_yesterday'] = 0.0

            # Extract features
            features_dict = self.predictor.feature_engineer.extract_features_sync(
                weather_data, sensor_data, record
            )
            feature_vector = [
                features_dict.get(name, 0.0)
                for name in self.predictor.feature_engineer.feature_names
            ]
            X_train_raw.append(feature_vector)
            y_train.append(actual_kwh)

        _LOGGER.debug("Feature extraction complete")
        return X_train_raw, y_train

    def _add_lag_features(self, sensor_data: Dict, record_time_local: datetime) -> Dict:
        """Add lag features from historical cache"""
        yesterday_dt = record_time_local - timedelta(days=1)
        yesterday_key = yesterday_dt.date().isoformat()
        yesterday_total = self.predictor._historical_cache['daily_productions'].get(yesterday_key, 0.0)
        sensor_data['production_yesterday'] = float(yesterday_total)

        same_hour_yesterday_key = f"{yesterday_key}_{record_time_local.hour:02d}"
        same_hour_yesterday = self.predictor._historical_cache['hourly_productions'].get(
            same_hour_yesterday_key, 0.0
        )
        sensor_data['production_same_hour_yesterday'] = float(same_hour_yesterday)

        return sensor_data

    async def scale_and_train(
        self,
        X_train_raw: List[List[float]],
        y_train: List[float]
    ) -> Tuple[Dict, float, float, float]:
        """Scale features and train model"""
        X_train_scaled_list = await self.predictor.hass.async_add_executor_job(
            self.predictor.scaler.fit_transform,
            X_train_raw,
            self.predictor.feature_engineer.feature_names
        )
        _LOGGER.info(
            f"Scaler fitted and training data transformed ({len(self.predictor.feature_engineer.feature_names)} features)"
        )

        weights_dict_raw, bias, accuracy, best_lambda = await self.predictor.hass.async_add_executor_job(
            self.predictor.trainer.train, X_train_scaled_list, y_train
        )
        _LOGGER.info(
            f"Ridge training complete. Accuracy (R-squared): {accuracy:.4f}, Best Lambda: {best_lambda:.4f}"
        )

        return weights_dict_raw, bias, accuracy, best_lambda

    async def create_learned_weights(
        self,
        weights_dict_raw: Dict,
        bias: float,
        accuracy: float,
        valid_records_count: int,
        training_start_time: datetime
    ) -> LearnedWeights:
        """Create LearnedWeights object"""
        mapped_weights = self.predictor.trainer.map_weights_to_features(
            weights_dict_raw,
            self.predictor.feature_engineer.feature_names
        )

        # Preserve correction factor from old weights
        old_weights = await self.predictor.data_manager.get_learned_weights()
        current_correction_factor = getattr(old_weights, 'correction_factor', 1.0)
        current_correction_factor = max(
            CORRECTION_FACTOR_MIN,
            min(CORRECTION_FACTOR_MAX, current_correction_factor)
        )
        _LOGGER.debug(f"Preserving fallback correction factor: {current_correction_factor:.3f}")

        # Check model improvement
        old_accuracy = getattr(old_weights, 'accuracy', 0.0) if old_weights else 0.0
        accuracy_improvement = accuracy - old_accuracy

        if old_accuracy > 0.1 and accuracy_improvement < -0.10:
            _LOGGER.warning(
                f"New model accuracy ({accuracy:.1%}) is significantly worse than "
                f"previous model ({old_accuracy:.1%}). Consider manual rollback from backup."
            )

        new_learned_weights = LearnedWeights(
            weights=mapped_weights,
            bias=bias,
            accuracy=accuracy,
            training_samples=valid_records_count,
            last_trained=training_start_time.isoformat(),
            model_version=ML_MODEL_VERSION,
            feature_names=self.predictor.feature_engineer.feature_names,
            feature_means=self.predictor.scaler.means,
            feature_stds=self.predictor.scaler.stds,
            correction_factor=current_correction_factor
        )

        if accuracy_improvement > 0:
            _LOGGER.info(
                f"Model improved! Accuracy: {old_accuracy:.1%} → {accuracy:.1%} (+{accuracy_improvement:.1%})"
            )
        else:
            _LOGGER.info(
                f"Model updated. Accuracy: {old_accuracy:.1%} → {accuracy:.1%} ({accuracy_improvement:+.1%})"
            )

        return new_learned_weights

    async def finalize_training(
        self,
        new_weights: LearnedWeights,
        accuracy: float,
        valid_records_count: int,
        training_records: List[Dict],
        training_start_time: datetime
    ) -> None:
        """Save weights and update predictor state"""
        await self.predictor.data_manager.save_learned_weights(new_weights)
        await self.predictor._update_hourly_profile(training_records)

        self.predictor.current_weights = new_weights
        self.predictor.current_accuracy = accuracy
        self.predictor.training_samples = valid_records_count
        self.predictor.last_training_time = training_start_time
        self.predictor.model_loaded = True

        self.predictor.prediction_orchestrator.update_strategies(
            weights=self.predictor.current_weights,
            profile=self.predictor.current_profile,
            accuracy=self.predictor.current_accuracy,
            peak_power_kw=self.predictor.peak_power_kw
        )
        _LOGGER.debug("Prediction orchestrator updated")

    async def send_training_notifications(
        self,
        success: bool,
        accuracy: float = None,
        sample_count: int = 0
    ) -> None:
        """Send training notifications if service available"""
        if not self.predictor.notification_service:
            return

        try:
            if success:
                await self.predictor.notification_service.show_training_complete(
                    success=True,
                    accuracy=accuracy * 100,
                    sample_count=sample_count
                )
            else:
                await self.predictor.notification_service.show_training_complete(
                    success=False,
                    accuracy=0,
                    sample_count=sample_count
                )
        except Exception as notify_err:
            _LOGGER.debug(f"Training notification failed: {notify_err}")
