# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from ..const import CORRECTION_FACTOR_MAX, CORRECTION_FACTOR_MIN, DATA_VERSION, ML_MODEL_VERSION

from ..core.core_helpers import SafeDateTimeUtil as dt_util

from ..ai import (
    HourlyProfile,
    LearnedWeights,
    PredictionRecord,
    create_default_hourly_profile,
    create_default_learned_weights,
)

_LOGGER = logging.getLogger(__name__)

class TypedDataAdapter:
    """Adapter class responsible for converting data between unstructured dictionaries"""

    @staticmethod
    def dict_to_prediction_record(data: Dict[str, Any]) -> PredictionRecord:
        """Converts a dictionary to a PredictionRecord dataclass instance @zara"""
        if isinstance(data, PredictionRecord):
            return data

        try:

            timestamp = data.get("timestamp", dt_util.now().isoformat())
            predicted = float(data.get("predicted_value", 0.0))

            actual_raw = data.get("actual_value")
            actual = float(actual_raw) if actual_raw is not None else None
            weather = data.get("weather_data", {})
            sensor = data.get("sensor_data", {})
            accuracy = float(data.get("accuracy", 0.0))
            version = data.get("model_version", ML_MODEL_VERSION)

            return PredictionRecord(
                timestamp=timestamp,
                predicted_value=predicted,
                actual_value=actual,
                weather_data=weather,
                sensor_data=sensor,
                accuracy=accuracy,
                model_version=version,
            )

        except (ValueError, TypeError, KeyError) as e:
            _LOGGER.error("Failed to convert dictionary to PredictionRecord: %s. Data: %s", e, data)

            raise ValueError(f"Invalid data for PredictionRecord conversion: {e}") from e

    @staticmethod
    def dict_to_learned_weights(data: Dict[str, Any]) -> LearnedWeights:
        """Converts a dictionary to a LearnedWeights dataclass instance @zara"""
        if isinstance(data, LearnedWeights):
            return data

        try:

            weights = data.get("weights")
            if weights is None:

                weights = data.get("weather_weights", {})
                if weights:
                    _LOGGER.debug("Using 'weather_weights' as fallback for 'weights' field.")

            bias = float(data.get("bias", 0.0))

            default_feature_names = [
                "temperature",
                "humidity",
                "cloudiness",
                "wind_speed",
                "hour_of_day",
                "seasonal_factor",
                "weather_trend",
                "production_yesterday",
                "production_last_hour",
            ]
            feature_names = data.get("feature_names", default_feature_names)
            if not isinstance(feature_names, list) or not feature_names:
                _LOGGER.warning(
                    "Invalid or missing 'feature_names' in weights data, using default list."
                )
                feature_names = default_feature_names

            feature_means = data.get("feature_means", {})
            feature_stds = data.get("feature_stds", {})

            accuracy = float(data.get("accuracy", 0.0))
            training_samples = int(data.get("training_samples", 0))
            last_trained = data.get(
                "last_trained", dt_util.now().isoformat()
            )
            model_version = data.get("model_version", ML_MODEL_VERSION)

            correction_factor_raw = data.get("correction_factor", 1.0)
            try:
                correction_factor = float(correction_factor_raw)

                correction_factor = max(
                    CORRECTION_FACTOR_MIN, min(CORRECTION_FACTOR_MAX, correction_factor)
                )
            except (ValueError, TypeError):
                _LOGGER.warning(
                    f"Invalid correction_factor '{correction_factor_raw}', using default 1.0."
                )
                correction_factor = 1.0

            learned_weights_instance = LearnedWeights(
                weights=weights if isinstance(weights, dict) else {},
                bias=bias,
                feature_names=feature_names,
                feature_means=(
                    feature_means if isinstance(feature_means, dict) else {}
                ),
                feature_stds=feature_stds if isinstance(feature_stds, dict) else {},
                accuracy=accuracy,
                training_samples=training_samples,
                last_trained=last_trained,
                model_version=model_version,
                algorithm_used=data.get("algorithm_used", "ridge"),
                correction_factor=correction_factor,

                weather_weights=data.get("weather_weights", {}),
                seasonal_factors=data.get("seasonal_factors", {}),
                feature_importance=data.get("feature_importance", {}),
            )
            _LOGGER.debug("Successfully converted dictionary to LearnedWeights object.")
            return learned_weights_instance

        except Exception as e:

            _LOGGER.error(
                "Failed to convert dictionary to LearnedWeights: %s. Returning default weights.",
                e,
                exc_info=True,
            )

            return create_default_learned_weights()

    @staticmethod
    def learned_weights_to_dict(weights: LearnedWeights) -> Dict[str, Any]:
        """Converts a LearnedWeights dataclass instance back into a dictionary @zara"""
        if not isinstance(weights, LearnedWeights):
            _LOGGER.error(
                "Invalid input: learned_weights_to_dict expects a LearnedWeights instance."
            )

            return TypedDataAdapter.learned_weights_to_dict(create_default_learned_weights())

        return {

            "weights": weights.weights,
            "bias": weights.bias,
            "feature_names": weights.feature_names,
            "feature_means": weights.feature_means,
            "feature_stds": weights.feature_stds,

            "accuracy": weights.accuracy,
            "training_samples": weights.training_samples,
            "last_trained": weights.last_trained,
            "model_version": weights.model_version,
            "algorithm_used": weights.algorithm_used,
            "correction_factor": weights.correction_factor,

            "weather_weights": weights.weather_weights,
            "seasonal_factors": weights.seasonal_factors,
            "feature_importance": weights.feature_importance,

            "file_format_version": DATA_VERSION,
            "last_saved": dt_util.now().isoformat(),
        }

    @staticmethod
    def dict_to_hourly_profile(data: Dict[str, Any]) -> HourlyProfile:
        """Converts a dictionary to an HourlyProfile dataclass instance @zara"""
        if isinstance(data, HourlyProfile):
            return data

        try:

            hourly_averages_raw = data.get("hourly_averages", {})

            # Convert hourly_averages, handling old dict format migration
            hourly_averages: Dict[str, float] = {}
            if isinstance(hourly_averages_raw, dict):
                for k, v in hourly_averages_raw.items():
                    if isinstance(v, dict):
                        # Old format: {"count": 0, "total": 0.0, "average": 0.5}
                        hourly_averages[str(k)] = float(v.get("average", 0.0))
                    elif isinstance(v, (int, float)):
                        hourly_averages[str(k)] = float(v)
                    else:
                        hourly_averages[str(k)] = 0.0

            samples_count = int(data.get("samples_count", 0))
            last_updated = data.get(
                "last_updated", dt_util.now().isoformat()
            )
            confidence = float(data.get("confidence", 0.1))

            hourly_factors = data.get("hourly_factors", {})
            seasonal_adjustment = data.get("seasonal_adjustment", {})

            hourly_profile_instance = HourlyProfile(
                hourly_averages=hourly_averages,
                samples_count=samples_count,
                last_updated=last_updated,
                confidence=confidence,

                hourly_factors=hourly_factors if isinstance(hourly_factors, dict) else {},
                seasonal_adjustment=(
                    seasonal_adjustment if isinstance(seasonal_adjustment, dict) else {}
                ),
            )
            _LOGGER.debug("Successfully converted dictionary to HourlyProfile object.")
            return hourly_profile_instance

        except Exception as e:
            _LOGGER.error(
                "Failed to convert dictionary to HourlyProfile: %s. Returning default profile.",
                e,
                exc_info=True,
            )

            return create_default_hourly_profile()

    @staticmethod
    def hourly_profile_to_dict(profile: HourlyProfile) -> Dict[str, Any]:
        """Converts an HourlyProfile dataclass instance back into a dictionary @zara"""
        if not isinstance(profile, HourlyProfile):
            _LOGGER.error(
                "Invalid input: hourly_profile_to_dict expects an HourlyProfile instance."
            )
            return TypedDataAdapter.hourly_profile_to_dict(create_default_hourly_profile())

        return {

            "hourly_averages": profile.hourly_averages,

            "samples_count": profile.samples_count,
            "last_updated": profile.last_updated,
            "confidence": profile.confidence,

            "hourly_factors": profile.hourly_factors,
            "seasonal_adjustment": profile.seasonal_adjustment,

            "file_format_version": DATA_VERSION,
            "last_saved": dt_util.now().isoformat(),
        }

    @staticmethod
    def prediction_record_to_dict(record: PredictionRecord) -> Dict[str, Any]:
        """Converts a PredictionRecord dataclass instance into a dictionary @zara"""
        if not isinstance(record, PredictionRecord):
            _LOGGER.error(
                "Invalid input: prediction_record_to_dict expects a PredictionRecord instance."
            )

            return {}

        return {
            "timestamp": record.timestamp,
            "predicted_value": record.predicted_value,
            "actual_value": record.actual_value,
            "weather_data": record.weather_data,
            "sensor_data": record.sensor_data,
            "accuracy": record.accuracy,
            "model_version": record.model_version,

        }
