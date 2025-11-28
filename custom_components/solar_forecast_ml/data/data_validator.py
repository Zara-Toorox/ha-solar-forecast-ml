"""Data Validation Module for Solar Forecast ML Integration V10.0.0 @zara

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
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..const import DATA_VERSION, MIN_TRAINING_DATA_POINTS

_LOGGER = logging.getLogger(__name__)

class DataValidator:
    """Validates data integrity for ML and forecast data"""

    @staticmethod
    def validate_prediction_data(data: Dict[str, Any]) -> bool:
        """Validate prediction data structure @zara"""
        required_fields = ["timestamp", "prediction_kwh", "confidence"]

        for field in required_fields:
            if field not in data:
                _LOGGER.error(f"Missing required field in prediction data: {field}")
                return False

        if not isinstance(data.get("prediction_kwh"), (int, float)):
            _LOGGER.error("Invalid type for prediction_kwh")
            return False

        if not isinstance(data.get("confidence"), (int, float)):
            _LOGGER.error("Invalid type for confidence")
            return False

        if data["prediction_kwh"] < 0:
            _LOGGER.error("Negative prediction value")
            return False

        if not 0 <= data["confidence"] <= 1:
            _LOGGER.error("Confidence out of range [0, 1]")
            return False

        return True

    @staticmethod
    def validate_sample_data(sample: Dict[str, Any]) -> bool:
        """Validate ML sample data structure @zara"""
        required_fields = ["timestamp", "actual_power", "features"]

        for field in required_fields:
            if field not in sample:
                _LOGGER.error(f"Missing required field in sample: {field}")
                return False

        features = sample.get("features", {})
        if not isinstance(features, dict):
            _LOGGER.error("Features must be a dictionary")
            return False

        min_features = ["hour", "temperature", "cloud_cover"]
        for feature in min_features:
            if feature not in features:
                _LOGGER.warning(f"Missing recommended feature: {feature}")

        return True

    @staticmethod
    def validate_model_state(state: Dict[str, Any]) -> bool:
        """Validate model state data @zara"""
        required_fields = ["version", "model_loaded", "training_samples"]

        for field in required_fields:
            if field not in state:
                _LOGGER.error(f"Missing required field in model state: {field}")
                return False

        if state.get("version") != DATA_VERSION:
            _LOGGER.warning(
                f"Model state version mismatch: {state.get('version')} != {DATA_VERSION}"
            )

        training_samples = state.get("training_samples", 0)
        if training_samples < 0:
            _LOGGER.error("Negative training sample count")
            return False

        return True

    @staticmethod
    def validate_daily_forecast(forecast: Dict[str, Any]) -> bool:
        """Validate daily forecast data @zara"""
        required_fields = ["date", "prediction_kwh"]

        for field in required_fields:
            if field not in forecast:
                _LOGGER.error(f"Missing required field in forecast: {field}")
                return False

        try:
            if forecast.get("date"):
                datetime.fromisoformat(forecast["date"])
        except (ValueError, TypeError):
            _LOGGER.error(f"Invalid date format: {forecast.get('date')}")
            return False

        prediction = forecast.get("prediction_kwh")
        if prediction is not None and (not isinstance(prediction, (int, float)) or prediction < 0):
            _LOGGER.error(f"Invalid prediction value: {prediction}")
            return False

        return True

    @staticmethod
    def check_data_quality(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check overall data quality metrics @zara"""
        if not samples:
            return {
                "total_samples": 0,
                "valid_samples": 0,
                "quality_score": 0.0,
                "issues": ["No samples available"],
            }

        valid_count = sum(1 for s in samples if DataValidator.validate_sample_data(s))
        quality_score = valid_count / len(samples) if samples else 0.0

        issues = []
        if len(samples) < MIN_TRAINING_DATA_POINTS:
            issues.append(f"Insufficient samples: {len(samples)} < {MIN_TRAINING_DATA_POINTS}")

        if quality_score < 0.9:
            issues.append(f"Low quality score: {quality_score:.2%}")

        return {
            "total_samples": len(samples),
            "valid_samples": valid_count,
            "quality_score": quality_score,
            "sufficient_for_training": len(samples) >= MIN_TRAINING_DATA_POINTS
            and quality_score >= 0.8,
            "issues": issues,
        }
