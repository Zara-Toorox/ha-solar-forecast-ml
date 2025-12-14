"""Data Validation Module for Solar Forecast ML Integration V12.0.0 @zara

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

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..const import DATA_VERSION, MIN_TRAINING_DATA_POINTS

_LOGGER = logging.getLogger(__name__)

# wttr.in cache settings (must match data_multi_weather_client.py)
WTTR_CACHE_MAX_AGE = 6 * 3600  # 6 hours

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

    @staticmethod
    def validate_wttr_cache(data: Dict[str, Any]) -> bool:
        """Validate wttr.in cache data structure @zara

        Args:
            data: Cache data dict

        Returns:
            True if valid, False otherwise
        """
        # Check required top-level fields
        if not isinstance(data, dict):
            _LOGGER.error("wttr.in cache: data is not a dict")
            return False

        if "version" not in data:
            _LOGGER.warning("wttr.in cache: missing version field")

        if "metadata" not in data:
            _LOGGER.error("wttr.in cache: missing metadata field")
            return False

        if "forecast" not in data:
            _LOGGER.error("wttr.in cache: missing forecast field")
            return False

        # Validate metadata
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            _LOGGER.error("wttr.in cache: metadata is not a dict")
            return False

        if "fetched_at" not in metadata:
            _LOGGER.error("wttr.in cache: missing fetched_at in metadata")
            return False

        # Validate fetched_at is valid ISO datetime (None is valid for new installation)
        fetched_at = metadata["fetched_at"]
        if fetched_at is not None:
            try:
                datetime.fromisoformat(fetched_at)
            except (ValueError, TypeError) as e:
                _LOGGER.error(f"wttr.in cache: invalid fetched_at format: {e}")
                return False
        else:
            _LOGGER.debug("wttr.in cache: fetched_at is null (normal for new installation)")

        # Validate forecast structure
        forecast = data.get("forecast", {})
        if not isinstance(forecast, dict):
            _LOGGER.error("wttr.in cache: forecast is not a dict")
            return False

        if not forecast:
            _LOGGER.debug("wttr.in cache: forecast is empty (normal for new installation)")
            return True  # Empty but valid structure

        # Validate at least one day entry
        for date_str, hours in forecast.items():
            # Validate date format
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                _LOGGER.warning(f"wttr.in cache: invalid date format: {date_str}")
                continue

            if not isinstance(hours, dict):
                _LOGGER.error(f"wttr.in cache: hours for {date_str} is not a dict")
                return False

            # Validate hour entries
            for hour_key, hour_data in hours.items():
                if not isinstance(hour_data, dict):
                    continue

                # Check for required fields in hour data
                if "cloud_cover" not in hour_data:
                    _LOGGER.warning(
                        f"wttr.in cache: missing cloud_cover for {date_str} hour {hour_key}"
                    )

        return True

    @staticmethod
    def validate_and_create_wttr_cache(
        cache_file: Path,
        latitude: float,
        longitude: float,
    ) -> Dict[str, Any]:
        """Validate wttr.in cache file, create if missing or invalid @zara

        Args:
            cache_file: Path to the cache file
            latitude: Location latitude
            longitude: Location longitude

        Returns:
            Dict with validation result:
            {
                "valid": bool,
                "created": bool,
                "repaired": bool,
                "message": str
            }
        """
        result = {
            "valid": False,
            "created": False,
            "repaired": False,
            "message": "",
        }

        try:
            # Ensure directory exists
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists
            if not cache_file.exists():
                # Create empty cache file
                empty_cache = DataValidator._create_empty_wttr_cache(latitude, longitude)
                DataValidator._write_cache_file(cache_file, empty_cache)
                result["created"] = True
                result["valid"] = True
                result["message"] = "Created new wttr.in cache file"
                _LOGGER.info(f"Created new wttr.in cache file: {cache_file}")
                return result

            # File exists - validate it
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                _LOGGER.warning(f"wttr.in cache corrupted, recreating: {e}")
                empty_cache = DataValidator._create_empty_wttr_cache(latitude, longitude)
                DataValidator._write_cache_file(cache_file, empty_cache)
                result["repaired"] = True
                result["valid"] = True
                result["message"] = "Repaired corrupted wttr.in cache file"
                return result

            # Validate structure
            if DataValidator.validate_wttr_cache(data):
                result["valid"] = True
                result["message"] = "wttr.in cache is valid"
                return result

            # Invalid structure - repair it
            _LOGGER.warning("wttr.in cache has invalid structure, recreating")
            empty_cache = DataValidator._create_empty_wttr_cache(latitude, longitude)
            DataValidator._write_cache_file(cache_file, empty_cache)
            result["repaired"] = True
            result["valid"] = True
            result["message"] = "Repaired invalid wttr.in cache structure"
            return result

        except Exception as e:
            result["message"] = f"Error validating wttr.in cache: {e}"
            _LOGGER.error(result["message"])
            return result

    @staticmethod
    def _create_empty_wttr_cache(latitude: float, longitude: float) -> Dict[str, Any]:
        """Create empty wttr.in cache structure."""
        return {
            "version": "1.0",
            "metadata": {
                "fetched_at": datetime.now().isoformat(),
                "source": "wttr.in",
                "latitude": latitude,
                "longitude": longitude,
                "cache_max_age_hours": WTTR_CACHE_MAX_AGE / 3600,
                "created_empty": True,
            },
            "forecast": {},
        }

    @staticmethod
    def _write_cache_file(cache_file: Path, data: Dict[str, Any]) -> None:
        """Write cache file atomically."""
        temp_file = cache_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(cache_file)
