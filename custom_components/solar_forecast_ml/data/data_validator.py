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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..const import DATA_VERSION, MIN_TRAINING_DATA_POINTS
from .data_schemas import get_schema

_LOGGER = logging.getLogger(__name__)

# wttr.in cache settings
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

    @staticmethod
    def validate_learning_config(data: Dict[str, Any]) -> bool:
        """Validate learning_config.json structure @zara

        Args:
            data: Config data dict

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, dict):
            _LOGGER.error("learning_config: data is not a dict")
            return False

        # Check required top-level fields
        required_sections = ["learning_parameters", "weather_adjustment",
                            "elevation_adjustment", "physics_defaults"]

        for section in required_sections:
            if section not in data:
                _LOGGER.warning(f"learning_config: missing section '{section}'")

        # Validate learning_parameters
        lp = data.get("learning_parameters", {})
        if not isinstance(lp, dict):
            _LOGGER.error("learning_config: learning_parameters is not a dict")
            return False

        # Check key learning parameters
        if "smoothing" in lp:
            smoothing = lp["smoothing"]
            if not isinstance(smoothing, dict):
                _LOGGER.error("learning_config: smoothing is not a dict")
                return False

        if "efficiency_clamps" in lp:
            clamps = lp["efficiency_clamps"]
            if not isinstance(clamps, dict):
                _LOGGER.error("learning_config: efficiency_clamps is not a dict")
                return False

        # Validate physics_defaults
        physics = data.get("physics_defaults", {})
        if physics:
            albedo = physics.get("albedo")
            if albedo is not None and not (0.0 <= albedo <= 1.0):
                _LOGGER.warning(f"learning_config: albedo out of range [0,1]: {albedo}")

            sys_eff = physics.get("system_efficiency")
            if sys_eff is not None and not (0.0 <= sys_eff <= 1.0):
                _LOGGER.warning(f"learning_config: system_efficiency out of range [0,1]: {sys_eff}")

        return True

    @staticmethod
    def validate_and_create_learning_config(
        config_file: Path,
    ) -> Dict[str, Any]:
        """Validate learning_config.json, create if missing or invalid @zara

        Args:
            config_file: Path to the config file

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
            config_file.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists
            if not config_file.exists():
                # Create default config file
                default_config = DataValidator._create_default_learning_config()
                DataValidator._write_cache_file(config_file, default_config)
                result["created"] = True
                result["valid"] = True
                result["message"] = "Created new learning_config.json file"
                _LOGGER.info(f"Created new learning_config.json: {config_file}")
                return result

            # File exists - validate it
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                _LOGGER.warning(f"learning_config.json corrupted, recreating: {e}")
                default_config = DataValidator._create_default_learning_config()
                DataValidator._write_cache_file(config_file, default_config)
                result["repaired"] = True
                result["valid"] = True
                result["message"] = "Repaired corrupted learning_config.json file"
                return result

            # Validate structure
            if DataValidator.validate_learning_config(data):
                result["valid"] = True
                result["message"] = "learning_config.json is valid"
                return result

            # Invalid structure - repair it
            _LOGGER.warning("learning_config.json has invalid structure, recreating")
            default_config = DataValidator._create_default_learning_config()
            DataValidator._write_cache_file(config_file, default_config)
            result["repaired"] = True
            result["valid"] = True
            result["message"] = "Repaired invalid learning_config.json structure"
            return result

        except Exception as e:
            result["message"] = f"Error validating learning_config.json: {e}"
            _LOGGER.error(result["message"])
            return result

    @staticmethod
    def _create_default_learning_config() -> Dict[str, Any]:
        """Create default learning_config.json structure @zara

        Uses centralized schema from data_schemas.py (Single Source of Truth).
        """
        return get_schema("learning_config")

    @staticmethod
    def validate_learned_weights(weights: Dict[str, Any]) -> Dict[str, Any]:
        """Validate learned_weights.json structure with attention support @zara

        Args:
            weights: Weights data dict from learned_weights.json

        Returns:
            Dict with validation result:
            {
                "valid": bool,
                "has_attention": bool,
                "issues": List[str],
                "model_info": Dict[str, Any]
            }
        """
        result = {
            "valid": True,
            "has_attention": False,
            "issues": [],
            "model_info": {},
        }

        if not isinstance(weights, dict):
            result["valid"] = False
            result["issues"].append("Weights data is not a dictionary")
            return result

        # Required LSTM gate weights
        required_lstm_fields = ["Wf", "Wi", "Wc", "Wo", "bf", "bi", "bc", "bo"]
        # Required output layer weights
        required_output_fields = ["Wy", "by"]
        # Required architecture info
        required_info_fields = ["input_size", "hidden_size"]

        # Check required LSTM fields
        for field in required_lstm_fields:
            if field not in weights:
                result["valid"] = False
                result["issues"].append(f"Missing required LSTM field: {field}")

        # Check output layer fields
        for field in required_output_fields:
            if field not in weights:
                result["valid"] = False
                result["issues"].append(f"Missing required output field: {field}")

        # Check architecture info
        for field in required_info_fields:
            if field not in weights:
                result["issues"].append(f"Missing architecture info: {field}")

        # Extract model info
        result["model_info"] = {
            "input_size": weights.get("input_size"),
            "hidden_size": weights.get("hidden_size"),
            "sequence_length": weights.get("sequence_length", 24),
            "num_outputs": weights.get("num_outputs", 1),
            "training_samples": weights.get("training_samples", 0),
            "accuracy": weights.get("accuracy"),
            "rmse": weights.get("rmse"),
            "last_trained": weights.get("last_trained"),
        }

        # Check attention mechanism
        has_attention = weights.get("has_attention", False)
        result["has_attention"] = has_attention
        result["model_info"]["has_attention"] = has_attention

        if has_attention:
            # Attention-specific fields required when has_attention=True
            attention_fields = ["W_query", "W_key", "W_value", "W_attn_out", "b_attn_out"]

            for field in attention_fields:
                if field not in weights:
                    result["valid"] = False
                    result["issues"].append(f"Missing attention field: {field}")

            # Validate attention weight shapes if present
            hidden_size = weights.get("hidden_size", 32)
            if "W_query" in weights:
                w_query = weights["W_query"]
                if isinstance(w_query, list):
                    expected_shape = (hidden_size, hidden_size)
                    actual_shape = (len(w_query), len(w_query[0]) if w_query else 0)
                    if actual_shape != expected_shape:
                        result["issues"].append(
                            f"W_query shape mismatch: expected {expected_shape}, got {actual_shape}"
                        )

            if "W_attn_out" in weights:
                w_attn_out = weights["W_attn_out"]
                if isinstance(w_attn_out, list):
                    expected_shape = (hidden_size, hidden_size * 2)
                    actual_shape = (len(w_attn_out), len(w_attn_out[0]) if w_attn_out else 0)
                    if actual_shape != expected_shape:
                        result["issues"].append(
                            f"W_attn_out shape mismatch: expected {expected_shape}, got {actual_shape}"
                        )

        # Validate weight arrays are not empty
        for field in required_lstm_fields + required_output_fields:
            if field in weights:
                value = weights[field]
                if isinstance(value, list) and len(value) == 0:
                    result["valid"] = False
                    result["issues"].append(f"Empty weight array: {field}")

        # Validate accuracy if present
        accuracy = weights.get("accuracy")
        if accuracy is not None:
            if not isinstance(accuracy, (int, float)):
                result["issues"].append(f"Invalid accuracy type: {type(accuracy)}")
            elif not -1.0 <= accuracy <= 1.0:
                result["issues"].append(f"Accuracy out of range [-1, 1]: {accuracy}")

        return result

    @staticmethod
    def validate_and_repair_learned_weights(
        weights_file: Path,
    ) -> Dict[str, Any]:
        """Validate learned_weights.json, return validation status @zara

        Note: This method does NOT auto-repair weights as that would
        require retraining. It only validates and reports issues.

        Args:
            weights_file: Path to learned_weights.json

        Returns:
            Dict with validation result including model info
        """
        result = {
            "exists": False,
            "valid": False,
            "has_attention": False,
            "issues": [],
            "model_info": {},
            "message": "",
        }

        try:
            if not weights_file.exists():
                result["message"] = "No learned_weights.json found (model not trained)"
                return result

            result["exists"] = True

            # Load and parse
            try:
                with open(weights_file, "r", encoding="utf-8") as f:
                    weights = json.load(f)
            except json.JSONDecodeError as e:
                result["message"] = f"Corrupted weights file: {e}"
                result["issues"].append("JSON parse error - retraining required")
                return result

            # Validate structure
            validation = DataValidator.validate_learned_weights(weights)
            result["valid"] = validation["valid"]
            result["has_attention"] = validation["has_attention"]
            result["issues"] = validation["issues"]
            result["model_info"] = validation["model_info"]

            if validation["valid"]:
                attn_str = " (with attention)" if validation["has_attention"] else ""
                result["message"] = f"Weights valid{attn_str}"
            else:
                result["message"] = f"Invalid weights: {', '.join(validation['issues'][:3])}"

            return result

        except Exception as e:
            result["message"] = f"Error validating weights: {e}"
            result["issues"].append(str(e))
            return result
