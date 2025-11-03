"""
Typed Data Adapter for Solar Forecast ML Integration.
Handles conversion between dictionaries and dataclasses defined in ml_types.py.

Copyright (C) 2025 Zara-Toorox

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
from typing import Dict, Any, Optional

# Import the dataclasses and default creators
from ..ml.ml_types import (
    PredictionRecord, LearnedWeights, HourlyProfile,
    create_default_learned_weights, create_default_hourly_profile
)
# Import constants needed for defaults or validation
from ..const import DATA_VERSION, ML_MODEL_VERSION, CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX
# Import SafeDateTimeUtil for timestamps
from ..core.helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class TypedDataAdapter:
    """
    Adapter class responsible for converting data between unstructured dictionaries
    (typically from JSON) and the strongly-typed dataclasses used internally.
    Provides robustness against missing keys and handles defaults.
    """

    @staticmethod
    def dict_to_prediction_record(data: Dict[str, Any]) -> PredictionRecord:
        """
        Converts a dictionary to a PredictionRecord dataclass instance.

        Args:
            data: The input dictionary.

        Returns:
            A PredictionRecord instance.

        Raises:
            ValueError: If conversion fails due to invalid data types or missing required keys
                       (after attempting defaults). Catches specific errors and re-raises.
        """
        if isinstance(data, PredictionRecord):
            return data # Return if already the correct type

        try:
            # Extract values with defaults for optional/potentially missing fields
            timestamp = data.get("timestamp", dt_util.utcnow().isoformat())
            predicted = float(data.get("predicted_value", 0.0))
            # Handle actual_value which can be None
            actual_raw = data.get("actual_value")
            actual = float(actual_raw) if actual_raw is not None else None
            weather = data.get("weather_data", {})
            sensor = data.get("sensor_data", {})
            accuracy = float(data.get("accuracy", 0.0)) # Default accuracy if missing
            version = data.get("model_version", ML_MODEL_VERSION) # Default version

            # Create the dataclass instance (validation happens in __post_init__)
            return PredictionRecord(
                timestamp=timestamp,
                predicted_value=predicted,
                actual_value=actual,
                weather_data=weather,
                sensor_data=sensor,
                accuracy=accuracy,
                model_version=version
            )

        except (ValueError, TypeError, KeyError) as e:
            _LOGGER.error("Failed to convert dictionary to PredictionRecord: %s. Data: %s", e, data)
            # Re-raise as ValueError to indicate conversion failure
            raise ValueError(f"Invalid data for PredictionRecord conversion: {e}") from e

    @staticmethod
    def dict_to_learned_weights(data: Dict[str, Any]) -> LearnedWeights:
        """
        Converts a dictionary to a LearnedWeights dataclass instance.
        Handles missing keys gracefully for backward compatibility with older formats.
        Applies defaults and clamps values where necessary.

        Args:
            data: The input dictionary (likely loaded from learned_weights.json).

        Returns:
            A LearnedWeights instance. Returns default weights if conversion fails significantly.
        """
        if isinstance(data, LearnedWeights):
            return data # Return if already the correct type

        try:
            # --- Provide defaults for potentially missing fields ---
            # Handle 'weights' vs older 'weather_weights'
            weights = data.get("weights")
            if weights is None:
                 # Fallback: Try to use 'weather_weights' if 'weights' is missing
                 weights = data.get("weather_weights", {})
                 if weights:
                      _LOGGER.debug("Using 'weather_weights' as fallback for 'weights' field.")
                 # If both are missing, it will default to {} below.


            bias = float(data.get("bias", 0.0))

            # Feature names: Critical for mapping weights. Default cautiously.
            default_feature_names = [ # Match FeatureEngineer base features
                "temperature", "humidity", "cloudiness", "wind_speed",
                "hour_of_day", "seasonal_factor", "weather_trend",
                "production_yesterday", "production_last_hour"
            ]
            feature_names = data.get("feature_names", default_feature_names)
            if not isinstance(feature_names, list) or not feature_names:
                 _LOGGER.warning("Invalid or missing 'feature_names' in weights data, using default list.")
                 feature_names = default_feature_names

            # Scaler state (means and stds) - default to empty dicts if missing
            feature_means = data.get("feature_means", {})
            feature_stds = data.get("feature_stds", {})

            # Metadata with defaults
            accuracy = float(data.get("accuracy", 0.0)) # Default to 0.0 if never trained
            training_samples = int(data.get("training_samples", 0))
            last_trained = data.get("last_trained", dt_util.utcnow().isoformat()) # Default to now if missing
            model_version = data.get("model_version", ML_MODEL_VERSION)

            # Fallback correction factor
            correction_factor_raw = data.get("correction_factor", 1.0)
            try:
                 correction_factor = float(correction_factor_raw)
                 # Clamp to defined range
                 correction_factor = max(CORRECTION_FACTOR_MIN, min(CORRECTION_FACTOR_MAX, correction_factor))
            except (ValueError, TypeError):
                 _LOGGER.warning(f"Invalid correction_factor '{correction_factor_raw}', using default 1.0.")
                 correction_factor = 1.0


            # --- Create Dataclass Instance ---
            # Validation (clamping, range checks) happens in __post_init__
            learned_weights_instance = LearnedWeights(
                weights=weights if isinstance(weights, dict) else {}, # Ensure dict
                bias=bias,
                feature_names=feature_names,
                feature_means=feature_means if isinstance(feature_means, dict) else {}, # Ensure dict
                feature_stds=feature_stds if isinstance(feature_stds, dict) else {}, # Ensure dict
                accuracy=accuracy,
                training_samples=training_samples,
                last_trained=last_trained,
                model_version=model_version,
                correction_factor=correction_factor,
                # Deprecated fields (load if present, otherwise default factory)
                weather_weights=data.get("weather_weights", {}),
                seasonal_factors=data.get("seasonal_factors", {}),
                feature_importance=data.get("feature_importance", {})
            )
            _LOGGER.debug("Successfully converted dictionary to LearnedWeights object.")
            return learned_weights_instance

        except Exception as e:
            # Catch broader errors during conversion (e.g., unexpected data types)
            _LOGGER.error("Failed to convert dictionary to LearnedWeights: %s. Returning default weights.", e, exc_info=True)
            # Return a default instance on failure to allow integration to continue
            return create_default_learned_weights()


    @staticmethod
    def learned_weights_to_dict(weights: LearnedWeights) -> Dict[str, Any]:
        """
        Converts a LearnedWeights dataclass instance back into a dictionary
        suitable for JSON serialization. Includes all current fields.

        Args:
            weights: The LearnedWeights instance.

        Returns:
            A dictionary representation.
        """
        if not isinstance(weights, LearnedWeights):
            _LOGGER.error("Invalid input: learned_weights_to_dict expects a LearnedWeights instance.")
            # Return dict from default? Or raise error? Return default dict for now.
            return TypedDataAdapter.learned_weights_to_dict(create_default_learned_weights())

        # Directly access attributes from the dataclass instance
        return {
            # Core fields
            "weights": weights.weights,
            "bias": weights.bias,
            "feature_names": weights.feature_names,
            "feature_means": weights.feature_means,
            "feature_stds": weights.feature_stds,
            # Metadata
            "accuracy": weights.accuracy,
            "training_samples": weights.training_samples,
            "last_trained": weights.last_trained,
            "model_version": weights.model_version,
            "correction_factor": weights.correction_factor,
            # Deprecated/Auxiliary fields
            "weather_weights": weights.weather_weights,
            "seasonal_factors": weights.seasonal_factors,
            "feature_importance": weights.feature_importance,
            # Add file format version and update timestamp for traceability
            "file_format_version": DATA_VERSION, # Use const for file format version
            "last_saved": dt_util.utcnow().isoformat()
        }


    @staticmethod
    def dict_to_hourly_profile(data: Dict[str, Any]) -> HourlyProfile:
        """
        Converts a dictionary to an HourlyProfile dataclass instance.
        Handles missing keys and ensures the structure is valid.

        Args:
            data: The input dictionary (likely loaded from hourly_profile.json).

        Returns:
            An HourlyProfile instance. Returns default profile if conversion fails.
        """
        if isinstance(data, HourlyProfile):
            return data # Return if already the correct type

        try:
            # --- Extract data with defaults ---
            # Primary data: hourly averages (keys '0' to '23')
            hourly_averages_raw = data.get("hourly_averages", {})
            # Ensure it's a dict, convert keys to str if needed (though post_init handles missing hours)
            hourly_averages = {str(k): float(v) for k, v in hourly_averages_raw.items()} if isinstance(hourly_averages_raw, dict) else {}

            # Metadata
            samples_count = int(data.get("samples_count", 0))
            last_updated = data.get("last_updated", dt_util.utcnow().isoformat())
            confidence = float(data.get("confidence", 0.1)) # Default to low confidence

            # Deprecated fields (load if present, default factory handles if absent)
            hourly_factors = data.get("hourly_factors", {})
            seasonal_adjustment = data.get("seasonal_adjustment", {})


            # --- Create Dataclass Instance ---
            # Validation and default hour filling happens in __post_init__
            hourly_profile_instance = HourlyProfile(
                hourly_averages=hourly_averages,
                samples_count=samples_count,
                last_updated=last_updated,
                confidence=confidence,
                # Pass deprecated fields if loaded
                hourly_factors=hourly_factors if isinstance(hourly_factors, dict) else {},
                seasonal_adjustment=seasonal_adjustment if isinstance(seasonal_adjustment, dict) else {}
            )
            _LOGGER.debug("Successfully converted dictionary to HourlyProfile object.")
            return hourly_profile_instance

        except Exception as e:
            _LOGGER.error("Failed to convert dictionary to HourlyProfile: %s. Returning default profile.", e, exc_info=True)
            # Return a default instance on failure
            return create_default_hourly_profile()


    @staticmethod
    def hourly_profile_to_dict(profile: HourlyProfile) -> Dict[str, Any]:
        """
        Converts an HourlyProfile dataclass instance back into a dictionary
        suitable for JSON serialization.

        Args:
            profile: The HourlyProfile instance.

        Returns:
            A dictionary representation.
        """
        if not isinstance(profile, HourlyProfile):
             _LOGGER.error("Invalid input: hourly_profile_to_dict expects an HourlyProfile instance.")
             return TypedDataAdapter.hourly_profile_to_dict(create_default_hourly_profile())

        return {
            # Core fields
            "hourly_averages": profile.hourly_averages, # Keys are already strings '0'-'23'
            # Metadata
            "samples_count": profile.samples_count,
            "last_updated": profile.last_updated,
            "confidence": profile.confidence,
            # Deprecated fields
            "hourly_factors": profile.hourly_factors,
            "seasonal_adjustment": profile.seasonal_adjustment,
            # File format info
            "file_format_version": DATA_VERSION,
            "last_saved": dt_util.utcnow().isoformat()
        }


    @staticmethod
    def prediction_record_to_dict(record: PredictionRecord) -> Dict[str, Any]:
        """
        Converts a PredictionRecord dataclass instance into a dictionary
        suitable for JSON serialization.

        Args:
            record: The PredictionRecord instance.

        Returns:
            A dictionary representation.
        """
        if not isinstance(record, PredictionRecord):
             _LOGGER.error("Invalid input: prediction_record_to_dict expects a PredictionRecord instance.")
             # Cannot easily create a default record, return empty dict or raise? Empty dict safer.
             return {}

        return {
            "timestamp": record.timestamp,
            "predicted_value": record.predicted_value,
            "actual_value": record.actual_value, # Will be null if Optional[float] is None
            "weather_data": record.weather_data,
            "sensor_data": record.sensor_data,
            "accuracy": record.accuracy,
            "model_version": record.model_version
            # No added file format version here as it's part of a list usually
        }