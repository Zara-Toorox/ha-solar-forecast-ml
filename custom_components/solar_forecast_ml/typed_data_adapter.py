"""
Typed Data Adapter for Solar Forecast ML Integration.
PROGRESSIVE UPGRADE v5.1.0: Feature Normalization Support

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

from .ml_types import (
    PredictionRecord, LearnedWeights, HourlyProfile,
    create_default_learned_weights, create_default_hourly_profile
)

_LOGGER = logging.getLogger(__name__)


class TypedDataAdapter:
    """
    Adapter for converting between Dicts and Typed Dataclasses.
    PROGRESSIVE UPGRADE: Feature Normalization Support
    """
    
    @staticmethod
    def dict_to_prediction_record(data: Dict[str, Any]) -> PredictionRecord:
        """Converts Dict to PredictionRecord Dataclass."""
        try:
            if isinstance(data, PredictionRecord):
                return data
            
            return PredictionRecord(
                timestamp=data.get("timestamp", datetime.now().isoformat()),
                predicted_value=float(data.get("predicted_value", 0.0)),
                actual_value=float(data["actual_value"]) if data.get("actual_value") is not None else None,
                weather_data=data.get("weather_data", {}),
                sensor_data=data.get("sensor_data", {}),
                accuracy=float(data.get("accuracy", 0.0)),
                model_version=data.get("model_version", "1.0")
            )
            
        except Exception as e:
            _LOGGER.error("Failed to convert dict to PredictionRecord: %s", str(e))
            raise
    
    @staticmethod
    def dict_to_learned_weights(data: Dict[str, Any]) -> LearnedWeights:
        """
        Converts Dict to LearnedWeights Dataclass.
        Handles missing keys and provides defaults for backward compatibility.
        PROGRESSIVE UPGRADE: feature_means and feature_stds Support
        """
        try:
            if isinstance(data, LearnedWeights):
                return data
            
            # --- Provide defaults for older versions ---
            weather_weights = data.get("weather_weights", {})
            if not weather_weights:
                weather_weights = {
                    "temperature": 0.3,
                    "humidity": 0.1,
                    "cloudiness": 0.4,
                    "wind_speed": 0.1,
                    "pressure": 0.1
                }
            
            seasonal_factors = data.get("seasonal_factors", {})
            if not seasonal_factors:
                seasonal_factors = {
                    "spring": 1.0,
                    "summer": 1.0,
                    "autumn": 1.0,
                    "winter": 1.0
                }
            
            # Clamp correction_factor to a safe range
            correction_factor = float(data.get("correction_factor", 1.0))
            correction_factor = max(0.1, min(5.0, correction_factor))
            
            bias = float(data.get("bias", 0.0))
            
            # Fallback for old models before 'weights' existed
            weights = data.get("weights", weather_weights.copy())
            
            # Fallback for feature names
            feature_names = data.get("feature_names", [
                "temperature", "humidity", "cloudiness", 
                "wind_speed", "hour_of_day", "seasonal_factor"
            ])
            
            # Add new normalization fields
            feature_means = data.get("feature_means", {})
            feature_stds = data.get("feature_stds", {})
            
            # --- Create Dataclass ---
            return LearnedWeights(
                weather_weights=weather_weights,
                seasonal_factors=seasonal_factors,
                correction_factor=correction_factor,
                accuracy=float(data.get("accuracy", 0.5)),
                training_samples=int(data.get("training_samples", 0)),
                last_trained=data.get("last_trained", datetime.now().isoformat()),
                model_version=data.get("model_version", "1.0"),
                feature_importance=data.get("feature_importance", {}),
                bias=bias,
                weights=weights,
                feature_names=feature_names,
                feature_means=feature_means,
                feature_stds=feature_stds
            )
            
        except Exception as e:
            _LOGGER.error("Failed to convert dict to LearnedWeights: %s. Creating default.", str(e))
            return create_default_learned_weights()
    
    @staticmethod
    def learned_weights_to_dict(weights: LearnedWeights) -> Dict[str, Any]:
        """
        Converts LearnedWeights to Dict for JSON storage.
        PROGRESSIVE UPGRADE: feature_means and feature_stds included
        """
        # The dataclass always has feature_means/stds (even if empty dicts),
        # so we can add them directly without hasattr checks.
        return {
            "version": "1.0",
            "weather_weights": weights.weather_weights,
            "seasonal_factors": weights.seasonal_factors,
            "correction_factor": weights.correction_factor,
            "accuracy": weights.accuracy,
            "training_samples": weights.training_samples,
            "last_trained": weights.last_trained,
            "model_version": weights.model_version,
            "feature_importance": weights.feature_importance,
            "bias": weights.bias,
            "weights": weights.weights,
            "feature_names": weights.feature_names,
            "feature_means": weights.feature_means,
            "feature_stds": weights.feature_stds,
            "updated": datetime.now().isoformat()
        }
    
    @staticmethod
    def dict_to_hourly_profile(data: Dict[str, Any]) -> HourlyProfile:
        """Converts Dict to HourlyProfile Dataclass."""
        try:
            if isinstance(data, HourlyProfile):
                return data
            
            hourly_factors = data.get("hourly_factors", {})
            if len(hourly_factors) != 24:
                # Ensure a 24-hour structure if data is partial or invalid
                hourly_factors = {str(h): 1.0 for h in range(24)}
            
            # *** BUGFIX ***
            # Removed redundant logic for populating hourly_averages.
            # The HourlyProfile dataclass __post_init__ handles this.
            # We just pass the dictionary (even if empty).
            hourly_averages = data.get("hourly_averages", {})
            
            return HourlyProfile(
                hourly_factors=hourly_factors,
                samples_count=int(data.get("samples_count", 0)),
                last_updated=data.get("last_updated", datetime.now().isoformat()),
                confidence=float(data.get("confidence", 0.5)),
                seasonal_adjustment=data.get("seasonal_adjustment", {}),
                hourly_averages=hourly_averages
            )
            
        except Exception as e:
            _LOGGER.error("Failed to convert dict to HourlyProfile: %s. Creating default.", str(e))
            return create_default_hourly_profile()
    
    @staticmethod
    def hourly_profile_to_dict(profile: HourlyProfile) -> Dict[str, Any]:
        """Converts HourlyProfile to Dict for JSON storage."""
        return {
            "version": "1.0",
            "hourly_factors": profile.hourly_factors,
            "samples_count": profile.samples_count,
            "last_updated": profile.last_updated,
            "confidence": profile.confidence,
            "seasonal_adjustment": profile.seasonal_adjustment,
            "hourly_averages": profile.hourly_averages
        }
    
    @staticmethod
    def prediction_record_to_dict(record: PredictionRecord) -> Dict[str, Any]:
        """Converts PredictionRecord to Dict for JSON storage."""
        return {
            "timestamp": record.timestamp,
            "predicted_value": record.predicted_value,
            "actual_value": record.actual_value,
            "weather_data": record.weather_data,
            "sensor_data": record.sensor_data,
            "accuracy": record.accuracy,
            "model_version": record.model_version
        }