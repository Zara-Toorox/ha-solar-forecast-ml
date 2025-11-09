"""
ML Type Definitions and Data Classes

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

import math
import logging # Import logging for potential use in validation
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, tzinfo # Ensure timezone is imported
from enum import Enum

# Import SafeDateTimeUtil if needed for default timestamps
from ..core.core_helpers import SafeDateTimeUtil as dt_util
# Import constants if needed for defaults or validation ranges
from ..const import ML_MODEL_VERSION, CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX

_LOGGER = logging.getLogger(__name__)


# --- Enums ---
class ErrorCategory(Enum):
    """Enumeration of error categories within the ML system"""
    DATA_INTEGRITY = "data_integrity"
    MODEL_TRAINING = "model_training"
    PREDICTION = "prediction"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency" # Added for dependency issues
    EXTERNAL_SERVICE = "external_service" # e.g., Weather API
    PERFORMANCE = "performance"
    UNKNOWN = "unknown"


# --- Dataclasses for Data Structures ---

@dataclass
class PredictionRecord:
    """Represents a single prediction record typically stored in history"""
    timestamp: str          # ISO 8601 timestamp string (preferably UTC)
    predicted_value: float  # The predicted energy value (e.g., kWh)
    actual_value: Optional[float] # The actual energy value (if available)
    weather_data: Dict[str, Any] # Weather conditions at the time of prediction
    sensor_data: Dict[str, Any]  # External sensor readings used for prediction
    accuracy: float         # Calculated accuracy (0.0 to 1.0) based on actual vs predicted
    model_version: str      # Version of the ML model used for this prediction

    def __post_init__(self):
        """Perform validation after initialization"""
        if self.predicted_value < 0:
            _LOGGER.warning(f"PredictionRecord created with negative predicted_value: {self.predicted_value}. Clamping to 0.")
            self.predicted_value = 0.0
            # raise ValueError("predicted_value cannot be negative")
        if self.actual_value is not None and self.actual_value < 0:
            _LOGGER.warning(f"PredictionRecord created with negative actual_value: {self.actual_value}. Clamping to 0.")
            self.actual_value = 0.0
            # raise ValueError("actual_value cannot be negative")
        if not (0.0 <= self.accuracy <= 1.0):
             _LOGGER.warning(f"PredictionRecord accuracy {self.accuracy} outside [0, 1] range. Clamping.")
             self.accuracy = max(0.0, min(1.0, self.accuracy))
             # raise ValueError("accuracy must be between 0.0 and 1.0")
        # Validate timestamp format? Could be expensive. Assume valid ISO string for now.


@dataclass
class LearnedWeights:
    """Stores the learned parameters weights bias of the ML model"""
    # Core model parameters
    weights: Dict[str, float]           # Weights for each named feature
    bias: float                         # Bias (intercept) term
    feature_names: List[str]            # Order of features corresponding to weights during training

    # Scaler state (for normalization)
    feature_means: Dict[str, float]     # Mean of each feature used for scaling
    feature_stds: Dict[str, float]      # Standard deviation of each feature used for scaling

    # Metadata
    accuracy: float                     # R-squared accuracy achieved during last training (0.0 to 1.0)
    training_samples: int               # Number of samples used in the last training
    last_trained: str                   # ISO 8601 timestamp string (UTC) of last training
    model_version: str = ML_MODEL_VERSION # Version of the model structure/logic

    # Fallback / Auxiliary Parameters
    # Learned factor applied by rule-based strategy if ML fails
    correction_factor: float = 1.0

    # Deprecated fields (kept for potential backward compatibility during loading)
    weather_weights: Dict[str, float] = field(default_factory=dict)
    seasonal_factors: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict) # Can be calculated post-training

    def __post_init__(self):
        """Validate fields after initialization"""
        if not (CORRECTION_FACTOR_MIN <= self.correction_factor <= CORRECTION_FACTOR_MAX):
            _LOGGER.warning(f"LearnedWeights correction_factor {self.correction_factor} outside valid range "
                            f"[{CORRECTION_FACTOR_MIN}, {CORRECTION_FACTOR_MAX}]. Clamping.")
            self.correction_factor = max(CORRECTION_FACTOR_MIN, min(CORRECTION_FACTOR_MAX, self.correction_factor))
            # raise ValueError(f"correction_factor must be between {CORRECTION_FACTOR_MIN} and {CORRECTION_FACTOR_MAX}")
        if not (0.0 <= self.accuracy <= 1.0):
             _LOGGER.warning(f"LearnedWeights accuracy {self.accuracy} outside [0, 1] range. Clamping.")
             self.accuracy = max(0.0, min(1.0, self.accuracy))
            # raise ValueError("accuracy must be between 0.0 and 1.0")
        if self.training_samples < 0:
            _LOGGER.error(f"LearnedWeights training_samples cannot be negative ({self.training_samples}). Setting to 0.")
            self.training_samples = 0
            # raise ValueError("training_samples cannot be negative")

        # Ensure feature names list matches keys in weights, means, stds if possible?
        # This might be too strict if features evolve. Log a warning if discrepancies found.
        if set(self.feature_names) != set(self.weights.keys()) and self.weights:
             _LOGGER.debug("Feature names list does not perfectly match keys in 'weights' dictionary.")
        if set(self.feature_names) != set(self.feature_means.keys()) and self.feature_means:
             _LOGGER.debug("Feature names list does not perfectly match keys in 'feature_means' dictionary.")
        if set(self.feature_names) != set(self.feature_stds.keys()) and self.feature_stds:
             _LOGGER.debug("Feature names list does not perfectly match keys in 'feature_stds' dictionary.")


@dataclass
class HourlyProfile:
    """Represents the typical hourly production profile often used as a fallback"""
    # Calculated median/average production value for each hour (0-23)
    # Keys should be strings '0' through '23' for JSON compatibility
    hourly_averages: Dict[str, float]

    # Metadata
    samples_count: int      # Number of valid hourly samples used to build the profile
    last_updated: str       # ISO 8601 timestamp string (UTC) when profile was last calculated
    confidence: float = 0.5 # Confidence score (0.0 to 1.0) based on sample count/variance

    # Deprecated fields (can be removed if loading handles their absence)
    hourly_factors: Dict[str, float] = field(default_factory=dict)
    seasonal_adjustment: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and ensure profile structure"""
        if self.samples_count < 0:
            _LOGGER.error(f"HourlyProfile samples_count cannot be negative ({self.samples_count}). Setting to 0.")
            self.samples_count = 0
            # raise ValueError("samples_count cannot be negative")
        if not (0.0 <= self.confidence <= 1.0):
             _LOGGER.warning(f"HourlyProfile confidence {self.confidence} outside [0, 1] range. Clamping.")
             self.confidence = max(0.0, min(1.0, self.confidence))
            # raise ValueError("confidence must be between 0.0 and 1.0")

        # Ensure all hours (0-23 as strings) are present in hourly_averages, default to 0.0
        for hour in range(24):
            hour_str = str(hour)
            if hour_str not in self.hourly_averages:
                 _LOGGER.debug(f"HourlyProfile missing average for hour {hour_str}, setting to 0.0.")
                 self.hourly_averages[hour_str] = 0.0
            elif self.hourly_averages[hour_str] < 0:
                 _LOGGER.warning(f"HourlyProfile average for hour {hour_str} is negative ({self.hourly_averages[hour_str]}). Clamping to 0.0.")
                 self.hourly_averages[hour_str] = 0.0


# --- Default Creation Functions ---

def create_default_learned_weights() -> LearnedWeights:
    """Creates a LearnedWeights object with default values used for initialization"""
    # CRITICAL FIX 1: Match FeatureEngineer.feature_names (27 features)
    # Base features (18)
    default_feature_names = [
        "temperature", "humidity", "cloudiness", "wind_speed",
        "hour_of_day", "seasonal_factor", "weather_trend",
        "production_yesterday", "production_same_hour_yesterday",
        "cloudiness_primary", "cloud_impact", "sunshine_factor",
        # BETA EXPANSION: Additional weather features (optional sensors)
        "rain", "uv_index", "lux",
        # IMPROVEMENT 7: Cloudiness trend features
        "cloudiness_trend_1h", "cloudiness_trend_3h", "cloudiness_volatility"
    ]

    # Polynomial features (4)
    default_feature_names.extend([
        "temperature_sq", "cloudiness_sq", "hour_of_day_sq", "seasonal_factor_sq"
    ])

    # Interaction features (5)
    default_feature_names.extend([
        "cloudiness_x_hour", "temperature_x_seasonal", "humidity_x_cloudiness",
        "wind_x_hour", "weather_trend_x_seasonal"
    ])

    _LOGGER.info(f"Creating default LearnedWeights with {len(default_feature_names)} features (triggers immediate training)")

    return LearnedWeights(
        weights={}, # Empty weights trigger training
        bias=0.0,
        feature_names=default_feature_names,
        feature_means={}, # Scaler not fitted yet
        feature_stds={},  # Scaler not fitted yet
        accuracy=0.0, # Start with zero accuracy (triggers training)
        training_samples=0, # No samples yet (triggers training)
        last_trained=dt_util.now().isoformat(), # Timestamp of creation (LOCAL time)
        model_version=ML_MODEL_VERSION,
        correction_factor=1.0, # Default fallback factor
        # Deprecated fields initialized empty
        weather_weights={},
        seasonal_factors={},
        feature_importance={}
    )


def create_default_hourly_profile() -> HourlyProfile:
    """Creates a default HourlyProfile typically with a simple sine curve approximation"""
    default_hourly_averages: Dict[str, float] = {}
    # Simple sine curve approximation for hours 6 AM to 6 PM (18:00)
    daylight_hours = 12
    start_hour = 6
    peak_value = 1.0 # Normalized peak, actual value depends on capacity/weather

    for hour in range(24):
        hour_str = str(hour)
        if start_hour <= hour < start_hour + daylight_hours:
            # Calculate position within daylight hours (0 to 1)
            relative_hour = (hour - start_hour) / daylight_hours
            # Apply sine function (0 at start/end, 1 at midpoint)
            # Add small offset to avoid issues right at the edges if needed
            sine_value = math.sin(relative_hour * math.pi + 0.01)
            default_hourly_averages[hour_str] = max(0.0, sine_value * peak_value) # Ensure non-negative
        else:
            default_hourly_averages[hour_str] = 0.0 # Night time

    return HourlyProfile(
        hourly_averages=default_hourly_averages,
        samples_count=0, # No real samples used
        last_updated=dt_util.now().isoformat(), # LOCAL time
        confidence=0.1, # Low confidence for default profile
        # Deprecated fields initialized empty
        hourly_factors={},
        seasonal_adjustment={}
    )


# --- Validation Function ---

def validate_prediction_record(record: Dict[str, Any]) -> bool:
    """Validates a dictionary intended to be converted into a PredictionRecord"""
    required_fields = [
        "timestamp", "predicted_value", "weather_data",
        "sensor_data", "accuracy", "model_version"
        # actual_value is Optional
    ]

    # Check for missing required fields
    for field_name in required_fields:
        if field_name not in record:
            raise ValueError(f"Missing required field in prediction record: '{field_name}'")

    # Check types and ranges
    if not isinstance(record["timestamp"], str): # Basic check, could parse if needed
        raise ValueError("prediction record 'timestamp' must be a string")
    if not isinstance(record["predicted_value"], (int, float)) or record["predicted_value"] < 0:
        raise ValueError("prediction record 'predicted_value' must be a non-negative number")
    if not isinstance(record["weather_data"], dict):
        raise ValueError("prediction record 'weather_data' must be a dictionary")
    if not isinstance(record["sensor_data"], dict):
        raise ValueError("prediction record 'sensor_data' must be a dictionary")
    if not isinstance(record["accuracy"], (int, float)) or not (0.0 <= record["accuracy"] <= 1.0):
        raise ValueError("prediction record 'accuracy' must be a number between 0.0 and 1.0")
    if not isinstance(record["model_version"], str):
        raise ValueError("prediction record 'model_version' must be a string")

    # Check optional actual_value if present
    actual_val = record.get("actual_value")
    if actual_val is not None and (not isinstance(actual_val, (int, float)) or actual_val < 0):
        raise ValueError("prediction record 'actual_value' must be a non-negative number or None")

    return True # Validation passed

# Note: sanitize_weather_data function removed as it's more related to
# weather service input processing than core ML types. It can reside in
# weather_service.py or a dedicated data cleaning module if needed.