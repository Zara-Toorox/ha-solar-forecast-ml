"""ML Type Definitions and Data Classes V12.2.0 @zara

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
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from enum import Enum
from typing import Any, Dict, List, Optional

from ..const import CORRECTION_FACTOR_MAX, CORRECTION_FACTOR_MIN, ML_MODEL_VERSION

from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Enumeration of error categories within the ML system"""

    DATA_INTEGRITY = "data_integrity"
    MODEL_TRAINING = "model_training"
    PREDICTION = "prediction"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    EXTERNAL_SERVICE = "external_service"
    PERFORMANCE = "performance"
    UNKNOWN = "unknown"

@dataclass
class PredictionRecord:
    """Represents a single prediction record typically stored in history"""

    timestamp: str
    predicted_value: float
    actual_value: Optional[float]
    weather_data: Dict[str, Any]
    sensor_data: Dict[str, Any]
    accuracy: float
    model_version: str

    def __post_init__(self):
        """Perform validation after initialization @zara"""
        if self.predicted_value < 0:
            _LOGGER.warning(
                f"PredictionRecord created with negative predicted_value: {self.predicted_value}. Clamping to 0."
            )
            self.predicted_value = 0.0

        if self.actual_value is not None and self.actual_value < 0:
            _LOGGER.warning(
                f"PredictionRecord created with negative actual_value: {self.actual_value}. Clamping to 0."
            )
            self.actual_value = 0.0

        if not (0.0 <= self.accuracy <= 1.0):
            _LOGGER.warning(
                f"PredictionRecord accuracy {self.accuracy} outside [0, 1] range. Clamping."
            )
            self.accuracy = max(0.0, min(1.0, self.accuracy))

@dataclass
class LearnedWeights:
    """Stores the learned parameters weights bias of the ML model"""

    weights: Dict[str, Any]
    bias: float
    feature_names: List[str]

    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]

    accuracy: float
    training_samples: int
    last_trained: str
    model_version: str = ML_MODEL_VERSION

    algorithm_used: str = "ridge"

    correction_factor: float = 1.0

    weather_weights: Dict[str, float] = field(default_factory=dict)
    seasonal_factors: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(
        default_factory=dict
    )

    def __post_init__(self):
        """Validate fields after initialization @zara"""
        if not (CORRECTION_FACTOR_MIN <= self.correction_factor <= CORRECTION_FACTOR_MAX):
            _LOGGER.warning(
                f"LearnedWeights correction_factor {self.correction_factor} outside valid range "
                f"[{CORRECTION_FACTOR_MIN}, {CORRECTION_FACTOR_MAX}]. Clamping."
            )
            self.correction_factor = max(
                CORRECTION_FACTOR_MIN, min(CORRECTION_FACTOR_MAX, self.correction_factor)
            )

        if not (0.0 <= self.accuracy <= 1.0):
            _LOGGER.debug(
                f"LearnedWeights accuracy {self.accuracy} outside [0, 1] range. Clamping to valid range."
            )
            self.accuracy = max(0.0, min(1.0, self.accuracy))

        if self.training_samples < 0:
            _LOGGER.error(
                f"LearnedWeights training_samples cannot be negative ({self.training_samples}). Setting to 0."
            )
            self.training_samples = 0

        if self.algorithm_used == "ridge" and set(self.feature_names) != set(self.weights.keys()) and self.weights:
            _LOGGER.debug(
                "Feature names list does not perfectly match keys in 'weights' dictionary."
            )
        if set(self.feature_names) != set(self.feature_means.keys()) and self.feature_means:
            _LOGGER.debug(
                "Feature names list does not perfectly match keys in 'feature_means' dictionary."
            )
        if set(self.feature_names) != set(self.feature_stds.keys()) and self.feature_stds:
            _LOGGER.debug(
                "Feature names list does not perfectly match keys in 'feature_stds' dictionary."
            )

@dataclass
class HourlyProfile:
    """Represents the typical hourly production profile often used as a fallback"""

    hourly_averages: Dict[str, float]

    samples_count: int
    last_updated: str
    confidence: float = 0.5

    hourly_factors: Dict[str, float] = field(default_factory=dict)
    seasonal_adjustment: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and ensure profile structure @zara"""
        if self.samples_count < 0:
            _LOGGER.error(
                f"HourlyProfile samples_count cannot be negative ({self.samples_count}). Setting to 0."
            )
            self.samples_count = 0

        if not (0.0 <= self.confidence <= 1.0):
            _LOGGER.warning(
                f"HourlyProfile confidence {self.confidence} outside [0, 1] range. Clamping."
            )
            self.confidence = max(0.0, min(1.0, self.confidence))

        for hour in range(24):
            hour_str = str(hour)
            if hour_str not in self.hourly_averages:
                _LOGGER.debug(f"HourlyProfile missing average for hour {hour_str}, setting to 0.0.")
                self.hourly_averages[hour_str] = 0.0
            else:
                value = self.hourly_averages[hour_str]
                # Handle old dict format: {"count": 0, "total": 0.0, "average": 0.5}
                if isinstance(value, dict):
                    self.hourly_averages[hour_str] = float(value.get("average", 0.0))
                elif not isinstance(value, (int, float)):
                    _LOGGER.warning(
                        f"HourlyProfile average for hour {hour_str} has invalid type ({type(value)}). Setting to 0.0."
                    )
                    self.hourly_averages[hour_str] = 0.0
                elif value < 0:
                    _LOGGER.warning(
                        f"HourlyProfile average for hour {hour_str} is negative ({value}). Clamping to 0.0."
                    )
                    self.hourly_averages[hour_str] = 0.0

def create_default_learned_weights() -> LearnedWeights:
    """Creates a LearnedWeights object with default values - V3 with 14 features @zara

    CRITICAL: These feature names MUST match the production files!
    Production reference: /Volumes/config/solar_forecast_ml/ml/learned_weights.json
    """

    default_feature_names = [
        "hour_of_day",
        "day_of_year",
        "season_encoded",
        "weather_temp",
        "weather_solar_radiation_wm2",
        "weather_wind",
        "weather_humidity",
        "weather_rain",
        "weather_clouds",
        "sun_elevation_deg",
        "theoretical_max_kwh",
        "clear_sky_radiation_wm2",
        "production_yesterday",
        "production_same_hour_yesterday",
    ]

    _LOGGER.info(
        f"Creating default LearnedWeights with {len(default_feature_names)} V3 features (triggers immediate training)"
    )

    return LearnedWeights(
        weights={},
        bias=0.0,
        feature_names=default_feature_names,
        feature_means={},
        feature_stds={},
        accuracy=0.0,
        training_samples=0,
        last_trained=dt_util.now().isoformat(),
        model_version=ML_MODEL_VERSION,
        correction_factor=1.0,

        weather_weights={},
        seasonal_factors={},
        feature_importance={},
    )

def create_default_hourly_profile() -> HourlyProfile:
    """Creates a default HourlyProfile typically with a simple sine curve approximation @zara"""
    default_hourly_averages: Dict[str, float] = {}

    daylight_hours = 12
    start_hour = 6
    peak_value = 1.0

    for hour in range(24):
        hour_str = str(hour)
        if start_hour <= hour < start_hour + daylight_hours:

            relative_hour = (hour - start_hour) / daylight_hours

            sine_value = math.sin(relative_hour * math.pi + 0.01)
            default_hourly_averages[hour_str] = max(
                0.0, sine_value * peak_value
            )
        else:
            default_hourly_averages[hour_str] = 0.0

    return HourlyProfile(
        hourly_averages=default_hourly_averages,
        samples_count=0,
        last_updated=dt_util.now().isoformat(),
        confidence=0.1,

        hourly_factors={},
        seasonal_adjustment={},
    )

def validate_prediction_record(record: Dict[str, Any]) -> bool:
    """Validates a dictionary intended to be converted into a PredictionRecord @zara"""
    required_fields = [
        "timestamp",
        "predicted_value",
        "weather_data",
        "sensor_data",
        "accuracy",
        "model_version",

    ]

    for field_name in required_fields:
        if field_name not in record:
            raise ValueError(f"Missing required field in prediction record: '{field_name}'")

    if not isinstance(record["timestamp"], str):
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

    actual_val = record.get("actual_value")
    if actual_val is not None and (not isinstance(actual_val, (int, float)) or actual_val < 0):
        raise ValueError("prediction record 'actual_value' must be a non-negative number or None")

    return True
