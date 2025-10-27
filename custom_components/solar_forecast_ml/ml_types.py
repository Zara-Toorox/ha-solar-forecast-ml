"""
ML Data Types fÃƒÂ¼r Solar Forecast ML Integration.
PROGRESSIVE UPGRADE v5.1.0: Feature Normalisierung Support

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
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class ErrorCategory(Enum):
    """Error Categories fÃƒÂ¼r ML System."""
    DATA_INTEGRITY = "data_integrity"
    MODEL_TRAINING = "model_training"
    PREDICTION = "prediction"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    PERFORMANCE = "performance"


@dataclass
class PredictionRecord:
    """Individual prediction record for training history."""
    timestamp: str
    predicted_value: float
    actual_value: Optional[float]
    weather_data: Dict[str, Any]
    sensor_data: Dict[str, Any]
    accuracy: float
    model_version: str
    
    def __post_init__(self):
        if self.predicted_value < 0:
            raise ValueError("predicted_value cannot be negative")
        if self.actual_value is not None and self.actual_value < 0:
            raise ValueError("actual_value cannot be negative")
        if not (0.0 <= self.accuracy <= 1.0):
            raise ValueError("accuracy must be between 0 and 1")


@dataclass
class LearnedWeights:
    """
    Learned model weights and metadata.
    PROGRESSIVE UPGRADE: Feature Normalisierung Support
    """
    weather_weights: Dict[str, float]
    seasonal_factors: Dict[str, float]
    correction_factor: float
    accuracy: float
    training_samples: int
    last_trained: str
    model_version: str = "1.0"
    feature_importance: Dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    feature_means: Dict[str, float] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.1 <= self.correction_factor <= 5.0):
            raise ValueError("correction_factor must be between 0.1 and 5.0")
        if not (0.0 <= self.accuracy <= 1.0):
            raise ValueError("accuracy must be between 0 and 1")
        if self.training_samples < 0:
            raise ValueError("training_samples cannot be negative")
        
        if not self.feature_names:
            self.feature_names = [
                "temperature", "humidity", "cloudiness", 
                "wind_speed", "hour_of_day", "seasonal_factor"
            ]
        
        if not self.weights:
            self.weights = self.weather_weights.copy()


@dataclass
class HourlyProfile:
    """
    Hourly production profile for time-based predictions.
    """
    hourly_factors: Dict[str, float]
    samples_count: int
    last_updated: str
    confidence: float = 0.5
    seasonal_adjustment: Dict[str, float] = field(default_factory=dict)
    hourly_averages: Dict[int, float] = field(default_factory=dict)
    
    def __post_init__(self):
        for hour in range(24):
            if str(hour) not in self.hourly_factors:
                self.hourly_factors[str(hour)] = 1.0
        
        if self.samples_count < 0:
            raise ValueError("samples_count cannot be negative")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0 and 1")
        
        if not self.hourly_averages:
            import math
            for hour in range(24):
                if 6 <= hour <= 18:
                    angle = ((hour - 6) / 12) * math.pi
                    self.hourly_averages[hour] = max(0.0, math.sin(angle) * 5.0)
                else:
                    self.hourly_averages[hour] = 0.0


@dataclass
class WeatherFeatures:
    """Structured weather features for ML training."""
    temperature: float
    humidity: float
    cloudiness: float
    wind_speed: float
    pressure: float
    hour_of_day: int
    seasonal_factor: float
    weather_trend: float = 0.0
    uv_index: Optional[float] = None
    
    def __post_init__(self):
        if not (-50 <= self.temperature <= 60):
            raise ValueError("temperature out of valid range [-50, 60]")
        if not (0 <= self.humidity <= 100):
            raise ValueError("humidity must be between 0 and 100")
        if not (0 <= self.cloudiness <= 100):
            raise ValueError("cloudiness must be between 0 and 100")
        if not (0 <= self.hour_of_day <= 23):
            raise ValueError("hour_of_day must be between 0 and 23")
        if self.wind_speed < 0:
            raise ValueError("wind_speed cannot be negative")
        if self.pressure < 800 or self.pressure > 1200:
            raise ValueError("pressure out of realistic range [800, 1200]")


@dataclass
class TrainingDataset:
    """Complete training dataset for ML model."""
    features: List[WeatherFeatures]
    targets: List[float]
    timestamps: List[str]
    data_version: str
    quality_score: float = 1.0
    
    def __post_init__(self):
        if not (len(self.features) == len(self.targets) == len(self.timestamps)):
            raise ValueError("features, targets, and timestamps must have same length")
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError("quality_score must be between 0 and 1")
        for target in self.targets:
            if target < 0:
                raise ValueError("solar production targets cannot be negative")


@dataclass 
class ModelMetrics:
    """Performance metrics for ML model evaluation."""
    mae: float
    rmse: float
    mape: float
    r2: float
    accuracy_percentage: float
    sample_count: int
    
    def __post_init__(self):
        if self.mae < 0:
            raise ValueError("MAE cannot be negative")
        if self.rmse < 0:
            raise ValueError("RMSE cannot be negative")
        if self.mape < 0:
            raise ValueError("MAPE cannot be negative")
        if not (-1.0 <= self.r2 <= 1.0):
            raise ValueError("RÃ‚Â² must be between -1 and 1")
        if not (0.0 <= self.accuracy_percentage <= 100.0):
            raise ValueError("accuracy_percentage must be between 0 and 100")
        if self.sample_count < 0:
            raise ValueError("sample_count cannot be negative")


@dataclass
class PredictionContext:
    """Context information for making predictions."""
    location: Dict[str, float]
    system_capacity: float
    system_type: str
    local_timezone: str
    prediction_horizon_hours: int = 24
    
    def __post_init__(self):
        if "lat" not in self.location or "lon" not in self.location:
            raise ValueError("location must contain 'lat' and 'lon'")
        if not (-90 <= self.location["lat"] <= 90):
            raise ValueError("latitude must be between -90 and 90")
        if not (-180 <= self.location["lon"] <= 180):
            raise ValueError("longitude must be between -180 and 180")
        if self.system_capacity <= 0:
            raise ValueError("system_capacity must be positive")
        if not (1 <= self.prediction_horizon_hours <= 168):
            raise ValueError("prediction_horizon_hours must be between 1 and 168")


def create_default_learned_weights() -> LearnedWeights:
    """
    Create default learned weights for initialization.
    """
    return LearnedWeights(
        weather_weights={
            "temperature": 0.3,
            "humidity": 0.1,
            "cloudiness": 0.4,
            "wind_speed": 0.1,
            "pressure": 0.1
        },
        seasonal_factors={
            "spring": 1.0,
            "summer": 1.0,
            "autumn": 1.0,
            "winter": 1.0
        },
        correction_factor=1.0,
        accuracy=0.5,
        training_samples=0,
        last_trained=dt_util.utcnow().isoformat(),
        model_version="1.0",
        bias=0.0,
        weights={},
        feature_names=[
            "temperature", "humidity", "cloudiness", 
            "wind_speed", "hour_of_day", "seasonal_factor"
        ],
        feature_means={},
        feature_stds={}
    )


def create_default_hourly_profile() -> HourlyProfile:
    """
    Create default hourly profile for initialization.
    """
    import math
    hourly_averages = {}
    for hour in range(24):
        if 6 <= hour <= 18:
            angle = ((hour - 6) / 12) * math.pi
            hourly_averages[hour] = max(0.0, math.sin(angle) * 5.0)
        else:
            hourly_averages[hour] = 0.0
    
    return HourlyProfile(
        hourly_factors={str(hour): 1.0 for hour in range(24)},
        samples_count=0,
        last_updated=dt_util.utcnow().isoformat(),
        confidence=0.5,
        hourly_averages=hourly_averages
    )


def validate_prediction_record(record: Dict[str, Any]) -> bool:
    """Validate a prediction record dictionary."""
    required_fields = [
        "timestamp", "predicted_value", "weather_data", 
        "sensor_data", "accuracy", "model_version"
    ]
    
    for field in required_fields:
        if field not in record:
            raise ValueError(f"Missing required field: {field}")
    
    if not isinstance(record["predicted_value"], (int, float)):
        raise ValueError("predicted_value must be numeric")
    
    if record["actual_value"] is not None and not isinstance(record["actual_value"], (int, float)):
        raise ValueError("actual_value must be numeric or None")
    
    if not isinstance(record["accuracy"], (int, float)):
        raise ValueError("accuracy must be numeric")
    
    if record["predicted_value"] < 0:
        raise ValueError("predicted_value cannot be negative")
    
    if record["actual_value"] is not None and record["actual_value"] < 0:
        raise ValueError("actual_value cannot be negative")
    
    if not (0.0 <= record["accuracy"] <= 1.0):
        raise ValueError("accuracy must be between 0 and 1")
    
    return True


def sanitize_weather_data(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize and validate weather data."""
    sanitized = {}
    
    temp = weather_data.get("temperature", 15.0)
    sanitized["temperature"] = max(-50, min(60, float(temp)))
    
    humidity = weather_data.get("humidity", 60.0)
    sanitized["humidity"] = max(0, min(100, float(humidity)))
    
    clouds = weather_data.get("cloudiness", 50.0)
    sanitized["cloudiness"] = max(0, min(100, float(clouds)))
    
    wind = weather_data.get("wind_speed", 5.0)
    sanitized["wind_speed"] = max(0, float(wind))
    
    pressure = weather_data.get("pressure", 1013.0)
    sanitized["pressure"] = max(800, min(1200, float(pressure)))
    
    return sanitized
