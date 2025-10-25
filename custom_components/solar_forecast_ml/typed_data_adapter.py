"""
Typed Data Adapter für Solar Forecast ML Integration.
✅ NEU: Konvertiert Dict → Dataclass für Type-Safety // von Zara

Copyright (C) 2025 Zara-Toorox
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .ml_types import (
    PredictionRecord, LearnedWeights, HourlyProfile,
    create_default_learned_weights, create_default_hourly_profile
)
from .exceptions import DataIntegrityException, ValidationException

_LOGGER = logging.getLogger(__name__)


class TypedDataAdapter:
    """
    Adapter für Konvertierung zwischen Dict (JSON) und Dataclasses.
    ✅ NEU: Type-Safety ohne JSON-Struktur zu brechen // von Zara
    """
    
    @staticmethod
    def dict_to_learned_weights(data: Dict[str, Any]) -> LearnedWeights:
        """
        Konvertiert Dict zu LearnedWeights Dataclass.
        ✅ ROBUST: Validierung + Defaults // von Zara
        """
        try:
            # Extrahiere weather_weights // von Zara
            weather_weights = data.get("weather_weights", {})
            if not weather_weights:
                weather_weights = {
                    "temperature": 0.3,
                    "humidity": 0.1,
                    "cloudiness": 0.4,
                    "wind_speed": 0.1,
                    "pressure": 0.1
                }
            
            # Extrahiere seasonal_factors // von Zara
            seasonal_factors = data.get("seasonal_factors", {})
            if not seasonal_factors:
                seasonal_factors = {
                    "spring": 1.0,
                    "summer": 1.0,
                    "autumn": 1.0,
                    "winter": 1.0
                }
            
            # Correction factor mit Bounds Check // von Zara
            correction_factor = float(data.get("correction_factor", 1.0))
            correction_factor = max(0.1, min(5.0, correction_factor))
            
            # Bias mit Default // von Zara
            bias = float(data.get("bias", 0.0))
            
            # Feature names // von Zara
            feature_names = data.get("feature_names", [
                "temperature", "humidity", "cloudiness", 
                "wind_speed", "hour_of_day", "seasonal_factor"
            ])
            
            # Weights als Dict (für lineares Modell) // von Zara
            weights = data.get("weights", weather_weights.copy())
            
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
                feature_names=feature_names
            )
            
        except Exception as e:
            _LOGGER.error("❌ Failed to convert dict to LearnedWeights: %s", str(e))
            return create_default_learned_weights()
    
    @staticmethod
    def learned_weights_to_dict(weights: LearnedWeights) -> Dict[str, Any]:
        """
        Konvertiert LearnedWeights zu Dict für JSON-Speicherung.
        ✅ VOLLSTÄNDIG: Alle Felder // von Zara
        """
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
            "updated": datetime.now().isoformat()
        }
    
    @staticmethod
    def dict_to_hourly_profile(data: Dict[str, Any]) -> HourlyProfile:
        """
        Konvertiert Dict zu HourlyProfile Dataclass.
        ✅ ROBUST: Validierung + Defaults // von Zara
        """
        try:
            # Hourly factors (24 Stunden) // von Zara
            hourly_factors = data.get("hourly_factors", {})
            if len(hourly_factors) != 24:
                hourly_factors = {str(h): 1.0 for h in range(24)}
            
            # Hourly averages (für Profile-Based Prediction) // von Zara
            hourly_averages = data.get("hourly_averages", {})
            if not hourly_averages:
                # Default: Sinus-Kurve für Tagesverlauf // von Zara
                import math
                hourly_averages = {}
                for hour in range(24):
                    if 6 <= hour <= 18:
                        # Sinus-Kurve zwischen Sonnenauf- und untergang // von Zara
                        angle = ((hour - 6) / 12) * math.pi
                        hourly_averages[hour] = max(0.0, math.sin(angle) * 5.0)
                    else:
                        hourly_averages[hour] = 0.0
            
            return HourlyProfile(
                hourly_factors=hourly_factors,
                samples_count=int(data.get("samples_count", 0)),
                last_updated=data.get("last_updated", datetime.now().isoformat()),
                confidence=float(data.get("confidence", 0.5)),
                seasonal_adjustment=data.get("seasonal_adjustment", {}),
                hourly_averages=hourly_averages
            )
            
        except Exception as e:
            _LOGGER.error("❌ Failed to convert dict to HourlyProfile: %s", str(e))
            return create_default_hourly_profile()
    
    @staticmethod
    def hourly_profile_to_dict(profile: HourlyProfile) -> Dict[str, Any]:
        """
        Konvertiert HourlyProfile zu Dict für JSON-Speicherung.
        ✅ VOLLSTÄNDIG: Alle Felder // von Zara
        """
        return {
            "version": "1.0",
            "hourly_factors": profile.hourly_factors,
            "samples_count": profile.samples_count,
            "last_updated": profile.last_updated,
            "confidence": profile.confidence,
            "seasonal_adjustment": profile.seasonal_adjustment,
            "hourly_averages": profile.hourly_averages,
            "updated": datetime.now().isoformat()
        }
    
    @staticmethod
    def dict_to_prediction_record(data: Dict[str, Any]) -> PredictionRecord:
        """
        Konvertiert Dict zu PredictionRecord Dataclass.
        ✅ VALIDIERT: Vollständige Prüfung // von Zara
        """
        try:
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
            raise ValidationException(
                f"Invalid prediction record: {str(e)}",
                field_name="prediction_record"
            )
    
    @staticmethod
    def prediction_record_to_dict(record: PredictionRecord) -> Dict[str, Any]:
        """
        Konvertiert PredictionRecord zu Dict.
        ✅ VOLLSTÄNDIG // von Zara
        """
        return {
            "timestamp": record.timestamp,
            "predicted_value": record.predicted_value,
            "actual_value": record.actual_value,
            "weather_data": record.weather_data,
            "sensor_data": record.sensor_data,
            "accuracy": record.accuracy,
            "model_version": record.model_version
        }
    
    @staticmethod
    def validate_and_convert_history(history_data: Dict[str, Any]) -> List[PredictionRecord]:
        """
        Validiert und konvertiert komplette History.
        ✅ BATCH-PROZESS: Effizient // von Zara
        """
        records = []
        predictions_list = history_data.get("predictions", [])
        
        for i, pred_dict in enumerate(predictions_list):
            try:
                record = TypedDataAdapter.dict_to_prediction_record(pred_dict)
                records.append(record)
            except Exception as e:
                _LOGGER.debug("âš ⚠ Skipping invalid history record %d: %s", i, str(e))
                continue
        
        return records
