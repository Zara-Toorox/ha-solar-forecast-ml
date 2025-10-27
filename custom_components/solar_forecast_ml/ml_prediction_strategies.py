"""
Prediction Strategies mit Strategy Pattern.

Copyright (C) 2025 Zara-Toorox

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""
import math
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass
from .ml_types import LearnedWeights, HourlyProfile
from .exceptions import MLModelException

_LOGGER = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    prediction: float
    confidence: float
    method: str
    features_used: Dict[str, float]
    model_accuracy: Optional[float] = None


class PredictionStrategy(ABC):
    
    @abstractmethod
    async def predict(self, features: Dict[str, float]) -> PredictionResult:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class MLModelStrategy(PredictionStrategy):
    
    def __init__(self, weights: LearnedWeights, current_accuracy: float):
        self.weights = weights
        self.current_accuracy = current_accuracy
    
    async def predict(self, features: Dict[str, float]) -> PredictionResult:
        if not self.weights:
            raise MLModelException("No trained weights available")
        
        prediction = self.weights.bias
        
        for feature_name, feature_value in features.items():
            weight = self.weights.weights.get(feature_name, 0.0)
            prediction += weight * feature_value
        
        prediction = max(0.0, min(prediction, 15000.0))
        confidence = self._calculate_confidence(features)
        
        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            method="ml_model",
            features_used=features,
            model_accuracy=self.current_accuracy
        )
    
    def is_available(self) -> bool:
        return self.weights is not None
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        if not self.current_accuracy:
            return 0.5
        
        base_confidence = self.current_accuracy
        
        if features.get("production_yesterday", 0.0) > 0:
            base_confidence *= 1.1
        
        weather_stability = features.get("weather_stability", 0.5)
        base_confidence *= (0.8 + 0.2 * weather_stability)
        
        return min(1.0, max(0.0, base_confidence))


class ProfileStrategy(PredictionStrategy):
    
    def __init__(self, profile: HourlyProfile):
        self.profile = profile
    
    async def predict(self, features: Dict[str, float]) -> PredictionResult:
        if not self.profile:
            raise MLModelException("No hourly profile available")
        
        hour = int(features.get("hour_of_day", 12))
        base_prediction = self.profile.hourly_averages.get(str(hour), 0.0)
        
        cloudiness = features.get("cloudiness", 50.0)
        cloud_factor = (100 - cloudiness) / 100.0
        
        seasonal_factor = features.get("seasonal_factor", 0.5)
        
        adjusted_prediction = base_prediction * cloud_factor * (0.5 + seasonal_factor)
        prediction = max(0.0, min(adjusted_prediction, 15000.0))
        
        confidence = 0.6
        
        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            method="hourly_profile",
            features_used=features,
            model_accuracy=None
        )
    
    def is_available(self) -> bool:
        return self.profile is not None and self.profile.samples_count > 10


class FallbackStrategy(PredictionStrategy):
    
    async def predict(self, features: Dict[str, float]) -> PredictionResult:
        hour_of_day = features.get("hour_of_day", 12.0)
        cloudiness = features.get("cloudiness", 50.0)
        seasonal_factor = features.get("seasonal_factor", 0.5)
        
        if hour_of_day < 6 or hour_of_day > 20:
            prediction = 0.0
        else:
            hour_factor = math.sin((hour_of_day - 6) * math.pi / 14)
            cloud_factor = (100 - cloudiness) / 100.0
            base_peak = 5000.0
            prediction = base_peak * hour_factor * cloud_factor * seasonal_factor
        
        prediction = max(0.0, prediction)
        
        return PredictionResult(
            prediction=prediction,
            confidence=0.3,
            method="simple_fallback",
            features_used=features,
            model_accuracy=None
        )
    
    def is_available(self) -> bool:
        return True


class PredictionOrchestrator:
    
    def __init__(self):
        self.strategies: list[PredictionStrategy] = []
    
    def register_strategy(self, strategy: PredictionStrategy) -> None:
        self.strategies.append(strategy)
    
    async def predict(self, features: Dict[str, float]) -> PredictionResult:
        for strategy in self.strategies:
            if strategy.is_available():
                try:
                    return await strategy.predict(features)
                except Exception as e:
                    _LOGGER.warning(f"Strategy {strategy.__class__.__name__} failed: {e}")
                    continue
        
        fallback = FallbackStrategy()
        return await fallback.predict(features)
    
    def update_strategies(
        self, 
        weights: Optional[LearnedWeights] = None,
        profile: Optional[HourlyProfile] = None,
        accuracy: float = 0.0
    ) -> None:
        self.strategies.clear()
        
        if weights:
            self.strategies.append(MLModelStrategy(weights, accuracy))
        
        if profile:
            self.strategies.append(ProfileStrategy(profile))
        
        self.strategies.append(FallbackStrategy())
