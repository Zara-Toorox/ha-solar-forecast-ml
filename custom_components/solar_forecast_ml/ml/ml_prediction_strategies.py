"""
ML Prediction Strategy Implementations

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
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass
from ..ml.ml_types import LearnedWeights, HourlyProfile
from ..core.core_exceptions import MLModelException

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
    """ML Model-based prediction strategy using learned weights"""
    
    def __init__(self, weights: LearnedWeights, current_accuracy: float, peak_power_kw: float = 0.0):
        self.weights = weights
        self.current_accuracy = current_accuracy
        # Peak power in kW - used to calculate max hourly production in kWh
        self.peak_power_kw = float(peak_power_kw) if peak_power_kw else 0.0
        
        # Calculate max hourly production in kWh
        # Under perfect conditions: 1 kWp ≈ 1 kWh per hour
        # Apply safety margin (20%) for theoretical peak conditions
        from ..const import HOURLY_PRODUCTION_SAFETY_MARGIN, DEFAULT_MAX_HOURLY_KWH
        if self.peak_power_kw > 0:
            self.max_hourly_kwh = self.peak_power_kw * HOURLY_PRODUCTION_SAFETY_MARGIN
        else:
            self.max_hourly_kwh = DEFAULT_MAX_HOURLY_KWH  # Fallback if not configured
            _LOGGER.warning(f"Peak power not configured, using fallback max hourly: {self.max_hourly_kwh} kWh")
    
    async def predict(self, features) -> PredictionResult:
        """Make prediction from features (dict or list)

        Args:
            features: Either Dict[str, float] (V1) or List[float] (V2)

        Returns:
            PredictionResult with prediction and metadata
        """
        if not self.weights:
            raise MLModelException("No trained weights available")

        # Linear prediction: prediction = bias + sum(weight_i * feature_i)
        prediction = self.weights.bias

        # V2: Handle list-based features (ordered by feature_names)
        if isinstance(features, list):
            if not hasattr(self.weights, 'feature_names') or not self.weights.feature_names:
                raise MLModelException("Cannot predict with list features: feature_names not available in weights")

            for i, feature_value in enumerate(features):
                if i >= len(self.weights.feature_names):
                    _LOGGER.warning(f"Feature index {i} exceeds feature_names length, skipping")
                    continue

                feature_name = self.weights.feature_names[i]
                weight = self.weights.weights.get(feature_name, 0.0)
                prediction += weight * feature_value

        # V1: Handle dict-based features
        elif isinstance(features, dict):
            for feature_name, feature_value in features.items():
                weight = self.weights.weights.get(feature_name, 0.0)
                prediction += weight * feature_value

        else:
            raise MLModelException(f"Unsupported features type: {type(features)}. Expected dict or list.")

        # Apply physical limits: Clip to realistic hourly production range
        # Min: 0 kWh (no negative production)
        # Max: peak_power_kw * safety_margin (theoretical maximum under perfect conditions)
        prediction = max(0.0, min(prediction, self.max_hourly_kwh))

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
    
    def _calculate_confidence(self, features) -> float:
        """Calculate confidence from features (dict or list)"""
        if not self.current_accuracy:
            return 0.5

        base_confidence = self.current_accuracy

        # Try to get production_yesterday for confidence boost
        production_yesterday = 0.0
        if isinstance(features, dict):
            production_yesterday = features.get("production_yesterday", 0.0)
        elif isinstance(features, list) and hasattr(self.weights, 'feature_names'):
            # Find production_yesterday in feature_names
            try:
                idx = self.weights.feature_names.index("production_yesterday")
                if idx < len(features):
                    production_yesterday = features[idx]
            except (ValueError, AttributeError):
                pass

        if production_yesterday > 0:
            base_confidence *= 1.1

        # Try to get weather_stability
        weather_stability = 0.5
        if isinstance(features, dict):
            weather_stability = features.get("weather_stability", 0.5)
        elif isinstance(features, list) and hasattr(self.weights, 'feature_names'):
            try:
                idx = self.weights.feature_names.index("weather_stability")
                if idx < len(features):
                    weather_stability = features[idx]
            except (ValueError, AttributeError):
                pass

        base_confidence *= (0.8 + 0.2 * weather_stability)

        return min(1.0, max(0.0, base_confidence))


class ProfileStrategy(PredictionStrategy):
    """Profile-based prediction strategy using historical hourly averages"""

    def __init__(self, profile: HourlyProfile, peak_power_kw: float = 0.0):
        self.profile = profile
        self.peak_power_kw = float(peak_power_kw) if peak_power_kw else 0.0

        # Calculate max hourly production based on system capacity
        from ..const import DEFAULT_MAX_HOURLY_KWH, HOURLY_PRODUCTION_SAFETY_MARGIN
        if self.peak_power_kw > 0:
            # Use configured capacity with safety margin (same as ML and Fallback)
            self.max_hourly_kwh = self.peak_power_kw * HOURLY_PRODUCTION_SAFETY_MARGIN
        else:
            # Fallback to default if not configured
            self.max_hourly_kwh = DEFAULT_MAX_HOURLY_KWH * 2.0  # 6.0 kWh default
            _LOGGER.warning(f"ProfileStrategy: peak_power_kw not configured, using default {self.max_hourly_kwh} kWh")
    
    async def predict(self, features: Dict[str, float]) -> PredictionResult:
        if not self.profile:
            raise MLModelException("No hourly profile available")
        
        hour = int(features.get("hour_of_day", 12))
        base_prediction = self.profile.hourly_averages.get(str(hour), 0.0)
        
        cloudiness = features.get("cloudiness", 50.0)
        cloud_factor = (100 - cloudiness) / 100.0
        
        seasonal_factor = features.get("seasonal_factor", 0.5)
        
        adjusted_prediction = base_prediction * cloud_factor * (0.5 + seasonal_factor)
        # Clip to reasonable range (profile-based, so use generous max)
        prediction = max(0.0, min(adjusted_prediction, self.max_hourly_kwh))
        
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
    """WARNING FIX 5 Simple fallback strategy using solar capacity"""

    def __init__(self, peak_power_kw: float = 0.0):
        """Initialize fallback strategy with system capacity"""
        from ..const import HOURLY_PRODUCTION_SAFETY_MARGIN, DEFAULT_MAX_HOURLY_KWH
        if peak_power_kw > 0:
            # Use configured capacity with safety margin
            self.base_peak_kwh = peak_power_kw * HOURLY_PRODUCTION_SAFETY_MARGIN
        else:
            # Fallback to default if not configured
            self.base_peak_kwh = DEFAULT_MAX_HOURLY_KWH
            _LOGGER.warning(f"FallbackStrategy: solar_capacity not configured, using default {self.base_peak_kwh} kWh")

    async def predict(self, features: Dict[str, float]) -> PredictionResult:
        hour_of_day = features.get("hour_of_day", 12.0)
        cloudiness = features.get("cloudiness", 50.0)
        seasonal_factor = features.get("seasonal_factor", 0.5)

        if hour_of_day < 6 or hour_of_day > 20:
            prediction = 0.0
        else:
            hour_factor = math.sin((hour_of_day - 6) * math.pi / 14)
            cloud_factor = (100 - cloudiness) / 100.0
            # WARNING FIX 5: Use configured solar capacity instead of hardcoded 5000W
            prediction = self.base_peak_kwh * hour_factor * cloud_factor * seasonal_factor

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
        self.peak_power_kw: float = 0.0  # Store for fallback

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

        # WARNING FIX 5: Use stored peak_power_kw for emergency fallback
        fallback = FallbackStrategy(self.peak_power_kw)
        return await fallback.predict(features)
    
    def update_strategies(
        self,
        weights: Optional[LearnedWeights] = None,
        profile: Optional[HourlyProfile] = None,
        accuracy: float = 0.0,
        peak_power_kw: float = 0.0  # Peak power in kW
    ) -> None:
        """Update available prediction strategies with new data"""
        self.strategies.clear()
        self.peak_power_kw = peak_power_kw  # Store for emergency fallback

        if weights:
            self.strategies.append(MLModelStrategy(weights, accuracy, peak_power_kw))

        if profile:
            # CRITICAL FIX: Pass peak_power_kw to ProfileStrategy for correct max values
            self.strategies.append(ProfileStrategy(profile, peak_power_kw))

        # WARNING FIX 5: Pass peak_power_kw to FallbackStrategy
        self.strategies.append(FallbackStrategy(peak_power_kw))