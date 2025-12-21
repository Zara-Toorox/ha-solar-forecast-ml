"""ML Prediction Strategy Implementations V12.2.0 @zara

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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from ..core.core_exceptions import MLModelException
from ..ml.ml_types import HourlyProfile, LearnedWeights

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

    def __init__(
        self, weights: LearnedWeights, current_accuracy: float, peak_power_kw: float = 0.0
    ):
        self.weights = weights
        self.current_accuracy = current_accuracy

        self.peak_power_kw = float(peak_power_kw) if peak_power_kw else 0.0

        self.algorithm_used = getattr(weights, 'algorithm_used', 'ridge') if weights else 'ridge'

        self._historical_sequence = None

        self._lstm_trainer = None
        self._sequence_builder = None
        if self.algorithm_used == "tiny_lstm" and weights:
            try:
                from ..ml.ml_tiny_lstm import TinyLSTM
                from ..ml.ml_sequence_builder import SequenceBuilder

                lstm_weights = weights.weights
                self._lstm_trainer = TinyLSTM(
                    input_size=lstm_weights.get('input_size', 14),
                    hidden_size=lstm_weights.get('hidden_size', 32),
                    sequence_length=lstm_weights.get('sequence_length', 24)
                )
                self._lstm_trainer.set_weights(lstm_weights)
                self._sequence_builder = SequenceBuilder(
                    sequence_length=lstm_weights.get('sequence_length', 24)
                )
                _LOGGER.info("MLModelStrategy initialized with TinyLSTM inference + SequenceBuilder")
            except Exception as e:
                _LOGGER.error(f"Failed to initialize LSTM for inference: {e}. Falling back to Ridge.")
                self.algorithm_used = "ridge"

        from ..const import DEFAULT_MAX_HOURLY_KWH, HOURLY_PRODUCTION_SAFETY_MARGIN

        if self.peak_power_kw > 0:
            self.max_hourly_kwh = self.peak_power_kw * HOURLY_PRODUCTION_SAFETY_MARGIN
        else:
            self.max_hourly_kwh = DEFAULT_MAX_HOURLY_KWH
            _LOGGER.warning(
                f"Peak power not configured, using fallback max hourly: {self.max_hourly_kwh} kWh"
            )

    def set_historical_sequence(self, recent_hours: Optional[list]) -> None:
        """Set historical data for LSTM sequence building @zara"""
        self._historical_sequence = recent_hours
        if recent_hours:
            _LOGGER.debug(f"Historical sequence set: {len(recent_hours)} hours")

    async def predict(self, features) -> PredictionResult:
        """Make prediction from features (dict or list) @zara"""
        if not self.weights:
            raise MLModelException("No trained weights available")

        if self.algorithm_used == "tiny_lstm" and self._lstm_trainer is not None:
            prediction = await self._predict_lstm(features)
        else:

            prediction = self._predict_ridge(features)

        theoretical_max_kwh = None
        if isinstance(features, list) and hasattr(self.weights, "feature_names"):
            try:
                idx = self.weights.feature_names.index("theoretical_max_kwh")
                if idx < len(features):

                    theoretical_max_scaled = features[idx]

                    feature_mean = self.weights.feature_means.get("theoretical_max_kwh", 0.0)
                    feature_std = self.weights.feature_stds.get("theoretical_max_kwh", 1.0)
                    theoretical_max_kwh = (theoretical_max_scaled * feature_std) + feature_mean
                    # Reduced logging - only log at TRACE level (commented out to reduce log spam)
            except (ValueError, AttributeError, KeyError) as e:
                _LOGGER.debug(f"Could not extract theoretical_max from features: {e}")

        effective_max = self.max_hourly_kwh

        if theoretical_max_kwh is not None and theoretical_max_kwh > 0:

            theoretical_max_with_margin = theoretical_max_kwh * 1.2
            effective_max = min(self.max_hourly_kwh, theoretical_max_with_margin)

            # Physics cap is applied silently - reduces log spam

        prediction_before_cap = prediction
        prediction = max(0.0, min(prediction, effective_max))

        # Capping applied silently - reduces log spam

        confidence = self._calculate_confidence(features)

        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            method=f"ml_model_{self.algorithm_used}",
            features_used=features,
            model_accuracy=self.current_accuracy,
        )

    def _predict_ridge(self, features) -> float:
        """Ridge regression prediction: bias + sum(weight_i * feature_i) @zara"""
        prediction = self.weights.bias

        if isinstance(features, list):
            if not hasattr(self.weights, "feature_names") or not self.weights.feature_names:
                raise MLModelException(
                    "Cannot predict with list features: feature_names not available in weights"
                )

            for i, feature_value in enumerate(features):
                if i >= len(self.weights.feature_names):
                    _LOGGER.warning(f"Feature index {i} exceeds feature_names length, skipping")
                    continue

                feature_name = self.weights.feature_names[i]
                weight = self.weights.weights.get(feature_name, 0.0)
                contribution = weight * feature_value
                prediction += contribution

        elif isinstance(features, dict):
            for feature_name, feature_value in features.items():
                weight = self.weights.weights.get(feature_name, 0.0)
                prediction += weight * feature_value

        else:
            raise MLModelException(
                f"Unsupported features type: {type(features)}. Expected dict or list."
            )

        return prediction

    async def _predict_lstm(self, features) -> float:
        """TinyLSTM neural network prediction @zara"""
        if self._lstm_trainer is None:
            raise MLModelException("LSTM trainer not initialized")

        try:
            import numpy as np

            sequence = None
            sequence_source = "unknown"

            if (self._historical_sequence and
                self._sequence_builder and
                len(self._historical_sequence) >= self._lstm_trainer.sequence_length):

                try:

                    feature_names = getattr(self.weights, 'feature_names', None)
                    if feature_names:
                        real_sequence = self._sequence_builder.create_sequences_for_inference(
                            recent_hours=self._historical_sequence,
                            feature_names=feature_names
                        )
                        if real_sequence is not None:
                            # Use directly - TinyLSTM.forward() expects (seq_len, features), not (batch, seq_len, features)
                            sequence = real_sequence  # Shape: (24, 14)
                            sequence_source = "real_24h_history"
                            _LOGGER.debug(
                                f"LSTM using real 24h sequence from {len(self._historical_sequence)} historical hours"
                            )
                except Exception as seq_err:
                    _LOGGER.warning(f"Failed to build real sequence: {seq_err}")
                    sequence = None

            if sequence is None:
                if isinstance(features, list):
                    feature_array = np.array(features, dtype=np.float32)
                elif isinstance(features, dict):

                    feature_array = np.array([
                        features.get(name, 0.0) for name in self.weights.feature_names
                    ], dtype=np.float32)
                else:
                    raise MLModelException(f"Unsupported features type: {type(features)}")

                sequence_length = self._lstm_trainer.sequence_length
                # Create (seq_len, features) directly - NO batch dimension
                sequence = np.tile(feature_array, (sequence_length, 1))  # Shape: (24, 14)
                sequence_source = "pseudo_sequence"

                if self._historical_sequence:
                    _LOGGER.warning(
                        f"LSTM fallback to pseudo-sequence: historical data available "
                        f"({len(self._historical_sequence)} hours) but sequence build failed"
                    )
                else:
                    _LOGGER.debug("LSTM using pseudo-sequence (no historical data available)")

            prediction = self._lstm_trainer.predict(sequence)

            _LOGGER.debug(f"LSTM prediction ({sequence_source}): {prediction:.4f} kWh")
            return float(prediction)

        except Exception as e:
            _LOGGER.error(f"LSTM prediction failed: {e}. Falling back to Ridge.")

            return self._predict_ridge(features)

    def is_available(self) -> bool:
        return self.weights is not None

    def _calculate_confidence(self, features) -> float:
        """Calculate confidence from features (dict or list) @zara"""
        if not self.current_accuracy:
            return 0.5

        base_confidence = self.current_accuracy

        production_yesterday = 0.0
        if isinstance(features, dict):
            production_yesterday = features.get("production_yesterday", 0.0)
        elif isinstance(features, list) and hasattr(self.weights, "feature_names"):

            try:
                idx = self.weights.feature_names.index("production_yesterday")
                if idx < len(features):
                    production_yesterday = features[idx]
            except (ValueError, AttributeError):
                pass

        if production_yesterday > 0:
            base_confidence *= 1.1

        weather_stability = 0.5
        if isinstance(features, dict):
            weather_stability = features.get("weather_stability", 0.5)
        elif isinstance(features, list) and hasattr(self.weights, "feature_names"):
            try:
                idx = self.weights.feature_names.index("weather_stability")
                if idx < len(features):
                    weather_stability = features[idx]
            except (ValueError, AttributeError):
                pass

        base_confidence *= 0.8 + 0.2 * weather_stability

        return min(1.0, max(0.0, base_confidence))

class ProfileStrategy(PredictionStrategy):
    """Profile-based prediction strategy using historical hourly averages"""

    def __init__(self, profile: HourlyProfile, peak_power_kw: float = 0.0):
        self.profile = profile
        self.peak_power_kw = float(peak_power_kw) if peak_power_kw else 0.0

        from ..const import DEFAULT_MAX_HOURLY_KWH, HOURLY_PRODUCTION_SAFETY_MARGIN

        if self.peak_power_kw > 0:

            self.max_hourly_kwh = self.peak_power_kw * HOURLY_PRODUCTION_SAFETY_MARGIN
        else:

            self.max_hourly_kwh = DEFAULT_MAX_HOURLY_KWH * 2.0
            _LOGGER.warning(
                f"ProfileStrategy: peak_power_kw not configured, using default {self.max_hourly_kwh} kWh"
            )

    async def predict(self, features: Dict[str, float]) -> PredictionResult:
        if not self.profile:
            raise MLModelException("No hourly profile available")

        hour = int(features.get("hour_of_day", 12))
        base_prediction = self.profile.hourly_averages.get(str(hour), 0.0)

        cloudiness = features.get("cloudiness", 50.0)
        cloud_factor = (100 - cloudiness) / 100.0

        seasonal_factor = features.get("seasonal_factor", 0.5)

        adjusted_prediction = base_prediction * cloud_factor * (0.5 + seasonal_factor)

        prediction = max(0.0, min(adjusted_prediction, self.max_hourly_kwh))

        confidence = 0.6

        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            method="hourly_profile",
            features_used=features,
            model_accuracy=None,
        )

    def is_available(self) -> bool:
        return self.profile is not None and self.profile.samples_count > 10

class FallbackStrategy(PredictionStrategy):
    """WARNING FIX 5 Simple fallback strategy using solar capacity"""

    def __init__(self, peak_power_kw: float = 0.0):
        """Initialize fallback strategy with system capacity @zara"""
        from ..const import DEFAULT_MAX_HOURLY_KWH, HOURLY_PRODUCTION_SAFETY_MARGIN

        if peak_power_kw > 0:

            self.base_peak_kwh = peak_power_kw * HOURLY_PRODUCTION_SAFETY_MARGIN
        else:

            self.base_peak_kwh = DEFAULT_MAX_HOURLY_KWH
            _LOGGER.warning(
                f"FallbackStrategy: solar_capacity not configured, using default {self.base_peak_kwh} kWh"
            )

    async def predict(self, features) -> PredictionResult:
        """Make prediction from features (dict or list) @zara"""

        if isinstance(features, dict):
            hour_of_day = features.get("hour_of_day", 12.0)
            cloudiness = features.get("cloudiness", 50.0)
            seasonal_factor = features.get("seasonal_factor", 0.5)
            features_dict = features
        else:

            hour_of_day = 12.0
            cloudiness = 50.0
            seasonal_factor = 0.5
            features_dict = {}

        if hour_of_day < 6 or hour_of_day > 20:
            prediction = 0.0
        else:
            hour_factor = math.sin((hour_of_day - 6) * math.pi / 14)
            cloud_factor = (100 - cloudiness) / 100.0

            prediction = self.base_peak_kwh * hour_factor * cloud_factor * seasonal_factor

        prediction = max(0.0, prediction)

        return PredictionResult(
            prediction=prediction,
            confidence=0.3,
            method="simple_fallback",
            features_used=features_dict,
            model_accuracy=None,
        )

    def is_available(self) -> bool:
        return True

class PredictionOrchestrator:

    def __init__(self):
        self.strategies: list[PredictionStrategy] = []
        self.peak_power_kw: float = 0.0
        self._historical_hours: Optional[list] = None

    def register_strategy(self, strategy: PredictionStrategy) -> None:
        self.strategies.append(strategy)

    def set_historical_data(self, recent_hours: Optional[list]) -> None:
        """Set historical data for LSTM sequence building @zara"""
        self._historical_hours = recent_hours

        for strategy in self.strategies:
            if isinstance(strategy, MLModelStrategy):
                strategy.set_historical_sequence(recent_hours)

    async def predict(self, features: Dict[str, float]) -> PredictionResult:
        if self._historical_hours:
            for strategy in self.strategies:
                if isinstance(strategy, MLModelStrategy):
                    strategy.set_historical_sequence(self._historical_hours)

        for strategy in self.strategies:
            if strategy.is_available():
                try:
                    return await strategy.predict(features)
                except Exception as e:
                    _LOGGER.warning(f"Strategy {strategy.__class__.__name__} failed: {e}")
                    continue

        fallback = FallbackStrategy(self.peak_power_kw)
        return await fallback.predict(features)

    def update_strategies(
        self,
        weights: Optional[LearnedWeights] = None,
        profile: Optional[HourlyProfile] = None,
        accuracy: float = 0.0,
        peak_power_kw: float = 0.0,
    ) -> None:
        """Update available prediction strategies with new data"""
        self.strategies.clear()
        self.peak_power_kw = peak_power_kw

        if weights:
            ml_strategy = MLModelStrategy(weights, accuracy, peak_power_kw)
            if self._historical_hours:
                ml_strategy.set_historical_sequence(self._historical_hours)
            self.strategies.append(ml_strategy)

        if profile:

            self.strategies.append(ProfileStrategy(profile, peak_power_kw))

        self.strategies.append(FallbackStrategy(peak_power_kw))
