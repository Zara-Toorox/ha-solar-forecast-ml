"""Adaptive Trainer - Selects best ML algorithm based on data & hardware V12.2.0 @zara

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

import asyncio
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

_psutil: Optional[Any] = None

def _ensure_psutil() -> Optional[Any]:
    """Try to import psutil (optional dependency) @zara"""
    global _psutil
    if _psutil is None:
        try:
            import psutil
            _psutil = psutil
            return psutil
        except ImportError:
            _LOGGER.debug("psutil not available - memory checks disabled")
            return None
    return _psutil

class AlgorithmType(Enum):
    """Available ML algorithms"""
    RIDGE = "ridge"
    TINY_LSTM = "tiny_lstm"
    AUTO = "auto"

class AdaptiveTrainer:
    """
    Selects and trains best ML algorithm based on available data and hardware.
    """

    def __init__(
        self,
        algorithm: str = "auto",
        enable_lstm: bool = True,
        min_samples_for_lstm: int = 100,
        min_memory_mb: int = 200
    ):
        """
        Initialize AdaptiveTrainer.

        Args:
            algorithm: "ridge", "tiny_lstm", or "auto"
            enable_lstm: Allow TinyLSTM (can be disabled by user)
            min_samples_for_lstm: Minimum samples to use LSTM
            min_memory_mb: Minimum free memory for LSTM
        """
        self.algorithm_choice = algorithm
        self.enable_lstm = enable_lstm
        self.min_samples_for_lstm = min_samples_for_lstm
        self.min_memory_mb = min_memory_mb

        from .ml_trainer import RidgeTrainer
        from .ml_tiny_lstm import TinyLSTM
        from .ml_sequence_builder import SequenceBuilder

        self.ridge_trainer = RidgeTrainer()
        self.lstm_trainer: Optional[TinyLSTM] = None
        self.sequence_builder = SequenceBuilder(sequence_length=24)

        _LOGGER.info(
            f"AdaptiveTrainer initialized: algorithm={algorithm}, "
            f"enable_lstm={enable_lstm}, min_samples={min_samples_for_lstm}"
        )

    def _check_system_resources(self) -> Dict[str, Any]:
        """Check available system resources. @zara"""
        psutil = _ensure_psutil()

        if psutil is None:

            return {
                "available_memory_mb": 500,
                "cpu_count": 1,
                "has_sufficient_memory": True
            }

        try:
            memory = psutil.virtual_memory()
            return {
                "available_memory_mb": memory.available / (1024 * 1024),
                "cpu_count": psutil.cpu_count() or 1,
                "has_sufficient_memory": (
                    memory.available / (1024 * 1024) >= self.min_memory_mb
                )
            }
        except Exception as e:
            _LOGGER.warning(f"Could not check system resources: {e}")
            return {
                "available_memory_mb": 0,
                "cpu_count": 1,
                "has_sufficient_memory": False
            }

    def select_algorithm(
        self,
        sample_count: int,
        force_algorithm: Optional[str] = None
    ) -> AlgorithmType:
        """
        Select best algorithm based on data & hardware.

        Decision Tree:
        1. User forced algorithm? → Use it
        2. Sample count < min_samples_for_lstm? → Ridge
        3. LSTM disabled by user? → Ridge
        4. Insufficient memory? → Ridge
        5. Otherwise → TinyLSTM

        Args:
            sample_count: Number of training samples available
            force_algorithm: Override automatic selection

        Returns:
            Selected algorithm type
        """

        if force_algorithm:
            try:
                algo = AlgorithmType(force_algorithm)
                _LOGGER.info(f"Algorithm forced by user: {force_algorithm}")
                return algo
            except ValueError:
                _LOGGER.warning(f"Invalid forced algorithm: {force_algorithm}, using auto")

        if sample_count < self.min_samples_for_lstm:
            _LOGGER.info(
                f"Using Ridge: Only {sample_count} samples "
                f"(need {self.min_samples_for_lstm} for LSTM)"
            )
            return AlgorithmType.RIDGE

        if not self.enable_lstm:
            _LOGGER.info("Using Ridge: TinyLSTM disabled by user")
            return AlgorithmType.RIDGE

        resources = self._check_system_resources()
        if not resources["has_sufficient_memory"]:
            _LOGGER.warning(
                f"Using Ridge: Insufficient memory "
                f"({resources['available_memory_mb']:.0f} MB available, "
                f"need {self.min_memory_mb} MB for LSTM)"
            )
            return AlgorithmType.RIDGE

        _LOGGER.info(
            f"Using TinyLSTM: {sample_count} samples, "
            f"{resources['available_memory_mb']:.0f} MB available"
        )
        return AlgorithmType.TINY_LSTM

    async def train(
        self,
        X_train: List[List[float]],
        y_train: List[float],
        feature_names: List[str],
        hourly_predictions: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, Any], float, str]:
        """
        Train using selected algorithm.

        Args:
            X_train: Training features (flat format for Ridge)
            y_train: Training targets
            feature_names: Feature names (14 features)
            hourly_predictions: Raw data (for LSTM sequence building)

        Returns:
            weights: Model weights (format depends on algorithm)
            accuracy: Model accuracy (R²)
            algorithm_used: Name of algorithm used
        """

        force_algorithm = None
        if self.algorithm_choice and self.algorithm_choice != "auto":
            force_algorithm = self.algorithm_choice
            _LOGGER.info(f"User requested algorithm: {force_algorithm}")

        algorithm = self.select_algorithm(len(X_train), force_algorithm=force_algorithm)

        try:
            if algorithm == AlgorithmType.TINY_LSTM:

                if hourly_predictions is None:
                    _LOGGER.warning("No hourly predictions for LSTM - falling back to Ridge")
                    raise ValueError("hourly_predictions required for LSTM")

                _LOGGER.info("Building sequences for TinyLSTM training...")
                X_sequences, y_targets = self.sequence_builder.build_sequences_from_predictions(
                    hourly_predictions, feature_names
                )

                if len(X_sequences) < 10:
                    _LOGGER.warning(
                        f"Not enough sequences ({len(X_sequences)}) - falling back to Ridge"
                    )
                    raise ValueError("Insufficient sequences for LSTM")

                stats = self.sequence_builder.get_sequence_stats(X_sequences, y_targets)
                _LOGGER.info(
                    f"Sequence stats: {stats['count']} sequences, "
                    f"target range [{stats['target_min']:.2f}, {stats['target_max']:.2f}]"
                )

                weights, accuracy = await self._train_lstm_async(X_sequences, y_targets)
                return weights, accuracy, "tiny_lstm"

            else:

                _LOGGER.info("Training Ridge Regression...")
                weights_dict, accuracy, _, _ = self.ridge_trainer.train(X_train, y_train)
                return weights_dict, accuracy, "ridge"

        except Exception as e:
            _LOGGER.error(f"Training failed with {algorithm.value}: {e}. Falling back to Ridge.")

            try:
                weights_dict, accuracy, _, _ = self.ridge_trainer.train(X_train, y_train)
                return weights_dict, accuracy, "ridge_fallback"
            except Exception as ridge_error:
                _LOGGER.error(f"Ridge fallback also failed: {ridge_error}")
                raise

    async def _train_lstm_async(
        self,
        X_sequences: List[Any],
        y_targets: List[float]
    ) -> Tuple[Dict[str, Any], float]:
        """
        Train TinyLSTM in thread pool (non-blocking).

        Uses asyncio.run_in_executor to avoid blocking HA event loop.
        """
        from .ml_tiny_lstm import TinyLSTM

        loop = asyncio.get_running_loop()

        def _train_sync():
            """Sync training function for executor @zara"""
            try:

                try:
                    os.nice(10)
                except (OSError, AttributeError):
                    pass

                if self.lstm_trainer is None:
                    self.lstm_trainer = TinyLSTM(
                        input_size=14,
                        hidden_size=32,
                        sequence_length=24,
                        learning_rate=0.001,
                        dropout=0.2
                    )

                import asyncio
                result = asyncio.run(
                    self.lstm_trainer.train(
                        X_sequences=X_sequences,
                        y_targets=y_targets,
                        epochs=100,
                        batch_size=16,
                        validation_split=0.2,
                        early_stopping_patience=10
                    )
                )

                return result

            except Exception as e:
                _LOGGER.error(f"LSTM training thread failed: {e}", exc_info=True)
                raise

        _LOGGER.info("Starting LSTM training in background thread...")
        result = await loop.run_in_executor(None, _train_sync)

        if not result.get('success', False):
            raise ValueError("LSTM training failed")

        accuracy = result.get('accuracy', 0.0)
        weights = self.lstm_trainer.get_weights()

        _LOGGER.info(f"LSTM training complete: accuracy={accuracy:.3f}")

        return weights, accuracy

    def predict(
        self,
        features: List[float],
        algorithm_used: str,
        weights: Dict[str, Any],
        recent_hours: Optional[List[Dict[str, Any]]] = None,
        feature_names: Optional[List[str]] = None
    ) -> float:
        """
        Predict using trained model.

        Args:
            features: Feature vector (14 features, flat)
            algorithm_used: Algorithm name ("ridge" or "tiny_lstm")
            weights: Model weights
            recent_hours: Last 24 hours (for LSTM sequence building)
            feature_names: Feature names (for LSTM)

        Returns:
            Prediction (kWh)
        """
        try:
            if algorithm_used == "tiny_lstm":

                if recent_hours is None or feature_names is None:
                    _LOGGER.warning("Cannot use LSTM for prediction - missing data")
                    raise ValueError("Missing data for LSTM prediction")

                sequence = self.sequence_builder.create_sequences_for_inference(
                    recent_hours, feature_names
                )

                if sequence is None:
                    raise ValueError("Failed to build inference sequence")

                from .ml_tiny_lstm import TinyLSTM

                if self.lstm_trainer is None:
                    self.lstm_trainer = TinyLSTM(
                        input_size=weights.get('input_size', 14),
                        hidden_size=weights.get('hidden_size', 32),
                        sequence_length=weights.get('sequence_length', 24)
                    )

                self.lstm_trainer.set_weights(weights)

                prediction = self.lstm_trainer.predict(sequence)
                return prediction

            else:

                import numpy as np

                X = np.array([features])

                X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])

                weight_vector = np.array([weights.get(str(i), 0.0) for i in range(len(features) + 1)])

                prediction = np.dot(X_with_bias, weight_vector)[0]

                return float(prediction)

        except Exception as e:
            _LOGGER.error(f"Prediction failed with {algorithm_used}: {e}")
            raise

    def get_algorithm_info(self, algorithm_used: str) -> Dict[str, Any]:
        """Get information about the algorithm used @zara"""
        if algorithm_used == "tiny_lstm":
            if self.lstm_trainer:
                return {
                    'algorithm': 'TinyLSTM',
                    'type': 'Recurrent Neural Network',
                    'hidden_size': self.lstm_trainer.hidden_size,
                    'sequence_length': self.lstm_trainer.sequence_length,
                    'parameters': self._count_lstm_parameters(),
                    'model_size_kb': self.lstm_trainer.get_model_size_kb()
                }
            else:
                return {'algorithm': 'TinyLSTM', 'type': 'Recurrent Neural Network'}
        else:
            return {
                'algorithm': 'Ridge Regression',
                'type': 'Linear Model',
                'regularization': 'L2'
            }

    def _count_lstm_parameters(self) -> int:
        """Count total parameters in LSTM @zara"""
        if self.lstm_trainer is None:
            return 0

        total = (
            self.lstm_trainer.Wf.size + self.lstm_trainer.Wi.size +
            self.lstm_trainer.Wc.size + self.lstm_trainer.Wo.size +
            self.lstm_trainer.bf.size + self.lstm_trainer.bi.size +
            self.lstm_trainer.bc.size + self.lstm_trainer.bo.size +
            self.lstm_trainer.Wy.size + self.lstm_trainer.by.size
        )
        return int(total)
