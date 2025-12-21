"""ML Model Loader for Solar Forecast ML Integration V12.2.0 @zara

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
from pathlib import Path
from typing import Any, Dict, Optional

from ..ml.ml_types import LearnedWeights, create_default_learned_weights

_LOGGER = logging.getLogger(__name__)

class MLModelLoader:
    """Handles loading and initialization of ML models"""

    def __init__(self, data_manager):
        """Initialize model loader @zara"""
        self.data_manager = data_manager
        self._loaded_weights: Optional[LearnedWeights] = None

    async def load_weights(self) -> Optional[LearnedWeights]:
        """Load learned weights from storage @zara"""
        try:
            weights_data = await self.data_manager.load_learned_weights()

            if not weights_data or not weights_data.get("coefficients"):
                _LOGGER.info("No learned weights found, using defaults")
                return create_default_learned_weights()

            self._loaded_weights = LearnedWeights(**weights_data)
            _LOGGER.info(f"Loaded weights with {len(self._loaded_weights.coefficients)} features")

            return self._loaded_weights

        except Exception as e:
            _LOGGER.error(f"Failed to load weights: {e}", exc_info=True)
            return create_default_learned_weights()

    async def save_weights(self, weights: LearnedWeights) -> bool:
        """Save learned weights to storage @zara"""
        try:
            weights_dict = {
                "coefficients": weights.coefficients,
                "intercept": weights.intercept,
                "feature_names": weights.feature_names,
                "model_version": weights.model_version,
                "training_date": weights.training_date,
                "accuracy": weights.accuracy,
            }

            success = await self.data_manager.save_learned_weights(weights_dict)

            if success:
                self._loaded_weights = weights
                _LOGGER.info("Successfully saved learned weights")

            return success

        except Exception as e:
            _LOGGER.error(f"Failed to save weights: {e}", exc_info=True)
            return False

    async def load_model_state(self) -> Dict[str, Any]:
        """Load model state information @zara"""
        try:
            state = await self.data_manager.load_model_state()
            return state if state else {}
        except Exception as e:
            _LOGGER.error(f"Failed to load model state: {e}")
            return {}

    async def save_model_state(self, state: Dict[str, Any]) -> bool:
        """Save model state information @zara"""
        try:
            return await self.data_manager.save_model_state(state)
        except Exception as e:
            _LOGGER.error(f"Failed to save model state: {e}")
            return False

    def get_loaded_weights(self) -> Optional[LearnedWeights]:
        """Get currently loaded weights @zara"""
        return self._loaded_weights

    def is_model_loaded(self) -> bool:
        """Check if model is currently loaded @zara"""
        return self._loaded_weights is not None and len(self._loaded_weights.coefficients) > 0
