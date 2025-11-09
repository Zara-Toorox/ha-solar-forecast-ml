"""
ML Inference Engine for Solar Forecast ML Integration

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
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from ..ml.ml_types import LearnedWeights

_LOGGER = logging.getLogger(__name__)

# Lazy NumPy import
_np = None

def _ensure_numpy():
    """Lazily import NumPy"""
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
        except ImportError as e:
            _LOGGER.error("NumPy required but not available")
            raise
    return _np


class MLInferenceEngine:
    """Handles ML model inference and predictions"""
    
    def __init__(self, weights: Optional[LearnedWeights] = None):
        """Initialize inference engine"""
        self.weights = weights
    
    def set_weights(self, weights: LearnedWeights) -> None:
        """Set model weights for inference"""
        self.weights = weights
        _LOGGER.debug(f"Weights set with {len(weights.coefficients)} features")
    
    def predict(self, features: Dict[str, float]) -> Optional[float]:
        """Make prediction using loaded weights"""
        if not self.weights:
            _LOGGER.error("No weights loaded for inference")
            return None
        
        try:
            np = _ensure_numpy()
            
            # Create feature vector in correct order
            feature_vector = []
            for feature_name in self.weights.feature_names:
                if feature_name not in features:
                    _LOGGER.warning(f"Missing feature: {feature_name}, using 0.0")
                    feature_vector.append(0.0)
                else:
                    feature_vector.append(features[feature_name])
            
            # Convert to numpy array
            X = np.array(feature_vector).reshape(1, -1)
            
            # Make prediction: y = X @ coefficients + intercept
            coefficients = np.array(self.weights.coefficients)
            prediction = float(np.dot(X, coefficients) + self.weights.intercept)
            
            return prediction
            
        except ImportError:
            _LOGGER.error("NumPy not available for inference")
            return None
        except Exception as e:
            _LOGGER.error(f"Prediction failed: {e}", exc_info=True)
            return None
    
    def predict_batch(self, feature_list: List[Dict[str, float]]) -> List[Optional[float]]:
        """Make predictions for multiple feature sets"""
        return [self.predict(features) for features in feature_list]
    
    def calculate_confidence(
        self,
        features: Dict[str, float],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate confidence score for prediction"""
        if not self.weights or not self.weights.accuracy:
            return 0.5
        
        # Base confidence from model accuracy
        confidence = self.weights.accuracy
        
        # Adjust based on feature completeness
        if self.weights.feature_names:
            present_features = sum(1 for f in self.weights.feature_names if f in features)
            feature_completeness = present_features / len(self.weights.feature_names)
            confidence *= feature_completeness
        
        # Ensure confidence is in valid range
        return max(0.0, min(1.0, confidence))
    
    def is_ready(self) -> bool:
        """Check if inference engine is ready to make predictions"""
        return (
            self.weights is not None and
            len(self.weights.coefficients) > 0 and
            len(self.weights.feature_names) > 0
        )
