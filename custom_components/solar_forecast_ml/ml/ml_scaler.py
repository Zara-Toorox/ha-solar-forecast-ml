"""Data Scaling for ML Models V10.0.0 @zara

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
from typing import Any, Dict, List, Optional, Tuple

_np: Optional[Any] = None

_LOGGER = logging.getLogger(__name__)

def _ensure_numpy() -> Any:
    """Lazily imports and returns the NumPy module raising ImportError if unavailable @zara"""
    global _np
    if _np is None:
        try:
            import numpy as np

            _np = np

        except ImportError as e:
            _LOGGER.error(
                "NumPy is required for StandardScaler but could not be imported. "
                "Please ensure it is installed. Error: %s",
                e,
            )
            raise ImportError(f"NumPy library is required but could not be imported: {e}") from e
    return _np

class StandardScaler:
    """Standardizes features by removing the mean and scaling to unit variance"""

    def __init__(self):
        """Initializes the StandardScaler @zara"""
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.is_fitted: bool = False
        self._feature_names_order: List[str] = []

    def fit(self, X: List[List[float]], feature_names: List[str]) -> None:
        """Compute the mean and standard deviation to be used for later scaling @zara"""
        np = _ensure_numpy()
        _LOGGER.debug(
            "Fitting StandardScaler with %d samples and %d features.", len(X), len(feature_names)
        )

        if not X or not isinstance(X, list) or not isinstance(X[0], list):
            raise ValueError("Input X must be a non-empty list of lists (feature vectors).")
        if len(X[0]) != len(feature_names):
            raise ValueError(
                f"Number of features in X ({len(X[0])}) does not match "
                f"number of feature names ({len(feature_names)})."
            )

        try:
            X_array = np.array(X, dtype=float)
            if X_array.ndim != 2:
                raise ValueError("Input X must be a 2D array-like structure.")

            self.means = {}
            self.stds = {}
            self._feature_names_order = feature_names

            for i, name in enumerate(feature_names):
                col = X_array[:, i]
                mean_val = float(np.mean(col))
                std_val = float(np.std(col))

                if std_val < 0.01:
                    _LOGGER.debug(
                        f"Feature '{name}' has low/zero variance (std={std_val:.6f}). "
                        f"Using std=1.0 to maintain dimension consistency."
                    )
                    std_val = 1.0

                self.means[name] = mean_val
                self.stds[name] = std_val

            self.is_fitted = True
            _LOGGER.info("StandardScaler fitted successfully for features: %s", feature_names)

        except Exception as e:
            _LOGGER.error("Error during StandardScaler fitting: %s", e, exc_info=True)
            self.is_fitted = False

            raise ValueError(f"Failed to fit StandardScaler: {e}") from e

    def transform(self, X: List[List[float]], feature_names: List[str]) -> List[List[float]]:
        """Perform standardization by centering and scaling @zara"""
        np = _ensure_numpy()
        if not self.is_fitted:
            raise ValueError("StandardScaler must be fitted before calling transform.")
        if not X or not isinstance(X, list) or not isinstance(X[0], list):
            raise ValueError("Input X for transform must be a non-empty list of lists.")
        if len(X[0]) != len(feature_names):
            raise ValueError(
                f"Number of features in X ({len(X[0])}) does not match "
                f"number of feature names ({len(feature_names)})."
            )

        if feature_names != self._feature_names_order:

            _LOGGER.warning(
                "Feature names or order in transform differ from fit. "
                "Ensure columns correspond correctly."
            )

        try:
            X_array = np.array(X, dtype=float)
            if X_array.ndim != 2:
                raise ValueError("Input X for transform must be 2D array-like.")

            X_scaled = np.zeros_like(X_array)

            means_array = np.array([self.means.get(name, 0.0) for name in feature_names])
            stds_array = np.array([self.stds.get(name, 1.0) for name in feature_names])

            stds_array = np.where(stds_array < 1e-8, 1.0, stds_array)
            X_scaled = (X_array - means_array) / stds_array

            for i, name in enumerate(feature_names):
                if name not in self.means:
                    _LOGGER.warning(
                        f"Feature '{name}' was not present during fitting. Leaving it unscaled."
                    )
                    X_scaled[:, i] = X_array[:, i]

            return X_scaled.tolist()

        except Exception as e:
            _LOGGER.error("Error during StandardScaler transformation: %s", e, exc_info=True)
            raise ValueError(f"Failed to transform data: {e}") from e

    def fit_transform(self, X: List[List[float]], feature_names: List[str]) -> List[List[float]]:
        """Fit to data then transform it Equivalent to calling fit then transform @zara"""
        self.fit(X, feature_names)
        return self.transform(X, feature_names)

    def transform_single(self, features):
        """Transform a single sample provided as a dictionary or list @zara"""
        if not self.is_fitted:
            _LOGGER.debug("Scaler not fitted, returning original features for single transform.")
            return features

        if isinstance(features, list):
            if not self._feature_names_order:
                _LOGGER.warning(
                    "Cannot scale list-based features: feature_names_order not available"
                )
                return features

            scaled_features = []
            for i, value in enumerate(features):
                if i >= len(self._feature_names_order):
                    _LOGGER.warning(
                        f"Feature index {i} exceeds feature_names_order length, keeping original"
                    )
                    scaled_features.append(value)
                    continue

                feature_name = self._feature_names_order[i]
                mean = self.means.get(feature_name)
                std = self.stds.get(feature_name)

                if mean is not None and std is not None:
                    scaled_value = (value - mean) / std if std > 1e-8 else 0.0
                    scaled_features.append(scaled_value)
                else:

                    scaled_features.append(value)

            return scaled_features

        elif isinstance(features, dict):
            scaled_features: Dict[str, float] = {}
            for name, value in features.items():
                mean = self.means.get(name)
                std = self.stds.get(name)

                if mean is not None and std is not None:

                    scaled_value = (value - mean) / std if std > 1e-8 else 0.0
                    scaled_features[name] = scaled_value
                else:

                    _LOGGER.debug(
                        f"Feature '{name}' not seen during fit, keeping original value in single transform."
                    )
                    scaled_features[name] = value

            return scaled_features

        else:
            _LOGGER.error(f"Unsupported features type: {type(features)}. Expected dict or list.")
            return features

    def get_state(self) -> Dict[str, Any]:
        """Get the internal state of the scaler means stds fitted status for serialization @zara"""
        return {
            "means": self.means,
            "stds": self.stds,
            "is_fitted": self.is_fitted,
            "feature_names_order": self._feature_names_order,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load the internal state of the scaler from a dictionary eg loaded from file @zara"""
        _LOGGER.debug("Setting StandardScaler state.")
        try:
            self.means = state.get("means", {})
            self.stds = state.get("stds", {})
            self.is_fitted = state.get("is_fitted", False)
            self._feature_names_order = state.get("feature_names_order", [])

            if self.is_fitted and (not self.means or not self.stds):
                _LOGGER.warning(
                    "Scaler state loaded as 'fitted' but means or stds are missing. Resetting to not fitted."
                )
                self.is_fitted = False
                self.means = {}
                self.stds = {}
                self._feature_names_order = []
            elif self.is_fitted:
                _LOGGER.info(
                    "StandardScaler state loaded successfully (%d features).", len(self.means)
                )

        except Exception as e:
            _LOGGER.error(
                "Failed to set StandardScaler state: %s. Resetting state.", e, exc_info=True
            )

            self.means = {}
            self.stds = {}
            self.is_fitted = False
            self._feature_names_order = []
