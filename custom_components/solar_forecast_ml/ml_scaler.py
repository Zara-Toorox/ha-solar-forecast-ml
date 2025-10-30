"""
StandardScaler for Feature Normalization in Solar Forecast ML.
Calculates mean and standard deviation for scaling features.

Copyright (C) 2025 Zara-Toorox

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple

# Lazy import NumPy
_np: Optional[Any] = None

_LOGGER = logging.getLogger(__name__)

def _ensure_numpy() -> Any:
    """Lazily imports and returns the NumPy module, raising ImportError if unavailable."""
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
            # Log only once on successful import
            # _LOGGER.info("NumPy library loaded successfully for StandardScaler.")
        except ImportError as e:
            _LOGGER.error("NumPy is required for StandardScaler but could not be imported. "
                          "Please ensure it is installed. Error: %s", e)
            raise ImportError(f"NumPy library is required but could not be imported: {e}") from e
    return _np


class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Calculates mean and standard deviation during `fit` and uses them for `transform`.
    Handles feature names for mapping means/stds correctly.
    """

    def __init__(self):
        """Initializes the StandardScaler."""
        self.means: Dict[str, float] = {}       # Stores mean for each feature name
        self.stds: Dict[str, float] = {}        # Stores standard deviation for each feature name
        self.is_fitted: bool = False            # Flag indicating if scaler has been fitted
        self._feature_names_order: List[str] = [] # Stores the order of features during fit

    def fit(self, X: List[List[float]], feature_names: List[str]) -> None:
        """
        Compute the mean and standard deviation to be used for later scaling.

        Args:
            X: List of feature vectors (samples x features).
            feature_names: List of names corresponding to the columns in X.

        Raises:
            ValueError: If input dimensions mismatch or data is invalid.
            ImportError: If NumPy is not available.
        """
        np = _ensure_numpy()
        _LOGGER.debug("Fitting StandardScaler with %d samples and %d features.", len(X), len(feature_names))

        if not X or not isinstance(X, list) or not isinstance(X[0], list):
             raise ValueError("Input X must be a non-empty list of lists (feature vectors).")
        if len(X[0]) != len(feature_names):
            raise ValueError(f"Number of features in X ({len(X[0])}) does not match "
                             f"number of feature names ({len(feature_names)}).")

        try:
            X_array = np.array(X, dtype=float) # Ensure float type
            if X_array.ndim != 2:
                 raise ValueError("Input X must be a 2D array-like structure.")

            self.means = {}
            self.stds = {}
            self._feature_names_order = feature_names # Store the order

            # Calculate mean and std deviation for each feature column
            for i, name in enumerate(feature_names):
                col = X_array[:, i]
                mean_val = float(np.mean(col))
                std_val = float(np.std(col))

                self.means[name] = mean_val
                # Use a small epsilon (or 1.0) for features with zero variance to avoid division by zero
                self.stds[name] = std_val if std_val > 1e-8 else 1.0

            self.is_fitted = True
            _LOGGER.info("StandardScaler fitted successfully for features: %s", feature_names)

        except Exception as e:
            _LOGGER.error("Error during StandardScaler fitting: %s", e, exc_info=True)
            self.is_fitted = False # Mark as not fitted on error
            # Re-raise or wrap in a custom exception if needed
            raise ValueError(f"Failed to fit StandardScaler: {e}") from e

    def transform(self, X: List[List[float]], feature_names: List[str]) -> List[List[float]]:
        """
        Perform standardization by centering and scaling.

        Args:
            X: List of feature vectors (samples x features) to transform.
            feature_names: List of names corresponding to the columns in X
                           (must match the order used during `fit`).

        Returns:
            List of transformed feature vectors.

        Raises:
            ValueError: If the scaler is not fitted, dimensions mismatch, or data is invalid.
            ImportError: If NumPy is not available.
        """
        np = _ensure_numpy()
        if not self.is_fitted:
            raise ValueError("StandardScaler must be fitted before calling transform.")
        if not X or not isinstance(X, list) or not isinstance(X[0], list):
             raise ValueError("Input X for transform must be a non-empty list of lists.")
        if len(X[0]) != len(feature_names):
            raise ValueError(f"Number of features in X ({len(X[0])}) does not match "
                             f"number of feature names ({len(feature_names)}).")
        # Ensure the feature names provided match those used during fit
        if feature_names != self._feature_names_order:
             # This check can be strict or relaxed depending on requirements
             _LOGGER.warning("Feature names or order in transform differ from fit. "
                           "Ensure columns correspond correctly.")
             # Reorder columns if needed? For now, assume order matches or is intended.


        try:
            X_array = np.array(X, dtype=float)
            if X_array.ndim != 2:
                 raise ValueError("Input X for transform must be 2D array-like.")

            X_scaled = np.zeros_like(X_array)

            # Apply scaling using stored means/stds based on feature names
            for i, name in enumerate(feature_names):
                mean = self.means.get(name)
                std = self.stds.get(name)

                if mean is None or std is None:
                     # This happens if transform is called with features not seen during fit
                     _LOGGER.warning(f"Feature '{name}' was not present during fitting. Leaving it unscaled.")
                     X_scaled[:, i] = X_array[:, i] # Keep original value
                elif std == 0: # Should be handled by epsilon during fit, but double-check
                     X_scaled[:, i] = 0.0 # Set to 0 if std dev was 0
                else:
                     X_scaled[:, i] = (X_array[:, i] - mean) / std

            return X_scaled.tolist() # Convert back to list of lists

        except Exception as e:
            _LOGGER.error("Error during StandardScaler transformation: %s", e, exc_info=True)
            raise ValueError(f"Failed to transform data: {e}") from e

    def fit_transform(self, X: List[List[float]], feature_names: List[str]) -> List[List[float]]:
        """
        Fit to data, then transform it. Equivalent to calling fit() then transform().
        """
        self.fit(X, feature_names)
        return self.transform(X, feature_names)

    def transform_single(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Transform a single sample provided as a dictionary (feature_name: value).

        Args:
            features: Dictionary representing a single sample.

        Returns:
            Dictionary with scaled feature values. Returns original dict if not fitted.
        """
        if not self.is_fitted:
            _LOGGER.debug("Scaler not fitted, returning original features for single transform.")
            return features # Return original if scaler hasn't been trained

        scaled_features: Dict[str, float] = {}
        for name, value in features.items():
            mean = self.means.get(name)
            std = self.stds.get(name)

            if mean is not None and std is not None:
                # Apply scaling if feature was seen during fit
                scaled_value = (value - mean) / std if std > 1e-8 else 0.0
                scaled_features[name] = scaled_value
            else:
                # Keep features not seen during fit unscaled
                _LOGGER.debug(f"Feature '{name}' not seen during fit, keeping original value in single transform.")
                scaled_features[name] = value

        return scaled_features

    def get_state(self) -> Dict[str, Any]:
        """
        Get the internal state of the scaler (means, stds, fitted status) for serialization.
        """
        return {
            'means': self.means,
            'stds': self.stds,
            'is_fitted': self.is_fitted,
            'feature_names_order': self._feature_names_order # Include feature order
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Load the internal state of the scaler from a dictionary (e.g., loaded from file).
        """
        _LOGGER.debug("Setting StandardScaler state.")
        try:
            self.means = state.get('means', {})
            self.stds = state.get('stds', {})
            self.is_fitted = state.get('is_fitted', False)
            self._feature_names_order = state.get('feature_names_order', []) # Load feature order

            # Basic validation after loading state
            if self.is_fitted and (not self.means or not self.stds):
                 _LOGGER.warning("Scaler state loaded as 'fitted' but means or stds are missing. Resetting to not fitted.")
                 self.is_fitted = False
                 self.means = {}
                 self.stds = {}
                 self._feature_names_order = []
            elif self.is_fitted:
                 _LOGGER.info("StandardScaler state loaded successfully (%d features).", len(self.means))

        except Exception as e:
            _LOGGER.error("Failed to set StandardScaler state: %s. Resetting state.", e, exc_info=True)
            # Reset to default state on error
            self.means = {}
            self.stds = {}
            self.is_fitted = False
            self._feature_names_order = []