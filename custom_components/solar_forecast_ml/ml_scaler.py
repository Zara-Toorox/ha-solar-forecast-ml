"""
StandardScaler fÃ¼r Feature Normalisierung.

Copyright (C) 2025 Zara-Toorox

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""
from typing import Dict, Any, List


def _ensure_numpy():
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
        except ImportError as e:
            raise ImportError(f"NumPy konnte nicht importiert werden: {e}")
    return _np


_np = None


class StandardScaler:
    
    def __init__(self):
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.is_fitted = False
    
    def fit(self, X: List[List[float]], feature_names: List[str]) -> None:
        np = _ensure_numpy()
        X_array = np.array(X)
        
        self.means = {}
        self.stds = {}
        
        for i, name in enumerate(feature_names):
            col = X_array[:, i]
            self.means[name] = float(np.mean(col))
            std = float(np.std(col))
            self.stds[name] = std if std > 1e-8 else 1.0
        
        self.is_fitted = True
    
    def transform(self, X: List[List[float]], feature_names: List[str]) -> List[List[float]]:
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        np = _ensure_numpy()
        X_array = np.array(X)
        X_scaled = np.zeros_like(X_array)
        
        for i, name in enumerate(feature_names):
            mean = self.means.get(name, 0.0)
            std = self.stds.get(name, 1.0)
            X_scaled[:, i] = (X_array[:, i] - mean) / std
        
        return X_scaled.tolist()
    
    def fit_transform(self, X: List[List[float]], feature_names: List[str]) -> List[List[float]]:
        self.fit(X, feature_names)
        return self.transform(X, feature_names)
    
    def transform_single(self, features: Dict[str, float]) -> Dict[str, float]:
        if not self.is_fitted:
            return features
        
        scaled = {}
        for name, value in features.items():
            mean = self.means.get(name, 0.0)
            std = self.stds.get(name, 1.0)
            scaled[name] = (value - mean) / std
        
        return scaled
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'means': self.means,
            'stds': self.stds,
            'is_fitted': self.is_fitted
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        self.means = state.get('means', {})
        self.stds = state.get('stds', {})
        self.is_fitted = state.get('is_fitted', False)
