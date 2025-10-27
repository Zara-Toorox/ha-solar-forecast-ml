"""
Training Logic fÃ¼r ML Model mit Ridge Regression.

Copyright (C) 2025 Zara-Toorox

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""
import logging
from typing import Dict, List, Tuple
from .exceptions import MLModelException

_LOGGER = logging.getLogger(__name__)


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


class RidgeTrainer:
    
    def __init__(self):
        self.best_lambda = 0.1
    
    async def train(
        self, 
        X_train: List[List[float]], 
        y_train: List[float]
    ) -> Tuple[Dict[str, float], float, float, float]:
        try:
            np = _ensure_numpy()
            
            X_array = np.array(X_train)
            y_array = np.array(y_train)
            
            X_with_bias = np.column_stack([X_array, np.ones(len(X_array))])
            
            lambda_candidates = [0.001, 0.01, 0.1, 1.0, 10.0]
            best_lambda = 0.1
            best_score = -np.inf
            
            split_idx = int(len(X_train) * 0.8)
            X_train_split = X_with_bias[:split_idx]
            y_train_split = y_array[:split_idx]
            X_test = X_with_bias[split_idx:]
            y_test = y_array[split_idx:]
            
            if len(X_test) > 5:
                for lambda_val in lambda_candidates:
                    XtX = np.dot(X_train_split.T, X_train_split)
                    Xty = np.dot(X_train_split.T, y_train_split)
                    
                    regularization = lambda_val * np.eye(XtX.shape[0])
                    XtX_reg = XtX + regularization
                    
                    try:
                        weights_with_bias = np.linalg.solve(XtX_reg, Xty)
                        
                        predictions = np.dot(X_test, weights_with_bias)
                        ss_res = np.sum((y_test - predictions) ** 2)
                        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                        
                        if ss_tot > 0:
                            r_squared = 1 - (ss_res / ss_tot)
                            if r_squared > best_score:
                                best_score = r_squared
                                best_lambda = lambda_val
                    except np.linalg.LinAlgError:
                        continue
            
            XtX = np.dot(X_with_bias.T, X_with_bias)
            Xty = np.dot(X_with_bias.T, y_array)
            
            regularization = best_lambda * np.eye(XtX.shape[0])
            XtX_reg = XtX + regularization
            
            weights_with_bias = np.linalg.solve(XtX_reg, Xty)
            
            feature_weights = weights_with_bias[:-1]
            bias = weights_with_bias[-1]
            
            weights_dict = {}
            for i, weight in enumerate(feature_weights):
                weights_dict[f"feature_{i}"] = float(weight)
            
            predictions = np.dot(X_with_bias, weights_with_bias)
            ss_res = np.sum((y_array - predictions) ** 2)
            ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
            
            _LOGGER.info(
                f"Ridge Regression: ss_res={ss_res:.2f}, ss_tot={ss_tot:.2f}, "
                f"mean(y)={np.mean(y_array):.2f}"
            )
            
            if ss_tot > 1e-8:
                r_squared = 1 - (ss_res / ss_tot)
                accuracy = max(0.0, min(1.0, r_squared))
                _LOGGER.info(f"RÂ²={r_squared:.4f}, accuracy={accuracy:.4f}")
            else:
                accuracy = 0.0
                _LOGGER.warning("ss_tot near zero - all target values identical?")
            
            if accuracy == 0.0:
                _LOGGER.warning(
                    "CRITICAL: accuracy=0.0 nach Training! "
                    f"Samples={len(y_train)}, y_range=[{np.min(y_array):.2f}, {np.max(y_array):.2f}]"
                )
            
            self.best_lambda = best_lambda
            _LOGGER.info(f"Ridge Training: Lambda={best_lambda}, RÂ²={accuracy:.4f}")
            
            return weights_dict, float(bias), accuracy, best_lambda
            
        except Exception as e:
            _LOGGER.error(f"Ridge model training failed: {e}")
            raise MLModelException(f"Ridge regression training failed: {e}")
    
    def map_weights_to_features(
        self, 
        weights_dict: Dict[str, float], 
        feature_names: List[str]
    ) -> Dict[str, float]:
        mapped_weights = {}
        for i, feature_name in enumerate(feature_names):
            key = f"feature_{i}"
            if key in weights_dict:
                mapped_weights[feature_name] = weights_dict[key]
        return mapped_weights
