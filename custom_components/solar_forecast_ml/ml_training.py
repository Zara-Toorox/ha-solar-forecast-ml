"""
Training Logic for ML Model with Ridge Regression.

Copyright (C) 2025 Zara-Toorox

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
from .exceptions import MLModelException

_LOGGER = logging.getLogger(__name__)

# Lazy import for NumPy
_np: Optional[Any] = None

def _ensure_numpy() -> Any:
    """
    Lazily imports and returns the NumPy module.
    Raises ImportError if NumPy is not installed.
    """
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
        except ImportError as e:
            raise ImportError(f"NumPy could not be imported: {e}")
    return _np


class RidgeTrainer:
    """
    Handles the training of a Ridge Regression model, including
    hyperparameter (lambda) tuning via a simple validation split.
    """
    
    def __init__(self):
        """Initializes the trainer with a default lambda."""
        self.best_lambda: float = 0.1
    
    def train(
        self, 
        X_train: List[List[float]], 
        y_train: List[float]
    ) -> Tuple[Dict[str, float], float, float, float]:
        """
        Trains the Ridge Regression model on the provided data.

        Args:
            X_train: List of feature vectors (training data).
            y_train: List of target values.

        Returns:
            A tuple containing:
            - (Dict[str, float]): Dictionary of feature weights (e.g., {"feature_0": 0.5, ...}).
            - (float): The calculated bias (intercept) term.
            - (float): The R-squared score on the full training set (clamped between 0.0 and 1.0).
            - (float): The best lambda value found during validation.
            
        Raises:
            MLModelException: If training fails for any reason.
        """
        try:
            np = _ensure_numpy()
            
            X_array = np.array(X_train)
            y_array = np.array(y_train)
            
            # Add a bias (intercept) column (vector of ones)
            X_with_bias = np.column_stack([X_array, np.ones(len(X_array))])
            
            # --- Hyperparameter Tuning (Find best lambda) ---
            lambda_candidates = [0.001, 0.01, 0.1, 1.0, 10.0]
            best_lambda = 0.1  # Default fallback
            best_score = -np.inf
            
            # Create an 80/20 train/validation split from the training data
            split_idx = int(len(X_train) * 0.8)
            X_train_split = X_with_bias[:split_idx]
            y_train_split = y_array[:split_idx]
            X_val_split = X_with_bias[split_idx:]
            y_val_split = y_array[split_idx:]
            
            # Only perform validation if the validation set is large enough
            if len(X_val_split) > 5:
                for lambda_val in lambda_candidates:
                    try:
                        # Solve (XtX + lambda*I) * w = Xty
                        XtX = np.dot(X_train_split.T, X_train_split)
                        Xty = np.dot(X_train_split.T, y_train_split)
                        
                        regularization = lambda_val * np.eye(XtX.shape[0])
                        XtX_reg = XtX + regularization
                        
                        weights_with_bias = np.linalg.solve(XtX_reg, Xty)
                        
                        # Evaluate on validation set
                        predictions = np.dot(X_val_split, weights_with_bias)
                        ss_res = np.sum((y_val_split - predictions) ** 2)
                        ss_tot = np.sum((y_val_split - np.mean(y_val_split)) ** 2)
                        
                        if ss_tot > 1e-8:
                            r_squared = 1 - (ss_res / ss_tot)
                            if r_squared > best_score:
                                best_score = r_squared
                                best_lambda = lambda_val
                    except np.linalg.LinAlgError:
                        # Matrix might be singular, skip this lambda
                        continue
            
            # --- Final Training (on all data) ---
            # Now train on the full dataset using the best lambda
            XtX_full = np.dot(X_with_bias.T, X_with_bias)
            Xty_full = np.dot(X_with_bias.T, y_array)
            
            final_regularization = best_lambda * np.eye(XtX_full.shape[0])
            XtX_full_reg = XtX_full + final_regularization
            
            final_weights_with_bias = np.linalg.solve(XtX_full_reg, Xty_full)
            
            # Separate features weights from the bias term
            feature_weights = final_weights_with_bias[:-1]
            bias = final_weights_with_bias[-1]
            
            weights_dict = {}
            for i, weight in enumerate(feature_weights):
                weights_dict[f"feature_{i}"] = float(weight)
            
            # --- Calculate final accuracy (R-squared) on the *entire* training set ---
            predictions_full = np.dot(X_with_bias, final_weights_with_bias)
            ss_res_full = np.sum((y_array - predictions_full) ** 2)
            ss_tot_full = np.sum((y_array - np.mean(y_array)) ** 2)
            
            _LOGGER.info(
                f"Ridge Regression: ss_res={ss_res_full:.2f}, ss_tot={ss_tot_full:.2f}, "
                f"mean(y)={np.mean(y_array):.2f}"
            )
            
            accuracy = 0.0
            if ss_tot_full > 1e-8:
                r_squared_full = 1 - (ss_res_full / ss_tot_full)
                # Clamp R-squared between 0 and 1
                accuracy = max(0.0, min(1.0, r_squared_full))
                _LOGGER.info(f"Training R-squared={r_squared_full:.4f}, accuracy={accuracy:.4f}")
            else:
                _LOGGER.warning("ss_tot near zero - all target values identical?")
            
            if accuracy == 0.0:
                _LOGGER.warning(
                    "CRITICAL: accuracy=0.0 after training! "
                    f"Samples={len(y_train)}, y_range=[{np.min(y_array):.2f}, {np.max(y_array):.2f}]"
                )
            
            self.best_lambda = best_lambda
            _LOGGER.info(f"Ridge Training complete: Best Lambda={best_lambda}, Training R-squared={accuracy:.4f}")
            
            return weights_dict, float(bias), accuracy, best_lambda
            
        except Exception as e:
            _LOGGER.error(f"Ridge model training failed: {e}", exc_info=True)
            raise MLModelException(f"Ridge regression training failed: {e}")
    
    def map_weights_to_features(
        self, 
        weights_dict: Dict[str, float], 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Maps the generic 'feature_i' weights to actual feature names.

        Args:
            weights_dict: A dictionary from 'train' (e.g., {"feature_0": 0.5}).
            feature_names: A list of feature names in the correct order.

        Returns:
            A dictionary mapping feature names to their weights.
        """
        mapped_weights = {}
        for i, feature_name in enumerate(feature_names):
            key = f"feature_{i}"
            if key in weights_dict:
                mapped_weights[feature_name] = weights_dict[key]
        return mapped_weights