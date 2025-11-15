"""
Machine Learning Model Trainer

Ridge Regression training with adaptive alpha calculation and L2 regularization.
Uses NumPy matrix operations for efficient model training.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

# Import specific exception type
from ..core.core_exceptions import MLModelException
from ..core.core_helpers import SafeDateTimeUtil as dt_util

# Lazy import NumPy
_np: Optional[Any] = None

_LOGGER = logging.getLogger(__name__)


# --- Helper for Lazy NumPy Import ---
def _ensure_numpy() -> Any:
    """Lazily imports and returns the NumPy module raising ImportError if unavailable"""
    global _np
    if _np is None:
        try:
            import numpy as np

            _np = np
            # _LOGGER.info("NumPy library loaded successfully for RidgeTrainer.")
        except ImportError as e:
            _LOGGER.error(
                "NumPy is required for RidgeTrainer but could not be imported. "
                "Please ensure it is installed. Error: %s",
                e,
            )
            raise ImportError(f"NumPy library is required for RidgeTrainer: {e}") from e
    return _np


# --- Ridge Trainer Class ---
class RidgeTrainer:
    """Handles the training of a Ridge Regression model Linear Regression with L2 Re..."""

    def __init__(self, default_lambda: float = 0.1):
        """Initializes the RidgeTrainer"""
        self.best_lambda: float = default_lambda  # Stores the lambda found during tuning

    def train(
        self, X_train: List[List[float]], y_train: List[float]
    ) -> Tuple[Dict[str, float], float, float, float]:
        """Trains the Ridge Regression model using the provided training data"""
        _LOGGER.info(f"Starting Ridge Regression training with {len(y_train)} samples.")
        training_start_time = dt_util.now()  # Use LOCAL time for consistency

        try:
            np = _ensure_numpy()  # Ensure NumPy is loaded

            # --- Data Preparation ---
            if not X_train or not y_train or len(X_train) != len(y_train):
                raise ValueError(
                    "X_train and y_train must be non-empty and have the same number of samples."
                )
            if len(y_train) < 5:  # Need minimum samples for split and matrix operations
                raise ValueError(
                    f"Insufficient samples for training ({len(y_train)}), minimum 5 required."
                )

            X_array = np.array(X_train, dtype=float)
            y_array = np.array(y_train, dtype=float)

            # Add a bias column (intercept term) - column of ones
            X_with_bias = np.hstack([X_array, np.ones((X_array.shape[0], 1))])
            num_features_incl_bias = X_with_bias.shape[1]

            # --- Hyperparameter Tuning (Lambda Selection) ---
            # Adaptive alpha - More samples = Less regularization needed
            # Formula: alpha inversely proportional to sample count
            # Examples: 50 samples -> alpha=1.0 (high reg), 500 samples -> alpha=0.1 (low reg)
            adaptive_alpha = max(0.01, min(1.0, 100.0 / len(y_train)))
            _LOGGER.info(
                f"Using adaptive Ridge alpha={adaptive_alpha:.4f} for {len(y_train)} samples"
            )

            # Define candidate values for the regularization strength (lambda or alpha)
            # Build candidates around the adaptive alpha
            lambda_candidates = [
                adaptive_alpha * 0.01,
                adaptive_alpha * 0.1,
                adaptive_alpha,
                adaptive_alpha * 10.0,
                adaptive_alpha * 100.0,
                adaptive_alpha * 1000.0,
            ]
            best_lambda_found = adaptive_alpha  # Initialize with adaptive value
            best_validation_score = -np.inf  # Track best R-squared on validation set

            # Create an 80/20 train/validation split from the provided training data
            split_idx = int(len(X_train) * 0.8)
            X_train_internal = X_with_bias[:split_idx]
            y_train_internal = y_array[:split_idx]
            X_val_internal = X_with_bias[split_idx:]
            y_val_internal = y_array[split_idx:]

            _LOGGER.debug(
                f"Internal split: {len(y_train_internal)} training, {len(y_val_internal)} validation samples."
            )

            # Perform validation only if the validation set is large enough
            min_validation_samples = max(
                5, num_features_incl_bias
            )  # Need at least as many samples as features
            if len(X_val_internal) >= min_validation_samples:
                _LOGGER.debug(f"Performing lambda tuning with candidates: {lambda_candidates}")

                # HIGH PRIORITY FIX: Compute common matrices once outside loop (performance optimization)
                XtX = X_train_internal.T @ X_train_internal
                Xty = X_train_internal.T @ y_train_internal
                identity_matrix = np.eye(num_features_incl_bias)

                for lambda_val in lambda_candidates:
                    try:
                        # --- Solve Ridge Equation: (X_T * X + lambda * I) * w = X_T * y ---
                        # Regularization term (Identity matrix scaled by lambda)
                        regularization_term = lambda_val * identity_matrix

                        # Add regularization to the covariance matrix
                        XtX_regularized = XtX + regularization_term

                        # Solve the linear system for weights (including bias)
                        weights_with_bias = np.linalg.solve(XtX_regularized, Xty)

                        # --- Evaluate on Validation Set ---
                        predictions_val = X_val_internal @ weights_with_bias
                        # Calculate Residual Sum of Squares (RSS)
                        ss_res_val = np.sum((y_val_internal - predictions_val) ** 2)
                        # Calculate Total Sum of Squares (TSS)
                        ss_tot_val = np.sum((y_val_internal - np.mean(y_val_internal)) ** 2)

                        # Calculate R-squared score (Coefficient of Determination)
                        if (
                            ss_tot_val > 1e-8
                        ):  # Avoid division by zero if all validation targets are the same
                            r_squared_val = 1.0 - (ss_res_val / ss_tot_val)
                            _LOGGER.debug(
                                f"  Lambda={lambda_val:.3f}, Validation R2={r_squared_val:.4f}"
                            )
                            # Update best lambda if this score is better
                            if r_squared_val > best_validation_score:
                                best_validation_score = r_squared_val
                                best_lambda_found = lambda_val
                        else:
                            _LOGGER.debug(
                                f"  Lambda={lambda_val:.3f}, Validation TSS is near zero, skipping R2 calculation"
                            )

                    except np.linalg.LinAlgError:
                        # Matrix might be singular or ill-conditioned for this lambda
                        _LOGGER.warning(
                            f"Linear algebra error during validation for lambda={lambda_val}. Skipping."
                        )
                        continue  # Try next lambda value
            else:
                _LOGGER.debug(
                    f"Validation set too small ({len(X_val_internal)} samples), "
                    f"skipping lambda tuning. Using default lambda={best_lambda_found}."
                )

            # --- Final Training using Best Lambda on Full Dataset ---
            _LOGGER.info(
                f"Training final model on {len(y_array)} samples using lambda={best_lambda_found:.4f}..."
            )
            try:
                XtX_full = X_with_bias.T @ X_with_bias
                Xty_full = X_with_bias.T @ y_array

                final_regularization_term = best_lambda_found * np.eye(num_features_incl_bias)
                XtX_full_regularized = XtX_full + final_regularization_term

                # Solve for the final weights (including bias)
                final_weights_with_bias = np.linalg.solve(XtX_full_regularized, Xty_full)

                # Separate feature weights from the bias term (last element)
                feature_weights = final_weights_with_bias[:-1]
                bias = final_weights_with_bias[-1]

                # Create dictionary mapping generic feature index to weight
                weights_dict_generic = {
                    f"feature_{i}": float(w) for i, w in enumerate(feature_weights)
                }

                # --- Calculate Final Accuracy (R-squared) on the *Entire* Training Set ---
                predictions_full = X_with_bias @ final_weights_with_bias
                ss_res_full = np.sum((y_array - predictions_full) ** 2)
                ss_tot_full = np.sum((y_array - np.mean(y_array)) ** 2)

                _LOGGER.debug(
                    f"Final Model Evaluation: RSS={ss_res_full:.2f}, TSS={ss_tot_full:.2f}, "
                    f"Mean Target={np.mean(y_array):.2f}"
                )

                # Calculate R-squared, clamping between 0.0 and 1.0
                final_accuracy = 0.0
                if ss_tot_full > 1e-8:
                    r_squared_full = 1.0 - (ss_res_full / ss_tot_full)
                    # Clamp R-squared: Handle cases where model fits worse than mean (R < 0)
                    # or perfect fit issues (R slightly > 1 due to float precision)
                    final_accuracy = max(0.0, min(1.0, r_squared_full))
                    _LOGGER.debug(
                        f"Training R-squared (raw)={r_squared_full:.4f}, Clamped Accuracy={final_accuracy:.4f}"
                    )
                else:
                    _LOGGER.warning(
                        "Total Sum of Squares (TSS) is near zero, possibly constant target values. Accuracy set to 0.0."
                    )
                    # Check if predictions match the constant target
                    if np.allclose(predictions_full, y_array):
                        final_accuracy = 1.0  # Perfect fit for constant data
                        _LOGGER.debug(
                            "Target values are constant, and predictions match. Accuracy set to 1.0."
                        )

                # Log critical warning if accuracy is exactly 0 after training
                if final_accuracy == 0.0 and ss_tot_full > 1e-8:
                    _LOGGER.critical(
                        "Model training resulted in 0.0 accuracy! "
                        f"Check input data quality and feature relevance. Samples={len(y_train)}, "
                        f"Target Range=[{np.min(y_array):.2f}, {np.max(y_array):.2f}]"
                    )

                # Store the best lambda found
                self.best_lambda = best_lambda_found
                training_end_time = dt_util.now()
                duration = (training_end_time - training_start_time).total_seconds()

                _LOGGER.info(
                    f"Ridge Training completed in {duration:.2f}s. "
                    f"Best Lambda={best_lambda_found:.4f}, Final Training Accuracy={final_accuracy:.4f}"
                )

                return weights_dict_generic, float(bias), final_accuracy, best_lambda_found

            except np.linalg.LinAlgError as final_linalg_err:
                _LOGGER.error(
                    f"Linear algebra error during final model training: {final_linalg_err}",
                    exc_info=True,
                )
                raise MLModelException(
                    f"Ridge training failed during final solve: {final_linalg_err}"
                ) from final_linalg_err

        except ImportError:
            # Handle missing numpy import error specifically
            raise  # Re-raise ImportError, caught by caller
        except ValueError as ve:
            # Handle data validation errors
            _LOGGER.error(f"Invalid input data for training: {ve}", exc_info=True)
            raise MLModelException(f"Invalid data provided for Ridge training: {ve}") from ve
        except Exception as e:
            # Catch any other unexpected errors during the process
            _LOGGER.error(f"Ridge model training failed unexpectedly: {e}", exc_info=True)
            raise MLModelException(f"Unexpected error during Ridge training: {e}") from e

    def map_weights_to_features(
        self, weights_dict_generic: Dict[str, float], feature_names: List[str]
    ) -> Dict[str, float]:
        """Maps the generic feature_i weights from training to actual feature names"""
        if not weights_dict_generic or not feature_names:
            _LOGGER.warning("Cannot map weights: input dict or feature names list is empty.")
            return {}

        mapped_weights: Dict[str, float] = {}
        expected_feature_count = len(feature_names)

        for i, feature_name in enumerate(feature_names):
            generic_key = f"feature_{i}"
            weight = weights_dict_generic.get(generic_key)

            if weight is not None:
                mapped_weights[feature_name] = weight
            else:
                _LOGGER.warning(
                    f"Weight for '{generic_key}' ({feature_name}) not found in trained weights dict. Setting to 0.0."
                )
                mapped_weights[feature_name] = 0.0  # Assign 0 if missing

        # Sanity check: Ensure number of mapped weights matches expected feature count
        if len(mapped_weights) != expected_feature_count:
            _LOGGER.warning(
                f"Mismatch in mapped weights count ({len(mapped_weights)}) "
                f"and expected feature count ({expected_feature_count})."
            )

        return mapped_weights
