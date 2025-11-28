"""ML Residual Trainer - Train ML on residuals (actual - physics) V10.0.0 @zara

This is part of the Physics-First, ML-Enhanced architecture.

The key insight: Instead of training ML to predict absolute power,
we train it to predict the RESIDUAL (difference) between physics
prediction and actual production. This captures:
- Local effects not modeled by physics (shadows, obstructions)
- Systematic biases in the physics model
- Weather forecast errors
- Panel degradation over time

Final prediction = physics_prediction + ml_residual

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)


class ResidualTrainer:
    """
    Trains ML model on residuals (actual - physics prediction).

    This approach has several advantages:
    1. Physics handles deterministic components (sun position, geometry)
    2. ML learns non-deterministic components (shadows, errors, local effects)
    3. Better generalization - ML learns smaller corrections
    4. Faster convergence - residuals are smaller than absolute values
    """

    def __init__(
        self,
        data_dir: Path,
        system_capacity_kwp: float,
        skip_load: bool = False,
    ):
        """
        Initialize ResidualTrainer.

        Args:
            data_dir: Directory for data files
            system_capacity_kwp: System capacity in kWp
            skip_load: If True, skip loading state (use async_load_state later)
        """
        self.data_dir = data_dir
        self.system_capacity_kwp = system_capacity_kwp

        # State file for residual model
        self.state_file = data_dir / "ml" / "residual_model_state.json"

        # Physics engine for computing base predictions
        self._physics_engine = None
        self._geometry_learner = None

        # Residual statistics for normalization
        self.residual_stats = {
            "mean": 0.0,
            "std": 1.0,
            "min": -1.0,
            "max": 1.0,
            "sample_count": 0,
        }

        # Model weights for residual prediction
        self.residual_weights: Dict[str, Any] = {}
        self.model_type = "ridge"  # Can be "ridge" or "tiny_lstm"

        # Track if state is loaded
        self._state_loaded = False

        # Load existing state (skip if called from async context)
        if not skip_load:
            self._load_state()
            self._state_loaded = True

        _LOGGER.info(
            "ResidualTrainer initialized: %d samples, model=%s",
            self.residual_stats["sample_count"],
            self.model_type,
        )

    async def async_load_state(self) -> None:
        """Load state asynchronously - use this in async context @zara"""
        if self._state_loaded:
            return

        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_state_sync)
        self._state_loaded = True

    def _load_state(self) -> None:
        """Load saved state from file (sync version for __init__).

        Uses run_in_executor pattern to avoid blocking the event loop.
        Called during __init__ which may run in async context.
        """
        if not self.state_file.exists():
            return

        try:
            self._load_state_sync()
        except Exception as e:
            _LOGGER.warning("Failed to load residual model state: %s", e)

    def _load_state_sync(self) -> None:
        """Synchronous state loading - call from executor in async context @zara"""
        if not self.state_file.exists():
            return

        with open(self.state_file, "r") as f:
            data = json.load(f)

        self.residual_stats = data.get("residual_stats", self.residual_stats)
        self.residual_weights = data.get("weights", {})
        self.model_type = data.get("model_type", "ridge")

        _LOGGER.debug(
            "Loaded residual model: %d samples, type=%s",
            self.residual_stats.get("sample_count", 0),
            self.model_type,
        )

    async def save_state(self) -> None:
        """Save current state to file (async to avoid blocking event loop)."""
        try:
            data = {
                "version": "1.0",
                "residual_stats": self.residual_stats,
                "weights": self.residual_weights,
                "model_type": self.model_type,
                "saved_at": datetime.now().isoformat(),
                "system_capacity_kwp": self.system_capacity_kwp,
            }

            def _write_sync():
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.state_file, "w") as f:
                    json.dump(data, f, indent=2)

            import asyncio
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _write_sync)

            _LOGGER.debug("Saved residual model state")
        except Exception as e:
            _LOGGER.error("Failed to save residual model state: %s", e)

    def _get_physics_engine(self):
        """Get or create physics engine (sync version for compute methods).

        Note: This uses skip_load=True to avoid blocking. The state must be
        loaded separately via async_ensure_physics_engine() before calling
        any compute methods in async context.
        """
        if self._physics_engine is None:
            from ..physics import PhysicsEngine, GeometryLearner

            # Try to load geometry learner with learned geometry
            # Use skip_load=True to avoid blocking event loop
            try:
                self._geometry_learner = GeometryLearner(
                    data_path=self.data_dir,
                    system_capacity_kwp=self.system_capacity_kwp,
                    skip_load=True,  # Don't block event loop
                )
                # Note: State needs to be loaded via async_ensure_physics_engine()
                # For now, use default geometry - will be updated on async load
                self._physics_engine = self._geometry_learner.get_physics_engine()
            except Exception as e:
                _LOGGER.warning("Could not load geometry learner: %s", e)
                # Fallback to default geometry
                self._physics_engine = PhysicsEngine(
                    system_capacity_kwp=self.system_capacity_kwp,
                )

        return self._physics_engine

    async def async_ensure_physics_engine(self) -> None:
        """Ensure physics engine is loaded with state - use in async context @zara"""
        if self._physics_engine is None:
            from ..physics import PhysicsEngine, GeometryLearner

            try:
                self._geometry_learner = GeometryLearner(
                    data_path=self.data_dir,
                    system_capacity_kwp=self.system_capacity_kwp,
                    skip_load=True,  # Don't block event loop
                )
                await self._geometry_learner.async_load_state()
                self._physics_engine = self._geometry_learner.get_physics_engine()
                _LOGGER.debug(
                    "Physics engine loaded async: tilt=%.1f, azimuth=%.1f",
                    self._geometry_learner.estimate.tilt_deg,
                    self._geometry_learner.estimate.azimuth_deg,
                )
            except Exception as e:
                _LOGGER.warning("Could not load geometry learner async: %s", e)
                self._physics_engine = PhysicsEngine(
                    system_capacity_kwp=self.system_capacity_kwp,
                )
        elif self._geometry_learner is not None and not self._geometry_learner._state_loaded:
            # Physics engine exists but state not loaded
            await self._geometry_learner.async_load_state()
            self._physics_engine = self._geometry_learner.get_physics_engine()

    def compute_physics_prediction(
        self,
        weather_data: Dict[str, Any],
        astronomy_data: Dict[str, Any],
    ) -> float:
        """
        Compute physics-based prediction for a single hour.

        Args:
            weather_data: Weather data with ghi, direct_radiation, diffuse_radiation, temperature
            astronomy_data: Astronomy data with elevation_deg, azimuth_deg

        Returns:
            Physics prediction in kWh
        """
        engine = self._get_physics_engine()

        result = engine.calculate_hourly_forecast(
            weather_data=weather_data,
            astronomy_data=astronomy_data,
        )

        return result.get("physics_prediction_kwh", 0.0)

    def compute_residuals(
        self,
        training_records: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], float, float, float]]:
        """
        Compute residuals for training records.

        For each record, computes:
        - physics_prediction: What physics model predicts
        - actual: What was actually produced
        - residual: actual - physics_prediction

        Args:
            training_records: List of training records with weather and actual_kwh

        Returns:
            List of (record, physics_pred, actual, residual) tuples
        """
        results = []

        for record in training_records:
            try:
                # Extract data from record
                actual_kwh = record.get("actual_kwh")
                if actual_kwh is None or actual_kwh < 0:
                    continue

                # Get weather data
                weather = record.get("corrected_weather", record.get("weather", {}))
                ghi = weather.get("ghi", weather.get("solar_radiation_wm2", 0)) or 0
                dni = weather.get("direct_radiation", 0) or 0
                dhi = weather.get("diffuse_radiation", 0) or 0
                temp = weather.get("temperature", 15) or 15

                # Get astronomy data
                astro = record.get("astronomy", {})
                elevation = astro.get("elevation_deg", 0) or 0
                azimuth = astro.get("azimuth_deg", 180) or 180

                # Skip if no meaningful irradiance
                if ghi <= 0 and dni <= 0:
                    continue

                # Compute physics prediction
                physics_pred = self.compute_physics_prediction(
                    weather_data={
                        "ghi": ghi,
                        "direct_radiation": dni,
                        "diffuse_radiation": dhi,
                        "temperature": temp,
                    },
                    astronomy_data={
                        "elevation_deg": elevation,
                        "azimuth_deg": azimuth,
                    },
                )

                # Compute residual
                residual = actual_kwh - physics_pred

                results.append((record, physics_pred, actual_kwh, residual))

            except Exception as e:
                _LOGGER.debug("Failed to compute residual for record: %s", e)
                continue

        return results

    def update_residual_stats(
        self,
        residuals: List[float],
    ) -> None:
        """
        Update residual statistics for normalization.

        Args:
            residuals: List of residual values
        """
        if not residuals:
            return

        import numpy as np

        residuals_array = np.array(residuals)

        # Compute statistics
        self.residual_stats = {
            "mean": float(np.mean(residuals_array)),
            "std": float(np.std(residuals_array)) or 1.0,
            "min": float(np.min(residuals_array)),
            "max": float(np.max(residuals_array)),
            "sample_count": len(residuals),
        }

        _LOGGER.info(
            "Residual stats: mean=%.4f, std=%.4f, range=[%.4f, %.4f], n=%d",
            self.residual_stats["mean"],
            self.residual_stats["std"],
            self.residual_stats["min"],
            self.residual_stats["max"],
            self.residual_stats["sample_count"],
        )

    async def train_residual_model(
        self,
        training_records: List[Dict[str, Any]],
        algorithm: str = "auto",
    ) -> Tuple[bool, float, str]:
        """
        Train ML model on residuals.

        Args:
            training_records: Training records with actual_kwh and weather
            algorithm: "ridge", "tiny_lstm", or "auto"

        Returns:
            Tuple of (success, accuracy, algorithm_used)
        """
        try:
            # Ensure physics engine is loaded with state (async to avoid blocking)
            await self.async_ensure_physics_engine()

            # Compute residuals
            residual_data = self.compute_residuals(training_records)

            if len(residual_data) < 10:
                _LOGGER.warning(
                    "Not enough residual samples for training: %d < 10",
                    len(residual_data),
                )
                return False, 0.0, "none"

            # Extract residuals and update stats
            residuals = [r[3] for r in residual_data]
            self.update_residual_stats(residuals)

            # Log residual analysis
            physics_preds = [r[1] for r in residual_data]
            actuals = [r[2] for r in residual_data]

            import numpy as np
            physics_mae = np.mean(np.abs(np.array(actuals) - np.array(physics_preds)))
            _LOGGER.info(
                "Physics model MAE: %.4f kWh (this is what ML will try to improve)",
                physics_mae,
            )

            # Prepare features for residual training
            X_train = []
            y_train = []

            for record, physics_pred, actual, residual in residual_data:
                # Build feature vector for residual prediction
                features = self._build_residual_features(record, physics_pred)
                if features:
                    X_train.append(features)
                    y_train.append(residual)

            if len(X_train) < 10:
                _LOGGER.warning("Not enough valid feature vectors: %d", len(X_train))
                return False, 0.0, "none"

            # Train the model
            from .ml_adaptive_trainer import AdaptiveTrainer

            trainer = AdaptiveTrainer(
                algorithm=algorithm,
                enable_lstm=True,
                min_samples_for_lstm=100,
            )

            # For residual training, we use the feature names
            feature_names = self._get_residual_feature_names()

            # Train
            weights, accuracy, algo_used = await trainer.train(
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_names,
            )

            # Store weights
            self.residual_weights = weights
            self.model_type = algo_used

            # Save state
            await self.save_state()

            _LOGGER.info(
                "Residual model trained: algorithm=%s, accuracy=%.3f, samples=%d",
                algo_used,
                accuracy,
                len(X_train),
            )

            return True, accuracy, algo_used

        except Exception as e:
            _LOGGER.error("Residual training failed: %s", e, exc_info=True)
            return False, 0.0, "error"

    def _build_residual_features(
        self,
        record: Dict[str, Any],
        physics_pred: float,
    ) -> Optional[List[float]]:
        """
        Build feature vector for residual prediction.

        Features specifically designed to predict residuals:
        1. Physics prediction (the baseline)
        2. Time features (hour, month, season)
        3. Weather features (clouds, temperature, humidity)
        4. Astronomy features (elevation, azimuth)
        5. Historical residuals (if available)

        Args:
            record: Training record
            physics_pred: Physics prediction for this record

        Returns:
            Feature vector or None if invalid
        """
        try:
            # Extract data
            weather = record.get("corrected_weather", record.get("weather", {}))
            astro = record.get("astronomy", {})

            # Parse datetime
            timestamp = record.get("timestamp", record.get("datetime"))
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                dt = timestamp or datetime.now()

            # Feature 0: Physics prediction (normalized by capacity)
            physics_norm = physics_pred / max(self.system_capacity_kwp, 0.1)

            # Feature 1-2: Time features
            hour = dt.hour
            month = dt.month

            # Feature 3: Hour as sine (captures daily pattern)
            import math
            hour_sin = math.sin(2 * math.pi * hour / 24)

            # Feature 4: Hour as cosine
            hour_cos = math.cos(2 * math.pi * hour / 24)

            # Feature 5: Month as sine (captures seasonal pattern)
            month_sin = math.sin(2 * math.pi * month / 12)

            # Feature 6: Month as cosine
            month_cos = math.cos(2 * math.pi * month / 12)

            # Feature 7: Cloud cover (normalized)
            clouds = (weather.get("clouds", 50) or 50) / 100.0

            # Feature 8: Temperature deviation from 25°C (normalized)
            temp = weather.get("temperature", 15) or 15
            temp_dev = (temp - 25) / 30.0  # Normalize roughly to [-1, 1]

            # Feature 9: Humidity (normalized)
            humidity = (weather.get("humidity", 50) or 50) / 100.0

            # Feature 10: Sun elevation (normalized)
            elevation = (astro.get("elevation_deg", 0) or 0) / 90.0

            # Feature 11: Sun azimuth relative to South (normalized)
            azimuth = astro.get("azimuth_deg", 180) or 180
            azimuth_dev = (azimuth - 180) / 180.0  # 0 at South, +/- at East/West

            # Feature 12: GHI/theoretical ratio (captures cloud/obstruction effects)
            ghi = weather.get("ghi", weather.get("solar_radiation_wm2", 0)) or 0
            theoretical = astro.get("clear_sky_solar_radiation_wm2", 100) or 100
            ghi_ratio = min(ghi / max(theoretical, 1), 2.0)  # Cap at 2.0

            # Feature 13: DNI fraction (indicates beam vs diffuse)
            dni = weather.get("direct_radiation", 0) or 0
            dni_fraction = dni / max(ghi, 1) if ghi > 0 else 0.0

            features = [
                physics_norm,    # 0
                hour_sin,        # 1
                hour_cos,        # 2
                month_sin,       # 3
                month_cos,       # 4
                clouds,          # 5
                temp_dev,        # 6
                humidity,        # 7
                elevation,       # 8
                azimuth_dev,     # 9
                ghi_ratio,       # 10
                dni_fraction,    # 11
                hour / 24.0,     # 12 (raw hour, normalized)
                month / 12.0,    # 13 (raw month, normalized)
            ]

            return features

        except Exception as e:
            _LOGGER.debug("Failed to build residual features: %s", e)
            return None

    def _get_residual_feature_names(self) -> List[str]:
        """Get feature names for residual model."""
        return [
            "physics_norm",
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos",
            "clouds",
            "temp_dev",
            "humidity",
            "elevation",
            "azimuth_dev",
            "ghi_ratio",
            "dni_fraction",
            "hour_norm",
            "month_norm",
        ]

    def predict_residual(
        self,
        record: Dict[str, Any],
        physics_pred: float,
    ) -> float:
        """
        Predict residual for a given record.

        Args:
            record: Record with weather and astronomy data
            physics_pred: Physics prediction for this record

        Returns:
            Predicted residual (to be added to physics_pred)
        """
        if not self.residual_weights:
            # No trained model - return 0 (use physics only)
            return 0.0

        try:
            features = self._build_residual_features(record, physics_pred)
            if features is None:
                return 0.0

            from .ml_adaptive_trainer import AdaptiveTrainer

            trainer = AdaptiveTrainer()
            residual = trainer.predict(
                features=features,
                algorithm_used=self.model_type,
                weights=self.residual_weights,
            )

            # Clamp residual to reasonable range
            max_residual = self.system_capacity_kwp * 0.5  # Max 50% of capacity
            residual = max(-max_residual, min(max_residual, residual))

            return residual

        except Exception as e:
            _LOGGER.debug("Residual prediction failed: %s", e)
            return 0.0

    def predict_with_physics(
        self,
        weather_data: Dict[str, Any],
        astronomy_data: Dict[str, Any],
        record: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Predict power using physics + ML residual ensemble.

        This is the main prediction method that combines:
        1. Physics-based prediction (POA calculation)
        2. ML residual correction

        Args:
            weather_data: Weather data
            astronomy_data: Astronomy data
            record: Optional full record for additional features

        Returns:
            Dict with prediction details
        """
        # Physics prediction
        physics_pred = self.compute_physics_prediction(weather_data, astronomy_data)

        # ML residual
        if record is None:
            record = {"weather": weather_data, "astronomy": astronomy_data}

        ml_residual = self.predict_residual(record, physics_pred)

        # Ensemble prediction
        final_pred = max(0.0, physics_pred + ml_residual)

        # Compute weights based on confidence
        physics_weight = 1.0
        ml_weight = 0.0

        if self.residual_stats["sample_count"] >= 50:
            ml_weight = min(0.5, self.residual_stats["sample_count"] / 200.0)
            physics_weight = 1.0 - ml_weight

        weighted_pred = physics_weight * physics_pred + ml_weight * (physics_pred + ml_residual)
        weighted_pred = max(0.0, weighted_pred)

        return {
            "physics_prediction_kwh": round(physics_pred, 4),
            "ml_residual_kwh": round(ml_residual, 4),
            "raw_ensemble_kwh": round(final_pred, 4),
            "weighted_ensemble_kwh": round(weighted_pred, 4),
            "final_prediction_kwh": round(weighted_pred, 4),
            "physics_weight": round(physics_weight, 3),
            "ml_weight": round(ml_weight, 3),
            "residual_samples": self.residual_stats["sample_count"],
            "model_type": self.model_type,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the residual model."""
        return {
            "type": "residual_ensemble",
            "model_type": self.model_type,
            "sample_count": self.residual_stats["sample_count"],
            "residual_mean": self.residual_stats["mean"],
            "residual_std": self.residual_stats["std"],
            "has_weights": bool(self.residual_weights),
            "feature_count": len(self._get_residual_feature_names()),
        }
