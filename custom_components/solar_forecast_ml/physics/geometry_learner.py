"""Geometry Learner for Solar Forecast ML Integration V12.2.0 @zara

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .physics_engine import (
    PhysicsEngine,
    SunPosition,
    IrradianceData,
    PanelGeometry,
)
from .panel_group_calculator import PanelGroupCalculator, PanelGroup

_LOGGER = logging.getLogger(__name__)


@dataclass
class LearningConfig:
    """Configuration for learning parameters, loaded from learning_config.json. @zara"""
    # Learning parameters
    baseline_prediction_kwh: float = 0.05
    shadow_detection_efficiency: float = 0.2
    shadow_hour_threshold: float = 0.7

    # Smoothing parameters for adaptive learning
    smoothing_aggressive_threshold: float = 1.0  # >100% deviation
    smoothing_aggressive_old: float = 0.3
    smoothing_aggressive_new: float = 0.7
    smoothing_fast_threshold: float = 0.5  # >50% deviation
    smoothing_fast_old: float = 0.5
    smoothing_fast_new: float = 0.5
    smoothing_normal_old: float = 0.8
    smoothing_normal_new: float = 0.2

    # Efficiency clamps
    efficiency_min: float = 0.1
    efficiency_max: float = 5.0
    efficiency_conservative_min: float = 0.3
    efficiency_conservative_max: float = 3.0

    # Weather adjustment
    weather_cloud_factor: float = 0.5

    # Elevation adjustment
    elevation_low_threshold: float = 10.0
    elevation_low_factor: float = 0.15
    elevation_high_factor: float = 0.08

    # Physics defaults
    physics_albedo: float = 0.2
    physics_system_efficiency: float = 0.90

    @classmethod
    def load_from_file(cls, data_path: Path) -> "LearningConfig":
        """Load config from learning_config.json synchronously. Falls back to defaults if not found. @zara

        Note: This method performs blocking file I/O. Use async_load_from_file()
        when calling from an async context (e.g., Home Assistant event loop).
        """
        config_file = data_path / "physics" / "learning_config.json"

        if not config_file.exists():
            _LOGGER.debug("No learning_config.json found, using defaults")
            return cls()

        try:
            with open(config_file, "r") as f:
                data = json.load(f)

            return cls._parse_config_data(data, config_file)

        except Exception as e:
            _LOGGER.warning("Failed to load learning_config.json: %s, using defaults", e)
            return cls()

    @classmethod
    async def async_load_from_file(cls, data_path: Path) -> "LearningConfig":
        """Load config from learning_config.json asynchronously. Falls back to defaults if not found. @zara

        Use this method when calling from an async context (e.g., Home Assistant event loop)
        to avoid blocking file I/O.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, cls.load_from_file, data_path)

    @classmethod
    def _parse_config_data(cls, data: dict, config_file: Path) -> "LearningConfig":
        """Parse config data dict into LearningConfig instance. @zara"""
        lp = data.get("learning_parameters", {})
        smoothing = lp.get("smoothing", {})
        clamps = lp.get("efficiency_clamps", {})
        weather = data.get("weather_adjustment", {})
        elevation = data.get("elevation_adjustment", {})
        physics = data.get("physics_defaults", {})

        config = cls(
            baseline_prediction_kwh=lp.get("baseline_prediction_kwh", 0.05),
            shadow_detection_efficiency=lp.get("shadow_detection_efficiency", 0.2),
            shadow_hour_threshold=lp.get("shadow_hour_threshold", 0.7),

            smoothing_aggressive_threshold=smoothing.get("aggressive", {}).get("deviation_threshold", 1.0),
            smoothing_aggressive_old=smoothing.get("aggressive", {}).get("old_weight", 0.3),
            smoothing_aggressive_new=smoothing.get("aggressive", {}).get("new_weight", 0.7),
            smoothing_fast_threshold=smoothing.get("fast", {}).get("deviation_threshold", 0.5),
            smoothing_fast_old=smoothing.get("fast", {}).get("old_weight", 0.5),
            smoothing_fast_new=smoothing.get("fast", {}).get("new_weight", 0.5),
            smoothing_normal_old=smoothing.get("normal", {}).get("old_weight", 0.8),
            smoothing_normal_new=smoothing.get("normal", {}).get("new_weight", 0.2),

            efficiency_min=clamps.get("min_efficiency", 0.1),
            efficiency_max=clamps.get("max_efficiency", 5.0),
            efficiency_conservative_min=clamps.get("conservative_min", 0.3),
            efficiency_conservative_max=clamps.get("conservative_max", 3.0),

            weather_cloud_factor=weather.get("cloud_cover_factor", 0.5),

            elevation_low_threshold=elevation.get("low_elevation_threshold_deg", 10.0),
            elevation_low_factor=elevation.get("low_elevation_factor", 0.15),
            elevation_high_factor=elevation.get("high_elevation_factor", 0.08),

            physics_albedo=physics.get("albedo", 0.2),
            physics_system_efficiency=physics.get("system_efficiency", 0.90),
        )

        _LOGGER.info("Loaded learning config from %s", config_file)
        return config


# Global config instance (loaded lazily)
_learning_config: Optional[LearningConfig] = None


def get_learning_config(data_path: Optional[Path] = None) -> LearningConfig:
    """Get the learning config, loading from file if not yet loaded (sync version). @zara

    Note: This function performs blocking file I/O on first call.
    Use async_get_learning_config() when calling from an async context.
    """
    global _learning_config
    if _learning_config is None:
        if data_path is not None:
            _learning_config = LearningConfig.load_from_file(data_path)
        else:
            _learning_config = LearningConfig()
    return _learning_config


async def async_get_learning_config(data_path: Optional[Path] = None) -> LearningConfig:
    """Get the learning config asynchronously, loading from file if not yet loaded. @zara

    Use this function when calling from an async context (e.g., Home Assistant event loop)
    to avoid blocking file I/O.
    """
    global _learning_config
    if _learning_config is None:
        if data_path is not None:
            _learning_config = await LearningConfig.async_load_from_file(data_path)
        else:
            _learning_config = LearningConfig()
    return _learning_config


def reload_learning_config(data_path: Path) -> LearningConfig:
    """Force reload of learning config from file (sync version). @zara"""
    global _learning_config
    _learning_config = LearningConfig.load_from_file(data_path)
    return _learning_config


async def async_reload_learning_config(data_path: Path) -> LearningConfig:
    """Force reload of learning config from file asynchronously. @zara"""
    global _learning_config
    _learning_config = await LearningConfig.async_load_from_file(data_path)
    return _learning_config


@dataclass
class ClearSkyDataPoint:
    """A single clear-sky observation for geometry learning. @zara"""

    timestamp: str
    sun_elevation_deg: float
    sun_azimuth_deg: float
    actual_power_kwh: float
    ghi_wm2: float
    dni_wm2: float
    dhi_wm2: float
    ambient_temp_c: float
    cloud_cover_percent: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "sun_elevation_deg": round(self.sun_elevation_deg, 2),
            "sun_azimuth_deg": round(self.sun_azimuth_deg, 2),
            "actual_power_kwh": round(self.actual_power_kwh, 4),
            "ghi_wm2": round(self.ghi_wm2, 1),
            "dni_wm2": round(self.dni_wm2, 1),
            "dhi_wm2": round(self.dhi_wm2, 1),
            "ambient_temp_c": round(self.ambient_temp_c, 1),
            "cloud_cover_percent": round(self.cloud_cover_percent, 1),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClearSkyDataPoint":
        return cls(
            timestamp=data["timestamp"],
            sun_elevation_deg=data["sun_elevation_deg"],
            sun_azimuth_deg=data["sun_azimuth_deg"],
            actual_power_kwh=data["actual_power_kwh"],
            ghi_wm2=data["ghi_wm2"],
            dni_wm2=data["dni_wm2"],
            dhi_wm2=data["dhi_wm2"],
            ambient_temp_c=data["ambient_temp_c"],
            cloud_cover_percent=data["cloud_cover_percent"],
        )


@dataclass
class GeometryEstimate:
    """Current geometry estimate with confidence metrics. @zara"""

    tilt_deg: float = 30.0
    azimuth_deg: float = 180.0  # Default azimuth (South)
    confidence: float = 0.0     # 0 = default, 1 = fully converged
    sample_count: int = 0
    last_updated: str = ""
    convergence_history: list = field(default_factory=list)
    error_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tilt_deg": round(self.tilt_deg, 2),
            "azimuth_deg": round(self.azimuth_deg, 2),
            "confidence": round(self.confidence, 4),
            "sample_count": self.sample_count,
            "last_updated": self.last_updated,
            "convergence_history": self.convergence_history[-10:],  # Last 10 updates
            "error_metrics": self.error_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeometryEstimate":
        return cls(
            tilt_deg=data.get("tilt_deg", 30.0),
            azimuth_deg=data.get("azimuth_deg", 180.0),
            confidence=data.get("confidence", 0.0),
            sample_count=data.get("sample_count", 0),
            last_updated=data.get("last_updated", ""),
            convergence_history=data.get("convergence_history", []),
            error_metrics=data.get("error_metrics", {}),
        )


class GeometryLearner:
    """Learns panel geometry from production data using optimization. @zara"""

    # Clear-sky detection thresholds
    MAX_CLOUD_COVER = 20.0           # Max cloud cover for clear-sky (%)
    MIN_GHI_RATIO = 0.75             # Min GHI/theoretical ratio for clear-sky
    MIN_SUN_ELEVATION = 15.0         # Min sun elevation for valid data (deg)

    # Learning parameters
    MIN_SAMPLES_FOR_ESTIMATE = 20    # Min samples before first estimate
    MIN_SAMPLES_FOR_CONFIDENCE = 50  # Min samples for moderate confidence
    FULL_CONFIDENCE_SAMPLES = 150    # Samples for full confidence

    # Optimization parameters
    LEARNING_RATE_INITIAL = 0.5      # Initial learning rate
    LEARNING_RATE_MIN = 0.05         # Minimum learning rate
    DAMPING_FACTOR = 0.1             # Levenberg-Marquardt damping

    # Geometry constraints
    TILT_MIN = 0.0
    TILT_MAX = 90.0
    AZIMUTH_MIN = 0.0
    AZIMUTH_MAX = 360.0

    def __init__(
        self,
        data_path: Path,
        system_capacity_kwp: float,
        initial_tilt: float = 30.0,
        initial_azimuth: float = 180.0,
        skip_load: bool = False,
    ):
        """Initialize the geometry learner. @zara"""
        self.data_path = data_path
        self.system_capacity_kwp = system_capacity_kwp

        # State file (in physics/ subdirectory for better organization)
        self.state_file = data_path / "physics" / "learned_geometry.json"

        # Current estimate
        self.estimate = GeometryEstimate(
            tilt_deg=initial_tilt,
            azimuth_deg=initial_azimuth,
        )

        # Data points for learning
        self.data_points: list[ClearSkyDataPoint] = []

        # Physics engine for calculations
        self._physics = PhysicsEngine(
            system_capacity_kwp=system_capacity_kwp,
            panel_tilt_deg=initial_tilt,
            panel_azimuth_deg=initial_azimuth,
        )

        # Track if state is loaded
        self._state_loaded = False

        # Load existing state (skip if called from async context)
        if not skip_load:
            self._load_state()
            self._state_loaded = True

        _LOGGER.info(
            "GeometryLearner initialized: tilt=%.1f, azimuth=%.1f, samples=%d, confidence=%.2f",
            self.estimate.tilt_deg,
            self.estimate.azimuth_deg,
            self.estimate.sample_count,
            self.estimate.confidence,
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
        """Load saved state from file. @zara"""
        if not self.state_file.exists():
            _LOGGER.debug("No existing geometry state found")
            return

        try:
            # Use sync file read - will be wrapped in executor by caller if needed
            self._load_state_sync()
        except Exception as e:
            _LOGGER.warning("Failed to load geometry state: %s", e)

    def _load_state_sync(self) -> None:
        """Synchronous state loading - call from executor in async context @zara"""
        if not self.state_file.exists():
            return

        with open(self.state_file, "r") as f:
            data = json.load(f)

        self.estimate = GeometryEstimate.from_dict(data.get("estimate", {}))
        self.data_points = [
            ClearSkyDataPoint.from_dict(p)
            for p in data.get("data_points", [])
        ]

        # Update physics engine with loaded geometry
        self._physics.update_geometry(
            self.estimate.tilt_deg,
            self.estimate.azimuth_deg,
            self.estimate.confidence,
        )

        _LOGGER.info(
            "Loaded geometry state: tilt=%.1f, azimuth=%.1f, samples=%d",
            self.estimate.tilt_deg,
            self.estimate.azimuth_deg,
            len(self.data_points),
        )

    async def save_state(self) -> None:
        """Save current state to file. @zara"""
        try:
            data = {
                "version": "1.0",
                "estimate": self.estimate.to_dict(),
                "data_points": [p.to_dict() for p in self.data_points[-500:]],  # Keep last 500
                "metadata": {
                    "system_capacity_kwp": self.system_capacity_kwp,
                    "saved_at": datetime.now().isoformat(),
                },
            }

            def _write_sync():
                self.data_path.mkdir(parents=True, exist_ok=True)
                with open(self.state_file, "w") as f:
                    json.dump(data, f, indent=2)

            import asyncio
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _write_sync)

            _LOGGER.debug("Saved geometry state: %d samples", len(self.data_points))

        except Exception as e:
            _LOGGER.error("Failed to save geometry state: %s", e)

    def is_clear_sky(
        self,
        cloud_cover_percent: float,
        ghi_wm2: float,
        theoretical_max_wm2: float,
    ) -> bool:
        """Check if conditions are suitable for geometry learning. @zara"""
        # Check cloud cover
        if cloud_cover_percent > self.MAX_CLOUD_COVER:
            return False

        # Check GHI ratio (if theoretical is available)
        if theoretical_max_wm2 > 50:  # Avoid division issues at low values
            ghi_ratio = ghi_wm2 / theoretical_max_wm2
            if ghi_ratio < self.MIN_GHI_RATIO:
                return False

        return True

    def add_data_point(
        self,
        timestamp: str,
        sun_elevation_deg: float,
        sun_azimuth_deg: float,
        actual_power_kwh: float,
        ghi_wm2: float,
        dni_wm2: float,
        dhi_wm2: float,
        ambient_temp_c: float,
        cloud_cover_percent: float,
        theoretical_max_wm2: float = 0.0,
    ) -> bool:
        """Add a data point for geometry learning. @zara"""
        # Validate sun position
        if sun_elevation_deg < self.MIN_SUN_ELEVATION:
            _LOGGER.debug("Sun too low for geometry learning: %.1f deg", sun_elevation_deg)
            return False

        # Validate production (must have meaningful production)
        if actual_power_kwh <= 0.01:  # 10 Wh minimum
            _LOGGER.debug("Production too low for geometry learning: %.4f kWh", actual_power_kwh)
            return False

        # Check clear-sky conditions
        if not self.is_clear_sky(cloud_cover_percent, ghi_wm2, theoretical_max_wm2):
            _LOGGER.debug(
                "Not clear-sky: clouds=%.1f%%, GHI=%.1f, theoretical=%.1f",
                cloud_cover_percent,
                ghi_wm2,
                theoretical_max_wm2,
            )
            return False

        # Validate irradiance
        if ghi_wm2 < 50 or dni_wm2 < 10:
            _LOGGER.debug("Irradiance too low: GHI=%.1f, DNI=%.1f", ghi_wm2, dni_wm2)
            return False

        # Create and add data point
        point = ClearSkyDataPoint(
            timestamp=timestamp,
            sun_elevation_deg=sun_elevation_deg,
            sun_azimuth_deg=sun_azimuth_deg,
            actual_power_kwh=actual_power_kwh,
            ghi_wm2=ghi_wm2,
            dni_wm2=dni_wm2,
            dhi_wm2=dhi_wm2,
            ambient_temp_c=ambient_temp_c,
            cloud_cover_percent=cloud_cover_percent,
        )

        self.data_points.append(point)
        _LOGGER.debug(
            "Added geometry data point: sun=(%.1f, %.1f), power=%.4f kWh",
            sun_elevation_deg,
            sun_azimuth_deg,
            actual_power_kwh,
        )

        return True

    def _calculate_prediction_error(
        self,
        tilt_deg: float,
        azimuth_deg: float,
        point: ClearSkyDataPoint,
    ) -> float:
        """Calculate prediction error for a given geometry. @zara"""
        geometry = PanelGeometry(tilt_deg=tilt_deg, azimuth_deg=azimuth_deg)
        sun = SunPosition(
            elevation_deg=point.sun_elevation_deg,
            azimuth_deg=point.sun_azimuth_deg,
        )
        irradiance = IrradianceData(
            ghi=point.ghi_wm2,
            dni=point.dni_wm2,
            dhi=point.dhi_wm2,
        )

        result = self._physics.calculate_power_output(
            irradiance, sun, point.ambient_temp_c, geometry
        )

        error = (result.power_kwh - point.actual_power_kwh) ** 2
        return error

    def _calculate_total_error(
        self,
        tilt_deg: float,
        azimuth_deg: float,
    ) -> float:
        """Calculate total squared error for all data points. @zara"""
        total_error = 0.0
        for point in self.data_points:
            total_error += self._calculate_prediction_error(tilt_deg, azimuth_deg, point)
        return total_error

    def _calculate_gradient(
        self,
        tilt_deg: float,
        azimuth_deg: float,
        delta: float = 0.5,
    ) -> tuple[float, float]:
        """Calculate numerical gradient of error function. @zara"""
        base_error = self._calculate_total_error(tilt_deg, azimuth_deg)

        # Gradient for tilt
        error_tilt_plus = self._calculate_total_error(tilt_deg + delta, azimuth_deg)
        error_tilt_minus = self._calculate_total_error(tilt_deg - delta, azimuth_deg)
        grad_tilt = (error_tilt_plus - error_tilt_minus) / (2 * delta)

        # Gradient for azimuth
        error_az_plus = self._calculate_total_error(tilt_deg, azimuth_deg + delta)
        error_az_minus = self._calculate_total_error(tilt_deg, azimuth_deg - delta)
        grad_azimuth = (error_az_plus - error_az_minus) / (2 * delta)

        return (grad_tilt, grad_azimuth)

    async def optimize_geometry(self) -> bool:
        """Run optimization to find best geometry. @zara"""
        if len(self.data_points) < self.MIN_SAMPLES_FOR_ESTIMATE:
            _LOGGER.debug(
                "Not enough samples for optimization: %d < %d",
                len(self.data_points),
                self.MIN_SAMPLES_FOR_ESTIMATE,
            )
            return False

        # Current estimate as starting point
        tilt = self.estimate.tilt_deg
        azimuth = self.estimate.azimuth_deg

        # Adaptive learning rate based on samples
        sample_ratio = min(1.0, len(self.data_points) / self.FULL_CONFIDENCE_SAMPLES)
        learning_rate = self.LEARNING_RATE_INITIAL * (1 - 0.7 * sample_ratio)
        learning_rate = max(self.LEARNING_RATE_MIN, learning_rate)

        # Gradient descent iterations
        max_iterations = 50
        min_improvement = 0.0001
        prev_error = self._calculate_total_error(tilt, azimuth)

        _LOGGER.debug(
            "Starting geometry optimization: tilt=%.1f, azimuth=%.1f, error=%.6f",
            tilt, azimuth, prev_error
        )

        for iteration in range(max_iterations):
            # Calculate gradient
            grad_tilt, grad_azimuth = self._calculate_gradient(tilt, azimuth)

            # Levenberg-Marquardt update with damping
            # This prevents oscillation and ensures convergence
            damping = self.DAMPING_FACTOR * (1 + iteration / 10)

            # Update parameters
            new_tilt = tilt - learning_rate * grad_tilt / (1 + damping * abs(grad_tilt))
            new_azimuth = azimuth - learning_rate * grad_azimuth / (1 + damping * abs(grad_azimuth))

            # Constrain to valid ranges
            new_tilt = max(self.TILT_MIN, min(self.TILT_MAX, new_tilt))
            new_azimuth = new_azimuth % 360  # Wrap azimuth

            # Calculate new error
            new_error = self._calculate_total_error(new_tilt, new_azimuth)

            # Check for improvement
            improvement = (prev_error - new_error) / max(prev_error, 0.001)

            if improvement < min_improvement and iteration > 10:
                _LOGGER.debug(
                    "Optimization converged at iteration %d: improvement=%.6f",
                    iteration,
                    improvement,
                )
                break

            if new_error < prev_error:
                tilt = new_tilt
                azimuth = new_azimuth
                prev_error = new_error

        # Calculate final error metrics
        final_error = self._calculate_total_error(tilt, azimuth)
        rmse = math.sqrt(final_error / len(self.data_points))

        # Calculate confidence based on samples and error
        sample_confidence = min(1.0, len(self.data_points) / self.FULL_CONFIDENCE_SAMPLES)
        error_confidence = max(0.0, 1.0 - rmse * 10)  # Lower error = higher confidence
        confidence = sample_confidence * 0.7 + error_confidence * 0.3

        # Update estimate
        old_tilt = self.estimate.tilt_deg
        old_azimuth = self.estimate.azimuth_deg

        self.estimate.tilt_deg = tilt
        self.estimate.azimuth_deg = azimuth
        self.estimate.confidence = confidence
        self.estimate.sample_count = len(self.data_points)
        self.estimate.last_updated = datetime.now().isoformat()
        self.estimate.error_metrics = {
            "rmse_kwh": round(rmse, 6),
            "total_error": round(final_error, 6),
            "samples_used": len(self.data_points),
        }

        # Track convergence history
        self.estimate.convergence_history.append({
            "timestamp": datetime.now().isoformat(),
            "tilt_deg": round(tilt, 2),
            "azimuth_deg": round(azimuth, 2),
            "confidence": round(confidence, 4),
            "rmse_kwh": round(rmse, 6),
            "delta_tilt": round(tilt - old_tilt, 2),
            "delta_azimuth": round(azimuth - old_azimuth, 2),
        })

        # Update physics engine
        self._physics.update_geometry(tilt, azimuth, confidence)

        _LOGGER.info(
            "Geometry optimized: tilt=%.1f (delta=%.1f), azimuth=%.1f (delta=%.1f), "
            "confidence=%.2f, RMSE=%.4f kWh",
            tilt,
            tilt - old_tilt,
            azimuth,
            azimuth - old_azimuth,
            confidence,
            rmse,
        )

        # Save state
        await self.save_state()

        return True

    def get_current_estimate(self) -> dict:
        """Get the current geometry estimate. @zara"""
        return self.estimate.to_dict()

    def get_physics_engine(self) -> PhysicsEngine:
        """Get the physics engine with current geometry. @zara"""
        return self._physics

    async def bulk_add_historical_data(
        self,
        historical_samples: list[dict],
    ) -> dict:
        """Add multiple historical data points for bulk training. @zara"""
        accepted_count = 0
        rejected_count = 0
        clear_sky_count = 0

        _LOGGER.info(f"Processing {len(historical_samples)} historical samples for GeometryLearner...")

        for sample in historical_samples:
            try:
                # Validate required fields
                required_fields = [
                    "timestamp", "sun_elevation_deg", "sun_azimuth_deg",
                    "actual_power_kwh", "ghi_wm2", "dni_wm2", "dhi_wm2",
                    "ambient_temp_c", "cloud_cover_percent"
                ]

                if not all(sample.get(f) is not None for f in required_fields):
                    rejected_count += 1
                    continue

                # Try to add the data point
                was_accepted = self.add_data_point(
                    timestamp=sample["timestamp"],
                    sun_elevation_deg=sample["sun_elevation_deg"],
                    sun_azimuth_deg=sample["sun_azimuth_deg"],
                    actual_power_kwh=sample["actual_power_kwh"],
                    ghi_wm2=sample["ghi_wm2"],
                    dni_wm2=sample["dni_wm2"],
                    dhi_wm2=sample["dhi_wm2"],
                    ambient_temp_c=sample["ambient_temp_c"],
                    cloud_cover_percent=sample["cloud_cover_percent"],
                    theoretical_max_wm2=sample.get("theoretical_max_wm2", 0.0),
                )

                if was_accepted:
                    accepted_count += 1
                    # Count clear-sky samples
                    if sample["cloud_cover_percent"] <= self.MAX_CLOUD_COVER:
                        clear_sky_count += 1
                else:
                    rejected_count += 1

            except Exception as e:
                _LOGGER.debug(f"Error processing sample: {e}")
                rejected_count += 1

        _LOGGER.info(
            f"Bulk import complete: {accepted_count} accepted (clear-sky: {clear_sky_count}), "
            f"{rejected_count} rejected"
        )

        # Run optimization if we have enough samples
        optimization_result = None
        if accepted_count > 0 and len(self.data_points) >= self.MIN_SAMPLES_FOR_ESTIMATE:
            _LOGGER.info("Running geometry optimization after bulk import...")
            success = await self.optimize_geometry()
            if success:
                optimization_result = self.estimate.to_dict()

        return {
            "samples_processed": len(historical_samples),
            "accepted": accepted_count,
            "rejected": rejected_count,
            "clear_sky_samples": clear_sky_count,
            "total_data_points": len(self.data_points),
            "optimization_ran": optimization_result is not None,
            "current_estimate": self.estimate.to_dict(),
        }


WEATHER_CLEAR_MAX_CLOUDS = 20
WEATHER_CLOUDY_MIN_CLOUDS = 40


@dataclass
class PanelGroupEfficiency:
    """Learned efficiency factors for a single panel group. @zara"""
    name: str
    configured_tilt_deg: float
    configured_azimuth_deg: float
    power_kwp: float
    energy_sensor: Optional[str] = None
    learned_efficiency_factor: float = 1.0
    learned_shadow_hours: List[int] = field(default_factory=list)
    sample_count: int = 0
    confidence: float = 0.0
    hourly_efficiency: Dict[int, float] = field(default_factory=dict)
    learning_history: List[Dict[str, Any]] = field(default_factory=list)
    avg_cloud_cover_at_learning: Dict[int, float] = field(default_factory=dict)
    sample_count_per_hour: Dict[int, int] = field(default_factory=dict)
    avg_sun_elevation_at_learning: Dict[int, float] = field(default_factory=dict)
    hourly_efficiency_clear: Dict[int, float] = field(default_factory=dict)
    hourly_efficiency_cloudy: Dict[int, float] = field(default_factory=dict)
    sample_count_clear: Dict[int, int] = field(default_factory=dict)
    sample_count_cloudy: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "configured_tilt_deg": round(self.configured_tilt_deg, 2),
            "configured_azimuth_deg": round(self.configured_azimuth_deg, 2),
            "power_kwp": round(self.power_kwp, 3),
            "learned_efficiency_factor": round(self.learned_efficiency_factor, 4),
            "learned_shadow_hours": self.learned_shadow_hours,
            "sample_count": self.sample_count,
            "confidence": round(self.confidence, 4),
            "hourly_efficiency": {str(k): round(v, 4) for k, v in self.hourly_efficiency.items()},
            "learning_history": self.learning_history[-30:],
            "avg_cloud_cover_at_learning": {str(k): round(v, 1) for k, v in self.avg_cloud_cover_at_learning.items()},
            "sample_count_per_hour": {str(k): v for k, v in self.sample_count_per_hour.items()},
            "avg_sun_elevation_at_learning": {str(k): round(v, 1) for k, v in self.avg_sun_elevation_at_learning.items()},
            "hourly_efficiency_clear": {str(k): round(v, 4) for k, v in self.hourly_efficiency_clear.items()},
            "hourly_efficiency_cloudy": {str(k): round(v, 4) for k, v in self.hourly_efficiency_cloudy.items()},
            "sample_count_clear": {str(k): v for k, v in self.sample_count_clear.items()},
            "sample_count_cloudy": {str(k): v for k, v in self.sample_count_cloudy.items()},
        }
        if self.energy_sensor:
            result["energy_sensor"] = self.energy_sensor
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "PanelGroupEfficiency":
        hourly_eff = data.get("hourly_efficiency", {})
        hourly_eff_int = {int(k): v for k, v in hourly_eff.items()}
        avg_cloud = data.get("avg_cloud_cover_at_learning", {})
        avg_cloud_int = {int(k): v for k, v in avg_cloud.items()}
        sample_per_hour = data.get("sample_count_per_hour", {})
        sample_per_hour_int = {int(k): v for k, v in sample_per_hour.items()}
        avg_elevation = data.get("avg_sun_elevation_at_learning", {})
        avg_elevation_int = {int(k): v for k, v in avg_elevation.items()}
        hourly_eff_clear = data.get("hourly_efficiency_clear", {})
        hourly_eff_clear_int = {int(k): v for k, v in hourly_eff_clear.items()}
        hourly_eff_cloudy = data.get("hourly_efficiency_cloudy", {})
        hourly_eff_cloudy_int = {int(k): v for k, v in hourly_eff_cloudy.items()}
        sample_count_clear = data.get("sample_count_clear", {})
        sample_count_clear_int = {int(k): v for k, v in sample_count_clear.items()}
        sample_count_cloudy = data.get("sample_count_cloudy", {})
        sample_count_cloudy_int = {int(k): v for k, v in sample_count_cloudy.items()}
        return cls(
            name=data.get("name", "Unknown"),
            configured_tilt_deg=data.get("configured_tilt_deg", 30.0),
            configured_azimuth_deg=data.get("configured_azimuth_deg", 180.0),
            power_kwp=data.get("power_kwp", 0.0),
            energy_sensor=data.get("energy_sensor"),
            learned_efficiency_factor=data.get("learned_efficiency_factor", 1.0),
            learned_shadow_hours=data.get("learned_shadow_hours", []),
            sample_count=data.get("sample_count", 0),
            confidence=data.get("confidence", 0.0),
            hourly_efficiency=hourly_eff_int,
            learning_history=data.get("learning_history", []),
            avg_cloud_cover_at_learning=avg_cloud_int,
            sample_count_per_hour=sample_per_hour_int,
            avg_sun_elevation_at_learning=avg_elevation_int,
            hourly_efficiency_clear=hourly_eff_clear_int,
            hourly_efficiency_cloudy=hourly_eff_cloudy_int,
            sample_count_clear=sample_count_clear_int,
            sample_count_cloudy=sample_count_cloudy_int,
        )


class PanelGroupEfficiencyLearner:
    """Learns efficiency factors per panel group from production data. @zara

    When panel groups are configured, the geometry is KNOWN (user-provided).
    This learner focuses on discovering:
    - Per-group efficiency factors (shadows, soiling, degradation)
    - Hourly efficiency patterns (morning shadows, evening shadows)
    - Systematic deviations from physics model

    This is complementary to GeometryLearner:
    - GeometryLearner: Learns unknown geometry (tilt/azimuth) when not configured
    - PanelGroupEfficiencyLearner: Learns efficiency when geometry IS configured
    """

    MIN_SAMPLES_FOR_ESTIMATE = 10  # Reduced for winter months @zara
    MIN_SAMPLES_FOR_CONFIDENCE = 30
    FULL_CONFIDENCE_SAMPLES = 100
    MAX_CLOUD_COVER = 50.0  # Relaxed for winter - allow more cloudy days @zara
    MIN_SUN_ELEVATION = 3.0  # Lowered for winter to capture morning/evening hours @zara
    MIN_GHI = 20  # Lowered from 50 to allow more learning data in winter @zara

    def __init__(
        self,
        data_path: Path,
        panel_groups: List[Dict[str, Any]],
        skip_load: bool = False,
        _config: Optional[LearningConfig] = None,
        _panel_group_calculator: Optional[PanelGroupCalculator] = None,
    ):
        """Initialize the panel group efficiency learner. @zara

        Note: When calling from an async context, use async_create() factory method
        to avoid blocking file I/O.

        Args:
            data_path: Path to data directory
            panel_groups: List of panel group configurations
            skip_load: If True, skip loading state from file
            _config: Pre-loaded LearningConfig (internal, used by async_create)
            _panel_group_calculator: Pre-created calculator (internal, used by async_create)
        """
        self.data_path = data_path
        # State file (in physics/ subdirectory for better organization)
        self.state_file = data_path / "physics" / "learned_panel_group_efficiency.json"

        # Load learning config from JSON (externalized parameters)
        # Use pre-loaded config if provided (from async_create), otherwise load sync
        if _config is not None:
            self._config = _config
        else:
            self._config = get_learning_config(data_path)

        self._panel_groups_config = panel_groups
        # Pass data_path so PanelGroupCalculator can load physics defaults from config
        # Use pre-created calculator if provided (from async_create)
        if _panel_group_calculator is not None:
            self._panel_group_calculator = _panel_group_calculator
        else:
            self._panel_group_calculator = PanelGroupCalculator(
                panel_groups=panel_groups,
                data_path=data_path
            )

        self.group_efficiencies: List[PanelGroupEfficiency] = []
        for idx, group in enumerate(panel_groups):
            self.group_efficiencies.append(PanelGroupEfficiency(
                name=group.get("name", f"Gruppe {idx + 1}"),
                configured_tilt_deg=float(group.get("tilt", 30)),
                configured_azimuth_deg=float(group.get("azimuth", 180)),
                power_kwp=float(group.get("power_wp", 0)) / 1000.0,
                energy_sensor=group.get("energy_sensor"),
            ))

        self.data_points: List[ClearSkyDataPoint] = []
        self.total_samples = 0
        self.last_updated = ""
        self._state_loaded = False

        if not skip_load:
            self._load_state()
            self._state_loaded = True

        _LOGGER.info(
            "PanelGroupEfficiencyLearner initialized: %d groups, total %.2f kWp",
            len(self.group_efficiencies),
            self._panel_group_calculator.total_capacity_kwp,
        )

    @classmethod
    async def async_create(
        cls,
        data_path: Path,
        panel_groups: List[Dict[str, Any]],
        skip_load: bool = False,
    ) -> "PanelGroupEfficiencyLearner":
        """Async factory method to create PanelGroupEfficiencyLearner without blocking. @zara

        Use this method when creating PanelGroupEfficiencyLearner from an async context
        (e.g., Home Assistant event loop) to avoid blocking file I/O.

        Args:
            data_path: Path to data directory
            panel_groups: List of panel group configurations
            skip_load: If True, skip loading state from file

        Returns:
            Initialized PanelGroupEfficiencyLearner instance
        """
        # Load config asynchronously
        config = await async_get_learning_config(data_path)

        # Create PanelGroupCalculator asynchronously
        calculator = await PanelGroupCalculator.async_create(
            panel_groups=panel_groups,
            data_path=data_path
        )

        # Create instance with pre-loaded components (skip_load=True to avoid sync I/O)
        instance = cls(
            data_path=data_path,
            panel_groups=panel_groups,
            skip_load=True,  # We'll load state async below
            _config=config,
            _panel_group_calculator=calculator,
        )

        # Load state asynchronously if needed
        if not skip_load:
            await instance.async_load_state()

        return instance

    async def async_load_state(self) -> None:
        """Load state asynchronously. @zara"""
        if self._state_loaded:
            return

        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_state_sync)
        self._state_loaded = True

    def _load_state(self) -> None:
        """Load saved state from file. @zara"""
        if not self.state_file.exists():
            _LOGGER.debug("No existing panel group efficiency state found")
            return

        try:
            self._load_state_sync()
        except Exception as e:
            _LOGGER.warning("Failed to load panel group efficiency state: %s", e)

    def _load_state_sync(self) -> None:
        """Synchronous state loading. @zara"""
        if not self.state_file.exists():
            return

        with open(self.state_file, "r") as f:
            data = json.load(f)

        version = data.get("version", "1.0")
        if version not in ("2.0", "2.1"):
            _LOGGER.info("Old geometry state version, starting fresh for panel groups")
            return

        saved_groups = data.get("panel_groups", [])
        for saved in saved_groups:
            for eff in self.group_efficiencies:
                if eff.name == saved.get("name"):
                    eff.learned_efficiency_factor = saved.get("learned_efficiency_factor", 1.0)
                    eff.learned_shadow_hours = saved.get("learned_shadow_hours", [])
                    eff.sample_count = saved.get("sample_count", 0)
                    eff.confidence = saved.get("confidence", 0.0)
                    eff.hourly_efficiency = {
                        int(k): v for k, v in saved.get("hourly_efficiency", {}).items()
                    }
                    eff.avg_cloud_cover_at_learning = {
                        int(k): v for k, v in saved.get("avg_cloud_cover_at_learning", {}).items()
                    }
                    eff.sample_count_per_hour = {
                        int(k): v for k, v in saved.get("sample_count_per_hour", {}).items()
                    }
                    eff.avg_sun_elevation_at_learning = {
                        int(k): v for k, v in saved.get("avg_sun_elevation_at_learning", {}).items()
                    }
                    eff.hourly_efficiency_clear = {
                        int(k): v for k, v in saved.get("hourly_efficiency_clear", {}).items()
                    }
                    eff.hourly_efficiency_cloudy = {
                        int(k): v for k, v in saved.get("hourly_efficiency_cloudy", {}).items()
                    }
                    eff.sample_count_clear = {
                        int(k): v for k, v in saved.get("sample_count_clear", {}).items()
                    }
                    eff.sample_count_cloudy = {
                        int(k): v for k, v in saved.get("sample_count_cloudy", {}).items()
                    }
                    break

        self.data_points = [
            ClearSkyDataPoint.from_dict(p)
            for p in data.get("data_points", [])
        ]
        self.total_samples = data.get("total_samples", len(self.data_points))
        self.last_updated = data.get("last_updated", "")

        _LOGGER.info(
            "Loaded panel group efficiency state: %d groups, %d samples",
            len(self.group_efficiencies),
            self.total_samples,
        )

    async def save_state(self) -> None:
        """Save current state to file. @zara"""
        try:
            data = {
                "version": "2.1",
                "mode": "panel_groups",
                "panel_groups": [g.to_dict() for g in self.group_efficiencies],
                "data_points": [p.to_dict() for p in self.data_points[-500:]],
                "total_samples": self.total_samples,
                "last_updated": datetime.now().isoformat(),
                "metadata": {
                    "total_capacity_kwp": self._panel_group_calculator.total_capacity_kwp,
                    "group_count": len(self.group_efficiencies),
                },
            }

            def _write_sync():
                self.data_path.mkdir(parents=True, exist_ok=True)
                with open(self.state_file, "w") as f:
                    json.dump(data, f, indent=2)

            import asyncio
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _write_sync)

            _LOGGER.debug("Saved panel group efficiency state: %d samples", self.total_samples)

        except Exception as e:
            _LOGGER.error("Failed to save panel group efficiency state: %s", e)

    def add_data_point(
        self,
        timestamp: str,
        hour: int,
        sun_elevation_deg: float,
        sun_azimuth_deg: float,
        actual_power_kwh: float,
        ghi_wm2: float,
        dni_wm2: float,
        dhi_wm2: float,
        ambient_temp_c: float,
        cloud_cover_percent: float,
        per_group_actuals: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Add a data point for efficiency learning.

        Args:
            per_group_actuals: Optional dict mapping group_name to actual_kwh.
                               When provided, enables per-group learning instead
                               of distributing total actual by contribution %.

        @zara
        """
        if sun_elevation_deg < self.MIN_SUN_ELEVATION:
            return False

        if actual_power_kwh <= 0.001:
            return False

        if cloud_cover_percent > self.MAX_CLOUD_COVER:
            return False

        if ghi_wm2 < self.MIN_GHI:
            return False

        irradiance = IrradianceData(ghi=ghi_wm2, dni=dni_wm2, dhi=dhi_wm2)
        sun = SunPosition(elevation_deg=sun_elevation_deg, azimuth_deg=sun_azimuth_deg)

        physics_result = self._panel_group_calculator.calculate_total_power(
            irradiance, sun, ambient_temp_c
        )

        if physics_result.total_power_kwh > 0.001:
            if per_group_actuals:
                self._learn_from_per_group_actuals(
                    hour=hour,
                    physics_result=physics_result,
                    per_group_actuals=per_group_actuals,
                    cloud_cover=cloud_cover_percent,
                    sun_elevation_deg=sun_elevation_deg,
                )
            else:
                # Fallback: Distribute total actual by contribution percentage
                efficiency_ratio = actual_power_kwh / physics_result.total_power_kwh

                for idx, group_result in enumerate(physics_result.group_results):
                    if idx < len(self.group_efficiencies):
                        eff = self.group_efficiencies[idx]
                        if hour not in eff.hourly_efficiency:
                            eff.hourly_efficiency[hour] = efficiency_ratio
                        else:
                            old_eff = eff.hourly_efficiency[hour]
                            eff.hourly_efficiency[hour] = old_eff * 0.9 + efficiency_ratio * 0.1
                        eff.sample_count += 1

        point = ClearSkyDataPoint(
            timestamp=timestamp,
            sun_elevation_deg=sun_elevation_deg,
            sun_azimuth_deg=sun_azimuth_deg,
            actual_power_kwh=actual_power_kwh,
            ghi_wm2=ghi_wm2,
            dni_wm2=dni_wm2,
            dhi_wm2=dhi_wm2,
            ambient_temp_c=ambient_temp_c,
            cloud_cover_percent=cloud_cover_percent,
        )
        self.data_points.append(point)
        self.total_samples += 1

        return True

    def _learn_from_per_group_actuals(
        self,
        hour: int,
        physics_result,
        per_group_actuals: Dict[str, float],
        cloud_cover: float = 50.0,
        sun_elevation_deg: float = 20.0,
    ) -> None:
        """Learn efficiency from per-group actual production data.

        This is the preferred learning method when energy sensors are
        configured per panel group.

        @zara
        """
        for idx, group_result in enumerate(physics_result.group_results):
            if idx >= len(self.group_efficiencies):
                continue

            eff = self.group_efficiencies[idx]
            group_name = eff.name

            group_actual_kwh = per_group_actuals.get(group_name)

            if group_actual_kwh is None or group_actual_kwh <= 0:
                continue

            group_predicted_kwh = group_result.power_kwh

            # If physics prediction is too low, use existing hourly_efficiency as baseline
            # This handles cases where DNI/DHI=0 causes unrealistic physics predictions
            if group_predicted_kwh <= 0.001:
                # Fallback: Compare actual to what we would have predicted with current efficiency
                existing_eff = eff.hourly_efficiency.get(hour, eff.learned_efficiency_factor)
                # Use configurable baseline (from learning_config.json)
                baseline_prediction = self._config.baseline_prediction_kwh
                if group_actual_kwh > 0.01:
                    # Learn from actual vs baseline
                    efficiency_ratio = group_actual_kwh / baseline_prediction
                    efficiency_ratio = min(self._config.efficiency_conservative_max, efficiency_ratio)
                    _LOGGER.debug(
                        "Low physics prediction for %s @ %02d:00, using baseline. actual=%.4f, baseline=%.4f, eff=%.3f",
                        group_name, hour, group_actual_kwh, baseline_prediction, efficiency_ratio
                    )
                else:
                    # Very low actual production - this is shadow/no-sun situation
                    # Reduce efficiency significantly (configurable shadow detection)
                    efficiency_ratio = self._config.shadow_detection_efficiency
                    _LOGGER.debug(
                        "Very low production for %s @ %02d:00: actual=%.4f -> setting eff=%.2f (shadow detected)",
                        group_name, hour, group_actual_kwh, efficiency_ratio
                    )
            else:
                efficiency_ratio = group_actual_kwh / group_predicted_kwh

            # Clamp extreme ratios but don't reject them - they contain valuable information!
            # Very high ratios (>max) indicate physics underestimation
            # Very low ratios (<min) indicate shadowing
            if efficiency_ratio > self._config.efficiency_max:
                _LOGGER.info(
                    "High efficiency ratio for %s @ %02d:00: %.3f (actual=%.4f, predicted=%.4f) - clamping to %.1f",
                    group_name, hour, efficiency_ratio, group_actual_kwh, group_predicted_kwh,
                    self._config.efficiency_conservative_max
                )
                efficiency_ratio = self._config.efficiency_conservative_max
            elif efficiency_ratio < self._config.efficiency_min:
                _LOGGER.info(
                    "Low efficiency ratio for %s @ %02d:00: %.3f (actual=%.4f, predicted=%.4f) - clamping to %.1f",
                    group_name, hour, efficiency_ratio, group_actual_kwh, group_predicted_kwh,
                    self._config.efficiency_min
                )
                efficiency_ratio = self._config.efficiency_min

            if hour not in eff.hourly_efficiency:
                eff.hourly_efficiency[hour] = efficiency_ratio
                eff.avg_cloud_cover_at_learning[hour] = cloud_cover
                eff.avg_sun_elevation_at_learning[hour] = sun_elevation_deg
                eff.sample_count_per_hour[hour] = 1
            else:
                old_eff = eff.hourly_efficiency[hour]

                # Adaptive learning rate: faster when deviation is large (configurable via learning_config.json)
                deviation_ratio = abs(efficiency_ratio - old_eff) / max(old_eff, 0.1)

                if deviation_ratio > self._config.smoothing_aggressive_threshold:
                    # Very large deviation - learn aggressively
                    smoothing_old = self._config.smoothing_aggressive_old
                    smoothing_new = self._config.smoothing_aggressive_new
                    _LOGGER.info(
                        "Aggressive learning for %s @ %02d:00: deviation=%.0f%%, "
                        "old_eff=%.3f -> new_eff=%.3f (%.0f/%.0f smoothing)",
                        group_name, hour, deviation_ratio * 100, old_eff, efficiency_ratio,
                        smoothing_old * 100, smoothing_new * 100
                    )
                elif deviation_ratio > self._config.smoothing_fast_threshold:
                    # Moderate deviation - learn faster
                    smoothing_old = self._config.smoothing_fast_old
                    smoothing_new = self._config.smoothing_fast_new
                    _LOGGER.info(
                        "Fast learning for %s @ %02d:00: deviation=%.0f%%, "
                        "old_eff=%.3f -> new_eff=%.3f (%.0f/%.0f smoothing)",
                        group_name, hour, deviation_ratio * 100, old_eff, efficiency_ratio,
                        smoothing_old * 100, smoothing_new * 100
                    )
                else:
                    # Small deviation - normal smoothing
                    smoothing_old = self._config.smoothing_normal_old
                    smoothing_new = self._config.smoothing_normal_new

                eff.hourly_efficiency[hour] = old_eff * smoothing_old + efficiency_ratio * smoothing_new
                old_cloud = eff.avg_cloud_cover_at_learning.get(hour, cloud_cover)
                eff.avg_cloud_cover_at_learning[hour] = old_cloud * smoothing_old + cloud_cover * smoothing_new
                old_elevation = eff.avg_sun_elevation_at_learning.get(hour, sun_elevation_deg)
                eff.avg_sun_elevation_at_learning[hour] = old_elevation * smoothing_old + sun_elevation_deg * smoothing_new
                eff.sample_count_per_hour[hour] = eff.sample_count_per_hour.get(hour, 0) + 1

            eff.sample_count += 1

            self._learn_weather_specific_efficiency(
                eff, hour, efficiency_ratio, cloud_cover, smoothing_old, smoothing_new
            )

            _LOGGER.debug(
                "Per-group learning for %s @ %02d:00: actual=%.4f, predicted=%.4f, eff=%.3f, clouds=%.0f%%, elev=%.1f°",
                group_name, hour, group_actual_kwh, group_predicted_kwh, efficiency_ratio, cloud_cover, sun_elevation_deg
            )

    def _learn_weather_specific_efficiency(
        self,
        eff: PanelGroupEfficiency,
        hour: int,
        efficiency_ratio: float,
        cloud_cover: float,
        smoothing_old: float,
        smoothing_new: float,
    ) -> None:
        """Learn weather-specific efficiency (clear vs cloudy). @zara"""
        if cloud_cover <= WEATHER_CLEAR_MAX_CLOUDS:
            if hour not in eff.hourly_efficiency_clear:
                eff.hourly_efficiency_clear[hour] = efficiency_ratio
                eff.sample_count_clear[hour] = 1
            else:
                old_eff = eff.hourly_efficiency_clear[hour]
                eff.hourly_efficiency_clear[hour] = old_eff * smoothing_old + efficiency_ratio * smoothing_new
                eff.sample_count_clear[hour] = eff.sample_count_clear.get(hour, 0) + 1
            _LOGGER.debug(
                "Clear-sky learning for %s @ %02d:00: eff=%.3f, clouds=%.0f%%",
                eff.name, hour, eff.hourly_efficiency_clear[hour], cloud_cover
            )
        elif cloud_cover >= WEATHER_CLOUDY_MIN_CLOUDS:
            if hour not in eff.hourly_efficiency_cloudy:
                eff.hourly_efficiency_cloudy[hour] = efficiency_ratio
                eff.sample_count_cloudy[hour] = 1
            else:
                old_eff = eff.hourly_efficiency_cloudy[hour]
                eff.hourly_efficiency_cloudy[hour] = old_eff * smoothing_old + efficiency_ratio * smoothing_new
                eff.sample_count_cloudy[hour] = eff.sample_count_cloudy.get(hour, 0) + 1
            _LOGGER.debug(
                "Cloudy-sky learning for %s @ %02d:00: eff=%.3f, clouds=%.0f%%",
                eff.name, hour, eff.hourly_efficiency_cloudy[hour], cloud_cover
            )

    async def optimize_efficiency(self) -> bool:
        """Calculate optimal efficiency factors per group. @zara"""
        if self.total_samples < self.MIN_SAMPLES_FOR_ESTIMATE:
            _LOGGER.debug(
                "Not enough samples for efficiency optimization: %d < %d",
                self.total_samples,
                self.MIN_SAMPLES_FOR_ESTIMATE,
            )
            return False

        for eff in self.group_efficiencies:
            if eff.hourly_efficiency:
                # Use configurable range to capture real efficiency variations
                # Values >1.0 mean panels produce MORE than physics model predicts
                # Values <1.0 mean panels produce LESS than physics model predicts
                # Both are valid learning signals, not errors!
                valid_efficiencies = [
                    e for e in eff.hourly_efficiency.values()
                    if self._config.efficiency_min < e < self._config.efficiency_max
                ]
                if valid_efficiencies:
                    eff.learned_efficiency_factor = sum(valid_efficiencies) / len(valid_efficiencies)

                shadow_hours = []
                for hour, hour_eff in eff.hourly_efficiency.items():
                    if hour_eff < self._config.shadow_hour_threshold:
                        shadow_hours.append(hour)
                eff.learned_shadow_hours = sorted(shadow_hours)

            sample_confidence = min(1.0, eff.sample_count / self.FULL_CONFIDENCE_SAMPLES)
            eff.confidence = sample_confidence

        self.last_updated = datetime.now().isoformat()

        await self.save_state()

        _LOGGER.info(
            "Panel group efficiency optimized: %d groups, %d total samples",
            len(self.group_efficiencies),
            self.total_samples,
        )

        return True

    def get_efficiency_for_hour(
        self,
        group_index: int,
        hour: int,
        current_cloud_cover: Optional[float] = None,
        current_sun_elevation: Optional[float] = None,
        current_dni: Optional[float] = None,
        current_dhi: Optional[float] = None,
        current_ghi: Optional[float] = None,
    ) -> float:
        """Get learned efficiency factor for a group at a specific hour.

        The learned efficiency reflects how well the panel performs relative to
        physics prediction. This is primarily valid for DIRECT radiation (DNI).
        For DIFFUSE radiation (DHI), panel orientation matters much less, so
        efficiency should be closer to 1.0.

        When DNI/GHI data is available, we weight the learned efficiency by the
        direct radiation fraction. This ensures:
        - Clear sky (high DNI): Use mostly learned efficiency (preserves accuracy)
        - Cloudy sky (high DHI): Use efficiency closer to 1.0 (physics is accurate)

        Args:
            group_index: Index of the panel group
            hour: Hour of day (0-23)
            current_cloud_cover: Current cloud cover percentage (0-100).
            current_sun_elevation: Current sun elevation in degrees.
            current_dni: Direct Normal Irradiance in W/m2 (NEW)
            current_dhi: Diffuse Horizontal Irradiance in W/m2 (NEW)
            current_ghi: Global Horizontal Irradiance in W/m2 (NEW)

        @zara
        """
        if group_index >= len(self.group_efficiencies):
            return 1.0

        eff = self.group_efficiencies[group_index]

        learned_eff = self._get_weather_specific_efficiency(eff, hour, current_cloud_cover)

        if current_ghi is not None and current_ghi > 20:
            direct_fraction = self._calculate_direct_fraction(
                current_dni, current_dhi, current_ghi, current_cloud_cover
            )

            if learned_eff <= 1.0:
                DIFFUSE_LOSS_RETENTION = 0.3
                diffuse_efficiency = 1.0 - (1.0 - learned_eff) * DIFFUSE_LOSS_RETENTION
            else:
                DIFFUSE_GAIN_RETENTION = 0.15
                diffuse_efficiency = 1.0 + (learned_eff - 1.0) * DIFFUSE_GAIN_RETENTION

            weighted_efficiency = (
                direct_fraction * learned_eff +
                (1 - direct_fraction) * diffuse_efficiency
            )

            _LOGGER.debug(
                "DNI-weighted efficiency for %s @ %02d:00: "
                "learned=%.3f, direct_fraction=%.2f, diffuse_eff=%.3f, weighted=%.3f",
                eff.name, hour, learned_eff, direct_fraction, diffuse_efficiency, weighted_efficiency
            )

            if hour in eff.hourly_efficiency:
                learned_elevation = eff.avg_sun_elevation_at_learning.get(hour)
                if current_sun_elevation is not None and learned_elevation is not None:
                    weighted_efficiency = self._adjust_efficiency_for_sun_elevation(
                        learned_eff=weighted_efficiency,
                        current_elevation=current_sun_elevation,
                        learned_elevation=learned_elevation,
                        group_name=eff.name,
                        hour=hour,
                    )

            return max(
                self._config.efficiency_conservative_min,
                min(self._config.efficiency_conservative_max, weighted_efficiency)
            )

        if hour in eff.hourly_efficiency:
            learned_cloud = eff.avg_cloud_cover_at_learning.get(hour)
            learned_elevation = eff.avg_sun_elevation_at_learning.get(hour)

            if current_cloud_cover is not None and learned_cloud is not None:
                learned_eff = self._adjust_efficiency_for_weather(
                    learned_eff=learned_eff,
                    current_cloud_cover=current_cloud_cover,
                    learned_cloud_cover=learned_cloud,
                )

            if current_sun_elevation is not None and learned_elevation is not None:
                learned_eff = self._adjust_efficiency_for_sun_elevation(
                    learned_eff=learned_eff,
                    current_elevation=current_sun_elevation,
                    learned_elevation=learned_elevation,
                    group_name=eff.name,
                    hour=hour,
                )

            return learned_eff

        return eff.learned_efficiency_factor

    def _get_weather_specific_efficiency(
        self,
        eff: PanelGroupEfficiency,
        hour: int,
        current_cloud_cover: Optional[float],
    ) -> float:
        """Get weather-specific efficiency if available. @zara"""
        if current_cloud_cover is not None:
            if current_cloud_cover <= WEATHER_CLEAR_MAX_CLOUDS:
                if hour in eff.hourly_efficiency_clear:
                    _LOGGER.debug(
                        "Using CLEAR efficiency for %s @ %02d:00: %.3f (clouds=%.0f%%)",
                        eff.name, hour, eff.hourly_efficiency_clear[hour], current_cloud_cover
                    )
                    return eff.hourly_efficiency_clear[hour]
            elif current_cloud_cover >= WEATHER_CLOUDY_MIN_CLOUDS:
                if hour in eff.hourly_efficiency_cloudy:
                    _LOGGER.debug(
                        "Using CLOUDY efficiency for %s @ %02d:00: %.3f (clouds=%.0f%%)",
                        eff.name, hour, eff.hourly_efficiency_cloudy[hour], current_cloud_cover
                    )
                    return eff.hourly_efficiency_cloudy[hour]
            else:
                clear_eff = eff.hourly_efficiency_clear.get(hour)
                cloudy_eff = eff.hourly_efficiency_cloudy.get(hour)
                if clear_eff is not None and cloudy_eff is not None:
                    blend = (current_cloud_cover - WEATHER_CLEAR_MAX_CLOUDS) / (WEATHER_CLOUDY_MIN_CLOUDS - WEATHER_CLEAR_MAX_CLOUDS)
                    blended_eff = clear_eff * (1 - blend) + cloudy_eff * blend
                    _LOGGER.debug(
                        "Using BLENDED efficiency for %s @ %02d:00: %.3f (clear=%.3f, cloudy=%.3f, blend=%.2f)",
                        eff.name, hour, blended_eff, clear_eff, cloudy_eff, blend
                    )
                    return blended_eff

        if hour in eff.hourly_efficiency:
            return eff.hourly_efficiency[hour]
        return eff.learned_efficiency_factor

    def _calculate_direct_fraction(
        self,
        dni: Optional[float],
        dhi: Optional[float],
        ghi: float,
        cloud_cover: Optional[float] = None,
    ) -> float:
        """Calculate the fraction of direct radiation in total irradiance.

        Args:
            dni: Direct Normal Irradiance in W/m2
            dhi: Diffuse Horizontal Irradiance in W/m2
            ghi: Global Horizontal Irradiance in W/m2
            cloud_cover: Cloud cover percentage as fallback

        Returns:
            Fraction of direct radiation (0.0 = all diffuse, 1.0 = all direct)

        @zara
        """
        if dni is not None and ghi > 0:
            # Direct calculation from DNI/GHI ratio
            # Note: DNI can be > GHI at low sun angles, so we clamp
            direct_fraction = max(0.0, min(1.0, dni / ghi))
        elif dhi is not None and ghi > 0:
            # Calculate from DHI: direct = GHI - DHI (approximately)
            direct_fraction = max(0.0, min(1.0, (ghi - dhi) / ghi))
        elif cloud_cover is not None:
            # Fallback: Estimate direct fraction from cloud cover
            # Clear sky (~0% clouds): ~85% direct
            # Overcast (~100% clouds): ~15% direct
            direct_fraction = max(0.0, min(1.0, 0.15 + (100 - cloud_cover) / 100 * 0.70))
        else:
            # Conservative default
            direct_fraction = 0.5

        return direct_fraction

    def _adjust_efficiency_for_weather(
        self,
        learned_eff: float,
        current_cloud_cover: float,
        learned_cloud_cover: float,
    ) -> float:
        """Adjust learned efficiency based on weather difference.

        If efficiency was learned at high cloud cover but current weather is clear,
        we should expect HIGHER actual production relative to physics model.
        Conversely, if learned at clear sky but current is cloudy, expect LOWER.

        The adjustment is based on the principle that cloud cover affects the
        ratio between diffuse and direct radiation. At high cloud cover, panels
        receive more diffuse light (which affects tilt efficiency differently).

        @zara
        """
        cloud_diff = learned_cloud_cover - current_cloud_cover

        if abs(cloud_diff) < 10:
            return learned_eff

        # Weather cloud factor from config (default 0.5)
        adjustment_factor = 1.0 + (cloud_diff / 100.0) * self._config.weather_cloud_factor

        adjusted_eff = learned_eff * adjustment_factor

        adjusted_eff = max(
            self._config.efficiency_conservative_min,
            min(self._config.efficiency_conservative_max, adjusted_eff)
        )

        _LOGGER.debug(
            "Weather adjustment: learned_eff=%.3f, clouds %.0f%% -> %.0f%%, "
            "adjustment=%.3f, result=%.3f",
            learned_eff, learned_cloud_cover, current_cloud_cover,
            adjustment_factor, adjusted_eff
        )

        return adjusted_eff

    def _adjust_efficiency_for_sun_elevation(
        self,
        learned_eff: float,
        current_elevation: float,
        learned_elevation: float,
        group_name: str = "",
        hour: int = 0,
    ) -> float:
        """Adjust learned efficiency based on sun elevation difference.

        This accounts for seasonal shadow variations:
        - Same hour in summer (high sun) vs winter (low sun) can have vastly
          different shadow patterns
        - If efficiency was learned at high elevation but current is low,
          there may be more shadowing -> reduce efficiency
        - If learned at low elevation but current is high, less shadowing
          -> but be careful not to over-boost

        The correction is asymmetric:
        - Lower sun than learned: Stronger correction (shadows are worse)
        - Higher sun than learned: Weaker correction (can't exceed physics)

        @zara
        """
        elevation_diff = current_elevation - learned_elevation

        # Small differences don't need correction
        if abs(elevation_diff) < 5:
            return learned_eff

        # Calculate correction factor (configurable via learning_config.json)
        # Negative diff = current sun is LOWER than when learned
        # This typically means MORE shadowing in winter
        if elevation_diff < 0:
            # Sun is lower than learned -> expect MORE shadows
            # Stronger correction (default: -15% per 10° lower)
            correction_factor = 1.0 + (elevation_diff / 10.0) * self._config.elevation_low_factor
        else:
            # Sun is higher than learned -> expect LESS shadows
            # Weaker correction (default: +8% per 10° higher, can't exceed physics)
            correction_factor = 1.0 + (elevation_diff / 10.0) * self._config.elevation_high_factor

        # Clamp to reasonable range
        correction_factor = max(self._config.efficiency_conservative_min, min(1.5, correction_factor))

        adjusted_eff = learned_eff * correction_factor

        # Final clamp
        adjusted_eff = max(
            self._config.efficiency_min,
            min(self._config.efficiency_conservative_max, adjusted_eff)
        )

        _LOGGER.debug(
            "Elevation adjustment for %s @ %02d:00: learned_elev=%.1f°, current=%.1f°, "
            "diff=%.1f°, correction=%.3f, eff %.3f -> %.3f",
            group_name, hour, learned_elevation, current_elevation,
            elevation_diff, correction_factor, learned_eff, adjusted_eff
        )

        return adjusted_eff

    def get_panel_group_calculator(self) -> PanelGroupCalculator:
        """Get the panel group calculator. @zara"""
        return self._panel_group_calculator

    def has_groups_with_sensors(self) -> bool:
        """Check if any panel group has an energy sensor configured. @zara"""
        return any(eff.energy_sensor for eff in self.group_efficiencies)

    def get_groups_with_sensors(self) -> List[PanelGroupEfficiency]:
        """Get list of panel groups that have energy sensors configured. @zara"""
        return [eff for eff in self.group_efficiencies if eff.energy_sensor]

    def get_current_estimate(self) -> dict:
        """Get the current efficiency estimates. @zara"""
        return {
            "mode": "panel_groups",
            "total_samples": self.total_samples,
            "last_updated": self.last_updated,
            "groups": [g.to_dict() for g in self.group_efficiencies],
        }

    async def bulk_add_historical_data(
        self,
        historical_samples: List[Dict[str, Any]],
    ) -> dict:
        """Add multiple historical data points for bulk training. @zara"""
        accepted_count = 0
        rejected_count = 0

        _LOGGER.info(
            f"Processing {len(historical_samples)} historical samples "
            f"for PanelGroupEfficiencyLearner..."
        )

        for sample in historical_samples:
            try:
                required_fields = [
                    "timestamp", "sun_elevation_deg", "sun_azimuth_deg",
                    "actual_power_kwh", "ghi_wm2", "dni_wm2", "dhi_wm2",
                    "ambient_temp_c", "cloud_cover_percent"
                ]

                if not all(sample.get(f) is not None for f in required_fields):
                    rejected_count += 1
                    continue

                hour = 12
                ts = sample.get("timestamp", "")
                if "T" in ts:
                    try:
                        hour = int(ts.split("T")[1].split(":")[0])
                    except (ValueError, IndexError):
                        pass

                # Extract per_group_actuals if available in the sample
                per_group_actuals = sample.get("panel_group_actuals")

                was_accepted = self.add_data_point(
                    timestamp=sample["timestamp"],
                    hour=hour,
                    sun_elevation_deg=sample["sun_elevation_deg"],
                    sun_azimuth_deg=sample["sun_azimuth_deg"],
                    actual_power_kwh=sample["actual_power_kwh"],
                    ghi_wm2=sample["ghi_wm2"],
                    dni_wm2=sample["dni_wm2"],
                    dhi_wm2=sample["dhi_wm2"],
                    ambient_temp_c=sample["ambient_temp_c"],
                    cloud_cover_percent=sample["cloud_cover_percent"],
                    per_group_actuals=per_group_actuals,
                )

                if was_accepted:
                    accepted_count += 1
                else:
                    rejected_count += 1

            except Exception as e:
                _LOGGER.debug(f"Error processing sample: {e}")
                rejected_count += 1

        _LOGGER.info(
            f"Bulk import complete: {accepted_count} accepted, {rejected_count} rejected"
        )

        optimization_result = None
        if accepted_count > 0 and self.total_samples >= self.MIN_SAMPLES_FOR_ESTIMATE:
            _LOGGER.info("Running efficiency optimization after bulk import...")
            success = await self.optimize_efficiency()
            if success:
                optimization_result = self.get_current_estimate()

        return {
            "samples_processed": len(historical_samples),
            "accepted": accepted_count,
            "rejected": rejected_count,
            "total_data_points": self.total_samples,
            "optimization_ran": optimization_result is not None,
            "current_estimate": self.get_current_estimate(),
        }
