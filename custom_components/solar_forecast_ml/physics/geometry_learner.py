"""Geometry Learner for Solar Forecast ML Integration V10.0.0 @zara

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
from typing import Optional

from .physics_engine import (
    PhysicsEngine,
    SunPosition,
    IrradianceData,
    PanelGeometry,
)

_LOGGER = logging.getLogger(__name__)


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

        # State file
        self.state_file = data_path / "learned_geometry.json"

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
