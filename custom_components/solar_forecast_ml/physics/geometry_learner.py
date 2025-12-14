"""Geometry Learner for Solar Forecast ML Integration V12.0.0 @zara

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


@dataclass
class PanelGroupEfficiency:
    """Learned efficiency factors for a single panel group. @zara"""
    name: str
    configured_tilt_deg: float
    configured_azimuth_deg: float
    power_kwp: float
    energy_sensor: Optional[str] = None  # Optional kWh sensor for per-group learning
    learned_efficiency_factor: float = 1.0
    learned_shadow_hours: List[int] = field(default_factory=list)
    sample_count: int = 0
    confidence: float = 0.0
    hourly_efficiency: Dict[int, float] = field(default_factory=dict)
    learning_history: List[Dict[str, Any]] = field(default_factory=list)

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
            "learning_history": self.learning_history[-30:],  # Keep last 30 entries
        }
        if self.energy_sensor:
            result["energy_sensor"] = self.energy_sensor
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "PanelGroupEfficiency":
        hourly_eff = data.get("hourly_efficiency", {})
        hourly_eff_int = {int(k): v for k, v in hourly_eff.items()}
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
    MIN_SUN_ELEVATION = 5.0  # Lowered for winter when sun is low @zara

    def __init__(
        self,
        data_path: Path,
        panel_groups: List[Dict[str, Any]],
        skip_load: bool = False,
    ):
        """Initialize the panel group efficiency learner. @zara"""
        self.data_path = data_path
        # State file (in physics/ subdirectory for better organization)
        self.state_file = data_path / "physics" / "learned_panel_group_efficiency.json"

        self._panel_groups_config = panel_groups
        self._panel_group_calculator = PanelGroupCalculator(panel_groups=panel_groups)

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
        if version != "2.0":
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
                "version": "2.0",
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

        if ghi_wm2 < 50:
            return False

        irradiance = IrradianceData(ghi=ghi_wm2, dni=dni_wm2, dhi=dhi_wm2)
        sun = SunPosition(elevation_deg=sun_elevation_deg, azimuth_deg=sun_azimuth_deg)

        physics_result = self._panel_group_calculator.calculate_total_power(
            irradiance, sun, ambient_temp_c
        )

        if physics_result.total_power_kwh > 0.001:
            if per_group_actuals:
                # Per-group learning using actual sensor data
                self._learn_from_per_group_actuals(
                    hour=hour,
                    physics_result=physics_result,
                    per_group_actuals=per_group_actuals,
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

            # Get actual production for this group
            group_actual_kwh = per_group_actuals.get(group_name)

            if group_actual_kwh is None or group_actual_kwh <= 0:
                continue

            # Get physics prediction for this group
            group_predicted_kwh = group_result.power_kwh

            if group_predicted_kwh <= 0.001:
                continue

            # Calculate efficiency ratio for this specific group
            efficiency_ratio = group_actual_kwh / group_predicted_kwh

            # Sanity check: efficiency should be reasonable (0.1 to 2.0)
            if not (0.1 < efficiency_ratio < 2.0):
                _LOGGER.debug(
                    "Unusual efficiency ratio for %s: %.3f (actual=%.4f, predicted=%.4f)",
                    group_name, efficiency_ratio, group_actual_kwh, group_predicted_kwh
                )
                continue

            # Update hourly efficiency with exponential moving average
            if hour not in eff.hourly_efficiency:
                eff.hourly_efficiency[hour] = efficiency_ratio
            else:
                old_eff = eff.hourly_efficiency[hour]
                # Use stronger weight (0.2) for actual sensor data vs. distributed data (0.1)
                eff.hourly_efficiency[hour] = old_eff * 0.8 + efficiency_ratio * 0.2

            eff.sample_count += 1

            _LOGGER.debug(
                "Per-group learning for %s @ %02d:00: actual=%.4f, predicted=%.4f, eff=%.3f",
                group_name, hour, group_actual_kwh, group_predicted_kwh, efficiency_ratio
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
                valid_efficiencies = [
                    e for e in eff.hourly_efficiency.values()
                    if 0.3 < e < 1.5
                ]
                if valid_efficiencies:
                    eff.learned_efficiency_factor = sum(valid_efficiencies) / len(valid_efficiencies)

                shadow_hours = []
                for hour, hour_eff in eff.hourly_efficiency.items():
                    if hour_eff < 0.7:
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

    def get_efficiency_for_hour(self, group_index: int, hour: int) -> float:
        """Get learned efficiency factor for a group at a specific hour. @zara"""
        if group_index >= len(self.group_efficiencies):
            return 1.0

        eff = self.group_efficiencies[group_index]

        if hour in eff.hourly_efficiency:
            return eff.hourly_efficiency[hour]

        return eff.learned_efficiency_factor

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
