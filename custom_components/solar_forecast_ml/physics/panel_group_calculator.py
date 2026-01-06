# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .physics_engine import (
    IrradianceData,
    PanelGeometry,
    PhysicsEngine,
    POAResult,
    PowerResult,
    SunPosition,
)
from .physics_calibrator import PhysicsCalibrator

_LOGGER = logging.getLogger(__name__)


def _load_physics_defaults_from_config_sync(data_path: Optional[Path]) -> dict:
    """Load physics defaults from learning_config.json synchronously (for executor). @zara"""
    defaults = {
        "albedo": 0.2,
        "system_efficiency": 0.90
    }

    if data_path is None:
        return defaults

    config_file = data_path / "physics" / "learning_config.json"
    if not config_file.exists():
        return defaults

    try:
        with open(config_file, "r") as f:
            data = json.load(f)

        physics = data.get("physics_defaults", {})
        defaults["albedo"] = physics.get("albedo", 0.2)
        defaults["system_efficiency"] = physics.get("system_efficiency", 0.90)

        _LOGGER.debug(
            "Loaded physics defaults from config: albedo=%.2f, system_eff=%.2f",
            defaults["albedo"], defaults["system_efficiency"]
        )
    except Exception as e:
        _LOGGER.warning("Failed to load physics defaults from config: %s", e)

    return defaults


async def _load_physics_defaults_from_config(data_path: Optional[Path]) -> dict:
    """Load physics defaults from learning_config.json asynchronously. @zara"""
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _load_physics_defaults_from_config_sync, data_path)


@dataclass
class PanelGroup:
    """Represents a group of solar panels with the same orientation. @zara"""

    name: str
    power_wp: float  # Total power in Watts peak
    azimuth_deg: float  # Panel azimuth (0=North, 90=East, 180=South, 270=West)
    tilt_deg: float  # Panel tilt angle in degrees (0=horizontal, 90=vertical)
    energy_sensor: Optional[str] = None  # Optional kWh sensor for per-group learning

    @property
    def power_kwp(self) -> float:
        """Power in kWp. @zara"""
        return self.power_wp / 1000.0

    @property
    def geometry(self) -> PanelGeometry:
        """Get panel geometry for this group. @zara"""
        return PanelGeometry(tilt_deg=self.tilt_deg, azimuth_deg=self.azimuth_deg)

    @property
    def has_energy_sensor(self) -> bool:
        """Check if this group has an energy sensor configured. @zara"""
        return self.energy_sensor is not None and len(self.energy_sensor) > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization. @zara"""
        result = {
            "name": self.name,
            "power_wp": self.power_wp,
            "power_kwp": round(self.power_kwp, 3),
            "azimuth_deg": self.azimuth_deg,
            "tilt_deg": self.tilt_deg,
        }
        if self.energy_sensor:
            result["energy_sensor"] = self.energy_sensor
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "PanelGroup":
        """Create PanelGroup from dictionary. @zara"""
        return cls(
            name=data.get("name", "Unknown"),
            power_wp=float(data.get("power_wp", 0)),
            azimuth_deg=float(data.get("azimuth", 180)),
            tilt_deg=float(data.get("tilt", 30)),
            energy_sensor=data.get("energy_sensor"),
        )


@dataclass
class PanelGroupResult:
    """Result of power calculation for a single panel group. @zara"""

    group: PanelGroup
    power_kwh: float
    poa_result: POAResult
    contribution_percent: float = 0.0
    actual_kwh: Optional[float] = None  # Actual production from energy sensor

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization. @zara"""
        result = {
            "group_name": self.group.name,
            "power_wp": self.group.power_wp,
            "azimuth_deg": self.group.azimuth_deg,
            "tilt_deg": self.group.tilt_deg,
            "power_kwh": round(self.power_kwh, 4),
            "contribution_percent": round(self.contribution_percent, 1),
            "poa_total_wm2": round(self.poa_result.poa_total, 2),
            "aoi_deg": round(self.poa_result.aoi_deg, 1),
        }
        if self.group.energy_sensor:
            result["energy_sensor"] = self.group.energy_sensor
        if self.actual_kwh is not None:
            result["actual_kwh"] = round(self.actual_kwh, 4)
        return result


@dataclass
class MultiGroupResult:
    """Combined result of power calculation for all panel groups. @zara"""

    total_power_kwh: float
    group_results: list[PanelGroupResult] = field(default_factory=list)
    total_capacity_kwp: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization. @zara"""
        return {
            "total_power_kwh": round(self.total_power_kwh, 4),
            "total_capacity_kwp": round(self.total_capacity_kwp, 3),
            "group_count": len(self.group_results),
            "groups": [r.to_dict() for r in self.group_results],
        }


class PanelGroupCalculator:
    """Calculator for multi-group solar panel systems. @zara

    This class handles the calculation of power output for solar systems
    with multiple panel groups that have different orientations.

    Example:
        - Group 1: 1200W facing South (180°) at 30° tilt
        - Group 2: 900W facing West (270°) at 10° tilt

    Each group's output is calculated separately based on its orientation
    relative to the sun position, then summed for total system output.
    """

    # Physical constants (inherited from PhysicsEngine)
    TEMP_COEFFICIENT = -0.004  # Temperature coefficient for crystalline Si (%/K)
    STC_TEMPERATURE = 25.0  # Standard Test Conditions temperature (C)
    # Default values (can be overridden via learning_config.json)
    DEFAULT_ALBEDO = 0.2  # Ground reflectivity
    DEFAULT_SYSTEM_EFF = 0.90  # System efficiency (inverter, cables, etc.)

    def __init__(
        self,
        panel_groups: list[PanelGroup] | list[dict] | None = None,
        albedo: Optional[float] = None,
        system_efficiency: Optional[float] = None,
        data_path: Optional[Path] = None,
        _skip_config_load: bool = False,
    ):
        """Initialize the panel group calculator. @zara

        Args:
            panel_groups: List of PanelGroup objects or dicts with group config
            albedo: Ground reflectivity (default from config or 0.2)
            system_efficiency: Overall system efficiency (default from config or 0.90)
            data_path: Optional path to data directory for loading config
            _skip_config_load: Internal flag - if True, skip sync config loading
                               (used by async_create factory method)
        """
        # Load defaults from config if available
        if not _skip_config_load and (albedo is None or system_efficiency is None):
            config_defaults = _load_physics_defaults_from_config_sync(data_path)
        else:
            config_defaults = {"albedo": 0.2, "system_efficiency": 0.90}

        self.albedo = albedo if albedo is not None else config_defaults["albedo"]
        self.system_efficiency = (
            system_efficiency if system_efficiency is not None
            else config_defaults["system_efficiency"]
        )
        self._groups: list[PanelGroup] = []
        self._calibrator: Optional[PhysicsCalibrator] = None
        self._calibration_enabled: bool = True  # Can be disabled for testing

        if panel_groups:
            self.set_panel_groups(panel_groups)

        _LOGGER.debug(
            "PanelGroupCalculator initialized with %d groups, total %.2f kWp, albedo=%.2f, sys_eff=%.2f",
            len(self._groups),
            self.total_capacity_kwp,
            self.albedo,
            self.system_efficiency,
        )

    @classmethod
    async def async_create(
        cls,
        panel_groups: list["PanelGroup"] | list[dict] | None = None,
        albedo: Optional[float] = None,
        system_efficiency: Optional[float] = None,
        data_path: Optional[Path] = None,
    ) -> "PanelGroupCalculator":
        """Async factory method to create PanelGroupCalculator without blocking. @zara

        Use this method when creating PanelGroupCalculator from an async context
        (e.g., Home Assistant event loop) to avoid blocking file I/O.

        Args:
            panel_groups: List of PanelGroup objects or dicts with group config
            albedo: Ground reflectivity (default from config or 0.2)
            system_efficiency: Overall system efficiency (default from config or 0.90)
            data_path: Optional path to data directory for loading config

        Returns:
            Initialized PanelGroupCalculator instance
        """
        # Load config asynchronously first
        if albedo is None or system_efficiency is None:
            config_defaults = await _load_physics_defaults_from_config(data_path)
            if albedo is None:
                albedo = config_defaults["albedo"]
            if system_efficiency is None:
                system_efficiency = config_defaults["system_efficiency"]

        # Now create instance with pre-loaded values (skip sync config load)
        return cls(
            panel_groups=panel_groups,
            albedo=albedo,
            system_efficiency=system_efficiency,
            data_path=data_path,
            _skip_config_load=True,
        )

    def set_panel_groups(self, panel_groups: list[PanelGroup] | list[dict]) -> None:
        """Set or update panel groups configuration. @zara"""
        self._groups = []

        for idx, group in enumerate(panel_groups):
            if isinstance(group, PanelGroup):
                self._groups.append(group)
            elif isinstance(group, dict):
                # Parse from dict format
                self._groups.append(
                    PanelGroup(
                        name=group.get("name", f"Gruppe {idx + 1}"),
                        power_wp=float(group.get("power_wp", 0)),
                        azimuth_deg=float(group.get("azimuth", 180)),
                        tilt_deg=float(group.get("tilt", 30)),
                        energy_sensor=group.get("energy_sensor"),
                    )
                )

        _LOGGER.info(
            "Panel groups updated: %d groups, total %.2f kWp",
            len(self._groups),
            self.total_capacity_kwp,
        )

    @property
    def groups(self) -> list[PanelGroup]:
        """Get all panel groups. @zara"""
        return self._groups

    @property
    def group_count(self) -> int:
        """Get number of panel groups. @zara"""
        return len(self._groups)

    @property
    def total_capacity_wp(self) -> float:
        """Total system capacity in Watts peak. @zara"""
        return sum(g.power_wp for g in self._groups)

    @property
    def total_capacity_kwp(self) -> float:
        """Total system capacity in kWp. @zara"""
        return self.total_capacity_wp / 1000.0

    @property
    def has_groups(self) -> bool:
        """Check if panel groups are configured. @zara"""
        return len(self._groups) > 0

    def set_calibrator(self, calibrator: PhysicsCalibrator) -> None:
        """Set the physics calibrator for self-learning corrections. @zara

        Args:
            calibrator: PhysicsCalibrator instance
        """
        self._calibrator = calibrator
        _LOGGER.info("PhysicsCalibrator attached to PanelGroupCalculator")

    def get_calibration_factor(
        self,
        group_name: str,
        hour: Optional[int] = None,
        cloud_cover: Optional[float] = None,
        sun_elevation: Optional[float] = None,
    ) -> float:
        """Get calibration correction factor for a group. @zara

        Args:
            group_name: Name of the panel group
            hour: Optional hour for hourly-specific factor
            cloud_cover: Optional cloud cover (0-100) for weather-bucket-specific factor
            sun_elevation: Optional sun elevation (degrees) for LOW_SUN bucket detection

        Returns:
            Correction factor (1.0 = no correction)
        """
        if not self._calibration_enabled or self._calibrator is None:
            return 1.0
        return self._calibrator.get_correction_factor(group_name, hour, cloud_cover, sun_elevation)

    def calculate_group_power(
        self,
        group: PanelGroup,
        irradiance: IrradianceData,
        sun: SunPosition,
        ambient_temp_c: float,
        hour: Optional[int] = None,
        cloud_cover: Optional[float] = None,
    ) -> PanelGroupResult:
        """Calculate power output for a single panel group with calibration. @zara

        Args:
            group: The panel group to calculate for
            irradiance: Solar irradiance components (GHI, DNI, DHI)
            sun: Current sun position
            ambient_temp_c: Ambient temperature in Celsius
            hour: Optional hour for hourly-specific calibration factor
            cloud_cover: Optional cloud cover (0-100) for weather-bucket-specific factor

        Returns:
            PanelGroupResult with power output and details
        """
        # Create a temporary PhysicsEngine for this group
        engine = PhysicsEngine(
            system_capacity_kwp=group.power_kwp,
            panel_tilt_deg=group.tilt_deg,
            panel_azimuth_deg=group.azimuth_deg,
            albedo=self.albedo,
            system_efficiency=self.system_efficiency,
        )

        # Calculate POA irradiance for this group's orientation
        poa_result = engine.calculate_poa_irradiance(irradiance, sun, group.geometry)

        # Calculate power output
        power_result = engine.calculate_power_output(
            irradiance, sun, ambient_temp_c, group.geometry
        )

        # Apply calibration correction factor (with weather bucket + LOW_SUN support)
        # Pass sun elevation from SunPosition for LOW_SUN bucket detection
        raw_power = power_result.power_kwh
        calibration_factor = self.get_calibration_factor(
            group.name, hour, cloud_cover, sun.elevation_deg
        )
        calibrated_power = raw_power * calibration_factor


        return PanelGroupResult(
            group=group,
            power_kwh=calibrated_power,
            poa_result=poa_result,
        )

    def calculate_total_power(
        self,
        irradiance: IrradianceData,
        sun: SunPosition,
        ambient_temp_c: float,
        hour: Optional[int] = None,
        cloud_cover: Optional[float] = None,
    ) -> MultiGroupResult:
        """Calculate total power output for all panel groups with calibration. @zara

        Args:
            irradiance: Solar irradiance components (GHI, DNI, DHI)
            sun: Current sun position
            ambient_temp_c: Ambient temperature in Celsius
            hour: Optional hour for hourly-specific calibration factors
            cloud_cover: Optional cloud cover (0-100) for weather-bucket-specific factors

        Returns:
            MultiGroupResult with total power and per-group breakdown
        """
        if not self._groups:
            return MultiGroupResult(
                total_power_kwh=0.0,
                group_results=[],
                total_capacity_kwp=0.0,
            )

        group_results: list[PanelGroupResult] = []
        total_power_kwh = 0.0

        for group in self._groups:
            result = self.calculate_group_power(
                group, irradiance, sun, ambient_temp_c, hour, cloud_cover
            )
            group_results.append(result)
            total_power_kwh += result.power_kwh

        # Calculate contribution percentages
        if total_power_kwh > 0:
            for result in group_results:
                result.contribution_percent = (result.power_kwh / total_power_kwh) * 100

        return MultiGroupResult(
            total_power_kwh=total_power_kwh,
            group_results=group_results,
            total_capacity_kwp=self.total_capacity_kwp,
        )

    def calculate_hourly_forecast(
        self,
        weather_data: dict,
        astronomy_data: dict,
    ) -> dict:
        """Calculate power forecast from weather and astronomy data for all groups. @zara

        This is the main entry point for forecast calculations with panel groups.

        Args:
            weather_data: Dict with weather info (ghi, dni, dhi, temperature, etc.)
            astronomy_data: Dict with sun position (elevation_deg, azimuth_deg)

        Returns:
            Dict with forecast results including per-group breakdown
        """
        # Extract irradiance
        ghi = weather_data.get("ghi", 0.0) or 0.0
        dni = weather_data.get("direct_radiation", 0.0) or 0.0
        dhi = weather_data.get("diffuse_radiation", 0.0) or 0.0

        irradiance = IrradianceData(ghi=ghi, dni=dni, dhi=dhi)

        # Extract sun position
        elevation = astronomy_data.get("elevation_deg", 0.0) or 0.0
        azimuth = astronomy_data.get("azimuth_deg", 180.0) or 180.0

        sun = SunPosition(elevation_deg=elevation, azimuth_deg=azimuth)

        # Extract temperature
        ambient_temp = weather_data.get("temperature", 15.0) or 15.0

        # Calculate for all groups
        result = self.calculate_total_power(irradiance, sun, ambient_temp)

        return {
            "physics_prediction_kwh": round(result.total_power_kwh, 4),
            "total_capacity_kwp": round(result.total_capacity_kwp, 3),
            "group_count": len(result.group_results),
            "groups": result.to_dict()["groups"],
            "input": {
                "ghi_wm2": ghi,
                "dni_wm2": dni,
                "dhi_wm2": dhi,
                "sun_elevation_deg": elevation,
                "sun_azimuth_deg": azimuth,
                "ambient_temp_c": ambient_temp,
            },
        }

    def get_optimal_hours_by_group(
        self,
        hourly_sun_positions: list[dict],
        irradiance_data: list[IrradianceData],
        ambient_temp_c: float = 15.0,
    ) -> dict[str, int]:
        """Find the optimal production hour for each panel group. @zara

        Different panel orientations have different peak production times:
        - East-facing panels peak in the morning
        - South-facing panels peak around noon
        - West-facing panels peak in the afternoon

        Args:
            hourly_sun_positions: List of sun positions by hour (0-23)
            irradiance_data: List of irradiance data by hour
            ambient_temp_c: Ambient temperature

        Returns:
            Dict mapping group name to optimal hour (0-23)
        """
        optimal_hours: dict[str, int] = {}

        for group in self._groups:
            max_power = 0.0
            best_hour = 12  # Default to noon

            for hour, (sun_data, irrad) in enumerate(
                zip(hourly_sun_positions, irradiance_data)
            ):
                sun = SunPosition(
                    elevation_deg=sun_data.get("elevation_deg", 0),
                    azimuth_deg=sun_data.get("azimuth_deg", 180),
                )

                result = self.calculate_group_power(group, irrad, sun, ambient_temp_c)

                if result.power_kwh > max_power:
                    max_power = result.power_kwh
                    best_hour = hour

            optimal_hours[group.name] = best_hour

        return optimal_hours

    def to_dict(self) -> dict:
        """Convert calculator state to dictionary. @zara"""
        return {
            "total_capacity_kwp": round(self.total_capacity_kwp, 3),
            "group_count": self.group_count,
            "albedo": self.albedo,
            "system_efficiency": self.system_efficiency,
            "groups": [g.to_dict() for g in self._groups],
        }

    def __repr__(self) -> str:
        """String representation. @zara"""
        return (
            f"PanelGroupCalculator("
            f"groups={self.group_count}, "
            f"total={self.total_capacity_kwp:.2f}kWp)"
        )
