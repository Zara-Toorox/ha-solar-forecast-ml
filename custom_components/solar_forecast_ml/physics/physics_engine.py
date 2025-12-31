# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional

_LOGGER = logging.getLogger(__name__)


@dataclass
class SunPosition:
    """Sun position in the sky. @zara"""

    elevation_deg: float
    azimuth_deg: float

    @property
    def zenith_deg(self) -> float:
        """Solar zenith angle (90 - elevation). @zara"""
        return 90.0 - self.elevation_deg

    @property
    def elevation_rad(self) -> float:
        """Elevation in radians. @zara"""
        return math.radians(self.elevation_deg)

    @property
    def azimuth_rad(self) -> float:
        """Azimuth in radians. @zara"""
        return math.radians(self.azimuth_deg)

    @property
    def zenith_rad(self) -> float:
        """Zenith in radians. @zara"""
        return math.radians(self.zenith_deg)


@dataclass
class PanelGeometry:
    """Solar panel geometry configuration. @zara"""

    tilt_deg: float
    azimuth_deg: float

    @property
    def tilt_rad(self) -> float:
        """Tilt in radians. @zara"""
        return math.radians(self.tilt_deg)

    @property
    def azimuth_rad(self) -> float:
        """Azimuth in radians. @zara"""
        return math.radians(self.azimuth_deg)


@dataclass
class IrradianceData:
    """Solar irradiance components from weather data. @zara"""

    ghi: float
    dni: float
    dhi: float

    def is_valid(self) -> bool:
        """Check if irradiance values are physically plausible. @zara"""
        if self.ghi < 0 or self.dni < 0 or self.dhi < 0:
            return False
        if self.ghi > 1400:
            return False
        return True


@dataclass
class POAResult:
    """Result of POA calculation. @zara"""

    poa_total: float
    poa_beam: float
    poa_diffuse: float
    poa_ground: float
    aoi_deg: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization. @zara"""
        return {
            "poa_total_wm2": round(self.poa_total, 2),
            "poa_beam_wm2": round(self.poa_beam, 2),
            "poa_diffuse_wm2": round(self.poa_diffuse, 2),
            "poa_ground_wm2": round(self.poa_ground, 2),
            "aoi_deg": round(self.aoi_deg, 2),
        }


@dataclass
class PowerResult:
    """Result of power calculation. @zara"""

    power_kwh: float
    poa_wm2: float
    temp_correction: float
    system_efficiency: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization. @zara"""
        return {
            "power_kwh": round(self.power_kwh, 4),
            "poa_wm2": round(self.poa_wm2, 2),
            "temp_correction": round(self.temp_correction, 4),
            "system_efficiency": round(self.system_efficiency, 4),
        }


class PhysicsEngine:
    """Physics-based solar power calculation engine. @zara"""

    # Physical constants
    TEMP_COEFFICIENT = -0.004  # Temperature coefficient for crystalline Si (%/K)
    STC_TEMPERATURE = 25.0     # Standard Test Conditions temperature (C)
    NOCT = 45.0                # Nominal Operating Cell Temperature (C)
    NOCT_IRRADIANCE = 800.0    # Irradiance at NOCT conditions (W/m2)
    NOCT_AMBIENT = 20.0        # Ambient temperature at NOCT (C)

    # Default system parameters
    DEFAULT_TILT = 30.0        # Default panel tilt (degrees)
    DEFAULT_AZIMUTH = 180.0    # Default panel azimuth (South)
    DEFAULT_ALBEDO = 0.2       # Ground reflectivity (typical for grass/soil)
    DEFAULT_SYSTEM_EFF = 0.90  # System efficiency (inverter, cables, etc.)

    def __init__(
        self,
        system_capacity_kwp: float,
        panel_tilt_deg: Optional[float] = None,
        panel_azimuth_deg: Optional[float] = None,
        albedo: float = DEFAULT_ALBEDO,
        system_efficiency: float = DEFAULT_SYSTEM_EFF,
    ):
        """Initialize the physics engine. @zara"""
        self.system_capacity_kwp = system_capacity_kwp
        self.albedo = albedo
        self.system_efficiency = system_efficiency

        # Panel geometry (can be updated by GeometryLearner)
        self._geometry = PanelGeometry(
            tilt_deg=panel_tilt_deg if panel_tilt_deg is not None else self.DEFAULT_TILT,
            azimuth_deg=panel_azimuth_deg if panel_azimuth_deg is not None else self.DEFAULT_AZIMUTH,
        )

        # Geometry confidence (0 = default, 1 = fully learned)
        self._geometry_confidence = 0.0 if panel_tilt_deg is None else 1.0

        # PhysicsEngine initialization logging removed to reduce log spam

    @property
    def geometry(self) -> PanelGeometry:
        """Current panel geometry. @zara"""
        return self._geometry

    @property
    def geometry_confidence(self) -> float:
        """Confidence in geometry (0-1). @zara"""
        return self._geometry_confidence

    def update_geometry(
        self,
        tilt_deg: float,
        azimuth_deg: float,
        confidence: float,
    ) -> None:
        """Update panel geometry from GeometryLearner. @zara"""
        self._geometry = PanelGeometry(tilt_deg=tilt_deg, azimuth_deg=azimuth_deg)
        self._geometry_confidence = confidence
        _LOGGER.info(
            "Geometry updated: tilt=%.1f, azimuth=%.1f, confidence=%.2f",
            tilt_deg,
            azimuth_deg,
            confidence,
        )

    def calculate_angle_of_incidence(
        self,
        sun: SunPosition,
        geometry: Optional[PanelGeometry] = None,
    ) -> float:
        """Calculate the Angle of Incidence (AOI) between sun and panel. @zara"""
        if geometry is None:
            geometry = self._geometry

        # Handle sun below horizon
        if sun.elevation_deg <= 0:
            return 90.0  # No direct irradiance

        # Calculate cosine of AOI
        cos_aoi = (
            math.cos(sun.zenith_rad) * math.cos(geometry.tilt_rad)
            + math.sin(sun.zenith_rad)
            * math.sin(geometry.tilt_rad)
            * math.cos(sun.azimuth_rad - geometry.azimuth_rad)
        )

        # Clamp to valid range for acos
        cos_aoi = max(-1.0, min(1.0, cos_aoi))

        # Convert to degrees
        aoi_deg = math.degrees(math.acos(cos_aoi))

        return aoi_deg

    def calculate_poa_irradiance(
        self,
        irradiance: IrradianceData,
        sun: SunPosition,
        geometry: Optional[PanelGeometry] = None,
    ) -> POAResult:
        """Calculate Plane of Array (POA) irradiance using simplified Perez model. @zara"""
        if geometry is None:
            geometry = self._geometry

        # Handle completely invalid data
        if not irradiance.is_valid():
            return POAResult(
                poa_total=0.0,
                poa_beam=0.0,
                poa_diffuse=0.0,
                poa_ground=0.0,
                aoi_deg=90.0,
            )

        # For low sun elevations (0-5°), there can still be diffuse radiation
        # especially for flat panels that can capture twilight/dawn light
        is_twilight = -2.0 <= sun.elevation_deg <= 5.0
        sun_below_horizon = sun.elevation_deg <= 0

        # No direct beam when sun is below horizon
        poa_beam = 0.0

        if not sun_below_horizon:
            # Calculate AOI for direct beam
            aoi_deg = self.calculate_angle_of_incidence(sun, geometry)
            aoi_rad = math.radians(aoi_deg)

            # POA beam component
            # Only positive when sun is in front of panel (AOI < 90)
            if aoi_deg < 90:
                poa_beam = irradiance.dni * math.cos(aoi_rad)
        else:
            aoi_deg = 90.0

        # POA diffuse component (isotropic sky model)
        # This works even at very low sun angles - flat panels can capture diffuse light
        # For twilight conditions, scale down diffuse based on sun angle
        diffuse_factor = 1.0
        if is_twilight and sun.elevation_deg < 3.0:
            # Gradual reduction for very low sun angles (from 3° down to -2°)
            diffuse_factor = max(0.0, (sun.elevation_deg + 2.0) / 5.0)

        poa_diffuse = irradiance.dhi * (1 + math.cos(geometry.tilt_rad)) / 2 * diffuse_factor

        # POA ground reflection component
        poa_ground = irradiance.ghi * self.albedo * (1 - math.cos(geometry.tilt_rad)) / 2 * diffuse_factor

        # Total POA
        poa_total = max(0.0, poa_beam + poa_diffuse + poa_ground)

        return POAResult(
            poa_total=poa_total,
            poa_beam=max(0.0, poa_beam),
            poa_diffuse=max(0.0, poa_diffuse),
            poa_ground=max(0.0, poa_ground),
            aoi_deg=aoi_deg,
        )

    def calculate_cell_temperature(
        self,
        ambient_temp_c: float,
        poa_wm2: float,
    ) -> float:
        """Estimate cell temperature using NOCT model. @zara"""
        if poa_wm2 <= 0:
            return ambient_temp_c

        temp_rise = (self.NOCT - self.NOCT_AMBIENT) * (poa_wm2 / self.NOCT_IRRADIANCE)
        cell_temp = ambient_temp_c + temp_rise

        return cell_temp

    def calculate_temperature_correction(
        self,
        cell_temp_c: float,
    ) -> float:
        """Calculate temperature correction factor for power output. @zara"""
        correction = 1.0 + self.TEMP_COEFFICIENT * (cell_temp_c - self.STC_TEMPERATURE)

        # Clamp to reasonable range
        correction = max(0.5, min(1.2, correction))

        return correction

    def calculate_power_output(
        self,
        irradiance: IrradianceData,
        sun: SunPosition,
        ambient_temp_c: float,
        geometry: Optional[PanelGeometry] = None,
    ) -> PowerResult:
        """Calculate expected power output for one hour. @zara"""
        if geometry is None:
            geometry = self._geometry

        # Calculate POA
        poa = self.calculate_poa_irradiance(irradiance, sun, geometry)

        if poa.poa_total <= 0:
            return PowerResult(
                power_kwh=0.0,
                poa_wm2=0.0,
                temp_correction=1.0,
                system_efficiency=self.system_efficiency,
            )

        # Calculate cell temperature
        cell_temp = self.calculate_cell_temperature(ambient_temp_c, poa.poa_total)

        # Calculate temperature correction
        temp_correction = self.calculate_temperature_correction(cell_temp)

        # Calculate power (kWh for 1 hour)
        # POA in W/m2, divide by 1000 to get kW/kWp, multiply by kWp
        power_kwh = (
            (poa.poa_total / 1000.0)
            * self.system_capacity_kwp
            * temp_correction
            * self.system_efficiency
        )

        # Ensure non-negative
        power_kwh = max(0.0, power_kwh)

        return PowerResult(
            power_kwh=power_kwh,
            poa_wm2=poa.poa_total,
            temp_correction=temp_correction,
            system_efficiency=self.system_efficiency,
        )

    def calculate_hourly_forecast(
        self,
        weather_data: dict,
        astronomy_data: dict,
    ) -> dict:
        """Calculate power forecast from weather and astronomy data. @zara"""
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

        # Calculate power
        result = self.calculate_power_output(irradiance, sun, ambient_temp)

        # Calculate POA for detailed output
        poa = self.calculate_poa_irradiance(irradiance, sun)

        return {
            "physics_prediction_kwh": round(result.power_kwh, 4),
            "poa_irradiance": poa.to_dict(),
            "power_calculation": result.to_dict(),
            "geometry": {
                "tilt_deg": self._geometry.tilt_deg,
                "azimuth_deg": self._geometry.azimuth_deg,
                "confidence": self._geometry_confidence,
            },
            "input": {
                "ghi_wm2": ghi,
                "dni_wm2": dni,
                "dhi_wm2": dhi,
                "sun_elevation_deg": elevation,
                "sun_azimuth_deg": azimuth,
                "ambient_temp_c": ambient_temp,
            },
        }

    def estimate_geometry_from_production(
        self,
        sun: SunPosition,
        actual_power_kwh: float,
        irradiance: IrradianceData,
        ambient_temp_c: float,
    ) -> Optional[tuple[float, float]]:
        """Estimate panel geometry from actual production data. @zara"""

        if actual_power_kwh <= 0 or irradiance.ghi <= 50:
            return None

        if sun.elevation_deg < 10:
            return None  # Too low for reliable estimation

        # Estimate POA needed for this power
        # P = (POA / 1000) * kWp * eta_temp * eta_system
        # POA = P * 1000 / (kWp * eta_temp * eta_system)

        # Assume default temperature for estimation
        cell_temp = self.calculate_cell_temperature(ambient_temp_c, irradiance.ghi)
        temp_correction = self.calculate_temperature_correction(cell_temp)

        poa_needed = (
            actual_power_kwh * 1000.0
            / (self.system_capacity_kwp * temp_correction * self.system_efficiency)
        )

        # This is just an indication - full optimization done by GeometryLearner
        return (poa_needed, sun.azimuth_deg)  # Placeholder
