# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from homeassistant.util import dt as dt_util

from ..astronomy.astronomy_cache_manager import get_cache_manager
from ..core.core_helpers import SafeDateTimeUtil as dt_util_safe
from ..data.data_weather_corrector import WeatherForecastCorrector
from ..ai import SeasonalAdjuster, AIPredictor
from ..physics import (
    PhysicsEngine,
    PanelGroupCalculator,
    PanelGroup,
    IrradianceData,
    SunPosition,
    PhysicsCalibrator,
)
from .forecast_strategy_base import ForecastResult, ForecastStrategy
from .forecast_weather_calculator import WeatherCalculator

_LOGGER = logging.getLogger(__name__)

# Hybrid prediction constants
MAX_AI_PHYSICS_DEVIATION = 0.5  # 50% max deviation before fallback to physics

# Low sun angle production constants
# At low sun angles, weather APIs systematically underestimate radiation.
# The effect is CLOUD-DEPENDENT:
# - Clear sky: APIs severely underestimate (up to 400% error at 10°)
# - Cloudy: Radiation is already diffuse, API estimates are more accurate
#
# Original twilight range (-2° to 5°) was too narrow - real underestimation extends to ~15°
TWILIGHT_MIN_ELEVATION = -2.0  # Degrees - civil twilight starts at -6°, but panels see light earlier
TWILIGHT_MAX_ELEVATION = 5.0   # Degrees - classic twilight (API reports ~0)
LOW_SUN_MAX_ELEVATION = 15.0   # Degrees - extended range for clear sky correction
TWILIGHT_MIN_DHI_WM2 = 15.0    # Minimum diffuse radiation estimate for twilight (W/m²)
TWILIGHT_MIN_GHI_WM2 = 8.0     # Minimum GHI for twilight conditions (W/m²)

# Cloud-dependent low sun correction thresholds
LOW_SUN_CLOUD_THRESHOLD = 50   # Above this cloud%, skip low-sun correction (already diffuse)
LOW_SUN_MORNING_HOURS = range(5, 12)  # Only apply LOW_SUN correction for morning hours (5-11)


class RuleBasedForecastStrategy(ForecastStrategy):
    """AI-based forecast strategy with physics fallback @zara"""

    def __init__(
        self,
        weather_calculator: WeatherCalculator,
        solar_capacity: float,
        orchestrator: Optional[Any] = None,
        panel_groups: Optional[List[Dict[str, Any]]] = None,
        ai_predictor: Optional[AIPredictor] = None,
    ):
        """Initialize strategy @zara"""
        super().__init__("rule_based")
        self.weather_calculator = weather_calculator
        self.solar_capacity = solar_capacity
        self.orchestrator = orchestrator
        self.ai_predictor = ai_predictor

        self.weather_corrector: Optional[WeatherForecastCorrector] = None
        self._corrector_init_attempted = False

        self.physics_engine: Optional[PhysicsEngine] = None
        self._physics_init_attempted = False

        self.panel_group_calculator: Optional[PanelGroupCalculator] = None
        self._panel_groups_config = panel_groups or []
        self._panel_groups_initialized = False

        # Physics calibrator for self-learning corrections
        self._physics_calibrator: Optional[PhysicsCalibrator] = None

        self.seasonal: Optional[SeasonalAdjuster] = None
        self._use_ai = False

        if panel_groups:
            _LOGGER.info(f"AI Hybrid Strategy: {len(panel_groups)} panel groups configured")
        if ai_predictor:
            _LOGGER.info("AI Hybrid Strategy: Local AI available")

    def set_physics_calibrator(self, calibrator: PhysicsCalibrator) -> None:
        """Set the physics calibrator for self-learning corrections @zara"""
        self._physics_calibrator = calibrator
        # If PanelGroupCalculator already exists, connect it
        if self.panel_group_calculator:
            self.panel_group_calculator.set_calibrator(calibrator)
            _LOGGER.info("PhysicsCalibrator connected to existing PanelGroupCalculator")
        else:
            _LOGGER.debug("PhysicsCalibrator stored, will connect when PanelGroupCalculator is created")

    def is_available(self) -> bool:
        """Always available as fallback @zara"""
        return True

    def get_priority(self) -> int:
        """Priority lower than AI strategy @zara"""
        return 50

    async def calculate_forecast(
        self,
        hourly_weather_forecast: List[Dict[str, Any]],
        sensor_data: Dict[str, Any],
        correction_factor: float,
        **kwargs,
    ) -> ForecastResult:
        """Calculate forecast using physics engine @zara"""
        try:
            capacity = float(sensor_data.get("solar_capacity", self.solar_capacity))
            if capacity <= 0:
                capacity = self.solar_capacity

            await self._ensure_components_initialized()

            if not hourly_weather_forecast:
                return self._empty_result()

            now = dt_util_safe.now()
            today = now.date()
            tomorrow = today + timedelta(days=1)
            day_after = today + timedelta(days=2)

            today_kwh = 0.0
            tomorrow_kwh = 0.0
            day_after_kwh = 0.0
            hourly_values = []

            best_hour = None
            best_hour_kwh = 0.0

            for hour_data in hourly_weather_forecast:
                hour_dt = self._parse_datetime(hour_data)
                if not hour_dt:
                    continue

                hour_date = hour_dt.date()
                hour = hour_dt.hour

                production, group_details = await self._calculate_hour_production(
                    hour_data, hour_dt, capacity
                )

                if hour_date == today:
                    if hour >= now.hour:
                        today_kwh += production
                        if production > best_hour_kwh:
                            best_hour_kwh = production
                            best_hour = hour
                elif hour_date == tomorrow:
                    tomorrow_kwh += production
                elif hour_date == day_after:
                    day_after_kwh += production

                hourly_entry = {
                    "hour": hour,
                    "datetime": hour_dt.isoformat(),
                    "date": hour_date.isoformat(),
                    "production_kwh": round(production, 3),
                }
                if group_details:
                    # Store detailed group info for diagnostics
                    hourly_entry["groups"] = group_details
                    # Store per-group predictions for TinyLSTM AI learning
                    hourly_entry["panel_group_predictions"] = {
                        g["name"]: round(g["power_kwh"], 4)
                        for g in group_details
                    }
                hourly_values.append(hourly_entry)

            month = now.month
            if self.seasonal:
                seasonal_factor = self.seasonal.get_factor(month)
                today_kwh *= seasonal_factor
                tomorrow_kwh *= seasonal_factor
                day_after_kwh *= seasonal_factor

            today_kwh *= correction_factor
            tomorrow_kwh *= correction_factor
            day_after_kwh *= correction_factor

            max_daily = capacity * 7.0
            today_kwh = min(today_kwh, max_daily)
            tomorrow_kwh = min(tomorrow_kwh, max_daily)
            day_after_kwh = min(day_after_kwh, max_daily)

            return ForecastResult(
                forecast_today=round(today_kwh, 2),
                forecast_tomorrow=round(tomorrow_kwh, 2),
                forecast_day_after_tomorrow=round(day_after_kwh, 2),
                confidence_today=70.0,
                confidence_tomorrow=60.0,
                confidence_day_after=50.0,
                method="rule_based" if not self._use_ai else "tinylstm_physics",
                calibrated=bool(self.panel_group_calculator and self.panel_group_calculator.has_groups),
                best_hour_today=best_hour,
                best_hour_production_kwh=round(best_hour_kwh, 3),
                hourly_values=hourly_values,
                model_accuracy=None,
                forecast_today_raw=today_kwh,
                safeguard_applied_today=False,
            )

        except Exception as e:
            _LOGGER.error(f"Forecast calculation failed: {e}")
            return self._empty_result()

    async def _ensure_components_initialized(self):
        """Initialize AI and physics components @zara"""
        if self.ai_predictor and self.ai_predictor.is_ready():
            self._use_ai = True
            _LOGGER.debug("Using Hybrid-AI for predictions")
        else:
            self._use_ai = False

        if not self._physics_init_attempted:
            self._physics_init_attempted = True
            try:
                self.physics_engine = PhysicsEngine(
                    system_capacity_kwp=self.solar_capacity
                )
                _LOGGER.debug("Physics engine initialized")
            except Exception as e:
                _LOGGER.warning(f"Physics engine init failed: {e}")

        # Initialize PanelGroupCalculator if groups configured
        if not self._panel_groups_initialized and self._panel_groups_config:
            self._panel_groups_initialized = True
            try:
                self.panel_group_calculator = PanelGroupCalculator(
                    panel_groups=self._panel_groups_config
                )
                _LOGGER.info(
                    f"PanelGroupCalculator initialized: {self.panel_group_calculator.group_count} groups, "
                    f"{self.panel_group_calculator.total_capacity_kwp:.2f} kWp"
                )
                # Connect PhysicsCalibrator if available
                if self._physics_calibrator:
                    self.panel_group_calculator.set_calibrator(self._physics_calibrator)
                    _LOGGER.info("PhysicsCalibrator connected to PanelGroupCalculator for self-learning")
            except Exception as e:
                _LOGGER.warning(f"PanelGroupCalculator init failed: {e}")

        if not self._corrector_init_attempted and self.orchestrator:
            self._corrector_init_attempted = True
            if hasattr(self.orchestrator, 'data_manager'):
                try:
                    self.weather_corrector = WeatherForecastCorrector(
                        self.orchestrator.hass,
                        self.orchestrator.data_manager
                    )
                except Exception as e:
                    _LOGGER.warning(f"Weather corrector init failed: {e}")

    async def _calculate_hour_production(
        self,
        hour_data: Dict[str, Any],
        hour_dt: datetime,
        capacity: float
    ) -> tuple[float, Optional[List[Dict[str, Any]]]]:
        """Hybrid Physics + ML prediction @zara

        Always calculates physics baseline, then blends with ML if:
        1. ML is ready and has positive R²
        2. ML prediction doesn't deviate >50% from physics

        Returns:
            Tuple of (total_production_kwh, group_details or None)
        """
        try:
            hour = hour_dt.hour
            clouds = float(hour_data.get("cloud_cover", hour_data.get("clouds", 50)))

            # STEP 1: Always calculate physics baseline
            physics_pred, physics_groups = self._calculate_physics_production(
                hour_data, hour_dt
            )

            # Get astronomy for confidence calculation
            astronomy_data = self._get_astronomy_for_hour(hour_dt)
            sun_elevation = astronomy_data.get("sun_elevation_deg", 30)

            # STEP 2: Try AI prediction if available
            ai_pred = 0.0
            ai_groups = None
            ai_confidence = 0.0

            if self._use_ai and self.ai_predictor:
                base_r2 = self.ai_predictor.current_accuracy or 0.0

                # Calculate dynamic confidence for this hour
                ai_confidence = self._calculate_hourly_ai_confidence(
                    base_r2, clouds, sun_elevation, hour
                )

                if ai_confidence > 0:
                    weather_data = {
                        "temperature": hour_data.get("temperature", 20),
                        "solar_radiation_wm2": hour_data.get("solar_radiation", hour_data.get("ghi", 0)),
                        "wind": hour_data.get("wind_speed", 3),
                        "humidity": hour_data.get("humidity", 70),
                        "rain": hour_data.get("precipitation", 0),
                        "clouds": clouds,
                        "dni": hour_data.get("direct_radiation", hour_data.get("dni", 0)),
                    }

                    group_predictions = await self.ai_predictor.predict_hour_per_group(
                        hour=hour,
                        weather_data=weather_data,
                        astronomy_data=astronomy_data,
                    )

                    if group_predictions:
                        ai_pred = sum(gp.power_kwh for gp in group_predictions)
                        ai_groups = []
                        for gp in group_predictions:
                            contribution = (gp.power_kwh / ai_pred * 100) if ai_pred > 0 else 0
                            ai_groups.append({
                                "name": gp.group_name,
                                "power_kwh": round(gp.power_kwh, 4),
                                "contribution_percent": round(contribution, 1),
                                "azimuth_deg": gp.azimuth,
                                "tilt_deg": gp.tilt,
                                "capacity_kwp": gp.capacity_kwp,
                            })

            # STEP 3: Check for deviation fallback
            use_hybrid = False
            if ai_confidence > 0 and ai_pred > 0 and physics_pred > 0:
                deviation = abs(ai_pred - physics_pred) / physics_pred
                if deviation <= MAX_AI_PHYSICS_DEVIATION:
                    use_hybrid = True

            # STEP 4: Blend or use physics only
            if use_hybrid:
                final_pred = physics_pred * (1 - ai_confidence) + ai_pred * ai_confidence

                if ai_groups and physics_groups:
                    final_groups = self._blend_group_details(
                        physics_groups, ai_groups, ai_confidence
                    )
                else:
                    final_groups = ai_groups or physics_groups

                if final_groups:
                    for g in final_groups:
                        g["source"] = f"hybrid_ai{int(ai_confidence*100)}"

                _LOGGER.debug(
                    f"Hybrid hour {hour}: physics={physics_pred:.3f}, ai={ai_pred:.3f}, "
                    f"conf={ai_confidence:.1%}, final={final_pred:.3f}"
                )
            else:
                final_pred = physics_pred
                final_groups = physics_groups
                if final_groups:
                    for g in final_groups:
                        g["source"] = "physics"

            return max(0.0, min(final_pred, capacity * 1.1)), final_groups

        except Exception as e:
            _LOGGER.debug(f"Hour calculation failed: {e}")
            production, _ = self._calculate_physics_production(hour_data, hour_dt)
            return production, None

    def _calculate_physics_production(
        self,
        hour_data: Dict[str, Any],
        hour_dt: datetime
    ) -> tuple[float, Optional[List[Dict[str, Any]]]]:
        """Physics-based calculation with panel groups and self-learning calibration @zara

        The PanelGroupCalculator now applies learned calibration factors
        based on historical Actual vs Physics comparisons.

        Includes twilight handling: At low sun angles (-2° to 5°), weather APIs
        often report 0 radiation, but real production occurs due to diffuse light.

        Returns:
            Tuple of (total_production_kwh, group_details or None)
        """
        try:
            ghi = float(hour_data.get("solar_radiation", hour_data.get("ghi", 0)))
            dni = float(hour_data.get("direct_radiation", hour_data.get("dni", 0)))
            dhi = float(hour_data.get("diffuse_radiation", hour_data.get("dhi", 0)))
            temp = float(hour_data.get("temperature", 15))
            clouds = float(hour_data.get("cloud_cover", hour_data.get("clouds", 50)))

            # Get astronomy data for sun position FIRST (needed for twilight check)
            astronomy = self._get_astronomy_for_hour(hour_dt)
            sun_elevation = astronomy.get("sun_elevation_deg", 0)
            sun_azimuth = astronomy.get("sun_azimuth_deg", 180)
            clear_sky_radiation = astronomy.get("clear_sky_radiation_wm2", 0)

            # Check for twilight conditions and apply minimum radiation if needed
            is_twilight = TWILIGHT_MIN_ELEVATION <= sun_elevation <= TWILIGHT_MAX_ELEVATION

            # NEW: Check for low sun angle with clear sky IN THE MORNING (extended correction range)
            # At low sun angles (<15°) with clear/fair sky (<50% clouds), APIs underestimate
            # This effect is primarily a MORNING phenomenon (evening has more atmospheric haze)
            is_low_sun_clear = (
                TWILIGHT_MAX_ELEVATION < sun_elevation <= LOW_SUN_MAX_ELEVATION
                and clouds < LOW_SUN_CLOUD_THRESHOLD
                and hour_dt.hour in LOW_SUN_MORNING_HOURS
            )

            if ghi <= 0:
                if is_twilight:
                    # Twilight: Apply minimum radiation estimates
                    # Weather APIs often report 0, but diffuse light exists
                    ghi, dhi = self._estimate_twilight_radiation(sun_elevation, clouds)
                    dni = 0.0  # No direct beam at very low angles
                    _LOGGER.debug(
                        f"Twilight adjustment for hour {hour_dt.hour}: "
                        f"elevation={sun_elevation:.1f}°, estimated GHI={ghi:.1f}, DHI={dhi:.1f}"
                    )
                else:
                    # Not twilight and no radiation - truly dark
                    return 0.0, None
            elif is_low_sun_clear:
                # Low sun angle with clear sky: Apply cloud-dependent correction
                # APIs underestimate because they model direct beam poorly at low angles
                ghi, dhi, dni = self._correct_low_sun_radiation(
                    sun_elevation, clouds, ghi, dhi, dni, clear_sky_radiation
                )
                _LOGGER.debug(
                    f"Low-sun clear-sky correction for hour {hour_dt.hour}: "
                    f"elevation={sun_elevation:.1f}°, clouds={clouds:.0f}%, "
                    f"corrected GHI={ghi:.1f}, DHI={dhi:.1f}"
                )

            # Extract hour for calibration lookup
            hour = hour_dt.hour

            # If we have panel groups, use PanelGroupCalculator
            if self.panel_group_calculator and self.panel_group_calculator.has_groups:
                irradiance = IrradianceData(
                    ghi=ghi,
                    dni=dni if dni > 0 else ghi * 0.7,  # Estimate DNI from GHI if not available
                    dhi=dhi if dhi > 0 else ghi * 0.3,  # Estimate DHI from GHI if not available
                )
                sun = SunPosition(
                    elevation_deg=sun_elevation,
                    azimuth_deg=sun_azimuth,
                )

                # Pass hour and cloud_cover for weather-bucket-specific calibration factors
                result = self.panel_group_calculator.calculate_total_power(
                    irradiance=irradiance,
                    sun=sun,
                    ambient_temp_c=temp,
                    hour=hour,
                    cloud_cover=clouds,  # For weather bucket calibration
                )

                # Build group details from physics calculation
                group_details = []
                for gr in result.group_results:
                    group_details.append({
                        "name": gr.group.name,
                        "power_kwh": round(gr.power_kwh, 4),
                        "contribution_percent": round(gr.contribution_percent, 1),
                        "poa_wm2": round(gr.poa_result.poa_total, 1),
                        "aoi_deg": round(gr.poa_result.aoi_deg, 1),
                    })

                return result.total_power_kwh, group_details

            # Simple fallback without panel groups
            efficiency = self._calculate_efficiency(clouds, temp)
            power_kw = (ghi / 1000.0) * self.solar_capacity * efficiency
            return max(0.0, min(power_kw, self.solar_capacity * 1.1)), None

        except Exception as e:
            _LOGGER.debug(f"Physics production calc failed: {e}")
            return 0.0, None

    def _get_astronomy_for_hour(self, hour_dt: datetime) -> Dict[str, Any]:
        """Get astronomy data for specific hour @zara"""
        try:
            cache_manager = get_cache_manager()
            if not cache_manager or not cache_manager.is_loaded():
                return self._default_astronomy(hour_dt.hour)

            date_key = hour_dt.date().isoformat()
            day_data = cache_manager.get_day_data(date_key)
            if not day_data:
                return self._default_astronomy(hour_dt.hour)

            hourly = day_data.get("hourly", {})
            hour_str = str(hour_dt.hour)
            hour_astro = hourly.get(hour_str, {})

            return {
                "sun_elevation_deg": hour_astro.get("elevation_deg", 30),
                "sun_azimuth_deg": hour_astro.get("azimuth_deg", 180),
                "theoretical_max_kwh": hour_astro.get("theoretical_max_pv_kwh", 0.5),
                "clear_sky_radiation_wm2": hour_astro.get("clear_sky_solar_radiation_wm2", 500),
                "max_elevation_today": day_data.get("max_elevation_deg", 60),
            }

        except Exception:
            return self._default_astronomy(hour_dt.hour)

    def _default_astronomy(self, hour: int) -> Dict[str, Any]:
        """Default astronomy values @zara"""
        elevation = max(0, 45 - abs(hour - 12) * 7)
        # Approximate azimuth based on hour
        azimuth = 90 + (hour - 6) * 15  # 90° at 6:00, 180° at 12:00, 270° at 18:00
        azimuth = max(90, min(270, azimuth))
        return {
            "sun_elevation_deg": elevation,
            "sun_azimuth_deg": azimuth,
            "theoretical_max_kwh": 0.5 if 6 <= hour <= 18 else 0.0,
            "clear_sky_radiation_wm2": max(0, 500 - abs(hour - 12) * 50),
            "max_elevation_today": 60,
        }

    def _calculate_efficiency(self, clouds: float, temp: float) -> float:
        """Calculate efficiency based on conditions @zara"""
        cloud_factor = 1.0 - (clouds / 100.0) * 0.75
        cloud_factor = max(0.1, min(1.0, cloud_factor))

        temp_factor = 1.0
        if temp > 25:
            temp_factor = 1.0 - (temp - 25) * 0.004
        elif temp < 10:
            temp_factor = 1.0 + (10 - temp) * 0.002
        temp_factor = max(0.85, min(1.05, temp_factor))

        return cloud_factor * temp_factor * 0.85

    def _estimate_twilight_radiation(
        self,
        sun_elevation: float,
        cloud_cover: float
    ) -> tuple[float, float]:
        """Estimate minimum radiation during twilight hours @zara

        At very low sun angles (-2° to 5°), weather APIs often report 0 W/m²
        but real diffuse light exists. Historical data shows consistent
        production of ~0.02-0.04 kWh during these hours regardless of clouds.

        This is because:
        1. Diffuse sky radiation exists even before/after direct sunlight
        2. At low angles, the atmosphere scatters light broadly
        3. Cloud cover has minimal impact at these angles (it's all diffuse anyway)

        Args:
            sun_elevation: Sun elevation in degrees
            cloud_cover: Cloud cover percentage (0-100)

        Returns:
            Tuple of (estimated_ghi, estimated_dhi) in W/m²
        """
        # Normalize elevation within twilight range to 0-1
        # -2° → 0.0, 5° → 1.0
        elevation_range = TWILIGHT_MAX_ELEVATION - TWILIGHT_MIN_ELEVATION  # 7 degrees
        elevation_factor = (sun_elevation - TWILIGHT_MIN_ELEVATION) / elevation_range
        elevation_factor = max(0.0, min(1.0, elevation_factor))

        # At twilight, cloud cover has minimal impact (empirically observed)
        # Production is nearly constant at ~0.03 kWh regardless of clouds
        # Only apply a small reduction for very heavy overcast
        cloud_factor = 1.0 - (cloud_cover / 100.0) * 0.15  # Max 15% reduction

        # Base minimum DHI scales with elevation factor
        # At -2°: ~3 W/m², at 5°: ~15 W/m²
        min_dhi = TWILIGHT_MIN_DHI_WM2 * (0.3 + 0.7 * elevation_factor) * cloud_factor

        # GHI in twilight is almost entirely diffuse
        min_ghi = min_dhi * 1.1  # GHI slightly higher than DHI

        return max(TWILIGHT_MIN_GHI_WM2, min_ghi), max(3.0, min_dhi)

    def _correct_low_sun_radiation(
        self,
        sun_elevation: float,
        cloud_cover: float,
        api_ghi: float,
        api_dhi: float,
        api_dni: float,
        clear_sky_ghi: float,
    ) -> tuple[float, float, float]:
        """Correct radiation estimates for low sun angles with clear/fair sky @zara

        At sun elevations between 5° and 15°, weather APIs systematically
        underestimate radiation, especially with clear skies. This is because:

        1. At low angles, atmospheric path is long → more scattering
        2. APIs model direct beam poorly at these angles
        3. The sky dome contributes significant diffuse radiation

        The correction is CLOUD-DEPENDENT:
        - Clear (0-25%): Strong correction (APIs miss scattered light)
        - Fair (25-50%): Moderate correction
        - Cloudy (>50%): No correction (radiation is already diffuse)

        NOTE: This provides an INITIAL estimate. The PhysicsCalibrator learns
        actual correction factors for LOW_SUN buckets (clear_low_sun, fair_low_sun)
        which will override this once sufficient data is collected.

        Args:
            sun_elevation: Sun elevation in degrees (5-15° range)
            cloud_cover: Cloud cover percentage (0-100)
            api_ghi: GHI from weather API (W/m²)
            api_dhi: DHI from weather API (W/m²)
            api_dni: DNI from weather API (W/m²)
            clear_sky_ghi: Clear sky GHI from astronomy cache (W/m²)

        Returns:
            Tuple of (corrected_ghi, corrected_dhi, corrected_dni) in W/m²
        """
        # Skip if cloudy - radiation is already diffuse, API is accurate
        if cloud_cover >= LOW_SUN_CLOUD_THRESHOLD:
            return api_ghi, api_dhi, api_dni

        # Check if we have learned LOW_SUN bucket factors from the calibrator
        # If so, skip the hardcoded correction - the calibrator will apply learned factors
        has_learned_low_sun_factor = self._has_learned_low_sun_factor(sun_elevation, cloud_cover)
        if has_learned_low_sun_factor:
            _LOGGER.debug(
                f"Low-sun: Using learned calibration factors instead of initial estimate "
                f"(elev={sun_elevation:.1f}°, clouds={cloud_cover:.0f}%)"
            )
            return api_ghi, api_dhi, api_dni

        # Calculate INITIAL correction factor based on sun elevation
        # This is used until the calibrator has learned better factors
        # Lower elevation = stronger correction needed
        # At 5°: factor ~2.5, at 15°: factor ~1.0
        elevation_normalized = (sun_elevation - TWILIGHT_MAX_ELEVATION) / (
            LOW_SUN_MAX_ELEVATION - TWILIGHT_MAX_ELEVATION
        )  # 0.0 at 5°, 1.0 at 15°
        elevation_normalized = max(0.0, min(1.0, elevation_normalized))

        # Base correction factor: exponential decay from 2.5 to 1.0
        base_factor = 1.0 + 1.5 * (1.0 - elevation_normalized) ** 2

        # Cloud-dependent scaling: less clouds = more correction needed
        # At 0% clouds: full correction
        # At 50% clouds: no correction
        cloud_scaling = 1.0 - (cloud_cover / LOW_SUN_CLOUD_THRESHOLD)
        cloud_scaling = max(0.0, cloud_scaling)

        # Final correction factor
        correction_factor = 1.0 + (base_factor - 1.0) * cloud_scaling

        # Apply correction to GHI
        corrected_ghi = api_ghi * correction_factor

        # Use clear sky as upper bound if available
        if clear_sky_ghi > 0:
            # Don't exceed clear sky radiation (reduced by cloud cover)
            max_ghi = clear_sky_ghi * (1.0 - cloud_cover / 100.0 * 0.7)
            corrected_ghi = min(corrected_ghi, max(api_ghi, max_ghi))

        # At low angles, most radiation is diffuse
        # DHI should be 70-90% of GHI at these angles
        dhi_ratio = 0.85 - (sun_elevation - 5) * 0.015  # 0.85 at 5°, 0.70 at 15°
        dhi_ratio = max(0.70, min(0.85, dhi_ratio))

        corrected_dhi = max(api_dhi, corrected_ghi * dhi_ratio)

        # DNI at low angles is minimal - keep API value or reduce
        corrected_dni = min(api_dni, corrected_ghi * (1.0 - dhi_ratio))

        _LOGGER.debug(
            f"Low-sun INITIAL correction: elev={sun_elevation:.1f}°, clouds={cloud_cover:.0f}%, "
            f"factor={correction_factor:.2f}, GHI {api_ghi:.1f}→{corrected_ghi:.1f}, "
            f"DHI {api_dhi:.1f}→{corrected_dhi:.1f} (will be learned over time)"
        )

        return corrected_ghi, corrected_dhi, corrected_dni

    def _has_learned_low_sun_factor(self, sun_elevation: float, cloud_cover: float) -> bool:
        """Check if calibrator has learned LOW_SUN bucket factors @zara

        Args:
            sun_elevation: Sun elevation in degrees
            cloud_cover: Cloud cover percentage (0-100)

        Returns:
            True if calibrator has learned factors for this LOW_SUN condition
        """
        if not self.panel_group_calculator or not self.panel_group_calculator._calibrator:
            return False

        calibrator = self.panel_group_calculator._calibrator

        # Determine which LOW_SUN bucket this would be
        if cloud_cover <= 25:
            bucket_name = "clear_low_sun"
        elif cloud_cover <= 50:
            bucket_name = "fair_low_sun"
        else:
            return True  # Cloudy - no LOW_SUN correction needed anyway

        # Check if any group has learned this bucket
        for group_name, factors in calibrator._factors.items():
            bucket_data = factors.bucket_factors.get(bucket_name)
            if bucket_data and bucket_data.sample_count >= 1:
                return True

        return False

    def _calculate_hourly_ai_confidence(
        self,
        base_r2: float,
        clouds: float,
        sun_elevation: float,
        hour: int,
    ) -> float:
        """Calculate dynamic AI confidence for specific hour @zara

        Factors:
        1. Base: R² score of the model
        2. Clouds: High cloud cover → less trustworthy (more variance)
        3. Sun elevation: Low elevation → less data, less confidence
        4. Time of day: Edge hours → less trustworthy
        """
        if base_r2 <= 0:
            return 0.0

        # Base confidence from R²
        if base_r2 < 0.5:
            base_conf = base_r2
        elif base_r2 < 0.8:
            base_conf = 0.5 + (base_r2 - 0.5) * 1.333
        else:
            base_conf = 0.9 + min((base_r2 - 0.8), 0.2) * 0.5

        # Cloud factor: 0-30% → 1.0, 30-70% → 0.9, 70-100% → 0.7
        if clouds < 30:
            cloud_factor = 1.0
        elif clouds < 70:
            cloud_factor = 0.9
        else:
            cloud_factor = 0.7

        # Sun elevation factor: < 10° → 0.7, 10-30° → 0.85, > 30° → 1.0
        if sun_elevation < 10:
            elevation_factor = 0.7
        elif sun_elevation < 30:
            elevation_factor = 0.85
        else:
            elevation_factor = 1.0

        # Time of day factor: 10-14h → 1.0, otherwise → 0.85
        if 10 <= hour <= 14:
            hour_factor = 1.0
        else:
            hour_factor = 0.85

        return base_conf * cloud_factor * elevation_factor * hour_factor

    def _blend_group_details(
        self,
        physics_groups: List[Dict[str, Any]],
        ai_groups: List[Dict[str, Any]],
        ai_weight: float,
    ) -> List[Dict[str, Any]]:
        """Blend physics and AI group predictions @zara"""
        blended = []
        physics_weight = 1 - ai_weight

        for pg, ag in zip(physics_groups, ai_groups):
            power = pg["power_kwh"] * physics_weight + ag["power_kwh"] * ai_weight
            blended.append({
                "name": pg.get("name", ag.get("name")),
                "power_kwh": round(power, 4),
                "contribution_percent": 0,
                "physics_kwh": round(pg["power_kwh"], 4),
                "ai_kwh": round(ag["power_kwh"], 4),
                "ai_weight": round(ai_weight, 2),
            })

        # Recalculate contribution percentages
        total = sum(g["power_kwh"] for g in blended)
        for g in blended:
            g["contribution_percent"] = round(g["power_kwh"] / total * 100, 1) if total > 0 else 0

        return blended

    def _parse_datetime(self, hour_data: Dict[str, Any]) -> Optional[datetime]:
        """Parse datetime from hour data @zara"""
        try:
            dt_str = hour_data.get("datetime") or hour_data.get("time")
            if dt_str:
                if isinstance(dt_str, datetime):
                    return dt_str
                return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            pass
        return None

    def _empty_result(self) -> ForecastResult:
        """Return empty result @zara"""
        return ForecastResult(
            forecast_today=0.0,
            forecast_tomorrow=0.0,
            forecast_day_after_tomorrow=0.0,
            confidence_today=0.0,
            confidence_tomorrow=0.0,
            confidence_day_after=0.0,
            method="rule_based_empty",
            calibrated=False,
            best_hour_today=12,
            best_hour_production_kwh=0.0,
            hourly_values=[],
            model_accuracy=None,
            forecast_today_raw=0.0,
            safeguard_applied_today=False,
        )
