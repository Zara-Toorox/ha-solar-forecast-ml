"""Rule-Based Forecast Strategy V12.2.0 @zara

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""

import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from homeassistant.util import dt as dt_util

from ..astronomy.astronomy_cache_manager import get_cache_manager
from ..core.core_helpers import SafeDateTimeUtil as dt_util_safe
from ..data.data_weather_corrector import WeatherForecastCorrector
from ..ml.ml_pattern_learner import PatternLearner
from ..ml.ml_residual_trainer import ResidualTrainer
from ..physics import PhysicsEngine, GeometryLearner, PanelGroupCalculator, PanelGroupEfficiencyLearner
from .forecast_strategy_base import ForecastResult, ForecastStrategy
from .forecast_weather_calculator import WeatherCalculator

_LOGGER = logging.getLogger(__name__)

class RuleBasedForecastStrategy(ForecastStrategy):
    """
    Rule-Based Forecast Strategy - DUMB CONSUMER

    Uses corrected weather data from weather_forecast_corrected.json.
    Does NOT calculate weather factors - just uses prepared data.
    """

    def __init__(
        self,
        weather_calculator: WeatherCalculator,
        solar_capacity: float,
        orchestrator: Optional[Any] = None,
        panel_groups: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize the Rule-Based Forecast Strategy"""
        super().__init__("rule_based")
        self.weather_calculator = weather_calculator
        self.solar_capacity = solar_capacity
        self.orchestrator = orchestrator

        self.PEAK_KW_PER_KWP = 0.95
        self.TOMORROW_DISCOUNT_FACTOR = 0.92

        self.MAX_REALISTIC_DAILY_KWH_PER_KWP = 8.0

        self.weather_corrector: Optional[WeatherForecastCorrector] = None
        self._corrector_init_attempted = False

        self.pattern_learner: Optional[PatternLearner] = None
        self._pattern_learner_init_attempted = False

        self.physics_engine: Optional[PhysicsEngine] = None
        self.geometry_learner: Optional[GeometryLearner] = None
        self._physics_init_attempted = False

        self.panel_group_calculator: Optional[PanelGroupCalculator] = None
        self._panel_groups_config = panel_groups or []
        self._panel_groups_init_attempted = False

        self.panel_group_efficiency_learner: Optional[PanelGroupEfficiencyLearner] = None
        self._efficiency_learner_init_attempted = False

        self.residual_trainer: Optional[ResidualTrainer] = None
        self._residual_init_attempted = False

        # Feature flags
        self.use_physics_engine = True      # Use physics-based POA calculation
        self.use_ml_residual = True         # Use ML residual enhancement
        self.use_panel_groups = bool(panel_groups)  # Use panel group calculation

        if panel_groups:
            _LOGGER.info(
                f"RuleBasedForecastStrategy initialized with {len(panel_groups)} panel groups"
            )
        _LOGGER.debug("RuleBasedForecastStrategy (Physics-First + ML-Enhanced) initialized.")

    def is_available(self) -> bool:
        """This strategy is always available as a fallback @zara"""
        return True

    def get_priority(self) -> int:
        """Returns the priority of this strategy Lower than ML strategy @zara"""
        return 50

    async def calculate_forecast(
        self,
        hourly_weather_forecast: List[Dict[str, Any]],
        sensor_data: Dict[str, Any],
        correction_factor: float,
        **kwargs,
    ) -> ForecastResult:
        """
        Calculate forecast using corrected weather data.

        DUMB CONSUMER: Uses prepared weather values from weather_forecast_corrected.json.
        Does NOT calculate weather factors.
        """
        _LOGGER.debug("Calculating forecast using Rule-based (Dumb Consumer) strategy...")

        try:

            try:
                base_capacity_kwp = float(sensor_data.get("solar_capacity", self.solar_capacity))
                if base_capacity_kwp <= 0:
                    _LOGGER.warning(
                        f"Solar capacity ({base_capacity_kwp}kWp) is zero or negative. Using fallback 1.0 kWp."
                    )
                    base_capacity_kwp = 1.0
            except (ValueError, TypeError):
                _LOGGER.warning(
                    f"Invalid solar_capacity in sensor_data, using default {self.solar_capacity} kWp."
                )
                base_capacity_kwp = self.solar_capacity

            if self.weather_corrector is None and not self._corrector_init_attempted:
                self._corrector_init_attempted = True
                if self.orchestrator and hasattr(self.orchestrator, 'data_manager'):
                    from ..data.data_weather_corrector import WeatherForecastCorrector
                    self.weather_corrector = WeatherForecastCorrector(
                        self.orchestrator.hass,
                        self.orchestrator.data_manager
                    )
                    _LOGGER.debug("Weather corrector initialized")
                else:
                    _LOGGER.error("Cannot initialize weather corrector - no data_manager available")

            if self.weather_corrector is None:

                return await self._calculate_forecast_fallback(
                    hourly_weather_forecast,
                    base_capacity_kwp,
                    correction_factor
                )

            if self.pattern_learner is None and not self._pattern_learner_init_attempted:
                self._pattern_learner_init_attempted = True
                if self.orchestrator and hasattr(self.orchestrator, 'data_manager'):
                    self.pattern_learner = PatternLearner(
                        self.orchestrator.data_manager.data_dir
                    )
                    await self.pattern_learner.load_patterns()
                    _LOGGER.info("Pattern learner initialized (KI mode enabled)")
                else:
                    _LOGGER.warning("Pattern learner not available - using baseline only")

            if self.panel_group_calculator is None and not self._panel_groups_init_attempted:
                self._panel_groups_init_attempted = True
                if self._panel_groups_config:
                    try:
                        # Get data_path for config loading
                        data_path = None
                        if self.orchestrator and hasattr(self.orchestrator, 'data_manager'):
                            data_path = Path(self.orchestrator.data_manager.data_dir)

                        # Use async factory to avoid blocking file I/O in event loop
                        self.panel_group_calculator = await PanelGroupCalculator.async_create(
                            panel_groups=self._panel_groups_config,
                            data_path=data_path,
                        )
                        self.use_panel_groups = True
                        _LOGGER.info(
                            "PanelGroupCalculator initialized: %d groups, total %.2f kWp",
                            self.panel_group_calculator.group_count,
                            self.panel_group_calculator.total_capacity_kwp,
                        )
                    except Exception as e:
                        _LOGGER.warning(f"Could not initialize PanelGroupCalculator: {e}")
                        self.use_panel_groups = False

            # Initialize PanelGroupEfficiencyLearner for per-group efficiency factors
            if (self.panel_group_efficiency_learner is None
                and not self._efficiency_learner_init_attempted
                and self.use_panel_groups):
                self._efficiency_learner_init_attempted = True
                try:
                    if self.orchestrator and hasattr(self.orchestrator, 'data_manager'):
                        data_dir = Path(self.orchestrator.data_manager.data_dir)
                        # Use async factory to avoid blocking file I/O in event loop
                        self.panel_group_efficiency_learner = await PanelGroupEfficiencyLearner.async_create(
                            data_path=data_dir,
                            panel_groups=self._panel_groups_config,
                        )
                        _LOGGER.info(
                            "PanelGroupEfficiencyLearner loaded: %d groups, %d total samples",
                            len(self._panel_groups_config),
                            self.panel_group_efficiency_learner.total_samples,
                        )
                except Exception as e:
                    _LOGGER.warning(f"Could not initialize PanelGroupEfficiencyLearner: {e}")

            if self.physics_engine is None and not self._physics_init_attempted and not self.use_panel_groups:
                self._physics_init_attempted = True
                try:
                    if self.orchestrator and hasattr(self.orchestrator, 'data_manager'):
                        data_dir = Path(self.orchestrator.data_manager.data_dir)

                        # Initialize Geometry Learner first (loads learned tilt/azimuth)
                        # Use skip_load=True and async_load_state to avoid blocking event loop
                        self.geometry_learner = GeometryLearner(
                            data_path=data_dir,
                            system_capacity_kwp=base_capacity_kwp,
                            skip_load=True,  # Don't block event loop
                        )
                        await self.geometry_learner.async_load_state()

                        # Get physics engine with learned geometry
                        self.physics_engine = self.geometry_learner.get_physics_engine()

                        _LOGGER.info(
                            "Physics Engine initialized (Physics-First mode): "
                            f"tilt={self.physics_engine.geometry.tilt_deg:.1f}°, "
                            f"azimuth={self.physics_engine.geometry.azimuth_deg:.1f}°, "
                            f"confidence={self.physics_engine.geometry_confidence:.2f}"
                        )
                    else:
                        # Fallback: Initialize without geometry learning
                        self.physics_engine = PhysicsEngine(
                            system_capacity_kwp=base_capacity_kwp,
                        )
                        _LOGGER.info("Physics Engine initialized with default geometry")
                except Exception as e:
                    _LOGGER.warning(f"Could not initialize Physics Engine: {e}. Using legacy calculation.")

            # Initialize Residual Trainer for ML enhancement (Phase 4)
            if self.residual_trainer is None and not self._residual_init_attempted:
                self._residual_init_attempted = True
                try:
                    if self.orchestrator and hasattr(self.orchestrator, 'data_manager'):
                        data_dir = Path(self.orchestrator.data_manager.data_dir)
                        # Use skip_load=True and async_load_state to avoid blocking event loop
                        # Pass panel_groups so ResidualTrainer uses correct geometry
                        self.residual_trainer = ResidualTrainer(
                            data_dir=data_dir,
                            system_capacity_kwp=base_capacity_kwp,
                            panel_groups=self._panel_groups_config if self.use_panel_groups else None,
                            skip_load=True,  # Don't block event loop
                        )
                        await self.residual_trainer.async_load_state()
                        model_info = self.residual_trainer.get_model_info()
                        _LOGGER.info(
                            "ResidualTrainer initialized: %d samples, model=%s, panel_groups=%s",
                            model_info.get("sample_count", 0),
                            model_info.get("model_type", "none"),
                            "yes" if self.use_panel_groups else "no",
                        )
                except Exception as e:
                    _LOGGER.warning(f"Could not initialize ResidualTrainer: {e}")

            rb_overall_correction = 1.0
            try:
                if self.weather_corrector.corrected_file.exists():
                    corrected_data = await self.weather_corrector._read_json_file(
                        self.weather_corrector.corrected_file, None
                    )
                    if corrected_data:
                        rb_correction_data = corrected_data.get("metadata", {}).get("rb_overall_correction", {})
                        rb_overall_correction = rb_correction_data.get("factor", 1.0)
                        confidence = rb_correction_data.get("confidence", 0.0)
                        _LOGGER.info(
                            f"RB Overall Correction loaded: factor={rb_overall_correction:.3f}, "
                            f"confidence={confidence:.2f}"
                        )
            except Exception as e:
                _LOGGER.warning(f"Could not load rb_overall_correction: {e}. Using default 1.0")

            total_today_kwh = 0.0
            total_tomorrow_kwh = 0.0
            total_day_after_kwh = 0.0

            best_hour_today = None
            best_hour_production = 0.0

            now_local = dt_util_safe.now()
            today_date = now_local.date()
            tomorrow_date = today_date + timedelta(days=1)
            day_after_tomorrow_date = today_date + timedelta(days=2)

            hourly_values = []

            for hour_data in hourly_weather_forecast:
                try:
                    hour_dt_local = hour_data.get("local_datetime")
                    if not hour_dt_local:
                        continue

                    if isinstance(hour_dt_local, str):
                        hour_dt_local = dt_util_safe.parse_datetime(hour_dt_local)
                        if not hour_dt_local:
                            continue

                    hour_dt_local = dt_util_safe.as_local(hour_dt_local)

                    if hour_dt_local < now_local:
                        continue

                    hour_date = hour_dt_local.date()
                    hour_local = hour_dt_local.hour

                    corrected_weather = await self.weather_corrector.get_corrected_weather_for_hour(hour_dt_local)

                    if not corrected_weather:
                        _LOGGER.debug(f"No corrected weather for {hour_dt_local} - skipping hour")
                        continue

                    # Extract weather data
                    solar_radiation_wm2 = corrected_weather.get("solar_radiation_wm2", 0) or 0
                    ghi = corrected_weather.get("ghi", solar_radiation_wm2) or solar_radiation_wm2
                    dni = corrected_weather.get("direct_radiation", 0) or 0
                    dhi = corrected_weather.get("diffuse_radiation", 0) or 0
                    # NOTE: Don't use "or 100" here! 0% clouds is valid and means clear sky @zara
                    cloud_cover_raw = corrected_weather.get("clouds")
                    cloud_cover = cloud_cover_raw if cloud_cover_raw is not None else 50
                    temperature = corrected_weather.get("temperature", 15) or 15

                    if ghi <= 0 and solar_radiation_wm2 <= 0:
                        cache_manager = get_cache_manager()
                        hour_astro = None
                        if cache_manager:
                            hour_astro = cache_manager.get_hourly_data(hour_date.isoformat(), hour_local)

                        sun_elevation = hour_astro.get("elevation_deg", 0) if hour_astro else 0
                        clear_sky_rad = hour_astro.get("clear_sky_solar_radiation_wm2", 0) if hour_astro else 0

                        if sun_elevation > 0 and clear_sky_rad > 0:
                            ghi = clear_sky_rad
                            solar_radiation_wm2 = clear_sky_rad
                            _LOGGER.debug(
                                f"Hour {hour_local}: GHI=0 but sun_elevation={sun_elevation:.1f}° - "
                                f"using clear_sky fallback: {clear_sky_rad:.1f} W/m²"
                            )
                        else:
                            hourly_kwh = 0.0
                            panel_group_predictions = None
                            continue

                    if ghi > 0 or solar_radiation_wm2 > 0:
                        # Get astronomy data for this hour
                        cache_manager = get_cache_manager()
                        hour_astro = None
                        if cache_manager:
                            hour_astro = cache_manager.get_hourly_data(hour_date.isoformat(), hour_local)

                        sun_elevation = 0.0
                        sun_azimuth = 180.0
                        if hour_astro:
                            sun_elevation = hour_astro.get("elevation_deg", 0) or 0
                            sun_azimuth = hour_astro.get("azimuth_deg", 180) or 180

                        # Initialize per-group predictions (will be populated if panel groups are used)
                        panel_group_predictions = None

                        if self.use_panel_groups and self.panel_group_calculator:
                            # Calculate using panel groups
                            group_result = self.panel_group_calculator.calculate_hourly_forecast(
                                weather_data={
                                    "ghi": ghi or solar_radiation_wm2,
                                    "direct_radiation": dni,
                                    "diffuse_radiation": dhi,
                                    "temperature": temperature,
                                },
                                astronomy_data={
                                    "elevation_deg": sun_elevation,
                                    "azimuth_deg": sun_azimuth,
                                },
                            )

                            # Apply per-group learned efficiency factors if available
                            physics_pred = 0.0
                            efficiency_applied = False

                            if self.panel_group_efficiency_learner:
                                # Calculate with per-group efficiency factors
                                # NEW: Pass DNI/DHI/GHI for physics-based efficiency weighting
                                for idx, grp in enumerate(group_result.get("groups", [])):
                                    group_kwh = grp.get("power_kwh", 0)
                                    efficiency = self.panel_group_efficiency_learner.get_efficiency_for_hour(
                                        group_index=idx,
                                        hour=hour_local,
                                        current_cloud_cover=cloud_cover,
                                        current_sun_elevation=sun_elevation,
                                        current_dni=dni,
                                        current_dhi=dhi,
                                        current_ghi=ghi or solar_radiation_wm2,
                                    )
                                    if efficiency != 1.0:
                                        efficiency_applied = True
                                    physics_pred += group_kwh * efficiency

                                # Panel Groups efficiency logging removed to reduce log spam
                            else:
                                # Fallback: Use raw physics prediction without efficiency factors
                                physics_pred = group_result["physics_prediction_kwh"]

                            # Panel Groups calculation logging removed to reduce log spam

                            # For panel groups, we store the group breakdown in physics_result
                            physics_result = {
                                "physics_prediction_kwh": physics_pred,
                                "poa_irradiance": {"poa_total_wm2": 0},  # Average not applicable
                                "groups": group_result.get("groups", []),
                            }

                            seasonal_factor = 1.0
                            if self.pattern_learner:
                                strategy_mode = self.pattern_learner.get_strategy_mode(cloud_cover)
                                # Apply cloud_impacts for both "cloudy" (>80%) and "mixed" (40-80%) conditions
                                if strategy_mode in ("cloudy", "mixed"):
                                    cloud_efficiency, _ = self.pattern_learner.get_cloud_impact(
                                        hour_local, cloud_cover
                                    )
                                    # cloud_efficiency can be > 1.0 (boost) or < 1.0 (dampen)
                                    # Blend with physics: use learned efficiency directly but dampen extreme values
                                    if cloud_efficiency >= 1.0:
                                        # Boost case: diffuse light better than expected
                                        seasonal_factor = 1.0 + 0.5 * (cloud_efficiency - 1.0)
                                    else:
                                        # Dampen case: worse than expected
                                        seasonal_factor = 0.7 + 0.3 * cloud_efficiency

                            # Apply final corrections to panel group result
                            hourly_kwh = (
                                physics_pred
                                * correction_factor       # User/learned correction
                                * rb_overall_correction   # RB overall learned factor
                                * seasonal_factor         # Seasonal/local adjustment
                            )

                            # Panel Groups corrections logging removed to reduce log spam

                            # Build per-group predictions for storage and later comparison
                            # Apply same correction factors to each group proportionally
                            total_correction = correction_factor * rb_overall_correction * seasonal_factor
                            panel_group_predictions = {}
                            for idx, grp in enumerate(group_result.get("groups", [])):
                                group_name = grp.get("group_name", f"Group {idx+1}")
                                raw_kwh = grp.get("power_kwh", 0)
                                # Apply learned efficiency if available
                                # NEW: Pass DNI/DHI/GHI for physics-based efficiency weighting
                                if self.panel_group_efficiency_learner:
                                    eff = self.panel_group_efficiency_learner.get_efficiency_for_hour(
                                        idx, hour_local,
                                        current_cloud_cover=cloud_cover,
                                        current_sun_elevation=sun_elevation,
                                        current_dni=dni,
                                        current_dhi=dhi,
                                        current_ghi=ghi or solar_radiation_wm2,
                                    )
                                else:
                                    eff = 1.0
                                # Final prediction for this group
                                group_pred_kwh = raw_kwh * eff * total_correction
                                panel_group_predictions[group_name] = round(group_pred_kwh, 4)

                        elif self.use_physics_engine and self.physics_engine:
                            # Use Physics Engine for accurate POA calculation (single orientation)
                            physics_result = self.physics_engine.calculate_hourly_forecast(
                                weather_data={
                                    "ghi": ghi or solar_radiation_wm2,
                                    "direct_radiation": dni,
                                    "diffuse_radiation": dhi,
                                    "temperature": temperature,
                                },
                                astronomy_data={
                                    "elevation_deg": sun_elevation,
                                    "azimuth_deg": sun_azimuth,
                                },
                            )

                            # Base physics prediction
                            physics_pred = physics_result["physics_prediction_kwh"]

                            ml_residual = 0.0
                            ml_weight = 0.0
                            if self.use_ml_residual and self.residual_trainer:
                                try:
                                    # Build record for residual prediction
                                    record = {
                                        "weather": corrected_weather,
                                        "astronomy": hour_astro or {},
                                        "datetime": hour_dt_local.isoformat(),
                                    }
                                    ensemble_result = self.residual_trainer.predict_with_physics(
                                        weather_data={
                                            "ghi": ghi or solar_radiation_wm2,
                                            "direct_radiation": dni,
                                            "diffuse_radiation": dhi,
                                            "temperature": temperature,
                                        },
                                        astronomy_data={
                                            "elevation_deg": sun_elevation,
                                            "azimuth_deg": sun_azimuth,
                                        },
                                        record=record,
                                    )
                                    ml_residual = ensemble_result.get("ml_residual_kwh", 0.0)
                                    ml_weight = ensemble_result.get("ml_weight", 0.0)
                                    # Use the weighted ensemble as base
                                    hourly_kwh = ensemble_result.get("final_prediction_kwh", physics_pred)
                                except Exception as e:
                                    _LOGGER.debug(f"ML residual prediction failed: {e}")
                                    hourly_kwh = physics_pred
                            else:
                                hourly_kwh = physics_pred

                            # Apply additional learned corrections (Pattern Learner)
                            # These are seasonal/local adjustments from actual production data
                            seasonal_factor = 1.0
                            if self.pattern_learner:
                                strategy_mode = self.pattern_learner.get_strategy_mode(cloud_cover)
                                # Apply cloud_impacts for both "cloudy" (>80%) and "mixed" (40-80%) conditions
                                if strategy_mode in ("cloudy", "mixed"):
                                    cloud_efficiency, _ = self.pattern_learner.get_cloud_impact(
                                        hour_local, cloud_cover
                                    )
                                    # cloud_efficiency can be > 1.0 (boost) or < 1.0 (dampen)
                                    # Blend with physics: use learned efficiency directly but dampen extreme values
                                    if cloud_efficiency >= 1.0:
                                        # Boost case: diffuse light better than expected
                                        seasonal_factor = 1.0 + 0.5 * (cloud_efficiency - 1.0)
                                    else:
                                        # Dampen case: worse than expected
                                        seasonal_factor = 0.7 + 0.3 * cloud_efficiency

                            # Apply final corrections
                            hourly_kwh = (
                                hourly_kwh
                                * correction_factor       # User/learned correction
                                * rb_overall_correction   # RB overall learned factor
                                * seasonal_factor         # Seasonal/local adjustment
                            )

                            if _LOGGER.isEnabledFor(logging.DEBUG):
                                _LOGGER.debug(
                                    f"Physics+ML calculation - Hour {hour_local}: "
                                    f"GHI={ghi:.1f}, DNI={dni:.1f}, DHI={dhi:.1f} W/m², "
                                    f"sun=({sun_elevation:.1f}°, {sun_azimuth:.1f}°), "
                                    f"POA={physics_result['poa_irradiance']['poa_total_wm2']:.1f} W/m², "
                                    f"physics={physics_pred:.4f} kWh, "
                                    f"ml_residual={ml_residual:.4f} (weight={ml_weight:.2f}), "
                                    f"seasonal={seasonal_factor:.3f} → {hourly_kwh:.4f} kWh"
                                )

                        else:
                            # ============================================================
                            # LEGACY FALLBACK (wenn Physics Engine nicht verfügbar)
                            # Behält alte Logik für Rückwärtskompatibilität
                            # ============================================================
                            hour_factor = self._get_hour_factor(hour_local, hour_dt_local)

                            geometry_factor = 1.0
                            strategy_mode = "baseline"

                            if self.pattern_learner:
                                strategy_mode = self.pattern_learner.get_strategy_mode(cloud_cover)

                                if strategy_mode == "sunny":
                                    if cache_manager and hour_astro:
                                        day_astro = cache_manager.get_day_data(hour_date.isoformat())
                                        max_elevation = None
                                        if day_astro and "hourly" in day_astro:
                                            max_elevation = max(
                                                (h.get("elevation_deg", 0) for h in day_astro["hourly"].values()),
                                                default=None
                                            )
                                        if sun_elevation > 0:
                                            geometry_factor, _ = self.pattern_learner.get_geometry_factor(
                                                sun_elevation, hour_date.month, max_elevation
                                            )
                                else:
                                    cloud_efficiency, _ = self.pattern_learner.get_cloud_impact(
                                        hour_local, cloud_cover
                                    )
                                    geometry_factor = cloud_efficiency

                            # Legacy formula (has double-scaling issue but kept as fallback)
                            hourly_kwh = (
                                (solar_radiation_wm2 / 1000.0)
                                * base_capacity_kwp
                                * hour_factor
                                * self.PEAK_KW_PER_KWP
                                * correction_factor
                                * rb_overall_correction
                                * geometry_factor
                            )

                            if _LOGGER.isEnabledFor(logging.DEBUG):
                                _LOGGER.debug(
                                    f"Legacy calculation - Hour {hour_local}: "
                                    f"solar_radiation={solar_radiation_wm2:.1f}W/m², "
                                    f"hour_factor={hour_factor:.3f}, geometry={geometry_factor:.2f}, "
                                    f"clouds={cloud_cover:.0f}% → hourly_kwh={hourly_kwh:.3f}"
                                )

                        # Apply hourly cap to prevent unrealistic values
                        # Use panel group total capacity if available, otherwise base capacity
                        if self.use_panel_groups and self.panel_group_calculator:
                            effective_capacity_kwp = self.panel_group_calculator.total_capacity_kwp
                        else:
                            effective_capacity_kwp = base_capacity_kwp
                        max_hourly_kwh = effective_capacity_kwp * self.PEAK_KW_PER_KWP * 0.95
                        if hourly_kwh > max_hourly_kwh:
                            _LOGGER.debug(
                                f"Hourly cap: {hourly_kwh:.3f} kWh → {max_hourly_kwh:.3f} kWh "
                                f"(hour {hour_local}, exceeds theoretical max)"
                            )
                            hourly_kwh = max_hourly_kwh

                    hourly_kwh = max(0.0, hourly_kwh)

                    # Build hourly value entry
                    hourly_entry = {
                        "hour": hour_local,
                        "datetime": hour_dt_local.isoformat(),
                        "production_kwh": round(hourly_kwh, 3),
                        "date": hour_date.isoformat(),
                    }

                    # Add per-group predictions if available (panel groups mode)
                    if panel_group_predictions:
                        hourly_entry["panel_group_predictions"] = panel_group_predictions

                    hourly_values.append(hourly_entry)

                    if hour_date == today_date:
                        total_today_kwh += hourly_kwh
                        if hourly_kwh > best_hour_production:
                            best_hour_today = hour_local
                            best_hour_production = hourly_kwh
                    elif hour_date == tomorrow_date:
                        total_tomorrow_kwh += hourly_kwh
                    elif hour_date == day_after_tomorrow_date:
                        total_day_after_kwh += hourly_kwh

                except Exception as e_inner:
                    _LOGGER.warning(
                        f"Failed to process hour {hour_data.get('local_hour')}: {e_inner}"
                    )
                    continue

            # Rule-based iteration complete logging removed to reduce log spam

            # Store raw values BEFORE any discounts or caps (for ForecastResult)
            raw_today_kwh = total_today_kwh
            raw_tomorrow_kwh = total_tomorrow_kwh
            raw_day_after_kwh = total_day_after_kwh

            # Apply discount factors for future days (weather uncertainty)
            tomorrow_discount = self.TOMORROW_DISCOUNT_FACTOR
            day_after_discount = self.TOMORROW_DISCOUNT_FACTOR * 0.95

            total_tomorrow_kwh *= tomorrow_discount
            total_day_after_kwh *= day_after_discount

            # Apply discount factors to hourly values as well (consistency fix)
            tomorrow_date_str = tomorrow_date.isoformat()
            day_after_date_str = day_after_tomorrow_date.isoformat()
            for entry in hourly_values:
                entry_date = entry.get("date")
                if entry_date == tomorrow_date_str:
                    entry["production_kwh"] = round(entry["production_kwh"] * tomorrow_discount, 3)
                    # Also scale panel_group_predictions if present
                    if "panel_group_predictions" in entry:
                        for grp_name in entry["panel_group_predictions"]:
                            entry["panel_group_predictions"][grp_name] = round(
                                entry["panel_group_predictions"][grp_name] * tomorrow_discount, 4
                            )
                elif entry_date == day_after_date_str:
                    entry["production_kwh"] = round(entry["production_kwh"] * day_after_discount, 3)
                    if "panel_group_predictions" in entry:
                        for grp_name in entry["panel_group_predictions"]:
                            entry["panel_group_predictions"][grp_name] = round(
                                entry["panel_group_predictions"][grp_name] * day_after_discount, 4
                            )

            # Apply daily plausibility cap (max realistic kWh per day)
            # Use effective capacity for panel groups, otherwise base capacity
            if self.use_panel_groups and self.panel_group_calculator:
                cap_capacity_kwp = self.panel_group_calculator.total_capacity_kwp
            else:
                cap_capacity_kwp = base_capacity_kwp
            max_realistic_daily_kwh = cap_capacity_kwp * self.MAX_REALISTIC_DAILY_KWH_PER_KWP

            today_forecast_kwh = min(total_today_kwh, max_realistic_daily_kwh)
            tomorrow_forecast_kwh = min(total_tomorrow_kwh, max_realistic_daily_kwh)
            day_after_forecast_kwh = min(total_day_after_kwh, max_realistic_daily_kwh)

            # Log if daily cap was applied
            if (total_today_kwh > max_realistic_daily_kwh or
                total_tomorrow_kwh > max_realistic_daily_kwh or
                total_day_after_kwh > max_realistic_daily_kwh):
                _LOGGER.warning(
                    f"Daily plausibility cap applied (max {max_realistic_daily_kwh:.2f} kWh): "
                    f"Today {total_today_kwh:.2f}→{today_forecast_kwh:.2f}, "
                    f"Tomorrow {total_tomorrow_kwh:.2f}→{tomorrow_forecast_kwh:.2f}, "
                    f"DayAfter {total_day_after_kwh:.2f}→{day_after_forecast_kwh:.2f}"
                )

            try:
                current_yield = sensor_data.get("current_yield")
                if current_yield is not None and current_yield > 0:
                    current_yield_float = float(current_yield)

                    if current_yield_float > today_forecast_kwh:
                        remaining_hours = max(0, 21 - now_local.hour)
                        total_production_hours = 15

                        additional_forecast = 0.0
                        if total_production_hours > 0 and remaining_hours > 0:
                            remaining_fraction = remaining_hours / total_production_hours
                            additional_forecast = today_forecast_kwh * remaining_fraction
                        else:
                            additional_forecast = today_forecast_kwh * 0.1

                        adjusted_today_forecast = current_yield_float + additional_forecast

                        _LOGGER.info(
                            f"Minimum forecast adjustment (Rule): Current yield {current_yield_float:.2f} kWh > "
                            f"Original forecast {today_forecast_kwh:.2f} kWh. "
                            f"Adjusted to {adjusted_today_forecast:.2f} kWh."
                        )

                        original_today_forecast = today_forecast_kwh
                        today_forecast_kwh = adjusted_today_forecast

                        if original_today_forecast > 0:
                            adjustment_ratio = today_forecast_kwh / original_today_forecast
                            tomorrow_forecast_kwh = tomorrow_forecast_kwh * adjustment_ratio
                            day_after_forecast_kwh = day_after_forecast_kwh * adjustment_ratio

            except Exception as e:
                _LOGGER.debug(f"Minimum forecast check (Rule) could not be performed: {e}")

            correction_deviation = abs(1.0 - correction_factor)
            confidence_base = max(0.0, 1.0 - correction_deviation * 0.5)
            confidence_today = max(30.0, min(95.0, confidence_base * 85.0))
            confidence_tomorrow = confidence_today * 0.9
            confidence_day_after = confidence_tomorrow * 0.85

            result = ForecastResult(
                forecast_today=today_forecast_kwh,
                forecast_tomorrow=tomorrow_forecast_kwh,
                forecast_day_after_tomorrow=day_after_forecast_kwh,
                confidence_today=confidence_today,
                confidence_tomorrow=confidence_tomorrow,
                confidence_day_after=confidence_day_after,
                method="rule_based_iterative",
                calibrated=True,
                base_capacity=base_capacity_kwp,
                correction_factor=correction_factor,
                best_hour_today=best_hour_today,
                best_hour_production_kwh=(
                    best_hour_production if best_hour_today is not None else None
                ),
                hourly_values=hourly_values,
                forecast_today_raw=raw_today_kwh,
                forecast_tomorrow_raw=raw_tomorrow_kwh,
                forecast_day_after_raw=raw_day_after_kwh,
                safeguard_applied_today=today_forecast_kwh < total_today_kwh,
                safeguard_applied_tomorrow=tomorrow_forecast_kwh < total_tomorrow_kwh,
                safeguard_applied_day_after=day_after_forecast_kwh < total_day_after_kwh,
            )

            _LOGGER.info(
                f"Rule-based (Dumb Consumer) Forecast successful: "
                f"Today={result.forecast_today:.2f} kWh, "
                f"Tomorrow={result.forecast_tomorrow:.2f} kWh, "
                f"Day After={result.forecast_day_after_tomorrow:.2f} kWh, "
                f"Confidence={result.confidence_today:.1f}%, "
                f"(CorrectionFactor={correction_factor:.2f})"
            )

            return result

        except Exception as e:
            _LOGGER.error(
                f"Rule-based (Iterative) forecast calculation failed unexpectedly: {e}",
                exc_info=True,
            )

            _LOGGER.warning("Using emergency fallback for Rule-based forecast.")
            fallback_capacity = self.solar_capacity if self.solar_capacity > 0 else 2.0
            emergency_yield = fallback_capacity * 1.5

            return ForecastResult(
                forecast_today=emergency_yield,
                forecast_tomorrow=emergency_yield * 0.9,
                forecast_day_after_tomorrow=emergency_yield * 0.8,
                confidence_today=20.0,
                confidence_tomorrow=15.0,
                confidence_day_after=10.0,
                method="emergency_fallback_rule",
                calibrated=False,
                base_capacity=fallback_capacity,
            )

    def _get_hour_factor(self, hour: int, hour_datetime: Optional[datetime] = None) -> float:
        """Calculates a factor 0.0 to 1.0 based on sun curve (sine), using astronomy cache @zara"""
        try:

            if self.orchestrator and hour_datetime:
                if not self.orchestrator.is_production_hour(hour_datetime):
                    return 0.0

            if hour_datetime:
                try:
                    cache_manager = get_cache_manager()
                    if cache_manager.is_loaded():

                        date_str = hour_datetime.date().isoformat()
                        window = cache_manager.get_production_window(date_str)

                        if window:
                            window_start_str, window_end_str = window

                            window_start = datetime.fromisoformat(window_start_str).replace(
                                tzinfo=None
                            )
                            window_end = datetime.fromisoformat(window_end_str).replace(tzinfo=None)

                            start_hour = window_start.hour
                            end_hour = window_end.hour

                            if hour < start_hour or hour > end_hour:
                                return 0.0

                            total_duration = end_hour - start_hour
                            if total_duration <= 0:
                                return 0.0

                            hour_pos = (hour + 0.5 - start_hour) / total_duration
                            if not (0.0 <= hour_pos <= 1.0):
                                return 0.0

                            factor = math.sin(hour_pos * math.pi)
                            return max(0.0, factor)

                except Exception as e:
                    _LOGGER.debug(f"Failed to use astronomy cache for hour factor: {e}")

            if hour_datetime:
                month = hour_datetime.month
                if month in [11, 12, 1, 2]:
                    start_hour, end_hour = 7, 16
                elif month in [5, 6, 7, 8]:
                    start_hour, end_hour = 5, 20
                else:
                    start_hour, end_hour = 6, 18
            else:

                if hour >= 22 or hour <= 5:
                    return 0.0
                start_hour = 7
                end_hour = 20

            if hour < start_hour or hour > end_hour:
                return 0.0

            total_duration = end_hour - start_hour
            if total_duration <= 0:
                return 0.0

            hour_pos = (hour + 0.5 - start_hour) / total_duration
            if not (0.0 <= hour_pos <= 1.0):
                return 0.0

            factor = math.sin(hour_pos * math.pi)
            return max(0.0, factor)

        except Exception:
            return 0.5

    async def _calculate_forecast_fallback(
        self,
        hourly_weather_forecast: List[Dict[str, Any]],
        base_capacity_kwp: float,
        correction_factor: float
    ) -> ForecastResult:
        """
        Fallback method using old weather factor calculation.

        Used only if weather_forecast_corrected.json is not available.
        """
        _LOGGER.warning("Using fallback forecast method with weather factor calculation")

        total_today_kwh = 0.0
        total_tomorrow_kwh = 0.0
        total_day_after_kwh = 0.0

        best_hour_today = None
        best_hour_production = 0.0

        now_local = dt_util_safe.now()
        today_date = now_local.date()
        tomorrow_date = today_date + timedelta(days=1)
        day_after_tomorrow_date = today_date + timedelta(days=2)

        hourly_values = []

        for hour_data in hourly_weather_forecast:
            try:
                hour_dt_local = hour_data.get("local_datetime")
                if not hour_dt_local:
                    continue

                if isinstance(hour_dt_local, str):
                    hour_dt_local = dt_util_safe.parse_datetime(hour_dt_local)
                    if not hour_dt_local:
                        continue

                hour_dt_local = dt_util_safe.as_local(hour_dt_local)

                if hour_dt_local < now_local:
                    continue

                hour_date = hour_dt_local.date()
                hour_local = hour_dt_local.hour

                hour_factor = self._get_hour_factor(hour_local, hour_dt_local)

                combined_weather_factor = (
                    self.weather_calculator.calculate_combined_weather_factor(
                        hour_data, include_seasonal=True
                    )
                )

                hourly_kwh = (
                    base_capacity_kwp
                    * self.PEAK_KW_PER_KWP
                    * hour_factor
                    * combined_weather_factor
                    * correction_factor
                )

                hourly_kwh = max(0.0, hourly_kwh)

                hourly_values.append(
                    {
                        "hour": hour_local,
                        "datetime": hour_dt_local.isoformat(),
                        "production_kwh": round(hourly_kwh, 3),
                        "date": hour_date.isoformat(),
                    }
                )

                if hour_date == today_date:
                    total_today_kwh += hourly_kwh
                    if hourly_kwh > best_hour_production:
                        best_hour_today = hour_local
                        best_hour_production = hourly_kwh
                elif hour_date == tomorrow_date:
                    total_tomorrow_kwh += hourly_kwh
                elif hour_date == day_after_tomorrow_date:
                    total_day_after_kwh += hourly_kwh

            except Exception as e_inner:
                _LOGGER.warning(f"Failed to process hour {hour_data.get('local_hour')}: {e_inner}")
                continue

        total_tomorrow_kwh *= self.TOMORROW_DISCOUNT_FACTOR
        total_day_after_kwh *= self.TOMORROW_DISCOUNT_FACTOR * 0.95

        min_forecast_kwh = 0.0
        max_realistic_kwh = base_capacity_kwp * self.MAX_REALISTIC_DAILY_KWH_PER_KWP

        today_forecast_kwh = max(min_forecast_kwh, min(total_today_kwh, max_realistic_kwh))
        tomorrow_forecast_kwh = max(min_forecast_kwh, min(total_tomorrow_kwh, max_realistic_kwh))
        day_after_forecast_kwh = max(min_forecast_kwh, min(total_day_after_kwh, max_realistic_kwh))

        confidence_today = 50.0
        confidence_tomorrow = 45.0
        confidence_day_after = 40.0

        return ForecastResult(
            forecast_today=today_forecast_kwh,
            forecast_tomorrow=tomorrow_forecast_kwh,
            forecast_day_after_tomorrow=day_after_forecast_kwh,
            confidence_today=confidence_today,
            confidence_tomorrow=confidence_tomorrow,
            confidence_day_after=confidence_day_after,
            method="rule_based_fallback",
            calibrated=True,
            base_capacity=base_capacity_kwp,
            correction_factor=correction_factor,
            best_hour_today=best_hour_today,
            best_hour_production_kwh=best_hour_production if best_hour_today is not None else None,
            hourly_values=hourly_values,
        )
