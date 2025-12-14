"""Pattern-Based Learning for Adaptive Solar Forecasting V12.0.0 @zara

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

import asyncio
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

_LOGGER = logging.getLogger(__name__)

class PatternLearner:
    """Learn and apply patterns from historical solar production data"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.patterns_file = data_dir / "ml" / "learned_patterns.json"
        self.patterns: Optional[Dict] = None

    async def load_patterns(self) -> Dict:
        """Load patterns from file @zara"""

        def _load_sync():
            try:
                if not self.patterns_file.exists():
                    _LOGGER.warning("Patterns file not found, creating default learned_patterns.json")
                    default_patterns = self._get_default_patterns()

                    self.patterns_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.patterns_file, "w") as f:
                        json.dump(default_patterns, f, indent=2)
                    _LOGGER.info("✓ Default learned_patterns.json created")

                    return default_patterns

                with open(self.patterns_file, "r") as f:
                    patterns = json.load(f)

                _LOGGER.info(
                    f"Patterns loaded: {patterns['metadata']['total_learning_days']} learning days, "
                    f"confidence varies by pattern"
                )
                return patterns

            except Exception as e:
                _LOGGER.error(f"Error loading patterns: {e}")
                return self._get_default_patterns()

        loop = asyncio.get_running_loop()
        self.patterns = await loop.run_in_executor(None, _load_sync)
        return self.patterns

    def _get_default_patterns(self) -> Dict:
        """Return default bootstrap patterns (used until learning starts) @zara"""
        return {
            "version": "1.0.0",
            "geometry_factors": {
                "sun_elevation_ranges": {
                    "0_5": {"factor": 2.5, "samples": 0, "confidence": 0.3},
                    "5_10": {"factor": 2.2, "samples": 0, "confidence": 0.3},
                    "10_15": {"factor": 1.9, "samples": 0, "confidence": 0.3},
                    "15_20": {"factor": 1.6, "samples": 0, "confidence": 0.3},
                    "20_25": {"factor": 1.4, "samples": 0, "confidence": 0.3},
                    "25_30": {"factor": 1.2, "samples": 0, "confidence": 0.3},
                    "30_plus": {"factor": 1.1, "samples": 0, "confidence": 0.3},
                }
            },
            "geometry_corrections": {
                "monthly": {}
            },
            "cloud_impacts": {"hour_patterns": {}},
            "seasonal_adjustments": {"months": {}},
            "metadata": {"total_learning_days": 0}
        }

    def get_geometry_factor(
        self,
        sun_elevation: float,
        month: int = None,
        max_elevation_of_day: float = None
    ) -> Tuple[float, float]:
        """
        Get geometry correction factor for tilted panels

        Now supports peak-detection for more accurate factors

        Args:
            sun_elevation: Sun elevation in degrees
            month: Optional month (1-12) for seasonal adjustment
            max_elevation_of_day: Optional max elevation of the day for peak detection

        Returns:
            (factor, confidence) tuple
        """
        if not self.patterns:
            return 1.0, 0.0

        if sun_elevation < 5:
            bucket_base = "0_5"
        elif sun_elevation < 10:
            bucket_base = "5_10"
        elif sun_elevation < 15:
            bucket_base = "10_15"
        elif sun_elevation < 20:
            bucket_base = "15_20"
        elif sun_elevation < 25:
            bucket_base = "20_25"
        elif sun_elevation < 30:
            bucket_base = "25_30"
        else:
            bucket_base = "30_plus"

        is_peak = False
        if max_elevation_of_day and sun_elevation >= max_elevation_of_day * 0.95:
            is_peak = True
            bucket = f"{bucket_base}_peak"
        else:
            bucket = f"{bucket_base}_side"

        geo_ranges = self.patterns["geometry_factors"]["sun_elevation_ranges"]
        geo_data = geo_ranges.get(bucket)

        if not geo_data:

            geo_data = geo_ranges.get(bucket_base, {"factor": 1.0, "confidence": 0.0})
            bucket = bucket_base

        base_factor = geo_data["factor"]
        confidence = geo_data["confidence"]

        if is_peak and bucket != bucket_base:
            _LOGGER.debug(
                f"Peak-aware geometry: elevation {sun_elevation:.1f}° is PEAK "
                f"(max {max_elevation_of_day:.1f}°), using bucket '{bucket}' "
                f"factor {base_factor:.2f}"
            )

        if month and "geometry_corrections" in self.patterns:
            monthly_corrections = self.patterns["geometry_corrections"].get("monthly", {})
            month_str = str(month)
            correction_data = monthly_corrections.get(month_str)

            if correction_data and correction_data.get("confidence", 0) >= 0.4:
                correction_factor = correction_data["correction_factor"]
                base_factor = base_factor * correction_factor

                _LOGGER.debug(
                    f"Geometry correction applied: {geo_data['factor']:.2f} × {correction_factor:.2f} = {base_factor:.2f} "
                    f"(month {month}, confidence {correction_data['confidence']:.0%})"
                )

        if month and "seasonal_adjustments" in self.patterns:
            seasonal = self.patterns["seasonal_adjustments"]["months"].get(str(month), {})
            if seasonal.get("samples", 0) > 5:

                if sun_elevation < 15:

                    seasonal_factor = seasonal.get("morning_boost", 1.0)
                else:

                    seasonal_factor = seasonal.get("midday_factor", 1.0)

                base_factor = (base_factor + seasonal_factor) / 2

        return base_factor, confidence

    def get_cloud_impact(
        self,
        hour: int,
        cloud_cover: float,
        cloud_cover_variability: float = None
    ) -> Tuple[float, float]:
        """Get cloud impact efficiency for specific hour and cloud coverage @zara

        Args:
            hour: Hour of day (0-23)
            cloud_cover: Cloud cover percentage (0-100)
            cloud_cover_variability: Optional std dev of cloud cover over the hour.
                                     High variability indicates potential for sun breakthroughs.
        """
        if not self.patterns:
            base_efficiency = self._default_cloud_efficiency(cloud_cover)
            # Apply breakthrough factor for variable conditions
            if cloud_cover_variability is not None and cloud_cover_variability > 15:
                breakthrough_factor = self._calculate_breakthrough_factor(
                    cloud_cover, cloud_cover_variability
                )
                base_efficiency = min(1.0, base_efficiency * breakthrough_factor)
            return base_efficiency, 0.3

        hour_patterns = self.patterns.get("cloud_impacts", {}).get("hour_patterns", {})
        hour_data = hour_patterns.get(str(hour))

        if not hour_data:
            base_efficiency = self._default_cloud_efficiency(cloud_cover)
            if cloud_cover_variability is not None and cloud_cover_variability > 15:
                breakthrough_factor = self._calculate_breakthrough_factor(
                    cloud_cover, cloud_cover_variability
                )
                base_efficiency = min(1.0, base_efficiency * breakthrough_factor)
            return base_efficiency, 0.3

        if cloud_cover < 30:
            bucket = "0_30"
        elif cloud_cover < 70:
            bucket = "30_70"
        else:
            bucket = "70_100"

        cloud_data = hour_data.get(bucket, {
            "efficiency": self._default_cloud_efficiency(cloud_cover),
            "confidence": 0.3
        })

        efficiency = cloud_data.get("efficiency", 0.5)
        confidence = cloud_data.get("confidence", 0.3)

        # Apply breakthrough factor for variable/mixed conditions
        if cloud_cover_variability is not None and cloud_cover_variability > 15:
            breakthrough_factor = self._calculate_breakthrough_factor(
                cloud_cover, cloud_cover_variability
            )
            efficiency = min(1.0, efficiency * breakthrough_factor)
            _LOGGER.debug(
                f"Cloud breakthrough factor applied: hour={hour}, "
                f"cloud_cover={cloud_cover:.0f}%, variability={cloud_cover_variability:.1f}, "
                f"factor={breakthrough_factor:.2f}, final_efficiency={efficiency:.2f}"
            )

        return efficiency, confidence

    def _calculate_breakthrough_factor(
        self,
        cloud_cover: float,
        variability: float
    ) -> float:
        """Calculate breakthrough factor for variable cloud conditions.

        When clouds are variable (high std dev), there's potential for sun
        to break through, increasing efficiency above what static cloud cover suggests.

        Args:
            cloud_cover: Average cloud cover (0-100%)
            variability: Standard deviation of cloud cover

        Returns:
            Multiplier for efficiency (1.0 = no change, >1.0 = breakthrough potential)
        """
        # Only apply for partially cloudy conditions (40-90%)
        if cloud_cover < 40 or cloud_cover > 90:
            return 1.0

        # Higher variability = more breakthrough potential
        # variability of 20 gives ~1.15x, 30 gives ~1.25x, 40 gives ~1.35x
        variability_factor = 1.0 + (variability - 15) * 0.01

        # Less breakthrough potential at very high cloud cover
        cloud_damping = 1.0 - (max(0, cloud_cover - 70) / 60)  # 1.0 at 70%, 0.67 at 90%

        breakthrough = 1.0 + (variability_factor - 1.0) * cloud_damping

        return min(1.5, max(1.0, breakthrough))  # Clamp to [1.0, 1.5]

    def _default_cloud_efficiency(self, cloud_cover: float) -> float:
        """Default cloud efficiency curve (bootstrap values) @zara"""
        if cloud_cover < 30:
            return 0.95
        elif cloud_cover < 70:
            return 0.65
        else:
            return 0.35

    def get_strategy_mode(self, cloud_cover: float) -> str:
        """Determine which strategy to use based on conditions @zara"""

        if cloud_cover > 80:
            return "cloudy"
        elif cloud_cover > 40:
            return "mixed"  # Wechselhaft - lernt sowohl Geometrie als auch Cloud-Impacts
        else:
            return "sunny"

    async def update_pattern_from_day(
        self,
        target_date: date,
        actual_production_kwh: float,
        hourly_actuals: Dict[int, float],
        hourly_conditions: Dict[int, Dict],
        astronomy_data: Dict,
        peak_power_w: float = None,
        system_capacity_kwp: float = None
    ):
        """
        Update patterns based on a completed day's data

        Args:
            target_date: Date of the data
            actual_production_kwh: Total production for the day
            hourly_actuals: Production by hour {hour: kwh}
            hourly_conditions: Weather conditions by hour {hour: {clouds, temp, etc}}
            astronomy_data: Astronomy data for the day
            peak_power_w: Peak power in Watts (used for breakthrough detection)
            system_capacity_kwp: System capacity in kWp (for breakthrough ratio calculation)
        """
        if not self.patterns:
            await self.load_patterns()

        _LOGGER.info(f"Learning from {target_date}: {actual_production_kwh:.2f} kWh")

        avg_clouds = sum(
            h.get("cloud_cover_percent", 100) for h in hourly_conditions.values()
        ) / max(len(hourly_conditions), 1)

        strategy_mode = self.get_strategy_mode(avg_clouds)

        if strategy_mode == "sunny":
            # Klarer Tag - nur Geometrie lernen
            await self._update_geometry_factors(
                target_date, hourly_actuals, astronomy_data
            )
            self.patterns["metadata"]["clear_sky_days_detected"] = (
                self.patterns["metadata"].get("clear_sky_days_detected", 0) + 1
            )
        elif strategy_mode == "mixed":
            # Wechselhafter Tag - BEIDE Faktoren lernen (wichtig für Cloud-Durchbrüche!)
            await self._update_geometry_factors(
                target_date, hourly_actuals, astronomy_data
            )
            await self._update_cloud_impacts(
                hourly_actuals, hourly_conditions, astronomy_data,
                peak_power_w=peak_power_w,
                system_capacity_kwp=system_capacity_kwp
            )
            self.patterns["metadata"]["mixed_days_detected"] = (
                self.patterns["metadata"].get("mixed_days_detected", 0) + 1
            )
            _LOGGER.info(
                f"Mixed day learning: Updated both geometry and cloud impacts "
                f"(avg clouds: {avg_clouds:.0f}%, peak_power: {peak_power_w or 'N/A'}W)"
            )
        else:
            # Bewölkter Tag - nur Cloud-Impacts lernen
            await self._update_cloud_impacts(
                hourly_actuals, hourly_conditions, astronomy_data,
                peak_power_w=peak_power_w,
                system_capacity_kwp=system_capacity_kwp
            )
            self.patterns["metadata"]["cloudy_days_detected"] = (
                self.patterns["metadata"].get("cloudy_days_detected", 0) + 1
            )

        self.patterns["metadata"]["total_learning_days"] = (
            self.patterns["metadata"].get("total_learning_days", 0) + 1
        )
        self.patterns["metadata"]["last_pattern_update"] = datetime.now().isoformat()
        self.patterns["last_updated"] = datetime.now().isoformat()

        await self._save_patterns()

    async def _update_geometry_factors(
        self,
        target_date: date,
        hourly_actuals: Dict[int, float],
        astronomy_data: Dict
    ):
        """Update geometry factors from clear-sky day

        Now writes to peak-aware buckets for better accuracy
        """
        hourly_astro = astronomy_data.get("hourly", {})
        month = target_date.month

        max_elevation = 0.0
        for hour_str, hour_data in hourly_astro.items():
            elev = hour_data.get("elevation_deg", 0)
            if elev > max_elevation:
                max_elevation = elev

        _LOGGER.debug(f"Learning geometry factors: max_elevation={max_elevation:.1f}°")

        for hour, actual_kwh in hourly_actuals.items():
            if actual_kwh < 0.01:
                continue

            hour_astro = hourly_astro.get(str(hour), {})
            theoretical_kwh = hour_astro.get("theoretical_max_pv_kwh", 0)
            elevation = hour_astro.get("elevation_deg", 0)

            if theoretical_kwh < 0.01 or elevation < 0:
                continue

            measured_factor = actual_kwh / theoretical_kwh

            measured_factor = max(0.8, min(5.0, measured_factor))

            if elevation < 5:
                bucket_base = "0_5"
            elif elevation < 10:
                bucket_base = "5_10"
            elif elevation < 15:
                bucket_base = "10_15"
            elif elevation < 20:
                bucket_base = "15_20"
            elif elevation < 25:
                bucket_base = "20_25"
            elif elevation < 30:
                bucket_base = "25_30"
            else:
                bucket_base = "30_plus"

            is_peak = False
            if max_elevation > 0 and elevation >= max_elevation * 0.95:
                is_peak = True
                bucket = f"{bucket_base}_peak"
            else:
                bucket = f"{bucket_base}_side"

            geo_ranges = self.patterns["geometry_factors"]["sun_elevation_ranges"]

            if bucket not in geo_ranges:
                geo_ranges[bucket] = {
                    "factor": 1.0,
                    "samples": 0,
                    "confidence": 0.3,
                    "description": f"{bucket_base} ({'peak' if is_peak else 'side'}) hours"
                }

            current = geo_ranges[bucket]

            samples = current["samples"]
            if samples == 0:

                new_factor = measured_factor
            else:

                alpha = 0.1
                new_factor = alpha * measured_factor + (1 - alpha) * current["factor"]

            geo_ranges[bucket]["factor"] = round(new_factor, 3)
            geo_ranges[bucket]["samples"] = samples + 1
            geo_ranges[bucket]["confidence"] = min(0.95, 0.3 + samples * 0.05)

            _LOGGER.debug(
                f"Geometry factor updated: elevation {elevation:.1f}° "
                f"(max {max_elevation:.1f}°, bucket {bucket}, is_peak={is_peak}), "
                f"factor {current['factor']:.2f} → {new_factor:.2f} "
                f"(samples: {samples + 1}, confidence: {geo_ranges[bucket]['confidence']:.2f})"
            )

    async def _update_cloud_impacts(
        self,
        hourly_actuals: Dict[int, float],
        hourly_conditions: Dict[int, Dict],
        astronomy_data: Dict,
        peak_power_w: float = None,
        system_capacity_kwp: float = None
    ):
        """Update cloud impact patterns from cloudy/mixed day

        Args:
            hourly_actuals: Actual production by hour
            hourly_conditions: Weather conditions by hour
            astronomy_data: Astronomy data for the day
            peak_power_w: Peak power in Watts (for breakthrough detection)
            system_capacity_kwp: System capacity for normalization
        """
        hourly_astro = astronomy_data.get("hourly", {})

        # Calculate if there was a significant peak (breakthrough indicator)
        has_breakthrough = False
        if peak_power_w and system_capacity_kwp:
            # If peak was > 50% of capacity despite clouds, there was breakthrough
            peak_ratio = peak_power_w / (system_capacity_kwp * 1000)
            avg_clouds = sum(
                h.get("cloud_cover_percent", 100) for h in hourly_conditions.values()
            ) / max(len(hourly_conditions), 1)

            if peak_ratio > 0.5 and avg_clouds > 60:
                has_breakthrough = True
                _LOGGER.info(
                    f"Cloud breakthrough detected: peak={peak_power_w:.0f}W "
                    f"({peak_ratio*100:.0f}% of capacity) with {avg_clouds:.0f}% avg clouds"
                )

        for hour, actual_kwh in hourly_actuals.items():
            if actual_kwh < 0.01:
                continue

            conditions = hourly_conditions.get(hour, {})
            cloud_cover = conditions.get("cloud_cover_percent", 100)

            hour_astro = hourly_astro.get(str(hour), {})
            theoretical_kwh = hour_astro.get("theoretical_max_pv_kwh", 0)

            if theoretical_kwh < 0.01:
                continue

            efficiency = actual_kwh / theoretical_kwh
            # Allow efficiency > 1.0 for cases where diffuse light performs better than expected
            # Cap at 3.0 to prevent extreme outliers from corrupting the model
            efficiency = max(0.05, min(3.0, efficiency))

            if cloud_cover < 30:
                bucket = "0_30"
            elif cloud_cover < 70:
                bucket = "30_70"
            else:
                bucket = "70_100"

            hour_str = str(hour)
            if hour_str not in self.patterns["cloud_impacts"]["hour_patterns"]:
                self.patterns["cloud_impacts"]["hour_patterns"][hour_str] = {
                    "0_30": {"efficiency": 0.95, "samples": 0, "confidence": 0.3},
                    "30_70": {"efficiency": 0.65, "samples": 0, "confidence": 0.3},
                    "70_100": {"efficiency": 0.35, "samples": 0, "confidence": 0.3}
                }

            current = self.patterns["cloud_impacts"]["hour_patterns"][hour_str][bucket]
            samples = current["samples"]

            # Use higher learning rate when breakthrough is detected
            # This helps the model learn that high clouds don't always mean low production
            if has_breakthrough and bucket == "70_100" and efficiency > current["efficiency"]:
                alpha = 0.25  # Learn faster from positive surprises
            else:
                alpha = 0.15

            if samples == 0:
                new_efficiency = efficiency
            else:
                new_efficiency = alpha * efficiency + (1 - alpha) * current["efficiency"]

            current["efficiency"] = round(new_efficiency, 3)
            current["samples"] = samples + 1
            current["confidence"] = min(0.95, 0.3 + samples * 0.05)

            if has_breakthrough and bucket == "70_100":
                _LOGGER.debug(
                    f"Breakthrough learning hour {hour}: cloud={cloud_cover:.0f}%, "
                    f"efficiency {current['efficiency']:.2f} → {new_efficiency:.2f} (alpha={alpha})"
                )

    async def _save_patterns(self):
        """Save patterns to file @zara"""

        def _save_sync():
            try:
                self.patterns_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.patterns_file, "w", encoding="utf-8") as f:
                    json.dump(self.patterns, f, indent=2, ensure_ascii=False)
                _LOGGER.debug(f"Patterns saved to {self.patterns_file}")
            except Exception as e:
                _LOGGER.error(f"Error saving patterns: {e}", exc_info=True)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _save_sync)

    async def apply_aggressive_correction(
        self,
        target_date: date,
        forecast_raw: float,
        actual_kwh: float,
        avg_cloud_cover: float,
    ) -> None:
        """Aggressive correction system - compares raw forecast vs actual and corrects geometry factors."""
        if not self.patterns:
            await self.load_patterns()

        if forecast_raw <= 0 or actual_kwh <= 0:
            _LOGGER.debug("Skipping aggressive correction: zero values")
            return

        error_ratio = abs(forecast_raw - actual_kwh) / forecast_raw
        ERROR_THRESHOLD = 0.10

        if error_ratio < ERROR_THRESHOLD:
            _LOGGER.debug(f"Aggressive correction not needed: error {error_ratio:.1%} < threshold {ERROR_THRESHOLD:.0%}")
            return

        is_clear_sky = avg_cloud_cover < 50

        if not is_clear_sky:
            _LOGGER.debug(f"Skipping aggressive correction: cloudy day ({avg_cloud_cover:.0f}% clouds)")
            return

        correction_ratio = actual_kwh / forecast_raw

        MIN_CORRECTION_RATIO = 0.5
        MAX_CORRECTION_RATIO = 1.8

        if correction_ratio < MIN_CORRECTION_RATIO or correction_ratio > MAX_CORRECTION_RATIO:
            _LOGGER.warning(
                f"⚠️ Skipping aggressive correction for {target_date}: "
                f"correction_ratio {correction_ratio:.2f} outside valid range [{MIN_CORRECTION_RATIO}-{MAX_CORRECTION_RATIO}]. "
                f"Forecast: {forecast_raw:.2f} kWh, Actual: {actual_kwh:.2f} kWh. "
                f"This indicates a forecast bug, not a geometry issue."
            )
            return

        month = str(target_date.month)

        if "geometry_corrections" not in self.patterns:
            self.patterns["geometry_corrections"] = {"monthly": {}}

        monthly_corrections = self.patterns["geometry_corrections"]["monthly"]

        if month not in monthly_corrections:
            monthly_corrections[month] = {
                "correction_factor": 1.0,
                "rolling_7d_history": [],
                "samples": 0,
                "confidence": 0.0,
                "last_updated": None
            }

        correction_data = monthly_corrections[month]
        history = correction_data["rolling_7d_history"]

        history.append(correction_ratio)
        if len(history) > 7:
            history.pop(0)

        avg_correction = sum(history) / len(history)

        correction_data["correction_factor"] = round(avg_correction, 3)
        correction_data["rolling_7d_history"] = history
        correction_data["samples"] = correction_data.get("samples", 0) + 1
        correction_data["confidence"] = min(0.95, 0.3 + len(history) * 0.1)
        correction_data["last_updated"] = datetime.now().isoformat()

        await self._save_patterns()

        _LOGGER.info(
            f"Correction learned: Forecast {forecast_raw:.2f} kWh vs Actual {actual_kwh:.2f} kWh (error: {error_ratio:.0%}). "
            f"Ratio: {correction_ratio:.2f} → 7d avg: {avg_correction:.2f} "
            f"(month {month}, {len(history)}/7 samples, confidence {correction_data['confidence']:.0%})"
        )
