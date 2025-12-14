"""Shadow Detection & Performance Loss Analysis V12.0.0 @zara

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
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

class ShadowDetector:
    """
    Detects and analyzes solar array shading using multiple methods

    Methods:
    1. Theory Ratio - Compare actual vs theoretical_max (universal, fast)
    2. Correlation-Based Detection - Compare Lux drop vs Production drop (accurate)
    3. Ensemble - Weighted combination of both methods

    V12.0.0: Uses correlation between Lux sensor and production to distinguish
    between cloud cover (affects both equally) and local shadows (only affects panels).
    """

    def __init__(
        self,
        data_manager=None,
        panel_groups: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize shadow detector @zara"""
        self.data_manager = data_manager
        self._panel_groups = panel_groups or []
        self._use_panel_groups = bool(panel_groups)

        self.SHADOW_THRESHOLD_LIGHT = 0.05
        self.SHADOW_THRESHOLD_MODERATE = 0.15
        self.SHADOW_THRESHOLD_HEAVY = 0.30

        self.CORRELATION_THRESHOLD_BASE = 0.15

        self.WEIGHT_THEORY = 0.60
        self.WEIGHT_SENSOR = 0.40

        if self._use_panel_groups:
            _LOGGER.info(
                f"Shadow Detector initialized with {len(self._panel_groups)} panel groups"
            )
        else:
            _LOGGER.info("Shadow Detector initialized (V12.0.0: Correlation-Based Detection)")

    async def detect_shadow_theory_ratio(
        self,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Method 1: Theory Ratio - Fast & Universal

        Compares actual production with theoretical maximum from clear sky model.
        Works WITHOUT any external sensors (astronomy cache only).

        Args:
            prediction: Hourly prediction dict with actual_kwh and astronomy data

        Returns:
            Shadow detection result with classification and confidence
        """
        try:
            actual_kwh = prediction.get("actual_kwh", 0)
            astronomy = prediction.get("astronomy", {})
            theoretical_max = astronomy.get("theoretical_max_kwh", 0)

            if theoretical_max < 0.01:
                return {
                    "method": "theory_ratio",
                    "shadow_type": "night",
                    "shadow_percent": 100.0,
                    "confidence": 1.0,
                    "efficiency_ratio": 0.0,
                    "interpretation": "No solar production expected (sun below horizon)"
                }

            efficiency_ratio = actual_kwh / theoretical_max

            shadow_ratio = max(0.0, 1.0 - efficiency_ratio)
            shadow_percent = shadow_ratio * 100.0

            if shadow_ratio < self.SHADOW_THRESHOLD_LIGHT:
                shadow_type = "none"
                confidence = 0.95
                interpretation = "Excellent production - minimal losses"
                possible_causes = []
            elif shadow_ratio < self.SHADOW_THRESHOLD_MODERATE:
                shadow_type = "light"
                confidence = 0.85
                interpretation = "Light shadowing or atmospheric effects"
                possible_causes = ["Thin clouds", "Dust/pollen on panels", "Morning/evening haze"]
            elif shadow_ratio < self.SHADOW_THRESHOLD_HEAVY:
                shadow_type = "moderate"
                confidence = 0.80
                interpretation = "Moderate shadowing detected"
                possible_causes = ["Clouds", "Building shadows", "Horizon obstruction", "Panel soiling"]
            else:
                shadow_type = "heavy"
                confidence = 0.75
                interpretation = "Heavy shadowing or system issue"
                possible_causes = ["Dense clouds", "Major obstruction", "Rain", "Snow on panels", "System fault"]

            loss_kwh = theoretical_max - actual_kwh

            result = {
                "method": "theory_ratio",
                "shadow_type": shadow_type,
                "shadow_percent": round(shadow_percent, 1),
                "confidence": confidence,
                "efficiency_ratio": round(efficiency_ratio, 3),
                "loss_kwh": round(loss_kwh, 3),
                "loss_percent": round(shadow_percent, 1),
                "interpretation": interpretation,
                "possible_causes": possible_causes,
                "theoretical_max_kwh": theoretical_max,
                "actual_kwh": actual_kwh
            }

            per_group_data = astronomy.get("theoretical_max_per_group", [])
            if per_group_data and self._use_panel_groups:
                per_group_analysis = self._analyze_per_group_shadows(
                    actual_kwh, theoretical_max, per_group_data
                )
                result["per_group_analysis"] = per_group_analysis

            return result

        except Exception as e:
            _LOGGER.error(f"Theory ratio shadow detection failed: {e}", exc_info=True)
            return {
                "method": "theory_ratio",
                "error": str(e),
                "shadow_type": "error",
                "confidence": 0.0
            }

    def _analyze_per_group_shadows(
        self,
        actual_kwh: float,
        theoretical_max_total: float,
        per_group_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Analyze shadow detection per panel group. @zara

        Since we only have total actual_kwh, we estimate per-group performance
        by assuming the ratio is proportional to theoretical max per group.
        """
        per_group_analysis = []

        if not per_group_data or theoretical_max_total <= 0:
            return per_group_analysis

        total_group_theoretical = sum(
            g.get("theoretical_kwh", 0) for g in per_group_data
        )
        if total_group_theoretical <= 0:
            return per_group_analysis

        overall_efficiency = actual_kwh / theoretical_max_total if theoretical_max_total > 0 else 0

        for group in per_group_data:
            group_name = group.get("name", "Unknown")
            group_theoretical = group.get("theoretical_kwh", 0)
            group_tilt = group.get("tilt_deg", 30)
            group_azimuth = group.get("azimuth_deg", 180)

            estimated_actual = group_theoretical * overall_efficiency
            group_shadow_percent = (1.0 - overall_efficiency) * 100 if overall_efficiency < 1.0 else 0

            if group_shadow_percent < 5:
                shadow_type = "none"
                interpretation = "Normale Produktion"
            elif group_shadow_percent < 15:
                shadow_type = "light"
                interpretation = "Leichte Verschattung"
            elif group_shadow_percent < 30:
                shadow_type = "moderate"
                interpretation = "Mäßige Verschattung"
            else:
                shadow_type = "heavy"
                interpretation = "Starke Verschattung"

            per_group_analysis.append({
                "name": group_name,
                "tilt_deg": group_tilt,
                "azimuth_deg": group_azimuth,
                "theoretical_kwh": round(group_theoretical, 4),
                "estimated_actual_kwh": round(estimated_actual, 4),
                "shadow_type": shadow_type,
                "shadow_percent": round(group_shadow_percent, 1),
                "interpretation": interpretation,
            })

        return per_group_analysis

    async def detect_shadow_sensor_fusion(
        self,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Method 2: Correlation-Based Detection - High Accuracy (V12.0.0)

        Uses CORRELATION between Lux drop and Production drop to distinguish:
        - CLOUDS: Affect both Lux sensor AND panels equally (correlated drops)
        - LOCAL SHADOW: Only affects panels, not the Lux sensor (uncorrelated)

        Key insight: Lux sensor is usually NOT at the panels!
        - If Lux drops 40% AND Production drops 40% → CLOUDS
        - If Lux drops 10% BUT Production drops 60% → PANEL SHADOW

        Args:
            prediction: Hourly prediction dict with weather, astronomy, and optional sensor data

        Returns:
            Shadow detection result with cloud/obstruction classification
        """
        try:

            weather = prediction.get("weather_corrected") or prediction.get("weather_forecast", {})
            astronomy = prediction.get("astronomy", {})
            sensor = prediction.get("sensor_actual", {})

            cloud_cover = weather.get("clouds", weather.get("cloud_cover", 0))
            actual_kwh = prediction.get("actual_kwh", 0)
            theoretical_max = astronomy.get("theoretical_max_kwh", 0)
            clear_sky_rad = astronomy.get("clear_sky_radiation_wm2", 0)
            sun_elevation = astronomy.get("sun_elevation_deg", astronomy.get("elevation_deg", 0))
            temperature = weather.get("temperature", 15)

            if theoretical_max < 0.01:
                return {
                    "method": "sensor_fusion",
                    "shadow_type": "night",
                    "confidence": 1.0,
                    "mode": "night"
                }

            sensor_lux = sensor.get("lux")
            has_lux_sensor = sensor_lux is not None and sensor_lux >= 0

            if has_lux_sensor:

                return await self._detect_with_lux_correlation(
                    prediction, cloud_cover, clear_sky_rad, sensor_lux,
                    actual_kwh, theoretical_max, sun_elevation, temperature
                )
            else:

                return await self._detect_with_cloud_cover(
                    prediction, cloud_cover, actual_kwh, theoretical_max
                )

        except Exception as e:
            _LOGGER.error(f"Sensor fusion shadow detection failed: {e}", exc_info=True)
            return {
                "method": "sensor_fusion",
                "error": str(e),
                "shadow_type": "error",
                "confidence": 0.0
            }

    async def _detect_with_lux_correlation(
        self,
        prediction: Dict[str, Any],
        cloud_cover: float,
        clear_sky_rad: float,
        sensor_lux: float,
        actual_kwh: float,
        theoretical_max: float,
        sun_elevation: float,
        temperature: float
    ) -> Dict[str, Any]:
        """
        V12.0.0: CORRELATION-BASED shadow detection WITH Lux sensor

        Key insight: The Lux sensor is usually NOT at the solar panels!

        Logic:
        - If Lux drops AND Production drops by similar amounts → CLOUDS
        - If Production drops MORE than Lux → PANEL SHADOW (building/tree)
        - If Lux drops MORE than Production → SENSOR SHADOW (rare)

        This distinguishes between regional effects (clouds) and local effects (shadows).
        """

        if sun_elevation < 10:
            return {
                "method": "sensor_fusion",
                "mode": "lux_sensor_low_angle",
                "shadow_type": "low_sun",
                "shadow_percent": 0.0,
                "confidence": 0.3,
                "root_cause": "low_sun_angle",
                "interpretation": f"Sun elevation too low ({sun_elevation:.1f}°) for reliable detection",
                "lux_analysis": {
                    "sensor_lux": round(sensor_lux, 1),
                    "clear_sky_lux": round(clear_sky_rad * 75.0, 1),
                    "sun_elevation_deg": round(sun_elevation, 1)
                }
            }

        if temperature < 0 and actual_kwh < theoretical_max * 0.15:
            efficiency_ratio = actual_kwh / theoretical_max if theoretical_max > 0 else 0
            shadow_percent = max(0.0, (1.0 - efficiency_ratio) * 100.0)
            return {
                "method": "sensor_fusion",
                "mode": "lux_sensor_frost",
                "shadow_type": "frost_snow",
                "shadow_percent": round(shadow_percent, 1),
                "confidence": 0.85,
                "root_cause": "panel_frost_snow",
                "interpretation": f"Possible frost/snow on panels (temp: {temperature:.1f}°C, only {efficiency_ratio*100:.0f}% efficiency)",
                "lux_analysis": {
                    "sensor_lux": round(sensor_lux, 1),
                    "temperature_c": round(temperature, 1)
                }
            }

        clear_sky_lux = clear_sky_rad * 75.0

        if clear_sky_lux < 100:

            return {
                "method": "sensor_fusion",
                "mode": "lux_sensor",
                "shadow_type": "low_radiation",
                "shadow_percent": 0.0,
                "confidence": 0.5,
                "root_cause": "low_radiation",
                "interpretation": "Clear-sky radiation too low for meaningful analysis"
            }

        lux_drop = 1.0 - (sensor_lux / clear_sky_lux)
        production_drop = 1.0 - (actual_kwh / theoretical_max)

        lux_drop = max(0.0, min(1.0, lux_drop))
        production_drop = max(0.0, min(1.0, production_drop))

        threshold = self.CORRELATION_THRESHOLD_BASE

        if cloud_cover > 80:
            threshold += 0.10
        elif cloud_cover > 50:
            threshold += 0.05

        if sun_elevation < 20:
            threshold += 0.10
        elif sun_elevation < 30:
            threshold += 0.05

        threshold = min(threshold, 0.35)

        correlation_diff = production_drop - lux_drop

        if abs(correlation_diff) < threshold:

            shadow_type = "weather_clouds"
            root_cause = "weather_clouds"
            confidence = 0.88
            interpretation = f"Cloud-based shading (Lux: -{lux_drop*100:.0f}%, Prod: -{production_drop*100:.0f}% - correlated)"
            possible_causes = ["Cloud cover", "Overcast sky", "Atmospheric haze"]

        elif correlation_diff > threshold:

            shadow_type = "panel_shadow"
            root_cause = "building_tree_obstruction"
            confidence = 0.82
            interpretation = f"Local panel shadow (Lux: -{lux_drop*100:.0f}%, Prod: -{production_drop*100:.0f}% - uncorrelated)"
            possible_causes = ["Building shadow", "Tree obstruction", "Horizon blocking", "Panel soiling"]

        else:

            shadow_type = "sensor_shadow"
            root_cause = "weather_better_than_forecast"
            confidence = 0.70
            interpretation = f"Lux sensor affected more than panels (Lux: -{lux_drop*100:.0f}%, Prod: -{production_drop*100:.0f}%)"
            possible_causes = ["Lux sensor in partial shadow", "Panels in better position"]

        shadow_percent = production_drop * 100.0

        if shadow_percent < self.SHADOW_THRESHOLD_LIGHT * 100:
            shadow_severity = "none"
        elif shadow_percent < self.SHADOW_THRESHOLD_MODERATE * 100:
            shadow_severity = "light"
        elif shadow_percent < self.SHADOW_THRESHOLD_HEAVY * 100:
            shadow_severity = "moderate"
        else:
            shadow_severity = "heavy"

        return {
            "method": "sensor_fusion",
            "mode": "lux_sensor",
            "shadow_type": shadow_severity,
            "shadow_percent": round(shadow_percent, 1),
            "confidence": confidence,
            "interpretation": interpretation,
            "root_cause": root_cause,
            "possible_causes": possible_causes,
            "lux_analysis": {
                "sensor_lux": round(sensor_lux, 1),
                "clear_sky_lux": round(clear_sky_lux, 1),
                "lux_drop_percent": round(lux_drop * 100, 1),
                "production_drop_percent": round(production_drop * 100, 1),
                "correlation_diff_percent": round(correlation_diff * 100, 1),
                "threshold_used_percent": round(threshold * 100, 1),
                "sun_elevation_deg": round(sun_elevation, 1),
                "cloud_cover_percent": round(cloud_cover, 1)
            }
        }

    async def _detect_with_cloud_cover(
        self,
        prediction: Dict[str, Any],
        cloud_cover: float,
        actual_kwh: float,
        theoretical_max: float
    ) -> Dict[str, Any]:
        """
        Fallback detection WITHOUT Lux sensor (cloud-based only)

        Estimates expected production based on cloud cover and compares with actual.
        Less accurate than correlation-based detection but still useful.

        V12.0.0: cloud_cover is now correctly read from "clouds" field.
        """

        cloud_factor = math.exp(-0.035 * cloud_cover)
        cloud_factor = max(0.03, cloud_factor)
        expected_kwh = theoretical_max * cloud_factor

        if expected_kwh > 0.01:
            deviation = actual_kwh - expected_kwh
            deviation_percent = (deviation / expected_kwh) * 100.0
        else:
            deviation_percent = 0.0

        if abs(deviation_percent) < 15:
            shadow_type = "match_forecast"
            confidence = 0.75
            interpretation = "Production matches cloud cover forecast"
            root_cause = "weather_clouds"
        elif deviation_percent < -25:
            shadow_type = "worse_than_forecast"
            confidence = 0.70
            interpretation = f"Production lower than cloud forecast suggests ({abs(deviation_percent):.0f}% worse)"
            root_cause = "possible_obstruction"
            possible_causes = ["Local obstruction", "Panel soiling", "Underperforming system", "Thicker clouds than forecast"]
        elif deviation_percent > 25:
            shadow_type = "better_than_forecast"
            confidence = 0.70
            interpretation = "Production better than cloud forecast suggests"
            root_cause = "clearer_than_forecast"
            possible_causes = []
        else:
            shadow_type = "normal"
            confidence = 0.75
            interpretation = "Normal variation from cloud-adjusted forecast"
            root_cause = "normal_variation"
            possible_causes = []

        efficiency_ratio = actual_kwh / theoretical_max
        shadow_percent = max(0.0, (1.0 - efficiency_ratio) * 100.0)

        return {
            "method": "sensor_fusion",
            "mode": "cloud_based",
            "shadow_type": shadow_type,
            "shadow_percent": round(shadow_percent, 1),
            "confidence": confidence,
            "interpretation": interpretation,
            "root_cause": root_cause,
            "possible_causes": possible_causes if 'possible_causes' in locals() else [],
            "cloud_analysis": {
                "cloud_cover_percent": cloud_cover,
                "expected_kwh": round(expected_kwh, 3),
                "actual_kwh": actual_kwh,
                "deviation_percent": round(deviation_percent, 1)
            }
        }

    async def detect_shadow_ensemble(
        self,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ensemble Detection - Combines both methods for best results

        Weighted combination:
        - Theory Ratio: 60% (reliable baseline)
        - Sensor Fusion: 40% (adds weather/lux context)

        Args:
            prediction: Hourly prediction dict

        Returns:
            Combined shadow detection result with all method outputs
        """
        try:

            theory_result = await self.detect_shadow_theory_ratio(prediction)
            fusion_result = await self.detect_shadow_sensor_fusion(prediction)

            if theory_result.get("shadow_type") == "error":
                _LOGGER.warning("Theory ratio failed, using sensor fusion only")
                return {
                    **fusion_result,
                    "ensemble_mode": "fusion_only",
                    "methods": {
                        "theory_ratio": theory_result,
                        "sensor_fusion": fusion_result
                    }
                }

            if fusion_result.get("shadow_type") == "error":
                _LOGGER.warning("Sensor fusion failed, using theory ratio only")
                return {
                    **theory_result,
                    "ensemble_mode": "theory_only",
                    "methods": {
                        "theory_ratio": theory_result,
                        "sensor_fusion": fusion_result
                    }
                }

            if theory_result.get("shadow_type") == "night":
                return {
                    **theory_result,
                    "ensemble_mode": "night",
                    "root_cause": "night",
                    "methods": {
                        "theory_ratio": theory_result,
                        "sensor_fusion": fusion_result
                    }
                }

            theory_shadow = theory_result.get("shadow_percent", 0)
            fusion_shadow = fusion_result.get("shadow_percent", 0)

            ensemble_shadow = (
                theory_shadow * self.WEIGHT_THEORY +
                fusion_shadow * self.WEIGHT_SENSOR
            )

            theory_conf = theory_result.get("confidence", 0)
            fusion_conf = fusion_result.get("confidence", 0)

            ensemble_confidence = (
                theory_conf * self.WEIGHT_THEORY +
                fusion_conf * self.WEIGHT_SENSOR
            )

            if ensemble_shadow < self.SHADOW_THRESHOLD_LIGHT * 100:
                ensemble_type = "none"
            elif ensemble_shadow < self.SHADOW_THRESHOLD_MODERATE * 100:
                ensemble_type = "light"
            elif ensemble_shadow < self.SHADOW_THRESHOLD_HEAVY * 100:
                ensemble_type = "moderate"
            else:
                ensemble_type = "heavy"

            root_cause = fusion_result.get("root_cause")
            if not root_cause or root_cause == "unknown":

                possible_causes = theory_result.get("possible_causes", [])
                if possible_causes and ensemble_type in ["moderate", "heavy"]:

                    root_cause = "possible_obstruction"
                elif ensemble_type == "none":
                    root_cause = "normal_variation"
                else:
                    root_cause = "unknown"

            fusion_mode = fusion_result.get("mode", "unknown")

            return {
                "method": "ensemble",
                "ensemble_mode": "full",
                "shadow_type": ensemble_type,
                "shadow_percent": round(ensemble_shadow, 1),
                "confidence": round(ensemble_confidence, 2),
                "root_cause": root_cause,
                "fusion_mode": fusion_mode,
                "efficiency_ratio": theory_result.get("efficiency_ratio", 0),
                "loss_kwh": theory_result.get("loss_kwh", 0),
                "theoretical_max_kwh": theory_result.get("theoretical_max_kwh", 0),
                "interpretation": self._generate_ensemble_interpretation(
                    ensemble_type, ensemble_shadow, root_cause, fusion_result
                ),
                "methods": {
                    "theory_ratio": theory_result,
                    "sensor_fusion": fusion_result
                },
                "weights": {
                    "theory": self.WEIGHT_THEORY,
                    "sensor_fusion": self.WEIGHT_SENSOR
                }
            }

        except Exception as e:
            _LOGGER.error(f"Ensemble shadow detection failed: {e}", exc_info=True)
            return {
                "method": "ensemble",
                "error": str(e),
                "shadow_type": "error",
                "confidence": 0.0
            }

    def _generate_ensemble_interpretation(
        self,
        shadow_type: str,
        shadow_percent: float,
        root_cause: str,
        fusion_result: Dict[str, Any]
    ) -> str:
        """Generate human-readable interpretation of ensemble result"""

        if shadow_type == "none":
            base = f"Excellent production ({100 - shadow_percent:.0f}% of theoretical max)"
        elif shadow_type == "light":
            base = f"Light shadowing ({shadow_percent:.0f}% loss)"
        elif shadow_type == "moderate":
            base = f"Moderate shadowing ({shadow_percent:.0f}% loss)"
        else:
            base = f"Heavy shadowing ({shadow_percent:.0f}% loss)"

        if root_cause == "weather_clouds":
            context = "primarily due to cloud cover"
        elif root_cause == "building_tree_obstruction":
            context = "likely due to local obstruction (building/tree)"
        elif root_cause == "normal_variation":
            context = "within normal operational range"
        else:
            context = "cause unclear, monitor system"

        return f"{base} - {context}"

class PerformanceLossAnalyzer:
    """
    Analyzes performance losses due to shading over time

    Provides daily summaries, trends, and pattern detection.
    """

    def __init__(self, data_manager=None):
        """Initialize performance loss analyzer @zara"""
        self.data_manager = data_manager
        _LOGGER.info("Performance Loss Analyzer initialized")

    async def analyze_daily_shadow(
        self,
        date: str,
        hourly_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze shadow performance for a specific day

        Args:
            date: Date string (YYYY-MM-DD)
            hourly_predictions: List of hourly prediction dicts with shadow_detection

        Returns:
            Daily shadow analysis summary
        """
        try:

            day_predictions = [
                p for p in hourly_predictions
                if p.get("target_date") == date
                and p.get("shadow_detection") is not None
                and p.get("shadow_detection", {}).get("shadow_type") not in ["night", "error"]
            ]

            if not day_predictions:
                return {
                    "date": date,
                    "error": "No shadow detection data available",
                    "total_hours_analyzed": 0
                }

            shadow_hours = []
            total_loss_kwh = 0.0
            total_theoretical_kwh = 0.0

            for pred in day_predictions:
                shadow_det = pred.get("shadow_detection", {})
                shadow_percent = shadow_det.get("shadow_percent", 0)
                loss_kwh = shadow_det.get("loss_kwh", 0)

                if shadow_percent > 15:
                    shadow_hours.append({
                        "hour": pred.get("target_hour"),
                        "shadow_percent": shadow_percent,
                        "shadow_type": shadow_det.get("shadow_type"),
                        "root_cause": shadow_det.get("root_cause")
                    })

                total_loss_kwh += loss_kwh
                total_theoretical_kwh += shadow_det.get("theoretical_max_kwh", 0)

            peak_shadow_hour = None
            peak_shadow_percent = 0.0

            for pred in day_predictions:
                shadow_det = pred.get("shadow_detection", {})
                shadow_percent = shadow_det.get("shadow_percent", 0)

                if shadow_percent > peak_shadow_percent:
                    peak_shadow_percent = shadow_percent
                    peak_shadow_hour = pred.get("target_hour")

            if total_theoretical_kwh > 0:
                daily_loss_percent = (total_loss_kwh / total_theoretical_kwh) * 100.0
            else:
                daily_loss_percent = 0.0

            root_causes = {}
            for pred in day_predictions:
                cause = pred.get("shadow_detection", {}).get("root_cause", "unknown")
                root_causes[cause] = root_causes.get(cause, 0) + 1

            dominant_cause = max(root_causes, key=root_causes.get) if root_causes else "unknown"

            return {
                "date": date,
                "total_hours_analyzed": len(day_predictions),
                "shadow_hours_count": len(shadow_hours),
                "shadow_hours": [h["hour"] for h in shadow_hours],
                "peak_shadow_hour": peak_shadow_hour,
                "peak_shadow_percent": round(peak_shadow_percent, 1),
                "cumulative_loss_kwh": round(total_loss_kwh, 3),
                "cumulative_theoretical_kwh": round(total_theoretical_kwh, 3),
                "daily_loss_percent": round(daily_loss_percent, 1),
                "root_causes": root_causes,
                "dominant_cause": dominant_cause,
                "interpretation": self._interpret_daily_shadow(
                    len(shadow_hours), daily_loss_percent, dominant_cause
                )
            }

        except Exception as e:
            _LOGGER.error(f"Daily shadow analysis failed: {e}", exc_info=True)
            return {
                "date": date,
                "error": str(e),
                "total_hours_analyzed": 0
            }

    def _interpret_daily_shadow(
        self,
        shadow_hours: int,
        daily_loss_percent: float,
        dominant_cause: str
    ) -> str:
        """Generate interpretation of daily shadow analysis"""

        if shadow_hours == 0:
            return "Excellent day - no significant shadowing detected"

        if daily_loss_percent < 10:
            severity = "minimal"
        elif daily_loss_percent < 25:
            severity = "moderate"
        else:
            severity = "significant"

        cause_text = {
            "weather_clouds": "due to cloud cover",
            "building_tree_obstruction": "due to building/tree obstruction",
            "normal_variation": "within normal range",
            "unknown": "cause unclear"
        }.get(dominant_cause, "")

        return f"{severity.capitalize()} shadowing detected ({shadow_hours}h affected, {daily_loss_percent:.0f}% loss) {cause_text}"

_shadow_detector = None
_performance_analyzer = None

def get_shadow_detector(
    data_manager=None,
    panel_groups: Optional[List[Dict[str, Any]]] = None,
) -> ShadowDetector:
    """Get or create shadow detector instance @zara"""
    global _shadow_detector
    if _shadow_detector is None:
        _shadow_detector = ShadowDetector(
            data_manager=data_manager,
            panel_groups=panel_groups,
        )
    elif panel_groups and not _shadow_detector._use_panel_groups:
        _shadow_detector._panel_groups = panel_groups
        _shadow_detector._use_panel_groups = True
    return _shadow_detector

def get_performance_analyzer(data_manager=None) -> PerformanceLossAnalyzer:
    """Get or create performance analyzer instance @zara"""
    global _performance_analyzer
    if _performance_analyzer is None:
        _performance_analyzer = PerformanceLossAnalyzer(data_manager)
    return _performance_analyzer
