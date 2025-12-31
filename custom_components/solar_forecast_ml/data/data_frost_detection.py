# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
import math
from typing import Dict, Optional

_LOGGER = logging.getLogger(__name__)

class FrostDetector:
    """
    Detects frost/ice on solar panels based on meteorological conditions.

    V12.0.0 Enhanced Algorithm:
    1. Dewpoint calculation (Magnus formula) for physical frost probability
    2. Frost margin analysis (temperature - dewpoint)
    3. Correlation-based detection: Compare actual radiation drop vs expected cloud drop
    4. Dynamic thresholds based on frost conditions and wind
    5. Combined scoring (0-10) with physics-based validation

    Detection Logic:
    - Frost forms when surface temperature < dewpoint AND dewpoint < 0°C
    - Correlation detection: If radiation drops MORE than clouds explain → FROST!
    - Wind factor: Low wind enables frost formation

    Score >= 8: Heavy frost (exclude from ML training)
    Score 5-7: Light frost (monitor)
    Score < 5: No frost
    """

    def __init__(self):
        """Initialize frost detector @zara"""
        self.frost_score_threshold_heavy = 8
        self.frost_score_threshold_light = 5

        self.CORRELATION_THRESHOLD_BASE = 0.20

    def detect_frost(
        self,
        temperature_c: Optional[float],
        humidity_percent: Optional[float],
        wind_speed_ms: Optional[float],
        solar_radiation_wm2: Optional[float],
        theoretical_max_wm2: Optional[float],
        cloud_cover_percent: Optional[float] = None,
    ) -> Dict:
        """
        Detect frost on solar panels using enhanced correlation-based algorithm

        V12.0.0 Enhanced Algorithm:
        1. Calculate dewpoint from temperature and humidity (Magnus formula)
        2. Determine frost probability from frost margin (temp - dewpoint)
        3. Correlate actual radiation drop with expected cloud drop
        4. Apply dynamic thresholds based on conditions
        5. Combined scoring with physics-based validation

        Args:
            temperature_c: Air temperature in Celsius
            humidity_percent: Relative humidity (0-100)
            wind_speed_ms: Wind speed in m/s
            solar_radiation_wm2: Measured solar radiation (W/m²)
            theoretical_max_wm2: Theoretical max radiation from astronomy (W/m²)
            cloud_cover_percent: Cloud coverage 0-100% (optional, from weather)

        Returns:
            Dictionary with:
                - frost_detected: None | "light_frost" | "heavy_frost"
                - frost_score: Score 0-10
                - confidence: 0.0-1.0
                - actual_vs_expected_pct: Radiation comparison
                - indicators: Dict with individual metrics
                - reason: Human-readable explanation
                - frost_analysis: Dict with correlation analysis (V12.0.0)
        """

        result = {
            "frost_detected": None,
            "frost_score": 0,
            "confidence": 0.0,
            "actual_vs_expected_pct": None,
            "indicators": {},
            "reason": None,
            "frost_analysis": {
                "dewpoint_c": None,
                "frost_margin_c": None,
                "frost_probability": 0.0,
                "correlation_diff_percent": None,
                "threshold_used_percent": None,
                "detection_method": "correlation_enhanced",
                "wind_frost_factor": None,
                "physical_frost_possible": False
            }
        }

        if any(val is None for val in [temperature_c, humidity_percent, wind_speed_ms,
                                       solar_radiation_wm2, theoretical_max_wm2]):
            result["reason"] = "Missing sensor data"
            result["frost_analysis"]["detection_method"] = "skipped_missing_data"
            return result

        if theoretical_max_wm2 < 5:
            result["reason"] = "Nighttime (no solar radiation expected)"
            result["frost_analysis"]["detection_method"] = "skipped_nighttime"
            return result

        dewpoint_c = self._calculate_dewpoint(temperature_c, humidity_percent)
        result["frost_analysis"]["dewpoint_c"] = round(dewpoint_c, 2)

        frost_margin = temperature_c - dewpoint_c
        result["frost_analysis"]["frost_margin_c"] = round(frost_margin, 2)

        physical_frost_possible = dewpoint_c < 0 and frost_margin < 3.0
        result["frost_analysis"]["physical_frost_possible"] = physical_frost_possible

        wind_frost_factor = self._calculate_wind_frost_factor(wind_speed_ms)
        result["frost_analysis"]["wind_frost_factor"] = round(wind_frost_factor, 2)

        frost_probability = self._calculate_frost_probability(
            temperature_c, dewpoint_c, frost_margin, wind_frost_factor
        )
        result["frost_analysis"]["frost_probability"] = round(frost_probability, 3)

        if temperature_c > 3 or (temperature_c > 1 and dewpoint_c > 0):
            result["reason"] = f"No frost conditions (temp={temperature_c:.1f}°C, dewpoint={dewpoint_c:.1f}°C)"
            result["frost_analysis"]["detection_method"] = "ruled_out_physics"
            return result

        expected_radiation = self._calculate_expected_radiation(
            theoretical_max_wm2, cloud_cover_percent
        )

        if expected_radiation < 3:
            result["reason"] = f"Expected radiation too low ({expected_radiation:.1f} W/m²)"
            result["frost_analysis"]["detection_method"] = "skipped_low_radiation"
            return result

        cloud_drop = 1.0 - (expected_radiation / theoretical_max_wm2) if theoretical_max_wm2 > 0 else 0
        actual_drop = 1.0 - (solar_radiation_wm2 / theoretical_max_wm2) if theoretical_max_wm2 > 0 else 0

        cloud_drop = max(0.0, min(1.0, cloud_drop))
        actual_drop = max(0.0, min(1.0, actual_drop))

        correlation_diff = actual_drop - cloud_drop
        result["frost_analysis"]["correlation_diff_percent"] = round(correlation_diff * 100, 1)

        actual_vs_expected = solar_radiation_wm2 / expected_radiation if expected_radiation > 0 else 1.0
        result["actual_vs_expected_pct"] = round(actual_vs_expected * 100, 1)

        threshold = self._calculate_dynamic_threshold(
            frost_probability, wind_frost_factor, cloud_cover_percent
        )
        result["frost_analysis"]["threshold_used_percent"] = round(threshold * 100, 1)

        frost_score = 0

        temp_score = 0
        if temperature_c < -5:
            temp_score = 3
        elif temperature_c < -2:
            temp_score = 2
        elif temperature_c < 0:
            temp_score = 1
        frost_score += temp_score

        humidity_score = 0
        if frost_margin < 1.0 and dewpoint_c < 0:
            humidity_score = 2
        elif frost_margin < 2.0 and dewpoint_c < 0:
            humidity_score = 2
        elif humidity_percent > 85:
            humidity_score = 2
        elif frost_margin < 3.0 and dewpoint_c < 0:
            humidity_score = 1
        elif humidity_percent > 75:
            humidity_score = 1
        frost_score += humidity_score

        wind_score = 0
        if wind_speed_ms < 1:
            wind_score = 2
        elif wind_speed_ms < 3:
            wind_score = 1
        frost_score += wind_score

        radiation_score = 0
        if correlation_diff > threshold and physical_frost_possible:

            if correlation_diff > 0.40:
                radiation_score = 3
            elif correlation_diff > 0.25:
                radiation_score = 2
            elif correlation_diff > threshold:
                radiation_score = 1
        else:

            if actual_vs_expected < 0.4:
                radiation_score = 3
            elif actual_vs_expected < 0.6:
                radiation_score = 2
            elif actual_vs_expected < 0.8:
                radiation_score = 1

        frost_score += radiation_score

        probability_bonus = 0
        if frost_probability > 0.8:
            probability_bonus = 2
        elif frost_probability > 0.5:
            probability_bonus = 1
        frost_score += probability_bonus

        frost_score = min(10, frost_score)

        result["frost_score"] = frost_score
        result["indicators"] = {
            "temperature_c": temperature_c,
            "humidity_percent": humidity_percent,
            "wind_speed_ms": wind_speed_ms,
            "radiation_deficit_pct": round((1 - actual_vs_expected) * 100, 1),
            "temp_score": temp_score,
            "humidity_score": humidity_score,
            "wind_score": wind_score,
            "radiation_score": radiation_score,
            "probability_bonus": probability_bonus,
            "dewpoint_c": round(dewpoint_c, 1),
            "frost_margin_c": round(frost_margin, 1)
        }

        if frost_score >= self.frost_score_threshold_heavy:
            result["frost_detected"] = "heavy_frost"
            result["confidence"] = min(1.0, frost_score / 10)

            if correlation_diff > threshold and physical_frost_possible:
                cause = f"correlation-based (unexplained drop {correlation_diff*100:.0f}%)"
            else:
                cause = f"radiation deficit {(1-actual_vs_expected)*100:.0f}%"

            result["reason"] = (
                f"Heavy frost detected (score {frost_score}/10, {cause}): "
                f"temp={temperature_c:.1f}°C, dewpoint={dewpoint_c:.1f}°C, "
                f"frost_margin={frost_margin:.1f}°C, wind={wind_speed_ms:.1f}m/s"
            )

        elif frost_score >= self.frost_score_threshold_light:
            result["frost_detected"] = "light_frost"
            result["confidence"] = frost_score / 10
            result["reason"] = (
                f"Light frost detected (score {frost_score}/10): "
                f"temp={temperature_c:.1f}°C, dewpoint={dewpoint_c:.1f}°C, "
                f"frost_margin={frost_margin:.1f}°C"
            )

        else:
            if physical_frost_possible:
                result["reason"] = (
                    f"Frost conditions present but no significant impact (score {frost_score}/10): "
                    f"dewpoint={dewpoint_c:.1f}°C, margin={frost_margin:.1f}°C"
                )
            else:
                result["reason"] = f"No frost detected (score {frost_score}/10)"

        return result

    def _calculate_dewpoint(self, temperature_c: float, humidity_percent: float) -> float:
        """Calculate dewpoint temperature using Magnus formula @zara"""

        a = 17.27
        b = 237.7

        humidity_percent = max(1.0, min(100.0, humidity_percent))

        try:
            alpha = (a * temperature_c) / (b + temperature_c) + math.log(humidity_percent / 100.0)
            dewpoint = (b * alpha) / (a - alpha)
            return dewpoint
        except (ValueError, ZeroDivisionError):

            return temperature_c - ((100 - humidity_percent) / 5.0)

    def _calculate_wind_frost_factor(self, wind_speed_ms: float) -> float:
        """Calculate wind influence on frost formation @zara"""
        if wind_speed_ms < 0.5:
            return 1.0
        elif wind_speed_ms < 1.5:
            return 0.85
        elif wind_speed_ms < 2.5:
            return 0.65
        elif wind_speed_ms < 4.0:
            return 0.40
        elif wind_speed_ms < 6.0:
            return 0.20
        else:
            return 0.05

    def _calculate_frost_probability(
        self,
        temperature_c: float,
        dewpoint_c: float,
        frost_margin: float,
        wind_factor: float
    ) -> float:
        """
        Calculate physical frost probability based on meteorological conditions

        Frost formation requires:
        1. Dewpoint below 0°C (sublimation instead of condensation)
        2. Surface temperature approaching dewpoint
        3. Low wind (allows radiative cooling)

        Args:
            temperature_c: Air temperature in Celsius
            dewpoint_c: Calculated dewpoint in Celsius
            frost_margin: Temperature - Dewpoint difference
            wind_factor: Wind influence factor (0-1)

        Returns:
            Probability 0.0-1.0
        """

        if dewpoint_c > 0:
            return 0.0

        if temperature_c > 3:
            return 0.0

        if frost_margin < 0:

            margin_probability = 0.95
        elif frost_margin < 1.0:
            margin_probability = 0.90
        elif frost_margin < 2.0:
            margin_probability = 0.75
        elif frost_margin < 3.0:
            margin_probability = 0.50
        elif frost_margin < 5.0:
            margin_probability = 0.25
        else:
            margin_probability = 0.05

        if temperature_c < -5:
            temp_factor = 1.0
        elif temperature_c < -2:
            temp_factor = 0.9
        elif temperature_c < 0:
            temp_factor = 0.8
        elif temperature_c < 2:
            temp_factor = 0.5
        else:
            temp_factor = 0.2

        probability = margin_probability * temp_factor * wind_factor

        return min(1.0, max(0.0, probability))

    def _calculate_dynamic_threshold(
        self,
        frost_probability: float,
        wind_factor: float,
        cloud_cover_percent: Optional[float]
    ) -> float:
        """
        Calculate dynamic correlation threshold based on conditions

        Lower threshold = more sensitive to frost detection
        Higher threshold = requires more evidence

        Args:
            frost_probability: Physical frost probability (0-1)
            wind_factor: Wind influence factor (0-1)
            cloud_cover_percent: Cloud coverage (0-100) or None

        Returns:
            Threshold for correlation difference (0.05-0.35)
        """
        threshold = self.CORRELATION_THRESHOLD_BASE

        if frost_probability > 0.7:
            threshold -= 0.08
        elif frost_probability > 0.4:
            threshold -= 0.04

        if wind_factor > 0.8:
            threshold -= 0.04

        if cloud_cover_percent is not None:
            if cloud_cover_percent > 85:
                threshold += 0.08
            elif cloud_cover_percent > 70:
                threshold += 0.04

        return max(0.05, min(0.35, threshold))

    def _calculate_expected_radiation(
        self,
        theoretical_max_wm2: float,
        cloud_cover_percent: Optional[float]
    ) -> float:
        """
        Calculate expected radiation based on cloud cover

        Uses exponential cloud transmission model from WeatherCalculator
        """
        if cloud_cover_percent is None:

            cloud_cover_percent = 50.0

        cloud_factor = math.exp(-0.008 * cloud_cover_percent)

        expected_radiation = theoretical_max_wm2 * cloud_factor

        return expected_radiation

    def should_exclude_from_training(self, frost_result: Dict) -> bool:
        """Determine if hour should be excluded from ML training @zara"""
        return frost_result.get("frost_detected") == "heavy_frost"
