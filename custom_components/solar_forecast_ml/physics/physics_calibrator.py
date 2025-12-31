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
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

# Calibration constants
MIN_PRODUCTION_KWH = 0.01  # Minimum production to consider for calibration (standard hours)
MIN_PRODUCTION_KWH_MORNING = 0.003  # Lower threshold for morning hours (7-10) to enable learning
MAX_CORRECTION_FACTOR = 10.0  # Maximum allowed correction factor (allows up to 10x correction)
MIN_CORRECTION_FACTOR = 0.1  # Minimum allowed correction factor (allows down to 0.1x)
ROLLING_WINDOW_DAYS = 21  # Days for rolling average (21 days = stable, stays within similar sun positions)
MIN_SAMPLES_FOR_FACTOR = 1  # Minimum days needed to calculate a factor (start learning immediately)
MIN_SAMPLES_FOR_FULL_CONFIDENCE = 7  # Days needed for full confidence (increased for 21-day window)
OUTLIER_THRESHOLD = 3.0  # Standard deviations for outlier detection

# Weather bucket thresholds (cloud cover percentages)
BUCKET_CLEAR_MAX = 25      # 0-25% clouds = CLEAR
BUCKET_FAIR_MAX = 50       # 25-50% clouds = FAIR
BUCKET_CLOUDY_MAX = 75     # 50-75% clouds = CLOUDY
                           # 75-100% clouds = OVERCAST

# Precipitation threshold for RAINY bucket
BUCKET_RAIN_THRESHOLD = 1.0  # mm/h for RAINY bucket

# Low sun angle thresholds (for special MORNING buckets only)
LOW_SUN_MIN_ELEVATION = 5.0   # Below this is twilight (handled separately)
LOW_SUN_MAX_ELEVATION = 15.0  # Above this, normal buckets apply
LOW_SUN_MORNING_HOURS = range(5, 12)  # Only apply LOW_SUN correction for morning hours (5-11)
# At low sun angles with clear/fair sky, weather APIs severely underestimate radiation
# This effect is primarily observed in the MORNING, not evening (evening has more atmospheric haze)


class WeatherBucket(Enum):
    """Weather condition buckets for calibration.

    Includes special LOW_SUN variants for morning/evening hours where
    weather APIs systematically underestimate radiation at clear/fair skies.

    Also includes precipitation buckets for rain and snow conditions.
    """
    CLEAR = "clear"                  # 0-25% clouds - direct radiation dominates
    CLEAR_LOW_SUN = "clear_low_sun"  # 0-25% clouds + sun 5-15° - APIs underestimate
    FAIR = "fair"                    # 25-50% clouds - mixed conditions
    FAIR_LOW_SUN = "fair_low_sun"    # 25-50% clouds + sun 5-15° - APIs underestimate
    CLOUDY = "cloudy"                # 50-75% clouds - mostly diffuse
    CLOUDY_LOW_SUN = "cloudy_low_sun"  # 50-75% clouds + sun 5-15°
    OVERCAST = "overcast"            # 75-100% clouds - fully diffuse
    # Precipitation buckets - higher priority than cloud-based buckets
    RAINY = "rainy"                  # Precipitation > 1mm/h - significantly reduced output
    RAINY_LOW_SUN = "rainy_low_sun"  # Rainy + sun 5-15° - very low output
    SNOWY = "snowy"                  # Snow-covered panels - near-zero output


@dataclass
class BucketFactors:
    """Calibration factors for a specific weather bucket."""
    global_factor: float = 1.0
    hourly_factors: Dict[int, float] = field(default_factory=dict)
    sample_count: int = 0
    confidence: float = 0.0


@dataclass
class HourlyCalibration:
    """Calibration data for a specific hour."""

    hour: int
    physics_kwh: float
    actual_kwh: float
    ratio: float  # actual / physics
    date: str
    group_name: Optional[str] = None


@dataclass
class GroupCalibrationFactors:
    """Calibration factors for a panel group."""

    group_name: str
    global_factor: float = 1.0  # Overall correction factor (fallback)
    hourly_factors: Dict[int, float] = field(default_factory=dict)  # Per-hour factors (fallback)
    bucket_factors: Dict[str, BucketFactors] = field(default_factory=dict)  # Weather-specific factors
    sample_count: int = 0
    confidence: float = 0.0  # 0-1 confidence based on sample count and consistency
    last_updated: Optional[str] = None


@dataclass
class CalibrationResult:
    """Result of calibration calculation."""

    success: bool
    message: str
    groups_calibrated: int = 0
    total_samples: int = 0
    avg_correction_factor: float = 1.0


class PhysicsCalibrator:
    """Self-learning calibrator for Physics Engine.

    Compares Physics predictions with actual production and learns
    correction factors to improve future predictions.
    """

    def __init__(self, data_dir: Path):
        """Initialize the calibrator.

        Args:
            data_dir: Path to solar_forecast_ml data directory
        """
        self.data_dir = data_dir
        self.physics_dir = data_dir / "physics"
        self.stats_dir = data_dir / "stats"
        self.config_file = self.physics_dir / "learning_config.json"
        self.history_file = self.physics_dir / "calibration_history.json"

        # Ensure physics directory exists
        self.physics_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache of calibration factors
        self._factors: Dict[str, GroupCalibrationFactors] = {}
        self._history: List[Dict[str, Any]] = []

        _LOGGER.info("PhysicsCalibrator initialized")

    async def async_init(self) -> bool:
        """Async initialization - load existing calibration data."""
        try:
            await self._load_config()
            await self._load_history()

            if self._factors:
                groups = list(self._factors.keys())
                _LOGGER.info(
                    f"PhysicsCalibrator loaded calibration for {len(groups)} groups: {groups}"
                )
            else:
                _LOGGER.info("PhysicsCalibrator: No existing calibration data found")

            return True
        except Exception as e:
            _LOGGER.error(f"PhysicsCalibrator initialization failed: {e}")
            return False

    async def _load_config(self) -> None:
        """Load calibration config from learning_config.json."""
        import asyncio

        if not self.config_file.exists():
            self._factors = {}
            return

        def _read():
            with open(self.config_file, "r") as f:
                return json.load(f)

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, _read)

            # Parse group factors
            groups_data = data.get("group_calibration", {})
            for group_name, group_data in groups_data.items():
                # Parse bucket factors if present
                bucket_factors_data = group_data.get("bucket_factors", {})
                bucket_factors: Dict[str, BucketFactors] = {}
                for bucket_name, bf_data in bucket_factors_data.items():
                    bucket_factors[bucket_name] = BucketFactors(
                        global_factor=bf_data.get("global_factor", 1.0),
                        hourly_factors={
                            int(h): f for h, f in bf_data.get("hourly_factors", {}).items()
                        },
                        sample_count=bf_data.get("sample_count", 0),
                        confidence=bf_data.get("confidence", 0.0),
                    )

                self._factors[group_name] = GroupCalibrationFactors(
                    group_name=group_name,
                    global_factor=group_data.get("global_factor", 1.0),
                    hourly_factors={
                        int(h): f for h, f in group_data.get("hourly_factors", {}).items()
                    },
                    bucket_factors=bucket_factors,
                    sample_count=group_data.get("sample_count", 0),
                    confidence=group_data.get("confidence", 0.0),
                    last_updated=group_data.get("last_updated"),
                )

            # Log bucket info
            bucket_count = sum(len(f.bucket_factors) for f in self._factors.values())
            _LOGGER.debug(f"Loaded calibration config: {len(self._factors)} groups, {bucket_count} bucket factors")

        except Exception as e:
            _LOGGER.warning(f"Could not load calibration config: {e}")
            self._factors = {}

    async def _load_history(self) -> None:
        """Load calibration history."""
        import asyncio

        if not self.history_file.exists():
            self._history = []
            return

        def _read():
            with open(self.history_file, "r") as f:
                return json.load(f)

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, _read)
            self._history = data.get("history", [])

            # Keep only last ROLLING_WINDOW_DAYS
            cutoff = (datetime.now() - timedelta(days=ROLLING_WINDOW_DAYS)).isoformat()
            self._history = [h for h in self._history if h.get("date", "") >= cutoff[:10]]

        except Exception as e:
            _LOGGER.warning(f"Could not load calibration history: {e}")
            self._history = []

    async def _save_config(self) -> None:
        """Save calibration config to learning_config.json."""
        import asyncio

        # Build config structure
        group_calibration = {}
        total_bucket_factors = 0
        for group_name, factors in self._factors.items():
            # Serialize bucket factors
            bucket_factors_data = {}
            for bucket_name, bf in factors.bucket_factors.items():
                bucket_factors_data[bucket_name] = {
                    "global_factor": round(bf.global_factor, 4),
                    "hourly_factors": {
                        str(h): round(f, 4) for h, f in bf.hourly_factors.items()
                    },
                    "sample_count": bf.sample_count,
                    "confidence": round(bf.confidence, 3),
                }
                total_bucket_factors += 1

            group_calibration[group_name] = {
                "global_factor": round(factors.global_factor, 4),
                "hourly_factors": {
                    str(h): round(f, 4) for h, f in factors.hourly_factors.items()
                },
                "bucket_factors": bucket_factors_data,
                "sample_count": factors.sample_count,
                "confidence": round(factors.confidence, 3),
                "last_updated": factors.last_updated,
            }

        # Calculate overall system efficiency adjustment
        avg_global = 1.0
        if self._factors:
            avg_global = sum(f.global_factor for f in self._factors.values()) / len(self._factors)

        config = {
            "version": "2.0",  # Bumped version for bucket support
            "updated_at": datetime.now().isoformat(),
            "physics_defaults": {
                "albedo": 0.2,
                "system_efficiency": 0.90,
                "learned_efficiency_factor": round(avg_global, 4),
            },
            "group_calibration": group_calibration,
            "metadata": {
                "rolling_window_days": ROLLING_WINDOW_DAYS,
                "min_samples": MIN_SAMPLES_FOR_FACTOR,
                "total_groups": len(self._factors),
                "total_bucket_factors": total_bucket_factors,
                "weather_buckets": [b.value for b in WeatherBucket],
            },
        }

        def _write():
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _write)
            _LOGGER.debug(f"Saved calibration config with {total_bucket_factors} bucket factors")
        except Exception as e:
            _LOGGER.error(f"Failed to save calibration config: {e}")

    async def _save_history(self) -> None:
        """Save calibration history."""
        import asyncio

        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "history": self._history,
        }

        def _write():
            with open(self.history_file, "w") as f:
                json.dump(data, f, indent=2)

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _write)
        except Exception as e:
            _LOGGER.error(f"Failed to save calibration history: {e}")

    def _classify_weather(
        self,
        cloud_cover: float,
        sun_elevation: Optional[float] = None,
        hour: Optional[int] = None,
        precipitation_mm: Optional[float] = None,
        snow_covered_panels: bool = False,
    ) -> WeatherBucket:
        """Classify weather conditions into weather bucket.

        Priority order:
        1. SNOWY - Snow-covered panels (highest priority, near-zero output)
        2. RAINY - Active precipitation > 1mm/h
        3. Cloud-based buckets with LOW_SUN variants for morning hours

        At low sun angles (5-15°) with clear or fair skies IN THE MORNING,
        weather APIs systematically underestimate radiation. These conditions
        get special LOW_SUN buckets for separate calibration factors.

        The LOW_SUN effect is primarily a MORNING phenomenon:
        - Morning: Clear atmosphere, less haze → APIs underestimate
        - Evening: More atmospheric particles/haze → APIs are more accurate

        Args:
            cloud_cover: Cloud cover percentage (0-100)
            sun_elevation: Sun elevation in degrees (optional, enables LOW_SUN detection)
            hour: Hour of day (0-23, optional, required for LOW_SUN morning check)
            precipitation_mm: Precipitation in mm/h (optional, enables RAINY detection)
            snow_covered_panels: Whether panels are snow-covered (from weather tracker)

        Returns:
            WeatherBucket enum value
        """
        # Priority 1: Snow-covered panels - highest priority
        if snow_covered_panels:
            return WeatherBucket.SNOWY

        # Check for low sun angle conditions (5-15° elevation) IN THE MORNING ONLY
        is_morning_low_sun = (
            sun_elevation is not None
            and LOW_SUN_MIN_ELEVATION < sun_elevation <= LOW_SUN_MAX_ELEVATION
            and hour is not None
            and hour in LOW_SUN_MORNING_HOURS
        )

        # Priority 2: Active precipitation (rain/snow falling)
        if precipitation_mm is not None and precipitation_mm > BUCKET_RAIN_THRESHOLD:
            if is_morning_low_sun:
                return WeatherBucket.RAINY_LOW_SUN
            return WeatherBucket.RAINY

        # Priority 3: Cloud-based classification
        if cloud_cover <= BUCKET_CLEAR_MAX:
            # Clear sky - check for morning low sun
            if is_morning_low_sun:
                return WeatherBucket.CLEAR_LOW_SUN
            return WeatherBucket.CLEAR
        elif cloud_cover <= BUCKET_FAIR_MAX:
            # Fair sky - check for morning low sun
            if is_morning_low_sun:
                return WeatherBucket.FAIR_LOW_SUN
            return WeatherBucket.FAIR
        elif cloud_cover <= BUCKET_CLOUDY_MAX:
            # Cloudy - check for morning low sun
            if is_morning_low_sun:
                return WeatherBucket.CLOUDY_LOW_SUN
            return WeatherBucket.CLOUDY
        else:
            # Overcast - no low sun variant (fully diffuse)
            return WeatherBucket.OVERCAST

    def get_correction_factor(
        self,
        group_name: str,
        hour: Optional[int] = None,
        cloud_cover: Optional[float] = None,
        sun_elevation: Optional[float] = None,
    ) -> float:
        """Get correction factor for a group, hour, and weather condition.

        Args:
            group_name: Name of the panel group
            hour: Optional hour (0-23) for hourly factor
            cloud_cover: Optional cloud cover (0-100) for weather-specific factor
            sun_elevation: Optional sun elevation (degrees) for low-sun detection

        Returns:
            Correction factor (multiply physics prediction by this)
        """
        if group_name not in self._factors:
            return 1.0

        factors = self._factors[group_name]

        # Priority 1: Weather bucket + hour specific factor (with sun elevation + hour for morning check)
        if cloud_cover is not None:
            bucket = self._classify_weather(cloud_cover, sun_elevation, hour)
            bucket_data = factors.bucket_factors.get(bucket.value)

            if bucket_data:
                # Check for hourly factor within this bucket
                if hour is not None and hour in bucket_data.hourly_factors:
                    return bucket_data.hourly_factors[hour]
                # Use bucket's global factor
                if bucket_data.sample_count >= MIN_SAMPLES_FOR_FACTOR:
                    return bucket_data.global_factor

            # Fallback for LOW_SUN buckets: try base bucket if no LOW_SUN data yet
            if bucket in (WeatherBucket.CLEAR_LOW_SUN, WeatherBucket.FAIR_LOW_SUN):
                base_bucket = WeatherBucket.CLEAR if bucket == WeatherBucket.CLEAR_LOW_SUN else WeatherBucket.FAIR
                base_bucket_data = factors.bucket_factors.get(base_bucket.value)
                if base_bucket_data and base_bucket_data.sample_count >= MIN_SAMPLES_FOR_FACTOR:
                    # Use base bucket factor as starting point (will be learned over time)
                    return base_bucket_data.global_factor

        # Priority 2: Hour-specific factor (fallback, weather-agnostic)
        if hour is not None and hour in factors.hourly_factors:
            return factors.hourly_factors[hour]

        # Fall back to global factor
        return factors.global_factor

    def get_all_factors(self) -> Dict[str, GroupCalibrationFactors]:
        """Get all calibration factors."""
        return self._factors.copy()

    async def calibrate_from_daily_data(
        self,
        date: str,
        group_data: Dict[str, List[Dict[str, Any]]],
    ) -> CalibrationResult:
        """Perform calibration from daily production data.

        Args:
            date: Date string (YYYY-MM-DD)
            group_data: Dict mapping group_name to list of hourly data with:
                - hour: int
                - physics_kwh: float (predicted)
                - actual_kwh: float (measured)
                - cloud_cover: float (0-100, optional but recommended)
                - sun_elevation: float (degrees, optional - enables LOW_SUN bucket learning)
                - precipitation_mm: float (mm/h, optional - enables RAINY bucket learning)
                - snow_covered_panels: bool (optional - enables SNOWY bucket learning)

        Returns:
            CalibrationResult with success status and details
        """
        try:
            total_samples = 0
            daily_ratios: Dict[str, List[float]] = {}
            hourly_ratios: Dict[str, Dict[int, List[float]]] = {}
            # Bucket-specific ratios (including LOW_SUN variants)
            bucket_ratios: Dict[str, Dict[str, List[float]]] = {}  # group -> bucket -> ratios
            bucket_hourly_ratios: Dict[str, Dict[str, Dict[int, List[float]]]] = {}  # group -> bucket -> hour -> ratios

            # Collect ratios for each group
            for group_name, hours in group_data.items():
                daily_ratios[group_name] = []
                hourly_ratios[group_name] = {}
                bucket_ratios[group_name] = {b.value: [] for b in WeatherBucket}
                bucket_hourly_ratios[group_name] = {b.value: {} for b in WeatherBucket}

                for hour_data in hours:
                    hour = hour_data.get("hour", 0)
                    physics = hour_data.get("physics_kwh", 0)
                    actual = hour_data.get("actual_kwh", 0)
                    cloud_cover = hour_data.get("cloud_cover")
                    sun_elevation = hour_data.get("sun_elevation")  # For LOW_SUN detection
                    precipitation_mm = hour_data.get("precipitation_mm")  # For RAINY bucket
                    snow_covered_panels = hour_data.get("snow_covered_panels", False)  # For SNOWY bucket

                    # Use lower threshold for morning hours (7-10) to enable learning
                    # Morning hours often have low physics predictions but real production
                    is_morning = 7 <= hour <= 10
                    min_threshold = MIN_PRODUCTION_KWH_MORNING if is_morning else MIN_PRODUCTION_KWH

                    # Skip if values too small (but use lower threshold for morning)
                    if physics < min_threshold or actual < min_threshold:
                        continue

                    ratio = actual / physics

                    # Skip extreme outliers
                    if ratio < MIN_CORRECTION_FACTOR or ratio > MAX_CORRECTION_FACTOR:
                        _LOGGER.debug(
                            f"Skipping outlier: {group_name} hour {hour}, "
                            f"ratio={ratio:.2f} (physics={physics:.3f}, actual={actual:.3f})"
                        )
                        continue

                    # Add to global ratios (fallback)
                    daily_ratios[group_name].append(ratio)

                    if hour not in hourly_ratios[group_name]:
                        hourly_ratios[group_name][hour] = []
                    hourly_ratios[group_name][hour].append(ratio)

                    # Add to bucket-specific ratios if cloud_cover available
                    # Now includes LOW_SUN bucket detection using sun_elevation + hour (morning only)
                    # Also includes RAINY/SNOWY bucket detection
                    if cloud_cover is not None or snow_covered_panels or precipitation_mm:
                        bucket = self._classify_weather(
                            cloud_cover if cloud_cover is not None else 50.0,  # Default to fair
                            sun_elevation,
                            hour,
                            precipitation_mm,
                            snow_covered_panels,
                        )
                        bucket_ratios[group_name][bucket.value].append(ratio)

                        if hour not in bucket_hourly_ratios[group_name][bucket.value]:
                            bucket_hourly_ratios[group_name][bucket.value][hour] = []
                        bucket_hourly_ratios[group_name][bucket.value][hour].append(ratio)

                        # Log special bucket assignments for debugging
                        if bucket in (WeatherBucket.CLEAR_LOW_SUN, WeatherBucket.FAIR_LOW_SUN, WeatherBucket.CLOUDY_LOW_SUN):
                            _LOGGER.debug(
                                f"Morning low-sun calibration sample: {group_name} hour {hour}, "
                                f"bucket={bucket.value}, ratio={ratio:.2f}, "
                                f"sun_elev={sun_elevation:.1f}°, clouds={cloud_cover:.0f}%"
                            )
                        elif bucket in (WeatherBucket.RAINY, WeatherBucket.RAINY_LOW_SUN):
                            _LOGGER.debug(
                                f"Rainy calibration sample: {group_name} hour {hour}, "
                                f"bucket={bucket.value}, ratio={ratio:.2f}, "
                                f"precipitation={precipitation_mm:.1f}mm/h"
                            )
                        elif bucket == WeatherBucket.SNOWY:
                            _LOGGER.debug(
                                f"Snowy calibration sample: {group_name} hour {hour}, "
                                f"bucket={bucket.value}, ratio={ratio:.2f}, "
                                f"snow_covered_panels=True"
                            )

                    total_samples += 1

            if total_samples == 0:
                return CalibrationResult(
                    success=False,
                    message="No valid samples for calibration",
                )

            # Add to history
            history_entry = {
                "date": date,
                "groups": {},
            }

            for group_name, ratios in daily_ratios.items():
                if ratios:
                    avg_ratio = sum(ratios) / len(ratios)

                    # Build bucket summary for history (including hourly data)
                    bucket_summary = {}
                    for bucket_name, b_ratios in bucket_ratios.get(group_name, {}).items():
                        if b_ratios:
                            # Get bucket-specific hourly ratios
                            b_hourly = bucket_hourly_ratios.get(group_name, {}).get(bucket_name, {})
                            bucket_summary[bucket_name] = {
                                "avg_ratio": round(sum(b_ratios) / len(b_ratios), 4),
                                "sample_count": len(b_ratios),
                                "hourly": {
                                    str(h): round(sum(r) / len(r), 4)
                                    for h, r in b_hourly.items()
                                    if r
                                },
                            }

                    history_entry["groups"][group_name] = {
                        "avg_ratio": round(avg_ratio, 4),
                        "sample_count": len(ratios),
                        "hourly": {
                            str(h): round(sum(r) / len(r), 4)
                            for h, r in hourly_ratios[group_name].items()
                            if r
                        },
                        "buckets": bucket_summary,
                    }

            self._history.append(history_entry)

            # Recalculate factors from history
            await self._recalculate_factors()

            # Save updated data
            await self._save_config()
            await self._save_history()

            # Calculate average factor for result
            avg_factor = 1.0
            if self._factors:
                avg_factor = sum(f.global_factor for f in self._factors.values()) / len(self._factors)

            _LOGGER.info(
                f"✓ Physics calibration complete for {date}: "
                f"{len(group_data)} groups, {total_samples} samples, "
                f"avg correction factor = {avg_factor:.2f}"
            )

            return CalibrationResult(
                success=True,
                message=f"Calibration successful: {total_samples} samples processed",
                groups_calibrated=len(group_data),
                total_samples=total_samples,
                avg_correction_factor=avg_factor,
            )

        except Exception as e:
            _LOGGER.error(f"Calibration failed: {e}", exc_info=True)
            return CalibrationResult(
                success=False,
                message=f"Calibration error: {e}",
            )

    async def _recalculate_factors(self) -> None:
        """Recalculate calibration factors from history including weather buckets."""
        # Collect all ratios per group and hour
        group_ratios: Dict[str, List[float]] = {}
        hourly_ratios: Dict[str, Dict[int, List[float]]] = {}
        # NEW: Bucket-specific collections
        bucket_ratios: Dict[str, Dict[str, List[float]]] = {}  # group -> bucket -> ratios
        bucket_hourly_ratios: Dict[str, Dict[str, Dict[int, List[float]]]] = {}

        for entry in self._history:
            for group_name, group_data in entry.get("groups", {}).items():
                # Initialize structures
                if group_name not in group_ratios:
                    group_ratios[group_name] = []
                    hourly_ratios[group_name] = {}
                    bucket_ratios[group_name] = {b.value: [] for b in WeatherBucket}
                    bucket_hourly_ratios[group_name] = {b.value: {} for b in WeatherBucket}

                # Global ratio
                avg_ratio = group_data.get("avg_ratio")
                if avg_ratio:
                    group_ratios[group_name].append(avg_ratio)

                # Hourly ratios
                for hour_str, ratio in group_data.get("hourly", {}).items():
                    hour = int(hour_str)
                    if hour not in hourly_ratios[group_name]:
                        hourly_ratios[group_name][hour] = []
                    hourly_ratios[group_name][hour].append(ratio)

                # NEW: Bucket ratios from history (including hourly)
                for bucket_name, bucket_data in group_data.get("buckets", {}).items():
                    if bucket_name in bucket_ratios[group_name]:
                        b_avg_ratio = bucket_data.get("avg_ratio")
                        if b_avg_ratio:
                            bucket_ratios[group_name][bucket_name].append(b_avg_ratio)

                        # Bucket-specific hourly ratios
                        for hour_str, ratio in bucket_data.get("hourly", {}).items():
                            hour = int(hour_str)
                            if hour not in bucket_hourly_ratios[group_name][bucket_name]:
                                bucket_hourly_ratios[group_name][bucket_name][hour] = []
                            bucket_hourly_ratios[group_name][bucket_name][hour].append(ratio)

        # Calculate factors
        for group_name, ratios in group_ratios.items():
            if len(ratios) < MIN_SAMPLES_FOR_FACTOR:
                continue

            # Remove outliers using IQR method
            clean_ratios = self._remove_outliers(ratios)
            if not clean_ratios:
                clean_ratios = ratios

            global_factor = sum(clean_ratios) / len(clean_ratios)
            global_factor = max(MIN_CORRECTION_FACTOR, min(MAX_CORRECTION_FACTOR, global_factor))

            # Calculate hourly factors (fallback)
            hourly_factors = {}
            for hour, hour_ratios in hourly_ratios.get(group_name, {}).items():
                if len(hour_ratios) >= MIN_SAMPLES_FOR_FACTOR:
                    clean_hour_ratios = self._remove_outliers(hour_ratios)
                    if clean_hour_ratios:
                        factor = sum(clean_hour_ratios) / len(clean_hour_ratios)
                        factor = max(MIN_CORRECTION_FACTOR, min(MAX_CORRECTION_FACTOR, factor))
                        hourly_factors[hour] = factor

            # NEW: Calculate bucket-specific factors (including hourly)
            calculated_bucket_factors: Dict[str, BucketFactors] = {}
            for bucket in WeatherBucket:
                b_ratios = bucket_ratios.get(group_name, {}).get(bucket.value, [])
                if len(b_ratios) >= MIN_SAMPLES_FOR_FACTOR:
                    clean_b_ratios = self._remove_outliers(b_ratios)
                    if not clean_b_ratios:
                        clean_b_ratios = b_ratios

                    b_factor = sum(clean_b_ratios) / len(clean_b_ratios)
                    b_factor = max(MIN_CORRECTION_FACTOR, min(MAX_CORRECTION_FACTOR, b_factor))

                    # Calculate bucket-specific hourly factors
                    b_hourly_factors: Dict[int, float] = {}
                    b_hourly_data = bucket_hourly_ratios.get(group_name, {}).get(bucket.value, {})
                    for hour, hour_ratios in b_hourly_data.items():
                        if len(hour_ratios) >= MIN_SAMPLES_FOR_FACTOR:
                            clean_hour_ratios = self._remove_outliers(hour_ratios)
                            if clean_hour_ratios:
                                factor = sum(clean_hour_ratios) / len(clean_hour_ratios)
                                factor = max(MIN_CORRECTION_FACTOR, min(MAX_CORRECTION_FACTOR, factor))
                                b_hourly_factors[hour] = factor

                    # Bucket confidence
                    b_std_dev = self._calculate_std_dev(clean_b_ratios) if len(clean_b_ratios) > 1 else 0.5
                    b_consistency = max(0, 1 - b_std_dev)
                    b_sample_confidence = min(1.0, len(b_ratios) / MIN_SAMPLES_FOR_FULL_CONFIDENCE)
                    b_confidence = (b_consistency + b_sample_confidence) / 2

                    calculated_bucket_factors[bucket.value] = BucketFactors(
                        global_factor=b_factor,
                        hourly_factors=b_hourly_factors,
                        sample_count=len(b_ratios),
                        confidence=b_confidence,
                    )

            # Calculate confidence based on sample count and consistency
            std_dev = self._calculate_std_dev(clean_ratios) if len(clean_ratios) > 1 else 0.5
            consistency = max(0, 1 - std_dev)  # Lower std_dev = higher consistency
            sample_confidence = min(1.0, len(ratios) / MIN_SAMPLES_FOR_FULL_CONFIDENCE)
            confidence = (consistency + sample_confidence) / 2

            self._factors[group_name] = GroupCalibrationFactors(
                group_name=group_name,
                global_factor=global_factor,
                hourly_factors=hourly_factors,
                bucket_factors=calculated_bucket_factors,
                sample_count=len(ratios),
                confidence=confidence,
                last_updated=datetime.now().isoformat(),
            )

            # Enhanced logging with bucket info
            bucket_info = ", ".join(
                f"{b}={bf.global_factor:.2f}({bf.sample_count})"
                for b, bf in calculated_bucket_factors.items()
            ) or "none"
            _LOGGER.debug(
                f"Calibration factor for {group_name}: "
                f"global={global_factor:.3f}, "
                f"hourly={len(hourly_factors)} hours, "
                f"buckets=[{bucket_info}], "
                f"confidence={confidence:.2f}"
            )

    def _remove_outliers(self, values: List[float]) -> List[float]:
        """Remove outliers using IQR method."""
        if len(values) < 4:
            return values

        sorted_vals = sorted(values)
        q1_idx = len(sorted_vals) // 4
        q3_idx = 3 * len(sorted_vals) // 4
        q1 = sorted_vals[q1_idx]
        q3 = sorted_vals[q3_idx]
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        return [v for v in values if lower <= v <= upper]

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    async def collect_daily_calibration_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect yesterday's data for calibration from hourly_predictions.json.

        Returns:
            Dict mapping group_name to list of hourly calibration data
        """
        import asyncio

        predictions_file = self.stats_dir / "hourly_predictions.json"
        panel_cache_file = self.stats_dir / "panel_group_today_cache.json"

        yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()

        result: Dict[str, List[Dict[str, Any]]] = {}

        # First, load cloud_cover data from hourly_predictions for weather bucket learning
        cloud_cover_by_hour: Dict[int, float] = {}
        if predictions_file.exists():
            try:
                def _read_predictions():
                    with open(predictions_file, "r") as f:
                        return json.load(f)

                loop = asyncio.get_event_loop()
                pred_data = await loop.run_in_executor(None, _read_predictions)

                for pred in pred_data.get("predictions", []):
                    target_date = pred.get("target_date", "")
                    # Accept both yesterday and today (for end-of-day calibration)
                    if target_date not in [yesterday, datetime.now().date().isoformat()]:
                        continue

                    hour = pred.get("target_hour", 0)
                    # Get cloud_cover from weather_corrected or weather_forecast
                    weather = pred.get("weather_corrected") or pred.get("weather_forecast") or {}
                    clouds = weather.get("clouds")
                    if clouds is not None:
                        cloud_cover_by_hour[hour] = float(clouds)

                if cloud_cover_by_hour:
                    _LOGGER.debug(f"Loaded cloud_cover for {len(cloud_cover_by_hour)} hours")

            except Exception as e:
                _LOGGER.warning(f"Could not load cloud_cover from predictions: {e}")

        # Try panel_group_today_cache first (has per-group data)
        if panel_cache_file.exists():
            try:
                def _read():
                    with open(panel_cache_file, "r") as f:
                        return json.load(f)

                loop = asyncio.get_event_loop()
                cache_data = await loop.run_in_executor(None, _read)

                cache_date = cache_data.get("date", "")
                groups = cache_data.get("groups", {})

                # We need yesterday's data, but cache might have today's
                # For now, use whatever is in the cache as a starting point
                for group_name, group_info in groups.items():
                    result[group_name] = []
                    hourly = group_info.get("hourly", [])

                    for hour_data in hourly:
                        hour = hour_data.get("hour", 0)
                        pred = hour_data.get("prediction_kwh", 0)
                        actual = hour_data.get("actual_kwh")

                        if actual is not None and pred > 0:
                            entry = {
                                "hour": hour,
                                "physics_kwh": pred,
                                "actual_kwh": actual,
                            }
                            # Add cloud_cover if available (enables weather bucket learning)
                            if hour in cloud_cover_by_hour:
                                entry["cloud_cover"] = cloud_cover_by_hour[hour]
                            result[group_name].append(entry)

                if result:
                    samples_with_clouds = sum(
                        1 for group in result.values()
                        for entry in group
                        if "cloud_cover" in entry
                    )
                    _LOGGER.debug(
                        f"Collected calibration data from panel_group_today_cache: "
                        f"{sum(len(v) for v in result.values())} samples "
                        f"({samples_with_clouds} with cloud_cover)"
                    )
                    return result

            except Exception as e:
                _LOGGER.warning(f"Could not read panel_group_today_cache: {e}")

        # Fallback to hourly_predictions.json
        if predictions_file.exists():
            try:
                def _read():
                    with open(predictions_file, "r") as f:
                        return json.load(f)

                loop = asyncio.get_event_loop()
                pred_data = await loop.run_in_executor(None, _read)

                predictions = pred_data.get("predictions", [])

                # Group by panel group from predictions
                for pred in predictions:
                    target_date = pred.get("target_date", "")
                    if target_date != yesterday:
                        continue

                    hour = pred.get("target_hour", 0)
                    actual = pred.get("actual_kwh")

                    if actual is None:
                        continue

                    # Get cloud_cover for weather bucket learning
                    weather = pred.get("weather_corrected") or pred.get("weather_forecast") or {}
                    cloud_cover = weather.get("clouds")

                    # Check for panel group predictions
                    group_preds = pred.get("panel_group_predictions")
                    if group_preds:
                        for gp in group_preds:
                            group_name = gp.get("name", "Unknown")
                            physics = gp.get("power_kwh", 0)

                            # We need per-group actual - approximate from contribution
                            contrib = gp.get("contribution_percent", 50) / 100
                            group_actual = actual * contrib

                            if group_name not in result:
                                result[group_name] = []

                            entry = {
                                "hour": hour,
                                "physics_kwh": physics,
                                "actual_kwh": group_actual,
                            }
                            if cloud_cover is not None:
                                entry["cloud_cover"] = float(cloud_cover)
                            result[group_name].append(entry)
                    else:
                        # Single group fallback
                        pred_kwh = pred.get("prediction_kwh", 0)
                        if "Default" not in result:
                            result["Default"] = []
                        entry = {
                            "hour": hour,
                            "physics_kwh": pred_kwh,
                            "actual_kwh": actual,
                        }
                        if cloud_cover is not None:
                            entry["cloud_cover"] = float(cloud_cover)
                        result["Default"].append(entry)

                if result:
                    _LOGGER.debug(
                        f"Collected calibration data from hourly_predictions: "
                        f"{sum(len(v) for v in result.values())} samples for {yesterday}"
                    )

            except Exception as e:
                _LOGGER.warning(f"Could not read hourly_predictions: {e}")

        return result

    async def run_daily_calibration(self) -> CalibrationResult:
        """Run daily calibration using yesterday's data.

        This should be called once per day (e.g., at midnight or early morning).

        Returns:
            CalibrationResult with calibration outcome
        """
        yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()

        _LOGGER.info(f"Starting daily physics calibration for {yesterday}")

        # Collect data
        group_data = await self.collect_daily_calibration_data()

        if not group_data:
            _LOGGER.warning("No calibration data available for yesterday")
            return CalibrationResult(
                success=False,
                message="No calibration data available",
            )

        # Run calibration
        return await self.calibrate_from_daily_data(yesterday, group_data)

    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of current calibration state including weather buckets."""
        summary = {
            "calibrated_groups": len(self._factors),
            "history_days": len(self._history),
            "weather_buckets": [b.value for b in WeatherBucket],
            "groups": {},
        }

        for group_name, factors in self._factors.items():
            # Build bucket summary
            bucket_summary = {}
            for bucket_name, bf in factors.bucket_factors.items():
                bucket_summary[bucket_name] = {
                    "factor": round(bf.global_factor, 3),
                    "samples": bf.sample_count,
                    "confidence": round(bf.confidence, 2),
                }

            summary["groups"][group_name] = {
                "global_factor": round(factors.global_factor, 3),
                "hourly_factors_count": len(factors.hourly_factors),
                "bucket_factors": bucket_summary,
                "sample_count": factors.sample_count,
                "confidence": round(factors.confidence, 2),
                "last_updated": factors.last_updated,
            }

        return summary
