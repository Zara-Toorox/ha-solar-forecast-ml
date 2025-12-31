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
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


class FeatureEngineer:
    """Extract normalized features for Multi-Output AI @zara"""

    # 17 base features - same for all groups
    BASE_FEATURE_NAMES = [
        "hour_of_day",
        "day_of_year",
        "season_encoded",
        "weather_temp",
        "weather_ghi",
        "weather_wind",
        "weather_humidity",
        "weather_rain",
        "weather_clouds",
        "sun_elevation_deg",
        "theoretical_max_kwh",
        "clear_sky_radiation",
        "production_yesterday",
        "production_same_hour",
        "weather_dni",
        "dni_ratio",
        "elevation_normalized",
    ]

    # 3 group-specific features - vary per panel group
    GROUP_FEATURE_NAMES = [
        "group_azimuth_normalized",   # azimuth / 360
        "group_tilt_normalized",       # tilt / 90
        "group_capacity_normalized",   # group_kwp / total_kwp
    ]

    # All 20 features combined
    FEATURE_NAMES = BASE_FEATURE_NAMES + GROUP_FEATURE_NAMES

    # Normalization ranges for each feature
    FEATURE_RANGES = {
        # Base features
        "hour_of_day": (0, 23),
        "day_of_year": (1, 365),
        "season_encoded": (0, 3),
        "weather_temp": (-20, 45),
        "weather_ghi": (0, 1200),
        "weather_wind": (0, 30),
        "weather_humidity": (0, 100),
        "weather_rain": (0, 50),
        "weather_clouds": (0, 100),
        "sun_elevation_deg": (-10, 90),
        "theoretical_max_kwh": (0, 2),
        "clear_sky_radiation": (0, 1200),
        "production_yesterday": (0, 20),
        "production_same_hour": (0, 3),
        "weather_dni": (0, 1000),
        "dni_ratio": (0, 1.5),
        "elevation_normalized": (0, 1),
        # Group-specific features (already normalized 0-1)
        "group_azimuth_normalized": (0, 1),
        "group_tilt_normalized": (0, 1),
        "group_capacity_normalized": (0, 1),
    }

    def __init__(self, dni_tracker: Optional[Any] = None):
        """Initialize with optional DNI tracker for dni_ratio @zara"""
        self.dni_tracker = dni_tracker
        _LOGGER.info(f"FeatureEngineer initialized: {len(self.FEATURE_NAMES)} features (normalized)")

    @property
    def feature_names(self) -> List[str]:
        """Feature names @zara"""
        return self.FEATURE_NAMES

    @property
    def feature_count(self) -> int:
        """Number of features @zara"""
        return len(self.FEATURE_NAMES)

    def extract_base_features(self, record: Dict[str, Any]) -> Optional[List[float]]:
        """Extract 17 base normalized features (without group-specific) @zara

        Returns:
            List of 17 normalized base features, or None on error
        """
        try:
            hour = float(record.get("target_hour", 12))
            day_of_year = float(record.get("target_day_of_year", 180))
            month = int(record.get("target_month", 6))

            season_str = record.get("target_season", "")
            if season_str:
                season_map = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
                season = float(season_map.get(season_str, 2))
            else:
                if month in [12, 1, 2]:
                    season = 0.0
                elif month in [3, 4, 5]:
                    season = 1.0
                elif month in [6, 7, 8]:
                    season = 2.0
                else:
                    season = 3.0

            weather = record.get("weather_corrected", {})
            temp = self._safe_float(weather.get("temperature"), 15.0)
            ghi = self._safe_float(weather.get("solar_radiation_wm2"), 0.0)
            wind = self._safe_float(weather.get("wind"), 3.0)
            humidity = self._safe_float(weather.get("humidity"), 70.0)
            rain = self._safe_float(weather.get("rain"), 0.0)
            clouds = self._safe_float(weather.get("clouds"), 50.0)
            dni = self._safe_float(weather.get("dni"), 0.0)

            astronomy = record.get("astronomy", {})
            elevation = self._safe_float(astronomy.get("sun_elevation_deg"), -30.0)
            theoretical_max = self._safe_float(astronomy.get("theoretical_max_kwh"), 0.0)
            clear_sky = self._safe_float(astronomy.get("clear_sky_radiation_wm2"), 0.0)
            max_elevation = self._safe_float(astronomy.get("max_elevation_today"), 60.0)

            prod_yesterday = self._safe_float(record.get("production_yesterday"), 0.0)
            prod_same_hour = self._safe_float(
                record.get("production_same_hour_yesterday"), 0.0
            )

            dni_ratio = self._calculate_dni_ratio(dni, int(hour))
            elev_norm = elevation / max_elevation if max_elevation > 0 else 0.0
            elev_norm = max(0.0, min(1.0, elev_norm))

            # Raw base features (17)
            raw_features = [
                hour,
                day_of_year,
                season,
                temp,
                ghi,
                wind,
                humidity,
                rain,
                clouds,
                elevation,
                theoretical_max,
                clear_sky,
                prod_yesterday,
                prod_same_hour,
                dni,
                dni_ratio,
                elev_norm,
            ]

            if len(raw_features) != len(self.BASE_FEATURE_NAMES):
                _LOGGER.error(
                    f"Base feature count mismatch: {len(raw_features)} vs {len(self.BASE_FEATURE_NAMES)}"
                )
                return None

            # Normalize base features to 0-1 range
            normalized = []
            for val, name in zip(raw_features, self.BASE_FEATURE_NAMES):
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    _LOGGER.warning(f"Invalid {name}: {val}, using 0.5")
                    normalized.append(0.5)
                else:
                    min_val, max_val = self.FEATURE_RANGES.get(name, (0, 1))
                    norm = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                    norm = max(0.0, min(1.0, norm))
                    normalized.append(norm)

            return normalized

        except Exception as e:
            _LOGGER.error(f"Base feature extraction failed: {e}")
            return None

    def extract_group_features(
        self,
        group_azimuth: float,
        group_tilt: float,
        group_capacity_kwp: float,
        total_capacity_kwp: float,
    ) -> List[float]:
        """Extract 3 normalized group-specific features @zara

        Args:
            group_azimuth: Panel group azimuth in degrees (0-360)
            group_tilt: Panel group tilt in degrees (0-90)
            group_capacity_kwp: Group capacity in kWp
            total_capacity_kwp: Total system capacity in kWp

        Returns:
            List of 3 normalized group features
        """
        # Normalize azimuth: 0-360 -> 0-1
        azimuth_norm = max(0.0, min(1.0, group_azimuth / 360.0))

        # Normalize tilt: 0-90 -> 0-1
        tilt_norm = max(0.0, min(1.0, group_tilt / 90.0))

        # Normalize capacity: group_kwp / total_kwp -> 0-1
        if total_capacity_kwp > 0:
            capacity_norm = max(0.0, min(1.0, group_capacity_kwp / total_capacity_kwp))
        else:
            capacity_norm = 0.5

        return [azimuth_norm, tilt_norm, capacity_norm]

    def extract(
        self,
        record: Dict[str, Any],
        group_azimuth: float = 180.0,
        group_tilt: float = 30.0,
        group_capacity_kwp: float = 1.0,
        total_capacity_kwp: float = 1.0,
    ) -> Optional[List[float]]:
        """Extract full 20-feature vector for a specific panel group @zara

        Args:
            record: Data record with weather, astronomy, production data
            group_azimuth: Panel group azimuth in degrees (0-360), default 180 (south)
            group_tilt: Panel group tilt in degrees (0-90), default 30
            group_capacity_kwp: Group capacity in kWp
            total_capacity_kwp: Total system capacity in kWp

        Returns:
            List of 20 normalized features, or None on error
        """
        # Extract 17 base features
        base_features = self.extract_base_features(record)
        if base_features is None:
            return None

        # Extract 3 group-specific features
        group_features = self.extract_group_features(
            group_azimuth, group_tilt, group_capacity_kwp, total_capacity_kwp
        )

        # Combine: 17 base + 3 group = 20 features
        return base_features + group_features

    def extract_for_all_groups(
        self,
        record: Dict[str, Any],
        panel_groups: List[Dict[str, Any]],
    ) -> Optional[List[List[float]]]:
        """Extract 20-feature vectors for all panel groups @zara

        Args:
            record: Data record with weather, astronomy, production data
            panel_groups: List of panel group configs with azimuth, tilt, capacity_kwp

        Returns:
            List of feature vectors (one per group), or None on error
        """
        # Extract base features once (same for all groups)
        base_features = self.extract_base_features(record)
        if base_features is None:
            return None

        # Calculate total capacity (supports power_wp in Watts or capacity_kwp/kwp in kWp)
        total_capacity = sum(
            self._get_group_capacity_kwp(g) for g in panel_groups
        )

        # Generate feature vector for each group
        all_features = []
        for group in panel_groups:
            azimuth = self._safe_float(group.get("azimuth", 180.0), 180.0)
            tilt = self._safe_float(group.get("tilt", 30.0), 30.0)
            capacity = self._get_group_capacity_kwp(group)

            group_features = self.extract_group_features(
                azimuth, tilt, capacity, total_capacity
            )

            # Combine: 17 base + 3 group = 20 features
            all_features.append(base_features + group_features)

        return all_features

    def extract_combined_for_multi_output(
        self,
        record: Dict[str, Any],
        panel_groups: List[Dict[str, Any]],
    ) -> Optional[List[float]]:
        """Extract combined feature vector for Multi-Output LSTM @zara

        Creates a single feature vector containing:
        - 17 base features (weather, astronomy, time)
        - 3 features per panel group (azimuth, tilt, capacity)

        Total features = 17 + 3 * num_groups

        This is the CORRECT way to do multi-output: the network sees
        ALL group information and can predict ALL group outputs.

        Args:
            record: Data record with weather, astronomy, production data
            panel_groups: List of panel group configs

        Returns:
            Single combined feature vector, or None on error

        Example for 2 groups:
            [17 base features] + [3 group1 features] + [3 group2 features]
            = 23 total features
        """
        # Extract 17 base features (same for all groups)
        base_features = self.extract_base_features(record)
        if base_features is None:
            return None

        if not panel_groups:
            # No groups defined - return only base features
            return base_features

        # Calculate total capacity
        total_capacity = sum(
            self._get_group_capacity_kwp(g) for g in panel_groups
        )

        # Start with base features
        combined = list(base_features)

        # Append 3 features for EACH group
        for group in panel_groups:
            azimuth = self._safe_float(group.get("azimuth", 180.0), 180.0)
            tilt = self._safe_float(group.get("tilt", 30.0), 30.0)
            capacity = self._get_group_capacity_kwp(group)

            group_features = self.extract_group_features(
                azimuth, tilt, capacity, total_capacity
            )
            combined.extend(group_features)

        return combined

    def get_combined_feature_count(self, num_groups: int) -> int:
        """Calculate total feature count for combined multi-output @zara

        Args:
            num_groups: Number of panel groups (0 = single output mode)

        Returns:
            Total feature count: 17 + 3 * num_groups
        """
        if num_groups <= 0:
            return len(self.BASE_FEATURE_NAMES)  # 17
        return len(self.BASE_FEATURE_NAMES) + len(self.GROUP_FEATURE_NAMES) * num_groups

    def _get_group_capacity_kwp(self, group: Dict[str, Any]) -> float:
        """Get group capacity in kWp from various config formats @zara

        Supports:
        - capacity_kwp: direct kWp value
        - kwp: direct kWp value
        - power_wp: Watts (converted to kWp by /1000)
        """
        if "capacity_kwp" in group:
            return self._safe_float(group["capacity_kwp"], 1.0)
        elif "kwp" in group:
            return self._safe_float(group["kwp"], 1.0)
        elif "power_wp" in group:
            return self._safe_float(group["power_wp"], 1000.0) / 1000.0
        return 1.0

    def _safe_float(self, value: Any, default: float) -> float:
        """Convert value to float with fallback @zara"""
        try:
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    def _calculate_dni_ratio(self, dni: float, hour: int) -> float:
        """Calculate DNI ratio using tracker or fallback @zara"""
        if self.dni_tracker:
            max_dni = self.dni_tracker.get_max_dni(hour)
            if max_dni > 0:
                return min(1.5, dni / max_dni)
        return 0.5

    def get_defaults(self) -> List[float]:
        """Return default normalized feature vector @zara"""
        return [0.5] * len(self.FEATURE_NAMES)

    def validate(self, record: Dict[str, Any]) -> bool:
        """Check if record has required fields @zara"""
        required = ["target_hour", "target_day_of_year", "weather_corrected", "astronomy"]

        for field in required:
            if field not in record:
                _LOGGER.warning(f"Missing field: {field}")
                return False

        weather = record.get("weather_corrected", {})
        weather_fields = ["temperature", "solar_radiation_wm2", "wind", "humidity", "clouds"]
        for field in weather_fields:
            if field not in weather:
                _LOGGER.warning(f"Missing weather field: {field}")
                return False

        astronomy = record.get("astronomy", {})
        astro_fields = ["sun_elevation_deg", "theoretical_max_kwh", "clear_sky_radiation_wm2"]
        for field in astro_fields:
            if field not in astronomy:
                _LOGGER.warning(f"Missing astronomy field: {field}")
                return False

        return True
