"""ML Data Loader - Load training data from new structure"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import numpy as np
import logging
import asyncio

_LOGGER = logging.getLogger(__name__)


class MLDataLoader:
    """Load and prepare training data from hourly_predictions.json"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.hourly_file = data_dir / "stats" / "hourly_predictions.json"
        self.summaries_file = data_dir / "stats" / "daily_summaries.json"

    async def load_training_data_async(
        self,
        min_days: int = 30,
        max_days: int = 365,
        exclude_outliers: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load training data for ML model (async, non-blocking)

        Returns:
            features: np.ndarray shape (n_samples, n_features)
            targets: np.ndarray shape (n_samples,)
            feature_names: List of feature names
        """
        def _load_sync():
            try:
                if not self.hourly_file.exists():
                    _LOGGER.warning("hourly_predictions.json not found")
                    return np.array([]), np.array([]), []

                with open(self.hourly_file, 'r') as f:
                    data = json.load(f)

                return data
            except Exception as e:
                _LOGGER.error(f"Error reading training data file: {e}", exc_info=True)
                return None

        try:
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, _load_sync)

            if data is None or not isinstance(data, dict):
                return np.array([]), np.array([]), []

            predictions = data.get("predictions", [])

            # Filter: Only predictions with actual values
            training_predictions = [
                p for p in predictions
                if p.get("actual_kwh") is not None
            ]

            if len(training_predictions) < min_days * 10:
                _LOGGER.warning(
                    f"Not enough training data: {len(training_predictions)} samples "
                    f"(need at least {min_days * 10})"
                )
                return np.array([]), np.array([]), []

            # Optional: Exclude outliers
            if exclude_outliers:
                training_predictions = [
                    p for p in training_predictions
                    if not p.get("flags", {}).get("is_outlier", False)
                ]

            # Extract features and targets
            features_list = []
            targets_list = []

            for p in training_predictions:
                features = self._extract_features(p)
                target = p["actual_kwh"]

                if features is not None:
                    features_list.append(features)
                    targets_list.append(target)

            if not features_list:
                return np.array([]), np.array([]), []

            features_array = np.array(features_list)
            targets_array = np.array(targets_list)
            feature_names = self._get_feature_names()

            _LOGGER.info(
                f"Loaded {len(features_array)} training samples with "
                f"{len(feature_names)} features"
            )

            return features_array, targets_array, feature_names

        except Exception as e:
            _LOGGER.error(f"Error loading training data: {e}", exc_info=True)
            return np.array([]), np.array([]), []

    def load_training_data(
        self,
        min_days: int = 30,
        max_days: int = 365,
        exclude_outliers: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """DEPRECATED: Sync version - use load_training_data_async() instead"""
        try:
            if not self.hourly_file.exists():
                _LOGGER.warning("hourly_predictions.json not found")
                return np.array([]), np.array([]), []

            with open(self.hourly_file, 'r') as f:
                data = json.load(f)

            predictions = data.get("predictions", [])
            training_predictions = [
                p for p in predictions
                if p.get("actual_kwh") is not None
            ]

            if len(training_predictions) < min_days * 10:
                _LOGGER.warning(
                    f"Not enough training data: {len(training_predictions)} samples "
                    f"(need at least {min_days * 10})"
                )
                return np.array([]), np.array([]), []

            if exclude_outliers:
                training_predictions = [
                    p for p in training_predictions
                    if not p.get("flags", {}).get("is_outlier", False)
                ]

            features_list = []
            targets_list = []

            for p in training_predictions:
                features = self._extract_features(p)
                target = p["actual_kwh"]

                if features is not None:
                    features_list.append(features)
                    targets_list.append(target)

            if not features_list:
                return np.array([]), np.array([]), []

            features_array = np.array(features_list)
            targets_array = np.array(targets_list)
            feature_names = self._get_feature_names()

            _LOGGER.info(
                f"Loaded {len(features_array)} training samples with "
                f"{len(feature_names)} features"
            )

            return features_array, targets_array, feature_names

        except Exception as e:
            _LOGGER.error(f"Error loading training data: {e}", exc_info=True)
            return np.array([]), np.array([]), []

    def _extract_features(self, prediction: Dict) -> Optional[List[float]]:
        """Extract feature vector from prediction"""
        try:
            wf = prediction.get("weather_forecast", {})
            sensor = prediction.get("sensor_actual", {})
            astro = prediction.get("astronomy", {})
            hist = prediction.get("historical_context", {})

            # Season mapping
            season_map = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}

            features = [
                # Time features
                prediction.get("target_hour", 0),
                prediction.get("target_day_of_year", 0),
                prediction.get("target_month", 0),
                prediction.get("target_day_of_week", 0),
                season_map.get(prediction.get("target_season", "summer"), 2),

                # Astronomy
                astro.get("sun_elevation_deg", 0) or 0,
                astro.get("hours_after_sunrise", 0) or 0,
                astro.get("hours_before_sunset", 0) or 0,

                # Weather forecast
                wf.get("temperature_c", 15) or 15,
                wf.get("cloud_cover_percent", 50) or 50,
                wf.get("humidity_percent", 50) or 50,
                wf.get("wind_speed_ms", 0) or 0,
                wf.get("precipitation_mm", 0) or 0,
                wf.get("pressure_hpa", 1013) or 1013,

                # Sensors (with fallbacks)
                self._get_lux_feature(sensor, wf),
                sensor.get("temperature_c") or wf.get("temperature_c", 15) or 15,
                sensor.get("humidity_percent") or wf.get("humidity_percent", 50) or 50,

                # Historical
                hist.get("yesterday_same_hour", 0) or 0,
                hist.get("same_hour_avg_7days", 0) or 0,
                sensor.get("current_yield_kwh", 0) or 0,
            ]

            # Validate: no NaN values
            if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in features):
                return None

            return features

        except Exception as e:
            _LOGGER.debug(f"Error extracting features: {e}")
            return None

    def _get_lux_feature(self, sensor: Dict, weather: Dict) -> float:
        """Get LUX value - prefer sensor, fallback to solar radiation estimate"""
        lux = sensor.get("lux")
        if lux is not None:
            return float(lux)

        # Estimate from solar radiation if available
        solar_rad = weather.get("solar_radiation_wm2")
        if solar_rad:
            # Rough conversion: 1 W/m² ≈ 100 lux (varies)
            return float(solar_rad) * 100

        return 0.0

    def _get_feature_names(self) -> List[str]:
        """Get feature names in order"""
        return [
            "hour",
            "day_of_year",
            "month",
            "day_of_week",
            "season",
            "sun_elevation_deg",
            "hours_after_sunrise",
            "hours_before_sunset",
            "temperature_forecast",
            "cloud_cover_forecast",
            "humidity_forecast",
            "wind_speed_forecast",
            "precipitation_forecast",
            "pressure_forecast",
            "lux",
            "temperature_actual",
            "humidity_actual",
            "production_yesterday_same_hour",
            "production_7days_avg_same_hour",
            "current_yield_today",
        ]

    async def get_learned_patterns_async(self) -> Dict:
        """Get learned patterns from daily summaries (async, non-blocking)"""
        def _load_sync():
            try:
                if not self.summaries_file.exists():
                    return {}

                with open(self.summaries_file, 'r') as f:
                    data = json.load(f)

                return data
            except Exception as e:
                _LOGGER.error(f"Error reading summaries file: {e}")
                return None

        try:
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, _load_sync)

            if data is None or not isinstance(data, dict):
                return {}

            summaries = data.get("summaries", [])

            # Aggregate patterns across days
            all_patterns = []
            for summary in summaries:
                patterns = summary.get("patterns", [])
                all_patterns.extend(patterns)

            # Group by type
            patterns_by_type = {}
            for pattern in all_patterns:
                ptype = pattern.get("type")
                if ptype not in patterns_by_type:
                    patterns_by_type[ptype] = []
                patterns_by_type[ptype].append(pattern)

            return patterns_by_type

        except Exception as e:
            _LOGGER.error(f"Error loading learned patterns: {e}")
            return {}

    def get_learned_patterns(self) -> Dict:
        """DEPRECATED: Sync version - use get_learned_patterns_async() instead"""
        try:
            if not self.summaries_file.exists():
                return {}

            with open(self.summaries_file, 'r') as f:
                data = json.load(f)

            summaries = data.get("summaries", [])

            all_patterns = []
            for summary in summaries:
                patterns = summary.get("patterns", [])
                all_patterns.extend(patterns)

            patterns_by_type = {}
            for pattern in all_patterns:
                ptype = pattern.get("type")
                if ptype not in patterns_by_type:
                    patterns_by_type[ptype] = []
                patterns_by_type[ptype].append(pattern)

            return patterns_by_type

        except Exception as e:
            _LOGGER.error(f"Error loading learned patterns: {e}")
            return {}
