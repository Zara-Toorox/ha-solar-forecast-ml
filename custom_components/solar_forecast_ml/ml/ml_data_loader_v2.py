"""
ML Data Loader V2 - Load training data from hourly_predictions.json + astronomy_cache.json

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)


class MLDataLoaderV2:
    """Load and prepare training data from hourly_predictions.json + astronomy_cache.json"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.hourly_predictions_file = data_dir / "stats" / "hourly_predictions.json"
        self.astronomy_cache_file = data_dir / "stats" / "astronomy_cache.json"
        self.hourly_samples_file = data_dir / "stats" / "hourly_samples.json"

    def load_training_data(
        self, min_samples: int = 10, exclude_outliers: bool = True
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Load training records from hourly_predictions.json + astronomy_cache.json

        Returns:
            records: List of training records with all features
            count: Number of valid records
        """

        def _load_sync():
            try:
                # Load hourly predictions
                if not self.hourly_predictions_file.exists():
                    _LOGGER.warning(
                        f"hourly_predictions.json not found at {self.hourly_predictions_file}"
                    )
                    return [], 0

                with open(self.hourly_predictions_file, "r") as f:
                    predictions_data = json.load(f)

                return predictions_data
            except Exception as e:
                _LOGGER.error(f"Error loading hourly_predictions.json: {e}", exc_info=True)
                return None

        try:
            # Synchronous loading (will be called from async context via run_in_executor)
            predictions_data = _load_sync()

            if predictions_data is None or not isinstance(predictions_data, dict):
                return [], 0

            predictions = predictions_data.get("predictions", [])
            _LOGGER.info(f"Loaded {len(predictions)} predictions from hourly_predictions.json")

            # Filter: Only predictions with actual values
            training_predictions = [p for p in predictions if p.get("actual_kwh") is not None]
            _LOGGER.info(f"Filtered to {len(training_predictions)} predictions with actual_kwh")

            if len(training_predictions) < min_samples:
                _LOGGER.warning(
                    f"Not enough training data: {len(training_predictions)} samples "
                    f"(need at least {min_samples})"
                )
                return [], 0

            # Optional: Exclude outliers
            if exclude_outliers:
                before_count = len(training_predictions)
                training_predictions = [
                    p
                    for p in training_predictions
                    if not p.get("flags", {}).get("is_outlier", False)
                ]
                _LOGGER.info(f"Excluded {before_count - len(training_predictions)} outliers")

            # Load astronomy cache
            astronomy_cache = self._load_astronomy_cache()

            # Load historical cache for lag features
            historical_cache = self._load_historical_cache()

            # Enrich predictions with astronomy data and lag features
            enriched_records = []
            for prediction in training_predictions:
                try:
                    record = self._enrich_prediction(prediction, astronomy_cache, historical_cache)
                    if record:
                        enriched_records.append(record)
                except Exception as e:
                    _LOGGER.debug(f"Failed to enrich prediction {prediction.get('id')}: {e}")
                    continue

            _LOGGER.info(
                f"Successfully enriched {len(enriched_records)} training records "
                f"from {len(training_predictions)} predictions"
            )

            return enriched_records, len(enriched_records)

        except Exception as e:
            _LOGGER.error(f"Error loading training data: {e}", exc_info=True)
            return [], 0

    def _load_astronomy_cache(self) -> Dict[str, Any]:
        """Load astronomy cache"""
        try:
            if not self.astronomy_cache_file.exists():
                _LOGGER.warning(f"astronomy_cache.json not found")
                return {}

            with open(self.astronomy_cache_file, "r") as f:
                data = json.load(f)

            days = data.get("days", {})
            _LOGGER.info(f"Loaded astronomy cache with {len(days)} days")
            return days

        except Exception as e:
            _LOGGER.error(f"Error loading astronomy cache: {e}")
            return {}

    def _load_historical_cache(self) -> Dict[str, Dict[str, float]]:
        """Load historical production cache for lag features from hourly_predictions.json"""
        try:
            # Use hourly_predictions.json directly (which has actual_kwh)
            # Use correct attribute name
            if not self.hourly_predictions_file.exists():
                _LOGGER.warning(f"hourly_predictions.json not found - cannot load lag features")
                return {"daily_productions": {}, "hourly_productions": {}}

            with open(self.hourly_predictions_file, "r") as f:
                data = json.load(f)

            predictions = data.get("predictions", [])

            # Build caches from hourly_predictions
            daily_productions = {}
            hourly_productions = {}

            # Group by date to calculate daily totals
            date_groups = {}
            for pred in predictions:
                target_date = pred.get("target_date")
                actual_kwh = pred.get("actual_kwh")
                target_hour = pred.get("target_hour")

                if not target_date or actual_kwh is None:
                    continue

                # Store hourly value
                hour_key = f"{target_date}_{target_hour:02d}"
                hourly_productions[hour_key] = actual_kwh

                # Accumulate for daily total
                if target_date not in date_groups:
                    date_groups[target_date] = []
                date_groups[target_date].append(actual_kwh)

            # Calculate daily totals
            for date, hourly_values in date_groups.items():
                daily_total = sum(hourly_values)
                daily_productions[date] = daily_total

            _LOGGER.info(
                f"Built historical cache from hourly_predictions: "
                f"{len(daily_productions)} days, {len(hourly_productions)} hours"
            )

            return {
                "daily_productions": daily_productions,
                "hourly_productions": hourly_productions,
            }

        except Exception as e:
            _LOGGER.error(f"Error loading historical cache: {e}")
            return {"daily_productions": {}, "hourly_productions": {}}

    def _enrich_prediction(
        self,
        prediction: Dict[str, Any],
        astronomy_cache: Dict[str, Any],
        historical_cache: Dict[str, Dict[str, float]],
    ) -> Optional[Dict[str, Any]]:
        """Enrich a prediction with astronomy data and lag features"""

        # Extract basic info
        target_date = prediction.get("target_date")
        target_hour = prediction.get("target_hour")

        if not target_date or target_hour is None:
            return None

        # Start with prediction data
        record = {
            # Copy all prediction data
            "timestamp": prediction.get("target_datetime"),
            "actual_kwh": prediction.get("actual_kwh"),
            # Time features
            "target_hour": target_hour,
            "target_day_of_year": prediction.get("target_day_of_year"),
            "target_month": prediction.get("target_month"),
            "target_day_of_week": prediction.get("target_day_of_week"),
            "target_season": prediction.get("target_season"),
            "prediction_created_hour": prediction.get("prediction_created_hour"),
            # Weather forecast
            "weather_data": prediction.get("weather_forecast", {}),
            # Sensor data
            "sensor_data": prediction.get("sensor_actual", {}),
            # Astronomy basic
            "astronomy_basic": prediction.get("astronomy", {}),
            # Flags
            "is_production_hour": prediction.get("flags", {}).get("is_production_hour", False),
        }

        # Add astronomy advanced features
        if target_date in astronomy_cache:
            day_data = astronomy_cache[target_date]
            hourly_data = day_data.get("hourly", {})
            hour_str = str(target_hour)

            if hour_str in hourly_data:
                hour_astro = hourly_data[hour_str]
                record["astronomy_advanced"] = {
                    "elevation_deg": hour_astro.get("elevation_deg"),
                    "azimuth_deg": hour_astro.get("azimuth_deg"),
                    "clear_sky_solar_radiation_wm2": hour_astro.get(
                        "clear_sky_solar_radiation_wm2"
                    ),
                    "theoretical_max_pv_kwh": hour_astro.get("theoretical_max_pv_kwh"),
                    "hours_since_solar_noon": hour_astro.get("hours_since_solar_noon"),
                    "day_progress_ratio": hour_astro.get("day_progress_ratio"),
                }
            else:
                record["astronomy_advanced"] = {}
        else:
            record["astronomy_advanced"] = {}

        # Add lag features
        try:
            dt = datetime.fromisoformat(target_date)
            from datetime import timedelta

            yesterday_date = (dt - timedelta(days=1)).date().isoformat()

            yesterday_total = historical_cache["daily_productions"].get(yesterday_date, 0.0)
            yesterday_same_hour_key = f"{yesterday_date}_{target_hour:02d}"
            yesterday_same_hour = historical_cache["hourly_productions"].get(
                yesterday_same_hour_key, 0.0
            )

            record["sensor_data"]["production_yesterday"] = yesterday_total
            record["sensor_data"]["production_same_hour_yesterday"] = yesterday_same_hour

        except Exception as e:
            _LOGGER.debug(f"Failed to add lag features: {e}")
            record["sensor_data"]["production_yesterday"] = 0.0
            record["sensor_data"]["production_same_hour_yesterday"] = 0.0

        return record

    def get_feature_names(self) -> List[str]:
        """Get feature names that will be extracted"""
        return [
            # Time features (6)
            "target_hour",
            "target_day_of_year",
            "target_month",
            "target_day_of_week",
            "target_season_encoded",
            "prediction_horizon",
            # Weather forecast (9)
            "weather_temp_c",
            "weather_cloud_percent",
            "weather_humidity_percent",
            "weather_wind_ms",
            "weather_precipitation_mm",
            "weather_pressure_hpa",
            "weather_feels_like_c",
            "weather_dew_point_c",
            "weather_visibility_km",
            # Sensor actual (6)
            "sensor_lux",
            "sensor_uv_index",
            "sensor_temp",
            "sensor_humidity",
            "sensor_rain",
            "sensor_wind",
            # Astronomy basic (4)
            "astro_basic_elevation",
            "astro_basic_hours_after_sunrise",
            "astro_basic_hours_before_sunset",
            "astro_basic_daylight_hours",
            # Astronomy advanced (6)
            "astro_elevation_deg",
            "astro_azimuth_deg",
            "astro_clear_sky_radiation_wm2",
            "astro_theoretical_max_kwh",
            "astro_hours_since_solar_noon",
            "astro_day_progress_ratio",
            # Lag features (2)
            "production_yesterday",
            "production_same_hour_yesterday",
            # Context (1)
            "is_production_hour",
            # Derived features (10)
            "sunshine_percent",
            "cloud_impact",
            "temp_elevation_interaction",
            "radiation_efficiency",
            "hour_of_day_sq",
            "elevation_sq",
            "cloud_x_hour",
            "temp_x_season",
            "radiation_x_cloud",
            "seasonal_factor",
        ]
