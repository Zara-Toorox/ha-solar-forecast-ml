"""
Feature Engineering V2 - For hourly_predictions.json + astronomy_cache.json

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import math
import logging
from typing import Dict, Any, List

_LOGGER = logging.getLogger(__name__)


class FeatureEngineerV2:
    """Feature engineering for V2 data structure"""

    def __init__(self):
        """Initialize feature engineer"""
        self.feature_names = self._build_feature_names()
        _LOGGER.info(f"FeatureEngineerV2 initialized with {len(self.feature_names)} features")

    def _build_feature_names(self) -> List[str]:
        """Build complete feature list"""
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

    def extract_features(self, record: Dict[str, Any]) -> List[float]:
        """Extract feature vector from enriched record"""
        try:
            # Extract data sections
            weather = record.get('weather_data', {})
            sensor = record.get('sensor_data', {})
            astro_basic = record.get('astronomy_basic', {})
            astro_advanced = record.get('astronomy_advanced', {})

            # --- TIME FEATURES (6) ---
            target_hour = float(record.get('target_hour', 12))
            target_day_of_year = float(record.get('target_day_of_year', 180))
            target_month = float(record.get('target_month', 6))
            target_day_of_week = float(record.get('target_day_of_week', 0))

            # Encode season: winter=0, spring=1, summer=2, autumn=3
            season_str = record.get('target_season', 'summer')
            season_map = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
            target_season_encoded = float(season_map.get(season_str, 2))

            prediction_created_hour = float(record.get('prediction_created_hour', target_hour))
            prediction_horizon = abs(target_hour - prediction_created_hour)

            # --- WEATHER FORECAST (9) ---
            weather_temp_c = self._safe_float(weather.get('temperature_c'), default=15.0)
            weather_cloud_percent = self._safe_float(weather.get('cloud_cover_percent'), default=50.0)
            weather_humidity_percent = self._safe_float(weather.get('humidity_percent'), default=60.0)
            weather_wind_ms = self._safe_float(weather.get('wind_speed_ms'), default=5.0)
            weather_precipitation_mm = self._safe_float(weather.get('precipitation_mm'), default=0.0)
            weather_pressure_hpa = self._safe_float(weather.get('pressure_hpa'), default=1013.0)
            weather_feels_like_c = self._safe_float(weather.get('feels_like_c'), default=weather_temp_c)
            weather_dew_point_c = self._safe_float(weather.get('dew_point_c'), default=weather_temp_c - 5.0)
            weather_visibility_km = self._safe_float(weather.get('visibility_km'), default=10.0)

            # --- SENSOR ACTUAL (6) ---
            sensor_lux = self._safe_float(sensor.get('lux'), default=0.0)
            sensor_uv_index = self._safe_float(sensor.get('uv_index'), default=0.0)
            sensor_temp = self._safe_float(sensor.get('temperature'), default=weather_temp_c)
            sensor_humidity = self._safe_float(sensor.get('humidity'), default=weather_humidity_percent)
            sensor_rain = self._safe_float(sensor.get('rain'), default=0.0)
            sensor_wind = self._safe_float(sensor.get('wind_speed'), default=weather_wind_ms)

            # --- ASTRONOMY BASIC (4) ---
            astro_basic_elevation = self._safe_float(astro_basic.get('sun_elevation_deg'), default=0.0)
            astro_basic_hours_after_sunrise = self._safe_float(astro_basic.get('hours_after_sunrise'), default=0.0)
            astro_basic_hours_before_sunset = self._safe_float(astro_basic.get('hours_before_sunset'), default=0.0)
            astro_basic_daylight_hours = self._safe_float(astro_basic.get('daylight_hours'), default=12.0)

            # --- ASTRONOMY ADVANCED (6) ---
            astro_elevation_deg = self._safe_float(astro_advanced.get('elevation_deg'), default=astro_basic_elevation)
            astro_azimuth_deg = self._safe_float(astro_advanced.get('azimuth_deg'), default=180.0)
            astro_clear_sky_radiation_wm2 = self._safe_float(astro_advanced.get('clear_sky_solar_radiation_wm2'), default=0.0)
            astro_theoretical_max_kwh = self._safe_float(astro_advanced.get('theoretical_max_pv_kwh'), default=0.0)
            astro_hours_since_solar_noon = self._safe_float(astro_advanced.get('hours_since_solar_noon'), default=0.0)
            astro_day_progress_ratio = self._safe_float(astro_advanced.get('day_progress_ratio'), default=0.5)

            # --- LAG FEATURES (2) ---
            production_yesterday = self._safe_float(sensor.get('production_yesterday'), default=0.0)
            production_same_hour_yesterday = self._safe_float(sensor.get('production_same_hour_yesterday'), default=0.0)

            # --- CONTEXT (1) ---
            is_production_hour = float(record.get('is_production_hour', False))

            # --- DERIVED FEATURES (10) ---
            sunshine_percent = max(0.0, 100.0 - weather_cloud_percent)
            cloud_impact = weather_cloud_percent ** 1.5
            temp_elevation_interaction = weather_temp_c * max(0.0, astro_elevation_deg)

            # Radiation efficiency (actual vs theoretical)
            radiation_efficiency = 0.0
            if astro_theoretical_max_kwh > 0.01:
                # Use sensor lux as proxy for actual radiation
                radiation_efficiency = min(1.0, sensor_lux / max(1.0, astro_clear_sky_radiation_wm2))

            hour_of_day_sq = target_hour ** 2
            elevation_sq = max(0.0, astro_elevation_deg) ** 2
            cloud_x_hour = weather_cloud_percent * target_hour

            # Seasonal factor (cosine, peaks in summer)
            seasonal_factor = 0.5 + 0.5 * math.cos((target_day_of_year - 172) * 2 * math.pi / 365.25)
            temp_x_season = weather_temp_c * seasonal_factor

            radiation_x_cloud = astro_clear_sky_radiation_wm2 * (sunshine_percent / 100.0)

            # Assemble feature vector
            features = [
                # Time (6)
                target_hour,
                target_day_of_year,
                target_month,
                target_day_of_week,
                target_season_encoded,
                prediction_horizon,

                # Weather (9)
                weather_temp_c,
                weather_cloud_percent,
                weather_humidity_percent,
                weather_wind_ms,
                weather_precipitation_mm,
                weather_pressure_hpa,
                weather_feels_like_c,
                weather_dew_point_c,
                weather_visibility_km,

                # Sensor (6)
                sensor_lux,
                sensor_uv_index,
                sensor_temp,
                sensor_humidity,
                sensor_rain,
                sensor_wind,

                # Astro Basic (4)
                astro_basic_elevation,
                astro_basic_hours_after_sunrise,
                astro_basic_hours_before_sunset,
                astro_basic_daylight_hours,

                # Astro Advanced (6)
                astro_elevation_deg,
                astro_azimuth_deg,
                astro_clear_sky_radiation_wm2,
                astro_theoretical_max_kwh,
                astro_hours_since_solar_noon,
                astro_day_progress_ratio,

                # Lag (2)
                production_yesterday,
                production_same_hour_yesterday,

                # Context (1)
                is_production_hour,

                # Derived (10)
                sunshine_percent,
                cloud_impact,
                temp_elevation_interaction,
                radiation_efficiency,
                hour_of_day_sq,
                elevation_sq,
                cloud_x_hour,
                temp_x_season,
                radiation_x_cloud,
                seasonal_factor,
            ]

            # Validate
            if len(features) != len(self.feature_names):
                _LOGGER.error(
                    f"Feature count mismatch! Expected {len(self.feature_names)}, got {len(features)}"
                )
                return None

            # Check for NaN/Inf
            for i, val in enumerate(features):
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    _LOGGER.warning(f"Invalid value for feature {self.feature_names[i]}: {val}, setting to 0.0")
                    features[i] = 0.0

            return features

        except Exception as e:
            _LOGGER.error(f"Feature extraction failed: {e}", exc_info=True)
            return None

    def _safe_float(self, value: Any, default: float = 0.0, min_val: float = None, max_val: float = None) -> float:
        """Safely convert value to float with bounds"""
        try:
            if value is None:
                return default

            val = float(value)

            if math.isnan(val) or math.isinf(val):
                return default

            if min_val is not None:
                val = max(min_val, val)

            if max_val is not None:
                val = min(max_val, val)

            return val

        except (ValueError, TypeError):
            return default
