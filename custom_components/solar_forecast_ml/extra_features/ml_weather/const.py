# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML - ML Weather
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************
"""Constants for the ML Weather integration."""

from datetime import timedelta
from homeassistant.const import Platform

# Base component constants
DOMAIN = "ml_weather"
NAME = "ML Weather"
VERSION = "6.2.0"  # Must match manifest.json
ATTRIBUTION = "Data provided by Solar Forecast ML (Multi-Expert Blended & Corrected)"

# Platforms
PLATFORMS = [
    Platform.SENSOR,
    Platform.WEATHER,
]

# Update interval
DEFAULT_SCAN_INTERVAL = timedelta(minutes=15)

# Data keys
ML_WEATHER_COORDINATOR = "ml_weather_coordinator"
ML_WEATHER_DATA = "ml_weather_data"

# Config keys
CONF_DATA_PATH = "data_path"
CONF_NAME = "name"

# Paths
# Source path (Solar Forecast ML weather integration cache - updated 5x daily)
# Falls back to weather_forecast_corrected.json if integration cache not available
DEFAULT_SOURCE_PATH = "/config/solar_forecast_ml/stats/weather_integration_ml.json"
FALLBACK_SOURCE_PATH = "/config/solar_forecast_ml/stats/weather_forecast_corrected.json"

# Cache directory (own cache for independence)
CACHE_DIR = "/config/sfml_weather"
CACHE_FILE = "weather_cache.json"

# PV Forecast source path
PV_FORECAST_SOURCE_PATH = "/config/solar_forecast_ml/stats/daily_forecasts.json"

# Legacy alias for config flow
DEFAULT_DATA_PATH = DEFAULT_SOURCE_PATH

# Forecast attributes (extra beyond standard HA)
ATTR_FORECAST_CLOUD_COVERAGE = "cloud_coverage"
ATTR_FORECAST_CLOUD_COVER_LOW = "cloud_cover_low"
ATTR_FORECAST_CLOUD_COVER_MID = "cloud_cover_mid"
ATTR_FORECAST_CLOUD_COVER_HIGH = "cloud_cover_high"
ATTR_FORECAST_PRESSURE = "pressure"
ATTR_FORECAST_HUMIDITY = "humidity"
ATTR_FORECAST_SOLAR_RADIATION = "solar_radiation"
ATTR_FORECAST_DIRECT_RADIATION = "direct_radiation"
ATTR_FORECAST_DIFFUSE_RADIATION = "diffuse_radiation"

# Sensor types
SENSOR_TYPE_TEMPERATURE = "temperature"
SENSOR_TYPE_HUMIDITY = "humidity"
SENSOR_TYPE_PRESSURE = "pressure"
SENSOR_TYPE_WIND_SPEED = "wind_speed"
SENSOR_TYPE_CLOUD_COVERAGE = "cloud_coverage"
SENSOR_TYPE_PRECIPITATION = "precipitation"
SENSOR_TYPE_SOLAR_RADIATION = "solar_radiation"
SENSOR_TYPE_DIRECT_RADIATION = "direct_radiation"
SENSOR_TYPE_DIFFUSE_RADIATION = "diffuse_radiation"
SENSOR_TYPE_CLOUD_COVER_LOW = "cloud_cover_low"
SENSOR_TYPE_CLOUD_COVER_MID = "cloud_cover_mid"
SENSOR_TYPE_CLOUD_COVER_HIGH = "cloud_cover_high"
SENSOR_TYPE_VISIBILITY = "visibility"
SENSOR_TYPE_FOG_DETECTED = "fog_detected"
SENSOR_TYPE_FOG_TYPE = "fog_type"

# PV Forecast sensor types
SENSOR_TYPE_PV_FORECAST_TODAY = "pv_forecast_today"
SENSOR_TYPE_PV_FORECAST_TOMORROW = "pv_forecast_tomorrow"
SENSOR_TYPE_PV_FORECAST_DAY_AFTER = "pv_forecast_day_after_tomorrow"

# Weather conditions mapping (clouds % to condition)
# Ranges are [low, high) - upper bound is exclusive, except 100 which is handled separately
CONDITION_MAP = {
    (0, 10): "sunny",
    (10, 25): "partlycloudy",
    (25, 50): "partlycloudy",
    (50, 75): "cloudy",
    (75, 101): "cloudy",  # Include 100%
}

# Forecast constants
FORECAST_HOURS = 120  # Number of hours for hourly forecast (5 days)
FORECAST_DAYS = 5     # Number of days for daily forecast

# Rain thresholds for condition determination (mm)
RAIN_THRESHOLD_LIGHT = 0.0   # Any rain
RAIN_THRESHOLD_MODERATE = 0.5  # Moderate rain

# Wind speed conversion factor (m/s to km/h)
WIND_SPEED_CONVERSION = 3.6

# Cache constants
CACHE_MAX_AGE_HOURS = 24  # Maximum cache age before considered stale
CACHE_MAX_SIZE_MB = 10    # Maximum cache size in MB
