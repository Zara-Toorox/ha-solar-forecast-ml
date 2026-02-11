# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Solar Forecast ML - Constants @zara"""

from datetime import timedelta
from homeassistant.const import Platform

# Integration Identity @zara
DOMAIN = "solar_forecast_ml"
NAME = "Solar Forecast ML"
VERSION = "16.4.2"
SOFTWARE_VERSION = VERSION
AI_VERSION = "2.0"
INTEGRATION_MODEL = "Solar Forecast ML V16"

PLATFORMS = [Platform.SENSOR]

# Configuration Keys @zara
CONF_WEATHER_ENTITY = "weather_entity"
CONF_POWER_ENTITY = "power_entity"
CONF_SOLAR_YIELD_TODAY = "solar_yield_today"
CONF_SOLAR_CAPACITY = "solar_capacity"

# Optional Sensors @zara
CONF_TOTAL_CONSUMPTION_TODAY = "total_consumption_today"
CONF_GRID_IMPORT_TODAY = "grid_import_today"
CONF_GRID_EXPORT_TODAY = "grid_export_today"
CONF_RAIN_SENSOR = "rain_sensor"
CONF_LUX_SENSOR = "lux_sensor"
CONF_TEMP_SENSOR = "temp_sensor"
CONF_WIND_SENSOR = "wind_sensor"
CONF_HUMIDITY_SENSOR = "humidity_sensor"
CONF_PRESSURE_SENSOR = "pressure_sensor"
CONF_SOLAR_RADIATION_SENSOR = "solar_radiation_sensor"

# Panel Groups @zara
CONF_PANEL_GROUPS = "panel_groups"
CONF_PANEL_GROUP_POWER = "power_wp"
CONF_PANEL_GROUP_AZIMUTH = "azimuth"
CONF_PANEL_GROUP_TILT = "tilt"
CONF_PANEL_GROUP_NAME = "name"
CONF_PANEL_GROUP_ENERGY_SENSOR = "energy_sensor"

# Inverter Clipping @zara
CONF_INVERTER_MAX_POWER = "inverter_max_power"
DEFAULT_INVERTER_MAX_POWER = 0.0
INVERTER_CLIPPING_THRESHOLD = 0.95

# Default Panel Values @zara
DEFAULT_PANEL_AZIMUTH = 180
DEFAULT_PANEL_TILT = 30

# External Sensor Mapping @zara
EXTERNAL_SENSOR_MAPPING = {
    "temperature": CONF_TEMP_SENSOR,
    "humidity": CONF_HUMIDITY_SENSOR,
    "wind_speed": CONF_WIND_SENSOR,
    "rain": CONF_RAIN_SENSOR,
    "pressure": CONF_PRESSURE_SENSOR,
    "solar_radiation": CONF_SOLAR_RADIATION_SENSOR,
    "lux": CONF_LUX_SENSOR,
}

# Weather Configuration @zara
CONF_WEATHER_PREFERENCE = "weather_preference"
CONF_FALLBACK_ENTITY = "fallback_weather_entity"
WEATHER_PREFERENCE_DWD = "dwd"
WEATHER_PREFERENCE_GENERIC = "generic"
WEATHER_FALLBACK_DEFAULT = "weather.home"

# Feature Toggles @zara
CONF_UPDATE_INTERVAL = "update_interval"
CONF_DIAGNOSTIC = "diagnostic"
CONF_HOURLY = "hourly"
CONF_NOTIFY_STARTUP = "notify_startup"
CONF_NOTIFY_FORECAST = "notify_forecast"
CONF_NOTIFY_LEARNING = "notify_learning"
CONF_NOTIFY_SUCCESSFUL_LEARNING = "notify_successful_learning"
CONF_NOTIFY_FROST = "notify_frost"
CONF_NOTIFY_FOG = "notify_fog"
CONF_NOTIFY_WEATHER_ALERT = "notify_weather_alert"
CONF_NOTIFY_SNOW_COVERED = "notify_snow_covered_panels"
CONF_LEARNING_ENABLED = "learning_enabled"

# Adaptive Forecast @zara
CONF_ADAPTIVE_FORECAST_MODE = "adaptive_forecast_mode"
DEFAULT_ADAPTIVE_FORECAST_MODE = False
ADAPTIVE_DEVIATION_THRESHOLD_PERCENT = 35
ADAPTIVE_DEVIATION_MIN_THRESHOLD_KWH = 0.10
ADAPTIVE_DEVIATION_PERCENT_OF_FORECAST = 0.10
ADAPTIVE_MIN_REMAINING_HOURS = 4
ADAPTIVE_CLOUD_COVER_DIFF_THRESHOLD = 25
ADAPTIVE_CHECK_HOUR = 12
ADAPTIVE_CHECK_MINUTE = 30

# Winter Mode @zara
CONF_WINTER_MODE = "winter_mode"
DEFAULT_WINTER_MODE = True
WINTER_MONTHS = [11, 12, 1, 2]
WINTER_CLOUD_PENALTY_FACTOR = 1.25
WINTER_LOW_SUN_THRESHOLD = 25
WINTER_MIN_BUCKET_SAMPLES = 5

# MPPT Throttling Detection @zara
CONF_ZERO_EXPORT_MODE = "zero_export_mode"
CONF_HAS_BATTERY = "has_battery"
CONF_SOLAR_TO_BATTERY_SENSOR = "solar_to_battery_sensor"
DEFAULT_ZERO_EXPORT_MODE = False
DEFAULT_HAS_BATTERY = False
MPPT_THROTTLE_BATTERY_POWER_THRESHOLD = 50.0
MPPT_THROTTLE_PRODUCTION_RATIO = 0.5
MPPT_CLEAR_SKY_GHI_THRESHOLD = 400
MPPT_CLEAR_SKY_CLOUDS_MAX = 30
THROTTLE_REASON_FULL_BATTERY = "full_battery_zero_export"
THROTTLE_REASON_ZERO_EXPORT = "zero_export_limited"

# ML Configuration @zara
CONF_ML_ALGORITHM = "ml_algorithm"
CONF_ENABLE_TINY_LSTM = "enable_tiny_lstm"
DEFAULT_ML_ALGORITHM = "auto"
DEFAULT_ENABLE_TINY_LSTM = True

# Weather Expert API Keys @zara
CONF_PIRATE_WEATHER_API_KEY = "pirate_weather_api_key"

# Defaults @zara
DEFAULT_SOLAR_CAPACITY = 5.0
UPDATE_INTERVAL = timedelta(minutes=60)

# Production Limits @zara
PEAK_POWER_UNIT = "kW"
MAX_HOURLY_PRODUCTION_FACTOR = 1.0
HOURLY_PRODUCTION_SAFETY_MARGIN = 1.2
DEFAULT_MAX_HOURLY_KWH = 3.0

# Daily Tasks @zara
DAILY_UPDATE_HOUR = 6
DAILY_VERIFICATION_HOUR = 21

# Directory Structure @zara
BASE_DATA_DIR = f"/config/{DOMAIN}"
ML_DIR = "ml"
STATS_DIR = "stats"
DATA_DIR = "data"
IMPORTS_DIR = "imports"
EXPORTS_DIR = "exports"
BACKUPS_DIR = "backups"
ASSETS_DIR = "assets"
DOCS_DIR = "docs"
EXPORTS_REPORTS_DIR = "reports"
EXPORTS_PICTURES_DIR = "pictures"
EXPORTS_STATISTICS_DIR = "statistics"
BACKUPS_AUTO_DIR = "auto"
BACKUPS_MANUAL_DIR = "manual"

# ML Training @zara
DATA_VERSION = "1.0"
MIN_TRAINING_DATA_POINTS = 50
BACKUP_RETENTION_DAYS = 30
MAX_BACKUP_FILES = 10

# ML Model @zara
ML_MODEL_VERSION = "1.0"
MODEL_ACCURACY_THRESHOLD = 0.75
PREDICTION_CONFIDENCE_THRESHOLD = 0.6
CORRECTION_FACTOR_MIN = 0.5
CORRECTION_FACTOR_MAX = 1.5

# Weather Validation @zara
WEATHER_TEMP_MIN = -50.0
WEATHER_TEMP_MAX = 60.0
WEATHER_HUMIDITY_MIN = 0.0
WEATHER_HUMIDITY_MAX = 100.0
WEATHER_CLOUDS_MIN = 0.0
WEATHER_CLOUDS_MAX = 100.0

# Circuit Breaker @zara
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60
CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 3

# Sensor Attributes @zara
ATTR_FORECAST_TODAY = "forecast_today"
ATTR_FORECAST_TOMORROW = "forecast_tomorrow"
ATTR_WEATHER_CONDITION = "weather_condition"
ATTR_LEARNING_STATUS = "learning_status"
ATTR_LAST_LEARNING = "last_learning"
ATTR_MODEL_ACCURACY = "model_accuracy"
ATTR_WEATHER_SOURCE = "weather_source"
ATTR_RETRY_COUNT = "retry_count"
ATTR_FALLBACK_ACTIVE = "fallback_active"

# AI Services @zara
SERVICE_RETRAIN_AI_MODEL = "retrain_ai_model"
SERVICE_RESET_AI_MODEL = "reset_ai_model"
SERVICE_RUN_GRID_SEARCH = "run_grid_search"
SERVICE_ANALYZE_FEATURE_IMPORTANCE = "analyze_feature_importance"

# Grid Search @zara
CONF_GRID_SEARCH_ENABLED = "grid_search_enabled"
CONF_GRID_SEARCH_INTERVAL_DAYS = "grid_search_interval_days"
DEFAULT_GRID_SEARCH_ENABLED = False
DEFAULT_GRID_SEARCH_INTERVAL_DAYS = 30
MIN_GRID_SEARCH_INTERVAL_DAYS = 14

# Emergency Services @zara
SERVICE_RUN_ALL_DAY_END_TASKS = "run_all_day_end_tasks"

# Testing Services @zara
SERVICE_TEST_MORNING_ROUTINE = "test_morning_routine"
SERVICE_TEST_RETROSPECTIVE_FORECAST = "test_retrospective_forecast"

# Astronomy Services @zara
SERVICE_BUILD_ASTRONOMY_CACHE = "build_astronomy_cache"
SERVICE_REFRESH_CACHE_TODAY = "refresh_cache_today"

# Weather Services @zara
SERVICE_RUN_WEATHER_CORRECTION = "run_weather_correction"
SERVICE_REFRESH_OPEN_METEO_CACHE = "refresh_open_meteo_cache"
SERVICE_REFRESH_MULTI_WEATHER = "refresh_multi_weather"

# Notification Services @zara
SERVICE_SEND_DAILY_BRIEFING = "send_daily_briefing"

# Shadow Detection Services @zara V16.2
SERVICE_BACKFILL_SHADOW_DETECTION = "backfill_shadow_detection"


# Icons @zara
ICON_SOLAR = "mdi:solar-power"
ICON_FORECAST = "mdi:weather-sunny"
ICON_LEARNING = "mdi:brain"
ICON_SHADOW_NONE = "mdi:weather-sunny"
ICON_SHADOW_LIGHT = "mdi:weather-partly-cloudy"
ICON_SHADOW_MODERATE = "mdi:weather-cloudy"
ICON_SHADOW_HEAVY = "mdi:weather-cloudy-alert"
ICON_SHADOW_ANALYSIS = "mdi:weather-sunset"

# Units @zara
UNIT_KWH = "kWh"
UNIT_PERCENTAGE = "%"

# Logging @zara
LOGGER_NAME = f"custom_components.{DOMAIN}"

# Update Intervals @zara
COORDINATOR_UPDATE_INTERVAL = timedelta(minutes=60)
LEARNING_UPDATE_INTERVAL = timedelta(hours=1)
CLEANUP_INTERVAL = timedelta(days=1)

# Threading @zara
MAX_CONCURRENT_OPERATIONS = 3
THREAD_POOL_SIZE = 2

# Validation Limits @zara
MIN_SOLAR_CAPACITY = 0.1
MAX_SOLAR_CAPACITY = 1000.0

# Status Values @zara
STATUS_INITIALIZING = "initializing"
STATUS_READY = "ready"
STATUS_LEARNING = "learning"
STATUS_FORECASTING = "forecasting"
STATUS_ERROR = "error"
STATUS_OFFLINE = "offline"

# Device Info @zara
DEVICE_MANUFACTURER = "Zara-Toorox"

# Production Hours @zara
SUN_BUFFER_HOURS = 1.5
FALLBACK_PRODUCTION_START_HOUR = 5
FALLBACK_PRODUCTION_END_HOUR = 21

# ============================================
# Coordinator Data Keys @zara
# ============================================
DATA_KEY_FORECAST_TODAY = "forecast_today"
DATA_KEY_FORECAST_TOMORROW = "forecast_tomorrow"
DATA_KEY_FORECAST_DAY_AFTER = "forecast_day_after_tomorrow"
DATA_KEY_HOURLY_FORECAST = "hourly_forecast"
DATA_KEY_CURRENT_WEATHER = "current_weather"
DATA_KEY_EXTERNAL_SENSORS = "external_sensors"
DATA_KEY_PRODUCTION_TIME = "production_time"
DATA_KEY_PEAK_TODAY = "peak_today"
DATA_KEY_YIELD_TODAY = "yield_today"
DATA_KEY_EXPECTED_DAILY_PRODUCTION = "expected_daily_production"
DATA_KEY_STATISTICS = "statistics"

# ============================================
# Production Time Sub-Keys @zara
# ============================================
PROD_TIME_ACTIVE = "active"
PROD_TIME_DURATION_SECONDS = "duration_seconds"
PROD_TIME_START_TIME = "start_time"
PROD_TIME_END_TIME = "end_time"

# ============================================
# Peak Today Sub-Keys @zara
# ============================================
PEAK_TODAY_POWER_W = "power_w"
PEAK_TODAY_AT = "at"

# ============================================
# Yield Today Sub-Keys @zara
# ============================================
YIELD_TODAY_KWH = "kwh"
YIELD_TODAY_SENSOR = "sensor"

# ============================================
# Statistics Sub-Keys @zara
# ============================================
STATS_ALL_TIME_PEAK = "all_time_peak"
STATS_CURRENT_MONTH = "current_month"
STATS_CURRENT_WEEK = "current_week"
STATS_LAST_7_DAYS = "last_7_days"
STATS_LAST_30_DAYS = "last_30_days"
STATS_AVG_ACCURACY = "avg_accuracy"
STATS_YIELD_KWH = "yield_kwh"
STATS_AVG_YIELD_KWH = "avg_yield_kwh"
STATS_CONSUMPTION_KWH = "consumption_kwh"

# ============================================
# Hourly Predictions Cache Keys @zara
# ============================================
CACHE_HOURLY_PREDICTIONS = "_hourly_predictions_cache"
CACHE_PREDICTIONS = "predictions"
CACHE_BEST_HOUR_TODAY = "best_hour_today"

# ============================================
# Prediction Entry Keys @zara
# ============================================
PRED_TARGET_DATE = "target_date"
PRED_TARGET_HOUR = "target_hour"
PRED_PREDICTION_KWH = "prediction_kwh"
PRED_PREDICTED_KWH = "predicted_kwh"

# ============================================
# External Sensors Keys @zara
# ============================================
EXT_SENSOR_SOLAR_YIELD_TODAY = "solar_yield_today"
EXT_SENSOR_TOTAL_CONSUMPTION_TODAY = "total_consumption_today"
EXT_SENSOR_GRID_IMPORT_TODAY = "grid_import_today"
EXT_SENSOR_GRID_EXPORT_TODAY = "grid_export_today"

# ============================================
# Forecast Source Keys @zara
# ============================================
FORECAST_KEY_TODAY = "today"
FORECAST_KEY_TOMORROW = "tomorrow"
FORECAST_KEY_DAY_AFTER = "day_after_tomorrow"
FORECAST_KEY_HOURLY = "hourly"
FORECAST_KEY_METHOD = "method"
