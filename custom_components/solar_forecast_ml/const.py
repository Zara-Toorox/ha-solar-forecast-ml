"""
Konstanten fÃ¼r die Solar Forecast ML Integration.
âœ“ Version 4.0 - Bereinigt ohne OpenWeatherMap
âœ“ Intelligente DWD-Retry Logic implementiert

Copyright (C) 2025 Zara-Toorox
"""
from datetime import timedelta
from homeassistant.const import Platform

# Integration info # von Zara
DOMAIN = "solar_forecast_ml"
NAME = "Solar Forecast ML"
VERSION = "4.0"

# Versionierung - Zentral verwaltet - von Zara
RELEASE_VERSION = "5.0"  # Hauptversion der Integration - von Zara
SOFTWARE_VERSION = "4.0"  # Software-Version - von Zara
ML_VERSION = "4.0"  # ML-Modell Version - von Zara
INTEGRATION_MODEL = "v5.0"  # Display-Version fÃ¼r Device Info - von Zara

# Platforms # von Zara
PLATFORMS = [Platform.SENSOR, Platform.BUTTON]

# Configuration keys # von Zara
CONF_CURRENT_POWER = "current_power"
CONF_FORECAST_SOLAR = "forecast_solar"
CONF_LUX_SENSOR = "lux_sensor"
CONF_PLANT_KWP = "plant_kwp"
CONF_POWER_ENTITY = "power_entity"
CONF_RAIN_SENSOR = "rain_sensor"
CONF_SOLAR_YIELD_TODAY = "solar_yield_today"  # Täglicher Solar-Ertrag in kWh - von Zara
CONF_TEMP_SENSOR = "temp_sensor"
CONF_TOTAL_CONSUMPTION_TODAY = "total_consumption_today"
CONF_UV_SENSOR = "uv_sensor"
CONF_WEATHER_ENTITY = "weather_entity"
CONF_WIND_SENSOR = "wind_sensor"

# âœ“ NEU: Weather Preference & Retry Configuration # von Zara
CONF_WEATHER_PREFERENCE = "weather_preference"
CONF_FALLBACK_ENTITY = "fallback_weather_entity"

WEATHER_PREFERENCE_DWD = "dwd"
WEATHER_PREFERENCE_GENERIC = "generic"

# Retry-Delays: sofort, 1min, 2min, 3min, 4min # von Zara
WEATHER_RETRY_DELAYS = [0, 60, 120, 180, 240]
WEATHER_MAX_RETRY_TIME = 300  # 5 Minuten Maximum # von Zara
WEATHER_FALLBACK_DEFAULT = "weather.home"

# Options keys # von Zara
CONF_UPDATE_INTERVAL = "update_interval"
CONF_DIAGNOSTIC = "diagnostic"
CONF_HOURLY = "hourly"
CONF_NOTIFY_STARTUP = "notify_startup"
CONF_NOTIFY_FORECAST = "notify_forecast"
CONF_NOTIFY_LEARNING = "notify_learning"
CONF_NOTIFY_SUCCESSFUL_LEARNING = "notify_successful_learning"

# Legacy config keys (for backward compatibility) # von Zara
CONF_SOLAR_CAPACITY = "solar_capacity"
CONF_LEARNING_ENABLED = "learning_enabled"
CONF_HOURLY_LEARNING_ENABLED = "hourly_learning_enabled"

# âœ“ NEU: Panel Configuration # von Zara
CONF_PANEL_EFFICIENCY = "panel_efficiency"
CONF_AZIMUTH = "azimuth"
CONF_TILT = "tilt"

# Default values # von Zara
DEFAULT_SOLAR_CAPACITY = 5.0
DEFAULT_PANEL_EFFICIENCY = 0.18  # 18% # von Zara
DEFAULT_AZIMUTH = 180.0  # SÃ¼d # von Zara
DEFAULT_TILT = 30.0  # 30Â° # von Zara
DEFAULT_LEARNING_ENABLED = True
DEFAULT_HOURLY_LEARNING_ENABLED = False
UPDATE_INTERVAL = timedelta(minutes=30)

# File names # von Zara
PREDICTION_HISTORY_FILE = "prediction_history.json"
LEARNED_WEIGHTS_FILE = "learned_weights.json"
HOURLY_PROFILE_FILE = "hourly_profile.json"
MODEL_STATE_FILE = "model_state.json"
ERROR_LOG_FILE = "error_log.json"

# Data management # von Zara
DATA_VERSION = "1.0"
MAX_PREDICTION_HISTORY = 365
MIN_TRAINING_DATA_POINTS = 7
BACKUP_RETENTION_DAYS = 30
MAX_BACKUP_FILES = 10

# ML Constants # von Zara
MODEL_ACCURACY_THRESHOLD = 0.75
ML_MODEL_VERSION = "1.0"
MIN_SAMPLES_FOR_TRAINING = 10
MAX_TRAINING_SAMPLES = 1000
FEATURE_VECTOR_SIZE = 7
PREDICTION_CONFIDENCE_THRESHOLD = 0.6

# ML Training Parameters # von Zara
LEARNING_RATE = 0.01
REGULARIZATION_FACTOR = 0.1
CONVERGENCE_TOLERANCE = 1e-6
MAX_TRAINING_ITERATIONS = 1000
CROSS_VALIDATION_FOLDS = 5

# ML Feature Engineering # von Zara
WEATHER_FEATURE_WEIGHTS = {
    "temperature": 0.25,
    "humidity": 0.15,
    "cloudiness": 0.35,
    "wind_speed": 0.10,
    "pressure": 0.10,
    "hour_of_day": 0.05
}

# Weather Data Validation Ranges # von Zara
WEATHER_TEMP_MIN = -50.0  # Â°C # von Zara
WEATHER_TEMP_MAX = 60.0   # Â°C # von Zara
WEATHER_HUMIDITY_MIN = 0.0
WEATHER_HUMIDITY_MAX = 100.0
WEATHER_CLOUDS_MIN = 0.0
WEATHER_CLOUDS_MAX = 100.0
WEATHER_WIND_MAX = 150.0  # km/h # von Zara
WEATHER_PRESSURE_MIN = 800.0   # hPa # von Zara
WEATHER_PRESSURE_MAX = 1200.0  # hPa # von Zara

# Error Handling Constants # von Zara
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60
CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 3
MAX_FAILURES_PER_HOUR = 20

# Retry Configuration # von Zara
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 1.0
DEFAULT_RETRY_MAX_DELAY = 60.0
DEFAULT_RETRY_EXPONENTIAL_BASE = 2.0

# ML Constants # von Zara
DEFAULT_WEATHER_WEIGHTS = {
    "temperature": 0.3,
    "clouds": 0.4,
    "humidity": 0.1,
    "wind_speed": 0.1,
    "pressure": 0.1
}

DEFAULT_SEASONAL_FACTORS = {
    "spring": 1.0,
    "summer": 1.0,
    "autumn": 1.0,
    "winter": 1.0
}

# Sensor attributes # von Zara
ATTR_FORECAST_TODAY = "forecast_today"
ATTR_FORECAST_TOMORROW = "forecast_tomorrow"
ATTR_WEATHER_CONDITION = "weather_condition"
ATTR_LEARNING_STATUS = "learning_status"
ATTR_LAST_LEARNING = "last_learning"
ATTR_MODEL_ACCURACY = "model_accuracy"
ATTR_WEATHER_SOURCE = "weather_source"  # âœ“ NEU fÃ¼r State-Machine # von Zara
ATTR_RETRY_COUNT = "retry_count"  # âœ“ NEU fÃ¼r State-Machine # von Zara
ATTR_FALLBACK_ACTIVE = "fallback_active"  # âœ“ NEU fÃ¼r State-Machine # von Zara

# Button identifiers # von Zara
BUTTON_MANUAL_FORECAST = "manual_forecast"
BUTTON_RETRAIN_MODEL = "retrain_model"
BUTTON_RESET_DATA = "reset_data"

# Service names # von Zara
SERVICE_MANUAL_FORECAST = "manual_forecast"
SERVICE_RETRAIN_MODEL = "retrain_model"
SERVICE_RESET_LEARNING_DATA = "reset_learning_data"

# Weather entity types # von Zara
WEATHER_TYPE_DWD = "dwd"
WEATHER_TYPE_GENERIC = "generic"

# Supported weather entities (priorisiert) # von Zara
PREFERRED_WEATHER_ENTITIES = [
    "weather.dwd_weather",       # PrioritÃ¤t 1: DWD (Deutschland) # von Zara
    "weather.met",               # PrioritÃ¤t 2: Met.no (Europa) # von Zara
    "weather.forecast_home",     # PrioritÃ¤t 3: Forecast.Solar # von Zara
    "weather.home",              # PrioritÃ¤t 4: Generic HA # von Zara
    "weather.hourly",            # PrioritÃ¤t 5: Hourly Fallback # von Zara
]

# Icons # von Zara
ICON_SOLAR = "mdi:solar-power"
ICON_FORECAST = "mdi:weather-sunny"
ICON_LEARNING = "mdi:brain"
ICON_BUTTON = "mdi:play-circle"

# Units # von Zara
UNIT_KWH = "kWh"
UNIT_PERCENTAGE = "%"

# Error codes # von Zara
ERROR_WEATHER_UNAVAILABLE = "weather_unavailable"
ERROR_DATA_CORRUPTION = "data_corruption"
ERROR_ML_MODEL_FAILED = "ml_model_failed"
ERROR_INSUFFICIENT_DATA = "insufficient_data"

# Helper Constants # von Zara
DATA_DIR = "/config/solar_forecast_ml"
DEFAULT_BASE_CAPACITY = 2.13
HISTORY_FILE = f"{DATA_DIR}/prediction_history.json"
HOURLY_PROFILE_FILE = f"{DATA_DIR}/hourly_profile.json"
OLD_HISTORY_FILE = "/config/custom_components/solar_forecast_ml/prediction_history.json"
OLD_HOURLY_PROFILE_FILE = "/config/custom_components/solar_forecast_ml/hourly_profile.json"
OLD_WEIGHTS_FILE = "/config/custom_components/solar_forecast_ml/learned_weights.json"
WEIGHTS_FILE = f"{DATA_DIR}/learned_weights.json"

# Logging # von Zara
LOGGER_NAME = f"custom_components.{DOMAIN}"

# Update intervals for different components # von Zara
COORDINATOR_UPDATE_INTERVAL = timedelta(minutes=30)
LEARNING_UPDATE_INTERVAL = timedelta(hours=1)
CLEANUP_INTERVAL = timedelta(days=1)

# Retry settings # von Zara
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds # von Zara
EXPONENTIAL_BACKOFF = True

# File I/O settings # von Zara
FILE_ENCODING = "utf-8"
JSON_INDENT = 2
ATOMIC_WRITE_ENABLED = True

# Performance settings # von Zara
MAX_CONCURRENT_OPERATIONS = 3
THREAD_POOL_SIZE = 2
MEMORY_CACHE_SIZE = 100

# Validation limits # von Zara
MIN_SOLAR_CAPACITY = 0.1
MAX_SOLAR_CAPACITY = 100.0
MIN_FORECAST_HOURS = 1
MAX_FORECAST_HOURS = 72

# Weather forecast limits # von Zara
MIN_TEMPERATURE = -50
MAX_TEMPERATURE = 60
MIN_HUMIDITY = 0
MAX_HUMIDITY = 100
MIN_CLOUDS = 0
MAX_CLOUDS = 100

# Learning parameters # von Zara
MIN_LEARNING_SAMPLES = 7
MAX_LEARNING_SAMPLES = 365
LEARNING_DECAY_FACTOR = 0.95

# Forecast accuracy thresholds # von Zara
ACCURACY_EXCELLENT = 95
ACCURACY_GOOD = 85
ACCURACY_FAIR = 70
ACCURACY_POOR = 50

# Status messages # von Zara
STATUS_INITIALIZING = "initializing"
STATUS_READY = "ready"
STATUS_LEARNING = "learning"
STATUS_FORECASTING = "forecasting"
STATUS_ERROR = "error"
STATUS_OFFLINE = "offline"

# Device info # von Zara
DEVICE_MANUFACTURER = "Zara-Toorox"
DEVICE_MODEL = "Solar Forecast ML"
DEVICE_SW_VERSION = VERSION

# ML Prediction Bounds # von Zara
PREDICTION_MIN_VALUE = 0.0
PREDICTION_MAX_VALUE = 50.0  # kWh per day maximum # von Zara
CORRECTION_FACTOR_MIN = 0.1
CORRECTION_FACTOR_MAX = 5.0

# Data Quality Thresholds # von Zara
DATA_QUALITY_MIN_SAMPLES = 5
DATA_QUALITY_MAX_AGE_DAYS = 90
DATA_QUALITY_MIN_ACCURACY = 0.3

# Health Check Constants # von Zara
HEALTH_CHECK_INTERVAL = timedelta(minutes=15)
SERVICE_HEALTH_TIMEOUT = 10  # seconds # von Zara
PERFORMANCE_DEGRADED_THRESHOLD = 0.7

# Backup Constants # von Zara
BACKUP_SCHEDULE_HOUR = 3  # 3 AM daily backup # von Zara
BACKUP_BEFORE_TRAINING = True
BACKUP_MAX_AGE_DAYS = 30

# Notification Constants # von Zara
NOTIFICATION_CRITICAL_ERRORS = True
NOTIFICATION_MODEL_RETRAINED = True
NOTIFICATION_DATA_CORRUPTION = True
NOTIFICATION_BACKUP_FAILED = True

# Feature Engineering Constants # von Zara
HOUR_NORMALIZATION_FACTOR = 24
SEASONAL_MONTH_MAPPING = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring", 
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn"
}

# Production Time Constants # von Zara
SUNRISE_HOUR = 6
SUNSET_HOUR = 19
PEAK_PRODUCTION_START = 10
PEAK_PRODUCTION_END = 14

# System Limits # von Zara
MAX_CONCURRENT_PREDICTIONS = 5
MAX_TRAINING_TIME_MINUTES = 30
MAX_MEMORY_USAGE_MB = 500
