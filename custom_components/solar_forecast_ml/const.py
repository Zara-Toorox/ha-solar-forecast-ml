# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from datetime import timedelta

from homeassistant.const import Platform

DOMAIN = "solar_forecast_ml"
NAME = "Solar Forecast ML"
VERSION = "14.0.0"
RELEASE_VERSION = "14.0.0"
RELEASE_NAME = "Sarpeidon"
SOFTWARE_VERSION = "14.0.0"
INTEGRATION_MODEL = "V14.0.0"
AI_VERSION = "14.0.0"

PLATFORMS = [Platform.SENSOR]

CONF_WEATHER_ENTITY = "weather_entity"
CONF_POWER_ENTITY = "power_entity"
CONF_SOLAR_YIELD_TODAY = "solar_yield_today"
CONF_SOLAR_CAPACITY = "solar_capacity"

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

# Panel Groups Configuration
CONF_PANEL_GROUPS = "panel_groups"
CONF_PANEL_GROUP_POWER = "power_wp"
CONF_PANEL_GROUP_AZIMUTH = "azimuth"
CONF_PANEL_GROUP_TILT = "tilt"
CONF_PANEL_GROUP_NAME = "name"
CONF_PANEL_GROUP_ENERGY_SENSOR = "energy_sensor"

# Inverter Clipping Configuration
CONF_INVERTER_MAX_POWER = "inverter_max_power"
DEFAULT_INVERTER_MAX_POWER = 0.0  # 0 = disabled (no clipping)
INVERTER_CLIPPING_THRESHOLD = 0.95  # 95% of max = considered clipped

# Default Panel Group Values
DEFAULT_PANEL_AZIMUTH = 180  # South
DEFAULT_PANEL_TILT = 30  # 30 degrees

EXTERNAL_SENSOR_MAPPING = {
    "temperature": CONF_TEMP_SENSOR,
    "humidity": CONF_HUMIDITY_SENSOR,
    "wind_speed": CONF_WIND_SENSOR,
    "rain": CONF_RAIN_SENSOR,
    "pressure": CONF_PRESSURE_SENSOR,
    "solar_radiation": CONF_SOLAR_RADIATION_SENSOR,
    "lux": CONF_LUX_SENSOR,
}

CONF_WEATHER_PREFERENCE = "weather_preference"
CONF_FALLBACK_ENTITY = "fallback_weather_entity"
WEATHER_PREFERENCE_DWD = "dwd"
WEATHER_PREFERENCE_GENERIC = "generic"
WEATHER_FALLBACK_DEFAULT = "weather.home"

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

# Adaptive Forecast Mode Configuration
CONF_ADAPTIVE_FORECAST_MODE = "adaptive_forecast_mode"
DEFAULT_ADAPTIVE_FORECAST_MODE = False  # Default OFF - User muss explizit aktivieren

# Adaptive Forecast Thresholds (nicht vom User konfigurierbar - SFML entscheidet autonom)
ADAPTIVE_DEVIATION_THRESHOLD_PERCENT = 35      # Mindestabweichung in Prozent
ADAPTIVE_DEVIATION_MIN_THRESHOLD_KWH = 0.10    # Absolutes Minimum (schützt vor Sensor-Rauschen)
ADAPTIVE_DEVIATION_PERCENT_OF_FORECAST = 0.10  # 10% der Tagesprognose als dynamische Schwelle
ADAPTIVE_MIN_REMAINING_HOURS = 4           # Mindestens 4h bis Sunset
ADAPTIVE_CLOUD_COVER_DIFF_THRESHOLD = 25   # Prozentpunkte Cloud-Cover Differenz
ADAPTIVE_CHECK_HOUR = 12                   # Uhrzeit für den Check (12:30)
ADAPTIVE_CHECK_MINUTE = 30

# V13.1: Winter Mode Configuration
# Enables enhanced algorithms for low sun angles (Nov-Feb)
CONF_WINTER_MODE = "winter_mode"
DEFAULT_WINTER_MODE = True  # Default ON - aktiviert automatisch für bessere Winter-Prognosen

# Winter Mode Constants (nicht vom User konfigurierbar)
WINTER_MONTHS = [11, 12, 1, 2]  # November bis Februar
WINTER_CLOUD_PENALTY_FACTOR = 1.25  # Clouds wirken 25% stärker im Winter
WINTER_LOW_SUN_THRESHOLD = 25  # Grad - unter diesem Winkel greift Winter-Korrektur
WINTER_MIN_BUCKET_SAMPLES = 5  # Mindest-Samples für Winter-Bucket-Faktoren

# V13.2: Learning Data Backup Protection
# Protects learning data from loss during HA backup restore
CONF_LEARNING_BACKUP_PROTECTION = "learning_backup_protection"
DEFAULT_LEARNING_BACKUP_PROTECTION = True  # Default ON - protects by default

# Share Backup Configuration (not user-configurable)
SHARE_BACKUP_DIR = "/share/solar_forecast_ml_backup"
SHARE_BACKUP_FRESHNESS_HOURS = 72  # 3 days threshold for stale data detection

# =============================================================================
# V13.2.0: MPPT Throttling Detection
# Erkennt gedrosselte Produktion bei Nulleinspeisung / vollem Akku
# Verhindert ML-Training aus kontaminierten Daten
# =============================================================================

# Config Options
CONF_ZERO_EXPORT_MODE = "zero_export_mode"  # Toggle: Nulleinspeisung aktiv
CONF_HAS_BATTERY = "has_battery"  # Toggle: Akku im System vorhanden
CONF_SOLAR_TO_BATTERY_SENSOR = "solar_to_battery_sensor"  # Power Sensor: PV → Akku

# Default Values
DEFAULT_ZERO_EXPORT_MODE = False  # User muss explizit aktivieren
DEFAULT_HAS_BATTERY = False  # User muss explizit aktivieren

# Detection Thresholds (nicht vom User konfigurierbar)
MPPT_THROTTLE_BATTERY_POWER_THRESHOLD = 50.0  # W - unter 50W gilt als "Akku voll"
MPPT_THROTTLE_PRODUCTION_RATIO = 0.5  # actual/physics < 50% = gedrosselt
MPPT_CLEAR_SKY_GHI_THRESHOLD = 400  # W/m² - gute Sonnenbedingungen
MPPT_CLEAR_SKY_CLOUDS_MAX = 30  # % - maximal 30% Wolken für "klaren Himmel"

# Throttle Reasons (für Logging und hourly_predictions.json)
THROTTLE_REASON_FULL_BATTERY = "full_battery_zero_export"
THROTTLE_REASON_ZERO_EXPORT = "zero_export_limited"

CONF_ML_ALGORITHM = "ml_algorithm"
CONF_ENABLE_TINY_LSTM = "enable_tiny_lstm"
DEFAULT_ML_ALGORITHM = "auto"
DEFAULT_ENABLE_TINY_LSTM = True

# Weather Expert API Keys
CONF_PIRATE_WEATHER_API_KEY = "pirate_weather_api_key"

DEFAULT_SOLAR_CAPACITY = 5.0
UPDATE_INTERVAL = timedelta(minutes=60)

PEAK_POWER_UNIT = "kW"
MAX_HOURLY_PRODUCTION_FACTOR = 1.0
HOURLY_PRODUCTION_SAFETY_MARGIN = 1.2
DEFAULT_MAX_HOURLY_KWH = 3.0

DAILY_UPDATE_HOUR = 6
DAILY_VERIFICATION_HOUR = 21

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

LEARNED_WEIGHTS_FILE = "learned_weights.json"
SEASONAL_FILE = "seasonal.json"
DNI_TRACKER_FILE = "dni_tracker.json"
HOURLY_PROFILE_FILE = "hourly_profile.json"
MODEL_STATE_FILE = "model_state.json"

DAILY_FORECASTS_FILE = "daily_forecasts.json"

COORDINATOR_STATE_FILE = "coordinator_state.json"
PRODUCTION_TIME_STATE_FILE = "production_time_state.json"

DATA_VERSION = "1.0"
MIN_TRAINING_DATA_POINTS = 50
BACKUP_RETENTION_DAYS = 30
MAX_BACKUP_FILES = 10

ML_MODEL_VERSION = "1.0"
MODEL_ACCURACY_THRESHOLD = 0.75
PREDICTION_CONFIDENCE_THRESHOLD = 0.6
CORRECTION_FACTOR_MIN = 0.5
CORRECTION_FACTOR_MAX = 1.5

WEATHER_TEMP_MIN = -50.0
WEATHER_TEMP_MAX = 60.0
WEATHER_HUMIDITY_MIN = 0.0
WEATHER_HUMIDITY_MAX = 100.0
WEATHER_CLOUDS_MIN = 0.0
WEATHER_CLOUDS_MAX = 100.0

CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60
CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 3

ATTR_FORECAST_TODAY = "forecast_today"
ATTR_FORECAST_TOMORROW = "forecast_tomorrow"
ATTR_WEATHER_CONDITION = "weather_condition"
ATTR_LEARNING_STATUS = "learning_status"
ATTR_LAST_LEARNING = "last_learning"
ATTR_MODEL_ACCURACY = "model_accuracy"
ATTR_WEATHER_SOURCE = "weather_source"
ATTR_RETRY_COUNT = "retry_count"
ATTR_FALLBACK_ACTIVE = "fallback_active"

# AI Services
SERVICE_RETRAIN_AI_MODEL = "retrain_ai_model"
SERVICE_RESET_AI_MODEL = "reset_ai_model"
SERVICE_RUN_GRID_SEARCH = "run_grid_search"
SERVICE_ANALYZE_FEATURE_IMPORTANCE = "analyze_feature_importance"

# Grid-Search Configuration
CONF_GRID_SEARCH_ENABLED = "grid_search_enabled"
CONF_GRID_SEARCH_INTERVAL_DAYS = "grid_search_interval_days"
DEFAULT_GRID_SEARCH_ENABLED = False  # Opt-in, not default
DEFAULT_GRID_SEARCH_INTERVAL_DAYS = 30  # Monthly
MIN_GRID_SEARCH_INTERVAL_DAYS = 14  # Not more often than every 2 weeks

# Emergency Services
SERVICE_RUN_ALL_DAY_END_TASKS = "run_all_day_end_tasks"

# Testing Services
SERVICE_TEST_MORNING_ROUTINE = "test_morning_routine"
SERVICE_TEST_RETROSPECTIVE_FORECAST = "test_retrospective_forecast"

# Astronomy Services
SERVICE_BUILD_ASTRONOMY_CACHE = "build_astronomy_cache"
SERVICE_REFRESH_CACHE_TODAY = "refresh_cache_today"

# Weather Services
SERVICE_RUN_WEATHER_CORRECTION = "run_weather_correction"
SERVICE_REFRESH_OPEN_METEO_CACHE = "refresh_open_meteo_cache"
SERVICE_REFRESH_MULTI_WEATHER = "refresh_multi_weather"

# Notification Services
SERVICE_SEND_DAILY_BRIEFING = "send_daily_briefing"

# Installation Services
SERVICE_INSTALL_EXTRA_FEATURES = "install_extra_features"

# Data Management Services
SERVICE_BORG_ASSIMILATION_REVERSE = "borg_assimilation_reverse"

# Physics Calibration Services
SERVICE_RESCUE_CALIBRATION = "rescue_calibration"
SERVICE_REPAIR_CALIBRATION = "repair_calibration"
SERVICE_WINTER_BUCKET_CORRECTION = "winter_bucket_correction_v13"

ICON_SOLAR = "mdi:solar-power"
ICON_FORECAST = "mdi:weather-sunny"
ICON_LEARNING = "mdi:brain"
ICON_SHADOW_NONE = "mdi:weather-sunny"
ICON_SHADOW_LIGHT = "mdi:weather-partly-cloudy"
ICON_SHADOW_MODERATE = "mdi:weather-cloudy"
ICON_SHADOW_HEAVY = "mdi:weather-cloudy-alert"
ICON_SHADOW_ANALYSIS = "mdi:weather-sunset"

UNIT_KWH = "kWh"
UNIT_PERCENTAGE = "%"

LOGGER_NAME = f"custom_components.{DOMAIN}"

FILE_ENCODING = "utf-8"
JSON_INDENT = 2
ATOMIC_WRITE_ENABLED = True

COORDINATOR_UPDATE_INTERVAL = timedelta(minutes=60)
LEARNING_UPDATE_INTERVAL = timedelta(hours=1)
CLEANUP_INTERVAL = timedelta(days=1)

MAX_CONCURRENT_OPERATIONS = 3
THREAD_POOL_SIZE = 2

MIN_SOLAR_CAPACITY = 0.1
MAX_SOLAR_CAPACITY = 1000.0

STATUS_INITIALIZING = "initializing"
STATUS_READY = "ready"
STATUS_LEARNING = "learning"
STATUS_FORECASTING = "forecasting"
STATUS_ERROR = "error"
STATUS_OFFLINE = "offline"

DEVICE_MANUFACTURER = "Zara-Toorox"

SUN_BUFFER_HOURS = 1.5
FALLBACK_PRODUCTION_START_HOUR = 5
FALLBACK_PRODUCTION_END_HOUR = 21

