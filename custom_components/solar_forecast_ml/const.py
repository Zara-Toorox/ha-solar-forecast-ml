"""
Constants for the Solar Forecast ML Integration.

Copyright (C) 2025 Zara-Toorox

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""
from datetime import timedelta
from homeassistant.const import Platform

DOMAIN = "solar_forecast_ml"
NAME = "Solar Forecast ML"
# [Version Update] Set version numbers to 6.2.1
VERSION = "6.8.1"
RELEASE_VERSION = "6.8.1"
SOFTWARE_VERSION = "6.8.1"
INTEGRATION_MODEL = "v6.8.1" # Model string might differ based on ML changes
ML_VERSION = "8.0.0" # Keep ML Model version unless model structure changed significantly

PLATFORMS = [Platform.SENSOR, Platform.BUTTON]

# --- Core Configuration Keys ---
CONF_WEATHER_ENTITY = "weather_entity"
CONF_POWER_ENTITY = "power_entity"
CONF_SOLAR_YIELD_TODAY = "solar_yield_today"
CONF_SOLAR_CAPACITY = "solar_capacity" # Changed from plant_kwp

# --- Optional Configuration Keys ---
CONF_TOTAL_CONSUMPTION_TODAY = "total_consumption_today"
CONF_RAIN_SENSOR = "rain_sensor"
CONF_LUX_SENSOR = "lux_sensor"
CONF_TEMP_SENSOR = "temp_sensor"
CONF_WIND_SENSOR = "wind_sensor"
CONF_UV_SENSOR = "uv_sensor"
CONF_HUMIDITY_SENSOR = "humidity_sensor"

# --- External Sensor Mapping (Single Source of Truth) ---
# Maps internal keys to config entry keys for centralized sensor handling
EXTERNAL_SENSOR_MAPPING = {
    'temperature': CONF_TEMP_SENSOR,
    'humidity': CONF_HUMIDITY_SENSOR,
    'wind_speed': CONF_WIND_SENSOR,
    'rain': CONF_RAIN_SENSOR,
    'uv_index': CONF_UV_SENSOR,
    'lux': CONF_LUX_SENSOR,
}

# --- Removed Configuration Keys (Dead Ends) ---
# CONF_CURRENT_POWER = "current_power" # Removed - Duplicated logic with CONF_POWER_ENTITY
# CONF_FORECAST_SOLAR = "forecast_solar" # Removed - Not used anywhere
# CONF_PLANT_KWP = "plant_kwp" # Removed - Replaced by CONF_SOLAR_CAPACITY

# --- Weather Preference (Keep for potential future use or remove if definitely unused) ---
CONF_WEATHER_PREFERENCE = "weather_preference"
CONF_FALLBACK_ENTITY = "fallback_weather_entity" # Check if still used
WEATHER_PREFERENCE_DWD = "dwd"
WEATHER_PREFERENCE_GENERIC = "generic"
WEATHER_FALLBACK_DEFAULT = "weather.home" # Default fallback weather entity

# --- Options Flow Keys ---
CONF_UPDATE_INTERVAL = "update_interval"
CONF_DIAGNOSTIC = "diagnostic"
CONF_HOURLY = "hourly" # Renamed from enable_hourly
CONF_NOTIFY_STARTUP = "notify_startup"
CONF_NOTIFY_FORECAST = "notify_forecast"
CONF_NOTIFY_LEARNING = "notify_learning"
CONF_NOTIFY_SUCCESSFUL_LEARNING = "notify_successful_learning"
CONF_LEARNING_ENABLED = "learning_enabled" # <-- WIEDER HINZUGEFÃƒÆ’Ã…â€œGT

# --- Deprecated/Legacy Configuration Keys (Keep for potential migration logic if needed) ---
# CONF_PANEL_EFFICIENCY = "panel_efficiency" # Might be needed for migration from older versions
# CONF_AZIMUTH = "azimuth"
# CONF_TILT = "tilt"

# --- Default Values ---
DEFAULT_SOLAR_CAPACITY = 5.0 # kWp
# DEFAULT_PANEL_EFFICIENCY = 0.18 # Removed defaults for deprecated keys unless needed for migration
# DEFAULT_AZIMUTH = 180.0
# DEFAULT_TILT = 30.0
UPDATE_INTERVAL = timedelta(minutes=30) # Default Coordinator update interval

# --- Physical Limits and Units ---
PEAK_POWER_UNIT = "kW"  # System-wide unit for peak power
MAX_HOURLY_PRODUCTION_FACTOR = 1.0  # kWh per hour â‰ˆ kWp under perfect conditions
# Safety margin for hourly production (20% above theoretical max)
HOURLY_PRODUCTION_SAFETY_MARGIN = 1.2
# Fallback max hourly production if peak power not configured (in kWh)
DEFAULT_MAX_HOURLY_KWH = 3.0

# --- Scheduling Constants ---
DAILY_UPDATE_HOUR = 6      # Hour for morning forecast update
DAILY_VERIFICATION_HOUR = 21 # Hour for evening verification

# --- File Names (Relative to DATA_DIR) ---
PREDICTION_HISTORY_FILE = "prediction_history.json"
LEARNED_WEIGHTS_FILE = "learned_weights.json"
HOURLY_PROFILE_FILE = "hourly_profile.json"
MODEL_STATE_FILE = "model_state.json"
# ERROR_LOG_FILE = "error_log.json" # Not used by data_manager, maybe error_handler?
HOURLY_SAMPLES_FILE = "hourly_samples.json"
# HOURLY_STATE_FILE = "hourly_state.json" # Not used

# --- Data Management Constants ---
DATA_DIR = f"/config/{DOMAIN}" # Use domain in path for uniqueness
DATA_VERSION = "1.0" # Version for the data format within JSON files
MAX_PREDICTION_HISTORY = 365 # Max days of prediction records to keep
MAX_HOURLY_SAMPLES = 1440 # Max hourly samples (e.g., 60 days * 24 hours)
MIN_TRAINING_DATA_POINTS = 50 # <-- GEÃƒÆ’Ã¢â‚¬Å¾NDERT (StabilitÃƒÆ’Ã‚Â¤t)
BACKUP_RETENTION_DAYS = 30 # How long to keep backups (if implemented)
MAX_BACKUP_FILES = 10      # Max number of backup files (if implemented)

# --- ML Model Constants ---
ML_MODEL_VERSION = "1.0" # Internal version of the ML model logic/features
MODEL_ACCURACY_THRESHOLD = 0.75 # Target accuracy threshold for retraining checks
PREDICTION_CONFIDENCE_THRESHOLD = 0.6 # Minimum confidence for certain actions (if used)
CORRECTION_FACTOR_MIN = 0.5 # Min value for learned fallback correction factor
CORRECTION_FACTOR_MAX = 1.5 # Max value for learned fallback correction factor

# --- Weather Calculation Constants (Example, might be in weather_calculator) ---
WEATHER_TEMP_MIN = -50.0
WEATHER_TEMP_MAX = 60.0
WEATHER_HUMIDITY_MIN = 0.0
WEATHER_HUMIDITY_MAX = 100.0
WEATHER_CLOUDS_MIN = 0.0
WEATHER_CLOUDS_MAX = 100.0
# ... other weather related constants if needed centrally

# --- Circuit Breaker Constants (Used by error_handling_service) ---
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60 # seconds
CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 3

# --- Attribute Names (Used in sensor states) ---
ATTR_FORECAST_TODAY = "forecast_today"
ATTR_FORECAST_TOMORROW = "forecast_tomorrow"
ATTR_WEATHER_CONDITION = "weather_condition"
ATTR_LEARNING_STATUS = "learning_status"
ATTR_LAST_LEARNING = "last_learning"
ATTR_MODEL_ACCURACY = "model_accuracy"
ATTR_WEATHER_SOURCE = "weather_source"
ATTR_RETRY_COUNT = "retry_count" # Check if used
ATTR_FALLBACK_ACTIVE = "fallback_active" # Check if used

# --- Button Keys (Used in button platform) ---
BUTTON_MANUAL_FORECAST = "manual_forecast"
BUTTON_MANUAL_LEARNING = "manual_learning" # Changed from retrain_model for consistency
# BUTTON_RESET_DATA = "reset_data" # Removed? Check services.yaml

# --- Service Names (Used for service registration) ---
SERVICE_MANUAL_FORECAST = "manual_forecast" # Check services.yaml
SERVICE_RETRAIN_MODEL = "force_retrain" # Match services.yaml
SERVICE_RESET_LEARNING_DATA = "reset_model" # Match services.yaml

# --- Other Constants ---
ICON_SOLAR = "mdi:solar-power"
ICON_FORECAST = "mdi:weather-sunny"
ICON_LEARNING = "mdi:brain"
ICON_BUTTON = "mdi:play-circle-outline"

UNIT_KWH = "kWh"
UNIT_PERCENTAGE = "%"

LOGGER_NAME = f"custom_components.{DOMAIN}" # For consistency in logging

FILE_ENCODING = "utf-8"
JSON_INDENT = 2
ATOMIC_WRITE_ENABLED = True # Flag if atomic writes are used (they are)

# --- Timing / Intervals ---
COORDINATOR_UPDATE_INTERVAL = timedelta(minutes=30) # Default update interval
LEARNING_UPDATE_INTERVAL = timedelta(hours=1) # How often samples are collected
CLEANUP_INTERVAL = timedelta(days=1) # How often cleanup tasks might run

# --- Performance / Limits ---
MAX_CONCURRENT_OPERATIONS = 3 # Example limit for concurrent tasks
THREAD_POOL_SIZE = 2 # For blocking operations in executor

# --- Validation Ranges ---
MIN_SOLAR_CAPACITY = 0.1
MAX_SOLAR_CAPACITY = 1000.0 # Increased upper limit from 100.0
# MIN_TEMPERATURE = -50 # Defined above
# MAX_TEMPERATURE = 60
# MIN_HUMIDITY = 0
# MAX_HUMIDITY = 100
# MIN_CLOUDS = 0
# MAX_CLOUDS = 100

# --- Status Strings ---
STATUS_INITIALIZING = "initializing"
STATUS_READY = "ready"
STATUS_LEARNING = "learning"
STATUS_FORECASTING = "forecasting"
STATUS_ERROR = "error"
STATUS_OFFLINE = "offline" # Or unavailable?

# --- Device Info ---
DEVICE_MANUFACTURER = "Zara-Toorox"
# DEVICE_MODEL = INTEGRATION_MODEL # Use constant defined above
# DEVICE_SW_VERSION = VERSION # Use constant defined above

# --- Sun Calculation Constants ---
SUN_BUFFER_HOURS = 1.5 # <-- GEÃƒÆ’Ã¢â‚¬Å¾NDERT (Block 4)
FALLBACK_PRODUCTION_START_HOUR = 5
FALLBACK_PRODUCTION_END_HOUR = 21
