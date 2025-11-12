"""
Constants for Solar Forecast ML Integration

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
VERSION = "8.6.0"
RELEASE_VERSION = "8.6.0"
SOFTWARE_VERSION = "8.6.0"
INTEGRATION_MODEL = "v8.6.0"
ML_VERSION = "8.0.0"

PLATFORMS = [Platform.SENSOR, Platform.BUTTON]

# --- Core Configuration Keys ---
CONF_WEATHER_ENTITY = "weather_entity"
CONF_POWER_ENTITY = "power_entity"
CONF_SOLAR_YIELD_TODAY = "solar_yield_today"
CONF_SOLAR_CAPACITY = "solar_capacity" # Changed from plant_kwp

# --- Optional Configuration Keys ---
CONF_TOTAL_CONSUMPTION_TODAY = "total_consumption_today"
CONF_GRID_IMPORT_TODAY = "grid_import_today"
CONF_GRID_EXPORT_TODAY = "grid_export_today"
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
CONF_LEARNING_ENABLED = "learning_enabled"

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
MAX_HOURLY_PRODUCTION_FACTOR = 1.0  # kWh per hour per kWp under perfect conditions
# Safety margin for hourly production (20% above theoretical max)
HOURLY_PRODUCTION_SAFETY_MARGIN = 1.2
# Fallback max hourly production if peak power not configured (in kWh)
DEFAULT_MAX_HOURLY_KWH = 3.0

# --- Scheduling Constants ---
DAILY_UPDATE_HOUR = 6      # Hour for morning forecast update
DAILY_VERIFICATION_HOUR = 21 # Hour for evening verification

# --- Directory Structure (Hierarchical) ---
BASE_DATA_DIR = f"/config/{DOMAIN}"  # Base directory for all integration data

# Subdirectories
ML_DIR = "ml"                    # Machine Learning models, weights, profiles
STATS_DIR = "stats"              # Statistics & forecast history
DATA_DIR = "data"                # Runtime state files
IMPORTS_DIR = "imports"          # User imports (external forecasts)
EXPORTS_DIR = "exports"          # User exports (reports, pictures, statistics)
BACKUPS_DIR = "backups"          # Automatic and manual backups
ASSETS_DIR = "assets"            # Internal assets (logos, icons)
DOCS_DIR = "docs"                # Documentation files

# Exports subdirectories
EXPORTS_REPORTS_DIR = "reports"
EXPORTS_PICTURES_DIR = "pictures"
EXPORTS_STATISTICS_DIR = "statistics"

# Backups subdirectories
BACKUPS_AUTO_DIR = "auto"
BACKUPS_MANUAL_DIR = "manual"

# --- File Names (Relative paths within subdirectories) ---
# ML Files
LEARNED_WEIGHTS_FILE = "learned_weights.json"
HOURLY_PROFILE_FILE = "hourly_profile.json"
HOURLY_SAMPLES_FILE = "hourly_samples.json"
MODEL_STATE_FILE = "model_state.json"

# Stats Files
DAILY_FORECASTS_FILE = "daily_forecasts.json"  # NEW: Expected daily production history
PREDICTION_HISTORY_FILE = "prediction_history.json"

# Runtime State Files
COORDINATOR_STATE_FILE = "coordinator_state.json"
PRODUCTION_TIME_STATE_FILE = "production_time_state.json"  # NEW: For production time persistence

# --- Data Management Constants ---
DATA_VERSION = "1.0" # Version for the data format within JSON files
MAX_PREDICTION_HISTORY = 365 # Max days of prediction records to keep
MAX_HOURLY_SAMPLES = 1440 # Max hourly samples (e.g., 60 days * 24 hours)
MIN_TRAINING_DATA_POINTS = 10  # Minimum samples for stable training (lowered for V2 testing)
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
BUTTON_MANUAL_LEARNING = "manual_learning" # Changed from retrain_model for consistency
# BUTTON_RESET_DATA = "reset_data" # Removed? Check services.yaml

# --- Service Names (Used for service registration) ---
SERVICE_RETRAIN_MODEL = "force_retrain" # Match services.yaml
SERVICE_RESET_LEARNING_DATA = "reset_model" # Match services.yaml
SERVICE_FINALIZE_DAY = "finalize_day" # Emergency day-end service
SERVICE_MOVE_TO_HISTORY = "move_to_history" # Emergency history service
SERVICE_CALCULATE_STATS = "calculate_stats" # Emergency statistics service
SERVICE_RUN_ALL_DAY_END_TASKS = "run_all_day_end_tasks" # Emergency complete day-end

# TESTING SERVICES - Simulate scheduled tasks for debugging (100% code-accurate replicas)
SERVICE_TEST_MORNING_ROUTINE = "test_morning_routine"  # Test: Complete 6 AM routine (weather + forecast + best hour)
SERVICE_TEST_BEST_HOUR = "test_best_hour"  # Test: Calculate best production hour using V2 sun-aware algorithm
SERVICE_TEST_TOMORROW_LOCK = "test_tomorrow_lock"  # Test: Simulate 12 PM tomorrow forecast lock
SERVICE_TEST_DAY_AFTER_SAVE = "test_day_after_save"  # Test: Simulate 6 AM day after tomorrow save (unlocked)
SERVICE_TEST_DAY_AFTER_LOCK = "test_day_after_lock"  # Test: Simulate 18 PM day after tomorrow lock
SERVICE_TEST_HOURLY_UPDATE = "test_hourly_update"  # Test: Trigger hourly actual update task (runs every hour at :05)

# MANUAL CONTROL SERVICES - Direct user control (use with caution!)
SERVICE_FORCE_TODAY_LOCK = "force_today_lock"  # Force: Override today's forecast lock (6 AM task with overwrite)

SERVICE_COLLECT_HOURLY_SAMPLE = "collect_hourly_sample"
SERVICE_NIGHT_CLEANUP = "night_cleanup" # Manual night cleanup (remove duplicates and zero-production samples)
SERVICE_RUN_ALL_SCHEDULED_TASKS = "run_all_scheduled_tasks" # Testing: Run all scheduled tasks in sequence
SERVICE_GENERATE_CHART = "generate_chart" # Generate forecast vs actual chart
SERVICE_RELOAD_SCHEDULED_TASKS = "reload_scheduled_tasks"  # Reload: Re-register all scheduled task listeners

# NEW: ML-Optimized Data Structure Debug Services
SERVICE_DEBUG_CREATE_HOURLY_PREDICTIONS = "debug_create_hourly_predictions"  # Debug: Manually create hourly predictions
SERVICE_DEBUG_UPDATE_HOURLY_ACTUAL = "debug_update_hourly_actual"  # Debug: Manually update actual value for specific hour
SERVICE_DEBUG_CREATE_DAILY_SUMMARY = "debug_create_daily_summary"  # Debug: Manually create daily summary
SERVICE_DEBUG_SHOW_PREDICTION = "debug_show_prediction"  # Debug: Show detailed prediction info
SERVICE_MIGRATE_DATA = "migrate_data"  # Migration: Migrate from old prediction_history.json
SERVICE_VALIDATE_DATA = "validate_data"  # Validation: Validate data integrity

# NEW: Astronomy Cache Services
SERVICE_BUILD_ASTRONOMY_CACHE = "build_astronomy_cache"  # Build astronomy cache for date range
SERVICE_EXTRACT_MAX_PEAKS = "extract_max_peaks"  # Extract max peak records from history
SERVICE_REFRESH_CACHE_TODAY = "refresh_cache_today"  # Refresh cache for today + next 7 days

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
SUN_BUFFER_HOURS = 1.5  # Buffer hours for sun calculations
FALLBACK_PRODUCTION_START_HOUR = 5
FALLBACK_PRODUCTION_END_HOUR = 21

# ============================================================================
# BATTERY MANAGEMENT CONSTANTS (v9.0.0 - Watt-based System)
# ============================================================================
# Completely separate from Solar/ML - no interference with existing code
#
# V9.0.0 BREAKING CHANGE:
# - New watt-based energy flow calculation
# - All sensors use Watt (W) for power measurements
# - Energy (kWh) calculated by integration over time
# - More accurate and universal for all battery systems

# --- NEW: Watt-based Battery Configuration (v9.0.0) ---
# REQUIRED sensors - all power values in Watt (W)
CONF_BATTERY_POWER_SENSOR = "battery_power_sensor"  # Battery power (W, +charge/-discharge)
CONF_BATTERY_SOC_SENSOR = "battery_soc_sensor"  # State of Charge (%, 0-100)
CONF_BATTERY_CAPACITY = "battery_capacity"  # Battery capacity (kWh)
CONF_SOLAR_PRODUCTION_SENSOR = "solar_production_sensor"  # Solar production (W, ≥0)
CONF_INVERTER_OUTPUT_SENSOR = "inverter_output_sensor"  # Inverter AC output (W, ≥0)
CONF_HOUSE_CONSUMPTION_SENSOR = "house_consumption_sensor"  # House consumption (W, ≥0)
CONF_GRID_IMPORT_SENSOR = "grid_import_sensor"  # Grid import (W, ≥0)
CONF_GRID_EXPORT_SENSOR = "grid_export_sensor"  # Grid export (W, ≥0)

# OPTIONAL sensors
CONF_GRID_CHARGE_POWER_SENSOR = "grid_charge_power_sensor"  # Grid charge power (W, ≥0)
CONF_BATTERY_TEMPERATURE_SENSOR = "battery_temperature_sensor"  # Temperature (°C)

# --- LEGACY Configuration (v8.x - DEPRECATED) ---
# These are kept for backwards compatibility but will show deprecation warnings
CONF_BATTERY_ENABLED = "battery_enabled"
CONF_BATTERY_SOC_ENTITY = "battery_soc_entity"  # DEPRECATED: use CONF_BATTERY_SOC_SENSOR
CONF_BATTERY_POWER_ENTITY = "battery_power_entity"  # DEPRECATED: use CONF_BATTERY_POWER_SENSOR
CONF_BATTERY_GRID_CHARGE_POWER_ENTITY = "battery_grid_charge_power_entity"  # DEPRECATED
CONF_BATTERY_CHARGE_TODAY_ENTITY = "battery_charge_today_entity"  # DEPRECATED: calculated from power
CONF_BATTERY_DISCHARGE_TODAY_ENTITY = "battery_discharge_today_entity"  # DEPRECATED: calculated from power
CONF_BATTERY_TEMPERATURE_ENTITY = "battery_temperature_entity"  # DEPRECATED: use CONF_BATTERY_TEMPERATURE_SENSOR

# --- Electricity Price Configuration ---
CONF_ELECTRICITY_COUNTRY = "electricity_country"  # Country for electricity prices (DE/AT)
CONF_ELECTRICITY_ENABLED = "electricity_enabled"  # Enable electricity price features

# --- aWATTar API Configuration (Free, No Registration Required) ---
# aWATTar provides free electricity spot prices for DE and AT
# No API key needed - 100 requests per day under fair use
# Data source: EPEX Spot, updated daily at 14:00

# --- Battery Defaults ---
DEFAULT_BATTERY_CAPACITY = 10.0  # kWh
MIN_BATTERY_CAPACITY = 0.5
MAX_BATTERY_CAPACITY = 1000.0
DEFAULT_ELECTRICITY_COUNTRY = "DE"

# --- Electricity Price Constants ---
ELECTRICITY_PRICE_UPDATE_HOUR = 13  # ENTSO-E publishes day-ahead prices around 13:00
ELECTRICITY_PRICE_CACHE_FILE = "electricity_prices.json"
MAX_ELECTRICITY_PRICE_HISTORY = 90  # Days of price history to keep

# --- Battery Sensor Icons ---
ICON_BATTERY = "mdi:battery"
ICON_BATTERY_CHARGING = "mdi:battery-charging"
ICON_BATTERY_DISCHARGING = "mdi:battery-minus"
ICON_ELECTRICITY_PRICE = "mdi:currency-eur"
ICON_CHARGING_RECOMMENDATION = "mdi:lightbulb-on"

# --- Battery Data Files ---
BATTERY_STATE_FILE = "battery_state.json"
BATTERY_STATISTICS_FILE = "battery_statistics.json"

# --- Charging Strategy Constants ---
CHARGING_PRICE_PERCENTILE_CHEAP = 25  # Below 25th percentile = cheap
CHARGING_PRICE_PERCENTILE_EXPENSIVE = 75  # Above 75th percentile = expensive
MIN_CHARGING_DURATION_HOURS = 2  # Minimum recommended charging duration

# --- Battery Efficiency ---
DEFAULT_BATTERY_EFFICIENCY = 0.9  # 90% round-trip efficiency
BATTERY_SELF_DISCHARGE_RATE = 0.02  # 2% per day

# --- Units ---
UNIT_EURO_PER_KWH = "€/kWh"
UNIT_CENT_PER_KWH = "ct/kWh"
UNIT_WATT = "W"
UNIT_HOURS = "h"

# --- Battery Sensor Unique IDs ---
BATTERY_SOC_SENSOR = "soc"
BATTERY_POWER_SENSOR = "power"
BATTERY_RUNTIME_REMAINING_SENSOR = "runtime_remaining"
BATTERY_EFFICIENCY_SENSOR = "efficiency"

# LEGACY v8.x (still used for forecast/cost sensors - not deleted)
BATTERY_EXPECTED_CHARGE_SOLAR_SENSOR = "expected_charge_solar"
BATTERY_CHARGE_FROM_SOLAR_SENSOR = "charge_from_solar"
BATTERY_CHARGE_FROM_GRID_SENSOR = "charge_from_grid"

ELECTRICITY_PRICE_CURRENT_SENSOR = "price_current"
ELECTRICITY_PRICE_NEXT_HOUR_SENSOR = "price_next_hour"
ELECTRICITY_PRICE_AVG_TODAY_SENSOR = "price_avg_today"
ELECTRICITY_PRICE_AVG_WEEK_SENSOR = "price_avg_week"
ELECTRICITY_PRICE_MIN_TODAY_SENSOR = "price_min_today"
ELECTRICITY_PRICE_MAX_TODAY_SENSOR = "price_max_today"
ELECTRICITY_CHEAPEST_HOUR_TODAY_SENSOR = "cheapest_hour_today"
ELECTRICITY_MOST_EXPENSIVE_HOUR_TODAY_SENSOR = "most_expensive_hour_today"
ELECTRICITY_CHARGING_RECOMMENDATION_SENSOR = "charging_recommendation"
ELECTRICITY_SAVINGS_TODAY_SENSOR = "savings_today"

# --- Battery Cost & Profit Sensors ---
BATTERY_SOLAR_DISCHARGE_TODAY_SENSOR = "solar_discharge_today"
BATTERY_GRID_DISCHARGE_TODAY_SENSOR = "grid_discharge_today"
BATTERY_GRID_CHARGE_COST_TODAY_SENSOR = "grid_charge_cost_today"
BATTERY_SOLAR_SAVINGS_TODAY_SENSOR = "solar_savings_today"
BATTERY_GRID_ARBITRAGE_PROFIT_TODAY_SENSOR = "grid_arbitrage_profit_today"
BATTERY_TOTAL_PROFIT_TODAY_SENSOR = "total_profit_today"
BATTERY_GRID_CHARGE_MONTH_SENSOR = "grid_charge_month"
BATTERY_TOTAL_PROFIT_MONTH_SENSOR = "total_profit_month"
BATTERY_TOTAL_PROFIT_YEAR_SENSOR = "total_profit_year"

# --- Autarky & Self-Consumption (with Battery) ---
AUTARKY_WITH_BATTERY_SENSOR = "autarky_with_battery"
SELF_CONSUMPTION_WITH_BATTERY_SENSOR = "self_consumption_with_battery"
GRID_EXPORT_TODAY_SENSOR = "grid_export_today"
GRID_IMPORT_TODAY_SENSOR = "grid_import_today"
DIRECT_SOLAR_CONSUMPTION_SENSOR = "direct_solar_consumption"

# --- NEW: Energy Flow Sensors (v9.0.0) ---
SOLAR_TO_HOUSE_SENSOR = "solar_to_house"
SOLAR_TO_BATTERY_SENSOR = "solar_to_battery"
SOLAR_TO_GRID_SENSOR = "solar_to_grid"
GRID_TO_HOUSE_SENSOR = "grid_to_house"
GRID_TO_BATTERY_SENSOR = "grid_to_battery"
BATTERY_TO_HOUSE_SENSOR = "battery_to_house"
