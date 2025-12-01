"""Constants for Solar Forecast ML Integration V10.0.0 @zara

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
VERSION = "10.0.1"
RELEASE_VERSION = "10.0.1"
RELEASE_NAME = "Lyra"
SOFTWARE_VERSION = "10.0.1"
INTEGRATION_MODEL = "V10.0.1"
ML_VERSION = "10.0.0"

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
CONF_LEARNING_ENABLED = "learning_enabled"

CONF_ML_ALGORITHM = "ml_algorithm"
CONF_ENABLE_TINY_LSTM = "enable_tiny_lstm"
DEFAULT_ML_ALGORITHM = "auto"
DEFAULT_ENABLE_TINY_LSTM = True

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

# ML Services
SERVICE_RETRAIN_MODEL = "force_retrain"
SERVICE_RESET_LEARNING_DATA = "reset_model"

# Emergency Services
SERVICE_RUN_ALL_DAY_END_TASKS = "run_all_day_end_tasks"

# Testing Services
SERVICE_TEST_MORNING_ROUTINE = "test_morning_routine"

# Astronomy Services
SERVICE_BUILD_ASTRONOMY_CACHE = "build_astronomy_cache"
SERVICE_REFRESH_CACHE_TODAY = "refresh_cache_today"

# Weather Services
SERVICE_RUN_WEATHER_CORRECTION = "run_weather_correction"
SERVICE_REFRESH_OPEN_METEO_CACHE = "refresh_open_meteo_cache"
SERVICE_BOOTSTRAP_FROM_HISTORY = "bootstrap_from_history"

# Physics Services
SERVICE_BOOTSTRAP_PHYSICS_FROM_HISTORY = "bootstrap_physics_from_history"

# Notification Services
SERVICE_SEND_DAILY_BRIEFING = "send_daily_briefing"

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

CONF_BATTERY_POWER_SENSOR = "battery_power_sensor"
CONF_BATTERY_SOC_SENSOR = "battery_soc_sensor"
CONF_BATTERY_CAPACITY = "battery_capacity"
CONF_SOLAR_PRODUCTION_SENSOR = "solar_production_sensor"
CONF_INVERTER_OUTPUT_SENSOR = "inverter_output_sensor"
CONF_HOUSE_CONSUMPTION_SENSOR = "house_consumption_sensor"
CONF_GRID_IMPORT_SENSOR = "grid_import_sensor"
CONF_GRID_EXPORT_SENSOR = "grid_export_sensor"

CONF_GRID_CHARGE_POWER_SENSOR = "grid_charge_power_sensor"
CONF_BATTERY_TEMPERATURE_SENSOR = "battery_temperature_sensor"

CONF_BATTERY_ENABLED = "battery_enabled"
CONF_BATTERY_SOC_ENTITY = "battery_soc_entity"
CONF_BATTERY_POWER_ENTITY = "battery_power_entity"
CONF_BATTERY_GRID_CHARGE_POWER_ENTITY = "battery_grid_charge_power_entity"
CONF_BATTERY_CHARGE_TODAY_ENTITY = "battery_charge_today_entity"
CONF_BATTERY_DISCHARGE_TODAY_ENTITY = "battery_discharge_today_entity"
CONF_BATTERY_TEMPERATURE_ENTITY = "battery_temperature_entity"

CONF_ELECTRICITY_COUNTRY = "electricity_country"
CONF_ELECTRICITY_ENABLED = "electricity_enabled"

DEFAULT_BATTERY_CAPACITY = 10.0
MIN_BATTERY_CAPACITY = 0.5
MAX_BATTERY_CAPACITY = 1000.0
DEFAULT_ELECTRICITY_COUNTRY = "DE"

ELECTRICITY_PRICE_UPDATE_HOUR = 13
ELECTRICITY_PRICE_CACHE_FILE = "electricity_prices.json"
MAX_ELECTRICITY_PRICE_HISTORY = 90

ICON_BATTERY = "mdi:battery"
ICON_BATTERY_CHARGING = "mdi:battery-charging"
ICON_BATTERY_DISCHARGING = "mdi:battery-minus"
ICON_ELECTRICITY_PRICE = "mdi:currency-eur"
ICON_CHARGING_RECOMMENDATION = "mdi:lightbulb-on"

BATTERY_STATE_FILE = "battery_state.json"
BATTERY_STATISTICS_FILE = "battery_statistics.json"

CHARGING_PRICE_PERCENTILE_CHEAP = 25
CHARGING_PRICE_PERCENTILE_EXPENSIVE = 75
MIN_CHARGING_DURATION_HOURS = 2

DEFAULT_BATTERY_EFFICIENCY = 0.9
BATTERY_SELF_DISCHARGE_RATE = 0.02

UNIT_EURO_PER_KWH = "€/kWh"
UNIT_CENT_PER_KWH = "ct/kWh"
UNIT_WATT = "W"
UNIT_HOURS = "h"

BATTERY_SOC_SENSOR = "soc"
BATTERY_POWER_SENSOR = "power"
BATTERY_RUNTIME_REMAINING_SENSOR = "runtime_remaining"
BATTERY_EFFICIENCY_SENSOR = "efficiency"

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

BATTERY_SOLAR_DISCHARGE_TODAY_SENSOR = "solar_discharge_today"
BATTERY_GRID_DISCHARGE_TODAY_SENSOR = "grid_discharge_today"
BATTERY_GRID_CHARGE_COST_TODAY_SENSOR = "grid_charge_cost_today"
BATTERY_SOLAR_SAVINGS_TODAY_SENSOR = "solar_savings_today"
BATTERY_GRID_ARBITRAGE_PROFIT_TODAY_SENSOR = "grid_arbitrage_profit_today"
BATTERY_TOTAL_PROFIT_TODAY_SENSOR = "total_profit_today"
BATTERY_GRID_CHARGE_MONTH_SENSOR = "grid_charge_month"
BATTERY_TOTAL_PROFIT_MONTH_SENSOR = "total_profit_month"
BATTERY_TOTAL_PROFIT_YEAR_SENSOR = "total_profit_year"

AUTARKY_WITH_BATTERY_SENSOR = "autarky_with_battery"
SELF_CONSUMPTION_WITH_BATTERY_SENSOR = "self_consumption_with_battery"
GRID_EXPORT_TODAY_SENSOR = "grid_export_today"
GRID_IMPORT_TODAY_SENSOR = "grid_import_today"
DIRECT_SOLAR_CONSUMPTION_SENSOR = "direct_solar_consumption"

SOLAR_TO_HOUSE_SENSOR = "solar_to_house"
SOLAR_TO_BATTERY_SENSOR = "solar_to_battery"
SOLAR_TO_GRID_SENSOR = "solar_to_grid"
GRID_TO_HOUSE_SENSOR = "grid_to_house"
GRID_TO_BATTERY_SENSOR = "grid_to_battery"
BATTERY_TO_HOUSE_SENSOR = "battery_to_house"
