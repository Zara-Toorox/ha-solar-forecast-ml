# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""Constants for SFML Stats integration."""
from __future__ import annotations

from pathlib import Path
from typing import Final

DOMAIN: Final = "sfml_stats"
NAME: Final = "SFML Stats"
VERSION: Final = "6.4.2"

SOLAR_FORECAST_ML_BASE: Final = Path("solar_forecast_ml")
SOLAR_FORECAST_ML_STATS: Final = SOLAR_FORECAST_ML_BASE / "stats"
SOLAR_FORECAST_ML_AI: Final = SOLAR_FORECAST_ML_BASE / "ai"
SOLAR_FORECAST_ML_DATA: Final = SOLAR_FORECAST_ML_BASE / "data"
SOLAR_FORECAST_ML_PHYSICS: Final = SOLAR_FORECAST_ML_BASE / "physics"

# Stats files
SOLAR_DAILY_SUMMARIES: Final = "daily_summaries.json"
SOLAR_HOURLY_PREDICTIONS: Final = "hourly_predictions.json"
SOLAR_ASTRONOMY_CACHE: Final = "astronomy_cache.json"

# AI files (in ai/ directory)
SOLAR_LEARNED_WEIGHTS: Final = "learned_weights.json"
SOLAR_SEASONAL: Final = "seasonal.json"
SOLAR_DNI_TRACKER: Final = "dni_tracker.json"

GRID_PRICE_MONITOR_BASE: Final = Path("grid_price_monitor")
GRID_PRICE_MONITOR_DATA: Final = GRID_PRICE_MONITOR_BASE / "data"

GRID_PRICE_HISTORY: Final = "price_history.json"
GRID_STATISTICS: Final = "statistics.json"
GRID_PRICE_CACHE: Final = "price_cache.json"

SFML_STATS_BASE: Final = Path("sfml_stats")
SFML_STATS_WEEKLY: Final = SFML_STATS_BASE / "weekly"
SFML_STATS_MONTHLY: Final = SFML_STATS_BASE / "monthly"
SFML_STATS_CHARTS: Final = SFML_STATS_BASE / "charts"
SFML_STATS_REPORTS: Final = SFML_STATS_BASE / "reports"
SFML_STATS_CACHE: Final = SFML_STATS_BASE / ".cache"
SFML_STATS_DATA: Final = SFML_STATS_BASE / "data"

DAILY_ENERGY_HISTORY: Final = "daily_energy_history.json"
HOURLY_BILLING_HISTORY: Final = "hourly_billing_history.json"

EXPORT_DIRECTORIES: Final = [
    SFML_STATS_BASE,
    SFML_STATS_WEEKLY,
    SFML_STATS_MONTHLY,
    SFML_STATS_CHARTS,
    SFML_STATS_REPORTS,
    SFML_STATS_CACHE,
    SFML_STATS_DATA,
]

WEEKLY_REPORT_PATTERN: Final = "weekly_report_KW{week:02d}_{year}.png"
MONTHLY_REPORT_PATTERN: Final = "monthly_report_{year}_{month:02d}.png"

CHART_SIZE_WEEKLY: Final = (16, 20)
CHART_SIZE_MONTHLY: Final = (18, 24)
CHART_DPI: Final = 150

WEEKLY_REPORT_DAY: Final = 6
WEEKLY_REPORT_HOUR: Final = 23
MONTHLY_REPORT_DAY: Final = 1
MONTHLY_REPORT_HOUR: Final = 2

UPDATE_INTERVAL: Final = 3600

COLORS: Final = {
    # Hintergrund mit mehr Tiefe
    "background": "#0d1117",
    "background_light": "#161b22",
    "background_card": "#21262d",
    "background_card_hover": "#30363d",

    # Text mit besserem Kontrast
    "text_primary": "#f0f6fc",
    "text_secondary": "#8b949e",
    "text_muted": "#6e7681",

    # Solar-Farben mit Glow-Effekt-Unterstützung
    "solar_yellow": "#ffd60a",
    "solar_orange": "#ff9500",
    "solar_gold": "#ffb627",

    # Neon-Akzente
    "neon_cyan": "#00d4ff",
    "neon_green": "#39ff14",
    "neon_pink": "#ff2e97",
    "neon_purple": "#bf5af2",

    # Preise mit Gradient-Support
    "price_green": "#2ecc71",
    "price_red": "#e74c3c",
    "price_yellow": "#f1c40f",

    # ML/Prediction Farben
    "ml_purple": "#a855f7",
    "rule_based_blue": "#3b82f6",

    # Actual vs Predicted
    "actual": "#10b981",
    "predicted": "#6366f1",

    # Genauigkeit mit Farbverlauf
    "accuracy_good": "#22c55e",
    "accuracy_medium": "#eab308",
    "accuracy_bad": "#ef4444",

    # Grid und Borders (subtiler)
    "grid": "#30363d",
    "border": "#3d444d",
    "border_glow": "#58a6ff",

    # Glassmorphism-Unterstützung
    "glass_bg": "rgba(255, 255, 255, 0.05)",
    "glass_border": "rgba(255, 255, 255, 0.1)",

    # Gradient-Endpunkte
    "gradient_start": "#667eea",
    "gradient_end": "#764ba2",
    "solar_gradient_start": "#f093fb",
    "solar_gradient_end": "#f5576c",
}

LOGGER_NAME: Final = "sfml_stats"

CONF_GENERATE_WEEKLY: Final = "generate_weekly"
CONF_GENERATE_MONTHLY: Final = "generate_monthly"
CONF_AUTO_GENERATE: Final = "auto_generate"
CONF_THEME: Final = "theme"
CONF_DASHBOARD_STYLE: Final = "dashboard_style"

THEME_DARK: Final = "dark"
THEME_LIGHT: Final = "light"

DASHBOARD_STYLE_3D: Final = "3d"
DASHBOARD_STYLE_2D: Final = "2d"

DEFAULT_GENERATE_WEEKLY: Final = True
DEFAULT_GENERATE_MONTHLY: Final = True
DEFAULT_AUTO_GENERATE: Final = True
DEFAULT_THEME: Final = THEME_DARK
DEFAULT_DASHBOARD_STYLE: Final = DASHBOARD_STYLE_3D

CONF_SENSOR_SOLAR_POWER: Final = "sensor_solar_power"
CONF_SENSOR_SOLAR_TO_HOUSE: Final = "sensor_solar_to_house"
CONF_SENSOR_SOLAR_TO_BATTERY: Final = "sensor_solar_to_battery"
CONF_SENSOR_BATTERY_TO_HOUSE: Final = "sensor_battery_to_house"
CONF_SENSOR_BATTERY_TO_GRID: Final = "sensor_battery_to_grid"
CONF_SENSOR_GRID_TO_HOUSE: Final = "sensor_grid_to_house"
CONF_SENSOR_GRID_TO_BATTERY: Final = "sensor_grid_to_battery"
CONF_SENSOR_HOUSE_TO_GRID: Final = "sensor_house_to_grid"
CONF_SENSOR_BATTERY_SOC: Final = "sensor_battery_soc"
CONF_SENSOR_BATTERY_POWER: Final = "sensor_battery_power"
CONF_SENSOR_HOME_CONSUMPTION: Final = "sensor_home_consumption"
CONF_SENSOR_SOLAR_YIELD_DAILY: Final = "sensor_solar_yield_daily"
CONF_SENSOR_GRID_IMPORT_DAILY: Final = "sensor_grid_import_daily"
CONF_SENSOR_GRID_IMPORT_YEARLY: Final = "sensor_grid_import_yearly"
CONF_SENSOR_BATTERY_CHARGE_SOLAR_DAILY: Final = "sensor_battery_charge_solar_daily"
CONF_SENSOR_BATTERY_CHARGE_GRID_DAILY: Final = "sensor_battery_charge_grid_daily"
CONF_SENSOR_PRICE_TOTAL: Final = "sensor_price_total"

CONF_SENSOR_SMARTMETER_IMPORT_KWH: Final = "sensor_smartmeter_import_kwh"
CONF_SENSOR_SMARTMETER_EXPORT_KWH: Final = "sensor_smartmeter_export_kwh"
CONF_SENSOR_SOLAR_YIELD_TOTAL_KWH: Final = "sensor_solar_yield_total_kwh"

CONF_SENSOR_SMARTMETER_IMPORT: Final = "sensor_smartmeter_import"
CONF_SENSOR_SMARTMETER_EXPORT: Final = "sensor_smartmeter_export"

CONF_WEATHER_ENTITY: Final = "weather_entity"

CONF_SENSOR_PANEL1_POWER: Final = "sensor_panel1_power"
CONF_SENSOR_PANEL1_MAX_TODAY: Final = "sensor_panel1_max_today"
CONF_SENSOR_PANEL2_POWER: Final = "sensor_panel2_power"
CONF_SENSOR_PANEL2_MAX_TODAY: Final = "sensor_panel2_max_today"
CONF_SENSOR_PANEL3_POWER: Final = "sensor_panel3_power"
CONF_SENSOR_PANEL3_MAX_TODAY: Final = "sensor_panel3_max_today"
CONF_SENSOR_PANEL4_POWER: Final = "sensor_panel4_power"
CONF_SENSOR_PANEL4_MAX_TODAY: Final = "sensor_panel4_max_today"

CONF_PANEL1_NAME: Final = "panel1_name"
CONF_PANEL2_NAME: Final = "panel2_name"
CONF_PANEL3_NAME: Final = "panel3_name"
CONF_PANEL4_NAME: Final = "panel4_name"

DEFAULT_PANEL1_NAME: Final = "Panel 1"
DEFAULT_PANEL2_NAME: Final = "Panel 2"
DEFAULT_PANEL3_NAME: Final = "Panel 3"
DEFAULT_PANEL4_NAME: Final = "Panel 4"

CONF_BILLING_START_DAY: Final = "billing_start_day"
CONF_BILLING_START_MONTH: Final = "billing_start_month"
CONF_BILLING_PRICE_MODE: Final = "billing_price_mode"
CONF_BILLING_FIXED_PRICE: Final = "billing_fixed_price"
CONF_FEED_IN_TARIFF: Final = "feed_in_tariff"

# Panel Group Name Mapping (SFML Stats überschreibt Solar Forecast ML Namen)
CONF_PANEL_GROUP_NAMES: Final = "panel_group_names"

PRICE_MODE_FIXED: Final = "fixed"
PRICE_MODE_DYNAMIC: Final = "dynamic"

DEFAULT_BILLING_START_DAY: Final = 1
DEFAULT_BILLING_START_MONTH: Final = 1
DEFAULT_BILLING_PRICE_MODE: Final = PRICE_MODE_DYNAMIC
DEFAULT_BILLING_FIXED_PRICE: Final = 35.0
DEFAULT_FEED_IN_TARIFF: Final = 8.1

ENERGY_FLOW_SENSORS: Final = [
    CONF_SENSOR_SOLAR_POWER,
    CONF_SENSOR_SOLAR_TO_HOUSE,
    CONF_SENSOR_SOLAR_TO_BATTERY,
    CONF_SENSOR_BATTERY_TO_HOUSE,
    CONF_SENSOR_BATTERY_TO_GRID,
    CONF_SENSOR_GRID_TO_HOUSE,
    CONF_SENSOR_GRID_TO_BATTERY,
    CONF_SENSOR_HOUSE_TO_GRID,
    CONF_SENSOR_BATTERY_SOC,
    CONF_SENSOR_BATTERY_POWER,
    CONF_SENSOR_HOME_CONSUMPTION,
    CONF_SENSOR_SOLAR_YIELD_DAILY,
    CONF_SENSOR_GRID_IMPORT_DAILY,
    CONF_SENSOR_GRID_IMPORT_YEARLY,
    CONF_SENSOR_BATTERY_CHARGE_SOLAR_DAILY,
    CONF_SENSOR_BATTERY_CHARGE_GRID_DAILY,
    CONF_SENSOR_PRICE_TOTAL,
    CONF_SENSOR_SMARTMETER_IMPORT_KWH,
    CONF_SENSOR_SMARTMETER_EXPORT_KWH,
    CONF_SENSOR_SOLAR_YIELD_TOTAL_KWH,
]

PANEL_SENSORS: Final = [
    CONF_SENSOR_PANEL1_POWER,
    CONF_SENSOR_PANEL1_MAX_TODAY,
    CONF_SENSOR_PANEL2_POWER,
    CONF_SENSOR_PANEL2_MAX_TODAY,
    CONF_SENSOR_PANEL3_POWER,
    CONF_SENSOR_PANEL3_MAX_TODAY,
    CONF_SENSOR_PANEL4_POWER,
    CONF_SENSOR_PANEL4_MAX_TODAY,
]

# =============================================================================
# Consumer Sensors (Wärmepumpe, Heizstab, Wallbox) - Optional
# =============================================================================

# Wärmepumpe (Heat Pump)
CONF_SENSOR_HEATPUMP_POWER: Final = "sensor_heatpump_power"
CONF_SENSOR_HEATPUMP_DAILY: Final = "sensor_heatpump_daily"
CONF_SENSOR_HEATPUMP_COP: Final = "sensor_heatpump_cop"

# Heizstab (Heating Rod)
CONF_SENSOR_HEATINGROD_POWER: Final = "sensor_heatingrod_power"
CONF_SENSOR_HEATINGROD_DAILY: Final = "sensor_heatingrod_daily"

# Wallbox (EV Charger)
CONF_SENSOR_WALLBOX_POWER: Final = "sensor_wallbox_power"
CONF_SENSOR_WALLBOX_DAILY: Final = "sensor_wallbox_daily"
CONF_SENSOR_WALLBOX_STATE: Final = "sensor_wallbox_state"

# Consumer Sensor Lists
CONSUMER_SENSORS: Final = [
    CONF_SENSOR_HEATPUMP_POWER,
    CONF_SENSOR_HEATPUMP_DAILY,
    CONF_SENSOR_HEATPUMP_COP,
    CONF_SENSOR_HEATINGROD_POWER,
    CONF_SENSOR_HEATINGROD_DAILY,
    CONF_SENSOR_WALLBOX_POWER,
    CONF_SENSOR_WALLBOX_DAILY,
    CONF_SENSOR_WALLBOX_STATE,
]

# Consumer default COP (if no COP sensor configured)
DEFAULT_HEATPUMP_COP: Final = 3.5

# =============================================================================
# Performance & Caching Constants
# =============================================================================

# Billing Calculator
RIEMANN_MAX_GAP_HOURS: Final = 4.0
BILLING_CACHE_TTL_SECONDS: Final = 60
LOG_BUFFER_MAX_SIZE: Final = 1000

# Power Sources Collector
POWER_COLLECTION_INTERVAL_SECONDS: Final = 300
POWER_DATA_RETENTION_DAYS: Final = 7

# API Caching
API_CACHE_TTL_SECONDS: Final = 30
MAX_HISTORY_HOURS: Final = 168  # 7 days

# Weather
WEATHER_HISTORY_DAYS: Final = 365
SUN_HOURS_RADIATION_THRESHOLD: Final = 100  # W/m²

# File Operations
FILE_RETRY_COUNT: Final = 3
FILE_RETRY_DELAY_SECONDS: Final = 0.1

# Scheduled Jobs
DAILY_AGGREGATION_HOUR: Final = 23
DAILY_AGGREGATION_MINUTE: Final = 55
DAILY_AGGREGATION_SECOND: Final = 0

# =============================================================================
# Monthly Tariff Feature Constants
# =============================================================================

MONTHLY_TARIFFS_FILE: Final = "monthly_tariffs.json"

# Configuration keys for tariff defaults
CONF_REFERENCE_PRICE: Final = "reference_price"
CONF_EEG_IMPORT_PRICE: Final = "eeg_import_price"
CONF_EEG_FEED_IN_PRICE: Final = "eeg_feed_in_price"
CONF_GRID_FEE_BASE: Final = "grid_fee_base"
CONF_GRID_FEE_SCALING: Final = "grid_fee_scaling"

# Default values for tariff configuration (in ct/kWh)
DEFAULT_REFERENCE_PRICE: Final = 26.0  # What electricity would cost without PV
DEFAULT_EEG_IMPORT_PRICE: Final = 18.0  # EEG community import price
DEFAULT_EEG_FEED_IN_PRICE: Final = 12.0  # EEG community feed-in price
DEFAULT_GRID_FEE_BASE: Final = 13.0  # Base grid fee at high consumption
DEFAULT_GRID_FEE_SCALING: Final = True  # Enable grid fee scaling by consumption

# Grid fee scaling thresholds (annual kWh)
GRID_FEE_THRESHOLD_HIGH: Final = 5000  # Above this: base fee
GRID_FEE_THRESHOLD_MEDIUM: Final = 2500  # Above this: 1.3x base
GRID_FEE_THRESHOLD_LOW: Final = 1000  # Above this: 1.6x base
# Below LOW: 2.0x base

# Scaling factors for grid fees
GRID_FEE_FACTOR_HIGH: Final = 1.0
GRID_FEE_FACTOR_MEDIUM: Final = 1.3
GRID_FEE_FACTOR_LOW: Final = 1.6
GRID_FEE_FACTOR_VERY_LOW: Final = 2.0

# =============================================================================
# Forecast Comparison Feature Constants
# =============================================================================

# External forecast entities (optional)
CONF_FORECAST_ENTITY_1: Final = "forecast_entity_1"
CONF_FORECAST_ENTITY_2: Final = "forecast_entity_2"
CONF_FORECAST_ENTITY_1_NAME: Final = "forecast_entity_1_name"
CONF_FORECAST_ENTITY_2_NAME: Final = "forecast_entity_2_name"

DEFAULT_FORECAST_ENTITY_1_NAME: Final = "Externe Prognose 1"
DEFAULT_FORECAST_ENTITY_2_NAME: Final = "Externe Prognose 2"

# Data file for external forecasts history
EXTERNAL_FORECASTS_HISTORY: Final = "external_forecasts_history.json"

# Forecast comparison settings
FORECAST_COMPARISON_RETENTION_DAYS: Final = 30
FORECAST_COMPARISON_CHART_DAYS: Final = 7

# Scheduled job times for forecast comparison
# Morning: Collect forecast values for today (before they change during the day)
FORECAST_MORNING_HOUR: Final = 8
FORECAST_MORNING_MINUTE: Final = 0
# Evening: Collect actual production for the day (after day is complete)
FORECAST_EVENING_HOUR: Final = 23
FORECAST_EVENING_MINUTE: Final = 50

# Chart generation time (after data collection)
FORECAST_CHART_HOUR: Final = 0
FORECAST_CHART_MINUTE: Final = 5
