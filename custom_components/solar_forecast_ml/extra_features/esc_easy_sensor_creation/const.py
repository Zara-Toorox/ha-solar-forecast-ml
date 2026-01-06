# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML - ESC
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************
"""Constants for ESC Easy Sensor Creation."""

DOMAIN = "esc_easy_sensor_creation"

# Sensor Types
SENSOR_TYPE_SUM = "sum"
SENSOR_TYPE_SQL = "sql_statistics"
SENSOR_TYPE_KWH_HELPER = "kwh_helper"  # Neuer Typ zur Unterscheidung im Flow
SENSOR_TYPE_DELTA = "delta"  # Neuer: Verlauf-Delta (z.B. heute vs. gestern)
SENSOR_TYPE_BATTERY = "battery_charge"  # Neuer: Akku Ladung/Entladung mit Filter + Riemann
SENSOR_TYPE_SFML = "sfml_sensors"  # SFML: Power + Yield + Daily Yield
SENSOR_TYPE_SFML_PANEL = "sfml_panel_groups"  # SFML Panelgruppen: 1-4 Strings mit Power + Yield + Daily

# SQL Statistics Types
SQL_STAT_AVG_TODAY = "avg_today"
SQL_STAT_AVG_MONTH = "avg_month"
SQL_STAT_AVG_YEAR = "avg_year"

SQL_STAT_MAX_TODAY = "max_today"
SQL_STAT_MAX_MONTH = "max_month"
SQL_STAT_MAX_YEAR = "max_year"

SQL_STAT_MIN_TODAY = "min_today"
SQL_STAT_MIN_MONTH = "min_month"
SQL_STAT_MIN_YEAR = "min_year"

# Delta Periods
DELTA_PERIOD_TODAY_YESTERDAY = "today_vs_yesterday"
DELTA_PERIOD_MONTH_PREV = "month_vs_prev"

# Battery Modes
BATTERY_MODE_CHARGE = "charge"  # Positive Werte (Ladung)
BATTERY_MODE_DISCHARGE = "discharge"  # Negative Werte (Entladung, absolut)

# Device Classes (erweitert um battery)
DEVICE_CLASS_POWER = "power"
DEVICE_CLASS_ENERGY = "energy"
DEVICE_CLASS_TEMPERATURE = "temperature"
DEVICE_CLASS_HUMIDITY = "humidity"
DEVICE_CLASS_BATTERY = "battery"
DEVICE_CLASS_MONETARY = "monetary"
DEVICE_CLASS_VOLTAGE = "voltage"
DEVICE_CLASS_CURRENT = "current"
DEVICE_CLASS_NONE = "none"

DEVICE_CLASSES = [
    DEVICE_CLASS_NONE,
    DEVICE_CLASS_POWER,
    DEVICE_CLASS_ENERGY,
    DEVICE_CLASS_TEMPERATURE,
    DEVICE_CLASS_HUMIDITY,
    DEVICE_CLASS_BATTERY,
    DEVICE_CLASS_MONETARY,
    DEVICE_CLASS_VOLTAGE,
    DEVICE_CLASS_CURRENT,
]

# Unit Fallbacks aus DeviceClass (für Stabilität)
DEVICE_CLASS_TO_UNIT = {
    "power": "W",
    "energy": "kWh",
    "temperature": "°C",
    "humidity": "%",
    "battery": "%",
    "monetary": "€",
    "voltage": "V",
    "current": "A",
    "none": None,
}