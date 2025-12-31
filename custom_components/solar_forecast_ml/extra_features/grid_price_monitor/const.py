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

# ============================================================================
# DOMAIN & VERSION CONSTANTS
# ============================================================================
DOMAIN = "grid_price_monitor"
NAME = "Grid Price Monitor"
VERSION = "1.0.0"

# ============================================================================
# PLATFORMS
# ============================================================================
PLATFORMS = [Platform.SENSOR, Platform.BINARY_SENSOR]

# ============================================================================
# API CONFIGURATION
# ============================================================================
AWATTAR_API_URL_DE = "https://api.awattar.de/v1/marketdata"
AWATTAR_API_URL_AT = "https://api.awattar.at/v1/marketdata"

API_TIMEOUT = 30  # seconds

# ============================================================================
# CONFIGURATION KEYS
# ============================================================================
CONF_COUNTRY = "country"
CONF_VAT_RATE = "vat_rate"
CONF_GRID_FEE = "grid_fee"
CONF_TAXES_FEES = "taxes_fees"
CONF_PROVIDER_MARKUP = "provider_markup"
CONF_MAX_PRICE = "max_price"

# Calibration option
CONF_USE_CALIBRATION = "use_calibration"
CONF_CALIBRATION_PRICE = "calibration_price"
CONF_BATTERY_POWER_SENSOR = "battery_power_sensor"

# ============================================================================
# DEFAULT VALUES
# ============================================================================
DEFAULT_COUNTRY = "DE"
DEFAULT_GRID_FEE = 8.0  # ct/kWh typical German grid fee (brutto)
DEFAULT_TAXES_FEES = 5.0  # ct/kWh taxes and fees (brutto)
DEFAULT_PROVIDER_MARKUP = 1.0  # ct/kWh provider margin (brutto)
DEFAULT_MAX_PRICE = 30.0  # ct/kWh threshold for "cheap" electricity

# ============================================================================
# VAT RATES
# ============================================================================
VAT_RATE_DE = 19  # 19% MwSt Germany (default)
VAT_RATE_AT = 20  # 20% MwSt Austria (default)
VAT_RATE_REDUCED_DE = 7  # 7% reduced VAT Germany

# VAT options for selector
VAT_OPTIONS = [
    {"value": 19, "label": "19% (Standard DE)"},
    {"value": 20, "label": "20% (Standard AT)"},
    {"value": 7, "label": "7% (Ermäßigt DE)"},
    {"value": 0, "label": "0% (Keine MwSt)"},
]

# ============================================================================
# COUNTRY OPTIONS
# ============================================================================
COUNTRY_OPTIONS = {
    "DE": "Germany",
    "AT": "Austria",
}

# ============================================================================
# UPDATE INTERVALS
# ============================================================================
UPDATE_INTERVAL = timedelta(minutes=5)
PRICE_FETCH_INTERVAL = timedelta(hours=1)

# ============================================================================
# SENSOR KEYS
# ============================================================================
SENSOR_SPOT_PRICE = "spot_price"
SENSOR_TOTAL_PRICE = "total_price"
SENSOR_SPOT_PRICE_NEXT_HOUR = "spot_price_next_hour"
SENSOR_TOTAL_PRICE_NEXT_HOUR = "total_price_next_hour"
SENSOR_CHEAPEST_HOUR_TODAY = "cheapest_hour_today"
SENSOR_MOST_EXPENSIVE_HOUR_TODAY = "most_expensive_hour_today"
SENSOR_AVERAGE_PRICE_TODAY = "average_price_today"
SENSOR_PRICES_TODAY = "prices_today"
SENSOR_PRICES_TOMORROW = "prices_tomorrow"

BINARY_SENSOR_CHEAP_ENERGY = "cheap_energy"

# Battery sensors
SENSOR_BATTERY_POWER = "battery_power"
SENSOR_BATTERY_CHARGED_TODAY = "battery_charged_today"
SENSOR_BATTERY_CHARGED_WEEK = "battery_charged_week"
SENSOR_BATTERY_CHARGED_MONTH = "battery_charged_month"

# ============================================================================
# ICONS
# ============================================================================
ICON_PRICE = "mdi:currency-eur"
ICON_SPOT = "mdi:chart-line"
ICON_CHEAP = "mdi:lightning-bolt"
ICON_CLOCK = "mdi:clock-outline"
ICON_AVERAGE = "mdi:chart-bar"
ICON_BATTERY = "mdi:battery-charging"
ICON_BATTERY_ENERGY = "mdi:battery-plus"
ICON_CALENDAR_TODAY = "mdi:calendar-today"
ICON_CALENDAR_TOMORROW = "mdi:calendar-arrow-right"

# ============================================================================
# UNITS
# ============================================================================
UNIT_CT_KWH = "ct/kWh"

# ============================================================================
# ATTRIBUTES
# ============================================================================
ATTR_FORECAST_TODAY = "forecast_today"
ATTR_FORECAST_TOMORROW = "forecast_tomorrow"
ATTR_NEXT_CHEAP_HOUR = "next_cheap_hour"
ATTR_CHEAP_HOURS_TODAY = "cheap_hours_today"
ATTR_CHEAP_HOURS_TOMORROW = "cheap_hours_tomorrow"
ATTR_PRICE_TREND = "price_trend"
ATTR_LAST_UPDATE = "last_update"
ATTR_DATA_SOURCE = "data_source"
