# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""User-friendly messages for Home Assistant logs @zara"""

from typing import Any


class UserMessages:
    """Centralized user messages @zara"""

    # AI Training @zara
    AI_LEARNING_PHASE = (
        "Learning phase: AI model is still collecting data. "
        "So far {samples} data points captured (range: {min_val:.2f}-{max_val:.2f} kWh). "
        "Predictions improve with each sunny day. "
        "System uses rule-based forecasting."
    )

    AI_LEARNING_PHASE_SIMPLE = (
        "Learning phase: AI model collecting data ({samples} points). "
        "Rule-based forecasting active - will improve automatically."
    )

    AI_NOT_ENOUGH_RESIDUALS = (
        "Learning phase: Not enough comparison data for AI training yet "
        "({count} of 10 required). System uses physics model."
    )

    AI_TRAINING_SUCCESS = (
        "AI training completed successfully. "
        "Accuracy: {accuracy:.1%}, data points: {samples}, duration: {duration:.1f}s"
    )

    AI_TRAINING_LAMBDA = (
        "AI model optimized (Lambda={lambda_val:.4f}, {samples} data points)"
    )

    AI_TSS_ZERO = (
        "Learning phase: Too little variation in production data. "
        "This is normal for new installations or cloudy periods. "
        "System waiting for more sunny days."
    )

    AI_LINALG_ERROR = (
        "AI calculation: Numerical problem during optimization. "
        "System uses alternative calculation."
    )

    # Weather Data @zara
    WEATHER_CACHE_UPDATING = "Weather data is being updated. No action required."

    WEATHER_CACHE_NOT_FOUND = "Weather cache not found. Weather data will be fetched."

    WEATHER_NO_FORECAST_DATA = (
        "No weather forecast for {date} in cache. "
        "Will be fetched automatically on next update."
    )

    WEATHER_API_ERROR = (
        "Weather service temporarily unreachable. "
        "Next attempt in {retry_minutes} minutes. "
        "Forecast based on last available data."
    )

    WEATHER_PRECISION_SKIP = "Weather precision calculation skipped - data being collected."

    WEATHER_FALLBACK_ACTIVE = (
        "Weather data temporarily unavailable. "
        "Forecast based on default values."
    )

    # Astronomy @zara
    ASTRONOMY_CACHE_BUILDING = (
        "Solar position data being calculated for the first time... "
        "This may take a few seconds."
    )

    ASTRONOMY_CACHE_NOT_FOUND = "Solar position data being recalculated."

    ASTRONOMY_CACHE_READY = "Solar position data calculated for {days} days."

    ASTRONOMY_CACHE_ERROR = (
        "Solar position calculation failed for {date}. "
        "System uses default values."
    )

    # Forecasts @zara
    FORECAST_TODAY_SAVED = "Daily forecast saved: {kwh:.2f} kWh (source: {source})"

    FORECAST_TOMORROW_SAVED = "Tomorrow's forecast saved: {kwh:.2f} kWh"

    FORECAST_LOCKED = "Forecast for {date} already set. No update needed."

    FORECAST_ADJUSTED = (
        "Forecast adjusted: Current production ({current:.2f} kWh) "
        "exceeds original forecast ({original:.2f} kWh). "
        "New forecast: {adjusted:.2f} kWh"
    )

    FORECAST_FALLBACK = "AI forecast unavailable. Rule-based forecast being used."

    FORECAST_ALL_FAILED = (
        "Forecast could not be created. "
        "Please check internet connection and weather service."
    )

    # Production Tracking @zara
    PRODUCTION_TRACKING_STARTED = "Production monitoring started for {entity}"

    PRODUCTION_TRACKING_DISABLED = (
        "No power entity configured. "
        "Production time tracking is disabled."
    )

    PRODUCTION_NEW_PEAK = "New daily record: {power_w:.0f}W at {time}"

    PRODUCTION_ALL_TIME_PEAK = "NEW ALL-TIME RECORD: {power_w:.0f}W on {date}"

    # Configuration @zara
    CONFIG_SOLAR_CAPACITY_ZERO = (
        "Configuration problem: PV system capacity is 0 or negative. "
        "Please correct in settings. "
        "System uses fallback value of 1.0 kWp."
    )

    CONFIG_POWER_ENTITY_MISSING = (
        "Setup incomplete: No power sensor configured. "
        "Please select in integration settings."
    )

    CONFIG_DIRECTORY_ERROR = (
        "No write permissions in configuration directory. "
        "Please check permissions for '{path}'."
    )

    # Initialization @zara
    INIT_COORDINATOR_READY = "Solar Forecast ready ({mode}, {capacity} kWp)"

    INIT_AI_READY = "AI model initialized and ready."

    INIT_AI_DISABLED = "AI functions disabled. Rule-based forecasting active."

    INIT_CLEAN_SLATE = "New installation detected - data structure being created..."

    INIT_CLEAN_SLATE_COMPLETE = "New installation complete. System is operational."

    INIT_DEPENDENCIES_MISSING = (
        "Some optional dependencies are missing. "
        "AI functions are limited."
    )

    # Scheduled Tasks @zara
    TASK_MORNING_ROUTINE_START = "Morning routine started for {date}"

    TASK_MORNING_ROUTINE_SUCCESS = "Morning routine completed successfully."

    TASK_MORNING_ROUTINE_RETRY = "Morning routine failed. Retry in {wait}s..."

    TASK_END_OF_DAY_START = "End-of-day routine started"

    TASK_END_OF_DAY_SUCCESS = "End-of-day routine successful. Accuracy: {accuracy:.1%}"

    TASK_MIDNIGHT_ROTATION = "Midnight rotation: Forecasts prepared for new day."

    # Sensors @zara
    SENSOR_UNAVAILABLE = (
        "Sensor '{entity}' not available. "
        "Please check if the sensor is configured correctly."
    )

    SENSOR_INVALID_VALUE = (
        "Invalid value from sensor '{entity}': {value}. "
        "Skipped for calculation."
    )

    # Shadow Detection @zara
    SHADOW_DETECTION_INIT = "Shadow analysis initialized."

    SHADOW_DETECTION_FALLBACK = "Shadow analysis: Alternative method being used."

    # General Errors @zara
    ERROR_UNEXPECTED = (
        "Unexpected error occurred. "
        "System attempting to continue. Details in debug log."
    )

    ERROR_SERVICE_UNAVAILABLE = (
        "Service temporarily unavailable. "
        "Automatic retry in progress."
    )

    @classmethod
    def format(cls, message_key: str, **kwargs: Any) -> str:
        """Format message with parameters @zara"""
        message_template = getattr(cls, message_key, None)
        if message_template is None:
            return message_key

        try:
            return message_template.format(**kwargs)
        except KeyError:
            return message_template

    @classmethod
    def get(cls, message_key: str) -> str:
        """Get message template @zara"""
        return getattr(cls, message_key, message_key)


def user_msg(key: str, **kwargs: Any) -> str:
    """Format user message @zara"""
    return UserMessages.format(key, **kwargs)
