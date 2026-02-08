# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Energy platform for Solar Forecast ML.

Provides async_get_solar_forecast() so HA Energy Dashboard can display
this integration under "Forecast production".
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import (
    CACHE_HOURLY_PREDICTIONS,
    CACHE_PREDICTIONS,
    DOMAIN,
    PRED_PREDICTION_KWH,
    PRED_PREDICTED_KWH,
    PRED_TARGET_DATE,
    PRED_TARGET_HOUR,
)

_LOGGER = logging.getLogger(__name__)


async def async_get_solar_forecast(
    hass: HomeAssistant, config_entry_id: str
) -> dict[str, Any] | None:
    """Return solar forecast data for the Energy Dashboard. @zara

    HA Energy Dashboard expects:
        {"wh_hours": {"ISO_TIMESTAMP": wh_value, ...}}

    Values must be in Watt-hours (Wh), timestamps in ISO format with timezone.
    We provide hourly forecasts for today and tomorrow.
    """
    if DOMAIN not in hass.data or config_entry_id not in hass.data[DOMAIN]:
        _LOGGER.debug("Solar Forecast ML not ready for energy forecast")
        return None

    coordinator = hass.data[DOMAIN][config_entry_id]
    tz = dt_util.get_time_zone(str(hass.config.time_zone)) or dt_util.UTC

    wh_hours: dict[str, float] = {}

    # --- Today: from coordinator cache ---
    today_predictions = _get_today_predictions(coordinator)
    today_str = dt_util.now().date().isoformat()
    _add_predictions_to_wh_hours(today_predictions, today_str, tz, wh_hours)

    # --- Tomorrow: from database ---
    tomorrow_str = (dt_util.now().date() + timedelta(days=1)).isoformat()
    tomorrow_predictions = await _get_tomorrow_predictions(coordinator, tomorrow_str)
    _add_predictions_to_wh_hours(tomorrow_predictions, tomorrow_str, tz, wh_hours)

    if not wh_hours:
        _LOGGER.debug("No hourly predictions available for energy forecast")
        return None

    _LOGGER.debug("Energy forecast: %d hourly entries provided", len(wh_hours))
    return {"wh_hours": wh_hours}


def _get_today_predictions(coordinator) -> list[dict[str, Any]]:
    """Get today's hourly predictions from coordinator cache. @zara"""
    hourly_data = getattr(coordinator, CACHE_HOURLY_PREDICTIONS, None)
    if not hourly_data or not isinstance(hourly_data, dict):
        return []
    return hourly_data.get(CACHE_PREDICTIONS, [])


async def _get_tomorrow_predictions(
    coordinator, tomorrow_str: str
) -> list[dict[str, Any]]:
    """Get tomorrow's hourly predictions from database. @zara"""
    try:
        if hasattr(coordinator, "data_manager") and coordinator.data_manager:
            predictions = await coordinator.data_manager.get_hourly_predictions(
                tomorrow_str, with_shadow=False, with_weather_actual=False
            )
            return predictions if predictions else []
    except Exception as e:
        _LOGGER.warning("Could not load tomorrow's predictions for energy forecast: %s", e)
    return []


def _add_predictions_to_wh_hours(
    predictions: list[dict[str, Any]],
    date_str: str,
    tz,
    wh_hours: dict[str, float],
) -> None:
    """Convert prediction list to wh_hours format. @zara

    Each prediction has target_date (YYYY-MM-DD) and target_hour (0-23).
    Values are in kWh, converted to Wh (* 1000).
    """
    for pred in predictions:
        pred_date = pred.get(PRED_TARGET_DATE)
        if pred_date != date_str:
            continue

        hour = pred.get(PRED_TARGET_HOUR)
        if hour is None:
            continue

        kwh = pred.get(PRED_PREDICTION_KWH) or pred.get(PRED_PREDICTED_KWH, 0.0)
        if kwh is None or kwh <= 0:
            continue

        try:
            dt = datetime(
                int(date_str[:4]),
                int(date_str[5:7]),
                int(date_str[8:10]),
                int(hour),
                0, 0,
                tzinfo=tz,
            )
            wh_hours[dt.isoformat()] = round(kwh * 1000, 1)
        except (ValueError, TypeError) as e:
            _LOGGER.debug("Skipping invalid prediction entry: %s", e)
