# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""WebSocket API for realtime updates. @zara"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback

if TYPE_CHECKING:
    from homeassistant.components.websocket_api import ActiveConnection

from ..readers.solar_reader import SolarDataReader

_LOGGER = logging.getLogger(__name__)

_subscriptions: dict[str, set] = {}


def _get_config_paths(hass: HomeAssistant) -> tuple[Path, Path]:
    """Get config paths dynamically from Home Assistant. @zara"""
    config_path = Path(hass.config.path())
    solar_path = config_path / "solar_forecast_ml"
    grid_path = config_path / "grid_price_monitor"
    return solar_path, grid_path


def _get_solar_reader(hass: HomeAssistant) -> SolarDataReader:
    """Get a SolarDataReader instance for database access. @zara"""
    config_path = Path(hass.config.path())
    return SolarDataReader(config_path)


async def async_setup_websocket(hass: HomeAssistant) -> None:
    """Register WebSocket commands. @zara"""
    websocket_api.async_register_command(hass, websocket_subscribe_updates)
    websocket_api.async_register_command(hass, websocket_get_dashboard_data)
    _LOGGER.info("SFML Stats WebSocket commands registered")


@websocket_api.websocket_command(
    {
        "type": "sfml_stats/subscribe",
        "interval": int,
    }
)
@websocket_api.async_response
async def websocket_subscribe_updates(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Subscribe to realtime updates. @zara"""
    msg_id = msg["id"]
    interval = msg.get("interval", 30)

    _LOGGER.debug("WebSocket Subscribe: msg_id=%s, interval=%s", msg_id, interval)

    connection.send_result(msg_id, {"subscribed": True, "interval": interval})

    async def send_updates():
        """Send periodic updates. @zara"""
        while True:
            try:
                data = await _get_realtime_data(hass)
                connection.send_message(
                    websocket_api.event_message(
                        msg_id,
                        {
                            "type": "update",
                            "timestamp": datetime.now().isoformat(),
                            "data": data,
                        }
                    )
                )
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                _LOGGER.error("Error sending WebSocket update: %s", e)
                break

    task = asyncio.create_task(send_updates())

    @callback
    def on_disconnect():
        task.cancel()

    connection.subscriptions[msg_id] = on_disconnect


@websocket_api.websocket_command(
    {
        "type": "sfml_stats/dashboard_data",
    }
)
@websocket_api.async_response
async def websocket_get_dashboard_data(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Get all dashboard data at once. @zara"""
    solar_path, grid_path = _get_config_paths(hass)

    result = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "solar": {},
        "prices": {},
        "kpis": {},
    }

    try:
        import aiofiles

        try:
            reader = _get_solar_reader(hass)
            week_ago = date.today() - timedelta(days=7)
            summaries = await reader.async_get_daily_summaries(
                start_date=week_ago,
                end_date=date.today()
            )
            result["solar"]["summaries"] = [
                {
                    "date": s.date.isoformat(),
                    "overall": {
                        "predicted_total_kwh": s.predicted_total_kwh,
                        "actual_total_kwh": s.actual_total_kwh,
                        "accuracy_percent": s.accuracy_percent,
                    }
                }
                for s in summaries
            ]
        except Exception as e:
            _LOGGER.error("Error loading summaries from database: %s", e)
            result["solar"]["summaries"] = []

        try:
            reader = _get_solar_reader(hass)
            two_days_ago = date.today() - timedelta(days=2)
            all_predictions = []
            current = two_days_ago
            while current <= date.today():
                day_predictions = await reader.async_get_hourly_predictions(target_date=current)
                for p in day_predictions:
                    all_predictions.append({
                        "target_datetime": p.target_datetime.isoformat(),
                        "target_hour": p.target_hour,
                        "target_date": p.target_date.isoformat(),
                        "prediction_kwh": p.prediction_kwh,
                        "actual_kwh": p.actual_kwh,
                    })
                current += timedelta(days=1)
            result["solar"]["predictions"] = all_predictions[-48:]
        except Exception as e:
            _LOGGER.error("Error loading predictions from database: %s", e)
            result["solar"]["predictions"] = []

        prices_path = grid_path / "data" / "price_history.json"
        if prices_path.exists():
            async with aiofiles.open(prices_path, "r") as f:
                prices = json.loads(await f.read())
                result["prices"]["history"] = prices.get("prices", [])[-48:]

        if result["solar"].get("summaries"):
            recent = result["solar"]["summaries"]
            result["kpis"]["week_production"] = sum(
                s.get("overall", {}).get("actual_total_kwh", 0) for s in recent
            )
            result["kpis"]["avg_accuracy"] = sum(
                s.get("overall", {}).get("accuracy_percent", 0) for s in recent
            ) / len(recent) if recent else 0

        if result["prices"].get("history"):
            prices_list = [p["price_net"] for p in result["prices"]["history"] if p.get("price_net")]
            result["kpis"]["price_current"] = prices_list[-1] if prices_list else 0
            result["kpis"]["price_min"] = min(prices_list) if prices_list else 0
            result["kpis"]["price_max"] = max(prices_list) if prices_list else 0

        connection.send_result(msg["id"], result)

    except Exception as e:
        _LOGGER.error("Error getting dashboard data: %s", e)
        connection.send_error(msg["id"], "error", str(e))


async def _get_realtime_data(hass: HomeAssistant) -> dict:
    """Collect realtime data for WebSocket updates. @zara"""
    solar_path, grid_path = _get_config_paths(hass)

    data = {
        "hour": datetime.now().hour,
        "minute": datetime.now().minute,
    }

    try:
        import aiofiles

        try:
            reader = _get_solar_reader(hass)
            now = datetime.now()
            predictions = await reader.async_get_hourly_predictions(target_date=now.date())
            current = next((p for p in predictions if p.target_hour == now.hour), None)

            if current:
                data["solar"] = {
                    "prediction": current.prediction_kwh,
                    "actual": current.actual_kwh,
                    "clouds": current.clouds,
                    "radiation": current.solar_radiation,
                }
        except Exception as e:
            _LOGGER.error("Error loading realtime prediction from database: %s", e)

        prices_path = grid_path / "data" / "price_history.json"
        if prices_path.exists():
            async with aiofiles.open(prices_path, "r") as f:
                prices = json.loads(await f.read())
                if prices.get("prices"):
                    latest = prices["prices"][-1]
                    data["price"] = {
                        "current": latest.get("price_net", 0),
                        "hour": latest.get("hour"),
                    }

        weather_path = solar_path / "stats" / "hourly_weather_actual.json"
        if weather_path.exists():
            async with aiofiles.open(weather_path, "r") as f:
                weather = json.loads(await f.read())
                today = datetime.now().date().isoformat()
                hour = str(datetime.now().hour)
                if today in weather.get("hourly_data", {}) and hour in weather["hourly_data"][today]:
                    data["weather"] = weather["hourly_data"][today][hour]

    except Exception as e:
        _LOGGER.error("Error getting realtime data: %s", e)

    return data
