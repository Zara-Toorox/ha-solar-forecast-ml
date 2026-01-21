"""WebSocket API for realtime updates.

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
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback

if TYPE_CHECKING:
    from homeassistant.components.websocket_api import ActiveConnection

_LOGGER = logging.getLogger(__name__)

_subscriptions: dict[str, set] = {}


def _get_config_paths(hass: HomeAssistant) -> tuple[Path, Path]:
    """Get config paths dynamically from Home Assistant."""
    config_path = Path(hass.config.path())
    solar_path = config_path / "solar_forecast_ml"
    grid_path = config_path / "grid_price_monitor"
    return solar_path, grid_path


async def async_setup_websocket(hass: HomeAssistant) -> None:
    """Register WebSocket commands."""
    websocket_api.async_register_command(hass, websocket_subscribe_updates)
    websocket_api.async_register_command(hass, websocket_get_dashboard_data)
    _LOGGER.info("SFML Stats WebSocket commands registered")


@websocket_api.websocket_command(
    {
        "type": "sfml_stats_lite/subscribe",
        "interval": int,
    }
)
@websocket_api.async_response
async def websocket_subscribe_updates(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Subscribe to realtime updates."""
    msg_id = msg["id"]
    interval = msg.get("interval", 30)

    _LOGGER.debug("WebSocket Subscribe: msg_id=%s, interval=%s", msg_id, interval)

    connection.send_result(msg_id, {"subscribed": True, "interval": interval})

    async def send_updates():
        """Send periodic updates."""
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
        "type": "sfml_stats_lite/dashboard_data",
    }
)
@websocket_api.async_response
async def websocket_get_dashboard_data(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Get all dashboard data at once."""
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

        summaries_path = solar_path / "stats" / "daily_summaries.json"
        if summaries_path.exists():
            async with aiofiles.open(summaries_path, "r") as f:
                summaries = json.loads(await f.read())
                result["solar"]["summaries"] = summaries.get("summaries", [])[-7:]

        predictions_path = solar_path / "stats" / "hourly_predictions.json"
        if predictions_path.exists():
            async with aiofiles.open(predictions_path, "r") as f:
                predictions = json.loads(await f.read())
                result["solar"]["predictions"] = predictions.get("predictions", [])[-48:]

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
    """Collect realtime data for WebSocket updates."""
    solar_path, grid_path = _get_config_paths(hass)

    data = {
        "hour": datetime.now().hour,
        "minute": datetime.now().minute,
    }

    try:
        import aiofiles

        predictions_path = solar_path / "stats" / "hourly_predictions.json"
        if predictions_path.exists():
            async with aiofiles.open(predictions_path, "r") as f:
                predictions = json.loads(await f.read())
                now = datetime.now()
                current = next(
                    (p for p in predictions.get("predictions", [])
                     if p.get("target_date") == now.date().isoformat()
                     and p.get("target_hour") == now.hour),
                    None
                )
                if current:
                    data["solar"] = {
                        "prediction": current.get("prediction_kwh", 0),
                        "actual": current.get("actual_kwh"),
                        "clouds": current.get("weather_forecast", {}).get("clouds", 0),
                        "radiation": current.get("weather_forecast", {}).get("solar_radiation_wm2", 0),
                    }

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
