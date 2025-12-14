"""REST API views for SFML Stats Dashboard. @zara

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

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web
from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant

from ..const import (
    DOMAIN,
    CONF_SENSOR_SOLAR_POWER,
    CONF_SENSOR_SOLAR_TO_HOUSE,
    CONF_SENSOR_SOLAR_TO_BATTERY,
    CONF_SENSOR_BATTERY_TO_HOUSE,
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
    CONF_WEATHER_ENTITY,
    CONF_SENSOR_PANEL1_POWER,
    CONF_SENSOR_PANEL1_MAX_TODAY,
    CONF_SENSOR_PANEL2_POWER,
    CONF_SENSOR_PANEL2_MAX_TODAY,
    CONF_SENSOR_PANEL3_POWER,
    CONF_SENSOR_PANEL3_MAX_TODAY,
    CONF_SENSOR_PANEL4_POWER,
    CONF_SENSOR_PANEL4_MAX_TODAY,
    CONF_PANEL1_NAME,
    CONF_PANEL2_NAME,
    CONF_PANEL3_NAME,
    CONF_PANEL4_NAME,
    DEFAULT_PANEL1_NAME,
    DEFAULT_PANEL2_NAME,
    DEFAULT_PANEL3_NAME,
    DEFAULT_PANEL4_NAME,
)

if TYPE_CHECKING:
    from aiohttp.web import Request, Response

_LOGGER = logging.getLogger(__name__)

SOLAR_PATH: Path | None = None
GRID_PATH: Path | None = None
HASS: HomeAssistant | None = None


async def async_setup_views(hass: HomeAssistant) -> None:
    """Register all API views. @zara"""
    global SOLAR_PATH, GRID_PATH, HASS

    HASS = hass
    config_path = Path(hass.config.path())
    SOLAR_PATH = config_path / "solar_forecast_ml"
    GRID_PATH = config_path / "grid_price_monitor"

    _LOGGER.debug("SFML Stats paths: Solar=%s, Grid=%s", SOLAR_PATH, GRID_PATH)

    hass.http.register_view(DashboardView())
    hass.http.register_view(SolarDataView())
    hass.http.register_view(PriceDataView())
    hass.http.register_view(SummaryDataView())
    hass.http.register_view(RealtimeDataView())
    hass.http.register_view(StaticFilesView())
    hass.http.register_view(EnergyFlowView())
    hass.http.register_view(StatisticsView())
    hass.http.register_view(BillingDataView())
    hass.http.register_view(ExportSolarAnalyticsView())
    hass.http.register_view(ExportBatteryAnalyticsView())
    hass.http.register_view(ExportHouseAnalyticsView())
    hass.http.register_view(ExportGridAnalyticsView())
    hass.http.register_view(WeatherHistoryView())
    hass.http.register_view(ExportWeatherAnalyticsView())
    hass.http.register_view(PowerSourcesHistoryView())
    hass.http.register_view(ExportPowerSourcesView())

    _LOGGER.info("SFML Stats API views registered")


async def _read_json_file(path: Path | None) -> dict | None:
    """Read a JSON file asynchronously. @zara"""
    if path is None:
        _LOGGER.warning("Path is None - was async_setup_views called?")
        return None
    if not path.exists():
        _LOGGER.debug("File not found: %s", path)
        return None
    try:
        import aiofiles
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)
            _LOGGER.debug("Successfully loaded: %s (%d bytes)", path, len(content))
            return data
    except Exception as e:
        _LOGGER.error("Error reading %s: %s", path, e)
        return None


class DashboardView(HomeAssistantView):
    """Main view serving the Vue.js app. @zara"""

    url = "/api/sfml_stats/dashboard"
    name = "api:sfml_stats:dashboard"
    requires_auth = False
    cors_allowed = True

    async def get(self, request: Request) -> Response:
        """Return the dashboard HTML page. @zara"""
        frontend_path = Path(__file__).parent.parent / "frontend" / "dist" / "index.html"

        if not frontend_path.exists():
            html_content = self._get_fallback_html()
        else:
            import aiofiles
            async with aiofiles.open(frontend_path, "r", encoding="utf-8") as f:
                html_content = await f.read()

        return web.Response(
            text=html_content,
            content_type="text/html",
            headers={
                "X-Frame-Options": "SAMEORIGIN",
                "Content-Security-Policy": "frame-ancestors 'self'",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )

    def _get_fallback_html(self) -> str:
        """Return fallback HTML when build is not present. @zara"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>SFML Stats Dashboard</title>
    <style>
        body {
            background: #0a0a1a;
            color: #fff;
            font-family: system-ui;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .message { text-align: center; }
        h1 { color: #00ffff; }
    </style>
</head>
<body>
    <div class="message">
        <h1>SFML Stats Dashboard</h1>
        <p>Frontend wird geladen...</p>
        <p style="color: #666;">Falls diese Meldung bleibt, wurde das Frontend noch nicht gebaut.</p>
    </div>
</body>
</html>"""


class StaticFilesView(HomeAssistantView):
    """Serve static files (JS, CSS, Assets). @zara"""

    url = "/api/sfml_stats/assets/{filename:.*}"
    name = "api:sfml_stats:assets"
    requires_auth = False

    async def get(self, request: Request, filename: str) -> Response:
        """Return a static file. @zara"""
        frontend_path = Path(__file__).parent.parent / "frontend" / "dist" / "assets" / filename

        if not frontend_path.exists():
            return web.Response(status=404, text="Not found")

        content_type = "application/octet-stream"
        if filename.endswith(".js"):
            content_type = "application/javascript"
        elif filename.endswith(".css"):
            content_type = "text/css"
        elif filename.endswith(".svg"):
            content_type = "image/svg+xml"
        elif filename.endswith(".png"):
            content_type = "image/png"
        elif filename.endswith(".woff2"):
            content_type = "font/woff2"

        import aiofiles
        async with aiofiles.open(frontend_path, "rb") as f:
            content = await f.read()

        return web.Response(body=content, content_type=content_type)


class SolarDataView(HomeAssistantView):
    """API for solar data. @zara"""

    url = "/api/sfml_stats/solar"
    name = "api:sfml_stats:solar"
    requires_auth = False

    async def get(self, request: Request) -> Response:
        """Return solar data. @zara"""
        days = int(request.query.get("days", 7))
        include_hourly = request.query.get("hourly", "true").lower() == "true"

        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "data": {},
        }

        forecasts_data = await _read_json_file(SOLAR_PATH / "stats" / "daily_forecasts.json")
        if forecasts_data and "history" in forecasts_data and len(forecasts_data["history"]) > 0:
            cutoff = date.today() - timedelta(days=days)
            result["data"]["daily"] = [
                {
                    "date": h["date"],
                    "overall": {
                        "predicted_total_kwh": h.get("predicted_kwh", 0),
                        "actual_total_kwh": h.get("actual_kwh", 0),
                        "accuracy_percent": h.get("accuracy", 0),
                        "peak_kwh": (h.get("peak_power_w", 0) or 0) / 1000,
                    }
                }
                for h in forecasts_data["history"]
                if date.fromisoformat(h["date"]) >= cutoff
            ]
        else:
            summaries = await _read_json_file(SOLAR_PATH / "stats" / "daily_summaries.json")
            if summaries and "summaries" in summaries:
                cutoff = date.today() - timedelta(days=days)
                result["data"]["daily"] = [
                    s for s in summaries["summaries"]
                    if date.fromisoformat(s["date"]) >= cutoff
                ]

        if include_hourly:
            predictions = await _read_json_file(SOLAR_PATH / "stats" / "hourly_predictions.json")
            if predictions and "predictions" in predictions:
                cutoff = date.today() - timedelta(days=days)
                result["data"]["hourly"] = [
                    p for p in predictions["predictions"]
                    if date.fromisoformat(p.get("target_date", "1970-01-01")) >= cutoff
                ]

        weather = await _read_json_file(SOLAR_PATH / "stats" / "hourly_weather_actual.json")
        if weather and "hourly_data" in weather:
            cutoff = date.today() - timedelta(days=days)
            result["data"]["weather"] = {
                k: v for k, v in weather["hourly_data"].items()
                if date.fromisoformat(k) >= cutoff
            }

        weather_corrected = await _read_json_file(SOLAR_PATH / "stats" / "weather_forecast_corrected.json")
        if weather_corrected and "forecast" in weather_corrected:
            cutoff = date.today() - timedelta(days=days)
            result["data"]["weather_corrected"] = {
                k: v for k, v in weather_corrected["forecast"].items()
                if date.fromisoformat(k) >= cutoff
            }

        ml_state = await _read_json_file(SOLAR_PATH / "ml" / "model_state.json")
        if ml_state:
            result["data"]["ml_state"] = ml_state

        if forecasts_data and "today" in forecasts_data:
            forecast_day = forecasts_data["today"].get("forecast_day", {})
            forecast_tomorrow = forecasts_data["today"].get("forecast_tomorrow", {})
            forecast_day_after = forecasts_data["today"].get("forecast_day_after_tomorrow", {})
            result["data"]["forecasts"] = {
                "today": {
                    "prediction_kwh": forecast_day.get("prediction_kwh"),
                    "prediction_kwh_display": forecast_day.get("prediction_kwh_display"),
                },
                "tomorrow": {
                    "date": forecast_tomorrow.get("date"),
                    "prediction_kwh": forecast_tomorrow.get("prediction_kwh"),
                    "prediction_kwh_display": forecast_tomorrow.get("prediction_kwh_display"),
                },
                "day_after_tomorrow": {
                    "date": forecast_day_after.get("date"),
                    "prediction_kwh": forecast_day_after.get("prediction_kwh"),
                    "prediction_kwh_display": forecast_day_after.get("prediction_kwh_display"),
                },
            }

            if "history" in forecasts_data:
                result["data"]["history"] = [
                    {
                        "date": h["date"],
                        "predicted_kwh": h.get("predicted_kwh", 0),
                        "actual_kwh": h.get("actual_kwh", 0),
                        "accuracy": h.get("accuracy", 0),
                        "peak_power_w": h.get("peak_power_w"),
                        "peak_at": h.get("peak_at"),
                        "consumption_kwh": h.get("consumption_kwh", 0),
                        "production_hours": h.get("production_hours"),
                    }
                    for h in forecasts_data["history"]
                ]

            if "statistics" in forecasts_data:
                result["data"]["statistics"] = forecasts_data["statistics"]

        astronomy = await _read_json_file(SOLAR_PATH / "stats" / "astronomy_cache.json")
        if astronomy and "days" in astronomy:
            cutoff_str = (date.today() - timedelta(days=days)).isoformat()
            result["data"]["astronomy"] = {
                k: {
                    "daylight_hours": v.get("daylight_hours"),
                    "sunrise": v.get("sunrise_local"),
                    "sunset": v.get("sunset_local"),
                }
                for k, v in astronomy["days"].items()
                if k >= cutoff_str
            }

        return web.json_response(result)


class PriceDataView(HomeAssistantView):
    """API for electricity price data. @zara"""

    url = "/api/sfml_stats/prices"
    name = "api:sfml_stats:prices"
    requires_auth = False

    async def get(self, request: Request) -> Response:
        """Return price data. @zara"""
        days = int(request.query.get("days", 7))

        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "data": {},
        }

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        price_cache = await _read_json_file(GRID_PATH / "data" / "price_cache.json")
        if price_cache and "prices" in price_cache:
            result["data"]["prices"] = [
                {
                    "timestamp": p["timestamp"],
                    "date": p.get("date"),
                    "hour": p.get("hour"),
                    "price_net": p.get("price", p.get("price_net", 0)),
                    "price_total": p.get("total_price", 0),
                }
                for p in price_cache["prices"]
                if datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00")) >= cutoff
            ]
        else:
            prices = await _read_json_file(GRID_PATH / "data" / "price_history.json")
            if prices and "prices" in prices:
                result["data"]["prices"] = [
                    {
                        "timestamp": p["timestamp"],
                        "date": p.get("date"),
                        "hour": p.get("hour"),
                        "price_net": p.get("price_net", 0),
                        "price_total": None,
                    }
                    for p in prices["prices"]
                    if datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00")) >= cutoff
                ]

        stats = await _read_json_file(GRID_PATH / "data" / "statistics.json")
        if stats:
            result["data"]["statistics"] = stats

        return web.json_response(result)


class SummaryDataView(HomeAssistantView):
    """API for aggregated dashboard data. @zara"""

    url = "/api/sfml_stats/summary"
    name = "api:sfml_stats:summary"
    requires_auth = False

    async def get(self, request: Request) -> Response:
        """Return a summary for the dashboard. @zara"""
        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "kpis": {},
            "today": {},
            "week": {},
        }

        summaries = await _read_json_file(SOLAR_PATH / "stats" / "daily_summaries.json")
        today = date.today()
        week_ago = today - timedelta(days=7)

        if summaries and "summaries" in summaries:
            today_data = next(
                (s for s in summaries["summaries"] if s["date"] == today.isoformat()),
                None
            )
            if today_data:
                result["today"] = {
                    "production": today_data.get("overall", {}).get("actual_total_kwh", 0),
                    "forecast": today_data.get("overall", {}).get("predicted_total_kwh", 0),
                    "accuracy": today_data.get("overall", {}).get("accuracy_percent", 0),
                    "peak_hour": today_data.get("overall", {}).get("peak_hour"),
                    "peak_kwh": today_data.get("overall", {}).get("peak_kwh", 0),
                }

            week_data = [
                s for s in summaries["summaries"]
                if date.fromisoformat(s["date"]) >= week_ago
            ]
            if week_data:
                result["week"] = {
                    "total_production": sum(
                        s.get("overall", {}).get("actual_total_kwh", 0) for s in week_data
                    ),
                    "total_forecast": sum(
                        s.get("overall", {}).get("predicted_total_kwh", 0) for s in week_data
                    ),
                    "avg_accuracy": sum(
                        s.get("overall", {}).get("accuracy_percent", 0) for s in week_data
                    ) / len(week_data) if week_data else 0,
                    "days_count": len(week_data),
                }

        prices = await _read_json_file(GRID_PATH / "data" / "price_history.json")
        if prices and "prices" in prices:
            recent_prices = [p["price_net"] for p in prices["prices"][-48:] if p.get("price_net")]
            if recent_prices:
                result["kpis"]["price_current"] = recent_prices[-1] if recent_prices else 0
                result["kpis"]["price_avg"] = sum(recent_prices) / len(recent_prices)
                result["kpis"]["price_min"] = min(recent_prices)
                result["kpis"]["price_max"] = max(recent_prices)

        ml_state = await _read_json_file(SOLAR_PATH / "ml" / "model_state.json")
        if ml_state:
            result["kpis"]["ml_accuracy"] = ml_state.get("current_accuracy", 0)
            result["kpis"]["ml_training_days"] = ml_state.get("training_days", 0)

        def extract_time(iso_string: str | None) -> str | None:
            """Extract HH:MM from ISO string. @zara"""
            if not iso_string:
                return None
            try:
                if "T" in iso_string:
                    time_part = iso_string.split("T")[1]
                    return time_part[:5]
                return iso_string[:5]
            except Exception:
                return None

        astronomy = await _read_json_file(SOLAR_PATH / "stats" / "astronomy_cache.json")
        today_str = date.today().isoformat()
        today_astronomy = {}
        if astronomy and "days" in astronomy:
            today_astronomy = astronomy["days"].get(today_str, {})

        forecasts = await _read_json_file(SOLAR_PATH / "stats" / "daily_forecasts.json")
        if forecasts and "today" in forecasts:
            production_time = forecasts["today"].get("production_time", {})
            start_time = production_time.get("start_time")
            end_time = production_time.get("end_time")

            if not start_time:
                start_time = today_astronomy.get("production_window_start")
            if not end_time:
                end_time = today_astronomy.get("production_window_end")

            result["production_time"] = {
                "active": production_time.get("active", False),
                "start_time": extract_time(start_time),
                "end_time": extract_time(end_time),
                "duration_seconds": production_time.get("duration_seconds", 0),
            }

        if today_astronomy:
            result["sun_times"] = {
                "sunrise": extract_time(today_astronomy.get("sunrise_local")),
                "sunset": extract_time(today_astronomy.get("sunset_local")),
            }

        return web.json_response(result)


class RealtimeDataView(HomeAssistantView):
    """API for realtime data (polled by frontend). @zara"""

    url = "/api/sfml_stats/realtime"
    name = "api:sfml_stats:realtime"
    requires_auth = False

    async def get(self, request: Request) -> Response:
        """Return current realtime data. @zara"""
        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "current_hour": datetime.now().hour,
            "data": {},
        }

        predictions = await _read_json_file(SOLAR_PATH / "stats" / "hourly_predictions.json")
        if predictions and "predictions" in predictions:
            now = datetime.now()
            current = next(
                (p for p in predictions["predictions"]
                 if p.get("target_date") == now.date().isoformat()
                 and p.get("target_hour") == now.hour),
                None
            )
            if current:
                result["data"]["solar"] = {
                    "prediction_kwh": current.get("prediction_kwh", 0),
                    "actual_kwh": current.get("actual_kwh"),
                    "weather": current.get("weather_forecast", {}),
                    "astronomy": current.get("astronomy", {}),
                }

        prices = await _read_json_file(GRID_PATH / "data" / "price_history.json")
        if prices and "prices" in prices:
            now = datetime.now()
            current_price = next(
                (p for p in reversed(prices["prices"])
                 if datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00")).hour == now.hour),
                None
            )
            if current_price:
                result["data"]["price"] = {
                    "current": current_price.get("price_net", 0),
                    "hour": current_price.get("hour"),
                }

        weather = await _read_json_file(SOLAR_PATH / "stats" / "hourly_weather_actual.json")
        if weather and "hourly_data" in weather:
            today_str = date.today().isoformat()
            hour_str = str(datetime.now().hour)
            if today_str in weather["hourly_data"]:
                current_weather = weather["hourly_data"][today_str].get(hour_str, {})
                result["data"]["weather_actual"] = current_weather

        return web.json_response(result)


def _get_config() -> dict[str, Any]:
    """Get current configuration from the first config entry. @zara"""
    if HASS is None:
        _LOGGER.debug("_get_config: HASS is None")
        return {}

    entries = HASS.data.get(DOMAIN, {})
    _LOGGER.debug("_get_config: entries=%s", entries)

    for entry_id, entry_data in entries.items():
        if isinstance(entry_data, dict) and "config" in entry_data:
            _LOGGER.debug("_get_config: Found config in entry %s: %s", entry_id, entry_data["config"])
            return entry_data["config"]

    config_entries = HASS.config_entries.async_entries(DOMAIN)
    if config_entries:
        entry = config_entries[0]
        _LOGGER.debug("_get_config: Fallback to ConfigEntry.data: %s", dict(entry.data))
        return dict(entry.data)

    _LOGGER.debug("_get_config: No config found")
    return {}


def _get_sensor_value(entity_id: str | None) -> float | None:
    """Read current value from a sensor. @zara"""
    if not entity_id or not HASS:
        return None

    state = HASS.states.get(entity_id)
    if state is None or state.state in ("unknown", "unavailable"):
        return None

    try:
        return float(state.state)
    except (ValueError, TypeError):
        return None


def _get_weather_data(entity_id: str | None) -> dict[str, Any] | None:
    """Read weather data from a Home Assistant weather entity. @zara"""
    if not entity_id or not HASS:
        return None

    state = HASS.states.get(entity_id)
    if state is None or state.state in ("unknown", "unavailable"):
        return None

    attrs = state.attributes
    return {
        "state": state.state,  # z.B. "sunny", "cloudy", etc.
        "temperature": attrs.get("temperature"),
        "humidity": attrs.get("humidity"),
        "wind_speed": attrs.get("wind_speed"),
        "wind_bearing": attrs.get("wind_bearing"),
        "pressure": attrs.get("pressure"),
        "cloud_coverage": attrs.get("cloud_coverage"),
        "visibility": attrs.get("visibility"),
        "uv_index": attrs.get("uv_index"),
    }


class EnergyFlowView(HomeAssistantView):
    """API for energy flow data from Home Assistant sensors. @zara"""

    url = "/api/sfml_stats/energy_flow"
    name = "api:sfml_stats:energy_flow"
    requires_auth = False

    async def get(self, request: Request) -> Response:
        """Return current energy flow data. @zara"""
        config = _get_config()

        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "flows": {
                "solar_power": _get_sensor_value(config.get(CONF_SENSOR_SOLAR_POWER)),
                "solar_to_house": _get_sensor_value(config.get(CONF_SENSOR_SOLAR_TO_HOUSE)),
                "solar_to_battery": _get_sensor_value(config.get(CONF_SENSOR_SOLAR_TO_BATTERY)),
                "battery_to_house": _get_sensor_value(config.get(CONF_SENSOR_BATTERY_TO_HOUSE)),
                "grid_to_house": _get_sensor_value(config.get(CONF_SENSOR_GRID_TO_HOUSE)),
                "grid_to_battery": _get_sensor_value(config.get(CONF_SENSOR_GRID_TO_BATTERY)),
                "house_to_grid": _get_sensor_value(config.get(CONF_SENSOR_HOUSE_TO_GRID)),
            },
            "battery": {
                "soc": _get_sensor_value(config.get(CONF_SENSOR_BATTERY_SOC)),
                "power": _get_sensor_value(config.get(CONF_SENSOR_BATTERY_POWER)),
            },
            "home": {
                "consumption": _get_sensor_value(config.get(CONF_SENSOR_HOME_CONSUMPTION)),
            },
            "statistics": {
                "solar_yield_daily": _get_sensor_value(config.get(CONF_SENSOR_SOLAR_YIELD_DAILY)),
                "grid_import_daily": _get_sensor_value(config.get(CONF_SENSOR_GRID_IMPORT_DAILY)),
                "grid_import_yearly": _get_sensor_value(config.get(CONF_SENSOR_GRID_IMPORT_YEARLY)),
                "battery_charge_solar_daily": _get_sensor_value(config.get(CONF_SENSOR_BATTERY_CHARGE_SOLAR_DAILY)),
                "battery_charge_grid_daily": _get_sensor_value(config.get(CONF_SENSOR_BATTERY_CHARGE_GRID_DAILY)),
                "price_total": _get_sensor_value(config.get(CONF_SENSOR_PRICE_TOTAL)),
            },
            "configured_sensors": {
                "solar_power": config.get(CONF_SENSOR_SOLAR_POWER),
                "solar_to_house": config.get(CONF_SENSOR_SOLAR_TO_HOUSE),
                "solar_to_battery": config.get(CONF_SENSOR_SOLAR_TO_BATTERY),
                "battery_to_house": config.get(CONF_SENSOR_BATTERY_TO_HOUSE),
                "grid_to_house": config.get(CONF_SENSOR_GRID_TO_HOUSE),
                "grid_to_battery": config.get(CONF_SENSOR_GRID_TO_BATTERY),
                "house_to_grid": config.get(CONF_SENSOR_HOUSE_TO_GRID),
                "battery_soc": config.get(CONF_SENSOR_BATTERY_SOC),
                "home_consumption": config.get(CONF_SENSOR_HOME_CONSUMPTION),
                "solar_yield_daily": config.get(CONF_SENSOR_SOLAR_YIELD_DAILY),
                "weather_entity": config.get(CONF_WEATHER_ENTITY),
            },
            "panels": self._get_panel_data(config),
            "weather_ha": _get_weather_data(config.get(CONF_WEATHER_ENTITY)),
            "sun_position": await self._get_sun_position(),
            "current_price": await self._get_current_price(),
        }

        return web.json_response(result)

    async def _get_current_price(self) -> dict[str, Any] | None:
        """Read current electricity price from price_cache.json. @zara"""
        price_cache = await _read_json_file(GRID_PATH / "data" / "price_cache.json")
        if not price_cache or "prices" not in price_cache:
            return None

        today_str = date.today().isoformat()
        current_hour = datetime.now().hour

        for p in price_cache["prices"]:
            if p.get("date") == today_str and p.get("hour") == current_hour:
                return {
                    "total_price": p.get("total_price"),
                    "net_price": p.get("price"),
                    "hour": current_hour,
                }
        return None

    async def _get_sun_position(self) -> dict[str, Any] | None:
        """Read current sun position from astronomy_cache.json. @zara"""
        astronomy = await _read_json_file(SOLAR_PATH / "stats" / "astronomy_cache.json")
        if not astronomy or "days" not in astronomy:
            return None

        today_str = date.today().isoformat()
        today_data = astronomy["days"].get(today_str)
        if not today_data:
            return None

        current_hour = datetime.now().hour
        hourly = today_data.get("hourly", {})
        current_hourly = hourly.get(str(current_hour), {})

        azimuth = current_hourly.get("azimuth_deg", 0)
        direction = self._azimuth_to_direction(azimuth)

        return {
            "elevation_deg": current_hourly.get("elevation_deg"),
            "azimuth_deg": azimuth,
            "direction": direction,
            "sunrise": self._extract_time(today_data.get("sunrise_local")),
            "sunset": self._extract_time(today_data.get("sunset_local")),
            "solar_noon": self._extract_time(today_data.get("solar_noon_local")),
            "daylight_hours": today_data.get("daylight_hours"),
        }

    def _azimuth_to_direction(self, azimuth: float) -> str:
        """Convert azimuth degrees to cardinal direction. @zara"""
        if azimuth is None:
            return "—"
        azimuth = azimuth % 360
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = int((azimuth + 22.5) / 45) % 8
        return directions[index]

    def _extract_time(self, iso_string: str | None) -> str | None:
        """Extract HH:MM from ISO string. @zara"""
        if not iso_string:
            return None
        try:
            if "T" in iso_string:
                time_part = iso_string.split("T")[1]
                return time_part[:5]
            return iso_string[:5]
        except Exception:
            return None

    def _get_panel_data(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Read panel data from configured sensors. @zara"""
        panels = []

        if config.get(CONF_SENSOR_PANEL1_POWER):
            panels.append({
                "id": 1,
                "name": config.get(CONF_PANEL1_NAME, DEFAULT_PANEL1_NAME),
                "power": _get_sensor_value(config.get(CONF_SENSOR_PANEL1_POWER)),
                "max_today": _get_sensor_value(config.get(CONF_SENSOR_PANEL1_MAX_TODAY)),
            })

        if config.get(CONF_SENSOR_PANEL2_POWER):
            panels.append({
                "id": 2,
                "name": config.get(CONF_PANEL2_NAME, DEFAULT_PANEL2_NAME),
                "power": _get_sensor_value(config.get(CONF_SENSOR_PANEL2_POWER)),
                "max_today": _get_sensor_value(config.get(CONF_SENSOR_PANEL2_MAX_TODAY)),
            })

        if config.get(CONF_SENSOR_PANEL3_POWER):
            panels.append({
                "id": 3,
                "name": config.get(CONF_PANEL3_NAME, DEFAULT_PANEL3_NAME),
                "power": _get_sensor_value(config.get(CONF_SENSOR_PANEL3_POWER)),
                "max_today": _get_sensor_value(config.get(CONF_SENSOR_PANEL3_MAX_TODAY)),
            })

        if config.get(CONF_SENSOR_PANEL4_POWER):
            panels.append({
                "id": 4,
                "name": config.get(CONF_PANEL4_NAME, DEFAULT_PANEL4_NAME),
                "power": _get_sensor_value(config.get(CONF_SENSOR_PANEL4_POWER)),
                "max_today": _get_sensor_value(config.get(CONF_SENSOR_PANEL4_MAX_TODAY)),
            })

        return panels


class StatisticsView(HomeAssistantView):
    """API for statistics data from JSON files. @zara"""

    url = "/api/sfml_stats/statistics"
    name = "api:sfml_stats:statistics"
    requires_auth = False

    async def get(self, request: Request) -> Response:
        """Return statistics data from Solar Forecast ML JSON files. @zara"""
        # Normal statistics response
        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "peaks": {},
            "production": {},
            "statistics": {},
        }

        forecasts = await _read_json_file(SOLAR_PATH / "stats" / "daily_forecasts.json")
        if forecasts:
            today_data = forecasts.get("today", {})
            peak_today = today_data.get("peak_today", {})
            result["peaks"]["today"] = {
                "power_w": peak_today.get("power_w"),
                "at": peak_today.get("at"),
            }

            stats = forecasts.get("statistics", {})
            all_time_peak = stats.get("all_time_peak", {})
            result["peaks"]["all_time"] = {
                "power_w": all_time_peak.get("power_w"),
                "date": all_time_peak.get("date"),
                "at": all_time_peak.get("at"),
            }

            forecast_day_data = today_data.get("forecast_day", {})
            result["production"]["today"] = {
                "forecast_kwh": forecast_day_data.get("prediction_kwh"),
                "forecast_kwh_display": forecast_day_data.get("prediction_kwh_display"),
                "yield_kwh": today_data.get("yield_today", {}).get("kwh"),
            }

            forecast_tomorrow_data = today_data.get("forecast_tomorrow", {})
            result["production"]["tomorrow"] = {
                "forecast_kwh": forecast_tomorrow_data.get("prediction_kwh"),
                "forecast_kwh_display": forecast_tomorrow_data.get("prediction_kwh_display"),
            }

            predictions = await _read_json_file(SOLAR_PATH / "stats" / "hourly_predictions.json")
            result["best_hour"] = {"hour": None, "prediction_kwh": None}
            if predictions and "predictions" in predictions:
                today_str = date.today().isoformat()
                today_preds = [
                    p for p in predictions["predictions"]
                    if p.get("target_date") == today_str and p.get("prediction_kwh")
                ]
                if today_preds:
                    best = max(today_preds, key=lambda x: x.get("prediction_kwh", 0))
                    result["best_hour"] = {
                        "hour": best.get("target_hour"),
                        "prediction_kwh": best.get("prediction_kwh"),
                    }

            result["statistics"]["current_week"] = stats.get("current_week", {})
            result["statistics"]["current_month"] = stats.get("current_month", {})
            result["statistics"]["last_7_days"] = stats.get("last_7_days", {})
            result["statistics"]["last_30_days"] = stats.get("last_30_days", {})
            result["statistics"]["last_365_days"] = stats.get("last_365_days", {})

            history = forecasts.get("history", [])
            result["history"] = [
                h for h in history[:365]  # Return up to 365 days for year view
                if h.get("actual_kwh") is not None or h.get("yield_kwh") is not None
            ]

        result["panel_groups"] = await self._get_panel_group_data()

        return web.json_response(result)

    async def _get_panel_group_data(self) -> dict[str, Any]:
        """Extract panel group predictions and actuals for today. @zara"""
        predictions = await _read_json_file(SOLAR_PATH / "stats" / "hourly_predictions.json")
        if not predictions or "predictions" not in predictions:
            return {"available": False, "groups": {}}

        today_str = date.today().isoformat()
        today_preds = [
            p for p in predictions["predictions"]
            if p.get("target_date") == today_str
        ]

        if not today_preds:
            return {"available": False, "groups": {}}

        group_names = set()
        for p in today_preds:
            if p.get("panel_group_predictions"):
                group_names.update(p["panel_group_predictions"].keys())
            if p.get("panel_group_actuals"):
                group_names.update(p["panel_group_actuals"].keys())

        if not group_names:
            return {"available": False, "groups": {}}

        groups = {}
        for group_name in sorted(group_names):
            group_data = {
                "name": group_name,
                "prediction_total_kwh": 0.0,
                "actual_total_kwh": 0.0,
                "hourly": [],
            }

            for p in today_preds:
                hour = p.get("target_hour")
                pred_kwh = None
                actual_kwh = None

                if p.get("panel_group_predictions"):
                    pred_kwh = p["panel_group_predictions"].get(group_name)
                if p.get("panel_group_actuals"):
                    actual_kwh = p["panel_group_actuals"].get(group_name)

                if pred_kwh is not None:
                    group_data["prediction_total_kwh"] += pred_kwh
                if actual_kwh is not None:
                    group_data["actual_total_kwh"] += actual_kwh

                group_data["hourly"].append({
                    "hour": hour,
                    "prediction_kwh": pred_kwh,
                    "actual_kwh": actual_kwh,
                })

            # Calculate accuracy
            if group_data["prediction_total_kwh"] > 0 and group_data["actual_total_kwh"] > 0:
                group_data["accuracy_percent"] = min(
                    100,
                    (group_data["actual_total_kwh"] / group_data["prediction_total_kwh"]) * 100
                )
            else:
                group_data["accuracy_percent"] = None

            groups[group_name] = group_data

        result = {"available": True, "groups": groups}
        await self._save_panel_group_cache(result, today_str)

        return result

    async def _save_panel_group_cache(self, data: dict[str, Any], today_str: str) -> None:
        """Save panel group data to cache file. @zara"""
        try:
            cache_path = SOLAR_PATH / "stats" / "panel_group_today_cache.json"
            cache_data = {
                "date": today_str,
                "last_updated": datetime.now().isoformat(),
                **data
            }
            import aiofiles
            async with aiofiles.open(cache_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(cache_data, indent=2))
        except Exception as e:
            _LOGGER.warning("Failed to save panel group cache: %s", e)


class BillingDataView(HomeAssistantView):
    """API for billing and annual balance data. @zara"""

    url = "/api/sfml_stats/billing"
    name = "api:sfml_stats:billing"
    requires_auth = False

    async def get(self, request: Request) -> Response:
        """Return billing configuration and annual balance data. @zara"""
        if HASS is None:
            return web.json_response({
                "success": False,
                "error": "Home Assistant not initialized",
            })

        billing_calculator = None
        entries = HASS.data.get(DOMAIN, {})
        for entry_id, entry_data in entries.items():
            if isinstance(entry_data, dict) and "billing_calculator" in entry_data:
                billing_calculator = entry_data["billing_calculator"]
                break

        if billing_calculator is None:
            return web.json_response({
                "success": False,
                "error": "BillingCalculator not initialized",
            })

        try:
            billing_data = await billing_calculator.async_calculate_billing()
        except Exception as err:
            _LOGGER.error("Error in billing calculation: %s", err)
            return web.json_response({
                "success": False,
                "error": str(err),
            })

        return web.json_response(billing_data)


class ExportSolarAnalyticsView(HomeAssistantView):
    """View to export solar analytics as PNG (Matplotlib)."""

    url = "/api/sfml_stats/export_solar_analytics"
    name = "api:sfml_stats:export_solar_analytics"
    requires_auth = False

    async def post(self, request: web.Request) -> web.Response:
        """Generate and return solar analytics PNG."""
        try:
            # Parse request JSON
            data = await request.json()
            period = data.get("period", "week")
            stats = data.get("stats", {})
            history = data.get("data", [])

            _LOGGER.info("Generating solar analytics export: period=%s, data_points=%d", period, len(history))

            # Import chart class
            from ..charts.solar_analytics import SolarAnalyticsChart

            # Generate chart
            chart = SolarAnalyticsChart(
                period=period,
                stats=stats,
                data=history
            )

            # Render to PNG bytes
            png_bytes = await chart.async_render()

            # Return as PNG
            return web.Response(
                body=png_bytes,
                content_type="image/png",
                headers={
                    "Content-Disposition": f'attachment; filename="solar_analytics_{period}.png"'
                }
            )

        except Exception as err:
            _LOGGER.error("Error generating solar analytics export: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)


class ExportBatteryAnalyticsView(HomeAssistantView):
    """View to export battery analytics as PNG (Matplotlib)."""

    url = "/api/sfml_stats/export_battery_analytics"
    name = "api:sfml_stats:export_battery_analytics"
    requires_auth = False

    async def post(self, request: web.Request) -> web.Response:
        """Generate and return battery analytics PNG."""
        try:
            data = await request.json()
            period = data.get("period", "week")
            stats = data.get("stats", {})
            history = data.get("data", [])

            _LOGGER.info("Generating battery analytics export: period=%s, data_points=%d", period, len(history))

            from ..charts.battery_analytics import BatteryAnalyticsChart

            chart = BatteryAnalyticsChart(
                period=period,
                stats=stats,
                data=history
            )

            png_bytes = await chart.async_render()

            return web.Response(
                body=png_bytes,
                content_type="image/png",
                headers={
                    "Content-Disposition": f'attachment; filename="battery_analytics_{period}.png"'
                }
            )

        except Exception as err:
            _LOGGER.error("Error generating battery analytics export: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)


class ExportHouseAnalyticsView(HomeAssistantView):
    """View to export house analytics as PNG (Matplotlib)."""

    url = "/api/sfml_stats/export_house_analytics"
    name = "api:sfml_stats:export_house_analytics"
    requires_auth = False

    async def post(self, request: web.Request) -> web.Response:
        """Generate and return house analytics PNG."""
        try:
            data = await request.json()
            period = data.get("period", "week")
            stats = data.get("stats", {})
            history = data.get("data", [])

            _LOGGER.info("Generating house analytics export: period=%s, data_points=%d", period, len(history))

            from ..charts.house_analytics import HouseAnalyticsChart

            chart = HouseAnalyticsChart(
                period=period,
                stats=stats,
                data=history
            )

            png_bytes = await chart.async_render()

            return web.Response(
                body=png_bytes,
                content_type="image/png",
                headers={
                    "Content-Disposition": f'attachment; filename="house_analytics_{period}.png"'
                }
            )

        except Exception as err:
            _LOGGER.error("Error generating house analytics export: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)


class ExportGridAnalyticsView(HomeAssistantView):
    """View to export grid analytics as PNG (Matplotlib)."""

    url = "/api/sfml_stats/export_grid_analytics"
    name = "api:sfml_stats:export_grid_analytics"
    requires_auth = False

    async def post(self, request: web.Request) -> web.Response:
        """Generate and return grid analytics PNG."""
        try:
            data = await request.json()
            period = data.get("period", "week")
            stats = data.get("stats", {})
            history = data.get("data", [])

            _LOGGER.info("Generating grid analytics export: period=%s, data_points=%d", period, len(history))

            from ..charts.grid_analytics import GridAnalyticsChart

            chart = GridAnalyticsChart(
                period=period,
                stats=stats,
                data=history
            )

            png_bytes = await chart.async_render()

            return web.Response(
                body=png_bytes,
                content_type="image/png",
                headers={
                    "Content-Disposition": f'attachment; filename="grid_analytics_{period}.png"'
                }
            )

        except Exception as err:
            _LOGGER.error("Error generating grid analytics export: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)


class WeatherHistoryView(HomeAssistantView):
    """View to get weather history data."""

    url = "/api/sfml_stats/weather_history"
    name = "api:sfml_stats:weather_history"
    requires_auth = False

    async def get(self, request: web.Request) -> web.Response:
        """Get weather history."""
        try:
            from ..weather_collector import WeatherDataCollector

            data_path = Path(HASS.config.path()) / "sfml_stats_weather"
            collector = WeatherDataCollector(HASS, data_path)

            history = await collector.get_history(days=365)
            stats = await collector.get_statistics()

            return web.json_response({
                "success": True,
                "data": history,
                "stats": stats
            })

        except Exception as err:
            _LOGGER.error("Error fetching weather history: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)


class ExportWeatherAnalyticsView(HomeAssistantView):
    """View to export weather analytics as PNG."""

    url = "/api/sfml_stats/export_weather_analytics"
    name = "api:sfml_stats:export_weather_analytics"
    requires_auth = False

    async def post(self, request: web.Request) -> web.Response:
        """Generate and return weather analytics PNG."""
        try:
            data = await request.json()
            period = data.get("period", "week")
            stats = data.get("stats", {})
            history = data.get("data", [])

            _LOGGER.info("Generating weather analytics export: period=%s, data_points=%d", period, len(history))

            from ..charts.weather_analytics import WeatherAnalyticsChart

            chart = WeatherAnalyticsChart(
                period=period,
                stats=stats,
                data=history
            )

            png_bytes = await chart.async_render()

            return web.Response(
                body=png_bytes,
                content_type="image/png",
                headers={
                    "Content-Disposition": f'attachment; filename="weather_analytics_{period}.png"'
                }
            )

        except Exception as err:
            _LOGGER.error("Error generating weather analytics export: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)


class PowerSourcesHistoryView(HomeAssistantView):
    """View to get power sources history data from HA Recorder. @zara"""

    url = "/api/sfml_stats/power_sources_history"
    name = "api:sfml_stats:power_sources_history"
    requires_auth = False

    async def get(self, request: web.Request) -> web.Response:
        """Get power sources history from Home Assistant Recorder. @zara"""
        try:
            hours = int(request.query.get("hours", 24))
            hours = min(hours, 168)  # Max 7 days

            config = _get_config()

            # Get configured sensor entity IDs
            sensors = {
                "solar_to_house": config.get(CONF_SENSOR_SOLAR_TO_HOUSE),
                "battery_to_house": config.get(CONF_SENSOR_BATTERY_TO_HOUSE),
                "grid_to_house": config.get(CONF_SENSOR_GRID_TO_HOUSE),
                "home_consumption": config.get(CONF_SENSOR_HOME_CONSUMPTION),
                "battery_soc": config.get(CONF_SENSOR_BATTERY_SOC),
            }

            # Filter out None values
            entity_ids = [eid for eid in sensors.values() if eid]

            if not entity_ids:
                return web.json_response({
                    "success": False,
                    "error": "No sensors configured"
                })

            # Get history from recorder
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)

            history_data = await self._get_recorder_history(
                entity_ids, start_time, end_time
            )

            # Process and align data
            processed_data = self._process_history(history_data, sensors, start_time, end_time)

            # Check if we got any actual data
            has_data = any(
                any(d.get(k) is not None for k in ['solar_to_house', 'battery_to_house', 'grid_to_house', 'home_consumption'])
                for d in processed_data
            )

            # If no data from recorder, try power sources collector data
            data_source = "recorder"
            if not has_data:
                _LOGGER.info("No data from recorder, trying power sources collector data")
                collector_data = await self._get_power_sources_collector_data(hours)
                if collector_data:
                    processed_data = collector_data
                    data_source = "collector"
                    _LOGGER.info("Got %d entries from power sources collector", len(collector_data))
                else:
                    # Last resort: try hourly file fallback
                    file_data = await self._get_hourly_history_from_file()
                    if file_data:
                        processed_data = file_data
                        data_source = "hourly_file"
                        _LOGGER.info("Got %d entries from hourly file", len(file_data))

            return web.json_response({
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "hours": hours,
                "sensors": sensors,
                "data": processed_data,
                "data_source": data_source
            })

        except Exception as err:
            _LOGGER.error("Error fetching power sources history: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)

    async def _get_recorder_history(
        self,
        entity_ids: list[str],
        start_time: datetime,
        end_time: datetime
    ) -> dict[str, list]:
        """Fetch history from Home Assistant Recorder. @zara"""
        if HASS is None:
            _LOGGER.error("HASS is None in _get_recorder_history")
            return {}

        _LOGGER.debug("Fetching history for entities: %s from %s to %s", entity_ids, start_time, end_time)

        # Try multiple methods to get history data
        # Method 1: Use get_significant_states via recorder instance executor
        # Note: get_significant_states is a SYNC function, must run in executor
        try:
            from homeassistant.components.recorder import get_instance
            from homeassistant.components.recorder import history as recorder_history

            if hasattr(recorder_history, 'get_significant_states'):
                instance = get_instance(HASS)

                # get_significant_states is synchronous - must run in executor
                def _get_history_sync():
                    return recorder_history.get_significant_states(
                        HASS,
                        start_time,
                        end_time,
                        entity_ids,
                        significant_changes_only=False,
                        include_start_time_state=True,
                    )

                history_data = await instance.async_add_executor_job(_get_history_sync)

                if history_data:
                    _LOGGER.info("Got history data via get_significant_states: %d entities", len(history_data))
                    return history_data
            else:
                _LOGGER.debug("get_significant_states not available")

        except Exception as e:
            _LOGGER.warning("Method 1 (get_significant_states via executor) failed: %s", e)

        # Fallback - collect current states and build minimal history
        _LOGGER.warning("All recorder methods failed, falling back to current state")
        return await self._get_history_fallback(entity_ids)

    async def _get_history_fallback(
        self,
        entity_ids: list[str],
    ) -> dict[str, list]:
        """Fallback: Get current states when recorder fails. @zara"""
        if HASS is None:
            return {}

        result = {}
        for entity_id in entity_ids:
            state = HASS.states.get(entity_id)
            if state:
                # Create a simple state object that matches the expected format
                result[entity_id] = [state]
                _LOGGER.debug("Fallback: Got current state for %s: %s", entity_id, state.state)

        return result

    async def _get_hourly_history_from_file(self) -> list[dict]:
        """Get hourly history from our own data file as alternative. @zara"""
        try:
            hourly_path = Path(HASS.config.path()) / "sfml_stats" / "data" / "hourly_billing_history.json"

            if not hourly_path.exists():
                return []

            import aiofiles
            import json

            async with aiofiles.open(hourly_path, 'r') as f:
                content = await f.read()
                data = json.loads(content)

            hours_data = data.get("hours", {})
            result = []

            for hour_key, hour_data in sorted(hours_data.items()):
                result.append({
                    "timestamp": hour_key + ":00",
                    "solar_to_house": hour_data.get("solar_to_house_kwh", 0) * 1000,  # Convert to W (avg)
                    "battery_to_house": hour_data.get("battery_to_house_kwh", 0) * 1000,
                    "grid_to_house": hour_data.get("grid_to_house_kwh", 0) * 1000,
                    "home_consumption": hour_data.get("home_consumption_kwh", 0) * 1000,
                    "battery_soc": None,
                })

            return result
        except Exception as e:
            _LOGGER.error("Error reading hourly history file: %s", e)
            return []

    async def _get_power_sources_collector_data(self, hours: int) -> list[dict]:
        """Get data from power sources collector file. @zara"""
        try:
            collector_path = Path(HASS.config.path()) / "sfml_stats" / "data" / "power_sources_history.json"

            if not collector_path.exists():
                _LOGGER.debug("Power sources collector file not found")
                return []

            import aiofiles
            import json

            async with aiofiles.open(collector_path, 'r') as f:
                content = await f.read()
                data = json.loads(content)

            data_points = data.get("data_points", [])
            if not data_points:
                return []

            # Filter by time
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            filtered = []

            for dp in data_points:
                try:
                    ts = datetime.fromisoformat(dp["timestamp"].replace('Z', '+00:00'))
                    if ts > cutoff:
                        filtered.append(dp)
                except (ValueError, KeyError):
                    continue

            _LOGGER.debug("Power sources collector: %d points after filtering", len(filtered))
            return sorted(filtered, key=lambda x: x["timestamp"])

        except Exception as e:
            _LOGGER.error("Error reading power sources collector file: %s", e)
            return []

    def _process_history(
        self,
        history_data: dict[str, list],
        sensors: dict[str, str | None],
        start_time: datetime,
        end_time: datetime
    ) -> list[dict]:
        """Process and align history data into time series. @zara"""
        # Create time buckets (5-minute intervals)
        interval_minutes = 5
        buckets = []
        current_time = start_time

        while current_time <= end_time:
            buckets.append({
                "timestamp": current_time.isoformat(),
                "solar_to_house": None,
                "battery_to_house": None,
                "grid_to_house": None,
                "home_consumption": None,
                "battery_soc": None,
            })
            current_time += timedelta(minutes=interval_minutes)

        # Fill buckets with sensor data
        for sensor_key, entity_id in sensors.items():
            if not entity_id or entity_id not in history_data:
                continue

            states = history_data[entity_id]
            if not states:
                continue

            # Sort states by time
            sorted_states = sorted(states, key=lambda s: s.last_updated if hasattr(s, 'last_updated') else s.last_changed)

            state_idx = 0
            for bucket in buckets:
                bucket_time = datetime.fromisoformat(bucket["timestamp"])
                if bucket_time.tzinfo is None:
                    bucket_time = bucket_time.replace(tzinfo=timezone.utc)

                # Find the most recent state before bucket time
                while (state_idx < len(sorted_states) - 1):
                    next_state = sorted_states[state_idx + 1]
                    next_time = next_state.last_updated if hasattr(next_state, 'last_updated') else next_state.last_changed
                    if next_time.tzinfo is None:
                        next_time = next_time.replace(tzinfo=timezone.utc)
                    if next_time <= bucket_time:
                        state_idx += 1
                    else:
                        break

                if state_idx < len(sorted_states):
                    state = sorted_states[state_idx]
                    try:
                        value = float(state.state)
                        bucket[sensor_key] = round(value, 3)
                    except (ValueError, TypeError):
                        pass

        return buckets


class ExportPowerSourcesView(HomeAssistantView):
    """View to export power sources chart as PNG. @zara"""

    url = "/api/sfml_stats/export_power_sources"
    name = "api:sfml_stats:export_power_sources"
    requires_auth = False

    async def post(self, request: web.Request) -> web.Response:
        """Generate and return power sources PNG. @zara"""
        try:
            data = await request.json()
            period = data.get("period", "today")
            stats = data.get("stats", {})
            history = data.get("data", [])

            _LOGGER.info("Generating power sources export: period=%s, data_points=%d", period, len(history))

            from ..charts.power_sources import PowerSourcesChart

            chart = PowerSourcesChart(
                period=period,
                stats=stats,
                data=history
            )

            png_bytes = await chart.async_render()

            return web.Response(
                body=png_bytes,
                content_type="image/png",
                headers={
                    "Content-Disposition": f'attachment; filename="power_sources_{period}.png"'
                }
            )

        except Exception as err:
            _LOGGER.error("Error generating power sources export: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)
