"""REST API views for SFML Stats Dashboard.

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

import functools
import ipaddress
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web
from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant


# ============================================================
# Local Network Security - Only allow local network access
# ============================================================
LOCAL_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("::1/128"),
]


def _get_client_ip(request: web.Request) -> str:
    """Extract real client IP from request."""
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    peername = request.transport.get_extra_info("peername")
    if peername:
        return peername[0]
    return "unknown"


def _is_local_ip(ip_str: str) -> bool:
    """Check if IP is in local network range."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in network for network in LOCAL_NETWORKS)
    except ValueError:
        return False


def local_only(func):
    """Decorator: Block external (non-local) requests."""
    @functools.wraps(func)
    async def wrapper(self, request: web.Request, *args, **kwargs):
        client_ip = _get_client_ip(request)
        if not _is_local_ip(client_ip):
            logging.getLogger(__name__).warning(
                "Blocked external access from %s to %s", client_ip, request.path
            )
            return web.Response(
                text="<!DOCTYPE html><html><head><title>Access Denied</title></head>"
                "<body style='font-family:sans-serif;text-align:center;padding:50px;'>"
                "<h1>403 - Access Denied</h1>"
                "<p>SFML Stats Lite is only accessible from the local network.</p>"
                "</body></html>",
                status=403,
                content_type="text/html"
            )
        return await func(self, request, *args, **kwargs)
    return wrapper

from ..const import (
    DOMAIN,
    VERSION,
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
    CONF_FEED_IN_TARIFF,
    DEFAULT_FEED_IN_TARIFF,
    CONF_PANEL_GROUP_NAMES,
)

if TYPE_CHECKING:
    from aiohttp.web import Request, Response

_LOGGER = logging.getLogger(__name__)

SOLAR_PATH: Path | None = None
GRID_PATH: Path | None = None
HASS: HomeAssistant | None = None


async def async_setup_views(hass: HomeAssistant) -> None:
    """Register all API views."""
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
    hass.http.register_view(CSSFilesView())
    hass.http.register_view(JSFilesView())
    hass.http.register_view(BackgroundImageView())
    hass.http.register_view(EnergyFlowView())
    hass.http.register_view(StatisticsView())
    hass.http.register_view(BillingDataView())
    hass.http.register_view(WeatherHistoryView())
    hass.http.register_view(PowerSourcesHistoryView())
    hass.http.register_view(EnergySourcesDailyStatsView())
    hass.http.register_view(HealthCheckView())
    hass.http.register_view(ClothingRecommendationView())
    hass.http.register_view(ForecastComparisonView())

    _LOGGER.info("SFML Stats API views registered")


async def _read_json_file(path: Path | None) -> dict | None:
    """Read a JSON file asynchronously."""
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
    """Main view serving the Vue.js app."""

    url = "/api/sfml_stats_lite/dashboard"
    name = "api:sfml_stats_lite:dashboard"
    requires_auth = False
    cors_allowed = True

    @local_only
    async def get(self, request: Request) -> Response:
        """Return the dashboard HTML page."""
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
        """Return fallback HTML when build is not present."""
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
    """Serve static files (JS, CSS, Assets)."""

    url = "/api/sfml_stats_lite/assets/{filename:.*}"
    name = "api:sfml_stats_lite:assets"
    requires_auth = False

    @local_only
    async def get(self, request: Request, filename: str) -> Response:
        """Return a static file.

        Supports files from:
        - frontend/dist/assets/  (images, fonts)
        - frontend/dist/css/     (stylesheets)
        - frontend/dist/js/      (javascript modules)
        """
        frontend_path = None

        # Determine the subdirectory based on file type
        if filename.startswith("css/") or filename.endswith(".css"):
            subdir = "css"
            # Remove css/ prefix if present
            clean_filename = filename[4:] if filename.startswith("css/") else filename
        elif filename.startswith("js/") or filename.endswith(".js"):
            subdir = "js"
            # Remove js/ prefix if present
            clean_filename = filename[3:] if filename.startswith("js/") else filename
        else:
            subdir = "assets"
            clean_filename = filename

        # Try via HASS.config.path() first (works in Docker)
        if HASS is not None:
            frontend_path = Path(HASS.config.path()) / "custom_components" / "sfml_stats_lite" / "frontend" / "dist" / subdir / clean_filename
            if not frontend_path.exists():
                # Fallback to assets folder for backward compatibility
                frontend_path = Path(HASS.config.path()) / "custom_components" / "sfml_stats_lite" / "frontend" / "dist" / "assets" / filename
                if not frontend_path.exists():
                    frontend_path = None

        # Fallback via __file__
        if frontend_path is None:
            frontend_path = Path(__file__).parent.parent / "frontend" / "dist" / subdir / clean_filename
            if not frontend_path.exists():
                # Fallback to assets folder
                frontend_path = Path(__file__).parent.parent / "frontend" / "dist" / "assets" / filename

        if not frontend_path.exists():
            _LOGGER.warning("Static file not found: %s (tried %s)", filename, frontend_path)
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


class CSSFilesView(HomeAssistantView):
    """Serve CSS files."""

    url = "/api/sfml_stats_lite/css/{filename:.*}"
    name = "api:sfml_stats_lite:css"
    requires_auth = False

    @local_only
    async def get(self, request: Request, filename: str) -> Response:
        """Return a CSS file."""
        frontend_path = Path(__file__).parent.parent / "frontend" / "dist" / "css" / filename

        if not frontend_path.exists():
            return web.Response(status=404, text="Not found")

        import aiofiles
        async with aiofiles.open(frontend_path, "rb") as f:
            content = await f.read()

        return web.Response(body=content, content_type="text/css")


class JSFilesView(HomeAssistantView):
    """Serve JavaScript files."""

    url = "/api/sfml_stats_lite/js/{filename:.*}"
    name = "api:sfml_stats_lite:js"
    requires_auth = False

    @local_only
    async def get(self, request: Request, filename: str) -> Response:
        """Return a JavaScript file."""
        frontend_path = Path(__file__).parent.parent / "frontend" / "dist" / "js" / filename

        if not frontend_path.exists():
            return web.Response(status=404, text="Not found")

        import aiofiles
        async with aiofiles.open(frontend_path, "rb") as f:
            content = await f.read()

        return web.Response(body=content, content_type="application/javascript")


class BackgroundImageView(HomeAssistantView):
    """Serve background image."""

    url = "/api/sfml_stats_lite/background.png"
    name = "api:sfml_stats_lite:background"
    requires_auth = False

    @local_only
    async def get(self, request: Request) -> Response:
        """Return the background image."""
        frontend_path = Path(__file__).parent.parent / "frontend" / "dist" / "background.png"

        if not frontend_path.exists():
            return web.Response(status=404, text="Not found")

        import aiofiles
        async with aiofiles.open(frontend_path, "rb") as f:
            content = await f.read()

        return web.Response(
            body=content,
            content_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"}
        )


class SolarDataView(HomeAssistantView):
    """API for solar data."""

    url = "/api/sfml_stats_lite/solar"
    name = "api:sfml_stats_lite:solar"
    requires_auth = False

    @local_only
    async def get(self, request: Request) -> Response:
        """Return solar data."""
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

        ai_weights = await _read_json_file(SOLAR_PATH / "ai" / "learned_weights.json")
        if ai_weights:
            result["data"]["ai_state"] = ai_weights

        if forecasts_data and "today" in forecasts_data:
            forecast_day = forecasts_data["today"].get("forecast_day", {})
            forecast_tomorrow = forecasts_data["today"].get("forecast_tomorrow", {})
            forecast_day_after = forecasts_data["today"].get("forecast_day_after_tomorrow", {})
            result["data"]["forecasts"] = {
                "today": {
                    "prediction_kwh": forecast_day.get("prediction_kwh"),
                },
                "tomorrow": {
                    "date": forecast_tomorrow.get("date"),
                    "prediction_kwh": forecast_tomorrow.get("prediction_kwh"),
                },
                "day_after_tomorrow": {
                    "date": forecast_day_after.get("date"),
                    "prediction_kwh": forecast_day_after.get("prediction_kwh"),
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

        # Multi-day hourly forecast (Heute, Morgen, Übermorgen)
        multi_day = await _read_json_file(SOLAR_PATH / "stats" / "multi_day_hourly_forecast.json")
        if multi_day and "days" in multi_day:
            result["data"]["multi_day_hourly"] = multi_day["days"]

        return web.json_response(result)


class PriceDataView(HomeAssistantView):
    """API for electricity price data."""

    url = "/api/sfml_stats_lite/prices"
    name = "api:sfml_stats_lite:prices"
    requires_auth = False

    @local_only
    async def get(self, request: Request) -> Response:
        """Return price data."""
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
    """API for aggregated dashboard data."""

    url = "/api/sfml_stats_lite/summary"
    name = "api:sfml_stats_lite:summary"
    requires_auth = False

    @local_only
    async def get(self, request: Request) -> Response:
        """Return a summary for the dashboard."""
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

        ai_weights = await _read_json_file(SOLAR_PATH / "ai" / "learned_weights.json")
        if ai_weights:
            result["kpis"]["ai_training_samples"] = ai_weights.get("training_samples", 0)

        def extract_time(iso_string: str | None) -> str | None:
            """Extract HH:MM from ISO string."""
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
    """API for realtime data (polled by frontend)."""

    url = "/api/sfml_stats_lite/realtime"
    name = "api:sfml_stats_lite:realtime"
    requires_auth = False

    @local_only
    async def get(self, request: Request) -> Response:
        """Return current realtime data."""
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
    """Get current configuration from the first config entry."""
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
    """Read current value from a sensor."""
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
    """Read weather data from a Home Assistant weather entity."""
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
    """API for energy flow data from Home Assistant sensors."""

    url = "/api/sfml_stats_lite/energy_flow"
    name = "api:sfml_stats_lite:energy_flow"
    requires_auth = False

    @local_only
    async def get(self, request: Request) -> Response:
        """Return current energy flow data."""
        config = _get_config()

        # Battery configuration check
        battery_configured = config.get(CONF_SENSOR_BATTERY_SOC) is not None
        battery_soc = _get_sensor_value(config.get(CONF_SENSOR_BATTERY_SOC)) if battery_configured else None
        battery_power = _get_sensor_value(config.get(CONF_SENSOR_BATTERY_POWER)) if battery_configured else None
        battery_to_house = _get_sensor_value(config.get(CONF_SENSOR_BATTERY_TO_HOUSE)) if battery_configured else None
        grid_to_battery = _get_sensor_value(config.get(CONF_SENSOR_GRID_TO_BATTERY)) if battery_configured else None

        # If battery not configured, don't show battery flows
        if not battery_configured:
            battery_to_house = None
            grid_to_battery = None

        # Solar kann NIEMALS negativ sein - korrigiere negative Werte
        solar_power = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_POWER))
        solar_to_house = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_TO_HOUSE))
        solar_to_battery = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_TO_BATTERY)) if battery_configured else None
        if solar_power is not None and solar_power < 0:
            solar_power = 0.0
        if solar_to_house is not None and solar_to_house < 0:
            solar_to_house = 0.0
        if solar_to_battery is not None and solar_to_battery < 0:
            solar_to_battery = 0.0

        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "flows": {
                "solar_power": solar_power,
                "solar_to_house": solar_to_house,
                "solar_to_battery": solar_to_battery,
                "battery_to_house": battery_to_house,
                "grid_to_house": _get_sensor_value(config.get(CONF_SENSOR_GRID_TO_HOUSE)),
                "grid_to_battery": grid_to_battery,
                "house_to_grid": _get_sensor_value(config.get(CONF_SENSOR_HOUSE_TO_GRID)),
            },
            "battery": {
                "soc": battery_soc,
                "power": battery_power,
                "configured": battery_configured,
            },
            "home": {
                "consumption": _get_sensor_value(config.get(CONF_SENSOR_HOME_CONSUMPTION)),
            },
            "statistics": {
                "solar_yield_daily": _get_sensor_value(config.get(CONF_SENSOR_SOLAR_YIELD_DAILY)),
                "grid_import_daily": _get_sensor_value(config.get(CONF_SENSOR_GRID_IMPORT_DAILY)),
                "grid_import_yearly": _get_sensor_value(config.get(CONF_SENSOR_GRID_IMPORT_YEARLY)),
                "battery_charge_solar_daily": _get_sensor_value(config.get(CONF_SENSOR_BATTERY_CHARGE_SOLAR_DAILY)) if battery_configured else None,
                "battery_charge_grid_daily": _get_sensor_value(config.get(CONF_SENSOR_BATTERY_CHARGE_GRID_DAILY)) if battery_configured else None,
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
            "feed_in_tariff": config.get(CONF_FEED_IN_TARIFF, DEFAULT_FEED_IN_TARIFF),
            "panels": self._get_panel_data(config),
            "weather_ha": _get_weather_data(config.get(CONF_WEATHER_ENTITY)),
            "sun_position": await self._get_sun_position(),
            "current_price": await self._get_current_price(),
        }

        return web.json_response(result)

    async def _get_current_price(self) -> dict[str, Any] | None:
        """Read current electricity price from price_cache.json."""
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
        """Read current sun position from astronomy_cache.json."""
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
        """Convert azimuth degrees to cardinal direction."""
        if azimuth is None:
            return "—"
        azimuth = azimuth % 360
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = int((azimuth + 22.5) / 45) % 8
        return directions[index]

    def _extract_time(self, iso_string: str | None) -> str | None:
        """Extract HH:MM from ISO string."""
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
        """Read panel data from configured sensors."""
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
    """API for statistics data from JSON files."""

    url = "/api/sfml_stats_lite/statistics"
    name = "api:sfml_stats_lite:statistics"
    requires_auth = False

    @local_only
    async def get(self, request: Request) -> Response:
        """Return statistics data from Solar Forecast ML JSON files."""
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
                "yield_kwh": today_data.get("yield_today", {}).get("kwh"),
            }

            forecast_tomorrow_data = today_data.get("forecast_tomorrow", {})
            result["production"]["tomorrow"] = {
                "forecast_kwh": forecast_tomorrow_data.get("prediction_kwh"),
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
        """Extract panel group predictions and actuals for today."""
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

        # Get panel group name mappings from config
        config = _get_config()
        name_mapping = config.get(CONF_PANEL_GROUP_NAMES, {})

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
            # Apply name mapping if configured
            display_name = name_mapping.get(group_name, group_name)
            group_data = {
                "name": group_name,
                "display_name": display_name,
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
        """Save panel group data to cache file."""
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
    """API for billing and annual balance data."""

    url = "/api/sfml_stats_lite/billing"
    name = "api:sfml_stats_lite:billing"
    requires_auth = False

    @local_only
    async def get(self, request: Request) -> Response:
        """Return billing configuration and annual balance data."""
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


class WeatherHistoryView(HomeAssistantView):
    """View to get weather history data."""

    url = "/api/sfml_stats_lite/weather_history"
    name = "api:sfml_stats_lite:weather_history"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Get weather history."""
        try:
            from ..weather_collector import WeatherDataCollector

            data_path = Path(HASS.config.path()) / "sfml_stats_lite" / "weather"
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


class PowerSourcesHistoryView(HomeAssistantView):
    """View to get power sources history data from HA Recorder."""

    url = "/api/sfml_stats_lite/power_sources_history"
    name = "api:sfml_stats_lite:power_sources_history"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Get power sources history from Home Assistant Recorder."""
        try:
            hours = int(request.query.get("hours", 24))
            hours = min(hours, 168)  # Max 7 days

            config = _get_config()

            # Get configured sensor entity IDs
            sensors = {
                "solar_power": config.get(CONF_SENSOR_SOLAR_POWER),
                "solar_to_house": config.get(CONF_SENSOR_SOLAR_TO_HOUSE),
                "solar_to_battery": config.get(CONF_SENSOR_SOLAR_TO_BATTERY),
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
                any(d.get(k) is not None for k in ['solar_power', 'solar_to_house', 'solar_to_battery', 'battery_to_house', 'grid_to_house', 'home_consumption'])
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
        """Fetch history from Home Assistant Recorder."""
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
        """Fallback: Get current states when recorder fails."""
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
        """Get hourly history from our own data file as alternative."""
        try:
            hourly_path = Path(HASS.config.path()) / "sfml_stats_lite" / "data" / "hourly_billing_history.json"

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
        """Get data from power sources collector file."""
        try:
            collector_path = Path(HASS.config.path()) / "sfml_stats_lite" / "data" / "power_sources_history.json"

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
        """Process and align history data into time series."""
        # Create time buckets (5-minute intervals)
        interval_minutes = 5
        buckets = []
        current_time = start_time

        while current_time <= end_time:
            buckets.append({
                "timestamp": current_time.isoformat(),
                "solar_power": None,
                "solar_to_house": None,
                "solar_to_battery": None,
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


class EnergySourcesDailyStatsView(HomeAssistantView):
    """API for daily energy sources statistics."""

    url = "/api/sfml_stats_lite/energy_sources_daily_stats"
    name = "api:sfml_stats_lite:energy_sources_daily_stats"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Get daily energy sources statistics."""
        try:
            days = int(request.query.get("days", 7))
            days = min(days, 365)  # Max 1 year

            # Get power sources collector from hass.data
            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized"
                })

            # Try to get collector from entry data
            collector = None
            entries = HASS.data.get(DOMAIN, {})
            for entry_id, entry_data in entries.items():
                if isinstance(entry_data, dict) and "power_sources_collector" in entry_data:
                    collector = entry_data["power_sources_collector"]
                    break

            if collector is None:
                # Fallback: read directly from file
                data_path = Path(HASS.config.path()) / "sfml_stats_lite" / "data" / "energy_sources_daily_stats.json"
                if data_path.exists():
                    import aiofiles
                    async with aiofiles.open(data_path, 'r') as f:
                        content = await f.read()
                        daily_stats = json.loads(content)
                else:
                    daily_stats = {"days": {}}
            else:
                daily_stats = await collector.get_daily_stats(days)

            # Merge with daily_energy_history.json for more complete data
            history_path = Path(HASS.config.path()) / "sfml_stats_lite" / "data" / "daily_energy_history.json"
            if history_path.exists():
                import aiofiles
                async with aiofiles.open(history_path, 'r') as f:
                    content = await f.read()
                    history_data = json.loads(content)
                    history_days = history_data.get("days", {})

                    # Merge history data into daily_stats (add missing days)
                    for date_str, day_data in history_days.items():
                        if date_str not in daily_stats.get("days", {}):
                            # Convert history format to daily_stats format
                            daily_stats.setdefault("days", {})[date_str] = {
                                "date": date_str,
                                "solar_to_house_kwh": day_data.get("solar_to_house_kwh", 0),
                                "solar_to_battery_kwh": day_data.get("battery_charge_solar_kwh", 0),
                                "battery_to_house_kwh": day_data.get("battery_to_house_kwh", 0),
                                "battery_charge_grid_kwh": day_data.get("battery_charge_grid_kwh", 0),
                                "grid_to_house_kwh": day_data.get("grid_import_kwh", 0),
                                "grid_export_kwh": day_data.get("grid_export_kwh", 0),
                                "home_consumption_kwh": day_data.get("home_consumption_kwh", 0),
                                "solar_yield_kwh": day_data.get("solar_yield_kwh", 0),
                                "price_ct_kwh": day_data.get("price_ct_kwh", 0),
                                "autarky_percent": day_data.get("autarky_percent", 0),
                                "self_consumption_percent": day_data.get("self_consumption_percent", 0),
                                "avg_soc": day_data.get("avg_soc", 0),
                                "min_soc": day_data.get("min_soc", 0),
                                "max_soc": day_data.get("max_soc", 0),
                                "peak_battery_power_w": day_data.get("peak_battery_power_w", 0),
                                "peak_consumption_w": 0,
                            }
                        else:
                            # Merge additional fields from history into existing day data
                            existing = daily_stats["days"][date_str]
                            if existing.get("peak_battery_power_w") is None or existing.get("peak_battery_power_w") == 0:
                                existing["peak_battery_power_w"] = day_data.get("peak_battery_power_w", 0)

            # Also get current sensor values for real-time display
            config = _get_config()
            # Solar kann NIEMALS negativ sein
            solar_to_house_val = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_TO_HOUSE))
            solar_to_battery_val = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_TO_BATTERY))
            if solar_to_house_val is not None and solar_to_house_val < 0:
                solar_to_house_val = 0.0
            if solar_to_battery_val is not None and solar_to_battery_val < 0:
                solar_to_battery_val = 0.0
            current_values = {
                "solar_yield_daily": _get_sensor_value(config.get(CONF_SENSOR_SOLAR_YIELD_DAILY)),
                "solar_to_house": solar_to_house_val,
                "solar_to_battery": solar_to_battery_val,
                "battery_to_house": _get_sensor_value(config.get(CONF_SENSOR_BATTERY_TO_HOUSE)),
                "grid_to_house": _get_sensor_value(config.get(CONF_SENSOR_GRID_TO_HOUSE)),
                "home_consumption": _get_sensor_value(config.get(CONF_SENSOR_HOME_CONSUMPTION)),
            }

            return web.json_response({
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "days_requested": days,
                "daily_stats": daily_stats.get("days", {}),
                "current_values": current_values,
            })

        except Exception as err:
            _LOGGER.error("Error fetching energy sources daily stats: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)


class HealthCheckView(HomeAssistantView):
    """Health check endpoint for monitoring."""

    url = "/api/sfml_stats_lite/health"
    name = "api:sfml_stats_lite:health"
    requires_auth = False
    cors_allowed = True

    @local_only
    async def get(self, request: Request) -> Response:
        """Return health status."""
        status = "healthy"
        checks = {}

        # Check HASS
        if HASS is None:
            status = "degraded"
            checks["hass"] = {"status": "error", "message": "Home Assistant not available"}
        else:
            checks["hass"] = {"status": "ok"}

        # Check Solar Path
        if SOLAR_PATH and SOLAR_PATH.exists():
            checks["solar_path"] = {"status": "ok", "path": str(SOLAR_PATH)}
        else:
            checks["solar_path"] = {"status": "missing", "path": str(SOLAR_PATH) if SOLAR_PATH else None}

        # Check Grid Path
        if GRID_PATH and GRID_PATH.exists():
            checks["grid_path"] = {"status": "ok", "path": str(GRID_PATH)}
        else:
            checks["grid_path"] = {"status": "missing", "path": str(GRID_PATH) if GRID_PATH else None}

        # Check config entries
        config = _get_config()
        if config:
            checks["config"] = {"status": "ok", "sensors_configured": len([k for k, v in config.items() if v])}
        else:
            checks["config"] = {"status": "missing"}
            status = "degraded"

        return web.json_response({
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "version": VERSION,
            "checks": checks,
        })


class ClothingRecommendationView(HomeAssistantView):
    """API for clothing recommendation based on weather."""

    url = "/api/sfml_stats_lite/clothing"
    name = "api:sfml_stats_lite:clothing"
    requires_auth = False
    cors_allowed = True

    @local_only
    async def get(self, request: Request) -> Response:
        """Return clothing recommendation based on current weather."""
        try:
            from ..clothing_recommendation import get_recommendation

            config = _get_config()
            weather_entity = config.get(CONF_WEATHER_ENTITY)

            # Get weather data from HA entity
            weather_data = _get_weather_data(weather_entity) if weather_entity else None

            if not weather_data:
                # Fallback to Solar Forecast ML weather data
                weather_file = await _read_json_file(SOLAR_PATH / "stats" / "hourly_weather_actual.json")
                if weather_file and "hourly_data" in weather_file:
                    today_str = date.today().isoformat()
                    hour_str = str(datetime.now().hour)
                    if today_str in weather_file["hourly_data"]:
                        current_hour = weather_file["hourly_data"][today_str].get(hour_str, {})
                        weather_data = {
                            "temperature": current_hour.get("temperature_2m"),
                            "humidity": current_hour.get("relative_humidity_2m"),
                            "wind_speed": current_hour.get("wind_speed_10m"),
                            "precipitation": current_hour.get("precipitation"),
                            "cloud_cover": current_hour.get("cloud_cover"),
                            "pressure": current_hour.get("surface_pressure"),
                            "uv_index": current_hour.get("uv_index", 0),
                            "radiation": current_hour.get("direct_radiation", 0),
                        }

            if not weather_data:
                return web.json_response({
                    "success": False,
                    "error": "No weather data available",
                    "hint": "Configure a weather entity or ensure Solar Forecast ML is providing weather data",
                })

            # Get hourly forecast for better prediction
            forecast_hours = None
            weather_corrected = await _read_json_file(SOLAR_PATH / "stats" / "weather_forecast_corrected.json")
            if weather_corrected and "forecast" in weather_corrected:
                today_str = date.today().isoformat()
                if today_str in weather_corrected["forecast"]:
                    forecast_hours = list(weather_corrected["forecast"][today_str].values())

            # Build weather input dict
            weather_input = {
                "temperature": weather_data.get("temperature", 15),
                "humidity": weather_data.get("humidity", 50),
                "wind_speed": weather_data.get("wind_speed", 0),
                "precipitation": weather_data.get("precipitation", 0),
                "cloud_cover": weather_data.get("cloud_cover", weather_data.get("cloud_coverage", 50)),
                "pressure": weather_data.get("pressure", 1013),
                "uv_index": weather_data.get("uv_index", 0),
                "radiation": weather_data.get("radiation", 0),
            }

            recommendation = get_recommendation(weather_input, forecast_hours)

            return web.json_response({
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "recommendation": {
                    "unterbekleidung": recommendation.unterbekleidung,
                    "unterbekleidung_icon": recommendation.unterbekleidung_icon,
                    "oberbekleidung": recommendation.oberbekleidung,
                    "oberbekleidung_icon": recommendation.oberbekleidung_icon,
                    "jacke": recommendation.jacke,
                    "jacke_icon": recommendation.jacke_icon,
                    "kopfbedeckung": recommendation.kopfbedeckung,
                    "kopfbedeckung_icon": recommendation.kopfbedeckung_icon,
                    "zusaetze": recommendation.zusaetze,
                    "zusaetze_icons": recommendation.zusaetze_icons,
                    "text_de": recommendation.text_de,
                    "text_en": recommendation.text_en,
                },
                "weather_summary": recommendation.weather_summary,
            })

        except Exception as err:
            _LOGGER.error("Error getting clothing recommendation: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err),
            }, status=500)


class ForecastComparisonView(HomeAssistantView):
    """API for forecast comparison data."""

    url = "/api/sfml_stats_lite/forecast_comparison"
    name = "api:sfml_stats_lite:forecast_comparison"
    requires_auth = False
    cors_allowed = True

    @local_only
    async def get(self, request: Request) -> Response:
        """Return forecast comparison data for chart rendering."""
        try:
            from pathlib import Path
            from ..readers.forecast_comparison_reader import ForecastComparisonReader

            days = int(request.query.get("days", 7))
            config_path = Path(HASS.config.path()) if HASS else None

            if not config_path:
                return web.json_response({
                    "success": False,
                    "error": "HASS not initialized",
                }, status=500)

            reader = ForecastComparisonReader(config_path)

            if not reader.is_available:
                return web.json_response({
                    "success": True,
                    "data": {
                        "dates": [],
                        "actual": [],
                        "sfml": [],
                        "external_1": None,
                        "external_1_name": None,
                        "external_2": None,
                        "external_2_name": None,
                        "stats": {
                            "days_count": 0,
                            "days_with_actual": 0,
                            "sfml_avg_accuracy": None,
                            "external_1_avg_accuracy": None,
                            "external_2_avg_accuracy": None,
                            "best_forecast": None,
                        },
                    },
                    "message": "No forecast comparison data available yet",
                })

            chart_data = await reader.async_get_chart_data(days=days)

            return web.json_response({
                "success": True,
                "data": chart_data,
            })

        except Exception as err:
            _LOGGER.error("Error getting forecast comparison data: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err),
            }, status=500)
