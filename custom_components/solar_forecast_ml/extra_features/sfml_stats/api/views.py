# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""REST API views for SFML Stats dashboard. @zara"""
from __future__ import annotations

import asyncio
import functools
import ipaddress
import json
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

import aiosqlite
from aiohttp import web
from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant


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
    """Extract real client IP from request. @zara"""
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
    """Check if IP is in local network range. @zara"""
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in network for network in LOCAL_NETWORKS)
    except ValueError:
        return False


def local_only(func):
    """Decorator: Block external (non-local) requests. @zara"""
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
                "<p>SFML Stats is only accessible from the local network.</p>"
                "</body></html>",
                status=403,
                content_type="text/html"
            )
        return await func(self, request, *args, **kwargs)
    return wrapper

from ..const import (
    DOMAIN,
    VERSION,
    CONF_SENSOR_SOLAR_TO_HOUSE,
    CONF_SENSOR_SOLAR_TO_BATTERY,
    CONF_SENSOR_BATTERY_TO_HOUSE,
    CONF_SENSOR_GRID_TO_HOUSE,
    CONF_SENSOR_GRID_TO_BATTERY,
    CONF_SENSOR_HOUSE_TO_GRID,
    CONF_SENSOR_SMARTMETER_IMPORT,
    CONF_SENSOR_SMARTMETER_EXPORT,
    CONF_SENSOR_BATTERY_SOC,
    CONF_SENSOR_BATTERY_POWER,
    CONF_SENSOR_HOME_CONSUMPTION,
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
    CONF_BILLING_PRICE_MODE,
    DEFAULT_BILLING_PRICE_MODE,
    CONF_PANEL_GROUP_NAMES,
    CONF_DASHBOARD_STYLE,
    DEFAULT_DASHBOARD_STYLE,
    CONF_THEME,
    DEFAULT_THEME,
    CONF_SENSOR_HEATPUMP_POWER,
    CONF_SENSOR_HEATPUMP_DAILY,
    CONF_SENSOR_HEATPUMP_COP,
    CONF_SENSOR_HEATINGROD_POWER,
    CONF_SENSOR_HEATINGROD_DAILY,
    CONF_SENSOR_WALLBOX_POWER,
    CONF_SENSOR_WALLBOX_DAILY,
    CONF_SENSOR_WALLBOX_STATE,
    DEFAULT_HEATPUMP_COP,
    SOLAR_FORECAST_DB,
    CONF_FORECAST_ENTITY_1_NAME,
    CONF_FORECAST_ENTITY_2_NAME,
    DEFAULT_FORECAST_ENTITY_1_NAME,
    DEFAULT_FORECAST_ENTITY_2_NAME,
)
from ..utils import get_json_cache, read_json_safe
from ..readers.solar_reader import SolarDataReader, DailyForecast
from ..readers.weather_reader import WeatherDataReader
from ..sfml_data_reader import SFMLDataReader

if TYPE_CHECKING:
    from aiohttp.web import Request, Response

_LOGGER = logging.getLogger(__name__)


class APIContext:
    """Singleton context for API views. @zara"""

    _instance: "APIContext | None" = None

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the context. @zara"""
        self.hass = hass
        self.config_path = Path(hass.config.path())
        self.solar_path = self.config_path / "solar_forecast_ml"
        self.grid_path = self.config_path / "grid_price_monitor"

    @classmethod
    def get(cls) -> "APIContext":
        """Get the singleton instance. @zara"""
        if cls._instance is None:
            raise RuntimeError("APIContext not initialized - call initialize() first")
        return cls._instance

    @classmethod
    def initialize(cls, hass: HomeAssistant) -> "APIContext":
        """Initialize the singleton instance. @zara"""
        cls._instance = cls(hass)
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if context is initialized. @zara"""
        return cls._instance is not None


SOLAR_PATH: Path | None = None
GRID_PATH: Path | None = None
HASS: HomeAssistant | None = None


@asynccontextmanager
async def _get_db() -> AsyncIterator[aiosqlite.Connection]:
    """Get DB connection via manager with direct fallback. @zara"""
    from ..storage.db_connection_manager import get_manager
    manager = get_manager()
    if manager is not None and manager.is_connected:
        yield await manager.get_connection()
        return
    db_path = Path(HASS.config.path()) / SOLAR_FORECAST_DB
    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


async def async_setup_views(hass: HomeAssistant) -> None:
    """Register all API views. @zara"""
    global SOLAR_PATH, GRID_PATH, HASS

    ctx = APIContext.initialize(hass)

    HASS = hass
    config_path = Path(hass.config.path())
    SOLAR_PATH = config_path / "solar_forecast_ml"
    GRID_PATH = config_path / "grid_price_monitor"

    _LOGGER.debug("SFML Stats paths: Solar=%s, Grid=%s", ctx.solar_path, ctx.grid_path)

    hass.http.register_view(HealthCheckView())
    hass.http.register_view(DashboardView())
    hass.http.register_view(LcarsDashboardView())
    hass.http.register_view(TariffDashboardView())
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
    hass.http.register_view(WeatherComparisonView())
    hass.http.register_view(ExportWeatherAnalyticsView())
    hass.http.register_view(PowerSourcesHistoryView())
    hass.http.register_view(ExportPowerSourcesView())
    hass.http.register_view(EnergySourcesDailyStatsView())
    hass.http.register_view(ClothingRecommendationView())

    hass.http.register_view(MonthlyTariffsView())
    hass.http.register_view(MonthlyTariffDetailView())
    hass.http.register_view(MonthlyTariffFinalizeView())
    hass.http.register_view(MonthlyTariffsExportView())
    hass.http.register_view(MonthlyTariffsDefaultsView())

    hass.http.register_view(ExportWeeklyReportView())

    hass.http.register_view(BackgroundImageView())

    hass.http.register_view(ForecastComparisonView())
    hass.http.register_view(ForecastComparisonChartView())
    hass.http.register_view(ShadowAnalyticsView())
    hass.http.register_view(AIStatusView())

    hass.http.register_view(DashboardSettingsView())

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


def _get_solar_reader() -> SolarDataReader:
    """Get a SolarDataReader instance for database access. @zara"""
    if SOLAR_PATH is None:
        raise RuntimeError("SOLAR_PATH not initialized - was async_setup_views called?")
    return SolarDataReader(SOLAR_PATH.parent)


async def _get_today_yield_from_db() -> float | None:
    """Get today's solar yield from prediction_panel_groups (sum of all panels). @zara"""
    try:
        async with _get_db() as db:
            async with db.execute(
                """SELECT SUM(ppg.actual_kwh) as total
                   FROM prediction_panel_groups ppg
                   JOIN hourly_predictions hp ON hp.prediction_id = ppg.prediction_id
                   WHERE hp.target_date = ?""",
                (date.today().isoformat(),)
            ) as cursor:
                row = await cursor.fetchone()

        if row and row["total"] is not None:
            return row["total"]
        return None
    except Exception as e:
        _LOGGER.error("Error getting today's yield from database: %s", e)
        return None


async def _get_weather_from_db(days: int = 7) -> dict[str, dict[str, dict]]:
    """Get hourly weather data from database. @zara"""
    try:
        if SOLAR_PATH is None:
            return {}

        weather_reader = WeatherDataReader(SOLAR_PATH.parent)
        if not weather_reader.is_available:
            _LOGGER.debug("Weather database not available")
            return {}

        cutoff = date.today() - timedelta(days=days)
        hourly_weather = await weather_reader.async_get_hourly_weather(start_date=cutoff)

        if not hourly_weather:
            _LOGGER.debug("No hourly weather data in database")
            return {}

        result: dict[str, dict[str, dict]] = {}
        for w in hourly_weather:
            date_str = w.date.isoformat()
            hour_str = str(w.hour)

            if date_str not in result:
                result[date_str] = {}

            result[date_str][hour_str] = {
                "temperature_c": w.temperature_c,
                "humidity_percent": w.humidity_percent,
                "wind_speed_ms": w.wind_speed_ms,
                "precipitation_mm": w.precipitation_mm,
                "solar_radiation_wm2": w.solar_radiation_wm2,
                "cloud_cover_percent": w.cloud_cover_percent,
            }

        _LOGGER.debug("Loaded weather data for %d days from database", len(result))
        return result

    except Exception as e:
        _LOGGER.error("Error getting weather from database: %s", e)
        return {}


async def _get_weather_forecast_from_db(days: int = 3) -> dict[str, dict[str, dict]]:
    """Get corrected weather forecast from database. @zara"""
    try:
        if SOLAR_PATH is None:
            return {}

        weather_reader = WeatherDataReader(SOLAR_PATH.parent)
        if not weather_reader.is_available:
            _LOGGER.debug("Weather database not available for forecast")
            return {}

        start = date.today()
        end = start + timedelta(days=days)
        forecast_weather = await weather_reader.async_get_forecast_weather(
            start_date=start, end_date=end
        )

        if not forecast_weather:
            _LOGGER.debug("No forecast weather data in database")
            return {}

        result: dict[str, dict[str, dict]] = {}
        for w in forecast_weather:
            date_str = w.date.isoformat()
            hour_str = str(w.hour)

            if date_str not in result:
                result[date_str] = {}

            result[date_str][hour_str] = {
                "temperature": w.temperature_c,
                "humidity": w.humidity_percent,
                "wind_speed": w.wind_speed_ms,
                "precipitation": w.precipitation_mm,
                "cloud_cover": w.cloud_cover_percent,
                "solar_radiation_wm2": w.solar_radiation_wm2,
            }

        _LOGGER.debug("Loaded forecast weather for %d days from database", len(result))
        return result

    except Exception as e:
        _LOGGER.error("Error getting weather forecast from database: %s", e)
        return {}


class HealthCheckView(HomeAssistantView):
    """Health check endpoint for monitoring. @zara"""

    url = "/api/sfml_stats/health"
    name = "api:sfml_stats:health"
    requires_auth = False

    @local_only
    async def get(self, request: Request) -> Response:
        """Return health status. @zara"""
        try:
            ctx = APIContext.get()

            checks = {
                "solar_data_available": ctx.solar_path.exists(),
                "grid_data_available": ctx.grid_path.exists(),
                "integration_loaded": DOMAIN in ctx.hass.data,
                "config_entries_present": len(
                    ctx.hass.config_entries.async_entries(DOMAIN)
                ) > 0,
            }

            if checks["solar_data_available"]:
                checks["solar_stats_available"] = (
                    ctx.solar_path / "solar_forecast.db"
                ).exists()
            else:
                checks["solar_stats_available"] = False

            if checks["grid_data_available"]:
                checks["grid_prices_available"] = (
                    ctx.grid_path / "data" / "price_cache.json"
                ).exists()
            else:
                checks["grid_prices_available"] = False

            critical_checks = [
                checks["integration_loaded"],
                checks["config_entries_present"],
            ]
            all_healthy = all(critical_checks)
            degraded = not all(checks.values()) and all_healthy

            if all_healthy and not degraded:
                status = "healthy"
                status_code = 200
            elif degraded:
                status = "degraded"
                status_code = 200
            else:
                status = "unhealthy"
                status_code = 503

            return web.json_response(
                {
                    "status": status,
                    "version": VERSION,
                    "checks": checks,
                    "timestamp": datetime.now().isoformat(),
                },
                status=status_code,
            )

        except RuntimeError:
            return web.json_response(
                {
                    "status": "unhealthy",
                    "version": VERSION,
                    "error": "API context not initialized",
                    "timestamp": datetime.now().isoformat(),
                },
                status=503,
            )
        except Exception as err:
            _LOGGER.error("Health check error: %s", err)
            return web.json_response(
                {
                    "status": "error",
                    "version": VERSION,
                    "error": str(err),
                    "timestamp": datetime.now().isoformat(),
                },
                status=500,
            )


class TariffDashboardView(HomeAssistantView):
    """Serves the monthly tariff management page. @zara"""

    url = "/api/sfml_stats/tariffs"
    name = "api:sfml_stats:tariffs"
    requires_auth = False
    cors_allowed = True

    @local_only
    async def get(self, request: Request) -> Response:
        """Return the tariffs HTML page. @zara"""
        tariff_html_path = Path(__file__).parent.parent / "frontend" / "dist" / "tariffs.html"

        if not tariff_html_path.exists():
            return web.Response(
                text="Tariff dashboard not found. Please check installation.",
                status=404,
                content_type="text/plain"
            )

        try:
            import aiofiles
            async with aiofiles.open(tariff_html_path, "r", encoding="utf-8") as f:
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
        except Exception as err:
            _LOGGER.error("Error loading tariff dashboard: %s", err)
            return web.Response(
                text=f"Error loading tariff dashboard: {err}",
                status=500,
                content_type="text/plain"
            )


class DashboardView(HomeAssistantView):
    """Main view serving the Vue.js app. @zara"""

    url = "/api/sfml_stats/dashboard"
    name = "api:sfml_stats:dashboard"
    requires_auth = False
    cors_allowed = True

    @local_only
    async def get(self, request: Request) -> Response:
        """Return the dashboard HTML page. @zara"""
        frontend_path = None

        if HASS is not None:
            frontend_path = Path(HASS.config.path()) / "custom_components" / "sfml_stats" / "frontend" / "dist" / "index.html"
            if not frontend_path.exists():
                frontend_path = None

        if frontend_path is None:
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


class LcarsDashboardView(HomeAssistantView):
    """LCARS Star Trek style dashboard view. @zara"""

    url = "/api/sfml_stats/lcars"
    name = "api:sfml_stats:lcars"
    requires_auth = False
    cors_allowed = True

    @local_only
    async def get(self, request: Request) -> Response:
        """Return the LCARS dashboard HTML page. @zara"""
        frontend_path = None

        if HASS is not None:
            frontend_path = Path(HASS.config.path()) / "custom_components" / "sfml_stats" / "frontend" / "dist" / "index-lcars.html"
            if not frontend_path.exists():
                frontend_path = None

        if frontend_path is None:
            frontend_path = Path(__file__).parent.parent / "frontend" / "dist" / "index-lcars.html"

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
        """Return fallback HTML when LCARS build is missing. @zara"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>SFML Stats - LCARS</title>
    <style>
        body {
            background: #000;
            color: #FF9966;
            font-family: 'Arial Narrow', Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            text-transform: uppercase;
        }
        .message { text-align: center; }
        h1 { color: #FF9966; letter-spacing: 0.2em; }
        a { color: #9999FF; }
    </style>
</head>
<body>
    <div class="message">
        <h1>LCARS Interface</h1>
        <p>Frontend wird initialisiert...</p>
        <p style="color: #666;">Falls diese Meldung bleibt, wurde das LCARS-Frontend noch nicht erstellt.</p>
        <p><a href="/api/sfml_stats/dashboard">Zum Standard-Dashboard</a></p>
    </div>
</body>
</html>"""


class StaticFilesView(HomeAssistantView):
    """Serve static files (JS, CSS, Assets). @zara"""

    url = "/api/sfml_stats/assets/{filename:.*}"
    name = "api:sfml_stats:assets"
    requires_auth = False

    @local_only
    async def get(self, request: Request, filename: str) -> Response:
        """Return a static file. @zara"""
        frontend_path = None

        if filename.startswith("css/") or filename.endswith(".css"):
            subdir = "css"
            clean_filename = filename[4:] if filename.startswith("css/") else filename
        elif filename.startswith("js/") or filename.endswith(".js"):
            subdir = "js"
            clean_filename = filename[3:] if filename.startswith("js/") else filename
        else:
            subdir = "assets"
            clean_filename = filename

        if HASS is not None:
            frontend_path = Path(HASS.config.path()) / "custom_components" / "sfml_stats" / "frontend" / "dist" / subdir / clean_filename
            if not frontend_path.exists():
                frontend_path = Path(HASS.config.path()) / "custom_components" / "sfml_stats" / "frontend" / "dist" / "assets" / filename
                if not frontend_path.exists():
                    frontend_path = None

        if frontend_path is None:
            frontend_path = Path(__file__).parent.parent / "frontend" / "dist" / subdir / clean_filename
            if not frontend_path.exists():
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
        elif filename.endswith(".webp"):
            content_type = "image/webp"
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

    @local_only
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
            try:
                reader = _get_solar_reader()
                cutoff = date.today() - timedelta(days=days)
                summaries = await reader.async_get_daily_summaries(
                    start_date=cutoff,
                    end_date=date.today()
                )
                result["data"]["daily"] = [
                    {
                        "date": s.date.isoformat(),
                        "overall": {
                            "predicted_total_kwh": s.predicted_total_kwh,
                            "actual_total_kwh": s.actual_total_kwh,
                            "accuracy_percent": s.accuracy_percent,
                            "error_kwh": s.error_kwh,
                            "production_hours": s.production_hours,
                            "peak_hour": s.peak_hour,
                            "peak_kwh": s.peak_kwh,
                        },
                        "time_windows": {
                            "morning_accuracy": s.morning_accuracy,
                            "midday_accuracy": s.midday_accuracy,
                            "afternoon_accuracy": s.afternoon_accuracy,
                        },
                        "ml_metrics": {
                            "mae": s.ml_mae,
                            "rmse": s.ml_rmse,
                            "r2_score": s.ml_r2_score,
                        }
                    }
                    for s in summaries
                ]
            except Exception as e:
                _LOGGER.error("Error loading daily summaries from database: %s", e)
                result["data"]["daily"] = []

        if include_hourly:
            try:
                reader = _get_solar_reader()
                cutoff_date = date.today() - timedelta(days=days)
                all_predictions = []
                current = cutoff_date
                while current <= date.today():
                    day_predictions = await reader.async_get_hourly_predictions(target_date=current)
                    for p in day_predictions:
                        all_predictions.append({
                            "target_datetime": p.target_datetime.isoformat(),
                            "target_hour": p.target_hour,
                            "target_date": p.target_date.isoformat(),
                            "prediction_kwh": p.prediction_kwh,
                            "actual_kwh": p.actual_kwh,
                            "accuracy_percent": p.accuracy_percent,
                            "error_kwh": p.error_kwh,
                            "prediction_method": p.prediction_method,
                            "ml_contribution_percent": p.ml_contribution_percent,
                            "confidence": p.confidence,
                            "temperature": p.temperature,
                            "solar_radiation": p.solar_radiation,
                            "clouds": p.clouds,
                            "sun_elevation": p.sun_elevation,
                            "theoretical_max_kwh": p.theoretical_max_kwh,
                        })
                    current += timedelta(days=1)
                result["data"]["hourly"] = all_predictions
            except Exception as e:
                _LOGGER.error("Error loading hourly predictions from database: %s", e)
                result["data"]["hourly"] = []

        weather_db = await _get_weather_from_db(days=days)
        if weather_db:
            result["data"]["weather"] = weather_db

        weather_corrected = await _get_weather_forecast_from_db(days=days)
        if weather_corrected:
            result["data"]["weather_corrected"] = weather_corrected

        try:
            reader = _get_solar_reader()
            model_state = await reader.async_get_model_state()
            if model_state:
                result["data"]["ai_state"] = {
                    "model_loaded": model_state.model_loaded,
                    "algorithm_used": model_state.algorithm_used,
                    "training_samples": model_state.training_samples,
                    "current_accuracy": model_state.current_accuracy,
                    "last_training": model_state.last_training.isoformat() if model_state.last_training else None,
                    "peak_power_kw": model_state.peak_power_kw,
                    "feature_weights": model_state.feature_weights,
                    "feature_importance": model_state.feature_importance,
                }
        except Exception as e:
            _LOGGER.error("Error loading AI model state from database: %s", e)

        try:
            reader = _get_solar_reader()
            daily_forecasts = await reader.async_get_daily_forecasts()

            today_fc = daily_forecasts.get("today")
            tomorrow_fc = daily_forecasts.get("tomorrow")
            day_after_fc = daily_forecasts.get("day_after_tomorrow")

            result["data"]["forecasts"] = {
                "today": {
                    "prediction_kwh": today_fc.prediction_kwh if today_fc else None,
                    "prediction_kwh_display": f"{today_fc.prediction_kwh:.2f}" if today_fc else None,
                },
                "tomorrow": {
                    "date": tomorrow_fc.forecast_date.isoformat() if tomorrow_fc and tomorrow_fc.forecast_date else None,
                    "prediction_kwh": tomorrow_fc.prediction_kwh if tomorrow_fc else None,
                    "prediction_kwh_display": f"{tomorrow_fc.prediction_kwh:.2f}" if tomorrow_fc else None,
                },
                "day_after_tomorrow": {
                    "date": day_after_fc.forecast_date.isoformat() if day_after_fc and day_after_fc.forecast_date else None,
                    "prediction_kwh": day_after_fc.prediction_kwh if day_after_fc else None,
                    "prediction_kwh_display": f"{day_after_fc.prediction_kwh:.2f}" if day_after_fc else None,
                },
            }
        except Exception as e:
            _LOGGER.error("Error loading forecasts from database: %s", e)
            result["data"]["forecasts"] = {
                "today": {"prediction_kwh": None, "prediction_kwh_display": None},
                "tomorrow": {"date": None, "prediction_kwh": None, "prediction_kwh_display": None},
                "day_after_tomorrow": {"date": None, "prediction_kwh": None, "prediction_kwh_display": None},
            }

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

        try:
            reader = _get_solar_reader()
            today = date.today()
            tomorrow = today + timedelta(days=1)
            day_after = today + timedelta(days=2)

            multi_day_data = {}

            for target_date in [today, tomorrow, day_after]:
                predictions = await reader.async_get_hourly_predictions(target_date=target_date)
                if predictions:
                    hourly_data = []
                    for p in predictions:
                        hourly_data.append({
                            "hour": p.target_hour,
                            "prediction_kwh": p.prediction_kwh,
                            "actual_kwh": p.actual_kwh,
                        })
                    multi_day_data[target_date.isoformat()] = {
                        "date": target_date.isoformat(),
                        "hourly": hourly_data,
                        "total_kwh": sum(p.prediction_kwh for p in predictions),
                    }

            if multi_day_data:
                result["data"]["multi_day_hourly"] = multi_day_data

        except Exception as e:
            _LOGGER.error("Error loading multi-day hourly forecast from database: %s", e)

        return web.json_response(result)


class PriceDataView(HomeAssistantView):
    """API for electricity price data. @zara"""

    url = "/api/sfml_stats/prices"
    name = "api:sfml_stats:prices"
    requires_auth = False

    @local_only
    async def get(self, request: Request) -> Response:
        """Return price data. @zara"""
        days = int(request.query.get("days", 7))

        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "data": {},
        }

        try:
            cutoff_date = (date.today() - timedelta(days=days)).isoformat()

            async with _get_db() as db:
                async with db.execute("""
                    SELECT timestamp, hour, price_net, total_price,
                           date(timestamp, '+1 hour') as price_date
                    FROM GPM_price_history
                    WHERE date(timestamp, '+1 hour') >= ?
                    ORDER BY timestamp
                """, (cutoff_date,)) as cursor:
                    rows = await cursor.fetchall()

            if rows:
                result["data"]["prices"] = [
                    {
                        "timestamp": row["timestamp"],
                        "date": row["price_date"],
                        "hour": row["hour"],
                        "price_net": row["price_net"] or 0,
                        "price_total": row["total_price"] or 0,
                    }
                    for row in rows
                ]
        except Exception as e:
            _LOGGER.error("Error loading prices from DB: %s", e)

        return web.json_response(result)


class SummaryDataView(HomeAssistantView):
    """API for aggregated dashboard data. @zara"""

    url = "/api/sfml_stats/summary"
    name = "api:sfml_stats:summary"
    requires_auth = False

    @local_only
    async def get(self, request: Request) -> Response:
        """Return a summary for the dashboard. @zara"""
        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "kpis": {},
            "today": {},
            "week": {},
        }

        today = date.today()
        week_ago = today - timedelta(days=7)

        try:
            reader = _get_solar_reader()
            summaries = await reader.async_get_daily_summaries(
                start_date=week_ago,
                end_date=today
            )

            today_data = next((s for s in summaries if s.date == today), None)
            if today_data:
                result["today"] = {
                    "production": today_data.actual_total_kwh,
                    "forecast": today_data.predicted_total_kwh,
                    "accuracy": today_data.accuracy_percent,
                    "peak_hour": today_data.peak_hour,
                    "peak_kwh": today_data.peak_kwh,
                }

            if summaries:
                result["week"] = {
                    "total_production": sum(s.actual_total_kwh for s in summaries),
                    "total_forecast": sum(s.predicted_total_kwh for s in summaries),
                    "avg_accuracy": sum(s.accuracy_percent for s in summaries) / len(summaries) if summaries else 0,
                    "days_count": len(summaries),
                }
        except Exception as e:
            _LOGGER.error("Error loading daily summaries from database: %s", e)

        try:
            async with _get_db() as db:
                async with db.execute("""
                    SELECT price_net, total_price
                    FROM GPM_price_history
                    WHERE timestamp >= datetime('now', '-48 hours')
                    ORDER BY timestamp
                """) as cursor:
                    price_rows = await cursor.fetchall()
            if price_rows:
                recent_prices = [r["price_net"] for r in price_rows if r["price_net"]]
                if recent_prices:
                    result["kpis"]["price_current"] = recent_prices[-1]
                    result["kpis"]["price_avg"] = sum(recent_prices) / len(recent_prices)
                    result["kpis"]["price_min"] = min(recent_prices)
                    result["kpis"]["price_max"] = max(recent_prices)
        except Exception as e:
            _LOGGER.error("Error loading price KPIs from DB: %s", e)

        try:
            reader = _get_solar_reader()
            model_state = await reader.async_get_model_state()
            if model_state:
                result["kpis"]["ai_training_samples"] = model_state.training_samples
        except Exception as e:
            _LOGGER.error("Error loading AI model state from database: %s", e)

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

    @local_only
    async def get(self, request: Request) -> Response:
        """Return current realtime data. @zara"""
        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "current_hour": datetime.now().hour,
            "data": {},
        }

        try:
            reader = _get_solar_reader()
            now = datetime.now()
            predictions = await reader.async_get_hourly_predictions(target_date=now.date())
            current = next((p for p in predictions if p.target_hour == now.hour), None)

            actual_kwh = current.actual_kwh if current else None

            if current:
                result["data"]["solar"] = {
                    "prediction_kwh": current.prediction_kwh,
                    "actual_kwh": actual_kwh,
                    "weather": {
                        "temperature": current.temperature,
                        "solar_radiation": current.solar_radiation,
                        "clouds": current.clouds,
                    },
                    "astronomy": {
                        "sun_elevation": current.sun_elevation,
                        "theoretical_max_kwh": current.theoretical_max_kwh,
                    },
                }
        except Exception as e:
            _LOGGER.error("Error loading realtime prediction from database: %s", e)

        try:
            now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")
            current_hour = now.hour
            async with _get_db() as db:
                async with db.execute("""
                    SELECT price_net, hour
                    FROM GPM_price_history
                    WHERE date(timestamp, '+1 hour') = ? AND hour = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (today_str, current_hour)) as cursor:
                    price_row = await cursor.fetchone()
            if price_row:
                result["data"]["price"] = {
                    "current": price_row["price_net"] or 0,
                    "hour": price_row["hour"],
                }
        except Exception as e:
            _LOGGER.error("Error loading current price from DB: %s", e)

        today_str = date.today().isoformat()
        hour_str = str(datetime.now().hour)
        weather_db = await _get_weather_from_db(days=1)
        if weather_db and today_str in weather_db and hour_str in weather_db[today_str]:
            result["data"]["weather_actual"] = weather_db[today_str][hour_str]

        return web.json_response(result)


def _get_config() -> dict[str, Any]:
    """Get current configuration from the first config entry. @zara"""
    if HASS is None:
        return {}

    entries = HASS.data.get(DOMAIN, {})

    for entry_id, entry_data in entries.items():
        if isinstance(entry_data, dict) and "config" in entry_data:
            return entry_data["config"]

    config_entries = HASS.config_entries.async_entries(DOMAIN)
    if config_entries:
        entry = config_entries[0]
        return dict(entry.data)

    return {}


def _get_sfml_panel_groups() -> list[dict[str, Any]]:
    """Get panel groups configuration from SFML. @zara"""
    if HASS is None:
        return []

    try:
        sfml_domain = "solar_forecast_ml"
        entries = HASS.data.get(sfml_domain, {})

        for entry_id, entry_data in entries.items():
            if hasattr(entry_data, "panel_groups"):
                panel_groups = entry_data.panel_groups
                if panel_groups:
                    _LOGGER.debug("Found %d panel groups from SFML coordinator", len(panel_groups))
                    return panel_groups
        return []
    except Exception as e:
        _LOGGER.debug("Could not get SFML panel groups: %s", e)
        return []


def _read_panel_group_sensor(entity_id: str) -> float | None:
    """Read current value from a panel group energy sensor. @zara"""
    if not HASS or not entity_id:
        return None

    try:
        state = HASS.states.get(entity_id)
        if state is None or state.state in ["unavailable", "unknown", None]:
            return None

        value = float(state.state)

        unit = state.attributes.get("unit_of_measurement", "")
        if unit.lower() == "wh":
            value = value / 1000.0

        return round(value, 3)
    except (ValueError, TypeError):
        return None


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

    wind_speed = attrs.get("wind_speed")
    if wind_speed is not None:
        wind_speed_unit = attrs.get("wind_speed_unit", "km/h")
        if wind_speed_unit == "km/h":
            wind_speed = round(wind_speed / 3.6, 1)

    return {
        "state": state.state,
        "temperature": attrs.get("temperature"),
        "humidity": attrs.get("humidity"),
        "wind_speed": wind_speed,
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

    @local_only
    async def get(self, request: Request) -> Response:
        """Return current energy flow data. @zara"""
        config = _get_config()

        sfml_reader = SFMLDataReader(HASS)

        solar_power = sfml_reader.get_live_power()
        solar_to_house = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_TO_HOUSE))
        solar_to_battery = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_TO_BATTERY))

        if solar_power is not None and solar_power < 0:
            solar_power = 0.0
        if solar_to_house is not None and solar_to_house < 0:
            solar_to_house = 0.0
        if solar_to_battery is not None and solar_to_battery < 0:
            solar_to_battery = 0.0

        if solar_power is not None and solar_power <= 0:
            solar_to_house = 0.0
            solar_to_battery = 0.0
        elif solar_power is not None and solar_to_battery is not None:
            solar_to_battery = min(solar_to_battery, solar_power)
        if solar_power is not None and solar_to_house is not None:
            solar_to_house = min(solar_to_house, solar_power)

        # Fallback: solar_to_house berechnen wenn kein Sensor konfiguriert
        if solar_to_house is None and solar_power is not None and solar_power > 0:
            solar_to_battery_val = solar_to_battery or 0
            solar_to_house = max(0, solar_power - solar_to_battery_val)

        battery_configured = config.get(CONF_SENSOR_BATTERY_SOC) is not None
        battery_soc = _get_sensor_value(config.get(CONF_SENSOR_BATTERY_SOC)) if battery_configured else None
        battery_power = _get_sensor_value(config.get(CONF_SENSOR_BATTERY_POWER)) if battery_configured else None
        battery_to_house = _get_sensor_value(config.get(CONF_SENSOR_BATTERY_TO_HOUSE)) if battery_configured else None
        grid_to_battery = _get_sensor_value(config.get(CONF_SENSOR_GRID_TO_BATTERY)) if battery_configured else None

        if not battery_configured:
            solar_to_battery = None

        if battery_configured and (battery_power is None or battery_power == 0):
            charge_power = (solar_to_battery or 0) + (grid_to_battery or 0)
            discharge_power = battery_to_house or 0
            if charge_power > 0:
                battery_power = charge_power
            elif discharge_power > 0:
                battery_power = -discharge_power

        solar_yield_daily_db = await _get_today_yield_from_db()
        solar_yield_daily = solar_yield_daily_db if solar_yield_daily_db is not None else sfml_reader.get_live_yield()

        # grid_to_house / house_to_grid: Fallback auf Smartmeter-Sensoren
        grid_to_house = _get_sensor_value(config.get(CONF_SENSOR_GRID_TO_HOUSE))
        if grid_to_house is None:
            grid_to_house = _get_sensor_value(config.get(CONF_SENSOR_SMARTMETER_IMPORT))

        house_to_grid = _get_sensor_value(config.get(CONF_SENSOR_HOUSE_TO_GRID))
        if house_to_grid is None:
            house_to_grid = _get_sensor_value(config.get(CONF_SENSOR_SMARTMETER_EXPORT))

        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "flows": {
                "solar_power": solar_power,
                "solar_to_house": solar_to_house,
                "solar_to_battery": solar_to_battery,
                "battery_to_house": battery_to_house,
                "grid_to_house": grid_to_house,
                "grid_to_battery": grid_to_battery,
                "house_to_grid": house_to_grid,
            },
            "battery": {
                "soc": battery_soc,
                "power": battery_power,
            },
            "home": {
                "consumption": _get_sensor_value(config.get(CONF_SENSOR_HOME_CONSUMPTION)),
            },
            "statistics": {
                "solar_yield_daily": solar_yield_daily,
                "grid_import_daily": _get_sensor_value(config.get(CONF_SENSOR_GRID_IMPORT_DAILY)),
                "grid_import_yearly": _get_sensor_value(config.get(CONF_SENSOR_GRID_IMPORT_YEARLY)),
                "battery_charge_solar_daily": _get_sensor_value(config.get(CONF_SENSOR_BATTERY_CHARGE_SOLAR_DAILY)),
                "battery_charge_grid_daily": _get_sensor_value(config.get(CONF_SENSOR_BATTERY_CHARGE_GRID_DAILY)),
                "price_total": _get_sensor_value(config.get(CONF_SENSOR_PRICE_TOTAL)),
            },
            "configured_sensors": {
                "solar_power": sfml_reader.get_power_entity_id() or "(from SFML)",
                "solar_to_house": config.get(CONF_SENSOR_SOLAR_TO_HOUSE),
                "solar_to_battery": config.get(CONF_SENSOR_SOLAR_TO_BATTERY),
                "battery_to_house": config.get(CONF_SENSOR_BATTERY_TO_HOUSE),
                "grid_to_house": config.get(CONF_SENSOR_GRID_TO_HOUSE),
                "grid_to_battery": config.get(CONF_SENSOR_GRID_TO_BATTERY),
                "house_to_grid": config.get(CONF_SENSOR_HOUSE_TO_GRID),
                "battery_soc": config.get(CONF_SENSOR_BATTERY_SOC),
                "home_consumption": config.get(CONF_SENSOR_HOME_CONSUMPTION),
                "solar_yield_daily": sfml_reader.get_yield_entity_id() or "(from SFML)",
                "weather_entity": config.get(CONF_WEATHER_ENTITY),
            },
            "panels": self._get_panel_data(config),
            "consumers": self._get_consumer_data(config),
            "weather_ha": _get_weather_data(config.get(CONF_WEATHER_ENTITY)),
            "sun_position": await self._get_sun_position(),
            "current_price": await self._get_current_price(),
            "feed_in_tariff": config.get(CONF_FEED_IN_TARIFF, DEFAULT_FEED_IN_TARIFF),
            "price_mode": config.get(CONF_BILLING_PRICE_MODE, DEFAULT_BILLING_PRICE_MODE),
        }

        return web.json_response(result)

    async def _get_current_price(self) -> dict[str, Any] | None:
        """Read current electricity price from GPM_price_history DB. @zara"""
        try:
            today_str = date.today().isoformat()
            current_hour = datetime.now().hour

            async with _get_db() as db:
                async with db.execute("""
                    SELECT price_net, total_price, hour
                    FROM GPM_price_history
                    WHERE date(timestamp, '+1 hour') = ? AND hour = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (today_str, current_hour)) as cursor:
                    row = await cursor.fetchone()

            if row:
                return {
                    "total_price": row["total_price"],
                    "net_price": row["price_net"],
                    "hour": current_hour,
                }
        except Exception as e:
            _LOGGER.error("Error loading current price from DB: %s", e)
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

    def _get_consumer_data(self, config: dict[str, Any]) -> dict[str, Any]:
        """Read consumer data from configured sensors. @zara"""
        consumers = {
            "heatpump": None,
            "heatingrod": None,
            "wallbox": None,
        }

        if config.get(CONF_SENSOR_HEATPUMP_POWER):
            power = _get_sensor_value(config.get(CONF_SENSOR_HEATPUMP_POWER))
            daily = _get_sensor_value(config.get(CONF_SENSOR_HEATPUMP_DAILY))
            cop = _get_sensor_value(config.get(CONF_SENSOR_HEATPUMP_COP))

            if cop is None:
                cop = DEFAULT_HEATPUMP_COP

            consumers["heatpump"] = {
                "power": power,
                "daily_kwh": daily,
                "cop": cop,
                "configured": True,
            }

        if config.get(CONF_SENSOR_HEATINGROD_POWER):
            power = _get_sensor_value(config.get(CONF_SENSOR_HEATINGROD_POWER))
            daily = _get_sensor_value(config.get(CONF_SENSOR_HEATINGROD_DAILY))

            consumers["heatingrod"] = {
                "power": power,
                "daily_kwh": daily,
                "configured": True,
            }

        if config.get(CONF_SENSOR_WALLBOX_POWER):
            power = _get_sensor_value(config.get(CONF_SENSOR_WALLBOX_POWER))
            daily = _get_sensor_value(config.get(CONF_SENSOR_WALLBOX_DAILY))
            state = _get_sensor_value(config.get(CONF_SENSOR_WALLBOX_STATE))

            consumers["wallbox"] = {
                "power": power,
                "daily_kwh": daily,
                "state": state,
                "configured": True,
            }

        return consumers


class StatisticsView(HomeAssistantView):
    """API for statistics data from JSON files. @zara"""

    url = "/api/sfml_stats/statistics"
    name = "api:sfml_stats:statistics"
    requires_auth = False

    @local_only
    async def get(self, request: Request) -> Response:
        """Return statistics data from SFML SQLite database. @zara"""
        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "peaks": {},
            "production": {},
            "statistics": {},
        }

        reader = _get_solar_reader()

        try:
            daily_forecasts = await reader.async_get_daily_forecasts()

            today_forecast = daily_forecasts.get("today")
            if today_forecast:
                result["production"]["today"] = {
                    "forecast_kwh": today_forecast.prediction_kwh,
                    "forecast_kwh_display": f"{today_forecast.prediction_kwh:.2f}",
                    "yield_kwh": None,
                    "source": today_forecast.source,
                    "locked": today_forecast.locked,
                }
            else:
                result["production"]["today"] = {
                    "forecast_kwh": None,
                    "forecast_kwh_display": None,
                    "yield_kwh": None,
                }

            tomorrow_forecast = daily_forecasts.get("tomorrow")
            if tomorrow_forecast:
                result["production"]["tomorrow"] = {
                    "forecast_kwh": tomorrow_forecast.prediction_kwh,
                    "forecast_kwh_display": f"{tomorrow_forecast.prediction_kwh:.2f}",
                    "date": tomorrow_forecast.forecast_date.isoformat() if tomorrow_forecast.forecast_date else None,
                }
            else:
                result["production"]["tomorrow"] = {
                    "forecast_kwh": None,
                    "forecast_kwh_display": None,
                }

        except Exception as e:
            _LOGGER.error("Error loading daily forecasts from database: %s", e)
            result["production"]["today"] = {"forecast_kwh": None, "forecast_kwh_display": None, "yield_kwh": None}
            result["production"]["tomorrow"] = {"forecast_kwh": None, "forecast_kwh_display": None}

        try:
            sfml_reader = SFMLDataReader(HASS)
            yield_value = sfml_reader.get_live_yield()
            if yield_value is not None:
                result["production"]["today"]["yield_kwh"] = yield_value
        except Exception as e:
            _LOGGER.debug("Could not load yield from SFML: %s", e)

        try:
            async with _get_db() as conn:
                async with conn.execute(
                    "SELECT peak_power_w, peak_power_time, peak_record_w, peak_record_date, peak_record_time, production_time_today FROM production_time_state WHERE id = 1"
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        result["peaks"]["today"] = {
                            "power_w": row["peak_power_w"],
                            "at": row["peak_power_time"],
                        }
                        result["peaks"]["record"] = {
                            "power_w": row["peak_record_w"],
                            "date": row["peak_record_date"],
                            "at": row["peak_record_time"],
                        }
                        result["production_time"] = row["production_time_today"]
                    else:
                        result["peaks"]["today"] = {"power_w": None, "at": None}
        except Exception as e:
            _LOGGER.debug("Could not load today's peak from database: %s", e)
            result["peaks"]["today"] = {"power_w": None, "at": None}

        try:
            async with reader._get_db_connection() as conn:
                async with conn.execute(
                    """SELECT all_time_peak_power_w, all_time_peak_date, all_time_peak_at
                       FROM daily_statistics WHERE id = 1"""
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        result["peaks"]["all_time"] = {
                            "power_w": row["all_time_peak_power_w"],
                            "date": row["all_time_peak_date"],
                            "at": row["all_time_peak_at"],
                        }
                    else:
                        result["peaks"]["all_time"] = {"power_w": None, "date": None, "at": None}
        except Exception as e:
            _LOGGER.debug("Could not load all-time peak from database: %s", e)
            result["peaks"]["all_time"] = {"power_w": None, "date": None, "at": None}

        try:
            today_preds = await reader.async_get_hourly_predictions(target_date=date.today())

            result["best_hour"] = {"hour": None, "prediction_kwh": None}
            if today_preds:
                preds_with_values = [p for p in today_preds if p.prediction_kwh > 0]
                if preds_with_values:
                    best = max(preds_with_values, key=lambda x: x.prediction_kwh)
                    result["best_hour"] = {
                        "hour": best.target_hour,
                        "prediction_kwh": best.prediction_kwh,
                    }
        except Exception as e:
            _LOGGER.error("Error loading best hour from database: %s", e)
            result["best_hour"] = {"hour": None, "prediction_kwh": None}

        try:
            async with reader._get_db_connection() as conn:
                async with conn.execute(
                    """SELECT
                        current_week_period, current_week_yield_kwh, current_week_consumption_kwh, current_week_days,
                        current_month_period, current_month_yield_kwh, current_month_consumption_kwh, current_month_avg_autarky, current_month_days,
                        last_7_days_avg_yield_kwh, last_7_days_avg_accuracy, last_7_days_total_yield_kwh,
                        last_30_days_avg_yield_kwh, last_30_days_avg_accuracy, last_30_days_total_yield_kwh,
                        last_365_days_avg_yield_kwh, last_365_days_total_yield_kwh
                       FROM daily_statistics WHERE id = 1"""
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        result["statistics"]["current_week"] = {
                            "period": row["current_week_period"],
                            "yield_kwh": row["current_week_yield_kwh"],
                            "consumption_kwh": row["current_week_consumption_kwh"],
                            "days": row["current_week_days"],
                        }
                        result["statistics"]["current_month"] = {
                            "period": row["current_month_period"],
                            "yield_kwh": row["current_month_yield_kwh"],
                            "consumption_kwh": row["current_month_consumption_kwh"],
                            "avg_autarky": row["current_month_avg_autarky"],
                            "days": row["current_month_days"],
                        }
                        result["statistics"]["last_7_days"] = {
                            "avg_yield_kwh": row["last_7_days_avg_yield_kwh"],
                            "avg_accuracy": row["last_7_days_avg_accuracy"],
                            "total_yield_kwh": row["last_7_days_total_yield_kwh"],
                        }
                        result["statistics"]["last_30_days"] = {
                            "avg_yield_kwh": row["last_30_days_avg_yield_kwh"],
                            "avg_accuracy": row["last_30_days_avg_accuracy"],
                            "total_yield_kwh": row["last_30_days_total_yield_kwh"],
                        }
                        result["statistics"]["last_365_days"] = {
                            "avg_yield_kwh": row["last_365_days_avg_yield_kwh"],
                            "total_yield_kwh": row["last_365_days_total_yield_kwh"],
                        }
        except Exception as e:
            _LOGGER.debug("Could not load statistics from database: %s", e)

        try:
            async with reader._get_db_connection() as conn:
                async with conn.execute(
                    """SELECT date, predicted_total_kwh, actual_total_kwh,
                              accuracy_percent, peak_power_w, peak_hour,
                              peak_power_time, production_hours
                       FROM daily_summaries
                       WHERE date < date('now')
                       ORDER BY date DESC
                       LIMIT 365"""
                ) as cursor:
                    rows = await cursor.fetchall()
                    history = []
                    for row in rows:
                        predicted = row["predicted_total_kwh"] or 0
                        actual = row["actual_total_kwh"] or 0
                        acc = row["accuracy_percent"]
                        if not acc or acc <= 0:
                            if predicted > 0 and actual > 0:
                                acc = max(0, min(100, 100 - abs((actual - predicted) / predicted) * 100))
                            else:
                                acc = 0
                        history.append({
                            "date": row["date"],
                            "predicted_kwh": predicted,
                            "actual_kwh": actual,
                            "accuracy": round(acc, 1),
                            "peak_power_w": row["peak_power_w"],
                            "peak_hour": row["peak_hour"],
                            "peak_at": row["peak_power_time"],
                            "production_hours": row["production_hours"],
                        })
                    result["history"] = history

                # Hourly production data for heatmap
                async with conn.execute(
                    """SELECT target_date, target_hour, actual_kwh
                       FROM hourly_predictions
                       WHERE actual_kwh IS NOT NULL
                         AND target_date >= date('now', '-365 days')
                       ORDER BY target_date, target_hour"""
                ) as cursor:
                    hourly_rows = await cursor.fetchall()
                    hourly_data: dict[str, dict[int, float]] = {}
                    for row in hourly_rows:
                        date_str = row["target_date"]
                        if date_str not in hourly_data:
                            hourly_data[date_str] = {}
                        hourly_data[date_str][row["target_hour"]] = row["actual_kwh"]
                    result["hourly_production"] = hourly_data
        except Exception as e:
            _LOGGER.debug("Could not load history from database: %s", e)
            result["history"] = []
            result["hourly_production"] = {}

        result["panel_groups"] = await self._get_panel_group_data()

        return web.json_response(result)

    async def _get_panel_group_data(self) -> dict[str, Any]:
        """Extract panel group predictions and actuals for today. @zara"""
        current_hour = datetime.now().hour

        try:
            reader = _get_solar_reader()
            panel_groups_by_name = await reader.async_get_panel_group_data(target_date=date.today())

            if not panel_groups_by_name:
                return {"available": False, "groups": {}}
        except Exception as e:
            _LOGGER.error("Error loading panel group data from database: %s", e)
            return {"available": False, "groups": {}}

        config = _get_config()
        name_mapping = config.get(CONF_PANEL_GROUP_NAMES, {})
        if not isinstance(name_mapping, dict):
            name_mapping = {}

        groups = {}
        for group_name, hourly_data in panel_groups_by_name.items():
            display_name = name_mapping.get(group_name, group_name)

            group_data = {
                "name": display_name,
                "original_name": group_name,
                "prediction_day_kwh": 0.0,
                "prediction_until_now_kwh": 0.0,
                "actual_until_now_kwh": 0.0,
                "hourly": [],
            }

            for panel_data in hourly_data:
                pred_kwh = panel_data.prediction_kwh
                actual_kwh = panel_data.actual_kwh
                hour = panel_data.target_hour if panel_data.target_hour is not None else 0

                if pred_kwh is not None:
                    group_data["prediction_day_kwh"] += pred_kwh

                if hour <= current_hour:
                    if pred_kwh is not None:
                        group_data["prediction_until_now_kwh"] += pred_kwh
                    if actual_kwh is not None:
                        group_data["actual_until_now_kwh"] += actual_kwh

                group_data["hourly"].append({
                    "hour": hour,
                    "prediction_kwh": pred_kwh,
                    "actual_kwh": actual_kwh,
                })

            group_data["actual_source"] = "db_hourly_sum"

            group_data["prediction_total_kwh"] = group_data["prediction_day_kwh"]
            group_data["actual_total_kwh"] = group_data["actual_until_now_kwh"]

            if group_data["prediction_until_now_kwh"] > 0 and group_data["actual_until_now_kwh"] > 0:
                deviation_percent = abs(
                    (group_data["actual_until_now_kwh"] - group_data["prediction_until_now_kwh"])
                    / group_data["prediction_until_now_kwh"]
                ) * 100
                group_data["accuracy_percent"] = max(0, min(100, 100 - deviation_percent))
            else:
                group_data["accuracy_percent"] = None

            groups[display_name] = group_data

        return {"available": True, "groups": groups}



class BillingDataView(HomeAssistantView):
    """API for billing and annual balance data. @zara"""

    url = "/api/sfml_stats/billing"
    name = "api:sfml_stats:billing"
    requires_auth = False

    @local_only
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
    """Export solar analytics as PNG. @zara"""

    url = "/api/sfml_stats/export_solar_analytics"
    name = "api:sfml_stats:export_solar_analytics"
    requires_auth = False

    @local_only
    async def post(self, request: web.Request) -> web.Response:
        """Generate and return solar analytics PNG. @zara"""
        try:
            data = await request.json()
            period = data.get("period", "week")
            stats = data.get("stats", {})
            history = data.get("data", [])

            _LOGGER.info("Generating solar analytics export: period=%s, data_points=%d", period, len(history))

            from ..charts.solar_analytics import SolarAnalyticsChart

            chart = SolarAnalyticsChart(
                period=period,
                stats=stats,
                data=history
            )

            png_bytes = await chart.async_render()

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
    """Export battery analytics as PNG. @zara"""

    url = "/api/sfml_stats/export_battery_analytics"
    name = "api:sfml_stats:export_battery_analytics"
    requires_auth = False

    @local_only
    async def post(self, request: web.Request) -> web.Response:
        """Generate and return battery analytics PNG. @zara"""
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
    """Export house analytics as PNG. @zara"""

    url = "/api/sfml_stats/export_house_analytics"
    name = "api:sfml_stats:export_house_analytics"
    requires_auth = False

    @local_only
    async def post(self, request: web.Request) -> web.Response:
        """Generate and return house analytics PNG. @zara"""
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
    """Export grid analytics as PNG. @zara"""

    url = "/api/sfml_stats/export_grid_analytics"
    name = "api:sfml_stats:export_grid_analytics"
    requires_auth = False

    @local_only
    async def post(self, request: web.Request) -> web.Response:
        """Generate and return grid analytics PNG. @zara"""
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
    """Get weather history data. @zara"""

    url = "/api/sfml_stats/weather_history"
    name = "api:sfml_stats:weather_history"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Get weather history. @zara"""
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


class WeatherComparisonView(HomeAssistantView):
    """Get actual vs forecast weather comparison data. @zara"""

    url = "/api/sfml_stats/weather_comparison"
    name = "api:sfml_stats:weather_comparison"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Get actual vs forecast weather comparison data. @zara"""
        try:
            from ..weather_collector import WeatherDataCollector

            days = int(request.query.get("days", 7))
            days = min(days, 30)

            data_path = Path(HASS.config.path()) / "sfml_stats_weather"
            collector = WeatherDataCollector(HASS, data_path)

            comparison = await collector.get_comparison_data(days=days)

            return web.json_response(comparison)

        except Exception as err:
            _LOGGER.error("Error fetching weather comparison: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)


class ExportWeatherAnalyticsView(HomeAssistantView):
    """Export weather analytics as PNG. @zara"""

    url = "/api/sfml_stats/export_weather_analytics"
    name = "api:sfml_stats:export_weather_analytics"
    requires_auth = False

    @local_only
    async def post(self, request: web.Request) -> web.Response:
        """Generate and return weather analytics PNG. @zara"""
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

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Get power sources history from Home Assistant Recorder. @zara"""
        try:
            hours = int(request.query.get("hours", 24))
            hours = min(hours, 168)

            config = _get_config()
            sfml_reader = SFMLDataReader(HASS)

            sensors = {
                "solar_power": sfml_reader.get_power_entity_id(),
                "solar_to_house": config.get(CONF_SENSOR_SOLAR_TO_HOUSE),
                "solar_to_battery": config.get(CONF_SENSOR_SOLAR_TO_BATTERY),
                "battery_to_house": config.get(CONF_SENSOR_BATTERY_TO_HOUSE),
                "grid_to_house": config.get(CONF_SENSOR_GRID_TO_HOUSE),
                "home_consumption": config.get(CONF_SENSOR_HOME_CONSUMPTION),
                "battery_soc": config.get(CONF_SENSOR_BATTERY_SOC),
            }

            entity_ids = [eid for eid in sensors.values() if eid]

            if not entity_ids:
                return web.json_response({
                    "success": False,
                    "error": "No sensors configured"
                })

            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)

            history_data = await self._get_recorder_history(
                entity_ids, start_time, end_time
            )
            processed_data = self._process_history(history_data, sensors, start_time, end_time)
            data_source = "recorder"

            # Check flow sensors specifically (not just solar_power)
            flow_keys = ['solar_to_house', 'grid_to_house', 'home_consumption']
            flow_data_count = sum(
                1 for d in processed_data
                if any(d.get(k) is not None and d.get(k, 0) > 0 for k in flow_keys)
            )
            has_data = flow_data_count >= 3  # Need at least 3 valid data points

            if not has_data:
                collector_data = await self._get_power_sources_collector_data(hours)
                if collector_data:
                    processed_data = collector_data
                    data_source = "db"
                    _LOGGER.debug("Power history from DB: %d points", len(collector_data))

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

        try:
            from homeassistant.components.recorder import get_instance
            from homeassistant.components.recorder import history as recorder_history

            if hasattr(recorder_history, 'get_significant_states'):
                instance = get_instance(HASS)

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
                    "solar_to_house": hour_data.get("solar_to_house_kwh", 0) * 1000,
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
        """Get data from stats_power_sources DB table. @zara"""
        try:
            async with _get_db() as conn:
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

                async with conn.execute("""
                    SELECT timestamp, solar_power_w, solar_to_house_w,
                           solar_to_battery_w, battery_to_house_w,
                           grid_to_house_w, house_consumption_w
                    FROM stats_power_sources
                    WHERE timestamp >= ?
                    ORDER BY timestamp
                """, (cutoff,)) as cursor:
                    rows = await cursor.fetchall()

                if not rows:
                    return []

                result = [
                    {
                        "timestamp": row["timestamp"],
                        "solar_power": row["solar_power_w"] or 0,
                        "solar_to_house": row["solar_to_house_w"] or 0,
                        "solar_to_battery": row["solar_to_battery_w"] or 0,
                        "battery_to_house": row["battery_to_house_w"] or 0,
                        "grid_to_house": row["grid_to_house_w"] or 0,
                        "home_consumption": row["house_consumption_w"] or 0,
                        "battery_soc": None,
                    }
                    for row in rows
                ]
                _LOGGER.debug("Power sources from DB: %d points", len(result))
                return result

        except Exception as e:
            _LOGGER.error("Error reading power sources from DB: %s", e)
            return []

    def _process_history(
        self,
        history_data: dict[str, list],
        sensors: dict[str, str | None],
        start_time: datetime,
        end_time: datetime
    ) -> list[dict]:
        """Process and align history data into time series. @zara"""
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

        for sensor_key, entity_id in sensors.items():
            if not entity_id or entity_id not in history_data:
                continue

            states = history_data[entity_id]
            if not states:
                continue

            sorted_states = sorted(states, key=lambda s: s.last_updated if hasattr(s, 'last_updated') else s.last_changed)

            state_idx = 0
            for bucket in buckets:
                bucket_time = datetime.fromisoformat(bucket["timestamp"])
                if bucket_time.tzinfo is None:
                    bucket_time = bucket_time.replace(tzinfo=timezone.utc)

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

    @local_only
    async def post(self, request: web.Request) -> web.Response:
        """Generate and return power sources PNG. @zara"""
        try:
            data = await request.json()
            period = data.get("period", "today")
            stats = data.get("stats", {})
            history = data.get("data", [])

            if hasattr(stats, '__dict__'):
                stats = dict(stats)

            _LOGGER.info("Generating power sources export: period=%s, data_points=%d, stats_keys=%s",
                         period, len(history), list(stats.keys()) if stats else [])

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
            import traceback
            _LOGGER.error("Error generating power sources export: %s\n%s", err, traceback.format_exc())
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)


class EnergySourcesDailyStatsView(HomeAssistantView):
    """API for daily energy sources statistics. @zara"""

    url = "/api/sfml_stats/energy_sources_daily_stats"
    name = "api:sfml_stats:energy_sources_daily_stats"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Get daily energy sources statistics. @zara"""
        try:
            days = int(request.query.get("days", 7))
            days = min(days, 365)

            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized"
                })

            collector = None
            entries = HASS.data.get(DOMAIN, {})
            for entry_id, entry_data in entries.items():
                if isinstance(entry_data, dict) and "power_sources_collector" in entry_data:
                    collector = entry_data["power_sources_collector"]
                    break

            if collector is None:
                data_path = Path(HASS.config.path()) / "sfml_stats" / "data" / "energy_sources_daily_stats.json"
                if data_path.exists():
                    import aiofiles
                    async with aiofiles.open(data_path, 'r') as f:
                        content = await f.read()
                        daily_stats = json.loads(content)
                else:
                    daily_stats = {"days": {}}
            else:
                daily_stats = await collector.get_daily_stats(days)

            history_path = Path(HASS.config.path()) / "sfml_stats" / "data" / "daily_energy_history.json"
            if history_path.exists():
                import aiofiles
                async with aiofiles.open(history_path, 'r') as f:
                    content = await f.read()
                    history_data = json.loads(content)
                    history_days = history_data.get("days", {})

                    for date_str, day_data in history_days.items():
                        if date_str not in daily_stats.get("days", {}):
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
                                "peak_consumption_w": day_data.get("peak_battery_power_w", 0),
                            }
                        else:
                            existing = daily_stats["days"][date_str]
                            if existing.get("peak_battery_power_w") is None or existing.get("peak_battery_power_w") == 0:
                                existing["peak_battery_power_w"] = day_data.get("peak_battery_power_w", 0)
                            if existing.get("home_consumption_kwh") is None or existing.get("home_consumption_kwh") == 0:
                                existing["home_consumption_kwh"] = day_data.get("home_consumption_kwh", 0)
                            if existing.get("autarky_percent") is None or existing.get("autarky_percent") == 0:
                                existing["autarky_percent"] = day_data.get("autarky_percent", 0)
                            if existing.get("self_consumption_percent") is None or existing.get("self_consumption_percent") == 0:
                                existing["self_consumption_percent"] = day_data.get("self_consumption_percent", 0)
                            if existing.get("peak_consumption_w") is None or existing.get("peak_consumption_w") == 0:
                                existing["peak_consumption_w"] = day_data.get("peak_battery_power_w", 0)

            config = _get_config()
            sfml_reader = SFMLDataReader(HASS)

            solar_power_val = sfml_reader.get_live_power()
            solar_to_house_val = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_TO_HOUSE))
            solar_to_battery_val = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_TO_BATTERY))
            if solar_power_val is not None and solar_power_val < 0:
                solar_power_val = 0.0
            if solar_to_house_val is not None and solar_to_house_val < 0:
                solar_to_house_val = 0.0
            if solar_to_battery_val is not None and solar_to_battery_val < 0:
                solar_to_battery_val = 0.0
            if solar_power_val is not None and solar_power_val <= 0:
                solar_to_house_val = 0.0
                solar_to_battery_val = 0.0
            elif solar_power_val is not None:
                if solar_to_battery_val is not None:
                    solar_to_battery_val = min(solar_to_battery_val, solar_power_val)
                if solar_to_house_val is not None:
                    solar_to_house_val = min(solar_to_house_val, solar_power_val)
            current_values = {
                "solar_yield_daily": sfml_reader.get_live_yield(),
                "solar_to_house": solar_to_house_val,
                "solar_to_battery": solar_to_battery_val,
                "battery_to_house": _get_sensor_value(config.get(CONF_SENSOR_BATTERY_TO_HOUSE)),
                "grid_to_house": _get_sensor_value(config.get(CONF_SENSOR_GRID_TO_HOUSE)),
                "home_consumption": _get_sensor_value(config.get(CONF_SENSOR_HOME_CONSUMPTION)),
            }

            # Hourly data from stats_hourly_billing + prices from GPM
            hourly_profile = {}
            hourly_grid = {}
            daily_prices = {}
            try:
                cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                async with _get_db() as conn:
                    # Hourly consumption + grid import profile
                    async with conn.execute(
                        """SELECT date, hour,
                                  COALESCE(grid_to_house_kwh, 0) + COALESCE(solar_to_house_kwh, 0)
                                  + COALESCE(battery_to_house_kwh, 0) AS consumption_kwh,
                                  COALESCE(grid_to_house_kwh, 0) + COALESCE(grid_to_battery_kwh, 0) AS grid_import_kwh
                           FROM stats_hourly_billing
                           WHERE date >= ?
                           ORDER BY date, hour""",
                        (cutoff_date,)
                    ) as cursor:
                        for row in await cursor.fetchall():
                            d = row["date"]
                            h = row["hour"]
                            hourly_profile.setdefault(d, {})[h] = round(row["consumption_kwh"], 4)
                            hourly_grid.setdefault(d, {})[h] = round(row["grid_import_kwh"], 4)

                    # Daily average prices from GPM
                    async with conn.execute(
                        """SELECT date, average_total FROM GPM_daily_averages
                           WHERE date >= ? ORDER BY date""",
                        (cutoff_date,)
                    ) as cursor:
                        for row in await cursor.fetchall():
                            daily_prices[row["date"]] = round(row["average_total"], 2)
            except Exception as e:
                _LOGGER.debug("Could not load hourly/price data: %s", e)

            return web.json_response({
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "days_requested": days,
                "daily_stats": daily_stats.get("days", {}),
                "current_values": current_values,
                "hourly_consumption": hourly_profile,
                "hourly_grid": hourly_grid,
                "daily_prices": daily_prices,
            })

        except Exception as err:
            _LOGGER.error("Error fetching energy sources daily stats: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)


class ClothingRecommendationView(HomeAssistantView):
    """API for clothing recommendation based on weather data. @zara"""

    url = "/api/sfml_stats/clothing_recommendation"
    name = "api:sfml_stats:clothing_recommendation"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Get clothing recommendation based on current weather. @zara"""
        try:
            from ..clothing_recommendation import get_recommendation

            weather_data = await self._get_weather_data()
            if not weather_data:
                return web.json_response({
                    "success": False,
                    "error": "No weather data available"
                })

            forecast_hours = await self._get_forecast_hours()

            recommendation = get_recommendation(weather_data, forecast_hours)

            return web.json_response({
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "recommendation": {
                    "unterbekleidung": {
                        "name": recommendation.unterbekleidung,
                        "icon": recommendation.unterbekleidung_icon,
                    },
                    "oberbekleidung": {
                        "name": recommendation.oberbekleidung,
                        "icon": recommendation.oberbekleidung_icon,
                    },
                    "jacke": {
                        "name": recommendation.jacke,
                        "icon": recommendation.jacke_icon,
                    },
                    "kopfbedeckung": {
                        "name": recommendation.kopfbedeckung,
                        "icon": recommendation.kopfbedeckung_icon,
                    },
                    "zusaetze": [
                        {"name": name, "icon": icon}
                        for name, icon in zip(recommendation.zusaetze, recommendation.zusaetze_icons)
                    ],
                    "text_de": recommendation.text_de,
                    "text_en": recommendation.text_en,
                },
                "weather": recommendation.weather_summary,
            })

        except Exception as err:
            _LOGGER.error("Error generating clothing recommendation: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)

    async def _get_weather_data(self) -> dict | None:
        """Get current weather data from SFML database. @zara"""
        today_str = date.today().isoformat()
        current_hour = str(datetime.now().hour)

        weather_actual_db = await _get_weather_from_db(days=1)
        if weather_actual_db and today_str in weather_actual_db and current_hour in weather_actual_db[today_str]:
            hour_data = weather_actual_db[today_str][current_hour]
            return {
                "temperature": hour_data.get("temperature_c", 15),
                "humidity": hour_data.get("humidity_percent", 50),
                "wind_speed": hour_data.get("wind_speed_ms", 0),
                "precipitation": hour_data.get("precipitation_mm", 0),
                "cloud_cover": hour_data.get("cloud_cover_percent", 50),
                "pressure": 1013,
                "radiation": hour_data.get("solar_radiation_wm2", 0),
                "uv_index": 0,
            }

        forecast_data = await _get_weather_forecast_from_db(days=1)
        if forecast_data and today_str in forecast_data:
            hour_data = forecast_data[today_str].get(current_hour, {})
            if hour_data:
                return {
                    "temperature": hour_data.get("temperature", 15),
                    "humidity": hour_data.get("humidity", 50),
                    "wind_speed": hour_data.get("wind_speed", 0),
                    "precipitation": hour_data.get("precipitation", 0),
                    "cloud_cover": hour_data.get("cloud_cover", 50),
                    "pressure": 1013,
                    "radiation": hour_data.get("solar_radiation_wm2", 0),
                    "uv_index": 0,
                }

        config = _get_config()
        weather_ha = _get_weather_data(config.get(CONF_WEATHER_ENTITY))
        if weather_ha:
            return {
                "temperature": weather_ha.get("temperature", 15),
                "humidity": weather_ha.get("humidity", 50),
                "wind_speed": weather_ha.get("wind_speed", 0),
                "precipitation": 0,
                "cloud_cover": weather_ha.get("cloud_coverage", 50),
                "pressure": weather_ha.get("pressure", 1013),
                "radiation": 0,
                "uv_index": weather_ha.get("uv_index", 0),
            }

        return None

    async def _get_forecast_hours(self) -> list[dict] | None:
        """Get hourly forecast for rain probability from database. @zara"""
        forecast_data = await _get_weather_forecast_from_db(days=1)
        if not forecast_data:
            return None

        today_str = date.today().isoformat()
        current_hour = datetime.now().hour
        forecast_hours = []

        if today_str in forecast_data:
            for hour in range(current_hour, 24):
                hour_data = forecast_data[today_str].get(str(hour), {})
                if hour_data:
                    precip = hour_data.get("precipitation", 0) or 0
                    precip_prob = 100 if precip > 0 else 0
                    forecast_hours.append({
                        "hour": hour,
                        "precipitation_probability": precip_prob,
                        "precipitation": precip,
                    })

        return forecast_hours if forecast_hours else None


class MonthlyTariffsView(HomeAssistantView):
    """API for monthly tariffs management (EEG/Energy Sharing support). @zara"""

    url = "/api/sfml_stats/monthly_tariffs"
    name = "api:sfml_stats:monthly_tariffs"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Get monthly tariffs data. @zara"""
        try:
            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized",
                })

            tariff_manager = None
            entries = HASS.data.get(DOMAIN, {})
            for entry_id, entry_data in entries.items():
                if isinstance(entry_data, dict) and "monthly_tariff_manager" in entry_data:
                    tariff_manager = entry_data["monthly_tariff_manager"]
                    break

            if tariff_manager is None:
                return web.json_response({
                    "success": False,
                    "error": "MonthlyTariffManager not initialized",
                })

            year = int(request.query.get("year", date.today().year))
            include_empty = request.query.get("include_empty", "false").lower() == "true"

            summary = await tariff_manager.get_year_summary(year)

            return web.json_response({
                "success": True,
                "timestamp": datetime.now().isoformat(),
                **summary,
            })

        except Exception as err:
            _LOGGER.error("Error fetching monthly tariffs: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)


class MonthlyTariffDetailView(HomeAssistantView):
    """API for single month tariff details. @zara"""

    url = "/api/sfml_stats/monthly_tariffs/{year}/{month}"
    name = "api:sfml_stats:monthly_tariff_detail"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request, year: str, month: str) -> web.Response:
        """Get detailed data for a specific month. @zara"""
        try:
            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized",
                })

            tariff_manager = self._get_tariff_manager()
            if tariff_manager is None:
                return web.json_response({
                    "success": False,
                    "error": "MonthlyTariffManager not initialized",
                })

            month_data = await tariff_manager.get_monthly_data(int(year), int(month))

            return web.json_response({
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "data": month_data,
            })

        except Exception as err:
            _LOGGER.error("Error fetching month detail: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)

    @local_only
    async def post(self, request: web.Request, year: str, month: str) -> web.Response:
        """Update overrides for a specific month. @zara"""
        try:
            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized",
                })

            tariff_manager = self._get_tariff_manager()
            if tariff_manager is None:
                return web.json_response({
                    "success": False,
                    "error": "MonthlyTariffManager not initialized",
                })

            data = await request.json()
            overrides = data.get("overrides", {})

            success = await tariff_manager.set_monthly_override(
                int(year), int(month), overrides
            )

            if success:
                month_data = await tariff_manager.get_monthly_data(int(year), int(month))
                return web.json_response({
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "data": month_data,
                })
            else:
                return web.json_response({
                    "success": False,
                    "error": "Failed to save overrides",
                }, status=500)

        except Exception as err:
            _LOGGER.error("Error updating month overrides: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)

    def _get_tariff_manager(self):
        """Get tariff manager from HASS data. @zara"""
        if HASS is None:
            return None
        entries = HASS.data.get(DOMAIN, {})
        for entry_id, entry_data in entries.items():
            if isinstance(entry_data, dict) and "monthly_tariff_manager" in entry_data:
                return entry_data["monthly_tariff_manager"]
        return None


class MonthlyTariffFinalizeView(HomeAssistantView):
    """API to finalize a month (mark as billed). @zara"""

    url = "/api/sfml_stats/monthly_tariffs/{year}/{month}/finalize"
    name = "api:sfml_stats:monthly_tariff_finalize"
    requires_auth = False

    @local_only
    async def post(self, request: web.Request, year: str, month: str) -> web.Response:
        """Finalize a month and optionally recalculate history. @zara"""
        try:
            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized",
                })

            tariff_manager = self._get_tariff_manager()
            if tariff_manager is None:
                return web.json_response({
                    "success": False,
                    "error": "MonthlyTariffManager not initialized",
                })

            data = await request.json() if request.body_exists else {}
            recalculate = data.get("recalculate_history", True)

            result = await tariff_manager.finalize_month(
                int(year), int(month), recalculate_history=recalculate
            )

            return web.json_response({
                "success": True,
                "timestamp": datetime.now().isoformat(),
                **result,
            })

        except Exception as err:
            _LOGGER.error("Error finalizing month: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)

    async def delete(self, request: web.Request, year: str, month: str) -> web.Response:
        """Remove finalization from a month. @zara"""
        try:
            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized",
                })

            tariff_manager = self._get_tariff_manager()
            if tariff_manager is None:
                return web.json_response({
                    "success": False,
                    "error": "MonthlyTariffManager not initialized",
                })

            success = await tariff_manager.unfinalize_month(int(year), int(month))

            return web.json_response({
                "success": success,
                "timestamp": datetime.now().isoformat(),
            })

        except Exception as err:
            _LOGGER.error("Error unfinalizing month: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)

    def _get_tariff_manager(self):
        """Get tariff manager from HASS data. @zara"""
        if HASS is None:
            return None
        entries = HASS.data.get(DOMAIN, {})
        for entry_id, entry_data in entries.items():
            if isinstance(entry_data, dict) and "monthly_tariff_manager" in entry_data:
                return entry_data["monthly_tariff_manager"]
        return None


class MonthlyTariffsExportView(HomeAssistantView):
    """API to export monthly tariffs as CSV. @zara"""

    url = "/api/sfml_stats/monthly_tariffs/export"
    name = "api:sfml_stats:monthly_tariffs_export"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Export monthly tariffs as CSV. @zara"""
        try:
            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized",
                })

            tariff_manager = self._get_tariff_manager()
            if tariff_manager is None:
                return web.json_response({
                    "success": False,
                    "error": "MonthlyTariffManager not initialized",
                })

            today = date.today()
            start = request.query.get("start", f"{today.year}-01")
            end = request.query.get("end", f"{today.year}-{today.month:02d}")

            start_year, start_month = map(int, start.split("-"))
            end_year, end_month = map(int, end.split("-"))

            csv_content = await tariff_manager.export_csv(
                start_year, start_month, end_year, end_month
            )

            filename = f"monthly_tariffs_{start}_{end}.csv"

            return web.Response(
                text=csv_content,
                content_type="text/csv",
                charset="utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"'
                }
            )

        except Exception as err:
            _LOGGER.error("Error exporting monthly tariffs: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)

    def _get_tariff_manager(self):
        """Get tariff manager from HASS data. @zara"""
        if HASS is None:
            return None
        entries = HASS.data.get(DOMAIN, {})
        for entry_id, entry_data in entries.items():
            if isinstance(entry_data, dict) and "monthly_tariff_manager" in entry_data:
                return entry_data["monthly_tariff_manager"]
        return None


class MonthlyTariffsDefaultsView(HomeAssistantView):
    """API to manage default tariff settings. @zara"""

    url = "/api/sfml_stats/monthly_tariffs/defaults"
    name = "api:sfml_stats:monthly_tariffs_defaults"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Get current default tariff settings. @zara"""
        try:
            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized",
                })

            tariff_manager = self._get_tariff_manager()
            if tariff_manager is None:
                return web.json_response({
                    "success": False,
                    "error": "MonthlyTariffManager not initialized",
                })

            defaults = tariff_manager._get_defaults()

            return web.json_response({
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "defaults": defaults,
            })

        except Exception as err:
            _LOGGER.error("Error fetching defaults: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)

    @local_only
    async def post(self, request: web.Request) -> web.Response:
        """Update default tariff settings. @zara"""
        try:
            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized",
                })

            tariff_manager = self._get_tariff_manager()
            if tariff_manager is None:
                return web.json_response({
                    "success": False,
                    "error": "MonthlyTariffManager not initialized",
                })

            data = await request.json()
            success = await tariff_manager.update_defaults(data)

            if success:
                defaults = tariff_manager._get_defaults()
                return web.json_response({
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "defaults": defaults,
                })
            else:
                return web.json_response({
                    "success": False,
                    "error": "Failed to save defaults",
                }, status=500)

        except Exception as err:
            _LOGGER.error("Error updating defaults: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)

    def _get_tariff_manager(self):
        """Get tariff manager from HASS data. @zara"""
        if HASS is None:
            return None
        entries = HASS.data.get(DOMAIN, {})
        for entry_id, entry_data in entries.items():
            if isinstance(entry_data, dict) and "monthly_tariff_manager" in entry_data:
                return entry_data["monthly_tariff_manager"]
        return None


class ExportWeeklyReportView(HomeAssistantView):
    """Generate and export weekly report as PNG. @zara"""

    url = "/api/sfml_stats/export_weekly_report"
    name = "api:sfml_stats:export_weekly_report"
    requires_auth = False

    @local_only
    async def post(self, request: web.Request) -> web.Response:
        """Generate and return weekly report PNG. @zara"""
        try:
            from datetime import date
            from pathlib import Path
            from ..charts.weekly_report import WeeklyReportChart
            from ..storage import DataValidator

            try:
                data = await request.json()
            except Exception:
                data = {}

            year = data.get("year")
            week = data.get("week")

            if year is None or week is None:
                today = date.today()
                iso = today.isocalendar()
                year = year or iso[0]
                week = week or iso[1]

            _LOGGER.info("Generating weekly report: KW %d/%d (Modern Redesign)", week, year)

            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized"
                }, status=500)

            validator = None
            entries = HASS.data.get(DOMAIN, {})
            for entry_id, entry_data in entries.items():
                if isinstance(entry_data, dict) and "validator" in entry_data:
                    validator = entry_data["validator"]
                    break

            if validator is None:
                return web.json_response({
                    "success": False,
                    "error": "DataValidator not initialized"
                }, status=500)

            chart = WeeklyReportChart(validator)
            fig = await chart.generate(year=year, week=week)

            import io
            from concurrent.futures import ThreadPoolExecutor

            def _render_to_bytes():
                buf = io.BytesIO()
                fig.savefig(
                    buf,
                    format="png",
                    dpi=150,
                    bbox_inches="tight",
                    facecolor=chart.styles.background,
                    edgecolor="none"
                )
                buf.seek(0)
                import matplotlib.pyplot as plt
                plt.close(fig)
                return buf.getvalue()

            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                png_bytes = await loop.run_in_executor(pool, _render_to_bytes)

            save_path = await chart.save(year=year, week=week)
            _LOGGER.info("Weekly report saved to: %s", save_path)

            return web.Response(
                body=png_bytes,
                content_type="image/png",
                headers={
                    "Content-Disposition": f'attachment; filename="weekly_report_KW{week:02d}_{year}.png"',
                    "X-Save-Path": str(save_path),
                }
            )

        except Exception as err:
            _LOGGER.error("Error generating weekly report: %s", err, exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Generate weekly report for current week. @zara"""
        from datetime import date

        today = date.today()
        iso = today.isocalendar()

        class MockRequest:
            async def json(self):
                return {"year": iso[0], "week": iso[1]}

        mock = MockRequest()
        return await self.post(mock)


class BackgroundImageView(HomeAssistantView):
    """Serve the dashboard background image. @zara"""

    url = "/api/sfml_stats/background"
    name = "api:sfml_stats:background"
    requires_auth = False

    async def get(self, request: web.Request) -> web.Response:
        """Return the background image. @zara"""
        from pathlib import Path

        bg_path = None
        paths_tried = []

        if HASS is not None:
            primary = Path(HASS.config.path()) / "custom_components" / "sfml_stats" / "frontend" / "dist" / "assets" / "background.webp"
            paths_tried.append(str(primary))
            if primary.exists():
                bg_path = primary

            if bg_path is None:
                alt_path = Path(HASS.config.path()) / "sfml_stats" / "background.webp"
                paths_tried.append(str(alt_path))
                if alt_path.exists():
                    bg_path = alt_path

        if bg_path is None:
            file_path = Path(__file__).parent.parent / "frontend" / "dist" / "assets" / "background.webp"
            paths_tried.append(str(file_path))
            if file_path.exists():
                bg_path = file_path

        if bg_path is None:
            _LOGGER.warning("Background image not found. Tried: %s", paths_tried)
            return web.Response(status=404, text=f"Background image not found. Tried: {paths_tried}")

        try:
            with open(bg_path, "rb") as f:
                image_data = f.read()

            return web.Response(
                body=image_data,
                content_type="image/webp",
                headers={
                    "Cache-Control": "public, max-age=86400",
                }
            )
        except Exception as err:
            _LOGGER.error("Error serving background image: %s", err)
            return web.Response(status=500, text=str(err))


class ForecastComparisonView(HomeAssistantView):
    """Get forecast comparison data. @zara"""

    url = "/api/sfml_stats/forecast_comparison"
    name = "api:sfml_stats:forecast_comparison"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Return forecast comparison data as JSON. @zara"""
        try:
            from ..readers.forecast_comparison_reader import ForecastComparisonReader

            days = int(request.query.get("days", "7"))
            days = min(max(days, 1), 30)

            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized"
                }, status=500)

            db_path = Path(HASS.config.path()) / SOLAR_FORECAST_DB
            config = _get_config()
            ext1_name = config.get(CONF_FORECAST_ENTITY_1_NAME, DEFAULT_FORECAST_ENTITY_1_NAME)
            ext2_name = config.get(CONF_FORECAST_ENTITY_2_NAME, DEFAULT_FORECAST_ENTITY_2_NAME)

            reader = ForecastComparisonReader(db_path, ext1_name, ext2_name)

            if not reader.is_available:
                return web.json_response({
                    "success": False,
                    "error": "No forecast comparison data available yet",
                    "hint": "Data is collected daily at 23:50"
                }, status=404)

            chart_data = await reader.async_get_chart_data(days=days)

            return web.json_response({
                "success": True,
                "data": chart_data,
            })

        except Exception as err:
            _LOGGER.error("Error getting forecast comparison data: %s", err)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)


class ForecastComparisonChartView(HomeAssistantView):
    """Generate and return forecast comparison chart as PNG. @zara"""

    url = "/api/sfml_stats/forecast_comparison_chart"
    name = "api:sfml_stats:forecast_comparison_chart"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Generate and return forecast comparison chart as PNG. @zara"""
        try:
            from ..charts.forecast_comparison import ForecastComparisonChart
            from ..storage import DataValidator

            days = int(request.query.get("days", "7"))
            days = min(max(days, 1), 30)

            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized"
                }, status=500)

            validator = None
            entries = HASS.data.get(DOMAIN, {})
            for entry_id, entry_data in entries.items():
                if isinstance(entry_data, dict) and "validator" in entry_data:
                    validator = entry_data["validator"]
                    break

            if validator is None:
                return web.json_response({
                    "success": False,
                    "error": "DataValidator not initialized"
                }, status=500)

            _LOGGER.info("Generating forecast comparison chart (%d days)", days)

            chart = ForecastComparisonChart(validator)
            fig = await chart.generate(days=days)

            import io
            from concurrent.futures import ThreadPoolExecutor

            def _render_to_bytes():
                buf = io.BytesIO()
                fig.savefig(
                    buf,
                    format="png",
                    dpi=150,
                    bbox_inches="tight",
                    facecolor=chart.styles.background,
                    edgecolor="none",
                )
                import matplotlib.pyplot as plt
                plt.close(fig)
                buf.seek(0)
                return buf.getvalue()

            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                png_bytes = await loop.run_in_executor(executor, _render_to_bytes)

            return web.Response(
                body=png_bytes,
                content_type="image/png",
                headers={
                    "Content-Disposition": "inline; filename=forecast_comparison.png",
                    "Cache-Control": "no-cache",
                }
            )

        except Exception as err:
            _LOGGER.error("Error generating forecast comparison chart: %s", err)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)

    @local_only
    async def post(self, request: web.Request) -> web.Response:
        """Handle POST request for forecast comparison chart. @zara"""
        try:
            data = await request.json()
            days = data.get("days", 7)
        except Exception:
            days = 7

        class MockRequest:
            query = {"days": str(days)}

        return await self.get(MockRequest())


class ShadowAnalyticsView(HomeAssistantView):
    """Get shadow analytics data from DB. @zara"""

    url = "/api/sfml_stats/shadow_analytics"
    name = "api:sfml_stats:shadow_analytics"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Return shadow analytics data as JSON. @zara"""
        try:
            if HASS is None:
                return web.json_response({"success": False, "error": "HASS not initialized"}, status=500)

            days = int(request.query.get("days", "30"))
            days = min(max(days, 7), 365)
            cutoff = (date.today() - timedelta(days=days)).isoformat()

            async with _get_db() as conn:
                # --- Heatmap: shadow_percent per date/hour ---
                heatmap: dict[str, dict[int, dict]] = {}
                async with conn.execute(
                    """SELECT p.target_date, p.target_hour,
                              h.shadow_percent, h.shadow_type, h.root_cause,
                              h.efficiency_ratio, h.loss_kwh, h.confidence
                       FROM hourly_shadow_detection h
                       JOIN hourly_predictions p ON h.prediction_id = p.prediction_id
                       WHERE p.target_date >= ?
                         AND p.target_hour >= 7 AND p.target_hour <= 18
                       ORDER BY p.target_date, p.target_hour""",
                    (cutoff,),
                ) as cursor:
                    for row in await cursor.fetchall():
                        d = row["target_date"]
                        hr = row["target_hour"]
                        if d not in heatmap:
                            heatmap[d] = {}
                        heatmap[d][hr] = {
                            "pct": round(row["shadow_percent"] or 0, 1),
                            "type": row["shadow_type"],
                            "cause": row["root_cause"],
                            "eff": round(row["efficiency_ratio"] or 0, 2),
                            "loss": round(row["loss_kwh"] or 0, 3),
                        }

                # --- Causes distribution ---
                causes: dict[str, int] = {}
                async with conn.execute(
                    """SELECT h.root_cause, COUNT(*) as cnt
                       FROM hourly_shadow_detection h
                       JOIN hourly_predictions p ON h.prediction_id = p.prediction_id
                       WHERE p.target_date >= ?
                         AND p.target_hour >= 7 AND p.target_hour <= 18
                         AND h.shadow_type <> 'none'
                       GROUP BY h.root_cause
                       ORDER BY cnt DESC""",
                    (cutoff,),
                ) as cursor:
                    for row in await cursor.fetchall():
                        cause = row["root_cause"] or "unknown"
                        if cause != "night":
                            causes[cause] = row["cnt"]

                # --- Daily loss from daily_summary_shadow_analysis ---
                daily_loss: list[dict] = []
                async with conn.execute(
                    """SELECT date, shadow_hours_count, cumulative_loss_kwh
                       FROM daily_summary_shadow_analysis
                       WHERE date >= ?
                       ORDER BY date""",
                    (cutoff,),
                ) as cursor:
                    for row in await cursor.fetchall():
                        daily_loss.append({
                            "date": row["date"],
                            "hours": row["shadow_hours_count"],
                            "loss_kwh": round(row["cumulative_loss_kwh"] or 0, 3),
                        })

                # --- Aggregate stats ---
                total_loss = sum(d["loss_kwh"] for d in daily_loss)
                total_shadow_hours = sum(d["hours"] for d in daily_loss)
                days_with_shadow = sum(1 for d in daily_loss if d["hours"] > 0)

                avg_efficiency = None
                async with conn.execute(
                    """SELECT AVG(h.efficiency_ratio) as avg_eff
                       FROM hourly_shadow_detection h
                       JOIN hourly_predictions p ON h.prediction_id = p.prediction_id
                       WHERE p.target_date >= ?
                         AND p.target_hour >= 7 AND p.target_hour <= 18
                         AND h.shadow_type <> 'none'""",
                    (cutoff,),
                ) as cursor:
                    row = await cursor.fetchone()
                    if row and row["avg_eff"] is not None:
                        avg_efficiency = round(row["avg_eff"], 3)

                dominant_cause = max(causes, key=causes.get) if causes else None

                # --- Hourly pattern (avg shadow per hour of day) ---
                hourly_pattern: dict[int, dict] = {}
                async with conn.execute(
                    """SELECT p.target_hour,
                              AVG(h.shadow_percent) as avg_pct,
                              COUNT(*) as total,
                              SUM(CASE WHEN h.shadow_type <> 'none' THEN 1 ELSE 0 END) as shadow_cnt
                       FROM hourly_shadow_detection h
                       JOIN hourly_predictions p ON h.prediction_id = p.prediction_id
                       WHERE p.target_date >= ?
                         AND p.target_hour >= 7 AND p.target_hour <= 18
                       GROUP BY p.target_hour
                       ORDER BY p.target_hour""",
                    (cutoff,),
                ) as cursor:
                    for row in await cursor.fetchall():
                        hr = row["target_hour"]
                        hourly_pattern[hr] = {
                            "avg_pct": round(row["avg_pct"] or 0, 1),
                            "occurrence_rate": round(row["shadow_cnt"] / row["total"] * 100, 1) if row["total"] > 0 else 0,
                        }

                # --- Learning state ---
                learning = {}
                async with conn.execute("SELECT * FROM shadow_pattern_config WHERE id = 1") as cursor:
                    row = await cursor.fetchone()
                    if row:
                        learning = {
                            "days_learned": row["total_days_learned"],
                            "hours_learned": row["total_hours_learned"],
                            "last_date": row["last_learning_date"],
                            "patterns_detected": row["patterns_detected"],
                        }

            return web.json_response({
                "success": True,
                "data": {
                    "stats": {
                        "total_loss_kwh": round(total_loss, 2),
                        "shadow_hours": total_shadow_hours,
                        "days_with_shadow": days_with_shadow,
                        "days_analyzed": len(daily_loss),
                        "avg_efficiency": avg_efficiency,
                        "dominant_cause": dominant_cause,
                    },
                    "heatmap": heatmap,
                    "causes": causes,
                    "daily_loss": daily_loss,
                    "hourly_pattern": hourly_pattern,
                    "learning": learning,
                },
            })

        except Exception as err:
            _LOGGER.error("Error getting shadow analytics: %s", err)
            return web.json_response({"success": False, "error": str(err)}, status=500)


class AIStatusView(HomeAssistantView):
    """Get basic AI/ML model status. @zara"""

    url = "/api/sfml_stats/ai_status"
    name = "api:sfml_stats:ai_status"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Return AI model status as JSON. @zara"""
        try:
            if HASS is None:
                return web.json_response({"success": False, "error": "HASS not initialized"}, status=500)

            async with _get_db() as conn:
                # Active model info
                model_info = {}
                async with conn.execute(
                    "SELECT * FROM ai_learned_weights_meta WHERE id = 1"
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        model_info = {
                            "active_model": row["active_model"],
                            "training_samples": row["training_samples"],
                            "accuracy": round((row["accuracy"] or 0) * 100, 1),
                            "rmse": round(row["rmse"] or 0, 4),
                            "last_trained": row["last_trained"],
                        }

                # Grid search history
                grid_search = {}
                async with conn.execute(
                    "SELECT COUNT(*) as runs, MAX(accuracy) as best_acc FROM ai_grid_search_results"
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        grid_search = {
                            "total_runs": row["runs"],
                            "best_accuracy": round((row["best_acc"] or 0) * 100, 1),
                        }

                # Shadow learning state
                shadow_learning = {}
                async with conn.execute(
                    "SELECT * FROM shadow_pattern_config WHERE id = 1"
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        shadow_learning = {
                            "days_learned": row["total_days_learned"],
                            "patterns_detected": row["patterns_detected"],
                            "last_date": row["last_learning_date"],
                        }

                # Drift events count
                drift_count = 0
                async with conn.execute(
                    "SELECT COUNT(*) as cnt FROM drift_events WHERE event_date >= date('now', '-7 days')"
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        drift_count = row["cnt"]

                # Training data size
                hourly_count = 0
                async with conn.execute(
                    "SELECT COUNT(*) as cnt FROM hourly_predictions WHERE actual_kwh IS NOT NULL"
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        hourly_count = row["cnt"]

            return web.json_response({
                "success": True,
                "data": {
                    "model": model_info,
                    "grid_search": grid_search,
                    "shadow_learning": shadow_learning,
                    "drift_events_7d": drift_count,
                    "training_data_points": hourly_count,
                },
            })

        except Exception as err:
            _LOGGER.error("Error getting AI status: %s", err)
            return web.json_response({"success": False, "error": str(err)}, status=500)


class DashboardSettingsView(HomeAssistantView):
    """API endpoint for dashboard settings. @zara"""

    url = "/api/sfml_stats/dashboard_settings"
    name = "api:sfml_stats:dashboard_settings"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Return current dashboard settings. @zara"""
        config = _get_config()

        return web.json_response({
            "success": True,
            "data": {
                "dashboard_style": config.get(CONF_DASHBOARD_STYLE, DEFAULT_DASHBOARD_STYLE),
                "theme": config.get(CONF_THEME, DEFAULT_THEME),
            }
        })

    @local_only
    async def post(self, request: web.Request) -> web.Response:
        """Update dashboard settings for current session. @zara"""
        try:
            data = await request.json()
            dashboard_style = data.get("dashboard_style")
            theme = data.get("theme")

            if HASS is not None and DOMAIN in HASS.data:
                entries = HASS.data.get(DOMAIN, {})
                for entry_id, entry_data in entries.items():
                    if isinstance(entry_data, dict):
                        if "session_settings" not in entry_data:
                            entry_data["session_settings"] = {}
                        if dashboard_style:
                            entry_data["session_settings"]["dashboard_style"] = dashboard_style
                        if theme:
                            entry_data["session_settings"]["theme"] = theme
                        break

            return web.json_response({
                "success": True,
                "message": "Settings updated for this session"
            })

        except Exception as err:
            _LOGGER.error("Error updating dashboard settings: %s", err)
            return web.json_response({
                "success": False,
                "error": str(err)
            }, status=500)
