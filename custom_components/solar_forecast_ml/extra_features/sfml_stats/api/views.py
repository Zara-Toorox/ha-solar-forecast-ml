# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""REST API views for SFML Stats Dashboard."""
from __future__ import annotations

import asyncio
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
    """Extract real client IP from request. @zara"""
    # Cloudflare specific header (highest priority)
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()

    # Standard proxy header
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    # Direct connection
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
    CONF_DASHBOARD_STYLE,
    DEFAULT_DASHBOARD_STYLE,
    CONF_THEME,
    DEFAULT_THEME,
    # Consumer sensors (Wärmepumpe, Heizstab, Wallbox)
    CONF_SENSOR_HEATPUMP_POWER,
    CONF_SENSOR_HEATPUMP_DAILY,
    CONF_SENSOR_HEATPUMP_COP,
    CONF_SENSOR_HEATINGROD_POWER,
    CONF_SENSOR_HEATINGROD_DAILY,
    CONF_SENSOR_WALLBOX_POWER,
    CONF_SENSOR_WALLBOX_DAILY,
    CONF_SENSOR_WALLBOX_STATE,
    DEFAULT_HEATPUMP_COP,
)
from ..utils import get_json_cache, read_json_safe

if TYPE_CHECKING:
    from aiohttp.web import Request, Response

_LOGGER = logging.getLogger(__name__)


class APIContext:
    """Singleton context for API views. @zara

    Replaces global variables with a proper singleton pattern.
    Provides access to Home Assistant instance and paths.
    """

    _instance: "APIContext | None" = None

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the context. @zara"""
        self.hass = hass
        self.config_path = Path(hass.config.path())
        self.solar_path = self.config_path / "solar_forecast_ml"
        self.grid_path = self.config_path / "grid_price_monitor"

    @classmethod
    def get(cls) -> "APIContext":
        """Get the singleton instance. @zara

        Raises:
            RuntimeError: If context has not been initialized.
        """
        if cls._instance is None:
            raise RuntimeError("APIContext not initialized - call initialize() first")
        return cls._instance

    @classmethod
    def initialize(cls, hass: HomeAssistant) -> "APIContext":
        """Initialize the singleton instance. @zara

        Args:
            hass: Home Assistant instance.

        Returns:
            The initialized APIContext.
        """
        cls._instance = cls(hass)
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if context is initialized. @zara"""
        return cls._instance is not None


# Backwards compatibility - these will be removed in future versions
SOLAR_PATH: Path | None = None
GRID_PATH: Path | None = None
HASS: HomeAssistant | None = None


async def async_setup_views(hass: HomeAssistant) -> None:
    """Register all API views. @zara"""
    global SOLAR_PATH, GRID_PATH, HASS

    # Initialize new APIContext
    ctx = APIContext.initialize(hass)

    # Maintain backwards compatibility
    HASS = hass
    config_path = Path(hass.config.path())
    SOLAR_PATH = config_path / "solar_forecast_ml"
    GRID_PATH = config_path / "grid_price_monitor"

    _LOGGER.debug("SFML Stats paths: Solar=%s, Grid=%s", ctx.solar_path, ctx.grid_path)

    hass.http.register_view(HealthCheckView())
    hass.http.register_view(DashboardView())
    hass.http.register_view(LcarsDashboardView())
    hass.http.register_view(HelpView())
    hass.http.register_view(HelpSFMLView())
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

    # Monthly Tariffs (EEG/Energy Sharing support)
    hass.http.register_view(MonthlyTariffsView())
    hass.http.register_view(MonthlyTariffDetailView())
    hass.http.register_view(MonthlyTariffFinalizeView())
    hass.http.register_view(MonthlyTariffsExportView())
    hass.http.register_view(MonthlyTariffsDefaultsView())

    # Weekly Report Export (Modern Redesign)
    hass.http.register_view(ExportWeeklyReportView())

    # Background Image for Dashboard
    hass.http.register_view(BackgroundImageView())

    # Forecast Comparison
    hass.http.register_view(ForecastComparisonView())
    hass.http.register_view(ForecastComparisonChartView())

    # Dashboard Settings (for style toggle)
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


class HealthCheckView(HomeAssistantView):
    """Health check endpoint for monitoring. @zara

    Returns the health status of the SFML Stats integration,
    including availability of data sources and configuration status.
    """

    url = "/api/sfml_stats/health"
    name = "api:sfml_stats:health"
    requires_auth = False

    @local_only
    async def get(self, request: Request) -> Response:
        """Return health status. @zara"""
        try:
            ctx = APIContext.get()

            # Check various health indicators
            checks = {
                "solar_data_available": ctx.solar_path.exists(),
                "grid_data_available": ctx.grid_path.exists(),
                "integration_loaded": DOMAIN in ctx.hass.data,
                "config_entries_present": len(
                    ctx.hass.config_entries.async_entries(DOMAIN)
                ) > 0,
            }

            # Check for specific data files
            if checks["solar_data_available"]:
                checks["solar_stats_available"] = (
                    ctx.solar_path / "stats" / "daily_summaries.json"
                ).exists()
            else:
                checks["solar_stats_available"] = False

            if checks["grid_data_available"]:
                checks["grid_prices_available"] = (
                    ctx.grid_path / "data" / "price_cache.json"
                ).exists()
            else:
                checks["grid_prices_available"] = False

            # Determine overall health
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
            # APIContext not initialized
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

        # Try via hass.config.path() first (works in Docker)
        if HASS is not None:
            frontend_path = Path(HASS.config.path()) / "custom_components" / "sfml_stats" / "frontend" / "dist" / "index.html"
            if not frontend_path.exists():
                frontend_path = None

        # Fallback via __file__
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

        # Try via hass.config.path() first (works in Docker)
        if HASS is not None:
            frontend_path = Path(HASS.config.path()) / "custom_components" / "sfml_stats" / "frontend" / "dist" / "index-lcars.html"
            if not frontend_path.exists():
                frontend_path = None

        # Fallback via __file__
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
        """Return fallback HTML when LCARS build is not present. @zara"""
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


class HelpView(HomeAssistantView):
    """Help page with sensor configuration documentation. @zara"""

    url = "/api/sfml_stats/help"
    name = "api:sfml_stats:help"
    requires_auth = False
    cors_allowed = True

    @local_only
    async def get(self, request: Request) -> Response:
        """Return the help page HTML. @zara"""
        frontend_path = None

        # Try via hass.config.path() first (works in Docker)
        if HASS is not None:
            frontend_path = Path(HASS.config.path()) / "custom_components" / "sfml_stats" / "frontend" / "dist" / "help.html"
            if not frontend_path.exists():
                frontend_path = None

        # Fallback via __file__
        if frontend_path is None:
            frontend_path = Path(__file__).parent.parent / "frontend" / "dist" / "help.html"

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
        """Return fallback HTML when help page is not present. @zara"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>SFML Stats - Help</title>
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
        h1 { color: #00d4ff; }
        a { color: #a855f7; }
    </style>
</head>
<body>
    <div class="message">
        <h1>Sensor Configuration Help</h1>
        <p>Help page is being loaded...</p>
        <p style="color: #666;">If this message persists, the help page was not built yet.</p>
        <p><a href="/api/sfml_stats/dashboard">Back to Dashboard</a></p>
    </div>
</body>
</html>"""


class HelpSFMLView(HomeAssistantView):
    """Solar Forecast ML help page with sensor configuration documentation. @zara"""

    url = "/api/sfml_stats/help-sfml"
    name = "api:sfml_stats:help_sfml"
    requires_auth = False
    cors_allowed = True

    @local_only
    async def get(self, request: Request) -> Response:
        """Return the SFML help page HTML. @zara"""
        frontend_path = None

        # Try via hass.config.path() first (works in Docker)
        if HASS is not None:
            frontend_path = Path(HASS.config.path()) / "custom_components" / "sfml_stats" / "frontend" / "dist" / "help-sfml.html"
            if not frontend_path.exists():
                frontend_path = None

        # Fallback via __file__
        if frontend_path is None:
            frontend_path = Path(__file__).parent.parent / "frontend" / "dist" / "help-sfml.html"

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
        """Return fallback HTML when SFML help page is not present. @zara"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Solar Forecast ML - Help</title>
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
        h1 { color: #ff9500; }
        a { color: #ffd60a; }
    </style>
</head>
<body>
    <div class="message">
        <h1>☀️ Solar Forecast ML - Sensor Help</h1>
        <p>Help page is being loaded...</p>
        <p style="color: #666;">If this message persists, the help page was not built yet.</p>
        <p><a href="/api/sfml_stats/dashboard">Back to Dashboard</a></p>
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
        """Return a static file. @zara

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

        # Try via hass.config.path() first (works in Docker)
        if HASS is not None:
            frontend_path = Path(HASS.config.path()) / "custom_components" / "sfml_stats" / "frontend" / "dist" / subdir / clean_filename
            if not frontend_path.exists():
                # Fallback to assets folder for backward compatibility
                frontend_path = Path(HASS.config.path()) / "custom_components" / "sfml_stats" / "frontend" / "dist" / "assets" / filename
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

        # Multi-day hourly forecast (Heute, Morgen, Übermorgen)
        multi_day = await _read_json_file(SOLAR_PATH / "stats" / "multi_day_hourly_forecast.json")
        if multi_day and "days" in multi_day:
            result["data"]["multi_day_hourly"] = multi_day["days"]

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
            # Support both "price_net" and "price" field names for compatibility
            recent_prices = [
                p.get("price_net") or p.get("price") or 0
                for p in prices["prices"][-48:]
                if p.get("price_net") or p.get("price")
            ]
            if recent_prices:
                result["kpis"]["price_current"] = recent_prices[-1] if recent_prices else 0
                result["kpis"]["price_avg"] = sum(recent_prices) / len(recent_prices)
                result["kpis"]["price_min"] = min(recent_prices)
                result["kpis"]["price_max"] = max(recent_prices)

        ai_weights = await _read_json_file(SOLAR_PATH / "ai" / "learned_weights.json")
        if ai_weights:
            result["kpis"]["ai_training_samples"] = ai_weights.get("training_samples", 0)

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
        # SFML domain - coordinator is stored directly (not as dict)
        sfml_domain = "solar_forecast_ml"
        entries = HASS.data.get(sfml_domain, {})

        for entry_id, entry_data in entries.items():
            # Coordinator is stored directly, not as {"coordinator": ...}
            # Check if entry_data has panel_groups attribute (it's the coordinator itself)
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

        # Convert Wh to kWh if needed
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

    # Windgeschwindigkeit: Konvertiere km/h zu m/s falls nötig
    # HA Weather-Entitäten liefern wind_speed_unit als Attribut
    wind_speed = attrs.get("wind_speed")
    if wind_speed is not None:
        wind_speed_unit = attrs.get("wind_speed_unit", "km/h")
        if wind_speed_unit == "km/h":
            wind_speed = round(wind_speed / 3.6, 1)  # km/h -> m/s
        # Wenn bereits m/s, keine Konvertierung nötig

    return {
        "state": state.state,  # z.B. "sunny", "cloudy", etc.
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

        # DEBUG: Log configured sensor keys
        _LOGGER.info(
            "EnergyFlowView: solar_to_house config key = %s, solar_to_battery config key = %s",
            config.get(CONF_SENSOR_SOLAR_TO_HOUSE),
            config.get(CONF_SENSOR_SOLAR_TO_BATTERY),
        )

        # Solar kann NIEMALS negativ sein - korrigiere negative Werte
        solar_power = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_POWER))
        solar_to_house = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_TO_HOUSE))
        solar_to_battery = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_TO_BATTERY))

        # DEBUG: Log sensor values
        _LOGGER.info(
            "EnergyFlowView: solar_power = %s, solar_to_house = %s, solar_to_battery = %s",
            solar_power, solar_to_house, solar_to_battery,
        )
        if solar_power is not None and solar_power < 0:
            solar_power = 0.0
        if solar_to_house is not None and solar_to_house < 0:
            solar_to_house = 0.0
        if solar_to_battery is not None and solar_to_battery < 0:
            solar_to_battery = 0.0

        # Wenn keine Solarproduktion, kann auch nichts zur Batterie/Haus fließen
        # Dies korrigiert fehlerhafte Sensor-Berechnungen
        if solar_power is not None and solar_power <= 0:
            solar_to_house = 0.0
            solar_to_battery = 0.0
        # solar_to_battery kann nie größer sein als solar_power
        elif solar_power is not None and solar_to_battery is not None:
            solar_to_battery = min(solar_to_battery, solar_power)
        # solar_to_house kann nie größer sein als solar_power
        if solar_power is not None and solar_to_house is not None:
            solar_to_house = min(solar_to_house, solar_power)

        # Prüfe ob Batterie konfiguriert ist (battery_soc ist der Haupt-Indikator)
        battery_configured = config.get(CONF_SENSOR_BATTERY_SOC) is not None
        battery_soc = _get_sensor_value(config.get(CONF_SENSOR_BATTERY_SOC)) if battery_configured else None
        battery_power = _get_sensor_value(config.get(CONF_SENSOR_BATTERY_POWER)) if battery_configured else None
        battery_to_house = _get_sensor_value(config.get(CONF_SENSOR_BATTERY_TO_HOUSE)) if battery_configured else None
        grid_to_battery = _get_sensor_value(config.get(CONF_SENSOR_GRID_TO_BATTERY)) if battery_configured else None

        # Wenn keine Batterie konfiguriert, auch solar_to_battery auf None setzen
        if not battery_configured:
            solar_to_battery = None

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
            "consumers": self._get_consumer_data(config),
            "weather_ha": _get_weather_data(config.get(CONF_WEATHER_ENTITY)),
            "sun_position": await self._get_sun_position(),
            "current_price": await self._get_current_price(),
            "feed_in_tariff": config.get(CONF_FEED_IN_TARIFF, DEFAULT_FEED_IN_TARIFF),
        }

        # DEBUG: Log final result flows
        _LOGGER.info(
            "EnergyFlowView FINAL: solar_to_house = %s, solar_to_battery = %s",
            result["flows"]["solar_to_house"],
            result["flows"]["solar_to_battery"],
        )

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
                # Support both "price_net" and "price" field names for compatibility
                net_price = p.get("price_net") or p.get("price")
                return {
                    "total_price": p.get("total_price"),
                    "net_price": net_price,
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

    def _get_consumer_data(self, config: dict[str, Any]) -> dict[str, Any]:
        """Read consumer data from configured sensors (WP, Heizstab, Wallbox). @zara"""
        consumers = {
            "heatpump": None,
            "heatingrod": None,
            "wallbox": None,
        }

        # Wärmepumpe (Heat Pump)
        if config.get(CONF_SENSOR_HEATPUMP_POWER):
            power = _get_sensor_value(config.get(CONF_SENSOR_HEATPUMP_POWER))
            daily = _get_sensor_value(config.get(CONF_SENSOR_HEATPUMP_DAILY))
            cop = _get_sensor_value(config.get(CONF_SENSOR_HEATPUMP_COP))

            # Fallback to default COP if not configured
            if cop is None:
                cop = DEFAULT_HEATPUMP_COP

            consumers["heatpump"] = {
                "power": power,
                "daily_kwh": daily,
                "cop": cop,
                "configured": True,
            }

        # Heizstab (Heating Rod)
        if config.get(CONF_SENSOR_HEATINGROD_POWER):
            power = _get_sensor_value(config.get(CONF_SENSOR_HEATINGROD_POWER))
            daily = _get_sensor_value(config.get(CONF_SENSOR_HEATINGROD_DAILY))

            consumers["heatingrod"] = {
                "power": power,
                "daily_kwh": daily,
                "configured": True,
            }

        # Wallbox (EV Charger)
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
        # Get live sensor values from SFML panel group config
        sfml_groups = _get_sfml_panel_groups()
        live_sensor_values: dict[str, float] = {}
        for group in sfml_groups:
            group_name = group.get("name", "")
            energy_sensor = group.get("energy_sensor", "")
            if group_name and energy_sensor:
                value = _read_panel_group_sensor(energy_sensor)
                if value is not None:
                    live_sensor_values[group_name] = value

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

        # Get panel group name mapping from config
        config = _get_config()
        name_mapping = config.get(CONF_PANEL_GROUP_NAMES, {})
        if not isinstance(name_mapping, dict):
            name_mapping = {}

        groups = {}
        for group_name in sorted(group_names):
            # Apply name mapping: use custom name if configured, otherwise original name
            display_name = name_mapping.get(group_name, group_name)

            group_data = {
                "name": display_name,
                "original_name": group_name,  # Keep original for reference
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

            # Use live sensor value if available (more accurate than hourly sum)
            original_name = group_data.get("original_name", group_name)
            if original_name in live_sensor_values:
                group_data["actual_total_kwh"] = live_sensor_values[original_name]
                group_data["actual_source"] = "live_sensor"
            else:
                group_data["actual_source"] = "hourly_sum"

            # Calculate accuracy: 100% - |deviation%|
            # Accuracy can never be >100% or <0%
            if group_data["prediction_total_kwh"] > 0 and group_data["actual_total_kwh"] > 0:
                deviation_percent = abs(
                    (group_data["actual_total_kwh"] - group_data["prediction_total_kwh"])
                    / group_data["prediction_total_kwh"]
                ) * 100
                group_data["accuracy_percent"] = max(0, min(100, 100 - deviation_percent))
            else:
                group_data["accuracy_percent"] = None

            # Use display_name as key for the groups dict
            groups[display_name] = group_data

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
    """View to export solar analytics as PNG (Matplotlib)."""

    url = "/api/sfml_stats/export_solar_analytics"
    name = "api:sfml_stats:export_solar_analytics"
    requires_auth = False

    @local_only
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

    @local_only
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

    @local_only
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

    @local_only
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

    @local_only
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


class WeatherComparisonView(HomeAssistantView):
    """View to get IST vs KI weather comparison data."""

    url = "/api/sfml_stats/weather_comparison"
    name = "api:sfml_stats:weather_comparison"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Get IST vs KI weather comparison data."""
        try:
            from ..weather_collector import WeatherDataCollector

            days = int(request.query.get("days", 7))
            days = min(days, 30)  # Max 30 days

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
    """View to export weather analytics as PNG."""

    url = "/api/sfml_stats/export_weather_analytics"
    name = "api:sfml_stats:export_weather_analytics"
    requires_auth = False

    @local_only
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

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Get power sources history from Home Assistant Recorder. @zara"""
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

            # Try collector data FIRST - it's more reliable than recorder
            data_source = "collector"
            collector_data = await self._get_power_sources_collector_data(hours)

            if collector_data and len(collector_data) > 0:
                processed_data = collector_data
                _LOGGER.info("Got %d entries from power sources collector", len(collector_data))
            else:
                # Fallback to recorder if collector has no data
                _LOGGER.info("No collector data, trying recorder")
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(hours=hours)

                history_data = await self._get_recorder_history(
                    entity_ids, start_time, end_time
                )

                # Process and align data
                processed_data = self._process_history(history_data, sensors, start_time, end_time)
                data_source = "recorder"

                # Check if we got any actual data from recorder
                has_data = any(
                    any(d.get(k) is not None for k in ['solar_power', 'solar_to_house', 'solar_to_battery', 'battery_to_house', 'grid_to_house', 'home_consumption'])
                    for d in processed_data
                )

                if not has_data:
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

            # Convert reactive proxy to plain dict if needed
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

            # Merge with daily_energy_history.json for more complete data
            history_path = Path(HASS.config.path()) / "sfml_stats" / "data" / "daily_energy_history.json"
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
                                "peak_consumption_w": day_data.get("peak_battery_power_w", 0),  # Use battery peak as proxy
                            }
                        else:
                            # Merge additional fields from history into existing day data
                            existing = daily_stats["days"][date_str]
                            if existing.get("peak_battery_power_w") is None or existing.get("peak_battery_power_w") == 0:
                                existing["peak_battery_power_w"] = day_data.get("peak_battery_power_w", 0)
                            # Also merge home_consumption, autarky, etc. if missing
                            if existing.get("home_consumption_kwh") is None or existing.get("home_consumption_kwh") == 0:
                                existing["home_consumption_kwh"] = day_data.get("home_consumption_kwh", 0)
                            if existing.get("autarky_percent") is None or existing.get("autarky_percent") == 0:
                                existing["autarky_percent"] = day_data.get("autarky_percent", 0)
                            if existing.get("self_consumption_percent") is None or existing.get("self_consumption_percent") == 0:
                                existing["self_consumption_percent"] = day_data.get("self_consumption_percent", 0)
                            if existing.get("peak_consumption_w") is None or existing.get("peak_consumption_w") == 0:
                                existing["peak_consumption_w"] = day_data.get("peak_battery_power_w", 0)

            # Also get current sensor values for real-time display
            config = _get_config()
            # Solar kann NIEMALS negativ sein
            solar_power_val = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_POWER))
            solar_to_house_val = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_TO_HOUSE))
            solar_to_battery_val = _get_sensor_value(config.get(CONF_SENSOR_SOLAR_TO_BATTERY))
            if solar_power_val is not None and solar_power_val < 0:
                solar_power_val = 0.0
            if solar_to_house_val is not None and solar_to_house_val < 0:
                solar_to_house_val = 0.0
            if solar_to_battery_val is not None and solar_to_battery_val < 0:
                solar_to_battery_val = 0.0
            # Wenn keine Solarproduktion, kann auch nichts zur Batterie/Haus fließen
            if solar_power_val is not None and solar_power_val <= 0:
                solar_to_house_val = 0.0
                solar_to_battery_val = 0.0
            elif solar_power_val is not None:
                if solar_to_battery_val is not None:
                    solar_to_battery_val = min(solar_to_battery_val, solar_power_val)
                if solar_to_house_val is not None:
                    solar_to_house_val = min(solar_to_house_val, solar_power_val)
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

            # Get weather data from Solar Forecast ML cache
            weather_data = await self._get_weather_data()
            if not weather_data:
                return web.json_response({
                    "success": False,
                    "error": "No weather data available"
                })

            # Get hourly forecast for rain probability
            forecast_hours = await self._get_forecast_hours()

            # Generate recommendation
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
        """Get current weather data from Solar Forecast ML. @zara"""
        # Try open_meteo_cache.json first
        cache_data = await _read_json_file(SOLAR_PATH / "data" / "open_meteo_cache.json")
        if cache_data and "forecast" in cache_data:
            today_str = date.today().isoformat()
            current_hour = datetime.now().hour

            if today_str in cache_data["forecast"]:
                hour_data = cache_data["forecast"][today_str].get(str(current_hour), {})
                if hour_data:
                    return {
                        "temperature": hour_data.get("temperature", 15),
                        "humidity": hour_data.get("humidity", 50),
                        "wind_speed": hour_data.get("wind_speed", 0),
                        "precipitation": hour_data.get("precipitation", 0),
                        "cloud_cover": hour_data.get("cloud_cover", 50),
                        "pressure": hour_data.get("pressure", 1013),
                        "radiation": hour_data.get("ghi", 0) or hour_data.get("direct_radiation", 0),
                        "uv_index": hour_data.get("uv_index", 0),
                    }

        # Fallback: try hourly_weather_actual.json
        weather_actual = await _read_json_file(SOLAR_PATH / "stats" / "hourly_weather_actual.json")
        if weather_actual and "hourly_data" in weather_actual:
            today_str = date.today().isoformat()
            current_hour = str(datetime.now().hour)

            if today_str in weather_actual["hourly_data"]:
                hour_data = weather_actual["hourly_data"][today_str].get(current_hour, {})
                if hour_data:
                    return {
                        "temperature": hour_data.get("temperature", 15),
                        "humidity": hour_data.get("humidity", 50),
                        "wind_speed": hour_data.get("wind_speed", 0),
                        "precipitation": hour_data.get("precipitation", 0),
                        "cloud_cover": hour_data.get("cloud_cover", 50),
                        "pressure": hour_data.get("pressure", 1013),
                        "radiation": hour_data.get("radiation", 0),
                        "uv_index": hour_data.get("uv_index", 0),
                    }

        # Last fallback: HA weather entity
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
        """Get hourly forecast for rain probability. @zara"""
        cache_data = await _read_json_file(SOLAR_PATH / "data" / "open_meteo_cache.json")
        if not cache_data or "forecast" not in cache_data:
            return None

        today_str = date.today().isoformat()
        current_hour = datetime.now().hour
        forecast_hours = []

        if today_str in cache_data["forecast"]:
            for hour in range(current_hour, 24):
                hour_data = cache_data["forecast"][today_str].get(str(hour), {})
                if hour_data:
                    forecast_hours.append({
                        "hour": hour,
                        "precipitation_probability": hour_data.get("precipitation_probability", 0),
                        "precipitation": hour_data.get("precipitation", 0),
                    })

        return forecast_hours if forecast_hours else None


class MonthlyTariffsView(HomeAssistantView):
    """API for monthly tariffs management (EEG/Energy Sharing support). @zara"""

    url = "/api/sfml_stats/monthly_tariffs"
    name = "api:sfml_stats:monthly_tariffs"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Get monthly tariffs data. @zara

        Query params:
        - year: Year to fetch (default: current year)
        - include_empty: Include months without data (default: false)
        """
        try:
            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized",
                })

            # Get tariff manager from entry data
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
        """Update overrides for a specific month. @zara

        Request body:
        {
            "overrides": {
                "import_price_ct": 32.5,
                "export_price_ct": 7.2,
                "reference_price_ct": 26.0,
                "grid_fee_ct": 18.0,
                "eeg_share_percent": 45.0
            }
        }
        """
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
                # Return updated data
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
        """Finalize a month and optionally recalculate history. @zara

        Request body:
        {
            "recalculate_history": true  // Optional, default true
        }
        """
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
        """Export monthly tariffs as CSV. @zara

        Query params:
        - start: Start month in YYYY-MM format (default: January of current year)
        - end: End month in YYYY-MM format (default: current month)
        """
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
        """Update default tariff settings. @zara

        Request body:
        {
            "reference_price_ct": 26.0,
            "feed_in_tariff_ct": 8.1,
            "eeg_import_price_ct": 18.0,
            "eeg_feed_in_ct": 12.0,
            "grid_fee_base_ct": 13.0,
            "grid_fee_scaling_enabled": true
        }
        """
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
    """View to generate and export weekly report as PNG. @zara

    GET/POST /api/sfml_stats/export_weekly_report
    Optional JSON body: {"year": 2025, "week": 1}
    If not provided, uses current week.

    Returns PNG image and saves to sfml_stats/weekly/ folder.
    """

    url = "/api/sfml_stats/export_weekly_report"
    name = "api:sfml_stats:export_weekly_report"
    requires_auth = False

    @local_only
    async def post(self, request: web.Request) -> web.Response:
        """Generate and return weekly report PNG."""
        try:
            from datetime import date
            from pathlib import Path
            from ..charts.weekly_report import WeeklyReportChart
            from ..storage import DataValidator

            # Parse optional parameters
            try:
                data = await request.json()
            except Exception:
                data = {}

            year = data.get("year")
            week = data.get("week")

            # Default to current week
            if year is None or week is None:
                today = date.today()
                iso = today.isocalendar()
                year = year or iso[0]
                week = week or iso[1]

            _LOGGER.info("Generating weekly report: KW %d/%d (Modern Redesign)", week, year)

            # Get validator from HASS data
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

            # Create chart and generate
            chart = WeeklyReportChart(validator)
            fig = await chart.generate(year=year, week=week)

            # Render to PNG bytes
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

            # Also save to file
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
        """GET method - generate with default parameters (current week)."""
        from datetime import date

        today = date.today()
        iso = today.isocalendar()

        # Create response directly
        class MockRequest:
            async def json(self):
                return {"year": iso[0], "week": iso[1]}

        mock = MockRequest()
        return await self.post(mock)


class BackgroundImageView(HomeAssistantView):
    """Serve the dashboard background image. @zara

    GET /api/sfml_stats/background
    Returns the background image from frontend/dist/background.png
    """

    url = "/api/sfml_stats/background"
    name = "api:sfml_stats:background"
    requires_auth = False

    async def get(self, request: web.Request) -> web.Response:
        """Return the background image."""
        from pathlib import Path

        bg_path = None
        paths_tried = []

        # Try paths in order of preference
        # 1. First try via hass.config.path() (works in Docker container)
        if HASS is not None:
            # Primary: custom_components path
            primary = Path(HASS.config.path()) / "custom_components" / "sfml_stats" / "frontend" / "dist" / "background.png"
            paths_tried.append(str(primary))
            if primary.exists():
                bg_path = primary

            # Alternative: sfml_stats data folder
            if bg_path is None:
                alt_path = Path(HASS.config.path()) / "sfml_stats" / "background.png"
                paths_tried.append(str(alt_path))
                if alt_path.exists():
                    bg_path = alt_path

        # 2. Fallback via __file__ (for development/testing)
        if bg_path is None:
            file_path = Path(__file__).parent.parent / "frontend" / "dist" / "background.png"
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
                content_type="image/png",
                headers={
                    "Cache-Control": "public, max-age=86400",  # Cache for 24h
                }
            )
        except Exception as err:
            _LOGGER.error("Error serving background image: %s", err)
            return web.Response(status=500, text=str(err))


class ForecastComparisonView(HomeAssistantView):
    """View to get forecast comparison data for the last 7 days. @zara

    GET /api/sfml_stats/forecast_comparison
    Optional query params: ?days=7

    Returns JSON with comparison data for charting.
    """

    url = "/api/sfml_stats/forecast_comparison"
    name = "api:sfml_stats:forecast_comparison"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Return forecast comparison data as JSON."""
        try:
            from ..readers.forecast_comparison_reader import ForecastComparisonReader

            # Parse optional days parameter
            days = int(request.query.get("days", "7"))
            days = min(max(days, 1), 30)  # Clamp between 1 and 30

            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized"
                }, status=500)

            config_path = Path(HASS.config.path())
            reader = ForecastComparisonReader(config_path)

            if not reader.is_available:
                return web.json_response({
                    "success": False,
                    "error": "No forecast comparison data available yet",
                    "hint": "Data is collected daily at 23:50"
                }, status=404)

            # Use async_get_chart_data() which returns the correct format for frontend
            # Format: {dates: [], actual: [], sfml: [], external_1: [], external_2: [], stats: {}}
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
    """View to generate and return forecast comparison chart as PNG. @zara

    GET/POST /api/sfml_stats/forecast_comparison_chart
    Optional JSON body: {"days": 7}

    Returns PNG image.
    """

    url = "/api/sfml_stats/forecast_comparison_chart"
    name = "api:sfml_stats:forecast_comparison_chart"
    requires_auth = False

    @local_only
    async def get(self, request: web.Request) -> web.Response:
        """Generate and return forecast comparison chart as PNG."""
        try:
            from ..charts.forecast_comparison import ForecastComparisonChart
            from ..storage import DataValidator

            # Parse optional days parameter
            days = int(request.query.get("days", "7"))
            days = min(max(days, 1), 30)  # Clamp between 1 and 30

            if HASS is None:
                return web.json_response({
                    "success": False,
                    "error": "Home Assistant not initialized"
                }, status=500)

            # Get validator from HASS data
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

            # Create chart and generate
            chart = ForecastComparisonChart(validator)
            fig = await chart.generate(days=days)

            # Render to PNG bytes
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
        """Handle POST request (same as GET)."""
        try:
            data = await request.json()
            days = data.get("days", 7)
        except Exception:
            days = 7

        # Create mock request with days parameter
        class MockRequest:
            query = {"days": str(days)}

        return await self.get(MockRequest())


class DashboardSettingsView(HomeAssistantView):
    """API endpoint for dashboard settings (theme, style). @zara

    GET /api/sfml_stats/dashboard_settings
    Returns current dashboard settings including dashboard_style and theme.

    POST /api/sfml_stats/dashboard_settings
    Updates dashboard settings (stores in session, does not persist to config).
    """

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
        """Update dashboard settings (session only, not persisted). @zara

        This allows the frontend to temporarily switch styles without
        modifying the Home Assistant config entry.
        """
        try:
            data = await request.json()
            dashboard_style = data.get("dashboard_style")
            theme = data.get("theme")

            # Store in HASS data for session (not persisted)
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
