# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML - ML Weather
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************
"""DataUpdateCoordinator for ML Weather."""

import logging
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    DEFAULT_DB_PATH,
    DEFAULT_SCAN_INTERVAL,
    CONF_DATA_PATH,
    CONDITION_MAP,
    FORECAST_HOURS,
    FORECAST_DAYS,
    RAIN_THRESHOLD_LIGHT,
    RAIN_THRESHOLD_MODERATE,
)
from .db_reader import DatabaseReader

_LOGGER = logging.getLogger(__name__)

# Retry constants
MAX_CONSECUTIVE_FAILURES = 5
BACKOFF_MULTIPLIER = 2  # Double interval on each failure
MAX_BACKOFF_MINUTES = 60  # Cap at 1 hour


class MLWeatherCoordinator(DataUpdateCoordinator):
    """Coordinator to manage fetching ML corrected weather data."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        self.hass = hass
        self.entry = entry
        db_path = entry.data.get(CONF_DATA_PATH, DEFAULT_DB_PATH)
        self._db_reader = DatabaseReader(hass, db_path)
        self._raw_data: dict = {}
        self._pv_forecast_data: dict = {}
        self._consecutive_failures: int = 0
        self._base_interval = DEFAULT_SCAN_INTERVAL

        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=DEFAULT_SCAN_INTERVAL,
        )

    async def async_config_entry_first_refresh(self) -> None:
        """Initialize DB connection and perform first refresh."""
        await self._db_reader.async_connect()
        await super().async_config_entry_first_refresh()

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch data from SQLite database with retry backoff."""
        try:
            # Calculate date range: today through FORECAST_DAYS ahead
            now = dt_util.now()
            start_date = now.strftime("%Y-%m-%d")
            end_date = (now + timedelta(days=FORECAST_DAYS)).strftime("%Y-%m-%d")

            # Read weather forecast from DB
            forecast_data = await self._db_reader.async_get_weather_forecast(
                start_date, end_date
            )

            if not forecast_data:
                self._handle_failure()
                raise UpdateFailed("No weather data available from database")

            self._raw_data = forecast_data

            # Read PV forecast from DB
            self._pv_forecast_data = await self._db_reader.async_get_pv_forecast()

            # Get version
            version = await self._db_reader.async_get_weather_version()

            # Success - reset failure counter and interval
            self._handle_success()

            return self._process_data(forecast_data, version)

        except UpdateFailed:
            raise
        except Exception as err:
            self._handle_failure()
            raise UpdateFailed(f"Error loading weather data from database: {err}") from err

    def _handle_failure(self) -> None:
        """Handle update failure with exponential backoff."""
        self._consecutive_failures += 1

        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            # Apply exponential backoff
            backoff_minutes = min(
                self._base_interval.total_seconds() / 60 * (BACKOFF_MULTIPLIER ** (self._consecutive_failures - MAX_CONSECUTIVE_FAILURES + 1)),
                MAX_BACKOFF_MINUTES
            )
            self.update_interval = timedelta(minutes=backoff_minutes)
            _LOGGER.warning(
                "Multiple consecutive failures (%d). "
                "Backing off update interval to %.1f minutes",
                self._consecutive_failures,
                backoff_minutes
            )
        else:
            _LOGGER.debug(
                "Update failure %d of %d before backoff",
                self._consecutive_failures,
                MAX_CONSECUTIVE_FAILURES
            )

    def _handle_success(self) -> None:
        """Handle successful update - reset counters."""
        if self._consecutive_failures > 0:
            _LOGGER.info(
                "Update successful after %d failures. Resetting interval.",
                self._consecutive_failures
            )
        self._consecutive_failures = 0
        self.update_interval = self._base_interval

    def _process_data(self, forecast_data: dict, version: str) -> dict[str, Any]:
        """Process forecast data into usable format."""
        # Get current weather (current hour)
        now = dt_util.now()
        today_str = now.strftime("%Y-%m-%d")
        current_hour = str(now.hour)

        current = self._get_hour_data(forecast_data, today_str, current_hour)

        return {
            "current": current,
            "forecast": forecast_data,
            "version": version,
            "metadata": {},
        }

    def _get_hour_data(self, forecast: dict, date_str: str, hour: str) -> dict[str, Any]:
        """Get weather data for a specific hour."""
        day_data = forecast.get(date_str, {})
        hour_data = day_data.get(hour, {})

        if not hour_data:
            _LOGGER.debug(f"No data for {date_str} hour {hour}, using defaults")
            return self._default_weather()

        return {
            "temperature": hour_data.get("temperature"),
            "humidity": hour_data.get("humidity"),
            "pressure": hour_data.get("pressure"),
            "wind_speed": hour_data.get("wind"),
            "precipitation": hour_data.get("rain", 0),
            "cloud_coverage": hour_data.get("clouds"),
            "cloud_cover_low": hour_data.get("cloud_cover_low"),
            "cloud_cover_mid": hour_data.get("cloud_cover_mid"),
            "cloud_cover_high": hour_data.get("cloud_cover_high"),
            "solar_radiation": hour_data.get("solar_radiation_wm2"),
            "direct_radiation": hour_data.get("direct_radiation"),
            "diffuse_radiation": hour_data.get("diffuse_radiation"),
            "visibility": hour_data.get("visibility_m"),
            "fog_detected": hour_data.get("fog_detected"),
            "fog_type": hour_data.get("fog_type"),
            "condition": self._get_condition(hour_data),
        }

    def _default_weather(self) -> dict[str, Any]:
        """Return default weather values."""
        return {
            "temperature": None,
            "humidity": None,
            "pressure": None,
            "wind_speed": None,
            "precipitation": 0,
            "cloud_coverage": None,
            "cloud_cover_low": None,
            "cloud_cover_mid": None,
            "cloud_cover_high": None,
            "solar_radiation": None,
            "direct_radiation": None,
            "diffuse_radiation": None,
            "visibility": None,
            "fog_detected": None,
            "fog_type": None,
            "condition": "unknown",
        }

    def _get_condition(self, hour_data: dict) -> str:
        """Determine weather condition from cloud coverage and rain."""
        rain = hour_data.get("rain", 0) or 0
        clouds = hour_data.get("clouds", 0) or 0

        # Check for precipitation first (using constants)
        if rain > RAIN_THRESHOLD_MODERATE:
            return "rainy"
        if rain > RAIN_THRESHOLD_LIGHT:
            return "rainy"

        # Check cloud coverage using condition map
        for (low, high), condition in CONDITION_MAP.items():
            if low <= clouds < high:
                return condition

        # Fallback for edge cases
        return "sunny"

    def get_current_weather(self) -> dict[str, Any]:
        """Get current weather data."""
        if self.data:
            return self.data.get("current", self._default_weather())
        return self._default_weather()

    def get_hourly_forecast(self) -> list[dict[str, Any]]:
        """Get hourly forecast for the next FORECAST_HOURS hours."""
        if not self.data:
            return []

        forecast_data = self.data.get("forecast", {})
        forecasts = []
        now = dt_util.now()

        # Generate forecasts for next FORECAST_HOURS hours
        for hours_ahead in range(FORECAST_HOURS):
            forecast_time = now + timedelta(hours=hours_ahead)
            date_str = forecast_time.strftime("%Y-%m-%d")
            hour_str = str(forecast_time.hour)

            hour_data = self._get_hour_data(forecast_data, date_str, hour_str)

            if hour_data.get("temperature") is not None:
                forecasts.append({
                    "datetime": forecast_time.isoformat(),
                    "temperature": hour_data.get("temperature"),
                    "humidity": hour_data.get("humidity"),
                    "pressure": hour_data.get("pressure"),
                    "wind_speed": hour_data.get("wind_speed"),
                    "precipitation": hour_data.get("precipitation", 0),
                    "cloud_coverage": hour_data.get("cloud_coverage"),
                    "condition": hour_data.get("condition", "unknown"),
                    "solar_radiation": hour_data.get("solar_radiation"),
                    "direct_radiation": hour_data.get("direct_radiation"),
                    "diffuse_radiation": hour_data.get("diffuse_radiation"),
                })

        return forecasts

    def get_daily_forecast(self) -> list[dict[str, Any]]:
        """Get daily forecast summary."""
        if not self.data:
            return []

        forecast_data = self.data.get("forecast", {})
        daily_forecasts = []

        # Get unique dates from forecast - only from today onwards (timezone-aware)
        today_str = dt_util.now().strftime("%Y-%m-%d")
        dates = sorted([d for d in forecast_data.keys() if d >= today_str])

        for date_str in dates[:FORECAST_DAYS]:  # Next FORECAST_DAYS days starting from today
            day_data = forecast_data.get(date_str, {})

            if not day_data:
                continue

            # Calculate daily aggregates
            temps = []
            humidities = []
            pressures = []
            winds = []
            rain_total = 0
            clouds = []
            solar_total = 0
            conditions = []

            for hour_str, hour_data in day_data.items():
                if hour_data.get("temperature") is not None:
                    temps.append(hour_data["temperature"])
                if hour_data.get("humidity") is not None:
                    humidities.append(hour_data["humidity"])
                if hour_data.get("pressure") is not None:
                    pressures.append(hour_data["pressure"])
                if hour_data.get("wind") is not None:
                    winds.append(hour_data["wind"])
                if hour_data.get("rain") is not None:
                    rain_total += hour_data["rain"]
                if hour_data.get("clouds") is not None:
                    clouds.append(hour_data["clouds"])
                if hour_data.get("solar_radiation_wm2") is not None:
                    solar_total += hour_data["solar_radiation_wm2"]

                conditions.append(self._get_condition(hour_data))

            if not temps:
                continue

            # Determine dominant condition
            condition_counts = {}
            for c in conditions:
                condition_counts[c] = condition_counts.get(c, 0) + 1
            dominant_condition = max(condition_counts, key=condition_counts.get)

            daily_forecasts.append({
                "datetime": f"{date_str}T12:00:00",
                "temperature": round(sum(temps) / len(temps), 1) if temps else None,
                "templow": round(min(temps), 1) if temps else None,
                "temphigh": round(max(temps), 1) if temps else None,
                "humidity": round(sum(humidities) / len(humidities), 1) if humidities else None,
                "pressure": round(sum(pressures) / len(pressures), 1) if pressures else None,
                "wind_speed": round(sum(winds) / len(winds), 1) if winds else None,
                "precipitation": round(rain_total, 1),
                "cloud_coverage": round(sum(clouds) / len(clouds), 1) if clouds else None,
                "condition": dominant_condition,
                "solar_radiation_total": round(solar_total, 1),
            })

        return daily_forecasts

    async def async_get_db_info(self) -> dict[str, Any]:
        """Get database status information."""
        return await self._db_reader.async_get_db_info()

    @property
    def db_path(self) -> str:
        """Return the path to the database file."""
        return self._db_reader.db_path

    def get_pv_forecast(self) -> dict[str, Any]:
        """Get PV forecast data for today, tomorrow, and day after tomorrow."""
        if not self._pv_forecast_data:
            return {
                "today": None,
                "tomorrow": None,
                "day_after_tomorrow": None,
            }

        return {
            "today": self._pv_forecast_data.get("today"),
            "today_source": self._pv_forecast_data.get("today_source"),
            "today_locked": self._pv_forecast_data.get("today_locked", False),
            "tomorrow": self._pv_forecast_data.get("tomorrow"),
            "tomorrow_date": self._pv_forecast_data.get("tomorrow_date"),
            "tomorrow_source": self._pv_forecast_data.get("tomorrow_source"),
            "day_after_tomorrow": self._pv_forecast_data.get("day_after_tomorrow"),
            "day_after_tomorrow_date": self._pv_forecast_data.get("day_after_tomorrow_date"),
            "day_after_tomorrow_source": self._pv_forecast_data.get("day_after_tomorrow_source"),
        }

    async def async_close(self) -> None:
        """Close database connection."""
        await self._db_reader.async_close()
