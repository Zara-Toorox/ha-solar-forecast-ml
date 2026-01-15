# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML - ML Weather
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************
"""DataUpdateCoordinator for ML Weather."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    DEFAULT_SOURCE_PATH,
    DEFAULT_SCAN_INTERVAL,
    CONF_DATA_PATH,
    CONDITION_MAP,
    PV_FORECAST_SOURCE_PATH,
    FORECAST_HOURS,
    FORECAST_DAYS,
    RAIN_THRESHOLD_LIGHT,
    RAIN_THRESHOLD_MODERATE,
)
from .cache_manager import CacheManager

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
        self._source_path = entry.data.get(CONF_DATA_PATH, DEFAULT_SOURCE_PATH)
        self._raw_data: dict = {}
        self._pv_forecast_data: dict = {}
        self._consecutive_failures: int = 0
        self._base_interval = DEFAULT_SCAN_INTERVAL

        # Initialize cache manager
        self._cache_manager = CacheManager(hass, self._source_path)

        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=DEFAULT_SCAN_INTERVAL,
        )

    async def async_config_entry_first_refresh(self) -> None:
        """Initialize cache and perform first refresh."""
        await self._cache_manager.async_initialize()
        await super().async_config_entry_first_refresh()

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch data via cache manager with retry backoff."""
        try:
            # Update cache from source (if newer) and get data
            data = await self._cache_manager.async_update_cache()

            if data is None:
                self._handle_failure()
                raise UpdateFailed("No weather data available (source and cache empty)")

            self._raw_data = data

            # Load PV forecast data
            self._pv_forecast_data = await self.hass.async_add_executor_job(
                self._load_pv_forecast
            )

            # Success - reset failure counter and interval
            self._handle_success()

            return self._process_data(data)

        except UpdateFailed:
            raise
        except json.JSONDecodeError as err:
            self._handle_failure()
            raise UpdateFailed(f"Invalid JSON in weather data: {err}") from err
        except PermissionError as err:
            self._handle_failure()
            raise UpdateFailed(f"Permission denied accessing weather data: {err}") from err
        except OSError as err:
            self._handle_failure()
            raise UpdateFailed(f"OS error loading weather data: {err}") from err
        except Exception as err:
            self._handle_failure()
            raise UpdateFailed(f"Unexpected error loading weather data: {err}") from err

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

    def _load_pv_forecast(self) -> dict[str, Any]:
        """Load PV forecast data from Solar Forecast ML."""
        pv_path = Path(PV_FORECAST_SOURCE_PATH)
        if not pv_path.exists():
            _LOGGER.debug(f"PV forecast file not found: {PV_FORECAST_SOURCE_PATH}")
            return {}

        try:
            with open(pv_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            _LOGGER.warning(f"Error loading PV forecast: {err}")
            return {}

    def _process_data(self, raw_data: dict) -> dict[str, Any]:
        """Process raw data into usable format."""
        forecast_data = raw_data.get("forecast", {})

        # Get current weather (current hour) - use timezone-aware datetime
        now = dt_util.now()
        today_str = now.strftime("%Y-%m-%d")
        current_hour = str(now.hour)

        current = self._get_hour_data(forecast_data, today_str, current_hour)

        return {
            "current": current,
            "forecast": forecast_data,
            "version": raw_data.get("version", "unknown"),
            "metadata": raw_data.get("metadata", {}),
            "cache_info": None,  # Will be populated on demand
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
        now = dt_util.now()  # Use timezone-aware datetime

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

    async def async_get_cache_info(self) -> dict[str, Any]:
        """Get cache status information."""
        return await self._cache_manager.async_get_cache_info()

    @property
    def cache_path(self) -> str:
        """Return the path to the cache file."""
        return self._cache_manager.cache_path

    def get_pv_forecast(self) -> dict[str, Any]:
        """Get PV forecast data for today, tomorrow, and day after tomorrow."""
        if not self._pv_forecast_data:
            return {
                "today": None,
                "tomorrow": None,
                "day_after_tomorrow": None,
            }

        today_data = self._pv_forecast_data.get("today", {})

        # Today's forecast
        forecast_day = today_data.get("forecast_day", {})
        today_kwh = forecast_day.get("prediction_kwh")

        # Tomorrow's forecast
        forecast_tomorrow = today_data.get("forecast_tomorrow", {})
        tomorrow_kwh = forecast_tomorrow.get("prediction_kwh")

        # Day after tomorrow's forecast
        forecast_day_after = today_data.get("forecast_day_after_tomorrow", {})
        day_after_kwh = forecast_day_after.get("prediction_kwh")

        return {
            "today": today_kwh,
            "today_source": forecast_day.get("source"),
            "today_locked": forecast_day.get("locked", False),
            "tomorrow": tomorrow_kwh,
            "tomorrow_date": forecast_tomorrow.get("date"),
            "tomorrow_source": forecast_tomorrow.get("source"),
            "day_after_tomorrow": day_after_kwh,
            "day_after_tomorrow_date": forecast_day_after.get("date"),
            "day_after_tomorrow_source": forecast_day_after.get("source"),
        }
