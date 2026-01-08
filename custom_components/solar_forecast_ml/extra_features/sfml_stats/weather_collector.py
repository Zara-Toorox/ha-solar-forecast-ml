# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""Weather Data Collector for SFML Stats.

Lädt Wetterdaten aus Solar Forecast ML's hourly_weather_actual.json.
Diese Daten werden bereits von Solar Forecast ML gesammelt und gespeichert.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from collections import defaultdict

import aiofiles

from homeassistant.core import HomeAssistant

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

# Solar Forecast ML weather data files
SOLAR_FORECAST_ML_WEATHER = "solar_forecast_ml/stats/hourly_weather_actual.json"
SOLAR_FORECAST_ML_WEATHER_FORECAST = "solar_forecast_ml/stats/weather_forecast_corrected.json"


class WeatherDataCollector:
    """Lädt Wetterdaten aus Solar Forecast ML."""

    def __init__(self, hass: HomeAssistant, data_path: Path) -> None:
        """Initialize collector."""
        self.hass = hass
        self.data_path = data_path

        # Ensure data directory exists
        self.data_path.mkdir(parents=True, exist_ok=True)

    async def collect_daily_data(self) -> None:
        """Wetterdaten werden von Solar Forecast ML gesammelt - nichts zu tun."""
        _LOGGER.debug("Weather data is provided by Solar Forecast ML")

    async def get_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get last N days of weather history from Solar Forecast ML.

        Lädt die Wetterdaten aus der hourly_weather_actual.json Datei,
        die von Solar Forecast ML gepflegt wird.
        """
        sfml_data = await self._load_from_solar_forecast_ml()

        if not sfml_data:
            _LOGGER.warning(
                "No weather data available from Solar Forecast ML. "
                "Make sure Solar Forecast ML integration is installed and configured."
            )
            return []

        # Return last N days
        return sfml_data[-days:] if len(sfml_data) > days else sfml_data

    async def _load_from_solar_forecast_ml(self) -> list[dict[str, Any]]:
        """Load and convert weather data from Solar Forecast ML hourly_weather_actual.json.

        Converts the hourly data format to daily aggregates.
        """
        sfml_path = Path(self.hass.config.path()) / SOLAR_FORECAST_ML_WEATHER

        if not sfml_path.exists():
            _LOGGER.debug(
                "Solar Forecast ML weather file not found at %s",
                sfml_path
            )
            return []

        try:
            async with aiofiles.open(sfml_path, "r", encoding="utf-8") as f:
                content = await f.read()
                sfml_data = json.loads(content)

            hourly_data = sfml_data.get("hourly_data", {})

            if not hourly_data:
                _LOGGER.debug("No hourly_data in Solar Forecast ML weather file")
                return []

            # Group hourly data by date and aggregate
            daily_temps: dict[str, list[float]] = defaultdict(list)
            daily_humidity: dict[str, list[float]] = defaultdict(list)
            daily_wind: dict[str, list[float]] = defaultdict(list)
            daily_rain: dict[str, list[float]] = defaultdict(list)
            daily_radiation: dict[str, list[float]] = defaultdict(list)
            daily_clouds: dict[str, list[float]] = defaultdict(list)

            for date_str, hours in hourly_data.items():
                if not isinstance(hours, dict):
                    continue

                for hour_str, values in hours.items():
                    if not isinstance(values, dict):
                        continue

                    # Temperature
                    temp = values.get("temperature_c")
                    if temp is not None:
                        try:
                            daily_temps[date_str].append(float(temp))
                        except (ValueError, TypeError):
                            pass

                    # Humidity
                    humidity = values.get("humidity_percent")
                    if humidity is not None:
                        try:
                            daily_humidity[date_str].append(float(humidity))
                        except (ValueError, TypeError):
                            pass

                    # Wind (stored as m/s in hourly_weather_actual.json)
                    wind = values.get("wind_speed_ms")
                    if wind is not None:
                        try:
                            daily_wind[date_str].append(float(wind))
                        except (ValueError, TypeError):
                            pass

                    # Rain
                    rain = values.get("precipitation_mm")
                    if rain is not None:
                        try:
                            daily_rain[date_str].append(float(rain))
                        except (ValueError, TypeError):
                            pass

                    # Solar radiation
                    radiation = values.get("solar_radiation_wm2")
                    if radiation is not None:
                        try:
                            daily_radiation[date_str].append(float(radiation))
                        except (ValueError, TypeError):
                            pass

                    # Cloud cover (only available during daylight hours)
                    clouds = values.get("cloud_cover_percent")
                    if clouds is not None:
                        try:
                            daily_clouds[date_str].append(float(clouds))
                        except (ValueError, TypeError):
                            pass

            # Load solar production data from daily_summaries.json for correlation
            solar_by_date: dict[str, float] = {}
            summaries_path = Path(self.hass.config.path()) / "solar_forecast_ml" / "stats" / "daily_summaries.json"
            if summaries_path.exists():
                try:
                    async with aiofiles.open(summaries_path, "r", encoding="utf-8") as f:
                        summaries_content = await f.read()
                        summaries_data = json.loads(summaries_content)
                    for summary in summaries_data.get("summaries", []):
                        date_key = summary.get("date")
                        actual_kwh = summary.get("overall", {}).get("actual_total_kwh", 0)
                        if date_key and actual_kwh:
                            solar_by_date[date_key] = actual_kwh
                except Exception as err:
                    _LOGGER.warning("Could not load solar summaries: %s", err)

            # Convert to daily data format
            daily_data = []
            for date_str in sorted(daily_temps.keys()):
                temps = daily_temps[date_str]
                if not temps:
                    continue

                humidity_vals = daily_humidity.get(date_str, [])
                wind_vals = daily_wind.get(date_str, [])
                rain_vals = daily_rain.get(date_str, [])
                radiation_vals = daily_radiation.get(date_str, [])
                cloud_vals = daily_clouds.get(date_str, [])

                # Calculate sun hours: count hours where radiation > 100 W/m²
                sun_hours = sum(1 for r in radiation_vals if r > 100) if radiation_vals else 0

                # Calculate values (with sanity limits)
                humidity_val = min(100.0, round(sum(humidity_vals) / len(humidity_vals), 1)) if humidity_vals else 0
                wind_val = round(sum(wind_vals) / len(wind_vals), 1) if wind_vals else 0
                wind_max_val = round(max(wind_vals), 1) if wind_vals else 0
                radiation_val = max(0.0, round(sum(radiation_vals) / len(radiation_vals), 1)) if radiation_vals else 0
                rain_val = max(0.0, round(sum(rain_vals), 1)) if rain_vals else 0
                clouds_val = min(100.0, round(sum(cloud_vals) / len(cloud_vals), 1)) if cloud_vals else 0

                daily_data.append({
                    "date": date_str,
                    "temp_avg": round(sum(temps) / len(temps), 1),
                    "temp_max": round(max(temps), 1),
                    "temp_min": round(min(temps), 1),
                    # Primary field names (used by frontend)
                    "humidity": humidity_val,
                    "wind": wind_val,
                    "radiation": radiation_val,
                    "rain": rain_val,
                    "clouds": clouds_val,
                    # Aliases for chart compatibility
                    "humidity_avg": humidity_val,
                    "wind_avg": wind_val,
                    "wind_max": wind_max_val,
                    "radiation_avg": radiation_val,
                    "rain_total": rain_val,
                    "sun_hours": sun_hours,
                    "solar_kwh": solar_by_date.get(date_str, 0),
                })

            _LOGGER.info(
                "Loaded %d days of weather history from Solar Forecast ML",
                len(daily_data)
            )
            return daily_data

        except json.JSONDecodeError as err:
            _LOGGER.error("Error parsing Solar Forecast ML weather file: %s", err)
            return []
        except Exception as err:
            _LOGGER.error("Error loading Solar Forecast ML weather data: %s", err, exc_info=True)
            return []

    async def get_comparison_data(self, days: int = 7) -> dict[str, Any]:
        """Get IST vs KI comparison data for weather analytics.

        Returns actual (IST) weather data alongside KI forecast data for comparison.
        """
        # Load IST data (actual measurements)
        ist_data = await self._load_from_solar_forecast_ml()

        # Load KI forecast data
        ki_data = await self._load_ki_forecast_data()

        if not ist_data:
            _LOGGER.warning("No IST weather data available for comparison")
            return {"success": False, "error": "No IST data available"}

        # Get the last N days of IST data
        ist_data = ist_data[-days:] if len(ist_data) > days else ist_data

        # Match KI data to IST dates
        comparison = []
        for ist_day in ist_data:
            date_str = ist_day.get("date")
            ki_day = ki_data.get(date_str, {})

            comparison.append({
                "date": date_str,
                # IST (actual) values
                "temp_ist": ist_day.get("temp_avg", 0),
                "temp_ist_max": ist_day.get("temp_max", 0),
                "temp_ist_min": ist_day.get("temp_min", 0),
                "radiation_ist": ist_day.get("radiation", 0),
                "clouds_ist": ist_day.get("clouds", 0),  # Real cloud cover from sensors
                "humidity_ist": ist_day.get("humidity", 0),
                "wind_ist": ist_day.get("wind", 0),
                "rain_ist": ist_day.get("rain", 0),
                # KI (forecast) values
                "temp_ki": ki_day.get("temp_avg", 0),
                "temp_ki_max": ki_day.get("temp_max", 0),
                "temp_ki_min": ki_day.get("temp_min", 0),
                "radiation_ki": ki_day.get("radiation", 0),
                "clouds_ki": ki_day.get("clouds", 0),
                "humidity_ki": ki_day.get("humidity", 0),
                "wind_ki": ki_day.get("wind", 0),
                "rain_ki": ki_day.get("rain", 0),
            })

        # Calculate accuracy metrics
        if comparison:
            temp_errors = [abs(c["temp_ist"] - c["temp_ki"]) for c in comparison if c["temp_ki"] != 0]
            rad_errors = [abs(c["radiation_ist"] - c["radiation_ki"]) for c in comparison if c["radiation_ki"] != 0]

            avg_temp_error = sum(temp_errors) / len(temp_errors) if temp_errors else 0
            avg_rad_error = sum(rad_errors) / len(rad_errors) if rad_errors else 0

            # Accuracy as percentage (100% = perfect, lower = worse)
            temp_accuracy = max(0, 100 - avg_temp_error * 10)  # 10°C error = 0%
            rad_accuracy = max(0, 100 - avg_rad_error / 5)  # 500 W/m² error = 0%
        else:
            temp_accuracy = 0
            rad_accuracy = 0

        return {
            "success": True,
            "data": comparison,
            "stats": {
                "temp_accuracy": round(temp_accuracy, 1),
                "radiation_accuracy": round(rad_accuracy, 1),
                "overall_accuracy": round((temp_accuracy + rad_accuracy) / 2, 1),
                "days_compared": len(comparison)
            }
        }

    async def _load_ki_forecast_data(self) -> dict[str, dict[str, Any]]:
        """Load KI forecast data from weather_forecast_corrected.json.

        Returns a dict keyed by date with aggregated daily forecasts.
        """
        forecast_path = Path(self.hass.config.path()) / SOLAR_FORECAST_ML_WEATHER_FORECAST

        if not forecast_path.exists():
            _LOGGER.debug("KI forecast file not found at %s", forecast_path)
            return {}

        try:
            async with aiofiles.open(forecast_path, "r", encoding="utf-8") as f:
                content = await f.read()
                forecast_data = json.loads(content)

            hourly_forecast = forecast_data.get("forecast", {})

            if not hourly_forecast or not isinstance(hourly_forecast, dict):
                _LOGGER.debug("No forecast data in KI weather file")
                return {}

            # Aggregate hourly forecasts to daily
            daily_forecasts: dict[str, dict[str, Any]] = {}

            for date_str, hours in hourly_forecast.items():
                if not isinstance(hours, dict):
                    continue

                temps = []
                radiations = []
                clouds = []
                humidities = []
                winds = []
                rains = []

                for hour_str, values in hours.items():
                    if not isinstance(values, dict):
                        continue

                    if values.get("temperature") is not None:
                        temps.append(float(values["temperature"]))
                    if values.get("solar_radiation_wm2") is not None:
                        radiations.append(float(values["solar_radiation_wm2"]))
                    if values.get("clouds") is not None:
                        clouds.append(float(values["clouds"]))
                    if values.get("humidity") is not None:
                        humidities.append(float(values["humidity"]))
                    if values.get("wind") is not None:
                        winds.append(float(values["wind"]))
                    if values.get("rain") is not None:
                        rains.append(float(values["rain"]))

                if temps:
                    daily_forecasts[date_str] = {
                        "temp_avg": round(sum(temps) / len(temps), 1),
                        "temp_max": round(max(temps), 1),
                        "temp_min": round(min(temps), 1),
                        "radiation": round(sum(radiations) / len(radiations), 1) if radiations else 0,
                        "clouds": round(sum(clouds) / len(clouds), 1) if clouds else 0,
                        "humidity": round(sum(humidities) / len(humidities), 1) if humidities else 0,
                        "wind": round(sum(winds) / len(winds), 1) if winds else 0,
                        "rain": round(sum(rains), 1) if rains else 0,
                    }

            _LOGGER.info("Loaded KI forecast data for %d days", len(daily_forecasts))
            return daily_forecasts

        except json.JSONDecodeError as err:
            _LOGGER.error("Error parsing KI forecast file: %s", err)
            return {}
        except Exception as err:
            _LOGGER.error("Error loading KI forecast data: %s", err, exc_info=True)
            return {}

    async def get_statistics(self) -> dict[str, Any]:
        """Calculate weather statistics from Solar Forecast ML history."""
        history_data = await self.get_history(days=365)

        if not history_data:
            return {
                "avgTemp": 0,
                "maxTemp": 0,
                "minTemp": 0,
                "totalRain": 0,
                "avgWind": 0,
                "sunHours": 0
            }

        week_data = history_data[-7:] if len(history_data) >= 7 else history_data
        month_data = history_data[-30:] if len(history_data) >= 30 else history_data

        # Sum sun hours for the week
        total_sun_hours = sum(d.get("sun_hours", 0) for d in week_data)

        return {
            "avgTemp": round(sum(d.get("temp_avg", 0) for d in week_data) / len(week_data), 1) if week_data else 0,
            "maxTemp": round(max((d.get("temp_max", 0) for d in history_data), default=0), 1),
            "minTemp": round(min((d.get("temp_min", 0) for d in history_data), default=0), 1),
            "totalRain": round(sum(d.get("rain_total", 0) for d in month_data), 1),
            "avgWind": round(sum(d.get("wind_avg", 0) for d in history_data) / len(history_data), 1) if history_data else 0,
            "sunHours": total_sun_hours
        }
