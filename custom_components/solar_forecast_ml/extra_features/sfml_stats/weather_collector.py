# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Weather Data Collector for SFML Stats. @zara"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from collections import defaultdict

from homeassistant.core import HomeAssistant

from .readers.solar_reader import SolarDataReader
from .readers.weather_reader import WeatherDataReader

_LOGGER = logging.getLogger(__name__)


class WeatherDataCollector:
    """Collects weather data from Solar Forecast ML. @zara"""

    def __init__(self, hass: HomeAssistant, data_path: Path) -> None:
        """Initialize collector. @zara"""
        self.hass = hass
        self.data_path = data_path

        self.data_path.mkdir(parents=True, exist_ok=True)

    async def collect_daily_data(self) -> None:
        """Weather data is provided by Solar Forecast ML. @zara"""
        _LOGGER.debug("Weather data is provided by Solar Forecast ML")

    async def get_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get last N days of weather history from Solar Forecast ML. @zara"""
        sfml_data = await self._load_from_solar_forecast_ml()

        if not sfml_data:
            _LOGGER.warning(
                "No weather data available from Solar Forecast ML. "
                "Make sure Solar Forecast ML integration is installed and configured."
            )
            return []

        return sfml_data[-days:] if len(sfml_data) > days else sfml_data

    async def _load_from_solar_forecast_ml(self) -> list[dict[str, Any]]:
        """Load and convert weather data from Solar Forecast ML database. @zara"""
        try:
            config_path = Path(self.hass.config.path())
            weather_reader = WeatherDataReader(config_path)

            if not weather_reader.is_available:
                _LOGGER.debug("Solar Forecast ML database not found")
                return []

            hourly_weather = await weather_reader.async_get_hourly_weather()

            if not hourly_weather:
                _LOGGER.debug("No hourly weather data in database")
                return []

            daily_temps: dict[str, list[float]] = defaultdict(list)
            daily_humidity: dict[str, list[float]] = defaultdict(list)
            daily_wind: dict[str, list[float]] = defaultdict(list)
            daily_rain: dict[str, list[float]] = defaultdict(list)
            daily_radiation: dict[str, list[float]] = defaultdict(list)
            daily_clouds: dict[str, list[float]] = defaultdict(list)

            for weather in hourly_weather:
                date_str = weather.date.isoformat()

                if weather.temperature_c is not None:
                    daily_temps[date_str].append(weather.temperature_c)

                if weather.humidity_percent is not None:
                    daily_humidity[date_str].append(weather.humidity_percent)

                if weather.wind_speed_ms is not None:
                    daily_wind[date_str].append(weather.wind_speed_ms)

                if weather.precipitation_mm is not None:
                    daily_rain[date_str].append(weather.precipitation_mm)

                if weather.solar_radiation_wm2 is not None:
                    daily_radiation[date_str].append(weather.solar_radiation_wm2)

                if weather.cloud_cover_percent is not None:
                    daily_clouds[date_str].append(weather.cloud_cover_percent)

            solar_by_date: dict[str, float] = {}
            try:
                config_path = Path(self.hass.config.path())
                reader = SolarDataReader(config_path)
                summaries = await reader.async_get_daily_summaries()
                for summary in summaries:
                    date_key = summary.date.isoformat()
                    actual_kwh = summary.actual_total_kwh
                    if date_key and actual_kwh:
                        solar_by_date[date_key] = actual_kwh
            except Exception as err:
                _LOGGER.warning("Could not load solar summaries from database: %s", err)

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

                sun_hours = sum(1 for r in radiation_vals if r > 100) if radiation_vals else 0

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
                    "humidity": humidity_val,
                    "wind": wind_val,
                    "radiation": radiation_val,
                    "rain": rain_val,
                    "clouds": clouds_val,
                    "humidity_avg": humidity_val,
                    "wind_avg": wind_val,
                    "wind_max": wind_max_val,
                    "radiation_avg": radiation_val,
                    "rain_total": rain_val,
                    "sun_hours": sun_hours,
                    "solar_kwh": solar_by_date.get(date_str, 0),
                })

            _LOGGER.info(
                "Loaded %d days of weather history from Solar Forecast ML database",
                len(daily_data)
            )
            return daily_data

        except Exception as err:
            _LOGGER.error("Error loading Solar Forecast ML weather data from database: %s", err, exc_info=True)
            return []

    async def get_comparison_data(self, days: int = 7) -> dict[str, Any]:
        """Get IST vs KI comparison data for weather analytics. @zara"""
        ist_data = await self._load_from_solar_forecast_ml()

        ki_data = await self._load_ki_forecast_data()

        if not ist_data:
            _LOGGER.warning("No IST weather data available for comparison")
            return {"success": False, "error": "No IST data available"}

        ist_data = ist_data[-days:] if len(ist_data) > days else ist_data

        comparison = []
        for ist_day in ist_data:
            date_str = ist_day.get("date")
            ki_day = ki_data.get(date_str, {})

            comparison.append({
                "date": date_str,
                "temp_ist": ist_day.get("temp_avg", 0),
                "temp_ist_max": ist_day.get("temp_max", 0),
                "temp_ist_min": ist_day.get("temp_min", 0),
                "radiation_ist": ist_day.get("radiation", 0),
                "clouds_ist": ist_day.get("clouds", 0),
                "humidity_ist": ist_day.get("humidity", 0),
                "wind_ist": ist_day.get("wind", 0),
                "rain_ist": ist_day.get("rain", 0),
                "temp_ki": ki_day.get("temp_avg", 0),
                "temp_ki_max": ki_day.get("temp_max", 0),
                "temp_ki_min": ki_day.get("temp_min", 0),
                "radiation_ki": ki_day.get("radiation", 0),
                "clouds_ki": ki_day.get("clouds", 0),
                "humidity_ki": ki_day.get("humidity", 0),
                "wind_ki": ki_day.get("wind", 0),
                "rain_ki": ki_day.get("rain", 0),
            })

        if comparison:
            temp_errors = [abs(c["temp_ist"] - c["temp_ki"]) for c in comparison if c["temp_ki"] != 0]
            rad_errors = [abs(c["radiation_ist"] - c["radiation_ki"]) for c in comparison if c["radiation_ki"] != 0]

            avg_temp_error = sum(temp_errors) / len(temp_errors) if temp_errors else 0
            avg_rad_error = sum(rad_errors) / len(rad_errors) if rad_errors else 0

            temp_accuracy = max(0, 100 - avg_temp_error * 10)
            rad_accuracy = max(0, 100 - avg_rad_error / 5)
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
        """Load KI forecast data from database. @zara"""
        try:
            config_path = Path(self.hass.config.path())
            weather_reader = WeatherDataReader(config_path)

            if not weather_reader.is_available:
                _LOGGER.debug("Solar Forecast ML database not found")
                return {}

            hourly_forecast = await weather_reader.async_get_forecast_weather()

            if not hourly_forecast:
                _LOGGER.debug("No forecast data in database")
                return {}

            daily_forecasts: dict[str, dict[str, Any]] = defaultdict(lambda: {
                "temps": [],
                "radiations": [],
                "clouds": [],
                "humidities": [],
                "winds": [],
                "rains": []
            })

            for weather in hourly_forecast:
                date_str = weather.date.isoformat()
                day = daily_forecasts[date_str]

                if weather.temperature_c is not None:
                    day["temps"].append(weather.temperature_c)
                if weather.solar_radiation_wm2 is not None:
                    day["radiations"].append(weather.solar_radiation_wm2)
                if weather.cloud_cover_percent is not None:
                    day["clouds"].append(weather.cloud_cover_percent)
                if weather.humidity_percent is not None:
                    day["humidities"].append(weather.humidity_percent)
                if weather.wind_speed_ms is not None:
                    day["winds"].append(weather.wind_speed_ms)
                if weather.precipitation_mm is not None:
                    day["rains"].append(weather.precipitation_mm)

            result = {}
            for date_str, data in daily_forecasts.items():
                if data["temps"]:
                    result[date_str] = {
                        "temp_avg": round(sum(data["temps"]) / len(data["temps"]), 1),
                        "temp_max": round(max(data["temps"]), 1),
                        "temp_min": round(min(data["temps"]), 1),
                        "radiation": round(sum(data["radiations"]) / len(data["radiations"]), 1) if data["radiations"] else 0,
                        "clouds": round(sum(data["clouds"]) / len(data["clouds"]), 1) if data["clouds"] else 0,
                        "humidity": round(sum(data["humidities"]) / len(data["humidities"]), 1) if data["humidities"] else 0,
                        "wind": round(sum(data["winds"]) / len(data["winds"]), 1) if data["winds"] else 0,
                        "rain": round(sum(data["rains"]), 1) if data["rains"] else 0,
                    }

            _LOGGER.info("Loaded KI forecast data for %d days from database", len(result))
            return result

        except Exception as err:
            _LOGGER.error("Error loading KI forecast data from database: %s", err, exc_info=True)
            return {}

    async def get_statistics(self) -> dict[str, Any]:
        """Calculate weather statistics from Solar Forecast ML history. @zara"""
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

        total_sun_hours = sum(d.get("sun_hours", 0) for d in week_data)

        return {
            "avgTemp": round(sum(d.get("temp_avg", 0) for d in week_data) / len(week_data), 1) if week_data else 0,
            "maxTemp": round(max((d.get("temp_max", 0) for d in history_data), default=0), 1),
            "minTemp": round(min((d.get("temp_min", 0) for d in history_data), default=0), 1),
            "totalRain": round(sum(d.get("rain_total", 0) for d in month_data), 1),
            "avgWind": round(sum(d.get("wind_avg", 0) for d in history_data) / len(history_data), 1) if history_data else 0,
            "sunHours": total_sun_hours
        }
