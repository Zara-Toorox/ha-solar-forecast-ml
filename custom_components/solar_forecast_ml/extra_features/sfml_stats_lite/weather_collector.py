"""Weather Data Collector for SFML Stats.

Lädt Wetterdaten aus Solar Forecast ML's hourly_weather_actual.json.
Diese Daten werden bereits von Solar Forecast ML gesammelt und gespeichert.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from collections import defaultdict

from homeassistant.core import HomeAssistant

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

# Solar Forecast ML weather data file
SOLAR_FORECAST_ML_WEATHER = "solar_forecast_ml/stats/hourly_weather_actual.json"


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

    def _read_json_file(self, file_path: Path) -> dict[str, Any] | None:
        """Read JSON file synchronously (runs in executor)."""
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def _load_from_solar_forecast_ml(self) -> list[dict[str, Any]]:
        """Load and convert weather data from Solar Forecast ML hourly_weather_actual.json.

        Converts the hourly data format to daily aggregates.
        """
        sfml_path = Path(self.hass.config.path()) / SOLAR_FORECAST_ML_WEATHER

        try:
            sfml_data = await self.hass.async_add_executor_job(
                self._read_json_file, sfml_path
            )

            if sfml_data is None:
                _LOGGER.debug(
                    "Solar Forecast ML weather file not found at %s",
                    sfml_path
                )
                return []

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

                    # Wind
                    wind = values.get("wind_speed_kmh")
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

                # Calculate sun hours: count hours where radiation > 100 W/m²
                sun_hours = sum(1 for r in radiation_vals if r > 100) if radiation_vals else 0

                daily_data.append({
                    "date": date_str,
                    "temp_avg": round(sum(temps) / len(temps), 1),
                    "temp_max": round(max(temps), 1),
                    "temp_min": round(min(temps), 1),
                    "humidity_avg": round(sum(humidity_vals) / len(humidity_vals), 1) if humidity_vals else 0,
                    "wind_avg": round(sum(wind_vals) / len(wind_vals), 1) if wind_vals else 0,
                    "wind_max": round(max(wind_vals), 1) if wind_vals else 0,
                    "radiation_avg": round(sum(radiation_vals) / len(radiation_vals), 1) if radiation_vals else 0,
                    "sun_hours": sun_hours,
                    "rain_total": round(sum(rain_vals), 1) if rain_vals else 0,
                    "rain": round(sum(rain_vals), 1) if rain_vals else 0,
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
            "avgTemp": round(sum(d.get("temp_avg", 0) for d in week_data) / len(week_data), 1),
            "maxTemp": round(max((d.get("temp_max", 0) for d in history_data), default=0), 1),
            "minTemp": round(min((d.get("temp_min", 0) for d in history_data), default=0), 1),
            "totalRain": round(sum(d.get("rain_total", 0) for d in month_data), 1),
            "avgWind": round(sum(d.get("wind_avg", 0) for d in history_data) / len(history_data), 1) if history_data else 0,
            "sunHours": total_sun_hours
        }
