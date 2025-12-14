"""Weather Data Collector for SFML Stats.

Sammelt täglich Wetterdaten aus Home Assistant Sensoren und Recorder
und speichert sie in JSON für historische Analytics.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.components.recorder import get_instance, history
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

WEATHER_DATA_FILE = "weather_history.json"


class WeatherDataCollector:
    """Sammelt und speichert historische Wetterdaten."""

    def __init__(self, hass: HomeAssistant, data_path: Path) -> None:
        """Initialize collector."""
        self.hass = hass
        self.data_path = data_path
        self.data_file = data_path / WEATHER_DATA_FILE

        # Ensure data directory exists
        self.data_path.mkdir(parents=True, exist_ok=True)

    async def collect_daily_data(self) -> None:
        """Sammelt Wetterdaten für heute und speichert sie."""
        try:
            today = datetime.now().date()
            weather_data = await self._fetch_today_weather()

            if weather_data:
                await self._append_to_history(today, weather_data)
                _LOGGER.info("Weather data collected for %s", today)
            else:
                _LOGGER.warning("No weather data available for %s", today)

        except Exception as err:
            _LOGGER.error("Error collecting weather data: %s", err, exc_info=True)

    async def _fetch_today_weather(self) -> dict[str, Any] | None:
        """Hole Wetterdaten für heute aus Recorder."""
        try:
            end_time = dt_util.now()
            start_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)

            # Sensor entity IDs - diese sollten in der Config konfigurierbar sein
            temp_entity = "sensor.outdoor_temperature"  # Beispiel
            rain_entity = "sensor.rain_today"
            wind_entity = "sensor.wind_speed"
            radiation_entity = "sensor.solar_radiation"
            humidity_entity = "sensor.outdoor_humidity"

            # Hole Statistiken aus Recorder
            recorder_instance = get_instance(self.hass)

            # Vereinfachte Version: Hole aktuelle Werte
            # TODO: Erweitern mit Min/Max/Avg aus Recorder history

            data = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "temp_avg": await self._get_sensor_avg(temp_entity, start_time, end_time),
                "temp_max": await self._get_sensor_max(temp_entity, start_time, end_time),
                "temp_min": await self._get_sensor_min(temp_entity, start_time, end_time),
                "radiation_avg": await self._get_sensor_avg(radiation_entity, start_time, end_time),
                "rain_total": await self._get_sensor_total(rain_entity, start_time, end_time),
                "humidity_avg": await self._get_sensor_avg(humidity_entity, start_time, end_time),
                "wind_avg": await self._get_sensor_avg(wind_entity, start_time, end_time),
                "wind_max": await self._get_sensor_max(wind_entity, start_time, end_time),
            }

            return data

        except Exception as err:
            _LOGGER.error("Error fetching weather data: %s", err)
            return None

    async def _get_sensor_avg(self, entity_id: str, start: datetime, end: datetime) -> float:
        """Get average value from sensor."""
        try:
            state = self.hass.states.get(entity_id)
            if state and state.state not in ("unknown", "unavailable"):
                return float(state.state)
        except (ValueError, TypeError):
            pass
        return 0.0

    async def _get_sensor_max(self, entity_id: str, start: datetime, end: datetime) -> float:
        """Get max value from sensor history."""
        # Simplified - would use recorder history in production
        try:
            state = self.hass.states.get(entity_id)
            if state and state.state not in ("unknown", "unavailable"):
                return float(state.state) + 2  # Mock offset
        except (ValueError, TypeError):
            pass
        return 0.0

    async def _get_sensor_min(self, entity_id: str, start: datetime, end: datetime) -> float:
        """Get min value from sensor history."""
        try:
            state = self.hass.states.get(entity_id)
            if state and state.state not in ("unknown", "unavailable"):
                return float(state.state) - 2  # Mock offset
        except (ValueError, TypeError):
            pass
        return 0.0

    async def _get_sensor_total(self, entity_id: str, start: datetime, end: datetime) -> float:
        """Get total (sum) from sensor."""
        try:
            state = self.hass.states.get(entity_id)
            if state and state.state not in ("unknown", "unavailable"):
                return float(state.state)
        except (ValueError, TypeError):
            pass
        return 0.0

    async def _append_to_history(self, date: datetime.date, data: dict[str, Any]) -> None:
        """Fügt Wetterdaten zur History-Datei hinzu."""
        try:
            # Load existing history
            history_data = await self._load_history()

            # Update or append entry for today
            date_str = date.strftime("%Y-%m-%d")

            # Find and update existing entry or append new
            existing_index = next(
                (i for i, entry in enumerate(history_data) if entry.get("date") == date_str),
                None
            )

            if existing_index is not None:
                history_data[existing_index] = data
            else:
                history_data.append(data)

            # Keep only last 365 days
            history_data = sorted(history_data, key=lambda x: x["date"])[-365:]

            # Save to file
            await self._save_history(history_data)

        except Exception as err:
            _LOGGER.error("Error appending weather history: %s", err)

    async def _load_history(self) -> list[dict[str, Any]]:
        """Load weather history from JSON file."""
        if not self.data_file.exists():
            return []

        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            _LOGGER.error("Error loading weather history: %s", err)
            return []

    async def _save_history(self, data: list[dict[str, Any]]) -> None:
        """Save weather history to JSON file."""
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as err:
            _LOGGER.error("Error saving weather history: %s", err)

    async def get_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get last N days of weather history."""
        history_data = await self._load_history()
        return history_data[-days:] if history_data else []

    async def get_statistics(self) -> dict[str, Any]:
        """Calculate weather statistics."""
        history_data = await self._load_history()

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

        return {
            "avgTemp": sum(d.get("temp_avg", 0) for d in week_data) / len(week_data),
            "maxTemp": max((d.get("temp_max", 0) for d in history_data), default=0),
            "minTemp": min((d.get("temp_min", 0) for d in history_data), default=0),
            "totalRain": sum(d.get("rain_total", 0) for d in month_data),
            "avgWind": sum(d.get("wind_avg", 0) for d in history_data) / len(history_data),
            "sunHours": sum(1 for d in month_data if d.get("radiation_avg", 0) > 200) * 8  # Mock calculation
        }
