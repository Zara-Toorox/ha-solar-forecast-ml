"""Power Sources Data Collector for SFML Stats. @zara

Collects power flow data every few minutes for the Power Sources chart.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

from .const import (
    CONF_SENSOR_SOLAR_TO_HOUSE,
    CONF_SENSOR_BATTERY_TO_HOUSE,
    CONF_SENSOR_GRID_TO_HOUSE,
    CONF_SENSOR_HOME_CONSUMPTION,
    CONF_SENSOR_BATTERY_SOC,
)

_LOGGER = logging.getLogger(__name__)

# Collection interval in seconds (5 minutes)
COLLECTION_INTERVAL = 300

# Keep data for 7 days
MAX_DATA_AGE_DAYS = 7


class PowerSourcesCollector:
    """Collects power sources data periodically. @zara"""

    def __init__(self, hass: HomeAssistant, config: dict[str, Any], data_path: Path) -> None:
        """Initialize the collector. @zara"""
        self.hass = hass
        self.config = config
        self.data_path = data_path
        self.data_file = data_path / "power_sources_history.json"
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the data collection task. @zara"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._collection_loop())
        _LOGGER.info("Power Sources Collector started")

    async def stop(self) -> None:
        """Stop the data collection task. @zara"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        _LOGGER.info("Power Sources Collector stopped")

    async def _collection_loop(self) -> None:
        """Main collection loop. @zara"""
        while self._running:
            try:
                await self._collect_data()
            except Exception as e:
                _LOGGER.error("Error collecting power sources data: %s", e)

            # Wait for next collection
            await asyncio.sleep(COLLECTION_INTERVAL)

    async def _collect_data(self) -> None:
        """Collect current power values. @zara"""
        now = datetime.now(timezone.utc)

        # Get sensor values
        solar_to_house = self._get_sensor_value(CONF_SENSOR_SOLAR_TO_HOUSE)
        battery_to_house = self._get_sensor_value(CONF_SENSOR_BATTERY_TO_HOUSE)
        grid_to_house = self._get_sensor_value(CONF_SENSOR_GRID_TO_HOUSE)
        home_consumption = self._get_sensor_value(CONF_SENSOR_HOME_CONSUMPTION)
        battery_soc = self._get_sensor_value(CONF_SENSOR_BATTERY_SOC)

        # Create data point
        data_point = {
            "timestamp": now.isoformat(),
            "solar_to_house": solar_to_house,
            "battery_to_house": battery_to_house,
            "grid_to_house": grid_to_house,
            "home_consumption": home_consumption,
            "battery_soc": battery_soc,
        }

        # Load existing data
        data = await self._load_data()

        # Add new data point
        data["data_points"].append(data_point)

        # Clean old data
        cutoff = now - timedelta(days=MAX_DATA_AGE_DAYS)
        data["data_points"] = [
            dp for dp in data["data_points"]
            if datetime.fromisoformat(dp["timestamp"].replace('Z', '+00:00')) > cutoff
        ]

        # Update metadata
        data["last_updated"] = now.isoformat()
        data["points_count"] = len(data["data_points"])

        # Save data
        await self._save_data(data)

        _LOGGER.debug(
            "Collected power data: solar=%.1f, battery=%.1f, grid=%.1f, consumption=%.1f, soc=%s",
            solar_to_house or 0, battery_to_house or 0, grid_to_house or 0,
            home_consumption or 0, battery_soc
        )

    def _get_sensor_value(self, config_key: str) -> float | None:
        """Get sensor value from Home Assistant. @zara"""
        entity_id = self.config.get(config_key)
        if not entity_id:
            return None

        state = self.hass.states.get(entity_id)
        if not state or state.state in ('unknown', 'unavailable'):
            return None

        try:
            return float(state.state)
        except (ValueError, TypeError):
            return None

    async def _load_data(self) -> dict[str, Any]:
        """Load existing data from file. @zara"""
        if not self.data_file.exists():
            return {
                "version": 1,
                "created": datetime.now(timezone.utc).isoformat(),
                "last_updated": None,
                "points_count": 0,
                "data_points": []
            }

        try:
            async with aiofiles.open(self.data_file, 'r') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            _LOGGER.error("Error loading power sources data: %s", e)
            return {
                "version": 1,
                "created": datetime.now(timezone.utc).isoformat(),
                "last_updated": None,
                "points_count": 0,
                "data_points": []
            }

    async def _save_data(self, data: dict[str, Any]) -> None:
        """Save data to file. @zara"""
        self.data_path.mkdir(parents=True, exist_ok=True)

        try:
            async with aiofiles.open(self.data_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            _LOGGER.error("Error saving power sources data: %s", e)

    async def get_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get historical data for the specified number of hours. @zara"""
        data = await self._load_data()

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        filtered = [
            dp for dp in data.get("data_points", [])
            if datetime.fromisoformat(dp["timestamp"].replace('Z', '+00:00')) > cutoff
        ]

        return sorted(filtered, key=lambda x: x["timestamp"])
