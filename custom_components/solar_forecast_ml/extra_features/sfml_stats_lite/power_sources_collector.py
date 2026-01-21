"""Power Sources Data Collector for SFML Stats.

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
    CONF_SENSOR_SOLAR_POWER,
    CONF_SENSOR_SOLAR_TO_HOUSE,
    CONF_SENSOR_SOLAR_TO_BATTERY,
    CONF_SENSOR_BATTERY_TO_HOUSE,
    CONF_SENSOR_GRID_TO_HOUSE,
    CONF_SENSOR_HOME_CONSUMPTION,
    CONF_SENSOR_BATTERY_SOC,
    CONF_SENSOR_SOLAR_YIELD_DAILY,
)

_LOGGER = logging.getLogger(__name__)

# Collection interval in seconds (5 minutes)
COLLECTION_INTERVAL = 300

# Keep data for 7 days
MAX_DATA_AGE_DAYS = 7


class PowerSourcesCollector:
    """Collects power sources data periodically."""

    def __init__(self, hass: HomeAssistant, config: dict[str, Any], data_path: Path) -> None:
        """Initialize the collector."""
        self.hass = hass
        self.config = config
        self.data_path = data_path
        self.data_file = data_path / "power_sources_history.json"
        self.daily_stats_file = data_path / "energy_sources_daily_stats.json"
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the data collection task."""
        if self._running:
            return

        # Ensure data directory exists and initialize files
        await self._ensure_data_files()

        self._running = True
        self._task = asyncio.create_task(self._collection_loop())
        _LOGGER.info("Power Sources Collector started")

    async def _ensure_data_files(self) -> None:
        """Ensure data directory and files exist on first installation."""
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Initialize power_sources_history.json if not exists
        if not self.data_file.exists():
            initial_data = {
                "version": 2,
                "created": datetime.now(timezone.utc).isoformat(),
                "last_updated": None,
                "points_count": 0,
                "data_points": []
            }
            await self._save_json(self.data_file, initial_data)
            _LOGGER.info("Created power_sources_history.json")

        # Initialize energy_sources_daily_stats.json if not exists
        if not self.daily_stats_file.exists():
            initial_stats = {
                "version": 1,
                "created": datetime.now(timezone.utc).isoformat(),
                "last_updated": None,
                "days": {}
            }
            await self._save_json(self.daily_stats_file, initial_stats)
            _LOGGER.info("Created energy_sources_daily_stats.json")

    async def _save_json(self, filepath: Path, data: dict[str, Any]) -> None:
        """Save data to a JSON file."""
        try:
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            _LOGGER.error("Error saving %s: %s", filepath, e)

    async def stop(self) -> None:
        """Stop the data collection task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        _LOGGER.info("Power Sources Collector stopped")

    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._running:
            try:
                await self._collect_data()
            except Exception as e:
                _LOGGER.error("Error collecting power sources data: %s", e)

            # Wait for next collection
            await asyncio.sleep(COLLECTION_INTERVAL)

    async def _collect_data(self) -> None:
        """Collect current power values."""
        now = datetime.now(timezone.utc)

        # Get sensor values
        solar_power = self._get_sensor_value(CONF_SENSOR_SOLAR_POWER)
        solar_to_house = self._get_sensor_value(CONF_SENSOR_SOLAR_TO_HOUSE)
        solar_to_battery = self._get_sensor_value(CONF_SENSOR_SOLAR_TO_BATTERY)
        battery_to_house = self._get_sensor_value(CONF_SENSOR_BATTERY_TO_HOUSE)
        grid_to_house = self._get_sensor_value(CONF_SENSOR_GRID_TO_HOUSE)
        home_consumption = self._get_sensor_value(CONF_SENSOR_HOME_CONSUMPTION)
        battery_soc = self._get_sensor_value(CONF_SENSOR_BATTERY_SOC)

        # Solar kann NIEMALS negativ sein - korrigiere negative Werte
        if solar_power is not None and solar_power < 0:
            solar_power = 0.0
        if solar_to_house is not None and solar_to_house < 0:
            solar_to_house = 0.0
        if solar_to_battery is not None and solar_to_battery < 0:
            solar_to_battery = 0.0

        # Create data point
        data_point = {
            "timestamp": now.isoformat(),
            "solar_power": solar_power,
            "solar_to_house": solar_to_house,
            "solar_to_battery": solar_to_battery,
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
            "Collected power data: solar_power=%.1f, solar_to_house=%.1f, solar_to_battery=%.1f, battery=%.1f, grid=%.1f, consumption=%.1f, soc=%s",
            solar_power or 0, solar_to_house or 0, solar_to_battery or 0, battery_to_house or 0, grid_to_house or 0,
            home_consumption or 0, battery_soc
        )

        # Update daily stats
        await self._update_daily_stats(now, data_point)

    def _get_sensor_value(self, config_key: str) -> float | None:
        """Get sensor value from Home Assistant."""
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
        """Load existing data from file."""
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
        """Save data to file."""
        self.data_path.mkdir(parents=True, exist_ok=True)

        try:
            async with aiofiles.open(self.data_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            _LOGGER.error("Error saving power sources data: %s", e)

    async def get_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get historical data for the specified number of hours."""
        data = await self._load_data()

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        filtered = [
            dp for dp in data.get("data_points", [])
            if datetime.fromisoformat(dp["timestamp"].replace('Z', '+00:00')) > cutoff
        ]

        return sorted(filtered, key=lambda x: x["timestamp"])

    async def _update_daily_stats(self, now: datetime, data_point: dict[str, Any]) -> None:
        """Update daily statistics with current data point."""
        # Get today's date string (local time)
        local_now = now.astimezone()
        today_str = local_now.strftime("%Y-%m-%d")

        # Load daily stats
        daily_stats = await self._load_daily_stats()

        # Initialize today if not exists
        if today_str not in daily_stats["days"]:
            daily_stats["days"][today_str] = self._create_empty_day_stats(today_str)

        today = daily_stats["days"][today_str]

        # Convert W to kWh for 5-minute interval: W * (5/60) / 1000
        interval_hours = COLLECTION_INTERVAL / 3600  # 5 minutes = 0.0833 hours

        # Accumulate energy values
        if data_point.get("solar_to_house") is not None:
            today["solar_to_house_kwh"] += (data_point["solar_to_house"] * interval_hours) / 1000
        if data_point.get("solar_to_battery") is not None:
            today["solar_to_battery_kwh"] += (data_point["solar_to_battery"] * interval_hours) / 1000
        if data_point.get("battery_to_house") is not None:
            today["battery_to_house_kwh"] += (data_point["battery_to_house"] * interval_hours) / 1000
        if data_point.get("grid_to_house") is not None:
            today["grid_to_house_kwh"] += (data_point["grid_to_house"] * interval_hours) / 1000
        if data_point.get("home_consumption") is not None:
            today["consumption_kwh"] += (data_point["home_consumption"] * interval_hours) / 1000

        # Calculate solar total (sum of solar_to_house and solar_to_battery)
        today["solar_total_kwh"] = today["solar_to_house_kwh"] + today["solar_to_battery_kwh"]

        # Also try to get the actual solar_yield_daily from sensor for comparison
        solar_yield_daily = self._get_sensor_value(CONF_SENSOR_SOLAR_YIELD_DAILY)
        if solar_yield_daily is not None:
            today["solar_yield_sensor_kwh"] = solar_yield_daily

        # Calculate autarky and self-consumption
        if today["consumption_kwh"] > 0:
            self_produced = today["solar_to_house_kwh"] + today["battery_to_house_kwh"]
            today["autarky_percent"] = min(100, (self_produced / today["consumption_kwh"]) * 100)

        total_produced = today["solar_total_kwh"]
        if total_produced > 0:
            today["self_consumption_percent"] = min(100, (today["consumption_kwh"] / total_produced) * 100)

        # Track battery SOC statistics
        if data_point.get("battery_soc") is not None:
            soc = data_point["battery_soc"]
            # Initialize SOC fields if missing (for existing data)
            if "soc_readings_count" not in today:
                today["soc_readings_count"] = 0
                today["soc_readings_sum"] = 0.0
                today["min_soc"] = 100.0
                today["max_soc"] = 0.0
                today["avg_soc"] = 0.0

            today["soc_readings_count"] += 1
            today["soc_readings_sum"] += soc
            today["avg_soc"] = today["soc_readings_sum"] / today["soc_readings_count"]
            today["min_soc"] = min(today.get("min_soc", 100.0), soc)
            today["max_soc"] = max(today.get("max_soc", 0.0), soc)

        # Track peak battery power (absolute value - both charging and discharging)
        if data_point.get("battery_to_house") is not None:
            battery_power = abs(data_point["battery_to_house"])
            if "peak_battery_power_w" not in today:
                today["peak_battery_power_w"] = 0.0
            today["peak_battery_power_w"] = max(today.get("peak_battery_power_w", 0.0), battery_power)

        today["last_updated"] = now.isoformat()
        today["data_points_count"] += 1

        # Update metadata
        daily_stats["last_updated"] = now.isoformat()

        # Save daily stats
        await self._save_json(self.daily_stats_file, daily_stats)

    def _create_empty_day_stats(self, date_str: str) -> dict[str, Any]:
        """Create empty statistics structure for a new day."""
        return {
            "date": date_str,
            "solar_total_kwh": 0.0,
            "solar_to_house_kwh": 0.0,
            "solar_to_battery_kwh": 0.0,
            "battery_to_house_kwh": 0.0,
            "grid_to_house_kwh": 0.0,
            "consumption_kwh": 0.0,
            "solar_yield_sensor_kwh": None,
            "autarky_percent": 0.0,
            "self_consumption_percent": 0.0,
            "avg_soc": 0.0,
            "min_soc": 100.0,
            "max_soc": 0.0,
            "soc_readings_count": 0,
            "soc_readings_sum": 0.0,
            "peak_battery_power_w": 0.0,
            "data_points_count": 0,
            "last_updated": None,
        }

    async def _load_daily_stats(self) -> dict[str, Any]:
        """Load daily statistics from file."""
        if not self.daily_stats_file.exists():
            return {
                "version": 1,
                "created": datetime.now(timezone.utc).isoformat(),
                "last_updated": None,
                "days": {}
            }

        try:
            async with aiofiles.open(self.daily_stats_file, 'r') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            _LOGGER.error("Error loading daily stats: %s", e)
            return {
                "version": 1,
                "created": datetime.now(timezone.utc).isoformat(),
                "last_updated": None,
                "days": {}
            }

    async def get_daily_stats(self, days: int = 7) -> dict[str, Any]:
        """Get daily statistics for the specified number of days."""
        daily_stats = await self._load_daily_stats()

        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        filtered_days = {
            date: stats for date, stats in daily_stats.get("days", {}).items()
            if date >= cutoff
        }

        return {
            "version": daily_stats.get("version", 1),
            "last_updated": daily_stats.get("last_updated"),
            "days": filtered_days
        }
