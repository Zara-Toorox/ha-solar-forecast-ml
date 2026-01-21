"""Daily energy aggregator for SFML Stats.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import aiofiles

from homeassistant.core import HomeAssistant

from ..const import (
    DOMAIN,
    SFML_STATS_DATA,
    DAILY_ENERGY_HISTORY,
    CONF_SENSOR_SOLAR_YIELD_DAILY,
    CONF_SENSOR_GRID_IMPORT_DAILY,
    CONF_SENSOR_BATTERY_CHARGE_SOLAR_DAILY,
    CONF_SENSOR_BATTERY_CHARGE_GRID_DAILY,
    CONF_SENSOR_HOUSE_TO_GRID,
    CONF_SENSOR_PRICE_TOTAL,
    CONF_SENSOR_BATTERY_TO_HOUSE,
    CONF_SENSOR_SOLAR_TO_HOUSE,
    CONF_SENSOR_SOLAR_TO_BATTERY,
    CONF_SENSOR_GRID_TO_HOUSE,
    CONF_SENSOR_HOME_CONSUMPTION,
    CONF_BILLING_PRICE_MODE,
    CONF_BILLING_FIXED_PRICE,
    PRICE_MODE_FIXED,
    DEFAULT_BILLING_FIXED_PRICE,
)

_LOGGER = logging.getLogger(__name__)


class DailyEnergyAggregator:
    """Aggregate and store daily energy values."""

    def __init__(self, hass: HomeAssistant, config_path: Path) -> None:
        """Initialize the aggregator."""
        self._hass = hass
        self._config_path = config_path
        self._data_path = config_path / SFML_STATS_DATA
        self._history_file = self._data_path / DAILY_ENERGY_HISTORY

    def _get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        entries = self._hass.data.get(DOMAIN, {})
        for entry_id, entry_data in entries.items():
            if isinstance(entry_data, dict) and "config" in entry_data:
                return entry_data["config"]

        config_entries = self._hass.config_entries.async_entries(DOMAIN)
        if config_entries:
            return dict(config_entries[0].data)
        return {}

    def _get_sensor_value(self, entity_id: str | None) -> float | None:
        """Read current value from a sensor."""
        if not entity_id:
            return None

        state = self._hass.states.get(entity_id)
        if state is None or state.state in ("unknown", "unavailable"):
            return None

        try:
            return float(state.state)
        except (ValueError, TypeError):
            return None

    async def _load_history(self) -> dict[str, Any]:
        """Load existing history file."""
        if not self._history_file.exists():
            return {"days": {}, "last_updated": None}

        try:
            async with aiofiles.open(self._history_file, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as err:
            _LOGGER.error("Error loading history: %s", err)
            return {"days": {}, "last_updated": None}

    async def _save_history(self, history: dict[str, Any]) -> bool:
        """Save history file."""
        self._data_path.mkdir(parents=True, exist_ok=True)

        try:
            async with aiofiles.open(self._history_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(history, indent=2, ensure_ascii=False))
            return True
        except Exception as err:
            _LOGGER.error("Error saving history: %s", err)
            return False

    async def async_aggregate_daily(self) -> bool:
        """Aggregate daily values and save them."""
        config = self._get_config()
        today_str = date.today().isoformat()

        _LOGGER.info("Starting daily energy aggregation for %s", today_str)

        daily_data = {
            "solar_yield_kwh": self._get_sensor_value(
                config.get(CONF_SENSOR_SOLAR_YIELD_DAILY)
            ),
            "grid_import_kwh": self._get_sensor_value(
                config.get(CONF_SENSOR_GRID_IMPORT_DAILY)
            ),
            "battery_charge_solar_kwh": self._get_sensor_value(
                config.get(CONF_SENSOR_BATTERY_CHARGE_SOLAR_DAILY)
            ),
            "battery_charge_grid_kwh": self._get_sensor_value(
                config.get(CONF_SENSOR_BATTERY_CHARGE_GRID_DAILY)
            ),
            "grid_export_kwh": self._get_sensor_value(
                config.get(CONF_SENSOR_HOUSE_TO_GRID)
            ),
            "price_ct_kwh": self._get_sensor_value(
                config.get(CONF_SENSOR_PRICE_TOTAL)
            ),
        }

        solar_yield = daily_data["solar_yield_kwh"] or 0
        battery_charge_solar = daily_data["battery_charge_solar_kwh"] or 0

        daily_data["solar_to_house_kwh"] = max(0, solar_yield - battery_charge_solar)
        daily_data["timestamp"] = datetime.now().isoformat()

        # Get data from energy_sources_daily_stats for battery_to_house and other values
        await self._merge_from_daily_stats(today_str, daily_data)

        price_mode = config.get(CONF_BILLING_PRICE_MODE, "dynamic")
        fixed_price = config.get(CONF_BILLING_FIXED_PRICE, DEFAULT_BILLING_FIXED_PRICE)
        daily_data["price_mode"] = price_mode
        if price_mode == PRICE_MODE_FIXED:
            daily_data["fixed_price_ct"] = fixed_price

        history = await self._load_history()
        history["days"][today_str] = daily_data
        history["last_updated"] = datetime.now().isoformat()

        success = await self._save_history(history)

        if success:
            _LOGGER.info(
                "Daily aggregation saved: Solar=%.2f kWh, Grid import=%.2f kWh, Battery discharged=%.2f kWh",
                solar_yield,
                daily_data["grid_import_kwh"] or 0,
                daily_data.get("battery_to_house_kwh") or 0
            )

        return success

    async def _merge_from_daily_stats(self, today_str: str, daily_data: dict[str, Any]) -> None:
        """Merge data from energy_sources_daily_stats.json."""
        daily_stats_file = self._data_path / "energy_sources_daily_stats.json"

        if not daily_stats_file.exists():
            return

        try:
            async with aiofiles.open(daily_stats_file, "r", encoding="utf-8") as f:
                content = await f.read()
                stats = json.loads(content)

            today_stats = stats.get("days", {}).get(today_str, {})

            # Add battery_to_house_kwh from power sources collector
            if today_stats.get("battery_to_house_kwh") is not None:
                daily_data["battery_to_house_kwh"] = today_stats["battery_to_house_kwh"]

            # Add consumption_kwh if available
            if today_stats.get("consumption_kwh") is not None:
                daily_data["home_consumption_kwh"] = today_stats["consumption_kwh"]

            # Add autarky and self-consumption percentages
            if today_stats.get("autarky_percent") is not None:
                daily_data["autarky_percent"] = today_stats["autarky_percent"]
            if today_stats.get("self_consumption_percent") is not None:
                daily_data["self_consumption_percent"] = today_stats["self_consumption_percent"]

            # Add battery SOC statistics
            if today_stats.get("avg_soc") is not None:
                daily_data["avg_soc"] = today_stats["avg_soc"]
            if today_stats.get("min_soc") is not None:
                daily_data["min_soc"] = today_stats["min_soc"]
            if today_stats.get("max_soc") is not None:
                daily_data["max_soc"] = today_stats["max_soc"]

            # Add peak battery power
            if today_stats.get("peak_battery_power_w") is not None:
                daily_data["peak_battery_power_w"] = today_stats["peak_battery_power_w"]

        except Exception as err:
            _LOGGER.warning("Error merging from daily stats: %s", err)

    async def async_get_billing_period_data(
        self,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        """Get aggregated data for a billing period."""
        history = await self._load_history()

        result = {
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "days_with_data": 0,
            "days_in_period": (end_date - start_date).days + 1,
            "total_solar_yield_kwh": 0,
            "total_grid_import_kwh": 0,
            "total_grid_export_kwh": 0,
            "total_battery_charge_solar_kwh": 0,
            "total_battery_charge_grid_kwh": 0,
            "total_solar_to_house_kwh": 0,
            "daily_data": [],
        }

        current = start_date
        while current <= end_date:
            day_str = current.isoformat()
            if day_str in history.get("days", {}):
                day_data = history["days"][day_str]
                result["days_with_data"] += 1

                result["total_solar_yield_kwh"] += day_data.get("solar_yield_kwh") or 0
                result["total_grid_import_kwh"] += day_data.get("grid_import_kwh") or 0
                result["total_grid_export_kwh"] += day_data.get("grid_export_kwh") or 0
                result["total_battery_charge_solar_kwh"] += day_data.get("battery_charge_solar_kwh") or 0
                result["total_battery_charge_grid_kwh"] += day_data.get("battery_charge_grid_kwh") or 0
                result["total_solar_to_house_kwh"] += day_data.get("solar_to_house_kwh") or 0

                result["daily_data"].append({
                    "date": day_str,
                    **day_data
                })

            current = current + timedelta(days=1)

        return result
