"""Sensor Platform for Grid Price Monitor Integration V1.0.0 @zara

Provides sensors for electricity spot prices, total prices, and statistics.

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

import logging
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN

if TYPE_CHECKING:
    from .coordinator import GridPriceMonitorCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Grid Price Monitor sensors @zara"""
    # Lazy imports to avoid blocking the event loop during module import
    from .sensors import (
        GridPriceSpotSensor,
        GridPriceTotalSensor,
        GridPriceSpotNextHourSensor,
        GridPriceTotalNextHourSensor,
        GridPricesTodaySensor,
        GridPricesTomorrowSensor,
        GridPriceCheapestHourSensor,
        GridPriceMostExpensiveHourSensor,
        GridPriceAverageSensor,
        BatteryPowerSensor,
        BatteryChargedTodaySensor,
        BatteryChargedWeekSensor,
        BatteryChargedMonthSensor,
    )

    coordinator: "GridPriceMonitorCoordinator" = hass.data[DOMAIN][entry.entry_id]

    sensors = [
        GridPriceSpotSensor(coordinator, entry),
        GridPriceTotalSensor(coordinator, entry),
        GridPriceSpotNextHourSensor(coordinator, entry),
        GridPriceTotalNextHourSensor(coordinator, entry),
        GridPricesTodaySensor(coordinator, entry),
        GridPricesTomorrowSensor(coordinator, entry),
        GridPriceCheapestHourSensor(coordinator, entry),
        GridPriceMostExpensiveHourSensor(coordinator, entry),
        GridPriceAverageSensor(coordinator, entry),
    ]

    # Add battery sensors if configured
    if coordinator.has_battery_sensor:
        sensors.extend([
            BatteryPowerSensor(coordinator, entry),
            BatteryChargedTodaySensor(coordinator, entry),
            BatteryChargedWeekSensor(coordinator, entry),
            BatteryChargedMonthSensor(coordinator, entry),
        ])
        _LOGGER.debug("Battery tracking sensors added")

    async_add_entities(sensors)
    _LOGGER.debug("Added %d sensors for Grid Price Monitor", len(sensors))
