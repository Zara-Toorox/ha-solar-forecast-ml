# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

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
