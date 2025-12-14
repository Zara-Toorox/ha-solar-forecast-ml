"""Binary Sensor Platform for Grid Price Monitor Integration V1.0.0 @zara

Provides binary sensor for cheap energy detection.

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
from typing import Any, TYPE_CHECKING

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

if TYPE_CHECKING:
    from .coordinator import GridPriceMonitorCoordinator

from .const import (
    ATTR_CHEAP_HOURS_TODAY,
    ATTR_CHEAP_HOURS_TOMORROW,
    ATTR_FORECAST_TODAY,
    ATTR_FORECAST_TOMORROW,
    ATTR_LAST_UPDATE,
    ATTR_NEXT_CHEAP_HOUR,
    ATTR_PRICE_TREND,
    BINARY_SENSOR_CHEAP_ENERGY,
    DOMAIN,
    ICON_CHEAP,
    NAME,
    VERSION,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Grid Price Monitor binary sensors @zara"""
    # Lazy import to avoid blocking the event loop during module import
    from .coordinator import GridPriceMonitorCoordinator

    coordinator: GridPriceMonitorCoordinator = hass.data[DOMAIN][entry.entry_id]

    async_add_entities([GridPriceCheapEnergySensor(coordinator, entry)])
    _LOGGER.debug("Added binary sensor for Grid Price Monitor")


class GridPriceCheapEnergySensor(
    CoordinatorEntity["GridPriceMonitorCoordinator"], BinarySensorEntity
):
    """Binary sensor indicating if current energy price is cheap @zara"""

    _attr_has_entity_name = True
    _attr_device_class = BinarySensorDeviceClass.POWER

    def __init__(
        self, coordinator: GridPriceMonitorCoordinator, entry: ConfigEntry
    ) -> None:
        """Initialize the binary sensor @zara"""
        super().__init__(coordinator)

        self._attr_unique_id = f"{entry.entry_id}_{BINARY_SENSOR_CHEAP_ENERGY}"
        self._attr_name = "Cheap Energy"
        self._attr_icon = ICON_CHEAP
        self._entry = entry

    @property
    def available(self) -> bool:
        """Return True if entity is available @zara

        Entity is available when coordinator has valid data.
        """
        return self.coordinator.last_update_success and self.coordinator.data is not None

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info @zara"""
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            name=NAME,
            manufacturer="Zara-Toorox",
            model="Grid Price Monitor",
            sw_version=VERSION,
        )

    @property
    def is_on(self) -> bool | None:
        """Return true if energy is cheap @zara"""
        if self.coordinator.data:
            return self.coordinator.data.get("is_cheap", False)
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes @zara"""
        if not self.coordinator.data:
            return {}

        return {
            "current_total_price": self.coordinator.data.get("total_price"),
            "max_price_threshold": self.coordinator.data.get("max_price_threshold"),
            "spot_price": self.coordinator.data.get("spot_price"),
            "markup_total": self.coordinator.data.get("markup_total"),
            ATTR_NEXT_CHEAP_HOUR: self.coordinator.data.get("next_cheap_hour"),
            "next_cheap_timestamp": self.coordinator.data.get("next_cheap_timestamp"),
            ATTR_CHEAP_HOURS_TODAY: self.coordinator.data.get("cheap_hours_today", []),
            ATTR_CHEAP_HOURS_TOMORROW: self.coordinator.data.get("cheap_hours_tomorrow", []),
            ATTR_PRICE_TREND: self.coordinator.data.get("price_trend"),
            ATTR_FORECAST_TODAY: self.coordinator.data.get("forecast_today", []),
            ATTR_FORECAST_TOMORROW: self.coordinator.data.get("forecast_tomorrow", []),
            ATTR_LAST_UPDATE: self.coordinator.data.get("last_update"),
        }
