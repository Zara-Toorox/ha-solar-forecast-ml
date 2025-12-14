"""Base Sensor Class for Grid Price Monitor Integration V1.0.0 @zara

Provides the base class for all Grid Price Monitor sensors.

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

from typing import TYPE_CHECKING

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from ..const import DOMAIN, NAME, VERSION

if TYPE_CHECKING:
    from ..coordinator import GridPriceMonitorCoordinator


class GridPriceBaseSensor(CoordinatorEntity["GridPriceMonitorCoordinator"], SensorEntity):
    """Base class for Grid Price Monitor sensors @zara"""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: "GridPriceMonitorCoordinator",
        entry: ConfigEntry,
        sensor_type: str,
        name: str,
        icon: str,
    ) -> None:
        """Initialize the sensor @zara

        Args:
            coordinator: Data update coordinator
            entry: Config entry
            sensor_type: Type identifier for the sensor
            name: Display name
            icon: MDI icon name
        """
        super().__init__(coordinator)

        self._attr_unique_id = f"{entry.entry_id}_{sensor_type}"
        self._attr_name = name
        self._attr_icon = icon
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
