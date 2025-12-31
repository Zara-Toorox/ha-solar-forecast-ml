# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.config_entries import ConfigEntry

from ..const import (
    ICON_AVERAGE,
    ICON_CLOCK,
    SENSOR_AVERAGE_PRICE_TODAY,
    SENSOR_CHEAPEST_HOUR_TODAY,
    SENSOR_MOST_EXPENSIVE_HOUR_TODAY,
    UNIT_CT_KWH,
)
from .base import GridPriceBaseSensor

if TYPE_CHECKING:
    from ..coordinator import GridPriceMonitorCoordinator


class GridPriceCheapestHourSensor(GridPriceBaseSensor):
    """Sensor for cheapest hour today @zara"""

    def __init__(
        self, coordinator: "GridPriceMonitorCoordinator", entry: ConfigEntry
    ) -> None:
        """Initialize cheapest hour sensor @zara"""
        super().__init__(
            coordinator,
            entry,
            SENSOR_CHEAPEST_HOUR_TODAY,
            "Cheapest Hour Today",
            ICON_CLOCK,
        )

    @property
    def native_value(self) -> str | None:
        """Return the cheapest hour @zara"""
        if self.coordinator.data:
            hour = self.coordinator.data.get("cheapest_hour_today")
            if hour is not None:
                return f"{hour:02d}:00"
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes @zara"""
        if not self.coordinator.data:
            return {}

        return {
            "total_price": self.coordinator.data.get("cheapest_price_today"),
            "spot_price_net": self.coordinator.data.get("cheapest_spot_net"),
        }


class GridPriceMostExpensiveHourSensor(GridPriceBaseSensor):
    """Sensor for most expensive hour today @zara"""

    def __init__(
        self, coordinator: "GridPriceMonitorCoordinator", entry: ConfigEntry
    ) -> None:
        """Initialize most expensive hour sensor @zara"""
        super().__init__(
            coordinator,
            entry,
            SENSOR_MOST_EXPENSIVE_HOUR_TODAY,
            "Most Expensive Hour Today",
            ICON_CLOCK,
        )

    @property
    def native_value(self) -> str | None:
        """Return the most expensive hour @zara"""
        if self.coordinator.data:
            hour = self.coordinator.data.get("most_expensive_hour_today")
            if hour is not None:
                return f"{hour:02d}:00"
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes @zara"""
        if not self.coordinator.data:
            return {}

        return {
            "total_price": self.coordinator.data.get("most_expensive_price_today"),
            "spot_price_net": self.coordinator.data.get("most_expensive_spot_net"),
        }


class GridPriceAverageSensor(GridPriceBaseSensor):
    """Sensor for average price today @zara"""

    _attr_native_unit_of_measurement = UNIT_CT_KWH
    _attr_device_class = SensorDeviceClass.MONETARY
    # Note: monetary device_class does not support state_class=measurement

    def __init__(
        self, coordinator: "GridPriceMonitorCoordinator", entry: ConfigEntry
    ) -> None:
        """Initialize average price sensor @zara"""
        super().__init__(
            coordinator,
            entry,
            SENSOR_AVERAGE_PRICE_TODAY,
            "Average Price Today",
            ICON_AVERAGE,
        )

    @property
    def native_value(self) -> float | None:
        """Return the average total price @zara"""
        if self.coordinator.data:
            return self.coordinator.data.get("average_price_today")
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes @zara"""
        if not self.coordinator.data:
            return {}

        return {
            "spot_price_net_average": self.coordinator.data.get("average_spot_net"),
        }
