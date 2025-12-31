# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy, UnitOfPower

from ..const import (
    ICON_BATTERY,
    ICON_BATTERY_ENERGY,
    SENSOR_BATTERY_CHARGED_MONTH,
    SENSOR_BATTERY_CHARGED_TODAY,
    SENSOR_BATTERY_CHARGED_WEEK,
    SENSOR_BATTERY_POWER,
)
from .base import GridPriceBaseSensor

if TYPE_CHECKING:
    from ..coordinator import GridPriceMonitorCoordinator


class BatteryPowerSensor(GridPriceBaseSensor):
    """Sensor for current battery charging power @zara"""

    _attr_native_unit_of_measurement = UnitOfPower.WATT
    _attr_device_class = SensorDeviceClass.POWER
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(
        self, coordinator: "GridPriceMonitorCoordinator", entry: ConfigEntry
    ) -> None:
        """Initialize battery power sensor @zara"""
        super().__init__(
            coordinator,
            entry,
            SENSOR_BATTERY_POWER,
            "Battery Power",
            ICON_BATTERY,
        )

    @property
    def native_value(self) -> float | None:
        """Return the current battery power @zara"""
        if self.coordinator.data:
            return self.coordinator.data.get("battery_power")
        return None


class BatteryChargedTodaySensor(GridPriceBaseSensor):
    """Sensor for energy charged to battery today @zara"""

    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_device_class = SensorDeviceClass.ENERGY
    # Use TOTAL instead of TOTAL_INCREASING because this value resets daily
    _attr_state_class = SensorStateClass.TOTAL

    def __init__(
        self, coordinator: "GridPriceMonitorCoordinator", entry: ConfigEntry
    ) -> None:
        """Initialize battery charged today sensor @zara"""
        super().__init__(
            coordinator,
            entry,
            SENSOR_BATTERY_CHARGED_TODAY,
            "Battery Charged Today",
            ICON_BATTERY_ENERGY,
        )

    @property
    def native_value(self) -> float | None:
        """Return energy charged today @zara"""
        if self.coordinator.data:
            return self.coordinator.data.get("battery_charged_today")
        return None


class BatteryChargedWeekSensor(GridPriceBaseSensor):
    """Sensor for energy charged to battery this week @zara"""

    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_device_class = SensorDeviceClass.ENERGY
    # Use TOTAL instead of TOTAL_INCREASING because this value resets weekly
    _attr_state_class = SensorStateClass.TOTAL

    def __init__(
        self, coordinator: "GridPriceMonitorCoordinator", entry: ConfigEntry
    ) -> None:
        """Initialize battery charged week sensor @zara"""
        super().__init__(
            coordinator,
            entry,
            SENSOR_BATTERY_CHARGED_WEEK,
            "Battery Charged Week",
            ICON_BATTERY_ENERGY,
        )

    @property
    def native_value(self) -> float | None:
        """Return energy charged this week @zara"""
        if self.coordinator.data:
            return self.coordinator.data.get("battery_charged_week")
        return None


class BatteryChargedMonthSensor(GridPriceBaseSensor):
    """Sensor for energy charged to battery this month @zara"""

    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_device_class = SensorDeviceClass.ENERGY
    # Use TOTAL instead of TOTAL_INCREASING because this value resets monthly
    _attr_state_class = SensorStateClass.TOTAL

    def __init__(
        self, coordinator: "GridPriceMonitorCoordinator", entry: ConfigEntry
    ) -> None:
        """Initialize battery charged month sensor @zara"""
        super().__init__(
            coordinator,
            entry,
            SENSOR_BATTERY_CHARGED_MONTH,
            "Battery Charged Month",
            ICON_BATTERY_ENERGY,
        )

    @property
    def native_value(self) -> float | None:
        """Return energy charged this month @zara"""
        if self.coordinator.data:
            return self.coordinator.data.get("battery_charged_month")
        return None
