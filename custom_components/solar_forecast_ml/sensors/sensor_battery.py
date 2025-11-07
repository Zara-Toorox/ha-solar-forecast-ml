"""
Battery Sensors

Provides battery monitoring sensors completely separate from Solar/ML sensors
No interference with existing solar functionality

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from typing import Optional, Dict, Any

from homeassistant.components.sensor import (
    SensorEntity,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    PERCENTAGE,
    UnitOfEnergy,
    UnitOfPower,
    UnitOfTime,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from ..const import (
    DOMAIN,
    INTEGRATION_MODEL,
    SOFTWARE_VERSION,
    ICON_BATTERY,
    ICON_BATTERY_CHARGING,
    ICON_BATTERY_DISCHARGING,
    UNIT_KWH,
    UNIT_WATT,
    UNIT_HOURS,
    BATTERY_SOC_SENSOR,
    BATTERY_POWER_SENSOR,
    BATTERY_CHARGE_TODAY_SENSOR,
    BATTERY_DISCHARGE_TODAY_SENSOR,
    BATTERY_RUNTIME_REMAINING_SENSOR,
    BATTERY_EFFICIENCY_SENSOR,
)
from ..coordinator import SolarForecastMLCoordinator

_LOGGER = logging.getLogger(__name__)


class BaseBatterySensor(CoordinatorEntity, SensorEntity):
    """Base class for battery sensors"""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: SolarForecastMLCoordinator,
        entry: ConfigEntry,
        sensor_key: str,
        name: str,
    ):
        """Initialize the base battery sensor"""
        super().__init__(coordinator)
        self.entry = entry
        self._sensor_key = sensor_key

        self._attr_unique_id = f"{entry.entry_id}_battery_{sensor_key}"
        self._attr_translation_key = f"battery_{sensor_key}"

        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, f"{entry.entry_id}_battery")},
            name="Battery Management",
            manufacturer="Zara-Toorox",
            model=f"{INTEGRATION_MODEL} - Battery",
            sw_version=SOFTWARE_VERSION,
            via_device=(DOMAIN, entry.entry_id),
        )

    @property
    def available(self) -> bool:
        """Return if entity is available"""
        return (
            self.coordinator.last_update_success
            and self.coordinator.battery_collector is not None
            and self.coordinator.battery_collector.is_configured()
        )


class BatterySOCSensor(BaseBatterySensor):
    """Battery State of Charge sensor (%)"""

    _attr_device_class = SensorDeviceClass.BATTERY
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_icon = ICON_BATTERY

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize SOC sensor"""
        super().__init__(coordinator, entry, BATTERY_SOC_SENSOR, "State of Charge")

    @property
    def native_value(self) -> Optional[float]:
        """Return the state of charge"""
        if not self.coordinator.battery_collector:
            return None
        return self.coordinator.battery_collector.get_state_of_charge()

    @property
    def icon(self) -> str:
        """Return dynamic icon based on SOC level"""
        soc = self.native_value
        if soc is None:
            return ICON_BATTERY

        if self.coordinator.battery_collector.is_charging():
            return ICON_BATTERY_CHARGING
        elif self.coordinator.battery_collector.is_discharging():
            return ICON_BATTERY_DISCHARGING

        # Dynamic battery icon based on level
        if soc >= 90:
            return "mdi:battery"
        elif soc >= 80:
            return "mdi:battery-90"
        elif soc >= 70:
            return "mdi:battery-80"
        elif soc >= 60:
            return "mdi:battery-70"
        elif soc >= 50:
            return "mdi:battery-60"
        elif soc >= 40:
            return "mdi:battery-50"
        elif soc >= 30:
            return "mdi:battery-40"
        elif soc >= 20:
            return "mdi:battery-30"
        elif soc >= 10:
            return "mdi:battery-20"
        else:
            return "mdi:battery-10"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes"""
        if not self.coordinator.battery_collector:
            return {}

        return {
            "capacity_kwh": self.coordinator.battery_collector.battery_capacity,
            "remaining_kwh": self.coordinator.battery_collector.get_remaining_capacity(),
            "empty_kwh": self.coordinator.battery_collector.get_empty_capacity(),
            "is_charging": self.coordinator.battery_collector.is_charging(),
            "is_discharging": self.coordinator.battery_collector.is_discharging(),
        }


class BatteryPowerSensor(BaseBatterySensor):
    """Battery current power sensor (W) - positive=charging, negative=discharging"""

    _attr_device_class = SensorDeviceClass.POWER
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfPower.WATT
    _attr_icon = ICON_BATTERY

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize power sensor"""
        super().__init__(coordinator, entry, BATTERY_POWER_SENSOR, "Power")

    @property
    def native_value(self) -> Optional[float]:
        """Return the battery power"""
        if not self.coordinator.battery_collector:
            return None
        return self.coordinator.battery_collector.get_battery_power()

    @property
    def icon(self) -> str:
        """Return dynamic icon based on charging state"""
        power = self.native_value
        if power is None:
            return ICON_BATTERY

        if power > 50:
            return ICON_BATTERY_CHARGING
        elif power < -50:
            return ICON_BATTERY_DISCHARGING
        else:
            return ICON_BATTERY


class BatteryChargeTodaySensor(BaseBatterySensor):
    """Battery charged energy today (kWh)"""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_icon = "mdi:battery-arrow-up"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize charge today sensor"""
        super().__init__(coordinator, entry, BATTERY_CHARGE_TODAY_SENSOR, "Charge Today")

    @property
    def native_value(self) -> Optional[float]:
        """Return energy charged today"""
        if not self.coordinator.battery_collector:
            return None
        return self.coordinator.battery_collector.get_charge_today()


class BatteryDischargeTodaySensor(BaseBatterySensor):
    """Battery discharged energy today (kWh)"""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_icon = "mdi:battery-arrow-down"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize discharge today sensor"""
        super().__init__(coordinator, entry, BATTERY_DISCHARGE_TODAY_SENSOR, "Discharge Today")

    @property
    def native_value(self) -> Optional[float]:
        """Return energy discharged today"""
        if not self.coordinator.battery_collector:
            return None
        return self.coordinator.battery_collector.get_discharge_today()


class BatteryRuntimeRemainingSensor(BaseBatterySensor):
    """Estimated battery runtime remaining (hours)"""

    _attr_device_class = SensorDeviceClass.DURATION
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfTime.HOURS
    _attr_icon = "mdi:timer-outline"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize runtime sensor"""
        super().__init__(coordinator, entry, BATTERY_RUNTIME_REMAINING_SENSOR, "Runtime Remaining")

    @property
    def native_value(self) -> Optional[float]:
        """Return estimated runtime in hours"""
        if not self.coordinator.battery_collector:
            return None

        # Get current consumption (if available)
        consumption = self.coordinator.data.get("current_consumption", 0) if self.coordinator.data else 0

        if consumption <= 0:
            return None

        return self.coordinator.battery_collector.get_runtime_remaining(consumption)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes"""
        if not self.coordinator.battery_collector:
            return {}

        runtime = self.native_value
        if runtime is None:
            return {"status": "No consumption data"}

        return {
            "remaining_capacity_kwh": self.coordinator.battery_collector.get_remaining_capacity(),
            "current_consumption_w": self.coordinator.data.get("current_consumption", 0) if self.coordinator.data else 0,
        }


class BatteryEfficiencySensor(BaseBatterySensor):
    """Battery efficiency sensor (%)"""

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_icon = "mdi:gauge"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize efficiency sensor"""
        super().__init__(coordinator, entry, BATTERY_EFFICIENCY_SENSOR, "Efficiency")

    @property
    def native_value(self) -> Optional[float]:
        """Return battery efficiency"""
        if not self.coordinator.battery_collector:
            return None

        charge_today = self.coordinator.battery_collector.get_charge_today()
        discharge_today = self.coordinator.battery_collector.get_discharge_today()

        if charge_today <= 0:
            return None

        # Efficiency = (Discharged / Charged) * 100
        efficiency = (discharge_today / charge_today) * 100

        return round(min(efficiency, 100.0), 1)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes"""
        if not self.coordinator.battery_collector:
            return {}

        return {
            "charge_today": self.coordinator.battery_collector.get_charge_today(),
            "discharge_today": self.coordinator.battery_collector.get_discharge_today(),
        }


# Export all battery sensors
BATTERY_SENSORS = [
    BatterySOCSensor,
    BatteryPowerSensor,
    BatteryChargeTodaySensor,
    BatteryDischargeTodaySensor,
    BatteryRuntimeRemainingSensor,
    BatteryEfficiencySensor,
]
