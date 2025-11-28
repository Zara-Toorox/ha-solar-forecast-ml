"""Battery Sensors V10.0.0 @zara

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

import logging
from typing import Any, Dict, Optional

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
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
    BATTERY_EFFICIENCY_SENSOR,
    BATTERY_POWER_SENSOR,
    BATTERY_RUNTIME_REMAINING_SENSOR,
    BATTERY_SOC_SENSOR,
    BATTERY_TO_HOUSE_SENSOR,
    DOMAIN,
    GRID_TO_BATTERY_SENSOR,
    GRID_TO_HOUSE_SENSOR,
    ICON_BATTERY,
    ICON_BATTERY_CHARGING,
    ICON_BATTERY_DISCHARGING,
    INTEGRATION_MODEL,
    SOFTWARE_VERSION,
    SOLAR_TO_BATTERY_SENSOR,
    SOLAR_TO_GRID_SENSOR,
    SOLAR_TO_HOUSE_SENSOR,
    UNIT_HOURS,
    UNIT_KWH,
    UNIT_WATT,
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
        """Return if entity is available @zara"""
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
        """Initialize SOC sensor @zara"""
        super().__init__(coordinator, entry, BATTERY_SOC_SENSOR, "State of Charge")

    @property
    def native_value(self) -> Optional[float]:
        """Return the state of charge @zara"""
        if not self.coordinator.battery_collector:
            return None
        return self.coordinator.battery_collector.get_state_of_charge()

    @property
    def icon(self) -> str:
        """Return dynamic icon based on SOC level @zara"""
        soc = self.native_value
        if soc is None:
            return ICON_BATTERY

        if self.coordinator.battery_collector.is_charging():
            return ICON_BATTERY_CHARGING
        elif self.coordinator.battery_collector.is_discharging():
            return ICON_BATTERY_DISCHARGING

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
        """Return additional attributes @zara"""
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
        """Initialize power sensor @zara"""
        super().__init__(coordinator, entry, BATTERY_POWER_SENSOR, "Power")

    @property
    def native_value(self) -> Optional[float]:
        """Return the battery power @zara"""
        if not self.coordinator.battery_collector:
            _LOGGER.debug("BatteryPowerSensor: No battery_collector available")
            return None

        power = self.coordinator.battery_collector.get_battery_power()
        _LOGGER.debug(f"BatteryPowerSensor: Returning power value: {power} W")
        return power

    @property
    def icon(self) -> str:
        """Return dynamic icon based on charging state @zara"""
        power = self.native_value
        if power is None:
            return ICON_BATTERY

        if power > 50:
            return ICON_BATTERY_CHARGING
        elif power < -50:
            return ICON_BATTERY_DISCHARGING
        else:
            return ICON_BATTERY

class BatteryRuntimeRemainingSensor(BaseBatterySensor):
    """Estimated battery runtime remaining (hours)"""

    _attr_device_class = SensorDeviceClass.DURATION
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfTime.HOURS
    _attr_icon = "mdi:timer-outline"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize runtime sensor @zara"""
        super().__init__(coordinator, entry, BATTERY_RUNTIME_REMAINING_SENSOR, "Runtime Remaining")

    @property
    def native_value(self) -> Optional[float]:
        """Return estimated runtime in hours @zara"""
        if not self.coordinator.battery_collector:
            return None

        consumption = (
            self.coordinator.data.get("current_consumption", 0) if self.coordinator.data else 0
        )

        if consumption <= 0:
            return None

        return self.coordinator.battery_collector.get_runtime_remaining(consumption)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes @zara"""
        if not self.coordinator.battery_collector:
            return {}

        runtime = self.native_value
        if runtime is None:
            return {"status": "No consumption data"}

        return {
            "remaining_capacity_kwh": self.coordinator.battery_collector.get_remaining_capacity(),
            "current_consumption_w": (
                self.coordinator.data.get("current_consumption", 0) if self.coordinator.data else 0
            ),
        }

class BatteryEfficiencySensor(BaseBatterySensor):
    """Battery efficiency sensor (%) - v9.0.0 based on energy flows"""

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_icon = "mdi:gauge"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize efficiency sensor @zara"""
        super().__init__(coordinator, entry, BATTERY_EFFICIENCY_SENSOR, "Efficiency")

    def _get_battery_coordinator(self):
        """Get battery coordinator from hass data @zara"""
        battery_key = f"{self.entry.entry_id}_battery"
        return self.coordinator.hass.data.get(DOMAIN, {}).get(battery_key)

    @property
    def native_value(self) -> Optional[float]:
        """Return battery efficiency (discharge / charge * 100) @zara"""
        battery_coordinator = self._get_battery_coordinator()
        if not battery_coordinator or not battery_coordinator.persistence:
            return None

        summary = battery_coordinator.persistence.get_today_summary()

        total_charge = summary.get("solar_to_battery_kwh", 0.0) + summary.get(
            "grid_to_battery_kwh", 0.0
        )

        total_discharge = summary.get("battery_to_house_kwh", 0.0)

        if total_charge <= 0:
            return None

        efficiency = (total_discharge / total_charge) * 100

        return round(min(efficiency, 100.0), 1)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes @zara"""
        battery_coordinator = self._get_battery_coordinator()
        if not battery_coordinator or not battery_coordinator.persistence:
            return {}

        summary = battery_coordinator.persistence.get_today_summary()

        total_charge = summary.get("solar_to_battery_kwh", 0.0) + summary.get(
            "grid_to_battery_kwh", 0.0
        )
        total_discharge = summary.get("battery_to_house_kwh", 0.0)

        return {
            "total_charge_today_kwh": round(total_charge, 3),
            "total_discharge_today_kwh": round(total_discharge, 3),
            "solar_charge_kwh": round(summary.get("solar_to_battery_kwh", 0.0), 3),
            "grid_charge_kwh": round(summary.get("grid_to_battery_kwh", 0.0), 3),
        }

class BaseEnergyFlowSensor(BaseBatterySensor):
    """Base class for v9.0.0 energy flow sensors"""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR

    def __init__(
        self,
        coordinator: SolarForecastMLCoordinator,
        entry: ConfigEntry,
        sensor_key: str,
        name: str,
        icon: str,
    ):
        """Initialize energy flow sensor"""
        super().__init__(coordinator, entry, sensor_key, name)
        self._attr_icon = icon

    def _get_battery_coordinator(self):
        """Get battery coordinator from hass data @zara"""
        battery_key = f"{self.entry.entry_id}_battery"
        return self.coordinator.hass.data.get(DOMAIN, {}).get(battery_key)

    @property
    def available(self) -> bool:
        """Return if entity is available (only if using v9.0.0 config) @zara"""
        if not super().available:
            return False

        battery_coordinator = self._get_battery_coordinator()
        return (
            battery_coordinator is not None
            and hasattr(battery_coordinator, "data_collector")
            and battery_coordinator.data_collector.using_new_config
        )

class SolarToHouseSensor(BaseEnergyFlowSensor):
    """Solar → House energy flow (kWh) - Direct solar consumption"""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        super().__init__(
            coordinator, entry, SOLAR_TO_HOUSE_SENSOR, "Solar to House", "mdi:solar-power"
        )

    @property
    def native_value(self) -> Optional[float]:
        """Return solar to house energy @zara"""
        battery_coordinator = self._get_battery_coordinator()
        if not battery_coordinator or not battery_coordinator.persistence:
            return None
        summary = battery_coordinator.persistence.get_today_summary()
        return summary.get("solar_to_house_kwh", 0.0)

class SolarToBatterySensor(BaseEnergyFlowSensor):
    """Solar → Battery energy flow (kWh) - Solar charging"""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            SOLAR_TO_BATTERY_SENSOR,
            "Solar to Battery",
            "mdi:solar-power-variant",
        )

    @property
    def native_value(self) -> Optional[float]:
        """Return solar to battery energy @zara"""
        battery_coordinator = self._get_battery_coordinator()
        if not battery_coordinator or not battery_coordinator.persistence:
            return None
        summary = battery_coordinator.persistence.get_today_summary()
        return summary.get("solar_to_battery_kwh", 0.0)

class SolarToGridSensor(BaseEnergyFlowSensor):
    """Solar → Grid energy flow (kWh) - Grid export"""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            SOLAR_TO_GRID_SENSOR,
            "Solar to Grid",
            "mdi:transmission-tower-export",
        )

    @property
    def native_value(self) -> Optional[float]:
        """Return solar to grid energy @zara"""
        battery_coordinator = self._get_battery_coordinator()
        if not battery_coordinator or not battery_coordinator.persistence:
            return None
        summary = battery_coordinator.persistence.get_today_summary()
        return summary.get("solar_to_grid_kwh", 0.0)

class GridToHouseSensor(BaseEnergyFlowSensor):
    """Grid → House energy flow (kWh) - Grid consumption"""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            GRID_TO_HOUSE_SENSOR,
            "Grid to House",
            "mdi:transmission-tower-import",
        )

    @property
    def native_value(self) -> Optional[float]:
        """Return grid to house energy @zara"""
        battery_coordinator = self._get_battery_coordinator()
        if not battery_coordinator or not battery_coordinator.persistence:
            return None
        summary = battery_coordinator.persistence.get_today_summary()
        return summary.get("grid_to_house_kwh", 0.0)

class GridToBatterySensor(BaseEnergyFlowSensor):
    """Grid → Battery energy flow (kWh) - Grid charging"""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            GRID_TO_BATTERY_SENSOR,
            "Grid to Battery",
            "mdi:battery-charging-high",
        )

    @property
    def native_value(self) -> Optional[float]:
        """Return grid to battery energy @zara"""
        battery_coordinator = self._get_battery_coordinator()
        if not battery_coordinator or not battery_coordinator.persistence:
            return None
        summary = battery_coordinator.persistence.get_today_summary()
        return summary.get("grid_to_battery_kwh", 0.0)

class BatteryToHouseSensor(BaseEnergyFlowSensor):
    """Battery → House energy flow (kWh) - Battery discharge"""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            BATTERY_TO_HOUSE_SENSOR,
            "Battery to House",
            "mdi:battery-arrow-down",
        )

    @property
    def native_value(self) -> Optional[float]:
        """Return battery to house energy @zara"""
        battery_coordinator = self._get_battery_coordinator()
        if not battery_coordinator or not battery_coordinator.persistence:
            return None
        summary = battery_coordinator.persistence.get_today_summary()
        return summary.get("battery_to_house_kwh", 0.0)

BATTERY_SENSORS = [

    BatterySOCSensor,
    BatteryPowerSensor,
    BatteryRuntimeRemainingSensor,
    BatteryEfficiencySensor,

    SolarToHouseSensor,
    SolarToBatterySensor,
    SolarToGridSensor,
    GridToHouseSensor,
    GridToBatterySensor,
    BatteryToHouseSensor,
]
