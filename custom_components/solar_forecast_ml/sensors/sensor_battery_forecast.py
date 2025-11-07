"""
Battery Forecast Sensors

Provides battery charging forecasts based on solar predictions
Integrates with solar forecast WITHOUT modifying solar/ML code

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
from homeassistant.const import UnitOfEnergy
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from ..const import (
    DOMAIN,
    INTEGRATION_MODEL,
    SOFTWARE_VERSION,
    BATTERY_EXPECTED_CHARGE_SOLAR_SENSOR,
    BATTERY_CHARGE_FROM_SOLAR_SENSOR,
    BATTERY_CHARGE_FROM_GRID_SENSOR,
    GRID_EXPORT_TODAY_SENSOR,
    GRID_IMPORT_TODAY_SENSOR,
    DIRECT_SOLAR_CONSUMPTION_SENSOR,
    AUTARKY_WITH_BATTERY_SENSOR,
    SELF_CONSUMPTION_WITH_BATTERY_SENSOR,
)
from ..coordinator import SolarForecastMLCoordinator

_LOGGER = logging.getLogger(__name__)


class BaseBatteryForecastSensor(CoordinatorEntity, SensorEntity):
    """Base class for battery forecast sensors"""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: SolarForecastMLCoordinator,
        entry: ConfigEntry,
        sensor_key: str,
        name: str,
    ):
        """Initialize the base battery forecast sensor"""
        super().__init__(coordinator)
        self.entry = entry
        self._sensor_key = sensor_key

        self._attr_unique_id = f"{entry.entry_id}_battery_forecast_{sensor_key}"
        self._attr_translation_key = f"battery_forecast_{sensor_key}"

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
        )


class BatteryExpectedChargeSolarSensor(BaseBatteryForecastSensor):
    """Expected battery charge from solar today (kWh)"""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_icon = "mdi:solar-panel"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize expected charge sensor"""
        super().__init__(coordinator, entry, BATTERY_EXPECTED_CHARGE_SOLAR_SENSOR, "Expected Charge from Solar")

    @property
    def native_value(self) -> Optional[float]:
        """Calculate expected charge from solar"""
        if not self.coordinator.data or not self.coordinator.battery_collector:
            return None

        # Get remaining solar forecast today
        remaining_solar = self.coordinator.data.get("prediction_kwh", 0)

        # Get current battery SOC
        soc = self.coordinator.battery_collector.get_state_of_charge()
        if soc is None:
            return None

        # Get empty capacity
        empty_capacity = self.coordinator.battery_collector.get_empty_capacity()

        # Estimate how much can go to battery
        # Simplified: Assume 50% of remaining solar goes to battery (rest to consumption)
        expected_to_battery = min(remaining_solar * 0.5, empty_capacity)

        return round(expected_to_battery, 2)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes"""
        if not self.coordinator.data or not self.coordinator.battery_collector:
            return {}

        return {
            "remaining_solar_forecast": self.coordinator.data.get("prediction_kwh", 0),
            "empty_capacity": self.coordinator.battery_collector.get_empty_capacity(),
            "current_soc": self.coordinator.battery_collector.get_state_of_charge(),
        }


class BatteryChargeFromSolarSensor(BaseBatteryForecastSensor):
    """Battery charged from solar today (kWh)"""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_icon = "mdi:solar-power"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize charge from solar sensor"""
        super().__init__(coordinator, entry, BATTERY_CHARGE_FROM_SOLAR_SENSOR, "Charge from Solar")

    @property
    def native_value(self) -> Optional[float]:
        """Return energy charged from solar today"""
        # This would require tracking - for now return estimate
        # TODO: Implement actual tracking in future version
        if not self.coordinator.battery_collector:
            return None

        # Placeholder: Return 0 for now, will be implemented with tracking
        return 0.0


class BatteryChargeFromGridSensor(BaseBatteryForecastSensor):
    """Battery charged from grid today (kWh)"""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_icon = "mdi:transmission-tower"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize charge from grid sensor"""
        super().__init__(coordinator, entry, BATTERY_CHARGE_FROM_GRID_SENSOR, "Charge from Grid")

    @property
    def native_value(self) -> Optional[float]:
        """Return energy charged from grid today"""
        if not self.coordinator.battery_collector:
            return None

        # Simplified: Total charge minus solar charge
        total_charge = self.coordinator.battery_collector.get_charge_today()
        # For now, assume all is from grid (will be refined with tracking)
        return total_charge


class GridExportTodaySensor(BaseBatteryForecastSensor):
    """Grid export today (kWh) - energy fed into grid"""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_icon = "mdi:transmission-tower-export"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize grid export sensor"""
        super().__init__(coordinator, entry, GRID_EXPORT_TODAY_SENSOR, "Grid Export")

    @property
    def native_value(self) -> Optional[float]:
        """Return grid export today"""
        # Placeholder for future implementation
        return 0.0


class GridImportTodaySensor(BaseBatteryForecastSensor):
    """Grid import today (kWh) - energy taken from grid"""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_icon = "mdi:transmission-tower-import"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize grid import sensor"""
        super().__init__(coordinator, entry, GRID_IMPORT_TODAY_SENSOR, "Grid Import")

    @property
    def native_value(self) -> Optional[float]:
        """Return grid import today"""
        # Placeholder for future implementation
        return 0.0


class DirectSolarConsumptionSensor(BaseBatteryForecastSensor):
    """Direct solar consumption today (kWh) - solar used directly without battery"""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_icon = "mdi:flash"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize direct consumption sensor"""
        super().__init__(coordinator, entry, DIRECT_SOLAR_CONSUMPTION_SENSOR, "Direct Solar Consumption")

    @property
    def native_value(self) -> Optional[float]:
        """Return direct solar consumption"""
        # Placeholder for future implementation
        return 0.0


class AutarkyWithBatterySensor(BaseBatteryForecastSensor):
    """Autarky with battery (%) - independence from grid"""

    _attr_device_class = SensorDeviceClass.POWER_FACTOR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "%"
    _attr_icon = "mdi:home-battery"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize autarky sensor"""
        super().__init__(coordinator, entry, AUTARKY_WITH_BATTERY_SENSOR, "Autarky with Battery")

    @property
    def native_value(self) -> Optional[float]:
        """Calculate autarky percentage"""
        # Autarky = (1 - GridImport / TotalConsumption) * 100
        # Placeholder for future implementation
        return 0.0


class SelfConsumptionWithBatterySensor(BaseBatteryForecastSensor):
    """Self-consumption with battery (%) - solar energy self-used"""

    _attr_device_class = SensorDeviceClass.POWER_FACTOR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "%"
    _attr_icon = "mdi:recycle"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize self-consumption sensor"""
        super().__init__(coordinator, entry, SELF_CONSUMPTION_WITH_BATTERY_SENSOR, "Self Consumption with Battery")

    @property
    def native_value(self) -> Optional[float]:
        """Calculate self-consumption percentage"""
        # Self-Consumption = (1 - GridExport / TotalProduction) * 100
        # Placeholder for future implementation
        return 0.0


# Export all battery forecast sensors
BATTERY_FORECAST_SENSORS = [
    BatteryExpectedChargeSolarSensor,
    BatteryChargeFromSolarSensor,
    BatteryChargeFromGridSensor,
    GridExportTodaySensor,
    GridImportTodaySensor,
    DirectSolarConsumptionSensor,
    AutarkyWithBatterySensor,
    SelfConsumptionWithBatterySensor,
]
