"""
Battery Cost & Profit Sensors

Provides financial tracking for battery charging/discharging
Uses BatteryCoordinator (separate from Solar)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from homeassistant.components.sensor import (
    SensorEntity,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CURRENCY_EURO
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.const import UnitOfEnergy

from ..const import (
    DOMAIN,
    INTEGRATION_MODEL,
    SOFTWARE_VERSION,
    UNIT_KWH,
    BATTERY_SOLAR_DISCHARGE_TODAY_SENSOR,
    BATTERY_GRID_DISCHARGE_TODAY_SENSOR,
    BATTERY_GRID_CHARGE_COST_TODAY_SENSOR,
    BATTERY_SOLAR_SAVINGS_TODAY_SENSOR,
    BATTERY_GRID_ARBITRAGE_PROFIT_TODAY_SENSOR,
    BATTERY_TOTAL_PROFIT_TODAY_SENSOR,
    BATTERY_GRID_CHARGE_MONTH_SENSOR,
    BATTERY_TOTAL_PROFIT_MONTH_SENSOR,
    BATTERY_TOTAL_PROFIT_YEAR_SENSOR,
)
from ..battery.battery_coordinator import BatteryCoordinator

_LOGGER = logging.getLogger(__name__)


class BaseBatteryCostSensor(CoordinatorEntity, SensorEntity):
    """Base class for battery cost sensors"""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: BatteryCoordinator,
        entry: ConfigEntry,
        sensor_key: str,
        name: str,
    ):
        """Initialize the base battery cost sensor"""
        super().__init__(coordinator)
        self.entry = entry
        self._sensor_key = sensor_key

        self._attr_unique_id = f"{entry.entry_id}_battery_cost_{sensor_key}"
        self._attr_translation_key = f"battery_cost_{sensor_key}"

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
            and self.coordinator.persistence is not None
        )


class BatterySolarDischargeTodaySensor(BaseBatteryCostSensor):
    """Solar discharge today (kWh) - energy from solar origin"""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_icon = "mdi:solar-power-variant"

    def __init__(self, coordinator: BatteryCoordinator, entry: ConfigEntry):
        """Initialize solar discharge sensor"""
        super().__init__(coordinator, entry, BATTERY_SOLAR_DISCHARGE_TODAY_SENSOR, "Solar Discharge Today")

    @property
    def native_value(self) -> Optional[float]:
        """Return solar discharge kWh today"""
        summary = self.coordinator.persistence.get_today_summary()
        return summary.get('solar_discharge_kwh', 0.0)


class BatteryGridDischargeTodaySensor(BaseBatteryCostSensor):
    """Grid discharge today (kWh) - energy from grid origin"""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_icon = "mdi:transmission-tower-export"

    def __init__(self, coordinator: BatteryCoordinator, entry: ConfigEntry):
        """Initialize grid discharge sensor"""
        super().__init__(coordinator, entry, BATTERY_GRID_DISCHARGE_TODAY_SENSOR, "Grid Discharge Today")

    @property
    def native_value(self) -> Optional[float]:
        """Return grid discharge kWh today"""
        summary = self.coordinator.persistence.get_today_summary()
        return summary.get('grid_discharge_kwh', 0.0)


class BatteryGridChargeCostTodaySensor(BaseBatteryCostSensor):
    """Grid charge cost today (€) - what you paid for grid charging"""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = CURRENCY_EURO
    _attr_icon = "mdi:currency-eur"

    def __init__(self, coordinator: BatteryCoordinator, entry: ConfigEntry):
        """Initialize grid charge cost sensor"""
        super().__init__(coordinator, entry, BATTERY_GRID_CHARGE_COST_TODAY_SENSOR, "Grid Charge Cost Today")

    @property
    def native_value(self) -> Optional[float]:
        """Return grid charge cost in € today"""
        summary = self.coordinator.persistence.get_today_summary()
        return round(summary.get('grid_charge_cost_eur', 0.0), 2)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes"""
        summary = self.coordinator.persistence.get_today_summary()
        grid_kwh = summary.get('grid_charge_kwh', 0.0)
        cost = summary.get('grid_charge_cost_eur', 0.0)

        attrs = {
            'grid_charge_kwh': round(grid_kwh, 2),
        }

        if grid_kwh > 0:
            avg_price = (cost / grid_kwh) * 100  # Convert to Cent/kWh
            attrs['avg_price_cent_kwh'] = round(avg_price, 2)

        return attrs


class BatterySolarSavingsTodaySensor(BaseBatteryCostSensor):
    """Solar savings today (€) - money saved by using solar from battery"""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = CURRENCY_EURO
    _attr_icon = "mdi:piggy-bank"

    def __init__(self, coordinator: BatteryCoordinator, entry: ConfigEntry):
        """Initialize solar savings sensor"""
        super().__init__(coordinator, entry, BATTERY_SOLAR_SAVINGS_TODAY_SENSOR, "Solar Savings Today")

    @property
    def native_value(self) -> Optional[float]:
        """Return solar savings in € today"""
        summary = self.coordinator.persistence.get_today_summary()
        return round(summary.get('solar_savings_eur', 0.0), 2)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes"""
        summary = self.coordinator.persistence.get_today_summary()
        return {
            'solar_discharge_kwh': round(summary.get('solar_discharge_kwh', 0.0), 2),
            'avg_discharge_price': summary.get('summary', {}).get('avg_discharge_price', 0.0),
        }


class BatteryGridArbitrageProfitTodaySensor(BaseBatteryCostSensor):
    """Grid arbitrage profit today (€) - profit from buy-low-sell-high"""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = CURRENCY_EURO
    _attr_icon = "mdi:chart-line"

    def __init__(self, coordinator: BatteryCoordinator, entry: ConfigEntry):
        """Initialize grid arbitrage profit sensor"""
        super().__init__(coordinator, entry, BATTERY_GRID_ARBITRAGE_PROFIT_TODAY_SENSOR, "Grid Arbitrage Profit Today")

    @property
    def native_value(self) -> Optional[float]:
        """Return grid arbitrage profit in € today"""
        summary = self.coordinator.persistence.get_today_summary()
        return round(summary.get('grid_arbitrage_profit_eur', 0.0), 2)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes"""
        summary = self.coordinator.persistence.get_today_summary()
        return {
            'grid_discharge_kwh': round(summary.get('grid_discharge_kwh', 0.0), 2),
            'avg_charge_price': summary.get('summary', {}).get('avg_grid_charge_price', 0.0),
            'avg_discharge_price': summary.get('summary', {}).get('avg_discharge_price', 0.0),
        }


class BatteryTotalProfitTodaySensor(BaseBatteryCostSensor):
    """Total profit today (€) - solar savings + arbitrage - grid charge cost"""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = CURRENCY_EURO
    _attr_icon = "mdi:cash-multiple"

    def __init__(self, coordinator: BatteryCoordinator, entry: ConfigEntry):
        """Initialize total profit sensor"""
        super().__init__(coordinator, entry, BATTERY_TOTAL_PROFIT_TODAY_SENSOR, "Total Profit Today")

    @property
    def native_value(self) -> Optional[float]:
        """Return total profit in € today"""
        summary = self.coordinator.persistence.get_today_summary()
        return round(summary.get('total_profit_eur', 0.0), 2)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return detailed breakdown"""
        summary = self.coordinator.persistence.get_today_summary()
        return {
            'solar_savings': round(summary.get('solar_savings_eur', 0.0), 2),
            'grid_arbitrage_profit': round(summary.get('grid_arbitrage_profit_eur', 0.0), 2),
            'grid_charge_cost': round(summary.get('grid_charge_cost_eur', 0.0), 2),
            'breakdown': (
                f"+{summary.get('solar_savings_eur', 0.0):.2f}€ solar "
                f"+{summary.get('grid_arbitrage_profit_eur', 0.0):.2f}€ arbitrage "
                f"-{summary.get('grid_charge_cost_eur', 0.0):.2f}€ cost"
            ),
        }


class BatteryGridChargeMonthSensor(BaseBatteryCostSensor):
    """Grid charge this month (kWh)"""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_icon = "mdi:transmission-tower"

    def __init__(self, coordinator: BatteryCoordinator, entry: ConfigEntry):
        """Initialize grid charge month sensor"""
        super().__init__(coordinator, entry, BATTERY_GRID_CHARGE_MONTH_SENSOR, "Grid Charge Month")

    @property
    def native_value(self) -> Optional[float]:
        """Return grid charge kWh this month"""
        now = datetime.now()
        summary = self.coordinator.persistence.get_month_summary(now.year, now.month)
        return round(summary.get('grid_charge_kwh', 0.0), 2)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes"""
        now = datetime.now()
        summary = self.coordinator.persistence.get_month_summary(now.year, now.month)
        return {
            'year_month': f"{now.year}-{now.month:02d}",
            'days_tracked': summary.get('days_tracked', 0),
            'grid_charge_cost': round(summary.get('grid_charge_cost_eur', 0.0), 2),
        }


class BatteryTotalProfitMonthSensor(BaseBatteryCostSensor):
    """Total profit this month (€)"""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = CURRENCY_EURO
    _attr_icon = "mdi:cash-multiple"

    def __init__(self, coordinator: BatteryCoordinator, entry: ConfigEntry):
        """Initialize total profit month sensor"""
        super().__init__(coordinator, entry, BATTERY_TOTAL_PROFIT_MONTH_SENSOR, "Total Profit Month")

    @property
    def native_value(self) -> Optional[float]:
        """Return total profit € this month"""
        now = datetime.now()
        summary = self.coordinator.persistence.get_month_summary(now.year, now.month)
        return round(summary.get('total_profit_eur', 0.0), 2)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return detailed breakdown"""
        now = datetime.now()
        summary = self.coordinator.persistence.get_month_summary(now.year, now.month)
        return {
            'year_month': f"{now.year}-{now.month:02d}",
            'days_tracked': summary.get('days_tracked', 0),
            'solar_savings': round(summary.get('solar_savings_eur', 0.0), 2),
            'grid_arbitrage_profit': round(summary.get('grid_arbitrage_profit_eur', 0.0), 2),
            'grid_charge_cost': round(summary.get('grid_charge_cost_eur', 0.0), 2),
        }


class BatteryTotalProfitYearSensor(BaseBatteryCostSensor):
    """Total profit this year (€)"""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = CURRENCY_EURO
    _attr_icon = "mdi:cash-multiple"

    def __init__(self, coordinator: BatteryCoordinator, entry: ConfigEntry):
        """Initialize total profit year sensor"""
        super().__init__(coordinator, entry, BATTERY_TOTAL_PROFIT_YEAR_SENSOR, "Total Profit Year")

    @property
    def native_value(self) -> Optional[float]:
        """Return total profit € this year"""
        now = datetime.now()
        summary = self.coordinator.persistence.get_year_summary(now.year)
        return round(summary.get('total_profit_eur', 0.0), 2)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return detailed breakdown"""
        now = datetime.now()
        summary = self.coordinator.persistence.get_year_summary(now.year)
        return {
            'year': now.year,
            'months_tracked': summary.get('months_tracked', 0),
            'solar_savings': round(summary.get('solar_savings_eur', 0.0), 2),
            'grid_arbitrage_profit': round(summary.get('grid_arbitrage_profit_eur', 0.0), 2),
            'grid_charge_cost': round(summary.get('grid_charge_cost_eur', 0.0), 2),
        }


# Export all battery cost sensors
BATTERY_COST_SENSORS = [
    BatterySolarDischargeTodaySensor,
    BatteryGridDischargeTodaySensor,
    BatteryGridChargeCostTodaySensor,
    BatterySolarSavingsTodaySensor,
    BatteryGridArbitrageProfitTodaySensor,
    BatteryTotalProfitTodaySensor,
    BatteryGridChargeMonthSensor,
    BatteryTotalProfitMonthSensor,
    BatteryTotalProfitYearSensor,
]
