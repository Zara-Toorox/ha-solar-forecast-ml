"""
Battery Data Collector

Collects battery sensor data from Home Assistant entities
Completely independent from Solar/ML components

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
from homeassistant.core import HomeAssistant, State
from homeassistant.config_entries import ConfigEntry

from ..const import (
    CONF_BATTERY_SOC_ENTITY,
    CONF_BATTERY_POWER_ENTITY,
    CONF_BATTERY_CHARGE_TODAY_ENTITY,
    CONF_BATTERY_DISCHARGE_TODAY_ENTITY,
    CONF_BATTERY_TEMPERATURE_ENTITY,
    CONF_BATTERY_CAPACITY,
    DEFAULT_BATTERY_CAPACITY,
)

_LOGGER = logging.getLogger(__name__)


class BatteryDataCollector:
    """Collects battery data from Home Assistant entities"""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """
        Initialize battery data collector

        Args:
            hass: Home Assistant instance
            entry: Config entry with battery configuration
        """
        self.hass = hass
        self.entry = entry

        # Get configured entities from options (with fallback to data for backwards compatibility)
        self.soc_entity = entry.options.get(CONF_BATTERY_SOC_ENTITY) or entry.data.get(CONF_BATTERY_SOC_ENTITY)
        self.power_entity = entry.options.get(CONF_BATTERY_POWER_ENTITY) or entry.data.get(CONF_BATTERY_POWER_ENTITY)
        self.charge_today_entity = entry.options.get(CONF_BATTERY_CHARGE_TODAY_ENTITY) or entry.data.get(CONF_BATTERY_CHARGE_TODAY_ENTITY)
        self.discharge_today_entity = entry.options.get(CONF_BATTERY_DISCHARGE_TODAY_ENTITY) or entry.data.get(CONF_BATTERY_DISCHARGE_TODAY_ENTITY)
        self.temperature_entity = entry.options.get(CONF_BATTERY_TEMPERATURE_ENTITY) or entry.data.get(CONF_BATTERY_TEMPERATURE_ENTITY)

        # Get battery capacity from options (with fallback to data)
        self.battery_capacity = entry.options.get(CONF_BATTERY_CAPACITY) or entry.data.get(CONF_BATTERY_CAPACITY, DEFAULT_BATTERY_CAPACITY)

        _LOGGER.info(
            f"BatteryDataCollector initialized - "
            f"Capacity: {self.battery_capacity} kWh, "
            f"SOC Entity: {self.soc_entity or 'Not configured'}, "
            f"Power Entity: {self.power_entity or 'Not configured'}"
        )

    def _get_entity_state(self, entity_id: Optional[str]) -> Optional[State]:
        """
        Get state of an entity

        Args:
            entity_id: Entity ID to query

        Returns:
            State object or None
        """
        if not entity_id:
            return None

        state = self.hass.states.get(entity_id)
        if state is None:
            _LOGGER.debug(f"Battery entity not found: {entity_id} - Please check entity ID in configuration")

        return state

    def _get_numeric_state(self, entity_id: Optional[str], default: float = 0.0) -> float:
        """
        Get numeric state value from entity

        Args:
            entity_id: Entity ID to query
            default: Default value if entity not found or invalid

        Returns:
            Numeric state value or default
        """
        state = self._get_entity_state(entity_id)

        if state is None:
            return default

        try:
            return float(state.state)
        except (ValueError, TypeError):
            _LOGGER.warning(f"Invalid numeric state for {entity_id}: {state.state}")
            return default

    def get_state_of_charge(self) -> Optional[float]:
        """
        Get current battery State of Charge (%)

        Returns:
            SOC percentage (0-100) or None
        """
        soc = self._get_numeric_state(self.soc_entity)

        if soc < 0 or soc > 100:
            _LOGGER.warning(f"Invalid SOC value: {soc}%")
            return None

        return round(soc, 1)

    def get_battery_power(self) -> Optional[float]:
        """
        Get current battery charge/discharge power (W)
        Positive = charging, Negative = discharging

        Returns:
            Power in Watts or None
        """
        return self._get_numeric_state(self.power_entity, 0.0)

    def get_charge_today(self) -> float:
        """
        Get total energy charged today (kWh)

        Returns:
            Energy in kWh
        """
        return self._get_numeric_state(self.charge_today_entity, 0.0)

    def get_discharge_today(self) -> float:
        """
        Get total energy discharged today (kWh)

        Returns:
            Energy in kWh
        """
        return self._get_numeric_state(self.discharge_today_entity, 0.0)

    def get_battery_temperature(self) -> Optional[float]:
        """
        Get battery temperature (°C)

        Returns:
            Temperature in °C or None
        """
        if not self.temperature_entity:
            return None

        return self._get_numeric_state(self.temperature_entity)

    def get_remaining_capacity(self) -> float:
        """
        Calculate remaining battery capacity (kWh)

        Returns:
            Remaining capacity in kWh
        """
        soc = self.get_state_of_charge()
        if soc is None:
            return 0.0

        return round((soc / 100) * self.battery_capacity, 2)

    def get_empty_capacity(self) -> float:
        """
        Calculate empty battery capacity (kWh)

        Returns:
            Empty capacity in kWh
        """
        soc = self.get_state_of_charge()
        if soc is None:
            return self.battery_capacity

        return round(((100 - soc) / 100) * self.battery_capacity, 2)

    def is_charging(self) -> bool:
        """
        Check if battery is currently charging

        Returns:
            True if charging (power > 0)
        """
        power = self.get_battery_power()
        return power is not None and power > 50  # > 50W threshold to avoid noise

    def is_discharging(self) -> bool:
        """
        Check if battery is currently discharging

        Returns:
            True if discharging (power < 0)
        """
        power = self.get_battery_power()
        return power is not None and power < -50  # < -50W threshold to avoid noise

    def get_runtime_remaining(self, consumption_watts: float) -> Optional[float]:
        """
        Calculate remaining runtime based on current consumption

        Args:
            consumption_watts: Current consumption in Watts

        Returns:
            Remaining runtime in hours or None
        """
        if consumption_watts <= 0:
            return None

        remaining_kwh = self.get_remaining_capacity()
        if remaining_kwh <= 0:
            return 0.0

        # Runtime = Capacity (kWh) / Consumption (kW)
        runtime_hours = remaining_kwh / (consumption_watts / 1000)

        return round(runtime_hours, 1)

    def get_full_charge_time(self) -> Optional[float]:
        """
        Calculate time until battery is fully charged

        Returns:
            Time in hours or None if not charging
        """
        if not self.is_charging():
            return None

        power = self.get_battery_power()
        if power is None or power <= 0:
            return None

        empty_capacity = self.get_empty_capacity()
        if empty_capacity <= 0:
            return 0.0

        # Charge time = Empty capacity (kWh) / Charge power (kW)
        charge_time_hours = empty_capacity / (power / 1000)

        return round(charge_time_hours, 1)

    def get_battery_data(self) -> Dict[str, Any]:
        """
        Get all battery data as dictionary

        Returns:
            Dictionary with all battery data
        """
        soc = self.get_state_of_charge()
        power = self.get_battery_power()

        return {
            'soc': soc,
            'soc_percent': soc,
            'power': power,
            'power_w': power,
            'charge_today': self.get_charge_today(),
            'discharge_today': self.get_discharge_today(),
            'temperature': self.get_battery_temperature(),
            'capacity': self.battery_capacity,
            'remaining_capacity': self.get_remaining_capacity(),
            'empty_capacity': self.get_empty_capacity(),
            'is_charging': self.is_charging(),
            'is_discharging': self.is_discharging(),
            'timestamp': datetime.now().isoformat(),
        }

    def is_configured(self) -> bool:
        """
        Check if battery is properly configured

        Returns:
            True if at least SOC entity is configured
        """
        return self.soc_entity is not None

    def validate_entities(self) -> Dict[str, bool]:
        """
        Validate all configured entities exist

        Returns:
            Dictionary with entity validation results
        """
        return {
            'soc': self._get_entity_state(self.soc_entity) is not None,
            'power': self._get_entity_state(self.power_entity) is not None,
            'charge_today': self._get_entity_state(self.charge_today_entity) is not None,
            'discharge_today': self._get_entity_state(self.discharge_today_entity) is not None,
            'temperature': self._get_entity_state(self.temperature_entity) is not None if self.temperature_entity else True,
        }
