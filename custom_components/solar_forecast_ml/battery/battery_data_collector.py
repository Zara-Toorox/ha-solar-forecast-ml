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
    # NEW v9.0.0 constants
    CONF_BATTERY_POWER_SENSOR,
    CONF_BATTERY_SOC_SENSOR,
    CONF_SOLAR_PRODUCTION_SENSOR,
    CONF_INVERTER_OUTPUT_SENSOR,
    CONF_HOUSE_CONSUMPTION_SENSOR,
    CONF_GRID_IMPORT_SENSOR,
    CONF_GRID_EXPORT_SENSOR,
    CONF_GRID_CHARGE_POWER_SENSOR,
    CONF_BATTERY_TEMPERATURE_SENSOR,
    # LEGACY v8.x constants
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
        """Initialize battery data collector Args: hass: Home Assistant instance entry: Config entry with battery configuration"""
        self.hass = hass
        self.entry = entry

        # NEW v9.0.0: Watt-based sensors (preferred)
        self.battery_power_sensor = entry.options.get(CONF_BATTERY_POWER_SENSOR) or entry.data.get(CONF_BATTERY_POWER_SENSOR)
        self.battery_soc_sensor = entry.options.get(CONF_BATTERY_SOC_SENSOR) or entry.data.get(CONF_BATTERY_SOC_SENSOR)
        self.solar_production_sensor = entry.options.get(CONF_SOLAR_PRODUCTION_SENSOR) or entry.data.get(CONF_SOLAR_PRODUCTION_SENSOR)
        self.inverter_output_sensor = entry.options.get(CONF_INVERTER_OUTPUT_SENSOR) or entry.data.get(CONF_INVERTER_OUTPUT_SENSOR)
        self.house_consumption_sensor = entry.options.get(CONF_HOUSE_CONSUMPTION_SENSOR) or entry.data.get(CONF_HOUSE_CONSUMPTION_SENSOR)
        self.grid_import_sensor = entry.options.get(CONF_GRID_IMPORT_SENSOR) or entry.data.get(CONF_GRID_IMPORT_SENSOR)
        self.grid_export_sensor = entry.options.get(CONF_GRID_EXPORT_SENSOR) or entry.data.get(CONF_GRID_EXPORT_SENSOR)
        self.grid_charge_power_sensor = entry.options.get(CONF_GRID_CHARGE_POWER_SENSOR) or entry.data.get(CONF_GRID_CHARGE_POWER_SENSOR)
        self.battery_temperature_sensor = entry.options.get(CONF_BATTERY_TEMPERATURE_SENSOR) or entry.data.get(CONF_BATTERY_TEMPERATURE_SENSOR)

        # LEGACY v8.x: Old entity configuration (fallback for backwards compatibility)
        self.soc_entity = entry.options.get(CONF_BATTERY_SOC_ENTITY) or entry.data.get(CONF_BATTERY_SOC_ENTITY)
        self.power_entity = entry.options.get(CONF_BATTERY_POWER_ENTITY) or entry.data.get(CONF_BATTERY_POWER_ENTITY)
        self.charge_today_entity = entry.options.get(CONF_BATTERY_CHARGE_TODAY_ENTITY) or entry.data.get(CONF_BATTERY_CHARGE_TODAY_ENTITY)
        self.discharge_today_entity = entry.options.get(CONF_BATTERY_DISCHARGE_TODAY_ENTITY) or entry.data.get(CONF_BATTERY_DISCHARGE_TODAY_ENTITY)
        self.temperature_entity = entry.options.get(CONF_BATTERY_TEMPERATURE_ENTITY) or entry.data.get(CONF_BATTERY_TEMPERATURE_ENTITY)

        # Get battery capacity
        self.battery_capacity = entry.options.get(CONF_BATTERY_CAPACITY) or entry.data.get(CONF_BATTERY_CAPACITY, DEFAULT_BATTERY_CAPACITY)

        # Cache for log throttling (only log when value changes)
        self._last_logged_power = None

        # Determine if using new v9.0 config or legacy v8.x
        self.using_new_config = bool(self.battery_power_sensor and self.battery_soc_sensor
                                     and self.solar_production_sensor and self.inverter_output_sensor
                                     and self.house_consumption_sensor and self.grid_import_sensor
                                     and self.grid_export_sensor)

        if self.using_new_config:
            _LOGGER.info(
                f"BatteryDataCollector initialized (v9.0 watt-based) - "
                f"Capacity: {self.battery_capacity} kWh, "
                f"Battery Power: {self.battery_power_sensor}, "
                f"Solar: {self.solar_production_sensor}, "
                f"Inverter: {self.inverter_output_sensor}, "
                f"Grid Import: {self.grid_import_sensor}, "
                f"Grid Export: {self.grid_export_sensor}, "
                f"House: {self.house_consumption_sensor}"
            )
        else:
            _LOGGER.warning(
                f"BatteryDataCollector using LEGACY v8.x configuration - "
                f"Please migrate to v9.0 watt-based sensors for better accuracy! "
                f"Capacity: {self.battery_capacity} kWh, "
                f"SOC: {self.soc_entity or 'Not configured'}, "
                f"Power: {self.power_entity or 'Not configured'}"
            )

    def _get_entity_state(self, entity_id: Optional[str]) -> Optional[State]:
        """Get state of an entity Args: entity_id: Entity ID to query Returns: State object or None"""
        if not entity_id:
            return None

        state = self.hass.states.get(entity_id)
        if state is None:
            _LOGGER.warning(
                f"Battery entity '{entity_id}' not found in Home Assistant! "
                f"Please check the entity ID in the integration configuration. "
                f"The entity may not exist or has a typo."
            )
            return None

        # Check if state is unavailable or unknown
        if state.state in ('unavailable', 'unknown', 'none', 'None'):
            _LOGGER.debug(
                f"Battery entity '{entity_id}' state is '{state.state}'. "
                f"Entity may not be ready yet or device is offline."
            )

        return state

    def _get_numeric_state(self, entity_id: Optional[str], default: float = 0.0) -> float:
        """Get numeric state value from entity Args: entity_id: Entity ID to query default: Default value if entity not found or invalid Returns: Numeric state value or default"""
        state = self._get_entity_state(entity_id)

        if state is None:
            return default

        try:
            return float(state.state)
        except (ValueError, TypeError):
            _LOGGER.debug(f"Invalid numeric state for {entity_id}: {state.state} - using default {default}")
            return default

    def _convert_to_kwh(self, value: float, entity_id: Optional[str]) -> float:
        """Convert energy value to kWh if needed

        Detects unit from entity attributes and converts Wh to kWh automatically.
        Also handles sensors without unit_of_measurement by checking value range.

        IMPORTANT: Uses heuristic fallback even when unit is set to "kWh" because
        some integration sensors incorrectly label Wh values as kWh.

        Args:
            value: Energy value from sensor
            entity_id: Entity ID to check unit

        Returns:
            Energy value in kWh
        """
        if entity_id is None or value == 0.0:
            return value

        state = self._get_entity_state(entity_id)
        if state is None:
            return value

        # Get unit from entity attributes
        unit = state.attributes.get('unit_of_measurement', '').lower()

        # If unit explicitly says Wh or W·h, convert to kWh
        if unit in ['wh', 'w·h', 'watthour', 'watthours']:
            converted = value / 1000.0
            _LOGGER.debug(f"Converted {entity_id} from {value} Wh to {converted} kWh")
            return converted

        # Heuristic check: Battery charge/discharge today is typically 0-50 kWh
        # If value > 100, it's likely in Wh and needs conversion
        # Apply this EVEN if unit says "kWh" because some sensors are misconfigured
        if value > 100.0:
            converted = value / 1000.0

            if unit in ['kwh', 'kw·h', 'kilowatthour', 'kilowatthours']:
                _LOGGER.warning(
                    f"Sensor {entity_id} has unit='{unit}' but value={value:.2f} seems too high for kWh. "
                    f"Auto-converting to {converted:.3f} kWh. "
                    f"Please fix the sensor configuration - it likely delivers Wh but is labeled as kWh!"
                )
            else:
                _LOGGER.info(
                    f"Auto-detected Wh for {entity_id} (value={value:.2f}) - "
                    f"Converting to {converted:.3f} kWh."
                )
            return converted

        # Value is reasonable for kWh (0-100), keep as-is
        return value

    def get_state_of_charge(self) -> Optional[float]:
        """Get current battery State of Charge (%)

        Returns:
            SOC percentage (0-100) or None
        """
        # v9.0.0: Use new sensor if available
        if self.battery_soc_sensor:
            soc = self._get_numeric_state(self.battery_soc_sensor)
        # LEGACY v8.x: Fallback to old entity
        elif self.soc_entity:
            soc = self._get_numeric_state(self.soc_entity)
        else:
            return None

        if soc is None:
            return None

        if soc < 0 or soc > 100:
            _LOGGER.warning(f"Invalid SOC value: {soc}%")
            return None

        return round(soc, 1)

    def get_battery_power(self) -> Optional[float]:
        """Get current battery charge/discharge power (W)

        Positive = charging, Negative = discharging

        Returns:
            Power in Watts or None
        """
        # v9.0.0: Use new sensor if available
        if self.battery_power_sensor:
            entity_id = self.battery_power_sensor
        # LEGACY v8.x: Fallback to old entity
        elif self.power_entity:
            entity_id = self.power_entity
        else:
            _LOGGER.debug("Battery power sensor not configured")
            return None

        state = self._get_entity_state(entity_id)
        if state is None:
            _LOGGER.warning(f"Battery power sensor '{entity_id}' not found or unavailable")
            return None

        if state.state in ('unknown', 'unavailable', 'none', 'None'):
            _LOGGER.debug(f"Battery power sensor '{entity_id}' state is: {state.state}")
            return None

        try:
            power = float(state.state)
            # Only log when value changes significantly (avoid spam)
            if self._last_logged_power is None or abs(power - self._last_logged_power) > 5.0:
                _LOGGER.debug(f"Battery power from '{entity_id}': {power} W")
                self._last_logged_power = power
            return power
        except (ValueError, TypeError) as e:
            _LOGGER.warning(f"Invalid battery power state for '{entity_id}': {state.state} - Error: {e}")
            return None

    def get_charge_today(self) -> float:
        """Get total energy charged today (kWh) Returns: Energy in kWh"""
        raw_value = self._get_numeric_state(self.charge_today_entity, 0.0)
        return self._convert_to_kwh(raw_value, self.charge_today_entity)

    def get_discharge_today(self) -> float:
        """Get total energy discharged today (kWh) Returns: Energy in kWh"""
        raw_value = self._get_numeric_state(self.discharge_today_entity, 0.0)
        return self._convert_to_kwh(raw_value, self.discharge_today_entity)

    def get_battery_temperature(self) -> Optional[float]:
        """Get battery temperature (°C) Returns: Temperature in °C or None"""
        if not self.temperature_entity:
            return None

        return self._get_numeric_state(self.temperature_entity)

    def get_remaining_capacity(self) -> float:
        """Calculate remaining battery capacity (kWh) Returns: Remaining capacity in kWh"""
        soc = self.get_state_of_charge()
        if soc is None:
            return 0.0

        return round((soc / 100) * self.battery_capacity, 2)

    def get_empty_capacity(self) -> float:
        """Calculate empty battery capacity (kWh) Returns: Empty capacity in kWh"""
        soc = self.get_state_of_charge()
        if soc is None:
            return self.battery_capacity

        return round(((100 - soc) / 100) * self.battery_capacity, 2)

    def is_charging(self) -> bool:
        """Check if battery is currently charging Returns: True if charging (power > 0)"""
        power = self.get_battery_power()
        return power is not None and power > 50  # > 50W threshold to avoid noise

    def is_discharging(self) -> bool:
        """Check if battery is currently discharging Returns: True if discharging (power < 0)"""
        power = self.get_battery_power()
        return power is not None and power < -50  # < -50W threshold to avoid noise

    def get_runtime_remaining(self, consumption_watts: float) -> Optional[float]:
        """Calculate remaining runtime based on current consumption Args: consumption_watts: Current consumption in Watts Returns: Remaining runtime in hours or None"""
        if consumption_watts <= 0:
            return None

        remaining_kwh = self.get_remaining_capacity()
        if remaining_kwh <= 0:
            return 0.0

        # Runtime = Capacity (kWh) / Consumption (kW)
        runtime_hours = remaining_kwh / (consumption_watts / 1000)

        return round(runtime_hours, 1)

    def get_full_charge_time(self) -> Optional[float]:
        """Calculate time until battery is fully charged Returns: Time in hours or None if not charging"""
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
        """Get all battery data as dictionary Returns: Dictionary with all battery data"""
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
        """Check if battery is properly configured

        Returns:
            True if at least SOC sensor is configured (v9.0.0 or LEGACY v8.x)
        """
        # v9.0.0: Check new config
        if self.battery_soc_sensor is not None:
            return True
        # LEGACY v8.x: Check old config
        if self.soc_entity is not None:
            return True
        return False

    def validate_entities(self) -> Dict[str, bool]:
        """Validate all configured entities exist Returns: Dictionary with entity validation results"""
        return {
            'soc': self._get_entity_state(self.soc_entity) is not None,
            'power': self._get_entity_state(self.power_entity) is not None,
            'charge_today': self._get_entity_state(self.charge_today_entity) is not None,
            'discharge_today': self._get_entity_state(self.discharge_today_entity) is not None,
            'temperature': self._get_entity_state(self.temperature_entity) is not None if self.temperature_entity else True,
        }

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get diagnostic information for troubleshooting Returns: Dictionary with diagnostic data"""
        info = {
            'configured_entities': {
                'soc_entity': self.soc_entity,
                'power_entity': self.power_entity,
                'charge_today_entity': self.charge_today_entity,
                'discharge_today_entity': self.discharge_today_entity,
                'temperature_entity': self.temperature_entity,
            },
            'entity_states': {},
            'entity_validation': self.validate_entities(),
            'battery_capacity': self.battery_capacity,
        }

        # Get raw states
        for key, entity_id in info['configured_entities'].items():
            if entity_id:
                state = self._get_entity_state(entity_id)
                if state:
                    info['entity_states'][key] = {
                        'state': state.state,
                        'attributes': dict(state.attributes),
                        'last_changed': state.last_changed.isoformat(),
                    }
                else:
                    info['entity_states'][key] = 'NOT_FOUND'
            else:
                info['entity_states'][key] = 'NOT_CONFIGURED'

        return info

    # ========================================================================
    # NEW v9.0.0: Watt-based Power Sensor Getters
    # ========================================================================

    def get_solar_production(self) -> float:
        """Get current solar production (W)

        Returns:
            Solar production in Watts (≥0)
        """
        if self.using_new_config:
            return max(0.0, self._get_numeric_state(self.solar_production_sensor, 0.0))
        # Legacy fallback: No solar sensor in old config
        return 0.0

    def get_inverter_output(self) -> float:
        """Get current inverter AC output (W)

        Returns:
            Inverter output in Watts (≥0) - AC power to house (Solar+Battery combined)
        """
        if self.using_new_config:
            return max(0.0, self._get_numeric_state(self.inverter_output_sensor, 0.0))
        # Legacy fallback: No inverter sensor in old config
        return 0.0

    def get_house_consumption(self) -> float:
        """Get current house consumption (W)

        Returns:
            House consumption in Watts (≥0)
        """
        if self.using_new_config:
            return max(0.0, self._get_numeric_state(self.house_consumption_sensor, 0.0))
        # Legacy fallback: No house consumption sensor in old config
        return 0.0

    def get_grid_import(self) -> float:
        """Get current grid import (W)

        Returns:
            Grid import power in Watts (≥0)
        """
        if self.using_new_config:
            return max(0.0, self._get_numeric_state(self.grid_import_sensor, 0.0))
        # Legacy fallback: No grid sensor in old config
        return 0.0

    def get_grid_export(self) -> float:
        """Get current grid export (W)

        Returns:
            Grid export power in Watts (≥0)
        """
        if self.using_new_config:
            return max(0.0, self._get_numeric_state(self.grid_export_sensor, 0.0))
        # Legacy fallback: No grid sensor in old config
        return 0.0

    def get_grid_charge_power(self) -> float:
        """Get current grid charge power (W)

        Returns:
            Grid charge power in Watts (≥0)
        """
        if self.grid_charge_power_sensor:
            return max(0.0, self._get_numeric_state(self.grid_charge_power_sensor, 0.0))
        return 0.0

    def get_all_power_sensors(self) -> Dict[str, float]:
        """Get all current power sensor values

        Returns:
            Dictionary with all power values in Watts
        """
        return {
            'battery_power_w': self.get_battery_power() or 0.0,
            'soc_percent': self.get_state_of_charge() or 0.0,
            'solar_production_w': self.get_solar_production(),
            'inverter_output_w': self.get_inverter_output(),
            'grid_import_w': self.get_grid_import(),
            'grid_export_w': self.get_grid_export(),
            'house_consumption_w': self.get_house_consumption(),
            'grid_charge_power_w': self.get_grid_charge_power(),
            'temperature_c': self.get_battery_temperature(),
            'timestamp': datetime.now().isoformat(),
        }
