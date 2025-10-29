"""
Sensor Data Collector Module
Collects and processes all sensor data centrally

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""
import logging
from typing import Any, Dict, Optional

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry

from .const import (
    CONF_TEMP_SENSOR, CONF_WIND_SENSOR, CONF_RAIN_SENSOR,
    CONF_UV_SENSOR, CONF_LUX_SENSOR, CONF_CURRENT_POWER
)

_LOGGER = logging.getLogger(__name__)


class SensorDataCollector:
    """
    Central class for sensor data collection.
    """
    
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        self.hass = hass
        self.entry = entry
        
        # Map internal keys to the config entry keys
        self._sensor_configs = {
            'temp_sensor': CONF_TEMP_SENSOR,
            'wind_sensor': CONF_WIND_SENSOR,
            'rain_sensor': CONF_RAIN_SENSOR,
            'uv_sensor': CONF_UV_SENSOR,
            'lux_sensor': CONF_LUX_SENSOR,
            'current_power': CONF_CURRENT_POWER
        }
    
    @staticmethod
    def strip_entity_id(entity_id_raw: Any) -> Optional[str]:
        """Safely strips an entity ID string, returns None if invalid."""
        if isinstance(entity_id_raw, str) and entity_id_raw:
            return entity_id_raw.strip()
        return None
    
    def get_sensor_entity_id(self, config_key: str) -> Optional[str]:
        """Gets a specific sensor entity ID from the config entry."""
        entity_id_raw = self.entry.data.get(config_key)
        return self.strip_entity_id(entity_id_raw)
    
    def get_sensor_value(self, entity_id: str) -> Optional[float]:
        """
        Gets and validates the state of an entity ID, converting it to float.
        Returns None if unavailable, unknown, or not a valid number.
        """
        if not entity_id:
            return None
        
        state = self.hass.states.get(entity_id)
        if not state or state.state in ['unavailable', 'unknown', 'None', '']:
            return None
        
        try:
            return float(state.state)
        except (ValueError, TypeError):
            _LOGGER.warning(f"Could not parse sensor value for {entity_id}: {state.state}")
            return None
    
    def collect_all_sensor_data(self, solar_capacity: float, power_entity: Optional[str]) -> Dict[str, Any]:
        """
        Collects all configured sensor data into a dictionary.
        This includes 'current_power'.
        """
        sensor_data = {
            'solar_capacity': solar_capacity,
            'power_entity': power_entity
        }
        
        for key, config_key in self._sensor_configs.items():
            entity_id = self.get_sensor_entity_id(config_key)
            if entity_id:
                value = self.get_sensor_value(entity_id)
                if value is not None:
                    sensor_data[key] = value
                    _LOGGER.debug(f"Sensor {key} loaded: {value}")
        
        return sensor_data
    
    def collect_sensor_data_dict(self) -> Dict[str, Optional[float]]:
        """
        Collects external sensor data, *excluding* 'current_power'.
        Used for populating weather-related data.
        """
        sensor_data_dict = {}
        
        for key, config_key in self._sensor_configs.items():
            if key == 'current_power':
                continue  # Skip internal power sensor
            
            entity_id = self.get_sensor_entity_id(config_key)
            if entity_id:
                sensor_data_dict[key] = self.get_sensor_value(entity_id)
            else:
                sensor_data_dict[key] = None
        
        return sensor_data_dict
    
    async def wait_for_external_sensors(self, max_wait: int = 25) -> int:
        """
        Waits at startup for at least one external sensor to become available.
        This prevents immediate errors if weather sensors are slow to load.
        """
        import asyncio
        
        _LOGGER.info("Waiting for external sensors (max %ds)...", max_wait)
        
        wait_interval = 2
        total_waited = 0
        
        while total_waited < max_wait:
            available_count = 0
            sensor_status = []
            
            for key, config_key in self._sensor_configs.items():
                if key == 'current_power':
                    continue  # We are waiting for *external* weather sensors
                
                entity_id = self.get_sensor_entity_id(config_key)
                
                if entity_id:
                    value = self.get_sensor_value(entity_id)
                    if value is not None:
                        available_count += 1
                        sensor_status.append(f"{key}=OK")
                    else:
                        sensor_status.append(f"{key}=FAIL")
            
            if available_count > 0:
                _LOGGER.info(
                    "External sensors ready: %d available after %ds [%s]",
                    available_count, total_waited, ", ".join(sensor_status)
                )
                return available_count
            
            await asyncio.sleep(wait_interval)
            total_waited += wait_interval
        
        _LOGGER.warning("No external sensors available after %ds", max_wait)
        return 0