"""
Helper module for external sensor displays.
Common base class and helper functions for external sensors.
Version 1.1 - by Zara (Bugfix by Gemini)

Copyright (C) 2025 Zara-Toorox

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
from datetime import datetime
from typing import Optional, Any, Dict

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import callback, HomeAssistant, State
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.event import async_track_state_change_event

from ..core.helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


def format_time_ago(last_changed: datetime) -> str:
    """
    Formats timestamp as 'X min/h ago' - by Zara
    Space-saving: <1min = '< 1 min ago' - by Zara
    
    Args:
        last_changed: Timestamp of the last change
        
    Returns:
        Formatted string like "5 min ago" or "2 h ago"
    """
    now = dt_util.utcnow()
    delta = now - last_changed
    
    seconds = delta.total_seconds()
    
    if seconds < 60:
        return "< 1 min ago"  # Space-saving for < 1 minute - by Zara
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} min ago"
    else:
        hours = int(seconds / 3600)
        return f"{hours} h ago"


class BaseExternalSensor:
    """
    Common base for external sensor displays with LIVE updates - by Zara
    
    This class implements:
    - LIVE State Change Tracking
    - Uniform error handling
    - Timestamp formatting
    - Availability check
    """
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    hass: HomeAssistant  # Make hass available to the class

    def __init__(self, coordinator, entry: ConfigEntry, sensor_config: Dict[str, Any]):
        """
        Initializes the external sensor - by Zara
        
        Args:
            coordinator: DataUpdateCoordinator
            entry: ConfigEntry
            sensor_config: Dict with configuration:
                - key: Suffix for unique_id (e.g., 'temp_sensor')
                - config_key: Key for Config (e.g., CONF_TEMP_SENSOR)
                - name: Display name
                - icon: MDI Icon
                - unit: Default unit (optional)
                - device_class: Device class (optional)
                - format_string: Format string for display (optional)
        """
        self._sensor_config = sensor_config
        self.entry = entry
        self.coordinator = coordinator
        
        # Set attributes
        self._attr_unique_id = f"{entry.entry_id}_{sensor_config['key']}"
        self._attr_name = sensor_config['name']
        self._attr_icon = sensor_config['icon']
        self._attr_device_class = sensor_config.get('device_class')
        self._attr_native_unit_of_measurement = sensor_config.get('unit')

    @staticmethod
    def strip_entity_id(entity_id_raw: Any) -> Optional[str]:
        """Safely strips an entity ID string, returns None if invalid."""
        if isinstance(entity_id_raw, str) and entity_id_raw:
            return entity_id_raw.strip()
        return None

    @property
    def _sensor_entity_id(self) -> Optional[str]:
        """
        Gets the entity ID of the sensor to track from the ConfigEntry.
        This is the correct, centralized logic.
        """
        config_key = self._sensor_config.get('config_key')
        if not config_key:
            _LOGGER.error(f"Missing 'config_key' for sensor {self._attr_name}")
            return None
        
        entity_id_raw = self.entry.data.get(config_key)
        return self.strip_entity_id(entity_id_raw)

    @property
    def available(self) -> bool:
        """External sensors are always available (they show their own status messages) - by Zara"""
        return True
    
    async def async_added_to_hass(self) -> None:
        """Registers LIVE update listener - by Zara"""
        await super().async_added_to_hass()
        
        sensor_entity_id = self._sensor_entity_id
        
        if sensor_entity_id:
            _LOGGER.debug(f"Tracking external sensor {sensor_entity_id} for {self._attr_name}")
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass,
                    [sensor_entity_id],
                    self._handle_external_sensor_update
                )
            )
        else:
            _LOGGER.debug(f"No external sensor configured for {self._attr_name}")
    
    @callback
    def _handle_external_sensor_update(self, event) -> None:
        """Triggers update on external sensor change - by Zara"""
        _LOGGER.debug(f"External sensor {event.data.get('entity_id')} updated, refreshing {self._attr_name}")
        self.async_write_ha_state()
    
    @property
    def native_value(self) -> str:
        """
        Gets value from the configured sensor with timestamp - by Zara
        
        Returns:
            Formatted string with value, unit, and timestamp
            or an error message
        """
        sensor_entity_id = self._sensor_entity_id
        
        if not sensor_entity_id:
            return "Not configured"
        
        state = self.hass.states.get(sensor_entity_id)
        if not state:
            return "Entity not found"
        
        try:
            # Check availability
            if state.state in ['unavailable', 'unknown', 'none', None]:
                return "Unavailable"
            
            # Format timestamp
            time_ago = format_time_ago(state.last_changed)
            
            # Get unit (override default if state provides one)
            unit = state.attributes.get('unit_of_measurement', self._attr_native_unit_of_measurement or "")
            
            # Format output
            return self._format_value(state.state, unit, time_ago)
            
        except Exception as e:
            _LOGGER.warning(f"Error reading {self._sensor_config['name']}: {e}")
            return "Error"
    
    def _get_unit(self, state: State) -> Optional[str]:
        """
        Determines the unit of the sensor - by Zara
        (This is redundant as of v1.1, native_value handles it)
        
        Args:
            state: State object of the sensor
            
        Returns:
            Unit or None
        """
        unit_key = self._sensor_config.get('unit_key', 'unit_of_measurement')
        default_unit = self._sensor_config.get('unit')
        
        return state.attributes.get(unit_key, default_unit)
    
    def _format_value(self, value: str, unit: Optional[str], time_ago: str) -> str:
        """
        Formats sensor value for display - by Zara
        
        Args:
            value: Sensor value
            unit: Unit (optional)
            time_ago: Timestamp string
            
        Returns:
            Formatted string
        """
        format_string = self._sensor_config.get('format_string', '{value} {unit} ({time})')
        
        # If a specific format is defined
        if '{value}' in format_string:
            result = format_string.replace('{value}', str(value))
            if unit:
                result = result.replace('{unit}', str(unit))
            else:
                result = result.replace(' {unit}', '')  # Remove unit placeholder
            result = result.replace('{time}', time_ago)
            return result
        
        # Standard format
        if unit:
            return f"{value} {unit} ({time_ago})"
        else:
            return f"{value} ({time_ago})"