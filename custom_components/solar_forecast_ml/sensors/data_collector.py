"""
Sensor Data Collector Module
Collects and processes configured external sensor data centrally.

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
import asyncio # Import asyncio for sleep
from typing import Any, Dict, Optional

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry

# Import centralized external sensor mapping
from ..const import EXTERNAL_SENSOR_MAPPING

_LOGGER = logging.getLogger(__name__)


class SensorDataCollector:
    """
    Central class for collecting and accessing configured external sensor data.
    Provides methods to safely retrieve entity IDs and their float values.
    Uses centralized EXTERNAL_SENSOR_MAPPING from const.py as single source of truth.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """Initialize the SensorDataCollector."""
        self.hass = hass
        self.entry = entry

        # Use centralized sensor mapping from const.py
        # Maps internal keys (e.g., 'temperature', 'humidity') to config entry keys
        self._sensor_configs = EXTERNAL_SENSOR_MAPPING
        _LOGGER.debug("SensorDataCollector initialized with centralized sensor mapping.")

    @staticmethod
    def strip_entity_id(entity_id_raw: Any) -> Optional[str]:
        """
        Safely strips whitespace from a potential entity ID string.
        Returns the stripped ID or None if the input is invalid (not a non-empty string).
        """
        if isinstance(entity_id_raw, str) and entity_id_raw.strip():
            return entity_id_raw.strip()
        return None # Return None for invalid input like None, empty string, or non-string types

    def get_sensor_entity_id(self, internal_key: str) -> Optional[str]:
        """
        Gets the configured entity ID for a specific internal sensor key (e.g., 'temperature').
        Reads the corresponding config key (e.g., CONF_TEMP_SENSOR) from the ConfigEntry data.

        Args:
            internal_key: The internal key used (e.g., 'temperature', 'humidity', 'wind_speed').

        Returns:
            The stripped entity ID string, or None if not configured or invalid.
        """
        config_key = self._sensor_configs.get(internal_key)
        if not config_key:
            # This indicates a programming error (mismatch between internal keys and _sensor_configs)
            _LOGGER.error(f"Internal error: No config key found for internal sensor key '{internal_key}'")
            return None

        entity_id_raw = self.entry.data.get(config_key)
        return self.strip_entity_id(entity_id_raw) # Use stripping helper

    def get_sensor_value(self, entity_id: str | None) -> Optional[float]:
        """
        Gets the current state of the specified entity ID and attempts to convert it to a float.

        Args:
            entity_id: The entity ID string to fetch the state from.

        Returns:
            The float value of the sensor's state, or None if the entity is not found,
            unavailable, unknown, or its state cannot be converted to a float.
        """
        if not entity_id:
            # Don't log here, expected if sensor is optional and not configured
            # _LOGGER.debug("No entity ID provided for get_sensor_value.")
            return None

        state = self.hass.states.get(entity_id)
        # Check if state object exists and has a valid state string
        if not state or state.state in ['unavailable', 'unknown', 'None', '', None]:
            _LOGGER.debug(f"Sensor '{entity_id}' is unavailable or has an invalid state: {state.state if state else 'Not found'}")
            return None

        # Attempt to convert the state string to a float
        try:
            # Handle potential non-numeric characters (e.g., units) robustly
            cleaned_state = str(state.state).split(" ")[0].replace(",", ".")
            value = float(cleaned_state)
            # Optional: Add range checks if necessary for specific sensors
            # if internal_key == 'temperature' and not (-50 <= value <= 60):
            #    _LOGGER.warning(f"Temperature value {value} for {entity_id} outside expected range.")
            #    return None
            return value
        except (ValueError, TypeError):
            # Log a warning if conversion fails
            _LOGGER.warning(f"Could not parse sensor value for '{entity_id}' to float: '{state.state}'")
            return None

    def collect_all_sensor_data_dict(self) -> Dict[str, Optional[float]]:
        """
        Collects the current values of all configured *external* sensors into a dictionary.
        This dictionary is typically used for ML feature engineering or logging.
        Keys are the internal names (e.g., 'temperature', 'humidity', 'wind_speed').
        Values are float or None.
        """
        sensor_data_dict: Dict[str, Optional[float]] = {}

        # Iterate through the configured external sensors
        for internal_key in self._sensor_configs.keys():
            entity_id = self.get_sensor_entity_id(internal_key)
            # get_sensor_value handles None entity_id gracefully
            sensor_data_dict[internal_key] = self.get_sensor_value(entity_id)

        _LOGGER.debug(f"Collected external sensor data: {sensor_data_dict}")
        return sensor_data_dict

    # Note: `collect_all_sensor_data` was removed as its logic was specific
    # to the old coordinator structure and mixed config data (capacity, power_entity)
    # with sensor readings. `collect_all_sensor_data_dict` is the cleaner replacement
    # for getting just the *external sensor readings*.

    async def wait_for_external_sensors(self, max_wait: int = 25) -> int:
        """
        Waits during startup for at least one configured external sensor to become available
        (i.e., provide a valid float value). Prevents immediate errors if sensors load slowly.

        Args:
            max_wait: Maximum time in seconds to wait.

        Returns:
            The number of external sensors that became available during the wait.
        """
        _LOGGER.info("Waiting for external sensors to become available (max %ds)...", max_wait)

        wait_interval = 2 # Check every 2 seconds
        total_waited = 0
        available_sensor_keys = set() # Track which sensors are available

        configured_external_sensors = [
             key for key in self._sensor_configs.keys() if self.get_sensor_entity_id(key)
        ]

        if not configured_external_sensors:
             _LOGGER.info("No external sensors configured, skipping wait.")
             return 0

        _LOGGER.debug(f"Configured external sensors to wait for: {configured_external_sensors}")

        while total_waited < max_wait:
            current_available_count = 0
            sensor_status_log = []

            # Check status of each configured external sensor
            for internal_key in configured_external_sensors:
                entity_id = self.get_sensor_entity_id(internal_key) # Re-get ID just in case? No, should be stable.
                value = self.get_sensor_value(entity_id) # Checks availability and parse

                if value is not None:
                    # Sensor is available and provides a valid number
                    current_available_count += 1
                    available_sensor_keys.add(internal_key)
                    sensor_status_log.append(f"{internal_key}=OK")
                else:
                    # Sensor not available or invalid state
                    sensor_status_log.append(f"{internal_key}=WAITING")

            # Log current status concisely
            _LOGGER.debug(f"Sensor availability check ({total_waited}s): [{', '.join(sensor_status_log)}]")

            # Check if at least one sensor is available
            if current_available_count > 0:
                _LOGGER.info(
                    f"At least one external sensor ({current_available_count}/{len(configured_external_sensors)}) "
                    f"became available after {total_waited}s. Proceeding."
                )
                return current_available_count # Return count of available sensors

            # Wait before next check
            await asyncio.sleep(wait_interval)
            total_waited += wait_interval

        # Loop finished without any sensor becoming available
        _LOGGER.warning(
            f"No external sensors became available after waiting {max_wait}s. "
            f"Integration will continue, but predictions might be less accurate initially."
        )
        return 0 # Return 0 if timeout reached
