# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

# *****************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# Refactored: JSON replaced with DatabaseManager @zara
# *****************************************************************************

"""
Sensor Data Collector for Solar Forecast ML.
Central class for collecting and accessing configured external sensor data.
All state is managed via the coordinator - no direct file I/O.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from ..const import EXTERNAL_SENSOR_MAPPING

_LOGGER = logging.getLogger(__name__)


class SensorDataCollector:
    """Central class for collecting and accessing configured external sensor data. @zara

    This class collects live sensor values from Home Assistant entities.
    It does not perform any file I/O - all persistence is handled by the coordinator.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """Initialize the SensorDataCollector. @zara"""
        self.hass = hass
        self.entry = entry
        self._sensor_configs = EXTERNAL_SENSOR_MAPPING
        _LOGGER.debug("SensorDataCollector initialized with centralized sensor mapping.")

    @staticmethod
    def strip_entity_id(entity_id_raw: Any) -> Optional[str]:
        """Safely strips whitespace from a potential entity ID string. @zara"""
        if isinstance(entity_id_raw, str) and entity_id_raw.strip():
            return entity_id_raw.strip()
        return None

    def get_sensor_entity_id(self, internal_key: str) -> Optional[str]:
        """Gets the configured entity ID for a specific internal sensor key. @zara

        Args:
            internal_key: Internal key like 'temperature', 'humidity', etc.

        Returns:
            Entity ID string or None if not configured
        """
        config_key = self._sensor_configs.get(internal_key)
        if not config_key:
            _LOGGER.error(
                f"Internal error: No config key found for internal sensor key '{internal_key}'"
            )
            return None

        entity_id_raw = self.entry.data.get(config_key)
        return self.strip_entity_id(entity_id_raw)

    def get_sensor_value(self, entity_id: Optional[str]) -> Optional[float]:
        """Gets the current state of the specified entity ID. @zara

        Args:
            entity_id: Home Assistant entity ID

        Returns:
            Float value or None if unavailable
        """
        if not entity_id:
            return None

        state = self.hass.states.get(entity_id)

        if not state or state.state in ["unavailable", "unknown", "None", "", None]:
            _LOGGER.debug(
                f"Sensor '{entity_id}' is unavailable or has an invalid state: "
                f"{state.state if state else 'Not found'}"
            )
            return None

        try:
            # Clean state value and convert to float
            cleaned_state = str(state.state).split(" ")[0].replace(",", ".")
            value = float(cleaned_state)
            return value
        except (ValueError, TypeError):
            _LOGGER.warning(
                f"Could not parse sensor value for '{entity_id}' to float: '{state.state}'"
            )
            return None

    def get_sensor_value_by_key(self, internal_key: str) -> Optional[float]:
        """Get sensor value by internal key. @zara

        Args:
            internal_key: Internal key like 'temperature', 'humidity', etc.

        Returns:
            Float value or None if unavailable
        """
        entity_id = self.get_sensor_entity_id(internal_key)
        return self.get_sensor_value(entity_id)

    def collect_all_sensor_data_dict(self) -> Dict[str, Optional[float]]:
        """Collects the current values of all configured external sensors. @zara

        Returns:
            Dictionary mapping internal keys to their current values
        """
        sensor_data_dict: Dict[str, Optional[float]] = {}

        for internal_key in self._sensor_configs.keys():
            entity_id = self.get_sensor_entity_id(internal_key)
            sensor_data_dict[internal_key] = self.get_sensor_value(entity_id)

        _LOGGER.debug(f"Collected external sensor data: {sensor_data_dict}")
        return sensor_data_dict

    def get_available_sensors(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all configured sensors. @zara

        Returns:
            Dictionary with sensor status information
        """
        status = {}

        for internal_key in self._sensor_configs.keys():
            entity_id = self.get_sensor_entity_id(internal_key)

            if not entity_id:
                status[internal_key] = {
                    "configured": False,
                    "available": False,
                    "entity_id": None,
                    "value": None,
                }
                continue

            value = self.get_sensor_value(entity_id)
            status[internal_key] = {
                "configured": True,
                "available": value is not None,
                "entity_id": entity_id,
                "value": value,
            }

        return status

    def count_available_sensors(self) -> tuple[int, int]:
        """Count configured and available sensors. @zara

        Returns:
            Tuple of (available_count, configured_count)
        """
        configured = 0
        available = 0

        for internal_key in self._sensor_configs.keys():
            entity_id = self.get_sensor_entity_id(internal_key)
            if entity_id:
                configured += 1
                if self.get_sensor_value(entity_id) is not None:
                    available += 1

        return available, configured

    async def wait_for_external_sensors(self, max_wait: int = 25) -> int:
        """Waits during startup for at least one configured external sensor. @zara

        Args:
            max_wait: Maximum seconds to wait

        Returns:
            Number of available sensors after waiting
        """
        _LOGGER.info("Waiting for external sensors to become available (max %ds)...", max_wait)

        wait_interval = 2
        total_waited = 0
        available_sensor_keys = set()

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

            for internal_key in configured_external_sensors:
                entity_id = self.get_sensor_entity_id(internal_key)
                value = self.get_sensor_value(entity_id)

                if value is not None:
                    current_available_count += 1
                    available_sensor_keys.add(internal_key)
                    sensor_status_log.append(f"{internal_key}=OK")
                else:
                    sensor_status_log.append(f"{internal_key}=WAITING")

            _LOGGER.debug(
                f"Sensor availability check ({total_waited}s): [{', '.join(sensor_status_log)}]"
            )

            if current_available_count > 0:
                _LOGGER.info(
                    f"At least one external sensor ({current_available_count}/"
                    f"{len(configured_external_sensors)}) became available after {total_waited}s. "
                    "Proceeding."
                )
                return current_available_count

            await asyncio.sleep(wait_interval)
            total_waited += wait_interval

        _LOGGER.warning(
            f"No external sensors became available after waiting {max_wait}s. "
            "Integration will continue, but predictions might be less accurate initially."
        )
        return 0

    def get_weather_data(self) -> Dict[str, Optional[float]]:
        """Get current weather-related sensor data. @zara

        Returns:
            Dictionary with weather sensor values
        """
        weather_keys = [
            "temperature", "humidity", "wind_speed", "rain",
            "pressure", "solar_radiation", "illuminance"
        ]

        return {
            key: self.get_sensor_value_by_key(key)
            for key in weather_keys
            if key in self._sensor_configs
        }

    def get_power_data(self) -> Dict[str, Optional[float]]:
        """Get current power-related sensor data. @zara

        Returns:
            Dictionary with power sensor values
        """
        power_keys = ["current_power", "solar_yield_today"]

        return {
            key: self.get_sensor_value_by_key(key)
            for key in power_keys
            if key in self._sensor_configs
        }

    def validate_sensor_config(self) -> Dict[str, Any]:
        """Validate sensor configuration. @zara

        Returns:
            Validation result with details about issues
        """
        issues = []
        warnings = []

        # Check required sensors
        required_sensors = ["current_power", "solar_yield_today"]
        for sensor_key in required_sensors:
            entity_id = self.get_sensor_entity_id(sensor_key)
            if not entity_id:
                issues.append(f"Required sensor '{sensor_key}' not configured")
            elif self.get_sensor_value(entity_id) is None:
                warnings.append(f"Required sensor '{sensor_key}' currently unavailable")

        # Check optional but recommended sensors
        recommended_sensors = ["temperature", "solar_radiation"]
        for sensor_key in recommended_sensors:
            entity_id = self.get_sensor_entity_id(sensor_key)
            if not entity_id:
                warnings.append(f"Recommended sensor '{sensor_key}' not configured")

        available, configured = self.count_available_sensors()

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "available_sensors": available,
            "configured_sensors": configured,
            "status": self.get_available_sensors(),
        }
