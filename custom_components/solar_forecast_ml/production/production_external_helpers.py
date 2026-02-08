# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
External Helpers for Solar Forecast ML.

Provides base classes and utilities for external sensor displays
with live updates.

@author zara
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.event import async_track_state_change_event

from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


def format_time_ago(last_changed: datetime) -> str:
    """Format timestamp as 'X min ago' or 'X h ago'. @zara"""
    now = dt_util.now()
    delta = now - last_changed

    seconds = delta.total_seconds()

    if seconds < 60:
        return "< 1 min ago"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} min ago"
    else:
        hours = int(seconds / 3600)
        return f"{hours} h ago"


class BaseExternalSensor:
    """Common base for external sensor displays with LIVE updates. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    hass: HomeAssistant

    def __init__(self, coordinator, entry: ConfigEntry, sensor_config: Dict[str, Any]):
        """Initialize the external sensor. @zara"""
        self._sensor_config = sensor_config
        self.entry = entry
        self.coordinator = coordinator

        self._attr_unique_id = f"{entry.entry_id}_{sensor_config['key']}"
        self._attr_name = sensor_config["name"]
        self._attr_icon = sensor_config["icon"]
        self._attr_device_class = sensor_config.get("device_class")
        self._attr_native_unit_of_measurement = sensor_config.get("unit")

    @staticmethod
    def strip_entity_id(entity_id_raw: Any) -> Optional[str]:
        """Safely strip an entity ID string, returns None if invalid. @zara"""
        if isinstance(entity_id_raw, str) and entity_id_raw:
            return entity_id_raw.strip()
        return None

    @property
    def _sensor_entity_id(self) -> Optional[str]:
        """Get the entity ID of the sensor to track from the ConfigEntry. @zara"""
        config_key = self._sensor_config.get("config_key")
        if not config_key:
            _LOGGER.error("Missing 'config_key' for sensor %s", self._attr_name)
            return None

        entity_id_raw = self.entry.data.get(config_key)
        return self.strip_entity_id(entity_id_raw)

    @property
    def available(self) -> bool:
        """External sensors are always available (they show their own status messages). @zara"""
        return True

    async def async_added_to_hass(self) -> None:
        """Register LIVE update listener. @zara"""
        await super().async_added_to_hass()

        sensor_entity_id = self._sensor_entity_id

        if sensor_entity_id:
            _LOGGER.debug(
                "Tracking external sensor %s for %s",
                sensor_entity_id,
                self._attr_name
            )
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, [sensor_entity_id], self._handle_external_sensor_update
                )
            )
        else:
            _LOGGER.debug("No external sensor configured for %s", self._attr_name)

    @callback
    def _handle_external_sensor_update(self, event) -> None:
        """Trigger update on external sensor change. @zara"""
        _LOGGER.debug(
            "External sensor %s updated, refreshing %s",
            event.data.get("entity_id"),
            self._attr_name
        )
        self.async_write_ha_state()

    @property
    def native_value(self) -> str:
        """Get value from the configured sensor with timestamp. @zara"""
        sensor_entity_id = self._sensor_entity_id

        if not sensor_entity_id:
            return "Not configured"

        state = self.hass.states.get(sensor_entity_id)
        if not state:
            return "Entity not found"

        try:
            if state.state in ["unavailable", "unknown", "none", None]:
                return "Unavailable"

            time_ago = format_time_ago(state.last_changed)

            unit = state.attributes.get(
                "unit_of_measurement", self._attr_native_unit_of_measurement or ""
            )

            return self._format_value(state.state, unit, time_ago)

        except Exception as e:
            _LOGGER.warning("Error reading %s: %s", self._sensor_config["name"], e)
            return "Error"

    def _get_unit(self, state: State) -> Optional[str]:
        """Determine the unit of the sensor. @zara"""
        unit_key = self._sensor_config.get("unit_key", "unit_of_measurement")
        default_unit = self._sensor_config.get("unit")

        return state.attributes.get(unit_key, default_unit)

    def _format_value(self, value: str, unit: Optional[str], time_ago: str) -> str:
        """Format sensor value for display. @zara"""
        format_string = self._sensor_config.get("format_string", "{value} {unit} ({time})")

        if "{value}" in format_string:
            result = format_string.replace("{value}", str(value))
            if unit:
                result = result.replace("{unit}", str(unit))
            else:
                result = result.replace(" {unit}", "")
            result = result.replace("{time}", time_ago)
            return result

        if unit:
            return f"{value} {unit} ({time_ago})"
        else:
            return f"{value} ({time_ago})"


class SensorValueExtractor:
    """Helper class to extract and validate sensor values. @zara"""

    def __init__(self, hass: HomeAssistant):
        """Initialize the sensor value extractor. @zara"""
        self.hass = hass

    def get_numeric_value(
        self,
        entity_id: Optional[str],
        default: float = 0.0
    ) -> float:
        """Get numeric value from a sensor entity. @zara"""
        if not entity_id:
            return default

        state = self.hass.states.get(entity_id)
        if not state or state.state in ["unavailable", "unknown", "none", None]:
            return default

        try:
            return float(state.state)
        except (ValueError, TypeError):
            return default

    def get_string_value(
        self,
        entity_id: Optional[str],
        default: str = ""
    ) -> str:
        """Get string value from a sensor entity. @zara"""
        if not entity_id:
            return default

        state = self.hass.states.get(entity_id)
        if not state or state.state in ["unavailable", "unknown", "none", None]:
            return default

        return str(state.state)

    def is_sensor_available(self, entity_id: Optional[str]) -> bool:
        """Check if a sensor is available and has a valid state. @zara"""
        if not entity_id:
            return False

        state = self.hass.states.get(entity_id)
        return (
            state is not None
            and state.state not in ["unavailable", "unknown", "none", None]
        )

    def get_sensor_with_timestamp(
        self,
        entity_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Get sensor value with timestamp information. @zara"""
        if not entity_id:
            return None

        state = self.hass.states.get(entity_id)
        if not state or state.state in ["unavailable", "unknown", "none", None]:
            return None

        try:
            value = float(state.state)
        except (ValueError, TypeError):
            value = state.state

        return {
            "value": value,
            "unit": state.attributes.get("unit_of_measurement"),
            "last_changed": state.last_changed,
            "last_updated": state.last_updated,
            "time_ago": format_time_ago(state.last_changed),
        }
