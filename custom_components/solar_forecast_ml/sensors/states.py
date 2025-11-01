"""
State Sensor platform for Solar Forecast ML Integration.
Contains sensors that reflect the state of other entities.

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

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.event import async_track_state_change_event

from ..const import (
    DOMAIN, INTEGRATION_MODEL, SOFTWARE_VERSION, ML_VERSION,
    CONF_TEMP_SENSOR, CONF_WIND_SENSOR, CONF_RAIN_SENSOR,
    CONF_UV_SENSOR, CONF_LUX_SENSOR, CONF_POWER_ENTITY, CONF_SOLAR_YIELD_TODAY,
    CONF_HUMIDITY_SENSOR
)

_LOGGER = logging.getLogger(__name__)


# --- Base Class for Entity State Sensors ---
class BaseEntityStateSensor(SensorEntity):
    """
    Base class for sensors that track the state of another entity.
    - Tracks state changes of a source entity via events
    - Reports the raw state of the source entity
    - Always available to show its status
    - Event-driven updates (no polling required)
    """

    _attr_has_entity_name = True
    _attr_should_poll = False  # Event-driven, no polling needed
    # Default category is DIAGNOSTIC, can be overridden by subclasses
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        source_entity_id_key: Optional[str],  # Key to look up in entry.data
        unique_id_key: str,
        translation_key: str,
        icon: str
    ):
        """Initialize the state sensor."""
        self.hass = hass
        self.entry = entry
        self._source_entity_id_key = source_entity_id_key
        self._attr_unique_id = f"{entry.entry_id}_{unique_id_key}"
        self._attr_translation_key = translation_key
        self._attr_icon = icon
        self._source_entity_id: Optional[str] = None  # Will be set in async_added_to_hass

        # Common device info
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Solar Forecast ML",
            manufacturer="Zara-Toorox",
            model=INTEGRATION_MODEL,
            sw_version=f"SW {SOFTWARE_VERSION} | ML {ML_VERSION}",
        )

    @property
    def source_entity_id(self) -> Optional[str]:
        """Get the source entity ID from config entry data."""
        if self._source_entity_id is None and self._source_entity_id_key:
            entity_id_raw = self.entry.data.get(self._source_entity_id_key)
            if isinstance(entity_id_raw, str) and entity_id_raw.strip():
                self._source_entity_id = entity_id_raw.strip()
            else:
                # Explicitly set to empty string if not configured or invalid
                self._source_entity_id = ""
        # Return None if not configured, otherwise the stripped ID
        return self._source_entity_id if self._source_entity_id else None

    @property
    def available(self) -> bool:
        """This sensor is always available to show the state (or lack thereof)."""
        return True

    async def async_added_to_hass(self) -> None:
        """Register state change listener."""
        await super().async_added_to_hass()
        source_id = self.source_entity_id  # Use property to get ID
        if source_id:
            _LOGGER.debug(f"StateSensor {self.entity_id} tracking {source_id}")
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, [source_id], self._handle_sensor_update
                )
            )
            # Fetch initial state - False = no async_update call, just write state
            self.async_schedule_update_ha_state(False)
        else:
            _LOGGER.debug(f"StateSensor {self.entity_id} has no source entity configured ({self._source_entity_id_key})")
            # Update state to "Not configured" - False = no async_update call
            self.async_schedule_update_ha_state(False)

    @callback
    def _handle_sensor_update(self, event) -> None:
        """Handle state update from the source entity."""
        source_entity = event.data.get('entity_id')
        new_state = event.data.get('new_state')
        old_state = event.data.get('old_state')
        
        # Extract values for logging
        new_value = new_state.state if new_state else "None"
        old_value = old_state.state if old_state else "None"
        
        _LOGGER.debug(
            f"StateSensor {self.entity_id} received update from {source_entity}: "
            f"{old_value} Ã¢â€ â€™ {new_value}"
        )
        
        self.async_write_ha_state()

    @property
    def native_value(self) -> str:
        """Return the state of the source entity as a string."""
        source_id = self.source_entity_id
        if not source_id:
            return "Not configured"

        state = self.hass.states.get(source_id)
        if state is None:
            return "Entity not found"

        return state.state  # Return the raw state string

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return attributes about the source entity."""
        source_id = self.source_entity_id
        base_attrs = {
            "source_entity_id": source_id,
            "source_entity_configured_key": self._source_entity_id_key
        }

        if not source_id:
            base_attrs["status"] = "not_configured"
            return base_attrs

        state = self.hass.states.get(source_id)
        if state is None:
            base_attrs["status"] = "entity_not_found"
            base_attrs["state"] = None
            return base_attrs

        status = "ok"
        if state.state in ['unavailable', 'unknown']:
            status = state.state

        base_attrs.update({
            "status": status,
            "state": state.state,
            "unit_of_measurement": state.attributes.get('unit_of_measurement'),
            "last_updated": state.last_updated.isoformat() if state.last_updated else None,
            "last_changed": state.last_changed.isoformat() if state.last_changed else None,
        })
        return base_attrs


# --- Specific State Sensors ---

class ExternalTempSensor(BaseEntityStateSensor):
    """State sensor for the configured external temperature sensor."""
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        super().__init__(
            hass,
            entry,
            source_entity_id_key=CONF_TEMP_SENSOR,
            unique_id_key="external_temp_state",
            translation_key="external_temp_state",
            icon="mdi:thermometer-check"
        )


class ExternalHumiditySensor(BaseEntityStateSensor):
    """State sensor for the configured external humidity sensor."""
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        super().__init__(
            hass,
            entry,
            source_entity_id_key=CONF_HUMIDITY_SENSOR,
            unique_id_key="external_humidity_state",
            translation_key="external_humidity_state",
            icon="mdi:water-percent-alert"
        )


class ExternalWindSensor(BaseEntityStateSensor):
    """State sensor for the configured external wind sensor."""
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        super().__init__(
            hass,
            entry,
            source_entity_id_key=CONF_WIND_SENSOR,
            unique_id_key="external_wind_state",
            translation_key="external_wind_state",
            icon="mdi:weather-windy-variant"
        )


class ExternalRainSensor(BaseEntityStateSensor):
    """State sensor for the configured external rain sensor."""
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        super().__init__(
            hass,
            entry,
            source_entity_id_key=CONF_RAIN_SENSOR,
            unique_id_key="external_rain_state",
            translation_key="external_rain_state",
            icon="mdi:weather-rainy-check"
        )


class ExternalUVSensor(BaseEntityStateSensor):
    """State sensor for the configured external UV sensor."""
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        super().__init__(
            hass,
            entry,
            source_entity_id_key=CONF_UV_SENSOR,
            unique_id_key="external_uv_state",
            translation_key="external_uv_state",
            icon="mdi:sun-wireless-outline"
        )


class ExternalLuxSensor(BaseEntityStateSensor):
    """State sensor for the configured external illuminance sensor."""
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        super().__init__(
            hass,
            entry,
            source_entity_id_key=CONF_LUX_SENSOR,
            unique_id_key="external_lux_state",
            translation_key="external_lux_state",
            icon="mdi:brightness-5-check"
        )


# --- State Sensors for Core Entities ---

class PowerSensorStateSensor(BaseEntityStateSensor):
    """State sensor for the configured main power sensor."""
    # Override category to make it a normal sensor
    _attr_entity_category = None

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        super().__init__(
            hass,
            entry,
            source_entity_id_key=CONF_POWER_ENTITY,
            unique_id_key="power_sensor_state",
            translation_key="power_sensor_state",
            icon="mdi:flash-alert-outline"
        )


class YieldSensorStateSensor(BaseEntityStateSensor):
    """State sensor for the configured daily yield sensor."""
    # Override category to make it a normal sensor
    _attr_entity_category = None

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        super().__init__(
            hass,
            entry,
            source_entity_id_key=CONF_SOLAR_YIELD_TODAY,
            unique_id_key="yield_sensor_state",
            translation_key="yield_sensor_state",
            icon="mdi:counter"
        )
