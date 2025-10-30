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
from homeassistant.core import HomeAssistant, callback, State
from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.update_coordinator import CoordinatorEntity # Needed for coordinator access

from .const import (
    DOMAIN, INTEGRATION_MODEL, SOFTWARE_VERSION, ML_VERSION,
    CONF_TEMP_SENSOR, CONF_WIND_SENSOR, CONF_RAIN_SENSOR,
    CONF_UV_SENSOR, CONF_LUX_SENSOR, CONF_POWER_ENTITY, CONF_SOLAR_YIELD_TODAY,
    CONF_HUMIDITY_SENSOR # <-- NEU (Block 3)
)
from .coordinator import SolarForecastMLCoordinator
from .helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


# --- Base Class for Entity State Sensors ---
class BaseEntityStateSensor(CoordinatorEntity, SensorEntity): # Inherit CoordinatorEntity to access coordinator
    """
    Base class for sensors that track the state of another entity.
    - Tracks state changes of a source entity.
    - Reports the raw state of the source entity.
    - Always available to show its status.
    """

    _attr_has_entity_name = True
    # Default category is DIAGNOSTIC, can be overridden by subclasses
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        coordinator: SolarForecastMLCoordinator, # Accept coordinator
        entry: ConfigEntry,
        source_entity_id_key: Optional[str], # Key to look up in entry.data
        unique_id_key: str,
        name: str,
        icon: str
    ):
        """Initialize the state sensor."""
        super().__init__(coordinator) # Initialize CoordinatorEntity
        self.entry = entry
        self._source_entity_id_key = source_entity_id_key
        self._attr_unique_id = f"{entry.entry_id}_{unique_id_key}"
        self._attr_name = name
        self._attr_icon = icon
        self._source_entity_id: Optional[str] = None # Will be set in async_added_to_hass

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
        # Return "" if not configured, otherwise the stripped ID
        return self._source_entity_id if self._source_entity_id else None

    @property
    def available(self) -> bool:
        """This sensor is always available to show the state (or lack thereof)."""
        return True

    async def async_added_to_hass(self) -> None:
        """Register state change listener."""
        await super().async_added_to_hass()
        source_id = self.source_entity_id # Use property to get ID
        if source_id:
            _LOGGER.debug(f"StateSensor {self.entity_id} tracking {source_id}")
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, [source_id], self._handle_sensor_update
                )
            )
            # Fetch initial state
            self.async_schedule_update_ha_state(True)
        else:
            _LOGGER.debug(f"StateSensor {self.entity_id} has no source entity configured ({self._source_entity_id_key})")
            # Update state to "Not configured"
            self.async_schedule_update_ha_state(True)


    @callback
    def _handle_sensor_update(self, event) -> None:
        """Handle state update from the source entity."""
        _LOGGER.debug(f"StateSensor {self.entity_id} received update from {event.data.get('entity_id')}")
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

        return state.state # Return the raw state string

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
    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id_key=CONF_TEMP_SENSOR,
            unique_id_key="external_temp_state",
            name="External Temperature Sensor State",
            icon="mdi:thermometer-check"
        )

# --- KORREKTUR (Block 3) ---
# Ersetzt die alte Platzhalter-Klasse
class ExternalHumiditySensor(BaseEntityStateSensor):
    """State sensor for the configured external humidity sensor."""
    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id_key=CONF_HUMIDITY_SENSOR, # Verwendet die korrekte Konstante
            unique_id_key="external_humidity_state",
            name="External Humidity Sensor State",
            icon="mdi:water-percent-alert"
        )
# --- ENDE KORREKTUR ---

class ExternalWindSensor(BaseEntityStateSensor):
    """State sensor for the configured external wind sensor."""
    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id_key=CONF_WIND_SENSOR,
            unique_id_key="external_wind_state",
            name="External Wind Sensor State",
            icon="mdi:weather-windy-variant"
        )

class ExternalRainSensor(BaseEntityStateSensor):
    """State sensor for the configured external rain sensor."""
    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id_key=CONF_RAIN_SENSOR,
            unique_id_key="external_rain_state",
            name="External Rain Sensor State",
            icon="mdi:weather-rainy-check"
        )

class ExternalUVSensor(BaseEntityStateSensor):
    """State sensor for the configured external UV sensor."""
    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id_key=CONF_UV_SENSOR,
            unique_id_key="external_uv_state",
            name="External UV Sensor State",
            icon="mdi:sun-wireless-outline" # Changed icon slightly
        )

class ExternalLuxSensor(BaseEntityStateSensor):
    """State sensor for the configured external illuminance sensor."""
    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id_key=CONF_LUX_SENSOR,
            unique_id_key="external_lux_state",
            name="External Illuminance Sensor State",
            icon="mdi:brightness-5-check"
        )

# --- State Sensors for Core Entities ---

class PowerSensorStateSensor(BaseEntityStateSensor):
    """State sensor for the configured main power sensor."""
    # [FIX] Override category to make it a normal sensor
    _attr_entity_category = None

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id_key=CONF_POWER_ENTITY, # Use the correct config key
            unique_id_key="power_sensor_state",
            name="Power Sensor State",
            icon="mdi:flash-alert-outline" # Icon indicating status/check
        )

class YieldSensorStateSensor(BaseEntityStateSensor):
    """State sensor for the configured daily yield sensor."""
    # [FIX] Override category to make it a normal sensor
    _attr_entity_category = None

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id_key=CONF_SOLAR_YIELD_TODAY, # Use the correct config key
            unique_id_key="yield_sensor_state",
            name="Yield Sensor State",
            icon="mdi:counter" # Icon indicating status/check
        )