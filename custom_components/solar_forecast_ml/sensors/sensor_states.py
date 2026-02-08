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
State sensors for tracking external sensor states in Solar Forecast ML.
These sensors monitor the state of configured external entities.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy, UnitOfPower
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.event import async_track_state_change_event

from ..const import (
    CONF_HUMIDITY_SENSOR,
    CONF_LUX_SENSOR,
    CONF_POWER_ENTITY,
    CONF_PRESSURE_SENSOR,
    CONF_RAIN_SENSOR,
    CONF_SOLAR_RADIATION_SENSOR,
    CONF_SOLAR_YIELD_TODAY,
    CONF_TEMP_SENSOR,
    CONF_WIND_SENSOR,
    DOMAIN,
    INTEGRATION_MODEL,
    AI_VERSION,
    SOFTWARE_VERSION,
)

_LOGGER = logging.getLogger(__name__)


class BaseEntityStateSensor(SensorEntity):
    """Base class for sensors that track the state of another entity. @zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        source_entity_id_key: Optional[str],
        unique_id_key: str,
        translation_key: str,
        icon: str,
    ):
        """Initialize the state sensor. @zara"""
        self.hass = hass
        self.entry = entry
        self._source_entity_id_key = source_entity_id_key
        self._attr_unique_id = f"{entry.entry_id}_ml_{unique_id_key}"
        self._attr_translation_key = translation_key
        self._attr_icon = icon
        self._source_entity_id: Optional[str] = None

        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Solar Forecast ML",
            manufacturer="Zara-Toorox",
            model=INTEGRATION_MODEL,
            sw_version=f"SW {SOFTWARE_VERSION} | AI {AI_VERSION}",
        )

    @property
    def source_entity_id(self) -> Optional[str]:
        """Get the source entity ID from config entry data. @zara"""
        if self._source_entity_id is None and self._source_entity_id_key:
            entity_id_raw = self.entry.data.get(self._source_entity_id_key)
            if isinstance(entity_id_raw, str) and entity_id_raw.strip():
                self._source_entity_id = entity_id_raw.strip()
            else:
                self._source_entity_id = ""

        return self._source_entity_id if self._source_entity_id else None

    @property
    def available(self) -> bool:
        """This sensor is always available to show the state or lack thereof. @zara"""
        return True

    async def async_added_to_hass(self) -> None:
        """Register state change listener. @zara"""
        await super().async_added_to_hass()
        source_id = self.source_entity_id
        if source_id:
            self.async_on_remove(
                async_track_state_change_event(self.hass, [source_id], self._handle_sensor_update)
            )
            self.async_schedule_update_ha_state(False)
        else:
            self.async_schedule_update_ha_state(False)

    @callback
    def _handle_sensor_update(self, event) -> None:
        """Handle state update from the source entity. @zara"""
        self.async_write_ha_state()

    @property
    def native_value(self) -> str:
        """Return the state of the source entity as a string. @zara"""
        source_id = self.source_entity_id
        if not source_id:
            return "Not configured"

        state = self.hass.states.get(source_id)
        if state is None:
            return "Entity not found"

        return state.state

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return attributes about the source entity. @zara"""
        source_id = self.source_entity_id
        base_attrs = {
            "source_entity_id": source_id,
            "source_entity_configured_key": self._source_entity_id_key,
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
        if state.state in ["unavailable", "unknown"]:
            status = state.state

        base_attrs.update(
            {
                "status": status,
                "state": state.state,
                "unit_of_measurement": state.attributes.get("unit_of_measurement"),
                "last_updated": state.last_updated.isoformat() if state.last_updated else None,
                "last_changed": state.last_changed.isoformat() if state.last_changed else None,
            }
        )
        return base_attrs


class ExternalSensorsStatusSensor(SensorEntity):
    """Sensor showing status of all configured external sensors. @zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """Initialize the external sensors status sensor. @zara"""
        self.hass = hass
        self.entry = entry
        self._attr_unique_id = f"{entry.entry_id}_ml_external_sensors_status"
        self._attr_translation_key = "external_sensors_status"
        self._attr_icon = "mdi:sensor-check"
        self._attr_name = "External Sensors Status"

        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Solar Forecast ML",
            manufacturer="Zara-Toorox",
            model=INTEGRATION_MODEL,
            sw_version=f"SW {SOFTWARE_VERSION} | AI {AI_VERSION}",
        )

        self._sensor_config = {
            "temperature": CONF_TEMP_SENSOR,
            "humidity": CONF_HUMIDITY_SENSOR,
            "wind_speed": CONF_WIND_SENSOR,
            "rain": CONF_RAIN_SENSOR,
            "illuminance": CONF_LUX_SENSOR,
            "pressure": CONF_PRESSURE_SENSOR,
            "solar_radiation": CONF_SOLAR_RADIATION_SENSOR,
        }

    @property
    def available(self) -> bool:
        """Sensor is always available. @zara"""
        return True

    async def async_added_to_hass(self) -> None:
        """Register state change listeners for all external sensors. @zara"""
        await super().async_added_to_hass()

        entity_ids_to_track = []
        for sensor_key in self._sensor_config.values():
            entity_id = self.entry.data.get(sensor_key)
            if entity_id and isinstance(entity_id, str) and entity_id.strip():
                entity_ids_to_track.append(entity_id.strip())

        if entity_ids_to_track:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, entity_ids_to_track, self._handle_sensor_update
                )
            )

        self.async_schedule_update_ha_state(False)

    @callback
    def _handle_sensor_update(self, event) -> None:
        """Handle state update from external sensors. @zara"""
        self.async_write_ha_state()

    @property
    def native_value(self) -> str:
        """Return overall status of external sensors. @zara"""
        configured_count = 0
        ok_count = 0
        unavailable_count = 0
        error_count = 0

        for sensor_key in self._sensor_config.values():
            entity_id = self.entry.data.get(sensor_key)
            if not entity_id or not isinstance(entity_id, str) or not entity_id.strip():
                continue

            configured_count += 1
            entity_id = entity_id.strip()
            state = self.hass.states.get(entity_id)

            if state is None:
                error_count += 1
            elif state.state in ["unavailable", "unknown"]:
                unavailable_count += 1
            else:
                ok_count += 1

        if configured_count == 0:
            return "No sensors configured"
        elif error_count > 0:
            return f"Error: {error_count} sensor(s) not found"
        elif unavailable_count == configured_count:
            return "All sensors unavailable"
        elif unavailable_count > 0:
            return f"Partial: {ok_count}/{configured_count} OK"
        else:
            return f"OK: {ok_count}/{configured_count} sensors"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return detailed status of each external sensor. @zara"""
        sensors_status = {}

        for sensor_name, sensor_key in self._sensor_config.items():
            entity_id = self.entry.data.get(sensor_key)

            if not entity_id or not isinstance(entity_id, str) or not entity_id.strip():
                sensors_status[sensor_name] = {
                    "status": "not_configured",
                    "entity_id": None,
                    "state": None,
                }
                continue

            entity_id = entity_id.strip()
            state = self.hass.states.get(entity_id)

            if state is None:
                sensors_status[sensor_name] = {
                    "status": "not_found",
                    "entity_id": entity_id,
                    "state": None,
                }
            elif state.state in ["unavailable", "unknown"]:
                sensors_status[sensor_name] = {
                    "status": state.state,
                    "entity_id": entity_id,
                    "state": state.state,
                }
            else:
                sensors_status[sensor_name] = {
                    "status": "ok",
                    "entity_id": entity_id,
                    "state": state.state,
                    "unit": state.attributes.get("unit_of_measurement"),
                }

        return {
            "sensors": sensors_status,
            "configured_count": sum(
                1 for s in sensors_status.values() if s["status"] != "not_configured"
            ),
            "ok_count": sum(1 for s in sensors_status.values() if s["status"] == "ok"),
            "unavailable_count": sum(
                1 for s in sensors_status.values() if s["status"] in ["unavailable", "unknown"]
            ),
            "error_count": sum(1 for s in sensors_status.values() if s["status"] == "not_found"),
        }


class PowerSensorStateSensor(BaseEntityStateSensor):
    """State sensor for the configured main power sensor. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_device_class = SensorDeviceClass.POWER
    _attr_native_unit_of_measurement = UnitOfPower.WATT
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(
            hass,
            entry,
            source_entity_id_key=CONF_POWER_ENTITY,
            unique_id_key="power_sensor_state",
            translation_key="power_sensor_state",
            icon="mdi:flash-alert-outline",
        )
        self._attr_name = "Power Sensor State"

    @property
    def native_value(self) -> Optional[float]:
        """Return the power value as float with units. @zara"""
        source_id = self.source_entity_id
        if not source_id:
            return None

        state = self.hass.states.get(source_id)
        if not state or state.state in ["unavailable", "unknown", "none", None, ""]:
            return None

        try:
            cleaned_state = str(state.state).split(" ")[0].replace(",", ".")
            return float(cleaned_state)
        except (ValueError, TypeError):
            _LOGGER.warning(f"Cannot parse power value from '{source_id}': {state.state}")
            return None


class YieldSensorStateSensor(BaseEntityStateSensor):
    """State sensor for the configured daily yield sensor. @zara"""

    _attr_entity_category = None
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_suggested_display_precision = 2

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(
            hass,
            entry,
            source_entity_id_key=CONF_SOLAR_YIELD_TODAY,
            unique_id_key="yield_sensor_state",
            translation_key="yield_sensor_state",
            icon="mdi:counter",
        )
        self._attr_name = "Yield Sensor State"

    @property
    def native_value(self) -> Optional[float]:
        """Return the yield value as float with units. @zara"""
        source_id = self.source_entity_id
        if not source_id:
            return None

        state = self.hass.states.get(source_id)
        if not state or state.state in ["unavailable", "unknown", "none", None, ""]:
            return None

        try:
            cleaned_state = str(state.state).split(" ")[0].replace(",", ".")
            return float(cleaned_state)
        except (ValueError, TypeError):
            _LOGGER.warning(f"Cannot parse yield value from '{source_id}': {state.state}")
            return None

