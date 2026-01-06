# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML - ESC
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************
"""ESC Binary Sensor Platform."""
from __future__ import annotations

import logging
from homeassistant.components.binary_sensor import BinarySensorEntity, BinarySensorDeviceClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    """Set up binary sensor platform."""
    if config_entry.data.get("sensor_type") != "binary_threshold":
        return
    config = config_entry.data
    async_add_entities([ESCThresholdBinarySensor(hass, config_entry, config)])
    _LOGGER.info(f"Setting up ESC binary sensor: {config.get('sensor_name')}")

class ESCThresholdBinarySensor(BinarySensorEntity):
    """Binary sensor for threshold alarms."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry, config: dict):
        self.hass = hass
        self._config = config
        self._attr_name = config.get("sensor_name", "Threshold Alarm")
        self._attr_unique_id = config_entry.entry_id
        self._source_sensor_id = config["source_sensor"]
        self._threshold = config["threshold"]
        self._above_threshold = config["above_threshold"]  # True: on if >, False: on if <
        self._attr_device_class = BinarySensorDeviceClass.PROBLEM if config.get("device_class") == "problem" else None
        self._attr_is_on = False
        self._attr_device_info = {
            "identifiers": {(DOMAIN, DOMAIN)},
            "name": "ESC Easy Sensor Creation",
        }
        self._attr_extra_state_attributes = {
            "source_sensor": self._source_sensor_id,
            "threshold": self._threshold,
            "direction": "above" if self._above_threshold else "below"
        }

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(
            async_track_state_change_event(self.hass, [self._source_sensor_id], self._handle_state_change)
        )
        await self._async_update_state()

    @callback
    def _handle_state_change(self, event):
        self.async_schedule_update_ha_state(True)

    async def async_update(self):
        await self._async_update_state()

    async def _async_update_state(self):
        if (state := self.hass.states.get(self._source_sensor_id)) and state.state not in ("unknown", "unavailable"):
            try:
                value = float(state.state)
                if (self._above_threshold and value > self._threshold) or (not self._above_threshold and value < self._threshold):
                    self._attr_is_on = True
                else:
                    self._attr_is_on = False
            except (ValueError, TypeError):
                _LOGGER.warning(f"Could not parse state of {self._source_sensor_id}.")
                self._attr_is_on = False
        else:
            self._attr_is_on = False