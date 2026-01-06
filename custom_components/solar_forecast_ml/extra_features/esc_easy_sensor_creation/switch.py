# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML - ESC
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************
"""ESC Switch Platform."""
from __future__ import annotations

import logging
from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    """Set up switch platform."""
    if config_entry.data.get("sensor_type") != "toggle_switch":
        return
    config = config_entry.data
    async_add_entities([ESCToggleSwitch(hass, config_entry, config)])
    _LOGGER.info(f"Setting up ESC switch: {config.get('sensor_name')}")

class ESCToggleSwitch(SwitchEntity):
    """Toggle switch for sensor control."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry, config: dict):
        self.hass = hass
        self._config = config
        self._attr_name = config.get("sensor_name", "Toggle Control")
        self._attr_unique_id = config_entry.entry_id
        self._target_entity = config["target_entity"]
        self._action = config.get("action", "pause")
        self._attr_is_on = True  # Default on
        self._attr_icon = "mdi:power-switch"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, DOMAIN)},
            "name": "ESC Easy Sensor Creation",
        }
        self._attr_extra_state_attributes = {
            "target_entity": self._target_entity,
            "action": self._action
        }

    async def async_turn_on(self, **kwargs):
        """Turn on (e.g. resume)."""
        self._attr_is_on = True
        # Trigger action on target (z.B. set attr 'paused' = False)
        self.hass.states.async_set(self._target_entity, self.hass.states.get(self._target_entity).state, {"esc_paused": False})
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs):
        """Turn off (e.g. pause)."""
        self._attr_is_on = False
        # Trigger action
        self.hass.states.async_set(self._target_entity, self.hass.states.get(self._target_entity).state, {"esc_paused": True})
        self.async_write_ha_state()

    async def async_update(self):
        """Update state from target."""
        target_state = self.hass.states.get(self._target_entity)
        if target_state and target_state.attributes.get("esc_paused"):
            self._attr_is_on = False
        else:
            self._attr_is_on = True