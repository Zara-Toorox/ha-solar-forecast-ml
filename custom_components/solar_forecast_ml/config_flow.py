"""
Config flow for the Solar Forecast ML integration.
Defines the user interface for adding and configuring the integration.

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
from __future__ import annotations
from typing import Any
import voluptuous as vol
import logging

from homeassistant import config_entries
# --- HIER DIE KORREKTUR ---
from homeassistant.config_entries import OptionsFlowWithReload, SOURCE_RECONFIGURE # Added OptionsFlowWithReload
# --- ENDE KORREKTUR ---
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    # Config keys
    CONF_WEATHER_ENTITY,
    CONF_POWER_ENTITY,
    CONF_SOLAR_YIELD_TODAY,
    CONF_TOTAL_CONSUMPTION_TODAY,
    CONF_SOLAR_CAPACITY,
    CONF_RAIN_SENSOR,
    CONF_LUX_SENSOR,
    CONF_TEMP_SENSOR,
    CONF_WIND_SENSOR,
    CONF_UV_SENSOR,
    CONF_HUMIDITY_SENSOR, # <-- NEU (Block 3)
    # Options keys
    CONF_UPDATE_INTERVAL,
    CONF_DIAGNOSTIC,
    CONF_HOURLY,
    CONF_NOTIFY_STARTUP,
    CONF_NOTIFY_FORECAST,
    CONF_NOTIFY_LEARNING,
    CONF_NOTIFY_SUCCESSFUL_LEARNING,
    # Defaults / Constants needed
    DEFAULT_SOLAR_CAPACITY
)

_LOGGER = logging.getLogger(__name__)

# --- Helper function to safely get defaults ---
def _get_default(data: dict | None, key: str, default: Any = vol.UNDEFINED):
    """Safely get default value for schema."""
    if data is None:
        return default
    value = data.get(key)
    return value if value is not None and value != "" else default

# --- Schema Definition ---
def _get_base_schema(defaults: dict | None) -> vol.Schema:
    """Returns the base schema for user and reconfigure steps."""
    if defaults is None:
        defaults = {}

    return vol.Schema({
        vol.Required(
            CONF_WEATHER_ENTITY,
            default=_get_default(defaults, CONF_WEATHER_ENTITY, "")
        ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["weather"])),
        vol.Required(
            CONF_POWER_ENTITY,
            default=_get_default(defaults, CONF_POWER_ENTITY, "")
        ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
        vol.Required(
            CONF_SOLAR_YIELD_TODAY,
            default=_get_default(defaults, CONF_SOLAR_YIELD_TODAY, "")
        ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),

        vol.Optional(
            CONF_TOTAL_CONSUMPTION_TODAY,
            default=_get_default(defaults, CONF_TOTAL_CONSUMPTION_TODAY)
        ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),

        vol.Optional(
            CONF_SOLAR_CAPACITY,
            default=_get_default(defaults, CONF_SOLAR_CAPACITY, DEFAULT_SOLAR_CAPACITY)
        ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=1000.0)),

        vol.Optional(
            CONF_RAIN_SENSOR,
            default=_get_default(defaults, CONF_RAIN_SENSOR)
        ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
        vol.Optional(
            CONF_LUX_SENSOR,
            default=_get_default(defaults, CONF_LUX_SENSOR)
        ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
        vol.Optional(
            CONF_TEMP_SENSOR,
            default=_get_default(defaults, CONF_TEMP_SENSOR)
        ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
        vol.Optional(
            CONF_WIND_SENSOR,
            default=_get_default(defaults, CONF_WIND_SENSOR)
        ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
        vol.Optional(
            CONF_UV_SENSOR,
            default=_get_default(defaults, CONF_UV_SENSOR)
        ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
        # --- NEU (Block 3) ---
        vol.Optional(
            CONF_HUMIDITY_SENSOR,
            default=_get_default(defaults, CONF_HUMIDITY_SENSOR)
        ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
        # --- ENDE NEU ---
    })


@config_entries.HANDLERS.register(DOMAIN)
class SolarForecastMLConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handles the configuration flow for Solar Forecast ML."""
    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry):
        """Redirect users to the options flow handler."""
        return SolarForecastMLOptionsFlow(config_entry)

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the initial setup step."""
        errors = {}
        prefill_data = user_input if user_input is not None else {}

        if user_input is not None:
            # Basic validation
            if not user_input.get(CONF_WEATHER_ENTITY): errors[CONF_WEATHER_ENTITY] = "required"
            if not user_input.get(CONF_POWER_ENTITY): errors[CONF_POWER_ENTITY] = "required"
            if not user_input.get(CONF_SOLAR_YIELD_TODAY): errors[CONF_SOLAR_YIELD_TODAY] = "required"
            try:
                capacity = user_input.get(CONF_SOLAR_CAPACITY)
                if capacity is not None:
                     float_cap = float(capacity)
                     if not (0.1 <= float_cap <= 1000.0): errors[CONF_SOLAR_CAPACITY] = "invalid_capacity"
            except (ValueError, TypeError): errors[CONF_SOLAR_CAPACITY] = "invalid_input"

            if errors:
                return self.async_show_form(
                    step_id="user", data_schema=_get_base_schema(prefill_data), errors=errors
                )

            # --- Data Cleaning and Entry Creation ---
            unique_id = user_input[CONF_WEATHER_ENTITY].strip()
            await self.async_set_unique_id(unique_id)
            self._abort_if_unique_id_configured()

            cleaned_data = {}
            for key, value in user_input.items():
                if isinstance(value, str):
                    cleaned_value = value.strip()
                    cleaned_data[key] = cleaned_value if cleaned_value else ""
                elif key == CONF_SOLAR_CAPACITY:
                     cleaned_data[key] = value if value is not None else DEFAULT_SOLAR_CAPACITY
                elif value is None:
                    cleaned_data[key] = ""
                else:
                    cleaned_data[key] = value

            if CONF_SOLAR_CAPACITY not in cleaned_data or cleaned_data[CONF_SOLAR_CAPACITY] == "":
                 cleaned_data[CONF_SOLAR_CAPACITY] = DEFAULT_SOLAR_CAPACITY

            return self.async_create_entry(title="Solar Forecast ML", data=cleaned_data)

        # Show initial form
        return self.async_show_form(
            step_id="user",
            data_schema=_get_base_schema({CONF_SOLAR_CAPACITY: DEFAULT_SOLAR_CAPACITY}),
            errors={},
        )

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the reconfiguration step."""
        if self.source != SOURCE_RECONFIGURE: return self.async_abort(reason="not_reconfigure")
        entry = self.hass.config_entries.async_get_entry(self.context["entry_id"])
        if entry is None: return self.async_abort(reason="entry_not_found")

        errors = {}
        prefill_data = dict(entry.data)

        if user_input is not None:
            prefill_data.update(user_input)
            # Basic validation
            if not user_input.get(CONF_WEATHER_ENTITY, "").strip(): errors[CONF_WEATHER_ENTITY] = "required"
            if not user_input.get(CONF_POWER_ENTITY, "").strip(): errors[CONF_POWER_ENTITY] = "required"
            if not user_input.get(CONF_SOLAR_YIELD_TODAY, "").strip(): errors[CONF_SOLAR_YIELD_TODAY] = "required"
            try:
                capacity = user_input.get(CONF_SOLAR_CAPACITY)
                if capacity is not None and capacity != "":
                     float_cap = float(capacity)
                     if not (0.1 <= float_cap <= 1000.0): errors[CONF_SOLAR_CAPACITY] = "invalid_capacity"
            except (ValueError, TypeError): errors[CONF_SOLAR_CAPACITY] = "invalid_input"

            if errors:
                return self.async_show_form(
                    step_id="reconfigure", data_schema=_get_base_schema(prefill_data), errors=errors
                )

            # --- Data Cleaning and Entry Update ---
            new_unique_id = user_input.get(CONF_WEATHER_ENTITY, "").strip()
            old_unique_id = entry.unique_id or ""
            if new_unique_id != old_unique_id:
                # Check for conflicts before updating HA entry (HA might handle this too)
                if self._async_current_entries(include_ignore=False):
                    for existing_entry in self._async_current_entries(include_ignore=False):
                        if existing_entry.unique_id == new_unique_id and existing_entry.entry_id != entry.entry_id:
                             errors["base"] = "already_configured"
                             return self.async_show_form(
                                 step_id="reconfigure", data_schema=_get_base_schema(prefill_data), errors=errors
                             )
                # Update unique_id if needed (HA does this via async_update_reload_and_abort)

            cleaned_data = {}
            for key, value in user_input.items():
                if isinstance(value, str):
                    cleaned_value = value.strip()
                    cleaned_data[key] = cleaned_value
                elif key == CONF_SOLAR_CAPACITY:
                     if value is None or value == "": cleaned_data[key] = DEFAULT_SOLAR_CAPACITY
                     else: cleaned_data[key] = float(value)
                elif value is None: cleaned_data[key] = ""
                else: cleaned_data[key] = value

            if cleaned_data.get(CONF_SOLAR_CAPACITY) == "":
                 cleaned_data[CONF_SOLAR_CAPACITY] = DEFAULT_SOLAR_CAPACITY

            return self.async_update_reload_and_abort(
                entry, data=cleaned_data, reason="reconfigure_successful"
            )

        # Show reconfigure form prefilled
        return self.async_show_form(
            step_id="reconfigure", data_schema=_get_base_schema(prefill_data), errors=errors
        )


class SolarForecastMLOptionsFlow(OptionsFlowWithReload):
    """Handles the options flow with automatic reload after changes."""

    # No __init__ needed, self.config_entry is provided by base class

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        # This init is deprecated, remove it.
        # self.config_entry = config_entry
        # Instead, access via self.config_entry directly in methods.
        pass # Keep init empty or remove completely. Let's remove it.

    # def __init__(self, config_entry: config_entries.ConfigEntry) -> None: <-- REMOVE THIS METHOD
    #    """Initialize options flow."""
    #    self.config_entry = config_entry


    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage the options."""
        errors = {}
        if user_input is not None:
            # Validation
            interval = user_input.get(CONF_UPDATE_INTERVAL, 3600)
            try:
                interval_sec = int(interval)
                if not (300 <= interval_sec <= 86400): errors[CONF_UPDATE_INTERVAL] = "invalid_interval"
            except (ValueError, TypeError): errors[CONF_UPDATE_INTERVAL] = "invalid_input"

            if errors:
                 options_schema = self._get_options_schema()
                 return self.async_show_form(
                     step_id="init",
                     data_schema=self.add_suggested_values_to_schema(
                         options_schema, user_input or self.config_entry.options
                     ),
                     errors=errors,
                 )

            # Update options
            updated_options = {
                 **self.config_entry.options,
                 **user_input,
                 CONF_DIAGNOSTIC: user_input.get(CONF_DIAGNOSTIC, True)
            }
            return self.async_create_entry(title="", data=updated_options)

        # Show form
        options_schema = self._get_options_schema()
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                options_schema, self.config_entry.options
            ),
            errors=errors,
        )

    def _get_options_schema(self) -> vol.Schema:
         """Define the schema for the options form."""
         current_options = self.config_entry.options

         return vol.Schema({
             vol.Optional(CONF_UPDATE_INTERVAL, default=current_options.get(CONF_UPDATE_INTERVAL, 1800)): # Default 30 min
                 vol.All(vol.Coerce(int), vol.Range(min=300, max=86400)),
             vol.Optional(CONF_DIAGNOSTIC, default=current_options.get(CONF_DIAGNOSTIC, True)): bool,
             vol.Optional(CONF_HOURLY, default=current_options.get(CONF_HOURLY, False)): bool,
             vol.Optional(CONF_NOTIFY_STARTUP, default=current_options.get(CONF_NOTIFY_STARTUP, True)): bool,
             vol.Optional(CONF_NOTIFY_FORECAST, default=current_options.get(CONF_NOTIFY_FORECAST, False)): bool,
             vol.Optional(CONF_NOTIFY_LEARNING, default=current_options.get(CONF_NOTIFY_LEARNING, False)): bool,
             vol.Optional(CONF_NOTIFY_SUCCESSFUL_LEARNING, default=current_options.get(CONF_NOTIFY_SUCCESSFUL_LEARNING, True)): bool,
         })