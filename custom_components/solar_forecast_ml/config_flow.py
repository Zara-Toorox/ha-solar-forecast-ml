"""
Configuration Flow for Solar Forecast ML Integration

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

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries

# --- HIER DIE KORREKTUR ---
from homeassistant.config_entries import (  # Added OptionsFlowWithReload
    SOURCE_RECONFIGURE,
    OptionsFlowWithReload,
)

# --- ENDE KORREKTUR ---
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import CONF_HUMIDITY_SENSOR  # <-- NEU (Block 3)
from .const import (  # Config keys; Options keys; Defaults / Constants needed; Battery Management (v9.0.0 - Watt-based); NEW v9.0.0 watt-based sensors
    CONF_BATTERY_CAPACITY,
    CONF_BATTERY_ENABLED,
    CONF_BATTERY_POWER_SENSOR,
    CONF_BATTERY_SOC_SENSOR,
    CONF_BATTERY_TEMPERATURE_SENSOR,
    CONF_DIAGNOSTIC,
    CONF_ELECTRICITY_COUNTRY,
    CONF_ELECTRICITY_ENABLED,
    CONF_GRID_CHARGE_POWER_SENSOR,
    CONF_GRID_EXPORT_SENSOR,
    CONF_GRID_EXPORT_TODAY,
    CONF_GRID_IMPORT_SENSOR,
    CONF_GRID_IMPORT_TODAY,
    CONF_HOURLY,
    CONF_HOUSE_CONSUMPTION_SENSOR,
    CONF_INVERTER_OUTPUT_SENSOR,
    CONF_LUX_SENSOR,
    CONF_NOTIFY_FORECAST,
    CONF_NOTIFY_LEARNING,
    CONF_NOTIFY_STARTUP,
    CONF_NOTIFY_SUCCESSFUL_LEARNING,
    CONF_POWER_ENTITY,
    CONF_RAIN_SENSOR,
    CONF_SOLAR_CAPACITY,
    CONF_SOLAR_PRODUCTION_SENSOR,
    CONF_SOLAR_YIELD_TODAY,
    CONF_TEMP_SENSOR,
    CONF_TOTAL_CONSUMPTION_TODAY,
    CONF_UPDATE_INTERVAL,
    CONF_UV_SENSOR,
    CONF_WEATHER_ENTITY,
    CONF_WIND_SENSOR,
    DEFAULT_BATTERY_CAPACITY,
    DEFAULT_ELECTRICITY_COUNTRY,
    DEFAULT_SOLAR_CAPACITY,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


# --- Helper function to safely get defaults ---
def _get_default(data: dict | None, key: str, default: Any = vol.UNDEFINED):
    """Safely get default value for schema"""
    if data is None:
        return default
    value = data.get(key)
    return value if value is not None and value != "" else default


# --- Schema Definition ---
def _get_base_schema(defaults: dict | None) -> vol.Schema:
    """Returns the base schema for user and reconfigure steps"""
    if defaults is None:
        defaults = {}

    return vol.Schema(
        {
            vol.Required(
                CONF_WEATHER_ENTITY, default=_get_default(defaults, CONF_WEATHER_ENTITY, "")
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["weather"])),
            vol.Required(
                CONF_POWER_ENTITY, default=_get_default(defaults, CONF_POWER_ENTITY, "")
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Required(
                CONF_SOLAR_YIELD_TODAY, default=_get_default(defaults, CONF_SOLAR_YIELD_TODAY, "")
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_TOTAL_CONSUMPTION_TODAY,
                default=_get_default(defaults, CONF_TOTAL_CONSUMPTION_TODAY),
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_GRID_IMPORT_TODAY, default=_get_default(defaults, CONF_GRID_IMPORT_TODAY)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_GRID_EXPORT_TODAY, default=_get_default(defaults, CONF_GRID_EXPORT_TODAY)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_SOLAR_CAPACITY,
                default=_get_default(defaults, CONF_SOLAR_CAPACITY, DEFAULT_SOLAR_CAPACITY),
            ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=1000.0)),
            vol.Optional(
                CONF_RAIN_SENSOR, default=_get_default(defaults, CONF_RAIN_SENSOR)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_LUX_SENSOR, default=_get_default(defaults, CONF_LUX_SENSOR)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_TEMP_SENSOR, default=_get_default(defaults, CONF_TEMP_SENSOR)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_WIND_SENSOR, default=_get_default(defaults, CONF_WIND_SENSOR)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_UV_SENSOR, default=_get_default(defaults, CONF_UV_SENSOR)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            # --- NEU (Block 3) ---
            vol.Optional(
                CONF_HUMIDITY_SENSOR, default=_get_default(defaults, CONF_HUMIDITY_SENSOR)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            # --- ENDE NEU ---
            # --- Battery Management (v9.0.0 - Watt-based) ---
            vol.Optional(
                CONF_BATTERY_CAPACITY,
                default=_get_default(defaults, CONF_BATTERY_CAPACITY, DEFAULT_BATTERY_CAPACITY),
            ): vol.All(vol.Coerce(float), vol.Range(min=0.5, max=1000.0)),
            # NEW v9.0.0: Watt-based sensors (REQUIRED for full functionality)
            vol.Optional(
                CONF_BATTERY_POWER_SENSOR, default=_get_default(defaults, CONF_BATTERY_POWER_SENSOR)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_BATTERY_SOC_SENSOR, default=_get_default(defaults, CONF_BATTERY_SOC_SENSOR)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_SOLAR_PRODUCTION_SENSOR,
                default=_get_default(defaults, CONF_SOLAR_PRODUCTION_SENSOR),
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_INVERTER_OUTPUT_SENSOR,
                default=_get_default(defaults, CONF_INVERTER_OUTPUT_SENSOR),
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_HOUSE_CONSUMPTION_SENSOR,
                default=_get_default(defaults, CONF_HOUSE_CONSUMPTION_SENSOR),
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_GRID_IMPORT_SENSOR, default=_get_default(defaults, CONF_GRID_IMPORT_SENSOR)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_GRID_EXPORT_SENSOR, default=_get_default(defaults, CONF_GRID_EXPORT_SENSOR)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_GRID_CHARGE_POWER_SENSOR,
                default=_get_default(defaults, CONF_GRID_CHARGE_POWER_SENSOR),
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Optional(
                CONF_BATTERY_TEMPERATURE_SENSOR,
                default=_get_default(defaults, CONF_BATTERY_TEMPERATURE_SENSOR),
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            # --- Electricity Price Configuration ---
            vol.Optional(
                CONF_ELECTRICITY_COUNTRY,
                default=_get_default(
                    defaults, CONF_ELECTRICITY_COUNTRY, DEFAULT_ELECTRICITY_COUNTRY
                ),
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=["DE", "AT"], mode=selector.SelectSelectorMode.DROPDOWN
                )
            ),
        }
    )


@config_entries.HANDLERS.register(DOMAIN)
class SolarForecastMLConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handles the configuration flow for Solar Forecast ML"""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry):
        """Redirect users to the options flow handler"""
        return SolarForecastMLOptionsFlow()

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the initial setup step"""
        errors = {}
        prefill_data = user_input if user_input is not None else {}

        if user_input is not None:
            # Basic validation
            if not user_input.get(CONF_WEATHER_ENTITY):
                errors[CONF_WEATHER_ENTITY] = "required"
            if not user_input.get(CONF_POWER_ENTITY):
                errors[CONF_POWER_ENTITY] = "required"
            if not user_input.get(CONF_SOLAR_YIELD_TODAY):
                errors[CONF_SOLAR_YIELD_TODAY] = "required"
            try:
                capacity = user_input.get(CONF_SOLAR_CAPACITY)
                if capacity is not None:
                    float_cap = float(capacity)
                    if not (0.1 <= float_cap <= 1000.0):
                        errors[CONF_SOLAR_CAPACITY] = "invalid_capacity"
            except (ValueError, TypeError):
                errors[CONF_SOLAR_CAPACITY] = "invalid_input"

            if errors:
                return self.async_show_form(
                    step_id="user", data_schema=_get_base_schema(prefill_data), errors=errors
                )

            # Check if Met.no is used - show warning if not
            weather_entity = user_input[CONF_WEATHER_ENTITY].strip()
            state = self.hass.states.get(weather_entity)
            if state:
                # Check integration platform (met, dwd, openweathermap, etc.)
                integration_domain = state.entity_id.split(".")[1].split("_")[0] if state.entity_id else None

                # Alternative: Check via entity registry
                try:
                    from homeassistant.helpers import entity_registry as er
                    entity_reg = er.async_get(self.hass)
                    entity_entry = entity_reg.async_get(weather_entity)
                    if entity_entry:
                        integration_domain = entity_entry.platform
                except Exception:
                    pass  # Fallback to state check

                # Show warning if not Met.no
                if integration_domain and integration_domain.lower() not in ["met", "met_no"]:
                    # Store user_input in instance variable for continuation
                    self._user_input = user_input
                    return await self.async_step_weather_warning()

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

    async def async_step_weather_warning(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Show warning if Met.no is not used"""
        if user_input is not None:
            # User clicked "Continue Anyway" - proceed with setup
            if hasattr(self, "_user_input"):
                user_data = self._user_input
                delattr(self, "_user_input")  # Clean up

                # --- Data Cleaning and Entry Creation ---
                unique_id = user_data[CONF_WEATHER_ENTITY].strip()
                await self.async_set_unique_id(unique_id)
                self._abort_if_unique_id_configured()

                cleaned_data = {}
                for key, value in user_data.items():
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

        # Show warning form
        return self.async_show_form(
            step_id="weather_warning",
            data_schema=vol.Schema({}),
            description_placeholders={
                "warning_message": (
                    "⚠️ WARNUNG: Suboptimale Wetter-Integration\n\n"
                    "Du nutzt keine Met.no Integration!\n\n"
                    "Für beste Ergebnisse empfehlen wir **Met.no**, da andere "
                    "Integrationen (z.B. DWD) keine numerischen Wetterdaten liefern "
                    "(cloud_cover_percent, temperature_c).\n\n"
                    "**Folge:**\n"
                    "- Vorhersagen können ungenau sein (Über-Vorhersagen bei Regen)\n"
                    "- System nutzt condition-aware Fallbacks (weniger präzise)\n\n"
                    "**Empfehlung:**\n"
                    "1. Home Assistant → Einstellungen → Integrationen\n"
                    "2. '+ Integration hinzufügen' → 'Met.no'\n"
                    "3. Standort auswählen (kostenlos, kein API-Key nötig)\n"
                    "4. Diese Integration neu konfigurieren mit Met.no\n\n"
                    "Möchtest du trotzdem fortfahren?"
                )
            },
        )

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the reconfiguration step"""
        if self.source != SOURCE_RECONFIGURE:
            return self.async_abort(reason="not_reconfigure")
        entry = self.hass.config_entries.async_get_entry(self.context["entry_id"])
        if entry is None:
            return self.async_abort(reason="entry_not_found")

        errors = {}
        prefill_data = dict(entry.data)

        if user_input is not None:
            prefill_data.update(user_input)
            # Basic validation
            if not user_input.get(CONF_WEATHER_ENTITY, "").strip():
                errors[CONF_WEATHER_ENTITY] = "required"
            if not user_input.get(CONF_POWER_ENTITY, "").strip():
                errors[CONF_POWER_ENTITY] = "required"
            if not user_input.get(CONF_SOLAR_YIELD_TODAY, "").strip():
                errors[CONF_SOLAR_YIELD_TODAY] = "required"
            try:
                capacity = user_input.get(CONF_SOLAR_CAPACITY)
                if capacity is not None and capacity != "":
                    float_cap = float(capacity)
                    if not (0.1 <= float_cap <= 1000.0):
                        errors[CONF_SOLAR_CAPACITY] = "invalid_capacity"
            except (ValueError, TypeError):
                errors[CONF_SOLAR_CAPACITY] = "invalid_input"

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
                        if (
                            existing_entry.unique_id == new_unique_id
                            and existing_entry.entry_id != entry.entry_id
                        ):
                            errors["base"] = "already_configured"
                            return self.async_show_form(
                                step_id="reconfigure",
                                data_schema=_get_base_schema(prefill_data),
                                errors=errors,
                            )
                # Update unique_id if needed (HA does this via async_update_reload_and_abort)

            cleaned_data = {}
            for key, value in user_input.items():
                if isinstance(value, str):
                    cleaned_value = value.strip()
                    cleaned_data[key] = cleaned_value
                elif key == CONF_SOLAR_CAPACITY:
                    if value is None or value == "":
                        cleaned_data[key] = DEFAULT_SOLAR_CAPACITY
                    else:
                        cleaned_data[key] = float(value)
                elif value is None:
                    cleaned_data[key] = ""
                else:
                    cleaned_data[key] = value

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
    """Handles the options flow with automatic reload after changes"""

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage the options"""
        errors = {}
        if user_input is not None:
            # Validation
            interval = user_input.get(CONF_UPDATE_INTERVAL, 3600)
            try:
                interval_sec = int(interval)
                if not (300 <= interval_sec <= 86400):
                    errors[CONF_UPDATE_INTERVAL] = "invalid_interval"
            except (ValueError, TypeError):
                errors[CONF_UPDATE_INTERVAL] = "invalid_input"

            if errors:
                return self.async_show_form(
                    step_id="init",
                    data_schema=self._get_options_schema(),
                    errors=errors,
                )

            # Update options (Boolean flags and settings go to options)
            # Battery sensor entities, capacity, and country are now configured via reconfigure flow (in data)
            updated_options = {
                **self.config_entry.options,
                **{
                    k: v
                    for k, v in user_input.items()
                    if k
                    in [
                        CONF_UPDATE_INTERVAL,
                        CONF_DIAGNOSTIC,
                        CONF_HOURLY,
                        CONF_NOTIFY_STARTUP,
                        CONF_NOTIFY_FORECAST,
                        CONF_NOTIFY_LEARNING,
                        CONF_NOTIFY_SUCCESSFUL_LEARNING,
                        CONF_BATTERY_ENABLED,  # Only battery enable flag
                        CONF_ELECTRICITY_ENABLED,  # Only electricity enable flag
                    ]
                },
            }

            # Return updated options to OptionsFlowWithReload
            return self.async_create_entry(title="", data=updated_options)

        # Show form with current options (defaults are already in schema)
        options_schema = self._get_options_schema()

        return self.async_show_form(
            step_id="init",
            data_schema=options_schema,
            errors=errors,
        )

    def _get_options_schema(self) -> vol.Schema:
        """Define the schema for the options form"""
        current_options = self.config_entry.options

        return vol.Schema(
            {
                vol.Optional(
                    CONF_UPDATE_INTERVAL, default=current_options.get(CONF_UPDATE_INTERVAL, 1800)
                ): vol.All(  # Default 30 min
                    vol.Coerce(int), vol.Range(min=300, max=86400)
                ),
                vol.Optional(
                    CONF_DIAGNOSTIC, default=current_options.get(CONF_DIAGNOSTIC, True)
                ): bool,
                vol.Optional(CONF_HOURLY, default=current_options.get(CONF_HOURLY, False)): bool,
                vol.Optional(
                    CONF_NOTIFY_STARTUP, default=current_options.get(CONF_NOTIFY_STARTUP, True)
                ): bool,
                vol.Optional(
                    CONF_NOTIFY_FORECAST, default=current_options.get(CONF_NOTIFY_FORECAST, False)
                ): bool,
                vol.Optional(
                    CONF_NOTIFY_LEARNING, default=current_options.get(CONF_NOTIFY_LEARNING, False)
                ): bool,
                vol.Optional(
                    CONF_NOTIFY_SUCCESSFUL_LEARNING,
                    default=current_options.get(CONF_NOTIFY_SUCCESSFUL_LEARNING, True),
                ): bool,
                # Battery Management - Only enable flags, capacity and entities are in reconfigure
                vol.Optional(
                    CONF_BATTERY_ENABLED, default=current_options.get(CONF_BATTERY_ENABLED, False)
                ): bool,
                # Electricity Prices - Only enable flag, country is in reconfigure
                vol.Optional(
                    CONF_ELECTRICITY_ENABLED,
                    default=current_options.get(CONF_ELECTRICITY_ENABLED, False),
                ): bool,
            }
        )
