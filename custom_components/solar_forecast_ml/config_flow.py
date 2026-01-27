# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries

from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_ADAPTIVE_FORECAST_MODE,
    CONF_DIAGNOSTIC,
    CONF_ENABLE_TINY_LSTM,
    CONF_HAS_BATTERY,
    CONF_HOURLY,
    CONF_HUMIDITY_SENSOR,
    CONF_INVERTER_MAX_POWER,
    CONF_LUX_SENSOR,
    CONF_ML_ALGORITHM,
    CONF_NOTIFY_FORECAST,
    CONF_NOTIFY_LEARNING,
    CONF_NOTIFY_STARTUP,
    CONF_NOTIFY_SUCCESSFUL_LEARNING,
    CONF_PANEL_GROUP_AZIMUTH,
    CONF_PANEL_GROUP_ENERGY_SENSOR,
    CONF_PANEL_GROUP_NAME,
    CONF_PANEL_GROUP_POWER,
    CONF_PANEL_GROUP_TILT,
    CONF_PANEL_GROUPS,
    CONF_PIRATE_WEATHER_API_KEY,
    CONF_POWER_ENTITY,
    CONF_PRESSURE_SENSOR,
    CONF_RAIN_SENSOR,
    CONF_SOLAR_CAPACITY,
    CONF_SOLAR_RADIATION_SENSOR,
    CONF_SOLAR_TO_BATTERY_SENSOR,
    CONF_SOLAR_YIELD_TODAY,
    CONF_TEMP_SENSOR,
    CONF_TOTAL_CONSUMPTION_TODAY,
    CONF_UPDATE_INTERVAL,
    CONF_WIND_SENSOR,
    CONF_WINTER_MODE,
    CONF_ZERO_EXPORT_MODE,
    CONF_LEARNING_BACKUP_PROTECTION,
    DEFAULT_ADAPTIVE_FORECAST_MODE,
    DEFAULT_HAS_BATTERY,
    DEFAULT_LEARNING_BACKUP_PROTECTION,
    DEFAULT_ENABLE_TINY_LSTM,
    DEFAULT_INVERTER_MAX_POWER,
    DEFAULT_ML_ALGORITHM,
    DEFAULT_PANEL_AZIMUTH,
    DEFAULT_PANEL_TILT,
    DEFAULT_SOLAR_CAPACITY,
    DEFAULT_WINTER_MODE,
    DEFAULT_ZERO_EXPORT_MODE,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

def _get_default(data: dict | None, key: str, default: Any = vol.UNDEFINED):
    """Safely get default value for schema @zara"""
    if data is None:
        return default
    value = data.get(key)
    return value if value is not None and value != "" else default

def _get_base_schema(defaults: dict | None, is_reconfigure: bool = False) -> vol.Schema:
    """Returns the base schema for user and reconfigure steps @zara

    Args:
        defaults: Dictionary with default/suggested values
        is_reconfigure: If True, use suggested_value instead of default for optional
                       entity selectors, allowing them to be cleared
    """
    if defaults is None:
        defaults = {}

    # For optional entity selectors in reconfigure mode, we use suggested_value
    # instead of default, which allows the user to clear the field
    def optional_entity_key(key: str):
        """Create vol.Optional key with suggested_value for reconfigure mode."""
        value = defaults.get(key)
        if is_reconfigure and value:
            # Use description with suggested_value - this shows the value but allows clearing
            return vol.Optional(key, description={"suggested_value": value})
        elif value:
            return vol.Optional(key, default=value)
        else:
            return vol.Optional(key)

    return vol.Schema(
        {
            # NOTE: CONF_WEATHER_ENTITY removed - Open-Meteo is SINGLE SOURCE
            # NOTE: CONF_GRID_IMPORT/EXPORT removed - Battery Management removed
            vol.Required(
                CONF_POWER_ENTITY, default=_get_default(defaults, CONF_POWER_ENTITY, "")
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            vol.Required(
                CONF_SOLAR_YIELD_TODAY, default=_get_default(defaults, CONF_SOLAR_YIELD_TODAY, "")
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
            optional_entity_key(CONF_TOTAL_CONSUMPTION_TODAY): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["sensor"])
            ),
            vol.Optional(
                CONF_SOLAR_CAPACITY,
                default=_get_default(defaults, CONF_SOLAR_CAPACITY, DEFAULT_SOLAR_CAPACITY),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.1,
                    max=1000.0,
                    step=0.01,
                    mode=selector.NumberSelectorMode.BOX,
                    unit_of_measurement="kWp",
                )
            ),
            vol.Optional(
                CONF_INVERTER_MAX_POWER,
                default=_get_default(defaults, CONF_INVERTER_MAX_POWER, DEFAULT_INVERTER_MAX_POWER),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.0,
                    max=1000.0,
                    step=0.1,
                    mode=selector.NumberSelectorMode.BOX,
                    unit_of_measurement="kW",
                )
            ),
            optional_entity_key(CONF_RAIN_SENSOR): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["sensor"])
            ),
            optional_entity_key(CONF_LUX_SENSOR): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["sensor"])
            ),
            optional_entity_key(CONF_TEMP_SENSOR): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["sensor"])
            ),
            optional_entity_key(CONF_WIND_SENSOR): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["sensor"])
            ),
            optional_entity_key(CONF_HUMIDITY_SENSOR): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["sensor"])
            ),
            optional_entity_key(CONF_PRESSURE_SENSOR): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["sensor"])
            ),
            optional_entity_key(CONF_SOLAR_RADIATION_SENSOR): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["sensor"])
            ),
        }
    )

def _parse_panel_groups(panel_groups_str: str) -> list[dict]:
    """Parse panel groups from string format to list of dicts.

    Supported formats:
    - Old: "power_wp/azimuth/tilt" (e.g., "1200/180/30, 900/270/10")
    - New: "power_wp/azimuth/tilt/energy_sensor" (e.g., "870/180/47/sensor.pv_gruppe1_energy")

    The energy_sensor is optional and enables per-group learning.

    @zara
    """
    if not panel_groups_str or not panel_groups_str.strip():
        return []

    groups = []
    # Split by comma or newline
    entries = [e.strip() for e in panel_groups_str.replace('\n', ',').split(',') if e.strip()]

    for idx, entry in enumerate(entries):
        parts = [p.strip() for p in entry.split('/')]
        if len(parts) >= 3:
            try:
                power_wp = float(parts[0])
                azimuth = float(parts[1])
                tilt = float(parts[2])

                # Validate ranges
                if power_wp <= 0 or power_wp > 100000:
                    continue
                if azimuth < 0 or azimuth > 360:
                    continue
                if tilt < 0 or tilt > 90:
                    continue

                group_data = {
                    CONF_PANEL_GROUP_NAME: f"Gruppe {idx + 1}",
                    CONF_PANEL_GROUP_POWER: power_wp,
                    CONF_PANEL_GROUP_AZIMUTH: azimuth,
                    CONF_PANEL_GROUP_TILT: tilt,
                }

                # Optional: Parse energy sensor (4th parameter)
                if len(parts) >= 4 and parts[3]:
                    energy_sensor = parts[3].strip()
                    # Basic validation: must look like an entity_id
                    if "." in energy_sensor and len(energy_sensor) > 3:
                        group_data[CONF_PANEL_GROUP_ENERGY_SENSOR] = energy_sensor

                groups.append(group_data)
            except (ValueError, TypeError):
                continue

    return groups


def _format_panel_groups(panel_groups: list[dict]) -> str:
    """Format panel groups from list of dicts to string format.

    Includes energy_sensor if configured.

    @zara
    """
    if not panel_groups:
        return ""

    # Format numbers: use int if whole number, otherwise keep float precision
    def fmt(v):
        return str(int(v)) if v == int(v) else str(v)

    lines = []
    for group in panel_groups:
        power = group.get(CONF_PANEL_GROUP_POWER, 0)
        azimuth = group.get(CONF_PANEL_GROUP_AZIMUTH, DEFAULT_PANEL_AZIMUTH)
        tilt = group.get(CONF_PANEL_GROUP_TILT, DEFAULT_PANEL_TILT)
        energy_sensor = group.get(CONF_PANEL_GROUP_ENERGY_SENSOR)

        if energy_sensor:
            lines.append(f"{fmt(power)}/{fmt(azimuth)}/{fmt(tilt)}/{energy_sensor}")
        else:
            lines.append(f"{fmt(power)}/{fmt(azimuth)}/{fmt(tilt)}")

    return ", ".join(lines)


def _calculate_total_capacity_from_groups(panel_groups: list[dict]) -> float:
    """Calculate total capacity in kWp from panel groups.

    @zara
    """
    if not panel_groups:
        return DEFAULT_SOLAR_CAPACITY

    total_wp = sum(g.get(CONF_PANEL_GROUP_POWER, 0) for g in panel_groups)
    return round(total_wp / 1000.0, 2)  # Convert Wp to kWp


@config_entries.HANDLERS.register(DOMAIN)
class SolarForecastMLConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handles the configuration flow for Solar Forecast ML"""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow. @zara"""
        self._user_input: dict[str, Any] = {}
        self._reconfigure_entry: config_entries.ConfigEntry | None = None

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry):
        """Redirect users to the options flow handler @zara"""
        return SolarForecastMLOptionsFlow()

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the initial setup step @zara"""
        errors = {}
        prefill_data = user_input if user_input is not None else {}

        if user_input is not None:

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

            # NOTE: Weather entity check REMOVED - Open-Meteo is SINGLE SOURCE

            unique_id = f"solar_forecast_ml_{user_input.get(CONF_POWER_ENTITY, 'default')}"
            await self.async_set_unique_id(unique_id)
            self._abort_if_unique_id_configured()

            cleaned_data = {}
            # List of optional sensor keys - empty values should not be stored
            optional_sensor_keys = [
                CONF_TOTAL_CONSUMPTION_TODAY,
                CONF_RAIN_SENSOR,
                CONF_LUX_SENSOR,
                CONF_TEMP_SENSOR,
                CONF_WIND_SENSOR,
                CONF_HUMIDITY_SENSOR,
                CONF_PRESSURE_SENSOR,
                CONF_SOLAR_RADIATION_SENSOR,
            ]
            for key, value in user_input.items():
                if isinstance(value, str):
                    cleaned_value = value.strip()
                    # Skip empty optional sensors - don't store them at all
                    if not cleaned_value and key in optional_sensor_keys:
                        continue
                    cleaned_data[key] = cleaned_value if cleaned_value else ""
                elif key in (CONF_SOLAR_CAPACITY, CONF_INVERTER_MAX_POWER):
                    # NumberSelector returns float, ensure proper handling
                    try:
                        cleaned_data[key] = float(value) if value is not None else (
                            DEFAULT_SOLAR_CAPACITY if key == CONF_SOLAR_CAPACITY else DEFAULT_INVERTER_MAX_POWER
                        )
                    except (ValueError, TypeError):
                        cleaned_data[key] = DEFAULT_SOLAR_CAPACITY if key == CONF_SOLAR_CAPACITY else DEFAULT_INVERTER_MAX_POWER
                elif value is None:
                    # Skip None values for optional sensors
                    if key in optional_sensor_keys:
                        continue
                    cleaned_data[key] = ""
                else:
                    cleaned_data[key] = value

            if CONF_SOLAR_CAPACITY not in cleaned_data or cleaned_data[CONF_SOLAR_CAPACITY] == "":
                cleaned_data[CONF_SOLAR_CAPACITY] = DEFAULT_SOLAR_CAPACITY
            if CONF_INVERTER_MAX_POWER not in cleaned_data:
                cleaned_data[CONF_INVERTER_MAX_POWER] = DEFAULT_INVERTER_MAX_POWER

            # Store user input and proceed to panel groups configuration
            self._user_input = cleaned_data
            return await self.async_step_panel_groups()

        return self.async_show_form(
            step_id="user",
            data_schema=_get_base_schema({CONF_SOLAR_CAPACITY: DEFAULT_SOLAR_CAPACITY}),
            errors={},
        )

    async def async_step_panel_groups(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the panel groups configuration step @zara

        Allows users to configure multiple panel groups with different orientations.
        Format: power_wp/azimuth/tilt (e.g., "1200/180/30, 900/270/10")
        """
        errors = {}

        if user_input is not None:
            panel_groups_str = user_input.get("panel_groups_input", "").strip()

            # V12.9.1: Panel groups are now REQUIRED (min 1, max 4)
            if not panel_groups_str:
                errors["panel_groups_input"] = "panel_groups_required"
            else:
                # Parse the panel groups
                panel_groups = _parse_panel_groups(panel_groups_str)

                if not panel_groups:
                    errors["panel_groups_input"] = "invalid_panel_format"
                elif len(panel_groups) > 4:
                    errors["panel_groups_input"] = "too_many_panel_groups"
                else:
                    # Validate energy sensors if configured
                    sensor_errors = await self._validate_panel_group_sensors(panel_groups)
                    if sensor_errors:
                        errors["panel_groups_input"] = sensor_errors
                    else:
                        # Store panel groups and calculate total capacity
                        self._user_input[CONF_PANEL_GROUPS] = panel_groups
                        total_capacity = _calculate_total_capacity_from_groups(panel_groups)
                        self._user_input[CONF_SOLAR_CAPACITY] = total_capacity

                        _LOGGER.info(
                            "Panel groups configured: %d groups, total %.2f kWp",
                            len(panel_groups),
                            total_capacity
                        )

            if not errors:
                return self.async_create_entry(title="Solar Forecast ML", data=self._user_input)

        # Get existing panel groups if reconfiguring
        existing_groups = self._user_input.get(CONF_PANEL_GROUPS, [])
        default_value = _format_panel_groups(existing_groups) if existing_groups else ""

        # V12.9.1: Panel groups are now REQUIRED
        schema = vol.Schema({
            vol.Required("panel_groups_input", default=default_value): selector.TextSelector(
                selector.TextSelectorConfig(
                    multiline=True,
                    type=selector.TextSelectorType.TEXT,
                )
            ),
        })

        return self.async_show_form(
            step_id="panel_groups",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "example": "1200/180/30, 900/270/10",
                "format": "Leistung(Wp)/Azimut(°)/Neigung(°)",
            },
        )

    async def _validate_panel_group_sensors(
        self, panel_groups: list[dict]
    ) -> str | None:
        """Validate energy sensors configured for panel groups.

        Returns:
            Error string if validation fails, None if all sensors are valid

        @zara
        """
        for group in panel_groups:
            entity_id = group.get(CONF_PANEL_GROUP_ENERGY_SENSOR)
            if not entity_id:
                continue

            # Check if entity exists
            state = self.hass.states.get(entity_id)

            if state is None:
                return f"energy_sensor_not_found: {entity_id}"

            if state.state in ["unavailable", "unknown"]:
                # Allow unavailable sensors at config time (might become available later)
                _LOGGER.warning(
                    "Energy sensor %s is currently %s - will retry at runtime",
                    entity_id, state.state
                )
                continue

            # Try to parse as numeric
            try:
                float(state.state)
            except (ValueError, TypeError):
                return f"energy_sensor_not_numeric: {entity_id}"

            # Check unit (optional but recommended)
            unit = state.attributes.get("unit_of_measurement", "")
            if unit and unit.lower() not in ["kwh", "wh"]:
                _LOGGER.warning(
                    "Energy sensor %s has unexpected unit: %s (expected kWh or Wh)",
                    entity_id, unit
                )

        return None

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the reconfiguration step @zara"""
        if self.source != SOURCE_RECONFIGURE:
            return self.async_abort(reason="not_reconfigure")
        entry = self.hass.config_entries.async_get_entry(self.context["entry_id"])
        if entry is None:
            return self.async_abort(reason="entry_not_found")

        errors = {}
        prefill_data = dict(entry.data)

        if user_input is not None:
            prefill_data.update(user_input)

            power_entity = user_input.get(CONF_POWER_ENTITY, "")
            if isinstance(power_entity, str):
                power_entity = power_entity.strip()
            if not power_entity:
                errors[CONF_POWER_ENTITY] = "required"

            yield_entity = user_input.get(CONF_SOLAR_YIELD_TODAY, "")
            if isinstance(yield_entity, str):
                yield_entity = yield_entity.strip()
            if not yield_entity:
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
                    step_id="reconfigure", data_schema=_get_base_schema(prefill_data, is_reconfigure=True), errors=errors
                )

            new_unique_id = f"solar_forecast_ml_{user_input.get(CONF_POWER_ENTITY, 'default')}"
            old_unique_id = entry.unique_id or ""
            if new_unique_id != old_unique_id:
                # Check for duplicates with other entries
                for existing_entry in self._async_current_entries(include_ignore=False):
                    if (
                        existing_entry.unique_id == new_unique_id
                        and existing_entry.entry_id != entry.entry_id
                    ):
                        errors["base"] = "already_configured"
                        return self.async_show_form(
                            step_id="reconfigure",
                            data_schema=_get_base_schema(prefill_data, is_reconfigure=True),
                            errors=errors,
                        )
                # Update unique_id for this entry
                await self.async_set_unique_id(new_unique_id)

            cleaned_data = {}
            # List of optional sensor keys that can be removed
            optional_sensor_keys = [
                CONF_TOTAL_CONSUMPTION_TODAY,
                CONF_RAIN_SENSOR,
                CONF_LUX_SENSOR,
                CONF_TEMP_SENSOR,
                CONF_WIND_SENSOR,
                CONF_HUMIDITY_SENSOR,
                CONF_PRESSURE_SENSOR,
                CONF_SOLAR_RADIATION_SENSOR,
            ]
            for key, value in user_input.items():
                if isinstance(value, str):
                    cleaned_value = value.strip()
                    # Skip empty optional sensors - they should be removed, not stored as ""
                    if not cleaned_value and key in optional_sensor_keys:
                        continue  # Don't add to cleaned_data = effectively removes the key
                    cleaned_data[key] = cleaned_value
                elif key in (CONF_SOLAR_CAPACITY, CONF_INVERTER_MAX_POWER):
                    # NumberSelector returns float, ensure proper handling
                    try:
                        if value is None or value == "":
                            cleaned_data[key] = DEFAULT_SOLAR_CAPACITY if key == CONF_SOLAR_CAPACITY else DEFAULT_INVERTER_MAX_POWER
                        else:
                            cleaned_data[key] = float(value)
                    except (ValueError, TypeError):
                        cleaned_data[key] = DEFAULT_SOLAR_CAPACITY if key == CONF_SOLAR_CAPACITY else DEFAULT_INVERTER_MAX_POWER
                elif value is None:
                    # Skip None values for optional sensors
                    if key in optional_sensor_keys:
                        continue
                    cleaned_data[key] = ""
                else:
                    cleaned_data[key] = value

            if cleaned_data.get(CONF_SOLAR_CAPACITY) == "":
                cleaned_data[CONF_SOLAR_CAPACITY] = DEFAULT_SOLAR_CAPACITY
            if CONF_INVERTER_MAX_POWER not in cleaned_data:
                cleaned_data[CONF_INVERTER_MAX_POWER] = DEFAULT_INVERTER_MAX_POWER

            # Preserve existing panel groups if not reconfiguring them
            if CONF_PANEL_GROUPS in entry.data and CONF_PANEL_GROUPS not in cleaned_data:
                cleaned_data[CONF_PANEL_GROUPS] = entry.data[CONF_PANEL_GROUPS]

            # Store for panel groups step
            self._user_input = cleaned_data
            self._reconfigure_entry = entry
            return await self.async_step_reconfigure_panel_groups()

        return self.async_show_form(
            step_id="reconfigure", data_schema=_get_base_schema(prefill_data, is_reconfigure=True), errors=errors
        )

    async def async_step_reconfigure_panel_groups(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle panel groups reconfiguration step @zara"""
        errors = {}
        entry = getattr(self, '_reconfigure_entry', None)
        if entry is None:
            return self.async_abort(reason="entry_not_found")

        if user_input is not None:
            panel_groups_str = user_input.get("panel_groups_input", "").strip()

            # V12.9.1: Panel groups are now REQUIRED (min 1, max 4)
            if not panel_groups_str:
                errors["panel_groups_input"] = "panel_groups_required"
            else:
                panel_groups = _parse_panel_groups(panel_groups_str)

                if not panel_groups:
                    errors["panel_groups_input"] = "invalid_panel_format"
                elif len(panel_groups) > 4:
                    errors["panel_groups_input"] = "too_many_panel_groups"
                else:
                    # Validate energy sensors if configured
                    sensor_errors = await self._validate_panel_group_sensors(panel_groups)
                    if sensor_errors:
                        errors["panel_groups_input"] = sensor_errors
                    else:
                        self._user_input[CONF_PANEL_GROUPS] = panel_groups
                        total_capacity = _calculate_total_capacity_from_groups(panel_groups)
                        self._user_input[CONF_SOLAR_CAPACITY] = total_capacity

            if not errors:
                return self.async_update_reload_and_abort(
                    entry, data=self._user_input, reason="reconfigure_successful"
                )

        # Get existing panel groups
        existing_groups = self._user_input.get(CONF_PANEL_GROUPS, [])
        default_value = _format_panel_groups(existing_groups) if existing_groups else ""

        # V12.9.1: Panel groups are now REQUIRED
        schema = vol.Schema({
            vol.Required("panel_groups_input", default=default_value): selector.TextSelector(
                selector.TextSelectorConfig(
                    multiline=True,
                    type=selector.TextSelectorType.TEXT,
                )
            ),
        })

        return self.async_show_form(
            step_id="reconfigure_panel_groups",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "example": "1200/180/30, 900/270/10",
                "format": "Leistung(Wp)/Azimut(°)/Neigung(°)",
            },
        )

class SolarForecastMLOptionsFlow(OptionsFlow):
    """Handles the options flow for Solar Forecast ML"""

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage the options @zara"""
        errors = {}
        if user_input is not None:
            # Validate and convert update_interval to int
            interval = user_input.get(CONF_UPDATE_INTERVAL, 1800)
            try:
                interval_sec = int(float(interval))  # Handle both int and float inputs
                if not (300 <= interval_sec <= 86400):
                    errors[CONF_UPDATE_INTERVAL] = "invalid_interval"
                else:
                    user_input[CONF_UPDATE_INTERVAL] = interval_sec  # Store as int
            except (ValueError, TypeError):
                errors[CONF_UPDATE_INTERVAL] = "invalid_input"

            if errors:
                return self.async_show_form(
                    step_id="init",
                    data_schema=self._get_options_schema(),
                    errors=errors,
                )

            # Build updated options with only valid keys
            valid_keys = [
                CONF_UPDATE_INTERVAL,
                CONF_DIAGNOSTIC,
                CONF_HOURLY,
                CONF_NOTIFY_STARTUP,
                CONF_NOTIFY_FORECAST,
                CONF_NOTIFY_LEARNING,
                CONF_NOTIFY_SUCCESSFUL_LEARNING,
                CONF_ML_ALGORITHM,
                CONF_ENABLE_TINY_LSTM,
                CONF_PIRATE_WEATHER_API_KEY,
                CONF_ADAPTIVE_FORECAST_MODE,
                CONF_WINTER_MODE,
                CONF_LEARNING_BACKUP_PROTECTION,
                # V13.2.0: MPPT Throttling Detection
                CONF_ZERO_EXPORT_MODE,
                CONF_HAS_BATTERY,
                CONF_SOLAR_TO_BATTERY_SENSOR,
            ]
            updated_options = {
                **self.config_entry.options,
                **{k: v for k, v in user_input.items() if k in valid_keys},
            }
            # Remove empty API keys
            if not updated_options.get(CONF_PIRATE_WEATHER_API_KEY):
                updated_options.pop(CONF_PIRATE_WEATHER_API_KEY, None)

            return self.async_create_entry(title="", data=updated_options)

        options_schema = self._get_options_schema()

        return self.async_show_form(
            step_id="init",
            data_schema=options_schema,
            errors=errors,
        )

    def _get_options_schema(self) -> vol.Schema:
        """Define the schema for the options form @zara"""
        current_options = self.config_entry.options

        # Safely get update_interval with type conversion
        try:
            update_interval = int(current_options.get(CONF_UPDATE_INTERVAL, 1800))
            if not 300 <= update_interval <= 86400:
                update_interval = 1800
        except (ValueError, TypeError):
            update_interval = 1800

        return vol.Schema(
            {
                vol.Optional(
                    CONF_UPDATE_INTERVAL, default=update_interval
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=300,
                        max=86400,
                        step=60,
                        mode=selector.NumberSelectorMode.BOX,
                        unit_of_measurement="s",
                    )
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
                vol.Optional(
                    CONF_ML_ALGORITHM,
                    default=current_options.get(CONF_ML_ALGORITHM, DEFAULT_ML_ALGORITHM),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            selector.SelectOptionDict(value="auto", label="Automatic (Recommended)"),
                            selector.SelectOptionDict(value="ridge", label="Ridge Regression (Fast)"),
                            selector.SelectOptionDict(value="tiny_lstm", label="Neural Network (Better Accuracy)"),
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                        translation_key="ml_algorithm",
                    )
                ),
                vol.Optional(
                    CONF_ENABLE_TINY_LSTM,
                    default=current_options.get(CONF_ENABLE_TINY_LSTM, DEFAULT_ENABLE_TINY_LSTM),
                ): bool,
                # Adaptive Forecast Mode - autonome Mittags-Korrektur
                vol.Optional(
                    CONF_ADAPTIVE_FORECAST_MODE,
                    default=current_options.get(CONF_ADAPTIVE_FORECAST_MODE, DEFAULT_ADAPTIVE_FORECAST_MODE),
                ): bool,
                # V13.1: Winter Mode - verbesserte Algorithmen für niedrige Sonnenstände
                vol.Optional(
                    CONF_WINTER_MODE,
                    default=current_options.get(CONF_WINTER_MODE, DEFAULT_WINTER_MODE),
                ): bool,
                # V13.2: Learning Backup Protection - schützt Lerndaten vor Backup-Restore
                vol.Optional(
                    CONF_LEARNING_BACKUP_PROTECTION,
                    default=current_options.get(CONF_LEARNING_BACKUP_PROTECTION, DEFAULT_LEARNING_BACKUP_PROTECTION),
                ): bool,
                # =============================================================
                # V13.2.0: MPPT Throttling Detection
                # Verhindert ML-Training aus gedrosselten Produktionswerten
                # =============================================================
                # Toggle: Nulleinspeisung aktiv (z.B. Shelly Nulleinspeisung)
                vol.Optional(
                    CONF_ZERO_EXPORT_MODE,
                    default=current_options.get(CONF_ZERO_EXPORT_MODE, DEFAULT_ZERO_EXPORT_MODE),
                ): bool,
                # Toggle: Akku im System vorhanden
                vol.Optional(
                    CONF_HAS_BATTERY,
                    default=current_options.get(CONF_HAS_BATTERY, DEFAULT_HAS_BATTERY),
                ): bool,
                # Sensor: Solar-zu-Akku Leistung (nur wenn has_battery = true)
                vol.Optional(
                    CONF_SOLAR_TO_BATTERY_SENSOR,
                    description={"suggested_value": current_options.get(CONF_SOLAR_TO_BATTERY_SENSOR, "")},
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=["sensor"],
                        device_class=["power"],
                        multiple=False,
                    )
                ),
                # Weather Expert API Keys
                vol.Optional(
                    CONF_PIRATE_WEATHER_API_KEY,
                    description={"suggested_value": current_options.get(CONF_PIRATE_WEATHER_API_KEY, "")},
                ): selector.TextSelector(
                    selector.TextSelectorConfig(
                        type=selector.TextSelectorType.PASSWORD,
                    )
                ),
            }
        )
