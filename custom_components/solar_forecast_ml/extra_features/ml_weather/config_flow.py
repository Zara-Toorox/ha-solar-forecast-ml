# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML - ML Weather
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************
"""Config flow for ML Weather integration."""

import logging
import os
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult

from .const import (
    DOMAIN,
    CONF_DATA_PATH,
    CONF_NAME,
    DEFAULT_DATA_PATH,
)

_LOGGER = logging.getLogger(__name__)


class MLWeatherConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for ML Weather."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate the data path
            data_path = user_input.get(CONF_DATA_PATH, DEFAULT_DATA_PATH)

            # Check if file exists
            file_exists = await self.hass.async_add_executor_job(
                os.path.exists, data_path
            )

            if not file_exists:
                errors["base"] = "file_not_found"
            else:
                # Create unique ID based on name
                name = user_input.get(CONF_NAME, "ML Weather")
                await self.async_set_unique_id(f"ml_weather_{name.lower().replace(' ', '_')}")
                self._abort_if_unique_id_configured()

                return self.async_create_entry(
                    title=name,
                    data=user_input,
                )

        # Show the form
        data_schema = vol.Schema(
            {
                vol.Required(CONF_NAME, default="ML Weather"): str,
                vol.Required(CONF_DATA_PATH, default=DEFAULT_DATA_PATH): str,
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return MLWeatherOptionsFlow(config_entry)


class MLWeatherOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for ML Weather."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate the data path
            data_path = user_input.get(CONF_DATA_PATH, DEFAULT_DATA_PATH)

            file_exists = await self.hass.async_add_executor_job(
                os.path.exists, data_path
            )

            if not file_exists:
                errors["base"] = "file_not_found"
            else:
                return self.async_create_entry(title="", data=user_input)

        current_path = self.config_entry.data.get(CONF_DATA_PATH, DEFAULT_DATA_PATH)

        data_schema = vol.Schema(
            {
                vol.Required(CONF_DATA_PATH, default=current_path): str,
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=data_schema,
            errors=errors,
        )
