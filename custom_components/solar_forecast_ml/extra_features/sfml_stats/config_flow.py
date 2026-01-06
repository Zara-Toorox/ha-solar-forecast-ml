"""Config flow for SFML Stats integration. @zara

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
import platform
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    NAME,
    CONF_GENERATE_WEEKLY,
    CONF_GENERATE_MONTHLY,
    CONF_AUTO_GENERATE,
    CONF_THEME,
    DEFAULT_GENERATE_WEEKLY,
    DEFAULT_GENERATE_MONTHLY,
    DEFAULT_AUTO_GENERATE,
    DEFAULT_THEME,
    THEME_DARK,
    THEME_LIGHT,
    CONF_SENSOR_SOLAR_POWER,
    CONF_SENSOR_SOLAR_TO_HOUSE,
    CONF_SENSOR_SOLAR_TO_BATTERY,
    CONF_SENSOR_BATTERY_TO_HOUSE,
    CONF_SENSOR_BATTERY_TO_GRID,
    CONF_SENSOR_GRID_TO_HOUSE,
    CONF_SENSOR_GRID_TO_BATTERY,
    CONF_SENSOR_HOUSE_TO_GRID,
    CONF_SENSOR_BATTERY_SOC,
    CONF_SENSOR_BATTERY_POWER,
    CONF_SENSOR_HOME_CONSUMPTION,
    CONF_SENSOR_SOLAR_YIELD_DAILY,
    CONF_SENSOR_GRID_IMPORT_DAILY,
    CONF_SENSOR_GRID_IMPORT_YEARLY,
    CONF_SENSOR_BATTERY_CHARGE_SOLAR_DAILY,
    CONF_SENSOR_BATTERY_CHARGE_GRID_DAILY,
    CONF_SENSOR_PRICE_TOTAL,
    CONF_WEATHER_ENTITY,
    CONF_SENSOR_SMARTMETER_IMPORT,
    CONF_SENSOR_SMARTMETER_EXPORT,
    CONF_SENSOR_PANEL1_POWER,
    CONF_SENSOR_PANEL1_MAX_TODAY,
    CONF_SENSOR_PANEL2_POWER,
    CONF_SENSOR_PANEL2_MAX_TODAY,
    CONF_SENSOR_PANEL3_POWER,
    CONF_SENSOR_PANEL3_MAX_TODAY,
    CONF_SENSOR_PANEL4_POWER,
    CONF_SENSOR_PANEL4_MAX_TODAY,
    CONF_PANEL1_NAME,
    CONF_PANEL2_NAME,
    CONF_PANEL3_NAME,
    CONF_PANEL4_NAME,
    DEFAULT_PANEL1_NAME,
    DEFAULT_PANEL2_NAME,
    DEFAULT_PANEL3_NAME,
    DEFAULT_PANEL4_NAME,
    CONF_BILLING_START_DAY,
    CONF_BILLING_START_MONTH,
    CONF_BILLING_PRICE_MODE,
    CONF_BILLING_FIXED_PRICE,
    CONF_FEED_IN_TARIFF,
    PRICE_MODE_FIXED,
    PRICE_MODE_DYNAMIC,
    DEFAULT_BILLING_START_DAY,
    DEFAULT_BILLING_START_MONTH,
    DEFAULT_BILLING_PRICE_MODE,
    DEFAULT_BILLING_FIXED_PRICE,
    DEFAULT_FEED_IN_TARIFF,
    CONF_PANEL_GROUP_NAMES,
)

_LOGGER = logging.getLogger(__name__)


def _is_raspberry_pi() -> bool:
    """Check if the system is running on a Raspberry Pi. @zara"""
    try:
        # Überprüfung 1: CPU-Architektur
        machine = platform.machine().lower()
        if machine in ('armv7l', 'aarch64', 'armv6l'):
            # Könnte ein Raspberry Pi sein, weitere Überprüfung
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    # Raspberry Pi hat typischerweise BCM-Chips
                    if 'raspberry pi' in cpuinfo or 'bcm' in cpuinfo:
                        return True
            except (FileNotFoundError, PermissionError):
                # Wenn /proc/cpuinfo nicht lesbar ist, nur auf Architektur basieren
                # Vorsichtig sein und ARM-Systeme als potenziell Raspberry Pi behandeln
                _LOGGER.warning(
                    "Cannot read /proc/cpuinfo, but ARM architecture detected (%s). "
                    "Assuming Raspberry Pi for safety.", machine
                )
                return True
        return False
    except Exception as e:
        _LOGGER.error("Error detecting Raspberry Pi: %s", e)
        # Im Fehlerfall: lieber sicher sein
        return False


def _is_proxmox() -> bool:
    """Check if the system is running on Proxmox VE. @zara"""
    try:
        # Proxmox-spezifische Dateien/Verzeichnisse prüfen
        proxmox_indicators = [
            '/etc/pve',  # Proxmox VE Konfigurationsverzeichnis
            '/usr/bin/pvesh',  # Proxmox API Shell
            '/usr/bin/pveversion',  # Proxmox Version Tool
        ]

        for indicator in proxmox_indicators:
            try:
                from pathlib import Path
                if Path(indicator).exists():
                    _LOGGER.info("Proxmox VE detected via %s", indicator)
                    return True
            except Exception:
                pass

        # Zusätzlich: Überprüfung der Kernel-Version auf "pve"
        try:
            import os
            kernel_version = os.uname().release.lower()
            if 'pve' in kernel_version:
                _LOGGER.info("Proxmox VE detected via kernel version: %s", kernel_version)
                return True
        except Exception:
            pass

        return False
    except Exception as e:
        _LOGGER.error("Error detecting Proxmox: %s", e)
        return False


def get_entity_selector(domain: str = "sensor") -> selector.EntitySelector:
    """Create an entity selector for the specified domain. @zara"""
    return selector.EntitySelector(
        selector.EntitySelectorConfig(
            domain=domain,
            multiple=False,
        )
    )


def get_entity_selector_optional() -> selector.Selector:
    """Create a text selector that allows clearing/removing the entity. @zara

    Uses a text selector to allow empty values.
    This solves the issue where users cannot delete wrongly configured entities.
    """
    return selector.TextSelector(
        selector.TextSelectorConfig(
            type=selector.TextSelectorType.TEXT,
        )
    )


class SFMLStatsConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for SFML Stats. @zara"""

    VERSION = 2

    def __init__(self) -> None:
        """Initialize the config flow. @zara"""
        self._data: dict[str, Any] = {}

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Handle the initial step - Basic Settings. @zara"""
        errors: dict[str, str] = {}

        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        # Überprüfung: Raspberry Pi ist nicht unterstützt
        if _is_raspberry_pi():
            _LOGGER.error(
                "Installation on Raspberry Pi is not supported due to performance limitations "
                "with matplotlib and chart generation."
            )
            return self.async_abort(reason="raspberry_pi_not_supported")

        # Warnung: Proxmox VE wird nicht empfohlen
        if _is_proxmox():
            _LOGGER.warning(
                "Installation on Proxmox VE detected. Please note that running Home Assistant "
                "directly on Proxmox VE (instead of in a VM/LXC) is not recommended and may "
                "cause performance issues with matplotlib chart generation."
            )
            return self.async_abort(reason="proxmox_not_recommended")

        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_energy_flow()

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required(
                    CONF_AUTO_GENERATE,
                    default=DEFAULT_AUTO_GENERATE,
                ): bool,
                vol.Required(
                    CONF_GENERATE_WEEKLY,
                    default=DEFAULT_GENERATE_WEEKLY,
                ): bool,
                vol.Required(
                    CONF_GENERATE_MONTHLY,
                    default=DEFAULT_GENERATE_MONTHLY,
                ): bool,
                vol.Required(
                    CONF_THEME,
                    default=DEFAULT_THEME,
                ): vol.In({
                    THEME_DARK: "Dark",
                    THEME_LIGHT: "Light",
                }),
            }),
            errors=errors,
        )

    async def async_step_energy_flow(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Handle step 2 - Energy Flow Sensors. @zara"""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_battery()

        return self.async_show_form(
            step_id="energy_flow",
            data_schema=vol.Schema({
                vol.Optional(CONF_SENSOR_SOLAR_POWER): get_entity_selector(),
                vol.Optional(CONF_SENSOR_SOLAR_TO_HOUSE): get_entity_selector(),
                vol.Optional(CONF_SENSOR_SOLAR_TO_BATTERY): get_entity_selector(),
                vol.Optional(CONF_SENSOR_GRID_TO_HOUSE): get_entity_selector(),
                vol.Optional(CONF_SENSOR_GRID_TO_BATTERY): get_entity_selector(),
                vol.Optional(CONF_SENSOR_HOUSE_TO_GRID): get_entity_selector(),
                vol.Optional(CONF_SENSOR_SMARTMETER_IMPORT): get_entity_selector(),
                vol.Optional(CONF_SENSOR_SMARTMETER_EXPORT): get_entity_selector(),
            }),
            errors=errors,
        )

    async def async_step_battery(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Handle step 3 - Battery Sensors. @zara"""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_statistics()

        return self.async_show_form(
            step_id="battery",
            data_schema=vol.Schema({
                vol.Optional(CONF_SENSOR_BATTERY_SOC): get_entity_selector(),
                vol.Optional(CONF_SENSOR_BATTERY_POWER): get_entity_selector(),
                vol.Optional(CONF_SENSOR_BATTERY_TO_HOUSE): get_entity_selector(),
                vol.Optional(CONF_SENSOR_BATTERY_TO_GRID): get_entity_selector(),
                vol.Optional(CONF_SENSOR_HOME_CONSUMPTION): get_entity_selector(),
            }),
            errors=errors,
        )

    async def async_step_statistics(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Handle step 4 - Statistics Sensors. @zara"""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_panels()

        return self.async_show_form(
            step_id="statistics",
            data_schema=vol.Schema({
                vol.Optional(CONF_SENSOR_SOLAR_YIELD_DAILY): get_entity_selector(),
                vol.Optional(CONF_SENSOR_GRID_IMPORT_DAILY): get_entity_selector(),
                vol.Optional(CONF_SENSOR_GRID_IMPORT_YEARLY): get_entity_selector(),
                vol.Optional(CONF_SENSOR_BATTERY_CHARGE_SOLAR_DAILY): get_entity_selector(),
                vol.Optional(CONF_SENSOR_BATTERY_CHARGE_GRID_DAILY): get_entity_selector(),
                vol.Optional(CONF_SENSOR_PRICE_TOTAL): get_entity_selector(),
                vol.Optional(CONF_WEATHER_ENTITY): get_entity_selector("weather"),
            }),
            errors=errors,
        )

    async def async_step_panels(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Handle step 5 - Panel Sensors (optional). @zara"""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_billing()

        return self.async_show_form(
            step_id="panels",
            data_schema=vol.Schema({
                vol.Optional(CONF_PANEL1_NAME, default=DEFAULT_PANEL1_NAME): str,
                vol.Optional(CONF_SENSOR_PANEL1_POWER): get_entity_selector(),
                vol.Optional(CONF_SENSOR_PANEL1_MAX_TODAY): get_entity_selector(),
                vol.Optional(CONF_PANEL2_NAME, default=DEFAULT_PANEL2_NAME): str,
                vol.Optional(CONF_SENSOR_PANEL2_POWER): get_entity_selector(),
                vol.Optional(CONF_SENSOR_PANEL2_MAX_TODAY): get_entity_selector(),
                vol.Optional(CONF_PANEL3_NAME, default=DEFAULT_PANEL3_NAME): str,
                vol.Optional(CONF_SENSOR_PANEL3_POWER): get_entity_selector(),
                vol.Optional(CONF_SENSOR_PANEL3_MAX_TODAY): get_entity_selector(),
                vol.Optional(CONF_PANEL4_NAME, default=DEFAULT_PANEL4_NAME): str,
                vol.Optional(CONF_SENSOR_PANEL4_POWER): get_entity_selector(),
                vol.Optional(CONF_SENSOR_PANEL4_MAX_TODAY): get_entity_selector(),
            }),
            errors=errors,
        )

    async def async_step_billing(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Handle step 6 - Billing / Energy Balance Configuration. @zara"""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_panel_group_names()

        months = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }

        days = {i: str(i) for i in range(1, 29)}

        return self.async_show_form(
            step_id="billing",
            data_schema=vol.Schema({
                vol.Required(
                    CONF_BILLING_START_DAY,
                    default=DEFAULT_BILLING_START_DAY,
                ): vol.In(days),
                vol.Required(
                    CONF_BILLING_START_MONTH,
                    default=DEFAULT_BILLING_START_MONTH,
                ): vol.In(months),
                vol.Required(
                    CONF_BILLING_PRICE_MODE,
                    default=DEFAULT_BILLING_PRICE_MODE,
                ): vol.In({
                    PRICE_MODE_DYNAMIC: "Dynamic price (from Grid Price Monitor)",
                    PRICE_MODE_FIXED: "Fixed price (manual entry)",
                }),
                vol.Optional(
                    CONF_BILLING_FIXED_PRICE,
                    default=DEFAULT_BILLING_FIXED_PRICE,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=100,
                        step=0.01,
                        unit_of_measurement="ct/kWh",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_FEED_IN_TARIFF,
                    default=DEFAULT_FEED_IN_TARIFF,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=50,
                        step=0.1,
                        unit_of_measurement="ct/kWh",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
            }),
            errors=errors,
        )

    async def async_step_panel_group_names(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Handle step 7 - Panel Group Names (override Solar Forecast ML names). @zara"""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Parse the input: "Gruppe 1=String Süd, Gruppe 2=String West"
            names_mapping = {}
            raw_input = user_input.get("panel_group_names_input", "").strip()
            if raw_input:
                for entry in raw_input.split(","):
                    entry = entry.strip()
                    if "=" in entry:
                        parts = entry.split("=", 1)
                        old_name = parts[0].strip()
                        new_name = parts[1].strip()
                        if old_name and new_name:
                            names_mapping[old_name] = new_name

            self._data[CONF_PANEL_GROUP_NAMES] = names_mapping
            return self.async_create_entry(
                title=NAME,
                data=self._data,
            )

        # Show example format
        return self.async_show_form(
            step_id="panel_group_names",
            data_schema=vol.Schema({
                vol.Optional("panel_group_names_input", default=""): selector.TextSelector(
                    selector.TextSelectorConfig(
                        type=selector.TextSelectorType.TEXT,
                        multiline=True,
                    )
                ),
            }),
            errors=errors,
            description_placeholders={
                "example": "Gruppe 1=String Süd, Gruppe 2=String West"
            },
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> SFMLStatsOptionsFlow:
        """Get the options flow for this handler. @zara"""
        return SFMLStatsOptionsFlow(config_entry)


class SFMLStatsOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for SFML Stats. @zara"""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow. @zara"""
        self._config_entry = config_entry

    def _process_sensor_input(
        self,
        user_input: dict[str, Any],
        sensor_keys: list[str],
    ) -> dict[str, Any]:
        """Process sensor input and update config entry data. @zara

        Helper method to reduce code duplication in sensor configuration steps.
        Removes keys with empty values and updates keys with new values.

        Args:
            user_input: User input from form.
            sensor_keys: List of configuration keys to process.

        Returns:
            Updated configuration data.
        """
        new_data = {**self._config_entry.data}
        for key in sensor_keys:
            value = user_input.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                new_data.pop(key, None)
            else:
                new_data[key] = value
        return new_data

    def _build_sensor_schema(
        self,
        sensor_keys: list[str],
    ) -> vol.Schema:
        """Build schema for sensor configuration form. @zara

        Helper method to build consistent sensor configuration schemas.

        Args:
            sensor_keys: List of configuration keys for the schema.

        Returns:
            Voluptuous schema for the form.
        """
        current = self._config_entry.data
        schema_dict = {}
        for key in sensor_keys:
            schema_dict[vol.Optional(key, default=current.get(key, ""))] = (
                get_entity_selector_optional()
            )
        return vol.Schema(schema_dict)

    async def async_step_init(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage the options - Menu. @zara"""
        return self.async_show_menu(
            step_id="init",
            menu_options=["general", "energy_flow", "battery", "statistics", "panels", "billing", "panel_group_names"],
        )

    async def async_step_general(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage general options. @zara"""
        if user_input is not None:
            new_data = {**self._config_entry.data, **user_input}
            self.hass.config_entries.async_update_entry(
                self._config_entry, data=new_data
            )
            return self.async_create_entry(title="", data={})

        current = self._config_entry.data

        return self.async_show_form(
            step_id="general",
            data_schema=vol.Schema({
                vol.Required(
                    CONF_AUTO_GENERATE,
                    default=current.get(CONF_AUTO_GENERATE, DEFAULT_AUTO_GENERATE),
                ): bool,
                vol.Required(
                    CONF_GENERATE_WEEKLY,
                    default=current.get(CONF_GENERATE_WEEKLY, DEFAULT_GENERATE_WEEKLY),
                ): bool,
                vol.Required(
                    CONF_GENERATE_MONTHLY,
                    default=current.get(CONF_GENERATE_MONTHLY, DEFAULT_GENERATE_MONTHLY),
                ): bool,
                vol.Required(
                    CONF_THEME,
                    default=current.get(CONF_THEME, DEFAULT_THEME),
                ): vol.In({
                    THEME_DARK: "Dark",
                    THEME_LIGHT: "Light",
                }),
            }),
        )

    async def async_step_energy_flow(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage energy flow sensor options. @zara"""
        energy_flow_keys = [
            CONF_SENSOR_SOLAR_POWER, CONF_SENSOR_SOLAR_TO_HOUSE,
            CONF_SENSOR_SOLAR_TO_BATTERY, CONF_SENSOR_GRID_TO_HOUSE,
            CONF_SENSOR_GRID_TO_BATTERY, CONF_SENSOR_HOUSE_TO_GRID,
            CONF_SENSOR_SMARTMETER_IMPORT, CONF_SENSOR_SMARTMETER_EXPORT,
        ]

        if user_input is not None:
            new_data = self._process_sensor_input(user_input, energy_flow_keys)
            self.hass.config_entries.async_update_entry(
                self._config_entry, data=new_data
            )
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="energy_flow",
            data_schema=self._build_sensor_schema(energy_flow_keys),
        )

    async def async_step_battery(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage battery sensor options. @zara"""
        battery_keys = [
            CONF_SENSOR_BATTERY_SOC, CONF_SENSOR_BATTERY_POWER,
            CONF_SENSOR_BATTERY_TO_HOUSE, CONF_SENSOR_BATTERY_TO_GRID,
            CONF_SENSOR_HOME_CONSUMPTION,
        ]

        if user_input is not None:
            new_data = self._process_sensor_input(user_input, battery_keys)
            self.hass.config_entries.async_update_entry(
                self._config_entry, data=new_data
            )
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="battery",
            data_schema=self._build_sensor_schema(battery_keys),
        )

    async def async_step_statistics(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage statistics sensor options. @zara"""
        statistics_keys = [
            CONF_SENSOR_SOLAR_YIELD_DAILY, CONF_SENSOR_GRID_IMPORT_DAILY,
            CONF_SENSOR_GRID_IMPORT_YEARLY, CONF_SENSOR_BATTERY_CHARGE_SOLAR_DAILY,
            CONF_SENSOR_BATTERY_CHARGE_GRID_DAILY, CONF_SENSOR_PRICE_TOTAL,
            CONF_WEATHER_ENTITY,
        ]

        if user_input is not None:
            new_data = self._process_sensor_input(user_input, statistics_keys)
            self.hass.config_entries.async_update_entry(
                self._config_entry, data=new_data
            )
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="statistics",
            data_schema=self._build_sensor_schema(statistics_keys),
        )

    async def async_step_panels(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage panel sensor options. @zara"""
        panel_keys = [
            CONF_PANEL1_NAME, CONF_SENSOR_PANEL1_POWER, CONF_SENSOR_PANEL1_MAX_TODAY,
            CONF_PANEL2_NAME, CONF_SENSOR_PANEL2_POWER, CONF_SENSOR_PANEL2_MAX_TODAY,
            CONF_PANEL3_NAME, CONF_SENSOR_PANEL3_POWER, CONF_SENSOR_PANEL3_MAX_TODAY,
            CONF_PANEL4_NAME, CONF_SENSOR_PANEL4_POWER, CONF_SENSOR_PANEL4_MAX_TODAY,
        ]

        if user_input is not None:
            new_data = self._process_sensor_input(user_input, panel_keys)
            self.hass.config_entries.async_update_entry(
                self._config_entry, data=new_data
            )
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="panels",
            data_schema=self._build_panel_schema(),
        )

    def _build_panel_schema(self) -> vol.Schema:
        """Build schema for panel configuration form. @zara

        Panels have a special schema with name fields (str) and sensor fields.
        """
        current = self._config_entry.data
        panel_configs = [
            (CONF_PANEL1_NAME, DEFAULT_PANEL1_NAME, CONF_SENSOR_PANEL1_POWER, CONF_SENSOR_PANEL1_MAX_TODAY),
            (CONF_PANEL2_NAME, DEFAULT_PANEL2_NAME, CONF_SENSOR_PANEL2_POWER, CONF_SENSOR_PANEL2_MAX_TODAY),
            (CONF_PANEL3_NAME, DEFAULT_PANEL3_NAME, CONF_SENSOR_PANEL3_POWER, CONF_SENSOR_PANEL3_MAX_TODAY),
            (CONF_PANEL4_NAME, DEFAULT_PANEL4_NAME, CONF_SENSOR_PANEL4_POWER, CONF_SENSOR_PANEL4_MAX_TODAY),
        ]

        schema_dict = {}
        for name_key, default_name, power_key, max_key in panel_configs:
            schema_dict[vol.Optional(name_key, default=current.get(name_key, default_name))] = str
            schema_dict[vol.Optional(power_key, default=current.get(power_key, ""))] = get_entity_selector_optional()
            schema_dict[vol.Optional(max_key, default=current.get(max_key, ""))] = get_entity_selector_optional()

        return vol.Schema(schema_dict)

    async def async_step_billing(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage billing and energy balance options. @zara"""
        if user_input is not None:
            new_data = {**self._config_entry.data}
            new_data.update(user_input)
            self.hass.config_entries.async_update_entry(
                self._config_entry, data=new_data
            )
            return self.async_create_entry(title="", data={})

        current = self._config_entry.data

        months = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }

        days = {i: str(i) for i in range(1, 29)}

        schema_dict = {
            vol.Required(
                CONF_BILLING_START_DAY,
                default=current.get(CONF_BILLING_START_DAY, DEFAULT_BILLING_START_DAY),
            ): vol.In(days),
            vol.Required(
                CONF_BILLING_START_MONTH,
                default=current.get(CONF_BILLING_START_MONTH, DEFAULT_BILLING_START_MONTH),
            ): vol.In(months),
            vol.Required(
                CONF_BILLING_PRICE_MODE,
                default=current.get(CONF_BILLING_PRICE_MODE, DEFAULT_BILLING_PRICE_MODE),
            ): vol.In({
                PRICE_MODE_DYNAMIC: "Dynamic price (from Grid Price Monitor)",
                PRICE_MODE_FIXED: "Fixed price (manual entry)",
            }),
            vol.Optional(
                CONF_BILLING_FIXED_PRICE,
                default=current.get(CONF_BILLING_FIXED_PRICE, DEFAULT_BILLING_FIXED_PRICE),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=100,
                    step=0.01,
                    unit_of_measurement="ct/kWh",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_FEED_IN_TARIFF,
                default=current.get(CONF_FEED_IN_TARIFF, DEFAULT_FEED_IN_TARIFF),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=50,
                    step=0.1,
                    unit_of_measurement="ct/kWh",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
        }

        return self.async_show_form(
            step_id="billing",
            data_schema=vol.Schema(schema_dict),
        )

    async def async_step_panel_group_names(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage panel group name mappings (override Solar Forecast ML names). @zara"""
        if user_input is not None:
            # Parse the input: "Gruppe 1=String Süd, Gruppe 2=String West"
            names_mapping = {}
            raw_input = user_input.get("panel_group_names_input", "").strip()
            if raw_input:
                for entry in raw_input.split(","):
                    entry = entry.strip()
                    if "=" in entry:
                        parts = entry.split("=", 1)
                        old_name = parts[0].strip()
                        new_name = parts[1].strip()
                        if old_name and new_name:
                            names_mapping[old_name] = new_name

            new_data = {**self._config_entry.data}
            new_data[CONF_PANEL_GROUP_NAMES] = names_mapping
            self.hass.config_entries.async_update_entry(
                self._config_entry, data=new_data
            )
            return self.async_create_entry(title="", data={})

        current = self._config_entry.data
        # Convert dict back to string format for editing
        existing_mapping = current.get(CONF_PANEL_GROUP_NAMES, {})
        if existing_mapping and isinstance(existing_mapping, dict):
            default_value = ", ".join(f"{k}={v}" for k, v in existing_mapping.items())
        else:
            default_value = ""

        return self.async_show_form(
            step_id="panel_group_names",
            data_schema=vol.Schema({
                vol.Optional("panel_group_names_input", default=default_value): selector.TextSelector(
                    selector.TextSelectorConfig(
                        type=selector.TextSelectorType.TEXT,
                        multiline=True,
                    )
                ),
            }),
            description_placeholders={
                "example": "Gruppe 1=String Süd, Gruppe 2=String West"
            },
        )
