# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""Config flow for SFML Stats integration. @zara

Simplified setup flow:
- Step 1: Welcome
- Step 2: All sensors on one page (grouped)
- Step 3: Helper check
- Step 4: Settings (billing, theme)
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
    CONF_DASHBOARD_STYLE,
    DEFAULT_GENERATE_WEEKLY,
    DEFAULT_GENERATE_MONTHLY,
    DEFAULT_AUTO_GENERATE,
    DEFAULT_THEME,
    DEFAULT_DASHBOARD_STYLE,
    THEME_DARK,
    THEME_LIGHT,
    DASHBOARD_STYLE_3D,
    DASHBOARD_STYLE_2D,
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
    CONF_SENSOR_PANEL2_POWER,
    CONF_SENSOR_PANEL3_POWER,
    CONF_SENSOR_PANEL4_POWER,
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
    CONF_BILLING_FIXED_PRICE,
    CONF_FEED_IN_TARIFF,
    DEFAULT_BILLING_START_DAY,
    DEFAULT_BILLING_START_MONTH,
    DEFAULT_BILLING_FIXED_PRICE,
    DEFAULT_FEED_IN_TARIFF,
    CONF_PANEL_GROUP_NAMES,
    CONF_FORECAST_ENTITY_1,
    CONF_FORECAST_ENTITY_2,
    CONF_FORECAST_ENTITY_1_NAME,
    CONF_FORECAST_ENTITY_2_NAME,
    DEFAULT_FORECAST_ENTITY_1_NAME,
    DEFAULT_FORECAST_ENTITY_2_NAME,
    # Consumer sensors (Wärmepumpe, Heizstab, Wallbox)
    CONF_SENSOR_HEATPUMP_POWER,
    CONF_SENSOR_HEATPUMP_DAILY,
    CONF_SENSOR_HEATPUMP_COP,
    CONF_SENSOR_HEATINGROD_POWER,
    CONF_SENSOR_HEATINGROD_DAILY,
    CONF_SENSOR_WALLBOX_POWER,
    CONF_SENSOR_WALLBOX_DAILY,
    CONF_SENSOR_WALLBOX_STATE,
)
from .sensor_helpers import check_and_suggest_helpers

_LOGGER = logging.getLogger(__name__)

# Month names for translations
MONTHS_DE = {
    1: "Januar", 2: "Februar", 3: "März", 4: "April",
    5: "Mai", 6: "Juni", 7: "Juli", 8: "August",
    9: "September", 10: "Oktober", 11: "November", 12: "Dezember"
}

MONTHS_EN = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# All sensor keys for configuration - used in both initial and options flow
ALL_SENSOR_KEYS = [
    # === SOLAR (4) ===
    CONF_SENSOR_SOLAR_POWER,
    CONF_SENSOR_SOLAR_TO_HOUSE,
    CONF_SENSOR_SOLAR_TO_BATTERY,
    CONF_SENSOR_SOLAR_YIELD_DAILY,
    # === GRID (7) ===
    CONF_SENSOR_GRID_TO_HOUSE,
    CONF_SENSOR_GRID_TO_BATTERY,
    CONF_SENSOR_HOUSE_TO_GRID,
    CONF_SENSOR_GRID_IMPORT_DAILY,
    CONF_SENSOR_GRID_IMPORT_YEARLY,
    CONF_SENSOR_SMARTMETER_IMPORT,
    CONF_SENSOR_SMARTMETER_EXPORT,
    # === BATTERY (6) ===
    CONF_SENSOR_BATTERY_SOC,
    CONF_SENSOR_BATTERY_POWER,
    CONF_SENSOR_BATTERY_TO_HOUSE,
    CONF_SENSOR_BATTERY_TO_GRID,
    CONF_SENSOR_BATTERY_CHARGE_SOLAR_DAILY,
    CONF_SENSOR_BATTERY_CHARGE_GRID_DAILY,
    # === HOME (1) ===
    CONF_SENSOR_HOME_CONSUMPTION,
    # === ANDERE (2) ===
    CONF_SENSOR_PRICE_TOTAL,
    CONF_WEATHER_ENTITY,
]

# Consumer sensor keys - Optional (Wärmepumpe, Heizstab, Wallbox)
CONSUMER_SENSOR_KEYS = [
    # === WÄRMEPUMPE (3) ===
    CONF_SENSOR_HEATPUMP_POWER,
    CONF_SENSOR_HEATPUMP_DAILY,
    CONF_SENSOR_HEATPUMP_COP,
    # === HEIZSTAB (2) ===
    CONF_SENSOR_HEATINGROD_POWER,
    CONF_SENSOR_HEATINGROD_DAILY,
    # === WALLBOX (3) ===
    CONF_SENSOR_WALLBOX_POWER,
    CONF_SENSOR_WALLBOX_DAILY,
    CONF_SENSOR_WALLBOX_STATE,
]


def _is_raspberry_pi() -> bool:
    """Check if the system is running on a Raspberry Pi. @zara"""
    try:
        machine = platform.machine().lower()
        if machine in ('armv7l', 'aarch64', 'armv6l'):
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    if 'raspberry pi' in cpuinfo or 'bcm' in cpuinfo:
                        return True
            except (FileNotFoundError, PermissionError):
                _LOGGER.warning(
                    "Cannot read /proc/cpuinfo, but ARM architecture detected (%s). "
                    "Assuming Raspberry Pi for safety.", machine
                )
                return True
        return False
    except Exception as e:
        _LOGGER.error("Error detecting Raspberry Pi: %s", e)
        return False


def _is_proxmox() -> bool:
    """Check if the system is running on Proxmox VE. @zara"""
    try:
        proxmox_indicators = [
            '/etc/pve',
            '/usr/bin/pvesh',
            '/usr/bin/pveversion',
        ]
        for indicator in proxmox_indicators:
            try:
                from pathlib import Path
                if Path(indicator).exists():
                    _LOGGER.info("Proxmox VE detected via %s", indicator)
                    return True
            except Exception:
                pass
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
    """Create a text selector that allows clearing/removing the entity. @zara"""
    return selector.TextSelector(
        selector.TextSelectorConfig(
            type=selector.TextSelectorType.TEXT,
        )
    )


class SFMLStatsConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for SFML Stats. @zara

    Simplified setup flow:
    Step 1: Welcome screen
    Step 2: All sensors on one page (grouped)
    Step 3: Helper check
    Step 4: Settings (billing, theme)
    """

    VERSION = 5

    def __init__(self) -> None:
        """Initialize the config flow. @zara"""
        self._data: dict[str, Any] = {}
        self._helper_yaml: str = ""

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Handle the initial step - Welcome screen. @zara"""
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        # Platform checks
        if _is_raspberry_pi():
            _LOGGER.error(
                "Installation on Raspberry Pi is not supported due to performance limitations."
            )
            return self.async_abort(reason="raspberry_pi_not_supported")

        if _is_proxmox():
            _LOGGER.warning(
                "Installation on Proxmox VE detected. Running HA directly on Proxmox is not recommended."
            )
            return self.async_abort(reason="proxmox_not_recommended")

        if user_input is not None:
            # User clicked "Next", proceed to sensor configuration
            return await self.async_step_sensors()

        # Show welcome screen
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({}),
        )

    async def async_step_sensors(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Handle sensor configuration - All sensors on one page. @zara"""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Filter out empty values and save
            for key, value in user_input.items():
                if value and isinstance(value, str) and value.strip():
                    self._data[key] = value.strip()
            # Check for missing kWh sensors
            return await self.async_step_helpers()

        # Build sensor schema with all sensors
        schema_dict = {}
        for key in ALL_SENSOR_KEYS:
            schema_dict[vol.Optional(key, default="")] = get_entity_selector_optional()

        return self.async_show_form(
            step_id="sensors",
            data_schema=vol.Schema(schema_dict),
            errors=errors,
        )

    async def async_step_helpers(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Handle helper check - Check for missing kWh sensors. @zara"""
        if user_input is not None:
            return await self.async_step_settings()

        # Check for missing daily sensors
        missing_helpers, self._helper_yaml = await check_and_suggest_helpers(
            self.hass,
            self._data,
        )

        # If no missing sensors, skip to settings
        if not missing_helpers:
            return await self.async_step_settings()

        # Show helper suggestion
        missing_names = [h.friendly_name for h in missing_helpers]

        return self.async_show_form(
            step_id="helpers",
            data_schema=vol.Schema({
                vol.Optional("show_yaml", default=False): bool,
            }),
            description_placeholders={
                "missing_count": str(len(missing_helpers)),
                "missing_sensors": ", ".join(missing_names),
                "yaml_config": self._helper_yaml,
            },
        )

    async def async_step_settings(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Handle settings - General settings. @zara"""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._data.update(user_input)
            # Set empty panel group names mapping
            self._data[CONF_PANEL_GROUP_NAMES] = {}
            return self.async_create_entry(
                title=NAME,
                data=self._data,
            )

        days = {i: str(i) for i in range(1, 29)}

        return self.async_show_form(
            step_id="settings",
            data_schema=vol.Schema({
                vol.Required(
                    CONF_AUTO_GENERATE,
                    default=DEFAULT_AUTO_GENERATE,
                ): bool,
                vol.Required(
                    CONF_THEME,
                    default=DEFAULT_THEME,
                ): vol.In({
                    THEME_DARK: "Dark",
                    THEME_LIGHT: "Light",
                }),
                vol.Required(
                    CONF_DASHBOARD_STYLE,
                    default=DEFAULT_DASHBOARD_STYLE,
                ): vol.In({
                    DASHBOARD_STYLE_3D: "3D Isometrisch",
                    DASHBOARD_STYLE_2D: "2D Klassisch",
                }),
                vol.Required(
                    CONF_BILLING_START_MONTH,
                    default=DEFAULT_BILLING_START_MONTH,
                ): vol.In(MONTHS_DE),
                vol.Required(
                    CONF_BILLING_START_DAY,
                    default=DEFAULT_BILLING_START_DAY,
                ): vol.In(days),
                vol.Required(
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
                vol.Required(
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

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> SFMLStatsOptionsFlow:
        """Get the options flow for this handler. @zara"""
        return SFMLStatsOptionsFlow(config_entry)


class SFMLStatsOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for SFML Stats. @zara

    Simplified menu structure:
    1. Sensoren - All sensor configuration
    2. Einstellungen - Theme, billing, reports
    3. Erweitert - Panels, group names
    """

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow. @zara"""
        self._config_entry = config_entry

    def _process_sensor_input(
        self,
        user_input: dict[str, Any],
        sensor_keys: list[str],
    ) -> dict[str, Any]:
        """Process sensor input and update config entry data. @zara"""
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

        WICHTIG: Verwende suggested_value statt default!
        Mit default= wird beim Löschen eines Feldes der alte Wert verwendet.
        Mit suggested_value= wird das Feld vorausgefüllt, aber Löschen funktioniert.
        """
        current = self._config_entry.data
        schema_dict = {}
        for key in sensor_keys:
            schema_dict[vol.Optional(
                key,
                description={"suggested_value": current.get(key, "")}
            )] = get_entity_selector_optional()
        return vol.Schema(schema_dict)

    async def async_step_init(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage the options - Simplified Menu. @zara"""
        if user_input is not None:
            next_step = user_input.get("menu_choice")
            if next_step == "sensors":
                return await self.async_step_sensors()
            elif next_step == "consumers":
                return await self.async_step_consumers()
            elif next_step == "settings":
                return await self.async_step_settings()
            elif next_step == "advanced":
                return await self.async_step_advanced()

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required("menu_choice", default="sensors"): vol.In({
                    "sensors": "Sensoren",
                    "consumers": "Verbraucher (WP, Heizstab, Wallbox)",
                    "settings": "Einstellungen",
                    "advanced": "Erweitert",
                }),
            }),
        )

    async def async_step_sensors(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage ALL sensor options on one page. @zara"""
        if user_input is not None:
            new_data = self._process_sensor_input(user_input, ALL_SENSOR_KEYS)
            self.hass.config_entries.async_update_entry(
                self._config_entry, data=new_data
            )
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="sensors",
            data_schema=self._build_sensor_schema(ALL_SENSOR_KEYS),
        )

    async def async_step_consumers(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage consumer sensor options (Wärmepumpe, Heizstab, Wallbox). @zara"""
        if user_input is not None:
            new_data = self._process_sensor_input(user_input, CONSUMER_SENSOR_KEYS)
            self.hass.config_entries.async_update_entry(
                self._config_entry, data=new_data
            )
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="consumers",
            data_schema=self._build_sensor_schema(CONSUMER_SENSOR_KEYS),
        )

    async def async_step_settings(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage settings - Theme, billing, reports. @zara"""
        if user_input is not None:
            new_data = {**self._config_entry.data, **user_input}
            self.hass.config_entries.async_update_entry(
                self._config_entry, data=new_data
            )
            return self.async_create_entry(title="", data={})

        current = self._config_entry.data
        days = {i: str(i) for i in range(1, 29)}

        return self.async_show_form(
            step_id="settings",
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
                vol.Required(
                    CONF_DASHBOARD_STYLE,
                    default=current.get(CONF_DASHBOARD_STYLE, DEFAULT_DASHBOARD_STYLE),
                ): vol.In({
                    DASHBOARD_STYLE_3D: "3D Isometrisch",
                    DASHBOARD_STYLE_2D: "2D Klassisch",
                }),
                vol.Required(
                    CONF_BILLING_START_MONTH,
                    default=current.get(CONF_BILLING_START_MONTH, DEFAULT_BILLING_START_MONTH),
                ): vol.In(MONTHS_DE),
                vol.Required(
                    CONF_BILLING_START_DAY,
                    default=current.get(CONF_BILLING_START_DAY, DEFAULT_BILLING_START_DAY),
                ): vol.In(days),
                vol.Required(
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
                vol.Required(
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
            }),
        )

    async def async_step_advanced(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Advanced options sub-menu. @zara"""
        if user_input is not None:
            next_step = user_input.get("menu_choice")
            if next_step == "panels":
                return await self.async_step_panels()
            elif next_step == "panel_group_names":
                return await self.async_step_panel_group_names()
            elif next_step == "forecast_comparison":
                return await self.async_step_forecast_comparison()

        return self.async_show_form(
            step_id="advanced",
            data_schema=vol.Schema({
                vol.Required("menu_choice", default="panels"): vol.In({
                    "panels": "PV-Panels",
                    "panel_group_names": "Panel-Gruppen Namen",
                    "forecast_comparison": "Prognose-Vergleich",
                }),
            }),
        )

    async def async_step_panels(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage panel sensor options. @zara"""
        panel_keys = [
            CONF_PANEL1_NAME, CONF_SENSOR_PANEL1_POWER,
            CONF_PANEL2_NAME, CONF_SENSOR_PANEL2_POWER,
            CONF_PANEL3_NAME, CONF_SENSOR_PANEL3_POWER,
            CONF_PANEL4_NAME, CONF_SENSOR_PANEL4_POWER,
        ]

        if user_input is not None:
            new_data = {**self._config_entry.data}
            for key in panel_keys:
                value = user_input.get(key)
                if value is None or (isinstance(value, str) and not value.strip()):
                    new_data.pop(key, None)
                else:
                    new_data[key] = value
            self.hass.config_entries.async_update_entry(
                self._config_entry, data=new_data
            )
            return self.async_create_entry(title="", data={})

        current = self._config_entry.data

        return self.async_show_form(
            step_id="panels",
            data_schema=vol.Schema({
                vol.Optional(CONF_PANEL1_NAME, default=current.get(CONF_PANEL1_NAME, DEFAULT_PANEL1_NAME)): str,
                vol.Optional(CONF_SENSOR_PANEL1_POWER, description={"suggested_value": current.get(CONF_SENSOR_PANEL1_POWER, "")}): get_entity_selector_optional(),
                vol.Optional(CONF_PANEL2_NAME, default=current.get(CONF_PANEL2_NAME, DEFAULT_PANEL2_NAME)): str,
                vol.Optional(CONF_SENSOR_PANEL2_POWER, description={"suggested_value": current.get(CONF_SENSOR_PANEL2_POWER, "")}): get_entity_selector_optional(),
                vol.Optional(CONF_PANEL3_NAME, default=current.get(CONF_PANEL3_NAME, DEFAULT_PANEL3_NAME)): str,
                vol.Optional(CONF_SENSOR_PANEL3_POWER, description={"suggested_value": current.get(CONF_SENSOR_PANEL3_POWER, "")}): get_entity_selector_optional(),
                vol.Optional(CONF_PANEL4_NAME, default=current.get(CONF_PANEL4_NAME, DEFAULT_PANEL4_NAME)): str,
                vol.Optional(CONF_SENSOR_PANEL4_POWER, description={"suggested_value": current.get(CONF_SENSOR_PANEL4_POWER, "")}): get_entity_selector_optional(),
            }),
        )

    async def async_step_panel_group_names(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage panel group name mappings. @zara"""
        if user_input is not None:
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

    async def async_step_forecast_comparison(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage forecast comparison entities for 7-day chart. @zara"""
        forecast_keys = [
            CONF_FORECAST_ENTITY_1,
            CONF_FORECAST_ENTITY_1_NAME,
            CONF_FORECAST_ENTITY_2,
            CONF_FORECAST_ENTITY_2_NAME,
        ]

        if user_input is not None:
            new_data = {**self._config_entry.data}
            for key in forecast_keys:
                value = user_input.get(key)
                if value is None or (isinstance(value, str) and not value.strip()):
                    new_data.pop(key, None)
                else:
                    new_data[key] = value
            self.hass.config_entries.async_update_entry(
                self._config_entry, data=new_data
            )
            return self.async_create_entry(title="", data={})

        current = self._config_entry.data

        return self.async_show_form(
            step_id="forecast_comparison",
            data_schema=vol.Schema({
                vol.Optional(
                    CONF_FORECAST_ENTITY_1,
                    description={"suggested_value": current.get(CONF_FORECAST_ENTITY_1, "")},
                ): get_entity_selector_optional(),
                vol.Optional(
                    CONF_FORECAST_ENTITY_1_NAME,
                    default=current.get(CONF_FORECAST_ENTITY_1_NAME, DEFAULT_FORECAST_ENTITY_1_NAME),
                ): str,
                vol.Optional(
                    CONF_FORECAST_ENTITY_2,
                    description={"suggested_value": current.get(CONF_FORECAST_ENTITY_2, "")},
                ): get_entity_selector_optional(),
                vol.Optional(
                    CONF_FORECAST_ENTITY_2_NAME,
                    default=current.get(CONF_FORECAST_ENTITY_2_NAME, DEFAULT_FORECAST_ENTITY_2_NAME),
                ): str,
            }),
        )
