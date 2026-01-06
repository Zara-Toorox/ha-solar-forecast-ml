# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML - ESC
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************
"""Config flow for ESC Easy Sensor Creation."""
from __future__ import annotations
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.helpers import selector
from homeassistant.core import HomeAssistant

from .const import (
    DOMAIN, SENSOR_TYPE_SUM, SENSOR_TYPE_SQL, SENSOR_TYPE_KWH_HELPER,
    SENSOR_TYPE_DELTA, SENSOR_TYPE_BATTERY, SENSOR_TYPE_SFML, SENSOR_TYPE_SFML_PANEL,
    SQL_STAT_AVG_TODAY, SQL_STAT_AVG_MONTH, SQL_STAT_AVG_YEAR,
    SQL_STAT_MAX_TODAY, SQL_STAT_MAX_MONTH, SQL_STAT_MAX_YEAR,
    SQL_STAT_MIN_TODAY, SQL_STAT_MIN_MONTH, SQL_STAT_MIN_YEAR,
    DELTA_PERIOD_TODAY_YESTERDAY, DELTA_PERIOD_MONTH_PREV,
    BATTERY_MODE_CHARGE, BATTERY_MODE_DISCHARGE,
    DEVICE_CLASSES, DEVICE_CLASS_NONE, DEVICE_CLASS_POWER
)

async def _get_helper_names(hass: HomeAssistant) -> list[str]:
    """Return a list of existing integration helper names."""
    return [
        entry.data.get("name")
        for entry in hass.config_entries.async_entries("integration")
        if entry.data.get("name")
    ]


class ESCConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for ESC."""
    VERSION = 3
    data: dict

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        if user_input is not None:
            self.data = user_input
            sensor_type = self.data["sensor_type"]

            if sensor_type == SENSOR_TYPE_SQL:
                return await self.async_step_select_single_sensor()
            if sensor_type == SENSOR_TYPE_KWH_HELPER:
                return await self.async_step_kwh_config()
            if sensor_type == SENSOR_TYPE_DELTA:
                return await self.async_step_select_delta_sensor()
            if sensor_type == SENSOR_TYPE_BATTERY:
                return await self.async_step_battery_mode()
            if sensor_type == SENSOR_TYPE_SFML:
                return await self.async_step_sfml_config()
            if sensor_type == SENSOR_TYPE_SFML_PANEL:
                return await self.async_step_sfml_panel_count()
            if sensor_type == "binary_threshold":  # Neuer Typ für Binary
                return await self.async_step_binary_threshold()
            if sensor_type == "toggle_switch":  # Neuer für Switch
                return await self.async_step_toggle_target()
            # Default to SUM sensor
            return await self.async_step_select_sensors()

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required("sensor_type"): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=[
                        {"label": "kWh Sensor (erstellt Riemann Sensor für Energie-Dashboard)", "value": SENSOR_TYPE_KWH_HELPER},
                        {"label": "Summe (mehrere Sensoren addieren)", "value": SENSOR_TYPE_SUM},
                        {"label": "Verlaufs-Statistik (Durchschnitt, MIN, MAX)", "value": SENSOR_TYPE_SQL},
                        {"label": "Verlauf-Delta (z.B. heute vs. gestern)", "value": SENSOR_TYPE_DELTA},
                        {"label": "Akku Ladung/Entladung (positiv/negativ filtern + Riemann)", "value": SENSOR_TYPE_BATTERY},
                        {"label": "SFML Sensoren (Power + Yield + Daily)", "value": SENSOR_TYPE_SFML},
                        {"label": "SFML Panelgruppen (1-4 Strings)", "value": SENSOR_TYPE_SFML_PANEL},
                        {"label": "Binary Threshold (Alarm bei >/< Wert)", "value": "binary_threshold"},
                        {"label": "Toggle Switch (Sensor pausieren/resetten)", "value": "toggle_switch"},
                    ], mode=selector.SelectSelectorMode.LIST)
                ),
            })
        )

    async def async_step_kwh_config(self, user_input=None):
        """Handle the configuration for the kWh helper."""
        if user_input is not None:
            existing_names = await _get_helper_names(self.hass)
            if user_input["sensor_name"] in existing_names:
                return self.async_show_form(
                    step_id="kwh_config",
                    data_schema=self.add_suggested_values_to_schema(
                        vol.Schema({
                            vol.Required("source_sensor"): selector.EntitySelector(
                                selector.EntitySelectorConfig(domain="sensor")
                            ),
                            vol.Required("sensor_name"): str,
                        }),
                        user_input,
                    ),
                    errors={"base": "name_exists"},
                )

            # Create the native integration helper
            await self.hass.config_entries.flow.async_init(
                "integration",
                context={"source": "user"},
                data={
                    "name": user_input["sensor_name"],
                    "source": user_input["source_sensor"],
                    "unit_prefix": "k",
                    "unit_time": "h",
                    "method": "left",
                },
            )
            return self.async_abort(reason="kwh_helper_created")

        return self.async_show_form(
            step_id="kwh_config",
            data_schema=vol.Schema({
                vol.Required("source_sensor"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor")
                ),
                vol.Required("sensor_name", default="Täglicher Verbrauch"): str,
            }),
        )

    async def async_step_battery_mode(self, user_input=None):
        """Handle battery mode selection (charge/discharge)."""
        if user_input is not None:
            self.data["battery_mode"] = user_input["battery_mode"]
            return await self.async_step_battery_sensor()

        return self.async_show_form(
            step_id="battery_mode",
            data_schema=vol.Schema({
                vol.Required("battery_mode"): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=[
                        {"label": "Positive Werte (Ladung, z.B. PV/Solar)", "value": BATTERY_MODE_CHARGE},
                        {"label": "Negative Werte (Entladung, z.B. Akku/Auto)", "value": BATTERY_MODE_DISCHARGE},
                    ], mode=selector.SelectSelectorMode.LIST)
                ),
            })
        )

    async def async_step_battery_sensor(self, user_input=None):
        """Select source for battery."""
        if user_input is not None:
            self.data["source_sensor"] = user_input["source_sensor"]
            # Create filtered W-Sensor + Riemann Helper
            await self._create_battery_entities()
            return self.async_create_entry(
                title=f"Akku {self.data['battery_mode'].title()}",
                data=self.data
            )

        return self.async_show_form(
            step_id="battery_sensor",
            data_schema=vol.Schema({
                vol.Required("source_sensor"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor", device_class="power")  # Nur Power-Sensoren
                ),
            })
        )

    async def _create_battery_entities(self):
        """Create W-filter Sensor and Riemann Helper."""
        source = self.data["source_sensor"]
        mode = self.data["battery_mode"]
        name_base = f"Akku {mode.title()}"
        
        # 1. Filtered W-Sensor (nur positive/absolute negative)
        w_config = {
            "sensor_type": "battery_w",
            "source_sensors": [source],
            "battery_mode": mode,
            "sensor_name": f"{name_base} (W)",
            "device_class": DEVICE_CLASS_POWER
        }
        # Hier würdest du den W-Sensor via sensor.py erstellen, aber da's im Flow ist, speichere in data
        self.data["w_sensor_config"] = w_config
        
        # 2. Riemann Helper (kWh, left Riemann) – nutzt original Source (HA handhabt es; für Filter: Passe an filtered Entity an, wenn verfügbar)
        helper_name = f"{name_base} (kWh)"
        existing_names = await _get_helper_names(self.hass)
        if helper_name not in existing_names:
            await self.hass.config_entries.flow.async_init(
                "integration",
                context={"source": "user"},
                data={
                    "name": helper_name,
                    "source": source,  # Könnte auf filtered W geändert werden, sobald Entity-ID bekannt
                    "unit_prefix": "k",
                    "unit_time": "h",
                    "method": "left",
                    "state_class": "total_increasing"  # Für Energy-Dashboard
                },
            )

    async def async_step_sfml_config(self, user_input=None):
        """Handle SFML sensor configuration."""
        if user_input is not None:
            self.data["source_sensor"] = user_input["source_sensor"]
            self.data["sensor_name"] = user_input.get("sensor_name", "SFML")

            # Prepare config - sensors are created in sensor.py
            self._prepare_sfml_config()

            return self.async_create_entry(
                title=f"SFML {self.data['sensor_name']}",
                data=self.data
            )

        return self.async_show_form(
            step_id="sfml_config",
            data_schema=vol.Schema({
                vol.Required("source_sensor"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor", device_class="power")
                ),
                vol.Required("sensor_name", default="Solar"): str,
            })
        )

    def _prepare_sfml_config(self):
        """Prepare SFML config - all sensors are created internally via sensor.py."""
        source = self.data["source_sensor"]

        # Store source for all SFML sensors
        self.data["sfml_power_source"] = source

        # All 3 sensors (Power, Yield, Daily Yield) are created in sensor.py:
        # - ESCSFMLPowerSensor: Filters positive W values
        # - ESCSFMLYieldSensor: Riemann integration for total kWh
        # - ESCSFMLDailyYieldSensor: Daily kWh with midnight reset

    async def async_step_sfml_panel_count(self, user_input=None):
        """Select number of panel groups (1-4)."""
        if user_input is not None:
            self.data["panel_count"] = int(user_input["panel_count"])
            self.data["panels"] = []
            return await self.async_step_sfml_panel_config()

        return self.async_show_form(
            step_id="sfml_panel_count",
            data_schema=vol.Schema({
                vol.Required("panel_count", default="2"): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=[
                        {"label": "1 Panelgruppe", "value": "1"},
                        {"label": "2 Panelgruppen", "value": "2"},
                        {"label": "3 Panelgruppen", "value": "3"},
                        {"label": "4 Panelgruppen", "value": "4"},
                    ], mode=selector.SelectSelectorMode.DROPDOWN)
                ),
            })
        )

    async def async_step_sfml_panel_config(self, user_input=None):
        """Configure each panel group."""
        current_panel = len(self.data["panels"]) + 1
        total_panels = self.data["panel_count"]

        if user_input is not None:
            # Store this panel's config
            self.data["panels"].append({
                "source_sensor": user_input["source_sensor"],
                "panel_name": user_input.get("panel_name", f"Grp{current_panel:02d}"),
            })

            # Check if more panels to configure
            if len(self.data["panels"]) < total_panels:
                return await self.async_step_sfml_panel_config()

            # All panels configured - create entry
            return self.async_create_entry(
                title=f"SFML Panelgruppen ({total_panels}x)",
                data=self.data
            )

        return self.async_show_form(
            step_id="sfml_panel_config",
            data_schema=vol.Schema({
                vol.Required("source_sensor"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor", device_class="power")
                ),
                vol.Required("panel_name", default=f"Grp{current_panel:02d}"): str,
            }),
            description_placeholders={
                "current": str(current_panel),
                "total": str(total_panels),
            }
        )

    async def async_step_select_sensors(self, user_input=None):
        """Handle sensor selection for Sum."""
        if user_input is not None:
            self.data.update(user_input)
            return await self.async_step_device_class()

        return self.async_show_form(
            step_id="select_sensors",
            data_schema=vol.Schema({
                vol.Required("source_sensors"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor", multiple=True)
                ),
            })
        )

    async def async_step_select_single_sensor(self, user_input=None):
        """Handle sensor selection for History Statistics."""
        if user_input is not None:
            self.data["source_sensors"] = [user_input["source_sensors"]]
            return await self.async_step_select_sql_stat_type()

        return self.async_show_form(
            step_id="select_single_sensor",
            data_schema=vol.Schema({
                vol.Required("source_sensors"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor")
                ),
            })
        )

    async def async_step_select_delta_sensor(self, user_input=None):
        """Handle sensor selection for Delta."""
        if user_input is not None:
            self.data["source_sensor"] = user_input["source_sensor"]
            return await self.async_step_delta_period()

        return self.async_show_form(
            step_id="select_delta_sensor",
            data_schema=vol.Schema({
                vol.Required("source_sensor"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor")
                ),
            })
        )

    async def async_step_delta_period(self, user_input=None):
        """Select period for Delta."""
        if user_input is not None:
            self.data["delta_period"] = user_input["delta_period"]
            return await self.async_step_device_class()

        return self.async_show_form(
            step_id="delta_period",
            data_schema=vol.Schema({
                vol.Required("delta_period"): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=[
                        {"label": "Heute vs. Gestern", "value": DELTA_PERIOD_TODAY_YESTERDAY},
                        {"label": "Dieser Monat vs. Vorheriger", "value": DELTA_PERIOD_MONTH_PREV},
                    ], mode=selector.SelectSelectorMode.DROPDOWN)
                ),
            })
        )

    async def async_step_select_sql_stat_type(self, user_input=None):
        """Select statistic type for History sensor."""
        if user_input is not None:
            self.data.update(user_input)
            return await self.async_step_device_class()

        return self.async_show_form(
            step_id="select_sql_stat_type",
            data_schema=vol.Schema({
                vol.Required("sql_stat_type"): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=[
                        {"label": "Durchschnitt - Heute", "value": SQL_STAT_AVG_TODAY},
                        {"label": "Durchschnitt - Dieser Monat", "value": SQL_STAT_AVG_MONTH},
                        {"label": "Durchschnitt - Dieses Jahr", "value": SQL_STAT_AVG_YEAR},
                        {"label": "Maximum - Heute", "value": SQL_STAT_MAX_TODAY},
                        {"label": "Maximum - Dieser Monat", "value": SQL_STAT_MAX_MONTH},
                        {"label": "Maximum - Dieses Jahr", "value": SQL_STAT_MAX_YEAR},
                        {"label": "Minimum - Heute", "value": SQL_STAT_MIN_TODAY},
                        {"label": "Minimum - Dieser Monat", "value": SQL_STAT_MIN_MONTH},
                        {"label": "Minimum - Dieses Jahr", "value": SQL_STAT_MIN_YEAR},
                    ], mode=selector.SelectSelectorMode.DROPDOWN)
                ),
            })
        )

    async def async_step_binary_threshold(self, user_input=None):
        """Handle binary threshold config."""
        if user_input is not None:
            self.data.update(user_input)
            return await self.async_step_device_class()  # Binary hat auch Class (z.B. heat)

        return self.async_show_form(
            step_id="binary_threshold",
            data_schema=vol.Schema({
                vol.Required("source_sensor"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor")
                ),
                vol.Required("threshold"): vol.All(vol.Coerce(float), vol.Range(min=-1000, max=1000)),
                vol.Required("above_threshold"): bool,  # True: On wenn >, False: On wenn <
            })
        )

    async def async_step_toggle_target(self, user_input=None):
        """Handle toggle target config."""
        if user_input is not None:
            self.data["target_entity"] = user_input["target_entity"]
            self.data["action"] = user_input.get("action", "pause")  # z.B. pause oder reset
            return self.async_create_entry(
                title=f"Toggle für {user_input['target_entity']}",
                data=self.data
            )

        return self.async_show_form(
            step_id="toggle_target",
            data_schema=vol.Schema({
                vol.Required("target_entity"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor", "switch"])
                ),
                vol.Optional("action"): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=[
                        {"label": "Pausieren", "value": "pause"},
                        {"label": "Resetten", "value": "reset"},
                    ], mode=selector.SelectSelectorMode.DROPDOWN)
                ),
            })
        )
        
    async def async_step_device_class(self, user_input=None):
        """Select the device class."""
        if user_input is not None:
            device_class = user_input["device_class"]
            self.data["device_class"] = None if device_class == DEVICE_CLASS_NONE else device_class
            return await self.async_step_name_sensor()

        return self.async_show_form(
            step_id="device_class",
            data_schema=vol.Schema({
                vol.Required("device_class", default=DEVICE_CLASS_NONE): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=DEVICE_CLASSES, mode=selector.SelectSelectorMode.DROPDOWN)
                ),
            })
        )

    async def async_step_name_sensor(self, user_input=None):
        """Final step to name the sensor."""
        if user_input is not None:
            self.data["sensor_name"] = user_input["sensor_name"]
            return self.async_create_entry(title=self.data["sensor_name"], data=self.data)

        suggested_name = "Neuer Sensor"
        if self.data.get("sensor_type") == SENSOR_TYPE_SUM:
            suggested_name = "Summen-Sensor"
        elif self.data.get("sensor_type") == SENSOR_TYPE_SQL:
            suggested_name = self.data.get("sql_stat_type", "Statistik").replace("_", " ").title()
        elif self.data.get("sensor_type") == SENSOR_TYPE_DELTA:
            suggested_name = self.data.get("delta_period", "Delta").replace("_", " ").title()

        return self.async_show_form(
            step_id="name_sensor",
            data_schema=vol.Schema({vol.Required("sensor_name", default=suggested_name): str}),
        )