"""Battery Sensor Auto-Discovery Service V10.0.0 @zara

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

import logging
from typing import Any, Dict

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
import voluptuous as vol

from ..battery.battery_sensor_discovery import BatterySensorDiscovery
from ..const import (
    CONF_BATTERY_POWER_SENSOR,
    CONF_BATTERY_SOC_SENSOR,
    CONF_BATTERY_TEMPERATURE_SENSOR,
    CONF_GRID_CHARGE_POWER_SENSOR,
    CONF_GRID_EXPORT_SENSOR,
    CONF_GRID_IMPORT_SENSOR,
    CONF_HOUSE_CONSUMPTION_SENSOR,
    CONF_INVERTER_OUTPUT_SENSOR,
    CONF_SOLAR_PRODUCTION_SENSOR,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

SERVICE_DISCOVER_BATTERY_SENSORS = "discover_battery_sensors"
SERVICE_VALIDATE_BATTERY_SENSORS = "validate_battery_sensors"
SERVICE_AUTO_CONFIGURE_BATTERY = "auto_configure_battery"

DISCOVER_SCHEMA = vol.Schema(
    {
        vol.Optional("integration"): cv.string,
        vol.Optional("auto_apply", default=False): cv.boolean,
    }
)

VALIDATE_SCHEMA = vol.Schema(
    {
        vol.Required("battery_power_sensor"): cv.entity_id,
        vol.Required("battery_soc_sensor"): cv.entity_id,
        vol.Required("solar_production_sensor"): cv.entity_id,
        vol.Required("inverter_output_sensor"): cv.entity_id,
        vol.Required("house_consumption_sensor"): cv.entity_id,
        vol.Required("grid_import_sensor"): cv.entity_id,
        vol.Required("grid_export_sensor"): cv.entity_id,
        vol.Optional("grid_charge_power_sensor"): cv.entity_id,
        vol.Optional("battery_temperature_sensor"): cv.entity_id,
    }
)

async def _apply_discovered_sensors(hass: HomeAssistant, discovered: Dict[str, str]) -> Dict[str, Any]:
    """Apply discovered sensors to the Solar Forecast ML config entry @zara"""

    config_entries = hass.config_entries.async_entries(DOMAIN)

    if not config_entries:
        _LOGGER.error("No Solar Forecast ML config entry found")
        return {
            "success": False,
            "error": "No Solar Forecast ML integration configured",
        }

    entry: ConfigEntry = config_entries[0]

    new_options = dict(entry.options)

    sensor_mapping = {
        "battery_power_sensor": CONF_BATTERY_POWER_SENSOR,
        "battery_soc_sensor": CONF_BATTERY_SOC_SENSOR,
        "solar_production_sensor": CONF_SOLAR_PRODUCTION_SENSOR,
        "inverter_output_sensor": CONF_INVERTER_OUTPUT_SENSOR,
        "house_consumption_sensor": CONF_HOUSE_CONSUMPTION_SENSOR,
        "grid_import_sensor": CONF_GRID_IMPORT_SENSOR,
        "grid_export_sensor": CONF_GRID_EXPORT_SENSOR,
        "grid_charge_power_sensor": CONF_GRID_CHARGE_POWER_SENSOR,
        "battery_temperature_sensor": CONF_BATTERY_TEMPERATURE_SENSOR,
    }

    updated_sensors = []
    for discovered_key, config_key in sensor_mapping.items():
        if discovered.get(discovered_key):
            new_options[config_key] = discovered[discovered_key]
            updated_sensors.append(discovered_key)
            _LOGGER.info(f"Applied {discovered_key}: {discovered[discovered_key]}")

    hass.config_entries.async_update_entry(entry, options=new_options)

    _LOGGER.info(
        f"Successfully applied {len(updated_sensors)} sensors to config entry. "
        f"Reloading integration..."
    )

    await hass.config_entries.async_reload(entry.entry_id)

    return {
        "success": True,
        "updated_sensors": updated_sensors,
        "sensor_count": len(updated_sensors),
        "message": f"Successfully configured {len(updated_sensors)} battery sensors. Integration reloaded.",
    }

async def async_setup_services(hass: HomeAssistant) -> None:
    """Register battery discovery services @zara"""

    async def handle_discover_battery_sensors(call: ServiceCall) -> Dict[str, Any]:
        """Handle battery sensor discovery service call @zara"""
        integration = call.data.get("integration")
        auto_apply = call.data.get("auto_apply", False)

        discovery = BatterySensorDiscovery(hass)

        if integration:

            _LOGGER.info(f"Starting battery sensor discovery for integration: {integration}")
            discovered = discovery.discover_sensors(integration)

            validation = discovery.validate_discovered_sensors(discovered)

            result = {
                "integration": integration,
                "discovered": discovered,
                "validation": validation,
            }

            if auto_apply and validation.get("valid"):
                _LOGGER.info("Auto-apply enabled - updating configuration entry")
                apply_result = await _apply_discovered_sensors(hass, discovered)
                result["applied"] = apply_result
        else:

            _LOGGER.info("Starting battery sensor discovery for all supported integrations")
            all_discovered = discovery.discover_all_integrations()

            result = {
                "integrations": list(all_discovered.keys()),
                "results": all_discovered,
                "supported_integrations": discovery.get_supported_integrations(),
            }

            if auto_apply and all_discovered:
                for integ_name, integ_discovered in all_discovered.items():
                    validation = discovery.validate_discovered_sensors(integ_discovered)
                    if validation.get("valid"):
                        _LOGGER.info(
                            f"Auto-apply enabled - using first valid integration: {integ_name}"
                        )
                        apply_result = await _apply_discovered_sensors(hass, integ_discovered)
                        result["applied"] = apply_result
                        result["applied_integration"] = integ_name
                        break

        _LOGGER.info(f"Battery sensor discovery completed: {result}")
        return result

    async def handle_validate_battery_sensors(call: ServiceCall) -> Dict[str, Any]:
        """Handle battery sensor validation service call @zara"""
        sensors = dict(call.data)

        discovery = BatterySensorDiscovery(hass)
        validation = discovery.validate_discovered_sensors(sensors)

        _LOGGER.info(f"Battery sensor validation completed: {validation}")
        return validation

    async def handle_auto_configure_battery(call: ServiceCall) -> Dict[str, Any]:
        """Handle automatic battery configuration service call @zara"""
        integration = call.data.get("integration")

        discovery = BatterySensorDiscovery(hass)

        if integration:
            _LOGGER.info(f"Auto-configuring battery sensors for: {integration}")
            discovered = discovery.discover_sensors(integration)
        else:
            _LOGGER.info("Auto-configuring battery sensors (searching all integrations)")
            all_discovered = discovery.discover_all_integrations()

            if not all_discovered:
                return {
                    "success": False,
                    "error": "No supported integrations found with battery sensors",
                }

            discovered = None
            integration = None
            for integ_name, integ_discovered in all_discovered.items():
                validation = discovery.validate_discovered_sensors(integ_discovered)
                if validation.get("valid"):
                    discovered = integ_discovered
                    integration = integ_name
                    break

            if not discovered:
                return {
                    "success": False,
                    "error": "No valid battery sensor configuration found",
                    "tried_integrations": list(all_discovered.keys()),
                }

        validation = discovery.validate_discovered_sensors(discovered)

        if not validation.get("valid"):
            return {
                "success": False,
                "integration": integration,
                "discovered": discovered,
                "validation": validation,
                "error": "Discovered sensors failed validation",
            }

        _LOGGER.info(f"Validation passed - applying configuration for {integration}")
        apply_result = await _apply_discovered_sensors(hass, discovered)

        return {
            "success": apply_result.get("success", False),
            "integration": integration,
            "discovered": discovered,
            "validation": validation,
            "applied": apply_result,
        }

    hass.services.async_register(
        DOMAIN,
        SERVICE_DISCOVER_BATTERY_SENSORS,
        handle_discover_battery_sensors,
        schema=DISCOVER_SCHEMA,
        supports_response=True,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_VALIDATE_BATTERY_SENSORS,
        handle_validate_battery_sensors,
        schema=VALIDATE_SCHEMA,
        supports_response=True,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_AUTO_CONFIGURE_BATTERY,
        handle_auto_configure_battery,
        schema=vol.Schema({vol.Optional("integration"): cv.string}),
        supports_response=True,
    )

    _LOGGER.info(
        f"Battery discovery services registered: "
        f"{SERVICE_DISCOVER_BATTERY_SENSORS}, {SERVICE_VALIDATE_BATTERY_SENSORS}, "
        f"{SERVICE_AUTO_CONFIGURE_BATTERY}"
    )

async def async_unload_services(hass: HomeAssistant) -> None:
    """Unregister battery discovery services @zara"""
    hass.services.async_remove(DOMAIN, SERVICE_DISCOVER_BATTERY_SENSORS)
    hass.services.async_remove(DOMAIN, SERVICE_VALIDATE_BATTERY_SENSORS)
    hass.services.async_remove(DOMAIN, SERVICE_AUTO_CONFIGURE_BATTERY)

    _LOGGER.info("Battery discovery services unregistered")
