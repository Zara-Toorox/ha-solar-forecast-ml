"""Battery Sensor Auto-Discovery V10.0.0 @zara

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
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

_LOGGER = logging.getLogger(__name__)

class SensorMapping:
    """Sensor mapping configuration for a specific integration"""

    def __init__(
        self,
        integration: str,
        battery_power: List[str],
        battery_soc: List[str],
        solar_production: List[str],
        inverter_output: List[str],
        house_consumption: List[str],
        grid_import: List[str],
        grid_export: List[str],
        grid_charge_power: Optional[List[str]] = None,
        battery_temperature: Optional[List[str]] = None,
    ):
        """Initialize sensor mapping

        Args:
            integration: Integration domain (e.g., 'anker_solix', 'huawei_solar')
            battery_power: List of possible entity_id patterns for battery power
            battery_soc: List of possible entity_id patterns for SOC
            solar_production: List of possible entity_id patterns for solar production
            inverter_output: List of possible entity_id patterns for inverter output
            house_consumption: List of possible entity_id patterns for house consumption
            grid_import: List of possible entity_id patterns for grid import
            grid_export: List of possible entity_id patterns for grid export
            grid_charge_power: Optional list for grid charge power sensors
            battery_temperature: Optional list for temperature sensors
        """
        self.integration = integration
        self.battery_power = battery_power
        self.battery_soc = battery_soc
        self.solar_production = solar_production
        self.inverter_output = inverter_output
        self.house_consumption = house_consumption
        self.grid_import = grid_import
        self.grid_export = grid_export
        self.grid_charge_power = grid_charge_power or []
        self.battery_temperature = battery_temperature or []

KNOWN_INTEGRATIONS: Dict[str, SensorMapping] = {
    "anker_solix": SensorMapping(
        integration="anker_solix",
        battery_power=[
            "charging_power",
            "akkuleistung",
            "battery_power",
        ],
        battery_soc=[
            "state_of_charge",
            "ladestand",
            "battery_soc",
        ],
        solar_production=[
            "input_power",
            "solarleistung",
            "solarbank_input_power",
            "solar_power",
        ],
        inverter_output=[
            "output_power",
            "dc_ausgangsleistung",
            "solarbank_output_power",
            "micro_inverter_power",
            "mwr_leistung",
        ],
        house_consumption=[
            "home_load_power",
            "ac_hausabgabe",
            "home_usage_avg",
            "ac_to_home_load",
            "hausabgabe",
        ],
        grid_import=[
            "grid_to_home_power",
            "grid_import_avg",
            "netzbezug",
        ],
        grid_export=[
            "photovoltaic_to_grid_power",
            "grid_export_avg",
            "einspe",
            "netzeinspeisung",
        ],
        grid_charge_power=[
            "grid_to_battery_power",
            "netzaufladung",
            "grid_charge",
        ],
        battery_temperature=[
            "temperature",
            "temperatur",
        ],
    ),
    "huawei_solar": SensorMapping(
        integration="huawei_solar",
        battery_power=[
            "battery_charge_discharge_power",
        ],
        battery_soc=[
            "battery_state_of_capacity",
        ],
        solar_production=[
            "input_power",
        ],
        inverter_output=[
            "active_power",
        ],
        house_consumption=[
            "power_meter_active_power",
        ],
        grid_import=[
            "grid_power",
        ],
        grid_export=[
            "grid_power",
        ],
        battery_temperature=[
            "battery_temperature",
        ],
    ),
    "fronius": SensorMapping(
        integration="fronius",
        battery_power=[
            "battery_power",
        ],
        battery_soc=[
            "battery_soc",
            "battery_state_of_charge",
        ],
        solar_production=[
            "solar_production",
            "inverter_power",
        ],
        inverter_output=[
            "inverter_power",
        ],
        house_consumption=[
            "load_power",
        ],
        grid_import=[
            "grid_import",
        ],
        grid_export=[
            "grid_export",
        ],
    ),
}

class BatterySensorDiscovery:
    """Auto-discovery service for battery sensors"""

    def __init__(self, hass: HomeAssistant):
        """Initialize discovery service @zara"""
        self.hass = hass
        self.entity_registry = er.async_get(hass)

    def _get_all_sensors(self) -> List[er.RegistryEntry]:
        """Get all sensor entities from the registry @zara"""
        return [
            entry
            for entry in self.entity_registry.entities.values()
            if entry.domain == "sensor" and not entry.disabled
        ]

    def _match_sensor(
        self, sensors: List[er.RegistryEntry], patterns: List[str], integration: str
    ) -> Optional[str]:
        """Match sensor by entity_id pattern and integration

        Args:
            sensors: List of available sensors
            patterns: List of patterns to match (e.g., ['charging_power', 'battery_power'])
            integration: Integration domain to filter by

        Returns:
            Entity ID of matched sensor or None
        """
        for sensor in sensors:

            if sensor.platform != integration:
                continue

            entity_suffix = sensor.entity_id.split(".", 1)[1] if "." in sensor.entity_id else ""

            for pattern in patterns:
                if pattern.lower() in entity_suffix.lower():
                    _LOGGER.debug(
                        f"Matched sensor '{sensor.entity_id}' for pattern '{pattern}' "
                        f"(integration: {integration})"
                    )
                    return sensor.entity_id

        return None

    def discover_sensors(self, integration: str) -> Dict[str, Optional[str]]:
        """Discover battery sensors for a specific integration @zara"""
        if integration not in KNOWN_INTEGRATIONS:
            _LOGGER.warning(
                f"Integration '{integration}' not supported for auto-discovery. "
                f"Supported integrations: {', '.join(KNOWN_INTEGRATIONS.keys())}"
            )
            return {}

        mapping = KNOWN_INTEGRATIONS[integration]
        sensors = self._get_all_sensors()

        discovered = {
            "battery_power_sensor": self._match_sensor(sensors, mapping.battery_power, integration),
            "battery_soc_sensor": self._match_sensor(sensors, mapping.battery_soc, integration),
            "solar_production_sensor": self._match_sensor(
                sensors, mapping.solar_production, integration
            ),
            "inverter_output_sensor": self._match_sensor(
                sensors, mapping.inverter_output, integration
            ),
            "house_consumption_sensor": self._match_sensor(
                sensors, mapping.house_consumption, integration
            ),
            "grid_import_sensor": self._match_sensor(sensors, mapping.grid_import, integration),
            "grid_export_sensor": self._match_sensor(sensors, mapping.grid_export, integration),
        }

        if mapping.grid_charge_power:
            discovered["grid_charge_power_sensor"] = self._match_sensor(
                sensors, mapping.grid_charge_power, integration
            )

        if mapping.battery_temperature:
            discovered["battery_temperature_sensor"] = self._match_sensor(
                sensors, mapping.battery_temperature, integration
            )

        found_count = sum(1 for v in discovered.values() if v is not None)
        _LOGGER.info(
            f"Auto-discovery for '{integration}' completed: {found_count}/{len(discovered)} sensors found"
        )

        for key, value in discovered.items():
            if value:
                _LOGGER.info(f"  ✓ {key}: {value}")
            else:
                _LOGGER.warning(f"  ✗ {key}: Not found")

        return discovered

    def discover_all_integrations(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Discover sensors from all supported integrations @zara"""
        results = {}

        for integration in KNOWN_INTEGRATIONS.keys():
            discovered = self.discover_sensors(integration)
            if any(v is not None for v in discovered.values()):
                results[integration] = discovered

        return results

    def get_supported_integrations(self) -> List[str]:
        """Get list of supported integrations @zara"""
        return list(KNOWN_INTEGRATIONS.keys())

    def validate_discovered_sensors(self, discovered: Dict[str, Optional[str]]) -> Dict[str, Any]:
        """Validate discovered sensors @zara"""

        core_required = [
            "battery_power_sensor",
            "battery_soc_sensor",
            "solar_production_sensor",
            "inverter_output_sensor",
            "house_consumption_sensor",
        ]

        grid_sensors = ["grid_import_sensor", "grid_export_sensor"]

        errors = []
        warnings = []

        for sensor_key in core_required:
            if not discovered.get(sensor_key):
                errors.append(f"Required sensor '{sensor_key}' not found")
                continue

            entity_id = discovered[sensor_key]
            state = self.hass.states.get(entity_id)

            if state is None:
                errors.append(f"Sensor '{entity_id}' does not exist in Home Assistant")
            elif state.state in ("unavailable", "unknown"):
                warnings.append(f"Sensor '{entity_id}' is currently unavailable")

        grid_found = sum(1 for sensor in grid_sensors if discovered.get(sensor))

        if grid_found == 0:

            warnings.append(
                "No Smart Meter sensors found (grid_import/export). "
                "Grid tracking will be limited. Consider installing a Smart Meter for full energy flow tracking."
            )
        elif grid_found == 1:

            missing = [s for s in grid_sensors if not discovered.get(s)]
            errors.append(
                f"Incomplete grid sensor configuration: {missing[0]} is missing. "
                f"Either provide both grid sensors or neither."
            )
        else:

            for sensor_key in grid_sensors:
                entity_id = discovered[sensor_key]
                state = self.hass.states.get(entity_id)

                if state is None:
                    errors.append(f"Sensor '{entity_id}' does not exist in Home Assistant")
                elif state.state in ("unavailable", "unknown"):
                    warnings.append(f"Sensor '{entity_id}' is currently unavailable")

        optional_sensors = ["grid_charge_power_sensor", "battery_temperature_sensor"]
        for sensor_key in optional_sensors:
            if discovered.get(sensor_key):
                entity_id = discovered[sensor_key]
                state = self.hass.states.get(entity_id)

                if state is None:
                    warnings.append(f"Optional sensor '{entity_id}' does not exist")
                elif state.state in ("unavailable", "unknown"):
                    warnings.append(f"Optional sensor '{entity_id}' is currently unavailable")

        is_valid = len(errors) == 0

        required_count = len(core_required)
        if grid_found == 2:
            required_count += 2

        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "discovered_count": sum(1 for v in discovered.values() if v is not None),
            "required_count": required_count,
            "has_smart_meter": grid_found == 2,
        }
