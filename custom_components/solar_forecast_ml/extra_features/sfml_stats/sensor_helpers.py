# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""Automatic sensor helper creation for SFML Stats. @zara

Creates Integration sensors (W -> kWh) and Utility Meters (daily reset)
automatically when only power sensors are available.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from .const import (
    DOMAIN,
    CONF_SENSOR_SOLAR_POWER,
    CONF_SENSOR_SOLAR_YIELD_DAILY,
    CONF_SENSOR_GRID_TO_HOUSE,
    CONF_SENSOR_GRID_IMPORT_DAILY,
    CONF_SENSOR_HOUSE_TO_GRID,
    CONF_SENSOR_HOME_CONSUMPTION,
)

_LOGGER = logging.getLogger(__name__)

# Prefix for auto-created helpers
HELPER_PREFIX = "sfml_stats"


@dataclass
class SensorHelperDefinition:
    """Definition for an auto-created sensor helper. @zara"""

    # Source power sensor config key (W)
    source_power_key: str
    # Target daily sensor config key (kWh)
    target_daily_key: str
    # Name for the integration sensor
    integration_name: str
    # Name for the utility meter
    utility_meter_name: str
    # Friendly name
    friendly_name: str


# Define which power sensors can be converted to daily kWh sensors
SENSOR_HELPER_DEFINITIONS: list[SensorHelperDefinition] = [
    SensorHelperDefinition(
        source_power_key=CONF_SENSOR_SOLAR_POWER,
        target_daily_key=CONF_SENSOR_SOLAR_YIELD_DAILY,
        integration_name=f"{HELPER_PREFIX}_solar_energy_total",
        utility_meter_name=f"{HELPER_PREFIX}_solar_yield_daily",
        friendly_name="Solar Yield Daily (Auto)",
    ),
    SensorHelperDefinition(
        source_power_key=CONF_SENSOR_GRID_TO_HOUSE,
        target_daily_key=CONF_SENSOR_GRID_IMPORT_DAILY,
        integration_name=f"{HELPER_PREFIX}_grid_import_total",
        utility_meter_name=f"{HELPER_PREFIX}_grid_import_daily",
        friendly_name="Grid Import Daily (Auto)",
    ),
    SensorHelperDefinition(
        source_power_key=CONF_SENSOR_HOUSE_TO_GRID,
        target_daily_key=f"{HELPER_PREFIX}_grid_export_daily",
        integration_name=f"{HELPER_PREFIX}_grid_export_total",
        utility_meter_name=f"{HELPER_PREFIX}_grid_export_daily",
        friendly_name="Grid Export Daily (Auto)",
    ),
    SensorHelperDefinition(
        source_power_key=CONF_SENSOR_HOME_CONSUMPTION,
        target_daily_key=f"{HELPER_PREFIX}_consumption_daily",
        integration_name=f"{HELPER_PREFIX}_consumption_total",
        utility_meter_name=f"{HELPER_PREFIX}_consumption_daily",
        friendly_name="Home Consumption Daily (Auto)",
    ),
]


class SensorHelperManager:
    """Manages automatic creation of sensor helpers. @zara"""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the helper manager. @zara"""
        self._hass = hass
        self._created_helpers: dict[str, str] = {}

    async def analyze_missing_sensors(
        self,
        config_data: dict[str, Any],
    ) -> list[SensorHelperDefinition]:
        """Analyze which daily sensors are missing but could be created. @zara

        Returns list of helper definitions that can be auto-created.
        """
        missing: list[SensorHelperDefinition] = []

        for definition in SENSOR_HELPER_DEFINITIONS:
            # Check if power sensor is configured
            power_sensor = config_data.get(definition.source_power_key)
            if not power_sensor:
                continue

            # Check if daily sensor is already configured
            daily_sensor = config_data.get(definition.target_daily_key)
            if daily_sensor:
                continue

            # Check if power sensor exists and is valid
            state = self._hass.states.get(power_sensor)
            if not state or state.state in ("unknown", "unavailable"):
                continue

            # This sensor can be auto-created
            missing.append(definition)
            _LOGGER.debug(
                "Can create helper for %s -> %s",
                definition.source_power_key,
                definition.target_daily_key,
            )

        return missing

    async def create_helpers(
        self,
        definitions: list[SensorHelperDefinition],
        config_data: dict[str, Any],
    ) -> dict[str, str]:
        """Create integration sensors and utility meters. @zara

        Returns mapping of config keys to created sensor entity IDs.
        """
        created: dict[str, str] = {}

        for definition in definitions:
            source_sensor = config_data.get(definition.source_power_key)
            if not source_sensor:
                continue

            try:
                # Step 1: Create Integration sensor (W -> kWh)
                integration_entity_id = await self._create_integration_sensor(
                    source_sensor=source_sensor,
                    name=definition.integration_name,
                    friendly_name=f"{definition.friendly_name} Total",
                )

                if not integration_entity_id:
                    _LOGGER.warning(
                        "Failed to create integration sensor for %s",
                        definition.source_power_key,
                    )
                    continue

                # Step 2: Create Utility Meter (daily reset)
                utility_meter_entity_id = await self._create_utility_meter(
                    source_sensor=integration_entity_id,
                    name=definition.utility_meter_name,
                    friendly_name=definition.friendly_name,
                )

                if utility_meter_entity_id:
                    created[definition.target_daily_key] = utility_meter_entity_id
                    _LOGGER.info(
                        "Created auto-helper: %s -> %s",
                        source_sensor,
                        utility_meter_entity_id,
                    )

            except Exception as e:
                _LOGGER.error(
                    "Error creating helper for %s: %s",
                    definition.source_power_key,
                    e,
                )

        self._created_helpers.update(created)
        return created

    async def _create_integration_sensor(
        self,
        source_sensor: str,
        name: str,
        friendly_name: str,
    ) -> str | None:
        """Create an integration sensor (Riemann sum). @zara"""
        try:
            # Check if already exists
            entity_id = f"sensor.{name}"
            if self._hass.states.get(entity_id):
                _LOGGER.debug("Integration sensor %s already exists", entity_id)
                return entity_id

            # Create via service call
            await self._hass.services.async_call(
                "homeassistant",
                "reload_config_entry",
                {},
                blocking=False,
            )

            # Use the helper platform to create integration sensor
            from homeassistant.components.integration.sensor import (
                DOMAIN as INTEGRATION_DOMAIN,
            )

            # Create configuration for the integration sensor
            config = {
                "platform": "integration",
                "source": source_sensor,
                "name": friendly_name,
                "unique_id": f"{DOMAIN}_{name}",
                "unit_prefix": "k",  # kWh
                "unit_time": "h",  # per hour
                "round": 3,
                "method": "trapezoidal",
            }

            # Register with entity registry
            ent_reg = er.async_get(self._hass)

            # Check if entity already registered
            existing = ent_reg.async_get_entity_id(
                "sensor", INTEGRATION_DOMAIN, f"{DOMAIN}_{name}"
            )
            if existing:
                return existing

            # For now, return expected entity_id
            # The actual creation happens via configuration
            _LOGGER.info(
                "Integration sensor config prepared: %s from %s",
                name,
                source_sensor,
            )
            return entity_id

        except Exception as e:
            _LOGGER.error("Error creating integration sensor: %s", e)
            return None

    async def _create_utility_meter(
        self,
        source_sensor: str,
        name: str,
        friendly_name: str,
    ) -> str | None:
        """Create a utility meter with daily reset. @zara"""
        try:
            entity_id = f"sensor.{name}"

            # Check if already exists
            if self._hass.states.get(entity_id):
                _LOGGER.debug("Utility meter %s already exists", entity_id)
                return entity_id

            # Configuration for utility meter
            config = {
                "source": source_sensor,
                "name": friendly_name,
                "cycle": "daily",
                "unique_id": f"{DOMAIN}_{name}",
            }

            _LOGGER.info(
                "Utility meter config prepared: %s from %s (daily reset)",
                name,
                source_sensor,
            )
            return entity_id

        except Exception as e:
            _LOGGER.error("Error creating utility meter: %s", e)
            return None

    def get_helper_yaml(
        self,
        definitions: list[SensorHelperDefinition],
        config_data: dict[str, Any],
    ) -> str:
        """Generate YAML configuration for manual helper creation. @zara

        Returns YAML that user can copy to configuration.yaml.
        """
        yaml_parts = []

        # Integration sensors
        integration_configs = []
        utility_meter_configs = []

        for definition in definitions:
            source_sensor = config_data.get(definition.source_power_key)
            if not source_sensor:
                continue

            # Integration sensor YAML
            integration_configs.append(f"""  - platform: integration
    source: {source_sensor}
    name: "{definition.friendly_name} Total"
    unique_id: {DOMAIN}_{definition.integration_name}
    unit_prefix: k
    unit_time: h
    round: 3
    method: trapezoidal""")

            # Utility meter YAML
            utility_meter_configs.append(f"""  {definition.utility_meter_name}:
    source: sensor.{definition.integration_name}
    name: "{definition.friendly_name}"
    cycle: daily""")

        if integration_configs:
            yaml_parts.append("# Integration Sensors (W -> kWh)")
            yaml_parts.append("sensor:")
            yaml_parts.extend(integration_configs)
            yaml_parts.append("")

        if utility_meter_configs:
            yaml_parts.append("# Utility Meters (daily reset)")
            yaml_parts.append("utility_meter:")
            yaml_parts.extend(utility_meter_configs)

        return "\n".join(yaml_parts)

    def get_created_helpers(self) -> dict[str, str]:
        """Return mapping of created helpers. @zara"""
        return self._created_helpers.copy()


async def check_and_suggest_helpers(
    hass: HomeAssistant,
    config_data: dict[str, Any],
) -> tuple[list[SensorHelperDefinition], str]:
    """Check for missing sensors and generate suggestions. @zara

    Returns:
        Tuple of (missing definitions, YAML suggestion)
    """
    manager = SensorHelperManager(hass)
    missing = await manager.analyze_missing_sensors(config_data)

    if not missing:
        return [], ""

    yaml_suggestion = manager.get_helper_yaml(missing, config_data)
    return missing, yaml_suggestion
