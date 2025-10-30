"""
Sensor platform setup for Solar Forecast ML Integration.
Imports and registers sensor entities based on configuration.

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

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, CONF_HOURLY, CONF_DIAGNOSTIC
from .coordinator import SolarForecastMLCoordinator

# Import sensors from their respective modules
from .sensor_core import (
    SolarForecastSensor,
    NextHourSensor,
    PeakProductionHourSensor,
    ProductionTimeSensor,
    AverageYieldSensor,
    AutarkySensor,
)
from .sensor_diagnostic import (
    DiagnosticStatusSensor,
    SolarAccuracySensor,
    YesterdayDeviationSensor,
    LastCoordinatorUpdateSensor,
    UpdateAgeSensor,
    LastMLTrainingSensor,
    NextScheduledUpdateSensor,
    MLServiceStatusSensor,
    MLMetricsSensor,
    CoordinatorHealthSensor,
    DataFilesStatusSensor,
    SunGuardWindowSensor, # <-- NEU (Block 4)
)
from .sensor_states import (
    ExternalTempSensor,
    ExternalHumiditySensor, # <-- NEU (Block 3)
    ExternalWindSensor,
    ExternalRainSensor,
    ExternalUVSensor,
    ExternalLuxSensor,
    PowerSensorStateSensor,
    YieldSensorStateSensor,
)


_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Solar Forecast ML sensors from config entry."""
    coordinator: SolarForecastMLCoordinator = hass.data[DOMAIN][entry.entry_id]

    # Determine which sensor sets to add based on options
    diagnostic_mode_enabled = entry.options.get(CONF_DIAGNOSTIC, True) # Default to True
    enable_hourly = entry.options.get(CONF_HOURLY, False) # Default to False

    _LOGGER.info(
        f"Setting up sensors: Diagnostic Mode={'Enabled' if diagnostic_mode_enabled else 'Disabled'}, "
        f"Hourly Sensor={'Enabled' if enable_hourly else 'Disabled'}"
    )

    # Core sensors are always added
    core_entities = [
        SolarForecastSensor(coordinator, entry, "today"),
        SolarForecastSensor(coordinator, entry, "tomorrow"),
        PeakProductionHourSensor(coordinator, entry),
        ProductionTimeSensor(coordinator, entry), # The live-updating sensor
        AverageYieldSensor(coordinator, entry),
        AutarkySensor(coordinator, entry),
    ]

    entities_to_add = core_entities

    # Add diagnostic sensors if enabled
    if diagnostic_mode_enabled:
        diagnostic_entities = [
            DiagnosticStatusSensor(coordinator, entry),
            SolarAccuracySensor(coordinator, entry),
            YesterdayDeviationSensor(coordinator, entry),
            LastCoordinatorUpdateSensor(coordinator, entry),
            UpdateAgeSensor(coordinator, entry),
            LastMLTrainingSensor(coordinator, entry),
            NextScheduledUpdateSensor(coordinator, entry),
            MLServiceStatusSensor(coordinator, entry),
            MLMetricsSensor(coordinator, entry),
            CoordinatorHealthSensor(coordinator, entry),
            DataFilesStatusSensor(coordinator, entry),
            SunGuardWindowSensor(coordinator, entry), # <-- NEU (Block 4)
            # State sensors are now separate but often considered diagnostic
            ExternalTempSensor(coordinator, entry),
            ExternalHumiditySensor(coordinator, entry), # <-- NEU (Block 3)
            ExternalWindSensor(coordinator, entry),
            ExternalRainSensor(coordinator, entry),
            ExternalUVSensor(coordinator, entry),
            ExternalLuxSensor(coordinator, entry),
            PowerSensorStateSensor(coordinator, entry), # Now a normal sensor
            YieldSensorStateSensor(coordinator, entry), # Now a normal sensor
        ]
        entities_to_add.extend(diagnostic_entities)
        _LOGGER.debug(f"Adding {len(diagnostic_entities)} diagnostic and state sensors.")
    else:
        # If diagnostic mode is off, still add the NORMAL state sensors
        state_entities = [
             PowerSensorStateSensor(coordinator, entry),
             YieldSensorStateSensor(coordinator, entry),
        ]
        entities_to_add.extend(state_entities)
        _LOGGER.debug(f"Diagnostic mode disabled. Adding {len(state_entities)} state sensors.")


    # Add hourly sensor if enabled
    if enable_hourly:
        entities_to_add.append(NextHourSensor(coordinator, entry))
        _LOGGER.debug("Adding Next Hour Forecast sensor.")

    # Register all selected entities
    async_add_entities(entities_to_add, True) # True enables automatic polling via coordinator where applicable
    _LOGGER.info(f"Successfully added {len(entities_to_add)} Solar Forecast ML sensors.")

# All sensor class definitions have been moved to sensor_core.py,
# sensor_diagnostic.py, and sensor_states.py.