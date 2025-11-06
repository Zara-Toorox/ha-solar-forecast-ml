"""
Sensor Platform for Solar Forecast ML Integration

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

from .const import DOMAIN, CONF_DIAGNOSTIC, CONF_HOURLY

# Import core sensors from sensors module
from .sensors.sensor_base import (
    SolarForecastSensor,
    PeakProductionHourSensor,
    ProductionTimeSensor,
    AverageYieldSensor,
    AutarkySensor,
    NextHourSensor,
    ExpectedDailyProductionSensor,
    MaxPeakTodaySensor,
    MaxPeakAllTimeSensor,
    ForecastDayAfterTomorrowSensor,
    MonthlyYieldSensor,
    MonthlyConsumptionSensor,
    WeeklyYieldSensor,
    WeeklyConsumptionSensor,
    AverageYield7DaysSensor,
    AverageYield30DaysSensor,
    AverageAutarkyMonthSensor,
    AverageAccuracy30DaysSensor,
)

# Import diagnostic sensors
from .sensors.sensor_diagnostic import (
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
)

# Import state sensors
from .sensors.sensor_states import (
    ExternalTempSensor,
    ExternalHumiditySensor,
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
) -> bool:
    """Set up Solar Forecast ML sensors from config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    
    # Determine which sensor sets to add based on options
    diagnostic_mode_enabled = entry.options.get(CONF_DIAGNOSTIC, True)
    enable_hourly = entry.options.get(CONF_HOURLY, False)
    
    _LOGGER.info(
        f"Setting up sensors: Diagnostic Mode={'Enabled' if diagnostic_mode_enabled else 'Disabled'}, "
        f"Hourly Sensor={'Enabled' if enable_hourly else 'Disabled'}"
    )
    
    # Core sensors are always added
    core_entities = [
        SolarForecastSensor(coordinator, entry, "remaining"),
        SolarForecastSensor(coordinator, entry, "tomorrow"),
        PeakProductionHourSensor(coordinator, entry),
        ProductionTimeSensor(coordinator, entry),
        AverageYieldSensor(coordinator, entry),
        AutarkySensor(coordinator, entry),
        ExpectedDailyProductionSensor(coordinator, entry),
        MaxPeakTodaySensor(coordinator, entry),
        MaxPeakAllTimeSensor(coordinator, entry),
        ForecastDayAfterTomorrowSensor(coordinator, entry),
        MonthlyYieldSensor(coordinator, entry),
        MonthlyConsumptionSensor(coordinator, entry),
        WeeklyYieldSensor(coordinator, entry),
        WeeklyConsumptionSensor(coordinator, entry),
        AverageYield7DaysSensor(coordinator, entry),
        AverageYield30DaysSensor(coordinator, entry),
        AverageAutarkyMonthSensor(coordinator, entry),
        AverageAccuracy30DaysSensor(coordinator, entry),
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
            # State sensors use hass instead of coordinator
            ExternalTempSensor(hass, entry),
            ExternalHumiditySensor(hass, entry),
            ExternalWindSensor(hass, entry),
            ExternalRainSensor(hass, entry),
            ExternalUVSensor(hass, entry),
            ExternalLuxSensor(hass, entry),
            PowerSensorStateSensor(hass, entry),
            YieldSensorStateSensor(hass, entry),
        ]
        entities_to_add.extend(diagnostic_entities)
        _LOGGER.debug(f"Adding {len(diagnostic_entities)} diagnostic and state sensors.")
    else:
        # If diagnostic mode is off, still add the state sensors
        state_entities = [
            PowerSensorStateSensor(hass, entry),
            YieldSensorStateSensor(hass, entry),
        ]
        entities_to_add.extend(state_entities)
        _LOGGER.debug(f"Diagnostic mode disabled. Adding {len(state_entities)} state sensors.")
    
    # Add hourly sensor if enabled
    if enable_hourly:
        entities_to_add.append(NextHourSensor(coordinator, entry))
        _LOGGER.debug("Adding Next Hour Forecast sensor.")
    
    # Register all selected entities
    async_add_entities(entities_to_add, True)
    _LOGGER.info(f"Successfully added {len(entities_to_add)} Solar Forecast ML sensors.")
    
    return True
