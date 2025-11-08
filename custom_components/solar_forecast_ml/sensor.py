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

from .const import DOMAIN, CONF_DIAGNOSTIC, CONF_HOURLY, CONF_BATTERY_ENABLED, CONF_ELECTRICITY_ENABLED

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
    YesterdayDeviationSensor,
    LastCoordinatorUpdateSensor,
    LastMLTrainingSensor,
    NextScheduledUpdateSensor,
    MLServiceStatusSensor,
    MLMetricsSensor,
    CoordinatorHealthSensor,
    DataFilesStatusSensor,
    CloudinessTrend1hSensor,
    CloudinessTrend3hSensor,
    CloudinessVolatilitySensor,
    NextProductionStartSensor,
)

# Import state sensors
from .sensors.sensor_states import (
    ExternalSensorsStatusSensor,
    PowerSensorStateSensor,
    YieldSensorStateSensor,
)

# ========================================================================
# BATTERY MANAGEMENT SENSORS (v8.3.0) - Completely separate from Solar
# ========================================================================
from .sensors.sensor_battery import BATTERY_SENSORS
from .sensors.sensor_electricity_price import ELECTRICITY_PRICE_SENSORS
from .sensors.sensor_battery_forecast import BATTERY_FORECAST_SENSORS
from .sensors.sensor_battery_cost import BATTERY_COST_SENSORS

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> bool:
    """Set up Solar Forecast ML sensors from config entry by @Zara"""
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
            YesterdayDeviationSensor(coordinator, entry),
            LastCoordinatorUpdateSensor(coordinator, entry),
            LastMLTrainingSensor(coordinator, entry),
            NextScheduledUpdateSensor(coordinator, entry),
            MLServiceStatusSensor(coordinator, entry),
            MLMetricsSensor(coordinator, entry),
            CoordinatorHealthSensor(coordinator, entry),
            DataFilesStatusSensor(coordinator, entry),
            # Cloudiness Trend Sensors
            CloudinessTrend1hSensor(coordinator, entry),
            CloudinessTrend3hSensor(coordinator, entry),
            CloudinessVolatilitySensor(coordinator, entry),
            # Production Start Sensor
            NextProductionStartSensor(coordinator, entry),
            # State sensors use hass instead of coordinator
            ExternalSensorsStatusSensor(hass, entry),
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

    # ========================================================================
    # BATTERY MANAGEMENT SENSORS (v8.3.0)
    # ========================================================================
    battery_enabled = entry.options.get(CONF_BATTERY_ENABLED, False)
    electricity_enabled = entry.options.get(CONF_ELECTRICITY_ENABLED, False)

    if battery_enabled:
        # Get BatteryCoordinator (separate from Solar coordinator)
        battery_coordinator_key = f"{entry.entry_id}_battery"
        battery_coordinator = hass.data[DOMAIN].get(battery_coordinator_key)

        if battery_coordinator:
            # Battery sensors use Solar coordinator (legacy)
            battery_entities = [sensor_class(coordinator, entry) for sensor_class in BATTERY_SENSORS]
            battery_forecast_entities = [sensor_class(coordinator, entry) for sensor_class in BATTERY_FORECAST_SENSORS]

            # Battery COST sensors use BatteryCoordinator (new)
            battery_cost_entities = [sensor_class(battery_coordinator, entry) for sensor_class in BATTERY_COST_SENSORS]

            entities_to_add.extend(battery_entities)
            entities_to_add.extend(battery_forecast_entities)
            entities_to_add.extend(battery_cost_entities)

            _LOGGER.info(
                f"Battery Management enabled - Adding {len(battery_entities)} battery sensors, "
                f"{len(battery_forecast_entities)} forecast sensors, "
                f"{len(battery_cost_entities)} cost sensors"
            )
        else:
            _LOGGER.warning("Battery enabled but BatteryCoordinator not found - skipping battery sensors")

    if electricity_enabled:
        electricity_entities = [sensor_class(coordinator, entry) for sensor_class in ELECTRICITY_PRICE_SENSORS]
        entities_to_add.extend(electricity_entities)
        _LOGGER.info(f"Electricity Prices enabled - Adding {len(electricity_entities)} electricity price sensors")

    # ========================================================================

    # Register all selected entities
    async_add_entities(entities_to_add, True)
    _LOGGER.info(f"Successfully added {len(entities_to_add)} total sensors (Solar + Battery Management).")

    return True
