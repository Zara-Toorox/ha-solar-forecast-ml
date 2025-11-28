"""Sensor Platform for Solar Forecast ML Integration V10.0.0 @zara

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

from .const import (
    CONF_BATTERY_ENABLED,
    CONF_DIAGNOSTIC,
    CONF_ELECTRICITY_ENABLED,
    CONF_HOURLY,
    DOMAIN,
)

from .sensors.sensor_base import (
    AverageAccuracy30DaysSensor,
    AverageYield7DaysSensor,
    AverageYield30DaysSensor,
    ExpectedDailyProductionSensor,
    ForecastDayAfterTomorrowSensor,
    MaxPeakAllTimeSensor,
    MaxPeakTodaySensor,
    MonthlyYieldSensor,
    NextHourSensor,
    PeakProductionHourSensor,
    ProductionTimeSensor,
    SolarForecastSensor,
    WeeklyYieldSensor,
)

from .sensors.sensor_battery import BATTERY_SENSORS
from .sensors.sensor_battery_cost import BATTERY_COST_SENSORS
from .sensors.sensor_battery_forecast import BATTERY_FORECAST_SENSORS

from .sensors.sensor_diagnostic import (
    ActivePredictionModelSensor,
    DataFilesStatusSensor,
    DiagnosticStatusSensor,
    MLMetricsSensor,
    MLServiceStatusSensor,
    MLTrainingReadinessSensor,
    NextProductionStartSensor,
    PatternCountSensor,
    PhysicsSamplesSensor,
)

from .sensors.sensor_electricity_price import ELECTRICITY_PRICE_SENSORS

from .sensors.sensor_states import (
    ExternalSensorsStatusSensor,
    PowerSensorStateSensor,
    YieldSensorStateSensor,
)

from .sensors.sensor_system_status import SystemStatusSensor

from .sensors.sensor_shadow_detection import SHADOW_DETECTION_SENSORS

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> bool:
    """Set up Solar Forecast ML sensors from config entry"""
    coordinator = hass.data[DOMAIN][entry.entry_id]

    diagnostic_mode_enabled = entry.options.get(CONF_DIAGNOSTIC, True)
    enable_hourly = entry.options.get(CONF_HOURLY, False)

    _LOGGER.info(
        f"Setting up sensors: Diagnostic Mode={'Enabled' if diagnostic_mode_enabled else 'Disabled'}, "
        f"Hourly Sensor={'Enabled' if enable_hourly else 'Disabled'}"
    )

    system_status_sensor = SystemStatusSensor(coordinator, entry.entry_id)
    coordinator.system_status_sensor = system_status_sensor

    essential_production_entities = [
        system_status_sensor,
        ExpectedDailyProductionSensor(coordinator, entry),
        SolarForecastSensor(coordinator, entry, "remaining"),
        SolarForecastSensor(coordinator, entry, "tomorrow"),
        ForecastDayAfterTomorrowSensor(coordinator, entry),
        PeakProductionHourSensor(coordinator, entry),
        ProductionTimeSensor(coordinator, entry),
        MaxPeakTodaySensor(coordinator, entry),
        MaxPeakAllTimeSensor(coordinator, entry),
        PowerSensorStateSensor(hass, entry),
        YieldSensorStateSensor(hass, entry),
        NextHourSensor(coordinator, entry),
    ]

    entities_to_add = essential_production_entities

    statistics_entities = [
        AverageYield7DaysSensor(coordinator, entry),
        AverageYield30DaysSensor(coordinator, entry),
        WeeklyYieldSensor(coordinator, entry),
        MonthlyYieldSensor(coordinator, entry),
        AverageAccuracy30DaysSensor(coordinator, entry),

    ]
    entities_to_add.extend(statistics_entities)

    essential_diagnostic_entities = [
        DataFilesStatusSensor(coordinator, entry),
    ]
    entities_to_add.extend(essential_diagnostic_entities)

    if diagnostic_mode_enabled:
        diagnostic_entities = [
            DiagnosticStatusSensor(coordinator, entry),
            ExternalSensorsStatusSensor(hass, entry),
            NextProductionStartSensor(coordinator, entry),
            MLServiceStatusSensor(coordinator, entry),
            MLMetricsSensor(coordinator, entry),
            MLTrainingReadinessSensor(coordinator, entry),
            ActivePredictionModelSensor(coordinator, entry),
            PatternCountSensor(coordinator, entry),
            PhysicsSamplesSensor(coordinator, entry),
        ]
        entities_to_add.extend(diagnostic_entities)
        _LOGGER.info(f"Diagnostic mode enabled - Adding {len(diagnostic_entities)} advanced diagnostic sensors.")

    battery_enabled = entry.options.get(CONF_BATTERY_ENABLED, False)
    electricity_enabled = entry.options.get(CONF_ELECTRICITY_ENABLED, False)

    if battery_enabled:

        battery_coordinator_key = f"{entry.entry_id}_battery"
        battery_coordinator = hass.data[DOMAIN].get(battery_coordinator_key)

        if battery_coordinator:

            battery_entities = [
                sensor_class(coordinator, entry) for sensor_class in BATTERY_SENSORS
            ]
            battery_forecast_entities = [
                sensor_class(coordinator, entry) for sensor_class in BATTERY_FORECAST_SENSORS
            ]

            battery_cost_entities = [
                sensor_class(battery_coordinator, entry) for sensor_class in BATTERY_COST_SENSORS
            ]

            entities_to_add.extend(battery_entities)
            entities_to_add.extend(battery_forecast_entities)
            entities_to_add.extend(battery_cost_entities)

            _LOGGER.info(
                f"Battery Management enabled - Adding {len(battery_entities)} battery sensors, "
                f"{len(battery_forecast_entities)} forecast sensors, "
                f"{len(battery_cost_entities)} cost sensors"
            )
        else:
            _LOGGER.warning(
                "Battery enabled but BatteryCoordinator not found - skipping battery sensors"
            )

    if electricity_enabled:
        electricity_entities = [
            sensor_class(coordinator, entry) for sensor_class in ELECTRICITY_PRICE_SENSORS
        ]
        entities_to_add.extend(electricity_entities)
        _LOGGER.info(
            f"Electricity Prices enabled - Adding {len(electricity_entities)} electricity price sensors"
        )

    shadow_detection_entities = [
        sensor_class(coordinator, entry) for sensor_class in SHADOW_DETECTION_SENSORS
    ]
    entities_to_add.extend(shadow_detection_entities)
    _LOGGER.info(
        f"Shadow Detection enabled - Adding {len(shadow_detection_entities)} shadow detection sensors"
    )

    async_add_entities(entities_to_add, True)
    _LOGGER.info(
        f"Successfully added {len(entities_to_add)} total sensors (Solar + Battery Management)."
    )

    return True
