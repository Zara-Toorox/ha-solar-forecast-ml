"""Sensor Platform for Solar Forecast ML Integration V12.2.0 @zara

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
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_DIAGNOSTIC,
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

    # Clean up entities that should no longer exist based on current config
    await _cleanup_orphaned_entities(hass, entry, diagnostic_mode_enabled)

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

    shadow_detection_entities = [
        sensor_class(coordinator, entry) for sensor_class in SHADOW_DETECTION_SENSORS
    ]
    entities_to_add.extend(shadow_detection_entities)
    _LOGGER.info(
        f"Shadow Detection enabled - Adding {len(shadow_detection_entities)} shadow detection sensors"
    )

    async_add_entities(entities_to_add, True)
    _LOGGER.info(f"Successfully added {len(entities_to_add)} total sensors.")

    return True


async def _cleanup_orphaned_entities(
    hass: HomeAssistant,
    entry: ConfigEntry,
    diagnostic_enabled: bool,
) -> None:
    """Remove entities from registry that should no longer exist based on config @zara

    This ensures that when a user disables diagnostic mode, the diagnostic sensors
    are properly removed and don't reappear after restart.
    """
    ent_reg = er.async_get(hass)

    # Patterns for diagnostic entities that should be removed when diagnostic mode is disabled
    diagnostic_patterns = [
        "diagnostic_status",
        "external_sensors_status",
        "next_production_start",
        "ml_service_status",
        "ml_metrics",
        "ml_training_readiness",
        "active_prediction_model",
        "pattern_count",
        "physics_samples",
    ]

    entities_removed = 0

    for entity_entry in list(ent_reg.entities.values()):
        # Only process entities for this config entry
        if entity_entry.config_entry_id != entry.entry_id:
            continue

        # Check if this is a diagnostic entity
        unique_id_lower = str(entity_entry.unique_id).lower()

        if not diagnostic_enabled:
            # Remove diagnostic entities when diagnostic mode is disabled
            for pattern in diagnostic_patterns:
                if pattern in unique_id_lower:
                    _LOGGER.debug(f"Removing disabled diagnostic entity: {entity_entry.entity_id}")
                    ent_reg.async_remove(entity_entry.entity_id)
                    entities_removed += 1
                    break

    if entities_removed > 0:
        _LOGGER.info(f"Cleaned up {entities_removed} orphaned entities based on current config")
