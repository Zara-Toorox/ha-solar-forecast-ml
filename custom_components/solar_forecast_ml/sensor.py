# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

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
    AIRmseSensor,
    DataFilesStatusSensor,
    MLMetricsSensor,
    NextProductionStartSensor,
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
            ExternalSensorsStatusSensor(hass, entry),
            NextProductionStartSensor(coordinator, entry),
            MLMetricsSensor(coordinator, entry),
            AIRmseSensor(coordinator, entry),
            ActivePredictionModelSensor(coordinator, entry),
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
