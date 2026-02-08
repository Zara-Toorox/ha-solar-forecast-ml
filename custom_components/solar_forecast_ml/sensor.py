# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Solar Forecast ML V16.2.0 - Sensor Platform.

async_setup_entry for sensor platform - creates all sensor entities.
All data operations use DatabaseManager (no JSON).

@zara
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
    VERSION,
)

from .sensors.sensor_base import (
    AverageAccuracy30DaysSensor,
    AverageYield7DaysSensor,
    AverageYield30DaysSensor,
    AverageYieldSensor,
    ExpectedDailyProductionSensor,
    ForecastDayAfterTomorrowSensor,
    MaxPeakAllTimeSensor,
    MaxPeakTodaySensor,
    MonthlyConsumptionSensor,
    MonthlyYieldSensor,
    NextHourSensor,
    PeakProductionHourSensor,
    ProductionTimeSensor,
    SolarForecastSensor,
    WeeklyConsumptionSensor,
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
from .sensors.sensor_drift_detection import DRIFT_DETECTION_SENSORS

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> bool:
    """Set up Solar Forecast ML sensors from config entry. @zara

    Creates all sensor entities based on configuration options.
    - Essential production sensors (always created)
    - Statistics sensors (always created)
    - Diagnostic sensors (if diagnostic mode enabled)
    - Shadow detection sensors (always created)
    """
    coordinator = hass.data[DOMAIN][entry.entry_id]

    diagnostic_mode_enabled = entry.options.get(CONF_DIAGNOSTIC, True)
    enable_hourly = entry.options.get(CONF_HOURLY, False)

    _LOGGER.info(
        f"Setting up sensors V{VERSION}: "
        f"Diagnostic Mode={'Enabled' if diagnostic_mode_enabled else 'Disabled'}, "
        f"Hourly Sensor={'Enabled' if enable_hourly else 'Disabled'}"
    )

    # Clean up orphaned entities @zara
    await _cleanup_orphaned_entities(hass, entry, diagnostic_mode_enabled)

    # Create system status sensor and connect to coordinator @zara
    system_status_sensor = SystemStatusSensor(coordinator, entry.entry_id)
    coordinator.system_status_sensor = system_status_sensor

    # Essential production sensors (always created) @zara
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

    # Statistics sensors (always created) @zara
    statistics_entities = [
        AverageYieldSensor(coordinator, entry),
        AverageYield7DaysSensor(coordinator, entry),
        AverageYield30DaysSensor(coordinator, entry),
        WeeklyYieldSensor(coordinator, entry),
        WeeklyConsumptionSensor(coordinator, entry),
        MonthlyYieldSensor(coordinator, entry),
        MonthlyConsumptionSensor(coordinator, entry),
        AverageAccuracy30DaysSensor(coordinator, entry),
    ]
    entities_to_add.extend(statistics_entities)

    # Essential diagnostic entities (always created) @zara
    essential_diagnostic_entities = [
        DataFilesStatusSensor(coordinator, entry),
    ]
    entities_to_add.extend(essential_diagnostic_entities)

    # Advanced diagnostic sensors (only if diagnostic mode enabled) @zara
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
        _LOGGER.info(
            f"Diagnostic mode enabled - Adding {len(diagnostic_entities)} advanced diagnostic sensors."
        )

    # Shadow detection sensors (always created) @zara
    shadow_detection_entities = [
        sensor_class(coordinator, entry) for sensor_class in SHADOW_DETECTION_SENSORS
    ]
    entities_to_add.extend(shadow_detection_entities)
    _LOGGER.info(
        f"Shadow Detection enabled - Adding {len(shadow_detection_entities)} shadow detection sensors"
    )

    # V17.0.0: Drift detection sensor (diagnostic, always created) @zara
    drift_detection_entities = [
        sensor_class(coordinator, entry) for sensor_class in DRIFT_DETECTION_SENSORS
    ]
    entities_to_add.extend(drift_detection_entities)
    _LOGGER.info(
        f"Drift Detection enabled - Adding {len(drift_detection_entities)} drift detection sensors"
    )

    async_add_entities(entities_to_add, True)
    _LOGGER.info(f"Successfully added {len(entities_to_add)} total sensors.")

    return True


async def _cleanup_orphaned_entities(
    hass: HomeAssistant,
    entry: ConfigEntry,
    diagnostic_enabled: bool,
) -> None:
    """Remove entities from registry that should no longer exist based on config. @zara

    Ensures that when a user disables diagnostic mode, the diagnostic sensors
    are properly removed and don't reappear after restart.
    """
    ent_reg = er.async_get(hass)

    # Patterns for diagnostic entities to remove when diagnostic mode is disabled @zara
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
        # Only process entities for this config entry @zara
        if entity_entry.config_entry_id != entry.entry_id:
            continue

        unique_id_lower = str(entity_entry.unique_id).lower()

        if not diagnostic_enabled:
            # Remove diagnostic entities when diagnostic mode is disabled @zara
            for pattern in diagnostic_patterns:
                if pattern in unique_id_lower:
                    _LOGGER.debug(
                        f"Removing disabled diagnostic entity: {entity_entry.entity_id}"
                    )
                    ent_reg.async_remove(entity_entry.entity_id)
                    entities_removed += 1
                    break

    if entities_removed > 0:
        _LOGGER.info(
            f"Cleaned up {entities_removed} orphaned entities based on current config"
        )
