"""
Sensor platform for Solar Forecast ML Integration.
COMPLETE with all original sensors + 13 new diagnostic sensors
LIVE updates for external sensors, timestamps and production time
Version 6.0.1 - KORRIGIERTE VERSION (Fix für Attribut-Benennung) - by Zara & Gemini

Copyright (C) 2025 Zara-Toorox

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""
import logging
from datetime import timedelta
from typing import Any, Dict, Optional

from homeassistant.components.sensor import (
    SensorEntity,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy, PERCENTAGE, UnitOfTemperature, UnitOfSpeed
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN, 
    CONF_TEMP_SENSOR, 
    CONF_WIND_SENSOR, 
    CONF_RAIN_SENSOR,
    CONF_UV_SENSOR, 
    CONF_LUX_SENSOR,
    CONF_HOURLY,
    CONF_DIAGNOSTIC,  # <-- HINZUGEFÜGT
    UPDATE_INTERVAL,
    INTEGRATION_MODEL,
    SOFTWARE_VERSION,
    ML_VERSION, DAILY_UPDATE_HOUR, DAILY_VERIFICATION_HOUR
)
from .sensor_external_helpers import BaseExternalSensor, format_time_ago
from .helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


ML_STATE_TRANSLATIONS = {
    "uninitialized": "Not yet trained",
    "training": "Training in progress",
    "ready": "Ready",
    "degraded": "Limited",
    "error": "Error"
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Solar Forecast ML sensors."""
    coordinator = hass.data[DOMAIN][entry.entry_id]

    # === NEUE LOGIK START ===

    # 1. Hole den Optionswert für den Diagnose-Modus
    #    Default ist True, damit bestehende Installationen die Sensoren nicht verlieren
    diagnostic_mode_enabled = entry.options.get(CONF_DIAGNOSTIC, True)
    _LOGGER.info(f"Diagnostic mode is {'enabled' if diagnostic_mode_enabled else 'disabled'}")

    # 2. Erstelle separate Listen
    
    # Core-Sensoren, die immer hinzugefügt werden
    core_entities = [
        SolarForecastSensor(coordinator, entry, "today"),
        SolarForecastSensor(coordinator, entry, "tomorrow"),
        PeakProductionHourSensor(coordinator, entry),
        ProductionTimeSensor(coordinator, entry),
        AverageYieldSensor(coordinator, entry),
        AutarkySensor(coordinator, entry),
    ]

    # Diagnostische Sensoren, nur hinzufügen, wenn aktiviert
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
        
        # Externe Status-Sensoren (sind per Definition diagnostisch)
        ExternalTempSensor(coordinator, entry),
        ExternalHumiditySensor(coordinator, entry),
        ExternalWindSensor(coordinator, entry),
        ExternalRainSensor(coordinator, entry),
        ExternalUVSensor(coordinator, entry),
        ExternalLuxSensor(coordinator, entry),
        PowerSensorStateSensor(coordinator, entry),
        YieldSensorStateSensor(coordinator, entry),
    ]

    entities_to_add = core_entities

    # 3. Füge diagnostische Sensoren bedingt hinzu
    if diagnostic_mode_enabled:
        entities_to_add.extend(diagnostic_entities)
        _LOGGER.debug(f"Adding {len(diagnostic_entities)} diagnostic sensors.")
    else:
        _LOGGER.debug("Diagnostic sensors are disabled by options.")

    # 4. Stunden-Sensor-Prüfung (bleibt separat)
    enable_hourly = entry.options.get(CONF_HOURLY, entry.data.get(CONF_HOURLY, False))
    if enable_hourly:
        entities_to_add.append(NextHourSensor(coordinator, entry))

    async_add_entities(entities_to_add, True)
    
    # === NEUE LOGIK ENDE ===


class BaseSolarSensor(CoordinatorEntity, SensorEntity):

    _attr_has_entity_name = True

    def __init__(self, coordinator, entry: ConfigEntry):
        """Initialize the base sensor."""
        super().__init__(coordinator)
        self.entry = entry
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Solar Forecast ML", 
            manufacturer="Zara-Toorox",
            model=INTEGRATION_MODEL,
            sw_version=f"SW {SOFTWARE_VERSION} | ML {ML_VERSION}",
            configuration_url="https://github.com/Zara-Toorox/ha-solar-forecast-ml",
        )

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return self.coordinator.last_update_success


class SolarForecastSensor(BaseSolarSensor):

    def __init__(self, coordinator, entry: ConfigEntry, key: str):
        super().__init__(coordinator, entry)
        self._key = key
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        clean_name = "Solar Forecast Today" if key == "today" else "Solar Forecast Tomorrow"
        self._attr_name = clean_name
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:solar-power"

    @property
    def native_value(self):
        if self.coordinator.data:
            if self._key == "today":
                return self.coordinator.data.get("forecast_today", 0.0)
            elif self._key == "tomorrow":
                return self.coordinator.data.get("forecast_tomorrow", 0.0)
        return 0.0


class NextHourSensor(BaseSolarSensor):

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_next_hour"
        self._attr_name = "Next Hour Forecast"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:clock-fast"
        
    @property
    def native_value(self):
        value = getattr(self.coordinator, 'next_hour_pred', 0.0)
        return round(value, 2) if value is not None else 0.0


class PeakProductionHourSensor(BaseSolarSensor):

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_peak_production_hour"
        self._attr_name = "Best Hour for Consumption"
        self._attr_icon = "mdi:battery-charging-high"

    @property
    def native_value(self):
        return getattr(self.coordinator, 'peak_production_time_today', 'Unknown')


# =============================================================================
# KORREKTE VERSION: ProductionTimeSensor – NUR ANZEIGE AUS COORDINATOR
# =============================================================================
class ProductionTimeSensor(BaseSolarSensor):

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_production_time"
        self._attr_name = "Production Time Today"
        self._attr_icon = "mdi:timer-outline"

    @property
    def native_value(self):
        """Liest den Wert direkt aus dem Coordinator."""
        # Dieser Wert wird bereits im Coordinator durch den
        # ProductionTimeCalculator (aus production_calculator.py)
        # live aktualisiert und hier nur noch angezeigt.
        return getattr(self.coordinator, 'production_time_today', 'Initializing...')


class AverageYieldSensor(BaseSolarSensor):

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_average_yield"
        self._attr_name = "Average Monthly Yield"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_device_class = None
        self._attr_icon = "mdi:chart-line"

    @property
    def native_value(self):
        return getattr(self.coordinator, 'avg_month_yield', 0.0)


class AutarkySensor(BaseSolarSensor):

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_autarky"
        self._attr_name = "Self-Sufficiency"
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:home-battery"

    @property
    def native_value(self):
        return getattr(self.coordinator, 'autarky', 0.0)


class DiagnosticStatusSensor(BaseSolarSensor):

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_status"
        self._attr_name = "System Status"
        self._attr_icon = "mdi:information"

    @property
    def native_value(self):
        return getattr(self.coordinator, 'diagnostic_status', 'Unknown')


class SolarAccuracySensor(BaseSolarSensor):

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_accuracy"
        self._attr_name = "Yesterday Accuracy"
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:target"

    @property
    def native_value(self):
        return getattr(self.coordinator, 'yesterday_accuracy', 0.0)


class YesterdayDeviationSensor(BaseSolarSensor):

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_yesterday_deviation"
        self._attr_name = "Yesterday Deviation"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.MEASUREMENT
        # *** KORREKTUR: device_class entfernt, um state_class Konflikt zu lösen ***
        self._attr_device_class = None
        self._attr_icon = "mdi:delta"

    @property
    def native_value(self):
        return getattr(self.coordinator, 'yesterday_deviation', 0.0)


class LastCoordinatorUpdateSensor(BaseSolarSensor):

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_coordinator_update"
        self._attr_name = "Last Update Timestamp"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:clock-check"
        
    @property
    def native_value(self):
        return getattr(self.coordinator, 'last_update_time', None)
    
    @property
    def extra_state_attributes(self) -> dict:
        last_update = getattr(self.coordinator, 'last_update_time', None)
        return {
            "last_update_iso": last_update.isoformat() if last_update else None,
            "time_ago": format_time_ago(last_update) if last_update else "Never"
        }


class UpdateAgeSensor(BaseSolarSensor):

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_update_age"
        self._attr_name = "Data Age"
        self._attr_icon = "mdi:timer-sand"
        
    @property
    def native_value(self):
        last_update = getattr(self.coordinator, 'last_update_time', None)
        return format_time_ago(last_update) if last_update else "No data"
    
    @property
    def extra_state_attributes(self) -> dict:
        last_update = getattr(self.coordinator, 'last_update_time', None)
        if not last_update:
            return {"status": "no_data"}
        
        age = (dt_util.now() - last_update).total_seconds()
        return {
            "age_seconds": int(age),
            "age_minutes": round(age / 60, 1),
            "status": "fresh" if age < 600 else "stale" if age < 3600 else "outdated"
        }


class LastMLTrainingSensor(BaseSolarSensor):

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_ml_training"
        self._attr_name = "Last ML Training"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:brain"
        
    @property
    def native_value(self):
        ml_predictor = getattr(self.coordinator, 'ml_predictor', None)
        if not ml_predictor:
            return None
        return getattr(ml_predictor, 'last_training_time', None)
    
    @property
    def extra_state_attributes(self) -> dict:
        ml_predictor = getattr(self.coordinator, 'ml_predictor', None)
        if not ml_predictor:
            return {
                "status": "ML not available",
                "model_state": "unavailable"
            }
        
        last_training = getattr(ml_predictor, 'last_training_time', None)
        
        # *** KORREKTUR: Enum-Objekt in String (value) umwandeln ***
        model_state_obj = getattr(ml_predictor, 'model_state', None)
        model_state_val = 'unknown'
        if hasattr(model_state_obj, 'value'):
            model_state_val = model_state_obj.value
        elif isinstance(model_state_obj, str):
            model_state_val = model_state_obj
        
        attrs = {
            "model_state": model_state_val,
            "model_state_text": ML_STATE_TRANSLATIONS.get(model_state_val, 'Unknown')
        }
        
        if last_training:
            attrs["training_completed"] = True
            attrs["time_since_training"] = format_time_ago(last_training)
        else:
            attrs["training_completed"] = False
            attrs["note"] = "No training performed yet"
        
        return attrs


class NextScheduledUpdateSensor(BaseSolarSensor):

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_next_scheduled_update"
        self._attr_name = "Next Scheduled Update"
        self._attr_icon = "mdi:clock-outline"
        
    @property
    def native_value(self):
        now = dt_util.now()
        scheduled_updates = [
            f"{DAILY_UPDATE_HOUR:02d}:00",
            f"{DAILY_VERIFICATION_HOUR:02d}:00"
        ]
        current_time = now.strftime("%H:%M")
        
        for time_str in sorted(scheduled_updates):
            if time_str > current_time:
                return f"Today {time_str}"
        
        return f"Tomorrow {scheduled_updates[0]}"
    
    @property
    def extra_state_attributes(self) -> dict:
        return {
            "daily_update_hour": DAILY_UPDATE_HOUR,
            "daily_verification_hour": DAILY_VERIFICATION_HOUR,
            "update_interval_minutes": UPDATE_INTERVAL.total_seconds() / 60
        }


class MLServiceStatusSensor(BaseSolarSensor):

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_service_status"
        self._attr_name = "ML Service Status"
        self._attr_icon = "mdi:robot"
        
    @property
    def native_value(self):
        ml_predictor = getattr(self.coordinator, 'ml_predictor', None)
        if not ml_predictor:
            return "Unavailable"
        
        # *** KORREKTUR: Enum-Objekt in String (value) umwandeln ***
        model_state_obj = getattr(ml_predictor, 'model_state', None)
        model_state_val = 'unknown'
        if hasattr(model_state_obj, 'value'):
            model_state_val = model_state_obj.value
        elif isinstance(model_state_obj, str):
            model_state_val = model_state_obj
        
        return ML_STATE_TRANSLATIONS.get(model_state_val, model_state_val.capitalize())
    
    @property
    def extra_state_attributes(self) -> dict:
        ml_predictor = getattr(self.coordinator, 'ml_predictor', None)
        if not ml_predictor:
            return {"status": "unavailable"}
        
        # *** KORREKTUR: Enum-Objekt in String (value) umwandeln ***
        model_state_obj = getattr(ml_predictor, 'model_state', None)
        model_state_val = 'unknown'
        if hasattr(model_state_obj, 'value'):
            model_state_val = model_state_obj.value
        elif isinstance(model_state_obj, str):
            model_state_val = model_state_obj
            
        return {
            "model_state": model_state_val,
            # *** KORREKTUR 1: 'samples_count' zu 'training_samples' geändert ***
            "samples_count": getattr(ml_predictor, 'training_samples', 0),
            "model_loaded": hasattr(ml_predictor, 'model') and ml_predictor.model is not None
        }


class MLMetricsSensor(BaseSolarSensor):

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_metrics"
        self._attr_name = "ML Metrics"
        self._attr_icon = "mdi:chart-box"
        
    @property
    def native_value(self):
        ml_predictor = getattr(self.coordinator, 'ml_predictor', None)
        if not ml_predictor:
            return "No data"
        
        # *** KORREKTUR 2: 'samples_count' zu 'training_samples' geändert ***
        samples = getattr(ml_predictor, 'training_samples', 0)
        return f"{samples} samples"
    
    @property
    def extra_state_attributes(self) -> dict:
        ml_predictor = getattr(self.coordinator, 'ml_predictor', None)
        if not ml_predictor:
            return {"status": "unavailable"}
        
        return {
            # *** KORREKTUR 3: 'samples_count' zu 'training_samples' geändert ***
            "samples_collected": getattr(ml_predictor, 'training_samples', 0),
            "features_count": len(getattr(ml_predictor, 'feature_names', [])),
            "last_prediction_error": getattr(ml_predictor, 'last_error', None)
        }


class CoordinatorHealthSensor(BaseSolarSensor):

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_coordinator_health"
        self._attr_name = "Coordinator Health"
        self._attr_icon = "mdi:heart-pulse"
        
    @property
    def native_value(self):
        if not self.coordinator.last_update_success:
            return "Failed"
        
        last_update = getattr(self.coordinator, 'last_update_time', None)
        if not last_update:
            return "Initializing"
        
        age = (dt_util.now() - last_update).total_seconds()
        if age < 600:
            return "Healthy"
        elif age < 3600:
            return "Degraded"
        else:
            return "Stale"
    
    @property
    def extra_state_attributes(self) -> dict:
        return {
            "last_update_success": self.coordinator.last_update_success,
            "update_count": getattr(self.coordinator, 'update_count', 0),
            "error_count": getattr(self.coordinator, 'error_count', 0)
        }


class DataFilesStatusSensor(BaseSolarSensor):

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_data_files_status"
        self._attr_name = "Data Files Status"
        self._attr_icon = "mdi:file-document"
        
    @property
    def native_value(self):
        ml_predictor = getattr(self.coordinator, 'ml_predictor', None)
        if not ml_predictor:
            return "Unknown"
        
        # *** KORREKTUR 4: 'samples_count' zu 'training_samples' geändert ***
        samples_count = getattr(ml_predictor, 'training_samples', 0)
        if samples_count > 0:
            return "Available"
        return "Empty"
    
    @property
    def extra_state_attributes(self) -> dict:
        ml_predictor = getattr(self.coordinator, 'ml_predictor', None)
        if not ml_predictor:
            return {"status": "unknown"}
        
        return {
            # *** KORREKTUR 5: 'samples_count' zu 'training_samples' geändert ***
            "samples_count": getattr(ml_predictor, 'training_samples', 0),
            "data_source": "persistent_storage"
        }

#############################################################################
#   NEW BASE CLASS FOR ENTITY STATE SENSORS
#############################################################################
class BaseEntityStateSensor(BaseSolarSensor):
    """
    Base class for diagnostic sensors that track the state of another entity.
    - Is Diagnostic
    - Is always available
    - Tracks state changes of a source entity
    - Reports the raw state of the source entity
    """
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry, source_entity_id: Optional[str], unique_id_key: str, name: str, icon: str):
        """Initialize the state sensor."""
        super().__init__(coordinator, entry)
        self._source_entity_id = source_entity_id
        self._attr_unique_id = f"{entry.entry_id}_{unique_id_key}"
        self._attr_name = name
        self._attr_icon = icon
    
    @property
    def available(self) -> bool:
        """This sensor is always available to show its state."""
        return True
    
    async def async_added_to_hass(self) -> None:
        """Register state change listener."""
        await super().async_added_to_hass()
        if self._source_entity_id:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, [self._source_entity_id], self._handle_sensor_update
                )
            )
    
    @callback
    def _handle_sensor_update(self, event) -> None:
        """Handle state update."""
        self.async_write_ha_state()
    
    @property
    def native_value(self) -> str:
        """Return the state of the source entity."""
        if not self._source_entity_id:
            return "Not configured"
        
        state = self.hass.states.get(self._source_entity_id)
        if state is None:
            return "Entity not found"
        
        return state.state
    
    @property
    def extra_state_attributes(self) -> dict:
        """Return attributes of the source entity."""
        if not self._source_entity_id:
            return {"status": "not_configured", "entity_id": None}
        
        state = self.hass.states.get(self._source_entity_id)
        if state is None:
            return {"status": "entity_not_found", "entity_id": self._source_entity_id, "state": None}
        
        return {
            "status": "ok" if state.state not in ['unavailable', 'unknown'] else state.state,
            "entity_id": self._source_entity_id,
            "state": state.state,
            "unit_of_measurement": state.attributes.get('unit_of_measurement'),
            "last_updated": state.last_updated.isoformat() if state.last_updated else None
        }

#############################################################################
#   REFACTORED STATE SENSORS
#############################################################################

class ExternalTempSensor(BaseEntityStateSensor):
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id=entry.data.get(CONF_TEMP_SENSOR),
            unique_id_key="external_temp_state",
            name="External Temperature Sensor State",
            icon="mdi:thermometer-check"
        )

class ExternalHumiditySensor(BaseEntityStateSensor):
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id=entry.data.get('humidity_sensor'),
            unique_id_key="external_humidity_state",
            name="External Humidity Sensor State",
            icon="mdi:water-percent-alert"
        )

class ExternalWindSensor(BaseEntityStateSensor):
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id=entry.data.get(CONF_WIND_SENSOR),
            unique_id_key="external_wind_state",
            name="External Wind Sensor State",
            icon="mdi:weather-windy-variant"
        )

class ExternalRainSensor(BaseEntityStateSensor):
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id=entry.data.get(CONF_RAIN_SENSOR),
            unique_id_key="external_rain_state",
            name="External Rain Sensor State",
            icon="mdi:weather-rainy-check"
        )

class ExternalUVSensor(BaseEntityStateSensor):
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id=entry.data.get(CONF_UV_SENSOR),
            unique_id_key="external_uv_state",
            name="External UV Sensor State",
            icon="mdi:weather-sunny-alert"
        )

class ExternalLuxSensor(BaseEntityStateSensor):
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id=entry.data.get(CONF_LUX_SENSOR),
            unique_id_key="external_lux_state",
            name="External Illuminance Sensor State",
            icon="mdi:brightness-5-check"
        )

# ConfiguredPowerEntitySensor (Removed as requested)

# PowerSensorStateSensor is now refactored
class PowerSensorStateSensor(BaseEntityStateSensor):
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id=getattr(coordinator, 'power_entity', None),
            unique_id_key="power_sensor_state",
            name="Power Sensor State",
            icon="mdi:information"
        )
        
# ConfiguredYieldEntitySensor (Removed as requested)

# YieldSensorStateSensor is now refactored
class YieldSensorStateSensor(BaseEntityStateSensor):
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(
            coordinator,
            entry,
            source_entity_id=getattr(coordinator, 'solar_yield_today', None),
            unique_id_key="yield_sensor_state",
            name="Yield Sensor State",
            icon="mdi:information"
        )