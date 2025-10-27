"""
Sensor platform für Solar Forecast ML Integration.
VOLLSTÄNDIG mit allen ursprünglichen Sensoren + 13 neuen Diagnose-Sensoren
LIVE-Updates für externe Sensoren, Zeitstempel und Produktionszeit
Version 4.10.0 - UTF-8 Fix + Status-Mapper + Erweiterte Anzeigen - von Zara

Copyright (C) 2025 Zara-Toorox

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
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN, 
    CONF_TEMP_SENSOR, 
    CONF_WIND_SENSOR, 
    CONF_RAIN_SENSOR,
    CONF_UV_SENSOR, 
    CONF_LUX_SENSOR,
    CONF_HOURLY,        # Fix: Import für Hourly Sensor Option - von Zara
    UPDATE_INTERVAL,
    INTEGRATION_MODEL,  # Fix: Device Info
    SOFTWARE_VERSION,   # Fix: Device Info
    ML_VERSION, DAILY_UPDATE_HOUR, DAILY_VERIFICATION_HOUR  # Tägliche Update-Zeiten - von Zara
)
from .sensor_external_helpers import BaseExternalSensor, format_time_ago

_LOGGER = logging.getLogger(__name__)


# Status-Mapper für ML Model States - von Zara
ML_STATE_TRANSLATIONS = {
    "uninitialized": "Noch nicht trainiert",
    "training": "Training läuft",
    "ready": "Einsatzbereit",
    "degraded": "Eingeschränkt",
    "error": "Fehler"
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Solar Forecast ML sensors."""
    coordinator = hass.data[DOMAIN][entry.entry_id]

    # • COMPLETE sensor suite - alle ursprünglichen Sensoren - von Zara
    entities_to_add = [
        # Original Status and diagnostic sensors - von Zara
        DiagnosticStatusSensor(coordinator, entry),
        SolarAccuracySensor(coordinator, entry),
        YesterdayDeviationSensor(coordinator, entry),
        
        # Main forecast sensors - von Zara
        SolarForecastSensor(coordinator, entry, "heute"),
        SolarForecastSensor(coordinator, entry, "morgen"),
        
        # Production analysis sensors - von Zara
        PeakProductionHourSensor(coordinator, entry),
        ProductionTimeSensor(coordinator, entry),
        AverageYieldSensor(coordinator, entry),
        AutarkySensor(coordinator, entry),
        
        # • NEU: Zeitstempel & Aktualität Diagnose-Sensoren (4 Sensoren) - von Zara
        LastCoordinatorUpdateSensor(coordinator, entry),
        UpdateAgeSensor(coordinator, entry),
        LastMLTrainingSensor(coordinator, entry),
        NextScheduledUpdateSensor(coordinator, entry),
        
        # • NEU: Service-Status Diagnose-Sensoren (3 Sensoren) - von Zara
        MLServiceStatusSensor(coordinator, entry),
        MLMetricsSensor(coordinator, entry),
        CoordinatorHealthSensor(coordinator, entry),
        DataFilesStatusSensor(coordinator, entry),
        
        # • NEU: Externe Sensoren Diagnose-Anzeige (6 Sensoren) - von Zara
        ExternalTempSensor(coordinator, entry),
        ExternalHumiditySensor(coordinator, entry),
        ExternalWindSensor(coordinator, entry),
        ExternalRainSensor(coordinator, entry),
        ExternalUVSensor(coordinator, entry),
        ExternalLuxSensor(coordinator, entry),
        
        ConfiguredPowerEntitySensor(coordinator, entry),
        PowerSensorStateSensor(coordinator, entry),
        ConfiguredYieldEntitySensor(coordinator, entry),
        YieldSensorStateSensor(coordinator, entry),
    ]

    # • Conditional hourly sensor - prüfe entry.data UND entry.options - von Zara
    enable_hourly = entry.options.get(CONF_HOURLY, entry.data.get(CONF_HOURLY, False))  # Fix: Nutze CONF_HOURLY Konstante - von Zara
    if enable_hourly:
        entities_to_add.append(NextHourSensor(coordinator, entry))

    async_add_entities(entities_to_add, True)


class BaseSolarSensor(CoordinatorEntity, SensorEntity):
    """• Safe base class for all sensors."""

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


# ============================================================================
# ORIGINAL SENSOREN (unverändert)
# ============================================================================

class SolarForecastSensor(BaseSolarSensor):
    """• Main forecast sensors (Heute/Morgen)."""

    def __init__(self, coordinator, entry: ConfigEntry, key: str):
        super().__init__(coordinator, entry)
        self._key = key
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        clean_name = "Solar Prognose Heute" if key == "heute" else "Solar Prognose Morgen"
        self._attr_name = clean_name
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:solar-power"

    @property
    def native_value(self):
        if self.coordinator.data:
            if self._key == "heute":
                return self.coordinator.data.get("forecast_today", 0.0)
            elif self._key == "morgen":
                return self.coordinator.data.get("forecast_tomorrow", 0.0)
        return 0.0


class NextHourSensor(BaseSolarSensor):
    """• Hourly forecast sensor."""

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_naechste_stunde"
        self._attr_name = "Prognose nächste Stunde"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:clock-fast"
        
    @property
    def native_value(self):
        # Sichere None-Prüfung vor round() - von Zara
        value = getattr(self.coordinator, 'next_hour_pred', 0.0)
        return round(value, 2) if value is not None else 0.0


class PeakProductionHourSensor(BaseSolarSensor):
    """• Best hour for consumption sensor."""

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_peak_production_hour"
        self._attr_name = "Beste Stunde für Verbraucher"
        self._attr_icon = "mdi:battery-charging-high"

    @property
    def native_value(self):
        return getattr(self.coordinator, 'peak_production_time_today', 'Unbekannt')


class ProductionTimeSensor(BaseSolarSensor):
    """• Daily production time sensor."""

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_production_time"
        self._attr_name = "Produktionszeit Heute"
        self._attr_icon = "mdi:timer-outline"
        self._timer_remove = None
        self._attr_native_value = "Initialisierung..."

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._timer_remove = async_track_time_interval(
            self.hass,
            self._async_update_value,
            timedelta(seconds=60)
        )
        self._async_update_value(None)

    async def async_will_remove_from_hass(self) -> None:
        await super().async_will_remove_from_hass()
        if self._timer_remove:
            self._timer_remove()
            self._timer_remove = None

    @callback
    def _async_update_value(self, now) -> None:
        if hasattr(self.coordinator, 'production_time_calculator'):
            self._attr_native_value = self.coordinator.production_time_calculator.get_production_time()
            self.async_write_ha_state()

    @property
    def native_value(self):
        return self._attr_native_value


class AverageYieldSensor(BaseSolarSensor):
    """• Average monthly yield sensor."""

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_average_yield"
        self._attr_name = "Durchschnittsertrag Monat"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:chart-line"

    @property
    def native_value(self):
        # Sichere None-Prüfung vor round() - von Zara
        value = getattr(self.coordinator, 'avg_month_yield', 0.0)
        return round(value, 2) if value is not None else 0.0


class AutarkySensor(BaseSolarSensor):
    """• Self-sufficiency sensor."""

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_autarky"
        self._attr_name = "Autarkie Heute"
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:home-battery"

    @property
    def native_value(self):
        # Sichere None-Prüfung vor round() - von Zara
        value = getattr(self.coordinator, 'autarky_today', 0.0)
        return round(value, 1) if value is not None else 0.0


class DiagnosticStatusSensor(BaseSolarSensor):
    """• Overall system status sensor."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_diagnostic_status"
        self._attr_name = "System Status"
        self._attr_icon = "mdi:information-outline"

    @property
    def native_value(self):
        return getattr(self.coordinator, 'diagnostic_status', 'Unbekannt')


class SolarAccuracySensor(BaseSolarSensor):
    """• Yesterday's accuracy sensor."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_yesterday_accuracy"
        self._attr_name = "Genauigkeit Gestern"
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:target"

    @property
    def native_value(self):
        # Sichere None-Prüfung vor round() - von Zara
        value = getattr(self.coordinator, 'yesterday_accuracy', 0.0)
        return round(value, 1) if value is not None else 0.0


class YesterdayDeviationSensor(BaseSolarSensor):
    """• Yesterday's deviation sensor."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_yesterday_deviation"
        self._attr_name = "Abweichung Gestern"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:delta"

    @property
    def native_value(self):
        # Sichere None-Prüfung vor round() - von Zara
        value = getattr(self.coordinator, 'yesterday_deviation', 0.0)
        return round(value, 2) if value is not None else 0.0


# ============================================================================
# • NEU: ZEITSTEMPEL & AKUALITÄT DIAGNOSE-SENSOREN - von Zara
# ============================================================================

class LastCoordinatorUpdateSensor(BaseSolarSensor):
    """Zeigt letztes Coordinator Update an - von Zara"""
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_coordinator_update"
        self._attr_name = "Letztes Update"
        self._attr_icon = "mdi:update"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
    
    @property
    def native_value(self):
        """Gibt Zeitstempel des letzten Updates zurück - von Zara"""
        return self.coordinator.last_update_time


class UpdateAgeSensor(BaseSolarSensor):
    """Zeigt Alter des letzten Updates in Minuten - von Zara"""
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_update_age"
        self._attr_name = "Update Alter"
        self._attr_icon = "mdi:clock-outline"
        self._attr_native_unit_of_measurement = "min"
    
    @property
    def native_value(self) -> int:
        """Berechnet Minuten seit letztem Update - von Zara"""
        if not self.coordinator.last_update_time:
            return None
        
        now = dt_util.now()
        age = (now - self.coordinator.last_update_time).total_seconds() / 60
        return int(age)


class LastMLTrainingSensor(BaseSolarSensor):
    """Zeigt Zeitpunkt des letzten ML-Trainings - von Zara"""
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_ml_training"
        self._attr_name = "Letztes ML-Training"
        self._attr_icon = "mdi:brain"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
    
    @property
    def available(self) -> bool:
        """
        ✓ VERBESSERT: Sensor nur verfügbar wenn ML trainiert wurde - von Zara
        Zeigt "unavailable" statt "Unbekannt" wenn noch nie trainiert
        """
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return False
        
        # Prüfe ob last_training_time existiert UND nicht None ist - von Zara
        if hasattr(ml_predictor, "last_training_time"):
            return ml_predictor.last_training_time is not None
        
        return False
    
    @property
    def native_value(self):
        """Holt letzten Trainingszeitpunkt vom ML Predictor - von Zara"""
        ml_predictor = self.coordinator.ml_predictor
        
        if not ml_predictor:
            return None
        
        if hasattr(ml_predictor, 'last_training_time'):
            return ml_predictor.last_training_time
        
        return None
    
    @property
    def extra_state_attributes(self) -> dict:
        """Zusätzliche Trainingsinfos - von Zara"""
        ml_predictor = self.coordinator.ml_predictor
        
        if not ml_predictor:
            return {
                "status": "ML nicht verfügbar",
                "hinweis": "Installieren Sie ML-Dependencies"
            }
        
        attrs = {}
        
        if hasattr(ml_predictor, 'last_training_time') and ml_predictor.last_training_time:
            attrs["training_durchgeführt"] = True
            attrs["letztes_training"] = ml_predictor.last_training_time.isoformat()
        else:
            attrs["training_durchgeführt"] = False
            attrs["hinweis"] = "Noch kein Training durchgeführt"
        
        if hasattr(ml_predictor, 'current_accuracy'):
            attrs["accuracy_percent"] = round(ml_predictor.current_accuracy * 100, 1)
        
        return attrs


class NextScheduledUpdateSensor(BaseSolarSensor):
    """Zeigt nächstes geplantes Update - von Zara"""
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_next_scheduled_update"
        self._attr_name = "Nächstes Update"
        self._attr_icon = "mdi:clock-check-outline"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
    
    @property
    def native_value(self):
        """Berechnet nächsten Update-Zeitpunkt - von Zara"""
        if not self.coordinator.last_update_time:
            return None
        
        # Nächstes Update = letztes Update + Interval - von Zara
        next_update = self.coordinator.last_update_time + UPDATE_INTERVAL
        return next_update


# ============================================================================
# • NEU: SERVICE-STATUS DIAGNOSE-SENSOREN - von Zara
# ============================================================================

class CoordinatorHealthSensor(BaseSolarSensor):
    """Zeigt Coordinator Health Status - von Zara"""
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_coordinator_health"
        self._attr_name = "Coordinator Status"
        self._attr_icon = "mdi:heart-pulse"
    
    @property
    def native_value(self) -> str:
        """Status basierend auf last_update_success - von Zara"""
        if self.coordinator.last_update_success:
            return "OK"
        return "Fehler"
    
    @property
    def extra_state_attributes(self) -> dict:
        """Zusätzliche Coordinator-Infos - von Zara"""
        return {
            "last_success": self.coordinator.last_update_success,
            "last_update": self.coordinator.last_update_time.isoformat() if self.coordinator.last_update_time else None,
            "update_interval_minutes": int(UPDATE_INTERVAL.total_seconds() / 60),
        }


class DataFilesStatusSensor(BaseSolarSensor):
    """Zeigt Status der Datendateien an - von Zara"""
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_data_files_status"
        self._attr_name = "Datendateien Status"
        self._attr_icon = "mdi:file-check"
        
        # Cache für async Daten - von Zara
        self._cached_status: Optional[dict] = None
        self._cached_file_count: int = 0
    
    async def async_added_to_hass(self) -> None:
        """Registriert Coordinator-Update-Listener - von Zara"""
        await super().async_added_to_hass()
        
        # Initiales Laden der Daten - von Zara
        await self._update_cached_data()
    
    @callback
    def _handle_coordinator_update(self) -> None:
        """Update Cache bei Coordinator-Änderungen - von Zara"""
        self.hass.async_create_task(self._update_cached_data())
        super()._handle_coordinator_update()
    
    async def _update_cached_data(self) -> None:
        """Lädt Daten async und cached sie - von Zara"""
        try:
            data_manager = getattr(self.coordinator, 'data_manager', None)
            
            if not data_manager:
                self._cached_status = None
                self._cached_file_count = 0
                return
            
            # Rufe async Methoden auf - von Zara
            if hasattr(data_manager, 'get_data_status'):
                self._cached_status = await data_manager.get_data_status()
            
            if hasattr(data_manager, 'get_data_files_count'):
                self._cached_file_count = await data_manager.get_data_files_count()
                
        except Exception as e:
            _LOGGER.warning(f"Fehler beim Aktualisieren der Datendatei-Infos: {e}")
            self._cached_status = None
            self._cached_file_count = 0
    
    @property
    def native_value(self) -> str:
        """Gibt gecachten Status zurück - von Zara"""
        if self._cached_status is None:
            return "Initialisierung..."
        
        if not self._cached_status.get('all_present', False):
            available = self._cached_status.get('available', 0)
            total = self._cached_status.get('total', 4)
            return f"Unvollständig ({available}/{total})"
        
        return "Vollständig"
    
    @property
    def extra_state_attributes(self) -> dict:
        """Detailinfos zu Datendateien aus Cache - von Zara"""
        if self._cached_status is None:
            return {"status": "Wird geladen..."}
        
        data_manager = getattr(self.coordinator, 'data_manager', None)
        
        attrs = {
            "dateien_gesamt": self._cached_status.get('total', 4),
            "dateien_vorhanden": self._cached_status.get('available', 0),
            "status_text": self._cached_status.get('status_text', 'Unbekannt'),
        }
        
        # Füge Details zu einzelnen Dateien hinzu - von Zara
        files = self._cached_status.get('files', {})
        for file_name, exists in files.items():
            attrs[f"datei_{file_name}"] = "Vorhanden" if exists else "Fehlend"
        
        # Füge Verzeichnispfad hinzu - von Zara
        if data_manager and hasattr(data_manager, 'get_data_directory'):
            attrs["verzeichnis"] = str(data_manager.get_data_directory())
        
        return attrs


class MLServiceStatusSensor(BaseSolarSensor):
    """Zeigt ML Service Status mit übersetzten States - von Zara"""
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_service_status"
        self._attr_name = "ML Service Status"
        self._attr_icon = "mdi:robot"
    
    @property
    def native_value(self) -> str:
        """Übersetzter ML State mit detaillierten Fehlermeldungen - von Zara"""
        ml_predictor = self.coordinator.ml_predictor
        
        if not ml_predictor:
            return "ML-Dependencies fehlen"
        
        try:
            if hasattr(ml_predictor, 'get_model_health'):
                health = ml_predictor.get_model_health()
                state_value = health.state.value if health.state else "unknown"
                return ML_STATE_TRANSLATIONS.get(state_value, state_value.capitalize())
            return "Status nicht abrufbar"
        except Exception as e:
            _LOGGER.debug(f"ML Service Status Fehler: {e}")
            return "Fehler beim Abruf"
    
    @property
    def extra_state_attributes(self) -> dict:
        """Basis ML-Infos mit erweiterten Diagnosedaten - von Zara"""
        ml_predictor = self.coordinator.ml_predictor
        
        if not ml_predictor:
            return {
                "status": "ML Predictor nicht initialisiert",
                "hinweis": "Prüfen Sie ob sklearn, pandas, numpy installiert sind",
                "model_loaded": False
            }
        
        try:
            if hasattr(ml_predictor, 'get_model_health'):
                health = ml_predictor.get_model_health()
                state_raw = health.state.value if health.state else "unknown"
                
                return {
                    "model_state_raw": state_raw,
                    "model_state_übersetzt": ML_STATE_TRANSLATIONS.get(state_raw, state_raw),
                    "model_loaded": health.model_loaded,
                    "training_samples": health.training_samples,
                    "initialisiert": True
                }
            
            return {
                "model_loaded": getattr(ml_predictor, 'model_loaded', False),
                "status": "Basis-Info",
                "initialisiert": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "Fehler beim Abruf"
            }


class MLMetricsSensor(BaseSolarSensor):
    """Zeigt detaillierte ML-Metriken - von Zara"""
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_metrics"
        self._attr_name = "ML Metriken"
        self._attr_icon = "mdi:chart-line"
    
    @property
    def native_value(self) -> str:
        """Zeigt aktuellen Model State übersetzt - von Zara"""
        ml_predictor = self.coordinator.ml_predictor
        
        if not ml_predictor:
            return "ML nicht verfügbar"
        
        try:
            if hasattr(ml_predictor, 'get_model_health'):
                health = ml_predictor.get_model_health()
                state_value = health.state.value if health.state else "unknown"
                return ML_STATE_TRANSLATIONS.get(state_value, state_value.capitalize())
            return "Keine Metriken"
        except Exception:
            return "Fehler"
    
    @property
    def extra_state_attributes(self) -> dict:
        """Detaillierte ML-Metriken als Attributes - von Zara"""
        ml_predictor = self.coordinator.ml_predictor
        
        if not ml_predictor:
            return {
                "status": "ML Predictor nicht verfügbar",
                "model_loaded": False,
                "hinweis": "Installieren Sie: pip install scikit-learn pandas numpy"
            }
        
        try:
            # Hole Model Health - von Zara
            if hasattr(ml_predictor, 'get_model_health'):
                health = ml_predictor.get_model_health()
                
                # Übersetze State - von Zara
                state_raw = health.state.value if health.state else "unknown"
                state_translated = ML_STATE_TRANSLATIONS.get(state_raw, state_raw.capitalize())
                
                return {
                    "model_state": state_translated,
                    "model_state_raw": state_raw,
                    "model_loaded": health.model_loaded,
                    "last_training": health.last_training.isoformat() if health.last_training else None,
                    "current_accuracy": round(health.current_accuracy * 100, 1),
                    "training_samples": health.training_samples,
                    "feature_count": len(health.features_available),
                    "features": ", ".join(health.features_available[:5]) + "..." if len(health.features_available) > 5 else ", ".join(health.features_available),
                    "avg_prediction_time_ms": round(health.performance_metrics.get("avg_prediction_time", 0) * 1000, 2),
                    "error_rate_percent": round(health.performance_metrics.get("error_rate", 0) * 100, 1),
                    "memory_usage_mb": round(health.performance_metrics.get("memory_usage_mb", 0), 1),
                }
            
            # Fallback wenn get_model_health nicht verfügbar - von Zara
            return {
                "model_loaded": getattr(ml_predictor, 'model_loaded', False),
                "current_accuracy": round(getattr(ml_predictor, 'current_accuracy', 0) * 100, 1),
                "training_samples": getattr(ml_predictor, 'training_samples', 0),
                "status": "Basis-Metriken"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "Fehler beim Abrufen der Metriken"
            }


class ExternalTempSensor(BaseExternalSensor, BaseSolarSensor):
    """Zeigt Wert des konfigurierten Temperatur-Sensors - von Zara"""
    
    def __init__(self, coordinator, entry: ConfigEntry):
        """Initialisiert Temperatur-Sensor - von Zara"""
        sensor_config = {
            'config_key': CONF_TEMP_SENSOR,
            'unique_id_suffix': 'external_temp',
            'name': 'Temperatur',
            'icon': 'mdi:thermometer',
            'default_unit': '°C'
        }
        BaseSolarSensor.__init__(self, coordinator, entry)
        BaseExternalSensor.__init__(self, coordinator, entry, sensor_config)


class ExternalHumiditySensor(BaseSolarSensor):
    """Zeigt Luftfeuchtigkeit von Weather Entity - von Zara"""
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_external_humidity"
        self._attr_name = "Luftfeuchtigkeit"
        self._attr_icon = "mdi:water-percent"
    
    @property
    def available(self) -> bool:
        """Sensor immer verfügbar - von Zara"""
        return True
    
    async def async_added_to_hass(self) -> None:
        """Register state change listener für LIVE-Updates - von Zara"""
        await super().async_added_to_hass()
        
        weather_entity = self.coordinator.current_weather_entity
        if weather_entity:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass,
                    [weather_entity],
                    self._handle_external_sensor_update
                )
            )
    
    @callback
    def _handle_external_sensor_update(self, event) -> None:
        """Update bei Änderung - von Zara"""
        self.async_write_ha_state()
    
    @property
    def native_value(self) -> str:
        """Holt Luftfeuchtigkeit von Weather Entity - von Zara"""
        weather_entity = self.coordinator.current_weather_entity
        
        if not weather_entity:
            return "Keine Weather Entity konfiguriert"
        
        state = self.coordinator.hass.states.get(weather_entity)
        if not state or state.state in ['unavailable', 'unknown', 'none', None]:
            return "Weather Entity nicht verfügbar"
        
        try:
            humidity = state.attributes.get('humidity')
            if humidity is None:
                return "Keine Luftfeuchtigkeit verfügbar"
            
            time_ago = format_time_ago(state.last_changed)
            return f"{humidity} % ({time_ago})"
            
        except Exception as e:
            _LOGGER.warning(f"Fehler beim Lesen der Luftfeuchtigkeit: {e}")
            return "Fehler beim Auslesen"


class ExternalWindSensor(BaseExternalSensor, BaseSolarSensor):
    """Zeigt Wert des konfigurierten Wind-Sensors - von Zara"""
    
    def __init__(self, coordinator, entry: ConfigEntry):
        """Initialisiert Wind-Sensor - von Zara"""
        sensor_config = {
            'config_key': CONF_WIND_SENSOR,
            'unique_id_suffix': 'external_wind',
            'name': 'Windgeschwindigkeit',
            'icon': 'mdi:weather-windy',
            'default_unit': 'km/h'
        }
        BaseSolarSensor.__init__(self, coordinator, entry)
        BaseExternalSensor.__init__(self, coordinator, entry, sensor_config)


class ExternalRainSensor(BaseExternalSensor, BaseSolarSensor):
    """Zeigt Wert des konfigurierten Regen-Sensors - von Zara"""
    
    def __init__(self, coordinator, entry: ConfigEntry):
        """Initialisiert Regen-Sensor - von Zara"""
        sensor_config = {
            'config_key': CONF_RAIN_SENSOR,
            'unique_id_suffix': 'external_rain',
            'name': 'Niederschlag',
            'icon': 'mdi:weather-rainy',
            'default_unit': 'mm'
        }
        BaseSolarSensor.__init__(self, coordinator, entry)
        BaseExternalSensor.__init__(self, coordinator, entry, sensor_config)


class ExternalUVSensor(BaseExternalSensor, BaseSolarSensor):
    """Zeigt Wert des konfigurierten UV-Index-Sensors - von Zara"""
    
    def __init__(self, coordinator, entry: ConfigEntry):
        """Initialisiert UV-Sensor - von Zara"""
        sensor_config = {
            'config_key': CONF_UV_SENSOR,
            'unique_id_suffix': 'external_uv',
            'name': 'UV-Index',
            'icon': 'mdi:weather-sunny-alert',
            'default_unit': None,
            'format_string': 'UV-Index: {value} ({time})'
        }
        BaseSolarSensor.__init__(self, coordinator, entry)
        BaseExternalSensor.__init__(self, coordinator, entry, sensor_config)


class ExternalLuxSensor(BaseExternalSensor, BaseSolarSensor):
    """Zeigt Wert des konfigurierten Lux/Helligkeits-Sensors - von Zara"""
    
    def __init__(self, coordinator, entry: ConfigEntry):
        """Initialisiert Lux-Sensor - von Zara"""
        sensor_config = {
            'config_key': CONF_LUX_SENSOR,
            'unique_id_suffix': 'external_lux',
            'name': 'Helligkeit',
            'icon': 'mdi:brightness-6',
            'default_unit': 'lx'
        }
        BaseSolarSensor.__init__(self, coordinator, entry)
        BaseExternalSensor.__init__(self, coordinator, entry, sensor_config)


class ConfiguredPowerEntitySensor(BaseSolarSensor):
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_configured_power_entity"
        self._attr_name = "Konfigurierte Power Entity"
        self._attr_icon = "mdi:lightning-bolt"
    
    @property
    def available(self) -> bool:
        return True
    
    @property
    def native_value(self) -> str:
        power_entity = getattr(self.coordinator, 'power_entity', None)
        if not power_entity:
            return "Nicht konfiguriert"
        return str(power_entity)
    
    @property
    def extra_state_attributes(self) -> dict:
        power_entity = getattr(self.coordinator, 'power_entity', None)
        if not power_entity:
            return {"status": "not_configured"}
        
        state = self.hass.states.get(power_entity)
        if state is None:
            return {
                "status": "entity_not_found",
                "entity_exists": False
            }
        
        return {
            "status": "configured",
            "entity_exists": True,
            "entity_id": power_entity
        }


class PowerSensorStateSensor(BaseSolarSensor):
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_power_sensor_state"
        self._attr_name = "Power Sensor State"
        self._attr_icon = "mdi:information"
    
    @property
    def available(self) -> bool:
        return True
    
    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        
        power_entity = getattr(self.coordinator, 'power_entity', None)
        if power_entity:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass,
                    [power_entity],
                    self._handle_sensor_update
                )
            )
    
    @callback
    def _handle_sensor_update(self, event) -> None:
        self.async_write_ha_state()
    
    @property
    def native_value(self) -> str:
        power_entity = getattr(self.coordinator, 'power_entity', None)
        
        if not power_entity:
            return "Nicht konfiguriert"
        
        state = self.hass.states.get(power_entity)
        if state is None:
            return "Entity nicht gefunden"
        
        return state.state
    
    @property
    def extra_state_attributes(self) -> dict:
        power_entity = getattr(self.coordinator, 'power_entity', None)
        
        if not power_entity:
            return {
                "status": "not_configured",
                "entity_id": None
            }
        
        state = self.hass.states.get(power_entity)
        if state is None:
            return {
                "status": "entity_not_found",
                "entity_id": power_entity,
                "state": None
            }
        
        return {
            "status": "ok" if state.state not in ['unavailable', 'unknown'] else state.state,
            "entity_id": power_entity,
            "state": state.state,
            "unit_of_measurement": state.attributes.get('unit_of_measurement'),
            "last_updated": state.last_updated.isoformat() if state.last_updated else None
        }


class ConfiguredYieldEntitySensor(BaseSolarSensor):
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_configured_yield_entity"
        self._attr_name = "Konfigurierte Yield Entity"
        self._attr_icon = "mdi:solar-power"
    
    @property
    def available(self) -> bool:
        return True
    
    @property
    def native_value(self) -> str:
        yield_entity = getattr(self.coordinator, 'solar_yield_today', None)
        if not yield_entity:
            return "Nicht konfiguriert"
        return str(yield_entity)
    
    @property
    def extra_state_attributes(self) -> dict:
        yield_entity = getattr(self.coordinator, 'solar_yield_today', None)
        if not yield_entity:
            return {"status": "not_configured"}
        
        state = self.hass.states.get(yield_entity)
        if state is None:
            return {
                "status": "entity_not_found",
                "entity_exists": False
            }
        
        return {
            "status": "configured",
            "entity_exists": True,
            "entity_id": yield_entity
        }


class YieldSensorStateSensor(BaseSolarSensor):
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_yield_sensor_state"
        self._attr_name = "Yield Sensor State"
        self._attr_icon = "mdi:information"
    
    @property
    def available(self) -> bool:
        return True
    
    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        
        yield_entity = getattr(self.coordinator, 'solar_yield_today', None)
        if yield_entity:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass,
                    [yield_entity],
                    self._handle_sensor_update
                )
            )
    
    @callback
    def _handle_sensor_update(self, event) -> None:
        self.async_write_ha_state()
    
    @property
    def native_value(self) -> str:
        yield_entity = getattr(self.coordinator, 'solar_yield_today', None)
        
        if not yield_entity:
            return "Nicht konfiguriert"
        
        state = self.hass.states.get(yield_entity)
        if state is None:
            return "Entity nicht gefunden"
        
        return state.state
    
    @property
    def extra_state_attributes(self) -> dict:
        yield_entity = getattr(self.coordinator, 'solar_yield_today', None)
        
        if not yield_entity:
            return {
                "status": "not_configured",
                "entity_id": None
            }
        
        state = self.hass.states.get(yield_entity)
        if state is None:
            return {
                "status": "entity_not_found",
                "entity_id": yield_entity,
                "state": None
            }
        
        return {
            "status": "ok" if state.state not in ['unavailable', 'unknown'] else state.state,
            "entity_id": yield_entity,
            "state": state.state,
            "unit_of_measurement": state.attributes.get('unit_of_measurement'),
            "last_updated": state.last_updated.isoformat() if state.last_updated else None
        }