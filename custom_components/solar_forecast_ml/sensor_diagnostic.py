"""
Diagnostic Sensor platform for Solar Forecast ML Integration.
Contains sensors for monitoring and troubleshooting.

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
from datetime import datetime, timedelta # Import timedelta
from typing import Any, Dict, Optional

from homeassistant.components.sensor import (
    SensorEntity,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy, PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

# Import BaseSolarSensor from the core sensor file
from .sensor_core import BaseSolarSensor
from .coordinator import SolarForecastMLCoordinator
from .helpers import SafeDateTimeUtil as dt_util # Keep alias for function calls
from .sensor_external_helpers import format_time_ago # Helper for timestamps
from .const import UPDATE_INTERVAL, DAILY_UPDATE_HOUR, DAILY_VERIFICATION_HOUR
from .ml_predictor import ModelState # Import Enum for state mapping

_LOGGER = logging.getLogger(__name__)

# Translations for ML state enum
ML_STATE_TRANSLATIONS = {
    ModelState.UNINITIALIZED.value: "Not yet trained",
    ModelState.TRAINING.value: "Training in progress",
    ModelState.READY.value: "Ready",
    ModelState.DEGRADED.value: "Degraded",
    ModelState.ERROR.value: "Error",
    "unavailable": "Unavailable", # Add state for when predictor is None
    "unknown": "Unknown" # Fallback
}


# --- Diagnostic Sensors ---

class DiagnosticStatusSensor(BaseSolarSensor):
    """Sensor showing the overall diagnostic status of the coordinator."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the diagnostic status sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_diagnostic_status" # Changed from _status to avoid clash
        self._attr_name = "System Diagnostic Status"
        self._attr_icon = "mdi:stethoscope" # More diagnostic icon

    @property
    def native_value(self) -> str | None:
        """Return the diagnostic status string."""
        # Value is directly on the coordinator instance
        return getattr(self.coordinator, 'diagnostic_status', None)


class SolarAccuracySensor(BaseSolarSensor):
    """Sensor showing the forecast accuracy from the previous day."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the accuracy sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_yesterday_accuracy" # Use clearer ID
        self._attr_name = "Yesterday Forecast Accuracy"
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:target-arrow" # More specific icon

    @property
    def native_value(self) -> float | None:
        """Return the accuracy percentage."""
        # Get value from coordinator, default to None if not set
        accuracy = getattr(self.coordinator, 'yesterday_accuracy', None)
        # Ensure value is within 0-100 range if it exists
        return max(0.0, min(100.0, accuracy)) if accuracy is not None else None


class YesterdayDeviationSensor(BaseSolarSensor):
    """Sensor showing the absolute forecast deviation (error) from the previous day."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the deviation sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_yesterday_deviation"
        self._attr_name = "Yesterday Forecast Deviation"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_device_class = None # Avoid conflict with state_class
        self._attr_icon = "mdi:delta"

    @property
    def native_value(self) -> float | None:
        """Return the deviation in kWh."""
        # Get value from coordinator, default to None if not set
        deviation = getattr(self.coordinator, 'last_day_error_kwh', None) # Use the correct attribute name
        return max(0.0, deviation) if deviation is not None else None # Ensure non-negative


class LastCoordinatorUpdateSensor(BaseSolarSensor):
    """Sensor showing the timestamp of the last successful coordinator update."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the last update sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_coordinator_update"
        self._attr_name = "Last Update Timestamp"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:clock-check-outline" # Changed icon slightly

    @property
    def native_value(self) -> datetime | None: # Use standard datetime hint
        """Return the timestamp of the last successful update."""
        # Use last_update_success_time for accuracy
        return getattr(self.coordinator, 'last_update_success_time', None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context like time ago."""
        last_update = getattr(self.coordinator, 'last_update_success_time', None)
        # Use coordinator.last_update from base class for last attempt time
        last_attempt = getattr(self.coordinator, 'last_update', None) # Base coordinator attribute
        return {
            "last_update_iso": last_update.isoformat() if last_update else None,
            "time_ago": format_time_ago(last_update) if last_update else "Never",
            "last_attempt_iso": last_attempt.isoformat() if last_attempt else None, # Use base attribute
        }


class UpdateAgeSensor(BaseSolarSensor):
    """Sensor showing how long ago the last successful update occurred."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the update age sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_update_age"
        self._attr_name = "Data Age"
        self._attr_icon = "mdi:timer-sand"
        # No unit, state is a formatted string

    @property
    def native_value(self) -> str:
        """Return the formatted time since the last successful update."""
        last_update = getattr(self.coordinator, 'last_update_success_time', None)
        if last_update:
            # Use utcnow for age calculation
            return format_time_ago(last_update)
        elif getattr(self.coordinator, 'last_update', None): # Check if any update attempt was made
             return "Update Failed"
        else:
             return "No data yet"


    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide age in seconds and a status."""
        last_update = getattr(self.coordinator, 'last_update_success_time', None)
        if not last_update:
            status = "initializing" if not getattr(self.coordinator, 'last_update', None) else "failed"
            return {"status": status, "age_seconds": None, "age_minutes": None}

        age_delta = dt_util.utcnow() - last_update # Use utcnow
        age_seconds = age_delta.total_seconds()

        status = "fresh"
        # Compare age to the actual update interval being used by coordinator
        interval_seconds = self.coordinator.update_interval.total_seconds() if self.coordinator.update_interval else UPDATE_INTERVAL.total_seconds()
        if age_seconds > (interval_seconds * 1.5): # Older than 1.5 intervals
            status = "stale"
        if age_seconds > 3600: # Older than 1 hour is definitely outdated
            status = "outdated"

        return {
            "age_seconds": int(age_seconds),
            "age_minutes": round(age_seconds / 60, 1),
            "status": status,
            "last_success_iso": last_update.isoformat()
        }


class LastMLTrainingSensor(BaseSolarSensor):
    """Sensor showing the timestamp of the last successful ML model training."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the last ML training sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_ml_training"
        self._attr_name = "Last ML Training"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:brain"

    @property
    def native_value(self) -> datetime | None: # Use standard datetime hint
        """Return the timestamp of the last successful training."""
        # Value is directly on the coordinator instance (synced from predictor)
        return getattr(self.coordinator, 'last_successful_learning', None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide model state and time since training."""
        ml_predictor = getattr(getattr(self.coordinator, 'service_manager', {}), 'ml_predictor', None) # Access via service_manager
        if not ml_predictor:
            return {
                "status": "ML unavailable",
                "model_state": "unavailable",
                "model_state_text": ML_STATE_TRANSLATIONS["unavailable"],
                "time_since_training": None,
                "training_completed": False
            }

        last_training = getattr(ml_predictor, 'last_training_time', None) # Get directly from predictor

        # Get model state enum and convert its value to string
        model_state_obj = getattr(ml_predictor, 'model_state', ModelState.UNINITIALIZED)
        model_state_val = model_state_obj.value if isinstance(model_state_obj, ModelState) else str(model_state_obj)

        attrs = {
            "status": "ML available",
            "model_state": model_state_val,
            "model_state_text": ML_STATE_TRANSLATIONS.get(model_state_val, 'Unknown')
        }

        if last_training:
            attrs["training_completed"] = True
            attrs["time_since_training"] = format_time_ago(last_training)
        else:
            attrs["training_completed"] = False
            attrs["time_since_training"] = "Never"
            attrs["note"] = "No training performed yet or ML service recently restarted"

        return attrs


class NextScheduledUpdateSensor(BaseSolarSensor):
    """Sensor showing the time of the next scheduled coordinator update/verification."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the next scheduled update sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_next_scheduled_task" # Changed ID slightly
        self._attr_name = "Next Scheduled Task"
        self._attr_icon = "mdi:calendar-clock"

    # --- HIER DIE KORREKTUR ---
    @property
    def available(self) -> bool:
        """This sensor is always available as it calculates based on time."""
        return True
    # --- ENDE KORREKTUR ---

    @property
    def native_value(self) -> str:
        """Return the time string for the next scheduled task."""
        now_local = dt_util.as_local(dt_util.utcnow()) # Use local time for comparison

        # Define scheduled times in local timezone (using constants)
        morning_update_time = now_local.replace(hour=DAILY_UPDATE_HOUR, minute=0, second=5, microsecond=0) # Add seconds offset
        evening_verify_time = now_local.replace(hour=DAILY_VERIFICATION_HOUR, minute=0, second=10, microsecond=0) # Add seconds offset

        next_tasks = []
        # Check if morning update is still pending today
        if now_local < morning_update_time:
            next_tasks.append(("Morning Update", morning_update_time))
        # Check if evening verification is still pending today
        if now_local < evening_verify_time:
             next_tasks.append(("Evening Verify", evening_verify_time))

        # If both tasks are past for today, the next task is tomorrow morning
        if not next_tasks:
             next_morning = morning_update_time + timedelta(days=1)
             return f"Tomorrow {next_morning.strftime('%H:%M')} (Morning Update)"

        # Find the soonest task for today
        next_tasks.sort(key=lambda x: x[1]) # Sort by time
        task_name, task_time = next_tasks[0] # Get the first one

        return f"Today {task_time.strftime('%H:%M')} ({task_name})"


    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide the configured schedule times."""
        return {
            "daily_update_time": f"{DAILY_UPDATE_HOUR:02d}:00:05", # Show exact trigger time
            "daily_verification_time": f"{DAILY_VERIFICATION_HOUR:02d}:00:10", # Show exact trigger time
        }


class MLServiceStatusSensor(BaseSolarSensor):
    """Sensor reflecting the operational state of the ML Predictor service."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the ML service status sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_service_status"
        self._attr_name = "ML Service Status"
        # Icon updated dynamically

    @property
    def native_value(self) -> str:
        """Return a human-readable status of the ML service."""
        ml_predictor = getattr(getattr(self.coordinator, 'service_manager', {}), 'ml_predictor', None)
        if not ml_predictor:
            dep_ok = getattr(self.coordinator, 'dependencies_ok', False)
            return "Unavailable (Dependencies Missing)" if not dep_ok else "Unavailable (Init Failed)"

        model_state_obj = getattr(ml_predictor, 'model_state', ModelState.UNINITIALIZED)
        model_state_val = model_state_obj.value if isinstance(model_state_obj, ModelState) else str(model_state_obj)
        return ML_STATE_TRANSLATIONS.get(model_state_val, 'Unknown')

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed ML service state."""
        service_manager = getattr(self.coordinator, 'service_manager', None)
        if not service_manager:
             return {"status": "unavailable", "reason": "Service Manager not found"}

        ml_predictor = service_manager.ml_predictor
        if not ml_predictor:
            dep_ok = getattr(self.coordinator, 'dependencies_ok', False)
            reason = "Dependencies missing" if not dep_ok else "Initialization failed"
            return {"status": "unavailable", "reason": reason}

        model_state_obj = getattr(ml_predictor, 'model_state', ModelState.UNINITIALIZED)
        model_state_val = model_state_obj.value if isinstance(model_state_obj, ModelState) else str(model_state_obj)

        return {
            "status": "available" if model_state_val != "unavailable" else "unavailable",
            "model_state": model_state_val,
            "training_samples": getattr(ml_predictor, 'training_samples', 0),
            "model_loaded": getattr(ml_predictor, 'model_loaded', False),
            "current_accuracy": round(getattr(ml_predictor, 'current_accuracy', 0.0), 4) if getattr(ml_predictor, 'current_accuracy', None) is not None else None,
            "last_training_iso": getattr(ml_predictor, 'last_training_time', None).isoformat() if getattr(ml_predictor, 'last_training_time', None) else None
        }

    @property
    def icon(self) -> str:
        """Dynamically change icon based on ML status."""
        state_val = self.native_value # Get the translated state string
        if state_val == ML_STATE_TRANSLATIONS[ModelState.READY.value]: return "mdi:robot-happy-outline"
        elif state_val == ML_STATE_TRANSLATIONS[ModelState.TRAINING.value]: return "mdi:robot-confused-outline"
        elif state_val == ML_STATE_TRANSLATIONS[ModelState.ERROR.value]: return "mdi:robot-dead-outline"
        else: return "mdi:robot-off-outline"


class MLMetricsSensor(BaseSolarSensor):
    """Sensor providing key metrics about the ML model's data and features."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the ML metrics sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_metrics"
        self._attr_name = "ML Model Metrics"
        self._attr_icon = "mdi:chart-box-outline"

    @property
    def native_value(self) -> str:
        """Return the number of training samples used and accuracy."""
        ml_predictor = getattr(getattr(self.coordinator, 'service_manager', {}), 'ml_predictor', None)
        if not ml_predictor: return "ML Unavailable"
        samples = getattr(ml_predictor, 'training_samples', 0)
        accuracy = getattr(ml_predictor, 'current_accuracy', None)
        acc_str = f"{accuracy*100:.1f}%" if accuracy is not None else "N/A"
        return f"{samples} Samples | Acc: {acc_str}"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed metrics."""
        service_manager = getattr(self.coordinator, 'service_manager', None)
        if not service_manager or not service_manager.ml_predictor: return {"status": "unavailable"}

        ml_predictor = service_manager.ml_predictor
        feature_engineer = getattr(ml_predictor, 'feature_engineer', None)
        feature_count = len(feature_engineer.feature_names) if feature_engineer else 0
        perf_metrics = getattr(ml_predictor, 'performance_metrics', {})
        current_weights = getattr(ml_predictor, 'current_weights', None)

        return {
            "status": "available",
            "training_samples": getattr(ml_predictor, 'training_samples', 0),
            "features_count": feature_count,
            "current_accuracy": round(getattr(ml_predictor, 'current_accuracy', 0.0), 4) if getattr(ml_predictor, 'current_accuracy', None) is not None else None,
            "model_version": getattr(current_weights, 'model_version', None) if current_weights else None,
            "avg_prediction_time_ms": round(perf_metrics.get('avg_prediction_time_ms', 0.0), 2),
            "prediction_success_rate": round(1.0 - perf_metrics.get('error_rate', 0.0), 3),
            "total_predictions": perf_metrics.get('total_predictions', 0)
        }


class CoordinatorHealthSensor(BaseSolarSensor):
    """Sensor reflecting the health and performance of the DataUpdateCoordinator."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the coordinator health sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_coordinator_health"
        self._attr_name = "Coordinator Health"
        self._attr_icon = "mdi:heart-pulse"

    @property
    def native_value(self) -> str:
        """Return a simple health status string."""
        last_success_time = getattr(self.coordinator, 'last_update_success_time', None)
        last_update_success_flag = getattr(self.coordinator, 'last_update_success', True)

        if not last_update_success_flag and last_success_time is None: return "Failed Initializing"
        elif not last_update_success_flag: return "Update Failed"
        if not last_success_time: return "Initializing"

        age_seconds = (dt_util.utcnow() - last_success_time).total_seconds()
        interval_seconds = self.coordinator.update_interval.total_seconds() if self.coordinator.update_interval else UPDATE_INTERVAL.total_seconds()

        if age_seconds < (interval_seconds * 1.5): return "Healthy"
        elif age_seconds < (interval_seconds * 3): return "Delayed"
        else: return "Stale"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed coordinator performance metrics."""
        last_success_time = getattr(self.coordinator, 'last_update_success_time', None)
        last_attempt_time = getattr(self.coordinator, 'last_update', None) # Base class attribute

        return {
            "last_update_successful": getattr(self.coordinator, 'last_update_success', False),
            "last_success_time_iso": last_success_time.isoformat() if last_success_time else None,
            "last_attempt_time_iso": last_attempt_time.isoformat() if last_attempt_time else None,
            "time_since_last_success": format_time_ago(last_success_time) if last_success_time else "Never",
            "update_interval_seconds": self.coordinator.update_interval.total_seconds() if self.coordinator.update_interval else None,
        }


class DataFilesStatusSensor(BaseSolarSensor):
    """Sensor checking the presence and basic validity of essential data files."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the data files status sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_data_files_status"
        self._attr_name = "Data Files Status"
        self._attr_icon = "mdi:file-check-outline"
        self._data_manager = getattr(coordinator, 'data_manager', None)

    @property
    def native_value(self) -> str:
        """Return a summary status of the data files."""
        if not self._data_manager: return "Unknown (No DataManager)"

        ml_predictor = getattr(getattr(self.coordinator, 'service_manager', {}), 'ml_predictor', None)
        model_loaded = getattr(ml_predictor, 'model_loaded', False) if ml_predictor else False
        samples_count = 0
        if ml_predictor:
             # Versuche, die Samples direkt vom DataManager zu holen, falls der Predictor noch nicht voll initialisiert ist
             samples_count = getattr(ml_predictor, 'training_samples', 0)
             if samples_count == 0 and hasattr(ml_predictor, 'current_profile') and ml_predictor.current_profile:
                 samples_count = getattr(ml_predictor.current_profile, 'samples_count', 0)


        if model_loaded and samples_count > 0: return "OK"
        elif samples_count > 0: return "Initializing (Samples OK)"
        elif model_loaded: return "Warning (Weights OK, No Samples)"
        else: return "Waiting for data / Not Trained"


    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide status based on loaded model state rather than file checks."""
        if not self._data_manager: return {"status": "unknown", "reason": "DataManager unavailable"}

        ml_predictor = getattr(getattr(self.coordinator, 'service_manager', {}), 'ml_predictor', None)
        weights_status = "Not Loaded/Missing"
        profile_status = "Not Loaded/Missing"
        samples_count = 0

        if ml_predictor:
             weights_status = "Loaded" if getattr(ml_predictor, 'model_loaded', False) else "Not Loaded/Init Failed"
             profile = getattr(ml_predictor, 'current_profile', None)
             profile_status = "Loaded" if profile else "Not Loaded/Init Failed"
             samples_count = getattr(ml_predictor, 'training_samples', 0)
             if samples_count == 0 and profile:
                 samples_count = getattr(profile, 'samples_count', 0)

        return {
            "status": self.native_value,
            "learned_weights_status": weights_status,
            "hourly_profile_status": profile_status,
            "hourly_samples_count": samples_count, # Zeigt die Samples aus dem Profil, wenn Trainingssamples 0 sind
            "prediction_history_status": "Managed",
            "model_state_status": "Managed",
            "data_directory": str(getattr(self._data_manager, 'data_dir', 'Unknown'))
        }

# --- NEU (Block 4): SunGuard Sensor ---
class SunGuardWindowSensor(BaseSolarSensor):
    """Sensor showing the calculated solar production window (local time)."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the sun guard window sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_sun_guard_window"
        self._attr_name = "Sun Guard Production Window"
        self._attr_icon = "mdi:window-open-variant" # Passendes Icon

    @property
    def native_value(self) -> str | None:
        """Return the formatted production window string from the coordinator."""
        # Der Koordinator berechnet diesen Wert bereits in _update_sensor_properties
        return getattr(self.coordinator, 'sun_guard_window', "N/A")

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide the raw status from the coordinator."""
        return {
            "status": getattr(self.coordinator, 'sun_guard_status', "Unknown"),
            "buffer_hours": getattr(self.coordinator.service_manager.sun_guard, 'buffer_hours', None) 
                            if self.coordinator.service_manager and self.coordinator.service_manager.sun_guard else None
        }
# --- ENDE NEU ---