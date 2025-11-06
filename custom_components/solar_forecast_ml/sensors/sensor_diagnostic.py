"""
Diagnostic Sensors

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

# Import BaseSolarSensor from the base sensor module
from .sensor_base import BaseSolarSensor
from ..coordinator import SolarForecastMLCoordinator
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..ml.ml_external_helpers import format_time_ago
from ..const import UPDATE_INTERVAL, DAILY_UPDATE_HOUR, DAILY_VERIFICATION_HOUR
from ..ml.ml_predictor import ModelState # Import Enum for state mapping

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
        self._attr_translation_key = "diagnostic_status"
        self._attr_icon = "mdi:stethoscope" # More diagnostic icon
        self._attr_name = "Diagnostic Status"

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
        self._attr_translation_key = "yesterday_accuracy"
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:target-arrow" # More specific icon
        self._attr_name = "Yesterday Accuracy"

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
        self._attr_translation_key = "yesterday_deviation"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_device_class = None # Avoid conflict with state_class
        self._attr_icon = "mdi:delta"
        self._attr_name = "Yesterday Deviation"

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
        self._attr_translation_key = "last_update_timestamp"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:clock-check-outline" # Changed icon slightly
        self._attr_name = "Last Update"

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
        self._attr_translation_key = "data_age"
        self._attr_icon = "mdi:timer-sand"
        self._attr_name = "Update Age"
        # No unit, state is a formatted string

    @property
    def native_value(self) -> str:
        """Return how long ago the last update was, in a human-readable format."""
        last_update = getattr(self.coordinator, 'last_update_success_time', None)
        if not last_update:
            return "Never"
        return format_time_ago(last_update)


class LastMLTrainingSensor(BaseSolarSensor):
    """Sensor showing the timestamp of the last ML model training."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the last ML training sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_ml_training"
        self._attr_translation_key = "last_ml_training"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:school-outline" # Training icon
        self._attr_name = "Last ML Training"

    @property
    def native_value(self) -> datetime | None:
        """Return the timestamp of the last ML training."""
        # FIXED: Direct access to ml_predictor instead of service_manager
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor: return None
        return getattr(ml_predictor, 'last_training_time', None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context like time ago."""
        # FIXED: Direct access to ml_predictor instead of service_manager
        ml_predictor = self.coordinator.ml_predictor
        last_training = getattr(ml_predictor, 'last_training_time', None) if ml_predictor else None
        return {
            "last_training_iso": last_training.isoformat() if last_training else None,
            "time_ago": format_time_ago(last_training) if last_training else "Never",
        }


class NextScheduledUpdateSensor(BaseSolarSensor):
    """Sensor showing the time of the next scheduled update (evening verify)."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the next scheduled update sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_next_scheduled_update"
        self._attr_translation_key = "next_scheduled_update"
        self._attr_icon = "mdi:calendar-clock"
        self._attr_name = "Next Scheduled Update"
        # No device class for now, it's a formatted string

    @property
    def native_value(self) -> str:
        """Return the time of the next scheduled update in a readable format."""
        now = dt_util.now()
        today_update = dt_util.now().replace(hour=DAILY_UPDATE_HOUR, minute=0, second=0, microsecond=0)
        evening_verify = dt_util.now().replace(hour=DAILY_VERIFICATION_HOUR, minute=0, second=0, microsecond=0)

        # If before today's update, next update is today_update
        if now < today_update:
            next_time = today_update
            event_type = "Morning Forecast"
        # If before today's verification, next update is evening_verify
        elif now < evening_verify:
            next_time = evening_verify
            event_type = "Evening Verify"
        # Otherwise, next update is tomorrow's morning update
        else:
            next_time = (dt_util.now() + timedelta(days=1)).replace(hour=DAILY_UPDATE_HOUR, minute=0, second=0, microsecond=0)
            event_type = "Morning Forecast"

        return f"{next_time.strftime('%H:%M')} ({event_type})"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide more details about the schedule."""
        now = dt_util.now()
        today_update = dt_util.now().replace(hour=DAILY_UPDATE_HOUR, minute=0, second=0, microsecond=0)
        evening_verify = dt_util.now().replace(hour=DAILY_VERIFICATION_HOUR, minute=0, second=0, microsecond=0)

        if now < today_update:
            next_time = today_update
            event_type = "Morning Forecast"
        elif now < evening_verify:
            next_time = evening_verify
            event_type = "Evening Verify"
        else:
            next_time = (dt_util.now() + timedelta(days=1)).replace(hour=DAILY_UPDATE_HOUR, minute=0, second=0, microsecond=0)
            event_type = "Morning Forecast"

        return {
            "next_update_time_iso": next_time.isoformat(),
            "event_type": event_type,
            "morning_forecast_time": f"{DAILY_UPDATE_HOUR}:00",
            "evening_verify_time": f"{DAILY_VERIFICATION_HOUR}:00",
        }


class MLServiceStatusSensor(BaseSolarSensor):
    """Sensor showing the status of the ML prediction service."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the ML service status sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_service_status"
        self._attr_translation_key = "ml_service_status"
        self._attr_icon = "mdi:robot-outline"
        self._attr_name = "ML Service Status"

    @property
    def native_value(self) -> str:
        """Return the ML service status as a translated string."""
        # FIXED: Direct access to ml_predictor instead of service_manager
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor: return ML_STATE_TRANSLATIONS["unavailable"]
        state_enum = getattr(ml_predictor, 'model_state', None)
        if state_enum is None:
            return ML_STATE_TRANSLATIONS["unknown"]
        state_str = state_enum.value if hasattr(state_enum, 'value') else str(state_enum)
        return ML_STATE_TRANSLATIONS.get(state_str, ML_STATE_TRANSLATIONS["unknown"])

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed service status."""
        # FIXED: Direct access to ml_predictor instead of service_manager
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return {"status": "unavailable"}

        state_enum = getattr(ml_predictor, 'model_state', None)
        state_str = state_enum.value if hasattr(state_enum, 'value') and state_enum else "unknown"
        model_loaded = getattr(ml_predictor, 'model_loaded', False)
        can_predict = getattr(ml_predictor, 'can_predict', False)

        return {
            "model_state": state_str,
            "model_loaded": model_loaded,
            "can_predict": can_predict,
            "training_samples": getattr(ml_predictor, 'training_samples', 0),
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
        self._attr_translation_key = "ml_metrics"
        self._attr_icon = "mdi:chart-box-outline"
        self._attr_name = "ML Metrics"

    @property
    def native_value(self) -> str:
        """Return the number of training samples used and accuracy."""
        # FIXED: Direct access to ml_predictor instead of service_manager
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor: return "ML Unavailable"
        samples = getattr(ml_predictor, 'training_samples', 0)
        accuracy = getattr(ml_predictor, 'current_accuracy', None)
        acc_str = f"{accuracy*100:.1f}%" if accuracy is not None else "N/A"
        return f"{samples} Samples | Acc: {acc_str}"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed metrics."""
        # FIXED: Direct access to ml_predictor instead of service_manager
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor: return {"status": "unavailable"}

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
        self._attr_translation_key = "coordinator_health"
        self._attr_icon = "mdi:heart-pulse"
        self._attr_name = "Coordinator Health"

    @property
    def native_value(self) -> str:
        """Return a simple health status string."""
        last_success_time = getattr(self.coordinator, 'last_update_success_time', None)
        last_update_success_flag = getattr(self.coordinator, 'last_update_success', True)

        if not last_update_success_flag and last_success_time is None: return "Failed Initializing"
        elif not last_update_success_flag: return "Update Failed"
        if not last_success_time: return "Initializing"

        age_seconds = (dt_util.now() - last_success_time).total_seconds() # LOCAL time - last_success_time is LOCAL
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
        self._attr_translation_key = "data_files_status"
        self._attr_icon = "mdi:file-check-outline"
        self._attr_name = "Data Files Status"
        self._data_manager = getattr(coordinator, 'data_manager', None)

    @property
    def native_value(self) -> str:
        """Return a summary status of the data files."""
        if not self._data_manager: return "Unknown (No DataManager)"

        # FIXED: Direct access to ml_predictor instead of service_manager
        ml_predictor = self.coordinator.ml_predictor
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

        # FIXED: Direct access to ml_predictor instead of service_manager
        ml_predictor = self.coordinator.ml_predictor
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