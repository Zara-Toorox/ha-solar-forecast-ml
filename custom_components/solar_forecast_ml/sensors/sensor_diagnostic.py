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
    """Sensor showing the overall diagnostic status of the coordinator by Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the diagnostic status sensor by Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_diagnostic_status" # Changed from _status to avoid clash
        self._attr_translation_key = "diagnostic_status"
        self._attr_icon = "mdi:stethoscope" # More diagnostic icon
        self._attr_name = "Diagnostic Status"

    @property
    def native_value(self) -> str | None:
        """Return the diagnostic status string by Zara"""
        # Value is directly on the coordinator instance
        return getattr(self.coordinator, 'diagnostic_status', None)


class YesterdayDeviationSensor(BaseSolarSensor):
    """Sensor showing the absolute forecast deviation error from the previous day by Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the deviation sensor by Zara"""
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
        """Return the deviation in kWh by Zara"""
        # Get value from coordinator, default to None if not set
        deviation = getattr(self.coordinator, 'last_day_error_kwh', None) # Use the correct attribute name
        return max(0.0, deviation) if deviation is not None else None # Ensure non-negative


class LastCoordinatorUpdateSensor(BaseSolarSensor):
    """Sensor showing the timestamp of the last successful coordinator update by Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the last update sensor by Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_coordinator_update"
        self._attr_translation_key = "last_update_timestamp"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:clock-check-outline" # Changed icon slightly
        self._attr_name = "Last Update"

    @property
    def native_value(self) -> datetime | None: # Use standard datetime hint
        """Return the timestamp of the last successful update by Zara"""
        # Use last_update_success_time for accuracy
        return getattr(self.coordinator, 'last_update_success_time', None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context like time ago by Zara"""
        last_update = getattr(self.coordinator, 'last_update_success_time', None)
        # Use coordinator.last_update from base class for last attempt time
        last_attempt = getattr(self.coordinator, 'last_update', None) # Base coordinator attribute
        return {
            "last_update_iso": last_update.isoformat() if last_update else None,
            "time_ago": format_time_ago(last_update) if last_update else "Never",
            "last_attempt_iso": last_attempt.isoformat() if last_attempt else None, # Use base attribute
        }


class LastMLTrainingSensor(BaseSolarSensor):
    """Sensor showing the timestamp of the last ML model training by Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the last ML training sensor by Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_ml_training"
        self._attr_translation_key = "last_ml_training"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:school-outline" # Training icon
        self._attr_name = "Last ML Training"

    @property
    def native_value(self) -> datetime | None:
        """Return the timestamp of the last ML training by Zara"""
        # FIXED: Direct access to ml_predictor instead of service_manager
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor: return None
        return getattr(ml_predictor, 'last_training_time', None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context like time ago by Zara"""
        # FIXED: Direct access to ml_predictor instead of service_manager
        ml_predictor = self.coordinator.ml_predictor
        last_training = getattr(ml_predictor, 'last_training_time', None) if ml_predictor else None
        return {
            "last_training_iso": last_training.isoformat() if last_training else None,
            "time_ago": format_time_ago(last_training) if last_training else "Never",
        }


class NextScheduledUpdateSensor(BaseSolarSensor):
    """Sensor showing the time of the next scheduled update evening verify by Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the next scheduled update sensor by Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_next_scheduled_update"
        self._attr_translation_key = "next_scheduled_update"
        self._attr_icon = "mdi:calendar-clock"
        self._attr_name = "Next Scheduled Update"
        # No device class for now, it's a formatted string

    @property
    def native_value(self) -> str:
        """Return the time of the next scheduled update in a readable format by Zara"""
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
        """Provide more details about the schedule by Zara"""
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
    """Sensor showing the status of the ML prediction service by Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the ML service status sensor by Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_service_status"
        self._attr_translation_key = "ml_service_status"
        self._attr_icon = "mdi:robot-outline"
        self._attr_name = "ML Service Status"

    @property
    def native_value(self) -> str:
        """Return the ML service status as a translated string by Zara"""
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
        """Provide detailed service status by Zara"""
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
        """Dynamically change icon based on ML status by Zara"""
        state_val = self.native_value # Get the translated state string
        if state_val == ML_STATE_TRANSLATIONS[ModelState.READY.value]: return "mdi:robot-happy-outline"
        elif state_val == ML_STATE_TRANSLATIONS[ModelState.TRAINING.value]: return "mdi:robot-confused-outline"
        elif state_val == ML_STATE_TRANSLATIONS[ModelState.ERROR.value]: return "mdi:robot-dead-outline"
        else: return "mdi:robot-off-outline"


class MLMetricsSensor(BaseSolarSensor):
    """Sensor providing key metrics about the ML models data and features by Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the ML metrics sensor by Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_metrics"
        self._attr_translation_key = "ml_metrics"
        self._attr_icon = "mdi:chart-box-outline"
        self._attr_name = "ML Metrics"

    @property
    def native_value(self) -> str:
        """Return the number of training samples used and accuracy by Zara"""
        # FIXED: Direct access to ml_predictor instead of service_manager
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor: return "ML Unavailable"
        samples = getattr(ml_predictor, 'training_samples', 0)
        accuracy = getattr(ml_predictor, 'current_accuracy', None)
        acc_str = f"{accuracy*100:.1f}%" if accuracy is not None else "N/A"
        return f"{samples} Samples | Acc: {acc_str}"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed metrics by Zara"""
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
    """Sensor reflecting the health and performance of the DataUpdateCoordinator by Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the coordinator health sensor by Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_coordinator_health"
        self._attr_translation_key = "coordinator_health"
        self._attr_icon = "mdi:heart-pulse"
        self._attr_name = "Coordinator Health"

    @property
    def native_value(self) -> str:
        """Return a simple health status string by Zara"""
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
        """Provide detailed coordinator performance metrics by Zara"""
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
    """Sensor showing count of available data files by Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the data files status sensor by Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_data_files_status"
        self._attr_translation_key = "data_files_status"
        self._attr_icon = "mdi:file-multiple-outline"
        self._attr_name = "Data Files Status"
        self._data_manager = getattr(coordinator, 'data_manager', None)

    def _check_file_exists(self, file_path) -> bool:
        """Check if a file exists by Zara"""
        try:
            from pathlib import Path
            return Path(file_path).exists()
        except Exception:
            return False

    @property
    def native_value(self) -> str:
        """Return count of available vs required files by Zara"""
        if not self._data_manager:
            return "0/0"

        from ..const import (
            LEARNED_WEIGHTS_FILE, HOURLY_PROFILE_FILE, HOURLY_SAMPLES_FILE,
            MODEL_STATE_FILE, DAILY_FORECASTS_FILE, PREDICTION_HISTORY_FILE
        )

        required_files = [
            LEARNED_WEIGHTS_FILE,
            HOURLY_PROFILE_FILE,
            HOURLY_SAMPLES_FILE,
            MODEL_STATE_FILE,
            DAILY_FORECASTS_FILE,
            PREDICTION_HISTORY_FILE
        ]

        available_count = 0
        ml_dir = self._data_manager.data_dir / "ml"
        stats_dir = self._data_manager.data_dir / "stats"

        for filename in required_files:
            if filename in [LEARNED_WEIGHTS_FILE, HOURLY_PROFILE_FILE, HOURLY_SAMPLES_FILE, MODEL_STATE_FILE]:
                file_path = ml_dir / filename
            else:
                file_path = stats_dir / filename

            if self._check_file_exists(file_path):
                available_count += 1

        total = len(required_files)
        return f"{available_count}/{total}"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return detailed file status by Zara"""
        if not self._data_manager:
            return {"status": "unavailable"}

        from ..const import (
            LEARNED_WEIGHTS_FILE, HOURLY_PROFILE_FILE, HOURLY_SAMPLES_FILE,
            MODEL_STATE_FILE, DAILY_FORECASTS_FILE, PREDICTION_HISTORY_FILE
        )

        ml_dir = self._data_manager.data_dir / "ml"
        stats_dir = self._data_manager.data_dir / "stats"

        files_status = {
            "learned_weights": self._check_file_exists(ml_dir / LEARNED_WEIGHTS_FILE),
            "hourly_profile": self._check_file_exists(ml_dir / HOURLY_PROFILE_FILE),
            "hourly_samples": self._check_file_exists(ml_dir / HOURLY_SAMPLES_FILE),
            "model_state": self._check_file_exists(ml_dir / MODEL_STATE_FILE),
            "daily_forecasts": self._check_file_exists(stats_dir / DAILY_FORECASTS_FILE),
            "prediction_history": self._check_file_exists(stats_dir / PREDICTION_HISTORY_FILE)
        }

        return {
            "files": files_status,
            "total_available": sum(1 for exists in files_status.values() if exists),
            "total_required": len(files_status),
            "data_directory": str(self._data_manager.data_dir)
        }


# --- Cloudiness Trend Sensors (IMPROVEMENT 7) ---

class CloudinessTrend1hSensor(BaseSolarSensor):
    """Sensor showing cloudiness change in the last 1 hour by Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the cloudiness trend 1h sensor by Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_cloudiness_trend_1h"
        self._attr_translation_key = "cloudiness_trend_1h"
        self._attr_native_unit_of_measurement = "%"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:trending-up"
        self._attr_name = "Cloudiness Trend 1h"

    @property
    def native_value(self) -> float | None:
        """Return the 1-hour cloudiness trend by Zara"""
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return None

        # Calculate trends on demand
        try:
            trends = ml_predictor._calculate_cloudiness_trends()
            return round(trends.get('cloudiness_trend_1h', 0.0), 1)
        except Exception as e:
            _LOGGER.debug(f"Failed to get cloudiness_trend_1h: {e}")
            return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context by Zara"""
        value = self.native_value
        if value is None:
            return {"status": "unavailable"}

        # Interpret trend
        if value > 10:
            interpretation = "Increasingly cloudy"
        elif value > 5:
            interpretation = "Slightly increasing"
        elif value < -10:
            interpretation = "Increasingly sunny"
        elif value < -5:
            interpretation = "Slightly clearing"
        else:
            interpretation = "Stabil"

        return {
            "interpretation": interpretation,
            "description": "Cloud change in last hour (positive = more clouds)"
        }


class CloudinessTrend3hSensor(BaseSolarSensor):
    """Sensor showing cloudiness change in the last 3 hours by Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the cloudiness trend 3h sensor by Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_cloudiness_trend_3h"
        self._attr_translation_key = "cloudiness_trend_3h"
        self._attr_native_unit_of_measurement = "%"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:chart-line"
        self._attr_name = "Cloudiness Trend 3h"

    @property
    def native_value(self) -> float | None:
        """Return the 3-hour cloudiness trend by Zara"""
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return None

        try:
            trends = ml_predictor._calculate_cloudiness_trends()
            return round(trends.get('cloudiness_trend_3h', 0.0), 1)
        except Exception as e:
            _LOGGER.debug(f"Failed to get cloudiness_trend_3h: {e}")
            return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context by Zara"""
        value = self.native_value
        if value is None:
            return {"status": "unavailable"}

        # Interpret trend
        if value > 20:
            interpretation = "Strongly increasing clouds"
        elif value > 10:
            interpretation = "Increasingly cloudy"
        elif value < -20:
            interpretation = "Strongly clearing"
        elif value < -10:
            interpretation = "Increasingly sunny"
        else:
            interpretation = "Relatively stable"

        return {
            "interpretation": interpretation,
            "description": "Cloud change in last 3 hours"
        }


class CloudinessVolatilitySensor(BaseSolarSensor):
    """Sensor showing cloudiness volatility standard deviation by Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the cloudiness volatility sensor by Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_cloudiness_volatility"
        self._attr_translation_key = "cloudiness_volatility"
        self._attr_native_unit_of_measurement = "%"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:waves"
        self._attr_name = "Cloudiness Volatility"

    @property
    def native_value(self) -> float | None:
        """Return the cloudiness volatility std dev by Zara"""
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return None

        try:
            trends = ml_predictor._calculate_cloudiness_trends()
            return round(trends.get('cloudiness_volatility', 0.0), 1)
        except Exception as e:
            _LOGGER.debug(f"Failed to get cloudiness_volatility: {e}")
            return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context by Zara"""
        value = self.native_value
        if value is None:
            return {"status": "unavailable"}

        # Interpret volatility
        if value > 30:
            interpretation = "Very volatile"
        elif value > 15:
            interpretation = "Volatile"
        elif value < 5:
            interpretation = "Very stable"
        else:
            interpretation = "Stabil"

        return {
            "interpretation": interpretation,
            "description": "Cloud volatility (std dev over 3h)"
        }