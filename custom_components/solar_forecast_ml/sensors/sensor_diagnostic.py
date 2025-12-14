"""Diagnostic Sensors - MIGRATED VERSION V12.0.0 @zara

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
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, UnitOfEnergy
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from ..astronomy.astronomy_cache_manager import get_cache_manager
from ..const import DAILY_UPDATE_HOUR, DAILY_VERIFICATION_HOUR, UPDATE_INTERVAL
from ..coordinator import SolarForecastMLCoordinator
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..ml.ml_external_helpers import format_time_ago
from ..ml.ml_predictor import ModelState
from .sensor_base import BaseSolarSensor
from .sensor_mixins import CoordinatorPropertySensorMixin

_LOGGER = logging.getLogger(__name__)

ML_STATE_TRANSLATIONS = {
    ModelState.UNINITIALIZED.value: "Not yet trained",
    ModelState.TRAINING.value: "Training in progress",
    ModelState.READY.value: "Ready",
    ModelState.DEGRADED.value: "Degraded",
    ModelState.ERROR.value: "Error",
    "unavailable": "Unavailable",
    "unknown": "Unknown",
}

class DiagnosticStatusSensor(CoordinatorPropertySensorMixin, BaseSolarSensor):
    """Sensor showing the overall diagnostic status of the coordinator"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:stethoscope"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_diagnostic_status"
        self._attr_translation_key = "diagnostic_status"
        self._attr_name = "Diagnostic Status"

    def get_coordinator_value(self) -> str | None:
        """Get value from coordinator @zara"""
        return getattr(self.coordinator, "diagnostic_status", None)

class YesterdayDeviationSensor(CoordinatorPropertySensorMixin, BaseSolarSensor):
    """Sensor showing the absolute forecast deviation error"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_device_class = None
    _attr_icon = "mdi:delta"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_yesterday_deviation"
        self._attr_translation_key = "yesterday_deviation"
        self._attr_name = "Yesterday Deviation"

    def get_coordinator_value(self) -> float | None:
        """Get value from coordinator @zara"""
        deviation = getattr(self.coordinator, "last_day_error_kwh", None)
        return max(0.0, deviation) if deviation is not None else None

class CloudinessTrend1hSensor(CoordinatorPropertySensorMixin, BaseSolarSensor):
    """Sensor showing cloudiness change in the last 1 hour"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = None
    _attr_state_class = None
    _attr_icon = "mdi:weather-partly-cloudy"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_cloudiness_trend_1h"
        self._attr_translation_key = "cloudiness_trend_1h"

    def get_coordinator_value(self) -> str | None:
        """Get text interpretation from coordinator cache @zara"""
        try:
            value = self.coordinator.cloudiness_trend_1h

            if value > 10:
                return "getting_cloudier"
            elif value > 5:
                return "slightly_cloudier"
            elif value < -10:
                return "getting_clearer"
            elif value < -5:
                return "slightly_clearer"
            else:
                return "stable"
        except Exception as e:
            _LOGGER.debug(f"Failed to get cloudiness_trend_1h: {e}")
            return None

    @property
    def icon(self) -> str:
        """Dynamic icon based on trend @zara"""
        try:
            value = self.coordinator.cloudiness_trend_1h
            if value > 10:
                return "mdi:weather-cloudy-arrow-right"
            elif value > 5:
                return "mdi:weather-partly-cloudy"
            elif value < -10:
                return "mdi:weather-sunny-alert"
            elif value < -5:
                return "mdi:weather-sunny"
            else:
                return "mdi:minus-circle-outline"
        except Exception:
            return "mdi:weather-partly-cloudy"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide numeric details @zara"""
        try:
            value = self.coordinator.cloudiness_trend_1h
            return {
                "change_percent": round(value, 1),
                "description": "Cloud change in last hour (positive = more clouds)",
            }
        except Exception:
            return {"status": "unavailable"}

class CloudinessTrend3hSensor(CoordinatorPropertySensorMixin, BaseSolarSensor):
    """Sensor showing cloudiness change in the last 3 hours"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = None
    _attr_state_class = None
    _attr_icon = "mdi:weather-partly-cloudy"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_cloudiness_trend_3h"
        self._attr_translation_key = "cloudiness_trend_3h"

    def get_coordinator_value(self) -> str | None:
        """Get text interpretation from coordinator cache @zara"""
        try:
            value = self.coordinator.cloudiness_trend_3h

            if value > 20:
                return "much_cloudier"
            elif value > 10:
                return "getting_cloudier"
            elif value < -20:
                return "much_clearer"
            elif value < -10:
                return "getting_clearer"
            else:
                return "relatively_stable"
        except Exception as e:
            _LOGGER.debug(f"Failed to get cloudiness_trend_3h: {e}")
            return None

    @property
    def icon(self) -> str:
        """Dynamic icon based on trend @zara"""
        try:
            value = self.coordinator.cloudiness_trend_3h
            if value > 20:
                return "mdi:weather-pouring"
            elif value > 10:
                return "mdi:weather-cloudy-arrow-right"
            elif value < -20:
                return "mdi:weather-sunny-alert"
            elif value < -10:
                return "mdi:weather-sunny"
            else:
                return "mdi:minus-circle-outline"
        except Exception:
            return "mdi:weather-partly-cloudy"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide numeric details @zara"""
        try:
            value = self.coordinator.cloudiness_trend_3h
            return {
                "change_percent": round(value, 1),
                "description": "Cloud change in last 3 hours",
            }
        except Exception:
            return {"status": "unavailable"}

class CloudinessVolatilitySensor(CoordinatorPropertySensorMixin, BaseSolarSensor):
    """Sensor showing weather stability index (inverted volatility)"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:waves"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_cloudiness_volatility"
        self._attr_translation_key = "cloudiness_volatility"

    def get_coordinator_value(self) -> float | None:
        """Get stability index from coordinator cache (inverted volatility) @zara"""

        try:
            volatility = self.coordinator.cloudiness_volatility

            stability_index = max(0.0, min(100.0, 100.0 - volatility))
            return round(stability_index, 1)
        except Exception as e:
            _LOGGER.debug(f"Failed to get cloudiness_volatility: {e}")
            return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context @zara"""
        value = self.native_value
        if value is None:
            return {"status": "unavailable"}

        if value > 95:
            interpretation = "very_stable"
        elif value > 85:
            interpretation = "stable"
        elif value > 70:
            interpretation = "moderate"
        elif value > 60:
            interpretation = "variable"
        else:
            interpretation = "very_variable"

        raw_volatility = 100.0 - value

        return {
            "interpretation": interpretation,
            "stability_index": round(value, 1),
            "raw_volatility": round(raw_volatility, 1),
        }

class NextProductionStartSensor(BaseSolarSensor):
    """Sensor showing when next solar production starts"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = None
    _attr_state_class = None
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:weather-sunset-up"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_next_production_start"
        self._attr_translation_key = "next_production_start"
        self._attr_name = "Next Production Start"

    @property
    def native_value(self) -> datetime | None:
        """Return next production start time in LOCAL timezone from in-memory astronomy cache @zara"""
        try:
            now_local = dt_util.now()
            today = now_local.date()

            cache_manager = get_cache_manager()
            if not cache_manager.is_loaded():
                _LOGGER.debug("Astronomy cache not loaded - cannot calculate next production start (normal on fresh install)")
                return None

            date_str = today.isoformat()
            day_data = cache_manager.get_day_data(date_str)

            if day_data:
                window_start_str = day_data.get("production_window_start")
                if window_start_str:
                    window_start = self._parse_datetime_aware(window_start_str, now_local.tzinfo)
                    if window_start and window_start > now_local:
                        return window_start

            tomorrow = today + timedelta(days=1)
            tomorrow_str = tomorrow.isoformat()
            tomorrow_data = cache_manager.get_day_data(tomorrow_str)

            if tomorrow_data:
                window_start_str = tomorrow_data.get("production_window_start")
                if window_start_str:
                    window_start = self._parse_datetime_aware(window_start_str, now_local.tzinfo)
                    if window_start:
                        return window_start

            _LOGGER.debug(f"No production window data available for {today} or {tomorrow} (normal on fresh install)")
            return None

        except Exception as e:
            _LOGGER.debug(f"Failed to calculate next production start: {e} (normal on fresh install)")
            return None

    def _parse_datetime_aware(self, dt_string: str, default_tz) -> Optional[datetime]:
        """Parse datetime string ensuring timezone awareness @zara

        Handles both offset-naive and offset-aware datetime strings.
        If the string has no timezone info, applies the default timezone.
        """
        try:
            if not dt_string:
                return None
            parsed = datetime.fromisoformat(dt_string)
            # If naive (no timezone), make it aware using default timezone
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=default_tz)
            return parsed
        except (ValueError, TypeError) as e:
            _LOGGER.debug(f"Could not parse datetime '{dt_string}': {e}")
            return None

    @property
    def icon(self) -> str:
        """Dynamic icon based on time until production @zara"""
        try:
            start_time = self.native_value
            if not start_time:
                return "mdi:weather-sunset-up"

            now = dt_util.now()
            time_until = start_time - now

            if time_until.total_seconds() < 3600:
                return "mdi:weather-sunny-alert"
            elif time_until.total_seconds() < 7200:
                return "mdi:weather-sunset-up"
            else:
                return "mdi:sleep"

        except Exception:
            return "mdi:weather-sunset-up"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context from astronomy cache @zara"""
        try:
            start_time = self.native_value
            if not start_time:
                return {"status": "unavailable"}

            now = dt_util.now()
            time_until = start_time - now

            end_time = None
            duration = None

            cache_manager = get_cache_manager()
            if cache_manager.is_loaded():

                target_date = start_time.date()
                date_str = target_date.isoformat()
                day_data = cache_manager.get_day_data(date_str)

                if day_data:
                    window_end_str = day_data.get("production_window_end")
                    if window_end_str:
                        # Use timezone-aware parsing @zara
                        end_time = self._parse_datetime_aware(window_end_str, now.tzinfo)

                        if end_time and start_time:
                            duration_td = end_time - start_time
                            hours = int(duration_td.total_seconds() // 3600)
                            minutes = int((duration_td.total_seconds() % 3600) // 60)
                            duration = f"{hours}h {minutes}m"
            else:
                _LOGGER.error("Astronomy cache not loaded - cannot get production end time")

            total_seconds = int(time_until.total_seconds())
            if total_seconds < 0:
                starts_in = "Production active"
            else:
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                starts_in = f"{hours}h {minutes}m"

            if start_time.date() == now.date():
                day = "Heute"
            elif start_time.date() == (now + timedelta(days=1)).date():
                day = "Morgen"
            else:
                day = start_time.strftime("%d.%m.%Y")

            return {
                "start_time": start_time.strftime("%H:%M"),
                "end_time": end_time.strftime("%H:%M") if end_time else "Unknown",
                "duration": duration if duration else "Unknown",
                "starts_in": starts_in,
                "day": day,
                "production_window": (
                    f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}"
                    if end_time
                    else "Unknown"
                ),
            }

        except Exception as e:
            _LOGGER.error(f"Failed to get extra attributes: {e}", exc_info=True)
            return {"status": "error"}

class LastCoordinatorUpdateSensor(BaseSolarSensor):
    """Sensor showing the timestamp of the last successful coordinator update"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_last_coordinator_update"
        self._attr_translation_key = "last_update_timestamp"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:clock-check-outline"
        self._attr_name = "Last Update"

    @property
    def native_value(self) -> datetime | None:
        """Return the timestamp @zara"""
        return getattr(self.coordinator, "last_update_success_time", None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context @zara"""
        last_update = getattr(self.coordinator, "last_update_success_time", None)
        last_attempt = getattr(self.coordinator, "last_update", None)
        return {
            "last_update_iso": last_update.isoformat() if last_update else None,
            "time_ago": format_time_ago(last_update) if last_update else "Never",
            "last_attempt_iso": last_attempt.isoformat() if last_attempt else None,
        }

class LastMLTrainingSensor(BaseSolarSensor):
    """Sensor showing the timestamp of the last ML model training"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_last_ml_training"
        self._attr_translation_key = "last_ml_training"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:school-outline"
        self._attr_name = "Last ML Training"

    @property
    def native_value(self) -> datetime | None:
        """Return the timestamp @zara"""
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return None
        return getattr(ml_predictor, "last_training_time", None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context @zara"""
        ml_predictor = self.coordinator.ml_predictor
        last_training = getattr(ml_predictor, "last_training_time", None) if ml_predictor else None
        return {
            "last_training_iso": last_training.isoformat() if last_training else None,
            "time_ago": format_time_ago(last_training) if last_training else "Never",
        }

class NextScheduledUpdateSensor(BaseSolarSensor):
    """Sensor showing the time of the next scheduled update"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_next_scheduled_update"
        self._attr_translation_key = "next_scheduled_update"
        self._attr_icon = "mdi:calendar-clock"
        self._attr_name = "Next Scheduled Update"

    @property
    def native_value(self) -> str:
        """Return the time of next scheduled task Actual active tasks: - 00:00 Reset Expected Production - 03:00 Weekly ML Training (Sunday only) - 06:00 Morning Forecast - 06:15/30/45 Forecast Retries - 23:05 Intelligent ML Training Check - 23:30 End of Day Workflow @zara"""
        now = dt_util.now()

        tasks = [
            (0, 0, "Reset Expected"),
            (3, 0, "Weekly ML Training" if now.weekday() == 6 else None),
            (DAILY_UPDATE_HOUR, 0, "Morning Forecast"),
            (DAILY_UPDATE_HOUR, 15, "Forecast Retry #1"),
            (DAILY_UPDATE_HOUR, 30, "Forecast Retry #2"),
            (DAILY_UPDATE_HOUR, 45, "Forecast Retry #3"),
            (23, 5, "ML Training Check"),
            (23, 30, "End of Day"),
        ]

        tasks = [(h, m, t) for h, m, t in tasks if t is not None]

        for hour, minute, task_name in tasks:
            task_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now < task_time:
                return f"{task_time.strftime('%H:%M')} ({task_name})"

        next_time = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return f"{next_time.strftime('%H:%M')} (Reset Expected)"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide more details about scheduled tasks @zara"""
        now = dt_util.now()

        tasks = [
            (0, 0, "Reset Expected"),
            (3, 0, "Weekly ML Training" if now.weekday() == 6 else None),
            (DAILY_UPDATE_HOUR, 0, "Morning Forecast"),
            (DAILY_UPDATE_HOUR, 15, "Forecast Retry #1"),
            (DAILY_UPDATE_HOUR, 30, "Forecast Retry #2"),
            (DAILY_UPDATE_HOUR, 45, "Forecast Retry #3"),
            (23, 5, "ML Training Check"),
            (23, 30, "End of Day"),
        ]

        tasks = [(h, m, t) for h, m, t in tasks if t is not None]

        next_time = None
        event_type = None
        for hour, minute, task_name in tasks:
            task_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now < task_time:
                next_time = task_time
                event_type = task_name
                break

        if next_time is None:
            next_time = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            event_type = "Reset Expected"

        return {
            "next_update_time_iso": next_time.isoformat(),
            "event_type": event_type,
            "is_sunday": now.weekday() == 6,
            "morning_forecast_time": f"{DAILY_UPDATE_HOUR}:00",
            "end_of_day_time": "23:30",
            "ml_training_check_time": "23:05",
        }

class MLServiceStatusSensor(BaseSolarSensor):
    """Sensor showing the status of the ML prediction service"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_ml_service_status"
        self._attr_translation_key = "ml_service_status"
        self._attr_icon = "mdi:robot-outline"
        self._attr_name = "ML Service Status"

    @property
    def native_value(self) -> str:
        """Return the ML service status @zara"""
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return ML_STATE_TRANSLATIONS["unavailable"]
        state_enum = getattr(ml_predictor, "model_state", None)
        if state_enum is None:
            return ML_STATE_TRANSLATIONS["unknown"]
        state_str = state_enum.value if hasattr(state_enum, "value") else str(state_enum)
        return ML_STATE_TRANSLATIONS.get(state_str, ML_STATE_TRANSLATIONS["unknown"])

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed service status @zara"""
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return {"status": "unavailable"}

        state_enum = getattr(ml_predictor, "model_state", None)
        state_str = state_enum.value if hasattr(state_enum, "value") and state_enum else "unknown"
        model_loaded = getattr(ml_predictor, "model_loaded", False)
        can_predict = getattr(ml_predictor, "can_predict", False)

        return {
            "model_state": state_str,
            "model_loaded": model_loaded,
            "can_predict": can_predict,
            "training_samples": getattr(ml_predictor, "training_samples", 0),
            "last_training_iso": (
                getattr(ml_predictor, "last_training_time", None).isoformat()
                if getattr(ml_predictor, "last_training_time", None)
                else None
            ),
        }

    @property
    def icon(self) -> str:
        """Dynamically change icon @zara"""
        state_val = self.native_value
        if state_val == ML_STATE_TRANSLATIONS[ModelState.READY.value]:
            return "mdi:robot-happy-outline"
        elif state_val == ML_STATE_TRANSLATIONS[ModelState.TRAINING.value]:
            return "mdi:robot-confused-outline"
        elif state_val == ML_STATE_TRANSLATIONS[ModelState.ERROR.value]:
            return "mdi:robot-dead-outline"
        else:
            return "mdi:robot-off-outline"

class MLMetricsSensor(BaseSolarSensor):
    """Sensor providing key metrics about the ML model"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_ml_metrics"
        self._attr_translation_key = "ml_metrics"
        self._attr_icon = "mdi:chart-box-outline"
        self._attr_name = "ML Metrics"

    @property
    def native_value(self) -> str:
        """Return the metrics @zara"""
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return "ML Unavailable"
        samples = getattr(ml_predictor, "training_samples", 0)
        accuracy = getattr(ml_predictor, "current_accuracy", None)
        acc_str = f"{accuracy*100:.1f}%" if accuracy is not None else "N/A"
        return f"{samples} Samples | Acc: {acc_str}"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed metrics @zara"""
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return {"status": "unavailable"}

        feature_engineer = getattr(ml_predictor, "feature_engineer", None)
        feature_count = len(feature_engineer.feature_names) if feature_engineer else 0
        perf_metrics = getattr(ml_predictor, "performance_metrics", {})
        current_weights = getattr(ml_predictor, "current_weights", None)

        return {
            "status": "available",
            "training_samples": getattr(ml_predictor, "training_samples", 0),
            "features_count": feature_count,
            "current_accuracy": (
                round(getattr(ml_predictor, "current_accuracy", 0.0), 4)
                if getattr(ml_predictor, "current_accuracy", None) is not None
                else None
            ),
            "model_version": (
                getattr(current_weights, "model_version", None) if current_weights else None
            ),
            "avg_prediction_time_ms": round(perf_metrics.get("avg_prediction_time_ms", 0.0), 2),
            "prediction_success_rate": round(1.0 - perf_metrics.get("error_rate", 0.0), 3),
            "total_predictions": perf_metrics.get("total_predictions", 0),
        }

class ActivePredictionModelSensor(BaseSolarSensor):
    """Sensor showing which prediction model/strategy is currently active"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_active_prediction_model"
        self._attr_translation_key = "active_prediction_model"
        self._attr_icon = "mdi:brain"
        self._attr_name = "Active Prediction Model"

    @property
    def native_value(self) -> str:
        """Return active model/strategy @zara"""
        orchestrator = getattr(self.coordinator, "forecast_orchestrator", None)
        if not orchestrator:
            return "Unknown"

        active_strategy_name = getattr(orchestrator, "active_strategy_name", None)

        # Check ML predictor for algorithm info
        ml_predictor = self.coordinator.ml_predictor
        ml_available = ml_predictor and getattr(ml_predictor, "model_loaded", False)

        # Determine display based on ML availability and active strategy
        if ml_available:
            algorithm = getattr(ml_predictor, "algorithm_used", None)
            if algorithm == "tiny_lstm":
                return "Neural Network (TinyLSTM)"
            elif algorithm == "ridge":
                return "ML (Ridge Regression)"
            else:
                return "ML + Rule-Based Blend"

        # Check which strategies are available
        ml_strategy = getattr(orchestrator, "ml_strategy", None)
        rb_strategy = getattr(orchestrator, "rule_based_strategy", None)

        if ml_strategy and getattr(ml_strategy, "is_available", lambda: False)():
            return "ML Strategy"
        elif rb_strategy and getattr(rb_strategy, "is_available", lambda: False)():
            return "Rule-Based"
        else:
            return "Automatic"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed model information @zara"""
        orchestrator = getattr(self.coordinator, "forecast_orchestrator", None)
        ml_predictor = self.coordinator.ml_predictor

        attrs = {
            "strategy_available": orchestrator is not None,
        }

        if orchestrator:
            attrs["active_strategy_name"] = getattr(orchestrator, "active_strategy_name", "unknown")

            # Check strategy availability
            ml_strategy = getattr(orchestrator, "ml_strategy", None)
            rb_strategy = getattr(orchestrator, "rule_based_strategy", None)

            attrs["ml_strategy_available"] = ml_strategy is not None and getattr(ml_strategy, "is_available", lambda: False)()
            attrs["rule_based_strategy_available"] = rb_strategy is not None and getattr(rb_strategy, "is_available", lambda: False)()

        if ml_predictor:
            attrs["ml_available"] = True
            attrs["ml_algorithm"] = getattr(ml_predictor, "algorithm_used", "unknown")
            attrs["ml_model_loaded"] = getattr(ml_predictor, "model_loaded", False)
            attrs["ml_training_samples"] = getattr(ml_predictor, "training_samples", 0)
            accuracy = getattr(ml_predictor, "current_accuracy", 0.0)
            attrs["ml_accuracy"] = round(accuracy, 3) if accuracy is not None else 0.0

            if attrs["ml_algorithm"] == "tiny_lstm":
                attrs["lstm_enabled"] = True
                attrs["lstm_hidden_size"] = 32
                attrs["lstm_lookback_hours"] = 24
        else:
            attrs["ml_available"] = False

        return attrs

class CoordinatorHealthSensor(BaseSolarSensor):
    """Sensor reflecting the health of the DataUpdateCoordinator"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_coordinator_health"
        self._attr_translation_key = "coordinator_health"
        self._attr_icon = "mdi:heart-pulse"
        self._attr_name = "Coordinator Health"

    @property
    def native_value(self) -> str:
        """Return health status @zara"""
        last_success_time = getattr(self.coordinator, "last_update_success_time", None)
        last_update_success_flag = getattr(self.coordinator, "last_update_success", True)

        if not last_update_success_flag and last_success_time is None:
            return "Failed Initializing"
        elif not last_update_success_flag:
            return "Update Failed"
        if not last_success_time:
            return "Initializing"

        age_seconds = (dt_util.now() - last_success_time).total_seconds()
        interval_seconds = (
            self.coordinator.update_interval.total_seconds()
            if self.coordinator.update_interval
            else UPDATE_INTERVAL.total_seconds()
        )

        if age_seconds < (interval_seconds * 1.5):
            return "Healthy"
        elif age_seconds < (interval_seconds * 3):
            return "Delayed"
        else:
            return "Stale"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed metrics @zara"""
        last_success_time = getattr(self.coordinator, "last_update_success_time", None)
        last_attempt_time = getattr(self.coordinator, "last_update", None)

        return {
            "last_update_successful": getattr(self.coordinator, "last_update_success", False),
            "last_success_time_iso": last_success_time.isoformat() if last_success_time else None,
            "last_attempt_time_iso": last_attempt_time.isoformat() if last_attempt_time else None,
            "time_since_last_success": (
                format_time_ago(last_success_time) if last_success_time else "Never"
            ),
            "update_interval_seconds": (
                self.coordinator.update_interval.total_seconds()
                if self.coordinator.update_interval
                else None
            ),
        }

class DataFilesStatusSensor(BaseSolarSensor):
    """Sensor showing count of available data files"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_data_files_status"
        self._attr_translation_key = "data_files_status"
        self._attr_icon = "mdi:file-multiple-outline"
        self._attr_name = "Data Files Status"
        self._data_manager = getattr(coordinator, "data_manager", None)

    def _check_file_exists(self, file_path) -> bool:
        """Check if a file exists @zara"""
        try:
            from pathlib import Path

            return Path(file_path).exists()
        except Exception:
            return False

    @property
    def native_value(self) -> str:
        """Return count of files @zara"""
        if not self._data_manager:
            return "0/0"

        from ..const import (
            DAILY_FORECASTS_FILE,
            HOURLY_PROFILE_FILE,
            LEARNED_WEIGHTS_FILE,
            MODEL_STATE_FILE,
        )

        ml_files_required = [
            LEARNED_WEIGHTS_FILE,
            HOURLY_PROFILE_FILE,
            MODEL_STATE_FILE,
        ]

        stats_files = [
            DAILY_FORECASTS_FILE,
            "hourly_predictions.json",
            "astronomy_cache.json",
            "weather_forecast_corrected.json",
            "weather_precision_daily.json",
            "daily_summaries.json",
        ]

        available_count = 0
        ml_dir = self._data_manager.data_dir / "ml"
        stats_dir = self._data_manager.data_dir / "stats"

        for filename in ml_files_required:
            file_path = ml_dir / filename
            if self._check_file_exists(file_path):
                available_count += 1

        for filename in stats_files:
            file_path = stats_dir / filename
            if self._check_file_exists(file_path):
                available_count += 1

        total_required = len(ml_files_required) + len(stats_files)
        return f"{available_count}/{total_required}"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return detailed file status @zara"""
        if not self._data_manager:
            return {"status": "unavailable"}

        from ..const import (
            DAILY_FORECASTS_FILE,
            HOURLY_PROFILE_FILE,
            LEARNED_WEIGHTS_FILE,
            MODEL_STATE_FILE,
        )

        ml_dir = self._data_manager.data_dir / "ml"
        stats_dir = self._data_manager.data_dir / "stats"

        files_required = {

            "learned_weights": self._check_file_exists(ml_dir / LEARNED_WEIGHTS_FILE),
            "hourly_profile": self._check_file_exists(ml_dir / HOURLY_PROFILE_FILE),
            "model_state": self._check_file_exists(ml_dir / MODEL_STATE_FILE),

            "daily_forecasts": self._check_file_exists(stats_dir / DAILY_FORECASTS_FILE),
            "hourly_predictions": self._check_file_exists(stats_dir / "hourly_predictions.json"),
            "astronomy_cache": self._check_file_exists(stats_dir / "astronomy_cache.json"),
            "weather_forecast_corrected": self._check_file_exists(stats_dir / "weather_forecast_corrected.json"),
            "weather_precision_daily": self._check_file_exists(stats_dir / "weather_precision_daily.json"),
            "daily_summaries": self._check_file_exists(stats_dir / "daily_summaries.json"),
        }

        return {
            "files": files_required,
            "total_available": sum(1 for exists in files_required.values() if exists),
            "total_required": len(files_required),
            "data_directory": str(self._data_manager.data_dir),
        }

class MLTrainingReadinessSensor(BaseSolarSensor):
    """
    Sensor showing ML Training Readiness status

    Replaces confusing "Samples" counter with clear training readiness status.
    Shows users if system is ready for training and provides guidance.
    """

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:progress-check"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_training_readiness"
        self._attr_translation_key = "training_readiness"
        self._attr_name = "Training Readiness"

    def _get_training_ready_count(self) -> int:
        """Count samples that are actually usable for training @zara"""
        try:

            training_ready_count = getattr(self.coordinator, "_training_ready_count", None)

            if training_ready_count is not None:
                return training_ready_count

            return 0

        except Exception as e:
            _LOGGER.debug(f"Error getting training-ready samples: {e}")
            return 0

    def _get_status(self, ready: int) -> str:
        """Get status category @zara"""
        if ready < 50:
            return "collecting"
        elif ready < 200:
            return "early"
        elif ready < 500:
            return "ready"
        else:
            return "excellent"

    def _get_status_label(self, ready: int) -> str:
        """Get user-friendly status label @zara"""
        if ready < 50:
            return f"Collecting ({ready}/50)"
        elif ready < 200:
            return f"Early Training ({ready}/200)"
        elif ready < 500:
            return f"Ready ({ready})"
        else:
            return f"Excellent ({ready})"

    def _get_recommendation(self, ready: int) -> str:
        """Get recommendation based on sample count @zara"""
        if ready < 50:
            return "Sammle weiter Daten (mindestens 3-5 Tage)"
        elif ready < 100:
            return "Erstes experimentelles Training möglich"
        elif ready < 200:
            return "Training möglich - mehr Daten = bessere Ergebnisse"
        elif ready < 500:
            return "Optimale Basis für Training!"
        else:
            return "Perfekt! Du kannst jetzt ein sehr gutes Modell trainieren"

    def _get_days_collecting(self) -> int:
        """Estimate days of data collection @zara"""
        try:
            ready = self._get_training_ready_count()
            if ready == 0:
                return 0

            estimated_days = max(1, ready // 12)
            return estimated_days

        except Exception:
            return 0

    @property
    def native_value(self) -> str:
        """Return training readiness status @zara"""
        ready = self._get_training_ready_count()
        return self._get_status_label(ready)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed readiness information @zara"""
        ready = self._get_training_ready_count()

        ml_predictor = self.coordinator.ml_predictor
        total_samples = getattr(ml_predictor, "training_samples", 0) if ml_predictor else 0

        days_collecting = self._get_days_collecting()
        status = self._get_status(ready)

        estimated_days_to_ready = max(0, (200 - ready) / 12) if ready < 200 else 0

        return {
            "training_ready_samples": ready,
            "total_samples": total_samples,
            "minimum_required": 200,
            "recommended": 500,
            "readiness_percent": min(100, int(ready / 200 * 100)),
            "status": status,
            "days_collecting": days_collecting,
            "estimated_days_to_ready": round(estimated_days_to_ready, 1),
            "can_train": ready >= 50,
            "recommendation": self._get_recommendation(ready),
            "status_emoji": {
                "collecting": "🔴",
                "early": "🟡",
                "ready": "🟢",
                "excellent": "⭐",
            }.get(status, "⚪"),
        }

    @property
    def icon(self) -> str:
        """Dynamic icon based on readiness @zara"""
        ready = self._get_training_ready_count()
        status = self._get_status(ready)

        return {
            "collecting": "mdi:progress-clock",
            "early": "mdi:progress-alert",
            "ready": "mdi:progress-check",
            "excellent": "mdi:progress-star",
        }.get(status, "mdi:progress-question")


class PatternCountSensor(BaseSolarSensor):
    """Sensor showing the number of learned pattern days from PatternLearner"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:chart-timeline-variant"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_pattern_count"
        self._attr_translation_key = "pattern_count"
        self._attr_name = "Pattern Count"

    def _get_pattern_data(self) -> Dict[str, Any]:
        """Load pattern data from learned_patterns.json using cached data @zara

        IMPORTANT: This method is called from native_value property which runs in event loop.
        We use cached data that is loaded asynchronously by the coordinator.
        """
        try:
            # Try to get cached data from coordinator first (async-safe)
            if hasattr(self.coordinator, "data") and self.coordinator.data:
                cached_patterns = self.coordinator.data.get("_cached_patterns")
                if cached_patterns is not None:
                    return cached_patterns

            # Fallback: Return empty - data will be loaded on next coordinator update
            # This avoids blocking file I/O in event loop
            return {}
        except Exception as e:
            _LOGGER.debug(f"Error loading pattern data: {e}")
            return {}

    @property
    def native_value(self) -> int:
        """Return total samples from PatternLearner (sun elevation ranges) @zara"""
        try:
            patterns = self._get_pattern_data()
            if not patterns:
                return 0

            # Count actual samples from sun_elevation_ranges
            geometry_factors = patterns.get("geometry_factors", {})
            sun_ranges = geometry_factors.get("sun_elevation_ranges", {})

            total_samples = sum(
                bucket.get("samples", 0)
                for bucket in sun_ranges.values()
                if isinstance(bucket, dict)
            )

            # If samples exist, return that count
            if total_samples > 0:
                return total_samples

            # Fallback to metadata (might be outdated)
            metadata = patterns.get("metadata", {})
            return metadata.get("total_learning_days", 0)
        except Exception as e:
            _LOGGER.debug(f"Error getting pattern count: {e}")
            return 0

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed pattern information @zara"""
        try:
            patterns = self._get_pattern_data()
            if not patterns:
                return {"status": "no_patterns"}

            metadata = patterns.get("metadata", {})
            geometry_factors = patterns.get("geometry_factors", {})
            sun_ranges = geometry_factors.get("sun_elevation_ranges", {})

            # Count learned buckets with samples
            learned_buckets = sum(
                1 for bucket in sun_ranges.values()
                if isinstance(bucket, dict) and bucket.get("samples", 0) > 0
            )
            total_buckets = len([b for b in sun_ranges.values() if isinstance(b, dict)])

            # Count total samples
            total_samples = sum(
                bucket.get("samples", 0)
                for bucket in sun_ranges.values()
                if isinstance(bucket, dict)
            )

            # Get geometry corrections
            geo_corrections = patterns.get("geometry_corrections", {})
            monthly_corrections = geo_corrections.get("monthly", {})

            return {
                "total_samples": total_samples,
                "total_learning_days": metadata.get("total_learning_days", 0),
                "learned_buckets": f"{learned_buckets}/{total_buckets}",
                "monthly_corrections_count": len(monthly_corrections),
                "has_seasonal_data": len(monthly_corrections) >= 3,
                "pattern_version": patterns.get("version", "unknown"),
            }
        except Exception as e:
            _LOGGER.debug(f"Error getting pattern attributes: {e}")
            return {"status": "error", "message": str(e)}


class PhysicsSamplesSensor(BaseSolarSensor):
    """Sensor showing the number of physics/ML samples from ResidualTrainer or GeometryLearner"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:solar-panel"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_physics_samples"
        self._attr_translation_key = "physics_samples"
        self._attr_name = "Physics Samples"

    def _get_physics_data(self) -> Dict[str, Any]:
        """Load physics data from residual_model_state.json or learned_geometry.json using cached data @zara

        IMPORTANT: This method is called from native_value property which runs in event loop.
        We use cached data that is loaded asynchronously by the coordinator.
        """
        try:
            # Try to get cached data from coordinator first (async-safe)
            if hasattr(self.coordinator, "data") and self.coordinator.data:
                cached_physics = self.coordinator.data.get("_cached_physics")
                if cached_physics is not None:
                    return cached_physics

            # Fallback: Return empty - data will be loaded on next coordinator update
            # This avoids blocking file I/O in event loop
            return {}
        except Exception as e:
            _LOGGER.debug(f"Error loading physics data: {e}")
            return {}

    @property
    def native_value(self) -> int:
        """Return sample count from ResidualTrainer or GeometryLearner @zara"""
        try:
            data = self._get_physics_data()
            if not data:
                return 0

            source = data.get("_source", "")

            # ResidualTrainer format
            if source == "residual_model_state":
                residual_stats = data.get("residual_stats", {})
                return residual_stats.get("sample_count", 0)

            # GeometryLearner format
            if source == "learned_geometry":
                estimate = data.get("estimate", {})
                return estimate.get("sample_count", 0)

            return 0
        except Exception as e:
            _LOGGER.debug(f"Error getting physics samples: {e}")
            return 0

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed physics/ML information @zara"""
        try:
            data = self._get_physics_data()
            if not data:
                return {"status": "not_initialized"}

            source = data.get("_source", "unknown")

            # ResidualTrainer format (Physics-First architecture)
            if source == "residual_model_state":
                residual_stats = data.get("residual_stats", {})
                sample_count = residual_stats.get("sample_count", 0)

                # Determine learning status
                if sample_count == 0:
                    status = "collecting"
                elif sample_count < 50:
                    status = "early_learning"
                elif sample_count < 200:
                    status = "learning"
                elif sample_count < 500:
                    status = "good"
                else:
                    status = "excellent"

                return {
                    "source": "ResidualTrainer (Physics-First)",
                    "sample_count": sample_count,
                    "model_type": data.get("model_type", "unknown"),
                    "residual_mean": round(residual_stats.get("mean", 0), 4),
                    "residual_std": round(residual_stats.get("std", 0), 4),
                    "system_capacity_kwp": data.get("system_capacity_kwp", "unknown"),
                    "learning_status": status,
                    "saved_at": data.get("saved_at", "never"),
                }

            # GeometryLearner format
            if source == "learned_geometry":
                estimate = data.get("estimate", {})
                config = data.get("config", {})
                sample_count = estimate.get("sample_count", 0)
                confidence = estimate.get("confidence", 0.0)

                # Determine learning status
                if sample_count == 0:
                    status = "collecting"
                elif sample_count < 20:
                    status = "early_learning"
                elif sample_count < 50:
                    status = "learning"
                elif confidence < 0.5:
                    status = "converging"
                else:
                    status = "ready"

                return {
                    "source": "GeometryLearner",
                    "sample_count": sample_count,
                    "learned_tilt_deg": round(estimate.get("tilt_deg", 30.0), 1),
                    "learned_azimuth_deg": round(estimate.get("azimuth_deg", 180.0), 1),
                    "confidence": round(confidence, 2),
                    "learning_status": status,
                    "last_updated": estimate.get("last_updated", "never"),
                    "system_capacity_kwp": config.get("system_capacity_kwp", "unknown"),
                }

            return {"status": "unknown_source", "source": source}
        except Exception as e:
            _LOGGER.debug(f"Error getting physics attributes: {e}")
            return {"status": "error", "message": str(e)}
