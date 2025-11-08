"""
Diagnostic Sensors - MIGRATED VERSION

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
    SensorEntity,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy, PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .sensor_base import BaseSolarSensor
from .sensor_mixins import CoordinatorPropertySensorMixin
from ..coordinator import SolarForecastMLCoordinator
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..ml.ml_external_helpers import format_time_ago
from ..const import UPDATE_INTERVAL, DAILY_UPDATE_HOUR, DAILY_VERIFICATION_HOUR
from ..ml.ml_predictor import ModelState

_LOGGER = logging.getLogger(__name__)

# Translations for ML state enum
ML_STATE_TRANSLATIONS = {
    ModelState.UNINITIALIZED.value: "Not yet trained",
    ModelState.TRAINING.value: "Training in progress",
    ModelState.READY.value: "Ready",
    ModelState.DEGRADED.value: "Degraded",
    ModelState.ERROR.value: "Error",
    "unavailable": "Unavailable",
    "unknown": "Unknown"
}


# =============================================================================
# MIGRATED COORDINATOR-PROPERTY SENSORS (Using Mixin)
# =============================================================================

class DiagnosticStatusSensor(CoordinatorPropertySensorMixin, BaseSolarSensor):
    """Sensor showing the overall diagnostic status of the coordinator by @Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:stethoscope"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize by @Zara"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_diagnostic_status"
        self._attr_translation_key = "diagnostic_status"
        self._attr_name = "Diagnostic Status"

    def get_coordinator_value(self) -> str | None:
        """Get value from coordinator by @Zara"""
        return getattr(self.coordinator, 'diagnostic_status', None)


class YesterdayDeviationSensor(CoordinatorPropertySensorMixin, BaseSolarSensor):
    """Sensor showing the absolute forecast deviation error by @Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_device_class = None
    _attr_icon = "mdi:delta"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize by @Zara"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_yesterday_deviation"
        self._attr_translation_key = "yesterday_deviation"
        self._attr_name = "Yesterday Deviation"

    def get_coordinator_value(self) -> float | None:
        """Get value from coordinator by @Zara"""
        deviation = getattr(self.coordinator, 'last_day_error_kwh', None)
        return max(0.0, deviation) if deviation is not None else None


class CloudinessTrend1hSensor(CoordinatorPropertySensorMixin, BaseSolarSensor):
    """Sensor showing cloudiness change in the last 1 hour by @Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = None  # Text interpretation, no unit
    _attr_state_class = None  # Not a measurement anymore
    _attr_icon = "mdi:weather-partly-cloudy"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize by @Zara"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_cloudiness_trend_1h"
        self._attr_translation_key = "cloudiness_trend_1h"
        self._attr_name = "Cloudiness Trend 1h"

    def get_coordinator_value(self) -> str | None:
        """Get text interpretation from coordinator cache by @Zara"""
        try:
            value = self.coordinator.cloudiness_trend_1h

            if value > 10:
                return "Increasingly cloudy"
            elif value > 5:
                return "Slightly increasing"
            elif value < -10:
                return "Increasingly sunny"
            elif value < -5:
                return "Slightly clearing"
            else:
                return "Stable"
        except Exception as e:
            _LOGGER.debug(f"Failed to get cloudiness_trend_1h: {e}")
            return None

    @property
    def icon(self) -> str:
        """Dynamic icon based on trend by @Zara"""
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
        """Provide numeric details by @Zara"""
        try:
            value = self.coordinator.cloudiness_trend_1h
            return {
                "change_percent": round(value, 1),
                "description": "Cloud change in last hour (positive = more clouds)"
            }
        except Exception:
            return {"status": "unavailable"}


class CloudinessTrend3hSensor(CoordinatorPropertySensorMixin, BaseSolarSensor):
    """Sensor showing cloudiness change in the last 3 hours by @Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = None  # Text interpretation, no unit
    _attr_state_class = None  # Not a measurement anymore
    _attr_icon = "mdi:weather-partly-cloudy"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize by @Zara"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_cloudiness_trend_3h"
        self._attr_translation_key = "cloudiness_trend_3h"
        self._attr_name = "Cloudiness Trend 3h"

    def get_coordinator_value(self) -> str | None:
        """Get text interpretation from coordinator cache by @Zara"""
        try:
            value = self.coordinator.cloudiness_trend_3h

            if value > 20:
                return "Strongly increasing clouds"
            elif value > 10:
                return "Increasingly cloudy"
            elif value < -20:
                return "Strongly clearing"
            elif value < -10:
                return "Increasingly sunny"
            else:
                return "Relatively stable"
        except Exception as e:
            _LOGGER.debug(f"Failed to get cloudiness_trend_3h: {e}")
            return None

    @property
    def icon(self) -> str:
        """Dynamic icon based on trend by @Zara"""
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
        """Provide numeric details by @Zara"""
        try:
            value = self.coordinator.cloudiness_trend_3h
            return {
                "change_percent": round(value, 1),
                "description": "Cloud change in last 3 hours"
            }
        except Exception:
            return {"status": "unavailable"}


class CloudinessVolatilitySensor(CoordinatorPropertySensorMixin, BaseSolarSensor):
    """Sensor showing cloudiness volatility by @Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:waves"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize by @Zara"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_cloudiness_volatility"
        self._attr_translation_key = "cloudiness_volatility"
        self._attr_name = "Cloudiness Volatility"

    def get_coordinator_value(self) -> float | None:
        """Get value from coordinator cache by @Zara"""
        # Cache is updated every coordinator update (every 15 min)
        try:
            return round(self.coordinator.cloudiness_volatility, 1)
        except Exception as e:
            _LOGGER.debug(f"Failed to get cloudiness_volatility: {e}")
            return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context by @Zara"""
        value = self.native_value
        if value is None:
            return {"status": "unavailable"}

        if value > 30:
            interpretation = "Very volatile"
        elif value > 15:
            interpretation = "Volatile"
        elif value < 5:
            interpretation = "Very stable"
        else:
            interpretation = "Stable"

        return {
            "interpretation": interpretation,
            "description": "Cloud volatility (std dev over 3h)"
        }


class NextProductionStartSensor(BaseSolarSensor):
    """Sensor showing when next solar production starts by @Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = None
    _attr_state_class = None
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:weather-sunset-up"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize by @Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_next_production_start"
        self._attr_translation_key = "next_production_start"
        self._attr_name = "Next Production Start"

    @property
    def native_value(self) -> datetime | None:
        """Return next production start time in LOCAL timezone by @Zara"""
        try:
            # Get sun.sun entity
            sun_entity = self.hass.states.get('sun.sun')
            if not sun_entity:
                _LOGGER.debug("sun.sun entity not available")
                return None

            # Get next sunrise (UTC)
            next_rising_str = sun_entity.attributes.get('next_rising')
            if not next_rising_str:
                _LOGGER.debug("next_rising attribute not found in sun.sun")
                return None

            # Parse to datetime (UTC)
            next_rising_utc = dt_util.parse_datetime(next_rising_str)
            if not next_rising_utc:
                _LOGGER.debug(f"Could not parse next_rising: {next_rising_str}")
                return None

            # Convert to LOCAL timezone
            next_rising_local = dt_util.as_local(next_rising_utc)

            # Production starts 60 minutes before sunrise
            production_start = next_rising_local - timedelta(minutes=60)

            # Check if production has already started today
            now_local = dt_util.now()
            if production_start.date() < now_local.date():
                # Production start was in the past (yesterday or earlier)
                # Get tomorrow's sunrise
                # sun.sun only provides next_rising, so we need to add a day
                production_start = production_start + timedelta(days=1)

            return production_start

        except Exception as e:
            _LOGGER.debug(f"Failed to calculate next production start: {e}")
            return None

    @property
    def icon(self) -> str:
        """Dynamic icon based on time until production by @Zara"""
        try:
            start_time = self.native_value
            if not start_time:
                return "mdi:weather-sunset-up"

            now = dt_util.now()
            time_until = start_time - now

            if time_until.total_seconds() < 3600:  # < 1 hour
                return "mdi:weather-sunny-alert"
            elif time_until.total_seconds() < 7200:  # < 2 hours
                return "mdi:weather-sunset-up"
            else:
                return "mdi:sleep"

        except Exception:
            return "mdi:weather-sunset-up"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context by @Zara"""
        try:
            start_time = self.native_value
            if not start_time:
                return {"status": "unavailable"}

            now = dt_util.now()
            time_until = start_time - now

            # Get sun.sun for sunset info
            sun_entity = self.hass.states.get('sun.sun')
            next_setting_str = sun_entity.attributes.get('next_setting') if sun_entity else None

            end_time = None
            duration = None
            if next_setting_str:
                next_setting_utc = dt_util.parse_datetime(next_setting_str)
                if next_setting_utc:
                    next_setting_local = dt_util.as_local(next_setting_utc)
                    # Production ends 60 minutes after sunset
                    end_time = next_setting_local + timedelta(minutes=60)

                    # Calculate duration
                    if end_time and start_time:
                        duration_td = end_time - start_time
                        hours = int(duration_td.total_seconds() // 3600)
                        minutes = int((duration_td.total_seconds() % 3600) // 60)
                        duration = f"{hours}h {minutes}m"

            # Format starts_in countdown
            total_seconds = int(time_until.total_seconds())
            if total_seconds < 0:
                starts_in = "Production active"
            else:
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                starts_in = f"{hours}h {minutes}m"

            # Determine day
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
                "production_window": f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}" if end_time else "Unknown"
            }

        except Exception as e:
            _LOGGER.debug(f"Failed to get extra attributes: {e}")
            return {"status": "error"}


# =============================================================================
# COMPLEX SENSORS (Keep original implementation)
# =============================================================================

class LastCoordinatorUpdateSensor(BaseSolarSensor):
    """Sensor showing the timestamp of the last successful coordinator update by @Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize by @Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_coordinator_update"
        self._attr_translation_key = "last_update_timestamp"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:clock-check-outline"
        self._attr_name = "Last Update"

    @property
    def native_value(self) -> datetime | None:
        """Return the timestamp by @Zara"""
        return getattr(self.coordinator, 'last_update_success_time', None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context by @Zara"""
        last_update = getattr(self.coordinator, 'last_update_success_time', None)
        last_attempt = getattr(self.coordinator, 'last_update', None)
        return {
            "last_update_iso": last_update.isoformat() if last_update else None,
            "time_ago": format_time_ago(last_update) if last_update else "Never",
            "last_attempt_iso": last_attempt.isoformat() if last_attempt else None,
        }


class LastMLTrainingSensor(BaseSolarSensor):
    """Sensor showing the timestamp of the last ML model training by @Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize by @Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_ml_training"
        self._attr_translation_key = "last_ml_training"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:school-outline"
        self._attr_name = "Last ML Training"

    @property
    def native_value(self) -> datetime | None:
        """Return the timestamp by @Zara"""
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return None
        return getattr(ml_predictor, 'last_training_time', None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context by @Zara"""
        ml_predictor = self.coordinator.ml_predictor
        last_training = getattr(ml_predictor, 'last_training_time', None) if ml_predictor else None
        return {
            "last_training_iso": last_training.isoformat() if last_training else None,
            "time_ago": format_time_ago(last_training) if last_training else "Never",
        }


class NextScheduledUpdateSensor(BaseSolarSensor):
    """Sensor showing the time of the next scheduled update by @Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize by @Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_next_scheduled_update"
        self._attr_translation_key = "next_scheduled_update"
        self._attr_icon = "mdi:calendar-clock"
        self._attr_name = "Next Scheduled Update"

    @property
    def native_value(self) -> str:
        """Return the time of next scheduled task by @Zara Actual active tasks: - 00:00 Reset Expected Production - 03:00 Weekly ML Training (Sunday only) - 06:00 Morning Forecast - 06:15/30/45 Forecast Retries - 23:05 Intelligent ML Training Check - 23:30 End of Day Workflow"""
        now = dt_util.now()

        # Define all actual scheduled tasks for today
        tasks = [
            (0, 0, "Reset Expected"),
            (3, 0, "Weekly ML Training" if now.weekday() == 6 else None),  # Sunday only
            (DAILY_UPDATE_HOUR, 0, "Morning Forecast"),
            (DAILY_UPDATE_HOUR, 15, "Forecast Retry #1"),
            (DAILY_UPDATE_HOUR, 30, "Forecast Retry #2"),
            (DAILY_UPDATE_HOUR, 45, "Forecast Retry #3"),
            (23, 5, "ML Training Check"),
            (23, 30, "End of Day"),
        ]

        # Filter out None tasks (e.g., Sunday-only on other days)
        tasks = [(h, m, t) for h, m, t in tasks if t is not None]

        # Find next task today
        for hour, minute, task_name in tasks:
            task_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now < task_time:
                return f"{task_time.strftime('%H:%M')} ({task_name})"

        # No more tasks today, show first task tomorrow
        next_time = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return f"{next_time.strftime('%H:%M')} (Reset Expected)"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide more details about scheduled tasks by @Zara"""
        now = dt_util.now()

        # Define all actual scheduled tasks for today
        tasks = [
            (0, 0, "Reset Expected"),
            (3, 0, "Weekly ML Training" if now.weekday() == 6 else None),  # Sunday only
            (DAILY_UPDATE_HOUR, 0, "Morning Forecast"),
            (DAILY_UPDATE_HOUR, 15, "Forecast Retry #1"),
            (DAILY_UPDATE_HOUR, 30, "Forecast Retry #2"),
            (DAILY_UPDATE_HOUR, 45, "Forecast Retry #3"),
            (23, 5, "ML Training Check"),
            (23, 30, "End of Day"),
        ]

        # Filter out None tasks
        tasks = [(h, m, t) for h, m, t in tasks if t is not None]

        # Find next task
        next_time = None
        event_type = None
        for hour, minute, task_name in tasks:
            task_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now < task_time:
                next_time = task_time
                event_type = task_name
                break

        # No more tasks today, show first task tomorrow
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
    """Sensor showing the status of the ML prediction service by @Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize by @Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_service_status"
        self._attr_translation_key = "ml_service_status"
        self._attr_icon = "mdi:robot-outline"
        self._attr_name = "ML Service Status"

    @property
    def native_value(self) -> str:
        """Return the ML service status by @Zara"""
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return ML_STATE_TRANSLATIONS["unavailable"]
        state_enum = getattr(ml_predictor, 'model_state', None)
        if state_enum is None:
            return ML_STATE_TRANSLATIONS["unknown"]
        state_str = state_enum.value if hasattr(state_enum, 'value') else str(state_enum)
        return ML_STATE_TRANSLATIONS.get(state_str, ML_STATE_TRANSLATIONS["unknown"])

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed service status by @Zara"""
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
        """Dynamically change icon by @Zara"""
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
    """Sensor providing key metrics about the ML model by @Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize by @Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_metrics"
        self._attr_translation_key = "ml_metrics"
        self._attr_icon = "mdi:chart-box-outline"
        self._attr_name = "ML Metrics"

    @property
    def native_value(self) -> str:
        """Return the metrics by @Zara"""
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return "ML Unavailable"
        samples = getattr(ml_predictor, 'training_samples', 0)
        accuracy = getattr(ml_predictor, 'current_accuracy', None)
        acc_str = f"{accuracy*100:.1f}%" if accuracy is not None else "N/A"
        return f"{samples} Samples | Acc: {acc_str}"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed metrics by @Zara"""
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return {"status": "unavailable"}

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
    """Sensor reflecting the health of the DataUpdateCoordinator by @Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize by @Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_coordinator_health"
        self._attr_translation_key = "coordinator_health"
        self._attr_icon = "mdi:heart-pulse"
        self._attr_name = "Coordinator Health"

    @property
    def native_value(self) -> str:
        """Return health status by @Zara"""
        last_success_time = getattr(self.coordinator, 'last_update_success_time', None)
        last_update_success_flag = getattr(self.coordinator, 'last_update_success', True)

        if not last_update_success_flag and last_success_time is None:
            return "Failed Initializing"
        elif not last_update_success_flag:
            return "Update Failed"
        if not last_success_time:
            return "Initializing"

        age_seconds = (dt_util.now() - last_success_time).total_seconds()
        interval_seconds = self.coordinator.update_interval.total_seconds() if self.coordinator.update_interval else UPDATE_INTERVAL.total_seconds()

        if age_seconds < (interval_seconds * 1.5):
            return "Healthy"
        elif age_seconds < (interval_seconds * 3):
            return "Delayed"
        else:
            return "Stale"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed metrics by @Zara"""
        last_success_time = getattr(self.coordinator, 'last_update_success_time', None)
        last_attempt_time = getattr(self.coordinator, 'last_update', None)

        return {
            "last_update_successful": getattr(self.coordinator, 'last_update_success', False),
            "last_success_time_iso": last_success_time.isoformat() if last_success_time else None,
            "last_attempt_time_iso": last_attempt_time.isoformat() if last_attempt_time else None,
            "time_since_last_success": format_time_ago(last_success_time) if last_success_time else "Never",
            "update_interval_seconds": self.coordinator.update_interval.total_seconds() if self.coordinator.update_interval else None,
        }


class DataFilesStatusSensor(BaseSolarSensor):
    """Sensor showing count of available data files by @Zara"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize by @Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_data_files_status"
        self._attr_translation_key = "data_files_status"
        self._attr_icon = "mdi:file-multiple-outline"
        self._attr_name = "Data Files Status"
        self._data_manager = getattr(coordinator, 'data_manager', None)

    def _check_file_exists(self, file_path) -> bool:
        """Check if a file exists by @Zara"""
        try:
            from pathlib import Path
            return Path(file_path).exists()
        except Exception:
            return False

    @property
    def native_value(self) -> str:
        """Return count of files by @Zara"""
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
        """Return detailed file status by @Zara"""
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
