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
from ..astronomy.astronomy_cache_manager import get_cache_manager

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
    """Sensor showing the overall diagnostic status of the coordinator"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:stethoscope"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_diagnostic_status"
        self._attr_translation_key = "diagnostic_status"
        self._attr_name = "Diagnostic Status"

    def get_coordinator_value(self) -> str | None:
        """Get value from coordinator"""
        return getattr(self.coordinator, 'diagnostic_status', None)


class YesterdayDeviationSensor(CoordinatorPropertySensorMixin, BaseSolarSensor):
    """Sensor showing the absolute forecast deviation error"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_device_class = None
    _attr_icon = "mdi:delta"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_yesterday_deviation"
        self._attr_translation_key = "yesterday_deviation"
        self._attr_name = "Yesterday Deviation"

    def get_coordinator_value(self) -> float | None:
        """Get value from coordinator"""
        deviation = getattr(self.coordinator, 'last_day_error_kwh', None)
        return max(0.0, deviation) if deviation is not None else None


class CloudinessTrend1hSensor(CoordinatorPropertySensorMixin, BaseSolarSensor):
    """Sensor showing cloudiness change in the last 1 hour"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = None  # Text interpretation, no unit
    _attr_state_class = None  # Not a measurement anymore
    _attr_icon = "mdi:weather-partly-cloudy"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_cloudiness_trend_1h"
        self._attr_translation_key = "cloudiness_trend_1h"

    def get_coordinator_value(self) -> str | None:
        """Get text interpretation from coordinator cache"""
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
        """Dynamic icon based on trend"""
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
        """Provide numeric details"""
        try:
            value = self.coordinator.cloudiness_trend_1h
            return {
                "change_percent": round(value, 1),
                "description": "Cloud change in last hour (positive = more clouds)"
            }
        except Exception:
            return {"status": "unavailable"}


class CloudinessTrend3hSensor(CoordinatorPropertySensorMixin, BaseSolarSensor):
    """Sensor showing cloudiness change in the last 3 hours"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = None  # Text interpretation, no unit
    _attr_state_class = None  # Not a measurement anymore
    _attr_icon = "mdi:weather-partly-cloudy"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_cloudiness_trend_3h"
        self._attr_translation_key = "cloudiness_trend_3h"

    def get_coordinator_value(self) -> str | None:
        """Get text interpretation from coordinator cache"""
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
        """Dynamic icon based on trend"""
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
        """Provide numeric details"""
        try:
            value = self.coordinator.cloudiness_trend_3h
            return {
                "change_percent": round(value, 1),
                "description": "Cloud change in last 3 hours"
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
        """Initialize"""
        BaseSolarSensor.__init__(self, coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_cloudiness_volatility"
        self._attr_translation_key = "cloudiness_volatility"

    def get_coordinator_value(self) -> float | None:
        """Get stability index from coordinator cache (inverted volatility)"""
        # Cache is updated every coordinator update (every 15 min)
        try:
            volatility = self.coordinator.cloudiness_volatility
            # Convert volatility to stability index: 100% = very stable, 0% = very unstable
            # Cap volatility at 100 to ensure stability index doesn't go negative
            stability_index = max(0.0, min(100.0, 100.0 - volatility))
            return round(stability_index, 1)
        except Exception as e:
            _LOGGER.debug(f"Failed to get cloudiness_volatility: {e}")
            return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context"""
        value = self.native_value
        if value is None:
            return {"status": "unavailable"}

        # Interpret stability index (higher = more stable)
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

        # Also provide raw volatility for reference
        raw_volatility = 100.0 - value

        return {
            "interpretation": interpretation,
            "stability_index": round(value, 1),
            "raw_volatility": round(raw_volatility, 1)
        }


class NextProductionStartSensor(BaseSolarSensor):
    """Sensor showing when next solar production starts"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = None
    _attr_state_class = None
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:weather-sunset-up"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_next_production_start"
        self._attr_translation_key = "next_production_start"
        self._attr_name = "Next Production Start"

    @property
    def native_value(self) -> datetime | None:
        """
        Return next production start time in LOCAL timezone from in-memory astronomy cache

        Falls back to sun.sun entity if cache unavailable
        """
        try:
            now_local = dt_util.now()
            today = now_local.date()

            # Try in-memory astronomy cache first (no I/O blocking!)
            cache_manager = get_cache_manager()
            if cache_manager.is_loaded():
                # Check today first
                date_str = today.isoformat()
                day_data = cache_manager.get_day_data(date_str)

                if day_data:
                    window_start_str = day_data.get("production_window_start")
                    if window_start_str:
                        # Parse and keep timezone info (required for TIMESTAMP sensors)
                        window_start = datetime.fromisoformat(window_start_str)

                        # If production start is in the future today, return it
                        if window_start > now_local:
                            return window_start

                # Otherwise get tomorrow's production window
                tomorrow = today + timedelta(days=1)
                tomorrow_str = tomorrow.isoformat()
                tomorrow_data = cache_manager.get_day_data(tomorrow_str)

                if tomorrow_data:
                    window_start_str = tomorrow_data.get("production_window_start")
                    if window_start_str:
                        # Parse and keep timezone info (required for TIMESTAMP sensors)
                        window_start = datetime.fromisoformat(window_start_str)
                        return window_start

            # Fallback to sun.sun entity if cache unavailable
            _LOGGER.debug("Astronomy cache unavailable, falling back to sun.sun entity")
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
        """Dynamic icon based on time until production"""
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
        """
        Provide additional context from astronomy cache

        Falls back to sun.sun entity if cache unavailable
        """
        try:
            start_time = self.native_value
            if not start_time:
                return {"status": "unavailable"}

            now = dt_util.now()
            time_until = start_time - now

            end_time = None
            duration = None

            # Try in-memory astronomy cache first (no I/O blocking!)
            cache_manager = get_cache_manager()
            if cache_manager.is_loaded():
                # Get production end time from cache
                # Check which day start_time is on
                target_date = start_time.date()
                date_str = target_date.isoformat()
                day_data = cache_manager.get_day_data(date_str)

                if day_data:
                    window_end_str = day_data.get("production_window_end")
                    if window_end_str:
                        # Parse and keep timezone info
                        end_time = datetime.fromisoformat(window_end_str)

                        # Calculate duration
                        if end_time and start_time:
                            duration_td = end_time - start_time
                            hours = int(duration_td.total_seconds() // 3600)
                            minutes = int((duration_td.total_seconds() % 3600) // 60)
                            duration = f"{hours}h {minutes}m"

            # Fallback to sun.sun entity if cache unavailable
            if not end_time:
                _LOGGER.debug("Using sun.sun fallback for production end time")
                sun_entity = self.hass.states.get('sun.sun')
                next_setting_str = sun_entity.attributes.get('next_setting') if sun_entity else None

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
    """Sensor showing the timestamp of the last successful coordinator update"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_last_coordinator_update"
        self._attr_translation_key = "last_update_timestamp"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:clock-check-outline"
        self._attr_name = "Last Update"

    @property
    def native_value(self) -> datetime | None:
        """Return the timestamp"""
        return getattr(self.coordinator, 'last_update_success_time', None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context"""
        last_update = getattr(self.coordinator, 'last_update_success_time', None)
        last_attempt = getattr(self.coordinator, 'last_update', None)
        return {
            "last_update_iso": last_update.isoformat() if last_update else None,
            "time_ago": format_time_ago(last_update) if last_update else "Never",
            "last_attempt_iso": last_attempt.isoformat() if last_attempt else None,
        }


class LastMLTrainingSensor(BaseSolarSensor):
    """Sensor showing the timestamp of the last ML model training"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_last_ml_training"
        self._attr_translation_key = "last_ml_training"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:school-outline"
        self._attr_name = "Last ML Training"

    @property
    def native_value(self) -> datetime | None:
        """Return the timestamp"""
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return None
        return getattr(ml_predictor, 'last_training_time', None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context"""
        ml_predictor = self.coordinator.ml_predictor
        last_training = getattr(ml_predictor, 'last_training_time', None) if ml_predictor else None
        return {
            "last_training_iso": last_training.isoformat() if last_training else None,
            "time_ago": format_time_ago(last_training) if last_training else "Never",
        }


class NextScheduledUpdateSensor(BaseSolarSensor):
    """Sensor showing the time of the next scheduled update"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_next_scheduled_update"
        self._attr_translation_key = "next_scheduled_update"
        self._attr_icon = "mdi:calendar-clock"
        self._attr_name = "Next Scheduled Update"

    @property
    def native_value(self) -> str:
        """Return the time of next scheduled task Actual active tasks: - 00:00 Reset Expected Production - 03:00 Weekly ML Training (Sunday only) - 06:00 Morning Forecast - 06:15/30/45 Forecast Retries - 23:05 Intelligent ML Training Check - 23:30 End of Day Workflow"""
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
        """Provide more details about scheduled tasks"""
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
    """Sensor showing the status of the ML prediction service"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_ml_service_status"
        self._attr_translation_key = "ml_service_status"
        self._attr_icon = "mdi:robot-outline"
        self._attr_name = "ML Service Status"

    @property
    def native_value(self) -> str:
        """Return the ML service status"""
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
        """Provide detailed service status"""
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
        """Dynamically change icon"""
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
        """Initialize"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_ml_metrics"
        self._attr_translation_key = "ml_metrics"
        self._attr_icon = "mdi:chart-box-outline"
        self._attr_name = "ML Metrics"

    @property
    def native_value(self) -> str:
        """Return the metrics"""
        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            return "ML Unavailable"
        samples = getattr(ml_predictor, 'training_samples', 0)
        accuracy = getattr(ml_predictor, 'current_accuracy', None)
        acc_str = f"{accuracy*100:.1f}%" if accuracy is not None else "N/A"
        return f"{samples} Samples | Acc: {acc_str}"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed metrics"""
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
    """Sensor reflecting the health of the DataUpdateCoordinator"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_coordinator_health"
        self._attr_translation_key = "coordinator_health"
        self._attr_icon = "mdi:heart-pulse"
        self._attr_name = "Coordinator Health"

    @property
    def native_value(self) -> str:
        """Return health status"""
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
        """Provide detailed metrics"""
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
    """Sensor showing count of available data files"""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_data_files_status"
        self._attr_translation_key = "data_files_status"
        self._attr_icon = "mdi:file-multiple-outline"
        self._attr_name = "Data Files Status"
        self._data_manager = getattr(coordinator, 'data_manager', None)

    def _check_file_exists(self, file_path) -> bool:
        """Check if a file exists"""
        try:
            from pathlib import Path
            return Path(file_path).exists()
        except Exception:
            return False

    @property
    def native_value(self) -> str:
        """Return count of files"""
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
        """Return detailed file status"""
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


class MLTrainingReadinessSensor(BaseSolarSensor):
    """
    Sensor showing ML Training Readiness status

    Replaces confusing "Samples" counter with clear training readiness status.
    Shows users if system is ready for training and provides guidance.
    """
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:progress-check"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_training_readiness"
        self._attr_translation_key = "training_readiness"
        self._attr_name = "Training Readiness"

    def _get_training_ready_count(self) -> int:
        """
        Count samples that are actually usable for V2 training

        V2 Training uses hourly_predictions.json:
        - Must have actual_kwh (measured production)
        - Must have all core features (weather_forecast, sensor_actual, astronomy)
        - Enriched with astronomy_cache.json data

        NOTE: Uses cached value from coordinator to avoid blocking I/O
        Coordinator updates this value during its regular update cycle
        """
        try:
            # V2: Get cached count from coordinator (updated during refresh)
            # This avoids blocking I/O in sensor property getter
            training_ready_count = getattr(self.coordinator, '_v2_training_ready_count', None)

            if training_ready_count is not None:
                return training_ready_count

            # Fallback: Return 0 if not yet initialized
            # Coordinator will populate this value on next update
            return 0

        except Exception as e:
            _LOGGER.debug(f"Error getting V2 training-ready samples: {e}")
            return 0

    def _get_status(self, ready: int) -> str:
        """Get status category"""
        if ready < 50:
            return "collecting"
        elif ready < 200:
            return "early"
        elif ready < 500:
            return "ready"
        else:
            return "excellent"

    def _get_status_label(self, ready: int) -> str:
        """Get user-friendly status label"""
        if ready < 50:
            return f"Collecting ({ready}/50)"
        elif ready < 200:
            return f"Early Training ({ready}/200)"
        elif ready < 500:
            return f"Ready ({ready})"
        else:
            return f"Excellent ({ready})"

    def _get_recommendation(self, ready: int) -> str:
        """Get recommendation based on sample count"""
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
        """
        Estimate days of data collection

        NOTE: Simplified calculation based on sample count
        Assumes ~12 production hours per day average
        """
        try:
            ready = self._get_training_ready_count()
            if ready == 0:
                return 0

            # Rough estimate: 12 production hours per day
            estimated_days = max(1, ready // 12)
            return estimated_days

        except Exception:
            return 0

    @property
    def native_value(self) -> str:
        """Return training readiness status"""
        ready = self._get_training_ready_count()
        return self._get_status_label(ready)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed readiness information"""
        ready = self._get_training_ready_count()

        # Also get total samples for comparison
        ml_predictor = self.coordinator.ml_predictor
        total_samples = getattr(ml_predictor, 'training_samples', 0) if ml_predictor else 0

        days_collecting = self._get_days_collecting()
        status = self._get_status(ready)

        # Estimated days to ready (assuming ~12 production hours per day)
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
                "excellent": "⭐"
            }.get(status, "⚪"),
        }

    @property
    def icon(self) -> str:
        """Dynamic icon based on readiness"""
        ready = self._get_training_ready_count()
        status = self._get_status(ready)

        return {
            "collecting": "mdi:progress-clock",
            "early": "mdi:progress-alert",
            "ready": "mdi:progress-check",
            "excellent": "mdi:progress-star"
        }.get(status, "mdi:progress-question")
