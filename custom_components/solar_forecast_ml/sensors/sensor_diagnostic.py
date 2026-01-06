# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

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
from ..ai import ModelState, format_time_ago
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
        self._attr_unique_id = f"{entry.entry_id}_ml_last_ai_training"
        self._attr_translation_key = "last_ai_training"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:school-outline"
        self._attr_name = "Last AI Training"

    @property
    def native_value(self) -> datetime | None:
        """Return the timestamp @zara"""
        ai_predictor = self.coordinator.ai_predictor
        if not ai_predictor:
            return None
        return getattr(ai_predictor, "last_training_time", None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context @zara"""
        ai_predictor = self.coordinator.ai_predictor
        last_training = getattr(ai_predictor, "last_training_time", None) if ai_predictor else None
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
        """Return the time of next scheduled task Actual active tasks: - 00:00 Reset Expected Production - 03:00 Weekly AI Training (Sunday only) - 06:00 Morning Forecast - 06:15/30/45 Forecast Retries - 23:05 Intelligent AI Training Check - 23:30 End of Day Workflow @zara"""
        now = dt_util.now()

        tasks = [
            (0, 0, "Reset Expected"),
            (3, 0, "Weekly AI Training" if now.weekday() == 6 else None),
            (DAILY_UPDATE_HOUR, 0, "Morning Forecast"),
            (DAILY_UPDATE_HOUR, 15, "Forecast Retry #1"),
            (DAILY_UPDATE_HOUR, 30, "Forecast Retry #2"),
            (DAILY_UPDATE_HOUR, 45, "Forecast Retry #3"),
            (23, 5, "AI Training Check"),
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
            (3, 0, "Weekly AI Training" if now.weekday() == 6 else None),
            (DAILY_UPDATE_HOUR, 0, "Morning Forecast"),
            (DAILY_UPDATE_HOUR, 15, "Forecast Retry #1"),
            (DAILY_UPDATE_HOUR, 30, "Forecast Retry #2"),
            (DAILY_UPDATE_HOUR, 45, "Forecast Retry #3"),
            (23, 5, "AI Training Check"),
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

class MLMetricsSensor(BaseSolarSensor):
    """Sensor providing key metrics about the AI prediction model"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_ml_metrics"
        self._attr_translation_key = "ml_metrics"
        self._attr_icon = "mdi:chart-box-outline"
        self._attr_name = "AI Metrics"

    @property
    def native_value(self) -> str:
        """Return AI model status with R² score @zara"""
        ai_predictor = self.coordinator.ai_predictor
        if not ai_predictor:
            return "AI Unavailable"

        model_loaded = getattr(ai_predictor, "model_loaded", False)
        if not model_loaded and not ai_predictor.is_ready():
            return "AI Not Trained"

        accuracy = getattr(ai_predictor, "current_accuracy", None)
        if accuracy is not None:
            # R² score interpretation
            if accuracy >= 0.8:
                quality = "Excellent"
            elif accuracy >= 0.6:
                quality = "Good"
            elif accuracy >= 0.4:
                quality = "Fair"
            else:
                quality = "Learning"
            return f"AI {quality} | R²: {accuracy:.2f}"
        else:
            return "AI Ready | R²: N/A"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed AI model metrics @zara"""
        ai_predictor = self.coordinator.ai_predictor
        if not ai_predictor:
            return {"status": "unavailable"}

        # Get AI model details
        accuracy = getattr(ai_predictor, "current_accuracy", None)
        training_samples = getattr(ai_predictor, "training_samples", 0)
        num_groups = getattr(ai_predictor, "num_groups", 1)
        total_capacity = getattr(ai_predictor, "total_capacity", 0.0)
        last_training = getattr(ai_predictor, "last_training_time", None)

        # Calculate confidence level from R²
        confidence = ai_predictor.get_base_ai_confidence() if hasattr(ai_predictor, "get_base_ai_confidence") else 0.0

        # Get feature count
        feature_engineer = getattr(ai_predictor, "feature_engineer", None)
        feature_count = len(feature_engineer.feature_names) if feature_engineer and hasattr(feature_engineer, "feature_names") else 0

        # Model quality assessment
        if accuracy is None:
            quality_assessment = "not_evaluated"
        elif accuracy >= 0.8:
            quality_assessment = "excellent"
        elif accuracy >= 0.6:
            quality_assessment = "good"
        elif accuracy >= 0.4:
            quality_assessment = "fair"
        elif accuracy >= 0.2:
            quality_assessment = "learning"
        else:
            quality_assessment = "poor"

        return {
            "status": "ready" if ai_predictor.is_ready() else "not_ready",
            "r2_score": round(accuracy, 4) if accuracy is not None else None,
            "quality_assessment": quality_assessment,
            "ai_confidence": round(confidence * 100, 1),
            "training_data_points": training_samples,
            "feature_count": feature_count,
            "panel_groups": num_groups,
            "total_capacity_kwp": round(total_capacity, 2),
            "last_training": last_training.isoformat() if last_training else None,
            "last_training_ago": format_time_ago(last_training) if last_training else "Never",
        }


class AIRmseSensor(BaseSolarSensor):
    """Sensor showing AI model RMSE (Root Mean Squared Error) in kWh @zara

    RMSE provides an intuitive measure of prediction accuracy:
    - Shows average error in the same unit as predictions (kWh)
    - Lower is better (0 = perfect predictions)
    - Example: RMSE of 1.5 means predictions are off by ~1.5 kWh on average
    """

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_device_class = None
    _attr_icon = "mdi:target"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_ai_rmse"
        self._attr_translation_key = "ai_rmse"
        self._attr_name = "AI RMSE"

    @property
    def native_value(self) -> float | None:
        """Return RMSE value in kWh @zara"""
        ai_predictor = self.coordinator.ai_predictor
        if not ai_predictor:
            return None

        rmse = getattr(ai_predictor, "current_rmse", None)
        if rmse is not None:
            return round(rmse, 3)
        return None

    @property
    def icon(self) -> str:
        """Dynamic icon based on RMSE quality @zara"""
        rmse = self.native_value
        if rmse is None:
            return "mdi:target"
        elif rmse < 0.5:
            return "mdi:target"  # Excellent
        elif rmse < 1.0:
            return "mdi:target-account"  # Good
        elif rmse < 2.0:
            return "mdi:target-variant"  # Fair
        else:
            return "mdi:bullseye"  # Needs improvement

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide RMSE interpretation and context @zara"""
        ai_predictor = self.coordinator.ai_predictor
        if not ai_predictor:
            return {"status": "unavailable"}

        rmse = getattr(ai_predictor, "current_rmse", None)
        r2 = getattr(ai_predictor, "current_accuracy", None)

        if rmse is None:
            return {
                "status": "not_trained",
                "description": "RMSE wird nach dem ersten Training verfügbar",
            }

        # Quality assessment based on typical solar production
        if rmse < 0.3:
            quality = "excellent"
            description = "Sehr präzise Vorhersagen"
        elif rmse < 0.5:
            quality = "very_good"
            description = "Sehr gute Vorhersagegenauigkeit"
        elif rmse < 1.0:
            quality = "good"
            description = "Gute Vorhersagegenauigkeit"
        elif rmse < 1.5:
            quality = "fair"
            description = "Akzeptable Genauigkeit, Modell lernt weiter"
        elif rmse < 2.5:
            quality = "moderate"
            description = "Moderate Genauigkeit, mehr Trainingsdaten nötig"
        else:
            quality = "learning"
            description = "Modell braucht mehr Zeit zum Lernen"

        return {
            "status": "ready",
            "rmse_kwh": round(rmse, 3),
            "quality": quality,
            "description": description,
            "r2_score": round(r2, 4) if r2 is not None else None,
            "interpretation": f"Durchschnittliche Abweichung: ±{rmse:.2f} kWh pro Stunde",
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
        """Return active model/strategy: AI-Hybrid, AI, Physics, or Automatic @zara"""
        orchestrator = getattr(self.coordinator, "forecast_orchestrator", None)
        ai_predictor = self.coordinator.ai_predictor

        # Check if AI is ready
        ai_ready = ai_predictor is not None and ai_predictor.is_ready()

        # Check if Physics is available
        physics_available = False
        if orchestrator:
            rb_strategy = getattr(orchestrator, "rule_based_strategy", None)
            physics_available = rb_strategy is not None and getattr(rb_strategy, "is_available", lambda: False)()

        # Determine mode
        if ai_ready and physics_available:
            return "AI-Hybrid"
        elif ai_ready:
            return "AI"
        elif physics_available:
            return "Physics"
        else:
            return "Automatic"

    @property
    def icon(self) -> str:
        """Dynamic icon based on active mode @zara"""
        mode = self.native_value
        if mode == "AI-Hybrid":
            return "mdi:brain"
        elif mode == "AI":
            return "mdi:robot"
        elif mode == "Physics":
            return "mdi:atom"
        else:
            return "mdi:auto-fix"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed model information @zara"""
        orchestrator = getattr(self.coordinator, "forecast_orchestrator", None)
        ai_predictor = self.coordinator.ai_predictor

        # Check component availability
        ai_ready = ai_predictor is not None and ai_predictor.is_ready()
        physics_available = False
        if orchestrator:
            rb_strategy = getattr(orchestrator, "rule_based_strategy", None)
            physics_available = rb_strategy is not None and getattr(rb_strategy, "is_available", lambda: False)()

        attrs = {
            "mode": self.native_value,
            "ai_available": ai_ready,
            "physics_available": physics_available,
        }

        # Mode description
        mode_descriptions = {
            "AI-Hybrid": "AI predictions validated and enhanced by physics model",
            "AI": "Pure AI-based predictions",
            "Physics": "Physics-based model (AI not yet trained)",
            "Automatic": "Automatic fallback mode",
        }
        attrs["mode_description"] = mode_descriptions.get(self.native_value, "Unknown")

        if ai_predictor:
            attrs["ai_ready"] = ai_ready
            attrs["ai_training_samples"] = getattr(ai_predictor, "training_samples", 0)
            accuracy = getattr(ai_predictor, "current_accuracy", None)
            attrs["ai_r2_score"] = round(accuracy, 3) if accuracy is not None else None
            attrs["ai_confidence"] = round(ai_predictor.get_base_ai_confidence() * 100, 1) if hasattr(ai_predictor, "get_base_ai_confidence") else 0.0

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
    """Sensor showing count of available data files (25 total in production)"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    # Production file structure - 25 files total
    EXPECTED_FILES = {
        "ai": [
            "dni_tracker.json",
            "learned_weights.json",
            "seasonal.json",
        ],
        "data": [
            "coordinator_state.json",
            "open_meteo_cache.json",
            "production_time_state.json",
            "weather_source_weights.json",
            "wttr_in_cache.json",
        ],
        "physics": [
            "calibration_history.json",
            "learning_config.json",
        ],
        "stats": [
            "astronomy_cache.json",
            "daily_forecasts.json",
            "daily_summaries.json",
            "forecast_drift_log.json",
            "hourly_predictions.json",
            "hourly_weather_actual.json",
            "multi_day_hourly_forecast.json",
            "panel_group_sensor_state.json",
            "panel_group_today_cache.json",
            "weather_forecast_corrected.json",
            "weather_precision_daily.json",
            "weather_source_learning.json",
            "yield_cache.json",
        ],
    }

    # Cache validity in seconds - file status doesn't change frequently
    CACHE_TTL_SECONDS = 60

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_data_files_status"
        self._attr_translation_key = "data_files_status"
        self._attr_icon = "mdi:file-multiple-outline"
        self._attr_name = "Data Files Status"
        self._data_manager = getattr(coordinator, "data_manager", None)
        # Cache for file status to avoid repeated I/O operations
        self._cached_status: Dict[str, Dict[str, bool]] = {}
        self._cache_timestamp: float = 0

    def _check_file_exists(self, file_path) -> bool:
        """Check if a file exists @zara"""
        try:
            return file_path.exists()
        except Exception:
            return False

    def _get_file_status(self) -> Dict[str, Dict[str, bool]]:
        """Get status of all expected files by directory. Uses caching to avoid slow I/O."""
        import time

        if not self._data_manager:
            return {}

        # Return cached result if still valid
        current_time = time.monotonic()
        if self._cached_status and (current_time - self._cache_timestamp) < self.CACHE_TTL_SECONDS:
            return self._cached_status

        # Rebuild cache
        status = {}
        for directory, files in self.EXPECTED_FILES.items():
            dir_path = self._data_manager.data_dir / directory
            status[directory] = {}
            for filename in files:
                file_path = dir_path / filename
                status[directory][filename] = self._check_file_exists(file_path)

        self._cached_status = status
        self._cache_timestamp = current_time
        return status

    @property
    def native_value(self) -> str:
        """Return count of files (e.g. '25/25')"""
        if not self._data_manager:
            return "0/25"

        status = self._get_file_status()
        available = sum(
            sum(1 for exists in files.values() if exists)
            for files in status.values()
        )
        total = sum(len(files) for files in self.EXPECTED_FILES.values())

        return f"{available}/{total}"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return detailed file status by directory"""
        if not self._data_manager:
            return {"status": "unavailable"}

        status = self._get_file_status()

        # Flatten for attributes
        files_flat = {}
        for directory, files in status.items():
            for filename, exists in files.items():
                key = f"{directory}/{filename}".replace(".json", "")
                files_flat[key] = exists

        available = sum(1 for exists in files_flat.values() if exists)
        total = len(files_flat)

        return {
            "files": files_flat,
            "by_directory": {
                dir_name: {
                    "available": sum(1 for e in files.values() if e),
                    "total": len(files),
                }
                for dir_name, files in status.items()
            },
            "total_available": available,
            "total_required": total,
            "all_files_present": available == total,
            "data_directory": str(self._data_manager.data_dir),
        }


class PhysicsSamplesSensor(BaseSolarSensor):
    """Sensor showing AI training status @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:brain"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_physics_samples"
        self._attr_translation_key = "physics_samples"
        self._attr_name = "AI Samples"

    def _get_physics_data(self) -> Dict[str, Any]:
        """Load AI weights from ai/learned_weights.json using cached data @zara

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
            _LOGGER.debug(f"Error loading AI data: {e}")
            return {}

    @property
    def native_value(self) -> int:
        """Return AI training sample count @zara"""
        try:
            data = self._get_physics_data()
            if not data:
                return 0

            # TinyLSTM weights format
            return data.get("training_samples", 0)
        except Exception as e:
            _LOGGER.debug(f"Error getting AI samples: {e}")
            return 0

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide AI status information @zara"""
        try:
            data = self._get_physics_data()
            if not data:
                return {"status": "not_initialized"}

            source = data.get("_source", "unknown")
            training_samples = data.get("training_samples", 0)

            # Determine learning status
            if training_samples == 0:
                status = "untrained"
            elif training_samples < 50:
                status = "early_learning"
            elif training_samples < 200:
                status = "learning"
            elif training_samples < 500:
                status = "good"
            else:
                status = "excellent"

            return {
                "source": "TinyLSTM",
                "training_samples": training_samples,
                "learning_status": status,
                "model_type": "lstm",
                "hidden_size": data.get("hidden_size", 32),
                "input_size": data.get("input_size", 17),
            }
        except Exception as e:
            _LOGGER.debug(f"Error getting AI attributes: {e}")
            return {"status": "error", "message": str(e)}
