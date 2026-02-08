# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

# *****************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# Refactored: JSON replaced with DatabaseManager @zara
# *****************************************************************************

"""
Diagnostic sensors for Solar Forecast ML.
All sensors read from coordinator cache or DB - no direct file I/O.
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
from homeassistant.helpers.entity import EntityCategory

from ..const import DAILY_UPDATE_HOUR, UPDATE_INTERVAL
from ..coordinator import SolarForecastMLCoordinator
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..ai import ModelState, format_time_ago
from .sensor_base import BaseSolarSensor

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


class YesterdayDeviationSensor(BaseSolarSensor):
    """Sensor showing the absolute forecast deviation error. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_device_class = None
    _attr_icon = "mdi:delta"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_yesterday_deviation"
        self._attr_translation_key = "yesterday_deviation"
        self._attr_name = "Yesterday Deviation"

    @property
    def native_value(self) -> Optional[float]:
        """Get value from coordinator. @zara"""
        deviation = getattr(self.coordinator, "last_day_error_kwh", None)
        return max(0.0, deviation) if deviation is not None else None


class CloudinessTrend1hSensor(BaseSolarSensor):
    """Sensor showing cloudiness change in the last 1 hour. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = None
    _attr_state_class = None
    _attr_icon = "mdi:weather-partly-cloudy"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_cloudiness_trend_1h"
        self._attr_translation_key = "cloudiness_trend_1h"

    @property
    def native_value(self) -> Optional[str]:
        """Get text interpretation from coordinator cache. @zara"""
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
        """Dynamic icon based on trend. @zara"""
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
        """Provide numeric details. @zara"""
        try:
            value = self.coordinator.cloudiness_trend_1h
            return {
                "change_percent": round(value, 1),
                "description": "Cloud change in last hour (positive = more clouds)",
            }
        except Exception:
            return {"status": "unavailable"}


class CloudinessTrend3hSensor(BaseSolarSensor):
    """Sensor showing cloudiness change in the last 3 hours. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = None
    _attr_state_class = None
    _attr_icon = "mdi:weather-partly-cloudy"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_cloudiness_trend_3h"
        self._attr_translation_key = "cloudiness_trend_3h"

    @property
    def native_value(self) -> Optional[str]:
        """Get text interpretation from coordinator cache. @zara"""
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
        """Dynamic icon based on trend. @zara"""
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
        """Provide numeric details. @zara"""
        try:
            value = self.coordinator.cloudiness_trend_3h
            return {
                "change_percent": round(value, 1),
                "description": "Cloud change in last 3 hours",
            }
        except Exception:
            return {"status": "unavailable"}


class CloudinessVolatilitySensor(BaseSolarSensor):
    """Sensor showing weather stability index (inverted volatility). @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:waves"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_cloudiness_volatility"
        self._attr_translation_key = "cloudiness_volatility"

    @property
    def native_value(self) -> Optional[float]:
        """Get stability index from coordinator cache (inverted volatility). @zara"""
        try:
            volatility = self.coordinator.cloudiness_volatility
            stability_index = max(0.0, min(100.0, 100.0 - volatility))
            return round(stability_index, 1)
        except Exception as e:
            _LOGGER.debug(f"Failed to get cloudiness_volatility: {e}")
            return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context. @zara"""
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
    """Sensor showing when next solar production starts. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = None
    _attr_state_class = None
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:weather-sunset-up"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_next_production_start"
        self._attr_translation_key = "next_production_start"
        self._attr_name = "Next Production Start"

    @property
    def native_value(self) -> Optional[datetime]:
        """Return next production start time from astronomy cache. @zara"""
        try:
            from ..astronomy.astronomy_cache_manager import get_cache_manager

            now_local = dt_util.now()
            today = now_local.date()

            cache_manager = get_cache_manager(self.coordinator.data_manager._db_manager)
            if not cache_manager.is_loaded():
                _LOGGER.debug("Astronomy cache not loaded - cannot calculate next production start")
                return None

            day_data = cache_manager.get_day_data(today)

            if day_data:
                window_start_str = day_data.get("production_window_start")
                if window_start_str:
                    window_start = self._parse_datetime_aware(window_start_str, now_local.tzinfo)
                    if window_start and window_start > now_local:
                        return window_start

            # Check tomorrow
            tomorrow = today + timedelta(days=1)
            tomorrow_data = cache_manager.get_day_data(tomorrow)

            if tomorrow_data:
                window_start_str = tomorrow_data.get("production_window_start")
                if window_start_str:
                    window_start = self._parse_datetime_aware(window_start_str, now_local.tzinfo)
                    if window_start:
                        return window_start

            return None

        except Exception as e:
            _LOGGER.debug(f"Failed to calculate next production start: {e}")
            return None

    def _parse_datetime_aware(self, dt_string: str, default_tz) -> Optional[datetime]:
        """Parse datetime string ensuring timezone awareness. @zara"""
        try:
            if not dt_string:
                return None
            parsed = datetime.fromisoformat(dt_string)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=default_tz)
            return parsed
        except (ValueError, TypeError) as e:
            _LOGGER.debug(f"Could not parse datetime '{dt_string}': {e}")
            return None

    @property
    def icon(self) -> str:
        """Dynamic icon based on time until production. @zara"""
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
        """Provide additional context from astronomy cache. @zara"""
        try:
            from ..astronomy.astronomy_cache_manager import get_cache_manager

            start_time = self.native_value
            if not start_time:
                return {"status": "unavailable"}

            now = dt_util.now()
            time_until = start_time - now

            end_time = None
            duration = None

            cache_manager = get_cache_manager(self.coordinator.data_manager._db_manager)
            if cache_manager.is_loaded():
                target_date = start_time.date()
                day_data = cache_manager.get_day_data(target_date)

                if day_data:
                    window_end_str = day_data.get("production_window_end")
                    if window_end_str:
                        end_time = self._parse_datetime_aware(window_end_str, now.tzinfo)
                        if end_time and start_time:
                            duration_td = end_time - start_time
                            hours = int(duration_td.total_seconds() // 3600)
                            minutes = int((duration_td.total_seconds() % 3600) // 60)
                            duration = f"{hours}h {minutes}m"

            total_seconds = int(time_until.total_seconds())
            if total_seconds < 0:
                starts_in = "Production active"
            else:
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                starts_in = f"{hours}h {minutes}m"

            if start_time.date() == now.date():
                day = "Today"
            elif start_time.date() == (now + timedelta(days=1)).date():
                day = "Tomorrow"
            else:
                day = start_time.strftime("%Y-%m-%d")

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
    """Sensor showing the timestamp of the last successful coordinator update. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_last_coordinator_update"
        self._attr_translation_key = "last_update_timestamp"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:clock-check-outline"
        self._attr_name = "Last Update"

    @property
    def native_value(self) -> Optional[datetime]:
        """Return the timestamp. @zara"""
        return getattr(self.coordinator, "last_update_success_time", None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context. @zara"""
        last_update = getattr(self.coordinator, "last_update_success_time", None)
        last_attempt = getattr(self.coordinator, "last_update", None)
        return {
            "last_update_iso": last_update.isoformat() if isinstance(last_update, datetime) else (last_update if isinstance(last_update, str) else None),
            "time_ago": format_time_ago(last_update) if isinstance(last_update, datetime) else "Never",
            "last_attempt_iso": last_attempt.isoformat() if isinstance(last_attempt, datetime) else (last_attempt if isinstance(last_attempt, str) else None),
        }


class LastMLTrainingSensor(BaseSolarSensor):
    """Sensor showing the timestamp of the last ML model training. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_last_ai_training"
        self._attr_translation_key = "last_ai_training"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:school-outline"
        self._attr_name = "Last AI Training"

    @property
    def native_value(self) -> Optional[datetime]:
        """Return the timestamp. @zara"""
        ai_predictor = self.coordinator.ai_predictor
        if not ai_predictor:
            return None
        return getattr(ai_predictor, "last_training_time", None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide additional context. @zara"""
        ai_predictor = self.coordinator.ai_predictor
        last_training = getattr(ai_predictor, "last_training_time", None) if ai_predictor else None
        return {
            "last_training_iso": last_training.isoformat() if isinstance(last_training, datetime) else (last_training if isinstance(last_training, str) else None),
            "time_ago": format_time_ago(last_training) if isinstance(last_training, datetime) else "Never",
        }


class NextScheduledUpdateSensor(BaseSolarSensor):
    """Sensor showing the time of the next scheduled update. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_next_scheduled_update"
        self._attr_translation_key = "next_scheduled_update"
        self._attr_icon = "mdi:calendar-clock"
        self._attr_name = "Next Scheduled Update"

    @property
    def native_value(self) -> str:
        """Return the time of next scheduled task. @zara"""
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
        """Provide more details about scheduled tasks. @zara"""
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
    """Sensor providing key metrics about the AI prediction model. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_ml_metrics"
        self._attr_translation_key = "ml_metrics"
        self._attr_icon = "mdi:chart-box-outline"
        self._attr_name = "AI Metrics"

    @property
    def native_value(self) -> str:
        """Return AI model status with R2 score. @zara"""
        ai_predictor = self.coordinator.ai_predictor
        if not ai_predictor:
            return "AI Unavailable"

        model_loaded = getattr(ai_predictor, "model_loaded", False)
        if not model_loaded and not ai_predictor.is_ready():
            return "AI Not Trained"

        accuracy = getattr(ai_predictor, "current_accuracy", None)
        if accuracy is not None:
            if accuracy >= 0.8:
                quality = "Excellent"
            elif accuracy >= 0.6:
                quality = "Good"
            elif accuracy >= 0.4:
                quality = "Fair"
            else:
                quality = "Learning"
            return f"AI {quality} | R2: {accuracy:.2f}"
        else:
            return "AI Ready | R2: N/A"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide detailed AI model metrics. @zara"""
        ai_predictor = self.coordinator.ai_predictor
        if not ai_predictor:
            return {"status": "unavailable"}

        accuracy = getattr(ai_predictor, "current_accuracy", None)
        training_samples = getattr(ai_predictor, "training_samples", 0)
        num_groups = getattr(ai_predictor, "num_groups", 1)
        total_capacity = getattr(ai_predictor, "total_capacity", 0.0)
        last_training = getattr(ai_predictor, "last_training_time", None)

        confidence = ai_predictor.get_base_ai_confidence() if hasattr(ai_predictor, "get_base_ai_confidence") else 0.0

        feature_engineer = getattr(ai_predictor, "feature_engineer", None)
        feature_count = len(feature_engineer.feature_names) if feature_engineer and hasattr(feature_engineer, "feature_names") else 0

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
            "last_training": last_training.isoformat() if isinstance(last_training, datetime) else (last_training if isinstance(last_training, str) else None),
            "last_training_ago": format_time_ago(last_training) if isinstance(last_training, datetime) else "Never",
        }


class AIRmseSensor(BaseSolarSensor):
    """Sensor showing AI model RMSE (Root Mean Squared Error) in kWh. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_device_class = None
    _attr_icon = "mdi:target"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_ai_rmse"
        self._attr_translation_key = "ai_rmse"
        self._attr_name = "AI RMSE"

    @property
    def native_value(self) -> Optional[float]:
        """Return RMSE value in kWh. @zara"""
        ai_predictor = self.coordinator.ai_predictor
        if not ai_predictor:
            return None

        rmse = getattr(ai_predictor, "current_rmse", None)
        if rmse is not None:
            return round(rmse, 3)
        return None

    @property
    def icon(self) -> str:
        """Dynamic icon based on RMSE quality. @zara"""
        rmse = self.native_value
        if rmse is None:
            return "mdi:target"
        elif rmse < 0.5:
            return "mdi:target"
        elif rmse < 1.0:
            return "mdi:target-account"
        elif rmse < 2.0:
            return "mdi:target-variant"
        else:
            return "mdi:bullseye"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide RMSE interpretation and context. @zara"""
        ai_predictor = self.coordinator.ai_predictor
        if not ai_predictor:
            return {"status": "unavailable"}

        rmse = getattr(ai_predictor, "current_rmse", None)
        r2 = getattr(ai_predictor, "current_accuracy", None)

        if rmse is None:
            return {
                "status": "not_trained",
                "description": "RMSE will be available after first training",
            }

        if rmse < 0.3:
            quality = "excellent"
            description = "Very precise predictions"
        elif rmse < 0.5:
            quality = "very_good"
            description = "Very good prediction accuracy"
        elif rmse < 1.0:
            quality = "good"
            description = "Good prediction accuracy"
        elif rmse < 1.5:
            quality = "fair"
            description = "Acceptable accuracy, model is learning"
        elif rmse < 2.5:
            quality = "moderate"
            description = "Moderate accuracy, more training data needed"
        else:
            quality = "learning"
            description = "Model needs more time to learn"

        return {
            "status": "ready",
            "rmse_kwh": round(rmse, 3),
            "quality": quality,
            "description": description,
            "r2_score": round(r2, 4) if r2 is not None else None,
            "interpretation": f"Average deviation: +/-{rmse:.2f} kWh per hour",
        }


class ActivePredictionModelSensor(BaseSolarSensor):
    """Sensor showing which prediction model/strategy is currently active. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_active_prediction_model"
        self._attr_translation_key = "active_prediction_model"
        self._attr_icon = "mdi:brain"
        self._attr_name = "Active Prediction Model"

    @property
    def native_value(self) -> str:
        """Return active model/strategy. @zara"""
        orchestrator = getattr(self.coordinator, "forecast_orchestrator", None)
        ai_predictor = self.coordinator.ai_predictor

        ai_ready = ai_predictor is not None and ai_predictor.is_ready()

        physics_available = False
        if orchestrator:
            rb_strategy = getattr(orchestrator, "rule_based_strategy", None)
            physics_available = rb_strategy is not None and getattr(rb_strategy, "is_available", lambda: False)()

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
        """Dynamic icon based on active mode. @zara"""
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
        """Provide detailed model information. @zara"""
        orchestrator = getattr(self.coordinator, "forecast_orchestrator", None)
        ai_predictor = self.coordinator.ai_predictor

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
    """Sensor reflecting the health of the DataUpdateCoordinator. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_coordinator_health"
        self._attr_translation_key = "coordinator_health"
        self._attr_icon = "mdi:heart-pulse"
        self._attr_name = "Coordinator Health"

    @property
    def native_value(self) -> str:
        """Return health status. @zara"""
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
        """Provide detailed metrics. @zara"""
        last_success_time = getattr(self.coordinator, "last_update_success_time", None)
        last_attempt_time = getattr(self.coordinator, "last_update", None)

        return {
            "last_update_successful": getattr(self.coordinator, "last_update_success", False),
            "last_success_time_iso": last_success_time.isoformat() if isinstance(last_success_time, datetime) else (last_success_time if isinstance(last_success_time, str) else None),
            "last_attempt_time_iso": last_attempt_time.isoformat() if isinstance(last_attempt_time, datetime) else (last_attempt_time if isinstance(last_attempt_time, str) else None),
            "time_since_last_success": (
                format_time_ago(last_success_time) if isinstance(last_success_time, datetime) else "Never"
            ),
            "update_interval_seconds": (
                self.coordinator.update_interval.total_seconds()
                if self.coordinator.update_interval
                else None
            ),
        }


class DataFilesStatusSensor(BaseSolarSensor):
    """Sensor showing database status. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    CACHE_TTL_SECONDS = 60

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_data_files_status"
        self._attr_translation_key = "data_files_status"
        self._attr_icon = "mdi:database"
        self._attr_name = "Database Status"
        self._cached_status: Dict[str, Any] = {}
        self._cache_timestamp: float = 0

    @property
    def native_value(self) -> str:
        """Return database status. @zara"""
        db = self.db_manager
        if not db:
            return "Not Connected"

        return "Connected"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return detailed database status. @zara"""
        db = self.db_manager
        if not db:
            return {"status": "unavailable"}

        return {
            "db_path": str(db.db_path),
            "connected": db._db is not None,
        }


class PhysicsSamplesSensor(BaseSolarSensor):
    """Sensor showing AI training status. @zara"""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:brain"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize. @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_physics_samples"
        self._attr_translation_key = "physics_samples"
        self._attr_name = "AI Samples"

    @property
    def native_value(self) -> int:
        """Return AI training sample count. @zara"""
        try:
            ai_predictor = self.coordinator.ai_predictor
            if ai_predictor:
                return getattr(ai_predictor, "training_samples", 0)
            return 0
        except Exception as e:
            _LOGGER.debug(f"Error getting AI samples: {e}")
            return 0

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Provide AI status information. @zara"""
        try:
            ai_predictor = self.coordinator.ai_predictor
            if not ai_predictor:
                return {"status": "not_initialized"}

            training_samples = getattr(ai_predictor, "training_samples", 0)

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
                "hidden_size": getattr(ai_predictor, "hidden_size", 32),
                "input_size": getattr(ai_predictor, "input_size", 17),
            }
        except Exception as e:
            _LOGGER.debug(f"Error getting AI attributes: {e}")
            return {"status": "error", "message": str(e)}
