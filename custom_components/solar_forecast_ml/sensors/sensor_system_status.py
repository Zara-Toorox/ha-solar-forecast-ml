# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from homeassistant.components.sensor import SensorEntity
from homeassistant.core import callback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from ..const import DOMAIN
from ..coordinator import SolarForecastMLCoordinator

_LOGGER = logging.getLogger(__name__)

class SystemStatusSensor(CoordinatorEntity, SensorEntity):
    """Sensor showing the system status and last events"""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry_id: str):
        """Initialize the system status sensor @zara"""
        super().__init__(coordinator)
        self._attr_name = "System Status"
        self._attr_unique_id = f"{entry_id}_ml_system_status"
        self._attr_has_entity_name = True

        self._attr_native_value = "initializing"

        self._recent_events: deque = deque(maxlen=10)

        self._last_event_type: str = "startup"
        self._last_event_time: Optional[datetime] = datetime.now()
        self._last_event_status: str = "initializing"
        self._last_event_summary: str = "System is starting up..."
        self._last_event_details: Dict[str, Any] = {}

        self._warnings: List[str] = []

        _LOGGER.info("System Status Sensor initialized")

    @property
    def device_info(self):
        """Return device information @zara"""
        return {
            "identifiers": {(DOMAIN, self.coordinator.entry.entry_id)},
            "name": "Solar Forecast ML",
            "manufacturer": "Zara-Toorox",
            "model": "Solar Forecast ML AI-Version",
        }

    @property
    def icon(self) -> str:
        """Return the icon based on state @zara"""
        state = self._attr_native_value

        if state == "ok":
            return "mdi:check-circle"
        elif state == "warning":
            return "mdi:alert"
        elif state == "error":
            return "mdi:alert-circle"
        elif state == "running":
            return "mdi:loading"
        else:
            return "mdi:information"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return sensor attributes @zara"""

        ai_predictor = self.coordinator.ai_predictor

        ml_status = "unknown"
        ml_samples = 0
        ml_accuracy = None
        ml_last_training = None
        ml_next_check = None

        if ai_predictor:
            ml_status = (
                ai_predictor.model_state.value
                if hasattr(ai_predictor, "model_state")
                else "unknown"
            )
            ml_samples = getattr(ai_predictor, "training_samples", 0)
            ml_accuracy = getattr(ai_predictor, "current_accuracy", None)
            ml_last_training = getattr(ai_predictor, "last_training_time", None)

        last_event_time_str = None
        if self._last_event_time:
            last_event_time_str = self._last_event_time.isoformat()

        ml_last_training_str = None
        if ml_last_training:
            ml_last_training_str = ml_last_training.isoformat()

        recent_events = []
        for event in self._recent_events:
            event_copy = event.copy()
            if "time" in event_copy and isinstance(event_copy["time"], datetime):
                event_copy["time"] = event_copy["time"].isoformat()
            recent_events.append(event_copy)

        hourly_today = self._get_hourly_forecast_for_day("today")
        hourly_tomorrow = self._get_hourly_forecast_for_day("tomorrow")
        hourly_day_after = self._get_hourly_forecast_for_day("day_after_tomorrow")

        return {

            "last_event_type": self._last_event_type,
            "last_event_time": last_event_time_str,
            "last_event_status": self._last_event_status,
            "last_event_summary": self._last_event_summary,
            "last_event_details": self._last_event_details,

            "ml_model_status": ml_status,
            "ml_samples_total": ml_samples,
            "ml_model_accuracy": round(ml_accuracy * 100, 1) if ml_accuracy else None,
            "ml_last_training": ml_last_training_str,
            "ml_next_training_check": ml_next_check,
            "forecast_source": self._get_forecast_source(),
            "yesterday_accuracy": self.coordinator.yesterday_accuracy,
            "yesterday_deviation_kwh": self.coordinator.last_day_error_kwh,

            "hourly_forecast_today": hourly_today,
            "hourly_forecast_tomorrow": hourly_tomorrow,
            "hourly_forecast_day_after_tomorrow": hourly_day_after,

            "warnings": self._warnings,
            "warnings_count": len(self._warnings),

            "recent_events": recent_events,
            "recent_events_count": len(self._recent_events),
        }

    def _get_forecast_source(self) -> str:
        """Determine current forecast source @zara"""
        if hasattr(self.coordinator, "ai_predictor") and self.coordinator.ai_predictor:
            if self.coordinator.ai_predictor.is_ready():
                return "tinylstm"
        return "physics"

    def update_status(
        self,
        event_type: str,
        event_status: str,
        event_summary: str,
        event_details: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
    ) -> None:
        """Update sensor with new event information"""
        from ..core.core_helpers import SafeDateTimeUtil as dt_util

        now = dt_util.now()

        self._last_event_type = event_type
        self._last_event_time = now
        self._last_event_status = event_status
        self._last_event_summary = event_summary
        self._last_event_details = event_details or {}

        if warnings is not None:
            self._warnings = warnings

        event_record = {
            "type": event_type,
            "time": now,
            "status": event_status,
            "summary": event_summary,
        }
        self._recent_events.append(event_record)

        self._attr_native_value = self._calculate_state()

        self.async_write_ha_state()

        _LOGGER.debug(
            f"Status updated: event={event_type}, status={event_status}, state={self._attr_native_value}"
        )

    def _calculate_state(self) -> str:
        """Calculate overall system state @zara"""

        if self._last_event_status == "failed":
            return "error"

        if any("CRITICAL" in w.upper() for w in self._warnings):
            return "error"

        if len(self._warnings) > 0:
            return "warning"

        if self._last_event_status == "partial":
            return "warning"

        ai_predictor = self.coordinator.ai_predictor
        if ai_predictor and hasattr(ai_predictor, "model_state"):
            ml_state = ai_predictor.model_state.value
            if ml_state in ["degraded", "error"]:
                return "warning"

        if self._last_event_status == "running":
            return "running"

        return "ok"

    async def async_added_to_hass(self) -> None:
        """Run when entity is added to hass @zara"""
        await super().async_added_to_hass()

        self.update_status(
            event_type="initialization",
            event_status="success",
            event_summary="Solar Forecast ML erfolgreich initialisiert",
        )

        _LOGGER.info("System Status Sensor successfully added to Home Assistant")

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator @zara"""

        self._warnings = self._collect_warnings()
        self._attr_native_value = self._calculate_state()
        self.async_write_ha_state()

    def _collect_warnings(self) -> List[str]:
        """Collect current system warnings @zara"""
        warnings = []

        ai_predictor = self.coordinator.ai_predictor
        if ai_predictor:

            if hasattr(ai_predictor, "last_training_time") and ai_predictor.last_training_time:
                from datetime import timedelta

                from ..core.core_helpers import SafeDateTimeUtil as dt_util

                # Ensure both datetimes are timezone-aware before subtraction
                last_training = dt_util.ensure_local(ai_predictor.last_training_time)
                training_age = dt_util.now() - last_training
                if training_age > timedelta(days=14):
                    warnings.append(f"Letztes ML Training vor {training_age.days} Tagen")

            if hasattr(ai_predictor, "training_samples"):
                from ..const import MIN_TRAINING_DATA_POINTS

                if ai_predictor.training_samples < MIN_TRAINING_DATA_POINTS:
                    warnings.append(
                        f"Nicht genug Samples für Training: {ai_predictor.training_samples}/{MIN_TRAINING_DATA_POINTS}"
                    )

        if (
            hasattr(self.coordinator, "weather_fallback_active")
            and self.coordinator.weather_fallback_active
        ):
            warnings.append("Wetter-Service im Fallback-Modus")

        return warnings

    def _get_hourly_forecast_for_day(self, day: str) -> List[Dict[str, Any]]:
        """Extract hourly forecast data for a specific day @zara"""
        from datetime import timedelta

        from ..core.core_helpers import SafeDateTimeUtil as dt_util

        try:

            if not self.coordinator.data or not self.coordinator.data.get("hourly_forecast"):
                return []

            hourly_forecast = self.coordinator.data.get("hourly_forecast", [])
            if not hourly_forecast:
                return []

            now = dt_util.now()
            if day == "today":
                target_date = now.date()
            elif day == "tomorrow":
                target_date = (now + timedelta(days=1)).date()
            elif day == "day_after_tomorrow":
                target_date = (now + timedelta(days=2)).date()
            else:
                return []

            result = []
            for hour_data in hourly_forecast:
                try:

                    hour_dt = hour_data.get("local_datetime")
                    if not hour_dt:
                        continue

                    if isinstance(hour_dt, str):
                        hour_dt = dt_util.parse_datetime(hour_dt)
                        if not hour_dt:
                            continue

                    if hour_dt.date() == target_date:
                        result.append(
                            {
                                "hour": hour_dt.hour,
                                "datetime": hour_dt.isoformat(),
                                "production_kwh": round(hour_data.get("production_kwh", 0.0), 3),
                            }
                        )

                except Exception as e:
                    _LOGGER.debug(f"Error processing hourly data entry: {e}")
                    continue

            result.sort(key=lambda x: x["hour"])
            return result

        except Exception as e:
            _LOGGER.debug(f"Error extracting hourly forecast for {day}: {e}")
            return []
