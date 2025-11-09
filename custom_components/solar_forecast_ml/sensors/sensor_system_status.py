"""
System Status Sensor for Solar Forecast ML

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
from datetime import datetime
from typing import Any, Dict, Optional, List
from collections import deque

from homeassistant.components.sensor import SensorEntity
from homeassistant.core import callback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from ..coordinator import SolarForecastMLCoordinator
from ..const import DOMAIN

_LOGGER = logging.getLogger(__name__)


class SystemStatusSensor(CoordinatorEntity, SensorEntity):
    """Sensor showing the system status and last events"""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry_id: str):
        """Initialize the system status sensor"""
        super().__init__(coordinator)
        self._attr_name = "System Status"
        self._attr_unique_id = f"{entry_id}_ml_system_status"
        self._attr_has_entity_name = True

        # State tracking
        self._attr_native_value = "initializing"

        # Event tracking (max 10 recent events)
        self._recent_events: deque = deque(maxlen=10)

        # Last event details
        self._last_event_type: Optional[str] = None
        self._last_event_time: Optional[datetime] = None
        self._last_event_status: Optional[str] = None
        self._last_event_summary: Optional[str] = None
        self._last_event_details: Dict[str, Any] = {}

        # Warnings
        self._warnings: List[str] = []

        _LOGGER.info("System Status Sensor initialized")

    @property
    def device_info(self):
        """Return device information"""
        return {
            "identifiers": {(DOMAIN, self.coordinator.entry.entry_id)},
            "name": "Solar Forecast ML",
            "manufacturer": "Zara-Toorox",
            "model": "Solar Forecast ML Integration",
        }

    @property
    def icon(self) -> str:
        """Return the icon based on state"""
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
        """Return sensor attributes"""
        # Get ML predictor info
        ml_predictor = self.coordinator.ml_predictor

        ml_status = "unknown"
        ml_samples = 0
        ml_accuracy = None
        ml_last_training = None
        ml_next_check = None

        if ml_predictor:
            ml_status = ml_predictor.model_state.value if hasattr(ml_predictor, 'model_state') else "unknown"
            ml_samples = getattr(ml_predictor, 'training_samples', 0)
            ml_accuracy = getattr(ml_predictor, 'current_accuracy', None)
            ml_last_training = getattr(ml_predictor, 'last_training_time', None)

        # Convert datetime to ISO format
        last_event_time_str = None
        if self._last_event_time:
            last_event_time_str = self._last_event_time.isoformat()

        ml_last_training_str = None
        if ml_last_training:
            ml_last_training_str = ml_last_training.isoformat()

        # Build recent events list
        recent_events = []
        for event in self._recent_events:
            event_copy = event.copy()
            if 'time' in event_copy and isinstance(event_copy['time'], datetime):
                event_copy['time'] = event_copy['time'].isoformat()
            recent_events.append(event_copy)

        # Get hourly forecast data if available
        hourly_today = self._get_hourly_forecast_for_day("today")
        hourly_tomorrow = self._get_hourly_forecast_for_day("tomorrow")
        hourly_day_after = self._get_hourly_forecast_for_day("day_after_tomorrow")

        return {
            # ==================== LETZTES EVENT ====================
            "last_event_type": self._last_event_type,
            "last_event_time": last_event_time_str,
            "last_event_status": self._last_event_status,
            "last_event_summary": self._last_event_summary,
            "last_event_details": self._last_event_details,

            # ==================== SYSTEM HEALTH ====================
            "ml_model_status": ml_status,
            "ml_samples_total": ml_samples,
            "ml_model_accuracy": round(ml_accuracy * 100, 1) if ml_accuracy else None,
            "ml_last_training": ml_last_training_str,
            "ml_next_training_check": ml_next_check,

            "forecast_source": self._get_forecast_source(),
            "yesterday_accuracy": self.coordinator.yesterday_accuracy,
            "yesterday_deviation_kwh": self.coordinator.last_day_error_kwh,

            # ==================== HOURLY FORECASTS ====================
            "hourly_forecast_today": hourly_today,
            "hourly_forecast_tomorrow": hourly_tomorrow,
            "hourly_forecast_day_after_tomorrow": hourly_day_after,

            # ==================== WARNINGS ====================
            "warnings": self._warnings,
            "warnings_count": len(self._warnings),

            # ==================== EVENT HISTORY ====================
            "recent_events": recent_events,
            "recent_events_count": len(self._recent_events),
        }

    def _get_forecast_source(self) -> str:
        """Determine current forecast source"""
        if hasattr(self.coordinator, 'forecast_orchestrator'):
            orchestrator = self.coordinator.forecast_orchestrator
            if hasattr(orchestrator, 'ml_strategy') and orchestrator.ml_strategy:
                if hasattr(orchestrator.ml_strategy, 'is_available') and orchestrator.ml_strategy.is_available():
                    return "ml"
        return "weather"

    def update_status(
        self,
        event_type: str,
        event_status: str,
        event_summary: str,
        event_details: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None
    ) -> None:
        """Update sensor with new event information"""
        from ..core.core_helpers import SafeDateTimeUtil as dt_util

        now = dt_util.now()

        # Update last event
        self._last_event_type = event_type
        self._last_event_time = now
        self._last_event_status = event_status
        self._last_event_summary = event_summary
        self._last_event_details = event_details or {}

        # Update warnings
        if warnings is not None:
            self._warnings = warnings

        # Add to event history
        event_record = {
            "type": event_type,
            "time": now,
            "status": event_status,
            "summary": event_summary
        }
        self._recent_events.append(event_record)

        # Calculate new state
        self._attr_native_value = self._calculate_state()

        # Trigger update
        self.async_write_ha_state()

        _LOGGER.debug(
            f"Status updated: event={event_type}, status={event_status}, state={self._attr_native_value}"
        )

    def _calculate_state(self) -> str:
        """Calculate overall system state"""
        # Error wenn letztes Event fehlgeschlagen
        if self._last_event_status == "failed":
            return "error"

        # Error wenn kritische Warnings
        if any("CRITICAL" in w.upper() for w in self._warnings):
            return "error"

        # Warning wenn Warnings vorhanden
        if len(self._warnings) > 0:
            return "warning"

        # Warning wenn letztes Event nur teilweise erfolgreich
        if self._last_event_status == "partial":
            return "warning"

        # Warning wenn ML degraded
        ml_predictor = self.coordinator.ml_predictor
        if ml_predictor and hasattr(ml_predictor, 'model_state'):
            ml_state = ml_predictor.model_state.value
            if ml_state in ["degraded", "error"]:
                return "warning"

        # Running während Events
        if self._last_event_status == "running":
            return "running"

        # Alles OK
        return "ok"

    async def async_added_to_hass(self) -> None:
        """Run when entity is added to hass"""
        await super().async_added_to_hass()

        # Set initial status now that sensor is registered
        self.update_status(
            event_type="initialization",
            event_status="success",
            event_summary="Solar Forecast ML erfolgreich initialisiert"
        )

        _LOGGER.info("System Status Sensor successfully added to Home Assistant")

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator"""
        # Update state calculations based on coordinator data
        self._warnings = self._collect_warnings()
        self._attr_native_value = self._calculate_state()
        self.async_write_ha_state()

    def _collect_warnings(self) -> List[str]:
        """Collect current system warnings"""
        warnings = []

        # Check ML status
        ml_predictor = self.coordinator.ml_predictor
        if ml_predictor:
            # Check training age
            if hasattr(ml_predictor, 'last_training_time') and ml_predictor.last_training_time:
                from ..core.core_helpers import SafeDateTimeUtil as dt_util
                from datetime import timedelta

                training_age = dt_util.now() - ml_predictor.last_training_time
                if training_age > timedelta(days=14):
                    warnings.append(f"Letztes ML Training vor {training_age.days} Tagen")

            # Check sample count
            if hasattr(ml_predictor, 'training_samples'):
                from ..const import MIN_TRAINING_DATA_POINTS
                if ml_predictor.training_samples < MIN_TRAINING_DATA_POINTS:
                    warnings.append(
                        f"Nicht genug Samples für Training: {ml_predictor.training_samples}/{MIN_TRAINING_DATA_POINTS}"
                    )

        # Check weather service
        if hasattr(self.coordinator, 'weather_fallback_active') and self.coordinator.weather_fallback_active:
            warnings.append("Wetter-Service im Fallback-Modus")

        return warnings

    def _get_hourly_forecast_for_day(self, day: str) -> List[Dict[str, Any]]:
        """Extract hourly forecast data for a specific day

        Args:
            day: "today", "tomorrow", or "day_after_tomorrow"

        Returns:
            List of dicts with hour and production_kwh, or empty list if not available
        """
        from ..core.core_helpers import SafeDateTimeUtil as dt_util
        from datetime import timedelta

        try:
            # Get coordinator data
            if not self.coordinator.data or not self.coordinator.data.get("hourly_forecast"):
                return []

            hourly_forecast = self.coordinator.data.get("hourly_forecast", [])
            if not hourly_forecast:
                return []

            # Determine target date
            now = dt_util.now()
            if day == "today":
                target_date = now.date()
            elif day == "tomorrow":
                target_date = (now + timedelta(days=1)).date()
            elif day == "day_after_tomorrow":
                target_date = (now + timedelta(days=2)).date()
            else:
                return []

            # Filter hourly data for target date
            result = []
            for hour_data in hourly_forecast:
                try:
                    # Get hour datetime
                    hour_dt = hour_data.get("local_datetime")
                    if not hour_dt:
                        continue

                    # Parse if string
                    if isinstance(hour_dt, str):
                        hour_dt = dt_util.parse_datetime(hour_dt)
                        if not hour_dt:
                            continue

                    # Check if this hour belongs to target date
                    if hour_dt.date() == target_date:
                        result.append({
                            "hour": hour_dt.hour,
                            "datetime": hour_dt.isoformat(),
                            "production_kwh": round(hour_data.get("production_kwh", 0.0), 3)
                        })

                except Exception as e:
                    _LOGGER.debug(f"Error processing hourly data entry: {e}")
                    continue

            # Sort by hour
            result.sort(key=lambda x: x["hour"])
            return result

        except Exception as e:
            _LOGGER.debug(f"Error extracting hourly forecast for {day}: {e}")
            return []
