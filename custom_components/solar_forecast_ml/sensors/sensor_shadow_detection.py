"""Shadow Detection Sensors V10.0.0 @zara

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
from typing import Any, Dict, List, Optional

from homeassistant.components.sensor import SensorEntity, SensorDeviceClass, SensorStateClass
from homeassistant.const import PERCENTAGE
from homeassistant.core import callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from ..const import DOMAIN, INTEGRATION_MODEL, SOFTWARE_VERSION, ML_VERSION

_LOGGER = logging.getLogger(__name__)

def _get_today_predictions_from_cache(coordinator) -> List[Dict[str, Any]]:
    """Get today's predictions from coordinator cache - NO FILE I/O! @zara"""
    try:
        if not coordinator:
            return []

        cache = getattr(coordinator, "_hourly_predictions_cache", None)
        if not cache:
            _LOGGER.debug("No hourly predictions cache available in coordinator")
            return []

        today = dt_util.now().date().isoformat()

        return [
            p for p in cache.get("predictions", [])
            if p.get("target_date") == today
        ]
    except Exception as e:
        _LOGGER.debug(f"Error getting today predictions from cache: {e}")
        return []

def _get_hourly_predictions_handler(coordinator):
    """Return an hourly_predictions handler, falling back to DataManager if @zara"""
    handler = None
    try:
        if coordinator and coordinator.data:
            handler = coordinator.data.get("hourly_predictions_handler")
        if not handler and hasattr(coordinator, "data_manager"):
            handler = getattr(coordinator.data_manager, "hourly_predictions", None)
    except Exception:
        handler = None
    return handler

def _filter_valid_shadow_predictions(predictions: List[Dict]) -> List[Dict]:
    """Filter predictions to only include valid shadow detection entries @zara"""
    return [
        p for p in predictions
        if p.get("shadow_detection") is not None
        and p.get("shadow_detection", {}).get("shadow_type") not in ["night", "error", None]
    ]

class ShadowCurrentSensor(CoordinatorEntity, SensorEntity):
    """Sensor for current hour shadow detection status.

    Uses internal cache to avoid repeated data lookups.
    """

    def __init__(self, coordinator, entry):
        """Initialize the sensor @zara"""
        super().__init__(coordinator)
        self._entry = entry
        self._attr_has_entity_name = True
        self._attr_translation_key = "shadow_current"
        self._attr_unique_id = f"{entry.entry_id}_shadow_current"
        self._attr_device_class = None
        self._attr_state_class = None
        self._attr_icon = "mdi:weather-sunny-alert"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Solar Forecast ML",
            manufacturer="Zara-Toorox",
            model=INTEGRATION_MODEL,
            sw_version=f"SW {SOFTWARE_VERSION} | ML {ML_VERSION}",
            configuration_url="https://github.com/Zara-Toorox/ha-solar-forecast-ml",
        )

        self._cached_prediction: Optional[Dict[str, Any]] = None
        self._cache_hour: Optional[int] = None

    def _get_current_prediction(self) -> Optional[Dict[str, Any]]:
        """Get current hour prediction with caching to avoid repeated lookups @zara"""
        now = dt_util.now()
        current_hour = now.hour

        if self._cache_hour == current_hour and self._cached_prediction is not None:
            return self._cached_prediction

        try:
            current_date = now.date().isoformat()
            prediction_id = f"{current_date}_{current_hour}"

            cache = getattr(self.coordinator, "_hourly_predictions_cache", None)
            if cache and cache.get("predictions"):
                self._cached_prediction = next(
                    (p for p in cache["predictions"] if p.get("id") == prediction_id), None
                )
            else:
                self._cached_prediction = None

            self._cache_hour = current_hour

            return self._cached_prediction

        except Exception as e:
            _LOGGER.debug(f"Error refreshing prediction cache: {e}")
            self._cached_prediction = None
            self._cache_hour = current_hour
            return None

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates - invalidate cache. @zara"""

        self._cache_hour = None
        super()._handle_coordinator_update()

    @property
    def available(self) -> bool:
        """Return if entity is available based on valid prediction data. @zara"""
        if not self.coordinator.last_update_success:
            return False
        hourly_predictions = _get_hourly_predictions_handler(self.coordinator)
        return hourly_predictions is not None

    @property
    def native_value(self) -> Optional[str]:
        """Return the current shadow status @zara"""
        prediction = self._get_current_prediction()

        if not prediction:
            return "no_data"

        shadow_det = prediction.get("shadow_detection", {})
        shadow_type = shadow_det.get("shadow_type", "unknown")

        type_to_state = {
            "none": "clear",
            "light": "light_shadow",
            "moderate": "moderate_shadow",
            "heavy": "heavy_shadow",
            "night": "night",
            "error": "error",
            "unknown": "no_data"
        }

        return type_to_state.get(shadow_type, "no_data")

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional shadow detection attributes @zara"""
        prediction = self._get_current_prediction()

        if not prediction:
            return {"status": "no_prediction_data"}

        try:
            shadow_det = prediction.get("shadow_detection", {})

            attrs = {
                "shadow_type": shadow_det.get("shadow_type", "unknown"),
                "shadow_percent": shadow_det.get("shadow_percent", 0),
                "confidence": shadow_det.get("confidence", 0),
                "root_cause": shadow_det.get("root_cause", "unknown"),
                "interpretation": shadow_det.get("interpretation", "N/A"),
                "efficiency_ratio": shadow_det.get("efficiency_ratio", 0),
                "loss_kwh": shadow_det.get("loss_kwh", 0),
            }

            methods = shadow_det.get("methods", {})
            if methods:
                theory = methods.get("theory_ratio", {})
                fusion = methods.get("sensor_fusion", {})

                attrs["method_theory_shadow"] = theory.get("shadow_percent", 0)
                attrs["method_theory_confidence"] = theory.get("confidence", 0)
                attrs["method_fusion_shadow"] = fusion.get("shadow_percent", 0)
                attrs["method_fusion_confidence"] = fusion.get("confidence", 0)
                attrs["method_fusion_mode"] = fusion.get("mode", "unknown")

            attrs["actual_kwh"] = prediction.get("actual_kwh", 0)
            attrs["theoretical_max_kwh"] = shadow_det.get("theoretical_max_kwh", 0)

            return attrs

        except Exception as e:
            _LOGGER.error(f"Error getting shadow current attributes: {e}")
            return {"error": str(e)}

    @property
    def icon(self) -> str:
        """Return icon based on shadow status @zara"""
        prediction = self._get_current_prediction()

        if not prediction:
            return "mdi:help-circle"

        shadow_det = prediction.get("shadow_detection", {})
        shadow_type = shadow_det.get("shadow_type", "unknown")

        icon_map = {
            "none": "mdi:weather-sunny",
            "light": "mdi:weather-partly-cloudy",
            "moderate": "mdi:weather-cloudy",
            "heavy": "mdi:weather-cloudy-alert",
            "night": "mdi:weather-night",
            "error": "mdi:alert-circle"
        }

        return icon_map.get(shadow_type, "mdi:help-circle")

class ShadowTodaySensor(CoordinatorEntity, SensorEntity):
    """Sensor for today's cumulative shadow analysis.

    Refactored to use coordinator cache and internal data caching.
    """

    def __init__(self, coordinator, entry):
        """Initialize the sensor @zara"""
        super().__init__(coordinator)
        self._entry = entry
        self._attr_has_entity_name = True
        self._attr_translation_key = "shadow_today"
        self._attr_unique_id = f"{entry.entry_id}_shadow_today"
        self._attr_device_class = None

        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_native_unit_of_measurement = "hours"
        self._attr_icon = "mdi:weather-sunset"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Solar Forecast ML",
            manufacturer="Zara-Toorox",
            model=INTEGRATION_MODEL,
            sw_version=f"SW {SOFTWARE_VERSION} | ML {ML_VERSION}",
            configuration_url="https://github.com/Zara-Toorox/ha-solar-forecast-ml",
        )

        self._cached_analysis: Optional[Dict[str, Any]] = None
        self._cache_date: Optional[str] = None

    def _get_today_analysis(self) -> Dict[str, Any]:
        """Calculate today's shadow analysis with caching @zara"""
        today = dt_util.now().date().isoformat()

        if self._cache_date == today and self._cached_analysis is not None:
            return self._cached_analysis

        try:
            today_predictions = _get_today_predictions_from_cache(self.coordinator)
            valid_predictions = _filter_valid_shadow_predictions(today_predictions)

            if not valid_predictions:
                self._cached_analysis = {"status": "no_data", "shadow_hours": 0}
                self._cache_date = today
                return self._cached_analysis

            shadow_types = {"none": 0, "light": 0, "moderate": 0, "heavy": 0}
            total_loss_kwh = 0.0
            total_theoretical_kwh = 0.0
            shadow_hours_list = []
            peak_shadow_hour = None
            peak_shadow_percent = 0.0

            for pred in valid_predictions:
                shadow_det = pred.get("shadow_detection", {})
                shadow_type = shadow_det.get("shadow_type", "unknown")

                if shadow_type in shadow_types:
                    shadow_types[shadow_type] += 1

                if shadow_type in ["moderate", "heavy"]:
                    hour = pred.get("target_hour")
                    if hour is not None:
                        shadow_hours_list.append(hour)

                loss_kwh = shadow_det.get("loss_kwh")
                theoretical_max = shadow_det.get("theoretical_max_kwh")

                if isinstance(loss_kwh, (int, float)) and loss_kwh >= 0:
                    total_loss_kwh += loss_kwh
                if isinstance(theoretical_max, (int, float)) and theoretical_max >= 0:
                    total_theoretical_kwh += theoretical_max

                shadow_percent = shadow_det.get("shadow_percent", 0)
                if isinstance(shadow_percent, (int, float)) and shadow_percent > peak_shadow_percent:
                    peak_shadow_percent = shadow_percent
                    peak_shadow_hour = pred.get("target_hour")

            if total_theoretical_kwh > 0:
                daily_loss_percent = (total_loss_kwh / total_theoretical_kwh) * 100.0
            else:
                daily_loss_percent = 0.0

            self._cached_analysis = {
                "shadow_hours": float(len(shadow_hours_list)),
                "shadow_hours_count": len(shadow_hours_list),
                "shadow_hours_list": shadow_hours_list,
                "none_count": shadow_types["none"],
                "light_count": shadow_types["light"],
                "moderate_count": shadow_types["moderate"],
                "heavy_count": shadow_types["heavy"],
                "total_analyzed_hours": len(valid_predictions),
                "peak_shadow_hour": peak_shadow_hour,
                "peak_shadow_percent": round(peak_shadow_percent, 1),
                "cumulative_loss_kwh": round(total_loss_kwh, 3),
                "daily_loss_percent": round(daily_loss_percent, 1),
                "date": today
            }
            self._cache_date = today

            return self._cached_analysis

        except Exception as e:
            _LOGGER.error(f"Error calculating shadow today analysis: {e}")
            self._cached_analysis = {"status": "error", "error": str(e), "shadow_hours": 0}
            self._cache_date = today
            return self._cached_analysis

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates - invalidate cache. @zara"""
        self._cache_date = None
        super()._handle_coordinator_update()

    @property
    def available(self) -> bool:
        """Return if entity is available based on valid prediction data. @zara"""
        if not self.coordinator.last_update_success:
            return False
        hourly_predictions = _get_hourly_predictions_handler(self.coordinator)
        return hourly_predictions is not None

    @property
    def native_value(self) -> Optional[float]:
        """Return number of shadow hours today @zara"""
        analysis = self._get_today_analysis()
        return analysis.get("shadow_hours", 0)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return today's shadow analysis attributes @zara"""
        analysis = self._get_today_analysis()

        if analysis.get("status") == "no_data":
            return {"status": "no_data"}

        if analysis.get("status") == "error":
            return {"error": analysis.get("error", "unknown")}

        return {
            "shadow_hours_count": analysis.get("shadow_hours_count", 0),
            "shadow_hours": analysis.get("shadow_hours_list", []),
            "none_count": analysis.get("none_count", 0),
            "light_count": analysis.get("light_count", 0),
            "moderate_count": analysis.get("moderate_count", 0),
            "heavy_count": analysis.get("heavy_count", 0),
            "total_analyzed_hours": analysis.get("total_analyzed_hours", 0),
            "peak_shadow_hour": analysis.get("peak_shadow_hour"),
            "peak_shadow_percent": analysis.get("peak_shadow_percent", 0),
            "cumulative_loss_kwh": analysis.get("cumulative_loss_kwh", 0),
            "daily_loss_percent": analysis.get("daily_loss_percent", 0),
            "date": analysis.get("date", "")
        }

class PerformanceLossTodaySensor(CoordinatorEntity, SensorEntity):
    """Sensor for today's performance loss due to shading.

    Refactored to use coordinator cache and internal data caching.
    """

    def __init__(self, coordinator, entry):
        """Initialize the sensor @zara"""
        super().__init__(coordinator)
        self._entry = entry
        self._attr_has_entity_name = True
        self._attr_translation_key = "performance_loss_today"
        self._attr_unique_id = f"{entry.entry_id}_performance_loss_today"
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_native_unit_of_measurement = "kWh"
        self._attr_icon = "mdi:solar-power-variant"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Solar Forecast ML",
            manufacturer="Zara-Toorox",
            model=INTEGRATION_MODEL,
            sw_version=f"SW {SOFTWARE_VERSION} | ML {ML_VERSION}",
            configuration_url="https://github.com/Zara-Toorox/ha-solar-forecast-ml",
        )

        self._cached_analysis: Optional[Dict[str, Any]] = None
        self._cache_date: Optional[str] = None

    def _get_performance_analysis(self) -> Dict[str, Any]:
        """Calculate today's performance loss analysis with caching @zara"""
        today = dt_util.now().date().isoformat()

        if self._cache_date == today and self._cached_analysis is not None:
            return self._cached_analysis

        try:
            today_predictions = _get_today_predictions_from_cache(self.coordinator)

            all_with_shadow = [
                p for p in today_predictions
                if p.get("shadow_detection") is not None
            ]

            valid_predictions = _filter_valid_shadow_predictions(today_predictions)

            if not all_with_shadow:
                self._cached_analysis = {"status": "no_data", "total_loss_kwh": 0.0, "loss_percent": 0}
                self._cache_date = today
                return self._cached_analysis

            total_loss = 0.0
            for p in all_with_shadow:
                loss_kwh = p.get("shadow_detection", {}).get("loss_kwh")
                if isinstance(loss_kwh, (int, float)) and loss_kwh >= 0:
                    total_loss += loss_kwh

            if not valid_predictions:
                self._cached_analysis = {
                    "status": "partial",
                    "total_loss_kwh": round(total_loss, 3),
                    "loss_percent": 0,
                    "hours_analyzed": 0
                }
                self._cache_date = today
                return self._cached_analysis

            total_actual = 0.0
            total_theoretical = 0.0
            root_causes: Dict[str, int] = {}

            for pred in valid_predictions:

                actual = pred.get("actual_kwh")
                if isinstance(actual, (int, float)) and actual >= 0:
                    total_actual += actual

                shadow_det = pred.get("shadow_detection", {})
                theoretical = shadow_det.get("theoretical_max_kwh")
                if isinstance(theoretical, (int, float)) and theoretical >= 0:
                    total_theoretical += theoretical

                cause = shadow_det.get("root_cause", "unknown")
                root_causes[cause] = root_causes.get(cause, 0) + 1

            if total_theoretical > 0:
                overall_efficiency = (total_actual / total_theoretical) * 100.0
                loss_percent = (total_loss / total_theoretical) * 100.0
            else:
                overall_efficiency = 0.0
                loss_percent = 0.0

            dominant_cause = max(root_causes, key=root_causes.get) if root_causes else "unknown"

            self._cached_analysis = {
                "total_loss_kwh": round(total_loss, 3),
                "total_actual_kwh": round(total_actual, 3),
                "total_theoretical_kwh": round(total_theoretical, 3),
                "overall_efficiency_percent": round(overall_efficiency, 1),
                "loss_percent": round(loss_percent, 1),
                "root_causes": root_causes,
                "dominant_cause": dominant_cause,
                "hours_analyzed": len(valid_predictions),
                "date": today
            }
            self._cache_date = today

            return self._cached_analysis

        except Exception as e:
            _LOGGER.error(f"Error calculating performance loss analysis: {e}")
            self._cached_analysis = {"status": "error", "error": str(e), "total_loss_kwh": 0.0, "loss_percent": 0}
            self._cache_date = today
            return self._cached_analysis

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates - invalidate cache. @zara"""
        self._cache_date = None
        super()._handle_coordinator_update()

    @property
    def available(self) -> bool:
        """Return if entity is available based on valid prediction data. @zara"""
        if not self.coordinator.last_update_success:
            return False
        hourly_predictions = _get_hourly_predictions_handler(self.coordinator)
        return hourly_predictions is not None

    @property
    def native_value(self) -> Optional[float]:
        """Return cumulative kWh lost today due to shading @zara"""
        analysis = self._get_performance_analysis()
        return analysis.get("total_loss_kwh", 0.0)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return performance loss analysis attributes @zara"""
        analysis = self._get_performance_analysis()

        if analysis.get("status") == "no_data":
            return {"status": "no_data"}

        if analysis.get("status") == "error":
            return {"error": analysis.get("error", "unknown")}

        return {
            "total_actual_kwh": analysis.get("total_actual_kwh", 0),
            "total_theoretical_kwh": analysis.get("total_theoretical_kwh", 0),
            "total_loss_kwh": analysis.get("total_loss_kwh", 0),
            "overall_efficiency_percent": analysis.get("overall_efficiency_percent", 0),
            "loss_percent": analysis.get("loss_percent", 0),
            "root_causes": analysis.get("root_causes", {}),
            "dominant_cause": analysis.get("dominant_cause", "unknown"),
            "hours_analyzed": analysis.get("hours_analyzed", 0),
            "date": analysis.get("date", "")
        }

    @property
    def icon(self) -> str:
        """Return icon based on loss severity @zara"""
        analysis = self._get_performance_analysis()
        loss_percent = analysis.get("loss_percent", 0)

        if not isinstance(loss_percent, (int, float)):
            return "mdi:solar-power-variant"

        if loss_percent < 10:
            return "mdi:solar-power"
        elif loss_percent < 25:
            return "mdi:solar-power-variant"
        else:
            return "mdi:solar-power-variant-outline"

SHADOW_DETECTION_SENSORS = [
    ShadowCurrentSensor,
    ShadowTodaySensor,
    PerformanceLossTodaySensor,
]
