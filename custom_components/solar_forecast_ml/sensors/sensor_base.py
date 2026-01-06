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
from typing import Optional

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, UnitOfEnergy
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from ..const import DOMAIN, INTEGRATION_MODEL, AI_VERSION, SOFTWARE_VERSION
from ..coordinator import SolarForecastMLCoordinator
from .sensor_mixins import (
    AlwaysAvailableFileBasedMixin,
    FileBasedSensorMixin,
    StatisticsFileBasedMixin,
)

_LOGGER = logging.getLogger(__name__)

class BaseSolarSensor(CoordinatorEntity, SensorEntity):
    """Base class for core sensors updated by the coordinator"""

    _attr_has_entity_name = True

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the base sensor @zara"""
        super().__init__(coordinator)
        self.entry = entry
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Solar Forecast ML",
            manufacturer="Zara-Toorox",
            model=INTEGRATION_MODEL,
            sw_version=f"SW {SOFTWARE_VERSION} | AI {AI_VERSION}",
            configuration_url="https://github.com/Zara-Toorox/ha-solar-forecast-ml",
        )

    @property
    def available(self) -> bool:
        """Return if entity is available based on coordinator @zara"""

        return self.coordinator.last_update_success and self.coordinator.data is not None

class SolarForecastSensor(SensorEntity):
    """Sensor for todays or tomorrows solar forecast"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry, key: str):
        """Initialize the forecast sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        self._key = key

        self._key_mapping = {
            "remaining": {"data_key": "prediction_kwh", "translation_key": "today_forecast"},
            "tomorrow": {"data_key": "forecast_tomorrow", "translation_key": "tomorrow_forecast"},
        }

        if key not in self._key_mapping:
            raise ValueError(f"Invalid sensor key: {key}. Must be 'remaining' or 'tomorrow'")

        config = self._key_mapping[key]
        self._data_key = config["data_key"]

        self._attr_unique_id = f"{entry.entry_id}_ml_forecast_{key}"
        self._attr_translation_key = config["translation_key"]
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:solar-power"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Return if entity is available @zara"""
        return True

    @property
    def native_value(self) -> float:
        """Return the forecast value @zara"""
        if self._key == "tomorrow":

            if not self._coordinator.data:
                return 0.0
            return self._coordinator.data.get(self._data_key) or 0.0

        production_time_sensor = self.hass.states.get(f"sensor.{DOMAIN}_production_time")
        if production_time_sensor and production_time_sensor.state == "00:00:00":

            return 0.0

        try:
            from homeassistant.util import dt as dt_util

            hourly_data = getattr(self._coordinator, "_hourly_predictions_cache", None)

            if not hourly_data:
                try:
                    hourly_data = self._coordinator.data_manager.hourly_predictions._read_json()
                except Exception:
                    return 0.0

            if not hourly_data or not isinstance(hourly_data, dict):
                return 0.0

            predictions = hourly_data.get("predictions", [])
            if not predictions:
                return 0.0

            now = dt_util.now()
            current_hour = now.hour
            today_str = now.strftime("%Y-%m-%d")

            remaining_kwh = 0.0
            for pred in predictions:
                if (
                    pred.get("target_date") == today_str
                    and pred.get("target_hour", -1) >= current_hour
                ):

                    kwh = pred.get("prediction_kwh") or pred.get("predicted_kwh", 0.0)
                    remaining_kwh += kwh if kwh is not None else 0.0

            return round(remaining_kwh, 2)

        except Exception as e:
            _LOGGER.warning(f"Failed to calculate remaining from hourly predictions: {e}")
            return 0.0

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass @zara"""
        await super().async_added_to_hass()

        self.async_on_remove(self._coordinator.async_add_listener(self._handle_coordinator_update))

        if self._key == "remaining":
            production_time_entity = f"sensor.{DOMAIN}_production_time"
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, production_time_entity, self._handle_production_time_change
                )
            )

        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates @zara"""
        self.async_write_ha_state()

    @callback
    def _handle_production_time_change(self, event) -> None:
        """Handle production time sensor changes - update state when production stops @zara"""
        self.async_write_ha_state()

class NextHourSensor(AlwaysAvailableFileBasedMixin, SensorEntity):
    """Sensor for the next hours solar forecast - reads directly from hourly_predictions.json"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the next hour sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        AlwaysAvailableFileBasedMixin.__init__(self)

        self._upcoming_hours = []

        self._attr_unique_id = f"{entry.entry_id}_ml_next_hour_forecast"
        self._attr_translation_key = "next_hour_forecast"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:clock-fast"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def native_value(self) -> float:
        """Return next hour forecast or 0.0 if no data @zara"""
        return self._cached_value if self._cached_value is not None else 0.0

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes showing all upcoming hours for the day @zara"""
        if not self._upcoming_hours:
            return {}

        attributes = {}

        for i, hour_data in enumerate(self._upcoming_hours, start=1):
            attributes[f"hour_{i}"] = hour_data.get("kwh", 0.0)
            attributes[f"hour_{i}_time"] = hour_data.get("time", "")

        total_upcoming = sum(h.get("kwh", 0.0) for h in self._upcoming_hours)
        attributes["total_upcoming"] = round(total_upcoming, 2)

        attributes["hours_count"] = len(self._upcoming_hours)

        attributes["hours_list"] = self._upcoming_hours

        return attributes

    async def _load_from_file(self) -> None:
        """Load next hour forecast and upcoming hours from hourly_predictions.json @zara"""
        try:
            from homeassistant.util import dt as dt_util

            hourly_data = await self._coordinator.data_manager.hourly_predictions._read_json_async()
            if not hourly_data or not isinstance(hourly_data, dict):
                self._cached_value = None
                self._upcoming_hours = []
                return

            predictions = hourly_data.get("predictions", [])
            if not predictions:
                self._cached_value = None
                self._upcoming_hours = []
                return

            now_local = dt_util.now()
            today = now_local.date().isoformat()
            current_hour = now_local.hour

            upcoming_predictions = [
                pred
                for pred in predictions
                if pred.get("target_date") == today and pred.get("target_hour", -1) > current_hour
            ]

            upcoming_predictions.sort(key=lambda p: p.get("target_hour", 0))

            self._upcoming_hours = [
                {
                    "time": f"{pred.get('target_hour', 0):02d}:00",
                    "kwh": pred.get("prediction_kwh", 0.0),
                }
                for pred in upcoming_predictions
            ]

            if upcoming_predictions:
                self._cached_value = upcoming_predictions[0].get("prediction_kwh", 0.0)
                next_hour = upcoming_predictions[0].get("target_hour")
                _LOGGER.debug(
                    f"NextHourSensor: Found prediction for {today} {next_hour:02d}:00 = {self._cached_value} kWh"
                )
            else:
                self._cached_value = 0.0
                _LOGGER.debug(f"NextHourSensor: No upcoming predictions found for {today}")

        except Exception as e:
            _LOGGER.warning(f"Failed to load NextHourSensor from hourly_predictions.json: {e}")
            self._cached_value = None
            self._upcoming_hours = []

    def extract_value_from_file(self, forecast_data: dict) -> Optional[float]:
        """Not used anymore - kept for compatibility with mixin @zara"""
        return None

class PeakProductionHourSensor(AlwaysAvailableFileBasedMixin, SensorEntity):
    """Sensor showing best production hour - reads from best_hour_today field in hourly_predictions.json"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the peak hour sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        AlwaysAvailableFileBasedMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_peak_production_hour"
        self._attr_translation_key = "peak_production_hour"
        self._attr_icon = "mdi:solar-power-variant-outline"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def native_value(self) -> str:
        """Return peak hour or '--:--' if no data @zara"""
        return self._cached_value if self._cached_value is not None else "--:--"

    async def _load_from_file(self) -> None:
        """Load best hour from hourly_predictions.json best_hour_today field @zara"""
        try:

            best_hour_str = (
                await self._coordinator.data_manager.hourly_predictions.get_best_hour_string()
            )

            if best_hour_str:
                self._cached_value = best_hour_str
                _LOGGER.debug(f"PeakProductionHourSensor: Loaded best hour = {best_hour_str}")
            else:
                self._cached_value = None
                _LOGGER.debug(f"PeakProductionHourSensor: No best_hour_today found in file")

        except Exception as e:
            _LOGGER.warning(f"Failed to load PeakProductionHourSensor: {e}")
            self._cached_value = None

    def extract_value_from_file(self, forecast_data: dict) -> Optional[str]:
        """Not used anymore - kept for compatibility with mixin @zara"""
        return None

class AverageYieldSensor(BaseSolarSensor):
    """Sensor for the calculated average monthly yield"""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average yield sensor @zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_average_yield"
        self._attr_translation_key = "average_yield"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:chart-line"

    @property
    def native_value(self) -> float | None:
        """Return the average monthly yield @zara"""
        value = getattr(self.coordinator, "avg_month_yield", None)
        return value if value is not None and value > 0 else None

class AutarkySensor(SensorEntity):
    """Sensor for the calculated self-sufficiency Autarky with live updates"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the autarky sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        self._yield_entity = entry.data.get("solar_yield_today")
        self._consumption_entity = entry.data.get("total_consumption_today")
        self._grid_import_entity = entry.data.get("grid_import_today")
        self._grid_export_entity = entry.data.get("grid_export_today")

        self._attr_unique_id = f"{entry.entry_id}_ml_self_sufficiency"
        self._attr_translation_key = "self_sufficiency"
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:home-lightning-bolt-outline"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Return if entity is available @zara"""
        return (
            self._coordinator.last_update_success
            and self._coordinator.data is not None
            and self.native_value is not None
        )

    @property
    def native_value(self) -> float | None:
        """Return the autarky percentage - LIVE calculation @zara"""

        coord_value = getattr(self._coordinator, "autarky_today", None)
        if coord_value is not None:
            return max(0.0, min(100.0, coord_value))

        if self._grid_import_entity and self._consumption_entity:
            try:
                grid_import_state = self.hass.states.get(self._grid_import_entity)
                consumption_state = self.hass.states.get(self._consumption_entity)

                if (
                    grid_import_state
                    and consumption_state
                    and grid_import_state.state not in ["unavailable", "unknown", "none", None, ""]
                    and consumption_state.state not in ["unavailable", "unknown", "none", None, ""]
                ):

                    grid_import_val = float(
                        str(grid_import_state.state).split()[0].replace(",", ".")
                    )
                    consumption_val = float(
                        str(consumption_state.state).split()[0].replace(",", ".")
                    )

                    if consumption_val > 0:

                        autarky = ((consumption_val - grid_import_val) / consumption_val) * 100.0
                        return max(0.0, min(100.0, autarky))
            except (ValueError, TypeError, AttributeError):
                pass

        if self._yield_entity and self._consumption_entity:
            try:
                yield_state = self.hass.states.get(self._yield_entity)
                consumption_state = self.hass.states.get(self._consumption_entity)

                if (
                    yield_state
                    and consumption_state
                    and yield_state.state not in ["unavailable", "unknown", "none", None, ""]
                    and consumption_state.state not in ["unavailable", "unknown", "none", None, ""]
                ):

                    yield_val = float(str(yield_state.state).split()[0].replace(",", "."))
                    consumption_val = float(
                        str(consumption_state.state).split()[0].replace(",", ".")
                    )

                    if consumption_val > 0:
                        autarky = (yield_val / consumption_val) * 100.0
                        return max(0.0, min(100.0, autarky))
            except (ValueError, TypeError, AttributeError):
                pass

        return None

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass @zara"""
        await super().async_added_to_hass()

        self.async_on_remove(self._coordinator.async_add_listener(self._handle_coordinator_update))

        if self._yield_entity:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, self._yield_entity, self._handle_sensor_change
                )
            )
        if self._consumption_entity:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, self._consumption_entity, self._handle_sensor_change
                )
            )

        if self._grid_import_entity:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, self._grid_import_entity, self._handle_sensor_change
                )
            )
        if self._grid_export_entity:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, self._grid_export_entity, self._handle_sensor_change
                )
            )

        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates @zara"""
        self.async_write_ha_state()

    @callback
    def _handle_sensor_change(self, event) -> None:
        """Handle yield or consumption sensor state changes @zara"""
        self.async_write_ha_state()

class ExpectedDailyProductionSensor(FileBasedSensorMixin, SensorEntity):
    """Sensor for expected daily production morning snapshot"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the expected daily production sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        FileBasedSensorMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_expected_daily_production"
        self._attr_translation_key = "expected_daily_production"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:solar-power-variant"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> Optional[float]:
        """Extract expected daily production from daily_forecasts.json @zara"""
        today = forecast_data.get("today", {})
        forecast_day = today.get("forecast_day", {})
        return forecast_day.get("prediction_kwh")

class ProductionTimeSensor(FileBasedSensorMixin, SensorEntity):
    """Sensor for production time today - reads directly from daily_forecastsjson"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the production time sensor @zara"""
        self.entry = entry
        self._coordinator = coordinator
        FileBasedSensorMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_production_time"
        self._attr_translation_key = "production_time"
        self._attr_icon = "mdi:timer-outline"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> Optional[str]:
        """Extract production time from daily_forecasts.json @zara"""
        today = forecast_data.get("today", {})
        production_time = today.get("production_time", {})
        duration_seconds = production_time.get("duration_seconds", 0)

        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

class MaxPeakTodaySensor(AlwaysAvailableFileBasedMixin, SensorEntity):
    """Sensor for todays maximum power peak"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the max peak today sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        AlwaysAvailableFileBasedMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_max_peak_today"
        self._attr_translation_key = "max_peak_today"
        self._attr_native_unit_of_measurement = "W"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_device_class = SensorDeviceClass.POWER
        self._attr_icon = "mdi:lightning-bolt"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def native_value(self) -> float:
        """Return max peak or 0 if no data @zara"""
        return self._cached_value if self._cached_value is not None else 0.0

    def extract_value_from_file(self, forecast_data: dict) -> Optional[float]:
        """Extract max peak today from daily_forecasts.json @zara"""
        today = forecast_data.get("today", {})
        peak_today = today.get("peak_today", {})
        return peak_today.get("power_w", 0.0)

class MaxPeakAllTimeSensor(AlwaysAvailableFileBasedMixin, SensorEntity):
    """Sensor for all-time maximum power peak"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the max peak all time sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        self._cached_date: Optional[str] = None
        AlwaysAvailableFileBasedMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_max_peak_all_time"
        self._attr_translation_key = "max_peak_all_time"
        self._attr_native_unit_of_measurement = "W"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_device_class = SensorDeviceClass.POWER
        self._attr_icon = "mdi:lightning-bolt-circle"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def native_value(self) -> float:
        """Return max peak all time or 0 if no data @zara"""
        return self._cached_value if self._cached_value is not None else 0.0

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes @zara"""
        if self._cached_date:
            return {"date": self._cached_date}
        return {}

    def extract_value_from_file(self, forecast_data: dict) -> Optional[float]:
        """Extract max peak all time from daily_forecasts.json @zara"""
        statistics = forecast_data.get("statistics", {})
        all_time_peak = statistics.get("all_time_peak", {})

        self._cached_date = all_time_peak.get("date")
        return all_time_peak.get("power_w", 0.0)

class ForecastDayAfterTomorrowSensor(FileBasedSensorMixin, SensorEntity):
    """Sensor for day after tomorrows solar forecast"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the day after tomorrow sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        FileBasedSensorMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_forecast_day_after_tomorrow"
        self._attr_translation_key = "forecast_day_after_tomorrow"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:calendar-arrow-right"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> Optional[float]:
        """Extract day after tomorrow forecast from daily_forecasts.json @zara"""
        today = forecast_data.get("today", {})
        day_after = today.get("forecast_day_after_tomorrow", {})
        return day_after.get("prediction_kwh")

class MonthlyYieldSensor(StatisticsFileBasedMixin, SensorEntity):
    """Sensor for current months total yield"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the monthly yield sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        StatisticsFileBasedMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_monthly_yield"
        self._attr_translation_key = "monthly_yield"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:calendar-month"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> float:
        """Extract monthly yield from daily_forecasts.json @zara"""
        statistics = forecast_data.get("statistics", {})
        current_month = statistics.get("current_month", {})
        return current_month.get("yield_kwh", 0.0)

class MonthlyConsumptionSensor(StatisticsFileBasedMixin, SensorEntity):
    """Sensor for current months total consumption"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the monthly consumption sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        StatisticsFileBasedMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_monthly_consumption"
        self._attr_translation_key = "monthly_consumption"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:home-lightning-bolt"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> float:
        """Extract monthly consumption from daily_forecasts.json @zara"""
        statistics = forecast_data.get("statistics", {})
        current_month = statistics.get("current_month", {})
        return current_month.get("consumption_kwh", 0.0)

class WeeklyYieldSensor(StatisticsFileBasedMixin, SensorEntity):
    """Sensor for current weeks total yield"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the weekly yield sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        StatisticsFileBasedMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_weekly_yield"
        self._attr_translation_key = "weekly_yield"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:calendar-week"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> float:
        """Extract weekly yield from daily_forecasts.json @zara"""
        statistics = forecast_data.get("statistics", {})
        current_week = statistics.get("current_week", {})
        return current_week.get("yield_kwh", 0.0)

class WeeklyConsumptionSensor(StatisticsFileBasedMixin, SensorEntity):
    """Sensor for current weeks total consumption"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the weekly consumption sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        StatisticsFileBasedMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_weekly_consumption"
        self._attr_translation_key = "weekly_consumption"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:home-lightning-bolt-outline"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> float:
        """Extract weekly consumption from daily_forecasts.json @zara"""
        statistics = forecast_data.get("statistics", {})
        current_week = statistics.get("current_week", {})
        return current_week.get("consumption_kwh", 0.0)

class AverageYield7DaysSensor(FileBasedSensorMixin, SensorEntity):
    """Sensor for average daily yield over last 7 days"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average yield 7 days sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        FileBasedSensorMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_avg_yield_7d"
        self._attr_translation_key = "avg_yield_7d"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:chart-line"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> Optional[float]:
        """Extract average yield 7 days from daily_forecasts.json @zara"""
        statistics = forecast_data.get("statistics", {})
        last_7d = statistics.get("last_7_days", {})
        return last_7d.get("avg_yield_kwh")

class AverageYield30DaysSensor(FileBasedSensorMixin, SensorEntity):
    """Sensor for average daily yield over last 30 days"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average yield 30 days sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        FileBasedSensorMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_avg_yield_30d"
        self._attr_translation_key = "avg_yield_30d"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:chart-bar"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> Optional[float]:
        """Extract average yield 30 days from daily_forecasts.json @zara"""
        statistics = forecast_data.get("statistics", {})
        last_30d = statistics.get("last_30_days", {})
        return last_30d.get("avg_yield_kwh")

class AverageAutarkyMonthSensor(FileBasedSensorMixin, SensorEntity):
    """Sensor for average autarky for current month"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average autarky month sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        FileBasedSensorMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_avg_autarky_month"
        self._attr_translation_key = "avg_autarky_month"
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:leaf"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> Optional[float]:
        """Extract average autarky month from daily_forecasts.json @zara"""
        statistics = forecast_data.get("statistics", {})
        current_month = statistics.get("current_month", {})
        return current_month.get("avg_autarky")

class AverageAccuracy30DaysSensor(FileBasedSensorMixin, SensorEntity):
    """Sensor for average accuracy over last 30 days"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average accuracy 30 days sensor @zara"""
        self._coordinator = coordinator
        self.entry = entry
        FileBasedSensorMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_avg_accuracy_30d"
        self._attr_translation_key = "avg_accuracy_30d"
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:target"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> Optional[float]:
        """Extract average accuracy 30 days from daily_forecasts.json @zara"""
        statistics = forecast_data.get("statistics", {})
        last_30d = statistics.get("last_30_days", {})
        return last_30d.get("avg_accuracy")
