"""
Base Sensor Class

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
from typing import Optional

from homeassistant.components.sensor import (
    SensorEntity,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy, PERCENTAGE
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from ..const import DOMAIN, INTEGRATION_MODEL, SOFTWARE_VERSION, ML_VERSION
from ..coordinator import SolarForecastMLCoordinator
from .sensor_mixins import FileBasedSensorMixin, StatisticsFileBasedMixin, AlwaysAvailableFileBasedMixin

_LOGGER = logging.getLogger(__name__)


# --- Base Class for Coordinator-based Sensors in this file ---
class BaseSolarSensor(CoordinatorEntity, SensorEntity):
    """Base class for core sensors updated by the coordinator by @Zara"""

    _attr_has_entity_name = True

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the base sensor by @Zara"""
        super().__init__(coordinator)
        self.entry = entry
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Solar Forecast ML",
            manufacturer="Zara-Toorox",
            model=INTEGRATION_MODEL,
            sw_version=f"SW {SOFTWARE_VERSION} | ML {ML_VERSION}",
            configuration_url="https://github.com/Zara-Toorox/ha-solar-forecast-ml",
        )

    @property
    def available(self) -> bool:
        """Return if entity is available based on coordinator by @Zara"""
        # Check if coordinator has successfully run at least once
        return self.coordinator.last_update_success and self.coordinator.data is not None


# --- Core Sensors ---

class SolarForecastSensor(SensorEntity):
    """Sensor for todays or tomorrows solar forecast by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry, key: str):
        """Initialize the forecast sensor by @Zara"""
        self._coordinator = coordinator
        self.entry = entry
        self._key = key
        self._yield_entity = entry.data.get("solar_yield_today") if key == "remaining" else None
        self._cached_total_forecast: Optional[float] = None
        
        # Mapping zwischen Entity-Key und Data-Key
        self._key_mapping = {
            "remaining": {"data_key": "prediction_kwh", "translation_key": "today_forecast"},
            "tomorrow": {"data_key": "forecast_tomorrow", "translation_key": "tomorrow_forecast"}
        }
        
        # Validierung
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
        """Return if entity is available by @Zara"""
        return True

    @property
    def native_value(self) -> float:
        """Return the forecast value by @Zara"""
        if self._key == "tomorrow":
            # Tomorrow: Just return coordinator data
            if not self._coordinator.data:
                return 0.0
            return self._coordinator.data.get(self._data_key) or 0.0
        
        # "remaining" key: Read from daily_forecasts.json, then subtract yield
        if self._cached_total_forecast is None:
            return 0.0
        
        total_forecast = self._cached_total_forecast
        
        # Get current yield LIVE
        current_yield = 0.0
        if self._yield_entity:
            yield_state = self.hass.states.get(self._yield_entity)
            if yield_state and yield_state.state not in ["unavailable", "unknown", "none", None, ""]:
                try:
                    cleaned_state = str(yield_state.state).split(" ")[0].replace(",", ".")
                    current_yield = float(cleaned_state)
                except (ValueError, TypeError):
                    pass
        
        # Calculate remaining
        remaining = max(0.0, total_forecast - current_yield)
        return round(remaining, 2)

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass by @Zara"""
        await super().async_added_to_hass()
        
        # For "remaining": Load initial forecast from file
        if self._key == "remaining":
            await self._load_forecast_from_file()
        
        # Listen to coordinator updates
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        
        # For "remaining": Also listen to yield sensor changes
        if self._key == "remaining" and self._yield_entity:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, self._yield_entity, self._handle_yield_change
                )
            )
            # Listening to yield sensor for live updates (debug log removed)
        
        # Initial state
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates - reload file for remaining by @Zara"""
        if self._key == "remaining":
            self.hass.async_create_task(self._reload_forecast_and_update())
        else:
            self.async_write_ha_state()

    async def _reload_forecast_and_update(self) -> None:
        """Reload forecast from file and update state by @Zara"""
        await self._load_forecast_from_file()
        self.async_write_ha_state()

    async def _load_forecast_from_file(self) -> None:
        """Load todays LOCKED forecast from daily_forecastsjson by @Zara"""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                today = forecast_data.get("today", {})
                forecast_day = today.get("forecast_day", {})
                self._cached_total_forecast = forecast_day.get("prediction_kwh")
                # Loaded total forecast from file (debug log removed)
            else:
                self._cached_total_forecast = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load forecast from file: {e}")
            self._cached_total_forecast = None

    @callback
    def _handle_yield_change(self, event) -> None:
        """Handle yield sensor state changes only for remaining by @Zara"""
        self.async_write_ha_state()


class NextHourSensor(AlwaysAvailableFileBasedMixin, SensorEntity):
    """Sensor for the next hours solar forecast - reads from daily_forecastsjson by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the next hour sensor by @Zara"""
        self._coordinator = coordinator
        self.entry = entry
        AlwaysAvailableFileBasedMixin.__init__(self)

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
        """Return next hour forecast or 0.0 if no data by @Zara"""
        return self._cached_value if self._cached_value is not None else 0.0

    def extract_value_from_file(self, forecast_data: dict) -> Optional[float]:
        """Extract next hour forecast from daily_forecasts.json by @Zara"""
        today = forecast_data.get("today", {})
        next_hour = today.get("forecast_next_hour", {})
        return next_hour.get("prediction_kwh")


class PeakProductionHourSensor(AlwaysAvailableFileBasedMixin, SensorEntity):
    """Sensor indicating the forecasted best production hour from morning forecast by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the peak hour sensor by @Zara"""
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
        """Return peak hour or '--:--' if no data by @Zara"""
        return self._cached_value if self._cached_value is not None else "--:--"

    def extract_value_from_file(self, forecast_data: dict) -> Optional[str]:
        """Extract forecasted best hour from daily_forecasts.json by @Zara"""
        today = forecast_data.get("today", {})
        forecast_best_hour = today.get("forecast_best_hour", {})
        hour = forecast_best_hour.get("hour")
        return f"{hour:02d}:00" if hour is not None else None


class AverageYieldSensor(BaseSolarSensor):
    """Sensor for the calculated average monthly yield by @Zara"""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average yield sensor by @Zara"""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_average_yield"
        self._attr_translation_key = "average_yield"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:chart-line"

    @property
    def native_value(self) -> float | None:
        """Return the average monthly yield by @Zara"""
        value = getattr(self.coordinator, 'avg_month_yield', None)
        return value if value is not None and value > 0 else None


class AutarkySensor(SensorEntity):
    """Sensor for the calculated self-sufficiency Autarky with live updates by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the autarky sensor by @Zara"""
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
        """Return if entity is available by @Zara"""
        return (self._coordinator.last_update_success and 
                self._coordinator.data is not None and
                self.native_value is not None)

    @property
    def native_value(self) -> float | None:
        """Return the autarky percentage - LIVE calculation by @Zara"""
        # Try coordinator value first (updated during coordinator refresh)
        coord_value = getattr(self._coordinator, 'autarky_today', None)
        if coord_value is not None:
            return max(0.0, min(100.0, coord_value))

        # Calculate live with Grid Import if available
        if self._grid_import_entity and self._consumption_entity:
            try:
                grid_import_state = self.hass.states.get(self._grid_import_entity)
                consumption_state = self.hass.states.get(self._consumption_entity)

                if (grid_import_state and consumption_state and
                    grid_import_state.state not in ["unavailable", "unknown", "none", None, ""] and
                    consumption_state.state not in ["unavailable", "unknown", "none", None, ""]):

                    grid_import_val = float(str(grid_import_state.state).split()[0].replace(",", "."))
                    consumption_val = float(str(consumption_state.state).split()[0].replace(",", "."))

                    if consumption_val > 0:
                        # Autarky = (Consumption - Grid Import) / Consumption * 100
                        autarky = ((consumption_val - grid_import_val) / consumption_val) * 100.0
                        return max(0.0, min(100.0, autarky))
            except (ValueError, TypeError, AttributeError):
                pass

        # Fallback to old calculation (Yield / Consumption) if Grid Import not configured
        if self._yield_entity and self._consumption_entity:
            try:
                yield_state = self.hass.states.get(self._yield_entity)
                consumption_state = self.hass.states.get(self._consumption_entity)

                if (yield_state and consumption_state and
                    yield_state.state not in ["unavailable", "unknown", "none", None, ""] and
                    consumption_state.state not in ["unavailable", "unknown", "none", None, ""]):

                    yield_val = float(str(yield_state.state).split()[0].replace(",", "."))
                    consumption_val = float(str(consumption_state.state).split()[0].replace(",", "."))

                    if consumption_val > 0:
                        autarky = (yield_val / consumption_val) * 100.0
                        return max(0.0, min(100.0, autarky))
            except (ValueError, TypeError, AttributeError):
                pass

        return None

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass by @Zara"""
        await super().async_added_to_hass()

        # Listen to coordinator updates
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )

        # Listen to yield and consumption sensor changes for live updates
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
        # Listen to grid import/export sensor changes for live updates
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

        # Initial state
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates by @Zara"""
        self.async_write_ha_state()

    @callback
    def _handle_sensor_change(self, event) -> None:
        """Handle yield or consumption sensor state changes by @Zara"""
        self.async_write_ha_state()


class ExpectedDailyProductionSensor(FileBasedSensorMixin, SensorEntity):
    """Sensor for expected daily production morning snapshot by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the expected daily production sensor by @Zara"""
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
        """Extract expected daily production from daily_forecasts.json by @Zara"""
        today = forecast_data.get("today", {})
        forecast_day = today.get("forecast_day", {})
        return forecast_day.get("prediction_kwh")


# --- Production Time Sensor (Reads from daily_forecasts.json) ---

class ProductionTimeSensor(FileBasedSensorMixin, SensorEntity):
    """Sensor for production time today - reads directly from daily_forecastsjson by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the production time sensor by @Zara"""
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
        """Extract production time from daily_forecasts.json by @Zara"""
        today = forecast_data.get("today", {})
        production_time = today.get("production_time", {})
        duration_seconds = production_time.get("duration_seconds", 0)

        # Format seconds into HH:MM:SS
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


# --- New Statistical Sensors from daily_forecasts.json ---

class MaxPeakTodaySensor(AlwaysAvailableFileBasedMixin, SensorEntity):
    """Sensor for todays maximum power peak by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the max peak today sensor by @Zara"""
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
        """Return max peak or 0 if no data by @Zara"""
        return self._cached_value if self._cached_value is not None else 0.0

    def extract_value_from_file(self, forecast_data: dict) -> Optional[float]:
        """Extract max peak today from daily_forecasts.json by @Zara"""
        today = forecast_data.get("today", {})
        peak_today = today.get("peak_today", {})
        return peak_today.get("power_w", 0.0)


class MaxPeakAllTimeSensor(AlwaysAvailableFileBasedMixin, SensorEntity):
    """Sensor for all-time maximum power peak by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the max peak all time sensor by @Zara"""
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
        """Return max peak all time or 0 if no data by @Zara"""
        return self._cached_value if self._cached_value is not None else 0.0

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes by @Zara"""
        if self._cached_date:
            return {"date": self._cached_date}
        return {}

    def extract_value_from_file(self, forecast_data: dict) -> Optional[float]:
        """Extract max peak all time from daily_forecasts.json by @Zara"""
        statistics = forecast_data.get("statistics", {})
        all_time_peak = statistics.get("all_time_peak", {})
        # Store date for extra_state_attributes
        self._cached_date = all_time_peak.get("date")
        return all_time_peak.get("power_w", 0.0)


class ForecastDayAfterTomorrowSensor(FileBasedSensorMixin, SensorEntity):
    """Sensor for day after tomorrows solar forecast by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the day after tomorrow sensor by @Zara"""
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
        """Extract day after tomorrow forecast from daily_forecasts.json by @Zara"""
        today = forecast_data.get("today", {})
        day_after = today.get("forecast_day_after_tomorrow", {})
        return day_after.get("prediction_kwh")


class MonthlyYieldSensor(StatisticsFileBasedMixin, SensorEntity):
    """Sensor for current months total yield by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the monthly yield sensor by @Zara"""
        self._coordinator = coordinator
        self.entry = entry
        StatisticsFileBasedMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_monthly_yield"
        self._attr_translation_key = "monthly_yield"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL  # FIX: Not INCREASING - resets monthly
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:calendar-month"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> float:
        """Extract monthly yield from daily_forecasts.json by @Zara"""
        statistics = forecast_data.get("statistics", {})
        current_month = statistics.get("current_month", {})
        return current_month.get("yield_kwh", 0.0)


class MonthlyConsumptionSensor(StatisticsFileBasedMixin, SensorEntity):
    """Sensor for current months total consumption by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the monthly consumption sensor by @Zara"""
        self._coordinator = coordinator
        self.entry = entry
        StatisticsFileBasedMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_monthly_consumption"
        self._attr_translation_key = "monthly_consumption"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL  # FIX: Not INCREASING - resets monthly
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:home-lightning-bolt"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> float:
        """Extract monthly consumption from daily_forecasts.json by @Zara"""
        statistics = forecast_data.get("statistics", {})
        current_month = statistics.get("current_month", {})
        return current_month.get("consumption_kwh", 0.0)


class WeeklyYieldSensor(StatisticsFileBasedMixin, SensorEntity):
    """Sensor for current weeks total yield by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the weekly yield sensor by @Zara"""
        self._coordinator = coordinator
        self.entry = entry
        StatisticsFileBasedMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_weekly_yield"
        self._attr_translation_key = "weekly_yield"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL  # FIX: Not INCREASING - resets weekly
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:calendar-week"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> float:
        """Extract weekly yield from daily_forecasts.json by @Zara"""
        statistics = forecast_data.get("statistics", {})
        current_week = statistics.get("current_week", {})
        return current_week.get("yield_kwh", 0.0)


class WeeklyConsumptionSensor(StatisticsFileBasedMixin, SensorEntity):
    """Sensor for current weeks total consumption by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the weekly consumption sensor by @Zara"""
        self._coordinator = coordinator
        self.entry = entry
        StatisticsFileBasedMixin.__init__(self)

        self._attr_unique_id = f"{entry.entry_id}_ml_weekly_consumption"
        self._attr_translation_key = "weekly_consumption"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL  # FIX: Not INCREASING - resets weekly
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:home-lightning-bolt-outline"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    def extract_value_from_file(self, forecast_data: dict) -> float:
        """Extract weekly consumption from daily_forecasts.json by @Zara"""
        statistics = forecast_data.get("statistics", {})
        current_week = statistics.get("current_week", {})
        return current_week.get("consumption_kwh", 0.0)


class AverageYield7DaysSensor(FileBasedSensorMixin, SensorEntity):
    """Sensor for average daily yield over last 7 days by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average yield 7 days sensor by @Zara"""
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
        """Extract average yield 7 days from daily_forecasts.json by @Zara"""
        statistics = forecast_data.get("statistics", {})
        last_7d = statistics.get("last_7_days", {})
        return last_7d.get("avg_yield_kwh")


class AverageYield30DaysSensor(FileBasedSensorMixin, SensorEntity):
    """Sensor for average daily yield over last 30 days by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average yield 30 days sensor by @Zara"""
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
        """Extract average yield 30 days from daily_forecasts.json by @Zara"""
        statistics = forecast_data.get("statistics", {})
        last_30d = statistics.get("last_30_days", {})
        return last_30d.get("avg_yield_kwh")


class AverageAutarkyMonthSensor(FileBasedSensorMixin, SensorEntity):
    """Sensor for average autarky for current month by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average autarky month sensor by @Zara"""
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
        """Extract average autarky month from daily_forecasts.json by @Zara"""
        statistics = forecast_data.get("statistics", {})
        current_month = statistics.get("current_month", {})
        return current_month.get("avg_autarky")


class AverageAccuracy30DaysSensor(FileBasedSensorMixin, SensorEntity):
    """Sensor for average accuracy over last 30 days by @Zara"""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average accuracy 30 days sensor by @Zara"""
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
        """Extract average accuracy 30 days from daily_forecasts.json by @Zara"""
        statistics = forecast_data.get("statistics", {})
        last_30d = statistics.get("last_30_days", {})
        return last_30d.get("avg_accuracy")

