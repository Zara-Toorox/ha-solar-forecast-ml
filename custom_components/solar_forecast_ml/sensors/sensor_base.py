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

_LOGGER = logging.getLogger(__name__)


# --- Base Class for Coordinator-based Sensors in this file ---
class BaseSolarSensor(CoordinatorEntity, SensorEntity):
    """Base class for core sensors updated by the coordinator."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the base sensor."""
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
        """Return if entity is available based on coordinator."""
        # Check if coordinator has successfully run at least once
        return self.coordinator.last_update_success and self.coordinator.data is not None


# --- Core Sensors ---

class SolarForecastSensor(SensorEntity):
    """
    Sensor for today's or tomorrow's solar forecast.
    "remaining" type reads from daily_forecasts.json and subtracts live yield.
    """

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry, key: str):
        """Initialize the forecast sensor."""
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
        
        self._attr_unique_id = f"{entry.entry_id}_{key}"
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
        """Return if entity is available."""
        return True

    @property
    def native_value(self) -> float:
        """
        Return the forecast value.
        'remaining': LOCKED forecast from daily_forecasts.json minus current yield
        'tomorrow': From coordinator data
        """
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
        """Run when entity about to be added to hass."""
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
            _LOGGER.debug(f"Remaining forecast sensor listening to {self._yield_entity}")
        
        # Initial state
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates - reload file for 'remaining'."""
        if self._key == "remaining":
            self.hass.async_create_task(self._reload_forecast_and_update())
        else:
            self.async_write_ha_state()

    async def _reload_forecast_and_update(self) -> None:
        """Reload forecast from file and update state."""
        await self._load_forecast_from_file()
        self.async_write_ha_state()

    async def _load_forecast_from_file(self) -> None:
        """Load today's LOCKED forecast from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                today = forecast_data.get("today", {})
                forecast_day = today.get("forecast_day", {})
                self._cached_total_forecast = forecast_day.get("prediction_kwh")
                _LOGGER.debug(f"Loaded total forecast from file: {self._cached_total_forecast} kWh")
            else:
                self._cached_total_forecast = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load forecast from file: {e}")
            self._cached_total_forecast = None

    @callback
    def _handle_yield_change(self, event) -> None:
        """Handle yield sensor state changes (only for 'remaining')."""
        self.async_write_ha_state()


class NextHourSensor(SensorEntity):
    """Sensor for the next hour's solar forecast - reads from daily_forecasts.json."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the next hour sensor."""
        self._coordinator = coordinator
        self.entry = entry
        self._cached_value: Optional[float] = None
        
        self._attr_unique_id = f"{entry.entry_id}_next_hour"
        self._attr_translation_key = "next_hour_forecast"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:clock-fast"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Sensor is available if we have a forecast set and it's active."""
        return self._cached_value is not None

    @property
    def native_value(self) -> float | None:
        """Return the next hour forecast value."""
        return self._cached_value

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        
        # Load initial value from file
        await self._load_from_file()
        
        # Listen to coordinator updates
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        
        # Initial state
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates - reload from file."""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value from file and update state."""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load next hour forecast from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                today = forecast_data.get("today", {})
                next_hour = today.get("forecast_next_hour", {})
                
                # Get prediction value
                self._cached_value = next_hour.get("prediction_kwh")
                    
                _LOGGER.debug(f"Loaded next hour forecast: {self._cached_value}")
            else:
                self._cached_value = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load next hour forecast from file: {e}")
            self._cached_value = None


class PeakProductionHourSensor(BaseSolarSensor):
    """Sensor indicating the estimated peak production hour."""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the peak hour sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_peak_production_hour"
        self._attr_translation_key = "peak_production_hour"
        self._attr_icon = "mdi:solar-power-variant-outline"

    @property
    def native_value(self) -> str | None:
        """Return the estimated peak production hour."""
        value = getattr(self.coordinator, 'peak_production_time_today', None)
        return value if value != "Calculating..." else None


class AverageYieldSensor(BaseSolarSensor):
    """Sensor for the calculated average monthly yield."""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average yield sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_average_yield"
        self._attr_translation_key = "average_yield"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:chart-line"

    @property
    def native_value(self) -> float | None:
        """Return the average monthly yield."""
        value = getattr(self.coordinator, 'avg_month_yield', None)
        return value if value is not None and value > 0 else None


class AutarkySensor(SensorEntity):
    """Sensor for the calculated self-sufficiency (Autarky) with live updates."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the autarky sensor."""
        self._coordinator = coordinator
        self.entry = entry
        self._yield_entity = entry.data.get("solar_yield_today")
        self._consumption_entity = entry.data.get("total_consumption_today")
        
        self._attr_unique_id = f"{entry.entry_id}_autarky"
        self._attr_translation_key = "self_sufficiency"
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:home-lightning-bolt-outline"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return (self._coordinator.last_update_success and 
                self._coordinator.data is not None and
                self.native_value is not None)

    @property
    def native_value(self) -> float | None:
        """Return the autarky percentage - LIVE calculation."""
        # Try coordinator value first (updated during coordinator refresh)
        coord_value = getattr(self._coordinator, 'autarky_today', None)
        if coord_value is not None:
            return max(0.0, min(100.0, coord_value))
        
        # Fallback: Calculate live
        if not (self._yield_entity and self._consumption_entity):
            return None
            
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
        """Run when entity about to be added to hass."""
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
        
        # Initial state
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates."""
        self.async_write_ha_state()

    @callback
    def _handle_sensor_change(self, event) -> None:
        """Handle yield or consumption sensor state changes."""
        self.async_write_ha_state()


class ExpectedDailyProductionSensor(SensorEntity):
    """
    Sensor for expected daily production (morning snapshot).
    Reads DIRECTLY from daily_forecasts.json current.prediction_kwh.
    """

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the expected daily production sensor."""
        self._coordinator = coordinator
        self.entry = entry
        self._cached_value: Optional[float] = None
        
        self._attr_unique_id = f"{entry.entry_id}_expected_daily_production"
        self._attr_translation_key = "expected_daily_production"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:solar-power-variant"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Sensor is available if we have a forecast set."""
        return self._cached_value is not None

    @property
    def native_value(self) -> float | None:
        """Return the cached expected daily production value."""
        return self._cached_value

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        
        # Load initial value from file
        await self._load_from_file()
        
        # Listen to coordinator updates
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        
        # Initial state
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates - reload from file."""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value from file and update state."""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load expected daily production from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                today = forecast_data.get("today", {})
                forecast_day = today.get("forecast_day", {})
                self._cached_value = forecast_day.get("prediction_kwh")
                _LOGGER.debug(f"Loaded expected daily production from file: {self._cached_value}")
            else:
                self._cached_value = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load expected daily production from file: {e}")
            self._cached_value = None


# --- Production Time Sensor (Reads from daily_forecasts.json) ---

class ProductionTimeSensor(SensorEntity):
    """
    Sensor for production time today - reads directly from daily_forecasts.json.
    100% restart-safe as data is persisted in file.
    """

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the production time sensor."""
        self.entry = entry
        self._coordinator = coordinator
        self._cached_value: Optional[str] = None

        self._attr_unique_id = f"{entry.entry_id}_production_time"
        self._attr_translation_key = "production_time"
        self._attr_icon = "mdi:timer-outline"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def native_value(self) -> str:
        """Return current production time from file."""
        return self._cached_value if self._cached_value else "00:00:00"

    @property
    def available(self) -> bool:
        """Sensor is available if we have data."""
        return self._cached_value is not None

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        
        # Load initial value from file
        await self._load_from_file()
        
        # Listen to coordinator updates (triggers re-loading)
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        
        # Initial state
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates - trigger reload and state update."""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value from file and update state."""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load production time from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                today = forecast_data.get("today", {})
                production_time = today.get("production_time", {})
                
                duration_seconds = production_time.get("duration_seconds", 0)
                
                # Format seconds into HH:MM:SS
                hours, remainder = divmod(duration_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                self._cached_value = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

                _LOGGER.debug(f"Loaded production time: {self._cached_value}")
            else:
                self._cached_value = "00:00:00"
        except Exception as e:
            _LOGGER.warning(f"Failed to load production time from file: {e}")
            self._cached_value = "00:00:00"


# --- New Statistical Sensors from daily_forecasts.json ---

class MaxPeakTodaySensor(SensorEntity):
    """Sensor for today's maximum power peak."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the max peak today sensor."""
        self._coordinator = coordinator
        self.entry = entry
        self._cached_value: Optional[float] = None
        
        self._attr_unique_id = f"{entry.entry_id}_max_peak_today"
        self._attr_translation_key = "max_peak_today"
        self._attr_native_unit_of_measurement = "W"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_device_class = SensorDeviceClass.POWER
        self._attr_icon = "mdi:lightning-bolt"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Sensor is available if we have a value."""
        return self._cached_value is not None and self._cached_value > 0

    @property
    def native_value(self) -> float | None:
        """Return today's max peak power."""
        return self._cached_value

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        await self._load_from_file()
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates."""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value from file and update state."""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load max peak today from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                today = forecast_data.get("today", {})
                peak_today = today.get("peak_today", {})
                self._cached_value = peak_today.get("power_w", 0.0)
                _LOGGER.debug(f"Loaded max peak today: {self._cached_value}W")
            else:
                self._cached_value = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load max peak today: {e}")
            self._cached_value = None


class MaxPeakAllTimeSensor(SensorEntity):
    """Sensor for all-time maximum power peak."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the max peak all time sensor."""
        self._coordinator = coordinator
        self.entry = entry
        self._cached_value: Optional[float] = None
        self._cached_date: Optional[str] = None
        
        self._attr_unique_id = f"{entry.entry_id}_max_peak_all_time"
        self._attr_translation_key = "max_peak_all_time"
        self._attr_native_unit_of_measurement = "W"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_device_class = SensorDeviceClass.POWER
        self._attr_icon = "mdi:lightning-bolt-circle"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Sensor is available if we have a value."""
        return self._cached_value is not None and self._cached_value > 0

    @property
    def native_value(self) -> float | None:
        """Return all-time max peak power."""
        return self._cached_value

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes."""
        if self._cached_date:
            return {"date": self._cached_date}
        return {}

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        await self._load_from_file()
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates."""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value from file and update state."""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load max peak all time from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                statistics = forecast_data.get("statistics", {})
                all_time_peak = statistics.get("all_time_peak", {})
                self._cached_value = all_time_peak.get("power_w", 0.0)
                self._cached_date = all_time_peak.get("date")
                _LOGGER.debug(f"Loaded max peak all time: {self._cached_value}W on {self._cached_date}")
            else:
                self._cached_value = None
                self._cached_date = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load max peak all time: {e}")
            self._cached_value = None
            self._cached_date = None


class ForecastDayAfterTomorrowSensor(SensorEntity):
    """Sensor for day after tomorrow's solar forecast."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the day after tomorrow sensor."""
        self._coordinator = coordinator
        self.entry = entry
        self._cached_value: Optional[float] = None
        
        self._attr_unique_id = f"{entry.entry_id}_forecast_day_after_tomorrow"
        self._attr_translation_key = "forecast_day_after_tomorrow"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:calendar-arrow-right"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Sensor is available if we have a forecast."""
        return self._cached_value is not None

    @property
    def native_value(self) -> float | None:
        """Return day after tomorrow forecast."""
        return self._cached_value

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        await self._load_from_file()
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates."""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value from file and update state."""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load day after tomorrow forecast from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                today = forecast_data.get("today", {})
                day_after = today.get("forecast_day_after_tomorrow", {})
                self._cached_value = day_after.get("prediction_kwh")
                _LOGGER.debug(f"Loaded day after tomorrow forecast: {self._cached_value}")
            else:
                self._cached_value = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load day after tomorrow forecast: {e}")
            self._cached_value = None


class MonthlyYieldSensor(SensorEntity):
    """Sensor for current month's total yield."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the monthly yield sensor."""
        self._coordinator = coordinator
        self.entry = entry
        self._cached_value: Optional[float] = None
        
        self._attr_unique_id = f"{entry.entry_id}_monthly_yield"
        self._attr_translation_key = "monthly_yield"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL_INCREASING
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:calendar-month"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Sensor is available if we have a value."""
        return True  # Always available, will show 0.0 if no data

    @property
    def native_value(self) -> float | None:
        """Return monthly yield."""
        return self._cached_value

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        await self._load_from_file()
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates."""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value from file and update state."""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load monthly yield from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                statistics = forecast_data.get("statistics", {})
                current_month = statistics.get("current_month", {})
                self._cached_value = current_month.get("yield_kwh", 0.0)
                _LOGGER.debug(f"Loaded monthly yield: {self._cached_value}")
            else:
                self._cached_value = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load monthly yield: {e}")
            self._cached_value = None


class MonthlyConsumptionSensor(SensorEntity):
    """Sensor for current month's total consumption."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the monthly consumption sensor."""
        self._coordinator = coordinator
        self.entry = entry
        self._cached_value: Optional[float] = None
        
        self._attr_unique_id = f"{entry.entry_id}_monthly_consumption"
        self._attr_translation_key = "monthly_consumption"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL_INCREASING
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:home-lightning-bolt"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Sensor is available if we have a value."""
        return True  # Always available, will show 0.0 if no data

    @property
    def native_value(self) -> float | None:
        """Return monthly consumption."""
        return self._cached_value

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        await self._load_from_file()
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates."""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value from file and update state."""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load monthly consumption from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                statistics = forecast_data.get("statistics", {})
                current_month = statistics.get("current_month", {})
                self._cached_value = current_month.get("consumption_kwh", 0.0)
                _LOGGER.debug(f"Loaded monthly consumption: {self._cached_value}")
            else:
                self._cached_value = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load monthly consumption: {e}")
            self._cached_value = None


class WeeklyYieldSensor(SensorEntity):
    """Sensor for current week's total yield."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the weekly yield sensor."""
        self._coordinator = coordinator
        self.entry = entry
        self._cached_value: Optional[float] = None
        
        self._attr_unique_id = f"{entry.entry_id}_weekly_yield"
        self._attr_translation_key = "weekly_yield"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL_INCREASING
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:calendar-week"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Sensor is available if we have a value."""
        return True  # Always available, will show 0.0 if no data

    @property
    def native_value(self) -> float | None:
        """Return weekly yield."""
        return self._cached_value

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        await self._load_from_file()
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates."""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value from file and update state."""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load weekly yield from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                statistics = forecast_data.get("statistics", {})
                current_week = statistics.get("current_week", {})
                self._cached_value = current_week.get("yield_kwh", 0.0)
                _LOGGER.debug(f"Loaded weekly yield: {self._cached_value}")
            else:
                self._cached_value = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load weekly yield: {e}")
            self._cached_value = None


class WeeklyConsumptionSensor(SensorEntity):
    """Sensor for current week's total consumption."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the weekly consumption sensor."""
        self._coordinator = coordinator
        self.entry = entry
        self._cached_value: Optional[float] = None
        
        self._attr_unique_id = f"{entry.entry_id}_weekly_consumption"
        self._attr_translation_key = "weekly_consumption"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL_INCREASING
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:home-lightning-bolt-outline"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Sensor is available if we have a value."""
        return True  # Always available, will show 0.0 if no data

    @property
    def native_value(self) -> float | None:
        """Return weekly consumption."""
        return self._cached_value

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        await self._load_from_file()
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates."""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value from file and update state."""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load weekly consumption from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                statistics = forecast_data.get("statistics", {})
                current_week = statistics.get("current_week", {})
                self._cached_value = current_week.get("consumption_kwh", 0.0)
                _LOGGER.debug(f"Loaded weekly consumption: {self._cached_value}")
            else:
                self._cached_value = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load weekly consumption: {e}")
            self._cached_value = None


class AverageYield7DaysSensor(SensorEntity):
    """Sensor for average daily yield over last 7 days."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average yield 7 days sensor."""
        self._coordinator = coordinator
        self.entry = entry
        self._cached_value: Optional[float] = None
        
        self._attr_unique_id = f"{entry.entry_id}_avg_yield_7d"
        self._attr_translation_key = "avg_yield_7d"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:chart-line"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Sensor is available if we have a value."""
        return self._cached_value is not None

    @property
    def native_value(self) -> float | None:
        """Return average daily yield for last 7 days."""
        return self._cached_value

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        await self._load_from_file()
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates."""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value from file and update state."""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load average yield 7 days from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                statistics = forecast_data.get("statistics", {})
                last_7d = statistics.get("last_7_days", {})
                self._cached_value = last_7d.get("avg_yield_kwh")
                _LOGGER.debug(f"Loaded avg yield 7d: {self._cached_value}")
            else:
                self._cached_value = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load avg yield 7d: {e}")
            self._cached_value = None


class AverageYield30DaysSensor(SensorEntity):
    """Sensor for average daily yield over last 30 days."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average yield 30 days sensor."""
        self._coordinator = coordinator
        self.entry = entry
        self._cached_value: Optional[float] = None
        
        self._attr_unique_id = f"{entry.entry_id}_avg_yield_30d"
        self._attr_translation_key = "avg_yield_30d"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:chart-bar"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Sensor is available if we have a value."""
        return self._cached_value is not None

    @property
    def native_value(self) -> float | None:
        """Return average daily yield for last 30 days."""
        return self._cached_value

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        await self._load_from_file()
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates."""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value from file and update state."""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load average yield 30 days from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                statistics = forecast_data.get("statistics", {})
                last_30d = statistics.get("last_30_days", {})
                self._cached_value = last_30d.get("avg_yield_kwh")
                _LOGGER.debug(f"Loaded avg yield 30d: {self._cached_value}")
            else:
                self._cached_value = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load avg yield 30d: {e}")
            self._cached_value = None


class AverageAutarkyMonthSensor(SensorEntity):
    """Sensor for average autarky for current month."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average autarky month sensor."""
        self._coordinator = coordinator
        self.entry = entry
        self._cached_value: Optional[float] = None
        
        self._attr_unique_id = f"{entry.entry_id}_avg_autarky_month"
        self._attr_translation_key = "avg_autarky_month"
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:leaf"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Sensor is available if we have a value."""
        return self._cached_value is not None

    @property
    def native_value(self) -> float | None:
        """Return average autarky for current month."""
        return self._cached_value

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        await self._load_from_file()
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates."""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value from file and update state."""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load average autarky month from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                statistics = forecast_data.get("statistics", {})
                current_month = statistics.get("current_month", {})
                self._cached_value = current_month.get("avg_autarky")
                _LOGGER.debug(f"Loaded avg autarky month: {self._cached_value}")
            else:
                self._cached_value = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load avg autarky month: {e}")
            self._cached_value = None


class AverageAccuracy30DaysSensor(SensorEntity):
    """Sensor for average accuracy over last 30 days."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the average accuracy 30 days sensor."""
        self._coordinator = coordinator
        self.entry = entry
        self._cached_value: Optional[float] = None
        
        self._attr_unique_id = f"{entry.entry_id}_avg_accuracy_30d"
        self._attr_translation_key = "avg_accuracy_30d"
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:target"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    @property
    def available(self) -> bool:
        """Sensor is available if we have a value."""
        return self._cached_value is not None

    @property
    def native_value(self) -> float | None:
        """Return average accuracy for last 30 days."""
        return self._cached_value

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        await self._load_from_file()
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates."""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value from file and update state."""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load average accuracy 30 days from daily_forecasts.json."""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                statistics = forecast_data.get("statistics", {})
                last_30d = statistics.get("last_30_days", {})
                self._cached_value = last_30d.get("avg_accuracy")
                _LOGGER.debug(f"Loaded avg accuracy 30d: {self._cached_value}")
            else:
                self._cached_value = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load avg accuracy 30d: {e}")
            self._cached_value = None

