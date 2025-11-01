"""
Core Sensor platform for Solar Forecast ML Integration.
Contains the primary user-facing sensors.

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
from ..production.tracker import ProductionTimeCalculator

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

class SolarForecastSensor(BaseSolarSensor):
    """Sensor for today's or tomorrow's solar forecast."""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry, key: str):
        """Initialize the forecast sensor."""
        super().__init__(coordinator, entry)
        self._key = key
        
        # Mapping zwischen Entity-Key und Data-Key
        self._key_mapping = {
            "remaining": {"data_key": "forecast_today", "translation_key": "today_forecast"},
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
        self._attr_state_class = SensorStateClass.TOTAL # Represents total energy for the day
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:solar-power"

    @property
    def native_value(self) -> float | None:
        """Return the forecast value."""
        # Access coordinator data safely using mapped data key
        if self.coordinator.data:
            return self.coordinator.data.get(self._data_key) # Returns None if key missing
        return None # Return None if data is not available


class NextHourSensor(BaseSolarSensor):
    """Sensor for the next hour's solar forecast."""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the next hour sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_next_hour"
        self._attr_translation_key = "next_hour_forecast"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        # --- HIER DIE KORREKTUR ---
        self._attr_state_class = SensorStateClass.TOTAL # Use TOTAL for energy amount in a period
        # --- ENDE KORREKTUR ---
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:clock-fast"

    @property
    def native_value(self) -> float | None:
        """Return the next hour forecast value."""
        value = getattr(self.coordinator, 'next_hour_pred', None)
        return round(value, 3) if value is not None else None # Use more precision for hourly


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


class AutarkySensor(BaseSolarSensor):
    """Sensor for the calculated self-sufficiency (Autarky)."""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the autarky sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_autarky"
        self._attr_translation_key = "self_sufficiency"
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:home-lightning-bolt-outline"

    @property
    def native_value(self) -> float | None:
        """Return the autarky percentage."""
        value = getattr(self.coordinator, 'autarky_today', None)
        return max(0.0, min(100.0, value)) if value is not None else None


class ExpectedDailyProductionSensor(BaseSolarSensor):
    """Sensor for expected daily production (6 AM snapshot)."""

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the expected daily production sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_expected_daily_production"
        self._attr_translation_key = "expected_daily_production"
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_icon = "mdi:solar-power-variant"

    @property
    def native_value(self) -> float | None:
        """Return the frozen daily production value from 6 AM."""
        return getattr(self.coordinator, 'expected_daily_production', None)


# --- Production Time Sensor (Live Update, NOT Coordinator based) ---
class ProductionTimeSensor(SensorEntity):
    """Sensor for live tracking of production time today."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize the live production time sensor."""
        self.entry = entry
        self._coordinator = coordinator
        self._production_calculator: Optional[ProductionTimeCalculator] = None
        self._power_entity_id: Optional[str] = None
        self._current_value: str = "Initializing..."

        self._attr_unique_id = f"{entry.entry_id}_production_time"
        self._attr_translation_key = "production_time"
        self._attr_icon = "mdi:timer-outline"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )
        self._attr_should_poll = False

    @property
    def native_value(self) -> str:
        """Return the current production time string."""
        return self._current_value

    @property
    def available(self) -> bool:
        """Sensor is available if the calculator is running and has a valid value."""
        return self._current_value not in ["Not available", "Error", "Initializing..."]


    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()

        calculator = getattr(self._coordinator, 'production_time_calculator', None)
        if isinstance(calculator, ProductionTimeCalculator):
            self._production_calculator = calculator
            self._power_entity_id = calculator.power_entity
            _LOGGER.debug(f"ProductionTimeSensor linked to calculator tracking '{self._power_entity_id or 'None'}'")

            self._update_internal_state()

            if self._power_entity_id:
                self.async_on_remove(
                    async_track_state_change_event(
                        self.hass, self._power_entity_id, self._handle_update
                    )
                )
                _LOGGER.debug(f"ProductionTimeSensor listening to state changes of '{self._power_entity_id}'")
            else:
                 _LOGGER.warning("ProductionTimeSensor cannot listen for updates: Power entity ID not found in calculator.")
                 self._current_value = "Not available"

        else:
            _LOGGER.error("ProductionTimeCalculator instance not found or invalid in coordinator! Cannot track production time.")
            self._current_value = "Error"

        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updates from the coordinator (mainly for midnight reset)."""
        _LOGGER.debug("ProductionTimeSensor received coordinator update.")
        self._update_internal_state()
        self.async_write_ha_state()


    @callback
    async def _handle_update(self, event=None) -> None:
        """Handle state updates from the power sensor."""
        entity_id = event.data.get("entity_id") if event and event.data else "coordinator"
        _LOGGER.debug(f"ProductionTimeSensor update triggered by '{entity_id}'.")
        self._update_internal_state()
        self.async_write_ha_state()

    @callback
    def _update_internal_state(self) -> None:
        """Fetch the latest value from the calculator and store it."""
        if self._production_calculator:
            try:
                new_value = self._production_calculator.get_production_time()
                self._current_value = new_value

            except Exception as e:
                if self._current_value != "Error":
                     _LOGGER.error(f"Error getting production time from calculator: {e}", exc_info=True)
                     self._current_value = "Error"
        else:
             if self._current_value != "Error":
                  _LOGGER.error("ProductionTimeCalculator instance lost!")
                  self._current_value = "Error"
