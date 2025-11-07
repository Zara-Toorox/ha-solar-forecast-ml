"""
Button Platform for Solar Forecast ML Integration

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

from __future__ import annotations

import logging
from typing import Any # Keep Any if needed, otherwise remove

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

# Use constants for domain and potentially device info
from .const import DOMAIN, INTEGRATION_MODEL, SOFTWARE_VERSION, ML_VERSION
from .core.core_helpers import SafeDateTimeUtil as dt_util # Keep if used, remove if not
from .coordinator import SolarForecastMLCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the button entities from a config entry by Zara"""
    coordinator: SolarForecastMLCoordinator = hass.data[DOMAIN][entry.entry_id]

    # Define the base device info reused by buttons
    base_device_info = DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name="Solar Forecast ML", # Main device name
        manufacturer="Zara-Toorox",
        model=INTEGRATION_MODEL,
        sw_version=f"SW {SOFTWARE_VERSION} | ML {ML_VERSION}",
        entry_type="service", # Correct entry type
        configuration_url="https://github.com/Zara-Toorox/ha-solar-forecast-ml",
    )


    buttons = [
        ManualForecastButton(coordinator, entry, base_device_info),
        ManualLearningButton(coordinator, entry, base_device_info),
        # === [FIX] Backfill Button Removed ===
        # MLBackfillButton(coordinator, entry),
        # === END REMOVAL ===
    ]

    async_add_entities(buttons)
    # Update log message to reflect available buttons
    _LOGGER.info("Solar Forecast ML Buttons successfully set up: Manual Forecast, Manual Learning.")


# =============================================================================
# 1. Manual Forecast Button
# =============================================================================
class ManualForecastButton(ButtonEntity):
    """Button entity to trigger a manual forecast refresh via the coordinator by Zara"""

    _attr_has_entity_name = True
    _attr_translation_key = "manual_forecast"
    _attr_icon = "mdi:refresh-circle"

    def __init__(
            self,
            coordinator: SolarForecastMLCoordinator,
            entry: ConfigEntry,
            device_info: DeviceInfo # Accept base device info
        ) -> None:
        """Initialize the button by Zara"""
        # super().__init__() # Not strictly needed if async_press is defined

        self.coordinator = coordinator
        self._entry_id = entry.entry_id

        # Unique ID for this specific button entity
        self._attr_unique_id = f"{self._entry_id}_manual_forecast"
        # Link this entity to the main device
        self._attr_device_info = device_info

    async def async_press(self) -> None:
        """Handle the button press Only create new forecast if NOT locked by Zara"""
        _LOGGER.info("Manual Forecast button pressed - checking lock status...")
        
        # Check if today's forecast is already locked
        current_day = await self.coordinator.data_manager.get_current_day_forecast()
        
        if current_day and current_day.get("locked"):
            _LOGGER.warning(
                f"Expected daily production already locked for today "
                f"({current_day.get('date')}) with {current_day.get('prediction_kwh')} kWh. "
                f"Button has no effect after 06:00 lock."
            )
            return  # Exit - button does NOTHING if locked
        
        # NOT locked yet - create new forecast
        _LOGGER.info("No locked forecast found - creating new forecast...")
        
        # Force refresh
        await self.coordinator.force_refresh_with_weather_update()
        
        # Update expected_daily_production if data is available
        if self.coordinator.data and "forecast_today" in self.coordinator.data:
            today_value = self.coordinator.data.get("forecast_today")
            self.coordinator.expected_daily_production = today_value
            
            # Save to persistent storage (BOTH systems)
            # OLD system (coordinator_state.json)
            await self.coordinator.data_manager.save_expected_daily_production(today_value)
            
            # NEW system (daily_forecasts.json) - lock forecast with manual source
            await self.coordinator.data_manager.save_daily_forecast(
                prediction_kwh=today_value,
                source="manual_button"
            )
            
            _LOGGER.info(
                f"Expected daily production set to {today_value:.2f} kWh "
                f"(saved to persistent storage, source: manual_button)"
            )
            
            # Force sensor update
            self.coordinator.async_update_listeners()
            
            _LOGGER.info(
                f"Coordinator data after refresh: "
                f"today={today_value}, "
                f"tomorrow={self.coordinator.data.get('forecast_tomorrow')}, "
                f"method={self.coordinator.data.get('_forecast_method')}"
            )
        else:
            _LOGGER.error("ERROR: Coordinator data is None or missing forecast_today after force refresh!")
        
        _LOGGER.debug("Force refresh with weather update completed.")


# =============================================================================
# 2. Manual Learning Button
# =============================================================================
class ManualLearningButton(ButtonEntity):
    """Button entity to trigger manual ML model training by Zara"""

    _attr_has_entity_name = True
    _attr_translation_key = "manual_learning"
    _attr_icon = "mdi:brain"

    def __init__(
            self,
            coordinator: SolarForecastMLCoordinator,
            entry: ConfigEntry,
            device_info: DeviceInfo # Accept base device info
        ) -> None:
        """Initialize the button by Zara"""
        # super().__init__() # Not strictly needed

        self.coordinator = coordinator
        self._entry_id = entry.entry_id

        # Unique ID for this specific button entity
        self._attr_unique_id = f"{self._entry_id}_manual_learning"
        # Link this entity to the main device
        self._attr_device_info = device_info

    async def async_press(self) -> None:
        """Handle the button press trigger ML model training by Zara"""
        _LOGGER.info("Manual Learning button pressed - starting training process.")

        # FIXED: Direct access to ml_predictor instead of service_manager
        ml_predictor = self.coordinator.ml_predictor

        if not ml_predictor:
            _LOGGER.error("ML Predictor service is not available. Cannot start manual training.")
            # Optionally, show a persistent notification to the user
            # await self.hass.services.async_call(...)
            return

        try:
            # Trigger the training method on the predictor
            _LOGGER.info("Calling ml_predictor.train_model()...")
            result = await ml_predictor.train_model()
            timestamp = dt_util.now() # LOCAL time for consistency

            # Log result and potentially update coordinator state
            if result and result.success:
                accuracy_str = f"{result.accuracy * 100:.1f}%" if result.accuracy is not None else "N/A"
                _LOGGER.info(f"Manual ML training completed successfully. Accuracy: {accuracy_str}")
                # Update coordinator state if necessary (e.g., last training time)
                # This might already be handled within train_model or via coordinator listeners
                # Example:
                # self.coordinator.last_successful_learning = timestamp
                # self.coordinator.model_accuracy = result.accuracy
                # self.coordinator.async_set_updated_data(self.coordinator.data) # Trigger sensor updates

                # If the coordinator has a specific callback method:
                if hasattr(self.coordinator, "on_ml_training_complete"):
                     self.coordinator.on_ml_training_complete(
                         timestamp=timestamp,
                         accuracy=result.accuracy
                     )

            elif result:
                _LOGGER.error(f"Manual ML training failed: {result.error_message}")
            else:
                 _LOGGER.error("Manual ML training failed with an unexpected result structure.")

        except Exception as e:
            _LOGGER.error(f"An unexpected error occurred during manual ML training: {e}", exc_info=True)


# =============================================================================
# 3. Backfill Training Button (REMOVED)
# =============================================================================
# class MLBackfillButton(ButtonEntity): ...
# (Entire class and references removed)