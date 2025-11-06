"""
Solar Forecast ML Integration - Main Entry Point

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
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.typing import ConfigType

from .const import (
    DOMAIN,
    PLATFORMS,
    SERVICE_RETRAIN_MODEL,
    SERVICE_RESET_LEARNING_DATA,
    SERVICE_FINALIZE_DAY,
    SERVICE_MOVE_TO_HISTORY,
    SERVICE_CALCULATE_STATS,
    SERVICE_RUN_ALL_DAY_END_TASKS,
    SERVICE_TRIGGER_6AM_FORECAST,
    SERVICE_FORECAST_DAY_AFTER_TOMORROW,
    SERVICE_COLLECT_HOURLY_SAMPLE,
)

from .core.core_helpers import SafeDateTimeUtil as dt_util
from datetime import timedelta

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Solar Forecast ML integration (legacy)."""
    # Store empty dict for the domain
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Solar Forecast ML from a config entry."""
    # Deferring imports to avoid blocking the event loop during component loading
    from .coordinator import SolarForecastMLCoordinator
    from .core.core_dependency_handler import DependencyHandler
    from .services.service_notification import create_notification_service

    # Check dependencies
    dependency_handler = DependencyHandler()
    dependencies_ok = await dependency_handler.check_dependencies(hass)

    if not dependencies_ok:
        _LOGGER.warning("Some ML dependencies are missing. ML features will be disabled.")

    # Initialize data structure
    hass.data.setdefault(DOMAIN, {})

    # Create and initialize notification service
    notification_service = await create_notification_service(hass, entry)
    if notification_service:
        hass.data[DOMAIN]["notification_service"] = notification_service
        _LOGGER.debug("NotificationService created and stored in hass.data")
    else:
        _LOGGER.warning("NotificationService could not be created")

    # Create coordinator
    coordinator = SolarForecastMLCoordinator(
        hass,
        entry,
        dependencies_ok=dependencies_ok
    )

    # Setup coordinator
    setup_ok = await coordinator.async_setup()
    if not setup_ok:
        _LOGGER.error("Failed to setup coordinator")
        return False

    # Perform initial data fetch
    await coordinator.async_config_entry_first_refresh()

    # Store coordinator in hass.data
    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Forward entry setup to platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services
    await _async_register_services(hass, entry, coordinator)

    # Show startup notification after successful setup
    if notification_service:
        try:
            installed_packages = []
            missing_packages = []

            if dependencies_ok:
                # Get list of installed ML packages
                installed_packages = dependency_handler.get_installed_packages()
            else:
                # Get list of missing packages
                missing_packages = dependency_handler.get_missing_packages()

            await notification_service.show_startup_success(
                ml_mode=dependencies_ok,
                installed_packages=installed_packages,
                missing_packages=missing_packages
            )
            _LOGGER.debug("Startup notification triggered")
        except Exception as e:
            _LOGGER.warning(f"Failed to show startup notification: {e}", exc_info=True)

    _LOGGER.info("Solar Forecast ML integration setup completed successfully")
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.info("Unloading Solar Forecast ML integration...")
    
    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    if unload_ok:
        # Remove coordinator from hass.data
        coordinator = hass.data[DOMAIN].pop(entry.entry_id)
        
        # Cleanup coordinator
        if hasattr(coordinator, 'scheduled_tasks'):
            await coordinator.scheduled_tasks.async_cleanup()
        
        # Unregister services only if this is the last entry
        if not hass.data[DOMAIN]:
            _async_unregister_services(hass)
    
    _LOGGER.info("Solar Forecast ML integration unloaded successfully")
    return unload_ok


async def _async_register_services(
    hass: HomeAssistant,
    entry: ConfigEntry,
    coordinator: "SolarForecastMLCoordinator"
) -> None:
    """Register integration services."""
    
    async def handle_retrain_model(call: ServiceCall) -> None:
        """Handle force_retrain service call."""
        _LOGGER.info("Service called: force_retrain")
        try:
            if coordinator.ml_predictor:
                success = await coordinator.ml_predictor.force_training()
                if success:
                    _LOGGER.info("ML model retrained successfully via service")
                else:
                    _LOGGER.error("ML model retraining failed")
            else:
                _LOGGER.warning("ML predictor not available")
        except Exception as e:
            _LOGGER.error(f"Error during force_retrain service: {e}", exc_info=True)
    
    async def handle_reset_model(call: ServiceCall) -> None:
        """Handle reset_model service call."""
        _LOGGER.info("Service called: reset_model")
        try:
            if coordinator.ml_predictor:
                success = await coordinator.ml_predictor.reset_model()
                if success:
                    _LOGGER.info("ML model reset successfully via service")
                else:
                    _LOGGER.error("ML model reset failed")
            else:
                _LOGGER.warning("ML predictor not available")
        except Exception as e:
            _LOGGER.error(f"Error during reset_model service: {e}", exc_info=True)
    
    async def handle_finalize_day(call: ServiceCall) -> None:
        """Handle finalize_day service call."""
        _LOGGER.info("Service called: finalize_day (EMERGENCY)")
        try:
            if hasattr(coordinator, 'scheduled_tasks'):
                await coordinator.scheduled_tasks.finalize_day_task(None)
                _LOGGER.info("Day finalization completed via service")
            else:
                _LOGGER.error("Scheduled tasks manager not available")
        except Exception as e:
            _LOGGER.error(f"Error during finalize_day service: {e}", exc_info=True)
    
    async def handle_move_to_history(call: ServiceCall) -> None:
        """Handle move_to_history service call."""
        _LOGGER.info("Service called: move_to_history (EMERGENCY)")
        try:
            if hasattr(coordinator, 'scheduled_tasks'):
                await coordinator.scheduled_tasks.move_to_history_task(None)
                _LOGGER.info("Move to history completed via service")
            else:
                _LOGGER.error("Scheduled tasks manager not available")
        except Exception as e:
            _LOGGER.error(f"Error during move_to_history service: {e}", exc_info=True)
    
    async def handle_calculate_stats(call: ServiceCall) -> None:
        """Handle calculate_stats service call."""
        _LOGGER.info("Service called: calculate_stats (EMERGENCY)")
        try:
            success = await coordinator.data_manager.calculate_statistics()
            if success:
                _LOGGER.info("Statistics calculation completed via service")
            else:
                _LOGGER.error("Statistics calculation failed")
        except Exception as e:
            _LOGGER.error(f"Error during calculate_stats service: {e}", exc_info=True)
    
    async def handle_run_all_day_end_tasks(call: ServiceCall) -> None:
        """Handle run_all_day_end_tasks service call."""
        _LOGGER.info("Service called: run_all_day_end_tasks (EMERGENCY)")
        try:
            # Execute complete day-end sequence
            if hasattr(coordinator, 'scheduled_tasks'):
                await coordinator.scheduled_tasks.finalize_day_task(None)
                _LOGGER.info("1/3: Day finalization completed")
                
                await coordinator.scheduled_tasks.move_to_history_task(None)
                _LOGGER.info("2/3: Move to history completed")
                
                await coordinator.data_manager.calculate_statistics()
                _LOGGER.info("3/3: Statistics calculation completed")
                
                _LOGGER.info("All day-end tasks completed successfully via service")
            else:
                _LOGGER.error("Scheduled tasks manager not available")
        except Exception as e:
            _LOGGER.error(f"Error during run_all_day_end_tasks service: {e}", exc_info=True)
    
    async def handle_trigger_6am_forecast(call: ServiceCall) -> None:
        """Handle trigger_6am_forecast service call (DEBUGGING)."""
        _LOGGER.info("Service called: trigger_6am_forecast (DEBUGGING)")
        try:
            await coordinator.set_expected_daily_production()
            _LOGGER.info("6 AM forecast lock triggered successfully via service")
        except Exception as e:
            _LOGGER.error(f"Error during trigger_6am_forecast service: {e}", exc_info=True)

    async def handle_forecast_day_after_tomorrow(call: ServiceCall) -> None:
        """Handle forecast_day_after_tomorrow service call."""
        _LOGGER.info("Service called: forecast_day_after_tomorrow")
        try:
            await coordinator.forecast_day_after_tomorrow()
            _LOGGER.info("Forecast for day after tomorrow triggered successfully via service")
        except Exception as e:
            _LOGGER.error(f"Error during forecast_day_after_tomorrow service: {e}", exc_info=True)

    async def handle_collect_hourly_sample(call: ServiceCall) -> None:
        """Handle collect_hourly_sample service call."""
        _LOGGER.info("Service called: collect_hourly_sample")
        try:
            if not coordinator.ml_predictor:
                _LOGGER.error("ML predictor not available - cannot collect hourly sample")
                return

            if not coordinator.ml_predictor.sample_collector:
                _LOGGER.error("Sample collector not available - cannot collect hourly sample")
                return

            # Check if power entity is configured
            if not coordinator.ml_predictor.sample_collector.power_entity:
                _LOGGER.error(
                    "Power entity not configured in sample collector - cannot collect hourly sample. "
                    "Please check your integration configuration."
                )
                return

            # Collect sample for the previous full hour
            hour_to_collect_dt = dt_util.now() - timedelta(hours=1)
            _LOGGER.info(f"Collecting hourly sample for hour {hour_to_collect_dt.hour} (local)")

            await coordinator.ml_predictor.sample_collector.collect_sample(hour_to_collect_dt)

            _LOGGER.info(f"Hourly sample collection completed for hour {hour_to_collect_dt.hour} via service")

        except Exception as e:
            _LOGGER.error(f"Error during collect_hourly_sample service: {e}", exc_info=True)
    
    # Register all services
    hass.services.async_register(DOMAIN, SERVICE_RETRAIN_MODEL, handle_retrain_model)
    hass.services.async_register(DOMAIN, SERVICE_RESET_LEARNING_DATA, handle_reset_model)
    hass.services.async_register(DOMAIN, SERVICE_FINALIZE_DAY, handle_finalize_day)
    hass.services.async_register(DOMAIN, SERVICE_MOVE_TO_HISTORY, handle_move_to_history)
    hass.services.async_register(DOMAIN, SERVICE_CALCULATE_STATS, handle_calculate_stats)
    hass.services.async_register(DOMAIN, SERVICE_RUN_ALL_DAY_END_TASKS, handle_run_all_day_end_tasks)
    hass.services.async_register(DOMAIN, SERVICE_TRIGGER_6AM_FORECAST, handle_trigger_6am_forecast)
    hass.services.async_register(DOMAIN, SERVICE_FORECAST_DAY_AFTER_TOMORROW, handle_forecast_day_after_tomorrow)
    hass.services.async_register(DOMAIN, SERVICE_COLLECT_HOURLY_SAMPLE, handle_collect_hourly_sample)
    
    _LOGGER.info("All services registered successfully")


def _async_unregister_services(hass: HomeAssistant) -> None:
    """Unregister integration services."""
    services = [
        SERVICE_RETRAIN_MODEL,
        SERVICE_RESET_LEARNING_DATA,
        SERVICE_FINALIZE_DAY,
        SERVICE_MOVE_TO_HISTORY,
        SERVICE_CALCULATE_STATS,
        SERVICE_RUN_ALL_DAY_END_TASKS,
        SERVICE_TRIGGER_6AM_FORECAST,
        SERVICE_FORECAST_DAY_AFTER_TOMORROW,
        SERVICE_COLLECT_HOURLY_SAMPLE,
    ]
    
    for service in services:
        if hass.services.has_service(DOMAIN, service):
            hass.services.async_remove(DOMAIN, service)
    
    _LOGGER.info("All services unregistered")
