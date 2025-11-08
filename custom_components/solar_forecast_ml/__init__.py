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
    SERVICE_DEBUGGING_6AM_FORECAST,
    SERVICE_DEBUGGING_BEST_HOUR,
    SERVICE_DEBUGGING_TOMORROW_12PM,
    SERVICE_DEBUGGING_DAY_AFTER_TOMORROW_6AM,
    SERVICE_DEBUGGING_DAY_AFTER_TOMORROW_6PM,
    SERVICE_COLLECT_HOURLY_SAMPLE,
    SERVICE_NIGHT_CLEANUP,
    SERVICE_RUN_ALL_SCHEDULED_TASKS,
    CONF_BATTERY_ENABLED,
)

from .core.core_helpers import SafeDateTimeUtil as dt_util
from datetime import timedelta

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Solar Forecast ML integration legacy by Zara"""
    # Store empty dict for the domain
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Solar Forecast ML from a config entry by Zara"""
    # Deferring imports to avoid blocking the event loop during component loading
    from .coordinator import SolarForecastMLCoordinator
    from .core.core_dependency_handler import DependencyHandler
    from .services.service_notification import create_notification_service
    from .battery.battery_coordinator import BatteryCoordinator
    from .const import CONF_BATTERY_ENABLED

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

    # Create Solar Forecast coordinator
    coordinator = SolarForecastMLCoordinator(
        hass,
        entry,
        dependencies_ok=dependencies_ok
    )

    # Setup Solar coordinator
    setup_ok = await coordinator.async_setup()
    if not setup_ok:
        _LOGGER.error("Failed to setup Solar Forecast coordinator")
        return False

    # Perform initial data fetch
    await coordinator.async_config_entry_first_refresh()

    # Store Solar coordinator in hass.data
    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Setup Battery coordinator (separate, optional)
    battery_coordinator = None
    battery_enabled = entry.options.get(CONF_BATTERY_ENABLED, entry.data.get(CONF_BATTERY_ENABLED, False))

    if battery_enabled:
        try:
            _LOGGER.info("Battery management enabled - initializing BatteryCoordinator")
            battery_coordinator = BatteryCoordinator(hass, entry)

            battery_setup_ok = await battery_coordinator.async_setup()
            if battery_setup_ok:
                await battery_coordinator.async_config_entry_first_refresh()
                # Store Battery coordinator separately
                hass.data[DOMAIN][f"{entry.entry_id}_battery"] = battery_coordinator
                _LOGGER.info("BatteryCoordinator initialized successfully")
            else:
                _LOGGER.warning("BatteryCoordinator setup failed, continuing without battery features")
                battery_coordinator = None
        except Exception as e:
            _LOGGER.error(f"Error setting up BatteryCoordinator: {e}", exc_info=True)
            battery_coordinator = None

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

    # Final startup log with comprehensive status
    mode_str = "ML Mode (Full Features)" if dependencies_ok else "Fallback Mode (Rule-Based)"
    battery_str = "Enabled" if battery_coordinator else "Disabled"

    _LOGGER.info(
        "="*70 + "\n"
        "Solar Forecast ML v8 \"Sarpeidon\" 🌟 - Setup Complete! ✓\n"
        f"Mode: {mode_str} | Battery Management: {battery_str}\n"
        "\"The future is not set in stone, but with data we illuminate the path.\"\n"
        "Author: Zara-Toorox | Live long and prosper! 🖖\n" +
        "="*70
    )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry by Zara"""
    _LOGGER.info("Unloading Solar Forecast ML integration...")

    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        # Remove Solar coordinator from hass.data
        coordinator = hass.data[DOMAIN].pop(entry.entry_id)

        # Cleanup Solar coordinator
        if hasattr(coordinator, 'scheduled_tasks'):
            coordinator.scheduled_tasks.cancel_listeners()

        # Remove Battery coordinator if exists (separate cleanup)
        battery_key = f"{entry.entry_id}_battery"
        if battery_key in hass.data[DOMAIN]:
            battery_coordinator = hass.data[DOMAIN].pop(battery_key)
            # Battery coordinator cleanup (if needed in future)
            _LOGGER.info("BatteryCoordinator unloaded")

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
    """Register integration services by Zara"""
    
    async def handle_retrain_model(call: ServiceCall) -> None:
        """Handle force_retrain service call by Zara"""
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
        """Handle reset_model service call by Zara"""
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
        """Handle finalize_day service call by Zara"""
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
        """Handle move_to_history service call by Zara"""
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
        """Handle calculate_stats service call by Zara"""
        _LOGGER.info("Service called: calculate_stats (MANUAL)")
        try:
            # Check if statistics were recently calculated
            if coordinator._last_statistics_calculation:
                time_since = (dt_util.now() - coordinator._last_statistics_calculation).total_seconds()
                if time_since < 60:  # Less than 1 minute ago
                    _LOGGER.warning(
                        f"Statistics were calculated {time_since:.0f}s ago. "
                        f"Skipping to avoid excessive I/O. Wait at least 1 minute between calls."
                    )
                    return

            success = await coordinator.data_manager.calculate_statistics()
            if success:
                coordinator._last_statistics_calculation = dt_util.now()
                _LOGGER.info("Statistics calculation completed via service")
            else:
                _LOGGER.error("Statistics calculation failed")
        except Exception as e:
            _LOGGER.error(f"Error during calculate_stats service: {e}", exc_info=True)
    
    async def handle_run_all_day_end_tasks(call: ServiceCall) -> None:
        """Handle run_all_day_end_tasks service call by Zara"""
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
    
    async def handle_debugging_6am_forecast(call: ServiceCall) -> None:
        """Handle debugging_6am_forecast service call by Zara"""
        _LOGGER.info("Service called: debugging_6am_forecast (DEBUGGING)")
        try:
            await coordinator.set_expected_daily_production()
            _LOGGER.info("✓ 6 AM forecast logic executed: TODAY locked via service")
        except Exception as e:
            _LOGGER.error(f"✗ Error during debugging_6am_forecast service: {e}", exc_info=True)

    async def handle_debugging_best_hour(call: ServiceCall) -> None:
        """Handle debugging_best_hour service call by Zara"""
        _LOGGER.info("Service called: debugging_best_hour (DEBUGGING)")
        try:
            # Get required inputs (same as coordinator update)
            weather_service = coordinator.weather_service
            if not weather_service:
                _LOGGER.error("✗ Weather service not available - ABORTING")
                return

            # Fetch weather data
            current_weather = await weather_service.get_current_weather()
            hourly_forecast = await weather_service.get_processed_hourly_forecast()
            external_sensors = coordinator.sensor_collector.collect_all_sensor_data_dict()

            # Run forecast orchestrator (EXACT CODE from coordinator._async_update_data)
            forecast = await coordinator.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=coordinator.learned_correction_factor
            )

            # Extract best_hour (EXACT CODE from coordinator._save_forecasts_to_storage:668-684)
            best_hour = forecast.get("best_hour")
            best_hour_kwh = forecast.get("best_hour_kwh")

            if best_hour is not None and best_hour_kwh is not None:
                source = (
                    "ML"
                    if coordinator.forecast_orchestrator.ml_strategy
                    and coordinator.forecast_orchestrator.ml_strategy.is_available()
                    else "Weather"
                )

                await coordinator.data_manager.save_forecast_best_hour(
                    hour=best_hour,
                    prediction_kwh=best_hour_kwh,
                    source=source,
                )
                _LOGGER.info(f"✓ Best hour saved: {best_hour}:00 with {best_hour_kwh:.3f} kWh (source: {source})")
                coordinator.async_update_listeners()
            else:
                _LOGGER.warning("✗ Best hour not calculated in forecast - strategy may not support it")

        except Exception as e:
            _LOGGER.error(f"✗ Error during debugging_best_hour service: {e}", exc_info=True)

    async def handle_debugging_tomorrow_12pm(call: ServiceCall) -> None:
        """Handle debugging_tomorrow_12pm service call by Zara"""
        _LOGGER.info("Service called: debugging_tomorrow_12pm (DEBUGGING)")
        try:
            # Get current forecast data
            weather_service = coordinator.weather_service
            if not weather_service:
                _LOGGER.error("✗ Weather service not available - ABORTING")
                return

            # Fetch weather data
            current_weather = await weather_service.get_current_weather()
            hourly_forecast = await weather_service.get_processed_hourly_forecast()
            external_sensors = coordinator.sensor_collector.collect_all_sensor_data_dict()

            # Run forecast orchestrator
            forecast = await coordinator.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=coordinator.learned_correction_factor
            )

            tomorrow_kwh = forecast.get("tomorrow")
            if tomorrow_kwh is None:
                _LOGGER.error("✗ Tomorrow forecast not available - ABORTING")
                return

            # EXACT CODE from coordinator._save_forecasts_to_storage:686-696
            now_local = dt_util.now()
            tomorrow_date = now_local + timedelta(days=1)
            source = (
                "ML"
                if coordinator.forecast_orchestrator.ml_strategy
                and coordinator.forecast_orchestrator.ml_strategy.is_available()
                else "Weather"
            )

            await coordinator.data_manager.save_forecast_tomorrow(
                date=tomorrow_date,
                prediction_kwh=tomorrow_kwh,
                source=source,
                lock=True,  # LOCK at 12 PM
            )
            _LOGGER.info(f"✓ Tomorrow forecast LOCKED: {tomorrow_kwh:.2f} kWh")
            coordinator.async_update_listeners()

        except Exception as e:
            _LOGGER.error(f"✗ Error during debugging_tomorrow_12pm service: {e}", exc_info=True)

    async def handle_debugging_day_after_tomorrow_6am(call: ServiceCall) -> None:
        """Handle debugging_day_after_tomorrow_6am service call by Zara"""
        _LOGGER.info("Service called: debugging_day_after_tomorrow_6am (DEBUGGING)")
        try:
            # Get current forecast data
            weather_service = coordinator.weather_service
            if not weather_service:
                _LOGGER.error("✗ Weather service not available - ABORTING")
                return

            # Fetch weather data
            current_weather = await weather_service.get_current_weather()
            hourly_forecast = await weather_service.get_processed_hourly_forecast()
            external_sensors = coordinator.sensor_collector.collect_all_sensor_data_dict()

            # Run forecast orchestrator
            forecast = await coordinator.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=coordinator.learned_correction_factor
            )

            day_after_kwh = forecast.get("day_after_tomorrow")

            # ========== DEBUG RESULT ==========
            _LOGGER.info(f"\nFinal Forecast Results:")
            _LOGGER.info(f"  - Today: {forecast.get('today', 0):.2f} kWh")
            _LOGGER.info(f"  - Tomorrow: {forecast.get('tomorrow', 0):.2f} kWh")
            _LOGGER.info(f"  - Day After Tomorrow: {day_after_kwh if day_after_kwh else 0:.2f} kWh")
            _LOGGER.info(f"  - Method: {forecast.get('method', 'N/A')}")
            _LOGGER.info("=" * 80)
            # ========== END DEBUG RESULT ==========

            if day_after_kwh is None:
                _LOGGER.error("✗ Day after tomorrow forecast not available - ABORTING")
                return

            # EXACT CODE from coordinator._save_forecasts_to_storage:649-657
            # At 6 AM, day_after_tomorrow is saved UNLOCKED (lock=False by default)
            now_local = dt_util.now()
            day_after_date = now_local + timedelta(days=2)
            source = (
                "ML"
                if coordinator.forecast_orchestrator.ml_strategy
                and coordinator.forecast_orchestrator.ml_strategy.is_available()
                else "Weather"
            )

            await coordinator.data_manager.save_forecast_day_after(
                date=day_after_date,
                prediction_kwh=day_after_kwh,
                source=source,
                lock=False,  # UNLOCKED at 6 AM (this is the key difference!)
            )
            _LOGGER.info(f"✓ Day after tomorrow forecast saved UNLOCKED: {day_after_kwh:.2f} kWh")
            coordinator.async_update_listeners()

        except Exception as e:
            _LOGGER.error(f"✗ Error during debugging_day_after_tomorrow_6am service: {e}", exc_info=True)

    async def handle_debugging_day_after_tomorrow_6pm(call: ServiceCall) -> None:
        """Handle debugging_day_after_tomorrow_6pm service call by Zara"""
        _LOGGER.info("Service called: debugging_day_after_tomorrow_6pm (DEBUGGING)")
        try:
            # Get current forecast data
            weather_service = coordinator.weather_service
            if not weather_service:
                _LOGGER.error("✗ Weather service not available - ABORTING")
                return

            # Fetch weather data
            current_weather = await weather_service.get_current_weather()
            hourly_forecast = await weather_service.get_processed_hourly_forecast()
            external_sensors = coordinator.sensor_collector.collect_all_sensor_data_dict()

            # Run forecast orchestrator
            forecast = await coordinator.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=coordinator.learned_correction_factor
            )

            day_after_kwh = forecast.get("day_after_tomorrow")
            if day_after_kwh is None:
                _LOGGER.error("✗ Day after tomorrow forecast not available - ABORTING")
                return

            # EXACT CODE from coordinator._save_forecasts_to_storage:698-708
            now_local = dt_util.now()
            day_after_date = now_local + timedelta(days=2)
            source = (
                "ML"
                if coordinator.forecast_orchestrator.ml_strategy
                and coordinator.forecast_orchestrator.ml_strategy.is_available()
                else "Weather"
            )

            await coordinator.data_manager.save_forecast_day_after(
                date=day_after_date,
                prediction_kwh=day_after_kwh,
                source=source,
                lock=True,  # LOCK at 18 PM
            )
            _LOGGER.info(f"✓ Day after tomorrow forecast LOCKED: {day_after_kwh:.2f} kWh")
            coordinator.async_update_listeners()

        except Exception as e:
            _LOGGER.error(f"✗ Error during debugging_day_after_tomorrow_6pm service: {e}", exc_info=True)

    async def handle_night_cleanup(call: ServiceCall) -> None:
        """Handle night_cleanup service call by Zara"""
        _LOGGER.info("Service called: night_cleanup (MANUAL TEST)")
        try:
            if hasattr(coordinator, 'scheduled_tasks'):
                await coordinator.scheduled_tasks.scheduled_night_cleanup(None)
                _LOGGER.info("Night cleanup completed via service")
            else:
                _LOGGER.error("Scheduled tasks manager not available")
        except Exception as e:
            _LOGGER.error(f"Error during night_cleanup service: {e}", exc_info=True)

    async def handle_collect_hourly_sample(call: ServiceCall) -> None:
        """Handle collect_hourly_sample service call by Zara"""
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

    async def handle_run_all_scheduled_tasks(call: ServiceCall) -> None:
        """Handle run_all_scheduled_tasks service call - Execute all scheduled tasks in order"""
        _LOGGER.info("="*80)
        _LOGGER.info("SERVICE: run_all_scheduled_tasks - Starting complete task sequence")
        _LOGGER.info("="*80)

        try:
            # Get battery coordinator if available
            battery_enabled = entry.options.get(CONF_BATTERY_ENABLED, entry.data.get(CONF_BATTERY_ENABLED, False))
            battery_coordinator_key = f"{entry.entry_id}_battery"
            battery_coordinator = hass.data[DOMAIN].get(battery_coordinator_key) if battery_enabled else None

            # Task 1: 23:00 - Battery Daily Rollup
            if battery_coordinator:
                _LOGGER.info("TASK 1/10: [23:00] Battery Daily Rollup")
                try:
                    await battery_coordinator.async_daily_rollup()
                    _LOGGER.info("✓ Task 1/10 completed: Battery Daily Rollup")
                except Exception as e:
                    _LOGGER.error(f"✗ Task 1/10 failed: Battery Daily Rollup - {e}", exc_info=True)
            else:
                _LOGGER.info("⊘ Task 1/10 skipped: Battery Daily Rollup (battery disabled)")

            # Task 2: 23:15 - Fetch Electricity Prices
            if battery_coordinator and battery_coordinator.electricity_service:
                _LOGGER.info("TASK 2/10: [23:15] Fetch Electricity Prices (aWATTar)")
                try:
                    await battery_coordinator.async_refresh_prices()
                    _LOGGER.info("✓ Task 2/10 completed: Fetch Electricity Prices")
                except Exception as e:
                    _LOGGER.error(f"✗ Task 2/10 failed: Fetch Electricity Prices - {e}", exc_info=True)
            else:
                _LOGGER.info("⊘ Task 2/10 skipped: Fetch Electricity Prices (battery disabled)")

            # Task 3: 23:30 - Finalize Current Day
            _LOGGER.info("TASK 3/10: [23:30] Finalize Current Day")
            try:
                if hasattr(coordinator, 'scheduled_tasks'):
                    await coordinator.scheduled_tasks.finalize_day_task(None)
                    _LOGGER.info("✓ Task 3/10 completed: Finalize Current Day")
                else:
                    _LOGGER.warning("✗ Task 3/10 failed: Scheduled tasks manager not available")
            except Exception as e:
                _LOGGER.error(f"✗ Task 3/10 failed: Finalize Current Day - {e}", exc_info=True)

            # Task 4: 23:31 - Move to History
            _LOGGER.info("TASK 4/10: [23:31] Move to History")
            try:
                if hasattr(coordinator, 'scheduled_tasks'):
                    await coordinator.scheduled_tasks.move_to_history_task(None)
                    _LOGGER.info("✓ Task 4/10 completed: Move to History")
                else:
                    _LOGGER.warning("✗ Task 4/10 failed: Scheduled tasks manager not available")
            except Exception as e:
                _LOGGER.error(f"✗ Task 4/10 failed: Move to History - {e}", exc_info=True)

            # Task 5: 23:32 - Calculate Statistics
            _LOGGER.info("TASK 5/10: [23:32] Calculate Statistics")
            try:
                if hasattr(coordinator, 'scheduled_tasks'):
                    await coordinator.scheduled_tasks.calculate_stats_task(None)
                    _LOGGER.info("✓ Task 5/10 completed: Calculate Statistics")
                else:
                    _LOGGER.warning("✗ Task 5/10 failed: Scheduled tasks manager not available")
            except Exception as e:
                _LOGGER.error(f"✗ Task 5/10 failed: Calculate Statistics - {e}", exc_info=True)

            # Task 6: 23:45 - Reset Daily Data (skipped - no such task)
            _LOGGER.info("TASK 6/10: [23:45] Reset Daily Data")
            _LOGGER.info("⊘ Task 6/10 skipped: No reset_daily_data_task exists (handled by midnight)")

            # Task 7: 00:01 - Battery Midnight Reset
            _LOGGER.info("TASK 7/10: [00:01] Battery Midnight Reset")
            try:
                if hasattr(coordinator, 'scheduled_tasks'):
                    # Call with dt_util.now() instead of None to avoid strftime error
                    await coordinator.scheduled_tasks.reset_expected_production(dt_util.now())
                    _LOGGER.info("✓ Task 7/10 completed: Battery Midnight Reset")
                else:
                    _LOGGER.warning("✗ Task 7/10 failed: Scheduled tasks manager not available")
            except Exception as e:
                _LOGGER.error(f"✗ Task 7/10 failed: Battery Midnight Reset - {e}", exc_info=True)

            # Task 8: 02:00 - Night Cleanup (Remove duplicates & zero-production samples)
            _LOGGER.info("TASK 8/10: [02:00] Night Cleanup")
            try:
                if hasattr(coordinator, 'scheduled_tasks'):
                    await coordinator.scheduled_tasks.scheduled_night_cleanup(dt_util.now())
                    _LOGGER.info("✓ Task 8/10 completed: Night Cleanup")
                else:
                    _LOGGER.warning("✗ Task 8/10 failed: Scheduled tasks manager not available")
            except Exception as e:
                _LOGGER.error(f"✗ Task 8/10 failed: Night Cleanup - {e}", exc_info=True)

            # Task 9: 03:00 - ML Training (if ML mode available)
            _LOGGER.info("TASK 9/10: [03:00] ML Training")
            try:
                if hasattr(coordinator, 'ml_mode') and coordinator.ml_mode:
                    # Trigger ML retraining
                    await coordinator.ml_mode.train_model()
                    _LOGGER.info("✓ Task 9/10 completed: ML Training")
                else:
                    _LOGGER.info("⊘ Task 9/10 skipped: ML mode not available")
            except Exception as e:
                _LOGGER.error(f"✗ Task 9/10 failed: ML Training - {e}", exc_info=True)

            # Task 10: 06:00 - Morning Prediction
            _LOGGER.info("TASK 10/10: [06:00] Morning Prediction")
            try:
                if hasattr(coordinator, 'scheduled_tasks'):
                    # Call with dt_util.now() instead of None to avoid strftime error
                    await coordinator.scheduled_tasks.scheduled_morning_update(dt_util.now())
                    _LOGGER.info("✓ Task 10/10 completed: Morning Prediction")
                else:
                    _LOGGER.warning("✗ Task 10/10 failed: Scheduled tasks manager not available")
            except Exception as e:
                _LOGGER.error(f"✗ Task 10/10 failed: Morning Prediction - {e}", exc_info=True)

            _LOGGER.info("="*80)
            _LOGGER.info("SERVICE: run_all_scheduled_tasks - Complete!")
            _LOGGER.info("="*80)

        except Exception as e:
            _LOGGER.error(f"CRITICAL ERROR during run_all_scheduled_tasks: {e}", exc_info=True)

    # Register all services
    hass.services.async_register(DOMAIN, SERVICE_RETRAIN_MODEL, handle_retrain_model)
    hass.services.async_register(DOMAIN, SERVICE_RESET_LEARNING_DATA, handle_reset_model)
    hass.services.async_register(DOMAIN, SERVICE_FINALIZE_DAY, handle_finalize_day)
    hass.services.async_register(DOMAIN, SERVICE_MOVE_TO_HISTORY, handle_move_to_history)
    hass.services.async_register(DOMAIN, SERVICE_CALCULATE_STATS, handle_calculate_stats)
    hass.services.async_register(DOMAIN, SERVICE_RUN_ALL_DAY_END_TASKS, handle_run_all_day_end_tasks)
    hass.services.async_register(DOMAIN, SERVICE_DEBUGGING_6AM_FORECAST, handle_debugging_6am_forecast)
    hass.services.async_register(DOMAIN, SERVICE_DEBUGGING_BEST_HOUR, handle_debugging_best_hour)
    hass.services.async_register(DOMAIN, SERVICE_DEBUGGING_TOMORROW_12PM, handle_debugging_tomorrow_12pm)
    hass.services.async_register(DOMAIN, SERVICE_DEBUGGING_DAY_AFTER_TOMORROW_6AM, handle_debugging_day_after_tomorrow_6am)
    hass.services.async_register(DOMAIN, SERVICE_DEBUGGING_DAY_AFTER_TOMORROW_6PM, handle_debugging_day_after_tomorrow_6pm)
    hass.services.async_register(DOMAIN, SERVICE_COLLECT_HOURLY_SAMPLE, handle_collect_hourly_sample)
    hass.services.async_register(DOMAIN, SERVICE_NIGHT_CLEANUP, handle_night_cleanup)
    hass.services.async_register(DOMAIN, SERVICE_RUN_ALL_SCHEDULED_TASKS, handle_run_all_scheduled_tasks)

    _LOGGER.info("All services registered successfully")


def _async_unregister_services(hass: HomeAssistant) -> None:
    """Unregister integration services by Zara"""
    services = [
        SERVICE_RETRAIN_MODEL,
        SERVICE_RESET_LEARNING_DATA,
        SERVICE_FINALIZE_DAY,
        SERVICE_MOVE_TO_HISTORY,
        SERVICE_CALCULATE_STATS,
        SERVICE_RUN_ALL_DAY_END_TASKS,
        SERVICE_DEBUGGING_6AM_FORECAST,
        SERVICE_DEBUGGING_BEST_HOUR,
        SERVICE_DEBUGGING_TOMORROW_12PM,
        SERVICE_DEBUGGING_DAY_AFTER_TOMORROW_6AM,
        SERVICE_DEBUGGING_DAY_AFTER_TOMORROW_6PM,
        SERVICE_COLLECT_HOURLY_SAMPLE,
        SERVICE_NIGHT_CLEANUP,
        SERVICE_RUN_ALL_SCHEDULED_TASKS,
    ]

    for service in services:
        if hass.services.has_service(DOMAIN, service):
            hass.services.async_remove(DOMAIN, service)

    _LOGGER.info("All services unregistered")
