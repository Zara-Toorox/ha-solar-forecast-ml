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
    """Set up the Solar Forecast ML integration legacy by @Zara"""
    # Store empty dict for the domain
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Solar Forecast ML from a config entry by @Zara"""
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
    """Unload a config entry by @Zara"""
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
    """Register integration services using Service Registry by @Zara"""
    from .services.service_registry import ServiceRegistry

    registry = ServiceRegistry(hass, entry, coordinator)
    await registry.async_register_all_services()

    # Store registry for cleanup
    hass.data[DOMAIN]["service_registry"] = registry


def _async_unregister_services(hass: HomeAssistant) -> None:
    """Unregister integration services using Service Registry by @Zara"""
    registry = hass.data[DOMAIN].get("service_registry")
    if registry:
        registry.unregister_all_services()
    else:
        _LOGGER.warning("Service registry not found for cleanup")
