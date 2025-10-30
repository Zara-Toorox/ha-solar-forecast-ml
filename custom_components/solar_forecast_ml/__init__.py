"""
The Solar Forecast ML integration init file.

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
from datetime import timedelta
from pathlib import Path

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType
from homeassistant.components import recorder # Import recorder

from .const import (
    DOMAIN,
    CONF_SOLAR_CAPACITY, # Keep this one
    # CONF_PANEL_EFFICIENCY, # Removed
    # CONF_AZIMUTH,          # Removed
    # CONF_TILT,             # Removed
    CONF_WEATHER_ENTITY,
    CONF_WEATHER_PREFERENCE, # Keep if migration uses it
    CONF_FALLBACK_ENTITY,    # Keep if migration uses it
    DEFAULT_SOLAR_CAPACITY,
    # DEFAULT_PANEL_EFFICIENCY, # Removed
    # DEFAULT_AZIMUTH,          # Removed
    # DEFAULT_TILT,             # Removed
    WEATHER_PREFERENCE_GENERIC, # Keep if migration uses it
    WEATHER_FALLBACK_DEFAULT,   # Keep if migration uses it
    DATA_DIR,
    CONF_POWER_ENTITY,
    CONF_SOLAR_YIELD_TODAY,
    SOFTWARE_VERSION, # Use constant for version
    # --- NEU (Block 3): Service-Namen importieren ---
    SERVICE_RETRAIN_MODEL,
    SERVICE_RESET_LEARNING_DATA
    # --- ENDE NEU ---
)


_LOGGER = logging.getLogger(__name__)

# Define platforms (SENSOR and BUTTON)
PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.BUTTON]


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    version = config_entry.version

    _LOGGER.info(f"Checking migration for Solar Forecast ML from version {version}")

    # No specific migration needed from 6.x to 6.2.1 for now,
    # but keep the structure for future needs.
    # if version < X:
    #    ...
    #    hass.config_entries.async_update_entry(config_entry, data=new_data, version=X)
    #    _LOGGER.info("Migration to version X completed")

    # If no migration was performed, still return True
    return True


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Solar Forecast ML component."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Solar Forecast ML from a config entry."""
    _LOGGER.info(f"Setting up Solar Forecast ML integration - Version {SOFTWARE_VERSION}")

    # === LAZY IMPORTS ===
    from .coordinator import SolarForecastMLCoordinator
    from .notification_service import create_notification_service
    from .dependency_handler import DependencyHandler
    # === END LAZY IMPORTS ===

    _LOGGER.info("Initializing NotificationService...")
    notification_service = await create_notification_service(hass)

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN]["notification_service"] = notification_service
    _LOGGER.info("NotificationService initialized")

    # === START RECORDER PRE-FLIGHT CHECK ===
    _LOGGER.info("Performing Pre-Flight Check: Verifying recorder settings...")
    try:
        rec_instance = recorder.get_instance(hass)
        if not rec_instance or not rec_instance.is_running:
            _LOGGER.warning("Recorder component is not running. Historical data and ML training might be affected.")
            # Continue setup, but log warning. Training might still work if recorder starts later.
        else:
            # Check essential entities if recorder is running
            entities_to_check = {
                "Solar Daily Yield": entry.data.get(CONF_SOLAR_YIELD_TODAY),
                "Power Sensor": entry.data.get(CONF_POWER_ENTITY),
                "Weather Sensor": entry.data.get(CONF_WEATHER_ENTITY),
            }

            unrecorded_entities = []

            for name, entity_id in entities_to_check.items():
                if not entity_id:
                    _LOGGER.debug(f"Pre-Flight Check: {name} is not configured, skipping check.")
                    continue

                # Use the recorder instance's entity_filter to check if recorded
                if not rec_instance.entity_filter(entity_id):
                    _LOGGER.warning(
                        f"Pre-Flight Check FAILED: Entity '{entity_id}' ({name}) "
                        f"is NOT being recorded (filtered out by 'recorder' config). "
                        f"ML training and some calculations (like peak time) will fail."
                    )
                    unrecorded_entities.append(f"'{entity_id}' ({name})")

            if unrecorded_entities:
                entity_list_str = ", ".join(unrecorded_entities)
                # Ensure notification_service was created before calling methods on it
                if notification_service and hasattr(notification_service, 'show_permanent_warning'):
                    await notification_service.show_permanent_warning(
                        notification_id="recorder_preflight_check_failed",
                        title="Solar Forecast ML - Recorder Error",
                        message=(
                            "The following essential entities are NOT recorded by Home Assistant: "
                            f"{entity_list_str}. \n\n"
                            "The ML model CANNOT be trained and some features (like historical peak time) "
                            "will not work without this data. "
                            "Please check your 'recorder:' configuration in 'configuration.yaml' "
                            "and ensure these entities are NOT excluded (e.g., via 'exclude' rules)."
                        )
                    )
                else:
                    _LOGGER.error("Notification service not available to show recorder warning.")

                _LOGGER.error(
                    "Pre-Flight Check FAILED: ML Training requires recorder data "
                    f"for {entity_list_str}. A permanent notification has been created (if service available)."
                )
            else:
                _LOGGER.info("Pre-Flight Check OK: All critical entities configured are being recorded.")

    except AttributeError as e:
        # Catch errors if recorder methods change or instance is unexpected
        _LOGGER.error(f"Error during Pre-Flight Recorder Check (AttributeError): {e}", exc_info=True)
    except Exception as e:
        _LOGGER.error(f"Unexpected error during Pre-Flight Recorder Check: {e}", exc_info=True)
    # === END RECORDER PRE-FLIGHT CHECK ===

    _LOGGER.info("Checking dependencies (async)...")
    dependency_handler = DependencyHandler()

    # Pass hass to async dependency check
    dependencies_ok = await dependency_handler.check_dependencies(hass)

    # Fetch status after check for accurate reporting
    dep_status = await dependency_handler.get_dependency_status(hass)
    _LOGGER.info(f"Dependencies Status: {dep_status}")

    # Create the coordinator
    coordinator = SolarForecastMLCoordinator(
        hass=hass,
        entry=entry,
        dependencies_ok=dependencies_ok
    )
    
    # --- NEU (Block 3): Koordinator-Referenz im ServiceManager setzen ---
    if coordinator.service_manager:
        coordinator.service_manager.coordinator = coordinator
        _LOGGER.debug("Coordinator reference set in ServiceManager.")
    # --- ENDE NEU ---

    _LOGGER.info("Initializing DataManager...")
    try:
        # DataManager needs to be initialized before services that might use it
        await coordinator.data_manager.initialize()
        # Construct path correctly
        data_dir_path = hass.config.path(DATA_DIR)
        _LOGGER.info(f"DataManager initialized - Directory: {data_dir_path}")
    except Exception as err:
        _LOGGER.error(f"Failed to initialize DataManager: {err}")
        # Ensure notification_service exists before using
        if notification_service and hasattr(notification_service, 'show_installation_error'):
            await notification_service.show_installation_error(
                f"DataManager initialization failed: {err}"
            )
        else:
             _LOGGER.error("Notification service not available to show DataManager init error.")
        return False # Stop setup if DataManager fails

    _LOGGER.info("Starting initial data refresh (this includes service initialization)...")
    # This call now handles waiting for weather and initializing services correctly
    await coordinator.async_config_entry_first_refresh()

    # Store coordinator instance
    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Forward entry setup for SENSOR and BUTTON platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Add listener for options updates
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    
    # --- NEU (Block 3): Service-Registrierung ---
    _LOGGER.info("Registering integration services...")
    try:
        if coordinator.service_manager:
            # Service: force_retrain
            hass.services.async_register(
                DOMAIN,
                SERVICE_RETRAIN_MODEL, # "force_retrain"
                coordinator.service_manager._handle_service_retrain
            )
            
            # Service: reset_model
            hass.services.async_register(
                DOMAIN,
                SERVICE_RESET_LEARNING_DATA, # "reset_model"
                coordinator.service_manager._handle_service_reset
            )
            _LOGGER.info("Services 'force_retrain' and 'reset_model' registered successfully.")
        else:
            _LOGGER.error("ServiceManager not found on coordinator, cannot register services.")
            
    except Exception as e:
        _LOGGER.error(f"Failed to register services: {e}", exc_info=True)
    # --- ENDE NEU ---

    _LOGGER.info("Showing startup notification...")
    try:
        installed_deps = []
        missing_deps = []

        for pkg, status in dep_status.items():
            if status.get("satisfied", False): # Check 'satisfied' key
                version = status.get("version", "unknown")
                installed_deps.append(f"{pkg} ({version})")
            else:
                missing_deps.append(pkg)

        # Ensure notification_service exists
        if notification_service and hasattr(notification_service, 'show_startup_success'):
            if dependencies_ok:
                _LOGGER.info("All dependencies satisfied - ML Mode active")
                await notification_service.show_startup_success(
                    ml_mode=True,
                    installed_packages=installed_deps
                )
            else:
                _LOGGER.warning(f"Missing dependencies: {', '.join(missing_deps)} - Fallback Mode")
                await notification_service.show_startup_success(
                    ml_mode=False,
                    installed_packages=installed_deps,
                    missing_packages=missing_deps
                )
        else:
             _LOGGER.error("Notification service not available to show startup status.")


    except Exception as err:
        _LOGGER.warning(f"Failed to show startup notification: {err}")

    _LOGGER.info("Solar Forecast ML integration setup complete")
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.info("Unloading Solar Forecast ML integration")

    # Unload platforms (SENSOR, BUTTON)
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        # Retrieve and remove coordinator
        coordinator = hass.data[DOMAIN].pop(entry.entry_id, None)

        # Perform cleanup if coordinator exists
        if coordinator:
            # Stop production time tracking if calculator exists
            # Check the correct attribute name after refactoring
            tracker_instance = getattr(coordinator, 'production_time_calculator', None)
            if tracker_instance:
                tracker_instance.stop_tracking()
                _LOGGER.info("ProductionTimeCalculator listeners cleaned up.")

            # Cleanup ServiceManager resources (which cleans up predictor)
            if hasattr(coordinator, 'service_manager') and coordinator.service_manager:
                 await coordinator.service_manager.cleanup()
                 _LOGGER.info("ServiceManager resources cleaned up.")

            # Cleanup DataManager resources (like executor) - Moved after service cleanup
            if hasattr(coordinator, 'data_manager') and coordinator.data_manager:
                await coordinator.data_manager.cleanup()
                _LOGGER.info("DataManager resources cleaned up.")

            # Cancel scheduled tasks listeners
            if hasattr(coordinator, 'scheduled_tasks') and coordinator.scheduled_tasks:
                 coordinator.scheduled_tasks.cancel_listeners()
                 _LOGGER.info("Scheduled task listeners cancelled.")
                 
            # --- NEU (Block 3): Service-Deregistrierung ---
            _LOGGER.info("Deregistering integration services...")
            try:
                hass.services.async_remove(DOMAIN, SERVICE_RETRAIN_MODEL)
                hass.services.async_remove(DOMAIN, SERVICE_RESET_LEARNING_DATA)
                _LOGGER.info("Services deregistered successfully.")
            except Exception as e:
                _LOGGER.warning(f"Error during service deregistration: {e}")
            # --- ENDE NEU ---


        # Clean up domain data if this was the last entry
        if not hass.data[DOMAIN]:
            hass.data.pop(DOMAIN)
            _LOGGER.debug("Removed DOMAIN data as it was the last entry.")

    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    _LOGGER.info("Reloading Solar Forecast ML integration")
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)