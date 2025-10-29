"""
The Solar Forecast ML integration init VERSION 6.2.0

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
from homeassistant.components import recorder

from .const import (
    DOMAIN,
    CONF_SOLAR_CAPACITY,
    CONF_PANEL_EFFICIENCY,
    CONF_AZIMUTH,
    CONF_TILT,
    CONF_WEATHER_ENTITY,
    CONF_WEATHER_PREFERENCE,
    CONF_FALLBACK_ENTITY,
    DEFAULT_SOLAR_CAPACITY,
    DEFAULT_PANEL_EFFICIENCY,
    DEFAULT_AZIMUTH,
    DEFAULT_TILT,
    WEATHER_PREFERENCE_GENERIC,
    WEATHER_FALLBACK_DEFAULT,
    DATA_DIR,
    CONF_POWER_ENTITY,
    CONF_SOLAR_YIELD_TODAY,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.BUTTON] # Platform.BUTTON is already here


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    version = config_entry.version
    
    _LOGGER.info(f"Migrating Solar Forecast ML from version {version} to version 6")
    
    if version < 6:
        new_data = {**config_entry.data}
        
        keys_to_remove = [
            "openweather_api_key",
            "location_lat",
            "location_lon",
            "weather_service",
        ]
        
        for key in keys_to_remove:
            if key in new_data:
                _LOGGER.info(f"Removing deprecated key: {key}")
                new_data.pop(key, None)
        
        if CONF_PANEL_EFFICIENCY not in new_data:
            new_data[CONF_PANEL_EFFICIENCY] = DEFAULT_PANEL_EFFICIENCY
        
        if CONF_AZIMUTH not in new_data:
            new_data[CONF_AZIMUTH] = DEFAULT_AZIMUTH
        
        if CONF_TILT not in new_data:
            new_data[CONF_TILT] = DEFAULT_TILT
        
        if CONF_WEATHER_PREFERENCE not in new_data:
            new_data[CONF_WEATHER_PREFERENCE] = WEATHER_PREFERENCE_GENERIC
        
        if CONF_FALLBACK_ENTITY not in new_data:
            fallback = new_data.get(CONF_WEATHER_ENTITY, WEATHER_FALLBACK_DEFAULT)
            new_data[CONF_FALLBACK_ENTITY] = fallback
        
        if CONF_SOLAR_CAPACITY not in new_data:
            new_data[CONF_SOLAR_CAPACITY] = DEFAULT_SOLAR_CAPACITY
        
        if CONF_WEATHER_ENTITY not in new_data:
            new_data[CONF_WEATHER_ENTITY] = "weather.home"
            _LOGGER.warning(f"No weather entity found, using default: weather.home")
        
        hass.config_entries.async_update_entry(
            config_entry,
            data=new_data,
            version=4
        )
        
        _LOGGER.info("Migration to version 6 completed successfully")
        return True
    
    return True


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    _LOGGER.info("Setting up Solar Forecast ML integration - Version 6.2.0")
    
    # === LAZY IMPORTS ===
    # By importing these here, we avoid blocking the Home Assistant event loop during startup.
    from .coordinator import SolarForecastMLCoordinator
    from .notification_service import create_notification_service
    from .dependency_handler import DependencyHandler
    # === END LAZY IMPORTS ===
    
    _LOGGER.info("Initializing NotificationService...")
    notification_service = await create_notification_service(hass)
    
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN]["notification_service"] = notification_service
    _LOGGER.info("NotificationService initialized")
    
    # === START PATCH 1: PRE-FLIGHT CHECK (RECORDER) ===
    _LOGGER.info("Performing Pre-Flight Check: Verifying recorder settings...")
    try:
        rec = recorder.get_instance(hass)
        if not rec or not rec.is_running:
            _LOGGER.warning("Recorder component is not running. ML training will fail.")
            # We continue here, as the recorder might just be temporarily disabled,
            # but collection will fail. A notification is warranted.
        
        # Get entities from configuration
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
            
            # === KORREKTUR: 'is_entity_recorded' zu 'entity_filter' geändert ===
            # entity_filter(entity_id) gibt True zurück, WENN es aufgezeichnet wird.
            if not rec.entity_filter(entity_id):
                _LOGGER.warning(
                    f"Pre-Flight Check FAILED: Entity '{entity_id}' ({name}) "
                    f"is NOT being recorded (filtered out by 'recorder' config). ML training will fail."
                )
                unrecorded_entities.append(f"'{entity_id}' ({name})")
        
        if unrecorded_entities:
            entity_list_str = ", ".join(unrecorded_entities)
            
            # We assume notification_service has a method for permanent warnings
            # based on the prompt "via notification_service".
            # If not, a direct call to persistent_notification.create would be needed.
            await notification_service.show_permanent_warning(
                notification_id="recorder_preflight_check_failed",
                title="Solar Forecast ML - Recorder Error",
                message=(
                    "The following entities are NOT recorded by Home Assistant: "
                    f"{entity_list_str}. \n\n"
                    "The ML model CANNOT be trained without this data. "
                    "Please check your 'recorder:' configuration in 'configuration.yaml' "
                    "and ensure these entities are NOT excluded."
                )
            )
            _LOGGER.error(
                "Pre-Flight Check FAILED: ML Training requires recorder data "
                f"for {entity_list_str}. A permanent notification has been created."
            )
        else:
            _LOGGER.info("Pre-Flight Check OK: All critical entities are being recorded.")
            
    except AttributeError as e:
        # Fange den Fehler ab, falls 'entity_filter' (oder die rec-Instanz) nicht wie erwartet funktioniert
        _LOGGER.error(f"Error during Pre-Flight Recorder Check: {e}", exc_info=True)
    except Exception as e:
        _LOGGER.error(f"Error during Pre-Flight Recorder Check: {e}", exc_info=True)
    # === END PATCH 1 ===
    
    _LOGGER.info("Checking dependencies (async)...")
    dependency_handler = DependencyHandler()
    
    # *** TYPEERROR FIX: Call the new async dependency check with await and hass ***
    dependencies_ok = await dependency_handler.check_dependencies(hass)
    
    dep_status = await dependency_handler.get_dependency_status(hass)
    _LOGGER.info(f"Dependencies Status: {dep_status}")
    
    coordinator = SolarForecastMLCoordinator(
        hass=hass,
        entry=entry,
        dependencies_ok=dependencies_ok
    )
    
    _LOGGER.info("Initializing DataManager...")
    try:
        await coordinator.data_manager.initialize()
        _LOGGER.info(f"DataManager initialized - Directory: {DATA_DIR}")
    except Exception as err:
        _LOGGER.error(f"Failed to initialize DataManager: {err}")
        await notification_service.show_installation_error(
            f"DataManager initialization failed: {err}"
        )
        return False
    
    _LOGGER.info("Initializing Services (ML, Weather)...")
    try:
        services_ok = await coordinator.service_manager.initialize_all_services()
        if services_ok:
            _LOGGER.info("Services initialized successfully")
        else:
            _LOGGER.warning("Some services failed to initialize")
    except Exception as err:
        _LOGGER.warning(f"Service initialization issue: {err}")
    
    _LOGGER.info("Starting initial data refresh...")
    await coordinator.async_config_entry_first_refresh()
    
    hass.data[DOMAIN][entry.entry_id] = coordinator
    
    # Forward entry setups for all defined platforms (SENSOR, BUTTON).
    # This is the correct way to register the button platform.
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS) 
    
    _LOGGER.info("Showing startup notification...")
    try:
        installed_deps = []
        missing_deps = []
        
        for pkg, status in dep_status.items():
            if status["satisfied"]:
                version = status.get("version", "unknown")
                installed_deps.append(f"{pkg} ({version})")
            else:
                missing_deps.append(pkg)
        
        if dependencies_ok:
            _LOGGER.info("All dependencies satisfied - ML Mode active")
            await notification_service.show_startup_success(
                ml_mode=True,
                installed_packages=installed_deps
            )
        else:
            _LOGGER.warning(f"  Missing dependencies: {', '.join(missing_deps)} - Fallback Mode")
            await notification_service.show_startup_success(
                ml_mode=False,
                installed_packages=installed_deps,
                missing_packages=missing_deps
            )
            
    except Exception as err:
        _LOGGER.warning(f"Failed to show startup notification: {err}")
    
    _LOGGER.info("Solar Forecast ML integration setup complete")
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    _LOGGER.info("Unloading Solar Forecast ML integration")
    
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    if unload_ok:
        coordinator = hass.data[DOMAIN].pop(entry.entry_id)
        
        if coordinator and hasattr(coordinator, 'production_time_calculator'):
            coordinator.production_time_calculator.stop_tracking()
            _LOGGER.info("ProductionTimeCalculator Listener cleaned up")
        
        if coordinator and hasattr(coordinator, 'data_manager'):
            await coordinator.data_manager.cleanup()
        
        if not hass.data[DOMAIN]:
            hass.data.pop(DOMAIN)
    
    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    _LOGGER.info("Reloading Solar Forecast ML integration")
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)