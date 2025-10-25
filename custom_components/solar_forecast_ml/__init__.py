"""
The Solar Forecast ML integration.

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
# Version 4.9.2 - DependencyHandler Integration + Startbenachrichtigung Fix
from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType

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
)
from .coordinator import SolarForecastMLCoordinator
from .notification_service import create_notification_service
from .dependency_handler import DependencyHandler

_LOGGER = logging.getLogger(__name__)

# Alle Plattformen die geladen werden
PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.BUTTON]


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """
    Migrate old config entries to new version.
    
    Migriert von Version 1/2/3 zu Version 4 # von Zara
    """
    version = config_entry.version
    
    _LOGGER.info(f"Migrating Solar Forecast ML from version {version} to version 4")
    
    # Migration von Version < 4 zu Version 4
    if version < 4:
        # Kopiere alte Daten
        new_data = {**config_entry.data}
        
        # Entferne alte OpenWeatherMap-Keys
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
        
        # FÃƒÂ¼ge neue Panel-Config hinzu
        if CONF_PANEL_EFFICIENCY not in new_data:
            new_data[CONF_PANEL_EFFICIENCY] = DEFAULT_PANEL_EFFICIENCY
        
        if CONF_AZIMUTH not in new_data:
            new_data[CONF_AZIMUTH] = DEFAULT_AZIMUTH
        
        if CONF_TILT not in new_data:
            new_data[CONF_TILT] = DEFAULT_TILT
        
        # FÃƒÂ¼ge Weather Preference hinzu
        if CONF_WEATHER_PREFERENCE not in new_data:
            new_data[CONF_WEATHER_PREFERENCE] = WEATHER_PREFERENCE_GENERIC
        
        # FÃƒÂ¼ge Fallback Entity hinzu
        if CONF_FALLBACK_ENTITY not in new_data:
            fallback = new_data.get(CONF_WEATHER_ENTITY, WEATHER_FALLBACK_DEFAULT)
            new_data[CONF_FALLBACK_ENTITY] = fallback
        
        # Stelle sicher dass Solar Capacity vorhanden
        if CONF_SOLAR_CAPACITY not in new_data:
            new_data[CONF_SOLAR_CAPACITY] = DEFAULT_SOLAR_CAPACITY
        
        # Stelle sicher dass Weather Entity vorhanden
        if CONF_WEATHER_ENTITY not in new_data:
            new_data[CONF_WEATHER_ENTITY] = "weather.home"
            _LOGGER.warning(f"No weather entity found, using default: weather.home")
        
        # Update Entry mit neuen Daten
        hass.config_entries.async_update_entry(
            config_entry,
            data=new_data,
            version=4
        )
        
        _LOGGER.info("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â Migration to version 4 completed successfully")
        return True
    
    # Keine Migration nÃƒÂ¶tig
    return True


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Solar Forecast ML component # von Zara"""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Solar Forecast ML from a config entry # von Zara"""
    _LOGGER.info("Setting up Solar Forecast ML integration - Version 4.9.2")
    
    # ========================================================================
    # SCHRITT 1: Initialisiere NotificationService
    # ========================================================================
    _LOGGER.info("Initializing NotificationService...")
    notification_service = await create_notification_service(hass)
    
    # Speichere NotificationService global
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN]["notification_service"] = notification_service
    _LOGGER.info("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â NotificationService initialized")
    
    # ========================================================================
    # SCHRITT 2: PrÃƒÂ¼fe Dependencies
    # ========================================================================
    _LOGGER.info("Checking dependencies...")
    dependency_handler = DependencyHandler()
    dependencies_ok = dependency_handler.check_dependencies()
    
    # Hole detaillierten Dependency-Status
    dep_status = await dependency_handler.get_dependency_status(hass)
    _LOGGER.info(f"Dependencies Status: {dep_status}")
    
    # ========================================================================
    # SCHRITT 3: Erstelle Coordinator mit Dependencies-Status
    # ========================================================================
    coordinator = SolarForecastMLCoordinator(
        hass=hass,
        entry=entry,
        dependencies_ok=dependencies_ok
    )
    
    # ========================================================================
    # SCHRITT 4: Initialisiere DataManager
    # ========================================================================
    _LOGGER.info("Initializing DataManager...")
    try:
        await coordinator.data_manager.initialize()
        _LOGGER.info(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â DataManager initialized - Directory: {DATA_DIR}")
    except Exception as err:
        _LOGGER.error(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬â€ Failed to initialize DataManager: {err}")
        await notification_service.show_installation_error(
            f"DataManager Initialisierung fehlgeschlagen: {err}"
        )
        return False
    
    # ========================================================================
    # SCHRITT 4.5: Initialisiere Services (ML, Weather, etc.) - von Zara
    # ========================================================================
    _LOGGER.info("Initializing Services (ML, Weather)...")
    try:
        services_ok = await coordinator.service_manager.initialize_all_services()
        if services_ok:
            _LOGGER.info("Services initialized successfully")
        else:
            _LOGGER.warning("Some services failed to initialize")
    except Exception as err:
        _LOGGER.warning(f"Service initialization issue: {err}")
    
    # ========================================================================
    # SCHRITT 5: Initial Data Refresh
    # ========================================================================
    _LOGGER.info("Starting initial data refresh...")
    await coordinator.async_config_entry_first_refresh()
    
    # ========================================================================
    # SCHRITT 6: Speichere Coordinator
    # ========================================================================
    hass.data[DOMAIN][entry.entry_id] = coordinator
    
    # ========================================================================
    # SCHRITT 7: Setup Plattformen
    # ========================================================================
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    # ========================================================================
    # SCHRITT 8: Zeige Start-Benachrichtigung mit Dependencies-Info
    # ========================================================================
    _LOGGER.info("Showing startup notification...")
    try:
        # Sammle installierte und fehlende Dependencies
        installed_deps = []
        missing_deps = []
        
        for pkg, status in dep_status.items():
            if status["satisfied"]:
                version = status.get("version", "unknown")
                installed_deps.append(f"{pkg} ({version})")
            else:
                missing_deps.append(pkg)
        
        # Zeige Benachrichtigung mit allen Infos
        if dependencies_ok:
            # Alle Dependencies vorhanden - ML-Mode
            _LOGGER.info("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â All dependencies satisfied - ML Mode active")
            await notification_service.show_startup_success(
                ml_mode=True,
                installed_packages=installed_deps
            )
        else:
            # Dependencies fehlen - Fallback-Mode
            _LOGGER.warning(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Missing dependencies: {', '.join(missing_deps)} - Fallback Mode")
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
    """Unload a config entry # von Zara"""
    _LOGGER.info("Unloading Solar Forecast ML integration")
    
    # Unload alle Plattformen
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    if unload_ok:
        # Entferne Coordinator
        coordinator = hass.data[DOMAIN].pop(entry.entry_id)
        
        # Cleanup ProductionTimeCalculator Listener - von Zara
        if coordinator and hasattr(coordinator, 'production_time_calculator'):
            coordinator.production_time_calculator.stop_tracking()
            _LOGGER.info("âœ… ProductionTimeCalculator Listener bereinigt")
        
        # Cleanup DataManager
        if coordinator and hasattr(coordinator, 'data_manager'):
            await coordinator.data_manager.cleanup()
        
        # Wenn keine EintrÃ¤ge mehr, entferne Domain
        if not hass.data[DOMAIN]:
            hass.data.pop(DOMAIN)
    
    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry # von Zara"""
    _LOGGER.info("Reloading Solar Forecast ML integration")
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)
