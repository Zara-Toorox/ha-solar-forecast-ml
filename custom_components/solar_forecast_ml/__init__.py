"""The Solar Forecast ML integration."""
# Version 4.9.2 - DependencyHandler Integration + Startbenachrichtigung Fix # von Zara
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
from .dependency_handler import DependencyHandler  # von Zara

_LOGGER = logging.getLogger(__name__)

# Alle Plattformen die geladen werden # von Zara
PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.BUTTON]


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """
    Migrate old config entries to new version.
    
    Migriert von Version 1/2/3 zu Version 4 # von Zara
    """
    version = config_entry.version
    
    _LOGGER.info(f"Migrating Solar Forecast ML from version {version} to version 4")
    
    # Migration von Version < 4 zu Version 4 # von Zara
    if version < 4:
        # Kopiere alte Daten # von Zara
        new_data = {**config_entry.data}
        
        # Entferne alte OpenWeatherMap-Keys # von Zara
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
        
        # Füge neue Panel-Config hinzu # von Zara
        if CONF_PANEL_EFFICIENCY not in new_data:
            new_data[CONF_PANEL_EFFICIENCY] = DEFAULT_PANEL_EFFICIENCY
        
        if CONF_AZIMUTH not in new_data:
            new_data[CONF_AZIMUTH] = DEFAULT_AZIMUTH
        
        if CONF_TILT not in new_data:
            new_data[CONF_TILT] = DEFAULT_TILT
        
        # Füge Weather Preference hinzu # von Zara
        if CONF_WEATHER_PREFERENCE not in new_data:
            new_data[CONF_WEATHER_PREFERENCE] = WEATHER_PREFERENCE_GENERIC
        
        # Füge Fallback Entity hinzu # von Zara
        if CONF_FALLBACK_ENTITY not in new_data:
            fallback = new_data.get(CONF_WEATHER_ENTITY, WEATHER_FALLBACK_DEFAULT)
            new_data[CONF_FALLBACK_ENTITY] = fallback
        
        # Stelle sicher dass Solar Capacity vorhanden # von Zara
        if CONF_SOLAR_CAPACITY not in new_data:
            new_data[CONF_SOLAR_CAPACITY] = DEFAULT_SOLAR_CAPACITY
        
        # Stelle sicher dass Weather Entity vorhanden # von Zara
        if CONF_WEATHER_ENTITY not in new_data:
            new_data[CONF_WEATHER_ENTITY] = "weather.home"
            _LOGGER.warning(f"No weather entity found, using default: weather.home")
        
        # Update Entry mit neuen Daten # von Zara
        hass.config_entries.async_update_entry(
            config_entry,
            data=new_data,
            version=4
        )
        
        _LOGGER.info("✔ Migration to version 4 completed successfully")
        return True
    
    # Keine Migration nötig # von Zara
    return True


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Solar Forecast ML component # von Zara"""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Solar Forecast ML from a config entry # von Zara"""
    _LOGGER.info("Setting up Solar Forecast ML integration - Version 4.9.2")
    
    # ========================================================================
    # SCHRITT 1: Initialisiere NotificationService # von Zara
    # ========================================================================
    _LOGGER.info("Initializing NotificationService...")
    notification_service = await create_notification_service(hass)
    
    # Speichere NotificationService global # von Zara
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN]["notification_service"] = notification_service
    _LOGGER.info("✔ NotificationService initialized")
    
    # ========================================================================
    # SCHRITT 2: Prüfe Dependencies # von Zara
    # ========================================================================
    _LOGGER.info("Checking dependencies...")
    dependency_handler = DependencyHandler()
    dependencies_ok = dependency_handler.check_dependencies()
    
    # Hole detaillierten Dependency-Status # von Zara
    dep_status = await dependency_handler.get_dependency_status(hass)
    _LOGGER.info(f"Dependencies Status: {dep_status}")
    
    # ========================================================================
    # SCHRITT 3: Erstelle Coordinator mit Dependencies-Status # von Zara
    # ========================================================================
    coordinator = SolarForecastMLCoordinator(
        hass=hass,
        entry=entry,
        dependencies_ok=dependencies_ok  # von Zara
    )
    
    # ========================================================================
    # SCHRITT 4: Initialisiere DataManager # von Zara
    # ========================================================================
    _LOGGER.info("Initializing DataManager...")
    try:
        await coordinator.data_manager.initialize()
        _LOGGER.info(f"✔ DataManager initialized - Directory: {DATA_DIR}")
    except Exception as err:
        _LOGGER.error(f"✗ Failed to initialize DataManager: {err}")
        await notification_service.show_installation_error(
            f"DataManager Initialisierung fehlgeschlagen: {err}"
        )
        return False
    
    # ========================================================================
    # SCHRITT 5: Initial Data Refresh # von Zara
    # ========================================================================
    _LOGGER.info("Starting initial data refresh...")
    await coordinator.async_config_entry_first_refresh()
    
    # ========================================================================
    # SCHRITT 6: Speichere Coordinator # von Zara
    # ========================================================================
    hass.data[DOMAIN][entry.entry_id] = coordinator
    
    # ========================================================================
    # SCHRITT 7: Setup Plattformen # von Zara
    # ========================================================================
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    # ========================================================================
    # SCHRITT 8: Zeige Start-Benachrichtigung mit Dependencies-Info # von Zara
    # ========================================================================
    _LOGGER.info("Showing startup notification...")
    try:
        # Sammle installierte und fehlende Dependencies # von Zara
        installed_deps = []
        missing_deps = []
        
        for pkg, status in dep_status.items():
            if status["satisfied"]:
                version = status.get("version", "unknown")
                installed_deps.append(f"{pkg} ({version})")
            else:
                missing_deps.append(pkg)
        
        # Zeige Benachrichtigung mit allen Infos # von Zara
        if dependencies_ok:
            # Alle Dependencies vorhanden - ML-Mode # von Zara
            _LOGGER.info("✔ All dependencies satisfied - ML Mode active")
            await notification_service.show_startup_success(
                ml_mode=True,
                installed_packages=installed_deps  # von Zara
            )
        else:
            # Dependencies fehlen - Fallback-Mode # von Zara
            _LOGGER.warning(f"⚠️ Missing dependencies: {', '.join(missing_deps)} - Fallback Mode")
            await notification_service.show_startup_success(
                ml_mode=False,
                installed_packages=installed_deps,  # von Zara
                missing_packages=missing_deps
            )
            
    except Exception as err:
        _LOGGER.warning(f"Failed to show startup notification: {err}")
    
    _LOGGER.info("Solar Forecast ML integration setup complete")
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry # von Zara"""
    _LOGGER.info("Unloading Solar Forecast ML integration")
    
    # Unload alle Plattformen # von Zara
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    if unload_ok:
        # Entferne Coordinator # von Zara
        coordinator = hass.data[DOMAIN].pop(entry.entry_id)
        
        # Cleanup DataManager # von Zara
        if coordinator and hasattr(coordinator, 'data_manager'):
            await coordinator.data_manager.cleanup()
        
        # Wenn keine Einträge mehr, entferne Domain # von Zara
        if not hass.data[DOMAIN]:
            hass.data.pop(DOMAIN)
    
    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry # von Zara"""
    _LOGGER.info("Reloading Solar Forecast ML integration")
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)
