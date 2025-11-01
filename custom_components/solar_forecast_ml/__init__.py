"""
INIT for Solar Forecast ML Integration.
Orchestrates data fetching, processing, and state updates using a modular architecture.

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

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.components import recorder
from homeassistant.const import Platform
import logging

from .const import DOMAIN, CONF_SOLAR_YIELD_TODAY, CONF_POWER_ENTITY, CONF_WEATHER_ENTITY

# IMPORTE HIER ENTFERNT:
# from .coordinator import SolarForecastMLCoordinator (Verschoben)
# from .core.dependency_handler import DependencyHandler (Verschoben)

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SENSOR, Platform.BUTTON]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Solar Forecast ML from a config entry."""
    
    # +++ IMPORTE HIER EINGEFÃœGT +++
    # Importe werden erst hier ausgefÃ¼hrt, nachdem HA die Requirements (numpy, aiofiles)
    # aus manifest.json installiert hat.
    from .coordinator import SolarForecastMLCoordinator
    from .core.dependency_handler import DependencyHandler
    from .services.notification import create_notification_service
    # +++ ENDE +++
    
    try:
        rec_instance = recorder.get_instance(hass)
        
        if not rec_instance:
            _LOGGER.warning("Recorder component not found. Historical data and ML training might be affected.")
        else:
            recorder_running = True
            
            if hasattr(rec_instance, 'running'):
                recorder_running = rec_instance.running
            elif hasattr(rec_instance, 'started'):
                recorder_running = rec_instance.started
            
            if not recorder_running:
                _LOGGER.warning("Recorder component is not running yet. Historical data and ML training might be affected.")
            else:
                entities_to_check = {
                    "Solar Daily Yield": entry.data.get(CONF_SOLAR_YIELD_TODAY),
                    "Power Sensor": entry.data.get(CONF_POWER_ENTITY),
                    "Weather Sensor": entry.data.get(CONF_WEATHER_ENTITY),
                }
                
                unrecorded_entities = []
                
                for name, entity_id in entities_to_check.items():
                    if entity_id:
                        try:
                            if hasattr(rec_instance, 'entity_filter'):
                                if not rec_instance.entity_filter(entity_id):
                                    unrecorded_entities.append(f"{name} ({entity_id})")
                        except Exception as check_err:
                            _LOGGER.debug(f"Could not verify recorder status for {name}: {check_err}")
                
                if unrecorded_entities:
                    _LOGGER.warning(
                        f"The following entities are not being recorded by Home Assistant Recorder: "
                        f"{', '.join(unrecorded_entities)}. This may affect ML training accuracy. "
                        f"Consider adding them to your recorder configuration."
                    )
                else:
                    _LOGGER.info("All configured entities are being recorded - ready for ML training")
                    
    except Exception as e:
        _LOGGER.warning(f"Could not check recorder status: {e}")
    
    # +++ MIGRATION 6.4.0: Architekturwechsel - Einmaliger Reset +++
    import json
    from pathlib import Path
    from homeassistant.util import dt as dt_util
    
    data_dir = Path(f"/config/{DOMAIN}")
    versinfo_file = data_dir / "versinfo.json"
    migration_done = False
    
    # Check if migration was already performed (async file read)
    def _read_versinfo():
        try:
            if versinfo_file.exists():
                with open(versinfo_file, 'r', encoding='utf-8') as f:
                    return json.load(f).get("migration_6_4_0", False)
        except Exception:
            pass
        return False
    
    migration_done = await hass.async_add_executor_job(_read_versinfo)
    
    # Perform migration if not done yet
    if not migration_done:
        _LOGGER.warning("Architecture change detected (v6.4.0). Performing one-time ML data reset...")
        
        def _perform_migration():
            try:
                # Ensure data directory exists
                data_dir.mkdir(parents=True, exist_ok=True)
                
                # Delete all ML JSON files
                ml_files = [
                    "prediction_history.json",
                    "learned_weights.json", 
                    "hourly_profile.json",
                    "model_state.json",
                    "hourly_samples.json"
                ]
                
                deleted_count = 0
                for filename in ml_files:
                    file_path = data_dir / filename
                    if file_path.exists():
                        file_path.unlink()
                        deleted_count += 1
                
                # Create empty default files immediately
                default_files = {
                    "prediction_history.json": {
                        "version": "1.0",
                        "predictions": [],
                        "last_updated": None
                    },
                    "learned_weights.json": {
                        "version": "1.0",
                        "weights": {},
                        "bias": 0.0,
                        "feature_names": [
                            "temperature", "humidity", "cloudiness", "wind_speed",
                            "hour_of_day", "seasonal_factor", "weather_trend",
                            "production_yesterday"
                        ],
                        "feature_means": {},
                        "feature_stds": {},
                        "accuracy": 0.0,
                        "training_samples": 0,
                        "last_trained": dt_util.now().isoformat(),
                        "model_version": "1.0",
                        "correction_factor": 1.0,
                        "weather_weights": {},
                        "seasonal_factors": {},
                        "feature_importance": {}
                    },
                    "hourly_profile.json": {
                        "version": "1.0",
                        "hourly_averages": {str(h): 0.0 for h in range(24)},
                        "sample_counts": {str(h): 0 for h in range(24)},
                        "last_updated": None
                    },
                    "model_state.json": {
                        "version": "1.0",
                        "model_loaded": False,
                        "last_training": None,
                        "training_samples": 0,
                        "current_accuracy": 0.0,
                        "status": "uninitialized"
                    },
                    "hourly_samples.json": {
                        "version": "1.0",
                        "samples": [],
                        "count": 0,
                        "last_updated": None
                    }
                }
                
                for filename, content in default_files.items():
                    file_path = data_dir / filename
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(content, f, indent=2)
                
                # Create versinfo.json with migration flag
                versinfo_data = {
                    "version": "6.4.0",
                    "migration_6_4_0": True,
                    "migration_date": dt_util.now().isoformat(),
                    "description": "Architecture change - ML data reset completed"
                }
                
                with open(versinfo_file, 'w', encoding='utf-8') as f:
                    json.dump(versinfo_data, f, indent=2)
                
                return deleted_count
                
            except Exception as e:
                _LOGGER.error(f"Migration execution failed: {e}", exc_info=True)
                return 0
        
        try:
            deleted_count = await hass.async_add_executor_job(_perform_migration)
            _LOGGER.info(f"Migration 6.4.0 completed. Deleted {deleted_count} ML data files and recreated defaults.")
        except Exception as migration_err:
            _LOGGER.error(f"Migration 6.4.0 failed: {migration_err}", exc_info=True)
    # +++ ENDE MIGRATION +++
    
    dependency_handler = DependencyHandler()
    dependencies_ok = await dependency_handler.check_dependencies(hass)  # Async mit hass
    
    if not dependencies_ok:
        _LOGGER.warning("Some dependencies are missing. Integration will start with limited functionality.")
    
    # +++ SINGLETON: NotificationService initialisieren BEVOR Coordinator erstellt wird +++
    hass.data.setdefault(DOMAIN, {})
    if "notification_service" not in hass.data[DOMAIN]:
        try:
            notification_service = await create_notification_service(hass, entry)
            hass.data[DOMAIN]["notification_service"] = notification_service
            _LOGGER.info("NotificationService Singleton created and stored in hass.data")
            
            # âœ… Zeige Startup-Benachrichtigung mit Option-PrÃ¼fung
            ml_mode = dependencies_ok
            installed = ["numpy", "aiofiles"] if dependencies_ok else []
            missing = [] if dependencies_ok else ["numpy", "aiofiles"]
            
            await notification_service.show_startup_success(
                ml_mode=ml_mode,
                installed_packages=installed,
                missing_packages=missing
            )
        except Exception as e:
            _LOGGER.warning(f"Failed to initialize NotificationService Singleton: {e}")
    # +++ ENDE +++
    
    coordinator = SolarForecastMLCoordinator(
        hass=hass,
        entry=entry,
        dependencies_ok=dependencies_ok,
    )
    
    # Initialize DataManager (creates missing files, deploys import tools)
    _LOGGER.debug("Initializing DataManager...")
    dm_init_success = await coordinator.data_manager.initialize()
    if not dm_init_success:
        _LOGGER.warning("DataManager initialization had issues, but continuing setup.")
    
    await coordinator.async_config_entry_first_refresh()
    
    hass.data[DOMAIN][entry.entry_id] = coordinator
    
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    if unload_ok:
        # Import hier nÃ¶tig, da Koordinator nicht global verfÃ¼gbar ist
        from .coordinator import SolarForecastMLCoordinator
        
        coordinator: SolarForecastMLCoordinator = hass.data[DOMAIN].pop(entry.entry_id)
        
        # Cleanup-Logik (basierend auf deinem Original-Code)
        if hasattr(coordinator, 'scheduled_tasks') and coordinator.scheduled_tasks:
            coordinator.scheduled_tasks.cancel_listeners()
        elif hasattr(coordinator, 'scheduled_tasks_manager') and coordinator.scheduled_tasks_manager:
             # Fallback auf alten Namen, falls noch verwendet
            await coordinator.scheduled_tasks_manager.cleanup()
        
        # (Optional, aber empfohlen) Eigene Cleanup-Methode fÃ¼r den Koordinator
        if hasattr(coordinator, 'cleanup'):
                await coordinator.cleanup()
        
        # NotificationService Singleton wird NICHT entfernt - bleibt Ã¼ber alle Config Entries bestehen
    
    return unload_ok
