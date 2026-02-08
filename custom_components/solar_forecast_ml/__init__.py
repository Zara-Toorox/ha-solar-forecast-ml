# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Solar Forecast ML V16.2.0 - Main Integration Entry Point.

Setup, Unload, Remove, and Migration Entry Points.
Service Registration and File Logging Setup.
All data operations use DatabaseManager (no JSON).

@zara
"""


# PyArmor Runtime Path Setup - MUST be before any protected module imports
import sys
from pathlib import Path as _Path
_runtime_path = str(_Path(__file__).parent)
if _runtime_path not in sys.path:
    sys.path.insert(0, _runtime_path)

# Pre-load PyArmor runtime at module level (before async event loop)
try:
    import pyarmor_runtime_009810  # noqa: F401
except ImportError:
    pass  # Runtime not present (development mode)

import atexit
import asyncio
import logging
import queue
from datetime import timedelta
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType

from .const import (
    DOMAIN,
    PLATFORMS,
    VERSION,
)
from .core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)

# File logging globals @zara
_log_queue_listener: Optional[QueueListener] = None
_log_queue_handler: Optional[QueueHandler] = None
_logging_initialized: bool = False


async def _migrate_db_remove_default_panel_group(data_manager: "DataManager") -> bool:
    """V16 Migration: Remove obsolete 'Default' panel group from database. @zara

    Since V13, panel groups must be explicitly named. Pre-V13 installations may
    still have a 'Default' group that can contaminate forecasts and calibration.

    This migration uses DatabaseManager instead of JSON files.

    Returns:
        True if any changes were made, False if already clean.
    """
    changes_made = False

    try:
        db = data_manager._db_manager

        # Clean physics_calibration_groups @zara
        result = await db.fetchone(
            "SELECT COUNT(*) FROM physics_calibration_groups WHERE group_name = 'Default'"
        )
        if result and result[0] > 0:
            await db.execute(
                "DELETE FROM physics_calibration_groups WHERE group_name = 'Default'"
            )
            await db.execute(
                "DELETE FROM physics_calibration_hourly WHERE group_name = 'Default'"
            )
            await db.execute(
                "DELETE FROM physics_calibration_buckets WHERE group_name = 'Default'"
            )
            await db.execute(
                "DELETE FROM physics_calibration_bucket_hourly WHERE group_name = 'Default'"
            )
            _LOGGER.info("V16 Migration: Removed 'Default' from physics calibration tables")
            changes_made = True

        # Clean physics_calibration_history @zara
        result = await db.fetchone(
            "SELECT COUNT(*) FROM physics_calibration_history WHERE group_name = 'Default'"
        )
        if result and result[0] > 0:
            await db.execute(
                "DELETE FROM physics_calibration_history WHERE group_name = 'Default'"
            )
            _LOGGER.info("V16 Migration: Removed 'Default' from calibration history")
            changes_made = True

        if changes_made:
            _LOGGER.info("V16 Migration: 'Default' panel group cleanup completed")
        else:
            _LOGGER.debug("V16 Migration: No 'Default' panel group found - data already clean")

    except Exception as e:
        _LOGGER.warning(f"V16 Migration failed (non-critical): {e}")

    return changes_made


async def setup_file_logging(hass: HomeAssistant) -> None:
    """Setup non-blocking file logging using QueueHandler. @zara

    Prevents duplicate handlers on reload by checking initialization state.
    """
    global _log_queue_listener, _log_queue_handler, _logging_initialized

    if _logging_initialized and _log_queue_listener is not None:
        _LOGGER.debug("File logging already initialized - skipping (prevents duplicate handlers)")
        return

    def _setup_logging_sync():
        """Synchronous file operations - runs in executor. @zara"""
        global _log_queue_listener, _log_queue_handler, _logging_initialized

        try:
            integration_logger = logging.getLogger(__package__)

            # Remove any existing QueueHandlers to prevent accumulation @zara
            existing_queue_handlers = [
                h for h in integration_logger.handlers
                if isinstance(h, QueueHandler)
            ]
            for handler in existing_queue_handlers:
                _LOGGER.debug(f"Removing existing QueueHandler: {handler}")
                integration_logger.removeHandler(handler)

            log_dir = Path(hass.config.path("solar_forecast_ml/logs"))
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / "solar_forecast_ml.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(formatter)

            log_queue: queue.Queue = queue.Queue(-1)

            _log_queue_handler = QueueHandler(log_queue)
            _log_queue_handler.setLevel(logging.DEBUG)

            _log_queue_listener = QueueListener(
                log_queue,
                file_handler,
                respect_handler_level=True,
            )
            _log_queue_listener.start()

            atexit.register(_stop_queue_listener)

            integration_logger.addHandler(_log_queue_handler)
            integration_logger.setLevel(logging.DEBUG)

            _logging_initialized = True

            return str(log_file)

        except Exception as e:
            _LOGGER.error(f"Failed to setup file logging: {e}", exc_info=True)
            return None

    loop = asyncio.get_running_loop()
    log_file = await loop.run_in_executor(None, _setup_logging_sync)

    if log_file:
        _LOGGER.info(f"File logging enabled (non-blocking): {log_file}")


def _stop_queue_listener() -> None:
    """Stop the queue listener on shutdown. @zara"""
    global _log_queue_listener, _log_queue_handler, _logging_initialized

    if _log_queue_handler is not None:
        try:
            integration_logger = logging.getLogger(__package__)
            integration_logger.removeHandler(_log_queue_handler)
            _log_queue_handler = None
        except Exception:
            pass

    if _log_queue_listener is not None:
        try:
            _log_queue_listener.stop()
            _log_queue_listener = None
        except Exception:
            pass

    _logging_initialized = False


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Solar Forecast ML integration. @zara"""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Solar Forecast ML from a config entry. @zara"""
    from .coordinator import SolarForecastMLCoordinator
    from .core.core_dependency_handler import DependencyHandler
    from .services.service_notification import create_notification_service

    await setup_file_logging(hass)

    # Register update listener for option changes @zara
    entry.async_on_unload(entry.add_update_listener(_async_options_updated))

    # Check ML dependencies @zara
    dependency_handler = DependencyHandler()
    dependencies_ok = await dependency_handler.check_dependencies(hass)

    if not dependencies_ok:
        _LOGGER.warning("Some ML dependencies are missing. ML features will be disabled.")

    hass.data.setdefault(DOMAIN, {})

    # Create notification service @zara
    notification_service = await create_notification_service(hass, entry)
    if notification_service:
        hass.data[DOMAIN]["notification_service"] = notification_service
        _LOGGER.debug("NotificationService created and stored in hass.data")
    else:
        _LOGGER.warning("NotificationService could not be created")

    # Setup data directory @zara
    data_dir = Path(hass.config.path("solar_forecast_ml"))
    flag_file = Path(hass.config.path(".storage/solar_forecast_ml_v16_installed"))

    try:
        await hass.async_add_executor_job(lambda: data_dir.mkdir(parents=True, exist_ok=True))
    except Exception as e:
        _LOGGER.error(f"Failed to create data directory: {e}", exc_info=True)

    # Initialize coordinator @zara
    coordinator = SolarForecastMLCoordinator(hass, entry, dependencies_ok=dependencies_ok)

    # Run async setup (includes DataManager and DB initialization) @zara
    setup_ok = await coordinator.async_setup()
    if not setup_ok:
        _LOGGER.error("Failed to setup Solar Forecast coordinator")
        return False

    # V16 Migration: Remove 'Default' panel group in background (non-blocking) @zara
    async def _delayed_v16_migration():
        """Run V16 migration in background after HA startup."""
        if not coordinator.data_manager:
            return
        try:
            await asyncio.sleep(3)  # Short delay to not block startup
            await _migrate_db_remove_default_panel_group(coordinator.data_manager)
        except Exception as e:
            _LOGGER.warning(f"V16 Migration failed (non-critical): {e}")

    if coordinator.data_manager:
        hass.async_create_task(
            _delayed_v16_migration(),
            name="solar_forecast_ml_v16_migration"
        )

    # JSON Migration runs in background after startup to not block HA bootstrap @zara
    async def _delayed_json_migration():
        """Run JSON migration in background after HA startup."""
        if not coordinator.data_manager or not coordinator.data_manager._db_manager:
            return

        try:
            # Wait for HA to fully start
            await asyncio.sleep(10)
            _LOGGER.info("Starting JSON migration in background...")

            from .data.json_migration import run_json_migration
            migration_stats = await run_json_migration(hass, coordinator.data_manager._db_manager)

            if migration_stats.imported > 0 or migration_stats.updated > 0:
                _LOGGER.info(
                    f"JSON Migration completed: Imported={migration_stats.imported}, "
                    f"Updated={migration_stats.updated}, Skipped={migration_stats.skipped}, "
                    f"Errors={migration_stats.errors}"
                )
            else:
                _LOGGER.debug("JSON Migration: No data to migrate")
        except Exception as e:
            _LOGGER.warning(f"JSON Migration failed (non-critical): {e}", exc_info=True)

    # First refresh runs in background to not block HA startup @zara
    async def _delayed_first_refresh():
        """Run first data refresh in background after HA startup."""
        try:
            await asyncio.sleep(5)
            async with asyncio.timeout(60):
                await coordinator.async_config_entry_first_refresh()
            _LOGGER.info("First data refresh completed successfully")
        except asyncio.TimeoutError:
            _LOGGER.debug(
                "First data refresh timed out after 60s - using cached data (normal during startup)"
            )
        except Exception as e:
            _LOGGER.debug(f"First data refresh deferred: {e} - using cached data")

    hass.async_create_task(
        _delayed_first_refresh(),
        name="solar_forecast_ml_first_refresh"
    )

    hass.async_create_task(
        _delayed_json_migration(),
        name="solar_forecast_ml_json_migration"
    )

    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Forward entry setup to platforms @zara
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services in background to not block bootstrap @zara
    async def _delayed_service_registration():
        await asyncio.sleep(1)
        await _async_register_services(hass, entry, coordinator)
        _LOGGER.debug("Services registered successfully")

    hass.async_create_task(
        _delayed_service_registration(),
        name="solar_forecast_ml_service_registration"
    )

    # Show installation notification for new installs @zara
    notification_marker = Path(hass.config.path(".storage/solar_forecast_ml_v16_notified"))

    if not flag_file.exists():
        _LOGGER.info("╔══════════════════════════════════════════════════════════════════╗")
        _LOGGER.info("║  Solar Forecast ML — Sarpeidion AI & DB-Version               ║")
        _LOGGER.info("║  Fresh Installation — Database storage initialized             ║")
        _LOGGER.info("╚══════════════════════════════════════════════════════════════════╝")

        try:
            flag_content = (
                f"Solar Forecast ML V{VERSION}\n"
                f"Installed: {dt_util.now().isoformat()}\n"
                f"Database-based storage - no JSON migration needed\n"
            )
            await hass.async_add_executor_job(flag_file.write_text, flag_content)
        except Exception as e:
            _LOGGER.warning(f"Could not write installation flag: {e}")

    if not notification_marker.exists():
        async def _send_install_notification():
            await asyncio.sleep(2)
            await hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": "☀️ Solar Forecast ML — Sarpeidion AI & DB-Version",
                    "message": (
                        "Installation successful!\n\n"
                        "**Next Steps:**\n"
                        "1. Complete the setup (Settings -> Integrations)\n"
                        "2. Wait 10 minutes after configuration\n"
                        "3. Restart Home Assistant to refresh all caches\n\n"
                        "*\"Logic is the beginning of wisdom, not the end.\"* — Spock\n\n"
                        "by Zara-Toorox — Live long and prosper!"
                    ),
                    "notification_id": "solar_forecast_ml_v16_installed",
                },
            )
            await hass.async_add_executor_job(
                notification_marker.write_text,
                f"Installation notification shown at {dt_util.now().isoformat()}"
            )

        hass.async_create_task(
            _send_install_notification(),
            name="solar_forecast_ml_install_notification"
        )
        _LOGGER.info("Installation notification shown to user")

    # Show startup notification @zara
    if notification_service:
        try:
            installed_packages = []
            missing_packages = []

            if dependencies_ok:
                installed_packages = dependency_handler.get_installed_packages()
            else:
                missing_packages = dependency_handler.get_missing_packages()

            use_attention = False
            if coordinator.ai_predictor:
                use_attention = getattr(coordinator.ai_predictor, "use_attention", False)

            await notification_service.show_startup_success(
                ml_mode=dependencies_ok,
                installed_packages=installed_packages,
                missing_packages=missing_packages,
                use_attention=use_attention,
            )
            _LOGGER.debug("Startup notification triggered")
        except Exception as e:
            _LOGGER.warning(f"Failed to show startup notification: {e}", exc_info=True)

    mode_str = "Hybrid-KI (Full Features)" if dependencies_ok else "Fallback Mode (Rule-Based)"

    # Auto-sync extra features on update @zara
    try:
        from .services.service_extra_features import ExtraFeaturesInstaller

        extra_installer = ExtraFeaturesInstaller(hass)
        updated_features, _ = await extra_installer.sync_on_update()

        if updated_features:
            async def _send_update_notification():
                await asyncio.sleep(2)
                await hass.services.async_call(
                    "persistent_notification",
                    "create",
                    {
                        "title": "Extra Features Updated",
                        "message": (
                            f"The following extra features were updated:\n\n"
                            f"**{', '.join(updated_features)}**\n\n"
                            "Please **restart Home Assistant** to load the new versions."
                        ),
                        "notification_id": "solar_forecast_ml_extra_features_updated",
                    },
                )

            hass.async_create_task(
                _send_update_notification(),
                name="solar_forecast_ml_update_notification"
            )
    except Exception as e:
        _LOGGER.warning(f"Extra features sync failed: {e}")

    w = 61
    banner = [
        "╔" + "═" * w + "╗",
        "║" + "  Solar Forecast ML — Sarpeidion AI & DB-Version".ljust(w) + "║",
        "║" + f"  Mode: {mode_str}".ljust(w) + "║",
        "║" + '  "Logic is the beginning of wisdom, not the end." — Spock'.ljust(w) + "║",
        "╚" + "═" * w + "╝",
    ]
    _LOGGER.info("\n" + "\n".join(banner))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry. @zara

    Properly cleans up logging handlers on unload to prevent duplicate log entries.
    """
    from .astronomy.astronomy_cache_manager import reset_cache_manager

    _LOGGER.info("Unloading Solar Forecast ML integration...")

    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        coordinator = hass.data[DOMAIN].pop(entry.entry_id)

        await coordinator.async_shutdown()

        reset_cache_manager()

        if not hass.data[DOMAIN]:
            _async_unregister_services(hass)

            # Stop logging when last entry is unloaded @zara
            _stop_queue_listener()
            _LOGGER.debug("File logging stopped (last config entry unloaded)")

    _LOGGER.info("Solar Forecast ML integration unloaded successfully")
    return unload_ok


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle removal of a config entry - clean up entity registry. @zara

    Called when the user removes the integration completely.
    """
    from homeassistant.helpers import entity_registry as er

    _LOGGER.info("Removing Solar Forecast ML integration and cleaning up entities...")

    ent_reg = er.async_get(hass)

    # Find all entities for this config entry @zara
    entities_to_remove = [
        entity_entry.entity_id
        for entity_entry in ent_reg.entities.values()
        if entity_entry.config_entry_id == entry.entry_id
    ]

    # Remove all entities @zara
    for entity_id in entities_to_remove:
        _LOGGER.debug(f"Removing entity: {entity_id}")
        ent_reg.async_remove(entity_id)

    _LOGGER.info(f"Removed {len(entities_to_remove)} entities from registry")


async def _async_options_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update - reload integration to apply changes. @zara

    Called when the user changes options (diagnostic mode, etc.)
    """
    _LOGGER.info("Options updated, reloading integration to apply changes...")
    await hass.config_entries.async_reload(entry.entry_id)


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry. @zara

    Handle config entry migration when VERSION changes.
    Clean up orphaned entities from removed features.
    V16: All migrations use database, not JSON.
    """
    from homeassistant.helpers import entity_registry as er

    _LOGGER.debug(f"Migrating from version {config_entry.version}")

    ent_reg = er.async_get(hass)

    # Patterns for diagnostic entities to remove when diagnostic mode is disabled @zara
    diagnostic_patterns = [
        "diagnostic_status",
        "external_sensors_status",
        "next_production_start",
        "ml_service_status",
        "ml_metrics",
        "ml_training_readiness",
        "active_prediction_model",
        "pattern_count",
        "physics_samples",
    ]

    diagnostic_enabled = config_entry.options.get("diagnostic", True)

    if not diagnostic_enabled:
        entities_removed = 0
        for entity_entry in list(ent_reg.entities.values()):
            if entity_entry.config_entry_id != config_entry.entry_id:
                continue

            for pattern in diagnostic_patterns:
                if pattern in str(entity_entry.unique_id).lower():
                    _LOGGER.debug(f"Removing orphaned diagnostic entity: {entity_entry.entity_id}")
                    ent_reg.async_remove(entity_entry.entity_id)
                    entities_removed += 1
                    break

        if entities_removed > 0:
            _LOGGER.info(f"Removed {entities_removed} orphaned diagnostic entities")

    return True


async def _async_register_services(
    hass: HomeAssistant, entry: ConfigEntry, coordinator: "SolarForecastMLCoordinator"
) -> None:
    """Register integration services using Service Registry. @zara"""
    from .services.service_registry import ServiceRegistry

    registry = ServiceRegistry(hass, entry, coordinator)
    await registry.async_register_all_services()

    hass.data[DOMAIN]["service_registry"] = registry


def _async_unregister_services(hass: HomeAssistant) -> None:
    """Unregister integration services using Service Registry. @zara"""
    registry = hass.data[DOMAIN].get("service_registry")
    if registry:
        registry.unregister_all_services()
    else:
        _LOGGER.warning("Service registry not found for cleanup")
