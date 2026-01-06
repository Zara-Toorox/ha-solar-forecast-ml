# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

# PyArmor Runtime Path Setup - MUST be before any protected module imports
import sys
from pathlib import Path as _Path
_runtime_path = str(_Path(__file__).parent)
if _runtime_path not in sys.path:
    sys.path.insert(0, _runtime_path)

# Pre-load PyArmor runtime at module level (before async event loop)
# This prevents "blocking call to open" warning from platform.libc_ver()
try:
    import pyarmor_runtime_009810  # noqa: F401
except ImportError:
    pass  # Runtime not present (development mode)

import atexit
import logging
import queue
import threading
from datetime import timedelta
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.typing import ConfigType

from .const import (
    DOMAIN,
    PLATFORMS,
)
from .core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)

_log_queue_listener: QueueListener | None = None
_log_queue_handler: QueueHandler | None = None
_logging_initialized: bool = False

async def setup_file_logging(hass: HomeAssistant) -> None:
    """Setup non-blocking file logging using QueueHandler @zara

    CRITICAL FIX V12.4.0: Prevents duplicate handlers on reload.
    This function now checks if logging is already initialized and skips
    re-initialization to prevent log message duplication (2x, 3x, 6x entries).
    """
    global _log_queue_listener, _log_queue_handler, _logging_initialized

    # CRITICAL: Skip if already initialized to prevent handler accumulation
    if _logging_initialized and _log_queue_listener is not None:
        _LOGGER.debug("File logging already initialized - skipping (prevents duplicate handlers)")
        return

    def _setup_logging_sync():
        """Synchronous file operations - runs in executor to avoid blocking @zara"""
        global _log_queue_listener, _log_queue_handler, _logging_initialized

        try:
            integration_logger = logging.getLogger(__package__)

            # CRITICAL FIX: Remove any existing QueueHandlers first to prevent accumulation
            # This handles edge cases where _logging_initialized flag was reset but handlers remain
            existing_queue_handlers = [
                h for h in integration_logger.handlers
                if isinstance(h, QueueHandler)
            ]
            for handler in existing_queue_handlers:
                _LOGGER.debug(f"Removing existing QueueHandler to prevent duplication: {handler}")
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

    import asyncio

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
    """Set up the Solar Forecast ML integration @zara"""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Solar Forecast ML from a config entry @zara"""

    from .coordinator import SolarForecastMLCoordinator
    from .core.core_dependency_handler import DependencyHandler
    from .services.service_notification import create_notification_service

    await setup_file_logging(hass)

    # Register update listener for option changes (diagnostic mode, etc.)
    entry.async_on_unload(entry.add_update_listener(_async_options_updated))

    dependency_handler = DependencyHandler()
    dependencies_ok = await dependency_handler.check_dependencies(hass)

    if not dependencies_ok:
        _LOGGER.warning("Some ML dependencies are missing. ML features will be disabled.")

    hass.data.setdefault(DOMAIN, {})

    notification_service = await create_notification_service(hass, entry)
    if notification_service:
        hass.data[DOMAIN]["notification_service"] = notification_service
        _LOGGER.debug("NotificationService created and stored in hass.data")
    else:
        _LOGGER.warning("NotificationService could not be created")

    import shutil
    from pathlib import Path

    from .core.core_helpers import SafeDateTimeUtil as dt_util

    flag_file = Path(hass.config.path(".storage/solar_forecast_ml_v10_installed"))
    data_dir = Path(hass.config.path("solar_forecast_ml"))
    template_dir = Path(__file__).parent / "pre-installation"

    try:
        await hass.async_add_executor_job(lambda: data_dir.mkdir(parents=True, exist_ok=True))
    except Exception as e:
        _LOGGER.error(f"Failed to create data directory: {e}", exc_info=True)

    if not flag_file.exists():
        _LOGGER.warning("=" * 70)
        _LOGGER.warning("Solar Forecast ML - Clean Slate Installation")
        _LOGGER.warning("Removing all beta data and installing fresh template")
        _LOGGER.warning("=" * 70)

        try:

            if data_dir.exists():
                _LOGGER.info(f"Removing old beta data from: {data_dir}")
                await hass.async_add_executor_job(shutil.rmtree, data_dir)

            _LOGGER.info(f"Installing template structure from: {template_dir}")
            await hass.async_add_executor_job(shutil.copytree, template_dir, data_dir)

            flag_content = (
                f"Solar Forecast ML 'Sarpeidon'\n"
                f"Installed: {dt_util.now().isoformat()}\n"
                f"First stable production release\n"
                f"Template-based installation - no legacy migrations\n"
            )
            await hass.async_add_executor_job(flag_file.write_text, flag_content)

            _LOGGER.info("=" * 70)
            _LOGGER.info("✓ Clean Slate Installation completed successfully")
            _LOGGER.info("✓ Template structure deployed from pre-installation/")
            _LOGGER.info("=" * 70)

        except Exception as e:
            _LOGGER.error(f"Clean Slate Installation failed: {e}", exc_info=True)
            _LOGGER.error("Continuing with setup - data directory may be incomplete")
    else:
        _LOGGER.debug("Already installed (flag exists in .storage)")

    from .data.data_startup_initializer import StartupInitializer

    initializer_config = {
        "latitude": entry.data.get("latitude", hass.config.latitude),
        "longitude": entry.data.get("longitude", hass.config.longitude),
        "solar_capacity": entry.data.get("solar_capacity", 2.0),
        "timezone": str(hass.config.time_zone),
    }

    initializer = StartupInitializer(data_dir, initializer_config)

    try:
        init_success = await hass.async_add_executor_job(initializer.initialize_all)
        if not init_success:
            _LOGGER.error("Startup Initializer reported failures - check logs above")
    except Exception as e:
        _LOGGER.error(f"Startup Initializer crashed: {e}", exc_info=True)

    coordinator = SolarForecastMLCoordinator(hass, entry, dependencies_ok=dependencies_ok)

    setup_ok = await coordinator.async_setup()
    if not setup_ok:
        _LOGGER.error("Failed to setup Solar Forecast coordinator")
        return False

    # First refresh with timeout to prevent blocking HA startup if network/APIs unavailable
    import asyncio
    try:
        async with asyncio.timeout(60):
            await coordinator.async_config_entry_first_refresh()
    except asyncio.TimeoutError:
        _LOGGER.warning(
            "First data refresh timed out after 60s - continuing with cached/default data. "
            "This may happen if network or weather APIs are temporarily unavailable."
        )

    hass.data[DOMAIN][entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    await _async_register_services(hass, entry, coordinator)

    notification_marker = Path(hass.config.path(".storage/solar_forecast_ml_v10_notified"))

    if not notification_marker.exists():

        await hass.services.async_call(
            "persistent_notification",
            "create",
            {
                "title": "✅ Solar Forecast ML 'Sarpeidon' Installed",
                "message": (
                    "Installation of **Solar Forecast ML 'Sarpeidon'** was successful!\n\n"
                    "**Next Steps:**\n"
                    "1. Complete the setup (Settings → Integrations)\n"
                    "2. Wait **10 minutes** after configuration\n"
                    "3. Restart Home Assistant to refresh all caches with your data\n\n"
                    "**Note:** This is the first stable production release. "
                    "All beta data has been replaced with a clean template structure.\n\n"
                    "Good luck with your solar forecast!"
                ),
                "notification_id": "solar_forecast_ml_v10_installed",
            },
        )

        await hass.async_add_executor_job(
            notification_marker.write_text,
            f"Installation notification shown at {dt_util.now().isoformat()}"
        )
        _LOGGER.info("Installation notification shown to user")

    if notification_service:
        try:
            installed_packages = []
            missing_packages = []

            if dependencies_ok:

                installed_packages = dependency_handler.get_installed_packages()
            else:

                missing_packages = dependency_handler.get_missing_packages()

            # Check if attention mechanism is enabled
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

    mode_str = "ML Mode (Full Features)" if dependencies_ok else "Fallback Mode (Rule-Based)"

    # Auto-sync extra features on SFML update (version-based, not every start)
    try:
        from .services.service_extra_features import ExtraFeaturesInstaller

        extra_installer = ExtraFeaturesInstaller(hass)
        updated_features, _ = await extra_installer.sync_on_update()

        if updated_features:
            # Notify user that restart is needed for updated features
            await hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": "🔄 Extra Features Updated",
                    "message": (
                        f"The following extra features were updated:\n\n"
                        f"**{', '.join(updated_features)}**\n\n"
                        "Please **restart Home Assistant** to load the new versions."
                    ),
                    "notification_id": "solar_forecast_ml_extra_features_updated",
                },
            )
    except Exception as e:
        _LOGGER.debug(f"Extra features sync skipped: {e}")

    _LOGGER.info(
        "=" * 70 + "\n"
        'Solar Forecast ML "Sarpeidon" - Setup Complete!\n'
        f"Mode: {mode_str}\n"
        '"The future is not set in stone, but with data we illuminate the path."\n'
        "Author: Zara-Toorox | Live long and prosper!\n" + "=" * 70
    )

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry @zara

    CRITICAL FIX V12.4.0: Now properly cleans up logging handlers on unload
    to prevent duplicate log entries after reload.
    """
    _LOGGER.info("Unloading Solar Forecast ML integration...")

    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:

        coordinator = hass.data[DOMAIN].pop(entry.entry_id)

        await coordinator.async_shutdown()

        if not hass.data[DOMAIN]:
            _async_unregister_services(hass)

            # CRITICAL FIX V12.4.0: Stop logging when last entry is unloaded
            # This prevents handler accumulation on reload
            _stop_queue_listener()
            _LOGGER.debug("File logging stopped (last config entry unloaded)")

    _LOGGER.info("Solar Forecast ML integration unloaded successfully")
    return unload_ok


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle removal of a config entry - clean up entity registry @zara

    This is called when the user removes the integration completely.
    It ensures all entities are removed from the entity registry.
    """
    from homeassistant.helpers import entity_registry as er

    _LOGGER.info("Removing Solar Forecast ML integration and cleaning up entities...")

    # Get the entity registry
    ent_reg = er.async_get(hass)

    # Find all entities for this config entry
    entities_to_remove = [
        entity_entry.entity_id
        for entity_entry in ent_reg.entities.values()
        if entity_entry.config_entry_id == entry.entry_id
    ]

    # Remove all entities
    for entity_id in entities_to_remove:
        _LOGGER.debug(f"Removing entity: {entity_id}")
        ent_reg.async_remove(entity_id)

    _LOGGER.info(f"Removed {len(entities_to_remove)} entities from registry")


async def _async_options_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update - reload integration to apply changes @zara

    This is called when the user changes options (diagnostic mode, etc.)
    We need to reload the integration to properly add/remove sensors.
    """
    _LOGGER.info("Options updated, reloading integration to apply changes...")
    await hass.config_entries.async_reload(entry.entry_id)


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry @zara

    Handle config entry migration when VERSION changes.
    Clean up orphaned entities from removed features.
    """
    from homeassistant.helpers import entity_registry as er

    _LOGGER.debug(f"Migrating from version {config_entry.version}")

    # Clean up any orphaned diagnostic entities if diagnostic mode was disabled
    ent_reg = er.async_get(hass)

    # List of diagnostic entity unique_id patterns
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
        # Remove diagnostic entities from registry
        entities_removed = 0
        for entity_entry in list(ent_reg.entities.values()):
            if entity_entry.config_entry_id != config_entry.entry_id:
                continue

            # Check if this is a diagnostic entity that should be removed
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
    """Register integration services using Service Registry"""
    from .services.service_registry import ServiceRegistry

    registry = ServiceRegistry(hass, entry, coordinator)
    await registry.async_register_all_services()

    hass.data[DOMAIN]["service_registry"] = registry

def _async_unregister_services(hass: HomeAssistant) -> None:
    """Unregister integration services using Service Registry @zara"""
    registry = hass.data[DOMAIN].get("service_registry")
    if registry:
        registry.unregister_all_services()
    else:
        _LOGGER.warning("Service registry not found for cleanup")
