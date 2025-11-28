"""Solar Forecast ML Integration - Main Entry Point V10.0.0 @zara

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
    CONF_BATTERY_ENABLED,
    DOMAIN,
    PLATFORMS,
)
from .core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)

_log_queue_listener: QueueListener | None = None

async def setup_file_logging(hass: HomeAssistant) -> None:
    """Setup non-blocking file logging using QueueHandler @zara"""
    global _log_queue_listener

    def _setup_logging_sync():
        """Synchronous file operations - runs in executor to avoid blocking @zara"""
        global _log_queue_listener

        try:

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

            queue_handler = QueueHandler(log_queue)
            queue_handler.setLevel(logging.DEBUG)

            _log_queue_listener = QueueListener(
                log_queue,
                file_handler,
                respect_handler_level=True,
            )
            _log_queue_listener.start()

            atexit.register(_stop_queue_listener)

            integration_logger = logging.getLogger(__package__)
            integration_logger.addHandler(queue_handler)
            integration_logger.setLevel(logging.DEBUG)

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
    global _log_queue_listener
    if _log_queue_listener is not None:
        try:
            _log_queue_listener.stop()
            _log_queue_listener = None
        except Exception:
            pass

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Solar Forecast ML integration legacy @zara"""

    hass.data.setdefault(DOMAIN, {})
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Solar Forecast ML from a config entry @zara"""

    from .battery.battery_coordinator import BatteryCoordinator
    from .const import CONF_BATTERY_ENABLED
    from .coordinator import SolarForecastMLCoordinator
    from .core.core_dependency_handler import DependencyHandler
    from .services.service_notification import create_notification_service

    await setup_file_logging(hass)

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
                f"Solar Forecast ML 'Lyra'\n"
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
        "battery_enabled": entry.data.get(CONF_BATTERY_ENABLED, False),
        "battery_capacity": entry.data.get("battery_capacity", 10.0),
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

    await coordinator.async_config_entry_first_refresh()

    hass.data[DOMAIN][entry.entry_id] = coordinator

    battery_coordinator = None
    battery_enabled = entry.options.get(
        CONF_BATTERY_ENABLED, entry.data.get(CONF_BATTERY_ENABLED, False)
    )

    if battery_enabled:
        try:
            _LOGGER.info("Battery management enabled - initializing BatteryCoordinator")
            battery_coordinator = BatteryCoordinator(hass, entry)

            battery_setup_ok = await battery_coordinator.async_setup()
            if battery_setup_ok:
                await battery_coordinator.async_config_entry_first_refresh()

                hass.data[DOMAIN][f"{entry.entry_id}_battery"] = battery_coordinator
                _LOGGER.info("BatteryCoordinator initialized successfully")
            else:
                _LOGGER.warning(
                    "BatteryCoordinator setup failed, continuing without battery features"
                )
                battery_coordinator = None
        except Exception as e:
            _LOGGER.error(f"Error setting up BatteryCoordinator: {e}", exc_info=True)
            battery_coordinator = None

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    await _async_register_services(hass, entry, coordinator)

    notification_marker = Path(hass.config.path(".storage/solar_forecast_ml_v10_notified"))

    if not notification_marker.exists():

        await hass.services.async_call(
            "persistent_notification",
            "create",
            {
                "title": "✅ Solar Forecast ML 'Lyra' installiert",
                "message": (
                    "Die Installation von **Solar Forecast ML 'Lyra'** war erfolgreich!\n\n"
                    "**Nächste Schritte:**\n"
                    "1. Führen Sie die Einrichtung durch (Konfiguration → Integrationen)\n"
                    "2. Warten Sie **10 Minuten** nach der Konfiguration\n"
                    "3. Starten Sie Home Assistant **neu**, damit alle Caches mit Ihren Daten aktualisiert werden\n\n"
                    "**Hinweis:** Dies ist die erste stabile Production-Release. "
                    "Alle Beta-Daten wurden durch eine saubere Template-Struktur ersetzt.\n\n"
                    "Viel Erfolg mit Ihrer Solar-Prognose! ☀️"
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

            await notification_service.show_startup_success(
                ml_mode=dependencies_ok,
                installed_packages=installed_packages,
                missing_packages=missing_packages,
            )
            _LOGGER.debug("Startup notification triggered")
        except Exception as e:
            _LOGGER.warning(f"Failed to show startup notification: {e}", exc_info=True)

    mode_str = "ML Mode (Full Features)" if dependencies_ok else "Fallback Mode (Rule-Based)"
    battery_str = "Enabled" if battery_coordinator else "Disabled"

    _LOGGER.info(
        "=" * 70 + "\n"
        'Solar Forecast ML "Lyra" 🌟 - Setup Complete! ✓\n'
        f"Mode: {mode_str} | Battery Management: {battery_str}\n"
        '"The future is not set in stone, but with data we illuminate the path."\n'
        "Author: Zara-Toorox | Live long and prosper! 🖖\n" + "=" * 70
    )

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry @zara"""
    _LOGGER.info("Unloading Solar Forecast ML integration...")

    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:

        coordinator = hass.data[DOMAIN].pop(entry.entry_id)

        await coordinator.async_shutdown()

        battery_key = f"{entry.entry_id}_battery"
        if battery_key in hass.data[DOMAIN]:
            battery_coordinator = hass.data[DOMAIN].pop(battery_key)

            _LOGGER.info("BatteryCoordinator unloaded")

        if not hass.data[DOMAIN]:
            _async_unregister_services(hass)

    _LOGGER.info("Solar Forecast ML integration unloaded successfully")
    return unload_ok

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
