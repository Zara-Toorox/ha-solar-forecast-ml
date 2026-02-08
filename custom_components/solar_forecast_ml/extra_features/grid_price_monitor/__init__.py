# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

# Only import constants at module level - these are lightweight
from .const import DOMAIN, NAME, PLATFORMS, VERSION

if TYPE_CHECKING:
    from .coordinator import GridPriceMonitorCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Grid Price Monitor from a config entry.

    Uses background initialization to avoid blocking HA startup. @zara
    """
    # Lazy import to avoid blocking the event loop during module import
    from .coordinator import GridPriceMonitorCoordinator

    _LOGGER.info(
        "Setting up %s v%s",
        NAME,
        VERSION,
    )

    # Initialize domain data storage
    hass.data.setdefault(DOMAIN, {})

    # Create coordinator (lightweight, no blocking)
    coordinator = GridPriceMonitorCoordinator(hass, entry)

    # Store coordinator BEFORE background init
    # This allows platforms to set up even if data isn't ready yet
    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Set up platforms - they will show "unavailable" until data is ready
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register update listener for options changes
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    # Background initialization to avoid blocking HA startup @zara
    async def _background_initialization() -> None:
        """Initialize coordinator in background to not block HA startup."""
        import asyncio

        try:
            _LOGGER.debug("Grid Price Monitor: Starting background initialization")

            # Initialize persistent storage (creates /config/grid_price_monitor/ structure)
            await coordinator.async_initialize_storage()

            # Initialize battery tracker if configured
            await coordinator.async_setup_battery_tracker()

            # Fetch initial data with timeout to prevent indefinite blocking
            # Note: Use async_refresh() instead of async_config_entry_first_refresh()
            # because we're in a background task after setup has completed (state is LOADED)
            try:
                async with asyncio.timeout(60):
                    await coordinator.async_refresh()
            except asyncio.TimeoutError:
                _LOGGER.warning(
                    "Grid Price Monitor: First refresh timed out after 60s - "
                    "will retry at next scheduled update"
                )

            _LOGGER.info(
                "%s setup complete - monitoring %s electricity prices",
                NAME,
                entry.data.get("country", "DE"),
            )
        except Exception as err:
            _LOGGER.error("Grid Price Monitor background init failed: %s", err)

    # Start background initialization - does not block HA startup
    hass.async_create_background_task(
        _background_initialization(),
        f"{DOMAIN}_background_init_{entry.entry_id}",
    )

    _LOGGER.info("Grid Price Monitor basic setup complete - initialization continues in background")
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry @zara"""
    _LOGGER.debug("Unloading %s", NAME)

    # Shutdown coordinator components
    coordinator: GridPriceMonitorCoordinator = hass.data[DOMAIN].get(entry.entry_id)
    if coordinator:
        await coordinator.async_shutdown_battery_tracker()
        await coordinator.async_shutdown_storage()

    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update @zara"""
    _LOGGER.debug("Options updated, reloading %s", NAME)
    await hass.config_entries.async_reload(entry.entry_id)
