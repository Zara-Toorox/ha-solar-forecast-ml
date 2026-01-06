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
    """Set up Grid Price Monitor from a config entry @zara"""
    # Lazy import to avoid blocking the event loop during module import
    from .coordinator import GridPriceMonitorCoordinator

    _LOGGER.info(
        "Setting up %s v%s",
        NAME,
        VERSION,
    )

    # Initialize domain data storage
    hass.data.setdefault(DOMAIN, {})

    # Create and initialize coordinator
    coordinator = GridPriceMonitorCoordinator(hass, entry)

    # Initialize persistent storage (creates /config/grid_price_monitor/ structure)
    await coordinator.async_initialize_storage()

    # Initialize battery tracker if configured
    await coordinator.async_setup_battery_tracker()

    # Fetch initial data
    await coordinator.async_config_entry_first_refresh()

    # Store coordinator
    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Set up platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register update listener for options changes
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    _LOGGER.info(
        "%s setup complete - monitoring %s electricity prices",
        NAME,
        entry.data.get("country", "DE"),
    )

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
