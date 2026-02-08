# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML - ML Weather
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************
"""The ML Weather integration."""

import logging
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import (
    DOMAIN,
    PLATFORMS,
    ML_WEATHER_COORDINATOR,
    ML_WEATHER_DATA,
    CONF_DATA_PATH,
    DEFAULT_DB_PATH,
)
from .coordinator import MLWeatherCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old config entry from v1 (JSON) to v2 (database)."""
    if config_entry.version == 1:
        _LOGGER.info("Migrating ML Weather config entry from version 1 to 2")
        new_data = {**config_entry.data}
        old_path = new_data.get(CONF_DATA_PATH, "")
        if old_path.endswith(".json"):
            new_data[CONF_DATA_PATH] = DEFAULT_DB_PATH
            _LOGGER.info("Migrated data path from %s to %s", old_path, DEFAULT_DB_PATH)

        hass.config_entries.async_update_entry(config_entry, data=new_data, version=2)
        _LOGGER.info("Migration to version 2 successful")
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up ML Weather from a config entry."""
    _LOGGER.debug("Setting up ML Weather integration")

    coordinator = MLWeatherCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {
        ML_WEATHER_COORDINATOR: coordinator,
        ML_WEATHER_DATA: coordinator,
    }

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Close database connection
    coordinator: MLWeatherCoordinator = hass.data[DOMAIN][entry.entry_id][ML_WEATHER_COORDINATOR]
    await coordinator.async_close()

    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)
