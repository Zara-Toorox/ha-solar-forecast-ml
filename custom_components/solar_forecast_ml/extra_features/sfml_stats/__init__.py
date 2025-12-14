"""SFML Stats integration for Home Assistant. @zara

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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_change

from .const import (
    DOMAIN,
    NAME,
    VERSION,
    CONF_SENSOR_SMARTMETER_IMPORT_KWH,
)
from .storage import DataValidator
from .api import async_setup_views, async_setup_websocket
from .services.daily_aggregator import DailyEnergyAggregator
from .services.billing_calculator import BillingCalculator

if TYPE_CHECKING:
    pass

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = []


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the SFML Stats component. @zara"""
    _LOGGER.info("Initializing %s v%s", NAME, VERSION)

    hass.data.setdefault(DOMAIN, {})

    await async_setup_views(hass)
    await async_setup_websocket(hass)
    _LOGGER.info("SFML Stats Dashboard available at: /api/sfml_stats/dashboard")

    return True


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to new version. @zara"""
    _LOGGER.info(
        "Migrating SFML Stats from version %s to %s",
        config_entry.version, 2
    )

    if config_entry.version == 1:
        new_data = {**config_entry.data}
        hass.config_entries.async_update_entry(
            config_entry,
            data=new_data,
            version=2
        )
        _LOGGER.info("Migration to version 2 successful")

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up SFML Stats from a config entry. @zara"""
    _LOGGER.info("Setting up %s (Entry: %s)", NAME, entry.entry_id)

    validator = DataValidator(hass)
    init_success = await validator.async_initialize()

    if not init_success:
        _LOGGER.error("DataValidator could not be initialized")
        return False

    source_status = validator.source_status
    _LOGGER.info(
        "Source status: Solar Forecast ML=%s, Grid Price Monitor=%s",
        source_status.get("solar_forecast_ml", False),
        source_status.get("grid_price_monitor", False),
    )

    if not any(source_status.values()):
        _LOGGER.warning(
            "No source integration found. "
            "Please install Solar Forecast ML or Grid Price Monitor."
        )

    config_path = Path(hass.config.path())
    aggregator = DailyEnergyAggregator(hass, config_path)
    billing_calculator = BillingCalculator(hass, config_path, entry_data=dict(entry.data))

    # Initialize Power Sources Collector
    from .power_sources_collector import PowerSourcesCollector
    power_sources_path = config_path / "sfml_stats" / "data"
    power_sources_collector = PowerSourcesCollector(hass, dict(entry.data), power_sources_path)
    await power_sources_collector.start()

    hass.data[DOMAIN][entry.entry_id] = {
        "validator": validator,
        "config": dict(entry.data),
        "aggregator": aggregator,
        "billing_calculator": billing_calculator,
        "power_sources_collector": power_sources_collector,
    }

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    async def _daily_aggregation_job(now: datetime) -> None:
        """Run daily aggregation job. @zara"""
        _LOGGER.info("Starting scheduled daily energy aggregation")
        await aggregator.async_aggregate_daily()

    cancel_daily_job = async_track_time_change(
        hass,
        _daily_aggregation_job,
        hour=23,
        minute=55,
        second=0,
    )
    hass.data[DOMAIN][entry.entry_id]["cancel_daily_job"] = cancel_daily_job

    _LOGGER.info("Daily energy aggregation scheduled for 23:55")

    smartmeter_import_kwh = entry.data.get(CONF_SENSOR_SMARTMETER_IMPORT_KWH)

    if smartmeter_import_kwh:
        _LOGGER.info("Initializing billing baselines for kWh sensor: %s", smartmeter_import_kwh)
        await billing_calculator.async_ensure_baselines()
    else:
        _LOGGER.info("Billing calculation disabled - no kWh sensor configured")

    tree = await validator.async_get_directory_tree()
    _LOGGER.debug("Directory structure: %s", tree)

    await aggregator.async_aggregate_daily()

    _LOGGER.info(
        "%s successfully set up. Export path: %s",
        NAME,
        validator.export_base_path
    )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry. @zara"""
    _LOGGER.info("Unloading %s (Entry: %s)", NAME, entry.entry_id)

    if entry.entry_id in hass.data[DOMAIN]:
        entry_data = hass.data[DOMAIN][entry.entry_id]
        if "cancel_daily_job" in entry_data:
            entry_data["cancel_daily_job"]()
            _LOGGER.debug("Daily aggregation job cancelled")

        # Stop power sources collector
        if "power_sources_collector" in entry_data:
            await entry_data["power_sources_collector"].stop()
            _LOGGER.debug("Power sources collector stopped")

    if entry.entry_id in hass.data[DOMAIN]:
        del hass.data[DOMAIN][entry.entry_id]

    return True


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry. @zara"""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update - refresh cached config without full reload. @zara"""
    _LOGGER.info("Config entry updated, refreshing cached configuration")

    if entry.entry_id not in hass.data.get(DOMAIN, {}):
        _LOGGER.warning("Entry %s not found in hass.data, skipping update", entry.entry_id)
        return

    entry_data = hass.data[DOMAIN][entry.entry_id]
    new_config = dict(entry.data)

    entry_data["config"] = new_config

    if "billing_calculator" in entry_data:
        billing_calculator = entry_data["billing_calculator"]
        billing_calculator.update_config(new_config)
        _LOGGER.debug("BillingCalculator config updated")

    if "aggregator" in entry_data:
        aggregator = entry_data["aggregator"]
        if hasattr(aggregator, "update_config"):
            aggregator.update_config(new_config)
            _LOGGER.debug("DailyEnergyAggregator config updated")

    _LOGGER.info("Configuration refresh complete")
