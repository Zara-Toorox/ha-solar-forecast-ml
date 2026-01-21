"""SFML Stats integration for Home Assistant.

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
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_change

from .const import (
    DOMAIN,
    NAME,
    VERSION,
    CONF_SENSOR_SMARTMETER_IMPORT_KWH,
    CONF_WEATHER_ENTITY,
    CONF_FORECAST_ENTITY_1,
    CONF_FORECAST_ENTITY_2,
    DAILY_AGGREGATION_HOUR,
    DAILY_AGGREGATION_MINUTE,
    DAILY_AGGREGATION_SECOND,
    FORECAST_MORNING_HOUR,
    FORECAST_MORNING_MINUTE,
    FORECAST_EVENING_HOUR,
    FORECAST_EVENING_MINUTE,
)
from .storage import DataValidator
from .api import async_setup_views, async_setup_websocket
from .services.daily_aggregator import DailyEnergyAggregator
from .services.billing_calculator import BillingCalculator
from .services.forecast_comparison_collector import ForecastComparisonCollector

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = []


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the SFML Stats component."""
    _LOGGER.info("Initializing %s v%s", NAME, VERSION)

    hass.data.setdefault(DOMAIN, {})

    await async_setup_views(hass)
    await async_setup_websocket(hass)
    _LOGGER.info("SFML Stats Lite Dashboard available at: /api/sfml_stats_lite/dashboard")

    return True


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to new version."""
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
    """Set up SFML Stats from a config entry."""
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

    # Show persistent notification if no source integration is found
    if not any(source_status.values()):
        _LOGGER.warning(
            "No source integration found. "
            "Please install Solar Forecast ML or Grid Price Monitor."
        )
        try:
            from homeassistant.components.persistent_notification import async_create
            await async_create(
                hass,
                "Keine Quell-Integration gefunden. "
                "Bitte Solar Forecast ML oder Grid Price Monitor installieren, "
                "um alle Funktionen nutzen zu können.",
                title="SFML Stats Lite - Warnung",
                notification_id=f"{DOMAIN}_no_sources",
            )
        except Exception as err:
            _LOGGER.debug("Could not create persistent notification: %s", err)

    config_path = Path(hass.config.path())
    entry_config = dict(entry.data)
    aggregator = DailyEnergyAggregator(hass, config_path)
    billing_calculator = BillingCalculator(hass, config_path, entry_data=entry_config)

    # Initialize Power Sources Collector with error handling
    from .power_sources_collector import PowerSourcesCollector
    power_sources_path = config_path / "sfml_stats_lite" / "data"
    power_sources_collector = PowerSourcesCollector(hass, entry_config, power_sources_path)
    try:
        await power_sources_collector.start()
    except Exception as err:
        _LOGGER.error("Failed to start power sources collector: %s", err)
        # Collector is optional, integration can still function

    # Initialize Weather Collector if weather entity is configured
    weather_collector = None
    weather_entity = entry_config.get(CONF_WEATHER_ENTITY)
    if weather_entity:
        try:
            from .weather_collector import WeatherDataCollector
            weather_path = config_path / "sfml_stats_lite_weather"
            weather_collector = WeatherDataCollector(hass, weather_path)
            _LOGGER.info("Weather collector initialized for entity: %s", weather_entity)
        except Exception as err:
            _LOGGER.error("Failed to initialize weather collector: %s", err)
            # Weather collector is optional

    # Initialize Forecast Comparison Collector if external forecasts are configured
    forecast_comparison_collector = None
    forecast_entity_1 = entry_config.get(CONF_FORECAST_ENTITY_1)
    forecast_entity_2 = entry_config.get(CONF_FORECAST_ENTITY_2)
    if forecast_entity_1 or forecast_entity_2:
        try:
            forecast_comparison_collector = ForecastComparisonCollector(hass, config_path)
            _LOGGER.info("Forecast comparison collector initialized")
        except Exception as err:
            _LOGGER.error("Failed to initialize forecast comparison collector: %s", err)

    hass.data[DOMAIN][entry.entry_id] = {
        "validator": validator,
        "config": entry_config,
        "aggregator": aggregator,
        "billing_calculator": billing_calculator,
        "power_sources_collector": power_sources_collector,
        "weather_collector": weather_collector,
        "forecast_comparison_collector": forecast_comparison_collector,
    }

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    async def _daily_aggregation_job(now: datetime) -> None:
        """Run daily aggregation job."""
        _LOGGER.info("Starting scheduled daily energy aggregation")
        try:
            await aggregator.async_aggregate_daily()
        except Exception as err:
            _LOGGER.error("Daily aggregation failed: %s", err)

    cancel_daily_job = async_track_time_change(
        hass,
        _daily_aggregation_job,
        hour=DAILY_AGGREGATION_HOUR,
        minute=DAILY_AGGREGATION_MINUTE,
        second=DAILY_AGGREGATION_SECOND,
    )
    hass.data[DOMAIN][entry.entry_id]["cancel_daily_job"] = cancel_daily_job

    _LOGGER.info(
        "Daily energy aggregation scheduled for %02d:%02d",
        DAILY_AGGREGATION_HOUR,
        DAILY_AGGREGATION_MINUTE,
    )

    # Schedule Forecast Comparison Jobs if collector is initialized
    if forecast_comparison_collector:
        async def _forecast_morning_job(now: datetime) -> None:
            """Collect morning forecast values."""
            _LOGGER.info("Starting scheduled morning forecast collection")
            try:
                await forecast_comparison_collector.async_collect_morning_forecasts()
            except Exception as err:
                _LOGGER.error("Morning forecast collection failed: %s", err)

        async def _forecast_evening_job(now: datetime) -> None:
            """Collect evening actual production."""
            _LOGGER.info("Starting scheduled evening actual collection")
            try:
                await forecast_comparison_collector.async_collect_evening_actual()
            except Exception as err:
                _LOGGER.error("Evening actual collection failed: %s", err)

        cancel_morning_forecast = async_track_time_change(
            hass,
            _forecast_morning_job,
            hour=FORECAST_MORNING_HOUR,
            minute=FORECAST_MORNING_MINUTE,
            second=0,
        )
        cancel_evening_forecast = async_track_time_change(
            hass,
            _forecast_evening_job,
            hour=FORECAST_EVENING_HOUR,
            minute=FORECAST_EVENING_MINUTE,
            second=0,
        )
        hass.data[DOMAIN][entry.entry_id]["cancel_morning_forecast"] = cancel_morning_forecast
        hass.data[DOMAIN][entry.entry_id]["cancel_evening_forecast"] = cancel_evening_forecast

        _LOGGER.info(
            "Forecast comparison jobs scheduled: Morning %02d:%02d, Evening %02d:%02d",
            FORECAST_MORNING_HOUR, FORECAST_MORNING_MINUTE,
            FORECAST_EVENING_HOUR, FORECAST_EVENING_MINUTE,
        )

        # Collect historical data on first setup
        hass.async_create_background_task(
            forecast_comparison_collector.async_collect_historical(days=7),
            f"{DOMAIN}_forecast_historical",
        )

    smartmeter_import_kwh = entry.data.get(CONF_SENSOR_SMARTMETER_IMPORT_KWH)

    if smartmeter_import_kwh:
        _LOGGER.info("Initializing billing baselines for kWh sensor: %s", smartmeter_import_kwh)
        await billing_calculator.async_ensure_baselines()
    else:
        _LOGGER.debug("Billing calculation disabled - no kWh sensor configured")

    tree = await validator.async_get_directory_tree()
    _LOGGER.debug("Directory structure: %s", tree)

    # Run initial aggregation as background task to not block startup
    async def _initial_aggregation() -> None:
        """Run initial aggregation in background."""
        try:
            await aggregator.async_aggregate_daily()
        except Exception as err:
            _LOGGER.error("Initial aggregation failed: %s", err)

    hass.async_create_background_task(
        _initial_aggregation(),
        f"{DOMAIN}_initial_aggregation",
    )

    _LOGGER.info(
        "%s successfully set up. Export path: %s",
        NAME,
        validator.export_base_path
    )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.info("Unloading %s (Entry: %s)", NAME, entry.entry_id)

    if entry.entry_id not in hass.data.get(DOMAIN, {}):
        _LOGGER.warning("Entry %s not found in hass.data", entry.entry_id)
        return True

    entry_data = hass.data[DOMAIN][entry.entry_id]

    # Cancel scheduled jobs
    if "cancel_daily_job" in entry_data:
        try:
            entry_data["cancel_daily_job"]()
            _LOGGER.debug("Daily aggregation job cancelled")
        except Exception as err:
            _LOGGER.warning("Error cancelling daily job: %s", err)

    # Cancel forecast comparison jobs
    if "cancel_morning_forecast" in entry_data:
        try:
            entry_data["cancel_morning_forecast"]()
            _LOGGER.debug("Morning forecast job cancelled")
        except Exception as err:
            _LOGGER.warning("Error cancelling morning forecast job: %s", err)

    if "cancel_evening_forecast" in entry_data:
        try:
            entry_data["cancel_evening_forecast"]()
            _LOGGER.debug("Evening forecast job cancelled")
        except Exception as err:
            _LOGGER.warning("Error cancelling evening forecast job: %s", err)

    # Stop power sources collector
    if "power_sources_collector" in entry_data and entry_data["power_sources_collector"]:
        try:
            await entry_data["power_sources_collector"].stop()
            _LOGGER.debug("Power sources collector stopped")
        except Exception as err:
            _LOGGER.warning("Error stopping power sources collector: %s", err)

    # Weather collector doesn't need stopping (passive loader)
    if "weather_collector" in entry_data and entry_data["weather_collector"]:
        _LOGGER.debug("Weather collector cleaned up")

    # Dismiss persistent notification if it was created
    try:
        from homeassistant.components.persistent_notification import async_dismiss
        await async_dismiss(hass, f"{DOMAIN}_no_sources")
    except Exception:
        pass  # Notification might not exist

    # Clean up entry data
    del hass.data[DOMAIN][entry.entry_id]

    return True


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update - refresh cached config without full reload."""
    _LOGGER.info("Config entry updated, refreshing cached configuration")

    if entry.entry_id not in hass.data.get(DOMAIN, {}):
        _LOGGER.warning("Entry %s not found in hass.data, skipping update", entry.entry_id)
        return

    entry_data = hass.data[DOMAIN][entry.entry_id]
    new_config = dict(entry.data)

    entry_data["config"] = new_config

    # Update BillingCalculator
    if "billing_calculator" in entry_data and entry_data["billing_calculator"]:
        try:
            entry_data["billing_calculator"].update_config(new_config)
            _LOGGER.debug("BillingCalculator config updated")
        except Exception as err:
            _LOGGER.warning("Error updating BillingCalculator config: %s", err)

    # Update DailyEnergyAggregator
    if "aggregator" in entry_data and entry_data["aggregator"]:
        aggregator = entry_data["aggregator"]
        if hasattr(aggregator, "update_config"):
            try:
                aggregator.update_config(new_config)
                _LOGGER.debug("DailyEnergyAggregator config updated")
            except Exception as err:
                _LOGGER.warning("Error updating DailyEnergyAggregator config: %s", err)

    # Update PowerSourcesCollector
    if "power_sources_collector" in entry_data and entry_data["power_sources_collector"]:
        collector = entry_data["power_sources_collector"]
        if hasattr(collector, "update_config"):
            try:
                collector.update_config(new_config)
                _LOGGER.debug("PowerSourcesCollector config updated")
            except Exception as err:
                _LOGGER.warning("Error updating PowerSourcesCollector config: %s", err)

    _LOGGER.info("Configuration refresh complete")
