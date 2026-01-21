# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""SFML Stats integration for Home Assistant."""
from __future__ import annotations

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
    DAILY_AGGREGATION_HOUR,
    DAILY_AGGREGATION_MINUTE,
    DAILY_AGGREGATION_SECOND,
    FORECAST_MORNING_HOUR,
    FORECAST_MORNING_MINUTE,
    FORECAST_EVENING_HOUR,
    FORECAST_EVENING_MINUTE,
    FORECAST_CHART_HOUR,
    FORECAST_CHART_MINUTE,
    CONF_FORECAST_ENTITY_1,
    CONF_FORECAST_ENTITY_2,
)
from .storage import DataValidator
from .api import async_setup_views, async_setup_websocket
from .services.daily_aggregator import DailyEnergyAggregator
from .services.billing_calculator import BillingCalculator
from .services.monthly_tariff_manager import MonthlyTariffManager
from .services.forecast_comparison_collector import ForecastComparisonCollector

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
                title="SFML Stats - Warnung",
                notification_id=f"{DOMAIN}_no_sources",
            )
        except Exception as err:
            _LOGGER.debug("Could not create persistent notification: %s", err)

    config_path = Path(hass.config.path())
    entry_config = dict(entry.data)
    aggregator = DailyEnergyAggregator(hass, config_path)
    billing_calculator = BillingCalculator(hass, config_path, entry_data=entry_config)
    monthly_tariff_manager = MonthlyTariffManager(hass, config_path, entry_data=entry_config)

    # Initialize Power Sources Collector with error handling
    from .power_sources_collector import PowerSourcesCollector
    power_sources_path = config_path / "sfml_stats" / "data"
    power_sources_collector = PowerSourcesCollector(hass, entry_config, power_sources_path)
    try:
        await power_sources_collector.start()
    except Exception as err:
        _LOGGER.error("Failed to start power sources collector: %s", err)
        # Collector is optional, integration can still function

    # Initialize Weather Collector if weather entity is configured
    # Note: WeatherDataCollector is a passive loader that reads from Solar Forecast ML
    # It doesn't need start/stop - it loads data on demand via get_history()
    weather_collector = None
    weather_entity = entry_config.get(CONF_WEATHER_ENTITY)
    if weather_entity:
        try:
            from .weather_collector import WeatherDataCollector
            weather_path = config_path / "sfml_stats_weather"
            weather_collector = WeatherDataCollector(hass, weather_path)
            _LOGGER.info("Weather collector initialized for entity: %s", weather_entity)
        except Exception as err:
            _LOGGER.error("Failed to initialize weather collector: %s", err)
            # Weather collector is optional

    # Initialize Forecast Comparison Collector
    forecast_comparison_collector = ForecastComparisonCollector(hass, config_path)
    _LOGGER.info("Forecast comparison collector initialized")

    hass.data[DOMAIN][entry.entry_id] = {
        "validator": validator,
        "config": entry_config,
        "aggregator": aggregator,
        "billing_calculator": billing_calculator,
        "monthly_tariff_manager": monthly_tariff_manager,
        "power_sources_collector": power_sources_collector,
        "weather_collector": weather_collector,
        "forecast_comparison_collector": forecast_comparison_collector,
    }

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    async def _daily_aggregation_job(now: datetime) -> None:
        """Run daily aggregation job. @zara"""
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

    # Schedule forecast comparison morning job (08:00 - collect forecasts)
    async def _forecast_morning_job(now: datetime) -> None:
        """Run morning forecast collection job. @zara"""
        _LOGGER.info("Starting scheduled morning forecast collection")
        try:
            await forecast_comparison_collector.async_collect_morning_forecasts()
        except Exception as err:
            _LOGGER.error("Morning forecast collection failed: %s", err)

    cancel_forecast_morning_job = async_track_time_change(
        hass,
        _forecast_morning_job,
        hour=FORECAST_MORNING_HOUR,
        minute=FORECAST_MORNING_MINUTE,
        second=0,
    )
    hass.data[DOMAIN][entry.entry_id]["cancel_forecast_morning_job"] = cancel_forecast_morning_job

    _LOGGER.info(
        "Morning forecast collection scheduled for %02d:%02d",
        FORECAST_MORNING_HOUR,
        FORECAST_MORNING_MINUTE,
    )

    # Schedule forecast comparison evening job (23:50 - collect actual production)
    async def _forecast_evening_job(now: datetime) -> None:
        """Run evening actual production collection job. @zara"""
        _LOGGER.info("Starting scheduled evening actual collection")
        try:
            await forecast_comparison_collector.async_collect_evening_actual()
        except Exception as err:
            _LOGGER.error("Evening actual collection failed: %s", err)

    cancel_forecast_evening_job = async_track_time_change(
        hass,
        _forecast_evening_job,
        hour=FORECAST_EVENING_HOUR,
        minute=FORECAST_EVENING_MINUTE,
        second=0,
    )
    hass.data[DOMAIN][entry.entry_id]["cancel_forecast_evening_job"] = cancel_forecast_evening_job

    _LOGGER.info(
        "Evening actual collection scheduled for %02d:%02d",
        FORECAST_EVENING_HOUR,
        FORECAST_EVENING_MINUTE,
    )

    # Schedule forecast comparison chart generation job
    async def _forecast_chart_job(now: datetime) -> None:
        """Generate forecast comparison chart. @zara"""
        # Only generate if external forecasts are configured
        if not entry_config.get(CONF_FORECAST_ENTITY_1) and not entry_config.get(CONF_FORECAST_ENTITY_2):
            _LOGGER.debug("No external forecast entities configured, skipping chart generation")
            return

        _LOGGER.info("Starting scheduled forecast comparison chart generation")
        try:
            from .charts import ForecastComparisonChart
            chart = ForecastComparisonChart(validator)
            await chart.save()
            _LOGGER.info("Forecast comparison chart generated successfully")
        except Exception as err:
            _LOGGER.error("Forecast comparison chart generation failed: %s", err)

    cancel_chart_job = async_track_time_change(
        hass,
        _forecast_chart_job,
        hour=FORECAST_CHART_HOUR,
        minute=FORECAST_CHART_MINUTE,
        second=0,
    )
    hass.data[DOMAIN][entry.entry_id]["cancel_forecast_chart_job"] = cancel_chart_job

    _LOGGER.info(
        "Forecast comparison chart generation scheduled for %02d:%02d",
        FORECAST_CHART_HOUR,
        FORECAST_CHART_MINUTE,
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
        """Run initial aggregation in background. @zara"""
        try:
            await aggregator.async_aggregate_daily()
        except Exception as err:
            _LOGGER.error("Initial aggregation failed: %s", err)

    hass.async_create_background_task(
        _initial_aggregation(),
        f"{DOMAIN}_initial_aggregation",
    )

    # Run initial forecast comparison collection if no data exists
    async def _initial_forecast_collection() -> None:
        """Run initial forecast comparison collection if needed. @zara"""
        import asyncio

        try:
            from .readers.forecast_comparison_reader import ForecastComparisonReader
            reader = ForecastComparisonReader(config_path)

            # Check if data file exists or has less than 7 days
            needs_historical = False

            if not reader.is_available:
                needs_historical = True
                _LOGGER.info("No forecast comparison data found")
            else:
                # Check how many days we have
                comparison_days = await reader.async_get_comparison_days(days=7)
                days_with_data = sum(1 for d in comparison_days if d.has_data)
                if days_with_data < 7:
                    needs_historical = True
                    _LOGGER.info(
                        "Only %d days of forecast data found, will load historical data",
                        days_with_data
                    )

            if needs_historical:
                # Wait 60 seconds for all sensors to initialize after HA restart
                _LOGGER.info("Waiting 60s for sensors to initialize before collecting historical data")
                await asyncio.sleep(60)
                _LOGGER.info("Running historical forecast comparison collection")
                await forecast_comparison_collector.async_collect_historical(days=7)
            else:
                _LOGGER.debug("Forecast comparison data complete, skipping initial collection")
        except Exception as err:
            _LOGGER.error("Initial forecast comparison collection failed: %s", err)

    hass.async_create_background_task(
        _initial_forecast_collection(),
        f"{DOMAIN}_initial_forecast_collection",
    )

    _LOGGER.info(
        "%s successfully set up. Export path: %s",
        NAME,
        validator.export_base_path
    )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry. @zara"""
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

    if "cancel_forecast_morning_job" in entry_data:
        try:
            entry_data["cancel_forecast_morning_job"]()
            _LOGGER.debug("Morning forecast collection job cancelled")
        except Exception as err:
            _LOGGER.warning("Error cancelling morning forecast job: %s", err)

    if "cancel_forecast_evening_job" in entry_data:
        try:
            entry_data["cancel_forecast_evening_job"]()
            _LOGGER.debug("Evening actual collection job cancelled")
        except Exception as err:
            _LOGGER.warning("Error cancelling evening forecast job: %s", err)

    if "cancel_forecast_chart_job" in entry_data:
        try:
            entry_data["cancel_forecast_chart_job"]()
            _LOGGER.debug("Forecast comparison chart job cancelled")
        except Exception as err:
            _LOGGER.warning("Error cancelling forecast chart job: %s", err)

    # Stop power sources collector
    if "power_sources_collector" in entry_data and entry_data["power_sources_collector"]:
        try:
            await entry_data["power_sources_collector"].stop()
            _LOGGER.debug("Power sources collector stopped")
        except Exception as err:
            _LOGGER.warning("Error stopping power sources collector: %s", err)

    # Weather collector doesn't need stopping (passive loader)
    # Just log if it was present
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

    # Update MonthlyTariffManager
    if "monthly_tariff_manager" in entry_data and entry_data["monthly_tariff_manager"]:
        try:
            entry_data["monthly_tariff_manager"].update_config(new_config)
            entry_data["monthly_tariff_manager"].invalidate_cache()
            _LOGGER.debug("MonthlyTariffManager config updated")
        except Exception as err:
            _LOGGER.warning("Error updating MonthlyTariffManager config: %s", err)

    # ForecastComparisonCollector doesn't need config update
    # It reads config fresh from hass.data on each collection

    _LOGGER.info("Configuration refresh complete")
