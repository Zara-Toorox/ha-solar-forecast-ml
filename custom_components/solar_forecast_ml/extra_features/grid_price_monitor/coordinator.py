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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    CONF_BATTERY_POWER_SENSOR,
    CONF_COUNTRY,
    CONF_GRID_FEE,
    CONF_MAX_PRICE,
    CONF_PROVIDER_MARKUP,
    CONF_TAXES_FEES,
    CONF_VAT_RATE,
    DB_PATH,
    DEFAULT_COUNTRY,
    DEFAULT_GRID_FEE,
    DEFAULT_MAX_PRICE,
    DEFAULT_PROVIDER_MARKUP,
    DEFAULT_TAXES_FEES,
    DOMAIN,
    PRICE_FETCH_INTERVAL,
    UPDATE_INTERVAL,
    VAT_RATE_AT,
    VAT_RATE_DE,
)
from .core import ElectricityPriceService, BatteryTracker, PriceCalculator
from .storage import DataValidator, GPMDatabaseConnector, PriceCache, HistoryManager, StatisticsStore
from .helpers import GPMLogger, async_setup_gpm_logging

_LOGGER = logging.getLogger(__name__)


class GridPriceMonitorCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator for Grid Price Monitor data updates @zara"""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the coordinator @zara"""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=UPDATE_INTERVAL,
        )

        self.entry = entry
        self._country = entry.data.get(CONF_COUNTRY, DEFAULT_COUNTRY)
        self._grid_fee = entry.data.get(CONF_GRID_FEE, DEFAULT_GRID_FEE)
        self._taxes_fees = entry.data.get(CONF_TAXES_FEES, DEFAULT_TAXES_FEES)
        self._provider_markup = entry.data.get(CONF_PROVIDER_MARKUP, DEFAULT_PROVIDER_MARKUP)
        self._max_price = entry.data.get(CONF_MAX_PRICE, DEFAULT_MAX_PRICE)

        # Get VAT rate from config, fallback to country default
        default_vat = VAT_RATE_AT if self._country == "AT" else VAT_RATE_DE
        self._vat_rate = entry.data.get(CONF_VAT_RATE, default_vat)

        # Initialize price calculator
        self._calculator = PriceCalculator(
            vat_rate=self._vat_rate,
            grid_fee=self._grid_fee,
            taxes_fees=self._taxes_fees,
            provider_markup=self._provider_markup,
        )

        self._price_service = ElectricityPriceService(self._country)
        self._last_price_fetch: datetime | None = None

        # Battery tracker
        self._battery_power_sensor = entry.data.get(CONF_BATTERY_POWER_SENSOR, "")
        self._battery_tracker: BatteryTracker | None = None

        # Storage components (initialized in async_initialize_storage)
        self._db_connector: GPMDatabaseConnector | None = None
        self._data_validator: DataValidator | None = None
        self._price_cache: PriceCache | None = None
        self._history_manager: HistoryManager | None = None
        self._statistics_store: StatisticsStore | None = None
        self._gpm_logger: GPMLogger | None = None
        self._storage_initialized = False

    @property
    def vat_factor(self) -> float:
        """Get VAT factor (e.g. 1.19 for 19%) @zara"""
        return self._calculator.vat_factor

    @property
    def total_markup(self) -> float:
        """Calculate total markup (grid fee + taxes + provider) - all gross @zara"""
        return self._calculator.total_markup

    @property
    def max_price(self) -> float:
        """Get the configured max price threshold @zara"""
        return self._max_price

    @property
    def battery_tracker(self) -> BatteryTracker | None:
        """Get the battery tracker instance @zara"""
        return self._battery_tracker

    @property
    def has_battery_sensor(self) -> bool:
        """Check if battery power sensor is configured @zara"""
        return bool(self._battery_power_sensor)

    @property
    def gpm_logger(self) -> GPMLogger | None:
        """Get the GPM logger instance @zara"""
        return self._gpm_logger

    async def async_initialize_storage(self) -> None:
        """Initialize persistent storage components @zara"""
        if self._storage_initialized:
            return

        # Get config path from hass
        config_path = Path(self.hass.config.path())
        gpm_path = config_path / "grid_price_monitor"

        # Initialize database connector
        self._db_connector = GPMDatabaseConnector(DB_PATH)
        await self._db_connector.connect()

        # Initialize data validator (for logs directory + legacy cleanup)
        self._data_validator = DataValidator(gpm_path, self._db_connector, self.hass)
        await self._data_validator.async_validate_structure()

        # Initialize storage components with DB connector
        self._price_cache = PriceCache(self._db_connector)
        self._history_manager = HistoryManager(self._db_connector)
        self._statistics_store = StatisticsStore(self._db_connector)

        # Initialize GPM logger
        self._gpm_logger = await async_setup_gpm_logging(self._data_validator.logs_path, self.hass)

        # Load cached prices if available and valid
        await self._price_cache.async_load()
        if await self._price_cache.async_is_valid():
            cached_prices = await self._price_cache.async_get_prices()
            if cached_prices:
                self._price_service.set_prices_from_cache(cached_prices)
                _LOGGER.debug("Loaded %d prices from cache", len(cached_prices))
                if self._gpm_logger:
                    self._gpm_logger.info("Loaded %d prices from cache", len(cached_prices))

        # Load history
        await self._history_manager.async_load()

        # Load statistics
        await self._statistics_store.async_load()

        self._storage_initialized = True
        _LOGGER.info("GPM storage initialized (database: %s)", DB_PATH)

    async def async_shutdown_storage(self) -> None:
        """Shutdown storage components @zara"""
        if self._gpm_logger:
            self._gpm_logger.shutdown()
            self._gpm_logger = None

        # Close database connection
        if self._db_connector:
            await self._db_connector.close()
            self._db_connector = None

        self._storage_initialized = False

    async def async_setup_battery_tracker(self) -> None:
        """Initialize the battery tracker if configured @zara"""
        if self._battery_power_sensor:
            self._battery_tracker = BatteryTracker(
                self.hass,
                self.entry.entry_id,
                db=self._db_connector,
            )
            await self._battery_tracker.async_setup(self._battery_power_sensor)
            _LOGGER.debug("Battery tracker initialized for sensor: %s", self._battery_power_sensor)

    async def async_shutdown_battery_tracker(self) -> None:
        """Shutdown the battery tracker @zara"""
        if self._battery_tracker:
            await self._battery_tracker.async_unload()
            self._battery_tracker = None

    def calculate_total_price(self, spot_price_net: float) -> float:
        """Calculate total gross price from net spot price @zara

        Formula: (Spot_net x VAT_factor) + Grid_fee + Taxes + Provider_markup

        Args:
            spot_price_net: Net spot price in ct/kWh (from aWATTar)

        Returns:
            Total gross price in ct/kWh
        """
        return self._calculator.calculate_total_price(spot_price_net)

    def calculate_spot_from_total(self, total_price: float) -> float:
        """Reverse calculate: get net spot price from total gross price @zara

        Used for calibration and threshold calculations.

        Args:
            total_price: Total gross price in ct/kWh

        Returns:
            Net spot price in ct/kWh
        """
        return self._calculator.calculate_spot_from_total(total_price)

    def update_config(self) -> None:
        """Update configuration values from entry @zara"""
        self._country = self.entry.data.get(CONF_COUNTRY, DEFAULT_COUNTRY)
        self._grid_fee = self.entry.data.get(CONF_GRID_FEE, DEFAULT_GRID_FEE)
        self._taxes_fees = self.entry.data.get(CONF_TAXES_FEES, DEFAULT_TAXES_FEES)
        self._provider_markup = self.entry.data.get(CONF_PROVIDER_MARKUP, DEFAULT_PROVIDER_MARKUP)
        self._max_price = self.entry.data.get(CONF_MAX_PRICE, DEFAULT_MAX_PRICE)
        self._battery_power_sensor = self.entry.data.get(CONF_BATTERY_POWER_SENSOR, "")

        # Get VAT rate from config, fallback to country default
        default_vat = VAT_RATE_AT if self._country == "AT" else VAT_RATE_DE
        self._vat_rate = self.entry.data.get(CONF_VAT_RATE, default_vat)

        # Update calculator
        self._calculator.update_config(
            vat_rate=self._vat_rate,
            grid_fee=self._grid_fee,
            taxes_fees=self._taxes_fees,
            provider_markup=self._provider_markup,
        )

        _LOGGER.debug(
            "Configuration updated: markup=%.2f, vat=%d%%, max_price=%.2f",
            self.total_markup,
            self._vat_rate,
            self._max_price,
        )

        # Log config change
        if self._gpm_logger:
            self._gpm_logger.log_config_change({
                "vat_rate": self._vat_rate,
                "grid_fee": self._grid_fee,
                "taxes_fees": self._taxes_fees,
                "provider_markup": self._provider_markup,
                "max_price": self._max_price,
            })

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch and process price data @zara"""
        now = datetime.now(timezone.utc)

        # Fetch new prices if needed (every hour or if no data)
        should_fetch = (
            self._last_price_fetch is None
            or (now - self._last_price_fetch) >= PRICE_FETCH_INTERVAL
            or not self._price_service.has_data
        )

        if should_fetch:
            _LOGGER.debug("Fetching new price data from aWATTar API")
            prices = await self._price_service.fetch_day_ahead_prices()

            if prices is None:
                if not self._price_service.has_data:
                    # Log API failure
                    if self._gpm_logger:
                        self._gpm_logger.log_api_fetch(
                            success=False,
                            country=self._country,
                            error="No cached data available"
                        )
                    raise UpdateFailed("Failed to fetch price data and no cached data available")
                _LOGGER.warning("Failed to fetch new prices, using cached data")
                if self._gpm_logger:
                    self._gpm_logger.log_api_fetch(
                        success=False,
                        country=self._country,
                        error="Using cached data"
                    )
            else:
                self._last_price_fetch = now

                # Log successful fetch
                if self._gpm_logger:
                    self._gpm_logger.log_api_fetch(
                        success=True,
                        country=self._country,
                        entries=len(prices)
                    )

                # Enrich prices with total_price before caching
                enriched_prices = self._enrich_prices_with_total(prices)

                # Save to cache (with total_price included)
                if self._price_cache:
                    await self._price_cache.async_save(enriched_prices, self._country)

                # Add to history (with total_price for each entry)
                if self._history_manager:
                    await self._history_manager.async_add_prices(enriched_prices)

        # Update config in case it changed
        self.update_config()

        # Calculate current values
        spot_price_net = self._price_service.get_current_price()
        spot_price_next_net = self._price_service.get_next_hour_price()

        total_price = None
        total_price_next = None
        spot_price_gross = None
        spot_price_next_gross = None

        if spot_price_net is not None:
            spot_price_gross = round(spot_price_net * self.vat_factor, 2)
            total_price = self.calculate_total_price(spot_price_net)

        if spot_price_next_net is not None:
            spot_price_next_gross = round(spot_price_next_net * self.vat_factor, 2)
            total_price_next = self.calculate_total_price(spot_price_next_net)

        # Determine if current price is cheap
        is_cheap = total_price is not None and total_price < self._max_price

        # Log price update
        if self._gpm_logger and total_price is not None:
            self._gpm_logger.log_price_update(
                spot_price=spot_price_gross,
                total_price=total_price,
                is_cheap=is_cheap
            )

        # Get statistics
        cheapest = self._price_service.get_cheapest_hour_today()
        most_expensive = self._price_service.get_most_expensive_hour_today()
        average_net = self._price_service.get_average_price_today()

        # Calculate total for cheapest/most expensive
        cheapest_total = None
        most_expensive_total = None
        average_total = None

        if cheapest:
            cheapest_total = self.calculate_total_price(cheapest["price"])
        if most_expensive:
            most_expensive_total = self.calculate_total_price(most_expensive["price"])

        # Calculate average of TOTAL prices (not total of average net)
        # This is more accurate as it represents what users actually pay on average
        today_prices_for_avg = self._price_service.get_today_prices()
        if today_prices_for_avg:
            total_prices = [self.calculate_total_price(p["price"]) for p in today_prices_for_avg]
            average_total = round(sum(total_prices) / len(total_prices), 2)

            # Update statistics store
            if self._statistics_store:
                await self._statistics_store.async_update_daily_average(
                    date=now.date().isoformat(),
                    average_net=average_net,
                    average_total=average_total,
                    min_price=cheapest["price"] if cheapest else None,
                    max_price=most_expensive["price"] if most_expensive else None,
                )

                # Update monthly summary
                cheap_hours_count = sum(
                    1 for p in today_prices_for_avg
                    if self.calculate_total_price(p["price"]) < self._max_price
                )
                await self._statistics_store.async_update_monthly_summary(
                    year=now.year,
                    month=now.month,
                    average_price=average_total,
                    total_cheap_hours=cheap_hours_count,
                    country=self._country,
                )

        # Get forecast data (reuse today_prices_for_avg if available)
        today_prices = today_prices_for_avg if today_prices_for_avg else self._price_service.get_today_prices()
        tomorrow_prices = self._price_service.get_tomorrow_prices()

        # Calculate total prices for forecasts
        today_forecast = self._build_forecast_with_total(today_prices)
        tomorrow_forecast = self._build_forecast_with_total(tomorrow_prices)

        # Get cheap hours (based on total price < max_price)
        cheap_hours_today = self._get_cheap_hours_for_date(today_prices)
        cheap_hours_tomorrow = self._get_cheap_hours_for_date(tomorrow_prices)

        # Find next cheap hour
        next_cheap = self._find_next_cheap_hour()

        # Calculate price trend
        price_trend = self._calculate_trend(total_price, total_price_next)

        data = {
            # Current prices - NET (from exchange)
            "spot_price_net": spot_price_net,
            "spot_price_next_hour_net": spot_price_next_net,
            # Current prices - GROSS (with VAT)
            "spot_price": spot_price_gross,
            "spot_price_next_hour": spot_price_next_gross,
            # Total prices (what customer pays)
            "total_price": total_price,
            "total_price_next_hour": total_price_next,
            # Status
            "is_cheap": is_cheap,
            # Statistics
            "cheapest_hour_today": cheapest["hour"] if cheapest else None,
            "cheapest_price_today": cheapest_total,
            "cheapest_spot_net": cheapest["price"] if cheapest else None,
            "most_expensive_hour_today": most_expensive["hour"] if most_expensive else None,
            "most_expensive_price_today": most_expensive_total,
            "most_expensive_spot_net": most_expensive["price"] if most_expensive else None,
            "average_price_today": average_total,
            "average_spot_net": average_net,
            # Forecasts (as attributes)
            "forecast_today": today_forecast,
            "forecast_tomorrow": tomorrow_forecast,
            # Cheap hours
            "cheap_hours_today": cheap_hours_today,
            "cheap_hours_tomorrow": cheap_hours_tomorrow,
            "next_cheap_hour": next_cheap["hour"] if next_cheap else None,
            "next_cheap_timestamp": next_cheap["timestamp"].isoformat() if next_cheap else None,
            # Trend
            "price_trend": price_trend,
            # Meta
            "last_update": now.isoformat(),
            "data_source": "aWATTar",
            "country": self._country,
            # Config values for reference
            "markup_total": self.total_markup,
            "vat_rate": self._vat_rate,
            "vat_factor": self.vat_factor,
            "max_price_threshold": self._max_price,
        }

        # Write current price to DB for external integrations
        if self._db_connector and total_price is not None:
            try:
                current_hour = datetime.now().astimezone().strftime("%Y-%m-%dT%H:00:00%z")
                await self._db_connector.execute(
                    """INSERT INTO GPM_current_price
                       (id, timestamp, spot_price_net, spot_price_gross, total_price,
                        price_next_hour, is_cheap, average_today,
                        cheapest_today, most_expensive_today, country, last_updated)
                       VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                           timestamp = excluded.timestamp,
                           spot_price_net = excluded.spot_price_net,
                           spot_price_gross = excluded.spot_price_gross,
                           total_price = excluded.total_price,
                           price_next_hour = excluded.price_next_hour,
                           is_cheap = excluded.is_cheap,
                           average_today = excluded.average_today,
                           cheapest_today = excluded.cheapest_today,
                           most_expensive_today = excluded.most_expensive_today,
                           country = excluded.country,
                           last_updated = excluded.last_updated""",
                    (
                        current_hour,
                        spot_price_net,
                        spot_price_gross,
                        total_price,
                        total_price_next,
                        1 if is_cheap else 0,
                        average_total,
                        cheapest_total,
                        most_expensive_total,
                        self._country,
                        now.isoformat(),
                    ),
                )
            except Exception as err:
                _LOGGER.debug("Failed to write current price to DB: %s", err)

        # Add battery power directly from configured sensor
        if self._battery_power_sensor:
            battery_power = self._get_battery_power()
            data["battery_power"] = battery_power

            # Add tracker statistics if available
            if self._battery_tracker:
                battery_stats = self._battery_tracker.get_statistics()
                # Override with direct sensor value
                battery_stats["battery_power"] = battery_power
                data.update(battery_stats)

                # Log battery charging
                if self._gpm_logger:
                    self._gpm_logger.log_battery_charge(
                        power_w=battery_power,
                        energy_today_kwh=battery_stats.get("battery_charged_today", 0),
                        energy_month_kwh=battery_stats.get("battery_charged_month", 0),
                    )

                # Update battery totals in statistics
                if self._statistics_store:
                    await self._statistics_store.async_update_battery_totals(
                        today_kwh=battery_stats.get("battery_charged_today", 0),
                        week_kwh=battery_stats.get("battery_charged_week", 0),
                        month_kwh=battery_stats.get("battery_charged_month", 0),
                    )

        return data

    def _get_battery_power(self) -> float:
        """Get current battery power directly from sensor @zara"""
        if not self._battery_power_sensor:
            return 0.0

        state = self.hass.states.get(self._battery_power_sensor)
        if state is None or state.state in ("unknown", "unavailable"):
            return 0.0

        try:
            return float(state.state)
        except (ValueError, TypeError):
            _LOGGER.warning("Invalid battery power value: %s", state.state)
            return 0.0

    def _enrich_prices_with_total(
        self, prices: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Enrich price entries with calculated total_price @zara

        Adds the total_price field (what the customer actually pays) to each
        price entry before caching. This uses the current config values
        (VAT, grid fee, taxes, provider markup).

        IMPORTANT: Converts timestamp to LOCAL time to prevent users from
        triggering actions at wrong times (e.g., EV charging).

        Args:
            prices: List of price entries from API

        Returns:
            Enriched list with total_price and local timestamp
        """
        enriched = []
        for entry in prices:
            enriched_entry = entry.copy()
            spot_net = entry.get("price", 0)
            # Calculate total price: (Spot x VAT) + Grid + Taxes + Provider
            enriched_entry["total_price"] = self.calculate_total_price(spot_net)

            # Convert timestamp to LOCAL time for cache
            # This is critical for automation triggers!
            ts = entry.get("timestamp")
            if ts is not None:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                # Convert UTC to local timezone
                local_ts = ts.astimezone()
                enriched_entry["timestamp"] = local_ts.isoformat()

            enriched.append(enriched_entry)
        return enriched

    def _build_forecast_with_total(
        self, prices: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build forecast list with total prices @zara"""
        forecast = []
        for entry in prices:
            spot_net = entry["price"]
            spot_gross = round(spot_net * self.vat_factor, 2)
            total = self.calculate_total_price(spot_net)

            forecast.append({
                "hour": entry["hour"],
                "spot_price_net": spot_net,
                "spot_price": spot_gross,
                "total_price": total,
                "is_cheap": total < self._max_price,
            })
        return forecast

    def _get_cheap_hours_for_date(
        self, prices: list[dict[str, Any]]
    ) -> list[int]:
        """Get list of cheap hours for a date @zara"""
        cheap_hours = []
        for p in prices:
            total = self.calculate_total_price(p["price"])
            if total < self._max_price:
                cheap_hours.append(p["hour"])
        return cheap_hours

    def _find_next_cheap_hour(self) -> dict[str, Any] | None:
        """Find the next hour where total price is below max threshold @zara"""
        now = datetime.now(timezone.utc)

        # Use public method instead of private _price_cache
        for entry in self._price_service.get_all_prices():
            entry_ts = entry["timestamp"]
            if isinstance(entry_ts, str):
                entry_ts = datetime.fromisoformat(entry_ts)
            if entry_ts >= now:
                total = self.calculate_total_price(entry["price"])
                if total < self._max_price:
                    return {
                        "hour": entry["hour"],
                        "timestamp": entry_ts,
                        "total_price": total,
                    }
        return None

    def _calculate_trend(
        self, current: float | None, next_hour: float | None
    ) -> str:
        """Calculate price trend based on total prices @zara"""
        if current is None or next_hour is None:
            return "unknown"

        diff = next_hour - current
        if diff > 1:
            return "rising"
        elif diff < -1:
            return "falling"
        else:
            return "stable"
