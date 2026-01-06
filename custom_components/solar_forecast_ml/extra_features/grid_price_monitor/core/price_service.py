# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from ..const import (
    API_TIMEOUT,
    AWATTAR_API_URL_AT,
    AWATTAR_API_URL_DE,
)

_LOGGER = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


class ElectricityPriceService:
    """Service for fetching electricity spot prices from aWATTar API @zara"""

    def __init__(self, country: str = "DE") -> None:
        """Initialize the electricity price service @zara"""
        self.country = country.upper()
        self.api_url = AWATTAR_API_URL_DE if self.country == "DE" else AWATTAR_API_URL_AT
        self._price_cache: list[dict[str, Any]] = []
        self._last_update: datetime | None = None

    async def fetch_day_ahead_prices(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]] | None:
        """Fetch day-ahead prices from aWATTar API @zara

        Args:
            start_date: Start date for price data (default: today 00:00 UTC)
            end_date: End date for price data (default: 2 days from start)

        Returns:
            List of price dictionaries or None on error
        """
        if start_date is None:
            start_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        if end_date is None:
            end_date = start_date + timedelta(days=2)

        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        url = f"{self.api_url}?start={start_ts}&end={end_ts}"

        # Retry logic for resilience
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                timeout = aiohttp.ClientTimeout(total=API_TIMEOUT)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as response:
                        if response.status != 200:
                            _LOGGER.warning(
                                "aWATTar API returned status %s for %s (attempt %d/%d)",
                                response.status,
                                self.country,
                                attempt + 1,
                                MAX_RETRIES,
                            )
                            if attempt < MAX_RETRIES - 1:
                                await asyncio.sleep(RETRY_DELAY_SECONDS)
                                continue
                            return None

                        data = await response.json()
                        prices = self._parse_awattar_response(data)

                        if prices:
                            self._price_cache = prices
                            self._last_update = datetime.now(timezone.utc)
                            _LOGGER.debug(
                                "Fetched %d price entries for %s",
                                len(prices),
                                self.country,
                            )

                        return prices

            except asyncio.TimeoutError as err:
                last_error = err
                _LOGGER.warning(
                    "Timeout fetching prices (attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    err,
                )
            except aiohttp.ClientError as err:
                last_error = err
                _LOGGER.warning(
                    "Network error fetching prices (attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    err,
                )
            except Exception as err:
                last_error = err
                _LOGGER.error("Unexpected error fetching prices: %s", err, exc_info=True)
                return None  # Don't retry on unexpected errors

            # Wait before retrying
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY_SECONDS)

        _LOGGER.error(
            "Failed to fetch prices after %d attempts. Last error: %s",
            MAX_RETRIES,
            last_error,
        )
        return None

    def _parse_awattar_response(
        self, response: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Parse aWATTar API response into standardized format @zara

        Args:
            response: Raw API response

        Returns:
            List of parsed price entries
        """
        prices = []

        if "data" not in response:
            _LOGGER.error("Invalid aWATTar response: missing 'data' field")
            return prices

        for entry in response["data"]:
            try:
                timestamp_ms = entry.get("start_timestamp")
                market_price = entry.get("marketprice")

                if timestamp_ms is None or market_price is None:
                    continue

                # Store timestamp in UTC for accurate time comparisons
                timestamp_utc = datetime.fromtimestamp(
                    timestamp_ms / 1000, tz=timezone.utc
                )

                # Convert to local timezone for display values
                timestamp_local = timestamp_utc.astimezone()

                # Convert EUR/MWh to ct/kWh (divide by 10)
                price_ct_kwh = round(market_price / 10, 2)

                prices.append({
                    "timestamp": timestamp_utc,  # Keep UTC for comparisons
                    "price": price_ct_kwh,
                    "hour": timestamp_local.hour,  # Local hour for display
                    "date": timestamp_local.date().isoformat(),  # Local date
                })

            except (KeyError, TypeError, ValueError) as err:
                _LOGGER.warning("Error parsing price entry: %s", err)
                continue

        return prices

    def set_prices_from_cache(self, prices: list[dict[str, Any]]) -> None:
        """Set prices from external cache @zara

        Args:
            prices: List of price entries from cache
        """
        self._price_cache = prices
        _LOGGER.debug("Loaded %d prices from cache", len(prices))

    def get_current_price(self) -> float | None:
        """Get the current hour's spot price @zara"""
        now = datetime.now(timezone.utc)
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        for entry in self._price_cache:
            entry_ts = entry["timestamp"]
            if isinstance(entry_ts, str):
                entry_ts = datetime.fromisoformat(entry_ts)
            if entry_ts == current_hour:
                return entry["price"]

        return None

    def get_next_hour_price(self) -> float | None:
        """Get the next hour's spot price @zara"""
        now = datetime.now(timezone.utc)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

        for entry in self._price_cache:
            entry_ts = entry["timestamp"]
            if isinstance(entry_ts, str):
                entry_ts = datetime.fromisoformat(entry_ts)
            if entry_ts == next_hour:
                return entry["price"]

        return None

    def get_prices_for_date(self, date: datetime) -> list[dict[str, Any]]:
        """Get all prices for a specific date in LOCAL timezone @zara

        The date comparison is done in the local timezone to match user expectations.
        """
        # Convert to local timezone for correct date comparison
        if date.tzinfo is None or date.tzinfo == timezone.utc:
            local_date = date.astimezone().date()
        else:
            local_date = date.date()

        target_date = local_date.isoformat()

        result = []
        for p in self._price_cache:
            # Convert UTC timestamp to local date for comparison
            entry_ts = p["timestamp"]
            if isinstance(entry_ts, str):
                entry_ts = datetime.fromisoformat(entry_ts)
            # Convert to local timezone and get date
            entry_local_date = entry_ts.astimezone().date().isoformat()
            if entry_local_date == target_date:
                result.append(p)

        return result

    def get_today_prices(self) -> list[dict[str, Any]]:
        """Get all prices for today (local timezone) @zara"""
        # Use local time for "today"
        local_now = datetime.now().astimezone()
        return self.get_prices_for_date(local_now)

    def get_tomorrow_prices(self) -> list[dict[str, Any]]:
        """Get all prices for tomorrow (local timezone) @zara"""
        local_now = datetime.now().astimezone()
        tomorrow = local_now + timedelta(days=1)
        return self.get_prices_for_date(tomorrow)

    def get_cheapest_hour_today(self) -> dict[str, Any] | None:
        """Get the cheapest hour for today @zara"""
        today_prices = self.get_today_prices()
        if not today_prices:
            return None
        return min(today_prices, key=lambda x: x["price"])

    def get_most_expensive_hour_today(self) -> dict[str, Any] | None:
        """Get the most expensive hour for today @zara"""
        today_prices = self.get_today_prices()
        if not today_prices:
            return None
        return max(today_prices, key=lambda x: x["price"])

    def get_average_price_today(self) -> float | None:
        """Get the average price for today @zara"""
        today_prices = self.get_today_prices()
        if not today_prices:
            return None
        return round(sum(p["price"] for p in today_prices) / len(today_prices), 2)

    def get_cheap_hours(
        self, max_price: float, date: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Get all hours where price is below max_price @zara"""
        if date is None:
            prices = self._price_cache
        else:
            prices = self.get_prices_for_date(date)

        return [p for p in prices if p["price"] < max_price]

    def get_next_cheap_hour(self, max_price: float) -> dict[str, Any] | None:
        """Get the next hour where price is below max_price @zara"""
        now = datetime.now(timezone.utc)

        future_cheap = []
        for p in self._price_cache:
            entry_ts = p["timestamp"]
            if isinstance(entry_ts, str):
                entry_ts = datetime.fromisoformat(entry_ts)
            if entry_ts >= now and p["price"] < max_price:
                future_cheap.append(p)

        if not future_cheap:
            return None

        return min(future_cheap, key=lambda x: x["timestamp"] if isinstance(x["timestamp"], datetime) else datetime.fromisoformat(x["timestamp"]))

    def get_all_prices(self) -> list[dict[str, Any]]:
        """Get all cached prices @zara"""
        return self._price_cache.copy()

    @property
    def last_update(self) -> datetime | None:
        """Return the last update timestamp @zara"""
        return self._last_update

    @property
    def has_data(self) -> bool:
        """Check if price data is available @zara"""
        return len(self._price_cache) > 0
