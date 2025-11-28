"""Electricity Price Service - aWATTar Integration V10.0.0 @zara

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

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import aiohttp

_LOGGER = logging.getLogger(__name__)

AWATTAR_API_URL = {
    "AT": "https://api.awattar.at/v1/marketdata",
    "DE": "https://api.awattar.de/v1/marketdata",
}

class ElectricityPriceService:
    """Fetch electricity prices from aWATTar API (free, no registration)"""

    def __init__(self, api_key: Optional[str] = None, country: str = "DE"):
        """Initialize the electricity price service Args: api_key: Not needed for aWATTar (kept for compatibility) country: Country code (DE or AT) @zara"""
        self.country = country.upper()
        self.api_url = AWATTAR_API_URL.get(self.country)

        if not self.api_url:
            _LOGGER.warning(f"Unsupported country: {self.country}, falling back to DE")
            self.country = "DE"
            self.api_url = AWATTAR_API_URL["DE"]

        self._price_cache: Dict[str, List[Dict]] = {}
        self._last_update: Optional[datetime] = None

        _LOGGER.info(f"ElectricityPriceService initialized for {self.country} using aWATTar API")

    async def fetch_day_ahead_prices(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> Optional[Dict[str, List[Dict]]]:
        """Fetch day-ahead prices from aWATTar API Args: start_date: Start date (defaults to today 00:00 UTC) end_date: End date (defaults to tomorrow 23:59 UTC) Returns: Dictionary with price data or None on error Format: { 'prices': [ {'timestamp': datetime, 'price': float, 'hour': int}, ... ], 'currency': 'EUR', 'unit': 'Cent/kWh', 'country': 'DE' }"""
        try:

            if start_date is None:
                start_date = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
            if end_date is None:
                end_date = start_date + timedelta(days=2)

            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)

            url = f"{self.api_url}?start={start_ts}&end={end_ts}"

            _LOGGER.debug(
                f"Fetching aWATTar prices for {self.country} from {start_date} to {end_date}"
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        _LOGGER.error(f"aWATTar API error: HTTP {response.status}")
                        return None

                    data = await response.json()

                    prices = self._parse_awattar_response(data)

                    if not prices:
                        _LOGGER.warning("No price data received from aWATTar API")
                        return None

                    result = {
                        "prices": prices,
                        "currency": "EUR",
                        "unit": "Cent/kWh",
                        "country": self.country,
                    }

                    self._price_cache[self.country] = result
                    self._last_update = datetime.now(timezone.utc)

                    _LOGGER.info(f"Successfully fetched {len(prices)} price entries from aWATTar")
                    return result

        except aiohttp.ClientError as e:
            _LOGGER.error(f"Network error fetching aWATTar prices: {e}")
            return None
        except Exception as e:
            _LOGGER.error(f"Unexpected error fetching aWATTar prices: {e}")
            return None

    def _parse_awattar_response(self, data: Dict) -> List[Dict]:
        """Parse aWATTar JSON response Args: data: JSON response from aWATTar API Returns: List of price dictionaries with timestamp, price, and hour @zara"""
        prices = []

        try:
            if "data" not in data:
                _LOGGER.error("Invalid aWATTar response: missing 'data' field")
                return prices

            for entry in data["data"]:

                timestamp = datetime.fromtimestamp(entry["start_timestamp"] / 1000, tz=timezone.utc)

                price_eur_mwh = entry["marketprice"]
                price_cent_kwh = price_eur_mwh / 10

                prices.append(
                    {
                        "timestamp": timestamp,
                        "price": round(price_cent_kwh, 2),
                        "hour": timestamp.hour,
                    }
                )

            prices.sort(key=lambda x: x["timestamp"])

        except Exception as e:
            _LOGGER.error(f"Error parsing aWATTar response: {e}")

        return prices

    def get_current_price(self) -> Optional[float]:
        """Get current electricity price (Cent/kWh) Returns: Current price in Cent/kWh or None @zara"""
        if not self._price_cache or self.country not in self._price_cache:
            return None

        now = datetime.now(timezone.utc)

        prices = self._price_cache[self.country].get("prices", [])

        for price_entry in prices:
            price_time = price_entry["timestamp"]

            if price_time <= now < price_time + timedelta(hours=1):
                return price_entry["price"]

        return None

    def get_price_at_hour(self, hour: int) -> Optional[float]:
        """Get price at specific hour today (in local time) Args: hour: Hour of day in local time (0-23) Returns: Price at specified hour or None @zara"""
        if not self._price_cache or self.country not in self._price_cache:
            return None

        now_local = datetime.now()
        target_local = now_local.replace(hour=hour, minute=0, second=0, microsecond=0)

        local_offset = datetime.now() - datetime.utcnow()
        target_utc = target_local - local_offset

        prices = self._price_cache[self.country].get("prices", [])

        for price_entry in prices:
            price_time = price_entry["timestamp"].replace(tzinfo=None)
            target_utc_naive = target_utc.replace(tzinfo=None)

            if price_time <= target_utc_naive < price_time + timedelta(hours=1):
                return price_entry["price"]

        return None

    def get_average_price_today(self) -> Optional[float]:
        """Calculate average price for today (in local time) Returns: Average price in Cent/kWh or None @zara"""
        if not self._price_cache or self.country not in self._price_cache:
            return None

        today_local = datetime.now().date()

        local_offset = datetime.now() - datetime.utcnow()

        prices = self._price_cache[self.country].get("prices", [])

        today_prices = [
            p["price"]
            for p in prices

            if (p["timestamp"].replace(tzinfo=None) + local_offset).date() == today_local
        ]

        if not today_prices:
            return None

        return round(sum(today_prices) / len(today_prices), 2)

    def get_average_price_week(self) -> Optional[float]:
        """Calculate average price for this week Returns: Average price in Cent/kWh or None @zara"""
        if not self._price_cache or self.country not in self._price_cache:
            return None

        now = datetime.now(timezone.utc)
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        prices = self._price_cache[self.country].get("prices", [])

        week_prices = [p["price"] for p in prices if p["timestamp"] >= week_start]

        if not week_prices:
            return None

        return round(sum(week_prices) / len(week_prices), 2)

    def get_cheapest_hours(self, count: int = 3) -> List[Tuple[int, float]]:
        """Get cheapest hours today (in local time) Args: count: Number of cheapest hours to return Returns: List of (hour, price) tuples sorted by price (hour in local time) @zara"""
        if not self._price_cache or self.country not in self._price_cache:
            return []

        today_local = datetime.now().date()

        local_offset = datetime.now() - datetime.utcnow()

        prices = self._price_cache[self.country].get("prices", [])

        today_prices = [
            (
                (p["timestamp"].replace(tzinfo=None) + local_offset).hour,
                p["price"],
            )
            for p in prices

            if (p["timestamp"].replace(tzinfo=None) + local_offset).date() == today_local
        ]

        if not today_prices:
            return []

        today_prices.sort(key=lambda x: x[1])
        return today_prices[:count]

    def get_most_expensive_hours(self, count: int = 3) -> List[Tuple[int, float]]:
        """Get most expensive hours today (in local time) Args: count: Number of most expensive hours to return Returns: List of (hour, price) tuples sorted by price descending (hour in local time) @zara"""
        if not self._price_cache or self.country not in self._price_cache:
            return []

        today_local = datetime.now().date()

        local_offset = datetime.now() - datetime.utcnow()

        prices = self._price_cache[self.country].get("prices", [])

        today_prices = [
            (
                (p["timestamp"].replace(tzinfo=None) + local_offset).hour,
                p["price"],
            )
            for p in prices

            if (p["timestamp"].replace(tzinfo=None) + local_offset).date() == today_local
        ]

        if not today_prices:
            return []

        today_prices.sort(key=lambda x: x[1], reverse=True)
        return today_prices[:count]

    def should_charge_now(self, threshold_percentile: float = 25.0) -> bool:
        """Determine if current price is good for charging Args: threshold_percentile: Consider prices below this percentile as good (default 25%) Returns: True if current price is below threshold @zara"""
        current_price = self.get_current_price()
        avg_price = self.get_average_price_today()

        if current_price is None or avg_price is None:
            return False

        threshold = avg_price * (threshold_percentile / 100)

        return current_price <= threshold

    def get_last_update(self) -> Optional[datetime]:
        """Get timestamp of last price update Returns: Datetime of last update or None @zara"""
        return self._last_update
