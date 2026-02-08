# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Price data reader for SFML Stats. @zara"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any

import aiofiles

from ..const import (
    GRID_PRICE_MONITOR_DATA,
    GRID_PRICE_HISTORY,
    GRID_STATISTICS,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class HourlyPrice:
    """Hourly electricity price. @zara"""

    timestamp: datetime
    hour: int
    price_net: float

    @property
    def date(self) -> date:
        """Return the date. @zara"""
        return self.timestamp.date()


@dataclass
class DailyPriceStats:
    """Daily price statistics. @zara"""

    date: date
    average_net: float
    average_total: float
    min_price: float
    max_price: float

    @property
    def price_spread(self) -> float:
        """Price spread between min and max. @zara"""
        return self.max_price - self.min_price


@dataclass
class PriceExtremes:
    """Price extremes. @zara"""

    all_time_low: float
    all_time_high: float
    all_time_low_date: date | None
    all_time_high_date: date | None


@dataclass
class BatteryStats:
    """Battery charging statistics. @zara"""

    today_kwh: float
    week_kwh: float
    month_kwh: float


class PriceDataReader:
    """Reads and parses data from Grid Price Monitor. @zara"""

    def __init__(self, config_path: Path) -> None:
        """Initialize the price data reader. @zara"""
        self._config_path = config_path
        self._data_path = config_path / GRID_PRICE_MONITOR_DATA

    @property
    def is_available(self) -> bool:
        """Check if Grid Price Monitor data is available. @zara"""
        return self._data_path.exists()

    async def _read_json_file(self, file_path: Path) -> dict | None:
        """Read a JSON file asynchronously. @zara"""
        if not file_path.exists():
            _LOGGER.debug("File not found: %s", file_path)
            return None

        try:
            async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        except json.JSONDecodeError as err:
            _LOGGER.error("JSON parsing error in %s: %s", file_path, err)
            return None
        except Exception as err:
            _LOGGER.error("Error reading %s: %s", file_path, err)
            return None

    async def async_get_hourly_prices(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[HourlyPrice]:
        """Read the hourly prices. @zara"""
        file_path = self._data_path / GRID_PRICE_HISTORY
        data = await self._read_json_file(file_path)

        if not data or "prices" not in data:
            return []

        prices: list[HourlyPrice] = []

        for raw in data["prices"]:
            try:
                timestamp = datetime.fromisoformat(
                    raw["timestamp"].replace("Z", "+00:00")
                )
                price_date = timestamp.date()

                if start_date and price_date < start_date:
                    continue
                if end_date and price_date > end_date:
                    continue

                price = HourlyPrice(
                    timestamp=timestamp,
                    hour=raw.get("hour", timestamp.hour),
                    price_net=raw.get("price_net", 0.0),
                )
                prices.append(price)

            except Exception as err:
                _LOGGER.warning("Error parsing price: %s", err)
                continue

        prices.sort(key=lambda x: x.timestamp)

        return prices

    async def async_get_daily_stats(self) -> list[DailyPriceStats]:
        """Read the daily price statistics. @zara"""
        file_path = self._data_path / GRID_STATISTICS
        data = await self._read_json_file(file_path)

        if not data or "daily_averages" not in data:
            return []

        stats: list[DailyPriceStats] = []

        for raw in data["daily_averages"]:
            try:
                stats.append(DailyPriceStats(
                    date=date.fromisoformat(raw["date"]),
                    average_net=raw.get("average_net", 0.0),
                    average_total=raw.get("average_total", 0.0),
                    min_price=raw.get("min_price", 0.0),
                    max_price=raw.get("max_price", 0.0),
                ))
            except Exception as err:
                _LOGGER.warning("Error parsing daily stats: %s", err)
                continue

        stats.sort(key=lambda x: x.date)

        return stats

    async def async_get_price_extremes(self) -> PriceExtremes | None:
        """Read the price extremes. @zara"""
        file_path = self._data_path / GRID_STATISTICS
        data = await self._read_json_file(file_path)

        if not data or "price_extremes" not in data:
            return None

        try:
            extremes = data["price_extremes"]

            low_date = None
            high_date = None
            if extremes.get("all_time_low_date"):
                low_date = date.fromisoformat(extremes["all_time_low_date"])
            if extremes.get("all_time_high_date"):
                high_date = date.fromisoformat(extremes["all_time_high_date"])

            return PriceExtremes(
                all_time_low=extremes.get("all_time_low", 0.0),
                all_time_high=extremes.get("all_time_high", 0.0),
                all_time_low_date=low_date,
                all_time_high_date=high_date,
            )
        except Exception as err:
            _LOGGER.error("Error parsing price extremes: %s", err)
            return None

    async def async_get_battery_stats(self) -> BatteryStats | None:
        """Read the battery statistics. @zara"""
        file_path = self._data_path / GRID_STATISTICS
        data = await self._read_json_file(file_path)

        if not data or "battery_totals" not in data:
            return None

        try:
            battery = data["battery_totals"]
            return BatteryStats(
                today_kwh=battery.get("today_kwh", 0.0),
                week_kwh=battery.get("week_kwh", 0.0),
                month_kwh=battery.get("month_kwh", 0.0),
            )
        except Exception as err:
            _LOGGER.error("Error parsing battery stats: %s", err)
            return None

    async def async_get_prices_for_date(self, target_date: date) -> list[HourlyPrice]:
        """Get all prices for a specific date. @zara"""
        all_prices = await self.async_get_hourly_prices(
            start_date=target_date,
            end_date=target_date,
        )
        return [p for p in all_prices if p.date == target_date]

    async def async_get_weekly_stats(self, year: int, week: int) -> dict[str, Any]:
        """Calculate statistics for a specific calendar week. @zara"""
        all_prices = await self.async_get_hourly_prices()

        week_prices = [
            p for p in all_prices
            if p.timestamp.isocalendar()[0] == year
            and p.timestamp.isocalendar()[1] == week
        ]

        if not week_prices:
            return {"week": week, "year": year, "data_available": False}

        prices_by_hour: dict[int, list[float]] = {h: [] for h in range(24)}
        for price in week_prices:
            prices_by_hour[price.hour].append(price.price_net)

        avg_by_hour = {
            hour: sum(prices) / len(prices) if prices else 0.0
            for hour, prices in prices_by_hour.items()
        }

        sorted_hours = sorted(avg_by_hour.items(), key=lambda x: x[1])
        cheapest_hours = [h for h, _ in sorted_hours[:3]]
        expensive_hours = [h for h, _ in sorted_hours[-3:]]

        all_prices_values = [p.price_net for p in week_prices]

        return {
            "week": week,
            "year": year,
            "data_available": True,
            "hours_count": len(week_prices),
            "average_price": round(sum(all_prices_values) / len(all_prices_values), 2),
            "min_price": round(min(all_prices_values), 2),
            "max_price": round(max(all_prices_values), 2),
            "price_spread": round(max(all_prices_values) - min(all_prices_values), 2),
            "cheapest_hours": cheapest_hours,
            "expensive_hours": expensive_hours,
            "avg_price_by_hour": {h: round(v, 2) for h, v in avg_by_hour.items()},
            "hourly_prices": week_prices,
        }

    async def async_get_monthly_stats(self, year: int, month: int) -> dict[str, Any]:
        """Calculate statistics for a specific month. @zara"""
        all_prices = await self.async_get_hourly_prices()

        month_prices = [
            p for p in all_prices
            if p.timestamp.year == year and p.timestamp.month == month
        ]

        if not month_prices:
            return {"month": month, "year": year, "data_available": False}

        prices_by_date: dict[date, list[float]] = {}
        for price in month_prices:
            if price.date not in prices_by_date:
                prices_by_date[price.date] = []
            prices_by_date[price.date].append(price.price_net)

        daily_stats = []
        for day, prices in sorted(prices_by_date.items()):
            daily_stats.append({
                "date": day.isoformat(),
                "avg": round(sum(prices) / len(prices), 2),
                "min": round(min(prices), 2),
                "max": round(max(prices), 2),
            })

        all_prices_values = [p.price_net for p in month_prices]

        daily_avgs = [(d, sum(p) / len(p)) for d, p in prices_by_date.items()]
        cheapest_day = min(daily_avgs, key=lambda x: x[1])
        expensive_day = max(daily_avgs, key=lambda x: x[1])

        return {
            "month": month,
            "year": year,
            "data_available": True,
            "days_count": len(prices_by_date),
            "hours_count": len(month_prices),
            "average_price": round(sum(all_prices_values) / len(all_prices_values), 2),
            "min_price": round(min(all_prices_values), 2),
            "max_price": round(max(all_prices_values), 2),
            "cheapest_day": {
                "date": cheapest_day[0].isoformat(),
                "avg_price": round(cheapest_day[1], 2),
            },
            "expensive_day": {
                "date": expensive_day[0].isoformat(),
                "avg_price": round(expensive_day[1], 2),
            },
            "daily_stats": daily_stats,
            "hourly_prices": month_prices,
        }

    async def async_get_price_at_hour(
        self,
        target_date: date,
        hour: int,
    ) -> float | None:
        """Get the price for a specific hour. @zara"""
        prices = await self.async_get_prices_for_date(target_date)
        for price in prices:
            if price.hour == hour:
                return price.price_net
        return None
