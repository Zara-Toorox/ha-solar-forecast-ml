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
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class StatisticsStore:
    """Manages aggregated statistics storage @zara"""

    def __init__(self, statistics_file_path: Path, hass: "HomeAssistant | None" = None) -> None:
        """Initialize the statistics store @zara

        Args:
            statistics_file_path: Path to the statistics JSON file
            hass: Home Assistant instance for async executor jobs
        """
        self._stats_path = statistics_file_path
        self._stats_data: dict[str, Any] | None = None
        self._loaded = False
        self._dirty = False
        self._hass = hass

    async def _run_in_executor(self, func):
        """Run a function in executor, using hass if available @zara"""
        if self._hass:
            return await self._hass.async_add_executor_job(func)
        return await asyncio.get_running_loop().run_in_executor(None, func)

    async def async_load(self) -> bool:
        """Load statistics from disk asynchronously @zara

        Returns:
            True if statistics were loaded successfully
        """
        def _load() -> dict[str, Any] | None:
            try:
                if not self._stats_path.exists():
                    _LOGGER.debug("No statistics file found, starting fresh")
                    return None

                with open(self._stats_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            except (json.JSONDecodeError, IOError) as err:
                _LOGGER.warning("Failed to load statistics: %s", err)
                return None

        self._stats_data = await self._run_in_executor(_load)

        if self._stats_data is None:
            self._stats_data = self._create_empty_stats()
        else:
            self._loaded = True
            _LOGGER.debug("Loaded statistics from disk")

        return True

    def _create_empty_stats(self) -> dict[str, Any]:
        """Create an empty statistics structure @zara

        Returns:
            Empty statistics dictionary
        """
        now = datetime.now(timezone.utc).isoformat()
        return {
            "version": 1,
            "created": now,
            "last_updated": now,
            "daily_averages": [],
            "monthly_summaries": [],
            "price_extremes": {
                "all_time_low": None,
                "all_time_low_date": None,
                "all_time_high": None,
                "all_time_high_date": None,
            },
            "battery_totals": {
                "total_charged_kwh": 0.0,
                "total_cost_saved": 0.0,
            }
        }

    async def async_save(self) -> bool:
        """Save statistics to disk asynchronously @zara

        Returns:
            True if saved successfully
        """
        if not self._dirty:
            return True

        if not self._stats_data:
            return True

        self._stats_data["last_updated"] = datetime.now(timezone.utc).isoformat()

        def _save() -> bool:
            try:
                with open(self._stats_path, "w", encoding="utf-8") as f:
                    json.dump(self._stats_data, f, indent=2, ensure_ascii=False)
                return True
            except IOError as err:
                _LOGGER.error("Failed to save statistics: %s", err)
                return False

        result = await self._run_in_executor(_save)

        if result:
            self._dirty = False
            _LOGGER.debug("Saved statistics to disk")

        return result

    async def async_update_daily_average(
        self,
        date: str,
        average_net: float,
        average_total: float,
        min_price: float | None = None,
        max_price: float | None = None,
    ) -> None:
        """Update or add daily average statistics @zara

        Args:
            date: Date string (YYYY-MM-DD)
            average_net: Average net price for the day
            average_total: Average total price for the day
            min_price: Minimum price for the day
            max_price: Maximum price for the day
        """
        if not self._stats_data:
            self._stats_data = self._create_empty_stats()

        # Find existing entry or create new
        daily_averages = self._stats_data.get("daily_averages", [])
        existing_entry = None

        for entry in daily_averages:
            if entry.get("date") == date:
                existing_entry = entry
                break

        if existing_entry:
            existing_entry["average_net"] = average_net
            existing_entry["average_total"] = average_total
            if min_price is not None:
                existing_entry["min_price"] = min_price
            if max_price is not None:
                existing_entry["max_price"] = max_price
        else:
            new_entry = {
                "date": date,
                "average_net": average_net,
                "average_total": average_total,
            }
            if min_price is not None:
                new_entry["min_price"] = min_price
            if max_price is not None:
                new_entry["max_price"] = max_price
            daily_averages.append(new_entry)

        self._stats_data["daily_averages"] = daily_averages

        # Update extremes if we have min/max prices
        if min_price is not None and max_price is not None:
            await self._async_update_extremes(min_price, max_price, date)

        # Keep only last 730 days of daily averages
        if len(daily_averages) > 730:
            daily_averages.sort(key=lambda x: x.get("date", ""), reverse=True)
            self._stats_data["daily_averages"] = daily_averages[:730]

        self._dirty = True
        await self.async_save()

    async def _async_update_extremes(
        self,
        min_price: float,
        max_price: float,
        date: str
    ) -> None:
        """Update all-time price extremes @zara

        Args:
            min_price: Today's minimum price
            max_price: Today's maximum price
            date: Date string (YYYY-MM-DD)
        """
        if not self._stats_data:
            return

        extremes = self._stats_data.get("price_extremes", {})

        # Update all-time low
        current_low = extremes.get("all_time_low")
        if current_low is None or min_price < current_low:
            extremes["all_time_low"] = min_price
            extremes["all_time_low_date"] = date

        # Update all-time high
        current_high = extremes.get("all_time_high")
        if current_high is None or max_price > current_high:
            extremes["all_time_high"] = max_price
            extremes["all_time_high_date"] = date

        self._stats_data["price_extremes"] = extremes

    async def async_update_monthly_summary(
        self,
        year: int,
        month: int,
        average_price: float,
        total_cheap_hours: int,
        country: str
    ) -> None:
        """Update or add monthly summary statistics @zara

        Args:
            year: Year
            month: Month (1-12)
            average_price: Average price for the month
            total_cheap_hours: Total hours with cheap electricity
            country: Country code
        """
        if not self._stats_data:
            self._stats_data = self._create_empty_stats()

        month_key = f"{year}-{month:02d}"

        monthly_summaries = self._stats_data.get("monthly_summaries", [])
        existing_entry = None

        for entry in monthly_summaries:
            if entry.get("month") == month_key:
                existing_entry = entry
                break

        if existing_entry:
            existing_entry["average_price"] = average_price
            existing_entry["cheap_hours"] = total_cheap_hours
        else:
            monthly_summaries.append({
                "month": month_key,
                "average_price": average_price,
                "cheap_hours": total_cheap_hours,
                "country": country,
            })

        self._stats_data["monthly_summaries"] = monthly_summaries

        # Keep only last 24 months
        if len(monthly_summaries) > 24:
            monthly_summaries.sort(key=lambda x: x.get("month", ""), reverse=True)
            self._stats_data["monthly_summaries"] = monthly_summaries[:24]

        self._dirty = True
        await self.async_save()

    async def async_update_battery_totals(
        self,
        today_kwh: float,
        week_kwh: float,
        month_kwh: float
    ) -> None:
        """Update battery charging totals @zara

        Args:
            today_kwh: Energy charged today in kWh
            week_kwh: Energy charged this week in kWh
            month_kwh: Energy charged this month in kWh
        """
        if not self._stats_data:
            self._stats_data = self._create_empty_stats()

        battery_totals = self._stats_data.get("battery_totals", {})

        # Store current values (not cumulative here, just current stats)
        battery_totals["today_kwh"] = today_kwh
        battery_totals["week_kwh"] = week_kwh
        battery_totals["month_kwh"] = month_kwh

        self._stats_data["battery_totals"] = battery_totals
        self._dirty = True
        # Don't save on every update to avoid excessive I/O

    def get_daily_average(self, date: datetime) -> dict[str, Any] | None:
        """Get daily average for a specific date @zara

        Args:
            date: Date to get average for

        Returns:
            Daily average entry or None
        """
        if not self._stats_data:
            return None

        date_str = date.strftime("%Y-%m-%d")

        for entry in self._stats_data.get("daily_averages", []):
            if entry.get("date") == date_str:
                return entry

        return None

    def get_monthly_summary(self, year: int, month: int) -> dict[str, Any] | None:
        """Get monthly summary for a specific month @zara

        Args:
            year: Year
            month: Month (1-12)

        Returns:
            Monthly summary entry or None
        """
        if not self._stats_data:
            return None

        month_key = f"{year}-{month:02d}"

        for entry in self._stats_data.get("monthly_summaries", []):
            if entry.get("month") == month_key:
                return entry

        return None

    def get_price_extremes(self) -> dict[str, Any]:
        """Get all-time price extremes @zara

        Returns:
            Dictionary with price extremes
        """
        if not self._stats_data:
            return {}

        return self._stats_data.get("price_extremes", {})

    def get_battery_totals(self) -> dict[str, Any]:
        """Get battery charging totals @zara

        Returns:
            Dictionary with battery totals
        """
        if not self._stats_data:
            return {"total_charged_kwh": 0.0, "total_cost_saved": 0.0}

        return self._stats_data.get("battery_totals", {})

    def get_stats_summary(self) -> dict[str, Any]:
        """Get summary of all statistics @zara

        Returns:
            Dictionary with statistics summary
        """
        if not self._stats_data:
            return {"loaded": False}

        return {
            "loaded": self._loaded,
            "created": self._stats_data.get("created"),
            "last_updated": self._stats_data.get("last_updated"),
            "daily_average_count": len(self._stats_data.get("daily_averages", [])),
            "monthly_summary_count": len(self._stats_data.get("monthly_summaries", [])),
            "price_extremes": self.get_price_extremes(),
            "battery_totals": self.get_battery_totals(),
            "dirty": self._dirty,
        }
