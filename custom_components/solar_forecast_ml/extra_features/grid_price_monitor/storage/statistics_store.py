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
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .db_connector import GPMDatabaseConnector

_LOGGER = logging.getLogger(__name__)


class StatisticsStore:
    """Manages aggregated statistics storage in SQLite @zara"""

    def __init__(self, db: GPMDatabaseConnector) -> None:
        """Initialize the statistics store @zara

        Args:
            db: GPM database connector instance
        """
        self._db = db
        self._loaded = False

    async def async_load(self) -> bool:
        """Check if statistics exist in database @zara

        Returns:
            True if ready
        """
        try:
            row = await self._db.fetchone(
                "SELECT COUNT(*) as cnt FROM GPM_daily_averages"
            )
            if row and row["cnt"] > 0:
                self._loaded = True
                _LOGGER.debug("Loaded statistics from database")
            return True
        except Exception as err:
            _LOGGER.warning("Failed to load statistics: %s", err)
            return True

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
        try:
            await self._db.execute(
                """INSERT INTO GPM_daily_averages
                   (date, average_net, average_total, min_price, max_price)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(date) DO UPDATE SET
                       average_net = excluded.average_net,
                       average_total = excluded.average_total,
                       min_price = COALESCE(excluded.min_price, GPM_daily_averages.min_price),
                       max_price = COALESCE(excluded.max_price, GPM_daily_averages.max_price)""",
                (date, average_net, average_total, min_price, max_price),
            )

            # Update extremes if we have min/max prices
            if min_price is not None and max_price is not None:
                await self._async_update_extremes(min_price, max_price, date)

            # Keep only last 730 days
            await self._db.execute(
                """DELETE FROM GPM_daily_averages
                   WHERE date NOT IN (
                       SELECT date FROM GPM_daily_averages
                       ORDER BY date DESC LIMIT 730
                   )""",
            )

            self._loaded = True

        except Exception as err:
            _LOGGER.error("Failed to update daily average: %s", err)

    async def _async_update_extremes(
        self,
        min_price: float,
        max_price: float,
        date: str,
    ) -> None:
        """Update all-time price extremes @zara

        Args:
            min_price: Today's minimum price
            max_price: Today's maximum price
            date: Date string (YYYY-MM-DD)
        """
        try:
            row = await self._db.fetchone(
                "SELECT all_time_low, all_time_high FROM GPM_price_extremes WHERE id = 1"
            )

            new_low = min_price
            new_low_date = date
            new_high = max_price
            new_high_date = date

            if row:
                if row["all_time_low"] is not None and row["all_time_low"] <= min_price:
                    new_low = row["all_time_low"]
                    # Keep existing date - fetch it
                    existing = await self._db.fetchone(
                        "SELECT all_time_low_date FROM GPM_price_extremes WHERE id = 1"
                    )
                    new_low_date = existing["all_time_low_date"] if existing else date

                if row["all_time_high"] is not None and row["all_time_high"] >= max_price:
                    new_high = row["all_time_high"]
                    existing = await self._db.fetchone(
                        "SELECT all_time_high_date FROM GPM_price_extremes WHERE id = 1"
                    )
                    new_high_date = existing["all_time_high_date"] if existing else date

            await self._db.execute(
                """INSERT INTO GPM_price_extremes
                   (id, all_time_low, all_time_low_date, all_time_high, all_time_high_date)
                   VALUES (1, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       all_time_low = excluded.all_time_low,
                       all_time_low_date = excluded.all_time_low_date,
                       all_time_high = excluded.all_time_high,
                       all_time_high_date = excluded.all_time_high_date""",
                (new_low, new_low_date, new_high, new_high_date),
            )

        except Exception as err:
            _LOGGER.error("Failed to update price extremes: %s", err)

    async def async_update_monthly_summary(
        self,
        year: int,
        month: int,
        average_price: float,
        total_cheap_hours: int,
        country: str,
    ) -> None:
        """Update or add monthly summary statistics @zara

        Args:
            year: Year
            month: Month (1-12)
            average_price: Average price for the month
            total_cheap_hours: Total hours with cheap electricity
            country: Country code
        """
        month_key = f"{year}-{month:02d}"

        try:
            await self._db.execute(
                """INSERT INTO GPM_monthly_summaries
                   (month, average_price, cheap_hours, country)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(month) DO UPDATE SET
                       average_price = excluded.average_price,
                       cheap_hours = excluded.cheap_hours,
                       country = excluded.country""",
                (month_key, average_price, total_cheap_hours, country),
            )

            # Keep only last 24 months
            await self._db.execute(
                """DELETE FROM GPM_monthly_summaries
                   WHERE month NOT IN (
                       SELECT month FROM GPM_monthly_summaries
                       ORDER BY month DESC LIMIT 24
                   )""",
            )

        except Exception as err:
            _LOGGER.error("Failed to update monthly summary: %s", err)

    async def async_update_battery_totals(
        self,
        today_kwh: float,
        week_kwh: float,
        month_kwh: float,
    ) -> None:
        """Update battery charging totals @zara

        Args:
            today_kwh: Energy charged today in kWh
            week_kwh: Energy charged this week in kWh
            month_kwh: Energy charged this month in kWh
        """
        try:
            await self._db.execute(
                """INSERT INTO GPM_battery_totals
                   (id, today_kwh, week_kwh, month_kwh)
                   VALUES (1, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       today_kwh = excluded.today_kwh,
                       week_kwh = excluded.week_kwh,
                       month_kwh = excluded.month_kwh""",
                (today_kwh, week_kwh, month_kwh),
            )
        except Exception as err:
            _LOGGER.error("Failed to update battery totals: %s", err)

    async def async_get_daily_average(self, date: datetime) -> dict[str, Any] | None:
        """Get daily average for a specific date @zara

        Args:
            date: Date to get average for

        Returns:
            Daily average entry or None
        """
        date_str = date.strftime("%Y-%m-%d")
        try:
            row = await self._db.fetchone(
                """SELECT date, average_net, average_total, min_price, max_price
                   FROM GPM_daily_averages WHERE date = ?""",
                (date_str,),
            )
            if row:
                return {
                    "date": row["date"],
                    "average_net": row["average_net"],
                    "average_total": row["average_total"],
                    "min_price": row["min_price"],
                    "max_price": row["max_price"],
                }
        except Exception as err:
            _LOGGER.error("Failed to get daily average: %s", err)
        return None

    def get_daily_average(self, date: datetime) -> dict[str, Any] | None:
        """Sync wrapper - returns None, use async version @zara"""
        return None

    async def async_get_monthly_summary(
        self, year: int, month: int
    ) -> dict[str, Any] | None:
        """Get monthly summary for a specific month @zara

        Args:
            year: Year
            month: Month (1-12)

        Returns:
            Monthly summary entry or None
        """
        month_key = f"{year}-{month:02d}"
        try:
            row = await self._db.fetchone(
                """SELECT month, average_price, cheap_hours, country
                   FROM GPM_monthly_summaries WHERE month = ?""",
                (month_key,),
            )
            if row:
                return {
                    "month": row["month"],
                    "average_price": row["average_price"],
                    "cheap_hours": row["cheap_hours"],
                    "country": row["country"],
                }
        except Exception as err:
            _LOGGER.error("Failed to get monthly summary: %s", err)
        return None

    def get_monthly_summary(self, year: int, month: int) -> dict[str, Any] | None:
        """Sync wrapper - returns None, use async version @zara"""
        return None

    async def async_get_price_extremes(self) -> dict[str, Any]:
        """Get all-time price extremes @zara

        Returns:
            Dictionary with price extremes
        """
        try:
            row = await self._db.fetchone(
                """SELECT all_time_low, all_time_low_date,
                          all_time_high, all_time_high_date
                   FROM GPM_price_extremes WHERE id = 1"""
            )
            if row:
                return {
                    "all_time_low": row["all_time_low"],
                    "all_time_low_date": row["all_time_low_date"],
                    "all_time_high": row["all_time_high"],
                    "all_time_high_date": row["all_time_high_date"],
                }
        except Exception:
            pass
        return {}

    def get_price_extremes(self) -> dict[str, Any]:
        """Sync wrapper - returns empty dict @zara"""
        return {}

    async def async_get_battery_totals(self) -> dict[str, Any]:
        """Get battery charging totals @zara

        Returns:
            Dictionary with battery totals
        """
        try:
            row = await self._db.fetchone(
                "SELECT today_kwh, week_kwh, month_kwh FROM GPM_battery_totals WHERE id = 1"
            )
            if row:
                return {
                    "today_kwh": row["today_kwh"],
                    "week_kwh": row["week_kwh"],
                    "month_kwh": row["month_kwh"],
                }
        except Exception:
            pass
        return {"today_kwh": 0.0, "week_kwh": 0.0, "month_kwh": 0.0}

    def get_battery_totals(self) -> dict[str, Any]:
        """Sync wrapper - returns defaults @zara"""
        return {"total_charged_kwh": 0.0, "total_cost_saved": 0.0}

    async def async_get_stats_summary(self) -> dict[str, Any]:
        """Get summary of all statistics @zara

        Returns:
            Dictionary with statistics summary
        """
        try:
            daily_row = await self._db.fetchone(
                "SELECT COUNT(*) as cnt FROM GPM_daily_averages"
            )
            monthly_row = await self._db.fetchone(
                "SELECT COUNT(*) as cnt FROM GPM_monthly_summaries"
            )
            extremes = await self.async_get_price_extremes()
            battery = await self.async_get_battery_totals()

            return {
                "loaded": self._loaded,
                "daily_average_count": daily_row["cnt"] if daily_row else 0,
                "monthly_summary_count": monthly_row["cnt"] if monthly_row else 0,
                "price_extremes": extremes,
                "battery_totals": battery,
            }
        except Exception:
            return {"loaded": False}

    def get_stats_summary(self) -> dict[str, Any]:
        """Sync wrapper - returns basic info @zara"""
        return {"loaded": self._loaded}
