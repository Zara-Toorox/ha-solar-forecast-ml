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
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .db_connector import GPMDatabaseConnector

_LOGGER = logging.getLogger(__name__)

# History retention period (2 years)
HISTORY_RETENTION_DAYS = 730

# Maximum entries to keep (24 hours * 730 days = ~17,520 entries)
MAX_HISTORY_ENTRIES = 18000


class HistoryManager:
    """Manages long-term price history storage in SQLite @zara"""

    def __init__(self, db: GPMDatabaseConnector) -> None:
        """Initialize the history manager @zara

        Args:
            db: GPM database connector instance
        """
        self._db = db
        self._loaded = False

    async def async_load(self) -> bool:
        """Check if history data exists in database @zara

        Returns:
            True if history is ready
        """
        try:
            row = await self._db.fetchone(
                "SELECT COUNT(*) as cnt FROM GPM_price_history"
            )
            entry_count = row["cnt"] if row else 0
            if entry_count > 0:
                self._loaded = True
                _LOGGER.info("Loaded price history with %d entries", entry_count)

            # Clean old entries on load
            await self._async_cleanup_old_entries()

            return True

        except Exception as err:
            _LOGGER.warning("Failed to load price history: %s", err)
            return True  # Continue even if empty

    async def async_add_prices(self, prices: list[dict[str, Any]]) -> int:
        """Add new price entries to history @zara

        Args:
            prices: List of price entries to add

        Returns:
            Number of new entries added
        """
        if not prices:
            return 0

        params = []
        for price_entry in prices:
            timestamp = price_entry.get("timestamp")
            if isinstance(timestamp, datetime):
                timestamp_str = timestamp.isoformat()
            else:
                timestamp_str = str(timestamp)

            params.append((
                timestamp_str,
                price_entry.get("price", 0),
                price_entry.get("total_price"),
                price_entry.get("hour", 0),
            ))

        try:
            # INSERT OR IGNORE skips duplicates (timestamp is UNIQUE)
            await self._db.executemany(
                """INSERT OR IGNORE INTO GPM_price_history
                   (timestamp, price_net, total_price, hour) VALUES (?, ?, ?, ?)""",
                params,
            )

            # Count how many were actually new
            added_count = len(params)  # Approximate; exact count not critical
            if added_count > 0:
                self._loaded = True
                _LOGGER.debug("Added up to %d price entries to history", added_count)

                # Cleanup if needed
                await self._async_cleanup_old_entries()

            return added_count

        except Exception as err:
            _LOGGER.error("Failed to add prices to history: %s", err)
            return 0

    async def _async_cleanup_old_entries(self) -> int:
        """Remove entries older than retention period @zara

        Returns:
            Number of entries removed
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=HISTORY_RETENTION_DAYS)
            cutoff_str = cutoff_date.isoformat()

            # Delete old entries
            await self._db.execute(
                "DELETE FROM GPM_price_history WHERE timestamp < ?",
                (cutoff_str,),
                auto_commit=False,
            )

            # Enforce maximum entry limit
            row = await self._db.fetchone(
                "SELECT COUNT(*) as cnt FROM GPM_price_history"
            )
            total = row["cnt"] if row else 0

            if total > MAX_HISTORY_ENTRIES:
                overflow = total - MAX_HISTORY_ENTRIES
                await self._db.execute(
                    """DELETE FROM GPM_price_history WHERE id IN (
                           SELECT id FROM GPM_price_history
                           ORDER BY timestamp ASC LIMIT ?
                       )""",
                    (overflow,),
                )
                _LOGGER.info("Cleaned up %d old history entries", overflow)

            await self._db.commit()
            return 0

        except Exception as err:
            _LOGGER.error("Failed to cleanup history: %s", err)
            return 0

    async def async_get_prices_for_date(
        self,
        date: datetime,
    ) -> list[dict[str, Any]]:
        """Get all prices for a specific date @zara

        Args:
            date: Date to get prices for

        Returns:
            List of price entries for the date
        """
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        if start_of_day.tzinfo is None:
            start_of_day = start_of_day.replace(tzinfo=timezone.utc)
        if end_of_day.tzinfo is None:
            end_of_day = end_of_day.replace(tzinfo=timezone.utc)

        try:
            rows = await self._db.fetchall(
                """SELECT timestamp, price_net, total_price, hour FROM GPM_price_history
                   WHERE timestamp >= ? AND timestamp < ?
                   ORDER BY timestamp""",
                (start_of_day.isoformat(), end_of_day.isoformat()),
            )
            return [
                {
                    "timestamp": row["timestamp"],
                    "price_net": row["price_net"],
                    "total_price": row["total_price"],
                    "hour": row["hour"],
                }
                for row in rows
            ]
        except Exception as err:
            _LOGGER.error("Failed to get prices for date: %s", err)
            return []

    # Keep sync wrapper for backward compatibility in coordinator
    def get_prices_for_date(self, date: datetime) -> list[dict[str, Any]]:
        """Sync wrapper - returns empty, use async_get_prices_for_date() @zara"""
        return []

    async def async_get_prices_for_range(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        """Get all prices for a date range @zara

        Args:
            start_date: Start of range
            end_date: End of range

        Returns:
            List of price entries in range
        """
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        try:
            rows = await self._db.fetchall(
                """SELECT timestamp, price_net, total_price, hour FROM GPM_price_history
                   WHERE timestamp >= ? AND timestamp <= ?
                   ORDER BY timestamp""",
                (start_date.isoformat(), end_date.isoformat()),
            )
            return [
                {
                    "timestamp": row["timestamp"],
                    "price_net": row["price_net"],
                    "total_price": row["total_price"],
                    "hour": row["hour"],
                }
                for row in rows
            ]
        except Exception as err:
            _LOGGER.error("Failed to get prices for range: %s", err)
            return []

    def get_prices_for_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[dict[str, Any]]:
        """Sync wrapper - returns empty, use async_get_prices_for_range() @zara"""
        return []

    async def async_get_average_price_for_date(self, date: datetime) -> float | None:
        """Calculate average price for a specific date @zara

        Args:
            date: Date to calculate average for

        Returns:
            Average price or None if no data
        """
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        if start_of_day.tzinfo is None:
            start_of_day = start_of_day.replace(tzinfo=timezone.utc)
        if end_of_day.tzinfo is None:
            end_of_day = end_of_day.replace(tzinfo=timezone.utc)

        try:
            row = await self._db.fetchone(
                """SELECT AVG(price_net) as avg_price FROM GPM_price_history
                   WHERE timestamp >= ? AND timestamp < ?""",
                (start_of_day.isoformat(), end_of_day.isoformat()),
            )
            if row and row["avg_price"] is not None:
                return round(row["avg_price"], 4)
        except Exception as err:
            _LOGGER.error("Failed to get average price: %s", err)
        return None

    def get_average_price_for_date(self, date: datetime) -> float | None:
        """Sync wrapper - returns None, use async version @zara"""
        return None

    async def async_get_min_max_for_date(
        self, date: datetime
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Get minimum and maximum price entries for a date @zara

        Args:
            date: Date to check

        Returns:
            Tuple of (min_entry, max_entry) or (None, None)
        """
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        if start_of_day.tzinfo is None:
            start_of_day = start_of_day.replace(tzinfo=timezone.utc)
        if end_of_day.tzinfo is None:
            end_of_day = end_of_day.replace(tzinfo=timezone.utc)

        try:
            min_row = await self._db.fetchone(
                """SELECT timestamp, price_net, total_price, hour FROM GPM_price_history
                   WHERE timestamp >= ? AND timestamp < ?
                   ORDER BY price_net ASC LIMIT 1""",
                (start_of_day.isoformat(), end_of_day.isoformat()),
            )
            max_row = await self._db.fetchone(
                """SELECT timestamp, price_net, total_price, hour FROM GPM_price_history
                   WHERE timestamp >= ? AND timestamp < ?
                   ORDER BY price_net DESC LIMIT 1""",
                (start_of_day.isoformat(), end_of_day.isoformat()),
            )

            min_entry = None
            max_entry = None

            if min_row:
                min_entry = {
                    "timestamp": min_row["timestamp"],
                    "price_net": min_row["price_net"],
                    "total_price": min_row["total_price"],
                    "hour": min_row["hour"],
                }
            if max_row:
                max_entry = {
                    "timestamp": max_row["timestamp"],
                    "price_net": max_row["price_net"],
                    "total_price": max_row["total_price"],
                    "hour": max_row["hour"],
                }

            return min_entry, max_entry

        except Exception as err:
            _LOGGER.error("Failed to get min/max prices: %s", err)
            return None, None

    def get_min_max_for_date(
        self, date: datetime
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Sync wrapper - returns (None, None) @zara"""
        return None, None

    async def async_get_history_stats(self) -> dict[str, Any]:
        """Get statistics about the stored history @zara

        Returns:
            Dictionary with history statistics
        """
        try:
            row = await self._db.fetchone(
                """SELECT COUNT(*) as cnt,
                          MIN(timestamp) as oldest,
                          MAX(timestamp) as newest
                   FROM GPM_price_history"""
            )

            if not row:
                return {"loaded": False, "entry_count": 0}

            return {
                "loaded": self._loaded,
                "entry_count": row["cnt"],
                "oldest_entry": row["oldest"],
                "newest_entry": row["newest"],
            }

        except Exception as err:
            _LOGGER.error("Failed to get history stats: %s", err)
            return {"loaded": False, "entry_count": 0}

    def get_history_stats(self) -> dict[str, Any]:
        """Sync wrapper - returns basic info @zara"""
        return {
            "loaded": self._loaded,
            "entry_count": 0,
        }
