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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# History retention period (2 years)
HISTORY_RETENTION_DAYS = 730

# Maximum entries to keep (24 hours * 730 days = ~17,520 entries)
MAX_HISTORY_ENTRIES = 18000


class HistoryManager:
    """Manages long-term price history storage @zara"""

    def __init__(self, history_file_path: Path, hass: "HomeAssistant | None" = None) -> None:
        """Initialize the history manager @zara

        Args:
            history_file_path: Path to the history JSON file
            hass: Home Assistant instance for async executor jobs
        """
        self._history_path = history_file_path
        self._history_data: dict[str, Any] | None = None
        self._loaded = False
        self._dirty = False
        self._hass = hass

    async def _run_in_executor(self, func):
        """Run a function in executor, using hass if available @zara"""
        if self._hass:
            return await self._hass.async_add_executor_job(func)
        return await asyncio.get_running_loop().run_in_executor(None, func)

    async def async_load(self) -> bool:
        """Load history from disk asynchronously @zara

        Returns:
            True if history was loaded successfully
        """
        def _load() -> dict[str, Any] | None:
            try:
                if not self._history_path.exists():
                    _LOGGER.debug("No history file found, starting fresh")
                    return None

                with open(self._history_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            except (json.JSONDecodeError, IOError) as err:
                _LOGGER.warning("Failed to load price history: %s", err)
                return None

        self._history_data = await self._run_in_executor(_load)

        if self._history_data is None:
            self._history_data = self._create_empty_history()
        else:
            self._loaded = True
            entry_count = len(self._history_data.get("prices", []))
            _LOGGER.info("Loaded price history with %d entries", entry_count)

            # Clean old entries on load
            await self._async_cleanup_old_entries()

        return True

    def _create_empty_history(self) -> dict[str, Any]:
        """Create an empty history structure @zara

        Returns:
            Empty history dictionary
        """
        now = datetime.now(timezone.utc).isoformat()
        return {
            "version": 1,
            "created": now,
            "last_updated": now,
            "prices": []
        }

    async def async_save(self) -> bool:
        """Save history to disk asynchronously @zara

        Returns:
            True if saved successfully
        """
        if not self._dirty:
            return True

        if not self._history_data:
            return True

        self._history_data["last_updated"] = datetime.now(timezone.utc).isoformat()

        def _save() -> bool:
            try:
                with open(self._history_path, "w", encoding="utf-8") as f:
                    json.dump(self._history_data, f, indent=2, ensure_ascii=False)
                return True
            except IOError as err:
                _LOGGER.error("Failed to save price history: %s", err)
                return False

        result = await self._run_in_executor(_save)

        if result:
            self._dirty = False
            _LOGGER.debug("Saved price history to disk")

        return result

    async def async_add_prices(self, prices: list[dict[str, Any]]) -> int:
        """Add new price entries to history @zara

        Args:
            prices: List of price entries to add

        Returns:
            Number of new entries added
        """
        if not self._history_data:
            self._history_data = self._create_empty_history()

        existing_timestamps = set()
        for entry in self._history_data.get("prices", []):
            ts = entry.get("timestamp")
            if ts:
                existing_timestamps.add(ts)

        added_count = 0
        for price_entry in prices:
            # Create history entry
            timestamp = price_entry.get("timestamp")
            if isinstance(timestamp, datetime):
                timestamp_str = timestamp.isoformat()
            else:
                timestamp_str = str(timestamp)

            # Skip if already exists
            if timestamp_str in existing_timestamps:
                continue

            history_entry = {
                "timestamp": timestamp_str,
                "price_net": price_entry.get("price"),
                "hour": price_entry.get("hour"),
            }

            self._history_data["prices"].append(history_entry)
            existing_timestamps.add(timestamp_str)
            added_count += 1

        if added_count > 0:
            self._dirty = True
            _LOGGER.debug("Added %d new price entries to history", added_count)

            # Cleanup if needed
            await self._async_cleanup_old_entries()

            # Save to disk
            await self.async_save()

        return added_count

    async def _async_cleanup_old_entries(self) -> int:
        """Remove entries older than retention period @zara

        Returns:
            Number of entries removed
        """
        if not self._history_data:
            return 0

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=HISTORY_RETENTION_DAYS)
        cutoff_str = cutoff_date.isoformat()

        original_count = len(self._history_data.get("prices", []))

        # Filter out old entries
        self._history_data["prices"] = [
            entry for entry in self._history_data.get("prices", [])
            if entry.get("timestamp", "") >= cutoff_str
        ]

        # Also enforce maximum entry limit
        if len(self._history_data["prices"]) > MAX_HISTORY_ENTRIES:
            # Sort by timestamp and keep newest
            self._history_data["prices"].sort(
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )
            self._history_data["prices"] = self._history_data["prices"][:MAX_HISTORY_ENTRIES]

        removed_count = original_count - len(self._history_data["prices"])

        if removed_count > 0:
            self._dirty = True
            _LOGGER.info("Cleaned up %d old history entries", removed_count)

        return removed_count

    def get_prices_for_date(
        self,
        date: datetime
    ) -> list[dict[str, Any]]:
        """Get all prices for a specific date @zara

        Args:
            date: Date to get prices for

        Returns:
            List of price entries for the date
        """
        if not self._history_data:
            return []

        # Create date range strings
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        if start_of_day.tzinfo is None:
            start_of_day = start_of_day.replace(tzinfo=timezone.utc)
        if end_of_day.tzinfo is None:
            end_of_day = end_of_day.replace(tzinfo=timezone.utc)

        start_str = start_of_day.isoformat()
        end_str = end_of_day.isoformat()

        return [
            entry for entry in self._history_data.get("prices", [])
            if start_str <= entry.get("timestamp", "") < end_str
        ]

    def get_prices_for_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> list[dict[str, Any]]:
        """Get all prices for a date range @zara

        Args:
            start_date: Start of range
            end_date: End of range

        Returns:
            List of price entries in range
        """
        if not self._history_data:
            return []

        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        return [
            entry for entry in self._history_data.get("prices", [])
            if start_str <= entry.get("timestamp", "") <= end_str
        ]

    def get_average_price_for_date(self, date: datetime) -> float | None:
        """Calculate average price for a specific date @zara

        Args:
            date: Date to calculate average for

        Returns:
            Average price or None if no data
        """
        prices = self.get_prices_for_date(date)
        if not prices:
            return None

        total = sum(p.get("price_net", 0) for p in prices)
        return round(total / len(prices), 4)

    def get_min_max_for_date(
        self,
        date: datetime
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Get minimum and maximum price entries for a date @zara

        Args:
            date: Date to check

        Returns:
            Tuple of (min_entry, max_entry) or (None, None)
        """
        prices = self.get_prices_for_date(date)
        if not prices:
            return None, None

        min_entry = min(prices, key=lambda x: x.get("price_net", float("inf")))
        max_entry = max(prices, key=lambda x: x.get("price_net", float("-inf")))

        return min_entry, max_entry

    def get_history_stats(self) -> dict[str, Any]:
        """Get statistics about the stored history @zara

        Returns:
            Dictionary with history statistics
        """
        if not self._history_data:
            return {
                "loaded": False,
                "entry_count": 0,
            }

        prices = self._history_data.get("prices", [])
        timestamps = [p.get("timestamp", "") for p in prices if p.get("timestamp")]

        oldest = min(timestamps) if timestamps else None
        newest = max(timestamps) if timestamps else None

        return {
            "loaded": self._loaded,
            "entry_count": len(prices),
            "oldest_entry": oldest,
            "newest_entry": newest,
            "created": self._history_data.get("created"),
            "last_updated": self._history_data.get("last_updated"),
            "dirty": self._dirty,
        }
