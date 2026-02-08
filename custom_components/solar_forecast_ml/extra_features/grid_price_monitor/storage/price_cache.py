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
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .db_connector import GPMDatabaseConnector

_LOGGER = logging.getLogger(__name__)

# Cache validity duration (48 hours to cover day-ahead prices)
CACHE_VALIDITY_HOURS = 48


class PriceCache:
    """Manages cached electricity price data in SQLite @zara"""

    def __init__(self, db: GPMDatabaseConnector) -> None:
        """Initialize the price cache @zara

        Args:
            db: GPM database connector instance
        """
        self._db = db
        self._loaded = False

    async def async_load(self) -> bool:
        """Load cache metadata from database @zara

        Returns:
            True if cache was loaded successfully
        """
        try:
            row = await self._db.fetchone(
                "SELECT last_fetch, valid_until, country FROM GPM_price_cache_meta WHERE id = 1"
            )
            if row:
                self._loaded = True
                count_row = await self._db.fetchone(
                    "SELECT COUNT(*) as cnt FROM GPM_price_cache"
                )
                entry_count = count_row["cnt"] if count_row else 0
                _LOGGER.debug("Loaded price cache with %d entries", entry_count)
                return True

            return False

        except Exception as err:
            _LOGGER.warning("Failed to load price cache: %s", err)
            return False

    async def async_save(
        self,
        prices: list[dict[str, Any]],
        country: str,
    ) -> bool:
        """Save prices to cache in database @zara

        Args:
            prices: List of price entries
            country: Country code (DE/AT)

        Returns:
            True if saved successfully
        """
        try:
            # Use LOCAL time - critical for user-facing data!
            now_local = datetime.now().astimezone()
            valid_until_local = now_local + timedelta(hours=CACHE_VALIDITY_HOURS)

            # Update metadata
            await self._db.execute(
                """INSERT INTO GPM_price_cache_meta (id, last_fetch, valid_until, country)
                   VALUES (1, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       last_fetch = excluded.last_fetch,
                       valid_until = excluded.valid_until,
                       country = excluded.country""",
                (now_local.isoformat(), valid_until_local.isoformat(), country),
                auto_commit=False,
            )

            # Replace all cache entries
            await self._db.execute(
                "DELETE FROM GPM_price_cache", auto_commit=False
            )

            # Insert new entries
            params = []
            for entry in prices:
                ts = entry.get("timestamp")
                if isinstance(ts, datetime):
                    ts = ts.isoformat()
                params.append((
                    str(ts),
                    entry.get("price", 0),
                    entry.get("total_price"),
                    entry.get("hour", 0),
                ))

            await self._db.executemany(
                """INSERT OR REPLACE INTO GPM_price_cache
                   (timestamp, price, total_price, hour) VALUES (?, ?, ?, ?)""",
                params,
            )

            _LOGGER.debug(
                "Saved %d price entries to cache, valid until %s",
                len(prices),
                valid_until_local.isoformat(),
            )
            return True

        except Exception as err:
            _LOGGER.error("Failed to save price cache: %s", err)
            return False

    def is_valid(self) -> bool:
        """Check if cache is valid and not expired @zara

        Note: Uses in-memory check based on last loaded metadata.
        For async validation, call async_is_valid().

        Returns:
            True if cache was loaded (actual validity checked async)
        """
        return self._loaded

    async def async_is_valid(self) -> bool:
        """Check if cache is valid and not expired (async) @zara

        Returns:
            True if cache is valid
        """
        try:
            row = await self._db.fetchone(
                "SELECT valid_until FROM GPM_price_cache_meta WHERE id = 1"
            )
            if not row or not row["valid_until"]:
                return False

            valid_until = datetime.fromisoformat(row["valid_until"])
            now = datetime.now().astimezone()

            if valid_until.tzinfo is None:
                valid_until = valid_until.astimezone()

            return now < valid_until

        except (ValueError, TypeError, Exception):
            return False

    def get_prices(self) -> list[dict[str, Any]]:
        """Sync wrapper - returns empty, use async_get_prices() @zara"""
        return []

    async def async_get_prices(self) -> list[dict[str, Any]]:
        """Get cached prices from database @zara

        Returns:
            List of price entries or empty list
        """
        try:
            rows = await self._db.fetchall(
                "SELECT timestamp, price, total_price, hour FROM GPM_price_cache ORDER BY timestamp"
            )
            prices = []
            for row in rows:
                entry = {
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "price": row["price"],
                    "hour": row["hour"],
                }
                if row["total_price"] is not None:
                    entry["total_price"] = row["total_price"]
                prices.append(entry)
            return prices

        except Exception as err:
            _LOGGER.warning("Failed to get cached prices: %s", err)
            return []

    async def async_get_country(self) -> str | None:
        """Get the country code for cached prices @zara

        Returns:
            Country code or None
        """
        try:
            row = await self._db.fetchone(
                "SELECT country FROM GPM_price_cache_meta WHERE id = 1"
            )
            return row["country"] if row else None
        except Exception:
            return None

    def get_country(self) -> str | None:
        """Sync wrapper - returns None, use async_get_country() @zara"""
        return None

    async def async_get_last_fetch_time(self) -> datetime | None:
        """Get the timestamp of the last fetch @zara

        Returns:
            Datetime of last fetch or None
        """
        try:
            row = await self._db.fetchone(
                "SELECT last_fetch FROM GPM_price_cache_meta WHERE id = 1"
            )
            if row and row["last_fetch"]:
                return datetime.fromisoformat(row["last_fetch"])
        except (ValueError, TypeError, Exception):
            pass
        return None

    def get_last_fetch_time(self) -> datetime | None:
        """Sync wrapper - returns None, use async_get_last_fetch_time() @zara"""
        return None

    async def async_get_cache_info(self) -> dict[str, Any]:
        """Get cache status information @zara

        Returns:
            Dictionary with cache status
        """
        try:
            row = await self._db.fetchone(
                "SELECT last_fetch, valid_until, country FROM GPM_price_cache_meta WHERE id = 1"
            )
            count_row = await self._db.fetchone(
                "SELECT COUNT(*) as cnt FROM GPM_price_cache"
            )
            entry_count = count_row["cnt"] if count_row else 0

            return {
                "loaded": self._loaded,
                "valid": await self.async_is_valid(),
                "country": row["country"] if row else None,
                "last_fetch": row["last_fetch"] if row else None,
                "valid_until": row["valid_until"] if row else None,
                "entry_count": entry_count,
            }
        except Exception:
            return {
                "loaded": False,
                "valid": False,
                "country": None,
                "last_fetch": None,
                "valid_until": None,
                "entry_count": 0,
            }

    def get_cache_info(self) -> dict[str, Any]:
        """Sync wrapper - returns basic info @zara"""
        return {
            "loaded": self._loaded,
            "valid": self._loaded,
            "country": None,
            "last_fetch": None,
            "valid_until": None,
            "entry_count": 0,
        }

    async def async_clear(self) -> bool:
        """Clear the cache @zara

        Returns:
            True if cleared successfully
        """
        try:
            await self._db.execute(
                "DELETE FROM GPM_price_cache", auto_commit=False
            )
            await self._db.execute(
                """INSERT INTO GPM_price_cache_meta (id, last_fetch, valid_until, country)
                   VALUES (1, NULL, NULL, NULL)
                   ON CONFLICT(id) DO UPDATE SET
                       last_fetch = NULL,
                       valid_until = NULL,
                       country = NULL""",
            )
            self._loaded = False
            _LOGGER.debug("Price cache cleared")
            return True
        except Exception as err:
            _LOGGER.error("Failed to clear price cache: %s", err)
            return False
