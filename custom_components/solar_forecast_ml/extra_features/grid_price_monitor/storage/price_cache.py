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

# Cache validity duration (48 hours to cover day-ahead prices)
CACHE_VALIDITY_HOURS = 48


class PriceCache:
    """Manages cached electricity price data @zara"""

    def __init__(self, cache_file_path: Path, hass: "HomeAssistant | None" = None) -> None:
        """Initialize the price cache @zara

        Args:
            cache_file_path: Path to the cache JSON file
            hass: Home Assistant instance for async executor jobs
        """
        self._cache_path = cache_file_path
        self._cache_data: dict[str, Any] | None = None
        self._loaded = False
        self._hass = hass

    async def _run_in_executor(self, func):
        """Run a function in executor, using hass if available @zara"""
        if self._hass:
            return await self._hass.async_add_executor_job(func)
        return await asyncio.get_running_loop().run_in_executor(None, func)

    async def async_load(self) -> bool:
        """Load cache from disk asynchronously @zara

        Returns:
            True if cache was loaded successfully
        """
        def _load() -> dict[str, Any] | None:
            try:
                if not self._cache_path.exists():
                    _LOGGER.debug("No cache file found at %s", self._cache_path)
                    return None

                with open(self._cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            except (json.JSONDecodeError, IOError) as err:
                _LOGGER.warning("Failed to load price cache: %s", err)
                return None

        self._cache_data = await self._run_in_executor(_load)

        if self._cache_data:
            self._loaded = True
            _LOGGER.debug(
                "Loaded price cache with %d entries",
                len(self._cache_data.get("prices", []))
            )
            return True

        return False

    async def async_save(
        self,
        prices: list[dict[str, Any]],
        country: str
    ) -> bool:
        """Save prices to cache asynchronously @zara

        Args:
            prices: List of price entries
            country: Country code (DE/AT)

        Returns:
            True if saved successfully
        """
        # Use LOCAL time for JSON export - critical for user-facing data!
        now_local = datetime.now().astimezone()
        valid_until_local = now_local + timedelta(hours=CACHE_VALIDITY_HOURS)

        self._cache_data = {
            "version": 1,
            "last_fetch": now_local.isoformat(),
            "valid_until": valid_until_local.isoformat(),
            "country": country,
            "prices": self._serialize_prices(prices)
        }

        def _save() -> bool:
            try:
                with open(self._cache_path, "w", encoding="utf-8") as f:
                    json.dump(self._cache_data, f, indent=2, ensure_ascii=False)
                return True
            except IOError as err:
                _LOGGER.error("Failed to save price cache: %s", err)
                return False

        result = await self._run_in_executor(_save)

        if result:
            _LOGGER.debug(
                "Saved %d price entries to cache, valid until %s",
                len(prices),
                valid_until_local.isoformat()
            )

        return result

    def _serialize_prices(
        self,
        prices: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Serialize price entries for JSON storage @zara

        Args:
            prices: List of price entries with datetime objects

        Returns:
            Serialized list safe for JSON
        """
        serialized = []
        for entry in prices:
            serialized_entry = entry.copy()
            if "timestamp" in serialized_entry:
                ts = serialized_entry["timestamp"]
                if isinstance(ts, datetime):
                    serialized_entry["timestamp"] = ts.isoformat()
            serialized.append(serialized_entry)
        return serialized

    def _deserialize_prices(
        self,
        prices: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Deserialize price entries from JSON storage @zara

        Args:
            prices: List of serialized price entries

        Returns:
            List with datetime objects restored
        """
        deserialized = []
        for entry in prices:
            deserialized_entry = entry.copy()
            if "timestamp" in deserialized_entry:
                ts = deserialized_entry["timestamp"]
                if isinstance(ts, str):
                    deserialized_entry["timestamp"] = datetime.fromisoformat(ts)
            deserialized.append(deserialized_entry)
        return deserialized

    def is_valid(self) -> bool:
        """Check if cache is valid and not expired @zara

        Returns:
            True if cache is valid
        """
        if not self._cache_data:
            return False

        valid_until_str = self._cache_data.get("valid_until")
        if not valid_until_str:
            return False

        try:
            valid_until = datetime.fromisoformat(valid_until_str)
            # Use local time for comparison (matching how we save)
            now = datetime.now().astimezone()

            # Handle timezone-naive timestamps (legacy data)
            if valid_until.tzinfo is None:
                valid_until = valid_until.astimezone()

            return now < valid_until

        except (ValueError, TypeError):
            return False

    def get_prices(self) -> list[dict[str, Any]]:
        """Get cached prices @zara

        Returns:
            List of price entries or empty list
        """
        if not self._cache_data:
            return []

        prices = self._cache_data.get("prices", [])
        return self._deserialize_prices(prices)

    def get_country(self) -> str | None:
        """Get the country code for cached prices @zara

        Returns:
            Country code or None
        """
        if not self._cache_data:
            return None
        return self._cache_data.get("country")

    def get_last_fetch_time(self) -> datetime | None:
        """Get the timestamp of the last fetch @zara

        Returns:
            Datetime of last fetch or None
        """
        if not self._cache_data:
            return None

        last_fetch_str = self._cache_data.get("last_fetch")
        if last_fetch_str:
            try:
                return datetime.fromisoformat(last_fetch_str)
            except (ValueError, TypeError):
                pass
        return None

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache status information @zara

        Returns:
            Dictionary with cache status
        """
        return {
            "loaded": self._loaded,
            "valid": self.is_valid(),
            "country": self.get_country(),
            "last_fetch": self._cache_data.get("last_fetch") if self._cache_data else None,
            "valid_until": self._cache_data.get("valid_until") if self._cache_data else None,
            "entry_count": len(self._cache_data.get("prices", [])) if self._cache_data else 0,
        }

    async def async_clear(self) -> bool:
        """Clear the cache asynchronously @zara

        Returns:
            True if cleared successfully
        """
        self._cache_data = {
            "version": 1,
            "last_fetch": None,
            "valid_until": None,
            "country": None,
            "prices": []
        }

        def _clear() -> bool:
            try:
                with open(self._cache_path, "w", encoding="utf-8") as f:
                    json.dump(self._cache_data, f, indent=2)
                return True
            except IOError as err:
                _LOGGER.error("Failed to clear price cache: %s", err)
                return False

        result = await self._run_in_executor(_clear)

        if result:
            _LOGGER.debug("Price cache cleared")

        return result
