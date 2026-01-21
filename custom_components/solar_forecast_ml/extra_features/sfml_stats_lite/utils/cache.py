"""Caching utilities for SFML Stats Lite.

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
from __future__ import annotations

import asyncio
import functools
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, TypeVar

from ..const import API_CACHE_TTL_SECONDS

_LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class TTLCache:
    """Simple TTL (Time-To-Live) cache for async functions.

    Thread-safe cache that stores results with an expiration time.
    Expired entries are automatically ignored and refreshed on access.

    Example:
        cache = TTLCache(ttl_seconds=30)

        @cache.cached(key_func=lambda path: str(path))
        async def read_file(path: Path) -> dict:
            ...
    """

    def __init__(self, ttl_seconds: int = API_CACHE_TTL_SECONDS) -> None:
        """Initialize the cache."""
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> tuple[bool, Any]:
        """Get a value from cache."""
        async with self._lock:
            if key in self._cache:
                cached_time, cached_value = self._cache[key]
                if datetime.now() - cached_time < self._ttl:
                    return True, cached_value
                # Entry expired, remove it
                del self._cache[key]
        return False, None

    async def set(self, key: str, value: Any) -> None:
        """Set a value in cache."""
        async with self._lock:
            self._cache[key] = (datetime.now(), value)

    async def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
        return False

    async def clear(self) -> int:
        """Clear all cache entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    async def cleanup_expired(self) -> int:
        """Remove all expired entries."""
        async with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, (cached_time, _) in self._cache.items()
                if now - cached_time >= self._ttl
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def cached(self, key_func: Callable[..., str]) -> Callable:
        """Decorator for caching async function results."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                cache_key = key_func(*args, **kwargs)

                # Check cache
                found, cached_value = await self.get(cache_key)
                if found:
                    _LOGGER.debug("Cache hit for key: %s", cache_key)
                    return cached_value

                # Call function and cache result
                _LOGGER.debug("Cache miss for key: %s", cache_key)
                result = await func(*args, **kwargs)
                await self.set(cache_key, result)
                return result

            return wrapper
        return decorator

    @property
    def size(self) -> int:
        """Return current cache size."""
        return len(self._cache)

    @property
    def ttl_seconds(self) -> int:
        """Return TTL in seconds."""
        return int(self._ttl.total_seconds())


# Global cache instance for JSON file reads
_json_file_cache = TTLCache(ttl_seconds=API_CACHE_TTL_SECONDS)


def get_json_cache() -> TTLCache:
    """Get the global JSON file cache instance."""
    return _json_file_cache
