"""
Data Cache Module for Solar Forecast ML Integration

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
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class DataCache:
    """Handles caching logic for weather and forecast data"""
    
    def __init__(self, data_dir: Path):
        """Initialize the data cache"""
        self.data_dir = data_dir
        self.weather_cache_file = data_dir / "data" / "weather_cache.json"
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
    async def get_cached_forecast(self, key: str, max_age_hours: int = 1) -> Optional[Dict[str, Any]]:
        """Get cached forecast data if not expired"""
        if key not in self._cache:
            return None
            
        cache_time = self._cache_timestamps.get(key)
        if not cache_time:
            return None
            
        age = dt_util.now() - cache_time
        if age.total_seconds() > max_age_hours * 3600:
            _LOGGER.debug(f"Cache expired for key: {key}")
            return None
            
        _LOGGER.debug(f"Cache hit for key: {key}")
        return self._cache[key]
    
    async def set_cached_forecast(self, key: str, data: Dict[str, Any]) -> None:
        """Store forecast data in cache"""
        self._cache[key] = data
        self._cache_timestamps[key] = dt_util.now()
        _LOGGER.debug(f"Cached data for key: {key}")
    
    async def clear_cache(self, key: Optional[str] = None) -> None:
        """Clear cache data"""
        if key:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
            _LOGGER.debug(f"Cleared cache for key: {key}")
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
            _LOGGER.debug("Cleared all cache data")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "total_entries": len(self._cache),
            "keys": list(self._cache.keys()),
            "oldest_entry": min(self._cache_timestamps.values()) if self._cache_timestamps else None,
            "newest_entry": max(self._cache_timestamps.values()) if self._cache_timestamps else None,
        }
