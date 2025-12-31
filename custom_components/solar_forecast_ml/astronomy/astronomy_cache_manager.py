# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

_LOGGER = logging.getLogger(__name__)

class AstronomyCacheManager:
    """
    In-memory cache manager for astronomy data

    Provides fast, thread-safe synchronous access to astronomy cache
    without blocking I/O operations.
    """

    def __init__(self):
        """Initialize the cache manager @zara"""
        self._cache: Optional[Dict[str, Any]] = None
        self._lock = Lock()
        self._last_loaded: Optional[datetime] = None
        self._cache_file: Optional[Path] = None

    def initialize(self, cache_file: Path) -> bool:
        """Initialize cache from file (file-based, no reload needed) @zara"""
        self._cache_file = cache_file

        if not self._cache_file or not self._cache_file.exists():
            _LOGGER.debug("Astronomy cache file not found, cannot load")
            return False

        try:
            with self._lock:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                self._last_loaded = datetime.now()

            _LOGGER.debug(
                f"Astronomy cache loaded into memory: {len(self._cache.get('days', {}))} days"
            )
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to load astronomy cache: {e}")
            return False

    def get_day_data(self, date_str: str) -> Optional[Dict[str, Any]]:
        """Get astronomy data for a specific date (thread-safe) @zara"""
        if not self._cache:
            return None

        with self._lock:
            return self._cache.get("days", {}).get(date_str)

    def get_production_window(self, date_str: str) -> Optional[tuple]:
        """Get production window for a date @zara"""
        day_data = self.get_day_data(date_str)
        if not day_data:
            return None

        start = day_data.get("production_window_start")
        end = day_data.get("production_window_end")

        if start and end:
            return (start, end)
        return None

    def get_hourly_data(self, date_str: str, hour: int) -> Optional[Dict[str, Any]]:
        """Get hourly astronomy data for specific date and hour @zara"""
        day_data = self.get_day_data(date_str)
        if not day_data:
            return None

        return day_data.get("hourly", {}).get(str(hour))

    def get_pv_system_info(self) -> Optional[Dict[str, Any]]:
        """Get PV system information @zara"""
        if not self._cache:
            return None

        with self._lock:
            return self._cache.get("pv_system")

    def is_loaded(self) -> bool:
        """Check if cache is loaded @zara"""
        return self._cache is not None

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache metadata @zara"""
        if not self._cache:
            return {"loaded": False, "last_loaded": None, "total_days": 0}

        with self._lock:
            cache_info = self._cache.get("cache_info", {})
            return {
                "loaded": True,
                "last_loaded": self._last_loaded.isoformat() if self._last_loaded else None,
                "total_days": cache_info.get("total_days", 0),
                "date_range_start": cache_info.get("date_range_start"),
                "date_range_end": cache_info.get("date_range_end"),
            }

    def clear(self):
        """Clear the cache @zara"""
        with self._lock:
            self._cache = None
            self._last_loaded = None

_cache_manager: Optional[AstronomyCacheManager] = None

def get_cache_manager() -> AstronomyCacheManager:
    """Get the global cache manager instance @zara"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = AstronomyCacheManager()
    return _cache_manager
