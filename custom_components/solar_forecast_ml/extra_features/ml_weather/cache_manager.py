# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML - ML Weather
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************
"""Cache Manager for ML Weather - handles local caching of weather data."""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

# Source paths (Solar Forecast ML)
# Primary: weather_integration_ml.json (updated 5x daily for accurate weather display)
# Fallback: weather_forecast_corrected.json (updated 1-2x daily for solar forecasting)
SOURCE_PATH = "/config/solar_forecast_ml/stats/weather_integration_ml.json"
FALLBACK_SOURCE_PATH = "/config/solar_forecast_ml/stats/weather_forecast_corrected.json"

# Cache directory
CACHE_DIR = "/config/sfml_weather"
CACHE_FILE = "weather_cache.json"
CACHE_METADATA_FILE = "cache_metadata.json"


class CacheManager:
    """Manages local caching of weather data from Solar Forecast ML."""

    def __init__(self, hass, source_path: str = SOURCE_PATH, fallback_path: str = FALLBACK_SOURCE_PATH) -> None:
        """Initialize the cache manager."""
        self.hass = hass
        self._source_path = source_path
        self._fallback_path = fallback_path
        self._cache_dir = Path(CACHE_DIR)
        self._cache_file = self._cache_dir / CACHE_FILE
        self._metadata_file = self._cache_dir / CACHE_METADATA_FILE
        self._active_source: str = source_path  # Track which source is being used

    async def async_initialize(self) -> None:
        """Initialize cache directory structure."""
        await self.hass.async_add_executor_job(self._ensure_directories)

    def _ensure_directories(self) -> None:
        """Ensure cache directories exist."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_subdir = self._cache_dir / "cache"
        cache_subdir.mkdir(exist_ok=True)
        _LOGGER.debug(f"Cache directory ensured: {self._cache_dir}")

    async def async_update_cache(self) -> dict[str, Any] | None:
        """
        Update cache from source file if newer.
        Returns the cached data or None if update failed.
        """
        return await self.hass.async_add_executor_job(self._update_cache)

    def _update_cache(self) -> dict[str, Any] | None:
        """Synchronous cache update (runs in executor).

        Uses primary source (weather_integration_ml.json) if available,
        falls back to weather_forecast_corrected.json otherwise.
        """
        # Try primary source first
        source_path = Path(self._source_path)
        fallback_path = Path(self._fallback_path)

        # Determine which source to use
        active_path = None
        if source_path.exists():
            active_path = source_path
            self._active_source = str(source_path)
        elif fallback_path.exists():
            _LOGGER.info(
                f"Primary source not found ({self._source_path}), "
                f"using fallback: {self._fallback_path}"
            )
            active_path = fallback_path
            self._active_source = str(fallback_path)
        else:
            _LOGGER.warning(
                f"No source files found: {self._source_path} or {self._fallback_path}"
            )
            # Try to return existing cache
            return self._load_cache()

        try:
            # Check if source is newer than cache
            source_mtime = active_path.stat().st_mtime
            cache_mtime = self._cache_file.stat().st_mtime if self._cache_file.exists() else 0

            if source_mtime > cache_mtime:
                _LOGGER.debug(f"Source file is newer, updating cache from {active_path.name}")
                return self._copy_and_load(active_path, source_mtime)
            else:
                _LOGGER.debug("Cache is up to date")
                return self._load_cache()

        except Exception as err:
            _LOGGER.error(f"Error updating cache: {err}")
            # Try to return existing cache on error
            return self._load_cache()

    def _copy_and_load(self, source_path: Path, source_mtime: float) -> dict[str, Any] | None:
        """Copy source to cache and load data."""
        try:
            # Read source data
            with open(source_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Enrich with cache metadata
            cache_data = {
                "source": "solar_forecast_ml",
                "source_file": str(source_path),
                "cached_at": datetime.now().isoformat(),
                "source_modified": datetime.fromtimestamp(source_mtime).isoformat(),
                "data": data,
            }

            # Write to cache
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)

            # Update metadata
            self._update_metadata(source_mtime)

            _LOGGER.info(f"Cache updated from {source_path}")
            return data

        except Exception as err:
            _LOGGER.error(f"Error copying to cache: {err}")
            return None

    def _load_cache(self) -> dict[str, Any] | None:
        """Load data from cache file."""
        if not self._cache_file.exists():
            _LOGGER.debug("No cache file exists yet")
            return None

        try:
            with open(self._cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # Return the actual weather data, not the wrapper
            return cache_data.get("data", cache_data)

        except Exception as err:
            _LOGGER.error(f"Error loading cache: {err}")
            return None

    def _update_metadata(self, source_mtime: float) -> None:
        """Update cache metadata file."""
        metadata = {
            "version": "1.0",
            "last_update": datetime.now().isoformat(),
            "source_modified": datetime.fromtimestamp(source_mtime).isoformat(),
            "source_path": self._source_path,
            "cache_file": str(self._cache_file),
        }

        try:
            with open(self._metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as err:
            _LOGGER.warning(f"Could not update metadata: {err}")

    async def async_get_cache_info(self) -> dict[str, Any]:
        """Get information about the cache status."""
        return await self.hass.async_add_executor_job(self._get_cache_info)

    def _get_cache_info(self) -> dict[str, Any]:
        """Get cache status information."""
        primary_exists = Path(self._source_path).exists()
        fallback_exists = Path(self._fallback_path).exists()

        info = {
            "cache_dir": str(self._cache_dir),
            "cache_exists": self._cache_file.exists(),
            "primary_source": self._source_path,
            "primary_source_exists": primary_exists,
            "fallback_source": self._fallback_path,
            "fallback_source_exists": fallback_exists,
            "active_source": self._active_source,
            "using_high_frequency_cache": primary_exists,  # True if using 5x daily updates
        }

        if self._cache_file.exists():
            stat = self._cache_file.stat()
            info["cache_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            info["cache_size_kb"] = round(stat.st_size / 1024, 2)

        # Show info for active source
        active_path = Path(self._active_source)
        if active_path.exists():
            stat = active_path.stat()
            info["source_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r", encoding="utf-8") as f:
                    info["metadata"] = json.load(f)
            except Exception:
                pass

        return info

    @property
    def cache_path(self) -> str:
        """Return the path to the cache file."""
        return str(self._cache_file)
