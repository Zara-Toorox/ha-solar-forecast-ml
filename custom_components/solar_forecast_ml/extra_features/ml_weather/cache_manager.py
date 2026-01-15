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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from homeassistant.util import dt as dt_util

from .const import (
    DEFAULT_SOURCE_PATH,
    FALLBACK_SOURCE_PATH,
    CACHE_DIR,
    CACHE_FILE,
    CACHE_MAX_AGE_HOURS,
    CACHE_MAX_SIZE_MB,
)

_LOGGER = logging.getLogger(__name__)

# Cache metadata file name
CACHE_METADATA_FILE = "cache_metadata.json"


class CacheManager:
    """Manages local caching of weather data from Solar Forecast ML."""

    def __init__(
        self,
        hass,
        source_path: str = DEFAULT_SOURCE_PATH,
        fallback_path: str = FALLBACK_SOURCE_PATH
    ) -> None:
        """Initialize the cache manager."""
        self.hass = hass
        self._source_path = source_path
        self._fallback_path = fallback_path
        self._cache_dir = Path(CACHE_DIR)
        self._cache_file = self._cache_dir / CACHE_FILE
        self._metadata_file = self._cache_dir / CACHE_METADATA_FILE
        self._active_source: str = source_path  # Track which source is being used
        self._consecutive_failures: int = 0  # Track failures for backoff

    async def async_initialize(self) -> None:
        """Initialize cache directory structure."""
        await self.hass.async_add_executor_job(self._ensure_directories)

    def _ensure_directories(self) -> None:
        """Ensure cache directories exist."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_subdir = self._cache_dir / "cache"
            cache_subdir.mkdir(exist_ok=True)
            _LOGGER.debug("Cache directory ensured: %s", self._cache_dir)
        except PermissionError as err:
            _LOGGER.error("Permission denied creating cache directory: %s", err)
            raise
        except OSError as err:
            _LOGGER.error("OS error creating cache directory: %s", err)
            raise

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
                "Primary source not found (%s), using fallback: %s",
                self._source_path,
                self._fallback_path
            )
            active_path = fallback_path
            self._active_source = str(fallback_path)
        else:
            _LOGGER.warning(
                "No source files found: %s or %s",
                self._source_path,
                self._fallback_path
            )
            self._consecutive_failures += 1
            # Try to return existing cache (with staleness check)
            return self._load_cache(check_staleness=True)

        try:
            # Check if source is newer than cache
            source_mtime = active_path.stat().st_mtime
            cache_mtime = self._cache_file.stat().st_mtime if self._cache_file.exists() else 0

            if source_mtime > cache_mtime:
                _LOGGER.debug("Source file is newer, updating cache from %s", active_path.name)
                data = self._copy_and_load(active_path, source_mtime)
                if data:
                    self._consecutive_failures = 0  # Reset on success
                return data
            else:
                _LOGGER.debug("Cache is up to date")
                self._consecutive_failures = 0  # Reset on success
                return self._load_cache()

        except PermissionError as err:
            _LOGGER.error("Permission denied accessing source file: %s", err)
            self._consecutive_failures += 1
            return self._load_cache(check_staleness=True)
        except json.JSONDecodeError as err:
            _LOGGER.error("Invalid JSON in source file: %s", err)
            self._consecutive_failures += 1
            return self._load_cache(check_staleness=True)
        except OSError as err:
            _LOGGER.error("OS error updating cache: %s", err)
            self._consecutive_failures += 1
            return self._load_cache(check_staleness=True)
        except Exception as err:
            _LOGGER.error("Unexpected error updating cache: %s", err)
            self._consecutive_failures += 1
            return self._load_cache(check_staleness=True)

    def _copy_and_load(self, source_path: Path, source_mtime: float) -> dict[str, Any] | None:
        """Copy source to cache and load data."""
        try:
            # Check file size before loading
            file_size_mb = source_path.stat().st_size / (1024 * 1024)
            if file_size_mb > CACHE_MAX_SIZE_MB:
                _LOGGER.warning(
                    "Source file too large (%.2f MB > %d MB limit)",
                    file_size_mb,
                    CACHE_MAX_SIZE_MB
                )
                return None

            # Read source data
            with open(source_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Use timezone-aware datetime
            now = dt_util.now()

            # Enrich with cache metadata
            cache_data = {
                "source": "solar_forecast_ml",
                "source_file": str(source_path),
                "cached_at": now.isoformat(),
                "source_modified": datetime.fromtimestamp(source_mtime).isoformat(),
                "data": data,
            }

            # Write to cache
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)

            # Update metadata
            self._update_metadata(source_mtime)

            _LOGGER.info("Cache updated from %s", source_path)
            return data

        except json.JSONDecodeError as err:
            _LOGGER.error("Invalid JSON in source file %s: %s", source_path, err)
            return None
        except PermissionError as err:
            _LOGGER.error("Permission denied writing cache: %s", err)
            return None
        except OSError as err:
            _LOGGER.error("OS error copying to cache: %s", err)
            return None

    def _load_cache(self, check_staleness: bool = False) -> dict[str, Any] | None:
        """Load data from cache file.

        Args:
            check_staleness: If True, warn if cache is older than CACHE_MAX_AGE_HOURS.
        """
        if not self._cache_file.exists():
            _LOGGER.debug("No cache file exists yet")
            return None

        try:
            # Check cache age if requested
            if check_staleness:
                cache_mtime = self._cache_file.stat().st_mtime
                cache_age_hours = (datetime.now().timestamp() - cache_mtime) / 3600
                if cache_age_hours > CACHE_MAX_AGE_HOURS:
                    _LOGGER.warning(
                        "Cache is stale (%.1f hours old, max %d hours). "
                        "Data may be outdated.",
                        cache_age_hours,
                        CACHE_MAX_AGE_HOURS
                    )

            with open(self._cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # Return the actual weather data, not the wrapper
            return cache_data.get("data", cache_data)

        except json.JSONDecodeError as err:
            _LOGGER.error("Invalid JSON in cache file: %s", err)
            return None
        except PermissionError as err:
            _LOGGER.error("Permission denied reading cache: %s", err)
            return None
        except OSError as err:
            _LOGGER.error("OS error loading cache: %s", err)
            return None

    def _update_metadata(self, source_mtime: float) -> None:
        """Update cache metadata file."""
        now = dt_util.now()
        metadata = {
            "version": "1.0",
            "last_update": now.isoformat(),
            "source_modified": datetime.fromtimestamp(source_mtime).isoformat(),
            "source_path": self._source_path,
            "active_source": self._active_source,
            "cache_file": str(self._cache_file),
            "consecutive_failures": self._consecutive_failures,
        }

        try:
            with open(self._metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except OSError as err:
            _LOGGER.warning("Could not update metadata: %s", err)

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
            "consecutive_failures": self._consecutive_failures,
            "max_cache_age_hours": CACHE_MAX_AGE_HOURS,
        }

        if self._cache_file.exists():
            try:
                stat = self._cache_file.stat()
                cache_mtime = stat.st_mtime
                info["cache_modified"] = datetime.fromtimestamp(cache_mtime).isoformat()
                info["cache_size_kb"] = round(stat.st_size / 1024, 2)

                # Calculate cache age
                cache_age_hours = (datetime.now().timestamp() - cache_mtime) / 3600
                info["cache_age_hours"] = round(cache_age_hours, 2)
                info["cache_is_stale"] = cache_age_hours > CACHE_MAX_AGE_HOURS
            except OSError:
                pass

        # Show info for active source
        active_path = Path(self._active_source)
        if active_path.exists():
            try:
                stat = active_path.stat()
                info["source_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            except OSError:
                pass

        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r", encoding="utf-8") as f:
                    info["metadata"] = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        return info

    @property
    def cache_path(self) -> str:
        """Return the path to the cache file."""
        return str(self._cache_file)
