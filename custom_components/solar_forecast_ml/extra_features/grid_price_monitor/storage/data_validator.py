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
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Directory and file structure
DATA_DIR_NAME = "data"
LOGS_DIR_NAME = "logs"

# Data files
PRICE_HISTORY_FILE = "price_history.json"
PRICE_CACHE_FILE = "price_cache.json"
BATTERY_STATS_FILE = "battery_stats.json"
STATISTICS_FILE = "statistics.json"
CONFIG_BACKUP_FILE = "config_backup.json"

# File templates
EMPTY_PRICE_HISTORY = {
    "version": 1,
    "created": None,
    "last_updated": None,
    "prices": []
}

EMPTY_PRICE_CACHE = {
    "version": 1,
    "last_fetch": None,
    "valid_until": None,
    "country": None,
    "prices": []
}

EMPTY_BATTERY_STATS = {
    "version": 1,
    "created": None,
    "last_updated": None,
    "current_day": None,
    "current_week": None,
    "current_month": None,
    "energy_today_wh": 0.0,
    "energy_week_wh": 0.0,
    "energy_month_wh": 0.0,
    "history": []
}

EMPTY_STATISTICS = {
    "version": 1,
    "created": None,
    "last_updated": None,
    "daily_averages": [],
    "monthly_summaries": [],
    "price_extremes": {
        "all_time_low": None,
        "all_time_high": None,
        "lowest_day": None,
        "highest_day": None
    }
}


class DataValidator:
    """Validates and manages the data directory structure @zara"""

    def __init__(self, base_path: Path, hass: "HomeAssistant | None" = None) -> None:
        """Initialize the data validator @zara

        Args:
            base_path: Base path for GPM data (e.g., /config/grid_price_monitor)
            hass: Home Assistant instance for async executor jobs
        """
        self._base_path = base_path
        self._data_path = self._base_path / DATA_DIR_NAME
        self._logs_path = self._base_path / LOGS_DIR_NAME
        self._hass = hass

    @property
    def base_path(self) -> Path:
        """Get the base directory path @zara"""
        return self._base_path

    @property
    def data_path(self) -> Path:
        """Get the data directory path @zara"""
        return self._data_path

    @property
    def logs_path(self) -> Path:
        """Get the logs directory path @zara"""
        return self._logs_path

    @property
    def price_history_path(self) -> Path:
        """Get the price history file path @zara"""
        return self._data_path / PRICE_HISTORY_FILE

    @property
    def price_cache_path(self) -> Path:
        """Get the price cache file path @zara"""
        return self._data_path / PRICE_CACHE_FILE

    @property
    def battery_stats_path(self) -> Path:
        """Get the battery stats file path @zara"""
        return self._data_path / BATTERY_STATS_FILE

    @property
    def statistics_path(self) -> Path:
        """Get the statistics file path @zara"""
        return self._data_path / STATISTICS_FILE

    @property
    def config_backup_path(self) -> Path:
        """Get the config backup file path @zara"""
        return self._base_path / CONFIG_BACKUP_FILE

    def get_log_file_path(self, date: datetime | None = None) -> Path:
        """Get the log file path for a specific month @zara

        Args:
            date: Date for the log file (defaults to current date)

        Returns:
            Path to the monthly log file
        """
        if date is None:
            date = datetime.now()
        filename = f"gpm_{date.strftime('%Y-%m')}.log"
        return self._logs_path / filename

    async def _run_in_executor(self, func):
        """Run a function in executor, using hass if available @zara"""
        if self._hass:
            return await self._hass.async_add_executor_job(func)
        return await asyncio.get_running_loop().run_in_executor(None, func)

    async def async_validate_structure(self) -> bool:
        """Validate and create the complete directory structure @zara

        Returns:
            True if structure is valid/created successfully
        """
        try:
            # Create directories (non-blocking)
            await self._run_in_executor(self._ensure_directories)

            # Create/validate data files
            await self._async_ensure_data_files()

            _LOGGER.info(
                "Data structure validated successfully at %s",
                self._base_path
            )
            return True

        except Exception as err:
            _LOGGER.error("Failed to validate data structure: %s", err)
            return False

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist (sync) @zara"""
        directories = [
            self._base_path,
            self._data_path,
            self._logs_path,
        ]

        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                _LOGGER.debug("Created directory: %s", directory)

    async def _async_ensure_data_files(self) -> None:
        """Ensure all required data files exist with valid content @zara"""
        file_templates = [
            (self.price_history_path, EMPTY_PRICE_HISTORY),
            (self.price_cache_path, EMPTY_PRICE_CACHE),
            (self.battery_stats_path, EMPTY_BATTERY_STATS),
            (self.statistics_path, EMPTY_STATISTICS),
        ]

        now = datetime.now().isoformat()

        for file_path, template in file_templates:
            if not file_path.exists():
                # Create file with template
                data = template.copy()
                if "created" in data:
                    data["created"] = now
                await self._async_write_json(file_path, data)
                _LOGGER.debug("Created data file: %s", file_path)
            else:
                # Validate existing file
                await self._async_validate_json_file(file_path, template)

    async def _async_validate_json_file(
        self,
        file_path: Path,
        template: dict[str, Any]
    ) -> None:
        """Validate a JSON file and repair if necessary @zara

        Args:
            file_path: Path to the JSON file
            template: Template with required structure
        """
        try:
            data = await self._async_read_json(file_path)
            if data is None:
                # File is corrupted, recreate from template
                _LOGGER.warning(
                    "Corrupted file detected, recreating: %s",
                    file_path
                )
                data = template.copy()
                data["created"] = datetime.now().isoformat()
                await self._async_write_json(file_path, data)
        except Exception as err:
            _LOGGER.error("Error validating %s: %s", file_path, err)

    async def _async_read_json(self, file_path: Path) -> dict[str, Any] | None:
        """Read and parse a JSON file asynchronously @zara

        Args:
            file_path: Path to the JSON file

        Returns:
            Parsed JSON data or None if invalid
        """
        def _read() -> dict[str, Any] | None:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as err:
                _LOGGER.warning("Failed to read %s: %s", file_path, err)
                return None

        return await self._run_in_executor(_read)

    async def _async_write_json(
        self,
        file_path: Path,
        data: dict[str, Any]
    ) -> bool:
        """Write data to a JSON file asynchronously @zara

        Args:
            file_path: Path to the JSON file
            data: Data to write

        Returns:
            True if successful
        """
        def _write() -> bool:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
            except IOError as err:
                _LOGGER.error("Failed to write %s: %s", file_path, err)
                return False

        return await self._run_in_executor(_write)

    async def async_backup_config(self, config_data: dict[str, Any]) -> bool:
        """Backup the current configuration @zara

        Args:
            config_data: Configuration data to backup

        Returns:
            True if successful
        """
        backup_data = {
            "version": 1,
            "backup_time": datetime.now().isoformat(),
            "config": config_data
        }
        return await self._async_write_json(self.config_backup_path, backup_data)

    async def async_restore_config(self) -> dict[str, Any] | None:
        """Restore configuration from backup @zara

        Returns:
            Configuration data or None if no backup exists
        """
        if not self.config_backup_path.exists():
            return None

        data = await self._async_read_json(self.config_backup_path)
        if data and "config" in data:
            return data["config"]
        return None

    def get_storage_info(self) -> dict[str, Any]:
        """Get information about storage usage @zara

        Returns:
            Dictionary with storage information
        """
        info = {
            "base_path": str(self._base_path),
            "exists": self._base_path.exists(),
            "files": {}
        }

        if self._base_path.exists():
            for file_path in [
                self.price_history_path,
                self.price_cache_path,
                self.battery_stats_path,
                self.statistics_path,
                self.config_backup_path,
            ]:
                if file_path.exists():
                    stat = file_path.stat()
                    info["files"][file_path.name] = {
                        "size_bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }

        return info
