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
    from .db_connector import GPMDatabaseConnector

_LOGGER = logging.getLogger(__name__)

# Directory structure
LOGS_DIR_NAME = "logs"

# Legacy JSON files to clean up
LEGACY_DATA_DIR = "data"
LEGACY_FILES = [
    "price_cache.json",
    "price_history.json",
    "statistics.json",
    "battery_stats.json",
]
LEGACY_CONFIG_BACKUP = "config_backup.json"


class DataValidator:
    """Validates directory structure and manages config backup @zara"""

    def __init__(
        self,
        base_path: Path,
        db: GPMDatabaseConnector,
        hass: HomeAssistant | None = None,
    ) -> None:
        """Initialize the data validator @zara

        Args:
            base_path: Base path for GPM data (e.g., /config/grid_price_monitor)
            db: GPM database connector instance
            hass: Home Assistant instance for async executor jobs
        """
        self._base_path = base_path
        self._logs_path = self._base_path / LOGS_DIR_NAME
        self._db = db
        self._hass = hass

    @property
    def base_path(self) -> Path:
        """Get the base directory path @zara"""
        return self._base_path

    @property
    def logs_path(self) -> Path:
        """Get the logs directory path @zara"""
        return self._logs_path

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
        """Validate and create the directory structure @zara

        Returns:
            True if structure is valid/created successfully
        """
        try:
            await self._run_in_executor(self._ensure_directories)
            await self._async_cleanup_legacy_files()

            _LOGGER.info(
                "Data structure validated successfully at %s",
                self._base_path,
            )
            return True

        except Exception as err:
            _LOGGER.error("Failed to validate data structure: %s", err)
            return False

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist (sync) @zara"""
        directories = [
            self._base_path,
            self._logs_path,
        ]

        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                _LOGGER.debug("Created directory: %s", directory)

    async def _async_cleanup_legacy_files(self) -> None:
        """Remove legacy JSON files after migration to database @zara"""

        def _cleanup() -> None:
            data_dir = self._base_path / LEGACY_DATA_DIR
            removed = []

            # Remove legacy data files
            if data_dir.exists():
                for filename in LEGACY_FILES:
                    file_path = data_dir / filename
                    if file_path.exists():
                        file_path.unlink()
                        removed.append(filename)

                # Remove data directory if empty
                try:
                    data_dir.rmdir()
                    removed.append(LEGACY_DATA_DIR + "/")
                except OSError:
                    pass  # Directory not empty (other files present)

            # Remove legacy config backup
            config_backup = self._base_path / LEGACY_CONFIG_BACKUP
            if config_backup.exists():
                config_backup.unlink()
                removed.append(LEGACY_CONFIG_BACKUP)

            if removed:
                _LOGGER.info(
                    "Migrated to database, removed legacy files: %s",
                    ", ".join(removed),
                )

        await self._run_in_executor(_cleanup)

    async def async_backup_config(self, config_data: dict[str, Any]) -> bool:
        """Backup the current configuration to database @zara

        Args:
            config_data: Configuration data to backup

        Returns:
            True if successful
        """
        try:
            await self._db.execute(
                """INSERT INTO GPM_config_backup (id, backup_time, config_json)
                   VALUES (1, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       backup_time = excluded.backup_time,
                       config_json = excluded.config_json""",
                (datetime.now().isoformat(), json.dumps(config_data)),
            )
            return True
        except Exception as err:
            _LOGGER.error("Failed to backup config: %s", err)
            return False

    async def async_restore_config(self) -> dict[str, Any] | None:
        """Restore configuration from database backup @zara

        Returns:
            Configuration data or None if no backup exists
        """
        try:
            row = await self._db.fetchone(
                "SELECT config_json FROM GPM_config_backup WHERE id = 1"
            )
            if row and row["config_json"]:
                return json.loads(row["config_json"])
        except Exception as err:
            _LOGGER.error("Failed to restore config: %s", err)
        return None

    def get_storage_info(self) -> dict[str, Any]:
        """Get information about storage usage @zara

        Returns:
            Dictionary with storage information
        """
        info = {
            "base_path": str(self._base_path),
            "exists": self._base_path.exists(),
            "db_path": self._db.db_path,
        }

        # Add DB file size
        db_file = Path(self._db.db_path)
        if db_file.exists():
            info["db_size_bytes"] = db_file.stat().st_size

        return info
