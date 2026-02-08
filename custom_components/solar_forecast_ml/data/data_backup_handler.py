# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Backup Handler for Solar Forecast ML V16.2.0.
Manages database backup creation, cleanup, and restoration.
Uses DatabaseManager for all database operations.

@zara
"""

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from homeassistant.core import HomeAssistant

from ..const import BACKUP_RETENTION_DAYS, MAX_BACKUP_FILES
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from .db_manager import DatabaseManager
from .data_io import DataManagerIO

_LOGGER = logging.getLogger(__name__)


class DataBackupHandler(DataManagerIO):
    """Handles database backup creation and management. @zara

    Provides methods to:
    - Create automatic and manual backups
    - Cleanup old backups based on retention policy
    - List, restore, and delete backups
    - Get backup information
    """

    def __init__(
        self,
        hass: HomeAssistant,
        db_manager: DatabaseManager,
        data_dir: Path
    ):
        """Initialize the backup handler. @zara

        Args:
            hass: Home Assistant instance
            db_manager: DatabaseManager for database operations
            data_dir: Base data directory path
        """
        super().__init__(hass, db_manager)
        self.data_dir = Path(data_dir)
        self.backup_base = self.data_dir / "backups"

    def _get_backup_dir(self, backup_type: str) -> Path:
        """Get backup directory for a given type. @zara"""
        return self.backup_base / backup_type

    async def create_backup(
        self,
        backup_name: Optional[str] = None,
        backup_type: str = "manual"
    ) -> bool:
        """Create backup of the database. @zara

        Args:
            backup_name: Optional name for the backup
            backup_type: Type of backup ('auto' or 'manual')

        Returns:
            True if backup was created successfully
        """
        try:
            # Generate backup name if not provided @zara
            if not backup_name:
                timestamp = dt_util.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}"

            backup_dir = self._get_backup_dir(backup_type) / backup_name

            # Ensure backup directory exists @zara
            def ensure_and_copy():
                backup_dir.mkdir(parents=True, exist_ok=True)

                # Copy database file @zara
                db_path = Path(self.db.db_path)
                if db_path.exists():
                    dest_file = backup_dir / "solar_forecast.db"
                    shutil.copy2(db_path, dest_file)
                    return True
                return False

            success = await self.hass.async_add_executor_job(ensure_and_copy)

            if success:
                _LOGGER.info("Backup created: %s (type: %s)", backup_name, backup_type)
            else:
                _LOGGER.warning("Database file not found for backup")

            return success

        except Exception as e:
            _LOGGER.error("Failed to create backup: %s", e)
            return False

    async def cleanup_old_backups(
        self,
        backup_type: str = "auto",
        retention_days: Optional[int] = None
    ) -> int:
        """Remove backups older than retention period. @zara

        Args:
            backup_type: Type of backups to clean up
            retention_days: Number of days to retain (default from const)

        Returns:
            Number of backups removed
        """
        try:
            backup_dir = self._get_backup_dir(backup_type)

            def check_dir():
                return backup_dir.exists()

            if not await self.hass.async_add_executor_job(check_dir):
                return 0

            if retention_days is None:
                retention_days = BACKUP_RETENTION_DAYS

            cutoff_date = dt_util.now() - timedelta(days=retention_days)

            def do_cleanup():
                removed_count = 0
                for backup_folder in backup_dir.iterdir():
                    if backup_folder.is_dir():
                        folder_time = datetime.fromtimestamp(
                            backup_folder.stat().st_mtime
                        )
                        folder_time = dt_util.ensure_local(folder_time)

                        if folder_time < cutoff_date:
                            shutil.rmtree(backup_folder)
                            removed_count += 1
                            _LOGGER.info("Removed old backup: %s", backup_folder.name)

                return removed_count

            removed = await self.hass.async_add_executor_job(do_cleanup)

            if removed > 0:
                _LOGGER.info(
                    "Cleanup completed: removed %d backups older than %d days (type: %s)",
                    removed, retention_days, backup_type
                )

            return removed

        except Exception as e:
            _LOGGER.error("Failed to cleanup old backups: %s", e)
            return 0

    async def cleanup_excess_backups(
        self,
        backup_type: str = "auto",
        max_backups: Optional[int] = None
    ) -> int:
        """Remove oldest backups if count exceeds maximum. @zara

        Args:
            backup_type: Type of backups to clean up
            max_backups: Maximum number of backups to keep

        Returns:
            Number of backups removed
        """
        try:
            backup_dir = self._get_backup_dir(backup_type)

            def check_dir():
                return backup_dir.exists()

            if not await self.hass.async_add_executor_job(check_dir):
                return 0

            if max_backups is None:
                max_backups = MAX_BACKUP_FILES

            def do_cleanup():
                backup_folders = [
                    (folder, folder.stat().st_mtime)
                    for folder in backup_dir.iterdir()
                    if folder.is_dir()
                ]
                backup_folders.sort(key=lambda x: x[1])

                removed_count = 0
                while len(backup_folders) > max_backups:
                    oldest_folder, _ = backup_folders.pop(0)
                    shutil.rmtree(oldest_folder)
                    removed_count += 1
                    _LOGGER.info("Removed excess backup: %s", oldest_folder.name)

                return removed_count

            removed = await self.hass.async_add_executor_job(do_cleanup)

            if removed > 0:
                _LOGGER.info(
                    "Cleanup completed: removed %d excess backups (type: %s, max: %d)",
                    removed, backup_type, max_backups
                )

            return removed

        except Exception as e:
            _LOGGER.error("Failed to cleanup excess backups: %s", e)
            return 0

    async def list_backups(self, backup_type: Optional[str] = None) -> list[dict]:
        """List all available backups. @zara

        Args:
            backup_type: Optional filter by type ('auto' or 'manual')

        Returns:
            List of backup information dictionaries
        """
        try:
            backup_types = [backup_type] if backup_type else ["auto", "manual"]

            def get_backups():
                backups = []
                for btype in backup_types:
                    backup_dir = self._get_backup_dir(btype)
                    if not backup_dir.exists():
                        continue

                    for backup_folder in backup_dir.iterdir():
                        if backup_folder.is_dir():
                            stat = backup_folder.stat()

                            # Calculate total size @zara
                            total_size = sum(
                                f.stat().st_size
                                for f in backup_folder.rglob("*")
                                if f.is_file()
                            )

                            backups.append({
                                "name": backup_folder.name,
                                "type": btype,
                                "date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                "size_bytes": total_size,
                                "size_mb": round(total_size / (1024 * 1024), 2),
                                "path": str(backup_folder),
                            })

                backups.sort(key=lambda x: x["date"], reverse=True)
                return backups

            return await self.hass.async_add_executor_job(get_backups)

        except Exception as e:
            _LOGGER.error("Failed to list backups: %s", e)
            return []

    async def restore_backup(
        self,
        backup_name: str,
        backup_type: str = "manual"
    ) -> bool:
        """Restore database from backup. @zara

        Args:
            backup_name: Name of the backup to restore
            backup_type: Type of backup

        Returns:
            True if restore was successful
        """
        try:
            backup_dir = self._get_backup_dir(backup_type) / backup_name

            def check_exists():
                return backup_dir.exists()

            if not await self.hass.async_add_executor_job(check_exists):
                _LOGGER.error("Backup not found: %s", backup_name)
                return False

            # Create pre-restore backup @zara
            await self.create_backup(
                backup_name=f"pre_restore_{dt_util.now().strftime('%Y%m%d_%H%M%S')}",
                backup_type="auto",
            )

            # Close database connection @zara
            await self.db.close()

            # Restore database file @zara
            def do_restore():
                db_path = Path(self.db.db_path)
                backup_db = backup_dir / "solar_forecast.db"

                if backup_db.exists():
                    shutil.copy2(backup_db, db_path)
                    return True
                return False

            success = await self.hass.async_add_executor_job(do_restore)

            # Reconnect to database @zara
            await self.db.connect()

            if success:
                _LOGGER.info("Backup restored: %s", backup_name)
            else:
                _LOGGER.error("Backup database file not found: %s", backup_name)

            return success

        except Exception as e:
            _LOGGER.error("Failed to restore backup: %s", e)
            # Try to reconnect @zara
            try:
                await self.db.connect()
            except Exception:
                pass
            return False

    async def delete_backup(
        self,
        backup_name: str,
        backup_type: str = "manual"
    ) -> bool:
        """Delete a specific backup. @zara

        Args:
            backup_name: Name of the backup to delete
            backup_type: Type of backup

        Returns:
            True if deleted successfully
        """
        try:
            backup_dir = self._get_backup_dir(backup_type) / backup_name

            def do_delete():
                if not backup_dir.exists():
                    return False
                shutil.rmtree(backup_dir)
                return True

            success = await self.hass.async_add_executor_job(do_delete)

            if success:
                _LOGGER.info("Backup deleted: %s (type: %s)", backup_name, backup_type)
            else:
                _LOGGER.warning("Backup not found: %s", backup_name)

            return success

        except Exception as e:
            _LOGGER.error("Failed to delete backup: %s", e)
            return False

    async def get_backup_info(
        self,
        backup_name: str,
        backup_type: str = "manual"
    ) -> Optional[dict[str, Any]]:
        """Get detailed information about a specific backup. @zara

        Args:
            backup_name: Name of the backup
            backup_type: Type of backup

        Returns:
            Backup information dictionary or None
        """
        try:
            backup_dir = self._get_backup_dir(backup_type) / backup_name

            def get_info():
                if not backup_dir.exists():
                    return None

                stat = backup_dir.stat()
                files = []
                total_size = 0

                for backup_file in backup_dir.glob("*"):
                    if backup_file.is_file():
                        file_size = backup_file.stat().st_size
                        total_size += file_size
                        files.append({
                            "name": backup_file.name,
                            "size_bytes": file_size,
                            "size_kb": round(file_size / 1024, 2),
                        })

                return {
                    "name": backup_name,
                    "type": backup_type,
                    "date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "files_count": len(files),
                    "files": files,
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "path": str(backup_dir),
                }

            return await self.hass.async_add_executor_job(get_info)

        except Exception as e:
            _LOGGER.error("Failed to get backup info: %s", e)
            return None
