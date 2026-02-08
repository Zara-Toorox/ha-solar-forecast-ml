# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Data Persistence Layer for Solar Forecast ML V16.2.0.
Handles database backup, restore, and data migration.
All operations use DatabaseManager for SQLite access.

@zara
"""

import asyncio
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from homeassistant.core import HomeAssistant

from ..const import BACKUP_RETENTION_DAYS, MAX_BACKUP_FILES
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from .db_manager import DatabaseManager

_LOGGER = logging.getLogger(__name__)


class DataPersistence:
    """Handles database backup, restoration, and data migration. @zara

    Provides methods to:
    - Create database backups (copies of the SQLite file)
    - Restore from backups
    - Migrate legacy JSON data to database
    - Clean up old backups based on retention policy
    """

    def __init__(
        self,
        hass: HomeAssistant,
        db_manager: DatabaseManager,
        data_dir: Path
    ):
        """Initialize data persistence handler. @zara

        Args:
            hass: Home Assistant instance
            db_manager: DatabaseManager for database operations
            data_dir: Base data directory path
        """
        self.hass = hass
        self.db = db_manager
        self.data_dir = Path(data_dir)
        self.backup_dir = self.data_dir / "backups" / "auto"
        self.manual_backup_dir = self.data_dir / "backups" / "manual"

    async def create_backup(
        self,
        name: Optional[str] = None,
        manual: bool = False
    ) -> Optional[Path]:
        """Create a backup of the database file. @zara

        Args:
            name: Optional backup name
            manual: If True, store in manual backup directory

        Returns:
            Path to backup file or None on failure
        """
        try:
            backup_base = self.manual_backup_dir if manual else self.backup_dir

            # Ensure backup directory exists @zara
            def ensure_dir():
                backup_base.mkdir(parents=True, exist_ok=True)

            await self.hass.async_add_executor_job(ensure_dir)

            # Generate backup filename @zara
            timestamp = dt_util.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{name}_{timestamp}" if name else f"backup_{timestamp}"
            backup_file = backup_base / f"{backup_name}.db"

            # Copy the database file @zara
            def copy_db():
                db_path = Path(self.db.db_path)
                if db_path.exists():
                    shutil.copy2(db_path, backup_file)
                    return True
                return False

            success = await self.hass.async_add_executor_job(copy_db)

            if success:
                _LOGGER.info("Database backup created: %s", backup_file)

                # Cleanup old backups if automatic @zara
                if not manual:
                    await self._cleanup_old_backups()

                return backup_file
            else:
                _LOGGER.error("Database file not found for backup")
                return None

        except Exception as e:
            _LOGGER.error("Failed to create backup: %s", e, exc_info=True)
            return None

    async def restore_backup(self, backup_file: Path) -> bool:
        """Restore database from a backup file. @zara

        Args:
            backup_file: Path to the backup file

        Returns:
            True if restore was successful
        """
        try:
            def check_exists():
                return backup_file.exists()

            if not await self.hass.async_add_executor_job(check_exists):
                _LOGGER.error("Backup file not found: %s", backup_file)
                return False

            # Close current database connection @zara
            await self.db.close()

            # Create pre-restore backup @zara
            db_path = Path(self.db.db_path)

            def do_restore():
                # Backup current database before restore @zara
                if db_path.exists():
                    pre_restore = db_path.with_suffix(".db.pre_restore")
                    shutil.copy2(db_path, pre_restore)

                # Restore from backup @zara
                shutil.copy2(backup_file, db_path)

            await self.hass.async_add_executor_job(do_restore)

            # Reconnect to restored database @zara
            await self.db.connect()

            _LOGGER.info("Database restored from: %s", backup_file)
            return True

        except Exception as e:
            _LOGGER.error("Failed to restore backup: %s", e, exc_info=True)
            # Try to reconnect to database @zara
            try:
                await self.db.connect()
            except Exception:
                pass
            return False

    async def _cleanup_old_backups(self) -> None:
        """Remove old automatic backups based on retention policy. @zara"""
        try:
            def check_dir():
                return self.backup_dir.exists()

            if not await self.hass.async_add_executor_job(check_dir):
                return

            def do_cleanup():
                backups = sorted(
                    self.backup_dir.glob("*.db"),
                    key=lambda p: p.stat().st_mtime
                )

                removed_count = 0

                # Remove if exceeding max backup count @zara
                if len(backups) > MAX_BACKUP_FILES:
                    for backup in backups[:-MAX_BACKUP_FILES]:
                        backup.unlink()
                        removed_count += 1
                        _LOGGER.debug("Removed old backup (count limit): %s", backup.name)

                # Remove if older than retention period @zara
                cutoff_date = dt_util.now() - timedelta(days=BACKUP_RETENTION_DAYS)
                remaining_backups = list(self.backup_dir.glob("*.db"))
                for backup in remaining_backups:
                    mtime = datetime.fromtimestamp(backup.stat().st_mtime)
                    mtime = dt_util.ensure_local(mtime)
                    if mtime < cutoff_date:
                        backup.unlink()
                        removed_count += 1
                        _LOGGER.debug("Removed old backup (age limit): %s", backup.name)

                return removed_count

            removed = await self.hass.async_add_executor_job(do_cleanup)
            if removed > 0:
                _LOGGER.info("Cleaned up %d old backups", removed)

        except Exception as e:
            _LOGGER.error("Failed to cleanup old backups: %s", e)

    async def list_backups(self, manual: bool = False) -> list[dict[str, Any]]:
        """List available backups. @zara

        Args:
            manual: If True, list manual backups, else automatic

        Returns:
            List of backup info dictionaries
        """
        try:
            backup_base = self.manual_backup_dir if manual else self.backup_dir

            def get_backups():
                if not backup_base.exists():
                    return []

                backups = []
                for backup_file in sorted(
                    backup_base.glob("*.db"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                ):
                    stat = backup_file.stat()
                    backups.append({
                        "name": backup_file.stem,
                        "path": str(backup_file),
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": "manual" if manual else "automatic",
                    })
                return backups

            return await self.hass.async_add_executor_job(get_backups)

        except Exception as e:
            _LOGGER.error("Failed to list backups: %s", e)
            return []

    async def delete_backup(self, backup_name: str, manual: bool = False) -> bool:
        """Delete a specific backup. @zara

        Args:
            backup_name: Name of the backup (without extension)
            manual: If True, look in manual backup directory

        Returns:
            True if deleted successfully
        """
        try:
            backup_base = self.manual_backup_dir if manual else self.backup_dir
            backup_file = backup_base / f"{backup_name}.db"

            def do_delete():
                if not backup_file.exists():
                    return False
                backup_file.unlink()
                return True

            success = await self.hass.async_add_executor_job(do_delete)

            if success:
                _LOGGER.info("Deleted backup: %s", backup_name)
            else:
                _LOGGER.error("Backup not found: %s", backup_name)

            return success

        except Exception as e:
            _LOGGER.error("Failed to delete backup: %s", e)
            return False

    async def vacuum_database(self) -> bool:
        """Vacuum the database to reclaim space. @zara

        Returns:
            True if vacuum was successful
        """
        try:
            await self.db.vacuum()
            _LOGGER.info("Database vacuum completed")
            return True
        except Exception as e:
            _LOGGER.error("Failed to vacuum database: %s", e)
            return False

    async def get_database_info(self) -> dict[str, Any]:
        """Get information about the database. @zara

        Returns:
            Dictionary with database information
        """
        try:
            db_size = await self.db.get_db_size()
            db_path = Path(self.db.db_path)

            def get_file_info():
                if db_path.exists():
                    stat = db_path.stat()
                    return {
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }
                return {"created": None, "modified": None}

            file_info = await self.hass.async_add_executor_job(get_file_info)

            return {
                "path": str(db_path),
                "size_bytes": db_size,
                "size_mb": round(db_size / (1024 * 1024), 2),
                "exists": db_path.exists(),
                "connected": self.db._db is not None,
                **file_info,
            }

        except Exception as e:
            _LOGGER.error("Failed to get database info: %s", e)
            return {"error": str(e)}
