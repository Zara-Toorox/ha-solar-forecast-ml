# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from homeassistant.core import HomeAssistant

from ..const import BACKUP_RETENTION_DAYS, MAX_BACKUP_FILES
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from .data_io import DataManagerIO

_LOGGER = logging.getLogger(__name__)

class DataBackupHandler(DataManagerIO):
    """Handles backup creation and cleanup"""

    def __init__(self, hass: HomeAssistant, data_dir: Path):
        super().__init__(hass, data_dir)

    async def create_backup(
        self, backup_name: Optional[str] = None, backup_type: str = "manual"
    ) -> bool:
        """Create backup of all data files"""
        try:
            if not backup_name:
                timestamp = dt_util.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}"

            backup_dir = self.data_dir / "backups" / backup_type / backup_name
            await self._ensure_directory_exists(backup_dir)

            for subdir in ["stats", "data", "ml"]:
                src_dir = self.data_dir / subdir
                if src_dir.exists():
                    for file in src_dir.glob("*.json"):
                        dest_file = backup_dir / f"{subdir}_{file.name}"
                        shutil.copy2(file, dest_file)

            _LOGGER.info(f"Backup created: {backup_name} (type: {backup_type})")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to create backup: {e}")
            return False

    async def cleanup_old_backups(
        self, backup_type: str = "auto", retention_days: Optional[int] = None
    ) -> int:
        """Remove backups older than retention period"""
        try:
            backup_dir = self.data_dir / "backups" / backup_type
            if not backup_dir.exists():
                return 0

            if retention_days is None:
                retention_days = BACKUP_RETENTION_DAYS

            cutoff_date = dt_util.now() - timedelta(days=retention_days)
            removed_count = 0

            for backup_folder in backup_dir.iterdir():
                if backup_folder.is_dir():
                    folder_time = datetime.fromtimestamp(backup_folder.stat().st_mtime)

                    folder_time = dt_util.ensure_local(folder_time)

                    if folder_time < cutoff_date:
                        shutil.rmtree(backup_folder)
                        removed_count += 1
                        _LOGGER.info(f"Removed old backup: {backup_folder.name}")

            if removed_count > 0:
                _LOGGER.info(
                    f"Cleanup completed: removed {removed_count} backups "
                    f"older than {retention_days} days (type: {backup_type})"
                )

            return removed_count

        except Exception as e:
            _LOGGER.error(f"Failed to cleanup old backups: {e}")
            return 0

    async def cleanup_excess_backups(
        self, backup_type: str = "auto", max_backups: Optional[int] = None
    ) -> int:
        """Remove oldest backups if count exceeds maximum"""
        try:
            backup_dir = self.data_dir / "backups" / backup_type
            if not backup_dir.exists():
                return 0

            if max_backups is None:
                max_backups = MAX_BACKUP_FILES

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
                _LOGGER.info(f"Removed excess backup: {oldest_folder.name}")

            if removed_count > 0:
                _LOGGER.info(
                    f"Cleanup completed: removed {removed_count} excess backups "
                    f"(type: {backup_type}, max: {max_backups})"
                )

            return removed_count

        except Exception as e:
            _LOGGER.error(f"Failed to cleanup excess backups: {e}")
            return 0

    async def list_backups(self, backup_type: Optional[str] = None) -> list:
        """List all available backups @zara"""
        try:
            backups = []

            backup_types = [backup_type] if backup_type else ["auto", "manual"]

            for btype in backup_types:
                backup_dir = self.data_dir / "backups" / btype
                if not backup_dir.exists():
                    continue

                for backup_folder in backup_dir.iterdir():
                    if backup_folder.is_dir():
                        stat = backup_folder.stat()

                        total_size = sum(
                            f.stat().st_size for f in backup_folder.rglob("*") if f.is_file()
                        )

                        backups.append(
                            {
                                "name": backup_folder.name,
                                "type": btype,
                                "date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                "size_bytes": total_size,
                                "size_mb": round(total_size / (1024 * 1024), 2),
                                "path": str(backup_folder),
                            }
                        )

            backups.sort(key=lambda x: x["date"], reverse=True)

            return backups

        except Exception as e:
            _LOGGER.error(f"Failed to list backups: {e}")
            return []

    async def restore_backup(self, backup_name: str, backup_type: str = "manual") -> bool:
        """Restore data from backup @zara"""
        try:
            backup_dir = self.data_dir / "backups" / backup_type / backup_name

            if not backup_dir.exists():
                _LOGGER.error(f"Backup not found: {backup_name}")
                return False

            await self.create_backup(
                backup_name=f"pre_restore_{dt_util.now().strftime('%Y%m%d_%H%M%S')}",
                backup_type="auto",
            )

            restored_count = 0
            for backup_file in backup_dir.glob("*.json"):

                parts = backup_file.stem.split("_", 1)
                if len(parts) == 2:
                    subdir, filename = parts
                    dest_file = self.data_dir / subdir / f"{filename}.json"

                    await self._ensure_directory_exists(dest_file.parent)
                    shutil.copy2(backup_file, dest_file)
                    restored_count += 1

            _LOGGER.info(f"Backup restored: {backup_name} ({restored_count} files)")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to restore backup: {e}")
            return False

    async def delete_backup(self, backup_name: str, backup_type: str = "manual") -> bool:
        """Delete specific backup @zara"""
        try:
            backup_dir = self.data_dir / "backups" / backup_type / backup_name

            if not backup_dir.exists():
                _LOGGER.warning(f"Backup not found: {backup_name}")
                return False

            shutil.rmtree(backup_dir)
            _LOGGER.info(f"Backup deleted: {backup_name} (type: {backup_type})")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to delete backup: {e}")
            return False

    async def get_backup_info(
        self, backup_name: str, backup_type: str = "manual"
    ) -> Optional[dict]:
        """Get detailed info about specific backup"""
        try:
            backup_dir = self.data_dir / "backups" / backup_type / backup_name

            if not backup_dir.exists():
                return None

            stat = backup_dir.stat()

            files = []
            total_size = 0
            for backup_file in backup_dir.glob("*.json"):
                file_size = backup_file.stat().st_size
                total_size += file_size
                files.append(
                    {
                        "name": backup_file.name,
                        "size_bytes": file_size,
                        "size_kb": round(file_size / 1024, 2),
                    }
                )

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

        except Exception as e:
            _LOGGER.error(f"Failed to get backup info: {e}")
            return None
