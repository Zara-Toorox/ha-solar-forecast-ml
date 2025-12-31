# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import asyncio
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..const import BACKUP_RETENTION_DAYS, MAX_BACKUP_FILES
from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)

class DataPersistence:
    """Handles data backup restoration and migration"""

    def __init__(self, data_dir: Path):
        """Initialize data persistence handler @zara"""
        self.data_dir = data_dir
        self.backup_dir = data_dir / "backups" / "auto"
        self.manual_backup_dir = data_dir / "backups" / "manual"

    async def create_backup(
        self, name: Optional[str] = None, manual: bool = False
    ) -> Optional[Path]:
        """Create a backup of all data files"""
        try:

            backup_base = self.manual_backup_dir if manual else self.backup_dir
            backup_base.mkdir(parents=True, exist_ok=True)

            timestamp = dt_util.now().strftime("%Y%m%d_%H%M%S")
            if name:
                backup_name = f"{name}_{timestamp}"
            else:
                backup_name = f"backup_{timestamp}"

            backup_file = backup_base / f"{backup_name}.tar.gz"

            def create_tarball():
                import tarfile

                with tarfile.open(backup_file, "w:gz") as tar:
                    for subdir in ["ml", "stats", "data"]:
                        source_dir = self.data_dir / subdir
                        if source_dir.exists():
                            tar.add(source_dir, arcname=subdir)

            await asyncio.get_event_loop().run_in_executor(None, create_tarball)

            _LOGGER.info(f"Backup created: {backup_file}")

            if not manual:
                await self._cleanup_old_backups()

            return backup_file

        except Exception as e:
            _LOGGER.error(f"Failed to create backup: {e}", exc_info=True)
            return None

    async def restore_backup(self, backup_file: Path) -> bool:
        """Restore data from a backup file @zara"""
        try:
            if not backup_file.exists():
                _LOGGER.error(f"Backup file not found: {backup_file}")
                return False

            restore_dir = self.data_dir / "temp_restore"
            restore_dir.mkdir(parents=True, exist_ok=True)

            def extract_tarball():
                import tarfile

                with tarfile.open(backup_file, "r:gz") as tar:
                    tar.extractall(restore_dir)

            await asyncio.get_event_loop().run_in_executor(None, extract_tarball)

            for subdir in ["ml", "stats", "data"]:
                source = restore_dir / subdir
                target = self.data_dir / subdir

                if source.exists():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.move(str(source), str(target))

            shutil.rmtree(restore_dir)

            _LOGGER.info(f"Backup restored from: {backup_file}")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to restore backup: {e}", exc_info=True)
            return False

    async def _cleanup_old_backups(self) -> None:
        """Remove old automatic backups based on retention policy @zara"""
        try:
            if not self.backup_dir.exists():
                return

            backups = sorted(self.backup_dir.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime)

            if len(backups) > MAX_BACKUP_FILES:
                for backup in backups[:-MAX_BACKUP_FILES]:
                    backup.unlink()
                    _LOGGER.debug(f"Removed old backup (count limit): {backup.name}")

            cutoff_date = dt_util.now() - timedelta(days=BACKUP_RETENTION_DAYS)
            for backup in backups:
                mtime = datetime.fromtimestamp(backup.stat().st_mtime)
                if mtime < cutoff_date:
                    backup.unlink()
                    _LOGGER.debug(f"Removed old backup (age limit): {backup.name}")

        except Exception as e:
            _LOGGER.error(f"Failed to cleanup old backups: {e}")

    async def list_backups(self, manual: bool = False) -> List[Dict[str, Any]]:
        """List available backups @zara"""
        try:
            backup_base = self.manual_backup_dir if manual else self.backup_dir

            if not backup_base.exists():
                return []

            backups = []
            for backup_file in sorted(
                backup_base.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True
            ):
                stat = backup_file.stat()
                backups.append(
                    {
                        "name": backup_file.stem,
                        "path": str(backup_file),
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": "manual" if manual else "automatic",
                    }
                )

            return backups

        except Exception as e:
            _LOGGER.error(f"Failed to list backups: {e}")
            return []

    async def delete_backup(self, backup_name: str, manual: bool = False) -> bool:
        """Delete a specific backup @zara"""
        try:
            backup_base = self.manual_backup_dir if manual else self.backup_dir
            backup_file = backup_base / f"{backup_name}.tar.gz"

            if not backup_file.exists():
                _LOGGER.error(f"Backup not found: {backup_name}")
                return False

            backup_file.unlink()
            _LOGGER.info(f"Deleted backup: {backup_name}")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to delete backup: {e}")
            return False
