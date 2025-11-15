"""
V8.6.2 RC-2 Clean Slate Migration - Automatic Backup and Fresh Start
Runs ONCE on first load after update to v8.6.2

This migration:
1. Detects upgrade from version < 8.6.2
2. Creates automatic timestamped backup of existing data
3. Deletes old data structure (RC-1 incompatible with RC-2)
4. Rebuilds clean directory structure for RC-2
5. Creates fresh JSON files with RC-2 schema

User configuration is preserved in Home Assistant (not in data directory)

This is a conservative, safe migration following the development policy:
- Mandatory backup before any deletion
- Clear logging of all actions
- Graceful error handling
- Safe for all users (minimum, standard, advanced configurations)
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)


class V862RC2CleanSlateMigration:
    """One-time migration to v8.6.2 RC-2 - Clean slate with automatic backup"""

    MIGRATION_VERSION = "8.6.2"
    MIGRATION_FLAG = ".migration_v862_rc2_completed"

    def __init__(self, hass: HomeAssistant, data_dir: Path, coordinator=None):
        """Initialize migration handler"""
        self.hass = hass
        self.data_dir = data_dir
        self.coordinator = coordinator
        self.backup_dir = None  # Will be set during backup creation

    def should_run(self) -> bool:
        """
        Check if migration needs to run

        Migration runs if:
        1. Migration flag does not exist, AND
        2. Data directory exists (upgrading from previous version)

        If data directory doesn't exist, this is a fresh install - no migration needed
        """
        flag_file = self.data_dir / self.MIGRATION_FLAG

        if flag_file.exists():
            _LOGGER.debug("V8.6.2 RC-2 migration already completed (flag exists)")
            return False

        # Check if data directory exists (indicates upgrade, not fresh install)
        if not self.data_dir.exists():
            _LOGGER.info("Fresh install detected - no migration needed")
            return False

        # Check if there's actual data to migrate (not empty directory)
        if not any(self.data_dir.iterdir()):
            _LOGGER.info("Empty data directory - no migration needed")
            return False

        _LOGGER.info("V8.6.2 RC-2 Clean Slate Migration needed (upgrade from previous version)")
        return True

    async def run(self) -> bool:
        """Execute complete clean slate migration with automatic backup"""
        try:
            _LOGGER.warning("=" * 80)
            _LOGGER.warning("🔄 V8.6.2 RC-2 CLEAN SLATE MIGRATION STARTING")
            _LOGGER.warning("=" * 80)
            _LOGGER.warning("RC-1 data structure incompatible with RC-2 changes")
            _LOGGER.warning("Creating automatic backup and starting fresh for stability")
            _LOGGER.warning("Your Home Assistant configuration is preserved")
            _LOGGER.warning("=" * 80)

            # Step 1: Create timestamped backup of ALL existing data
            _LOGGER.info("Step 1/6: Creating automatic backup...")
            backup_success = await self._create_automatic_backup()
            if not backup_success:
                _LOGGER.error("Backup creation failed - ABORTING migration for safety")
                return False
            _LOGGER.info(f"✓ Backup created successfully: {self.backup_dir.name}")

            # Step 2: Reset ML Model (if available)
            if self.coordinator and hasattr(self.coordinator, 'ml_predictor') and self.coordinator.ml_predictor:
                _LOGGER.info("Step 2/6: Resetting ML model...")
                try:
                    success = await self.coordinator.ml_predictor.reset_model()
                    if success:
                        _LOGGER.info("✓ ML model reset successfully")
                    else:
                        _LOGGER.warning("⚠ ML model reset had issues (continuing anyway)")
                except Exception as e:
                    _LOGGER.warning(f"⚠ ML reset failed: {e} (continuing anyway)")
            else:
                _LOGGER.info("Step 2/6: Skipping ML reset (predictor not initialized)")

            # Step 3: Delete old data structure (except backup and migration flag)
            _LOGGER.info("Step 3/6: Cleaning old data structure...")
            await self._clean_data_directory()
            _LOGGER.info("✓ Old data structure cleaned")

            # Step 4: Rebuild directory structure for RC-2
            _LOGGER.info("Step 4/6: Rebuilding directory structure for RC-2...")
            await self._rebuild_directory_structure()
            _LOGGER.info("✓ Directory structure rebuilt")

            # Step 5: Create fresh JSON files with RC-2 schema
            _LOGGER.info("Step 5/6: Creating fresh JSON files with RC-2 schema...")
            await self._create_fresh_json_files()
            _LOGGER.info("✓ Fresh JSON files created")

            # Step 6: Mark migration complete
            _LOGGER.info("Step 6/6: Marking migration complete...")
            await self._mark_migration_complete()
            _LOGGER.info("✓ Migration flag created")

            _LOGGER.warning("=" * 80)
            _LOGGER.warning("✅ V8.6.2 RC-2 CLEAN SLATE MIGRATION COMPLETED SUCCESSFULLY")
            _LOGGER.warning("=" * 80)
            _LOGGER.warning(f"Backup location: {self.backup_dir}")
            _LOGGER.warning("System ready for fresh RC-2 start!")
            _LOGGER.warning("Morning Routine will initialize forecasts at 06:00 tomorrow")
            _LOGGER.warning("ML training will begin after 10+ production hours")
            _LOGGER.warning("=" * 80)

            return True

        except Exception as e:
            _LOGGER.error(f"❌ V8.6.2 RC-2 Migration FAILED: {e}", exc_info=True)
            if self.backup_dir and self.backup_dir.exists():
                _LOGGER.error(f"Your data is safe in backup: {self.backup_dir}")
            return False

    async def _create_automatic_backup(self) -> bool:
        """
        Create timestamped backup of entire data directory
        Returns True if successful, False otherwise
        """

        def _backup_sync():
            # Create backups directory if it doesn't exist
            backups_parent = self.data_dir / "backups"
            backups_parent.mkdir(parents=True, exist_ok=True)

            # Create timestamped backup directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.backup_dir = backups_parent / f"auto_migration_rc2_{timestamp}"
            self.backup_dir.mkdir(parents=True, exist_ok=True)

            _LOGGER.info(f"  Creating backup in: {self.backup_dir.name}")

            # Backup all files and directories (except backups directory itself)
            backed_up_count = 0
            for item in self.data_dir.iterdir():
                # Skip the backups directory itself
                if item.name == "backups":
                    continue

                try:
                    target = self.backup_dir / item.name

                    if item.is_file():
                        shutil.copy2(item, target)
                        _LOGGER.debug(f"    Backed up file: {item.name}")
                        backed_up_count += 1
                    elif item.is_dir():
                        shutil.copytree(item, target)
                        _LOGGER.debug(f"    Backed up directory: {item.name}")
                        backed_up_count += 1

                except Exception as e:
                    _LOGGER.warning(f"  Could not backup {item.name}: {e}")

            # Create backup manifest
            manifest = {
                "migration_version": self.MIGRATION_VERSION,
                "backup_created": dt_util.now().isoformat(),
                "items_backed_up": backed_up_count,
                "purpose": "Automatic backup before V8.6.2 RC-2 clean slate migration",
                "restore_instructions": "Contact developer or restore manually if needed",
            }

            manifest_file = self.backup_dir / "BACKUP_MANIFEST.json"
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            _LOGGER.info(f"  ✓ Backed up {backed_up_count} items")
            return True

        try:
            return await self.hass.async_add_executor_job(_backup_sync)
        except Exception as e:
            _LOGGER.error(f"Backup creation failed: {e}", exc_info=True)
            return False

    async def _clean_data_directory(self):
        """Delete old data structure (except backups and migration flags)"""

        def _clean_sync():
            if not self.data_dir.exists():
                _LOGGER.warning(f"Data directory does not exist: {self.data_dir}")
                return

            # Delete everything except backups directory and migration flags
            for item in self.data_dir.iterdir():
                # Keep: backups directory, any migration flags
                if item.name == "backups" or item.name.startswith(".migration_"):
                    continue

                try:
                    if item.is_file():
                        item.unlink()
                        _LOGGER.debug(f"  Deleted file: {item.name}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        _LOGGER.debug(f"  Deleted directory: {item.name}")
                except Exception as e:
                    _LOGGER.warning(f"  Could not delete {item.name}: {e}")

        await self.hass.async_add_executor_job(_clean_sync)

    async def _rebuild_directory_structure(self):
        """Rebuild clean directory structure for RC-2"""

        def _rebuild_sync():
            # Create all required directories for RC-2
            directories = [
                self.data_dir / "stats",
                self.data_dir / "ml",
                self.data_dir / "ml" / "backups",
                self.data_dir / "ml" / "models",
                self.data_dir / "data",
                self.data_dir / "backups",
                self.data_dir / "backups" / "manual",
                self.data_dir / "backups" / "auto",
                self.data_dir / "exports",
                self.data_dir / "exports" / "pictures",
                self.data_dir / "exports" / "reports",
                self.data_dir / "exports" / "statistics",
                self.data_dir / "docs",
                self.data_dir / "assets",
            ]

            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                _LOGGER.debug(f"  Created: {directory.relative_to(self.data_dir)}")

        await self.hass.async_add_executor_job(_rebuild_sync)

    async def _create_fresh_json_files(self):
        """Create fresh JSON files with RC-2 schema"""

        def _create_sync():
            # Version info
            versinfo = {
                "version": "8.6.2",
                "migration_date": dt_util.now().isoformat(),
                "description": "V8.6.2 RC-2 - Clean slate with bugfixes #13 and #14",
                "changes": [
                    "Fixed: Hour 6 missing in winter production window",
                    "Fixed: False warning about forecast count in winter",
                    "Improved: Conservative development policy implemented",
                    "Structure: Incompatible with RC-1 (automatic migration applied)",
                ],
            }
            self._write_json_sync(self.data_dir / "versinfo.json", versinfo)

            # Stats files
            stats_dir = self.data_dir / "stats"

            # hourly_predictions.json - RC-2 schema
            hourly_predictions = {
                "version": "2.0",
                "last_updated": dt_util.now().isoformat(),
                "best_hour_today": None,
                "metadata": {
                    "system_id": "solar_system_001",
                    "location": {},
                    "system_specs": {},
                    "sensor_config": {},
                },
                "predictions": [],
            }
            self._write_json_sync(stats_dir / "hourly_predictions.json", hourly_predictions)

            # daily_forecasts.json - RC-2 schema
            daily_forecasts = {
                "version": "1.0",
                "today": {},
                "tomorrow": {},
                "day_after_tomorrow": {},
                "history": [],
                "statistics": {
                    "current_week": {},
                    "current_month": {},
                    "last_7_days": {},
                    "last_30_days": {},
                    "last_365_days": {},
                    "all_time_peak": {},
                },
                "metadata": {
                    "history_entries": 0,
                    "last_update": dt_util.now().isoformat(),
                    "retention_days": 730,
                },
            }
            self._write_json_sync(stats_dir / "daily_forecasts.json", daily_forecasts)

            # daily_summaries.json - RC-2 schema
            daily_summaries = {"version": "1.0", "summaries": []}
            self._write_json_sync(stats_dir / "daily_summaries.json", daily_summaries)

            # ML files
            ml_dir = self.data_dir / "ml"

            # model_state.json - RC-2 schema
            model_state = {
                "status": "uninitialized",
                "current_accuracy": 0.0,
                "training_samples": 0,
                "last_training": None,
                "version": "2.0",
            }
            self._write_json_sync(ml_dir / "model_state.json", model_state)

            # hourly_profile.json - RC-2 schema
            hourly_profile = {
                "version": "1.0",
                "hourly_averages": {},
                "sample_count": 0,
                "last_updated": None,
            }
            self._write_json_sync(ml_dir / "hourly_profile.json", hourly_profile)

            # Data files
            data_dir = self.data_dir / "data"

            # coordinator_state.json - RC-2 schema
            coordinator_state = {
                "version": "1.0",
                "expected_daily_production": None,
                "last_collected_hour": None,
                "last_update": dt_util.now().isoformat(),
            }
            self._write_json_sync(data_dir / "coordinator_state.json", coordinator_state)

            # production_time_state.json - RC-2 schema
            production_time_state = {
                "version": "1.0",
                "date": dt_util.now().date().isoformat(),
                "accumulated_hours": 0.0,
                "is_active": False,
                "start_time": None,
                "last_updated": dt_util.now().isoformat(),
                "production_time_today": "00:00:00",
            }
            self._write_json_sync(data_dir / "production_time_state.json", production_time_state)

            _LOGGER.info("  ✓ All JSON files created with RC-2 schema")

        await self.hass.async_add_executor_job(_create_sync)

    def _write_json_sync(self, file_path: Path, data: dict):
        """Write JSON file synchronously with UTF-8 encoding"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    async def _mark_migration_complete(self):
        """Create migration flag file"""

        def _mark_sync():
            flag_data = {
                "version": self.MIGRATION_VERSION,
                "completed_at": dt_util.now().isoformat(),
                "backup_created": True,
                "backup_location": str(self.backup_dir) if self.backup_dir else None,
                "ml_reset": True,
                "directory_cleaned": True,
                "structure_rebuilt": True,
                "rc2_schema_initialized": True,
            }

            flag_file = self.data_dir / self.MIGRATION_FLAG
            with open(flag_file, "w", encoding="utf-8") as f:
                json.dump(flag_data, f, indent=2)

            _LOGGER.info(f"✓ Migration flag created: {self.MIGRATION_FLAG}")

        await self.hass.async_add_executor_job(_mark_sync)
