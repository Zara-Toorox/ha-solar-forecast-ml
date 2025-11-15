"""
V7 Clean Slate Migration - Complete Fresh Start
Runs ONCE on first load after update to v7.0

This migration:
1. Resets ML model completely
2. Deletes ALL files in /config/solar_forecast_ml/
3. Rebuilds clean directory structure
4. Creates fresh JSON files
5. Deploys documentation

User configuration is preserved in Home Assistant (not in data directory)
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


class V7CleanSlateMigration:
    """One-time migration to v7.0 - Complete fresh start"""

    MIGRATION_VERSION = "7.0.0"
    MIGRATION_FLAG = ".migration_v7_completed"

    def __init__(self, hass: HomeAssistant, data_dir: Path, coordinator=None):
        """Initialize migration handler"""
        self.hass = hass
        self.data_dir = data_dir
        self.coordinator = coordinator
        # Source docs directory in custom_components
        self.source_docs_dir = Path(__file__).parent.parent / "docs"

    def should_run(self) -> bool:
        """Check if migration needs to run"""
        flag_file = self.data_dir / self.MIGRATION_FLAG

        if flag_file.exists():
            _LOGGER.debug("V7 migration already completed (flag exists)")
            return False

        _LOGGER.info("V7 Clean Slate Migration needed")
        return True

    async def run(self) -> bool:
        """Execute complete clean slate migration"""
        try:
            _LOGGER.warning("=" * 80)
            _LOGGER.warning("🔄 V7 CLEAN SLATE MIGRATION STARTING")
            _LOGGER.warning("=" * 80)
            _LOGGER.warning("This will reset ML and rebuild data structure for stability")
            _LOGGER.warning("Your Home Assistant configuration is preserved")
            _LOGGER.warning("=" * 80)

            # Step 1: Reset ML Model (use existing service!)
            if self.coordinator and self.coordinator.ml_predictor:
                _LOGGER.info("Step 1/5: Resetting ML model...")
                success = await self.coordinator.ml_predictor.reset_model()
                if success:
                    _LOGGER.info("✓ ML model reset successfully")
                else:
                    _LOGGER.warning("⚠ ML model reset had issues (continuing anyway)")
            else:
                _LOGGER.info("Step 1/5: Skipping ML reset (predictor not initialized)")

            # Step 2: Delete ALL files in data directory (except migration flag)
            _LOGGER.info("Step 2/5: Cleaning data directory...")
            await self._clean_data_directory()
            _LOGGER.info("✓ Data directory cleaned")

            # Step 3: Rebuild directory structure
            _LOGGER.info("Step 3/5: Rebuilding directory structure...")
            await self._rebuild_directory_structure()
            _LOGGER.info("✓ Directory structure rebuilt")

            # Step 4: Create fresh JSON files
            _LOGGER.info("Step 4/5: Creating fresh JSON files...")
            await self._create_fresh_json_files()
            _LOGGER.info("✓ Fresh JSON files created")

            # Step 5: Deploy documentation
            _LOGGER.info("Step 5/5: Deploying documentation...")
            await self._deploy_documentation()
            _LOGGER.info("✓ Documentation deployed")

            # Mark migration complete
            await self._mark_migration_complete()

            _LOGGER.warning("=" * 80)
            _LOGGER.warning("✅ V7 CLEAN SLATE MIGRATION COMPLETED SUCCESSFULLY")
            _LOGGER.warning("=" * 80)
            _LOGGER.warning("System ready for fresh start!")
            _LOGGER.warning("ML training will begin after 10+ production hours")
            _LOGGER.warning("=" * 80)

            return True

        except Exception as e:
            _LOGGER.error(f"❌ V7 Migration FAILED: {e}", exc_info=True)
            return False

    async def _clean_data_directory(self):
        """Delete all files and subdirectories in data directory"""

        def _clean_sync():
            if not self.data_dir.exists():
                _LOGGER.warning(f"Data directory does not exist: {self.data_dir}")
                return

            # Delete everything except the migration flag (if it exists)
            for item in self.data_dir.iterdir():
                if item.name == self.MIGRATION_FLAG:
                    continue  # Keep migration flag

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
        """Rebuild clean directory structure"""

        def _rebuild_sync():
            # Create all required directories
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
        """Create fresh JSON files with clean defaults"""

        def _create_sync():
            # Version info
            versinfo = {
                "version": "7.0.0",
                "migration_date": dt_util.now().isoformat(),
                "description": "V7 Clean Slate - Fresh start with bug fixes",
            }
            self._write_json_sync(self.data_dir / "versinfo.json", versinfo)

            # Stats files
            stats_dir = self.data_dir / "stats"

            # hourly_predictions.json
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

            # daily_forecasts.json
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

            # daily_summaries.json
            daily_summaries = {"version": "1.0", "summaries": []}
            self._write_json_sync(stats_dir / "daily_summaries.json", daily_summaries)

            # ML files
            ml_dir = self.data_dir / "ml"

            # model_state.json
            model_state = {
                "status": "uninitialized",
                "current_accuracy": 0.0,
                "training_samples": 0,
                "last_training": None,
                "version": "2.0",
            }
            self._write_json_sync(ml_dir / "model_state.json", model_state)

            # hourly_profile.json
            hourly_profile = {
                "version": "1.0",
                "hourly_averages": {},
                "sample_count": 0,
                "last_updated": None,
            }
            self._write_json_sync(ml_dir / "hourly_profile.json", hourly_profile)

            # Data files
            data_dir = self.data_dir / "data"

            # coordinator_state.json
            coordinator_state = {
                "expected_daily_production": None,
                "last_collected_hour": None,
                "last_update": dt_util.now().isoformat(),
            }
            self._write_json_sync(data_dir / "coordinator_state.json", coordinator_state)

            # production_time_state.json
            production_time_state = {
                "is_active": False,
                "accumulated_hours": 0.0,
                "start_time": None,
                "last_update": dt_util.now().isoformat(),
            }
            self._write_json_sync(data_dir / "production_time_state.json", production_time_state)

            _LOGGER.info("  ✓ All JSON files created with clean defaults")

        await self.hass.async_add_executor_job(_create_sync)

    def _write_json_sync(self, file_path: Path, data: dict):
        """Write JSON file synchronously with UTF-8 encoding"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    async def _deploy_documentation(self):
        """Deploy documentation files from component docs/ to data docs/"""

        def _deploy_sync():
            target_docs_dir = self.data_dir / "docs"

            # Check if source docs directory exists
            if not self.source_docs_dir.exists():
                _LOGGER.warning(f"Source docs directory not found: {self.source_docs_dir}")
                _LOGGER.warning("Skipping documentation deployment")
                return

            # Copy all files from source to target
            deployed_count = 0
            for source_file in self.source_docs_dir.iterdir():
                if source_file.is_file():
                    target_file = target_docs_dir / source_file.name
                    shutil.copy2(source_file, target_file)
                    _LOGGER.debug(f"  ✓ Deployed: {source_file.name}")
                    deployed_count += 1

            if deployed_count > 0:
                _LOGGER.info(f"  ✓ Deployed {deployed_count} documentation file(s)")
            else:
                _LOGGER.warning("  No documentation files found to deploy")

        await self.hass.async_add_executor_job(_deploy_sync)

    async def _mark_migration_complete(self):
        """Create migration flag file"""

        def _mark_sync():
            flag_data = {
                "version": self.MIGRATION_VERSION,
                "completed_at": dt_util.now().isoformat(),
                "ml_reset": True,
                "directory_cleaned": True,
                "structure_rebuilt": True,
                "docs_deployed": True,
            }

            flag_file = self.data_dir / self.MIGRATION_FLAG
            with open(flag_file, "w", encoding="utf-8") as f:
                json.dump(flag_data, f, indent=2)

            _LOGGER.info(f"✓ Migration flag created: {self.MIGRATION_FLAG}")

        await self.hass.async_add_executor_job(_mark_sync)
