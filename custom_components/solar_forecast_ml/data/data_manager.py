"""
Data Manager for Solar Forecast ML Integration - FACADE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from ..ml.ml_types import HourlyProfile, LearnedWeights
from .data_backup_handler import DataBackupHandler
from .data_daily_summaries import DailySummariesHandler
from .data_forecast_handler import DataForecastHandler
from .data_hourly_predictions import HourlyPredictionsHandler
from .data_io import DataManagerIO
from .data_ml_handler import DataMLHandler
from .data_prediction_handler import DataPredictionHandler
from .data_schema_validator import DataSchemaValidator
from .data_state_handler import DataStateHandler
from .data_weather_accuracy import WeatherAccuracyTracker

_LOGGER = logging.getLogger(__name__)


class DataManager(DataManagerIO):
    """Data Manager API for Solar Forecast ML"""

    def __init__(self, hass: HomeAssistant, entry_id: str, data_dir: Path, error_handler=None):
        """Initialize the Data Manager Facade"""
        super().__init__(hass, data_dir)

        self.entry_id = entry_id
        self.error_handler = error_handler

        # Initialize specialized handlers
        self.forecast_handler = DataForecastHandler(hass, data_dir)
        self.ml_handler = DataMLHandler(hass, data_dir)
        self.prediction_handler = DataPredictionHandler(
            hass, data_dir
        )  # OLD - kept for compatibility
        self.state_handler = DataStateHandler(hass, data_dir)
        self.backup_handler = DataBackupHandler(hass, data_dir)
        self.weather_accuracy = WeatherAccuracyTracker(hass, data_dir)

        # NEW: ML-optimized handlers (with parent reference for atomic writes)
        self.hourly_predictions = HourlyPredictionsHandler(data_dir, data_manager=self)
        self.daily_summaries = DailySummariesHandler(data_dir, data_manager=self)

        # Keep file paths for backward compatibility
        self.daily_forecasts_file = self.forecast_handler.daily_forecasts_file
        self.learned_weights_file = self.ml_handler.learned_weights_file
        self.hourly_profile_file = self.ml_handler.hourly_profile_file
        self.model_state_file = self.ml_handler.model_state_file
        self.hourly_samples_file = self.ml_handler.hourly_samples_file
        self.prediction_history_file = self.prediction_handler.prediction_history_file
        self.coordinator_state_file = self.state_handler.coordinator_state_file
        self.weather_cache_file = self.state_handler.weather_cache_file
        self.production_time_state_file = self.state_handler.production_time_state_file

        # Keep adapters
        self.data_adapter = self.ml_handler.data_adapter

        _LOGGER.info("DataManager Facade initialized with specialized handlers")

    async def async_initialize(self) -> None:
        """Async initialization - ensures all files exist (non-blocking)"""
        await self.hourly_predictions.ensure_file_exists()
        await self.daily_summaries.ensure_file_exists()
        _LOGGER.debug("DataManager async initialization complete")

    async def initialize(self) -> bool:
        """Initialize data manager ensure directories exist and create default files"""
        try:
            # Ensure base directory exists
            await self._ensure_directory_exists(self.data_dir)

            # Create subdirectories according to structure
            await self._ensure_directory_exists(self.data_dir / "ml")
            await self._ensure_directory_exists(self.data_dir / "ml" / "models")
            await self._ensure_directory_exists(self.data_dir / "stats")
            await self._ensure_directory_exists(self.data_dir / "stats" / "accuracy_reports")
            await self._ensure_directory_exists(self.data_dir / "data")
            await self._ensure_directory_exists(self.data_dir / "imports")
            await self._ensure_directory_exists(self.data_dir / "exports")
            await self._ensure_directory_exists(self.data_dir / "exports" / "reports")
            await self._ensure_directory_exists(self.data_dir / "exports" / "pictures")
            await self._ensure_directory_exists(self.data_dir / "exports" / "statistics")
            await self._ensure_directory_exists(self.data_dir / "backups")
            await self._ensure_directory_exists(self.data_dir / "backups" / "auto")
            await self._ensure_directory_exists(self.data_dir / "backups" / "manual")
            await self._ensure_directory_exists(self.data_dir / "assets")
            await self._ensure_directory_exists(self.data_dir / "assets" / "images")
            await self._ensure_directory_exists(self.data_dir / "docs")

            # CRITICAL: Auto-migrate from v1 to v2 structure if needed
            # This runs BEFORE any handlers use the files
            _LOGGER.info("Checking for automatic migration from v1 to v2 structure...")
            await self._auto_migrate_v1_to_v2()

            # CRITICAL: Validate and migrate all JSON files FIRST
            # This ensures all files have correct schema before handlers use them
            _LOGGER.info("Starting JSON schema validation and migration")
            validator = DataSchemaValidator(self.hass, self.data_dir)
            validation_success = await validator.validate_and_migrate_all()

            if not validation_success:
                _LOGGER.warning(
                    "JSON schema validation completed with warnings - continuing initialization"
                )
            else:
                _LOGGER.info("JSON schema validation completed successfully")

            # Initialize handlers (they will use the validated files)
            await self.forecast_handler.ensure_daily_forecasts_file()
            await self.ml_handler.ensure_ml_files()
            await self.prediction_handler.ensure_prediction_history_file()
            await self.state_handler.ensure_state_files()

            # Deploy documentation files to docs directory
            await self._deploy_documentation()

            _LOGGER.info("DataManager Facade initialized successfully")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to initialize DataManager: {e}", exc_info=True)
            return False

    #
    # FORECAST METHODS - Delegate to DataForecastHandler
    #

    async def load_daily_forecasts(self) -> Dict[str, Any]:
        """Load daily forecasts"""
        return await self.forecast_handler.load_daily_forecasts()

    async def reset_today_block(self) -> bool:
        """Reset TODAY block at midnight"""
        return await self.forecast_handler.reset_today_block()

    async def save_forecast_day(
        self,
        prediction_kwh: float,
        source: str = "ML",
        lock: bool = True,
        force_overwrite: bool = False,
    ) -> bool:
        """Save todays daily forecast"""
        return await self.forecast_handler.save_forecast_day(
            prediction_kwh, source, lock, force_overwrite
        )

    async def save_forecast_tomorrow(
        self, date: datetime, prediction_kwh: float, source: str = "ML", lock: bool = False
    ) -> bool:
        """Save tomorrows forecast"""
        return await self.forecast_handler.save_forecast_tomorrow(
            date, prediction_kwh, source, lock
        )

    async def save_forecast_day_after(
        self, date: datetime, prediction_kwh: float, source: str = "ML", lock: bool = False
    ) -> bool:
        """Save day after tomorrows forecast"""
        return await self.forecast_handler.save_forecast_day_after(
            date, prediction_kwh, source, lock
        )

    async def save_forecast_best_hour(
        self, hour: int, prediction_kwh: float, source: str = "ML_Hourly"
    ) -> bool:
        """Save best hour forecast"""
        return await self.forecast_handler.save_forecast_best_hour(hour, prediction_kwh, source)

    async def save_actual_best_hour(self, hour: int, actual_kwh: float) -> bool:
        """Save actual best production hour"""
        return await self.forecast_handler.save_actual_best_hour(hour, actual_kwh)

    async def save_forecast_next_hour(
        self,
        hour_start: datetime,
        hour_end: datetime,
        prediction_kwh: float,
        source: str = "ML_Hourly",
    ) -> bool:
        """Save next hour forecast"""
        return await self.forecast_handler.save_forecast_next_hour(
            hour_start, hour_end, prediction_kwh, source
        )

    async def deactivate_next_hour_forecast(self) -> bool:
        """Deactivate next hour forecast"""
        return await self.forecast_handler.deactivate_next_hour_forecast()

    async def update_production_time(
        self,
        active: bool,
        duration_seconds: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        last_power_above_10w: Optional[datetime] = None,
        zero_power_since: Optional[datetime] = None,
    ) -> bool:
        """Update production time tracking"""
        return await self.forecast_handler.update_production_time(
            active, duration_seconds, start_time, end_time, last_power_above_10w, zero_power_since
        )

    async def update_peak_today(self, power_w: float, timestamp: datetime) -> bool:
        """Update todays peak power"""
        return await self.forecast_handler.update_peak_today(power_w, timestamp)

    async def update_all_time_peak(self, power_w: float, timestamp: datetime) -> bool:
        """Update all-time peak power"""
        return await self.forecast_handler.update_all_time_peak(power_w, timestamp)

    async def get_all_time_peak(self) -> Optional[float]:
        """Get all-time peak value"""
        return await self.forecast_handler.get_all_time_peak()

    async def finalize_today(
        self, yield_kwh: float, consumption_kwh: Optional[float] = None, production_seconds: int = 0
    ) -> bool:
        """Finalize today with actual values"""
        # Finalize in daily_forecasts.json
        success = await self.forecast_handler.finalize_today(
            yield_kwh, consumption_kwh, production_seconds
        )

        if success:
            # Calculate accuracy for prediction_history.json
            try:
                # Get today's forecast to calculate accuracy
                current_day = await self.get_current_day_forecast()
                accuracy_percent = None

                if current_day and "forecast_day" in current_day:
                    forecast_kwh = current_day["forecast_day"].get("prediction_kwh")
                    if forecast_kwh and forecast_kwh > 0:
                        error = abs(forecast_kwh - yield_kwh)
                        accuracy = max(0.0, 100.0 - (error / forecast_kwh * 100))
                        accuracy_percent = round(accuracy, 1)

                # Update prediction_history.json with actual values
                await self.update_today_predictions_actual(
                    actual_value=yield_kwh, accuracy=accuracy_percent
                )
                _LOGGER.debug(
                    f"Updated prediction_history.json: actual_value={yield_kwh:.2f} kWh, "
                    f"accuracy={f'{accuracy_percent:.1f}' if accuracy_percent else 'N/A'}%"
                )
            except Exception as e:
                _LOGGER.warning(f"Failed to update prediction_history with actual values: {e}")

        return success

    async def get_history(
        self, days: int = 30, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get history entries"""
        return await self.forecast_handler.get_history(days, start_date, end_date)

    async def rotate_forecasts_at_midnight(self) -> bool:
        """Rotate forecasts at midnight 000030"""
        return await self.forecast_handler.rotate_forecasts_at_midnight()

    async def save_learned_weights(self, weights: LearnedWeights) -> bool:
        """Save learned weights"""
        return await self.ml_handler.save_learned_weights(weights)

    async def load_learned_weights(self) -> Optional[LearnedWeights]:
        """Load learned weights"""
        return await self.ml_handler.load_learned_weights()

    async def get_learned_weights(self) -> Optional[LearnedWeights]:
        """Get learned weights"""
        return await self.ml_handler.get_learned_weights()

    async def delete_learned_weights(self) -> bool:
        """Delete learned weights"""
        return await self.ml_handler.delete_learned_weights()

    async def save_hourly_profile(self, profile: HourlyProfile) -> bool:
        """Save hourly profile"""
        return await self.ml_handler.save_hourly_profile(profile)

    async def load_hourly_profile(self) -> Optional[HourlyProfile]:
        """Load hourly profile"""
        return await self.ml_handler.load_hourly_profile()

    async def get_hourly_profile(self) -> Optional[HourlyProfile]:
        """Get hourly profile"""
        return await self.ml_handler.get_hourly_profile()

    async def save_model_state(self, state: Dict[str, Any]) -> bool:
        """Save model state"""
        return await self.ml_handler.save_model_state(state)

    async def load_model_state(self) -> Dict[str, Any]:
        """Load model state"""
        return await self.ml_handler.load_model_state()

    async def get_model_state(self) -> Dict[str, Any]:
        """Get model state"""
        return await self.ml_handler.get_model_state()

    async def update_model_state(
        self,
        model_loaded: Optional[bool] = None,
        last_training: Optional[str] = None,
        training_samples: Optional[int] = None,
        current_accuracy: Optional[float] = None,
        status: Optional[str] = None,
    ) -> bool:
        """Update model state partially"""
        return await self.ml_handler.update_model_state(
            model_loaded, last_training, training_samples, current_accuracy, status
        )

    async def add_hourly_sample(self, sample: Dict[str, Any]) -> bool:
        """Add hourly sample"""
        return await self.ml_handler.add_hourly_sample(sample)

    async def get_last_collected_hour(self) -> Optional[datetime]:
        """Get last collected hour timestamp"""
        return await self.state_handler.get_last_collected_hour()

    async def set_last_collected_hour(self, timestamp: datetime) -> bool:
        """Set last collected hour timestamp"""
        return await self.state_handler.set_last_collected_hour(timestamp)

    async def get_hourly_samples(
        self,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get hourly samples"""
        return await self.ml_handler.get_hourly_samples(limit, start_date, end_date)

    async def clear_hourly_samples(self) -> bool:
        """Clear hourly samples"""
        return await self.ml_handler.clear_hourly_samples()

    async def get_hourly_samples_count(self) -> int:
        """Get hourly samples count"""
        return await self.ml_handler.get_hourly_samples_count()

    async def get_all_training_records(self, days: int = 60) -> List[Dict[str, Any]]:
        """Get all training records hourly samples for the specified number of days"""
        from datetime import timedelta

        from ..core.core_helpers import SafeDateTimeUtil as dt_util

        # Calculate start date (days ago from now)
        now = dt_util.now()
        start_date = (now - timedelta(days=days)).isoformat()

        # Get hourly samples from the specified date range
        return await self.ml_handler.get_hourly_samples(
            limit=None, start_date=start_date, end_date=None  # No limit, get all  # Up to now
        )

    async def cleanup_duplicate_samples(self) -> Dict[str, int]:
        """Cleanup duplicate samples"""
        return await self.ml_handler.cleanup_duplicate_samples()

    async def cleanup_zero_production_samples(self) -> Dict[str, int]:
        """Cleanup zero-production samples"""
        return await self.ml_handler.cleanup_zero_production_samples()

    # Note: prediction_history.json methods removed - use hourly_predictions instead

    async def update_today_predictions_actual(
        self, actual_value: float, accuracy: Optional[float] = None
    ) -> bool:
        """Update todays predictions with actual value and accuracy"""
        return await self.prediction_handler.update_today_predictions_actual(actual_value, accuracy)

    async def save_expected_daily_production(self, value: float) -> bool:
        """Save expected daily production"""
        return await self.state_handler.save_expected_daily_production(value)

    async def load_expected_daily_production(self) -> Optional[float]:
        """Load expected daily production"""
        # Load daily forecasts to check for new system first
        daily_forecasts = await self.load_daily_forecasts()
        return await self.state_handler.load_expected_daily_production(
            check_daily_forecasts=True, daily_forecasts_data=daily_forecasts
        )

    async def clear_expected_daily_production(self) -> bool:
        """Clear expected daily production"""
        return await self.state_handler.clear_expected_daily_production()

    async def save_weather_cache(self, weather_data: Dict[str, Any]) -> bool:
        """Save weather cache"""
        return await self.state_handler.save_weather_cache(weather_data)

    async def load_weather_cache(self) -> Optional[Dict[str, Any]]:
        """Load weather cache"""
        return await self.state_handler.load_weather_cache()

    async def clear_weather_cache(self) -> bool:
        """Clear weather cache"""
        return await self.state_handler.clear_weather_cache()

    async def get_weather_cache_age(self) -> Optional[int]:
        """Get weather cache age in minutes"""
        return await self.state_handler.get_weather_cache_age()

    async def is_weather_cache_valid(self, max_age_minutes: int = 180) -> bool:
        """Check if weather cache is valid Default: 180 minutes (3 hours) for better resilience"""
        return await self.state_handler.is_weather_cache_valid(max_age_minutes)

    async def create_backup(
        self, backup_name: Optional[str] = None, backup_type: str = "manual"
    ) -> bool:
        """Create backup"""
        return await self.backup_handler.create_backup(backup_name, backup_type)

    async def cleanup_old_backups(
        self, backup_type: str = "auto", retention_days: Optional[int] = None
    ) -> int:
        """Cleanup old backups"""
        return await self.backup_handler.cleanup_old_backups(backup_type, retention_days)

    async def cleanup_excess_backups(
        self, backup_type: str = "auto", max_backups: Optional[int] = None
    ) -> int:
        """Cleanup excess backups"""
        return await self.backup_handler.cleanup_excess_backups(backup_type, max_backups)

    async def list_backups(self, backup_type: Optional[str] = None) -> list:
        """List backups"""
        return await self.backup_handler.list_backups(backup_type)

    async def restore_backup(self, backup_name: str, backup_type: str = "manual") -> bool:
        """Restore backup"""
        return await self.backup_handler.restore_backup(backup_name, backup_type)

    async def delete_backup(self, backup_name: str, backup_type: str = "manual") -> bool:
        """Delete backup"""
        return await self.backup_handler.delete_backup(backup_name, backup_type)

    async def get_backup_info(
        self, backup_name: str, backup_type: str = "manual"
    ) -> Optional[dict]:
        """Get backup info"""
        return await self.backup_handler.get_backup_info(backup_name, backup_type)

    async def save_daily_forecast(
        self, prediction_kwh: float, source: str = "auto_6am", force_overwrite: bool = False
    ) -> bool:
        """OLD METHOD - redirects to save_forecast_day"""
        return await self.save_forecast_day(prediction_kwh, source, force_overwrite=force_overwrite)

    async def get_current_day_forecast(self) -> Optional[Dict[str, Any]]:
        """OLD METHOD - returns today block"""
        try:
            data = await self.load_daily_forecasts()
            return data.get("today")
        except Exception:
            return None

    async def save_forecast_today(self, prediction_kwh: float, source: str = "ML") -> bool:
        """OLD METHOD - redirects to save_forecast_day"""
        return await self.save_forecast_day(prediction_kwh, source)

    async def save_production_tracking(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        duration_seconds: int = 0,
        duration_formatted: str = "00:00:00",
        currently_producing: bool = False,
        last_power_above_10w: Optional[datetime] = None,
        zero_power_streak_minutes: int = 0,
    ) -> bool:
        """OLD METHOD - redirects to update_production_time"""
        zero_power_since = None
        if zero_power_streak_minutes > 0 and last_power_above_10w:
            zero_power_since = last_power_above_10w + timedelta(minutes=zero_power_streak_minutes)

        return await self.update_production_time(
            active=currently_producing,
            duration_seconds=duration_seconds,
            start_time=start_time,
            end_time=end_time,
            last_power_above_10w=last_power_above_10w,
            zero_power_since=zero_power_since,
        )

    async def move_to_history(self) -> bool:
        """Move finalized today data to history"""
        return await self.forecast_handler.move_to_history()

    async def calculate_statistics(self) -> bool:
        """Calculate aggregated statistics"""
        return await self.forecast_handler.calculate_statistics()

    async def save_power_peak(
        self, power_w: float, timestamp: datetime, is_all_time: bool = False
    ) -> bool:
        """OLD METHOD - redirects to update_peak_today"""
        return await self.update_peak_today(power_w, timestamp)

    async def finalize_current_day(
        self,
        actual_yield_kwh: float,
        actual_consumption_kwh: Optional[float] = None,
        production_time_today: Optional[str] = None,
    ) -> bool:
        """OLD METHOD - redirects to finalize_today"""
        production_seconds = 0
        if production_time_today:
            try:
                parts = production_time_today.replace("h", "").replace("m", "").split()
                if len(parts) >= 2:
                    production_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60
            except Exception as e:
                _LOGGER.debug(f"Could not parse production time '{production_time_today}': {e}")

        return await self.finalize_today(
            actual_yield_kwh, actual_consumption_kwh, production_seconds
        )

    def _deploy_documentation_sync(self, integration_dir, target_dir, marker_file) -> int:
        """
        Synchronous helper to deploy documentation files

        v8.6.0: Deploy documentation to docs/ instead of import_tools to imports/
        - Deploys README_CRITICAL.md from integration root
        - Deploys PDF handbook from import_tools/ (if exists)
        - Creates user-friendly docs structure
        """
        import shutil

        files_copied = 0

        # Check if already deployed for this version (blocking)
        if marker_file.exists():
            return -1  # Signal "already deployed"

        try:
            # Source files to copy
            files_to_deploy = [
                (integration_dir / "README_CRITICAL.md", "README.md"),  # Main documentation
            ]

            # Check if import_tools PDF exists (transitional period)
            import_tools_pdf = (
                integration_dir / "import_tools" / "Solar Forecast ML Integration Handbuch.pdf"
            )
            if import_tools_pdf.exists():
                files_to_deploy.append((import_tools_pdf, "Handbuch.pdf"))

            # Copy files (blocking)
            for source, dest_name in files_to_deploy:
                if source.exists():
                    try:
                        dest_path = target_dir / dest_name
                        shutil.copy2(str(source), str(dest_path))
                        files_copied += 1
                    except Exception as e:
                        _LOGGER.warning(f"Failed to copy {source.name}: {e}")

            # Create marker file (blocking)
            marker_file.write_text("Documentation deployed for v8.6.0")

        except Exception as e:
            _LOGGER.warning(f"Error during documentation deployment: {e}")

        return files_copied

    async def _deploy_documentation(self) -> None:
        """
        Deploy documentation files to docs directory

        v8.6.0 CHANGE:
        - Replaces old import_tools deployment
        - Deploys to /config/solar_forecast_ml/docs/
        - Provides user-friendly access to documentation
        """
        try:
            from pathlib import Path

            # Source: custom_components/solar_forecast_ml/
            # Target: /config/solar_forecast_ml/docs/
            # Get the integration's base directory
            integration_dir = Path(__file__).parent.parent
            target_dir = self.data_dir / "docs"
            marker_file = target_dir / ".docs_deployed_v8.6"

            # Fast check: If already deployed for this version, skip
            if marker_file.exists():
                _LOGGER.debug("Documentation already deployed for v8.6, skipping")
                return

            # Spawn executor job for file operations
            files_copied = await self.hass.async_add_executor_job(
                self._deploy_documentation_sync, integration_dir, target_dir, marker_file
            )

            if files_copied > 0:
                _LOGGER.info(
                    f"✓ Documentation deployed: {files_copied} files to "
                    f"{target_dir.relative_to(self.data_dir.parent)}"
                )
            elif files_copied == -1:
                _LOGGER.debug("Documentation already deployed")

        except Exception as e:
            _LOGGER.warning(f"Failed to deploy documentation: {e}")
            # Non-critical error, continue initialization

    async def _auto_migrate_v1_to_v2(self) -> bool:
        """
        Automatically migrate from v1 (prediction_history.json) to v2 (hourly_predictions.json + daily_summaries.json)

        This runs silently on every startup. If old file exists, it migrates.
        If migration fails, it just creates clean new files.
        User doesn't need to do anything.
        """
        try:
            from .data_migration import DataMigration

            old_file = self.data_dir / "stats" / "prediction_history.json"

            # Check if old file exists
            if not old_file.exists():
                _LOGGER.debug("No old prediction_history.json found - skipping migration")
                return True

            # Check if we already migrated (new files exist with data)
            new_hourly = self.data_dir / "stats" / "hourly_predictions.json"
            new_daily = self.data_dir / "stats" / "daily_summaries.json"

            if new_hourly.exists() and new_daily.exists():
                _LOGGER.debug("New v2 structure already exists - skipping migration")
                return True

            # Migrate!
            _LOGGER.info("🔄 Auto-migrating from v1 to v2 structure...")
            migration = DataMigration(self.data_dir)

            # Run migration (not dry run)
            report = await migration.migrate(dry_run=False)

            if report["success"]:
                _LOGGER.info(
                    f"✅ Auto-migration successful! Converted {len(report['converted_dates'])} days of historical data"
                )
                _LOGGER.info(f"   Old file archived as: prediction_history_v1_archived.json")
                return True
            else:
                _LOGGER.warning("⚠️  Auto-migration had errors - creating clean new files instead")
                # Migration failed - that's OK, just continue with clean files
                return True

        except Exception as e:
            _LOGGER.warning(f"Auto-migration failed: {e} - continuing with clean files")
            # Migration error is non-critical - new files will be created automatically
            return True
