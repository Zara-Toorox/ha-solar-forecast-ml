"""
Data Manager for Solar Forecast ML Integration - FACADE

Delegates to specialized handlers for better organization.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

from homeassistant.core import HomeAssistant

from .data_io import DataManagerIO
from .data_forecast_handler import DataForecastHandler
from .data_ml_handler import DataMLHandler
from .data_prediction_handler import DataPredictionHandler
from .data_state_handler import DataStateHandler
from .data_backup_handler import DataBackupHandler

from ..ml.ml_types import LearnedWeights, HourlyProfile

_LOGGER = logging.getLogger(__name__)


class DataManager(DataManagerIO):
    """
    Data Manager API for Solar Forecast ML.
    Facade that delegates to specialized handlers.
    """

    def __init__(self, hass: HomeAssistant, entry_id: str, data_dir: Path, error_handler=None):
        """Initialize the Data Manager Facade."""
        super().__init__(hass, data_dir)

        self.entry_id = entry_id
        self.error_handler = error_handler

        # Initialize specialized handlers
        self.forecast_handler = DataForecastHandler(hass, data_dir)
        self.ml_handler = DataMLHandler(hass, data_dir)
        self.prediction_handler = DataPredictionHandler(hass, data_dir)
        self.state_handler = DataStateHandler(hass, data_dir)
        self.backup_handler = DataBackupHandler(hass, data_dir)

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

    async def initialize(self) -> bool:
        """Initialize data manager: ensure directories exist and create default files."""
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

            # Initialize handlers
            await self.forecast_handler.ensure_daily_forecasts_file()
            await self.ml_handler.ensure_ml_files()
            await self.prediction_handler.ensure_prediction_history_file()
            await self.state_handler.ensure_state_files()

            # Deploy import tools for beta testers
            await self._deploy_import_tools()

            _LOGGER.info("DataManager Facade initialized successfully")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to initialize DataManager: {e}", exc_info=True)
            return False

    # 
    # FORECAST METHODS - Delegate to DataForecastHandler
    #

    async def load_daily_forecasts(self) -> Dict[str, Any]:
        """Load daily forecasts."""
        return await self.forecast_handler.load_daily_forecasts()

    async def reset_today_block(self) -> bool:
        """Reset TODAY block at midnight."""
        return await self.forecast_handler.reset_today_block()

    async def save_forecast_day(
        self,
        prediction_kwh: float,
        source: str = "ML",
        lock: bool = True
    ) -> bool:
        """Save today's daily forecast."""
        return await self.forecast_handler.save_forecast_day(
            prediction_kwh, source, lock
        )

    async def save_forecast_tomorrow(
        self,
        date: datetime,
        prediction_kwh: float,
        source: str = "ML",
        lock: bool = False
    ) -> bool:
        """Save tomorrow's forecast."""
        return await self.forecast_handler.save_forecast_tomorrow(
            date, prediction_kwh, source, lock
        )

    async def save_forecast_day_after(
        self,
        date: datetime,
        prediction_kwh: float,
        source: str = "ML",
        lock: bool = False
    ) -> bool:
        """Save day after tomorrow's forecast."""
        return await self.forecast_handler.save_forecast_day_after(
            date, prediction_kwh, source, lock
        )

    async def save_forecast_best_hour(
        self,
        hour: int,
        prediction_kwh: float,
        source: str = "ML_Hourly"
    ) -> bool:
        """Save best hour forecast."""
        return await self.forecast_handler.save_forecast_best_hour(
            hour, prediction_kwh, source
        )

    async def save_forecast_next_hour(
        self,
        hour_start: datetime,
        hour_end: datetime,
        prediction_kwh: float,
        source: str = "ML_Hourly"
    ) -> bool:
        """Save next hour forecast."""
        return await self.forecast_handler.save_forecast_next_hour(
            hour_start, hour_end, prediction_kwh, source
        )

    async def deactivate_next_hour_forecast(self) -> bool:
        """Deactivate next hour forecast."""
        return await self.forecast_handler.deactivate_next_hour_forecast()

    async def update_production_time(
        self,
        active: bool,
        duration_seconds: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        last_power_above_10w: Optional[datetime] = None,
        zero_power_since: Optional[datetime] = None
    ) -> bool:
        """Update production time tracking."""
        return await self.forecast_handler.update_production_time(
            active, duration_seconds, start_time, end_time,
            last_power_above_10w, zero_power_since
        )

    async def update_peak_today(
        self,
        power_w: float,
        timestamp: datetime
    ) -> bool:
        """Update today's peak power."""
        return await self.forecast_handler.update_peak_today(power_w, timestamp)

    async def update_all_time_peak(
        self,
        power_w: float,
        timestamp: datetime
    ) -> bool:
        """Update all-time peak power."""
        return await self.forecast_handler.update_all_time_peak(power_w, timestamp)

    async def get_all_time_peak(self) -> Optional[float]:
        """Get all-time peak value."""
        return await self.forecast_handler.get_all_time_peak()

    async def finalize_today(
        self,
        yield_kwh: float,
        consumption_kwh: Optional[float] = None,
        production_seconds: int = 0
    ) -> bool:
        """Finalize today with actual values."""
        return await self.forecast_handler.finalize_today(
            yield_kwh, consumption_kwh, production_seconds
        )

    async def get_history(
        self,
        days: int = 30,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get history entries."""
        return await self.forecast_handler.get_history(days, start_date, end_date)

    async def rotate_forecasts_at_midnight(self) -> bool:
        """Rotate forecasts at midnight (00:00:30)."""
        return await self.forecast_handler.rotate_forecasts_at_midnight()

    async def save_learned_weights(self, weights: LearnedWeights) -> bool:
        """Save learned weights."""
        return await self.ml_handler.save_learned_weights(weights)

    async def load_learned_weights(self) -> Optional[LearnedWeights]:
        """Load learned weights."""
        return await self.ml_handler.load_learned_weights()

    async def get_learned_weights(self) -> Optional[LearnedWeights]:
        """Get learned weights."""
        return await self.ml_handler.get_learned_weights()

    async def save_hourly_profile(self, profile: HourlyProfile) -> bool:
        """Save hourly profile."""
        return await self.ml_handler.save_hourly_profile(profile)

    async def load_hourly_profile(self) -> Optional[HourlyProfile]:
        """Load hourly profile."""
        return await self.ml_handler.load_hourly_profile()

    async def get_hourly_profile(self) -> Optional[HourlyProfile]:
        """Get hourly profile."""
        return await self.ml_handler.get_hourly_profile()

    async def save_model_state(self, state: Dict[str, Any]) -> bool:
        """Save model state."""
        return await self.ml_handler.save_model_state(state)

    async def load_model_state(self) -> Dict[str, Any]:
        """Load model state."""
        return await self.ml_handler.load_model_state()

    async def get_model_state(self) -> Dict[str, Any]:
        """Get model state."""
        return await self.ml_handler.get_model_state()

    async def update_model_state(
        self,
        model_loaded: Optional[bool] = None,
        last_training: Optional[str] = None,
        training_samples: Optional[int] = None,
        current_accuracy: Optional[float] = None,
        status: Optional[str] = None
    ) -> bool:
        """Update model state partially."""
        return await self.ml_handler.update_model_state(
            model_loaded, last_training, training_samples,
            current_accuracy, status
        )

    async def add_hourly_sample(self, sample: Dict[str, Any]) -> bool:
        """Add hourly sample."""
        return await self.ml_handler.add_hourly_sample(sample)

    async def get_last_collected_hour(self) -> Optional[datetime]:
        """Get last collected hour timestamp."""
        return await self.state_handler.get_last_collected_hour()

    async def set_last_collected_hour(self, timestamp: datetime) -> bool:
        """Set last collected hour timestamp."""
        return await self.state_handler.set_last_collected_hour(timestamp)

    async def get_hourly_samples(
        self,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get hourly samples."""
        return await self.ml_handler.get_hourly_samples(limit, start_date, end_date)

    async def clear_hourly_samples(self) -> bool:
        """Clear hourly samples."""
        return await self.ml_handler.clear_hourly_samples()

    async def get_hourly_samples_count(self) -> int:
        """Get hourly samples count."""
        return await self.ml_handler.get_hourly_samples_count()

    async def cleanup_duplicate_samples(self) -> Dict[str, int]:
        """Cleanup duplicate samples."""
        return await self.ml_handler.cleanup_duplicate_samples()

    async def cleanup_zero_production_samples(self) -> Dict[str, int]:
        """Cleanup zero-production samples."""
        return await self.ml_handler.cleanup_zero_production_samples()

 
    async def save_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """Save prediction to history."""
        return await self.prediction_handler.save_prediction(prediction_data)

    async def get_predictions(
        self,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get predictions."""
        return await self.prediction_handler.get_predictions(
            limit, start_date, end_date
        )

    async def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Get latest prediction."""
        return await self.prediction_handler.get_latest_prediction()

    async def get_prediction_for_date(self, date: str) -> Optional[Dict[str, Any]]:
        """Get prediction for date."""
        return await self.prediction_handler.get_prediction_for_date(date)

    async def cleanup_old_predictions(self, days: int = 365) -> bool:
        """Cleanup old predictions."""
        return await self.prediction_handler.cleanup_old_predictions(days)

    async def calculate_accuracy_stats(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Calculate accuracy statistics."""
        return await self.prediction_handler.calculate_accuracy_stats(days)

    async def get_accuracy_trend(
        self,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get accuracy trend."""
        return await self.prediction_handler.get_accuracy_trend(days)

    async def get_predictions_count(self) -> int:
        """Get predictions count."""
        return await self.prediction_handler.get_predictions_count()


    async def save_expected_daily_production(self, value: float) -> bool:
        """Save expected daily production."""
        return await self.state_handler.save_expected_daily_production(value)

    async def load_expected_daily_production(self) -> Optional[float]:
        """Load expected daily production."""
        # Load daily forecasts to check for new system first
        daily_forecasts = await self.load_daily_forecasts()
        return await self.state_handler.load_expected_daily_production(
            check_daily_forecasts=True,
            daily_forecasts_data=daily_forecasts
        )

    async def clear_expected_daily_production(self) -> bool:
        """Clear expected daily production."""
        return await self.state_handler.clear_expected_daily_production()

    async def save_weather_cache(self, weather_data: Dict[str, Any]) -> bool:
        """Save weather cache."""
        return await self.state_handler.save_weather_cache(weather_data)

    async def load_weather_cache(self) -> Optional[Dict[str, Any]]:
        """Load weather cache."""
        return await self.state_handler.load_weather_cache()

    async def clear_weather_cache(self) -> bool:
        """Clear weather cache."""
        return await self.state_handler.clear_weather_cache()

    async def get_weather_cache_age(self) -> Optional[int]:
        """Get weather cache age in minutes."""
        return await self.state_handler.get_weather_cache_age()

    async def is_weather_cache_valid(self, max_age_minutes: int = 60) -> bool:
        """Check if weather cache is valid."""
        return await self.state_handler.is_weather_cache_valid(max_age_minutes)


    async def create_backup(
        self,
        backup_name: Optional[str] = None,
        backup_type: str = "manual"
    ) -> bool:
        """Create backup."""
        return await self.backup_handler.create_backup(backup_name, backup_type)

    async def cleanup_old_backups(
        self,
        backup_type: str = "auto",
        retention_days: Optional[int] = None
    ) -> int:
        """Cleanup old backups."""
        return await self.backup_handler.cleanup_old_backups(
            backup_type, retention_days
        )

    async def cleanup_excess_backups(
        self,
        backup_type: str = "auto",
        max_backups: Optional[int] = None
    ) -> int:
        """Cleanup excess backups."""
        return await self.backup_handler.cleanup_excess_backups(
            backup_type, max_backups
        )

    async def list_backups(
        self,
        backup_type: Optional[str] = None
    ) -> list:
        """List backups."""
        return await self.backup_handler.list_backups(backup_type)

    async def restore_backup(
        self,
        backup_name: str,
        backup_type: str = "manual"
    ) -> bool:
        """Restore backup."""
        return await self.backup_handler.restore_backup(backup_name, backup_type)

    async def delete_backup(
        self,
        backup_name: str,
        backup_type: str = "manual"
    ) -> bool:
        """Delete backup."""
        return await self.backup_handler.delete_backup(backup_name, backup_type)

    async def get_backup_info(
        self,
        backup_name: str,
        backup_type: str = "manual"
    ) -> Optional[dict]:
        """Get backup info."""
        return await self.backup_handler.get_backup_info(backup_name, backup_type)

    async def save_daily_forecast(self, prediction_kwh: float, source: str = "auto_6am") -> bool:
        """OLD METHOD - redirects to save_forecast_day()."""
        return await self.save_forecast_day(prediction_kwh, source)

    async def get_current_day_forecast(self) -> Optional[Dict[str, Any]]:
        """OLD METHOD - returns today block."""
        try:
            data = await self.load_daily_forecasts()
            return data.get("today")
        except Exception:
            return None

    async def save_forecast_today(self, prediction_kwh: float, source: str = "ML") -> bool:
        """OLD METHOD - redirects to save_forecast_day()."""
        return await self.save_forecast_day(prediction_kwh, source)

    async def save_production_tracking(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        duration_seconds: int = 0,
        duration_formatted: str = "00:00:00",
        currently_producing: bool = False,
        last_power_above_10w: Optional[datetime] = None,
        zero_power_streak_minutes: int = 0
    ) -> bool:
        """OLD METHOD - redirects to update_production_time()."""
        zero_power_since = None
        if zero_power_streak_minutes > 0 and last_power_above_10w:
            zero_power_since = last_power_above_10w + timedelta(minutes=zero_power_streak_minutes)
        
        return await self.update_production_time(
            active=currently_producing,
            duration_seconds=duration_seconds,
            start_time=start_time,
            end_time=end_time,
            last_power_above_10w=last_power_above_10w,
            zero_power_since=zero_power_since
        )

    async def move_to_history(self) -> bool:
        """Move finalized today data to history."""
        try:
            _LOGGER.info("Moving data to history (stub implementation)")
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to move to history: {e}")
            return False

    async def calculate_statistics(self) -> bool:
        """Calculate aggregated statistics."""
        try:
            _LOGGER.info("Calculating statistics (stub implementation)")
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to calculate statistics: {e}")
            return False

    async def save_power_peak(self, power_w: float, timestamp: datetime, is_all_time: bool = False) -> bool:
        """OLD METHOD - redirects to update_peak_today()."""
        return await self.update_peak_today(power_w, timestamp)

    async def finalize_current_day(
        self,
        actual_yield_kwh: float,
        actual_consumption_kwh: Optional[float] = None,
        production_time_today: Optional[str] = None
    ) -> bool:
        """OLD METHOD - redirects to finalize_today()."""
        production_seconds = 0
        if production_time_today:
            try:
                parts = production_time_today.replace("h", "").replace("m", "").split()
                if len(parts) >= 2:
                    production_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60
            except:
                pass
        
        return await self.finalize_today(actual_yield_kwh, actual_consumption_kwh, production_seconds)

    def _deploy_import_tools_sync(self, source_dir, target_dir, marker_file) -> int:
        """
        Synchronous helper to deploy import tools.
        This is a BLOCKING function to be run in an executor.

        Returns:
            Number of files copied, or -1 if already deployed, or 0 if no source
        """
        import shutil

        files_copied = 0

        # Check if source exists (blocking)
        if not source_dir.exists():
            return 0

        # Check if already deployed (blocking)
        if marker_file.exists():
            return -1  # Signal "already deployed"

        # Copy all files (blocking)
        for source_file in source_dir.iterdir():
            if source_file.is_file():
                target_file = target_dir / source_file.name
                shutil.copy2(str(source_file), str(target_file))
                files_copied += 1

        # Create marker (blocking)
        marker_file.write_text("Import tools deployed successfully")

        return files_copied

    async def _deploy_import_tools(self) -> None:
        """
        Deploy import_tools to the imports directory for beta testers.
        This makes it easy for users to access historical data import scripts.
        """
        try:
            from pathlib import Path

            # Source: custom_components/solar_forecast_ml/import_tools/
            # Target: /config/solar_forecast_ml/imports/

            # Get the integration's base directory
            integration_dir = Path(__file__).parent.parent
            source_dir = integration_dir / "import_tools"
            target_dir = self.data_dir / "imports"
            marker_file = target_dir / ".import_tools_deployed"

            # Run ALL blocking operations in executor
            files_copied = await self.hass.async_add_executor_job(
                self._deploy_import_tools_sync, source_dir, target_dir, marker_file
            )

            if files_copied == 0:
                _LOGGER.debug("No import_tools directory found, skipping deployment")
            elif files_copied == -1:
                _LOGGER.debug("Import tools already deployed, skipping")
            else:
                _LOGGER.info(
                    f"[OK] Import tools deployed: {files_copied} files copied to "
                    f"{target_dir.relative_to(self.data_dir.parent)}"
                )

        except Exception as e:
            _LOGGER.warning(f"Failed to deploy import tools: {e}")
            # Non-critical error, continue initialization
