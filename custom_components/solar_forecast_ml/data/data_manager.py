# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from ..ai import HourlyProfile, LearnedWeights
from .data_backup_handler import DataBackupHandler
from .data_daily_summaries import DailySummariesHandler
from .data_forecast_handler import DataForecastHandler
from .data_hourly_predictions import HourlyPredictionsHandler
from .data_io import DataManagerIO
from .data_ml_handler import DataMLHandler
from .data_schema_validator import DataSchemaValidator
from .data_state_handler import DataStateHandler

_LOGGER = logging.getLogger(__name__)

class DataManager(DataManagerIO):
    """Data Manager API for Solar Forecast ML"""

    def __init__(self, hass: HomeAssistant, entry_id: str, data_dir: Path, error_handler=None):
        """Initialize the Data Manager Facade @zara"""
        super().__init__(hass, data_dir)

        self.entry_id = entry_id
        self.error_handler = error_handler

        self.forecast_handler = DataForecastHandler(hass, data_dir)
        self.ml_handler = DataMLHandler(hass, data_dir, data_manager=self)
        self.state_handler = DataStateHandler(hass, data_dir)
        self.backup_handler = DataBackupHandler(hass, data_dir)

        self.hourly_predictions = HourlyPredictionsHandler(data_dir, data_manager=self)
        self.daily_summaries = DailySummariesHandler(data_dir, data_manager=self)

        self.daily_forecasts_file = self.forecast_handler.daily_forecasts_file
        self.learned_weights_file = self.ml_handler.learned_weights_file
        self.hourly_profile_file = self.ml_handler.hourly_profile_file
        self.model_state_file = self.ml_handler.model_state_file

        self.coordinator_state_file = self.state_handler.coordinator_state_file
        self.production_time_state_file = self.state_handler.production_time_state_file

        self.hourly_predictions_file = self.data_dir / "stats" / "hourly_predictions.json"
        self.astronomy_cache_file = self.data_dir / "stats" / "astronomy_cache.json"
        self.weather_precision_file = self.data_dir / "stats" / "weather_precision_daily.json"
        self.weather_corrected_file = self.data_dir / "stats" / "weather_forecast_corrected.json"
        self.daily_summaries_file = self.data_dir / "stats" / "daily_summaries.json"

        self.data_adapter = self.ml_handler.data_adapter

        _LOGGER.info("DataManager Facade initialized with specialized handlers")

    async def async_initialize(self) -> None:
        """Async initialization - Template-Based @zara"""
        _LOGGER.debug("DataManager async initialization complete (template-based, no action needed)")

    async def initialize(self) -> bool:
        """Initialize data manager - Schema-First Installation @zara"""
        try:
            _LOGGER.info("DataManager: Running schema validation (creates missing files)...")

            validator = DataSchemaValidator(self.hass, self.data_dir)
            validation_success = await validator.validate_and_migrate_all()

            if not validation_success:
                _LOGGER.warning(
                    "JSON schema validation completed with warnings - continuing initialization"
                )
            else:
                _LOGGER.info("JSON schema validation completed successfully ✓")

            required_files = [
                self.daily_forecasts_file,
                self.hourly_predictions_file,
                self.daily_summaries_file,
                self.model_state_file,
                self.hourly_profile_file,
                self.coordinator_state_file,
                self.production_time_state_file,
            ]

            missing = [f.name for f in required_files if not f.exists()]
            if missing:
                _LOGGER.error(
                    f"Critical files still missing after schema validation: {missing}\n"
                    f"This indicates a bug in the schema validator.\n"
                    f"Expected location: {self.data_dir}"
                )
                return False

            _LOGGER.info(f"DataManager: All {len(required_files)} required files present ✓")
            _LOGGER.info("DataManager initialized successfully")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to initialize DataManager: {e}", exc_info=True)
            return False

    async def load_daily_forecasts(self) -> Dict[str, Any]:
        """Load daily forecasts @zara"""
        return await self.forecast_handler.load_daily_forecasts()

    async def reset_today_block(self) -> bool:
        """Reset TODAY block at midnight @zara"""
        return await self.forecast_handler.reset_today_block()

    async def save_forecast_day(
        self,
        prediction_kwh: float,
        source: str = "ML",
        lock: bool = True,
        force_overwrite: bool = False,
        prediction_kwh_raw: float = None,
        safeguard_applied: bool = False,
    ) -> bool:
        """Save todays daily forecast"""
        return await self.forecast_handler.save_forecast_day(
            prediction_kwh, source, lock, force_overwrite, prediction_kwh_raw, safeguard_applied
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

    async def save_multi_day_hourly_forecast(
        self, hourly_forecast: list
    ) -> bool:
        """Save multi-day hourly forecast to JSON @zara"""
        return await self.forecast_handler.save_multi_day_hourly_forecast(hourly_forecast)

    async def save_actual_best_hour(self, hour: int, actual_kwh: float) -> bool:
        """Save actual best production hour @zara"""
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
        """Deactivate next hour forecast @zara"""
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
        """Update todays peak power @zara"""
        return await self.forecast_handler.update_peak_today(power_w, timestamp)

    async def update_all_time_peak(self, power_w: float, timestamp: datetime) -> bool:
        """Update all-time peak power @zara"""
        return await self.forecast_handler.update_all_time_peak(power_w, timestamp)

    async def save_power_peak(
        self, power_w: float, timestamp: datetime, is_all_time: bool = False
    ) -> bool:
        """Save power peak - updates today and optionally all-time @zara"""
        success = await self.update_peak_today(power_w, timestamp)
        if is_all_time:
            success = await self.update_all_time_peak(power_w, timestamp) and success
        return success

    async def get_all_time_peak(self) -> Optional[float]:
        """Get all-time peak value @zara"""
        return await self.forecast_handler.get_all_time_peak()

    async def finalize_today(
        self,
        yield_kwh: float,
        consumption_kwh: Optional[float] = None,
        production_seconds: int = 0,
        excluded_hours_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Finalize today with actual values"""

        return await self.forecast_handler.finalize_today(
            yield_kwh, consumption_kwh, production_seconds, excluded_hours_info
        )

    async def get_history(
        self, days: int = 30, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get history entries"""
        return await self.forecast_handler.get_history(days, start_date, end_date)

    async def rotate_forecasts_at_midnight(self) -> bool:
        """Rotate forecasts at midnight 000030 @zara"""
        return await self.forecast_handler.rotate_forecasts_at_midnight()

    async def save_learned_weights(self, weights: LearnedWeights) -> bool:
        """Save learned weights @zara"""
        return await self.ml_handler.save_learned_weights(weights)

    async def load_learned_weights(self) -> Optional[LearnedWeights]:
        """Load learned weights @zara"""
        return await self.ml_handler.load_learned_weights()

    async def get_learned_weights(self) -> Optional[LearnedWeights]:
        """Get learned weights @zara"""
        return await self.ml_handler.get_learned_weights()

    async def delete_learned_weights(self) -> bool:
        """Delete learned weights @zara"""
        return await self.ml_handler.delete_learned_weights()

    async def save_hourly_profile(self, profile: HourlyProfile) -> bool:
        """Save hourly profile @zara"""
        return await self.ml_handler.save_hourly_profile(profile)

    async def load_hourly_profile(self) -> Optional[HourlyProfile]:
        """Load hourly profile @zara"""
        return await self.ml_handler.load_hourly_profile()

    async def get_hourly_profile(self) -> Optional[HourlyProfile]:
        """Get hourly profile @zara"""
        return await self.ml_handler.get_hourly_profile()

    async def save_model_state(self, state: Dict[str, Any]) -> bool:
        """Save model state @zara"""
        return await self.ml_handler.save_model_state(state)

    async def load_model_state(self) -> Dict[str, Any]:
        """Load model state @zara"""
        return await self.ml_handler.load_model_state()

    async def get_model_state(self) -> Dict[str, Any]:
        """Get model state @zara"""
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

    async def get_last_collected_hour(self) -> Optional[datetime]:
        """Get last collected hour timestamp @zara"""
        return await self.state_handler.get_last_collected_hour()

    async def set_last_collected_hour(self, timestamp: datetime) -> bool:
        """Set last collected hour timestamp @zara"""
        return await self.state_handler.set_last_collected_hour(timestamp)

    async def save_expected_daily_production(self, value: float) -> bool:
        """Save expected daily production @zara"""
        return await self.state_handler.save_expected_daily_production(value)

    async def load_expected_daily_production(self) -> Optional[float]:
        """Load expected daily production @zara"""

        daily_forecasts = await self.load_daily_forecasts()
        return await self.state_handler.load_expected_daily_production(
            check_daily_forecasts=True, daily_forecasts_data=daily_forecasts
        )

    async def clear_expected_daily_production(self) -> bool:
        """Clear expected daily production @zara"""
        return await self.state_handler.clear_expected_daily_production()

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
        """List backups @zara"""
        return await self.backup_handler.list_backups(backup_type)

    async def restore_backup(self, backup_name: str, backup_type: str = "manual") -> bool:
        """Restore backup @zara"""
        return await self.backup_handler.restore_backup(backup_name, backup_type)

    async def delete_backup(self, backup_name: str, backup_type: str = "manual") -> bool:
        """Delete backup @zara"""
        return await self.backup_handler.delete_backup(backup_name, backup_type)

    async def get_backup_info(
        self, backup_name: str, backup_type: str = "manual"
    ) -> Optional[dict]:
        """Get backup info"""
        return await self.backup_handler.get_backup_info(backup_name, backup_type)

    async def get_current_day_forecast(self) -> Optional[Dict[str, Any]]:
        """Get today's forecast block from daily_forecasts.json @zara"""
        try:
            data = await self.load_daily_forecasts()
            return data.get("today")
        except Exception:
            return None

    async def save_daily_forecast(
        self, prediction_kwh: float, source: str = "auto_6am", force_overwrite: bool = False
    ) -> bool:
        """Save daily forecast - redirects to save_forecast_day @zara"""
        return await self.save_forecast_day(prediction_kwh, source, force_overwrite=force_overwrite)

    async def move_to_history(self) -> bool:
        """Move finalized today data to history @zara"""
        return await self.forecast_handler.move_to_history()

    async def calculate_statistics(self) -> bool:
        """Calculate aggregated statistics @zara"""
        return await self.forecast_handler.calculate_statistics()
