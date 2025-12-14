"""JSON Schema Validation and Migration for Solar Forecast ML Integration V12.0.0 @zara

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from ..const import (
    CORRECTION_FACTOR_MAX,
    CORRECTION_FACTOR_MIN,
    DATA_VERSION,
    ML_MODEL_VERSION,
)
from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)

class DataSchemaValidator:
    """Validates and migrates JSON files to ensure they match code expectations"""

    def __init__(self, hass: HomeAssistant, data_dir: Path):
        """Initialize the schema validator @zara"""
        self.hass = hass
        self.data_dir = data_dir
        self.migration_log = []

    async def validate_and_migrate_all(self) -> bool:
        """Validate and migrate all JSON files on startup @zara"""
        try:
            _LOGGER.info("=== Starting JSON Schema Validation and Migration ===")

            await self._ensure_directory_structure()

            success = True

            success &= await self._validate_learned_weights()
            success &= await self._validate_hourly_profile()
            success &= await self._validate_model_state()
            success &= await self._validate_learned_patterns()

            success &= await self._validate_daily_forecasts()
            success &= await self._validate_hourly_predictions()
            success &= await self._validate_daily_summaries()

            success &= await self._validate_weather_forecast_corrected()
            success &= await self._validate_forecast_drift_log()
            success &= await self._validate_weather_precision_daily()
            success &= await self._validate_hourly_weather_actual()
            success &= await self._validate_weather_cache()
            success &= await self._validate_open_meteo_cache()
            success &= await self._validate_wttr_in_cache()
            success &= await self._validate_weather_source_weights()
            success &= await self._validate_weather_source_learning()

            success &= await self._validate_coordinator_state()
            success &= await self._validate_production_time_state()

            success &= await self._validate_astronomy_cache()

            success &= await self._validate_learned_geometry()
            success &= await self._validate_learned_panel_group_efficiency()
            success &= await self._validate_panel_group_sensor_state()
            success &= await self._validate_residual_model_state()
            success &= await self._validate_yield_cache()

            if self.migration_log:
                _LOGGER.info("=== Migration Summary ===")
                for entry in self.migration_log:
                    _LOGGER.info(f"  - {entry}")
            else:
                _LOGGER.info("All JSON files valid - no migrations needed")

            _LOGGER.info("=== JSON Schema Validation Complete ===")
            return success

        except Exception as e:
            _LOGGER.error(f"Schema validation failed: {e}", exc_info=True)
            return False

    def _log_migration(self, message: str) -> None:
        """Log a migration action @zara"""
        self.migration_log.append(message)
        _LOGGER.info(f"MIGRATION: {message}")

    async def _ensure_directory_structure(self) -> None:
        """Ensure all required directories exist (critical for clean installs) @zara"""
        required_dirs = [
            self.data_dir,
            self.data_dir / "data",
            self.data_dir / "stats",
            self.data_dir / "ml",
            self.data_dir / "physics",  # Physics-based learning (V12.0.0)
            self.data_dir / "logs",
            self.data_dir / "backups",
            self.data_dir / "backups" / "auto",
            self.data_dir / "backups" / "manual",
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self._log_migration(f"Created directory: {dir_path.name}/")
                except Exception as e:
                    _LOGGER.error(f"Failed to create directory {dir_path}: {e}")
                    raise

        _LOGGER.debug(f"Directory structure verified: {len(required_dirs)} directories")

    async def _read_json(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Read JSON file async @zara"""
        try:
            import json

            import aiofiles

            if not file_path.exists():
                return None

            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            _LOGGER.error(f"Failed to read {file_path.name}: {e}")
            return None

    async def _write_json(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Write JSON file async atomically @zara"""
        try:
            import json

            import aiofiles

            file_path.parent.mkdir(parents=True, exist_ok=True)

            temp_file = file_path.with_suffix(".tmp")
            async with aiofiles.open(temp_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))

            await self.hass.async_add_executor_job(temp_file.replace, file_path)
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to write {file_path.name}: {e}")
            return False

    async def _file_exists(self, file_path: Path) -> bool:
        """Check if file exists async @zara"""
        return await self.hass.async_add_executor_job(file_path.exists)

    async def _ensure_directory(self, dir_path: Path) -> None:
        """Ensure directory exists @zara"""
        if not await self.hass.async_add_executor_job(dir_path.exists):
            await self.hass.async_add_executor_job(dir_path.mkdir, True, True)
            _LOGGER.debug(f"Created directory: {dir_path}")

    async def _migrate_file(self, old_path: Path, new_path: Path) -> bool:
        """Migrate file from old to new location @zara"""
        try:
            import shutil
            await self.hass.async_add_executor_job(shutil.move, str(old_path), str(new_path))
            _LOGGER.info(f"Migrated {old_path.name} to {new_path.parent.name}/")
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to migrate {old_path.name}: {e}")
            return False

    async def _validate_learned_weights(self) -> bool:
        """Validate learned_weights json @zara"""
        file_path = self.data_dir / "ml" / "learned_weights.json"

        data = await self._read_json(file_path)
        if data is None:

            self._log_migration(
                "learned_weights.json: File missing - will be created by V3 training (14 features)"
            )
            return True

        modified = False

        if not data.get("feature_names") or not isinstance(data["feature_names"], list):
            self._log_migration(
                f"learned_weights.json: Missing feature_names, will be set by V3 training"
            )
        else:

            feature_count = len(data["feature_names"])
            if feature_count == 14:
                self._log_migration(
                    f"learned_weights.json: Found {feature_count} features (V3 - correct)"
                )
            elif feature_count == 44:
                self._log_migration(
                    f"learned_weights.json: Found {feature_count} features (old V2 model, needs retraining with V3)"
                )
            else:
                self._log_migration(
                    f"learned_weights.json: Found {feature_count} features (expected 14 for V3, needs retraining)"
                )

        if "weights" not in data or not isinstance(data["weights"], dict):
            data["weights"] = {}
            modified = True

        if "bias" not in data:
            data["bias"] = 0.0
            modified = True

        if "feature_means" not in data or not isinstance(data["feature_means"], dict):
            data["feature_means"] = {}
            modified = True

        if "feature_stds" not in data or not isinstance(data["feature_stds"], dict):
            data["feature_stds"] = {}
            modified = True

        if "accuracy" not in data or not (0.0 <= data["accuracy"] <= 1.0):
            if "accuracy" in data:
                self._log_migration(
                    f"learned_weights.json: Clamping accuracy from {data['accuracy']} to valid range"
                )
            data["accuracy"] = max(0.0, min(1.0, data.get("accuracy", 0.0)))
            modified = True

        if "training_samples" not in data or data["training_samples"] < 0:
            data["training_samples"] = 0
            modified = True

        if "last_trained" not in data:
            data["last_trained"] = dt_util.now().isoformat()
            modified = True

        if "model_version" not in data:
            data["model_version"] = ML_MODEL_VERSION
            modified = True

        cf = data.get("correction_factor", 1.0)
        if not (CORRECTION_FACTOR_MIN <= cf <= CORRECTION_FACTOR_MAX):
            self._log_migration(
                f"learned_weights.json: Clamping correction_factor from {cf} to valid range"
            )
            data["correction_factor"] = max(CORRECTION_FACTOR_MIN, min(CORRECTION_FACTOR_MAX, cf))
            modified = True

        if "weather_weights" not in data:
            data["weather_weights"] = {}
            modified = True
        if "seasonal_factors" not in data:
            data["seasonal_factors"] = {}
            modified = True
        if "feature_importance" not in data:
            data["feature_importance"] = {}
            modified = True

        if modified:
            self._log_migration("learned_weights.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    def _get_expected_feature_names(self) -> List[str]:
        """Get the 14 expected feature names (V3 format - matches production) @zara"""
        return [
            "hour",
            "day_of_year",
            "month",
            "elevation",
            "azimuth",
            "cloud_cover",
            "temperature",
            "humidity",
            "wind_speed",
            "precipitation",
            "clear_sky_radiation",
            "theoretical_max",
            "production_yesterday",
            "production_same_hour_yesterday",
        ]

    def _create_default_learned_weights(self) -> Dict[str, Any]:
        """Create default learned weights structure - matches production format @zara"""
        return {
            "weights": {},
            "bias": 0.0,
            "feature_names": self._get_expected_feature_names(),
            "feature_means": {},
            "feature_stds": {},
            "accuracy": 0.0,
            "training_samples": 0,
            "last_trained": None,
            "model_version": ML_MODEL_VERSION,
            "algorithm_used": "ridge",
            "correction_factor": 1.0,
            "weather_weights": {},
            "seasonal_factors": {},
            "feature_importance": {},
            "file_format_version": "1.0",
            "last_saved": None,
        }

    async def _validate_hourly_profile(self) -> bool:
        """Validate hourly_profile json @zara"""
        file_path = self.data_dir / "ml" / "hourly_profile.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("hourly_profile.json: Creating new file with sine curve default")
            data = self._create_default_hourly_profile()
            return await self._write_json(file_path, data)

        modified = False

        if "hourly_averages" not in data or not isinstance(data["hourly_averages"], dict):
            data["hourly_averages"] = {}
            modified = True

        for hour in range(24):
            hour_str = str(hour)
            if hour_str not in data["hourly_averages"]:

                data["hourly_averages"][hour_str] = self._calculate_sine_hour(hour)
                modified = True
            elif data["hourly_averages"][hour_str] < 0:
                self._log_migration(f"hourly_profile.json: Clamping negative value for hour {hour}")
                data["hourly_averages"][hour_str] = 0.0
                modified = True

        if "samples_count" not in data or data["samples_count"] < 0:
            data["samples_count"] = 0
            modified = True

        if "last_updated" not in data:
            data["last_updated"] = dt_util.now().isoformat()
            modified = True

        if "confidence" not in data or not (0.0 <= data["confidence"] <= 1.0):
            data["confidence"] = 0.1
            modified = True

        if "hourly_factors" not in data:
            data["hourly_factors"] = {}
            modified = True
        if "seasonal_adjustment" not in data:
            data["seasonal_adjustment"] = {}
            modified = True

        if modified:
            self._log_migration("hourly_profile.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    def _calculate_sine_hour(self, hour: int) -> float:
        """Calculate sine curve value for an hour @zara"""
        start_hour = 6
        daylight_hours = 12

        if start_hour <= hour < start_hour + daylight_hours:
            relative_hour = (hour - start_hour) / daylight_hours
            sine_value = math.sin(relative_hour * math.pi + 0.01)
            return max(0.0, sine_value * 1.0)
        return 0.0

    def _create_default_hourly_profile(self) -> Dict[str, Any]:
        """Create default hourly profile structure - matches production format @zara"""
        hourly_averages = {}
        for hour in range(24):
            hourly_averages[str(hour)] = self._calculate_sine_hour(hour)

        return {
            "hourly_averages": hourly_averages,
            "samples_count": 0,
            "last_updated": None,
            "confidence": 0.1,
            "hourly_factors": {},
            "seasonal_adjustment": {},
            "file_format_version": "3.0.0",
            "last_saved": None,
        }

    async def _validate_model_state(self) -> bool:
        """Validate model_state json @zara"""
        file_path = self.data_dir / "ml" / "model_state.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("model_state.json: Creating new file")
            data = self._create_default_model_state()
            return await self._write_json(file_path, data)

        modified = False

        if "version" not in data:
            data["version"] = DATA_VERSION
            modified = True

        if "model_loaded" not in data or not isinstance(data["model_loaded"], bool):
            data["model_loaded"] = False
            modified = True

        if "last_training" not in data:
            data["last_training"] = None
            modified = True

        if "training_samples" not in data or data["training_samples"] < 0:
            data["training_samples"] = 0
            modified = True

        current_acc = data.get("current_accuracy")
        if (
            current_acc is None
            or not isinstance(current_acc, (int, float))
            or not (0.0 <= current_acc <= 1.0)
        ):
            data["current_accuracy"] = 0.0
            modified = True

        if "status" not in data or data["status"] not in [
            "uninitialized",
            "ready",
            "training",
            "error",
        ]:
            data["status"] = "uninitialized"
            modified = True

        if modified:
            self._log_migration("model_state.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_model_state(self) -> Dict[str, Any]:
        """Create default model state structure - matches production V1.0 format @zara"""
        return {
            "version": "1.0",
            "model_loaded": False,
            "last_training": None,
            "training_samples": 0,
            "current_accuracy": 0.0,
            "status": "uninitialized",
            "peak_power_kw": None,
            "model_info": {
                "version": "1.0",
                "type": None,
            },
            "performance_metrics": {
                "avg_prediction_time_ms": 0.0,
                "error_rate": 0.0,
            },
            "last_updated": None,
        }

    async def _validate_daily_forecasts(self) -> bool:
        """Validate daily_forecasts json @zara"""
        file_path = self.data_dir / "stats" / "daily_forecasts.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("daily_forecasts.json: Creating new file with full structure")
            data = self._create_default_daily_forecasts()
            return await self._write_json(file_path, data)

        modified = False

        if "current_day" in data or "forecasts" in data:
            self._log_migration(
                "daily_forecasts.json: Migrating from OLD structure to NEW structure"
            )
            data = self._create_default_daily_forecasts()
            modified = True

        if "version" not in data:
            data["version"] = DATA_VERSION
            modified = True

        if "today" not in data or not isinstance(data["today"], dict):
            data["today"] = self._create_default_today_block()
            modified = True
        else:
            if await self._validate_today_block(data["today"]):
                modified = True

        if "statistics" not in data or not isinstance(data["statistics"], dict):
            data["statistics"] = self._create_default_statistics_block()
            modified = True
        else:
            if await self._validate_statistics_block(data["statistics"]):
                modified = True

        if "history" not in data or not isinstance(data["history"], list):
            data["history"] = []
            modified = True

        if "metadata" not in data or not isinstance(data["metadata"], dict):
            data["metadata"] = {
                "retention_days": 730,
                "history_entries": len(data.get("history", [])),
                "last_update": None,
            }
            modified = True

        if modified:
            self._log_migration("daily_forecasts.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    async def _validate_today_block(self, today: Dict[str, Any]) -> bool:
        """Validate today block structure @zara"""
        modified = False

        if "date" not in today:
            today["date"] = None
            modified = True

        sub_blocks = [
            ("forecast_day", self._create_default_forecast_day),
            ("forecast_tomorrow", self._create_default_forecast_tomorrow),
            ("forecast_day_after_tomorrow", self._create_default_forecast_day_after),
            ("forecast_best_hour", self._create_default_forecast_best_hour),
            ("actual_best_hour", self._create_default_actual_best_hour),
            ("forecast_next_hour", self._create_default_forecast_next_hour),
            ("production_time", self._create_default_production_time),
            ("peak_today", self._create_default_peak_today),
            ("yield_today", self._create_default_yield_today),
            ("consumption_today", self._create_default_consumption_today),
            ("autarky", self._create_default_autarky),
            ("finalized", self._create_default_finalized),
        ]

        for block_name, create_func in sub_blocks:
            if block_name not in today or not isinstance(today[block_name], dict):
                today[block_name] = create_func()
                modified = True

        return modified

    async def _validate_statistics_block(self, stats: Dict[str, Any]) -> bool:
        """Validate statistics block structure @zara"""
        modified = False

        stat_blocks = [
            ("all_time_peak", self._create_default_all_time_peak),
            ("current_week", self._create_default_current_week),
            ("current_month", self._create_default_current_month),
            ("last_7_days", self._create_default_last_7_days),
            ("last_30_days", self._create_default_last_30_days),
            ("last_365_days", self._create_default_last_365_days),
        ]

        for block_name, create_func in stat_blocks:
            if block_name not in stats or not isinstance(stats[block_name], dict):
                stats[block_name] = create_func()
                modified = True

        return modified

    def _create_default_daily_forecasts(self) -> Dict[str, Any]:
        """Create complete default daily_forecasts structure - matches production V12.0.0 @zara"""
        return {
            "version": "3.0.0",
            "today": self._create_default_today_block(),
            "statistics": self._create_default_statistics_block(),
            "history": [],
            "metadata": {"retention_days": 730, "history_entries": 0, "last_update": None},
        }

    def _create_default_today_block(self) -> Dict[str, Any]:
        """Create default today block @zara"""
        return {
            "date": None,
            "forecast_day": self._create_default_forecast_day(),
            "forecast_tomorrow": self._create_default_forecast_tomorrow(),
            "forecast_day_after_tomorrow": self._create_default_forecast_day_after(),
            "forecast_best_hour": self._create_default_forecast_best_hour(),
            "actual_best_hour": self._create_default_actual_best_hour(),
            "forecast_next_hour": self._create_default_forecast_next_hour(),
            "production_time": self._create_default_production_time(),
            "peak_today": self._create_default_peak_today(),
            "yield_today": self._create_default_yield_today(),
            "consumption_today": self._create_default_consumption_today(),
            "autarky": self._create_default_autarky(),
            "finalized": self._create_default_finalized(),
        }

    def _create_default_forecast_day(self) -> Dict[str, Any]:
        """Create default forecast_day block - matches production V12.0.0 format @zara"""
        return {
            "prediction_kwh": None,
            "prediction_kwh_raw": None,
            "prediction_kwh_display": None,  # For intraday corrections (09:05/12:05)
            "safeguard_applied": False,
            "safeguard_reduction_kwh": 0.0,
            "locked": False,
            "locked_at": None,
            "source": None,
            "intraday_corrected": False,
            "intraday_corrected_at": None,
        }

    def _create_default_forecast_tomorrow(self) -> Dict[str, Any]:
        """Create default forecast_tomorrow block @zara"""
        return {
            "date": None,
            "prediction_kwh": None,
            "locked": False,
            "locked_at": None,
            "source": None,
            "updates": [],
        }

    def _create_default_forecast_day_after(self) -> Dict[str, Any]:
        """Create default forecast_day_after_tomorrow block @zara"""
        return {
            "date": None,
            "prediction_kwh": None,
            "locked": False,
            "next_update": None,
            "source": None,
            "updates": [],
        }

    def _create_default_forecast_best_hour(self) -> Dict[str, Any]:
        """Create default forecast_best_hour block @zara"""
        return {
            "hour": None,
            "prediction_kwh": None,
            "locked": False,
            "locked_at": None,
            "source": None,
        }

    def _create_default_actual_best_hour(self) -> Dict[str, Any]:
        """Create default actual_best_hour block @zara"""
        return {"hour": None, "actual_kwh": None, "saved_at": None}

    def _create_default_forecast_next_hour(self) -> Dict[str, Any]:
        """Create default forecast_next_hour block @zara"""
        return {"period": None, "prediction_kwh": None, "updated_at": None, "source": None}

    def _create_default_production_time(self) -> Dict[str, Any]:
        """Create default production_time block @zara"""
        return {
            "active": False,
            "duration_seconds": 0,
            "start_time": None,
            "end_time": None,
            "last_power_above_10w": None,
            "zero_power_since": None,
        }

    def _create_default_peak_today(self) -> Dict[str, Any]:
        """Create default peak_today block @zara"""
        return {"power_w": 0.0, "at": None}

    def _create_default_yield_today(self) -> Dict[str, Any]:
        """Create default yield_today block @zara"""
        return {"kwh": None, "sensor": None}

    def _create_default_consumption_today(self) -> Dict[str, Any]:
        """Create default consumption_today block @zara"""
        return {"kwh": None, "sensor": None}

    def _create_default_autarky(self) -> Dict[str, Any]:
        """Create default autarky block @zara"""
        return {"percent": None, "calculated_at": None}

    def _create_default_finalized(self) -> Dict[str, Any]:
        """Create default finalized block @zara"""
        return {
            "yield_kwh": None,
            "consumption_kwh": None,
            "production_hours": None,
            "accuracy_percent": None,
            "at": None,
        }

    def _create_default_statistics_block(self) -> Dict[str, Any]:
        """Create default statistics block @zara"""
        return {
            "all_time_peak": self._create_default_all_time_peak(),
            "current_week": self._create_default_current_week(),
            "current_month": self._create_default_current_month(),
            "last_7_days": self._create_default_last_7_days(),
            "last_30_days": self._create_default_last_30_days(),
            "last_365_days": self._create_default_last_365_days(),
        }

    def _create_default_all_time_peak(self) -> Dict[str, Any]:
        """Create default all_time_peak block @zara"""
        return {"power_w": 0.0, "date": None, "at": None}

    def _create_default_current_week(self) -> Dict[str, Any]:
        """Create default current_week block @zara"""
        return {
            "period": None,
            "date_range": None,
            "yield_kwh": 0.0,
            "consumption_kwh": 0.0,
            "days": 0,
            "updated_at": None,
        }

    def _create_default_current_month(self) -> Dict[str, Any]:
        """Create default current_month block @zara"""
        return {
            "period": None,
            "yield_kwh": 0.0,
            "consumption_kwh": 0.0,
            "avg_autarky": 0.0,
            "days": 0,
            "updated_at": None,
        }

    def _create_default_last_7_days(self) -> Dict[str, Any]:
        """Create default last_7_days block @zara"""
        return {
            "avg_yield_kwh": 0.0,
            "avg_accuracy": 0.0,
            "total_yield_kwh": 0.0,
            "calculated_at": None,
        }

    def _create_default_last_30_days(self) -> Dict[str, Any]:
        """Create default last_30_days block @zara"""
        return {
            "avg_yield_kwh": 0.0,
            "avg_accuracy": 0.0,
            "total_yield_kwh": 0.0,
            "calculated_at": None,
        }

    def _create_default_last_365_days(self) -> Dict[str, Any]:
        """Create default last_365_days block @zara"""
        return {"avg_yield_kwh": 0.0, "total_yield_kwh": 0.0, "calculated_at": None}

    async def _validate_coordinator_state(self) -> bool:
        """Validate coordinator_state json @zara"""
        file_path = self.data_dir / "data" / "coordinator_state.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("coordinator_state.json: Creating new file")
            data = self._create_default_coordinator_state()
            return await self._write_json(file_path, data)

        modified = False

        if "version" not in data:
            data["version"] = DATA_VERSION
            modified = True

        if "expected_daily_production" not in data:
            data["expected_daily_production"] = None
            modified = True

        if "last_set_date" not in data:
            data["last_set_date"] = None
            modified = True

        if "last_updated" not in data:
            data["last_updated"] = None
            modified = True

        if "last_collected_hour" not in data:
            data["last_collected_hour"] = None
            modified = True

        if modified:
            self._log_migration("coordinator_state.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_coordinator_state(self) -> Dict[str, Any]:
        """Create default coordinator_state structure - matches production V1.0 format @zara"""
        return {
            "version": "1.0",
            "expected_daily_production": None,
            "last_set_date": None,
            "last_updated": None,
            "last_collected_hour": None,
        }

    async def _validate_weather_forecast_corrected(self) -> bool:
        """Validate weather_forecast_corrected.json - SINGLE SOURCE OF TRUTH @zara"""
        file_path = self.data_dir / "stats" / "weather_forecast_corrected.json"

        data = await self._read_json(file_path)
        if data is None:

            self._log_migration(
                "weather_forecast_corrected.json: File missing - creating empty template"
            )
            data = {
                "version": "3.3",
                "forecast": {},
                "metadata": self._create_default_weather_corrected_metadata(),
            }
            return await self._write_json(file_path, data)

        modified = False

        if "version" not in data or data["version"] < "3.3":
            data["version"] = "3.3"
            modified = True

        if "forecast" not in data or not isinstance(data["forecast"], dict):
            self._log_migration(
                "weather_forecast_corrected.json: Missing 'forecast' block - needs weather refresh"
            )
            data["forecast"] = {}
            modified = True
        else:

            forecast = data["forecast"]
            if forecast:

                sample_checked = False
                for date_str, hours in forecast.items():
                    if not isinstance(hours, dict):
                        continue
                    for hour_str, hour_data in hours.items():
                        if not isinstance(hour_data, dict):
                            continue

                        required_fields = [
                            "temperature",
                            "solar_radiation_wm2",
                            "wind",
                            "humidity",
                            "clouds",
                        ]
                        missing = [f for f in required_fields if f not in hour_data]
                        if missing:
                            self._log_migration(
                                f"weather_forecast_corrected.json: Sample hour missing fields: {missing}"
                            )
                        sample_checked = True
                        break
                    if sample_checked:
                        break

        if "metadata" not in data or not isinstance(data["metadata"], dict):
            data["metadata"] = self._create_default_weather_corrected_metadata()
            modified = True
        else:

            meta = data["metadata"]
            if "corrections_applied" not in meta:
                meta["corrections_applied"] = {
                    "temperature": 0.0,
                    "solar_radiation_wm2": 1.0,
                    "clouds": 1.0,
                    "humidity": 1.0,
                    "wind": 1.0,
                    "rain": 1.0,
                    "pressure": 0.0,
                }
                modified = True
            if "confidence_scores" not in meta:
                meta["confidence_scores"] = {
                    "temperature": 0.0,
                    "solar_radiation_wm2": 0.0,
                    "clouds": 0.0,
                    "humidity": 0.0,
                    "wind": 0.0,
                    "rain": 0.0,
                    "pressure": 0.0,
                }
                modified = True

            # Remove deprecated fields if they exist (migration cleanup)
            if "cloud_blending" in meta:
                del meta["cloud_blending"]
                modified = True
            if "cloud_model" in meta:
                del meta["cloud_model"]
                modified = True

            if "rb_overall_correction" not in meta:
                meta["rb_overall_correction"] = {
                    "factor": 1.0,
                    "confidence": 0.0,
                    "sample_days": 0,
                    "last_updated": dt_util.now().isoformat(),
                    "note": "Learns from RB prediction vs actual over 7+ days. Accounts for installation-specific factors (tilt, orientation, degradation, etc.)",
                }
                modified = True

            if "self_healing" not in meta:
                meta["self_healing"] = self._create_default_self_healing_metadata()
                self._log_migration(
                    "weather_forecast_corrected.json: Added self_healing metadata block"
                )
                modified = True
            else:

                sh = meta["self_healing"]
                default_sh = self._create_default_self_healing_metadata()
                for key, default_val in default_sh.items():
                    if key not in sh:
                        sh[key] = default_val
                        modified = True

        if modified:
            self._log_migration("weather_forecast_corrected.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_weather_corrected_metadata(self) -> Dict[str, Any]:
        """Create default metadata for weather_forecast_corrected.json @zara

        IMPORTANT: This structure matches the MASTER JSON in production.
        Do NOT add fields that don't exist in the master!
        """
        return {
            "created": dt_util.now().isoformat(),
            "correction_source": "weather_precision_daily.json",
            "correction_method": "7_day_rolling_average",
            "sample_days": 0,
            "min_confidence": 0.0,
            "corrections_applied": {
                "temperature": 0.0,
                "solar_radiation_wm2": 1.0,
                "clouds": 1.0,
                "humidity": 1.0,
                "wind": 1.0,
                "rain": 1.0,
                "pressure": 0.0,
            },
            "confidence_scores": {
                "temperature": 0.0,
                "solar_radiation_wm2": 0.0,
                "clouds": 0.0,
                "humidity": 0.0,
                "wind": 0.0,
                "rain": 0.0,
                "pressure": 0.0,
            },
            "rb_overall_correction": {
                "factor": 1.0,
                "confidence": 0.0,
                "sample_days": 0,
                "last_updated": dt_util.now().isoformat(),
                "note": "Learns from RB prediction vs actual over 7+ days. Accounts for installation-specific factors (tilt, orientation, degradation, etc.)",
            },
            "self_healing": self._create_default_self_healing_metadata(),
        }

    def _create_default_self_healing_metadata(self) -> Dict[str, Any]:
        """Create default self_healing metadata structure for weather_forecast_corrected.json @zara"""
        return {
            "weather_cache_status": "ok",
            "open_meteo_status": "ok",
            "precision_data_status": "ok",
            "astronomy_cache_status": "ok",
            "consecutive_weather_failures": 0,
            "consecutive_open_meteo_failures": 0,
            "last_weather_error": None,
            "last_open_meteo_error": None,
            "last_fallback_used": None,
            "fallback_count_today": 0,
        }

    async def _validate_forecast_drift_log(self) -> bool:
        """Validate forecast_drift_log.json - Tracks forecast drift for analysis @zara

        This file logs forecast drift data for future pattern analysis.
        Used by the midday check V3.0 to understand forecast accuracy patterns.
        """
        file_path = self.data_dir / "stats" / "forecast_drift_log.json"

        data = await self._read_json(file_path)
        if data is None:
            # Create empty log file
            self._log_migration("forecast_drift_log.json: Creating new file")
            data = {
                "version": "1.0",
                "entries": [],
            }
            return await self._write_json(file_path, data)

        modified = False

        if "version" not in data:
            data["version"] = "1.0"
            modified = True

        if "entries" not in data or not isinstance(data["entries"], list):
            data["entries"] = []
            modified = True

        # Keep only last 30 entries (30 days of data)
        if len(data["entries"]) > 30:
            data["entries"] = data["entries"][-30:]
            modified = True

        if modified:
            self._log_migration("forecast_drift_log.json: Schema updated")
            return await self._write_json(file_path, data)

        return True

    async def _validate_weather_precision_daily(self) -> bool:
        """Validate weather_precision_daily.json - Correction factors @zara"""
        file_path = self.data_dir / "stats" / "weather_precision_daily.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("weather_precision_daily.json: Creating new file")
            data = self._create_default_weather_precision_daily()
            return await self._write_json(file_path, data)

        modified = False

        if "rolling_averages" not in data or not isinstance(data["rolling_averages"], dict):
            data["rolling_averages"] = self._create_default_rolling_averages()
            modified = True
        else:
            ra = data["rolling_averages"]

            if "correction_factors" in ra:

                cf = ra["correction_factors"]

                default_values = {
                    "temperature": 0.0,
                    "solar_radiation_wm2": 1.0,
                    "clouds": 1.0,
                    "humidity": 1.0,
                    "wind": 1.0,
                    "rain": 1.0,
                    "pressure": 0.0,
                }
                for field, default in default_values.items():
                    if field not in cf:
                        cf[field] = default
                        modified = True

                if "confidence" not in ra:
                    ra["confidence"] = {
                        "temperature": 0.0,
                        "solar_radiation_wm2": 0.0,
                        "clouds": 0.0,
                        "humidity": 0.0,
                        "wind": 0.0,
                        "rain": 0.0,
                        "pressure": 0.0,
                    }
                    modified = True
                else:

                    conf = ra["confidence"]
                    for field in default_values.keys():
                        if field not in conf:
                            conf[field] = 0.0
                            modified = True

                if "sample_days" not in ra:
                    ra["sample_days"] = 0
                    modified = True

                if "updated_at" not in ra:
                    ra["updated_at"] = None
                    modified = True
            else:

                for period in ["7_day", "30_day"]:
                    if period not in ra or not isinstance(ra[period], dict):
                        ra[period] = self._create_default_rolling_average_period()
                        modified = True
                    else:

                        if "correction_factors" not in ra[period]:
                            ra[period]["correction_factors"] = {
                                "temperature": 1.0,
                                "solar_radiation_wm2": 1.0,
                                "wind": 1.0,
                                "humidity": 1.0,
                                "rain": 1.0,
                            }
                            modified = True

                        if "confidence" not in ra[period]:
                            ra[period]["confidence"] = {
                                "temperature": 0.0,
                                "solar_radiation_wm2": 0.0,
                                "wind": 0.0,
                                "humidity": 0.0,
                                "rain": 0.0,
                            }
                            modified = True

                        if "sample_days" not in ra[period]:
                            ra[period]["sample_days"] = 0
                            modified = True

        if "daily_tracking" not in data:
            data["daily_tracking"] = {}
            modified = True

        if "metadata" not in data or not isinstance(data["metadata"], dict):

            if "7_day" in data.get("rolling_averages", {}):
                data["metadata"] = {
                    "created": dt_util.now().isoformat(),
                    "last_updated": None,
                    "total_days_tracked": 0,
                    "sensors_configured": [],
                    "sensors_optional": True,
                }
                modified = True

        if modified:
            self._log_migration("weather_precision_daily.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_weather_precision_daily(self) -> Dict[str, Any]:
        """Create default weather_precision_daily structure - matches production flat format @zara"""
        return {
            "daily_tracking": {},
            "rolling_averages": {
                "sample_days": 0,
                "correction_factors": {
                    "temperature": 0.0,
                    "solar_radiation_wm2": 1.0,
                    "clouds": 1.0,
                    "humidity": 1.0,
                    "wind": 1.0,
                    "rain": 1.0,
                    "pressure": 0.0,
                },
                "confidence": {
                    "temperature": 0.0,
                    "solar_radiation_wm2": 0.0,
                    "clouds": 0.0,
                    "humidity": 0.0,
                    "wind": 0.0,
                    "rain": 0.0,
                    "pressure": 0.0,
                },
                "updated_at": None,
            },
            "metadata": {
                "created": None,
                "last_updated": None,
                "total_days_tracked": 0,
                "sensors_configured": [],
                "sensors_optional": True,
            },
        }

    def _create_default_rolling_averages(self) -> Dict[str, Any]:
        """Create default rolling_averages structure - flat format @zara"""
        return {
            "sample_days": 0,
            "correction_factors": {
                "temperature": 0.0,
                "solar_radiation_wm2": 1.0,
                "clouds": 1.0,
                "humidity": 1.0,
                "wind": 1.0,
                "rain": 1.0,
                "pressure": 0.0,
            },
            "confidence": {
                "temperature": 0.0,
                "solar_radiation_wm2": 0.0,
                "clouds": 0.0,
                "humidity": 0.0,
                "wind": 0.0,
                "rain": 0.0,
                "pressure": 0.0,
            },
            "updated_at": None,
        }

    def _create_default_rolling_average_period(self) -> Dict[str, Any]:
        """Create default rolling average period structure (legacy nested format) @zara"""
        return {
            "sample_days": 0,
            "correction_factors": {
                "temperature": 1.0,
                "solar_radiation_wm2": 1.0,
                "wind": 1.0,
                "humidity": 1.0,
                "rain": 1.0,
            },
            "confidence": {
                "temperature": 0.0,
                "solar_radiation_wm2": 0.0,
                "wind": 0.0,
                "humidity": 0.0,
                "rain": 0.0,
            },
        }

    async def _validate_hourly_weather_actual(self) -> bool:
        """Validate hourly_weather_actual.json - Local sensor readings @zara"""
        file_path = self.data_dir / "stats" / "hourly_weather_actual.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("hourly_weather_actual.json: Creating new file")
            data = self._create_default_hourly_weather_actual()
            return await self._write_json(file_path, data)

        modified = False

        if "version" not in data or data["version"] < "1.1":
            data["version"] = "1.1"
            modified = True

        if "metadata" not in data or not isinstance(data["metadata"], dict):
            data["metadata"] = {
                "created_at": dt_util.now().isoformat(),
                "last_updated": None,
            }
            modified = True

        if "hourly_data" not in data or not isinstance(data["hourly_data"], dict):
            data["hourly_data"] = {}
            modified = True
        else:

            for date_str, hours in data["hourly_data"].items():
                if not isinstance(hours, dict):
                    continue
                for hour_str, hour_data in hours.items():
                    if not isinstance(hour_data, dict):
                        continue

                    if hour_data.get("frost_detected") is not None:

                        if "frost_analysis" not in hour_data:

                            hour_data["frost_analysis"] = self._create_default_frost_analysis()
                            modified = True
                        else:

                            fa = hour_data["frost_analysis"]
                            if not isinstance(fa, dict):
                                hour_data["frost_analysis"] = self._create_default_frost_analysis()
                                modified = True
                            else:

                                default_fa = self._create_default_frost_analysis()
                                for key, default_val in default_fa.items():
                                    if key not in fa:
                                        fa[key] = default_val
                                        modified = True

        if modified:
            self._log_migration("hourly_weather_actual.json: Schema updated and validated (V1.1 frost_analysis)")
            return await self._write_json(file_path, data)

        return True

    def _create_default_frost_analysis(self) -> Dict[str, Any]:
        """Create default frost_analysis structure for V12.0.0 enhanced frost detection @zara"""
        return {
            "dewpoint_c": None,
            "frost_margin_c": None,
            "frost_probability": 0.0,
            "correlation_diff_percent": None,
            "threshold_used_percent": None,
            "detection_method": "unknown",
            "wind_frost_factor": None,
            "physical_frost_possible": False
        }

    def _create_default_hourly_weather_actual(self) -> Dict[str, Any]:
        """Create default hourly_weather_actual structure @zara"""
        return {
            "version": "1.1",
            "metadata": {
                "created_at": dt_util.now().isoformat(),
                "last_updated": None,
                "frost_detection_version": "correlation_enhanced",
            },
            "hourly_data": {},
        }

    async def _validate_weather_cache(self) -> bool:
        """Validate weather_cache.json - maintains backwards compatibility placeholder. @zara"""
        file_path = self.data_dir / "data" / "weather_cache.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration(
                "weather_cache.json: Creating minimal placeholder (Open-Meteo is the data source)"
            )
            data = self._create_default_weather_cache()
            return await self._write_json(file_path, data)

        # Just ensure version marker exists
        modified = False
        if data.get("version") != "3.0":
            data["version"] = "3.0"
            data["_note"] = "Open-Meteo (open_meteo_cache.json) is the primary data source"
            data["_migration_date"] = dt_util.now().isoformat()
            modified = True

        if modified:
            self._log_migration(
                "weather_cache.json: Schema updated to v3.0"
            )
            return await self._write_json(file_path, data)

        return True

    def _create_default_weather_cache(self) -> Dict[str, Any]:
        """Create weather_cache placeholder for backwards compatibility. @zara"""
        return {
            "version": "3.0",
            "_note": "Open-Meteo (open_meteo_cache.json) is the primary data source",
            "_migration_date": dt_util.now().isoformat(),
            "cached_at": None,
            "forecast_hours": [],
        }

    async def _validate_learned_patterns(self) -> bool:
        """Validate learned_patterns.json - Pattern learning data @zara"""
        file_path = self.data_dir / "ml" / "learned_patterns.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("learned_patterns.json: Creating new file")
            data = self._create_default_learned_patterns()
            return await self._write_json(file_path, data)

        modified = False

        if "version" not in data:
            data["version"] = "1.0.0"
            modified = True

        required_blocks = [
            "geometry_factors",
            "geometry_corrections",
            "cloud_impacts",
            "seasonal_adjustments",
            "efficiency_curves",
            "metadata",
        ]

        for block in required_blocks:
            if block not in data:
                if block == "geometry_factors":
                    data[block] = self._create_default_geometry_factors()
                elif block == "geometry_corrections":
                    data[block] = {"monthly": {}}
                elif block == "cloud_impacts":
                    data[block] = {"description": "Learned cloud dampening", "hour_patterns": {}}
                elif block == "seasonal_adjustments":
                    data[block] = self._create_default_seasonal_adjustments()
                elif block == "efficiency_curves":
                    data[block] = self._create_default_efficiency_curves()
                elif block == "metadata":
                    data[block] = {
                        "created": dt_util.now().strftime("%Y-%m-%d"),
                        "total_learning_days": 0,
                        "clear_sky_days_detected": 0,
                        "cloudy_days_detected": 0,
                        "last_pattern_update": None,
                    }
                modified = True

        if modified:
            self._log_migration("learned_patterns.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_learned_patterns(self) -> Dict[str, Any]:
        """Create default learned_patterns structure @zara"""
        return {
            "version": "1.0.0",
            "last_updated": None,
            "geometry_factors": self._create_default_geometry_factors(),
            "geometry_corrections": {"monthly": {}},
            "cloud_impacts": {
                "description": "Learned cloud dampening by hour and cloud coverage bucket",
                "hour_patterns": {},
            },
            "seasonal_adjustments": self._create_default_seasonal_adjustments(),
            "efficiency_curves": self._create_default_efficiency_curves(),
            "metadata": {
                "created": dt_util.now().strftime("%Y-%m-%d"),
                "total_learning_days": 0,
                "clear_sky_days_detected": 0,
                "cloudy_days_detected": 0,
                "last_pattern_update": None,
            },
            "description": "Pattern-based learning data for adaptive solar forecasting",
        }

    def _create_default_geometry_factors(self) -> Dict[str, Any]:
        """Create default geometry factors @zara"""
        return {
            "description": "Learned correction factors for tilted panels by sun elevation",
            "sun_elevation_ranges": {
                "0_5": {"factor": 2.5, "samples": 0, "confidence": 0.3, "description": "Very low sun angle"},
                "5_10": {"factor": 2.0, "samples": 0, "confidence": 0.3, "description": "Low sun angle"},
                "10_15": {"factor": 1.6, "samples": 0, "confidence": 0.3, "description": "Medium-low sun angle"},
                "15_20": {"factor": 1.4, "samples": 0, "confidence": 0.3, "description": "Medium sun angle"},
                "20_25": {"factor": 1.2, "samples": 0, "confidence": 0.3, "description": "Medium-high sun angle"},
                "25_30": {"factor": 1.1, "samples": 0, "confidence": 0.3, "description": "High sun angle"},
                "30_plus": {"factor": 1.0, "samples": 0, "confidence": 0.3, "description": "Very high sun angle"},
            },
        }

    def _create_default_seasonal_adjustments(self) -> Dict[str, Any]:
        """Create default seasonal adjustments @zara"""
        months = {
            "1": {"name": "January", "morning_boost": 2.1, "midday_factor": 1.6, "samples": 0},
            "2": {"name": "February", "morning_boost": 2.0, "midday_factor": 1.5, "samples": 0},
            "3": {"name": "March", "morning_boost": 1.8, "midday_factor": 1.4, "samples": 0},
            "4": {"name": "April", "morning_boost": 1.6, "midday_factor": 1.3, "samples": 0},
            "5": {"name": "May", "morning_boost": 1.4, "midday_factor": 1.2, "samples": 0},
            "6": {"name": "June", "morning_boost": 1.2, "midday_factor": 1.1, "samples": 0},
            "7": {"name": "July", "morning_boost": 1.2, "midday_factor": 1.1, "samples": 0},
            "8": {"name": "August", "morning_boost": 1.3, "midday_factor": 1.2, "samples": 0},
            "9": {"name": "September", "morning_boost": 1.5, "midday_factor": 1.3, "samples": 0},
            "10": {"name": "October", "morning_boost": 1.7, "midday_factor": 1.4, "samples": 0},
            "11": {"name": "November", "morning_boost": 2.0, "midday_factor": 1.6, "samples": 0},
            "12": {"name": "December", "morning_boost": 2.2, "midday_factor": 1.7, "samples": 0},
        }
        return {"description": "Seasonal behavior patterns by month", "months": months}

    def _create_default_efficiency_curves(self) -> Dict[str, Any]:
        """Create default efficiency curves @zara"""
        return {
            "description": "Learned production efficiency at different radiation levels",
            "radiation_buckets": {
                "0_100": {"efficiency": 0.95, "samples": 0, "description": "Very low radiation"},
                "100_200": {"efficiency": 0.95, "samples": 0, "description": "Low radiation"},
                "200_400": {"efficiency": 0.95, "samples": 0, "description": "Medium radiation"},
                "400_600": {"efficiency": 0.95, "samples": 0, "description": "Good radiation"},
                "600_800": {"efficiency": 0.95, "samples": 0, "description": "High radiation"},
                "800_plus": {"efficiency": 0.95, "samples": 0, "description": "Very high radiation"},
            },
        }

    async def _validate_daily_summaries(self) -> bool:
        """Validate daily_summaries.json - Historical daily analysis @zara"""
        file_path = self.data_dir / "stats" / "daily_summaries.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("daily_summaries.json: Creating new file")
            data = self._create_default_daily_summaries()
            return await self._write_json(file_path, data)

        modified = False

        if "version" not in data:
            data["version"] = "2.0"
            modified = True

        if "summaries" not in data or not isinstance(data["summaries"], list):
            data["summaries"] = []
            modified = True

        if "last_updated" not in data:
            data["last_updated"] = None
            modified = True

        if modified:
            self._log_migration("daily_summaries.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_daily_summaries(self) -> Dict[str, Any]:
        """Create default daily_summaries structure @zara"""
        return {
            "version": "2.0",
            "last_updated": None,
            "summaries": [],
        }

    async def _validate_production_time_state(self) -> bool:
        """Validate production_time_state.json - Production time tracking @zara"""
        file_path = self.data_dir / "data" / "production_time_state.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("production_time_state.json: Creating new file")
            data = self._create_default_production_time_state()
            return await self._write_json(file_path, data)

        modified = False

        if "version" not in data:
            data["version"] = "1.0"
            modified = True

        required_fields = {
            "date": None,
            "accumulated_hours": 0.0,
            "is_active": False,
            "start_time": None,
            "last_updated": None,
            "production_time_today": "00:00:00",
        }

        for field, default in required_fields.items():
            if field not in data:
                data[field] = default
                modified = True

        if modified:
            self._log_migration("production_time_state.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_production_time_state(self) -> Dict[str, Any]:
        """Create default production_time_state structure @zara"""
        return {
            "version": "1.0",
            "date": None,
            "accumulated_hours": 0.0,
            "is_active": False,
            "start_time": None,
            "last_updated": None,
            "production_time_today": "00:00:00",
        }

    async def _validate_hourly_predictions(self) -> bool:
        """Validate hourly_predictions.json - Detailed hourly forecasts @zara"""
        file_path = self.data_dir / "stats" / "hourly_predictions.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("hourly_predictions.json: Creating new file")
            data = self._create_default_hourly_predictions()
            return await self._write_json(file_path, data)

        modified = False

        if "version" not in data:
            data["version"] = "2.0"
            modified = True

        if "predictions" not in data or not isinstance(data["predictions"], list):
            data["predictions"] = []
            modified = True

        if "last_updated" not in data:
            data["last_updated"] = None
            modified = True

        if modified:
            self._log_migration("hourly_predictions.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_hourly_predictions(self) -> Dict[str, Any]:
        """Create default hourly_predictions structure @zara"""
        return {
            "version": "2.0",
            "last_updated": None,
            "best_hour_today": None,
            "predictions": [],
        }

    async def _validate_open_meteo_cache(self) -> bool:
        """Validate open_meteo_cache.json - SINGLE SOURCE for all weather data @zara

        IMPORTANT: Open-Meteo is now the ONLY weather data source!
        - GHI (global_horizontal_irradiance) is used DIRECTLY
        - cloud_cover, temperature, humidity etc. come from Open-Meteo API
        - NO blending with other weather sources
        - NO physics-based cloud calculations needed
        """
        file_path = self.data_dir / "data" / "open_meteo_cache.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration(
                "open_meteo_cache.json: CRITICAL - Creating new file "
                "(This is the SINGLE SOURCE for all weather data!)"
            )
            data = self._create_default_open_meteo_cache()
            return await self._write_json(file_path, data)

        modified = False

        # Update to version 2.0 for direct radiation model
        if data.get("version", "1.0") < "2.0":
            data["version"] = "2.0"
            modified = True
            self._log_migration(
                "open_meteo_cache.json: Upgraded to V2.0 (direct radiation model)"
            )

        # Validate and ensure complete metadata
        if "metadata" not in data or not isinstance(data["metadata"], dict):
            data["metadata"] = self._create_default_open_meteo_metadata()
            modified = True
        else:
            meta = data["metadata"]

            # Required metadata fields
            required_meta = {
                "fetched_at": None,
                "latitude": None,
                "longitude": None,
                "hours_cached": 0,
                "days_cached": 0,
                "mode": "direct_radiation",
            }
            for field, default in required_meta.items():
                if field not in meta:
                    meta[field] = default
                    modified = True

            # Ensure mode is set to direct_radiation (new architecture)
            if meta.get("mode") != "direct_radiation":
                meta["mode"] = "direct_radiation"
                modified = True

        # Validate forecast structure
        if "forecast" not in data or not isinstance(data["forecast"], dict):
            data["forecast"] = {}
            self._log_migration(
                "open_meteo_cache.json: Missing forecast data - will be populated at next update"
            )
            modified = True
        else:
            forecast = data["forecast"]
            for date_str, hours in list(forecast.items()):
                if not isinstance(hours, dict):
                    self._log_migration(
                        f"open_meteo_cache.json: Invalid hours structure for {date_str} - removing"
                    )
                    del forecast[date_str]
                    modified = True
                    continue

                for hour_str, hour_data in list(hours.items()):
                    if not isinstance(hour_data, dict):
                        del hours[hour_str]
                        modified = True
                        continue

                    # Check for required radiation fields (direct radiation model)
                    has_ghi = hour_data.get("ghi") is not None
                    has_direct_diffuse = (
                        hour_data.get("direct_radiation") is not None or
                        hour_data.get("diffuse_radiation") is not None
                    )

                    # Radiation data is REQUIRED in new architecture
                    if not has_ghi and not has_direct_diffuse:
                        # Log warning but don't remove - might be night hours
                        hour_int = int(hour_str) if hour_str.isdigit() else 0
                        if 6 <= hour_int <= 20:  # Daytime hours
                            _LOGGER.debug(
                                f"open_meteo_cache.json: Missing radiation data for "
                                f"{date_str} hour {hour_str} (daytime)"
                            )

        # Check cache freshness - critical for single source
        if data.get("metadata", {}).get("fetched_at"):
            try:
                fetched_at = datetime.fromisoformat(data["metadata"]["fetched_at"])
                age_hours = (datetime.now() - fetched_at).total_seconds() / 3600

                if age_hours > 24:
                    self._log_migration(
                        f"open_meteo_cache.json: CRITICAL - Cache very stale ({age_hours:.1f}h old)! "
                        "Weather data quality may be degraded."
                    )
                elif age_hours > 12:
                    self._log_migration(
                        f"open_meteo_cache.json: Cache stale ({age_hours:.1f}h old) - "
                        "will be refreshed at next scheduled update"
                    )
            except (ValueError, TypeError):
                self._log_migration(
                    "open_meteo_cache.json: Invalid fetched_at timestamp"
                )

        # Log data quality metrics
        forecast = data.get("forecast", {})
        if forecast:
            total_hours = sum(len(hours) for hours in forecast.values())
            total_days = len(forecast)
            _LOGGER.debug(
                f"open_meteo_cache.json: Validated {total_hours} hours across {total_days} days"
            )

        if modified:
            self._log_migration("open_meteo_cache.json: Schema updated (SINGLE SOURCE validated)")
            return await self._write_json(file_path, data)

        return True

    def _create_default_open_meteo_metadata(self) -> Dict[str, Any]:
        """Create default metadata for open_meteo_cache @zara

        IMPORTANT: This structure matches the MASTER JSON in production.
        Do NOT add fields that don't exist in the master!
        """
        return {
            "fetched_at": None,
            "latitude": None,
            "longitude": None,
            "hours_cached": 0,
            "days_cached": 0,
            "mode": "direct_radiation",
        }

    def _create_default_open_meteo_cache(self) -> Dict[str, Any]:
        """Create default open_meteo_cache structure @zara

        IMPORTANT: This is the SINGLE SOURCE for all weather data!
        Open-Meteo provides direct GHI values - no calculations needed.
        """
        return {
            "version": "2.0",
            "metadata": self._create_default_open_meteo_metadata(),
            "forecast": {},
        }

    async def _validate_astronomy_cache(self) -> bool:
        """Validate astronomy_cache.json - Sun position and PV system data @zara"""
        file_path = self.data_dir / "stats" / "astronomy_cache.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration(
                "astronomy_cache.json: Creating new file - will be populated by morning routine"
            )
            data = self._create_default_astronomy_cache()
            return await self._write_json(file_path, data)

        modified = False

        if "version" not in data:
            data["version"] = "1.0"
            modified = True

        if "last_updated" not in data:
            data["last_updated"] = None
            modified = True

        if "location" not in data or not isinstance(data["location"], dict):
            data["location"] = {
                "latitude": None,
                "longitude": None,
                "elevation_m": 0,
                "timezone": "Europe/Berlin",
            }
            modified = True
        else:
            loc = data["location"]
            required_loc = ["latitude", "longitude", "elevation_m", "timezone"]
            for field in required_loc:
                if field not in loc:
                    loc[field] = None if field in ["latitude", "longitude"] else (0 if field == "elevation_m" else "Europe/Berlin")
                    modified = True

        if "pv_system" not in data or not isinstance(data["pv_system"], dict):
            data["pv_system"] = self._create_default_pv_system()
            modified = True
        else:
            pv = data["pv_system"]

            required_pv = {
                "installed_capacity_kwp": None,
                "max_peak_record_kwh": 0.0,
                "max_peak_date": None,
                "max_peak_hour": None,
            }
            for field, default in required_pv.items():
                if field not in pv:
                    pv[field] = default
                    modified = True

            if "hourly_max_peaks" not in pv or not isinstance(pv["hourly_max_peaks"], dict):
                pv["hourly_max_peaks"] = self._create_default_hourly_max_peaks()
                modified = True
            else:
                for hour in range(24):
                    hour_str = str(hour)
                    if hour_str not in pv["hourly_max_peaks"]:
                        pv["hourly_max_peaks"][hour_str] = {
                            "kwh": 0.0,
                            "date": None,
                            "conditions": {},
                        }
                        modified = True

        if "cache_info" not in data or not isinstance(data["cache_info"], dict):
            data["cache_info"] = {
                "total_days": 0,
                "days_back": 31,
                "days_ahead": 7,
                "date_range_start": None,
                "date_range_end": None,
                "success_count": 0,
                "error_count": 0,
            }
            modified = True

        if "days" not in data or not isinstance(data["days"], dict):
            data["days"] = {}
            modified = True

        if modified:
            self._log_migration("astronomy_cache.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_astronomy_cache(self) -> Dict[str, Any]:
        """Create default astronomy_cache structure matching production format @zara"""
        return {
            "version": "1.0",
            "last_updated": None,
            "location": {
                "latitude": None,
                "longitude": None,
                "elevation_m": 0,
                "timezone": "Europe/Berlin",
            },
            "pv_system": self._create_default_pv_system(),
            "cache_info": {
                "total_days": 0,
                "days_back": 31,
                "days_ahead": 7,
                "date_range_start": None,
                "date_range_end": None,
                "success_count": 0,
                "error_count": 0,
            },
            "days": {},
        }

    def _create_default_pv_system(self) -> Dict[str, Any]:
        """Create default pv_system structure for astronomy_cache @zara"""
        return {
            "installed_capacity_kwp": None,
            "max_peak_record_kwh": 0.0,
            "max_peak_date": None,
            "max_peak_hour": None,
            "max_peak_conditions": {
                "sun_elevation_deg": None,
                "cloud_cover_percent": None,
                "temperature_c": None,
                "solar_radiation_wm2": None,
            },
            "hourly_max_peaks": self._create_default_hourly_max_peaks(),
            "all_time_peak_power_kw": 0.0,
            "all_time_peak_date": None,
            "all_time_peak_at": None,
        }

    def _create_default_hourly_max_peaks(self) -> Dict[str, Any]:
        """Create default hourly_max_peaks structure (24 hours) @zara"""
        peaks = {}
        for hour in range(24):
            peaks[str(hour)] = {
                "kwh": 0.0,
                "date": None,
                "conditions": {},
            }
        return peaks

    async def _validate_learned_geometry(self) -> bool:
        """Validate learned_geometry.json - Panel geometry learning data @zara"""
        # New location in physics/ subdirectory (V12.0.0)
        file_path = self.data_dir / "physics" / "learned_geometry.json"
        old_file_path = self.data_dir / "learned_geometry.json"

        # Migrate from old location if needed
        if await self._file_exists(old_file_path) and not await self._file_exists(file_path):
            await self._ensure_directory(file_path.parent)
            await self._migrate_file(old_file_path, file_path)
            self._log_migration(
                "learned_geometry.json: Migrated to physics/ subdirectory"
            )

        data = await self._read_json(file_path)
        if data is None:
            # File doesn't exist - will be created by GeometryLearner when it collects data
            self._log_migration(
                "learned_geometry.json: File missing - will be created by GeometryLearner "
                "(Physics-First Architecture, requires clear-sky data collection)"
            )
            # Create default structure for immediate availability
            await self._ensure_directory(file_path.parent)
            data = self._create_default_learned_geometry()
            return await self._write_json(file_path, data)

        modified = False

        # Validate version
        if "version" not in data:
            data["version"] = "1.0"
            modified = True

        # Validate estimate block
        if "estimate" not in data or not isinstance(data["estimate"], dict):
            data["estimate"] = self._create_default_geometry_estimate()
            modified = True
        else:
            estimate = data["estimate"]
            # Required estimate fields with defaults
            estimate_defaults = {
                "tilt_deg": 30.0,
                "azimuth_deg": 180.0,
                "confidence": 0.0,
                "sample_count": 0,
                "last_updated": None,
                "convergence_history": [],
                "error_metrics": {},
            }
            for field, default in estimate_defaults.items():
                if field not in estimate:
                    estimate[field] = default
                    modified = True

            # Validate ranges
            if not (0 <= estimate.get("tilt_deg", 30) <= 90):
                estimate["tilt_deg"] = max(0, min(90, estimate.get("tilt_deg", 30)))
                modified = True
            if not (0 <= estimate.get("azimuth_deg", 180) <= 360):
                estimate["azimuth_deg"] = estimate.get("azimuth_deg", 180) % 360
                modified = True
            if not (0 <= estimate.get("confidence", 0) <= 1):
                estimate["confidence"] = max(0, min(1, estimate.get("confidence", 0)))
                modified = True

        # Validate data_points (optional - can be empty)
        if "data_points" not in data:
            data["data_points"] = []
            modified = True
        elif not isinstance(data["data_points"], list):
            data["data_points"] = []
            modified = True

        # Validate metadata
        if "metadata" not in data or not isinstance(data["metadata"], dict):
            data["metadata"] = {
                "system_capacity_kwp": None,
                "saved_at": None,
            }
            modified = True

        if modified:
            self._log_migration("learned_geometry.json: Schema updated (Physics-First Architecture)")
            return await self._write_json(file_path, data)

        return True

    def _create_default_learned_geometry(self) -> Dict[str, Any]:
        """Create default learned_geometry structure for Physics-First Architecture @zara

        IMPORTANT: This structure matches the MASTER JSON in production.
        Do NOT add fields that don't exist in the master!
        """
        return {
            "version": "1.0",
            "estimate": self._create_default_geometry_estimate(),
            "data_points": [],
            "metadata": {
                "system_capacity_kwp": None,
                "saved_at": None,
            },
        }

    def _create_default_geometry_estimate(self) -> Dict[str, Any]:
        """Create default geometry estimate block @zara"""
        return {
            "tilt_deg": 30.0,  # Default: 30° tilt (common for Central Europe)
            "azimuth_deg": 180.0,  # Default: South-facing
            "confidence": 0.0,  # No confidence until learning
            "sample_count": 0,
            "last_updated": None,
            "convergence_history": [],
            "error_metrics": {},
        }

    async def _validate_learned_panel_group_efficiency(self) -> bool:
        """Validate learned_panel_group_efficiency.json - Panel group efficiency learning data @zara

        This file is only relevant when panel_groups are configured.
        If the file doesn't exist and there are no panel groups, that's fine.
        """
        # New location in physics/ subdirectory (V12.0.0)
        file_path = self.data_dir / "physics" / "learned_panel_group_efficiency.json"
        old_file_path = self.data_dir / "learned_panel_group_efficiency.json"

        # Migrate from old location if needed
        if await self._file_exists(old_file_path) and not await self._file_exists(file_path):
            await self._ensure_directory(file_path.parent)
            await self._migrate_file(old_file_path, file_path)
            self._log_migration(
                "learned_panel_group_efficiency.json: Migrated to physics/ subdirectory"
            )

        data = await self._read_json(file_path)
        if data is None:
            # File doesn't exist - will be created by PanelGroupEfficiencyLearner
            # when panel groups are configured and bootstrap runs
            self._log_migration(
                "learned_panel_group_efficiency.json: File missing - will be created by "
                "PanelGroupEfficiencyLearner when panel groups are configured"
            )
            return True  # Not an error - file is optional

        modified = False

        # Validate version
        if data.get("version") != "2.0":
            data["version"] = "2.0"
            modified = True

        # Validate mode
        if data.get("mode") != "panel_groups":
            data["mode"] = "panel_groups"
            modified = True

        # Validate panel_groups array
        if "panel_groups" not in data or not isinstance(data["panel_groups"], list):
            data["panel_groups"] = []
            modified = True
        else:
            # Validate each panel group entry
            for group in data["panel_groups"]:
                if not isinstance(group, dict):
                    continue

                # Required fields with defaults
                group_defaults = {
                    "name": "Unknown",
                    "configured_tilt_deg": 30.0,
                    "configured_azimuth_deg": 180.0,
                    "power_kwp": 0.0,
                    "learned_efficiency_factor": 1.0,
                    "learned_shadow_hours": [],
                    "sample_count": 0,
                    "confidence": 0.0,
                    "hourly_efficiency": {},
                }
                for field, default in group_defaults.items():
                    if field not in group:
                        group[field] = default
                        modified = True

                # Validate ranges
                if not (0 <= group.get("configured_tilt_deg", 30) <= 90):
                    group["configured_tilt_deg"] = max(0, min(90, group.get("configured_tilt_deg", 30)))
                    modified = True
                if not (0 <= group.get("configured_azimuth_deg", 180) <= 360):
                    group["configured_azimuth_deg"] = group.get("configured_azimuth_deg", 180) % 360
                    modified = True
                if not (0 <= group.get("confidence", 0) <= 1):
                    group["confidence"] = max(0, min(1, group.get("confidence", 0)))
                    modified = True
                if not (0.1 <= group.get("learned_efficiency_factor", 1.0) <= 2.0):
                    group["learned_efficiency_factor"] = max(0.1, min(2.0, group.get("learned_efficiency_factor", 1.0)))
                    modified = True

        # Validate data_points (optional - keep last 500)
        if "data_points" not in data:
            data["data_points"] = []
            modified = True
        elif not isinstance(data["data_points"], list):
            data["data_points"] = []
            modified = True
        elif len(data["data_points"]) > 500:
            data["data_points"] = data["data_points"][-500:]
            modified = True

        # Validate total_samples
        if "total_samples" not in data or not isinstance(data["total_samples"], int):
            data["total_samples"] = len(data.get("data_points", []))
            modified = True

        # Validate last_updated
        if "last_updated" not in data:
            data["last_updated"] = None
            modified = True

        # Validate metadata
        if "metadata" not in data or not isinstance(data["metadata"], dict):
            data["metadata"] = {
                "total_capacity_kwp": 0.0,
                "group_count": len(data.get("panel_groups", [])),
            }
            modified = True

        if modified:
            self._log_migration("learned_panel_group_efficiency.json: Schema updated (Panel-Gruppen V12.0.0)")
            return await self._write_json(file_path, data)

        return True

    async def _validate_residual_model_state(self) -> bool:
        """Validate residual_model_state.json - ML Residual Trainer state @zara"""
        file_path = self.data_dir / "ml" / "residual_model_state.json"

        data = await self._read_json(file_path)
        if data is None:
            # File doesn't exist - will be created by ResidualTrainer during END_OF_DAY
            self._log_migration(
                "residual_model_state.json: File missing - will be created by ResidualTrainer "
                "(Physics-First Architecture, trains at END_OF_DAY Step 9/9)"
            )
            # Create default structure for immediate availability
            data = self._create_default_residual_model_state()
            return await self._write_json(file_path, data)

        modified = False

        # Validate version
        if "version" not in data:
            data["version"] = "1.0"
            modified = True

        # Validate residual_stats block
        if "residual_stats" not in data or not isinstance(data["residual_stats"], dict):
            data["residual_stats"] = self._create_default_residual_stats()
            modified = True
        else:
            stats = data["residual_stats"]
            # Required fields with defaults
            stats_defaults = {
                "mean": 0.0,
                "std": 1.0,
                "min": -1.0,
                "max": 1.0,
                "sample_count": 0,
            }
            for field, default in stats_defaults.items():
                if field not in stats:
                    stats[field] = default
                    modified = True

        # Validate weights (can be empty dict)
        if "weights" not in data:
            data["weights"] = {}
            modified = True
        elif not isinstance(data["weights"], dict):
            data["weights"] = {}
            modified = True

        # Validate model_type
        valid_model_types = ["ridge", "tiny_lstm", "none"]
        if data.get("model_type") not in valid_model_types:
            data["model_type"] = "ridge"
            modified = True

        # Validate optional metadata fields
        if "saved_at" not in data:
            data["saved_at"] = None
            modified = True

        if "system_capacity_kwp" not in data:
            data["system_capacity_kwp"] = None
            modified = True

        if modified:
            self._log_migration("residual_model_state.json: Schema updated (Physics-First Architecture)")
            return await self._write_json(file_path, data)

        return True

    def _create_default_residual_model_state(self) -> Dict[str, Any]:
        """Create default residual_model_state structure for Physics-First Architecture @zara

        IMPORTANT: This structure matches the MASTER JSON in production.
        Do NOT add fields that don't exist in the master!
        """
        return {
            "version": "1.0",
            "residual_stats": self._create_default_residual_stats(),
            "weights": {},
            "model_type": "ridge",
            "saved_at": None,
            "system_capacity_kwp": None,
        }

    def _create_default_residual_stats(self) -> Dict[str, Any]:
        """Create default residual_stats block @zara"""
        return {
            "mean": 0.0,
            "std": 1.0,
            "min": -1.0,
            "max": 1.0,
            "sample_count": 0,
        }

    async def _validate_yield_cache(self) -> bool:
        """Validate yield_cache.json - Persistent cache for hourly yield tracking @zara

        Location: data_dir/stats/yield_cache.json
        """
        file_path = self.data_dir / "stats" / "yield_cache.json"

        data = await self._read_json(file_path)
        if data is None:
            # File doesn't exist - create empty template
            # Will be populated by update_hourly_actuals() during production hours
            self._log_migration(
                "yield_cache.json: Creating new file - "
                "will be populated during production hours"
            )
            data = self._create_default_yield_cache()
            return await self._write_json(file_path, data)

        modified = False

        # Validate required fields
        required_fields = {
            "value": None,
            "time": None,
            "date": None,
        }

        for field, default in required_fields.items():
            if field not in data:
                data[field] = default
                modified = True

        # Validate value is numeric or None
        if data.get("value") is not None:
            try:
                data["value"] = float(data["value"])
                if data["value"] < 0:
                    self._log_migration(
                        f"yield_cache.json: Clamping negative value {data['value']} to 0"
                    )
                    data["value"] = 0.0
                    modified = True
            except (ValueError, TypeError):
                self._log_migration(
                    f"yield_cache.json: Invalid value '{data['value']}' - resetting to None"
                )
                data["value"] = None
                modified = True

        # Validate date format (YYYY-MM-DD)
        if data.get("date") is not None:
            try:
                datetime.strptime(data["date"], "%Y-%m-%d")
            except ValueError:
                self._log_migration(
                    f"yield_cache.json: Invalid date format '{data['date']}' - resetting"
                )
                data["date"] = None
                data["value"] = None  # Invalidate value too
                modified = True

        # Check if cache is stale (from a different day)
        if data.get("date"):
            today = dt_util.now().date().isoformat()
            if data["date"] != today:
                self._log_migration(
                    f"yield_cache.json: Stale cache from {data['date']} (today is {today}) - "
                    "will be refreshed at next hourly update"
                )
                # Don't reset here - let update_hourly_actuals handle it
                # This allows for proper day-transition handling

        if modified:
            self._log_migration("yield_cache.json: Schema validated and updated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_yield_cache(self) -> Dict[str, Any]:
        """Create default yield_cache structure @zara

        This cache stores the last known cumulative yield value
        to enable correct delta calculation after HA restarts.
        """
        return {
            "value": None,
            "time": None,
            "date": None,
        }

    async def _validate_wttr_in_cache(self) -> bool:
        """Validate wttr_in_cache.json - wttr.in API cache for Multi-Weather Blending @zara

        Location: data_dir/data/wttr_in_cache.json
        """
        file_path = self.data_dir / "data" / "wttr_in_cache.json"

        data = await self._read_json(file_path)
        if data is None:
            # File doesn't exist - create empty template
            # Will be populated by MultiWeatherBlender on first weather fetch
            self._log_migration(
                "wttr_in_cache.json: Creating empty template - "
                "will be populated when Multi-Weather Blending triggers"
            )
            data = self._create_default_wttr_in_cache()
            return await self._write_json(file_path, data)

        modified = False

        # Validate version
        if "version" not in data:
            data["version"] = "1.0"
            modified = True

        # Validate metadata
        if "metadata" not in data or not isinstance(data["metadata"], dict):
            data["metadata"] = {
                "fetched_at": None,
                "source": "wttr.in",
                "latitude": None,
                "longitude": None,
                "cache_max_age_hours": 6,
                "created_empty": True,
            }
            modified = True
        else:
            meta = data["metadata"]
            required_meta = {
                "fetched_at": None,
                "source": "wttr.in",
            }
            for field, default in required_meta.items():
                if field not in meta:
                    meta[field] = default
                    modified = True

        # Validate forecast structure
        if "forecast" not in data or not isinstance(data["forecast"], dict):
            data["forecast"] = {}
            modified = True

        if modified:
            self._log_migration("wttr_in_cache.json: Schema validated and updated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_wttr_in_cache(self) -> Dict[str, Any]:
        """Create default wttr_in_cache structure for Multi-Weather Blending @zara"""
        return {
            "version": "1.0",
            "metadata": {
                "fetched_at": None,
                "source": "wttr.in",
                "latitude": None,
                "longitude": None,
                "cache_max_age_hours": 6,
                "created_empty": True,
            },
            "forecast": {},
        }

    async def _validate_weather_source_weights(self) -> bool:
        """Validate weather_source_weights.json - Multi-Weather Blending weights @zara

        CRITICAL: This file enables weight learning for weather source blending.
        Without it, the WeatherSourceLearner cannot operate.

        Location: data_dir/data/weather_source_weights.json
        """
        file_path = self.data_dir / "data" / "weather_source_weights.json"

        data = await self._read_json(file_path)
        if data is None:
            # CRITICAL: Create this file for Multi-Weather Blending to work
            self._log_migration(
                "weather_source_weights.json: CRITICAL - Creating file with default weights "
                "(Required for Multi-Weather Blending weight learning)"
            )
            data = self._create_default_weather_source_weights()
            return await self._write_json(file_path, data)

        modified = False

        # Validate version
        if "version" not in data:
            data["version"] = "1.0"
            modified = True

        # Validate weights structure
        if "weights" not in data or not isinstance(data["weights"], dict):
            data["weights"] = {
                "open_meteo": 0.5,
                "wwo": 0.5,
            }
            modified = True
        else:
            weights = data["weights"]
            # Ensure default weights exist
            if "open_meteo" not in weights:
                weights["open_meteo"] = 0.5
                modified = True
            if "wwo" not in weights:
                weights["wwo"] = 0.5
                modified = True

            # Validate weights are numeric and in valid range
            for source in ["open_meteo", "wwo"]:
                try:
                    val = float(weights[source])
                    if not (0.0 <= val <= 1.0):
                        weights[source] = max(0.0, min(1.0, val))
                        modified = True
                except (ValueError, TypeError):
                    weights[source] = 0.5
                    modified = True

        # Validate learning_metadata
        if "learning_metadata" not in data or not isinstance(data["learning_metadata"], dict):
            data["learning_metadata"] = {
                "last_updated": None,
                "learning_days": 0,
                "created_at": dt_util.now().isoformat(),
                "source": "schema_validator_default",
            }
            modified = True

        if modified:
            self._log_migration("weather_source_weights.json: Schema validated and updated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_weather_source_weights(self) -> Dict[str, Any]:
        """Create default weather_source_weights structure for Multi-Weather Blending @zara

        IMPORTANT: These are the initial weights before any learning occurs.
        50/50 split gives equal weight to both sources until the system learns.
        """
        return {
            "version": "1.0",
            "weights": {
                "open_meteo": 0.5,
                "wwo": 0.5,
            },
            "learning_metadata": {
                "last_updated": None,
                "learning_days": 0,
                "created_at": dt_util.now().isoformat(),
                "source": "schema_validator_default",
            },
        }

    async def _validate_weather_source_learning(self) -> bool:
        """Validate weather_source_learning.json - Learning history for weight optimization @zara

        Location: data_dir/stats/weather_source_learning.json
        """
        file_path = self.data_dir / "stats" / "weather_source_learning.json"

        data = await self._read_json(file_path)
        if data is None:
            # File doesn't exist - create empty template
            # Will be populated by WeatherSourceLearner at 23:30 daily
            self._log_migration(
                "weather_source_learning.json: Creating empty template - "
                "will be populated by WeatherSourceLearner at 23:30 daily"
            )
            data = self._create_default_weather_source_learning()
            return await self._write_json(file_path, data)

        modified = False

        # Validate version
        if "version" not in data:
            data["version"] = "1.0"
            modified = True

        # Validate daily_history structure
        if "daily_history" not in data or not isinstance(data["daily_history"], dict):
            data["daily_history"] = {}
            modified = True

        # Validate metadata
        if "metadata" not in data or not isinstance(data["metadata"], dict):
            data["metadata"] = {
                "total_learning_days": 0,
                "last_learning_run": None,
                "created_at": dt_util.now().isoformat(),
            }
            modified = True

        # Keep only last 30 days of history
        if len(data.get("daily_history", {})) > 30:
            sorted_dates = sorted(data["daily_history"].keys())
            dates_to_remove = sorted_dates[:-30]
            for old_date in dates_to_remove:
                del data["daily_history"][old_date]
            modified = True
            self._log_migration(
                f"weather_source_learning.json: Trimmed history to last 30 days "
                f"(removed {len(dates_to_remove)} old entries)"
            )

        if modified:
            self._log_migration("weather_source_learning.json: Schema validated and updated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_weather_source_learning(self) -> Dict[str, Any]:
        """Create default weather_source_learning structure @zara"""
        return {
            "version": "1.0",
            "daily_history": {},
            "metadata": {
                "total_learning_days": 0,
                "last_learning_run": None,
                "created_at": dt_util.now().isoformat(),
            },
        }

    async def _validate_panel_group_sensor_state(self) -> bool:
        """Validate panel_group_sensor_state.json - State persistence for panel group energy sensors @zara

        This file persists the last-known kWh values for panel group energy sensors.
        Used for delta calculation across HA restarts.

        Location: data_dir/stats/panel_group_sensor_state.json
        """
        file_path = self.data_dir / "stats" / "panel_group_sensor_state.json"

        data = await self._read_json(file_path)
        if data is None:
            # File doesn't exist - create empty template
            # Will be populated by PanelGroupSensorReader when panel groups with energy sensors are configured
            self._log_migration(
                "panel_group_sensor_state.json: Creating empty template - "
                "will be populated when panel groups with energy sensors are used"
            )
            data = self._create_default_panel_group_sensor_state()
            return await self._write_json(file_path, data)

        modified = False

        # Validate last_updated field
        if "last_updated" not in data:
            data["last_updated"] = None
            modified = True

        # Validate last_values structure
        if "last_values" not in data or not isinstance(data["last_values"], dict):
            data["last_values"] = {}
            modified = True
        else:
            # Validate all values are numeric
            for group_name, value in list(data["last_values"].items()):
                if value is not None:
                    try:
                        float_val = float(value)
                        if float_val < 0:
                            self._log_migration(
                                f"panel_group_sensor_state.json: Clamping negative value for '{group_name}' to 0"
                            )
                            data["last_values"][group_name] = 0.0
                            modified = True
                    except (ValueError, TypeError):
                        self._log_migration(
                            f"panel_group_sensor_state.json: Invalid value for '{group_name}' - removing"
                        )
                        del data["last_values"][group_name]
                        modified = True

        if modified:
            self._log_migration("panel_group_sensor_state.json: Schema validated and updated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_panel_group_sensor_state(self) -> Dict[str, Any]:
        """Create default panel_group_sensor_state structure @zara"""
        return {
            "last_updated": None,
            "last_values": {},
        }
