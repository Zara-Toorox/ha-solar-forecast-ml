"""
JSON Schema Validation and Migration for Solar Forecast ML Integration

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
import math
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from homeassistant.core import HomeAssistant

from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..const import (
    DATA_VERSION, ML_MODEL_VERSION, CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX,
    MAX_PREDICTION_HISTORY
)

_LOGGER = logging.getLogger(__name__)


class DataSchemaValidator:
    """Validates and migrates JSON files to ensure they match code expectations"""

    def __init__(self, hass: HomeAssistant, data_dir: Path):
        """Initialize the schema validator"""
        self.hass = hass
        self.data_dir = data_dir
        self.migration_log = []

    async def validate_and_migrate_all(self) -> bool:
        """Validate and migrate all JSON files on startup"""
        try:
            _LOGGER.info("=== Starting JSON Schema Validation and Migration ===")

            # Validate/migrate each file in order of priority
            success = True

            # HIGH PRIORITY - Core ML files
            success &= await self._validate_learned_weights()
            success &= await self._validate_hourly_profile()
            success &= await self._validate_model_state()
            success &= await self._validate_hourly_samples()

            # HIGH PRIORITY - Central forecast file
            success &= await self._validate_daily_forecasts()

            # MEDIUM PRIORITY - History and state
            success &= await self._validate_prediction_history()
            success &= await self._validate_coordinator_state()

            # Report migration results
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
        """Log a migration action"""
        self.migration_log.append(message)
        _LOGGER.info(f"MIGRATION: {message}")

    async def _read_json(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Read JSON file async"""
        try:
            import aiofiles
            import json

            if not file_path.exists():
                return None

            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            _LOGGER.error(f"Failed to read {file_path.name}: {e}")
            return None

    async def _write_json(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Write JSON file async atomically"""
        try:
            import aiofiles
            import json

            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first
            temp_file = file_path.with_suffix('.tmp')
            async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))

            # Atomic rename
            await self.hass.async_add_executor_job(temp_file.replace, file_path)
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to write {file_path.name}: {e}")
            return False

    # =================================================================
    # LEARNED WEIGHTS VALIDATION
    # =================================================================

    async def _validate_learned_weights(self) -> bool:
        """Validate learned_weights json"""
        file_path = self.data_dir / "ml" / "learned_weights.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("learned_weights.json: Creating new file with default structure")
            data = self._create_default_learned_weights()
            return await self._write_json(file_path, data)

        modified = False

        # Validate 27 feature names
        expected_features = self._get_expected_feature_names()
        if not data.get("feature_names") or data["feature_names"] != expected_features:
            self._log_migration(f"learned_weights.json: Updating feature_names to 27 features")
            data["feature_names"] = expected_features
            modified = True

        # Ensure all required fields exist
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

        # Validate accuracy range
        if "accuracy" not in data or not (0.0 <= data["accuracy"] <= 1.0):
            if "accuracy" in data:
                self._log_migration(f"learned_weights.json: Clamping accuracy from {data['accuracy']} to valid range")
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

        # Validate correction_factor range
        cf = data.get("correction_factor", 1.0)
        if not (CORRECTION_FACTOR_MIN <= cf <= CORRECTION_FACTOR_MAX):
            self._log_migration(f"learned_weights.json: Clamping correction_factor from {cf} to valid range")
            data["correction_factor"] = max(CORRECTION_FACTOR_MIN, min(CORRECTION_FACTOR_MAX, cf))
            modified = True

        # Ensure deprecated fields exist for backward compatibility
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
        """Get the 27 expected feature names"""
        features = [
            # Base features (18)
            "temperature", "humidity", "cloudiness", "wind_speed",
            "hour_of_day", "seasonal_factor", "weather_trend",
            "production_yesterday", "production_same_hour_yesterday",
            "cloudiness_primary", "cloud_impact", "sunshine_factor",
            "rain", "uv_index", "lux",
            "cloudiness_trend_1h", "cloudiness_trend_3h", "cloudiness_volatility",
            # Polynomial features (4)
            "temperature_sq", "cloudiness_sq", "hour_of_day_sq", "seasonal_factor_sq",
            # Interaction features (5)
            "cloudiness_x_hour", "temperature_x_seasonal", "humidity_x_cloudiness",
            "wind_x_hour", "weather_trend_x_seasonal"
        ]
        return features

    def _create_default_learned_weights(self) -> Dict[str, Any]:
        """Create default learned weights structure"""
        return {
            "weights": {},
            "bias": 0.0,
            "feature_names": self._get_expected_feature_names(),
            "feature_means": {},
            "feature_stds": {},
            "accuracy": 0.0,
            "training_samples": 0,
            "last_trained": dt_util.now().isoformat(),
            "model_version": ML_MODEL_VERSION,
            "correction_factor": 1.0,
            "weather_weights": {},
            "seasonal_factors": {},
            "feature_importance": {}
        }

    # =================================================================
    # HOURLY PROFILE VALIDATION
    # =================================================================

    async def _validate_hourly_profile(self) -> bool:
        """Validate hourly_profile json"""
        file_path = self.data_dir / "ml" / "hourly_profile.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("hourly_profile.json: Creating new file with sine curve default")
            data = self._create_default_hourly_profile()
            return await self._write_json(file_path, data)

        modified = False

        # Ensure hourly_averages exists and has all 24 hours
        if "hourly_averages" not in data or not isinstance(data["hourly_averages"], dict):
            data["hourly_averages"] = {}
            modified = True

        for hour in range(24):
            hour_str = str(hour)
            if hour_str not in data["hourly_averages"]:
                # Use sine curve for missing hours
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

        # Validate confidence range
        if "confidence" not in data or not (0.0 <= data["confidence"] <= 1.0):
            data["confidence"] = 0.1
            modified = True

        # Deprecated fields
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
        """Calculate sine curve value for an hour"""
        start_hour = 6
        daylight_hours = 12

        if start_hour <= hour < start_hour + daylight_hours:
            relative_hour = (hour - start_hour) / daylight_hours
            sine_value = math.sin(relative_hour * math.pi + 0.01)
            return max(0.0, sine_value * 1.0)
        return 0.0

    def _create_default_hourly_profile(self) -> Dict[str, Any]:
        """Create default hourly profile structure"""
        hourly_averages = {}
        for hour in range(24):
            hourly_averages[str(hour)] = self._calculate_sine_hour(hour)

        return {
            "hourly_averages": hourly_averages,
            "samples_count": 0,
            "last_updated": dt_util.now().isoformat(),
            "confidence": 0.1,
            "hourly_factors": {},
            "seasonal_adjustment": {}
        }

    # =================================================================
    # MODEL STATE VALIDATION
    # =================================================================

    async def _validate_model_state(self) -> bool:
        """Validate model_state json"""
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

        if "current_accuracy" not in data or not (0.0 <= data["current_accuracy"] <= 1.0):
            data["current_accuracy"] = 0.0
            modified = True

        if "status" not in data or data["status"] not in ["uninitialized", "ready", "training", "error"]:
            data["status"] = "uninitialized"
            modified = True

        if modified:
            self._log_migration("model_state.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_model_state(self) -> Dict[str, Any]:
        """Create default model state structure"""
        return {
            "version": DATA_VERSION,
            "model_loaded": False,
            "last_training": None,
            "training_samples": 0,
            "current_accuracy": 0.0,
            "status": "uninitialized"
        }

    # =================================================================
    # HOURLY SAMPLES VALIDATION
    # =================================================================

    async def _validate_hourly_samples(self) -> bool:
        """Validate hourly_samples json"""
        file_path = self.data_dir / "ml" / "hourly_samples.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("hourly_samples.json: Creating new file")
            data = self._create_default_hourly_samples()
            return await self._write_json(file_path, data)

        modified = False

        if "version" not in data:
            data["version"] = DATA_VERSION
            modified = True

        if "samples" not in data or not isinstance(data["samples"], list):
            data["samples"] = []
            modified = True

        # Validate sample count matches array length
        actual_count = len(data["samples"])
        if "count" not in data or data["count"] != actual_count:
            data["count"] = actual_count
            modified = True

        if "last_updated" not in data:
            data["last_updated"] = None
            modified = True

        # Enforce 10000 sample limit
        if len(data["samples"]) > 10000:
            self._log_migration(f"hourly_samples.json: Trimming {len(data['samples']) - 10000} old samples")
            data["samples"] = data["samples"][-10000:]
            data["count"] = len(data["samples"])
            modified = True

        if modified:
            self._log_migration("hourly_samples.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    def _create_default_hourly_samples(self) -> Dict[str, Any]:
        """Create default hourly samples structure"""
        return {
            "version": DATA_VERSION,
            "samples": [],
            "count": 0,
            "last_updated": None
        }

    # =================================================================
    # DAILY FORECASTS VALIDATION (MOST COMPLEX)
    # =================================================================

    async def _validate_daily_forecasts(self) -> bool:
        """Validate daily_forecasts json"""
        file_path = self.data_dir / "stats" / "daily_forecasts.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("daily_forecasts.json: Creating new file with full structure")
            data = self._create_default_daily_forecasts()
            return await self._write_json(file_path, data)

        modified = False

        # Check for old structure and migrate
        if "current_day" in data or "forecasts" in data:
            self._log_migration("daily_forecasts.json: Migrating from OLD structure to NEW structure")
            data = self._create_default_daily_forecasts()
            modified = True

        # Validate version
        if "version" not in data:
            data["version"] = DATA_VERSION
            modified = True

        # Validate today block
        if "today" not in data or not isinstance(data["today"], dict):
            data["today"] = self._create_default_today_block()
            modified = True
        else:
            if await self._validate_today_block(data["today"]):
                modified = True

        # Validate statistics block
        if "statistics" not in data or not isinstance(data["statistics"], dict):
            data["statistics"] = self._create_default_statistics_block()
            modified = True
        else:
            if await self._validate_statistics_block(data["statistics"]):
                modified = True

        # Validate history
        if "history" not in data or not isinstance(data["history"], list):
            data["history"] = []
            modified = True

        # Validate metadata
        if "metadata" not in data or not isinstance(data["metadata"], dict):
            data["metadata"] = {
                "retention_days": 730,
                "history_entries": len(data.get("history", [])),
                "last_update": None
            }
            modified = True

        if modified:
            self._log_migration("daily_forecasts.json: Schema updated and validated")
            return await self._write_json(file_path, data)

        return True

    async def _validate_today_block(self, today: Dict[str, Any]) -> bool:
        """Validate today block structure"""
        modified = False

        if "date" not in today:
            today["date"] = None
            modified = True

        # Validate all sub-blocks
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
            ("finalized", self._create_default_finalized)
        ]

        for block_name, create_func in sub_blocks:
            if block_name not in today or not isinstance(today[block_name], dict):
                today[block_name] = create_func()
                modified = True

        return modified

    async def _validate_statistics_block(self, stats: Dict[str, Any]) -> bool:
        """Validate statistics block structure"""
        modified = False

        stat_blocks = [
            ("all_time_peak", self._create_default_all_time_peak),
            ("current_week", self._create_default_current_week),
            ("current_month", self._create_default_current_month),
            ("last_7_days", self._create_default_last_7_days),
            ("last_30_days", self._create_default_last_30_days),
            ("last_365_days", self._create_default_last_365_days)
        ]

        for block_name, create_func in stat_blocks:
            if block_name not in stats or not isinstance(stats[block_name], dict):
                stats[block_name] = create_func()
                modified = True

        return modified

    def _create_default_daily_forecasts(self) -> Dict[str, Any]:
        """Create complete default daily_forecasts structure"""
        return {
            "version": DATA_VERSION,
            "today": self._create_default_today_block(),
            "statistics": self._create_default_statistics_block(),
            "history": [],
            "metadata": {
                "retention_days": 730,
                "history_entries": 0,
                "last_update": None
            }
        }

    def _create_default_today_block(self) -> Dict[str, Any]:
        """Create default today block"""
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
            "finalized": self._create_default_finalized()
        }

    def _create_default_forecast_day(self) -> Dict[str, Any]:
        """Create default forecast_day block"""
        return {
            "prediction_kwh": None,
            "locked": False,
            "locked_at": None,
            "source": None
        }

    def _create_default_forecast_tomorrow(self) -> Dict[str, Any]:
        """Create default forecast_tomorrow block"""
        return {
            "date": None,
            "prediction_kwh": None,
            "locked": False,
            "locked_at": None,
            "source": None,
            "updates": []
        }

    def _create_default_forecast_day_after(self) -> Dict[str, Any]:
        """Create default forecast_day_after_tomorrow block"""
        return {
            "date": None,
            "prediction_kwh": None,
            "locked": False,
            "next_update": None,
            "source": None,
            "updates": []
        }

    def _create_default_forecast_best_hour(self) -> Dict[str, Any]:
        """Create default forecast_best_hour block"""
        return {
            "hour": None,
            "prediction_kwh": None,
            "locked": False,
            "locked_at": None,
            "source": None
        }

    def _create_default_actual_best_hour(self) -> Dict[str, Any]:
        """Create default actual_best_hour block"""
        return {
            "hour": None,
            "actual_kwh": None,
            "saved_at": None
        }

    def _create_default_forecast_next_hour(self) -> Dict[str, Any]:
        """Create default forecast_next_hour block"""
        return {
            "period": None,
            "prediction_kwh": None,
            "updated_at": None,
            "source": None
        }

    def _create_default_production_time(self) -> Dict[str, Any]:
        """Create default production_time block"""
        return {
            "active": False,
            "duration_seconds": 0,
            "start_time": None,
            "end_time": None,
            "last_power_above_10w": None,
            "zero_power_since": None
        }

    def _create_default_peak_today(self) -> Dict[str, Any]:
        """Create default peak_today block"""
        return {
            "power_w": 0.0,
            "at": None
        }

    def _create_default_yield_today(self) -> Dict[str, Any]:
        """Create default yield_today block"""
        return {
            "kwh": None,
            "sensor": None
        }

    def _create_default_consumption_today(self) -> Dict[str, Any]:
        """Create default consumption_today block"""
        return {
            "kwh": None,
            "sensor": None
        }

    def _create_default_autarky(self) -> Dict[str, Any]:
        """Create default autarky block"""
        return {
            "percent": None,
            "calculated_at": None
        }

    def _create_default_finalized(self) -> Dict[str, Any]:
        """Create default finalized block"""
        return {
            "yield_kwh": None,
            "consumption_kwh": None,
            "production_hours": None,
            "accuracy_percent": None,
            "at": None
        }

    def _create_default_statistics_block(self) -> Dict[str, Any]:
        """Create default statistics block"""
        return {
            "all_time_peak": self._create_default_all_time_peak(),
            "current_week": self._create_default_current_week(),
            "current_month": self._create_default_current_month(),
            "last_7_days": self._create_default_last_7_days(),
            "last_30_days": self._create_default_last_30_days(),
            "last_365_days": self._create_default_last_365_days()
        }

    def _create_default_all_time_peak(self) -> Dict[str, Any]:
        """Create default all_time_peak block"""
        return {
            "power_w": 0.0,
            "date": None,
            "at": None
        }

    def _create_default_current_week(self) -> Dict[str, Any]:
        """Create default current_week block"""
        return {
            "period": None,
            "date_range": None,
            "yield_kwh": 0.0,
            "consumption_kwh": 0.0,
            "days": 0,
            "updated_at": None
        }

    def _create_default_current_month(self) -> Dict[str, Any]:
        """Create default current_month block"""
        return {
            "period": None,
            "yield_kwh": 0.0,
            "consumption_kwh": 0.0,
            "avg_autarky": 0.0,
            "days": 0,
            "updated_at": None
        }

    def _create_default_last_7_days(self) -> Dict[str, Any]:
        """Create default last_7_days block"""
        return {
            "avg_yield_kwh": 0.0,
            "avg_accuracy": 0.0,
            "total_yield_kwh": 0.0,
            "calculated_at": None
        }

    def _create_default_last_30_days(self) -> Dict[str, Any]:
        """Create default last_30_days block"""
        return {
            "avg_yield_kwh": 0.0,
            "avg_accuracy": 0.0,
            "total_yield_kwh": 0.0,
            "calculated_at": None
        }

    def _create_default_last_365_days(self) -> Dict[str, Any]:
        """Create default last_365_days block"""
        return {
            "avg_yield_kwh": 0.0,
            "total_yield_kwh": 0.0,
            "calculated_at": None
        }

    # =================================================================
    # PREDICTION HISTORY VALIDATION
    # =================================================================

    async def _validate_prediction_history(self) -> bool:
        """Validate prediction_history json"""
        file_path = self.data_dir / "stats" / "prediction_history.json"
        old_file_path = self.data_dir / "data" / "prediction_history.json"

        data = await self._read_json(file_path)
        if data is None:
            self._log_migration("prediction_history.json: Creating new file")
            data = self._create_default_prediction_history()
            return await self._write_json(file_path, data)

        modified = False

        if "version" not in data:
            data["version"] = DATA_VERSION
            modified = True

        if "predictions" not in data or not isinstance(data["predictions"], list):
            data["predictions"] = []
            modified = True

        if "last_updated" not in data:
            data["last_updated"] = None
            modified = True

        # Enforce MAX_PREDICTION_HISTORY limit
        if len(data["predictions"]) > MAX_PREDICTION_HISTORY:
            self._log_migration(f"prediction_history.json: Trimming {len(data['predictions']) - MAX_PREDICTION_HISTORY} old predictions")
            data["predictions"] = data["predictions"][-MAX_PREDICTION_HISTORY:]
            modified = True

        if modified:
            self._log_migration("prediction_history.json: Schema updated and validated")
            await self._write_json(file_path, data)

        # CLEANUP: Remove old prediction_history.json from data/ directory if it exists
        # This file is a relict from before the stats/ subdirectory migration
        if old_file_path.exists():
            try:
                # Safety check: Only remove if new file has data
                if data.get("predictions") and len(data["predictions"]) > 0:
                    self._log_migration(
                        f"Removing obsolete data/prediction_history.json "
                        f"(backup exists in backups/pre_migration/)"
                    )
                    await self.hass.async_add_executor_job(old_file_path.unlink)
                else:
                    _LOGGER.warning(
                        "Old data/prediction_history.json exists but new file is empty - "
                        "keeping old file for safety. Manual review recommended."
                    )
            except Exception as e:
                _LOGGER.warning(f"Failed to remove old prediction_history.json: {e}")

        return True

    def _create_default_prediction_history(self) -> Dict[str, Any]:
        """Create default prediction_history structure"""
        return {
            "version": DATA_VERSION,
            "predictions": [],
            "last_updated": None
        }

    # =================================================================
    # COORDINATOR STATE VALIDATION
    # =================================================================

    async def _validate_coordinator_state(self) -> bool:
        """Validate coordinator_state json"""
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
        """Create default coordinator_state structure"""
        return {
            "version": DATA_VERSION,
            "expected_daily_production": None,
            "last_set_date": None,
            "last_updated": None,
            "last_collected_hour": None
        }
