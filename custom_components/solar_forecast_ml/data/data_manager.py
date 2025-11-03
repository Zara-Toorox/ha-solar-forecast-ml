"""
Data Manager API for the Solar Forecast ML Integration.
Provides high-level functions to access and modify ML data files,
inheriting low-level I/O operations from DataManagerIO.

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

import asyncio
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

from homeassistant.core import HomeAssistant

# Import IO Base Class
from ..data.io import DataManagerIO
from ..core.helpers import SafeDateTimeUtil as dt_util

from ..const import (
    MAX_PREDICTION_HISTORY, MIN_TRAINING_DATA_POINTS, DATA_VERSION,
    BACKUP_RETENTION_DAYS, MAX_BACKUP_FILES
)
from ..ml.ml_types import (
    LearnedWeights, HourlyProfile,
    create_default_learned_weights, create_default_hourly_profile,
    validate_prediction_record
)
from ..exceptions import DataIntegrityException
from ..core.helpers import SafeDateTimeUtil as dt_util
# Import TypedDataAdapter for conversions
from ..data.data_adapter import TypedDataAdapter

_LOGGER = logging.getLogger(__name__)


# Inherit from DataManagerIO to get file handling methods and lock
class DataManager(DataManagerIO):
    """
    Data Manager API for Solar Forecast ML.
    Handles the application logic for reading, writing, and managing ML data.
    Uses DataManagerIO for underlying file operations.
    CRITICAL FIX: All timestamp parsing now ensures LOCAL timezone.
    """

    def __init__(self, hass: HomeAssistant, entry_id: str, data_dir: Path, error_handler=None):
        """Initialize the Data Manager API."""
        # Initialize the base I/O class
        super().__init__(hass, data_dir)

        self.entry_id = entry_id
        self.error_handler = error_handler
        self.data_adapter = TypedDataAdapter()

        # Define specific file paths using the base data_dir
        self.prediction_history_file = self.data_dir / "prediction_history.json"
        self.learned_weights_file = self.data_dir / "learned_weights.json"
        self.hourly_profile_file = self.data_dir / "hourly_profile.json"
        self.model_state_file = self.data_dir / "model_state.json"
        self.hourly_samples_file = self.data_dir / "hourly_samples.json"
        self.coordinator_state_file = self.data_dir / "coordinator_state.json"

        # Defaults for _read_json_file
        self._prediction_history_default = {"version": DATA_VERSION, "predictions": [], "last_updated": None}
        self._hourly_samples_default = {"version": DATA_VERSION, "samples": [], "count": 0, "last_updated": None}
        self._model_state_default = {
            "version": DATA_VERSION, "model_loaded": False, "last_training": None,
            "training_samples": 0, "current_accuracy": 0.0, "status": "uninitialized",
        }
        self._coordinator_state_default = {
            "version": DATA_VERSION,
            "expected_daily_production": None,
            "last_set_date": None,
            "last_updated": None
        }

        _LOGGER.info("DataManager API initialized with LOCAL TIME enforcement")

    async def initialize(self) -> bool:
        """Initialize data manager: ensure directories exist and create default files."""
        try:
            # Ensure base and backup directories exist
            await self._ensure_directory_exists(self.data_dir)
            await self._ensure_directory_exists(self.data_dir / "backups")

            # Create default files if missing
            await self._initialize_missing_files()

            _LOGGER.info("DataManager initialized successfully")
            return True
        except Exception as e:
            _LOGGER.error(f"DataManager initialization failed: {e}")
            return False

    async def _initialize_missing_files(self):
        """Create default files if they don't exist."""
        files_to_check = [
            (self.prediction_history_file, self._prediction_history_default),
            (self.hourly_samples_file, self._hourly_samples_default),
            (self.model_state_file, self._model_state_default),
        ]

        for file_path, default_data in files_to_check:
            if not file_path.exists():
                await self._ensure_directory_exists(file_path.parent)
                await self._atomic_write_json(file_path, default_data)
                _LOGGER.info(f"Created default file: {file_path.name}")

    # --- Learned Weights Methods ---
    async def save_learned_weights(self, weights: LearnedWeights) -> bool:
        """Save learned weights to file."""
        try:
            weights_dict = self.data_adapter.learned_weights_to_dict(weights)
            await self._ensure_directory_exists(self.learned_weights_file.parent)
            await self._atomic_write_json(self.learned_weights_file, weights_dict)
            _LOGGER.info("Learned weights saved successfully")
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to save learned weights: {e}")
            return False

    async def load_learned_weights(self) -> Optional[LearnedWeights]:
        """Load learned weights from file."""
        try:
            data = await self._read_json_file(self.learned_weights_file, create_default_learned_weights())
            if data:
                return self.data_adapter.dict_to_learned_weights(data)
            return None
        except Exception as e:
            _LOGGER.error(f"Failed to load learned weights: {e}")
            return None

    async def get_learned_weights(self) -> Optional[LearnedWeights]:
        """Get learned weights. Alias for load_learned_weights()."""
        return await self.load_learned_weights()

    # --- Hourly Profile Methods ---
    async def save_hourly_profile(self, profile: HourlyProfile) -> bool:
        """Save hourly profile to file."""
        try:
            profile_dict = self.data_adapter.hourly_profile_to_dict(profile)
            await self._ensure_directory_exists(self.hourly_profile_file.parent)
            await self._atomic_write_json(self.hourly_profile_file, profile_dict)
            _LOGGER.info("Hourly profile saved successfully")
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to save hourly profile: {e}")
            return False

    async def load_hourly_profile(self) -> Optional[HourlyProfile]:
        """Load hourly profile from file."""
        try:
            data = await self._read_json_file(self.hourly_profile_file, create_default_hourly_profile())
            if data:
                return self.data_adapter.dict_to_hourly_profile(data)
            return None
        except Exception as e:
            _LOGGER.error(f"Failed to load hourly profile: {e}")
            return None

    async def get_hourly_profile(self) -> Optional[HourlyProfile]:
        """Get hourly profile. Alias for load_hourly_profile()."""
        return await self.load_hourly_profile()

    # --- Model State Methods ---
    async def save_model_state(self, state: Dict[str, Any]) -> bool:
        """Save model state to file."""
        try:
            await self._ensure_directory_exists(self.model_state_file.parent)
            await self._atomic_write_json(self.model_state_file, state)
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to save model state: {e}")
            return False

    async def load_model_state(self) -> Dict[str, Any]:
        """Load model state from file."""
        try:
            return await self._read_json_file(self.model_state_file, self._model_state_default)
        except Exception as e:
            _LOGGER.error(f"Failed to load model state: {e}")
            return self._model_state_default

    async def get_model_state(self) -> Dict[str, Any]:
        """Get model state. Alias for load_model_state()."""
        return await self.load_model_state()

    # --- Prediction History Methods ---
    async def save_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """Save a single prediction to history."""
        try:
            if not validate_prediction_record(prediction_data):
                _LOGGER.error("Prediction data validation failed")
                return False
            
            async with self._file_lock:
                history = await self._read_json_file(
                    self.prediction_history_file, 
                    self._prediction_history_default
                )
                
                if "predictions" not in history:
                    history["predictions"] = []
                
                history["predictions"].append(prediction_data)
                
                # Limit history size
                if len(history["predictions"]) > MAX_PREDICTION_HISTORY:
                    history["predictions"] = history["predictions"][-MAX_PREDICTION_HISTORY:]
                
                history["last_updated"] = dt_util.now().isoformat()
                
                await self._ensure_directory_exists(self.prediction_history_file.parent)
                await self._atomic_write_json_unlocked(self.prediction_history_file, history)
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to save prediction: {e}")
            return False

    async def add_prediction_record(self, record: Dict[str, Any]) -> bool:
        """Add a single prediction record to history. Alias for save_prediction()."""
        return await self.save_prediction(record)

    async def load_prediction_history(self) -> List[Dict[str, Any]]:
        """Load prediction history from file."""
        try:
            history = await self._read_json_file(self.prediction_history_file, self._prediction_history_default)
            return history.get("predictions", [])
        except Exception as e:
            _LOGGER.error(f"Failed to load prediction history: {e}")
            return []

    async def get_prediction_history(self) -> Dict[str, Any]:
        """Get complete prediction history as dictionary."""
        try:
            history = await self._read_json_file(
                self.prediction_history_file, 
                self._prediction_history_default
            )
            return history
        except Exception as e:
            _LOGGER.error(f"Failed to get prediction history: {e}")
            return self._prediction_history_default

    async def get_average_monthly_yield(self) -> Optional[float]:
        """
        Calculate average monthly yield from prediction history.
        CRITICAL FIX: Now ensures timestamps are in LOCAL timezone.
        """
        try:
            history = await self.get_prediction_history()
            predictions = history.get("predictions", [])
            
            if not predictions:
                _LOGGER.debug("No predictions available for average monthly yield calculation")
                return None
            
            cutoff_date = dt_util.now() - timedelta(days=30)
            valid_yields = []
            
            for pred in predictions:
                try:
                    timestamp = dt_util.parse_datetime(pred.get("timestamp"))
                    if timestamp:
                        # CRITICAL FIX: Force timestamp to LOCAL timezone
                        timestamp = dt_util.ensure_local(timestamp)
                    
                    actual_value = pred.get("actual_value")
                    
                    if timestamp and timestamp >= cutoff_date and actual_value is not None:
                        valid_yields.append(float(actual_value))
                except (ValueError, TypeError, KeyError):
                    continue
            
            if not valid_yields:
                _LOGGER.debug("No valid yield data in last 30 days")
                return None
            
            avg_yield = sum(valid_yields) / len(valid_yields)
            _LOGGER.debug(f"Average monthly yield: {avg_yield:.2f} kWh (from {len(valid_yields)} days)")
            return round(avg_yield, 2)
            
        except Exception as e:
            _LOGGER.error(f"Failed to calculate average monthly yield: {e}")
            return None

    # --- Hourly Samples Methods ---
    async def save_hourly_samples(self, samples: List[Dict[str, Any]]) -> bool:
        """Save hourly samples to file."""
        try:
            data = {
                "version": DATA_VERSION,
                "samples": samples,
                "count": len(samples),
                "last_updated": dt_util.now().isoformat()
            }
            await self._ensure_directory_exists(self.hourly_samples_file.parent)
            await self._atomic_write_json(self.hourly_samples_file, data)
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to save hourly samples: {e}")
            return False

    async def load_hourly_samples(self) -> List[Dict[str, Any]]:
        """Load hourly samples from file."""
        try:
            data = await self._read_json_file(self.hourly_samples_file, self._hourly_samples_default)
            return data.get("samples", [])
        except Exception as e:
            _LOGGER.error(f"Failed to load hourly samples: {e}")
            return []

    async def add_hourly_sample(self, sample: Dict[str, Any]) -> bool:
        """Add a single hourly sample to the collection."""
        try:
            async with self._file_lock:
                data = await self._read_json_file(
                    self.hourly_samples_file, 
                    self._hourly_samples_default
                )
                
                samples = data.get("samples", [])
                samples.append(sample)
                
                MAX_HOURLY_SAMPLES = 1440
                if len(samples) > MAX_HOURLY_SAMPLES:
                    samples = samples[-MAX_HOURLY_SAMPLES:]
                
                data["samples"] = samples
                data["count"] = len(samples)
                data["last_updated"] = dt_util.now().isoformat()
                
                await self._ensure_directory_exists(self.hourly_samples_file.parent)
                await self._atomic_write_json_unlocked(self.hourly_samples_file, data)
                
            _LOGGER.debug(f"Hourly sample added. Total samples: {len(samples)}")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Failed to add hourly sample: {e}")
            return False

    async def get_hourly_samples(self, days: int = 60) -> List[Dict[str, Any]]:
        """
        Get hourly samples from the last N days.
        CRITICAL FIX: Now ensures all timestamps are in LOCAL timezone.
        """
        try:
            all_samples = await self.load_hourly_samples()
            
            if not all_samples:
                _LOGGER.debug("No hourly samples available")
                return []
            
            cutoff_date = dt_util.now() - timedelta(days=days)
            filtered_samples = []
            
            for sample in all_samples:
                try:
                    timestamp = dt_util.parse_datetime(sample.get("timestamp"))
                    if timestamp:
                        # CRITICAL FIX: Force timestamp to LOCAL timezone
                        timestamp = dt_util.ensure_local(timestamp)
                        
                    if timestamp and timestamp >= cutoff_date:
                        filtered_samples.append(sample)
                except (ValueError, TypeError, KeyError):
                    continue
            
            _LOGGER.debug(f"Retrieved {len(filtered_samples)} hourly samples from last {days} days")
            return filtered_samples
            
        except Exception as e:
            _LOGGER.error(f"Failed to get hourly samples: {e}")
            return []

    async def get_last_collected_hour(self) -> Optional[datetime]:
        """
        Get timestamp of the last collected hourly sample.
        CRITICAL FIX: Returns LOCAL time.
        """
        try:
            samples = await self.load_hourly_samples()
            
            if not samples:
                return None
            
            latest_timestamp = None
            for sample in reversed(samples):
                try:
                    timestamp = dt_util.parse_datetime(sample.get("timestamp"))
                    if timestamp:
                        # CRITICAL FIX: Ensure timestamp is in LOCAL timezone
                        timestamp = dt_util.ensure_local(timestamp)
                        latest_timestamp = timestamp
                        break
                except (ValueError, TypeError, KeyError):
                    continue
            
            if latest_timestamp:
                _LOGGER.debug(f"Last collected hour: {latest_timestamp.isoformat()}")
            
            return latest_timestamp
            
        except Exception as e:
            _LOGGER.error(f"Failed to get last collected hour: {e}")
            return None

    async def set_last_collected_hour(self, timestamp: datetime) -> None:
        """
        Set last collected hour timestamp.
        
        Note: This is a no-op method because the actual sample is already stored
        in hourly_samples.json. The get_last_collected_hour() method reads from
        the samples directly, so no separate state storage is needed.
        
        This method exists only to maintain API compatibility with ml_sample_collector.
        """
        _LOGGER.debug(f"Sample collection timestamp recorded: {timestamp.isoformat()}")
        # No action needed - sample is already in hourly_samples.json
        pass

    async def get_all_training_records(self, days: int = 60) -> List[Dict[str, Any]]:
        """
        Get all training records (predictions + hourly samples) from the last N days.
        CRITICAL FIX: Now ensures all timestamps are in LOCAL timezone.
        """
        try:
            history_data = await self.get_prediction_history()
            predictions = history_data.get('predictions', [])
            hourly_samples = await self.load_hourly_samples()
            
            if not predictions and not hourly_samples:
                _LOGGER.debug("No training records available")
                return []
            
            cutoff_date = dt_util.now() - timedelta(days=days)
            
            filtered_predictions = []
            for pred in predictions:
                try:
                    timestamp = dt_util.parse_datetime(pred.get("timestamp"))
                    if timestamp:
                        # CRITICAL FIX: Force timestamp to LOCAL timezone
                        timestamp = dt_util.ensure_local(timestamp)
                        
                    if timestamp and timestamp >= cutoff_date:
                        filtered_predictions.append(pred)
                except (ValueError, TypeError, KeyError):
                    continue
            
            training_records = []
            
            for pred in filtered_predictions:
                record = pred.copy()
                record["record_type"] = "daily_prediction"
                training_records.append(record)
            
            for sample in hourly_samples:
                try:
                    # CRITICAL FIX: Ensure hourly sample timestamps are LOCAL
                    timestamp = dt_util.parse_datetime(sample.get("timestamp"))
                    if timestamp:
                        timestamp = dt_util.ensure_local(timestamp)
                        if timestamp >= cutoff_date:
                            record = sample.copy()
                            record["record_type"] = "hourly_sample"
                            training_records.append(record)
                except (ValueError, TypeError, KeyError):
                    continue
            
            training_records.sort(
                key=lambda x: dt_util.parse_datetime(x.get("timestamp")) or dt_util.now(),
                reverse=False
            )
            
            _LOGGER.debug(
                f"Retrieved {len(training_records)} training records "
                f"({len(filtered_predictions)} predictions + {len(hourly_samples)} hourly samples) "
                f"from last {days} days"
            )
            return training_records
            
        except Exception as e:
            _LOGGER.error(f"Failed to get training records: {e}")
            return []

    async def cleanup_duplicate_samples(self) -> Dict[str, int]:
        """Remove duplicate hourly samples based on timestamp."""
        try:
            async with self._file_lock:
                data = await self._read_json_file(
                    self.hourly_samples_file, 
                    self._hourly_samples_default
                )
                
                samples = data.get("samples", [])
                original_count = len(samples)
                
                if not samples:
                    return {"removed": 0, "remaining": 0}
                
                seen_timestamps = {}
                unique_samples = []
                
                for sample in samples:
                    timestamp_str = sample.get("timestamp")
                    if timestamp_str:
                        if timestamp_str not in seen_timestamps:
                            seen_timestamps[timestamp_str] = sample
                            unique_samples.append(sample)
                        else:
                            existing = seen_timestamps[timestamp_str]
                            if len(sample.keys()) > len(existing.keys()):
                                unique_samples.remove(existing)
                                unique_samples.append(sample)
                                seen_timestamps[timestamp_str] = sample
                    else:
                        unique_samples.append(sample)
                
                data["samples"] = unique_samples
                data["count"] = len(unique_samples)
                data["last_updated"] = dt_util.now().isoformat()
                
                await self._ensure_directory_exists(self.hourly_samples_file.parent)
                await self._atomic_write_json_unlocked(self.hourly_samples_file, data)
                
                removed_count = original_count - len(unique_samples)
                _LOGGER.info(f"Duplicate cleanup: {removed_count} duplicates removed, {len(unique_samples)} remaining")
                
                return {
                    "removed": removed_count,
                    "remaining": len(unique_samples)
                }
                
        except Exception as e:
            _LOGGER.error(f"Failed to cleanup duplicate samples: {e}")
            return {"removed": 0, "remaining": 0}

    async def cleanup_zero_production_samples(self) -> Dict[str, int]:
        """Remove hourly samples with zero or near-zero production."""
        try:
            async with self._file_lock:
                data = await self._read_json_file(
                    self.hourly_samples_file, 
                    self._hourly_samples_default
                )
                
                samples = data.get("samples", [])
                original_count = len(samples)
                
                if not samples:
                    return {"removed": 0, "remaining": 0}
                
                ZERO_THRESHOLD = 0.001
                filtered_samples = []
                
                for sample in samples:
                    try:
                        production = float(sample.get("production", 0) or 0)
                        if production > ZERO_THRESHOLD:
                            filtered_samples.append(sample)
                    except (ValueError, TypeError):
                        filtered_samples.append(sample)
                
                data["samples"] = filtered_samples
                data["count"] = len(filtered_samples)
                data["last_updated"] = dt_util.now().isoformat()
                
                await self._ensure_directory_exists(self.hourly_samples_file.parent)
                await self._atomic_write_json_unlocked(self.hourly_samples_file, data)
                
                removed_count = original_count - len(filtered_samples)
                _LOGGER.info(f"Zero-production cleanup: {removed_count} samples removed, {len(filtered_samples)} remaining")
                
                return {
                    "removed": removed_count,
                    "remaining": len(filtered_samples)
                }
                
        except Exception as e:
            _LOGGER.error(f"Failed to cleanup zero production samples: {e}")
            return {"removed": 0, "remaining": 0}

    # --- Weather Forecast Cache ---
    async def save_weather_forecast_cache(self, forecast_data: Dict[str, Any]) -> bool:
        """Save weather forecast cache."""
        try:
            cache_file = self.data_dir / "weather_cache.json"
            await self._ensure_directory_exists(cache_file.parent)
            await self._atomic_write_json(cache_file, forecast_data)
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to save weather cache: {e}")
            return False

    async def load_weather_forecast_cache(self) -> Optional[Dict[str, Any]]:
        """Load weather forecast cache with backward compatibility."""
        try:
            cache_file = self.data_dir / "weather_cache.json"
            if cache_file.exists():
                data = await self._read_json_file(cache_file, None)
                
                if data is None:
                    return None
                
                # Backward compatibility: If data is a list (old format), wrap it
                if isinstance(data, list):
                    _LOGGER.info("Converting old weather cache format (list) to new format (dict)")
                    return {
                        "forecast_hours": data,
                        "cached_at": dt_util.now().isoformat(),  # LOCAL time
                        "data_quality": {
                            "today_hours": 0,
                            "tomorrow_hours": 0,
                            "total_hours": len(data)
                        },
                        "converted_from_old_format": True
                    }
                
                # New format: Already a dict
                return data
            return None
        except Exception as e:
            _LOGGER.error(f"Failed to load weather cache: {e}")
            return None

    # --- Backup Methods ---
    async def create_backup(self, backup_name: Optional[str] = None) -> bool:
        """Create backup of all data files."""
        try:
            if not backup_name:
                timestamp = dt_util.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}"
            
            backup_dir = self.data_dir / "backups" / backup_name
            await self._ensure_directory_exists(backup_dir)
            
            # Copy all JSON files
            for file in self.data_dir.glob("*.json"):
                shutil.copy2(file, backup_dir / file.name)
            
            _LOGGER.info(f"Backup created: {backup_name}")
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to create backup: {e}")
            return False

    async def cleanup_old_backups(self):
        """Remove backups older than BACKUP_RETENTION_DAYS."""
        try:
            backup_dir = self.data_dir / "backups"
            if not backup_dir.exists():
                return
            
            cutoff_date = datetime.now() - timedelta(days=BACKUP_RETENTION_DAYS)
            
            for backup_folder in backup_dir.iterdir():
                if backup_folder.is_dir():
                    folder_time = datetime.fromtimestamp(backup_folder.stat().st_mtime)
                    if folder_time < cutoff_date:
                        shutil.rmtree(backup_folder)
                        _LOGGER.info(f"Removed old backup: {backup_folder.name}")
        except Exception as e:
            _LOGGER.error(f"Failed to cleanup old backups: {e}")

    # ==================================================================================
    # Coordinator State Persistence (Expected Daily Production)
    # ==================================================================================

    async def save_expected_daily_production(self, value: float) -> bool:
        """
        Save expected daily production value persistently.
        This value survives HA restarts until midnight reset.
        
        Args:
            value: Expected daily production in kWh
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            now_local = dt_util.as_local(dt_util.now())
            state = {
                "version": DATA_VERSION,
                "expected_daily_production": value,
                "last_set_date": now_local.date().isoformat(),
                "last_updated": now_local.isoformat()
            }
            
            success = await self._write_json_file(self.coordinator_state_file, state)
            if success:
                _LOGGER.debug(f"Expected daily production saved: {value:.2f} kWh")
            return success
            
        except Exception as e:
            _LOGGER.error(f"Failed to save expected daily production: {e}")
            return False

    async def load_expected_daily_production(self) -> Optional[float]:
        """
        Load expected daily production from persistent storage.
        Returns None if:
        - File doesn't exist
        - Value is from a different day (auto-expired)
        - Value is invalid
        
        Returns:
            Expected daily production value or None
        """
        try:
            state = await self._read_json_file(
                self.coordinator_state_file,
                self._coordinator_state_default
            )
            
            if not state:
                return None
            
            # Check if value is from today
            now_local = dt_util.as_local(dt_util.now())
            today_str = now_local.date().isoformat()
            last_set_date = state.get("last_set_date")
            
            if last_set_date != today_str:
                _LOGGER.debug(
                    f"Expected daily production expired "
                    f"(from {last_set_date}, today is {today_str})"
                )
                return None
            
            value = state.get("expected_daily_production")
            if value is not None:
                _LOGGER.debug(f"Loaded expected daily production: {value:.2f} kWh")
                return float(value)
            
            return None
            
        except Exception as e:
            _LOGGER.error(f"Failed to load expected daily production: {e}")
            return None

    async def clear_expected_daily_production(self) -> bool:
        """
        Clear expected daily production from persistent storage.
        Called at midnight reset.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            state = self._coordinator_state_default.copy()
            now_local = dt_util.as_local(dt_util.now())
            state["last_updated"] = now_local.isoformat()
            
            success = await self._write_json_file(self.coordinator_state_file, state)
            if success:
                _LOGGER.debug("Expected daily production cleared from persistent storage")
            return success
            
        except Exception as e:
            _LOGGER.error(f"Failed to clear expected daily production: {e}")
            return False
