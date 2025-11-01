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

from ..const import (
    MAX_PREDICTION_HISTORY, MIN_TRAINING_DATA_POINTS, DATA_VERSION,
    BACKUP_RETENTION_DAYS, MAX_BACKUP_FILES
)
from ..ml.types import (
    LearnedWeights, HourlyProfile,
    create_default_learned_weights, create_default_hourly_profile,
    validate_prediction_record
)
from ..exceptions import DataIntegrityException
from ..core.helpers import SafeDateTimeUtil as dt_util
# Import TypedDataAdapter for conversions
from ..data.adapter import TypedDataAdapter

_LOGGER = logging.getLogger(__name__)


# Inherit from DataManagerIO to get file handling methods and lock
class DataManager(DataManagerIO):
    """
    Data Manager API for Solar Forecast ML.
    Handles the application logic for reading, writing, and managing ML data.
    Uses DataManagerIO for underlying file operations.
    """

    def __init__(self, hass: HomeAssistant, entry_id: str, data_dir: Path, error_handler=None):
        """Initialize the Data Manager API."""
        # Initialize the base I/O class
        super().__init__(hass, data_dir)

        self.entry_id = entry_id
        self.error_handler = error_handler # Keep error handler if needed for logging/reporting
        self.data_adapter = TypedDataAdapter() # Adapter for type conversions

        # Define specific file paths using the base data_dir
        self.prediction_history_file = self.data_dir / "prediction_history.json"
        self.learned_weights_file = self.data_dir / "learned_weights.json"
        self.hourly_profile_file = self.data_dir / "hourly_profile.json"
        self.model_state_file = self.data_dir / "model_state.json"
        self.hourly_samples_file = self.data_dir / "hourly_samples.json"

        # Defaults for _read_json_file
        self._prediction_history_default = {"version": DATA_VERSION, "predictions": [], "last_updated": None}
        self._hourly_samples_default = {"version": DATA_VERSION, "samples": [], "count": 0, "last_updated": None}
        self._model_state_default = {
            "version": DATA_VERSION, "model_loaded": False, "last_training": None,
            "training_samples": 0, "current_accuracy": 0.0, "status": "uninitialized",
        }


        _LOGGER.info("DataManager API initialized (inherits IO operations)")

    async def initialize(self) -> bool:
        """
        Initialize data manager: ensure directories exist, create default files,
        run timestamp migration, and run data migrations.
        """
        try:
            # Ensure base and backup directories exist (uses inherited method)
            await self._ensure_directory_exists(self.data_dir)
            await self._ensure_directory_exists(self.data_dir / "backups")

            # Setup import tools (automatic deployment for historical data import)
            await self._setup_import_tools()

            # Create default files if they are missing
            await self._initialize_missing_files()

            # ===== TIMESTAMP MIGRATION (OPTION A) =====
            # Run before other migrations to ensure consistent timestamp format
            _LOGGER.info("Checking for timestamp migration to local time...")
            migration_success = await self.migrate_timestamps_to_local()
            if not migration_success:
                _LOGGER.warning("Timestamp migration had issues. Check logs. Continuing initialization.")
            # ===========================================

            # Run data migrations (structure updates)
            migration_success = await self.migrate_data()

            if migration_success:
                _LOGGER.info("DataManager API initialized successfully.")
                return True
            else:
                _LOGGER.warning("DataManager API initialized, but migration checks had issues.")
                # Continue even if migration had non-critical issues
                return True

        except Exception as e:
            _LOGGER.error(f"DataManager API initialization failed critically: {e}", exc_info=True)
            return False

    async def _initialize_missing_files(self) -> None:
        """Initialize missing data files with default content."""
        # Check and create each file using the inherited _atomic_write_json
        if not await self._file_exists(self.prediction_history_file):
            _LOGGER.info(f"Creating default file: {self.prediction_history_file.name}")
            await self._atomic_write_json(self.prediction_history_file, self._prediction_history_default)

        if not await self._file_exists(self.learned_weights_file):
            _LOGGER.info(f"Creating default file: {self.learned_weights_file.name}")
            default_weights_dict = self.data_adapter.learned_weights_to_dict(create_default_learned_weights())
            await self._atomic_write_json(self.learned_weights_file, default_weights_dict)

        if not await self._file_exists(self.hourly_profile_file):
            _LOGGER.info(f"Creating default file: {self.hourly_profile_file.name}")
            default_profile_dict = self.data_adapter.hourly_profile_to_dict(create_default_hourly_profile())
            await self._atomic_write_json(self.hourly_profile_file, default_profile_dict)

        if not await self._file_exists(self.model_state_file):
            _LOGGER.info(f"Creating default file: {self.model_state_file.name}")
            await self._atomic_write_json(self.model_state_file, self._model_state_default)

        if not await self._file_exists(self.hourly_samples_file):
            _LOGGER.info(f"Creating default file: {self.hourly_samples_file.name}")
            await self._atomic_write_json(self.hourly_samples_file, self._hourly_samples_default)


    async def _setup_import_tools(self) -> None:
        """
        Setup the import/ directory with historical data import tools.
        Copies import scripts and documentation from integration package to data directory.
        Only copies if files don't exist or if version is newer.
        
        This enables users to quickly bootstrap ML training by importing historical
        sensor data instead of waiting 30 days for natural data collection.
        """
        try:
            # Define import directory
            import_dir = self.data_dir / "import"
            
            # Ensure import directory exists
            await self._ensure_directory_exists(import_dir)
            
            # Get path to integration's import_tools directory
            # The integration root is two levels up (manager.py is in data/ subdirectory)
            integration_root = Path(__file__).parent.parent.resolve()
            tools_source_dir = integration_root / "import_tools"
            
            # Check if import_tools directory exists in integration
            if not await self.hass.async_add_executor_job(tools_source_dir.exists):
                _LOGGER.debug("import_tools directory not found in integration package. Skipping import tools setup.")
                return
            
            # Define files to copy
            tool_files = [
                "import_historical_data.py",
                "validate_csv_files.py",
                "IMPORT_README.md",
                "QUICK_START.md"
            ]
            
            # Copy each file if it doesn't exist or is outdated
            copied_count = 0
            updated_count = 0
            
            for filename in tool_files:
                source_file = tools_source_dir / filename
                target_file = import_dir / filename
                
                # Check if source file exists
                if not await self.hass.async_add_executor_job(source_file.exists):
                    _LOGGER.debug(f"Source file not found in import_tools: {filename}")
                    continue
                
                # Check if target file exists
                target_exists = await self._file_exists(target_file)
                
                # Determine if copy is needed
                should_copy = False
                reason = ""
                
                if not target_exists:
                    should_copy = True
                    reason = "File missing"
                else:
                    # Compare modification times to detect updates
                    try:
                        source_stat = await self.hass.async_add_executor_job(source_file.stat)
                        target_stat = await self.hass.async_add_executor_job(target_file.stat)
                        
                        if source_stat.st_mtime > target_stat.st_mtime:
                            should_copy = True
                            reason = "Source is newer"
                            updated_count += 1
                    except Exception as stat_err:
                        _LOGGER.debug(f"Could not compare timestamps for {filename}: {stat_err}")
                        # Don't copy if we can't determine - safer to keep existing file
                
                if should_copy:
                    # Copy file (preserving metadata like modification time)
                    await self.hass.async_add_executor_job(
                        shutil.copy2, str(source_file), str(target_file)
                    )
                    
                    if reason == "File missing":
                        copied_count += 1
                    
                    _LOGGER.info(f"Deployed import tool: {filename} ({reason})")
            
            # Log summary
            if copied_count > 0 or updated_count > 0:
                _LOGGER.info(
                    f"Import tools deployment complete: {copied_count} new file(s), "
                    f"{updated_count} updated file(s) in {import_dir}"
                )
            else:
                _LOGGER.debug("Import tools already up-to-date.")
            
        except Exception as e:
            # Don't fail initialization if import tools setup fails
            # This is a convenience feature, not critical for operation
            _LOGGER.warning(
                f"Failed to setup import tools (non-critical): {e}",
                exc_info=True
            )


    async def reset_for_architecture_change(self) -> bool:
        """
        Complete reset for architecture change (v6.4.0).
        Deletes all ML data files and recreates them empty.
        Sets migration flag in versinfo.json to prevent repeated execution.
        
        Note: This method is currently called directly from __init__.py synchronously.
        This async version is kept for potential future use.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            _LOGGER.warning("Starting architecture migration: Deleting all ML training data...")
            
            async with self._file_lock:
                # Delete all JSON files
                files_to_delete = [
                    self.prediction_history_file,
                    self.learned_weights_file,
                    self.hourly_profile_file,
                    self.model_state_file,
                    self.hourly_samples_file
                ]
                
                deleted_count = 0
                for file_path in files_to_delete:
                    if await self._file_exists(file_path):
                        try:
                            await self.hass.async_add_executor_job(file_path.unlink)
                            deleted_count += 1
                            _LOGGER.info(f"Deleted: {file_path.name}")
                        except Exception as del_err:
                            _LOGGER.error(f"Failed to delete {file_path.name}: {del_err}")
                
                _LOGGER.info(f"Deleted {deleted_count} ML data files")
                
                # Recreate empty default files
                _LOGGER.info("Creating fresh empty ML data files...")
                await self._initialize_missing_files()
                
                # Redeploy import tools
                _LOGGER.info("Redeploying import tools...")
                await self._setup_import_tools()
                
                # Write migration flag to versinfo.json
                versinfo_file = self.data_dir / "versinfo.json"
                versinfo_data = {
                    "version": "6.4.0",
                    "migration_6_4_0": True,
                    "migration_date": dt_util.now().isoformat(),
                    "description": "Architecture change - ML data reset"
                }
                await self._atomic_write_json_unlocked(versinfo_file, versinfo_data)
                
                _LOGGER.warning(
                    "Architecture migration complete. All ML training data has been reset. "
                    "ML training will start fresh from this point."
                )
                
            return True
            
        except Exception as e:
            _LOGGER.error(f"Architecture migration failed: {e}", exc_info=True)
            return False


    # --- API Methods for Data Access ---

    async def add_prediction_record(self, record: Dict[str, Any]) -> None:
        """
        Adds a prediction record to the history file. (Thread-safe)
        """
        try:
            validate_prediction_record(record) # Validate input structure first
            async with self._file_lock: # Acquire lock for read-modify-write
                # Read current history (uses inherited _read_json_file)
                history = await self._read_json_file(
                    self.prediction_history_file,
                    default_structure=self._prediction_history_default
                )

                # Modify data in memory
                predictions = history.get("predictions", [])
                predictions.append(record)

                # Enforce history limit
                if len(predictions) > MAX_PREDICTION_HISTORY:
                    predictions = predictions[-MAX_PREDICTION_HISTORY:]

                history["predictions"] = predictions
                history["last_updated"] = dt_util.now().isoformat()  # OPTION A: LOCAL TIME
                history["version"] = DATA_VERSION # Ensure version is set

                # Write back atomically (uses inherited unlocked write)
                await self._atomic_write_json_unlocked(self.prediction_history_file, history)

            _LOGGER.debug("Prediction record added successfully.")
        except Exception as e:
            _LOGGER.error(f"Failed to add prediction record: {e}", exc_info=True)
            # Re-raise as DataIntegrityException to signal failure
            raise DataIntegrityException(f"Failed to add prediction record: {str(e)}")


    async def update_today_predictions_actual(self, actual_value: float, accuracy: Optional[float] = None) -> None:
        """
        Updates prediction records from today with the actual value. (Thread-safe)
        """
        try:
            async with self._file_lock: # Acquire lock for read-modify-write
                # Read current history
                history_data = await self._read_json_file(
                    self.prediction_history_file,
                    default_structure=self._prediction_history_default
                )
                predictions = history_data.get('predictions', [])
                if not predictions:
                    _LOGGER.debug("No prediction history found to update.")
                    return # Nothing to do

                # Determine today's date in local timezone for comparison
                today_local = dt_util.now().date()  # OPTION A: DIRECT LOCAL DATE
                updated_count = 0
                needs_save = False

                # Iterate through predictions (consider iterating reversed for efficiency)
                for record in predictions:
                    try:
                        # Parse timestamp and get local date
                        record_timestamp = dt_util.parse_datetime(record.get('timestamp', ''))
                        if record_timestamp is None:
                            continue
                        record_date_local = dt_util.as_local(record_timestamp).date()

                        # Check if record is from today and needs updating
                        if record_date_local == today_local and record.get('actual_value') is None:
                            record['actual_value'] = actual_value
                            if accuracy is not None:
                                record['accuracy'] = accuracy # Store calculated accuracy
                            updated_count += 1
                            needs_save = True # Mark that we need to save changes

                    except (ValueError, KeyError, TypeError) as parse_error:
                        # Log and skip invalid records
                        _LOGGER.debug(f"Skipping record during update due to parsing error: {parse_error}")
                        continue

                # If any records were updated, write the changes back
                if needs_save:
                    history_data["last_updated"] = dt_util.now().isoformat()  # OPTION A: LOCAL TIME
                    history_data["version"] = DATA_VERSION
                    await self._atomic_write_json_unlocked(self.prediction_history_file, history_data)
                    _LOGGER.info(f"Updated {updated_count} prediction records from today with actual value.")
                else:
                     _LOGGER.debug("No prediction records found for today needing an actual value update.")

        except Exception as e:
            _LOGGER.error(f"Failed to update today's predictions: {e}", exc_info=True)
            raise DataIntegrityException(f"Failed to update today's predictions: {str(e)}")


    async def get_prediction_history(self) -> Dict[str, Any]:
        """Reads the entire prediction history."""
        # Uses inherited _read_json_file with appropriate default
        return await self._read_json_file(
            self.prediction_history_file,
            default_structure=self._prediction_history_default
        )


    async def get_all_training_records(self, days: int = 60) -> List[Dict[str, Any]]:
        """
        Prepares a list of records suitable for ML training, primarily from hourly samples.
        Filters invalid/incomplete samples.
        """
        all_training_records = []
        try:
            _LOGGER.debug(f"Loading hourly samples from last {days} days for training...")
            # Use get_hourly_samples API which reads the file
            samples_data = await self.get_hourly_samples(days=days)
            samples = samples_data.get('samples', [])

            if not samples:
                _LOGGER.warning("No hourly samples found for the specified period. Cannot prepare training data.")
                return []

            valid_sample_count = 0
            skipped_count = 0
            # Process each sample to create a training record
            for sample in samples:
                try:
                    # Basic validation: Check for essential keys and positive energy yield
                    actual_kwh = sample.get('actual_kwh')
                    weather_data = sample.get('weather_data')
                    sensor_data = sample.get('sensor_data')
                    timestamp = sample.get('timestamp')

                    if not all([timestamp, weather_data, sensor_data]) or actual_kwh is None or actual_kwh <= 0:
                        skipped_count += 1
                        continue # Skip incomplete or zero-yield samples

                    # Structure the record as expected by the trainer
                    training_record = {
                        'timestamp': timestamp,
                        'actual_value': actual_kwh, # Target value for training
                        'weather_data': weather_data,
                        'sensor_data': sensor_data,
                        'model_version': sample.get('model_version', DATA_VERSION), # Include version if available
                        'source': 'hourly_samples' # Mark the data source
                    }
                    all_training_records.append(training_record)
                    valid_sample_count += 1

                except Exception as e:
                     # Catch unexpected errors during sample processing
                     _LOGGER.warning(f"Skipping sample due to unexpected error ({e}): {sample.get('timestamp')}")
                     skipped_count += 1
                     continue

            # Sort records chronologically (important for some ML approaches)
            all_training_records.sort(key=lambda x: x['timestamp'])

            _LOGGER.info(f"Prepared {valid_sample_count} valid training records "
                         f"(skipped {skipped_count} invalid/incomplete) from {len(samples)} total hourly samples.")
            return all_training_records

        except Exception as e:
            _LOGGER.error(f"Failed to get training records from hourly samples: {e}", exc_info=True)
            return [] # Return empty list on failure


    async def get_learned_weights(self) -> Optional[LearnedWeights]:
        """Reads and parses the learned weights file."""
        try:
            # Read using inherited method
            data = await self._read_json_file(self.learned_weights_file, default_structure=None) # Default None to detect non-existence
            if data is None or not data: # Check if file didn't exist or was empty/invalid
                 _LOGGER.warning("Learned weights file not found or empty.")
                 return None

            # Validate basic structure before parsing
            # Add more checks if specific keys are absolutely required
            if "weights" not in data and "weather_weights" not in data:
                 _LOGGER.error("Learned weights file is corrupted or incomplete (missing weights).")
                 return None

            # Convert dict to dataclass using adapter
            return self.data_adapter.dict_to_learned_weights(data)

        except DataIntegrityException as e:
             _LOGGER.error(f"Data integrity error reading learned weights: {e}")
             return None
        except Exception as e:
            _LOGGER.error(f"Unexpected error getting learned weights: {e}", exc_info=True)
            return None


    async def save_learned_weights(self, weights: LearnedWeights) -> None:
        """Saves the learned weights dataclass to a JSON file. (Thread-safe)"""
        try:
            if not isinstance(weights, LearnedWeights):
                 raise TypeError("Invalid data type provided for learned weights.")

            # Convert dataclass to dict using adapter
            data = self.data_adapter.learned_weights_to_dict(weights)
            # Add metadata (redundant if adapter does it, but safe)
            data["version"] = DATA_VERSION
            data["last_updated"] = dt_util.now().isoformat()  # OPTION A: LOCAL TIME

            # Write atomically using inherited method (acquires lock)
            await self._atomic_write_json(self.learned_weights_file, data)
            _LOGGER.info("Learned weights saved successfully.")
        except Exception as e:
            _LOGGER.error(f"Failed to save learned weights: {e}", exc_info=True)
            raise DataIntegrityException(f"Failed to save learned weights: {str(e)}")


    async def get_hourly_profile(self) -> Optional[HourlyProfile]:
        """Reads and parses the hourly production profile file."""
        try:
            data = await self._read_json_file(self.hourly_profile_file, default_structure=None)
            if data is None or not data:
                 _LOGGER.warning("Hourly profile file not found or empty.")
                 return None

            # Basic validation
            if "hourly_averages" not in data and "hourly_factors" not in data: # Check old and new keys
                 _LOGGER.error("Hourly profile file is corrupted or incomplete.")
                 return None

            return self.data_adapter.dict_to_hourly_profile(data)
        except DataIntegrityException as e:
             _LOGGER.error(f"Data integrity error reading hourly profile: {e}")
             return None
        except Exception as e:
            _LOGGER.error(f"Unexpected error getting hourly profile: {e}", exc_info=True)
            return None


    async def save_hourly_profile(self, profile: HourlyProfile) -> None:
        """Saves the hourly profile dataclass to a JSON file. (Thread-safe)"""
        try:
            if not isinstance(profile, HourlyProfile):
                 raise TypeError("Invalid data type provided for hourly profile.")

            data = self.data_adapter.hourly_profile_to_dict(profile)
            data["version"] = DATA_VERSION
            data["last_updated"] = dt_util.now().isoformat()  # OPTION A: LOCAL TIME

            await self._atomic_write_json(self.hourly_profile_file, data)
            _LOGGER.info("Hourly profile saved successfully.")
        except Exception as e:
             _LOGGER.error(f"Failed to save hourly profile: {e}", exc_info=True)
             raise DataIntegrityException(f"Failed to save hourly profile: {str(e)}")


    async def get_model_state(self) -> Dict[str, Any]:
        """Reads the model state file."""
        # Use inherited _read_json_file with the correct default
        state = await self._read_json_file(
            self.model_state_file,
            default_structure=self._model_state_default
        )
        # Ensure backfill_run is not present (migration handled elsewhere if needed)
        state.pop("backfill_run", None)
        return state

    async def save_model_state(self, state: Dict[str, Any]) -> None:
        """Saves the model state dictionary to a JSON file. (Thread-safe)"""
        try:
            if not isinstance(state, dict):
                 raise TypeError("Invalid data type provided for model state (must be dict).")

            # Ensure required fields and remove obsolete ones
            state.setdefault("version", DATA_VERSION)
            state.pop("backfill_run", None) # Remove obsolete key
            state["last_updated"] = dt_util.now().isoformat()  # OPTION A: LOCAL TIME

            await self._atomic_write_json(self.model_state_file, state)
            _LOGGER.debug("Model state saved successfully.")
        except Exception as e:
             _LOGGER.error(f"Failed to save model state: {e}", exc_info=True)
             raise DataIntegrityException(f"Failed to save model state: {str(e)}")


    async def add_hourly_sample(self, sample: Dict[str, Any]) -> None:
        """Adds an hourly sample to the samples file. (Thread-safe)"""
        try:
            # Basic validation of the input sample
            if not all(k in sample for k in ["timestamp", "actual_kwh", "weather_data", "sensor_data"]):
                 _LOGGER.warning(f"Skipping incomplete hourly sample: {sample.get('timestamp')}")
                 return

            async with self._file_lock: # Acquire lock for read-modify-write
                # Read current samples
                samples_data = await self._read_json_file(
                    self.hourly_samples_file,
                    default_structure=self._hourly_samples_default
                )
                samples = samples_data.get("samples", [])

                # DUPLICATE CHECK: Prevent adding samples with existing timestamps
                timestamp = sample.get('timestamp')
                if any(s.get('timestamp') == timestamp for s in samples):
                    _LOGGER.warning(f"Sample for {timestamp} already existing, skiping Duplicate.")
                    return

                # Append new sample
                samples.append(sample)

                # Prune old samples: Keep last 60 days based on LOCAL DATE
                # OPTION A: Use local time for cutoff
                now_local = dt_util.now()
                cutoff_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=60)
                
                try:
                    # Filter based on timestamp, handle potential parsing errors
                    filtered_samples = []
                    for s in samples:
                        try:
                            ts = dt_util.parse_datetime(s['timestamp'])
                            if ts is None:
                                continue
                            # Convert to local for date comparison
                            ts_local = dt_util.as_local(ts)
                            if ts_local >= cutoff_local:
                                filtered_samples.append(s)
                        except (ValueError, KeyError, TypeError):
                            continue  # Skip samples with invalid timestamps
                            
                except Exception as parse_err:
                     _LOGGER.warning(f"Error parsing timestamp during hourly sample pruning: {parse_err}. Keeping all samples.")
                     filtered_samples = samples # Keep all if parsing fails

                # Update data structure
                samples_data["samples"] = filtered_samples
                samples_data["last_updated"] = dt_util.now().isoformat()  # OPTION A: LOCAL TIME
                samples_data["count"] = len(filtered_samples)
                samples_data["version"] = DATA_VERSION

                # Write back atomically
                await self._atomic_write_json_unlocked(self.hourly_samples_file, samples_data)

            _LOGGER.debug(f"Hourly sample added for {sample['timestamp']}")

        except Exception as e:
            _LOGGER.error(f"Failed to add hourly sample: {e}", exc_info=True)
            # Don't raise DataIntegrityException here, as one failed sample shouldn't stop the integration


    async def get_hourly_samples(self, days: int = 0) -> Dict[str, Any]:
        """
        Reads hourly samples, optionally filtered by the last N days.
        If days=0, returns all samples.
        """
        try:
            # Read the raw data
            data = await self._read_json_file(
                self.hourly_samples_file,
                default_structure=self._hourly_samples_default
            )

            # Filter if days > 0
            if days > 0:
                # OPTION A: Use local time for cutoff
                now_local = dt_util.now()
                cutoff_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
                
                samples = data.get("samples", [])
                filtered_samples = []
                parse_errors = 0
                for sample in samples:
                    try:
                        timestamp = dt_util.parse_datetime(sample.get("timestamp", ""))
                        if timestamp is None:
                            parse_errors += 1
                            continue
                        # Convert to local for comparison
                        timestamp_local = dt_util.as_local(timestamp)
                        if timestamp_local >= cutoff_local:
                            filtered_samples.append(sample)
                    except (ValueError, KeyError, TypeError):
                         parse_errors += 1
                         # Don't include samples with invalid timestamps in filtered result
                         continue
                if parse_errors > 0:
                     _LOGGER.debug(f"Skipped {parse_errors} samples with invalid timestamps during filtering.")

                # Return a new dict with the filtered samples
                return {
                    "version": data.get("version", DATA_VERSION),
                    "samples": filtered_samples,
                    "count": len(filtered_samples),
                    "last_updated": data.get("last_updated")
                }
            else:
                 # Return a copy of the full data if no filtering needed
                 return data.copy()

        except Exception as e:
             _LOGGER.error(f"Unexpected error getting hourly samples: {e}", exc_info=True)
             # Return default structure on error
             return self._hourly_samples_default.copy()


    async def get_average_monthly_yield(self) -> float:
        """
        Calculates the average monthly yield (kWh) based on the last 30 days
        of hourly samples.
        """
        try:
            # Get hourly samples for the last 30 days
            samples_data = await self.get_hourly_samples(days=30)
            samples = samples_data.get('samples', [])
            if not samples:
                _LOGGER.debug("No hourly samples found within the last 30 days for avg monthly yield calculation.")
                return 0.0

            # Aggregate hourly yields per day using LOCAL DATE (OPTION A)
            daily_totals_kwh: Dict[datetime.date, float] = {}
            valid_days = set()

            for sample in samples:
                try:
                    actual_kwh = sample.get('actual_kwh')
                    # Include 0 kWh values, but skip None or negative
                    if actual_kwh is None or actual_kwh < 0:
                        continue

                    # Parse timestamp and convert to local
                    timestamp = dt_util.parse_datetime(sample['timestamp'])
                    if timestamp is None:
                        continue
                    timestamp_local = dt_util.as_local(timestamp)
                    date_key_local = timestamp_local.date()  # OPTION A: LOCAL DATE

                    # Add to daily total
                    daily_totals_kwh[date_key_local] = daily_totals_kwh.get(date_key_local, 0.0) + actual_kwh
                    valid_days.add(date_key_local)

                except (ValueError, KeyError, TypeError):
                    # Log and skip samples with invalid timestamp or kwh value
                    _LOGGER.debug(f"Skipping invalid sample during yield aggregation: {sample.get('timestamp')}")
                    continue

            # Check if any valid daily totals were aggregated
            if not daily_totals_kwh:
                 _LOGGER.warning("No valid daily totals could be aggregated from hourly samples for avg monthly yield.")
                 return 0.0

            # Calculate the number of unique days with data in the period
            num_days_with_data = len(valid_days)
            if num_days_with_data == 0:
                 return 0.0

            # Calculate average daily yield based on available days
            total_yield_kwh = sum(daily_totals_kwh.values())
            avg_daily_yield_kwh = total_yield_kwh / num_days_with_data

            # Extrapolate to a 30-day month
            avg_monthly_yield_kwh = avg_daily_yield_kwh * 30

            _LOGGER.debug(
                f"Average Monthly Yield calculated: {avg_monthly_yield_kwh:.2f} kWh "
                f"(Based on {total_yield_kwh:.2f} kWh over {num_days_with_data} days within the last 30)"
            )
            return round(avg_monthly_yield_kwh, 2)

        except Exception as e:
            _LOGGER.error(f"Failed to calculate average monthly yield from hourly samples: {e}", exc_info=True)
            return 0.0 # Return 0.0 on any calculation error

    async def reset_ml_data(self) -> None:
        """
        Resets all ML-related data files to their default (empty) state.
        This includes weights, profiles, samples, history, and model state.
        (Thread-safe)
        """
        _LOGGER.warning("Executing ML Data Reset. All learned progress and history will be lost.")
        try:
            async with self._file_lock: # Acquire lock for all reset operations
                _LOGGER.debug("Resetting learned_weights.json...")
                default_weights_dict = self.data_adapter.learned_weights_to_dict(create_default_learned_weights())
                await self._atomic_write_json_unlocked(self.learned_weights_file, default_weights_dict)
                
                _LOGGER.debug("Resetting hourly_profile.json...")
                default_profile_dict = self.data_adapter.hourly_profile_to_dict(create_default_hourly_profile())
                await self._atomic_write_json_unlocked(self.hourly_profile_file, default_profile_dict)
                
                _LOGGER.debug("Resetting hourly_samples.json...")
                await self._atomic_write_json_unlocked(self.hourly_samples_file, self._hourly_samples_default)
                
                _LOGGER.debug("Resetting model_state.json...")
                await self._atomic_write_json_unlocked(self.model_state_file, self._model_state_default)
                
                _LOGGER.debug("Resetting prediction_history.json...")
                await self._atomic_write_json_unlocked(self.prediction_history_file, self._prediction_history_default)
                
            _LOGGER.info("All ML data files have been reset to their default state.")
            
        except Exception as e:
            _LOGGER.error(f"Failed during ML data reset: {e}", exc_info=True)
            # Re-raise as DataIntegrityException to signal failure
            raise DataIntegrityException(f"Failed to reset ML data files: {str(e)}")

    # ========================================================================
    # OPTION A: TIMESTAMP MIGRATION TO LOCAL TIME
    # ========================================================================

    async def migrate_timestamps_to_local(self) -> bool:
        """
        Migrates all UTC timestamps in data files to local time (one-time operation).
        Checks migration flag and skips if already completed.
        
        Returns:
            True if migration successful or not needed, False if failed.
        """
        try:
            # 1. Check if migration already completed
            state = await self.get_model_state()
            if state.get("timestamp_migration_to_local"):
                _LOGGER.debug("Timestamp migration to local time already completed. Skipping.")
                return True
            
            _LOGGER.info("Starting one-time timestamp migration from UTC to local time...")
            
            # 2. Optional: Create backup before migration
            backup_dir = self.data_dir / "backups" / "pre_migration"
            try:
                await self._ensure_directory_exists(backup_dir)
                await self._create_migration_backup(backup_dir)
                _LOGGER.info(f"Backup created in {backup_dir}")
            except Exception as backup_err:
                _LOGGER.warning(f"Backup before migration failed: {backup_err}. Continuing without backup.")
            
            # 3. Migrate each file
            all_success = True
            
            # 3.1 hourly_samples.json
            if not await self._migrate_hourly_samples_timestamps():
                all_success = False
            
            # 3.2 prediction_history.json
            if not await self._migrate_prediction_history_timestamps():
                all_success = False
            
            # 3.3 learned_weights.json
            if not await self._migrate_learned_weights_timestamps():
                all_success = False
            
            # 3.4 hourly_profile.json
            if not await self._migrate_hourly_profile_timestamps():
                all_success = False
            
            # 3.5 model_state.json (migrate own timestamps and set flag)
            if not await self._migrate_model_state_timestamps():
                all_success = False
            
            # 4. Report results
            if all_success:
                _LOGGER.info("Timestamp migration to local time completed successfully.")
            else:
                _LOGGER.warning("Timestamp migration partially failed. Some files skipped. Check logs.")
            
            return all_success
            
        except Exception as e:
            _LOGGER.error(f"Critical error during timestamp migration: {e}", exc_info=True)
            return False


    async def _create_migration_backup(self, backup_dir: Path) -> None:
        """Creates backup of data files before migration."""
        files_to_backup = [
            self.hourly_samples_file,
            self.prediction_history_file,
            self.learned_weights_file,
            self.hourly_profile_file,
            self.model_state_file
        ]
        
        for file_path in files_to_backup:
            if await self._file_exists(file_path):
                backup_path = backup_dir / file_path.name
                try:
                    await self.hass.async_add_executor_job(
                        lambda src=file_path, dst=backup_path: dst.write_bytes(src.read_bytes())
                    )
                    _LOGGER.debug(f"Backup created: {file_path.name}")
                except Exception as e:
                    _LOGGER.warning(f"Backup of {file_path.name} failed: {e}")


    def _migrate_timestamp_string(self, timestamp_str: str) -> str:
        """
        Converts a single timestamp string from UTC to local time.
        Heuristically detects UTC timestamps (+00:00 or Z suffix).
        
        Args:
            timestamp_str: ISO format timestamp
        
        Returns:
            ISO format timestamp in local time
        """
        try:
            # Parse the timestamp
            dt = dt_util.parse_datetime(timestamp_str)
            if dt is None:
                _LOGGER.debug(f"Could not parse timestamp: {timestamp_str}")
                return timestamp_str
            
            # Heuristic: Check if UTC (ends with +00:00 or Z)
            if timestamp_str.endswith('+00:00') or timestamp_str.endswith('Z'):
                # Convert UTC to local
                dt_local = dt_util.as_local(dt)
                migrated = dt_local.isoformat()
                _LOGGER.debug(f"Migrated UTC timestamp: {timestamp_str} {migrated}")
                return migrated
            else:
                # Already has non-UTC offset, likely already local
                _LOGGER.debug(f"Timestamp appears already local: {timestamp_str}")
                return timestamp_str
                
        except Exception as e:
            _LOGGER.warning(f"Error migrating timestamp '{timestamp_str}': {e}")
            return timestamp_str


    async def _migrate_hourly_samples_timestamps(self) -> bool:
        """Migrates timestamps in hourly_samples.json."""
        try:
            if not await self._file_exists(self.hourly_samples_file):
                _LOGGER.debug("hourly_samples.json does not exist, skipping migration.")
                return True
            
            _LOGGER.info("Migrating hourly_samples.json timestamps...")
            
            async with self._file_lock:
                data = await self._read_json_file(self.hourly_samples_file)
                
                if not data or not data.get("samples"):
                    _LOGGER.debug("hourly_samples.json is empty, nothing to migrate.")
                    return True
                
                migrated_count = 0
                samples = data.get("samples", [])
                
                for sample in samples:
                    if "timestamp" in sample:
                        old_ts = sample["timestamp"]
                        new_ts = self._migrate_timestamp_string(old_ts)
                        if new_ts != old_ts:
                            sample["timestamp"] = new_ts
                            migrated_count += 1
                
                # Migrate last_updated
                if "last_updated" in data and data["last_updated"]:
                    old_ts = data["last_updated"]
                    new_ts = self._migrate_timestamp_string(old_ts)
                    if new_ts != old_ts:
                        data["last_updated"] = new_ts
                        migrated_count += 1
                
                # Write back
                await self._atomic_write_json_unlocked(self.hourly_samples_file, data)
                
            _LOGGER.info(f"hourly_samples.json: {migrated_count} timestamps migrated.")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Migration of hourly_samples.json failed: {e}", exc_info=True)
            return False


    async def _migrate_prediction_history_timestamps(self) -> bool:
        """Migrates timestamps in prediction_history.json."""
        try:
            if not await self._file_exists(self.prediction_history_file):
                _LOGGER.debug("prediction_history.json does not exist, skipping migration.")
                return True
            
            _LOGGER.info("Migrating prediction_history.json timestamps...")
            
            async with self._file_lock:
                data = await self._read_json_file(self.prediction_history_file)
                
                if not data or not data.get("predictions"):
                    _LOGGER.debug("prediction_history.json is empty, nothing to migrate.")
                    return True
                
                migrated_count = 0
                predictions = data.get("predictions", [])
                
                for prediction in predictions:
                    if "timestamp" in prediction:
                        old_ts = prediction["timestamp"]
                        new_ts = self._migrate_timestamp_string(old_ts)
                        if new_ts != old_ts:
                            prediction["timestamp"] = new_ts
                            migrated_count += 1
                
                # Migrate last_updated
                if "last_updated" in data and data["last_updated"]:
                    old_ts = data["last_updated"]
                    new_ts = self._migrate_timestamp_string(old_ts)
                    if new_ts != old_ts:
                        data["last_updated"] = new_ts
                        migrated_count += 1
                
                await self._atomic_write_json_unlocked(self.prediction_history_file, data)
                
            _LOGGER.info(f"prediction_history.json: {migrated_count} timestamps migrated.")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Migration of prediction_history.json failed: {e}", exc_info=True)
            return False


    async def _migrate_learned_weights_timestamps(self) -> bool:
        """Migrates timestamps in learned_weights.json."""
        try:
            if not await self._file_exists(self.learned_weights_file):
                _LOGGER.debug("learned_weights.json does not exist, skipping migration.")
                return True
            
            _LOGGER.info("Migrating learned_weights.json timestamps...")
            
            async with self._file_lock:
                data = await self._read_json_file(self.learned_weights_file)
                
                if not data:
                    _LOGGER.debug("learned_weights.json is empty, nothing to migrate.")
                    return True
                
                migrated_count = 0
                
                # Only last_updated field
                if "last_updated" in data and data["last_updated"]:
                    old_ts = data["last_updated"]
                    new_ts = self._migrate_timestamp_string(old_ts)
                    if new_ts != old_ts:
                        data["last_updated"] = new_ts
                        migrated_count += 1
                
                await self._atomic_write_json_unlocked(self.learned_weights_file, data)
                
            _LOGGER.info(f"learned_weights.json: {migrated_count} timestamps migrated.")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Migration of learned_weights.json failed: {e}", exc_info=True)
            return False


    async def _migrate_hourly_profile_timestamps(self) -> bool:
        """Migrates timestamps in hourly_profile.json."""
        try:
            if not await self._file_exists(self.hourly_profile_file):
                _LOGGER.debug("hourly_profile.json does not exist, skipping migration.")
                return True
            
            _LOGGER.info("Migrating hourly_profile.json timestamps...")
            
            async with self._file_lock:
                data = await self._read_json_file(self.hourly_profile_file)
                
                if not data:
                    _LOGGER.debug("hourly_profile.json is empty, nothing to migrate.")
                    return True
                
                migrated_count = 0
                
                # Only last_updated field
                if "last_updated" in data and data["last_updated"]:
                    old_ts = data["last_updated"]
                    new_ts = self._migrate_timestamp_string(old_ts)
                    if new_ts != old_ts:
                        data["last_updated"] = new_ts
                        migrated_count += 1
                
                await self._atomic_write_json_unlocked(self.hourly_profile_file, data)
                
            _LOGGER.info(f"hourly_profile.json: {migrated_count} timestamps migrated.")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Migration of hourly_profile.json failed: {e}", exc_info=True)
            return False


    async def _migrate_model_state_timestamps(self) -> bool:
        """Migrates timestamps in model_state.json and sets migration flag."""
        try:
            if not await self._file_exists(self.model_state_file):
                _LOGGER.debug("model_state.json does not exist, creating with migration flag.")
                # New installation - create with flag set
                state = self._model_state_default.copy()
                state["timestamp_migration_to_local"] = True
                await self._atomic_write_json(self.model_state_file, state)
                return True
            
            _LOGGER.info("Migrating model_state.json timestamps...")
            
            async with self._file_lock:
                data = await self._read_json_file(self.model_state_file)
                
                if not data:
                    _LOGGER.debug("model_state.json is empty, creating default.")
                    data = self._model_state_default.copy()
                
                migrated_count = 0
                
                # Migrate last_training
                if "last_training" in data and data["last_training"]:
                    old_ts = data["last_training"]
                    new_ts = self._migrate_timestamp_string(old_ts)
                    if new_ts != old_ts:
                        data["last_training"] = new_ts
                        migrated_count += 1
                
                # Migrate last_updated
                if "last_updated" in data and data["last_updated"]:
                    old_ts = data["last_updated"]
                    new_ts = self._migrate_timestamp_string(old_ts)
                    if new_ts != old_ts:
                        data["last_updated"] = new_ts
                        migrated_count += 1
                
                # Set migration flag
                data["timestamp_migration_to_local"] = True
                
                await self._atomic_write_json_unlocked(self.model_state_file, data)
                
            _LOGGER.info(f"model_state.json: {migrated_count} timestamps migrated, flag set.")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Migration of model_state.json failed: {e}", exc_info=True)
            return False

    # ========================================================================
    # END TIMESTAMP MIGRATION
    # ========================================================================

    # --- DATA STRUCTURE MIGRATION (Remains here as it uses API methods) ---
    async def migrate_data(self) -> bool:
        """Runs data structure migrations for all known files."""
        all_success = True
        _LOGGER.info("Starting data structure migration check...")
        # List of migration functions and their corresponding file paths
        migrations = [
            (self._migrate_prediction_history, self.prediction_history_file),
            (self._migrate_learned_weights, self.learned_weights_file),
            (self._migrate_hourly_profile, self.hourly_profile_file),
            (self._migrate_model_state, self.model_state_file),
            (self._migrate_hourly_samples, self.hourly_samples_file),
        ]
        # Sequentially run migrations for existing files
        for migrate_func, file_path in migrations:
             # Check if file exists before attempting migration
             if await self._file_exists(file_path):
                 try:
                     await migrate_func(file_path) # Pass file path to migration func
                 except Exception as e:
                     _LOGGER.error(f"Migration failed for {file_path.name}: {e}", exc_info=True)
                     all_success = False # Mark failure but continue checking others
             else:
                 _LOGGER.debug(f"Migration skipped for {file_path.name}: File does not exist.")

        if all_success:
             _LOGGER.info("Data structure migration check completed successfully.")
        else:
             _LOGGER.warning("Data structure migration check completed with one or more errors.")
        return all_success


    async def _migrate_model_state(self, file_path: Path) -> None:
        """Migrates the model_state.json file, removing obsolete keys."""
        needs_save = False
        # Use lock for read-modify-write during migration
        async with self._file_lock:
            try:
                data = await self._read_json_file(file_path, default_structure={}) # Read unlocked
                if not data: return # Skip if read failed or empty

                current_version = data.get("version", "1.0")

                # Example Migration: Update version if needed
                if current_version != DATA_VERSION:
                    _LOGGER.info(f"Migrating model state {file_path.name} from v{current_version} to v{DATA_VERSION}")
                    data["version"] = DATA_VERSION
                    needs_save = True

                # Example Migration: Remove obsolete 'backfill_run' key
                if "backfill_run" in data:
                    _LOGGER.info(f"Removing obsolete 'backfill_run' key from {file_path.name}.")
                    data.pop("backfill_run")
                    needs_save = True

                # Add other migration steps here if needed for future versions...

                # Save if changes were made
                if needs_save:
                    data["last_updated"] = dt_util.now().isoformat()  # OPTION A: LOCAL TIME
                    await self._atomic_write_json_unlocked(file_path, data) # Write unlocked
                    _LOGGER.info(f"Migration successful for {file_path.name}.")
                else:
                    _LOGGER.debug(f"{file_path.name} is already up to date (v{DATA_VERSION}).")

            except Exception as e:
                _LOGGER.error(f"Error during migration of {file_path.name}: {e}", exc_info=True)
                raise # Re-raise to indicate migration failure


    # Placeholder migration functions (only check version)
    async def _migrate_prediction_history(self, file_path: Path) -> None:
        await self._check_and_update_version(file_path)

    async def _migrate_learned_weights(self, file_path: Path) -> None:
        await self._check_and_update_version(file_path)

    async def _migrate_hourly_profile(self, file_path: Path) -> None:
        await self._check_and_update_version(file_path)

    async def _migrate_hourly_samples(self, file_path: Path) -> None:
        await self._check_and_update_version(file_path)


    async def _check_and_update_version(self, file_path: Path) -> None:
        """Generic function to check and update the version field in JSON files."""
        needs_save = False
        # Use lock for read-modify-write
        async with self._file_lock:
            try:
                data = await self._read_json_file(file_path, default_structure={}) # Read unlocked
                if not data: return # Skip if read failed or empty

                current_version = data.get("version", "1.0") # Default to 1.0 if missing

                if current_version != DATA_VERSION:
                    _LOGGER.info(f"Updating version for {file_path.name} from v{current_version} to v{DATA_VERSION}")
                    data["version"] = DATA_VERSION
                    needs_save = True

                # Save if version was updated
                if needs_save:
                    data["last_updated"] = dt_util.now().isoformat()  # OPTION A: LOCAL TIME
                    await self._atomic_write_json_unlocked(file_path, data) # Write unlocked
                else:
                    _LOGGER.debug(f"{file_path.name} version is already up to date (v{DATA_VERSION}).")

            except Exception as e:
                _LOGGER.error(f"Failed to check/update version for {file_path.name}: {e}", exc_info=True)
                raise # Re-raise to indicate migration failure

    # ========================================================================
    # DUPLICATE CLEANUP & PERSISTENCE (Option 1 + 2)
    # ========================================================================

    async def cleanup_duplicate_samples(self) -> Dict[str, int]:
        """
        Entfernt Duplikate aus hourly_samples.json basierend auf Timestamp.
        Behere Duplikate.
        Erstellt automatisch ein Backup vor der Bereinigung.
        
        Returns:
            Dict mit 'removed' (Anzahl entfernter Duplikate) und 'remaining' (verbleibende Samples)
        """
        try:
            _LOGGER.info("Starting duplicate sample cleanup...")
            
            # 1. Erstelle Backup
            backup_success = await self._create_samples_backup()
            if not backup_success:
                _LOGGER.warning("Backup vor Cleanup fehlgeschlagen, fahre trotzdem fort.")
            
            async with self._file_lock:
                # 2. Lese aktuelle Daten
                samples_data = await self._read_json_file(
                    self.hourly_samples_file,
                    default_structure=self._hourly_samples_default
                )
                
                samples = samples_data.get('samples', [])
                if not samples:
                    _LOGGER.info("Keine Samples vorhanden, Cleanup skipped.")
                    return {'removed': 0, 'remaining': 0}
                
                original_count = len(samples)
                
                # 3. Entferne Duplikate (behalte ersten Eintrag)
                seen_timestamps = set()
                cleaned = []
                duplicates_removed = 0
                
                for sample in samples:
                    ts = sample.get('timestamp')
                    if ts and ts not in seen_timestamps:
                        seen_timestamps.add(ts)
                        cleaned.append(sample)
                    else:
                        duplicates_removed += 1
                        _LOGGER.debug(f"Duplicate entfernt: {ts}")
                
                # 4. Validiere bereinigteData
                if not self._validate_cleaned_samples(cleaned):
                    raise DataIntegrityException("Validierung der bereinigten Samples fehlgeschlagen.")
                
                # 5. Aktualisiere Struktur
                samples_data['samples'] = cleaned
                samples_data['count'] = len(cleaned)
                samples_data['last_updated'] = dt_util.now().isoformat()
                
                # 6. Schreibe zurÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ck
                await self._atomic_write_json_unlocked(self.hourly_samples_file, samples_data)
            
            result = {'removed': duplicates_removed, 'remaining': len(cleaned)}
            _LOGGER.info(
                f"Duplicate Cleanup abgeschlossen: {duplicates_removed} Duplikate entfernt, "
                f"{len(cleaned)} Samples verbleiben (von {original_count} )."
            )
            return result
            
        except Exception as e:
            _LOGGER.error(f"Fehler bei Duplicate Cleanup: {e}", exc_info=True)
            raise DataIntegrityException(f"Duplicate Cleanup fehlgeschlagen: {str(e)}")


    async def cleanup_zero_production_samples(self) -> Dict[str, int]:
        """
        Entfernt alle Samples mit production=0 aus hourly_samples.json.
        Wird nachts um 2:00 Uhr , um nur relevante Produktionsdaten zu behalten.
        Erstellt automatisch ein Backup vor der Bereinigung.
        
        Returns:
            Dict mit 'removed' (Anzahl entfernter Samples) und 'remaining' (verbleibende Samples)
        """
        try:
            _LOGGER.info("Starting zero-production sample cleanup...")
            
            backup_success = await self._create_samples_backup()
            if not backup_success:
                _LOGGER.warning("Backup vor Cleanup fehlgeschlagen, fahre trotzdem fort.")
            
            async with self._file_lock:
                samples_data = await self._read_json_file(
                    self.hourly_samples_file,
                    default_structure=self._hourly_samples_default
                )
                
                samples = samples_data.get('samples', [])
                if not samples:
                    _LOGGER.info("Keine Samples vorhanden, Cleanup skipped")
                    return {'removed': 0, 'remaining': 0}
                
                original_count = len(samples)
                
                cleaned = []
                zero_samples_removed = 0
                
                for sample in samples:
                    actual_kwh = sample.get('actual_kwh', 0.0)
                    try:
                        actual_kwh_float = float(actual_kwh)
                        if actual_kwh_float > 0.01:  # Threshold: 0.01 kWh = 10 Wh
                            cleaned.append(sample)
                        else:
                            zero_samples_removed += 1
                            _LOGGER.debug(f"Zero-production sample removed: {sample.get('timestamp')}")
                    except (ValueError, TypeError):
                        _LOGGER.warning(f"Invalid actual_kwh value: {actual_kwh}, keeping sample")
                        cleaned.append(sample)
                
                if not self._validate_cleaned_samples(cleaned):
                    raise DataIntegrityException("Validierung der bereinigten Samples fehlgeschlagen.")
                
                samples_data['samples'] = cleaned
                samples_data['count'] = len(cleaned)
                samples_data['last_updated'] = dt_util.now().isoformat()
                
                await self._atomic_write_json_unlocked(self.hourly_samples_file, samples_data)
            
            result = {'removed': zero_samples_removed, 'remaining': len(cleaned)}
            _LOGGER.info(
                f"Zero-production Cleanup abgeschlossen: {zero_samples_removed} Samples entfernt, "
                f"{len(cleaned)} Samples verbleiben (von {original_count} ursprueglich)."
            )
            return result
            
        except Exception as e:
            _LOGGER.error(f"Fehler bei Zero-production Cleanup: {e}", exc_info=True)
            raise DataIntegrityException(f"Zero-production Cleanup fehlgeschlagen: {str(e)}")

    async def _create_samples_backup(self) -> bool:
        """Erstellt Backup von hourly_samples.json vor Cleanup."""
        try:
            if not await self._file_exists(self.hourly_samples_file):
                _LOGGER.debug("Keine hourly_samples.json vorhanden, no backup needed.")
                return True
            
            backup_dir = self.data_dir / "backups"
            await self._ensure_directory_exists(backup_dir)
            
            timestamp_str = dt_util.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"hourly_samples_backup_{timestamp_str}.json"
            backup_path = backup_dir / backup_filename
            
            await self.hass.async_add_executor_job(
                lambda: backup_path.write_bytes(self.hourly_samples_file.read_bytes())
            )
            
            _LOGGER.info(f"Backup erstellt: {backup_filename}")
            
            # Cleanup old backups after creating new one
            await self._cleanup_old_backups()
            
            return True
            
        except Exception as e:
            _LOGGER.warning(f"Backup-Erstellung fehlgeschlagen: {e}")
            return False

    async def _cleanup_old_backups(self) -> None:
        """
        Bereinigt alte Backup-Dateien basierend auf Retention-Richtlinien.
        BehÃƒÂ¤lt maximal MAX_BACKUP_FILES neueste Backups und lÃƒÂ¶scht Backups ÃƒÂ¤lter als BACKUP_RETENTION_DAYS.
        """
        try:
            backup_dir = self.data_dir / "backups"
            if not await self._file_exists(backup_dir):
                return
            
            # Get all backup files
            backup_files = await self.hass.async_add_executor_job(
                lambda: sorted(
                    backup_dir.glob("hourly_samples_backup_*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True  # Newest first
                )
            )
            
            if not backup_files:
                return
            
            now = dt_util.now()
            cutoff_date = now - timedelta(days=BACKUP_RETENTION_DAYS)
            removed_count = 0
            
            # Keep only MAX_BACKUP_FILES newest files
            for i, backup_file in enumerate(backup_files):
                try:
                    # Remove if beyond max count
                    if i >= MAX_BACKUP_FILES:
                        await self.hass.async_add_executor_job(backup_file.unlink)
                        removed_count += 1
                        _LOGGER.debug(f"Removed backup (exceeded max count): {backup_file.name}")
                        continue
                    
                    # Remove if too old
                    file_mtime = await self.hass.async_add_executor_job(
                        lambda f=backup_file: datetime.fromtimestamp(f.stat().st_mtime, tz=dt_util.DEFAULT_TIME_ZONE)
                    )
                    
                    if file_mtime < cutoff_date:
                        await self.hass.async_add_executor_job(backup_file.unlink)
                        removed_count += 1
                        _LOGGER.debug(f"Removed backup (too old): {backup_file.name}")
                        
                except Exception as e:
                    _LOGGER.warning(f"Failed to remove backup {backup_file.name}: {e}")
                    continue
            
            if removed_count > 0:
                _LOGGER.info(f"Backup cleanup: {removed_count} old backups removed, {len(backup_files) - removed_count} retained")
                
        except Exception as e:
            _LOGGER.warning(f"Backup cleanup failed: {e}")

    def _validate_cleaned_samples(self, samples: List[Dict[str, Any]]) -> bool:
        """Validiert berein igte Samples vor dem Speichern."""
        try:
            if not isinstance(samples, list):
                _LOGGER.error("Bereingte Samples sind keine Liste.")
                return False
            
            # PrÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼fe ob alle Samples die required keys haben
            required_keys = ['timestamp', 'actual_kwh', 'weather_data', 'sensor_data']
            for sample in samples:
                if not all(k in sample for k in required_keys):
                    _LOGGER.error(f"Sample fehlt required keys: {sample.get('timestamp', 'unknown')}")
                    return False
            
            # PrÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼fe ob Timestamps unique sind
            timestamps = [s.get('timestamp') for s in samples]
            if len(timestamps) != len(set(timestamps)):
                _LOGGER.error("Bereinigtte Samples enthalten noch Duplikate!")
                return False
            
            return True
            
        except Exception as e:
            _LOGGER.error(f"Validierung fehlgeschlagen: {e}", exc_info=True)
            return False

    async def get_last_collected_hour(self) -> Optional[int]:
        """
        Liest die zuletzt erfolgreich gesammelte Stunde aus model_state.json.
        
        Returns:
            int (0-23) oder None wenn nicht gesetzt
        """
        try:
            state = await self.get_model_state()
            last_hour = state.get('last_collected_hour')
            if last_hour is not None and isinstance(last_hour, int) and 0 <= last_hour <= 23:
                return last_hour
            return None
        except Exception as e:
            _LOGGER.debug(f"Fehler beim Lesen von last_collected_hour: {e}")
            return None

    async def set_last_collected_hour(self, hour: int) -> None:
        """
        Speichert die zuletzt erfolgreich gesammelte Stunde in model_state.json.
        
        Args:
            hour: Stunde (0-23)
        """
        try:
            if not isinstance(hour, int) or hour < 0 or hour > 23:
                _LOGGER.error(f"not vaild: {hour}")
                return
            
            async with self._file_lock:
                state = await self.get_model_state()
                state['last_collected_hour'] = hour
                state['last_updated'] = dt_util.now().isoformat()
                await self._atomic_write_json_unlocked(self.model_state_file, state)
            
            _LOGGER.debug(f"last_collected_hour gesetzt auf: {hour}")
            
        except Exception as e:
            _LOGGER.error(f"Fehler beim Setzen von last_collected_hour: {e}", exc_info=True)

    async def _write_json_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Wrapper method for backward compatibility.
        Redirects to _atomic_write_json for file operations.
        """
        await self._atomic_write_json(file_path, data)

    async def load_weather_forecast_cache(self) -> Optional[Dict[str, Any]]:
        """
        Load weather forecast from cache file.
        Returns None if file doesn't exist or is invalid.
        """
        cache_file = self.data_dir / "weather_forecast_cache.json"
        if not cache_file.exists():
            _LOGGER.debug("Weather forecast cache file does not exist")
            return None
        
        try:
            data = await self._read_json_file(cache_file, default_structure={})
            if not data or not data.get("forecast_hours"):
                _LOGGER.debug("Weather forecast cache is empty or invalid")
                return None
            
            _LOGGER.debug(f"Loaded weather forecast cache: {len(data.get('forecast_hours', []))} hours")
            return data
            
        except Exception as e:
            _LOGGER.error(f"Failed to load weather forecast cache: {e}", exc_info=True)
            return None

    async def save_weather_forecast_cache(self, forecast_hours: List[Dict[str, Any]]) -> bool:
        """
        Save weather forecast with smart merge and deduplication.
        Ensures today + tomorrow coverage with quality validation.
        Uses LOCAL time for all operations.
        
        Args:
            forecast_hours: List of processed forecast entries with 'datetime' (UTC) and 'local_datetime' fields
            
        Returns:
            True if save successful, False otherwise
        """
        cache_file = self.data_dir / "weather_forecast_cache.json"
        
        try:
            # Load existing cache
            old_cache = await self.load_weather_forecast_cache()
            old_hours = old_cache.get("forecast_hours", []) if old_cache else []
            
            # Merge and deduplicate (using UTC datetime as key)
            merged = self._merge_and_deduplicate_forecast(old_hours, forecast_hours)
            
            # Filter time window: now-1h to now+48h (LOCAL time)
            filtered = self._filter_forecast_timewindow(merged)
            
            # Validate quality (today and tomorrow coverage)
            quality = self._validate_forecast_quality(filtered)
            
            # Build cache structure
            cache_data = {
                "version": DATA_VERSION,
                "last_updated": dt_util.now().isoformat(),  # LOCAL TIME
                "last_refresh": dt_util.now().isoformat(),  # LOCAL TIME
                "data_quality": quality,
                "forecast_hours": filtered
            }
            
            # Save atomically
            await self._atomic_write_json(cache_file, cache_data)
            
            _LOGGER.info(
                f"Saved weather forecast cache: {len(filtered)} hours, "
                f"today={quality['today_hours']}h, tomorrow={quality['tomorrow_hours']}h, "
                f"complete={quality['complete']}"
            )
            
            return True
            
        except Exception as e:
            _LOGGER.error(f"Failed to save weather forecast cache: {e}", exc_info=True)
            return False

    def _merge_and_deduplicate_forecast(
        self, 
        old_hours: List[Dict[str, Any]], 
        new_hours: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge old and new forecast data, deduplicate by UTC datetime.
        Newer entries overwrite older ones.
        
        Args:
            old_hours: Existing forecast data
            new_hours: New forecast data to merge
            
        Returns:
            Merged and deduplicated list sorted by datetime
        """
        # Use UTC datetime as deduplication key (unique identifier)
        all_hours = {}
        
        # Add old data first
        for entry in old_hours:
            dt_key = entry.get("datetime")  # UTC datetime string
            if dt_key:
                all_hours[dt_key] = entry
        
        # Add new data (overwrites duplicates)
        for entry in new_hours:
            dt_key = entry.get("datetime")  # UTC datetime string
            if dt_key:
                all_hours[dt_key] = entry
        
        # Convert back to list, sorted by UTC datetime
        merged = sorted(all_hours.values(), key=lambda x: x.get("datetime", ""))
        
        _LOGGER.debug(
            f"Merged forecast: {len(old_hours)} old + {len(new_hours)} new = {len(merged)} unique hours"
        )
        
        return merged

    def _filter_forecast_timewindow(self, forecast_hours: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter forecast to time window: now-30d to now+48h (LOCAL time).
        Keeps 30 days of historical forecasts for ML training.
        
        Args:
            forecast_hours: List of forecast entries with 'local_datetime' field
            
        Returns:
            Filtered list within time window
        """
        now_local = dt_util.now()  # Current time in LOCAL timezone
        start_time = now_local - timedelta(days=30)  # 30 days history
        end_time = now_local + timedelta(hours=48)
        
        filtered = []
        for entry in forecast_hours:
            local_dt_value = entry.get("local_datetime")
            if not local_dt_value:
                continue
            
            try:
                # Handle both datetime objects and ISO strings
                if isinstance(local_dt_value, str):
                    local_dt = dt_util.parse_datetime(local_dt_value)
                    if not local_dt:
                        continue
                else:
                    local_dt = local_dt_value
                
                # Ensure timezone-aware for comparison
                if local_dt.tzinfo is None:
                    local_dt = dt_util.as_local(local_dt)
                
                if start_time <= local_dt <= end_time:
                    filtered.append(entry)
                    
            except Exception as e:
                _LOGGER.debug(f"Failed to parse datetime for filtering: {e}")
                continue
        
        _LOGGER.debug(
            f"Filtered forecast: {len(forecast_hours)} -> {len(filtered)} hours "
            f"(window: -1h to +48h LOCAL time)"
        )
        
        return filtered

    def _validate_forecast_quality(self, forecast_hours: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate forecast data quality - ensure today and tomorrow coverage.
        Uses LOCAL time for date comparisons.
        
        Args:
            forecast_hours: List of forecast entries with 'local_datetime' field
            
        Returns:
            Quality metrics dictionary
        """
        now_local = dt_util.now()  # Current time in LOCAL timezone
        today_date = now_local.date()
        tomorrow_date = today_date + timedelta(days=1)
        
        today_hours = 0
        tomorrow_hours = 0
        
        for entry in forecast_hours:
            local_dt_value = entry.get("local_datetime")
            if not local_dt_value:
                continue
            
            try:
                # Handle both datetime objects and ISO strings
                if isinstance(local_dt_value, str):
                    local_dt = dt_util.parse_datetime(local_dt_value)
                    if not local_dt:
                        continue
                else:
                    local_dt = local_dt_value
                
                # Ensure timezone-aware
                if local_dt.tzinfo is None:
                    local_dt = dt_util.as_local(local_dt)
                
                entry_date = local_dt.date()
                
                if entry_date == today_date:
                    today_hours += 1
                elif entry_date == tomorrow_date:
                    tomorrow_hours += 1
                    
            except Exception as e:
                _LOGGER.debug(f"Failed to parse datetime for validation: {e}")
                continue
        
        # Minimum thresholds for quality
        min_today_hours = 6
        min_tomorrow_hours = 12
        
        complete = today_hours >= min_today_hours and tomorrow_hours >= min_tomorrow_hours
        
        quality = {
            "today_hours": today_hours,
            "tomorrow_hours": tomorrow_hours,
            "complete": complete,
            "min_today_hours": min_today_hours,
            "min_tomorrow_hours": min_tomorrow_hours
        }
        
        _LOGGER.debug(
            f"Forecast quality: today={today_hours}h (min {min_today_hours}), "
            f"tomorrow={tomorrow_hours}h (min {min_tomorrow_hours}), complete={complete}"
        )
        
        return quality
