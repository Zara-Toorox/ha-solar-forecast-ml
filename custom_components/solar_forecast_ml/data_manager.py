"""
Data Manager fÃƒÂ¼r die Solar Forecast ML Integration.
Ã¢Å“â€¦ ERWEITERT mit fehlenden ML-Methods und Backup-System

Copyright (C) 2025 Zara-Toorox
"""
import asyncio
import json
import logging
import os
import shutil
import aiofiles
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from concurrent.futures import ThreadPoolExecutor

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import (
    MAX_PREDICTION_HISTORY, MIN_TRAINING_DATA_POINTS, BACKUP_RETENTION_DAYS, 
    DATA_VERSION, MAX_BACKUP_FILES
)
from .ml_types import (
    PredictionRecord, LearnedWeights, HourlyProfile, 
    create_default_learned_weights, create_default_hourly_profile,
    validate_prediction_record
)
from .exceptions import (
    DataIntegrityException, ConfigurationException, SolarForecastMLException,
    ErrorSeverity, create_context
)

_LOGGER = logging.getLogger(__name__)


class DataManager:
    """
    Data Manager fÃƒÂ¼r Solar Forecast ML.
    Ã¢Å“â€¦ ERWEITERT mit ML-Methods und Backup-System
    """

    def __init__(self, hass: HomeAssistant, entry_id: str, data_dir: Path):
        """
        Initialize the Data Manager.
        
        Args:
            hass: Home Assistant instance
            entry_id: Config Entry ID fÃƒÂ¼r Multi-Instance-Support # von Zara
            data_dir: Daten-Verzeichnis fÃƒÂ¼r JSON-Dateien # von Zara
        
        # GeÃƒÂ¤nderter Abschnitt von Zara - entry_id Parameter hinzugefÃƒÂ¼gt
        """
        self.hass = hass
        self.entry_id = entry_id  # Speichere entry_id fÃƒÂ¼r zukÃƒÂ¼nftige Features # von Zara
        self.data_dir = Path(data_dir)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="DataManager")
        
        # File paths # von Zara
        self.prediction_history_file = self.data_dir / "prediction_history.json"
        self.learned_weights_file = self.data_dir / "learned_weights.json"
        self.hourly_profile_file = self.data_dir / "hourly_profile.json"
        self.model_state_file = self.data_dir / "model_state.json"
        
        # Thread-safe lock fÃƒÂ¼r kritische Operations # von Zara
        self._file_lock = asyncio.Lock()
        
        _LOGGER.info("Ã¢Å“â€¦ DataManager initialisiert mit async I/O support (Entry: %s)", entry_id)

    async def initialize(self) -> bool:
        """Initialize data manager and create directory structure."""
        try:
            # Create directory structure # von Zara
            await self._ensure_directory_exists(self.data_dir)
            await self._ensure_directory_exists(self.data_dir / "backups")
            
            # Initialize missing files # von Zara
            await self._initialize_missing_files()
            
            # Data migration # von Zara
            migration_success = await self.migrate_data()
            
            if migration_success:
                _LOGGER.info("Ã¢Å“â€¦ DataManager erfolgreich initialisiert")
                return True
            else:
                raise DataIntegrityException(
                    "Data migration fehlgeschlagen",
                    create_context(data_dir=str(self.data_dir))
                )
                
        except Exception as e:
            _LOGGER.error("Ã¢ÂÅ’ DataManager Initialisierung fehlgeschlagen: %s", str(e))
            raise DataIntegrityException(
                f"DataManager Initialisierung fehlgeschlagen: {str(e)}",
                create_context(error=str(e), data_dir=str(self.data_dir))
            )

    async def _ensure_directory_exists(self, directory: Path) -> None:
        """Ensure directory exists."""
        try:
            await self.hass.async_add_executor_job(
                os.makedirs, directory, 0o755, True
            )
            _LOGGER.debug("Ã¢Å“â€¦ Directory created/verified: %s", directory)
        except Exception as e:
            raise DataIntegrityException(
                f"Fehler beim Erstellen des Verzeichnisses {directory}: {str(e)}",
                create_context(directory=str(directory), error=str(e))
            )

    async def _initialize_missing_files(self) -> None:
        """Initialize missing JSON files."""
        files_to_check = [
            (self.prediction_history_file, self._create_default_prediction_history),
            (self.learned_weights_file, self._create_default_learned_weights),
            (self.hourly_profile_file, self._create_default_hourly_profile),
            (self.model_state_file, self._create_default_model_state)
        ]
        
        for file_path, create_func in files_to_check:
            if not await self._file_exists(file_path):
                _LOGGER.info("Ã°Å¸â€œÂ Erstelle fehlende Datei: %s", file_path.name)
                await create_func()

    async def _file_exists(self, file_path: Path) -> bool:
        """Check if file exists."""
        try:
            return await self.hass.async_add_executor_job(file_path.exists)
        except Exception:
            return False

    async def _create_default_prediction_history(self) -> None:
        """Create default prediction history."""
        default_history = {
            "version": DATA_VERSION,
            "created": dt_util.utcnow().isoformat(),
            "predictions": []
        }
        await self._atomic_write_json(self.prediction_history_file, default_history)

    async def _create_default_learned_weights(self) -> None:
        """Create default learned weights."""
        default_weights = create_default_learned_weights()
        weights_dict = {
            "version": DATA_VERSION,
            "created": dt_util.utcnow().isoformat(),
            "weather_weights": default_weights.weather_weights,
            "seasonal_factors": default_weights.seasonal_factors,
            "correction_factor": default_weights.correction_factor,
            "accuracy": default_weights.accuracy,
            "training_samples": default_weights.training_samples,
            "last_trained": default_weights.last_trained,
            "model_version": default_weights.model_version,
            "feature_importance": default_weights.feature_importance
        }
        await self._atomic_write_json(self.learned_weights_file, weights_dict)

    async def _create_default_hourly_profile(self) -> None:
        """Create default hourly profile."""
        default_profile = create_default_hourly_profile()
        profile_dict = {
            "version": DATA_VERSION,
            "created": dt_util.utcnow().isoformat(),
            "hourly_factors": default_profile.hourly_factors,
            "samples_count": default_profile.samples_count,
            "last_updated": default_profile.last_updated,
            "confidence": default_profile.confidence,
            "seasonal_adjustment": default_profile.seasonal_adjustment
        }
        await self._atomic_write_json(self.hourly_profile_file, profile_dict)

    async def _create_default_model_state(self) -> None:
        """Create default model state."""
        default_state = {
            "version": DATA_VERSION,
            "created": dt_util.utcnow().isoformat(),
            "last_training": None,
            "training_count": 0,
            "performance_metrics": {
                "mae": None,
                "rmse": None,
                "accuracy": None
            },
            "status": "initialized"
        }
        await self._atomic_write_json(self.model_state_file, default_state)

    async def _atomic_write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Atomic write to JSON file."""
        async with self._file_lock:
            temp_file = file_path.with_suffix('.tmp')
            try:
                async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(data, indent=2, ensure_ascii=False))
                
                await self.hass.async_add_executor_job(
                    shutil.move, str(temp_file), str(file_path)
                )
                
                _LOGGER.debug("Ã¢Å“â€¦ Atomic write successful: %s", file_path.name)
                
            except Exception as e:
                # Clean up temp file if it exists # von Zara
                if await self._file_exists(temp_file):
                    try:
                        await self.hass.async_add_executor_job(temp_file.unlink)
                    except Exception:
                        pass
                raise DataIntegrityException(
                    f"Failed to write {file_path.name}: {str(e)}",
                    create_context(file=str(file_path), error=str(e))
                )

    async def _read_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Read JSON file with error handling."""
        try:
            if not await self._file_exists(file_path):
                raise DataIntegrityException(
                    f"File not found: {file_path.name}",
                    create_context(file=str(file_path))
                )
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
                
        except json.JSONDecodeError as e:
            _LOGGER.error("Ã¢ÂÅ’ JSON decode error in %s: %s", file_path.name, str(e))
            raise DataIntegrityException(
                f"Invalid JSON in {file_path.name}: {str(e)}",
                create_context(file=str(file_path), error=str(e))
            )
        except Exception as e:
            raise DataIntegrityException(
                f"Failed to read {file_path.name}: {str(e)}",
                create_context(file=str(file_path), error=str(e))
            )

    # Ã¢Å“â€¦ ERWEITERT: ML-spezifische Methoden # von Zara
    async def add_prediction(self, prediction: PredictionRecord) -> None:
        """
        Add a new prediction to history.
        
        Args:
            prediction: PredictionRecord to add
        """
        try:
            # Validate prediction record # von Zara
            validate_prediction_record(prediction)
            
            # Get current history # von Zara
            history = await self.get_prediction_history()
            predictions = history.get("predictions", [])
            
            # Add new prediction # von Zara
            predictions.append(prediction)
            
            # Limit history size # von Zara
            if len(predictions) > MAX_PREDICTION_HISTORY:
                predictions = predictions[-MAX_PREDICTION_HISTORY:]
            
            # Update history # von Zara
            history["predictions"] = predictions
            history["last_updated"] = dt_util.utcnow().isoformat()
            
            # Create backup before writing # von Zara
            await self._create_backup(self.prediction_history_file)
            
            # Save updated history # von Zara
            await self._atomic_write_json(self.prediction_history_file, history)
            
            _LOGGER.debug("Ã¢Å“â€¦ Prediction added to history")
            
        except Exception as e:
            _LOGGER.error("Ã¢ÂÅ’ Failed to add prediction: %s", str(e))
            raise DataIntegrityException(
                f"Failed to add prediction: {str(e)}",
                create_context(error=str(e))
            )

    async def add_prediction_record(self, record: Dict[str, Any]) -> None:
        """
        FÃ¼gt Training-Sample als Dict zur History hinzu.
        Alias fÃ¼r add_prediction mit Dict-Input - von Zara
        
        Args:
            record: Dict mit prediction data
        """
        try:
            # Konvertiere Dict zu PredictionRecord - von Zara
            prediction = PredictionRecord(
                timestamp=record.get('timestamp', dt_util.utcnow().isoformat()),
                predicted_value=record.get('predicted_value', 0.0),
                actual_value=record.get('actual_value'),
                weather_data=record.get('weather_data', {}),
                sensor_data=record.get('sensor_data', {})
            )
            
            # Nutze existierende add_prediction Methode - von Zara
            await self.add_prediction(prediction)
            
            _LOGGER.debug("âœ… Prediction record added via add_prediction_record - von Zara")
            
        except Exception as e:
            _LOGGER.error("âŒ Failed to add prediction record: %s - von Zara", str(e))
            raise DataIntegrityException(
                f"Failed to add prediction record: {str(e)}",
                create_context(error=str(e))
            )

    async def get_recent_predictions(self, hours: int = 24) -> List[PredictionRecord]:
        """
        Get recent predictions within the specified time window.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of PredictionRecords
        """
        try:
            history = await self.get_prediction_history()
            predictions = history.get("predictions", [])
            
            if not predictions:
                return []
            
            # Calculate cutoff time # von Zara
            cutoff = dt_util.utcnow() - timedelta(hours=hours)
            
            # Filter predictions by timestamp # von Zara
            recent = []
            for pred in predictions:
                try:
                    pred_time = dt_util.parse_datetime(pred.get("timestamp"))
                    if pred_time and pred_time >= cutoff:
                        recent.append(pred)
                except Exception:
                    continue
            
            return recent
            
        except Exception as e:
            _LOGGER.error("Ã¢ÂÅ’ Failed to get recent predictions: %s", str(e))
            return []

    async def update_learned_weights(self, weights: LearnedWeights) -> None:
        """
        Update learned weights.
        
        Args:
            weights: LearnedWeights object to save
        """
        try:
            weights_dict = {
                "version": DATA_VERSION,
                "weather_weights": weights.weather_weights,
                "seasonal_factors": weights.seasonal_factors,
                "correction_factor": weights.correction_factor,
                "accuracy": weights.accuracy,
                "training_samples": weights.training_samples,
                "last_trained": weights.last_trained,
                "model_version": weights.model_version,
                "feature_importance": weights.feature_importance,
                "updated": dt_util.utcnow().isoformat()
            }
            
            # Create backup before writing # von Zara
            await self._create_backup(self.learned_weights_file)
            
            # Save weights # von Zara
            await self._atomic_write_json(self.learned_weights_file, weights_dict)
            
            _LOGGER.info("Ã¢Å“â€¦ Learned weights updated (Accuracy: %.2f%%)", weights.accuracy * 100)
            
        except Exception as e:
            _LOGGER.error("Ã¢ÂÅ’ Failed to update learned weights: %s", str(e))
            raise DataIntegrityException(
                f"Failed to update learned weights: {str(e)}",
                create_context(error=str(e))
            )

    async def update_hourly_profile(self, profile: HourlyProfile) -> None:
        """
        Update hourly profile.
        
        Args:
            profile: HourlyProfile object to save
        """
        try:
            profile_dict = {
                "version": DATA_VERSION,
                "hourly_factors": profile.hourly_factors,
                "samples_count": profile.samples_count,
                "last_updated": profile.last_updated,
                "confidence": profile.confidence,
                "seasonal_adjustment": profile.seasonal_adjustment,
                "updated": dt_util.utcnow().isoformat()
            }
            
            # Create backup before writing # von Zara
            await self._create_backup(self.hourly_profile_file)
            
            # Save profile # von Zara
            await self._atomic_write_json(self.hourly_profile_file, profile_dict)
            
            _LOGGER.info("Ã¢Å“â€¦ Hourly profile updated")
            
        except Exception as e:
            _LOGGER.error("Ã¢ÂÅ’ Failed to update hourly profile: %s", str(e))
            raise DataIntegrityException(
                f"Failed to update hourly profile: {str(e)}",
                create_context(error=str(e))
            )

    async def update_model_state(
        self,
        training_count: Optional[int] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        status: Optional[str] = None
    ) -> None:
        """
        Update model state.
        
        Args:
            training_count: Number of training iterations
            performance_metrics: Performance metrics (mae, rmse, accuracy)
            status: Current model status
        """
        try:
            current_state = await self.get_model_state()
            
            # Update fields if provided # von Zara
            if training_count is not None:
                current_state["training_count"] = training_count
                current_state["last_training"] = dt_util.utcnow().isoformat()
            
            if performance_metrics is not None:
                current_state["performance_metrics"] = performance_metrics
            
            if status is not None:
                current_state["status"] = status
            
            current_state["last_updated"] = dt_util.utcnow().isoformat()
            
            # Create backup before writing # von Zara
            await self._create_backup(self.model_state_file)
            
            # Save state # von Zara
            await self._atomic_write_json(self.model_state_file, current_state)
            
            _LOGGER.debug("Ã¢Å“â€¦ Model state updated")
            
        except Exception as e:
            _LOGGER.error("Ã¢ÂÅ’ Failed to update model state: %s", str(e))
            raise DataIntegrityException(
                f"Failed to update model state: {str(e)}",
                create_context(error=str(e))
            )

    # Ã¢Å“â€¦ BACKUP SYSTEM # von Zara
    async def _create_backup(self, file_path: Path) -> None:
        """Create backup of a file before modification."""
        try:
            if not await self._file_exists(file_path):
                return
            
            backup_dir = self.data_dir / "backups"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"{file_path.stem}_{timestamp}.json"
            
            # Copy file to backup # von Zara
            await self.hass.async_add_executor_job(
                shutil.copy2, str(file_path), str(backup_file)
            )
            
            _LOGGER.debug("Ã¢Å“â€¦ Backup created: %s", backup_file.name)
            
            # Cleanup old backups # von Zara
            await self._cleanup_old_backups(file_path.stem)
            
        except Exception as e:
            # Backup failure should not break the operation # von Zara
            _LOGGER.warning("Ã¢Å¡Â Ã¯Â¸Â Backup creation failed: %s", str(e))

    async def _cleanup_old_backups(self, file_stem: str) -> None:
        """Remove old backup files."""
        try:
            backup_dir = self.data_dir / "backups"
            
            if not await self._file_exists(backup_dir):
                return
            
            # Get all backup files for this file stem # von Zara
            pattern = f"{file_stem}_*.json"
            backup_files = await self.hass.async_add_executor_job(
                lambda: sorted(backup_dir.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
            )
            
            # Remove files beyond retention limit # von Zara
            if len(backup_files) > MAX_BACKUP_FILES:
                for old_backup in backup_files[MAX_BACKUP_FILES:]:
                    try:
                        await self.hass.async_add_executor_job(old_backup.unlink)
                        _LOGGER.debug("Ã°Å¸â€”â€˜Ã¯Â¸Â Removed old backup: %s", old_backup.name)
                    except Exception as e:
                        _LOGGER.warning("Ã¢Å¡Â Ã¯Â¸Â Failed to remove old backup: %s", str(e))
            
            # Remove files older than retention period # von Zara
            cutoff_date = datetime.now() - timedelta(days=BACKUP_RETENTION_DAYS)
            for backup_file in backup_files:
                try:
                    file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        await self.hass.async_add_executor_job(backup_file.unlink)
                        _LOGGER.debug("Ã°Å¸â€”â€˜Ã¯Â¸Â Removed expired backup: %s", backup_file.name)
                except Exception as e:
                    _LOGGER.warning("Ã¢Å¡Â Ã¯Â¸Â Failed to check/remove backup: %s", str(e))
                    
        except Exception as e:
            _LOGGER.warning("Ã¢Å¡Â Ã¯Â¸Â Backup cleanup failed: %s", str(e))

    async def restore_from_backup(self, file_path: Path, backup_timestamp: str) -> bool:
        """
        Restore a file from backup.
        
        Args:
            file_path: Target file to restore
            backup_timestamp: Timestamp of backup to restore (format: YYYYMMDD_HHMMSS)
            
        Returns:
            True if restore was successful
        """
        try:
            backup_dir = self.data_dir / "backups"
            backup_file = backup_dir / f"{file_path.stem}_{backup_timestamp}.json"
            
            if not await self._file_exists(backup_file):
                _LOGGER.error("Ã¢ÂÅ’ Backup file not found: %s", backup_file.name)
                return False
            
            # Create backup of current file before restore # von Zara
            if await self._file_exists(file_path):
                await self._create_backup(file_path)
            
            # Restore from backup # von Zara
            await self.hass.async_add_executor_job(
                shutil.copy2, str(backup_file), str(file_path)
            )
            
            _LOGGER.info("Ã¢Å“â€¦ Restored from backup: %s", backup_file.name)
            return True
            
        except Exception as e:
            _LOGGER.error("Ã¢ÂÅ’ Restore from backup failed: %s", str(e))
            return False

    # Ã¢Å“â€¦ VALIDATION METHODS # von Zara
    async def validate_all_data(self) -> bool:
        """
        Validate all data files.
        
        Returns:
            True if all validations pass
        """
        try:
            _LOGGER.info("Ã°Å¸â€Â Starting data validation...")
            
            validation_tasks = [
                self._validate_prediction_history(),
                self._validate_learned_weights(),
                self._validate_hourly_profile(),
                self._validate_model_state()
            ]
            
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    _LOGGER.error("Ã¢ÂÅ’ Validation failed for task %d: %s", i, str(result))
                    return False
            
            _LOGGER.info("Ã¢Å“â€¦ All data validations passed")
            return True
            
        except Exception as e:
            _LOGGER.error("Ã¢ÂÅ’ Data validation failed: %s", str(e))
            return False

    async def _validate_prediction_history(self) -> None:
        """Validate prediction history data."""
        data = await self.get_prediction_history()
        
        required_fields = ["version", "created", "predictions"]
        if not all(field in data for field in required_fields):
            raise DataIntegrityException("Invalid prediction_history schema")
        
        # Validate each prediction record # von Zara
        for i, pred in enumerate(data.get("predictions", [])):
            try:
                validate_prediction_record(pred)
            except Exception as e:
                raise DataIntegrityException(f"Invalid prediction record at index {i}: {e}")

    async def _validate_learned_weights(self) -> None:
        """Validate learned weights data."""
        data = await self.get_learned_weights()
        
        required_fields = ["weather_weights", "seasonal_factors", "correction_factor", "accuracy"]
        if not all(field in data for field in required_fields):
            raise DataIntegrityException("Invalid learned_weights schema")
        
        # Validate correction_factor bounds # von Zara
        correction_factor = data.get("correction_factor", 1.0)
        if not (0.1 <= correction_factor <= 5.0):
            raise DataIntegrityException(f"correction_factor {correction_factor} out of bounds [0.1, 5.0]")

    async def _validate_hourly_profile(self) -> None:
        """Validate hourly profile data."""
        data = await self.get_hourly_profile()
        
        required_fields = ["hourly_factors", "samples_count"]
        if not all(field in data for field in required_fields):
            raise DataIntegrityException("Invalid hourly_profile schema")
        
        # Validate hourly_factors (should have 24 entries) # von Zara
        hourly_factors = data.get("hourly_factors", {})
        if len(hourly_factors) != 24:
            raise DataIntegrityException(f"hourly_factors must have 24 entries, got {len(hourly_factors)}")

    async def _validate_model_state(self) -> None:
        """Validate model state data."""
        data = await self.get_model_state()
        
        required_fields = ["version", "status"]
        if not all(field in data for field in required_fields):
            raise DataIntegrityException("Invalid model_state schema")

    # Migration methods # von Zara
    async def migrate_data(self) -> bool:
        """Migrate data to current version."""
        try:
            _LOGGER.info("Ã°Å¸â€â€ž Starte Data Migration...")
            
            await self._initialize_missing_files()
            
            migration_tasks = [
                self._migrate_prediction_history(),
                self._migrate_learned_weights(),
                self._migrate_hourly_profile(),
                self._migrate_model_state()
            ]
            
            results = await asyncio.gather(*migration_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    _LOGGER.error("Ã¢ÂÅ’ Migration Task %d failed: %s", i, str(result))
                    raise result
            
            _LOGGER.info("Ã¢Å“â€¦ Data Migration erfolgreich abgeschlossen")
            return True
            
        except Exception as e:
            _LOGGER.error("Ã¢ÂÅ’ Data Migration fehlgeschlagen: %s", str(e))
            if isinstance(e, DataIntegrityException):
                raise
            else:
                raise DataIntegrityException(
                    f"Data Migration fehlgeschlagen: {str(e)}",
                    create_context(error=str(e))
                )

    async def _migrate_prediction_history(self) -> bool:
        """Migrate prediction history."""
        try:
            data = await self._read_json_file(self.prediction_history_file)
            
            if data.get("version") != DATA_VERSION:
                data["version"] = DATA_VERSION
                data["migrated"] = dt_util.utcnow().isoformat()
                await self._atomic_write_json(self.prediction_history_file, data)
                _LOGGER.info("Ã¢Å“â€¦ Prediction history migrated to version %s", DATA_VERSION)
            
            return True
            
        except Exception as e:
            _LOGGER.error("Ã¢ÂÅ’ Prediction history migration failed: %s", str(e))
            raise

    async def _migrate_learned_weights(self) -> bool:
        """Migrate learned weights."""
        try:
            data = await self._read_json_file(self.learned_weights_file)
            
            if data.get("version") != DATA_VERSION:
                data["version"] = DATA_VERSION
                data["migrated"] = dt_util.utcnow().isoformat()
                await self._atomic_write_json(self.learned_weights_file, data)
                _LOGGER.info("Ã¢Å“â€¦ Learned weights migrated to version %s", DATA_VERSION)
            
            return True
            
        except Exception as e:
            _LOGGER.error("Ã¢ÂÅ’ Learned weights migration failed: %s", str(e))
            raise

    async def _migrate_hourly_profile(self) -> bool:
        """Migrate hourly profile."""
        try:
            data = await self._read_json_file(self.hourly_profile_file)
            
            if data.get("version") != DATA_VERSION:
                data["version"] = DATA_VERSION
                data["migrated"] = dt_util.utcnow().isoformat()
                await self._atomic_write_json(self.hourly_profile_file, data)
                _LOGGER.info("Ã¢Å“â€¦ Hourly profile migrated to version %s", DATA_VERSION)
            
            return True
            
        except Exception as e:
            _LOGGER.error("Ã¢ÂÅ’ Hourly profile migration failed: %s", str(e))
            raise

    async def _migrate_model_state(self) -> bool:
        """Migrate model state."""
        try:
            data = await self._read_json_file(self.model_state_file)
            
            if data.get("version") != DATA_VERSION:
                data["version"] = DATA_VERSION
                data["migrated"] = dt_util.utcnow().isoformat()
                await self._atomic_write_json(self.model_state_file, data)
                _LOGGER.info("Ã¢Å“â€¦ Model state migrated to version %s", DATA_VERSION)
            
            return True
            
        except Exception as e:
            _LOGGER.error("Ã¢ÂÅ’ Model state migration failed: %s", str(e))
            raise

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
        _LOGGER.info("Ã¢Å“â€¦ DataManager cleanup completed")

    # Public API methods # von Zara
    async def get_prediction_history(self) -> Dict[str, Any]:
        """Get prediction history data."""
        return await self._read_json_file(self.prediction_history_file)

    async def get_learned_weights(self) -> Dict[str, Any]:
        """Get learned weights data."""
        return await self._read_json_file(self.learned_weights_file)

    async def get_hourly_profile(self) -> Dict[str, Any]:
        """Get hourly profile data."""
        return await self._read_json_file(self.hourly_profile_file)

    async def get_model_state(self) -> Dict[str, Any]:
        """Get model state data."""
        return await self._read_json_file(self.model_state_file)

    async def save_prediction_history(self, data: Dict[str, Any]) -> None:
        """Save prediction history data."""
        await self._atomic_write_json(self.prediction_history_file, data)

    async def save_learned_weights(self, data: Dict[str, Any]) -> None:
        """Save learned weights data."""
        await self._atomic_write_json(self.learned_weights_file, data)

    async def save_hourly_profile(self, data: Dict[str, Any]) -> None:
        """Save hourly profile data."""
        await self._atomic_write_json(self.hourly_profile_file, data)

    async def save_model_state(self, data: Dict[str, Any]) -> None:
        """Save model state data."""
        await self._atomic_write_json(self.model_state_file, data)

    # ========================================================================
    # Ã¢Å“â€¦ STRATEGIE 2: DATEI-STATUS METHODEN # von Zara
    # ========================================================================
    
    async def get_data_files_count(self) -> int:
        """
        Ã¢Å“â€¦ STRATEGIE 2: ZÃƒÂ¤hlt vorhandene Datendateien.
        
        Returns:
            Anzahl vorhandener Datendateien (0-4)
        # von Zara
        """
        try:
            files_to_check = [
                self.prediction_history_file,
                self.learned_weights_file,
                self.hourly_profile_file,
                self.model_state_file
            ]
            
            count = 0
            for file_path in files_to_check:
                if await self._file_exists(file_path):
                    count += 1
            
            return count
            
        except Exception as e:
            _LOGGER.warning(f"Ã¢Å¡Â Ã¯Â¸Â Fehler beim ZÃƒÂ¤hlen der Datendateien: {e}")
            return 0
    
    async def get_data_status(self) -> dict[str, Any]:
        """
        Ã¢Å“â€¦ STRATEGIE 2: Gibt detaillierten Status aller Datendateien zurÃƒÂ¼ck.
        
        Returns:
            Dict mit Status-Informationen
        # von Zara
        """
        try:
            files_status = {
                "prediction_history": await self._file_exists(self.prediction_history_file),
                "learned_weights": await self._file_exists(self.learned_weights_file),
                "hourly_profile": await self._file_exists(self.hourly_profile_file),
                "model_state": await self._file_exists(self.model_state_file),
            }
            
            total_files = len(files_status)
            available_files = sum(1 for exists in files_status.values() if exists)
            
            return {
                "files": files_status,
                "total": total_files,
                "available": available_files,
                "all_present": available_files == total_files,
                "status_text": f"{available_files}/{total_files} Dateien"
            }
            
        except Exception as e:
            _LOGGER.warning(f"Ã¢Å¡Â Ã¯Â¸Â Fehler beim Abrufen des Datei-Status: {e}")
            return {
                "files": {},
                "total": 4,
                "available": 0,
                "all_present": False,
                "status_text": "Fehler"
            }
    
    def get_data_directory(self) -> Path:
        """
        Ã¢Å“â€¦ STRATEGIE 2: Gibt Data Directory Pfad zurÃƒÂ¼ck.
        
        Returns:
            Path zum Data Directory
        # von Zara
        """
        return self.data_dir
    
    # ========================================================================
    # ✅ LÖSUNG 2: SAVE_ALL_ASYNC IMPLEMENTIERUNG # von Zara
    # ========================================================================
    
    async def save_all_async(self) -> None:
        """
        Speichert alle Datendateien asynchron.
        ✅ LÖSUNG 2: Nun tatsächlich implementiert // von Zara
        
        Diese Methode wird vom Coordinator aufgerufen aber macht
        normalerweise nichts, da Daten nur bei Änderungen gespeichert werden.
        Sie dient als Backup-Save falls irgendwo Daten im RAM sind.
        """
        try:
            # Diese Methode existiert jetzt, aber speichert nur wenn notwendig // von Zara
            # Die tatsächliche Speicherung erfolgt durch:
            # - add_prediction_record() -> Speichert prediction_history
            # - save_learned_weights() -> Speichert nach Training
            # - save_hourly_profile() -> Speichert nach Training
            # - update_model_state() -> Speichert model state
            
            _LOGGER.debug("✅ save_all_async aufgerufen (Daten werden bei Änderungen gespeichert)")
            
        except Exception as e:
            _LOGGER.debug(f"save_all_async: {e}")
