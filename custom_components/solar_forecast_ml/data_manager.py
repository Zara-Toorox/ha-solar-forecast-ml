"""
Data Manager für die Solar Forecast ML Integration.
✓ ERWEITERT mit fehlenden ML-Methods und Backup-System
• UPDATE PATCH: update_today_predictions_actual() für tagesbasierte actual_value Updates - von Zara

Copyright (C) 2025 Zara-Toorox

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
from .helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class DataManager:
    """
    Data Manager für Solar Forecast ML.
    ✓ ERWEITERT mit ML-Methods und Backup-System
    """

    def __init__(self, hass: HomeAssistant, entry_id: str, data_dir: Path, error_handler=None):
        """
        Initialize the Data Manager.
        
        Args:
            hass: Home Assistant instance
            entry_id: Config Entry ID für Multi-Instance-Support
            data_dir: Daten-Verzeichnis für JSON-Dateien
        
        # Geänderter Abschnitt von Zara - entry_id Parameter hinzugefügt
        """
        self.hass = hass
        self.entry_id = entry_id
        self.data_dir = Path(data_dir)
        self.error_handler = error_handler
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="DataManager")
        
        # File paths
        self.prediction_history_file = self.data_dir / "prediction_history.json"
        self.learned_weights_file = self.data_dir / "learned_weights.json"
        self.hourly_profile_file = self.data_dir / "hourly_profile.json"
        self.model_state_file = self.data_dir / "model_state.json"
        self.hourly_samples_file = self.data_dir / "hourly_samples.json"
        
        # Thread-safe lock für kritische Operations
        self._file_lock = asyncio.Lock()
        
        _LOGGER.info("DataManager initialisiert mit async I/O support (Entry: %s)", entry_id)

    async def initialize(self) -> bool:
        """Initialize data manager and create directory structure."""
        try:
            # Create directory structure
            await self._ensure_directory_exists(self.data_dir)
            await self._ensure_directory_exists(self.data_dir / "backups")
            
            # Initialize missing files
            await self._initialize_missing_files()
            
            # Data migration
            migration_success = await self.migrate_data()
            
            if migration_success:
                _LOGGER.info("DataManager erfolgreich initialisiert")
                return True
            else:
                _LOGGER.warning("DataManager initialisiert, aber Migration hatte Probleme")
                return True
                
        except Exception as e:
            _LOGGER.error(f"DataManager Initialisierung fehlgeschlagen: {e}")
            return False

    async def _ensure_directory_exists(self, directory: Path) -> None:
        """Ensure directory exists, create if not."""
        try:
            await self.hass.async_add_executor_job(lambda: directory.mkdir(parents=True, exist_ok=True))
        except Exception as e:
            raise DataIntegrityException(
                f"Failed to create directory {directory}: {str(e)}",
                context=create_context(directory=str(directory), error=str(e))
            )


    async def _get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes."""
        try:
            if await self._file_exists(file_path):
                return file_path.stat().st_size
            return 0
        except Exception:
            return 0

    async def _file_exists(self, file_path: Path) -> bool:
        """Check if file exists."""
        try:
            return await self.hass.async_add_executor_job(file_path.exists)
        except Exception:
            return False

    async def _initialize_missing_files(self) -> None:
        """Initialize missing data files with defaults."""
        # Prediction history
        if not await self._file_exists(self.prediction_history_file):
            await self._create_default_prediction_history()
        
        # Learned weights
        if not await self._file_exists(self.learned_weights_file):
            await self._create_default_learned_weights()
        
        # Hourly profile
        if not await self._file_exists(self.hourly_profile_file):
            await self._create_default_hourly_profile()
        
        # Model state
        if not await self._file_exists(self.model_state_file):
            await self._create_default_model_state()
        
        if not await self._file_exists(self.hourly_samples_file):
            await self._create_default_hourly_samples()

    async def _create_default_prediction_history(self) -> None:
        """Create default prediction history file."""
        default_history = {
            "version": DATA_VERSION,
            "predictions": [],
            "last_updated": dt_util.utcnow().isoformat()
        }
        await self._atomic_write_json(self.prediction_history_file, default_history)

    async def _create_default_learned_weights(self) -> None:
        """Create default learned weights file."""
        from dataclasses import asdict
        default_weights = asdict(create_default_learned_weights())
        default_weights["version"] = DATA_VERSION
        default_weights["last_updated"] = dt_util.utcnow().isoformat()
        await self._atomic_write_json(self.learned_weights_file, default_weights)

    async def _create_default_hourly_profile(self) -> None:
        """Create default hourly profile file."""
        from dataclasses import asdict
        default_profile = asdict(create_default_hourly_profile())
        default_profile["version"] = DATA_VERSION
        default_profile["last_updated"] = dt_util.utcnow().isoformat()
        await self._atomic_write_json(self.hourly_profile_file, default_profile)

    async def _create_default_model_state(self) -> None:
        """Create default model state file."""
        default_state = {
            "version": DATA_VERSION,
            "model_loaded": False,
            "last_training": None,
            "training_samples": 0,
            "current_accuracy": 0.0,
            "model_info": {
                "version": "1.0",
                "type": "ml_predictor"
            },
            "status": "initialized"
        }
        await self._atomic_write_json(self.model_state_file, default_state)

    async def _create_default_hourly_samples(self) -> None:
        default_samples = {
            "version": DATA_VERSION,
            "samples": [],
            "count": 0,
            "last_updated": dt_util.utcnow().isoformat()
        }
        await self._atomic_write_json(self.hourly_samples_file, default_samples)


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
                
                _LOGGER.debug("Atomic write successful: %s", file_path.name)
                
            except Exception as e:
                # Clean up temp file if it exists
                if await self._file_exists(temp_file):
                    try:
                        await self.hass.async_add_executor_job(temp_file.unlink)
                    except Exception:
                        pass
                raise DataIntegrityException(
                    f"Failed to write {file_path.name}: {str(e)}",
                    context=create_context(file=str(file_path), error=str(e))
                )

    async def _read_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Read JSON file with error handling."""
        try:
            if not await self._file_exists(file_path):
                raise DataIntegrityException(
                    f"File not found: {file_path.name}",
                    context=create_context(file=str(file_path))
                )
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
                
        except json.JSONDecodeError as e:
            _LOGGER.error("JSON decode error in %s: %s", file_path.name, str(e))
            raise DataIntegrityException(
                f"Invalid JSON in {file_path.name}: {str(e)}",
                context=create_context(file=str(file_path), error=str(e))
            )
        except Exception as e:
            raise DataIntegrityException(
                f"Failed to read {file_path.name}: {str(e)}",
                context=create_context(file=str(file_path), error=str(e))
            )

    # ✓ ERWEITERT: ML-spezifische Methoden
    async def add_prediction(self, prediction: PredictionRecord) -> None:
        """
        Add a new prediction to history.
        
        Args:
            prediction: PredictionRecord to add
        """
        try:
            # PredictionRecord hat bereits Validierung in __post_init__ - von Zara
            # Konvertiere PredictionRecord zu Dict für JSON - von Zara
            from dataclasses import asdict
            prediction_dict = asdict(prediction)
            
            # Get current history
            history = await self.get_prediction_history()
            predictions = history.get("predictions", [])
            
            # Add new prediction as dict - von Zara
            predictions.append(prediction_dict)
            
            # Limit history size
            if len(predictions) > MAX_PREDICTION_HISTORY:
                predictions = predictions[-MAX_PREDICTION_HISTORY:]
            
            # Update history
            history["predictions"] = predictions
            history["last_updated"] = dt_util.utcnow().isoformat()
            
            # Save
            await self.save_prediction_history(history)
            
            _LOGGER.debug("Prediction added to history")
            
        except Exception as e:
            _LOGGER.error(f"Failed to add prediction: {e}")
            raise DataIntegrityException(
                f"Failed to add prediction: {str(e)}",
                context=create_context(error=str(e))
            )

    async def add_prediction_record(self, record: Dict[str, Any]) -> None:
        """
        Fügt einen neuen Prediction Record zur History hinzu.
        
        Args:
            record: Dictionary mit Prediction-Daten
        """
        try:
            # Validiere Record
            validate_prediction_record(record)
            
            # Get current history
            history = await self.get_prediction_history()
            predictions = history.get("predictions", [])
            
            # Add new record
            predictions.append(record)
            
            # Limit history size
            if len(predictions) > MAX_PREDICTION_HISTORY:
                predictions = predictions[-MAX_PREDICTION_HISTORY:]
            
            # Update history
            history["predictions"] = predictions
            history["last_updated"] = dt_util.utcnow().isoformat()
            
            # Save
            await self.save_prediction_history(history)
            
            _LOGGER.debug("Prediction record added to history")
            
        except Exception as e:
            _LOGGER.error(f"Failed to add prediction record: {e}")
            raise DataIntegrityException(
                f"Failed to add prediction record: {str(e)}",
                context=create_context(error=str(e))
            )

    async def update_latest_prediction_actual(self, actual_value: float, accuracy: float = None) -> None:
        """
        Aktualisiert den letzten Prediction Record mit actual_value.
        DEPRECATED: Verwende update_today_predictions_actual() stattdessen - von Zara
        
        Args:
            actual_value: Tatsächlicher Tagesertrag in kWh
            accuracy: Optional berechnete Genauigkeit (0-1)
        """
        try:
            # Lade aktuelle History - von Zara
            history_data = await self.get_prediction_history()
            predictions = history_data.get('predictions', [])
            
            if not predictions:
                _LOGGER.warning("Keine Predictions vorhanden - Update übersprungen")
                return
            
            # Hole letzten Record (neuester zuerst) - von Zara
            latest_record = predictions[-1]
            
            # Update actual_value - von Zara
            latest_record['actual_value'] = actual_value
            
            # Update accuracy wenn gegeben - von Zara
            if accuracy is not None:
                latest_record['accuracy'] = accuracy
            
            # Speichere aktualisierte History - von Zara
            await self.save_prediction_history(history_data)
            
            if self.error_handler:
                file_size = await self._get_file_size(self.prediction_history_file)
                records_count = len(history_data.get("predictions", []))
                self.error_handler.log_json_operation(
                    file_name="prediction_history.json",
                    operation="add_record",
                    success=True,
                    file_size_bytes=file_size,
                    records_count=records_count
                )
            
            _LOGGER.info(
                f"✓ Latest prediction updated: "
                f"predicted={latest_record.get('predicted_value', 0):.2f}kWh, "
                f"actual={actual_value:.2f}kWh, "
                f"accuracy={accuracy*100:.1f}%" if accuracy else "no accuracy"
            )
            
        except Exception as e:
            _LOGGER.error(f"Failed to update latest prediction actual: {e}")
            raise DataIntegrityException(
                f"Failed to update prediction actual value: {str(e)}",
                context=create_context(error=str(e))
            )

    async def update_today_predictions_actual(self, actual_value: float, accuracy: float = None) -> None:
        """
        Aktualisiert ALLE Prediction Records vom heutigen Tag mit actual_value.
        Neue Methode für tagesbasierte Aktualisierung - von Zara
        
        Args:
            actual_value: Tagesertrag in kWh
            accuracy: Optional berechnete Genauigkeit (0-1)
        """
        try:
            # Lade aktuelle History - von Zara
            history_data = await self.get_prediction_history()
            predictions = history_data.get('predictions', [])
            
            if not predictions:
                _LOGGER.warning("Keine Predictions vorhanden - Update übersprungen")
                return
            
            today = dt_util.as_local(dt_util.utcnow()).date()
            
            updated_count = 0
            
            _LOGGER.debug(f"Suche Predictions lokales Datum: {today}")
            
            # Durchlaufe alle Records und update die vom heutigen Tag - von Zara
            for record in predictions:
                try:
                    # Parse timestamp - von Zara
                    record_timestamp = dt_util.parse_datetime(record.get('timestamp', ''))
                    record_date = dt_util.as_local(record_timestamp).date()
                    
                    # Prüfe ob Record von heute ist - von Zara
                    if record_date == today:
                        # Update actual_value - von Zara
                        record['actual_value'] = actual_value
                        
                        # Update accuracy wenn gegeben - von Zara
                        if accuracy is not None:
                            record['accuracy'] = accuracy
                        
                        updated_count += 1
                        
                except (ValueError, KeyError) as e:
                    _LOGGER.debug(f"Konnte Record nicht parsen, überspringe: {e}")
                    continue
            
            # Speichere aktualisierte History - von Zara
            if updated_count > 0:
                await self.save_prediction_history(history_data)
                
                _LOGGER.info(
                    f"✓ {updated_count} Predictions vom heutigen Tag aktualisiert: "
                    f"actual={actual_value:.2f}kWh"
                    + (f", accuracy={accuracy*100:.1f}%" if accuracy is not None else "")
                )
            else:
                _LOGGER.warning(f"⚠ Keine Predictions vom heutigen Tag gefunden (Suchte: {today})")
            
        except Exception as e:
            _LOGGER.error(f"❌ Failed to update today predictions actual: {e}")
            raise DataIntegrityException(
                f"Failed to update today predictions actual value: {str(e)}",
                context=create_context(error=str(e))
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
            
            # Calculate cutoff time
            cutoff = dt_util.utcnow() - timedelta(hours=hours)
            
            # Filter predictions by timestamp
            recent = []
            for pred in predictions:
                try:
                    pred_time = dt_util.parse_datetime(pred["timestamp"])
                    if pred_time >= cutoff:
                        # Convert dict to PredictionRecord
                        from .typed_data_adapter import TypedDataAdapter
                        adapter = TypedDataAdapter()
                        record = adapter.dict_to_prediction_record(pred)
                        recent.append(record)
                except (ValueError, KeyError) as e:
                    _LOGGER.debug(f"Skipping invalid prediction record: {e}")
                    continue
            
            return recent
            
        except Exception as e:
            _LOGGER.error(f"Failed to get recent predictions: {e}")
            return []

    def get_average_monthly_yield(self) -> float:
        try:
            history_file = self.prediction_history_file
            if not history_file.exists():
                return 0.0
            
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            predictions = data.get('predictions', [])
            if not predictions:
                return 0.0
            
            cutoff = dt_util.utcnow() - timedelta(days=30)
            
            daily_totals = {}
            for pred in predictions:
                try:
                    pred_time = dt_util.parse_datetime(pred['timestamp'])
                    if pred_time < cutoff:
                        continue
                    
                    actual_value = pred.get('actual_value')
                    if actual_value is None or actual_value <= 0:
                        continue
                    
                    date_key = pred_time.date().isoformat()
                    if date_key not in daily_totals:
                        daily_totals[date_key] = 0.0
                    daily_totals[date_key] += actual_value
                    
                except (ValueError, KeyError):
                    continue
            
            if not daily_totals:
                return 0.0
            
            total_yield = sum(daily_totals.values())
            days_count = len(daily_totals)
            
            if days_count == 0:
                return 0.0
            
            avg_daily_yield = total_yield / days_count
            monthly_yield = avg_daily_yield * 30
            
            return round(monthly_yield, 2)
            
        except Exception as e:
            _LOGGER.error(f"Failed to calculate average monthly yield: {e}")
            return 0.0

    async def get_prediction_history(self) -> Dict[str, Any]:
        """
        Get prediction history.
        
        Returns:
            Dictionary with prediction history
        """
        try:
            if not await self._file_exists(self.prediction_history_file):
                await self._create_default_prediction_history()
            
            return await self._read_json_file(self.prediction_history_file)
            
        except Exception as e:
            _LOGGER.error(f"Failed to get prediction history: {e}")
            return {
                "version": DATA_VERSION,
                "predictions": [],
                "last_updated": dt_util.utcnow().isoformat()
            }

    async def get_all_training_records(self, days: int = 60) -> List[Dict[str, Any]]:
        try:
            all_records = []
            
            history_data = await self.get_prediction_history()
            predictions = history_data.get('predictions', [])
            
            for pred in predictions:
                if pred.get('actual_value') is not None:
                    all_records.append({
                        'timestamp': pred['timestamp'],
                        'predicted_value': pred.get('predicted_value', 0.0),
                        'actual_value': pred['actual_value'],
                        'weather_data': pred.get('weather_data', {}),
                        'sensor_data': pred.get('sensor_data', {}),
                        'accuracy': pred.get('accuracy', 0.75),
                        'model_version': pred.get('model_version', DATA_VERSION),
                        'source': 'prediction_history'
                    })
            
            samples_data = await self.get_hourly_samples(days=days)
            samples = samples_data.get('samples', [])
            
            daily_aggregated = {}
            for sample in samples:
                try:
                    timestamp = dt_util.parse_datetime(sample['timestamp'])
                    date_key = timestamp.date().isoformat()
                    
                    if date_key not in daily_aggregated:
                        daily_aggregated[date_key] = {
                            'timestamp': timestamp.replace(hour=12, minute=0, second=0, microsecond=0).isoformat(),
                            'actual_value': 0.0,
                            'weather_data': sample.get('weather_data', {}),
                            'sensor_data': sample.get('sensor_data', {}),
                            'hourly_count': 0
                        }
                    
                    daily_aggregated[date_key]['actual_value'] += sample.get('actual_kwh', 0.0)
                    daily_aggregated[date_key]['hourly_count'] += 1
                    
                except (ValueError, KeyError) as e:
                    _LOGGER.debug(f"Skipping invalid sample: {e}")
                    continue
            
            for date_key, agg_data in daily_aggregated.items():
                all_records.append({
                    'timestamp': agg_data['timestamp'],
                    'predicted_value': agg_data['actual_value'],
                    'actual_value': agg_data['actual_value'],
                    'weather_data': agg_data['weather_data'],
                    'sensor_data': agg_data['sensor_data'],
                    'accuracy': 1.0,
                    'model_version': DATA_VERSION,
                    'source': 'hourly_samples',
                    'hourly_count': agg_data['hourly_count']
                })
            
            all_records.sort(key=lambda x: x['timestamp'])
            
            _LOGGER.info(
                f"Training records loaded: {len(predictions)} from predictions, "
                f"{len(daily_aggregated)} from hourly samples, total: {len(all_records)}"
            )
            
            return all_records
            
        except Exception as e:
            _LOGGER.error(f"Failed to get all training records: {e}")
            return []

    async def save_prediction_history(self, data: Dict[str, Any]) -> None:
        """
        Save prediction history.
        
        Args:
            data: Prediction history data to save
        """
        try:
            # Ensure version is set
            if "version" not in data:
                data["version"] = DATA_VERSION
            
            # Update timestamp
            data["last_updated"] = dt_util.utcnow().isoformat()
            
            # Atomic write
            await self._atomic_write_json(self.prediction_history_file, data)
            
            if self.error_handler:
                file_size = await self._get_file_size(self.prediction_history_file)
                records_count = len(data.get("predictions", []))
                self.error_handler.log_json_operation(
                    file_name="prediction_history.json",
                    operation="write",
                    success=True,
                    file_size_bytes=file_size,
                    records_count=records_count
                )
            
        except Exception as e:
            _LOGGER.error(f"Failed to save prediction history: {e}")
            if self.error_handler:
                self.error_handler.log_json_operation(
                    file_name="prediction_history.json",
                    operation="write",
                    success=False,
                    error_message=str(e)
                )
            raise DataIntegrityException(
                f"Failed to save prediction history: {str(e)}",
                context=create_context(error=str(e))
            )

    async def migrate_data(self) -> bool:
        """
        Migrate data to current version.
        
        Returns:
            True if migration successful
        """
        try:
            # Check each file and migrate if needed
            await self._migrate_prediction_history()
            await self._migrate_learned_weights()
            await self._migrate_hourly_profile()
            await self._migrate_model_state()
            
            _LOGGER.info("Data migration completed successfully")
            return True
            
        except Exception as e:
            _LOGGER.error(f"❌ Data migration failed: {e}")
            return False

    async def _migrate_prediction_history(self) -> None:
        """Migrate prediction history to current version."""
        try:
            if not await self._file_exists(self.prediction_history_file):
                return
            
            data = await self._read_json_file(self.prediction_history_file)
            current_version = data.get("version", "1.0")
            
            if current_version != DATA_VERSION:
                _LOGGER.info(f"Migrating prediction history from {current_version} to {DATA_VERSION}")
                data["version"] = DATA_VERSION
                await self._atomic_write_json(self.prediction_history_file, data)
                
        except Exception as e:
            _LOGGER.error(f"Failed to migrate prediction history: {e}")

    async def _migrate_learned_weights(self) -> None:
        """Migrate learned weights to current version."""
        try:
            if not await self._file_exists(self.learned_weights_file):
                return
            
            data = await self._read_json_file(self.learned_weights_file)
            current_version = data.get("version", "1.0")
            
            if current_version != DATA_VERSION:
                _LOGGER.info(f"Migrating learned weights from {current_version} to {DATA_VERSION}")
                data["version"] = DATA_VERSION
                await self._atomic_write_json(self.learned_weights_file, data)
                
        except Exception as e:
            _LOGGER.error(f"Failed to migrate learned weights: {e}")

    async def _migrate_hourly_profile(self) -> None:
        """Migrate hourly profile to current version."""
        try:
            if not await self._file_exists(self.hourly_profile_file):
                return
            
            data = await self._read_json_file(self.hourly_profile_file)
            current_version = data.get("version", "1.0")
            
            if current_version != DATA_VERSION:
                _LOGGER.info(f"Migrating hourly profile from {current_version} to {DATA_VERSION}")
                data["version"] = DATA_VERSION
                await self._atomic_write_json(self.hourly_profile_file, data)
                
        except Exception as e:
            _LOGGER.error(f"Failed to migrate hourly profile: {e}")

    async def _migrate_model_state(self) -> None:
        """Migrate model state to current version."""
        try:
            if not await self._file_exists(self.model_state_file):
                return
            
            data = await self._read_json_file(self.model_state_file)
            current_version = data.get("version", "1.0")
            
            if current_version != DATA_VERSION:
                _LOGGER.info(f"Migrating model state from {current_version} to {DATA_VERSION}")
                data["version"] = DATA_VERSION
                await self._atomic_write_json(self.model_state_file, data)
                
        except Exception as e:
            _LOGGER.error(f"Failed to migrate model state: {e}")

    async def cleanup_old_data(self, days: int = 90) -> None:
        """
        Clean up old prediction data.
        
        Args:
            days: Keep data from this many days
        """
        try:
            history = await self.get_prediction_history()
            predictions = history.get("predictions", [])
            
            if not predictions:
                return
            
            # Calculate cutoff
            cutoff = dt_util.utcnow() - timedelta(days=days)
            
            # Filter predictions
            filtered = [
                p for p in predictions
                if dt_util.parse_datetime(p["timestamp"]) >= cutoff
            ]
            
            # Update if changed
            if len(filtered) != len(predictions):
                history["predictions"] = filtered
                await self.save_prediction_history(history)
                _LOGGER.info(f"Cleaned up {len(predictions) - len(filtered)} old predictions")
                
        except Exception as e:
            _LOGGER.error(f"Failed to cleanup old data: {e}")

    async def get_learned_weights(self) -> Optional[LearnedWeights]:
        """Get learned weights."""
        try:
            if not await self._file_exists(self.learned_weights_file):
                return None
            
            data = await self._read_json_file(self.learned_weights_file)
            
            # Convert dict to LearnedWeights
            from .typed_data_adapter import TypedDataAdapter
            adapter = TypedDataAdapter()
            return adapter.dict_to_learned_weights(data)
            
        except Exception as e:
            _LOGGER.error(f"Failed to get learned weights: {e}")
            return None

    async def save_learned_weights(self, weights: LearnedWeights) -> None:
        """Save learned weights."""
        try:
            from dataclasses import asdict
            data = asdict(weights)
            data["version"] = DATA_VERSION
            data["last_updated"] = dt_util.utcnow().isoformat()
            
            await self._atomic_write_json(self.learned_weights_file, data)
            
            if self.error_handler:
                file_size = await self._get_file_size(self.learned_weights_file)
                self.error_handler.log_json_operation(
                    file_name="learned_weights.json",
                    operation="write",
                    success=True,
                    file_size_bytes=file_size
                )
            
            _LOGGER.debug("Learned weights saved")
            
        except Exception as e:
            _LOGGER.error(f"Failed to save learned weights: {e}")
            if self.error_handler:
                self.error_handler.log_json_operation(
                    file_name="learned_weights.json",
                    operation="write",
                    success=False,
                    error_message=str(e)
                )
            raise DataIntegrityException(
                f"Failed to save learned weights: {str(e)}",
                context=create_context(error=str(e))
            )

    async def get_hourly_profile(self) -> Optional[HourlyProfile]:
        """Get hourly profile."""
        try:
            if not await self._file_exists(self.hourly_profile_file):
                return None
            
            data = await self._read_json_file(self.hourly_profile_file)
            
            # Convert dict to HourlyProfile
            from .typed_data_adapter import TypedDataAdapter
            adapter = TypedDataAdapter()
            return adapter.dict_to_hourly_profile(data)
            
        except Exception as e:
            _LOGGER.error(f"Failed to get hourly profile: {e}")
            return None

    async def save_hourly_profile(self, profile: HourlyProfile) -> None:
        """Save hourly profile."""
        try:
            from dataclasses import asdict
            data = asdict(profile)
            data["version"] = DATA_VERSION
            data["last_updated"] = dt_util.utcnow().isoformat()
            
            await self._atomic_write_json(self.hourly_profile_file, data)
            
            if self.error_handler:
                file_size = await self._get_file_size(self.hourly_profile_file)
                self.error_handler.log_json_operation(
                    file_name="hourly_profile.json",
                    operation="write",
                    success=True,
                    file_size_bytes=file_size
                )
            
            _LOGGER.debug("Hourly profile saved")
            
        except Exception as e:
            _LOGGER.error(f"Failed to save hourly profile: {e}")
            if self.error_handler:
                self.error_handler.log_json_operation(
                    file_name="hourly_profile.json",
                    operation="write",
                    success=False,
                    error_message=str(e)
                )
            raise DataIntegrityException(
                f"Failed to save hourly profile: {str(e)}",
                context=create_context(error=str(e))
            )

    async def get_model_state(self) -> Dict[str, Any]:
        """Get model state."""
        try:
            if not await self._file_exists(self.model_state_file):
                await self._create_default_model_state()
            
            return await self._read_json_file(self.model_state_file)
            
        except Exception as e:
            _LOGGER.error(f"Failed to get model state: {e}")
            return {
                "version": DATA_VERSION,
                "model_loaded": False,
                "last_training": None,
                "training_samples": 0,
                "current_accuracy": 0.0,
                "status": "error"
            }

    async def save_model_state(self, state: Dict[str, Any]) -> None:
        """Save model state."""
        try:
            if "version" not in state:
                state["version"] = DATA_VERSION
            
            state["last_updated"] = dt_util.utcnow().isoformat()
            
            await self._atomic_write_json(self.model_state_file, state)
            
            if self.error_handler:
                file_size = await self._get_file_size(self.model_state_file)
                self.error_handler.log_json_operation(
                    file_name="model_state.json",
                    operation="write",
                    success=True,
                    file_size_bytes=file_size
                )
            
            _LOGGER.debug("Model state saved")
            
        except Exception as e:
            _LOGGER.error(f"Failed to save model state: {e}")
            if self.error_handler:
                self.error_handler.log_json_operation(
                    file_name="model_state.json",
                    operation="write",
                    success=False,
                    error_message=str(e)
                )
            raise DataIntegrityException(
                f"Failed to save model state: {str(e)}",
                context=create_context(error=str(e))
            )

    async def add_hourly_sample(self, sample: Dict[str, Any]) -> None:
        try:
            samples_data = await self.get_hourly_samples()
            samples = samples_data.get("samples", [])
            
            samples.append(sample)
            
            if len(samples) > 24 * 60:
                samples = samples[-24 * 60:]
            
            samples_data["samples"] = samples
            samples_data["last_updated"] = dt_util.utcnow().isoformat()
            samples_data["count"] = len(samples)
            
            await self._atomic_write_json(self.hourly_samples_file, samples_data)
            
            _LOGGER.debug("Hourly sample added")
            
        except Exception as e:
            _LOGGER.error(f"Failed to add hourly sample: {e}")
            raise DataIntegrityException(
                f"Failed to add hourly sample: {str(e)}",
                context=create_context(error=str(e))
            )

    async def get_hourly_samples(self, days: int = 60) -> Dict[str, Any]:
        try:
            if not await self._file_exists(self.hourly_samples_file):
                await self._create_default_hourly_samples()
            
            data = await self._read_json_file(self.hourly_samples_file)
            
            if days and days > 0:
                cutoff = dt_util.utcnow() - timedelta(days=days)
                samples = data.get("samples", [])
                filtered = []
                
                for sample in samples:
                    try:
                        timestamp = dt_util.parse_datetime(sample["timestamp"])
                        if timestamp >= cutoff:
                            filtered.append(sample)
                    except (ValueError, KeyError):
                        continue
                
                data["samples"] = filtered
                data["count"] = len(filtered)
            
            return data
            
        except Exception as e:
            _LOGGER.error(f"Failed to get hourly samples: {e}")
            return {
                "version": DATA_VERSION,
                "samples": [],
                "count": 0,
                "last_updated": None
            }


    async def save_all_async(self) -> None:
        """
        Zentrale Save-Methode Auto-Save System.
        Verifiziert Datenpersistenz und erstellt fehlende Dateien. // von Zara
        """
        try:
            # Prüfe kritische Dateien auf Existenz // von Zara
            critical_files = [
                (self.prediction_history_file, "prediction_history"),
                (self.learned_weights_file, "learned_weights"),
                (self.hourly_profile_file, "hourly_profile"),
                (self.model_state_file, "model_state"),
                (self.hourly_samples_file, "hourly_samples")
            ]
            
            missing_files = []
            for file_path, file_name in critical_files:
                if not await self._file_exists(file_path):
                    missing_files.append(file_name)
            
            if missing_files:
                _LOGGER.warning(f"Auto-Save: Fehlende Dateien erkannt: {missing_files}")
                # Erstelle fehlende Dateien mit Defaults // von Zara
                await self._initialize_missing_files()
                _LOGGER.info("Auto-Save: Fehlende Dateien wiederhergestellt")
            else:
                _LOGGER.debug("✓ Auto-Save: Alle Daten persistent")
            
        except Exception as e:
            _LOGGER.error(f"Auto-Save fehlgeschlagen: {e}")
            # Nicht re-raisen - Auto-Save ist nicht kritisch // von Zara

    async def cleanup(self) -> None:
        try:
            await self.hass.async_add_executor_job(self.shutdown)
            _LOGGER.info("DataManager cleanup complete")
        except Exception as e:
            _LOGGER.error(f"Error during DataManager cleanup: {e}")

    def shutdown(self) -> None:
        """Shutdown data manager and cleanup resources."""
        try:
            self._executor.shutdown(wait=False)
            _LOGGER.info("DataManager shutdown complete")
        except Exception as e:
            _LOGGER.error(f"Error during DataManager shutdown: {e}")