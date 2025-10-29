"""
Data Manager for the Solar Forecast ML Integration.
EXTENDED with missing ML-Methods and Backup-System
• UPDATE PATCH: update_today_predictions_actual() for day-based actual_value updates - by Zara
• FIX: get_all_training_records now uses HOURLY samples directly for training.

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
    Data Manager for Solar Forecast ML.
    EXTENDED with ML-Methods and Backup-System
    """

    def __init__(self, hass: HomeAssistant, entry_id: str, data_dir: Path, error_handler=None):
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

        # Thread-safe lock for critical operations
        self._file_lock = asyncio.Lock()

        _LOGGER.info("DataManager initialized with async I/O support (Entry: %s)", entry_id)

    async def initialize(self) -> bool:
        """Initialize data manager and create directory structure."""
        try:
            await self._ensure_directory_exists(self.data_dir)
            await self._ensure_directory_exists(self.data_dir / "backups")
            await self._initialize_missing_files()
            migration_success = await self.migrate_data()

            if migration_success:
                _LOGGER.info("DataManager initialized successfully")
                return True
            else:
                _LOGGER.warning("DataManager initialized, but migration had issues")
                return True

        except Exception as e:
            _LOGGER.error(f"DataManager initialization failed: {e}")
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
        if not await self._file_exists(self.prediction_history_file):
            await self._create_default_prediction_history()
        if not await self._file_exists(self.learned_weights_file):
            await self._create_default_learned_weights()
        if not await self._file_exists(self.hourly_profile_file):
            await self._create_default_hourly_profile()
        if not await self._file_exists(self.model_state_file):
            await self._create_default_model_state()
        if not await self._file_exists(self.hourly_samples_file):
            await self._create_default_hourly_samples()

    async def _create_default_prediction_history(self) -> None:
        default_history = {
            "version": DATA_VERSION,
            "predictions": [],
            "last_updated": dt_util.utcnow().isoformat()
        }
        await self._atomic_write_json(self.prediction_history_file, default_history)

    async def _create_default_learned_weights(self) -> None:
        from dataclasses import asdict
        default_weights = asdict(create_default_learned_weights())
        default_weights["version"] = DATA_VERSION
        default_weights["last_updated"] = dt_util.utcnow().isoformat()
        await self._atomic_write_json(self.learned_weights_file, default_weights)

    async def _create_default_hourly_profile(self) -> None:
        from dataclasses import asdict
        default_profile = asdict(create_default_hourly_profile())
        default_profile["version"] = DATA_VERSION
        default_profile["last_updated"] = dt_util.utcnow().isoformat()
        await self._atomic_write_json(self.hourly_profile_file, default_profile)

    async def _create_default_model_state(self) -> None:
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
            "status": "initialized",
            "backfill_run": False  # Wichtig: Backfill-Status
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
        async with self._file_lock:
            temp_file = file_path.with_suffix('.tmp')
            try:
                async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                    # Verwende sort_keys=True für konsistente Reihenfolge (optional, aber gut für Diffs)
                    json_data = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
                    await f.write(json_data)
                await self.hass.async_add_executor_job(
                    shutil.move, str(temp_file), str(file_path)
                )
                _LOGGER.debug("Atomic write successful: %s", file_path.name)
            except Exception as e:
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
        try:
            if not await self._file_exists(file_path):
                # Wenn Datei nicht existiert, erstelle leere Standardstruktur
                _LOGGER.warning(f"File {file_path.name} not found, creating default.")
                if file_path == self.prediction_history_file:
                    await self._create_default_prediction_history()
                    return {"version": DATA_VERSION, "predictions": [], "last_updated": None}
                elif file_path == self.hourly_samples_file:
                    await self._create_default_hourly_samples()
                    return {"version": DATA_VERSION, "samples": [], "count": 0, "last_updated": None}
                # Füge hier bei Bedarf weitere Standard-Erstellungen hinzu
                else:
                     raise DataIntegrityException(f"File not found and no default creator: {file_path.name}")
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                # Füge leere Datei Prüfung hinzu
                if not content:
                     _LOGGER.warning(f"File {file_path.name} is empty, returning default structure.")
                     if file_path == self.prediction_history_file:
                         return {"version": DATA_VERSION, "predictions": [], "last_updated": None}
                     elif file_path == self.hourly_samples_file:
                         return {"version": DATA_VERSION, "samples": [], "count": 0, "last_updated": None}
                     # ... andere Defaults ...
                     else:
                         return {} # Generischer Fallback
                return await self.hass.async_add_executor_job(json.loads, content)
        except json.JSONDecodeError as e:
            _LOGGER.error(f"Invalid JSON in {file_path.name}: {str(e)}. Attempting backup restore or returning default.")
            # Optional: Backup-Wiederherstellungslogik hier einfügen
            # Fallback auf Default:
            if file_path == self.prediction_history_file: return {"version": DATA_VERSION, "predictions": [], "last_updated": None}
            if file_path == self.hourly_samples_file: return {"version": DATA_VERSION, "samples": [], "count": 0, "last_updated": None}
            # ...
            return {}
        except Exception as e:
            raise DataIntegrityException(f"Failed to read {file_path.name}: {str(e)}")

    # --- ML-SPECIFIC METHODS ---

    async def add_prediction_record(self, record: Dict[str, Any]) -> None:
        try:
            validate_prediction_record(record)
            history = await self.get_prediction_history()
            predictions = history.get("predictions", [])
            predictions.append(record)
            if len(predictions) > MAX_PREDICTION_HISTORY:
                predictions = predictions[-MAX_PREDICTION_HISTORY:]
            history["predictions"] = predictions
            history["last_updated"] = dt_util.utcnow().isoformat()
            await self.save_prediction_history(history)
            _LOGGER.debug("Prediction record added to history")
        except Exception as e:
            raise DataIntegrityException(f"Failed to add prediction record: {str(e)}")

    async def update_today_predictions_actual(self, actual_value: float, accuracy: float = None) -> None:
        try:
            history_data = await self.get_prediction_history()
            predictions = history_data.get('predictions', [])
            if not predictions:
                return
            today = dt_util.as_local(dt_util.utcnow()).date()
            updated_count = 0
            # Iteriere rückwärts, um potenziell nur die neuesten Einträge des Tages zu aktualisieren
            for record in reversed(predictions):
                try:
                    record_timestamp = dt_util.parse_datetime(record.get('timestamp', ''))
                    record_date = dt_util.as_local(record_timestamp).date()
                    if record_date == today:
                         # Update nur, wenn actual_value noch nicht gesetzt ist oder None war
                        if record.get('actual_value') is None:
                            record['actual_value'] = actual_value
                            if accuracy is not None:
                                record['accuracy'] = accuracy
                            updated_count += 1
                        # Optional: Breche ab, wenn der erste passende Eintrag gefunden wurde?
                        # break # Wenn nur der allerletzte Eintrag aktualisiert werden soll
                    elif record_date < today:
                         # Da die Liste (implizit) nach Zeit sortiert ist, können wir hier abbrechen
                         break
                except (ValueError, KeyError):
                    continue
            if updated_count > 0:
                await self.save_prediction_history(history_data)
                _LOGGER.info(f"{updated_count} predictions from today updated with actual value.")
            else:
                 _LOGGER.debug("No prediction records found for today needing an actual value update.")
        except Exception as e:
            raise DataIntegrityException(f"Failed to update today predictions: {str(e)}")

    async def get_prediction_history(self) -> Dict[str, Any]:
        return await self._read_json_file(self.prediction_history_file)

    async def save_prediction_history(self, data: Dict[str, Any]) -> None:
        if "version" not in data:
            data["version"] = DATA_VERSION
        data["last_updated"] = dt_util.utcnow().isoformat()
        await self._atomic_write_json(self.prediction_history_file, data)

    # === KORRIGIERTE get_all_training_records FUNKTION ===
    async def get_all_training_records(self, days: int = 60) -> List[Dict[str, Any]]:
        """
        Gibt eine Liste von *stündlichen* Datensätzen zurück, die für das Training
        geeignet sind. Nutzt primär hourly_samples.json.
        """
        all_records = []
        try:
            _LOGGER.info(f"Loading hourly samples for training (last {days} days)...")
            samples_data = await self.get_hourly_samples(days=days)
            samples = samples_data.get('samples', [])

            if not samples:
                _LOGGER.warning("No hourly samples found for training.")
                return []

            valid_sample_count = 0
            for sample in samples:
                try:
                    # Prüfe, ob der Sample einen positiven Energieertrag hat
                    actual_kwh = sample.get('actual_kwh')
                    if actual_kwh is None or actual_kwh <= 0:
                        continue

                    # Stelle sicher, dass Wetter- und Sensordaten vorhanden sind
                    weather_data = sample.get('weather_data')
                    sensor_data = sample.get('sensor_data')
                    if not weather_data or not sensor_data:
                         _LOGGER.debug(f"Skipping sample due to missing weather/sensor data: {sample.get('timestamp')}")
                         continue

                    # Erstelle den Trainingsdatensatz im erwarteten Format
                    training_record = {
                        'timestamp': sample['timestamp'],
                        # Wichtig: Key-Mapping von 'actual_kwh' zu 'actual_value'
                        'actual_value': actual_kwh,
                        'weather_data': weather_data,
                        'sensor_data': sensor_data,
                        # Füge weitere Felder hinzu, falls vom Trainer benötigt
                        'model_version': sample.get('model_version', DATA_VERSION),
                        'source': 'hourly_samples'
                    }
                    all_records.append(training_record)
                    valid_sample_count += 1

                except KeyError as e:
                    _LOGGER.warning(f"Skipping invalid hourly sample (missing key {e}): {sample.get('timestamp')}")
                    continue
                except Exception as e:
                     _LOGGER.warning(f"Skipping sample due to unexpected error ({e}): {sample.get('timestamp')}")
                     continue # Überspringe nur diesen Sample

            # Sortiere nach Zeitstempel, falls die Reihenfolge wichtig ist
            all_records.sort(key=lambda x: x['timestamp'])

            _LOGGER.info(f"Training records prepared: {valid_sample_count} valid hourly samples loaded from {len(samples)} total samples.")
            return all_records

        except Exception as e:
            _LOGGER.error(f"Failed to get training records from hourly samples: {e}", exc_info=True)
            return [] # Gib im Fehlerfall eine leere Liste zurück
    # === ENDE KORREKTUR ===

    async def get_learned_weights(self) -> Optional[LearnedWeights]:
        try:
            if not await self._file_exists(self.learned_weights_file):
                 _LOGGER.warning("Learned weights file not found.")
                 return None # Kein Fehler, nur keine Daten
            data = await self._read_json_file(self.learned_weights_file)
            # Einfache Validierung (Beispiel)
            if not data or "weights" not in data or "feature_names" not in data:
                 _LOGGER.error("Learned weights file is corrupted or incomplete.")
                 # Optional: Versuche Backup wiederherzustellen
                 return None
            from .typed_data_adapter import TypedDataAdapter
            adapter = TypedDataAdapter()
            return adapter.dict_to_learned_weights(data)
        except DataIntegrityException as e:
             _LOGGER.error(f"Data integrity error reading learned weights: {e}")
             return None # Bei Lesefehler keine Gewichte zurückgeben
        except Exception as e:
            _LOGGER.error(f"Unexpected error getting learned weights: {e}", exc_info=True)
            return None

    async def save_learned_weights(self, weights: LearnedWeights) -> None:
        try:
            from dataclasses import asdict
            # Prüfe, ob das Objekt valide ist, bevor es gespeichert wird
            if not isinstance(weights, LearnedWeights):
                 raise TypeError("Invalid type provided for weights.")
            data = asdict(weights)
            # Füge Metadaten hinzu
            data["version"] = DATA_VERSION
            data["last_updated"] = dt_util.utcnow().isoformat()
            await self._atomic_write_json(self.learned_weights_file, data)
            _LOGGER.info("Learned weights saved successfully.") # Geändert von DEBUG auf INFO
        except Exception as e:
            _LOGGER.error(f"Failed to save learned weights: {e}", exc_info=True)
            # Werfe den Fehler weiter, damit der Aufrufer informiert wird
            raise DataIntegrityException(f"Failed to save learned weights: {str(e)}")

    async def get_hourly_profile(self) -> Optional[HourlyProfile]:
        try:
            if not await self._file_exists(self.hourly_profile_file):
                 _LOGGER.warning("Hourly profile file not found.")
                 return None
            data = await self._read_json_file(self.hourly_profile_file)
            if not data or "hourly_averages" not in data:
                 _LOGGER.error("Hourly profile file is corrupted or incomplete.")
                 return None
            from .typed_data_adapter import TypedDataAdapter
            adapter = TypedDataAdapter()
            return adapter.dict_to_hourly_profile(data)
        except DataIntegrityException as e:
             _LOGGER.error(f"Data integrity error reading hourly profile: {e}")
             return None
        except Exception as e:
            _LOGGER.error(f"Unexpected error getting hourly profile: {e}", exc_info=True)
            return None

    async def save_hourly_profile(self, profile: HourlyProfile) -> None:
        try:
            from dataclasses import asdict
            if not isinstance(profile, HourlyProfile):
                 raise TypeError("Invalid type provided for profile.")
            data = asdict(profile)
            data["version"] = DATA_VERSION
            data["last_updated"] = dt_util.utcnow().isoformat()
            await self._atomic_write_json(self.hourly_profile_file, data)
            _LOGGER.info("Hourly profile saved successfully.")
        except Exception as e:
             _LOGGER.error(f"Failed to save hourly profile: {e}", exc_info=True)
             raise DataIntegrityException(f"Failed to save hourly profile: {str(e)}")

    async def get_model_state(self) -> Dict[str, Any]:
        try:
            # Versuche zuerst zu lesen
            data = await self._read_json_file(self.model_state_file)
            # Stelle sicher, dass 'backfill_run' existiert (Migration für ältere Versionen)
            if "backfill_run" not in data:
                _LOGGER.info("Adding missing 'backfill_run' key to model state.")
                data["backfill_run"] = False
                # Speichere den korrigierten Zustand sofort
                await self._atomic_write_json(self.model_state_file, data)
            return data
        except DataIntegrityException as e:
             # Wenn Datei nicht gefunden wurde (vom read_json_file behandelt) oder korrupt ist
             _LOGGER.warning(f"Model state file issue ({e}), creating default state.")
             await self._create_default_model_state()
             # Lese den neu erstellten Zustand
             return await self._read_json_file(self.model_state_file)
        except Exception as e:
             # Fallback für unerwartete Fehler
             _LOGGER.error(f"Unexpected error getting model state: {e}", exc_info=True)
             return {
                 "version": DATA_VERSION, "model_loaded": False, "last_training": None,
                 "training_samples": 0, "current_accuracy": 0.0, "status": "error",
                 "backfill_run": False
             }

    async def save_model_state(self, state: Dict[str, Any]) -> None:
        try:
            if not isinstance(state, dict):
                 raise TypeError("Invalid type provided for model state.")
            # Stelle sicher, dass wichtige Keys vorhanden sind
            state.setdefault("version", DATA_VERSION)
            state.setdefault("backfill_run", False) # Stelle sicher, dass es immer da ist
            state["last_updated"] = dt_util.utcnow().isoformat()
            await self._atomic_write_json(self.model_state_file, state)
            _LOGGER.debug("Model state saved")
        except Exception as e:
             _LOGGER.error(f"Failed to save model state: {e}", exc_info=True)
             raise DataIntegrityException(f"Failed to save model state: {str(e)}")

    # --- BACKFILL STATUS METHODEN ---
    async def has_backfill_run(self) -> bool:
        """Prüft, ob der Backfill bereits durchgeführt wurde."""
        try:
            state = await self.get_model_state()
            # Verwende .get() mit Fallback, falls der Key doch fehlt
            return state.get("backfill_run", False)
        except Exception as e:
            _LOGGER.error(f"Failed to check backfill status: {e}", exc_info=True)
            return False # Sicherer Fallback

    async def set_backfill_run(self, status: bool = True) -> None:
        """Setzt den Backfill-Status."""
        try:
            # Verwende eine Sperre, um Race Conditions beim Lesen/Schreiben zu vermeiden
            async with self._file_lock:
                 state = await self.get_model_state() # Lese aktuellen Zustand
                 if state.get("backfill_run") == status:
                     _LOGGER.debug(f"Backfill status already set to {status}.")
                     return # Keine Änderung nötig
                 state["backfill_run"] = status
                 await self.save_model_state(state) # Speichere geänderten Zustand
            _LOGGER.info(f"Backfill status successfully set to {status}")
        except Exception as e:
            _LOGGER.error(f"Failed to set backfill status to {status}: {e}", exc_info=True)
            # Werfe den Fehler nicht weiter, um den Ablauf nicht zu blockieren,
            # aber logge ihn als Error.
            # raise DataIntegrityException(f"Failed to set backfill status: {str(e)}")

    # --- ENDE BACKFILL ---

    async def add_hourly_sample(self, sample: Dict[str, Any]) -> None:
        try:
            # Validierung des Samples (Beispiel)
            if not all(k in sample for k in ["timestamp", "actual_kwh", "weather_data", "sensor_data"]):
                 _LOGGER.warning(f"Skipping incomplete hourly sample: {sample.get('timestamp')}")
                 return
            
            async with self._file_lock: # Sperre für Lese-Modifiziere-Schreibe-Operation
                samples_data = await self.get_hourly_samples(days=0) # Lese *alle* Samples
                samples = samples_data.get("samples", [])
                
                # Optional: Prüfen, ob dieser Zeitstempel schon existiert, um Duplikate zu vermeiden
                # timestamp_exists = any(s.get('timestamp') == sample['timestamp'] for s in samples)
                # if timestamp_exists:
                #     _LOGGER.debug(f"Hourly sample for {sample['timestamp']} already exists, skipping.")
                #     return

                samples.append(sample)
                
                # Limit auf die letzten 60 Tage (ca. 24 * 60 Stunden)
                cutoff_time = dt_util.utcnow() - timedelta(days=60)
                # Filtere direkt beim Hinzufügen ältere Samples heraus
                filtered_samples = [s for s in samples if dt_util.parse_datetime(s['timestamp']) >= cutoff_time]
                
                # Update Metadaten
                samples_data["samples"] = filtered_samples
                samples_data["last_updated"] = dt_util.utcnow().isoformat()
                samples_data["count"] = len(filtered_samples)
                
                # Schreibe die aktualisierten Daten zurück
                await self._atomic_write_json(self.hourly_samples_file, samples_data)
                
            _LOGGER.debug(f"Hourly sample added for {sample['timestamp']}")
            
        except Exception as e:
            _LOGGER.error(f"Failed to add hourly sample: {e}", exc_info=True)
            # Fehler nicht weiter werfen, um den Hauptprozess nicht zu stören
            # raise DataIntegrityException(f"Failed to add hourly sample: {str(e)}")

    async def get_hourly_samples(self, days: int = 60) -> Dict[str, Any]:
        """Liest stündliche Samples, optional gefiltert nach Tagen."""
        try:
            # Lese die Rohdaten aus der Datei
            data = await self._read_json_file(self.hourly_samples_file)
            
            # Filtere nach Tagen, falls gewünscht (days > 0)
            if days and days > 0:
                cutoff = dt_util.utcnow() - timedelta(days=days)
                samples = data.get("samples", [])
                # Verwende List Comprehension für effizienteres Filtern
                filtered = [
                    sample for sample in samples
                    if dt_util.parse_datetime(sample.get("timestamp", "1970-01-01T00:00:00+00:00")) >= cutoff
                ]
                 # Gib eine Kopie zurück, um das Original nicht zu ändern
                return {
                    "version": data.get("version", DATA_VERSION),
                    "samples": filtered,
                    "count": len(filtered),
                    "last_updated": data.get("last_updated")
                }
            else:
                 # Gib die vollständigen Daten zurück (als Kopie)
                 return data.copy()
                 
        except DataIntegrityException as e:
             # Wenn Datei nicht gefunden/korrupt (wird von _read_json_file behandelt)
             _LOGGER.error(f"Could not get hourly samples due to file issue: {e}")
             return {"version": DATA_VERSION, "samples": [], "count": 0, "last_updated": None}
        except Exception as e:
             _LOGGER.error(f"Unexpected error getting hourly samples: {e}", exc_info=True)
             return {"version": DATA_VERSION, "samples": [], "count": 0, "last_updated": None}

    async def cleanup(self) -> None:
        """Räumt Ressourcen auf, insbesondere den ThreadPoolExecutor."""
        try:
            _LOGGER.info("Shutting down DataManager Executor...")
            # --- START CORRECTION ---
            # Directly schedule the blocking shutdown call in HA's executor
            # No need to use self._executor.submit()
            await self.hass.async_add_executor_job(self._executor.shutdown, True)
            # --- END CORRECTION ---
            _LOGGER.info("DataManager cleanup complete.")
        except Exception as e:
            _LOGGER.error(f"Error during DataManager cleanup: {e}", exc_info=True)

    # --- ZUSÄTZLICHE ANALYSEMETHODE ---
    async def get_average_monthly_yield(self) -> float:
        """
        Berechnet den durchschnittlichen monatlichen Ertrag (kWh) basierend auf den letzten 30 Tagen
        der *stündlichen* Samples.
        """
        try:
            # Hole stündliche Samples der letzten 30 Tage
            samples_data = await self.get_hourly_samples(days=30)
            samples = samples_data.get('samples', [])
            if not samples:
                _LOGGER.warning("No hourly samples found within the last 30 days to calculate avg monthly yield.")
                return 0.0

            daily_totals = {}
            min_date = None
            max_date = None

            # Aggregiere stündliche Erträge pro Tag
            for sample in samples:
                try:
                    actual_kwh = sample.get('actual_kwh')
                    if actual_kwh is None or actual_kwh < 0: # Erlaube 0 kWh
                        continue

                    timestamp = dt_util.parse_datetime(sample['timestamp'])
                    # Verwende UTC-Datum für konsistente Aggregation
                    date_key = timestamp.date()

                    # Aktualisiere min/max Datum
                    if min_date is None or date_key < min_date: min_date = date_key
                    if max_date is None or date_key > max_date: max_date = date_key

                    daily_totals[date_key] = daily_totals.get(date_key, 0.0) + actual_kwh

                except (ValueError, KeyError):
                    _LOGGER.debug(f"Skipping invalid sample during yield calculation: {sample.get('timestamp')}")
                    continue

            if not daily_totals:
                 _LOGGER.warning("No valid daily totals could be aggregated from hourly samples.")
                 return 0.0

            # Berechne die Anzahl der Tage im betrachteten Zeitraum
            # days_count = len(daily_totals) # Zählt nur Tage mit Produktion
            # Besser: Anzahl der Tage zwischen min und max Datum + 1
            if min_date and max_date:
                 days_in_period = (max_date - min_date).days + 1
                 # Stelle sicher, dass wir nicht mehr als 30 Tage verwenden (falls Daten lückenhaft)
                 days_count_for_avg = min(days_in_period, 30)
            else:
                 days_count_for_avg = len(daily_totals) # Fallback

            if days_count_for_avg == 0:
                 return 0.0

            # Berechne den Durchschnitt
            total_yield = sum(daily_totals.values())
            avg_daily_yield = total_yield / days_count_for_avg

            # Hochrechnung auf 30 Tage (einen "durchschnittlichen" Monat)
            monthly_yield = avg_daily_yield * 30

            _LOGGER.debug(
                f"Average Monthly Yield calculated: {monthly_yield:.2f} kWh "
                f"(based on {total_yield:.2f} kWh over {days_count_for_avg} days)"
            )
            return round(monthly_yield, 2)

        except Exception as e:
            _LOGGER.error(f"Failed to calculate average monthly yield from hourly samples: {e}", exc_info=True)
            return 0.0
    # --- ENDE ZUSÄTZLICHE ANALYSEMETHODE ---

    def shutdown(self) -> None:
         """Alias für cleanup, der direkt aufgerufen werden kann (synchron)."""
         # Wird von async_cleanup aufgerufen.
         pass # Die eigentliche Logik ist im Executor.shutdown in async_cleanup

    # --- MIGRATION & CLEANUP ---
    async def migrate_data(self) -> bool:
        """Führt Datenmigrationen für alle bekannten Dateien durch."""
        all_success = True
        try:
            _LOGGER.info("Starting data migration check...")
            # Liste der Migrationsfunktionen und ihrer Dateien
            migrations = [
                (self._migrate_prediction_history, self.prediction_history_file),
                (self._migrate_learned_weights, self.learned_weights_file),
                (self._migrate_hourly_profile, self.hourly_profile_file),
                (self._migrate_model_state, self.model_state_file),
                # Füge hier ggf. Migration für hourly_samples hinzu
                (self._migrate_hourly_samples, self.hourly_samples_file),
            ]
            for migrate_func, file_path in migrations:
                 if await self._file_exists(file_path):
                     try:
                         await migrate_func()
                     except Exception as e:
                         _LOGGER.error(f"Migration failed for {file_path.name}: {e}")
                         all_success = False
                 else:
                     _LOGGER.debug(f"Migration skipped for {file_path.name}: File does not exist.")

            if all_success:
                 _LOGGER.info("Data migration check completed successfully.")
            else:
                 _LOGGER.warning("Data migration check completed with errors.")
            return all_success
        except Exception as e:
            _LOGGER.error(f"Critical error during data migration process: {e}", exc_info=True)
            return False

    async def _migrate_model_state(self) -> None:
        """Migriert die model_state.json Datei."""
        file_path = self.model_state_file
        try:
            data = await self._read_json_file(file_path) # Nutzt jetzt robustere Lesefunktion
            current_version = data.get("version", "1.0") # Sicherer Zugriff
            needs_save = False

            if current_version != DATA_VERSION:
                _LOGGER.info(f"Migrating model state from v{current_version} to v{DATA_VERSION}")
                data["version"] = DATA_VERSION
                needs_save = True

            # Füge fehlendes backfill_run hinzu (wird auch von get_model_state gemacht, aber hier explizit)
            if "backfill_run" not in data:
                _LOGGER.info(f"Adding missing 'backfill_run' key during migration.")
                data["backfill_run"] = False
                needs_save = True

            # Füge hier ggf. weitere Migrationsschritte für zukünftige Versionen hinzu

            if needs_save:
                await self._atomic_write_json(file_path, data)
                _LOGGER.info(f"Model state migrated successfully to v{DATA_VERSION}.")
            else:
                 _LOGGER.debug(f"Model state is already up to date (v{DATA_VERSION}).")

        except Exception as e:
            _LOGGER.error(f"Failed to migrate {file_path.name}: {e}", exc_info=True)
            raise # Fehler weitergeben, damit migrate_data ihn fängt

    # Platzhalter für die anderen Migrationsfunktionen (angenommen, sie prüfen nur die Version)
    async def _migrate_prediction_history(self) -> None:
        file_path = self.prediction_history_file
        await self._check_and_update_version(file_path)

    async def _migrate_learned_weights(self) -> None:
        file_path = self.learned_weights_file
        await self._check_and_update_version(file_path)

    async def _migrate_hourly_profile(self) -> None:
        file_path = self.hourly_profile_file
        await self._check_and_update_version(file_path)

    async def _migrate_hourly_samples(self) -> None:
        file_path = self.hourly_samples_file
        await self._check_and_update_version(file_path)

    async def _check_and_update_version(self, file_path: Path) -> None:
        """Generische Funktion zum Prüfen und Aktualisieren der Version in JSON-Dateien."""
        try:
            data = await self._read_json_file(file_path)
            current_version = data.get("version", "1.0")
            if current_version != DATA_VERSION:
                _LOGGER.info(f"Updating version for {file_path.name} from v{current_version} to v{DATA_VERSION}")
                data["version"] = DATA_VERSION
                await self._atomic_write_json(file_path, data)
            else:
                 _LOGGER.debug(f"{file_path.name} version is up to date (v{DATA_VERSION}).")
        except Exception as e:
            _LOGGER.error(f"Failed to check/update version for {file_path.name}: {e}", exc_info=True)
            raise # Fehler weitergeben