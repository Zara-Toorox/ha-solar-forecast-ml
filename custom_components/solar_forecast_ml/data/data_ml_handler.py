"""
Data ML Handler for Solar Forecast ML Integration

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
from pathlib import Path
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from ..const import DATA_VERSION
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..ml.ml_types import (
    HourlyProfile,
    LearnedWeights,
    create_default_hourly_profile,
    create_default_learned_weights,
)
from .data_adapter import TypedDataAdapter
from .data_io import DataManagerIO

_LOGGER = logging.getLogger(__name__)


class DataMLHandler(DataManagerIO):
    """Handles ML-specific data weights profiles model state samples"""

    def __init__(self, hass: HomeAssistant, data_dir: Path):
        super().__init__(hass, data_dir)

        self.data_adapter = TypedDataAdapter()

        # ML files
        self.learned_weights_file = self.data_dir / "ml" / "learned_weights.json"
        self.hourly_profile_file = self.data_dir / "ml" / "hourly_profile.json"
        self.model_state_file = self.data_dir / "ml" / "model_state.json"
        self.hourly_samples_file = self.data_dir / "ml" / "hourly_samples.json"

        # Defaults
        self._model_state_default = {
            "version": DATA_VERSION,
            "model_loaded": False,
            "last_training": None,
            "training_samples": 0,
            "current_accuracy": 0.0,
            "status": "uninitialized",
        }

        self._hourly_samples_default = {
            "version": DATA_VERSION,
            "samples": [],
            "count": 0,
            "last_updated": None,
        }

    async def ensure_ml_files(self) -> None:
        """Ensure ML files exist with defaults"""
        # IMPORTANT: Do NOT create learned_weights.json if it doesn't exist
        # V2 training will create it with 44 features when training starts
        if not self.learned_weights_file.exists():
            _LOGGER.info(
                "learned_weights.json does not exist - will be created by V2 training (44 features)"
            )

        if not self.hourly_profile_file.exists():
            default_profile = create_default_hourly_profile()
            await self.save_hourly_profile(default_profile)

        if not self.model_state_file.exists():
            await self._atomic_write_json(self.model_state_file, self._model_state_default)

        if not self.hourly_samples_file.exists():
            await self._atomic_write_json(self.hourly_samples_file, self._hourly_samples_default)

    # =============================================================
    # LEARNED WEIGHTS Methods
    # =============================================================

    async def backup_learned_weights(self) -> bool:
        """Create a timestamped backup of current learned weights before saving new ones"""
        try:
            if not self.learned_weights_file.exists():
                return True  # Nothing to backup

            # Create backup filename with timestamp
            timestamp = dt_util.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.learned_weights_file.parent / "backups"
            await self._ensure_directory_exists(backup_dir)

            backup_file = backup_dir / f"learned_weights_{timestamp}.json"

            # Copy current weights to backup
            import shutil

            await self.hass.async_add_executor_job(
                shutil.copy2, str(self.learned_weights_file), str(backup_file)
            )

            _LOGGER.info(f"Backup created: {backup_file.name}")

            # Clean up old backups (keep last 5)
            backups = sorted(backup_dir.glob("learned_weights_*.json"), reverse=True)
            for old_backup in backups[5:]:
                await self.hass.async_add_executor_job(old_backup.unlink)
                _LOGGER.debug(f"Deleted old backup: {old_backup.name}")

            return True
        except Exception as e:
            _LOGGER.error(f"Failed to create backup: {e}")
            return False

    async def save_learned_weights(self, weights: LearnedWeights) -> bool:
        """Save learned weights to file with automatic backup"""
        try:
            # Create backup before overwriting
            await self.backup_learned_weights()

            weights_dict = self.data_adapter.learned_weights_to_dict(weights)
            await self._ensure_directory_exists(self.learned_weights_file.parent)
            await self._atomic_write_json(self.learned_weights_file, weights_dict)
            _LOGGER.info(
                "Learned weights saved successfully (accuracy: {:.1f}%, samples: {})".format(
                    weights.accuracy * 100, weights.training_samples
                )
            )
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to save learned weights: {e}")
            return False

    async def load_learned_weights(self) -> Optional[LearnedWeights]:
        """Load learned weights from file"""
        try:
            # Don't pass default_structure - return None if file doesn't exist
            # V2 training will create the file when needed
            if not self.learned_weights_file.exists():
                _LOGGER.debug("learned_weights.json does not exist - returning None")
                return None

            data = await self._read_json_file(self.learned_weights_file, None)
            if data:
                return self.data_adapter.dict_to_learned_weights(data)
            return None
        except Exception as e:
            _LOGGER.error(f"Failed to load learned weights: {e}")
            return None

    async def get_learned_weights(self) -> Optional[LearnedWeights]:
        """Get learned weights Alias for load_learned_weights"""
        return await self.load_learned_weights()

    async def delete_learned_weights(self) -> bool:
        """Delete learned weights file"""
        try:
            if self.learned_weights_file.exists():
                await self.hass.async_add_executor_job(self.learned_weights_file.unlink)
                _LOGGER.info("Learned weights file deleted successfully")
                return True
            else:
                _LOGGER.debug("Learned weights file does not exist, nothing to delete")
                return True
        except Exception as e:
            _LOGGER.error(f"Failed to delete learned weights: {e}")
            return False

    # =============================================================
    # HOURLY PROFILE Methods
    # =============================================================

    async def save_hourly_profile(self, profile: HourlyProfile) -> bool:
        """Save hourly profile to file"""
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
        """Load hourly profile from file"""
        try:
            data = await self._read_json_file(
                self.hourly_profile_file, create_default_hourly_profile()
            )
            if data:
                return self.data_adapter.dict_to_hourly_profile(data)
            return None
        except Exception as e:
            _LOGGER.error(f"Failed to load hourly profile: {e}")
            return None

    async def get_hourly_profile(self) -> Optional[HourlyProfile]:
        """Get hourly profile Alias for load_hourly_profile"""
        return await self.load_hourly_profile()

    # =============================================================
    # MODEL STATE Methods
    # =============================================================

    async def save_model_state(self, state: Dict[str, Any]) -> bool:
        """Save model state to file"""
        try:
            await self._ensure_directory_exists(self.model_state_file.parent)
            await self._atomic_write_json(self.model_state_file, state)
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to save model state: {e}")
            return False

    async def load_model_state(self) -> Dict[str, Any]:
        """Load model state from file"""
        try:
            return await self._read_json_file(self.model_state_file, self._model_state_default)
        except Exception as e:
            _LOGGER.error(f"Failed to load model state: {e}")
            return self._model_state_default

    async def get_model_state(self) -> Dict[str, Any]:
        """Get model state Alias for load_model_state"""
        return await self.load_model_state()

    async def update_model_state(
        self,
        model_loaded: Optional[bool] = None,
        last_training: Optional[str] = None,
        training_samples: Optional[int] = None,
        current_accuracy: Optional[float] = None,
        status: Optional[str] = None,
    ) -> bool:
        """Update model state partially"""
        try:
            state = await self.load_model_state()

            if model_loaded is not None:
                state["model_loaded"] = model_loaded
            if last_training is not None:
                state["last_training"] = last_training
            if training_samples is not None:
                state["training_samples"] = training_samples
            if current_accuracy is not None:
                state["current_accuracy"] = round(float(current_accuracy), 2)
            if status is not None:
                state["status"] = status

            return await self.save_model_state(state)

        except Exception as e:
            _LOGGER.error(f"Failed to update model state: {e}")
            return False

    # =============================================================
    # HOURLY SAMPLES Methods
    # =============================================================

    async def add_hourly_sample(self, sample: Dict[str, Any]) -> bool:
        """Add hourly sample for ML training"""
        try:
            # Get the lock specific to this file for read-modify-write operation
            file_lock = await self._get_file_lock(self.hourly_samples_file)

            async with file_lock:
                data = await self._read_json_file(
                    self.hourly_samples_file, self._hourly_samples_default
                )

                if "samples" not in data:
                    data["samples"] = []

                # Add timestamp if not present
                if "timestamp" not in sample:
                    sample["timestamp"] = dt_util.now().isoformat()

                data["samples"].append(sample)
                data["count"] = len(data["samples"])
                data["last_updated"] = dt_util.now().isoformat()

                # Keep last 10000 samples
                if len(data["samples"]) > 10000:
                    data["samples"] = data["samples"][-10000:]
                    data["count"] = len(data["samples"])

                # Use unlocked version since we already hold the lock
                await self._atomic_write_json_unlocked(self.hourly_samples_file, data)

                _LOGGER.debug(f"Hourly sample added (total: {data['count']})")
                return True

        except Exception as e:
            _LOGGER.error(f"Failed to add hourly sample: {e}", exc_info=True)
            return False

    async def get_hourly_samples(
        self,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get hourly samples with optional filtering"""
        try:
            data = await self._read_json_file(
                self.hourly_samples_file, self._hourly_samples_default
            )

            samples = data.get("samples", [])

            # Filter by date range if provided
            if start_date or end_date:
                filtered = []
                for sample in samples:
                    timestamp = sample.get("timestamp", "")
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue
                    filtered.append(sample)
                samples = filtered

            # Apply limit
            if limit:
                samples = samples[-limit:]

            return samples

        except Exception as e:
            _LOGGER.error(f"Failed to get hourly samples: {e}")
            return []

    async def clear_hourly_samples(self) -> bool:
        """Clear all hourly samples"""
        try:
            data = self._hourly_samples_default.copy()
            data["last_updated"] = dt_util.now().isoformat()

            await self._atomic_write_json(self.hourly_samples_file, data)
            _LOGGER.info("Hourly samples cleared")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to clear hourly samples: {e}")
            return False

    async def get_hourly_samples_count(self) -> int:
        """Get count of hourly samples"""
        try:
            data = await self._read_json_file(
                self.hourly_samples_file, self._hourly_samples_default
            )
            return data.get("count", 0)
        except Exception:
            return 0

    async def cleanup_duplicate_samples(self) -> Dict[str, int]:
        """Remove duplicate samples based on timestamp"""
        try:
            # Get the lock specific to this file for read-modify-write operation
            file_lock = await self._get_file_lock(self.hourly_samples_file)

            async with file_lock:
                data = await self._read_json_file(
                    self.hourly_samples_file, self._hourly_samples_default
                )

                samples = data.get("samples", [])
                initial_count = len(samples)

                # Remove duplicates based on timestamp
                seen_timestamps = set()
                unique_samples = []

                for sample in samples:
                    timestamp = sample.get("timestamp")
                    if timestamp and timestamp not in seen_timestamps:
                        seen_timestamps.add(timestamp)
                        unique_samples.append(sample)

                data["samples"] = unique_samples
                data["count"] = len(unique_samples)
                data["last_updated"] = dt_util.now().isoformat()

                # Use unlocked version since we already hold the lock
                await self._atomic_write_json_unlocked(self.hourly_samples_file, data)

                removed = initial_count - len(unique_samples)
                _LOGGER.info(
                    f"Cleanup: Removed {removed} duplicate samples, {len(unique_samples)} remaining"
                )

                return {"removed": removed, "remaining": len(unique_samples)}

        except Exception as e:
            _LOGGER.error(f"Failed to cleanup duplicate samples: {e}", exc_info=True)
            return {"removed": 0, "remaining": 0}

    async def cleanup_zero_production_samples(self) -> Dict[str, int]:
        """Remove samples with zero or None production"""
        try:
            # Get the lock specific to this file for read-modify-write operation
            file_lock = await self._get_file_lock(self.hourly_samples_file)

            async with file_lock:
                data = await self._read_json_file(
                    self.hourly_samples_file, self._hourly_samples_default
                )

                samples = data.get("samples", [])
                initial_count = len(samples)

                # Remove zero production samples
                valid_samples = [
                    sample
                    for sample in samples
                    if sample.get("actual_kwh") is not None and sample.get("actual_kwh") > 0
                ]

                data["samples"] = valid_samples
                data["count"] = len(valid_samples)
                data["last_updated"] = dt_util.now().isoformat()

                # Use unlocked version since we already hold the lock
                await self._atomic_write_json_unlocked(self.hourly_samples_file, data)

                removed = initial_count - len(valid_samples)
                _LOGGER.info(
                    f"Cleanup: Removed {removed} zero-production samples, {len(valid_samples)} remaining"
                )

                return {"removed": removed, "remaining": len(valid_samples)}

        except Exception as e:
            _LOGGER.error(f"Failed to cleanup zero-production samples: {e}", exc_info=True)
            return {"removed": 0, "remaining": 0}
