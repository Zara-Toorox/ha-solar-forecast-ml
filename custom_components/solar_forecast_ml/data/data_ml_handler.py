# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from ..const import DATA_VERSION
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..ai import (
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

    def __init__(self, hass: HomeAssistant, data_dir: Path, data_manager=None):
        super().__init__(hass, data_dir)

        self.data_adapter = TypedDataAdapter()
        self.data_manager = data_manager

        self.learned_weights_file = self.data_dir / "ai" / "learned_weights.json"
        self.hourly_profile_file = self.data_dir / "ai" / "seasonal.json"
        self.model_state_file = self.data_dir / "ai" / "dni_tracker.json"

        self._model_state_default = {
            "version": DATA_VERSION,
            "model_loaded": False,
            "last_training": None,
            "training_samples": 0,
            "current_accuracy": 0.0,
            "status": "uninitialized",
        }

    async def backup_learned_weights(self) -> bool:
        """Create a timestamped backup of current learned weights before saving new ones @zara"""
        try:
            if not self.learned_weights_file.exists():
                return True

            timestamp = dt_util.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.learned_weights_file.parent / "backups"
            await self._ensure_directory_exists(backup_dir)

            backup_file = backup_dir / f"learned_weights_{timestamp}.json"

            import shutil

            await self.hass.async_add_executor_job(
                shutil.copy2, str(self.learned_weights_file), str(backup_file)
            )

            _LOGGER.info(f"Backup created: {backup_file.name}")

            backups = sorted(backup_dir.glob("learned_weights_*.json"), reverse=True)
            for old_backup in backups[5:]:
                await self.hass.async_add_executor_job(old_backup.unlink)
                _LOGGER.debug(f"Deleted old backup: {old_backup.name}")

            return True
        except Exception as e:
            _LOGGER.error(f"Failed to create backup: {e}")
            return False

    async def save_learned_weights(self, weights: LearnedWeights) -> bool:
        """Save learned weights to file with automatic backup @zara"""
        try:

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
        """Load learned weights from file @zara"""
        try:

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
        """Get learned weights Alias for load_learned_weights @zara"""
        return await self.load_learned_weights()

    async def delete_learned_weights(self) -> bool:
        """Delete learned weights file @zara"""
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

    async def save_hourly_profile(self, profile: HourlyProfile) -> bool:
        """Save hourly profile to file @zara"""
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
        """Load hourly profile from file @zara"""
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
        """Get hourly profile Alias for load_hourly_profile @zara"""
        return await self.load_hourly_profile()

    async def save_model_state(self, state: Dict[str, Any]) -> bool:
        """Save model state to file @zara"""
        try:
            await self._ensure_directory_exists(self.model_state_file.parent)
            await self._atomic_write_json(self.model_state_file, state)
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to save model state: {e}")
            return False

    async def load_model_state(self) -> Dict[str, Any]:
        """Load model state from file @zara"""
        try:
            return await self._read_json_file(self.model_state_file, self._model_state_default)
        except Exception as e:
            _LOGGER.error(f"Failed to load model state: {e}")
            return self._model_state_default

    async def get_model_state(self) -> Dict[str, Any]:
        """Get model state Alias for load_model_state @zara"""
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

    async def get_hourly_samples(
        self,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get hourly samples for ML training

        V3: Reads from hourly_predictions.json (predictions with actual_kwh)
        This is the single source of truth for training data.
        """
        try:

            if self.data_manager:
                hourly_predictions_file = self.data_manager.hourly_predictions_file
            else:

                hourly_predictions_file = self.data_dir / "stats" / "hourly_predictions.json"

            if not hourly_predictions_file.exists():
                _LOGGER.warning("hourly_predictions.json not found")
                return []

            data = await self._read_json_file(hourly_predictions_file, None)
            if not data:
                _LOGGER.warning("Failed to load hourly_predictions.json")
                return []

            predictions = data.get("predictions", [])
            samples = [p for p in predictions if p.get("actual_kwh") is not None]

            if start_date or end_date:
                filtered = []
                for sample in samples:
                    timestamp = sample.get("target_datetime", sample.get("id", ""))
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue
                    filtered.append(sample)
                samples = filtered

            if limit:
                samples = samples[-limit:]

            return samples

        except Exception as e:
            _LOGGER.error(f"Failed to get hourly samples from hourly_predictions: {e}")
            return []

    async def get_hourly_samples_count(self) -> int:
        """Get count of hourly samples available for ML training @zara"""
        try:
            samples = await self.get_hourly_samples()
            return len(samples)
        except Exception:
            return 0
