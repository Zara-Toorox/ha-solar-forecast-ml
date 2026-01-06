# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import json
import logging
from pathlib import Path
from typing import Dict, Optional

_LOGGER = logging.getLogger(__name__)

DEFAULT_FACTORS = {
    1: 0.85,   # January
    2: 0.90,   # February
    3: 0.95,   # March
    4: 1.00,   # April
    5: 1.05,   # May
    6: 1.10,   # June
    7: 1.10,   # July
    8: 1.05,   # August
    9: 1.00,   # September
    10: 0.95,  # October
    11: 0.90,  # November
    12: 0.85,  # December
}


class SeasonalAdjuster:
    """Monthly adjustment factors for solar production @zara"""

    def __init__(self, storage_path: Optional[Path] = None, auto_load: bool = False):
        """Initialize with optional storage path @zara

        Args:
            storage_path: Path to JSON storage file
            auto_load: If True, load synchronously on init.
        """
        self.storage_path = storage_path
        self.factors: Dict[int, float] = DEFAULT_FACTORS.copy()
        self.sample_counts: Dict[int, int] = {m: 0 for m in range(1, 13)}
        self._loaded = False

        if auto_load and storage_path and storage_path.exists():
            self._load_sync()

        _LOGGER.info("SeasonalAdjuster initialized")

    async def async_load(self):
        """Load factors asynchronously (non-blocking) @zara"""
        if self._loaded or not self.storage_path or not self.storage_path.exists():
            return

        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)
        self._loaded = True

    def get_factor(self, month: int) -> float:
        """Get adjustment factor for month (1-12) @zara"""
        if month < 1 or month > 12:
            return 1.0
        return self.factors.get(month, 1.0)

    def update(self, month: int, actual: float, predicted: float):
        """Update factor based on actual vs predicted @zara"""
        if month < 1 or month > 12:
            return
        if predicted <= 0 or actual < 0:
            return

        ratio = actual / predicted
        ratio = max(0.5, min(1.5, ratio))

        count = self.sample_counts.get(month, 0)
        current = self.factors.get(month, 1.0)

        alpha = 0.1 if count < 10 else 0.05
        self.factors[month] = current * (1 - alpha) + ratio * alpha
        self.sample_counts[month] = count + 1

    def save(self):
        """Persist factors to storage (sync version, use async_save if possible) @zara"""
        if not self.storage_path:
            return

        try:
            self._save_sync()
            _LOGGER.debug("Seasonal factors saved")
        except Exception as e:
            _LOGGER.error(f"Failed to save seasonal factors: {e}")

    def _save_sync(self):
        """Internal sync save method @zara"""
        data = {
            "version": "1.0",
            "factors": {str(k): v for k, v in self.factors.items()},
            "sample_counts": {str(k): v for k, v in self.sample_counts.items()},
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    async def async_save(self, hass):
        """Persist factors to storage asynchronously (non-blocking) @zara"""
        if not self.storage_path:
            return

        try:
            await hass.async_add_executor_job(self._save_sync)
            _LOGGER.debug("Seasonal factors saved")
        except Exception as e:
            _LOGGER.error(f"Failed to save seasonal factors: {e}")

    def _load_sync(self):
        """Load factors from storage synchronously @zara"""
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            if "factors" in data:
                for k, v in data["factors"].items():
                    self.factors[int(k)] = float(v)

            if "sample_counts" in data:
                for k, v in data["sample_counts"].items():
                    self.sample_counts[int(k)] = int(v)

            self._loaded = True
            _LOGGER.info("Seasonal factors loaded")
        except Exception as e:
            _LOGGER.warning(f"Could not load seasonal factors: {e}")

    def get_stats(self) -> Dict[str, any]:
        """Get statistics for diagnostics @zara"""
        return {
            "factors": self.factors.copy(),
            "sample_counts": self.sample_counts.copy(),
            "min_factor": min(self.factors.values()),
            "max_factor": max(self.factors.values()),
        }
