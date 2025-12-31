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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

TRACKED_HOURS = range(6, 21)
HISTORY_DAYS = 7


class DniTracker:
    """Track max DNI per hour over 7 days for dni_ratio feature @zara"""

    def __init__(self, storage_path: Optional[Path] = None, auto_load: bool = False):
        """Initialize tracker with optional storage path @zara

        Args:
            storage_path: Path to JSON storage file
            auto_load: If True, load synchronously on init.
        """
        self.storage_path = storage_path
        self.max_dni: Dict[int, float] = {h: 0.0 for h in TRACKED_HOURS}
        self.history: Dict[int, List[float]] = {h: [] for h in TRACKED_HOURS}
        self.last_updated: Optional[str] = None
        self._loaded = False

        if auto_load and storage_path and storage_path.exists():
            self._load_sync()

        _LOGGER.info("DniTracker initialized")

    async def async_load(self):
        """Load tracker data asynchronously (non-blocking) @zara"""
        if self._loaded or not self.storage_path or not self.storage_path.exists():
            return

        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)
        self._loaded = True

    def get_max_dni(self, hour: int) -> float:
        """Get max DNI for hour over last 7 days @zara"""
        if hour not in self.max_dni:
            return 0.0
        return self.max_dni.get(hour, 0.0)

    def record_dni(self, hour: int, dni: float):
        """Record DNI value for hour (called during day) @zara"""
        if hour not in TRACKED_HOURS:
            return
        if dni < 0:
            return

        hist = self.history.get(hour, [])
        if len(hist) == 0 or dni > hist[-1]:
            if len(hist) == 0:
                hist.append(dni)
            else:
                hist[-1] = dni
            self.history[hour] = hist

    def end_of_day_update(self):
        """Finalize today's values and update max (call at 23:30) @zara"""
        for hour in TRACKED_HOURS:
            hist = self.history.get(hour, [])

            if len(hist) == 0 or hist[-1] == 0.0:
                hist.append(0.0)

            if len(hist) > HISTORY_DAYS:
                hist = hist[-HISTORY_DAYS:]

            self.history[hour] = hist
            self.max_dni[hour] = max(hist) if hist else 0.0

        self.last_updated = datetime.now().isoformat()
        self.save()

        _LOGGER.info("DNI tracker end-of-day update complete")

    def start_new_day(self):
        """Prepare for new day by adding placeholder @zara"""
        for hour in TRACKED_HOURS:
            hist = self.history.get(hour, [])
            hist.append(0.0)
            if len(hist) > HISTORY_DAYS:
                hist = hist[-HISTORY_DAYS:]
            self.history[hour] = hist

    def save(self):
        """Persist tracker data (sync version, use async_save if possible) @zara"""
        if not self.storage_path:
            return

        try:
            self._save_sync()
            _LOGGER.debug("DNI tracker saved")
        except Exception as e:
            _LOGGER.error(f"Failed to save DNI tracker: {e}")

    def _save_sync(self):
        """Internal sync save method @zara"""
        data = {
            "version": "1.0",
            "last_updated": self.last_updated,
            "max_dni": {str(k): v for k, v in self.max_dni.items()},
            "history": {str(k): v for k, v in self.history.items()},
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    async def async_save(self, hass):
        """Persist tracker data asynchronously (non-blocking) @zara"""
        if not self.storage_path:
            return

        try:
            await hass.async_add_executor_job(self._save_sync)
            _LOGGER.debug("DNI tracker saved")
        except Exception as e:
            _LOGGER.error(f"Failed to save DNI tracker: {e}")

    def _load_sync(self):
        """Load tracker data from storage synchronously @zara"""
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            self.last_updated = data.get("last_updated")

            if "max_dni" in data:
                for k, v in data["max_dni"].items():
                    self.max_dni[int(k)] = float(v)

            if "history" in data:
                for k, v in data["history"].items():
                    self.history[int(k)] = [float(x) for x in v]

            self._loaded = True
            _LOGGER.info("DNI tracker loaded")
        except Exception as e:
            _LOGGER.warning(f"Could not load DNI tracker: {e}")

    def get_stats(self) -> Dict[str, any]:
        """Get statistics for diagnostics @zara"""
        return {
            "last_updated": self.last_updated,
            "max_dni": self.max_dni.copy(),
            "history_lengths": {h: len(v) for h, v in self.history.items()},
        }
