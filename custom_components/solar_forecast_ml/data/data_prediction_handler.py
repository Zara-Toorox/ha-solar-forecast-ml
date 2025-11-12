"""
Data Prediction Handler for Solar Forecast ML Integration

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
from typing import Dict, Any, Optional, List

from homeassistant.core import HomeAssistant

from .data_io import DataManagerIO
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..const import DATA_VERSION, MAX_PREDICTION_HISTORY
from ..ml.ml_types import validate_prediction_record

_LOGGER = logging.getLogger(__name__)


class DataPredictionHandler(DataManagerIO):
    """Handles prediction history tracking and accuracy calculations"""

    def __init__(self, hass: HomeAssistant, data_dir: Path):
        super().__init__(hass, data_dir)
        
        self.prediction_history_file = self.data_dir / "stats" / "prediction_history.json"
        
        self._prediction_history_default = {
            "version": DATA_VERSION,
            "predictions": [],
            "last_updated": None
        }

    async def ensure_prediction_history_file(self) -> None:
        """Ensure prediction history file exists"""
        if not self.prediction_history_file.exists():
            await self._atomic_write_json(
                self.prediction_history_file,
                self._prediction_history_default
            )

    # ═════════════════════════════════════════════════════════════
    # PREDICTION HISTORY Methods - REMOVED
    # prediction_history.json is obsolete - hourly_predictions.json is the single source of truth
    # ═════════════════════════════════════════════════════════════

    async def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Get most recent prediction"""
        try:
            predictions = await self.get_predictions(limit=1)
            return predictions[0] if predictions else None
        except Exception:
            return None

    async def get_prediction_for_date(self, date: str) -> Optional[Dict[str, Any]]:
        """Get prediction for specific date"""
        try:
            predictions = await self.get_predictions()
            for pred in reversed(predictions):
                if pred.get("date") == date:
                    return pred
            return None
        except Exception:
            return None

    async def cleanup_old_predictions(self, days: int = 365) -> bool:
        """Remove predictions older than specified days"""
        try:
            # Get the lock specific to this file for read-modify-write operation
            file_lock = await self._get_file_lock(self.prediction_history_file)

            async with file_lock:
                history = await self._read_json_file(
                    self.prediction_history_file,
                    self._prediction_history_default
                )

                cutoff_date = (dt_util.now() - dt_util.timedelta(days=days)).date().isoformat()

                original_count = len(history["predictions"])
                history["predictions"] = [
                    pred for pred in history["predictions"]
                    if pred.get("date", "") >= cutoff_date
                ]

                removed_count = original_count - len(history["predictions"])

                if removed_count > 0:
                    history["last_updated"] = dt_util.now().isoformat()
                    # Use unlocked version since we already hold the lock
                    await self._atomic_write_json_unlocked(self.prediction_history_file, history)
                    _LOGGER.info(f"Removed {removed_count} old predictions (older than {days} days)")

                return True

        except Exception as e:
            _LOGGER.error(f"Failed to cleanup old predictions: {e}")
            return False

    # ═════════════════════════════════════════════════════════════
    # ACCURACY CALCULATIONS
    # ═════════════════════════════════════════════════════════════

    async def calculate_accuracy_stats(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Calculate accuracy statistics for recent predictions"""
        try:
            predictions = await self.get_predictions()
            
            if not predictions:
                return {
                    "count": 0,
                    "avg_accuracy": 0.0,
                    "median_accuracy": 0.0,
                    "min_accuracy": 0.0,
                    "max_accuracy": 0.0,
                    "calculated_at": dt_util.now().isoformat()
                }
            
            # Filter to recent predictions with accuracy data
            cutoff_date = (dt_util.now() - dt_util.timedelta(days=days)).date().isoformat()
            recent = [
                pred for pred in predictions
                if pred.get("date", "") >= cutoff_date
                and pred.get("accuracy") is not None
            ]
            
            if not recent:
                return {
                    "count": 0,
                    "avg_accuracy": 0.0,
                    "median_accuracy": 0.0,
                    "min_accuracy": 0.0,
                    "max_accuracy": 0.0,
                    "calculated_at": dt_util.now().isoformat()
                }
            
            accuracies = [pred["accuracy"] for pred in recent]
            
            # Calculate statistics
            avg_accuracy = sum(accuracies) / len(accuracies)
            sorted_acc = sorted(accuracies)
            median_accuracy = sorted_acc[len(sorted_acc) // 2]
            min_accuracy = min(accuracies)
            max_accuracy = max(accuracies)
            
            return {
                "count": len(recent),
                "avg_accuracy": round(avg_accuracy, 1),
                "median_accuracy": round(median_accuracy, 1),
                "min_accuracy": round(min_accuracy, 1),
                "max_accuracy": round(max_accuracy, 1),
                "period_days": days,
                "calculated_at": dt_util.now().isoformat()
            }
            
        except Exception as e:
            _LOGGER.error(f"Failed to calculate accuracy stats: {e}")
            return {
                "count": 0,
                "avg_accuracy": 0.0,
                "median_accuracy": 0.0,
                "min_accuracy": 0.0,
                "max_accuracy": 0.0,
                "error": str(e),
                "calculated_at": dt_util.now().isoformat()
            }

    async def get_accuracy_trend(
        self,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get daily accuracy trend for recent period"""
        try:
            predictions = await self.get_predictions()
            
            cutoff_date = (dt_util.now() - dt_util.timedelta(days=days)).date().isoformat()
            recent = [
                pred for pred in predictions
                if pred.get("date", "") >= cutoff_date
                and pred.get("accuracy") is not None
            ]
            
            # Build trend data
            trend = []
            for pred in recent:
                trend.append({
                    "date": pred.get("date"),
                    "accuracy": round(pred.get("accuracy", 0.0), 1),
                    "predicted_kwh": pred.get("predicted_kwh"),
                    "actual_kwh": pred.get("actual_kwh"),
                    "source": pred.get("source")
                })
            
            return trend
            
        except Exception as e:
            _LOGGER.error(f"Failed to get accuracy trend: {e}")
            return []

    async def get_predictions_count(self) -> int:
        """Get total count of predictions"""
        try:
            history = await self._read_json_file(
                self.prediction_history_file,
                self._prediction_history_default
            )
            return len(history.get("predictions", []))
        except Exception:
            return 0

    async def update_today_predictions_actual(
        self,
        actual_value: float,
        accuracy: Optional[float] = None
    ) -> bool:
        """Update todays prediction records with actual value and accuracy"""
        try:
            today_date = dt_util.now().date().isoformat()

            file_lock = await self._get_file_lock(self.prediction_history_file)

            async with file_lock:
                history = await self._read_json_file(
                    self.prediction_history_file,
                    self._prediction_history_default
                )

                predictions = history.get("predictions", [])
                updated_count = 0

                # Update all predictions for today
                for pred in predictions:
                    if pred.get("date") == today_date:
                        pred["actual_value"] = actual_value
                        if accuracy is not None:
                            pred["accuracy"] = accuracy
                        updated_count += 1

                if updated_count > 0:
                    history["last_updated"] = dt_util.now().isoformat()
                    await self._atomic_write_json_unlocked(self.prediction_history_file, history)
                    _LOGGER.debug(f"Updated {updated_count} prediction(s) for {today_date} with actual value {actual_value:.2f} kWh")
                    return True
                else:
                    _LOGGER.debug(f"No predictions found for {today_date} to update")
                    return False

        except Exception as e:
            _LOGGER.error(f"Failed to update today's predictions: {e}", exc_info=True)
            return False
