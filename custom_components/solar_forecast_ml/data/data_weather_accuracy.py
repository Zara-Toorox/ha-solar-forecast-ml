"""
Weather Forecast Accuracy Tracker

Tracks predicted vs actual weather conditions to measure forecast quality.
Useful for understanding weather provider reliability.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from homeassistant.core import HomeAssistant

from .data_io import DataManagerIO
from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class WeatherAccuracyTracker(DataManagerIO):
    """Tracks weather forecast accuracy over time"""

    def __init__(self, hass: HomeAssistant, data_dir: Path):
        super().__init__(hass, data_dir)
        self.weather_accuracy_file = self.data_dir / "stats" / "weather_accuracy.json"

        self._default_structure = {
            "version": "1.0",
            "retention_days": 30,
            "entries": [],  # List of accuracy measurements
            "statistics": {
                "last_7_days": {
                    "avg_cloud_error": None,
                    "avg_temp_error": None,
                    "sample_count": 0
                },
                "last_30_days": {
                    "avg_cloud_error": None,
                    "avg_temp_error": None,
                    "sample_count": 0
                }
            },
            "metadata": {
                "last_update": None,
                "total_entries": 0
            }
        }

    async def ensure_file(self) -> None:
        """Ensure weather accuracy file exists"""
        if not self.weather_accuracy_file.exists():
            await self._atomic_write_json(self.weather_accuracy_file, self._default_structure)
            _LOGGER.info("Created weather_accuracy.json")

    async def record_forecast_vs_actual(
        self,
        forecast_datetime: datetime,
        predicted_data: Dict[str, Any],
        actual_data: Dict[str, Any]
    ) -> bool:
        """Record a comparison between predicted and actual weather

        Args:
            forecast_datetime: The datetime this forecast was for
            predicted_data: Weather data that was forecasted
            actual_data: Weather data that actually occurred

        Returns:
            True if successfully saved
        """
        try:
            await self.ensure_file()
            data = await self._read_json_file(self.weather_accuracy_file, self._default_structure)

            # Calculate errors
            cloud_error = None
            temp_error = None

            if "cloud_cover" in predicted_data and "cloud_cover" in actual_data:
                cloud_error = abs(predicted_data["cloud_cover"] - actual_data["cloud_cover"])

            if "temperature" in predicted_data and "temperature" in actual_data:
                temp_error = abs(predicted_data["temperature"] - actual_data["temperature"])

            # Create entry
            entry = {
                "forecast_datetime": forecast_datetime.isoformat(),
                "recorded_at": dt_util.now().isoformat(),
                "predicted": {
                    "cloud_cover": predicted_data.get("cloud_cover"),
                    "temperature": predicted_data.get("temperature"),
                    "condition": predicted_data.get("condition"),
                },
                "actual": {
                    "cloud_cover": actual_data.get("cloud_cover"),
                    "temperature": actual_data.get("temperature"),
                    "condition": actual_data.get("condition"),
                },
                "errors": {
                    "cloud_cover_abs": cloud_error,
                    "temperature_abs": temp_error,
                }
            }

            # Add to entries
            data["entries"].append(entry)

            # Remove old entries (> retention_days)
            retention_days = data.get("retention_days", 30)
            cutoff = dt_util.now() - timedelta(days=retention_days)

            data["entries"] = [
                e for e in data["entries"]
                if dt_util.parse_datetime(e["recorded_at"]) >= cutoff
            ]

            # Update statistics
            await self._calculate_statistics(data)

            # Update metadata
            data["metadata"]["last_update"] = dt_util.now().isoformat()
            data["metadata"]["total_entries"] = len(data["entries"])

            await self._atomic_write_json(self.weather_accuracy_file, data)

            _LOGGER.debug(
                f"Weather accuracy recorded: cloud_error={cloud_error}%, temp_error={temp_error}°C"
            )
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to record weather accuracy: {e}", exc_info=True)
            return False

    async def _calculate_statistics(self, data: Dict[str, Any]) -> None:
        """Calculate accuracy statistics for different time periods"""
        now = dt_util.now()

        for period_name, days in [("last_7_days", 7), ("last_30_days", 30)]:
            cutoff = now - timedelta(days=days)

            relevant_entries = [
                e for e in data["entries"]
                if dt_util.parse_datetime(e["recorded_at"]) >= cutoff
            ]

            if relevant_entries:
                cloud_errors = [
                    e["errors"]["cloud_cover_abs"]
                    for e in relevant_entries
                    if e["errors"]["cloud_cover_abs"] is not None
                ]

                temp_errors = [
                    e["errors"]["temperature_abs"]
                    for e in relevant_entries
                    if e["errors"]["temperature_abs"] is not None
                ]

                data["statistics"][period_name] = {
                    "avg_cloud_error": round(sum(cloud_errors) / len(cloud_errors), 1) if cloud_errors else None,
                    "avg_temp_error": round(sum(temp_errors) / len(temp_errors), 1) if temp_errors else None,
                    "sample_count": len(relevant_entries)
                }
            else:
                data["statistics"][period_name] = {
                    "avg_cloud_error": None,
                    "avg_temp_error": None,
                    "sample_count": 0
                }

    async def get_statistics(self) -> Optional[Dict[str, Any]]:
        """Get current weather accuracy statistics

        Returns:
            Dictionary with statistics or None if not available
        """
        try:
            data = await self._read_json_file(self.weather_accuracy_file, self._default_structure)
            return data.get("statistics")
        except Exception as e:
            _LOGGER.error(f"Failed to get weather accuracy statistics: {e}")
            return None

    async def get_recent_entries(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent accuracy entries

        Args:
            days: Number of days to look back

        Returns:
            List of accuracy entries
        """
        try:
            data = await self._read_json_file(self.weather_accuracy_file, self._default_structure)
            cutoff = dt_util.now() - timedelta(days=days)

            return [
                e for e in data.get("entries", [])
                if dt_util.parse_datetime(e["recorded_at"]) >= cutoff
            ]
        except Exception as e:
            _LOGGER.error(f"Failed to get recent weather accuracy entries: {e}")
            return []
