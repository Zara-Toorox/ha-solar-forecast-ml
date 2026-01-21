"""Forecast comparison data reader for SFML Stats Lite.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import aiofiles

from ..const import (
    SFML_STATS_DATA,
    EXTERNAL_FORECASTS_HISTORY,
    FORECAST_COMPARISON_CHART_DAYS,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class ExternalForecast:
    """External forecast data for a day."""

    entity_id: str
    name: str
    forecast_kwh: float | None
    error_kwh: float | None = None
    accuracy_percent: float | None = None


@dataclass
class ForecastComparisonDay:
    """Forecast comparison data for a single day."""

    date: date
    actual_kwh: float | None
    sfml_forecast_kwh: float | None
    sfml_error_kwh: float | None = None
    sfml_accuracy_percent: float | None = None
    external_1: ExternalForecast | None = None
    external_2: ExternalForecast | None = None
    timestamp: datetime | None = None

    @property
    def has_data(self) -> bool:
        """Check if day has any useful data."""
        return self.actual_kwh is not None or self.sfml_forecast_kwh is not None


@dataclass
class ForecastComparisonStats:
    """Aggregated statistics for forecast comparison."""

    days_count: int
    days_with_actual: int

    sfml_avg_accuracy: float | None = None
    sfml_total_error_kwh: float = 0.0

    external_1_name: str | None = None
    external_1_avg_accuracy: float | None = None
    external_1_total_error_kwh: float = 0.0

    external_2_name: str | None = None
    external_2_avg_accuracy: float | None = None
    external_2_total_error_kwh: float = 0.0

    best_forecast: str | None = None


class ForecastComparisonReader:
    """Reads forecast comparison data for charts and analysis."""

    def __init__(self, config_path: Path) -> None:
        """Initialize the forecast comparison reader."""
        self._config_path = config_path
        self._data_path = config_path / SFML_STATS_DATA
        self._history_file = self._data_path / EXTERNAL_FORECASTS_HISTORY

    @property
    def is_available(self) -> bool:
        """Check if forecast comparison data is available."""
        return self._history_file.exists()

    async def _read_json_file(self, file_path: Path) -> dict | None:
        """Read a JSON file asynchronously."""
        if not file_path.exists():
            _LOGGER.debug("File not found: %s", file_path)
            return None

        try:
            async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        except json.JSONDecodeError as err:
            _LOGGER.error("JSON parsing error in %s: %s", file_path, err)
            return None
        except Exception as err:
            _LOGGER.error("Error reading %s: %s", file_path, err)
            return None

    async def async_get_comparison_days(
        self,
        days: int = FORECAST_COMPARISON_CHART_DAYS,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[ForecastComparisonDay]:
        """Get forecast comparison data for a date range."""
        data = await self._read_json_file(self._history_file)

        if not data or "days" not in data:
            return []

        # Determine date range
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=days - 1)

        result: list[ForecastComparisonDay] = []

        current = start_date
        while current <= end_date:
            day_str = current.isoformat()

            if day_str in data["days"]:
                raw = data["days"][day_str]
                comparison_day = self._parse_day(current, raw)
            else:
                # Create empty entry for missing day
                comparison_day = ForecastComparisonDay(
                    date=current,
                    actual_kwh=None,
                    sfml_forecast_kwh=None,
                )

            result.append(comparison_day)
            current = current + timedelta(days=1)

        return result

    def _parse_day(self, day_date: date, raw: dict[str, Any]) -> ForecastComparisonDay:
        """Parse a single day's data from JSON."""
        external_1 = None
        external_2 = None

        if "external_1" in raw:
            ext1 = raw["external_1"]
            external_1 = ExternalForecast(
                entity_id=ext1.get("entity_id", ""),
                name=ext1.get("name", "External 1"),
                forecast_kwh=ext1.get("forecast_kwh"),
                error_kwh=ext1.get("error_kwh"),
                accuracy_percent=ext1.get("accuracy_percent"),
            )

        if "external_2" in raw:
            ext2 = raw["external_2"]
            external_2 = ExternalForecast(
                entity_id=ext2.get("entity_id", ""),
                name=ext2.get("name", "External 2"),
                forecast_kwh=ext2.get("forecast_kwh"),
                error_kwh=ext2.get("error_kwh"),
                accuracy_percent=ext2.get("accuracy_percent"),
            )

        timestamp = None
        if raw.get("timestamp"):
            try:
                timestamp = datetime.fromisoformat(raw["timestamp"])
            except ValueError:
                pass

        return ForecastComparisonDay(
            date=day_date,
            actual_kwh=raw.get("actual_kwh"),
            sfml_forecast_kwh=raw.get("sfml_forecast_kwh"),
            sfml_error_kwh=raw.get("sfml_error_kwh"),
            sfml_accuracy_percent=raw.get("sfml_accuracy_percent"),
            external_1=external_1,
            external_2=external_2,
            timestamp=timestamp,
        )

    async def async_get_statistics(
        self,
        days: int = FORECAST_COMPARISON_CHART_DAYS,
    ) -> ForecastComparisonStats:
        """Calculate aggregated statistics for forecast comparison."""
        comparison_days = await self.async_get_comparison_days(days=days)

        days_with_actual = sum(1 for d in comparison_days if d.actual_kwh is not None and d.actual_kwh > 0)

        stats = ForecastComparisonStats(
            days_count=len(comparison_days),
            days_with_actual=days_with_actual,
        )

        if days_with_actual == 0:
            return stats

        # Calculate SFML statistics
        sfml_accuracies = [
            d.sfml_accuracy_percent for d in comparison_days
            if d.sfml_accuracy_percent is not None
        ]
        sfml_errors = [
            d.sfml_error_kwh for d in comparison_days
            if d.sfml_error_kwh is not None
        ]

        if sfml_accuracies:
            stats.sfml_avg_accuracy = round(sum(sfml_accuracies) / len(sfml_accuracies), 1)
        if sfml_errors:
            stats.sfml_total_error_kwh = round(sum(sfml_errors), 2)

        # Calculate External 1 statistics
        ext1_accuracies = []
        ext1_errors = []
        for d in comparison_days:
            if d.external_1:
                if d.external_1.accuracy_percent is not None:
                    ext1_accuracies.append(d.external_1.accuracy_percent)
                if d.external_1.error_kwh is not None:
                    ext1_errors.append(d.external_1.error_kwh)
                if d.external_1.name and not stats.external_1_name:
                    stats.external_1_name = d.external_1.name

        if ext1_accuracies:
            stats.external_1_avg_accuracy = round(sum(ext1_accuracies) / len(ext1_accuracies), 1)
        if ext1_errors:
            stats.external_1_total_error_kwh = round(sum(ext1_errors), 2)

        # Calculate External 2 statistics
        ext2_accuracies = []
        ext2_errors = []
        for d in comparison_days:
            if d.external_2:
                if d.external_2.accuracy_percent is not None:
                    ext2_accuracies.append(d.external_2.accuracy_percent)
                if d.external_2.error_kwh is not None:
                    ext2_errors.append(d.external_2.error_kwh)
                if d.external_2.name and not stats.external_2_name:
                    stats.external_2_name = d.external_2.name

        if ext2_accuracies:
            stats.external_2_avg_accuracy = round(sum(ext2_accuracies) / len(ext2_accuracies), 1)
        if ext2_errors:
            stats.external_2_total_error_kwh = round(sum(ext2_errors), 2)

        # Determine best forecast
        accuracies = {}
        if stats.sfml_avg_accuracy is not None:
            accuracies["SFML"] = stats.sfml_avg_accuracy
        if stats.external_1_avg_accuracy is not None:
            accuracies[stats.external_1_name or "External 1"] = stats.external_1_avg_accuracy
        if stats.external_2_avg_accuracy is not None:
            accuracies[stats.external_2_name or "External 2"] = stats.external_2_avg_accuracy

        if accuracies:
            stats.best_forecast = max(accuracies, key=accuracies.get)

        return stats

    async def async_get_chart_data(
        self,
        days: int = FORECAST_COMPARISON_CHART_DAYS,
    ) -> dict[str, Any]:
        """Get data formatted for chart rendering."""
        comparison_days = await self.async_get_comparison_days(days=days)
        stats = await self.async_get_statistics(days=days)

        # Prepare series data
        dates = [d.date.strftime("%d.%m") for d in comparison_days]
        actual = [d.actual_kwh for d in comparison_days]
        sfml = [d.sfml_forecast_kwh for d in comparison_days]

        external_1 = None
        external_1_name = None
        external_2 = None
        external_2_name = None

        # Check if we have external forecasts
        for d in comparison_days:
            if d.external_1 and d.external_1.forecast_kwh is not None:
                if external_1 is None:
                    external_1 = []
                    external_1_name = d.external_1.name
            if d.external_2 and d.external_2.forecast_kwh is not None:
                if external_2 is None:
                    external_2 = []
                    external_2_name = d.external_2.name

        if external_1 is not None:
            external_1 = [
                d.external_1.forecast_kwh if d.external_1 else None
                for d in comparison_days
            ]

        if external_2 is not None:
            external_2 = [
                d.external_2.forecast_kwh if d.external_2 else None
                for d in comparison_days
            ]

        return {
            "dates": dates,
            "actual": actual,
            "sfml": sfml,
            "external_1": external_1,
            "external_1_name": external_1_name or stats.external_1_name,
            "external_2": external_2,
            "external_2_name": external_2_name or stats.external_2_name,
            "stats": {
                "days_count": stats.days_count,
                "days_with_actual": stats.days_with_actual,
                "sfml_avg_accuracy": stats.sfml_avg_accuracy,
                "external_1_avg_accuracy": stats.external_1_avg_accuracy,
                "external_2_avg_accuracy": stats.external_2_avg_accuracy,
                "best_forecast": stats.best_forecast,
            },
        }
