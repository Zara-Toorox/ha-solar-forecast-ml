# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Forecast comparison data reader for SFML Stats (DB-only). @zara"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite

from ..const import FORECAST_COMPARISON_CHART_DAYS

_LOGGER = logging.getLogger(__name__)


def _compute_accuracy(actual: float | None, forecast: float | None) -> float | None:
    """Compute accuracy percent from actual and forecast values. @zara"""
    if actual is None or forecast is None:
        return None
    if actual <= 0 and forecast <= 0:
        return 100.0
    if actual <= 0 or forecast <= 0:
        return 0.0
    return round(max(0, min(100, 100 - abs((actual - forecast) / actual) * 100)), 1)


@dataclass
class ExternalForecast:
    """External forecast data for a day. @zara"""

    name: str
    forecast_kwh: float | None
    error_kwh: float | None = None
    accuracy_percent: float | None = None


@dataclass
class ForecastComparisonDay:
    """Forecast comparison data for a single day. @zara"""

    date: date
    actual_kwh: float | None
    sfml_forecast_kwh: float | None
    sfml_error_kwh: float | None = None
    sfml_accuracy_percent: float | None = None
    external_1: ExternalForecast | None = None
    external_2: ExternalForecast | None = None

    @property
    def has_data(self) -> bool:
        """Check if day has any useful data. @zara"""
        return self.actual_kwh is not None or self.sfml_forecast_kwh is not None


@dataclass
class ForecastComparisonStats:
    """Aggregated statistics for forecast comparison. @zara"""

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
    """Reads forecast comparison data from stats_forecast_comparison DB table. @zara"""

    def __init__(
        self,
        db_path: Path,
        external_1_name: str = "Externe Prognose 1",
        external_2_name: str = "Externe Prognose 2",
    ) -> None:
        """Initialize the forecast comparison reader. @zara"""
        self._db_path = db_path
        self._ext1_name = external_1_name
        self._ext2_name = external_2_name

    @asynccontextmanager
    async def _get_db(self) -> AsyncIterator[aiosqlite.Connection]:
        """Get DB connection via manager with direct fallback. @zara"""
        from ..storage.db_connection_manager import get_manager
        manager = get_manager()
        if manager is not None and manager.is_connected:
            yield await manager.get_connection()
            return
        async with aiosqlite.connect(str(self._db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            yield conn

    @property
    def is_available(self) -> bool:
        """Check if forecast comparison data is available. @zara"""
        return self._db_path.exists()

    async def async_get_comparison_days(
        self,
        days: int = FORECAST_COMPARISON_CHART_DAYS,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[ForecastComparisonDay]:
        """Get forecast comparison data for a date range from DB. @zara"""
        if not self._db_path.exists():
            return []

        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=days - 1)

        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        # Read from stats_forecast_comparison + enrich actual from daily_summaries
        db_rows: dict[str, dict[str, Any]] = {}

        try:
            async with self._get_db() as conn:
                # Main forecast comparison data
                async with conn.execute(
                    """SELECT date, actual_kwh, sfml_forecast_kwh,
                              sfml_accuracy_percent,
                              external_1_kwh, external_1_accuracy_percent,
                              external_2_kwh, external_2_accuracy_percent,
                              best_source
                       FROM stats_forecast_comparison
                       WHERE date >= ? AND date <= ?
                       ORDER BY date""",
                    (start_str, end_str),
                ) as cursor:
                    for row in await cursor.fetchall():
                        db_rows[row["date"]] = dict(row)

                # Enrich with daily_summaries for actual + sfml forecast
                async with conn.execute(
                    """SELECT date, actual_total_kwh, predicted_total_kwh
                       FROM daily_summaries
                       WHERE date >= ? AND date <= ?
                       ORDER BY date""",
                    (start_str, end_str),
                ) as cursor:
                    for row in await cursor.fetchall():
                        d = row["date"]
                        if d not in db_rows:
                            # Day exists in daily_summaries but not in forecast_comparison
                            db_rows[d] = {
                                "date": d,
                                "actual_kwh": row["actual_total_kwh"],
                                "sfml_forecast_kwh": row["predicted_total_kwh"],
                                "sfml_accuracy_percent": None,
                                "external_1_kwh": None,
                                "external_1_accuracy_percent": None,
                                "external_2_kwh": None,
                                "external_2_accuracy_percent": None,
                                "best_source": None,
                            }
                        else:
                            # Fill missing actual_kwh from daily_summaries
                            if db_rows[d].get("actual_kwh") is None and row["actual_total_kwh"] is not None:
                                db_rows[d]["actual_kwh"] = row["actual_total_kwh"]
                            # Fill missing sfml_forecast from daily_summaries
                            if db_rows[d].get("sfml_forecast_kwh") is None and row["predicted_total_kwh"] is not None:
                                db_rows[d]["sfml_forecast_kwh"] = row["predicted_total_kwh"]

        except Exception as err:
            _LOGGER.error("Error reading forecast comparison from DB: %s", err)
            return []

        # Build result for each day in range
        result: list[ForecastComparisonDay] = []
        current = start_date
        while current <= end_date:
            day_str = current.isoformat()

            if day_str in db_rows:
                raw = db_rows[day_str]
                comparison_day = self._build_day(current, raw)
            else:
                comparison_day = ForecastComparisonDay(
                    date=current,
                    actual_kwh=None,
                    sfml_forecast_kwh=None,
                )

            result.append(comparison_day)
            current += timedelta(days=1)

        return result

    def _build_day(self, day_date: date, raw: dict[str, Any]) -> ForecastComparisonDay:
        """Build a ForecastComparisonDay from DB row data. @zara"""
        actual = raw.get("actual_kwh")
        sfml_forecast = raw.get("sfml_forecast_kwh")

        # Compute SFML accuracy if missing/zero
        sfml_acc = raw.get("sfml_accuracy_percent")
        if (not sfml_acc or sfml_acc <= 0) and actual is not None and sfml_forecast is not None:
            sfml_acc = _compute_accuracy(actual, sfml_forecast)

        sfml_error = None
        if actual is not None and sfml_forecast is not None:
            sfml_error = round(sfml_forecast - actual, 3)

        # External 1
        external_1 = None
        ext1_kwh = raw.get("external_1_kwh")
        if ext1_kwh is not None:
            ext1_acc = raw.get("external_1_accuracy_percent")
            if (not ext1_acc or ext1_acc <= 0) and actual is not None:
                ext1_acc = _compute_accuracy(actual, ext1_kwh)
            ext1_error = None
            if actual is not None:
                ext1_error = round(ext1_kwh - actual, 3)
            external_1 = ExternalForecast(
                name=self._ext1_name,
                forecast_kwh=ext1_kwh,
                error_kwh=ext1_error,
                accuracy_percent=ext1_acc,
            )

        # External 2
        external_2 = None
        ext2_kwh = raw.get("external_2_kwh")
        if ext2_kwh is not None:
            ext2_acc = raw.get("external_2_accuracy_percent")
            if (not ext2_acc or ext2_acc <= 0) and actual is not None:
                ext2_acc = _compute_accuracy(actual, ext2_kwh)
            ext2_error = None
            if actual is not None:
                ext2_error = round(ext2_kwh - actual, 3)
            external_2 = ExternalForecast(
                name=self._ext2_name,
                forecast_kwh=ext2_kwh,
                error_kwh=ext2_error,
                accuracy_percent=ext2_acc,
            )

        return ForecastComparisonDay(
            date=day_date,
            actual_kwh=actual,
            sfml_forecast_kwh=sfml_forecast,
            sfml_error_kwh=sfml_error,
            sfml_accuracy_percent=sfml_acc,
            external_1=external_1,
            external_2=external_2,
        )

    async def async_get_statistics(
        self,
        days: int = FORECAST_COMPARISON_CHART_DAYS,
    ) -> ForecastComparisonStats:
        """Calculate aggregated statistics for forecast comparison. @zara"""
        comparison_days = await self.async_get_comparison_days(days=days)

        days_with_actual = sum(1 for d in comparison_days if d.actual_kwh is not None and d.actual_kwh > 0)

        stats = ForecastComparisonStats(
            days_count=len(comparison_days),
            days_with_actual=days_with_actual,
            external_1_name=self._ext1_name,
            external_2_name=self._ext2_name,
        )

        if days_with_actual == 0:
            return stats

        # SFML accuracies
        sfml_accuracies = [
            d.sfml_accuracy_percent for d in comparison_days
            if d.sfml_accuracy_percent is not None and d.sfml_accuracy_percent > 0
        ]
        sfml_errors = [
            d.sfml_error_kwh for d in comparison_days
            if d.sfml_error_kwh is not None
        ]

        if sfml_accuracies:
            stats.sfml_avg_accuracy = round(sum(sfml_accuracies) / len(sfml_accuracies), 1)
        if sfml_errors:
            stats.sfml_total_error_kwh = round(sum(sfml_errors), 2)

        # External 1 accuracies
        ext1_accuracies = []
        ext1_errors = []
        has_ext1 = False
        for d in comparison_days:
            if d.external_1:
                has_ext1 = True
                if d.external_1.accuracy_percent is not None and d.external_1.accuracy_percent > 0:
                    ext1_accuracies.append(d.external_1.accuracy_percent)
                if d.external_1.error_kwh is not None:
                    ext1_errors.append(d.external_1.error_kwh)

        if ext1_accuracies:
            stats.external_1_avg_accuracy = round(sum(ext1_accuracies) / len(ext1_accuracies), 1)
        if ext1_errors:
            stats.external_1_total_error_kwh = round(sum(ext1_errors), 2)
        if not has_ext1:
            stats.external_1_name = None

        # External 2 accuracies
        ext2_accuracies = []
        ext2_errors = []
        has_ext2 = False
        for d in comparison_days:
            if d.external_2:
                has_ext2 = True
                if d.external_2.accuracy_percent is not None and d.external_2.accuracy_percent > 0:
                    ext2_accuracies.append(d.external_2.accuracy_percent)
                if d.external_2.error_kwh is not None:
                    ext2_errors.append(d.external_2.error_kwh)

        if ext2_accuracies:
            stats.external_2_avg_accuracy = round(sum(ext2_accuracies) / len(ext2_accuracies), 1)
        if ext2_errors:
            stats.external_2_total_error_kwh = round(sum(ext2_errors), 2)
        if not has_ext2:
            stats.external_2_name = None

        # Determine best forecast
        accuracies = {}
        if stats.sfml_avg_accuracy is not None:
            accuracies["SFML"] = stats.sfml_avg_accuracy
        if stats.external_1_avg_accuracy is not None:
            accuracies[stats.external_1_name or "Extern 1"] = stats.external_1_avg_accuracy
        if stats.external_2_avg_accuracy is not None:
            accuracies[stats.external_2_name or "Extern 2"] = stats.external_2_avg_accuracy

        if accuracies:
            stats.best_forecast = max(accuracies, key=accuracies.get)

        return stats

    async def async_get_chart_data(
        self,
        days: int = FORECAST_COMPARISON_CHART_DAYS,
    ) -> dict[str, Any]:
        """Get data formatted for chart rendering. @zara"""
        comparison_days = await self.async_get_comparison_days(days=days)
        stats = await self.async_get_statistics(days=days)

        dates = [d.date.strftime("%d.%m") for d in comparison_days]
        actual = [d.actual_kwh for d in comparison_days]
        sfml = [d.sfml_forecast_kwh for d in comparison_days]

        # External forecasts
        external_1 = None
        external_1_name = None
        external_2 = None
        external_2_name = None

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
