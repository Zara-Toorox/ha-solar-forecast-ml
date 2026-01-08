# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""Solar data reader for SFML Stats."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any

import aiofiles

from ..const import (
    SOLAR_FORECAST_ML_STATS,
    SOLAR_FORECAST_ML_AI,
    SOLAR_DAILY_SUMMARIES,
    SOLAR_HOURLY_PREDICTIONS,
    SOLAR_LEARNED_WEIGHTS,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class DailySummary:
    """Daily solar production summary. @zara"""

    date: date
    day_of_week: int
    month: int
    season: str

    predicted_total_kwh: float
    actual_total_kwh: float
    accuracy_percent: float
    error_kwh: float
    production_hours: int
    peak_hour: int
    peak_kwh: float

    morning_accuracy: float | None = None
    midday_accuracy: float | None = None
    afternoon_accuracy: float | None = None

    ml_mae: float | None = None
    ml_rmse: float | None = None
    ml_r2_score: float | None = None

    shadow_hours_count: int = 0
    shadow_loss_kwh: float = 0.0
    frost_hours_count: int = 0

    raw_data: dict = field(default_factory=dict)


@dataclass
class HourlyPrediction:
    """Hourly prediction data. @zara"""

    target_datetime: datetime
    target_hour: int
    target_date: date

    prediction_kwh: float
    actual_kwh: float | None
    accuracy_percent: float | None
    error_kwh: float | None

    prediction_method: str
    ml_contribution_percent: float
    confidence: float

    temperature: float | None = None
    solar_radiation: float | None = None
    clouds: float | None = None

    sun_elevation: float | None = None
    theoretical_max_kwh: float | None = None


@dataclass
class ModelState:
    """ML model state. @zara"""

    model_loaded: bool
    algorithm_used: str
    training_samples: int
    current_accuracy: float
    last_training: datetime | None
    peak_power_kw: float

    feature_weights: dict[str, float] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)


class SolarDataReader:
    """Reads and parses data from Solar Forecast ML. @zara"""

    def __init__(self, config_path: Path) -> None:
        """Initialize the solar data reader. @zara"""
        self._config_path = config_path
        self._stats_path = config_path / SOLAR_FORECAST_ML_STATS
        self._ai_path = config_path / SOLAR_FORECAST_ML_AI

    @property
    def is_available(self) -> bool:
        """Check if Solar Forecast ML data is available. @zara"""
        return self._stats_path.exists() and self._ai_path.exists()

    async def _read_json_file(self, file_path: Path) -> dict | None:
        """Read a JSON file asynchronously. @zara"""
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

    async def async_get_daily_summaries(
        self,
        days: int | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[DailySummary]:
        """Read the daily summaries. @zara"""
        file_path = self._stats_path / SOLAR_DAILY_SUMMARIES
        data = await self._read_json_file(file_path)

        if not data or "summaries" not in data:
            return []

        summaries: list[DailySummary] = []

        for raw in data["summaries"]:
            try:
                summary_date = date.fromisoformat(raw["date"])

                if start_date and summary_date < start_date:
                    continue
                if end_date and summary_date > end_date:
                    continue

                overall = raw.get("overall", {})
                time_windows = raw.get("time_windows", {})
                ml_metrics = raw.get("ml_metrics", {}).get("model_performance", {})
                shadow = raw.get("shadow_analysis", {})
                frost = raw.get("frost_analysis", {})

                summary = DailySummary(
                    date=summary_date,
                    day_of_week=raw.get("day_of_week", 0),
                    month=raw.get("month", 1),
                    season=raw.get("season", "unknown"),
                    predicted_total_kwh=overall.get("predicted_total_kwh", 0.0),
                    actual_total_kwh=overall.get("actual_total_kwh", 0.0),
                    accuracy_percent=overall.get("accuracy_percent", 0.0),
                    error_kwh=overall.get("error_kwh", 0.0),
                    production_hours=overall.get("production_hours", 0),
                    peak_hour=overall.get("peak_hour", 12),
                    peak_kwh=overall.get("peak_kwh", 0.0),
                    morning_accuracy=time_windows.get("morning_7_10", {}).get("accuracy"),
                    midday_accuracy=time_windows.get("midday_11_14", {}).get("accuracy"),
                    afternoon_accuracy=time_windows.get("afternoon_15_17", {}).get("accuracy"),
                    ml_mae=ml_metrics.get("mae"),
                    ml_rmse=ml_metrics.get("rmse"),
                    ml_r2_score=ml_metrics.get("r2_score"),
                    shadow_hours_count=shadow.get("shadow_hours_count", 0),
                    shadow_loss_kwh=shadow.get("cumulative_loss_kwh", 0.0),
                    frost_hours_count=frost.get("total_affected_hours", 0),
                    raw_data=raw,
                )
                summaries.append(summary)

            except Exception as err:
                _LOGGER.warning("Error parsing summary: %s", err)
                continue

        summaries.sort(key=lambda x: x.date, reverse=True)

        if days is not None:
            summaries = summaries[:days]

        return summaries

    async def async_get_hourly_predictions(
        self,
        target_date: date | None = None,
        include_no_production: bool = False,
    ) -> list[HourlyPrediction]:
        """Read the hourly predictions. @zara"""
        file_path = self._stats_path / SOLAR_HOURLY_PREDICTIONS
        data = await self._read_json_file(file_path)

        if not data or "predictions" not in data:
            return []

        predictions: list[HourlyPrediction] = []

        for raw in data["predictions"]:
            try:
                pred_date = date.fromisoformat(raw["target_date"])

                if target_date and pred_date != target_date:
                    continue

                prediction_kwh = raw.get("prediction_kwh", 0.0)
                actual_kwh = raw.get("actual_kwh")

                if not include_no_production:
                    if prediction_kwh == 0.0 and (actual_kwh is None or actual_kwh == 0.0):
                        continue

                weather = raw.get("weather_forecast", {})
                astronomy = raw.get("astronomy", {})

                prediction = HourlyPrediction(
                    target_datetime=datetime.fromisoformat(raw["target_datetime"]),
                    target_hour=raw.get("target_hour", 0),
                    target_date=pred_date,
                    prediction_kwh=prediction_kwh,
                    actual_kwh=actual_kwh,
                    accuracy_percent=raw.get("accuracy_percent"),
                    error_kwh=raw.get("error_kwh"),
                    prediction_method=raw.get("prediction_method", "unknown"),
                    ml_contribution_percent=raw.get("ml_contribution_percent", 0.0),
                    confidence=raw.get("confidence", 0.0),
                    temperature=weather.get("temperature"),
                    solar_radiation=weather.get("solar_radiation_wm2"),
                    clouds=weather.get("clouds"),
                    sun_elevation=astronomy.get("sun_elevation_deg"),
                    theoretical_max_kwh=astronomy.get("theoretical_max_kwh"),
                )
                predictions.append(prediction)

            except Exception as err:
                _LOGGER.warning("Error parsing prediction: %s", err)
                continue

        return predictions

    async def async_get_model_state(self) -> ModelState | None:
        """Read the current ML model state from ai/learned_weights.json. @zara"""
        weights_path = self._ai_path / SOLAR_LEARNED_WEIGHTS
        weights_data = await self._read_json_file(weights_path)

        if not weights_data:
            return None

        try:
            last_training = None
            if weights_data.get("last_trained"):
                try:
                    last_training = datetime.fromisoformat(
                        weights_data["last_trained"].replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            # Check if model has trained weights
            model_loaded = bool(weights_data.get("Wf") and len(weights_data.get("Wf", [])) > 0)

            return ModelState(
                model_loaded=model_loaded,
                algorithm_used="TinyLSTM",
                training_samples=weights_data.get("trained_samples", 0),
                current_accuracy=0.0,  # Not stored in learned_weights.json
                last_training=last_training,
                peak_power_kw=0.0,  # Not stored in learned_weights.json
                feature_weights={},
            )

        except Exception as err:
            _LOGGER.error("Error parsing model state: %s", err)
            return None

    async def async_get_weekly_stats(self, year: int, week: int) -> dict[str, Any]:
        """Calculate statistics for a specific calendar week. @zara"""
        all_summaries = await self.async_get_daily_summaries()

        week_summaries = [
            s for s in all_summaries
            if s.date.isocalendar()[0] == year and s.date.isocalendar()[1] == week
        ]

        if not week_summaries:
            return {"week": week, "year": year, "data_available": False}

        total_predicted = sum(s.predicted_total_kwh for s in week_summaries)
        total_actual = sum(s.actual_total_kwh for s in week_summaries)
        avg_accuracy = (
            sum(s.accuracy_percent for s in week_summaries) / len(week_summaries)
            if week_summaries else 0.0
        )

        hourly_preds = await self.async_get_hourly_predictions()
        week_predictions = [
            p for p in hourly_preds
            if p.target_date.isocalendar()[0] == year
            and p.target_date.isocalendar()[1] == week
        ]

        avg_ml_contribution = (
            sum(p.ml_contribution_percent for p in week_predictions) / len(week_predictions)
            if week_predictions else 0.0
        )

        return {
            "week": week,
            "year": year,
            "data_available": True,
            "days_count": len(week_summaries),
            "total_predicted_kwh": round(total_predicted, 2),
            "total_actual_kwh": round(total_actual, 2),
            "average_accuracy_percent": round(avg_accuracy, 1),
            "avg_ml_contribution_percent": round(avg_ml_contribution, 1),
            "total_shadow_hours": sum(s.shadow_hours_count for s in week_summaries),
            "total_shadow_loss_kwh": round(
                sum(s.shadow_loss_kwh for s in week_summaries), 2
            ),
            "total_frost_hours": sum(s.frost_hours_count for s in week_summaries),
            "daily_summaries": week_summaries,
        }

    async def async_get_monthly_stats(self, year: int, month: int) -> dict[str, Any]:
        """Calculate statistics for a specific month. @zara"""
        all_summaries = await self.async_get_daily_summaries()

        month_summaries = [
            s for s in all_summaries
            if s.date.year == year and s.date.month == month
        ]

        if not month_summaries:
            return {"month": month, "year": year, "data_available": False}

        total_predicted = sum(s.predicted_total_kwh for s in month_summaries)
        total_actual = sum(s.actual_total_kwh for s in month_summaries)

        best_day = max(month_summaries, key=lambda x: x.actual_total_kwh)
        worst_day = min(month_summaries, key=lambda x: x.actual_total_kwh)

        return {
            "month": month,
            "year": year,
            "data_available": True,
            "days_count": len(month_summaries),
            "total_predicted_kwh": round(total_predicted, 2),
            "total_actual_kwh": round(total_actual, 2),
            "average_daily_kwh": round(total_actual / len(month_summaries), 2),
            "average_accuracy_percent": round(
                sum(s.accuracy_percent for s in month_summaries) / len(month_summaries), 1
            ),
            "best_day": {
                "date": best_day.date.isoformat(),
                "kwh": best_day.actual_total_kwh,
            },
            "worst_day": {
                "date": worst_day.date.isoformat(),
                "kwh": worst_day.actual_total_kwh,
            },
            "total_shadow_loss_kwh": round(
                sum(s.shadow_loss_kwh for s in month_summaries), 2
            ),
            "total_frost_hours": sum(s.frost_hours_count for s in month_summaries),
            "daily_summaries": month_summaries,
        }
