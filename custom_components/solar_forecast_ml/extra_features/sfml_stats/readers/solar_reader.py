# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Solar data reader for SFML Stats. @zara"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING, AsyncIterator

import aiosqlite

from ..const import SOLAR_FORECAST_DB

if TYPE_CHECKING:
    from ..storage.db_connection_manager import DatabaseConnectionManager

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


@dataclass
class PanelGroupData:
    """Panel group prediction and actual data. @zara"""

    group_name: str
    prediction_kwh: float
    actual_kwh: float | None = None
    target_hour: int | None = None


@dataclass
class DailyForecast:
    """Daily forecast data from SFML database, read-only. @zara"""

    forecast_type: str
    forecast_date: date
    prediction_kwh: float
    prediction_kwh_raw: float | None = None
    locked: bool = False
    source: str | None = None
    created_at: datetime | None = None


class SolarDataReader:
    """Reads and parses data from Solar Forecast ML SQLite database. @zara"""

    _db_manager: DatabaseConnectionManager | None = None

    def __init__(self, config_path: Path, db_manager: DatabaseConnectionManager | None = None) -> None:
        """Initialize the solar data reader. @zara"""
        self._config_path = config_path
        self._db_path = config_path / SOLAR_FORECAST_DB
        if db_manager is not None:
            SolarDataReader._db_manager = db_manager

    @property
    def is_available(self) -> bool:
        """Check if Solar Forecast ML database is available. @zara"""
        if self._db_manager is not None:
            return self._db_manager.is_available
        return self._db_path.exists()

    @asynccontextmanager
    async def _get_db_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Get a database connection from the manager. @zara"""
        from ..storage.db_connection_manager import get_manager

        manager = get_manager()
        if manager is not None and manager.is_connected:
            _LOGGER.debug("SolarDataReader: Using database connection manager")
            yield await manager.get_connection()
        else:
            _LOGGER.warning("SolarDataReader: Database manager not available, using direct connection (THIS CAUSES THREADING ERRORS)")
            conn = await aiosqlite.connect(str(self._db_path))
            conn.row_factory = aiosqlite.Row
            try:
                yield conn
            finally:
                await conn.close()

    async def async_get_daily_summaries(
        self,
        days: int | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[DailySummary]:
        """Read daily summaries from database. @zara"""
        if not self.is_available:
            _LOGGER.debug("Database not found: %s", self._db_path)
            return []

        try:
            async with self._get_db_connection() as conn:
                query = """
                    SELECT
                        ds.date, ds.day_of_week, ds.month, ds.season,
                        ds.predicted_total_kwh, ds.actual_total_kwh,
                        ds.accuracy_percent, ds.error_kwh, ds.production_hours,
                        ds.peak_hour, ds.peak_kwh, ds.peak_power_w,
                        ds.ml_mae, ds.ml_rmse, ds.ml_r2_score
                    FROM daily_summaries ds
                """
                params = []
                conditions = []

                if start_date:
                    conditions.append("ds.date >= ?")
                    params.append(start_date.isoformat())
                if end_date:
                    conditions.append("ds.date <= ?")
                    params.append(end_date.isoformat())

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY ds.date DESC"

                if days is not None:
                    query += f" LIMIT {days}"

                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                summaries: list[DailySummary] = []

                for row in rows:
                    try:
                        summary_date = date.fromisoformat(row["date"]) if isinstance(row["date"], str) else row["date"]

                        time_windows = await self._get_time_windows(conn, row["date"])

                        frost_data = await self._get_frost_analysis(conn, row["date"])

                        summary = DailySummary(
                            date=summary_date,
                            day_of_week=row["day_of_week"] or 0,
                            month=row["month"] or 1,
                            season=row["season"] or "unknown",
                            predicted_total_kwh=row["predicted_total_kwh"] or 0.0,
                            actual_total_kwh=row["actual_total_kwh"] or 0.0,
                            accuracy_percent=row["accuracy_percent"] or 0.0,
                            error_kwh=row["error_kwh"] or 0.0,
                            production_hours=row["production_hours"] or 0,
                            peak_hour=row["peak_hour"] or 12,
                            peak_kwh=row["peak_kwh"] or 0.0,
                            morning_accuracy=time_windows.get("morning_7_10", {}).get("accuracy"),
                            midday_accuracy=time_windows.get("midday_11_14", {}).get("accuracy"),
                            afternoon_accuracy=time_windows.get("afternoon_15_17", {}).get("accuracy"),
                            ml_mae=row["ml_mae"],
                            ml_rmse=row["ml_rmse"],
                            ml_r2_score=row["ml_r2_score"],
                            frost_hours_count=frost_data.get("total_affected_hours", 0) if frost_data else 0,
                            raw_data=dict(row),
                        )
                        summaries.append(summary)

                    except Exception as err:
                        _LOGGER.warning("Error parsing summary row: %s", err)
                        continue

                return summaries

        except Exception as err:
            _LOGGER.error("Error reading daily summaries from database: %s", err)
            return []

    async def _get_time_windows(self, conn: aiosqlite.Connection, date_str: str) -> dict[str, dict]:
        """Get time windows for a specific date. @zara"""
        try:
            async with conn.execute(
                """SELECT window_name, predicted_kwh, actual_kwh, accuracy, stable, hours_count
                   FROM daily_summary_time_windows WHERE date = ?""",
                (date_str,)
            ) as cursor:
                rows = await cursor.fetchall()

            return {
                row["window_name"]: {
                    "predicted_kwh": row["predicted_kwh"],
                    "actual_kwh": row["actual_kwh"],
                    "accuracy": row["accuracy"],
                    "stable": bool(row["stable"]),
                    "hours_count": row["hours_count"],
                }
                for row in rows
            }
        except Exception:
            return {}

    async def _get_frost_analysis(self, conn: aiosqlite.Connection, date_str: str) -> dict | None:
        """Get frost analysis for a specific date. @zara"""
        try:
            async with conn.execute(
                """SELECT hours_analyzed, frost_detected, total_affected_hours,
                          heavy_frost_hours, light_frost_hours
                   FROM daily_summary_frost_analysis WHERE date = ?""",
                (date_str,)
            ) as cursor:
                row = await cursor.fetchone()

            if not row:
                return None

            return {
                "hours_analyzed": row["hours_analyzed"],
                "frost_detected": bool(row["frost_detected"]),
                "total_affected_hours": row["total_affected_hours"],
                "heavy_frost_hours": row["heavy_frost_hours"],
                "light_frost_hours": row["light_frost_hours"],
            }
        except Exception:
            return None

    async def async_get_hourly_predictions(
        self,
        target_date: date | None = None,
        include_no_production: bool = False,
    ) -> list[HourlyPrediction]:
        """Read hourly predictions from database. @zara"""
        if not self.is_available:
            _LOGGER.debug("Database not found: %s", self._db_path)
            return []

        try:
            async with self._get_db_connection() as conn:
                query = """
                    SELECT
                        hp.target_datetime, hp.target_hour, hp.target_date,
                        hp.prediction_kwh, hp.actual_kwh, hp.accuracy_percent,
                        hp.error_kwh, hp.prediction_method, hp.ml_contribution_percent,
                        hp.confidence,
                        pw.temperature, pw.solar_radiation_wm2, pw.clouds,
                        pa.sun_elevation_deg, pa.theoretical_max_kwh
                    FROM hourly_predictions hp
                    LEFT JOIN prediction_weather pw ON hp.prediction_id = pw.prediction_id
                        AND pw.weather_type = 'forecast'
                    LEFT JOIN prediction_astronomy pa ON hp.prediction_id = pa.prediction_id
                """
                params = []
                conditions = []

                if target_date:
                    conditions.append("hp.target_date = ?")
                    params.append(target_date.isoformat())

                if not include_no_production:
                    conditions.append("(hp.prediction_kwh > 0 OR (hp.actual_kwh IS NOT NULL AND hp.actual_kwh > 0))")

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY hp.target_datetime"

                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                predictions: list[HourlyPrediction] = []

                for row in rows:
                    try:
                        pred_date = date.fromisoformat(row["target_date"]) if isinstance(row["target_date"], str) else row["target_date"]
                        target_dt = datetime.fromisoformat(row["target_datetime"]) if isinstance(row["target_datetime"], str) else row["target_datetime"]

                        prediction = HourlyPrediction(
                            target_datetime=target_dt,
                            target_hour=row["target_hour"] or 0,
                            target_date=pred_date,
                            prediction_kwh=row["prediction_kwh"] or 0.0,
                            actual_kwh=row["actual_kwh"],
                            accuracy_percent=row["accuracy_percent"],
                            error_kwh=row["error_kwh"],
                            prediction_method=row["prediction_method"] or "unknown",
                            ml_contribution_percent=row["ml_contribution_percent"] or 0.0,
                            confidence=row["confidence"] or 0.0,
                            temperature=row["temperature"],
                            solar_radiation=row["solar_radiation_wm2"],
                            clouds=row["clouds"],
                            sun_elevation=row["sun_elevation_deg"],
                            theoretical_max_kwh=row["theoretical_max_kwh"],
                        )
                        predictions.append(prediction)

                    except Exception as err:
                        _LOGGER.warning("Error parsing prediction row: %s", err)
                        continue

                return predictions

        except Exception as err:
            _LOGGER.error("Error reading hourly predictions from database: %s", err)
            return []

    async def async_get_model_state(self) -> ModelState | None:
        """Read the current ML model state from database. @zara"""
        if not self.is_available:
            return None

        try:
            async with self._get_db_connection() as conn:
                async with conn.execute(
                    """SELECT version, active_model, training_samples, last_trained,
                              accuracy, rmse
                       FROM ai_learned_weights_meta WHERE id = 1"""
                ) as cursor:
                    meta_row = await cursor.fetchone()

                if not meta_row:
                    return None

                last_training = None
                if meta_row["last_trained"]:
                    try:
                        last_training = datetime.fromisoformat(
                            str(meta_row["last_trained"]).replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

                async with conn.execute(
                    "SELECT COUNT(*) FROM ai_lstm_weights"
                ) as cursor:
                    lstm_count = (await cursor.fetchone())[0]

                model_loaded = lstm_count > 0

                return ModelState(
                    model_loaded=model_loaded,
                    algorithm_used=meta_row["active_model"] or "TinyLSTM",
                    training_samples=meta_row["training_samples"] or 0,
                    current_accuracy=meta_row["accuracy"] or 0.0,
                    last_training=last_training,
                    peak_power_kw=0.0,
                    feature_weights={},
                )

        except Exception as err:
            _LOGGER.error("Error reading model state from database: %s", err)
            return None

    async def async_get_weekly_stats(self, year: int, week: int) -> dict[str, Any]:
        """Calculate statistics for a specific calendar week. @zara"""
        start_date = date.fromisocalendar(year, week, 1)
        end_date = date.fromisocalendar(year, week, 7)
        week_summaries = await self.async_get_daily_summaries(
            start_date=start_date, end_date=end_date
        )

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
        import calendar as cal_mod
        start_date = date(year, month, 1)
        last_day = cal_mod.monthrange(year, month)[1]
        end_date = date(year, month, last_day)
        month_summaries = await self.async_get_daily_summaries(
            start_date=start_date, end_date=end_date
        )

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

    async def async_get_panel_group_data(
        self,
        target_date: date,
    ) -> dict[str, list[PanelGroupData]]:
        """Get panel group predictions and actuals for a specific date. @zara"""
        if not self.is_available:
            _LOGGER.debug("Database not found: %s", self._db_path)
            return {}

        try:
            async with self._get_db_connection() as conn:
                async with conn.execute(
                    """SELECT ppg.group_name, ppg.prediction_kwh, ppg.actual_kwh,
                              hp.target_hour
                       FROM prediction_panel_groups ppg
                       JOIN hourly_predictions hp ON hp.prediction_id = ppg.prediction_id
                       WHERE hp.target_date = ?
                       ORDER BY hp.target_hour, ppg.group_name""",
                    (target_date.isoformat(),)
                ) as cursor:
                    rows = await cursor.fetchall()

                if not rows:
                    return {}

                result: dict[str, list[PanelGroupData]] = {}
                for row in rows:
                    group_name = row["group_name"]
                    if group_name not in result:
                        result[group_name] = []

                    result[group_name].append(PanelGroupData(
                        group_name=group_name,
                        prediction_kwh=row["prediction_kwh"] or 0.0,
                        actual_kwh=row["actual_kwh"],
                        target_hour=row["target_hour"],
                    ))

                return result

        except Exception as err:
            _LOGGER.error("Error reading panel group data from database: %s", err)
            return {}

    async def async_get_daily_forecasts(
        self,
        forecast_types: list[str] | None = None,
    ) -> dict[str, DailyForecast]:
        """Read daily forecasts from SFML database. @zara"""
        if not self.is_available:
            _LOGGER.debug("Database not found: %s", self._db_path)
            return {}

        if forecast_types is None:
            forecast_types = ["today", "tomorrow", "day_after_tomorrow"]

        today = date.today()
        date_mapping = {
            "today": today,
            "tomorrow": today + timedelta(days=1),
            "day_after_tomorrow": today + timedelta(days=2),
        }

        result: dict[str, DailyForecast] = {}

        try:
            async with self._get_db_connection() as conn:
                for forecast_type in forecast_types:
                    expected_date = date_mapping.get(forecast_type)
                    if not expected_date:
                        continue

                    query = """
                        SELECT forecast_type, forecast_date, prediction_kwh,
                               prediction_kwh_raw, locked, source, created_at
                        FROM daily_forecasts
                        WHERE forecast_type = ? AND forecast_date = ?
                    """

                    async with conn.execute(
                        query, (forecast_type, expected_date.isoformat())
                    ) as cursor:
                        row = await cursor.fetchone()

                    if not row:
                        _LOGGER.debug(
                            "No daily_forecast found for type=%s, date=%s",
                            forecast_type, expected_date
                        )
                        continue

                    try:
                        forecast_date_val = row["forecast_date"]
                        if isinstance(forecast_date_val, str):
                            forecast_date_val = date.fromisoformat(forecast_date_val)

                        created_at_val = row["created_at"]
                        if isinstance(created_at_val, str):
                            created_at_val = datetime.fromisoformat(created_at_val)

                        forecast = DailyForecast(
                            forecast_type=row["forecast_type"],
                            forecast_date=forecast_date_val,
                            prediction_kwh=row["prediction_kwh"] or 0.0,
                            prediction_kwh_raw=row["prediction_kwh_raw"],
                            locked=bool(row["locked"]) if row["locked"] is not None else False,
                            source=row["source"],
                            created_at=created_at_val,
                        )
                        result[forecast_type] = forecast

                    except Exception as err:
                        _LOGGER.warning("Error parsing daily forecast row: %s", err)
                        continue

                _LOGGER.debug("Loaded %d daily forecasts from database", len(result))
                return result

        except Exception as err:
            _LOGGER.error("Error reading daily forecasts from database: %s", err)
            return {}
