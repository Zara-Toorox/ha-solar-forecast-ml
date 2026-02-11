# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""SFML Data Reader for SFML Stats. @zara"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

import aiosqlite

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

SFML_DOMAIN = "solar_forecast_ml"


class SFMLDataReader:
    """Reads solar data from SFML database. @zara"""

    def __init__(self, hass: HomeAssistant | None) -> None:
        """Initialize the SFML Data Reader. @zara"""
        self.hass = hass
        if hass is not None:
            self._db_path = Path(hass.config.path()) / "solar_forecast_ml" / "solar_forecast.db"
        else:
            self._db_path = Path("/config/solar_forecast_ml/solar_forecast.db")
            _LOGGER.warning("SFMLDataReader initialized with None hass - using default path")
        self._sfml_config: dict[str, Any] | None = None

    @asynccontextmanager
    async def _get_db(self) -> AsyncIterator[aiosqlite.Connection]:
        """Get DB connection via manager with direct fallback. @zara"""
        from .storage.db_connection_manager import get_manager
        manager = get_manager()
        if manager is not None and manager.is_connected:
            yield await manager.get_connection()
            return
        async with aiosqlite.connect(str(self._db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            yield conn

    @property
    def is_available(self) -> bool:
        """Check if SFML database is available. @zara"""
        return self._db_path.exists()

    def _get_sfml_config(self) -> dict[str, Any] | None:
        """Get SFML config entry data. @zara"""
        if self._sfml_config is not None:
            return self._sfml_config

        if self.hass is None:
            return None

        for entry in self.hass.config_entries.async_entries(SFML_DOMAIN):
            self._sfml_config = {**entry.data, **entry.options}
            return self._sfml_config
        return None

    def get_power_entity_id(self) -> str | None:
        """Get power sensor entity ID from SFML config. @zara"""
        config = self._get_sfml_config()
        if config:
            return config.get("power_entity")
        return None

    def get_yield_entity_id(self) -> str | None:
        """Get yield sensor entity ID from SFML config. @zara"""
        config = self._get_sfml_config()
        if config:
            return config.get("solar_yield_today")
        return None

    def get_live_power(self) -> float | None:
        """Get current solar power from HA sensor. @zara"""
        if self.hass is None:
            return None

        entity_id = self.get_power_entity_id()
        if not entity_id:
            return None

        state = self.hass.states.get(entity_id)
        if not state or state.state in ('unknown', 'unavailable'):
            return None

        try:
            return float(state.state)
        except (ValueError, TypeError):
            return None

    def get_live_yield(self) -> float | None:
        """Get current daily yield from HA sensor. @zara"""
        if self.hass is None:
            return None

        entity_id = self.get_yield_entity_id()
        if not entity_id:
            return None

        state = self.hass.states.get(entity_id)
        if not state or state.state in ('unknown', 'unavailable'):
            return None

        try:
            return float(state.state)
        except (ValueError, TypeError):
            return None

    async def get_today_data(self) -> dict[str, Any]:
        """Get today's solar data from SFML database. @zara"""
        if not self.is_available:
            return {}

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT yield_today_kwh, yield_today_sensor,
                           peak_today_power_w, peak_today_at,
                           consumption_today_kwh, autarky_percent
                    FROM daily_forecast_tracking
                    WHERE id = 1
                """) as cursor:
                    row = await cursor.fetchone()

                if row:
                    return {
                        "yield_kwh": row["yield_today_kwh"],
                        "yield_sensor": row["yield_today_sensor"],
                        "peak_power_w": row["peak_today_power_w"],
                        "peak_at": row["peak_today_at"],
                        "consumption_kwh": row["consumption_today_kwh"],
                        "autarky_percent": row["autarky_percent"],
                    }
        except Exception as e:
            _LOGGER.error("Error reading today's data from SFML DB: %s", e)

        return {}

    async def get_yield_cache(self) -> dict[str, Any]:
        """Get yield cache with timestamp. @zara"""
        if not self.is_available:
            return {}

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT value, time, date FROM yield_cache WHERE id = 1
                """) as cursor:
                    row = await cursor.fetchone()

                if row:
                    return {
                        "value": row["value"],
                        "time": row["time"],
                        "date": row["date"],
                    }
        except Exception as e:
            _LOGGER.error("Error reading yield cache from SFML DB: %s", e)

        return {}

    async def get_daily_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get daily solar production history from SFML database. @zara"""
        if not self.is_available:
            return []

        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            async with self._get_db() as db:

                async with db.execute("""
                    SELECT cache_date, group_name,
                           prediction_total_kwh, actual_total_kwh
                    FROM panel_group_daily_cache
                    WHERE cache_date >= ?
                    ORDER BY cache_date DESC, group_name
                """, (cutoff,)) as cursor:
                    rows = await cursor.fetchall()

            daily_data: dict[str, dict[str, Any]] = {}
            for row in rows:
                date = row["cache_date"]
                if date not in daily_data:
                    daily_data[date] = {
                        "date": date,
                        "actual_kwh": 0.0,
                        "predicted_kwh": 0.0,
                        "groups": {}
                    }

                actual = row["actual_total_kwh"] or 0
                predicted = row["prediction_total_kwh"] or 0

                daily_data[date]["actual_kwh"] += actual
                daily_data[date]["predicted_kwh"] += predicted
                daily_data[date]["groups"][row["group_name"]] = {
                    "actual_kwh": actual,
                    "predicted_kwh": predicted,
                }

            return list(daily_data.values())

        except Exception as e:
            _LOGGER.error("Error reading daily history from SFML DB: %s", e)
            return []

    async def get_daily_summaries(self, days: int = 30) -> list[dict[str, Any]]:
        """Get daily summaries from SFML database. @zara"""
        if not self.is_available:
            return []

        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT date, predicted_total_kwh, actual_total_kwh,
                           accuracy_percent, peak_power_w, peak_hour,
                           production_hours
                    FROM daily_summaries
                    WHERE date >= ?
                    ORDER BY date DESC
                """, (cutoff,)) as cursor:
                    rows = await cursor.fetchall()

            return [
                {
                    "date": row["date"],
                    "predicted_kwh": row["predicted_total_kwh"],
                    "actual_kwh": row["actual_total_kwh"],
                    "accuracy_percent": row["accuracy_percent"],
                    "peak_power_w": row["peak_power_w"],
                    "peak_hour": row["peak_hour"],
                    "production_hours": row["production_hours"],
                }
                for row in rows
            ]
        except Exception as e:
            _LOGGER.error("Error reading daily summaries from SFML DB: %s", e)
            return []

    async def get_hourly_history(self, days: int = 7) -> list[dict[str, Any]]:
        """Get hourly solar production history from SFML database. @zara"""
        if not self.is_available:
            return []

        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT cache_date, hour, group_name,
                           prediction_kwh, actual_kwh
                    FROM panel_group_daily_hourly
                    WHERE cache_date >= ?
                    ORDER BY cache_date DESC, hour DESC, group_name
                """, (cutoff,)) as cursor:
                    rows = await cursor.fetchall()

            hourly_data: dict[str, dict[str, Any]] = {}
            for row in rows:
                key = f"{row['cache_date']}_{row['hour']:02d}"
                if key not in hourly_data:
                    hourly_data[key] = {
                        "date": row["cache_date"],
                        "hour": row["hour"],
                        "actual_kwh": 0.0,
                        "predicted_kwh": 0.0,
                        "groups": {}
                    }

                actual = row["actual_kwh"] or 0
                predicted = row["prediction_kwh"] or 0

                hourly_data[key]["actual_kwh"] += actual
                hourly_data[key]["predicted_kwh"] += predicted
                hourly_data[key]["groups"][row["group_name"]] = {
                    "actual_kwh": actual,
                    "predicted_kwh": predicted,
                }

            return list(hourly_data.values())

        except Exception as e:
            _LOGGER.error("Error reading hourly history from SFML DB: %s", e)
            return []

    async def get_hourly_predictions(self, date: str | None = None) -> list[dict[str, Any]]:
        """Get hourly predictions for a specific date from SFML database. @zara"""
        if not self.is_available:
            return []

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT target_hour, prediction_kwh, actual_kwh,
                           accuracy_percent, confidence
                    FROM hourly_predictions
                    WHERE target_date = ?
                    ORDER BY target_hour
                """, (date,)) as cursor:
                    rows = await cursor.fetchall()

            return [
                {
                    "hour": row["target_hour"],
                    "predicted_kwh": row["prediction_kwh"],
                    "actual_kwh": row["actual_kwh"],
                    "accuracy_percent": row["accuracy_percent"],
                    "confidence": row["confidence"],
                }
                for row in rows
            ]
        except Exception as e:
            _LOGGER.error("Error reading hourly predictions from SFML DB: %s", e)
            return []

    def _pivot_panel_rows(self, rows, date_field: str = "target_date") -> list[dict[str, Any]]:
        """Pivot prediction_panel_groups rows into hourly yield dicts. @zara"""
        from collections import defaultdict
        by_hour: dict[tuple, dict[str, float]] = defaultdict(dict)
        dates_by_hour: dict[tuple, str] = {}
        for row in rows:
            key = (row[date_field], row["target_hour"])
            dates_by_hour[key] = row[date_field]
            by_hour[key][row["group_name"]] = row["actual_kwh"] or 0.0

        result = []
        for key in sorted(by_hour):
            groups = by_hour[key]
            entry = {
                "date": dates_by_hour[key],
                "hour": key[1],
                "yield_total_kwh": sum(groups.values()),
                "groups": groups,
            }
            result.append(entry)
        return result

    async def get_hourly_yield(
        self,
        date: str | None = None,
        hour: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get hourly solar yield from prediction_panel_groups. @zara"""
        if not self.is_available:
            return []

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        try:
            async with self._get_db() as db:

                query = """
                    SELECT hp.target_date, hp.target_hour,
                           ppg.group_name, ppg.actual_kwh
                    FROM prediction_panel_groups ppg
                    JOIN hourly_predictions hp ON hp.prediction_id = ppg.prediction_id
                    WHERE hp.target_date = ?"""
                params: list = [date]

                if hour is not None:
                    query += " AND hp.target_hour = ?"
                    params.append(hour)

                query += " ORDER BY hp.target_hour, ppg.group_name"

                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

            return self._pivot_panel_rows(rows)
        except Exception as e:
            _LOGGER.error("Error reading hourly yield from prediction_panel_groups: %s", e)
            return []

    async def get_daily_yield_from_hourly(self, date: str | None = None) -> dict[str, Any]:
        """Get daily yield totals by summing panel group actuals. @zara"""
        if not self.is_available:
            return {}

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT ppg.group_name,
                           SUM(ppg.actual_kwh) as total
                    FROM prediction_panel_groups ppg
                    JOIN hourly_predictions hp ON hp.prediction_id = ppg.prediction_id
                    WHERE hp.target_date = ?
                    GROUP BY ppg.group_name
                """, (date,)) as cursor:
                    rows = await cursor.fetchall()

            if not rows:
                return {}

            groups = {row["group_name"]: row["total"] or 0.0 for row in rows}
            return {
                "date": date,
                "yield_total_kwh": sum(groups.values()),
                "groups": groups,
            }
        except Exception as e:
            _LOGGER.error("Error reading daily yield from prediction_panel_groups: %s", e)

        return {}

    async def get_hourly_yield_range(
        self,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """Get hourly yield for a date range from prediction_panel_groups. @zara"""
        if not self.is_available:
            return []

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT hp.target_date, hp.target_hour,
                           ppg.group_name, ppg.actual_kwh
                    FROM prediction_panel_groups ppg
                    JOIN hourly_predictions hp ON hp.prediction_id = ppg.prediction_id
                    WHERE hp.target_date >= ? AND hp.target_date <= ?
                    ORDER BY hp.target_date, hp.target_hour, ppg.group_name
                """, (start_date, end_date)) as cursor:
                    rows = await cursor.fetchall()

            return self._pivot_panel_rows(rows)
        except Exception as e:
            _LOGGER.error("Error reading hourly yield range: %s", e)
            return []

    async def get_panel_groups(self) -> list[dict[str, Any]]:
        """Get panel group configurations from SFML. @zara"""
        config = self._get_sfml_config()
        if not config:
            return []

        panel_groups_str = config.get("panel_groups", "")
        if not panel_groups_str:
            return []

        groups = []
        for i, group_str in enumerate(panel_groups_str.split(",")):
            parts = group_str.strip().split("/")
            if len(parts) >= 3:
                group = {
                    "name": f"Gruppe {i+1}",
                    "power_wp": float(parts[0]) if parts[0] else 0,
                    "azimuth": float(parts[1]) if parts[1] else 0,
                    "tilt": float(parts[2]) if parts[2] else 0,
                }
                if len(parts) >= 4 and parts[3]:
                    group["energy_sensor"] = parts[3]
                groups.append(group)

        return groups

    async def get_panel_group_sensor_states(self) -> dict[str, float]:
        """Get current sensor values for all panel groups. @zara"""
        if not self.is_available:
            return {}

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT group_name, last_value, last_updated
                    FROM panel_group_sensor_state
                """) as cursor:
                    rows = await cursor.fetchall()

            return {
                row["group_name"]: {
                    "value": row["last_value"],
                    "last_updated": row["last_updated"],
                }
                for row in rows
            }
        except Exception as e:
            _LOGGER.error("Error reading panel group sensor states from SFML DB: %s", e)
            return {}

    async def get_forecast(self) -> dict[str, Any]:
        """Get current forecast from SFML database. @zara"""
        if not self.is_available:
            return {}

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT forecast_type, forecast_date, prediction_kwh, locked
                    FROM daily_forecasts
                    ORDER BY created_at DESC
                """) as cursor:
                    rows = await cursor.fetchall()

            forecasts = {}
            for row in rows:
                forecast_type = row["forecast_type"]
                if forecast_type not in forecasts:
                    forecasts[forecast_type] = {
                        "date": row["forecast_date"],
                        "prediction_kwh": row["prediction_kwh"],
                        "locked": row["locked"],
                    }

            return forecasts
        except Exception as e:
            _LOGGER.error("Error reading forecast from SFML DB: %s", e)
            return {}

    async def get_statistics(self) -> dict[str, Any]:
        """Get solar statistics from SFML database. @zara"""
        if not self.is_available:
            return {}

        try:
            async with self._get_db() as db:

                async with db.execute("""
                    SELECT * FROM daily_statistics WHERE id = 1
                """) as cursor:
                    stats_row = await cursor.fetchone()

                if not stats_row:
                    return {}

                return {
                    "all_time_peak": {
                        "power_w": stats_row["all_time_peak_power_w"],
                        "date": stats_row["all_time_peak_date"],
                        "at": stats_row["all_time_peak_at"],
                    },
                    "current_week": {
                        "period": stats_row["current_week_period"],
                        "yield_kwh": stats_row["current_week_yield_kwh"],
                        "consumption_kwh": stats_row["current_week_consumption_kwh"],
                        "days": stats_row["current_week_days"],
                    },
                    "current_month": {
                        "period": stats_row["current_month_period"],
                        "yield_kwh": stats_row["current_month_yield_kwh"],
                        "consumption_kwh": stats_row["current_month_consumption_kwh"],
                        "avg_autarky": stats_row["current_month_avg_autarky"],
                        "days": stats_row["current_month_days"],
                    },
                    "last_7_days": {
                        "avg_yield_kwh": stats_row["last_7_days_avg_yield_kwh"],
                        "avg_accuracy": stats_row["last_7_days_avg_accuracy"],
                        "total_yield_kwh": stats_row["last_7_days_total_yield_kwh"],
                    },
                    "last_30_days": {
                        "avg_yield_kwh": stats_row["last_30_days_avg_yield_kwh"],
                        "avg_accuracy": stats_row["last_30_days_avg_accuracy"],
                        "total_yield_kwh": stats_row["last_30_days_total_yield_kwh"],
                    },
                }
        except Exception as e:
            _LOGGER.error("Error reading statistics from SFML DB: %s", e)
            return {}

    async def get_all_solar_data(self) -> dict[str, Any]:
        """Get all solar data in one call. @zara"""
        return {
            "live": {
                "power_w": self.get_live_power(),
                "yield_kwh": self.get_live_yield(),
                "today": await self.get_today_data(),
            },
            "history": {
                "daily": await self.get_daily_history(days=30),
                "hourly": await self.get_hourly_history(days=7),
            },
            "forecast": await self.get_forecast(),
            "statistics": await self.get_statistics(),
            "panel_groups": await self.get_panel_groups(),
            "sfml_available": self.is_available,
        }
