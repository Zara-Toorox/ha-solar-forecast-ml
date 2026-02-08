# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Coordinator Update Helpers for Solar Forecast ML V16.2.0.
Provides helper methods for coordinator data updates and forecast generation.
Uses DatabaseManager for state operations.
"""

import logging
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

from homeassistant.helpers.update_coordinator import UpdateFailed

from .core_helpers import SafeDateTimeUtil as dt_util
from ..data.db_manager import DatabaseManager
from ..const import (
    DATA_KEY_FORECAST_TODAY,
    DATA_KEY_FORECAST_TOMORROW,
    DATA_KEY_FORECAST_DAY_AFTER,
    DATA_KEY_HOURLY_FORECAST,
    DATA_KEY_CURRENT_WEATHER,
    DATA_KEY_EXTERNAL_SENSORS,
    DATA_KEY_PRODUCTION_TIME,
    DATA_KEY_PEAK_TODAY,
    DATA_KEY_YIELD_TODAY,
    DATA_KEY_EXPECTED_DAILY_PRODUCTION,
    DATA_KEY_STATISTICS,
    PROD_TIME_ACTIVE,
    PROD_TIME_DURATION_SECONDS,
    PROD_TIME_START_TIME,
    PROD_TIME_END_TIME,
    PEAK_TODAY_POWER_W,
    PEAK_TODAY_AT,
    YIELD_TODAY_KWH,
    YIELD_TODAY_SENSOR,
    FORECAST_KEY_TODAY,
    FORECAST_KEY_TOMORROW,
    FORECAST_KEY_DAY_AFTER,
    FORECAST_KEY_HOURLY,
    FORECAST_KEY_METHOD,
    EXT_SENSOR_SOLAR_YIELD_TODAY,
    STATS_ALL_TIME_PEAK,
    STATS_LAST_7_DAYS,
    STATS_LAST_30_DAYS,
    STATS_AVG_ACCURACY,
    STATS_YIELD_KWH,
    STATS_AVG_YIELD_KWH,
    STATS_CONSUMPTION_KWH,
    STATS_CURRENT_MONTH,
    STATS_CURRENT_WEEK,
)

_LOGGER = logging.getLogger(__name__)


class CoordinatorUpdateHelpers:
    """Helper methods for coordinator data updates. @zara"""

    def __init__(self, coordinator: Any, db_manager: DatabaseManager):
        self.coordinator = coordinator
        self.db = db_manager

    async def fetch_weather_data(self) -> Tuple[Optional[Dict], Optional[list]]:
        """Fetch current weather and hourly forecast. @zara"""
        current_weather = None
        hourly_forecast = None

        if self.coordinator.weather_service:
            try:
                current_weather = await self.coordinator.weather_service.get_current_weather()
                hourly_forecast = (
                    await self.coordinator.weather_service.get_corrected_hourly_forecast()
                )
                self.coordinator._last_weather_update = dt_util.now()

                if not hourly_forecast or len(hourly_forecast) == 0:
                    _LOGGER.warning("Weather service returned no hourly forecast data")

            except Exception as e:
                _LOGGER.error(f"Error fetching weather data: {e}", exc_info=True)

        return current_weather, hourly_forecast

    async def generate_forecast(
        self,
        current_weather: Optional[Dict],
        hourly_forecast: Optional[list],
        external_sensors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate forecast using orchestrator. @zara"""
        forecast = await self.coordinator.forecast_orchestrator.orchestrate_forecast(
            current_weather=current_weather,
            hourly_forecast=hourly_forecast,
            external_sensors=external_sensors,
            ml_prediction_today=None,
            ml_prediction_tomorrow=None,
            correction_factor=self.coordinator.learned_correction_factor,
        )

        if not forecast:
            raise UpdateFailed("Forecast generation failed")

        hourly_count = len(forecast.get(FORECAST_KEY_HOURLY, []))
        _LOGGER.debug(
            f"Forecast generated: today={forecast.get(FORECAST_KEY_TODAY, 'N/A')} kWh, "
            f"tomorrow={forecast.get(FORECAST_KEY_TOMORROW, 'N/A')} kWh, "
            f"method={forecast.get(FORECAST_KEY_METHOD, 'unknown')}, "
            f"hourly_entries={hourly_count}"
        )
        return forecast

    async def build_coordinator_result(
        self,
        forecast: Dict[str, Any],
        current_weather: Optional[Dict],
        external_sensors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build coordinator result dictionary. @zara"""
        # Load statistics from database @zara
        statistics = await self._load_statistics()

        # Load peak today from database if not set on coordinator @zara
        peak_power_today = getattr(self.coordinator, "_peak_power_today", 0.0)
        peak_time_today = getattr(self.coordinator, "_peak_time_today", None)

        if peak_power_today == 0.0:
            today_peak = await self._get_today_peak()
            if today_peak:
                peak_power_today = today_peak.get("power_w", 0.0)
                peak_time_today = today_peak.get("time")

        result = {
            DATA_KEY_FORECAST_TODAY: forecast.get(FORECAST_KEY_TODAY),
            DATA_KEY_FORECAST_TOMORROW: forecast.get(FORECAST_KEY_TOMORROW),
            DATA_KEY_FORECAST_DAY_AFTER: forecast.get(FORECAST_KEY_DAY_AFTER),
            DATA_KEY_HOURLY_FORECAST: forecast.get(FORECAST_KEY_HOURLY, []) if self.coordinator.enable_hourly else [],
            DATA_KEY_CURRENT_WEATHER: current_weather,
            DATA_KEY_EXTERNAL_SENSORS: external_sensors,
            DATA_KEY_PRODUCTION_TIME: {
                PROD_TIME_ACTIVE: self.coordinator.production_time_calculator.is_active,
                PROD_TIME_DURATION_SECONDS: self.coordinator.production_time_calculator.total_seconds,
                PROD_TIME_START_TIME: self.coordinator.production_time_calculator.start_time,
                PROD_TIME_END_TIME: self.coordinator.production_time_calculator.end_time,
            },
            DATA_KEY_PEAK_TODAY: {
                PEAK_TODAY_POWER_W: peak_power_today,
                PEAK_TODAY_AT: peak_time_today,
            },
            DATA_KEY_YIELD_TODAY: {
                YIELD_TODAY_KWH: external_sensors.get(EXT_SENSOR_SOLAR_YIELD_TODAY),
                YIELD_TODAY_SENSOR: self.coordinator.solar_yield_today,
            },
            DATA_KEY_EXPECTED_DAILY_PRODUCTION: self.coordinator.expected_daily_production,
            DATA_KEY_STATISTICS: statistics,
        }
        return result

    async def _load_statistics(self) -> Dict[str, Any]:
        """Load statistics from database. @zara"""
        statistics = {
            STATS_ALL_TIME_PEAK: {},
            STATS_LAST_7_DAYS: {},
            STATS_LAST_30_DAYS: {},
            STATS_CURRENT_MONTH: {},
            STATS_CURRENT_WEEK: {},
        }

        try:
            # Load all-time peak from production_time_state @zara
            all_time_peak = await self.db.fetchone(
                """SELECT peak_record_w, peak_record_date, peak_record_time
                   FROM production_time_state
                   WHERE id = 1 AND peak_record_w IS NOT NULL"""
            )
            if all_time_peak and all_time_peak[0]:
                statistics[STATS_ALL_TIME_PEAK] = {
                    PEAK_TODAY_POWER_W: float(all_time_peak[0]),
                    PEAK_TODAY_AT: f"{all_time_peak[1]} {all_time_peak[2]}" if all_time_peak[1] and all_time_peak[2] else None,
                    "date": all_time_peak[1],
                }

            # Load 7-day statistics @zara
            stats_7d = await self.db.fetchone(
                """SELECT AVG(actual_total_kwh), AVG(accuracy_percent), SUM(actual_total_kwh)
                   FROM daily_summaries
                   WHERE date >= DATE('now', '-7 days') AND actual_total_kwh IS NOT NULL"""
            )
            if stats_7d:
                statistics[STATS_LAST_7_DAYS] = {
                    STATS_AVG_YIELD_KWH: float(stats_7d[0]) if stats_7d[0] else None,
                    STATS_AVG_ACCURACY: float(stats_7d[1]) if stats_7d[1] else None,
                    STATS_YIELD_KWH: float(stats_7d[2]) if stats_7d[2] else None,
                }

            # Load 30-day statistics @zara
            stats_30d = await self.db.fetchone(
                """SELECT AVG(actual_total_kwh), AVG(accuracy_percent), SUM(actual_total_kwh)
                   FROM daily_summaries
                   WHERE date >= DATE('now', '-30 days') AND actual_total_kwh IS NOT NULL"""
            )
            if stats_30d:
                statistics[STATS_LAST_30_DAYS] = {
                    STATS_AVG_YIELD_KWH: float(stats_30d[0]) if stats_30d[0] else None,
                    STATS_AVG_ACCURACY: float(stats_30d[1]) if stats_30d[1] else None,
                    STATS_YIELD_KWH: float(stats_30d[2]) if stats_30d[2] else None,
                }

            # Load current month statistics @zara
            month_start = dt_util.now().date().replace(day=1).isoformat()

            month_yield = await self.db.fetchone(
                """SELECT SUM(actual_total_kwh)
                   FROM daily_summaries
                   WHERE date >= ? AND actual_total_kwh IS NOT NULL""",
                (month_start,)
            )

            month_consumption = await self.db.fetchone(
                """SELECT SUM(consumption_kwh)
                   FROM forecast_history
                   WHERE date >= ? AND consumption_kwh IS NOT NULL""",
                (month_start,)
            )

            statistics[STATS_CURRENT_MONTH] = {
                STATS_YIELD_KWH: float(month_yield[0]) if month_yield and month_yield[0] else 0.0,
                STATS_CONSUMPTION_KWH: float(month_consumption[0]) if month_consumption and month_consumption[0] else 0.0,
            }

            # Load current week statistics @zara
            week_start = (dt_util.now().date() - timedelta(
                days=dt_util.now().date().weekday()
            )).isoformat()

            week_yield = await self.db.fetchone(
                """SELECT SUM(actual_total_kwh)
                   FROM daily_summaries
                   WHERE date >= ? AND actual_total_kwh IS NOT NULL""",
                (week_start,)
            )

            week_consumption = await self.db.fetchone(
                """SELECT SUM(consumption_kwh)
                   FROM forecast_history
                   WHERE date >= ? AND consumption_kwh IS NOT NULL""",
                (week_start,)
            )

            statistics[STATS_CURRENT_WEEK] = {
                STATS_YIELD_KWH: float(week_yield[0]) if week_yield and week_yield[0] else 0.0,
                STATS_CONSUMPTION_KWH: float(week_consumption[0]) if week_consumption and week_consumption[0] else 0.0,
            }

        except Exception as e:
            _LOGGER.warning(f"Failed to load statistics: {e}")

        return statistics

    async def _get_today_peak(self) -> Optional[Dict[str, Any]]:
        """Get today's peak power from production_time_state. @zara"""
        try:
            today = dt_util.now().date().isoformat()
            row = await self.db.fetchone(
                """SELECT peak_power_w, peak_power_time
                   FROM production_time_state
                   WHERE id = 1 AND date = ?""",
                (today,)
            )
            if row and row[0]:
                return {
                    "power_w": float(row[0]),
                    "time": str(row[1]) if row[1] else None,
                }
        except Exception as e:
            _LOGGER.debug(f"Could not get today's peak: {e}")
        return None

    async def save_forecasts(self, forecast_data: Dict[str, Any], hourly_forecast: list) -> None:
        """Save forecasts to database. Only writes if NOT locked (morning routine has priority). @zara"""
        try:
            now = dt_util.now()
            today_date = now.date().isoformat()
            tomorrow_date = (now.date() + timedelta(days=1)).isoformat()
            day_after_date = (now.date() + timedelta(days=2)).isoformat()

            forecasts_to_save = [
                ("today", today_date, forecast_data.get(FORECAST_KEY_TODAY)),
                ("tomorrow", tomorrow_date, forecast_data.get(FORECAST_KEY_TOMORROW)),
                ("day_after_tomorrow", day_after_date, forecast_data.get(FORECAST_KEY_DAY_AFTER)),
            ]

            for forecast_type, forecast_date, kwh_value in forecasts_to_save:
                if kwh_value is None:
                    continue

                # Check if locked - morning routine has exclusive write access when locked
                lock_check = await self.db.fetchone(
                    """SELECT locked FROM daily_forecasts
                        WHERE forecast_type = ? AND forecast_date = ?""",
                    (forecast_type, forecast_date)
                )

                if lock_check and lock_check[0]:
                    _LOGGER.debug(
                        "Skipping %s forecast update - locked by morning routine",
                        forecast_type
                    )
                    continue

                # Not locked - safe to write
                await self.db.execute(
                    """INSERT INTO daily_forecasts
                        (forecast_type, forecast_date, prediction_kwh, source)
                        VALUES (?, ?, ?, 'coordinator_update')
                        ON CONFLICT(forecast_type, forecast_date) DO UPDATE SET
                            prediction_kwh = excluded.prediction_kwh,
                            source = excluded.source
                        WHERE locked = 0 OR locked IS NULL""",
                    (forecast_type, forecast_date, kwh_value)
                )

            _LOGGER.debug("Forecasts saved to database (respecting locks)")

        except Exception as e:
            _LOGGER.error(f"Error saving forecasts to database: {e}", exc_info=True)

    async def handle_startup_recovery(self) -> None:
        """Handle startup recovery for missing forecasts. @zara"""
        try:
            today_forecast = await self.db.fetchone(
                """SELECT prediction_kwh, locked
                   FROM daily_forecasts
                   WHERE forecast_type = 'today' AND forecast_date = DATE('now')
                   ORDER BY created_at DESC LIMIT 1"""
            )

            now_local = dt_util.now()

            if not today_forecast or not today_forecast[1]:
                if now_local.hour < 12:
                    _LOGGER.info(
                        "System started without locked forecast (before 12:00) - initiating recovery"
                    )
                    if hasattr(self.coordinator, '_recovery_forecast_process'):
                        await self.coordinator._recovery_forecast_process(source="startup_recovery")
                else:
                    _LOGGER.info(
                        "System started late without forecast (after 12:00) - "
                        "using current forecast (NOT morning baseline)"
                    )
                    if self.coordinator.data and DATA_KEY_FORECAST_TODAY in self.coordinator.data:
                        forecast_value = self.coordinator.data.get(DATA_KEY_FORECAST_TODAY)
                        # Only insert if no entry exists - never overwrite existing (locked or not)
                        await self.db.execute(
                            """INSERT OR IGNORE INTO daily_forecasts
                               (forecast_type, forecast_date, prediction_kwh, source, locked)
                               VALUES ('today', DATE('now'), ?, ?, TRUE)""",
                            (forecast_value, f"late_startup_{now_local.hour:02d}:{now_local.minute:02d}")
                        )
                        _LOGGER.info(
                            f"Set forecast to current value: {forecast_value:.2f} kWh "
                            f"(not representative of morning prediction)"
                        )

        except Exception as e:
            _LOGGER.error(f"Error during startup recovery: {e}", exc_info=True)
