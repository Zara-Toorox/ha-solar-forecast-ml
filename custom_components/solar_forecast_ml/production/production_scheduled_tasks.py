"""Scheduled Task Management for Production Tracking V10.0.0 @zara

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

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_change

from .production_morning_routine import MorningRoutineHandler

from ..const import (
    CONF_HUMIDITY_SENSOR,
    CONF_LUX_SENSOR,
    CONF_PRESSURE_SENSOR,
    CONF_RAIN_SENSOR,
    CONF_SOLAR_RADIATION_SENSOR,
    CONF_TEMP_SENSOR,
    CONF_WIND_SENSOR,
    CORRECTION_FACTOR_MAX,
    CORRECTION_FACTOR_MIN,
    DAILY_UPDATE_HOUR,
    DAILY_VERIFICATION_HOUR,
)
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..data.data_manager import DataManager
from ..data.data_sensor_cloud_correction import SensorCloudCorrection
from ..ml.ml_types import LearnedWeights, create_default_learned_weights

if TYPE_CHECKING:
    from ..coordinator import SolarForecastMLCoordinator

_LOGGER = logging.getLogger(__name__)

class ScheduledTasksManager:
    """Manages scheduled tasks for the Solar Forecast ML integration"""

    def __init__(
        self,
        hass: HomeAssistant,
        coordinator: "SolarForecastMLCoordinator",
        solar_yield_today_entity_id: Optional[str],
        data_manager: DataManager,
    ):
        """Initialize the ScheduledTasksManager"""
        self.hass = hass
        self.coordinator = coordinator
        self.solar_yield_today_entity_id = solar_yield_today_entity_id
        self.data_manager = data_manager
        self._listeners = []

        self.morning_routine_handler = MorningRoutineHandler(data_manager, coordinator)

        solar_rad_sensor = coordinator.entry.data.get(CONF_SOLAR_RADIATION_SENSOR)
        lux_sensor = coordinator.entry.data.get(CONF_LUX_SENSOR)
        self.sensor_cloud_correction = SensorCloudCorrection(
            hass=hass,
            data_manager=data_manager,
            solar_radiation_sensor=solar_rad_sensor,
            lux_sensor=lux_sensor,
        )

        self._end_of_day_running = False

        self._pattern_learner = None
        self._weather_corrector = None

        _LOGGER.debug("ScheduledTasksManager initialized (with bulletproof morning routine handler).")

    def _get_pattern_learner(self):
        """Get or create PatternLearner instance (lazy initialization) @zara"""
        if self._pattern_learner is None:
            from ..ml.ml_pattern_learner import PatternLearner
            self._pattern_learner = PatternLearner(self.data_manager.data_dir)
            _LOGGER.debug("PatternLearner lazy-initialized")
        return self._pattern_learner

    def _get_weather_corrector(self):
        """Get or create WeatherForecastCorrector instance (lazy initialization) @zara"""
        if self._weather_corrector is None:
            from ..data.data_weather_corrector import WeatherForecastCorrector
            self._weather_corrector = WeatherForecastCorrector(
                self.hass,
                self.data_manager
            )
            _LOGGER.debug("WeatherForecastCorrector lazy-initialized")
        return self._weather_corrector

    def setup_listeners(self) -> None:
        """Register the time-based listeners with Home Assistant @zara"""
        self.cancel_listeners()

        local_now = dt_util.now()
        _LOGGER.info(f"All scheduled tasks use LOCAL time (async_track_time_change behavior)")

        remove_morning_1 = async_track_time_change(
            self.hass, self.morning_routine_complete, hour=0, minute=25, second=0
        )
        self._listeners.append(remove_morning_1)
        _LOGGER.info(f"Scheduled MORNING ROUTINE #1 for 00:25:00 LOCAL (Primary - after corrected forecast)")

        remove_morning_2 = async_track_time_change(
            self.hass, self.morning_routine_complete, hour=4, minute=15, second=0
        )
        self._listeners.append(remove_morning_2)
        _LOGGER.info(f"Scheduled MORNING ROUTINE #2 for 04:15:00 LOCAL (Refresh/Fallback)")

        remove_reset_expected = async_track_time_change(
            self.hass, self.reset_expected_production, hour=0, minute=0, second=0
        )
        self._listeners.append(remove_reset_expected)
        _LOGGER.info(f"Scheduled expected daily production reset for 00:00:00 LOCAL")

        remove_hourly_actuals = async_track_time_change(
            self.hass, self.update_hourly_actuals, minute=5, second=0
        )
        self._listeners.append(remove_hourly_actuals)
        _LOGGER.info(f"Scheduled HOURLY ACTUALS UPDATE for every hour at :05:00 LOCAL")

        if self.sensor_cloud_correction.is_enabled:
            remove_sensor_check = async_track_time_change(
                self.hass, self.sensor_cloud_morning_check, hour=9, minute=5, second=0
            )
            self._listeners.append(remove_sensor_check)
            _LOGGER.info(
                f"Scheduled SENSOR CLOUD CORRECTION for 09:05:00 LOCAL "
                f"(validates forecast against W/m²/Lux sensor)"
            )
        else:
            _LOGGER.debug(
                "Sensor cloud correction NOT scheduled - no W/m² or Lux sensor configured"
            )

        remove_end_of_day = async_track_time_change(
            self.hass, self.end_of_day_workflow, hour=23, minute=30, second=0
        )
        self._listeners.append(remove_end_of_day)
        _LOGGER.info(
            f"Scheduled END_OF_DAY_WORKFLOW for 23:30:00 LOCAL (Finalize+History+DailySummary+Stats+Cleanup)"
        )

        _LOGGER.info(
            "Daily Briefing: Configure automation to call 'solar_forecast_ml.send_daily_briefing' service "
            "(see examples/automation_daily_briefing.yaml)"
        )

    def cancel_listeners(self) -> None:
        """Remove any active time-based listeners @zara"""
        for remove_listener in self._listeners:
            try:
                remove_listener()
            except Exception as e:
                _LOGGER.warning(f"Error removing scheduled task listener: {e}")
        self._listeners = []
        _LOGGER.debug("Cancelled scheduled task listeners.")

    async def calculate_yesterday_deviation_on_startup(self) -> None:
        """Calculates the forecast deviation from yesterday using daily_forecasts.json @zara"""
        _LOGGER.info("Calculating yesterday's forecast deviation at startup...")

        try:

            daily_forecasts = await self.data_manager.load_daily_forecasts()

            if not daily_forecasts:
                _LOGGER.info(
                    "No daily_forecasts.json available. Cannot calculate yesterday's deviation."
                )
                self.coordinator.last_day_error_kwh = 0.0
                self.coordinator.yesterday_accuracy = 0.0
                return

            yesterday = daily_forecasts.get("yesterday", {})

            if not yesterday:
                _LOGGER.info("No yesterday data in daily_forecasts.json")
                self.coordinator.last_day_error_kwh = 0.0
                self.coordinator.yesterday_accuracy = 0.0
                return

            forecast_day = yesterday.get("forecast_day", {})
            actual_day = yesterday.get("actual_day", {})

            forecast_kwh = forecast_day.get("prediction_kwh")
            actual_kwh = actual_day.get("actual_kwh")

            if forecast_kwh is None or actual_kwh is None:
                _LOGGER.info("Yesterday data incomplete (missing forecast or actual)")
                self.coordinator.last_day_error_kwh = 0.0
                self.coordinator.yesterday_accuracy = 0.0
                return

            deviation = abs(forecast_kwh - actual_kwh)

            accuracy = 0.0
            if actual_kwh > 0.1:
                error_fraction = deviation / actual_kwh
                accuracy = max(0.0, 1.0 - error_fraction)
            elif forecast_kwh < 0.1 and actual_kwh < 0.1:
                accuracy = 1.0
            accuracy_pct = accuracy * 100.0

            _LOGGER.info(
                f"Yesterday's Deviation on Startup:\n"
                f"  Date:       {yesterday.get('date', 'unknown')}\n"
                f"  Forecast:   {forecast_kwh:.2f} kWh\n"
                f"  Actual:     {actual_kwh:.2f} kWh\n"
                f"  Deviation:  {deviation:.2f} kWh\n"
                f"  Accuracy:   {accuracy_pct:.1f}%"
            )

            self.coordinator.last_day_error_kwh = round(deviation, 2)
            self.coordinator.yesterday_accuracy = round(accuracy_pct, 1)

            statistics = daily_forecasts.get("statistics", {})
            current_month = statistics.get("current_month", {})
            if current_month and current_month.get("yield_kwh"):
                self.coordinator.avg_month_yield = round(current_month.get("yield_kwh", 0.0), 2)
                _LOGGER.info(
                    f"Loaded monthly average yield: {self.coordinator.avg_month_yield:.2f} kWh"
                )
            else:
                self.coordinator.avg_month_yield = 0.0

            self.coordinator.async_update_listeners()

        except Exception as e:
            _LOGGER.error(f"Error calculating yesterday's deviation on startup: {e}", exc_info=True)
            self.coordinator.last_day_error_kwh = 0.0
            self.coordinator.yesterday_accuracy = 0.0
            self.coordinator.avg_month_yield = 0.0

    @callback
    async def sensor_cloud_morning_check(self, now: datetime) -> None:
        """Callback for 09:05 sensor-based cloud correction @zara"""
        current_time = now if now is not None else dt_util.now()

        try:
            _LOGGER.info(
                f"🌅 SENSOR CLOUD CORRECTION triggered at "
                f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} LOCAL"
            )

            result = await self.sensor_cloud_correction.run_morning_check(current_time)

            if result.get("correction_applied"):
                _LOGGER.info(
                    f"☀️ Sensor correction applied! "
                    f"Sensor: {result.get('sensor_cloud_percent', 0):.0f}%, "
                    f"Forecast: {result.get('forecast_cloud_percent', 0):.0f}%, "
                    f"Reason: {result.get('reason')}"
                )

                await self.coordinator.async_request_refresh()
            else:
                _LOGGER.info(
                    f"✓ Sensor check completed - no correction needed. "
                    f"Reason: {result.get('reason')}"
                )

        except Exception as e:
            _LOGGER.error(f"Error in sensor cloud morning check: {e}", exc_info=True)

    @callback
    async def scheduled_morning_update(self, now: datetime) -> None:
        """Callback for the scheduled morning task Triggers a full forecast update @zara"""
        current_time = now if now is not None else dt_util.now()
        _LOGGER.info(
            f"Triggering daily morning forecast update (Local Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')})..."
        )

        try:

            await self.coordinator.async_request_refresh()
            _LOGGER.info("Morning forecast update request successful.")

            await asyncio.sleep(0.5)

            if self.coordinator.data and "forecast_today" in self.coordinator.data:
                forecast_today = self.coordinator.data.get("forecast_today")
                if forecast_today is not None:
                    self.coordinator.expected_daily_production = forecast_today

                    await self.data_manager.save_expected_daily_production(forecast_today)

                    self.coordinator.async_update_listeners()
                    _LOGGER.info(
                        f"Morning update complete: Expected daily production set to {forecast_today:.2f} kWh "
                        f"(saved to persistent storage)"
                    )
                else:
                    _LOGGER.error("Morning update: forecast_today is None after refresh!")
            else:
                _LOGGER.error(
                    "Morning update: coordinator.data is missing or has no forecast_today!"
                )

        except Exception as e:
            _LOGGER.error(f"Failed to complete morning forecast update: {e}", exc_info=True)

    @callback
    async def morning_routine_complete(self, now: datetime) -> None:
        """COMPLETE MORNING ROUTINE - Runs 2x daily (00:25 + 04:15) @zara"""
        current_time = now if now is not None else dt_util.now()

        _LOGGER.info(
            f"🌅 MORNING ROUTINE COMPLETE at {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        _LOGGER.info("=" * 80)

        try:

            _LOGGER.info("Step 1/2: Running morning forecast update...")
            await self.scheduled_morning_update(now)

            await asyncio.sleep(1.0)

            _LOGGER.info("Step 2/2: Creating hourly predictions...")
            await self.create_morning_hourly_predictions(now)

            _LOGGER.info("=" * 80)
            _LOGGER.info("✓ MORNING ROUTINE COMPLETE - All tasks successful")
            _LOGGER.info("=" * 80)

        except Exception as e:
            _LOGGER.error(f"✗ MORNING ROUTINE FAILED: {e}", exc_info=True)

    @callback
    async def scheduled_night_cleanup(self, now: datetime) -> None:
        """Night cleanup callback - redirects to end_of_day_workflow. @zara"""
        _LOGGER.debug("scheduled_night_cleanup redirecting to end_of_day_workflow")
        await self._night_cleanup_internal(now)

    @callback
    async def reset_expected_production(self, now: datetime) -> None:
        """Reset expected daily production at midnight @zara"""
        _LOGGER.info(
            f"Resetting expected daily production (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')})..."
        )
        try:
            await self.coordinator.reset_expected_daily_production()
            _LOGGER.info("Expected daily production reset successful.")
        except Exception as e:
            _LOGGER.error(f"Failed to reset expected daily production: {e}", exc_info=True)

    @callback
    async def set_expected_production(self, now: datetime) -> None:
        """Set expected daily production at 6 AM @zara"""
        _LOGGER.info(
            f"=== Setting expected daily production (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')}) ==="
        )
        try:

            if not self.coordinator:
                _LOGGER.error("Coordinator not available!")
                return

            if not self.coordinator.data:
                _LOGGER.warning("Coordinator data is None, forcing refresh...")
                await self.coordinator.async_request_refresh()

            await self.coordinator.set_expected_daily_production()

            saved_forecast = await self.data_manager.get_current_day_forecast()
            if (
                saved_forecast
                and saved_forecast.get("locked")
                and saved_forecast.get("prediction_kwh")
            ):
                _LOGGER.info(
                    f"Daily forecast validated: {saved_forecast.get('prediction_kwh')} kWh "
                    f"for {saved_forecast.get('date')}, locked=True"
                )
            else:
                _LOGGER.error("Daily forecast validation FAILED - forecast not properly saved!")

            _LOGGER.info("Expected daily production set successful.")
        except Exception as e:
            _LOGGER.error(f"Failed to set expected daily production: {e}", exc_info=True)

    @callback
    async def retry_forecast_setting(self, now: datetime, attempt: int) -> None:
        """Retry mechanism for setting forecast if 0600 failed @zara"""
        _LOGGER.info(
            f"=== Forecast Retry Attempt #{attempt} (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')}) ==="
        )

        try:

            current_forecast = await self.data_manager.get_current_day_forecast()

            forecast_day = current_forecast.get("forecast_day", {}) if current_forecast else {}
            if forecast_day.get("locked"):
                _LOGGER.info(
                    f"Forecast already locked with {forecast_day.get('prediction_kwh')} kWh - "
                    f"retry not needed"
                )
                return

            _LOGGER.warning(
                f"Forecast not locked at retry #{attempt} - initiating recovery process"
            )

            success = await self.coordinator._recovery_forecast_process(
                source=f"retry_06:{15*attempt:02d}"
            )

            if success:
                _LOGGER.info(f"Retry #{attempt} successful - forecast set")
            else:
                _LOGGER.error(f"Retry #{attempt} failed - forecast NOT set")
                if attempt == 3:
                    _LOGGER.critical(
                        "All retry attempts exhausted (06:00, 06:15, 06:30, 06:45) - "
                        "daily forecast NOT set!"
                    )

        except Exception as e:
            _LOGGER.error(f"Error during retry attempt #{attempt}: {e}", exc_info=True)

    @callback
    async def end_of_day_workflow(self, now: datetime) -> None:
        """Consolidated End-of-Day Workflow at 23:30 - All tasks in sequence @zara"""
        current_time = now if now is not None else dt_util.now()

        if self._end_of_day_running:
            _LOGGER.warning(
                f"END_OF_DAY_WORKFLOW already running - skipping duplicate trigger at "
                f"{current_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return

        self._end_of_day_running = True

        try:
            await self._end_of_day_workflow_internal(current_time)
        finally:

            self._end_of_day_running = False

    async def _end_of_day_workflow_internal(self, current_time: datetime) -> None:
        """Internal implementation of end-of-day workflow (called with mutex held). @zara"""
        _LOGGER.info(
            f"🌙 END_OF_DAY_WORKFLOW TRIGGERED at {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        _LOGGER.info(
            f"=== END_OF_DAY_WORKFLOW Started (Local Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}) ==="
        )

        workflow_start = asyncio.get_event_loop().time()

        steps_completed = 0
        total_steps = 9
        errors = []

        try:
            _LOGGER.info("Step 1/9: Finalizing day...")
            await self._finalize_day_internal(current_time)
            steps_completed += 1
            _LOGGER.info("✓ Step 1/9 completed")
        except Exception as e:
            _LOGGER.error(f"✗ Step 1/8 failed: {e}", exc_info=True)
            errors.append(f"Finalize: {str(e)}")

        try:
            _LOGGER.info("Step 2/9: Creating daily summary (ML analysis)...")
            today = current_time.date().isoformat()

            hourly_predictions = (
                await self.coordinator.data_manager.hourly_predictions.get_predictions_for_date(
                    today
                )
            )

            if hourly_predictions:
                success = await self.coordinator.data_manager.daily_summaries.create_daily_summary(
                    date=today, hourly_predictions=hourly_predictions
                )

                if success:
                    summary = await self.coordinator.data_manager.daily_summaries.get_summary(today)
                    _LOGGER.info(f"✓ Daily summary created:")
                    _LOGGER.info(
                        f"  Overall accuracy: {summary['overall']['accuracy_percent']:.1f}%"
                    )
                    _LOGGER.info(f"  Patterns detected: {len(summary.get('patterns', []))}")
                    _LOGGER.info(f"  Recommendations: {len(summary.get('recommendations', []))}")

                    for pattern in summary.get("patterns", []):
                        _LOGGER.warning(
                            f"  ⚠️  Pattern detected: {pattern['type']} at hours {pattern['hours']} "
                            f"({pattern['severity']} severity)"
                        )
                else:
                    _LOGGER.warning("Daily summary creation returned False")
            else:
                _LOGGER.warning(f"No hourly predictions found for {today} - skipping summary")

            steps_completed += 1
            _LOGGER.info("✓ Step 2/9 completed")

        except Exception as e:
            _LOGGER.error(f"✗ Step 2/9 failed: {e}", exc_info=True)
            errors.append(f"Daily summary: {str(e)}")

        try:
            _LOGGER.info("Step 3/9: Moving to history...")
            await self._move_to_history_internal(current_time)
            steps_completed += 1
            _LOGGER.info("✓ Step 3/9 completed")
        except Exception as e:
            _LOGGER.error(f"✗ Step 3/9 failed: {e}", exc_info=True)
            errors.append(f"History: {str(e)}")

        try:
            _LOGGER.info("Step 4/9: Updating yesterday deviation...")
            await self._update_yesterday_deviation_internal(current_time)
            steps_completed += 1
            _LOGGER.info("✓ Step 4/9 completed")
        except Exception as e:
            _LOGGER.error(f"✗ Step 4/9 failed: {e}", exc_info=True)
            errors.append(f"Deviation: {str(e)}")

        try:
            _LOGGER.info("Step 5/9: Calculating statistics...")
            await self._calculate_stats_internal(current_time)
            steps_completed += 1
            _LOGGER.info("✓ Step 5/9 completed")
        except Exception as e:
            _LOGGER.error(f"✗ Step 5/9 failed: {e}", exc_info=True)
            errors.append(f"Statistics: {str(e)}")

        try:
            _LOGGER.info("Step 6/9: Updating RB overall correction factor...")

            weather_corrector = self._get_weather_corrector()
            yesterday_str = (current_time - timedelta(days=1)).strftime("%Y-%m-%d")
            await weather_corrector.update_rb_correction(yesterday_str)
            steps_completed += 1
            _LOGGER.info("✓ Step 6/9 completed - RB correction updated (production metric)")
        except Exception as e:
            _LOGGER.error(f"✗ Step 6/9 failed: {e}", exc_info=True)
            errors.append(f"RB correction: {str(e)}")

        try:
            _LOGGER.info("Step 7/9: Running night cleanup...")
            await self._night_cleanup_internal(current_time)
            steps_completed += 1
            _LOGGER.info("✓ Step 7/9 completed")
        except Exception as e:
            _LOGGER.error(f"✗ Step 7/9 failed: {e}", exc_info=True)
            errors.append(f"Cleanup: {str(e)}")

        try:
            _LOGGER.info("Step 8/9: Learning patterns from today (KI)...")

            pattern_learner = self._get_pattern_learner()
            await pattern_learner.load_patterns()

            today = current_time.date()
            today_str = today.isoformat()

            daily_forecasts = await self.coordinator.data_manager.forecast_handler._read_json_file(
                self.coordinator.data_manager.daily_forecasts_file, None
            )

            if daily_forecasts:
                finalized = daily_forecasts.get("today", {}).get("finalized", {})
                actual_production_kwh = finalized.get("yield_kwh")

                if actual_production_kwh and actual_production_kwh > 0:

                    from ..astronomy.astronomy_cache_manager import get_cache_manager
                    import re as re_module
                    cache_manager = get_cache_manager()
                    astronomy_data = None
                    production_start_hour = 6
                    production_end_hour = 20

                    if cache_manager and cache_manager.is_loaded():
                        astronomy_data = cache_manager.get_day_data(today_str)
                        if astronomy_data:

                            prod_start = astronomy_data.get("production_window_start", "")
                            prod_end = astronomy_data.get("production_window_end", "")
                            start_match = re_module.search(r'T(\d{2}):', prod_start) if prod_start else None
                            end_match = re_module.search(r'T(\d{2}):', prod_end) if prod_end else None
                            if start_match:
                                production_start_hour = int(start_match.group(1))
                            if end_match:
                                production_end_hour = int(end_match.group(1))
                            _LOGGER.debug(
                                f"Production window for pattern learning: {production_start_hour}:00 - {production_end_hour}:00"
                            )
                    else:
                        _LOGGER.warning("Astronomy cache not loaded for pattern learning - using default window")

                    hourly_predictions = await self.coordinator.data_manager.hourly_predictions.get_predictions_for_date(today_str)

                    hourly_actuals = {}
                    hourly_conditions = {}

                    if hourly_predictions:
                        for pred in hourly_predictions:
                            hour = pred.get("target_hour")
                            actual_kwh = pred.get("actual_kwh")

                            if hour is None or actual_kwh is None:
                                continue
                            if hour < production_start_hour or hour > production_end_hour:
                                continue
                            if actual_kwh <= 0:
                                continue

                            hourly_actuals[hour] = actual_kwh

                            # FIX 2025-11-27: Use correct field name "weather_corrected" (not "weather")
                            # and correct field names: "temperature" (not "temp_c")
                            weather_data = pred.get("weather_corrected") or pred.get("weather_forecast") or {}
                            hourly_conditions[hour] = {
                                "cloud_cover_percent": weather_data.get("clouds", 100),
                                "temperature_c": weather_data.get("temperature", 0),
                                "solar_radiation_wm2": weather_data.get("solar_radiation_wm2", 0)
                            }

                    if hourly_actuals and astronomy_data:
                        await pattern_learner.update_pattern_from_day(
                            target_date=today,
                            actual_production_kwh=actual_production_kwh,
                            hourly_actuals=hourly_actuals,
                            hourly_conditions=hourly_conditions,
                            astronomy_data=astronomy_data
                        )

                        _LOGGER.info(
                            f"✓ Patterns learned from today: {actual_production_kwh:.2f} kWh total, "
                            f"{len(hourly_actuals)} hours with data"
                        )

                        try:
                            daily_forecasts = await self.coordinator.data_manager.load_daily_forecasts()
                            forecast_day = daily_forecasts.get("today", {}).get("forecast_day", {})
                            forecast_raw = forecast_day.get("prediction_kwh_raw")

                            if forecast_raw is not None:

                                try:
                                    from pathlib import Path
                                    import json

                                    corrected_file = Path(self.coordinator.data_manager.data_dir) / "stats" / "weather_forecast_corrected.json"

                                    def _load_json(file_path):
                                        """Helper to load JSON file (runs in executor) @zara"""
                                        if not file_path.exists():
                                            return None
                                        return json.loads(file_path.read_text())

                                    weather_corrected = await self.hass.async_add_executor_job(_load_json, corrected_file)
                                    if not weather_corrected:
                                        _LOGGER.warning("weather_forecast_corrected.json not found for clear-sky detection")
                                        avg_clouds = 100
                                        raise FileNotFoundError("Corrected forecast file missing")

                                    forecast_hours = weather_corrected.get("forecast", {}).get(today_str, {})

                                    start_hour = production_start_hour
                                    end_hour = production_end_hour

                                    production_clouds = [
                                        forecast_hours.get(str(h), {}).get("clouds", 0)
                                        for h in range(start_hour, end_hour + 1)
                                        if str(h) in forecast_hours
                                    ]

                                    avg_clouds = sum(production_clouds) / len(production_clouds) if production_clouds else 100

                                    _LOGGER.debug(
                                        f"Clear-sky detection: {len(production_clouds)} production hours "
                                        f"({start_hour}-{end_hour}), avg clouds: {avg_clouds:.1f}%"
                                    )
                                except Exception as weather_err:
                                    _LOGGER.warning(f"Failed to load weather forecast for clear-sky detection: {weather_err}")
                                    avg_clouds = 100

                                    start_hour = production_start_hour
                                    end_hour = production_end_hour

                                battery_peak_soc = await self._get_battery_peak_soc_today(today_str, start_hour, end_hour)

                                await pattern_learner.apply_aggressive_correction(
                                    target_date=today,
                                    forecast_raw=forecast_raw,
                                    actual_kwh=actual_production_kwh,
                                    avg_cloud_cover=avg_clouds,
                                    battery_peak_soc=battery_peak_soc
                                )
                            else:
                                _LOGGER.debug("No raw forecast available for aggressive correction")
                        except Exception as corr_err:
                            _LOGGER.warning(f"Aggressive correction failed: {corr_err}")

                    else:
                        _LOGGER.info(
                            "ℹ️ LEARNING: Insufficient hourly data for pattern learning today. "
                            "This is normal for the first few days after installation. "
                            "The system needs complete daylight hours with production data. "
                            "Pattern learning will automatically start once enough data is collected."
                        )
                else:
                    _LOGGER.info("No production today - skipping pattern learning (cloudy day or night)")

            steps_completed += 1
            _LOGGER.info("✓ Step 8/9 completed")

        except Exception as e:
            _LOGGER.error(f"✗ Step 8/9 failed: {e}", exc_info=True)
            errors.append(f"Pattern learning: {str(e)}")

        # Step 9: Residual Model Training (Phase 4 - ML Enhancement)
        try:
            _LOGGER.info("Step 9/9: Training Residual ML model...")
            await self._train_residual_model(current_time)
            steps_completed += 1
            _LOGGER.info("✓ Step 9/9 completed")
        except Exception as e:
            _LOGGER.error(f"✗ Step 9/9 failed: {e}", exc_info=True)
            errors.append(f"Residual training: {str(e)}")

        workflow_duration = asyncio.get_event_loop().time() - workflow_start

        if steps_completed == total_steps:
            _LOGGER.info(
                f"=== END_OF_DAY_WORKFLOW Completed Successfully ({steps_completed}/{total_steps} steps, {workflow_duration:.1f}s) ==="
            )

            try:
                if self.coordinator:
                    self.coordinator.update_system_status(
                        event_type="end_of_day_workflow",
                        event_status="success",
                        event_summary="Tagesabschluss erfolgreich abgeschlossen",
                        event_details={
                            "duration_seconds": round(workflow_duration, 1),
                            "steps_completed": f"{steps_completed}/{total_steps}",
                        },
                    )
            except Exception as e:
                _LOGGER.warning(f"Failed to update system status: {e}")
        else:
            _LOGGER.warning(
                f"=== END_OF_DAY_WORKFLOW Completed with Errors ({steps_completed}/{total_steps} steps, {workflow_duration:.1f}s) ==="
            )
            _LOGGER.warning(f"Errors: {'; '.join(errors)}")

            try:
                if self.coordinator:
                    self.coordinator.update_system_status(
                        event_type="end_of_day_workflow",
                        event_status="partial",
                        event_summary=f"Tagesabschluss teilweise erfolgreich ({steps_completed}/{total_steps})",
                        event_details={
                            "duration_seconds": round(workflow_duration, 1),
                            "steps_completed": f"{steps_completed}/{total_steps}",
                            "errors": errors,
                        },
                    )
            except Exception as e:
                _LOGGER.warning(f"Failed to update system status: {e}")

    async def _train_residual_model(self, now: datetime) -> None:
        """Train the Residual ML model for Physics+ML ensemble (Phase 4). @zara"""
        try:
            from ..ml.ml_residual_trainer import ResidualTrainer
            from ..ml.ml_data_loader_v3 import MLDataLoaderV3

            data_dir = self.data_manager.data_dir

            # Load training data
            data_loader = MLDataLoaderV3(data_dir, hass=self.hass)
            records, count = await data_loader.load_training_data(min_samples=20)

            if count < 20:
                _LOGGER.info(
                    f"Residual training: Not enough samples ({count} < 20). "
                    "Waiting for more production data."
                )
                return

            # Get system capacity
            solar_capacity = 2.0  # Default
            if self.coordinator:
                solar_capacity = getattr(self.coordinator, 'solar_capacity', 2.0) or 2.0

            # Initialize and train residual model
            residual_trainer = ResidualTrainer(
                data_dir=data_dir,
                system_capacity_kwp=solar_capacity,
            )

            success, accuracy, algo = await residual_trainer.train_residual_model(
                training_records=records,
                algorithm="auto",
            )

            if success:
                model_info = residual_trainer.get_model_info()
                _LOGGER.info(
                    f"✅ Residual model trained: accuracy={accuracy:.3f}, "
                    f"algorithm={algo}, samples={model_info['sample_count']}, "
                    f"residual_mean={model_info['residual_mean']:.4f} kWh"
                )
            else:
                _LOGGER.warning(f"Residual model training failed with algorithm {algo}")

        except ImportError as e:
            _LOGGER.debug(f"Residual trainer not available: {e}")
        except Exception as e:
            _LOGGER.error(f"Residual model training error: {e}", exc_info=True)
            raise

    async def _finalize_day_internal(self, now: datetime) -> None:
        """Internal: Finalize current day with actual values @zara"""
        current_time = now if now is not None else dt_util.now()

        try:

            actual_yield = 0.0
            if self.solar_yield_today_entity_id:
                yield_state = self.hass.states.get(self.solar_yield_today_entity_id)
                if yield_state and yield_state.state not in (None, "unknown", "unavailable"):
                    try:
                        actual_yield = float(yield_state.state)
                        _LOGGER.debug(
                            f"📊 Final Yield Sensor: {self.solar_yield_today_entity_id} = {actual_yield:.2f} kWh"
                        )
                    except (ValueError, TypeError) as e:
                        _LOGGER.warning(f"Invalid yield state: {yield_state.state} - {e}")
                else:
                    _LOGGER.warning(f"Yield sensor unavailable: {self.solar_yield_today_entity_id}")
            else:
                _LOGGER.warning("No solar_yield_today sensor configured")

            actual_consumption = None
            if self.coordinator.total_consumption_today:
                consumption_state = self.hass.states.get(self.coordinator.total_consumption_today)
                if consumption_state and consumption_state.state not in (
                    None,
                    "unknown",
                    "unavailable",
                ):
                    try:
                        actual_consumption = float(consumption_state.state)
                        _LOGGER.debug(f"Final consumption: {actual_consumption:.2f} kWh")
                    except (ValueError, TypeError) as e:
                        _LOGGER.warning(
                            f"Invalid consumption state: {consumption_state.state} - {e}"
                        )

            production_seconds = 0
            try:
                if (
                    hasattr(self.coordinator, "production_time_calculator")
                    and self.coordinator.production_time_calculator
                ):
                    production_hours = (
                        self.coordinator.production_time_calculator.get_production_hours()
                    )
                    if production_hours is not None:
                        production_seconds = int(production_hours * 3600)
                        _LOGGER.debug(
                            f"Production time: {production_seconds}s ({production_hours:.2f}h)"
                        )
            except Exception as e:
                _LOGGER.warning(f"Failed to get production time: {e}")

            try:
                await self._save_actual_best_hour()
            except Exception as e:
                _LOGGER.warning(f"Failed to save actual best hour: {e}")

            success = await self.data_manager.finalize_today(
                yield_kwh=actual_yield,
                consumption_kwh=actual_consumption,
                production_seconds=production_seconds,
            )

            if success:
                consumption_str = (
                    f"{actual_consumption:.2f}" if actual_consumption is not None else "N/A"
                )
                _LOGGER.info(
                    f"✓ Day finalized: Yield={actual_yield:.2f} kWh, "
                    f"Consumption={consumption_str} kWh"
                )

                try:
                    daily_forecasts = await self.data_manager.load_daily_forecasts()
                    if daily_forecasts:
                        statistics = daily_forecasts.get("statistics", {})
                        current_month = statistics.get("current_month", {})
                        if current_month and current_month.get("yield_kwh"):
                            self.coordinator.avg_month_yield = round(
                                current_month.get("yield_kwh", 0.0), 2
                            )
                            _LOGGER.info(
                                f"✓ Updated monthly average yield: {self.coordinator.avg_month_yield:.2f} kWh"
                            )
                except Exception as stats_error:
                    _LOGGER.warning(f"Failed to update monthly statistics: {stats_error}")
            else:
                _LOGGER.error("Failed to finalize day")

        except Exception as e:
            _LOGGER.error(f"Failed to finalize day: {e}", exc_info=True)

    async def _move_to_history_internal(self, now: datetime) -> None:
        """Internal: Move current day to history @zara"""
        try:
            success = await self.data_manager.move_to_history()

            if success:
                _LOGGER.info("✓ Moved to history")
            else:
                _LOGGER.error("Failed to move day to history")

        except Exception as e:
            _LOGGER.error(f"Failed to move to history: {e}", exc_info=True)

    async def _update_yesterday_deviation_internal(self, now: datetime) -> None:
        """Internal: Update yesterday deviation after moving to history @zara"""
        try:

            history = await self.data_manager.get_history(days=1)

            if not history or len(history) == 0:
                _LOGGER.warning("No history available to calculate yesterday deviation")
                self.coordinator.last_day_error_kwh = 0.0
                self.coordinator.yesterday_accuracy = 0.0
                return

            today_entry = history[0]

            forecast_kwh = today_entry.get("forecast_kwh", 0.0)
            actual_kwh = today_entry.get("actual_kwh", 0.0)

            if forecast_kwh is None or actual_kwh is None:
                _LOGGER.warning("History entry missing forecast_kwh or actual_kwh")
                self.coordinator.last_day_error_kwh = 0.0
                self.coordinator.yesterday_accuracy = 0.0
                return

            deviation = abs(forecast_kwh - actual_kwh)

            accuracy = 0.0
            if actual_kwh > 0.1:
                error_fraction = deviation / actual_kwh
                accuracy = max(0.0, 1.0 - error_fraction)
            elif forecast_kwh < 0.1 and actual_kwh < 0.1:
                accuracy = 1.0

            accuracy_pct = accuracy * 100.0

            self.coordinator.last_day_error_kwh = round(deviation, 2)
            self.coordinator.yesterday_accuracy = round(accuracy_pct, 1)
            self.coordinator.async_update_listeners()

            _LOGGER.info(
                f"✓ Yesterday Deviation updated:\n"
                f"  Date:       {today_entry.get('date', 'unknown')}\n"
                f"  Forecast:   {forecast_kwh:.2f} kWh\n"
                f"  Actual:     {actual_kwh:.2f} kWh\n"
                f"  Deviation:  {deviation:.2f} kWh\n"
                f"  Accuracy:   {accuracy_pct:.1f}%"
            )

        except Exception as e:
            _LOGGER.error(f"Failed to update yesterday deviation: {e}", exc_info=True)
            self.coordinator.last_day_error_kwh = 0.0
            self.coordinator.yesterday_accuracy = 0.0

    async def _calculate_stats_internal(self, now: datetime) -> None:
        """Internal: Calculate statistics @zara"""

        try:
            success = await self.data_manager.calculate_statistics()

            if success:
                _LOGGER.info("✓ Statistics calculated")

                try:
                    data = await self.data_manager.load_daily_forecasts()
                    stats_7d = data.get("statistics", {}).get("last_7_days", {})
                    if stats_7d:
                        _LOGGER.debug(
                            f"Last 7 days: Avg yield={stats_7d.get('avg_yield', 0):.2f} kWh, "
                            f"Avg accuracy={stats_7d.get('avg_accuracy', 0):.1f}%"
                        )
                except Exception:
                    pass
            else:
                _LOGGER.error("Failed to calculate statistics")

        except Exception as e:
            _LOGGER.error(f"Failed to calculate statistics: {e}", exc_info=True)

    async def _night_cleanup_internal(self, now: datetime) -> None:
        """Night cleanup internal - legacy method retained for compatibility. @zara"""
        _LOGGER.debug("Night cleanup completed (no action needed in current architecture)")

    @callback
    async def finalize_day_task(self, now: datetime) -> None:
        """Finalize day task - redirects to end_of_day_workflow. @zara"""
        _LOGGER.debug("finalize_day_task redirecting to _finalize_day_internal")
        await self._finalize_day_internal(now)

    @callback
    async def move_to_history_task(self, now: datetime) -> None:
        """Move to history task - redirects to end_of_day_workflow. @zara"""
        _LOGGER.debug("move_to_history_task redirecting to _move_to_history_internal")
        await self._move_to_history_internal(now)

    @callback
    async def calculate_stats_task(self, now: datetime) -> None:
        """Calculate stats task - redirects to end_of_day_workflow. @zara"""
        _LOGGER.debug("calculate_stats_task redirecting to _calculate_stats_internal")
        await self._calculate_stats_internal(now)

    async def _save_actual_best_hour(self) -> None:
        """Calculate and save the actual best production hour from todays hourly samples @zara"""
        try:
            today = dt_util.now().date()
            today_str = today.isoformat()

            today_samples = await self.data_manager.hourly_predictions.get_predictions_for_date(today_str)

            if not today_samples:
                _LOGGER.debug(
                    "No hourly samples from today - skipping actual best hour calculation"
                )
                return

            max_production = -1
            best_hour = None
            best_kwh = 0.0

            for sample in today_samples:
                production = sample.get("actual_kwh")
                if production is not None and production > max_production:
                    max_production = production
                    best_kwh = production

                    best_hour = sample.get("target_hour")

            if best_hour is not None and best_kwh > 0:

                success = await self.data_manager.save_actual_best_hour(
                    hour=best_hour, actual_kwh=best_kwh
                )

                if success:
                    _LOGGER.info(
                        f"✓ Actual best hour saved: {best_hour:02d}:00 with {best_kwh:.2f} kWh"
                    )
                else:
                    _LOGGER.warning("Failed to save actual best hour")
            else:
                _LOGGER.debug("No significant production today - actual best hour not saved")

        except Exception as e:
            _LOGGER.error(f"Error calculating actual best hour: {e}", exc_info=True)

    @callback
    async def create_morning_hourly_predictions(self, now: datetime) -> None:
        """MORNING TASK (6:00 AM): Create hourly predictions for today @zara"""
        current_time = now if now is not None else dt_util.now()

        _LOGGER.info(
            f"🌅 MORNING HOURLY PREDICTIONS TASK at {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        _LOGGER.info("=" * 80)
        _LOGGER.info("Creating hourly predictions for today (Single Source of Truth)")
        _LOGGER.info("=" * 80)

        try:
            today = current_time.date().isoformat()

            _LOGGER.info("Step 1/5: Fetching weather forecast...")
            weather_service = self.coordinator.weather_service
            if not weather_service:
                _LOGGER.error("✗ Weather service not available - ABORTING")
                return

            current_weather = await weather_service.get_current_weather()

            hourly_weather_forecast = await weather_service.get_corrected_hourly_forecast()

            if not hourly_weather_forecast:
                _LOGGER.error("✗ No corrected weather forecast available - ABORTING")
                return

            _LOGGER.info(f"✓ Corrected weather forecast retrieved: {len(hourly_weather_forecast)} hours")

            _LOGGER.info("Step 1b: Capturing morning forecast for weather precision tracking...")

            _LOGGER.info("✓ Weather operations managed by Pipeline Manager")

            _LOGGER.info("Step 2/5: Collecting sensor configuration...")

            sensor_config = {
                "temperature": self.coordinator.entry.data.get(CONF_TEMP_SENSOR) is not None,
                "humidity": self.coordinator.entry.data.get(CONF_HUMIDITY_SENSOR) is not None,
                "lux": self.coordinator.entry.data.get(CONF_LUX_SENSOR) is not None,
                "rain": self.coordinator.entry.data.get(CONF_RAIN_SENSOR) is not None,
                "wind_speed": self.coordinator.entry.data.get(CONF_WIND_SENSOR) is not None,
                "pressure": self.coordinator.entry.data.get(CONF_PRESSURE_SENSOR) is not None,
                "solar_radiation": self.coordinator.entry.data.get(CONF_SOLAR_RADIATION_SENSOR) is not None,
            }
            _LOGGER.info(
                f"✓ Sensor config: {sum(sensor_config.values())}/{len(sensor_config)} sensors available"
            )

            _LOGGER.info("Step 3/5: Generating ML forecast...")
            external_sensors = self.coordinator.sensor_collector.collect_all_sensor_data_dict()

            forecast = await self.coordinator.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_weather_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=self.coordinator.learned_correction_factor,
            )

            if not forecast or not forecast.get("hourly"):
                _LOGGER.error("✗ Forecast generation failed - no hourly data")
                return

            _LOGGER.info(
                f"✓ Forecast generated: {len(forecast.get('hourly', []))} hourly predictions"
            )
            _LOGGER.info(f"  Today total: {forecast.get('today', 0):.2f} kWh")
            _LOGGER.info(f"  Method: {forecast.get('method', 'unknown')}")

            _LOGGER.info("Step 4/5: Calculating astronomy data...")
            astronomy_data = await self._get_astronomy_data(current_time)
            _LOGGER.info(f"✓ Astronomy data calculated")

            _LOGGER.info("Step 5/5: Creating hourly predictions for TODAY only...")

            all_hourly = forecast.get("hourly", [])
            today_hourly = [h for h in all_hourly if h.get("date") == today]

            _LOGGER.info(
                f"Filtered hourly forecast: {len(all_hourly)} total → {len(today_hourly)} for today"
            )

            _LOGGER.info("🛡️  Using bulletproof morning routine handler (backup/restore + retry)")
            success = await self.morning_routine_handler.execute_morning_routine_with_retry(
                date=today,
                hourly_forecast=today_hourly,
                weather_hourly=hourly_weather_forecast,
                astronomy_data=astronomy_data,
                sensor_config=sensor_config,
            )

            if success:
                predictions = (
                    await self.coordinator.data_manager.hourly_predictions.get_predictions_for_date(
                        today
                    )
                )
                _LOGGER.info(f"\n✓✓✓ HOURLY PREDICTIONS CREATED SUCCESSFULLY ✓✓✓")
                _LOGGER.info(f"  Date: {today}")
                _LOGGER.info(f"  Total predictions: {len(predictions)}")
                _LOGGER.info(
                    f"  Total predicted: {sum(p['prediction_kwh'] for p in predictions):.2f} kWh"
                )

                _LOGGER.info(f"\n  Hourly breakdown:")
                for p in predictions:
                    _LOGGER.info(
                        f"    {p['target_hour']:02d}:00 - "
                        f"{p['prediction_kwh']:.3f} kWh - "
                        f"Conf: {p['confidence']:.0f}% - "
                        f"{'⭐ PEAK' if p['flags']['is_peak_hour'] else ''}"
                    )

                _LOGGER.info(f"\n  Saving forecasts to daily_forecasts.json...")

                today_kwh = forecast.get('today')
                today_raw = forecast.get('today_raw')
                safeguard_applied = forecast.get('safeguard_applied', False)

                await self.coordinator.data_manager.save_forecast_day(
                    prediction_kwh=today_kwh,
                    source=forecast.get('method', 'morning_routine'),
                    lock=True,
                    force_overwrite=True,
                    prediction_kwh_raw=today_raw,
                    safeguard_applied=safeguard_applied,
                )
                _LOGGER.info(f"  ✓ Today forecast saved: {today_kwh:.2f} kWh [LOCKED]")

                tomorrow_kwh = forecast.get('tomorrow')
                if tomorrow_kwh is not None:
                    tomorrow_date = current_time + timedelta(days=1)
                    await self.coordinator.data_manager.save_forecast_tomorrow(
                        date=tomorrow_date,
                        prediction_kwh=tomorrow_kwh,
                        source=forecast.get('method', 'morning_routine'),
                        lock=False,
                    )
                    _LOGGER.info(f"  ✓ Tomorrow forecast saved: {tomorrow_kwh:.2f} kWh [unlocked]")

                day_after_kwh = forecast.get('day_after_tomorrow')
                if day_after_kwh is not None:
                    day_after_date = current_time + timedelta(days=2)
                    await self.coordinator.data_manager.save_forecast_day_after(
                        date=day_after_date,
                        prediction_kwh=day_after_kwh,
                        source=forecast.get('method', 'morning_routine'),
                        lock=False,
                    )
                    _LOGGER.info(f"  ✓ Day after tomorrow forecast saved: {day_after_kwh:.2f} kWh [unlocked]")

                try:
                    hourly_data = (
                        await self.coordinator.data_manager.hourly_predictions._read_json_async()
                    )
                    self.coordinator._hourly_predictions_cache = hourly_data
                except Exception as cache_err:
                    _LOGGER.debug(f"Failed to update hourly predictions cache: {cache_err}")

                self.coordinator.async_update_listeners()
                _LOGGER.info(f"  ✓ Sensors notified to reload from hourly_predictions.json")

            else:
                _LOGGER.error("✗ Failed to create hourly predictions")

            _LOGGER.info("=" * 80)

        except Exception as e:
            _LOGGER.error(f"✗ Error in create_morning_hourly_predictions: {e}", exc_info=True)

    async def _get_astronomy_data(self, dt: datetime) -> Dict[str, Any]:
        """Get astronomy data for the day from astronomy cache @zara"""
        try:
            target_date = dt.date()
            date_str = target_date.isoformat()
            astronomy_cache_file = (
                self.coordinator.data_manager.data_dir / "stats" / "astronomy_cache.json"
            )

            day_data = None

            if astronomy_cache_file.exists():
                def _read_sync():
                    import json
                    with open(astronomy_cache_file, "r") as f:
                        return json.load(f)

                import asyncio
                loop = asyncio.get_running_loop()
                cache = await loop.run_in_executor(None, _read_sync)

                day_data = cache.get("days", {}).get(date_str)

            if not day_data:
                _LOGGER.warning(
                    f"No astronomy data for {date_str} in cache - calculating on-the-fly (Self-Healing)"
                )
                day_data = await self._calculate_astronomy_fallback(target_date)

                if not day_data:
                    _LOGGER.error(f"Failed to calculate astronomy fallback for {date_str}")
                    return {}

            return {
                "sunrise": day_data.get("sunrise_local"),
                "sunset": day_data.get("sunset_local"),
                "solar_noon": day_data.get("solar_noon_local"),
                "daylight_hours": day_data.get("daylight_hours"),
                "hourly": day_data.get(
                    "hourly", {}
                ),
            }

        except Exception as e:
            _LOGGER.error(f"Failed to get astronomy data: {e}", exc_info=True)
            return {}

    async def _calculate_astronomy_fallback(self, target_date: date) -> Optional[Dict[str, Any]]:
        """Calculate astronomy data on-the-fly when cache is unavailable (Self-Healing) @zara"""
        try:

            astronomy_cache = None
            if (
                self.coordinator.weather_pipeline_manager
                and self.coordinator.weather_pipeline_manager.astronomy_cache
            ):
                astronomy_cache = self.coordinator.weather_pipeline_manager.astronomy_cache

            if not astronomy_cache:
                _LOGGER.error("AstronomyCache not available in pipeline manager for fallback")
                return None

            if not all([
                astronomy_cache.latitude,
                astronomy_cache.longitude,
                astronomy_cache.timezone
            ]):
                _LOGGER.error("AstronomyCache location not initialized for fallback")
                return None

            system_capacity_kwp = self.coordinator.solar_capacity or 5.0

            day_data = await astronomy_cache.build_cache_for_date(
                target_date,
                system_capacity_kwp
            )

            if day_data:
                _LOGGER.info(
                    f"✓ Astronomy fallback calculated for {target_date}: "
                    f"sunrise={day_data.get('sunrise_local')}, "
                    f"sunset={day_data.get('sunset_local')}"
                )

            return day_data

        except Exception as e:
            _LOGGER.error(f"Astronomy fallback calculation failed: {e}", exc_info=True)
            return None

    async def _get_production_window_from_cache(self, target_date: date) -> Optional[tuple]:
        """Get production window from astronomy cache @zara"""
        try:

            astronomy_cache_file = (
                self.coordinator.data_manager.data_dir / "stats" / "astronomy_cache.json"
            )

            if not astronomy_cache_file.exists():
                return None

            def _read_sync():
                import json

                with open(astronomy_cache_file, "r") as f:
                    return json.load(f)

            import asyncio

            loop = asyncio.get_running_loop()
            cache = await loop.run_in_executor(None, _read_sync)

            date_str = target_date.isoformat()
            day_data = cache.get("days", {}).get(date_str)

            if not day_data:
                return None

            window_start_str = day_data.get("production_window_start")
            window_end_str = day_data.get("production_window_end")

            if not window_start_str or not window_end_str:
                return None

            window_start = datetime.fromisoformat(window_start_str).replace(tzinfo=None)
            window_end = datetime.fromisoformat(window_end_str).replace(tzinfo=None)

            _LOGGER.debug(
                f"Production window from cache: {window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}"
            )

            return (window_start, window_end)

        except Exception as e:
            _LOGGER.debug(f"Failed to get production window from cache: {e}")
            return None

    @callback
    async def update_hourly_actuals(self, now: datetime) -> None:
        """HOURLY TASK (every hour at :05): Update actual values for previous hour @zara"""
        current_time = now if now is not None else dt_util.now()
        _LOGGER.debug(f"🔔 update_hourly_actuals() called at {current_time.strftime('%H:%M:%S')}")
        _LOGGER.debug(f"   current_time.tzinfo = {current_time.tzinfo}")

        production_window = await self._get_production_window_from_cache(current_time.date())

        if not production_window:

            _LOGGER.info(
                "ℹ️ FRESH INSTALLATION: Astronomy cache not yet available. "
                "This is normal after a new installation or Home Assistant restart. "
                "The system will automatically populate the cache at the next scheduled update "
                "(00:05 or 04:15). If this message persists after 24 hours, please restart Home Assistant."
            )
            return

        production_start, production_end = production_window
        current_time_naive = current_time.replace(tzinfo=None)

        if not (production_start <= current_time_naive <= production_end):
            _LOGGER.debug(
                f"⏰ Skipping hourly update at {current_time.strftime('%H:%M:%S')} - "
                f"Outside production window ({production_start.strftime('%H:%M')} - {production_end.strftime('%H:%M')})"
            )
            return
        else:
            _LOGGER.debug(
                f"⏰ Inside production window: {current_time.strftime('%H:%M:%S')} is between "
                f"{production_start.strftime('%H:%M')} and {production_end.strftime('%H:%M')}"
            )

        previous_hour_dt = current_time - timedelta(hours=1)
        previous_hour = previous_hour_dt.hour
        today = current_time.date().isoformat()

        _LOGGER.info(
            f"⏰ HOURLY UPDATE at {current_time.strftime('%H:%M:%S')} - Updating actual for {previous_hour:02d}:00-{current_time.hour:02d}:00"
        )

        try:

            current_yield = None
            if self.solar_yield_today_entity_id:
                yield_state = self.hass.states.get(self.solar_yield_today_entity_id)
                if yield_state and yield_state.state not in (None, "unknown", "unavailable"):
                    try:
                        current_yield = float(yield_state.state)
                        _LOGGER.debug(
                            f"📊 Yield Sensor Read: {self.solar_yield_today_entity_id} = {current_yield:.3f} kWh"
                        )
                    except (ValueError, TypeError) as e:
                        _LOGGER.warning(f"Invalid yield sensor value: {yield_state.state} - {e}")

            if current_yield is None:
                _LOGGER.warning(
                    f"Cannot update hourly actual: yield sensor not available or invalid ({self.solar_yield_today_entity_id})"
                )
                return

            if not hasattr(self.coordinator, "_last_yield_cache"):
                self.coordinator._last_yield_cache = {}

            previous_yield = self.coordinator._last_yield_cache.get("value", None)
            previous_yield_time = self.coordinator._last_yield_cache.get("time", None)

            self.coordinator._last_yield_cache = {"value": current_yield, "time": current_time}

            if previous_yield is None:
                _LOGGER.info(
                    f"First hourly update - caching yield: {current_yield:.3f} kWh (no actual update yet)"
                )
                return

            if current_yield >= previous_yield:
                actual_kwh = current_yield - previous_yield
            else:

                _LOGGER.warning(
                    f"Yield sensor decreased ({previous_yield:.3f} → {current_yield:.3f} kWh) - likely reset, skipping hour"
                )
                return

            _LOGGER.info(
                f"Hourly production {previous_hour:02d}:00-{current_time.hour:02d}:00: {actual_kwh:.3f} kWh (Yield: {previous_yield:.3f} → {current_yield:.3f})"
            )

            sensor_data_raw = self.coordinator.sensor_collector.collect_all_sensor_data_dict()
            sensor_data = {
                "temperature_c": sensor_data_raw.get("temperature"),
                "humidity_percent": sensor_data_raw.get("humidity"),
                "lux": sensor_data_raw.get("lux"),
                "rain_mm": sensor_data_raw.get("rain"),
                "uv_index": sensor_data_raw.get("uv_index"),
                "wind_speed_ms": sensor_data_raw.get("wind_speed"),
                "current_yield_kwh": current_yield,
            }

            _LOGGER.debug(
                f"☁️ Weather Hour {previous_hour:02d}: Temp={sensor_data.get('temperature_c')}°C, "
                f"Humidity={sensor_data.get('humidity_percent')}%, Lux={sensor_data.get('lux')}W/m², "
                f"Rain={sensor_data.get('rain_mm')}mm, Wind={sensor_data.get('wind_speed_ms')}m/s"
            )

            weather_actual = None
            try:
                hourly_actual_file = self.coordinator.data_manager.data_dir / "stats" / "hourly_weather_actual.json"
                if hourly_actual_file.exists():
                    def _read_hourly_actual():
                        import json
                        with open(hourly_actual_file, "r") as f:
                            return json.load(f)

                    hourly_actual_data = await self.hass.async_add_executor_job(_read_hourly_actual)
                    hour_data = hourly_actual_data.get("hourly_data", {}).get(today, {}).get(str(previous_hour))

                    if hour_data:
                        weather_actual = {
                            "temperature_c": hour_data.get("temperature_c"),
                            "cloud_cover_percent": hour_data.get("cloud_cover_percent"),
                            "humidity_percent": hour_data.get("humidity_percent"),
                            "wind_speed_ms": hour_data.get("wind_speed_ms"),
                            "precipitation_mm": hour_data.get("precipitation_mm"),
                            "pressure_hpa": hour_data.get("pressure_hpa"),
                            "solar_radiation_wm2": hour_data.get("solar_radiation_wm2"),
                            "lux": hour_data.get("lux"),

                            "frost_detected": hour_data.get("frost_detected"),
                            "frost_score": hour_data.get("frost_score"),
                            "frost_confidence": hour_data.get("frost_confidence"),
                        }
                        _LOGGER.debug(f"Loaded weather_actual from hourly tracking (frost: {hour_data.get('frost_detected')})")
                    else:
                        _LOGGER.debug(f"No hourly weather data for {today} hour {previous_hour}")
            except Exception as e:
                _LOGGER.warning(f"Could not load hourly weather actual: {e}")

            if weather_actual is None and self.coordinator.weather_service:
                current_weather = await self.coordinator.weather_service.get_current_weather()
                if current_weather:
                    weather_actual = {
                        "temperature_c": current_weather.get("temperature"),
                        "cloud_cover_percent": current_weather.get("cloud_cover"),
                        "humidity_percent": current_weather.get("humidity"),
                        "wind_speed_ms": current_weather.get("wind_speed"),
                        "precipitation_mm": current_weather.get("precipitation"),
                        "pressure_hpa": current_weather.get("pressure"),
                    }
                    _LOGGER.debug("Using weather service (no frost data available)")

            astronomy_update = await self._calculate_astronomy_for_hour(current_time)
            success = await self.coordinator.data_manager.hourly_predictions.update_hourly_actual(
                date=today,
                hour=previous_hour,
                actual_kwh=actual_kwh,
                sensor_data=sensor_data,
                weather_actual=weather_actual,
                astronomy_update=astronomy_update,
            )

            if success:
                _LOGGER.info(f"✓ Updated actual for hour {previous_hour:02d}: {actual_kwh:.3f} kWh")

                try:
                    hourly_data = (
                        await self.coordinator.data_manager.hourly_predictions._read_json_async()
                    )
                    self.coordinator._hourly_predictions_cache = hourly_data
                except Exception as cache_err:
                    _LOGGER.debug(f"Failed to update hourly predictions cache: {cache_err}")

                self.coordinator.async_update_listeners()
            else:
                _LOGGER.warning(f"✗ Failed to update actual for hour {previous_hour:02d}")

        except Exception as e:
            _LOGGER.error(f"Error in update_hourly_actuals: {e}", exc_info=True)

    async def _calculate_astronomy_for_hour(self, dt: datetime) -> Dict[str, Any]:
        """Calculate astronomy features for ML training from astronomy_cache @zara"""
        try:

            from ..astronomy.astronomy_cache_manager import get_cache_manager

            cache_manager = get_cache_manager()
            if cache_manager:
                date_str = dt.date().isoformat()
                hour = dt.hour

                day_data = cache_manager.get_day_data(date_str)
                if day_data and "hourly" in day_data:
                    hourly_data = day_data["hourly"].get(str(hour), {})

                    if hourly_data:

                        sunrise_str = day_data.get("sunrise_local")
                        sunset_str = day_data.get("sunset_local")

                        hours_after_sunrise = None
                        hours_before_sunset = None

                        if sunrise_str and sunset_str:
                            try:
                                sunrise = datetime.fromisoformat(sunrise_str)
                                sunset = datetime.fromisoformat(sunset_str)
                                hours_after_sunrise = (dt - sunrise).total_seconds() / 3600.0
                                hours_before_sunset = (sunset - dt).total_seconds() / 3600.0
                            except:
                                pass

                        return {
                            "sun_elevation_deg": hourly_data.get("elevation_deg"),
                            "sun_azimuth_deg": hourly_data.get("azimuth_deg"),
                            "clear_sky_radiation_wm2": hourly_data.get(
                                "clear_sky_solar_radiation_wm2"
                            ),
                            "theoretical_max_kwh": hourly_data.get("theoretical_max_pv_kwh"),
                            "hours_since_solar_noon": hourly_data.get("hours_since_solar_noon"),
                            "day_progress_ratio": hourly_data.get("day_progress_ratio"),
                            "hours_after_sunrise": (
                                round(hours_after_sunrise, 2)
                                if hours_after_sunrise is not None
                                else None
                            ),
                            "hours_before_sunset": (
                                round(hours_before_sunset, 2)
                                if hours_before_sunset is not None
                                else None
                            ),
                        }

            _LOGGER.debug(
                f"ℹ️ Astronomy cache not yet available for {dt}. "
                "Cache will be populated at next morning routine (00:05 or 04:15)."
            )
            return {}

        except Exception as e:
            _LOGGER.debug(f"Failed to calculate astronomy for hour: {e}")
            return {}

    async def _get_battery_peak_soc_today(
        self,
        date_str: str,
        start_hour: int,
        end_hour: int
    ) -> float:
        """
        Get peak battery SOC during production hours to detect MPPT throttling.

        Args:
            date_str: Date in YYYY-MM-DD format
            start_hour: Production window start hour
            end_hour: Production window end hour

        Returns:
            Peak battery SOC (0-100%) or None if battery not configured/available
        """
        try:
            from ..const import CONF_BATTERY_SOC_SENSOR, CONF_BATTERY_ENABLED

            battery_enabled = self.coordinator.config_entry.options.get(
                CONF_BATTERY_ENABLED,
                self.coordinator.config_entry.data.get(CONF_BATTERY_ENABLED, False)
            )

            if not battery_enabled:
                _LOGGER.debug("Battery not enabled - skipping SOC check")
                return None

            battery_soc_sensor = self.coordinator.config_entry.options.get(
                CONF_BATTERY_SOC_SENSOR,
                self.coordinator.config_entry.data.get(CONF_BATTERY_SOC_SENSOR)
            )

            if not battery_soc_sensor:
                _LOGGER.debug("Battery SOC sensor not configured")
                return None

            from homeassistant.components.recorder import history
            from datetime import datetime, timedelta

            date_obj = datetime.fromisoformat(date_str)
            start_time = date_obj.replace(hour=start_hour, minute=0, second=0, microsecond=0)
            end_time = date_obj.replace(hour=end_hour, minute=59, second=59, microsecond=999999)

            history_data = await self.hass.async_add_executor_job(
                history.get_significant_states,
                self.hass,
                start_time,
                end_time,
                [battery_soc_sensor],
                None,
                True,
                True,
            )

            if not history_data or battery_soc_sensor not in history_data:
                _LOGGER.debug(f"No battery SOC history found for {date_str}")
                return None

            states = history_data[battery_soc_sensor]
            peak_soc = 0.0

            for state in states:
                try:
                    soc_value = float(state.state)
                    if soc_value > peak_soc:
                        peak_soc = soc_value
                except (ValueError, AttributeError):
                    continue

            if peak_soc > 0:
                _LOGGER.debug(
                    f"Battery peak SOC during production ({start_hour}-{end_hour}h): {peak_soc:.1f}%"
                )
                return peak_soc
            else:
                _LOGGER.debug("No valid battery SOC values found in history")
                return None

        except Exception as e:
            _LOGGER.warning(f"Failed to get battery peak SOC: {e}")
            return None
