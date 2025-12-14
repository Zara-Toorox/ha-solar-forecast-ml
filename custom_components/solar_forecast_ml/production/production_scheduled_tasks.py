"""Scheduled Task Management for Production Tracking V10.3.0 @zara

Dynamic scheduling based on sunrise time for optimal forecast accuracy.

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
import json
import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_change, async_track_point_in_time

from .production_morning_routine import MorningRoutineHandler

from ..services.service_system_report import SystemReportGenerator
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

# Minimum time for final forecast (ensures fresh weather data from 06:00 UTC model run)
MINIMUM_FINAL_FORECAST_TIME = time(5, 0)


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
        self._dynamic_listeners = []  # For sunrise-based dynamic tasks

        self.morning_routine_handler = MorningRoutineHandler(data_manager, coordinator)

        solar_rad_sensor = coordinator.entry.data.get(CONF_SOLAR_RADIATION_SENSOR)
        lux_sensor = coordinator.entry.data.get(CONF_LUX_SENSOR)
        self.sensor_cloud_correction = SensorCloudCorrection(
            hass=hass,
            data_manager=data_manager,
            solar_radiation_sensor=solar_rad_sensor,
            lux_sensor=lux_sensor,
            solar_yield_entity=solar_yield_today_entity_id,
        )

        self._end_of_day_running = False

        self._pattern_learner = None
        self._weather_corrector = None

        # Persistent yield cache file path
        self._yield_cache_file = data_manager.data_dir / "stats" / "yield_cache.json"

        _LOGGER.debug("ScheduledTasksManager initialized")

    def _get_pattern_learner(self):
        """Get or create PatternLearner instance (lazy initialization) @zara"""
        if self._pattern_learner is None:
            from ..ml.ml_pattern_learner import PatternLearner
            self._pattern_learner = PatternLearner(self.data_manager.data_dir)
        return self._pattern_learner

    def _get_weather_corrector(self):
        """Get or create WeatherForecastCorrector instance (lazy initialization) @zara"""
        if self._weather_corrector is None:
            from ..data.data_weather_corrector import WeatherForecastCorrector
            self._weather_corrector = WeatherForecastCorrector(
                self.hass,
                self.data_manager
            )
        return self._weather_corrector

    async def _load_yield_cache(self) -> Dict[str, Any]:
        """Load persistent yield cache from file @zara"""
        try:
            if not self._yield_cache_file.exists():
                return {}

            def _read():
                with open(self._yield_cache_file, "r") as f:
                    return json.load(f)

            return await self.hass.async_add_executor_job(_read)
        except Exception as e:
            _LOGGER.warning(f"Failed to load yield cache: {e}")
            return {}

    async def _save_yield_cache(self, cache_data: Dict[str, Any]) -> None:
        """Save yield cache to persistent file @zara"""
        try:
            def _write():
                with open(self._yield_cache_file, "w") as f:
                    json.dump(cache_data, f, indent=2, default=str)

            await self.hass.async_add_executor_job(_write)
        except Exception as e:
            _LOGGER.warning(f"Failed to save yield cache: {e}")

    async def recover_missing_hourly_actuals(self) -> int:
        """Recover missing hourly actuals using hourly yield deltas from sensor history @zara

        This method reconstructs missing actual_kwh values by:
        1. Reading the current cumulative yield from sensor
        2. Finding the last known yield value from hourly_predictions
        3. Distributing the delta across missing hours proportionally

        Returns:
            Number of hours recovered
        """
        current_time = dt_util.now()
        today = current_time.date().isoformat()

        try:
            # Get current yield from sensor
            current_yield = None
            if self.solar_yield_today_entity_id:
                yield_state = self.hass.states.get(self.solar_yield_today_entity_id)
                if yield_state and yield_state.state not in (None, "unknown", "unavailable"):
                    try:
                        current_yield = float(yield_state.state)
                    except (ValueError, TypeError):
                        pass

            if current_yield is None:
                _LOGGER.warning("Cannot recover hourly actuals: yield sensor unavailable")
                return 0

            # Get today's predictions
            predictions = await self.data_manager.hourly_predictions.get_predictions_for_date(today)
            if not predictions:
                _LOGGER.warning("Cannot recover hourly actuals: no predictions for today")
                return 0

            # Get production window
            production_window = await self._get_production_window_from_cache(current_time.date())
            if not production_window:
                _LOGGER.warning("Cannot recover hourly actuals: no production window")
                return 0

            production_start, production_end = production_window
            start_hour = production_start.hour
            end_hour = min(current_time.hour, production_end.hour)

            # Find missing hours and last known yield
            missing_hours = []
            last_known_yield = 0.0
            last_known_hour = None

            for pred in sorted(predictions, key=lambda x: x.get("target_hour", 0)):
                hour = pred.get("target_hour")
                if hour is None or hour < start_hour or hour > end_hour:
                    continue

                actual = pred.get("actual_kwh")
                sensor_yield = pred.get("sensor_actual", {}).get("current_yield_kwh")

                if actual is not None and sensor_yield is not None:
                    last_known_yield = sensor_yield
                    last_known_hour = hour
                elif hour < current_time.hour:  # Only recover past hours
                    missing_hours.append(hour)

            if not missing_hours:
                _LOGGER.debug("No missing hours to recover")
                return 0

            # Calculate total delta to distribute
            total_delta = current_yield - last_known_yield
            if total_delta <= 0:
                _LOGGER.debug(f"No production delta to distribute (current={current_yield}, last_known={last_known_yield})")
                return 0

            # Get theoretical max for each missing hour to weight distribution
            hour_weights = {}
            total_weight = 0.0

            for hour in missing_hours:
                pred = next((p for p in predictions if p.get("target_hour") == hour), None)
                if pred:
                    # Use prediction or theoretical max as weight
                    weight = pred.get("prediction_kwh", 0) or pred.get("astronomy", {}).get("theoretical_max_kwh", 0) or 1.0
                    hour_weights[hour] = max(weight, 0.01)  # Minimum weight to avoid division by zero
                    total_weight += hour_weights[hour]

            if total_weight == 0:
                total_weight = len(missing_hours)
                hour_weights = {h: 1.0 for h in missing_hours}

            # Distribute delta proportionally and update each hour
            recovered_count = 0
            cumulative_yield = last_known_yield

            _LOGGER.info(
                f"Recovering {len(missing_hours)} missing hours: {missing_hours}\n"
                f"  Total delta: {total_delta:.3f} kWh (current={current_yield:.3f}, last_known={last_known_yield:.3f})"
            )

            for hour in sorted(missing_hours):
                weight = hour_weights.get(hour, 1.0)
                hour_kwh = round((weight / total_weight) * total_delta, 3)

                cumulative_yield += hour_kwh

                # Collect sensor data (may be empty for past hours)
                sensor_data = {
                    "temperature_c": None,
                    "humidity_percent": None,
                    "lux": None,
                    "rain_mm": None,
                    "uv_index": None,
                    "wind_speed_ms": None,
                    "current_yield_kwh": round(cumulative_yield, 3),
                }

                # Get weather actual if available
                weather_actual = None
                try:
                    hourly_actual_file = self.data_manager.data_dir / "stats" / "hourly_weather_actual.json"
                    if hourly_actual_file.exists():
                        def _read_weather():
                            with open(hourly_actual_file, "r") as f:
                                return json.load(f)

                        hourly_actual_data = await self.hass.async_add_executor_job(_read_weather)
                        hour_data = hourly_actual_data.get("hourly_data", {}).get(today, {}).get(str(hour))
                        if hour_data:
                            weather_actual = {
                                "temperature_c": hour_data.get("temperature_c"),
                                "cloud_cover_percent": hour_data.get("cloud_cover_percent"),
                                "humidity_percent": hour_data.get("humidity_percent"),
                                "wind_speed_ms": hour_data.get("wind_speed_ms"),
                                "precipitation_mm": hour_data.get("precipitation_mm"),
                                "pressure_hpa": hour_data.get("pressure_hpa"),
                                "solar_radiation_wm2": hour_data.get("solar_radiation_wm2"),
                            }
                except Exception:
                    pass

                # Update the hour
                success = await self.data_manager.hourly_predictions.update_hourly_actual(
                    date=today,
                    hour=hour,
                    actual_kwh=hour_kwh,
                    sensor_data=sensor_data,
                    weather_actual=weather_actual,
                    astronomy_update=None,
                )

                if success:
                    recovered_count += 1
                    _LOGGER.debug(f"  Recovered hour {hour:02d}: {hour_kwh:.3f} kWh")

            # Update yield cache with current value
            await self._save_yield_cache({
                "value": current_yield,
                "time": current_time.isoformat(),
                "date": today,
            })

            # Update coordinator cache
            if recovered_count > 0:
                try:
                    hourly_data = await self.data_manager.hourly_predictions._read_json_async()
                    self.coordinator._hourly_predictions_cache = hourly_data
                except Exception:
                    pass

                self.coordinator.async_update_listeners()

            _LOGGER.info(f"Recovered {recovered_count}/{len(missing_hours)} missing hourly actuals")
            return recovered_count

        except Exception as e:
            _LOGGER.error(f"Error recovering missing hourly actuals: {e}", exc_info=True)
            return 0

    async def _get_sunrise_for_today(self) -> Optional[datetime]:
        """Get sunrise time for today from astronomy cache @zara"""
        try:
            today = dt_util.now().date()
            astronomy_cache_file = self.data_manager.data_dir / "stats" / "astronomy_cache.json"

            if not astronomy_cache_file.exists():
                _LOGGER.warning("Astronomy cache not found for dynamic scheduling")
                return None

            def _read_sync():
                with open(astronomy_cache_file, "r") as f:
                    return json.load(f)

            cache = await self.hass.async_add_executor_job(_read_sync)
            day_data = cache.get("days", {}).get(today.isoformat())

            if not day_data:
                _LOGGER.warning(f"No astronomy data for {today}")
                return None

            sunrise_str = day_data.get("sunrise_local")
            if not sunrise_str:
                return None

            sunrise = datetime.fromisoformat(sunrise_str)

            # DEFENSIVE: Ensure timezone-aware datetime to avoid comparison errors
            # Old cache files from StartupInitializer may lack timezone info
            if sunrise.tzinfo is None:
                local_tz = dt_util.now().tzinfo
                if local_tz:
                    sunrise = sunrise.replace(tzinfo=local_tz)
                    _LOGGER.debug(f"Added timezone {local_tz} to naive sunrise datetime")

            return sunrise

        except Exception as e:
            _LOGGER.error(f"Error getting sunrise: {e}")
            return None

    def _calculate_dynamic_forecast_time(self, sunrise: datetime) -> datetime:
        """Calculate the optimal time for final forecast based on sunrise @zara

        Formula: max(05:00, sunrise - 30min)

        This ensures:
        - Fresh weather data (06:00 UTC model run available after ~05:00 local)
        - Forecast is ready before production starts
        """
        forecast_time = sunrise - timedelta(minutes=30)

        # Ensure minimum time of 05:00
        min_time_today = datetime.combine(sunrise.date(), MINIMUM_FINAL_FORECAST_TIME)
        if sunrise.tzinfo:
            min_time_today = min_time_today.replace(tzinfo=sunrise.tzinfo)

        if forecast_time < min_time_today:
            forecast_time = min_time_today

        return forecast_time

    def _calculate_cloud_check_time(self, sunrise: datetime) -> datetime:
        """Calculate cloud check time: sunrise + 4 hours @zara"""
        return sunrise + timedelta(hours=4)

    async def _schedule_dynamic_tasks_for_today(self) -> None:
        """Schedule dynamic tasks based on today's sunrise @zara"""
        # Cancel any existing dynamic listeners
        for remove_listener in self._dynamic_listeners:
            try:
                remove_listener()
            except Exception:
                pass
        self._dynamic_listeners = []

        sunrise = await self._get_sunrise_for_today()
        if not sunrise:
            _LOGGER.warning("Could not get sunrise - using fallback times (05:30 forecast, 12:00 cloud check)")
            # Fallback to reasonable defaults
            now = dt_util.now()
            sunrise = datetime.combine(now.date(), time(8, 0))
            if now.tzinfo:
                sunrise = sunrise.replace(tzinfo=now.tzinfo)

        # Calculate dynamic times
        final_forecast_time = self._calculate_dynamic_forecast_time(sunrise)
        cloud_check_time = self._calculate_cloud_check_time(sunrise)

        now = dt_util.now()

        # Schedule final forecast if not yet passed
        if final_forecast_time > now:
            remove_final = async_track_point_in_time(
                self.hass,
                self._final_morning_forecast,
                final_forecast_time
            )
            self._dynamic_listeners.append(remove_final)
            _LOGGER.info(
                f"🌅 Dynamic scheduling: Final forecast at {final_forecast_time.strftime('%H:%M')} "
                f"(sunrise {sunrise.strftime('%H:%M')} - 30min)"
            )
        else:
            _LOGGER.debug(f"Final forecast time {final_forecast_time.strftime('%H:%M')} already passed")

        # Schedule cloud check if not yet passed
        if cloud_check_time > now:
            remove_cloud = async_track_point_in_time(
                self.hass,
                self._cloud_discrepancy_logging,
                cloud_check_time
            )
            self._dynamic_listeners.append(remove_cloud)
            _LOGGER.info(
                f"☁️ Dynamic scheduling: Cloud check at {cloud_check_time.strftime('%H:%M')} "
                f"(sunrise + 4h, logging only)"
            )
        else:
            _LOGGER.debug(f"Cloud check time {cloud_check_time.strftime('%H:%M')} already passed")

    async def _final_morning_forecast(self, now: datetime) -> None:
        """Final morning forecast with fresh weather data @zara

        This runs at max(05:00, sunrise - 30min) and:
        1. Fetches fresh weather data (Multi-Weather Blending)
        2. Creates the final locked forecast for today
        """
        current_time = now if now is not None else dt_util.now()

        _LOGGER.info(f"🌅 Final morning forecast triggered at {current_time.strftime('%H:%M')}")

        try:
            # Step 1: Update weather cache with fresh data
            if hasattr(self.coordinator, 'weather_pipeline_manager'):
                pipeline = self.coordinator.weather_pipeline_manager
                if hasattr(pipeline, 'multi_weather_blender') and pipeline.multi_weather_blender:
                    blender = pipeline.multi_weather_blender
                    blend_success = await blender.update_and_save_cache()
                    if blend_success:
                        stats = blender.get_blend_stats()
                        _LOGGER.info(
                            f"📡 Fresh weather data fetched: "
                            f"weights=OM:{stats['weights'].get('open_meteo', 0.5):.0%}/"
                            f"WWO:{stats['weights'].get('wwo', 0.5):.0%}"
                        )
                    else:
                        _LOGGER.warning("Weather update failed, using cached data")

            # Step 2: Run the complete morning routine
            await self.morning_routine_complete(now)

        except Exception as e:
            _LOGGER.error(f"Final morning forecast failed: {e}", exc_info=True)

    async def _cloud_discrepancy_logging(self, now: datetime) -> None:
        """Log cloud discrepancy for weight learning (no correction) @zara

        This runs at sunrise + 4h and:
        1. Compares sensor cloud cover with forecast
        2. Logs discrepancy for end-of-day weight learning
        3. Does NOT apply any correction
        """
        current_time = now if now is not None else dt_util.now()

        _LOGGER.info(f"☁️ Cloud discrepancy check at {current_time.strftime('%H:%M')} (logging only)")

        try:
            if not self.sensor_cloud_correction.cloud_sensors_enabled:
                _LOGGER.debug("Cloud sensors not configured - skipping check")
                return

            # Get sensor cloud cover
            sensor_cloud = await self.sensor_cloud_correction._get_sensor_cloud_estimate(datetime.now())
            if sensor_cloud is None:
                _LOGGER.debug("Could not read sensor cloud cover")
                return

            # Get forecast cloud cover (average for daylight hours)
            forecast_cloud = await self.sensor_cloud_correction._get_forecast_cloud_average(current_time)
            if forecast_cloud is None:
                _LOGGER.debug("Could not get forecast cloud cover")
                return

            discrepancy = sensor_cloud - forecast_cloud

            # Log for weight learning
            log_entry = {
                "timestamp": current_time.isoformat(),
                "sensor_cloud_percent": round(sensor_cloud, 1),
                "forecast_cloud_percent": round(forecast_cloud, 1),
                "discrepancy_percent": round(discrepancy, 1),
                "action": "logged_for_learning"
            }

            # Save to drift log for end-of-day analysis
            drift_log_file = self.data_manager.data_dir / "stats" / "forecast_drift_log.json"
            try:
                def _update_log():
                    existing = []
                    if drift_log_file.exists():
                        with open(drift_log_file, "r") as f:
                            data = json.load(f)
                            existing = data.get("entries", [])

                    existing.append(log_entry)
                    # Keep last 30 days
                    existing = existing[-720:]  # ~30 days * 24 hours

                    with open(drift_log_file, "w") as f:
                        json.dump({"entries": existing}, f, indent=2)

                await self.hass.async_add_executor_job(_update_log)
            except Exception as e:
                _LOGGER.debug(f"Could not save drift log: {e}")

            if abs(discrepancy) > 25:
                _LOGGER.info(
                    f"☁️ Cloud discrepancy detected:\n"
                    f"  Sensor: {sensor_cloud:.0f}%, Forecast: {forecast_cloud:.0f}%, "
                    f"Diff: {discrepancy:+.0f}%\n"
                    f"  → Logged for weight learning (no correction applied)"
                )
            else:
                _LOGGER.debug(
                    f"Cloud check: Sensor={sensor_cloud:.0f}%, Forecast={forecast_cloud:.0f}%, "
                    f"Diff={discrepancy:+.0f}% (within tolerance)"
                )

        except Exception as e:
            _LOGGER.error(f"Cloud discrepancy logging failed: {e}", exc_info=True)

    def setup_listeners(self) -> None:
        """Register the time-based listeners with Home Assistant @zara

        Fixed tasks:
        - 00:25: First forecast (locked)
        - 00:00: Reset expected production
        - XX:05: Hourly actuals update
        - 23:30: End-of-day workflow
        - 1st@07:30: Monthly report

        Dynamic tasks (scheduled daily at 00:30):
        - max(05:00, sunrise-30min): Final forecast with fresh weather
        - sunrise+4h: Cloud discrepancy logging (for weight learning)
        """
        self.cancel_listeners()

        # 00:25 - First forecast
        remove_morning_1 = async_track_time_change(
            self.hass, self.morning_routine_complete, hour=0, minute=25, second=0
        )
        self._listeners.append(remove_morning_1)

        # 00:30 - Schedule dynamic tasks for today
        remove_dynamic_scheduler = async_track_time_change(
            self.hass, self._schedule_dynamic_tasks_callback, hour=0, minute=30, second=0
        )
        self._listeners.append(remove_dynamic_scheduler)

        # 00:00 - Reset expected production
        remove_reset_expected = async_track_time_change(
            self.hass, self.reset_expected_production, hour=0, minute=0, second=0
        )
        self._listeners.append(remove_reset_expected)

        # XX:05 - Hourly actuals update
        remove_hourly_actuals = async_track_time_change(
            self.hass, self.update_hourly_actuals, minute=5, second=0
        )
        self._listeners.append(remove_hourly_actuals)

        # 23:30 - End-of-day workflow
        remove_end_of_day = async_track_time_change(
            self.hass, self.end_of_day_workflow, hour=23, minute=30, second=0
        )
        self._listeners.append(remove_end_of_day)

        # Monthly system report - runs at 07:30 on the 1st of each month
        remove_monthly_report = async_track_time_change(
            self.hass, self.generate_monthly_report, hour=7, minute=30, second=0
        )
        self._listeners.append(remove_monthly_report)

        _LOGGER.info(
            "Scheduled tasks registered:\n"
            "  Fixed: 00:25 (first forecast), 00:00 (reset), XX:05 (hourly), 23:30 (EOD)\n"
            "  Dynamic: Scheduled at 00:30 based on sunrise"
        )

        # Also schedule dynamic tasks now (for startup/reload scenarios)
        self.hass.async_create_task(self._schedule_dynamic_tasks_for_today())

    @callback
    async def _schedule_dynamic_tasks_callback(self, now: datetime) -> None:
        """Callback to schedule dynamic tasks at 00:30 @zara"""
        await self._schedule_dynamic_tasks_for_today()

    def cancel_listeners(self) -> None:
        """Remove any active time-based listeners @zara"""
        for remove_listener in self._listeners:
            try:
                remove_listener()
            except Exception as e:
                _LOGGER.warning(f"Error removing scheduled task listener: {e}")
        self._listeners = []

        for remove_listener in self._dynamic_listeners:
            try:
                remove_listener()
            except Exception as e:
                _LOGGER.warning(f"Error removing dynamic task listener: {e}")
        self._dynamic_listeners = []

    async def calculate_yesterday_deviation_on_startup(self) -> None:
        """Calculates the forecast deviation from yesterday using daily_forecasts.json @zara"""
        try:
            daily_forecasts = await self.data_manager.load_daily_forecasts()

            if not daily_forecasts:
                self.coordinator.last_day_error_kwh = 0.0
                self.coordinator.yesterday_accuracy = 0.0
                return

            yesterday = daily_forecasts.get("yesterday", {})

            if not yesterday:
                self.coordinator.last_day_error_kwh = 0.0
                self.coordinator.yesterday_accuracy = 0.0
                return

            forecast_day = yesterday.get("forecast_day", {})
            actual_day = yesterday.get("actual_day", {})

            forecast_kwh = forecast_day.get("prediction_kwh")
            actual_kwh = actual_day.get("actual_kwh")

            if forecast_kwh is None or actual_kwh is None:
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

            statistics = daily_forecasts.get("statistics", {})
            current_month = statistics.get("current_month", {})
            if current_month and current_month.get("yield_kwh"):
                self.coordinator.avg_month_yield = round(current_month.get("yield_kwh", 0.0), 2)
            else:
                self.coordinator.avg_month_yield = 0.0

            self.coordinator.async_update_listeners()
            _LOGGER.debug(f"Yesterday deviation: {deviation:.2f} kWh, accuracy: {accuracy_pct:.1f}%")

        except Exception as e:
            _LOGGER.error(f"Error calculating yesterday's deviation: {e}")
            self.coordinator.last_day_error_kwh = 0.0
            self.coordinator.yesterday_accuracy = 0.0
            self.coordinator.avg_month_yield = 0.0

    @callback
    async def scheduled_morning_update(self, now: datetime) -> None:
        """Callback for the scheduled morning task Triggers a full forecast update @zara"""
        try:
            await self.coordinator.async_request_refresh()
            await asyncio.sleep(0.5)

            if self.coordinator.data and "forecast_today" in self.coordinator.data:
                forecast_today = self.coordinator.data.get("forecast_today")
                if forecast_today is not None:
                    self.coordinator.expected_daily_production = forecast_today
                    await self.data_manager.save_expected_daily_production(forecast_today)
                    self.coordinator.async_update_listeners()
                else:
                    _LOGGER.error("Morning update: forecast_today is None")
            else:
                _LOGGER.error("Morning update: coordinator.data missing")

        except Exception as e:
            _LOGGER.error(f"Failed morning forecast update: {e}")

    @callback
    async def morning_routine_complete(self, now: datetime) -> None:
        """COMPLETE MORNING ROUTINE @zara"""
        current_time = now if now is not None else dt_util.now()

        try:
            await self.scheduled_morning_update(now)
            await asyncio.sleep(1.0)
            await self.create_morning_hourly_predictions(now)
            _LOGGER.info(f"Morning routine completed at {current_time.strftime('%H:%M')}")

        except Exception as e:
            _LOGGER.error(f"Morning routine failed: {e}")

    @callback
    async def reset_expected_production(self, now: datetime) -> None:
        """Reset expected daily production at midnight @zara"""
        try:
            await self.coordinator.reset_expected_daily_production()

            # Reset sensor correction daily state
            self.sensor_cloud_correction.reset_daily_state()

        except Exception as e:
            _LOGGER.error(f"Failed to reset expected daily production: {e}")

    @callback
    async def generate_monthly_report(self, now: datetime) -> None:
        """Generate monthly system report on the 1st of each month at 07:30. @zara"""
        current_time = now if now is not None else dt_util.now()

        # Only run on the 1st of the month
        if current_time.day != 1:
            return

        _LOGGER.info("Generating monthly system report...")

        try:
            report_generator = SystemReportGenerator(self.data_manager.data_dir)
            success = await report_generator.generate_report()

            if success:
                _LOGGER.info("Monthly system report generated successfully")
            else:
                _LOGGER.warning("Monthly system report generation failed")

        except Exception as e:
            _LOGGER.error(f"Error generating monthly report: {e}", exc_info=True)

    @callback
    async def end_of_day_workflow(self, now: datetime) -> None:
        """Consolidated End-of-Day Workflow at 23:30 - All tasks in sequence @zara"""
        current_time = now if now is not None else dt_util.now()

        if self._end_of_day_running:
            _LOGGER.debug("END_OF_DAY_WORKFLOW already running - skipping")
            return

        self._end_of_day_running = True

        try:
            await self._end_of_day_workflow_internal(current_time)
        finally:
            self._end_of_day_running = False

    async def _end_of_day_workflow_internal(self, current_time: datetime) -> None:
        """Internal implementation of end-of-day workflow (called with mutex held). @zara"""
        workflow_start = asyncio.get_event_loop().time()

        steps_completed = 0
        total_steps = 12  # Includes Panel Group Efficiency Learning
        errors = []

        try:
            await self._finalize_day_internal(current_time)
            steps_completed += 1
        except Exception as e:
            _LOGGER.error(f"Step 1 (finalize) failed: {e}")
            errors.append(f"Finalize: {str(e)}")

        try:
            today = current_time.date().isoformat()

            hourly_predictions = (
                await self.coordinator.data_manager.hourly_predictions.get_predictions_for_date(
                    today
                )
            )

            if hourly_predictions:
                await self.coordinator.data_manager.daily_summaries.create_daily_summary(
                    date=today, hourly_predictions=hourly_predictions
                )

            steps_completed += 1

        except Exception as e:
            _LOGGER.error(f"Step 2 (summary) failed: {e}")
            errors.append(f"Daily summary: {str(e)}")

        try:
            await self._move_to_history_internal(current_time)
            steps_completed += 1
        except Exception as e:
            _LOGGER.error(f"Step 3 (history) failed: {e}")
            errors.append(f"History: {str(e)}")

        try:
            await self._update_yesterday_deviation_internal(current_time)
            steps_completed += 1
        except Exception as e:
            _LOGGER.error(f"Step 4 (deviation) failed: {e}")
            errors.append(f"Deviation: {str(e)}")

        try:
            await self._calculate_stats_internal(current_time)
            steps_completed += 1
        except Exception as e:
            _LOGGER.error(f"Step 5 (stats) failed: {e}")
            errors.append(f"Statistics: {str(e)}")

        # Step 6: Weather Precision Calculation (compare forecast vs actual weather)
        try:
            await self._calculate_weather_precision(current_time)
            steps_completed += 1
        except Exception as e:
            _LOGGER.error(f"Step 6 (weather precision) failed: {e}")
            errors.append(f"Weather precision: {str(e)}")

        try:
            weather_corrector = self._get_weather_corrector()
            # Use today's date to include freshly finalized data in RB correction
            today_str = current_time.strftime("%Y-%m-%d")
            await weather_corrector.update_rb_correction(today_str)
            steps_completed += 1
        except Exception as e:
            _LOGGER.error(f"Step 7 (RB correction) failed: {e}")
            errors.append(f"RB correction: {str(e)}")

        try:
            await self._night_cleanup_internal(current_time)
            steps_completed += 1
        except Exception as e:
            _LOGGER.error(f"Step 8 (cleanup) failed: {e}")
            errors.append(f"Cleanup: {str(e)}")

        try:
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

                            weather_data = pred.get("weather_corrected") or pred.get("weather_forecast") or {}
                            hourly_conditions[hour] = {
                                "cloud_cover_percent": weather_data.get("clouds", 100),
                                "temperature_c": weather_data.get("temperature", 0),
                                "solar_radiation_wm2": weather_data.get("solar_radiation_wm2", 0)
                            }

                    if hourly_actuals and astronomy_data:
                        # Extract peak power for breakthrough detection
                        peak_power_w = finalized.get("peak_power_w")

                        # Get system capacity for breakthrough ratio calculation
                        system_capacity_kwp = None
                        if self.coordinator:
                            system_capacity_kwp = getattr(self.coordinator, 'solar_capacity', None)

                        await pattern_learner.update_pattern_from_day(
                            target_date=today,
                            actual_production_kwh=actual_production_kwh,
                            hourly_actuals=hourly_actuals,
                            hourly_conditions=hourly_conditions,
                            astronomy_data=astronomy_data,
                            peak_power_w=peak_power_w,
                            system_capacity_kwp=system_capacity_kwp
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

                                await pattern_learner.apply_aggressive_correction(
                                    target_date=today,
                                    forecast_raw=forecast_raw,
                                    actual_kwh=actual_production_kwh,
                                    avg_cloud_cover=avg_clouds,
                                )
                            else:
                                _LOGGER.debug("No raw forecast available for aggressive correction")
                        except Exception as corr_err:
                            _LOGGER.warning(f"Aggressive correction failed: {corr_err}")

            steps_completed += 1

        except Exception as e:
            _LOGGER.error(f"Step 9 (pattern learning) failed: {e}")
            errors.append(f"Pattern learning: {str(e)}")

        try:
            await self._train_residual_model(current_time)
            steps_completed += 1
        except Exception as e:
            _LOGGER.error(f"Step 10 (residual training) failed: {e}")
            errors.append(f"Residual training: {str(e)}")

        # Step 11: Train ML main model (learned_weights.json)
        try:
            await self._train_ml_main_model(current_time)
            steps_completed += 1
        except Exception as e:
            _LOGGER.error(f"Step 11 (ML main model training) failed: {e}")
            errors.append(f"ML main model: {str(e)}")

        # Step 12: Panel Group Efficiency Learning (if panel groups with sensors configured)
        try:
            await self._train_panel_group_efficiency(current_time)
            steps_completed += 1
        except Exception as e:
            _LOGGER.error(f"Step 12 (Panel Group Efficiency learning) failed: {e}")
            errors.append(f"Panel Group Efficiency: {str(e)}")

        workflow_duration = asyncio.get_event_loop().time() - workflow_start

        if steps_completed == total_steps:
            _LOGGER.info(f"End-of-day workflow completed ({steps_completed}/{total_steps} steps, {workflow_duration:.1f}s)")

            try:
                if self.coordinator:
                    self.coordinator.update_system_status(
                        event_type="end_of_day_workflow",
                        event_status="success",
                        event_summary="Tagesabschluss erfolgreich",
                        event_details={
                            "duration_seconds": round(workflow_duration, 1),
                            "steps_completed": f"{steps_completed}/{total_steps}",
                        },
                    )
            except Exception:
                pass
        else:
            _LOGGER.warning(f"End-of-day workflow partial ({steps_completed}/{total_steps}): {'; '.join(errors)}")

            try:
                if self.coordinator:
                    self.coordinator.update_system_status(
                        event_type="end_of_day_workflow",
                        event_status="partial",
                        event_summary=f"Tagesabschluss teilweise ({steps_completed}/{total_steps})",
                        event_details={
                            "duration_seconds": round(workflow_duration, 1),
                            "steps_completed": f"{steps_completed}/{total_steps}",
                            "errors": errors,
                        },
                    )
            except Exception:
                pass

    async def _train_residual_model(self, now: datetime) -> None:
        """Train the Residual ML model for Physics+ML ensemble (Phase 4). @zara"""
        try:
            from ..ml.ml_residual_trainer import ResidualTrainer
            from ..ml.ml_data_loader_v3 import MLDataLoaderV3

            data_dir = self.data_manager.data_dir
            # MLDataLoaderV3 expects the stats/ subdirectory as data_dir
            data_loader = MLDataLoaderV3(data_dir / "stats", hass=self.hass)
            records, count = await data_loader.load_training_data(min_samples=20)

            if count < 20:
                return

            solar_capacity = 2.0
            panel_groups = None
            if self.coordinator:
                solar_capacity = getattr(self.coordinator, 'solar_capacity', 2.0) or 2.0
                panel_groups = getattr(self.coordinator, 'panel_groups', None)

            # Pass panel_groups so ResidualTrainer uses correct geometry for multi-orientation systems
            residual_trainer = ResidualTrainer(
                data_dir=data_dir,
                system_capacity_kwp=solar_capacity,
                panel_groups=panel_groups,
                skip_load=True,  # Avoid blocking call in async context
            )
            await residual_trainer.async_load_state()  # Load state asynchronously

            success, accuracy, algo = await residual_trainer.train_residual_model(
                training_records=records,
                algorithm="auto",
            )

            if success:
                _LOGGER.debug(
                    f"Residual model trained: accuracy={accuracy:.3f}, algo={algo}, "
                    f"panel_groups={'yes' if panel_groups else 'no'}"
                )

        except ImportError:
            pass
        except Exception as e:
            _LOGGER.error(f"Residual model training error: {e}")
            raise

    async def _train_ml_main_model(self, now: datetime) -> None:
        """Train the ML main model (learned_weights.json) for direct predictions. @zara

        This trains the Ridge/LSTM model that populates learned_weights.json.
        Unlike ResidualTrainer (which corrects physics predictions), this model
        can make direct predictions when physics engine is not available.
        """
        try:
            if not self.coordinator or not self.coordinator.ml_predictor:
                _LOGGER.debug("ML main model training skipped: no ml_predictor available")
                return

            ml_predictor = self.coordinator.ml_predictor

            # Train the model - this will update learned_weights.json
            result = await ml_predictor.train_model()

            if result.success:
                _LOGGER.info(
                    f"ML main model trained: accuracy={result.accuracy:.3f}, "
                    f"samples={result.samples_used}, features={result.feature_count}"
                )

                # Update coordinator with new accuracy
                if self.coordinator:
                    self.coordinator.on_ml_training_complete(
                        timestamp=dt_util.now(),
                        accuracy=result.accuracy
                    )
            else:
                if result.error_message:
                    _LOGGER.warning(f"ML main model training incomplete: {result.error_message}")
                else:
                    _LOGGER.debug("ML main model training returned without success (may need more data)")

        except Exception as e:
            _LOGGER.error(f"ML main model training error: {e}")
            raise

    async def _train_panel_group_efficiency(self, now: datetime) -> None:
        """Train panel group efficiency factors from today's data. @zara

        This trains per-group efficiency factors when panel groups with energy sensors
        are configured. Uses actual per-group production data stored in hourly_predictions.
        """
        try:
            # Check if panel groups are configured
            panel_groups = getattr(self.coordinator, 'panel_groups', [])
            if not panel_groups:
                _LOGGER.debug("Panel group efficiency training skipped: no panel groups configured")
                return

            # Check if any panel group has an energy sensor
            has_any_sensor = any(g.get("energy_sensor") for g in panel_groups)
            if not has_any_sensor:
                _LOGGER.debug("Panel group efficiency training skipped: no energy sensors configured")
                return

            from ..physics import PanelGroupEfficiencyLearner
            from pathlib import Path

            data_dir = Path(self.data_manager.data_dir)

            # Initialize the learner (skip_load=True to avoid blocking call)
            efficiency_learner = PanelGroupEfficiencyLearner(
                data_path=data_dir,
                panel_groups=panel_groups,
                skip_load=True,
            )
            await efficiency_learner.async_load_state()

            # Get today's hourly predictions with panel_group_actuals
            today_str = now.strftime("%Y-%m-%d")
            predictions = await self.data_manager.hourly_predictions.get_predictions_for_date(today_str)

            if not predictions:
                _LOGGER.debug(f"Panel group efficiency training: no predictions for {today_str}")
                return

            # Count entries with panel_group_actuals
            entries_with_actuals = 0
            entries_added = 0

            for pred in predictions:
                panel_group_actuals = pred.get("panel_group_actuals")
                if not panel_group_actuals:
                    continue

                entries_with_actuals += 1

                # Extract data needed for learning
                target_date = pred.get("target_date", today_str)
                hour = pred.get("target_hour")  # Field is target_hour, not hour
                actual_kwh = pred.get("actual_kwh")

                if hour is None or actual_kwh is None:
                    continue

                # Get weather data from weather_corrected sub-dict
                weather_corrected = pred.get("weather_corrected", {}) or {}
                ghi = weather_corrected.get("ghi", 0) or 0
                dni = weather_corrected.get("direct_radiation", 0) or 0
                dhi = weather_corrected.get("diffuse_radiation", 0) or 0
                temperature = weather_corrected.get("temperature", 15) or 15
                cloud_cover = weather_corrected.get("clouds", 50) or 50

                # Get astronomy data from astronomy sub-dict
                astronomy = pred.get("astronomy", {}) or {}
                sun_elevation = astronomy.get("elevation_deg", 0) or 0
                sun_azimuth = astronomy.get("azimuth_deg", 180) or 180

                # Build timestamp for the data point
                timestamp = f"{target_date}T{hour:02d}:30:00"

                try:
                    efficiency_learner.add_data_point(
                        timestamp=timestamp,
                        hour=hour,
                        sun_elevation_deg=sun_elevation,
                        sun_azimuth_deg=sun_azimuth,
                        actual_power_kwh=actual_kwh,
                        ghi_wm2=ghi,
                        dni_wm2=dni,
                        dhi_wm2=dhi,
                        ambient_temp_c=temperature,
                        cloud_cover_percent=cloud_cover,
                        per_group_actuals=panel_group_actuals,
                    )
                    entries_added += 1
                except Exception as e:
                    _LOGGER.debug(f"Could not add data point for hour {hour}: {e}")

            if entries_added == 0:
                _LOGGER.debug(
                    f"Panel group efficiency training: no valid entries to add "
                    f"(found {entries_with_actuals} entries with actuals)"
                )
                return

            # Optimize efficiency factors (returns bool, not dict)
            success = await efficiency_learner.optimize_efficiency()

            if success:
                _LOGGER.info(
                    f"Panel group efficiency trained: {entries_added} samples, "
                    f"{len(panel_groups)} groups"
                )
            else:
                _LOGGER.debug(
                    "Panel group efficiency optimization: not enough samples yet"
                )

        except Exception as e:
            _LOGGER.error(f"Panel group efficiency training error: {e}")
            raise

    async def _calculate_weather_precision(self, now: datetime) -> None:
        """Calculate weather precision factors (compare forecast vs actual weather). @zara

        This compares the RAW Open-Meteo forecast with actual sensor readings
        to learn location-specific correction factors for weather predictions.
        """
        try:
            # Access weather pipeline manager through coordinator
            if not self.coordinator or not hasattr(self.coordinator, 'weather_pipeline_manager'):
                _LOGGER.debug("Weather precision skipped: no weather_pipeline_manager")
                return

            pipeline = self.coordinator.weather_pipeline_manager
            if not pipeline:
                _LOGGER.debug("Weather precision skipped: pipeline is None")
                return

            # Check required components
            if not pipeline.weather_precision or not pipeline.weather_actual_tracker or not pipeline.weather_corrector:
                _LOGGER.debug("Weather precision skipped: required components not initialized")
                return

            # Calculate for yesterday (full day of data available)
            yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")

            _LOGGER.info(f"Calculating weather precision for {yesterday}")

            success = await pipeline.weather_precision.calculate_correction_factors_v2(
                weather_actual_tracker=pipeline.weather_actual_tracker,
                weather_corrector=pipeline.weather_corrector,
                date_str=yesterday,
            )

            if success:
                _LOGGER.info(f"Weather precision calculated for {yesterday}")
            else:
                _LOGGER.debug(
                    f"Weather precision skipped for {yesterday} "
                    "(normal if no weather data collected that day)"
                )

        except Exception as e:
            _LOGGER.error(f"Weather precision calculation error: {e}")
            raise

    async def _finalize_day_internal(self, now: datetime) -> None:
        """Internal: Finalize current day with actual values @zara"""
        try:
            actual_yield = 0.0
            if self.solar_yield_today_entity_id:
                yield_state = self.hass.states.get(self.solar_yield_today_entity_id)
                if yield_state and yield_state.state not in (None, "unknown", "unavailable"):
                    try:
                        actual_yield = float(yield_state.state)
                    except (ValueError, TypeError):
                        pass

            actual_consumption = None
            if self.coordinator.total_consumption_today:
                consumption_state = self.hass.states.get(self.coordinator.total_consumption_today)
                if consumption_state and consumption_state.state not in (None, "unknown", "unavailable"):
                    try:
                        actual_consumption = float(consumption_state.state)
                    except (ValueError, TypeError):
                        pass

            production_seconds = 0
            try:
                if hasattr(self.coordinator, "production_time_calculator") and self.coordinator.production_time_calculator:
                    production_hours = self.coordinator.production_time_calculator.get_production_hours()
                    if production_hours is not None:
                        production_seconds = int(production_hours * 3600)
            except Exception:
                pass

            try:
                await self._save_actual_best_hour()
            except Exception:
                pass

            success = await self.data_manager.finalize_today(
                yield_kwh=actual_yield,
                consumption_kwh=actual_consumption,
                production_seconds=production_seconds,
            )

            if success:
                try:
                    daily_forecasts = await self.data_manager.load_daily_forecasts()
                    if daily_forecasts:
                        statistics = daily_forecasts.get("statistics", {})
                        current_month = statistics.get("current_month", {})
                        if current_month and current_month.get("yield_kwh"):
                            self.coordinator.avg_month_yield = round(current_month.get("yield_kwh", 0.0), 2)
                except Exception:
                    pass
            else:
                _LOGGER.error("Failed to finalize day")

        except Exception as e:
            _LOGGER.error(f"Failed to finalize day: {e}")

    async def _move_to_history_internal(self, now: datetime) -> None:
        """Internal: Move current day to history @zara"""
        try:
            success = await self.data_manager.move_to_history()
            if not success:
                _LOGGER.error("Failed to move day to history")
        except Exception as e:
            _LOGGER.error(f"Failed to move to history: {e}")

    async def _update_yesterday_deviation_internal(self, now: datetime) -> None:
        """Internal: Update yesterday deviation after moving to history @zara"""
        try:
            history = await self.data_manager.get_history(days=1)

            if not history or len(history) == 0:
                self.coordinator.last_day_error_kwh = 0.0
                self.coordinator.yesterday_accuracy = 0.0
                return

            today_entry = history[0]
            forecast_kwh = today_entry.get("forecast_kwh", 0.0)
            actual_kwh = today_entry.get("actual_kwh", 0.0)

            if forecast_kwh is None or actual_kwh is None:
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

            self.coordinator.last_day_error_kwh = round(deviation, 2)
            self.coordinator.yesterday_accuracy = round(accuracy * 100.0, 1)
            self.coordinator.async_update_listeners()

        except Exception as e:
            _LOGGER.error(f"Failed to update yesterday deviation: {e}")
            self.coordinator.last_day_error_kwh = 0.0
            self.coordinator.yesterday_accuracy = 0.0

    async def _calculate_stats_internal(self, now: datetime) -> None:
        """Internal: Calculate statistics @zara"""
        try:
            success = await self.data_manager.calculate_statistics()
            if not success:
                _LOGGER.error("Failed to calculate statistics")
        except Exception as e:
            _LOGGER.error(f"Failed to calculate statistics: {e}")

    async def _night_cleanup_internal(self, now: datetime) -> None:
        """Night cleanup internal - legacy method retained for compatibility. @zara"""
        pass

    async def _save_actual_best_hour(self) -> None:
        """Calculate and save the actual best production hour from todays hourly samples @zara"""
        try:
            today = dt_util.now().date()
            today_str = today.isoformat()

            today_samples = await self.data_manager.hourly_predictions.get_predictions_for_date(today_str)

            if not today_samples:
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
                await self.data_manager.save_actual_best_hour(hour=best_hour, actual_kwh=best_kwh)

        except Exception as e:
            _LOGGER.error(f"Error calculating actual best hour: {e}")

    @callback
    async def create_morning_hourly_predictions(self, now: datetime) -> None:
        """MORNING TASK: Create hourly predictions for today @zara"""
        current_time = now if now is not None else dt_util.now()

        try:
            today = current_time.date().isoformat()

            weather_service = self.coordinator.weather_service
            if not weather_service:
                _LOGGER.error("Weather service not available")
                return

            current_weather = await weather_service.get_current_weather()
            hourly_weather_forecast = await weather_service.get_corrected_hourly_forecast()

            if not hourly_weather_forecast:
                _LOGGER.error("No corrected weather forecast available")
                return

            sensor_config = {
                "temperature": self.coordinator.entry.data.get(CONF_TEMP_SENSOR) is not None,
                "humidity": self.coordinator.entry.data.get(CONF_HUMIDITY_SENSOR) is not None,
                "lux": self.coordinator.entry.data.get(CONF_LUX_SENSOR) is not None,
                "rain": self.coordinator.entry.data.get(CONF_RAIN_SENSOR) is not None,
                "wind_speed": self.coordinator.entry.data.get(CONF_WIND_SENSOR) is not None,
                "pressure": self.coordinator.entry.data.get(CONF_PRESSURE_SENSOR) is not None,
                "solar_radiation": self.coordinator.entry.data.get(CONF_SOLAR_RADIATION_SENSOR) is not None,
            }

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
                _LOGGER.error("Forecast generation failed - no hourly data")
                return

            astronomy_data = await self._get_astronomy_data(current_time)

            all_hourly = forecast.get("hourly", [])
            today_hourly = [h for h in all_hourly if h.get("date") == today]

            success = await self.morning_routine_handler.execute_morning_routine_with_retry(
                date=today,
                hourly_forecast=today_hourly,
                weather_hourly=hourly_weather_forecast,
                astronomy_data=astronomy_data,
                sensor_config=sensor_config,
            )

            if success:
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

                tomorrow_kwh = forecast.get('tomorrow')
                if tomorrow_kwh is not None:
                    tomorrow_date = current_time + timedelta(days=1)
                    await self.coordinator.data_manager.save_forecast_tomorrow(
                        date=tomorrow_date,
                        prediction_kwh=tomorrow_kwh,
                        source=forecast.get('method', 'morning_routine'),
                        lock=False,
                    )

                day_after_kwh = forecast.get('day_after_tomorrow')
                if day_after_kwh is not None:
                    day_after_date = current_time + timedelta(days=2)
                    await self.coordinator.data_manager.save_forecast_day_after(
                        date=day_after_date,
                        prediction_kwh=day_after_kwh,
                        source=forecast.get('method', 'morning_routine'),
                        lock=False,
                    )

                try:
                    hourly_data = await self.coordinator.data_manager.hourly_predictions._read_json_async()
                    self.coordinator._hourly_predictions_cache = hourly_data
                except Exception:
                    pass

                self.coordinator.async_update_listeners()
                _LOGGER.info(f"Hourly predictions created: {today_kwh:.2f} kWh")

            else:
                _LOGGER.error("Failed to create hourly predictions")

        except Exception as e:
            _LOGGER.error(f"Error in create_morning_hourly_predictions: {e}")

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
                day_data = await self._calculate_astronomy_fallback(target_date)

                if not day_data:
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
            _LOGGER.error(f"Failed to get astronomy data: {e}")
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
                return None

            if not all([
                astronomy_cache.latitude,
                astronomy_cache.longitude,
                astronomy_cache.timezone
            ]):
                return None

            system_capacity_kwp = self.coordinator.solar_capacity or 5.0

            day_data = await astronomy_cache.build_cache_for_date(
                target_date,
                system_capacity_kwp
            )

            return day_data

        except Exception as e:
            _LOGGER.error(f"Astronomy fallback calculation failed: {e}")
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

            return (window_start, window_end)

        except Exception:
            return None

    @callback
    async def update_hourly_actuals(self, now: datetime) -> None:
        """HOURLY TASK (every hour at :05): Update actual values for previous hour @zara

        Uses persistent yield cache to survive restarts.
        """
        current_time = now if now is not None else dt_util.now()

        production_window = await self._get_production_window_from_cache(current_time.date())

        if not production_window:
            return

        production_start, production_end = production_window
        current_time_naive = current_time.replace(tzinfo=None)

        if not (production_start <= current_time_naive <= production_end):
            return

        previous_hour_dt = current_time - timedelta(hours=1)
        previous_hour = previous_hour_dt.hour
        today = current_time.date().isoformat()

        try:
            current_yield = None
            if self.solar_yield_today_entity_id:
                yield_state = self.hass.states.get(self.solar_yield_today_entity_id)
                if yield_state and yield_state.state not in (None, "unknown", "unavailable"):
                    try:
                        current_yield = float(yield_state.state)
                    except (ValueError, TypeError):
                        pass

            if current_yield is None:
                return

            # Load persistent yield cache (survives restarts)
            yield_cache = await self._load_yield_cache()

            # Check if cache is from today
            cache_date = yield_cache.get("date")
            if cache_date != today:
                # New day - reset cache but still try to get previous yield from last hour's prediction
                yield_cache = {}

            previous_yield = yield_cache.get("value")

            # If no cache, try to recover from hourly_predictions
            if previous_yield is None:
                predictions = await self.data_manager.hourly_predictions.get_predictions_for_date(today)
                if predictions:
                    # Find last hour with sensor data
                    for pred in sorted(predictions, key=lambda x: x.get("target_hour", 0), reverse=True):
                        hour = pred.get("target_hour", 99)
                        if hour < current_time.hour:
                            sensor_yield = pred.get("sensor_actual", {}).get("current_yield_kwh")
                            if sensor_yield is not None:
                                previous_yield = sensor_yield
                                _LOGGER.debug(f"Recovered previous_yield={previous_yield} from hour {hour}")
                                break

            # Save current yield to persistent cache
            await self._save_yield_cache({
                "value": current_yield,
                "time": current_time.isoformat(),
                "date": today,
            })

            # Also keep in-memory for backwards compatibility
            self.coordinator._last_yield_cache = {"value": current_yield, "time": current_time}

            if previous_yield is None:
                _LOGGER.debug(
                    f"No previous yield available for hour {previous_hour} - "
                    f"first run after restart, will be available next hour"
                )
                return

            if current_yield >= previous_yield:
                actual_kwh = current_yield - previous_yield
            else:
                _LOGGER.debug(
                    f"Yield decreased (current={current_yield}, previous={previous_yield}) - "
                    f"possibly sensor reset, skipping update"
                )
                return

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

            weather_actual = None
            try:
                hourly_actual_file = self.coordinator.data_manager.data_dir / "stats" / "hourly_weather_actual.json"
                if hourly_actual_file.exists():
                    def _read_hourly_actual():
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
            except Exception:
                pass

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

            astronomy_update = await self._calculate_astronomy_for_hour(current_time)

            # Get inverter max power for clipping detection
            inverter_max_power = getattr(self.coordinator, 'inverter_max_power', 0.0)

            # Read panel group energy sensors if configured
            panel_group_actuals = await self._read_panel_group_actuals()

            success = await self.coordinator.data_manager.hourly_predictions.update_hourly_actual(
                date=today,
                hour=previous_hour,
                actual_kwh=actual_kwh,
                sensor_data=sensor_data,
                weather_actual=weather_actual,
                astronomy_update=astronomy_update,
                inverter_max_power=inverter_max_power,
                panel_group_actuals=panel_group_actuals,
            )

            if success:
                _LOGGER.debug(f"Updated hourly actual for {today} hour {previous_hour}: {actual_kwh:.3f} kWh")
                try:
                    hourly_data = await self.coordinator.data_manager.hourly_predictions._read_json_async()
                    self.coordinator._hourly_predictions_cache = hourly_data
                except Exception:
                    pass

                self.coordinator.async_update_listeners()

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

            return {}

        except Exception:
            return {}

    async def _read_panel_group_actuals(self) -> Optional[Dict[str, float]]:
        """Read actual production from panel group energy sensors. @zara

        Returns:
            Dict mapping group_name to hourly production (kWh), or None if no sensors configured
        """
        try:
            # Get panel group sensor reader from coordinator
            panel_group_sensor_reader = getattr(self.coordinator, 'panel_group_sensor_reader', None)

            if not panel_group_sensor_reader:
                return None

            if not panel_group_sensor_reader.has_any_sensor():
                return None

            # Get hourly production for all groups with sensors
            group_actuals = await panel_group_sensor_reader.get_all_hourly_productions()

            if not group_actuals:
                return None

            _LOGGER.debug(f"Read panel group actuals: {group_actuals}")

            # Optional: Check consistency with total yield
            if self.solar_yield_today_entity_id:
                yield_state = self.hass.states.get(self.solar_yield_today_entity_id)
                if yield_state and yield_state.state not in (None, "unknown", "unavailable"):
                    try:
                        # Note: This is cumulative, not hourly delta - just log for reference
                        current_total = float(yield_state.state)
                        _LOGGER.debug(f"Total yield sensor: {current_total:.3f} kWh")
                    except (ValueError, TypeError):
                        pass

            return group_actuals

        except Exception as e:
            _LOGGER.warning(f"Error reading panel group sensors: {e}")
            return None
