"""
Service Registry for Solar Forecast ML Integration

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
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Awaitable, Callable, Dict, List

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall

from ..const import (
    CONF_BATTERY_ENABLED,
    DOMAIN,
    SERVICE_BUILD_ASTRONOMY_CACHE,
    SERVICE_CALCULATE_STATS,
    SERVICE_COLLECT_HOURLY_SAMPLE,
    SERVICE_DEBUG_CREATE_DAILY_SUMMARY,
    SERVICE_DEBUG_CREATE_HOURLY_PREDICTIONS,
    SERVICE_DEBUG_SHOW_PREDICTION,
    SERVICE_DEBUG_UPDATE_HOURLY_ACTUAL,
    SERVICE_EXTRACT_MAX_PEAKS,
    SERVICE_FINALIZE_DAY,
    SERVICE_FORCE_TODAY_LOCK,
    SERVICE_GENERATE_CHART,
    SERVICE_MIGRATE_DATA,
    SERVICE_MOVE_TO_HISTORY,
    SERVICE_NIGHT_CLEANUP,
    SERVICE_REFRESH_CACHE_TODAY,
    SERVICE_RELOAD_SCHEDULED_TASKS,
    SERVICE_RESET_LEARNING_DATA,
    SERVICE_RETRAIN_MODEL,
    SERVICE_RUN_ALL_DAY_END_TASKS,
    SERVICE_RUN_ALL_SCHEDULED_TASKS,
    SERVICE_TEST_BEST_HOUR,
    SERVICE_TEST_DAY_AFTER_LOCK,
    SERVICE_TEST_DAY_AFTER_SAVE,
    SERVICE_TEST_HOURLY_UPDATE,
    SERVICE_TEST_MORNING_ROUTINE,
    SERVICE_TEST_TOMORROW_LOCK,
    SERVICE_VALIDATE_DATA,
)
from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


@dataclass
class ServiceDefinition:
    """Service definition for registration"""

    name: str
    handler: Callable[[ServiceCall], Awaitable[None]]
    description: str = ""


class ServiceRegistry:
    """Central service registry for Solar Forecast ML"""

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, coordinator: "SolarForecastMLCoordinator"
    ):
        """Initialize service registry"""
        self.hass = hass
        self.entry = entry
        self.coordinator = coordinator
        self._registered_services: List[str] = []

        # Astronomy service handler
        self._astronomy_handler = None

    async def async_register_all_services(self) -> None:
        """Register all services"""
        # Initialize astronomy handler
        from ..services.service_astronomy import AstronomyServiceHandler

        self._astronomy_handler = AstronomyServiceHandler(self.hass, self.entry, self.coordinator)
        await self._astronomy_handler.initialize()

        services = self._build_service_definitions()

        for service in services:
            self.hass.services.async_register(DOMAIN, service.name, service.handler)
            self._registered_services.append(service.name)
            _LOGGER.debug(f"Registered service: {service.name}")

        _LOGGER.info(f"Registered {len(services)} services successfully")

    def unregister_all_services(self) -> None:
        """Unregister all services"""
        for service_name in self._registered_services:
            if self.hass.services.has_service(DOMAIN, service_name):
                self.hass.services.async_remove(DOMAIN, service_name)

        count = len(self._registered_services)
        self._registered_services.clear()
        _LOGGER.info(f"Unregistered {count} services")

    def _build_service_definitions(self) -> List[ServiceDefinition]:
        """Build all service definitions"""
        return [
            # ML Services
            ServiceDefinition(
                name=SERVICE_RETRAIN_MODEL,
                handler=self._handle_retrain_model,
                description="Force ML model retraining",
            ),
            ServiceDefinition(
                name=SERVICE_RESET_LEARNING_DATA,
                handler=self._handle_reset_model,
                description="Reset ML model and learning data",
            ),
            # Day-End Services
            ServiceDefinition(
                name=SERVICE_FINALIZE_DAY,
                handler=self._handle_finalize_day,
                description="Emergency: Finalize current day",
            ),
            ServiceDefinition(
                name=SERVICE_MOVE_TO_HISTORY,
                handler=self._handle_move_to_history,
                description="Emergency: Move current day to history",
            ),
            ServiceDefinition(
                name=SERVICE_CALCULATE_STATS,
                handler=self._handle_calculate_stats,
                description="Manual: Calculate statistics",
            ),
            ServiceDefinition(
                name=SERVICE_RUN_ALL_DAY_END_TASKS,
                handler=self._handle_run_all_day_end_tasks,
                description="Emergency: Run all day-end tasks",
            ),
            # Testing Services - Scheduled task simulation
            ServiceDefinition(
                name=SERVICE_TEST_MORNING_ROUTINE,
                handler=self._handle_test_morning_routine,
                description="Test: Complete 6 AM morning routine",
            ),
            ServiceDefinition(
                name=SERVICE_TEST_BEST_HOUR,
                handler=self._handle_test_best_hour,
                description="Test: Calculate best production hour",
            ),
            ServiceDefinition(
                name=SERVICE_TEST_TOMORROW_LOCK,
                handler=self._handle_test_tomorrow_lock,
                description="Test: Simulate 12 PM tomorrow lock",
            ),
            ServiceDefinition(
                name=SERVICE_TEST_DAY_AFTER_SAVE,
                handler=self._handle_test_day_after_save,
                description="Test: Simulate 6 AM day after save",
            ),
            ServiceDefinition(
                name=SERVICE_TEST_DAY_AFTER_LOCK,
                handler=self._handle_test_day_after_lock,
                description="Test: Simulate 18 PM day after lock",
            ),
            ServiceDefinition(
                name=SERVICE_TEST_HOURLY_UPDATE,
                handler=self._handle_test_hourly_update,
                description="Test: Trigger hourly actual update",
            ),
            # Manual Control Services
            ServiceDefinition(
                name=SERVICE_FORCE_TODAY_LOCK,
                handler=self._handle_force_today_lock,
                description="Force: Override today's forecast lock",
            ),
            # Collection & Cleanup Services
            ServiceDefinition(
                name=SERVICE_COLLECT_HOURLY_SAMPLE,
                handler=self._handle_collect_hourly_sample,
                description="Collect hourly ML sample",
            ),
            ServiceDefinition(
                name=SERVICE_NIGHT_CLEANUP,
                handler=self._handle_night_cleanup,
                description="Manual: Night cleanup task",
            ),
            ServiceDefinition(
                name=SERVICE_RUN_ALL_SCHEDULED_TASKS,
                handler=self._handle_run_all_scheduled_tasks,
                description="Testing: Run all scheduled tasks",
            ),
            ServiceDefinition(
                name=SERVICE_GENERATE_CHART,
                handler=self._handle_generate_chart,
                description="Generate forecast vs actual chart",
            ),
            ServiceDefinition(
                name=SERVICE_RELOAD_SCHEDULED_TASKS,
                handler=self._handle_reload_scheduled_tasks,
                description="Reload: Re-register all scheduled task listeners",
            ),
            # NEW: ML-Optimized Data Structure Debug Services
            ServiceDefinition(
                name=SERVICE_DEBUG_CREATE_HOURLY_PREDICTIONS,
                handler=self._handle_debug_create_hourly_predictions,
                description="Debug: Manually create hourly predictions for a date",
            ),
            ServiceDefinition(
                name=SERVICE_DEBUG_UPDATE_HOURLY_ACTUAL,
                handler=self._handle_debug_update_hourly_actual,
                description="Debug: Manually update actual value for specific hour",
            ),
            ServiceDefinition(
                name=SERVICE_DEBUG_CREATE_DAILY_SUMMARY,
                handler=self._handle_debug_create_daily_summary,
                description="Debug: Manually create daily summary for a date",
            ),
            ServiceDefinition(
                name=SERVICE_DEBUG_SHOW_PREDICTION,
                handler=self._handle_debug_show_prediction,
                description="Debug: Show detailed prediction information",
            ),
            ServiceDefinition(
                name=SERVICE_MIGRATE_DATA,
                handler=self._handle_migrate_data,
                description="Migration: Migrate from old prediction_history.json",
            ),
            ServiceDefinition(
                name=SERVICE_VALIDATE_DATA,
                handler=self._handle_validate_data,
                description="Validation: Validate data integrity",
            ),
            # NEW: Astronomy Cache Services
            ServiceDefinition(
                name=SERVICE_BUILD_ASTRONOMY_CACHE,
                handler=self._handle_build_astronomy_cache,
                description="Build astronomy cache for date range",
            ),
            ServiceDefinition(
                name=SERVICE_EXTRACT_MAX_PEAKS,
                handler=self._handle_extract_max_peaks,
                description="Extract max peak records from history",
            ),
            ServiceDefinition(
                name=SERVICE_REFRESH_CACHE_TODAY,
                handler=self._handle_refresh_cache_today,
                description="Refresh cache for today + next 7 days",
            ),
        ]

    # ==========================================================================
    # SERVICE HANDLERS
    # ==========================================================================

    async def _handle_retrain_model(self, call: ServiceCall) -> None:
        """Handle force_retrain service"""
        _LOGGER.info("Service: force_retrain")
        try:
            if self.coordinator.ml_predictor:
                success = await self.coordinator.ml_predictor.force_training()
                if success:
                    _LOGGER.info("ML model retrained successfully")
                else:
                    _LOGGER.error("ML model retraining failed")
            else:
                _LOGGER.warning("ML predictor not available")
        except Exception as e:
            _LOGGER.error(f"Error in force_retrain: {e}", exc_info=True)

    async def _handle_reset_model(self, call: ServiceCall) -> None:
        """Handle reset_model service"""
        _LOGGER.info("Service: reset_model")
        try:
            if self.coordinator.ml_predictor:
                success = await self.coordinator.ml_predictor.reset_model()
                if success:
                    _LOGGER.info("ML model reset successfully")
                else:
                    _LOGGER.error("ML model reset failed")
            else:
                _LOGGER.warning("ML predictor not available")
        except Exception as e:
            _LOGGER.error(f"Error in reset_model: {e}", exc_info=True)

    async def _handle_finalize_day(self, call: ServiceCall) -> None:
        """Handle finalize_day service"""
        _LOGGER.info("Service: finalize_day (EMERGENCY)")
        try:
            if hasattr(self.coordinator, "scheduled_tasks"):
                await self.coordinator.scheduled_tasks._finalize_day_internal(None)
                _LOGGER.info("Day finalization completed")
            else:
                _LOGGER.error("Scheduled tasks manager not available")
        except Exception as e:
            _LOGGER.error(f"Error in finalize_day: {e}", exc_info=True)

    async def _handle_move_to_history(self, call: ServiceCall) -> None:
        """Handle move_to_history service"""
        _LOGGER.info("Service: move_to_history (EMERGENCY)")
        try:
            if hasattr(self.coordinator, "scheduled_tasks"):
                await self.coordinator.scheduled_tasks.move_to_history_task(None)
                _LOGGER.info("Move to history completed")
            else:
                _LOGGER.error("Scheduled tasks manager not available")
        except Exception as e:
            _LOGGER.error(f"Error in move_to_history: {e}", exc_info=True)

    async def _handle_calculate_stats(self, call: ServiceCall) -> None:
        """Handle calculate_stats service"""
        _LOGGER.info("Service: calculate_stats (MANUAL)")
        try:
            # Check if statistics were recently calculated
            if self.coordinator._last_statistics_calculation:
                time_since = (
                    dt_util.now() - self.coordinator._last_statistics_calculation
                ).total_seconds()
                if time_since < 60:
                    _LOGGER.warning(
                        f"Statistics calculated {time_since:.0f}s ago. "
                        f"Skipping to avoid excessive I/O. Wait at least 1 minute."
                    )
                    return

            success = await self.coordinator.data_manager.calculate_statistics()
            if success:
                self.coordinator._last_statistics_calculation = dt_util.now()
                _LOGGER.info("Statistics calculation completed")
            else:
                _LOGGER.error("Statistics calculation failed")
        except Exception as e:
            _LOGGER.error(f"Error in calculate_stats: {e}", exc_info=True)

    async def _handle_run_all_day_end_tasks(self, call: ServiceCall) -> None:
        """Handle run_all_day_end_tasks service"""
        _LOGGER.info("Service: run_all_day_end_tasks (EMERGENCY)")
        try:
            if hasattr(self.coordinator, "scheduled_tasks"):
                # end_of_day_workflow already includes all steps
                await self.coordinator.scheduled_tasks.end_of_day_workflow(None)
                _LOGGER.info("All day-end tasks completed")
            else:
                _LOGGER.error("Scheduled tasks manager not available")
        except Exception as e:
            _LOGGER.error(f"Error in run_all_day_end_tasks: {e}", exc_info=True)

    async def _handle_test_morning_routine(self, call: ServiceCall) -> None:
        """Handle test_morning_routine service - REPLAY 06:00 routine with STORED data

        This service does EXACTLY what the 06:00 routine does, but using STORED weather data
        from weather_cache.json instead of fetching new data from the API.

        This allows debugging the 06:00 routine without waiting until next morning - it replays
        the exact same process with the same data that was available at 06:00.

        Uses STORED data from:
        - weather_cache.json (weather forecast from 06:00)
        - Current sensor states (external sensors)

        Writes to files EXACTLY as 06:00 routine does:
        - hourly_predictions.json (creates daily predictions)
        - Updates cache and triggers sensor updates
        """
        from ..core.core_helpers import SafeDateTimeUtil as dt_util

        # SIMULATE 06:00 - Set time to 06:00 today morning
        actual_time = dt_util.now()
        current_time = actual_time.replace(hour=6, minute=0, second=0, microsecond=0)

        _LOGGER.info(
            f"🌅 TEST MORNING HOURLY PREDICTIONS (SIMULATE 06:00) - Actual time: {actual_time.strftime('%H:%M:%S')}"
        )
        _LOGGER.info("=" * 80)
        _LOGGER.info(
            f"⏰ SIMULATING: It is 06:00 today morning (pretending current_time = {current_time.strftime('%Y-%m-%d %H:%M:%S')})"
        )
        _LOGGER.info("Using: weather_cache.json from 06:00 + sensor states from 06:00")
        _LOGGER.info("This will calculate the FULL DAY as if it were 06:00 right now")
        _LOGGER.info("=" * 80)

        try:
            today = current_time.date().isoformat()

            # Step 1: Load STORED weather forecast from cache (REPLAY 06:00 with stored data)
            _LOGGER.info("Step 1/5: Loading STORED weather forecast from cache...")

            # Load weather_cache.json to replay what 06:00 had
            import json
            from pathlib import Path

            weather_cache_path = (
                Path(self.coordinator.data_manager.data_dir) / "data" / "weather_cache.json"
            )

            try:
                loop = asyncio.get_event_loop()

                def _load_cache():
                    with open(weather_cache_path, "r") as f:
                        return json.load(f)

                cache_data = await loop.run_in_executor(None, _load_cache)
                forecast_hours = cache_data.get("forecast_hours", [])

                if not forecast_hours:
                    _LOGGER.error("✗ No forecast data in weather cache - ABORTING")
                    return

                # Reconstruct hourly_weather_forecast from cached data
                hourly_weather_forecast = forecast_hours

                # Use first entry as "current weather" (simulates what 06:00 would have seen)
                current_weather = forecast_hours[0] if forecast_hours else None

                _LOGGER.info(
                    f"✓ STORED weather forecast loaded: {len(hourly_weather_forecast)} hours"
                )
                _LOGGER.info(f"  Cache timestamp: {cache_data.get('cached_at', 'unknown')}")

            except FileNotFoundError:
                _LOGGER.error(f"✗ Weather cache not found at {weather_cache_path} - ABORTING")
                return
            except Exception as e:
                _LOGGER.error(f"✗ Failed to load weather cache: {e} - ABORTING")
                return

            # Step 2: Get sensor configuration (EXACTLY as 06:00 routine line 862-872)
            _LOGGER.info("Step 2/5: Collecting sensor configuration...")
            sensor_config = {
                "temperature": hasattr(self.coordinator, "temp_sensor")
                and self.coordinator.temp_sensor is not None,
                "humidity": hasattr(self.coordinator, "humidity_sensor")
                and self.coordinator.humidity_sensor is not None,
                "lux": hasattr(self.coordinator, "lux_sensor")
                and self.coordinator.lux_sensor is not None,
                "rain": hasattr(self.coordinator, "rain_sensor")
                and self.coordinator.rain_sensor is not None,
                "uv_index": hasattr(self.coordinator, "uv_sensor")
                and self.coordinator.uv_sensor is not None,
                "wind_speed": hasattr(self.coordinator, "wind_sensor")
                and self.coordinator.wind_sensor is not None,
            }
            _LOGGER.info(
                f"✓ Sensor config: {sum(sensor_config.values())}/{len(sensor_config)} sensors available"
            )

            # Step 3: Create forecast with ML (EXACTLY as 06:00 routine line 874-893)
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

            # Step 4: Get astronomy data (EXACTLY as 06:00 routine line 895-898)
            _LOGGER.info("Step 4/5: Calculating astronomy data...")
            # Call the same method the real 06:00 routine calls
            astronomy_data = await self.coordinator.scheduled_tasks._get_astronomy_data(
                current_time
            )
            _LOGGER.info(f"✓ Astronomy data calculated")

            # Step 5: Create hourly predictions ONLY FOR TODAY (EXACTLY as 06:00 routine line 900-949)
            _LOGGER.info("Step 5/5: Creating hourly predictions for TODAY only...")

            # Filter hourly_forecast to only include TODAY's hours
            all_hourly = forecast.get("hourly", [])
            today_hourly = [h for h in all_hourly if h.get("date") == today]

            _LOGGER.info(
                f"Filtered hourly forecast: {len(all_hourly)} total → {len(today_hourly)} for today"
            )

            # WRITE TO FILE - EXACTLY as 06:00 routine does (line 909-915)
            success = (
                await self.coordinator.data_manager.hourly_predictions.create_daily_predictions(
                    date=today,
                    hourly_forecast=today_hourly,
                    weather_forecast=hourly_weather_forecast,
                    astronomy_data=astronomy_data,
                    sensor_config=sensor_config,
                )
            )

            if success:
                # Read back what was written - EXACTLY as 06:00 routine (line 918-932)
                predictions = (
                    await self.coordinator.data_manager.hourly_predictions.get_predictions_for_date(
                        today
                    )
                )
                _LOGGER.info(f"\n✓✓✓ HOURLY PREDICTIONS CREATED SUCCESSFULLY ✓✓✓")
                _LOGGER.info(f"  Date: {today}")
                _LOGGER.info(f"  Total predictions: {len(predictions)}")
                _LOGGER.info(
                    f"  Total predicted: {sum(p['predicted_kwh'] for p in predictions):.2f} kWh"
                )

                # Show hourly breakdown - EXACTLY as 06:00 routine (line 925-932)
                _LOGGER.info(f"\n  Hourly breakdown:")
                for p in predictions:
                    _LOGGER.info(
                        f"    {p['target_hour']:02d}:00 - "
                        f"{p['predicted_kwh']:.3f} kWh - "
                        f"Conf: {p['confidence']:.0f}% - "
                        f"{'⭐ PEAK' if p['flags']['is_peak_hour'] else ''}"
                    )

                # Update cache and notify sensors - EXACTLY as 06:00 routine (line 939-947)
                try:
                    hourly_data = (
                        await self.coordinator.data_manager.hourly_predictions._read_json_async()
                    )
                    self.coordinator._hourly_predictions_cache = hourly_data
                except Exception as cache_err:
                    _LOGGER.debug(f"Failed to update hourly predictions cache: {cache_err}")

                # Trigger sensor updates (NextHourSensor and PeakProductionHourSensor read from hourly_predictions.json)
                self.coordinator.async_update_listeners()
                _LOGGER.info(f"  ✓ Sensors notified to reload from hourly_predictions.json")
            else:
                _LOGGER.error("✗ Failed to create hourly predictions")

            _LOGGER.info("=" * 80)

        except Exception as e:
            _LOGGER.error(f"✗ Error in test_morning_routine: {e}", exc_info=True)

    async def _handle_force_today_lock(self, call: ServiceCall) -> None:
        """Handle force_today_lock service - manual 6 AM task trigger with force overwrite"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: force_today_lock - Manual Forecast Lock Override")
        _LOGGER.info("=" * 80)
        _LOGGER.info("⚠️  WARNING: MANUAL CONTROL - This OVERWRITES existing forecast locks!")
        _LOGGER.info("User manually triggered the 6 AM forecast lock task via Developer Tools")
        _LOGGER.info("")
        try:
            await self.coordinator.set_expected_daily_production()
            _LOGGER.info(
                "✓ Manual forecast lock successful: TODAY locked with force_overwrite=True"
            )
            _LOGGER.info("=" * 80)
        except Exception as e:
            _LOGGER.error(f"✗ Error in force_today_lock: {e}", exc_info=True)

    async def _handle_test_best_hour(self, call: ServiceCall) -> None:
        """Handle test_best_hour service - Test V2 sun-aware hybrid calculator"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: test_best_hour - Calculate Best Production Hour")
        _LOGGER.info("=" * 80)
        _LOGGER.info("⚠️  WARNING: TESTING SERVICE - Creates backup before use recommended!")
        _LOGGER.info("Using V2 Hybrid Calculator: ML → Profile → Solar Noon (sun-aware)")
        _LOGGER.info("")
        try:
            best_hour, best_hour_kwh = (
                await self.coordinator.best_hour_calculator.calculate_best_hour_today()
            )

            if best_hour is not None:
                # Determine source based on method used
                if best_hour_kwh is not None and best_hour_kwh > 0:
                    source = "ML-Hourly" if self.coordinator.ml_predictor else "Profile"
                else:
                    source = "Solar-Noon"

                await self.coordinator.data_manager.save_forecast_best_hour(
                    hour=best_hour,
                    prediction_kwh=best_hour_kwh if best_hour_kwh else 0.0,
                    source=source,
                )
                _LOGGER.info(
                    f"✓ Best hour saved: {best_hour:02d}:00 with {best_hour_kwh:.3f} kWh "
                    f"(source: {source}, sun-aware: ✓)"
                )
                _LOGGER.info("=" * 80)
                self.coordinator.async_update_listeners()
            else:
                _LOGGER.warning(
                    "✗ Could not calculate best hour - no sun/ML/profile data available"
                )
                _LOGGER.info("=" * 80)
        except Exception as e:
            _LOGGER.error(f"✗ Error in test_best_hour: {e}", exc_info=True)

    async def _handle_test_tomorrow_lock(self, call: ServiceCall) -> None:
        """Handle test_tomorrow_lock service - Simulate 12 PM tomorrow forecast lock"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: test_tomorrow_lock - Simulate 12 PM Tomorrow Lock")
        _LOGGER.info("=" * 80)
        _LOGGER.info("⚠️  WARNING: TESTING SERVICE - Creates backup before use recommended!")
        _LOGGER.info("This service replicates what happens at 12:00 PM for tomorrow's forecast")
        _LOGGER.info("")
        try:
            weather_service = self.coordinator.weather_service
            if not weather_service:
                _LOGGER.error("✗ Weather service not available - ABORTING")
                return

            current_weather = await weather_service.get_current_weather()
            hourly_forecast = await weather_service.get_processed_hourly_forecast()
            external_sensors = self.coordinator.sensor_collector.collect_all_sensor_data_dict()

            forecast = await self.coordinator.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=self.coordinator.learned_correction_factor,
            )

            tomorrow_kwh = forecast.get("tomorrow")
            if tomorrow_kwh is None:
                _LOGGER.error("✗ Tomorrow forecast not available - ABORTING")
                return

            now_local = dt_util.now()
            tomorrow_date = now_local + timedelta(days=1)
            source = (
                "ML"
                if self.coordinator.forecast_orchestrator.ml_strategy
                and self.coordinator.forecast_orchestrator.ml_strategy.is_available()
                else "Weather"
            )

            await self.coordinator.data_manager.save_forecast_tomorrow(
                date=tomorrow_date,
                prediction_kwh=tomorrow_kwh,
                source=source,
                lock=True,
            )
            _LOGGER.info(f"✓ Tomorrow forecast LOCKED: {tomorrow_kwh:.2f} kWh")
            _LOGGER.info("=" * 80)
            self.coordinator.async_update_listeners()
        except Exception as e:
            _LOGGER.error(f"✗ Error in test_tomorrow_lock: {e}", exc_info=True)

    async def _handle_test_day_after_save(self, call: ServiceCall) -> None:
        """Handle test_day_after_save service - Simulate 6 AM day after tomorrow save (unlocked)"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: test_day_after_save - Simulate 6 AM Day After Tomorrow Save")
        _LOGGER.info("=" * 80)
        _LOGGER.info("⚠️  WARNING: TESTING SERVICE - Creates backup before use recommended!")
        _LOGGER.info("This service saves day after tomorrow forecast UNLOCKED (lock=False)")
        _LOGGER.info("")
        try:
            weather_service = self.coordinator.weather_service
            if not weather_service:
                _LOGGER.error("✗ Weather service not available - ABORTING")
                return

            current_weather = await weather_service.get_current_weather()
            hourly_forecast = await weather_service.get_processed_hourly_forecast()
            external_sensors = self.coordinator.sensor_collector.collect_all_sensor_data_dict()

            forecast = await self.coordinator.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=self.coordinator.learned_correction_factor,
            )

            day_after_kwh = forecast.get("day_after_tomorrow")

            _LOGGER.info(f"\nFinal Forecast Results:")
            _LOGGER.info(f"  - Today: {forecast.get('today', 0):.2f} kWh")
            _LOGGER.info(f"  - Tomorrow: {forecast.get('tomorrow', 0):.2f} kWh")
            _LOGGER.info(f"  - Day After Tomorrow: {day_after_kwh if day_after_kwh else 0:.2f} kWh")
            _LOGGER.info(f"  - Method: {forecast.get('method', 'N/A')}")
            _LOGGER.info("=" * 80)

            if day_after_kwh is None:
                _LOGGER.error("✗ Day after tomorrow forecast not available - ABORTING")
                return

            now_local = dt_util.now()
            day_after_date = now_local + timedelta(days=2)
            source = (
                "ML"
                if self.coordinator.forecast_orchestrator.ml_strategy
                and self.coordinator.forecast_orchestrator.ml_strategy.is_available()
                else "Weather"
            )

            await self.coordinator.data_manager.save_forecast_day_after(
                date=day_after_date,
                prediction_kwh=day_after_kwh,
                source=source,
                lock=False,
            )
            _LOGGER.info(f"✓ Day after tomorrow forecast saved UNLOCKED: {day_after_kwh:.2f} kWh")
            _LOGGER.info("=" * 80)
            self.coordinator.async_update_listeners()
        except Exception as e:
            _LOGGER.error(f"✗ Error in test_day_after_save: {e}", exc_info=True)

    async def _handle_test_day_after_lock(self, call: ServiceCall) -> None:
        """Handle test_day_after_lock service - Simulate 18 PM day after tomorrow lock"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: test_day_after_lock - Simulate 18 PM Day After Tomorrow Lock")
        _LOGGER.info("=" * 80)
        _LOGGER.info("⚠️  WARNING: TESTING SERVICE - Creates backup before use recommended!")
        _LOGGER.info("This service locks day after tomorrow forecast at 18:00 PM")
        _LOGGER.info("")
        try:
            weather_service = self.coordinator.weather_service
            if not weather_service:
                _LOGGER.error("✗ Weather service not available - ABORTING")
                return

            current_weather = await weather_service.get_current_weather()
            hourly_forecast = await weather_service.get_processed_hourly_forecast()
            external_sensors = self.coordinator.sensor_collector.collect_all_sensor_data_dict()

            forecast = await self.coordinator.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=self.coordinator.learned_correction_factor,
            )

            day_after_kwh = forecast.get("day_after_tomorrow")
            if day_after_kwh is None:
                _LOGGER.error("✗ Day after tomorrow forecast not available - ABORTING")
                return

            now_local = dt_util.now()
            day_after_date = now_local + timedelta(days=2)
            source = (
                "ML"
                if self.coordinator.forecast_orchestrator.ml_strategy
                and self.coordinator.forecast_orchestrator.ml_strategy.is_available()
                else "Weather"
            )

            await self.coordinator.data_manager.save_forecast_day_after(
                date=day_after_date,
                prediction_kwh=day_after_kwh,
                source=source,
                lock=True,
            )
            _LOGGER.info(f"✓ Day after tomorrow forecast LOCKED: {day_after_kwh:.2f} kWh")
            _LOGGER.info("=" * 80)
            self.coordinator.async_update_listeners()
        except Exception as e:
            _LOGGER.error(f"✗ Error in test_day_after_lock: {e}", exc_info=True)

    async def _handle_test_hourly_update(self, call: ServiceCall) -> None:
        """Handle test_hourly_update service - Trigger hourly actual update task"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: test_hourly_update - Trigger Hourly Actual Update")
        _LOGGER.info("=" * 80)
        _LOGGER.info("⚠️  TEST SERVICE - Simulates the :05 hourly update task")
        _LOGGER.info("")
        try:
            if not hasattr(self.coordinator, "scheduled_tasks"):
                _LOGGER.error("✗ Scheduled tasks manager not available - ABORTING")
                return

            # Call the actual hourly update method
            now = dt_util.now()
            await self.coordinator.scheduled_tasks.update_hourly_actuals(now)

            _LOGGER.info("=" * 80)
            _LOGGER.info("✓ Hourly update completed")
            _LOGGER.info(
                "Check logs above for details (production window check, yield calculation, accuracy)"
            )
            _LOGGER.info("=" * 80)

            # Trigger sensor updates
            self.coordinator.async_update_listeners()

        except Exception as e:
            _LOGGER.error(f"✗ Error in test_hourly_update: {e}", exc_info=True)

    async def _handle_night_cleanup(self, call: ServiceCall) -> None:
        """Handle night_cleanup service"""
        _LOGGER.info("Service: night_cleanup (MANUAL TEST)")
        try:
            if hasattr(self.coordinator, "scheduled_tasks"):
                await self.coordinator.scheduled_tasks.scheduled_night_cleanup(None)
                _LOGGER.info("Night cleanup completed")
            else:
                _LOGGER.error("Scheduled tasks manager not available")
        except Exception as e:
            _LOGGER.error(f"Error in night_cleanup: {e}", exc_info=True)

    async def _handle_collect_hourly_sample(self, call: ServiceCall) -> None:
        """Handle collect_hourly_sample service"""
        _LOGGER.info("Service: collect_hourly_sample")
        try:
            if not self.coordinator.ml_predictor:
                _LOGGER.error("ML predictor not available")
                return

            if not self.coordinator.ml_predictor.sample_collector:
                _LOGGER.error("Sample collector not available")
                return

            if not self.coordinator.ml_predictor.sample_collector.power_entity:
                _LOGGER.error("Power entity not configured")
                return

            hour_to_collect_dt = dt_util.now() - timedelta(hours=1)
            _LOGGER.info(f"Collecting hourly sample for hour {hour_to_collect_dt.hour} (local)")

            await self.coordinator.ml_predictor.sample_collector.collect_sample(hour_to_collect_dt)

            _LOGGER.info(f"Hourly sample collection completed for hour {hour_to_collect_dt.hour}")
        except Exception as e:
            _LOGGER.error(f"Error in collect_hourly_sample: {e}", exc_info=True)

    async def _handle_run_all_scheduled_tasks(self, call: ServiceCall) -> None:
        """Handle run_all_scheduled_tasks service - COMPLETE TEST of all 18 scheduled tasks This service runs ALL scheduled tasks in chronological order for debugging and testing. Updated: 2025-11-08 - Complete rewrite to match actual task schedule"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: run_all_scheduled_tasks - COMPLETE TEST SEQUENCE (18 Tasks)")
        _LOGGER.info("=" * 80)

        task_count = 0
        success_count = 0
        failed_count = 0
        skipped_count = 0

        try:
            now = dt_util.now()

            # Check availability of components
            has_scheduled_tasks = hasattr(self.coordinator, "scheduled_tasks")
            has_ml_predictor = (
                hasattr(self.coordinator, "ml_predictor") and self.coordinator.ml_predictor
            )

            battery_enabled = self.entry.options.get(
                CONF_BATTERY_ENABLED, self.entry.data.get(CONF_BATTERY_ENABLED, False)
            )
            battery_coordinator_key = f"{self.entry.entry_id}_battery"
            battery_coordinator = (
                self.hass.data[DOMAIN].get(battery_coordinator_key) if battery_enabled else None
            )

            # =================================================================================
            # TASK 1: [23:05] ML Training Check (Daily)
            # =================================================================================
            task_count += 1
            _LOGGER.info(f"\nTASK {task_count}: [23:05] ML Training Check (Daily)")
            try:
                if has_ml_predictor:
                    await self.coordinator.ml_predictor._daily_training_check_callback()
                    _LOGGER.info(f"✓ Task {task_count} completed: ML Training Check")
                    success_count += 1
                else:
                    _LOGGER.info(f"⊘ Task {task_count} skipped: ML Predictor not available")
                    skipped_count += 1
            except Exception as e:
                _LOGGER.error(f"✗ Task {task_count} FAILED: ML Training Check - {e}", exc_info=True)
                failed_count += 1

            # =================================================================================
            # TASK 2: [23:30] End-of-Day Workflow (Consolidated: Finalize + History + Stats + Cleanup)
            # =================================================================================
            task_count += 1
            _LOGGER.info(f"\nTASK {task_count}: [23:30] End-of-Day Workflow (4 Steps)")
            try:
                if has_scheduled_tasks:
                    await self.coordinator.scheduled_tasks.end_of_day_workflow(now)
                    _LOGGER.info(f"✓ Task {task_count} completed: End-of-Day Workflow")
                    success_count += 1
                else:
                    _LOGGER.warning(
                        f"✗ Task {task_count} FAILED: Scheduled tasks manager not available"
                    )
                    failed_count += 1
            except Exception as e:
                _LOGGER.error(
                    f"✗ Task {task_count} FAILED: End-of-Day Workflow - {e}", exc_info=True
                )
                failed_count += 1

            # =================================================================================
            # TASK 3: [00:00] Reset Expected Production
            # =================================================================================
            task_count += 1
            _LOGGER.info(f"\nTASK {task_count}: [00:00] Reset Expected Production")
            try:
                if has_scheduled_tasks:
                    await self.coordinator.scheduled_tasks.reset_expected_production(now)
                    _LOGGER.info(f"✓ Task {task_count} completed: Reset Expected Production")
                    success_count += 1
                else:
                    _LOGGER.warning(
                        f"✗ Task {task_count} FAILED: Scheduled tasks manager not available"
                    )
                    failed_count += 1
            except Exception as e:
                _LOGGER.error(
                    f"✗ Task {task_count} FAILED: Reset Expected Production - {e}", exc_info=True
                )
                failed_count += 1

            # =================================================================================
            # TASK 4: [00:00:30] Midnight Forecast Rotation
            # =================================================================================
            task_count += 1
            _LOGGER.info(f"\nTASK {task_count}: [00:00:30] Midnight Forecast Rotation")
            try:
                await self.coordinator._rotate_forecasts_midnight()
                _LOGGER.info(f"✓ Task {task_count} completed: Midnight Forecast Rotation")
                success_count += 1
            except Exception as e:
                _LOGGER.error(
                    f"✗ Task {task_count} FAILED: Midnight Forecast Rotation - {e}", exc_info=True
                )
                failed_count += 1

            # =================================================================================
            # TASK 5: [03:00] Weekly ML Retraining (Sunday only)
            # =================================================================================
            task_count += 1
            _LOGGER.info(f"\nTASK {task_count}: [03:00] Weekly ML Retraining (Sunday only)")
            try:
                if has_ml_predictor:
                    if now.weekday() == 6:  # Sunday
                        await self.coordinator.ml_predictor.train_model()
                        _LOGGER.info(
                            f"✓ Task {task_count} completed: Weekly ML Retraining (Sunday)"
                        )
                        success_count += 1
                    else:
                        _LOGGER.info(
                            f"⊘ Task {task_count} skipped: Not Sunday (today is {now.strftime('%A')})"
                        )
                        skipped_count += 1
                else:
                    _LOGGER.info(f"⊘ Task {task_count} skipped: ML Predictor not available")
                    skipped_count += 1
            except Exception as e:
                _LOGGER.error(
                    f"✗ Task {task_count} FAILED: Weekly ML Retraining - {e}", exc_info=True
                )
                failed_count += 1

            # =================================================================================
            # TASK 6: [06:00] Morning Forecast Update
            # =================================================================================
            task_count += 1
            _LOGGER.info(f"\nTASK {task_count}: [06:00] Morning Forecast Update")
            try:
                if has_scheduled_tasks:
                    await self.coordinator.scheduled_tasks.scheduled_morning_update(now)
                    _LOGGER.info(f"✓ Task {task_count} completed: Morning Forecast Update")
                    success_count += 1
                else:
                    _LOGGER.warning(
                        f"✗ Task {task_count} FAILED: Scheduled tasks manager not available"
                    )
                    failed_count += 1
            except Exception as e:
                _LOGGER.error(
                    f"✗ Task {task_count} FAILED: Morning Forecast Update - {e}", exc_info=True
                )
                failed_count += 1

            # =================================================================================
            # TASK 7-9: [06:15, 06:30, 06:45] Forecast Retry Attempts
            # =================================================================================
            for attempt in [1, 2, 3]:
                task_count += 1
                retry_minute = 15 * attempt
                _LOGGER.info(
                    f"\nTASK {task_count}: [06:{retry_minute:02d}] Forecast Retry Attempt #{attempt}"
                )
                try:
                    if has_scheduled_tasks:
                        # Simulate retry time
                        retry_time = now.replace(hour=6, minute=retry_minute, second=10)
                        await self.coordinator.scheduled_tasks.retry_forecast_setting(
                            retry_time, attempt
                        )
                        _LOGGER.info(f"✓ Task {task_count} completed: Forecast Retry #{attempt}")
                        success_count += 1
                    else:
                        _LOGGER.warning(
                            f"✗ Task {task_count} FAILED: Scheduled tasks manager not available"
                        )
                        failed_count += 1
                except Exception as e:
                    _LOGGER.error(
                        f"✗ Task {task_count} FAILED: Forecast Retry #{attempt} - {e}",
                        exc_info=True,
                    )
                    failed_count += 1

            # =================================================================================
            # TASK 10: [REMOVED] Next Hour Forecast Update - Sensor now reads directly from hourly_predictions.json
            # =================================================================================
            # NOTE: No longer needed - NextHourSensor reads directly from hourly_predictions.json

            # =================================================================================
            # TASK 11: [XX:02] Hourly Sample Collection (Test with previous hour)
            # =================================================================================
            task_count += 1
            _LOGGER.info(f"\nTASK {task_count}: [XX:02] Hourly Sample Collection")
            try:
                if has_ml_predictor:
                    hour_to_collect = now - timedelta(hours=1)
                    await self.coordinator.ml_predictor.sample_collector.collect_sample(
                        hour_to_collect
                    )
                    _LOGGER.info(f"✓ Task {task_count} completed: Hourly Sample Collection")
                    success_count += 1
                else:
                    _LOGGER.info(f"⊘ Task {task_count} skipped: ML Predictor not available")
                    skipped_count += 1
            except Exception as e:
                _LOGGER.error(
                    f"✗ Task {task_count} FAILED: Hourly Sample Collection - {e}", exc_info=True
                )
                failed_count += 1

            # =================================================================================
            # OPTIONAL BATTERY TASKS (if battery management enabled)
            # =================================================================================
            if battery_coordinator:
                # Battery Task 1: Daily Rollup
                task_count += 1
                _LOGGER.info(f"\nTASK {task_count}: [BATTERY] Daily Rollup")
                try:
                    await battery_coordinator.async_daily_rollup()
                    _LOGGER.info(f"✓ Task {task_count} completed: Battery Daily Rollup")
                    success_count += 1
                except Exception as e:
                    _LOGGER.error(
                        f"✗ Task {task_count} FAILED: Battery Daily Rollup - {e}", exc_info=True
                    )
                    failed_count += 1

                # Battery Task 2: Electricity Prices Refresh
                task_count += 1
                _LOGGER.info(f"\nTASK {task_count}: [BATTERY] Electricity Prices Refresh")
                try:
                    if (
                        hasattr(battery_coordinator, "electricity_service")
                        and battery_coordinator.electricity_service
                    ):
                        await battery_coordinator.async_refresh_prices()
                        _LOGGER.info(f"✓ Task {task_count} completed: Electricity Prices Refresh")
                        success_count += 1
                    else:
                        _LOGGER.info(
                            f"⊘ Task {task_count} skipped: Electricity service not configured"
                        )
                        skipped_count += 1
                except Exception as e:
                    _LOGGER.error(
                        f"✗ Task {task_count} FAILED: Electricity Prices Refresh - {e}",
                        exc_info=True,
                    )
                    failed_count += 1
            else:
                _LOGGER.info(f"\n⊘ Battery Management Tasks skipped (battery disabled)")
                skipped_count += 2  # 2 battery tasks skipped

            # =================================================================================
            # FINAL SUMMARY
            # =================================================================================
            _LOGGER.info("=" * 80)
            _LOGGER.info("SERVICE: run_all_scheduled_tasks - COMPLETE TEST FINISHED!")
            _LOGGER.info("=" * 80)
            _LOGGER.info(f"Total Tasks:   {task_count}")
            _LOGGER.info(f"✓ Success:     {success_count}")
            _LOGGER.info(f"✗ Failed:      {failed_count}")
            _LOGGER.info(f"⊘ Skipped:     {skipped_count}")
            _LOGGER.info("=" * 80)

            if failed_count > 0:
                _LOGGER.warning(f"⚠️  {failed_count} task(s) FAILED - Check logs above for details")
            elif success_count == task_count:
                _LOGGER.info("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
            else:
                _LOGGER.info(
                    f"✅ All executable tasks completed ({success_count}/{task_count - skipped_count})"
                )

        except Exception as e:
            _LOGGER.error(f"CRITICAL ERROR in run_all_scheduled_tasks: {e}", exc_info=True)

    async def _handle_generate_chart(self, call: ServiceCall) -> None:
        """Handle generate_chart service"""
        _LOGGER.info("Service: generate_chart")
        try:
            from datetime import date

            from .service_chart_generator import ChartGenerator

            # Get parameters
            chart_type = call.data.get("chart_type", "daily")
            target_date_str = call.data.get("date")  # Optional: YYYY-MM-DD

            # Parse date if provided
            target_date = None
            if target_date_str:
                try:
                    target_date = date.fromisoformat(target_date_str)
                except ValueError:
                    _LOGGER.error(f"Invalid date format: {target_date_str}")
                    return

            # Get data directory from coordinator
            data_dir = self.coordinator.data_manager.data_dir

            # Create chart generator
            chart_gen = ChartGenerator(data_dir)

            # Generate chart based on type
            if chart_type == "daily":
                chart_path = await chart_gen.generate_daily_forecast_chart(target_date)
                if chart_path:
                    _LOGGER.info(f"✅ Daily forecast chart generated: {chart_path}")
                else:
                    _LOGGER.warning(
                        "❌ Failed to generate daily forecast chart (check if matplotlib is installed or if data is available for requested date)"
                    )

            elif chart_type == "weekly":
                chart_path = await chart_gen.generate_weekly_accuracy_chart()
                if chart_path:
                    _LOGGER.info(f"✅ Weekly accuracy chart generated: {chart_path}")
                else:
                    _LOGGER.warning(
                        "❌ Failed to generate weekly accuracy chart (check if matplotlib is installed or if enough data is available)"
                    )

            elif chart_type == "production_weather":
                chart_path = await chart_gen.generate_production_weather_chart(target_date)
                if chart_path:
                    _LOGGER.info(f"✅ Production vs Weather chart generated: {chart_path}")
                else:
                    _LOGGER.warning(
                        "❌ Failed to generate production-weather chart (check if matplotlib is installed or if data is available)"
                    )

            elif chart_type == "monthly_heatmap":
                chart_path = await chart_gen.generate_monthly_heatmap()
                if chart_path:
                    _LOGGER.info(f"✅ Monthly heatmap generated: {chart_path}")
                else:
                    _LOGGER.warning(
                        "❌ Failed to generate monthly heatmap (check if matplotlib is installed or if enough data is available)"
                    )

            elif chart_type == "sensor_correlation":
                chart_path = await chart_gen.generate_sensor_correlation_chart()
                if chart_path:
                    _LOGGER.info(f"✅ Sensor correlation chart generated: {chart_path}")
                else:
                    _LOGGER.warning(
                        "❌ Failed to generate sensor correlation chart (check if matplotlib is installed or if enough data is available)"
                    )

            else:
                _LOGGER.error(f"Unknown chart type: {chart_type}")

        except Exception as e:
            _LOGGER.error(f"Error generating chart: {e}", exc_info=True)

    async def _handle_reload_scheduled_tasks(self, call: ServiceCall) -> None:
        """Handle reload_scheduled_tasks service - Re-register all scheduled task listeners"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: reload_scheduled_tasks - Re-register Scheduled Task Listeners")
        _LOGGER.info("=" * 80)
        _LOGGER.info("⚠️  RELOAD SERVICE - Cancels and re-registers all scheduled task listeners")
        _LOGGER.info("")
        try:
            if not hasattr(self.coordinator, "scheduled_tasks"):
                _LOGGER.error("✗ Scheduled tasks manager not available - ABORTING")
                return

            # Cancel existing listeners
            _LOGGER.info("Step 1: Canceling existing listeners...")
            self.coordinator.scheduled_tasks.cancel_listeners()
            _LOGGER.info("  ✓ All existing listeners canceled")

            # Re-register all listeners
            _LOGGER.info("Step 2: Re-registering all scheduled task listeners...")
            self.coordinator.scheduled_tasks.setup_listeners()
            _LOGGER.info("  ✓ All scheduled task listeners re-registered")

            _LOGGER.info("")
            _LOGGER.info("=" * 80)
            _LOGGER.info("✓ Scheduled tasks reloaded successfully")
            _LOGGER.info("Scheduled tasks that were re-registered:")
            _LOGGER.info("  - 06:00 AM: Morning forecast update + hourly predictions creation")
            _LOGGER.info("  - Every hour at :05: Hourly actuals update")
            _LOGGER.info("  - 23:05 PM: ML training check")
            _LOGGER.info("  - 23:30 PM: End of day workflow")
            _LOGGER.info("=" * 80)

        except Exception as e:
            _LOGGER.error(f"✗ Error reloading scheduled tasks: {e}", exc_info=True)

    # ==========================================================================
    # NEW: ML-OPTIMIZED DATA STRUCTURE DEBUG SERVICE HANDLERS
    # ==========================================================================

    async def _handle_debug_create_hourly_predictions(self, call: ServiceCall) -> None:
        """Handle debug_create_hourly_predictions service"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: debug_create_hourly_predictions")
        _LOGGER.info("=" * 80)
        _LOGGER.info("⚠️  DEBUG SERVICE - Manually create hourly predictions")
        _LOGGER.info("")

        try:
            # Get date parameter (default: today)
            date_str = call.data.get("date")
            if date_str:
                target_date = dt_util.parse_datetime(date_str)
                if not target_date:
                    _LOGGER.error(f"Invalid date format: {date_str}")
                    return
            else:
                target_date = dt_util.now()

            date_str = target_date.date().isoformat()

            _LOGGER.info(f"Creating hourly predictions for: {date_str}")

            # Step 1: Fetch weather forecast
            _LOGGER.info("Step 1/5: Fetching weather forecast...")
            weather_service = self.coordinator.weather_service
            if not weather_service:
                _LOGGER.error("✗ Weather service not available")
                return

            hourly_forecast = await weather_service.get_processed_hourly_forecast()
            if not hourly_forecast:
                _LOGGER.error("✗ Failed to fetch hourly forecast")
                return
            _LOGGER.info(f"✓ Fetched {len(hourly_forecast)} hourly weather forecasts")

            # Step 2: Get sensor configuration
            _LOGGER.info("Step 2/5: Collecting sensor configuration...")
            sensor_config = {
                "temperature": hasattr(self.coordinator, "temp_sensor")
                and self.coordinator.temp_sensor is not None,
                "humidity": hasattr(self.coordinator, "humidity_sensor")
                and self.coordinator.humidity_sensor is not None,
                "lux": hasattr(self.coordinator, "lux_sensor")
                and self.coordinator.lux_sensor is not None,
                "rain": hasattr(self.coordinator, "rain_sensor")
                and self.coordinator.rain_sensor is not None,
                "uv_index": hasattr(self.coordinator, "uv_sensor")
                and self.coordinator.uv_sensor is not None,
                "wind_speed": hasattr(self.coordinator, "wind_speed_sensor")
                and self.coordinator.wind_speed_sensor is not None,
            }
            _LOGGER.info(f"✓ Sensor configuration: {sensor_config}")

            # Step 3: Generate ML forecast
            _LOGGER.info("Step 3/5: Generating ML hourly forecast...")
            current_weather = await weather_service.get_current_weather()
            external_sensors = self.coordinator.sensor_collector.collect_all_sensor_data_dict()

            forecast = await self.coordinator.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=self.coordinator.learned_correction_factor,
            )

            if not forecast or "hourly" not in forecast:
                _LOGGER.error("✗ Failed to generate hourly forecast")
                return
            _LOGGER.info(f"✓ Generated hourly forecast with {len(forecast['hourly'])} hours")

            # Step 4: Calculate astronomy data
            _LOGGER.info("Step 4/5: Calculating astronomy data...")
            astronomy_data = self._get_astronomy_data_for_debug(target_date)
            _LOGGER.info(
                f"✓ Astronomy: sunrise={astronomy_data.get('sunrise')}, sunset={astronomy_data.get('sunset')}"
            )

            # Step 5: Create hourly predictions
            _LOGGER.info("Step 5/5: Creating hourly predictions...")
            success = (
                await self.coordinator.data_manager.hourly_predictions.create_daily_predictions(
                    date=date_str,
                    hourly_forecast=forecast["hourly"],
                    weather_forecast=hourly_forecast,
                    astronomy_data=astronomy_data,
                    sensor_config=sensor_config,
                )
            )

            if success:
                _LOGGER.info(f"✓ Successfully created hourly predictions for {date_str}")
                _LOGGER.info("=" * 80)
            else:
                _LOGGER.error(f"✗ Failed to create hourly predictions for {date_str}")
                _LOGGER.info("=" * 80)

        except Exception as e:
            _LOGGER.error(f"Error in debug_create_hourly_predictions: {e}", exc_info=True)

    async def _handle_debug_update_hourly_actual(self, call: ServiceCall) -> None:
        """Handle debug_update_hourly_actual service"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: debug_update_hourly_actual")
        _LOGGER.info("=" * 80)
        _LOGGER.info("⚠️  DEBUG SERVICE - Manually update actual value for hour")
        _LOGGER.info("")

        try:
            # Get parameters
            date_str = call.data.get("date")  # YYYY-MM-DD
            hour = call.data.get("hour")  # 0-23
            actual_kwh = call.data.get("actual_kwh")  # float

            if not date_str or hour is None or actual_kwh is None:
                _LOGGER.error("Missing required parameters: date, hour, actual_kwh")
                return

            _LOGGER.info(f"Updating: date={date_str}, hour={hour}, actual_kwh={actual_kwh}")

            # Collect sensor data
            _LOGGER.info("Collecting sensor data...")
            sensor_data = self.coordinator.sensor_collector.collect_all_sensor_data_dict()
            _LOGGER.info(f"✓ Collected sensor data: {sensor_data}")

            # Get weather data
            _LOGGER.info("Fetching weather data...")
            weather_service = self.coordinator.weather_service
            weather_actual = None
            if weather_service:
                current_weather = await weather_service.get_current_weather()
                if current_weather:
                    weather_actual = {
                        "temperature_c": current_weather.get("temperature"),
                        "cloud_cover_percent": current_weather.get("cloud_cover"),
                        "humidity_percent": current_weather.get("humidity"),
                        "wind_speed_ms": current_weather.get("wind_speed"),
                        "precipitation_mm": current_weather.get("precipitation"),
                        "pressure_hpa": current_weather.get("pressure"),
                    }
                    _LOGGER.info(f"✓ Collected weather data")

            # Update hourly actual
            success = await self.coordinator.data_manager.hourly_predictions.update_hourly_actual(
                date=date_str,
                hour=hour,
                actual_kwh=actual_kwh,
                sensor_data=sensor_data,
                weather_actual=weather_actual,
            )

            if success:
                _LOGGER.info(f"✓ Successfully updated actual for {date_str} hour {hour}")
                _LOGGER.info("=" * 80)
            else:
                _LOGGER.error(f"✗ Failed to update actual for {date_str} hour {hour}")
                _LOGGER.info("=" * 80)

        except Exception as e:
            _LOGGER.error(f"Error in debug_update_hourly_actual: {e}", exc_info=True)

    async def _handle_debug_create_daily_summary(self, call: ServiceCall) -> None:
        """Handle debug_create_daily_summary service"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: debug_create_daily_summary")
        _LOGGER.info("=" * 80)
        _LOGGER.info("⚠️  DEBUG SERVICE - Manually create daily summary")
        _LOGGER.info("")

        try:
            # Get date parameter (default: today)
            date_str = call.data.get("date")
            if date_str:
                target_date = dt_util.parse_datetime(date_str)
                if not target_date:
                    _LOGGER.error(f"Invalid date format: {date_str}")
                    return
            else:
                target_date = dt_util.now()

            date_str = target_date.date().isoformat()

            _LOGGER.info(f"Creating daily summary for: {date_str}")

            # Load hourly predictions for this date
            _LOGGER.info("Loading hourly predictions...")
            hourly_predictions = (
                await self.coordinator.data_manager.hourly_predictions.get_predictions_for_date(
                    date_str
                )
            )

            if not hourly_predictions:
                _LOGGER.error(f"✗ No hourly predictions found for {date_str}")
                return

            _LOGGER.info(f"✓ Loaded {len(hourly_predictions)} hourly predictions")

            # Create daily summary
            _LOGGER.info("Creating daily summary...")
            success = await self.coordinator.data_manager.daily_summaries.create_daily_summary(
                date=date_str, hourly_predictions=hourly_predictions
            )

            if success:
                # Load and display summary
                summary = await self.coordinator.data_manager.daily_summaries.get_summary(date_str)
                if summary:
                    overall = summary.get("overall", {})
                    _LOGGER.info("")
                    _LOGGER.info("Daily Summary Created:")
                    _LOGGER.info(f"  Predicted Total: {overall.get('predicted_total_kwh', 0)} kWh")
                    _LOGGER.info(f"  Actual Total: {overall.get('actual_total_kwh', 0)} kWh")
                    _LOGGER.info(f"  Accuracy: {overall.get('accuracy_percent', 0)}%")
                    _LOGGER.info(f"  Error: {overall.get('error_kwh', 0)} kWh")
                    _LOGGER.info(f"  Production Hours: {overall.get('production_hours', 0)}")

                    patterns = summary.get("patterns", [])
                    if patterns:
                        _LOGGER.info("")
                        _LOGGER.info(f"  Detected Patterns: {len(patterns)}")
                        for pattern in patterns:
                            _LOGGER.info(
                                f"    - {pattern.get('type')}: {pattern.get('avg_error_percent')}% error"
                            )

                    recommendations = summary.get("recommendations", [])
                    if recommendations:
                        _LOGGER.info("")
                        _LOGGER.info(f"  Recommendations: {len(recommendations)}")
                        for rec in recommendations:
                            _LOGGER.info(f"    - {rec.get('action')}: {rec.get('reason')}")

                _LOGGER.info("")
                _LOGGER.info(f"✓ Successfully created daily summary for {date_str}")
                _LOGGER.info("=" * 80)
            else:
                _LOGGER.error(f"✗ Failed to create daily summary for {date_str}")
                _LOGGER.info("=" * 80)

        except Exception as e:
            _LOGGER.error(f"Error in debug_create_daily_summary: {e}", exc_info=True)

    async def _handle_debug_show_prediction(self, call: ServiceCall) -> None:
        """Handle debug_show_prediction service"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: debug_show_prediction")
        _LOGGER.info("=" * 80)
        _LOGGER.info("⚠️  DEBUG SERVICE - Show detailed prediction")
        _LOGGER.info("")

        try:
            # Get parameters
            date_str = call.data.get("date")
            hour = call.data.get("hour")

            if not date_str:
                _LOGGER.error("Missing required parameter: date")
                return

            _LOGGER.info(f"Loading prediction for: {date_str}, hour={hour}")

            if hour is not None:
                # Show specific hour
                prediction = self.coordinator.data_manager.hourly_predictions.get_prediction_by_id(
                    f"{date_str}_{hour}"
                )

                if not prediction:
                    _LOGGER.error(f"✗ Prediction not found: {date_str}_{hour}")
                    return

                _LOGGER.info("")
                _LOGGER.info(f"Prediction Details for {date_str} Hour {hour}:")
                _LOGGER.info(f"  ID: {prediction.get('id')}")
                _LOGGER.info(f"  Predicted: {prediction.get('predicted_kwh')} kWh")
                _LOGGER.info(f"  Actual: {prediction.get('actual_kwh')} kWh")
                _LOGGER.info(f"  Accuracy: {prediction.get('accuracy_percent')}%")
                _LOGGER.info(f"  Error: {prediction.get('error_kwh')} kWh")

                weather_forecast = prediction.get("weather_forecast", {})
                _LOGGER.info("")
                _LOGGER.info("  Weather Forecast:")
                _LOGGER.info(f"    Temperature: {weather_forecast.get('temperature_c')}°C")
                _LOGGER.info(f"    Cloud Cover: {weather_forecast.get('cloud_cover_percent')}%")
                _LOGGER.info(f"    Humidity: {weather_forecast.get('humidity_percent')}%")

                sensor_actual = prediction.get("sensor_actual", {})
                _LOGGER.info("")
                _LOGGER.info("  Sensor Actual:")
                _LOGGER.info(f"    Lux: {sensor_actual.get('lux')}")
                _LOGGER.info(f"    Temperature: {sensor_actual.get('temperature_c')}°C")
                _LOGGER.info(f"    Humidity: {sensor_actual.get('humidity_percent')}%")

                flags = prediction.get("flags", {})
                _LOGGER.info("")
                _LOGGER.info("  Flags:")
                _LOGGER.info(f"    Production Hour: {flags.get('is_production_hour')}")
                _LOGGER.info(f"    Peak Hour: {flags.get('is_peak_hour')}")
                _LOGGER.info(f"    Outlier: {flags.get('is_outlier')}")

            else:
                # Show all hours for date
                predictions = (
                    await self.coordinator.data_manager.hourly_predictions.get_predictions_for_date(
                        date_str
                    )
                )

                if not predictions:
                    _LOGGER.error(f"✗ No predictions found for {date_str}")
                    return

                _LOGGER.info("")
                _LOGGER.info(f"Predictions for {date_str}:")
                _LOGGER.info(f"  Total Hours: {len(predictions)}")
                _LOGGER.info("")
                _LOGGER.info("  Hour | Predicted | Actual   | Accuracy | Error")
                _LOGGER.info("  -----|-----------|----------|----------|-------")
                for p in predictions:
                    hour = p.get("target_hour")
                    predicted = p.get("predicted_kwh", 0)
                    actual = p.get("actual_kwh")
                    accuracy = p.get("accuracy_percent")
                    error = p.get("error_kwh")

                    actual_str = f"{actual:.3f}" if actual is not None else "N/A"
                    accuracy_str = f"{accuracy:.1f}%" if accuracy is not None else "N/A"
                    error_str = f"{error:.3f}" if error is not None else "N/A"

                    _LOGGER.info(
                        f"  {hour:02d}   | {predicted:.3f}   | {actual_str:8} | {accuracy_str:8} | {error_str:6}"
                    )

            _LOGGER.info("")
            _LOGGER.info("=" * 80)

        except Exception as e:
            _LOGGER.error(f"Error in debug_show_prediction: {e}", exc_info=True)

    async def _handle_migrate_data(self, call: ServiceCall) -> None:
        """Handle migrate_data service"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: migrate_data")
        _LOGGER.info("=" * 80)
        _LOGGER.info("⚠️  MIGRATION SERVICE - Migrate from old prediction_history.json")
        _LOGGER.info("")

        try:
            # Get dry_run parameter (default: True for safety)
            dry_run = call.data.get("dry_run", True)

            _LOGGER.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE MIGRATION'}")
            _LOGGER.info("")

            # Import migration module
            from ..data.data_migration import DataMigration

            # Create migration handler
            data_dir = self.coordinator.data_manager.data_dir
            migration = DataMigration(data_dir)

            # Run migration
            report = await migration.migrate(dry_run=dry_run)

            # Display report
            _LOGGER.info("")
            _LOGGER.info("Migration Report:")
            _LOGGER.info(f"  Success: {report['success']}")
            _LOGGER.info(f"  Old File Exists: {report['old_file_exists']}")
            _LOGGER.info(f"  Old Predictions Count: {report['old_predictions_count']}")
            _LOGGER.info(f"  Old Dates Count: {report['old_dates_count']}")
            _LOGGER.info(f"  Converted Dates: {len(report['converted_dates'])}")
            _LOGGER.info(f"  Skipped Dates: {len(report['skipped_dates'])}")

            if report["errors"]:
                _LOGGER.info("")
                _LOGGER.info("  Errors:")
                for error in report["errors"]:
                    _LOGGER.info(f"    - {error}")

            if report["warnings"]:
                _LOGGER.info("")
                _LOGGER.info("  Warnings:")
                for warning in report["warnings"]:
                    _LOGGER.info(f"    - {warning}")

            _LOGGER.info("")
            if report["success"]:
                if dry_run:
                    _LOGGER.info("✓ Dry run completed successfully - No files modified")
                    _LOGGER.info("  Run again with dry_run=false to perform actual migration")
                else:
                    _LOGGER.info("✓ Migration completed successfully")
            else:
                _LOGGER.error("✗ Migration failed - Check errors above")

            _LOGGER.info("=" * 80)

        except Exception as e:
            _LOGGER.error(f"Error in migrate_data: {e}", exc_info=True)

    async def _handle_validate_data(self, call: ServiceCall) -> None:
        """Handle validate_data service"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: validate_data")
        _LOGGER.info("=" * 80)
        _LOGGER.info("⚠️  VALIDATION SERVICE - Validate data integrity")
        _LOGGER.info("")

        try:
            # Import validation module
            from ..data.data_migration import DataValidator

            # Create validator
            data_dir = self.coordinator.data_manager.data_dir
            validator = DataValidator(data_dir)

            # Run validation
            report = await validator.validate()

            # Display report
            _LOGGER.info("")
            _LOGGER.info("Validation Report:")
            _LOGGER.info(f"  Valid: {report['valid']}")
            _LOGGER.info(f"  Files Checked: {', '.join(report['files_checked'])}")

            if report["errors"]:
                _LOGGER.info("")
                _LOGGER.info("  Errors:")
                for error in report["errors"]:
                    _LOGGER.info(f"    - {error}")

            if report["warnings"]:
                _LOGGER.info("")
                _LOGGER.info("  Warnings:")
                for warning in report["warnings"]:
                    _LOGGER.info(f"    - {warning}")

            _LOGGER.info("")
            if report["valid"]:
                _LOGGER.info("✓ Data validation passed")
            else:
                _LOGGER.error("✗ Data validation failed - Check errors above")

            _LOGGER.info("=" * 80)

        except Exception as e:
            _LOGGER.error(f"Error in validate_data: {e}", exc_info=True)

    def _get_astronomy_data_for_debug(self, target_date) -> Dict:
        """
        Get astronomy data for debug service from in-memory astronomy cache

        Falls back to sun.sun entity if cache unavailable
        """
        try:
            # Try in-memory astronomy cache first (no I/O blocking!)
            from ..astronomy.astronomy_cache_manager import get_cache_manager

            cache_manager = get_cache_manager()

            if cache_manager.is_loaded():
                # Get astronomy data for target date
                date_str = (
                    target_date.isoformat()
                    if hasattr(target_date, "isoformat")
                    else str(target_date)
                )
                day_data = cache_manager.get_day_data(date_str)

                if day_data:
                    return {
                        "sunrise": day_data.get("sunrise_local"),
                        "sunset": day_data.get("sunset_local"),
                        "solar_noon": day_data.get("solar_noon_local"),
                        "daylight_hours": day_data.get("daylight_hours"),
                        "hourly": day_data.get(
                            "hourly", {}
                        ),  # CRITICAL: Include hourly astronomy data for V2 features!
                    }

            # Fallback to sun.sun entity if cache unavailable
            _LOGGER.debug("Astronomy cache unavailable for debug, falling back to sun.sun entity")
            sun_entity = self.hass.states.get("sun.sun")

            if not sun_entity:
                _LOGGER.warning("Sun entity not available")
                return {}

            sunrise_str = sun_entity.attributes.get("next_rising")
            sunset_str = sun_entity.attributes.get("next_setting")

            if sunrise_str and sunset_str:
                from datetime import datetime

                # Parse strings to datetime objects
                # Handle both string and datetime formats
                if isinstance(sunrise_str, str):
                    sunrise = datetime.fromisoformat(sunrise_str.replace("Z", "+00:00"))
                else:
                    sunrise = sunrise_str

                if isinstance(sunset_str, str):
                    sunset = datetime.fromisoformat(sunset_str.replace("Z", "+00:00"))
                else:
                    sunset = sunset_str

                # Calculate solar noon (midpoint between sunrise and sunset)
                solar_noon = sunrise + (sunset - sunrise) / 2

                # Calculate daylight hours
                daylight_hours = (sunset - sunrise).total_seconds() / 3600

                return {
                    "sunrise": sunrise.isoformat(),
                    "sunset": sunset.isoformat(),
                    "solar_noon": solar_noon.isoformat(),
                    "daylight_hours": round(daylight_hours, 2),
                }

            return {}

        except Exception as e:
            _LOGGER.warning(f"Failed to get astronomy data: {e}")
            return {}

    # ==========================================================================
    # ASTRONOMY CACHE SERVICE HANDLERS
    # ==========================================================================

    async def _handle_build_astronomy_cache(self, call: ServiceCall) -> None:
        """Handle build_astronomy_cache service"""
        if self._astronomy_handler:
            await self._astronomy_handler.handle_build_astronomy_cache(call)

    async def _handle_extract_max_peaks(self, call: ServiceCall) -> None:
        """Handle extract_max_peaks service"""
        if self._astronomy_handler:
            await self._astronomy_handler.handle_extract_max_peaks(call)

    async def _handle_refresh_cache_today(self, call: ServiceCall) -> None:
        """Handle refresh_cache_today service"""
        if self._astronomy_handler:
            await self._astronomy_handler.handle_refresh_cache_today(call)
