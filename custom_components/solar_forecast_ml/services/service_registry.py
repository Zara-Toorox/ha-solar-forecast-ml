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

import logging
from dataclasses import dataclass
from typing import Callable, Awaitable, Dict, List
from datetime import timedelta

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.config_entries import ConfigEntry

from ..const import (
    DOMAIN,
    SERVICE_RETRAIN_MODEL,
    SERVICE_RESET_LEARNING_DATA,
    SERVICE_FINALIZE_DAY,
    SERVICE_MOVE_TO_HISTORY,
    SERVICE_CALCULATE_STATS,
    SERVICE_RUN_ALL_DAY_END_TASKS,
    SERVICE_DEBUGGING_6AM_FORECAST,
    SERVICE_DEBUGGING_BEST_HOUR,
    SERVICE_DEBUGGING_TOMORROW_12PM,
    SERVICE_DEBUGGING_DAY_AFTER_TOMORROW_6AM,
    SERVICE_DEBUGGING_DAY_AFTER_TOMORROW_6PM,
    SERVICE_COLLECT_HOURLY_SAMPLE,
    SERVICE_NIGHT_CLEANUP,
    SERVICE_RUN_ALL_SCHEDULED_TASKS,
    CONF_BATTERY_ENABLED,
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
    """Central service registry for Solar Forecast ML by @Zara"""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        coordinator: "SolarForecastMLCoordinator"
    ):
        """Initialize service registry by @Zara"""
        self.hass = hass
        self.entry = entry
        self.coordinator = coordinator
        self._registered_services: List[str] = []

    async def async_register_all_services(self) -> None:
        """Register all services by @Zara"""
        services = self._build_service_definitions()

        for service in services:
            self.hass.services.async_register(
                DOMAIN,
                service.name,
                service.handler
            )
            self._registered_services.append(service.name)
            _LOGGER.debug(f"Registered service: {service.name}")

        _LOGGER.info(f"Registered {len(services)} services successfully")

    def unregister_all_services(self) -> None:
        """Unregister all services by @Zara"""
        for service_name in self._registered_services:
            if self.hass.services.has_service(DOMAIN, service_name):
                self.hass.services.async_remove(DOMAIN, service_name)

        count = len(self._registered_services)
        self._registered_services.clear()
        _LOGGER.info(f"Unregistered {count} services")

    def _build_service_definitions(self) -> List[ServiceDefinition]:
        """Build all service definitions by @Zara"""
        return [
            # ML Services
            ServiceDefinition(
                name=SERVICE_RETRAIN_MODEL,
                handler=self._handle_retrain_model,
                description="Force ML model retraining"
            ),
            ServiceDefinition(
                name=SERVICE_RESET_LEARNING_DATA,
                handler=self._handle_reset_model,
                description="Reset ML model and learning data"
            ),

            # Day-End Services
            ServiceDefinition(
                name=SERVICE_FINALIZE_DAY,
                handler=self._handle_finalize_day,
                description="Emergency: Finalize current day"
            ),
            ServiceDefinition(
                name=SERVICE_MOVE_TO_HISTORY,
                handler=self._handle_move_to_history,
                description="Emergency: Move current day to history"
            ),
            ServiceDefinition(
                name=SERVICE_CALCULATE_STATS,
                handler=self._handle_calculate_stats,
                description="Manual: Calculate statistics"
            ),
            ServiceDefinition(
                name=SERVICE_RUN_ALL_DAY_END_TASKS,
                handler=self._handle_run_all_day_end_tasks,
                description="Emergency: Run all day-end tasks"
            ),

            # Debugging Services - Time-based forecast simulations
            ServiceDefinition(
                name=SERVICE_DEBUGGING_6AM_FORECAST,
                handler=self._handle_debugging_6am_forecast,
                description="Debug: Simulate 6 AM forecast lock"
            ),
            ServiceDefinition(
                name=SERVICE_DEBUGGING_BEST_HOUR,
                handler=self._handle_debugging_best_hour,
                description="Debug: Calculate best hour"
            ),
            ServiceDefinition(
                name=SERVICE_DEBUGGING_TOMORROW_12PM,
                handler=self._handle_debugging_tomorrow_12pm,
                description="Debug: Simulate 12 PM tomorrow forecast lock"
            ),
            ServiceDefinition(
                name=SERVICE_DEBUGGING_DAY_AFTER_TOMORROW_6AM,
                handler=self._handle_debugging_day_after_tomorrow_6am,
                description="Debug: Simulate 6 AM day after tomorrow forecast"
            ),
            ServiceDefinition(
                name=SERVICE_DEBUGGING_DAY_AFTER_TOMORROW_6PM,
                handler=self._handle_debugging_day_after_tomorrow_6pm,
                description="Debug: Simulate 18 PM day after tomorrow lock"
            ),

            # Collection & Cleanup Services
            ServiceDefinition(
                name=SERVICE_COLLECT_HOURLY_SAMPLE,
                handler=self._handle_collect_hourly_sample,
                description="Collect hourly ML sample"
            ),
            ServiceDefinition(
                name=SERVICE_NIGHT_CLEANUP,
                handler=self._handle_night_cleanup,
                description="Manual: Night cleanup task"
            ),
            ServiceDefinition(
                name=SERVICE_RUN_ALL_SCHEDULED_TASKS,
                handler=self._handle_run_all_scheduled_tasks,
                description="Testing: Run all scheduled tasks"
            ),
        ]

    # ==========================================================================
    # SERVICE HANDLERS
    # ==========================================================================

    async def _handle_retrain_model(self, call: ServiceCall) -> None:
        """Handle force_retrain service by @Zara"""
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
        """Handle reset_model service by @Zara"""
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
        """Handle finalize_day service by @Zara"""
        _LOGGER.info("Service: finalize_day (EMERGENCY)")
        try:
            if hasattr(self.coordinator, 'scheduled_tasks'):
                await self.coordinator.scheduled_tasks.finalize_day_task(None)
                _LOGGER.info("Day finalization completed")
            else:
                _LOGGER.error("Scheduled tasks manager not available")
        except Exception as e:
            _LOGGER.error(f"Error in finalize_day: {e}", exc_info=True)

    async def _handle_move_to_history(self, call: ServiceCall) -> None:
        """Handle move_to_history service by @Zara"""
        _LOGGER.info("Service: move_to_history (EMERGENCY)")
        try:
            if hasattr(self.coordinator, 'scheduled_tasks'):
                await self.coordinator.scheduled_tasks.move_to_history_task(None)
                _LOGGER.info("Move to history completed")
            else:
                _LOGGER.error("Scheduled tasks manager not available")
        except Exception as e:
            _LOGGER.error(f"Error in move_to_history: {e}", exc_info=True)

    async def _handle_calculate_stats(self, call: ServiceCall) -> None:
        """Handle calculate_stats service by @Zara"""
        _LOGGER.info("Service: calculate_stats (MANUAL)")
        try:
            # Check if statistics were recently calculated
            if self.coordinator._last_statistics_calculation:
                time_since = (dt_util.now() - self.coordinator._last_statistics_calculation).total_seconds()
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
        """Handle run_all_day_end_tasks service by @Zara"""
        _LOGGER.info("Service: run_all_day_end_tasks (EMERGENCY)")
        try:
            if hasattr(self.coordinator, 'scheduled_tasks'):
                await self.coordinator.scheduled_tasks.finalize_day_task(None)
                _LOGGER.info("1/3: Day finalization completed")

                await self.coordinator.scheduled_tasks.move_to_history_task(None)
                _LOGGER.info("2/3: Move to history completed")

                await self.coordinator.data_manager.calculate_statistics()
                _LOGGER.info("3/3: Statistics calculation completed")

                _LOGGER.info("All day-end tasks completed")
            else:
                _LOGGER.error("Scheduled tasks manager not available")
        except Exception as e:
            _LOGGER.error(f"Error in run_all_day_end_tasks: {e}", exc_info=True)

    async def _handle_debugging_6am_forecast(self, call: ServiceCall) -> None:
        """Handle debugging_6am_forecast service by @Zara"""
        _LOGGER.info("Service: debugging_6am_forecast (DEBUGGING)")
        try:
            await self.coordinator.set_expected_daily_production()
            _LOGGER.info("✓ 6 AM forecast logic executed: TODAY locked")
        except Exception as e:
            _LOGGER.error(f"✗ Error in debugging_6am_forecast: {e}", exc_info=True)

    async def _handle_debugging_best_hour(self, call: ServiceCall) -> None:
        """Handle debugging_best_hour service by @Zara"""
        _LOGGER.info("Service: debugging_best_hour (DEBUGGING)")
        try:
            best_hour, best_hour_kwh = await self.coordinator.best_hour_calculator.calculate_best_hour_today()

            if best_hour is not None and best_hour_kwh is not None:
                source = (
                    "Profile+Weather"
                    if self.coordinator.forecast_orchestrator.ml_strategy
                    and self.coordinator.forecast_orchestrator.ml_strategy.is_available()
                    else "Profile"
                )

                await self.coordinator.data_manager.save_forecast_best_hour(
                    hour=best_hour,
                    prediction_kwh=best_hour_kwh,
                    source=source,
                )
                _LOGGER.info(f"✓ Best hour saved: {best_hour}:00 with {best_hour_kwh:.3f} kWh (source: {source})")
                self.coordinator.async_update_listeners()
            else:
                _LOGGER.warning("✗ Could not calculate best hour - no profile data")
        except Exception as e:
            _LOGGER.error(f"✗ Error in debugging_best_hour: {e}", exc_info=True)

    async def _handle_debugging_tomorrow_12pm(self, call: ServiceCall) -> None:
        """Handle debugging_tomorrow_12pm service by @Zara"""
        _LOGGER.info("Service: debugging_tomorrow_12pm (DEBUGGING)")
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
                correction_factor=self.coordinator.learned_correction_factor
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
            self.coordinator.async_update_listeners()
        except Exception as e:
            _LOGGER.error(f"✗ Error in debugging_tomorrow_12pm: {e}", exc_info=True)

    async def _handle_debugging_day_after_tomorrow_6am(self, call: ServiceCall) -> None:
        """Handle debugging_day_after_tomorrow_6am service by @Zara"""
        _LOGGER.info("Service: debugging_day_after_tomorrow_6am (DEBUGGING)")
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
                correction_factor=self.coordinator.learned_correction_factor
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
            self.coordinator.async_update_listeners()
        except Exception as e:
            _LOGGER.error(f"✗ Error in debugging_day_after_tomorrow_6am: {e}", exc_info=True)

    async def _handle_debugging_day_after_tomorrow_6pm(self, call: ServiceCall) -> None:
        """Handle debugging_day_after_tomorrow_6pm service by @Zara"""
        _LOGGER.info("Service: debugging_day_after_tomorrow_6pm (DEBUGGING)")
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
                correction_factor=self.coordinator.learned_correction_factor
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
            self.coordinator.async_update_listeners()
        except Exception as e:
            _LOGGER.error(f"✗ Error in debugging_day_after_tomorrow_6pm: {e}", exc_info=True)

    async def _handle_night_cleanup(self, call: ServiceCall) -> None:
        """Handle night_cleanup service by @Zara"""
        _LOGGER.info("Service: night_cleanup (MANUAL TEST)")
        try:
            if hasattr(self.coordinator, 'scheduled_tasks'):
                await self.coordinator.scheduled_tasks.scheduled_night_cleanup(None)
                _LOGGER.info("Night cleanup completed")
            else:
                _LOGGER.error("Scheduled tasks manager not available")
        except Exception as e:
            _LOGGER.error(f"Error in night_cleanup: {e}", exc_info=True)

    async def _handle_collect_hourly_sample(self, call: ServiceCall) -> None:
        """Handle collect_hourly_sample service by @Zara"""
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
        """Handle run_all_scheduled_tasks service - COMPLETE TEST of all 18 scheduled tasks by @Zara This service runs ALL scheduled tasks in chronological order for debugging and testing. Updated: 2025-11-08 - Complete rewrite to match actual task schedule"""
        _LOGGER.info("="*80)
        _LOGGER.info("SERVICE: run_all_scheduled_tasks - COMPLETE TEST SEQUENCE (18 Tasks)")
        _LOGGER.info("="*80)

        task_count = 0
        success_count = 0
        failed_count = 0
        skipped_count = 0

        try:
            now = dt_util.now()

            # Check availability of components
            has_scheduled_tasks = hasattr(self.coordinator, 'scheduled_tasks')
            has_ml_predictor = hasattr(self.coordinator, 'ml_predictor') and self.coordinator.ml_predictor

            battery_enabled = self.entry.options.get(CONF_BATTERY_ENABLED, self.entry.data.get(CONF_BATTERY_ENABLED, False))
            battery_coordinator_key = f"{self.entry.entry_id}_battery"
            battery_coordinator = self.hass.data[DOMAIN].get(battery_coordinator_key) if battery_enabled else None

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
                    _LOGGER.warning(f"✗ Task {task_count} FAILED: Scheduled tasks manager not available")
                    failed_count += 1
            except Exception as e:
                _LOGGER.error(f"✗ Task {task_count} FAILED: End-of-Day Workflow - {e}", exc_info=True)
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
                    _LOGGER.warning(f"✗ Task {task_count} FAILED: Scheduled tasks manager not available")
                    failed_count += 1
            except Exception as e:
                _LOGGER.error(f"✗ Task {task_count} FAILED: Reset Expected Production - {e}", exc_info=True)
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
                _LOGGER.error(f"✗ Task {task_count} FAILED: Midnight Forecast Rotation - {e}", exc_info=True)
                failed_count += 1

            # =================================================================================
            # TASK 5: [03:00] Weekly ML Retraining (Sunday only)
            # =================================================================================
            task_count += 1
            _LOGGER.info(f"\nTASK {task_count}: [03:00] Weekly ML Retraining (Sunday only)")
            try:
                if has_ml_predictor:
                    if now.weekday() == 6:  # Sunday
                        await self.coordinator.ml_predictor.train_and_evaluate_model()
                        _LOGGER.info(f"✓ Task {task_count} completed: Weekly ML Retraining (Sunday)")
                        success_count += 1
                    else:
                        _LOGGER.info(f"⊘ Task {task_count} skipped: Not Sunday (today is {now.strftime('%A')})")
                        skipped_count += 1
                else:
                    _LOGGER.info(f"⊘ Task {task_count} skipped: ML Predictor not available")
                    skipped_count += 1
            except Exception as e:
                _LOGGER.error(f"✗ Task {task_count} FAILED: Weekly ML Retraining - {e}", exc_info=True)
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
                    _LOGGER.warning(f"✗ Task {task_count} FAILED: Scheduled tasks manager not available")
                    failed_count += 1
            except Exception as e:
                _LOGGER.error(f"✗ Task {task_count} FAILED: Morning Forecast Update - {e}", exc_info=True)
                failed_count += 1

            # =================================================================================
            # TASK 7-9: [06:15, 06:30, 06:45] Forecast Retry Attempts
            # =================================================================================
            for attempt in [1, 2, 3]:
                task_count += 1
                retry_minute = 15 * attempt
                _LOGGER.info(f"\nTASK {task_count}: [06:{retry_minute:02d}] Forecast Retry Attempt #{attempt}")
                try:
                    if has_scheduled_tasks:
                        # Simulate retry time
                        retry_time = now.replace(hour=6, minute=retry_minute, second=10)
                        await self.coordinator.scheduled_tasks.retry_forecast_setting(retry_time, attempt)
                        _LOGGER.info(f"✓ Task {task_count} completed: Forecast Retry #{attempt}")
                        success_count += 1
                    else:
                        _LOGGER.warning(f"✗ Task {task_count} FAILED: Scheduled tasks manager not available")
                        failed_count += 1
                except Exception as e:
                    _LOGGER.error(f"✗ Task {task_count} FAILED: Forecast Retry #{attempt} - {e}", exc_info=True)
                    failed_count += 1

            # =================================================================================
            # TASK 10: [XX:00:05] Next Hour Forecast Update (Test with current hour)
            # =================================================================================
            task_count += 1
            _LOGGER.info(f"\nTASK {task_count}: [XX:00:05] Next Hour Forecast Update")
            try:
                await self.coordinator._update_next_hour_forecast()
                _LOGGER.info(f"✓ Task {task_count} completed: Next Hour Forecast Update")
                success_count += 1
            except Exception as e:
                _LOGGER.error(f"✗ Task {task_count} FAILED: Next Hour Forecast Update - {e}", exc_info=True)
                failed_count += 1

            # =================================================================================
            # TASK 11: [XX:02] Hourly Sample Collection (Test with previous hour)
            # =================================================================================
            task_count += 1
            _LOGGER.info(f"\nTASK {task_count}: [XX:02] Hourly Sample Collection")
            try:
                if has_ml_predictor:
                    hour_to_collect = now - timedelta(hours=1)
                    await self.coordinator.ml_predictor.sample_collector.collect_sample(hour_to_collect)
                    _LOGGER.info(f"✓ Task {task_count} completed: Hourly Sample Collection")
                    success_count += 1
                else:
                    _LOGGER.info(f"⊘ Task {task_count} skipped: ML Predictor not available")
                    skipped_count += 1
            except Exception as e:
                _LOGGER.error(f"✗ Task {task_count} FAILED: Hourly Sample Collection - {e}", exc_info=True)
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
                    _LOGGER.error(f"✗ Task {task_count} FAILED: Battery Daily Rollup - {e}", exc_info=True)
                    failed_count += 1

                # Battery Task 2: Electricity Prices Refresh
                task_count += 1
                _LOGGER.info(f"\nTASK {task_count}: [BATTERY] Electricity Prices Refresh")
                try:
                    if hasattr(battery_coordinator, 'electricity_service') and battery_coordinator.electricity_service:
                        await battery_coordinator.async_refresh_prices()
                        _LOGGER.info(f"✓ Task {task_count} completed: Electricity Prices Refresh")
                        success_count += 1
                    else:
                        _LOGGER.info(f"⊘ Task {task_count} skipped: Electricity service not configured")
                        skipped_count += 1
                except Exception as e:
                    _LOGGER.error(f"✗ Task {task_count} FAILED: Electricity Prices Refresh - {e}", exc_info=True)
                    failed_count += 1
            else:
                _LOGGER.info(f"\n⊘ Battery Management Tasks skipped (battery disabled)")
                skipped_count += 2  # 2 battery tasks skipped

            # =================================================================================
            # FINAL SUMMARY
            # =================================================================================
            _LOGGER.info("="*80)
            _LOGGER.info("SERVICE: run_all_scheduled_tasks - COMPLETE TEST FINISHED!")
            _LOGGER.info("="*80)
            _LOGGER.info(f"Total Tasks:   {task_count}")
            _LOGGER.info(f"✓ Success:     {success_count}")
            _LOGGER.info(f"✗ Failed:      {failed_count}")
            _LOGGER.info(f"⊘ Skipped:     {skipped_count}")
            _LOGGER.info("="*80)

            if failed_count > 0:
                _LOGGER.warning(f"⚠️  {failed_count} task(s) FAILED - Check logs above for details")
            elif success_count == task_count:
                _LOGGER.info("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
            else:
                _LOGGER.info(f"✅ All executable tasks completed ({success_count}/{task_count - skipped_count})")

        except Exception as e:
            _LOGGER.error(f"CRITICAL ERROR in run_all_scheduled_tasks: {e}", exc_info=True)
