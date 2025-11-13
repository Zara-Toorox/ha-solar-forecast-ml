"""
Scheduled Task Management for Production Tracking

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
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_change

from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..ml.ml_types import LearnedWeights, create_default_learned_weights
from ..data.data_manager import DataManager
from ..const import DAILY_UPDATE_HOUR, DAILY_VERIFICATION_HOUR, CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX

if TYPE_CHECKING:
    from ..coordinator import SolarForecastMLCoordinator

_LOGGER = logging.getLogger(__name__)


class ScheduledTasksManager:
    """Manages scheduled tasks for the Solar Forecast ML integration"""

    def __init__(
        self,
        hass: HomeAssistant,
        coordinator: 'SolarForecastMLCoordinator',
        solar_yield_today_entity_id: Optional[str],
        data_manager: DataManager
    ):
        """Initialize the ScheduledTasksManager"""
        self.hass = hass
        self.coordinator = coordinator
        self.solar_yield_today_entity_id = solar_yield_today_entity_id
        self.data_manager = data_manager
        self._listeners = []

        _LOGGER.debug("ScheduledTasksManager initialized.")

    def setup_listeners(self) -> None:
        """Register the time-based listeners with Home Assistant"""
        self.cancel_listeners()

        # Calculate timezone offset (async_track_time_change uses UTC!)
        local_now = dt_util.now()
        _LOGGER.info(f"All scheduled tasks use LOCAL time (async_track_time_change behavior)")

        # Morning forecast update at 6 AM LOCAL
        # async_track_time_change uses LOCAL time, not UTC!
        remove_morning = async_track_time_change(
            self.hass,
            self.scheduled_morning_update,
            hour=DAILY_UPDATE_HOUR,
            minute=0,
            second=0
        )
        self._listeners.append(remove_morning)
        _LOGGER.info(f"Scheduled morning forecast update for {DAILY_UPDATE_HOUR:02d}:00:00 LOCAL")

        # Evening verification at 18:00 LOCAL - DEPRECATED (optional, can be removed)
        # Removed in consolidation - finalize_day at 23:30 handles everything
        # remove_evening = async_track_time_change(...)

        # Reset expected daily production at midnight LOCAL
        remove_reset_expected = async_track_time_change(
            self.hass,
            self.reset_expected_production,
            hour=0,
            minute=0,
            second=0
        )
        self._listeners.append(remove_reset_expected)
        _LOGGER.info(f"Scheduled expected daily production reset for 00:00:00 LOCAL")

        # Midnight forecast rotation at 00:00:30 LOCAL (after reset, rotate forecasts)
        remove_midnight_rotation = async_track_time_change(
            self.hass,
            self.midnight_forecast_rotation,
            hour=0,
            minute=0,
            second=30
        )
        self._listeners.append(remove_midnight_rotation)
        _LOGGER.info(f"Scheduled midnight forecast rotation for 00:00:30 LOCAL")

        # 06:00:00 is already registered for morning_update above
        # set_expected_production is now integrated into scheduled_morning_update
        # No separate listener needed

        # Forecast retry tasks at 06:15, 06:30, 06:45 LOCAL
        retry_times = [15, 30, 45]  # Minutes after DAILY_UPDATE_HOUR
        for attempt, retry_minute in enumerate(retry_times, start=1):
            # Create a wrapper callback that properly awaits the async method
            @callback
            def make_retry_callback(retry_attempt: int):
                """Factory function to create retry callback with correct attempt number"""
                async def retry_callback(now: datetime) -> None:
                    """Wrapper that awaits the async retry method"""
                    await self.retry_forecast_setting(now, retry_attempt)
                return retry_callback

            remove_retry = async_track_time_change(
                self.hass,
                make_retry_callback(attempt),
                hour=DAILY_UPDATE_HOUR,
                minute=retry_minute,
                second=10
            )
            self._listeners.append(remove_retry)
            _LOGGER.info(
                f"Scheduled forecast retry #{attempt} for {DAILY_UPDATE_HOUR:02d}:{retry_minute:02d}:10 LOCAL"
            )

        # NEW: 6:00 LOCAL - Create Hourly Predictions (Single Source of Truth)
        remove_hourly_predictions = async_track_time_change(
            self.hass,
            self.create_morning_hourly_predictions,
            hour=6,
            minute=0,
            second=0
        )
        self._listeners.append(remove_hourly_predictions)
        _LOGGER.info(f"Scheduled HOURLY PREDICTIONS CREATION for 06:00:00 LOCAL")

        # HOURLY: Every hour at :05 - Update actual values for previous hour
        remove_hourly_actuals = async_track_time_change(
            self.hass,
            self.update_hourly_actuals,
            minute=5,
            second=0
        )
        self._listeners.append(remove_hourly_actuals)
        _LOGGER.info(f"Scheduled HOURLY ACTUALS UPDATE for every hour at :05:00 LOCAL")

        # 23:30 LOCAL - END_OF_DAY_WORKFLOW (Consolidated: Finalize + History + DailySummary + Stats + Cleanup)
        remove_end_of_day = async_track_time_change(
            self.hass,
            self.end_of_day_workflow,
            hour=23,
            minute=30,
            second=0
        )
        self._listeners.append(remove_end_of_day)
        _LOGGER.info(f"Scheduled END_OF_DAY_WORKFLOW for 23:30:00 LOCAL (Finalize+History+DailySummary+Stats+Cleanup)")


    def cancel_listeners(self) -> None:
        """Remove any active time-based listeners"""
        for remove_listener in self._listeners:
            try:
                remove_listener()
            except Exception as e:
                _LOGGER.warning(f"Error removing scheduled task listener: {e}")
        self._listeners = []
        _LOGGER.debug("Cancelled scheduled task listeners.")


    async def calculate_yesterday_deviation_on_startup(self) -> None:
        """Calculates the forecast deviation from yesterday using daily_forecasts.json"""
        _LOGGER.info("Calculating yesterday's forecast deviation at startup...")

        try:
            # Load daily_forecasts.json
            daily_forecasts = await self.data_manager.load_daily_forecasts()

            if not daily_forecasts:
                _LOGGER.info("No daily_forecasts.json available. Cannot calculate yesterday's deviation.")
                self.coordinator.last_day_error_kwh = 0.0
                self.coordinator.yesterday_accuracy = 0.0
                return

            # Get yesterday data
            yesterday = daily_forecasts.get('yesterday', {})

            if not yesterday:
                _LOGGER.info("No yesterday data in daily_forecasts.json")
                self.coordinator.last_day_error_kwh = 0.0
                self.coordinator.yesterday_accuracy = 0.0
                return

            # Extract forecast and actual values
            forecast_day = yesterday.get('forecast_day', {})
            actual_day = yesterday.get('actual_day', {})

            forecast_kwh = forecast_day.get('prediction_kwh')
            actual_kwh = actual_day.get('actual_kwh')

            if forecast_kwh is None or actual_kwh is None:
                _LOGGER.info("Yesterday data incomplete (missing forecast or actual)")
                self.coordinator.last_day_error_kwh = 0.0
                self.coordinator.yesterday_accuracy = 0.0
                return

            # Calculate deviation and accuracy
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

            # Also load monthly statistics
            statistics = daily_forecasts.get('statistics', {})
            current_month = statistics.get('current_month', {})
            if current_month and current_month.get('yield_kwh'):
                self.coordinator.avg_month_yield = round(current_month.get('yield_kwh', 0.0), 2)
                _LOGGER.info(f"Loaded monthly average yield: {self.coordinator.avg_month_yield:.2f} kWh")
            else:
                self.coordinator.avg_month_yield = 0.0

            self.coordinator.async_update_listeners()

        except Exception as e:
            _LOGGER.error(f"Error calculating yesterday's deviation on startup: {e}", exc_info=True)
            self.coordinator.last_day_error_kwh = 0.0
            self.coordinator.yesterday_accuracy = 0.0
            self.coordinator.avg_month_yield = 0.0


    @callback
    async def scheduled_morning_update(self, now: datetime) -> None:
        """Callback for the scheduled morning task Triggers a full forecast update"""
        _LOGGER.info(f"Triggering daily morning forecast update (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')})...")

        try:
            # Step 1: Refresh Astronomy Cache for today + next 7 days
            if hasattr(self.coordinator, 'service_registry') and self.coordinator.service_registry:
                try:
                    if hasattr(self.coordinator.service_registry, '_astronomy_handler'):
                        _LOGGER.info("Refreshing astronomy cache for today + next 7 days...")
                        await self.coordinator.service_registry._astronomy_handler.handle_refresh_cache_today(None)
                        _LOGGER.info("Astronomy cache refresh complete.")
                except Exception as e:
                    _LOGGER.warning(f"Astronomy cache refresh failed (non-critical): {e}")

            # Step 2: Trigger coordinator refresh to get fresh forecast
            await self.coordinator.async_request_refresh()
            _LOGGER.info("Morning forecast update request successful.")
            
            # Wait a moment for refresh to complete
            await asyncio.sleep(0.5)
            
            # Second: Set expected_daily_production from the fresh forecast
            if self.coordinator.data and "forecast_today" in self.coordinator.data:
                forecast_today = self.coordinator.data.get("forecast_today")
                if forecast_today is not None:
                    self.coordinator.expected_daily_production = forecast_today
                    
                    # Save to persistent storage
                    await self.data_manager.save_expected_daily_production(forecast_today)
                    
                    self.coordinator.async_update_listeners()
                    _LOGGER.info(
                        f"Morning update complete: Expected daily production set to {forecast_today:.2f} kWh "
                        f"(saved to persistent storage)"
                    )
                else:
                    _LOGGER.error("Morning update: forecast_today is None after refresh!")
            else:
                _LOGGER.error("Morning update: coordinator.data is missing or has no forecast_today!")

        except Exception as e:
            _LOGGER.error(f"Failed to complete morning forecast update: {e}", exc_info=True)


    # REMOVED: scheduled_evening_verification() - Deprecated and replaced by end_of_day_workflow
    # The evening verification functionality is now handled in:
    # - _finalize_day_internal() at 23:30 (reads today's yield)
    # - _update_yesterday_deviation_internal() at 23:30 (updates deviation after move_to_history)
    # No longer needed as standalone function.

    @callback
    async def scheduled_night_cleanup(self, now: datetime) -> None:
        """DEPRECATED: Callback for the scheduled night cleanup task - now part of end_of_day_workflow"""
        _LOGGER.warning("scheduled_night_cleanup called directly (deprecated, now part of end_of_day_workflow)")
        await self._night_cleanup_internal(now)

    @callback
    async def reset_expected_production(self, now: datetime) -> None:
        """Reset expected daily production at midnight"""
        _LOGGER.info(f"Resetting expected daily production (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')})...")
        try:
            await self.coordinator.reset_expected_daily_production()
            _LOGGER.info("Expected daily production reset successful.")
        except Exception as e:
            _LOGGER.error(f"Failed to reset expected daily production: {e}", exc_info=True)

    @callback
    async def midnight_forecast_rotation(self, now: datetime) -> None:
        """Rotate forecasts at midnight: tomorrow → today, day_after → tomorrow"""
        _LOGGER.info(f"=== MIDNIGHT FORECAST ROTATION (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')}) ===")
        try:
            success = await self.data_manager.rotate_forecasts_at_midnight()
            if success:
                _LOGGER.info("✓ Midnight forecast rotation completed successfully")
            else:
                _LOGGER.warning("Midnight forecast rotation failed")
        except Exception as e:
            _LOGGER.error(f"Failed to rotate forecasts at midnight: {e}", exc_info=True)

    @callback
    async def set_expected_production(self, now: datetime) -> None:
        """Set expected daily production at 6 AM"""
        _LOGGER.info(f"=== Setting expected daily production (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')}) ===")
        try:
            # Check coordinator availability
            if not self.coordinator:
                _LOGGER.error("Coordinator not available!")
                return

            if not self.coordinator.data:
                _LOGGER.warning("Coordinator data is None, forcing refresh...")
                await self.coordinator.async_request_refresh()

            # Call coordinator method (includes internal wait logic)
            await self.coordinator.set_expected_daily_production()

            # Validate that forecast was successfully saved
            saved_forecast = await self.data_manager.get_current_day_forecast()
            if saved_forecast and saved_forecast.get("locked") and saved_forecast.get("prediction_kwh"):
                _LOGGER.info(
                    f"Daily forecast validated: {saved_forecast.get('prediction_kwh')} kWh "
                    f"for {saved_forecast.get('date')}, locked=True"
                )
            else:
                _LOGGER.error(
                    "Daily forecast validation FAILED - forecast not properly saved!"
                )

            _LOGGER.info("Expected daily production set successful.")
        except Exception as e:
            _LOGGER.error(f"Failed to set expected daily production: {e}", exc_info=True)

    @callback
    async def retry_forecast_setting(self, now: datetime, attempt: int) -> None:
        """Retry mechanism for setting forecast if 0600 failed"""
        _LOGGER.info(f"=== Forecast Retry Attempt #{attempt} (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')}) ===")
        
        try:
            # Check if forecast is already locked
            current_forecast = await self.data_manager.get_current_day_forecast()

            forecast_day = current_forecast.get("forecast_day", {}) if current_forecast else {}
            if forecast_day.get("locked"):
                _LOGGER.info(
                    f"Forecast already locked with {forecast_day.get('prediction_kwh')} kWh - "
                    f"retry not needed"
                )
                return
            
            # Forecast not locked - initiate recovery
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

    # ==================================================================================
    # ==================================================================================

    @callback
    async def end_of_day_workflow(self, now: datetime) -> None:
        """Consolidated End-of-Day Workflow at 23:30 - All tasks in sequence"""
        current_time = now if now is not None else dt_util.now()

        _LOGGER.info(f"🌙 END_OF_DAY_WORKFLOW TRIGGERED at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        _LOGGER.info(f"=== END_OF_DAY_WORKFLOW Started (Local Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}) ===")

        workflow_start = asyncio.get_event_loop().time()

        steps_completed = 0
        total_steps = 6  # UPDATED: Added daily summary step
        errors = []

        # STEP 1: Finalize Day (read solar_yield_today BEFORE midnight reset!)
        try:
            _LOGGER.info("Step 1/6: Finalizing day...")
            await self._finalize_day_internal(now)
            steps_completed += 1
            _LOGGER.info("✓ Step 1/6 completed")
        except Exception as e:
            _LOGGER.error(f"✗ Step 1/6 failed: {e}", exc_info=True)
            errors.append(f"Finalize: {str(e)}")

        # STEP 2: Move to History
        try:
            _LOGGER.info("Step 2/6: Moving to history...")
            await self._move_to_history_internal(now)
            steps_completed += 1
            _LOGGER.info("✓ Step 2/6 completed")
        except Exception as e:
            _LOGGER.error(f"✗ Step 2/6 failed: {e}", exc_info=True)
            errors.append(f"History: {str(e)}")

        # STEP 3: Update Yesterday Deviation (after history is moved)
        try:
            _LOGGER.info("Step 3/6: Updating yesterday deviation...")
            await self._update_yesterday_deviation_internal(now)
            steps_completed += 1
            _LOGGER.info("✓ Step 3/6 completed")
        except Exception as e:
            _LOGGER.error(f"✗ Step 3/6 failed: {e}", exc_info=True)
            errors.append(f"Deviation: {str(e)}")

        # STEP 4: Create Daily Summary (NEW!)
        try:
            _LOGGER.info("Step 4/6: Creating daily summary (ML analysis)...")
            today = current_time.date().isoformat()

            # Get hourly predictions for today
            hourly_predictions = await self.coordinator.data_manager.hourly_predictions.get_predictions_for_date(today)

            if hourly_predictions:
                success = await self.coordinator.data_manager.daily_summaries.create_daily_summary(
                    date=today,
                    hourly_predictions=hourly_predictions
                )

                if success:
                    summary = await self.coordinator.data_manager.daily_summaries.get_summary(today)
                    _LOGGER.info(f"✓ Daily summary created:")
                    _LOGGER.info(f"  Overall accuracy: {summary['overall']['accuracy_percent']:.1f}%")
                    _LOGGER.info(f"  Patterns detected: {len(summary.get('patterns', []))}")
                    _LOGGER.info(f"  Recommendations: {len(summary.get('recommendations', []))}")

                    # Log important patterns
                    for pattern in summary.get('patterns', []):
                        _LOGGER.warning(
                            f"  ⚠️  Pattern detected: {pattern['type']} at hours {pattern['hours']} "
                            f"({pattern['severity']} severity)"
                        )
                else:
                    _LOGGER.warning("Daily summary creation returned False")
            else:
                _LOGGER.warning(f"No hourly predictions found for {today} - skipping summary")

            steps_completed += 1
            _LOGGER.info("✓ Step 4/6 completed")

        except Exception as e:
            _LOGGER.error(f"✗ Step 4/6 failed: {e}", exc_info=True)
            errors.append(f"Daily summary: {str(e)}")

        # STEP 5: Calculate Statistics
        try:
            _LOGGER.info("Step 5/6: Calculating statistics...")
            await self._calculate_stats_internal(now)
            steps_completed += 1
            _LOGGER.info("✓ Step 5/6 completed")
        except Exception as e:
            _LOGGER.error(f"✗ Step 5/6 failed: {e}", exc_info=True)
            errors.append(f"Statistics: {str(e)}")

        # STEP 6: Night Cleanup
        try:
            _LOGGER.info("Step 6/6: Running night cleanup...")
            await self._night_cleanup_internal(now)
            steps_completed += 1
            _LOGGER.info("✓ Step 6/6 completed")
        except Exception as e:
            _LOGGER.error(f"✗ Step 6/6 failed: {e}", exc_info=True)
            errors.append(f"Cleanup: {str(e)}")

        workflow_duration = asyncio.get_event_loop().time() - workflow_start

        if steps_completed == total_steps:
            _LOGGER.info(f"=== END_OF_DAY_WORKFLOW Completed Successfully ({steps_completed}/{total_steps} steps, {workflow_duration:.1f}s) ===")

            # Update system status sensor
            try:
                if self.coordinator:
                    self.coordinator.update_system_status(
                        event_type="end_of_day_workflow",
                        event_status="success",
                        event_summary="Tagesabschluss erfolgreich abgeschlossen",
                        event_details={
                            "duration_seconds": round(workflow_duration, 1),
                            "steps_completed": f"{steps_completed}/{total_steps}"
                        }
                    )
            except Exception as e:
                _LOGGER.warning(f"Failed to update system status: {e}")
        else:
            _LOGGER.warning(
                f"=== END_OF_DAY_WORKFLOW Completed with Errors ({steps_completed}/{total_steps} steps, {workflow_duration:.1f}s) ==="
            )
            _LOGGER.warning(f"Errors: {'; '.join(errors)}")

            # Update system status with partial success
            try:
                if self.coordinator:
                    self.coordinator.update_system_status(
                        event_type="end_of_day_workflow",
                        event_status="partial",
                        event_summary=f"Tagesabschluss teilweise erfolgreich ({steps_completed}/{total_steps})",
                        event_details={
                            "duration_seconds": round(workflow_duration, 1),
                            "steps_completed": f"{steps_completed}/{total_steps}",
                            "errors": errors
                        }
                    )
            except Exception as e:
                _LOGGER.warning(f"Failed to update system status: {e}")

    async def _finalize_day_internal(self, now: datetime) -> None:
        """Internal: Finalize current day with actual values"""
        current_time = now if now is not None else dt_util.now()

        try:
            # Get final yield value
            actual_yield = 0.0
            if self.solar_yield_today_entity_id:
                yield_state = self.hass.states.get(self.solar_yield_today_entity_id)
                if yield_state and yield_state.state not in (None, "unknown", "unavailable"):
                    try:
                        actual_yield = float(yield_state.state)
                        _LOGGER.debug(f"Final yield: {actual_yield:.2f} kWh")
                    except (ValueError, TypeError) as e:
                        _LOGGER.warning(f"Invalid yield state: {yield_state.state} - {e}")
                else:
                    _LOGGER.warning(f"Yield sensor unavailable: {self.solar_yield_today_entity_id}")
            else:
                _LOGGER.warning("No solar_yield_today sensor configured")

            # Get final consumption value (optional)
            actual_consumption = None
            if self.coordinator.total_consumption_today:
                consumption_state = self.hass.states.get(self.coordinator.total_consumption_today)
                if consumption_state and consumption_state.state not in (None, "unknown", "unavailable"):
                    try:
                        actual_consumption = float(consumption_state.state)
                        _LOGGER.debug(f"Final consumption: {actual_consumption:.2f} kWh")
                    except (ValueError, TypeError) as e:
                        _LOGGER.warning(f"Invalid consumption state: {consumption_state.state} - {e}")

            # Get production time in seconds
            production_seconds = 0
            try:
                if hasattr(self.coordinator, 'production_time_calculator') and self.coordinator.production_time_calculator:
                    production_hours = self.coordinator.production_time_calculator.get_production_hours()
                    if production_hours is not None:
                        production_seconds = int(production_hours * 3600)
                        _LOGGER.debug(f"Production time: {production_seconds}s ({production_hours:.2f}h)")
            except Exception as e:
                _LOGGER.warning(f"Failed to get production time: {e}")

            # Calculate and save actual best hour
            try:
                await self._save_actual_best_hour()
            except Exception as e:
                _LOGGER.warning(f"Failed to save actual best hour: {e}")

            # Finalize day in data_manager (NEW METHOD)
            success = await self.data_manager.finalize_today(
                yield_kwh=actual_yield,
                consumption_kwh=actual_consumption,
                production_seconds=production_seconds
            )

            if success:
                consumption_str = f"{actual_consumption:.2f}" if actual_consumption is not None else "N/A"
                _LOGGER.info(
                    f"✓ Day finalized: Yield={actual_yield:.2f} kWh, "
                    f"Consumption={consumption_str} kWh"
                )

                # Update monthly average yield from statistics
                try:
                    daily_forecasts = await self.data_manager.load_daily_forecasts()
                    if daily_forecasts:
                        statistics = daily_forecasts.get('statistics', {})
                        current_month = statistics.get('current_month', {})
                        if current_month and current_month.get('yield_kwh'):
                            self.coordinator.avg_month_yield = round(current_month.get('yield_kwh', 0.0), 2)
                            _LOGGER.info(f"✓ Updated monthly average yield: {self.coordinator.avg_month_yield:.2f} kWh")
                except Exception as stats_error:
                    _LOGGER.warning(f"Failed to update monthly statistics: {stats_error}")

                # Auto-generate daily forecast chart after day finalization
                try:
                    from ..services.service_chart_generator import ChartGenerator
                    chart_gen = ChartGenerator(self.data_manager.data_dir)
                    chart_path = await chart_gen.generate_daily_forecast_chart()
                    if chart_path:
                        _LOGGER.info(f"✓ Daily chart generated: {chart_path}")
                    else:
                        _LOGGER.debug("Daily chart generation skipped (no data or matplotlib unavailable)")
                except Exception as chart_error:
                    _LOGGER.warning(f"Failed to auto-generate daily chart: {chart_error}")

                # Also generate production_weather chart
                try:
                    from ..services.service_chart_generator import ChartGenerator
                    chart_gen = ChartGenerator(self.data_manager.data_dir)
                    weather_chart_path = await chart_gen.generate_production_weather_chart()
                    if weather_chart_path:
                        _LOGGER.info(f"✓ Production-Weather chart generated: {weather_chart_path}")
                    else:
                        _LOGGER.debug("Production-Weather chart generation skipped (no data or matplotlib unavailable)")
                except Exception as weather_error:
                    _LOGGER.warning(f"Failed to auto-generate production-weather chart: {weather_error}")
            else:
                _LOGGER.error("Failed to finalize day")

        except Exception as e:
            _LOGGER.error(f"Failed to finalize day: {e}", exc_info=True)

    async def _move_to_history_internal(self, now: datetime) -> None:
        """Internal: Move current day to history"""
        try:
            success = await self.data_manager.move_to_history()

            if success:
                _LOGGER.info("✓ Moved to history")
            else:
                _LOGGER.error("Failed to move day to history")

        except Exception as e:
            _LOGGER.error(f"Failed to move to history: {e}", exc_info=True)

    async def _update_yesterday_deviation_internal(self, now: datetime) -> None:
        """Internal: Update yesterday deviation after moving to history"""
        try:
            # Get history (most recent entry should be today which just moved)
            history = await self.data_manager.get_history(days=1)

            if not history or len(history) == 0:
                _LOGGER.warning("No history available to calculate yesterday deviation")
                self.coordinator.last_day_error_kwh = 0.0
                self.coordinator.yesterday_accuracy = 0.0
                return

            # Most recent entry is what we just finalized (today moved to history)
            today_entry = history[0]

            forecast_kwh = today_entry.get('forecast_kwh', 0.0)
            actual_kwh = today_entry.get('actual_kwh', 0.0)

            if forecast_kwh is None or actual_kwh is None:
                _LOGGER.warning("History entry missing forecast_kwh or actual_kwh")
                self.coordinator.last_day_error_kwh = 0.0
                self.coordinator.yesterday_accuracy = 0.0
                return

            # Calculate deviation and accuracy
            deviation = abs(forecast_kwh - actual_kwh)

            accuracy = 0.0
            if actual_kwh > 0.1:
                error_fraction = deviation / actual_kwh
                accuracy = max(0.0, 1.0 - error_fraction)
            elif forecast_kwh < 0.1 and actual_kwh < 0.1:
                accuracy = 1.0

            accuracy_pct = accuracy * 100.0

            # Update coordinator
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
        """Internal: Calculate statistics"""

        try:
            success = await self.data_manager.calculate_statistics()

            if success:
                _LOGGER.info("✓ Statistics calculated")

                # Log some key stats if available
                try:
                    data = await self.data_manager.load_daily_forecasts()
                    stats_7d = data.get("statistics", {}).get("last_7_days", {})
                    if stats_7d:
                        _LOGGER.debug(
                            f"Last 7 days: Avg yield={stats_7d.get('avg_yield', 0):.2f} kWh, "
                            f"Avg accuracy={stats_7d.get('avg_accuracy', 0):.1f}%"
                        )
                except Exception:
                    pass  # Non-critical
            else:
                _LOGGER.error("Failed to calculate statistics")

        except Exception as e:
            _LOGGER.error(f"Failed to calculate statistics: {e}", exc_info=True)

    async def _night_cleanup_internal(self, now: datetime) -> None:
        """Internal: Night cleanup - remove duplicates and zero-production samples"""
        try:
            # Step 1: Remove duplicate samples
            duplicate_result = await self.data_manager.cleanup_duplicate_samples()
            _LOGGER.debug(
                f"Duplicate cleanup: {duplicate_result['removed']} removed, "
                f"{duplicate_result['remaining']} remaining"
            )

            # Step 2: Remove zero-production samples
            zero_result = await self.data_manager.cleanup_zero_production_samples()
            _LOGGER.debug(
                f"Zero-production cleanup: {zero_result['removed']} removed, "
                f"{zero_result['remaining']} remaining"
            )

            # Summary
            total_removed = duplicate_result['removed'] + zero_result['removed']
            _LOGGER.info(
                f"✓ Night cleanup: {total_removed} samples removed, "
                f"{zero_result['remaining']} remaining"
            )

        except Exception as e:
            _LOGGER.error(f"Failed to execute night cleanup: {e}", exc_info=True)

    # ==================================================================================
    # DEPRECATED: Old separate task functions (kept for service calls compatibility)
    # ==================================================================================

    @callback
    async def finalize_day_task(self, now: datetime) -> None:
        """DEPRECATED: Use end_of_day_workflow instead"""
        _LOGGER.warning("finalize_day_task called directly (deprecated, use end_of_day_workflow)")
        await self._finalize_day_internal(now)

    @callback
    async def move_to_history_task(self, now: datetime) -> None:
        """DEPRECATED: Use end_of_day_workflow instead"""
        _LOGGER.warning("move_to_history_task called directly (deprecated, use end_of_day_workflow)")
        await self._move_to_history_internal(now)

    @callback
    async def calculate_stats_task(self, now: datetime) -> None:
        """DEPRECATED: Use end_of_day_workflow instead"""
        _LOGGER.warning("calculate_stats_task called directly (deprecated, use end_of_day_workflow)")
        await self._calculate_stats_internal(now)

    async def _save_actual_best_hour(self) -> None:
        """Calculate and save the actual best production hour from todays hourly samples"""
        try:
            today = dt_util.now().date()
            hourly_samples = await self.data_manager.get_hourly_samples()

            # Filter samples from today
            today_samples = [
                s for s in hourly_samples
                if dt_util.parse_datetime(s.get("timestamp")).date() == today
            ]

            if not today_samples:
                _LOGGER.debug("No hourly samples from today - skipping actual best hour calculation")
                return

            # Find the hour with maximum production
            max_production = -1
            best_hour = None
            best_kwh = 0.0

            for sample in today_samples:
                production = sample.get("actual_kwh")
                if production is not None and production > max_production:
                    max_production = production
                    best_kwh = production
                    timestamp = dt_util.parse_datetime(sample.get("timestamp"))
                    if timestamp:
                        best_hour = timestamp.hour

            if best_hour is not None and best_kwh > 0:
                # Save to daily_forecasts.json
                success = await self.data_manager.save_actual_best_hour(
                    hour=best_hour,
                    actual_kwh=best_kwh
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

    # ==================================================================================
    # NEW: HOURLY PREDICTIONS TASKS
    # ==================================================================================

    @callback
    async def create_morning_hourly_predictions(self, now: datetime) -> None:
        """
        MORNING TASK (6:00 AM): Create hourly predictions for today

        This is the SINGLE SOURCE OF TRUTH for today's predictions.
        All other services read from this data.
        """
        current_time = now if now is not None else dt_util.now()

        _LOGGER.info(f"🌅 MORNING HOURLY PREDICTIONS TASK at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        _LOGGER.info("="*80)
        _LOGGER.info("Creating hourly predictions for today (Single Source of Truth)")
        _LOGGER.info("="*80)

        try:
            today = current_time.date().isoformat()

            # Step 1: Get weather forecast
            _LOGGER.info("Step 1/5: Fetching weather forecast...")
            weather_service = self.coordinator.weather_service
            if not weather_service:
                _LOGGER.error("✗ Weather service not available - ABORTING")
                return

            current_weather = await weather_service.get_current_weather()
            hourly_weather_forecast = await weather_service.get_processed_hourly_forecast()

            if not hourly_weather_forecast:
                _LOGGER.error("✗ No hourly weather forecast available - ABORTING")
                return

            _LOGGER.info(f"✓ Weather forecast retrieved: {len(hourly_weather_forecast)} hours")

            # Step 2: Get sensor configuration
            _LOGGER.info("Step 2/5: Collecting sensor configuration...")
            sensor_config = {
                "temperature": hasattr(self.coordinator, 'temp_sensor') and self.coordinator.temp_sensor is not None,
                "humidity": hasattr(self.coordinator, 'humidity_sensor') and self.coordinator.humidity_sensor is not None,
                "lux": hasattr(self.coordinator, 'lux_sensor') and self.coordinator.lux_sensor is not None,
                "rain": hasattr(self.coordinator, 'rain_sensor') and self.coordinator.rain_sensor is not None,
                "uv_index": hasattr(self.coordinator, 'uv_sensor') and self.coordinator.uv_sensor is not None,
                "wind_speed": hasattr(self.coordinator, 'wind_sensor') and self.coordinator.wind_sensor is not None,
            }
            _LOGGER.info(f"✓ Sensor config: {sum(sensor_config.values())}/{len(sensor_config)} sensors available")

            # Step 3: Create forecast with ML
            _LOGGER.info("Step 3/5: Generating ML forecast...")
            external_sensors = self.coordinator.sensor_collector.collect_all_sensor_data_dict()

            forecast = await self.coordinator.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_weather_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=self.coordinator.learned_correction_factor
            )

            if not forecast or not forecast.get("hourly"):
                _LOGGER.error("✗ Forecast generation failed - no hourly data")
                return

            _LOGGER.info(f"✓ Forecast generated: {len(forecast.get('hourly', []))} hourly predictions")
            _LOGGER.info(f"  Today total: {forecast.get('today', 0):.2f} kWh")
            _LOGGER.info(f"  Method: {forecast.get('method', 'unknown')}")

            # Step 4: Get astronomy data
            _LOGGER.info("Step 4/5: Calculating astronomy data...")
            astronomy_data = await self._get_astronomy_data(current_time)
            _LOGGER.info(f"✓ Astronomy data calculated")

            # Step 5: Create hourly predictions (ONLY FOR TODAY)
            _LOGGER.info("Step 5/5: Creating hourly predictions for TODAY only...")

            # Filter hourly_forecast to only include TODAY's hours
            all_hourly = forecast.get("hourly", [])
            today_hourly = [h for h in all_hourly if h.get("date") == today]

            _LOGGER.info(f"Filtered hourly forecast: {len(all_hourly)} total → {len(today_hourly)} for today")

            success = await self.coordinator.data_manager.hourly_predictions.create_daily_predictions(
                date=today,
                hourly_forecast=today_hourly,
                weather_forecast=hourly_weather_forecast,
                astronomy_data=astronomy_data,
                sensor_config=sensor_config
            )

            if success:
                predictions = await self.coordinator.data_manager.hourly_predictions.get_predictions_for_date(today)
                _LOGGER.info(f"\n✓✓✓ HOURLY PREDICTIONS CREATED SUCCESSFULLY ✓✓✓")
                _LOGGER.info(f"  Date: {today}")
                _LOGGER.info(f"  Total predictions: {len(predictions)}")
                _LOGGER.info(f"  Total predicted: {sum(p['predicted_kwh'] for p in predictions):.2f} kWh")

                # Show hourly breakdown
                _LOGGER.info(f"\n  Hourly breakdown:")
                for p in predictions:
                    _LOGGER.info(
                        f"    {p['target_hour']:02d}:00 - "
                        f"{p['predicted_kwh']:.3f} kWh - "
                        f"Conf: {p['confidence']:.0f}% - "
                        f"{'⭐ PEAK' if p['flags']['is_peak_hour'] else ''}"
                    )

                # NOTE: Do NOT save to daily_forecasts.json here!
                # The scheduled_morning_update task (also at 06:00) already handles that.
                # Saving twice would cause "already locked" warning.
                _LOGGER.debug(f"  Daily forecast ({forecast['today']:.2f} kWh) is saved by scheduled_morning_update task")

                try:
                    hourly_data = await self.coordinator.data_manager.hourly_predictions._read_json_async()
                    self.coordinator._hourly_predictions_cache = hourly_data
                except Exception as cache_err:
                    _LOGGER.debug(f"Failed to update hourly predictions cache: {cache_err}")

                # Trigger sensor updates (NextHourSensor and PeakProductionHourSensor read from hourly_predictions.json)
                self.coordinator.async_update_listeners()
                _LOGGER.info(f"  ✓ Sensors notified to reload from hourly_predictions.json")
            else:
                _LOGGER.error("✗ Failed to create hourly predictions")

            _LOGGER.info("="*80)

        except Exception as e:
            _LOGGER.error(f"✗ Error in create_morning_hourly_predictions: {e}", exc_info=True)

    async def _get_astronomy_data(self, dt: datetime) -> Dict[str, Any]:
        """
        Get astronomy data for the day from astronomy cache

        Falls back to sun.sun entity if cache is unavailable
        """
        try:
            # Try astronomy cache first
            target_date = dt.date()
            astronomy_cache_file = (
                self.coordinator.data_manager.data_dir /
                "stats" /
                "astronomy_cache.json"
            )

            if astronomy_cache_file.exists():
                def _read_sync():
                    import json
                    with open(astronomy_cache_file, 'r') as f:
                        return json.load(f)

                import asyncio
                loop = asyncio.get_running_loop()
                cache = await loop.run_in_executor(None, _read_sync)

                # Get astronomy data for target date
                date_str = target_date.isoformat()
                day_data = cache.get("days", {}).get(date_str)

                if day_data:
                    return {
                        "sunrise": day_data.get("sunrise_local"),
                        "sunset": day_data.get("sunset_local"),
                        "solar_noon": day_data.get("solar_noon_local"),
                        "daylight_hours": day_data.get("daylight_hours")
                    }

            # Fallback to sun.sun entity if cache unavailable
            _LOGGER.debug("Astronomy cache unavailable, falling back to sun.sun entity")
            sun_entity = self.hass.states.get("sun.sun")

            if sun_entity:
                sunrise_str = sun_entity.attributes.get("next_rising")
                sunset_str = sun_entity.attributes.get("next_setting")

                if sunrise_str and sunset_str:
                    sunrise = datetime.fromisoformat(sunrise_str.replace('Z', '+00:00'))
                    sunset = datetime.fromisoformat(sunset_str.replace('Z', '+00:00'))

                    # Calculate solar noon (midpoint)
                    solar_noon = sunrise + (sunset - sunrise) / 2

                    # Calculate daylight hours
                    daylight_hours = (sunset - sunrise).total_seconds() / 3600

                    return {
                        "sunrise": sunrise.isoformat(),
                        "sunset": sunset.isoformat(),
                        "solar_noon": solar_noon.isoformat(),
                        "daylight_hours": round(daylight_hours, 2)
                    }

            return {}

        except Exception as e:
            _LOGGER.warning(f"Failed to get astronomy data: {e}")
            return {}

    async def _get_production_window_from_cache(self, target_date: date) -> Optional[tuple]:
        """
        Get production window from astronomy cache

        Args:
            target_date: Date to get window for

        Returns:
            Tuple of (start_time, end_time) as naive datetime objects, or None if not available
        """
        try:
            # Check if astronomy cache exists
            astronomy_cache_file = (
                self.coordinator.data_manager.data_dir /
                "stats" /
                "astronomy_cache.json"
            )

            if not astronomy_cache_file.exists():
                return None

            # Read cache
            def _read_sync():
                import json
                with open(astronomy_cache_file, 'r') as f:
                    return json.load(f)

            import asyncio
            loop = asyncio.get_running_loop()
            cache = await loop.run_in_executor(None, _read_sync)

            # Get production window for target date
            date_str = target_date.isoformat()
            day_data = cache.get("days", {}).get(date_str)

            if not day_data:
                return None

            # Get production window times (already in Local Time with timezone)
            window_start_str = day_data.get("production_window_start")
            window_end_str = day_data.get("production_window_end")

            if not window_start_str or not window_end_str:
                return None

            # Parse ISO timestamps and convert to naive local time
            window_start = datetime.fromisoformat(window_start_str).replace(tzinfo=None)
            window_end = datetime.fromisoformat(window_end_str).replace(tzinfo=None)

            _LOGGER.debug(
                f"Production window from cache: {window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}"
            )

            return (window_start, window_end)

        except Exception as e:
            _LOGGER.debug(f"Failed to get production window from cache: {e}")
            return None

    def _get_production_window_from_sun_entity(self, current_time: datetime) -> Optional[tuple]:
        """
        Fallback: Get production window from sun.sun entity

        Args:
            current_time: Current time with timezone

        Returns:
            Tuple of (start_time, end_time) as naive datetime objects, or None if not available
        """
        try:
            sun_entity = self.hass.states.get("sun.sun")
            if not sun_entity:
                return None

            # Use last_rising (today's sunrise) and next_setting (today's sunset)
            sunrise_str = sun_entity.attributes.get("last_rising")
            sunset_str = sun_entity.attributes.get("next_setting")

            if not sunrise_str or not sunset_str:
                return None

            # Parse and convert to local time
            sunrise = datetime.fromisoformat(sunrise_str.replace('Z', '+00:00'))
            sunset = datetime.fromisoformat(sunset_str.replace('Z', '+00:00'))

            sunrise_local = sunrise.astimezone(current_time.tzinfo).replace(tzinfo=None)
            sunset_local = sunset.astimezone(current_time.tzinfo).replace(tzinfo=None)

            # Production window: Sunrise - 1h to Sunset + 1h
            production_start = sunrise_local - timedelta(hours=1)
            production_end = sunset_local + timedelta(hours=1)

            _LOGGER.debug(
                f"Production window from sun.sun: {production_start.strftime('%H:%M')} - {production_end.strftime('%H:%M')}"
            )

            return (production_start, production_end)

        except Exception as e:
            _LOGGER.debug(f"Failed to get production window from sun.sun: {e}")
            return None

    @callback
    async def update_hourly_actuals(self, now: datetime) -> None:
        """
        HOURLY TASK (every hour at :05): Update actual values for previous hour

        Example: At 12:05, update actual values for 11:00-12:00

        Only runs during production time window (Sunrise - 1h to Sunset + 1h)
        """
        current_time = now if now is not None else dt_util.now()
        _LOGGER.debug(f"🔔 update_hourly_actuals() called at {current_time.strftime('%H:%M:%S')}")
        _LOGGER.debug(f"   current_time.tzinfo = {current_time.tzinfo}")

        # Check if we're in production time window using astronomy cache
        production_window = await self._get_production_window_from_cache(current_time.date())

        if not production_window:
            # Fallback to sun.sun entity if astronomy cache not available
            _LOGGER.debug("Astronomy cache not available, falling back to sun.sun entity")
            production_window = self._get_production_window_from_sun_entity(current_time)

            if not production_window:
                # Last resort fallback: Use conservative default window based on season
                month = current_time.month
                # Northern hemisphere assumption - adjust for latitude if needed
                if 4 <= month <= 9:  # Summer (April-September): 05:00-21:00
                    default_start = current_time.replace(hour=5, minute=0, second=0, microsecond=0, tzinfo=None)
                    default_end = current_time.replace(hour=21, minute=0, second=0, microsecond=0, tzinfo=None)
                else:  # Winter (October-March): 06:30-17:30
                    default_start = current_time.replace(hour=6, minute=30, second=0, microsecond=0, tzinfo=None)
                    default_end = current_time.replace(hour=17, minute=30, second=0, microsecond=0, tzinfo=None)

                production_window = (default_start, default_end)
                _LOGGER.warning(
                    f"⏰ Using fallback production window (cache and sun.sun unavailable): "
                    f"{default_start.strftime('%H:%M')} - {default_end.strftime('%H:%M')}"
                )

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

        # Calculate previous hour
        previous_hour_dt = current_time - timedelta(hours=1)
        previous_hour = previous_hour_dt.hour
        today = current_time.date().isoformat()

        _LOGGER.info(f"⏰ HOURLY UPDATE at {current_time.strftime('%H:%M:%S')} - Updating actual for {previous_hour:02d}:00-{current_time.hour:02d}:00")

        try:
            # Get current yield sensor value from solar_yield_today entity
            current_yield = None
            if self.solar_yield_today_entity_id:
                yield_state = self.hass.states.get(self.solar_yield_today_entity_id)
                if yield_state and yield_state.state not in (None, "unknown", "unavailable"):
                    try:
                        current_yield = float(yield_state.state)
                    except (ValueError, TypeError) as e:
                        _LOGGER.warning(f"Invalid yield sensor value: {yield_state.state} - {e}")

            if current_yield is None:
                _LOGGER.warning(f"Cannot update hourly actual: yield sensor not available or invalid ({self.solar_yield_today_entity_id})")
                return

            # Get previous yield from cache
            if not hasattr(self.coordinator, '_last_yield_cache'):
                self.coordinator._last_yield_cache = {}

            previous_yield = self.coordinator._last_yield_cache.get('value', None)
            previous_yield_time = self.coordinator._last_yield_cache.get('time', None)

            # Store current yield for next hour
            self.coordinator._last_yield_cache = {
                'value': current_yield,
                'time': current_time
            }

            if previous_yield is None:
                _LOGGER.info(f"First hourly update - caching yield: {current_yield:.3f} kWh (no actual update yet)")
                return

            # Calculate hourly production (handle midnight rollover)
            if current_yield >= previous_yield:
                actual_kwh = current_yield - previous_yield
            else:
                # Yield sensor reset (e.g. midnight or restart) - skip this hour
                _LOGGER.warning(f"Yield sensor decreased ({previous_yield:.3f} → {current_yield:.3f} kWh) - likely reset, skipping hour")
                return

            _LOGGER.info(f"Hourly production {previous_hour:02d}:00-{current_time.hour:02d}:00: {actual_kwh:.3f} kWh (Yield: {previous_yield:.3f} → {current_yield:.3f})")

            # Collect and map sensor data
            sensor_data_raw = self.coordinator.sensor_collector.collect_all_sensor_data_dict()
            sensor_data = {
                "temperature_c": sensor_data_raw.get("temperature"),
                "humidity_percent": sensor_data_raw.get("humidity"),
                "lux": sensor_data_raw.get("lux"),
                "rain_mm": sensor_data_raw.get("rain"),
                "uv_index": sensor_data_raw.get("uv_index"),
                "wind_speed_ms": sensor_data_raw.get("wind_speed"),
                "current_power_w": None,
                "current_yield_kwh": current_yield
            }

            # Get weather data
            weather_actual = None
            if self.coordinator.weather_service:
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
            success = await self.coordinator.data_manager.hourly_predictions.update_hourly_actual(
                date=today,
                hour=previous_hour,
                actual_kwh=actual_kwh,
                sensor_data=sensor_data,
                weather_actual=weather_actual,
                astronomy_update=astronomy_update
            )

            if success:
                _LOGGER.info(f"✓ Updated actual for hour {previous_hour:02d}: {actual_kwh:.3f} kWh")

                try:
                    hourly_data = await self.coordinator.data_manager.hourly_predictions._read_json_async()
                    self.coordinator._hourly_predictions_cache = hourly_data
                except Exception as cache_err:
                    _LOGGER.debug(f"Failed to update hourly predictions cache: {cache_err}")

                self.coordinator.async_update_listeners()
            else:
                _LOGGER.warning(f"✗ Failed to update actual for hour {previous_hour:02d}")

        except Exception as e:
            _LOGGER.error(f"Error in update_hourly_actuals: {e}", exc_info=True)

    async def _calculate_astronomy_for_hour(self, dt: datetime) -> Dict[str, Any]:
        """Calculate astronomy features for ML training"""
        try:
            import math
            from astral import LocationInfo
            from astral.sun import sun, elevation, azimuth

            sun_entity = self.hass.states.get("sun.sun")

            if not sun_entity:
                return {}

            sunrise_str = sun_entity.attributes.get("next_rising") or sun_entity.attributes.get("last_rising")
            sunset_str = sun_entity.attributes.get("next_setting")

            if not sunrise_str or not sunset_str:
                return {}

            sunrise = datetime.fromisoformat(sunrise_str.replace('Z', '+00:00'))
            sunset = datetime.fromisoformat(sunset_str.replace('Z', '+00:00'))

            dt_local = dt.replace(tzinfo=None) if dt.tzinfo else dt
            sunrise_local = sunrise.astimezone(dt.tzinfo).replace(tzinfo=None) if dt.tzinfo else sunrise.replace(tzinfo=None)
            sunset_local = sunset.astimezone(dt.tzinfo).replace(tzinfo=None) if dt.tzinfo else sunset.replace(tzinfo=None)

            hours_after_sunrise = (dt_local - sunrise_local).total_seconds() / 3600 if dt_local > sunrise_local else 0
            hours_before_sunset = (sunset_local - dt_local).total_seconds() / 3600 if dt_local < sunset_local else 0

            solar_noon = sunrise_local + (sunset_local - sunrise_local) / 2
            hours_from_noon = abs((dt_local - solar_noon).total_seconds() / 3600)
            daylight_hours = (sunset_local - sunrise_local).total_seconds() / 3600

            if sunrise_local <= dt_local <= sunset_local:
                max_elevation = 90 - abs(dt.timetuple().tm_yday - 172) * 0.4
                sun_elevation_deg = max_elevation * (1 - (hours_from_noon / (daylight_hours / 2)) ** 2)
            else:
                sun_elevation_deg = 0.0

            return {
                "sun_elevation_deg": round(sun_elevation_deg, 2),
                "hours_after_sunrise": round(hours_after_sunrise, 2),
                "hours_before_sunset": round(hours_before_sunset, 2)
            }

        except Exception as e:
            _LOGGER.debug(f"Failed to calculate astronomy for hour: {e}")
            return {}

