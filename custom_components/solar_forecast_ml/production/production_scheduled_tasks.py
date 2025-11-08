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
    """Manages scheduled tasks for the Solar Forecast ML integration by @Zara"""

    def __init__(
        self,
        hass: HomeAssistant,
        coordinator: 'SolarForecastMLCoordinator',
        solar_yield_today_entity_id: Optional[str],
        data_manager: DataManager
    ):
        """Initialize the ScheduledTasksManager by @Zara"""
        self.hass = hass
        self.coordinator = coordinator
        self.solar_yield_today_entity_id = solar_yield_today_entity_id
        self.data_manager = data_manager
        self._listeners = []

        _LOGGER.debug("ScheduledTasksManager initialized.")

    def setup_listeners(self) -> None:
        """Register the time-based listeners with Home Assistant by @Zara"""
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

        # 23:30 LOCAL - END_OF_DAY_WORKFLOW (Consolidated: Finalize + History + Stats + Cleanup)
        remove_end_of_day = async_track_time_change(
            self.hass,
            self.end_of_day_workflow,
            hour=23,
            minute=30,
            second=0
        )
        self._listeners.append(remove_end_of_day)
        _LOGGER.info(f"Scheduled END_OF_DAY_WORKFLOW for 23:30:00 LOCAL (Finalize+History+Stats+Cleanup)")


    def cancel_listeners(self) -> None:
        """Remove any active time-based listeners by @Zara"""
        for remove_listener in self._listeners:
            try:
                remove_listener()
            except Exception as e:
                _LOGGER.warning(f"Error removing scheduled task listener: {e}")
        self._listeners = []
        _LOGGER.debug("Cancelled scheduled task listeners.")


    async def calculate_yesterday_deviation_on_startup(self) -> None:
        """Calculates the forecast deviation from the previous day upon Home Assistant s... by @Zara"""
        _LOGGER.info("Calculating yesterday's forecast deviation at startup...")
        deviation: float = 0.0

        try:
            # get_predictions() returns List[Dict] directly, not Dict with 'predictions' key
            predictions: List[Dict[str, Any]] = await self.data_manager.get_predictions()

            if not predictions:
                _LOGGER.info("No prediction history available. Cannot calculate yesterday's deviation.")
                self.coordinator.last_day_error_kwh = deviation
                self.coordinator.yesterday_accuracy = 0.0
                return

            yesterday_local_date: date = (dt_util.now() - timedelta(days=1)).date() # now() already returns LOCAL time
            _LOGGER.debug(f"Looking for records from yesterday: {yesterday_local_date}")

            last_yesterday_record: Optional[Dict[str, Any]] = None
            for pred in reversed(predictions):
                try:
                    timestamp_utc = dt_util.parse_datetime(pred.get('timestamp', ''))
                    if not timestamp_utc: continue

                    pred_date_local = dt_util.as_local(timestamp_utc).date()

                    if pred_date_local == yesterday_local_date:
                         if pred.get('forecast_kwh') is not None and pred.get('actual_kwh') is not None:
                              last_yesterday_record = pred
                              break
                         else:
                              _LOGGER.debug(f"Record found for {yesterday_local_date} but missing forecast_kwh/actual_kwh value.")
                              break

                    elif pred_date_local < yesterday_local_date:
                        _LOGGER.debug("Reached records older than yesterday.")
                        break

                except (ValueError, KeyError, TypeError) as parse_error:
                    _LOGGER.debug(f"Skipping record due to parsing error: {parse_error}")
                    continue

            predicted_kwh: Optional[float] = None
            actual_kwh: Optional[float] = None
            if last_yesterday_record:
                try:
                    predicted_kwh = float(last_yesterday_record.get('forecast_kwh', 0.0))
                    actual_kwh = float(last_yesterday_record.get('actual_kwh', 0.0))
                except (ValueError, TypeError):
                    _LOGGER.warning("Last yesterday record has invalid forecast_kwh/actual_kwh values.")
                    predicted_kwh = None
                    actual_kwh = None

            if predicted_kwh is not None and actual_kwh is not None:
                deviation = abs(predicted_kwh - actual_kwh)

                accuracy = 0.0
                if actual_kwh > 0.1:
                    error_fraction = deviation / actual_kwh
                    accuracy = max(0.0, 1.0 - error_fraction)
                elif predicted_kwh < 0.1 and actual_kwh < 0.1:
                    accuracy = 1.0
                accuracy_pct = accuracy * 100.0

                _LOGGER.info(
                    f"Yesterday's Deviation on Startup:\n"
                    f"  Predicted:  {predicted_kwh:.2f} kWh\n"
                    f"  Actual:     {actual_kwh:.2f} kWh\n"
                    f"  Deviation:  {deviation:.2f} kWh\n"
                    f"  Accuracy:   {accuracy_pct:.1f}%"
                )

                self.coordinator.last_day_error_kwh = round(deviation, 2)
                self.coordinator.yesterday_accuracy = round(accuracy_pct, 1)
                self.coordinator.async_update_listeners()
            else:
                _LOGGER.info("No valid prediction/actual values found for yesterday. Deviation set to 0.")
                self.coordinator.last_day_error_kwh = 0.0
                self.coordinator.yesterday_accuracy = 0.0

        except Exception as e:
            _LOGGER.error(f"Error calculating yesterday's deviation on startup: {e}", exc_info=True)
            self.coordinator.last_day_error_kwh = 0.0
            self.coordinator.yesterday_accuracy = 0.0


    @callback
    async def scheduled_morning_update(self, now: datetime) -> None:
        """Callback for the scheduled morning task Triggers a full forecast update by @Zara"""
        _LOGGER.info(f"Triggering daily morning forecast update (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')})...")

        try:
            # First: Trigger coordinator refresh to get fresh forecast
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
        """Reset expected daily production at midnight by @Zara"""
        _LOGGER.info(f"Resetting expected daily production (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')})...")
        try:
            await self.coordinator.reset_expected_daily_production()
            _LOGGER.info("Expected daily production reset successful.")
        except Exception as e:
            _LOGGER.error(f"Failed to reset expected daily production: {e}", exc_info=True)

    @callback
    async def set_expected_production(self, now: datetime) -> None:
        """Set expected daily production at 6 AM by @Zara"""
        _LOGGER.info(f"=== Setting expected daily production (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')}) ===")
        try:
            # Check coordinator availability
            if not self.coordinator:
                _LOGGER.error("Coordinator not available!")
                return
                
            if not self.coordinator.data:
                _LOGGER.warning("Coordinator data is None, forcing refresh...")
                await self.coordinator.async_request_refresh()
                await asyncio.sleep(1.0)
            
            # Call coordinator method
            await self.coordinator.set_expected_daily_production()

            # Wait for async file write to complete
            await asyncio.sleep(0.1)

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
        """Retry mechanism for setting forecast if 0600 failed by @Zara"""
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
        """Consolidated End-of-Day Workflow at 23:30 - All tasks in sequence by @Zara"""
        current_time = now if now is not None else dt_util.now()
        _LOGGER.info(f"=== END_OF_DAY_WORKFLOW Started (Local Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}) ===")

        workflow_start = asyncio.get_event_loop().time()

        try:
            # STEP 1: Finalize Day (read solar_yield_today BEFORE midnight reset!)
            _LOGGER.info("Step 1/4: Finalizing day...")
            await self._finalize_day_internal(now)
            await asyncio.sleep(1)  # 1s buffer

            # STEP 2: Move to History
            _LOGGER.info("Step 2/4: Moving to history...")
            await self._move_to_history_internal(now)
            await asyncio.sleep(1)  # 1s buffer

            # STEP 2.5: Update Yesterday Deviation (after history is moved)
            _LOGGER.info("Step 2.5/4: Updating yesterday deviation...")
            await self._update_yesterday_deviation_internal(now)
            await asyncio.sleep(1)  # 1s buffer

            # STEP 3: Calculate Statistics
            _LOGGER.info("Step 3/4: Calculating statistics...")
            await self._calculate_stats_internal(now)
            await asyncio.sleep(1)  # 1s buffer

            # STEP 4: Night Cleanup
            _LOGGER.info("Step 4/4: Running night cleanup...")
            await self._night_cleanup_internal(now)

            workflow_duration = asyncio.get_event_loop().time() - workflow_start
            _LOGGER.info(f"=== END_OF_DAY_WORKFLOW Completed (Duration: {workflow_duration:.1f}s) ===")

        except Exception as e:
            _LOGGER.error(f"END_OF_DAY_WORKFLOW failed: {e}", exc_info=True)

    async def _finalize_day_internal(self, now: datetime) -> None:
        """Internal: Finalize current day with actual values by @Zara"""
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
            if hasattr(self.coordinator, 'production_time_calculator') and self.coordinator.production_time_calculator:
                production_seconds = int(self.coordinator.production_time_calculator.get_production_hours() * 3600)
                _LOGGER.debug(f"Production time: {production_seconds}s")

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
                _LOGGER.info(
                    f"✓ Day finalized: Yield={actual_yield:.2f} kWh, "
                    f"Consumption={f'{actual_consumption:.2f}' if actual_consumption else 'N/A'} kWh"
                )
            else:
                _LOGGER.error("Failed to finalize day")

        except Exception as e:
            _LOGGER.error(f"Failed to finalize day: {e}", exc_info=True)

    async def _move_to_history_internal(self, now: datetime) -> None:
        """Internal: Move current day to history by @Zara"""
        try:
            success = await self.data_manager.move_to_history()

            if success:
                _LOGGER.info("✓ Moved to history")
            else:
                _LOGGER.error("Failed to move day to history")

        except Exception as e:
            _LOGGER.error(f"Failed to move to history: {e}", exc_info=True)

    async def _update_yesterday_deviation_internal(self, now: datetime) -> None:
        """Internal: Update yesterday deviation after moving to history by @Zara"""
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
        """Internal: Calculate statistics by @Zara"""

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
        """Internal: Night cleanup - remove duplicates and zero-production samples by @Zara"""
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
        """Calculate and save the actual best production hour from todays hourly samples by @Zara"""
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

