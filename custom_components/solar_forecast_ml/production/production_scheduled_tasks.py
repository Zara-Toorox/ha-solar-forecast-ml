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
    """Manages scheduled tasks for the Solar Forecast ML integration by Zara"""

    def __init__(
        self,
        hass: HomeAssistant,
        coordinator: 'SolarForecastMLCoordinator',
        solar_yield_today_entity_id: Optional[str],
        data_manager: DataManager
    ):
        """Initialize the ScheduledTasksManager by Zara"""
        self.hass = hass
        self.coordinator = coordinator
        self.solar_yield_today_entity_id = solar_yield_today_entity_id
        self.data_manager = data_manager
        self._listeners = []

        _LOGGER.debug("ScheduledTasksManager initialized.")

    def setup_listeners(self) -> None:
        """Register the time-based listeners with Home Assistant by Zara"""
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

        # Evening verification at 18:00 LOCAL
        remove_evening = async_track_time_change(
            self.hass,
            self.scheduled_evening_verification,
            hour=DAILY_VERIFICATION_HOUR,
            minute=0,
            second=10
        )
        self._listeners.append(remove_evening)
        _LOGGER.info(f"Scheduled evening verification for {DAILY_VERIFICATION_HOUR:02d}:00:10 LOCAL")

        # Night cleanup at 2 AM LOCAL
        remove_cleanup = async_track_time_change(
            self.hass,
            self.scheduled_night_cleanup,
            hour=2,
            minute=0,
            second=15
        )
        self._listeners.append(remove_cleanup)
        _LOGGER.info(f"Scheduled night cleanup for 02:00:15 LOCAL")

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

        # Set expected daily production at 6 AM LOCAL
        remove_set_expected = async_track_time_change(
            self.hass,
            self.set_expected_production,
            hour=6,
            minute=0,
            second=10
        )
        self._listeners.append(remove_set_expected)
        _LOGGER.info(f"Scheduled expected daily production set for 06:00:10 LOCAL")

        # Forecast retry tasks at 06:15, 06:30, 06:45 LOCAL
        retry_times = [15, 30, 45]  # Minutes after DAILY_UPDATE_HOUR
        for attempt, retry_minute in enumerate(retry_times, start=1):
            remove_retry = async_track_time_change(
                self.hass,
                lambda now, attempt=attempt: self.retry_forecast_setting(now, attempt=attempt),
                hour=DAILY_UPDATE_HOUR,
                minute=retry_minute,
                second=10
            )
            self._listeners.append(remove_retry)
            _LOGGER.info(
                f"Scheduled forecast retry #{attempt} for {DAILY_UPDATE_HOUR:02d}:{retry_minute:02d}:10 LOCAL"
            )

        # Daily forecasts system at 23:30-23:32 LOCAL

        # 23:30 LOCAL - Finalize current day
        remove_finalize = async_track_time_change(
            self.hass,
            self.finalize_day_task,
            hour=23,
            minute=30,
            second=0
        )
        self._listeners.append(remove_finalize)
        _LOGGER.info(f"Scheduled day finalization for 23:30:00 LOCAL")

        # 23:31 LOCAL - Move current day to history
        remove_history = async_track_time_change(
            self.hass,
            self.move_to_history_task,
            hour=23,
            minute=31,
            second=0
        )
        self._listeners.append(remove_history)
        _LOGGER.info(f"Scheduled history update for 23:31:00 LOCAL")

        # 23:32 LOCAL - Calculate statistics
        remove_stats = async_track_time_change(
            self.hass,
            self.calculate_stats_task,
            hour=23,
            minute=32,
            second=0
        )
        self._listeners.append(remove_stats)
        _LOGGER.info(f"Scheduled statistics calculation for 23:32:00 LOCAL")


    def cancel_listeners(self) -> None:
        """Remove any active time-based listeners by Zara"""
        for remove_listener in self._listeners:
            try:
                remove_listener()
            except Exception as e:
                _LOGGER.warning(f"Error removing scheduled task listener: {e}")
        self._listeners = []
        _LOGGER.debug("Cancelled scheduled task listeners.")


    async def calculate_yesterday_deviation_on_startup(self) -> None:
        """Calculates the forecast deviation from the previous day upon Home Assistant s... by Zara"""
        _LOGGER.info("Calculating yesterday's forecast deviation at startup...")
        deviation: float = 0.0

        try:
            # FIX: Use get_predictions() instead of get_prediction_history()
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
                         if pred.get('predicted_value') is not None and pred.get('actual_value') is not None:
                              last_yesterday_record = pred
                              break
                         else:
                              _LOGGER.debug(f"Record found for {yesterday_local_date} but missing predicted/actual value.")
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
                    predicted_kwh = float(last_yesterday_record.get('predicted_value', 0.0))
                    actual_kwh = float(last_yesterday_record.get('actual_value', 0.0))
                except (ValueError, TypeError):
                    _LOGGER.warning("Last yesterday record has invalid predicted/actual values.")
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
        """Callback for the scheduled morning task Triggers a full forecast update by Zara"""
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


    @callback
    async def scheduled_evening_verification(self, now: datetime) -> None:
        """Callback for the scheduled evening task Verifies todays forecast by Zara"""
        _LOGGER.info(f"Starting daily evening verification (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')})...")

        if not self.solar_yield_today_entity_id:
            _LOGGER.warning("Cannot perform evening verification: Solar Yield Today sensor not configured.")
            return

        actual_kwh: Optional[float] = None
        try:
            state = self.hass.states.get(self.solar_yield_today_entity_id)

            if not state or state.state in ['unavailable', 'unknown', 'none', None, '']:
                _LOGGER.error(f"Evening verification failed: Yield sensor '{self.solar_yield_today_entity_id}' is unavailable or state is invalid ('{state.state if state else 'None'}').")
                return

            try:
                cleaned_state = str(state.state).split(" ")[0].replace(",", ".")
                actual_kwh = float(cleaned_state)
                if actual_kwh < 0:
                     _LOGGER.warning(f"Actual yield from '{self.solar_yield_today_entity_id}' is negative ({actual_kwh} kWh). Treating as 0 kWh.")
                     actual_kwh = 0.0

            except (ValueError, TypeError):
                _LOGGER.error(f"Evening verification failed: Invalid non-numeric value from yield sensor '{self.solar_yield_today_entity_id}': '{state.state}'.")
                return

        except Exception as e:
             _LOGGER.error(f"Error accessing yield sensor '{self.solar_yield_today_entity_id}': {e}", exc_info=True)
             return

        predicted_kwh: Optional[float] = None
        if self.coordinator.data:
            predicted_kwh = self.coordinator.data.get("forecast_today")

        if predicted_kwh is None:
            _LOGGER.warning("Evening verification skipped: No 'forecast_today' value available in coordinator data.")
            return

        try:
            predicted_kwh = float(predicted_kwh)
            if predicted_kwh < 0: predicted_kwh = 0.0
        except (ValueError, TypeError):
             _LOGGER.error(f"Invalid predicted_kwh value from coordinator: {predicted_kwh}. Cannot verify.")
             return

        deviation_kwh: float = 0.0
        accuracy_pct: float = 0.0

        if actual_kwh is not None:
            deviation_kwh = abs(predicted_kwh - actual_kwh)

            if actual_kwh > 0.1:
                error_fraction = deviation_kwh / actual_kwh
                accuracy = max(0.0, 1.0 - error_fraction)
                accuracy_pct = accuracy * 100.0
            elif predicted_kwh < 0.1 and actual_kwh < 0.1:
                 accuracy = 1.0
                 accuracy_pct = 100.0
            else:
                 accuracy = 0.0
                 accuracy_pct = 0.0

            _LOGGER.info(
                f"Evening Verification Results:\n"
                f"  Predicted Today: {predicted_kwh:.2f} kWh\n"
                f"  Actual Today:    {actual_kwh:.2f} kWh\n"
                f"  Deviation:       {deviation_kwh:.2f} kWh\n"
                f"  Accuracy:        {accuracy_pct:.1f}%"
            )

            try:
                await self.data_manager.update_today_predictions_actual(actual_kwh, accuracy=accuracy)
                _LOGGER.debug("Updated today's prediction records in history with actual value and accuracy.")
            except Exception as e:
                _LOGGER.warning(f"Could not update today's prediction records in history: {e}")

            if predicted_kwh > 0.1 and actual_kwh >= 0.0:
                try:
                    ideal_factor = actual_kwh / predicted_kwh
                    new_correction_factor = max(CORRECTION_FACTOR_MIN, min(CORRECTION_FACTOR_MAX, ideal_factor))

                    _LOGGER.debug(f"Calculated new fallback correction factor based on today's ratio: "
                                  f"{actual_kwh:.2f} / {predicted_kwh:.2f} = {ideal_factor:.3f} -> Clamped: {new_correction_factor:.3f}")

                    current_weights = await self.data_manager.get_learned_weights()
                    if current_weights is None:
                         _LOGGER.info("No learned weights file found. Creating default weights to save correction factor.")
                         current_weights = create_default_learned_weights()

                    smoothed_factor = new_correction_factor

                    if abs(smoothed_factor - current_weights.correction_factor) > 0.01:
                        current_weights.correction_factor = smoothed_factor
                        await self.data_manager.save_learned_weights(current_weights)
                        _LOGGER.info(f"Updated and saved fallback correction factor: {smoothed_factor:.3f}")
                        self.coordinator.learned_correction_factor = smoothed_factor
                    else:
                         _LOGGER.debug("Correction factor change is minimal, not saving.")


                except ZeroDivisionError:
                     _LOGGER.warning("Predicted value was near zero, cannot update correction factor based on ratio.")
                except Exception as e:
                    _LOGGER.error(f"Failed to update fallback correction factor: {e}", exc_info=True)

            self.coordinator.last_day_error_kwh = round(deviation_kwh, 2)
            self.coordinator.yesterday_accuracy = round(accuracy_pct, 1)
            self.coordinator.async_update_listeners()

        else:
             _LOGGER.error("Evening verification failed: Actual kWh value was None unexpectedly.")


    @callback
    async def scheduled_night_cleanup(self, now: datetime) -> None:
        """Callback for the scheduled night cleanup task 200 AM by Zara"""
        current_time = now if now is not None else dt_util.now()
        _LOGGER.info(f"Starting night cleanup (Local Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')})...")

        try:
            # Step 1: Remove duplicate samples
            _LOGGER.info("Step 1/2: Removing duplicate samples...")
            duplicate_result = await self.data_manager.cleanup_duplicate_samples()
            _LOGGER.info(
                f"Duplicate cleanup completed: {duplicate_result['removed']} duplicates removed, "
                f"{duplicate_result['remaining']} samples remaining."
            )
            
            # Step 2: Remove zero-production samples
            _LOGGER.info("Step 2/2: Removing zero-production samples...")
            zero_result = await self.data_manager.cleanup_zero_production_samples()
            _LOGGER.info(
                f"Zero-production cleanup completed: {zero_result['removed']} samples removed, "
                f"{zero_result['remaining']} samples remaining."
            )
            
            # Summary
            total_removed = duplicate_result['removed'] + zero_result['removed']
            _LOGGER.info(
                f"Night cleanup finished: {total_removed} total samples removed "
                f"({duplicate_result['removed']} duplicates + {zero_result['removed']} zero-production), "
                f"{zero_result['remaining']} samples remaining."
            )

        except Exception as e:
            _LOGGER.error(f"Failed to execute night cleanup: {e}", exc_info=True)

    @callback
    async def reset_expected_production(self, now: datetime) -> None:
        """Reset expected daily production at midnight by Zara"""
        _LOGGER.info(f"Resetting expected daily production (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')})...")
        try:
            await self.coordinator.reset_expected_daily_production()
            _LOGGER.info("Expected daily production reset successful.")
        except Exception as e:
            _LOGGER.error(f"Failed to reset expected daily production: {e}", exc_info=True)

    @callback
    async def set_expected_production(self, now: datetime) -> None:
        """Set expected daily production at 6 AM by Zara"""
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
        """Retry mechanism for setting forecast if 0600 failed by Zara"""
        _LOGGER.info(f"=== Forecast Retry Attempt #{attempt} (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')}) ===")
        
        try:
            # Check if forecast is already locked
            current_forecast = await self.data_manager.get_current_day_forecast()
            
            if current_forecast and current_forecast.get("locked"):
                _LOGGER.info(
                    f"Forecast already locked with {current_forecast.get('prediction_kwh')} kWh - "
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
    # NEW: Daily Forecasts System Tasks (23:30-23:32)
    # ==================================================================================

    @callback
    async def finalize_day_task(self, now: datetime) -> None:
        """Finalize current day with actual values at 2330 by Zara"""
        current_time = now if now is not None else dt_util.now()
        _LOGGER.info(f"Finalizing day (Local Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')})...")

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
                    f"Day finalized successfully: Yield={actual_yield:.2f} kWh, "
                    f"Consumption={f'{actual_consumption:.2f}' if actual_consumption else 'N/A'} kWh"
                )
            else:
                _LOGGER.error("Failed to finalize day")
                
        except Exception as e:
            _LOGGER.error(f"Failed to finalize day: {e}", exc_info=True)

    @callback
    async def move_to_history_task(self, now: datetime) -> None:
        """Move current day to history at 2331 by Zara"""
        current_time = now if now is not None else dt_util.now()
        _LOGGER.info(f"Moving day to history (Local Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')})...")

        try:
            success = await self.data_manager.move_to_history()

            if success:
                _LOGGER.info("Current day moved to history successfully")
            else:
                _LOGGER.error("Failed to move day to history")

        except Exception as e:
            _LOGGER.error(f"Failed to move to history: {e}", exc_info=True)

    @callback
    async def calculate_stats_task(self, now: datetime) -> None:
        """Calculate statistics at 2332 by Zara"""
        current_time = now if now is not None else dt_util.now()
        _LOGGER.info(f"Calculating statistics (Local Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')})...")

        try:
            success = await self.data_manager.calculate_statistics()
            
            if success:
                _LOGGER.info("Statistics calculated successfully")
                
                # Log some key stats if available
                try:
                    data = await self.data_manager.load_daily_forecasts()
                    stats_7d = data.get("statistics", {}).get("last_7_days", {})
                    if stats_7d:
                        _LOGGER.info(
                            f"Last 7 days: Avg yield={stats_7d.get('avg_yield', 0):.2f} kWh, "
                            f"Avg accuracy={stats_7d.get('avg_accuracy', 0):.1f}%"
                        )
                except Exception:
                    pass  # Non-critical
            else:
                _LOGGER.error("Failed to calculate statistics")
                
        except Exception as e:
            _LOGGER.error(f"Failed to calculate statistics: {e}", exc_info=True)

    async def _save_actual_best_hour(self) -> None:
        """Calculate and save the actual best production hour from todays hourly samples by Zara"""
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

