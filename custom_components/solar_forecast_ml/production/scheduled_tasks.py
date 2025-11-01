"""
Scheduled Tasks Manager for Solar Forecast ML.
Handles daily forecast updates, evening verification, and startup calculations.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""
import logging
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_change

from ..core.helpers import SafeDateTimeUtil as dt_util
from ..ml.types import LearnedWeights, create_default_learned_weights
from ..data.manager import DataManager
from ..const import DAILY_UPDATE_HOUR, DAILY_VERIFICATION_HOUR, CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX

if TYPE_CHECKING:
    from ..coordinator import SolarForecastMLCoordinator

_LOGGER = logging.getLogger(__name__)


class ScheduledTasksManager:
    """
    Manages scheduled tasks for the Solar Forecast ML integration:
    - Morning forecast update trigger.
    - Evening forecast verification against actual yield.
    - Night cleanup of zero-production samples (2:00 AM).
    - Calculation of yesterday's deviation on startup.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        coordinator: 'SolarForecastMLCoordinator',
        solar_yield_today_entity_id: Optional[str],
        data_manager: DataManager
    ):
        """Initialize the ScheduledTasksManager."""
        self.hass = hass
        self.coordinator = coordinator
        self.solar_yield_today_entity_id = solar_yield_today_entity_id
        self.data_manager = data_manager
        self._listeners = []

        _LOGGER.debug("ScheduledTasksManager initialized.")

    def setup_listeners(self) -> None:
        """Register the time-based listeners with Home Assistant."""
        self.cancel_listeners()

        remove_morning = async_track_time_change(
            self.hass,
            self.scheduled_morning_update,
            hour=DAILY_UPDATE_HOUR,
            minute=0,
            second=5
        )
        self._listeners.append(remove_morning)
        _LOGGER.info(f"Scheduled morning forecast update trigger for {DAILY_UPDATE_HOUR:02d}:00:05 daily.")

        remove_evening = async_track_time_change(
            self.hass,
            self.scheduled_evening_verification,
            hour=DAILY_VERIFICATION_HOUR,
            minute=0,
            second=10
        )
        self._listeners.append(remove_evening)
        _LOGGER.info(f"Scheduled evening verification trigger for {DAILY_VERIFICATION_HOUR:02d}:00:10 daily.")

        remove_cleanup = async_track_time_change(
            self.hass,
            self.scheduled_night_cleanup,
            hour=2,
            minute=0,
            second=15
        )
        self._listeners.append(remove_cleanup)
        _LOGGER.info("Scheduled night cleanup of zero-production samples for 02:00:15 daily.")

        # Reset expected daily production at midnight
        remove_reset_expected = async_track_time_change(
            self.hass,
            self.reset_expected_production,
            hour=0,
            minute=0,
            second=0
        )
        self._listeners.append(remove_reset_expected)
        _LOGGER.info("Scheduled expected daily production reset for 00:00:00 daily.")

        # Set expected daily production at 6 AM
        remove_set_expected = async_track_time_change(
            self.hass,
            self.set_expected_production,
            hour=6,
            minute=0,
            second=0
        )
        self._listeners.append(remove_set_expected)
        _LOGGER.info("Scheduled expected daily production set for 06:00:00 daily.")

    def cancel_listeners(self) -> None:
        """Remove any active time-based listeners."""
        for remove_listener in self._listeners:
            try:
                remove_listener()
            except Exception as e:
                _LOGGER.warning(f"Error removing scheduled task listener: {e}")
        self._listeners = []
        _LOGGER.debug("Cancelled scheduled task listeners.")


    async def calculate_yesterday_deviation_on_startup(self) -> None:
        """
        Calculates the forecast deviation from the previous day upon Home Assistant startup.
        Reads the last prediction record from yesterday in the history to get
        the final predicted and actual values. Updates coordinator state.
        """
        _LOGGER.info("Calculating yesterday's forecast deviation at startup...")
        deviation: float = 0.0

        try:
            history_data = await self.data_manager.get_prediction_history()
            predictions: List[Dict[str, Any]] = history_data.get('predictions', [])

            if not predictions:
                _LOGGER.info("No prediction history available. Cannot calculate yesterday's deviation.")
                self.coordinator.last_day_error_kwh = deviation
                self.coordinator.yesterday_accuracy = 0.0
                return

            yesterday_local_date: date = (dt_util.as_local(dt_util.utcnow()) - timedelta(days=1)).date()
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
        """
        Callback for the scheduled morning task. Triggers a full forecast update.

        Args:
            now: The datetime object when the trigger occurred (local time).
        """
        _LOGGER.info(f"Triggering daily morning forecast update (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')})...")

        try:
            await self.coordinator.async_request_refresh()
            _LOGGER.info("Morning forecast update request successful.")

        except Exception as e:
            _LOGGER.error(f"Failed to trigger morning forecast update: {e}", exc_info=True)


    @callback
    async def scheduled_evening_verification(self, now: datetime) -> None:
        """
        Callback for the scheduled evening task. Verifies today's forecast
        against the actual yield sensor value, updates prediction records,
        and recalculates the fallback correction factor.

        Args:
            now: The datetime object when the trigger occurred (local time).
        """
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
        """
        Callback for the scheduled night cleanup task (2:00 AM).
        Removes duplicates and zero-production samples from hourly samples database.

        Args:
            now: The datetime object when the trigger occurred (local time).
        """
        _LOGGER.info(f"Starting night cleanup (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')})...")

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
        """Reset expected daily production at midnight."""
        _LOGGER.info(f"Resetting expected daily production (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')})...")
        try:
            await self.coordinator.reset_expected_daily_production()
            _LOGGER.info("Expected daily production reset successful.")
        except Exception as e:
            _LOGGER.error(f"Failed to reset expected daily production: {e}", exc_info=True)

    @callback
    async def set_expected_production(self, now: datetime) -> None:
        """Set expected daily production at 6 AM."""
        _LOGGER.info(f"Setting expected daily production (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')})...")
        try:
            await self.coordinator.set_expected_daily_production()
            _LOGGER.info("Expected daily production set successful.")
        except Exception as e:
            _LOGGER.error(f"Failed to set expected daily production: {e}", exc_info=True)
