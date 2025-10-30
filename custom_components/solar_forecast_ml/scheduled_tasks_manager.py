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
from datetime import datetime, timedelta, date # Import date
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_change # Import listener type

# Use SafeDateTimeUtil for consistent timezone handling
from .helpers import SafeDateTimeUtil as dt_util
# Import types for data handling
from .ml_types import LearnedWeights, create_default_learned_weights
from .data_manager import DataManager # Import DataManager for type hint
# Import constants for schedule times and factor limits
from .const import DAILY_UPDATE_HOUR, DAILY_VERIFICATION_HOUR, CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX

# Use TYPE_CHECKING to avoid circular import with coordinator if needed for type hints
if TYPE_CHECKING:
    from .coordinator import SolarForecastMLCoordinator

_LOGGER = logging.getLogger(__name__)


class ScheduledTasksManager:
    """
    Manages scheduled tasks for the Solar Forecast ML integration:
    - Morning forecast update trigger.
    - Evening forecast verification against actual yield.
    - Calculation of yesterday's deviation on startup.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        coordinator: 'SolarForecastMLCoordinator', # Use forward reference
        solar_yield_today_entity_id: Optional[str], # Renamed for clarity
        data_manager: DataManager
    ):
        """Initialize the ScheduledTasksManager."""
        self.hass = hass
        self.coordinator = coordinator
        self.solar_yield_today_entity_id = solar_yield_today_entity_id
        self.data_manager = data_manager
        self._listeners = [] # Store listener removal callbacks

        _LOGGER.debug("ScheduledTasksManager initialized.")

    def setup_listeners(self) -> None:
        """Register the time-based listeners with Home Assistant."""
        # Ensure listeners are not duplicated if called multiple times
        self.cancel_listeners()

        # Schedule morning update trigger
        remove_morning = async_track_time_change(
            self.hass,
            self.scheduled_morning_update, # Async callback method
            hour=DAILY_UPDATE_HOUR,
            minute=0,
            second=5 # Run slightly after the hour
        )
        self._listeners.append(remove_morning)
        _LOGGER.info(f"Scheduled morning forecast update trigger for {DAILY_UPDATE_HOUR:02d}:00:05 daily.")

        # Schedule evening verification trigger
        remove_evening = async_track_time_change(
            self.hass,
            self.scheduled_evening_verification, # Async callback method
            hour=DAILY_VERIFICATION_HOUR,
            minute=0,
            second=10 # Run slightly after the hour
        )
        self._listeners.append(remove_evening)
        _LOGGER.info(f"Scheduled evening verification trigger for {DAILY_VERIFICATION_HOUR:02d}:00:10 daily.")

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
        deviation: float = 0.0 # Default to 0 deviation

        try:
            # Get prediction history from data manager
            history_data = await self.data_manager.get_prediction_history()
            predictions: List[Dict[str, Any]] = history_data.get('predictions', [])

            if not predictions:
                _LOGGER.info("No prediction history available. Cannot calculate yesterday's deviation.")
                # Set coordinator state directly
                self.coordinator.last_day_error_kwh = deviation
                self.coordinator.yesterday_accuracy = 0.0 # No accuracy if no data
                return

            # Determine yesterday's date (local timezone)
            yesterday_local_date: date = (dt_util.as_local(dt_util.utcnow()) - timedelta(days=1)).date()
            _LOGGER.debug(f"Looking for records from yesterday: {yesterday_local_date}")

            # Find the *last* valid record from yesterday
            last_yesterday_record: Optional[Dict[str, Any]] = None
            for pred in reversed(predictions): # Iterate backwards efficiently
                try:
                    timestamp_utc = dt_util.parse_datetime(pred.get('timestamp', ''))
                    if not timestamp_utc: continue # Skip if timestamp invalid

                    pred_date_local = dt_util.as_local(timestamp_utc).date()

                    if pred_date_local == yesterday_local_date:
                         # Found the last (most recent) record for yesterday
                         # Check if it has the required values
                         if pred.get('predicted_value') is not None and pred.get('actual_value') is not None:
                              last_yesterday_record = pred
                              break # Stop searching
                         else:
                              # Record found, but missing values needed for calculation
                              _LOGGER.debug(f"Record found for {yesterday_local_date} but missing predicted/actual value.")
                              # Continue searching backwards in case an earlier record has it? Unlikely.
                              # For now, assume the last record should have it if verification ran.
                              break # Stop if last record is incomplete

                    elif pred_date_local < yesterday_local_date:
                        # Optimization: Stop searching once we go past yesterday
                        _LOGGER.debug("Reached records older than yesterday.")
                        break

                except (ValueError, KeyError, TypeError) as parse_error:
                    _LOGGER.debug(f"Skipping record due to parsing error: {parse_error}")
                    continue

            # --- Calculate Deviation ---
            predicted_kwh: Optional[float] = None
            actual_kwh: Optional[float] = None
            accuracy: float = 0.0

            if last_yesterday_record:
                try:
                    predicted_kwh = float(last_yesterday_record['predicted_value'])
                    actual_kwh = float(last_yesterday_record['actual_value'])

                    # Calculate absolute deviation
                    deviation = abs(predicted_kwh - actual_kwh)

                    # Calculate accuracy (percentage)
                    if actual_kwh > 0.1: # Avoid division by zero/small numbers
                        error_fraction = deviation / actual_kwh
                        accuracy = max(0.0, 1.0 - error_fraction) * 100.0 # As percentage
                    elif predicted_kwh < 0.1 and actual_kwh < 0.1:
                         accuracy = 100.0 # Consider accurate if both predicted and actual are near zero
                    else:
                         accuracy = 0.0 # Low accuracy if prediction was high but actual was near zero

                    _LOGGER.info(
                        f"Yesterday's ({yesterday_local_date}) deviation calculated: "
                        f"Predicted={predicted_kwh:.2f} kWh, Actual={actual_kwh:.2f} kWh, "
                        f"Deviation={deviation:.2f} kWh, Accuracy={accuracy:.1f}%"
                    )
                except (ValueError, TypeError, KeyError) as calc_err:
                     _LOGGER.error(f"Error calculating deviation/accuracy from record: {calc_err}. Record: {last_yesterday_record}")
                     deviation = 0.0 # Reset on calculation error
                     accuracy = 0.0
            else:
                _LOGGER.info(f"No complete prediction record found for yesterday ({yesterday_local_date}). "
                             "Deviation set to 0.")
                deviation = 0.0
                accuracy = 0.0

            # Update coordinator state
            self.coordinator.last_day_error_kwh = round(deviation, 2)
            self.coordinator.yesterday_accuracy = round(accuracy, 1) # Store accuracy

        except Exception as e:
            _LOGGER.error(f"Error calculating yesterday's deviation on startup: {e}", exc_info=True)
            # Set defaults on coordinator in case of failure
            self.coordinator.last_day_error_kwh = 0.0
            self.coordinator.yesterday_accuracy = 0.0


    @callback
    async def scheduled_morning_update(self, now: datetime) -> None:
        """
        Callback for the scheduled morning task. Triggers a coordinator refresh
        to generate the forecast for the day.

        Args:
            now: The datetime object when the trigger occurred (local time).
        """
        _LOGGER.info(f"Triggering daily morning forecast update (Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')})...")

        try:
            # Request the coordinator to update its data
            await self.coordinator.async_request_refresh()
            _LOGGER.info("Morning forecast update request successful.")

        except Exception as e:
            # Catch errors during the refresh request itself
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

        # Check if the yield sensor is configured
        if not self.solar_yield_today_entity_id:
            _LOGGER.warning("Cannot perform evening verification: Solar Yield Today sensor not configured.")
            return

        # --- Get Actual Yield ---
        actual_kwh: Optional[float] = None
        try:
            state = self.hass.states.get(self.solar_yield_today_entity_id)

            if not state or state.state in ['unavailable', 'unknown', 'none', None, '']:
                _LOGGER.error(f"Evening verification failed: Yield sensor '{self.solar_yield_today_entity_id}' is unavailable or state is invalid ('{state.state if state else 'None'}').")
                return # Cannot proceed without actual yield

            # Attempt to parse the state value
            try:
                # Handle potential units or commas
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
             return # Cannot proceed

        # --- Get Predicted Yield ---
        predicted_kwh: Optional[float] = None
        if self.coordinator.data:
            predicted_kwh = self.coordinator.data.get("forecast_today")

        if predicted_kwh is None:
            _LOGGER.warning("Evening verification skipped: No 'forecast_today' value available in coordinator data.")
            return

        # Ensure prediction is float and non-negative
        try:
            predicted_kwh = float(predicted_kwh)
            if predicted_kwh < 0: predicted_kwh = 0.0
        except (ValueError, TypeError):
             _LOGGER.error(f"Invalid predicted_kwh value from coordinator: {predicted_kwh}. Cannot verify.")
             return

        # --- Calculate Accuracy and Deviation ---
        deviation_kwh: float = 0.0
        accuracy_pct: float = 0.0

        if actual_kwh is not None: # Should always be true if we reached here
            deviation_kwh = abs(predicted_kwh - actual_kwh)

            # Calculate accuracy (percentage)
            if actual_kwh > 0.1: # Avoid division by zero/small numbers
                error_fraction = deviation_kwh / actual_kwh
                accuracy = max(0.0, 1.0 - error_fraction) # Accuracy fraction [0, 1]
                accuracy_pct = accuracy * 100.0
            elif predicted_kwh < 0.1 and actual_kwh < 0.1:
                 accuracy = 1.0 # Consider accurate if both near zero
                 accuracy_pct = 100.0
            else:
                 accuracy = 0.0 # Low accuracy if prediction high but actual near zero
                 accuracy_pct = 0.0

            _LOGGER.info(
                f"Evening Verification Results:\n"
                f"  Predicted Today: {predicted_kwh:.2f} kWh\n"
                f"  Actual Today:    {actual_kwh:.2f} kWh\n"
                f"  Deviation:       {deviation_kwh:.2f} kWh\n"
                f"  Accuracy:        {accuracy_pct:.1f}%"
            )

            # --- Update Prediction History ---
            try:
                # Call data manager to update all records from today with the actual value and accuracy
                await self.data_manager.update_today_predictions_actual(actual_kwh, accuracy=accuracy) # Pass accuracy fraction
                _LOGGER.debug("Updated today's prediction records in history with actual value and accuracy.")
            except Exception as e:
                _LOGGER.warning(f"Could not update today's prediction records in history: {e}")

            # --- Update Fallback Correction Factor ---
            # Adjust the factor based on today's actual vs predicted ratio
            if predicted_kwh > 0.1 and actual_kwh >= 0.0: # Avoid division by zero and learning from negative actuals
                try:
                    # Calculate the ideal correction factor for today
                    ideal_factor = actual_kwh / predicted_kwh
                    # Clamp the ideal factor to prevent extreme values
                    new_correction_factor = max(CORRECTION_FACTOR_MIN, min(CORRECTION_FACTOR_MAX, ideal_factor))

                    _LOGGER.debug(f"Calculated new fallback correction factor based on today's ratio: "
                                  f"{actual_kwh:.2f} / {predicted_kwh:.2f} = {ideal_factor:.3f} -> Clamped: {new_correction_factor:.3f}")

                    # Load the current weights (or create default)
                    current_weights = await self.data_manager.get_learned_weights()
                    if current_weights is None:
                         _LOGGER.info("No learned weights file found. Creating default weights to save correction factor.")
                         current_weights = create_default_learned_weights()

                    # Apply smoothing (e.g., exponential moving average) to prevent drastic jumps?
                    # Example: smooth_factor = 0.2
                    # smoothed_factor = (smooth_factor * new_correction_factor) + ((1 - smooth_factor) * current_weights.correction_factor)
                    # smoothed_factor = max(CORRECTION_FACTOR_MIN, min(CORRECTION_FACTOR_MAX, smoothed_factor))
                    # For now, apply directly:
                    smoothed_factor = new_correction_factor # No smoothing for now

                    # Update and save if changed significantly
                    if abs(smoothed_factor - current_weights.correction_factor) > 0.01:
                        current_weights.correction_factor = smoothed_factor
                        await self.data_manager.save_learned_weights(current_weights)
                        _LOGGER.info(f"Updated and saved fallback correction factor: {smoothed_factor:.3f}")
                        # Update coordinator's in-memory value as well
                        self.coordinator.learned_correction_factor = smoothed_factor
                    else:
                         _LOGGER.debug("Correction factor change is minimal, not saving.")


                except ZeroDivisionError:
                     _LOGGER.warning("Predicted value was near zero, cannot update correction factor based on ratio.")
                except Exception as e:
                    _LOGGER.error(f"Failed to update fallback correction factor: {e}", exc_info=True)

            # --- Update Coordinator State for Sensors ---
            self.coordinator.last_day_error_kwh = round(deviation_kwh, 2)
            self.coordinator.yesterday_accuracy = round(accuracy_pct, 1) # Store percentage for sensor
            # Trigger listener update for sensors reading these coordinator attributes
            self.coordinator.async_update_listeners()

        else:
             # This case should not be reachable if checks above work
             _LOGGER.error("Evening verification failed: Actual kWh value was None unexpectedly.")