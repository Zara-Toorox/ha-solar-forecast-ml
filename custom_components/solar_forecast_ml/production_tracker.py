"""
Live Production Time Tracker for Solar Forecast ML.
Tracks the duration of solar production based on power sensor state changes.

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
from datetime import datetime, timedelta, timezone
from typing import Optional

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_change

# Use SafeDateTimeUtil for timezone awareness
from .helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class ProductionTimeCalculator:
    """
    Live tracks production time based on power entity state changes.
    Uses a state machine to handle start, stop, and low-power/unavailable timeouts.
    """

    def __init__(self, hass: HomeAssistant, power_entity: Optional[str] = None):
        """Initialize the live production time tracker."""
        self.hass = hass
        self.power_entity = power_entity

        # State machine variables
        self._is_active = False  # Is production currently considered active?
        self._start_time: Optional[datetime] = None # When the current active phase started (UTC)
        self._accumulated_hours = 0.0 # Total hours accumulated today before the current phase
        self._zero_power_start: Optional[datetime] = None # Tracks start of zero/unavailable period (UTC)
        self._today_total_hours = 0.0 # Stores final value for the day after midnight reset

        # Configuration thresholds (using Watts for input sensor)
        self.MIN_POWER_THRESHOLD_W = 10.0  # Power (W) to start counting production
        self.ZERO_POWER_THRESHOLD_W = 1.0  # Power (W) below which the stop-timer starts
        self.ZERO_POWER_TIMEOUT = timedelta(minutes=5) # Time to wait at zero/unavailable before stopping

        # Listener removal callbacks
        self._state_listener_remove = None
        self._midnight_listener_remove = None

        _LOGGER.info("ProductionTimeCalculator (Live Tracking) initialized")

    def start_tracking(self) -> None:
        """Starts the live tracking listeners and initializes state based on current power."""
        if not self.power_entity:
            _LOGGER.warning("No power entity configured - production time tracking disabled.")
            return

        try:
            # Register listener for power entity state changes
            self._state_listener_remove = async_track_state_change_event(
                self.hass,
                [self.power_entity],
                self._handle_power_change # Callback function
            )

            # Register listener for midnight to reset daily counters
            self._midnight_listener_remove = async_track_time_change(
                self.hass,
                self._handle_midnight_reset, # Callback function
                hour=0,
                minute=0,
                second=1 # Run 1 second past midnight for clean day separation
            )

            # Initialize the state based on the current power sensor value
            _LOGGER.debug(f"Initializing ProductionTimeCalculator state for {self.power_entity}")
            current_state = self.hass.states.get(self.power_entity)
            now_utc = dt_util.utcnow() # Use UTC for internal tracking

            if current_state and current_state.state not in ["unavailable", "unknown"]:
                try:
                    current_power_w = float(current_state.state)

                    # Check if currently producing based on initial state
                    if current_power_w >= self.MIN_POWER_THRESHOLD_W:
                        self._is_active = True
                        # Approximate start time using last_changed (convert to UTC)
                        start_candidate_naive = getattr(current_state, 'last_changed', None)
                        if start_candidate_naive:
                             # Ensure last_changed is treated as local and converted to UTC
                             start_candidate_local = dt_util.as_local(start_candidate_naive)
                             self._start_time = start_candidate_local.astimezone(timezone.utc)
                        else:
                             self._start_time = now_utc # Fallback to current time

                        self._zero_power_start = None # Not in zero power state
                        _LOGGER.info(f"Initial state: Production detected ({current_power_w}W). "
                                     f"Approximate start time (UTC): {self._start_time}")
                    else:
                        # Not producing initially
                        self._is_active = False
                        self._start_time = None
                        self._zero_power_start = None
                        _LOGGER.info(f"Initial state: No production detected ({current_power_w}W).")

                except (ValueError, TypeError):
                     _LOGGER.warning(f"Could not parse initial state value for {self.power_entity}: "
                                     f"'{current_state.state}'. Assuming not producing.")
                     self._is_active = False # Safe default
            else:
                # Handle cases where sensor is unavailable or not found at startup
                state_str = "not found" if current_state is None else current_state.state
                _LOGGER.warning(f"Could not get valid initial state for {self.power_entity} (State: {state_str}). "
                                "Assuming not producing.")
                self._is_active = False # Safe default

            _LOGGER.info(f"Production time tracking started for {self.power_entity}")

        except Exception as e:
            _LOGGER.error(f"Error starting production time tracking: {e}", exc_info=True)


    @callback
    def _handle_power_change(self, event) -> None:
        """
        Callback triggered by state changes of the power entity.
        Handles the state machine logic for tracking production time.
        Treats 'unavailable' like '0 Watts' using the timeout mechanism.
        """
        try:
            new_state = event.data.get("new_state")
            now_utc = dt_util.utcnow() # Use UTC for internal logic
            power_w: Optional[float] = None
            is_unavailable = False

            # Determine power value and availability
            if not new_state or new_state.state in ["unavailable", "unknown"]:
                # --- Handle Unavailable/Unknown State ---
                is_unavailable = True
                power_w = 0.0 # Treat as zero power for timeout logic
                state_str = "None" if not new_state else new_state.state
                _LOGGER.debug(f"Power sensor {self.power_entity} reported state: {state_str}. "
                              f"Treating as {power_w}W for timeout check.")
            else:
                try:
                    power_str = str(new_state.state) # Ensure string for parsing
                    power_w = float(power_str)
                    # Check for negative values, treat them as 0
                    if power_w < 0:
                        _LOGGER.debug(f"Negative power value {power_w}W detected, treating as 0W.")
                        power_w = 0.0
                except (ValueError, TypeError):
                    # Ignore non-numeric states
                    _LOGGER.debug(f"Ignoring non-numeric state for {self.power_entity}: '{new_state.state}'")
                    return # Exit if state cannot be parsed as a number

            # --- State Machine Logic ---
            if power_w >= self.MIN_POWER_THRESHOLD_W:
                # --- State: Production Started or Continuing ---
                if not self._is_active:
                    # Transition: Inactive -> Active
                    self._is_active = True
                    self._start_time = now_utc # Production starts *now*
                    _LOGGER.debug(f"Production started: {power_w:.1f}W at {now_utc}")

                # If production is active (newly or continuing), reset the zero power timer
                if self._zero_power_start is not None:
                     _LOGGER.debug(f"Zero-Power/Unavailable timer reset due to production ({power_w:.1f}W) at {now_utc}")
                     self._zero_power_start = None
                # self._last_production_time = now_utc # Can be used for staleness check if needed

            elif self._is_active:
                # --- State: Potentially Stopping (Was active, now power < threshold) ---
                # Check if power is near zero OR if the sensor is unavailable
                if power_w < self.ZERO_POWER_THRESHOLD_W or is_unavailable:
                    # Start or check the zero power timeout
                    if self._zero_power_start is None:
                        # Timer not running, start it now
                        self._zero_power_start = now_utc
                        _LOGGER.debug(f"Zero-Power/Unavailable timer started: {power_w:.1f}W at {now_utc}")
                    else:
                        # Timer is running, check if timeout reached
                        elapsed_since_zero = now_utc - self._zero_power_start
                        if elapsed_since_zero >= self.ZERO_POWER_TIMEOUT:
                            # Timeout reached, stop the production tracking
                            # The logical stop time is when the zero power period began
                            stop_time_utc = self._zero_power_start
                            self._stop_production_tracking(stop_time_utc)
                            _LOGGER.debug(
                                f"Production stopped after {self.ZERO_POWER_TIMEOUT.total_seconds():.0f}s timeout "
                                f"detected at {now_utc} (Logical stop time: {stop_time_utc})"
                            )
                            # State is now inactive, timer reset inside _stop_production_tracking

                else:
                    # Power is low (e.g., 5W) but not zero/unavailable
                    # Reset the zero power timer if it was running
                    if self._zero_power_start is not None:
                        _LOGGER.debug(f"Zero-Power/Unavailable timer reset due to low power ({power_w:.1f}W) at {now_utc}")
                        self._zero_power_start = None
                    # Optionally update last production time if needed for other logic
                    # self._last_production_time = now_utc

            # else: # State: Inactive and power < threshold - Remain Inactive, do nothing

        except Exception as e:
            # Catch unexpected errors in the handler
            _LOGGER.error(f"Error in power change handler: {e}", exc_info=True)


    def _stop_production_tracking(self, stop_time_utc: datetime) -> None:
        """
        Internal method: Stops the current production phase, calculates the duration,
        adds it to the daily total, and resets the state machine.

        Args:
            stop_time_utc: The timestamp (UTC) when the production logically stopped.
        """
        if not self._is_active or not self._start_time:
             # Should not happen if state machine is correct, but log defensively
             _LOGGER.debug("Stop tracking called but not active or no start time recorded.")
             # Ensure state consistency
             self._is_active = False
             self._start_time = None
             self._zero_power_start = None
             return

        # Calculate duration of the just-ended phase
        # Ensure start_time is valid and before stop_time
        if stop_time_utc < self._start_time:
            _LOGGER.warning(f"Stop time ({stop_time_utc}) is before start time ({self._start_time}). "
                            "This might indicate clock issues or out-of-order events. Duration set to 0.")
            duration = timedelta(seconds=0)
        else:
            duration = stop_time_utc - self._start_time

        # Convert duration to hours
        hours_added = duration.total_seconds() / 3600.0

        # Sanity check for negative hours (shouldn't occur with above check)
        if hours_added < 0:
             _LOGGER.error(f"Calculated negative production hours ({hours_added:.4f}). Clamping to 0.")
             hours_added = 0.0

        # Add the duration to the accumulated total for today
        self._accumulated_hours += hours_added

        _LOGGER.info(
            f"Production phase ended. Duration: {hours_added:.2f}h. "
            f"Total accumulated today: {self._accumulated_hours:.2f}h."
        )

        # Reset the state machine for the next production phase
        self._is_active = False
        self._start_time = None
        self._zero_power_start = None # Ensure timer is reset

    @callback
    def _handle_midnight_reset(self, now_local: datetime) -> None:
        """
        Callback triggered shortly after midnight (local time).
        Finalizes the previous day's total and resets counters for the new day.
        """
        _LOGGER.info(f"Midnight reset triggered at local time {now_local}")

        # Use UTC midnight as the definitive boundary for calculations
        # Find the start of the current day in local time, then convert to UTC
        start_of_today_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        midnight_boundary_utc = start_of_today_local.astimezone(timezone.utc)


        # If production was active across the midnight boundary, stop it at UTC midnight
        if self._is_active:
            _LOGGER.info("Production was active across midnight. Stopping phase at UTC midnight.")
            self._stop_production_tracking(midnight_boundary_utc)
            # Note: _stop_production_tracking resets _is_active and _start_time

        # Finalize the total for the completed day
        self._today_total_hours = self._accumulated_hours # Store the final value
        _LOGGER.info(f"Final production time recorded for the previous day: {self._today_total_hours:.2f}h")

        # Reset counters for the *new* day starting now
        self._accumulated_hours = 0.0
        # _today_total_hours keeps the previous day's value until the *next* midnight reset
        # Ensure state machine is fully reset
        self._is_active = False
        self._start_time = None
        self._zero_power_start = None

        _LOGGER.info("Production time counters reset for the new day.")


    def get_production_time(self) -> str:
        """
        Returns the current production time accumulated *today* as a formatted string "Xh Ym".
        Calculates the ongoing duration if currently active.
        """
        try:
            if not self.power_entity:
                return "Not available" # Sensor not configured

            # Get the total hours as a float, including any current active phase
            total_hours_today = self.get_production_hours_float()

            # Handle zero or near-zero time
            if total_hours_today <= (1/3600): # Less than 1 second
                return "0h 0m"

            # Convert float hours to integer hours and minutes
            hours_int = int(total_hours_today)
            minutes_float = (total_hours_today - hours_int) * 60
            minutes_int = int(round(minutes_float)) # Round minutes to nearest integer

            # Handle edge case where rounding minutes results in 60
            if minutes_int == 60:
                hours_int += 1
                minutes_int = 0

            return f"{hours_int}h {minutes_int}m"

        except Exception as e:
            # Log error and return an error string
            _LOGGER.error(f"Error calculating formatted production time: {e}", exc_info=True)
            return "Error"

    def get_production_hours_float(self) -> float:
        """
        Returns the current production time accumulated *today* as a float value (hours).
        Includes the duration of the current phase if active.
        """
        try:
            # Start with hours accumulated from completed phases today
            current_total_hours = self._accumulated_hours

            # If currently in an active production phase, add the duration so far
            if self._is_active and self._start_time:
                now_utc = dt_util.utcnow()
                # Ensure start_time is valid and before now
                if now_utc >= self._start_time:
                    current_duration = now_utc - self._start_time
                    current_phase_hours = current_duration.total_seconds() / 3600.0
                    current_total_hours += current_phase_hours
                else:
                     # Log if clock issues detected, but don't add negative time
                     _LOGGER.warning(f"Current time {now_utc} is before active phase start time {self._start_time}. "
                                     "Ignoring current phase duration.")

            # Ensure the result is non-negative and round slightly for precision
            result = max(0.0, current_total_hours)
            # Optional: Debug log to trace calculation
            # _LOGGER.debug(f"Calculated production hours float: {result:.4f} "
            #               f"(Accumulated: {self._accumulated_hours:.4f}, Active: {self._is_active})")
            return round(result, 4) # Round to 4 decimal places

        except Exception as e:
             # Log error and return 0.0 as a safe fallback
             _LOGGER.error(f"Error calculating production hours float: {e}", exc_info=True)
             return 0.0


    def is_currently_producing(self) -> bool:
        """Returns true if the state machine considers production to be currently active."""
        return self._is_active

    def stop_tracking(self) -> None:
        """Stops the live tracking, removes listeners, and performs final cleanup."""
        _LOGGER.info("Stopping production time tracking...")

        # If tracking stops while active, finalize the current phase
        if self._is_active:
             _LOGGER.info("Production timer was active during stop. Finalizing last phase.")
             self._stop_production_tracking(dt_util.utcnow()) # Stop at current time

        # Remove state change listener
        if self._state_listener_remove:
            try:
                self._state_listener_remove()
                _LOGGER.debug("Removed power entity state listener.")
            except Exception as e:
                 _LOGGER.warning(f"Error removing state listener: {e}")
            self._state_listener_remove = None

        # Remove midnight reset listener
        if self._midnight_listener_remove:
            try:
                self._midnight_listener_remove()
                _LOGGER.debug("Removed midnight reset listener.")
            except Exception as e:
                 _LOGGER.warning(f"Error removing midnight listener: {e}")
            self._midnight_listener_remove = None

        _LOGGER.info("Production time tracking stopped and listeners removed.")