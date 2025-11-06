"""
Production Tracking and Monitoring

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
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_change

# CRITICAL: Use SafeDateTimeUtil for timezone awareness - NOT homeassistant.util.dt!
from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class ProductionTimeCalculator:
    """
    Live tracks production time based on power entity state changes.
    Uses a state machine to handle start, stop, and low-power/unavailable timeouts.
    CRITICAL FIX: All internal timestamps now use LOCAL time for consistency.
    """

    def __init__(self, hass: HomeAssistant, power_entity: Optional[str] = None, data_manager=None):
        """Initialize the live production time tracker with persistent state."""
        self.hass = hass
        self.power_entity = power_entity
        self.data_manager = data_manager

        # State machine variables - CRITICAL: Now using LOCAL time
        self._is_active = False  # Is production currently considered active?
        self._start_time: Optional[datetime] = None # When the current active phase started (LOCAL)
        self._accumulated_hours = 0.0 # Total hours accumulated today before the current phase
        self._zero_power_start: Optional[datetime] = None # Tracks start of zero/unavailable period (LOCAL)
        self._today_total_hours = 0.0 # Stores final value for the day after midnight reset
        self._end_time: Optional[datetime] = None  # When production stopped (LOCAL)
        self._last_power_above_10w: Optional[datetime] = None  # Last time power > 10W (LOCAL)
        self._zero_power_streak_minutes = 0  # Minutes at zero power


        # Configuration thresholds (using Watts for input sensor)
        self.MIN_POWER_THRESHOLD_W = 10.0  # Power (W) to start counting production
        self.ZERO_POWER_THRESHOLD_W = 10.0  # Power (W) below which the stop-timer starts
        self.ZERO_POWER_TIMEOUT = timedelta(minutes=5) # Time to wait at zero/unavailable before stopping

        # Listener removal callbacks
        self._state_listener_remove = None
        self._midnight_listener_remove = None
        self._periodic_save_listener_remove = None  # Periodic save timer

        # Persistent state management
        self._last_save_time: Optional[datetime] = None
        self._needs_save = False  # Flag to indicate if save is needed
        self.AUTOSAVE_INTERVAL = timedelta(seconds=30)  # Save every 30 seconds if needed

        _LOGGER.info("ProductionTimeCalculator (Live Tracking) initialized with LOCAL time and persistence")

    async def _load_persistent_state(self) -> bool:
        """
        Load production time state from persistent storage.
        Called at startup to restore state after restart.
        
        Returns:
            True if state was loaded and restored, False otherwise
        """
        if not self.data_manager:
            return False
        
        try:
            state_file = self.data_manager.production_time_state_file
            if not state_file.exists():
                _LOGGER.debug("No persistent production time state file found")
                return False
            
            state_data = await self.data_manager._read_json_file(state_file, {})
            
            if not state_data:
                return False
            
            # Verify it's for today
            saved_date = state_data.get("date")
            today_str = dt_util.now().date().isoformat()
            
            if saved_date != today_str:
                _LOGGER.info(f"Persistent state is from {saved_date}, not loading (today is {today_str})")
                return False
            
            # Restore state
            self._accumulated_hours = state_data.get("accumulated_hours", 0.0)
            self._is_active = state_data.get("is_active", False)
            
            # Restore start_time if active (convert to LOCAL)
            if self._is_active and state_data.get("start_time"):
                try:
                    # Parse and ensure LOCAL timezone
                    start_time_str = state_data["start_time"]
                    self._start_time = dt_util.parse_datetime(start_time_str)
                    if self._start_time:
                        self._start_time = dt_util.ensure_local(self._start_time)
                except Exception as e:
                    _LOGGER.warning(f"Could not restore start_time: {e}")
                    self._is_active = False
                    self._start_time = None
            
            _LOGGER.info(
                f"Loaded production time state from {state_file.name}: "
                f"accumulated={self._accumulated_hours:.2f}h, active={self._is_active}, "
                f"start_time={self._start_time.isoformat() if self._start_time else 'None'}"
            )
            
            return True
            
        except Exception as e:
            _LOGGER.error(f"Failed to load production time state: {e}")
            return False

    async def _save_persistent_state(self, force: bool = False) -> bool:
        """
        Save current production time state to persistent storage.
        
        Args:
            force: Force save regardless of time since last save
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.data_manager:
            return False
        
        try:
            now_local = dt_util.now()
            
            # Check if we should save (throttle to avoid excessive I/O)
            if not force and self._last_save_time:
                time_since_save = now_local - self._last_save_time
                if time_since_save < timedelta(seconds=10):  # Don't save more than every 10 seconds
                    return False  # Too soon, skip
            
            # Get current production time string
            production_time_str = self.get_production_time()
            
            # Prepare state data for production_time_state.json
            state = {
                "version": "1.0",
                "date": now_local.date().isoformat(),
                "accumulated_hours": round(self._accumulated_hours, 4),
                "is_active": self._is_active,
                "start_time": self._start_time.isoformat() if self._start_time else None,
                "last_updated": now_local.isoformat(),
                "production_time_today": production_time_str  # Add formatted string
            }
            
            # Save to production_time_state.json
            success = await self.data_manager._atomic_write_json(
                self.data_manager.production_time_state_file,
                state
            )
            
            if success:
                self._last_save_time = now_local
                _LOGGER.debug(
                    f"Saved production time state: "
                    f"accumulated={self._accumulated_hours:.2f}h, active={self._is_active}, "
                    f"time={production_time_str}"
                )
            
            return success
            
        except Exception as e:
            _LOGGER.error(f"Failed to save persistent production time state: {e}")
            return False

    async def start_tracking(self) -> None:
        """
        TODO 11: NOW ASYNC - Starts the live tracking listeners and initializes state.
        Loads persistent state FIRST before initializing from sensor.
        """
        _LOGGER.info(f"[PRODUCTION_TRACKER] start_tracking called with power_entity={self.power_entity}")
        
        if not self.power_entity:
            _LOGGER.warning("No power entity configured - production time tracking disabled.")
            return
        
        # Try to find entity with retries
        max_retries = 3
        retry_delay = 30
        
        for attempt in range(max_retries):
            test_state = self.hass.states.get(self.power_entity)
            if test_state is not None:
                _LOGGER.info(f"[PRODUCTION_TRACKER] Power entity found: {self.power_entity} = {test_state.state}")
                break
            else:
                if attempt < max_retries - 1:
                    _LOGGER.warning(
                        f"[PRODUCTION_TRACKER] Power entity '{self.power_entity}' not found yet. "
                        f"Retry {attempt + 1}/{max_retries} in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    _LOGGER.error(
                        f"[PRODUCTION_TRACKER] Power entity '{self.power_entity}' not found after {max_retries} retries! "
                        f"Please check your configuration."
                    )
                    # Don't list entities here to avoid async issues
                    _LOGGER.info(
                        f"[PRODUCTION_TRACKER] Check if entity name is correct. "
                        f"Expected format: 'sensor.xxx' without spaces"
                    )
                    return

        try:
            # TODO 11: Lade State ZUERST (synchron, da Methode jetzt async)
            state_loaded = await self._load_persistent_state()
            
            if state_loaded:
                _LOGGER.info(
                    f"[OK] Restored production time state: "
                    f"accumulated={self._accumulated_hours:.2f}h, "
                    f"active={self._is_active}"
                )
            else:
                _LOGGER.info("No persistent state found - will initialize from sensor")

            # Register listener for power entity state changes
            self._state_listener_remove = async_track_state_change_event(
                self.hass,
                [self.power_entity],
                self._handle_power_change # Callback function
            )
            _LOGGER.info(f"[PRODUCTION_TRACKER] State listener registered for {self.power_entity}")

            # Register listener for midnight to reset daily counters
            self._midnight_listener_remove = async_track_time_change(
                self.hass,
                self._handle_midnight_reset, # Callback function
                hour=0,
                minute=0,
                second=1 # Run 1 second past midnight for clean day separation
            )

            # Register periodic save timer (every 30 seconds)
            # Runs every 30 seconds at :00 and :30
            self._periodic_save_listener_remove = async_track_time_change(
                self.hass,
                self._handle_periodic_save, # Callback function
                second=[0, 30]  # At 0 and 30 seconds of every minute
            )
            _LOGGER.debug("Production time periodic save scheduled every 30 seconds")

            # Init NUR wenn kein State geladen wurde
            if not state_loaded:
                _LOGGER.info(f"[PRODUCTION_TRACKER] No state loaded - initializing from current sensor value")
                current_state = self.hass.states.get(self.power_entity)
                now_local = dt_util.now()
                
                _LOGGER.info(f"[PRODUCTION_TRACKER] Current sensor state object: {current_state}")
                if current_state:
                    _LOGGER.info(f"[PRODUCTION_TRACKER] Sensor state value: {current_state.state}")

                if current_state and current_state.state not in ["unavailable", "unknown"]:
                    try:
                        current_power_w = float(current_state.state)
                        _LOGGER.info(f"[PRODUCTION_TRACKER] Parsed power value: {current_power_w}W")

                        # Check if currently producing based on initial state
                        if current_power_w >= self.MIN_POWER_THRESHOLD_W:
                            self._is_active = True
                            # Approximate start time using last_changed (convert to LOCAL)
                            start_candidate_naive = getattr(current_state, 'last_changed', None)
                            if start_candidate_naive:
                                 # Ensure last_changed is in LOCAL timezone
                                 self._start_time = dt_util.ensure_local(start_candidate_naive)
                            else:
                                 self._start_time = now_local # Fallback to current time

                            self._zero_power_start = None # Not in zero power state
                            _LOGGER.info(f"Initial state: Production detected ({current_power_w}W). "
                                         f"Approximate start time (LOCAL): {self._start_time.strftime('%H:%M:%S')}")
                            self._needs_save = True  # Mark for save
                        else:
                             _LOGGER.info(f"Initial state: No production ({current_power_w}W).")
                             # Not producing - default state is already set
                    except (ValueError, TypeError) as e:
                         _LOGGER.warning(f"Could not parse initial power value: {current_state.state}. Treating as not producing. Error: {e}")
                else:
                    _LOGGER.info("Initial state: Power sensor unavailable or unknown. Starting with no production.")
            else:
                # State wurde geladen - prÃ¼fe ob wir den aktuellen Sensor-Status berÃ¼cksichtigen mÃ¼ssen
                current_state = self.hass.states.get(self.power_entity)
                _LOGGER.info(f"[PRODUCTION_TRACKER] State loaded - checking sensor. Current state: {current_state.state if current_state else 'None'}")
                if current_state and current_state.state not in ["unavailable", "unknown"]:
                    try:
                        current_power_w = float(current_state.state)
                        
                        # FALL 1: Wir sind aktiv aber Sensor zeigt 0 - starte Timeout
                        if self._is_active and current_power_w <= self.ZERO_POWER_THRESHOLD_W:
                            self._zero_power_start = dt_util.now()
                            _LOGGER.debug(
                                f"Restored active state but sensor shows zero power - "
                                f"starting timeout timer"
                            )
                        # FALL 2: Wir sind NICHT aktiv aber Sensor zeigt Produktion - STARTE TRACKING
                        elif not self._is_active and current_power_w >= self.MIN_POWER_THRESHOLD_W:
                            _LOGGER.info(
                                f"[PRODUCTION_TRACKER] [OK] Restored inactive state but sensor shows {current_power_w}W - "
                                f"starting production tracking now"
                            )
                            self._start_production_tracking(dt_util.now())
                    except (ValueError, TypeError):
                        pass

            _LOGGER.info(f"Production time tracking started for {self.power_entity}")

        except Exception as e:
            _LOGGER.error(f"Error starting production time tracking: {e}", exc_info=True)


    @callback
    def _handle_power_change(self, event) -> None:
        """
        Callback triggered by state changes of the power entity.
        Handles the state machine logic for tracking production time.
        Treats 'unavailable' like '0 Watts' using the timeout mechanism.
        CRITICAL FIX: Now using LOCAL time for all timestamps.
        """
        try:
            new_state = event.data.get("new_state")
            old_state = event.data.get("old_state")
            now_local = dt_util.now() # Use LOCAL time for internal logic
            power_w: Optional[float] = None
            
            _LOGGER.info(f"[PRODUCTION_TRACKER] Power change event received!")
            _LOGGER.info(f"[PRODUCTION_TRACKER] Old state: {old_state.state if old_state else 'None'}")
            _LOGGER.info(f"[PRODUCTION_TRACKER] New state: {new_state.state if new_state else 'None'}")
            _LOGGER.info(f"[PRODUCTION_TRACKER] Entity: {self.power_entity}")
            
            _LOGGER.debug(f"[PRODUCTION_TRACKER] Power change detected: entity={self.power_entity}, state={new_state.state if new_state else 'None'}")

            # Check if state is valid and parse power value
            if new_state and new_state.state not in ["unavailable", "unknown"]:
                try:
                    power_w = float(new_state.state)
                    _LOGGER.debug(f"[PRODUCTION_TRACKER] Parsed power: {power_w}W, is_active={self._is_active}, MIN_THRESHOLD={self.MIN_POWER_THRESHOLD_W}W")
                except (ValueError, TypeError):
                    power_w = None
                    _LOGGER.debug(f"Could not parse power value: {new_state.state}. Treating as unavailable.")

            # ** STATE MACHINE LOGIC **

            # ** Case 1: Sensor is unavailable or invalid **
            if power_w is None:
                if self._is_active:
                    # We were active, but sensor is now unavailable
                    if self._zero_power_start is None:
                        # Start timeout countdown
                        self._zero_power_start = now_local
                        _LOGGER.debug("Sensor unavailable while active. Starting timeout countdown.")
                    else:
                        # Timeout already in progress
                        elapsed = now_local - self._zero_power_start
                        if elapsed > self.ZERO_POWER_TIMEOUT:
                            # Timeout exceeded - stop tracking
                            _LOGGER.info("Timeout exceeded for sensor unavailable. Stopping production tracking.")
                            self._stop_production_tracking(now_local)
                else:
                    # Not active, sensor unavailable - nothing to do
                    _LOGGER.debug("Sensor unavailable while not active. No action needed.")

                return # Exit early if sensor unavailable

            # ** Case 2: Power >= MIN_POWER_THRESHOLD_W -> Production Active **
            if power_w >= self.MIN_POWER_THRESHOLD_W:
                _LOGGER.debug(f"[PRODUCTION_TRACKER] Case 2: Power {power_w}W >= {self.MIN_POWER_THRESHOLD_W}W threshold")
                if not self._is_active:
                    # Start production tracking
                    _LOGGER.info(f"[PRODUCTION_TRACKER] [OK] Starting production tracking at {now_local.strftime('%H:%M:%S')} (Power: {power_w}W)")
                    self._start_production_tracking(now_local)
                else:
                    # Already active - cancel any zero-power timeout if present
                    if self._zero_power_start is not None:
                        _LOGGER.debug(f"Power restored to {power_w}W - cancelling zero-power timeout.")
                        self._zero_power_start = None
                    # Continue tracking (ongoing production)

            # ** Case 3: Power < ZERO_POWER_THRESHOLD_W -> Potential Stop **
            elif power_w < self.ZERO_POWER_THRESHOLD_W:
                if self._is_active:
                    # Check if timeout is in progress
                    if self._zero_power_start is None:
                        # Start the timeout countdown
                        self._zero_power_start = now_local
                        _LOGGER.debug(f"Low power detected ({power_w}W). Starting timeout countdown.")
                    else:
                        # Timeout already in progress, check if expired
                        elapsed = now_local - self._zero_power_start
                        if elapsed > self.ZERO_POWER_TIMEOUT:
                             # Timeout exceeded - stop tracking
                             _LOGGER.info(f"Timeout exceeded for low power. Stopping production tracking at {now_local.strftime('%H:%M:%S')}")
                             self._stop_production_tracking(now_local)
                        else:
                             # Timeout not yet expired - keep waiting
                             self._zero_power_streak_minutes = int(elapsed.total_seconds() / 60)
                             _LOGGER.debug(f"Zero-power timeout in progress. Elapsed: {elapsed.total_seconds():.0f}s, Threshold: {self.ZERO_POWER_TIMEOUT.total_seconds():.0f}s")
                else:
                    # Not active and below threshold - nothing to do
                    _LOGGER.debug(f"Not active and power below threshold ({power_w}W). No action needed.")

            # Track last time we saw power > 10W
            if power_w is not None and power_w > self.ZERO_POWER_THRESHOLD_W:
                self._last_power_above_10w = now_local
                self._zero_power_streak_minutes = 0
                
            # Mark that we need to save the state
            self._needs_save = True

        except Exception as e:
            _LOGGER.error(f"Error handling power state change: {e}", exc_info=True)


    def _start_production_tracking(self, start_time_local: datetime) -> None:
        """
        Starts a new production phase. Sets the active flag and records the start time (LOCAL).
        
        Args:
            start_time_local: The LOCAL time when production started.
        """
        self._is_active = True
        self._start_time = start_time_local
        self._last_power_above_10w = start_time_local
        self._zero_power_streak_minutes = 0
        self._zero_power_start = None # Reset any pending timeout
        _LOGGER.debug(f"Production phase started. Start time (LOCAL): {start_time_local.strftime('%H:%M:%S')}")
        
        # Flag for periodic save
        self._needs_save = True


    def _stop_production_tracking(self, stop_time_local: datetime) -> None:
        """
        Stops the current production phase. Accumulates the time from this phase (LOCAL).
        
        Args:
            stop_time_local: The LOCAL time when production stopped.
        """
        if not self._is_active:
            _LOGGER.debug("Stop called but tracking was not active. Ignoring.")
            return

        if self._start_time:
            # Calculate the duration of this production phase
            production_duration = stop_time_local - self._start_time
            hours_in_phase = production_duration.total_seconds() / 3600.0
            self._accumulated_hours += hours_in_phase

            _LOGGER.debug(
                f"Production phase ended. Start: {self._start_time.strftime('%H:%M:%S')}, "
                f"Stop: {stop_time_local.strftime('%H:%M:%S')}, "
                f"Duration: {hours_in_phase:.2f}h, "
                f"Total Today: {self._accumulated_hours:.2f}h"
            )
        else:
            _LOGGER.warning("Stop called but no start time recorded. Cannot calculate duration.")

        # Reset state
        self._is_active = False
        self._start_time = None
        self._end_time = stop_time_local
        self._zero_power_start = None  # Reset timeout tracker
        
        # Flag for periodic save
        self._needs_save = True


    @callback
    def _handle_midnight_reset(self, now: datetime) -> None:
        """
        Callback for midnight reset of daily counters.
        Stores the final value for the previous day and resets for the new day.
        CRITICAL FIX: Using LOCAL time for consistency.
        """
        try:
            # Save the total hours for the previous day
            self._today_total_hours = self.get_production_hours()
            _LOGGER.info(f"Midnight reset - Final production time for previous day: {self._today_total_hours:.2f} hours")

            # If currently active, accumulate time up to midnight
            now_local = dt_util.now()
            if self._is_active and self._start_time:
                # Use actual midnight (start of new day)
                midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
                time_to_midnight = (midnight - self._start_time).total_seconds() / 3600.0
                if time_to_midnight > 0:
                    self._accumulated_hours += time_to_midnight
                # Reset start time to midnight for the new day
                self._start_time = midnight

            # Reset counters for the new day
            self._accumulated_hours = 0.0
            self._zero_power_start = None  # Reset any pending timeout
            self._end_time = None
            self._last_power_above_10w = None
            self._zero_power_streak_minutes = 0
            
            # Mark that we need to save the state
            self._needs_save = True

            _LOGGER.debug("Daily production time counters reset for new day (LOCAL time)")

        except Exception as e:
            _LOGGER.error(f"Error during midnight reset: {e}", exc_info=True)


    def get_production_hours(self) -> float:
        """
        Get total production hours for today.
        If currently active, includes time up to now.
        Returns value in hours (float).
        CRITICAL FIX: Using LOCAL time for consistency.
        """
        total = self._accumulated_hours

        if self._is_active and self._start_time:
            # Add the time from the current active phase (LOCAL)
            now_local = dt_util.now()
            current_phase_duration = (now_local - self._start_time).total_seconds() / 3600.0
            total += current_phase_duration

        return total


    def get_production_time(self) -> str:
        """
        Get formatted production time string (HH:MM:SS).
        If currently active, includes time up to now.
        CRITICAL FIX: Using LOCAL time for consistency.
        """
        total_hours = self.get_production_hours()

        # Convert to HH:MM:SS format
        hours = int(total_hours)
        minutes = int((total_hours - hours) * 60)
        seconds = int(((total_hours - hours) * 60 - minutes) * 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


    def is_currently_producing(self) -> bool:
        """Check if production is currently active."""
        return self._is_active
    
    @callback
    def _handle_periodic_save(self, now: datetime) -> None:
        """
        Callback for periodic save of production time state.
        Runs every 30 seconds and saves if _needs_save flag is set.
        
        CRITICAL FIX: Also checks timeout even without power events.
        This ensures production stops after 5 min at 0W even if no new events arrive.
        """
        try:
            now_local = dt_util.now()
            
            # CRITICAL: Periodic timeout check (not just event-based)
            if self._is_active and self._zero_power_start is not None:
                elapsed = now_local - self._zero_power_start
                if elapsed > self.ZERO_POWER_TIMEOUT:
                    _LOGGER.info(
                        f"[OK] Periodic timeout check: {elapsed.total_seconds():.0f}s elapsed "
                        f"(threshold: {self.ZERO_POWER_TIMEOUT.total_seconds():.0f}s). "
                        f"Stopping production tracking."
                    )
                    self._stop_production_tracking(now_local)
                    self._needs_save = True
                else:
                    # Update streak counter
                    self._zero_power_streak_minutes = int(elapsed.total_seconds() / 60)
            
            # Save if needed
            if self._needs_save and self.data_manager:
                _LOGGER.debug("Periodic save triggered - saving production state")
                # Create async task for saving (allowed in time_change callbacks)
                self.hass.async_create_task(self._save_state_async())
                self._needs_save = False
        except Exception as e:
            _LOGGER.error(f"Error during periodic save/timeout check: {e}")
    
    async def _save_state_async(self) -> None:
        """
        Async wrapper for saving both persistent state and production tracking.
        Called from periodic save handler.
        UPDATED: Uses new data_manager.update_production_time() method.
        """
        try:
            # Save to production_time_state.json
            await self._save_persistent_state(force=True)
            
            # Calculate current values
            now_local = dt_util.now()
            duration_seconds = int(self._accumulated_hours * 3600)
            if self._is_active and self._start_time:
                active_duration = (now_local - self._start_time).total_seconds()
                duration_seconds += int(active_duration)
            
            # Save to daily_forecasts.json using NEW METHOD
            await self.data_manager.update_production_time(
                active=self._is_active,
                duration_seconds=duration_seconds,
                start_time=self._start_time,
                end_time=self._end_time,
                last_power_above_10w=self._last_power_above_10w,
                zero_power_since=self._zero_power_start  # NEW: Pass timeout indicator
            )
            
            _LOGGER.debug(
                f"Production state saved: active={self._is_active}, "
                f"duration={duration_seconds}s, "
                f"zero_power_since={self._zero_power_start.isoformat() if self._zero_power_start else 'None'}"
            )
            
        except Exception as e:
            _LOGGER.error(f"Failed to save production state: {e}")
    
    async def stop_tracking(self) -> None:
        """
        Stops all tracking listeners and saves final state.
        Called when the integration is unloaded.
        """
        try:
            # Save final state before stopping
            if self.data_manager and self._needs_save:
                await self._save_state_async()
                
            # Remove listeners
            if self._state_listener_remove:
                self._state_listener_remove()
            if self._midnight_listener_remove:
                self._midnight_listener_remove()
            if self._periodic_save_listener_remove:
                self._periodic_save_listener_remove()
                
            # Final save if currently producing
            if self._is_active:
                now_local = dt_util.now()
                self._stop_production_tracking(now_local)
                if self.data_manager:
                    await self._save_state_async()

            _LOGGER.info("Production time tracking stopped")
        except Exception as e:
            _LOGGER.error(f"Error stopping production time tracking: {e}", exc_info=True)

    def get_today_total_hours(self) -> float:
        """
        Get the final production hours from the previous day.
        Only available after midnight reset, otherwise returns current day's value.
        """
        return self._today_total_hours if self._today_total_hours > 0 else self.get_production_hours()