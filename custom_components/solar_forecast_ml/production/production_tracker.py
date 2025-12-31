# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_change

from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..core.core_user_messages import user_msg

_LOGGER = logging.getLogger(__name__)

class ProductionTimeCalculator:
    """Live tracks production time based on power entity state changes"""

    def __init__(
        self,
        hass: HomeAssistant,
        power_entity: Optional[str] = None,
        data_manager=None,
        coordinator=None,
    ):
        """Initialize the live production time tracker with persistent state"""
        self.hass = hass
        self.power_entity = power_entity
        self.data_manager = data_manager
        self.coordinator = coordinator

        self._is_active = False
        self._start_time: Optional[datetime] = None
        self._accumulated_hours = 0.0
        self._zero_power_start: Optional[datetime] = (
            None
        )
        self._today_total_hours = 0.0
        self._end_time: Optional[datetime] = None
        self._last_power_above_10w: Optional[datetime] = None
        self._zero_power_streak_minutes = 0

        self.MIN_POWER_THRESHOLD_W = 10.0
        self.ZERO_POWER_THRESHOLD_W = 10.0
        self.ZERO_POWER_TIMEOUT = timedelta(
            minutes=5
        )

        self._state_listener_remove = None
        self._midnight_listener_remove = None
        self._periodic_save_listener_remove = None

        self._last_save_time: Optional[datetime] = None
        self._needs_save = False
        self.AUTOSAVE_INTERVAL = timedelta(seconds=30)

        _LOGGER.info(
            "ProductionTimeCalculator (Live Tracking) initialized with LOCAL time and persistence"
        )

    async def _load_persistent_state(self) -> bool:
        """Load production time state from persistent storage @zara"""
        if not self.data_manager:
            return False

        try:
            state_file = self.data_manager.production_time_state_file
            if not state_file.exists():

                return False

            state_data = await self.data_manager._read_json_file(state_file, {})

            if not state_data:
                return False

            saved_date = state_data.get("date")
            today_str = dt_util.now().date().isoformat()

            if saved_date != today_str:
                _LOGGER.info(
                    f"Persistent state is from {saved_date}, not loading (today is {today_str})"
                )
                return False

            self._accumulated_hours = state_data.get("accumulated_hours", 0.0)
            self._is_active = state_data.get("is_active", False)

            if self._is_active and state_data.get("start_time"):
                try:

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
        """Save current production time state to persistent storage @zara"""
        if not self.data_manager:
            return False

        try:
            now_local = dt_util.now()

            if not force and self._last_save_time:
                time_since_save = now_local - self._last_save_time
                if time_since_save < timedelta(seconds=10):
                    return False

            production_time_str = self.get_production_time()

            state = {
                "version": "1.0",
                "date": now_local.date().isoformat(),
                "accumulated_hours": round(self._accumulated_hours, 4),
                "is_active": self._is_active,
                "start_time": self._start_time.isoformat() if self._start_time else None,
                "last_updated": now_local.isoformat(),
                "production_time_today": production_time_str,
            }

            success = await self.data_manager._atomic_write_json(
                self.data_manager.production_time_state_file, state
            )

            if success:
                self._last_save_time = now_local

            return success

        except Exception as e:
            _LOGGER.error(f"Failed to save persistent production time state: {e}")
            return False

    async def _sync_to_daily_forecasts(self) -> None:
        """Sync production time state to daily_forecasts.json immediately after restore.

        This fixes the race condition where the ProductionTimeSensor reads from
        daily_forecasts.json before the periodic save (every 30s) updates it.
        Without this sync, users see 00:00:00 for up to 30 seconds after restart.
        """
        if not self.data_manager:
            return

        try:
            now_local = dt_util.now()
            duration_seconds = int(self._accumulated_hours * 3600)

            if self._is_active and self._start_time:
                active_duration = (now_local - self._start_time).total_seconds()
                duration_seconds += int(active_duration)

            await self.data_manager.update_production_time(
                active=self._is_active,
                duration_seconds=duration_seconds,
                start_time=self._start_time,
                end_time=self._end_time,
                last_power_above_10w=self._last_power_above_10w,
                zero_power_since=self._zero_power_start,
            )

            _LOGGER.info(
                f"[OK] Synced production time to daily_forecasts.json: "
                f"duration={duration_seconds}s ({self.get_production_time()})"
            )

        except Exception as e:
            _LOGGER.warning(f"Failed to sync production time to daily_forecasts: {e}")

    async def start_tracking(self) -> None:
        """Starts the live tracking listeners and initializes state @zara"""
        _LOGGER.info(
            f"[PRODUCTION_TRACKER] start_tracking called with power_entity={self.power_entity}"
        )

        if not self.power_entity:
            _LOGGER.info(user_msg('PRODUCTION_TRACKING_DISABLED'))
            return

        max_retries = 3
        retry_delay = 30

        for attempt in range(max_retries):
            test_state = self.hass.states.get(self.power_entity)
            if test_state is not None:
                _LOGGER.info(
                    f"[PRODUCTION_TRACKER] Power entity found: {self.power_entity} = {test_state.state}"
                )
                break
            else:
                if attempt < max_retries - 1:

                    log_func = _LOGGER.info if attempt == 0 else _LOGGER.warning
                    log_func(
                        f"[PRODUCTION_TRACKER] Power entity '{self.power_entity}' not available yet. "
                        f"Waiting for sensor to initialize (attempt {attempt + 1}/{max_retries}, retry in {retry_delay}s)..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    _LOGGER.warning(
                        user_msg('SENSOR_UNAVAILABLE', entity=self.power_entity)
                    )
                    _LOGGER.debug(
                        f"Expected format: 'sensor.xxx' - check configuration"
                    )
                    return

        try:

            state_loaded = await self._load_persistent_state()

            if state_loaded:
                _LOGGER.info(
                    f"[OK] Restored production time state: "
                    f"accumulated={self._accumulated_hours:.2f}h, "
                    f"active={self._is_active}"
                )
                # CRITICAL: Immediately sync to daily_forecasts.json so sensor shows correct value
                await self._sync_to_daily_forecasts()
            else:
                _LOGGER.info("No persistent state found - will initialize from sensor")

            self._state_listener_remove = async_track_state_change_event(
                self.hass, [self.power_entity], self._handle_power_change
            )
            _LOGGER.info(f"[PRODUCTION_TRACKER] State listener registered for {self.power_entity}")

            self._midnight_listener_remove = async_track_time_change(
                self.hass,
                self._handle_midnight_reset,
                hour=0,
                minute=0,
                second=1,
            )

            self._periodic_save_listener_remove = async_track_time_change(
                self.hass,
                self._handle_periodic_save,
                second=[0, 30],
            )

            if not state_loaded:
                _LOGGER.info(
                    f"[PRODUCTION_TRACKER] No state loaded - initializing from current sensor value"
                )
                current_state = self.hass.states.get(self.power_entity)
                now_local = dt_util.now()

                _LOGGER.info(f"[PRODUCTION_TRACKER] Current sensor state object: {current_state}")
                if current_state:
                    _LOGGER.info(f"[PRODUCTION_TRACKER] Sensor state value: {current_state.state}")

                if current_state and current_state.state not in ["unavailable", "unknown"]:
                    try:
                        current_power_w = float(current_state.state)
                        _LOGGER.info(f"[PRODUCTION_TRACKER] Parsed power value: {current_power_w}W")

                        if current_power_w >= self.MIN_POWER_THRESHOLD_W:
                            self._is_active = True

                            start_candidate_naive = getattr(current_state, "last_changed", None)
                            if start_candidate_naive:

                                self._start_time = dt_util.ensure_local(start_candidate_naive)
                            else:
                                self._start_time = now_local

                            self._zero_power_start = None
                            _LOGGER.info(
                                f"Initial state: Production detected ({current_power_w}W). "
                                f"Approximate start time (LOCAL): {self._start_time.strftime('%H:%M:%S')}"
                            )
                            self._needs_save = True
                        else:
                            _LOGGER.info(f"Initial state: No production ({current_power_w}W).")

                    except (ValueError, TypeError) as e:
                        _LOGGER.warning(
                            f"Could not parse initial power value: {current_state.state}. Treating as not producing. Error: {e}"
                        )
                else:
                    _LOGGER.info(
                        "Initial state: Power sensor unavailable or unknown. Starting with no production."
                    )
            else:

                current_state = self.hass.states.get(self.power_entity)
                _LOGGER.info(
                    f"[PRODUCTION_TRACKER] State loaded - checking sensor. Current state: {current_state.state if current_state else 'None'}"
                )
                if current_state and current_state.state not in ["unavailable", "unknown"]:
                    try:
                        current_power_w = float(current_state.state)

                        if self._is_active and current_power_w <= self.ZERO_POWER_THRESHOLD_W:
                            self._zero_power_start = dt_util.now()

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
        """Callback triggered by state changes of the power entity @zara"""
        try:
            new_state = event.data.get("new_state")
            old_state = event.data.get("old_state")
            now_local = dt_util.now()
            power_w: Optional[float] = None

            if new_state and new_state.state not in ["unavailable", "unknown"]:
                try:
                    power_w = float(new_state.state)

                except (ValueError, TypeError):
                    power_w = None

            if power_w is None:
                if self._is_active:

                    if self._zero_power_start is None:

                        self._zero_power_start = now_local

                    else:

                        elapsed = now_local - self._zero_power_start
                        if elapsed > self.ZERO_POWER_TIMEOUT:

                            _LOGGER.info(
                                "Timeout exceeded for sensor unavailable. Stopping production tracking."
                            )
                            self._stop_production_tracking(now_local)
                else:

                    pass

                return

            if power_w is not None and power_w > 0 and self.data_manager:
                try:

                    peak_updated = asyncio.create_task(
                        self.data_manager.update_peak_today(power_w, now_local)
                    )

                    if self.coordinator and peak_updated:

                        async def notify_on_peak_update():
                            try:
                                was_updated = await peak_updated
                                if was_updated:
                                    self.coordinator.async_update_listeners()
                            except Exception as e:

                                pass

                        asyncio.create_task(notify_on_peak_update())
                except Exception as e:

                    pass

            if power_w >= self.MIN_POWER_THRESHOLD_W:

                if not self._is_active:

                    _LOGGER.info(
                        f"[PRODUCTION_TRACKER] [OK] Starting production tracking at {now_local.strftime('%H:%M:%S')} (Power: {power_w}W)"
                    )
                    self._start_production_tracking(now_local)
                else:

                    if self._zero_power_start is not None:

                        self._zero_power_start = None

            elif power_w < self.ZERO_POWER_THRESHOLD_W:
                if self._is_active:

                    if self._zero_power_start is None:

                        self._zero_power_start = now_local

                    else:

                        elapsed = now_local - self._zero_power_start
                        if elapsed > self.ZERO_POWER_TIMEOUT:

                            _LOGGER.info(
                                f"Timeout exceeded for low power. Stopping production tracking at {now_local.strftime('%H:%M:%S')}"
                            )
                            self._stop_production_tracking(now_local)
                        else:

                            self._zero_power_streak_minutes = int(elapsed.total_seconds() / 60)

                else:

                    pass

            if power_w is not None and power_w > self.ZERO_POWER_THRESHOLD_W:
                self._last_power_above_10w = now_local
                self._zero_power_streak_minutes = 0

            self._needs_save = True

        except Exception as e:
            _LOGGER.error(f"Error handling power state change: {e}", exc_info=True)

    def _start_production_tracking(self, start_time_local: datetime) -> None:
        """Starts a new production phase Sets the active flag and records the start time... @zara"""
        self._is_active = True
        self._start_time = start_time_local
        self._last_power_above_10w = start_time_local
        self._zero_power_streak_minutes = 0
        self._zero_power_start = None

        self._needs_save = True

    def _stop_production_tracking(self, stop_time_local: datetime) -> None:
        """Stops the current production phase Accumulates the time from this phase LOCAL @zara"""
        if not self._is_active:

            return

        if self._start_time:

            production_duration = stop_time_local - self._start_time
            hours_in_phase = production_duration.total_seconds() / 3600.0
            self._accumulated_hours += hours_in_phase

        else:
            _LOGGER.warning("Stop called but no start time recorded. Cannot calculate duration.")

        self._is_active = False
        self._start_time = None
        self._end_time = stop_time_local
        self._zero_power_start = None

        self._needs_save = True

    @callback
    def _handle_midnight_reset(self, now: datetime) -> None:
        """Callback for midnight reset of daily counters @zara"""
        try:

            self._today_total_hours = self.get_production_hours()
            _LOGGER.info(
                f"Midnight reset - Final production time for previous day: {self._today_total_hours:.2f} hours"
            )

            now_local = dt_util.now()
            if self._is_active and self._start_time:

                midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
                time_to_midnight = (midnight - self._start_time).total_seconds() / 3600.0
                if time_to_midnight > 0:
                    self._accumulated_hours += time_to_midnight

                self._start_time = midnight

            self._accumulated_hours = 0.0
            self._zero_power_start = None
            self._end_time = None
            self._last_power_above_10w = None
            self._zero_power_streak_minutes = 0

            self._needs_save = True

        except Exception as e:
            _LOGGER.error(f"Error during midnight reset: {e}", exc_info=True)

    def get_production_hours(self) -> float:
        """Get total production hours for today @zara"""
        total = self._accumulated_hours

        if self._is_active and self._start_time:

            now_local = dt_util.now()
            current_phase_duration = (now_local - self._start_time).total_seconds() / 3600.0
            total += current_phase_duration

        return total

    def get_production_time(self) -> str:
        """Get formatted production time string HHMMSS @zara"""
        total_hours = self.get_production_hours()

        hours = int(total_hours)
        minutes = int((total_hours - hours) * 60)
        seconds = int(((total_hours - hours) * 60 - minutes) * 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def is_currently_producing(self) -> bool:
        """Check if production is currently active @zara"""
        return self._is_active

    @callback
    def _handle_periodic_save(self, now: datetime) -> None:
        """Callback for periodic save of production time state @zara"""
        try:
            now_local = dt_util.now()

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

                    self._zero_power_streak_minutes = int(elapsed.total_seconds() / 60)

            if self._needs_save and self.data_manager:

                self.hass.async_create_task(self._save_state_async())
                self._needs_save = False
        except Exception as e:
            _LOGGER.error(f"Error during periodic save/timeout check: {e}")

    async def _save_state_async(self) -> None:
        """Async wrapper for saving both persistent state and production tracking @zara"""
        try:

            await self._save_persistent_state(force=True)

            now_local = dt_util.now()
            duration_seconds = int(self._accumulated_hours * 3600)
            if self._is_active and self._start_time:
                active_duration = (now_local - self._start_time).total_seconds()
                duration_seconds += int(active_duration)

            await self.data_manager.update_production_time(
                active=self._is_active,
                duration_seconds=duration_seconds,
                start_time=self._start_time,
                end_time=self._end_time,
                last_power_above_10w=self._last_power_above_10w,
                zero_power_since=self._zero_power_start,
            )

        except Exception as e:
            _LOGGER.error(f"Failed to save production state: {e}")

    async def stop_tracking(self) -> None:
        """Stops all tracking listeners and saves final state @zara"""
        try:

            if self.data_manager and self._needs_save:
                await self._save_state_async()

            if self._state_listener_remove:
                try:
                    self._state_listener_remove()
                except ValueError:
                    pass  # Listener already removed
                self._state_listener_remove = None
            if self._midnight_listener_remove:
                try:
                    self._midnight_listener_remove()
                except ValueError:
                    pass  # Listener already removed
                self._midnight_listener_remove = None
            if self._periodic_save_listener_remove:
                try:
                    self._periodic_save_listener_remove()
                except ValueError:
                    pass  # Listener already removed
                self._periodic_save_listener_remove = None

            if self._is_active:
                now_local = dt_util.now()
                self._stop_production_tracking(now_local)
                if self.data_manager:
                    await self._save_state_async()

            _LOGGER.info("Production time tracking stopped")
        except Exception as e:
            _LOGGER.error(f"Error stopping production time tracking: {e}", exc_info=True)

    @property
    def is_active(self) -> bool:
        """Returns True if production is currently active @zara"""
        return self._is_active

    @property
    def total_seconds(self) -> int:
        """Returns total production time in seconds for today @zara"""
        total_hours = self.get_production_hours()
        return int(total_hours * 3600)

    @property
    def start_time(self) -> Optional[datetime]:
        """Returns the start time of currentlast production phase LOCAL @zara"""
        return self._start_time

    @property
    def end_time(self) -> Optional[datetime]:
        """Returns the end time of last production phase LOCAL @zara"""
        return self._end_time

    def get_today_total_hours(self) -> float:
        """Get the final production hours from the previous day @zara"""
        return (
            self._today_total_hours if self._today_total_hours > 0 else self.get_production_hours()
        )
