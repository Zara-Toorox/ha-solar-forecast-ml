"""
Production Calculator for Solar Forecast ML.
Calculates production time and other production metrics.
STRATEGY 2: ProductionTimeCalculator for Live-Tracking
Version 4.11.0 - Weighted Peak-Time Calculation

Copyright (C) 2025 Zara-Toorox

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
from datetime import datetime, timedelta
from typing import Optional, Any
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_change

from .helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class ProductionCalculator:
    """
    Calculates historical production metrics, such as total production time
    today or the weighted average peak production hour.
    """
    
    def __init__(self, hass: HomeAssistant):
        self.hass = hass
        
        self.MIN_PRODUCTION_POWER = 0.01  # min power (kW) to count as "producing"
        self.PRODUCTION_START_HOUR = 5    # 5 AM
        self.PRODUCTION_END_HOUR = 21     # 9 PM
        
        # Weighting for peak time calculation
        self.RECENT_DAYS_THRESHOLD = 3
        self.RECENT_WEIGHT = 0.7
        self.OLDER_WEIGHT = 0.3
        
        _LOGGER.debug("✓ ProductionCalculator initialized")
    
    async def calculate_production_time_today(
        self,
        power_entity: Optional[str]
    ) -> str:
        """
        Calculates the total time production was active today based on history.
        """
        try:
            if not power_entity:
                _LOGGER.debug("1️⃣ No power sensor configured")
                return "Not available"
            
            now = dt_util.utcnow()
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Call the blocking history function in an executor thread
            history = await self.hass.async_add_executor_job(
                self._get_state_history,
                power_entity,
                start_of_day,
                now
            )
            
            if not history:
                _LOGGER.debug("1️⃣ No history data available")
                return "Calculating..."
            
            production_minutes = 0
            
            for i in range(len(history) - 1):
                try:
                    power = float(history[i].state)
                    
                    if power >= self.MIN_PRODUCTION_POWER:
                        time_diff = history[i + 1].last_changed - history[i].last_changed
                        production_minutes += time_diff.total_seconds() / 60
                        
                except (ValueError, AttributeError):
                    continue
            
            hours = int(production_minutes // 60)
            minutes = int(production_minutes % 60)
            
            return f"{hours}h {minutes}m"
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Production time calculation failed: {e}")
            return "Calculation failed"
    
    def _get_state_history(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> list:
        """
        Synchronous helper to fetch state history.
        Must be run in an executor job.
        """
        try:
            # Import recorder component locally as required by HA best practices
            from homeassistant.components import recorder
            from homeassistant.components.recorder import history
            
            if not recorder.is_entity_recorded(self.hass, entity_id):
                _LOGGER.debug(f"1️⃣ Entity {entity_id} is not being recorded")
                return []
            
            states = history.state_changes_during_period(
                self.hass,
                start_time,
                end_time,
                entity_id,
                no_attributes=True
            )
            
            if entity_id in states:
                return states[entity_id]
            
            return []
            
        except Exception as e:
            _LOGGER.debug(f"History retrieval failed: {e}")
            return []
    
    async def calculate_peak_production_time(
        self,
        power_entity: Optional[str] = None
    ) -> str:
        """
        Calculates the historical peak production hour using a weighted average.
        Recent days are weighted more heavily.
        """
        try:
            _LOGGER.info("📍 START Peak-Time Calculation (Weighted Method)")
            
            if not power_entity:
                _LOGGER.debug("1️⃣ No power sensor for peak calculation")
                return "12:00"  # Fallback
            
            now = dt_util.now()
            start_time = now - timedelta(days=14)
            cutoff_recent = now - timedelta(days=self.RECENT_DAYS_THRESHOLD)
            
            _LOGGER.debug(
                f"📝 Analysis period: {start_time.strftime('%Y-%m-%d %H:%M')} "
                f"to {now.strftime('%Y-%m-%d %H:%M')}"
            )
            _LOGGER.debug(
                f"📠 Weighting: Last {self.RECENT_DAYS_THRESHOLD} days = {self.RECENT_WEIGHT*100:.0f}%, "
                f"Older days = {self.OLDER_WEIGHT*100:.0f}%"
            )
            
            states = await self.hass.async_add_executor_job(
                self._get_state_history,
                power_entity,
                start_time,
                now
            )
            
            if not states or len(states) < 10:
                _LOGGER.warning(
                    f"⚠️ Not enough data for peak calculation: "
                    f"{len(states) if states else 0} states found (min. 10 required)"
                )
                return "12:00"  # Fallback
            
            _LOGGER.debug(f"📈 {len(states)} states loaded from history")
            
            hourly_data = {hour: {'values': [], 'weights': []} for hour in range(24)}
            
            invalid_states = 0
            night_values = 0
            converted_watt = 0
            recent_count = 0
            older_count = 0
            
            for state in states:
                try:
                    if state.state in ["unavailable", "unknown", None]:
                        invalid_states += 1
                        continue
                    
                    power = float(state.state)
                    
                    if power < self.MIN_PRODUCTION_POWER:
                        night_values += 1
                        continue
                    
                    # Heuristic: If power > 100, assume it's in Watts, convert to kW
                    # This handles sensors that might report W instead of kW
                    if power > 100:
                        power = power / 1000.0
                        converted_watt += 1
                    
                    hour = state.last_changed.hour
                    
                    if not (self.PRODUCTION_START_HOUR <= hour <= self.PRODUCTION_END_HOUR):
                        night_values += 1
                        continue
                    
                    is_recent = state.last_changed >= cutoff_recent
                    weight = self.RECENT_WEIGHT if is_recent else self.OLDER_WEIGHT
                    
                    if is_recent:
                        recent_count += 1
                    else:
                        older_count += 1
                    
                    hourly_data[hour]['values'].append(power)
                    hourly_data[hour]['weights'].append(weight)
                
                except (ValueError, TypeError, AttributeError):
                    invalid_states += 1
                    continue
            
            _LOGGER.debug(
                f"📠 Data quality: {invalid_states} invalid, {night_values} night/low, "
                f"{converted_watt} W->kW converted"
            )
            _LOGGER.debug(
                f"⚖️ Weighted data: {recent_count} recent ({self.RECENT_WEIGHT*100:.0f}%), "
                f"{older_count} older ({self.OLDER_WEIGHT*100:.0f}%)"
            )
            
            hourly_weighted_averages = {}
            
            for hour, data in hourly_data.items():
                if data['values']:
                    weighted_sum = sum(v * w for v, w in zip(data['values'], data['weights']))
                    weight_sum = sum(data['weights'])
                    
                    if weight_sum > 0:
                        weighted_avg = weighted_sum / weight_sum
                        hourly_weighted_averages[hour] = weighted_avg
                        
                        _LOGGER.debug(
                            f"⏰ Hour {hour:02d}: {len(data['values'])} values, "
                            f"Weighted Avg = {weighted_avg:.3f} kW"
                        )
            
            if not hourly_weighted_averages:
                _LOGGER.warning("⚠️ No valid production data found")
                return "12:00"  # Fallback
            
            peak_hour = max(hourly_weighted_averages, key=hourly_weighted_averages.get)
            peak_value = hourly_weighted_averages[peak_hour]
            
            if not (self.PRODUCTION_START_HOUR <= peak_hour <= self.PRODUCTION_END_HOUR):
                _LOGGER.warning(
                    f"⚠️ Peak hour {peak_hour}:00 outside production time "
                    f"({self.PRODUCTION_START_HOUR}-{self.PRODUCTION_END_HOUR}), falling back to 12:00"
                )
                return "12:00"
            
            _LOGGER.info(
                f"✓ Peak hour found: {peak_hour:02d}:00 "
                f"(≈ {peak_value:.2f} kW weighted, "
                f"{len(hourly_data[peak_hour]['values'])} values)"
            )
            
            return f"{peak_hour:02d}:00"
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Peak-Time calculation failed: {e}", exc_info=True)
            return "12:00"
    
    def is_production_hours(self, hour: int = None) -> bool:
        """Checks if a given hour is within the defined production window."""
        try:
            if hour is None:
                hour = dt_util.utcnow().hour
            
            return self.PRODUCTION_START_HOUR <= hour <= self.PRODUCTION_END_HOUR
            
        except Exception:
            return True  # Fail open
    
    def estimate_remaining_production_hours(self) -> float:
        """Estimates remaining production hours for today."""
        try:
            now = dt_util.utcnow()
            current_hour = now.hour
            
            if current_hour >= self.PRODUCTION_END_HOUR:
                return 0.0  # Past production time
            
            if current_hour < self.PRODUCTION_START_HOUR:
                # Before production time, return full duration
                return float(self.PRODUCTION_END_HOUR - self.PRODUCTION_START_HOUR)
            
            # During production time
            remaining_full_hours = self.PRODUCTION_END_HOUR - current_hour - 1
            current_hour_fraction = 1.0 - (now.minute / 60.0)
            
            return float(remaining_full_hours) + current_hour_fraction
            
        except Exception:
            return 0.0


class ProductionTimeCalculator:
    """
    Live tracks production time based on power entity state changes.
    Uses a state machine to handle start, stop, and low-power timeouts.
    """
    
    def __init__(self, hass: HomeAssistant, power_entity: Optional[str] = None):
        self.hass = hass
        self.power_entity = power_entity
        
        self._is_active = False  # Is production currently active?
        self._start_time: Optional[datetime] = None
        self._accumulated_hours = 0.0
        self._last_production_time: Optional[datetime] = None
        self._zero_power_start: Optional[datetime] = None
        self._today_total_hours = 0.0
        
        # Power (in W) to start counting production
        self.MIN_POWER_THRESHOLD = 10.0
        # Power (in W) below which the stop-timer starts
        self.ZERO_POWER_THRESHOLD = 1.0
        # Time to wait at zero power before stopping
        self.ZERO_POWER_TIMEOUT = timedelta(minutes=5)
        
        self._state_listener_remove = None
        self._midnight_listener_remove = None
        
        _LOGGER.info("✓ ProductionTimeCalculator initialized")
    
    def start_tracking(self) -> None:
        """Starts the live tracking listeners."""
        if not self.power_entity:
            _LOGGER.info("1️⃣ No power entity - production time tracking disabled")
            return
        
        try:
            # Listen for state changes on the power entity
            self._state_listener_remove = async_track_state_change_event(
                self.hass,
                [self.power_entity],
                self._handle_power_change
            )
            
            # Listen for midnight to reset counters
            self._midnight_listener_remove = async_track_time_change(
                self.hass,
                self._handle_midnight_reset,
                hour=0,
                minute=0,
                second=0
            )
            
            _LOGGER.info(f"✓ Production time tracking started for {self.power_entity}")
            
        except Exception as e:
            _LOGGER.error(f"❌ Error starting tracking: {e}")
    
    @callback
    def _handle_power_change(self, event) -> None:
        """Callback to handle power entity state changes."""
        try:
            new_state = event.data.get("new_state")
            if not new_state or new_state.state in ["unavailable", "unknown"]:
                return
            
            try:
                power = float(new_state.state)
            except (ValueError, TypeError):
                return
            
            now = dt_util.utcnow()
            
            if power >= self.MIN_POWER_THRESHOLD:
                # --- State: Started/Producing ---
                if not self._is_active:
                    self._is_active = True
                    self._start_time = now
                    _LOGGER.debug(f"🟢 Production start: {power}W")
                
                # Reset zero-power timer
                self._zero_power_start = None
                self._last_production_time = now
            
            elif self._is_active:
                # --- State: Stopping? ---
                # Power is < MIN_POWER_THRESHOLD but we were active
                
                if power < self.ZERO_POWER_THRESHOLD:
                    # Power is near zero, start/check the timeout
                    if self._zero_power_start is None:
                        self._zero_power_start = now
                        _LOGGER.debug(f"⏱️ Zero-Power timer started: {power}W")
                    
                    elif now - self._zero_power_start >= self.ZERO_POWER_TIMEOUT:
                        # Timeout reached, stop production tracking
                        self._stop_production_tracking(now)
                        _LOGGER.debug(f"🕑 Production end after 5 min timeout")
                
                else:
                    # Power is low (e.g., 5W) but not zero, reset timer
                    self._zero_power_start = None
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Error in power change handler: {e}")
    
    def _stop_production_tracking(self, stop_time: datetime) -> None:
        """Internal: Stops a production phase and banks the time."""
        if not self._is_active or not self._start_time:
            return
        
        duration = stop_time - self._start_time
        hours = duration.total_seconds() / 3600.0
        
        self._accumulated_hours += hours
        self._today_total_hours = self._accumulated_hours
        
        _LOGGER.info(
            f"✓ Production phase ended: {hours:.2f}h "
            f"(Total today: {self._today_total_hours:.2f}h)"
        )
        
        # Reset state machine
        self._is_active = False
        self._start_time = None
        self._zero_power_start = None
    
    @callback
    def _handle_midnight_reset(self, now: datetime) -> None:
        """Callback to reset counters at midnight."""
        _LOGGER.info(f"🕛 Midnight-Reset: Today {self._today_total_hours:.2f}h produced")
        
        if self._is_active:
            # If production was active across midnight, stop it
            self._stop_production_tracking(now)
        
        self._accumulated_hours = 0.0
        self._today_total_hours = 0.0
        self._is_active = False
        self._start_time = None
        self._last_production_time = None
        self._zero_power_start = None
    
    def get_production_time(self) -> str:
        """Returns the current production time as a formatted string."""
        try:
            if not self.power_entity:
                return "Not available"
            
            total_hours = self.get_production_hours_float()
            
            if total_hours < 0.01:
                return "0h 0m"
            
            hours = int(total_hours)
            minutes = int((total_hours - hours) * 60)
            
            return f"{hours}h {minutes}m"
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Error retrieving production time: {e}")
            return "Error"
    
    def get_production_hours_float(self) -> float:
        """Returns the current production time as a float."""
        try:
            total_hours = self._accumulated_hours
            
            # If currently active, add the ongoing duration
            if self._is_active and self._start_time:
                now = dt_util.utcnow()
                current_duration = now - self._start_time
                total_hours += current_duration.total_seconds() / 3600.0
            
            return round(total_hours, 2)
            
        except Exception:
            return 0.0
    
    def is_currently_producing(self) -> bool:
        """Returns true if the system is currently in a production state."""
        return self._is_active
    
    def stop_tracking(self) -> None:
        """Stops the live tracking and removes listeners."""
        if self._state_listener_remove:
            self._state_listener_remove()
            self._state_listener_remove = None
        
        if self._midnight_listener_remove:
            self._midnight_listener_remove()
            self._midnight_listener_remove = None
        
        _LOGGER.info("✓ Production time tracking stopped")