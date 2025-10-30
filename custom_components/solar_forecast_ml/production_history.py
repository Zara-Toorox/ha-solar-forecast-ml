"""
Historical Production Calculator for Solar Forecast ML.
Calculates historical metrics like the weighted peak production hour
based on recorder history.

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
from datetime import datetime, timedelta
from typing import Optional

from homeassistant.core import HomeAssistant

# Use SafeDateTimeUtil for timezone awareness
from .helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class ProductionCalculator:
    """
    Calculates historical production metrics based on recorder data,
    primarily the weighted average peak production hour.
    """

    def __init__(self, hass: HomeAssistant):
        """Initialize the historical production calculator."""
        self.hass = hass

        # Thresholds and parameters for calculation
        self.MIN_PRODUCTION_POWER_KW = 0.01 # Minimum power in kW to consider as producing
        self.PRODUCTION_START_HOUR = 5     # Typical start hour for filtering data (local time)
        self.PRODUCTION_END_HOUR = 21      # Typical end hour for filtering data (local time)

        # Weighting parameters for peak time calculation
        self.RECENT_DAYS_THRESHOLD = 3 # How many recent days get higher weight
        self.RECENT_WEIGHT = 0.7       # Weight for recent days
        self.OLDER_WEIGHT = 0.3        # Weight for older days

        _LOGGER.debug("ProductionCalculator (Historical Analysis) initialized")

    def _extract_state_value(self, state) -> Optional[str]:
        """Safely extract the state value, handling both State objects and dicts."""
        if hasattr(state, 'state'):
            return state.state
        elif isinstance(state, dict) and 'state' in state:
            return state['state']
        else:
            return None

    def _extract_last_changed(self, state) -> Optional[datetime]:
        """Safely extract the last_changed timestamp, handling both State objects and dicts."""
        if hasattr(state, 'last_changed'):
            return state.last_changed
        elif isinstance(state, dict) and 'last_changed' in state:
            # Parse if it's a string (ISO format expected)
            last_changed_str = state['last_changed']
            if isinstance(last_changed_str, str):
                return dt_util.parse_datetime(last_changed_str)
            elif isinstance(last_changed_str, datetime):
                return last_changed_str
        return None

    def _get_state_history(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> list:
        """
        Synchronous helper to fetch state history via recorder.
        Must be run in an executor job. Checks recorder status and entity filter.
        """
        try:
            from homeassistant.components import recorder
            from homeassistant.components.recorder import history

            # Check recorder status
            rec_instance = recorder.get_instance(self.hass)
            if not rec_instance or not rec_instance.is_running:
                 _LOGGER.warning(f"Recorder is not running, cannot fetch history for {entity_id}")
                 return []

            # Check if entity is recorded
            if not rec_instance.entity_filter(entity_id):
                _LOGGER.debug(f"Entity {entity_id} is excluded by recorder configuration")
                return []

            # Fetch history using the core function
            states_dict = history.get_significant_states(
                hass=self.hass,
                start_time=start_time,
                end_time=end_time,
                entity_ids=[entity_id],
                minimal_response=True, # Fetch only state and last_changed
                no_attributes=True     # Attributes not needed for this calculation
            )

            if entity_id in states_dict:
                return states_dict[entity_id]

            _LOGGER.debug(f"No history found for {entity_id} in the specified period.")
            return []

        except Exception as e:
            # Log specific error during history retrieval
            _LOGGER.error(f"Error retrieving history for {entity_id}: {e}", exc_info=True)
            return []

    async def calculate_peak_production_time(
        self,
        power_entity: Optional[str] = None
    ) -> str:
        """
        Calculates the historical peak production hour using a weighted average.
        Correctly handles W -> kW conversion assuming the input sensor provides Watts.

        Args:
            power_entity: The entity ID of the power sensor (expected in Watts).

        Returns:
            The estimated peak hour as a string "HH:00" (e.g., "14:00"),
            or "12:00" as a fallback.
        """
        try:
            _LOGGER.info("Calculating historical peak production time (Weighted Method)...")

            if not power_entity:
                _LOGGER.warning("No power sensor entity provided for peak calculation. Using fallback 12:00.")
                return "12:00"

            # Use timezone-aware 'now' and calculate start time for history query (UTC based)
            now_utc = dt_util.utcnow()
            start_time_utc = now_utc - timedelta(days=14)
            # Use local 'now' for recent day cutoff comparison
            now_local = dt_util.as_local(now_utc)
            cutoff_recent_local = now_local - timedelta(days=self.RECENT_DAYS_THRESHOLD)


            _LOGGER.debug(
                f"Analysis period (UTC): {start_time_utc.strftime('%Y-%m-%d %H:%M:%S')} "
                f"to {now_utc.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            _LOGGER.debug(
                f"Weighting cutoff (Local Time): Before {cutoff_recent_local.strftime('%Y-%m-%d %H:%M:%S')} = {self.OLDER_WEIGHT*100:.0f}%, "
                f"On or after = {self.RECENT_WEIGHT*100:.0f}%"
            )

            # Fetch history in executor
            states = await self.hass.async_add_executor_job(
                self._get_state_history,
                power_entity,
                start_time_utc,
                now_utc # Fetch up to current time
            )

            # Check if sufficient data was returned
            min_required_states = 10 # Require at least a few data points
            if not states or len(states) < min_required_states:
                _LOGGER.warning(
                    f"Insufficient history data for peak calculation ({len(states) if states else 0} states found, "
                    f"minimum {min_required_states} required). Using fallback 12:00."
                )
                return "12:00"

            _LOGGER.debug(f"Processing {len(states)} states loaded from recorder history.")

            # Prepare dictionary to hold data per hour
            hourly_data = {hour: {'values_kw': [], 'weights': []} for hour in range(24)}

            # Counters for data quality assessment
            invalid_states_count = 0
            low_power_or_night_count = 0
            watt_conversions_count = 0
            recent_points_count = 0
            older_points_count = 0
            total_valid_points = 0

            # Process each state from history
            for state in states:
                try:
                    # Ignore unavailable/unknown states
                    state_value = self._extract_state_value(state)
                    if state_value in ["unavailable", "unknown"] or state_value is None:
                        invalid_states_count += 1
                        continue

                    # Convert state to float (power value)
                    power_str = str(state_value)
                    power_value = float(power_str)

                    # --- WATT to kW Conversion (Corrected) ---
                    # Assume input is Watts (as per config requirement)
                    power_kw = 0.0
                    if power_value > 0:
                        power_kw = power_value / 1000.0
                        watt_conversions_count += 1
                    # Keep power_kw as 0.0 if input was 0 or negative

                    # Filter out very low power values (below threshold in kW)
                    if power_kw < self.MIN_PRODUCTION_POWER_KW:
                        low_power_or_night_count += 1
                        continue

                    # Determine the hour (in local time) for grouping
                    # History provides timestamps usually in UTC, convert to local
                    state_time_utc = self._extract_last_changed(state)
                    if state_time_utc is None:
                        invalid_states_count += 1
                        continue
                    state_time_local = dt_util.as_local(state_time_utc)
                    hour_local = state_time_local.hour

                    # Filter out data outside typical production hours (local time)
                    if not (self.PRODUCTION_START_HOUR <= hour_local <= self.PRODUCTION_END_HOUR):
                        low_power_or_night_count += 1
                        continue

                    # Apply weighting based on recency (using local time comparison)
                    is_recent = state_time_local >= cutoff_recent_local
                    weight = self.RECENT_WEIGHT if is_recent else self.OLDER_WEIGHT

                    if is_recent:
                        recent_points_count += 1
                    else:
                        older_points_count += 1

                    # Store the kW value and its weight for the corresponding hour
                    hourly_data[hour_local]['values_kw'].append(power_kw)
                    hourly_data[hour_local]['weights'].append(weight)
                    total_valid_points += 1

                except (ValueError, TypeError, AttributeError) as parse_error:
                    # Log parsing errors but continue processing other states
                    state_value = self._extract_state_value(state)
                    _LOGGER.debug(f"Skipping state due to parsing error: {state_value} ({parse_error})")
                    invalid_states_count += 1
                    continue

            _LOGGER.debug(
                f"Data quality assessment: "
                f"{invalid_states_count} invalid states ignored, "
                f"{low_power_or_night_count} low power/night values ignored, "
                f"{watt_conversions_count} values converted W->kW."
            )
            _LOGGER.debug(
                f"Total valid data points used for calculation: {total_valid_points} "
                f"(Recent: {recent_points_count}, Older: {older_points_count})"
            )

            # Calculate weighted average power for each hour
            hourly_weighted_averages_kw = {}
            for hour, data in hourly_data.items():
                if data['values_kw']:
                    # Use numpy for potentially better performance if installed, but fallback
                    try:
                        import numpy as np
                        weighted_avg_kw = np.average(data['values_kw'], weights=data['weights'])
                    except ImportError:
                        # Fallback calculation without numpy
                        weighted_sum = sum(v * w for v, w in zip(data['values_kw'], data['weights']))
                        weight_sum = sum(data['weights'])
                        weighted_avg_kw = weighted_sum / weight_sum if weight_sum > 0 else 0.0

                    # Store only if average is positive
                    if weighted_avg_kw > 0:
                         hourly_weighted_averages_kw[hour] = weighted_avg_kw
                         _LOGGER.debug(
                             f"Hour {hour:02d} (Local): {len(data['values_kw'])} values, "
                             f"Weighted Avg = {weighted_avg_kw:.3f} kW"
                         )

            # Check if any valid hourly averages were calculated
            if not hourly_weighted_averages_kw:
                _LOGGER.warning(
                    "No valid production data found after filtering and averaging. "
                    "Cannot calculate peak time. Using fallback 12:00."
                )
                return "12:00"

            # Find the hour with the maximum weighted average power
            peak_hour = max(hourly_weighted_averages_kw, key=hourly_weighted_averages_kw.get)
            peak_value_kw = hourly_weighted_averages_kw[peak_hour]

            # Final sanity check: ensure peak hour is within the expected production window
            if not (self.PRODUCTION_START_HOUR <= peak_hour <= self.PRODUCTION_END_HOUR):
                _LOGGER.warning(
                    f"Calculated peak hour {peak_hour:02d}:00 is outside the defined production window "
                    f"({self.PRODUCTION_START_HOUR}-{self.PRODUCTION_END_HOUR}). This might indicate data issues. "
                    "Using fallback 12:00."
                )
                return "12:00"

            _LOGGER.info(
                f"Historical peak production hour determined: {peak_hour:02d}:00 "
                f"(Weighted Avg Power: {peak_value_kw:.2f} kW)"
            )

            # Return formatted hour string
            return f"{peak_hour:02d}:00"

        except Exception as e:
            # Catch unexpected errors during the process
            _LOGGER.error(f"Peak production time calculation failed unexpectedly: {e}", exc_info=True)
            return "12:00" # Safe fallback

# ProductionTimeCalculator class has been moved to production_tracker.py