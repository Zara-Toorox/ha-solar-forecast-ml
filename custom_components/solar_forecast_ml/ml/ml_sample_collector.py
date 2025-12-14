"""ML Training Sample Collector V12.0.0 @zara

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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from homeassistant.components.recorder import get_instance, history
from homeassistant.core import HomeAssistant, State
from homeassistant.util import dt as dt_util

from ..const import ML_MODEL_VERSION
from ..core.core_helpers import SafeDateTimeUtil
from ..data.data_manager import DataManager
from ..forecast.forecast_weather_calculator import WeatherCalculator

_LOGGER = logging.getLogger(__name__)

class SampleCollector:

    def __init__(self, hass: HomeAssistant, data_manager: DataManager):
        self.hass = hass
        self.data_manager = data_manager
        self._sample_lock = asyncio.Lock()
        self._forecast_cache: Dict[str, Any] = {}
        self.weather_entity: Optional[str] = None
        self.power_entity: Optional[str] = None
        self.temp_sensor: Optional[str] = None
        self.wind_sensor: Optional[str] = None
        self.rain_sensor: Optional[str] = None
        self.uv_sensor: Optional[str] = None
        self.lux_sensor: Optional[str] = None
        self.humidity_sensor: Optional[str] = None

        self._weather_calculator = WeatherCalculator()

    def _check_critical_sensors_available(self) -> bool:
        """Check if critical sensors especially power_entity are available @zara"""
        if not self.power_entity:
            _LOGGER.debug("Power entity not configured")
            return False

        state = self.hass.states.get(self.power_entity)
        if state is None or state.state in ["unavailable", "unknown", None]:
            _LOGGER.warning(
                f"Power entity {self.power_entity} is not available (state: {state.state if state else 'None'}). "
                f"Skipping sample collection - will retry next hour."
            )
            return False

        return True

    async def collect_sample(self, target_datetime: datetime) -> None:
        """Collects data for the specified target hour local time @zara"""

        try:
            recorder_instance = await self.hass.async_add_executor_job(
                lambda: get_instance(self.hass)
            )
            if not recorder_instance:
                _LOGGER.debug(
                    "Recorder not ready yet, skipping sample collection. "
                    "This is normal during startup."
                )
                return
        except Exception as e:
            _LOGGER.debug(f"Error checking recorder status: {e}. Skipping sample collection.")
            return

        target_local_hour = target_datetime.hour
        async with self._sample_lock:
            try:

                if not self._check_critical_sensors_available():
                    _LOGGER.info(
                        f"Critical sensors not yet available for hour {target_local_hour}. "
                        f"This is normal after restart. Will retry next hour."
                    )
                    return

                last_collected = await self.data_manager.get_last_collected_hour()

                target_datetime_local = target_datetime.replace(minute=0, second=0, microsecond=0)

                if (
                    last_collected
                    and last_collected.hour == target_datetime_local.hour
                    and last_collected.date() == target_datetime_local.date()
                ):
                    _LOGGER.debug(
                        f"Sample for hour {target_local_hour} (local) already persistently collected, skipping."
                    )
                    return

                sample_time_local = await self._collect_hourly_sample(target_datetime)

                if sample_time_local:
                    await self.data_manager.set_last_collected_hour(sample_time_local)
                    _LOGGER.debug(
                        f"Persistent status: Hour {target_local_hour} marked as collected at {sample_time_local.isoformat()}."
                    )

            except Exception as e:
                _LOGGER.error(
                    f"Hourly sample collection for hour {target_local_hour} failed: {e}",
                    exc_info=True,
                )

    async def _calculate_time_weighted_average(
        self,
        entity_id: str,
        start_time_utc: datetime,
        end_time_utc: datetime,
        attribute: Optional[str] = None,
    ) -> Optional[float]:
        """Calculates the time-weighted average of a sensor or attribute"""
        _LOGGER.debug(
            f"Calculating TWA for {entity_id} (Attribute: {attribute}) from {start_time_utc} to {end_time_utc}"
        )

        try:
            history_list = await self.hass.async_add_executor_job(
                history.get_significant_states,
                self.hass,
                start_time_utc,
                end_time_utc,
                [entity_id],
                None,
                True,
                True,
            )

            if not history_list or entity_id not in history_list:
                _LOGGER.warning(f"No history for TWA of {entity_id} found.")
                return None

            states = history_list[entity_id]
            if not states:
                _LOGGER.debug(f"History for TWA of {entity_id} is empty.")
                return None

        except Exception as e:
            _LOGGER.error(f"Error fetching history for TWA of {entity_id}: {e}")
            return None

        total_value_duration = 0.0
        total_duration_sec = 0.0

        initial_value = None
        for state in states:
            if state.last_updated <= start_time_utc:
                try:
                    val_str = state.attributes.get(attribute) if attribute else state.state
                    initial_value = float(val_str)
                except (ValueError, TypeError, AttributeError):
                    continue
            else:
                break

        if initial_value is None:
            _LOGGER.warning(
                f"Could not find initial value for TWA of {entity_id}. Start with first value in the window."
            )

            for state in states:
                if state.last_updated > start_time_utc:
                    try:
                        val_str = state.attributes.get(attribute) if attribute else state.state
                        initial_value = float(val_str)
                        break
                    except (ValueError, TypeError, AttributeError):
                        continue
            if initial_value is None:
                _LOGGER.warning(
                    f"No valid numeric values for TWA of {entity_id} (attr={attribute}) found in the entire period. "
                    f"Sensor may be unavailable or misconfigured. Returning None."
                )
                return None

        prev_value = initial_value
        prev_time = start_time_utc

        start_index_integration = 0
        while (
            start_index_integration < len(states)
            and states[start_index_integration].last_updated <= start_time_utc
        ):
            start_index_integration += 1

        for state in states[start_index_integration:]:
            try:
                current_time = min(state.last_updated, end_time_utc)
                duration_sec = (current_time - prev_time).total_seconds()

                if duration_sec > 0:
                    total_value_duration += prev_value * duration_sec
                    total_duration_sec += duration_sec

                val_str = state.attributes.get(attribute) if attribute else state.state
                prev_value = float(val_str)
                prev_time = current_time

                if prev_time >= end_time_utc:
                    break

            except (ValueError, TypeError, AttributeError):

                if val_str not in ("unavailable", "unknown", "none", None, ""):
                    _LOGGER.debug(
                        f"Invalid state '{val_str}' for TWA of {entity_id}, using previous value."
                    )
                prev_time = min(state.last_updated, end_time_utc)
            except Exception as e_loop:
                _LOGGER.warning(f"Error in TWA loop: {e_loop}")
                prev_time = min(state.last_updated, end_time_utc)

        if prev_time < end_time_utc:
            duration_sec = (end_time_utc - prev_time).total_seconds()
            total_value_duration += prev_value * duration_sec
            total_duration_sec += duration_sec

        if total_duration_sec == 0:
            _LOGGER.debug(f"TWA for {entity_id}: No duration (0s), use initial_value.")
            return initial_value

        average = total_value_duration / total_duration_sec
        _LOGGER.debug(f"TWA for {entity_id} (Attribute: {attribute}) = {average:.2f}")
        return average

    async def _get_dominant_condition(
        self, start_time_utc: datetime, end_time_utc: datetime, entity_id: str
    ) -> str:
        """Determines the weather condition that lasted the longest in the time window"""
        _LOGGER.debug(
            f"Determining dominant condition for {entity_id} from {start_time_utc} to {end_time_utc}"
        )
        try:
            history_list = await self.hass.async_add_executor_job(
                history.get_significant_states,
                self.hass,
                start_time_utc,
                end_time_utc,
                [entity_id],
                None,
                True,
                True,
            )

            if not history_list or entity_id not in history_list:
                _LOGGER.warning(f"No history for condition of {entity_id} found.")
                return "unknown"

            states = history_list[entity_id]
            if not states:
                _LOGGER.debug(f"History for condition of {entity_id} is empty.")
                return "unknown"

            durations: Dict[str, float] = {}

            prev_state_str = "unknown"
            for state in states:
                if state.last_updated <= start_time_utc:
                    prev_state_str = state.state
                else:
                    break

            prev_time = start_time_utc

            start_index_integration = 0
            while (
                start_index_integration < len(states)
                and states[start_index_integration].last_updated <= start_time_utc
            ):
                start_index_integration += 1

            for state in states[start_index_integration:]:
                current_time = min(state.last_updated, end_time_utc)
                duration_sec = (current_time - prev_time).total_seconds()

                if duration_sec > 0 and prev_state_str not in ["unknown", "unavailable"]:
                    durations[prev_state_str] = durations.get(prev_state_str, 0.0) + duration_sec

                prev_state_str = state.state
                prev_time = current_time
                if prev_time >= end_time_utc:
                    break

            if prev_time < end_time_utc:
                duration_sec = (end_time_utc - prev_time).total_seconds()
                if duration_sec > 0 and prev_state_str not in ["unknown", "unavailable"]:
                    durations[prev_state_str] = durations.get(prev_state_str, 0.0) + duration_sec

            if not durations:
                _LOGGER.warning(f"No valid conditions for {entity_id} found in the period.")
                return "unknown"

            dominant_condition = max(durations, key=durations.get)
            _LOGGER.debug(
                f"Dominant condition for {entity_id} is '{dominant_condition}' (Durations: {durations})"
            )
            return dominant_condition

        except Exception as e:
            _LOGGER.error(f"Error determining dominant condition: {e}", exc_info=True)
            return "unknown"

    async def _get_historical_forecast_data(
        self, target_time_utc: datetime
    ) -> Optional[Dict[str, Any]]:
        """Fetches historical forecast data from Open-Meteo cache"""
        try:
            import json
            open_meteo_file = self.data_manager.data_dir / "data" / "open_meteo_cache.json"

            if not open_meteo_file.exists():
                _LOGGER.debug("Open-Meteo cache not available")
                return None

            def _read_cache():
                with open(open_meteo_file, 'r') as f:
                    return json.load(f)

            cache = await self.hass.async_add_executor_job(_read_cache)
            forecast = cache.get("forecast", {})

            if not forecast:
                _LOGGER.debug("Open-Meteo cache empty")
                return None

            target_local = target_time_utc.astimezone()
            target_date = target_local.date().isoformat()
            target_hour = target_local.hour

            if target_date in forecast and str(target_hour) in forecast[target_date]:
                entry = forecast[target_date][str(target_hour)]

                weather_data = {
                    "temperature": entry.get("temperature", 15.0),
                    "humidity": entry.get("humidity", 60.0),
                    "cloud_cover": entry.get("cloud_cover", 50.0),
                    "wind_speed": entry.get("wind_speed", 5.0),
                    "pressure": entry.get("pressure", 1013.0),
                    "ghi": entry.get("ghi", 0.0),
                    "direct_radiation": entry.get("direct_radiation", 0.0),
                    "diffuse_radiation": entry.get("diffuse_radiation", 0.0),
                    "condition": "open-meteo",
                }

                _LOGGER.debug(
                    f"Open-Meteo data for {target_date} {target_hour}:00: "
                    f"temp={weather_data['temperature']} cloud={weather_data['cloud_cover']}% "
                    f"ghi={weather_data['ghi']}W/m²"
                )
                return weather_data

            _LOGGER.debug(f"No Open-Meteo data for {target_date} {target_hour}:00")
            return None

        except Exception as e:
            _LOGGER.error(f"Error fetching Open-Meteo data: {e}", exc_info=True)
            return None

    async def _get_historical_average_states(
        self, start_time_utc: datetime, end_time_utc: datetime
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Fetches historical weather data primarily from forecast cache"""
        _LOGGER.debug(f"Fetching historical data for {start_time_utc} to {end_time_utc}")

        weather_data = self._get_default_weather()
        sensor_data = self._get_default_sensor_data()

        mid_time_utc = start_time_utc + (end_time_utc - start_time_utc) / 2
        forecast_data = await self._get_historical_forecast_data(mid_time_utc)

        if forecast_data:

            weather_data.update(forecast_data)
            _LOGGER.debug(f"Using historical forecast for weather data")
            forecast_available = True
        else:
            _LOGGER.debug(f"No historical forecast available, use TWA for weather entity")
            forecast_available = False

        sensor_entities: List[Tuple[str, Optional[str], str, Dict]] = [
            (self.temp_sensor, None, "temperature", sensor_data),
            (self.wind_sensor, None, "wind_speed", sensor_data),
            (self.rain_sensor, None, "rain", sensor_data),
            (self.uv_sensor, None, "uv_index", sensor_data),
            (self.lux_sensor, None, "lux", sensor_data),
            (self.humidity_sensor, None, "humidity", sensor_data),
        ]

        tasks = []

        for entity_id, attr, key, target_dict in sensor_entities:
            if entity_id:
                tasks.append(
                    self._calculate_time_weighted_average(
                        entity_id, start_time_utc, end_time_utc, attr
                    )
                )
            else:
                tasks.append(asyncio.sleep(0, result=None))

        if not forecast_available and self.weather_entity:
            weather_fallback_attrs = [
                ("temperature", "temperature"),
                ("humidity", "humidity"),
                ("wind_speed", "wind_speed"),
                ("pressure", "pressure"),
                ("cloud_coverage", "cloud_cover"),
            ]
            for attr, key in weather_fallback_attrs:
                tasks.append(
                    self._calculate_time_weighted_average(
                        self.weather_entity, start_time_utc, end_time_utc, attr
                    )
                )

        if self.weather_entity and not forecast_available:
            tasks.append(
                self._get_dominant_condition(start_time_utc, end_time_utc, self.weather_entity)
            )
        else:
            tasks.append(asyncio.sleep(0, result=None))

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=30.0
            )
        except asyncio.TimeoutError:
            _LOGGER.error(f"Timeout fetching TWA values after 30 seconds - using defaults")
            return weather_data, sensor_data
        except Exception as e:
            _LOGGER.error(f"Error fetching TWA values: {e}", exc_info=True)

            return weather_data, sensor_data

        for i, (entity_id, attr, key, target_dict) in enumerate(sensor_entities):
            if i < len(results):
                result = results[i]

                if isinstance(result, Exception):
                    _LOGGER.debug(f"Sensor {entity_id} ({key}) raised exception: {result}")
                    continue
                if result is not None:
                    try:
                        target_dict[key] = max(0.0, float(result))
                    except (ValueError, TypeError) as e:
                        _LOGGER.debug(f"Cannot convert result {result} to float for {key}: {e}")

        if not forecast_available:
            offset = len(sensor_entities)
            weather_attrs = ["temperature", "humidity", "wind_speed", "pressure", "cloud_cover"]
            for i, key in enumerate(weather_attrs):
                idx = offset + i
                if idx < len(results):
                    result = results[idx]

                    if isinstance(result, Exception):
                        _LOGGER.debug(f"Weather attr {key} raised exception: {result}")
                        continue
                    if result is not None:
                        try:
                            weather_data[key] = max(0.0, float(result))
                        except (ValueError, TypeError) as e:
                            _LOGGER.debug(f"Cannot convert result {result} to float for {key}: {e}")

            if len(results) > offset + len(weather_attrs):
                condition_result = results[offset + len(weather_attrs)]
                if isinstance(condition_result, Exception):
                    _LOGGER.debug(f"Condition raised exception: {condition_result}")
                elif condition_result:
                    weather_data["condition"] = condition_result

        _LOGGER.debug(f"Historical weather data: {weather_data}")
        _LOGGER.debug(f"Historical sensor data: {sensor_data}")
        return weather_data, sensor_data

    async def _collect_hourly_sample(self, target_datetime: datetime) -> Optional[datetime]:
        """Collects data for the specified target hour local time @zara"""
        try:

            start_time_utc, end_time_utc, sample_time_local = self._get_utc_times_for_hour(
                target_datetime
            )

            if start_time_utc is None or end_time_utc is None or sample_time_local is None:
                _LOGGER.warning(
                    f"Could not determine UTC time window for local hour {target_datetime.hour}. Sample skipped."
                )
                return None

            actual_kwh = await self._perform_riemann_integration(start_time_utc, end_time_utc)
            if actual_kwh is None:
                _LOGGER.warning(
                    f"Could not fetch actual production (Riemann) for hour {target_datetime.hour} (local) (None). Sample skipped."
                )
                return None

            daily_total = await self._get_daily_production_so_far(end_time_utc)
            if daily_total is None:
                _LOGGER.warning(
                    f"Could not fetch daily total up to hour {target_datetime.hour}. Set to 0 for sample."
                )
                daily_total = 0.0

            percentage = 0.0
            if daily_total > 0.01:
                percentage = max(0.0, actual_kwh) / daily_total
            elif actual_kwh <= 0.01 and daily_total <= 0.01:
                percentage = 0.0

            weather_data, sensor_data = await self._get_historical_average_states(
                start_time_utc, end_time_utc
            )

            _LOGGER.debug(
                f"Hourly sample collected: {sample_time_local.strftime('%Y-%m-%d %H')}:00 local | "
                f"Actual={actual_kwh:.2f}kWh ({percentage*100:.1f}% of day)"
            )

            return sample_time_local

        except Exception as e:
            _LOGGER.error(
                f"Error collecting (historical) hourly sample for hour {target_datetime.hour}: {e}",
                exc_info=True,
            )
            return None

    def _get_utc_times_for_hour(
        self, target_datetime: datetime
    ) -> Tuple[Optional[datetime], Optional[datetime], Optional[datetime]]:
        """Calculates the UTC startend window and local sample timestamp"""
        try:
            now_local = SafeDateTimeUtil.now()

            start_time_local = target_datetime.replace(minute=0, second=0, microsecond=0)
            end_time_local = start_time_local + timedelta(hours=1)

            if end_time_local > now_local:
                _LOGGER.warning(
                    f"Attempt to collect data for future/current hour {start_time_local.hour} (up to {end_time_local}). Skip."
                )
                return None, None, None

            start_time_utc = start_time_local.astimezone(timezone.utc)
            end_time_utc = end_time_local.astimezone(timezone.utc)

            sample_time_local = start_time_local

            return start_time_utc, end_time_utc, sample_time_local

        except ValueError as e:
            _LOGGER.warning(
                f"Could not create local start time for hour {target_datetime.hour} (possibly daylight saving?): {e}"
            )
            return None, None, None
        except Exception as tz_err:
            _LOGGER.error(f"Error in timezone conversion for hour {target_datetime.hour}: {tz_err}")
            return None, None, None

    async def _safe_parse_yield(self, state_obj: Optional[Any]) -> Optional[float]:
        if not state_obj or state_obj.state in ["unavailable", "unknown", "none", None]:
            return None
        try:
            val = float(state_obj.state)
            return val if val >= 0 else 0.0
        except (ValueError, TypeError):
            try:
                cleaned_state = str(state_obj.state).split(" ")[0].replace(",", ".")
                val = float(cleaned_state)
                _LOGGER.debug(f"State '{state_obj.state}' normalized to {val}")
                return val if val >= 0 else 0.0
            except (ValueError, TypeError):
                _LOGGER.warning(
                    f"Could not convert normalized state to number: '{state_obj.state}'"
                )
                return None

    async def _perform_riemann_integration(
        self, start_time: datetime, end_time: datetime
    ) -> Optional[float]:
        if not self.power_entity:
            _LOGGER.debug("No power_entity configured for Riemann sum")
            return None

        _LOGGER.debug(
            f"Starting Riemann sum for {self.power_entity} from {start_time} to {end_time}"
        )

        try:
            if start_time.tzinfo is None or start_time.tzinfo.utcoffset(start_time) is None:
                _LOGGER.error("Riemann: start_time must be timezone-aware (UTC).")
                start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time.tzinfo is None or end_time.tzinfo.utcoffset(end_time) is None:
                _LOGGER.error("Riemann: end_time must be timezone-aware (UTC).")
                end_time = end_time.replace(tzinfo=timezone.utc)

            history_list = await self.hass.async_add_executor_job(
                history.get_significant_states,
                self.hass,
                start_time,
                end_time,
                [self.power_entity],
                None,
                True,
                True,
            )

            if not history_list or self.power_entity not in history_list:
                _LOGGER.warning(
                    f"No history for {self.power_entity} from {start_time} to {end_time} found."
                )
                current_state = self.hass.states.get(self.power_entity)
                if current_state is None:
                    _LOGGER.error(f"Power entity {self.power_entity} not found in Home Assistant!")
                    return None
                else:
                    _LOGGER.warning(
                        f"Sensor {self.power_entity} exists, but provides no history in the period. Result is 0.0 kWh."
                    )
                    return 0.0

            states = history_list[self.power_entity]

            if not states:
                _LOGGER.debug(f"History for {self.power_entity} in period is empty.")
                return 0.0

            total_wh = 0.0
            initial_state_value = 0.0
            initial_state_time = start_time
            found_initial = False
            for state in states:
                if state.last_updated <= start_time:
                    try:
                        initial_state_value = max(0.0, float(state.state))
                        initial_state_time = state.last_updated
                        found_initial = True
                    except (ValueError, TypeError):
                        continue
                else:
                    break

            if not found_initial:
                _LOGGER.warning(
                    f"Could not find valid state before/at {start_time} for Riemann start. Start with 0W."
                )
            prev_power = initial_state_value
            prev_time = start_time

            _LOGGER.debug(
                f"Riemann Start: Initial prev_power = {prev_power}W (based on state around {initial_state_time})"
            )

            start_index_integration = 0
            while (
                start_index_integration < len(states)
                and states[start_index_integration].last_updated <= start_time
            ):
                start_index_integration += 1

            for state in states[start_index_integration:]:
                try:
                    state_time = state.last_updated
                    current_end_time = min(state_time, end_time)
                    time_diff_seconds = (current_end_time - prev_time).total_seconds()

                    if time_diff_seconds <= 1e-6:
                        _LOGGER.debug(
                            f"Skip Riemann step: Time difference is zero or negative at {current_end_time}."
                        )
                        prev_time = current_end_time
                        try:
                            prev_power = max(0.0, float(state.state))
                        except (ValueError, TypeError):
                            pass
                        continue

                    time_diff_hours = time_diff_seconds / 3600.0
                    wh = prev_power * time_diff_hours
                    total_wh += wh

                    prev_time = current_end_time

                    try:
                        current_power = max(0.0, float(state.state))
                        prev_power = current_power
                    except (ValueError, TypeError):

                        if state.state not in ("unavailable", "unknown", "none", None, ""):
                            _LOGGER.debug(
                                f"Invalid state '{state.state}' at {state_time}. Continue with previous value {prev_power}W."
                            )

                    if prev_time >= end_time:
                        _LOGGER.debug(f"Riemann reached/exceeds end time {end_time}.")
                        break

                except Exception as loop_err:
                    _LOGGER.error(
                        f"Error in Riemann loop at time {state.last_updated}: {loop_err}",
                        exc_info=True,
                    )
                    if hasattr(state, "last_updated"):
                        prev_time = min(state.last_updated, end_time)
                    continue

            if prev_time < end_time:
                time_diff_seconds = (end_time - prev_time).total_seconds()
                if time_diff_seconds > 1e-6:
                    time_diff_hours = time_diff_seconds / 3600.0
                    wh = prev_power * time_diff_hours
                    total_wh += wh

            kwh = max(0.0, total_wh / 1000.0)
            _LOGGER.debug(
                f"Riemann integration completed: {kwh:.4f} kWh (from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')})"
            )
            return kwh

        except Exception as e:
            _LOGGER.error(
                f"Critical error in Riemann integration for {self.power_entity}: {e}", exc_info=True
            )
            return None

    async def _get_daily_production_so_far(self, end_of_hour_utc: datetime) -> Optional[float]:
        """Fetches the daily production so far via Riemann integration up to the end of ... @zara"""

        end_of_hour_local = SafeDateTimeUtil.as_local(end_of_hour_utc)
        start_of_day_local = end_of_hour_local.replace(hour=0, minute=0, second=0, microsecond=0)

        try:
            start_of_day_utc = start_of_day_local.astimezone(timezone.utc)
        except Exception as tz_err:
            _LOGGER.error(f"Error in timezone conversion for daily production: {tz_err}")
            return None

        _LOGGER.debug(
            f"Calculate daily production up to end of hour {end_of_hour_local.hour} (local): "
            f"UTC period: {start_of_day_utc.isoformat()} to {end_of_hour_utc.isoformat()}"
        )

        kwh = await self._perform_riemann_integration(start_of_day_utc, end_of_hour_utc)

        if kwh is not None:
            _LOGGER.debug(
                f"Daily yield up to end of hour {end_of_hour_local.hour} (Riemann): {kwh:.2f} kWh"
            )
        else:
            _LOGGER.warning(
                f"Fetching daily yield (up to hour {end_of_hour_local.hour}) via Riemann failed."
            )

        return kwh

    def _get_default_weather(self) -> Dict[str, Any]:
        """Returns default weather values @zara"""
        return {
            "temperature": 15.0,
            "humidity": 60.0,
            "cloud_cover": 50.0,
            "wind_speed": 5.0,
            "pressure": 1013.0,
            "condition": "unknown",
        }

    def _get_default_sensor_data(self) -> Dict[str, Any]:
        """Returns default sensor values @zara"""
        return {
            "temperature": 0.0,
            "wind_speed": 0.0,
            "rain": 0.0,
            "uv_index": 0.0,
            "lux": 0.0,
            "humidity": 0.0,
        }

    def set_forecast_cache(self, cache: Dict[str, Any]) -> None:
        self._forecast_cache = cache

    def configure_entities(
        self,
        weather_entity: Optional[str] = None,
        power_entity: Optional[str] = None,
        temp_sensor: Optional[str] = None,
        wind_sensor: Optional[str] = None,
        rain_sensor: Optional[str] = None,
        lux_sensor: Optional[str] = None,
        humidity_sensor: Optional[str] = None,
        pressure_sensor: Optional[str] = None,
        solar_radiation_sensor: Optional[str] = None,
        solar_yield_today: Optional[str] = None,
    ) -> None:
        """Configures the entity IDs used by the collector"""
        _LOGGER.debug("Configuring entities in SampleCollector...")
        self.weather_entity = weather_entity
        self.power_entity = power_entity
        self.temp_sensor = temp_sensor
        self.wind_sensor = wind_sensor
        self.rain_sensor = rain_sensor
        self.lux_sensor = lux_sensor
        self.humidity_sensor = humidity_sensor
        self.pressure_sensor = pressure_sensor
        self.solar_radiation_sensor = solar_radiation_sensor
        _LOGGER.debug(
            f"SampleCollector Entities: Weather='{weather_entity}', Power='{power_entity}', "
            f"Temp='{temp_sensor}', Wind='{wind_sensor}', Rain='{rain_sensor}', "
            f"Lux='{lux_sensor}', Humidity='{humidity_sensor}', "
            f"Pressure='{pressure_sensor}', SolarRadiation='{solar_radiation_sensor}'"
        )
