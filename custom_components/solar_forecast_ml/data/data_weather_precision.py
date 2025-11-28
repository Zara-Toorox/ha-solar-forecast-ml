"""Weather Precision Tracker - Learns weather forecast accuracy for local location V10.0.0 @zara

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

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from ..const import (
    CONF_HUMIDITY_SENSOR,
    CONF_LUX_SENSOR,
    CONF_PRESSURE_SENSOR,
    CONF_RAIN_SENSOR,
    CONF_SOLAR_RADIATION_SENSOR,
    CONF_TEMP_SENSOR,
    CONF_WIND_SENSOR,
)
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from .data_io import DataManagerIO

_LOGGER = logging.getLogger(__name__)

class WeatherPrecisionTracker(DataManagerIO):
    """Tracks weather forecast accuracy against local sensors

    ASYNC I/O: Inherits from DataManagerIO for non-blocking file operations.
    All file reads/writes use async methods to avoid blocking HA event loop.
    """

    def __init__(self, hass: HomeAssistant, config_entry, data_dir: Path):
        """Initialize the weather precision tracker @zara"""

        super().__init__(hass, data_dir)

        self.config_entry = config_entry
        self.precision_file = data_dir / "stats" / "weather_precision_daily.json"

        self.sensors = {
            "temp": config_entry.data.get(CONF_TEMP_SENSOR),
            "solar_radiation_wm2": config_entry.data.get(CONF_SOLAR_RADIATION_SENSOR),
            "humidity": config_entry.data.get(CONF_HUMIDITY_SENSOR),
            "wind": config_entry.data.get(CONF_WIND_SENSOR),
            "rain": config_entry.data.get(CONF_RAIN_SENSOR),
            "pressure": config_entry.data.get(CONF_PRESSURE_SENSOR),
        }

        _LOGGER.info(f"WeatherPrecisionTracker initialized with sensors: {[k for k, v in self.sensors.items() if v]}")

    async def _load_precision_data(self) -> Dict[str, Any]:
        """Load weather precision data from file (ASYNC) @zara"""
        if not self.precision_file.exists():
            return self._create_empty_precision_data()

        try:

            data = await self._read_json_file(self.precision_file, None)
            if data:
                _LOGGER.debug(f"Loaded weather precision data: {len(data.get('daily_tracking', {}))} days tracked")
                return data
            return self._create_empty_precision_data()
        except Exception as e:
            _LOGGER.error(f"Error loading weather precision data: {e}")
            return self._create_empty_precision_data()

    def _create_empty_precision_data(self) -> Dict[str, Any]:
        """Create empty weather precision data structure @zara"""
        return {
            "daily_tracking": {},
            "rolling_averages": {
                "sample_days": 0,
                "correction_factors": {
                    "temperature": 0.0,
                    "solar_radiation_wm2": 1.0,
                    "clouds": 1.0,
                    "humidity": 1.0,
                    "wind": 1.0,
                    "rain": 1.0,
                    "pressure": 0.0,
                },
                "confidence": {
                    "temperature": 0.0,
                    "solar_radiation_wm2": 0.0,
                    "clouds": 0.0,
                    "humidity": 0.0,
                    "wind": 0.0,
                    "rain": 0.0,
                    "pressure": 0.0,
                },
                "updated_at": None,
            },
            "metadata": {
                "created": dt_util.now().isoformat(),
                "last_updated": None,
                "total_days_tracked": 0,
                "sensors_configured": [k for k, v in self.sensors.items() if v],
                "sensors_optional": True
            }
        }

    async def _save_precision_data(self, data: Dict[str, Any]) -> bool:
        """Save weather precision data to file (ASYNC) @zara"""
        try:
            data["metadata"]["last_updated"] = dt_util.now().isoformat()

            await self._atomic_write_json(self.precision_file, data)

            _LOGGER.debug(f"Saved weather precision data to {self.precision_file}")
            return True

        except Exception as e:
            _LOGGER.error(f"Error saving weather precision data: {e}")
            return False

    async def capture_morning_forecast(self, weather_cache: Dict[str, Any]) -> bool:
        """Capture morning weather forecast at 06:00 LOCAL time @zara"""
        try:
            today = dt_util.now().date().isoformat()

            _LOGGER.info(f"Capturing morning weather forecast for {today}")

            precision_data = await self._load_precision_data()

            if today not in precision_data["daily_tracking"]:
                precision_data["daily_tracking"][today] = {
                    "forecast_captured": dt_util.now().isoformat(),
                    "hourly_data": {},
                    "daily_summary": None
                }

            today_forecast = weather_cache.get("forecasts", {}).get(today, {})
            if not today_forecast:
                _LOGGER.warning(f"No weather forecast found for {today} in weather cache")
                return False

            hourly_forecast = today_forecast.get("hourly", [])
            if not hourly_forecast:
                _LOGGER.warning(f"No hourly forecast data for {today}")
                return False

            for hour_data in hourly_forecast:
                hour = hour_data.get("hour")
                if hour is None:
                    continue

                precision_data["daily_tracking"][today]["hourly_data"][str(hour)] = {
                    "forecast": {
                        "temp": hour_data.get("temperature"),
                        "clouds": hour_data.get("cloud_cover"),
                        "humidity": hour_data.get("humidity"),
                        "wind": hour_data.get("wind_speed"),
                        "rain": hour_data.get("precipitation", 0),
                        "pressure": hour_data.get("pressure"),
                    },
                    "actual": None,
                    "deviation": None
                }

            await self._save_precision_data(precision_data)

            _LOGGER.info(f"Captured {len(hourly_forecast)} hourly forecasts for {today}")
            return True

        except Exception as e:
            _LOGGER.error(f"Error capturing morning forecast: {e}", exc_info=True)
            return False

    async def update_hourly_actuals(self) -> bool:
        """Update hourly actual values from sensors (called at :05 of each hour) @zara"""
        try:
            now = dt_util.now()

            target_hour = (now.hour - 1) % 24
            today = now.date().isoformat()

            _LOGGER.debug(f"Updating hourly actuals for {today} hour {target_hour}")

            precision_data = await self._load_precision_data()

            if today not in precision_data["daily_tracking"]:
                _LOGGER.debug(f"No forecast captured for {today} yet")
                return False

            hourly_data = precision_data["daily_tracking"][today]["hourly_data"]
            hour_key = str(target_hour)

            if hour_key not in hourly_data:
                _LOGGER.debug(f"No forecast for hour {target_hour}, skipping")
                return False

            actuals = {}

            if self.sensors["temp"]:
                temp_state = self.hass.states.get(self.sensors["temp"])
                if temp_state and temp_state.state not in ["unknown", "unavailable"]:
                    try:
                        actuals["temp"] = float(temp_state.state)
                    except (ValueError, TypeError):
                        pass

            if self.sensors["solar_radiation_wm2"]:
                solar_rad_state = self.hass.states.get(self.sensors["solar_radiation_wm2"])
                if solar_rad_state and solar_rad_state.state not in ["unknown", "unavailable"]:
                    try:
                        actuals["solar_radiation_wm2"] = float(solar_rad_state.state)
                    except (ValueError, TypeError):
                        pass

            if self.sensors["humidity"]:
                humidity_state = self.hass.states.get(self.sensors["humidity"])
                if humidity_state and humidity_state.state not in ["unknown", "unavailable"]:
                    try:
                        actuals["humidity"] = float(humidity_state.state)
                    except (ValueError, TypeError):
                        pass

            if self.sensors["wind"]:
                wind_state = self.hass.states.get(self.sensors["wind"])
                if wind_state and wind_state.state not in ["unknown", "unavailable"]:
                    try:
                        actuals["wind"] = float(wind_state.state)
                    except (ValueError, TypeError):
                        pass

            if self.sensors["rain"]:
                rain_state = self.hass.states.get(self.sensors["rain"])
                if rain_state and rain_state.state not in ["unknown", "unavailable"]:
                    try:
                        actuals["rain"] = float(rain_state.state)
                    except (ValueError, TypeError):
                        pass

            if actuals:
                hourly_data[hour_key]["actual"] = actuals
                precision_data["daily_tracking"][today]["hourly_data"] = hourly_data
                await self._save_precision_data(precision_data)

                _LOGGER.debug(f"Updated actuals for {today} hour {target_hour}: {list(actuals.keys())}")
                return True
            else:
                _LOGGER.debug(f"No sensor values available for hour {target_hour}")
                return False

        except Exception as e:
            _LOGGER.error(f"Error updating hourly actuals: {e}", exc_info=True)
            return False

    async def calculate_daily_accuracy(self, date_str: Optional[str] = None) -> bool:
        """Calculate daily accuracy at end of day (23:30 LOCAL time) @zara"""
        try:

            if date_str is None:
                yesterday = (dt_util.now() - timedelta(days=1)).date().isoformat()
                date_str = yesterday

            _LOGGER.info(f"Calculating daily accuracy for {date_str}")

            precision_data = await self._load_precision_data()

            if date_str not in precision_data["daily_tracking"]:
                _LOGGER.warning(f"No tracking data found for {date_str}")
                return False

            day_data = precision_data["daily_tracking"][date_str]
            hourly_data = day_data["hourly_data"]

            deviations_by_category = {
                "temp": [],
                "solar_radiation_wm2": [],
                "humidity": [],
                "wind": [],
                "rain": []
            }

            hours_tracked = 0

            for hour_key, hour_info in hourly_data.items():
                forecast = hour_info.get("forecast")
                actual = hour_info.get("actual")

                if not forecast or not actual:
                    continue

                hour_deviations = {}

                if "temp" in actual and forecast.get("temp") is not None:
                    forecast_temp = forecast["temp"]
                    actual_temp = actual["temp"]
                    if forecast_temp != 0:
                        deviation_pct = ((actual_temp - forecast_temp) / abs(forecast_temp)) * 100
                        hour_deviations["temp"] = round(deviation_pct, 1)
                        deviations_by_category["temp"].append(deviation_pct)

                if "solar_radiation_wm2" in actual and forecast.get("clouds") is not None:

                    cloud_pct = forecast["clouds"]
                    expected_lux = 100000 * (1 - cloud_pct / 100) + 5000
                    actual_lux = actual["solar_radiation_wm2"]
                    if expected_lux > 0:
                        deviation_pct = ((actual_lux - expected_lux) / expected_lux) * 100
                        hour_deviations["solar_radiation_wm2"] = round(deviation_pct, 1)
                        deviations_by_category["solar_radiation_wm2"].append(deviation_pct)

                if "humidity" in actual and forecast.get("humidity") is not None:
                    forecast_hum = forecast["humidity"]
                    actual_hum = actual["humidity"]
                    if forecast_hum != 0:
                        deviation_pct = ((actual_hum - forecast_hum) / abs(forecast_hum)) * 100
                        hour_deviations["humidity"] = round(deviation_pct, 1)
                        deviations_by_category["humidity"].append(deviation_pct)

                if "wind" in actual and forecast.get("wind") is not None:
                    forecast_wind = forecast["wind"]
                    actual_wind = actual["wind"]
                    if forecast_wind != 0:
                        deviation_pct = ((actual_wind - forecast_wind) / abs(forecast_wind)) * 100
                        hour_deviations["wind"] = round(deviation_pct, 1)
                        deviations_by_category["wind"].append(deviation_pct)

                if "rain" in actual and forecast.get("rain") is not None:
                    forecast_rain = forecast["rain"]
                    actual_rain = actual["rain"]

                    deviation_pct = (actual_rain - forecast_rain) * 100
                    hour_deviations["rain"] = round(deviation_pct, 1)
                    deviations_by_category["rain"].append(deviation_pct)

                hourly_data[hour_key]["deviation"] = hour_deviations
                hours_tracked += 1

            avg_deviations = {}
            max_deviations = {}

            for category, deviations in deviations_by_category.items():
                if deviations:
                    avg_deviations[category] = round(sum(deviations) / len(deviations), 1)
                    max_deviations[category] = round(max(deviations, key=abs), 1)
                else:
                    avg_deviations[category] = 0
                    max_deviations[category] = 0

            day_data["daily_summary"] = {
                "hours_tracked": hours_tracked,
                "sensors_available": [k for k, v in deviations_by_category.items() if v],
                "average_deviation_percent": avg_deviations,
                "max_deviation_percent": max_deviations
            }

            precision_data["daily_tracking"][date_str] = day_data

            self._update_rolling_averages(precision_data)

            await self._save_precision_data(precision_data)

            _LOGGER.info(
                f"Calculated daily accuracy for {date_str}: "
                f"{hours_tracked} hours tracked, avg deviations: {avg_deviations}"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Error calculating daily accuracy: {e}", exc_info=True)
            return False

    async def get_correction_factors(self) -> Dict[str, float]:
        """Get current 7-day rolling average correction factors (ASYNC) @zara"""
        try:
            precision_data = await self._load_precision_data()
            return precision_data["rolling_averages"]["correction_factors"]
        except Exception as e:
            _LOGGER.error(f"Error getting correction factors: {e}")
            return {
                "temperature": 0.0,
                "pressure": 0.0,
                "solar_radiation_wm2": 1.0,
                "wind": 1.0,
                "humidity": 1.0,
                "rain": 1.0,
                "clouds": 1.0
            }

    async def get_confidence_scores(self) -> Dict[str, float]:
        """Get current 7-day rolling average confidence scores (ASYNC) @zara"""
        try:
            precision_data = await self._load_precision_data()
            return precision_data["rolling_averages"]["confidence"]
        except Exception as e:
            _LOGGER.error(f"Error getting confidence scores: {e}")
            return {
                "temperature": 0.0,
                "solar_radiation_wm2": 0.0,
                "wind": 0.0,
                "humidity": 0.0,
                "rain": 0.0,
            }

    async def calculate_correction_factors_v2(
        self,
        weather_actual_tracker,
        weather_corrector,
        date_str: Optional[str] = None
    ) -> bool:
        """
        Calculate correction factors for ALL weather fields.

        Architecture:
            - Compares RAW Open-Meteo forecast (open_meteo_cache.json)
            - With hourly_weather_actual.json (local sensors)
            - Calculates precision factors for ALL fields
            - Updates weather_precision_daily.json with learned factors

        IMPORTANT: Must compare RAW forecast vs Actual, NOT corrected forecast!
        Otherwise we get circular corrections.

        Args:
            weather_actual_tracker: WeatherActualTracker instance
            weather_corrector: WeatherForecastCorrector instance
            date_str: Date to calculate (default: today)

        Returns:
            True if successful
        """
        try:

            if date_str is None:
                date_str = dt_util.now().date().isoformat()

            _LOGGER.info(f"Calculating correction factors for {date_str}")

            actual_weather_by_hour = await weather_actual_tracker.get_daily_actual_weather(date_str)

            if not actual_weather_by_hour:
                _LOGGER.info(
                    f"ℹ️ PRECISION: No actual weather data for {date_str}. "
                    "This is expected for days before installation or after a restart. "
                    "Weather precision tracking will start automatically with the next full day of data collection."
                )
                return False

            _LOGGER.info(f"Loaded actual weather for {len(actual_weather_by_hour)} hours")

            raw_forecast_file = self.data_dir / "data" / "open_meteo_cache.json"
            if not raw_forecast_file.exists():
                _LOGGER.warning("open_meteo_cache.json not found - skipping precision calculation")
                return False

            raw_forecast_data = await self._read_json_file(raw_forecast_file, None)
            if not raw_forecast_data:
                _LOGGER.warning("Could not read Open-Meteo cache - skipping")
                return False

            raw_forecast_by_hour = raw_forecast_data.get("forecast", {}).get(date_str, {})
            if not raw_forecast_by_hour:
                _LOGGER.warning(f"No RAW forecast data for {date_str} in Open-Meteo cache")
                return False

            _LOGGER.info(f"Loaded RAW Open-Meteo forecast for {len(raw_forecast_by_hour)} hours")

            hourly_comparisons = []

            for hour, actual_data in actual_weather_by_hour.items():
                hour_str = str(hour)

                if hour_str not in raw_forecast_by_hour:
                    _LOGGER.debug(f"No RAW forecast for hour {hour} - skipping")
                    continue

                raw_forecast_data = raw_forecast_by_hour[hour_str]

                forecast_for_comparison = self._convert_raw_forecast_to_comparison_format(raw_forecast_data)

                comparison = self._compare_weather_fields(forecast_for_comparison, actual_data, hour)
                if comparison:
                    hourly_comparisons.append(comparison)

            if not hourly_comparisons:
                _LOGGER.warning("No valid hour comparisons found")
                return False

            _LOGGER.info(f"Compared {len(hourly_comparisons)} hours successfully")

            daily_factors = self._calculate_daily_factors(hourly_comparisons)

            precision_data = await self._load_precision_data()

            if "daily_tracking" not in precision_data:
                precision_data["daily_tracking"] = {}

            precision_data["daily_tracking"][date_str] = {
                "date": date_str,
                "hours_tracked": len(hourly_comparisons),
                "daily_factors": daily_factors,
                "hourly_comparisons": hourly_comparisons,
                "calculated_at": dt_util.now().isoformat(),
            }

            self._update_rolling_averages(precision_data)

            await self._save_precision_data(precision_data)

            _LOGGER.info(
                f"✓ Correction factors calculated for {date_str}: "
                f"{len(hourly_comparisons)} hours, "
                f"factors: {daily_factors}"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Error calculating correction factors: {e}", exc_info=True)
            return False

    def _convert_raw_forecast_to_comparison_format(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Open-Meteo cache format to comparison format @zara

        Field mapping (Open-Meteo → Comparison):
        - temperature → temperature (same)
        - humidity → humidity (same)
        - pressure → pressure (same)
        - cloud_cover → clouds
        - ghi → solar_radiation_wm2
        - wind_speed → wind
        - precipitation → rain
        """
        return {
            "temperature": raw_data.get("temperature"),
            "solar_radiation_wm2": raw_data.get("ghi") or 0,
            "clouds": raw_data.get("cloud_cover"),
            "humidity": raw_data.get("humidity"),
            "wind": raw_data.get("wind_speed"),
            "rain": raw_data.get("precipitation") or 0,
            "pressure": raw_data.get("pressure"),
        }

    def _compare_weather_fields(
        self,
        forecast: Dict[str, Any],
        actual: Dict[str, Any],
        hour: int
    ) -> Optional[Dict[str, Any]]:
        """
        Compare forecast vs actual for a single hour.

        Returns:
            Dict with comparison results or None if no valid comparisons
        """
        comparison = {
            "hour": hour,
            "fields": {}
        }

        valid_comparisons = 0

        if "temperature_c" in actual and "temperature" in forecast:
            forecast_val = forecast["temperature"]
            actual_val = actual["temperature_c"]

            if forecast_val is not None and actual_val is not None:
                offset = actual_val - forecast_val
                comparison["fields"]["temperature"] = {
                    "forecast": forecast_val,
                    "actual": actual_val,
                    "offset": round(offset, 2)
                }
                valid_comparisons += 1

        if "solar_radiation_wm2" in actual and "solar_radiation_wm2" in forecast:
            forecast_val = forecast["solar_radiation_wm2"]
            actual_val = actual["solar_radiation_wm2"]

            if forecast_val is not None and actual_val is not None and forecast_val > 0:
                factor = actual_val / forecast_val
                comparison["fields"]["solar_radiation_wm2"] = {
                    "forecast": forecast_val,
                    "actual": actual_val,
                    "factor": round(factor, 3)
                }
                valid_comparisons += 1

        if "cloud_cover_percent" in actual and "clouds" in forecast:
            forecast_val = forecast["clouds"]
            actual_val = actual["cloud_cover_percent"]

            if forecast_val is not None and actual_val is not None and forecast_val != 0:
                factor = actual_val / forecast_val
                comparison["fields"]["clouds"] = {
                    "forecast": forecast_val,
                    "actual": actual_val,
                    "factor": round(factor, 3)
                }
                valid_comparisons += 1

        if "humidity_percent" in actual and "humidity" in forecast:
            forecast_val = forecast["humidity"]
            actual_val = actual["humidity_percent"]

            if forecast_val is not None and actual_val is not None and forecast_val != 0:
                factor = actual_val / forecast_val
                comparison["fields"]["humidity"] = {
                    "forecast": forecast_val,
                    "actual": actual_val,
                    "factor": round(factor, 3)
                }
                valid_comparisons += 1

        if "wind_speed_ms" in actual and "wind" in forecast:
            forecast_val = forecast["wind"]
            actual_val = actual["wind_speed_ms"]

            if forecast_val is not None and actual_val is not None and forecast_val > 0:
                factor = actual_val / forecast_val
                comparison["fields"]["wind"] = {
                    "forecast": forecast_val,
                    "actual": actual_val,
                    "factor": round(factor, 3)
                }
                valid_comparisons += 1

        if "precipitation_mm" in actual and "rain" in forecast:
            forecast_val = forecast["rain"]
            actual_val = actual["precipitation_mm"]

            if forecast_val is not None and actual_val is not None:
                difference = actual_val - forecast_val
                comparison["fields"]["rain"] = {
                    "forecast": forecast_val,
                    "actual": actual_val,
                    "difference": round(difference, 2)
                }
                valid_comparisons += 1

        if "pressure_hpa" in actual and "pressure" in forecast:
            forecast_val = forecast["pressure"]
            actual_val = actual["pressure_hpa"]

            if forecast_val is not None and actual_val is not None:
                offset = actual_val - forecast_val
                comparison["fields"]["pressure"] = {
                    "forecast": forecast_val,
                    "actual": actual_val,
                    "offset": round(offset, 2)
                }
                valid_comparisons += 1

        return comparison if valid_comparisons > 0 else None

    def _calculate_daily_factors(self, hourly_comparisons: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate daily average correction factors from hourly comparisons @zara"""

        field_offsets = {
            "temperature": [],
            "pressure": []
        }
        field_factors = {
            "solar_radiation_wm2": [],
            "clouds": [],
            "humidity": [],
            "wind": [],
        }
        field_differences = {
            "rain": []
        }

        for hour_comp in hourly_comparisons:
            fields = hour_comp.get("fields", {})

            for field_name, field_data in fields.items():

                if field_name in field_offsets:

                    if "offset" in field_data:
                        offset = field_data["offset"]

                        if -20.0 <= offset <= 20.0:
                            field_offsets[field_name].append(offset)
                elif field_name in field_factors:

                    if "factor" in field_data:
                        factor = field_data["factor"]

                        if 0.1 <= factor <= 10.0:
                            field_factors[field_name].append(factor)
                elif field_name in field_differences:

                    if "difference" in field_data:
                        diff = field_data["difference"]

                        if -50.0 <= diff <= 50.0:
                            field_differences[field_name].append(diff)

        daily_factors = {}

        for field_name, offsets in field_offsets.items():
            if offsets:
                avg_offset = sum(offsets) / len(offsets)
                daily_factors[field_name] = round(avg_offset, 3)
            else:
                daily_factors[field_name] = 0.0

        for field_name, factors in field_factors.items():
            if factors:
                avg_factor = sum(factors) / len(factors)
                daily_factors[field_name] = round(avg_factor, 3)
            else:
                daily_factors[field_name] = 1.0

        for field_name, diffs in field_differences.items():
            if diffs:
                avg_diff = sum(diffs) / len(diffs)
                daily_factors[field_name] = round(avg_diff, 3)
            else:
                daily_factors[field_name] = 0.0

        return daily_factors

    def _update_rolling_averages(self, precision_data: Dict[str, Any]) -> None:
        """Update 7-day rolling averages @zara"""
        try:
            daily_tracking = precision_data.get("daily_tracking", {})

            dates = sorted(daily_tracking.keys(), reverse=True)

            offset_fields = ["temperature", "pressure"]
            difference_fields = ["rain"]
            factor_fields = ["solar_radiation_wm2", "clouds", "humidity", "wind"]

            field_values = {
                "temperature": [],
                "solar_radiation_wm2": [],
                "clouds": [],
                "humidity": [],
                "wind": [],
                "rain": [],
                "pressure": []
            }

            valid_days = 0

            for date in dates[:7]:
                day_data = daily_tracking[date]
                daily_factors = day_data.get("daily_factors", {})

                for field_name in field_values.keys():
                    if field_name in daily_factors:
                        value = daily_factors[field_name]

                        if field_name in offset_fields or field_name in difference_fields:
                            field_values[field_name].append(value)
                        elif value != 1.0:
                            field_values[field_name].append(value)

                if daily_factors:
                    valid_days += 1

            rolling_averages = {}
            confidence_scores = {}

            for field_name, values in field_values.items():
                if values and len(values) >= 3:
                    avg_value = sum(values) / len(values)
                    rolling_averages[field_name] = round(avg_value, 3)

                    sample_confidence = min(1.0, len(values) / 7)

                    mean = avg_value
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    std_dev = math.sqrt(variance)
                    consistency = max(0.0, 1.0 - (std_dev * 2))

                    confidence = (sample_confidence + consistency) / 2
                    confidence_scores[field_name] = round(confidence, 2)
                else:

                    if field_name in offset_fields or field_name in difference_fields:
                        rolling_averages[field_name] = 0.0
                    else:
                        rolling_averages[field_name] = 1.0
                    confidence_scores[field_name] = 0.0

            precision_data["rolling_averages"] = {
                "sample_days": valid_days,
                "correction_factors": rolling_averages,
                "confidence": confidence_scores,
                "updated_at": dt_util.now().isoformat()
            }

            _LOGGER.debug(
                f"Updated 7-day rolling averages: "
                f"{valid_days} days, corrections: {rolling_averages}"
            )

        except Exception as e:
            _LOGGER.warning(f"Error updating rolling averages: {e}")
