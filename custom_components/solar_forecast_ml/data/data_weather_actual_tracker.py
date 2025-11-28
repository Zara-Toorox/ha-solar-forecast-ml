"""Weather Actual Tracker V10.0.0 @zara

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
from pathlib import Path
from typing import Dict, Optional, Any
import json

from .data_frost_detection import FrostDetector

_LOGGER = logging.getLogger(__name__)

class WeatherActualTracker:
    """Tracks actual weather conditions from local sensors"""

    def __init__(self, hass, data_dir: Path, config_entry=None):
        """Initialize weather actual tracker @zara"""
        self.hass = hass
        self.data_dir = data_dir
        self.stats_dir = data_dir / "stats"
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        self.actual_file = self.stats_dir / "hourly_weather_actual.json"

        self.frost_detector = FrostDetector()

        if config_entry:
            from ..const import (
                CONF_TEMP_SENSOR,
                CONF_HUMIDITY_SENSOR,
                CONF_WIND_SENSOR,
                CONF_RAIN_SENSOR,
                CONF_PRESSURE_SENSOR,
                CONF_SOLAR_RADIATION_SENSOR,
                CONF_LUX_SENSOR,
                CONF_WEATHER_ENTITY,
            )

            config = config_entry.data
            self.sensor_mapping = {
                "temperature_c": config.get(CONF_TEMP_SENSOR),
                "humidity_percent": config.get(CONF_HUMIDITY_SENSOR),
                "wind_speed_ms": config.get(CONF_WIND_SENSOR),
                "precipitation_mm": config.get(CONF_RAIN_SENSOR),
                "pressure_hpa": config.get(CONF_PRESSURE_SENSOR),
                "solar_radiation_wm2": config.get(CONF_SOLAR_RADIATION_SENSOR),
                "lux": config.get(CONF_LUX_SENSOR),
                "condition": config.get(CONF_WEATHER_ENTITY),

            }

            self.weather_entity_id = config.get(CONF_WEATHER_ENTITY)
            _LOGGER.info("Weather Actual Tracker initialized with config_entry sensors")
        else:

            self.sensor_mapping = {
                "temperature_c": "sensor.outdoor_temperature",
                "humidity_percent": "sensor.outdoor_humidity",
                "wind_speed_ms": "sensor.wind_speed",
                "precipitation_mm": "sensor.rain_rate",
                "pressure_hpa": "sensor.barometric_pressure",
                "solar_radiation_wm2": "sensor.solar_radiation",
                "condition": "weather.home",

            }
            self.weather_entity_id = "weather.home"
            _LOGGER.warning("Weather Actual Tracker initialized WITHOUT config_entry - using hardcoded defaults!")

    async def track_current_weather(self) -> Dict[str, Any]:
        """Track current weather conditions from local sensors @zara"""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        hour = now.hour

        _LOGGER.info(f"Tracking actual weather for {date_str} hour {hour}")

        weather_data = await self._read_sensors()

        if not weather_data:
            _LOGGER.warning("No local weather sensors available - skipping tracking")
            return {}

        weather_data["timestamp"] = now.isoformat()
        weather_data["source"] = "homeassistant_sensors"

        cloud_cover = await self._calculate_cloud_cover_from_radiation(
            weather_data.get("solar_radiation_wm2"),
            date_str,
            hour
        )
        if cloud_cover is not None:
            weather_data["cloud_cover_percent"] = cloud_cover
            _LOGGER.debug(f"Calculated cloud_cover_percent: {cloud_cover}% from W/m² sensor")

        frost_result = await self._detect_frost(weather_data, date_str, hour)
        if frost_result:
            weather_data["frost_detected"] = frost_result.get("frost_detected")
            weather_data["frost_score"] = frost_result.get("frost_score")
            weather_data["frost_confidence"] = frost_result.get("confidence")

            if "frost_analysis" in frost_result:
                weather_data["frost_analysis"] = frost_result["frost_analysis"]

            if frost_result.get("frost_detected"):

                frost_analysis = frost_result.get("frost_analysis", {})
                dewpoint = frost_analysis.get("dewpoint_c")
                frost_prob = frost_analysis.get("frost_probability", 0)
                corr_diff = frost_analysis.get("correlation_diff_percent")

                if dewpoint is not None:
                    _LOGGER.warning(
                        f"⚠️ FROST DETECTED (hour {hour}): {frost_result.get('reason')} "
                        f"[dewpoint={dewpoint}°C, prob={frost_prob:.0%}, corr_diff={corr_diff}%]"
                    )
                else:
                    _LOGGER.warning(
                        f"⚠️ FROST DETECTED (hour {hour}): {frost_result.get('reason')}"
                    )

        astronomy_fields = await self._get_astronomy_time_fields(date_str, hour)
        if astronomy_fields:
            weather_data["hours_after_sunrise"] = astronomy_fields.get("hours_after_sunrise")
            weather_data["hours_before_sunset"] = astronomy_fields.get("hours_before_sunset")
            _LOGGER.debug(
                f"Added astronomy fields: hours_after_sunrise={astronomy_fields.get('hours_after_sunrise')}, "
                f"hours_before_sunset={astronomy_fields.get('hours_before_sunset')}"
            )

        await self._save_hourly_data(date_str, hour, weather_data)

        _LOGGER.info(
            f"Saved actual weather for {date_str} {hour:02d}:00 "
            f"(fields: {', '.join(k for k, v in weather_data.items() if v is not None and k not in ['timestamp', 'source'])})"
        )

        return weather_data

    async def _read_sensors(self) -> Dict[str, Any]:
        """Read current values from all configured sensors @zara"""
        data = {}
        available_count = 0

        for field, entity_id in self.sensor_mapping.items():
            if not entity_id:
                continue

            try:
                state = self.hass.states.get(entity_id)

                if state is None:
                    _LOGGER.debug(f"Sensor {entity_id} not found (field: {field})")
                    continue

                value = state.state

                if value in ["unavailable", "unknown", None]:
                    _LOGGER.debug(f"Sensor {entity_id} unavailable (field: {field})")
                    continue

                if field == "condition":
                    data[field] = value
                    available_count += 1
                    _LOGGER.debug(f"✓ {field}: {value} (from {entity_id})")
                else:

                    try:
                        numeric_value = float(value)
                        data[field] = round(numeric_value, 2)
                        available_count += 1
                        _LOGGER.debug(f"✓ {field}: {numeric_value} (from {entity_id})")
                    except (ValueError, TypeError):
                        _LOGGER.debug(f"Could not convert {entity_id} value '{value}' to float")
                        continue

            except Exception as e:
                _LOGGER.debug(f"Error reading sensor {entity_id} (field: {field}): {e}")
                continue

        _LOGGER.info(f"Read {available_count} sensors successfully")
        return data if available_count > 0 else {}

    async def _calculate_cloud_cover_from_radiation(
        self,
        measured_wm2: Optional[float],
        date_str: str,
        hour: int
    ) -> Optional[float]:
        """
        Calculate actual cloud cover from measured solar radiation.

        Uses the ratio of measured W/m² to theoretical clear-sky maximum.
        This gives the ACTUAL cloud cover, not the forecasted one.

        Formula:
            cloud_cover = (1 - measured/theoretical_max) * 100

        Args:
            measured_wm2: Measured solar radiation from local sensor (W/m²)
            date_str: Date string (YYYY-MM-DD)
            hour: Hour (0-23)

        Returns:
            Cloud cover percentage (0-100) or None if calculation not possible
        """
        if measured_wm2 is None:
            _LOGGER.debug("No solar radiation sensor - cannot calculate cloud cover")
            return None

        try:

            astronomy_file = self.stats_dir / "astronomy_cache.json"

            if not astronomy_file.exists():
                _LOGGER.debug("Astronomy cache not found - cannot calculate cloud cover")
                return None

            astronomy_data = await self.hass.async_add_executor_job(
                self._read_json_sync, astronomy_file
            )

            day_data = astronomy_data.get("days", {}).get(date_str, {})
            hour_data = day_data.get("hourly", {}).get(str(hour), {})

            theoretical_max_wm2 = hour_data.get("clear_sky_solar_radiation_wm2")

            if theoretical_max_wm2 is None or theoretical_max_wm2 <= 0:
                _LOGGER.debug(f"No theoretical max for {date_str} hour {hour} - cannot calculate cloud cover")
                return None

            calibration_factor = await self._get_solar_radiation_correction_factor()

            calibrated_wm2 = measured_wm2 / calibration_factor if calibration_factor > 0 else measured_wm2

            ratio = calibrated_wm2 / theoretical_max_wm2

            ratio = max(0.0, min(1.0, ratio))

            cloud_cover = (1.0 - ratio) * 100.0

            cloud_cover = round(cloud_cover, 1)

            _LOGGER.debug(
                f"Cloud cover calculation: measured={measured_wm2} W/m², "
                f"calibration_factor={calibration_factor}, calibrated={calibrated_wm2:.1f} W/m², "
                f"theoretical_max={theoretical_max_wm2} W/m², "
                f"ratio={ratio:.2f}, cloud_cover={cloud_cover}%"
            )

            return cloud_cover

        except Exception as e:
            _LOGGER.warning(f"Error calculating cloud cover from radiation: {e}")
            return None

    async def _get_solar_radiation_correction_factor(self) -> float:
        """Load solar_radiation_wm2 correction factor from weather_forecast_corrected.json @zara"""
        try:
            corrected_file = self.stats_dir / "weather_forecast_corrected.json"

            if not corrected_file.exists():
                return 1.0

            data = await self.hass.async_add_executor_job(
                self._read_json_sync, corrected_file
            )

            factor = data.get("metadata", {}).get("corrections_applied", {}).get("solar_radiation_wm2", 1.0)

            if factor is None or factor <= 0:
                return 1.0

            return float(factor)

        except Exception as e:
            _LOGGER.debug(f"Could not load solar radiation correction factor: {e}")
            return 1.0

    async def _get_astronomy_time_fields(
        self,
        date_str: str,
        hour: int
    ) -> Optional[Dict[str, float]]:
        """
        Calculate hours_after_sunrise and hours_before_sunset from astronomy cache.

        Args:
            date_str: Date string (YYYY-MM-DD)
            hour: Hour (0-23)

        Returns:
            Dict with hours_after_sunrise and hours_before_sunset, or None if not available
        """
        try:
            astronomy_file = self.stats_dir / "astronomy_cache.json"

            if not astronomy_file.exists():
                _LOGGER.debug("Astronomy cache not found - cannot calculate time fields")
                return None

            astronomy_data = await self.hass.async_add_executor_job(
                self._read_json_sync, astronomy_file
            )

            day_data = astronomy_data.get("days", {}).get(date_str, {})

            if not day_data:
                _LOGGER.debug(f"No astronomy data for {date_str}")
                return None

            sunrise_str = day_data.get("sunrise_local")
            sunset_str = day_data.get("sunset_local")

            if not sunrise_str or not sunset_str:
                _LOGGER.debug(f"Missing sunrise/sunset for {date_str}")
                return None

            sunrise = datetime.fromisoformat(sunrise_str)
            sunset = datetime.fromisoformat(sunset_str)

            current_dt = datetime.fromisoformat(f"{date_str}T{hour:02d}:00:00")

            hours_after_sunrise = (current_dt - sunrise).total_seconds() / 3600.0
            hours_before_sunset = (sunset - current_dt).total_seconds() / 3600.0

            return {
                "hours_after_sunrise": round(hours_after_sunrise, 2),
                "hours_before_sunset": round(hours_before_sunset, 2),
            }

        except Exception as e:
            _LOGGER.debug(f"Error calculating astronomy time fields: {e}")
            return None

    async def _detect_frost(
        self,
        weather_data: Dict[str, Any],
        date_str: str,
        hour: int
    ) -> Optional[Dict]:
        """
        Detect frost on solar panels using FrostDetector

        Args:
            weather_data: Current weather sensor data
            date_str: Date string (YYYY-MM-D)
            hour: Hour (0-23)

        Returns:
            Frost detection result dict or None if detection not possible
        """
        try:

            astronomy_file = self.stats_dir / "astronomy_cache.json"

            if not astronomy_file.exists():
                _LOGGER.debug("Astronomy cache not found - skipping frost detection")
                return None

            astronomy_data = await self.hass.async_add_executor_job(
                self._read_json_sync, astronomy_file
            )

            day_data = astronomy_data.get("days", {}).get(date_str, {})
            hour_data = day_data.get("hourly", {}).get(str(hour), {})

            theoretical_max_wm2 = hour_data.get("clear_sky_solar_radiation_wm2")

            if theoretical_max_wm2 is None:
                _LOGGER.debug(f"No astronomy data for {date_str} hour {hour} - skipping frost detection")
                return None

            cloud_cover = None
            try:
                weather_cache_file = self.data_dir / "data" / "weather_cache.json"
                if weather_cache_file.exists():
                    cache_data = await self.hass.async_add_executor_job(
                        self._read_json_sync, weather_cache_file
                    )
                    forecast_day = cache_data.get("forecasts", {}).get(date_str, {})
                    hourly_forecasts = forecast_day.get("hourly", [])

                    for forecast in hourly_forecasts:
                        if forecast.get("local_hour") == hour:
                            cloud_cover = forecast.get("cloud_cover")
                            break
            except Exception as e:
                _LOGGER.debug(f"Could not load cloud cover from weather cache: {e}")

            frost_result = self.frost_detector.detect_frost(
                temperature_c=weather_data.get("temperature_c"),
                humidity_percent=weather_data.get("humidity_percent"),
                wind_speed_ms=weather_data.get("wind_speed_ms"),
                solar_radiation_wm2=weather_data.get("solar_radiation_wm2"),
                theoretical_max_wm2=theoretical_max_wm2,
                cloud_cover_percent=cloud_cover
            )

            return frost_result

        except Exception as e:
            _LOGGER.error(f"Error during frost detection: {e}", exc_info=True)
            return None

    async def _save_hourly_data(self, date_str: str, hour: int, weather_data: Dict[str, Any]):
        """Save hourly weather data to JSON file @zara"""
        try:

            if self.actual_file.exists():
                data = await self.hass.async_add_executor_job(
                    self._read_json_sync, self.actual_file
                )
            else:
                data = {
                    "version": "1.0",
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat(),
                    },
                    "hourly_data": {},
                }

            data["metadata"]["last_updated"] = datetime.now().isoformat()

            if date_str not in data["hourly_data"]:
                data["hourly_data"][date_str] = {}

            data["hourly_data"][date_str][str(hour)] = weather_data

            await self._cleanup_old_data(data)

            await self.hass.async_add_executor_job(
                self._write_json_sync, self.actual_file, data
            )

            _LOGGER.debug(f"Saved to {self.actual_file}")

        except Exception as e:
            _LOGGER.error(f"Error saving hourly weather data: {e}", exc_info=True)

    def _read_json_sync(self, file_path: Path) -> Dict[str, Any]:
        """Synchronous JSON file read (for executor) @zara"""
        with open(file_path, "r") as f:
            return json.load(f)

    def _write_json_sync(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Synchronous JSON file write (for executor) @zara"""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    async def _cleanup_old_data(self, data: Dict[str, Any]):
        """Remove data older than 7 days @zara"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

            dates_to_remove = [
                date_str
                for date_str in data["hourly_data"].keys()
                if date_str < cutoff_date
            ]

            for date_str in dates_to_remove:
                del data["hourly_data"][date_str]
                _LOGGER.debug(f"Removed old data for {date_str}")

        except Exception as e:
            _LOGGER.warning(f"Error during cleanup: {e}")

    async def get_actual_weather(
        self, date_str: str, hour: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get actual weather data for specific date and hour.

        Args:
            date_str: Date string (YYYY-MM-DD)
            hour: Hour (0-23)

        Returns:
            Weather data dict or None if not available
        """
        try:
            if not self.actual_file.exists():
                return None

            data = await self.hass.async_add_executor_job(
                self._read_json_sync, self.actual_file
            )

            return data.get("hourly_data", {}).get(date_str, {}).get(str(hour))

        except Exception as e:
            _LOGGER.error(f"Error reading actual weather for {date_str} {hour}: {e}")
            return None

    async def get_daily_actual_weather(self, date_str: str) -> Dict[int, Dict[str, Any]]:
        """Get all actual weather data for a specific day @zara"""
        try:
            if not self.actual_file.exists():
                return {}

            data = await self.hass.async_add_executor_job(
                self._read_json_sync, self.actual_file
            )

            daily_data = data.get("hourly_data", {}).get(date_str, {})

            return {int(hour): weather for hour, weather in daily_data.items()}

        except Exception as e:
            _LOGGER.error(f"Error reading daily actual weather for {date_str}: {e}")
            return {}
