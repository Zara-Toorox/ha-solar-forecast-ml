# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any
import json

from .data_frost_detection import FrostDetector
from .data_schemas import get_schema
from ..const import DOMAIN
from ..core.core_helpers import SafeDateTimeUtil as dt_util

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

        # Track last frost notification hour to avoid spam
        self._last_frost_notification_hour: Optional[str] = None

        # Snow tracking state (persisted to file, loaded on init)
        self._last_snow_notification_key: Optional[str] = None
        self._snow_tracking_initialized = False

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

    async def track_current_weather(self) -> Dict[str, Any]:
        """Track current weather conditions from local sensors @zara"""
        now = dt_util.now()
        date_str = now.strftime("%Y-%m-%d")
        hour = now.hour

        weather_data = await self._read_sensors()

        if not weather_data:
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

        frost_result = await self._detect_frost(weather_data, date_str, hour)
        if frost_result:
            weather_data["frost_detected"] = frost_result.get("frost_detected")
            weather_data["frost_score"] = frost_result.get("frost_score")
            weather_data["frost_confidence"] = frost_result.get("confidence")

            if "frost_analysis" in frost_result:
                weather_data["frost_analysis"] = frost_result["frost_analysis"]

            # Send notification for heavy frost
            if frost_result.get("frost_detected") == "heavy_frost":
                await self._send_frost_notification(frost_result, weather_data, date_str, hour)

        astronomy_fields = await self._get_astronomy_time_fields(date_str, hour)
        if astronomy_fields:
            weather_data["hours_after_sunrise"] = astronomy_fields.get("hours_after_sunrise")
            weather_data["hours_before_sunset"] = astronomy_fields.get("hours_before_sunset")

        # Snow tracking - runs 24/7, even at night
        snow_result = await self._check_snow_conditions(weather_data, date_str, hour)
        if snow_result:
            weather_data["snow_covered_panels"] = snow_result.get("snow_covered_panels", False)
            weather_data["snow_coverage_source"] = snow_result.get("source", "unknown")
            if snow_result.get("snow_event_detected"):
                weather_data["snow_event_detected"] = True

        await self._save_hourly_data(date_str, hour, weather_data)

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
                    continue

                value = state.state

                if value in ["unavailable", "unknown", None]:
                    continue

                if field == "condition":
                    data[field] = value
                    available_count += 1
                else:
                    try:
                        numeric_value = float(value)
                        data[field] = round(numeric_value, 2)
                        available_count += 1
                    except (ValueError, TypeError):
                        continue

            except Exception:
                continue

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
            return None

        try:
            astronomy_file = self.stats_dir / "astronomy_cache.json"

            if not astronomy_file.exists():
                return None

            astronomy_data = await self.hass.async_add_executor_job(
                self._read_json_sync, astronomy_file
            )

            day_data = astronomy_data.get("days", {}).get(date_str, {})
            hour_data = day_data.get("hourly", {}).get(str(hour), {})

            theoretical_max_wm2 = hour_data.get("clear_sky_solar_radiation_wm2")

            if theoretical_max_wm2 is None or theoretical_max_wm2 <= 0:
                return None

            calibration_factor = await self._get_solar_radiation_correction_factor()

            calibrated_wm2 = measured_wm2 / calibration_factor if calibration_factor > 0 else measured_wm2
            ratio = calibrated_wm2 / theoretical_max_wm2
            ratio = max(0.0, min(1.0, ratio))
            cloud_cover = (1.0 - ratio) * 100.0

            return round(cloud_cover, 1)

        except Exception:
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

        except Exception:
            return 1.0

    async def _get_astronomy_time_fields(
        self,
        date_str: str,
        hour: int
    ) -> Optional[Dict[str, float]]:
        """
        Calculate hours_after_sunrise and hours_before_sunset from astronomy cache.
        """
        try:
            astronomy_file = self.stats_dir / "astronomy_cache.json"

            if not astronomy_file.exists():
                return None

            astronomy_data = await self.hass.async_add_executor_job(
                self._read_json_sync, astronomy_file
            )

            day_data = astronomy_data.get("days", {}).get(date_str, {})

            if not day_data:
                return None

            sunrise_str = day_data.get("sunrise_local")
            sunset_str = day_data.get("sunset_local")

            if not sunrise_str or not sunset_str:
                return None

            sunrise = datetime.fromisoformat(sunrise_str)
            sunset = datetime.fromisoformat(sunset_str)

            current_dt = datetime.fromisoformat(f"{date_str}T{hour:02d}:00:00")
            # Fix timezone mismatch by using timezone from sunrise
            if sunrise.tzinfo is not None:
                current_dt = current_dt.replace(tzinfo=sunrise.tzinfo)

            hours_after_sunrise = (current_dt - sunrise).total_seconds() / 3600.0
            hours_before_sunset = (sunset - current_dt).total_seconds() / 3600.0

            return {
                "hours_after_sunrise": round(hours_after_sunrise, 2),
                "hours_before_sunset": round(hours_before_sunset, 2),
            }

        except Exception:
            return None

    async def _detect_frost(
        self,
        weather_data: Dict[str, Any],
        date_str: str,
        hour: int
    ) -> Optional[Dict]:
        """Detect frost on solar panels using FrostDetector @zara"""
        try:
            astronomy_file = self.stats_dir / "astronomy_cache.json"

            if not astronomy_file.exists():
                return None

            astronomy_data = await self.hass.async_add_executor_job(
                self._read_json_sync, astronomy_file
            )

            day_data = astronomy_data.get("days", {}).get(date_str, {})
            hour_data = day_data.get("hourly", {}).get(str(hour), {})

            theoretical_max_wm2 = hour_data.get("clear_sky_solar_radiation_wm2")

            if theoretical_max_wm2 is None:
                return None

            cloud_cover = None
            try:
                # Use open_meteo_cache.json (primary source) with correct format
                weather_cache_file = self.data_dir / "data" / "open_meteo_cache.json"
                if weather_cache_file.exists():
                    cache_data = await self.hass.async_add_executor_job(
                        self._read_json_sync, weather_cache_file
                    )
                    # Format: forecast[date][hour] (hour as string)
                    forecast_day = cache_data.get("forecast", {}).get(date_str, {})
                    hour_data_forecast = forecast_day.get(str(hour), {})
                    cloud_cover = hour_data_forecast.get("cloud_cover")
            except Exception:
                pass

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
            _LOGGER.error(f"Error during frost detection: {e}")
            return None

    async def _save_hourly_data(self, date_str: str, hour: int, weather_data: Dict[str, Any]):
        """Save hourly weather data to JSON file @zara"""
        try:
            if self.actual_file.exists():
                data = await self.hass.async_add_executor_job(
                    self._read_json_sync, self.actual_file
                )
            else:
                # Use centralized schema from data_schemas.py (Single Source of Truth)
                data = get_schema("hourly_weather_actual")
                data["metadata"]["created_at"] = dt_util.now().isoformat()

            data["metadata"]["last_updated"] = dt_util.now().isoformat()

            if date_str not in data["hourly_data"]:
                data["hourly_data"][date_str] = {}

            data["hourly_data"][date_str][str(hour)] = weather_data

            await self._cleanup_old_data(data)

            await self.hass.async_add_executor_job(
                self._write_json_sync, self.actual_file, data
            )

        except Exception as e:
            _LOGGER.error(f"Error saving hourly weather data: {e}")

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
            cutoff_date = (dt_util.now() - timedelta(days=7)).strftime("%Y-%m-%d")

            dates_to_remove = [
                date_str
                for date_str in data["hourly_data"].keys()
                if date_str < cutoff_date
            ]

            for date_str in dates_to_remove:
                del data["hourly_data"][date_str]

        except Exception:
            pass

    async def get_actual_weather(
        self, date_str: str, hour: int
    ) -> Optional[Dict[str, Any]]:
        """Get actual weather data for specific date and hour @zara"""
        try:
            if not self.actual_file.exists():
                return None

            data = await self.hass.async_add_executor_job(
                self._read_json_sync, self.actual_file
            )

            return data.get("hourly_data", {}).get(date_str, {}).get(str(hour))

        except Exception:
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

        except Exception:
            return {}

    async def _send_frost_notification(
        self,
        frost_result: Dict[str, Any],
        weather_data: Dict[str, Any],
        date_str: str,
        hour: int
    ) -> None:
        """Send notification when heavy frost is detected @zara"""
        try:
            # Avoid duplicate notifications for same hour
            hour_key = f"{date_str}_{hour}"
            if self._last_frost_notification_hour == hour_key:
                return

            # Get notification service from hass.data
            notification_service = self.hass.data.get(DOMAIN, {}).get("notification_service")
            if not notification_service:
                _LOGGER.debug("Notification service not available for frost warning")
                return

            frost_analysis = frost_result.get("frost_analysis", {})

            await notification_service.show_frost_warning(
                frost_score=frost_result.get("frost_score", 0),
                temperature_c=weather_data.get("temperature_c", 0.0),
                dewpoint_c=frost_analysis.get("dewpoint_c", 0.0),
                frost_margin_c=frost_analysis.get("frost_margin_c", 0.0),
                hour=hour,
                confidence=frost_result.get("confidence", 0.0),
            )

            self._last_frost_notification_hour = hour_key
            _LOGGER.info(f"Frost warning notification sent for {date_str} {hour:02d}:00")

        except Exception as e:
            _LOGGER.error(f"Error sending frost notification: {e}")

    # =========================================================================
    # SNOW TRACKING METHODS
    # =========================================================================

    async def _check_snow_conditions(
        self,
        weather_data: Dict[str, Any],
        date_str: str,
        hour: int
    ) -> Optional[Dict[str, Any]]:
        """Check for snow conditions and panel coverage.

        This method runs 24/7 (not just during production hours) to detect:
        1. Active snowfall (temp < 2°C + precipitation > 0.5mm)
        2. Persistent snow coverage (snow fell recently + temp still < 2°C)
        3. Snow melt (temp > 2°C for 2h OR temp > 5°C for 30min)

        @zara
        """
        try:
            result = {
                "snow_covered_panels": False,
                "snow_event_detected": False,
                "source": "none",
            }

            temp_c = weather_data.get("temperature_c")
            precip_mm = weather_data.get("precipitation_mm", 0.0)

            # Load current snow tracking state
            snow_tracking = await self._load_snow_tracking()

            # Step 1: Check for active snow event
            if temp_c is not None and temp_c < 2.0 and precip_mm > 0.5:
                # Active snowfall detected!
                result["snow_event_detected"] = True
                result["snow_covered_panels"] = True
                result["source"] = "active_snowfall"

                # Update tracking state
                now = dt_util.now()
                snow_tracking["last_snow_event"] = now.isoformat()
                snow_tracking["panels_covered_since"] = now.isoformat()
                snow_tracking["estimated_depth_mm"] = snow_tracking.get("estimated_depth_mm", 0.0) + (precip_mm * 10)
                snow_tracking["melt_started_at"] = None
                snow_tracking["above_threshold_since"] = None

                await self._save_snow_tracking(snow_tracking)
                await self._send_snow_notification(
                    snow_type="snowfall",
                    temp_c=temp_c,
                    precip_mm=precip_mm,
                    date_str=date_str,
                    hour=hour
                )

                _LOGGER.info(
                    f"Snow event detected: {precip_mm:.1f}mm at {temp_c:.1f}°C - "
                    f"Panels likely snow-covered"
                )
                return result

            # Step 2: Check for persistent snow coverage
            last_snow_event = snow_tracking.get("last_snow_event")
            if last_snow_event:
                try:
                    last_snow_dt = dt_util.parse_datetime(last_snow_event) or datetime.fromisoformat(last_snow_event)
                    hours_since_snow = (dt_util.now() - last_snow_dt).total_seconds() / 3600

                    # Snow likely still on panels if:
                    # - Snow fell within last 12 hours AND
                    # - Temperature is still below 2°C
                    if hours_since_snow < 12 and temp_c is not None and temp_c < 2.0:
                        result["snow_covered_panels"] = True
                        result["source"] = "persistence_check"
                        return result

                except (ValueError, TypeError):
                    pass

            # Step 3: Check for snow melt
            panels_covered_since = snow_tracking.get("panels_covered_since")
            if panels_covered_since and temp_c is not None:
                melt_detected = await self._check_snow_melt(
                    temp_c=temp_c,
                    snow_tracking=snow_tracking,
                    date_str=date_str,
                    hour=hour
                )

                if melt_detected:
                    # Reset snow tracking
                    snow_tracking["panels_covered_since"] = None
                    snow_tracking["estimated_depth_mm"] = 0.0
                    snow_tracking["melt_started_at"] = None
                    snow_tracking["above_threshold_since"] = None
                    await self._save_snow_tracking(snow_tracking)

                    result["snow_covered_panels"] = False
                    result["source"] = "melt_detected"

                    await self._send_snow_notification(
                        snow_type="melting",
                        temp_c=temp_c,
                        precip_mm=0,
                        date_str=date_str,
                        hour=hour
                    )

                    _LOGGER.info(f"Snow melt detected at {temp_c:.1f}°C - Panels clearing")
                    return result

            return result

        except Exception as e:
            _LOGGER.error(f"Error checking snow conditions: {e}")
            return None

    async def _check_snow_melt(
        self,
        temp_c: float,
        snow_tracking: Dict[str, Any],
        date_str: str,
        hour: int
    ) -> bool:
        """Check if snow is melting based on temperature thresholds.

        Melt conditions:
        - Temp > 2°C for 2+ hours, OR
        - Temp > 5°C for 30+ minutes (accelerated melt)

        @zara
        """
        now = dt_util.now()

        # Threshold 1: Temp > 5°C = fast melt (30 min)
        if temp_c > 5.0:
            above_threshold = snow_tracking.get("above_threshold_since")
            if above_threshold:
                try:
                    above_dt = dt_util.parse_datetime(above_threshold) or datetime.fromisoformat(above_threshold)
                    minutes_above = (now - above_dt).total_seconds() / 60
                    if minutes_above >= 30:
                        return True
                except (ValueError, TypeError):
                    pass
            else:
                snow_tracking["above_threshold_since"] = now.isoformat()
                await self._save_snow_tracking(snow_tracking)
            return False

        # Threshold 2: Temp > 2°C = normal melt (2 hours)
        if temp_c > 2.0:
            melt_started = snow_tracking.get("melt_started_at")
            if melt_started:
                try:
                    melt_dt = dt_util.parse_datetime(melt_started) or datetime.fromisoformat(melt_started)
                    hours_melting = (now - melt_dt).total_seconds() / 3600
                    if hours_melting >= 2:
                        return True
                except (ValueError, TypeError):
                    pass
            else:
                snow_tracking["melt_started_at"] = now.isoformat()
                await self._save_snow_tracking(snow_tracking)
            return False

        # Temp still < 2°C - reset melt timers
        if snow_tracking.get("melt_started_at") or snow_tracking.get("above_threshold_since"):
            snow_tracking["melt_started_at"] = None
            snow_tracking["above_threshold_since"] = None
            await self._save_snow_tracking(snow_tracking)

        return False

    async def _load_snow_tracking(self) -> Dict[str, Any]:
        """Load snow tracking state from file @zara"""
        try:
            if not self.actual_file.exists():
                return self._get_default_snow_tracking()

            data = await self.hass.async_add_executor_job(
                self._read_json_sync, self.actual_file
            )

            return data.get("snow_tracking", self._get_default_snow_tracking())

        except Exception as e:
            _LOGGER.debug(f"Error loading snow tracking: {e}")
            return self._get_default_snow_tracking()

    async def _save_snow_tracking(self, snow_tracking: Dict[str, Any]) -> None:
        """Save snow tracking state to file @zara"""
        try:
            if self.actual_file.exists():
                data = await self.hass.async_add_executor_job(
                    self._read_json_sync, self.actual_file
                )
            else:
                data = get_schema("hourly_weather_actual")
                data["metadata"]["created_at"] = dt_util.now().isoformat()

            data["snow_tracking"] = snow_tracking

            await self.hass.async_add_executor_job(
                self._write_json_sync, self.actual_file, data
            )

        except Exception as e:
            _LOGGER.error(f"Error saving snow tracking: {e}")

    def _get_default_snow_tracking(self) -> Dict[str, Any]:
        """Get default snow tracking structure @zara"""
        return {
            "last_snow_event": None,
            "estimated_depth_mm": 0.0,
            "panels_covered_since": None,
            "melt_started_at": None,
            "above_threshold_since": None,
        }

    async def _send_snow_notification(
        self,
        snow_type: str,
        temp_c: float,
        precip_mm: float,
        date_str: str,
        hour: int
    ) -> None:
        """Send notification for snow events @zara

        Args:
            snow_type: "snowfall" or "melting"
            temp_c: Current temperature
            precip_mm: Precipitation amount (for snowfall)
            date_str: Date string
            hour: Hour
        """
        try:
            # Avoid duplicate notifications for same event type per day
            notification_key = f"{date_str}_{snow_type}"
            if self._last_snow_notification_key == notification_key:
                return

            notification_service = self.hass.data.get(DOMAIN, {}).get("notification_service")
            if not notification_service:
                _LOGGER.debug("Notification service not available for snow warning")
                return

            if snow_type == "snowfall":
                await notification_service.show_snow_covered_warning(
                    temperature_c=temp_c,
                    precipitation_mm=precip_mm,
                    hour=hour,
                )
            elif snow_type == "melting":
                await notification_service.show_snow_melting_info(
                    temperature_c=temp_c,
                    hour=hour,
                )

            self._last_snow_notification_key = notification_key
            _LOGGER.info(f"Snow {snow_type} notification sent for {date_str} {hour:02d}:00")

        except Exception as e:
            _LOGGER.error(f"Error sending snow notification: {e}")

    async def detect_snow_from_production_anomaly(
        self,
        hour: int,
        weather_forecast: Dict[str, Any],
        actual_production: float,
    ) -> bool:
        """Detect snow when no rain sensor is available.

        For users WITHOUT rain sensors, we can detect snow by comparing:
        - API predicted precipitation > 0.5mm
        - Temperature < 2°C
        - Expected solar radiation > 100 W/m²
        - Actual production = 0 or near 0
        → Likely snow on panels

        This is called from data_hourly_predictions.py during actual updates.

        @zara
        """
        try:
            temp = weather_forecast.get("temperature", 10)
            precip = weather_forecast.get("precipitation_mm", 0)
            radiation = weather_forecast.get("solar_radiation_wm2", 0)

            if temp >= 2.0:
                return False
            if precip < 0.5:
                return False
            if radiation < 100:
                return False  # Night or very cloudy - can't tell
            if actual_production > 0.05:
                return False  # Some production = no snow cover

            _LOGGER.debug(
                f"Snow detected from production anomaly: "
                f"Temp={temp:.1f}°C, Precip={precip:.1f}mm, "
                f"Radiation={radiation:.0f}W/m², Production={actual_production:.3f}kWh"
            )
            return True

        except Exception as e:
            _LOGGER.debug(f"Error in snow detection from production anomaly: {e}")
            return False

    async def get_snow_tracking_status(self) -> Dict[str, Any]:
        """Get current snow tracking status for external use @zara"""
        snow_tracking = await self._load_snow_tracking()

        panels_covered = False
        if snow_tracking.get("panels_covered_since"):
            panels_covered = True

        return {
            "panels_covered": panels_covered,
            "last_snow_event": snow_tracking.get("last_snow_event"),
            "estimated_depth_mm": snow_tracking.get("estimated_depth_mm", 0.0),
            "panels_covered_since": snow_tracking.get("panels_covered_since"),
            "melt_in_progress": snow_tracking.get("melt_started_at") is not None,
        }
