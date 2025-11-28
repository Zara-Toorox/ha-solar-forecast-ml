"""Sensor-Based Cloud Correction - Morning Check at 9:00 V10.0.0 @zara

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
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from homeassistant.core import HomeAssistant

from .data_io import DataManagerIO

_LOGGER = logging.getLogger(__name__)

CLOUD_DISCREPANCY_THRESHOLD = 25

MIN_SUN_ELEVATION = 10

class SensorCloudCorrection(DataManagerIO):
    """Corrects cloud forecasts based on local sensor readings.

    ONLY active if user has configured a solar_radiation_sensor or lux_sensor.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        data_manager,
        solar_radiation_sensor: Optional[str] = None,
        lux_sensor: Optional[str] = None,
    ):
        """Initialize sensor-based cloud correction.

        Args:
            hass: Home Assistant instance
            data_manager: DataManager instance
            solar_radiation_sensor: Entity ID of W/m² sensor (optional)
            lux_sensor: Entity ID of Lux sensor (optional)
        """
        super().__init__(hass, data_manager.data_dir)

        self.data_manager = data_manager
        self.solar_radiation_sensor = solar_radiation_sensor
        self.lux_sensor = lux_sensor

        self.is_enabled = bool(solar_radiation_sensor or lux_sensor)

        if self.is_enabled:
            sensor_info = []
            if solar_radiation_sensor:
                sensor_info.append(f"W/m²={solar_radiation_sensor}")
            if lux_sensor:
                sensor_info.append(f"Lux={lux_sensor}")
            _LOGGER.info(
                f"SensorCloudCorrection enabled ({', '.join(sensor_info)}). "
                f"Morning check at 9:00 will validate cloud forecasts."
            )
        else:
            _LOGGER.debug(
                "SensorCloudCorrection disabled - no solar_radiation or lux sensor configured"
            )

    async def should_run_morning_check(self, current_time: datetime) -> bool:
        """Check if morning correction should run @zara"""
        if not self.is_enabled:
            return False

        if current_time.hour != 9 or current_time.minute > 10:
            return False

        return True

    async def run_morning_check(self, current_time: datetime) -> Dict[str, Any]:
        """Run the 9:00 morning cloud correction check @zara"""
        result = {
            "executed": False,
            "correction_applied": False,
            "reason": None,
            "sensor_cloud_percent": None,
            "forecast_cloud_percent": None,
            "discrepancy": None,
        }

        if not self.is_enabled:
            result["reason"] = "No sensor configured"
            return result

        try:
            _LOGGER.info("🌅 Running 9:00 sensor-based cloud correction check...")

            sensor_cloud = await self._get_sensor_cloud_estimate(current_time)
            if sensor_cloud is None:
                result["reason"] = "Could not calculate cloud cover from sensor"
                _LOGGER.warning(result["reason"])
                return result

            result["sensor_cloud_percent"] = sensor_cloud

            forecast_cloud = await self._get_forecast_cloud_average(current_time)
            if forecast_cloud is None:
                result["reason"] = "Could not get forecast cloud cover"
                _LOGGER.warning(result["reason"])
                return result

            result["forecast_cloud_percent"] = forecast_cloud

            discrepancy = forecast_cloud - sensor_cloud
            result["discrepancy"] = discrepancy
            result["executed"] = True

            _LOGGER.info(
                f"Morning check: Sensor={sensor_cloud:.0f}%, "
                f"Forecast={forecast_cloud:.0f}%, "
                f"Discrepancy={discrepancy:+.0f}%"
            )

            if discrepancy > CLOUD_DISCREPANCY_THRESHOLD:

                _LOGGER.info(
                    f"☀️ Forecast overestimated clouds by {discrepancy:.0f}%! "
                    f"Triggering forecast recalculation..."
                )

                correction_factor = sensor_cloud / forecast_cloud if forecast_cloud > 0 else 1.0
                await self._apply_sensor_correction(current_time, correction_factor, sensor_cloud)

                result["correction_applied"] = True
                result["reason"] = f"Forecast overestimated clouds by {discrepancy:.0f}%"

            elif discrepancy < -CLOUD_DISCREPANCY_THRESHOLD:

                _LOGGER.info(
                    f"☁️ Forecast underestimated clouds by {abs(discrepancy):.0f}%! "
                    f"Triggering forecast recalculation..."
                )

                correction_factor = sensor_cloud / forecast_cloud if forecast_cloud > 0 else 1.0
                await self._apply_sensor_correction(current_time, correction_factor, sensor_cloud)

                result["correction_applied"] = True
                result["reason"] = f"Forecast underestimated clouds by {abs(discrepancy):.0f}%"

            else:
                result["reason"] = f"Discrepancy {discrepancy:+.0f}% within tolerance (±{CLOUD_DISCREPANCY_THRESHOLD}%)"
                _LOGGER.info(f"✓ Forecast accuracy acceptable: {result['reason']}")

            return result

        except Exception as e:
            _LOGGER.error(f"Error in morning cloud check: {e}", exc_info=True)
            result["reason"] = f"Error: {str(e)}"
            return result

    async def _get_sensor_cloud_estimate(self, current_time: datetime) -> Optional[float]:
        """Calculate cloud cover percentage from sensor readings @zara"""
        try:

            actual_file = self.data_manager.data_dir / "stats" / "hourly_weather_actual.json"
            if not actual_file.exists():
                _LOGGER.warning("hourly_weather_actual.json not found")
                return None

            actual_data = await self._read_json_file(actual_file, None)
            if not actual_data:
                return None

            today = current_time.date().isoformat()
            today_data = actual_data.get("hourly_data", {}).get(today, {})

            cloud_values = []
            for hour in ["8", "9"]:
                hour_data = today_data.get(hour, {})

                cloud = hour_data.get("cloud_cover_percent")
                if cloud is not None:
                    cloud_values.append(cloud)
                    _LOGGER.debug(f"Hour {hour}: cloud_cover_percent = {cloud:.1f}%")

            if not cloud_values:
                _LOGGER.warning("No cloud cover data found for 8:00-9:00")
                return None

            avg_cloud = sum(cloud_values) / len(cloud_values)
            _LOGGER.debug(f"Sensor-based cloud estimate: {avg_cloud:.1f}% (from {len(cloud_values)} readings)")

            return avg_cloud

        except Exception as e:
            _LOGGER.error(f"Error getting sensor cloud estimate: {e}")
            return None

    async def _get_forecast_cloud_average(self, current_time: datetime) -> Optional[float]:
        """Get forecasted cloud cover for today (8:00-16:00 average) @zara"""
        try:
            corrected_file = self.data_manager.weather_corrected_file
            if not corrected_file.exists():
                _LOGGER.warning("weather_forecast_corrected.json not found")
                return None

            corrected_data = await self._read_json_file(corrected_file, None)
            if not corrected_data:
                return None

            today = current_time.date().isoformat()
            today_forecast = corrected_data.get("forecast", {}).get(today, {})

            if not today_forecast:
                _LOGGER.warning(f"No forecast data for {today}")
                return None

            cloud_values = []
            for hour in range(8, 17):
                hour_data = today_forecast.get(str(hour), {})
                clouds = hour_data.get("clouds")
                if clouds is not None:
                    cloud_values.append(clouds)

            if not cloud_values:
                _LOGGER.warning("No cloud forecast values found")
                return None

            avg_cloud = sum(cloud_values) / len(cloud_values)
            _LOGGER.debug(f"Forecast cloud average: {avg_cloud:.1f}% ({len(cloud_values)} hours)")

            return avg_cloud

        except Exception as e:
            _LOGGER.error(f"Error getting forecast cloud average: {e}")
            return None

    async def _apply_sensor_correction(
        self,
        current_time: datetime,
        correction_factor: float,
        sensor_cloud: float
    ) -> bool:
        """Apply sensor-based correction to today's forecast.

        Updates the remaining hours of today with corrected cloud values.

        Args:
            current_time: Current datetime
            correction_factor: Factor to apply (sensor_cloud / forecast_cloud)
            sensor_cloud: The sensor-measured cloud cover

        Returns:
            True if correction was applied
        """
        try:
            corrected_file = self.data_manager.weather_corrected_file
            if not corrected_file.exists():
                return False

            corrected_data = await self._read_json_file(corrected_file, None)
            if not corrected_data:
                return False

            today = current_time.date().isoformat()
            today_forecast = corrected_data.get("forecast", {}).get(today, {})

            if not today_forecast:
                return False

            current_hour = current_time.hour
            corrections_made = 0

            for hour in range(current_hour, 17):
                hour_str = str(hour)
                if hour_str in today_forecast:
                    old_clouds = today_forecast[hour_str].get("clouds")
                    if old_clouds is not None:

                        new_clouds = (sensor_cloud * 0.7) + (old_clouds * 0.3)
                        new_clouds = max(0, min(100, new_clouds))

                        today_forecast[hour_str]["clouds"] = round(new_clouds, 1)
                        today_forecast[hour_str]["sensor_corrected"] = True
                        corrections_made += 1

                        _LOGGER.debug(
                            f"Hour {hour}: clouds {old_clouds:.0f}% → {new_clouds:.0f}%"
                        )

            if "metadata" not in corrected_data:
                corrected_data["metadata"] = {}

            corrected_data["metadata"]["sensor_correction"] = {
                "applied_at": current_time.isoformat(),
                "sensor_cloud_percent": round(sensor_cloud, 1),
                "correction_factor": round(correction_factor, 3),
                "hours_corrected": corrections_made,
            }

            await self._atomic_write_json(corrected_file, corrected_data)

            _LOGGER.info(
                f"✅ Applied sensor correction to {corrections_made} hours "
                f"(clouds adjusted towards {sensor_cloud:.0f}%)"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Error applying sensor correction: {e}", exc_info=True)
            return False
