"""Sensor-Based Cloud Correction - Morning & Midday Checks V12.0.0 @zara

Two-stage intraday forecast correction:
1. 09:00 Morning Check - Primary correction (saves 70% of day)
2. 12:00 Midday Check - Secondary correction (only if needed)

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

# Thresholds for triggering corrections
CLOUD_DISCREPANCY_THRESHOLD = 25  # Cloud cover difference in %
PRODUCTION_DEVIATION_THRESHOLD = 25  # Production IST vs forecast in %
FORECAST_DRIFT_THRESHOLD = 20  # Weather forecast change in % points (Snapshot vs Corrected)

MIN_SUN_ELEVATION = 10

class SensorCloudCorrection(DataManagerIO):
    """Corrects cloud forecasts based on local sensor readings and production data.

    Two-stage correction system:
    1. 09:00 Morning Check - Compare sensor clouds with forecast, trigger full recalc if >25% off
    2. 12:00 Midday Check - Compare morning production with forecast, check for weather changes

    ALWAYS active (uses production data), enhanced if solar_radiation_sensor or lux_sensor configured.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        data_manager,
        solar_radiation_sensor: Optional[str] = None,
        lux_sensor: Optional[str] = None,
        solar_yield_entity: Optional[str] = None,
    ):
        """Initialize sensor-based cloud correction.

        Args:
            hass: Home Assistant instance
            data_manager: DataManager instance
            solar_radiation_sensor: Entity ID of W/m² sensor (optional)
            lux_sensor: Entity ID of Lux sensor (optional)
            solar_yield_entity: Entity ID of daily yield sensor (for production comparison)
        """
        super().__init__(hass, data_manager.data_dir)

        self.data_manager = data_manager
        self.solar_radiation_sensor = solar_radiation_sensor
        self.lux_sensor = lux_sensor
        self.solar_yield_entity = solar_yield_entity

        # Cloud correction needs sensors, but midday correction works with production data
        self.cloud_sensors_enabled = bool(solar_radiation_sensor or lux_sensor)
        self.is_enabled = True  # Always enabled for production-based midday check

        # Track if morning correction was applied (to avoid double-correction at midday)
        self._morning_correction_applied: bool = False
        self._morning_correction_factor: Optional[float] = None

        # Store original morning forecast for comparison
        self._morning_forecast_clouds: Optional[Dict[str, float]] = None

        if self.cloud_sensors_enabled:
            sensor_info = []
            if solar_radiation_sensor:
                sensor_info.append(f"W/m²={solar_radiation_sensor}")
            if lux_sensor:
                sensor_info.append(f"Lux={lux_sensor}")
            _LOGGER.info(
                f"SensorCloudCorrection enabled ({', '.join(sensor_info)}). "
                f"Morning check at 9:00, Midday check at 12:00."
            )
        else:
            _LOGGER.info(
                "SensorCloudCorrection: No cloud sensors, using production-based midday check only."
            )

    async def should_run_morning_check(self, current_time: datetime) -> bool:
        """Check if morning correction should run @zara"""
        if not self.cloud_sensors_enabled:
            return False

        if current_time.hour != 9 or current_time.minute > 10:
            return False

        return True

    async def should_run_midday_check(self, current_time: datetime) -> bool:
        """Check if midday correction should run (12:00-12:15) @zara"""
        if current_time.hour != 12 or current_time.minute > 15:
            return False

        # Skip if major morning correction was already applied
        if self._morning_correction_applied and self._morning_correction_factor:
            if abs(self._morning_correction_factor - 1.0) > 0.3:
                _LOGGER.debug(
                    f"Midday check skipped - major morning correction already applied "
                    f"(factor={self._morning_correction_factor:.2f})"
                )
                return False

        return True

    async def run_morning_check(self, current_time: datetime) -> Dict[str, Any]:
        """Run the 9:00 morning cloud correction check @zara

        Returns dict with:
            - executed: Whether check ran
            - correction_applied: Whether cloud values were adjusted
            - needs_forecast_recalc: Whether full forecast recalculation is needed
            - correction_factor: Factor to apply to remaining forecast
        """
        result = {
            "executed": False,
            "correction_applied": False,
            "needs_forecast_recalc": False,
            "correction_factor": 1.0,
            "reason": None,
            "sensor_cloud_percent": None,
            "forecast_cloud_percent": None,
            "discrepancy": None,
        }

        # Reset morning correction state
        self._morning_correction_applied = False
        self._morning_correction_factor = None

        if not self.cloud_sensors_enabled:
            result["reason"] = "No cloud sensor configured"
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
                # Forecast overestimated clouds → actual sky is clearer → more production expected
                _LOGGER.info(
                    f"☀️ Forecast overestimated clouds by {discrepancy:.0f}%! "
                    f"Triggering forecast recalculation..."
                )

                # Calculate production correction factor
                # If forecast said 80% clouds but actual is 40%, production should be ~1.5-2x higher
                cloud_ratio = (100 - sensor_cloud) / (100 - forecast_cloud) if forecast_cloud < 100 else 1.5
                production_factor = min(2.0, max(1.0, cloud_ratio))

                await self._apply_sensor_correction(current_time, cloud_ratio, sensor_cloud)

                result["correction_applied"] = True
                result["needs_forecast_recalc"] = True
                result["correction_factor"] = production_factor
                result["reason"] = f"Forecast overestimated clouds by {discrepancy:.0f}%"

                # Track for midday check
                self._morning_correction_applied = True
                self._morning_correction_factor = production_factor

            elif discrepancy < -CLOUD_DISCREPANCY_THRESHOLD:
                # Forecast underestimated clouds → actual sky is cloudier → less production expected
                _LOGGER.info(
                    f"☁️ Forecast underestimated clouds by {abs(discrepancy):.0f}%! "
                    f"Triggering forecast recalculation..."
                )

                # Calculate production correction factor
                cloud_ratio = (100 - sensor_cloud) / (100 - forecast_cloud) if forecast_cloud < 100 else 0.5
                production_factor = max(0.3, min(1.0, cloud_ratio))

                await self._apply_sensor_correction(current_time, cloud_ratio, sensor_cloud)

                result["correction_applied"] = True
                result["needs_forecast_recalc"] = True
                result["correction_factor"] = production_factor
                result["reason"] = f"Forecast underestimated clouds by {abs(discrepancy):.0f}%"

                # Track for midday check
                self._morning_correction_applied = True
                self._morning_correction_factor = production_factor

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

    async def run_midday_check(self, current_time: datetime) -> Dict[str, Any]:
        """Run the 12:05 midday correction check @zara"""
        result = {
            "executed": False,
            "correction_needed": False,
            "needs_forecast_recalc": False,
            "correction_factor": 1.0,
            "new_display_value": None,
            "reason": None,
            # Morning production data
            "morning_production_kwh": None,
            "morning_forecast_kwh": None,
            "morning_deviation_kwh": None,
            # Forecast drift data
            "forecast_drift_detected": False,
            "forecast_drift_percent": None,
            "snapshot_afternoon_clouds": None,
            "corrected_afternoon_clouds": None,
            # Original forecast data
            "original_daily_forecast": None,
            "original_afternoon_forecast": None,
            "new_afternoon_forecast": None,
        }

        try:
            _LOGGER.info("📊 Running 12:05 midday correction check (V3.0 - Forecast Drift Detection)...")

            # === STEP 1: Get current yield from sensor ===
            current_yield = await self._get_current_yield_from_sensor()
            if current_yield is None:
                current_yield = 0.0
                _LOGGER.warning("Could not get current yield from sensor, using 0.0")

            # === STEP 2: Get morning data (6-12 Uhr) ===
            morning_production = await self._get_morning_production(current_time)
            morning_forecast = await self._get_morning_forecast(current_time)

            if morning_forecast is None or morning_forecast < 0.001:
                morning_forecast = 0.001  # Avoid division by zero

            if morning_production is None:
                morning_production = current_yield  # Use sensor value as fallback

            result["morning_production_kwh"] = round(morning_production, 3)
            result["morning_forecast_kwh"] = round(morning_forecast, 3)

            # Morning deviation (can be positive or negative)
            morning_deviation = morning_production - morning_forecast
            result["morning_deviation_kwh"] = round(morning_deviation, 3)

            deviation_percent = (morning_deviation / morning_forecast) * 100 if morning_forecast > 0 else 0

            _LOGGER.info(
                f"Morning (6-12h): IST={morning_production:.3f} kWh, "
                f"Forecast={morning_forecast:.3f} kWh, Deviation={morning_deviation:+.3f} kWh ({deviation_percent:+.1f}%)"
            )

            # === STEP 3: Get original daily forecast ===
            original_daily_forecast = await self._get_original_daily_forecast()
            original_afternoon_forecast = await self._get_afternoon_forecast_sum(current_time)

            result["original_daily_forecast"] = original_daily_forecast
            result["original_afternoon_forecast"] = original_afternoon_forecast

            _LOGGER.info(
                f"Original forecast: Daily={original_daily_forecast:.3f} kWh, "
                f"Afternoon (12-sunset)={original_afternoon_forecast:.3f} kWh"
            )

            # === STEP 4: Check for FORECAST DRIFT ===
            forecast_drift = await self._check_forecast_drift(current_time)

            result["forecast_drift_percent"] = forecast_drift.get("drift_percent")
            result["snapshot_afternoon_clouds"] = forecast_drift.get("snapshot_clouds")
            result["corrected_afternoon_clouds"] = forecast_drift.get("corrected_clouds")

            drift_detected = (
                forecast_drift.get("drift_percent") is not None
                and abs(forecast_drift.get("drift_percent", 0)) > FORECAST_DRIFT_THRESHOLD
            )
            result["forecast_drift_detected"] = drift_detected

            if drift_detected:
                _LOGGER.info(
                    f"⚠️ FORECAST DRIFT DETECTED: {forecast_drift.get('drift_percent'):+.1f}% "
                    f"(Snapshot={forecast_drift.get('snapshot_clouds'):.0f}%, "
                    f"Corrected={forecast_drift.get('corrected_clouds'):.0f}%)"
                )
            else:
                drift_pct = forecast_drift.get('drift_percent')
                drift_str = f"{drift_pct:.1f}%" if drift_pct is not None else "N/A"
                _LOGGER.info(
                    f"✓ No significant forecast drift "
                    f"(Drift={drift_str}, threshold=±{FORECAST_DRIFT_THRESHOLD}%)"
                )

            # === STEP 5: Calculate new display value ===
            result["executed"] = True

            if drift_detected:
                # Scenario B: Forecast changed significantly
                # → Recalculate afternoon with NEW weather data
                new_afternoon = await self._calculate_new_afternoon_forecast(current_time)
                result["new_afternoon_forecast"] = new_afternoon

                if new_afternoon is not None:
                    # Sensor = Current yield + NEW afternoon forecast
                    new_display = current_yield + new_afternoon
                    result["new_display_value"] = round(new_display, 3)
                    result["correction_needed"] = True
                    result["needs_forecast_recalc"] = True
                    result["reason"] = (
                        f"Forecast drift {forecast_drift.get('drift_percent'):+.1f}% → "
                        f"Recalculated: {current_yield:.2f} + {new_afternoon:.2f} = {new_display:.2f} kWh"
                    )

                    _LOGGER.info(
                        f"📈 Scenario B (Forecast Drift): New display = {new_display:.3f} kWh "
                        f"(Current={current_yield:.3f} + NewAfternoon={new_afternoon:.3f})"
                    )
                else:
                    # Fallback: Use base calculation
                    base_value = original_daily_forecast + morning_deviation if original_daily_forecast else current_yield
                    result["new_display_value"] = round(base_value, 3)
                    result["correction_needed"] = True
                    result["reason"] = f"Forecast drift detected, but recalc failed. Using base: {base_value:.2f} kWh"

            else:
                # Scenario A: No significant forecast drift
                # → Sensor = IST + Original afternoon forecast
                if original_afternoon_forecast is not None and original_afternoon_forecast > 0:
                    # Base = Current yield + Original afternoon forecast
                    base_value = current_yield + original_afternoon_forecast
                    result["new_display_value"] = round(base_value, 3)

                    # Only trigger correction if there's significant morning deviation
                    if abs(deviation_percent) > PRODUCTION_DEVIATION_THRESHOLD:
                        result["correction_needed"] = True
                        result["needs_forecast_recalc"] = False  # Don't recalc, just update display
                        result["reason"] = (
                            f"Morning deviation {deviation_percent:+.1f}% → "
                            f"Base: {current_yield:.2f} + {original_afternoon_forecast:.2f} = {base_value:.2f} kWh"
                        )
                        _LOGGER.info(
                            f"📈 Scenario A (Base Correction): New display = {base_value:.3f} kWh "
                            f"(Current={current_yield:.3f} + OrigAfternoon={original_afternoon_forecast:.3f})"
                        )
                    else:
                        result["reason"] = (
                            f"Morning deviation within tolerance ({deviation_percent:+.1f}%), no correction needed"
                        )
                        _LOGGER.info(f"✓ {result['reason']}")
                else:
                    result["reason"] = "No afternoon forecast data available"
                    _LOGGER.warning(result["reason"])

            # === STEP 6: Log forecast drift for future analysis ===
            await self._log_forecast_drift(
                current_time=current_time,
                morning_deviation=morning_deviation,
                drift_percent=forecast_drift.get("drift_percent"),
                correction_applied=result["correction_needed"],
            )

            return result

        except Exception as e:
            _LOGGER.error(f"Error in midday check V3.0: {e}", exc_info=True)
            result["reason"] = f"Error: {str(e)}"
            return result

    async def _get_current_yield_from_sensor(self) -> Optional[float]:
        """Get current daily yield from sensor entity @zara"""
        try:
            if not self.solar_yield_entity:
                return None

            state = self.hass.states.get(self.solar_yield_entity)
            if state and state.state not in (None, "unknown", "unavailable"):
                return float(state.state)
            return None
        except (ValueError, TypeError):
            return None

    async def _get_original_daily_forecast(self) -> Optional[float]:
        """Get original daily forecast from daily_forecasts.json @zara"""
        try:
            forecasts_file = self.data_manager.data_dir / "stats" / "daily_forecasts.json"
            if not forecasts_file.exists():
                return None

            data = await self._read_json_file(forecasts_file, None)
            if not data:
                return None

            forecast_day = data.get("today", {}).get("forecast_day", {})
            # Use prediction_kwh (locked value), not prediction_kwh_display
            return forecast_day.get("prediction_kwh")

        except Exception as e:
            _LOGGER.error(f"Error getting original daily forecast: {e}")
            return None

    async def _get_afternoon_forecast_sum(self, current_time: datetime) -> Optional[float]:
        """Get sum of hourly forecasts for afternoon hours (12-sunset) @zara"""
        try:
            today = current_time.date().isoformat()

            predictions_file = self.data_manager.data_dir / "stats" / "hourly_predictions.json"
            if not predictions_file.exists():
                return None

            data = await self._read_json_file(predictions_file, None)
            if not data:
                return None

            total = 0.0
            hours_found = 0

            for pred in data.get("predictions", []):
                if pred.get("target_date") != today:
                    continue
                hour = pred.get("target_hour")
                if hour is None or hour < 12:
                    continue
                pred_kwh = pred.get("prediction_kwh")
                if pred_kwh is not None:
                    total += pred_kwh
                    hours_found += 1

            if hours_found == 0:
                return None

            _LOGGER.debug(f"Afternoon forecast sum (12+): {total:.3f} kWh from {hours_found} hours")
            return total

        except Exception as e:
            _LOGGER.error(f"Error getting afternoon forecast sum: {e}")
            return None

    async def _check_forecast_drift(self, current_time: datetime) -> Dict[str, Any]:
        """Compare Snapshot (11:45) with Corrected (04:15) afternoon clouds @zara

        Returns:
            dict with:
                - drift_percent: Cloud difference (positive = less clouds in snapshot)
                - snapshot_clouds: Average afternoon clouds from snapshot
                - corrected_clouds: Average afternoon clouds from corrected
        """
        result = {
            "drift_percent": None,
            "snapshot_clouds": None,
            "corrected_clouds": None,
        }

        try:
            today = current_time.date().isoformat()

            # === Read Snapshot (11:45) ===
            snapshot_file = self.data_manager.data_dir / "stats" / "weather_forecast_snapshot.json"
            if not snapshot_file.exists():
                _LOGGER.warning("Snapshot file not found - cannot detect forecast drift")
                return result

            snapshot_data = await self._read_json_file(snapshot_file, None)
            if not snapshot_data or not snapshot_data.get("forecast"):
                _LOGGER.warning("Snapshot has no forecast data")
                return result

            # === Read Corrected (04:15) ===
            corrected_file = self.data_manager.weather_corrected_file
            if not corrected_file.exists():
                _LOGGER.warning("Corrected file not found")
                return result

            corrected_data = await self._read_json_file(corrected_file, None)
            if not corrected_data or not corrected_data.get("forecast"):
                _LOGGER.warning("Corrected has no forecast data")
                return result

            # === Get afternoon clouds (12-17h) from both ===
            snapshot_forecast = snapshot_data.get("forecast", {}).get(today, {})
            corrected_forecast = corrected_data.get("forecast", {}).get(today, {})

            snapshot_clouds = []
            corrected_clouds = []

            for hour in range(12, 18):
                hour_str = str(hour)

                snap_hour = snapshot_forecast.get(hour_str, {})
                corr_hour = corrected_forecast.get(hour_str, {})

                snap_cloud = snap_hour.get("clouds")
                corr_cloud = corr_hour.get("clouds")

                if snap_cloud is not None:
                    snapshot_clouds.append(snap_cloud)
                if corr_cloud is not None:
                    corrected_clouds.append(corr_cloud)

            if not snapshot_clouds or not corrected_clouds:
                _LOGGER.warning("Insufficient cloud data for drift comparison")
                return result

            # Calculate averages
            snapshot_avg = sum(snapshot_clouds) / len(snapshot_clouds)
            corrected_avg = sum(corrected_clouds) / len(corrected_clouds)

            # Drift = Corrected - Snapshot (positive means corrected has MORE clouds)
            # So if drift is negative, the new forecast (snapshot) has MORE clouds
            drift_percent = corrected_avg - snapshot_avg

            result["snapshot_clouds"] = round(snapshot_avg, 1)
            result["corrected_clouds"] = round(corrected_avg, 1)
            result["drift_percent"] = round(drift_percent, 1)

            _LOGGER.debug(
                f"Forecast drift check: Corrected={corrected_avg:.1f}%, "
                f"Snapshot={snapshot_avg:.1f}%, Drift={drift_percent:+.1f}%"
            )

            return result

        except Exception as e:
            _LOGGER.error(f"Error checking forecast drift: {e}")
            return result

    async def _calculate_new_afternoon_forecast(self, current_time: datetime) -> Optional[float]:
        """Calculate new afternoon forecast based on fresh weather data @zara

        Uses the snapshot (11:45) weather data to recalculate afternoon production.
        This is called when significant forecast drift is detected.
        """
        try:
            # For now, we use a simple proportional adjustment based on cloud change
            # In a full implementation, this would trigger a full physics recalculation

            drift_data = await self._check_forecast_drift(current_time)
            original_afternoon = await self._get_afternoon_forecast_sum(current_time)

            if original_afternoon is None:
                return None

            drift_percent = drift_data.get("drift_percent", 0)

            # Simple adjustment: +10% clouds = -15% production (roughly)
            # drift_percent is Corrected - Snapshot
            # If positive: Corrected has more clouds, so Snapshot is clearer → more production
            # If negative: Snapshot has more clouds → less production

            cloud_production_factor = 1.5  # 1% cloud change = 1.5% production change
            production_adjustment = (drift_percent / 100) * cloud_production_factor

            # Apply adjustment
            new_afternoon = original_afternoon * (1 + production_adjustment)
            new_afternoon = max(0, new_afternoon)  # Cannot be negative

            _LOGGER.info(
                f"New afternoon forecast: Original={original_afternoon:.3f} kWh, "
                f"Adjustment={production_adjustment:+.1%}, New={new_afternoon:.3f} kWh"
            )

            return round(new_afternoon, 3)

        except Exception as e:
            _LOGGER.error(f"Error calculating new afternoon forecast: {e}")
            return None

    async def _log_forecast_drift(
        self,
        current_time: datetime,
        morning_deviation: float,
        drift_percent: Optional[float],
        correction_applied: bool,
    ) -> None:
        """Log forecast drift data for future analysis @zara"""
        try:
            log_file = self.data_manager.data_dir / "stats" / "forecast_drift_log.json"

            # Read existing log
            existing_data = await self._read_json_file(log_file, None)
            if existing_data is None:
                existing_data = {"version": "1.0", "entries": []}

            # Add new entry
            entry = {
                "date": current_time.date().isoformat(),
                "time": current_time.strftime("%H:%M"),
                "morning_deviation_kwh": round(morning_deviation, 3),
                "forecast_drift_percent": round(drift_percent, 1) if drift_percent is not None else None,
                "correction_applied": correction_applied,
            }

            existing_data["entries"].append(entry)

            # Keep only last 30 days
            if len(existing_data["entries"]) > 30:
                existing_data["entries"] = existing_data["entries"][-30:]

            await self._atomic_write_json(log_file, existing_data)

        except Exception as e:
            _LOGGER.debug(f"Could not log forecast drift: {e}")

    async def _get_morning_actual_clouds(self, current_time: datetime) -> Optional[float]:
        """Get actual cloud cover from morning hours (sensor data) @zara"""
        try:
            today = current_time.date().isoformat()

            actual_file = self.data_manager.data_dir / "stats" / "hourly_weather_actual.json"
            if not actual_file.exists():
                return None

            actual_data = await self._read_json_file(actual_file, None)
            if not actual_data:
                return None

            today_data = actual_data.get("hourly_data", {}).get(today, {})

            cloud_values = []
            for hour in range(8, 12):  # Morning hours 8-11
                hour_data = today_data.get(str(hour), {})
                cloud = hour_data.get("cloud_cover_percent")
                if cloud is not None:
                    cloud_values.append(cloud)

            if not cloud_values:
                return None

            return sum(cloud_values) / len(cloud_values)

        except Exception as e:
            _LOGGER.debug(f"Error getting morning actual clouds: {e}")
            return None

    async def _get_afternoon_forecast_clouds(self, current_time: datetime) -> Optional[float]:
        """Get forecasted cloud cover for afternoon hours (12-17) @zara"""
        try:
            today = current_time.date().isoformat()

            corrected_file = self.data_manager.weather_corrected_file
            if not corrected_file.exists():
                return None

            corrected_data = await self._read_json_file(corrected_file, None)
            if not corrected_data:
                return None

            today_forecast = corrected_data.get("forecast", {}).get(today, {})
            if not today_forecast:
                return None

            cloud_values = []
            for hour in range(12, 18):  # Afternoon hours 12-17
                hour_data = today_forecast.get(str(hour), {})
                clouds = hour_data.get("clouds")
                if clouds is not None:
                    cloud_values.append(clouds)

            if not cloud_values:
                return None

            return sum(cloud_values) / len(cloud_values)

        except Exception as e:
            _LOGGER.debug(f"Error getting afternoon forecast clouds: {e}")
            return None

    async def _get_morning_production(self, current_time: datetime) -> Optional[float]:
        """Get actual production from 6:00-12:00 today @zara"""
        try:
            today = current_time.date().isoformat()

            # Read from hourly_predictions.json
            predictions_file = self.data_manager.data_dir / "stats" / "hourly_predictions.json"
            if not predictions_file.exists():
                return None

            predictions_data = await self._read_json_file(predictions_file, None)
            if not predictions_data:
                return None

            # Sum up actual_kwh for hours 6-11 (morning)
            total_production = 0.0
            hours_found = 0

            predictions = predictions_data.get("predictions", [])
            for pred in predictions:
                if pred.get("target_date") != today:
                    continue
                hour = pred.get("target_hour")
                if hour is None or hour < 6 or hour >= 12:
                    continue
                actual_kwh = pred.get("actual_kwh")
                if actual_kwh is not None:
                    total_production += actual_kwh
                    hours_found += 1

            if hours_found == 0:
                _LOGGER.debug("No morning actual production data found")
                return None

            _LOGGER.debug(f"Morning production (6-12): {total_production:.3f} kWh from {hours_found} hours")
            return total_production

        except Exception as e:
            _LOGGER.error(f"Error getting morning production: {e}")
            return None

    async def _get_morning_forecast(self, current_time: datetime) -> Optional[float]:
        """Get forecasted production for 6:00-12:00 today @zara"""
        try:
            today = current_time.date().isoformat()

            # Read from hourly_predictions.json
            predictions_file = self.data_manager.data_dir / "stats" / "hourly_predictions.json"
            if not predictions_file.exists():
                return None

            predictions_data = await self._read_json_file(predictions_file, None)
            if not predictions_data:
                return None

            # Sum up prediction_kwh for hours 6-11 (morning)
            total_forecast = 0.0
            hours_found = 0

            predictions = predictions_data.get("predictions", [])
            for pred in predictions:
                if pred.get("target_date") != today:
                    continue
                hour = pred.get("target_hour")
                if hour is None or hour < 6 or hour >= 12:
                    continue
                pred_kwh = pred.get("prediction_kwh")
                if pred_kwh is not None:
                    total_forecast += pred_kwh
                    hours_found += 1

            if hours_found == 0:
                _LOGGER.debug("No morning forecast data found")
                return None

            _LOGGER.debug(f"Morning forecast (6-12): {total_forecast:.3f} kWh from {hours_found} hours")
            return total_forecast

        except Exception as e:
            _LOGGER.error(f"Error getting morning forecast: {e}")
            return None

    async def _check_afternoon_weather_change(self, current_time: datetime) -> Optional[float]:
        """Check if afternoon weather forecast changed significantly @zara

        Compares current weather forecast (12-17h) with the original morning forecast.

        Returns:
            Cloud cover change in percentage points (positive = less clouds now)
        """
        try:
            today = current_time.date().isoformat()

            # Get current weather forecast
            corrected_file = self.data_manager.weather_corrected_file
            if not corrected_file.exists():
                return None

            corrected_data = await self._read_json_file(corrected_file, None)
            if not corrected_data:
                return None

            today_forecast = corrected_data.get("forecast", {}).get(today, {})
            if not today_forecast:
                return None

            # Get afternoon cloud values (12-17)
            current_clouds = []
            for hour in range(12, 18):
                hour_data = today_forecast.get(str(hour), {})
                clouds = hour_data.get("clouds")
                if clouds is not None:
                    current_clouds.append(clouds)

            if not current_clouds:
                return None

            current_avg = sum(current_clouds) / len(current_clouds)

            # Check if we have stored morning forecast clouds
            metadata = corrected_data.get("metadata", {})
            original_clouds = metadata.get("original_afternoon_clouds")

            if original_clouds is None:
                # First time - store current as original
                if "metadata" not in corrected_data:
                    corrected_data["metadata"] = {}
                corrected_data["metadata"]["original_afternoon_clouds"] = round(current_avg, 1)
                corrected_data["metadata"]["original_stored_at"] = current_time.isoformat()
                await self._atomic_write_json(corrected_file, corrected_data)
                return None

            # Calculate change: positive means less clouds now
            cloud_change = original_clouds - current_avg

            _LOGGER.debug(
                f"Afternoon clouds: original={original_clouds:.1f}%, "
                f"current={current_avg:.1f}%, change={cloud_change:+.1f}%"
            )

            return cloud_change

        except Exception as e:
            _LOGGER.error(f"Error checking afternoon weather change: {e}")
            return None

    async def store_morning_forecast_reference(self, current_time: datetime) -> bool:
        """Store the morning weather forecast as reference for midday comparison @zara

        Should be called around 6:00-7:00 AM to capture the original forecast.
        """
        try:
            today = current_time.date().isoformat()

            corrected_file = self.data_manager.weather_corrected_file
            if not corrected_file.exists():
                return False

            corrected_data = await self._read_json_file(corrected_file, None)
            if not corrected_data:
                return False

            today_forecast = corrected_data.get("forecast", {}).get(today, {})
            if not today_forecast:
                return False

            # Store afternoon cloud average (12-17h)
            afternoon_clouds = []
            for hour in range(12, 18):
                hour_data = today_forecast.get(str(hour), {})
                clouds = hour_data.get("clouds")
                if clouds is not None:
                    afternoon_clouds.append(clouds)

            if afternoon_clouds:
                avg_clouds = sum(afternoon_clouds) / len(afternoon_clouds)

                if "metadata" not in corrected_data:
                    corrected_data["metadata"] = {}

                corrected_data["metadata"]["original_afternoon_clouds"] = round(avg_clouds, 1)
                corrected_data["metadata"]["original_stored_at"] = current_time.isoformat()

                await self._atomic_write_json(corrected_file, corrected_data)
                _LOGGER.debug(f"Stored morning reference: afternoon clouds avg={avg_clouds:.1f}%")
                return True

            return False

        except Exception as e:
            _LOGGER.error(f"Error storing morning forecast reference: {e}")
            return False

    def reset_daily_state(self) -> None:
        """Reset daily tracking state (call at midnight) @zara"""
        self._morning_correction_applied = False
        self._morning_correction_factor = None
        self._morning_forecast_clouds = None
