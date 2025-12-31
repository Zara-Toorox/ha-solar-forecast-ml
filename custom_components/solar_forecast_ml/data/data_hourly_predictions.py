# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from homeassistant.util import dt as dt_util

from ..const import DOMAIN
from .data_shadow_detection import get_shadow_detector

_LOGGER = logging.getLogger(__name__)

class HourlyPredictionsHandler:
    """Manages hourly predictions with ML-optimized structure"""

    def __init__(self, data_dir: Path, data_manager=None):
        self.data_dir = data_dir
        self.data_manager = data_manager
        self.hourly_file = data_dir / "stats" / "hourly_predictions.json"

    def _normalize_weather_fields(self, weather_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Normalize weather field names to ML convention (short names)"""
        if not weather_data:
            return None

        return {

            "temperature": (
                weather_data.get("temperature") or
                weather_data.get("temperature_c")
            ),

            "solar_radiation_wm2": (
                weather_data.get("solar_radiation_wm2") or
                weather_data.get("solar_radiation_wm2")
            ),

            "wind": (
                weather_data.get("wind") or
                weather_data.get("wind_speed") or
                weather_data.get("wind_speed_ms")
            ),

            "humidity": (
                weather_data.get("humidity") or
                weather_data.get("humidity_percent")
            ),

            "rain": (
                weather_data.get("rain") or
                weather_data.get("precipitation") or
                weather_data.get("precipitation_mm") or
                0.0
            ),

            "clouds": (
                weather_data.get("clouds") or
                weather_data.get("cloud_cover") or
                weather_data.get("cloud_cover_percent")
            ),

            "pressure": (
                weather_data.get("pressure") or
                weather_data.get("pressure_hpa")
            ),

            # DNI/DHI for physics engine POA calculation
            "direct_radiation": weather_data.get("direct_radiation"),

            "diffuse_radiation": weather_data.get("diffuse_radiation"),

            "source": weather_data.get("source", "unknown"),
        }

    async def create_daily_predictions(
        self,
        date: str,
        hourly_forecast: List[Dict[str, Any]],
        weather_forecast: List[Dict[str, Any]],
        astronomy_data: Dict[str, Any],
        sensor_config: Dict[str, bool],
        inverter_max_power: float = 0.0,
    ) -> bool:
        """
        Create all hourly predictions for a day (called at 6:00 AM)

        Args:
            date: Date string YYYY-MM-DD
            hourly_forecast: ML hourly predictions with production_kwh
            weather_forecast: Weather service hourly forecast
            astronomy_data: Sun position data
            sensor_config: Which sensors are available
            inverter_max_power: Max AC power in kW. If > 0, forecasts are capped to this value.

        Returns:
            True if successful
        """
        try:
            data = await self._read_json_async()
            prediction_created_at = dt_util.now()

            existing_ids_before = {p["id"] for p in data["predictions"] if "id" in p}
            _LOGGER.debug(f"Existing prediction IDs in DB: {len(existing_ids_before)}")

            predictions_before = len(data["predictions"])
            data["predictions"] = [p for p in data["predictions"] if p.get("target_date") != date]
            removed_count = predictions_before - len(data["predictions"])

            if removed_count > 0:
                _LOGGER.info(
                    f"✓ Removed {removed_count} existing predictions for {date} before creating new ones"
                )

            existing_ids = {p["id"] for p in data["predictions"] if "id" in p}

            _LOGGER.debug(
                f"Creating hourly predictions for {date}, got {len(hourly_forecast)} hours"
            )

            processed_hours = set()
            created_count = 0
            skipped_duplicates = 0

            for hour_data in hourly_forecast:
                hour = hour_data.get("hour")
                hour_datetime = hour_data.get("datetime")

                if hour is None or hour_datetime is None:
                    _LOGGER.warning(f"Skipping hour with missing data: {hour_data}")
                    continue

                prediction_id = f"{date}_{hour}"

                if prediction_id in existing_ids:
                    skipped_duplicates += 1
                    _LOGGER.warning(
                        f"✗ DUPLICATE PREVENTION: Prediction {prediction_id} already exists in DB - SKIPPING"
                    )
                    continue

                if hour in processed_hours:
                    skipped_duplicates += 1
                    _LOGGER.warning(
                        f"✗ DUPLICATE PREVENTION: Hour {hour} already processed in this execution - SKIPPING"
                    )
                    continue

                processed_hours.add(hour)
                existing_ids.add(prediction_id)
                created_count += 1

                weather = await self._find_corrected_weather_for_hour(date, hour)

                if not weather:
                    weather = self._find_weather_for_hour(weather_forecast, hour)

                astro = self._get_astronomy_for_hour(astronomy_data, hour)

                # Get raw prediction and apply inverter clipping if configured
                raw_prediction_kwh = hour_data.get("production_kwh", 0.0)
                if inverter_max_power > 0 and raw_prediction_kwh > inverter_max_power:
                    capped_prediction_kwh = inverter_max_power
                    _LOGGER.debug(
                        f"Forecast capped for hour {hour}: {raw_prediction_kwh:.2f} -> {capped_prediction_kwh:.2f} kWh "
                        f"(inverter max: {inverter_max_power} kW)"
                    )
                else:
                    capped_prediction_kwh = raw_prediction_kwh

                prediction = {

                    "id": f"{date}_{hour}",
                    "prediction_created_at": prediction_created_at.isoformat(),
                    "prediction_created_hour": prediction_created_at.hour,
                    "target_datetime": hour_datetime,
                    "target_date": date,
                    "target_hour": hour,
                    "target_day_of_week": datetime.fromisoformat(date).weekday(),
                    "target_day_of_year": datetime.fromisoformat(date).timetuple().tm_yday,
                    "target_month": datetime.fromisoformat(date).month,
                    "target_season": self._get_season(datetime.fromisoformat(date).month),

                    "prediction_kwh": capped_prediction_kwh,
                    "prediction_kwh_uncapped": raw_prediction_kwh if inverter_max_power > 0 else None,
                    "prediction_method": "blended",
                    "ml_contribution_percent": 0,
                    "model_version": "1.0",
                    "confidence": 75.0,

                    "actual_kwh": None,
                    "actual_measured_at": None,
                    "accuracy_percent": None,
                    "error_kwh": None,
                    "error_percent": None,

                    "weather_forecast": self._normalize_weather_fields(weather),

                    "weather_corrected": self._normalize_weather_fields(weather),

                    "weather_actual": None,

                    "sensor_actual": self._init_sensor_data(sensor_config),

                    "astronomy": astro,

                    "error_analysis": None,

                    "historical_context": None,

                    "production_metrics": {
                        "peak_power_today_kwh": None,
                        "production_hours_today": None,
                        "cumulative_today_kwh": None,
                    },

                    # Per-panel-group predictions (only when panel groups are configured)
                    "panel_group_predictions": hour_data.get("panel_group_predictions"),

                    "flags": {
                        "is_production_hour": hour_data.get("production_kwh", 0) > 0,
                        "is_peak_hour": False,
                        "is_outlier": False,
                        "has_weather_alert": False,
                        "weather_alert_type": None,  # "unexpected_rain", "unexpected_clouds", "sudden_storm", etc.
                        "exclude_from_learning": False,  # Master flag: True if this hour should be excluded from ALL learning
                        "has_sensor_data": False,
                        "sensor_data_complete": False,
                        "weather_forecast_updated": False,
                        "manual_override": False,
                        "inverter_clipped": False,  # Set to True if actual >= 95% of inverter max
                        "has_panel_group_predictions": hour_data.get("panel_group_predictions") is not None,
                    },

                    "quality": {
                        "prediction_confidence": self._get_confidence_level(75.0),
                        "weather_forecast_age_hours": 0,
                        "sensor_data_quality": "unknown",
                        "data_completeness_percent": 100.0,
                    },
                }

                data["predictions"].append(prediction)

            final_ids = [p["id"] for p in data["predictions"]]
            duplicate_ids = [id for id in final_ids if final_ids.count(id) > 1]

            if duplicate_ids:
                _LOGGER.error(
                    f"✗ CRITICAL: Duplicate IDs detected after creation: {set(duplicate_ids)}\n"
                    f"   This should NEVER happen! Aborting write to prevent corruption..."
                )
                return False

            best_hour_str = self._mark_peak_hour_and_get_best(data["predictions"], date)
            data["best_hour_today"] = best_hour_str

            data["last_updated"] = dt_util.now().isoformat()
            await self._write_json_atomic(data)

            _LOGGER.info(
                f"✓ Created {created_count} hourly predictions for {date}\n"
                f"   → Skipped {skipped_duplicates} duplicates\n"
                f"   → Best hour: {best_hour_str}\n"
                f"   → Total predictions in DB: {len(data['predictions'])}\n"
                f"   → Integrity check: PASSED (no duplicates)"
            )
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to create daily predictions: {e}", exc_info=True)
            return False

    async def update_hourly_actual(
        self,
        date: str,
        hour: int,
        actual_kwh: float,
        sensor_data: Dict[str, Any],
        weather_actual: Optional[Dict[str, Any]] = None,
        astronomy_update: Optional[Dict[str, Any]] = None,
        inverter_max_power: float = 0.0,
        panel_group_actuals: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Update actual values for a specific hour

        Called every hour (e.g. at 12:05 to update 11:00-12:00)

        Args:
            inverter_max_power: Max AC power in kW. If > 0 and actual >= 95% of max,
                               the hour is flagged as clipped and excluded from ML training.
            panel_group_actuals: Optional dict mapping group_name to actual_kwh for per-group learning.
                                e.g. {"Gruppe 1": 0.42, "Gruppe 2": 0.58}
        """
        try:
            data = await self._read_json_async()
            prediction_id = f"{date}_{hour}"

            prediction = next(
                (p for p in data["predictions"] if p.get("id") == prediction_id), None
            )

            if not prediction:
                _LOGGER.warning(f"No prediction found for {prediction_id}")
                return False

            prediction["actual_kwh"] = actual_kwh
            prediction["actual_measured_at"] = dt_util.now().isoformat()

            # Inverter clipping detection
            # If actual production is >= 95% of inverter max, mark as clipped
            # Clipped hours are excluded from ML training to prevent bias
            is_clipped = False
            if inverter_max_power > 0:
                clipping_threshold = inverter_max_power * 0.95
                if actual_kwh >= clipping_threshold:
                    is_clipped = True
                    _LOGGER.debug(
                        f"Inverter clipping detected for {prediction_id}: "
                        f"{actual_kwh:.2f} kWh >= {clipping_threshold:.2f} kWh (95% of {inverter_max_power} kW)"
                    )
            prediction["flags"]["inverter_clipped"] = is_clipped

            if prediction["prediction_kwh"] > 0:

                prediction["error_kwh"] = round(actual_kwh - prediction["prediction_kwh"], 3)
                prediction["error_percent"] = round(
                    ((actual_kwh - prediction["prediction_kwh"]) / prediction["prediction_kwh"])
                    * 100,
                    1,
                )

                accuracy = max(0.0, 100.0 - abs(prediction["error_percent"]))
                prediction["accuracy_percent"] = round(accuracy, 1)
            else:

                if actual_kwh < 0.001:
                    prediction["accuracy_percent"] = 100.0
                    prediction["error_kwh"] = 0.0
                    prediction["error_percent"] = 0.0
                else:
                    prediction["accuracy_percent"] = 0.0
                    prediction["error_kwh"] = round(actual_kwh, 3)
                    prediction["error_percent"] = None

            prediction["sensor_actual"] = sensor_data
            prediction["flags"]["has_sensor_data"] = True
            prediction["flags"]["sensor_data_complete"] = self._check_sensor_completeness(
                sensor_data
            )

            if weather_actual:
                prediction["weather_actual"] = weather_actual

            # Unexpected weather detection - compare actual vs forecast
            weather_alert = self._detect_unexpected_weather(
                weather_actual=weather_actual,
                weather_forecast=prediction.get("weather_forecast"),
                sensor_data=sensor_data,
            )
            if weather_alert:
                prediction["flags"]["has_weather_alert"] = True
                prediction["flags"]["weather_alert_type"] = weather_alert["type"]
                prediction["flags"]["exclude_from_learning"] = True
                _LOGGER.info(
                    f"⚠️ Weather alert for {prediction_id}: {weather_alert['type']} - "
                    f"{weather_alert['reason']}"
                )

                # Send Home Assistant notification
                await self._send_weather_alert_notification(
                    alert_type=weather_alert["type"],
                    reason=weather_alert["reason"],
                    hour=hour,
                    date_str=date,
                    weather_actual=weather_actual,
                    weather_forecast=prediction.get("weather_forecast"),
                )

            # Store per-panel-group actual production for group-specific learning
            if panel_group_actuals:
                prediction["panel_group_actuals"] = panel_group_actuals
                prediction["flags"]["has_panel_group_actuals"] = True
                _LOGGER.debug(
                    f"Stored panel group actuals for {prediction_id}: {panel_group_actuals}"
                )

                # Calculate per-group accuracy if predictions exist
                panel_group_predictions = prediction.get("panel_group_predictions")

                # Backfill: If predictions are missing, estimate from total prediction
                # This enables accuracy calculation even for hours created before per-group feature
                if not panel_group_predictions and prediction.get("prediction_kwh", 0) > 0:
                    total_pred = prediction["prediction_kwh"]
                    # Distribute proportionally based on actuals (best available approximation)
                    total_actual = sum(panel_group_actuals.values())
                    if total_actual > 0.001:
                        panel_group_predictions = {
                            name: (actual / total_actual) * total_pred
                            for name, actual in panel_group_actuals.items()
                        }
                        prediction["panel_group_predictions"] = panel_group_predictions
                        prediction["flags"]["has_panel_group_predictions"] = True
                        prediction["flags"]["panel_group_predictions_backfilled"] = True
                        _LOGGER.debug(
                            f"Backfilled panel_group_predictions for {prediction_id} from total prediction"
                        )

                if panel_group_predictions:
                    panel_group_accuracy = {}
                    for group_name, actual_kwh_group in panel_group_actuals.items():
                        pred_kwh_group = panel_group_predictions.get(group_name)
                        if pred_kwh_group is not None and pred_kwh_group > 0:
                            error_kwh = actual_kwh_group - pred_kwh_group
                            error_percent = (error_kwh / pred_kwh_group) * 100
                            accuracy = max(0.0, 100.0 - abs(error_percent))
                            panel_group_accuracy[group_name] = {
                                "prediction_kwh": round(pred_kwh_group, 4),
                                "actual_kwh": round(actual_kwh_group, 4),
                                "error_kwh": round(error_kwh, 4),
                                "error_percent": round(error_percent, 1),
                                "accuracy_percent": round(accuracy, 1),
                            }
                        elif actual_kwh_group < 0.001:
                            # Both near zero - perfect accuracy
                            panel_group_accuracy[group_name] = {
                                "prediction_kwh": round(pred_kwh_group or 0, 4),
                                "actual_kwh": round(actual_kwh_group, 4),
                                "error_kwh": 0.0,
                                "error_percent": 0.0,
                                "accuracy_percent": 100.0,
                            }
                    if panel_group_accuracy:
                        prediction["panel_group_accuracy"] = panel_group_accuracy
                        _LOGGER.debug(
                            f"Calculated panel group accuracy for {prediction_id}: "
                            f"{list(panel_group_accuracy.keys())}"
                        )

            if astronomy_update and prediction.get("astronomy"):
                prediction["astronomy"].update(astronomy_update)

            prediction["historical_context"] = self._get_historical_context(
                data["predictions"], date, hour
            )

            prediction["production_metrics"] = self._calculate_production_metrics(
                data["predictions"], date, sensor_data
            )

            try:
                if (
                    actual_kwh is not None
                    and prediction.get("astronomy", {}).get("theoretical_max_kwh") is not None
                ):
                    shadow_detector = get_shadow_detector(self.data_manager)
                    shadow_result = await shadow_detector.detect_shadow_ensemble(prediction)

                    prediction["shadow_detection"] = shadow_result

                    if shadow_result.get("shadow_type") not in ["error", "night"]:
                        prediction["performance_loss"] = {
                            "shadow_percent": shadow_result.get("shadow_percent", 0),
                            "shadow_type": shadow_result.get("shadow_type", "unknown"),
                            "loss_kwh": shadow_result.get("loss_kwh", 0),
                            "efficiency_ratio": shadow_result.get("efficiency_ratio", 0),
                            "root_cause": shadow_result.get("root_cause", "unknown"),
                            "confidence": shadow_result.get("confidence", 0)
                        }
                    else:

                        prediction["performance_loss"] = {
                            "shadow_type": shadow_result.get("shadow_type", "unknown"),
                            "note": shadow_result.get("interpretation", "N/A")
                        }

                    _LOGGER.debug(
                        f"Shadow detection for {prediction_id}: "
                        f"{shadow_result.get('shadow_type')} "
                        f"({shadow_result.get('shadow_percent', 0):.1f}%)"
                    )

            except Exception as shadow_error:
                _LOGGER.warning(
                    f"Shadow detection failed for {prediction_id}: {shadow_error}",
                    exc_info=False
                )

                prediction["shadow_detection"] = {
                    "error": str(shadow_error),
                    "shadow_type": "error",
                    "confidence": 0.0
                }

            data["last_updated"] = dt_util.now().isoformat()
            await self._write_json_atomic(data)

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to update hourly actual: {e}", exc_info=True)
            return False

    def get_prediction_by_id(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Get specific prediction by ID (e.g. '2025-11-10_12')"""
        try:
            asyncio.get_running_loop()
            _LOGGER.debug(
                "get_prediction_by_id called in event loop; returning None to avoid blocking I/O"
            )
            return None
        except RuntimeError:
            pass

        data = self._read_json()
        return next((p for p in data["predictions"] if p.get("id") == prediction_id), None)

    async def get_predictions_for_date(self, date: str) -> List[Dict[str, Any]]:
        """Get all predictions for a specific date"""
        data = await self._read_json_async()
        return [p for p in data["predictions"] if p.get("target_date") == date]

    def get_next_hour_prediction(self) -> Optional[Dict[str, Any]]:
        """Get prediction for next full hour"""
        now = dt_util.now()
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        prediction_id = f"{next_hour.date().isoformat()}_{next_hour.hour}"
        return self.get_prediction_by_id(prediction_id)

    async def get_best_hour_today(self) -> Optional[Dict[str, Any]]:
        """Get prediction with highest production for today"""
        today = dt_util.now().date().isoformat()
        today_predictions = await self.get_predictions_for_date(today)

        if not today_predictions:
            return None

        return max(today_predictions, key=lambda x: x.get("prediction_kwh", 0))

    async def get_best_hour_string(self) -> Optional[str]:
        """Get best hour as string (e.g. '11:00') from top-level field"""
        data = await self._read_json_async()
        return data.get("best_hour_today")

    def _find_weather_for_hour(self, weather_forecast: List[Dict], hour: int) -> Dict:
        """Find weather data for specific hour"""
        for w in weather_forecast:
            if w.get("local_hour") == hour:
                temp = w.get("temperature")
                humidity = w.get("humidity")

                dew_point = None
                if temp is not None and humidity is not None:
                    try:

                        import math

                        a = 17.27
                        b = 237.7
                        alpha = ((a * temp) / (b + temp)) + math.log(humidity / 100.0)
                        dew_point = (b * alpha) / (a - alpha)
                    except:
                        dew_point = None

                return {
                    "source": "met.no",
                    "temperature_c": temp,
                    "feels_like_c": temp,
                    "cloud_cover_percent": w.get("cloud_cover"),
                    "condition": w.get("condition"),
                    "humidity_percent": humidity,
                    "wind_speed_ms": w.get("wind_speed"),
                    "wind_direction": w.get("wind_direction"),
                    "precipitation_mm": w.get("precipitation"),
                    "pressure_hpa": w.get("pressure"),
                    "solar_radiation_wm2": None,
                    "uv_index": None,
                    "visibility_km": None,
                    "dew_point_c": round(dew_point, 1) if dew_point is not None else None,
                }
        return {}

    async def _find_corrected_weather_for_hour(self, date: str, hour: int) -> Dict:
        """Find corrected weather data for specific date and hour from weather_forecast_corrected.json"""
        try:
            import json
            import asyncio
            from pathlib import Path

            corrected_file = Path(self.data_dir) / "stats" / "weather_forecast_corrected.json"

            if not corrected_file.exists():
                _LOGGER.debug(f"Corrected weather file not found, using raw weather")
                return {}

            def _load_file():
                with open(corrected_file, 'r') as f:
                    return json.load(f)

            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, _load_file)

            if not data or "forecast" not in data:
                return {}

            forecast_by_date = data.get("forecast", {})

            day_forecast = forecast_by_date.get(date)
            if not day_forecast:
                _LOGGER.debug(f"No corrected weather found for date {date}")
                return {}

            hour_str = str(hour)
            corrected = day_forecast.get(hour_str)

            if not corrected:
                _LOGGER.debug(f"No corrected weather found for {date} {hour:02d}:00 (hour key: '{hour_str}')")
                return {}

            temp = corrected.get("temperature")
            humidity = corrected.get("humidity")

            dew_point = None
            if temp is not None and humidity is not None:
                try:
                    import math
                    a = 17.27
                    b = 237.7
                    alpha = ((a * temp) / (b + temp)) + math.log(humidity / 100.0)
                    dew_point = (b * alpha) / (a - alpha)
                except:
                    dew_point = None

            _LOGGER.debug(
                f"Using CORRECTED weather for {date} {hour:02d}:00 "
                f"(temp: {temp}°C, lux: {corrected.get('lux')}, clouds: {corrected.get('clouds')}%, "
                f"humidity: {humidity}%, wind: {corrected.get('wind')} m/s)"
            )

            return {
                "source": "met.no (corrected)",
                "temperature_c": temp,
                "feels_like_c": temp,
                "cloud_cover_percent": corrected.get("clouds"),
                "condition": None,
                "humidity_percent": humidity,
                "wind_speed_ms": corrected.get("wind"),
                "wind_direction": None,
                "precipitation_mm": corrected.get("rain"),
                "pressure_hpa": corrected.get("pressure"),
                "solar_radiation_wm2": corrected.get("solar_radiation_wm2"),
                # DNI/DHI for physics engine POA calculation
                "direct_radiation": corrected.get("direct_radiation"),
                "diffuse_radiation": corrected.get("diffuse_radiation"),
                "uv_index": None,
                "visibility_km": None,
                "dew_point_c": round(dew_point, 1) if dew_point is not None else None,
            }

        except Exception as e:
            _LOGGER.warning(f"Error loading corrected weather for {date} {hour:02d}:00: {e}")
            return {}

    def _get_astronomy_for_hour(self, astro_data: Dict, hour: int) -> Dict:
        """Get astronomy data for specific hour from astronomy_cache"""

        hourly_astro = astro_data.get("hourly", {}).get(str(hour), {})

        sunrise = astro_data.get("sunrise")
        sunset = astro_data.get("sunset")
        solar_noon = astro_data.get("solar_noon")

        hours_after_sunrise = None
        hours_before_sunset = None

        if sunrise and sunset:
            try:
                from datetime import datetime

                sunrise_time = (
                    datetime.fromisoformat(sunrise) if isinstance(sunrise, str) else sunrise
                )
                sunset_time = datetime.fromisoformat(sunset) if isinstance(sunset, str) else sunset

                if hasattr(sunrise_time, "date"):
                    hour_time = sunrise_time.replace(hour=hour, minute=30, second=0, microsecond=0)
                    hours_after_sunrise = (hour_time - sunrise_time).total_seconds() / 3600.0
                    hours_before_sunset = (sunset_time - hour_time).total_seconds() / 3600.0
            except Exception as e:
                _LOGGER.debug(f"Could not calculate sunrise/sunset hours: {e}")

        return {

            "sunrise": sunrise,
            "sunset": sunset,
            "solar_noon": solar_noon,
            "daylight_hours": astro_data.get("daylight_hours"),

            "sun_elevation_deg": hourly_astro.get("elevation_deg"),
            "sun_azimuth_deg": hourly_astro.get("azimuth_deg"),
            "clear_sky_radiation_wm2": hourly_astro.get("clear_sky_solar_radiation_wm2"),
            "theoretical_max_kwh": hourly_astro.get("theoretical_max_pv_kwh"),
            "hours_since_solar_noon": hourly_astro.get("hours_since_solar_noon"),
            "day_progress_ratio": hourly_astro.get("day_progress_ratio"),

            "hours_after_sunrise": (
                round(hours_after_sunrise, 2) if hours_after_sunrise is not None else None
            ),
            "hours_before_sunset": (
                round(hours_before_sunset, 2) if hours_before_sunset is not None else None
            ),
        }

    def _init_sensor_data(self, sensor_config: Dict[str, bool]) -> Dict:
        """Initialize sensor data structure based on config"""
        return {
            "temperature_c": None,
            "humidity_percent": None,
            "solar_radiation_wm2": None,
            "rain_mm": None,
            "uv_index": None,
            "wind_speed_ms": None,
            "current_yield_kwh": None,
        }

    def _get_season(self, month: int) -> str:
        """Get season from month"""
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "autumn"
        else:
            return "winter"

    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence percentage to level"""
        if confidence >= 80:
            return "high"
        elif confidence >= 60:
            return "medium"
        else:
            return "low"

    def _check_sensor_completeness(self, sensor_data: Dict) -> bool:
        """Check if all configured sensors have data"""
        non_null_count = sum(1 for v in sensor_data.values() if v is not None)
        return non_null_count >= 3

    def _detect_unexpected_weather(
        self,
        weather_actual: Optional[Dict[str, Any]],
        weather_forecast: Optional[Dict[str, Any]],
        sensor_data: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, str]]:
        """Detect unexpected weather events by comparing actual vs forecast.

        Returns alert dict if unexpected weather detected, None otherwise.

        Alert types:
        - unexpected_rain: Rain sensor shows precipitation, forecast had none
        - unexpected_clouds: Solar radiation significantly lower than forecast
        - sudden_storm: Large pressure drop indicating storm
        - unexpected_snow: Temperature < 2°C with precipitation

       
        """
        if not weather_actual and not sensor_data:
            return None

        # Get actual values from sensors or weather_actual
        actual_rain = None
        actual_temp = None
        actual_radiation = None
        actual_pressure = None

        if weather_actual:
            actual_rain = weather_actual.get("precipitation_mm")
            actual_temp = weather_actual.get("temperature_c")
            actual_radiation = weather_actual.get("solar_radiation_wm2")
            actual_pressure = weather_actual.get("pressure_hpa")

        # Override with sensor data if available (more accurate)
        if sensor_data:
            if sensor_data.get("rain_mm") is not None:
                actual_rain = sensor_data.get("rain_mm")
            if sensor_data.get("temperature_c") is not None:
                actual_temp = sensor_data.get("temperature_c")
            if sensor_data.get("solar_radiation_wm2") is not None:
                actual_radiation = sensor_data.get("solar_radiation_wm2")

        # Get forecast values
        forecast_rain = 0.0
        forecast_radiation = None
        forecast_clouds = None

        if weather_forecast:
            forecast_rain = weather_forecast.get("precipitation_mm") or 0.0
            forecast_radiation = weather_forecast.get("solar_radiation_wm2")
            forecast_clouds = weather_forecast.get("cloud_cover_percent")

        # Detection 1: Unexpected rain
        # Actual rain > 0.5mm but forecast had < 0.1mm
        if actual_rain is not None and actual_rain > 0.5 and forecast_rain < 0.1:
            return {
                "type": "unexpected_rain",
                "reason": f"Actual rain {actual_rain:.1f}mm, forecast {forecast_rain:.1f}mm",
            }

        # Detection 2: Unexpected snow
        # Temperature < 2°C and precipitation > 0.5mm
        if (
            actual_temp is not None
            and actual_rain is not None
            and actual_temp < 2.0
            and actual_rain > 0.5
            and forecast_rain < 0.1
        ):
            return {
                "type": "unexpected_snow",
                "reason": f"Temp {actual_temp:.1f}°C with {actual_rain:.1f}mm precipitation (not forecast)",
            }

        # Detection 3: Unexpected clouds / sudden radiation drop
        # Actual radiation < 40% of forecast (significant drop)
        if (
            actual_radiation is not None
            and forecast_radiation is not None
            and forecast_radiation > 100  # Only during daylight with meaningful forecast
            and actual_radiation < forecast_radiation * 0.4
        ):
            return {
                "type": "unexpected_clouds",
                "reason": f"Radiation {actual_radiation:.0f} W/m² vs forecast {forecast_radiation:.0f} W/m² ({actual_radiation/forecast_radiation*100:.0f}%)",
            }

        # Detection 4: Snow-covered panels (persistent snow from earlier)
        # Check if panels are marked as snow-covered from weather_actual_tracker
        if weather_actual and weather_actual.get("snow_covered_panels"):
            source = weather_actual.get("snow_coverage_source", "unknown")
            return {
                "type": "snow_covered_panels",
                "reason": f"Panels snow-covered (source: {source})",
            }

        return None

    def _get_historical_context(
        self, all_predictions: List[Dict], current_date: str, current_hour: int
    ) -> Dict[str, float]:
        """Get historical context for ML training"""
        try:
            current_dt = datetime.fromisoformat(current_date)

            yesterday = (current_dt - timedelta(days=1)).isoformat()
            yesterday_id = f"{yesterday}_{current_hour}"
            yesterday_pred = next(
                (
                    p
                    for p in all_predictions
                    if p.get("id") == yesterday_id and p.get("actual_kwh") is not None
                ),
                None,
            )
            yesterday_kwh = yesterday_pred.get("actual_kwh", 0.0) if yesterday_pred else 0.0

            last_7_days_kwh = []
            for days_back in range(1, 8):
                past_date = (current_dt - timedelta(days=days_back)).isoformat()
                past_id = f"{past_date}_{current_hour}"
                past_pred = next(
                    (
                        p
                        for p in all_predictions
                        if p.get("id") == past_id and p.get("actual_kwh") is not None
                    ),
                    None,
                )
                if past_pred:
                    last_7_days_kwh.append(past_pred.get("actual_kwh", 0.0))

            avg_7days = sum(last_7_days_kwh) / len(last_7_days_kwh) if last_7_days_kwh else 0.0

            return {
                "yesterday_same_hour": round(yesterday_kwh, 3),
                "same_hour_avg_7days": round(avg_7days, 3),
            }

        except Exception as e:
            _LOGGER.debug(f"Failed to get historical context: {e}")
            return {"yesterday_same_hour": 0.0, "same_hour_avg_7days": 0.0}

    def _calculate_production_metrics(
        self, all_predictions: List[Dict], current_date: str, sensor_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate daily production metrics"""
        try:
            today_predictions = [
                p
                for p in all_predictions
                if p.get("target_date") == current_date and p.get("actual_kwh") is not None
            ]

            if not today_predictions:
                return {
                    "peak_power_today_kwh": None,
                    "production_hours_today": None,
                    "cumulative_today_kwh": None,
                }

            peak_kwh = max(p.get("actual_kwh", 0.0) for p in today_predictions)

            production_hours = sum(1 for p in today_predictions if p.get("actual_kwh", 0.0) > 0.001)

            cumulative_kwh = sensor_data.get("current_yield_kwh")

            return {
                "peak_power_today_kwh": round(peak_kwh, 3),
                "production_hours_today": production_hours,
                "cumulative_today_kwh": (
                    round(cumulative_kwh, 3) if cumulative_kwh is not None else None
                ),
            }

        except Exception as e:
            _LOGGER.debug(f"Failed to calculate production metrics: {e}")
            return {
                "peak_power_today_kwh": None,
                "production_hours_today": None,
                "cumulative_today_kwh": None,
            }

    def _mark_peak_hour_and_get_best(self, predictions: List[Dict], date: str) -> Optional[str]:
        """Mark the hour with highest prediction as peak and return best hour string"""
        date_predictions = [p for p in predictions if p.get("target_date") == date]
        if date_predictions:
            peak = max(date_predictions, key=lambda x: x.get("prediction_kwh", 0))
            peak["flags"]["is_peak_hour"] = True
            best_hour = peak.get("target_hour")
            if best_hour is not None:
                return f"{best_hour:02d}:00"
        return None

    def _read_json(self) -> Dict:
        """Read JSON file (blocking - use in sync context only)"""
        try:
            with open(self.hourly_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self._ensure_file_exists()
            with open(self.hourly_file, "r") as f:
                return json.load(f)

    async def _read_json_async(self) -> Dict:
        """Read JSON file (non-blocking - use in async context)"""

        def _do_read():
            try:
                with open(self.hourly_file, "r") as f:
                    return json.load(f)
            except FileNotFoundError:
                self._ensure_file_exists()
                with open(self.hourly_file, "r") as f:
                    return json.load(f)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do_read)

    def _write_json(self, data: Dict):
        """Blocking I/O not allowed - use _write_json_atomic instead."""
        raise RuntimeError(
            "_write_json() removed - use _write_json_atomic() or call from executor."
        )

    async def _write_json_atomic(self, data: Dict):
        """Write JSON atomically using DataManager's thread-safe method"""
        if self.data_manager:
            await self.data_manager._atomic_write_json(self.hourly_file, data)
        else:

            def _do_write():
                temp_file = self.hourly_file.with_suffix(".tmp")
                with open(temp_file, "w") as f:
                    json.dump(data, f, indent=2)
                temp_file.replace(self.hourly_file)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _do_write)

    async def _send_weather_alert_notification(
        self,
        alert_type: str,
        reason: str,
        hour: int,
        date_str: str,
        weather_actual: Optional[Dict] = None,
        weather_forecast: Optional[Dict] = None,
    ) -> None:
        """Send a weather alert notification via Home Assistant"""
        try:
            # Access notification service via data_manager -> hass
            if not self.data_manager or not hasattr(self.data_manager, 'hass'):
                _LOGGER.debug("Cannot send weather alert notification: no hass access")
                return

            hass = self.data_manager.hass
            notification_service = hass.data.get(DOMAIN, {}).get("notification_service")

            if not notification_service:
                _LOGGER.debug("Notification service not available for weather alert")
                return

            await notification_service.show_weather_alert(
                alert_type=alert_type,
                reason=reason,
                hour=hour,
                date_str=date_str,
                weather_actual=weather_actual,
                weather_forecast=weather_forecast,
            )

            _LOGGER.info(f"Weather alert notification sent for {date_str} {hour:02d}:00")

        except Exception as e:
            _LOGGER.error(f"Error sending weather alert notification: {e}")

    def _ensure_file_exists(self):
        """Ensure the hourly predictions file exists with initial structure"""
        if not self.hourly_file.parent.exists():
            self.hourly_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.hourly_file.exists():
            initial_data = {
                "version": "2.0",
                "last_updated": datetime.now().isoformat(),
                "best_hour_today": None,
                "predictions": [],
            }
            with open(self.hourly_file, "w") as f:
                json.dump(initial_data, f, indent=2)
