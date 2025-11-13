"""Handler for hourly prediction data - ML optimized structure"""
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import logging
import asyncio
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)


class HourlyPredictionsHandler:
    """Manages hourly predictions with ML-optimized structure"""

    def __init__(self, data_dir: Path, data_manager=None):
        self.data_dir = data_dir
        self.data_manager = data_manager
        self.hourly_file = data_dir / "stats" / "hourly_predictions.json"
        # NOTE: File creation moved to async ensure method to avoid blocking I/O in __init__
        # Will be called on first async access

    async def ensure_file_exists(self):
        """Create file with initial structure (async, non-blocking)"""
        if not self.hourly_file.exists():
            self.hourly_file.parent.mkdir(parents=True, exist_ok=True)
            initial_data = {
                "version": "2.0",
                "last_updated": dt_util.now().isoformat(),
                "best_hour_today": None,
                "metadata": {
                    "system_id": "solar_system_001",
                    "location": {},
                    "system_specs": {},
                    "sensor_config": {}
                },
                "predictions": []
            }
            await self._write_json_atomic(initial_data)
            _LOGGER.info("Created new hourly_predictions.json")

    async def create_daily_predictions(
        self,
        date: str,
        hourly_forecast: List[Dict[str, Any]],
        weather_forecast: List[Dict[str, Any]],
        astronomy_data: Dict[str, Any],
        sensor_config: Dict[str, bool]
    ) -> bool:
        """
        Create all hourly predictions for a day (called at 6:00 AM)

        Args:
            date: Date string YYYY-MM-DD
            hourly_forecast: ML hourly predictions with production_kwh
            weather_forecast: Weather service hourly forecast
            astronomy_data: Sun position data
            sensor_config: Which sensors are available

        Returns:
            True if successful
        """
        try:
            data = await self._read_json_async()
            prediction_created_at = dt_util.now()

            # Remove old predictions for this date (if exists)
            data["predictions"] = [
                p for p in data["predictions"]
                if p.get("target_date") != date
            ]

            _LOGGER.debug(f"Creating hourly predictions for {date}, got {len(hourly_forecast)} hours")

            # Create prediction for each hour
            for hour_data in hourly_forecast:
                hour = hour_data.get("hour")
                hour_datetime = hour_data.get("datetime")

                if hour is None or hour_datetime is None:
                    _LOGGER.warning(f"Skipping hour with missing data: {hour_data}")
                    continue

                # Find matching weather data
                weather = self._find_weather_for_hour(weather_forecast, hour)

                # Find astronomy data for this hour
                astro = self._get_astronomy_for_hour(astronomy_data, hour)

                prediction = {
                    # Identification
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

                    # Prediction
                    "predicted_kwh": hour_data.get("production_kwh", 0.0),
                    "prediction_method": "blended",
                    "ml_contribution_percent": 0,
                    "model_version": "1.0",
                    "confidence": 75.0,

                    # Actual values (filled later)
                    "actual_kwh": None,
                    "actual_measured_at": None,
                    "accuracy_percent": None,
                    "error_kwh": None,
                    "error_percent": None,

                    # Weather forecast
                    "weather_forecast": weather,

                    # Weather actual (filled later)
                    "weather_actual": None,

                    # Sensor data (filled later)
                    "sensor_actual": self._init_sensor_data(sensor_config),

                    # Astronomy
                    "astronomy": astro,

                    # Error analysis (filled later)
                    "error_analysis": None,

                    # Historical context (filled later)
                    "historical_context": None,

                    # Production metrics (filled later with actual data)
                    "production_metrics": {
                        "peak_power_today_kwh": None,
                        "production_hours_today": None,
                        "cumulative_today_kwh": None
                    },

                    # Flags
                    "flags": {
                        "is_production_hour": hour_data.get("production_kwh", 0) > 0,
                        "is_peak_hour": False,
                        "is_outlier": False,
                        "has_weather_alert": False,
                        "has_sensor_data": False,
                        "sensor_data_complete": False,
                        "weather_forecast_updated": False,
                        "manual_override": False
                    },

                    # Quality
                    "quality": {
                        "prediction_confidence": self._get_confidence_level(75.0),
                        "weather_forecast_age_hours": 0,
                        "sensor_data_quality": "unknown",
                        "data_completeness_percent": 100.0
                    }
                }

                data["predictions"].append(prediction)

            # Mark peak hour and save best_hour_today
            best_hour_str = self._mark_peak_hour_and_get_best(data["predictions"], date)
            data["best_hour_today"] = best_hour_str

            data["last_updated"] = dt_util.now().isoformat()
            await self._write_json_atomic(data)

            _LOGGER.info(f"Created {len(hourly_forecast)} hourly predictions for {date} - Best hour: {best_hour_str}")
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
        astronomy_update: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update actual values for a specific hour

        Called every hour (e.g. at 12:05 to update 11:00-12:00)
        """
        try:
            data = await self._read_json_async()
            prediction_id = f"{date}_{hour}"

            # Find the prediction
            prediction = next(
                (p for p in data["predictions"] if p.get("id") == prediction_id),
                None
            )

            if not prediction:
                _LOGGER.warning(f"No prediction found for {prediction_id}")
                return False

            # Update actual values
            prediction["actual_kwh"] = actual_kwh
            prediction["actual_measured_at"] = dt_util.now().isoformat()

            # Calculate accuracy and error
            if prediction["predicted_kwh"] > 0:
                # Calculate error
                prediction["error_kwh"] = round(actual_kwh - prediction["predicted_kwh"], 3)
                prediction["error_percent"] = round(
                    ((actual_kwh - prediction["predicted_kwh"]) / prediction["predicted_kwh"]) * 100, 1
                )

                # Calculate accuracy: 100% - abs(error_percent), capped at 0-100%
                # Example: predicted=0.1, actual=0.2 → error=+100% → accuracy=0%
                # Example: predicted=0.1, actual=0.09 → error=-10% → accuracy=90%
                accuracy = max(0.0, 100.0 - abs(prediction["error_percent"]))
                prediction["accuracy_percent"] = round(accuracy, 1)
            else:
                # predicted_kwh is 0 (e.g., night hour)
                if actual_kwh < 0.001:  # Both predicted and actual are ~0
                    prediction["accuracy_percent"] = 100.0
                    prediction["error_kwh"] = 0.0
                    prediction["error_percent"] = 0.0
                else:  # Predicted 0 but got production (unexpected)
                    prediction["accuracy_percent"] = 0.0
                    prediction["error_kwh"] = round(actual_kwh, 3)
                    prediction["error_percent"] = None  # Cannot divide by zero

            # Update sensor data
            prediction["sensor_actual"] = sensor_data
            prediction["flags"]["has_sensor_data"] = True
            prediction["flags"]["sensor_data_complete"] = self._check_sensor_completeness(sensor_data)

            # Update weather actual
            if weather_actual:
                prediction["weather_actual"] = weather_actual

            if astronomy_update and prediction.get("astronomy"):
                prediction["astronomy"].update(astronomy_update)

            prediction["historical_context"] = self._get_historical_context(
                data["predictions"], date, hour
            )

            prediction["production_metrics"] = self._calculate_production_metrics(
                data["predictions"], date, sensor_data
            )

            data["last_updated"] = dt_util.now().isoformat()
            await self._write_json_atomic(data)

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to update hourly actual: {e}", exc_info=True)
            return False

    def get_prediction_by_id(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Get specific prediction by ID (e.g. '2025-11-10_12')"""
        data = self._read_json()
        return next(
            (p for p in data["predictions"] if p.get("id") == prediction_id),
            None
        )

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

        return max(today_predictions, key=lambda x: x.get("predicted_kwh", 0))

    async def get_best_hour_string(self) -> Optional[str]:
        """Get best hour as string (e.g. '11:00') from top-level field"""
        data = await self._read_json_async()
        return data.get("best_hour_today")

    def _find_weather_for_hour(self, weather_forecast: List[Dict], hour: int) -> Dict:
        """Find weather data for specific hour"""
        for w in weather_forecast:
            if w.get("local_hour") == hour:
                return {
                    "source": "met.no",
                    "temperature_c": w.get("temperature"),
                    "feels_like_c": w.get("temperature"),
                    "cloud_cover_percent": w.get("cloud_cover"),
                    "condition": w.get("condition"),
                    "humidity_percent": w.get("humidity"),
                    "wind_speed_ms": w.get("wind_speed"),
                    "wind_direction": w.get("wind_direction"),
                    "precipitation_mm": w.get("precipitation"),
                    "pressure_hpa": w.get("pressure"),
                    "solar_radiation_wm2": None,
                    "uv_index": None,
                    "visibility_km": None,
                    "dew_point_c": None
                }
        return {}

    def _get_astronomy_for_hour(self, astro_data: Dict, hour: int) -> Dict:
        """Get astronomy data for specific hour"""
        return {
            "sunrise": astro_data.get("sunrise"),
            "sunset": astro_data.get("sunset"),
            "solar_noon": astro_data.get("solar_noon"),
            "daylight_hours": astro_data.get("daylight_hours"),
            "sun_elevation_deg": None,
            "sun_azimuth_deg": None,
            "hours_after_sunrise": None,
            "hours_before_sunset": None
        }

    def _init_sensor_data(self, sensor_config: Dict[str, bool]) -> Dict:
        """Initialize sensor data structure based on config"""
        return {
            "temperature_c": None,
            "humidity_percent": None,
            "lux": None,
            "rain_mm": None,
            "uv_index": None,
            "wind_speed_ms": None,
            "current_power_w": None,
            "current_yield_kwh": None
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
        return non_null_count >= 3  # At least 3 sensors have data

    def _get_historical_context(self, all_predictions: List[Dict], current_date: str, current_hour: int) -> Dict[str, float]:
        """Get historical context for ML training"""
        try:
            current_dt = datetime.fromisoformat(current_date)

            yesterday = (current_dt - timedelta(days=1)).isoformat()
            yesterday_id = f"{yesterday}_{current_hour}"
            yesterday_pred = next(
                (p for p in all_predictions if p.get("id") == yesterday_id and p.get("actual_kwh") is not None),
                None
            )
            yesterday_kwh = yesterday_pred.get("actual_kwh", 0.0) if yesterday_pred else 0.0

            last_7_days_kwh = []
            for days_back in range(1, 8):
                past_date = (current_dt - timedelta(days=days_back)).isoformat()
                past_id = f"{past_date}_{current_hour}"
                past_pred = next(
                    (p for p in all_predictions if p.get("id") == past_id and p.get("actual_kwh") is not None),
                    None
                )
                if past_pred:
                    last_7_days_kwh.append(past_pred.get("actual_kwh", 0.0))

            avg_7days = sum(last_7_days_kwh) / len(last_7_days_kwh) if last_7_days_kwh else 0.0

            return {
                "yesterday_same_hour": round(yesterday_kwh, 3),
                "same_hour_avg_7days": round(avg_7days, 3)
            }

        except Exception as e:
            _LOGGER.debug(f"Failed to get historical context: {e}")
            return {
                "yesterday_same_hour": 0.0,
                "same_hour_avg_7days": 0.0
            }

    def _calculate_production_metrics(self, all_predictions: List[Dict], current_date: str, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate daily production metrics"""
        try:
            today_predictions = [
                p for p in all_predictions
                if p.get("target_date") == current_date and p.get("actual_kwh") is not None
            ]

            if not today_predictions:
                return {
                    "peak_power_today_kwh": None,
                    "production_hours_today": None,
                    "cumulative_today_kwh": None
                }

            # Find peak hour (highest actual_kwh)
            peak_kwh = max(p.get("actual_kwh", 0.0) for p in today_predictions)

            # Count production hours (hours where actual_kwh > 0.001)
            production_hours = sum(1 for p in today_predictions if p.get("actual_kwh", 0.0) > 0.001)

            # Get cumulative production today from current_yield_kwh sensor
            cumulative_kwh = sensor_data.get("current_yield_kwh")

            return {
                "peak_power_today_kwh": round(peak_kwh, 3),
                "production_hours_today": production_hours,
                "cumulative_today_kwh": round(cumulative_kwh, 3) if cumulative_kwh is not None else None
            }

        except Exception as e:
            _LOGGER.debug(f"Failed to calculate production metrics: {e}")
            return {
                "peak_power_today_kwh": None,
                "production_hours_today": None,
                "cumulative_today_kwh": None
            }

    def _mark_peak_hour_and_get_best(self, predictions: List[Dict], date: str) -> Optional[str]:
        """Mark the hour with highest prediction as peak and return best hour string"""
        date_predictions = [p for p in predictions if p.get("target_date") == date]
        if date_predictions:
            peak = max(date_predictions, key=lambda x: x.get("predicted_kwh", 0))
            peak["flags"]["is_peak_hour"] = True
            best_hour = peak.get("target_hour")
            if best_hour is not None:
                return f"{best_hour:02d}:00"
        return None

    def _read_json(self) -> Dict:
        """Read JSON file (blocking - use in sync context only)"""
        try:
            with open(self.hourly_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self._ensure_file_exists()
            with open(self.hourly_file, 'r') as f:
                return json.load(f)

    async def _read_json_async(self) -> Dict:
        """Read JSON file (non-blocking - use in async context)"""
        def _do_read():
            try:
                with open(self.hourly_file, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                self._ensure_file_exists()
                with open(self.hourly_file, 'r') as f:
                    return json.load(f)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do_read)

    def _write_json(self, data: Dict):
        """REMOVED: This method caused blocking I/O - use _write_json_atomic instead"""
        raise RuntimeError(
            "DEPRECATED: _write_json() removed to prevent blocking I/O. "
            "Use _write_json_atomic() instead or call from executor."
        )

    async def _write_json_atomic(self, data: Dict):
        """Write JSON atomically using DataManager's thread-safe method"""
        if self.data_manager:
            await self.data_manager._atomic_write_json(self.hourly_file, data)
        else:
            # Fallback during init (no data_manager yet)
            def _do_write():
                temp_file = self.hourly_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2)
                temp_file.replace(self.hourly_file)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _do_write)
