"""ML Data Loader V3 - Load training data with corrected weather V10.0.0 @zara

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..core.core_helpers import SafeDateTimeUtil as dt_util

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

class MLDataLoaderV3:
    """Load and prepare training data with corrected weather

    ASYNC I/O: Uses async_add_executor_job for non-blocking file operations.
    """

    def __init__(self, data_dir: Path, hass: Optional["HomeAssistant"] = None):
        self.data_dir = data_dir
        self.hass = hass
        self.hourly_predictions_file = data_dir / "hourly_predictions.json"
        self.astronomy_cache_file = data_dir / "astronomy_cache.json"
        self.weather_corrected_file = data_dir / "weather_forecast_corrected.json"
        self.weather_precision_file = data_dir / "stats" / "weather_precision_daily.json"

    def _read_json_sync(self, file_path: Path) -> Dict[str, Any]:
        """Synchronous JSON file read (for executor) @zara"""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def _read_json_async(self, file_path: Path) -> Dict[str, Any]:
        """Asynchronous JSON file read using executor @zara"""
        if self.hass:

            return await self.hass.async_add_executor_job(self._read_json_sync, file_path)
        else:

            return self._read_json_sync(file_path)

    async def load_training_data(
        self,
        min_samples: int = 10,
        exclude_outliers: bool = True
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Load training records from hourly_predictions.json with corrected weather (ASYNC)

        Args:
            min_samples: Minimum number of samples required
            exclude_outliers: Exclude outlier production values

        Returns:
            Tuple of (records, count)
        """
        try:

            if not self.hourly_predictions_file.exists():
                _LOGGER.warning(f"hourly_predictions.json not found")
                return [], 0

            hourly_data = await self._read_json_async(self.hourly_predictions_file)

            predictions = hourly_data.get("predictions", [])
            _LOGGER.info(f"Loaded {len(predictions)} predictions from hourly_predictions.json")

            astronomy_cache = await self._load_astronomy_cache()

            precision_data = await self._load_precision_data()

            historical_cache = await self._load_historical_cache()

            valid_predictions = [
                p for p in predictions
                if p.get("actual_kwh") is not None
            ]

            _LOGGER.info(f"Found {len(valid_predictions)} predictions with actual production")

            frost_excluded_count = 0
            non_frost_predictions = []
            for p in valid_predictions:

                weather_actual = p.get("weather_actual", {})
                frost_detected = weather_actual.get("frost_detected")

                if frost_detected == "heavy_frost":
                    frost_excluded_count += 1
                    frost_score = weather_actual.get("frost_score", 'N/A')
                    frost_confidence = weather_actual.get("frost_confidence", 0)
                    _LOGGER.info(
                        f"Excluding sample {p.get('datetime')} - heavy frost detected "
                        f"(score: {frost_score}, confidence: {frost_confidence:.0%})"
                    )
                else:
                    non_frost_predictions.append(p)

            if frost_excluded_count > 0:
                _LOGGER.info(
                    f"Excluded {frost_excluded_count} frost-affected samples from training "
                    f"({len(non_frost_predictions)} clean samples remaining)"
                )

            valid_predictions = non_frost_predictions

            if len(valid_predictions) < min_samples:
                _LOGGER.warning(
                    f"Not enough training samples: {len(valid_predictions)} < {min_samples}"
                )
                return [], 0

            records = []
            for prediction in valid_predictions:
                record = self._enrich_prediction(
                    prediction,
                    astronomy_cache,
                    historical_cache,
                    precision_data
                )

                if record:
                    records.append(record)

            if exclude_outliers and len(records) > 10:
                records = self._exclude_outliers(records)

            _LOGGER.info(f"Prepared {len(records)} training records")
            return records, len(records)

        except Exception as e:
            _LOGGER.error(f"Error loading training data: {e}", exc_info=True)
            return [], 0

    async def _load_astronomy_cache(self) -> Dict[str, Any]:
        """Load astronomy cache (ASYNC) @zara"""
        if not self.astronomy_cache_file.exists():
            _LOGGER.warning("Astronomy cache not found")
            return {}

        try:

            return await self._read_json_async(self.astronomy_cache_file)
        except Exception as e:
            _LOGGER.error(f"Error loading astronomy cache: {e}")
            return {}

    async def _load_precision_data(self) -> Dict[str, Any]:
        """Load weather precision data for historical corrections (ASYNC) @zara"""
        if not self.weather_precision_file.exists():
            _LOGGER.debug("Weather precision file not found, using defaults")
            return self._get_default_precision()

        try:

            return await self._read_json_async(self.weather_precision_file)
        except Exception as e:
            _LOGGER.error(f"Error loading precision data: {e}")
            return self._get_default_precision()

    def _get_default_precision(self) -> Dict[str, Any]:
        """Get default precision (no corrections) @zara"""
        return {
            "rolling_averages": {
                "30_day": {
                    "correction_factors": {
                        "temperature": 1.0,
                        "lux": 1.0,
                        "wind": 1.0,
                        "humidity": 1.0,
                        "rain": 1.0,
                    }
                }
            }
        }

    async def _load_historical_cache(self) -> Dict[str, Dict[str, float]]:
        """Load historical production data for LAG features (ASYNC) @zara"""
        try:
            if not self.hourly_predictions_file.exists():
                _LOGGER.warning("hourly_predictions.json not found - cannot load lag features")
                return {"daily_productions": {}, "hourly_productions": {}}

            data = await self._read_json_async(self.hourly_predictions_file)

            predictions = data.get("predictions", [])

            daily_productions = {}
            hourly_productions = {}

            date_groups = {}
            for pred in predictions:
                target_date = pred.get("target_date")
                actual_kwh = pred.get("actual_kwh")
                target_hour = pred.get("target_hour")

                if not target_date or actual_kwh is None:
                    continue

                hour_key = f"{target_date}_{target_hour:02d}"
                hourly_productions[hour_key] = actual_kwh

                if target_date not in date_groups:
                    date_groups[target_date] = []
                date_groups[target_date].append(actual_kwh)

            for date, hourly_values in date_groups.items():
                daily_total = sum(hourly_values)
                daily_productions[date] = daily_total

            _LOGGER.info(
                f"Built historical cache: {len(daily_productions)} days, {len(hourly_productions)} hours"
            )

            return {
                "daily_productions": daily_productions,
                "hourly_productions": hourly_productions,
            }

        except Exception as e:
            _LOGGER.error(f"Error loading historical cache: {e}")
            return {"daily_productions": {}, "hourly_productions": {}}

    def _enrich_prediction(
        self,
        prediction: Dict[str, Any],
        astronomy_cache: Dict[str, Any],
        historical_cache: Dict[str, Dict[str, float]],
        precision_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Enrich prediction with corrected weather, astronomy, and LAG features

        Args:
            prediction: Raw prediction from hourly_predictions.json
            astronomy_cache: Astronomy data
            historical_cache: Historical production data
            precision_data: Weather precision/correction data

        Returns:
            Enriched record or None if incomplete
        """
        try:

            target_date = prediction.get("target_date")
            target_hour = prediction.get("target_hour")
            actual_kwh = prediction.get("actual_kwh")

            if not target_date or target_hour is None or actual_kwh is None:
                return None

            target_datetime = datetime.fromisoformat(f"{target_date}T{target_hour:02d}:00:00")

            target_day_of_year = target_datetime.timetuple().tm_yday
            target_month = target_datetime.month

            if target_month in [12, 1, 2]:
                target_season = "winter"
            elif target_month in [3, 4, 5]:
                target_season = "spring"
            elif target_month in [6, 7, 8]:
                target_season = "summer"
            else:
                target_season = "autumn"

            weather_corrected = self._get_corrected_weather_for_training(
                prediction,
                precision_data,
                target_date
            )

            if not weather_corrected:
                _LOGGER.debug(f"No corrected weather for {target_date} {target_hour}:00")
                return None

            astronomy_data = self._get_astronomy_for_hour(
                astronomy_cache,
                target_date,
                target_hour
            )

            if not astronomy_data:
                _LOGGER.debug(f"No astronomy data for {target_date} {target_hour}:00")
                return None

            yesterday_date = (target_datetime - timedelta(days=1)).date().isoformat()
            production_yesterday = historical_cache["daily_productions"].get(yesterday_date, 0.0)

            yesterday_hour_key = f"{yesterday_date}_{target_hour:02d}"
            production_same_hour_yesterday = historical_cache["hourly_productions"].get(
                yesterday_hour_key, 0.0
            )

            record = {

                "target_date": target_date,
                "target_hour": target_hour,
                "target_datetime": target_datetime.isoformat(),

                "target_day_of_year": target_day_of_year,
                "target_month": target_month,
                "target_season": target_season,

                "weather_corrected": weather_corrected,

                "astronomy": astronomy_data,

                "production_yesterday": production_yesterday,
                "production_same_hour_yesterday": production_same_hour_yesterday,

                "actual_kwh": actual_kwh
            }

            return record

        except Exception as e:
            _LOGGER.error(f"Error enriching prediction: {e}")
            return None

    def _get_corrected_weather_for_training(
        self,
        prediction: Dict[str, Any],
        precision_data: Dict[str, Any],
        target_date: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get corrected weather for training (applies historical corrections)

        For training data, we reconstruct what the corrected forecast WOULD have been
        by applying the learned corrections to the historical forecast.

        Args:
            prediction: Prediction with weather_corrected or weather_forecast data
            precision_data: Weather precision data
            target_date: Target date

        Returns:
            Corrected weather dictionary or None
        """
        try:

            if "weather_corrected" in prediction:
                weather_corrected = prediction.get("weather_corrected")
                if weather_corrected:
                    return weather_corrected

            weather_forecast = prediction.get("weather_forecast", {})

            if not weather_forecast:
                return None

            corrections = precision_data.get("rolling_averages", {}).get("30_day", {}).get(
                "correction_factors",
                {"temperature": 1.0, "lux": 1.0, "wind": 1.0, "humidity": 1.0, "rain": 1.0}
            )

            original_temp = weather_forecast.get("temperature")
            original_humidity = weather_forecast.get("humidity")
            original_wind = weather_forecast.get("wind_speed")
            original_rain = weather_forecast.get("precipitation", 0)

            original_clouds = weather_forecast.get("cloud_cover", 50)
            ghi = weather_forecast.get("ghi") or weather_forecast.get("solar_radiation_wm2")
            direct_radiation = weather_forecast.get("direct_radiation", 0)
            diffuse_radiation = weather_forecast.get("diffuse_radiation", 0)

            if ghi is None and (direct_radiation or diffuse_radiation):
                ghi = (direct_radiation or 0) + (diffuse_radiation or 0)

            if ghi and ghi > 0:
                original_lux = ghi * 100
            else:
                original_lux = 100000 * (1 - original_clouds / 100) + 5000

            corrected_temp = original_temp * corrections["temperature"] if original_temp else 15.0
            corrected_lux = original_lux * corrections["lux"]
            corrected_humidity = original_humidity * corrections["humidity"] if original_humidity else 70.0
            corrected_wind = original_wind * corrections["wind"] if original_wind else 3.0
            corrected_rain = original_rain * corrections["rain"]

            solar_radiation_wm2 = ghi if ghi and ghi > 0 else corrected_lux / 100

            return {
                "temperature": corrected_temp,
                "solar_radiation_wm2": solar_radiation_wm2,
                "lux": corrected_lux,
                "wind": corrected_wind,
                "humidity": corrected_humidity,
                "rain": corrected_rain,
                "clouds": original_clouds
            }

        except Exception as e:
            _LOGGER.error(f"Error getting corrected weather for training: {e}")
            return None

    def _get_astronomy_for_hour(
        self,
        astronomy_cache: Dict[str, Any],
        target_date: str,
        target_hour: int
    ) -> Optional[Dict[str, Any]]:
        """Get astronomy data for specific hour"""
        try:

            day_data = astronomy_cache.get("days", {}).get(target_date)
            if not day_data:
                return None

            hourly_data = day_data.get("hourly", {})
            hour_data = hourly_data.get(str(target_hour))

            if not hour_data:
                return None

            return {
                "sun_elevation_deg": hour_data.get("elevation_deg", -30.0),
                "theoretical_max_kwh": hour_data.get("theoretical_max_pv_kwh", 0.0),
                "clear_sky_radiation_wm2": hour_data.get("clear_sky_solar_radiation_wm2", 0.0)
            }

        except Exception as e:
            _LOGGER.error(f"Error getting astronomy for hour: {e}")
            return None

    def _exclude_outliers(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Exclude outlier production values using IQR method @zara"""
        try:
            productions = [r["actual_kwh"] for r in records]

            productions_sorted = sorted(productions)
            n = len(productions_sorted)

            if n < 4:

                _LOGGER.debug(f"Not enough data for IQR filtering ({n} records)")
                return records

            q1 = productions_sorted[n // 4]
            q3 = productions_sorted[3 * n // 4]
            iqr = max(q3 - q1, 0.001)

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            filtered = [
                r for r in records
                if lower_bound <= r["actual_kwh"] <= upper_bound
            ]

            excluded = len(records) - len(filtered)
            if excluded > 0:
                _LOGGER.info(f"Excluded {excluded} outliers (bounds: {lower_bound:.2f}-{upper_bound:.2f} kWh)")

            return filtered

        except Exception as e:
            _LOGGER.error(f"Error excluding outliers: {e}")
            return records

    async def load_prediction_features(
        self,
        target_datetime: datetime,
        weather_corrected: Dict[str, Any],
        astronomy: Dict[str, Any],
        historical_cache: Dict[str, Dict[str, float]]
    ) -> Optional[Dict[str, Any]]:
        """
        Load features for making a prediction (ASYNC)

        Args:
            target_datetime: Target datetime for prediction
            weather_corrected: Corrected weather forecast
            astronomy: Astronomy data
            historical_cache: Historical production data

        Returns:
            Feature record or None
        """
        try:
            target_date = target_datetime.date().isoformat()
            target_hour = target_datetime.hour
            target_day_of_year = target_datetime.timetuple().tm_yday
            target_month = target_datetime.month

            if target_month in [12, 1, 2]:
                target_season = "winter"
            elif target_month in [3, 4, 5]:
                target_season = "spring"
            elif target_month in [6, 7, 8]:
                target_season = "summer"
            else:
                target_season = "autumn"

            yesterday_date = (target_datetime - timedelta(days=1)).date().isoformat()
            production_yesterday = historical_cache["daily_productions"].get(yesterday_date, 0.0)

            yesterday_hour_key = f"{yesterday_date}_{target_hour:02d}"
            production_same_hour_yesterday = historical_cache["hourly_productions"].get(
                yesterday_hour_key, 0.0
            )

            return {
                "target_date": target_date,
                "target_hour": target_hour,
                "target_datetime": target_datetime.isoformat(),
                "target_day_of_year": target_day_of_year,
                "target_month": target_month,
                "target_season": target_season,
                "weather_corrected": weather_corrected,
                "astronomy": astronomy,
                "production_yesterday": production_yesterday,
                "production_same_hour_yesterday": production_same_hour_yesterday
            }

        except Exception as e:
            _LOGGER.error(f"Error loading prediction features: {e}")
            return None
