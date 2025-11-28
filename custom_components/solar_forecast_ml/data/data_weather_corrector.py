"""Weather Forecast Corrector - Applies Precision Corrections to Open-Meteo Data @zara

Open-Meteo is the PRIMARY weather data source.
Precision correction factors from local sensors are APPLIED to create
the corrected forecast (Regional API + Local Sensor Learning).

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
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from ..core.core_helpers import SafeDateTimeUtil as dt_util
from .data_io import DataManagerIO
from .data_open_meteo_client import OpenMeteoClient, get_default_weather

_LOGGER = logging.getLogger(__name__)

DEFAULT_WEATHER = {
    "temperature": 15.0,
    "solar_radiation_wm2": 150.0,
    "wind": 3.0,
    "humidity": 70.0,
    "rain": 0.0,
    "clouds": 50.0,
    "pressure": 1013.0,
    "source": "default_fallback",
    "confidence": 0.1,
}


class WeatherForecastCorrector(DataManagerIO):
    """Creates corrected weather forecasts by applying precision factors to Open-Meteo data.

    Architecture:
    - Open-Meteo provides raw regional forecast data
    - Local sensors provide actual measurements (hourly_weather_actual.json)
    - Precision tracking calculates correction factors (weather_precision_daily.json)
    - This class APPLIES those factors to create weather_forecast_corrected.json
    """

    def __init__(self, hass: HomeAssistant, data_manager):
        super().__init__(hass, data_manager.data_dir)

        self.data_manager = data_manager
        self.stats_dir = data_manager.data_dir / "stats"
        self.corrected_file = data_manager.weather_corrected_file
        self.precision_file = self.stats_dir / "weather_precision_daily.json"

        self._open_meteo_client: Optional[OpenMeteoClient] = None
        self._open_meteo_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._consecutive_failures: int = 0
        self._last_error: Optional[str] = None

        try:
            latitude = hass.config.latitude
            longitude = hass.config.longitude
            if latitude and longitude:
                cache_file = data_manager.data_dir / "data" / "open_meteo_cache.json"
                self._open_meteo_client = OpenMeteoClient(
                    latitude, longitude,
                    cache_file=cache_file
                )
                _LOGGER.info(
                    f"WeatherForecastCorrector initialized "
                    f"(lat={latitude:.4f}, lon={longitude:.4f}) - Applies precision corrections"
                )
            else:
                _LOGGER.warning("No coordinates in HA config - Open-Meteo disabled")
        except Exception as e:
            _LOGGER.warning(f"Could not initialize Open-Meteo client: {e}")

    async def async_init(self) -> bool:
        if self._open_meteo_client:
            await self._open_meteo_client.async_init()
            await self._load_open_meteo_cache()
        return True

    async def _load_open_meteo_cache(self) -> bool:
        if not self._open_meteo_client:
            return False

        try:
            cached_forecast = self._open_meteo_client._cache.get("hourly_forecast", [])
            if not cached_forecast:
                _LOGGER.debug("No Open-Meteo data in client cache")
                return False

            self._open_meteo_cache.clear()
            for entry in cached_forecast:
                date = entry.get("date")
                hour = entry.get("hour")

                if date and hour is not None:
                    if date not in self._open_meteo_cache:
                        self._open_meteo_cache[date] = {}

                    direct = entry.get("direct_radiation") or 0
                    diffuse = entry.get("diffuse_radiation") or 0

                    self._open_meteo_cache[date][hour] = {
                        "temperature": entry.get("temperature"),
                        "humidity": entry.get("humidity"),
                        "cloud_cover": entry.get("cloud_cover"),
                        "precipitation": entry.get("precipitation"),
                        "wind_speed": entry.get("wind_speed"),
                        "pressure": entry.get("pressure"),
                        "direct_radiation": direct,
                        "diffuse_radiation": diffuse,
                        "ghi": entry.get("ghi") or (direct + diffuse),
                    }

            if self._open_meteo_cache:
                _LOGGER.info(
                    f"Loaded Open-Meteo cache: {len(cached_forecast)} hours, "
                    f"{len(self._open_meteo_cache)} days"
                )
                return True

            return False

        except Exception as e:
            _LOGGER.warning(f"Error loading Open-Meteo cache: {e}")
            return False

    async def _fetch_open_meteo_forecast(self) -> bool:
        if not self._open_meteo_client:
            _LOGGER.debug("Open-Meteo client not available")
            return False

        try:
            forecast = await self._open_meteo_client.get_hourly_forecast(hours=72)
            if not forecast:
                _LOGGER.warning("Open-Meteo returned no forecast data")
                return False

            self._open_meteo_cache.clear()
            for entry in forecast:
                date = entry.get("date")
                hour = entry.get("hour")

                if date and hour is not None:
                    if date not in self._open_meteo_cache:
                        self._open_meteo_cache[date] = {}

                    direct = entry.get("direct_radiation") or 0
                    diffuse = entry.get("diffuse_radiation") or 0

                    self._open_meteo_cache[date][hour] = {
                        "temperature": entry.get("temperature"),
                        "humidity": entry.get("humidity"),
                        "cloud_cover": entry.get("cloud_cover"),
                        "precipitation": entry.get("precipitation"),
                        "wind_speed": entry.get("wind_speed"),
                        "pressure": entry.get("pressure"),
                        "direct_radiation": direct,
                        "diffuse_radiation": diffuse,
                        "ghi": entry.get("ghi") or (direct + diffuse),
                    }

            _LOGGER.info(
                f"Fetched Open-Meteo forecast: {len(forecast)} hours, "
                f"{len(self._open_meteo_cache)} days"
            )
            return True

        except Exception as e:
            _LOGGER.warning(f"Error fetching Open-Meteo forecast: {e}")
            return False

    def _get_weather_for_hour(self, date_str: str, hour: int) -> Dict[str, Any]:
        """Get RAW weather for hour from Open-Meteo cache (before correction) @zara"""
        if date_str in self._open_meteo_cache:
            hour_data = self._open_meteo_cache[date_str].get(hour)
            if hour_data:
                return {
                    "temperature": hour_data.get("temperature"),
                    "solar_radiation_wm2": hour_data.get("ghi") or 0,
                    "wind": hour_data.get("wind_speed"),
                    "humidity": hour_data.get("humidity"),
                    "rain": hour_data.get("precipitation") or 0,
                    "clouds": hour_data.get("cloud_cover"),
                    "pressure": hour_data.get("pressure"),
                    "direct_radiation": hour_data.get("direct_radiation"),
                    "diffuse_radiation": hour_data.get("diffuse_radiation"),
                    "source": "open-meteo-raw",
                    "confidence": 0.95,
                }

        _LOGGER.debug(f"No Open-Meteo data for {date_str} {hour:02d}:00, using defaults")
        return self._get_default_weather_for_hour(hour, date_str)

    def _get_default_weather_for_hour(self, hour: int, date_str: str) -> Dict[str, Any]:
        import math
        hour_offset = (hour - 14) / 24.0 * 2 * math.pi
        temp_amplitude = 5.0
        temp = DEFAULT_WEATHER["temperature"] + temp_amplitude * math.cos(hour_offset)

        default = DEFAULT_WEATHER.copy()
        default["temperature"] = round(temp, 1)
        default["source"] = "default_fallback"
        default["confidence"] = 0.1

        return default

    async def _load_correction_factors(self) -> Dict[str, Any]:
        """Load precision correction factors from weather_precision_daily.json"""
        default_factors = {
            "temperature": 0.0,
            "solar_radiation_wm2": 1.0,
            "clouds": 1.0,
            "wind": 1.0,
            "humidity": 1.0,
            "rain": 1.0,
            "pressure": 0.0,
        }
        default_confidence = {k: 0.0 for k in default_factors}

        try:
            if not self.precision_file.exists():
                _LOGGER.debug("No precision file found - using default factors (no correction)")
                return {"factors": default_factors, "confidence": default_confidence, "sample_days": 0}

            precision_data = await self._read_json_file(self.precision_file, None)
            if not precision_data:
                return {"factors": default_factors, "confidence": default_confidence, "sample_days": 0}

            rolling = precision_data.get("rolling_averages", {})
            factors = rolling.get("correction_factors", default_factors)
            confidence = rolling.get("confidence", default_confidence)
            sample_days = rolling.get("sample_days", 0)

            for key in default_factors:
                if key not in factors:
                    factors[key] = default_factors[key]
                if key not in confidence:
                    confidence[key] = 0.0

            return {"factors": factors, "confidence": confidence, "sample_days": sample_days}

        except Exception as e:
            _LOGGER.warning(f"Error loading correction factors: {e}")
            return {"factors": default_factors, "confidence": default_confidence, "sample_days": 0}

    def _apply_correction(self, raw_value: float, factor: float, field_type: str) -> float:
        """Apply correction factor to a raw value based on field type"""
        if raw_value is None:
            return None

        if field_type in ("temperature", "pressure"):
            return round(raw_value + factor, 1)
        else:
            corrected = raw_value * factor
            if field_type == "clouds":
                corrected = max(0.0, min(100.0, corrected))
            elif field_type in ("solar_radiation_wm2", "wind", "humidity", "rain"):
                corrected = max(0.0, corrected)
            return round(corrected, 1)

    async def create_corrected_forecast(self, min_confidence: float = 0.0) -> bool:
        try:
            correction_data = await self._load_correction_factors()
            factors = correction_data["factors"]
            confidence = correction_data["confidence"]
            sample_days = correction_data["sample_days"]

            if sample_days > 0:
                _LOGGER.info(
                    f"Creating corrected forecast - applying {sample_days}-day precision factors "
                    f"(solar_rad={factors.get('solar_radiation_wm2', 1.0):.3f}, "
                    f"clouds={factors.get('clouds', 1.0):.3f})"
                )
            else:
                _LOGGER.info("Creating forecast - no precision data yet (using raw Open-Meteo values)")

            open_meteo_available = await self._fetch_open_meteo_forecast()
            if not open_meteo_available:
                _LOGGER.warning("Open-Meteo unavailable - using cached data or defaults")

            astronomy_cache_file = self.stats_dir / "astronomy_cache.json"
            astronomy_data = {}
            if astronomy_cache_file.exists():
                try:
                    astro_cache = await self._read_json_file(astronomy_cache_file, None)
                    if astro_cache:
                        astronomy_data = astro_cache.get("days", {})
                except Exception as e:
                    _LOGGER.warning(f"Could not load astronomy cache: {e}")

            forecast_by_date: Dict[str, Dict[str, Dict[str, Any]]] = {}

            for date_str in sorted(self._open_meteo_cache.keys()):
                day_data = self._open_meteo_cache[date_str]

                for hour in range(24):
                    weather = self._get_weather_for_hour(date_str, hour)

                    astronomy = {}
                    if date_str in astronomy_data:
                        hour_astro = astronomy_data[date_str].get("hourly", {}).get(str(hour), {})
                        astronomy = {
                            "sun_elevation": hour_astro.get("elevation_deg"),
                            "azimuth": hour_astro.get("azimuth_deg"),
                            "clear_sky_radiation": hour_astro.get("clear_sky_solar_radiation_wm2"),
                            "theoretical_max_kwh": hour_astro.get("theoretical_max_pv_kwh"),
                        }

                    if date_str not in forecast_by_date:
                        forecast_by_date[date_str] = {}

                    raw_solar = weather.get("solar_radiation_wm2") or 0
                    raw_clouds = weather.get("clouds")
                    raw_temp = weather.get("temperature")
                    raw_wind = weather.get("wind")
                    raw_humidity = weather.get("humidity")
                    raw_rain = weather.get("rain") or 0
                    raw_pressure = weather.get("pressure")

                    forecast_by_date[date_str][str(hour)] = {
                        "temperature": self._apply_correction(raw_temp, factors.get("temperature", 0.0), "temperature"),
                        "solar_radiation_wm2": self._apply_correction(raw_solar, factors.get("solar_radiation_wm2", 1.0), "solar_radiation_wm2"),
                        "wind": self._apply_correction(raw_wind, factors.get("wind", 1.0), "wind"),
                        "humidity": self._apply_correction(raw_humidity, factors.get("humidity", 1.0), "humidity"),
                        "rain": self._apply_correction(raw_rain, factors.get("rain", 1.0), "rain"),
                        "clouds": self._apply_correction(raw_clouds, factors.get("clouds", 1.0), "clouds"),
                        "pressure": self._apply_correction(raw_pressure, factors.get("pressure", 0.0), "pressure"),
                        "direct_radiation": weather.get("direct_radiation"),
                        "diffuse_radiation": weather.get("diffuse_radiation"),
                    }

            rb_correction = await self._get_preserved_rb_correction()

            corrected_data = {
                "version": "4.1",
                "forecast": forecast_by_date,
                "metadata": {
                    "created": dt_util.now().isoformat(),
                    "source": "open-meteo-corrected",
                    "mode": "precision_corrected",
                    "hours_forecast": sum(len(h) for h in forecast_by_date.values()),
                    "days_forecast": len(forecast_by_date),
                    "corrections_applied": {
                        "solar_radiation_wm2": factors.get("solar_radiation_wm2", 1.0),
                        "clouds": factors.get("clouds", 1.0),
                        "temperature": factors.get("temperature", 0.0),
                        "wind": factors.get("wind", 1.0),
                        "humidity": factors.get("humidity", 1.0),
                        "rain": factors.get("rain", 1.0),
                    },
                    "confidence_scores": confidence,
                    "precision_sample_days": sample_days,
                    "rb_overall_correction": rb_correction,
                    "self_healing": {
                        "open_meteo_status": "ok" if open_meteo_available else "using_cache",
                        "consecutive_failures": self._consecutive_failures,
                        "last_error": self._last_error,
                    },
                }
            }

            await self._atomic_write_json(self.corrected_file, corrected_data)

            total_hours = sum(len(h) for h in forecast_by_date.values())
            _LOGGER.info(
                f"Created corrected forecast: {total_hours} hours, {sample_days} days precision learning"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Error creating forecast: {e}", exc_info=True)
            return False

    async def _get_preserved_rb_correction(self) -> Dict[str, Any]:
        default_rb = {
            "factor": 1.0,
            "confidence": 0.0,
            "sample_days": 0,
            "last_updated": dt_util.now().isoformat(),
            "note": "Installation-specific correction factor"
        }

        if self.corrected_file.exists():
            try:
                existing = await self._read_json_file(self.corrected_file, None)
                if existing and "metadata" in existing:
                    existing_rb = existing["metadata"].get("rb_overall_correction", {})
                    if existing_rb and existing_rb.get("factor", 1.0) != 1.0:
                        return {
                            "factor": existing_rb.get("factor", 1.0),
                            "confidence": existing_rb.get("confidence", 0.0),
                            "sample_days": existing_rb.get("sample_days", 0),
                            "last_updated": existing_rb.get("last_updated", dt_util.now().isoformat()),
                            "note": existing_rb.get("note", "Installation-specific correction factor")
                        }
            except Exception:
                pass

        return default_rb

    async def get_corrected_weather_for_hour(self, target_datetime: datetime) -> Optional[Dict[str, Any]]:
        date_str = target_datetime.date().isoformat()
        hour = target_datetime.hour

        try:
            if self.corrected_file.exists():
                data = await self._read_json_file(self.corrected_file, None)
                if data:
                    forecast = data.get("forecast", {})
                    if date_str in forecast and str(hour) in forecast[date_str]:
                        result = forecast[date_str][str(hour)]
                        result["_source"] = "corrected_forecast"
                        result["_confidence"] = 0.95
                        return result

            weather = self._get_weather_for_hour(date_str, hour)
            weather["_source"] = "open_meteo_direct"
            return weather

        except Exception as e:
            _LOGGER.error(f"Error getting weather for hour: {e}")
            return self._get_default_weather_for_hour(hour, date_str)

    async def update_rb_correction(self, date_str: str) -> bool:
        try:
            _LOGGER.debug(f"Updating RB correction for {date_str}")

            predictions_file = self.stats_dir / "hourly_predictions.json"
            if not predictions_file.exists():
                return False

            predictions_data = await self._read_json_file(predictions_file, None)
            if not predictions_data:
                return False

            predictions = predictions_data.get("predictions", [])

            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            cutoff_date = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")

            daily_data = {}
            for pred in predictions:
                pred_date = pred.get("target_date")
                if not pred_date or pred_date < cutoff_date or pred_date > date_str:
                    continue

                actual_kwh = pred.get("actual_kwh")
                prediction_kwh = pred.get("prediction_kwh")
                ml_contrib = pred.get("ml_contribution_percent", 0)

                if actual_kwh is None or prediction_kwh is None or prediction_kwh <= 0:
                    continue

                rb_weight = 1.0 - (ml_contrib / 100.0)
                if rb_weight < 0.25:
                    continue

                if pred_date not in daily_data:
                    daily_data[pred_date] = {"rb_total": 0.0, "actual_total": 0.0, "hours": 0}

                daily_data[pred_date]["rb_total"] += prediction_kwh
                daily_data[pred_date]["actual_total"] += actual_kwh
                daily_data[pred_date]["hours"] += 1

            daily_factors = []
            for date, data in daily_data.items():
                if data["hours"] == 0 or data["rb_total"] == 0:
                    continue
                factor = data["actual_total"] / data["rb_total"]
                daily_factors.append(factor)

            if len(daily_factors) < 1:
                return False

            import statistics
            if len(daily_factors) >= 5:
                mean = statistics.mean(daily_factors)
                stdev = statistics.stdev(daily_factors)
                daily_factors = [f for f in daily_factors if abs(f - mean) <= 3 * stdev]

            if not daily_factors:
                return False

            avg_factor = sum(daily_factors) / len(daily_factors)
            sample_days = len(daily_factors)

            if sample_days == 1:
                confidence = 0.2
            elif sample_days < 3:
                confidence = 0.3
            elif sample_days < 7:
                confidence = 0.4
            elif sample_days < 30:
                confidence = 0.7
            else:
                confidence = 0.9

            if not self.corrected_file.exists():
                return False

            corrected_data = await self._read_json_file(self.corrected_file, None)
            if not corrected_data:
                return False

            if "metadata" not in corrected_data:
                corrected_data["metadata"] = {}

            corrected_data["metadata"]["rb_overall_correction"] = {
                "factor": round(avg_factor, 3),
                "confidence": round(confidence, 2),
                "sample_days": sample_days,
                "last_updated": dt_util.now().isoformat(),
                "note": "Installation-specific correction factor"
            }

            await self._atomic_write_json(self.corrected_file, corrected_data)

            _LOGGER.info(
                f"Updated RB correction: factor={avg_factor:.3f}, "
                f"confidence={confidence:.1f}, samples={sample_days}"
            )
            return True

        except Exception as e:
            _LOGGER.error(f"Error updating RB correction: {e}", exc_info=True)
            return False

    async def get_correction_summary(self) -> Dict[str, Any]:
        try:
            if self.corrected_file.exists():
                data = await self._read_json_file(self.corrected_file, None)
                if data and "metadata" in data:
                    return data["metadata"]
            return {}
        except Exception as e:
            _LOGGER.error(f"Error getting correction summary: {e}")
            return {}
