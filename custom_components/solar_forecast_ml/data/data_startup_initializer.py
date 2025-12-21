"""Startup Initializer - Guarantees all critical JSON files exist BEFORE async operations V12.2.0 @zara

IMPORTANT: All JSON structures created here MUST match the MASTER files in production!
Master location: /Volumes/config/solar_forecast_ml/
Do NOT add fields that don't exist in the master!

REFACTORED V12.2.0: This module now ONLY creates the 5 critical pre-async files:
1. Directory structure
2. open_meteo_cache.json
3. astronomy_cache.json
4. weather_forecast_corrected.json
5. daily_forecasts.json

All other files are created by DataSchemaValidator (async).

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
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

_LOGGER = logging.getLogger(__name__)


class StartupInitializer:
    """Synchronous initializer - guarantees critical files exist before async startup

    REFACTORED V12.2.0: Only handles 5 critical pre-async files.
    All other files are handled by DataSchemaValidator after async startup.
    """

    def __init__(self, data_dir: Path, config: Dict[str, Any]):
        """Initialize startup initializer @zara"""
        self.data_dir = Path(data_dir)
        self.config = config

        self.latitude = config.get("latitude", 52.52)
        self.longitude = config.get("longitude", 13.40)
        self.solar_capacity_kwp = config.get("solar_capacity", 2.0)
        self.timezone_str = config.get("timezone", "Europe/Berlin")

        # Parse timezone for datetime operations
        try:
            self.timezone = ZoneInfo(self.timezone_str)
        except Exception:
            _LOGGER.warning(f"Invalid timezone '{self.timezone_str}', falling back to Europe/Berlin")
            self.timezone = ZoneInfo("Europe/Berlin")
            self.timezone_str = "Europe/Berlin"

    def _make_tz_aware_iso(self, date_obj, hour: int, minute: int = 0) -> str:
        """Create timezone-aware ISO string for astronomy cache @zara

        CRITICAL: All datetime strings in astronomy_cache.json MUST include timezone
        to avoid 'can't compare offset-naive and offset-aware datetimes' errors.

        Args:
            date_obj: date object
            hour: hour (0-23)
            minute: minute (0-59)

        Returns:
            ISO string with timezone, e.g. "2025-12-12T06:00:00+01:00"
        """
        dt = datetime.combine(date_obj, time(hour, minute), tzinfo=self.timezone)
        return dt.isoformat()

    def initialize_all(self) -> bool:
        """Initialize ONLY critical JSON files synchronously @zara

        CRITICAL: This method MUST run BEFORE any async operations to prevent
        FileNotFoundError race conditions during first startup.

        REFACTORED V12.2.0: Only creates files that are needed BEFORE async startup.
        All other files are created/validated by DataSchemaValidator (async).

        Order matters:
        1. Create ALL directories first (required for any file operation)
        2. Create open_meteo_cache (first API fetch depends on it)
        3. Create astronomy_cache (first calculation depends on it)
        4. Create weather_forecast_corrected (derived from open_meteo)
        5. Create daily_forecasts (sensors access this immediately)

        All other files are handled by DataSchemaValidator after async startup.
        """
        _LOGGER.info("=" * 80)
        _LOGGER.info("STARTUP INITIALIZER - Creating critical pre-async files")
        _LOGGER.info("=" * 80)

        success = True

        # STEP 1: Create ALL required directories FIRST
        if not self._ensure_directory_structure():
            _LOGGER.error("Failed to create directory structure")
            success = False
        else:
            _LOGGER.info("Directory structure - Ready")

        # STEP 2: open_meteo_cache FIRST - API fetch depends on it!
        if not self._ensure_open_meteo_cache():
            _LOGGER.warning("open_meteo_cache.json - Created empty (will be filled by API)")
        else:
            _LOGGER.info("open_meteo_cache.json - Ready")

        # STEP 3: astronomy_cache (first calculation depends on it)
        if not self._ensure_astronomy_cache():
            _LOGGER.error("Failed to create astronomy_cache.json")
            success = False
        else:
            _LOGGER.info("astronomy_cache.json - Ready")

        # STEP 4: weather_forecast_corrected (derived from open_meteo)
        if not self._ensure_weather_forecast_corrected():
            _LOGGER.error("Failed to create weather_forecast_corrected.json")
            success = False
        else:
            _LOGGER.info("weather_forecast_corrected.json - Ready")

        # STEP 5: daily_forecasts.json - CRITICAL for sensors
        if not self._ensure_daily_forecasts():
            _LOGGER.error("Failed to create daily_forecasts.json")
            success = False
        else:
            _LOGGER.info("daily_forecasts.json - Ready")

        _LOGGER.info("=" * 80)
        if success:
            _LOGGER.info("STARTUP INITIALIZER - Critical files ready (5 files)")
            _LOGGER.info("Remaining files will be created by DataSchemaValidator")
        else:
            _LOGGER.error("STARTUP INITIALIZER - Some critical files failed")
        _LOGGER.info("=" * 80)

        return success

    def _ensure_directory_structure(self) -> bool:
        """Create ALL required directories synchronously @zara

        CRITICAL: Must run FIRST before any file operations to prevent
        FileNotFoundError when atomic writes try to create temp files.
        """
        required_dirs = [
            self.data_dir,
            self.data_dir / "data",
            self.data_dir / "stats",
            self.data_dir / "ml",
            self.data_dir / "physics",
            self.data_dir / "logs",
            self.data_dir / "backups",
            self.data_dir / "backups" / "auto",
        ]

        try:
            for directory in required_dirs:
                directory.mkdir(parents=True, exist_ok=True)
                _LOGGER.debug(f"Ensured directory exists: {directory}")

            _LOGGER.info(f"Created/verified {len(required_dirs)} directories")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to create directory structure: {e}", exc_info=True)
            return False

    def _ensure_open_meteo_cache(self) -> bool:
        """Create/validate open_meteo_cache.json matching MASTER structure @zara

        MASTER structure (from /Volumes/config/solar_forecast_ml/data/open_meteo_cache.json):
        - version: "2.0"
        - metadata: {fetched_at, latitude, longitude, hours_cached, days_cached, mode}
        - forecast: {date: {hour: {temperature, humidity, cloud_cover, precipitation, wind_speed,
                    pressure, direct_radiation, diffuse_radiation, ghi, global_tilted_irradiance}}}

        Returns True if cache exists and has data, False if empty/created new.
        """
        cache_file = self.data_dir / "data" / "open_meteo_cache.json"

        # Check if exists and has data
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                forecast = data.get("forecast", {})
                if forecast and len(forecast) > 0:
                    hours_cached = sum(len(h) for h in forecast.values())
                    _LOGGER.debug(f"open_meteo_cache.json exists with {hours_cached} hours")
                    return True
                else:
                    _LOGGER.info("open_meteo_cache.json exists but is empty - will create baseline")
            except Exception as e:
                _LOGGER.warning(f"Could not read open_meteo_cache.json: {e}")

        # Create new or replace empty cache
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            from datetime import date

            today = date.today()
            forecast_data = {}

            # Create 3 days of baseline forecast data
            for i in range(3):
                target_date = today + timedelta(days=i)
                date_str = target_date.isoformat()
                forecast_data[date_str] = {}

                for hour in range(24):
                    # Approximated baseline values
                    if 6 <= hour <= 18:
                        elevation_approx = max(0, 60 * (1 - abs(hour - 12) / 6))
                        clear_sky_ghi = max(0, 1000 * (elevation_approx / 60) ** 1.5)
                        # Approximate direct/diffuse split (70/30 on clear day)
                        direct_rad = int(clear_sky_ghi * 0.7 * 0.5)  # 50% for baseline clouds
                        diffuse_rad = int(clear_sky_ghi * 0.3 * 0.8)  # Less affected by clouds
                        ghi = int(clear_sky_ghi * 0.5)  # 50% baseline
                    else:
                        direct_rad = 0
                        diffuse_rad = 0
                        ghi = 0

                    # MASTER structure for hourly data
                    forecast_data[date_str][str(hour)] = {
                        "temperature": 10.0,
                        "humidity": 70.0,
                        "cloud_cover": 50.0,
                        "precipitation": 0.0,
                        "wind_speed": 3.0,
                        "pressure": 1013.0,
                        "direct_radiation": direct_rad,
                        "diffuse_radiation": diffuse_rad,
                        "ghi": ghi,
                        "global_tilted_irradiance": ghi,  # Simplified: same as GHI
                    }

            # MASTER structure for open_meteo_cache.json
            cache_data = {
                "version": "2.0",
                "metadata": {
                    "fetched_at": datetime.now().isoformat(),
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "hours_cached": sum(len(h) for h in forecast_data.values()),
                    "days_cached": len(forecast_data),
                    "mode": "direct_radiation",
                },
                "forecast": forecast_data,
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            _LOGGER.info(
                f"Created open_meteo_cache.json (MASTER structure) with {len(forecast_data)} days baseline - "
                f"API will update with real data"
            )
            return False  # Return False to indicate it's baseline, not real data

        except Exception as e:
            _LOGGER.error(f"Failed to create open_meteo_cache.json: {e}", exc_info=True)
            return False

    def _ensure_astronomy_cache(self) -> bool:
        """Create minimal astronomy_cache.json matching MASTER structure @zara

        MASTER structure (from /Volumes/config/solar_forecast_ml/stats/astronomy_cache.json):
        - version: "1.0"
        - last_updated, location, pv_system, cache_info, days
        - days[date].hourly[hour] has: elevation_deg, azimuth_deg, clear_sky_solar_radiation_wm2,
          theoretical_max_pv_kwh, hours_since_solar_noon, day_progress_ratio
        """
        cache_file = self.data_dir / "stats" / "astronomy_cache.json"

        if cache_file.exists():
            _LOGGER.debug("astronomy_cache.json already exists, skipping")
            return True

        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            from datetime import date

            today = date.today()
            days_data = {}

            for i in range(7):
                target_date = today + timedelta(days=i)
                date_str = target_date.isoformat()

                # Build hourly data matching MASTER structure
                hourly_data = {}
                for hour in range(24):
                    if 6 <= hour <= 18:
                        # Approximated values for startup
                        elevation_approx = max(0, 60 * (1 - abs(hour - 12) / 6))
                        azimuth_approx = 90 + (hour - 6) * 15  # ~90° at 6h to ~270° at 18h
                        clear_sky_rad = max(0, 1000 * (elevation_approx / 60) ** 1.5)
                        theoretical_max = (clear_sky_rad / 1000) * self.solar_capacity_kwp
                        hours_since_noon = hour - 12
                        day_progress = (hour - 6) / 12.0  # 0 at sunrise, 1 at sunset
                    else:
                        elevation_approx = 0
                        azimuth_approx = 0
                        clear_sky_rad = 0
                        theoretical_max = 0
                        hours_since_noon = hour - 12
                        day_progress = 0 if hour < 6 else 1.0

                    # MASTER structure for hourly data
                    hourly_data[str(hour)] = {
                        "elevation_deg": round(elevation_approx, 1),
                        "azimuth_deg": round(azimuth_approx, 1),
                        "clear_sky_solar_radiation_wm2": round(clear_sky_rad, 0),
                        "theoretical_max_pv_kwh": round(theoretical_max, 3),
                        "hours_since_solar_noon": hours_since_noon,
                        "day_progress_ratio": round(day_progress, 3),
                    }

                # MASTER structure for day entry
                # CRITICAL: All datetime strings MUST include timezone to avoid
                # "can't compare offset-naive and offset-aware datetimes" errors
                days_data[date_str] = {
                    "sunrise_local": self._make_tz_aware_iso(target_date, 6, 0),
                    "sunset_local": self._make_tz_aware_iso(target_date, 18, 0),
                    "solar_noon_local": self._make_tz_aware_iso(target_date, 12, 0),
                    "production_window_start": self._make_tz_aware_iso(target_date, 6, 0),
                    "production_window_end": self._make_tz_aware_iso(target_date, 18, 0),
                    "daylight_hours": 12.0,
                    "hourly": hourly_data,
                }

            # MASTER structure for astronomy_cache.json
            cache_data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "location": {
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "elevation_m": 0,
                    "timezone": self.timezone_str,
                },
                "pv_system": {
                    "installed_capacity_kwp": self.solar_capacity_kwp,
                },
                "cache_info": {
                    "total_days": len(days_data),
                    "days_back": 0,
                    "days_ahead": 7,
                    "date_range_start": today.isoformat(),
                    "date_range_end": (today + timedelta(days=6)).isoformat(),
                    "success_count": len(days_data),
                    "error_count": 0,
                },
                "days": days_data,
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            _LOGGER.info(f"Created astronomy_cache.json (MASTER structure) with 7 days")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to create astronomy_cache.json: {e}", exc_info=True)
            return False

    def _ensure_weather_forecast_corrected(self) -> bool:
        """Create weather_forecast_corrected.json matching MASTER structure @zara

        MASTER structure (from /Volumes/config/solar_forecast_ml/stats/weather_forecast_corrected.json):
        - version: "4.1"
        - forecast: {date: {hour: {temperature, solar_radiation_wm2, wind, humidity, rain, clouds, pressure, direct_radiation, diffuse_radiation}}}
        - metadata: {created, source, mode, hours_forecast, days_forecast, corrections_applied, confidence_scores, precision_sample_days, rb_overall_correction, self_healing}
        """
        corrected_file = self.data_dir / "stats" / "weather_forecast_corrected.json"

        if corrected_file.exists():
            _LOGGER.debug("weather_forecast_corrected.json already exists, skipping")
            return True

        try:
            corrected_file.parent.mkdir(parents=True, exist_ok=True)

            # PRIORITY 1: Try loading from open_meteo_cache.json (synchronous read)
            open_meteo_file = self.data_dir / "data" / "open_meteo_cache.json"
            if open_meteo_file.exists():
                try:
                    with open(open_meteo_file, "r") as f:
                        om_data = json.load(f)

                    forecast_data = om_data.get("forecast", {})
                    if forecast_data and len(forecast_data) > 0:
                        _LOGGER.info("Found Open-Meteo cache - using it for initial forecast")
                        return self._create_forecast_from_open_meteo(corrected_file, forecast_data)
                except Exception as e:
                    _LOGGER.warning(f"Could not load Open-Meteo cache: {e}")

            # PRIORITY 2: Fallback to baseline values (only if Open-Meteo unavailable)
            _LOGGER.warning("Open-Meteo cache not available - creating baseline forecast as fallback")
            return self._create_baseline_forecast(corrected_file)

        except Exception as e:
            _LOGGER.error(f"Failed to create weather_forecast_corrected.json: {e}", exc_info=True)
            return False

    def _create_forecast_from_open_meteo(self, corrected_file: Path, om_forecast: dict) -> bool:
        """Create corrected forecast from Open-Meteo data matching MASTER structure @zara"""
        try:
            forecast_by_date = {}

            for date_str, hours in om_forecast.items():
                forecast_by_date[date_str] = {}

                for hour_str, hour_data in hours.items():
                    # MASTER structure for hourly forecast data
                    forecast_by_date[date_str][hour_str] = {
                        "temperature": hour_data.get("temperature", 10.0),
                        "solar_radiation_wm2": hour_data.get("ghi", 0),
                        "wind": hour_data.get("wind_speed", 3.0),
                        "humidity": hour_data.get("humidity", 70.0),
                        "rain": hour_data.get("precipitation", 0.0),
                        "clouds": hour_data.get("cloud_cover", 50.0),
                        "pressure": hour_data.get("pressure", 1013.0),
                        "direct_radiation": hour_data.get("direct_radiation"),
                        "diffuse_radiation": hour_data.get("diffuse_radiation"),
                    }

            # MASTER structure for weather_forecast_corrected.json
            corrected_data = {
                "version": "4.1",
                "forecast": forecast_by_date,
                "metadata": self._create_master_metadata(
                    source="open-meteo-direct",
                    hours_forecast=sum(len(h) for h in forecast_by_date.values()),
                    days_forecast=len(forecast_by_date),
                ),
            }

            with open(corrected_file, "w", encoding="utf-8") as f:
                json.dump(corrected_data, f, indent=2, ensure_ascii=False)

            _LOGGER.info(
                f"Created weather_forecast_corrected.json (MASTER structure) from Open-Meteo: "
                f"{len(forecast_by_date)} days, {sum(len(h) for h in forecast_by_date.values())} hours"
            )
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to create forecast from Open-Meteo: {e}", exc_info=True)
            return False

    def _create_baseline_forecast(self, corrected_file: Path) -> bool:
        """FALLBACK ONLY: Create baseline forecast matching MASTER structure @zara"""
        try:
            from datetime import date

            today = date.today()
            forecast_by_date = {}

            for i in range(3):
                target_date = today + timedelta(days=i)
                date_str = target_date.isoformat()
                forecast_by_date[date_str] = {}

                for hour in range(24):
                    if 6 <= hour <= 18:
                        elevation_approx = max(0, 60 * (1 - abs(hour - 12) / 6))
                        clear_sky_rad = max(0, 1000 * (elevation_approx / 60) ** 1.5)
                    else:
                        clear_sky_rad = 0

                    baseline_radiation = int(clear_sky_rad * 0.5)

                    # MASTER structure for hourly forecast data
                    forecast_by_date[date_str][str(hour)] = {
                        "temperature": 10.0,
                        "solar_radiation_wm2": baseline_radiation,
                        "wind": 5.0,
                        "humidity": 60.0,
                        "rain": 0.0,
                        "clouds": 50.0,
                        "pressure": 1013.0,
                        "direct_radiation": None,
                        "diffuse_radiation": None,
                    }

            # MASTER structure for weather_forecast_corrected.json
            corrected_data = {
                "version": "4.1",
                "forecast": forecast_by_date,
                "metadata": self._create_master_metadata(
                    source="baseline_fallback",
                    hours_forecast=sum(len(h) for h in forecast_by_date.values()),
                    days_forecast=len(forecast_by_date),
                ),
            }

            with open(corrected_file, "w", encoding="utf-8") as f:
                json.dump(corrected_data, f, indent=2, ensure_ascii=False)

            _LOGGER.info("Created weather_forecast_corrected.json (MASTER structure) - baseline fallback")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to create baseline forecast: {e}", exc_info=True)
            return False

    def _create_master_metadata(self, source: str, hours_forecast: int, days_forecast: int) -> Dict[str, Any]:
        """Create metadata matching MASTER structure for weather_forecast_corrected.json @zara

        MASTER metadata keys: created, source, mode, hours_forecast, days_forecast,
        corrections_applied, confidence_scores, precision_sample_days, rb_overall_correction, self_healing
        """
        return {
            "created": datetime.now().isoformat(),
            "source": source,
            "mode": "direct_radiation",
            "hours_forecast": hours_forecast,
            "days_forecast": days_forecast,
            # MASTER: corrections_applied
            "corrections_applied": {
                "solar_radiation_wm2": 1.0,
                "clouds": 1.0,
                "temperature": 0.0,
                "wind": 1.0,
                "humidity": 1.0,
                "rain": 1.0,
            },
            # MASTER: confidence_scores
            "confidence_scores": {
                "temperature": 0.0,
                "solar_radiation_wm2": 0.0,
                "clouds": 0.0,
                "humidity": 0.0,
                "wind": 0.0,
                "rain": 0.0,
                "pressure": 0.0,
            },
            # MASTER: precision_sample_days
            "precision_sample_days": 0,
            # MASTER: rb_overall_correction
            "rb_overall_correction": {
                "factor": 1.0,
                "confidence": 0.0,
                "sample_days": 0,
                "last_updated": datetime.now().isoformat(),
                "note": "Installation-specific correction factor",
            },
            # MASTER: self_healing
            "self_healing": {
                "open_meteo_status": "ok",
                "consecutive_failures": 0,
                "last_error": None,
            },
        }

    def _ensure_daily_forecasts(self) -> bool:
        """Create daily_forecasts.json matching MASTER structure @zara

        CRITICAL: This file is accessed by many sensors and handlers.
        Must exist before async operations start.

        Location: stats/daily_forecasts.json
        """
        forecasts_file = self.data_dir / "stats" / "daily_forecasts.json"

        if forecasts_file.exists():
            _LOGGER.debug("daily_forecasts.json already exists, skipping")
            return True

        try:
            forecasts_file.parent.mkdir(parents=True, exist_ok=True)

            from datetime import date
            today = date.today()
            today_str = today.isoformat()

            # MASTER structure for daily_forecasts.json
            forecasts_data = {
                "version": "3.0.0",
                "today": {
                    "date": today_str,
                    "forecast_day": {
                        "prediction_kwh": None,
                        "prediction_kwh_raw": None,
                        "safeguard_applied": False,
                        "safeguard_reduction_kwh": 0.0,
                        "locked": False,
                        "locked_at": None,
                        "source": None,
                    },
                    "forecast_tomorrow": {
                        "date": None,
                        "prediction_kwh": None,
                        "locked": False,
                        "locked_at": None,
                        "source": None,
                        "updates": [],
                    },
                    "forecast_day_after_tomorrow": {
                        "date": None,
                        "prediction_kwh": None,
                        "locked": False,
                        "next_update": None,
                        "source": None,
                        "updates": [],
                    },
                    "forecast_best_hour": {
                        "hour": None,
                        "prediction_kwh": None,
                        "locked": False,
                        "locked_at": None,
                        "source": None,
                    },
                    "actual_best_hour": {
                        "hour": None,
                        "actual_kwh": None,
                        "saved_at": None,
                    },
                    "forecast_next_hour": {
                        "period": None,
                        "prediction_kwh": None,
                        "updated_at": None,
                        "source": None,
                    },
                    "production_time": {
                        "active": False,
                        "duration_seconds": 0,
                        "start_time": None,
                        "end_time": None,
                        "last_power_above_10w": None,
                        "zero_power_since": None,
                    },
                    "peak_today": {
                        "power_w": 0.0,
                        "at": None,
                    },
                    "yield_today": {
                        "kwh": None,
                        "sensor": None,
                    },
                    "consumption_today": {
                        "kwh": None,
                        "sensor": None,
                    },
                    "autarky": {
                        "percent": None,
                        "calculated_at": None,
                    },
                    "finalized": {
                        "yield_kwh": None,
                        "consumption_kwh": None,
                        "production_hours": None,
                        "accuracy_percent": None,
                        "at": None,
                    },
                },
                "statistics": {
                    "all_time_peak": {
                        "power_w": 0.0,
                        "date": None,
                        "at": None,
                    },
                    "current_week": {
                        "period": None,
                        "date_range": None,
                        "yield_kwh": 0.0,
                        "consumption_kwh": 0.0,
                        "days": 0,
                        "updated_at": None,
                    },
                    "current_month": {
                        "period": None,
                        "yield_kwh": 0.0,
                        "consumption_kwh": 0.0,
                        "avg_autarky": 0.0,
                        "days": 0,
                        "updated_at": None,
                    },
                    "last_7_days": {
                        "avg_yield_kwh": 0.0,
                        "avg_accuracy": 0.0,
                        "total_yield_kwh": 0.0,
                        "calculated_at": None,
                    },
                    "last_30_days": {
                        "avg_yield_kwh": 0.0,
                        "avg_accuracy": 0.0,
                        "total_yield_kwh": 0.0,
                        "calculated_at": None,
                    },
                    "last_365_days": {
                        "avg_yield_kwh": 0.0,
                        "total_yield_kwh": 0.0,
                        "calculated_at": None,
                    },
                },
                "history": [],
                "metadata": {
                    "retention_days": 730,
                    "history_entries": 0,
                    "last_update": None,
                },
            }

            with open(forecasts_file, "w", encoding="utf-8") as f:
                json.dump(forecasts_data, f, indent=2, ensure_ascii=False)

            _LOGGER.info(f"Created daily_forecasts.json (MASTER structure) for {today_str}")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to create daily_forecasts.json: {e}", exc_info=True)
            return False
