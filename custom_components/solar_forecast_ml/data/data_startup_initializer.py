# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import json
import logging
from datetime import datetime, timedelta, time, date
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

_LOGGER = logging.getLogger(__name__)


class StartupInitializer:
    """Synchronous initializer - guarantees critical files exist before async startup.

    Creates:
    1. Directory structure (ai/, data/, stats/, physics/, logs/, backups/auto/)
    2. data/open_meteo_cache.json
    3. stats/astronomy_cache.json
    4. stats/weather_forecast_corrected.json
    5. stats/daily_forecasts.json

    All other files are created by DataSchemaValidator (async).
    """

    def __init__(self, data_dir: Path, config: Dict[str, Any]):
        """Initialize startup initializer."""
        self.data_dir = Path(data_dir)
        self.config = config

        self.latitude = config.get("latitude", 52.52)
        self.longitude = config.get("longitude", 13.40)
        self.solar_capacity_kwp = config.get("solar_capacity", 2.0)
        self.timezone_str = config.get("timezone", "Europe/Berlin")

        try:
            self.timezone = ZoneInfo(self.timezone_str)
        except Exception:
            _LOGGER.warning(f"Invalid timezone '{self.timezone_str}', using Europe/Berlin")
            self.timezone = ZoneInfo("Europe/Berlin")
            self.timezone_str = "Europe/Berlin"

    def initialize_all(self) -> bool:
        """Initialize critical files synchronously.

        MUST run BEFORE any async operations to prevent race conditions.

        Returns:
            True if all critical files ready, False on error.
        """
        _LOGGER.info("=" * 60)
        _LOGGER.info("STARTUP INITIALIZER - Creating critical pre-async files")
        _LOGGER.info("=" * 60)

        success = True

        # Step 1: Create directory structure
        if not self._ensure_directories():
            _LOGGER.error("Failed to create directory structure")
            success = False
        else:
            _LOGGER.info("Directory structure ready")

        # Step 2: Create critical files (only if missing)
        if not self._ensure_open_meteo_cache():
            _LOGGER.warning("open_meteo_cache.json created with baseline data")
        else:
            _LOGGER.info("open_meteo_cache.json ready")

        if not self._ensure_astronomy_cache():
            _LOGGER.error("Failed to create astronomy_cache.json")
            success = False
        else:
            _LOGGER.info("astronomy_cache.json ready")

        if not self._ensure_weather_forecast_corrected():
            _LOGGER.error("Failed to create weather_forecast_corrected.json")
            success = False
        else:
            _LOGGER.info("weather_forecast_corrected.json ready")

        if not self._ensure_daily_forecasts():
            _LOGGER.error("Failed to create daily_forecasts.json")
            success = False
        else:
            _LOGGER.info("daily_forecasts.json ready")

        _LOGGER.info("=" * 60)
        if success:
            _LOGGER.info("STARTUP INITIALIZER complete - 4 critical files ready")
        else:
            _LOGGER.error("STARTUP INITIALIZER failed - some files missing")
        _LOGGER.info("=" * 60)

        return success

    def _ensure_directories(self) -> bool:
        """Create all required directories."""
        directories = [
            self.data_dir,
            self.data_dir / "ai",
            self.data_dir / "data",
            self.data_dir / "stats",
            self.data_dir / "physics",
            self.data_dir / "logs",
            self.data_dir / "backups",
            self.data_dir / "backups" / "auto",
        ]

        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            _LOGGER.debug(f"Created/verified {len(directories)} directories")
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to create directories: {e}")
            return False

    def _write_json(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Write JSON file atomically."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = file_path.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_file.replace(file_path)
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to write {file_path.name}: {e}")
            return False

    def _make_tz_aware_iso(self, date_obj: date, hour: int, minute: int = 0) -> str:
        """Create timezone-aware ISO string."""
        dt = datetime.combine(date_obj, time(hour, minute), tzinfo=self.timezone)
        return dt.isoformat()

    def _ensure_open_meteo_cache(self) -> bool:
        """Create open_meteo_cache.json if missing.

        Structure from production:
        - version: "2.0"
        - metadata: {fetched_at, latitude, longitude, hours_cached, days_cached, mode}
        - forecast: {date: {hour: {temperature, humidity, cloud_cover, ...}}}
        """
        cache_file = self.data_dir / "data" / "open_meteo_cache.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                if data.get("forecast") and len(data["forecast"]) > 0:
                    return True
            except Exception:
                pass

        # Create baseline data
        today = date.today()
        forecast_data = {}

        for i in range(3):
            target_date = today + timedelta(days=i)
            date_str = target_date.isoformat()
            forecast_data[date_str] = {}

            for hour in range(24):
                if 6 <= hour <= 18:
                    elevation = max(0, 60 * (1 - abs(hour - 12) / 6))
                    clear_sky_ghi = max(0, 1000 * (elevation / 60) ** 1.5)
                    direct_rad = int(clear_sky_ghi * 0.35)
                    diffuse_rad = int(clear_sky_ghi * 0.24)
                    ghi = int(clear_sky_ghi * 0.5)
                else:
                    direct_rad = diffuse_rad = ghi = 0

                forecast_data[date_str][str(hour)] = {
                    "temperature": 10.0,
                    "humidity": 70,
                    "cloud_cover": 50,
                    "cloud_cover_low": None,
                    "cloud_cover_mid": None,
                    "cloud_cover_high": None,
                    "precipitation": 0.0,
                    "wind_speed": 3.0,
                    "pressure": 1013.0,
                    "direct_radiation": direct_rad,
                    "diffuse_radiation": diffuse_rad,
                    "ghi": ghi,
                    "global_tilted_irradiance": ghi,
                    "blend_info": {
                        "sources": ["startup_baseline"],
                        "trigger": "initial_setup",
                    },
                    "source": "startup_baseline",
                }

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

        return self._write_json(cache_file, cache_data)

    def _ensure_astronomy_cache(self) -> bool:
        """Create astronomy_cache.json if missing.

        Structure from production:
        - version: "1.0"
        - last_updated, location, pv_system, cache_info, days
        - days[date]: {sunrise_local, sunset_local, solar_noon_local, daylight_hours, hourly}
        """
        cache_file = self.data_dir / "stats" / "astronomy_cache.json"

        if cache_file.exists():
            return True

        today = date.today()
        days_data = {}

        for i in range(7):
            target_date = today + timedelta(days=i)
            date_str = target_date.isoformat()

            hourly_data = {}
            for hour in range(24):
                if 6 <= hour <= 18:
                    elevation = max(0, 60 * (1 - abs(hour - 12) / 6))
                    azimuth = 90 + (hour - 6) * 15
                    clear_sky_rad = max(0, 1000 * (elevation / 60) ** 1.5)
                    theoretical_max = (clear_sky_rad / 1000) * self.solar_capacity_kwp
                else:
                    elevation = azimuth = clear_sky_rad = theoretical_max = 0

                hourly_data[str(hour)] = {
                    "elevation_deg": round(elevation, 1),
                    "azimuth_deg": round(azimuth, 1),
                    "clear_sky_solar_radiation_wm2": round(clear_sky_rad, 0),
                    "theoretical_max_pv_kwh": round(theoretical_max, 3),
                    "hours_since_solar_noon": hour - 12,
                    "day_progress_ratio": round((hour - 6) / 12.0, 3) if 6 <= hour <= 18 else 0,
                }

            days_data[date_str] = {
                "sunrise_local": self._make_tz_aware_iso(target_date, 6, 0),
                "sunset_local": self._make_tz_aware_iso(target_date, 18, 0),
                "solar_noon_local": self._make_tz_aware_iso(target_date, 12, 0),
                "production_window_start": self._make_tz_aware_iso(target_date, 6, 0),
                "production_window_end": self._make_tz_aware_iso(target_date, 18, 0),
                "daylight_hours": 12.0,
                "hourly": hourly_data,
            }

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
                "max_peak_record_kwh": 0.0,
                "max_peak_date": None,
                "max_peak_hour": None,
                "max_peak_conditions": {
                    "sun_elevation_deg": None,
                    "cloud_cover_percent": None,
                    "temperature_c": None,
                    "solar_radiation_wm2": None,
                },
                "hourly_max_peaks": {str(h): {"kwh": 0.0, "date": None, "conditions": {}} for h in range(24)},
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

        return self._write_json(cache_file, cache_data)

    def _ensure_weather_forecast_corrected(self) -> bool:
        """Create weather_forecast_corrected.json if missing.

        Structure from production:
        - version: "4.3"
        - forecast: {date: {hour: {temperature, solar_radiation_wm2, wind, humidity, rain, clouds, pressure, direct_radiation, diffuse_radiation}}}
        - metadata: {...}
        """
        corrected_file = self.data_dir / "stats" / "weather_forecast_corrected.json"

        if corrected_file.exists():
            return True

        # Try to use open_meteo_cache if available
        open_meteo_file = self.data_dir / "data" / "open_meteo_cache.json"
        forecast_by_date = {}

        if open_meteo_file.exists():
            try:
                with open(open_meteo_file, "r") as f:
                    om_data = json.load(f)
                om_forecast = om_data.get("forecast", {})

                for date_str, hours in om_forecast.items():
                    forecast_by_date[date_str] = {}
                    for hour_str, hour_data in hours.items():
                        forecast_by_date[date_str][hour_str] = {
                            "temperature": hour_data.get("temperature", 10.0),
                            "solar_radiation_wm2": hour_data.get("ghi", 0),
                            "wind": hour_data.get("wind_speed", 3.0),
                            "humidity": hour_data.get("humidity", 70),
                            "rain": hour_data.get("precipitation", 0.0),
                            "clouds": hour_data.get("cloud_cover", 50),
                            "cloud_cover_low": hour_data.get("cloud_cover_low"),
                            "cloud_cover_mid": hour_data.get("cloud_cover_mid"),
                            "cloud_cover_high": hour_data.get("cloud_cover_high"),
                            "pressure": hour_data.get("pressure", 1013.0),
                            "direct_radiation": hour_data.get("direct_radiation", 0),
                            "diffuse_radiation": hour_data.get("diffuse_radiation", 0),
                        }
            except Exception:
                pass

        # Fallback: create baseline
        if not forecast_by_date:
            today = date.today()
            for i in range(3):
                target_date = today + timedelta(days=i)
                date_str = target_date.isoformat()
                forecast_by_date[date_str] = {}

                for hour in range(24):
                    if 6 <= hour <= 18:
                        elevation = max(0, 60 * (1 - abs(hour - 12) / 6))
                        radiation = int(max(0, 1000 * (elevation / 60) ** 1.5) * 0.5)
                    else:
                        radiation = 0

                    forecast_by_date[date_str][str(hour)] = {
                        "temperature": 10.0,
                        "solar_radiation_wm2": radiation,
                        "wind": 5.0,
                        "humidity": 60,
                        "rain": 0.0,
                        "clouds": 50,
                        "cloud_cover_low": None,
                        "cloud_cover_mid": None,
                        "cloud_cover_high": None,
                        "pressure": 1013.0,
                        "direct_radiation": 0,
                        "diffuse_radiation": 0,
                    }

        corrected_data = {
            "version": "4.3",
            "forecast": forecast_by_date,
            "metadata": {
                "created": datetime.now().isoformat(),
                "source": "startup_initializer",
                "mode": "direct_radiation",
                "hours_forecast": sum(len(h) for h in forecast_by_date.values()),
                "days_forecast": len(forecast_by_date),
            },
        }

        return self._write_json(corrected_file, corrected_data)

    def _ensure_daily_forecasts(self) -> bool:
        """Create daily_forecasts.json if missing.

        Structure from production:
        - version: "3.0.0"
        - today: {date, forecast_day, forecast_tomorrow, forecast_day_after_tomorrow, ...}
        - statistics: {all_time_peak, current_week, current_month, last_7_days, last_30_days, last_365_days}
        - history: []
        - metadata: {retention_days, history_entries, last_update}
        """
        forecasts_file = self.data_dir / "stats" / "daily_forecasts.json"

        if forecasts_file.exists():
            return True

        today_str = date.today().isoformat()

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
                    "date": today_str,
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

        return self._write_json(forecasts_file, forecasts_data)
