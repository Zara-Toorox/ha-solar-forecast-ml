"""Startup Initializer - Guarantees all critical JSON files exist BEFORE async operations V10.0.0 @zara

IMPORTANT: All JSON structures created here MUST match the MASTER files in production!
Master location: /Volumes/config/solar_forecast_ml/
Do NOT add fields that don't exist in the master!

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
from typing import Any, Dict

_LOGGER = logging.getLogger(__name__)

class StartupInitializer:
    """Synchronous initializer - guarantees critical files exist before async startup"""

    def __init__(self, data_dir: Path, config: Dict[str, Any]):
        """Initialize startup initializer @zara"""
        self.data_dir = Path(data_dir)
        self.config = config

        self.latitude = config.get("latitude", 52.52)
        self.longitude = config.get("longitude", 13.40)
        self.solar_capacity_kwp = config.get("solar_capacity", 2.0)
        self.timezone_str = config.get("timezone", "Europe/Berlin")

    def initialize_all(self) -> bool:
        """Initialize ALL critical JSON files synchronously @zara"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("🚀 STARTUP INITIALIZER - Guaranteeing critical files exist")
        _LOGGER.info("=" * 80)

        success = True

        # CRITICAL: open_meteo_cache FIRST - other files may depend on it!
        if not self._ensure_open_meteo_cache():
            _LOGGER.warning("⚠️  open_meteo_cache.json - Created empty (will be filled by API)")
        else:
            _LOGGER.info("✅ open_meteo_cache.json - Ready")

        if not self._ensure_astronomy_cache():
            _LOGGER.error("❌ Failed to create astronomy_cache.json")
            success = False
        else:
            _LOGGER.info("✅ astronomy_cache.json - Ready")

        if not self._ensure_weather_forecast_corrected():
            _LOGGER.error("❌ Failed to create weather_forecast_corrected.json")
            success = False
        else:
            _LOGGER.info("✅ weather_forecast_corrected.json - Ready")

        if self.config.get("battery_enabled", False):
            if not self._ensure_battery_charge_history():
                _LOGGER.warning("⚠️  Failed to create battery_charge_history.json")
            else:
                _LOGGER.info("✅ battery_charge_history.json - Ready")

        _LOGGER.info("=" * 80)
        if success:
            _LOGGER.info("✅ STARTUP INITIALIZER - All critical files guaranteed")
        else:
            _LOGGER.error("❌ STARTUP INITIALIZER - Some files failed (check logs above)")
        _LOGGER.info("=" * 80)

        return success

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
                days_data[date_str] = {
                    "sunrise_local": f"{date_str}T06:00:00",
                    "sunset_local": f"{date_str}T18:00:00",
                    "solar_noon_local": f"{date_str}T12:00:00",
                    "production_window_start": f"{date_str}T06:00:00",
                    "production_window_end": f"{date_str}T18:00:00",
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

    def _ensure_battery_charge_history(self) -> bool:
        """Create battery_charge_history.json matching MASTER structure @zara

        MASTER structure (from /Volumes/config/solar_forecast_ml/data/battery_charge_history.json):
        - version: "1.0"
        - battery_capacity, last_update, daily, monthly, yearly
        """
        battery_file = self.data_dir / "data" / "battery_charge_history.json"

        if battery_file.exists():
            _LOGGER.debug("battery_charge_history.json already exists, skipping")
            return True

        try:
            battery_file.parent.mkdir(parents=True, exist_ok=True)

            # MASTER structure for battery_charge_history.json
            battery_data = {
                "version": "1.0",
                "battery_capacity": self.config.get("battery_capacity", 10.0),
                "last_update": None,
                "daily": {},
                "monthly": {},
                "yearly": {},
            }

            with open(battery_file, "w", encoding="utf-8") as f:
                json.dump(battery_data, f, indent=2, ensure_ascii=False)

            _LOGGER.info("Created battery_charge_history.json (MASTER structure)")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to create battery_charge_history.json: {e}", exc_info=True)
            return False
