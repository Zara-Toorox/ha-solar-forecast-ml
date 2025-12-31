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
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

_LOGGER = logging.getLogger(__name__)


from dataclasses import dataclass


@dataclass
class PanelGroupTheoreticalMax:
    """Theoretical max output for a single panel group. @zara"""
    name: str
    power_kwp: float
    azimuth_deg: float
    tilt_deg: float
    theoretical_kwh: float
    poa_wm2: float
    aoi_deg: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "power_kwp": round(self.power_kwp, 3),
            "azimuth_deg": self.azimuth_deg,
            "tilt_deg": self.tilt_deg,
            "theoretical_kwh": round(self.theoretical_kwh, 4),
            "poa_wm2": round(self.poa_wm2, 2),
            "aoi_deg": round(self.aoi_deg, 1),
        }

class AstronomyCache:
    """Calculate and cache astronomy data for solar forecasting"""

    def __init__(self, data_dir: Path, data_manager=None):
        self.data_dir = data_dir
        self.data_manager = data_manager
        self.cache_file = data_dir / "stats" / "astronomy_cache.json"
        self.cache_days_ahead = 7

        self.latitude: Optional[float] = None
        self.longitude: Optional[float] = None
        self.elevation_m: Optional[float] = None
        self.timezone: Optional[ZoneInfo] = None

        self._panel_groups: List[Dict[str, Any]] = []

    def set_panel_groups(self, panel_groups: List[Dict[str, Any]]) -> None:
        """Set panel groups for theoretical max calculations. @zara"""
        self._panel_groups = panel_groups or []
        if self._panel_groups:
            _LOGGER.info(
                f"AstronomyCache: Panel groups configured ({len(self._panel_groups)} groups)"
            )

    def initialize_location(
        self, latitude: float, longitude: float, timezone_str: str, elevation_m: float = 0
    ):
        """Initialize location parameters"""
        self.latitude = latitude
        self.longitude = longitude
        self.elevation_m = elevation_m
        self.timezone = ZoneInfo(timezone_str)
        _LOGGER.info(
            f"Astronomy Cache initialized: lat={latitude}, lon={longitude}, "
            f"tz={timezone_str}, elev={elevation_m}m"
        )

    def _calculate_sun_position(
        self, dt: datetime, latitude: float, longitude: float
    ) -> Tuple[float, float]:
        """
        Calculate sun elevation and azimuth for given time

        Args:
            dt: datetime (timezone-aware)
            latitude: Location latitude in degrees
            longitude: Location longitude in degrees

        Returns:
            (elevation_deg, azimuth_deg)
        """

        dt_utc = dt.astimezone(ZoneInfo('UTC'))

        year, month, day = dt_utc.year, dt_utc.month, dt_utc.day
        hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0

        if month <= 2:
            year -= 1
            month += 12

        a = math.floor(year / 100)
        b = 2 - a + math.floor(a / 4)

        jd = (
            math.floor(365.25 * (year + 4716))
            + math.floor(30.6001 * (month + 1))
            + day
            + b
            - 1524.5
            + hour / 24.0
        )

        t = (jd - 2451545.0) / 36525.0

        l0 = (280.46646 + 36000.76983 * t + 0.0003032 * t * t) % 360

        m = (357.52911 + 35999.05029 * t - 0.0001537 * t * t) % 360
        m_rad = math.radians(m)

        c = (
            (1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(m_rad)
            + (0.019993 - 0.000101 * t) * math.sin(2 * m_rad)
            + 0.000289 * math.sin(3 * m_rad)
        )

        true_long = (l0 + c) % 360

        epsilon = 23.439291 - 0.0130042 * t
        epsilon_rad = math.radians(epsilon)

        true_long_rad = math.radians(true_long)
        alpha = math.degrees(
            math.atan2(math.cos(epsilon_rad) * math.sin(true_long_rad), math.cos(true_long_rad))
        )

        delta = math.degrees(math.asin(math.sin(epsilon_rad) * math.sin(true_long_rad)))
        delta_rad = math.radians(delta)

        gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360

        lst = (gmst + longitude) % 360

        hour_angle = (lst - alpha) % 360
        if hour_angle > 180:
            hour_angle -= 360
        hour_angle_rad = math.radians(hour_angle)

        lat_rad = math.radians(latitude)

        sin_elevation = math.sin(lat_rad) * math.sin(delta_rad) + math.cos(lat_rad) * math.cos(
            delta_rad
        ) * math.cos(hour_angle_rad)
        elevation = math.degrees(math.asin(max(-1, min(1, sin_elevation))))

        cos_azimuth = (math.sin(delta_rad) - math.sin(lat_rad) * sin_elevation) / (
            math.cos(lat_rad) * math.cos(math.radians(elevation))
        )
        cos_azimuth = max(-1, min(1, cos_azimuth))
        azimuth = math.degrees(math.acos(cos_azimuth))

        if hour_angle > 0:
            azimuth = 360 - azimuth

        return elevation, azimuth

    def _calculate_sunrise_sunset(
        self, target_date: date, latitude: float, longitude: float, timezone: ZoneInfo
    ) -> Tuple[Optional[datetime], Optional[datetime], Optional[datetime]]:
        """
        Calculate sunrise, sunset, and solar noon for a given date

        Returns:
            (sunrise, sunset, solar_noon) all timezone-aware
        """

        noon_local = datetime.combine(target_date, datetime.min.time().replace(hour=12))
        noon_local = noon_local.replace(tzinfo=timezone)

        solar_noon = None
        max_elevation = -90

        for minute in range(10 * 60, 14 * 60):
            test_time = datetime.combine(target_date, datetime.min.time())
            test_time = test_time.replace(hour=minute // 60, minute=minute % 60, tzinfo=timezone)
            elevation, _ = self._calculate_sun_position(test_time, latitude, longitude)
            if elevation > max_elevation:
                max_elevation = elevation
                solar_noon = test_time

        if solar_noon is None:
            return None, None, None

        sunrise = None
        for hour in range(0, solar_noon.hour + 1):
            for minute in range(0, 60, 5):
                test_time = datetime.combine(target_date, datetime.min.time())
                test_time = test_time.replace(hour=hour, minute=minute, tzinfo=timezone)
                elevation, _ = self._calculate_sun_position(test_time, latitude, longitude)
                if elevation > -0.833:
                    sunrise = test_time
                    break
            if sunrise:
                break

        sunset = None
        for hour in range(solar_noon.hour, 24):
            for minute in range(0, 60, 5):
                test_time = datetime.combine(target_date, datetime.min.time())
                test_time = test_time.replace(hour=hour, minute=minute, tzinfo=timezone)
                elevation, _ = self._calculate_sun_position(test_time, latitude, longitude)
                if elevation < -0.833:
                    sunset = test_time
                    break
            if sunset:
                break

        return sunrise, sunset, solar_noon

    def _calculate_clear_sky_solar_radiation(self, elevation_deg: float, day_of_year: int) -> float:
        """Calculate clear sky solar radiation using simplified model @zara"""
        if elevation_deg <= 0:
            return 0.0

        solar_constant = 1367

        distance_factor = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)

        elevation_rad = math.radians(elevation_deg)
        air_mass = 1 / (math.sin(elevation_rad) + 0.50572 * (elevation_deg + 6.07995) ** -1.6364)
        transmission = 0.7 ** (air_mass**0.678)

        clear_sky_radiation = (
            solar_constant * distance_factor * math.sin(elevation_rad) * transmission
        )

        return max(0, clear_sky_radiation)

    def _calculate_theoretical_pv_output(
        self, solar_radiation_wm2: float, system_capacity_kwp: float, efficiency: float = 0.95
    ) -> float:
        """
        Calculate theoretical PV output for one hour

        Args:
            solar_radiation_wm2: Solar radiation in W/m²
            system_capacity_kwp: System capacity in kWp
            efficiency: Overall system efficiency (0.0-1.0)
                       Default 0.95 (modern inverters + cabling: 95-98%)
                       Panel efficiency already baked into kWp rating

        Returns:
            Theoretical output in kWh (for HORIZONTAL plane)
            Note: Actual tilted panels will produce more via geometry factor
        """

        stc_radiation = 1000.0

        pv_output_kwh = (
            system_capacity_kwp * (solar_radiation_wm2 / stc_radiation) * efficiency * 1.0
        )

        return max(0, pv_output_kwh)

    def _calculate_theoretical_pv_per_group(
        self,
        clear_sky_radiation_wm2: float,
        sun_elevation_deg: float,
        sun_azimuth_deg: float,
        efficiency: float = 0.95,
    ) -> Tuple[float, List[PanelGroupTheoreticalMax]]:
        """Calculate theoretical PV output per panel group based on orientation. @zara

        Uses proper physics calculations for each panel group's orientation
        relative to the sun position.

        Args:
            clear_sky_radiation_wm2: Clear sky GHI in W/m²
            sun_elevation_deg: Sun elevation angle in degrees
            sun_azimuth_deg: Sun azimuth angle in degrees
            efficiency: System efficiency (default 0.95)

        Returns:
            Tuple of (total_kwh, list of per-group results)
        """
        if not self._panel_groups or sun_elevation_deg <= 0:
            return 0.0, []

        group_results: List[PanelGroupTheoreticalMax] = []
        total_kwh = 0.0

        for idx, group in enumerate(self._panel_groups):
            group_name = group.get("name", f"Gruppe {idx + 1}")
            power_wp = float(group.get("power_wp", 0))
            power_kwp = power_wp / 1000.0
            azimuth_deg = float(group.get("azimuth", 180))
            tilt_deg = float(group.get("tilt", 30))

            if power_kwp <= 0:
                continue

            aoi_deg = self._calculate_aoi(
                sun_elevation_deg, sun_azimuth_deg, tilt_deg, azimuth_deg
            )

            poa_wm2 = self._calculate_poa_from_ghi(
                clear_sky_radiation_wm2, sun_elevation_deg, sun_azimuth_deg,
                tilt_deg, azimuth_deg, aoi_deg
            )

            stc_radiation = 1000.0
            theoretical_kwh = power_kwp * (poa_wm2 / stc_radiation) * efficiency

            group_results.append(PanelGroupTheoreticalMax(
                name=group_name,
                power_kwp=power_kwp,
                azimuth_deg=azimuth_deg,
                tilt_deg=tilt_deg,
                theoretical_kwh=theoretical_kwh,
                poa_wm2=poa_wm2,
                aoi_deg=aoi_deg,
            ))
            total_kwh += theoretical_kwh

        return total_kwh, group_results

    def _calculate_aoi(
        self,
        sun_elevation_deg: float,
        sun_azimuth_deg: float,
        panel_tilt_deg: float,
        panel_azimuth_deg: float,
    ) -> float:
        """Calculate Angle of Incidence (AOI) between sun and panel. @zara"""
        sun_zenith = 90.0 - sun_elevation_deg
        sun_zenith_rad = math.radians(sun_zenith)
        panel_tilt_rad = math.radians(panel_tilt_deg)
        azimuth_diff_rad = math.radians(sun_azimuth_deg - panel_azimuth_deg)

        cos_aoi = (
            math.cos(sun_zenith_rad) * math.cos(panel_tilt_rad)
            + math.sin(sun_zenith_rad) * math.sin(panel_tilt_rad) * math.cos(azimuth_diff_rad)
        )

        cos_aoi = max(-1.0, min(1.0, cos_aoi))
        aoi_deg = math.degrees(math.acos(cos_aoi))

        return aoi_deg

    def _calculate_poa_from_ghi(
        self,
        ghi_wm2: float,
        sun_elevation_deg: float,
        sun_azimuth_deg: float,
        panel_tilt_deg: float,
        panel_azimuth_deg: float,
        aoi_deg: float,
        albedo: float = 0.2,
    ) -> float:
        """Calculate Plane of Array irradiance from GHI for a specific panel orientation. @zara

        For clear sky conditions, we estimate DNI and DHI from GHI.
        """
        if ghi_wm2 <= 0 or sun_elevation_deg <= 0:
            return 0.0

        sun_elevation_rad = math.radians(sun_elevation_deg)

        dni_estimated = ghi_wm2 / max(0.01, math.sin(sun_elevation_rad)) * 0.85
        dhi_estimated = ghi_wm2 * 0.15

        dni_estimated = min(dni_estimated, 1000.0)
        dhi_estimated = max(dhi_estimated, ghi_wm2 * 0.1)

        if aoi_deg < 90:
            poa_beam = dni_estimated * math.cos(math.radians(aoi_deg))
        else:
            poa_beam = 0.0

        panel_tilt_rad = math.radians(panel_tilt_deg)
        poa_diffuse = dhi_estimated * (1 + math.cos(panel_tilt_rad)) / 2

        poa_ground = ghi_wm2 * albedo * (1 - math.cos(panel_tilt_rad)) / 2

        poa_total = poa_beam + poa_diffuse + poa_ground

        return max(0.0, poa_total)

    async def build_cache_for_date(
        self, target_date: date, system_capacity_kwp: float
    ) -> Optional[Dict]:
        """
        Build astronomy cache for a specific date

        Args:
            target_date: Date to calculate for
            system_capacity_kwp: PV system capacity

        Returns:
            Dictionary with astronomy data for the date
        """
        if not all([self.latitude, self.longitude, self.timezone]):
            _LOGGER.error("Astronomy Cache not initialized with location")
            return None

        def _build_sync():
            try:

                sunrise, sunset, solar_noon = self._calculate_sunrise_sunset(
                    target_date, self.latitude, self.longitude, self.timezone
                )

                if not sunrise or not sunset or not solar_noon:
                    _LOGGER.warning(f"Could not calculate sun times for {target_date}")
                    return None

                production_start = sunrise - timedelta(minutes=30)
                production_end = sunset + timedelta(minutes=30)

                daylight_hours = (sunset - sunrise).total_seconds() / 3600.0

                hourly_data = {}
                day_of_year = target_date.timetuple().tm_yday

                for hour in range(24):

                    hour_time = datetime.combine(target_date, datetime.min.time())
                    hour_time = hour_time.replace(hour=hour, minute=30, tzinfo=self.timezone)

                    elevation, azimuth = self._calculate_sun_position(
                        hour_time, self.latitude, self.longitude
                    )

                    clear_sky_sr = self._calculate_clear_sky_solar_radiation(elevation, day_of_year)

                    if self._panel_groups:
                        total_theoretical, group_results = self._calculate_theoretical_pv_per_group(
                            clear_sky_sr, elevation, azimuth
                        )
                        theoretical_pv = total_theoretical
                    else:
                        theoretical_pv = self._calculate_theoretical_pv_output(
                            clear_sky_sr, system_capacity_kwp
                        )
                        group_results = []

                    hours_since_noon = (hour_time - solar_noon).total_seconds() / 3600.0

                    if sunrise <= hour_time <= sunset:
                        day_progress = (hour_time - sunrise).total_seconds() / (
                            sunset - sunrise
                        ).total_seconds()
                    else:
                        day_progress = 0.0 if hour_time < sunrise else 1.0

                    hour_data = {
                        "elevation_deg": round(elevation, 2),
                        "azimuth_deg": round(azimuth, 2),
                        "clear_sky_solar_radiation_wm2": round(clear_sky_sr, 1),
                        "theoretical_max_pv_kwh": round(theoretical_pv, 4),
                        "hours_since_solar_noon": round(hours_since_noon, 2),
                        "day_progress_ratio": round(day_progress, 3),
                    }

                    if group_results:
                        hour_data["theoretical_max_per_group"] = [
                            g.to_dict() for g in group_results
                        ]

                    hourly_data[str(hour)] = hour_data

                return {
                    "sunrise_local": sunrise.isoformat(),
                    "sunset_local": sunset.isoformat(),
                    "solar_noon_local": solar_noon.isoformat(),
                    "production_window_start": production_start.isoformat(),
                    "production_window_end": production_end.isoformat(),
                    "daylight_hours": round(daylight_hours, 2),
                    "hourly": hourly_data,
                }

            except Exception as e:
                _LOGGER.error(f"Error building cache for {target_date}: {e}", exc_info=True)
                return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _build_sync)

    async def rebuild_cache(
        self,
        system_capacity_kwp: float,
        start_date: Optional[date] = None,
        days_back: int = 30,
        days_ahead: int = 7,
    ) -> Dict:
        """
        Rebuild entire astronomy cache

        Args:
            start_date: Starting date (default: today)
            days_back: Days to calculate backwards (default: 30)
            days_ahead: Days to calculate ahead (default: 7)
            system_capacity_kwp: PV system capacity

        Returns:
            Statistics about the rebuild
        """
        if start_date is None:
            start_date = datetime.now(self.timezone).date()

        _LOGGER.info(
            f"Rebuilding astronomy cache: {days_back} days back, "
            f"{days_ahead} days ahead from {start_date}"
        )

        cache_data = {

            "version": "1.0",
            "last_updated": datetime.now(self.timezone).isoformat(),

            "location": {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "elevation_m": self.elevation_m,
                "timezone": str(self.timezone),
            },

            "pv_system": {
                "installed_capacity_kwp": system_capacity_kwp

            },

            "cache_info": {
                "total_days": 0,
                "days_back": days_back,
                "days_ahead": days_ahead,
                "date_range_start": (start_date - timedelta(days=days_back)).isoformat(),
                "date_range_end": (start_date + timedelta(days=days_ahead)).isoformat(),
            },

            "days": {},
        }

        start_calc = start_date - timedelta(days=days_back)
        end_calc = start_date + timedelta(days=days_ahead)

        total_days = (end_calc - start_calc).days + 1
        success_count = 0
        error_count = 0

        # Build list of dates to process
        dates_to_process = []
        current_date = start_calc
        while current_date <= end_calc:
            dates_to_process.append(current_date)
            current_date += timedelta(days=1)

        # Process all dates in PARALLEL using asyncio.gather for faster startup
        # Each build_cache_for_date already uses run_in_executor internally
        _LOGGER.info(f"Astronomy cache: Processing {total_days} days in parallel...")

        async def build_with_date(d: date):
            """Helper to return (date_str, day_data) tuple"""
            day_data = await self.build_cache_for_date(d, system_capacity_kwp)
            return (d.isoformat(), day_data)

        results = await asyncio.gather(
            *[build_with_date(d) for d in dates_to_process],
            return_exceptions=True
        )

        # Collect results
        for result in results:
            if isinstance(result, Exception):
                _LOGGER.error(f"Astronomy cache build error: {result}")
                error_count += 1
            elif result[1] is not None:
                date_str, day_data = result
                cache_data["days"][date_str] = day_data
                success_count += 1
            else:
                error_count += 1

        _LOGGER.info(f"Astronomy cache: {success_count}/{total_days} days processed")

        cache_data["cache_info"]["total_days"] = success_count
        cache_data["cache_info"]["success_count"] = success_count
        cache_data["cache_info"]["error_count"] = error_count

        if self.data_manager:
            await self.data_manager._atomic_write_json(self.cache_file, cache_data)
        else:

            def _write_sync():
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, indent=2, sort_keys=False, ensure_ascii=False)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _write_sync)

        _LOGGER.info(
            f"Astronomy cache rebuilt: {success_count} days successful, " f"{error_count} errors"
        )

        return {
            "total_days": total_days,
            "success_count": success_count,
            "error_count": error_count,
            "cache_file": str(self.cache_file),
        }

    async def get_day_data(self, target_date: date) -> Optional[Dict]:
        """Get astronomy data for a specific date from cache @zara"""

        def _load_sync():
            try:
                if not self.cache_file.exists():
                    return None

                with open(self.cache_file, "r") as f:
                    cache = json.load(f)

                date_str = target_date.isoformat()
                return cache.get("days", {}).get(date_str)

            except Exception as e:
                _LOGGER.error(f"Error loading astronomy cache: {e}")
                return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _load_sync)

    async def get_hourly_data(self, target_date: date, target_hour: int) -> Optional[Dict]:
        """Get astronomy data for a specific hour @zara"""
        day_data = await self.get_day_data(target_date)
        if not day_data:
            return None

        return day_data.get("hourly", {}).get(str(target_hour))

    async def get_production_window(self, target_date: date) -> Optional[Tuple[datetime, datetime]]:
        """Get production window for a date @zara"""
        day_data = await self.get_day_data(target_date)
        if not day_data:
            return None

        try:
            start = datetime.fromisoformat(day_data["production_window_start"])
            end = datetime.fromisoformat(day_data["production_window_end"])
            return start, end
        except (KeyError, ValueError) as e:
            _LOGGER.error(f"Error parsing production window: {e}")
            return None
