"""
Astronomy Cache - Independent solar position and radiation calculations

Provides:
- Sun position (elevation, azimuth) for every hour
- Clear sky solar radiation calculations
- Production windows
- No dependency on sun.sun entity
- Atomic file writes with locks
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta, date
import json
import logging
import math
import asyncio
from zoneinfo import ZoneInfo

_LOGGER = logging.getLogger(__name__)


class AstronomyCache:
    """Calculate and cache astronomy data for solar forecasting"""

    def __init__(self, data_dir: Path, data_manager=None):
        self.data_dir = data_dir
        self.data_manager = data_manager
        self.cache_file = data_dir / "stats" / "astronomy_cache.json"
        self.cache_days_ahead = 7

        # Will be loaded from config
        self.latitude: Optional[float] = None
        self.longitude: Optional[float] = None
        self.elevation_m: Optional[float] = None
        self.timezone: Optional[ZoneInfo] = None

    def initialize_location(
        self,
        latitude: float,
        longitude: float,
        timezone_str: str,
        elevation_m: float = 0
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
        self,
        dt: datetime,
        latitude: float,
        longitude: float
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
        # Convert to Julian Day
        year, month, day = dt.year, dt.month, dt.day
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

        if month <= 2:
            year -= 1
            month += 12

        a = math.floor(year / 100)
        b = 2 - a + math.floor(a / 4)

        jd = (
            math.floor(365.25 * (year + 4716)) +
            math.floor(30.6001 * (month + 1)) +
            day + b - 1524.5 +
            hour / 24.0
        )

        # Time in Julian centuries since J2000.0
        t = (jd - 2451545.0) / 36525.0

        # Mean longitude of sun (degrees)
        l0 = (280.46646 + 36000.76983 * t + 0.0003032 * t * t) % 360

        # Mean anomaly (degrees)
        m = (357.52911 + 35999.05029 * t - 0.0001537 * t * t) % 360
        m_rad = math.radians(m)

        # Equation of center
        c = (
            (1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(m_rad) +
            (0.019993 - 0.000101 * t) * math.sin(2 * m_rad) +
            0.000289 * math.sin(3 * m_rad)
        )

        # True longitude
        true_long = (l0 + c) % 360

        # Obliquity of ecliptic
        epsilon = 23.439291 - 0.0130042 * t
        epsilon_rad = math.radians(epsilon)

        # Right ascension
        true_long_rad = math.radians(true_long)
        alpha = math.degrees(math.atan2(
            math.cos(epsilon_rad) * math.sin(true_long_rad),
            math.cos(true_long_rad)
        ))

        # Declination
        delta = math.degrees(math.asin(
            math.sin(epsilon_rad) * math.sin(true_long_rad)
        ))
        delta_rad = math.radians(delta)

        # Greenwich Mean Sidereal Time
        gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360

        # Local Sidereal Time
        lst = (gmst + longitude) % 360

        # Hour angle
        hour_angle = (lst - alpha) % 360
        if hour_angle > 180:
            hour_angle -= 360
        hour_angle_rad = math.radians(hour_angle)

        # Convert latitude to radians
        lat_rad = math.radians(latitude)

        # Calculate elevation
        sin_elevation = (
            math.sin(lat_rad) * math.sin(delta_rad) +
            math.cos(lat_rad) * math.cos(delta_rad) * math.cos(hour_angle_rad)
        )
        elevation = math.degrees(math.asin(max(-1, min(1, sin_elevation))))

        # Calculate azimuth
        cos_azimuth = (
            (math.sin(delta_rad) - math.sin(lat_rad) * sin_elevation) /
            (math.cos(lat_rad) * math.cos(math.radians(elevation)))
        )
        cos_azimuth = max(-1, min(1, cos_azimuth))
        azimuth = math.degrees(math.acos(cos_azimuth))

        if hour_angle > 0:
            azimuth = 360 - azimuth

        return elevation, azimuth

    def _calculate_sunrise_sunset(
        self,
        target_date: date,
        latitude: float,
        longitude: float,
        timezone: ZoneInfo
    ) -> Tuple[Optional[datetime], Optional[datetime], Optional[datetime]]:
        """
        Calculate sunrise, sunset, and solar noon for a given date

        Returns:
            (sunrise, sunset, solar_noon) all timezone-aware
        """
        # Use noon as reference point
        noon_local = datetime.combine(target_date, datetime.min.time().replace(hour=12))
        noon_local = noon_local.replace(tzinfo=timezone)

        # Find solar noon (when elevation is maximum)
        solar_noon = None
        max_elevation = -90

        for minute in range(10 * 60, 14 * 60):  # Search 10:00 - 14:00
            test_time = datetime.combine(target_date, datetime.min.time())
            test_time = test_time.replace(
                hour=minute // 60,
                minute=minute % 60,
                tzinfo=timezone
            )
            elevation, _ = self._calculate_sun_position(test_time, latitude, longitude)
            if elevation > max_elevation:
                max_elevation = elevation
                solar_noon = test_time

        if solar_noon is None:
            return None, None, None

        # Find sunrise (first time elevation > 0 before solar noon)
        sunrise = None
        for hour in range(0, solar_noon.hour + 1):
            for minute in range(0, 60, 5):
                test_time = datetime.combine(target_date, datetime.min.time())
                test_time = test_time.replace(hour=hour, minute=minute, tzinfo=timezone)
                elevation, _ = self._calculate_sun_position(test_time, latitude, longitude)
                if elevation > -0.833:  # Account for atmospheric refraction
                    sunrise = test_time
                    break
            if sunrise:
                break

        # Find sunset (first time elevation < 0 after solar noon)
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

    def _calculate_clear_sky_solar_radiation(
        self,
        elevation_deg: float,
        day_of_year: int
    ) -> float:
        """
        Calculate clear sky solar radiation using simplified model

        Args:
            elevation_deg: Sun elevation in degrees
            day_of_year: Day of year (1-365)

        Returns:
            Solar radiation in W/m²
        """
        if elevation_deg <= 0:
            return 0.0

        # Solar constant
        solar_constant = 1367  # W/m²

        # Earth-Sun distance correction
        distance_factor = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)

        # Atmospheric transmission (depends on elevation)
        elevation_rad = math.radians(elevation_deg)
        air_mass = 1 / (math.sin(elevation_rad) + 0.50572 * (elevation_deg + 6.07995)**-1.6364)
        transmission = 0.7**(air_mass**0.678)

        # Clear sky radiation
        clear_sky_radiation = (
            solar_constant *
            distance_factor *
            math.sin(elevation_rad) *
            transmission
        )

        return max(0, clear_sky_radiation)

    def _calculate_theoretical_pv_output(
        self,
        solar_radiation_wm2: float,
        system_capacity_kwp: float = 5.0,
        efficiency: float = 0.85
    ) -> float:
        """
        Calculate theoretical PV output for one hour

        Args:
            solar_radiation_wm2: Solar radiation in W/m²
            system_capacity_kwp: System capacity in kWp
            efficiency: Overall system efficiency (0.0-1.0)

        Returns:
            Theoretical output in kWh
        """
        # Standard Test Conditions: 1000 W/m²
        stc_radiation = 1000.0

        # PV output = capacity × (radiation / STC) × efficiency × time
        # time = 1 hour
        pv_output_kwh = (
            system_capacity_kwp *
            (solar_radiation_wm2 / stc_radiation) *
            efficiency *
            1.0  # 1 hour
        )

        return max(0, pv_output_kwh)

    async def build_cache_for_date(
        self,
        target_date: date,
        system_capacity_kwp: float = 5.0
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
                # Calculate sunrise, sunset, solar noon
                sunrise, sunset, solar_noon = self._calculate_sunrise_sunset(
                    target_date,
                    self.latitude,
                    self.longitude,
                    self.timezone
                )

                if not sunrise or not sunset or not solar_noon:
                    _LOGGER.warning(f"Could not calculate sun times for {target_date}")
                    return None

                # Production window: sunrise - 1h to sunset + 1h
                production_start = sunrise - timedelta(hours=1)
                production_end = sunset + timedelta(hours=1)

                # Calculate daylight hours
                daylight_hours = (sunset - sunrise).total_seconds() / 3600.0

                # Calculate hourly data (0-23)
                hourly_data = {}
                day_of_year = target_date.timetuple().tm_yday

                for hour in range(24):
                    # Middle of the hour
                    hour_time = datetime.combine(target_date, datetime.min.time())
                    hour_time = hour_time.replace(
                        hour=hour,
                        minute=30,
                        tzinfo=self.timezone
                    )

                    # Calculate sun position
                    elevation, azimuth = self._calculate_sun_position(
                        hour_time,
                        self.latitude,
                        self.longitude
                    )

                    # Calculate solar radiation
                    clear_sky_sr = self._calculate_clear_sky_solar_radiation(
                        elevation,
                        day_of_year
                    )

                    # Calculate theoretical PV output
                    theoretical_pv = self._calculate_theoretical_pv_output(
                        clear_sky_sr,
                        system_capacity_kwp
                    )

                    # Hours since solar noon (for ML feature)
                    hours_since_noon = (hour_time - solar_noon).total_seconds() / 3600.0

                    # Day progress ratio (0.0 at sunrise, 1.0 at sunset)
                    if sunrise <= hour_time <= sunset:
                        day_progress = (hour_time - sunrise).total_seconds() / (sunset - sunrise).total_seconds()
                    else:
                        day_progress = 0.0 if hour_time < sunrise else 1.0

                    hourly_data[str(hour)] = {
                        "elevation_deg": round(elevation, 2),
                        "azimuth_deg": round(azimuth, 2),
                        "clear_sky_solar_radiation_wm2": round(clear_sky_sr, 1),
                        "theoretical_max_pv_kwh": round(theoretical_pv, 4),
                        "hours_since_solar_noon": round(hours_since_noon, 2),
                        "day_progress_ratio": round(day_progress, 3)
                    }

                return {
                    "sunrise_local": sunrise.isoformat(),
                    "sunset_local": sunset.isoformat(),
                    "solar_noon_local": solar_noon.isoformat(),
                    "production_window_start": production_start.isoformat(),
                    "production_window_end": production_end.isoformat(),
                    "daylight_hours": round(daylight_hours, 2),
                    "hourly": hourly_data
                }

            except Exception as e:
                _LOGGER.error(f"Error building cache for {target_date}: {e}", exc_info=True)
                return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _build_sync)

    async def rebuild_cache(
        self,
        start_date: Optional[date] = None,
        days_back: int = 30,  # 30 days history
        days_ahead: int = 7,   # 7 days forecast
        system_capacity_kwp: float = 5.0
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
            # === USER INFO - Quick overview at top ===
            "version": "1.0",
            "last_updated": datetime.now(self.timezone).isoformat(),

            # === LOCATION ===
            "location": {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "elevation_m": self.elevation_m,
                "timezone": str(self.timezone)
            },

            # === PV SYSTEM ===
            "pv_system": {
                "installed_capacity_kwp": system_capacity_kwp
                # Max peaks will be added later by max_peak_tracker
            },

            # === CACHE INFO ===
            "cache_info": {
                "total_days": 0,  # Will be updated after processing
                "days_back": days_back,
                "days_ahead": days_ahead,
                "date_range_start": (start_date - timedelta(days=days_back)).isoformat(),
                "date_range_end": (start_date + timedelta(days=days_ahead)).isoformat()
            },

            # === DAILY ASTRONOMY DATA ===
            "days": {},

            # === LEGACY - For backwards compatibility ===
            "metadata": {
                "version": "1.0",
                "last_updated": datetime.now(self.timezone).isoformat(),
                "cache_days_ahead": days_ahead,
                "location": {
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "elevation_m": self.elevation_m,
                    "timezone": str(self.timezone)
                },
                "pv_system": {
                    "installed_capacity_kwp": system_capacity_kwp
                },
                "cache_info": {
                    "total_days": 0,
                    "days_back": days_back,
                    "days_ahead": days_ahead,
                    "date_range_start": (start_date - timedelta(days=days_back)).isoformat(),
                    "date_range_end": (start_date + timedelta(days=days_ahead)).isoformat()
                }
            }
        }

        # Calculate date range
        start_calc = start_date - timedelta(days=days_back)
        end_calc = start_date + timedelta(days=days_ahead)

        total_days = (end_calc - start_calc).days + 1
        success_count = 0
        error_count = 0

        current_date = start_calc
        while current_date <= end_calc:
            date_str = current_date.isoformat()

            day_data = await self.build_cache_for_date(
                current_date,
                system_capacity_kwp
            )

            if day_data:
                cache_data["days"][date_str] = day_data
                success_count += 1
            else:
                error_count += 1

            current_date += timedelta(days=1)

            # Log progress every 10 days
            if success_count % 10 == 0:
                _LOGGER.info(f"Astronomy cache: {success_count}/{total_days} days processed")

        # Update cache info with final counts (both top-level and legacy metadata)
        cache_data["cache_info"]["total_days"] = success_count
        cache_data["cache_info"]["success_count"] = success_count
        cache_data["cache_info"]["error_count"] = error_count

        cache_data["metadata"]["cache_info"]["total_days"] = success_count
        cache_data["metadata"]["cache_info"]["success_count"] = success_count
        cache_data["metadata"]["cache_info"]["error_count"] = error_count

        # Write cache atomically
        if self.data_manager:
            await self.data_manager._atomic_write_json(self.cache_file, cache_data)
        else:
            # Fallback: direct write with proper encoding
            def _write_sync():
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, sort_keys=False, ensure_ascii=False)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _write_sync)

        _LOGGER.info(
            f"Astronomy cache rebuilt: {success_count} days successful, "
            f"{error_count} errors"
        )

        return {
            "total_days": total_days,
            "success_count": success_count,
            "error_count": error_count,
            "cache_file": str(self.cache_file)
        }

    async def get_day_data(self, target_date: date) -> Optional[Dict]:
        """
        Get astronomy data for a specific date from cache

        Args:
            target_date: Date to retrieve

        Returns:
            Dictionary with astronomy data or None if not in cache
        """
        def _load_sync():
            try:
                if not self.cache_file.exists():
                    return None

                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)

                date_str = target_date.isoformat()
                return cache.get("days", {}).get(date_str)

            except Exception as e:
                _LOGGER.error(f"Error loading astronomy cache: {e}")
                return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _load_sync)

    async def get_hourly_data(
        self,
        target_date: date,
        target_hour: int
    ) -> Optional[Dict]:
        """
        Get astronomy data for a specific hour

        Args:
            target_date: Date
            target_hour: Hour (0-23)

        Returns:
            Dictionary with hour data or None
        """
        day_data = await self.get_day_data(target_date)
        if not day_data:
            return None

        return day_data.get("hourly", {}).get(str(target_hour))

    async def get_production_window(
        self,
        target_date: date
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Get production window for a date

        Args:
            target_date: Date

        Returns:
            (start, end) as timezone-aware datetimes or None
        """
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
