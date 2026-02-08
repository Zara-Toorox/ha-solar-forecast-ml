# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Astronomy Cache Module for Solar Forecast ML V16.2.0.
Calculates and caches solar position data using pvlib-style calculations.
All data is stored in SQLite database via DatabaseManager.
"""

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from ..data.db_manager import DatabaseManager

_LOGGER = logging.getLogger(__name__)


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
    """Calculate and cache astronomy data for solar forecasting using DB storage. @zara"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.latitude: Optional[float] = None
        self.longitude: Optional[float] = None
        self.elevation_m: Optional[float] = None
        self.timezone: Optional[ZoneInfo] = None
        self._panel_groups: List[Dict[str, Any]] = []

    def set_panel_groups(self, panel_groups: List[Dict[str, Any]]) -> None:
        """Set panel groups for theoretical max calculations. @zara"""
        self._panel_groups = panel_groups or []
        if self._panel_groups:
            _LOGGER.info(f"Panel groups configured: {len(self._panel_groups)} groups")

    def initialize_location(
        self, latitude: float, longitude: float, timezone_str: str, elevation_m: float = 0
    ):
        """Initialize location parameters and save to DB. @zara"""
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
        """Calculate sun elevation and azimuth for given time using astronomical formulas. @zara"""
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
        """Calculate sunrise, sunset, and solar noon for a given date. @zara"""
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
        """Calculate clear sky solar radiation using simplified model. @zara"""
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
        """Calculate theoretical PV output for one hour. @zara"""
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
        """Calculate theoretical PV output per panel group based on orientation. @zara"""
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
        """Calculate Angle of Incidence between sun and panel. @zara"""
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
        """Calculate Plane of Array irradiance from GHI for specific panel orientation. @zara"""
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
    ) -> bool:
        """Build astronomy cache for a specific date and save to DB. @zara"""
        if not all([self.latitude, self.longitude, self.timezone]):
            _LOGGER.error("Astronomy Cache not initialized with location")
            return False

        def _build_sync():
            try:
                sunrise, sunset, solar_noon = self._calculate_sunrise_sunset(
                    target_date, self.latitude, self.longitude, self.timezone
                )

                if not sunrise or not sunset or not solar_noon:
                    _LOGGER.warning(f"Could not calculate sun times for {target_date}")
                    return None

                daylight_hours = (sunset - sunrise).total_seconds() / 3600.0
                day_of_year = target_date.timetuple().tm_yday

                hourly_data = []
                for hour in range(24):
                    hour_time = datetime.combine(target_date, datetime.min.time())
                    hour_time = hour_time.replace(hour=hour, minute=30, tzinfo=self.timezone)

                    elevation, azimuth = self._calculate_sun_position(
                        hour_time, self.latitude, self.longitude
                    )

                    clear_sky_sr = self._calculate_clear_sky_solar_radiation(elevation, day_of_year)

                    group_results = None
                    if self._panel_groups:
                        total_theoretical, group_results = self._calculate_theoretical_pv_per_group(
                            clear_sky_sr, elevation, azimuth
                        )
                        theoretical_pv = total_theoretical
                    else:
                        theoretical_pv = self._calculate_theoretical_pv_output(
                            clear_sky_sr, system_capacity_kwp
                        )

                    hourly_data.append({
                        "hour": hour,
                        "elevation": elevation,
                        "azimuth": azimuth,
                        "clear_sky_radiation": clear_sky_sr,
                        "theoretical_max": theoretical_pv,
                        "sunrise": sunrise,
                        "sunset": sunset,
                        "solar_noon": solar_noon,
                        "daylight_hours": daylight_hours,
                        "group_results": group_results,  # Per-group POA for DB @zara
                    })

                return hourly_data

            except Exception as e:
                _LOGGER.error(f"Error building cache for {target_date}: {e}", exc_info=True)
                return None

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _build_sync)

        if not result:
            return False

        # Batch-Write: Alle 24 Stunden in einem einzigen DB-Commit
        batch_params = [
            (target_date, hour_data["hour"], hour_data["elevation"],
             hour_data["azimuth"], hour_data["clear_sky_radiation"],
             hour_data["theoretical_max"], hour_data["sunrise"],
             hour_data["sunset"], hour_data["solar_noon"],
             hour_data["daylight_hours"])
            for hour_data in result
        ]

        await self.db.executemany(
            """INSERT INTO astronomy_cache
               (cache_date, hour, sun_elevation_deg, sun_azimuth_deg,
                clear_sky_radiation_wm2, theoretical_max_kwh,
                sunrise, sunset, solar_noon, daylight_hours)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(cache_date, hour) DO UPDATE SET
                   sun_elevation_deg = excluded.sun_elevation_deg,
                   sun_azimuth_deg = excluded.sun_azimuth_deg,
                   clear_sky_radiation_wm2 = excluded.clear_sky_radiation_wm2,
                   theoretical_max_kwh = excluded.theoretical_max_kwh,
                   sunrise = excluded.sunrise,
                   sunset = excluded.sunset,
                   solar_noon = excluded.solar_noon,
                   daylight_hours = excluded.daylight_hours""",
            batch_params
        )

        # Save per-panel-group POA data @zara V16.1
        group_params = []
        for hour_data in result:
            group_results = hour_data.get("group_results")
            if group_results:
                for gr in group_results:
                    group_params.append((
                        target_date,
                        hour_data["hour"],
                        gr.name,
                        gr.power_kwp,
                        gr.azimuth_deg,
                        gr.tilt_deg,
                        gr.theoretical_kwh,
                        gr.poa_wm2,
                        gr.aoi_deg,
                    ))

        if group_params:
            await self.db.executemany(
                """INSERT INTO astronomy_cache_panel_groups
                   (cache_date, hour, group_name, power_kwp, azimuth_deg, tilt_deg,
                    theoretical_kwh, poa_wm2, aoi_deg)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(cache_date, hour, group_name) DO UPDATE SET
                       power_kwp = excluded.power_kwp,
                       azimuth_deg = excluded.azimuth_deg,
                       tilt_deg = excluded.tilt_deg,
                       theoretical_kwh = excluded.theoretical_kwh,
                       poa_wm2 = excluded.poa_wm2,
                       aoi_deg = excluded.aoi_deg""",
                group_params
            )

        return True

    async def rebuild_cache(
        self,
        system_capacity_kwp: float,
        start_date: Optional[date] = None,
        days_back: int = 30,
        days_ahead: int = 7,
    ) -> Dict:
        """Rebuild entire astronomy cache in database. @zara"""
        if start_date is None:
            start_date = datetime.now(self.timezone).date()

        _LOGGER.info(
            f"Rebuilding astronomy cache: {days_back} days back, "
            f"{days_ahead} days ahead from {start_date}"
        )

        await self.db.execute(
            """INSERT INTO astronomy_system_info
               (id, latitude, longitude, elevation_m, timezone, installed_capacity_kwp)
               VALUES (1, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   latitude = excluded.latitude,
                   longitude = excluded.longitude,
                   elevation_m = excluded.elevation_m,
                   timezone = excluded.timezone,
                   installed_capacity_kwp = excluded.installed_capacity_kwp,
                   updated_at = CURRENT_TIMESTAMP""",
            (self.latitude, self.longitude, self.elevation_m,
             str(self.timezone), system_capacity_kwp)
        )

        start_calc = start_date - timedelta(days=days_back)
        end_calc = start_date + timedelta(days=days_ahead)

        dates_to_process = []
        current_date = start_calc
        while current_date <= end_calc:
            dates_to_process.append(current_date)
            current_date += timedelta(days=1)

        _LOGGER.info(f"Processing {len(dates_to_process)} days in parallel...")

        results = await asyncio.gather(
            *[self.build_cache_for_date(d, system_capacity_kwp) for d in dates_to_process],
            return_exceptions=True
        )

        success_count = sum(1 for r in results if r is True)
        error_count = len(results) - success_count

        _LOGGER.info(f"Astronomy cache: {success_count}/{len(dates_to_process)} days processed")

        return {
            "total_days": len(dates_to_process),
            "success_count": success_count,
            "error_count": error_count,
        }

    async def get_day_data(self, target_date: date) -> Optional[Dict]:
        """Get astronomy data for a specific date from database. @zara"""
        rows = await self.db.fetchall(
            """SELECT hour, sun_elevation_deg, sun_azimuth_deg,
                      clear_sky_radiation_wm2, theoretical_max_kwh,
                      sunrise, sunset, solar_noon, daylight_hours
               FROM astronomy_cache
               WHERE cache_date = ?
               ORDER BY hour""",
            (target_date,)
        )

        if not rows:
            return None

        hourly = {}
        for row in rows:
            hourly[str(row[0])] = {
                "elevation_deg": row[1],
                "azimuth_deg": row[2],
                "clear_sky_solar_radiation_wm2": row[3],
                "theoretical_max_pv_kwh": row[4],
            }

        first_row = rows[0]
        return {
            "sunrise_local": first_row[5],
            "sunset_local": first_row[6],
            "solar_noon_local": first_row[7],
            "daylight_hours": first_row[8],
            "hourly": hourly,
        }

    async def get_hourly_data(self, target_date: date, target_hour: int) -> Optional[Dict]:
        """Get astronomy data for a specific hour from database. @zara"""
        row = await self.db.fetchone(
            """SELECT sun_elevation_deg, sun_azimuth_deg,
                      clear_sky_radiation_wm2, theoretical_max_kwh
               FROM astronomy_cache
               WHERE cache_date = ? AND hour = ?""",
            (target_date, target_hour)
        )

        if not row:
            return None

        return {
            "elevation_deg": row[0],
            "azimuth_deg": row[1],
            "clear_sky_solar_radiation_wm2": row[2],
            "theoretical_max_pv_kwh": row[3],
        }

    async def get_production_window(self, target_date: date) -> Optional[Tuple[datetime, datetime]]:
        """Get production window for a date from database. @zara"""
        row = await self.db.fetchone(
            "SELECT sunrise, sunset FROM astronomy_cache WHERE cache_date = ? LIMIT 1",
            (target_date,)
        )

        if not row:
            return None

        try:
            sunrise = datetime.fromisoformat(row[0])
            sunset = datetime.fromisoformat(row[1])
            start = sunrise - timedelta(minutes=30)
            end = sunset + timedelta(minutes=30)
            return start, end
        except (ValueError, TypeError) as e:
            _LOGGER.error(f"Error parsing production window: {e}")
            return None
