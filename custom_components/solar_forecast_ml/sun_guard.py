"""
Sun Guard for Solar Forecast ML Integration.
Determines the active solar production window based on sunrise/sunset
using Astral library calculations, applying a buffer. Provides fallback
logic if Astral calculation fails.

Copyright (C) 2025 Zara-Toorox
"""
import logging
from datetime import datetime, timedelta, UTC
from typing import Optional, Tuple

from homeassistant.core import HomeAssistant
from homeassistant.util.dt import get_time_zone  # <-- ECHTER HA-Helper!
from .helpers import SafeDateTimeUtil as dt_util
from .const import SUN_BUFFER_HOURS, FALLBACK_PRODUCTION_START_HOUR, FALLBACK_PRODUCTION_END_HOUR

try:
    from astral.location import LocationInfo
    from astral import sun as astral_sun
    _ASTRAL_AVAILABLE = True
except ImportError:
    _ASTRAL_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)


class SunGuard:
    def __init__(self, hass: HomeAssistant, buffer_hours: float = SUN_BUFFER_HOURS):
        self.hass = hass
        self.buffer_hours = buffer_hours
        self._cached_window_utc: Optional[Tuple[datetime, datetime]] = None
        self._cache_update_time_utc: Optional[datetime] = None
        self._cache_max_age = timedelta(minutes=15)
        self._fallback_start_hour: int = FALLBACK_PRODUCTION_START_HOUR
        self._fallback_end_hour: int = FALLBACK_PRODUCTION_END_HOUR
        self._last_checked_day_local: Optional[datetime.date] = None

        if not _ASTRAL_AVAILABLE:
            _LOGGER.warning("Astral library not found. SunGuard will rely on fallback hours.")

        _LOGGER.info(
            f"SunGuard initialized with +/- {self.buffer_hours:.1f}h buffer. "
            f"Fallback: {self._fallback_start_hour:02d}:00 - {self._fallback_end_hour:02d}:00 (Local). "
            f"Astral: {_ASTRAL_AVAILABLE}"
        )

    def _get_ha_location_observer(self):
        if not _ASTRAL_AVAILABLE:
            raise ImportError("Astral library is not installed.")

        try:
            latitude = self.hass.config.latitude
            longitude = self.hass.config.longitude
            elevation = self.hass.config.elevation
            tz_str = self.hass.config.time_zone  # z. B. "Europe/Berlin"

            # --- FIX: ECHTER HA-Helper für Timezone ---
            tzinfo_obj = get_time_zone(tz_str) if tz_str else UTC
            if tzinfo_obj is None:
                _LOGGER.error(f"Invalid timezone string: {tz_str}")
                raise ValueError("Invalid timezone configuration")

            loc_info = LocationInfo("HomeAssistant", "Region", tz_str or "UTC", latitude, longitude)
            observer = loc_info.observer
            observer.elevation = elevation
            return observer, tzinfo_obj

        except Exception as e:
            _LOGGER.error(f"Could not retrieve/process HA location for Astral: {e}")
            raise ValueError("Invalid HA location configuration for Astral") from e

    async def _calculate_and_cache_window(self) -> Optional[Tuple[datetime, datetime]]:
        if not _ASTRAL_AVAILABLE:
            return None

        now_utc = dt_util.utcnow()
        now_local = dt_util.as_local(now_utc)
        today_local_date = now_local.date()

        try:
            observer, tzinfo_obj = self._get_ha_location_observer()

            def _get_sun_times():
                return (
                    astral_sun.sunrise(observer, date=today_local_date, tzinfo=tzinfo_obj),
                    astral_sun.sunset(observer, date=today_local_date, tzinfo=tzinfo_obj)
                )

            sunrise_local, sunset_local = await self.hass.async_add_executor_job(_get_sun_times)

            sunrise_utc = sunrise_local.astimezone(UTC)
            sunset_utc = sunset_local.astimezone(UTC)

            if sunrise_utc >= sunset_utc:
                _LOGGER.error("Sunrise after sunset – invalid Astral result.")
                return None

            buffer = timedelta(hours=self.buffer_hours)
            window_start = sunrise_utc - buffer
            window_end = sunset_utc + buffer

            self._cached_window_utc = (window_start, window_end)
            self._cache_update_time_utc = now_utc
            self._last_checked_day_local = today_local_date

            _LOGGER.debug(f"Astral window: {window_start} - {window_end} UTC")
            return self._cached_window_utc

        except Exception as e:
            _LOGGER.error(f"Astral calculation failed: {e}", exc_info=True)
            self._cached_window_utc = None
            return None

    async def get_production_window_utc(self) -> Tuple[datetime, datetime]:
        now_utc = dt_util.utcnow()
        now_local_date = dt_util.as_local(now_utc).date()

        # Cache prüfen
        if (self._cached_window_utc
            and self._cache_update_time_utc
            and (now_utc - self._cache_update_time_utc < self._cache_max_age)
            and self._last_checked_day_local == now_local_date):
            return self._cached_window_utc

        # Astral versuchen
        if _ASTRAL_AVAILABLE:
            window = await self._calculate_and_cache_window()
            if window:
                return window

        # --- Fallback ---
        _LOGGER.warning("Using fallback production window.")
        now_local = dt_util.as_local(now_utc)

        try:
            tz_str = self.hass.config.time_zone
            # --- FIX: ECHTER HA-Helper ---
            tzinfo_obj = get_time_zone(tz_str) if tz_str else (now_local.tzinfo or UTC)
            if tzinfo_obj is None:
                _LOGGER.warning(f"Invalid timezone '{tz_str}', using UTC")
                tzinfo_obj = UTC

            start_local = now_local.replace(
                hour=self._fallback_start_hour, minute=0, second=0, microsecond=0
            ).replace(tzinfo=tzinfo_obj)
            end_local = now_local.replace(
                hour=self._fallback_end_hour, minute=0, second=0, microsecond=0
            ).replace(tzinfo=tzinfo_obj)

            start_utc = start_local.astimezone(UTC)
            end_utc = end_local.astimezone(UTC)

            if start_utc >= end_utc:
                _LOGGER.warning("Fallback window invalid, using full day.")
                start_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                end_utc = start_utc + timedelta(days=1) - timedelta(seconds=1)

            self._cached_window_utc = (start_utc, end_utc)
            self._cache_update_time_utc = now_utc
            self._last_checked_day_local = now_local_date

            _LOGGER.debug(f"Fallback window: {start_utc.strftime('%H:%M')} - {end_utc.strftime('%H:%M')} UTC")
            return (start_utc, end_utc)

        except Exception as e:
            _LOGGER.critical(f"Even fallback failed: {e}", exc_info=True)
            start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1) - timedelta(seconds=1)
            return (start, end)

    async def is_production_time(self, time_to_check_utc: Optional[datetime] = None) -> bool:
        check_time = time_to_check_utc or dt_util.utcnow()
        if check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=UTC)
        elif check_time.tzinfo != UTC:
            check_time = check_time.astimezone(UTC)

        try:
            start, end = await self.get_production_window_utc()
            return start <= check_time <= end
        except Exception as e:
            _LOGGER.error(f"Failed to check production time: {e}")
            return True  # Fail-open

    async def log_production_window(self) -> None:
        try:
            start_utc, end_utc = await self.get_production_window_utc()
            start_local = dt_util.as_local(start_utc)
            end_local = dt_util.as_local(end_utc)

            method = ("calculated via Astral"
                      if _ASTRAL_AVAILABLE and self._cached_window_utc
                      and self._cached_window_utc[0].hour != self._fallback_start_hour
                      else "fallback hours")

            _LOGGER.info(
                f"Current production window ({method}, Local): "
                f"{start_local.strftime('%Y-%m-%d %H:%M:%S')} - {end_local.strftime('%Y-%m-%d %H:%M:%S')} "
                f"(±{self.buffer_hours:.1f}h)"
            )
        except Exception as e:
            _LOGGER.error(f"Failed to log window: {e}")