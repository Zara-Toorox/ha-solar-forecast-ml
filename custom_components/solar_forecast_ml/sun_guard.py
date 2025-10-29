import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

from homeassistant.core import HomeAssistant
from .helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class SunGuard:
    def __init__(self, hass: HomeAssistant, buffer_hours: float = 2.0):
        self.hass = hass
        self.buffer_hours = buffer_hours
        self._cache: Optional[Tuple[datetime, datetime]] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=5)
        self._last_state: Optional[bool] = None
        self._fallback_start_hour = 5
        self._fallback_end_hour = 21
        self._last_day = None  # Neuer Tag-Tracker
        _LOGGER.info(f"SunGuard initialized with buffer of {self.buffer_hours} hours.")

    def is_production_time(self, hour_to_check: Optional[int] = None) -> bool:
        now = dt_util.now()
        today = now.date()

        # Invalidate cache at midnight
        if self._last_day != today:
            _LOGGER.debug("SunGuard: New day detected → cache invalidated")
            self._cache = None
            self._last_day = today

        check_time = now.replace(hour=hour_to_check, minute=0, second=0, microsecond=0) if hour_to_check is not None else now
        _LOGGER.debug(f"SunGuard.is_production_time: Checking → {check_time}")

        # === CACHE ===
        if self._cache and self._cache_time:
            cache_age = now - self._cache_time
            if cache_age < self._cache_duration:
                sunrise_buf, sunset_buf = self._cache
                is_prod = sunrise_buf <= check_time <= sunset_buf
                _LOGGER.debug(f"SunGuard (Cache): {sunrise_buf.strftime('%H:%M')} ≤ {check_time.strftime('%H:%M')} ≤ {sunset_buf.strftime('%H:%M')} → {is_prod}")
                return is_prod

        # === SUN ENTITY ===
        sun_state = self.hass.states.get("sun.sun")
        if not sun_state or sun_state.state in ['unavailable', 'unknown']:
            is_prod = self._fallback_start_hour <= check_time.hour < self._fallback_end_hour
            _LOGGER.debug(f"SunGuard (Fallback): {check_time.hour} in {self._fallback_start_hour}-{self._fallback_end_hour} → {is_prod}")
            return is_prod

        next_rising = sun_state.attributes.get("next_rising")
        next_setting = sun_state.attributes.get("next_setting")

        if not next_rising or not next_setting:
            is_prod = self._fallback_start_hour <= check_time.hour < self._fallback_end_hour
            _LOGGER.debug("SunGuard: Missing next_rising/setting → fallback")
            return is_prod

        try:
            next_rise_dt = dt_util.parse_datetime(next_rising)
            next_set_dt = dt_util.parse_datetime(next_setting)

            if not next_rise_dt or not next_set_dt:
                raise ValueError("Parse failed")

            # === HEUTIGER Sonnenaufgang ===
            if sun_state.state == "above_horizon":
                sunrise_today = next_rise_dt - timedelta(days=1)
            else:
                sunrise_today = next_rise_dt

            # === HEUTIGER Sonnenuntergang ===
            if sun_state.state == "below_horizon":
                sunset_today = next_set_dt - timedelta(days=1)
            else:
                sunset_today = next_set_dt

            # === SICHERSTELLEN: Beide sind HEUTE ===
            if sunrise_today.date() != today or sunset_today.date() != today:
                _LOGGER.debug(f"SunGuard: Calculated dates not today → fallback (sunrise: {sunrise_today.date()}, sunset: {sunset_today.date()})")
                is_prod = self._fallback_start_hour <= check_time.hour < self._fallback_end_hour
                return is_prod

            # === BUFFER ANWENDEN ===
            sunrise_buffered = sunrise_today - timedelta(hours=self.buffer_hours)
            sunset_buffered = sunset_today + timedelta(hours=self.buffer_hours)

            # === CACHE NUR MIT HEUTIGEM FENSTER ===
            self._cache = (sunrise_buffered, sunset_buffered)
            self._cache_time = now

            is_prod = sunrise_buffered <= check_time <= sunset_buffered
            _LOGGER.debug(
                f"SunGuard: {sunrise_buffered.strftime('%H:%M')} ≤ "
                f"{check_time.strftime('%H:%M')} ≤ {sunset_buffered.strftime('%H:%M')} → {is_prod}"
            )
            return is_prod

        except Exception as e:
            _LOGGER.warning(f"SunGuard: Error calculating window: {e}")
            return self._fallback_start_hour <= check_time.hour < self._fallback_end_hour

    def get_production_window(self) -> Tuple[datetime, datetime]:
        now = dt_util.now()
        today = now.date()
        sun_state = self.hass.states.get("sun.sun")

        if sun_state and sun_state.state not in ['unavailable', 'unknown']:
            next_rising = sun_state.attributes.get("next_rising")
            next_setting = sun_state.attributes.get("next_setting")
            if next_rising and next_setting:
                try:
                    next_rise = dt_util.parse_datetime(next_rising)
                    next_set = dt_util.parse_datetime(next_setting)
                    if next_rise and next_set:
                        sunrise = next_rise - timedelta(days=1) if sun_state.state == "above_horizon" else next_rise
                        sunset = next_set - timedelta(days=1) if sun_state.state == "below_horizon" else next_set
                        if sunrise.date() == today and sunset.date() == today:
                            return (
                                sunrise - timedelta(hours=self.buffer_hours),
                                sunset + timedelta(hours=self.buffer_hours)
                            )
                except:
                    pass

        return (
            now.replace(hour=self._fallback_start_hour, minute=0, second=0, microsecond=0),
            now.replace(hour=self._fallback_end_hour, minute=0, second=0, microsecond=0)
        )

    def log_production_window(self) -> None:
        sunrise, sunset = self.get_production_window()
        _LOGGER.info(f"SunGuard window: {sunrise.strftime('%H:%M')} - {sunset.strftime('%H:%M')} (Buffer: {self.buffer_hours}h)")