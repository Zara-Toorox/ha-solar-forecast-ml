import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from homeassistant.core import HomeAssistant

from .helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class SunGuard:

    def __init__(self, hass: HomeAssistant, buffer_hours: float = 1.0):
        self.hass = hass
        self.buffer_hours = buffer_hours
        self._cache: Optional[Tuple[datetime, datetime]] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=5)
        self._last_state: Optional[bool] = None
        self._fallback_start_hour = 5
        self._fallback_end_hour = 21

    def is_production_time(self) -> bool:
        now = dt_util.now()
        
        if self._cache and self._cache_time:
            cache_age = now - self._cache_time
            if cache_age < self._cache_duration:
                sunrise, sunset = self._cache
                is_production = sunrise <= now <= sunset
                self._log_state_change(is_production, "cache")
                return is_production
        
        sun_state = self.hass.states.get("sun.sun")
        
        if not sun_state or sun_state.state in ['unavailable', 'unknown']:
            is_production = self._fallback_start_hour <= now.hour < self._fallback_end_hour
            self._log_state_change(is_production, "fallback")
            return is_production
        
        next_rising = sun_state.attributes.get("next_rising")
        next_setting = sun_state.attributes.get("next_setting")
        
        if next_rising and next_setting:
            sunrise_dt = dt_util.parse_datetime(next_rising)
            sunset_dt = dt_util.parse_datetime(next_setting)
            
            if sunrise_dt and sunset_dt:
                if sun_state.state == "above_horizon":
                    sunrise_dt = sunrise_dt - timedelta(days=1)
                
                sunrise_buffered = sunrise_dt - timedelta(hours=self.buffer_hours)
                sunset_buffered = sunset_dt + timedelta(hours=self.buffer_hours)
                
                self._cache = (sunrise_buffered, sunset_buffered)
                self._cache_time = now
                
                is_production = sunrise_buffered <= now <= sunset_buffered
                self._log_state_change(is_production, "sun.sun", sunrise_buffered, sunset_buffered)
                return is_production
        
        is_production = self._fallback_start_hour <= now.hour < self._fallback_end_hour
        self._log_state_change(is_production, "fallback")
        return is_production

    def get_production_window(self) -> Tuple[datetime, datetime]:
        now = dt_util.now()
        sun_state = self.hass.states.get("sun.sun")
        
        if sun_state and sun_state.state not in ['unavailable', 'unknown']:
            next_rising = sun_state.attributes.get("next_rising")
            next_setting = sun_state.attributes.get("next_setting")
            
            if next_rising and next_setting:
                sunrise_dt = dt_util.parse_datetime(next_rising)
                sunset_dt = dt_util.parse_datetime(next_setting)
                
                if sunrise_dt and sunset_dt:
                    if sun_state.state == "above_horizon":
                        sunrise_dt = sunrise_dt - timedelta(days=1)
                    
                    sunrise_buffered = sunrise_dt - timedelta(hours=self.buffer_hours)
                    sunset_buffered = sunset_dt + timedelta(hours=self.buffer_hours)
                    return (sunrise_buffered, sunset_buffered)
        
        return (
            now.replace(hour=self._fallback_start_hour, minute=0, second=0, microsecond=0),
            now.replace(hour=self._fallback_end_hour, minute=0, second=0, microsecond=0)
        )

    def log_production_window(self) -> None:
        sunrise, sunset = self.get_production_window()
        _LOGGER.info(
            f"Ã¢Ëœâ‚¬Ã¯Â¸ÂÃ‚Â Sun Guard Produktionsfenster: "
            f"{sunrise.strftime('%H:%M')} - {sunset.strftime('%H:%M')} "
            f"(Buffer: {self.buffer_hours}h)"
        )

    def _log_state_change(
        self, 
        is_production: bool, 
        source: str,
        sunrise: Optional[datetime] = None,
        sunset: Optional[datetime] = None
    ) -> None:
        if self._last_state == is_production:
            return
        
        now = dt_util.now()
        
        if is_production:
            if sunrise and sunset:
                _LOGGER.info(
                    f"Ã°Å¸Å¸Â¢ DATENSAMMLUNG GESTARTET um {now.strftime('%H:%M:%S')} "
                    f"(Quelle: {source}, Fenster: {sunrise.strftime('%H:%M')}-{sunset.strftime('%H:%M')})"
                )
            else:
                _LOGGER.info(
                    f"Ã°Å¸Å¸Â¢ DATENSAMMLUNG GESTARTET um {now.strftime('%H:%M:%S')} "
                    f"(Quelle: {source})"
                )
        else:
            if sunset:
                _LOGGER.info(
                    f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â´ DATENSAMMLUNG PAUSIERT um {now.strftime('%H:%M:%S')} "
                    f"(Grund: AuÃƒÅ¸erhalb Produktionszeit, Quelle: {source}, Ende: {sunset.strftime('%H:%M')})"
                )
            else:
                _LOGGER.info(
                    f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â´ DATENSAMMLUNG PAUSIERT um {now.strftime('%H:%M:%S')} "
                    f"(Grund: AuÃƒÅ¸erhalb Produktionszeit, Quelle: {source})"
                )
        
        self._last_state = is_production
