import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from homeassistant.components import recorder
from .data_manager import DataManager
from .const import ML_MODEL_VERSION

_LOGGER = logging.getLogger(__name__)


class SampleCollector:
    
    def __init__(
        self,
        hass: HomeAssistant,
        data_manager: DataManager,
        sun_guard=None
    ):
        self.hass = hass
        self.data_manager = data_manager
        self.sun_guard = sun_guard
        self._last_sample_hour: Optional[int] = None
        self._sample_lock = asyncio.Lock()
        self._forecast_cache: Dict[str, Any] = {}
        self.weather_entity: Optional[str] = None
        self.power_entity: Optional[str] = None
        self.temp_sensor: Optional[str] = None
        self.wind_sensor: Optional[str] = None
        self.rain_sensor: Optional[str] = None
        self.uv_sensor: Optional[str] = None
        self.lux_sensor: Optional[str] = None
    
    async def collect_sample(self, current_hour: int) -> None:
        if self.sun_guard and not self.sun_guard.is_production_time():
            _LOGGER.debug(
                f"â›” Stunde {current_hour}: Außerhalb Produktionszeit - "
                f"Sample Collection übersprungen"
            )
            return
        
        async with self._sample_lock:
            try:
                if self._last_sample_hour == current_hour:
                    _LOGGER.debug(
                        f"Sample für Stunde {current_hour} bereits gesammelt, überspringe"
                    )
                    return
                
                self._last_sample_hour = current_hour
                
                await self._collect_hourly_sample(current_hour)
                
            except Exception as e:
                _LOGGER.error(f"Hourly sample collection failed: {e}", exc_info=True)
    
    async def _collect_hourly_sample(self, current_hour: int) -> None:
        try:
            actual_kwh = await self._get_actual_production_for_hour(current_hour)
            if actual_kwh is None:
                _LOGGER.debug(f"Keine Produktion für Stunde {current_hour}")
                return
            
            daily_total = await self._get_daily_production_so_far()
            if daily_total is None or daily_total <= 0:
                _LOGGER.debug(f"Kein Tagesertrag verfügbar für Percentage-Berechnung")
                percentage = 0.0
            else:
                percentage = actual_kwh / daily_total
            
            weather_data = await self._get_current_weather_data()
            sensor_data = await self._collect_current_sensor_data()
            
            now = dt_util.utcnow()
            sample_time = now.replace(hour=current_hour, minute=0, second=0, microsecond=0)
            
            sample = {
                "timestamp": sample_time.isoformat(),
                "actual_kwh": round(actual_kwh, 4),
                "daily_total": round(daily_total, 4) if daily_total else 0.0,
                "percentage_of_day": round(percentage, 4),
                "weather_data": weather_data,
                "sensor_data": sensor_data,
                "model_version": ML_MODEL_VERSION
            }
            
            await self.data_manager.add_hourly_sample(sample)
            
            _LOGGER.info(
                f"✅ Hourly Sample gespeichert: {current_hour}:00 Uhr | "
                f"Actual={actual_kwh:.2f}kWh ({percentage*100:.1f}% des Tages), "
                f"Daily={daily_total:.2f}kWh"
            )
            
        except Exception as e:
            _LOGGER.error(f"Fehler beim Sammeln des hourly samples: {e}", exc_info=True)
    
    async def _get_actual_production_for_hour(self, hour: int) -> Optional[float]:
        if not self.power_entity:
            _LOGGER.debug("Kein power_entity konfiguriert")
            return None
        
        try:
            now = dt_util.utcnow()
            
            start_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(hours=1)
            
            if end_time > now:
                end_time = now
            
            history_list = await self.hass.async_add_executor_job(
                recorder.history.get_significant_states,
                self.hass,
                start_time,
                end_time,
                [self.power_entity],
                None,
                True
            )
            
            if not history_list or self.power_entity not in history_list:
                _LOGGER.debug(f"Keine History für {self.power_entity} von {start_time} bis {end_time}")
                return None
            
            states = history_list[self.power_entity]
            
            if not states:
                return 0.0
            
            total_wh = 0.0
            prev_time = start_time
            
            for state in states:
                try:
                    power_w = float(state.state)
                    state_time = state.last_updated
                    
                    if state_time > prev_time:
                        time_diff_hours = (state_time - prev_time).total_seconds() / 3600.0
                        wh = power_w * time_diff_hours
                        total_wh += wh
                    
                    prev_time = state_time
                    
                except (ValueError, TypeError):
                    continue
            
            if prev_time < end_time:
                try:
                    last_power = float(states[-1].state)
                    time_diff_hours = (end_time - prev_time).total_seconds() / 3600.0
                    wh = last_power * time_diff_hours
                    total_wh += wh
                except (ValueError, TypeError):
                    pass
            
            kwh = total_wh / 1000.0
            
            _LOGGER.debug(
                f"Produktion Stunde {hour}: {kwh:.4f} kWh "
                f"({len(states)} Datenpunkte)"
            )
            
            return kwh
            
        except Exception as e:
            _LOGGER.error(f"Fehler beim Abrufen der Production History: {e}")
            return None
    
    async def _get_daily_production_so_far(self) -> Optional[float]:
        if not self.power_entity:
            return None
        
        try:
            now = dt_util.utcnow()
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            history_list = await self.hass.async_add_executor_job(
                recorder.history.get_significant_states,
                self.hass,
                start_of_day,
                now,
                [self.power_entity],
                None,
                True
            )
            
            if not history_list or self.power_entity not in history_list:
                return None
            
            states = history_list[self.power_entity]
            
            if not states:
                return 0.0
            
            total_wh = 0.0
            prev_time = start_of_day
            
            for state in states:
                try:
                    power_w = float(state.state)
                    state_time = state.last_updated
                    
                    if state_time > prev_time:
                        time_diff_hours = (state_time - prev_time).total_seconds() / 3600.0
                        wh = power_w * time_diff_hours
                        total_wh += wh
                    
                    prev_time = state_time
                    
                except (ValueError, TypeError):
                    continue
            
            if prev_time < now:
                try:
                    last_power = float(states[-1].state)
                    time_diff_hours = (now - prev_time).total_seconds() / 3600.0
                    wh = last_power * time_diff_hours
                    total_wh += wh
                except (ValueError, TypeError):
                    pass
            
            kwh = total_wh / 1000.0
            
            _LOGGER.debug(f"Tagesertrag bisher: {kwh:.2f} kWh")
            
            return kwh
            
        except Exception as e:
            _LOGGER.error(f"Fehler beim Abrufen des Tagesertrags: {e}")
            return None
    
    async def _collect_current_sensor_data(self) -> Dict[str, Any]:
        sensor_data = {}
        
        try:
            if self.temp_sensor:
                state = self.hass.states.get(self.temp_sensor)
                if state and state.state not in ['unavailable', 'unknown', 'none', None]:
                    try:
                        sensor_data['temperature'] = float(state.state)
                    except (ValueError, TypeError):
                        pass
            
            if self.wind_sensor:
                state = self.hass.states.get(self.wind_sensor)
                if state and state.state not in ['unavailable', 'unknown', 'none', None]:
                    try:
                        sensor_data['wind_speed'] = float(state.state)
                    except (ValueError, TypeError):
                        pass
            
            if self.rain_sensor:
                state = self.hass.states.get(self.rain_sensor)
                if state and state.state not in ['unavailable', 'unknown', 'none', None]:
                    try:
                        sensor_data['rain'] = float(state.state)
                    except (ValueError, TypeError):
                        pass
            
            if self.uv_sensor:
                state = self.hass.states.get(self.uv_sensor)
                if state and state.state not in ['unavailable', 'unknown', 'none', None]:
                    try:
                        sensor_data['uv_index'] = float(state.state)
                    except (ValueError, TypeError):
                        pass
            
            if self.lux_sensor:
                state = self.hass.states.get(self.lux_sensor)
                if state and state.state not in ['unavailable', 'unknown', 'none', None]:
                    try:
                        sensor_data['lux'] = float(state.state)
                    except (ValueError, TypeError):
                        pass
            
        except Exception as e:
            _LOGGER.debug(f"Fehler beim Sammeln von Sensor-Daten: {e}")
        
        return sensor_data
    
    async def _get_current_weather_data(self) -> Dict[str, Any]:
        try:
            if not self.weather_entity:
                return self._get_default_weather()
            
            weather_state = self.hass.states.get(self.weather_entity)
            if not weather_state:
                return self._get_default_weather()
            
            attrs = weather_state.attributes
            
            return {
                'temperature': float(attrs.get('temperature', 15.0)),
                'humidity': float(attrs.get('humidity', 60.0)),
                'cloudiness': float(attrs.get('cloud_coverage', 50.0)),
                'wind_speed': float(attrs.get('wind_speed', 5.0)),
                'pressure': float(attrs.get('pressure', 1013.0))
            }
            
        except Exception as e:
            _LOGGER.debug(f"Fehler beim Abrufen von Weather-Daten: {e}")
            return self._get_default_weather()
    
    def _get_default_weather(self) -> Dict[str, Any]:
        return {
            'temperature': 15.0,
            'humidity': 60.0,
            'cloudiness': 50.0,
            'wind_speed': 5.0,
            'pressure': 1013.0
        }
    
    def set_forecast_cache(self, cache: Dict[str, Any]) -> None:
        self._forecast_cache = cache
    
    def configure_entities(
        self,
        weather_entity: Optional[str] = None,
        power_entity: Optional[str] = None,
        temp_sensor: Optional[str] = None,
        wind_sensor: Optional[str] = None,
        rain_sensor: Optional[str] = None,
        uv_sensor: Optional[str] = None,
        lux_sensor: Optional[str] = None
    ) -> None:
        self.weather_entity = weather_entity
        self.power_entity = power_entity
        self.temp_sensor = temp_sensor
        self.wind_sensor = wind_sensor
        self.rain_sensor = rain_sensor
        self.uv_sensor = uv_sensor
        self.lux_sensor = lux_sensor
