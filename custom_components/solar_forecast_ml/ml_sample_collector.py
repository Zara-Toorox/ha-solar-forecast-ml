"""
Sample Collector for Solar Forecast ML
Handles hourly data collection, historical backfill, and state retrieval.
FIXED: Backfill now fetches historical sensor/weather data, not current.
REFACTORED VERSION: Modular structure with separate manager classes
Version 6.0.0 - Modular Architecture

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

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from homeassistant.core import HomeAssistant
from homeassistant.components import recorder
from homeassistant.components.recorder import history
from homeassistant.util import dt as dt_util

from .data_manager import DataManager
from .const import ML_MODEL_VERSION
from .helpers import SafeDateTimeUtil as dt_util_safe

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
        self.solar_yield_today: Optional[str] = None
        self.temp_sensor: Optional[str] = None
        self.wind_sensor: Optional[str] = None
        self.rain_sensor: Optional[str] = None
        self.uv_sensor: Optional[str] = None
        self.lux_sensor: Optional[str] = None

    async def collect_sample(self, current_hour: int, base_timestamp: Optional[datetime] = None) -> None:
        if not self.weather_entity or not self.solar_yield_today:
            _LOGGER.warning(
                f"Collector noch nicht vollständig konfiguriert (fehlende Kern-Entitäten). "
                f"Überspringe Sample für Stunde {current_hour}."
            )
            return

        if self.sun_guard and not base_timestamp: # Backfill ignoriert sun_guard
            try:
                if not self.sun_guard.is_production_time(hour_to_check=current_hour):
                    _LOGGER.debug(f"Hour {current_hour}: Outside production time - Skipped")
                    return
            except TypeError:
                if not self.sun_guard.is_production_time():
                    _LOGGER.debug(f"Hour {current_hour}: Outside production time - Skipped")
                    return

        async with self._sample_lock:
            try:
                # Sperre gilt nur für Standard-Sammeln, nicht für Backfill
                if self._last_sample_hour == current_hour and not base_timestamp:
                    _LOGGER.debug(f"Hour {current_hour}: Already collected")
                    return
                
                if not base_timestamp:
                    self._last_sample_hour = current_hour

                await self._collect_hourly_sample(current_hour, base_timestamp)
            except Exception as e:
                _LOGGER.error(f"Collection failed: {e}", exc_info=True)

    async def _collect_hourly_sample(self, current_hour: int, base_timestamp: Optional[datetime] = None) -> None:
        try:
            # === START PATCH 4: ML-Kern-Upgrade (Lag Features) ===
            # Hole Produktion der *aktuellen* Stunde
            actual_kwh = await self._get_actual_production_for_hour(current_hour, base_timestamp)
            
            # Hole Produktion der *letzten* Stunde
            # Wir rufen (current_hour - 1) auf. 
            # _get_actual_production_for_hour behandelt den Tageswechsel (Stunde -1 wird zu 23 Uhr am Vortag, wenn base_timestamp genutzt wird)
            last_hour_kwh = await self._get_actual_production_for_hour(current_hour - 1, base_timestamp)
            
            if actual_kwh is None:
                _LOGGER.debug(f"No data for hour {current_hour}")
                return
            
            # Wenn last_hour_kwh None ist, setzen wir es auf 0.0 für das Training
            if last_hour_kwh is None:
                _LOGGER.debug(f"No data for last hour ({current_hour - 1}), using 0.0")
                last_hour_kwh = 0.0
            # === ENDE PATCH 4 ===

            daily_total = await self._get_daily_production_so_far() or 0.0
            percentage = actual_kwh / daily_total if daily_total > 0 else 0.0

            # === FIX: Zeitstempel an Abruffunktionen übergeben ===
            weather_data = await self._get_current_weather_data(base_timestamp, current_hour)
            sensor_data = await self._collect_current_sensor_data(base_timestamp, current_hour)
            # === ENDE FIX ===
            
            # === START PATCH 4: ML-Kern-Upgrade (Lag Features) ===
            # Füge das neue Feature zum sensor_data dict hinzu
            sensor_data["production_last_hour"] = last_hour_kwh
            # === ENDE PATCH 4 ===

            if base_timestamp:
                # Nutze das übergebene Datum und setze die Stunde
                sample_time = base_timestamp.replace(hour=current_hour, minute=0, second=0, microsecond=0)
            else:
                # Standardverhalten: Nimm den aktuellen Tag
                now = dt_util_safe.utcnow()
                sample_time = now.replace(hour=current_hour, minute=0, second=0, microsecond=0)

            sample = {
                "timestamp": sample_time.isoformat(),
                "actual_kwh": round(actual_kwh, 4),
                "daily_total": round(daily_total, 4),
                "percentage_of_day": round(percentage, 4),
                "weather_data": weather_data,
                "sensor_data": sensor_data, # Enthält jetzt 'production_last_hour'
                "model_version": ML_MODEL_VERSION
            }

            await self.data_manager.add_hourly_sample(sample)
            _LOGGER.info(f"Saved: {sample_time.strftime('%Y-%m-%d %H')}:00 | {actual_kwh:.2f}kWh (LastHour: {last_hour_kwh:.2f}kWh)")

        except Exception as e:
            _LOGGER.error(f"Sample error: {e}", exc_info=True)

    async def _safe_parse_yield(self, state_obj: Optional[Any]) -> Optional[float]:
        if not state_obj or state_obj.state in ['unavailable', 'unknown', 'none', None]:
            _LOGGER.debug(f"Zustand ist ungültig (unavailable/none): {state_obj}")
            return None
        
        try:
            return float(state_obj.state)
        except (ValueError, TypeError):
            try:
                cleaned_state = str(state_obj.state).split(" ")[0].replace(",", ".")
                val = float(cleaned_state)
                _LOGGER.debug(f"Status '{state_obj.state}' wurde zu {val} normalisiert")
                return val
            except (ValueError, TypeError):
                _LOGGER.warning(f"Konnte normalisierten Status nicht in Zahl umwandeln: '{state_obj.state}'")
                return None

    async def _get_state_at_timestamp(self, entity_id: str, timestamp: datetime) -> Optional[Any]:
        _LOGGER.debug(f"Getting state for {entity_id} at or before {timestamp}")
        
        start_window = timestamp - timedelta(hours=1)
        
        try:
            history_list = await self.hass.async_add_executor_job(
                lambda: history.state_changes_during_period(
                    self.hass,
                    start_window,
                    timestamp,
                    entity_id,
                    no_attributes=False,  # === FIX: Muss False sein, um Attribute zu erhalten ===
                    include_start_time_state=False
                )
            )
            
            if entity_id in history_list and history_list[entity_id]:
                # Gib den letzten Zustand *im* Fenster zurück
                return history_list[entity_id][-1]
            
            _LOGGER.warning(f"No history found for {entity_id} in window {start_window} to {timestamp}")
            return None

        except Exception as e:
            _LOGGER.error(f"Error getting history for {entity_id}: {e}", exc_info=True)
            return None

    async def _get_actual_production_for_hour(self, hour: int, base_timestamp: Optional[datetime] = None) -> Optional[float]:
        if not self.solar_yield_today:
            _LOGGER.warning("solar_yield_today entity ID is not configured. Skipping hourly sample.")
            return None

        try:
            if base_timestamp:
                now_utc = base_timestamp
            else:
                now_utc = dt_util_safe.utcnow()

            # === START PATCH 4: Anpassung für (Stunde - 1) ===
            # Wenn Stunde -1 (z.B. 0 Uhr -> -1) übergeben wird, berechne die Zeit korrekt
            target_time = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=hour)
            
            start_of_hour = target_time.replace(minute=0, second=0, microsecond=0)
            end_of_hour = start_of_hour + timedelta(hours=1)
            # === ENDE PATCH 4 ===

            rec = recorder.get_instance(self.hass)
            if not rec or not rec.is_running:
                _LOGGER.warning("Recorder is not available or running.")
                return None

            end_state = await self._get_state_at_timestamp(self.solar_yield_today, end_of_hour)
            end_yield = await self._safe_parse_yield(end_state)
            
            if end_yield is None:
                _LOGGER.debug(f"No valid end state for {self.solar_yield_today} at {end_of_hour}")
                return None

            start_state = await self._get_state_at_timestamp(self.solar_yield_today, start_of_hour)
            start_yield = await self._safe_parse_yield(start_state)

            if start_yield is None:
                _LOGGER.debug(f"No valid start state. Using end state {end_yield} as delta.")
                return end_yield

            if end_yield < start_yield:
                _LOGGER.debug(f"Sensor reset detected (End: {end_yield} < Start: {start_yield}). Using {end_yield} as delta.")
                return end_yield

            delta = end_yield - start_yield
            return max(0.0, delta)

        except Exception as e:
            _LOGGER.error(f"Delta error in _get_actual_production_for_hour: {e}", exc_info=True)
            return None

    async def _get_daily_production_so_far(self) -> Optional[float]:
        if not self.solar_yield_today:
            return 0.0
        state = self.hass.states.get(self.solar_yield_today)
        return await self._safe_parse_yield(state)

    async def _collect_current_sensor_data(self, base_timestamp: Optional[datetime] = None, hour: Optional[int] = None) -> Dict[str, Any]:
        data = {}
        
        query_time = None
        if base_timestamp and hour is not None:
            # Für History: Nimm den Zustand zur Mitte der Stunde
            # === START PATCH 4: Anpassung für (Stunde - 1) ===
            query_time_base = base_timestamp.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=hour)
            query_time = query_time_base.replace(minute=30, second=0, microsecond=0)
            # === ENDE PATCH 4 ===

        for entity, key in [
            (self.temp_sensor, 'temperature'),
            (self.wind_sensor, 'wind_speed'),
            (self.rain_sensor, 'rain'),
            (self.uv_sensor, 'uv_index'),
            (self.lux_sensor, 'lux')
        ]:
            if entity:
                state_obj = None
                if query_time:
                    # Hole historischen Status
                    state_obj = await self._get_state_at_timestamp(entity, query_time)
                else:
                    # Hole aktuellen Status
                    state_obj = self.hass.states.get(entity)
                
                # Verwende safe_parse, um den Wert zu extrahieren
                val = await self._safe_parse_yield(state_obj)
                # Setze Wert oder 0.0 als Fallback
                data[key] = val if val is not None else 0.0
        return data

    async def _get_current_weather_data(self, base_timestamp: Optional[datetime] = None, hour: Optional[int] = None) -> Dict[str, Any]:
        if not self.weather_entity:
            return self._get_default_weather()

        ws = None
        if base_timestamp and hour is not None:
            # Hole historischen Status (Mitte der Stunde)
            # === START PATCH 4: Anpassung für (Stunde - 1) ===
            query_time_base = base_timestamp.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=hour)
            query_time = query_time_base.replace(minute=30, second=0, microsecond=0)
            # === ENDE PATCH 4 ===
            ws = await self._get_state_at_timestamp(self.weather_entity, query_time)
        else:
            # Hole aktuellen Status
            ws = self.hass.states.get(self.weather_entity)

        if not ws:
            _LOGGER.warning(f"Konnte Wetter-Status nicht abrufen (History: {bool(base_timestamp)})")
            return self._get_default_weather()
        
        # 'attributes' sollte jetzt da sein (dank no_attributes=False)
        if not hasattr(ws, 'attributes'):
             _LOGGER.warning(f"Wetter-Objekt hat keine Attribute: {ws}")
             return self._get_default_weather()

        a = ws.attributes
        
        cc = a.get('cloud_coverage', 50.0)
        if isinstance(cc, str):
            try: cc = float(cc.replace("%", ""))
            except: cc = 50.0
            
        return {
            # Sicherer Abruf: Prüfe 'Timperature' (Tippfehler im Original) und 'temperature'
            'temperature': float(a.get('Timperature', a.get('temperature', 15.0))),
            'humidity': float(a.get('humidity', 60.0)),
            'cloudiness': float(cc),
            'wind_speed': float(a.get('wind_speed', 5.0)),
            'pressure': float(a.get('pressure', 1013.0))
        }

    def _get_default_weather(self):
        return {'temperature': 15.0, 'humidity': 60.0, 'cloudiness': 50.0, 'wind_speed': 5.0, 'pressure': 1013.0}

    def set_forecast_cache(self, cache: Dict[str, Any]) -> None:
        self._forecast_cache = cache

    def configure_entities(
        self,
        weather_entity: Optional[str] = None,
        power_entity: Optional[str] = None,
        solar_yield_today: Optional[str] = None,
        temp_sensor: Optional[str] = None,
        wind_sensor: Optional[str] = None,
        rain_sensor: Optional[str] = None,
        uv_sensor: Optional[str] = None,
        lux_sensor: Optional[str] = None
    ) -> None:
        self.weather_entity = weather_entity
        self.power_entity = power_entity
        self.solar_yield_today = solar_yield_today
        self.temp_sensor = temp_sensor
        self.wind_sensor = wind_sensor
        self.rain_sensor = rain_sensor
        self.uv_sensor = uv_sensor
        self.lux_sensor = lux_sensor

    def get_all_entity_ids(self) -> List[str]:
        entities = set()
        if self.weather_entity:
            entities.add(self.weather_entity)
        if self.power_entity:
            entities.add(self.power_entity)
        if self.solar_yield_today:
            entities.add(self.solar_yield_today)
        if self.temp_sensor:
            entities.add(self.temp_sensor)
        if self.wind_sensor:
            entities.add(self.wind_sensor)
        if self.rain_sensor:
            entities.add(self.rain_sensor)
        if self.uv_sensor:
            entities.add(self.uv_sensor)
        if self.lux_sensor:
            entities.add(self.lux_sensor)
        if hasattr(self, 'sun_guard') and self.sun_guard and isinstance(self.sun_guard, str):
            entities.add(self.sun_guard)
        if not entities:
            _LOGGER.debug("get_all_entity_ids(): Keine Entity-IDs konfiguriert.")
        return list(entities)