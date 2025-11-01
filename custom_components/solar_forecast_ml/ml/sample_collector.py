"""
ML Sample Collector for Solar Forecast ML Integration.
Orchestrates data fetching, processing, and state updates using a modular architecture.

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
from datetime import datetime, timedelta, timezone # <-- timezone importiert
from typing import Dict, Any, Optional, List, Tuple
from homeassistant.core import HomeAssistant, State
from ..core.helpers import SafeDateTimeUtil
from homeassistant.util import dt as dt_util # <-- dt_util wird konsistent verwendet

# --- IMPORTE HIER ENTFERNT ---
# from homeassistant.components import recorder (Verschoben)
# from homeassistant.components.recorder import history (Verschoben)
# --- ENDE ENTFERNUNG ---

from ..data.manager import DataManager
from ..const import ML_MODEL_VERSION
from ..forecast.weather_calculator import WeatherCalculator # Import fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Condition-Mapping

_LOGGER = logging.getLogger(__name__)


class SampleCollector:

    def __init__(
        self,
        hass: HomeAssistant,
        data_manager: DataManager
    ):
        self.hass = hass
        self.data_manager = data_manager
        self._sample_lock = asyncio.Lock()
        self._forecast_cache: Dict[str, Any] = {}
        self.weather_entity: Optional[str] = None
        self.power_entity: Optional[str] = None
        self.temp_sensor: Optional[str] = None
        self.wind_sensor: Optional[str] = None
        self.rain_sensor: Optional[str] = None
        self.uv_sensor: Optional[str] = None
        self.lux_sensor: Optional[str] = None
        self.humidity_sensor: Optional[str] = None
        
        # Helper class for weather condition mapping
        self._weather_calculator = WeatherCalculator()
    
    async def collect_sample(self, target_local_hour: int) -> None:
        """
        Sammelt die Daten fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r die angegebene ZIELSTUNDE (Lokalzeit).
        This function is called by MLPredictor, der die korrekte
        target hour (the hour that just ended) ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼bergibt.
        """
        # 1. Sonnen-Check (Verwendet UTC intern, daher sicher)
        # 2. Lock und Duplikat-Check (PERSISTENTE PRÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œFUNG)
        async with self._sample_lock:
            try:
                last_collected = await self.data_manager.get_last_collected_hour()
                
                if last_collected == target_local_hour:
                    _LOGGER.debug(
                        f"Sample fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Stunde {target_local_hour} (Lokal) bereits persistent gesammelt, ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼berspringe."
                    )
                    return

                # 3. Sammle die Daten fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r die ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼bergebene Zielstunde
                await self._collect_hourly_sample(target_local_hour)

                # 4. Speichere erfolgreich gesammelte Stunde persistent
                await self.data_manager.set_last_collected_hour(target_local_hour)
                _LOGGER.debug(f"Persistenter Status: Stunde {target_local_hour} als gesammelt markiert.")

            except Exception as e:
                _LOGGER.error(f"Hourly sample collection fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Stunde {target_local_hour} fehlgeschlagen: {e}", exc_info=True)
    
    # --- (FEHLERBEHEBUNG) NEUE HILFSMETHODE fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Zeitgewichteten Durchschnitt ---
    async def _calculate_time_weighted_average(
        self,
        entity_id: str,
        start_time_utc: datetime,
        end_time_utc: datetime,
        attribute: Optional[str] = None
    ) -> Optional[float]:
        """
        Berechnet den zeitgewichteten Durchschnitt eines Sensors (oder Attributs) 
        innerhalb eines Zeitfensters.
        """
        
        # +++ IMPORT HIER EINGEFÃƒÆ’Ã†â€™Ãƒâ€¦Ã¢â‚¬Å“GT +++
        try:
            from homeassistant.components.recorder import history
        except ImportError:
            _LOGGER.error("Recorder component not available. Cannot calculate Time Weighted Average.")
            return None
        # +++ ENDE EINFÃƒÆ’Ã†â€™Ãƒâ€¦Ã¢â‚¬Å“GUNG +++
        
        _LOGGER.debug(f"Berechne TWA fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r {entity_id} (Attribut: {attribute}) von {start_time_utc} bis {end_time_utc}")
        
        # 1. Hole alle ZustÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤nde im Zeitfenster
        try:
            history_list = await self.hass.async_add_executor_job(
                history.get_significant_states,
                self.hass,
                start_time_utc,
                end_time_utc,
                [entity_id],
                None,
                True, # include_start_time_state = True
                True  # significant_changes_only = True (wichtig!)
            )
            
            if not history_list or entity_id not in history_list:
                _LOGGER.warning(f"Keine History fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r TWA von {entity_id} gefunden.")
                return None
            
            states = history_list[entity_id]
            if not states:
                _LOGGER.debug(f"History fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r TWA von {entity_id} ist leer.")
                return None
                
        except Exception as e:
            _LOGGER.error(f"Fehler beim Abrufen der History fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r TWA von {entity_id}: {e}")
            return None

        # 2. Berechne zeitgewichteten Durchschnitt
        total_value_duration = 0.0
        total_duration_sec = 0.0
        
        # Finde den initialen Wert (letzter Wert vor oder bei start_time)
        initial_value = None
        for state in states:
            if state.last_updated <= start_time_utc:
                try:
                    val_str = state.attributes.get(attribute) if attribute else state.state
                    initial_value = float(val_str)
                except (ValueError, TypeError, AttributeError):
                    continue # Suche weiter nach einem gÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ltigen numerischen Wert
            else:
                break # Zustand ist bereits nach start_time

        if initial_value is None:
            _LOGGER.warning(f"Konnte keinen initialen Wert fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r TWA von {entity_id} finden. Beginne mit erstem Wert im Fenster.")
            # Suche den ersten gÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ltigen Wert *im* Fenster
            for state in states:
                if state.last_updated > start_time_utc:
                     try:
                         val_str = state.attributes.get(attribute) if attribute else state.state
                         initial_value = float(val_str)
                         break
                     except (ValueError, TypeError, AttributeError):
                         continue
            if initial_value is None:
                 _LOGGER.error(f"Keine gÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ltigen numerischen Werte fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r TWA von {entity_id} im gesamten Zeitraum gefunden.")
                 return None

        prev_value = initial_value
        prev_time = start_time_utc
        
        # Iteriere ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ber alle ZustÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤nde *nach* start_time
        start_index_integration = 0
        while start_index_integration < len(states) and states[start_index_integration].last_updated <= start_time_utc:
            start_index_integration += 1

        for state in states[start_index_integration:]:
            try:
                current_time = min(state.last_updated, end_time_utc)
                duration_sec = (current_time - prev_time).total_seconds()

                if duration_sec > 0:
                    total_value_duration += prev_value * duration_sec
                    total_duration_sec += duration_sec
                
                # Aktualisiere prev_value fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r nÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤chsten Schritt
                val_str = state.attributes.get(attribute) if attribute else state.state
                prev_value = float(val_str)
                prev_time = current_time

                if prev_time >= end_time_utc:
                    break
                    
            except (ValueError, TypeError, AttributeError):
                # UngÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ltiger Zustand (z.B. 'unknown'), fahre mit vorigem Wert fort
                _LOGGER.debug(f"UngÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ltiger Zustand '{val_str}' bei TWA fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r {entity_id}, verwende vorigen Wert.")
                prev_time = min(state.last_updated, end_time_utc) # Zeit muss aber fortschreiten
            except Exception as e_loop:
                _LOGGER.warning(f"Fehler in TWA-Schleife: {e_loop}")
                prev_time = min(state.last_updated, end_time_utc)

        # Letztes Segment von prev_time bis end_time_utc
        if prev_time < end_time_utc:
            duration_sec = (end_time_utc - prev_time).total_seconds()
            total_value_duration += prev_value * duration_sec
            total_duration_sec += duration_sec
            
        if total_duration_sec == 0:
            _LOGGER.debug(f"TWA fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r {entity_id}: Keine Dauer (0s), verwende initial_value.")
            return initial_value # Nur ein Wert im Zeitfenster

        average = total_value_duration / total_duration_sec
        _LOGGER.debug(f"TWA fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r {entity_id} (Attribut: {attribute}) = {average:.2f}")
        return average

    # --- (FEHLERBEHEBUNG) NEUE HILFSMETHODE fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r dominanten Zustand ---
    async def _get_dominant_condition(
        self,
        start_time_utc: datetime, 
        end_time_utc: datetime,
        entity_id: str
    ) -> str:
        """
        Ermittelt den Wetter-Zustand (Condition), der im Zeitfenster
        am lÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ngsten angedauert hat.
        """
        
        # +++ IMPORT HIER EINGEFÃƒÆ’Ã†â€™Ãƒâ€¦Ã¢â‚¬Å“GT +++
        try:
            from homeassistant.components.recorder import history
        except ImportError:
            _LOGGER.error("Recorder component not available. Cannot get dominant condition.")
            return "unknown"
        # +++ ENDE EINFÃƒÆ’Ã†â€™Ãƒâ€¦Ã¢â‚¬Å“GUNG +++

        _LOGGER.debug(f"Ermittle dominante Condition fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r {entity_id} von {start_time_utc} bis {end_time_utc}")
        try:
            history_list = await self.hass.async_add_executor_job(
                history.get_significant_states,
                self.hass, start_time_utc, end_time_utc, [entity_id],
                None, True, True
            )
            
            if not history_list or entity_id not in history_list:
                _LOGGER.warning(f"Keine History fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Condition von {entity_id} gefunden.")
                return 'unknown'
            
            states = history_list[entity_id]
            if not states:
                _LOGGER.debug(f"History fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Condition von {entity_id} ist leer.")
                return 'unknown'

            durations: Dict[str, float] = {}
            
            # Initialen Zustand finden
            prev_state_str = 'unknown'
            for state in states:
                if state.last_updated <= start_time_utc:
                    prev_state_str = state.state
                else:
                    break # Zustand ist bereits nach start_time
            
            prev_time = start_time_utc

            start_index_integration = 0
            while start_index_integration < len(states) and states[start_index_integration].last_updated <= start_time_utc:
                start_index_integration += 1
                
            for state in states[start_index_integration:]:
                current_time = min(state.last_updated, end_time_utc)
                duration_sec = (current_time - prev_time).total_seconds()

                if duration_sec > 0 and prev_state_str not in ['unknown', 'unavailable']:
                    durations[prev_state_str] = durations.get(prev_state_str, 0.0) + duration_sec
                
                prev_state_str = state.state
                prev_time = current_time
                if prev_time >= end_time_utc:
                    break

            # Letztes Segment
            if prev_time < end_time_utc:
                duration_sec = (end_time_utc - prev_time).total_seconds()
                if duration_sec > 0 and prev_state_str not in ['unknown', 'unavailable']:
                    durations[prev_state_str] = durations.get(prev_state_str, 0.0) + duration_sec

            if not durations:
                _LOGGER.warning(f"Keine gÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ltigen Conditions fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r {entity_id} im Zeitraum gefunden.")
                return 'unknown'

            # Finde den Zustand mit der lÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ngsten Dauer
            dominant_condition = max(durations, key=durations.get)
            _LOGGER.debug(f"Dominante Condition fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r {entity_id} ist '{dominant_condition}' (Dauern: {durations})")
            return dominant_condition

        except Exception as e:
            _LOGGER.error(f"Fehler bei Ermittlung der dominanten Condition: {e}", exc_info=True)
            return 'unknown'

    async def _get_historical_forecast_data(
        self,
        target_time_utc: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Holt historische Forecast-Daten aus dem weather_forecast_cache.json
        fÃƒÂ¼r die angegebene Ziel-Zeit.
        
        Args:
            target_time_utc: Zielstunde in UTC
            
        Returns:
            Dict mit Forecast-Daten oder None falls nicht gefunden
        """
        try:
            # Lade Forecast-Cache
            cache = await self.data_manager.load_weather_forecast_cache()
            if not cache or not cache.get("forecast_hours"):
                _LOGGER.debug("Forecast-Cache leer oder nicht verfÃƒÂ¼gbar")
                return None
            
            forecast_hours = cache.get("forecast_hours", [])
            
            # Suche Forecast fÃƒÂ¼r Ziel-Stunde (mit 30min Toleranz)
            target_str = target_time_utc.isoformat()
            tolerance = timedelta(minutes=30)
            
            for entry in forecast_hours:
                entry_dt_str = entry.get("datetime")
                if not entry_dt_str:
                    continue
                
                try:
                    entry_dt = datetime.fromisoformat(entry_dt_str.replace('Z', '+00:00'))
                    time_diff = abs((entry_dt - target_time_utc).total_seconds())
                    
                    if time_diff <= tolerance.total_seconds():
                        # Forecast gefunden - extrahiere relevante Daten
                        weather_data = {
                            'temperature': entry.get('temperature', 15.0),
                            'humidity': entry.get('humidity', 60.0),
                            'cloud_cover': entry.get('cloud_coverage', 50.0),
                            'wind_speed': entry.get('wind_speed', 5.0),
                            'pressure': entry.get('pressure', 1013.0),
                            'condition': entry.get('condition', 'unknown')
                        }
                        
                        _LOGGER.debug(
                            f"Historischer Forecast gefunden fÃƒÂ¼r {target_time_utc.isoformat()}: "
                            f"temp={weather_data['temperature']}Ã‚Â°C, cloud={weather_data['cloud_cover']}%"
                        )
                        return weather_data
                        
                except (ValueError, TypeError) as e:
                    _LOGGER.debug(f"Fehler beim Parsen von Forecast-Eintrag: {e}")
                    continue
            
            _LOGGER.debug(f"Kein passender Forecast fÃƒÂ¼r {target_time_utc.isoformat()} im Cache gefunden")
            return None
            
        except Exception as e:
            _LOGGER.error(f"Fehler beim Abrufen historischer Forecast-Daten: {e}", exc_info=True)
            return None

    # --- (FEHLERBEHEBUNG & V1) VÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œLLIG NEU AUFGEBAUTE METHODE ---
    async def _get_historical_average_states(
        self, 
        start_time_utc: datetime, 
        end_time_utc: datetime
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Holt historische Wetterdaten primÃƒÂ¤r aus Forecast-Cache,
        Fallback auf Recorder TWA fÃƒÂ¼r Sensoren und Weather-Entity.
        
        Returns:
            (weather_data, sensor_data)
        """
        _LOGGER.debug(f"Rufe historische Daten ab fÃƒÂ¼r {start_time_utc} bis {end_time_utc}")
        
        weather_data = self._get_default_weather()
        sensor_data = self._get_default_sensor_data()
        
        # PRIORITÃƒâ€žT 1: Historischer Forecast aus Cache (Mitte des Zeitfensters)
        mid_time_utc = start_time_utc + (end_time_utc - start_time_utc) / 2
        forecast_data = await self._get_historical_forecast_data(mid_time_utc)
        
        if forecast_data:
            # Nutze Forecast-Daten fÃƒÂ¼r weather_data
            weather_data.update(forecast_data)
            _LOGGER.debug(f"Verwende historischen Forecast fÃƒÂ¼r Weather-Daten")
            forecast_available = True
        else:
            _LOGGER.debug(f"Kein historischer Forecast verfÃƒÂ¼gbar, nutze TWA fÃƒÂ¼r Weather-Entity")
            forecast_available = False
        
        # PRIORITÃƒâ€žT 2: Sensor-Daten via TWA (immer durchfÃƒÂ¼hren)
        # Definiere Sensor-EntitÃƒÂ¤ten (keine Weather-Entity-Attribute mehr!)
        sensor_entities: List[Tuple[str, Optional[str], str, Dict]] = [
            (self.temp_sensor, None, 'temperature', sensor_data),
            (self.wind_sensor, None, 'wind_speed', sensor_data),
            (self.rain_sensor, None, 'rain', sensor_data),
            (self.uv_sensor, None, 'uv_index', sensor_data),
            (self.lux_sensor, None, 'lux', sensor_data),
            (self.humidity_sensor, None, 'humidity', sensor_data),
        ]
        
        tasks = []
        
        # TWA-Tasks nur fÃƒÂ¼r konfigurierte Sensoren
        for entity_id, attr, key, target_dict in sensor_entities:
            if entity_id:
                tasks.append(self._calculate_time_weighted_average(entity_id, start_time_utc, end_time_utc, attr))
            else:
                tasks.append(asyncio.sleep(0, result=None))
        
        # Wenn kein Forecast verfÃƒÂ¼gbar: TWA fÃƒÂ¼r Weather-Entity-Attribute als Fallback
        if not forecast_available and self.weather_entity:
            weather_fallback_attrs = [
                ('temperature', 'temperature'),
                ('humidity', 'humidity'),
                ('wind_speed', 'wind_speed'),
                ('pressure', 'pressure'),
                ('cloud_coverage', 'cloud_cover'),
            ]
            for attr, key in weather_fallback_attrs:
                tasks.append(self._calculate_time_weighted_average(self.weather_entity, start_time_utc, end_time_utc, attr))
        
        # Condition: Immer via dominante Methode (falls Weather-Entity vorhanden)
        if self.weather_entity and not forecast_available:
            tasks.append(self._get_dominant_condition(start_time_utc, end_time_utc, self.weather_entity))
        else:
            tasks.append(asyncio.sleep(0, result=None))
        
        # Alle Tasks parallel ausfÃƒÂ¼hren
        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            _LOGGER.error(f"Fehler bei Abruf der TWA-Werte: {e}")
            # Falls Forecast vorhanden, gib diesen zurÃƒÂ¼ck, sonst Defaults
            return weather_data, sensor_data
        
        # Ergebnisse zuordnen
        # Sensor-Daten (erste 6 Tasks)
        for i, (entity_id, attr, key, target_dict) in enumerate(sensor_entities):
            if i < len(results) and results[i] is not None:
                target_dict[key] = max(0.0, results[i])
        
        # Weather-Entity Fallback (falls kein Forecast)
        if not forecast_available:
            offset = len(sensor_entities)
            weather_attrs = ['temperature', 'humidity', 'wind_speed', 'pressure', 'cloud_cover']
            for i, key in enumerate(weather_attrs):
                idx = offset + i
                if idx < len(results) and results[idx] is not None:
                    weather_data[key] = max(0.0, results[idx])
            
            # Condition (letzter Task)
            if len(results) > offset + len(weather_attrs):
                condition_result = results[offset + len(weather_attrs)]
                if condition_result:
                    weather_data['condition'] = condition_result
        
        _LOGGER.debug(f"Historische Wetterdaten: {weather_data}")
        _LOGGER.debug(f"Historische Sensordaten: {sensor_data}")
        return weather_data, sensor_data


    # --- (Verbesserung 1) ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œBERARBEITETE METHODE ---
    async def _collect_hourly_sample(self, target_local_hour: int) -> None:
        """Sammelt die Daten fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r die angegebene Zielstunde (Lokalzeit)
           unter Verwendung von historischen Recorder-Daten."""
        try:
            # 1. Definiere den UTC-Zeitraum fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r die Zielstunde
            start_time_utc, end_time_utc, sample_time_local = self._get_utc_times_for_hour(target_local_hour)
            
            if start_time_utc is None or end_time_utc is None or sample_time_local is None:
                _LOGGER.warning(f"Konnte UTC-Zeitfenster fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r lokale Stunde {target_local_hour} nicht bestimmen. Sample ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼bersprungen.")
                return

            # 2. Hole Ist-Wert (Riemann-Summe) fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r den UTC-Zeitraum
            actual_kwh = await self._perform_riemann_integration(start_time_utc, end_time_utc)
            if actual_kwh is None:
                _LOGGER.warning(f"Konnte Ist-Produktion (Riemann) fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Stunde {target_local_hour} (Lokal) nicht abrufen (None). Sample ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼bersprungen.")
                return # Skip sample if essential data missing

            # 3. Hole Tagesgesamtwert bis zum Ende der Zielstunde (Ende des UTC-Zeitraums)
            daily_total = await self._get_daily_production_so_far(end_time_utc)
            if daily_total is None:
                _LOGGER.warning(f"Konnte Tagesgesamtwert bis Stunde {target_local_hour} nicht abrufen. Setze auf 0 fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Sample.")
                daily_total = 0.0 # Setze auf 0, wenn Abruf fehlschlÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤gt

            percentage = 0.0
            if daily_total > 0.01: 
                percentage = max(0.0, actual_kwh) / daily_total
            elif actual_kwh <= 0.01 and daily_total <= 0.01:
                percentage = 0.0 

            # 4. (NEU) Hole historische Durchschnittsdaten fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Wetter und Sensoren
            weather_data, sensor_data = await self._get_historical_average_states(start_time_utc, end_time_utc)

            # 5. Sample zusammenstellen und speichern
            sample = {
                "timestamp": sample_time_local.isoformat(), # Speichere als LOKALEN ISO String
                "actual_kwh": round(actual_kwh, 4),
                "daily_total": round(daily_total, 4), 
                "percentage_of_day": round(percentage, 4),
                "weather_data": weather_data,
                "sensor_data": sensor_data,
                "model_version": ML_MODEL_VERSION
            }

            await self.data_manager.add_hourly_sample(sample)

            _LOGGER.info(
                f"Hourly Sample (HISTORICAL) gespeichert: {sample_time_local.strftime('%Y-%m-%d %H')}:00 local | "
                f"Actual={actual_kwh:.2f}kWh ({percentage*100:.1f}% des Tages), "
                f"TagesTotal={daily_total:.2f}kWh, "
                f"Wetter-Temp={weather_data.get('temperature'):.1f}C, "
                f"Wetter-Cloud={weather_data.get('cloud_cover'):.1f}%"
            )

        except Exception as e:
            _LOGGER.error(f"Fehler beim Sammeln des (historischen) hourly samples fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Stunde {target_local_hour}: {e}", exc_info=True)

    # --- (Verbesserung 1) NEUE HILFSMETHODE ---
    def _get_utc_times_for_hour(self, target_local_hour: int) -> Tuple[Optional[datetime], Optional[datetime], Optional[datetime]]:
        """
        Berechnet das UTC-Start/Ende-Fenster und den lokalen Sample-Zeitstempel 
        fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r eine gegebene lokale Zielstunde.
        """
        try:
            now_local = SafeDateTimeUtil.as_local(SafeDateTimeUtil.utcnow())
            
            # Nimm den Tag der aktuellen lokalen Zeit
            # und setze die Stunde auf die Zielstunde
            start_time_local_base = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Wenn die Zielstunde (z.B. 23) grÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¶ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¸er ist als die aktuelle Stunde (z.B. 0),
            # bedeutet das, dass der Trigger nach Mitternacht lief und wir den Vortag meinen.
            if target_local_hour > now_local.hour:
                start_time_local_base -= timedelta(days=1)
                _LOGGER.debug(f"Zielstunde {target_local_hour} liegt vor aktueller Stunde {now_local.hour}. Verwende Vortag: {start_time_local_base.date()}")
            
            start_time_local = start_time_local_base.replace(hour=target_local_hour)
            end_time_local = start_time_local + timedelta(hours=1)

            # PrÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼fen, ob der gesamte Zeitraum in der Vergangenheit liegt
            if end_time_local > now_local:
                 _LOGGER.warning(f"Versuch, Daten fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r zukÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼nftige/aktuelle Stunde {target_local_hour} (bis {end_time_local}) zu sammeln. ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œberspringe.")
                 return None, None, None # Kann keine zukÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼nftigen Daten sammeln

            # Konvertiere die LOKALEN Zeiten nach UTC fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r die Recorder-Abfrage
            start_time_utc = start_time_local.astimezone(timezone.utc)
            end_time_utc = end_time_local.astimezone(timezone.utc)
            
            # Der Sample-Zeitstempel ist der LOKALE Startzeitpunkt
            sample_time_local = start_time_local

            return start_time_utc, end_time_utc, sample_time_local
            
        except ValueError as e:
            _LOGGER.warning(f"Konnte lokale Startzeit fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Stunde {target_local_hour} nicht erstellen (evtl. Zeitumstellung?): {e}")
            return None, None, None
        except Exception as tz_err:
            _LOGGER.error(f"Fehler bei Zeitzonenkonvertierung fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Stunde {target_local_hour}: {tz_err}")
            return None, None, None

    async def _safe_parse_yield(self, state_obj: Optional[Any]) -> Optional[float]:
        if not state_obj or state_obj.state in ['unavailable', 'unknown', 'none', None]:
            return None
        try:
            val = float(state_obj.state)
            return val if val >= 0 else 0.0 
        except (ValueError, TypeError):
            try:
                cleaned_state = str(state_obj.state).split(" ")[0].replace(",", ".")
                val = float(cleaned_state)
                _LOGGER.debug(f"Status '{state_obj.state}' wurde zu {val} normalisiert")
                return val if val >= 0 else 0.0 
            except (ValueError, TypeError):
                _LOGGER.warning(f"Konnte normalisierten Status nicht in Zahl umwandeln: '{state_obj.state}'")
                return None

    # =========================================================================
    # Riemann-Summe (UnverÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ndert)
    # =========================================================================
    async def _perform_riemann_integration(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[float]:
        
        # +++ IMPORT HIER EINGEFÃƒÆ’Ã†â€™Ãƒâ€¦Ã¢â‚¬Å“GT +++
        try:
            from homeassistant.components.recorder import history
        except ImportError:
            _LOGGER.error("Recorder component not available. Cannot perform Riemann integration.")
            return None
        # +++ ENDE EINFÃƒÆ’Ã†â€™Ãƒâ€¦Ã¢â‚¬Å“GUNG +++

        if not self.power_entity:
            _LOGGER.debug("Kein power_entity konfiguriert fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Riemann-Summe")
            return None

        _LOGGER.debug(f"Starte Riemann-Summe fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r {self.power_entity} von {start_time} bis {end_time}")

        try:
            if start_time.tzinfo is None or start_time.tzinfo.utcoffset(start_time) is None:
                _LOGGER.error("Riemann: start_time muss timezone-aware (UTC) sein.")
                start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time.tzinfo is None or end_time.tzinfo.utcoffset(end_time) is None:
                _LOGGER.error("Riemann: end_time muss timezone-aware (UTC) sein.")
                end_time = end_time.replace(tzinfo=timezone.utc) 

            history_list = await self.hass.async_add_executor_job(
                history.get_significant_states,
                self.hass,
                start_time,
                end_time,
                [self.power_entity],
                None,
                True, 
                True  
            )

            if not history_list or self.power_entity not in history_list:
                _LOGGER.warning(f"Keine History fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r {self.power_entity} von {start_time} bis {end_time} gefunden.")
                current_state = self.hass.states.get(self.power_entity)
                if current_state is None:
                    _LOGGER.error(f"Power entity {self.power_entity} nicht in Home Assistant gefunden!")
                    return None 
                else:
                    _LOGGER.warning(f"Sensor {self.power_entity} existiert, aber liefert keine History im Zeitraum. Ergebnis ist 0.0 kWh.")
                    return 0.0 

            states = history_list[self.power_entity]

            if not states:
                _LOGGER.debug(f"History fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r {self.power_entity} in Zeitraum ist leer.")
                return 0.0

            total_wh = 0.0
            initial_state_value = 0.0
            initial_state_time = start_time
            found_initial = False
            for state in states:
                if state.last_updated <= start_time:
                    try:
                        initial_state_value = max(0.0, float(state.state))
                        initial_state_time = state.last_updated 
                        found_initial = True
                    except (ValueError, TypeError):
                        continue 
                else:
                    break 

            if not found_initial:
                _LOGGER.warning(f"Konnte keinen gÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ltigen Zustand vor/bei {start_time} fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Riemann-Start finden. Beginne mit 0W.")
            prev_power = initial_state_value
            prev_time = start_time 

            _LOGGER.debug(f"Riemann Start: Initialer prev_power = {prev_power}W (basierend auf Zustand um {initial_state_time})")

            start_index_integration = 0
            while start_index_integration < len(states) and states[start_index_integration].last_updated <= start_time:
                start_index_integration += 1

            _LOGGER.debug(f"Riemann Integration beginnt bei Index {start_index_integration} (Zeit > {start_time})")

            for state in states[start_index_integration:]:
                try:
                    state_time = state.last_updated
                    current_end_time = min(state_time, end_time)
                    time_diff_seconds = (current_end_time - prev_time).total_seconds()
                    
                    if time_diff_seconds <= 1e-6: 
                        _LOGGER.debug(f"ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œberspringe Riemann Schritt: Zeitdifferenz ist Null oder negativ bei {current_end_time}.")
                        prev_time = current_end_time
                        try: prev_power = max(0.0, float(state.state))
                        except (ValueError, TypeError): pass
                        continue

                    time_diff_hours = time_diff_seconds / 3600.0
                    wh = prev_power * time_diff_hours
                    total_wh += wh
                    _LOGGER.debug(f"Riemann Schritt: von {prev_time.strftime('%H:%M:%S.%f')} bis {current_end_time.strftime('%H:%M:%S.%f')} ({time_diff_seconds:.1f}s) mit {prev_power:.1f}W -> {wh:.2f}Wh dazu (Total: {total_wh:.2f}Wh)")

                    prev_time = current_end_time

                    try:
                         current_power = max(0.0, float(state.state))
                         prev_power = current_power 
                    except (ValueError, TypeError):
                         _LOGGER.debug(f"UngÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ltiger Zustand '{state.state}' bei {state_time}. Verwende vorherigen Wert {prev_power}W weiter.")

                    if prev_time >= end_time:
                        _LOGGER.debug(f"Riemann erreicht/ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼berschreitet Endzeit {end_time}.")
                        break

                except Exception as loop_err:
                     _LOGGER.error(f"Fehler in Riemann-Schleife bei Zeit {state.last_updated}: {loop_err}", exc_info=True)
                     if hasattr(state, 'last_updated'):
                         prev_time = min(state.last_updated, end_time)
                     continue

            if prev_time < end_time:
                time_diff_seconds = (end_time - prev_time).total_seconds()
                if time_diff_seconds > 1e-6:
                    time_diff_hours = time_diff_seconds / 3600.0
                    wh = prev_power * time_diff_hours 
                    total_wh += wh
                    _LOGGER.debug(f"Riemann Abschluss: von {prev_time.strftime('%H:%M:%S.%f')} bis {end_time.strftime('%H:%M:%S.%f')} ({time_diff_seconds:.1f}s) mit {prev_power:.1f}W -> {wh:.2f}Wh dazu (Final Total: {total_wh:.2f}Wh)")

            kwh = max(0.0, total_wh / 1000.0)
            _LOGGER.debug(f"Riemann-Summe Ergebnis: {kwh:.4f} kWh")
            return kwh

        except Exception as e:
            _LOGGER.error(f"Kritischer Fehler bei Riemann-Integration fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r {self.power_entity}: {e}", exc_info=True)
            return None

    # (UnverÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ndert)
    async def _get_actual_production_for_hour(self, target_local_hour: int) -> Optional[float]:
        _LOGGER.debug(f"Abruf der Ist-Produktion fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Stunde {target_local_hour} (Lokal)...")
        start_time_utc, end_time_utc, _ = self._get_utc_times_for_hour(target_local_hour)
        
        if start_time_utc is None or end_time_utc is None:
            _LOGGER.warning(f"Konnte UTC-Zeitfenster fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Stunde {target_local_hour} nicht bestimmen.")
            return None

        _LOGGER.debug(f"Frage Recorder ab fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Stunde {target_local_hour} Lokal: "
                      f"UTC-Zeitraum: {start_time_utc.isoformat()} bis {end_time_utc.isoformat()}")

        kwh = await self._perform_riemann_integration(start_time_utc, end_time_utc)

        if kwh is not None:
             _LOGGER.debug(
                f"Produktion Stunde {target_local_hour} (Lokal) [UTC: {start_time_utc.strftime('%H:%M')}-{end_time_utc.strftime('%H:%M')}] (Riemann): {kwh:.4f} kWh"
            )
        else:
             _LOGGER.warning(f"Riemann-Integration fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Stunde {target_local_hour} (Lokal) fehlgeschlagen.")

        return kwh
    
    # (UnverÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ndert, auÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¸er Parameter-Typ)
    async def _get_daily_production_so_far(self, end_of_hour_utc: datetime) -> Optional[float]:
        """Holt die Tagesproduktion (bisher) via Riemann-Integration bis zum Ende der Zielstunde (UTC)."""
        
        # Bestimme Start des Tages (Lokalzeit) basierend auf der Endzeit
        end_of_hour_local = SafeDateTimeUtil.as_local(end_of_hour_utc)
        start_of_day_local = end_of_hour_local.replace(hour=0, minute=0, second=0, microsecond=0)

        # Konvertiere nach UTC
        try:
            start_of_day_utc = start_of_day_local.astimezone(timezone.utc)
        except Exception as tz_err:
             _LOGGER.error(f"Fehler bei Zeitzonenkonvertierung fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r Tagesproduktion: {tz_err}")
             return None

        _LOGGER.debug(f"Berechne Tagesproduktion bis Ende Stunde {end_of_hour_local.hour} (Lokal): "
                      f"UTC-Zeitraum: {start_of_day_utc.isoformat()} bis {end_of_hour_utc.isoformat()}")

        kwh = await self._perform_riemann_integration(start_of_day_utc, end_of_hour_utc)

        if kwh is not None:
            _LOGGER.debug(f"Tagesertrag bis Ende Stunde {end_of_hour_local.hour} (Riemann): {kwh:.2f} kWh")
        else:
            _LOGGER.warning(f"Abruf des Tagesertrags (bis Stunde {end_of_hour_local.hour}) via Riemann fehlgeschlagen.")

        return kwh

    # --- (Verbesserung 1) ENTFERNT ---
    # async def _collect_current_sensor_data(self) -> Dict[str, Any]:
    # async def _get_current_weather_data(self) -> Dict[str, Any]:

    # --- (Verbesserung 1) NEUE HILFSMETHODEN ---
    def _get_default_weather(self) -> Dict[str, Any]:
        """Gibt Standard-Wetterwerte zurÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ck."""
        return {
            'temperature': 15.0, 'humidity': 60.0, 'cloud_cover': 50.0,
            'wind_speed': 5.0, 'pressure': 1013.0, 'condition': 'unknown'
        }

    def _get_default_sensor_data(self) -> Dict[str, Any]:
        """Gibt Standard-Sensorwerte zurÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ck."""
        return {
            'temperature': 0.0, 'wind_speed': 0.0, 'rain': 0.0,
            'uv_index': 0.0, 'lux': 0.0, 'humidity': 0.0
        }
    # --- ENDE ---

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
        lux_sensor: Optional[str] = None,
        humidity_sensor: Optional[str] = None, 
        solar_yield_today: Optional[str] = None 
    ) -> None:
        """Konfiguriert die vom Collector verwendeten EntitÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ts-IDs."""
        _LOGGER.debug("Konfiguriere EntitÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ten im SampleCollector...")
        self.weather_entity = weather_entity
        self.power_entity = power_entity
        self.temp_sensor = temp_sensor
        self.wind_sensor = wind_sensor
        self.rain_sensor = rain_sensor
        self.uv_sensor = uv_sensor
        self.lux_sensor = lux_sensor
        self.humidity_sensor = humidity_sensor 
        _LOGGER.debug(f"SampleCollector Entities: Weather='{weather_entity}', Power='{power_entity}', "
                      f"Temp='{temp_sensor}', Wind='{wind_sensor}', Rain='{rain_sensor}', "
                      f"UV='{uv_sensor}', Lux='{lux_sensor}', Humidity='{humidity_sensor}'")