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
from typing import Dict, Any, Optional
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util # <-- dt_util wird konsistent verwendet
from homeassistant.components import recorder
from homeassistant.components.recorder import history
from .data_manager import DataManager
from .const import ML_MODEL_VERSION
# from .helpers import SafeDateTimeUtil as dt_util_safe # Nicht verwendeter Import

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
        # _last_sample_hour entfernt - wird jetzt persistent in model_state.json gespeichert
        self._sample_lock = asyncio.Lock()
        self._forecast_cache: Dict[str, Any] = {}
        self.weather_entity: Optional[str] = None
        self.power_entity: Optional[str] = None
        self.temp_sensor: Optional[str] = None
        self.wind_sensor: Optional[str] = None
        self.rain_sensor: Optional[str] = None
        self.uv_sensor: Optional[str] = None
        self.lux_sensor: Optional[str] = None
        self.humidity_sensor: Optional[str] = None # <-- Aus Block 8
    
    # --- KORREKTUR: Persistente Stundenverfolgung + Weather Attribute Validierung ---
    async def collect_sample(self, target_local_hour: int) -> None:
        """
        Sammelt die Daten für die angegebene ZIELSTUNDE (Lokalzeit).
        Diese Funktion wird vom MLPredictor aufgerufen, der die korrekte
        Zielstunde (die Stunde, die gerade zu Ende gegangen ist) übergibt.
        """
        # 1. Sonnen-Check (Verwendet UTC intern, daher sicher)
        if self.sun_guard:
            is_prod_time = False
            # Erstelle einen Zeitstempel *innerhalb* der Zielstunde für den Check
            # z.B. wenn target_local_hour = 13, prüfe 13:30 Lokalzeit
            try:
                now_local = dt_util.as_local(dt_util.utcnow())
                check_time_local = now_local.replace(hour=target_local_hour, minute=30, second=0, microsecond=0)
                check_time_utc = check_time_local.astimezone(timezone.utc)

                if asyncio.iscoroutinefunction(self.sun_guard.is_production_time):
                    is_prod_time = await self.sun_guard.is_production_time(check_time_utc)
                else:
                    is_prod_time = self.sun_guard.is_production_time(check_time_utc)
            except Exception as sg_err:
                 _LOGGER.warning(f"Fehler beim Aufruf von sun_guard.is_production_time für Stunde {target_local_hour}: {sg_err}")
                 is_prod_time = True # Im Zweifel weitermachen

            if not is_prod_time:
                _LOGGER.debug(
                    f"Zielstunde {target_local_hour} (Lokal) liegt außerhalb der Produktionszeit laut SunGuard. Sample Collection übersprungen."
                )
                return

        # 2. Lock und Duplikat-Check (PERSISTENTE PRÜFUNG)
        async with self._sample_lock:
            try:
                # Prüfe persistente last_collected_hour aus model_state.json
                last_collected = await self.data_manager.get_last_collected_hour()
                
                if last_collected == target_local_hour:
                    _LOGGER.debug(
                        f"Sample für Stunde {target_local_hour} (Lokal) bereits persistent gesammelt, überspringe."
                    )
                    return

                # 3. Sammle die Daten für die übergebene Zielstunde
                await self._collect_hourly_sample(target_local_hour)

                # 4. Speichere erfolgreich gesammelte Stunde persistent
                await self.data_manager.set_last_collected_hour(target_local_hour)
                _LOGGER.debug(f"Persistenter Status: Stunde {target_local_hour} als gesammelt markiert.")

            except Exception as e:
                _LOGGER.error(f"Hourly sample collection für Stunde {target_local_hour} fehlgeschlagen: {e}", exc_info=True)
                # NICHT persistent speichern bei Fehler, so kann es später erneut versucht werden
    # --- ENDE KORREKTUR ---

    async def _collect_hourly_sample(self, target_local_hour: int) -> None:
        """Sammelt die Daten fÃƒÂ¼r die angegebene Zielstunde (Lokalzeit)."""
        try:
            # Hole Ist-Wert fÃƒÂ¼r die Zielstunde (lokal)
            actual_kwh = await self._get_actual_production_for_hour(target_local_hour)
            if actual_kwh is None:
                _LOGGER.warning(f"Konnte Ist-Produktion fÃƒÂ¼r Stunde {target_local_hour} (Lokal) nicht abrufen (None). Sample ÃƒÂ¼bersprungen.")
                return # Skip sample if essential data missing

            # Hole Tagesgesamtwert bis *jetzt* (Ende der Zielstunde)
            daily_total = await self._get_daily_production_so_far(target_local_hour)
            if daily_total is None:
                _LOGGER.warning(f"Konnte Tagesgesamtwert bis Stunde {target_local_hour} nicht abrufen. Setze auf 0 fÃƒÂ¼r Sample.")
                daily_total = 0.0 # Setze auf 0, wenn Abruf fehlschlÃƒÂ¤gt

            # Berechne prozentualen Anteil sicher
            percentage = 0.0
            if daily_total > 0.01: # Vermeide Division durch sehr kleine Zahlen
                percentage = max(0.0, actual_kwh) / daily_total # Stelle sicher, dass actual_kwh >= 0 ist
            elif actual_kwh <= 0.01 and daily_total <= 0.01:
                percentage = 0.0 # Korrekt, wenn beides 0 ist

            # Hole Wetter- und Sensordaten zum *Zeitpunkt der Sammlung* (ca. HH:02)
            # Dies ist ein Kompromiss, ideal wÃƒÂ¤ren Daten von HH:30, aber das ist komplexer.
            weather_data = await self._get_current_weather_data()
            sensor_data = await self._collect_current_sensor_data()

            # OPTION A: Zeitstempel fÃ¼r den *Beginn* der gesammelten Stunde (LOKAL speichern)
            now_local = dt_util.as_local(dt_util.utcnow())
            try:
                # Nimm den Tag der aktuellen Zeit (kÃƒÂ¶nnte nach Mitternacht sein)
                # und setze die Stunde auf die Zielstunde
                sample_time_local = now_local.replace(hour=target_local_hour, minute=0, second=0, microsecond=0)
                # Wenn die Zielstunde z.B. 23 ist und es ist 00:02, mÃƒÂ¼ssen wir einen Tag abziehen
                if target_local_hour == 23 and now_local.hour == 0:
                     sample_time_local -= timedelta(days=1)

            except ValueError as e:
                _LOGGER.error(f"Fehler beim Erstellen des lokalen Zeitstempels fÃƒÂ¼r Stunde {target_local_hour}: {e}. Verwende UTC-Ãƒâ€žquivalent.")
                # Fallback: Verwende aktuelle Zeit minus 1 Stunde
                sample_time_local = now_local - timedelta(hours=1)
                sample_time_local = sample_time_local.replace(minute=0, second=0, microsecond=0)

            # OPTION A: Speichere direkt als lokalen Timestamp (nicht nach UTC konvertieren)
            sample = {
                "timestamp": sample_time_local.isoformat(), # Speichere als LOKALEN ISO String
                "actual_kwh": round(actual_kwh, 4),
                "daily_total": round(daily_total, 4), # Speichere den Tageswert bis Ende der Stunde
                "percentage_of_day": round(percentage, 4),
                "weather_data": weather_data,
                "sensor_data": sensor_data,
                "model_version": ML_MODEL_VERSION
            }

            await self.data_manager.add_hourly_sample(sample)

            _LOGGER.info(
                f"Hourly Sample gespeichert (Riemann): {sample_time_local.strftime('%Y-%m-%d %H')}:00 local | "
                f"Actual={actual_kwh:.2f}kWh ({percentage*100:.1f}% des Tages), "
                f"DailyTotalAtHourEnd={daily_total:.2f}kWh"
            )

        except Exception as e:
            _LOGGER.error(f"Fehler beim Sammeln des hourly samples fÃƒÂ¼r Stunde {target_local_hour}: {e}", exc_info=True)

    # _safe_parse_yield wird fÃƒÂ¼r _collect_current_sensor_data benÃƒÂ¶tigt
    async def _safe_parse_yield(self, state_obj: Optional[Any]) -> Optional[float]:
        if not state_obj or state_obj.state in ['unavailable', 'unknown', 'none', None]:
            return None
        try:
            # Versuche direkte Konvertierung
            val = float(state_obj.state)
            return val if val >= 0 else 0.0 # Negativen Wert als 0 behandeln
        except (ValueError, TypeError):
            try:
                # Verbesserte Bereinigung
                cleaned_state = str(state_obj.state).split(" ")[0].replace(",", ".")
                val = float(cleaned_state)
                _LOGGER.debug(f"Status '{state_obj.state}' wurde zu {val} normalisiert")
                return val if val >= 0 else 0.0 # Negativen Wert als 0 behandeln
            except (ValueError, TypeError):
                _LOGGER.warning(f"Konnte normalisierten Status nicht in Zahl umwandeln: '{state_obj.state}'")
                return None

    # =========================================================================
    # KORREKTE Riemann-Summe (Links-Methode) - UnverÃƒÂ¤ndert von Batch 4
    # =========================================================================
    async def _perform_riemann_integration(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[float]:
        """
        FÃƒÂ¼hrt eine korrekte linke Riemann-Summenintegration fÃƒÂ¼r die
        konfigurierte power_entity durch.
        """
        if not self.power_entity:
            _LOGGER.debug("Kein power_entity konfiguriert fÃƒÂ¼r Riemann-Summe")
            return None

        _LOGGER.debug(f"Starte Riemann-Summe fÃƒÂ¼r {self.power_entity} von {start_time} bis {end_time}")

        try:
            # Stelle sicher, dass start_time und end_time UTC sind
            if start_time.tzinfo is None or start_time.tzinfo.utcoffset(start_time) is None:
                _LOGGER.error("Riemann: start_time muss timezone-aware (UTC) sein.")
                start_time = start_time.replace(tzinfo=timezone.utc) # Versuche zu reparieren
                # raise ValueError("start_time muss timezone-aware (UTC) sein fÃƒÂ¼r Recorder-Abfrage")
            if end_time.tzinfo is None or end_time.tzinfo.utcoffset(end_time) is None:
                _LOGGER.error("Riemann: end_time muss timezone-aware (UTC) sein.")
                end_time = end_time.replace(tzinfo=timezone.utc) # Versuche zu reparieren
                # raise ValueError("end_time muss timezone-aware (UTC) sein fÃƒÂ¼r Recorder-Abfrage")

            history_list = await self.hass.async_add_executor_job(
                history.get_significant_states,
                self.hass,
                start_time,
                end_time,
                [self.power_entity],
                None,
                True, # include_start_time_state = True
                True  # significant_changes_only = True
            )

            if not history_list or self.power_entity not in history_list:
                _LOGGER.warning(f"Keine History fÃƒÂ¼r {self.power_entity} von {start_time} bis {end_time} gefunden.")
                # PrÃƒÂ¼fe, ob der Sensor ÃƒÂ¼berhaupt existiert
                current_state = self.hass.states.get(self.power_entity)
                if current_state is None:
                    _LOGGER.error(f"Power entity {self.power_entity} nicht in Home Assistant gefunden!")
                    return None # Schwerwiegender Fehler
                else:
                    _LOGGER.warning(f"Sensor {self.power_entity} existiert, aber liefert keine History im Zeitraum. Ergebnis ist 0.0 kWh.")
                    return 0.0 # Kein Fehler, aber keine Daten -> 0 kWh

            states = history_list[self.power_entity]

            if not states:
                _LOGGER.debug(f"History fÃƒÂ¼r {self.power_entity} in Zeitraum ist leer.")
                return 0.0

            total_wh = 0.0

            # Initialisiere mit dem ersten Zustand (von *vor* start_time)
            # Finde den letzten gÃƒÂ¼ltigen Zustand vor oder genau bei start_time
            initial_state_value = 0.0
            initial_state_time = start_time
            found_initial = False
            for state in states:
                if state.last_updated <= start_time:
                    try:
                        initial_state_value = max(0.0, float(state.state))
                        initial_state_time = state.last_updated # Merke Zeit des gÃƒÂ¼ltigen Zustands
                        found_initial = True
                    except (ValueError, TypeError):
                        continue # UngÃƒÂ¼ltigen Zustand ignorieren
                else:
                    break # Zustand ist bereits nach start_time

            if not found_initial:
                _LOGGER.warning(f"Konnte keinen gÃƒÂ¼ltigen Zustand vor/bei {start_time} fÃƒÂ¼r Riemann-Start finden. Beginne mit 0W.")
            prev_power = initial_state_value
            prev_time = start_time # Integration beginnt immer bei start_time

            _LOGGER.debug(f"Riemann Start: Initialer prev_power = {prev_power}W (basierend auf Zustand um {initial_state_time})")


            # Iteriere ÃƒÂ¼ber alle ZustÃƒÂ¤nde, deren Zeit *nach* start_time liegt
            start_index_integration = 0
            while start_index_integration < len(states) and states[start_index_integration].last_updated <= start_time:
                start_index_integration += 1

            _LOGGER.debug(f"Riemann Integration beginnt bei Index {start_index_integration} (Zeit > {start_time})")

            for state in states[start_index_integration:]:
                try:
                    state_time = state.last_updated

                    # Stelle sicher, dass wir nur bis end_time integrieren
                    current_end_time = min(state_time, end_time)

                    # Berechne Energie = Leistung_DAFÃƒÅ“R * Zeitdifferenz
                    time_diff_seconds = (current_end_time - prev_time).total_seconds()
                    
                    # Ignoriere negative oder Null-Zeitdifferenzen
                    if time_diff_seconds <= 1e-6: # Kleine Toleranz fÃƒÂ¼r FlieÃƒÅ¸kommavergleiche
                        _LOGGER.debug(f"ÃƒÅ“berspringe Riemann Schritt: Zeitdifferenz ist Null oder negativ bei {current_end_time}.")
                        # Update prev_time, um Fortschritt zu machen, aber Leistung bleibt gleich
                        prev_time = current_end_time
                        # Update prev_power, falls der aktuelle Zustand gÃƒÂ¼ltig ist
                        try: prev_power = max(0.0, float(state.state))
                        except (ValueError, TypeError): pass
                        continue

                    time_diff_hours = time_diff_seconds / 3600.0
                    wh = prev_power * time_diff_hours
                    total_wh += wh
                    _LOGGER.debug(f"Riemann Schritt: von {prev_time.strftime('%H:%M:%S.%f')} bis {current_end_time.strftime('%H:%M:%S.%f')} ({time_diff_seconds:.1f}s) mit {prev_power:.1f}W -> {wh:.2f}Wh dazu (Total: {total_wh:.2f}Wh)")

                    # Update fÃƒÂ¼r die nÃƒÂ¤chste Schleife
                    prev_time = current_end_time

                    # Aktualisiere prev_power nur, wenn der aktuelle Zustand gÃƒÂ¼ltig ist
                    try:
                         current_power = max(0.0, float(state.state))
                         prev_power = current_power # Update nur bei gÃƒÂ¼ltigem Wert
                    except (ValueError, TypeError):
                         _LOGGER.debug(f"UngÃƒÂ¼ltiger Zustand '{state.state}' bei {state_time}. Verwende vorherigen Wert {prev_power}W weiter.")
                         # prev_power bleibt unverÃƒÂ¤ndert

                    # Stoppe, wenn wir end_time erreicht haben oder ÃƒÂ¼berschritten
                    if prev_time >= end_time:
                        _LOGGER.debug(f"Riemann erreicht/ÃƒÂ¼berschreitet Endzeit {end_time}.")
                        break

                except Exception as loop_err:
                     _LOGGER.error(f"Fehler in Riemann-Schleife bei Zeit {state.last_updated}: {loop_err}", exc_info=True)
                     if hasattr(state, 'last_updated'):
                         prev_time = min(state.last_updated, end_time)
                     continue

            # Berechne das letzte StÃƒÂ¼ck vom letzten Event bis end_time, falls noch nicht erreicht
            if prev_time < end_time:
                time_diff_seconds = (end_time - prev_time).total_seconds()
                if time_diff_seconds > 1e-6:
                    time_diff_hours = time_diff_seconds / 3600.0
                    wh = prev_power * time_diff_hours # Nutze die letzte bekannte gÃƒÂ¼ltige Leistung
                    total_wh += wh
                    _LOGGER.debug(f"Riemann Abschluss: von {prev_time.strftime('%H:%M:%S.%f')} bis {end_time.strftime('%H:%M:%S.%f')} ({time_diff_seconds:.1f}s) mit {prev_power:.1f}W -> {wh:.2f}Wh dazu (Final Total: {total_wh:.2f}Wh)")

            # Stelle sicher, dass das Ergebnis nicht negativ ist
            kwh = max(0.0, total_wh / 1000.0)
            _LOGGER.debug(f"Riemann-Summe Ergebnis: {kwh:.4f} kWh")
            return kwh

        except Exception as e:
            _LOGGER.error(f"Kritischer Fehler bei Riemann-Integration fÃƒÂ¼r {self.power_entity}: {e}", exc_info=True)
            return None

    # --- KORREKTUR (Block 2 & 9) ---
    async def _get_actual_production_for_hour(self, target_local_hour: int) -> Optional[float]:
        """
        Holt die Produktion der Stunde via Riemann-Integration.
        Verwendet die lokale Zeit fÃƒÂ¼r die Stundenauswahl.
        """
        _LOGGER.debug(f"Abruf der Ist-Produktion fÃƒÂ¼r Stunde {target_local_hour} (Lokal)...")
        # 1. Aktuelle Zeit in der LOKALEN Zeitzone von HA holen
        now_local = dt_util.as_local(dt_util.utcnow())

        # 2. Start- und Endzeit fÃƒÂ¼r die angeforderte LOKALE Stunde des aktuellen Tages erstellen
        try:
            # Nimm den Tag der aktuellen lokalen Zeit
            # und setze die Stunde auf die Zielstunde
            start_time_local = now_local.replace(hour=target_local_hour, minute=0, second=0, microsecond=0)
            # Wenn die Zielstunde (z.B. 23) grÃƒÂ¶ÃƒÅ¸er ist als die aktuelle Stunde (z.B. 0),
            # bedeutet das, dass der Trigger nach Mitternacht lief und wir den Vortag meinen.
            if target_local_hour > now_local.hour:
                start_time_local -= timedelta(days=1)
                _LOGGER.debug(f"Zielstunde {target_local_hour} liegt vor aktueller Stunde {now_local.hour}. Verwende Vortag: {start_time_local.date()}")

            end_time_local = start_time_local + timedelta(hours=1)
        except ValueError as e:
            _LOGGER.warning(f"Konnte lokale Startzeit fÃƒÂ¼r Stunde {target_local_hour} nicht erstellen (evtl. Zeitumstellung?): {e}")
            return None

        # 3. Sicherstellen, dass nicht in die Zukunft integriert wird (obwohl nicht mehr nÃƒÂ¶tig sein sollte)
        # if end_time_local > now_local:
        #     _LOGGER.debug(f"Integrationsende {end_time_local.strftime('%H:%M:%S')} liegt in der Zukunft. Begrenze auf {now_local.strftime('%H:%M:%S')}.")
        #     end_time_local = now_local # Begrenze auf die aktuelle lokale Zeit

        # 4. PrÃƒÂ¼fen, ob der gesamte Zeitraum in der Vergangenheit liegt
        if end_time_local > dt_util.as_local(dt_util.utcnow()):
             _LOGGER.warning(f"Versuch, Daten fÃƒÂ¼r zukÃƒÂ¼nftige Stunde {target_local_hour} (bis {end_time_local}) zu sammeln. ÃƒÅ“berspringe.")
             return None # Kann keine zukÃƒÂ¼nftigen Daten sammeln

        # 5. Konvertiere die LOKALEN Zeiten nach UTC fÃƒÂ¼r die Recorder-Abfrage
        try:
            start_time_utc = start_time_local.astimezone(timezone.utc)
            end_time_utc = end_time_local.astimezone(timezone.utc)
        except Exception as tz_err:
            _LOGGER.error(f"Fehler bei Zeitzonenkonvertierung fÃƒÂ¼r Stunde {target_local_hour}: {tz_err}")
            return None

        _LOGGER.debug(f"Frage Recorder ab fÃƒÂ¼r Stunde {target_local_hour} Lokal: "
                      f"UTC-Zeitraum: {start_time_utc.isoformat()} bis {end_time_utc.isoformat()}")

        # 6. FÃƒÂ¼hre die Integration mit den korrekten UTC-Zeiten durch
        kwh = await self._perform_riemann_integration(start_time_utc, end_time_utc)

        if kwh is not None:
             _LOGGER.debug(
                f"Produktion Stunde {target_local_hour} (Lokal) [UTC: {start_time_utc.strftime('%H:%M')}-{end_time_utc.strftime('%H:%M')}] (Riemann): {kwh:.4f} kWh"
            )
        else:
             _LOGGER.warning(f"Riemann-Integration fÃƒÂ¼r Stunde {target_local_hour} (Lokal) fehlgeschlagen.")

        return kwh
    # --- ENDE KORREKTUR ---

    async def _get_daily_production_so_far(self, target_local_hour: int) -> Optional[float]:
        """Holt die Tagesproduktion (bisher) via Riemann-Integration bis zum Ende der Zielstunde."""
        now_utc = dt_util.utcnow()
        now_local = dt_util.as_local(now_utc)

        # Bestimme Start des Tages (Lokalzeit)
        start_of_day_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        # Wenn die Zielstunde (z.B. 23) grÃƒÂ¶ÃƒÅ¸er ist als die aktuelle Stunde (z.B. 0),
        # bezieht sich der Tagesanfang auf den Vortag.
        if target_local_hour > now_local.hour:
            start_of_day_local -= timedelta(days=1)

        # Bestimme das Ende der Zielstunde (Lokalzeit)
        try:
            end_of_hour_local = start_of_day_local.replace(hour=target_local_hour, minute=59, second=59, microsecond=999999)
            # Wenn target_local_hour 23 ist, brauchen wir +1 Stunde, um das Ende zu bekommen
            if target_local_hour == 23:
                 end_of_hour_local = (start_of_day_local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                 end_of_hour_local = start_of_day_local.replace(hour=target_local_hour + 1, minute=0, second=0, microsecond=0)

        except ValueError as e:
            _LOGGER.warning(f"Konnte Endzeit fÃƒÂ¼r Tagesproduktion bis Stunde {target_local_hour} nicht bestimmen: {e}")
            return None

        # Konvertiere nach UTC
        try:
            start_of_day_utc = start_of_day_local.astimezone(timezone.utc)
            end_of_hour_utc = end_of_hour_local.astimezone(timezone.utc)
        except Exception as tz_err:
             _LOGGER.error(f"Fehler bei Zeitzonenkonvertierung fÃƒÂ¼r Tagesproduktion: {tz_err}")
             return None

        _LOGGER.debug(f"Berechne Tagesproduktion bis Ende Stunde {target_local_hour} (Lokal): "
                      f"UTC-Zeitraum: {start_of_day_utc.isoformat()} bis {end_of_hour_utc.isoformat()}")

        # Integriere von Tagesanfang (UTC) bis Ende der Zielstunde (UTC)
        kwh = await self._perform_riemann_integration(start_of_day_utc, end_of_hour_utc)

        if kwh is not None:
            _LOGGER.debug(f"Tagesertrag bis Ende Stunde {target_local_hour} (Riemann): {kwh:.2f} kWh")
        else:
            _LOGGER.warning(f"Abruf des Tagesertrags (bis Stunde {target_local_hour}) via Riemann fehlgeschlagen.")

        return kwh
    # =========================================================================
    # ENDE KORREKTUR
    # =========================================================================

    async def _collect_current_sensor_data(self) -> Dict[str, Any]:
        """Sammelt aktuelle Werte von konfigurierten optionalen Sensoren."""
        sensor_data = {}
        _LOGGER.debug("Sammle aktuelle optionale Sensorwerte...")
        try:
            entity_map = [
                (self.temp_sensor, 'temperature'),
                (self.wind_sensor, 'wind_speed'),
                (self.rain_sensor, 'rain'),
                (self.uv_sensor, 'uv_index'),
                (self.lux_sensor, 'lux'),
                (self.humidity_sensor, 'humidity') # Aus Block 8
            ]

            for entity, key in entity_map:
                if entity:
                    state_obj = self.hass.states.get(entity)
                    val = await self._safe_parse_yield(state_obj)
                    # Setze auf 0.0 wenn None zurÃƒÂ¼ckkommt oder negativ
                    sensor_data[key] = max(0.0, val) if val is not None else 0.0
                    _LOGGER.debug(f"Sensor {key} ({entity}): {sensor_data[key]}")
                else:
                    # FÃƒÂ¼ge den Key trotzdem hinzu mit 0.0, damit das Modell konsistente Features sieht
                    sensor_data[key] = 0.0
                    _LOGGER.debug(f"Sensor {key}: Nicht konfiguriert, setze auf 0.0")

        except Exception as e:
            _LOGGER.warning(f"Fehler beim Sammeln von Sensor-Daten: {e}")
            # FÃƒÂ¼lle fehlende Keys mit 0.0 auf, falls ein Fehler auftrat
            default_keys = ['temperature', 'wind_speed', 'rain', 'uv_index', 'lux', 'humidity']
            for key in default_keys:
                sensor_data.setdefault(key, 0.0)

        _LOGGER.debug(f"Gesammelte Sensordaten: {sensor_data}")
        return sensor_data

    async def _get_current_weather_data(self) -> Dict[str, Any]:
        """Holt aktuelle Wetterdaten vom konfigurierten Wettersensor."""
        _LOGGER.debug("Rufe aktuelle Wetterdaten ab...")
        try:
            if not self.weather_entity:
                _LOGGER.warning("Kein Wettersensor konfiguriert. Verwende Standardwerte.")
                return self._get_default_weather()

            ws = self.hass.states.get(self.weather_entity)

            if not ws or not hasattr(ws, 'attributes') or ws.state in ['unavailable', 'unknown']:
                _LOGGER.warning(f"Konnte Wetter-Status nicht abrufen oder ungültig: {self.weather_entity} (State: {ws.state if ws else 'None'})")
                return self._get_default_weather()

            attrs = ws.attributes

            # === KRITISCHE VALIDIERUNG: Prüfe ob Attribute geladen sind ===
            # Wenn temperature None ist, ist die Entity zwar verfügbar, aber Attribute noch nicht geladen
            temp_value = attrs.get('temperature')
            if temp_value is None:
                _LOGGER.error(
                    f"KRITISCH: Weather Entity {self.weather_entity} ist verfügbar, "
                    f"aber 'temperature' Attribut ist None (Entity lädt noch Daten). "
                    f"Sample wird mit Default-Werten gespeichert um Datenverlust zu vermeiden. "
                    f"ACHTUNG: Dies produziert ungültige Trainingsdaten!"
                )
                # Gebe Defaults zurück, aber mit deutlicher Warnung
                return self._get_default_weather()
            # ================================================================

            # Cloud Cover Logik (wie im Weather Service)
            cc_val = attrs.get('cloud_coverage', attrs.get('cloudiness'))
            cc = 50.0 # Default
            if cc_val is not None:
                try: cc = float(str(cc_val).replace("%", "").strip())
                except (ValueError, TypeError):
                    _LOGGER.debug(f"Konnte Cloud-Wert '{cc_val}' nicht parsen.")
                    # Fallback: Versuche Mapping aus Zustandswert
                    from .weather_calculator import WeatherCalculator # Lokaler Import
                    wc = WeatherCalculator()
                    cc = wc._map_condition_to_cloud_cover(ws.state)
            else: # Wenn beide Attribute fehlen
                from .weather_calculator import WeatherCalculator
                wc = WeatherCalculator()
                cc = wc._map_condition_to_cloud_cover(ws.state)


            # Hilfsfunktion zum sicheren Parsen von Float-Attributen
            # WICHTIG: Gibt None zurück wenn Attribut None ist (nicht mehr Default!)
            def safe_float(key, allow_none=False):
                val = attrs.get(key)
                if val is None:
                    if allow_none:
                        return None
                    else:
                        # Dies sollte nicht passieren da wir temperature bereits geprüft haben
                        _LOGGER.warning(f"Kritisches Attribut '{key}' ist None, Entity noch nicht initialisiert.")
                        raise ValueError(f"Required attribute '{key}' is None")
                try: 
                    return float(val)
                except (ValueError, TypeError):
                    _LOGGER.debug(f"Konnte Wetter-Attribut '{key}' ('{val}') nicht parsen.")
                    raise ValueError(f"Cannot parse attribute '{key}': {val}")

            # Daten sammeln und validieren (mit bereits geprüftem temp_value)
            weather_data = {
                'temperature': float(temp_value),  # Bereits validiert (nicht None)
                'humidity': max(0.0, min(100.0, safe_float('humidity'))), # Clamp 0-100
                'cloudiness': max(0.0, min(100.0, cc)), # Clamp 0-100
                'wind_speed': max(0.0, safe_float('wind_speed')), # Clamp >= 0
                'pressure': safe_float('pressure')
            }
            _LOGGER.debug(f"Wetterdaten erfolgreich abgerufen: {weather_data}")
            return weather_data

        except ValueError as ve:
            # Spezifischer Fehler: Attribute sind None oder nicht parsebar
            _LOGGER.error(
                f"Weather Entity Attribute nicht verfügbar: {ve}. "
                f"Entity lädt möglicherweise noch. Verwende Default-Werte. "
                f"WARNUNG: Dies erzeugt ungültige Trainingsdaten!"
            )
            return self._get_default_weather()
        except Exception as e:
            _LOGGER.warning(f"Unerwarteter Fehler beim Abrufen von Wetter-Daten: {e}", exc_info=True)
            return self._get_default_weather()

    def _get_default_weather(self) -> Dict[str, Any]:
        """Gibt Standard-Wetterwerte zurÃƒÂ¼ck."""
        return {
            'temperature': 15.0,
            'humidity': 60.0,
            'cloudiness': 50.0,
            'wind_speed': 5.0,
            'pressure': 1013.0
        }

    def set_forecast_cache(self, cache: Dict[str, Any]) -> None:
        self._forecast_cache = cache

    # --- KORREKTUR (Block 8): configure_entities Signatur erweitert ---
    def configure_entities(
        self,
        weather_entity: Optional[str] = None,
        power_entity: Optional[str] = None,
        temp_sensor: Optional[str] = None,
        wind_sensor: Optional[str] = None,
        rain_sensor: Optional[str] = None,
        uv_sensor: Optional[str] = None,
        lux_sensor: Optional[str] = None,
        humidity_sensor: Optional[str] = None, # <-- NEU
        solar_yield_today: Optional[str] = None # HinzugefÃƒÂ¼gt fÃƒÂ¼r KompatibilitÃƒÂ¤t
    ) -> None:
        """Konfiguriert die vom Collector verwendeten EntitÃƒÂ¤ts-IDs."""
        _LOGGER.debug("Konfiguriere EntitÃƒÂ¤ten im SampleCollector...")
        self.weather_entity = weather_entity
        self.power_entity = power_entity
        self.temp_sensor = temp_sensor
        self.wind_sensor = wind_sensor
        self.rain_sensor = rain_sensor
        self.uv_sensor = uv_sensor
        self.lux_sensor = lux_sensor
        self.humidity_sensor = humidity_sensor # <-- NEU
        _LOGGER.debug(f"SampleCollector Entities: Weather='{weather_entity}', Power='{power_entity}', "
                      f"Temp='{temp_sensor}', Wind='{wind_sensor}', Rain='{rain_sensor}', "
                      f"UV='{uv_sensor}', Lux='{lux_sensor}', Humidity='{humidity_sensor}'")
    # --- ENDE KORREKTUR ---