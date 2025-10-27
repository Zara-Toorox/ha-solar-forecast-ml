"""
Production Calculator für Solar Forecast ML.
Berechnet Produktionszeit und weitere Produktions-Metriken.
STRATEGIE 2: ProductionTimeCalculator für Live-Tracking
Version 4.11.0 - Gewichtete Peak-Zeit Berechnung

Copyright (C) 2025 Zara-Toorox

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
from typing import Optional, Any
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_change
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)


class ProductionCalculator:
    
    def __init__(self, hass: HomeAssistant):
        self.hass = hass
        
        self.MIN_PRODUCTION_POWER = 0.01
        self.PRODUCTION_START_HOUR = 5
        self.PRODUCTION_END_HOUR = 21
        
        self.RECENT_DAYS_THRESHOLD = 3
        self.RECENT_WEIGHT = 0.7
        self.OLDER_WEIGHT = 0.3
        
        _LOGGER.debug("✓ ProductionCalculator initialisiert")
    
    async def calculate_production_time_today(
        self,
        power_entity: Optional[str]
    ) -> str:
        try:
            if not power_entity:
                _LOGGER.debug("1️⃣ Kein Power-Sensor konfiguriert")
                return "Nicht verfügbar"
            
            now = dt_util.utcnow()
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            history = await self.hass.async_add_executor_job(
                self._get_state_history,
                power_entity,
                start_of_day,
                now
            )
            
            if not history:
                _LOGGER.debug("1️⃣ Keine History-Daten verfügbar")
                return "Berechnung läuft..."
            
            production_minutes = 0
            
            for i in range(len(history) - 1):
                try:
                    power = float(history[i].state)
                    
                    if power >= self.MIN_PRODUCTION_POWER:
                        time_diff = history[i + 1].last_changed - history[i].last_changed
                        production_minutes += time_diff.total_seconds() / 60
                        
                except (ValueError, AttributeError):
                    continue
            
            hours = int(production_minutes // 60)
            minutes = int(production_minutes % 60)
            
            return f"{hours}h {minutes}m"
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Produktionszeit-Berechnung fehlgeschlagen: {e}")
            return "Berechnung fehlgeschlagen"
    
    def _get_state_history(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> list:
        try:
            from homeassistant.components import recorder
            from homeassistant.components.recorder import history
            
            if not recorder.is_entity_recorded(self.hass, entity_id):
                _LOGGER.debug(f"1️⃣ Entity {entity_id} wird nicht aufgezeichnet")
                return []
            
            states = history.state_changes_during_period(
                self.hass,
                start_time,
                end_time,
                entity_id,
                no_attributes=True
            )
            
            if entity_id in states:
                return states[entity_id]
            
            return []
            
        except Exception as e:
            _LOGGER.debug(f"History-Abruf fehlgeschlagen: {e}")
            return []
    
    async def calculate_peak_production_time(
        self,
        power_entity: Optional[str] = None
    ) -> str:
        try:
            _LOGGER.info("🔍 START Peak-Zeit Berechnung (Gewichtete Methode)")
            
            if not power_entity:
                _LOGGER.debug("1️⃣ Kein Power-Sensor für Peak-Berechnung")
                return "12:00"
            
            now = dt_util.now()
            start_time = now - timedelta(days=14)
            cutoff_recent = now - timedelta(days=self.RECENT_DAYS_THRESHOLD)
            
            _LOGGER.debug(
                f"📅 Analyse-Zeitraum: {start_time.strftime('%Y-%m-%d %H:%M')} "
                f"bis {now.strftime('%Y-%m-%d %H:%M')}"
            )
            _LOGGER.debug(
                f"📊 Gewichtung: Letzte {self.RECENT_DAYS_THRESHOLD} Tage = {self.RECENT_WEIGHT*100:.0f}%, "
                f"Ältere Tage = {self.OLDER_WEIGHT*100:.0f}%"
            )
            
            states = await self.hass.async_add_executor_job(
                self._get_state_history,
                power_entity,
                start_time,
                now
            )
            
            if not states or len(states) < 10:
                _LOGGER.warning(
                    f"⚠️ Nicht genug Daten für Peak-Berechnung: "
                    f"{len(states) if states else 0} States gefunden (min. 10 benötigt)"
                )
                return "12:00"
            
            _LOGGER.debug(f"📈 {len(states)} States aus History geladen")
            
            hourly_data = {hour: {'values': [], 'weights': []} for hour in range(24)}
            
            invalid_states = 0
            night_values = 0
            converted_watt = 0
            recent_count = 0
            older_count = 0
            
            for state in states:
                try:
                    if state.state in ["unavailable", "unknown", None]:
                        invalid_states += 1
                        continue
                    
                    power = float(state.state)
                    
                    if power < self.MIN_PRODUCTION_POWER:
                        night_values += 1
                        continue
                    
                    if power > 100:
                        power = power / 1000.0
                        converted_watt += 1
                    
                    hour = state.last_changed.hour
                    
                    if not (self.PRODUCTION_START_HOUR <= hour <= self.PRODUCTION_END_HOUR):
                        night_values += 1
                        continue
                    
                    is_recent = state.last_changed >= cutoff_recent
                    weight = self.RECENT_WEIGHT if is_recent else self.OLDER_WEIGHT
                    
                    if is_recent:
                        recent_count += 1
                    else:
                        older_count += 1
                    
                    hourly_data[hour]['values'].append(power)
                    hourly_data[hour]['weights'].append(weight)
                
                except (ValueError, TypeError, AttributeError) as e:
                    invalid_states += 1
                    continue
            
            _LOGGER.debug(
                f"📊 Datenqualität: {invalid_states} invalid, {night_values} night/low, "
                f"{converted_watt} W→kW konvertiert"
            )
            _LOGGER.debug(
                f"⚖️ Gewichtete Daten: {recent_count} recent ({self.RECENT_WEIGHT*100:.0f}%), "
                f"{older_count} older ({self.OLDER_WEIGHT*100:.0f}%)"
            )
            
            hourly_weighted_averages = {}
            
            for hour, data in hourly_data.items():
                if data['values']:
                    weighted_sum = sum(v * w for v, w in zip(data['values'], data['weights']))
                    weight_sum = sum(data['weights'])
                    
                    if weight_sum > 0:
                        weighted_avg = weighted_sum / weight_sum
                        hourly_weighted_averages[hour] = weighted_avg
                        
                        _LOGGER.debug(
                            f"⏰ Stunde {hour:02d}: {len(data['values'])} Werte, "
                            f"Gewichteter Ø = {weighted_avg:.3f} kW"
                        )
            
            if not hourly_weighted_averages:
                _LOGGER.warning("⚠️ Keine gültigen Produktionsdaten gefunden")
                return "12:00"
            
            peak_hour = max(hourly_weighted_averages, key=hourly_weighted_averages.get)
            peak_value = hourly_weighted_averages[peak_hour]
            
            if not (self.PRODUCTION_START_HOUR <= peak_hour <= self.PRODUCTION_END_HOUR):
                _LOGGER.warning(
                    f"⚠️ Peak-Stunde {peak_hour}:00 außerhalb Produktionszeit "
                    f"({self.PRODUCTION_START_HOUR}-{self.PRODUCTION_END_HOUR}), Fallback 12:00"
                )
                return "12:00"
            
            _LOGGER.info(
                f"✓ Peak-Stunde gefunden: {peak_hour:02d}:00 Uhr "
                f"(≈ {peak_value:.2f} kW gewichtet, "
                f"{len(hourly_data[peak_hour]['values'])} Werte)"
            )
            
            return f"{peak_hour:02d}:00"
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Peak-Zeit Berechnung fehlgeschlagen: {e}", exc_info=True)
            return "12:00"
    
    def is_production_hours(self, hour: int = None) -> bool:
        try:
            if hour is None:
                hour = dt_util.utcnow().hour
            
            return self.PRODUCTION_START_HOUR <= hour <= self.PRODUCTION_END_HOUR
            
        except Exception:
            return True
    
    def estimate_remaining_production_hours(self) -> float:
        try:
            now = dt_util.utcnow()
            current_hour = now.hour
            
            if current_hour >= self.PRODUCTION_END_HOUR:
                return 0.0
            
            if current_hour < self.PRODUCTION_START_HOUR:
                return float(self.PRODUCTION_END_HOUR - self.PRODUCTION_START_HOUR)
            
            remaining_full_hours = self.PRODUCTION_END_HOUR - current_hour - 1
            current_hour_fraction = 1.0 - (now.minute / 60.0)
            
            return float(remaining_full_hours) + current_hour_fraction
            
        except Exception:
            return 0.0


class ProductionTimeCalculator:
    
    def __init__(self, hass: HomeAssistant, power_entity: Optional[str] = None):
        self.hass = hass
        self.power_entity = power_entity
        
        self._is_active = False
        self._start_time: Optional[datetime] = None
        self._accumulated_hours = 0.0
        self._last_production_time: Optional[datetime] = None
        self._zero_power_start: Optional[datetime] = None
        self._today_total_hours = 0.0
        
        self.MIN_POWER_THRESHOLD = 10.0
        self.ZERO_POWER_THRESHOLD = 1.0
        self.ZERO_POWER_TIMEOUT = timedelta(minutes=5)
        
        self._state_listener_remove = None
        self._midnight_listener_remove = None
        
        _LOGGER.info("✓ ProductionTimeCalculator initialisiert")
    
    def start_tracking(self) -> None:
        if not self.power_entity:
            _LOGGER.info("1️⃣ Kein Power-Entity - Produktionszeit-Tracking deaktiviert")
            return
        
        try:
            self._state_listener_remove = async_track_state_change_event(
                self.hass,
                [self.power_entity],
                self._handle_power_change
            )
            
            self._midnight_listener_remove = async_track_time_change(
                self.hass,
                self._handle_midnight_reset,
                hour=0,
                minute=0,
                second=0
            )
            
            _LOGGER.info(f"✓ Produktionszeit-Tracking gestartet für {self.power_entity}")
            
        except Exception as e:
            _LOGGER.error(f"❌ Fehler beim Starten des Trackings: {e}")
    
    @callback
    def _handle_power_change(self, event) -> None:
        try:
            new_state = event.data.get("new_state")
            if not new_state or new_state.state in ["unavailable", "unknown"]:
                return
            
            try:
                power = float(new_state.state)
            except (ValueError, TypeError):
                return
            
            now = dt_util.utcnow()
            
            if power >= self.MIN_POWER_THRESHOLD:
                if not self._is_active:
                    self._is_active = True
                    self._start_time = now
                    _LOGGER.debug(f"☀️ Produktions-Start: {power}W")
                
                self._zero_power_start = None
                self._last_production_time = now
            
            elif self._is_active:
                if power < self.ZERO_POWER_THRESHOLD:
                    if self._zero_power_start is None:
                        self._zero_power_start = now
                        _LOGGER.debug(f"⏱️ Zero-Power Timer gestartet: {power}W")
                    
                    elif now - self._zero_power_start >= self.ZERO_POWER_TIMEOUT:
                        self._stop_production_tracking(now)
                        _LOGGER.debug(f"🛑 Produktions-Ende nach 5 Min Timeout")
                
                else:
                    self._zero_power_start = None
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Fehler in Power Change Handler: {e}")
    
    def _stop_production_tracking(self, stop_time: datetime) -> None:
        if not self._is_active or not self._start_time:
            return
        
        duration = stop_time - self._start_time
        hours = duration.total_seconds() / 3600.0
        
        self._accumulated_hours += hours
        self._today_total_hours = self._accumulated_hours
        
        _LOGGER.info(
            f"✓ Produktionsphase beendet: {hours:.2f}h "
            f"(Gesamt heute: {self._today_total_hours:.2f}h)"
        )
        
        self._is_active = False
        self._start_time = None
        self._zero_power_start = None
    
    @callback
    def _handle_midnight_reset(self, now: datetime) -> None:
        _LOGGER.info(f"🕛 Mitternacht-Reset: Heute {self._today_total_hours:.2f}h produziert")
        
        if self._is_active:
            self._stop_production_tracking(now)
        
        self._accumulated_hours = 0.0
        self._today_total_hours = 0.0
        self._is_active = False
        self._start_time = None
        self._last_production_time = None
        self._zero_power_start = None
    
    def get_production_time(self) -> str:
        try:
            if not self.power_entity:
                return "Nicht verfügbar"
            
            total_hours = self._accumulated_hours
            
            if self._is_active and self._start_time:
                now = dt_util.utcnow()
                current_duration = now - self._start_time
                total_hours += current_duration.total_seconds() / 3600.0
            
            if total_hours < 0.01:
                return "0h 0m"
            
            hours = int(total_hours)
            minutes = int((total_hours - hours) * 60)
            
            return f"{hours}h {minutes}m"
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Fehler beim Abrufen der Produktionszeit: {e}")
            return "Fehler"
    
    def get_production_hours_float(self) -> float:
        try:
            total_hours = self._accumulated_hours
            
            if self._is_active and self._start_time:
                now = dt_util.utcnow()
                current_duration = now - self._start_time
                total_hours += current_duration.total_seconds() / 3600.0
            
            return round(total_hours, 2)
            
        except Exception:
            return 0.0
    
    def is_currently_producing(self) -> bool:
        return self._is_active
    
    def stop_tracking(self) -> None:
        if self._state_listener_remove:
            self._state_listener_remove()
            self._state_listener_remove = None
        
        if self._midnight_listener_remove:
            self._midnight_listener_remove()
            self._midnight_listener_remove = None
        
        _LOGGER.info("✓ Produktionszeit-Tracking gestoppt")
