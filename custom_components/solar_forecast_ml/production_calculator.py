"""
Production Calculator für Solar Forecast ML.
Berechnet Produktionszeit und weitere Produktions-Metriken.
STRATEGIE 2: ProductionTimeCalculator für Live-Tracking
Version 4.10.0 - Historische Peak-Zeit Berechnung

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
    """
    Berechnet Produktionszeit und verwandte Metriken.

    """
    
    def __init__(self, hass: HomeAssistant):
        """
        Initialisiere Production Calculator.
        
        Args:
            hass: HomeAssistant Instanz

        """
        self.hass = hass
        
        # Konstanten für Produktions-Berechnung
        self.MIN_PRODUCTION_POWER = 0.01  # Minimum Power (kW) für Produktion
        self.PRODUCTION_START_HOUR = 5   # FrÃƒÂ¼heste mÃƒÂ¶gliche Produktion
        self.PRODUCTION_END_HOUR = 21    # SpÃƒÂ¤teste mÃƒÂ¶gliche Produktion
        
        _LOGGER.debug("Ã¢Å“â€œ ProductionCalculator initialisiert")
    
    async def calculate_production_time_today(
        self,
        power_entity: Optional[str]
    ) -> str:
        """
        Berechnet heutige Produktionszeit aus Power-Sensor.
        
        Args:
            power_entity: Entity-ID des Power-Sensors (optional)
            
        Returns:
            Produktionszeit als String (z.B. "6h 30m") oder Fallback

        """
        try:
            # Kein Power-Sensor konfiguriert
            if not power_entity:
                _LOGGER.debug("ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¹ÃƒÂ¯Ã‚Â¸Ã‚Â Kein Power-Sensor konfiguriert")
                return "Nicht verfÃƒÂ¼gbar"
            
            # Hole History für heute
            now = dt_util.utcnow()
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Hole State History
            history = await self.hass.async_add_executor_job(
                self._get_state_history,
                power_entity,
                start_of_day,
                now
            )
            
            if not history:
                _LOGGER.debug("ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¹ÃƒÂ¯Ã‚Â¸Ã‚Â Keine History-Daten verfÃƒÂ¼gbar")
                return "Berechnung lÃƒÂ¤uft..."
            
            # ZÃƒÂ¤hle ZeitrÃƒÂ¤ume mit Produktion
            production_minutes = 0
            
            for i in range(len(history) - 1):
                try:
                    # Parse Power-Wert
                    power = float(history[i].state)
                    
                    # Wenn Power ÃƒÂ¼ber Minimum
                    if power >= self.MIN_PRODUCTION_POWER:
                        # Berechne Zeitdifferenz zum nÃƒÂ¤chsten State
                        time_diff = history[i + 1].last_changed - history[i].last_changed
                        production_minutes += time_diff.total_seconds() / 60
                        
                except (ValueError, AttributeError):
                    # Skip ungÃƒÂ¼ltige States
                    continue
            
            # Konvertiere zu Stunden und Minuten
            hours = int(production_minutes // 60)
            minutes = int(production_minutes % 60)
            
            return f"{hours}h {minutes}m"
            
        except Exception as e:
            _LOGGER.warning(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Produktionszeit-Berechnung fehlgeschlagen: {e}")
            return "Berechnung fehlgeschlagen"
    
    def _get_state_history(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> list:
        """
        Synchrone Hilfsfunktion für History-Abruf.
        
        Args:
            entity_id: Entity-ID
            start_time: Start-Zeitpunkt
            end_time: End-Zeitpunkt
            
        Returns:
            Liste von States

        """
        try:
            from homeassistant.components import recorder
            from homeassistant.components.recorder import history
            
            # PrÃƒÂ¼fe ob Recorder verfÃƒÂ¼gbar
            if not recorder.is_entity_recorded(self.hass, entity_id):
                _LOGGER.debug(f"ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¹ÃƒÂ¯Ã‚Â¸Ã‚Â Entity {entity_id} wird nicht aufgezeichnet")
                return []
            
            # Hole History
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
        """
        Berechnet beste Stunde basierend auf historischen Power-Daten.
        Analysiert letzte 14 Tage und findet Stunde mit hÃƒÂ¶chster Durchschnittsproduktion.
        
        Args:
            power_entity: Power-Sensor Entity-ID (optional)
            
        Returns:
            Peak-Zeit als String (z.B. "12:00")

        """
        try:
            # Fallback wenn kein Power-Sensor konfiguriert
            if not power_entity:
                _LOGGER.debug("ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¹ÃƒÂ¯Ã‚Â¸Ã‚Â Kein Power-Sensor für Peak-Berechnung")
                return "12:00"
            
            # Zeitraum: Letzte 14 Tage
            now = dt_util.now()
            start_time = now - timedelta(days=14)
            
            # Hole historische Daten
            states = await self.hass.async_add_executor_job(
                self._get_state_history,
                power_entity,
                start_time,
                now
            )
            
            if not states or len(states) < 10:
                _LOGGER.debug("ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¹ÃƒÂ¯Ã‚Â¸Ã‚Â Nicht genug Daten für Peak-Berechnung")
                return "12:00"
            
            # Sammle Produktionswerte pro Stunde
            hourly_production = {hour: [] for hour in range(24)}
            
            for state in states:
                try:
                    # Filtere ungÃƒÂ¼ltige States
                    if state.state in ["unavailable", "unknown", None]:
                        continue
                    
                    power = float(state.state)
                    
                    # Ignoriere Nacht-Werte und sehr kleine Werte
                    if power < self.MIN_PRODUCTION_POWER:
                        continue
                    
                    # Konvertiere zu kW falls in W
                    if power > 100:  # Vermutlich Watt
                        power = power / 1000.0
                    
                    # Extrahiere Stunde
                    hour = state.last_changed.hour
                    
                    # Nur Produktionsstunden berÃƒÂ¼cksichtigen
                    if self.PRODUCTION_START_HOUR <= hour <= self.PRODUCTION_END_HOUR:
                        hourly_production[hour].append(power)
                
                except (ValueError, TypeError, AttributeError):
                    continue
            
            # Berechne Durchschnitt pro Stunde
            hourly_averages = {}
            for hour, values in hourly_production.items():
                if values:  # Nur Stunden mit Daten
                    avg = sum(values) / len(values)
                    hourly_averages[hour] = avg
            
            # Finde Stunde mit hÃƒÂ¶chster Durchschnittsproduktion
            if not hourly_averages:
                _LOGGER.debug("ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¹ÃƒÂ¯Ã‚Â¸Ã‚Â Keine gÃƒÂ¼ltigen Produktionsdaten gefunden")
                return "12:00"
            
            peak_hour = max(hourly_averages, key=hourly_averages.get)
            peak_value = hourly_averages[peak_hour]
            
            _LOGGER.info(
                f"Ã¢Å“â€œ Peak-Stunde gefunden: {peak_hour}:00 Uhr "
                f"(ÃƒÆ’Ã‹Å“ {peak_value:.2f} kW aus {len(hourly_production[peak_hour])} Werten)"
            )
            
            return f"{peak_hour:02d}:00"
            
        except Exception as e:
            _LOGGER.warning(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Peak-Zeit Berechnung fehlgeschlagen: {e}")
            return "12:00"  # Safe fallback
    
    def is_production_hours(self, hour: int = None) -> bool:
        """
        PrÃƒÂ¼ft ob aktuell Produktionsstunden sind.
        
        Args:
            hour: Stunde zum PrÃƒÂ¼fen (optional, default: jetzt)
            
        Returns:
            True wenn Produktionsstunden

        """
        try:
            if hour is None:
                hour = dt_util.utcnow().hour
            
            return self.PRODUCTION_START_HOUR <= hour <= self.PRODUCTION_END_HOUR
            
        except Exception:
            return True  # Safe fallback
    
    def estimate_remaining_production_hours(self) -> float:
        """
        SchÃƒÂ¤tzt verbleibende Produktionsstunden für heute.
        
        Returns:
            GeschÃƒÂ¤tzte verbleibende Stunden

        """
        try:
            now = dt_util.utcnow()
            current_hour = now.hour
            
            # Nach Produktionsende
            if current_hour >= self.PRODUCTION_END_HOUR:
                return 0.0
            
            # Vor Produktionsbeginn
            if current_hour < self.PRODUCTION_START_HOUR:
                return float(self.PRODUCTION_END_HOUR - self.PRODUCTION_START_HOUR)
            
            # WÃƒÂ¤hrend Produktion
            remaining = self.PRODUCTION_END_HOUR - current_hour
            
            # BerÃƒÂ¼cksichtige Minuten
            remaining -= now.minute / 60.0
            
            return max(0.0, remaining)
            
        except Exception as e:
            _LOGGER.warning(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Remaining hours Berechnung fehlgeschlagen: {e}")
            return 8.0  # Safe fallback


# ============================================================================
# Ã¢Å“â€œ STRATEGIE 2: PRODUCTION TIME CALCULATOR MIT LIVE-TRACKING
# ============================================================================

class ProductionTimeCalculator:
    """
    Live-Tracking der Produktionszeit ÃƒÂ¼ber State Changes.
    Ã¢Å“â€œ STRATEGIE 2: Echtes Live-Tracking statt History-Queries

    """
    
    def __init__(self, hass: HomeAssistant, power_entity: Optional[str] = None):
        """
        Initialisiere ProductionTimeCalculator.
        
        Args:
            hass: HomeAssistant Instanz
            power_entity: Power-Sensor Entity-ID (optional)

        """
        self.hass = hass
        self.power_entity = power_entity
        
        # Tracking State
        self._is_active = False
        self._start_time: Optional[datetime] = None
        self._accumulated_hours = 0.0
        self._last_production_time: Optional[datetime] = None
        self._zero_power_start: Optional[datetime] = None
        self._today_total_hours = 0.0
        
        # Konstanten
        self.MIN_POWER_THRESHOLD = 10.0  # Watt
        self.ZERO_POWER_THRESHOLD = 1.0  # Watt
        self.ZERO_POWER_TIMEOUT = timedelta(minutes=5)  # 5 Minuten
        
        # Listener Cleanup
        self._state_listener_remove = None
        self._midnight_listener_remove = None
        
        _LOGGER.info("Ã¢Å“â€œ ProductionTimeCalculator initialisiert")
    
    def start_tracking(self) -> None:
        """
        Startet Produktionszeit-Tracking.

        """
        if not self.power_entity:
            _LOGGER.info("ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¹ÃƒÂ¯Ã‚Â¸Ã‚Â Kein Power-Entity - Produktionszeit-Tracking deaktiviert")
            return
        
        try:
            # State Change Listener für Power-Entity
            self._state_listener_remove = async_track_state_change_event(
                self.hass,
                [self.power_entity],
                self._handle_power_change
            )
            
            # Midnight Listener für Reset
            self._midnight_listener_remove = async_track_time_change(
                self.hass,
                self._handle_midnight_reset,
                hour=0,
                minute=0,
                second=0
            )
            
            _LOGGER.info(f"Ã¢Å“â€œ Produktionszeit-Tracking gestartet für {self.power_entity}")
            
        except Exception as e:
            _LOGGER.error(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Fehler beim Starten des Trackings: {e}")
    
    @callback
    def _handle_power_change(self, event) -> None:
        """
        Callback für Power State Changes.

        """
        try:
            new_state = event.data.get("new_state")
            if not new_state or new_state.state in ["unavailable", "unknown"]:
                return
            
            try:
                power = float(new_state.state)
            except (ValueError, TypeError):
                return
            
            now = dt_util.utcnow()
            
            # Power ÃƒÂ¼ber Threshold ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ Start/Continue Tracking
            if power >= self.MIN_POWER_THRESHOLD:
                if not self._is_active:
                    # Start neuer Produktionsphase
                    self._is_active = True
                    self._start_time = now
                    _LOGGER.debug(f"ÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¢ Produktions-Start: {power}W")
                
                # Reset Zero-Power Timer
                self._zero_power_start = None
                self._last_production_time = now
            
            # Power unter Threshold
            elif self._is_active:
                # PrÃƒÂ¼fe ob unter Zero-Threshold
                if power < self.ZERO_POWER_THRESHOLD:
                    # Start Zero-Power Timer wenn noch nicht gestartet
                    if self._zero_power_start is None:
                        self._zero_power_start = now
                        _LOGGER.debug(f"ÃƒÂ¢Ã‚ÂÃ‚Â±ÃƒÂ¯Ã‚Â¸Ã‚Â Zero-Power Timer gestartet: {power}W")
                    
                    # PrÃƒÂ¼fe Timeout
                    elif now - self._zero_power_start >= self.ZERO_POWER_TIMEOUT:
                        # Stoppe Tracking
                        self._stop_production_tracking(now)
                        _LOGGER.debug(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â´ Produktions-Ende nach 5 Min Timeout")
                
                # Zwischen Thresholds ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ Continue aber kein Zero-Timer
                else:
                    self._zero_power_start = None
            
        except Exception as e:
            _LOGGER.warning(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Fehler in Power Change Handler: {e}")
    
    def _stop_production_tracking(self, stop_time: datetime) -> None:
        """
        Stoppt Produktions-Tracking und speichert Stunden.

        """
        if not self._is_active or not self._start_time:
            return
        
        # Berechne Dauer
        duration = stop_time - self._start_time
        hours = duration.total_seconds() / 3600.0
        
        # Addiere zu akkumulierten Stunden
        self._accumulated_hours += hours
        self._today_total_hours = self._accumulated_hours
        
        _LOGGER.info(
            f"Ã¢Å“â€œ Produktionsphase beendet: {hours:.2f}h "
            f"(Gesamt heute: {self._today_total_hours:.2f}h)"
        )
        
        # Reset State
        self._is_active = False
        self._start_time = None
        self._zero_power_start = None
    
    @callback
    def _handle_midnight_reset(self, now: datetime) -> None:
        """
        Callback für Mitternacht-Reset.

        """
        _LOGGER.info(f"ÃƒÂ°Ã…Â¸Ã…â€™Ã¢â€žÂ¢ Mitternacht-Reset: Heute {self._today_total_hours:.2f}h produziert")
        
        # Wenn noch aktiv, stoppe zuerst
        if self._is_active:
            self._stop_production_tracking(now)
        
        # Reset für neuen Tag
        self._accumulated_hours = 0.0
        self._today_total_hours = 0.0
        self._is_active = False
        self._start_time = None
        self._last_production_time = None
        self._zero_power_start = None
    
    def get_production_time(self) -> str:
        """
        Gibt aktuelle Produktionszeit als formatierter String zurÃƒÂ¼ck.
        
        Returns:
            String wie "6h 30m" oder Status-Meldung

        """
        try:
            # Wenn keine Power-Entity konfiguriert
            if not self.power_entity:
                return "Nicht verfÃƒÂ¼gbar"
            
            # Berechne aktuelle Gesamtzeit
            total_hours = self._accumulated_hours
            
            # Wenn aktuell aktiv, addiere laufende Zeit
            if self._is_active and self._start_time:
                now = dt_util.utcnow()
                current_duration = now - self._start_time
                total_hours += current_duration.total_seconds() / 3600.0
            
            # Wenn noch keine Produktion
            if total_hours < 0.01:  # weniger als ~30 Sekunden
                return "0h 0m"
            
            # Formatiere als Stunden und Minuten
            hours = int(total_hours)
            minutes = int((total_hours - hours) * 60)
            
            return f"{hours}h {minutes}m"
            
        except Exception as e:
            _LOGGER.warning(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Fehler beim Abrufen der Produktionszeit: {e}")
            return "Fehler"
    
    def get_production_hours_float(self) -> float:
        """
        Gibt Produktionszeit als Float (Stunden) zurÃƒÂ¼ck.
        
        Returns:
            Stunden als Float

        """
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
        """
        PrÃƒÂ¼ft ob aktuell produziert wird.
        
        Returns:
            True wenn aktiv

        """
        return self._is_active
    
    def stop_tracking(self) -> None:
        """
        Stoppt Tracking und räumt Listener auf.

        """
        if self._state_listener_remove:
            self._state_listener_remove()
            self._state_listener_remove = None
        
        if self._midnight_listener_remove:
            self._midnight_listener_remove()
            self._midnight_listener_remove = None
        
        _LOGGER.info("Ã¢Å“â€œ Produktionszeit-Tracking gestoppt")
