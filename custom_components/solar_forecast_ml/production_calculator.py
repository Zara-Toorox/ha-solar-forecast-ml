"""
Production Calculator für Solar Forecast ML.
Berechnet Produktionszeit und weitere Produktions-Metriken.
✅ STRATEGIE 2: ProductionTimeCalculator für Live-Tracking # von Zara
Version 4.10.0 - Historische Peak-Zeit Berechnung # von Zara

Copyright (C) 2025 Zara-Toorox
# von Zara
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
    # von Zara
    """
    
    def __init__(self, hass: HomeAssistant):
        """
        Initialisiere Production Calculator.
        
        Args:
            hass: HomeAssistant Instanz
        # von Zara
        """
        self.hass = hass
        
        # Konstanten für Produktions-Berechnung # von Zara
        self.MIN_PRODUCTION_POWER = 0.01  # Minimum Power (kW) für Produktion # von Zara
        self.PRODUCTION_START_HOUR = 5   # Früheste mögliche Produktion # von Zara
        self.PRODUCTION_END_HOUR = 21    # Späteste mögliche Produktion # von Zara
        
        _LOGGER.debug("✅ ProductionCalculator initialisiert")
    
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
        # von Zara
        """
        try:
            # Kein Power-Sensor konfiguriert # von Zara
            if not power_entity:
                _LOGGER.debug("ℹ️ Kein Power-Sensor konfiguriert")
                return "Nicht verfügbar"
            
            # Hole History für heute # von Zara
            now = dt_util.utcnow()
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Hole State History # von Zara
            history = await self.hass.async_add_executor_job(
                self._get_state_history,
                power_entity,
                start_of_day,
                now
            )
            
            if not history:
                _LOGGER.debug("ℹ️ Keine History-Daten verfügbar")
                return "Berechnung läuft..."
            
            # Zähle Zeiträume mit Produktion # von Zara
            production_minutes = 0
            
            for i in range(len(history) - 1):
                try:
                    # Parse Power-Wert # von Zara
                    power = float(history[i].state)
                    
                    # Wenn Power über Minimum # von Zara
                    if power >= self.MIN_PRODUCTION_POWER:
                        # Berechne Zeitdifferenz zum nächsten State # von Zara
                        time_diff = history[i + 1].last_changed - history[i].last_changed
                        production_minutes += time_diff.total_seconds() / 60
                        
                except (ValueError, AttributeError):
                    # Skip ungültige States # von Zara
                    continue
            
            # Konvertiere zu Stunden und Minuten # von Zara
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
        """
        Synchrone Hilfsfunktion für History-Abruf.
        
        Args:
            entity_id: Entity-ID
            start_time: Start-Zeitpunkt
            end_time: End-Zeitpunkt
            
        Returns:
            Liste von States
        # von Zara
        """
        try:
            from homeassistant.components import recorder
            from homeassistant.components.recorder import history
            
            # Prüfe ob Recorder verfügbar # von Zara
            if not recorder.is_entity_recorded(self.hass, entity_id):
                _LOGGER.debug(f"ℹ️ Entity {entity_id} wird nicht aufgezeichnet")
                return []
            
            # Hole History # von Zara
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
        Analysiert letzte 14 Tage und findet Stunde mit höchster Durchschnittsproduktion.
        
        Args:
            power_entity: Power-Sensor Entity-ID (optional)
            
        Returns:
            Peak-Zeit als String (z.B. "12:00")
        # von Zara
        """
        try:
            # Fallback wenn kein Power-Sensor konfiguriert # von Zara
            if not power_entity:
                _LOGGER.debug("ℹ️ Kein Power-Sensor für Peak-Berechnung")
                return "12:00"
            
            # Zeitraum: Letzte 14 Tage # von Zara
            now = dt_util.now()
            start_time = now - timedelta(days=14)
            
            # Hole historische Daten # von Zara
            states = await self.hass.async_add_executor_job(
                self._get_state_history,
                power_entity,
                start_time,
                now
            )
            
            if not states or len(states) < 10:
                _LOGGER.debug("ℹ️ Nicht genug Daten für Peak-Berechnung")
                return "12:00"
            
            # Sammle Produktionswerte pro Stunde # von Zara
            hourly_production = {hour: [] for hour in range(24)}
            
            for state in states:
                try:
                    # Filtere ungültige States # von Zara
                    if state.state in ["unavailable", "unknown", None]:
                        continue
                    
                    power = float(state.state)
                    
                    # Ignoriere Nacht-Werte und sehr kleine Werte # von Zara
                    if power < self.MIN_PRODUCTION_POWER:
                        continue
                    
                    # Konvertiere zu kW falls in W # von Zara
                    if power > 100:  # Vermutlich Watt # von Zara
                        power = power / 1000.0
                    
                    # Extrahiere Stunde # von Zara
                    hour = state.last_changed.hour
                    
                    # Nur Produktionsstunden berücksichtigen # von Zara
                    if self.PRODUCTION_START_HOUR <= hour <= self.PRODUCTION_END_HOUR:
                        hourly_production[hour].append(power)
                
                except (ValueError, TypeError, AttributeError):
                    continue
            
            # Berechne Durchschnitt pro Stunde # von Zara
            hourly_averages = {}
            for hour, values in hourly_production.items():
                if values:  # Nur Stunden mit Daten # von Zara
                    avg = sum(values) / len(values)
                    hourly_averages[hour] = avg
            
            # Finde Stunde mit höchster Durchschnittsproduktion # von Zara
            if not hourly_averages:
                _LOGGER.debug("ℹ️ Keine gültigen Produktionsdaten gefunden")
                return "12:00"
            
            peak_hour = max(hourly_averages, key=hourly_averages.get)
            peak_value = hourly_averages[peak_hour]
            
            _LOGGER.info(
                f"✅ Peak-Stunde gefunden: {peak_hour}:00 Uhr "
                f"(Ø {peak_value:.2f} kW aus {len(hourly_production[peak_hour])} Werten)"
            )
            
            return f"{peak_hour:02d}:00"
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Peak-Zeit Berechnung fehlgeschlagen: {e}")
            return "12:00"  # Safe fallback # von Zara
    
    def is_production_hours(self, hour: int = None) -> bool:
        """
        Prüft ob aktuell Produktionsstunden sind.
        
        Args:
            hour: Stunde zum Prüfen (optional, default: jetzt)
            
        Returns:
            True wenn Produktionsstunden
        # von Zara
        """
        try:
            if hour is None:
                hour = dt_util.utcnow().hour
            
            return self.PRODUCTION_START_HOUR <= hour <= self.PRODUCTION_END_HOUR
            
        except Exception:
            return True  # Safe fallback # von Zara
    
    def estimate_remaining_production_hours(self) -> float:
        """
        Schätzt verbleibende Produktionsstunden für heute.
        
        Returns:
            Geschätzte verbleibende Stunden
        # von Zara
        """
        try:
            now = dt_util.utcnow()
            current_hour = now.hour
            
            # Nach Produktionsende # von Zara
            if current_hour >= self.PRODUCTION_END_HOUR:
                return 0.0
            
            # Vor Produktionsbeginn # von Zara
            if current_hour < self.PRODUCTION_START_HOUR:
                return float(self.PRODUCTION_END_HOUR - self.PRODUCTION_START_HOUR)
            
            # Während Produktion # von Zara
            remaining = self.PRODUCTION_END_HOUR - current_hour
            
            # Berücksichtige Minuten # von Zara
            remaining -= now.minute / 60.0
            
            return max(0.0, remaining)
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Remaining hours Berechnung fehlgeschlagen: {e}")
            return 8.0  # Safe fallback # von Zara


# ============================================================================
# ✅ STRATEGIE 2: PRODUCTION TIME CALCULATOR MIT LIVE-TRACKING # von Zara
# ============================================================================

class ProductionTimeCalculator:
    """
    Live-Tracking der Produktionszeit über State Changes.
    ✅ STRATEGIE 2: Echtes Live-Tracking statt History-Queries # von Zara
    # von Zara
    """
    
    def __init__(self, hass: HomeAssistant, power_entity: Optional[str] = None):
        """
        Initialisiere ProductionTimeCalculator.
        
        Args:
            hass: HomeAssistant Instanz
            power_entity: Power-Sensor Entity-ID (optional)
        # von Zara
        """
        self.hass = hass
        self.power_entity = power_entity
        
        # Tracking State # von Zara
        self._is_active = False
        self._start_time: Optional[datetime] = None
        self._accumulated_hours = 0.0
        self._last_production_time: Optional[datetime] = None
        self._zero_power_start: Optional[datetime] = None
        self._today_total_hours = 0.0
        
        # Konstanten # von Zara
        self.MIN_POWER_THRESHOLD = 10.0  # Watt # von Zara
        self.ZERO_POWER_THRESHOLD = 1.0  # Watt # von Zara
        self.ZERO_POWER_TIMEOUT = timedelta(minutes=5)  # 5 Minuten # von Zara
        
        # Listener Cleanup # von Zara
        self._state_listener_remove = None
        self._midnight_listener_remove = None
        
        _LOGGER.info("✅ ProductionTimeCalculator initialisiert")
    
    async def start_tracking(self) -> None:
        """
        Startet Produktionszeit-Tracking.
        # von Zara
        """
        if not self.power_entity:
            _LOGGER.info("ℹ️ Kein Power-Entity - Produktionszeit-Tracking deaktiviert")
            return
        
        try:
            # State Change Listener für Power-Entity # von Zara
            self._state_listener_remove = async_track_state_change_event(
                self.hass,
                [self.power_entity],
                self._handle_power_change
            )
            
            # Midnight Listener für Reset # von Zara
            self._midnight_listener_remove = async_track_time_change(
                self.hass,
                self._handle_midnight_reset,
                hour=0,
                minute=0,
                second=0
            )
            
            _LOGGER.info(f"✅ Produktionszeit-Tracking gestartet für {self.power_entity}")
            
        except Exception as e:
            _LOGGER.error(f"❌ Fehler beim Starten des Trackings: {e}")
    
    @callback
    def _handle_power_change(self, event) -> None:
        """
        Callback für Power State Changes.
        # von Zara
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
            
            # Power über Threshold → Start/Continue Tracking # von Zara
            if power >= self.MIN_POWER_THRESHOLD:
                if not self._is_active:
                    # Start neuer Produktionsphase # von Zara
                    self._is_active = True
                    self._start_time = now
                    _LOGGER.debug(f"🟢 Produktions-Start: {power}W")
                
                # Reset Zero-Power Timer # von Zara
                self._zero_power_start = None
                self._last_production_time = now
            
            # Power unter Threshold # von Zara
            elif self._is_active:
                # Prüfe ob unter Zero-Threshold # von Zara
                if power < self.ZERO_POWER_THRESHOLD:
                    # Start Zero-Power Timer wenn noch nicht gestartet # von Zara
                    if self._zero_power_start is None:
                        self._zero_power_start = now
                        _LOGGER.debug(f"⏱️ Zero-Power Timer gestartet: {power}W")
                    
                    # Prüfe Timeout # von Zara
                    elif now - self._zero_power_start >= self.ZERO_POWER_TIMEOUT:
                        # Stoppe Tracking # von Zara
                        self._stop_production_tracking(now)
                        _LOGGER.debug(f"🔴 Produktions-Ende nach 5 Min Timeout")
                
                # Zwischen Thresholds → Continue aber kein Zero-Timer # von Zara
                else:
                    self._zero_power_start = None
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Fehler in Power Change Handler: {e}")
    
    def _stop_production_tracking(self, stop_time: datetime) -> None:
        """
        Stoppt Produktions-Tracking und speichert Stunden.
        # von Zara
        """
        if not self._is_active or not self._start_time:
            return
        
        # Berechne Dauer # von Zara
        duration = stop_time - self._start_time
        hours = duration.total_seconds() / 3600.0
        
        # Addiere zu akkumulierten Stunden # von Zara
        self._accumulated_hours += hours
        self._today_total_hours = self._accumulated_hours
        
        _LOGGER.info(
            f"✅ Produktionsphase beendet: {hours:.2f}h "
            f"(Gesamt heute: {self._today_total_hours:.2f}h)"
        )
        
        # Reset State # von Zara
        self._is_active = False
        self._start_time = None
        self._zero_power_start = None
    
    @callback
    def _handle_midnight_reset(self, now: datetime) -> None:
        """
        Callback für Mitternacht-Reset.
        # von Zara
        """
        _LOGGER.info(f"🌙 Mitternacht-Reset: Heute {self._today_total_hours:.2f}h produziert")
        
        # Wenn noch aktiv, stoppe zuerst # von Zara
        if self._is_active:
            self._stop_production_tracking(now)
        
        # Reset für neuen Tag # von Zara
        self._accumulated_hours = 0.0
        self._today_total_hours = 0.0
        self._is_active = False
        self._start_time = None
        self._last_production_time = None
        self._zero_power_start = None
    
    def get_production_time(self) -> str:
        """
        Gibt aktuelle Produktionszeit als formatierter String zurück.
        
        Returns:
            String wie "6h 30m" oder Status-Meldung
        # von Zara
        """
        try:
            # Wenn keine Power-Entity konfiguriert # von Zara
            if not self.power_entity:
                return "Nicht verfügbar"
            
            # Berechne aktuelle Gesamtzeit # von Zara
            total_hours = self._accumulated_hours
            
            # Wenn aktuell aktiv, addiere laufende Zeit # von Zara
            if self._is_active and self._start_time:
                now = dt_util.utcnow()
                current_duration = now - self._start_time
                total_hours += current_duration.total_seconds() / 3600.0
            
            # Wenn noch keine Produktion # von Zara
            if total_hours < 0.01:  # weniger als ~30 Sekunden # von Zara
                return "0h 0m"
            
            # Formatiere als Stunden und Minuten # von Zara
            hours = int(total_hours)
            minutes = int((total_hours - hours) * 60)
            
            return f"{hours}h {minutes}m"
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Fehler beim Abrufen der Produktionszeit: {e}")
            return "Fehler"
    
    def get_production_hours_float(self) -> float:
        """
        Gibt Produktionszeit als Float (Stunden) zurück.
        
        Returns:
            Stunden als Float
        # von Zara
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
        Prüft ob aktuell produziert wird.
        
        Returns:
            True wenn aktiv
        # von Zara
        """
        return self._is_active
    
    async def stop_tracking(self) -> None:
        """
        Stoppt Tracking und räumt Listener auf.
        # von Zara
        """
        if self._state_listener_remove:
            self._state_listener_remove()
            self._state_listener_remove = None
        
        if self._midnight_listener_remove:
            self._midnight_listener_remove()
            self._midnight_listener_remove = None
        
        _LOGGER.info("✅ Produktionszeit-Tracking gestoppt")
