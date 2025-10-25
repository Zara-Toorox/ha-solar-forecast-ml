"""
Helper-Modul für externe Sensor-Anzeigen.
Gemeinsame Basis-Klasse und Hilfsfunktionen für externe Sensoren.
Version 1.0 - von Zara

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
import logging
from datetime import datetime
from typing import Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import callback
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)


def format_time_ago(last_changed: datetime) -> str:
    """
    Formatiert Zeitstempel als 'vor X Min/Std' - von Zara
    Platzsparender: <1min = '>1min' - von Zara
    
    Args:
        last_changed: Zeitpunkt der letzten Änderung
        
    Returns:
        Formatierter String wie "vor 5 Min." oder "vor 2 Std."
    """
    now = dt_util.utcnow()
    delta = now - last_changed
    
    if delta.total_seconds() < 60:
        return ">1min"  # Platzsparender für <1 Minute - von Zara
    elif delta.total_seconds() < 3600:
        minutes = int(delta.total_seconds() / 60)
        return f"vor {minutes} Min."
    else:
        hours = int(delta.total_seconds() / 3600)
        return f"vor {hours} Std."


class BaseExternalSensor:
    """
    Gemeinsame Basis für externe Sensor-Anzeigen mit LIVE-Updates - von Zara
    
    Diese Klasse implementiert:
    - LIVE State Change Tracking
    - Einheitliche Fehlerbehandlung
    - Zeitstempel-Formatierung
    - Verfügbarkeitsprüfung
    """
    
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    
    def __init__(self, coordinator, entry: ConfigEntry, sensor_config: dict):
        """
        Initialisiert externen Sensor - von Zara
        
        Args:
            coordinator: DataUpdateCoordinator
            entry: ConfigEntry
            sensor_config: Dict mit Konfiguration:
                - config_key: Key für Config (z.B. CONF_TEMP_SENSOR)
                - unique_id_suffix: Suffix für unique_id
                - name: Anzeigename
                - icon: MDI Icon
                - unit_key: Key für Einheit aus Attributes (optional)
                - default_unit: Standard-Einheit falls nicht gefunden (optional)
                - format_string: Format-String für Anzeige (optional)
        """
        # Muss von abgeleiteter Klasse bereits aufgerufen sein
        self._sensor_config = sensor_config
        self.entry = entry
        self.coordinator = coordinator
        
        # Attribute setzen
        self._attr_unique_id = f"{entry.entry_id}_{sensor_config['unique_id_suffix']}"
        self._attr_name = sensor_config['name']
        self._attr_icon = sensor_config['icon']
    
    @property
    def available(self) -> bool:
        """Externe Sensoren immer verfügbar (zeigen eigene Status-Meldungen) - von Zara"""
        return True
    
    async def async_added_to_hass(self) -> None:
        """Registriert LIVE-Update Listener - von Zara"""
        await super().async_added_to_hass()
        
        sensor_entity_id = self.entry.data.get(self._sensor_config['config_key'])
        if sensor_entity_id:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass,
                    [sensor_entity_id],
                    self._handle_external_sensor_update
                )
            )
    
    @callback
    def _handle_external_sensor_update(self, event) -> None:
        """Triggert Update bei Änderung des externen Sensors - von Zara"""
        self.async_write_ha_state()
    
    @property
    def native_value(self) -> str:
        """
        Holt Wert vom konfigurierten Sensor mit Zeitstempel - von Zara
        
        Returns:
            Formatierter String mit Wert, Einheit und Zeitstempel
            oder Fehlermeldung
        """
        sensor_entity_id = self.entry.data.get(self._sensor_config['config_key'])
        
        if not sensor_entity_id:
            return "Kein externer Sensor vorhanden"
        
        state = self.coordinator.hass.states.get(sensor_entity_id)
        if not state:
            return "Sensor nicht gefunden"
        
        try:
            # Prüfe Verfügbarkeit
            if state.state in ['unavailable', 'unknown', 'none', None]:
                return "Sensor nicht verfügbar"
            
            # Formatiere Zeitstempel
            time_ago = format_time_ago(state.last_changed)
            
            # Hole Einheit
            unit = self._get_unit(state)
            
            # Formatiere Ausgabe
            return self._format_value(state.state, unit, time_ago)
            
        except Exception as e:
            _LOGGER.warning(f"Fehler beim Lesen des {self._sensor_config['name']}: {e}")
            return "Fehler beim Auslesen"
    
    def _get_unit(self, state) -> Optional[str]:
        """
        Ermittelt Einheit des Sensors - von Zara
        
        Args:
            state: State-Objekt des Sensors
            
        Returns:
            Einheit oder None
        """
        unit_key = self._sensor_config.get('unit_key', 'unit_of_measurement')
        default_unit = self._sensor_config.get('default_unit')
        
        if default_unit is None:
            # Kein default_unit = keine Einheit verwenden
            return None
        
        return state.attributes.get(unit_key, default_unit)
    
    def _format_value(self, value: str, unit: Optional[str], time_ago: str) -> str:
        """
        Formatiert Sensor-Wert für Anzeige - von Zara
        
        Args:
            value: Sensor-Wert
            unit: Einheit (optional)
            time_ago: Zeitstempel-String
            
        Returns:
            Formatierter String
        """
        format_string = self._sensor_config.get('format_string', '{value} {unit} ({time})')
        
        # Wenn spezifisches Format definiert
        if '{value}' in format_string:
            result = format_string.replace('{value}', str(value))
            if unit:
                result = result.replace('{unit}', unit)
            else:
                result = result.replace(' {unit}', '')  # Entferne unit-Platzhalter
            result = result.replace('{time}', time_ago)
            return result
        
        # Standard-Format
        if unit:
            return f"{value} {unit} ({time_ago})"
        else:
            return f"{value} ({time_ago})"
