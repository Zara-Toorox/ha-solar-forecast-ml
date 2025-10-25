"""
Service Manager für Solar Forecast ML.
Managed Lifecycle aller Services (ML, Weather, Notification, Error Handler).
FIX: Keine doppelte Startup-Benachrichtigung mehr
Version 4.9.2 - Doppelte Notification Fix

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
from typing import Optional, Any
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


class ServiceManager:
    """
    Managed Lifecycle und Initialisierung aller Services.
    Kapselt Service-Management-Logik aus Coordinator.
    
    PATCH: Verbesserte Service-Initialisierung und Validation

    """
    
    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        data_manager,
        weather_entity: str,
        dependencies_ok: bool = False,  # NEU: Dependencies-Status
    ):
        """
        Initialisiere Service Manager.
        
        Args:
            hass: HomeAssistant Instanz
            entry: ConfigEntry
            data_manager: DataManager Instanz
            weather_entity: Weather Entity ID

        """
        self.hass = hass
        self.entry = entry
        self.data_manager = data_manager
        self.weather_entity = weather_entity
        
        # Service References
        self.error_handler: Optional[Any] = None
        self.weather_service: Optional[Any] = None
        self.ml_predictor: Optional[Any] = None
        self.notification_service: Optional[Any] = None
        
        # Status Flags
        self._services_initialized = False
        self._ml_ready = False
        self._initialization_lock = asyncio.Lock()
        
        # Dependency Status (wird von auÃƒÆ’Ã…Â¸en gesetzt)
        self.dependencies_installed = dependencies_ok
    
    # ========================================================================
    # GEÄNDERTER ABSCHNITT VON ZARA - FIX Problem 6: Bessere Service-Init
    # ========================================================================
    async def initialize_all_services(self) -> bool:
        """
        Initialisiert alle Services in korrekter Reihenfolge.
        
        Thread-safe mit Lock für parallele Aufrufe.
        
        PATCH: Verbesserte Fehlerbehandlung und Validation
        
        Returns:
            True wenn erfolgreich initialisiert

        """
        async with self._initialization_lock:
            if self._services_initialized:
                _LOGGER.debug(" Services bereits initialisiert")
                return True
            
            _LOGGER.info(" Initialisiere Services...")
            
            try:
                # STEP 1: Error Handler (immer benötigt)
                error_handler_ok = await self._initialize_error_handler()
                if not error_handler_ok:
                    _LOGGER.warning("Â  Error Handler Initialisierung fehlgeschlagen - fahre fort")
                
                # STEP 2: Notification Service
                # NEU: Versuche zuerst aus hass.data zu holen (falls bereits in __init__.py initialisiert)
                if not self.notification_service:
                    self.notification_service = self.hass.data.get(DOMAIN, {}).get("notification_service")
                    if self.notification_service:
                        _LOGGER.info(" NotificationService aus hass.data übernommen")
                    else:
                        # Falls nicht vorhanden, initialisiere neu
                        notif_ok = await self._initialize_notification_service()
                        if not notif_ok:
                            _LOGGER.warning("Â  Notification Service nicht verfügbar")
                else:
                    _LOGGER.debug(" Notification Service bereits gesetzt")
                
                # STEP 3: Weather Service
                weather_ok = await self._initialize_weather_service()
                if not weather_ok:
                    _LOGGER.warning("Â  Weather Service Initialisierung fehlgeschlagen - fahre fort")
                
                # STEP 4: ML Predictor (nur wenn Dependencies OK)
                # DEBUG: Check Dependencies
                # ML Predictor initialisieren
                ml_ok = await self._initialize_ml_predictor()
                if not ml_ok:
                    _LOGGER.warning("Â  ML Predictor nicht verfügbar - Fallback aktiv")
                    self._ml_ready = False

                
                self._services_initialized = True
                _LOGGER.info(f" Services initialisiert - ML Ready: {self._ml_ready}")
                
                # NEU: Validiere Service-Status
                self._log_service_status()
                return True
                
            except Exception as e:
                _LOGGER.error(f" Service Initialisierung fehlgeschlagen: {e}", exc_info=True)
                return False
    
    # ========================================================================
    # ENDE GEÄNDEERTER ABSCHNITT
    # ========================================================================
    
    def _log_service_status(self):
        """
        NEU: Logge Status aller Services für Debugging.

        """
        _LOGGER.info(" Service Status:")
        _LOGGER.info(f"  - Error Handler: {'' if self.error_handler else ''}")
        _LOGGER.info(f"  - Notification Service: {'' if self.notification_service else ''}")
        _LOGGER.info(f"  - Weather Service: {'' if self.weather_service else ''}")
        _LOGGER.info(f"  - ML Predictor: {'' if self.ml_predictor else ''}")
        _LOGGER.info(f"  - ML Ready: {'' if self._ml_ready else ''}")
    
    async def _initialize_error_handler(self) -> bool:
        """
        Initialisiert Error Handler Service.
        
       PATCH: Gibt Success Status zurück

        """
        try:
            from .error_handling_service import ErrorHandlingService
            
            # FIX von Zara: Keine Parameter mehr übergeben
            self.error_handler = ErrorHandlingService()
            _LOGGER.debug(" Error Handler initialisiert")
            return True
            
        except Exception as e:
            _LOGGER.error(f" Error Handler Initialisierung fehlgeschlagen: {e}", exc_info=True)
            return False
    
    async def _initialize_notification_service(self) -> bool:
        """
        Initialisiert Notification Service.
        
        PATCH: Gibt Success Status zurück

        """
        try:
            from .notification_service import NotificationService
            
            self.notification_service = NotificationService(
                self.hass,
                self.entry.entry_id
            )
            
            _LOGGER.debug(" Notification Service initialisiert")
            return True
            
        except Exception as e:
            _LOGGER.error(f" Notification Service Initialisierung fehlgeschlagen: {e}", exc_info=True)
            return False
    
    async def _initialize_weather_service(self) -> bool:
        """
        Initialisiert Weather Service.
        
        PATCH: Gibt Success Status zurück

        """
        try:
            from .weather_service import WeatherService
            
            # Prüfe ob Weather Entity existiert
            weather_state = self.hass.states.get(self.weather_entity)
            if not weather_state:
                _LOGGER.error(f" Weather Entity nicht gefunden: {self.weather_entity}")
                return False
            
            self.weather_service = WeatherService(
                self.hass,
                self.weather_entity,
                self.error_handler
            )
            
            # Initialisiere Weather Service
            weather_init_success = await self.weather_service.initialize()
            
            if weather_init_success:
                _LOGGER.debug(" Weather Service initialisiert")
                return True
            else:
                _LOGGER.error(" Weather Service Initialisierung fehlgeschlagen")
                return False
                
        except Exception as e:
            _LOGGER.warning(f"Â  Weather Service Initialisierung fehlgeschlagen: {e}")
            return False
    
    async def _initialize_ml_predictor(self) -> bool:
        """
        Initialisiert ML Predictor mit Dependencies.
        
        PATCH: Gibt Success Status zurück und bessere Validation
        FIX: Keine ML-Activation Benachrichtigung mehr (wird bereits in __init__.py gesendet)

        """
        try:
            # NEU: Prüfe ZUERST ob Dependencies vorhanden sind - von Zara
            if not self.dependencies_installed:
                _LOGGER.warning(
                    "âš ï¸ ML Dependencies fehlen (numpy, aiofiles) - "
                    "ML Predictor wird nicht initialisiert"
                )
                self._ml_ready = False
                return False
            
            from .ml_predictor import MLPredictor
            
            _LOGGER.info(" Initialisiere ML Predictor mit Dependencies...")
            
            # NEU: Validiere DataManager ist initialisiert
            if not self.data_manager:
                _LOGGER.error(" DataManager nicht verfügbar - ML kann nicht initialisiert werden")
                return False
            
            self.ml_predictor = MLPredictor(
                self.hass,
                self.data_manager,
                self.error_handler
            )
            
            # Initialize ML Predictor
            ml_init_success = await self.ml_predictor.initialize()
            
            if ml_init_success:
                self._ml_ready = True
                _LOGGER.info("ML Predictor erfolgreich initialisiert")
                
                # FIX: KEINE ML-Activation Benachrichtigung mehr hier // Basti
                # Die Startup-Benachrichtigung in __init__.py zeigt bereits den ML-Status
                
                return True
            else:
                _LOGGER.warning("ML Predictor Initialisierung fehlgeschlagen - Fallback aktiv")
                self._ml_ready = False
                return False
                
        except ImportError as e:
            _LOGGER.error(f" ML Predictor Import fehlgeschlagen (Dependencies fehlen?): {e}")
            self._ml_ready = False
            return False
        except Exception as e:
            _LOGGER.error(f" ML Predictor Initialisierung fehlgeschlagen: {e}", exc_info=True)
            self._ml_ready = False
            return False
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def is_ml_ready(self) -> bool:
        """
        Prüft ob ML bereit ist.
        
        Returns:
            True wenn ML Predictor verfügbar und gesund

        """
        if not self._ml_ready or not self.ml_predictor:
            return False
        
        # NEU: Zusätzliche Health-Check nur wenn verfügbar
        try:
            if hasattr(self.ml_predictor, 'is_healthy'):
                return self.ml_predictor.is_healthy()
            else:
                # Wenn keine is_healthy Methode, gehe davon aus dass es OK ist
                return True
        except Exception as e:
            _LOGGER.debug(f" ML Health check fehlgeschlagen: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """
        Prüft ob Services initialisiert sind.
        
        Returns:
            True wenn initialisiert

        """
        return self._services_initialized
    
    def get_service_status(self) -> dict[str, Any]:
        """
        NEU: Gibt Status aller Services zurück für Debugging/UI.
        
        Returns:
            Dict mit Service-Status

        """
        return {
            "initialized": self._services_initialized,
            "error_handler_available": self.error_handler is not None,
            "notification_service_available": self.notification_service is not None,
            "weather_service_available": self.weather_service is not None,
            "ml_predictor_available": self.ml_predictor is not None,
            "ml_ready": self._ml_ready,
            "dependencies_installed": self.dependencies_installed,
        }
