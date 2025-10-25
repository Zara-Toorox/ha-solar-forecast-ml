"""
Notification Service für Solar Forecast ML Integration.
✅ PRODUCTION READY: Async Factory Pattern
✅ FIX: Korrigierter Import ohne None-Fehler
✅ NEU: Zeigt installierte Dependencies in Startbenachrichtigung # von Zara
Version 4.8.3 # von Zara

Copyright (C) 2025 Zara-Toorox
"""
import logging
import asyncio
from typing import Optional, List
from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Notification IDs # von Zara
NOTIFICATION_ID_DEPENDENCIES = "solar_forecast_ml_dependencies"
NOTIFICATION_ID_INSTALLATION = "solar_forecast_ml_installation"
NOTIFICATION_ID_SUCCESS = "solar_forecast_ml_success"
NOTIFICATION_ID_ERROR = "solar_forecast_ml_error"
NOTIFICATION_ID_ML_ACTIVE = "solar_forecast_ml_ml_active"
NOTIFICATION_ID_STARTUP = "solar_forecast_ml_startup"


class NotificationService:
    """
    Service für Persistent Notifications in Home Assistant.
    ✅ ASYNC: Alle Methoden non-blocking
    ✅ SAFE: Error handling für alle Notification-Operationen
    ✅ FIX: Korrigierter Import-Mechanismus # von Zara
    """
    
    def __init__(self, hass: HomeAssistant):
        """
        Initialisiert Notification Service.
        
        ⚠️ NICHT DIREKT AUFRUFEN: Verwende create_notification_service()
        # von Zara
        """
        self.hass = hass
        self._initialized = False
        self._notification_lock = asyncio.Lock()
        _LOGGER.debug("🔧 NotificationService Instanz erstellt")
    
    async def initialize(self) -> bool:
        """
        Initialisiert den Notification Service.
        
        Returns:
            True wenn erfolgreich initialisiert
        # von Zara
        """
        try:
            async with self._notification_lock:
                if self._initialized:
                    _LOGGER.debug("✔ NotificationService bereits initialisiert")
                    return True
                
                # Prüfe ob persistent_notification Component geladen ist # von Zara
                if 'persistent_notification' not in self.hass.config.components:
                    _LOGGER.warning(
                        "⚠️ persistent_notification nicht verfügbar - "
                        "Notifications werden nicht angezeigt"
                    )
                    self._initialized = True
                    return False
                
                self._initialized = True
                _LOGGER.info("✅ NotificationService erfolgreich initialisiert")
                return True
                
        except Exception as e:
            _LOGGER.error(
                f"❌ Fehler bei NotificationService Initialisierung: {e}",
                exc_info=True
            )
            return False
    
    # ========================================================================
    # 🔧 FIX: Neuer Import-Mechanismus ohne None-Probleme # von Zara
    # ========================================================================
    async def _safe_create_notification(
        self,
        message: str,
        title: str,
        notification_id: str
    ) -> bool:
        """
        Erstellt Notification mit Error Handling.
        
        ✅ FIX: Verwendet hass.services.async_call statt direktem Import
        # von Zara
        """
        if not self._initialized:
            _LOGGER.warning(
                f"⚠️ NotificationService nicht initialisiert - "
                f"Notification '{notification_id}' wird nicht angezeigt"
            )
            return False
        
        try:
            # ✅ FIX: Verwende hass.services.async_call statt problematischem Import # von Zara
            await self.hass.services.async_call(
                domain="persistent_notification",
                service="create",
                service_data={
                    "message": message,
                    "title": title,
                    "notification_id": notification_id,
                },
                blocking=True,
            )
            _LOGGER.debug(f"🔧 Notification '{notification_id}' erstellt")
            return True
            
        except Exception as e:
            _LOGGER.error(
                f"❌ Fehler beim Erstellen von Notification '{notification_id}': {e}",
                exc_info=True
            )
            return False
    
    async def _safe_dismiss_notification(self, notification_id: str) -> bool:
        """
        Entfernt Notification mit Error Handling.
        
        ✅ FIX: Verwendet hass.services.async_call
        # von Zara
        """
        if not self._initialized:
            return False
        
        try:
            # ✅ FIX: Verwende hass.services.async_call # von Zara
            await self.hass.services.async_call(
                domain="persistent_notification",
                service="dismiss",
                service_data={
                    "notification_id": notification_id,
                },
                blocking=True,
            )
            _LOGGER.debug(f"🔧 Notification '{notification_id}' entfernt")
            return True
            
        except Exception as e:
            _LOGGER.warning(
                f"⚠️ Fehler beim Entfernen von Notification '{notification_id}': {e}"
            )
            return False
    # ========================================================================
    # ENDE FIX
    # ========================================================================
    
    async def show_startup_success(
        self, 
        ml_mode: bool = True,
        installed_packages: Optional[List[str]] = None,  # von Zara
        missing_packages: Optional[List[str]] = None
    ) -> bool:
        """
        Zeigt Start-Benachrichtigung mit Status der Integration.
        
        Args:
            ml_mode: True wenn ML-Features aktiv, False bei Fallback
            installed_packages: Liste installierter Pakete (informativ) # von Zara
            missing_packages: Optional - Liste fehlender Pakete bei Fallback
            
        Returns:
            True wenn erfolgreich
        # von Zara
        """
        try:
            # Erstelle Liste installierter Dependencies (informativ) # von Zara
            installed_list = ""
            if installed_packages:
                installed_items = "\n".join([f"- ✅ {pkg}" for pkg in installed_packages])
                installed_list = f"\n**Installierte Abhängigkeiten:**\n{installed_items}\n"
            
            if ml_mode:
                # ✅ ML-Mode aktiv - alle Dependencies vorhanden # von Zara
                message = f"""
**🥳 Solar Forecast ML erfolgreich gestartet!**

**Status: Full ML Mode ✅**

Die Integration läuft mit allen Features:
{installed_list}
✅ **Machine Learning aktiv**
- Ridge Regression Model
- 28 Features für Prognosen
- Automatisches Learning
- Pattern Recognition

✅ **Features aktiv**
- ML-basierte Vorhersagen
- Wetterbasierte Optimierung
- Kontinuierliche Verbesserung
- Seasonal Adjustments

**Die Integration ist einsatzbereit!**

Das ML-Model lernt kontinuierlich aus echten Daten und verbessert sich über die Zeit.
"""
                title = "✅ Solar Forecast ML - Erfolgreich gestartet"
                
            else:
                # ⚠️ Fallback-Mode - Dependencies fehlen # von Zara
                missing_list = ""
                if missing_packages:
                    missing_items = "\n".join([f"- ❌ {pkg}" for pkg in missing_packages])
                    missing_list = f"\n**Fehlende Pakete:**\n{missing_items}\n"
                
                message = f"""
**⚠️ Solar Forecast ML gestartet - Fallback Mode**

**Status: Basis-Funktionen aktiv**

Die Integration läuft, aber ML-Features sind nicht verfügbar:
{installed_list}{missing_list}

**Aktuelle Features:**
✅ Basis-Prognosen (regelbasiert)
✅ Wetterintegration
✅ Tagesproduktions-Berechnung
❌ ML-basierte Optimierung (fehlt)
❌ Pattern Recognition (fehlt)

**Lösung:**

Klicken Sie auf den Button **"ML-Abhängigkeiten installieren"** in den Geräteeinstellungen, um alle Features zu aktivieren.

Nach der Installation und Neustart:
- ML-Features werden aktiviert
- Präzisere Vorhersagen
- Automatisches Learning
"""
                title = "⚠️ Solar Forecast ML - Fallback Mode"
            
            return await self._safe_create_notification(
                message=message,
                title=title,
                notification_id=NOTIFICATION_ID_STARTUP
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Fehler bei show_startup_success: {e}", exc_info=True)
            return False
    
    async def show_dependencies_missing(self, missing_packages: List[str]) -> bool:
        """
        Zeigt Notification über fehlende Dependencies.
        
        Args:
            missing_packages: Liste fehlender Pakete
            
        Returns:
            True wenn erfolgreich
        # von Zara
        """
        try:
            # Erstelle Liste # von Zara
            missing_list = "\n".join([f"- {pkg}" for pkg in missing_packages])
            
            message = f"""
**⚠️ Solar Forecast ML - Dependencies fehlen**

Folgende Python-Pakete werden benötigt:

{missing_list}

**Was bedeutet das?**

Die Integration läuft im **Fallback-Modus** mit Basis-Funktionen. ML-Features sind nicht verfügbar.

**Lösung:**

Klicken Sie auf den Button **"ML-Abhängigkeiten installieren"** in den Geräteeinstellungen.

Die fehlenden Pakete werden automatisch installiert. Nach einem Neustart sind alle Features verfügbar.

**Alternative: Manuelle Installation**

Falls der Button nicht funktioniert, installieren Sie manuell per SSH:

```
docker exec homeassistant pip install --break-system-packages numpy aiofiles
```

Danach Home Assistant neu starten.
"""
            
            return await self._safe_create_notification(
                message=message,
                title="⚠️ Solar Forecast ML - Dependencies fehlen",
                notification_id=NOTIFICATION_ID_DEPENDENCIES
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Fehler bei show_dependencies_missing: {e}", exc_info=True)
            return False
    
    async def show_installation_progress(self, status: str, progress: int) -> bool:
        """
        Zeigt Installation Progress als Notification.
        
        Args:
            status: Status-Text
            progress: Fortschritt 0-100
            
        Returns:
            True wenn erfolgreich
        # von Zara
        """
        try:
            # Progress Bar mit Unicode-Zeichen # von Zara
            bar_length = 20
            filled = int(bar_length * progress / 100)
            bar = "â–ˆ" * filled + "â–'" * (bar_length - filled)
            
            message = f"""
**🔧 Installation läuft...**

{bar} {progress}%

**Status:** {status}

Bitte warten Sie, bis die Installation abgeschlossen ist.
"""
            
            return await self._safe_create_notification(
                message=message,
                title="🔧 Solar Forecast ML - Installation",
                notification_id=NOTIFICATION_ID_INSTALLATION
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Fehler bei show_installation_progress: {e}", exc_info=True)
            return False
    
    async def show_installation_success(self) -> bool:
        """
        Zeigt Success-Notification nach erfolgreicher Installation.
        
        Returns:
            True wenn erfolgreich
        # von Zara
        """
        try:
            # Entferne Progress-Notification # von Zara
            await self._safe_dismiss_notification(NOTIFICATION_ID_INSTALLATION)
            
            message = """
**✅ Installation erfolgreich!**

Alle ML-Abhängigkeiten wurden erfolgreich installiert:
- ✅ numpy installiert
- ✅ aiofiles installiert

**⚠️ Wichtig: Neustart erforderlich**

Bitte starten Sie Home Assistant neu, um die ML-Features zu aktivieren.

Nach dem Neustart:
- ML-Model wird automatisch trainiert
- Erweiterte Prognosen verfügbar
- Pattern Recognition aktiv

Die Integration läuft dann im **Full ML Mode**.
"""
            
            # Zeige neue Success-Notification # von Zara
            return await self._safe_create_notification(
                message=message,
                title="✅ Solar Forecast ML - Installation erfolgreich",
                notification_id=NOTIFICATION_ID_SUCCESS
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Fehler bei show_installation_success: {e}", exc_info=True)
            return False
    
    async def show_installation_error(
        self, 
        error_message: str,
        failed_packages: Optional[List[str]] = None
    ) -> bool:
        """
        Zeigt Error-Notification bei fehlgeschlagener Installation.
        
        Args:
            error_message: Fehler-Beschreibung
            failed_packages: Optional - Liste der fehlgeschlagenen Pakete
            
        Returns:
            True wenn erfolgreich
        # von Zara
        """
        try:
            # Entferne Progress-Notification # von Zara
            await self._safe_dismiss_notification(NOTIFICATION_ID_INSTALLATION)
            
            failed_list = ""
            if failed_packages:
                failed_list = "\n".join([f"- {pkg}" for pkg in failed_packages])
                failed_list = f"\n**Fehlgeschlagene Pakete:**\n{failed_list}\n"
            
            message = f"""
**❌ Installation fehlgeschlagen**

{error_message}
{failed_list}

**Manuelle Installation:**

Verbinden Sie sich per SSH mit Home Assistant und führen Sie aus:

```
docker exec homeassistant pip install --break-system-packages numpy aiofiles
```

Oder falls kein Docker:
```
pip install numpy aiofiles
```

Danach Home Assistant neu starten.

**Hilfe benötigt?**
Prüfen Sie die Logs für Details oder kontaktieren Sie den Support.
"""
            
            return await self._safe_create_notification(
                message=message,
                title="❌ Solar Forecast ML - Installation fehlgeschlagen",
                notification_id=NOTIFICATION_ID_ERROR
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Fehler bei show_installation_error: {e}", exc_info=True)
            return False
    
    async def show_ml_activated(self) -> bool:
        """
        Zeigt Notification wenn ML erfolgreich aktiviert wurde.
        
        Returns:
            True wenn erfolgreich
        # von Zara
        """
        try:
            message = """
**🥳 ML-Features aktiviert!**

Solar Forecast ML läuft jetzt im **Full ML Mode**:

✅ Ridge Regression Model aktiv
✅ 28 Features für Prognosen
✅ Automatisches Learning aktiv
✅ Pattern Recognition aktiv
✅ Seasonal Adjustments

**Was bedeutet das?**

Die Integration verwendet jetzt Machine Learning für:
- Präzisere Vorhersagen
- Anpassung an Ihr System
- Wetterbasierte Optimierung
- Kontinuierliche Verbesserung

Das Model wird automatisch mit echten Daten trainiert und verbessert sich über die Zeit.

**Status:** ML ist einsatzbereit
"""
            
            return await self._safe_create_notification(
                message=message,
                title="🥳 Solar Forecast ML - ML aktiviert",
                notification_id=NOTIFICATION_ID_ML_ACTIVE
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Fehler bei show_ml_activated: {e}", exc_info=True)
            return False
    
    async def dismiss_all(self) -> None:
        """
        Entfernt alle Solar Forecast ML Notifications.
        
        Nützlich beim Cleanup.
        # von Zara
        """
        try:
            notification_ids = [
                NOTIFICATION_ID_DEPENDENCIES,
                NOTIFICATION_ID_INSTALLATION,
                NOTIFICATION_ID_SUCCESS,
                NOTIFICATION_ID_ERROR,
                NOTIFICATION_ID_ML_ACTIVE,
                NOTIFICATION_ID_STARTUP
            ]
            
            for notification_id in notification_ids:
                await self._safe_dismiss_notification(notification_id)
                
            _LOGGER.debug("✅ Alle Notifications entfernt")
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Fehler beim Entfernen aller Notifications: {e}")


async def create_notification_service(hass: HomeAssistant) -> NotificationService:
    """
    Factory-Funktion zum Erstellen eines NotificationService.
    
    ✅ ASYNC: Korrekte Initialisierung
    ✅ SAFE: Error Handling
    
    Args:
        hass: HomeAssistant Instanz
        
    Returns:
        Initialisierter NotificationService
        
    Raises:
        Exception: Bei kritischen Initialisierungsfehlern
    # von Zara
    """
    try:
        service = NotificationService(hass)
        await service.initialize()
        return service
        
    except Exception as e:
        _LOGGER.error(
            f"❌ Fehler beim Erstellen von NotificationService: {e}",
            exc_info=True
        )
        # Gebe Service trotzdem zurück, aber nicht initialisiert # von Zara
        # So kann Integration weiterlaufen ohne Notifications # von Zara
        return NotificationService(hass)
