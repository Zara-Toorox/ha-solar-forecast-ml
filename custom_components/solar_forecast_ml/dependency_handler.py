"""Dependency handler for Solar Forecast ML integration."""
# Version 4.9.3 - Async Fix für Blocking I/O # von Zara
from __future__ import annotations

import logging
from typing import Any

# Fix: importlib.metadata statt getattr() # von Zara
try:
    from importlib.metadata import version as get_version
except ImportError:
    # Fallback für Python < 3.8 # von Zara
    from importlib_metadata import version as get_version

_LOGGER = logging.getLogger(__name__)


# Dependencies die benötigt werden # von Zara
REQUIRED_DEPENDENCIES = {
    "numpy": "1.21.0",
    "aiofiles": "23.0.0",
}


class DependencyHandler:
    """
    Handler für Abhängigkeitsprüfung.
    
    Vereinfacht: Nur Check, keine Installation.
    Home Assistant installiert automatisch aus manifest.json # von Zara
    """
    
    def __init__(self) -> None:
        """Initialize dependency handler # von Zara"""
        self._checked = False
        self._all_satisfied = False
        self._package_status = {}
    
    def check_package(self, package: str) -> bool:
        """
        Prüfe ob Package installiert und funktionsfähig ist.
        
        Args:
            package: Package-Name (z.B. "numpy")
            
        Returns:
            True wenn Package funktioniert # von Zara
        """
        try:
            if package == "numpy":
                import numpy as np
                # Test grundlegende Funktionalität # von Zara
                test_array = np.array([1, 2, 3])
                _ = test_array.mean()
                _LOGGER.debug(f"✅ {package} funktioniert (Version: {np.__version__})")
                return True
            elif package == "aiofiles":
                import aiofiles
                _LOGGER.debug(f"✅ {package} funktioniert")
                return True
            else:
                # Für andere Packages: Standard-Import # von Zara
                __import__(package)
                _LOGGER.debug(f"✅ {package} installiert")
                return True
                
        except Exception as e:
            _LOGGER.warning(f"❌ {package} nicht verfügbar: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """
        Prüfe alle Dependencies.
        
        Returns:
            True wenn alle vorhanden # von Zara
        """
        if self._checked:
            _LOGGER.debug(f"ℹ️ Dependencies bereits geprüft: {self._all_satisfied}")
            return self._all_satisfied
        
        _LOGGER.info("🔍 Prüfe Dependencies...")
        
        missing_deps = []
        
        for package in REQUIRED_DEPENDENCIES.keys():
            is_ok = self.check_package(package)
            self._package_status[package] = is_ok
            if not is_ok:
                missing_deps.append(package)
        
        if not missing_deps:
            _LOGGER.info("✅ Alle Dependencies vorhanden")
            self._checked = True
            self._all_satisfied = True
            return True
        
        _LOGGER.warning(f"⚠️ Fehlende Dependencies: {', '.join(missing_deps)}")
        _LOGGER.info("ℹ️ Home Assistant installiert diese automatisch beim nächsten Neustart")
        self._checked = True
        self._all_satisfied = False
        return False
    
    def _get_package_version_sync(self, package: str) -> str:
        """
        Blocking-Funktion zum Holen der Package-Version.
        Wird im Executor ausgeführt # von Zara
        """
        try:
            return get_version(package)
        except Exception:
            # Fallback für Packages ohne Metadaten # von Zara
            if package == "numpy":
                try:
                    import numpy as np
                    return np.__version__
                except:
                    pass
            return "unknown"
    
    async def get_dependency_status(self, hass=None) -> dict[str, Any]:
        """
        Hole Status aller Abhängigkeiten.
        Async-Version mit Executor für Blocking I/O # von Zara
        
        Args:
            hass: HomeAssistant instance für async_add_executor_job
            
        Returns:
            Dict mit Package-Status # von Zara
        """
        status = {}
        
        for package, min_version in REQUIRED_DEPENDENCIES.items():
            is_satisfied = self.check_package(package)
            
            # Fix: Blocking I/O in Executor ausführen # von Zara
            if hass:
                version = await hass.async_add_executor_job(
                    self._get_package_version_sync, package
                )
            else:
                # Fallback ohne hass (z.B. Tests) # von Zara
                version = self._get_package_version_sync(package)
            
            status[package] = {
                "installed": is_satisfied,
                "version": version,
                "required": min_version,
                "satisfied": is_satisfied,
            }
        
        return status
