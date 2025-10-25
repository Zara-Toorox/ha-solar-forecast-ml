"""
Dependency handler for Solar Forecast ML integration.

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
# Version 4.9.3 - Async Fix für Blocking I/O
from __future__ import annotations

import logging
from typing import Any

# Fix: importlib.metadata statt getattr()
try:
    from importlib.metadata import version as get_version
except ImportError:
    # Fallback für Python < 3.8
    from importlib_metadata import version as get_version

_LOGGER = logging.getLogger(__name__)


# Dependencies die benötigt werden
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
                # Test grundlegende Funktionalität
                test_array = np.array([1, 2, 3])
                _ = test_array.mean()
                _LOGGER.debug(f"✓ {package} funktioniert (Version: {np.__version__})")
                return True
            elif package == "aiofiles":
                import aiofiles
                _LOGGER.debug(f"✓ {package} funktioniert")
                return True
            else:
                # Für andere Packages: Standard-Import
                __import__(package)
                _LOGGER.debug(f"✓ {package} installiert")
                return True
                
        except Exception as e:
            _LOGGER.warning(f"âŒ {package} nicht verfügbar: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """
        Prüfe alle Dependencies.
        
        Returns:
            True wenn alle vorhanden # von Zara
        """
        if self._checked:
            _LOGGER.debug(f"â„¹ï¸ Dependencies bereits geprüft: {self._all_satisfied}")
            return self._all_satisfied
        
        _LOGGER.info("ðŸ” Prüfe Dependencies...")
        
        missing_deps = []
        
        for package in REQUIRED_DEPENDENCIES.keys():
            is_ok = self.check_package(package)
            self._package_status[package] = is_ok
            if not is_ok:
                missing_deps.append(package)
        
        if not missing_deps:
            _LOGGER.info("✓ Alle Dependencies vorhanden")
            self._checked = True
            self._all_satisfied = True
            return True
        
        _LOGGER.warning(f"âš ï¸ Fehlende Dependencies: {', '.join(missing_deps)}")
        _LOGGER.info("â„¹ï¸ Home Assistant installiert diese automatisch beim nächsten Neustart")
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
            # Fallback für Packages ohne Metadaten
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
            
            # Fix: Blocking I/O in Executor ausführen
            if hass:
                version = await hass.async_add_executor_job(
                    self._get_package_version_sync, package
                )
            else:
                # Fallback ohne hass (z.B. Tests)
                version = self._get_package_version_sync(package)
            
            status[package] = {
                "installed": is_satisfied,
                "version": version,
                "required": min_version,
                "satisfied": is_satisfied,
            }
        
        return status
