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
# Version 4.9.3 - Async Fix fÃ¼r Blocking I/O
from __future__ import annotations

import logging
from typing import Any

# Fix: importlib.metadata statt getattr()
try:
    from importlib.metadata import version as get_version
except ImportError:
    # Fallback fÃ¼r Python < 3.8
    from importlib_metadata import version as get_version

_LOGGER = logging.getLogger(__name__)


# Dependencies die benÃ¶tigt werden
REQUIRED_DEPENDENCIES = {
    "numpy": "1.21.0",
    "aiofiles": "23.0.0",
}


class DependencyHandler:
    """
    Handler fÃ¼r AbhÃ¤ngigkeitsprÃ¼fung.
    
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
        PrÃ¼fe ob Package installiert und funktionsfÃ¤hig ist.
        
        Args:
            package: Package-Name (z.B. "numpy")
            
        Returns:
            True wenn Package funktioniert # von Zara
        """
        try:
            if package == "numpy":
                import numpy as np
                # Test grundlegende FunktionalitÃ¤t
                test_array = np.array([1, 2, 3])
                _ = test_array.mean()
                _LOGGER.debug(f"âœ“ {package} funktioniert (Version: {np.__version__})")
                return True
            elif package == "aiofiles":
                import aiofiles
                _LOGGER.debug(f"âœ“ {package} funktioniert")
                return True
            else:
                # FÃ¼r andere Packages: Standard-Import
                __import__(package)
                _LOGGER.debug(f"âœ“ {package} installiert")
                return True
                
        except Exception as e:
            _LOGGER.warning(f"Ã¢ÂÅ’ {package} nicht verfÃ¼gbar: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """
        PrÃ¼fe alle Dependencies.
        
        Returns:
            True wenn alle vorhanden # von Zara
        """
        if self._checked:
            _LOGGER.debug(f"Ã¢â€žÂ¹Ã¯Â¸Â Dependencies bereits geprÃ¼ft: {self._all_satisfied}")
            return self._all_satisfied
        
        _LOGGER.info("Ã°Å¸â€Â PrÃ¼fe Dependencies...")
        
        missing_deps = []
        
        for package in REQUIRED_DEPENDENCIES.keys():
            is_ok = self.check_package(package)
            self._package_status[package] = is_ok
            if not is_ok:
                missing_deps.append(package)
        
        if not missing_deps:
            _LOGGER.info("âœ“ Alle Dependencies vorhanden")
            self._checked = True
            self._all_satisfied = True
            return True
        
        _LOGGER.warning(f"Ã¢Å¡Â Ã¯Â¸Â Fehlende Dependencies: {', '.join(missing_deps)}")
        _LOGGER.info("Ã¢â€žÂ¹Ã¯Â¸Â Home Assistant installiert diese automatisch beim nÃ¤chsten Neustart")
        self._checked = True
        self._all_satisfied = False
        return False
    
    def _get_package_version_sync(self, package: str) -> str:
        """
        Blocking-Funktion zum Holen der Package-Version.
        Wird im Executor ausgefÃ¼hrt # von Zara
        """
        try:
            return get_version(package)
        except Exception:
            # Fallback fÃ¼r Packages ohne Metadaten
            if package == "numpy":
                try:
                    import numpy as np
                    return np.__version__
                except:
                    pass
            return "unknown"
    
    async def get_dependency_status(self, hass=None) -> dict[str, Any]:
        """
        Hole Status aller AbhÃ¤ngigkeiten.
        Async-Version mit Executor fÃ¼r Blocking I/O # von Zara
        
        Args:
            hass: HomeAssistant instance fÃ¼r async_add_executor_job
            
        Returns:
            Dict mit Package-Status # von Zara
        """
        status = {}
        
        for package, min_version in REQUIRED_DEPENDENCIES.items():
            is_satisfied = self.check_package(package)
            
            # Fix: Blocking I/O in Executor ausfÃ¼hren
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
