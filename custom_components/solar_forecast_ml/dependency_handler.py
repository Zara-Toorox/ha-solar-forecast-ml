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
# Version 4.9.4 - Async Fix for Blocking I/O
from __future__ import annotations

import logging
from typing import Any

# Fix: importlib.metadata instead of getattr()
try:
    from importlib.metadata import version as get_version
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import version as get_version

_LOGGER = logging.getLogger(__name__)


# Dependencies that are required
REQUIRED_DEPENDENCIES = {
    "numpy": "1.21.0",
    "aiofiles": "23.0.0",
}


class DependencyHandler:
    """
    Handler for dependency checks.
    
    Simplified: Only check, no installation.
    Home Assistant installs automatically from manifest.json # by Zara
    """
    
    def __init__(self) -> None:
        """Initialize dependency handler # by Zara"""
        self._checked = False
        self._all_satisfied = False
        self._package_status = {}
    
    def _check_package_sync(self, package: str) -> bool:
        """
        Synchronous check if a package is installed and functional.
        This is a BLOCKING function intended to be run in an executor.
        
        Args:
            package: Package name (e.g., "numpy")
            
        Returns:
            True if package works # by Zara
        """
        try:
            if package == "numpy":
                import numpy as np
                # Test basic functionality
                test_array = np.array([1, 2, 3])
                _ = test_array.mean()
                _LOGGER.debug(f"✓ {package} is functional (Version: {np.__version__})")
                return True
            elif package == "aiofiles":
                import aiofiles
                _LOGGER.debug(f"✓ {package} is functional")
                return True
            else:
                # For other packages: standard import
                __import__(package)
                _LOGGER.debug(f"✓ {package} is installed")
                return True
                
        except Exception as e:
            _LOGGER.warning(f"✗ {package} is not available: {e}")
            return False
    
    async def check_dependencies(self, hass) -> bool:
        """
        Check all dependencies asynchronously.
        
        Args:
            hass: HomeAssistant instance for async_add_executor_job
            
        Returns:
            True if all are present # by Zara
        """
        if self._checked:
            _LOGGER.debug(f"ℹ️ Dependencies already checked: {self._all_satisfied}")
            return self._all_satisfied
        
        _LOGGER.info("🔍 Checking dependencies...")
        
        missing_deps = []
        
        for package in REQUIRED_DEPENDENCIES.keys():
            # Run the blocking import check in an executor
            is_ok = await hass.async_add_executor_job(
                self._check_package_sync, package
            )
            self._package_status[package] = is_ok
            if not is_ok:
                missing_deps.append(package)
        
        if not missing_deps:
            _LOGGER.info("✓ All dependencies are present")
            self._checked = True
            self._all_satisfied = True
            return True
        
        _LOGGER.warning(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
        _LOGGER.info("ℹ️ Home Assistant should install these automatically on the next restart")
        self._checked = True
        self._all_satisfied = False
        return False
    
    def _get_package_version_sync(self, package: str) -> str:
        """
        Blocking function to get the package version.
        Will be run in the executor # by Zara
        """
        try:
            return get_version(package)
        except Exception:
            # Fallback for packages without metadata
            if package == "numpy":
                try:
                    import numpy as np
                    return np.__version__
                except:
                    pass
            return "unknown"
    
    async def get_dependency_status(self, hass=None) -> dict[str, Any]:
        """
        Get the status of all dependencies.
        Async version with executor for blocking I/O # by Zara
        
        Args:
            hass: HomeAssistant instance for async_add_executor_job
            
        Returns:
            Dict with package status # by Zara
        """
        status = {}
        
        for package, min_version in REQUIRED_DEPENDENCIES.items():
            # Use the already checked status if available
            is_satisfied = self._package_status.get(package)
            
            # If not checked yet (e.g., direct call), run the check
            if is_satisfied is None:
                if hass:
                    is_satisfied = await hass.async_add_executor_job(
                        self._check_package_sync, package
                    )
                else:
                    is_satisfied = self._check_package_sync(package)
                self._package_status[package] = is_satisfied

            # Get version
            if hass:
                version = await hass.async_add_executor_job(
                    self._get_package_version_sync, package
                )
            else:
                # Fallback without hass (e.g., tests)
                version = self._get_package_version_sync(package)
            
            status[package] = {
                "installed": is_satisfied,
                "version": version,
                "required": min_version,
                "satisfied": is_satisfied,
            }
        
        return status