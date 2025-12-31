# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from __future__ import annotations

import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)

REQUIRED_DEPENDENCIES = {
    "numpy": "1.21.0",
    "aiofiles": "23.0.0",
}

class DependencyHandler:
    """Handler for dependency checks"""

    def __init__(self) -> None:
        """Initialize dependency handler @zara"""
        self._checked = False
        self._all_satisfied = False
        self._package_status = {}

    def _check_package_sync(self, package: str) -> bool:
        """Synchronous check if a package is installed and functional @zara"""
        try:
            if package == "numpy":
                import numpy as np

                test_array = np.array([1, 2, 3])
                _ = test_array.mean()
                _LOGGER.debug(f"[OK] {package} is functional (Version: {np.__version__})")
                return True
            elif package == "aiofiles":
                import aiofiles

                _LOGGER.debug(f"[OK] {package} is functional")
                return True
            else:

                __import__(package)
                _LOGGER.debug(f"[OK] {package} is installed")
                return True

        except Exception as e:
            _LOGGER.warning(f"[FAIL] {package} is not available: {e}")
            return False

    async def check_dependencies(self, hass=None) -> bool:
        """Check all dependencies asynchronously if hass provided or synchronously @zara"""
        if self._checked:
            _LOGGER.debug(f"Dependencies already checked: {self._all_satisfied}")
            return self._all_satisfied

        _LOGGER.info("Checking dependencies...")

        missing_deps = []

        for package in REQUIRED_DEPENDENCIES.keys():
            if hass:

                is_ok = await hass.async_add_executor_job(self._check_package_sync, package)
            else:

                is_ok = self._check_package_sync(package)

            self._package_status[package] = is_ok
            if not is_ok:
                missing_deps.append(package)

        if not missing_deps:
            _LOGGER.info("[OK] All dependencies are present")
            self._checked = True
            self._all_satisfied = True
            return True

        _LOGGER.warning(f"[WARN] Missing dependencies: {', '.join(missing_deps)}")
        _LOGGER.info("Home Assistant should install these automatically on the next restart")
        self._checked = True
        self._all_satisfied = False
        return False

    def _get_package_version_sync(self, package: str) -> str:
        """Blocking function to get the package version @zara"""

        try:
            from importlib.metadata import version as get_version
        except ImportError:

            try:
                from importlib_metadata import version as get_version
            except ImportError:
                _LOGGER.warning("Could not import 'importlib.metadata' or 'importlib_metadata'.")
                return "unknown (import error)"

        try:
            return get_version(package)
        except Exception:

            if package == "numpy":
                try:
                    import numpy as np

                    return np.__version__
                except Exception as e:
                    _LOGGER.debug(f"Could not get numpy version: {e}")
            return "unknown"

    async def get_dependency_status(self, hass=None) -> dict[str, Any]:
        """Get the status of all dependencies @zara"""
        status = {}

        for package, min_version in REQUIRED_DEPENDENCIES.items():

            is_satisfied = self._package_status.get(package)

            if is_satisfied is None:
                if hass:
                    is_satisfied = await hass.async_add_executor_job(
                        self._check_package_sync, package
                    )
                else:
                    is_satisfied = self._check_package_sync(package)
                self._package_status[package] = is_satisfied

            if hass:
                version = await hass.async_add_executor_job(self._get_package_version_sync, package)
            else:

                version = self._get_package_version_sync(package)

            status[package] = {
                "installed": is_satisfied,
                "version": version,
                "required": min_version,
                "satisfied": is_satisfied,
            }

        return status

    def get_installed_packages(self) -> list[str]:
        """Get list of installed package names @zara"""
        if not self._checked:
            _LOGGER.warning("Dependencies not checked yet, returning empty list")
            return []

        return [pkg for pkg, status in self._package_status.items() if status]

    def get_missing_packages(self) -> list[str]:
        """Get list of missing package names @zara"""
        if not self._checked:
            _LOGGER.warning("Dependencies not checked yet, returning empty list")
            return []

        return [pkg for pkg, status in self._package_status.items() if not status]
