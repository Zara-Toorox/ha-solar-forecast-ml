"""Extra Features Installation Service for Solar Forecast ML V12.2.0 @zara

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
import shutil
from pathlib import Path
from typing import List, Tuple

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class ExtraFeaturesInstaller:
    """Handles installation of extra feature components."""

    # Extra features available for installation
    EXTRA_FEATURES = [
        "grid_price_monitor",
        "sfml_stats",
        "sfml_stats_lite",
    ]

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the installer."""
        self.hass = hass
        self._source_base = Path(__file__).parent.parent / "extra_features"
        self._target_base = Path(__file__).parent.parent.parent  # custom_components/

    async def install_all(self) -> Tuple[List[str], List[str]]:
        """Install all extra features.

        Returns:
            Tuple of (installed_list, failed_list)
        """
        installed = []
        failed = []

        for feature in self.EXTRA_FEATURES:
            success = await self._install_feature(feature)
            if success:
                installed.append(feature)
            else:
                failed.append(feature)

        return installed, failed

    async def _install_feature(self, feature_name: str) -> bool:
        """Install a single extra feature.

        Args:
            feature_name: Name of the feature directory

        Returns:
            True if successful, False otherwise
        """
        source_path = self._source_base / feature_name
        target_path = self._target_base / feature_name

        # Validate source exists
        if not source_path.exists():
            _LOGGER.error(f"Extra feature source not found: {source_path}")
            return False

        # Check if manifest.json exists (valid component)
        if not (source_path / "manifest.json").exists():
            _LOGGER.error(f"Invalid component (no manifest.json): {feature_name}")
            return False

        try:
            # Check if already installed
            if target_path.exists():
                _LOGGER.info(f"Feature '{feature_name}' already installed at {target_path}")
                # Update existing installation
                await self.hass.async_add_executor_job(
                    self._copy_directory, source_path, target_path
                )
                _LOGGER.info(f"Updated existing installation: {feature_name}")
            else:
                # Fresh installation
                await self.hass.async_add_executor_job(
                    self._copy_directory, source_path, target_path
                )
                _LOGGER.info(f"Installed new feature: {feature_name}")

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to install '{feature_name}': {e}")
            return False

    def _copy_directory(self, source: Path, target: Path) -> None:
        """Copy directory recursively (synchronous, runs in executor).

        Args:
            source: Source directory path
            target: Target directory path
        """
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)

    def get_installation_status(self) -> dict:
        """Get current installation status of all extra features.

        Returns:
            Dict with feature names and their installation status
        """
        status = {}
        for feature in self.EXTRA_FEATURES:
            source_exists = (self._source_base / feature).exists()
            target_exists = (self._target_base / feature).exists()

            if target_exists:
                status[feature] = "installed"
            elif source_exists:
                status[feature] = "available"
            else:
                status[feature] = "missing"

        return status
