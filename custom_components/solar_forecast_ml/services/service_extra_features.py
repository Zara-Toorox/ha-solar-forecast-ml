# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class ExtraFeaturesInstaller:
    """Handles installation and auto-update of extra feature components @zara

    Features:
    - Dynamic discovery: Finds all subfolders with manifest.json automatically
    - Version-based updates: Only copies when source is newer than installed
    - Auto-sync on SFML update: Called during async_setup_entry
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the installer."""
        self.hass = hass
        self._source_base = Path(__file__).parent.parent / "extra_features"
        self._target_base = Path(__file__).parent.parent.parent  # custom_components/

    def _discover_extra_features(self) -> List[str]:
        """Dynamically discover all extra features with manifest.json @zara

        Returns:
            List of feature directory names that have a valid manifest.json
        """
        features = []

        if not self._source_base.exists():
            return features

        for item in self._source_base.iterdir():
            if item.is_dir() and (item / "manifest.json").exists():
                features.append(item.name)

        return sorted(features)

    def _get_version_from_manifest(self, manifest_path: Path) -> Optional[str]:
        """Extract version from manifest.json @zara"""
        try:
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    data = json.load(f)
                    return data.get("version")
        except Exception:
            pass
        return None

    def _compare_versions(self, source_version: Optional[str], target_version: Optional[str]) -> bool:
        """Check if source version is newer than target version @zara

        Returns:
            True if source is newer or target doesn't exist, False otherwise
        """
        if target_version is None:
            return True  # Not installed yet
        if source_version is None:
            return False  # Can't determine source version

        # Simple version comparison (works for semver like "1.0.0", "1.2.3")
        try:
            source_parts = [int(x) for x in source_version.split(".")]
            target_parts = [int(x) for x in target_version.split(".")]

            # Pad shorter version with zeros
            max_len = max(len(source_parts), len(target_parts))
            source_parts.extend([0] * (max_len - len(source_parts)))
            target_parts.extend([0] * (max_len - len(target_parts)))

            return source_parts > target_parts
        except (ValueError, AttributeError):
            # If version parsing fails, compare as strings
            return source_version != target_version

    async def sync_on_update(self) -> Tuple[List[str], List[str]]:
        """Sync extra features if SFML was updated @zara

        Only copies features where the source version is newer than installed.
        Called automatically during async_setup_entry.

        Returns:
            Tuple of (updated_list, skipped_list)
        """
        updated = []
        skipped = []

        features = await self.hass.async_add_executor_job(self._discover_extra_features)

        if not features:
            _LOGGER.debug("No extra features found in extra_features/")
            return updated, skipped

        for feature in features:
            source_path = self._source_base / feature
            target_path = self._target_base / feature

            source_manifest = source_path / "manifest.json"
            target_manifest = target_path / "manifest.json"

            # Get versions
            source_version = await self.hass.async_add_executor_job(
                self._get_version_from_manifest, source_manifest
            )
            target_version = await self.hass.async_add_executor_job(
                self._get_version_from_manifest, target_manifest
            )

            # Check if update needed
            needs_update = self._compare_versions(source_version, target_version)

            if needs_update:
                success = await self._install_feature(feature)
                if success:
                    if target_version:
                        _LOGGER.info(
                            f"Extra feature '{feature}' updated: {target_version} → {source_version}"
                        )
                    else:
                        _LOGGER.info(
                            f"Extra feature '{feature}' installed: v{source_version}"
                        )
                    updated.append(feature)
                else:
                    skipped.append(feature)
            else:
                _LOGGER.debug(
                    f"Extra feature '{feature}' is up-to-date (v{target_version})"
                )
                skipped.append(feature)

        if updated:
            _LOGGER.info(
                f"Extra features sync complete: {len(updated)} updated, {len(skipped)} up-to-date"
            )

        return updated, skipped

    async def install_all(self) -> Tuple[List[str], List[str]]:
        """Install/update all extra features (force mode) @zara

        Used by the manual service call - always copies regardless of version.

        Returns:
            Tuple of (installed_list, failed_list)
        """
        installed = []
        failed = []

        features = await self.hass.async_add_executor_job(self._discover_extra_features)

        if not features:
            _LOGGER.warning("No extra features found in extra_features/")
            return installed, failed

        for feature in features:
            success = await self._install_feature(feature)
            if success:
                installed.append(feature)
            else:
                failed.append(feature)

        return installed, failed

    async def _install_feature(self, feature_name: str) -> bool:
        """Install a single extra feature @zara

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
            await self.hass.async_add_executor_job(
                self._copy_directory, source_path, target_path
            )
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to install '{feature_name}': {e}")
            return False

    def _copy_directory(self, source: Path, target: Path) -> None:
        """Copy directory recursively (synchronous, runs in executor) @zara

        Args:
            source: Source directory path
            target: Target directory path
        """
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)

    def get_installation_status(self) -> Dict[str, dict]:
        """Get current installation status of all extra features @zara

        Returns:
            Dict with feature names and their installation status + versions
        """
        status = {}
        features = self._discover_extra_features()

        for feature in features:
            source_path = self._source_base / feature
            target_path = self._target_base / feature

            source_version = self._get_version_from_manifest(source_path / "manifest.json")
            target_version = self._get_version_from_manifest(target_path / "manifest.json")

            if target_path.exists():
                if self._compare_versions(source_version, target_version):
                    state = "update_available"
                else:
                    state = "installed"
            else:
                state = "available"

            status[feature] = {
                "state": state,
                "source_version": source_version,
                "installed_version": target_version,
            }

        return status
