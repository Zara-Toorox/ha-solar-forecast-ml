# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""Data validator for SFML Stats."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles

from ..const import (
    EXPORT_DIRECTORIES,
    SFML_STATS_BASE,
    SOLAR_FORECAST_ML_BASE,
    SOLAR_FORECAST_ML_STATS,
    SOLAR_FORECAST_ML_AI,
    GRID_PRICE_MONITOR_BASE,
    GRID_PRICE_MONITOR_DATA,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class DataValidator:
    """Validates and creates required directory structures. @zara"""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the data validator. @zara"""
        self._hass = hass
        self._config_path = Path(hass.config.path())
        self._initialized = False
        self._source_status: dict[str, bool] = {}

    @property
    def config_path(self) -> Path:
        """Return the Home Assistant config path. @zara"""
        return self._config_path

    @property
    def export_base_path(self) -> Path:
        """Return the SFML Stats export base path. @zara"""
        return self._config_path / SFML_STATS_BASE

    @property
    def is_initialized(self) -> bool:
        """Check if the validator has been initialized. @zara"""
        return self._initialized

    @property
    def source_status(self) -> dict[str, bool]:
        """Return the status of source integrations. @zara"""
        return self._source_status.copy()

    async def async_initialize(self) -> bool:
        """Initialize the directory structure. @zara"""
        _LOGGER.info("Initializing SFML Stats directory structure")

        try:
            await self._validate_sources()
            await self._create_export_directories()
            await self._create_gitignore()

            self._initialized = True
            _LOGGER.info(
                "SFML Stats directory structure initialized: %s",
                self.export_base_path
            )
            return True

        except Exception as err:
            _LOGGER.error("Error during initialization: %s", err)
            return False

    async def _validate_sources(self) -> None:
        """Validate availability of source integrations. @zara"""
        sources = {
            "solar_forecast_ml": {
                "base": SOLAR_FORECAST_ML_BASE,
                "required_dirs": [SOLAR_FORECAST_ML_STATS, SOLAR_FORECAST_ML_AI],
            },
            "grid_price_monitor": {
                "base": GRID_PRICE_MONITOR_BASE,
                "required_dirs": [GRID_PRICE_MONITOR_DATA],
            },
        }

        for source_name, source_config in sources.items():
            base_path = self._config_path / source_config["base"]
            is_available = await self._hass.async_add_executor_job(base_path.exists)

            if is_available:
                for required_dir in source_config["required_dirs"]:
                    dir_path = self._config_path / required_dir
                    dir_exists = await self._hass.async_add_executor_job(dir_path.exists)
                    if not dir_exists:
                        _LOGGER.warning(
                            "Source directory not found: %s",
                            dir_path
                        )
                        is_available = False
                        break

            self._source_status[source_name] = is_available

            if is_available:
                _LOGGER.info("Source available: %s (%s)", source_name, base_path)
            else:
                _LOGGER.warning(
                    "Source not available: %s - some statistics will not be generated",
                    source_name
                )

    async def _create_export_directories(self) -> None:
        """Create all export directories. @zara"""
        for directory in EXPORT_DIRECTORIES:
            dir_path = self._config_path / directory
            dir_exists = await self._hass.async_add_executor_job(dir_path.exists)
            if not dir_exists:
                await self._hass.async_add_executor_job(
                    lambda p=dir_path: p.mkdir(parents=True, exist_ok=True)
                )
                _LOGGER.debug("Directory created: %s", dir_path)
            else:
                _LOGGER.debug("Directory already exists: %s", dir_path)

    async def _create_gitignore(self) -> None:
        """Create a .gitignore in the export folder. @zara"""
        gitignore_path = self.export_base_path / ".gitignore"

        gitignore_exists = await self._hass.async_add_executor_job(gitignore_path.exists)
        if not gitignore_exists:
            gitignore_content = """# SFML Stats - Auto-generated files
.cache/
*.png
*.tmp
*.temp
*.log
"""
            async with aiofiles.open(gitignore_path, "w") as f:
                await f.write(gitignore_content)
            _LOGGER.debug(".gitignore created: %s", gitignore_path)

    def get_source_path(self, source: str, subpath: str | Path = "") -> Path | None:
        """Return the path to a source file. @zara"""
        if not self._source_status.get(source, False):
            return None

        base_paths = {
            "solar_forecast_ml": SOLAR_FORECAST_ML_BASE,
            "grid_price_monitor": GRID_PRICE_MONITOR_BASE,
        }

        base = base_paths.get(source)
        if base is None:
            return None

        return self._config_path / base / subpath

    def get_export_path(self, subpath: str | Path = "") -> Path:
        """Return the path to an export file. @zara"""
        return self.export_base_path / subpath

    async def async_validate_file_readable(self, file_path: Path) -> bool:
        """Check if a file is readable. @zara"""
        try:
            exists = await self._hass.async_add_executor_job(file_path.exists)
            if not exists:
                return False
            is_file = await self._hass.async_add_executor_job(file_path.is_file)
            return is_file
        except Exception:
            return False

    async def async_get_directory_tree(self) -> dict:
        """Return the current directory structure as a dict. @zara"""
        tree = {
            "export_base": str(self.export_base_path),
            "initialized": self._initialized,
            "sources": self._source_status.copy(),
            "directories": {},
        }

        for directory in EXPORT_DIRECTORIES:
            dir_path = self._config_path / directory
            dir_exists = await self._hass.async_add_executor_job(dir_path.exists)
            tree["directories"][str(directory)] = {
                "exists": dir_exists,
                "path": str(dir_path),
                "files": [],
            }

            if dir_exists:
                try:
                    def get_files(p: Path) -> list[str]:
                        """Helper for blocking file operations. @zara"""
                        return [f.name for f in p.glob("*") if f.is_file()]

                    file_names = await self._hass.async_add_executor_job(
                        get_files, dir_path
                    )
                    tree["directories"][str(directory)]["files"] = file_names
                except Exception:
                    pass

        return tree

    def __repr__(self) -> str:
        """Return string representation of the validator. @zara"""
        return (
            f"DataValidator("
            f"config_path={self._config_path}, "
            f"initialized={self._initialized}, "
            f"sources={self._source_status})"
        )
