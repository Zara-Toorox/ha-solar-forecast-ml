"""
Low-level Data I/O for Solar Forecast ML Integration.
Handles thread-safe file operations (read, write, directories).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""
import asyncio
import json
import logging
import shutil
import aiofiles
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from homeassistant.core import HomeAssistant

# Use constants for versioning, accessible by inheriting classes
from .const import DATA_VERSION
# Import specific exception types
from .exceptions import DataIntegrityException, create_context

_LOGGER = logging.getLogger(__name__)


class DataManagerIO:
    """
    Base class providing asynchronous, thread-safe file I/O operations
    for JSON data using atomic writes and an asyncio Lock.
    Blocking file operations are executed in a ThreadPoolExecutor.
    """

    def __init__(self, hass: HomeAssistant, data_dir: Path):
        """
        Initialize the I/O manager.

        Args:
            hass: HomeAssistant instance.
            data_dir: The base directory for data storage.
        """
        self.hass = hass
        self.data_dir = Path(data_dir) # Ensure it's a Path object
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="DataManagerIO")
        self._file_lock = asyncio.Lock() # Lock for critical read-modify-write sequences

        _LOGGER.debug("DataManagerIO initialized for directory: %s", self.data_dir)

    async def _ensure_directory_exists(self, directory: Path) -> None:
        """
        Ensure a directory exists, creating it if necessary. (Non-blocking)
        Uses executor for the blocking mkdir operation.
        """
        try:
            # Check existence first (often faster if dir exists)
            exists = await self.hass.async_add_executor_job(directory.is_dir)
            if not exists:
                _LOGGER.info("Creating directory: %s", directory)
                await self.hass.async_add_executor_job(
                    lambda: directory.mkdir(parents=True, exist_ok=True)
                )
        except Exception as e:
            raise DataIntegrityException(
                f"Failed to ensure directory exists: {directory}",
                context=create_context(directory=str(directory), error=str(e))
            )

    async def _get_file_size(self, file_path: Path) -> int:
        """
        Get the size of a file in bytes. (Non-blocking)
        Returns 0 if the file doesn't exist or an error occurs.
        Uses executor for the blocking stat operation.
        """
        try:
            if await self._file_exists(file_path):
                stat_result = await self.hass.async_add_executor_job(file_path.stat)
                return stat_result.st_size
            return 0
        except Exception as e:
            _LOGGER.warning("Could not get file size for %s: %s", file_path.name, e)
            return 0

    async def _file_exists(self, file_path: Path) -> bool:
        """
        Check if a file exists. (Non-blocking)
        Uses executor for the blocking exists check.
        """
        try:
            return await self.hass.async_add_executor_job(file_path.exists)
        except Exception as e:
            _LOGGER.warning("Could not check file existence for %s: %s", file_path.name, e)
            return False

    async def _atomic_write_json_unlocked(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Internal function for atomic JSON writing. Assumes the caller holds `_file_lock`.
        Uses a temporary file and `shutil.move` for atomicity.
        Blocking file operations (move, unlink) are run in the executor.
        """
        temp_file = file_path.with_suffix(f'.tmp_{asyncio.current_task().get_name()}') # Unique temp name
        try:
            # Asynchronously write JSON data to the temporary file
            async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                # Use compact encoding for potentially smaller files if needed, or indent for readability
                json_data = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
                await f.write(json_data)
                await f.flush() # Ensure data is written before moving

            # Atomically replace the target file with the temporary file (blocking operation)
            await self.hass.async_add_executor_job(
                shutil.move, str(temp_file), str(file_path)
            )
            _LOGGER.debug("Atomic write (unlocked) successful for: %s", file_path.name)

        except Exception as e:
            _LOGGER.error("Atomic write failed for %s: %s", file_path.name, e)
            # Clean up the temporary file if it exists (blocking operation)
            if await self._file_exists(temp_file):
                _LOGGER.debug("Attempting to clean up temporary file: %s", temp_file)
                try:
                    await self.hass.async_add_executor_job(temp_file.unlink)
                except Exception as unlink_e:
                    _LOGGER.warning("Failed to remove temporary file %s: %s", temp_file, unlink_e)
            # Re-raise the original exception wrapped in DataIntegrityException
            raise DataIntegrityException(
                f"Failed atomic write to {file_path.name}: {str(e)}",
                context=create_context(file=str(file_path), error=str(e), temp_file=str(temp_file))
            )
        finally:
            # Ensure temp file is removed even if move fails unexpectedly (belt-and-suspenders)
             if await self._file_exists(temp_file):
                 try: await self.hass.async_add_executor_job(temp_file.unlink)
                 except: pass


    async def _atomic_write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Public, thread-safe method for atomically writing JSON data to a file.
        Acquires the `_file_lock` before writing.
        """
        async with self._file_lock:
            await self._atomic_write_json_unlocked(file_path, data)

    async def _read_json_file(self, file_path: Path, default_structure: Dict | None = None) -> Dict[str, Any]:
        """
        Reads JSON data from a file asynchronously. (Non-blocking file read)
        Handles file not found, empty file, and JSON decode errors gracefully.
        Returns a default structure in case of errors or if the file is missing/empty.
        Does NOT attempt to write or create files.
        """
        if default_structure is None:
            default_structure = {} # Default to empty dict if not specified

        try:
            # Check existence using non-blocking helper
            if not await self._file_exists(file_path):
                _LOGGER.debug("File %s not found, returning default structure.", file_path.name)
                return default_structure # Return default if file doesn't exist

            # Read file content asynchronously
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

                # Handle empty file
                if not content:
                     _LOGGER.warning("File %s is empty, returning default structure.", file_path.name)
                     return default_structure # Return default if file is empty

            # Decode JSON content (potentially blocking for large files, run in executor)
            try:
                data = await self.hass.async_add_executor_job(json.loads, content)
                if not isinstance(data, dict):
                     _LOGGER.warning("JSON content in %s is not a dictionary, returning default.", file_path.name)
                     return default_structure
                return data
            except json.JSONDecodeError as e:
                _LOGGER.error("Invalid JSON content in %s: %s. Returning default structure.", file_path.name, e)
                return default_structure # Return default on decode error

        except FileNotFoundError:
             # Should be caught by _file_exists, but handle defensively
             _LOGGER.warning("File %s not found during read attempt, returning default.", file_path.name)
             return default_structure
        except Exception as e:
            # Catch other potential I/O or unexpected errors
            _LOGGER.error("Failed to read JSON file %s: %s", file_path.name, e, exc_info=True)
            # Wrap in DataIntegrityException? Maybe too harsh, return default for resilience.
            # raise DataIntegrityException(f"Failed to read {file_path.name}: {str(e)}")
            return default_structure # Return default on other errors


    async def cleanup(self) -> None:
        """
        Cleans up resources, specifically shutting down the ThreadPoolExecutor.
        Should be called during integration unload. (Non-blocking shutdown trigger)
        """
        try:
            _LOGGER.info("Shutting down DataManagerIO Executor...")
            # Use executor job to run the blocking shutdown method
            await self.hass.async_add_executor_job(self._executor.shutdown, True) # True waits for tasks
            _LOGGER.info("DataManagerIO Executor shut down successfully.")
        except Exception as e:
            _LOGGER.error("Error during DataManagerIO cleanup: %s", e, exc_info=True)