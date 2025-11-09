"""
Low-Level Data IO Operations

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

import asyncio
import json
import logging
import shutil
import re
from datetime import datetime, date
# --- IMPORT HIER ENTFERNT ---
# import aiofiles (Wird in die Funktionen verschoben)
# --- ENDE ENTFERNUNG ---
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from homeassistant.core import HomeAssistant

# Use constants for versioning, accessible by inheriting classes
from ..const import DATA_VERSION
# Import specific exception types
from ..core.core_exceptions import DataIntegrityException, create_context

_LOGGER = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON Encoder that converts datetime and date objects to ISO format str..."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            if isinstance(obj, datetime):
                # CRITICAL FIX: Force all datetimes to LOCAL timezone
                try:
                    from ..core.core_helpers import SafeDateTimeUtil as dt_util
                    obj = dt_util.ensure_local(obj)
                    # Converting datetime to local timezone (debug log removed)
                except Exception as e:
                    _LOGGER.warning(f"Failed to convert datetime to local timezone: {e}, using original")
            return obj.isoformat()
        return super().default(obj)


class DataManagerIO:
    """Base class providing asynchronous thread-safe file IO operations"""

    def __init__(self, hass: HomeAssistant, data_dir: Path):
        """Initialize the IO manager"""
        self.hass = hass
        self.data_dir = Path(data_dir) # Ensure it's a Path object
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="DataManagerIO")
        # Per-file locks: Each file gets its own lock to prevent unnecessary blocking
        self._file_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # Lock for accessing the locks dictionary
        # DataManagerIO initialized (debug log removed)

    async def _get_file_lock(self, file_path: Path) -> asyncio.Lock:
        """Get or create a lock for a specific file"""
        file_key = str(file_path)  # Use full path as key

        async with self._locks_lock:
            if file_key not in self._file_locks:
                self._file_locks[file_key] = asyncio.Lock()
                # Created new file lock (debug log removed)
            return self._file_locks[file_key]

    async def _ensure_directory_exists(self, directory: Path) -> None:
        """Ensure a directory exists creating it if necessary Non-blocking"""
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
        """Get the size of a file in bytes Non-blocking"""
        try:
            if await self._file_exists(file_path):
                stat_result = await self.hass.async_add_executor_job(file_path.stat)
                return stat_result.st_size
            return 0
        except Exception as e:
            _LOGGER.warning("Could not get file size for %s: %s", file_path.name, e)
            return 0

    async def _file_exists(self, file_path: Path) -> bool:
        """Check if a file exists Non-blocking"""
        try:
            return await self.hass.async_add_executor_job(file_path.exists)
        except Exception as e:
            _LOGGER.warning("Could not check file existence for %s: %s", file_path.name, e)
            return False

    async def _atomic_write_json_unlocked(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Internal function for atomic JSON writing Assumes the caller holds _file_lock"""

        # +++ IMPORT HIER EINGEF +++
        # Wird erst importiert, wenn die Funktion aufgerufen wird
        try:
            import aiofiles
        except ImportError:
            _LOGGER.error("AIOFiles ist nicht installiert. Dateischreiben fehlgeschlagen.")
            raise DataIntegrityException("AIOFiles dependency missing, cannot write file")
        # +++ ENDE EINF +++

        # Ensure parent directory exists
        parent_dir = file_path.parent
        await self._ensure_directory_exists(parent_dir)

        # Create safe temp filename by sanitizing task name (remove invalid filename characters)
        task_name = asyncio.current_task().get_name()
        safe_task_name = re.sub(r'[^\w\-_]', '_', task_name)[:50]  # Limit length and sanitize
        temp_file = file_path.with_suffix(f'.tmp_{safe_task_name}') # Unique temp name
        try:
            # Asynchronously write JSON data to the temporary file
            async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                # Use DateTimeEncoder to handle datetime objects automatically
                # CRITICAL: DateTimeEncoder now forces all datetimes to LOCAL timezone
                json_data = json.dumps(data, cls=DateTimeEncoder, indent=2, ensure_ascii=False, sort_keys=True)
                await f.write(json_data)
                await f.flush() # Ensure data is written before moving

            # Atomically replace the target file with the temporary file (blocking operation)
            await self.hass.async_add_executor_job(
                shutil.move, str(temp_file), str(file_path)
            )
            # Atomic write successful (debug log removed)

        except Exception as e:
            _LOGGER.error("Atomic write failed for %s: %s", file_path.name, e)
            # Clean up the temporary file if it exists (blocking operation)
            if await self._file_exists(temp_file):
                # Attempting to clean up temporary file (debug log removed)
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
                try:
                    await self.hass.async_add_executor_job(temp_file.unlink)
                except Exception as e:
                    # Could not remove temp file (debug log removed)
                    pass


    async def _atomic_write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Public thread-safe method for atomically writing JSON data to a file"""
        try:
            # Get the lock specific to this file
            file_lock = await self._get_file_lock(file_path)

            # Try to acquire lock with timeout to prevent infinite blocking
            # Waiting for file lock (debug log removed)
            async with asyncio.timeout(15.0):  # Increased from 5s to 15s
                async with file_lock:
                    # File lock acquired, writing (debug log removed)
                    await self._atomic_write_json_unlocked(file_path, data)
                    # File lock released (debug log removed)
        except asyncio.TimeoutError:
            _LOGGER.error(
                f"Timeout acquiring file lock for {file_path.name} after 15 seconds. "
                f"Another operation may be blocking the lock. This should not happen with per-file locks!"
            )
            raise DataIntegrityException(
                f"Failed to acquire file lock for {file_path.name} - timeout after 15s",
                context=create_context(file=str(file_path))
            )

    async def _read_json_file(self, file_path: Path, default_structure: Dict | None = None) -> Dict[str, Any]:
        """Reads JSON data from a file asynchronously Non-blocking file read"""
        
        # +++ IMPORT HIER EINGEF +++
        try:
            import aiofiles
        except ImportError:
            _LOGGER.error("AIOFiles ist nicht installiert. Dateilesen fehlgeschlagen.")
            if default_structure is None: return {}
            return default_structure
        # +++ ENDE EINF +++

        if default_structure is None:
            default_structure = {} # Default to empty dict if not specified

        try:
            # Check existence using non-blocking helper
            if not await self._file_exists(file_path):
                # File not found, returning default (debug log removed)
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
                # Allow both dict and list (list for weather_cache backward compatibility)
                if not isinstance(data, (dict, list)):
                     _LOGGER.warning("JSON content in %s is neither dict nor list, returning default.", file_path.name)
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
        """Cleans up resources specifically shutting down the ThreadPoolExecutor"""
        try:
            _LOGGER.info("Shutting down DataManagerIO Executor...")
            # Use executor job to run the blocking shutdown method
            await self.hass.async_add_executor_job(self._executor.shutdown, True) # True waits for tasks
            _LOGGER.info("DataManagerIO Executor shut down successfully.")
        except Exception as e:
            _LOGGER.error("Error during DataManagerIO cleanup: %s", e, exc_info=True)
