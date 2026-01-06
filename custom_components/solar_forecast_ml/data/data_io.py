# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict

from homeassistant.core import HomeAssistant

from ..const import DATA_VERSION

from ..core.core_exceptions import DataIntegrityException, create_context

_LOGGER = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON Encoder that converts datetime and date objects to ISO format str..."""

    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            if isinstance(obj, datetime):

                try:
                    from ..core.core_helpers import SafeDateTimeUtil as dt_util

                    obj = dt_util.ensure_local(obj)

                except Exception as e:
                    _LOGGER.warning(
                        f"Failed to convert datetime to local timezone: {e}, using original"
                    )
            return obj.isoformat()

        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, (bool, type(True), type(False))):
            return bool(obj)

        return super().default(obj)

class DataManagerIO:
    """Base class providing asynchronous thread-safe file IO operations"""

    def __init__(self, hass: HomeAssistant, data_dir: Path):
        """Initialize the IO manager @zara"""
        self.hass = hass
        self.data_dir = Path(data_dir)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="DataManagerIO")

        self._file_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()

    async def _get_file_lock(self, file_path: Path) -> asyncio.Lock:
        """Get or create a lock for a specific file @zara"""
        file_key = str(file_path)

        async with self._locks_lock:
            if file_key not in self._file_locks:
                self._file_locks[file_key] = asyncio.Lock()

            return self._file_locks[file_key]

    async def _ensure_directory_exists(self, directory: Path) -> None:
        """Ensure a directory exists creating it if necessary Non-blocking @zara"""
        try:

            exists = await self.hass.async_add_executor_job(directory.is_dir)
            if not exists:
                _LOGGER.info("Creating directory: %s", directory)
                await self.hass.async_add_executor_job(
                    lambda: directory.mkdir(parents=True, exist_ok=True)
                )
        except Exception as e:
            raise DataIntegrityException(
                f"Failed to ensure directory exists: {directory}",
                context=create_context(directory=str(directory), error=str(e)),
            )

    async def _get_file_size(self, file_path: Path) -> int:
        """Get the size of a file in bytes Non-blocking @zara"""
        try:
            if await self._file_exists(file_path):
                stat_result = await self.hass.async_add_executor_job(file_path.stat)
                return stat_result.st_size
            return 0
        except Exception as e:
            _LOGGER.warning("Could not get file size for %s: %s", file_path.name, e)
            return 0

    async def _file_exists(self, file_path: Path) -> bool:
        """Check if a file exists Non-blocking @zara"""
        try:
            return await self.hass.async_add_executor_job(file_path.exists)
        except Exception as e:
            _LOGGER.warning("Could not check file existence for %s: %s", file_path.name, e)
            return False

    async def _atomic_write_json_unlocked(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Internal function for atomic JSON writing Assumes the caller holds _file_lock @zara

        CRITICAL FIX (V12.0.1): Directory creation is now done INSIDE the synchronous
        executor function to prevent race conditions. Previously, the async check and
        sync mkdir could race with other tasks, causing FileNotFoundError on first startup.
        """

        # Pre-generate temp file path components
        task_name = asyncio.current_task().get_name()
        task_hash = hashlib.md5(task_name.encode()).hexdigest()[:8]
        timestamp = str(int(time.time() * 1000))[-6:]

        # All file operations are now done synchronously in the executor
        # to prevent race conditions between directory check and file creation
        def _ensure_dir_and_write():
            """Synchronous directory creation + write + move - all in one atomic block.

            CRITICAL: By doing mkdir inside the same executor call as the file write,
            we eliminate the race condition where another task could try to write
            between our async exists() check and the actual mkdir().
            """
            parent_dir = file_path.parent

            # ALWAYS ensure directory exists - mkdir with exist_ok=True is idempotent
            # This is safe to call even if directory exists
            parent_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique temp file name
            counter = 0
            while counter < 100:
                suffix = (
                    f".tmp_{task_hash}_{timestamp}"
                    if counter == 0
                    else f".tmp_{task_hash}_{timestamp}_{counter}"
                )
                temp_file = parent_dir / f"{file_path.stem}{suffix}"
                if not temp_file.exists():
                    break
                counter += 1
            else:
                import random
                suffix = f".tmp_{random.randint(100000, 999999)}"
                temp_file = parent_dir / f"{file_path.stem}{suffix}"

            try:
                # Write to temp file
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(
                        data, f, cls=DateTimeEncoder, indent=2, ensure_ascii=False, sort_keys=False
                    )
                    f.flush()
                    os.fsync(f.fileno())

                # Move temp file to final destination (atomic on POSIX)
                shutil.move(str(temp_file), str(file_path))

            except Exception as write_error:
                # Clean up temp file on error
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass
                raise write_error

            finally:
                # Final cleanup - remove temp file if it still exists
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass

        try:
            await self.hass.async_add_executor_job(_ensure_dir_and_write)

        except Exception as e:
            _LOGGER.error("Atomic write failed for %s: %s", file_path.name, e)
            raise DataIntegrityException(
                f"Failed atomic write to {file_path.name}: {str(e)}",
                context=create_context(file=str(file_path), error=str(e)),
            )

    async def _atomic_write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Public thread-safe method for atomically writing JSON data to a file @zara"""
        try:

            file_lock = await self._get_file_lock(file_path)

            async with asyncio.timeout(15.0):
                async with file_lock:

                    await self._atomic_write_json_unlocked(file_path, data)

        except asyncio.TimeoutError:
            _LOGGER.error(
                f"Timeout acquiring file lock for {file_path.name} after 15 seconds. "
                f"Another operation may be blocking the lock. This should not happen with per-file locks!"
            )
            raise DataIntegrityException(
                f"Failed to acquire file lock for {file_path.name} - timeout after 15s",
                context=create_context(file=str(file_path)),
            )

    async def _read_json_file(
        self, file_path: Path, default_structure: Dict | None = None
    ) -> Dict[str, Any]:
        """Reads JSON data from a file asynchronously Non-blocking file read"""

        if default_structure is None:
            default_structure = {}

        try:

            if not await self._file_exists(file_path):

                return default_structure

            def _read_file():
                """Synchronous file read in executor."""
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if not content:
                        return None
                    return content

            content = await self.hass.async_add_executor_job(_read_file)

            if content is None:
                _LOGGER.warning(
                    "File %s is empty, returning default structure.", file_path.name
                )
                return default_structure

            try:
                data = await self.hass.async_add_executor_job(json.loads, content)

                if not isinstance(data, (dict, list)):
                    _LOGGER.warning(
                        "JSON content in %s is neither dict nor list, returning default.",
                        file_path.name,
                    )
                    return default_structure
                return data
            except json.JSONDecodeError as e:
                _LOGGER.error(
                    "Invalid JSON content in %s: %s. Returning default structure.",
                    file_path.name,
                    e,
                )
                return default_structure

        except FileNotFoundError:

            _LOGGER.warning(
                "File %s not found during read attempt, returning default.", file_path.name
            )
            return default_structure
        except Exception as e:

            _LOGGER.error("Failed to read JSON file %s: %s", file_path.name, e, exc_info=True)

            return default_structure

    async def cleanup(self) -> None:
        """Cleans up resources specifically shutting down the ThreadPoolExecutor @zara"""
        try:
            _LOGGER.info("Shutting down DataManagerIO Executor...")

            await self.hass.async_add_executor_job(
                self._executor.shutdown, True
            )
            _LOGGER.info("DataManagerIO Executor shut down successfully.")
        except Exception as e:
            _LOGGER.error("Error during DataManagerIO cleanup: %s", e, exc_info=True)
