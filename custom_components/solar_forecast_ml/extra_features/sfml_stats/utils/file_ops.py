# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""File operations utilities for SFML Stats. @zara"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import aiofiles

from ..const import FILE_RETRY_COUNT, FILE_RETRY_DELAY_SECONDS

_LOGGER = logging.getLogger(__name__)


async def read_json_safe(
    path: Path,
    retries: int = FILE_RETRY_COUNT,
    delay: float = FILE_RETRY_DELAY_SECONDS,
) -> dict[str, Any] | None:
    """Read JSON file with retry logic. @zara"""
    for attempt in range(retries):
        try:
            if not path.exists():
                return None
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        except json.JSONDecodeError as err:
            _LOGGER.warning(
                "JSON decode error in %s (attempt %d/%d): %s",
                path, attempt + 1, retries, err
            )
            if attempt == retries - 1:
                _LOGGER.error("Failed to parse %s after %d attempts", path, retries)
                return None
        except IOError as err:
            _LOGGER.warning(
                "IO error reading %s (attempt %d/%d): %s",
                path, attempt + 1, retries, err
            )
            if attempt == retries - 1:
                _LOGGER.error("Failed to read %s after %d attempts", path, retries)
                return None
        except Exception as err:
            _LOGGER.error("Unexpected error reading %s: %s", path, err)
            return None

        await asyncio.sleep(delay * (attempt + 1))

    return None


async def write_json_safe(
    path: Path,
    data: dict[str, Any],
    retries: int = FILE_RETRY_COUNT,
    indent: int = 2,
) -> bool:
    """Write JSON file atomically with retry. @zara"""
    temp_path = path.with_suffix(".tmp")

    for attempt in range(retries):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=indent, ensure_ascii=False))

            temp_path.replace(path)

            return True

        except IOError as err:
            _LOGGER.warning(
                "IO error writing %s (attempt %d/%d): %s",
                path, attempt + 1, retries, err
            )
            if attempt == retries - 1:
                _LOGGER.error("Failed to write %s after %d attempts", path, retries)
                return False
        except Exception as err:
            _LOGGER.error("Unexpected error writing %s: %s", path, err)
            return False

        await asyncio.sleep(FILE_RETRY_DELAY_SECONDS * (attempt + 1))

    return False


async def append_to_file_safe(
    path: Path,
    content: str,
    retries: int = FILE_RETRY_COUNT,
) -> bool:
    """Append content to file with retry logic. @zara"""
    for attempt in range(retries):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(path, "a", encoding="utf-8") as f:
                await f.write(content)
            return True
        except IOError as err:
            _LOGGER.warning(
                "IO error appending to %s (attempt %d/%d): %s",
                path, attempt + 1, retries, err
            )
            if attempt == retries - 1:
                _LOGGER.error("Failed to append to %s after %d attempts", path, retries)
                return False
        except Exception as err:
            _LOGGER.error("Unexpected error appending to %s: %s", path, err)
            return False

        await asyncio.sleep(FILE_RETRY_DELAY_SECONDS * (attempt + 1))

    return False


def ensure_directory(path: Path) -> bool:
    """Ensure a directory exists. @zara"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as err:
        _LOGGER.error("Failed to create directory %s: %s", path, err)
        return False
