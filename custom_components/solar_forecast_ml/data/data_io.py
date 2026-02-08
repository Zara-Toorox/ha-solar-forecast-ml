# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Data IO Operations for Solar Forecast ML V16.2.0.
Provides base class for async database operations.
All file I/O replaced with database operations via DatabaseManager.

@zara
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from homeassistant.core import HomeAssistant

from .db_manager import DatabaseManager

_LOGGER = logging.getLogger(__name__)


class DataManagerIO:
    """Base class providing async database IO operations. @zara

    Replaces the old file-based IO with SQLite database operations.
    All subclasses should use self.db for database access.
    """

    def __init__(self, hass: HomeAssistant, db_manager: DatabaseManager):
        """Initialize the IO manager. @zara

        Args:
            hass: Home Assistant instance
            db_manager: DatabaseManager instance for all DB operations
        """
        self.hass = hass
        self.db = db_manager
        self._operation_lock = asyncio.Lock()
        self._initialized = False

        _LOGGER.debug("DataManagerIO initialized with DatabaseManager")

    async def ensure_initialized(self) -> bool:
        """Ensure the database connection is established. @zara"""
        if self._initialized:
            return True

        try:
            if self.db is None:
                _LOGGER.error("Database manager not initialized")
                return False
            if self.db._db is None:
                await self.db.connect()
            self._initialized = True
            return True
        except Exception as e:
            _LOGGER.error("Failed to initialize database connection: %s", e)
            return False

    async def execute_query(
        self,
        sql: str,
        parameters: tuple = (),
        timeout: float = 10.0
    ) -> bool:
        """Execute a database query with timeout. @zara

        Args:
            sql: SQL statement to execute
            parameters: Query parameters
            timeout: Timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            async with asyncio.timeout(timeout):
                async with self._operation_lock:
                    await self.db.execute(sql, parameters)
                    return True
        except asyncio.TimeoutError:
            _LOGGER.error("Database query timeout after %.1f seconds", timeout)
            return False
        except Exception as e:
            _LOGGER.error("Database query failed: %s", e)
            return False

    async def fetch_one(
        self,
        sql: str,
        parameters: tuple = ()
    ) -> Optional[Any]:
        """Fetch a single row from database. @zara

        Args:
            sql: SQL query
            parameters: Query parameters

        Returns:
            Row data or None if not found
        """
        try:
            return await self.db.fetchone(sql, parameters)
        except Exception as e:
            _LOGGER.error("Failed to fetch row: %s", e)
            return None

    async def fetch_all(
        self,
        sql: str,
        parameters: tuple = ()
    ) -> list:
        """Fetch all rows from database. @zara

        Args:
            sql: SQL query
            parameters: Query parameters

        Returns:
            List of rows or empty list on error
        """
        try:
            return await self.db.fetchall(sql, parameters)
        except Exception as e:
            _LOGGER.error("Failed to fetch rows: %s", e)
            return []

    async def get_db_stats(self) -> dict[str, Any]:
        """Get database statistics. @zara

        Returns:
            Dictionary with database stats
        """
        try:
            db_size = await self.db.get_db_size()
            return {
                "size_bytes": db_size,
                "size_mb": round(db_size / (1024 * 1024), 2),
                "connected": self.db._db is not None,
                "path": self.db.db_path,
            }
        except Exception as e:
            _LOGGER.error("Failed to get database stats: %s", e)
            return {
                "size_bytes": 0,
                "size_mb": 0,
                "connected": False,
                "error": str(e),
            }

    async def cleanup(self) -> None:
        """Cleanup resources and close database connection. @zara"""
        try:
            _LOGGER.info("Cleaning up DataManagerIO resources...")
            await self.db.close()
            self._initialized = False
            _LOGGER.info("DataManagerIO cleanup completed")
        except Exception as e:
            _LOGGER.error("Error during DataManagerIO cleanup: %s", e)
