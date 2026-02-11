# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Centralized database connection manager for SFML Stats. @zara"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

import aiosqlite

from ..const import SOLAR_FORECAST_DB

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


def get_manager() -> DatabaseConnectionManager | None:
    """Get the current database manager instance if available. @zara"""
    instance = DatabaseConnectionManager._instance
    _LOGGER.debug("get_manager() called, returning: %s (connected: %s)",
                  instance is not None,
                  instance.is_connected if instance else False)
    return instance


class DatabaseConnectionManager:
    """Singleton manager for SQLite database connections. @zara"""

    _instance: DatabaseConnectionManager | None = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, config_path: Path) -> None:
        """Initialize the database connection manager. @zara"""
        self._config_path = config_path
        self._db_path = config_path / SOLAR_FORECAST_DB
        self._connection: aiosqlite.Connection | None = None
        self._is_connected = False

    @classmethod
    async def get_instance(cls, hass: HomeAssistant) -> DatabaseConnectionManager:
        """Get or create the singleton instance. @zara"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    config_path = Path(hass.config.path())
                    cls._instance = cls(config_path)
                    await cls._instance.connect()
        return cls._instance

    @classmethod
    async def close_instance(cls) -> None:
        """Close and reset the singleton instance. @zara"""
        if cls._instance is not None:
            async with cls._lock:
                if cls._instance is not None:
                    await cls._instance.close()
                    cls._instance = None

    @property
    def db_path(self) -> Path:
        """Return the database file path. @zara"""
        return self._db_path

    @property
    def is_available(self) -> bool:
        """Check if database file exists. @zara"""
        return self._db_path.exists()

    @property
    def is_connected(self) -> bool:
        """Check if connection is active. @zara"""
        return self._is_connected and self._connection is not None

    async def connect(self) -> bool:
        """Establish database connection. @zara"""
        if self._is_connected and self._connection is not None:
            _LOGGER.debug("Database already connected")
            return True

        if not self.is_available:
            _LOGGER.warning("Database not found: %s", self._db_path)
            return False

        try:
            self._connection = await aiosqlite.connect(str(self._db_path))
            self._connection.row_factory = aiosqlite.Row
            self._is_connected = True
            _LOGGER.info("Database connection established: %s", self._db_path)
            return True
        except Exception as err:
            _LOGGER.error("Failed to connect to database: %s", err)
            self._connection = None
            self._is_connected = False
            return False

    async def close(self) -> None:
        """Close database connection. @zara"""
        if self._connection is not None:
            try:
                await self._connection.close()
                _LOGGER.info("Database connection closed")
            except Exception as err:
                _LOGGER.error("Error closing database connection: %s", err)
            finally:
                self._connection = None
                self._is_connected = False

    async def execute(self, query: str, params: tuple | list | None = None):
        """Execute a query and return cursor. @zara"""
        if not self.is_connected:
            raise RuntimeError("Database not connected")

        if params is None:
            params = []

        return self._connection.execute(query, params)

    async def get_connection(self) -> aiosqlite.Connection:
        """Get the active database connection. @zara"""
        if not self.is_connected:
            raise RuntimeError("Database not connected")
        return self._connection

    async def _ensure_connected(self) -> bool:
        """Verify connection is alive, reconnect if needed. @zara"""
        if self._connection is not None:
            try:
                await self._connection.execute("SELECT 1")
                return True
            except Exception:
                _LOGGER.warning("Database connection lost, attempting reconnect")
                self._connection = None
                self._is_connected = False

        return await self.connect()

    async def execute_read(self, query: str, params: tuple | list | None = None) -> list[aiosqlite.Row]:
        """Execute a read query with auto-reconnect. @zara"""
        if params is None:
            params = []

        for attempt in range(2):
            if not await self._ensure_connected():
                raise RuntimeError("Database not available")
            try:
                async with self._connection.execute(query, params) as cursor:
                    return await cursor.fetchall()
            except aiosqlite.OperationalError as err:
                if attempt == 0 and "database is locked" not in str(err):
                    _LOGGER.warning("Read query failed (attempt 1), retrying: %s", err)
                    self._connection = None
                    self._is_connected = False
                    continue
                raise

    async def execute_write(self, query: str, params: tuple | list | None = None) -> None:
        """Execute a write query with auto-commit and auto-reconnect. @zara"""
        if params is None:
            params = []

        for attempt in range(2):
            if not await self._ensure_connected():
                raise RuntimeError("Database not available")
            try:
                await self._connection.execute(query, params)
                await self._connection.commit()
                return
            except aiosqlite.OperationalError as err:
                if attempt == 0 and "database is locked" not in str(err):
                    _LOGGER.warning("Write query failed (attempt 1), retrying: %s", err)
                    self._connection = None
                    self._is_connected = False
                    continue
                raise

    @asynccontextmanager
    async def get_connection_ctx(self) -> AsyncIterator[aiosqlite.Connection]:
        """Context manager for multi-statement operations. @zara"""
        if not await self._ensure_connected():
            raise RuntimeError("Database not available")
        yield self._connection
