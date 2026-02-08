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

import aiosqlite

_LOGGER = logging.getLogger(__name__)


class GPMDatabaseConnector:
    """Lightweight SQLite connector for Grid Price Monitor @zara

    Uses the shared solar_forecast.db with GPM_ prefixed tables.
    """

    def __init__(self, db_path: str) -> None:
        """Initialize the database connector @zara

        Args:
            db_path: Absolute path to the SQLite database file
        """
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Establish database connection and ensure tables exist @zara"""
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row

        # Match SFML PRAGMA settings for shared DB compatibility
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.execute("PRAGMA journal_mode = DELETE")
        await self._db.execute("PRAGMA busy_timeout = 30000")

        await self._ensure_tables()
        _LOGGER.info(
            "GPM database connected: %s", self.db_path
        )

    async def close(self) -> None:
        """Close database connection @zara"""
        if self._db:
            await self._db.close()
            self._db = None
            _LOGGER.debug("GPM database connection closed")

    async def execute(
        self,
        sql: str,
        parameters: tuple = (),
        auto_commit: bool = True,
    ) -> None:
        """Execute a SQL statement @zara

        Args:
            sql: SQL statement
            parameters: Query parameters
            auto_commit: Whether to auto-commit after execution
        """
        await self._db.execute(sql, parameters)
        if auto_commit:
            await self._db.commit()

    async def fetchone(
        self,
        sql: str,
        parameters: tuple = (),
    ) -> aiosqlite.Row | None:
        """Execute SQL and fetch one row @zara

        Args:
            sql: SQL query
            parameters: Query parameters

        Returns:
            Single row or None
        """
        async with self._db.execute(sql, parameters) as cursor:
            return await cursor.fetchone()

    async def fetchall(
        self,
        sql: str,
        parameters: tuple = (),
    ) -> list[aiosqlite.Row]:
        """Execute SQL and fetch all rows @zara

        Args:
            sql: SQL query
            parameters: Query parameters

        Returns:
            List of rows
        """
        async with self._db.execute(sql, parameters) as cursor:
            return await cursor.fetchall()

    async def executemany(
        self,
        sql: str,
        parameters_list: list[tuple],
    ) -> int:
        """Execute SQL with multiple parameter sets @zara

        Args:
            sql: SQL statement
            parameters_list: List of parameter tuples

        Returns:
            Number of rows affected
        """
        await self._db.executemany(sql, parameters_list)
        await self._db.commit()
        return len(parameters_list)

    async def commit(self) -> None:
        """Commit current transaction @zara"""
        await self._db.commit()

    async def _ensure_tables(self) -> None:
        """Create all GPM tables if they don't exist @zara"""
        await self._db.executescript("""
            -- Price cache metadata (single row)
            CREATE TABLE IF NOT EXISTS GPM_price_cache_meta (
                id INTEGER PRIMARY KEY DEFAULT 1,
                last_fetch TEXT,
                valid_until TEXT,
                country TEXT,
                CHECK (id = 1)
            );

            -- Price cache entries
            CREATE TABLE IF NOT EXISTS GPM_price_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL UNIQUE,
                price REAL NOT NULL,
                total_price REAL,
                hour INTEGER NOT NULL
            );

            -- Price history (2 years retention)
            CREATE TABLE IF NOT EXISTS GPM_price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL UNIQUE,
                price_net REAL NOT NULL,
                total_price REAL,
                hour INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_gpm_price_history_ts
                ON GPM_price_history(timestamp);

            -- Daily average statistics
            CREATE TABLE IF NOT EXISTS GPM_daily_averages (
                date TEXT PRIMARY KEY,
                average_net REAL NOT NULL,
                average_total REAL NOT NULL,
                min_price REAL,
                max_price REAL
            );

            -- Monthly summary statistics
            CREATE TABLE IF NOT EXISTS GPM_monthly_summaries (
                month TEXT PRIMARY KEY,
                average_price REAL NOT NULL,
                cheap_hours INTEGER NOT NULL DEFAULT 0,
                country TEXT
            );

            -- All-time price extremes (single row)
            CREATE TABLE IF NOT EXISTS GPM_price_extremes (
                id INTEGER PRIMARY KEY DEFAULT 1,
                all_time_low REAL,
                all_time_low_date TEXT,
                all_time_high REAL,
                all_time_high_date TEXT,
                CHECK (id = 1)
            );

            -- Battery tracker statistics (single row, replaces HA Store)
            CREATE TABLE IF NOT EXISTS GPM_battery_stats (
                id INTEGER PRIMARY KEY DEFAULT 1,
                energy_today_wh REAL DEFAULT 0.0,
                energy_week_wh REAL DEFAULT 0.0,
                energy_month_wh REAL DEFAULT 0.0,
                current_day INTEGER,
                current_week INTEGER,
                current_month INTEGER,
                CHECK (id = 1)
            );

            -- Battery totals for statistics display (single row)
            CREATE TABLE IF NOT EXISTS GPM_battery_totals (
                id INTEGER PRIMARY KEY DEFAULT 1,
                today_kwh REAL DEFAULT 0.0,
                week_kwh REAL DEFAULT 0.0,
                month_kwh REAL DEFAULT 0.0,
                CHECK (id = 1)
            );

            -- Current price for external integrations (single row)
            CREATE TABLE IF NOT EXISTS GPM_current_price (
                id INTEGER PRIMARY KEY DEFAULT 1,
                timestamp TEXT NOT NULL,
                spot_price_net REAL,
                spot_price_gross REAL,
                total_price REAL,
                price_next_hour REAL,
                is_cheap INTEGER DEFAULT 0,
                average_today REAL,
                cheapest_today REAL,
                most_expensive_today REAL,
                country TEXT,
                last_updated TEXT NOT NULL,
                CHECK (id = 1)
            );

            -- Configuration backup (single row)
            CREATE TABLE IF NOT EXISTS GPM_config_backup (
                id INTEGER PRIMARY KEY DEFAULT 1,
                backup_time TEXT,
                config_json TEXT,
                CHECK (id = 1)
            );
        """)
        await self._db.commit()

        # Migrate existing tables: add total_price column if missing
        await self._migrate_tables()

        _LOGGER.debug("GPM database tables verified")

    async def _migrate_tables(self) -> None:
        """Run schema migrations for existing tables @zara"""
        try:
            # Check if total_price column exists in GPM_price_history
            async with self._db.execute(
                "PRAGMA table_info(GPM_price_history)"
            ) as cursor:
                columns = [row[1] for row in await cursor.fetchall()]

            if "total_price" not in columns:
                await self._db.execute(
                    "ALTER TABLE GPM_price_history ADD COLUMN total_price REAL"
                )
                await self._db.commit()
                _LOGGER.info("Migrated GPM_price_history: added total_price column")
        except Exception as err:
            _LOGGER.warning("Table migration check failed: %s", err)
