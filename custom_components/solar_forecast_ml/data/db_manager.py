# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Database Manager for Solar Forecast ML.
Handles all SQLite operations using aiosqlite.
"""

import aiofiles
import aiosqlite
import json
import logging
import numpy as np
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database operations for Solar Forecast ML."""

    def __init__(self, db_path: str):
        """Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Establish database connection and create schema if needed."""
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row

        # Enable foreign keys
        await self._db.execute("PRAGMA foreign_keys = ON")

        # Use DELETE mode for SMB compatibility (no .wal/.shm files) @zara
        await self._db.execute("PRAGMA journal_mode = DELETE")

        # Set busy timeout to 30 seconds (wait for locks instead of failing immediately)
        await self._db.execute("PRAGMA busy_timeout = 30000")

        await self._initialize_schema()
        _LOGGER.info("Database connected with DELETE mode and 30s timeout: %s", self.db_path)

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            _LOGGER.info("Database closed")

    async def _initialize_schema(self) -> None:
        """Initialize database schema - only runs migrations if schema exists.

        Schema is created by StartupInitializer (sync) during initial setup.
        This method only runs migrations to avoid duplicate schema execution.
        """
        # Check if schema already exists (created by StartupInitializer)
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='astronomy_cache'"
        ) as cursor:
            result = await cursor.fetchone()

        if result:
            # Schema already exists, only run migrations
            _LOGGER.debug("Database schema already exists, running migrations only")
            await self._run_migrations()
            return

        # Fallback: Schema doesn't exist (shouldn't happen normally)
        _LOGGER.warning("Schema not found, initializing from scratch")
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            _LOGGER.error("Schema file not found: %s", schema_path)
            return

        async with aiofiles.open(schema_path, "r") as f:
            schema_sql = await f.read()

        await self._db.executescript(schema_sql)
        await self._db.commit()
        _LOGGER.info("Database schema initialized")

        await self._run_migrations()

    async def _run_migrations(self) -> None:
        """Run database migrations for schema updates."""
        await self._migrate_hourly_weather_actual_condition()
        await self._migrate_hourly_predictions_v16()
        await self._migrate_shadow_analysis_table_v16()
        await self._migrate_weather_precision_v161()
        await self._migrate_astronomy_panel_groups_v161()
        await self._cleanup_ai_model_weights_blobs_v161()
        # V16.1 additions for frost/snow/visibility @zara
        await self._migrate_frost_analysis_table_v161()
        await self._migrate_visibility_learning_table_v161()
        await self._migrate_snow_tracking_table_v161()
        await self._migrate_weather_forecast_weather_code_v161()
        # V16.0.0 statistics & billing tables @zara
        await self._migrate_stats_tables_v1600()
        # V16.2 shadow pattern learning tables @zara
        await self._migrate_shadow_pattern_learning_v162()
        # V16.0.0 sensor storage tables @zara
        await self._migrate_sensor_storage_tables_v1600()
        # V16.0.0 raw prediction values + method performance learning @zara
        await self._migrate_raw_prediction_values_v1600()
        await self._migrate_method_performance_learning_v1600()
        # V16.0.0 schema completeness fixes @zara
        await self._migrate_hourly_weather_actual_snow_frost_v1600()
        await self._migrate_stats_daily_energy_columns_v1600()
        await self._migrate_weather_cache_open_meteo_code_v1600()
        await self._migrate_missing_indexes_v1600()
        # V16.0.0 peak power tracking refactoring @zara
        await self._migrate_peak_power_tracking_v1600()
        # V16.0.0 AI pipeline fixes @zara
        await self._migrate_dni_history_dedup_v1600()
        await self._migrate_grid_search_model_type_v1600()
        await self._migrate_prediction_weather_diffuse_v1600()
        await self._migrate_prediction_weather_direct_radiation_v1601()
        # V16.0.2 Ridge ensemble columns @zara
        await self._migrate_ensemble_columns_v1602()
        # V16.0.3 Per-group adaptive ensemble weights @zara
        await self._migrate_ensemble_group_weights_v1603()
        # V16.0.4 Ridge feature normalization persistence @zara
        await self._migrate_ridge_meta_normalization_v1604()

    async def _migrate_hourly_weather_actual_condition(self) -> None:
        """Add condition column to hourly_weather_actual if missing."""
        async with self._db.execute(
            "PRAGMA table_info(hourly_weather_actual)"
        ) as cursor:
            columns = await cursor.fetchall()
            column_names = {col[1] for col in columns}

        if "condition" not in column_names:
            await self._db.execute(
                "ALTER TABLE hourly_weather_actual ADD COLUMN condition TEXT"
            )
            await self._db.commit()
            _LOGGER.info("Migration: Added 'condition' column to hourly_weather_actual")

    async def _migrate_hourly_predictions_v16(self) -> None:
        """Add V16 fields to hourly_predictions if missing. @zara"""
        async with self._db.execute(
            "PRAGMA table_info(hourly_predictions)"
        ) as cursor:
            columns = await cursor.fetchall()
            column_names = {col[1] for col in columns}

        migrations = []

        if "weather_alert_type" not in column_names:
            migrations.append("ALTER TABLE hourly_predictions ADD COLUMN weather_alert_type TEXT")

        if "exclude_from_learning" not in column_names:
            migrations.append("ALTER TABLE hourly_predictions ADD COLUMN exclude_from_learning BOOLEAN DEFAULT FALSE")

        if "mppt_throttled" not in column_names:
            migrations.append("ALTER TABLE hourly_predictions ADD COLUMN mppt_throttled BOOLEAN DEFAULT FALSE")

        if "mppt_throttle_reason" not in column_names:
            migrations.append("ALTER TABLE hourly_predictions ADD COLUMN mppt_throttle_reason TEXT")

        if "has_panel_group_actuals" not in column_names:
            migrations.append("ALTER TABLE hourly_predictions ADD COLUMN has_panel_group_actuals BOOLEAN DEFAULT FALSE")

        if "panel_group_predictions_backfilled" not in column_names:
            migrations.append("ALTER TABLE hourly_predictions ADD COLUMN panel_group_predictions_backfilled BOOLEAN DEFAULT FALSE")

        if "adaptive_corrected" not in column_names:
            migrations.append("ALTER TABLE hourly_predictions ADD COLUMN adaptive_corrected BOOLEAN DEFAULT FALSE")

        if "adaptive_correction_time" not in column_names:
            migrations.append("ALTER TABLE hourly_predictions ADD COLUMN adaptive_correction_time TIMESTAMP")

        if migrations:
            for migration in migrations:
                await self._db.execute(migration)
            await self._db.commit()
            _LOGGER.info("Migration: Added %d V16 fields to hourly_predictions", len(migrations))

    async def _migrate_shadow_analysis_table_v16(self) -> None:
        """Create daily_summary_shadow_analysis table if missing. @zara"""
        # Check if table exists
        async with self._db.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table' AND name='daily_summary_shadow_analysis'"""
        ) as cursor:
            result = await cursor.fetchone()

        if not result:
            # Create table
            await self._db.execute(
                """CREATE TABLE IF NOT EXISTS daily_summary_shadow_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL UNIQUE,
                    shadow_hours_count INTEGER DEFAULT 0,
                    cumulative_loss_kwh REAL DEFAULT 0.0,
                    FOREIGN KEY (date) REFERENCES daily_summaries(date) ON DELETE CASCADE
                )"""
            )
            await self._db.commit()
            _LOGGER.info("Migration: Created daily_summary_shadow_analysis table")

    async def _migrate_weather_precision_v161(self) -> None:
        """Create weather precision tables for hourly and weather-specific factors. @zara

        V16.1 migration: Adds fine-grained precision learning with hourly factors
        and weather-specific (clear/cloudy) factors.
        """
        # Check if weather_precision_hourly_factors exists
        async with self._db.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table' AND name='weather_precision_hourly_factors'"""
        ) as cursor:
            result = await cursor.fetchone()

        if not result:
            await self._db.execute(
                """CREATE TABLE IF NOT EXISTS weather_precision_hourly_factors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hour INTEGER NOT NULL CHECK(hour >= 0 AND hour <= 23),
                    factor_type TEXT NOT NULL CHECK(factor_type IN (
                        'solar_radiation_wm2', 'clouds', 'temperature',
                        'humidity', 'wind', 'rain', 'pressure'
                    )),
                    factor_value REAL NOT NULL DEFAULT 1.0,
                    sample_count INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.0,
                    std_dev REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(hour, factor_type)
                )"""
            )
            await self._db.commit()
            _LOGGER.info("Migration: Created weather_precision_hourly_factors table")

        # Check if weather_precision_weather_specific exists
        async with self._db.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table' AND name='weather_precision_weather_specific'"""
        ) as cursor:
            result = await cursor.fetchone()

        if not result:
            await self._db.execute(
                """CREATE TABLE IF NOT EXISTS weather_precision_weather_specific (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    weather_type TEXT NOT NULL CHECK(weather_type IN ('clear', 'cloudy', 'mixed')),
                    factor_type TEXT NOT NULL CHECK(factor_type IN (
                        'solar_radiation_wm2', 'clouds', 'temperature',
                        'humidity', 'wind', 'rain', 'pressure'
                    )),
                    factor_value REAL NOT NULL DEFAULT 1.0,
                    sample_days INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(weather_type, factor_type)
                )"""
            )
            await self._db.commit()
            _LOGGER.info("Migration: Created weather_precision_weather_specific table")

    async def _migrate_astronomy_panel_groups_v161(self) -> None:
        """Create astronomy_cache_panel_groups table for per-group POA radiation. @zara

        V16.1 migration: Adds per-panel-group POA (Plane-of-Array) radiation data
        for accurate tilted irradiance calculations instead of horizontal GHI only.
        """
        async with self._db.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table' AND name='astronomy_cache_panel_groups'"""
        ) as cursor:
            result = await cursor.fetchone()

        if not result:
            await self._db.execute(
                """CREATE TABLE IF NOT EXISTS astronomy_cache_panel_groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_date DATE NOT NULL,
                    hour INTEGER NOT NULL CHECK(hour >= 0 AND hour <= 23),
                    group_name TEXT NOT NULL,
                    power_kwp REAL NOT NULL,
                    azimuth_deg REAL NOT NULL,
                    tilt_deg REAL NOT NULL,
                    theoretical_kwh REAL,
                    poa_wm2 REAL,
                    aoi_deg REAL,
                    UNIQUE(cache_date, hour, group_name)
                )"""
            )
            # Create index for fast lookups by date
            await self._db.execute(
                """CREATE INDEX IF NOT EXISTS idx_astronomy_panel_groups_date
                   ON astronomy_cache_panel_groups(cache_date)"""
            )
            await self._db.commit()
            _LOGGER.info("Migration: Created astronomy_cache_panel_groups table with index")

    async def _cleanup_ai_model_weights_blobs_v161(self) -> None:
        """Remove legacy JSON blobs from ai_model_weights table. @zara

        V16.1 cleanup: The ai_model_weights table was storing full weight dicts as JSON,
        which caused inconsistencies with has_attention flag. Now we use only
        structured tables (ai_lstm_weights, ai_ridge_weights, etc.).
        """
        async with self._db.execute(
            "SELECT COUNT(*) FROM ai_model_weights"
        ) as cursor:
            result = await cursor.fetchone()

        if result and result[0] > 0:
            await self._db.execute("DELETE FROM ai_model_weights")
            await self._db.commit()
            _LOGGER.info(
                "Migration V16.1: Deleted %d legacy JSON blobs from ai_model_weights. "
                "Structured tables are now the single source of truth.",
                result[0]
            )

        # V16.1: Fix Ridge weights dimension mismatch @zara
        # Check if Ridge weights have extra columns beyond flat_size + 1 (bias column)
        meta_row = await self.fetchone(
            "SELECT flat_size FROM ai_ridge_meta WHERE id = 1"
        )
        if meta_row and meta_row[0]:
            expected_flat_size = meta_row[0]
            # V16.0.4 FIX: Keep col 0..flat_size (flat_size = bias col), delete col > flat_size @zara
            async with self._db.execute(
                "SELECT COUNT(*) FROM ai_ridge_weights WHERE col_index > ?",
                (expected_flat_size,)
            ) as cursor:
                extra_count = await cursor.fetchone()

            if extra_count and extra_count[0] > 0:
                await self._db.execute(
                    "DELETE FROM ai_ridge_weights WHERE col_index > ?",
                    (expected_flat_size,)
                )
                await self._db.commit()
                _LOGGER.info(
                    "Migration V16.1: Removed %d extra Ridge weight entries (col_index > %d). "
                    "This fixes the flat_size mismatch.",
                    extra_count[0], expected_flat_size
                )

    async def _migrate_frost_analysis_table_v161(self) -> None:
        """Create daily_summary_frost_analysis table if missing. @zara V16.1

        This table stores frost detection results for each day.
        """
        async with self._db.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table' AND name='daily_summary_frost_analysis'"""
        ) as cursor:
            result = await cursor.fetchone()

        if not result:
            await self._db.execute(
                """CREATE TABLE IF NOT EXISTS daily_summary_frost_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL UNIQUE,
                    hours_analyzed INTEGER,
                    frost_detected BOOLEAN,
                    total_affected_hours INTEGER,
                    heavy_frost_hours INTEGER,
                    light_frost_hours INTEGER,
                    FOREIGN KEY (date) REFERENCES daily_summaries(date) ON DELETE CASCADE
                )"""
            )
            await self._db.commit()
            _LOGGER.info("Migration V16.1: Created daily_summary_frost_analysis table")

    async def _migrate_visibility_learning_table_v161(self) -> None:
        """Create/update visibility_learning table. @zara V16.1

        This table stores fog detection learning data.
        V16.1 adds: visibility_threshold_m, fog_visibility_threshold_m,
                    samples_below_threshold, samples_above_threshold
        """
        async with self._db.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table' AND name='visibility_learning'"""
        ) as cursor:
            result = await cursor.fetchone()

        if not result:
            await self._db.execute(
                """CREATE TABLE IF NOT EXISTS visibility_learning (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version TEXT DEFAULT '1.0',
                    has_solar_radiation_sensor BOOLEAN DEFAULT FALSE,
                    last_learning_date DATE,
                    total_fog_hours_learned INTEGER DEFAULT 0,
                    total_fog_light_hours_learned INTEGER DEFAULT 0,
                    bright_sky_fog_hits INTEGER DEFAULT 0,
                    pirate_weather_fog_hits INTEGER DEFAULT 0,
                    learning_sessions INTEGER DEFAULT 0,
                    fog_bright_sky_weight REAL DEFAULT 0.5,
                    fog_pirate_weather_weight REAL DEFAULT 0.5,
                    fog_light_bright_sky_weight REAL DEFAULT 0.5,
                    fog_light_pirate_weather_weight REAL DEFAULT 0.5,
                    visibility_threshold_m REAL DEFAULT 5000,
                    fog_visibility_threshold_m REAL DEFAULT 1000,
                    samples_below_threshold INTEGER DEFAULT 0,
                    samples_above_threshold INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            await self._db.commit()
            _LOGGER.info("Migration V16.1: Created visibility_learning table")
        else:
            # Table exists, add V16.1 columns if missing @zara
            async with self._db.execute("PRAGMA table_info(visibility_learning)") as cursor:
                columns = await cursor.fetchall()
                column_names = {col[1] for col in columns}

            migrations = []
            if "visibility_threshold_m" not in column_names:
                migrations.append(
                    "ALTER TABLE visibility_learning ADD COLUMN visibility_threshold_m REAL DEFAULT 5000"
                )
            if "fog_visibility_threshold_m" not in column_names:
                migrations.append(
                    "ALTER TABLE visibility_learning ADD COLUMN fog_visibility_threshold_m REAL DEFAULT 1000"
                )
            if "samples_below_threshold" not in column_names:
                migrations.append(
                    "ALTER TABLE visibility_learning ADD COLUMN samples_below_threshold INTEGER DEFAULT 0"
                )
            if "samples_above_threshold" not in column_names:
                migrations.append(
                    "ALTER TABLE visibility_learning ADD COLUMN samples_above_threshold INTEGER DEFAULT 0"
                )

            if migrations:
                for migration in migrations:
                    await self._db.execute(migration)
                await self._db.commit()
                _LOGGER.info("Migration V16.1: Added %d columns to visibility_learning", len(migrations))

    async def _migrate_snow_tracking_table_v161(self) -> None:
        """Create/update snow_tracking table. @zara V16.1

        This table stores snow coverage tracking data.
        V16.1 adds: melt_hours, detection_source, cleared_at
        """
        async with self._db.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table' AND name='snow_tracking'"""
        ) as cursor:
            result = await cursor.fetchone()

        if not result:
            # Create table with full V16.1 schema @zara
            await self._db.execute(
                """CREATE TABLE IF NOT EXISTS snow_tracking (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_snow_event TIMESTAMP,
                    panels_covered_since TIMESTAMP,
                    estimated_depth_mm REAL DEFAULT 0,
                    melt_started_at TIMESTAMP,
                    melt_hours REAL DEFAULT 0,
                    detection_source TEXT,
                    cleared_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            await self._db.commit()
            _LOGGER.info("Migration V16.1: Created snow_tracking table")
        else:
            # Table exists, add V16.1 columns if missing @zara
            async with self._db.execute("PRAGMA table_info(snow_tracking)") as cursor:
                columns = await cursor.fetchall()
                column_names = {col[1] for col in columns}

            migrations = []
            if "melt_hours" not in column_names:
                migrations.append("ALTER TABLE snow_tracking ADD COLUMN melt_hours REAL DEFAULT 0")
            if "detection_source" not in column_names:
                migrations.append("ALTER TABLE snow_tracking ADD COLUMN detection_source TEXT")
            if "cleared_at" not in column_names:
                migrations.append("ALTER TABLE snow_tracking ADD COLUMN cleared_at TIMESTAMP")

            if migrations:
                for migration in migrations:
                    await self._db.execute(migration)
                await self._db.commit()
                _LOGGER.info("Migration V16.1: Added %d columns to snow_tracking", len(migrations))

    async def _migrate_weather_forecast_weather_code_v161(self) -> None:
        """Add weather_code column to weather_forecast table. @zara V16.1

        Used for snow detection via Open-Meteo weather codes.
        """
        async with self._db.execute("PRAGMA table_info(weather_forecast)") as cursor:
            columns = await cursor.fetchall()
            column_names = {col[1] for col in columns}

        if "weather_code" not in column_names:
            await self._db.execute(
                "ALTER TABLE weather_forecast ADD COLUMN weather_code INTEGER"
            )
            await self._db.commit()
            _LOGGER.info("Migration V16.1: Added weather_code column to weather_forecast")

    async def _migrate_stats_tables_v1600(self) -> None:
        """Create statistics and billing tables if missing. @zara V16.0.0

        Creates tables for energy billing, tariff tracking, and forecast comparison.
        """
        tables_created = []

        # Check and create stats_settings
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stats_settings'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS stats_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                tables_created.append("stats_settings")

        # Check and create stats_monthly_tariffs
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stats_monthly_tariffs'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS stats_monthly_tariffs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        year_month TEXT NOT NULL UNIQUE,
                        year INTEGER NOT NULL,
                        month INTEGER NOT NULL,
                        price_ct_kwh REAL NOT NULL,
                        feed_in_tariff_ct REAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                tables_created.append("stats_monthly_tariffs")

        # Check and create stats_price_history
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stats_price_history'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS stats_price_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        datetime TEXT NOT NULL UNIQUE,
                        date TEXT NOT NULL,
                        hour INTEGER NOT NULL CHECK(hour >= 0 AND hour <= 23),
                        price_ct_kwh REAL NOT NULL,
                        price_source TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_stats_price_history_date ON stats_price_history(date)"
                )
                tables_created.append("stats_price_history")

        # Check and create stats_power_sources
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stats_power_sources'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS stats_power_sources (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL UNIQUE,
                        date TEXT NOT NULL,
                        hour INTEGER NOT NULL CHECK(hour >= 0 AND hour <= 23),
                        solar_power_w REAL DEFAULT 0,
                        grid_power_w REAL DEFAULT 0,
                        battery_power_w REAL DEFAULT 0,
                        house_consumption_w REAL DEFAULT 0,
                        solar_to_house_w REAL DEFAULT 0,
                        solar_to_battery_w REAL DEFAULT 0,
                        solar_to_grid_w REAL DEFAULT 0,
                        battery_to_house_w REAL DEFAULT 0,
                        grid_to_house_w REAL DEFAULT 0,
                        grid_to_battery_w REAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_stats_power_sources_date ON stats_power_sources(date)"
                )
                tables_created.append("stats_power_sources")

        # Check and create stats_hourly_billing
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stats_hourly_billing'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS stats_hourly_billing (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        hour_key TEXT NOT NULL UNIQUE,
                        date TEXT NOT NULL,
                        hour INTEGER NOT NULL CHECK(hour >= 0 AND hour <= 23),
                        grid_import_kwh REAL DEFAULT 0,
                        grid_import_cost_ct REAL DEFAULT 0,
                        grid_export_kwh REAL DEFAULT 0,
                        feed_in_revenue_ct REAL DEFAULT 0,
                        feed_in_tariff_ct REAL DEFAULT 0,
                        price_ct_kwh REAL DEFAULT 0,
                        grid_to_house_kwh REAL DEFAULT 0,
                        grid_to_house_cost_ct REAL DEFAULT 0,
                        grid_to_battery_kwh REAL DEFAULT 0,
                        grid_to_battery_cost_ct REAL DEFAULT 0,
                        solar_yield_kwh REAL DEFAULT 0,
                        solar_to_house_kwh REAL DEFAULT 0,
                        solar_to_battery_kwh REAL DEFAULT 0,
                        battery_to_house_kwh REAL DEFAULT 0,
                        home_consumption_kwh REAL DEFAULT 0,
                        data_source TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_stats_hourly_billing_date ON stats_hourly_billing(date)"
                )
                tables_created.append("stats_hourly_billing")

        # Check and create stats_daily_energy
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stats_daily_energy'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS stats_daily_energy (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL UNIQUE,
                        solar_yield_kwh REAL DEFAULT 0,
                        grid_import_kwh REAL DEFAULT 0,
                        grid_export_kwh REAL DEFAULT 0,
                        battery_charge_solar_kwh REAL DEFAULT 0,
                        battery_charge_grid_kwh REAL DEFAULT 0,
                        battery_to_house_kwh REAL DEFAULT 0,
                        solar_to_house_kwh REAL DEFAULT 0,
                        solar_to_battery_kwh REAL DEFAULT 0,
                        grid_to_house_kwh REAL DEFAULT 0,
                        home_consumption_kwh REAL DEFAULT 0,
                        self_consumption_kwh REAL DEFAULT 0,
                        autarkie_percent REAL DEFAULT 0,
                        avg_price_ct REAL DEFAULT 0,
                        total_cost_eur REAL DEFAULT 0,
                        feed_in_revenue_eur REAL DEFAULT 0,
                        savings_eur REAL DEFAULT 0,
                        peak_solar_w REAL DEFAULT 0,
                        peak_solar_time TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_stats_daily_energy_date ON stats_daily_energy(date)"
                )
                tables_created.append("stats_daily_energy")

        # Check and create stats_billing_totals
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stats_billing_totals'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS stats_billing_totals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        billing_start_date TEXT NOT NULL UNIQUE,
                        billing_start_day INTEGER DEFAULT 1,
                        billing_start_month INTEGER DEFAULT 1,
                        grid_import_kwh REAL DEFAULT 0,
                        grid_import_cost_eur REAL DEFAULT 0,
                        grid_export_kwh REAL DEFAULT 0,
                        feed_in_revenue_eur REAL DEFAULT 0,
                        solar_yield_kwh REAL DEFAULT 0,
                        solar_to_house_kwh REAL DEFAULT 0,
                        solar_to_battery_kwh REAL DEFAULT 0,
                        battery_to_house_kwh REAL DEFAULT 0,
                        grid_to_house_kwh REAL DEFAULT 0,
                        grid_to_house_cost_eur REAL DEFAULT 0,
                        grid_to_battery_kwh REAL DEFAULT 0,
                        grid_to_battery_cost_eur REAL DEFAULT 0,
                        home_consumption_kwh REAL DEFAULT 0,
                        self_consumption_kwh REAL DEFAULT 0,
                        autarkie_percent REAL DEFAULT 0,
                        savings_eur REAL DEFAULT 0,
                        net_benefit_eur REAL DEFAULT 0,
                        hours_count INTEGER DEFAULT 0,
                        avg_price_ct REAL DEFAULT 0,
                        avg_feed_in_tariff_ct REAL DEFAULT 0,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                tables_created.append("stats_billing_totals")

        # Check and create stats_forecast_comparison
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stats_forecast_comparison'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS stats_forecast_comparison (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL UNIQUE,
                        actual_kwh REAL,
                        sfml_forecast_kwh REAL,
                        sfml_accuracy_percent REAL,
                        external_1_kwh REAL,
                        external_1_accuracy_percent REAL,
                        external_2_kwh REAL,
                        external_2_accuracy_percent REAL,
                        best_source TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_stats_forecast_comparison_date ON stats_forecast_comparison(date)"
                )
                tables_created.append("stats_forecast_comparison")

        if tables_created:
            await self._db.commit()
            _LOGGER.info("Migration V16.0.0: Created stats tables: %s", ", ".join(tables_created))

    async def _migrate_shadow_pattern_learning_v162(self) -> None:
        """V16.2: Create shadow pattern learning tables. @zara"""
        tables_created = []

        # shadow_pattern_hourly
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='shadow_pattern_hourly'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS shadow_pattern_hourly (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        hour INTEGER NOT NULL CHECK(hour >= 0 AND hour <= 23),
                        shadow_occurrence_rate REAL DEFAULT 0.0,
                        avg_shadow_percent REAL DEFAULT 0.0,
                        std_dev_shadow_percent REAL DEFAULT 0.0,
                        pct_weather_clouds REAL DEFAULT 0.0,
                        pct_building_tree REAL DEFAULT 0.0,
                        pct_low_sun REAL DEFAULT 0.0,
                        pct_other REAL DEFAULT 0.0,
                        pattern_type TEXT DEFAULT 'unknown' CHECK(pattern_type IN (
                            'no_shadow', 'occasional', 'frequent', 'fixed_obstruction', 'unknown'
                        )),
                        confidence REAL DEFAULT 0.0,
                        sample_count INTEGER DEFAULT 0,
                        shadow_days INTEGER DEFAULT 0,
                        clear_days INTEGER DEFAULT 0,
                        first_learned DATE,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(hour)
                    )
                """)
                tables_created.append("shadow_pattern_hourly")

        # shadow_pattern_seasonal
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='shadow_pattern_seasonal'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS shadow_pattern_seasonal (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        month INTEGER NOT NULL CHECK(month >= 1 AND month <= 12),
                        hour INTEGER NOT NULL CHECK(hour >= 0 AND hour <= 23),
                        shadow_occurrence_rate REAL DEFAULT 0.0,
                        avg_shadow_percent REAL DEFAULT 0.0,
                        std_dev_shadow_percent REAL DEFAULT 0.0,
                        dominant_cause TEXT DEFAULT 'unknown',
                        sample_count INTEGER DEFAULT 0,
                        shadow_days INTEGER DEFAULT 0,
                        confidence REAL DEFAULT 0.0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(month, hour)
                    )
                """)
                tables_created.append("shadow_pattern_seasonal")

        # shadow_learning_history
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='shadow_learning_history'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS shadow_learning_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL,
                        hour INTEGER NOT NULL CHECK(hour >= 0 AND hour <= 23),
                        shadow_detected BOOLEAN NOT NULL,
                        shadow_type TEXT,
                        shadow_percent REAL,
                        root_cause TEXT,
                        confidence REAL,
                        sun_elevation_deg REAL,
                        cloud_cover_percent REAL,
                        theoretical_max_kwh REAL,
                        actual_kwh REAL,
                        efficiency_ratio REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date, hour)
                    )
                """)
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_shadow_learning_history_date ON shadow_learning_history(date)"
                )
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_shadow_learning_history_hour ON shadow_learning_history(hour)"
                )
                tables_created.append("shadow_learning_history")

        # shadow_pattern_config
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='shadow_pattern_config'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS shadow_pattern_config (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        rolling_window_days INTEGER DEFAULT 30,
                        min_samples_for_pattern INTEGER DEFAULT 7,
                        ema_alpha REAL DEFAULT 0.15,
                        fixed_obstruction_threshold REAL DEFAULT 0.7,
                        total_days_learned INTEGER DEFAULT 0,
                        total_hours_learned INTEGER DEFAULT 0,
                        last_learning_date DATE,
                        patterns_detected INTEGER DEFAULT 0,
                        fixed_obstructions_detected INTEGER DEFAULT 0,
                        version TEXT DEFAULT '1.0',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                tables_created.append("shadow_pattern_config")

        if tables_created:
            await self._db.commit()
            _LOGGER.info("Migration V16.2: Created shadow pattern learning tables: %s", ", ".join(tables_created))

    async def _migrate_sensor_storage_tables_v1600(self) -> None:
        """V16.0.0: Create sensor storage tables for live data. @zara

        Creates tables for:
        - sensor_power_live: Rolling 2-day power (Watt) data
        - sensor_energy_hourly: Permanent hourly energy (kWh) data
        - sensor_weather: Permanent 5-min weather sensor readings
        """
        tables_created = []

        # sensor_power_live
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sensor_power_live'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS sensor_power_live (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        power_watt REAL,
                        solar_to_battery_watt REAL
                    )
                """)
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sensor_power_live_timestamp ON sensor_power_live(timestamp)"
                )
                tables_created.append("sensor_power_live")

        # sensor_energy_hourly
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sensor_energy_hourly'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS sensor_energy_hourly (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        date DATE NOT NULL,
                        hour INTEGER NOT NULL CHECK(hour >= 0 AND hour <= 23),
                        yield_total_kwh REAL,
                        yield_gruppe1_kwh REAL,
                        yield_gruppe2_kwh REAL,
                        consumption_kwh REAL,
                        UNIQUE(date, hour)
                    )
                """)
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sensor_energy_hourly_date ON sensor_energy_hourly(date)"
                )
                tables_created.append("sensor_energy_hourly")

        # sensor_weather
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sensor_weather'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS sensor_weather (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        temperature REAL,
                        humidity REAL,
                        wind_speed REAL,
                        rain REAL,
                        lux REAL,
                        pressure REAL,
                        solar_radiation REAL
                    )
                """)
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sensor_weather_timestamp ON sensor_weather(timestamp)"
                )
                tables_created.append("sensor_weather")

        # sensor_monthly_stats
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sensor_monthly_stats'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS sensor_monthly_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        year INTEGER NOT NULL,
                        month INTEGER NOT NULL CHECK(month >= 1 AND month <= 12),
                        yield_total_kwh REAL,
                        consumption_total_kwh REAL,
                        avg_autarky_percent REAL,
                        avg_accuracy_percent REAL,
                        peak_power_w REAL,
                        peak_power_date DATE,
                        production_days INTEGER,
                        best_day_kwh REAL,
                        best_day_date DATE,
                        worst_day_kwh REAL,
                        worst_day_date DATE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(year, month)
                    )
                """)
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sensor_monthly_stats_year_month ON sensor_monthly_stats(year, month)"
                )
                tables_created.append("sensor_monthly_stats")

        if tables_created:
            await self._db.commit()
            _LOGGER.info("Migration V16.0.0: Created sensor storage tables: %s", ", ".join(tables_created))

    async def _migrate_raw_prediction_values_v1600(self) -> None:
        """V16.0.0: Add raw physics/AI prediction values to hourly_predictions and panel groups. @zara"""
        # hourly_predictions: physics_kwh, ai_kwh, ai_confidence
        async with self._db.execute(
            "PRAGMA table_info(hourly_predictions)"
        ) as cursor:
            columns = await cursor.fetchall()
            hp_columns = {col[1] for col in columns}

        hp_migrations = []
        if "physics_kwh" not in hp_columns:
            hp_migrations.append("ALTER TABLE hourly_predictions ADD COLUMN physics_kwh REAL")
        if "ai_kwh" not in hp_columns:
            hp_migrations.append("ALTER TABLE hourly_predictions ADD COLUMN ai_kwh REAL")
        if "ai_confidence" not in hp_columns:
            hp_migrations.append("ALTER TABLE hourly_predictions ADD COLUMN ai_confidence REAL")

        if hp_migrations:
            for migration in hp_migrations:
                await self._db.execute(migration)
            _LOGGER.info("Migration V16.0.0: Added %d raw value columns to hourly_predictions", len(hp_migrations))

        # prediction_panel_groups: physics_kwh, ai_kwh
        async with self._db.execute(
            "PRAGMA table_info(prediction_panel_groups)"
        ) as cursor:
            columns = await cursor.fetchall()
            pg_columns = {col[1] for col in columns}

        pg_migrations = []
        if "physics_kwh" not in pg_columns:
            pg_migrations.append("ALTER TABLE prediction_panel_groups ADD COLUMN physics_kwh REAL")
        if "ai_kwh" not in pg_columns:
            pg_migrations.append("ALTER TABLE prediction_panel_groups ADD COLUMN ai_kwh REAL")

        if pg_migrations:
            for migration in pg_migrations:
                await self._db.execute(migration)
            _LOGGER.info("Migration V16.0.0: Added %d raw value columns to prediction_panel_groups", len(pg_migrations))

        if hp_migrations or pg_migrations:
            await self._db.commit()

    async def _migrate_method_performance_learning_v1600(self) -> None:
        """V16.0.0: Create method_performance_learning table for blending weight learning. @zara"""
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='method_performance_learning'"
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS method_performance_learning (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cloud_bucket TEXT NOT NULL CHECK(cloud_bucket IN ('clear', 'partly_cloudy', 'overcast')),
                        hour_bucket TEXT NOT NULL CHECK(hour_bucket IN ('morning', 'midday', 'afternoon')),
                        physics_mae REAL DEFAULT 0.0,
                        ai_mae REAL DEFAULT 0.0,
                        blend_mae REAL DEFAULT 0.0,
                        ai_advantage_factor REAL DEFAULT 1.0,
                        sample_count INTEGER DEFAULT 0,
                        last_updated TIMESTAMP,
                        UNIQUE(cloud_bucket, hour_bucket)
                    )
                """)
                await self._db.commit()
                _LOGGER.info("Migration V16.0.0: Created method_performance_learning table")

    async def _migrate_hourly_weather_actual_snow_frost_v1600(self) -> None:
        """V16.0.0: Add frost/snow tracking columns to hourly_weather_actual. @zara

        These columns are used by data_weather_actual_tracker for frost notifications
        and snow event detection/clearing progress.
        """
        async with self._db.execute("PRAGMA table_info(hourly_weather_actual)") as cursor:
            columns = await cursor.fetchall()
            column_names = {col[1] for col in columns}

        migrations = []
        if "frost_notification_sent" not in column_names:
            migrations.append("ALTER TABLE hourly_weather_actual ADD COLUMN frost_notification_sent BOOLEAN DEFAULT 0")
        if "frost_type" not in column_names:
            migrations.append("ALTER TABLE hourly_weather_actual ADD COLUMN frost_type TEXT")
        if "snow_confidence" not in column_names:
            migrations.append("ALTER TABLE hourly_weather_actual ADD COLUMN snow_confidence REAL")
        if "snow_event_detected" not in column_names:
            migrations.append("ALTER TABLE hourly_weather_actual ADD COLUMN snow_event_detected BOOLEAN DEFAULT 0")
        if "snow_clearing_progress" not in column_names:
            migrations.append("ALTER TABLE hourly_weather_actual ADD COLUMN snow_clearing_progress REAL")

        if migrations:
            for migration in migrations:
                await self._db.execute(migration)
            await self._db.commit()
            _LOGGER.info("Migration V16.0.0: Added %d frost/snow columns to hourly_weather_actual", len(migrations))

    async def _migrate_stats_daily_energy_columns_v1600(self) -> None:
        """V16.0.0: Add missing energy tracking columns to stats_daily_energy. @zara

        Adds grid_to_battery, smartmeter, and consumer device columns.
        """
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stats_daily_energy'"
        ) as cursor:
            if not await cursor.fetchone():
                return

        async with self._db.execute("PRAGMA table_info(stats_daily_energy)") as cursor:
            columns = await cursor.fetchall()
            column_names = {col[1] for col in columns}

        migrations = []
        if "grid_to_battery_kwh" not in column_names:
            migrations.append("ALTER TABLE stats_daily_energy ADD COLUMN grid_to_battery_kwh REAL DEFAULT 0")
        if "smartmeter_import_kwh" not in column_names:
            migrations.append("ALTER TABLE stats_daily_energy ADD COLUMN smartmeter_import_kwh REAL DEFAULT 0")
        if "smartmeter_export_kwh" not in column_names:
            migrations.append("ALTER TABLE stats_daily_energy ADD COLUMN smartmeter_export_kwh REAL DEFAULT 0")
        if "consumer_heatpump_kwh" not in column_names:
            migrations.append("ALTER TABLE stats_daily_energy ADD COLUMN consumer_heatpump_kwh REAL DEFAULT 0")
        if "consumer_heatingrod_kwh" not in column_names:
            migrations.append("ALTER TABLE stats_daily_energy ADD COLUMN consumer_heatingrod_kwh REAL DEFAULT 0")
        if "consumer_wallbox_kwh" not in column_names:
            migrations.append("ALTER TABLE stats_daily_energy ADD COLUMN consumer_wallbox_kwh REAL DEFAULT 0")

        if migrations:
            for migration in migrations:
                await self._db.execute(migration)
            await self._db.commit()
            _LOGGER.info("Migration V16.0.0: Added %d columns to stats_daily_energy", len(migrations))

    async def _migrate_weather_cache_open_meteo_code_v1600(self) -> None:
        """V16.0.0: Add weather_code column to weather_cache_open_meteo. @zara"""
        async with self._db.execute("PRAGMA table_info(weather_cache_open_meteo)") as cursor:
            columns = await cursor.fetchall()
            column_names = {col[1] for col in columns}

        if "weather_code" not in column_names:
            await self._db.execute(
                "ALTER TABLE weather_cache_open_meteo ADD COLUMN weather_code INTEGER"
            )
            await self._db.commit()
            _LOGGER.info("Migration V16.0.0: Added weather_code column to weather_cache_open_meteo")

    async def _migrate_missing_indexes_v1600(self) -> None:
        """V16.0.0: Create missing indexes and clean up duplicates. @zara"""
        # Drop old index with wrong name (V16.1 migration used idx_astronomy_panel_groups_date
        # instead of idx_astronomy_cache_panel_groups_date)
        try:
            await self._db.execute("DROP INDEX IF EXISTS idx_astronomy_panel_groups_date")
        except Exception:
            pass

        indexes = [
            # Stats indexes
            "CREATE INDEX IF NOT EXISTS idx_stats_price_history_datetime ON stats_price_history(datetime)",
            "CREATE INDEX IF NOT EXISTS idx_stats_hourly_billing_hour_key ON stats_hourly_billing(hour_key)",
            "CREATE INDEX IF NOT EXISTS idx_stats_monthly_tariffs_year_month ON stats_monthly_tariffs(year_month)",
            "CREATE INDEX IF NOT EXISTS idx_stats_power_sources_timestamp ON stats_power_sources(timestamp)",
            # Astronomy panel groups indexes (V16.1 migration used wrong name + missed second index)
            "CREATE INDEX IF NOT EXISTS idx_astronomy_cache_panel_groups_date ON astronomy_cache_panel_groups(cache_date)",
            "CREATE INDEX IF NOT EXISTS idx_astronomy_cache_panel_groups_date_hour ON astronomy_cache_panel_groups(cache_date, hour)",
            # Weather precision hourly index (missing in V16.1 migration)
            "CREATE INDEX IF NOT EXISTS idx_weather_precision_hourly_hour ON weather_precision_hourly_factors(hour)",
        ]
        for idx_sql in indexes:
            try:
                await self._db.execute(idx_sql)
            except Exception:
                pass
        await self._db.commit()

    async def _migrate_peak_power_tracking_v1600(self) -> None:
        """V16.0.0: Add today's peak + all-time peak record columns to production_time_state
        and peak_power_time to daily_summaries for AI learning. @zara"""
        migrations_applied = []

        # Add peak tracking columns to production_time_state @zara
        async with self._db.execute("PRAGMA table_info(production_time_state)") as cursor:
            columns = await cursor.fetchall()
            column_names = {col[1] for col in columns}

        for col, col_type in [
            ("peak_power_w", "REAL DEFAULT 0"),
            ("peak_power_time", "TEXT"),
            ("peak_record_w", "REAL"),
            ("peak_record_date", "TEXT"),
            ("peak_record_time", "TEXT"),
        ]:
            if col not in column_names:
                await self._db.execute(
                    f"ALTER TABLE production_time_state ADD COLUMN {col} {col_type}"
                )
                migrations_applied.append(f"production_time_state.{col}")

        # Add peak_power_time to daily_summaries @zara
        async with self._db.execute("PRAGMA table_info(daily_summaries)") as cursor:
            columns = await cursor.fetchall()
            column_names = {col[1] for col in columns}

        if "peak_power_time" not in column_names:
            await self._db.execute(
                "ALTER TABLE daily_summaries ADD COLUMN peak_power_time TEXT"
            )
            migrations_applied.append("daily_summaries.peak_power_time")

        # Seed all-time record from astronomy_hourly_peaks @zara
        if migrations_applied:
            try:
                row = await self._db.fetchone(
                    """SELECT kwh * 1000 as peak_w, date, hour
                       FROM astronomy_hourly_peaks
                       WHERE kwh IS NOT NULL AND kwh > 0
                       ORDER BY kwh DESC LIMIT 1"""
                )
                if row and row[0]:
                    await self._db.execute(
                        """UPDATE production_time_state
                           SET peak_record_w = ?, peak_record_date = ?, peak_record_time = ?
                           WHERE id = 1 AND (peak_record_w IS NULL OR peak_record_w < ?)""",
                        (float(row[0]), str(row[1]), f"{int(row[2]):02d}:00", float(row[0]))
                    )
                    _LOGGER.info(
                        "Migration V16.0.0: Seeded all-time peak: %.1f W on %s at %02d:00",
                        float(row[0]), row[1], int(row[2])
                    )
            except Exception as e:
                _LOGGER.debug("Could not seed all-time peak: %s", e)

            await self._db.commit()
            _LOGGER.info("Migration V16.0.0: Peak power tracking: %s", ", ".join(migrations_applied))

    async def _migrate_dni_history_dedup_v1600(self) -> None:
        """V16.0.0: Deduplicate ai_dni_history table. @zara"""
        try:
            async with self._db.execute(
                """SELECT COUNT(*) FROM ai_dni_history
                   WHERE id NOT IN (
                       SELECT MIN(id) FROM ai_dni_history
                       GROUP BY hour, dni_value, date(recorded_at)
                   )"""
            ) as cursor:
                row = await cursor.fetchone()
            if row and row[0] > 0:
                await self._db.execute(
                    """DELETE FROM ai_dni_history WHERE id NOT IN (
                           SELECT MIN(id) FROM ai_dni_history
                           GROUP BY hour, dni_value, date(recorded_at)
                       )"""
                )
                await self._db.commit()
                _LOGGER.info(
                    "Migration V16.0.0: Removed %d duplicate DNI history entries", row[0]
                )
        except Exception as e:
            _LOGGER.debug("Migration V16.0.0 DNI dedup skipped: %s", e)

    async def _migrate_grid_search_model_type_v1600(self) -> None:
        """V16.0.0: Add model_type column to ai_grid_search_results. @zara"""
        async with self._db.execute(
            "PRAGMA table_info(ai_grid_search_results)"
        ) as cursor:
            columns = await cursor.fetchall()
            column_names = {col[1] for col in columns}

        if "model_type" not in column_names:
            await self._db.execute(
                "ALTER TABLE ai_grid_search_results ADD COLUMN model_type TEXT DEFAULT 'lstm'"
            )
            await self._db.commit()
            _LOGGER.info("Migration V16.0.0: Added 'model_type' to ai_grid_search_results")

    async def _migrate_prediction_weather_diffuse_v1600(self) -> None:
        """V16.0.0: Add diffuse_radiation column to prediction_weather. @zara"""
        async with self._db.execute(
            "PRAGMA table_info(prediction_weather)"
        ) as cursor:
            columns = await cursor.fetchall()
            column_names = {col[1] for col in columns}

        if "diffuse_radiation" not in column_names:
            await self._db.execute(
                "ALTER TABLE prediction_weather ADD COLUMN diffuse_radiation REAL"
            )
            await self._db.commit()
            _LOGGER.info("Migration V16.0.0: Added 'diffuse_radiation' to prediction_weather")

    async def _migrate_prediction_weather_direct_radiation_v1601(self) -> None:
        """V16.0.1: Add direct_radiation column to prediction_weather. @zara"""
        async with self._db.execute(
            "PRAGMA table_info(prediction_weather)"
        ) as cursor:
            columns = await cursor.fetchall()
            column_names = {col[1] for col in columns}

        if "direct_radiation" not in column_names:
            await self._db.execute(
                "ALTER TABLE prediction_weather ADD COLUMN direct_radiation REAL"
            )
            await self._db.commit()
            _LOGGER.info("Migration V16.0.1: Added 'direct_radiation' to prediction_weather")

    async def _migrate_ensemble_columns_v1602(self) -> None:
        """V16.0.2: Add lstm_kwh and ridge_kwh columns for ensemble diagnostics. @zara"""
        # hourly_predictions: lstm_kwh, ridge_kwh
        async with self._db.execute(
            "PRAGMA table_info(hourly_predictions)"
        ) as cursor:
            columns = await cursor.fetchall()
            hp_columns = {col[1] for col in columns}

        hp_migrations = []
        if "lstm_kwh" not in hp_columns:
            hp_migrations.append("ALTER TABLE hourly_predictions ADD COLUMN lstm_kwh REAL")
        if "ridge_kwh" not in hp_columns:
            hp_migrations.append("ALTER TABLE hourly_predictions ADD COLUMN ridge_kwh REAL")

        if hp_migrations:
            for migration in hp_migrations:
                await self._db.execute(migration)
            _LOGGER.info("Migration V16.0.2: Added %d ensemble columns to hourly_predictions", len(hp_migrations))

        # prediction_panel_groups: lstm_kwh, ridge_kwh
        async with self._db.execute(
            "PRAGMA table_info(prediction_panel_groups)"
        ) as cursor:
            columns = await cursor.fetchall()
            pg_columns = {col[1] for col in columns}

        pg_migrations = []
        if "lstm_kwh" not in pg_columns:
            pg_migrations.append("ALTER TABLE prediction_panel_groups ADD COLUMN lstm_kwh REAL")
        if "ridge_kwh" not in pg_columns:
            pg_migrations.append("ALTER TABLE prediction_panel_groups ADD COLUMN ridge_kwh REAL")

        if pg_migrations:
            for migration in pg_migrations:
                await self._db.execute(migration)
            _LOGGER.info("Migration V16.0.2: Added %d ensemble columns to prediction_panel_groups", len(pg_migrations))

        if hp_migrations or pg_migrations:
            await self._db.commit()

    async def _migrate_ensemble_group_weights_v1603(self) -> None:
        """V16.0.3: Create ensemble_group_weights table for per-group adaptive blending. @zara"""
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ensemble_group_weights'"
        ) as cursor:
            result = await cursor.fetchone()

        if not result:
            await self._db.execute(
                """CREATE TABLE IF NOT EXISTS ensemble_group_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_name TEXT NOT NULL,
                    cloud_bucket TEXT NOT NULL CHECK(cloud_bucket IN ('clear', 'partly_cloudy', 'overcast')),
                    hour_bucket TEXT NOT NULL CHECK(hour_bucket IN ('morning', 'midday', 'afternoon')),
                    lstm_weight REAL DEFAULT 0.85,
                    ridge_weight REAL DEFAULT 0.15,
                    lstm_mae REAL DEFAULT 0.0,
                    ridge_mae REAL DEFAULT 0.0,
                    sample_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP,
                    UNIQUE(group_name, cloud_bucket, hour_bucket)
                )"""
            )
            await self._db.commit()
            _LOGGER.info("Migration V16.0.3: Created ensemble_group_weights table")

    async def _migrate_ridge_meta_normalization_v1604(self) -> None:
        """V16.0.4: Add feature_means_json/feature_stds_json to ai_ridge_meta. @zara"""
        try:
            async with self._db.execute("PRAGMA table_info(ai_ridge_meta)") as cursor:
                columns = await cursor.fetchall()
                column_names = {col[1] for col in columns}
        except Exception:
            return  # Table doesn't exist yet

        if "feature_means_json" not in column_names:
            await self._db.execute(
                "ALTER TABLE ai_ridge_meta ADD COLUMN feature_means_json TEXT"
            )
            await self._db.commit()
            _LOGGER.info("Migration V16.0.4: Added feature_means_json to ai_ridge_meta")

        if "feature_stds_json" not in column_names:
            await self._db.execute(
                "ALTER TABLE ai_ridge_meta ADD COLUMN feature_stds_json TEXT"
            )
            await self._db.commit()
            _LOGGER.info("Migration V16.0.4: Added feature_stds_json to ai_ridge_meta")

    # ========================================================================
    # AI/ML Operations
    # ========================================================================

    async def save_seasonal_factors(self, factors: Dict[str, float], sample_counts: Dict[str, int]) -> None:
        """Save seasonal factors to database."""
        for month_str, factor in factors.items():
            month = int(month_str)
            sample_count = sample_counts.get(month_str, 0)
            await self._db.execute(
                """INSERT INTO ai_seasonal_factors (month, factor, sample_count, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(month) DO UPDATE SET
                       factor = excluded.factor,
                       sample_count = excluded.sample_count,
                       updated_at = excluded.updated_at""",
                (month, factor, sample_count, datetime.now())
            )
        await self._db.commit()

    async def get_seasonal_factors(self) -> Dict[int, Tuple[float, int]]:
        """Get all seasonal factors."""
        async with self._db.execute(
            "SELECT month, factor, sample_count FROM ai_seasonal_factors"
        ) as cursor:
            rows = await cursor.fetchall()
            return {row[0]: (row[1], row[2]) for row in rows}

    async def save_feature_importance(self, data: Dict[str, Any]) -> None:
        """Save feature importance analysis."""
        timestamp = datetime.fromisoformat(data["timestamp"])

        for feature, importance in data["feature_importance"].items():
            if feature in data["helpful_features"]:
                category = "helpful"
            elif feature in data["harmful_features"]:
                category = "harmful"
            else:
                category = "neutral"

            await self._db.execute(
                """INSERT INTO ai_feature_importance
                   (feature_name, importance, category, baseline_rmse, num_samples,
                    analysis_time_seconds, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (feature, importance, category, data.get("baseline_rmse"),
                 data.get("num_samples"), data.get("analysis_time_seconds"), timestamp)
            )
        await self._db.commit()

    async def get_latest_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get the latest feature importance analysis."""
        async with self._db.execute(
            """SELECT feature_name, importance, category
               FROM ai_feature_importance
               WHERE timestamp = (SELECT MAX(timestamp) FROM ai_feature_importance)"""
        ) as cursor:
            rows = await cursor.fetchall()
            if not rows:
                return None

            return {
                "feature_importance": {row[0]: row[1] for row in rows},
                "helpful_features": [row[0] for row in rows if row[2] == "helpful"],
                "harmful_features": [row[0] for row in rows if row[2] == "harmful"],
                "neutral_features": [row[0] for row in rows if row[2] == "neutral"]
            }

    async def save_grid_search_result(self, result: Dict[str, Any]) -> None:
        """Save grid search result."""
        timestamp = datetime.fromisoformat(result["timestamp"])

        for res in result.get("all_results", []):
            params = res["params"]
            await self._db.execute(
                """INSERT INTO ai_grid_search_results
                   (success, hidden_size, batch_size, learning_rate, accuracy,
                    epochs_trained, final_val_loss, duration_seconds, is_best_result,
                    hardware_info, timestamp, model_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (result["success"], params["hidden_size"], params["batch_size"],
                 params["learning_rate"], res.get("accuracy", 0), res.get("epochs_trained", 0),
                 res.get("final_val_loss", 0), res.get("duration_seconds", 0),
                 params == result["best_params"], json.dumps(result.get("hardware_info")),
                 timestamp, result.get("model_type", "lstm"))
            )
        await self._db.commit()

    async def save_dni_tracker(self, data: Dict[str, Any]) -> None:
        """Save DNI tracker data."""
        last_updated = datetime.fromisoformat(data["last_updated"]).date()

        for hour_str, max_dni in data["max_dni"].items():
            hour = int(hour_str)
            await self._db.execute(
                """INSERT INTO ai_dni_tracker (hour, max_dni, last_updated)
                   VALUES (?, ?, ?)
                   ON CONFLICT(hour) DO UPDATE SET
                       max_dni = excluded.max_dni,
                       last_updated = excluded.last_updated""",
                (hour, max_dni, last_updated)
            )

            if hour_str in data["history"]:
                await self._db.execute(
                    "DELETE FROM ai_dni_history WHERE hour = ?", (hour,)
                )
                for dni_value in data["history"][hour_str]:
                    await self._db.execute(
                        "INSERT INTO ai_dni_history (hour, dni_value) VALUES (?, ?)",
                        (hour, dni_value)
                    )
        await self._db.commit()

    async def save_model_weights(self, weights_data: Dict[str, Any]) -> None:
        """Save AI model weights to structured tables. @zara"""
        # Save meta data to ai_learned_weights_meta @zara
        await self._db.execute(
            """INSERT INTO ai_learned_weights_meta
               (id, version, active_model, training_samples, last_trained, accuracy, rmse, updated_at)
               VALUES (1, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   version = excluded.version,
                   active_model = excluded.active_model,
                   training_samples = excluded.training_samples,
                   last_trained = excluded.last_trained,
                   accuracy = excluded.accuracy,
                   rmse = excluded.rmse,
                   updated_at = excluded.updated_at""",
            (weights_data.get("version", "2.0"),
             weights_data.get("active_model", "none"),
             weights_data.get("training_samples", 0),
             weights_data.get("last_trained"),
             weights_data.get("accuracy"),
             weights_data.get("rmse"),
             datetime.now())
        )

        # Save LSTM weights if present @zara
        lstm_data = weights_data.get("lstm")
        if lstm_data:
            # Inject training metadata from parent into lstm_data @zara
            lstm_data["training_samples"] = weights_data.get("training_samples")
            lstm_data["accuracy"] = weights_data.get("accuracy")
            lstm_data["rmse"] = weights_data.get("rmse")
            await self._save_lstm_weights(lstm_data)

        # Save Ridge weights if present @zara
        ridge_data = weights_data.get("ridge")
        if ridge_data:
            # Inject training metadata from parent into ridge_data @zara
            ridge_data["training_samples"] = weights_data.get("training_samples")
            await self._save_ridge_weights(ridge_data)

        # V16.1 FIX: Removed JSON blob saving to ai_model_weights @zara
        # The structured tables (ai_lstm_weights, ai_ridge_weights, ai_lstm_meta, etc.)
        # are the single source of truth. JSON blobs caused has_attention inconsistency.

        await self._db.commit()
        _LOGGER.debug("Model weights saved to structured tables")

    async def _save_lstm_weights(self, lstm_data: Dict[str, Any]) -> None:
        """Save LSTM weights to ai_lstm_weights table. @zara"""
        # Save metadata to ai_lstm_meta @zara
        await self._db.execute(
            """INSERT INTO ai_lstm_meta
               (id, input_size, hidden_size, sequence_length, num_outputs, has_attention,
                training_samples, accuracy, rmse, updated_at)
               VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   input_size = excluded.input_size,
                   hidden_size = excluded.hidden_size,
                   sequence_length = excluded.sequence_length,
                   num_outputs = excluded.num_outputs,
                   has_attention = excluded.has_attention,
                   training_samples = excluded.training_samples,
                   accuracy = excluded.accuracy,
                   rmse = excluded.rmse,
                   updated_at = excluded.updated_at""",
            (lstm_data.get("input_size"),
             lstm_data.get("hidden_size"),
             lstm_data.get("sequence_length", 24),
             lstm_data.get("num_outputs"),
             lstm_data.get("has_attention", False),
             lstm_data.get("training_samples"),
             lstm_data.get("accuracy"),
             lstm_data.get("rmse"),
             datetime.now())
        )

        # Delete old weights before inserting new ones to avoid UNIQUE constraint errors @zara
        await self._db.execute("DELETE FROM ai_lstm_weights")

        # Save each weight type (including attention weights) @zara
        weight_types = ['Wf', 'Wi', 'Wc', 'Wo', 'bf', 'bi', 'bc', 'bo', 'Wy', 'by',
                        'W_query', 'W_key', 'W_value', 'W_attn_out', 'b_attn_out']
        for wtype in weight_types:
            if wtype not in lstm_data:
                continue
            weight_array = lstm_data[wtype]
            # Flatten if 2D @zara
            if isinstance(weight_array, list) and len(weight_array) > 0 and isinstance(weight_array[0], list):
                flat = [val for row in weight_array for val in row]
            elif isinstance(weight_array, list):
                flat = weight_array
            else:
                # Handle numpy arrays @zara
                if hasattr(weight_array, 'flatten'):
                    flat = weight_array.flatten().tolist()
                else:
                    flat = list(weight_array)
            # Insert each value @zara
            for idx, val in enumerate(flat):
                await self._db.execute(
                    "INSERT OR REPLACE INTO ai_lstm_weights (weight_type, weight_index, weight_value) VALUES (?, ?, ?)",
                    (wtype, idx, float(val))
                )

    async def _save_ridge_weights(self, ridge_data: Dict[str, Any]) -> None:
        """Save Ridge weights to ai_ridge_weights table. @zara"""
        # V16.0.4: Serialize feature_means/stds as JSON for persistence @zara
        feature_means_json = None
        feature_stds_json = None
        if ridge_data.get("feature_means") is not None:
            feature_means_json = json.dumps(ridge_data["feature_means"])
        if ridge_data.get("feature_stds") is not None:
            feature_stds_json = json.dumps(ridge_data["feature_stds"])

        # Save metadata to ai_ridge_meta @zara
        await self._db.execute(
            """INSERT INTO ai_ridge_meta
               (id, model_type, alpha, input_size, hidden_size, sequence_length,
                num_outputs, flat_size, trained_samples, loo_cv_score, accuracy, rmse,
                feature_means_json, feature_stds_json, updated_at)
               VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   model_type = excluded.model_type,
                   alpha = excluded.alpha,
                   input_size = excluded.input_size,
                   hidden_size = excluded.hidden_size,
                   sequence_length = excluded.sequence_length,
                   num_outputs = excluded.num_outputs,
                   flat_size = excluded.flat_size,
                   trained_samples = excluded.trained_samples,
                   loo_cv_score = excluded.loo_cv_score,
                   accuracy = excluded.accuracy,
                   rmse = excluded.rmse,
                   feature_means_json = excluded.feature_means_json,
                   feature_stds_json = excluded.feature_stds_json,
                   updated_at = excluded.updated_at""",
            (ridge_data.get("model_type", "TinyRidge"),
             ridge_data.get("alpha"),
             ridge_data.get("input_size"),
             ridge_data.get("hidden_size"),
             ridge_data.get("sequence_length", 24),
             ridge_data.get("num_outputs"),
             ridge_data.get("flat_size"),
             ridge_data.get("trained_samples"),
             ridge_data.get("loo_cv_score"),
             ridge_data.get("accuracy"),
             ridge_data.get("rmse"),
             feature_means_json,
             feature_stds_json,
             datetime.now())
        )

        # Clear existing weights @zara
        await self._db.execute("DELETE FROM ai_ridge_weights")

        weights = ridge_data.get("weights")
        if not weights:
            return

        # Save as sparse matrix @zara
        for row_idx, row in enumerate(weights):
            for col_idx, val in enumerate(row):
                if val != 0.0:  # Only save non-zero values @zara
                    await self._db.execute(
                        "INSERT INTO ai_ridge_weights (row_index, col_index, weight_value) VALUES (?, ?, ?)",
                        (row_idx, col_idx, val)
                    )

    async def get_model_weights(self) -> Optional[Dict[str, Any]]:
        """Get AI model weights from database. @zara

        V16.1 FIX: Now uses structured tables (ai_lstm_weights, ai_ridge_weights)
        as the primary source instead of JSON blobs in ai_model_weights.
        This ensures has_attention flag is consistent with actual attention weights.
        """
        # Primary: Load from structured tables @zara V16.1
        meta = await self.fetchone(
            """SELECT version, active_model, training_samples, last_trained,
                      accuracy, rmse
               FROM ai_learned_weights_meta WHERE id = 1"""
        )

        if not meta:
            return None

        result = {
            "version": meta[0] or "2.0",
            "active_model": meta[1] or "none",
            "training_samples": meta[2] or 0,
            "last_trained": str(meta[3]) if meta[3] else None,
            "accuracy": meta[4],
            "rmse": meta[5],
        }

        # Load LSTM weights from structured table @zara
        lstm_weights = await self._load_lstm_weights()
        if lstm_weights:
            result["lstm"] = lstm_weights
            result["has_attention"] = lstm_weights.get("has_attention", False)

        # Load Ridge weights from structured table @zara
        ridge_weights = await self._load_ridge_weights()
        if ridge_weights:
            result["ridge"] = ridge_weights

        return result

    async def _get_model_weights_legacy(self) -> Optional[Dict[str, Any]]:
        """Fallback to old ai_model_weights table. @zara"""
        rows = await self.fetchall("SELECT weight_type, weight_data FROM ai_model_weights")
        if not rows:
            return None
        return {row[0]: json.loads(row[1]) for row in rows}

    async def _load_lstm_weights(self) -> Optional[Dict[str, Any]]:
        """Load LSTM weights from ai_lstm_weights table. @zara"""
        rows = await self.fetchall(
            """SELECT weight_type, weight_index, weight_value
               FROM ai_lstm_weights ORDER BY weight_type, weight_index"""
        )
        if not rows:
            return None

        # Group by weight_type @zara
        weights_by_type: Dict[str, List[float]] = {}
        for row in rows:
            weight_type = row[0]
            if weight_type not in weights_by_type:
                weights_by_type[weight_type] = []
            weights_by_type[weight_type].append(row[2])

        # Determine dimensions from data @zara
        hidden_size = len(weights_by_type.get('bf', []))
        num_outputs = len(weights_by_type.get('by', []))

        if hidden_size == 0:
            return None

        # Calculate input_size from Wf shape @zara
        # Wf has shape (hidden_size, input_size + hidden_size)
        wf_len = len(weights_by_type.get('Wf', []))
        concat_size = wf_len // hidden_size if hidden_size > 0 else 0
        input_size = concat_size - hidden_size if concat_size > hidden_size else 0

        # Load metadata from ai_lstm_meta table @zara
        has_attention = False
        training_samples = 0
        accuracy = None
        rmse = None
        meta_row = await self.fetchone(
            "SELECT has_attention, training_samples, accuracy, rmse FROM ai_lstm_meta WHERE id = 1"
        )
        if meta_row:
            has_attention = bool(meta_row[0])
            training_samples = meta_row[1] or 0
            accuracy = meta_row[2]
            rmse = meta_row[3]

        result = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_outputs': num_outputs,
            'has_attention': has_attention,
            'training_samples': training_samples,
            'accuracy': accuracy,
            'rmse': rmse,
        }

        # Reshape weight matrices (hidden_size, concat_size) @zara
        for wtype in ['Wf', 'Wi', 'Wc', 'Wo']:
            if wtype in weights_by_type:
                flat = weights_by_type[wtype]
                if concat_size > 0:
                    result[wtype] = [flat[i*concat_size:(i+1)*concat_size] for i in range(hidden_size)]

        # Wy has shape (num_outputs, hidden_size) @zara
        if 'Wy' in weights_by_type and num_outputs > 0:
            flat = weights_by_type['Wy']
            result['Wy'] = [flat[i*hidden_size:(i+1)*hidden_size] for i in range(num_outputs)]

        # Biases are 1D @zara
        for btype in ['bf', 'bi', 'bc', 'bo', 'by']:
            if btype in weights_by_type:
                result[btype] = weights_by_type[btype]

        # V16.1 FIX: Verify attention weights actually exist before enabling @zara
        attention_keys = ['W_query', 'W_key', 'W_value', 'W_attn_out', 'b_attn_out']
        attention_weights_exist = all(k in weights_by_type for k in attention_keys)

        # Only enable attention if meta says so AND weights actually exist @zara
        if has_attention and attention_weights_exist:
            # W_query, W_key, W_value: shape (hidden_size, hidden_size)
            for wtype in ['W_query', 'W_key', 'W_value']:
                flat = weights_by_type[wtype]
                result[wtype] = [flat[i*hidden_size:(i+1)*hidden_size] for i in range(hidden_size)]

            # W_attn_out: shape (hidden_size, 2*hidden_size)
            flat = weights_by_type['W_attn_out']
            double_hidden = 2 * hidden_size
            result['W_attn_out'] = [flat[i*double_hidden:(i+1)*double_hidden] for i in range(hidden_size)]

            # b_attn_out: shape (hidden_size,)
            result['b_attn_out'] = weights_by_type['b_attn_out']
        elif has_attention and not attention_weights_exist:
            # Meta says attention but weights missing - fix the flag @zara V16.1
            result['has_attention'] = False
            _LOGGER.warning(
                "ai_lstm_meta.has_attention=True but attention weights missing in ai_lstm_weights. "
                "Setting has_attention=False."
            )

        return result

    async def _load_ridge_weights(self) -> Optional[Dict[str, Any]]:
        """Load Ridge weights from ai_ridge_weights table. @zara"""
        rows = await self.fetchall(
            """SELECT row_index, col_index, weight_value
               FROM ai_ridge_weights ORDER BY row_index, col_index"""
        )
        if not rows:
            return None

        # Determine dimensions from actual weights @zara
        max_row = max(row[0] for row in rows)
        max_col = max(row[1] for row in rows)
        num_outputs = max_row + 1
        num_features_from_weights = max_col + 1

        # Load metadata from ai_ridge_meta table @zara
        input_size = None
        sequence_length = 24
        flat_size = None
        alpha = None
        loo_cv_score = None
        trained_samples = 0
        accuracy = None
        rmse = None
        feature_means_json = None
        feature_stds_json = None
        meta_row = await self.fetchone(
            """SELECT input_size, sequence_length, num_outputs, flat_size,
                      alpha, loo_cv_score, trained_samples, accuracy, rmse,
                      feature_means_json, feature_stds_json
               FROM ai_ridge_meta WHERE id = 1"""
        )
        if meta_row:
            input_size = meta_row[0]
            sequence_length = meta_row[1] or 24
            flat_size = meta_row[3]
            alpha = meta_row[4]
            loo_cv_score = meta_row[5]
            trained_samples = meta_row[6] or 0
            accuracy = meta_row[7]
            rmse = meta_row[8]
            feature_means_json = meta_row[9]
            feature_stds_json = meta_row[10]

        # V16.0.4 FIX: Use meta flat_size + 1 for column count (includes bias column) @zara
        # Weights shape is (num_outputs, flat_size + 1) because _add_bias() adds a ones column
        if flat_size:
            col_count = flat_size + 1  # flat_size features + 1 bias
        else:
            col_count = num_features_from_weights  # Fallback to sparse dimensions

        # Safety: ensure matrix large enough for all stored weights
        if col_count < num_features_from_weights:
            col_count = num_features_from_weights

        # Create 2D array (sparse to dense) @zara
        weights = [[0.0] * col_count for _ in range(num_outputs)]
        for row in rows:
            weights[row[0]][row[1]] = row[2]

        # V16.0.4: Deserialize feature_means/stds from JSON @zara
        feature_means = None
        feature_stds = None
        if feature_means_json:
            try:
                feature_means = json.loads(feature_means_json)
            except (json.JSONDecodeError, TypeError):
                pass
        if feature_stds_json:
            try:
                feature_stds = json.loads(feature_stds_json)
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            'model_type': 'TinyRidge',
            'weights': weights,
            'num_outputs': num_outputs,
            'input_size': input_size if input_size else (flat_size // sequence_length if flat_size else num_features_from_weights),
            'flat_size': flat_size if flat_size else num_features_from_weights,  # V16.0.4: Use meta flat_size @zara
            'sequence_length': sequence_length,
            'alpha': alpha,
            'loo_cv_score': loo_cv_score,
            'trained_samples': trained_samples,
            'accuracy': accuracy,
            'rmse': rmse,
            'feature_means': feature_means,
            'feature_stds': feature_stds,
        }

    # ========================================================================
    # Physics Calibration Operations
    # ========================================================================

    async def save_learning_config(self, config: Dict[str, Any]) -> None:
        """Save physics learning config."""
        defaults = config["physics_defaults"]
        metadata = config["metadata"]

        await self._db.execute(
            """INSERT INTO physics_learning_config
               (id, version, albedo, system_efficiency, learned_efficiency_factor,
                rolling_window_days, min_samples, updated_at)
               VALUES (1, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   version = excluded.version,
                   albedo = excluded.albedo,
                   system_efficiency = excluded.system_efficiency,
                   learned_efficiency_factor = excluded.learned_efficiency_factor,
                   rolling_window_days = excluded.rolling_window_days,
                   min_samples = excluded.min_samples,
                   updated_at = excluded.updated_at""",
            (config["version"], defaults["albedo"], defaults["system_efficiency"],
             defaults["learned_efficiency_factor"], metadata["rolling_window_days"],
             metadata["min_samples"], datetime.fromisoformat(config["updated_at"]))
        )

        for group_name, group_data in config["group_calibration"].items():
            await self._save_group_calibration(group_name, group_data)

        await self._db.commit()

    async def _save_group_calibration(self, group_name: str, group_data: Dict[str, Any]) -> None:
        """Save calibration data for a panel group."""
        await self._db.execute(
            """INSERT INTO physics_calibration_groups
               (group_name, global_factor, sample_count, confidence, last_updated)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(group_name) DO UPDATE SET
                   global_factor = excluded.global_factor,
                   sample_count = excluded.sample_count,
                   confidence = excluded.confidence,
                   last_updated = excluded.last_updated""",
            (group_name, group_data["global_factor"], group_data["sample_count"],
             group_data["confidence"], datetime.fromisoformat(group_data["last_updated"]))
        )

        await self._db.execute(
            "DELETE FROM physics_calibration_hourly WHERE group_name = ?",
            (group_name,)
        )
        for hour_str, factor in group_data["hourly_factors"].items():
            hour = int(hour_str)
            await self._db.execute(
                "INSERT INTO physics_calibration_hourly (group_name, hour, factor) VALUES (?, ?, ?)",
                (group_name, hour, factor)
            )

        await self._db.execute(
            "DELETE FROM physics_calibration_buckets WHERE group_name = ?",
            (group_name,)
        )
        await self._db.execute(
            "DELETE FROM physics_calibration_bucket_hourly WHERE group_name = ?",
            (group_name,)
        )

        for bucket_name, bucket_data in group_data["bucket_factors"].items():
            await self._db.execute(
                """INSERT INTO physics_calibration_buckets
                   (group_name, bucket_name, global_factor, sample_count, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (group_name, bucket_name, bucket_data["global_factor"],
                 bucket_data["sample_count"], bucket_data["confidence"])
            )

            for hour_str, factor in bucket_data["hourly_factors"].items():
                hour = int(hour_str)
                await self._db.execute(
                    """INSERT INTO physics_calibration_bucket_hourly
                       (group_name, bucket_name, hour, factor) VALUES (?, ?, ?, ?)""",
                    (group_name, bucket_name, hour, factor)
                )

    async def get_group_calibration(self, group_name: str) -> Optional[Dict[str, Any]]:
        """Get calibration data for a panel group."""
        async with self._db.execute(
            """SELECT global_factor, sample_count, confidence, last_updated
               FROM physics_calibration_groups WHERE group_name = ?""",
            (group_name,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None

        hourly = {}
        async with self._db.execute(
            "SELECT hour, factor FROM physics_calibration_hourly WHERE group_name = ?",
            (group_name,)
        ) as cursor:
            async for row in cursor:
                hourly[str(row[0])] = row[1]

        buckets = {}
        async with self._db.execute(
            """SELECT bucket_name, global_factor, sample_count, confidence
               FROM physics_calibration_buckets WHERE group_name = ?""",
            (group_name,)
        ) as cursor:
            async for row in cursor:
                bucket_name = row[0]
                buckets[bucket_name] = {
                    "global_factor": row[1],
                    "sample_count": row[2],
                    "confidence": row[3],
                    "hourly_factors": {}
                }

        async with self._db.execute(
            "SELECT bucket_name, hour, factor FROM physics_calibration_bucket_hourly WHERE group_name = ?",
            (group_name,)
        ) as cursor:
            async for row in cursor:
                buckets[row[0]]["hourly_factors"][str(row[1])] = row[2]

        return {
            "global_factor": row[0],
            "sample_count": row[1],
            "confidence": row[2],
            "last_updated": row[3],
            "hourly_factors": hourly,
            "bucket_factors": buckets
        }

    async def save_calibration_history(self, history_data: List[Dict[str, Any]]) -> None:
        """Save calibration history."""
        for entry in history_data:
            entry_date = datetime.fromisoformat(entry["date"]).date()

            for group_name, group_data in entry["groups"].items():
                for hour_str, avg_ratio in group_data["hourly"].items():
                    hour = int(hour_str)
                    await self._db.execute(
                        """INSERT INTO physics_calibration_history
                           (date, group_name, bucket_name, hour, avg_ratio, sample_count, source)
                           VALUES (?, ?, NULL, ?, ?, ?, ?)
                           ON CONFLICT(date, group_name, bucket_name, hour) DO UPDATE SET
                               avg_ratio = excluded.avg_ratio,
                               sample_count = excluded.sample_count""",
                        (entry_date, group_name, hour, avg_ratio,
                         group_data["sample_count"], entry.get("source"))
                    )

                for bucket_name, bucket_data in group_data.get("buckets", {}).items():
                    for hour_str, avg_ratio in bucket_data["hourly"].items():
                        hour = int(hour_str)
                        await self._db.execute(
                            """INSERT INTO physics_calibration_history
                               (date, group_name, bucket_name, hour, avg_ratio, sample_count, source)
                               VALUES (?, ?, ?, ?, ?, ?, ?)
                               ON CONFLICT(date, group_name, bucket_name, hour) DO UPDATE SET
                                   avg_ratio = excluded.avg_ratio,
                                   sample_count = excluded.sample_count""",
                            (entry_date, group_name, bucket_name, hour, avg_ratio,
                             bucket_data["sample_count"], entry.get("source"))
                        )
        await self._db.commit()

    # ========================================================================
    # Weather Operations
    # ========================================================================

    async def save_weather_expert_weights(self, data: Dict[str, Any]) -> None:
        """Save weather expert weights."""
        for cloud_type, experts in data["weights"].items():
            for expert_name, weight in experts.items():
                await self._db.execute(
                    """INSERT INTO weather_expert_weights
                       (cloud_type, expert_name, weight, last_updated)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(cloud_type, expert_name) DO UPDATE SET
                           weight = excluded.weight,
                           last_updated = excluded.last_updated""",
                    (cloud_type, expert_name, weight, datetime.fromisoformat(data["metadata"]["last_updated"]))
                )
        await self._db.commit()

    async def get_weather_expert_weights(self, cloud_type: str) -> Dict[str, float]:
        """Get weather expert weights for a cloud type."""
        async with self._db.execute(
            "SELECT expert_name, weight FROM weather_expert_weights WHERE cloud_type = ?",
            (cloud_type,)
        ) as cursor:
            rows = await cursor.fetchall()
            return {row[0]: row[1] for row in rows}

    async def save_weather_source_weights(self, data: Dict[str, Any]) -> None:
        """Save weather source weights."""
        for source_name, weight in data["weights"].items():
            last_mae = data["learning_metadata"]["last_mae"].get(source_name)
            await self._db.execute(
                """INSERT INTO weather_source_weights
                   (source_name, weight, last_mae, version, last_updated)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(source_name) DO UPDATE SET
                       weight = excluded.weight,
                       last_mae = excluded.last_mae,
                       last_updated = excluded.last_updated""",
                (source_name, weight, last_mae, data["version"],
                 datetime.fromisoformat(data["learning_metadata"]["last_updated"]))
            )
        await self._db.commit()

    # ========================================================================
    # State Operations
    # ========================================================================

    async def save_coordinator_state(self, state: Dict[str, Any]) -> None:
        """Save coordinator state."""
        await self._db.execute(
            """INSERT INTO coordinator_state
               (id, expected_daily_production, last_set_date, last_updated)
               VALUES (1, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   expected_daily_production = excluded.expected_daily_production,
                   last_set_date = excluded.last_set_date,
                   last_updated = excluded.last_updated""",
            (state["expected_daily_production"],
             datetime.fromisoformat(state["last_set_date"]).date(),
             datetime.fromisoformat(state["last_updated"]))
        )
        await self._db.commit()

    async def get_coordinator_state(self) -> Optional[Dict[str, Any]]:
        """Get coordinator state."""
        async with self._db.execute(
            "SELECT expected_daily_production, last_set_date FROM coordinator_state WHERE id = 1"
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "expected_daily_production": row[0],
                "last_set_date": str(row[1])
            }

    async def save_production_time_state(self, state: Dict[str, Any]) -> None:
        """Save production time state."""
        await self._db.execute(
            """INSERT INTO production_time_state
               (id, date, accumulated_hours, is_active, start_time, production_time_today, last_updated)
               VALUES (1, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   date = excluded.date,
                   accumulated_hours = excluded.accumulated_hours,
                   is_active = excluded.is_active,
                   start_time = excluded.start_time,
                   production_time_today = excluded.production_time_today,
                   last_updated = excluded.last_updated""",
            (datetime.fromisoformat(state["date"]).date(),
             state["accumulated_hours"], state["is_active"],
             state.get("start_time"), state["production_time_today"],
             datetime.fromisoformat(state["last_updated"]))
        )
        await self._db.commit()

    async def save_panel_group_sensor_state(self, state: Dict[str, Any]) -> None:
        """Save panel group sensor state."""
        for group_name, last_value in state["last_values"].items():
            await self._db.execute(
                """INSERT INTO panel_group_sensor_state (group_name, last_value, last_updated)
                   VALUES (?, ?, ?)
                   ON CONFLICT(group_name) DO UPDATE SET
                       last_value = excluded.last_value,
                       last_updated = excluded.last_updated""",
                (group_name, last_value, datetime.fromisoformat(state["last_updated"]))
            )
        await self._db.commit()

    async def save_yield_cache(self, cache: Dict[str, Any]) -> None:
        """Save yield cache."""
        await self._db.execute(
            """INSERT INTO yield_cache (id, value, time, date)
               VALUES (1, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   value = excluded.value,
                   time = excluded.time,
                   date = excluded.date""",
            (cache["value"], datetime.fromisoformat(cache["time"]),
             datetime.fromisoformat(cache["date"]).date())
        )
        await self._db.commit()

    # ========================================================================
    # Daily Forecast Tracking Operations
    # ========================================================================

    async def save_daily_forecast_tracking(self, data: Dict[str, Any]) -> None:
        """Save daily forecast tracking data."""
        await self._db.execute(
            """INSERT INTO daily_forecast_tracking
               (id, date, forecast_best_hour, forecast_best_hour_kwh, forecast_best_hour_locked,
                forecast_best_hour_locked_at, forecast_best_hour_source,
                actual_best_hour, actual_best_hour_kwh, actual_best_hour_saved_at,
                forecast_next_hour_period, forecast_next_hour_kwh, forecast_next_hour_updated_at,
                forecast_next_hour_source, production_time_active, production_time_duration_seconds,
                production_time_start, production_time_end, production_time_last_power_above_10w,
                production_time_zero_power_since, peak_today_power_w, peak_today_at,
                yield_today_kwh, yield_today_sensor, consumption_today_kwh, consumption_today_sensor,
                autarky_percent, autarky_calculated_at, finalized_yield_kwh, finalized_consumption_kwh,
                finalized_production_hours, finalized_accuracy_percent, finalized_excluded_hours_count,
                finalized_excluded_hours_total, finalized_excluded_hours_ratio,
                finalized_excluded_hours_reasons, finalized_at, last_updated)
               VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   date = excluded.date,
                   forecast_best_hour = excluded.forecast_best_hour,
                   forecast_best_hour_kwh = excluded.forecast_best_hour_kwh,
                   forecast_best_hour_locked = excluded.forecast_best_hour_locked,
                   forecast_best_hour_locked_at = excluded.forecast_best_hour_locked_at,
                   forecast_best_hour_source = excluded.forecast_best_hour_source,
                   actual_best_hour = excluded.actual_best_hour,
                   actual_best_hour_kwh = excluded.actual_best_hour_kwh,
                   actual_best_hour_saved_at = excluded.actual_best_hour_saved_at,
                   forecast_next_hour_period = excluded.forecast_next_hour_period,
                   forecast_next_hour_kwh = excluded.forecast_next_hour_kwh,
                   forecast_next_hour_updated_at = excluded.forecast_next_hour_updated_at,
                   forecast_next_hour_source = excluded.forecast_next_hour_source,
                   production_time_active = excluded.production_time_active,
                   production_time_duration_seconds = excluded.production_time_duration_seconds,
                   production_time_start = excluded.production_time_start,
                   production_time_end = excluded.production_time_end,
                   production_time_last_power_above_10w = excluded.production_time_last_power_above_10w,
                   production_time_zero_power_since = excluded.production_time_zero_power_since,
                   peak_today_power_w = excluded.peak_today_power_w,
                   peak_today_at = excluded.peak_today_at,
                   yield_today_kwh = excluded.yield_today_kwh,
                   yield_today_sensor = excluded.yield_today_sensor,
                   consumption_today_kwh = excluded.consumption_today_kwh,
                   consumption_today_sensor = excluded.consumption_today_sensor,
                   autarky_percent = excluded.autarky_percent,
                   autarky_calculated_at = excluded.autarky_calculated_at,
                   finalized_yield_kwh = excluded.finalized_yield_kwh,
                   finalized_consumption_kwh = excluded.finalized_consumption_kwh,
                   finalized_production_hours = excluded.finalized_production_hours,
                   finalized_accuracy_percent = excluded.finalized_accuracy_percent,
                   finalized_excluded_hours_count = excluded.finalized_excluded_hours_count,
                   finalized_excluded_hours_total = excluded.finalized_excluded_hours_total,
                   finalized_excluded_hours_ratio = excluded.finalized_excluded_hours_ratio,
                   finalized_excluded_hours_reasons = excluded.finalized_excluded_hours_reasons,
                   finalized_at = excluded.finalized_at,
                   last_updated = excluded.last_updated""",
            (
                datetime.fromisoformat(data["date"]).date() if isinstance(data["date"], str) else data["date"],
                data.get("forecast_best_hour", {}).get("hour"),
                data.get("forecast_best_hour", {}).get("prediction_kwh"),
                data.get("forecast_best_hour", {}).get("locked", False),
                data.get("forecast_best_hour", {}).get("locked_at"),
                data.get("forecast_best_hour", {}).get("source"),
                data.get("actual_best_hour", {}).get("hour"),
                data.get("actual_best_hour", {}).get("actual_kwh"),
                data.get("actual_best_hour", {}).get("saved_at"),
                data.get("forecast_next_hour", {}).get("period"),
                data.get("forecast_next_hour", {}).get("prediction_kwh"),
                data.get("forecast_next_hour", {}).get("updated_at"),
                data.get("forecast_next_hour", {}).get("source"),
                data.get("production_time", {}).get("active", False),
                data.get("production_time", {}).get("duration_seconds"),
                data.get("production_time", {}).get("start_time"),
                data.get("production_time", {}).get("end_time"),
                data.get("production_time", {}).get("last_power_above_10w"),
                data.get("production_time", {}).get("zero_power_since"),
                data.get("peak_today", {}).get("power_w"),
                data.get("peak_today", {}).get("at"),
                data.get("yield_today", {}).get("kwh"),
                data.get("yield_today", {}).get("sensor"),
                data.get("consumption_today", {}).get("kwh"),
                data.get("consumption_today", {}).get("sensor"),
                data.get("autarky", {}).get("percent"),
                data.get("autarky", {}).get("calculated_at"),
                data.get("finalized", {}).get("yield_kwh"),
                data.get("finalized", {}).get("consumption_kwh"),
                data.get("finalized", {}).get("production_hours"),
                data.get("finalized", {}).get("accuracy_percent"),
                data.get("finalized", {}).get("excluded_hours", {}).get("count"),
                data.get("finalized", {}).get("excluded_hours", {}).get("total"),
                data.get("finalized", {}).get("excluded_hours", {}).get("ratio"),
                json.dumps(data.get("finalized", {}).get("excluded_hours", {}).get("reasons")),
                data.get("finalized", {}).get("at"),
                datetime.now()
            )
        )
        await self._db.commit()

    async def get_daily_forecast_tracking(self) -> Optional[Dict[str, Any]]:
        """Get daily forecast tracking data."""
        row = await self.fetchone("SELECT * FROM daily_forecast_tracking WHERE id = 1")
        if not row:
            return None

        return {
            "date": str(row["date"]),
            "forecast_best_hour": {
                "hour": row["forecast_best_hour"],
                "prediction_kwh": row["forecast_best_hour_kwh"],
                "locked": bool(row["forecast_best_hour_locked"]),
                "locked_at": str(row["forecast_best_hour_locked_at"]) if row["forecast_best_hour_locked_at"] else None,
                "source": row["forecast_best_hour_source"]
            } if row["forecast_best_hour"] is not None else {},
            "actual_best_hour": {
                "hour": row["actual_best_hour"],
                "actual_kwh": row["actual_best_hour_kwh"],
                "saved_at": str(row["actual_best_hour_saved_at"]) if row["actual_best_hour_saved_at"] else None
            } if row["actual_best_hour"] is not None else {},
            "forecast_next_hour": {
                "period": row["forecast_next_hour_period"],
                "prediction_kwh": row["forecast_next_hour_kwh"],
                "updated_at": str(row["forecast_next_hour_updated_at"]) if row["forecast_next_hour_updated_at"] else None,
                "source": row["forecast_next_hour_source"]
            } if row["forecast_next_hour_period"] is not None else {},
            "production_time": {
                "active": bool(row["production_time_active"]),
                "duration_seconds": row["production_time_duration_seconds"],
                "start_time": str(row["production_time_start"]) if row["production_time_start"] else None,
                "end_time": str(row["production_time_end"]) if row["production_time_end"] else None,
                "last_power_above_10w": str(row["production_time_last_power_above_10w"]) if row["production_time_last_power_above_10w"] else None,
                "zero_power_since": str(row["production_time_zero_power_since"]) if row["production_time_zero_power_since"] else None
            },
            "peak_today": {
                "power_w": row["peak_today_power_w"],
                "at": str(row["peak_today_at"]) if row["peak_today_at"] else None
            } if row["peak_today_power_w"] is not None else {},
            "yield_today": {
                "kwh": row["yield_today_kwh"],
                "sensor": row["yield_today_sensor"]
            } if row["yield_today_kwh"] is not None else {},
            "consumption_today": {
                "kwh": row["consumption_today_kwh"],
                "sensor": row["consumption_today_sensor"]
            } if row["consumption_today_kwh"] is not None else {},
            "autarky": {
                "percent": row["autarky_percent"],
                "calculated_at": str(row["autarky_calculated_at"]) if row["autarky_calculated_at"] else None
            } if row["autarky_percent"] is not None else {},
            "finalized": {
                "yield_kwh": row["finalized_yield_kwh"],
                "consumption_kwh": row["finalized_consumption_kwh"],
                "production_hours": row["finalized_production_hours"],
                "accuracy_percent": row["finalized_accuracy_percent"],
                "excluded_hours": {
                    "count": row["finalized_excluded_hours_count"],
                    "total": row["finalized_excluded_hours_total"],
                    "ratio": row["finalized_excluded_hours_ratio"],
                    "reasons": json.loads(row["finalized_excluded_hours_reasons"]) if row["finalized_excluded_hours_reasons"] else {}
                } if row["finalized_excluded_hours_count"] is not None else {},
                "at": str(row["finalized_at"]) if row["finalized_at"] else None
            } if row["finalized_yield_kwh"] is not None else {}
        }

    # ========================================================================
    # Daily Statistics Operations
    # ========================================================================

    async def save_daily_statistics(self, data: Dict[str, Any]) -> None:
        """Save daily statistics data."""
        await self._db.execute(
            """INSERT INTO daily_statistics
               (id, all_time_peak_power_w, all_time_peak_date, all_time_peak_at,
                current_week_period, current_week_date_range, current_week_yield_kwh,
                current_week_consumption_kwh, current_week_days, current_week_updated_at,
                current_month_period, current_month_yield_kwh, current_month_consumption_kwh,
                current_month_avg_autarky, current_month_days, current_month_updated_at,
                last_7_days_avg_yield_kwh, last_7_days_avg_accuracy, last_7_days_total_yield_kwh,
                last_7_days_calculated_at, last_30_days_avg_yield_kwh, last_30_days_avg_accuracy,
                last_30_days_total_yield_kwh, last_30_days_calculated_at, last_365_days_avg_yield_kwh,
                last_365_days_total_yield_kwh, last_365_days_calculated_at, last_updated)
               VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   all_time_peak_power_w = excluded.all_time_peak_power_w,
                   all_time_peak_date = excluded.all_time_peak_date,
                   all_time_peak_at = excluded.all_time_peak_at,
                   current_week_period = excluded.current_week_period,
                   current_week_date_range = excluded.current_week_date_range,
                   current_week_yield_kwh = excluded.current_week_yield_kwh,
                   current_week_consumption_kwh = excluded.current_week_consumption_kwh,
                   current_week_days = excluded.current_week_days,
                   current_week_updated_at = excluded.current_week_updated_at,
                   current_month_period = excluded.current_month_period,
                   current_month_yield_kwh = excluded.current_month_yield_kwh,
                   current_month_consumption_kwh = excluded.current_month_consumption_kwh,
                   current_month_avg_autarky = excluded.current_month_avg_autarky,
                   current_month_days = excluded.current_month_days,
                   current_month_updated_at = excluded.current_month_updated_at,
                   last_7_days_avg_yield_kwh = excluded.last_7_days_avg_yield_kwh,
                   last_7_days_avg_accuracy = excluded.last_7_days_avg_accuracy,
                   last_7_days_total_yield_kwh = excluded.last_7_days_total_yield_kwh,
                   last_7_days_calculated_at = excluded.last_7_days_calculated_at,
                   last_30_days_avg_yield_kwh = excluded.last_30_days_avg_yield_kwh,
                   last_30_days_avg_accuracy = excluded.last_30_days_avg_accuracy,
                   last_30_days_total_yield_kwh = excluded.last_30_days_total_yield_kwh,
                   last_30_days_calculated_at = excluded.last_30_days_calculated_at,
                   last_365_days_avg_yield_kwh = excluded.last_365_days_avg_yield_kwh,
                   last_365_days_total_yield_kwh = excluded.last_365_days_total_yield_kwh,
                   last_365_days_calculated_at = excluded.last_365_days_calculated_at,
                   last_updated = excluded.last_updated""",
            (
                data.get("all_time_peak", {}).get("power_w"),
                data.get("all_time_peak", {}).get("date"),
                data.get("all_time_peak", {}).get("at"),
                data.get("current_week", {}).get("period"),
                data.get("current_week", {}).get("date_range"),
                data.get("current_week", {}).get("yield_kwh"),
                data.get("current_week", {}).get("consumption_kwh"),
                data.get("current_week", {}).get("days"),
                data.get("current_week", {}).get("updated_at"),
                data.get("current_month", {}).get("period"),
                data.get("current_month", {}).get("yield_kwh"),
                data.get("current_month", {}).get("consumption_kwh"),
                data.get("current_month", {}).get("avg_autarky"),
                data.get("current_month", {}).get("days"),
                data.get("current_month", {}).get("updated_at"),
                data.get("last_7_days", {}).get("avg_yield_kwh"),
                data.get("last_7_days", {}).get("avg_accuracy"),
                data.get("last_7_days", {}).get("total_yield_kwh"),
                data.get("last_7_days", {}).get("calculated_at"),
                data.get("last_30_days", {}).get("avg_yield_kwh"),
                data.get("last_30_days", {}).get("avg_accuracy"),
                data.get("last_30_days", {}).get("total_yield_kwh"),
                data.get("last_30_days", {}).get("calculated_at"),
                data.get("last_365_days", {}).get("avg_yield_kwh"),
                data.get("last_365_days", {}).get("total_yield_kwh"),
                data.get("last_365_days", {}).get("calculated_at"),
                datetime.now()
            )
        )
        await self._db.commit()

    async def get_daily_statistics(self) -> Optional[Dict[str, Any]]:
        """Get daily statistics data."""
        row = await self.fetchone("SELECT * FROM daily_statistics WHERE id = 1")
        if not row:
            return None

        return {
            "all_time_peak": {
                "power_w": row["all_time_peak_power_w"],
                "date": str(row["all_time_peak_date"]) if row["all_time_peak_date"] else None,
                "at": str(row["all_time_peak_at"]) if row["all_time_peak_at"] else None
            } if row["all_time_peak_power_w"] is not None else {},
            "current_week": {
                "period": row["current_week_period"],
                "date_range": row["current_week_date_range"],
                "yield_kwh": row["current_week_yield_kwh"],
                "consumption_kwh": row["current_week_consumption_kwh"],
                "days": row["current_week_days"],
                "updated_at": str(row["current_week_updated_at"]) if row["current_week_updated_at"] else None
            } if row["current_week_period"] is not None else {},
            "current_month": {
                "period": row["current_month_period"],
                "yield_kwh": row["current_month_yield_kwh"],
                "consumption_kwh": row["current_month_consumption_kwh"],
                "avg_autarky": row["current_month_avg_autarky"],
                "days": row["current_month_days"],
                "updated_at": str(row["current_month_updated_at"]) if row["current_month_updated_at"] else None
            } if row["current_month_period"] is not None else {},
            "last_7_days": {
                "avg_yield_kwh": row["last_7_days_avg_yield_kwh"],
                "avg_accuracy": row["last_7_days_avg_accuracy"],
                "total_yield_kwh": row["last_7_days_total_yield_kwh"],
                "calculated_at": str(row["last_7_days_calculated_at"]) if row["last_7_days_calculated_at"] else None
            } if row["last_7_days_avg_yield_kwh"] is not None else {},
            "last_30_days": {
                "avg_yield_kwh": row["last_30_days_avg_yield_kwh"],
                "avg_accuracy": row["last_30_days_avg_accuracy"],
                "total_yield_kwh": row["last_30_days_total_yield_kwh"],
                "calculated_at": str(row["last_30_days_calculated_at"]) if row["last_30_days_calculated_at"] else None
            } if row["last_30_days_avg_yield_kwh"] is not None else {},
            "last_365_days": {
                "avg_yield_kwh": row["last_365_days_avg_yield_kwh"],
                "total_yield_kwh": row["last_365_days_total_yield_kwh"],
                "calculated_at": str(row["last_365_days_calculated_at"]) if row["last_365_days_calculated_at"] else None
            } if row["last_365_days_avg_yield_kwh"] is not None else {}
        }

    # ========================================================================
    # Forecast History Operations
    # ========================================================================

    async def save_forecast_history_entry(self, entry: Dict[str, Any]) -> None:
        """Save a single forecast history entry."""
        entry_date = datetime.fromisoformat(entry["date"]).date() if isinstance(entry["date"], str) else entry["date"]

        await self._db.execute(
            """INSERT INTO forecast_history
               (date, predicted_kwh, actual_kwh, consumption_kwh, autarky, accuracy,
                production_hours, peak_power, source, excluded_hours_count, excluded_hours_total,
                excluded_hours_ratio, excluded_hours_reasons, archived_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(date) DO UPDATE SET
                   predicted_kwh = excluded.predicted_kwh,
                   actual_kwh = excluded.actual_kwh,
                   consumption_kwh = excluded.consumption_kwh,
                   autarky = excluded.autarky,
                   accuracy = excluded.accuracy,
                   production_hours = excluded.production_hours,
                   peak_power = excluded.peak_power,
                   source = excluded.source,
                   excluded_hours_count = excluded.excluded_hours_count,
                   excluded_hours_total = excluded.excluded_hours_total,
                   excluded_hours_ratio = excluded.excluded_hours_ratio,
                   excluded_hours_reasons = excluded.excluded_hours_reasons,
                   archived_at = excluded.archived_at""",
            (
                entry_date,
                entry.get("predicted_kwh"),
                entry.get("actual_kwh"),
                entry.get("consumption_kwh"),
                entry.get("autarky"),
                entry.get("accuracy"),
                entry.get("production_hours"),
                entry.get("peak_power"),
                entry.get("source"),
                entry.get("excluded_hours", {}).get("count"),
                entry.get("excluded_hours", {}).get("total"),
                entry.get("excluded_hours", {}).get("ratio"),
                json.dumps(entry.get("excluded_hours", {}).get("reasons")),
                datetime.fromisoformat(entry["archived_at"]) if "archived_at" in entry else datetime.now()
            )
        )
        await self._db.commit()

    async def get_forecast_history(self, limit: int = 730) -> List[Dict[str, Any]]:
        """Get forecast history entries."""
        rows = await self.fetchall(
            f"SELECT * FROM forecast_history ORDER BY date DESC LIMIT {limit}"
        )

        return [
            {
                "date": str(row["date"]),
                "predicted_kwh": row["predicted_kwh"],
                "actual_kwh": row["actual_kwh"],
                "consumption_kwh": row["consumption_kwh"],
                "autarky": row["autarky"],
                "accuracy": row["accuracy"],
                "production_hours": row["production_hours"],
                "peak_power": row["peak_power"],
                "source": row["source"],
                "excluded_hours": {
                    "count": row["excluded_hours_count"],
                    "total": row["excluded_hours_total"],
                    "ratio": row["excluded_hours_ratio"],
                    "reasons": json.loads(row["excluded_hours_reasons"]) if row["excluded_hours_reasons"] else {}
                } if row["excluded_hours_count"] is not None else {},
                "archived_at": str(row["archived_at"]) if row["archived_at"] else None
            }
            for row in rows
        ]

    # ========================================================================
    # Shadow Detection Operations
    # ========================================================================

    async def save_hourly_shadow_detection(self, prediction_id: str, shadow_data: Dict[str, Any]) -> None:
        """Save shadow detection data for an hourly prediction."""
        await self._db.execute(
            """INSERT INTO hourly_shadow_detection
               (prediction_id, method, ensemble_mode, shadow_type, shadow_percent, confidence,
                root_cause, fusion_mode, efficiency_ratio, loss_kwh, theoretical_max_kwh,
                interpretation, theory_ratio_shadow_type, theory_ratio_shadow_percent,
                theory_ratio_confidence, theory_ratio_efficiency_ratio, theory_ratio_clear_sky_wm2,
                theory_ratio_actual_wm2, theory_ratio_loss_kwh, theory_ratio_root_cause,
                sensor_fusion_shadow_type, sensor_fusion_shadow_percent, sensor_fusion_confidence,
                sensor_fusion_efficiency_ratio, sensor_fusion_loss_kwh, sensor_fusion_root_cause,
                sensor_fusion_lux_factor, sensor_fusion_lux_shadow_percent,
                sensor_fusion_irradiance_factor, sensor_fusion_irradiance_shadow_percent,
                weight_theory_ratio, weight_sensor_fusion)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(prediction_id) DO UPDATE SET
                   method = excluded.method,
                   ensemble_mode = excluded.ensemble_mode,
                   shadow_type = excluded.shadow_type,
                   shadow_percent = excluded.shadow_percent,
                   confidence = excluded.confidence,
                   root_cause = excluded.root_cause,
                   fusion_mode = excluded.fusion_mode,
                   efficiency_ratio = excluded.efficiency_ratio,
                   loss_kwh = excluded.loss_kwh,
                   theoretical_max_kwh = excluded.theoretical_max_kwh,
                   interpretation = excluded.interpretation,
                   theory_ratio_shadow_type = excluded.theory_ratio_shadow_type,
                   theory_ratio_shadow_percent = excluded.theory_ratio_shadow_percent,
                   theory_ratio_confidence = excluded.theory_ratio_confidence,
                   theory_ratio_efficiency_ratio = excluded.theory_ratio_efficiency_ratio,
                   theory_ratio_clear_sky_wm2 = excluded.theory_ratio_clear_sky_wm2,
                   theory_ratio_actual_wm2 = excluded.theory_ratio_actual_wm2,
                   theory_ratio_loss_kwh = excluded.theory_ratio_loss_kwh,
                   theory_ratio_root_cause = excluded.theory_ratio_root_cause,
                   sensor_fusion_shadow_type = excluded.sensor_fusion_shadow_type,
                   sensor_fusion_shadow_percent = excluded.sensor_fusion_shadow_percent,
                   sensor_fusion_confidence = excluded.sensor_fusion_confidence,
                   sensor_fusion_efficiency_ratio = excluded.sensor_fusion_efficiency_ratio,
                   sensor_fusion_loss_kwh = excluded.sensor_fusion_loss_kwh,
                   sensor_fusion_root_cause = excluded.sensor_fusion_root_cause,
                   sensor_fusion_lux_factor = excluded.sensor_fusion_lux_factor,
                   sensor_fusion_lux_shadow_percent = excluded.sensor_fusion_lux_shadow_percent,
                   sensor_fusion_irradiance_factor = excluded.sensor_fusion_irradiance_factor,
                   sensor_fusion_irradiance_shadow_percent = excluded.sensor_fusion_irradiance_shadow_percent,
                   weight_theory_ratio = excluded.weight_theory_ratio,
                   weight_sensor_fusion = excluded.weight_sensor_fusion""",
            (
                prediction_id,
                shadow_data.get("method"),
                shadow_data.get("ensemble_mode"),
                shadow_data.get("shadow_type"),
                shadow_data.get("shadow_percent"),
                shadow_data.get("confidence"),
                shadow_data.get("root_cause"),
                shadow_data.get("fusion_mode"),
                shadow_data.get("efficiency_ratio"),
                shadow_data.get("loss_kwh"),
                shadow_data.get("theoretical_max_kwh"),
                shadow_data.get("interpretation"),
                shadow_data.get("methods", {}).get("theory_ratio", {}).get("shadow_type"),
                shadow_data.get("methods", {}).get("theory_ratio", {}).get("shadow_percent"),
                shadow_data.get("methods", {}).get("theory_ratio", {}).get("confidence"),
                shadow_data.get("methods", {}).get("theory_ratio", {}).get("efficiency_ratio"),
                shadow_data.get("methods", {}).get("theory_ratio", {}).get("clear_sky_wm2"),
                shadow_data.get("methods", {}).get("theory_ratio", {}).get("actual_wm2"),
                shadow_data.get("methods", {}).get("theory_ratio", {}).get("loss_kwh"),
                shadow_data.get("methods", {}).get("theory_ratio", {}).get("root_cause"),
                shadow_data.get("methods", {}).get("sensor_fusion", {}).get("shadow_type"),
                shadow_data.get("methods", {}).get("sensor_fusion", {}).get("shadow_percent"),
                shadow_data.get("methods", {}).get("sensor_fusion", {}).get("confidence"),
                shadow_data.get("methods", {}).get("sensor_fusion", {}).get("efficiency_ratio"),
                shadow_data.get("methods", {}).get("sensor_fusion", {}).get("loss_kwh"),
                shadow_data.get("methods", {}).get("sensor_fusion", {}).get("root_cause"),
                shadow_data.get("methods", {}).get("sensor_fusion", {}).get("lux_analysis", {}).get("factor"),
                shadow_data.get("methods", {}).get("sensor_fusion", {}).get("lux_analysis", {}).get("shadow_percent"),
                shadow_data.get("methods", {}).get("sensor_fusion", {}).get("irradiance_factor"),
                shadow_data.get("methods", {}).get("sensor_fusion", {}).get("irradiance_shadow_percent"),
                shadow_data.get("weights", {}).get("theory_ratio"),
                shadow_data.get("weights", {}).get("sensor_fusion")
            )
        )
        await self._db.commit()

    # ========================================================================
    # Additional Hourly Prediction Data Operations
    # ========================================================================

    async def save_hourly_production_metrics(self, prediction_id: str, metrics: Dict[str, Any]) -> None:
        """Save production metrics for an hourly prediction."""
        await self._db.execute(
            """INSERT INTO hourly_production_metrics (prediction_id, peak_power_today_kwh, production_hours_today, cumulative_today_kwh)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(prediction_id) DO UPDATE SET
                   peak_power_today_kwh = excluded.peak_power_today_kwh,
                   production_hours_today = excluded.production_hours_today,
                   cumulative_today_kwh = excluded.cumulative_today_kwh""",
            (prediction_id, metrics.get("peak_power_today_kwh"), metrics.get("production_hours_today"), metrics.get("cumulative_today_kwh"))
        )
        await self._db.commit()

    async def save_hourly_historical_context(self, prediction_id: str, context: Dict[str, Any]) -> None:
        """Save historical context for an hourly prediction."""
        await self._db.execute(
            """INSERT INTO hourly_historical_context (prediction_id, yesterday_same_hour, same_hour_avg_7days)
               VALUES (?, ?, ?)
               ON CONFLICT(prediction_id) DO UPDATE SET
                   yesterday_same_hour = excluded.yesterday_same_hour,
                   same_hour_avg_7days = excluded.same_hour_avg_7days""",
            (prediction_id, context.get("yesterday_same_hour"), context.get("same_hour_avg_7days"))
        )
        await self._db.commit()

    async def save_hourly_panel_group_accuracy(self, prediction_id: str, group_name: str, accuracy_data: Dict[str, Any]) -> None:
        """Save panel group accuracy data for an hourly prediction."""
        await self._db.execute(
            """INSERT INTO hourly_panel_group_accuracy
               (prediction_id, group_name, prediction_kwh, actual_kwh, error_kwh, error_percent, accuracy_percent)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(prediction_id, group_name) DO UPDATE SET
                   prediction_kwh = excluded.prediction_kwh,
                   actual_kwh = excluded.actual_kwh,
                   error_kwh = excluded.error_kwh,
                   error_percent = excluded.error_percent,
                   accuracy_percent = excluded.accuracy_percent""",
            (prediction_id, group_name, accuracy_data.get("prediction_kwh"), accuracy_data.get("actual_kwh"),
             accuracy_data.get("error_kwh"), accuracy_data.get("error_percent"), accuracy_data.get("accuracy_percent"))
        )
        await self._db.commit()

    # ========================================================================
    # Daily Summary Additional Data Operations
    # ========================================================================

    async def save_daily_patterns(self, date_val: date, patterns: List[Dict[str, Any]]) -> None:
        """Save patterns for a daily summary."""
        # Clear existing patterns for this date
        await self._db.execute("DELETE FROM daily_patterns WHERE date = ?", (date_val,))

        # Insert new patterns
        for pattern in patterns:
            await self._db.execute(
                """INSERT INTO daily_patterns
                   (date, pattern_type, hours, severity, avg_error_percent, confidence,
                    first_detected, occurrence_count, seasonal)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (date_val, pattern.get("type"), json.dumps(pattern.get("hours", [])),
                 pattern.get("severity"), pattern.get("avg_error_percent"), pattern.get("confidence"),
                 datetime.fromisoformat(pattern["first_detected"]) if "first_detected" in pattern else None,
                 pattern.get("occurrence_count"), pattern.get("seasonal", False))
            )
        await self._db.commit()

    async def save_daily_recommendations(self, date_val: date, recommendations: List[Dict[str, Any]]) -> None:
        """Save recommendations for a daily summary."""
        # Clear existing recommendations for this date
        await self._db.execute("DELETE FROM daily_recommendations WHERE date = ?", (date_val,))

        # Insert new recommendations
        for rec in recommendations:
            await self._db.execute(
                """INSERT INTO daily_recommendations
                   (date, recommendation_type, priority, action, hours, factor, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (date_val, rec.get("type"), rec.get("priority"), rec.get("action"),
                 json.dumps(rec.get("hours", [])), rec.get("factor"), rec.get("reason"))
            )
        await self._db.commit()

    # ========================================================================
    # Utility Methods
    # ========================================================================

    async def vacuum(self) -> None:
        """Vacuum the database to reclaim space."""
        await self._db.commit()
        await self._db.execute("VACUUM")
        _LOGGER.info("Database vacuumed")

    async def get_db_size(self) -> int:
        """Get database file size in bytes."""
        return Path(self.db_path).stat().st_size

    async def execute(self, sql: str, parameters: tuple = (), auto_commit: bool = True) -> None:
        """Execute a raw SQL statement.

        Args:
            sql: The SQL statement to execute
            parameters: Parameters for the SQL statement
            auto_commit: If True, automatically commit after execution (default: True)
        """
        if self._db is None:
            _LOGGER.debug("Database is closed, skipping execute")
            return
        await self._db.execute(sql, parameters)
        if auto_commit:
            await self._db.commit()

    async def commit(self) -> None:
        """Manually commit the current transaction."""
        if self._db is None:
            return
        await self._db.commit()

    async def executemany(self, sql: str, parameters_list: List[tuple]) -> int:
        """Execute a SQL statement with multiple parameter sets (batch insert).

        Returns the number of rows affected.
        """
        if self._db is None:
            _LOGGER.debug("Database is closed, skipping executemany")
            return 0
        if not parameters_list:
            return 0
        await self._db.executemany(sql, parameters_list)
        await self._db.commit()
        return len(parameters_list)

    async def fetchone(self, sql: str, parameters: tuple = ()) -> Optional[aiosqlite.Row]:
        """Execute a query and fetch one row."""
        if self._db is None:
            _LOGGER.debug("Database is closed, returning None from fetchone")
            return None
        async with self._db.execute(sql, parameters) as cursor:
            return await cursor.fetchone()

    async def fetchall(self, sql: str, parameters: tuple = ()) -> List[aiosqlite.Row]:
        """Execute a query and fetch all rows."""
        if self._db is None:
            _LOGGER.debug("Database is closed, returning empty list from fetchall")
            return []
        async with self._db.execute(sql, parameters) as cursor:
            return await cursor.fetchall()
