# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Schema Validator for Solar Forecast ML V16.0.0.
Validates and ensures database schema integrity on startup.
Replaces JSON file validation with database table validation.

@zara
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from .db_manager import DatabaseManager
from .data_io import DataManagerIO

_LOGGER = logging.getLogger(__name__)


# Required database tables with their essential columns @zara
REQUIRED_TABLES = {
    "ai_seasonal_factors": ["month", "factor", "sample_count"],
    "ai_feature_importance": ["feature_name", "importance", "category"],
    "ai_grid_search_results": ["success", "hidden_size", "accuracy"],
    "ai_dni_tracker": ["hour", "max_dni"],
    "ai_model_weights": ["weight_type", "weight_data"],
    "physics_learning_config": ["albedo", "system_efficiency"],
    "physics_calibration_groups": ["group_name", "global_factor"],
    "physics_calibration_hourly": ["group_name", "hour", "factor"],
    "physics_calibration_buckets": ["group_name", "bucket_name", "global_factor"],
    "physics_calibration_history": ["date", "group_name", "avg_ratio"],
    "weather_forecast": ["forecast_date", "hour", "temperature", "clouds"],
    "weather_expert_weights": ["cloud_type", "expert_name", "weight"],
    "weather_source_weights": ["source_name", "weight"],
    "hourly_predictions": ["prediction_id", "target_date", "prediction_kwh"],
    "daily_forecasts": ["forecast_type", "forecast_date", "prediction_kwh"],
    "daily_summaries": ["date", "predicted_total_kwh", "actual_total_kwh"],
    "astronomy_cache": ["cache_date", "hour", "sun_elevation_deg"],
    "coordinator_state": ["expected_daily_production", "last_set_date"],
    "production_time_state": ["date", "accumulated_hours"],
    "panel_group_sensor_state": ["group_name", "last_value"],
    "yield_cache": ["value", "time", "date"],
    "visibility_learning": ["visibility_threshold_m", "fog_visibility_threshold_m"],
}

# Default values for singleton config tables @zara
DEFAULT_PHYSICS_LEARNING_CONFIG = {
    "albedo": 0.2,
    "system_efficiency": 0.9,
    "learned_efficiency_factor": 1.0,
    "rolling_window_days": 21,
    "min_samples": 1,
}

DEFAULT_VISIBILITY_LEARNING = {
    "visibility_threshold_m": 10000,
    "fog_visibility_threshold_m": 1000,
    "samples_below_threshold": 0,
    "samples_above_threshold": 0,
}


class DataSchemaValidator(DataManagerIO):
    """Validates database schema and ensures required tables exist. @zara

    Replaces the old JSON file validation with database table validation.
    On startup, ensures all required tables exist and have proper structure.
    """

    def __init__(self, hass: HomeAssistant, db_manager: DatabaseManager):
        """Initialize the schema validator. @zara

        Args:
            hass: Home Assistant instance
            db_manager: DatabaseManager instance for DB operations
        """
        super().__init__(hass, db_manager)
        self.migration_log: List[str] = []
        self.healed_tables: List[str] = []
        _LOGGER.debug("DataSchemaValidator initialized with DatabaseManager")

    async def validate_and_migrate_all(self) -> bool:
        """Validate all database tables and ensure schema integrity. @zara

        This method:
        1. Verifies all required tables exist
        2. Checks essential columns are present
        3. Initializes singleton config tables with defaults if empty
        4. Logs all validation actions

        Returns:
            True if all validations pass, False otherwise
        """
        try:
            _LOGGER.info("=== Database Schema Validation Starting ===")

            success = True

            # Validate all required tables exist
            for table_name, required_columns in REQUIRED_TABLES.items():
                table_valid = await self._validate_table(table_name, required_columns)
                if not table_valid:
                    success = False
                    self._log(f"Table validation failed: {table_name}")

            # Initialize singleton config tables if needed
            await self._ensure_physics_learning_config()
            await self._ensure_visibility_learning()
            await self._ensure_coordinator_state()

            # Log summary
            if self.migration_log:
                _LOGGER.info("=== Schema Validation Summary ===")
                for entry in self.migration_log:
                    _LOGGER.info("  %s", entry)
            else:
                _LOGGER.info("All database tables valid - no changes needed")

            if self.healed_tables:
                _LOGGER.info(
                    "Initialized %d table(s): %s",
                    len(self.healed_tables),
                    ", ".join(self.healed_tables)
                )

            _LOGGER.info("=== Database Schema Validation Complete ===")
            return success

        except Exception as e:
            _LOGGER.error("Schema validation failed: %s", e, exc_info=True)
            return False

    def _log(self, message: str) -> None:
        """Log a validation action. @zara"""
        self.migration_log.append(message)
        _LOGGER.info("SCHEMA: %s", message)

    async def _validate_table(
        self,
        table_name: str,
        required_columns: List[str]
    ) -> bool:
        """Validate that a table exists and has required columns. @zara

        Args:
            table_name: Name of the table to validate
            required_columns: List of required column names

        Returns:
            True if table is valid, False otherwise
        """
        try:
            # Check if table exists
            row = await self.fetch_one(
                """SELECT name FROM sqlite_master
                   WHERE type='table' AND name=?""",
                (table_name,)
            )

            if not row:
                _LOGGER.warning("Table %s does not exist", table_name)
                return False

            # Check columns using PRAGMA
            columns = await self.fetch_all(
                f"PRAGMA table_info({table_name})"
            )

            existing_columns = {col[1] for col in columns}  # col[1] is column name

            # Check for missing required columns
            missing = set(required_columns) - existing_columns
            if missing:
                _LOGGER.warning(
                    "Table %s missing columns: %s",
                    table_name, ", ".join(missing)
                )
                return False

            return True

        except Exception as e:
            _LOGGER.error(
                "Failed to validate table %s: %s",
                table_name, e
            )
            return False

    async def _ensure_physics_learning_config(self) -> None:
        """Ensure physics_learning_config has default values. @zara"""
        try:
            row = await self.fetch_one(
                "SELECT id FROM physics_learning_config WHERE id = 1"
            )

            if not row:
                await self.execute_query(
                    """INSERT INTO physics_learning_config
                       (id, albedo, system_efficiency, learned_efficiency_factor,
                        rolling_window_days, min_samples, updated_at)
                       VALUES (1, ?, ?, ?, ?, ?, ?)""",
                    (
                        DEFAULT_PHYSICS_LEARNING_CONFIG["albedo"],
                        DEFAULT_PHYSICS_LEARNING_CONFIG["system_efficiency"],
                        DEFAULT_PHYSICS_LEARNING_CONFIG["learned_efficiency_factor"],
                        DEFAULT_PHYSICS_LEARNING_CONFIG["rolling_window_days"],
                        DEFAULT_PHYSICS_LEARNING_CONFIG["min_samples"],
                        datetime.now(),
                    )
                )
                self._log("Created default physics_learning_config")
                self.healed_tables.append("physics_learning_config")

        except Exception as e:
            _LOGGER.error("Failed to ensure physics_learning_config: %s", e)

    async def _ensure_visibility_learning(self) -> None:
        """Ensure visibility_learning has default values. @zara"""
        try:
            row = await self.fetch_one(
                "SELECT id FROM visibility_learning WHERE id = 1"
            )

            if not row:
                await self.execute_query(
                    """INSERT INTO visibility_learning
                       (id, visibility_threshold_m, fog_visibility_threshold_m,
                        samples_below_threshold, samples_above_threshold, last_updated)
                       VALUES (1, ?, ?, ?, ?, ?)""",
                    (
                        DEFAULT_VISIBILITY_LEARNING["visibility_threshold_m"],
                        DEFAULT_VISIBILITY_LEARNING["fog_visibility_threshold_m"],
                        DEFAULT_VISIBILITY_LEARNING["samples_below_threshold"],
                        DEFAULT_VISIBILITY_LEARNING["samples_above_threshold"],
                        datetime.now(),
                    )
                )
                self._log("Created default visibility_learning")
                self.healed_tables.append("visibility_learning")

        except Exception as e:
            _LOGGER.error("Failed to ensure visibility_learning: %s", e)

    async def _ensure_coordinator_state(self) -> None:
        """Ensure coordinator_state has default values. @zara"""
        try:
            row = await self.fetch_one(
                "SELECT id FROM coordinator_state WHERE id = 1"
            )

            if not row:
                await self.execute_query(
                    """INSERT INTO coordinator_state
                       (id, expected_daily_production, last_set_date, last_updated)
                       VALUES (1, 0, ?, ?)""",
                    (
                        datetime.now().date(),
                        datetime.now(),
                    )
                )
                self._log("Created default coordinator_state")
                self.healed_tables.append("coordinator_state")

        except Exception as e:
            _LOGGER.error("Failed to ensure coordinator_state: %s", e)

    async def validate_table_integrity(self, table_name: str) -> Dict[str, Any]:
        """Check integrity of a specific table. @zara

        Args:
            table_name: Name of table to check

        Returns:
            Dictionary with integrity check results
        """
        result = {
            "table": table_name,
            "exists": False,
            "row_count": 0,
            "columns": [],
            "indexes": [],
            "issues": [],
        }

        try:
            # Check table exists
            row = await self.fetch_one(
                """SELECT name FROM sqlite_master
                   WHERE type='table' AND name=?""",
                (table_name,)
            )

            if not row:
                result["issues"].append("Table does not exist")
                return result

            result["exists"] = True

            # Get row count
            count_row = await self.fetch_one(
                f"SELECT COUNT(*) FROM {table_name}"
            )
            result["row_count"] = count_row[0] if count_row else 0

            # Get column info
            columns = await self.fetch_all(
                f"PRAGMA table_info({table_name})"
            )
            result["columns"] = [
                {
                    "name": col[1],
                    "type": col[2],
                    "notnull": bool(col[3]),
                    "default": col[4],
                    "pk": bool(col[5]),
                }
                for col in columns
            ]

            # Get index info
            indexes = await self.fetch_all(
                f"PRAGMA index_list({table_name})"
            )
            result["indexes"] = [idx[1] for idx in indexes]

            return result

        except Exception as e:
            _LOGGER.error(
                "Failed to check integrity of %s: %s",
                table_name, e
            )
            result["issues"].append(f"Error: {str(e)}")
            return result

    async def get_schema_report(self) -> Dict[str, Any]:
        """Generate a comprehensive schema report. @zara

        Returns:
            Dictionary with schema validation report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "tables_checked": 0,
            "tables_valid": 0,
            "tables_missing": [],
            "table_details": {},
        }

        for table_name in REQUIRED_TABLES.keys():
            report["tables_checked"] += 1

            integrity = await self.validate_table_integrity(table_name)
            report["table_details"][table_name] = integrity

            if integrity["exists"]:
                report["tables_valid"] += 1
            else:
                report["tables_missing"].append(table_name)

        return report

    async def check_foreign_keys(self) -> Dict[str, Any]:
        """Check foreign key integrity. @zara

        Returns:
            Dictionary with foreign key check results
        """
        try:
            # Enable foreign key checks
            await self.execute_query("PRAGMA foreign_keys = ON")

            # Run foreign key check
            violations = await self.fetch_all("PRAGMA foreign_key_check")

            return {
                "valid": len(violations) == 0,
                "violations_count": len(violations),
                "violations": [
                    {
                        "table": v[0],
                        "rowid": v[1],
                        "parent_table": v[2],
                        "fk_index": v[3],
                    }
                    for v in violations
                ],
            }

        except Exception as e:
            _LOGGER.error("Failed to check foreign keys: %s", e)
            return {
                "valid": False,
                "error": str(e),
                "violations_count": -1,
            }

    async def vacuum_database(self) -> bool:
        """Vacuum database to reclaim space. @zara

        Returns:
            True if successful, False otherwise
        """
        try:
            await self.db.vacuum()
            self._log("Database vacuumed successfully")
            return True
        except Exception as e:
            _LOGGER.error("Failed to vacuum database: %s", e)
            return False

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics. @zara

        Returns:
            Dictionary with database statistics
        """
        try:
            stats = await self.get_db_stats()

            # Count rows in main tables
            table_counts = {}
            for table_name in [
                "hourly_predictions",
                "daily_summaries",
                "weather_forecast",
                "astronomy_cache",
            ]:
                row = await self.fetch_one(
                    f"SELECT COUNT(*) FROM {table_name}"
                )
                table_counts[table_name] = row[0] if row else 0

            stats["table_row_counts"] = table_counts
            return stats

        except Exception as e:
            _LOGGER.error("Failed to get database stats: %s", e)
            return {
                "error": str(e),
                "connected": False,
            }
