# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from homeassistant.core import HomeAssistant

from .data_schemas import get_schema, SCHEMA_FILE_PATHS

_LOGGER = logging.getLogger(__name__)


class DataSchemaValidator:
    """Validates and creates JSON files using centralized schemas from data_schemas.py.

    This validator uses the Single Source of Truth pattern:
    - All schemas are defined in data_schemas.py
    - Uses get_schema() to get current schemas
    - Self-healing adds missing fields based on the central schema

    Directories (30 JSON files total):
    - / (root): .migrations_completed.json
    - ai/: dni_tracker.json, grid_search_results.json, learned_weights.json, seasonal.json
    - data/: bright_sky_cache.json, coordinator_state.json, open_meteo_cache.json,
             pirate_weather_cache.json, production_time_state.json,
             weather_expert_weights.json, weather_source_weights.json, wttr_in_cache.json
    - physics/: calibration_history.json, learning_config.json
    - stats/: astronomy_cache.json, daily_forecasts.json, daily_summaries.json,
              forecast_drift_log.json, hourly_predictions.json, hourly_weather_actual.json,
              multi_day_hourly_forecast.json, panel_group_sensor_state.json,
              panel_group_today_cache.json, retrospective_forecast.json,
              weather_expert_learning.json, weather_forecast_corrected.json,
              weather_precision_daily.json, weather_source_learning.json, yield_cache.json
    """

    def __init__(self, hass: HomeAssistant, data_dir: Path):
        """Initialize the schema validator."""
        self.hass = hass
        self.data_dir = data_dir
        self.migration_log = []
        self.healed_files = []

    async def validate_and_migrate_all(self) -> bool:
        """Validate and create all JSON files on startup.

        Uses centralized schemas from data_schemas.py.
        Iterates over SCHEMA_FILE_PATHS to validate all files.
        """
        try:
            _LOGGER.info("=== JSON Schema Validation Starting ===")

            await self._ensure_directories()

            success = True

            # Validate all files using centralized schemas
            for schema_name, relative_path in SCHEMA_FILE_PATHS.items():
                # Special handling for learned_weights (AI model architecture)
                if schema_name == "learned_weights":
                    success &= await self._validate_learned_weights()
                    continue

                file_path = self.data_dir / relative_path
                file_name = relative_path.split("/")[-1]
                schema = get_schema(schema_name)

                success &= await self._validate_and_heal(file_path, schema, file_name)

            if self.migration_log:
                _LOGGER.info("=== Schema Validation Summary ===")
                for entry in self.migration_log:
                    _LOGGER.info(f"  {entry}")
            else:
                _LOGGER.info("All JSON files valid - no changes needed")

            if self.healed_files:
                _LOGGER.info(f"Self-healed {len(self.healed_files)} file(s): {', '.join(self.healed_files)}")

            _LOGGER.info("=== JSON Schema Validation Complete ===")
            return success

        except Exception as e:
            _LOGGER.error(f"Schema validation failed: {e}", exc_info=True)
            return False

    def _log(self, message: str) -> None:
        """Log a validation action."""
        self.migration_log.append(message)
        _LOGGER.info(f"SCHEMA: {message}")

    def _ensure_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
        file_name: str,
        path: str = ""
    ) -> tuple[Dict[str, Any], bool]:
        """Recursively ensure data matches schema, adding missing fields.

        Self-healing logic:
        - If a key exists in schema but not in data -> add it with default value
        - If both exist and are dicts -> recurse to check nested structure
        - Never overwrite existing values
        - Returns (healed_data, was_healed)

        Args:
            data: The existing data to heal
            schema: The expected schema with default values
            file_name: Name of the file (for logging)
            path: Current path in the nested structure (for logging)

        Returns:
            Tuple of (healed_data, was_healed_flag)
        """
        healed = False

        for key, default_value in schema.items():
            current_path = f"{path}.{key}" if path else key

            if key not in data:
                # Key missing - add it with default value
                data[key] = default_value
                healed = True
                _LOGGER.debug(f"Self-heal {file_name}: added missing '{current_path}'")

            elif isinstance(default_value, dict) and isinstance(data[key], dict):
                # Both are dicts - recurse to check nested structure
                data[key], nested_healed = self._ensure_schema(
                    data[key], default_value, file_name, current_path
                )
                if nested_healed:
                    healed = True

        return data, healed

    async def _validate_and_heal(
        self,
        file_path: Path,
        schema: Dict[str, Any],
        file_name: str
    ) -> bool:
        """Validate file exists and heal missing fields.

        Args:
            file_path: Path to the JSON file
            schema: Expected schema with default values
            file_name: Display name for logging

        Returns:
            True if validation/healing succeeded
        """
        data = await self._read_json(file_path)

        if data is None:
            # File doesn't exist - create with full schema
            self._log(f"Creating {file_name}")
            return await self._write_json(file_path, schema)

        # File exists - check and heal missing fields
        healed_data, was_healed = self._ensure_schema(data, schema, file_name)

        if was_healed:
            self._log(f"Self-healed {file_name} (added missing fields)")
            self.healed_files.append(file_name)
            return await self._write_json(file_path, healed_data)

        return True

    async def _ensure_directories(self) -> None:
        """Ensure all directories exist."""
        dirs = [
            self.data_dir / "ai",
            self.data_dir / "data",
            self.data_dir / "stats",
            self.data_dir / "physics",
            self.data_dir / "logs",
            self.data_dir / "backups" / "auto",
        ]
        for d in dirs:
            if not d.exists():
                await self.hass.async_add_executor_job(d.mkdir, True, True)

    async def _read_json(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Read JSON file."""
        try:
            import json
            import aiofiles

            if not file_path.exists():
                return None

            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            _LOGGER.warning(f"Failed to read {file_path.name}: {e}")
            return None

    async def _write_json(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Write JSON file atomically."""
        try:
            import json
            import aiofiles

            file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = file_path.with_suffix(".tmp")

            async with aiofiles.open(temp_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))

            await self.hass.async_add_executor_job(temp_file.replace, file_path)
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to write {file_path.name}: {e}")
            return False

    # =========================================================================
    # SPECIAL CASE: LEARNED_WEIGHTS
    # =========================================================================

    async def _validate_learned_weights(self) -> bool:
        """Validate ai/learned_weights.json.

        IMPORTANT: Do NOT create a default file here!
        The AI model architecture (input_size, num_outputs) depends on the
        panel_groups configuration which is not available in this validator.

        If the file doesn't exist, the AI predictor will:
        1. Initialize with the correct architecture based on panel_groups
        2. Set state to UNTRAINED
        3. Train when sufficient data is available

        Creating a default file with wrong input_size/num_outputs causes
        "Model architecture mismatch" errors on startup.
        """
        file_path = self.data_dir / "ai" / "learned_weights.json"
        data = await self._read_json(file_path)

        if data is None:
            # Do NOT create - let AIPredictor handle this
            self._log("learned_weights.json missing (AI will train when ready)")
            return True

        # Validate existing file has required structure
        # If corrupted, delete it so AIPredictor creates fresh
        required_keys = ["hidden_size"]
        if not all(key in data for key in required_keys):
            self._log("learned_weights.json corrupted, removing for fresh training")
            await self.hass.async_add_executor_job(file_path.unlink)
            return True

        return True
