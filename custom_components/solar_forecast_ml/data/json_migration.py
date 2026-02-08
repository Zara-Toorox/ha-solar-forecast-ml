# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import asyncio
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from homeassistant.core import HomeAssistant

from .db_manager import DatabaseManager

_LOGGER = logging.getLogger(__name__)


def _sql_value(val: Any) -> Union[None, str, int, float, bool]:
    """Convert a value to a SQLite-safe type (no BLOBs).

    Prevents lists, dicts, and other complex objects from being stored as BLOBs.
    """
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (datetime, date)):
        return val.isoformat() if hasattr(val, 'isoformat') else str(val)
    if isinstance(val, (list, dict)):
        return json.dumps(val)
    return str(val)

MIGRATION_FLAG = ".storage/solar_forecast_ml_json_migrated_v3"

JSON_FILE_MAPPING = {
    "ai/seasonal.json": "ai_seasonal_factors",
    "ai/feature_importance.json": "ai_feature_importance",
    "ai/grid_search_results.json": "ai_grid_search_results",
    "ai/dni_tracker.json": "ai_dni_tracker",
    "ai/learned_weights.json": "ai_learned_weights_meta",
    "physics/calibration_history.json": "physics_calibration_history",
    "physics/learning_config.json": "physics_learning_config",
    "data/coordinator_state.json": "coordinator_state",
    "data/production_time_state.json": "production_time_state",
    "data/weather_expert_weights.json": "weather_expert_weights",
    "data/weather_source_weights.json": "weather_source_weights",
    "data/wttr_in_cache.json": "weather_cache_wttr_in",
    "data/bright_sky_cache.json": "weather_cache_bright_sky",
    "data/pirate_weather_cache.json": "weather_cache_pirate_weather",
    "data/open_meteo_cache.json": "weather_cache_open_meteo",
    "stats/hourly_predictions.json": "hourly_predictions",
    "stats/daily_forecasts.json": "daily_forecasts",
    "stats/daily_summaries.json": "daily_summaries",
    "stats/weather_forecast_corrected.json": "weather_forecast",
    "stats/weather_integration_ml.json": "weather_forecast",
    "stats/astronomy_cache.json": "astronomy_cache",
    "stats/weather_expert_learning.json": "weather_expert_learning",
    "stats/weather_source_learning.json": "weather_source_learning",
    "stats/panel_group_sensor_state.json": "panel_group_sensor_state",
    "stats/yield_cache.json": "yield_cache",
    "stats/visibility_learning.json": "visibility_learning",
    "stats/forecast_drift_log.json": "forecast_drift_log",
    "stats/hourly_weather_actual.json": "hourly_weather_actual",
    "stats/weather_precision_daily.json": "weather_precision_daily",
    "stats/panel_group_today_cache.json": "panel_group_daily_cache",
    "stats/multi_day_hourly_forecast.json": "multi_day_hourly_forecast",
    "stats/retrospective_forecast.json": "retrospective_forecast",
}


class JsonMigrationStats:
    def __init__(self):
        self.imported = 0
        self.updated = 0
        self.skipped = 0
        self.errors = 0
        self.error_files: List[str] = []

    def __str__(self) -> str:
        return (
            f"Imported: {self.imported}, Updated: {self.updated}, "
            f"Skipped: {self.skipped}, Errors: {self.errors}"
        )


class JsonMigrator:
    def __init__(self, hass: HomeAssistant, db_manager: DatabaseManager):
        self.hass = hass
        self.db = db_manager
        self.base_path = Path(hass.config.path("solar_forecast_ml"))
        self.stats = JsonMigrationStats()

    async def should_migrate(self) -> bool:
        flag_path = Path(self.hass.config.path(MIGRATION_FLAG))
        return not flag_path.exists()

    async def dry_run(self) -> Dict[str, Any]:
        result = {
            "flag_exists": not await self.should_migrate(),
            "base_path": str(self.base_path),
            "base_path_exists": self.base_path.exists(),
            "files_found": [],
            "files_missing": [],
            "file_details": {},
            "total_entries": 0,
            "validation_errors": [],
        }

        for relative_path, table_name in JSON_FILE_MAPPING.items():
            full_path = self.base_path / relative_path
            if full_path.exists():
                result["files_found"].append(relative_path)
                try:
                    data = await self._load_json(full_path)
                    if data is None:
                        result["validation_errors"].append(f"{relative_path}: Invalid JSON")
                        continue

                    entry_count = self._count_entries(relative_path, data)
                    result["file_details"][relative_path] = {
                        "table": table_name,
                        "entries": entry_count,
                        "size_bytes": full_path.stat().st_size,
                    }
                    result["total_entries"] += entry_count
                except Exception as e:
                    result["validation_errors"].append(f"{relative_path}: {e}")
            else:
                result["files_missing"].append(relative_path)

        return result

    def _count_entries(self, file_path: str, data: Dict[str, Any]) -> int:
        if isinstance(data, list):
            return len(data)

        count_keys = {
            "ai/seasonal.json": lambda d: len(d.get("factors", {})),
            "ai/feature_importance.json": lambda d: len(d.get("feature_importance", {})),
            "ai/grid_search_results.json": lambda d: len(d.get("all_results", [])),
            "ai/dni_tracker.json": lambda d: len(d.get("max_dni", {})),
            "ai/learned_weights.json": lambda d: 2,
            "physics/calibration_history.json": lambda d: sum(len(e.get("groups", {})) for e in d.get("history", [])),
            "physics/learning_config.json": lambda d: 1 + len(d.get("group_calibration", {})),
            "data/coordinator_state.json": lambda d: 1,
            "data/production_time_state.json": lambda d: 1,
            "data/weather_expert_weights.json": lambda d: sum(len(e) for e in d.get("weights", {}).values()),
            "data/weather_source_weights.json": lambda d: len(d.get("weights", {})),
            "data/wttr_in_cache.json": lambda d: sum(len(h) for h in d.get("forecast", {}).values() if isinstance(h, dict)),
            "data/bright_sky_cache.json": lambda d: sum(len(h) for h in d.get("forecast", {}).values() if isinstance(h, dict)),
            "data/pirate_weather_cache.json": lambda d: sum(len(h) for h in d.get("forecast", {}).values() if isinstance(h, dict)),
            "data/open_meteo_cache.json": lambda d: sum(len(h) for h in d.get("forecast", {}).values() if isinstance(h, dict)),
            "stats/hourly_predictions.json": lambda d: len(d.get("predictions", [])),
            "stats/daily_forecasts.json": lambda d: len(d.get("history", [])) + 3,
            "stats/daily_summaries.json": lambda d: len(d.get("summaries", [])),
            "stats/astronomy_cache.json": lambda d: sum(len(day.get("hourly", {})) for day in d.get("days", {}).values()),
            "stats/weather_forecast_corrected.json": lambda d: sum(len(h) for h in d.get("forecast", {}).values() if isinstance(h, dict)),
            "stats/weather_integration_ml.json": lambda d: sum(len(h) for h in d.get("forecast", {}).values() if isinstance(h, dict)),
            "stats/weather_expert_learning.json": lambda d: sum(len(day.get("mae_by_expert", {})) for day in d.get("daily_history", {}).values()),
            "stats/weather_source_learning.json": lambda d: sum(1 for day in d.get("daily_history", {}).values() for k in day if k.startswith("mae_")),
            "stats/hourly_weather_actual.json": lambda d: sum(len(h) for h in d.get("hourly_data", {}).values() if isinstance(h, dict)),
            "stats/yield_cache.json": lambda d: 1,
            "stats/visibility_learning.json": lambda d: 1,
            "stats/retrospective_forecast.json": lambda d: 1 + len(d.get("hourly_predictions", [])),
            "stats/forecast_drift_log.json": lambda d: len(d.get("entries", [])),
            "stats/weather_precision_daily.json": lambda d: sum(len(day.get("hourly_comparisons", [])) for day in d.get("daily_tracking", {}).values()),
            "stats/multi_day_hourly_forecast.json": lambda d: sum(len(day.get("hourly", [])) for day in d.get("days", {}).values()),
            "stats/panel_group_today_cache.json": lambda d: len(d.get("groups", {})),
        }

        counter = count_keys.get(file_path)
        if counter:
            try:
                return counter(data)
            except:
                return 0

        if isinstance(data, dict):
            for key in ["entries", "hourly", "predictions", "forecasts", "summaries", "groups"]:
                if key in data:
                    val = data[key]
                    return len(val) if isinstance(val, (list, dict)) else 0
        return 1

    async def run_migration(self) -> JsonMigrationStats:
        if not await self.should_migrate():
            _LOGGER.info("JSON migration already completed - skipping")
            return self.stats

        _LOGGER.info("=" * 60)
        _LOGGER.info("Starting JSON to SQLite migration...")
        _LOGGER.info("=" * 60)

        json_files = await self._find_json_files()
        if not json_files:
            _LOGGER.info("No JSON files found for migration")
            await self._set_migration_flag()
            return self.stats

        total_files = len(json_files)
        _LOGGER.info(f"Found {total_files} JSON files to migrate")

        for idx, json_path in enumerate(json_files, 1):
            relative_path = str(json_path.relative_to(self.base_path))
            _LOGGER.info(f"[{idx}/{total_files}] Migrating {relative_path}...")
            await self._migrate_file(json_path)
            _LOGGER.info(f"[{idx}/{total_files}] Completed {relative_path} - Running total: {self.stats}")

        await self._set_migration_flag()

        _LOGGER.info("=" * 60)
        _LOGGER.info(f"JSON MIGRATION COMPLETED SUCCESSFULLY")
        _LOGGER.info(f"Final stats: {self.stats}")
        _LOGGER.info("=" * 60)
        if self.stats.error_files:
            _LOGGER.warning(f"Failed files: {', '.join(self.stats.error_files)}")

        return self.stats

    async def _find_json_files(self) -> List[Path]:
        def _scan_files():
            files = []
            for relative_path in JSON_FILE_MAPPING.keys():
                full_path = self.base_path / relative_path
                if full_path.exists():
                    files.append(full_path)
            return files

        return await self.hass.async_add_executor_job(_scan_files)

    async def _migrate_file(self, json_path: Path) -> None:
        relative_path = str(json_path.relative_to(self.base_path))
        table_name = JSON_FILE_MAPPING.get(relative_path)

        if not table_name:
            self.stats.skipped += 1
            return

        try:
            data = await self._load_json(json_path)
            if data is None:
                self.stats.errors += 1
                self.stats.error_files.append(relative_path)
                return

            await self._migrate_data(relative_path, table_name, data)

        except Exception as e:
            _LOGGER.error(f"Migration failed for {relative_path}: {e}")
            self.stats.errors += 1
            self.stats.error_files.append(relative_path)

    async def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        def _read():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                _LOGGER.error(f"Invalid JSON in {path}: {e}")
                return None
            except Exception as e:
                _LOGGER.error(f"Failed to read {path}: {e}")
                return None

        return await self.hass.async_add_executor_job(_read)

    async def _migrate_data(self, file_path: str, table_name: str, data: Dict[str, Any]) -> None:
        migrator_map = {
            "ai/seasonal.json": self._migrate_seasonal,
            "ai/feature_importance.json": self._migrate_feature_importance,
            "ai/grid_search_results.json": self._migrate_grid_search,
            "ai/dni_tracker.json": self._migrate_dni_tracker,
            "ai/learned_weights.json": self._migrate_learned_weights,
            "physics/calibration_history.json": self._migrate_calibration_history,
            "physics/learning_config.json": self._migrate_learning_config,
            "data/coordinator_state.json": self._migrate_coordinator_state,
            "data/production_time_state.json": self._migrate_production_time_state,
            "data/weather_expert_weights.json": self._migrate_weather_expert_weights,
            "data/weather_source_weights.json": self._migrate_weather_source_weights,
            "data/wttr_in_cache.json": self._migrate_wttr_cache,
            "data/bright_sky_cache.json": self._migrate_bright_sky_cache,
            "data/pirate_weather_cache.json": self._migrate_pirate_weather_cache,
            "data/open_meteo_cache.json": self._migrate_open_meteo_cache,
            "stats/hourly_predictions.json": self._migrate_hourly_predictions,
            "stats/daily_forecasts.json": self._migrate_daily_forecasts,
            "stats/daily_summaries.json": self._migrate_daily_summaries,
            "stats/weather_forecast_corrected.json": self._migrate_weather_forecast,
            "stats/weather_integration_ml.json": self._migrate_weather_forecast,
            "stats/astronomy_cache.json": self._migrate_astronomy_cache,
            "stats/weather_expert_learning.json": self._migrate_weather_expert_learning,
            "stats/weather_source_learning.json": self._migrate_weather_source_learning,
            "stats/panel_group_sensor_state.json": self._migrate_panel_group_sensor_state,
            "stats/yield_cache.json": self._migrate_yield_cache,
            "stats/visibility_learning.json": self._migrate_visibility_learning,
            "stats/forecast_drift_log.json": self._migrate_forecast_drift_log,
            "stats/hourly_weather_actual.json": self._migrate_hourly_weather_actual,
            "stats/weather_precision_daily.json": self._migrate_weather_precision_daily,
            "stats/panel_group_today_cache.json": self._migrate_panel_group_daily_cache,
            "stats/multi_day_hourly_forecast.json": self._migrate_multi_day_forecast,
            "stats/retrospective_forecast.json": self._migrate_retrospective_forecast,
        }

        migrator = migrator_map.get(file_path)
        if migrator:
            await migrator(data)
        else:
            self.stats.skipped += 1

    def _format_hardware_info(self, hw: Optional[Dict[str, Any]]) -> Optional[str]:
        if not hw or not isinstance(hw, dict):
            return None
        parts = []
        if hw.get("machine_type"):
            parts.append(hw["machine_type"])
        if hw.get("architecture"):
            parts.append(hw["architecture"])
        if hw.get("cpu_count"):
            parts.append(f"{hw['cpu_count']} CPUs")
        if hw.get("is_container"):
            parts.append("container")
        if hw.get("is_raspberry_pi"):
            parts.append("RPi")
        return " ".join(parts) if parts else None

    async def _migrate_seasonal(self, data: Dict[str, Any]) -> None:
        factors = data.get("factors", {})
        sample_counts = data.get("sample_counts", {})
        version = data.get("version", "1.0")

        for month_str, factor in factors.items():
            try:
                month = int(month_str)
                sample_count = sample_counts.get(month_str, 0)

                existing = await self.db.fetchone(
                    "SELECT id FROM ai_seasonal_factors WHERE month = ?", (month,)
                )

                if existing:
                    await self.db.execute(
                        "UPDATE ai_seasonal_factors SET factor = ?, sample_count = ?, version = ?, updated_at = ? WHERE month = ?",
                        (factor, sample_count, version, datetime.now(), month)
                    )
                    self.stats.updated += 1
                else:
                    await self.db.execute(
                        "INSERT INTO ai_seasonal_factors (month, factor, sample_count, version, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (month, factor, sample_count, version, datetime.now())
                    )
                    self.stats.imported += 1
            except Exception as e:
                _LOGGER.debug(f"Skip seasonal month {month_str}: {e}")
                self.stats.skipped += 1

    async def _migrate_feature_importance(self, data: Dict[str, Any]) -> None:
        timestamp = data.get("timestamp")
        if not timestamp:
            self.stats.skipped += 1
            return

        try:
            ts = datetime.fromisoformat(timestamp)
        except:
            self.stats.skipped += 1
            return

        features = data.get("feature_importance", {})
        helpful = data.get("helpful_features", [])
        harmful = data.get("harmful_features", [])

        for feature_name, importance in features.items():
            if feature_name in helpful:
                category = "helpful"
            elif feature_name in harmful:
                category = "harmful"
            else:
                category = "neutral"

            try:
                existing = await self.db.fetchone(
                    "SELECT id FROM ai_feature_importance WHERE feature_name = ? AND timestamp = ?",
                    (feature_name, ts)
                )

                if existing:
                    await self.db.execute(
                        "UPDATE ai_feature_importance SET importance = ?, category = ? WHERE feature_name = ? AND timestamp = ?",
                        (importance, category, feature_name, ts)
                    )
                    self.stats.updated += 1
                else:
                    await self.db.execute(
                        "INSERT INTO ai_feature_importance (feature_name, importance, category, baseline_rmse, num_samples, analysis_time_seconds, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (feature_name, importance, category, data.get("baseline_rmse"), data.get("num_samples"), data.get("analysis_time_seconds"), ts)
                    )
                    self.stats.imported += 1
            except Exception as e:
                _LOGGER.debug(f"Skip feature {feature_name}: {e}")
                self.stats.skipped += 1

    async def _migrate_grid_search(self, data: Dict[str, Any]) -> None:
        results = data.get("all_results", [])
        timestamp = data.get("timestamp")
        best_params = data.get("best_params", {})

        if not timestamp:
            self.stats.skipped += 1
            return

        try:
            ts = datetime.fromisoformat(timestamp)
        except:
            self.stats.skipped += 1
            return

        for res in results:
            params = res.get("params", {})
            is_best = params == best_params

            try:
                await self.db.execute(
                    "INSERT OR IGNORE INTO ai_grid_search_results (success, hidden_size, batch_size, learning_rate, accuracy, epochs_trained, final_val_loss, duration_seconds, is_best_result, hardware_info, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (data.get("success", False), params.get("hidden_size"), params.get("batch_size"), params.get("learning_rate"), res.get("accuracy"), res.get("epochs_trained"), res.get("final_val_loss"), res.get("duration_seconds"), is_best, self._format_hardware_info(data.get("hardware_info")), ts)
                )
                self.stats.imported += 1
            except Exception as e:
                _LOGGER.debug(f"Skip grid search result: {e}")
                self.stats.skipped += 1

    async def _migrate_dni_tracker(self, data: Dict[str, Any]) -> None:
        max_dni = data.get("max_dni", {})
        history = data.get("history", {})
        last_updated = data.get("last_updated")
        version = data.get("version", "1.0")

        if not last_updated:
            self.stats.skipped += 1
            return

        try:
            lu_date = datetime.fromisoformat(last_updated).date()
        except:
            self.stats.skipped += 1
            return

        for hour_str, dni_value in max_dni.items():
            try:
                hour = int(hour_str)
                if hour < 6 or hour > 20:
                    continue

                existing = await self.db.fetchone(
                    "SELECT id FROM ai_dni_tracker WHERE hour = ?", (hour,)
                )

                if existing:
                    await self.db.execute(
                        "UPDATE ai_dni_tracker SET max_dni = ?, version = ?, last_updated = ? WHERE hour = ?",
                        (dni_value, version, lu_date, hour)
                    )
                    self.stats.updated += 1
                else:
                    await self.db.execute(
                        "INSERT INTO ai_dni_tracker (hour, max_dni, version, last_updated) VALUES (?, ?, ?, ?)",
                        (hour, dni_value, version, lu_date)
                    )
                    self.stats.imported += 1

                if hour_str in history:
                    for hist_value in history[hour_str]:
                        await self.db.execute(
                            "INSERT OR IGNORE INTO ai_dni_history (hour, dni_value) VALUES (?, ?)",
                            (hour, hist_value)
                        )
            except Exception as e:
                _LOGGER.debug(f"Skip DNI hour {hour_str}: {e}")
                self.stats.skipped += 1

    async def _migrate_learned_weights(self, data: Dict[str, Any]) -> None:
        try:
            existing = await self.db.fetchone("SELECT id FROM ai_learned_weights_meta WHERE id = 1")

            meta_values = (
                data.get("version", "2.0"),
                data.get("active_model", "tiny_lstm"),
                data.get("training_samples"),
                data.get("last_trained"),
                data.get("accuracy"),
                data.get("rmse"),
                datetime.now()
            )

            if existing:
                await self.db.execute(
                    "UPDATE ai_learned_weights_meta SET version = ?, active_model = ?, training_samples = ?, last_trained = ?, accuracy = ?, rmse = ?, updated_at = ? WHERE id = 1",
                    meta_values
                )
                self.stats.updated += 1
            else:
                await self.db.execute(
                    "INSERT INTO ai_learned_weights_meta (id, version, active_model, training_samples, last_trained, accuracy, rmse, updated_at) VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
                    meta_values
                )
                self.stats.imported += 1

            # Ridge data
            ridge_data = data.get("ridge", {})
            if ridge_data:
                # Ridge weights
                ridge_weights = ridge_data.get("weights", [])
                if ridge_weights and isinstance(ridge_weights, list):
                    for row_idx, row in enumerate(ridge_weights):
                        if isinstance(row, list):
                            for col_idx, val in enumerate(row):
                                if val is not None and val != 0:
                                    await self.db.execute(
                                        "INSERT OR REPLACE INTO ai_ridge_weights (row_index, col_index, weight_value) VALUES (?, ?, ?)",
                                        (row_idx, col_idx, val)
                                    )
                                    self.stats.imported += 1

                # Ridge metadata
                ridge_meta_existing = await self.db.fetchone("SELECT id FROM ai_ridge_meta WHERE id = 1")
                ridge_meta_values = (
                    ridge_data.get("model_type", "TinyRidge"),
                    ridge_data.get("alpha"),
                    ridge_data.get("input_size"),
                    ridge_data.get("hidden_size"),
                    ridge_data.get("sequence_length"),
                    ridge_data.get("num_outputs"),
                    ridge_data.get("flat_size"),
                    ridge_data.get("trained_samples"),
                    ridge_data.get("loo_cv_score"),
                    ridge_data.get("accuracy"),
                    ridge_data.get("rmse"),
                    datetime.now()
                )

                if ridge_meta_existing:
                    await self.db.execute(
                        "UPDATE ai_ridge_meta SET model_type = ?, alpha = ?, input_size = ?, hidden_size = ?, sequence_length = ?, num_outputs = ?, flat_size = ?, trained_samples = ?, loo_cv_score = ?, accuracy = ?, rmse = ?, updated_at = ? WHERE id = 1",
                        ridge_meta_values
                    )
                    self.stats.updated += 1
                else:
                    await self.db.execute(
                        "INSERT INTO ai_ridge_meta (id, model_type, alpha, input_size, hidden_size, sequence_length, num_outputs, flat_size, trained_samples, loo_cv_score, accuracy, rmse, updated_at) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        ridge_meta_values
                    )
                    self.stats.imported += 1

                # Ridge normalization (feature_means and feature_stds)
                feature_means = ridge_data.get("feature_means", [])
                feature_stds = ridge_data.get("feature_stds", [])
                if feature_means and feature_stds:
                    for idx, (mean, std) in enumerate(zip(feature_means, feature_stds)):
                        if mean is not None and std is not None:
                            await self.db.execute(
                                "INSERT OR REPLACE INTO ai_ridge_normalization (feature_index, feature_mean, feature_std) VALUES (?, ?, ?)",
                                (idx, mean, std)
                            )
                            self.stats.imported += 1

            # LSTM data
            lstm_data = data.get("lstm", {})
            if lstm_data:
                # LSTM weights (including attention weights)
                lstm_weight_types = ['Wf', 'Wi', 'Wc', 'Wo', 'bf', 'bi', 'bc', 'bo', 'Wy', 'by', 'W_query', 'W_key', 'W_value', 'W_attn_out', 'b_attn_out']
                for wt in lstm_weight_types:
                    wt_values = lstm_data.get(wt, [])
                    if isinstance(wt_values, list):
                        flat_idx = 0
                        for row in wt_values:
                            if isinstance(row, list):
                                for val in row:
                                    if isinstance(val, (int, float)):
                                        await self.db.execute(
                                            "INSERT OR REPLACE INTO ai_lstm_weights (weight_type, weight_index, weight_value) VALUES (?, ?, ?)",
                                            (wt, flat_idx, val)
                                        )
                                        self.stats.imported += 1
                                        flat_idx += 1
                            elif isinstance(row, (int, float)):
                                await self.db.execute(
                                    "INSERT OR REPLACE INTO ai_lstm_weights (weight_type, weight_index, weight_value) VALUES (?, ?, ?)",
                                    (wt, flat_idx, row)
                                )
                                self.stats.imported += 1
                                flat_idx += 1

                # LSTM metadata
                lstm_meta_existing = await self.db.fetchone("SELECT id FROM ai_lstm_meta WHERE id = 1")
                lstm_meta_values = (
                    lstm_data.get("input_size"),
                    lstm_data.get("hidden_size"),
                    lstm_data.get("sequence_length"),
                    lstm_data.get("num_outputs"),
                    lstm_data.get("has_attention", False),
                    lstm_data.get("training_samples") or data.get("training_samples"),
                    lstm_data.get("accuracy") or data.get("accuracy"),
                    lstm_data.get("rmse") or data.get("rmse"),
                    datetime.now()
                )

                if lstm_meta_existing:
                    await self.db.execute(
                        "UPDATE ai_lstm_meta SET input_size = ?, hidden_size = ?, sequence_length = ?, num_outputs = ?, has_attention = ?, training_samples = ?, accuracy = ?, rmse = ?, updated_at = ? WHERE id = 1",
                        lstm_meta_values
                    )
                    self.stats.updated += 1
                else:
                    await self.db.execute(
                        "INSERT INTO ai_lstm_meta (id, input_size, hidden_size, sequence_length, num_outputs, has_attention, training_samples, accuracy, rmse, updated_at) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        lstm_meta_values
                    )
                    self.stats.imported += 1

        except Exception as e:
            _LOGGER.debug(f"Skip learned weights: {e}")
            self.stats.skipped += 1

    async def _migrate_calibration_history(self, data: Dict[str, Any]) -> None:
        entries = data if isinstance(data, list) else data.get("history", data.get("entries", []))

        for entry in entries:
            entry_date = entry.get("date")
            if not entry_date:
                continue

            try:
                date_obj = datetime.fromisoformat(entry_date).date()
            except:
                continue

            groups = entry.get("groups", {})
            for group_name, group_data in groups.items():
                hourly = group_data.get("hourly", {})
                sample_count = group_data.get("sample_count", 0)

                # Gruppen-Level hourly (bucket_name = NULL)
                for hour_str, avg_ratio in hourly.items():
                    try:
                        hour = int(hour_str)
                        existing = await self.db.fetchone(
                            "SELECT id FROM physics_calibration_history WHERE date = ? AND group_name = ? AND hour = ? AND bucket_name IS NULL",
                            (date_obj, group_name, hour)
                        )

                        if existing:
                            await self.db.execute(
                                "UPDATE physics_calibration_history SET avg_ratio = ?, sample_count = ? WHERE id = ?",
                                (avg_ratio, sample_count, existing[0])
                            )
                            self.stats.updated += 1
                        else:
                            await self.db.execute(
                                "INSERT INTO physics_calibration_history (date, group_name, bucket_name, hour, avg_ratio, sample_count, source) VALUES (?, ?, NULL, ?, ?, ?, ?)",
                                (date_obj, group_name, hour, avg_ratio, sample_count, entry.get("source"))
                            )
                            self.stats.imported += 1
                    except Exception as e:
                        _LOGGER.debug(f"Skip calibration entry: {e}")
                        self.stats.skipped += 1

                # Bucket-Level hourly (mit bucket_name)
                buckets = group_data.get("buckets", {})
                for bucket_name, bucket_data in buckets.items():
                    bucket_hourly = bucket_data.get("hourly", {})
                    bucket_sample_count = bucket_data.get("sample_count", 0)

                    for hour_str, avg_ratio in bucket_hourly.items():
                        try:
                            hour = int(hour_str)
                            existing = await self.db.fetchone(
                                "SELECT id FROM physics_calibration_history WHERE date = ? AND group_name = ? AND hour = ? AND bucket_name = ?",
                                (date_obj, group_name, hour, bucket_name)
                            )

                            if existing:
                                await self.db.execute(
                                    "UPDATE physics_calibration_history SET avg_ratio = ?, sample_count = ? WHERE id = ?",
                                    (avg_ratio, bucket_sample_count, existing[0])
                                )
                                self.stats.updated += 1
                            else:
                                await self.db.execute(
                                    "INSERT INTO physics_calibration_history (date, group_name, bucket_name, hour, avg_ratio, sample_count, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (date_obj, group_name, bucket_name, hour, avg_ratio, bucket_sample_count, entry.get("source"))
                                )
                                self.stats.imported += 1
                        except Exception as e:
                            _LOGGER.debug(f"Skip bucket calibration entry: {e}")
                            self.stats.skipped += 1

    async def _migrate_learning_config(self, data: Dict[str, Any]) -> None:
        try:
            defaults = data.get("physics_defaults", {})
            metadata = data.get("metadata", {})
            updated_at = data.get("updated_at")

            if updated_at:
                try:
                    updated_at = datetime.fromisoformat(updated_at)
                except:
                    updated_at = datetime.now()
            else:
                updated_at = datetime.now()

            existing = await self.db.fetchone("SELECT id FROM physics_learning_config WHERE id = 1")

            if existing:
                await self.db.execute(
                    "UPDATE physics_learning_config SET version = ?, albedo = ?, system_efficiency = ?, learned_efficiency_factor = ?, rolling_window_days = ?, min_samples = ?, updated_at = ? WHERE id = 1",
                    (data.get("version", "3.0"), defaults.get("albedo", 0.2), defaults.get("system_efficiency", 0.9), defaults.get("learned_efficiency_factor", 1.0), metadata.get("rolling_window_days", 21), metadata.get("min_samples", 1), updated_at)
                )
                self.stats.updated += 1
            else:
                await self.db.execute(
                    "INSERT INTO physics_learning_config (id, version, albedo, system_efficiency, learned_efficiency_factor, rolling_window_days, min_samples, updated_at) VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
                    (data.get("version", "3.0"), defaults.get("albedo", 0.2), defaults.get("system_efficiency", 0.9), defaults.get("learned_efficiency_factor", 1.0), metadata.get("rolling_window_days", 21), metadata.get("min_samples", 1), updated_at)
                )
                self.stats.imported += 1

            for group_name, group_data in data.get("group_calibration", {}).items():
                await self._migrate_group_calibration(group_name, group_data)
        except Exception as e:
            _LOGGER.debug(f"Skip learning config: {e}")
            self.stats.skipped += 1

    async def _migrate_group_calibration(self, group_name: str, group_data: Dict[str, Any]) -> None:
        try:
            last_updated = group_data.get("last_updated")
            if last_updated:
                try:
                    last_updated = datetime.fromisoformat(last_updated)
                except:
                    last_updated = datetime.now()
            else:
                last_updated = datetime.now()

            existing = await self.db.fetchone(
                "SELECT id FROM physics_calibration_groups WHERE group_name = ?", (group_name,)
            )

            if existing:
                await self.db.execute(
                    "UPDATE physics_calibration_groups SET global_factor = ?, sample_count = ?, confidence = ?, last_updated = ? WHERE group_name = ?",
                    (group_data.get("global_factor", 1.0), group_data.get("sample_count", 0), group_data.get("confidence", 0.0), last_updated, group_name)
                )
                self.stats.updated += 1
            else:
                await self.db.execute(
                    "INSERT INTO physics_calibration_groups (group_name, global_factor, sample_count, confidence, last_updated) VALUES (?, ?, ?, ?, ?)",
                    (group_name, group_data.get("global_factor", 1.0), group_data.get("sample_count", 0), group_data.get("confidence", 0.0), last_updated)
                )
                self.stats.imported += 1

            for hour_str, factor in group_data.get("hourly_factors", {}).items():
                try:
                    hour = int(hour_str)
                    await self.db.execute(
                        "INSERT OR REPLACE INTO physics_calibration_hourly (group_name, hour, factor) VALUES (?, ?, ?)",
                        (group_name, hour, factor)
                    )
                except:
                    pass

            for bucket_name, bucket_data in group_data.get("bucket_factors", {}).items():
                await self.db.execute(
                    "INSERT OR REPLACE INTO physics_calibration_buckets (group_name, bucket_name, global_factor, sample_count, confidence) VALUES (?, ?, ?, ?, ?)",
                    (group_name, bucket_name, bucket_data.get("global_factor", 1.0), bucket_data.get("sample_count", 0), bucket_data.get("confidence", 0.0))
                )

                for hour_str, factor in bucket_data.get("hourly_factors", {}).items():
                    try:
                        hour = int(hour_str)
                        await self.db.execute(
                            "INSERT OR REPLACE INTO physics_calibration_bucket_hourly (group_name, bucket_name, hour, factor) VALUES (?, ?, ?, ?)",
                            (group_name, bucket_name, hour, factor)
                        )
                    except:
                        pass
        except Exception as e:
            _LOGGER.debug(f"Skip group calibration {group_name}: {e}")

    async def _migrate_coordinator_state(self, data: Dict[str, Any]) -> None:
        try:
            last_set_date = data.get("last_set_date")
            last_updated = data.get("last_updated")

            if last_set_date:
                try:
                    last_set_date = datetime.fromisoformat(last_set_date).date()
                except:
                    last_set_date = None

            if last_updated:
                try:
                    last_updated = datetime.fromisoformat(last_updated)
                except:
                    last_updated = datetime.now()
            else:
                last_updated = datetime.now()

            existing = await self.db.fetchone("SELECT id FROM coordinator_state WHERE id = 1")

            if existing:
                await self.db.execute(
                    "UPDATE coordinator_state SET expected_daily_production = ?, last_set_date = ?, last_updated = ? WHERE id = 1",
                    (data.get("expected_daily_production"), last_set_date, last_updated)
                )
                self.stats.updated += 1
            else:
                await self.db.execute(
                    "INSERT INTO coordinator_state (id, expected_daily_production, last_set_date, last_updated) VALUES (1, ?, ?, ?)",
                    (data.get("expected_daily_production"), last_set_date, last_updated)
                )
                self.stats.imported += 1
        except Exception as e:
            _LOGGER.debug(f"Skip coordinator state: {e}")
            self.stats.skipped += 1

    async def _migrate_production_time_state(self, data: Dict[str, Any]) -> None:
        try:
            date_val = data.get("date")
            if date_val:
                try:
                    date_val = datetime.fromisoformat(date_val).date()
                except:
                    date_val = datetime.now().date()
            else:
                date_val = datetime.now().date()

            last_updated = data.get("last_updated")
            if last_updated:
                try:
                    last_updated = datetime.fromisoformat(last_updated)
                except:
                    last_updated = datetime.now()
            else:
                last_updated = datetime.now()

            existing = await self.db.fetchone("SELECT id FROM production_time_state WHERE id = 1")

            if existing:
                await self.db.execute(
                    "UPDATE production_time_state SET date = ?, accumulated_hours = ?, is_active = ?, start_time = ?, production_time_today = ?, last_updated = ? WHERE id = 1",
                    (date_val, data.get("accumulated_hours", 0), data.get("is_active", False), data.get("start_time"), data.get("production_time_today"), last_updated)
                )
                self.stats.updated += 1
            else:
                await self.db.execute(
                    "INSERT INTO production_time_state (id, date, accumulated_hours, is_active, start_time, production_time_today, last_updated) VALUES (1, ?, ?, ?, ?, ?, ?)",
                    (date_val, data.get("accumulated_hours", 0), data.get("is_active", False), data.get("start_time"), data.get("production_time_today"), last_updated)
                )
                self.stats.imported += 1
        except Exception as e:
            _LOGGER.debug(f"Skip production time state: {e}")
            self.stats.skipped += 1

    async def _migrate_weather_expert_weights(self, data: Dict[str, Any]) -> None:
        weights = data.get("weights", {})
        metadata = data.get("metadata", {})
        last_updated = metadata.get("last_updated")

        if last_updated:
            try:
                last_updated = datetime.fromisoformat(last_updated)
            except:
                last_updated = datetime.now()
        else:
            last_updated = datetime.now()

        for cloud_type, experts in weights.items():
            for expert_name, weight in experts.items():
                try:
                    await self.db.execute(
                        "INSERT OR REPLACE INTO weather_expert_weights (cloud_type, expert_name, weight, last_updated) VALUES (?, ?, ?, ?)",
                        (cloud_type, expert_name, weight, last_updated)
                    )
                    self.stats.imported += 1
                except Exception as e:
                    _LOGGER.debug(f"Skip weather expert weight: {e}")
                    self.stats.skipped += 1

        # Snow prediction stats
        snow_stats = data.get("snow_prediction_stats", {})
        if snow_stats:
            try:
                snow_last_updated = snow_stats.get("last_updated")
                if snow_last_updated:
                    try:
                        snow_last_updated = datetime.fromisoformat(snow_last_updated)
                    except:
                        snow_last_updated = None

                existing = await self.db.fetchone("SELECT id FROM weather_expert_snow_stats WHERE id = 1")
                if existing:
                    await self.db.execute(
                        "UPDATE weather_expert_snow_stats SET total_predictions = ?, correct_predictions = ?, accuracy = ?, last_updated = ? WHERE id = 1",
                        (snow_stats.get("total_predictions", 0), snow_stats.get("correct_predictions", 0), snow_stats.get("accuracy", 0.0), snow_last_updated)
                    )
                    self.stats.updated += 1
                else:
                    await self.db.execute(
                        "INSERT INTO weather_expert_snow_stats (id, total_predictions, correct_predictions, accuracy, last_updated) VALUES (1, ?, ?, ?, ?)",
                        (snow_stats.get("total_predictions", 0), snow_stats.get("correct_predictions", 0), snow_stats.get("accuracy", 0.0), snow_last_updated)
                    )
                    self.stats.imported += 1
            except Exception as e:
                _LOGGER.debug(f"Skip snow prediction stats: {e}")

    async def _migrate_weather_source_weights(self, data: Dict[str, Any]) -> None:
        weights = data.get("weights", {})
        learning_metadata = data.get("learning_metadata", {})
        last_mae = learning_metadata.get("last_mae", {})
        last_updated = learning_metadata.get("last_updated")

        if last_updated:
            try:
                last_updated = datetime.fromisoformat(last_updated)
            except:
                last_updated = datetime.now()
        else:
            last_updated = datetime.now()

        # Parse last_learning_date
        last_learning_date = learning_metadata.get("last_learning_date")
        if last_learning_date:
            try:
                last_learning_date = datetime.fromisoformat(last_learning_date).date()
            except:
                last_learning_date = None

        for source_name, weight in weights.items():
            try:
                await self.db.execute(
                    """INSERT OR REPLACE INTO weather_source_weights
                    (source_name, weight, last_mae, version, last_learning_date, comparison_hours,
                     smoothing_factor_used, smoothing_factor_default, accelerated_learning, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (source_name, weight, last_mae.get(source_name), data.get("version", "1.1"),
                     last_learning_date, learning_metadata.get("comparison_hours"),
                     learning_metadata.get("smoothing_factor_used"), learning_metadata.get("smoothing_factor_default", 0.3),
                     learning_metadata.get("accelerated_learning", False), last_updated)
                )
                self.stats.imported += 1
            except Exception as e:
                _LOGGER.debug(f"Skip weather source weight: {e}")
                self.stats.skipped += 1

    async def _migrate_wttr_cache(self, data: Dict[str, Any]) -> None:
        await self._migrate_weather_cache_generic(data, "weather_cache_wttr_in", ["cloud_cover", "temperature", "humidity", "wind_speed", "precipitation", "pressure", "source"])

    async def _migrate_bright_sky_cache(self, data: Dict[str, Any]) -> None:
        await self._migrate_weather_cache_generic(data, "weather_cache_bright_sky", ["cloud_cover"])

    async def _migrate_pirate_weather_cache(self, data: Dict[str, Any]) -> None:
        await self._migrate_weather_cache_generic(data, "weather_cache_pirate_weather", ["cloud_cover"])

    async def _migrate_open_meteo_cache(self, data: Dict[str, Any]) -> None:
        await self._migrate_weather_cache_generic(data, "weather_cache_open_meteo", ["temperature", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "humidity", "wind_speed", "precipitation", "pressure", "direct_radiation", "diffuse_radiation", "ghi", "global_tilted_irradiance", "visibility_m", "source"])

    async def _migrate_weather_cache_generic(self, data: Dict[str, Any], table_name: str, fields: List[str]) -> None:
        forecast = data.get("forecast", {})
        metadata = data.get("metadata", {})
        fetched_at = metadata.get("fetched_at") or metadata.get("last_updated")
        if fetched_at and isinstance(fetched_at, str):
            try:
                fetched_at = datetime.fromisoformat(fetched_at)
            except:
                fetched_at = None

        records = []
        if isinstance(forecast, dict):
            for date_str, hours_data in forecast.items():
                try:
                    forecast_date = datetime.fromisoformat(date_str).date()
                except:
                    self.stats.skipped += 1
                    continue

                if not isinstance(hours_data, dict):
                    continue

                for hour_str, hour_data in hours_data.items():
                    try:
                        hour = int(hour_str)

                        if isinstance(hour_data, dict):
                            field_values = [_sql_value(hour_data.get(f)) for f in fields]
                        elif isinstance(hour_data, (int, float)) and len(fields) == 1:
                            field_values = [_sql_value(hour_data)]
                        else:
                            continue

                        records.append((forecast_date, hour, *field_values, _sql_value(fetched_at)))
                    except Exception as e:
                        _LOGGER.debug(f"Skip weather cache entry: {e}")
                        self.stats.skipped += 1

        if records:
            try:
                placeholders = ", ".join(["?"] * (len(fields) + 3))
                field_names = ", ".join(["forecast_date", "hour"] + fields + ["fetched_at"])
                count = await self.db.executemany(
                    f"INSERT OR REPLACE INTO {table_name} ({field_names}) VALUES ({placeholders})",
                    records
                )
                self.stats.imported += count
            except Exception as e:
                _LOGGER.error(f"Batch insert {table_name} failed: {e}")
                self.stats.errors += 1

    async def _migrate_hourly_predictions(self, data: Dict[str, Any]) -> None:
        predictions = data if isinstance(data, list) else data.get("predictions", [])
        total_predictions = len(predictions)

        if total_predictions > 0:
            _LOGGER.info(f"   → Migrating {total_predictions} hourly predictions...")

        # Use manual transaction for better performance
        batch_counter = 0
        processed_counter = 0
        batch_size = 100  # Commit every 100 predictions

        # Create a wrapper for execute with auto_commit disabled by default
        async def execute_no_commit(sql, params):
            await self.db.execute(sql, params, auto_commit=False)

        for pred in predictions:
            prediction_id = pred.get("id") or pred.get("prediction_id")
            if not prediction_id:
                continue

            try:
                existing = await self.db.fetchone(
                    "SELECT id FROM hourly_predictions WHERE prediction_id = ?", (prediction_id,)
                )

                # Check if main prediction exists, but continue to migrate child tables
                main_prediction_exists = existing is not None

                target_datetime = pred.get("target_datetime")
                if target_datetime and isinstance(target_datetime, str):
                    target_datetime = datetime.fromisoformat(target_datetime)

                target_date = pred.get("target_date")
                if target_date and isinstance(target_date, str):
                    target_date = datetime.fromisoformat(target_date).date()

                prediction_created_at = pred.get("prediction_created_at")
                if prediction_created_at and isinstance(prediction_created_at, str):
                    prediction_created_at = datetime.fromisoformat(prediction_created_at)

                # Parse actual_measured_at
                actual_measured_at = pred.get("actual_measured_at")
                if actual_measured_at and isinstance(actual_measured_at, str):
                    actual_measured_at = datetime.fromisoformat(actual_measured_at)

                # Extract flags and quality
                flags = pred.get("flags", {})
                quality = pred.get("quality", {})

                # Only insert main prediction if it doesn't exist yet
                if not main_prediction_exists:
                    await execute_no_commit(
                        """INSERT INTO hourly_predictions (prediction_id, prediction_created_at, prediction_created_hour, target_datetime, target_date, target_hour, target_day_of_week, target_day_of_year, target_month, target_season, prediction_kwh, prediction_kwh_uncapped, prediction_method, ml_contribution_percent, model_version, confidence, actual_kwh, actual_measured_at, accuracy_percent, error_kwh, error_percent, is_production_hour, is_peak_hour, is_outlier, has_weather_alert, has_sensor_data, sensor_data_complete, weather_forecast_updated, manual_override, inverter_clipped, has_panel_group_predictions, prediction_confidence, weather_forecast_age_hours, sensor_data_quality, data_completeness_percent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (prediction_id, prediction_created_at, pred.get("prediction_created_hour"), target_datetime, target_date, pred.get("target_hour"), pred.get("target_day_of_week"), pred.get("target_day_of_year"), pred.get("target_month"), pred.get("target_season"), pred.get("prediction_kwh", 0), pred.get("prediction_kwh_uncapped"), pred.get("prediction_method"), pred.get("ml_contribution_percent"), pred.get("model_version"), pred.get("confidence"), pred.get("actual_kwh"), actual_measured_at, pred.get("accuracy_percent"), pred.get("error_kwh"), pred.get("error_percent"), flags.get("is_production_hour", False), flags.get("is_peak_hour", False), flags.get("is_outlier", False), flags.get("has_weather_alert", False), flags.get("has_sensor_data", False), flags.get("sensor_data_complete", False), flags.get("weather_forecast_updated", False), flags.get("manual_override", False), flags.get("inverter_clipped", False), flags.get("has_panel_group_predictions", False), quality.get("prediction_confidence"), quality.get("weather_forecast_age_hours"), quality.get("sensor_data_quality"), quality.get("data_completeness_percent")),
                    )
                    self.stats.imported += 1
                else:
                    # V16.0.4: Update actual values from JSON into existing records @zara
                    json_actual = pred.get("actual_kwh")
                    if json_actual is not None:
                        await execute_no_commit(
                            """UPDATE hourly_predictions SET
                               actual_kwh = ?,
                               actual_measured_at = ?,
                               accuracy_percent = ?,
                               error_kwh = ?,
                               error_percent = ?
                            WHERE prediction_id = ?
                              AND (actual_kwh IS NULL OR actual_kwh = 0)""",
                            (json_actual, actual_measured_at,
                             pred.get("accuracy_percent"),
                             pred.get("error_kwh"),
                             pred.get("error_percent"),
                             prediction_id)
                        )
                        self.stats.updated += 1
                    else:
                        self.stats.skipped += 1

                # Always migrate child tables (using INSERT OR REPLACE for idempotency)

                sensor = pred.get("sensor_actual", {})
                if sensor and isinstance(sensor, dict):
                    try:
                        await execute_no_commit(
                            "INSERT OR REPLACE INTO prediction_sensor_actual (prediction_id, temperature_c, humidity_percent, solar_radiation_wm2, rain_mm, uv_index, wind_speed_ms, current_yield_kwh, lux) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (prediction_id, sensor.get("temperature_c"), sensor.get("humidity_percent"), sensor.get("solar_radiation_wm2"), sensor.get("rain_mm"), sensor.get("uv_index"), sensor.get("wind_speed_ms"), sensor.get("current_yield_kwh"), sensor.get("lux"))
                        )
                        self.stats.imported += 1
                    except:
                        pass

                # Migrate weather data (forecast, corrected, actual) to prediction_weather
                for weather_type, weather_key in [("forecast", "weather_forecast"), ("corrected", "weather_corrected"), ("actual", "weather_actual")]:
                    weather_data = pred.get(weather_key)
                    if weather_data and isinstance(weather_data, dict):
                        try:
                            # weather_actual uses different field names than forecast/corrected
                            if weather_type == "actual":
                                temp = weather_data.get("temperature_c")
                                wind = weather_data.get("wind_speed_ms")
                                humidity = weather_data.get("humidity_percent")
                                rain = weather_data.get("precipitation_mm")
                                clouds = weather_data.get("cloud_cover_percent")
                                pressure = weather_data.get("pressure_hpa")
                                lux = weather_data.get("lux")
                                frost_detected = weather_data.get("frost_detected")
                                frost_score = weather_data.get("frost_score")
                                frost_confidence = weather_data.get("frost_confidence")
                            else:
                                temp = weather_data.get("temperature")
                                wind = weather_data.get("wind")
                                humidity = weather_data.get("humidity")
                                rain = weather_data.get("rain")
                                clouds = weather_data.get("clouds")
                                pressure = weather_data.get("pressure")
                                lux = None
                                frost_detected = None
                                frost_score = None
                                frost_confidence = None
                            await execute_no_commit(
                                "INSERT OR REPLACE INTO prediction_weather (prediction_id, weather_type, temperature, solar_radiation_wm2, wind, humidity, rain, clouds, pressure, source, lux, frost_detected, frost_score, frost_confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (prediction_id, weather_type, _sql_value(temp), _sql_value(weather_data.get("solar_radiation_wm2")), _sql_value(wind), _sql_value(humidity), _sql_value(rain), _sql_value(clouds), _sql_value(pressure), _sql_value(weather_data.get("source")), _sql_value(lux), _sql_value(frost_detected), _sql_value(frost_score), _sql_value(frost_confidence))
                            )
                            self.stats.imported += 1
                        except:
                            pass

                # Migrate astronomy data to prediction_astronomy
                astronomy = pred.get("astronomy")
                if astronomy and isinstance(astronomy, dict):
                    try:
                        await execute_no_commit(
                            "INSERT OR REPLACE INTO prediction_astronomy (prediction_id, sunrise, sunset, solar_noon, daylight_hours, sun_elevation_deg, sun_azimuth_deg, clear_sky_radiation_wm2, theoretical_max_kwh, hours_since_solar_noon, day_progress_ratio, hours_after_sunrise, hours_before_sunset) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (prediction_id, _sql_value(astronomy.get("sunrise")), _sql_value(astronomy.get("sunset")), _sql_value(astronomy.get("solar_noon")), _sql_value(astronomy.get("daylight_hours")), _sql_value(astronomy.get("sun_elevation_deg")), _sql_value(astronomy.get("sun_azimuth_deg")), _sql_value(astronomy.get("clear_sky_radiation_wm2")), _sql_value(astronomy.get("theoretical_max_kwh")), _sql_value(astronomy.get("hours_since_solar_noon")), _sql_value(astronomy.get("day_progress_ratio")), _sql_value(astronomy.get("hours_after_sunrise")), _sql_value(astronomy.get("hours_before_sunset")))
                        )
                        self.stats.imported += 1
                    except:
                        pass

                # panel_group_predictions is a Dict: {"Gruppe 1": 0.0243, "Gruppe 2": 0.0133}
                panel_predictions = pred.get("panel_group_predictions") or {}
                panel_actuals = pred.get("panel_group_actuals") or {}

                # Collect all group names from both predictions and actuals
                all_groups = set(panel_predictions.keys()) | set(panel_actuals.keys())

                for group_name in all_groups:
                    pred_kwh = panel_predictions.get(group_name)
                    actual_kwh = panel_actuals.get(group_name)
                    if pred_kwh is not None or actual_kwh is not None:
                        try:
                            await execute_no_commit(
                                "INSERT OR REPLACE INTO prediction_panel_groups (prediction_id, group_name, prediction_kwh, actual_kwh) VALUES (?, ?, ?, ?)",
                                (prediction_id, group_name, pred_kwh or 0, actual_kwh)
                            )
                            self.stats.imported += 1
                        except:
                            pass

                # Migrate shadow_detection to hourly_shadow_detection
                shadow = pred.get("shadow_detection")
                if shadow and isinstance(shadow, dict):
                    try:
                        methods = shadow.get("methods", {})
                        theory_ratio = methods.get("theory_ratio", {})
                        sensor_fusion = methods.get("sensor_fusion", {})
                        weights = shadow.get("weights", {})

                        await execute_no_commit(
                            """INSERT OR REPLACE INTO hourly_shadow_detection (
                                prediction_id, method, ensemble_mode, shadow_type, shadow_percent,
                                confidence, root_cause, fusion_mode, efficiency_ratio, loss_kwh,
                                theoretical_max_kwh, interpretation,
                                theory_ratio_shadow_type, theory_ratio_shadow_percent, theory_ratio_confidence,
                                theory_ratio_efficiency_ratio, theory_ratio_clear_sky_wm2, theory_ratio_actual_wm2,
                                theory_ratio_loss_kwh, theory_ratio_root_cause,
                                sensor_fusion_shadow_type, sensor_fusion_shadow_percent, sensor_fusion_confidence,
                                sensor_fusion_efficiency_ratio, sensor_fusion_loss_kwh, sensor_fusion_root_cause,
                                sensor_fusion_lux_factor, sensor_fusion_lux_shadow_percent,
                                sensor_fusion_irradiance_factor, sensor_fusion_irradiance_shadow_percent,
                                weight_theory_ratio, weight_sensor_fusion
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                prediction_id,
                                _sql_value(shadow.get("method")),
                                _sql_value(shadow.get("ensemble_mode")),
                                _sql_value(shadow.get("shadow_type")),
                                _sql_value(shadow.get("shadow_percent")),
                                _sql_value(shadow.get("confidence")),
                                _sql_value(shadow.get("root_cause")),
                                _sql_value(shadow.get("fusion_mode")),
                                _sql_value(shadow.get("efficiency_ratio")),
                                _sql_value(shadow.get("loss_kwh")),
                                _sql_value(shadow.get("theoretical_max_kwh")),
                                _sql_value(shadow.get("interpretation")),
                                _sql_value(theory_ratio.get("shadow_type")),
                                _sql_value(theory_ratio.get("shadow_percent")),
                                _sql_value(theory_ratio.get("confidence")),
                                _sql_value(theory_ratio.get("efficiency_ratio")),
                                _sql_value(theory_ratio.get("clear_sky_wm2")),
                                _sql_value(theory_ratio.get("actual_wm2")),
                                _sql_value(theory_ratio.get("loss_kwh")),
                                _sql_value(theory_ratio.get("root_cause")),
                                _sql_value(sensor_fusion.get("shadow_type")),
                                _sql_value(sensor_fusion.get("shadow_percent")),
                                _sql_value(sensor_fusion.get("confidence")),
                                _sql_value(sensor_fusion.get("efficiency_ratio")),
                                _sql_value(sensor_fusion.get("loss_kwh")),
                                _sql_value(sensor_fusion.get("root_cause")),
                                _sql_value(sensor_fusion.get("lux_factor")),
                                _sql_value(sensor_fusion.get("lux_shadow_percent")),
                                _sql_value(sensor_fusion.get("irradiance_factor")),
                                _sql_value(sensor_fusion.get("irradiance_shadow_percent")),
                                _sql_value(weights.get("theory_ratio")),
                                _sql_value(weights.get("sensor_fusion"))
                            )
                        )
                        self.stats.imported += 1
                    except Exception as e:
                        _LOGGER.debug(f"Skip shadow_detection: {e}")

                # Migrate production_metrics to hourly_production_metrics
                prod_metrics = pred.get("production_metrics")
                if prod_metrics and isinstance(prod_metrics, dict):
                    try:
                        await execute_no_commit(
                            """INSERT OR REPLACE INTO hourly_production_metrics (
                                prediction_id, peak_power_today_kwh, production_hours_today, cumulative_today_kwh
                            ) VALUES (?, ?, ?, ?)""",
                            (
                                prediction_id,
                                _sql_value(prod_metrics.get("peak_power_today_kwh")),
                                _sql_value(prod_metrics.get("production_hours_today")),
                                _sql_value(prod_metrics.get("cumulative_today_kwh"))
                            )
                        )
                        self.stats.imported += 1
                    except Exception as e:
                        _LOGGER.debug(f"Skip production_metrics: {e}")

                # Migrate historical_context to hourly_historical_context
                hist_context = pred.get("historical_context")
                if hist_context and isinstance(hist_context, dict):
                    try:
                        await execute_no_commit(
                            """INSERT OR REPLACE INTO hourly_historical_context (
                                prediction_id, yesterday_same_hour, same_hour_avg_7days
                            ) VALUES (?, ?, ?)""",
                            (
                                prediction_id,
                                _sql_value(hist_context.get("yesterday_same_hour")),
                                _sql_value(hist_context.get("same_hour_avg_7days"))
                            )
                        )
                        self.stats.imported += 1
                    except Exception as e:
                        _LOGGER.debug(f"Skip historical_context: {e}")

                # Migrate panel_group_accuracy to hourly_panel_group_accuracy
                panel_accuracy = pred.get("panel_group_accuracy")
                if panel_accuracy and isinstance(panel_accuracy, dict):
                    for group_name, accuracy_data in panel_accuracy.items():
                        if accuracy_data and isinstance(accuracy_data, dict):
                            try:
                                await execute_no_commit(
                                    """INSERT OR REPLACE INTO hourly_panel_group_accuracy (
                                        prediction_id, group_name, prediction_kwh, actual_kwh,
                                        error_kwh, error_percent, accuracy_percent
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                    (
                                        prediction_id,
                                        group_name,
                                        _sql_value(accuracy_data.get("prediction_kwh")),
                                        _sql_value(accuracy_data.get("actual_kwh")),
                                        _sql_value(accuracy_data.get("error_kwh")),
                                        _sql_value(accuracy_data.get("error_percent")),
                                        _sql_value(accuracy_data.get("accuracy_percent"))
                                    )
                                )
                                self.stats.imported += 1
                            except Exception as e:
                                _LOGGER.debug(f"Skip panel_group_accuracy for {group_name}: {e}")

                # Batch commit for performance (after successful processing of all child tables)
                batch_counter += 1
                processed_counter += 1
                if batch_counter >= batch_size:
                    await self.db.commit()
                    batch_counter = 0
                    if total_predictions > 500:
                        progress_percent = (processed_counter / total_predictions) * 100
                        _LOGGER.info(f"   → Progress: {processed_counter}/{total_predictions} ({progress_percent:.1f}%)")

            except Exception as e:
                _LOGGER.debug(f"Skip hourly prediction: {e}")
                self.stats.skipped += 1

        # Final commit for remaining predictions
        if batch_counter > 0:
            await self.db.commit()

    async def _migrate_daily_forecasts(self, data: Dict[str, Any]) -> None:
        history = data.get("history", [])
        today = data.get("today", {})

        total_entries = len(history) + (1 if today else 0)
        if total_entries > 0:
            _LOGGER.info(f"   → Migrating {total_entries} daily forecasts...")

        if today and today.get("date"):
            try:
                today_date = today.get("date")
                if isinstance(today_date, str):
                    today_date = datetime.fromisoformat(today_date).date()

                type_mapping = {
                    "forecast_day": "today",
                    "forecast_tomorrow": "tomorrow",
                    "forecast_day_after_tomorrow": "day_after_tomorrow"
                }

                for json_key, db_type in type_mapping.items():
                    fc_data = today.get(json_key)
                    if fc_data and isinstance(fc_data, dict):
                        pred_kwh = fc_data.get("prediction_kwh")
                        pred_raw = fc_data.get("prediction_kwh_raw", pred_kwh)
                        safeguard = fc_data.get("safeguard_applied", False)
                        safeguard_red = fc_data.get("safeguard_reduction_kwh")
                        locked = fc_data.get("locked", False)
                        locked_at = fc_data.get("locked_at")
                        source = fc_data.get("source")
                        fc_date = fc_data.get("date")
                        if fc_date and isinstance(fc_date, str):
                            fc_date = datetime.fromisoformat(fc_date).date()
                        else:
                            fc_date = today_date

                        if locked_at and isinstance(locked_at, str):
                            locked_at = datetime.fromisoformat(locked_at)

                        if pred_kwh is not None:
                            await self.db.execute(
                                "INSERT OR REPLACE INTO daily_forecasts (forecast_type, forecast_date, prediction_kwh, prediction_kwh_raw, safeguard_applied, safeguard_reduction_kwh, locked, locked_at, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (db_type, fc_date, pred_kwh, pred_raw, safeguard, safeguard_red, locked, locked_at, source)
                            )
                            self.stats.imported += 1

                        updates = fc_data.get("updates", [])
                        for upd in updates:
                            upd_kwh = upd.get("prediction_kwh")
                            upd_at = upd.get("updated_at")
                            upd_source = upd.get("source")
                            if upd_at and isinstance(upd_at, str):
                                upd_at = datetime.fromisoformat(upd_at)
                            if upd_kwh is not None:
                                await self.db.execute(
                                    "INSERT INTO daily_forecast_updates (forecast_type, forecast_date, prediction_kwh, source, updated_at) VALUES (?, ?, ?, ?, ?)",
                                    (db_type, fc_date, upd_kwh, upd_source, upd_at)
                                )
                                self.stats.imported += 1
            except Exception as e:
                _LOGGER.debug(f"Skip today forecast: {e}")

        for entry in history:
            date_str = entry.get("date")
            if not date_str:
                continue

            try:
                forecast_date = datetime.fromisoformat(date_str).date() if isinstance(date_str, str) else date_str
                predicted = entry.get("predicted_kwh")
                actual = entry.get("actual_kwh")
                accuracy = entry.get("accuracy")
                prod_hours = entry.get("production_hours")
                peak_w = entry.get("peak_power_w")
                source = entry.get("forecast_source")

                existing = await self.db.fetchone(
                    "SELECT id FROM daily_summaries WHERE date = ?", (forecast_date,)
                )

                if existing:
                    await self.db.execute(
                        "UPDATE daily_summaries SET predicted_total_kwh = COALESCE(predicted_total_kwh, ?), actual_total_kwh = COALESCE(actual_total_kwh, ?), accuracy_percent = COALESCE(accuracy_percent, ?), production_hours = COALESCE(production_hours, ?), peak_power_w = COALESCE(peak_power_w, ?) WHERE id = ?",
                        (predicted, actual, accuracy, prod_hours, peak_w, existing[0])
                    )
                    self.stats.updated += 1
                else:
                    await self.db.execute(
                        "INSERT INTO daily_summaries (date, predicted_total_kwh, actual_total_kwh, accuracy_percent, production_hours, peak_power_w) VALUES (?, ?, ?, ?, ?, ?)",
                        (forecast_date, predicted, actual, accuracy, prod_hours, peak_w)
                    )
                    self.stats.imported += 1
            except Exception as e:
                _LOGGER.debug(f"Skip daily forecast history: {e}")
                self.stats.skipped += 1

        # Migrate daily_forecast_tracking (singleton table with current tracking data)
        tracking_date = data.get("date")
        if tracking_date:
            try:
                if isinstance(tracking_date, str):
                    tracking_date = datetime.fromisoformat(tracking_date).date()

                forecast_best_hour = data.get("forecast_best_hour", {})
                actual_best_hour = data.get("actual_best_hour", {})
                forecast_next_hour = data.get("forecast_next_hour", {})
                production_time = data.get("production_time", {})
                peak_today = data.get("peak_today", {})
                yield_today = data.get("yield_today", {})
                consumption_today = data.get("consumption_today", {})
                autarky = data.get("autarky", {})
                finalized = data.get("finalized", {})
                excluded_hours = finalized.get("excluded_hours", {})

                # Parse timestamps
                forecast_best_hour_locked_at = forecast_best_hour.get("locked_at")
                if forecast_best_hour_locked_at and isinstance(forecast_best_hour_locked_at, str):
                    forecast_best_hour_locked_at = datetime.fromisoformat(forecast_best_hour_locked_at)

                actual_best_hour_saved_at = actual_best_hour.get("saved_at")
                if actual_best_hour_saved_at and isinstance(actual_best_hour_saved_at, str):
                    actual_best_hour_saved_at = datetime.fromisoformat(actual_best_hour_saved_at)

                forecast_next_hour_updated_at = forecast_next_hour.get("updated_at")
                if forecast_next_hour_updated_at and isinstance(forecast_next_hour_updated_at, str):
                    forecast_next_hour_updated_at = datetime.fromisoformat(forecast_next_hour_updated_at)

                production_time_start = production_time.get("start_time")
                if production_time_start and isinstance(production_time_start, str):
                    production_time_start = datetime.fromisoformat(production_time_start)

                production_time_end = production_time.get("end_time")
                if production_time_end and isinstance(production_time_end, str):
                    production_time_end = datetime.fromisoformat(production_time_end)

                production_time_last_power = production_time.get("last_power_above_10w")
                if production_time_last_power and isinstance(production_time_last_power, str):
                    production_time_last_power = datetime.fromisoformat(production_time_last_power)

                production_time_zero_power = production_time.get("zero_power_since")
                if production_time_zero_power and isinstance(production_time_zero_power, str):
                    production_time_zero_power = datetime.fromisoformat(production_time_zero_power)

                peak_today_at = peak_today.get("at")
                if peak_today_at and isinstance(peak_today_at, str):
                    peak_today_at = datetime.fromisoformat(peak_today_at)

                autarky_calculated_at = autarky.get("calculated_at")
                if autarky_calculated_at and isinstance(autarky_calculated_at, str):
                    autarky_calculated_at = datetime.fromisoformat(autarky_calculated_at)

                finalized_at = finalized.get("at")
                if finalized_at and isinstance(finalized_at, str):
                    finalized_at = datetime.fromisoformat(finalized_at)

                await self.db.execute(
                    """INSERT OR REPLACE INTO daily_forecast_tracking (
                        date, forecast_best_hour, forecast_best_hour_kwh, forecast_best_hour_locked,
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
                        finalized_excluded_hours_reasons, finalized_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        tracking_date,
                        _sql_value(forecast_best_hour.get("hour")),
                        _sql_value(forecast_best_hour.get("prediction_kwh")),
                        _sql_value(forecast_best_hour.get("locked")),
                        forecast_best_hour_locked_at,
                        _sql_value(forecast_best_hour.get("source")),
                        _sql_value(actual_best_hour.get("hour")),
                        _sql_value(actual_best_hour.get("actual_kwh")),
                        actual_best_hour_saved_at,
                        _sql_value(forecast_next_hour.get("period")),
                        _sql_value(forecast_next_hour.get("prediction_kwh")),
                        forecast_next_hour_updated_at,
                        _sql_value(forecast_next_hour.get("source")),
                        _sql_value(production_time.get("active")),
                        _sql_value(production_time.get("duration_seconds")),
                        production_time_start,
                        production_time_end,
                        production_time_last_power,
                        production_time_zero_power,
                        _sql_value(peak_today.get("power_w")),
                        peak_today_at,
                        _sql_value(yield_today.get("kwh")),
                        _sql_value(yield_today.get("sensor")),
                        _sql_value(consumption_today.get("kwh")),
                        _sql_value(consumption_today.get("sensor")),
                        _sql_value(autarky.get("percent")),
                        autarky_calculated_at,
                        _sql_value(finalized.get("yield_kwh")),
                        _sql_value(finalized.get("consumption_kwh")),
                        _sql_value(finalized.get("production_hours")),
                        _sql_value(finalized.get("accuracy_percent")),
                        _sql_value(excluded_hours.get("count")),
                        _sql_value(excluded_hours.get("total")),
                        _sql_value(excluded_hours.get("ratio")),
                        _sql_value(excluded_hours.get("reasons")),
                        finalized_at
                    )
                )
                self.stats.imported += 1
            except Exception as e:
                _LOGGER.debug(f"Skip daily_forecast_tracking: {e}")

        # Migrate daily_statistics (singleton table with cumulative statistics)
        stats = data.get("statistics", {})
        if stats:
            try:
                all_time_peak = stats.get("all_time_peak", {})
                current_week = stats.get("current_week", {})
                current_month = stats.get("current_month", {})
                last_7_days = stats.get("last_7_days", {})
                last_30_days = stats.get("last_30_days", {})
                last_365_days = stats.get("last_365_days", {})

                # Parse timestamps and dates
                all_time_peak_date = all_time_peak.get("date")
                if all_time_peak_date and isinstance(all_time_peak_date, str):
                    all_time_peak_date = datetime.fromisoformat(all_time_peak_date).date()

                all_time_peak_at = all_time_peak.get("at")
                if all_time_peak_at and isinstance(all_time_peak_at, str):
                    all_time_peak_at = datetime.fromisoformat(all_time_peak_at)

                current_week_updated_at = current_week.get("updated_at")
                if current_week_updated_at and isinstance(current_week_updated_at, str):
                    current_week_updated_at = datetime.fromisoformat(current_week_updated_at)

                current_month_updated_at = current_month.get("updated_at")
                if current_month_updated_at and isinstance(current_month_updated_at, str):
                    current_month_updated_at = datetime.fromisoformat(current_month_updated_at)

                last_7_days_calculated_at = last_7_days.get("calculated_at")
                if last_7_days_calculated_at and isinstance(last_7_days_calculated_at, str):
                    last_7_days_calculated_at = datetime.fromisoformat(last_7_days_calculated_at)

                last_30_days_calculated_at = last_30_days.get("calculated_at")
                if last_30_days_calculated_at and isinstance(last_30_days_calculated_at, str):
                    last_30_days_calculated_at = datetime.fromisoformat(last_30_days_calculated_at)

                last_365_days_calculated_at = last_365_days.get("calculated_at")
                if last_365_days_calculated_at and isinstance(last_365_days_calculated_at, str):
                    last_365_days_calculated_at = datetime.fromisoformat(last_365_days_calculated_at)

                await self.db.execute(
                    """INSERT OR REPLACE INTO daily_statistics (
                        id, all_time_peak_power_w, all_time_peak_date, all_time_peak_at,
                        current_week_period, current_week_date_range, current_week_yield_kwh,
                        current_week_consumption_kwh, current_week_days, current_week_updated_at,
                        current_month_period, current_month_yield_kwh, current_month_consumption_kwh,
                        current_month_avg_autarky, current_month_days, current_month_updated_at,
                        last_7_days_avg_yield_kwh, last_7_days_avg_accuracy, last_7_days_total_yield_kwh, last_7_days_calculated_at,
                        last_30_days_avg_yield_kwh, last_30_days_avg_accuracy, last_30_days_total_yield_kwh, last_30_days_calculated_at,
                        last_365_days_avg_yield_kwh, last_365_days_total_yield_kwh, last_365_days_calculated_at
                    ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        _sql_value(all_time_peak.get("power_w")),
                        all_time_peak_date,
                        all_time_peak_at,
                        _sql_value(current_week.get("period")),
                        _sql_value(current_week.get("date_range")),
                        _sql_value(current_week.get("yield_kwh")),
                        _sql_value(current_week.get("consumption_kwh")),
                        _sql_value(current_week.get("days")),
                        current_week_updated_at,
                        _sql_value(current_month.get("period")),
                        _sql_value(current_month.get("yield_kwh")),
                        _sql_value(current_month.get("consumption_kwh")),
                        _sql_value(current_month.get("avg_autarky")),
                        _sql_value(current_month.get("days")),
                        current_month_updated_at,
                        _sql_value(last_7_days.get("avg_yield_kwh")),
                        _sql_value(last_7_days.get("avg_accuracy")),
                        _sql_value(last_7_days.get("total_yield_kwh")),
                        last_7_days_calculated_at,
                        _sql_value(last_30_days.get("avg_yield_kwh")),
                        _sql_value(last_30_days.get("avg_accuracy")),
                        _sql_value(last_30_days.get("total_yield_kwh")),
                        last_30_days_calculated_at,
                        _sql_value(last_365_days.get("avg_yield_kwh")),
                        _sql_value(last_365_days.get("total_yield_kwh")),
                        last_365_days_calculated_at
                    )
                )
                self.stats.imported += 1
            except Exception as e:
                _LOGGER.debug(f"Skip daily_statistics: {e}")

    async def _migrate_daily_summaries(self, data: Dict[str, Any]) -> None:
        summaries = data if isinstance(data, list) else data.get("summaries", [])

        for summary in summaries:
            date_val = summary.get("date")
            if not date_val:
                continue

            try:
                if isinstance(date_val, str):
                    date_val = datetime.fromisoformat(date_val).date()

                overall = summary.get("overall", {})
                if not overall:
                    overall = summary

                existing = await self.db.fetchone(
                    "SELECT id FROM daily_summaries WHERE date = ?", (date_val,)
                )

                predicted = overall.get("predicted_total_kwh") or overall.get("predicted_kwh")
                actual = overall.get("actual_total_kwh") or overall.get("actual_kwh")
                accuracy = overall.get("accuracy_percent") or overall.get("accuracy")
                error = overall.get("error_kwh") or overall.get("absolute_error_kwh")
                prod_hours = overall.get("production_hours")
                peak_w = overall.get("peak_power_w")
                peak_hour = overall.get("peak_hour")

                if existing:
                    await self.db.execute(
                        "UPDATE daily_summaries SET predicted_total_kwh = ?, actual_total_kwh = ?, accuracy_percent = ?, error_kwh = ?, production_hours = ?, peak_power_w = ?, peak_hour = ? WHERE id = ?",
                        (predicted, actual, accuracy, error, prod_hours, peak_w, peak_hour, existing[0])
                    )
                    self.stats.updated += 1
                else:
                    await self.db.execute(
                        "INSERT INTO daily_summaries (date, predicted_total_kwh, actual_total_kwh, accuracy_percent, error_kwh, production_hours, peak_power_w, peak_hour, month, season) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (date_val, predicted, actual, accuracy, error, prod_hours, peak_w, peak_hour, summary.get("month"), summary.get("season"))
                    )
                    self.stats.imported += 1

                time_windows = summary.get("time_windows", {})
                if time_windows and isinstance(time_windows, dict):
                    for window_name, tw in time_windows.items():
                        if not isinstance(tw, dict):
                            continue
                        try:
                            await self.db.execute(
                                "INSERT OR REPLACE INTO daily_summary_time_windows (date, window_name, predicted_kwh, actual_kwh, accuracy, stable, hours_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (date_val, window_name, tw.get("predicted_kwh"), tw.get("actual_kwh"), tw.get("accuracy"), tw.get("stable"), tw.get("hours_count"))
                            )
                            self.stats.imported += 1
                        except:
                            pass

                frost = summary.get("frost_analysis", {})
                if frost and isinstance(frost, dict):
                    try:
                        await self.db.execute(
                            "INSERT OR REPLACE INTO daily_summary_frost_analysis (date, hours_analyzed, frost_detected, total_affected_hours, heavy_frost_hours, light_frost_hours) VALUES (?, ?, ?, ?, ?, ?)",
                            (date_val, frost.get("hours_analyzed"), frost.get("frost_detected"), frost.get("total_affected_hours"), frost.get("heavy_frost_hours"), frost.get("light_frost_hours"))
                        )
                        self.stats.imported += 1
                    except:
                        pass
            except Exception as e:
                _LOGGER.debug(f"Skip daily summary: {e}")
                self.stats.skipped += 1

    async def _migrate_weather_forecast(self, data: Dict[str, Any]) -> None:
        forecast = data.get("forecast", {})
        records = []

        if isinstance(forecast, dict):
            for date_str, hours_data in forecast.items():
                try:
                    forecast_date = datetime.fromisoformat(date_str).date()
                except:
                    self.stats.skipped += 1
                    continue

                if not isinstance(hours_data, dict):
                    continue

                for hour_str, fc in hours_data.items():
                    if not isinstance(fc, dict):
                        continue

                    try:
                        hour = int(hour_str)
                        records.append((
                            forecast_date, hour,
                            _sql_value(fc.get("temperature")),
                            _sql_value(fc.get("solar_radiation_wm2")),
                            _sql_value(fc.get("wind")),
                            _sql_value(fc.get("humidity")),
                            _sql_value(fc.get("rain")),
                            _sql_value(fc.get("clouds")),
                            _sql_value(fc.get("cloud_cover_low")),
                            _sql_value(fc.get("cloud_cover_mid")),
                            _sql_value(fc.get("cloud_cover_high")),
                            _sql_value(fc.get("pressure")),
                            _sql_value(fc.get("direct_radiation")),
                            _sql_value(fc.get("diffuse_radiation")),
                            _sql_value(fc.get("visibility_m")),
                            _sql_value(fc.get("fog_detected")),
                            _sql_value(fc.get("fog_type"))
                        ))
                    except Exception as e:
                        _LOGGER.debug(f"Skip weather forecast: {e}")
                        self.stats.skipped += 1

        if records:
            try:
                count = await self.db.executemany(
                    "INSERT OR REPLACE INTO weather_forecast (forecast_date, hour, temperature, solar_radiation_wm2, wind, humidity, rain, clouds, cloud_cover_low, cloud_cover_mid, cloud_cover_high, pressure, direct_radiation, diffuse_radiation, visibility_m, fog_detected, fog_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    records
                )
                self.stats.imported += count
            except Exception as e:
                _LOGGER.error(f"Batch insert weather forecast failed: {e}")
                self.stats.errors += 1

    async def _migrate_astronomy_cache(self, data: Dict[str, Any]) -> None:
        days = data.get("days", {})
        records = []

        if isinstance(days, dict):
            for date_str, day_data in days.items():
                if not isinstance(day_data, dict):
                    continue

                try:
                    cache_date = datetime.fromisoformat(date_str).date()
                except:
                    self.stats.skipped += 1
                    continue

                sunrise = _sql_value(day_data.get("sunrise_local"))
                sunset = _sql_value(day_data.get("sunset_local"))
                solar_noon = _sql_value(day_data.get("solar_noon_local"))
                daylight_hours = _sql_value(day_data.get("daylight_hours"))

                hourly = day_data.get("hourly", [])
                if isinstance(hourly, list):
                    for hour_idx, hour_data in enumerate(hourly):
                        if not isinstance(hour_data, dict):
                            continue
                        try:
                            records.append((
                                cache_date, hour_idx,
                                _sql_value(hour_data.get("elevation_deg")),
                                _sql_value(hour_data.get("azimuth_deg")),
                                _sql_value(hour_data.get("clear_sky_solar_radiation_wm2")),
                                _sql_value(hour_data.get("theoretical_max_pv_kwh")),
                                sunrise, sunset, solar_noon, daylight_hours
                            ))
                        except Exception as e:
                            _LOGGER.debug(f"Skip astronomy cache hour: {e}")
                            self.stats.skipped += 1
                elif isinstance(hourly, dict):
                    for hour_str, hour_data in hourly.items():
                        if not isinstance(hour_data, dict):
                            continue
                        try:
                            hour = int(hour_str)
                            records.append((
                                cache_date, hour,
                                _sql_value(hour_data.get("elevation_deg")),
                                _sql_value(hour_data.get("azimuth_deg")),
                                _sql_value(hour_data.get("clear_sky_solar_radiation_wm2")),
                                _sql_value(hour_data.get("theoretical_max_pv_kwh")),
                                sunrise, sunset, solar_noon, daylight_hours
                            ))
                        except Exception as e:
                            _LOGGER.debug(f"Skip astronomy cache hour: {e}")
                            self.stats.skipped += 1

        if records:
            try:
                count = await self.db.executemany(
                    "INSERT OR REPLACE INTO astronomy_cache (cache_date, hour, sun_elevation_deg, sun_azimuth_deg, clear_sky_radiation_wm2, theoretical_max_kwh, sunrise, sunset, solar_noon, daylight_hours) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    records
                )
                self.stats.imported += count
            except Exception as e:
                _LOGGER.error(f"Batch insert astronomy cache failed: {e}")
                self.stats.errors += 1

        location = data.get("location", {})
        pv_system = data.get("pv_system", {})
        if location or pv_system:
            try:
                await self.db.execute(
                    "INSERT OR REPLACE INTO astronomy_system_info (id, latitude, longitude, elevation_m, timezone, installed_capacity_kwp) VALUES (1, ?, ?, ?, ?, ?)",
                    (location.get("latitude"), location.get("longitude"), location.get("elevation_m"), location.get("timezone"), pv_system.get("installed_capacity_kwp"))
                )
            except:
                pass

    async def _migrate_weather_expert_learning(self, data: Dict[str, Any]) -> None:
        daily_history = data.get("daily_history", {})

        if isinstance(daily_history, dict):
            for date_str, day_data in daily_history.items():
                if not isinstance(day_data, dict):
                    continue

                try:
                    date_val = datetime.fromisoformat(date_str).date()
                except:
                    continue

                learned_at = day_data.get("learned_at")
                if learned_at and isinstance(learned_at, str):
                    try:
                        learned_at = datetime.fromisoformat(learned_at)
                    except:
                        learned_at = datetime.now()
                else:
                    learned_at = datetime.now()

                comparison_hours = day_data.get("comparison_hours", 0)

                mae_by_cloud_type = day_data.get("mae_by_cloud_type", {})
                weights_updated = day_data.get("weights_updated", {})

                for cloud_type, experts_mae in mae_by_cloud_type.items():
                    if not isinstance(experts_mae, dict):
                        continue

                    cloud_weights = weights_updated.get(cloud_type, {})

                    for expert_name, mae in experts_mae.items():
                        weight_after = cloud_weights.get(expert_name, 0) if isinstance(cloud_weights, dict) else 0

                        try:
                            await self.db.execute(
                                "INSERT OR REPLACE INTO weather_expert_learning (date, cloud_type, expert_name, mae, weight_after, comparison_hours, learned_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (date_val, cloud_type, expert_name, mae, weight_after, comparison_hours, learned_at)
                            )
                            self.stats.imported += 1
                        except Exception as e:
                            _LOGGER.debug(f"Skip weather expert learning: {e}")
                            self.stats.skipped += 1

    async def _migrate_weather_source_learning(self, data: Dict[str, Any]) -> None:
        daily_history = data.get("daily_history", {})

        if isinstance(daily_history, dict):
            for date_str, day_data in daily_history.items():
                if not isinstance(day_data, dict):
                    continue

                try:
                    date_val = datetime.fromisoformat(date_str).date()
                except:
                    continue

                learned_at = day_data.get("learned_at")
                if learned_at and isinstance(learned_at, str):
                    try:
                        learned_at = datetime.fromisoformat(learned_at)
                    except:
                        learned_at = datetime.now()
                else:
                    learned_at = datetime.now()

                weights_after = day_data.get("weights_after", {})

                for key, value in day_data.items():
                    if key.startswith("mae_"):
                        source_name = key[4:]
                        weight_after = weights_after.get(source_name, 0)

                        try:
                            await self.db.execute(
                                "INSERT OR REPLACE INTO weather_source_learning (date, source_name, mae, weight_after, learned_at) VALUES (?, ?, ?, ?, ?)",
                                (date_val, source_name, value, weight_after, learned_at)
                            )
                            self.stats.imported += 1
                        except Exception as e:
                            _LOGGER.debug(f"Skip weather source learning: {e}")
                            self.stats.skipped += 1

    async def _migrate_panel_group_sensor_state(self, data: Dict[str, Any]) -> None:
        last_values = data.get("last_values", {})
        last_updated = data.get("last_updated")

        if last_updated and isinstance(last_updated, str):
            try:
                last_updated = datetime.fromisoformat(last_updated)
            except:
                last_updated = datetime.now()
        else:
            last_updated = datetime.now()

        for group_name, last_value in last_values.items():
            try:
                await self.db.execute(
                    "INSERT OR REPLACE INTO panel_group_sensor_state (group_name, last_value, last_updated) VALUES (?, ?, ?)",
                    (group_name, last_value, last_updated)
                )
                self.stats.imported += 1
            except Exception as e:
                _LOGGER.debug(f"Skip panel group sensor state: {e}")
                self.stats.skipped += 1

    async def _migrate_yield_cache(self, data: Dict[str, Any]) -> None:
        try:
            value = data.get("value")
            time_val = data.get("time")
            date_val = data.get("date")

            if time_val and isinstance(time_val, str):
                time_val = datetime.fromisoformat(time_val)

            if date_val and isinstance(date_val, str):
                date_val = datetime.fromisoformat(date_val).date()

            await self.db.execute(
                "INSERT OR REPLACE INTO yield_cache (id, value, time, date) VALUES (1, ?, ?, ?)",
                (value, time_val, date_val)
            )
            self.stats.imported += 1
        except Exception as e:
            _LOGGER.debug(f"Skip yield cache: {e}")
            self.stats.skipped += 1

    async def _migrate_visibility_learning(self, data: Dict[str, Any]) -> None:
        try:
            stats = data.get("stats", {})
            weights = data.get("weights", {})

            last_learning_date = stats.get("last_learning_date")
            if last_learning_date and isinstance(last_learning_date, str):
                try:
                    last_learning_date = datetime.fromisoformat(last_learning_date).date()
                except:
                    last_learning_date = None

            last_updated = data.get("updated_at")
            if last_updated and isinstance(last_updated, str):
                try:
                    last_updated = datetime.fromisoformat(last_updated)
                except:
                    last_updated = datetime.now()
            else:
                last_updated = datetime.now()

            # Extract fog weights
            fog_weights = weights.get("fog", {})
            fog_light_weights = weights.get("fog_light", {})

            await self.db.execute(
                "INSERT OR REPLACE INTO visibility_learning (id, version, has_solar_radiation_sensor, last_learning_date, total_fog_hours_learned, total_fog_light_hours_learned, bright_sky_fog_hits, pirate_weather_fog_hits, learning_sessions, fog_bright_sky_weight, fog_pirate_weather_weight, fog_light_bright_sky_weight, fog_light_pirate_weather_weight, last_updated) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (data.get("version", "1.0"), data.get("has_solar_radiation_sensor", False), last_learning_date, stats.get("total_fog_hours_learned", 0), stats.get("total_fog_light_hours_learned", 0), stats.get("bright_sky_fog_hits", 0), stats.get("pirate_weather_fog_hits", 0), stats.get("learning_sessions", 0), fog_weights.get("bright_sky", 0.5), fog_weights.get("pirate_weather", 0.5), fog_light_weights.get("bright_sky", 0.5), fog_light_weights.get("pirate_weather", 0.5), last_updated)
            )
            self.stats.imported += 1
        except Exception as e:
            _LOGGER.debug(f"Skip visibility learning: {e}")
            self.stats.skipped += 1

    async def _migrate_forecast_drift_log(self, data: Dict[str, Any]) -> None:
        entries = data if isinstance(data, list) else data.get("entries", [])

        for entry in entries:
            timestamp = entry.get("timestamp")
            if not timestamp:
                date_str = entry.get("date")
                time_str = entry.get("time")
                if date_str and time_str:
                    timestamp = f"{date_str}T{time_str}:00"
                elif date_str:
                    timestamp = f"{date_str}T00:00:00"

            if not timestamp:
                self.stats.skipped += 1
                continue

            try:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)

                if entry.get("morning_deviation_kwh") is not None or entry.get("forecast_drift_percent") is not None:
                    entry_type = "morning_correction"
                elif entry.get("sensor_cloud_percent") is not None or entry.get("discrepancy_percent") is not None:
                    entry_type = "cloud_discrepancy"
                else:
                    entry_type = "morning_correction"

                await self.db.execute(
                    "INSERT INTO forecast_drift_log (timestamp, entry_type, morning_deviation_kwh, forecast_drift_percent, correction_applied, sensor_cloud_percent, forecast_cloud_percent, discrepancy_percent, action) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (timestamp, entry_type, entry.get("morning_deviation_kwh"), entry.get("forecast_drift_percent"), entry.get("correction_applied"), entry.get("sensor_cloud_percent"), entry.get("forecast_cloud_percent"), entry.get("discrepancy_percent"), entry.get("action"))
                )
                self.stats.imported += 1
            except Exception as e:
                _LOGGER.debug(f"Skip forecast drift log: {e}")
                self.stats.skipped += 1

    async def _migrate_hourly_weather_actual(self, data: Dict[str, Any]) -> None:
        hourly_data = data.get("hourly_data", {})
        records = []

        if isinstance(hourly_data, dict):
            for date_str, hours_data in hourly_data.items():
                try:
                    date_val = datetime.fromisoformat(date_str).date()
                except:
                    self.stats.skipped += 1
                    continue

                if not isinstance(hours_data, dict):
                    continue

                for hour_str, entry in hours_data.items():
                    if not isinstance(entry, dict):
                        continue

                    try:
                        hour = int(hour_str)
                        frost_analysis = entry.get("frost_analysis", {})
                        records.append((
                            date_val, hour,
                            _sql_value(entry.get("temperature_c")),
                            _sql_value(entry.get("humidity_percent")),
                            _sql_value(entry.get("wind_speed_ms")),
                            _sql_value(entry.get("precipitation_mm")),
                            _sql_value(entry.get("pressure_hpa")),
                            _sql_value(entry.get("solar_radiation_wm2")),
                            _sql_value(entry.get("lux")),
                            _sql_value(entry.get("timestamp")),
                            _sql_value(entry.get("source")),
                            _sql_value(entry.get("cloud_cover_percent")),
                            _sql_value(entry.get("cloud_cover_source")),
                            _sql_value(entry.get("frost_detected")),
                            _sql_value(entry.get("frost_score")),
                            _sql_value(entry.get("frost_confidence")),
                            _sql_value(frost_analysis.get("dewpoint_c")),
                            _sql_value(frost_analysis.get("frost_margin_c")),
                            _sql_value(frost_analysis.get("frost_probability")),
                            _sql_value(frost_analysis.get("correlation_diff_percent")),
                            _sql_value(frost_analysis.get("detection_method")),
                            _sql_value(frost_analysis.get("wind_frost_factor")),
                            _sql_value(frost_analysis.get("physical_frost_possible")),
                            _sql_value(entry.get("hours_after_sunrise")),
                            _sql_value(entry.get("hours_before_sunset")),
                            _sql_value(entry.get("snow_covered_panels")),
                            _sql_value(entry.get("snow_coverage_source"))
                        ))
                    except Exception as e:
                        _LOGGER.debug(f"Skip hourly weather actual: {e}")
                        self.stats.skipped += 1

        if records:
            try:
                count = await self.db.executemany(
                    "INSERT OR REPLACE INTO hourly_weather_actual (date, hour, temperature_c, humidity_percent, wind_speed_ms, precipitation_mm, pressure_hpa, solar_radiation_wm2, lux, timestamp, source, cloud_cover_percent, cloud_cover_source, frost_detected, frost_score, frost_confidence, dewpoint_c, frost_margin_c, frost_probability, correlation_diff_percent, detection_method, wind_frost_factor, physical_frost_possible, hours_after_sunrise, hours_before_sunset, snow_covered_panels, snow_coverage_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    records
                )
                self.stats.imported += count
            except Exception as e:
                _LOGGER.error(f"Batch insert hourly weather actual failed: {e}")
                self.stats.errors += 1

        snow = data.get("snow_tracking", {})
        if snow and isinstance(snow, dict):
            try:
                last_snow = snow.get("last_snow_event")
                if last_snow and isinstance(last_snow, str):
                    last_snow = datetime.fromisoformat(last_snow)

                panels_covered = snow.get("panels_covered_since")
                if panels_covered and isinstance(panels_covered, str):
                    panels_covered = datetime.fromisoformat(panels_covered)

                melt_started = snow.get("melt_started_at")
                if melt_started and isinstance(melt_started, str):
                    melt_started = datetime.fromisoformat(melt_started)

                await self.db.execute(
                    "INSERT OR REPLACE INTO snow_tracking (id, last_snow_event, panels_covered_since, estimated_depth_mm, melt_started_at, updated_at) VALUES (1, ?, ?, ?, ?, ?)",
                    (last_snow, panels_covered, snow.get("estimated_depth_mm", 0), melt_started, datetime.now())
                )
                self.stats.imported += 1
            except Exception as e:
                _LOGGER.debug(f"Skip snow tracking: {e}")

    async def _migrate_weather_precision_daily(self, data: Dict[str, Any]) -> None:
        daily_tracking = data.get("daily_tracking", {})
        hourly_records = []
        summary_records = []

        if isinstance(daily_tracking, dict):
            for date_str, day_data in daily_tracking.items():
                if not isinstance(day_data, dict):
                    continue

                try:
                    date_val = datetime.fromisoformat(date_str).date()
                except:
                    self.stats.skipped += 1
                    continue

                for hourly in day_data.get("hourly_comparisons", []):
                    hour = hourly.get("hour")
                    if hour is None:
                        continue

                    fields = hourly.get("fields", {})
                    temp = fields.get("temperature", {})
                    hum = fields.get("humidity", {})
                    wind = fields.get("wind", {})
                    rain = fields.get("rain", {})
                    solar = fields.get("solar_radiation_wm2", fields.get("solar", {}))
                    clouds = fields.get("clouds", {})

                    try:
                        hourly_records.append((
                            date_val, hour,
                            _sql_value(temp.get("forecast")), _sql_value(temp.get("actual")), _sql_value(temp.get("offset")),
                            _sql_value(hum.get("forecast")), _sql_value(hum.get("actual")), _sql_value(hum.get("factor")),
                            _sql_value(wind.get("forecast")), _sql_value(wind.get("actual")), _sql_value(wind.get("factor")),
                            _sql_value(rain.get("forecast")), _sql_value(rain.get("actual")), _sql_value(rain.get("difference")),
                            _sql_value(solar.get("forecast")), _sql_value(solar.get("actual")), _sql_value(solar.get("factor")),
                            _sql_value(clouds.get("forecast")), _sql_value(clouds.get("actual")), _sql_value(clouds.get("factor"))
                        ))
                    except Exception as e:
                        _LOGGER.debug(f"Skip weather precision hourly: {e}")
                        self.stats.skipped += 1

                daily_factors = day_data.get("daily_factors", {})
                hours_tracked = day_data.get("hours_tracked", len(day_data.get("hourly_comparisons", [])))
                summary_records.append((
                    date_val, hours_tracked,
                    _sql_value(daily_factors.get("temperature")), _sql_value(daily_factors.get("pressure")),
                    _sql_value(daily_factors.get("solar_radiation_wm2")), _sql_value(daily_factors.get("clouds")),
                    _sql_value(daily_factors.get("humidity")), _sql_value(daily_factors.get("wind")), _sql_value(daily_factors.get("rain"))
                ))

        if hourly_records:
            try:
                count = await self.db.executemany(
                    "INSERT OR REPLACE INTO weather_precision_daily (date, hour, temp_forecast, temp_actual, temp_offset, humidity_forecast, humidity_actual, humidity_factor, wind_forecast, wind_actual, wind_factor, rain_forecast, rain_actual, rain_difference, solar_forecast, solar_actual, solar_factor, clouds_forecast, clouds_actual, clouds_factor) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    hourly_records
                )
                self.stats.imported += count
            except Exception as e:
                _LOGGER.error(f"Batch insert weather precision daily failed: {e}")
                self.stats.errors += 1

        if summary_records:
            try:
                await self.db.executemany(
                    "INSERT OR REPLACE INTO weather_precision_daily_summary (date, hours_tracked, avg_temp_offset, avg_pressure_offset, avg_solar_factor, avg_clouds_factor, avg_humidity_factor, avg_wind_factor, avg_rain_diff) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    summary_records
                )
            except:
                pass

        # Migrate rolling_averages.correction_factors to weather_precision_factors
        rolling_averages = data.get("rolling_averages", {})
        correction_factors = rolling_averages.get("correction_factors", {})
        confidence_data = rolling_averages.get("confidence", {})

        if correction_factors:
            try:
                # Calculate average confidence
                conf_values = [v for v in confidence_data.values() if isinstance(v, (int, float))]
                avg_confidence = sum(conf_values) / len(conf_values) if conf_values else 0.0

                await self.db.execute(
                    """INSERT OR REPLACE INTO weather_precision_factors
                       (id, temperature_factor, solar_factor, cloud_factor, wind_factor,
                        humidity_factor, rain_factor, pressure_factor, sample_days, confidence, updated_at)
                       VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                    (
                        _sql_value(correction_factors.get("temperature", 0.0)),
                        _sql_value(correction_factors.get("solar_radiation_wm2", 1.0)),
                        _sql_value(correction_factors.get("clouds", 1.0)),
                        _sql_value(correction_factors.get("wind", 1.0)),
                        _sql_value(correction_factors.get("humidity", 1.0)),
                        _sql_value(correction_factors.get("rain", 0.0)),
                        _sql_value(correction_factors.get("pressure", 0.0)),
                        _sql_value(rolling_averages.get("sample_days", 0)),
                        _sql_value(avg_confidence)
                    )
                )
                self.stats.imported += 1
            except Exception as e:
                _LOGGER.debug(f"Skip weather precision factors: {e}")

    async def _migrate_panel_group_daily_cache(self, data: Dict[str, Any]) -> None:
        cache_date = data.get("date")
        if not cache_date:
            self.stats.skipped += 1
            return

        try:
            if isinstance(cache_date, str):
                cache_date = datetime.fromisoformat(cache_date).date()
        except:
            self.stats.skipped += 1
            return

        cache_records = []
        hourly_records = []

        groups = data.get("groups", {})
        if isinstance(groups, dict):
            for group_name, group_data in groups.items():
                if not isinstance(group_data, dict):
                    continue

                try:
                    cache_records.append((
                        cache_date, group_name,
                        _sql_value(group_data.get("prediction_total_kwh")),
                        _sql_value(group_data.get("actual_total_kwh"))
                    ))

                    for hour_data in group_data.get("hourly", []):
                        hour = hour_data.get("hour")
                        if hour is not None:
                            hourly_records.append((
                                cache_date, group_name, hour,
                                _sql_value(hour_data.get("prediction_kwh")),
                                _sql_value(hour_data.get("actual_kwh"))
                            ))
                except Exception as e:
                    _LOGGER.debug(f"Skip panel group {group_name}: {e}")
                    self.stats.skipped += 1

        if cache_records:
            try:
                count = await self.db.executemany(
                    "INSERT OR REPLACE INTO panel_group_daily_cache (cache_date, group_name, prediction_total_kwh, actual_total_kwh) VALUES (?, ?, ?, ?)",
                    cache_records
                )
                self.stats.imported += count
            except Exception as e:
                _LOGGER.error(f"Batch insert panel group daily cache failed: {e}")
                self.stats.errors += 1

        if hourly_records:
            try:
                await self.db.executemany(
                    "INSERT OR REPLACE INTO panel_group_daily_hourly (cache_date, group_name, hour, prediction_kwh, actual_kwh) VALUES (?, ?, ?, ?, ?)",
                    hourly_records
                )
            except Exception as e:
                _LOGGER.error(f"Batch insert panel group daily hourly failed: {e}")
                self.stats.errors += 1

    async def _migrate_multi_day_forecast(self, data: Dict[str, Any]) -> None:
        days = data.get("days", {})
        forecast_records = []
        panel_records = []

        if isinstance(days, dict):
            for date_str, day_data in days.items():
                if not isinstance(day_data, dict):
                    continue

                try:
                    forecast_date = datetime.fromisoformat(date_str).date()
                except:
                    self.stats.skipped += 1
                    continue

                day_type = day_data.get("day_type", "today")

                for entry in day_data.get("hourly", []):
                    hour = entry.get("hour")
                    if hour is None:
                        continue

                    try:
                        forecast_records.append((
                            forecast_date, day_type, hour,
                            _sql_value(entry.get("prediction_kwh")),
                            _sql_value(entry.get("cloud_cover")),
                            _sql_value(entry.get("temperature")),
                            _sql_value(entry.get("solar_radiation_wm2")),
                            _sql_value(entry.get("weather_source"))
                        ))

                        for panel in entry.get("panel_groups", []):
                            group_name = panel.get("name")
                            if group_name:
                                panel_records.append((
                                    forecast_date, hour, group_name,
                                    _sql_value(panel.get("power_kwh")),
                                    _sql_value(panel.get("contribution_percent")),
                                    _sql_value(panel.get("poa_wm2")),
                                    _sql_value(panel.get("aoi_deg")),
                                    _sql_value(panel.get("source"))
                                ))
                    except Exception as e:
                        _LOGGER.debug(f"Skip multi day forecast: {e}")
                        self.stats.skipped += 1

        if forecast_records:
            try:
                count = await self.db.executemany(
                    "INSERT OR REPLACE INTO multi_day_hourly_forecast (forecast_date, day_type, hour, prediction_kwh, cloud_cover, temperature, solar_radiation_wm2, weather_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    forecast_records
                )
                self.stats.imported += count
            except Exception as e:
                _LOGGER.error(f"Batch insert multi day forecast failed: {e}")
                self.stats.errors += 1

        if panel_records:
            try:
                count = await self.db.executemany(
                    "INSERT OR REPLACE INTO multi_day_hourly_forecast_panels (forecast_date, hour, group_name, power_kwh, contribution_percent, poa_wm2, aoi_deg, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    panel_records
                )
                self.stats.imported += count
            except Exception as e:
                _LOGGER.error(f"Batch insert multi day forecast panels failed: {e}")
                self.stats.errors += 1

    async def _migrate_retrospective_forecast(self, data: Dict[str, Any]) -> None:
        try:
            forecast_summary = data.get("forecast_summary", {})
            simulation_context = data.get("simulation_context", {})

            generated_at = data.get("generated_at")
            if generated_at and isinstance(generated_at, str):
                generated_at = datetime.fromisoformat(generated_at)

            target_date = simulation_context.get("target_date")
            if target_date and isinstance(target_date, str):
                try:
                    target_date = datetime.fromisoformat(target_date).date()
                except:
                    target_date = None

            await self.db.execute(
                "INSERT OR REPLACE INTO retrospective_forecast (id, version, generated_at, simulated_forecast_time, sunrise_today, target_date, today_kwh, today_kwh_raw, safeguard_applied, tomorrow_kwh, day_after_tomorrow_kwh, method, confidence, best_hour, best_hour_kwh) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _sql_value(data.get("version", "1.0")),
                    _sql_value(generated_at),
                    _sql_value(simulation_context.get("simulated_forecast_time")),
                    _sql_value(simulation_context.get("sunrise_today")),
                    target_date,
                    _sql_value(forecast_summary.get("today_kwh")),
                    _sql_value(forecast_summary.get("today_kwh_raw")),
                    _sql_value(forecast_summary.get("safeguard_applied", False)),
                    _sql_value(forecast_summary.get("tomorrow_kwh")),
                    _sql_value(forecast_summary.get("day_after_tomorrow_kwh")),
                    _sql_value(forecast_summary.get("method")),
                    _sql_value(forecast_summary.get("confidence")),
                    _sql_value(forecast_summary.get("best_hour")),
                    _sql_value(forecast_summary.get("best_hour_kwh"))
                )
            )
            self.stats.imported += 1

            await self.db.execute("DELETE FROM retrospective_forecast_hourly")

            hourly_records = []
            for hour_data in data.get("hourly_predictions", []):
                hour = hour_data.get("hour")
                if hour is not None:
                    weather = hour_data.get("weather", {})
                    astronomy = hour_data.get("astronomy", {})
                    hourly_records.append((
                        hour,
                        _sql_value(hour_data.get("prediction_kwh")),
                        _sql_value(weather.get("temperature_c")),
                        _sql_value(weather.get("cloud_cover_percent")),
                        _sql_value(weather.get("humidity_percent")),
                        _sql_value(weather.get("wind_speed_ms")),
                        _sql_value(weather.get("precipitation_mm")),
                        _sql_value(weather.get("direct_radiation")),
                        _sql_value(weather.get("diffuse_radiation")),
                        _sql_value(weather.get("visibility_m")),
                        _sql_value(weather.get("fog_detected")),
                        _sql_value(weather.get("fog_type")),
                        _sql_value(astronomy.get("sun_elevation_deg")),
                        _sql_value(astronomy.get("sun_azimuth_deg")),
                        _sql_value(astronomy.get("theoretical_max_kwh")),
                        _sql_value(astronomy.get("clear_sky_radiation_wm2"))
                    ))

            if hourly_records:
                count = await self.db.executemany(
                    "INSERT OR REPLACE INTO retrospective_forecast_hourly (hour, prediction_kwh, temperature_c, cloud_cover_percent, humidity_percent, wind_speed_ms, precipitation_mm, direct_radiation, diffuse_radiation, visibility_m, fog_detected, fog_type, sun_elevation_deg, sun_azimuth_deg, theoretical_max_kwh, clear_sky_radiation_wm2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    hourly_records
                )
                self.stats.imported += count
        except Exception as e:
            _LOGGER.debug(f"Skip retrospective forecast: {e}")
            self.stats.skipped += 1

    async def _set_migration_flag(self) -> None:
        def _write_flag():
            flag_path = Path(self.hass.config.path(MIGRATION_FLAG))
            flag_path.parent.mkdir(parents=True, exist_ok=True)
            flag_path.write_text(
                f"JSON to SQLite migration completed\n"
                f"Timestamp: {datetime.now().isoformat()}\n"
                f"Stats: {self.stats}\n"
            )

        await self.hass.async_add_executor_job(_write_flag)
        _LOGGER.info("Migration flag set")


async def run_json_migration(hass: HomeAssistant, db_manager: DatabaseManager) -> JsonMigrationStats:
    migrator = JsonMigrator(hass, db_manager)
    return await migrator.run_migration()


async def dry_run_json_migration(hass: HomeAssistant, db_manager: DatabaseManager) -> Dict[str, Any]:
    migrator = JsonMigrator(hass, db_manager)
    return await migrator.dry_run()
