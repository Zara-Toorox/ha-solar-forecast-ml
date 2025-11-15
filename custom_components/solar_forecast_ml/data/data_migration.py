"""Migration utilities for converting old prediction_history.json to new structure"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)


class DataMigration:
    """Migrate from old prediction_history.json to new hourly_predictions.json"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.stats_dir = data_dir / "stats"

        # Old files
        self.old_prediction_history = self.stats_dir / "prediction_history.json"
        self.old_prediction_history_backup = self.stats_dir / "prediction_history_backup_v1.json"

        # New files
        self.new_hourly_predictions = self.stats_dir / "hourly_predictions.json"
        self.new_daily_summaries = self.stats_dir / "daily_summaries.json"

    async def migrate(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Migrate old data to new structure

        Args:
            dry_run: If True, don't actually write files, just analyze

        Returns:
            Migration report with statistics
        """
        _LOGGER.info("=" * 80)
        _LOGGER.info("DATA MIGRATION: prediction_history.json → new structure")
        _LOGGER.info("=" * 80)
        _LOGGER.info(f"Mode: {'DRY RUN (no files written)' if dry_run else 'LIVE MIGRATION'}")
        _LOGGER.info("")

        report = {
            "success": False,
            "dry_run": dry_run,
            "old_file_exists": False,
            "old_predictions_count": 0,
            "old_dates_count": 0,
            "converted_dates": [],
            "skipped_dates": [],
            "errors": [],
            "warnings": [],
        }

        try:
            # Step 1: Check if old file exists
            if not self.old_prediction_history.exists():
                _LOGGER.warning("Old prediction_history.json not found - nothing to migrate")
                report["warnings"].append("No old file found")
                report["success"] = True
                return report

            report["old_file_exists"] = True

            # Step 2: Load old data
            _LOGGER.info("Step 1/6: Loading old prediction_history.json...")

            def _load_old():
                with open(self.old_prediction_history, "r") as f:
                    return json.load(f)

            # Run in executor to avoid blocking
            loop = asyncio.get_running_loop()
            old_data = await loop.run_in_executor(None, _load_old)

            old_predictions = old_data.get("predictions", [])
            report["old_predictions_count"] = len(old_predictions)
            _LOGGER.info(f"  Loaded {len(old_predictions)} predictions from old file")

            # Step 3: Analyze old data structure
            _LOGGER.info("Step 2/6: Analyzing old data...")
            analysis = self._analyze_old_data(old_predictions)

            report["old_dates_count"] = len(analysis["dates"])
            _LOGGER.info(f"  Found {len(analysis['dates'])} unique dates")
            _LOGGER.info(
                f"  Date range: {analysis['date_range']['min']} to {analysis['date_range']['max']}"
            )
            _LOGGER.info(
                f"  Average predictions per day: {analysis['avg_predictions_per_day']:.1f}"
            )
            _LOGGER.info(f"  Duplicates detected: {analysis['duplicates_count']}")

            # Step 4: Group by date and deduplicate
            _LOGGER.info("Step 3/6: Grouping and deduplicating...")
            grouped = self._group_by_date(old_predictions)
            _LOGGER.info(f"  Grouped into {len(grouped)} days")

            # Step 5: Convert to new structure
            _LOGGER.info("Step 4/6: Converting to new structure...")
            conversion_results = self._convert_to_new_structure(grouped)

            report["converted_dates"] = conversion_results["converted_dates"]
            report["skipped_dates"] = conversion_results["skipped_dates"]

            _LOGGER.info(
                f"  Successfully converted: {len(conversion_results['converted_dates'])} days"
            )
            _LOGGER.info(
                f"  Skipped (insufficient data): {len(conversion_results['skipped_dates'])} days"
            )

            # Step 6: Write new files
            if not dry_run:
                _LOGGER.info("Step 5/6: Writing new files...")

                # Create backup before deletion
                _LOGGER.info(f"  Creating backup: {self.old_prediction_history_backup.name}")
                import shutil

                shutil.copy2(self.old_prediction_history, self.old_prediction_history_backup)

                # Write daily summaries from converted data
                if conversion_results.get("daily_summaries"):
                    _LOGGER.info(f"  Writing daily_summaries.json...")
                    self._write_daily_summaries(conversion_results["daily_summaries"])

                _LOGGER.info("Step 6/6: Cleanup...")
                # Delete old file (v8.6.0: Clean migration - no archive)
                self.old_prediction_history.unlink()
                _LOGGER.info(f"  ✓ Old prediction_history.json deleted (backup kept)")
                _LOGGER.info(f"  ✓ Backup available at: {self.old_prediction_history_backup.name}")

            else:
                _LOGGER.info("Step 5/6: SKIPPED (dry run)")
                _LOGGER.info("Step 6/6: SKIPPED (dry run)")

            report["success"] = True

            _LOGGER.info("")
            _LOGGER.info("=" * 80)
            _LOGGER.info("MIGRATION COMPLETED SUCCESSFULLY")
            _LOGGER.info("=" * 80)

            return report

        except Exception as e:
            _LOGGER.error(f"Migration failed: {e}", exc_info=True)
            report["errors"].append(str(e))
            return report

    def _analyze_old_data(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Analyze old data structure"""
        dates = set()
        dates_list = []

        for p in predictions:
            date = p.get("date")
            if date:
                dates.add(date)
                dates_list.append(date)

        dates_sorted = sorted(dates)

        # Count duplicates
        from collections import Counter

        date_counts = Counter(dates_list)
        duplicates = sum(1 for count in date_counts.values() if count > 1)

        return {
            "dates": dates_sorted,
            "date_range": {
                "min": dates_sorted[0] if dates_sorted else None,
                "max": dates_sorted[-1] if dates_sorted else None,
            },
            "avg_predictions_per_day": len(predictions) / len(dates) if dates else 0,
            "duplicates_count": duplicates,
        }

    def _group_by_date(self, predictions: List[Dict]) -> Dict[str, List[Dict]]:
        """Group predictions by date"""
        grouped = {}

        for p in predictions:
            date = p.get("date")
            if not date:
                continue

            if date not in grouped:
                grouped[date] = []

            grouped[date].append(p)

        return grouped

    def _convert_to_new_structure(self, grouped: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Convert old daily predictions to new structure (creates daily summaries)"""
        converted_dates = []
        skipped_dates = []
        daily_summaries = []

        for date, predictions in grouped.items():
            try:
                # Pick the "best" prediction from duplicates
                best_prediction = self._pick_best_prediction(predictions)

                if not best_prediction:
                    skipped_dates.append(date)
                    continue

                # Create a daily summary from old data
                summary = self._create_summary_from_old_prediction(date, best_prediction)

                if summary:
                    daily_summaries.append(summary)
                    converted_dates.append(date)
                else:
                    skipped_dates.append(date)

            except Exception as e:
                _LOGGER.warning(f"Failed to convert {date}: {e}")
                skipped_dates.append(date)

        return {
            "converted_dates": converted_dates,
            "skipped_dates": skipped_dates,
            "daily_summaries": daily_summaries,
        }

    def _pick_best_prediction(self, predictions: List[Dict]) -> Optional[Dict]:
        """Pick the best prediction from duplicates"""
        if not predictions:
            return None

        # Prefer predictions with actual_value
        with_actual = [p for p in predictions if p.get("actual_value") is not None]

        if with_actual:
            # Pick the one with most complete data
            scored = []
            for p in with_actual:
                score = 0

                # Has actual value
                if p.get("actual_value") is not None:
                    score += 10

                # Has sensor data
                sensor_data = p.get("sensor_data", {})
                if sensor_data:
                    score += sum(1 for v in sensor_data.values() if v is not None)

                # Has weather data
                weather_data = p.get("weather_data", {})
                if weather_data:
                    score += 1

                # Has accuracy
                if p.get("accuracy") is not None and p.get("accuracy") > 0:
                    score += 5

                scored.append((score, p))

            # Return highest scored
            scored.sort(reverse=True, key=lambda x: x[0])
            return scored[0][1]

        # If no actual values, just return the first one
        return predictions[0]

    def _create_summary_from_old_prediction(self, date: str, prediction: Dict) -> Optional[Dict]:
        """Create a daily summary from old prediction format"""
        try:
            dt = datetime.fromisoformat(date)

            # Old prediction has daily totals, not hourly
            predicted_total = prediction.get("predicted_value", 0)
            actual_total = prediction.get("actual_value")

            # Calculate accuracy
            accuracy = 0.0
            if predicted_total > 0 and actual_total is not None:
                accuracy = actual_total / predicted_total * 100

            error = (actual_total - predicted_total) if actual_total is not None else None

            summary = {
                "date": date,
                "day_of_week": dt.weekday(),
                "day_of_year": dt.timetuple().tm_yday,
                "month": dt.month,
                "season": self._get_season(dt.month),
                "week_of_year": dt.isocalendar()[1],
                "overall": {
                    "predicted_total_kwh": round(predicted_total, 2),
                    "actual_total_kwh": (
                        round(actual_total, 2) if actual_total is not None else None
                    ),
                    "accuracy_percent": round(accuracy, 1),
                    "error_kwh": round(error, 2) if error is not None else None,
                    "error_percent": (
                        round((error / predicted_total * 100), 1)
                        if predicted_total > 0 and error is not None
                        else None
                    ),
                    "production_hours": None,
                    "peak_power_w": None,
                    "peak_hour": None,
                    "peak_kwh": None,
                },
                "hourly_stats": {
                    "total_hours_predicted": 0,
                    "hours_with_actual_data": 0,
                    "best_hour": None,
                    "worst_hour": None,
                    "mean_hourly_accuracy": None,
                    "std_hourly_accuracy": None,
                },
                "time_windows": {},
                "weather_analysis": {},
                "patterns": [],
                "ml_metrics": {},
                "recommendations": [],
                "comparison": {},
                "_migrated_from_v1": True,
                "_original_timestamp": prediction.get("timestamp"),
                "_original_source": prediction.get("source"),
            }

            return summary

        except Exception as e:
            _LOGGER.warning(f"Failed to create summary for {date}: {e}")
            return None

    def _get_season(self, month: int) -> str:
        """Get season from month"""
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "autumn"
        else:
            return "winter"

    def _write_daily_summaries(self, summaries: List[Dict]):
        """Write daily summaries to file"""
        data = {"version": "2.0", "last_updated": dt_util.now().isoformat(), "summaries": summaries}

        def _do_write():
            temp_file = self.new_daily_summaries.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.new_daily_summaries)

        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in async context - run in executor
            loop.run_in_executor(None, _do_write)
        except RuntimeError:
            # No event loop - just run directly
            _do_write()

        _LOGGER.info(f"  ✓ Written {len(summaries)} daily summaries")


class DataValidator:
    """Validate data integrity after migration"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.stats_dir = data_dir / "stats"

    async def validate(self) -> Dict[str, Any]:
        """Validate data files"""
        _LOGGER.info("=" * 80)
        _LOGGER.info("DATA VALIDATION")
        _LOGGER.info("=" * 80)

        report = {"valid": True, "files_checked": [], "errors": [], "warnings": []}

        # Check hourly_predictions.json
        hourly_file = self.stats_dir / "hourly_predictions.json"
        if hourly_file.exists():
            _LOGGER.info("Checking hourly_predictions.json...")
            result = await self._validate_hourly_predictions(hourly_file)
            report["files_checked"].append("hourly_predictions.json")

            if not result["valid"]:
                report["valid"] = False
                report["errors"].extend(result["errors"])

            if result["warnings"]:
                report["warnings"].extend(result["warnings"])

            _LOGGER.info(f"  Status: {'✓ VALID' if result['valid'] else '✗ INVALID'}")
            _LOGGER.info(f"  Predictions: {result['predictions_count']}")
            _LOGGER.info(f"  Dates: {result['dates_count']}")
        else:
            _LOGGER.info("hourly_predictions.json not found (will be created by scheduled tasks)")

        # Check daily_summaries.json
        summaries_file = self.stats_dir / "daily_summaries.json"
        if summaries_file.exists():
            _LOGGER.info("Checking daily_summaries.json...")
            result = await self._validate_daily_summaries(summaries_file)
            report["files_checked"].append("daily_summaries.json")

            if not result["valid"]:
                report["valid"] = False
                report["errors"].extend(result["errors"])

            if result["warnings"]:
                report["warnings"].extend(result["warnings"])

            _LOGGER.info(f"  Status: {'✓ VALID' if result['valid'] else '✗ INVALID'}")
            _LOGGER.info(f"  Summaries: {result['summaries_count']}")
        else:
            _LOGGER.info("daily_summaries.json not found (will be created at end of day)")

        _LOGGER.info("=" * 80)
        _LOGGER.info(f"VALIDATION {'PASSED' if report['valid'] else 'FAILED'}")
        _LOGGER.info("=" * 80)

        return report

    async def _validate_hourly_predictions(self, file_path: Path) -> Dict[str, Any]:
        """Validate hourly_predictions.json structure"""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "predictions_count": 0,
            "dates_count": 0,
        }

        try:

            def _load_file():
                with open(file_path, "r") as f:
                    return json.load(f)

            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, _load_file)

            # Check required fields
            if "version" not in data:
                result["errors"].append("Missing 'version' field")
                result["valid"] = False

            if "predictions" not in data:
                result["errors"].append("Missing 'predictions' field")
                result["valid"] = False
                return result

            predictions = data["predictions"]
            result["predictions_count"] = len(predictions)

            # Check unique dates
            dates = set(p.get("target_date") for p in predictions if p.get("target_date"))
            result["dates_count"] = len(dates)

            # Validate sample predictions
            if predictions:
                required_fields = [
                    "id",
                    "target_date",
                    "target_hour",
                    "predicted_kwh",
                    "weather_forecast",
                    "flags",
                ]

                sample = predictions[0]
                for field in required_fields:
                    if field not in sample:
                        result["errors"].append(f"Missing required field in predictions: {field}")
                        result["valid"] = False

            # Check for duplicates
            ids = [p.get("id") for p in predictions]
            if len(ids) != len(set(ids)):
                result["warnings"].append("Duplicate prediction IDs found")

        except json.JSONDecodeError as e:
            result["errors"].append(f"Invalid JSON: {e}")
            result["valid"] = False
        except Exception as e:
            result["errors"].append(f"Validation error: {e}")
            result["valid"] = False

        return result

    async def _validate_daily_summaries(self, file_path: Path) -> Dict[str, Any]:
        """Validate daily_summaries.json structure"""
        result = {"valid": True, "errors": [], "warnings": [], "summaries_count": 0}

        try:

            def _load_file():
                with open(file_path, "r") as f:
                    return json.load(f)

            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, _load_file)

            # Check required fields
            if "version" not in data:
                result["errors"].append("Missing 'version' field")
                result["valid"] = False

            if "summaries" not in data:
                result["errors"].append("Missing 'summaries' field")
                result["valid"] = False
                return result

            summaries = data["summaries"]
            result["summaries_count"] = len(summaries)

            # Validate sample summary
            if summaries:
                required_fields = ["date", "overall", "hourly_stats", "time_windows"]

                sample = summaries[0]
                for field in required_fields:
                    if field not in sample:
                        result["errors"].append(f"Missing required field in summaries: {field}")
                        result["valid"] = False

        except json.JSONDecodeError as e:
            result["errors"].append(f"Invalid JSON: {e}")
            result["valid"] = False
        except Exception as e:
            result["errors"].append(f"Validation error: {e}")
            result["valid"] = False

        return result
