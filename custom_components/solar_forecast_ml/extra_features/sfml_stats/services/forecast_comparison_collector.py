# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""Forecast comparison collector for SFML Stats.

Collects daily forecast values from:
- SFML (Solar Forecast ML) predictions
- Up to 2 external forecast entities (e.g., Solcast, Forecast.Solar)
- Actual production sensor

Stores data for 7-day comparison charts.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import aiofiles

from homeassistant.core import HomeAssistant

from ..const import (
    DOMAIN,
    SFML_STATS_DATA,
    SOLAR_FORECAST_ML_STATS,
    SOLAR_DAILY_SUMMARIES,
    EXTERNAL_FORECASTS_HISTORY,
    FORECAST_COMPARISON_RETENTION_DAYS,
    CONF_SENSOR_SOLAR_YIELD_DAILY,
    CONF_FORECAST_ENTITY_1,
    CONF_FORECAST_ENTITY_2,
    CONF_FORECAST_ENTITY_1_NAME,
    CONF_FORECAST_ENTITY_2_NAME,
    DEFAULT_FORECAST_ENTITY_1_NAME,
    DEFAULT_FORECAST_ENTITY_2_NAME,
)

_LOGGER = logging.getLogger(__name__)


class ForecastComparisonCollector:
    """Collect and store forecast comparison data. @zara"""

    def __init__(self, hass: HomeAssistant, config_path: Path) -> None:
        """Initialize the collector. @zara"""
        self._hass = hass
        self._config_path = config_path
        self._data_path = config_path / SFML_STATS_DATA
        self._history_file = self._data_path / EXTERNAL_FORECASTS_HISTORY
        self._sfml_summaries_file = config_path / SOLAR_FORECAST_ML_STATS / SOLAR_DAILY_SUMMARIES

    def _get_config(self) -> dict[str, Any]:
        """Get current configuration. @zara"""
        entries = self._hass.data.get(DOMAIN, {})
        for entry_id, entry_data in entries.items():
            if isinstance(entry_data, dict) and "config" in entry_data:
                return entry_data["config"]

        config_entries = self._hass.config_entries.async_entries(DOMAIN)
        if config_entries:
            return dict(config_entries[0].data)
        return {}

    def _get_sensor_value(self, entity_id: str | None) -> float | None:
        """Read current value from a sensor. @zara"""
        if not entity_id:
            return None

        state = self._hass.states.get(entity_id)
        if state is None or state.state in ("unknown", "unavailable"):
            return None

        try:
            return float(state.state)
        except (ValueError, TypeError):
            return None

    async def _load_history(self) -> dict[str, Any]:
        """Load existing history file. @zara"""
        if not self._history_file.exists():
            return {
                "days": {},
                "metadata": {
                    "last_updated": None,
                    "retention_days": FORECAST_COMPARISON_RETENTION_DAYS,
                }
            }

        try:
            async with aiofiles.open(self._history_file, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as err:
            _LOGGER.error("Error loading forecast history: %s", err)
            return {
                "days": {},
                "metadata": {
                    "last_updated": None,
                    "retention_days": FORECAST_COMPARISON_RETENTION_DAYS,
                }
            }

    async def _save_history(self, history: dict[str, Any]) -> bool:
        """Save history file. @zara"""
        self._data_path.mkdir(parents=True, exist_ok=True)

        try:
            async with aiofiles.open(self._history_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(history, indent=2, ensure_ascii=False))
            return True
        except Exception as err:
            _LOGGER.error("Error saving forecast history: %s", err)
            return False

    async def _get_sfml_forecast(self, day_str: str) -> float | None:
        """Get SFML forecast for a specific day. @zara

        For today/tomorrow: reads from daily_forecasts.json (live predictions)
        For past days: reads from daily_summaries.json (historical data)
        """
        today_str = date.today().isoformat()
        tomorrow_str = (date.today() + timedelta(days=1)).isoformat()

        # For today or tomorrow, use daily_forecasts.json
        if day_str in (today_str, tomorrow_str):
            forecasts_file = self._config_path / SOLAR_FORECAST_ML_STATS / "daily_forecasts.json"
            if forecasts_file.exists():
                try:
                    async with aiofiles.open(forecasts_file, "r", encoding="utf-8") as f:
                        content = await f.read()
                        data = json.loads(content)

                    today_data = data.get("today", {})

                    if day_str == today_str:
                        forecast_day = today_data.get("forecast_day", {})
                        prediction = forecast_day.get("prediction_kwh")
                        if prediction is not None:
                            _LOGGER.debug("SFML forecast for today from daily_forecasts.json: %.2f kWh", prediction)
                            return prediction

                    elif day_str == tomorrow_str:
                        forecast_tomorrow = today_data.get("forecast_tomorrow", {})
                        prediction = forecast_tomorrow.get("prediction_kwh")
                        if prediction is not None:
                            _LOGGER.debug("SFML forecast for tomorrow from daily_forecasts.json: %.2f kWh", prediction)
                            return prediction

                except Exception as err:
                    _LOGGER.warning("Error reading daily_forecasts.json: %s", err)

        # For past days or fallback, use daily_summaries.json
        if not self._sfml_summaries_file.exists():
            _LOGGER.debug("SFML summaries file not found: %s", self._sfml_summaries_file)
            return None

        try:
            async with aiofiles.open(self._sfml_summaries_file, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)

            summaries = data.get("summaries", [])
            for summary in summaries:
                if summary.get("date") == day_str:
                    overall = summary.get("overall", {})
                    return overall.get("predicted_total_kwh")

            _LOGGER.debug("No SFML forecast found for %s", day_str)
            return None

        except Exception as err:
            _LOGGER.warning("Error reading SFML summaries: %s", err)
            return None

    async def _cleanup_old_entries(self, history: dict[str, Any]) -> None:
        """Remove entries older than retention period. @zara"""
        retention_days = history.get("metadata", {}).get(
            "retention_days", FORECAST_COMPARISON_RETENTION_DAYS
        )
        cutoff_date = date.today() - timedelta(days=retention_days)
        cutoff_str = cutoff_date.isoformat()

        days_to_remove = [
            day_str for day_str in history.get("days", {})
            if day_str < cutoff_str
        ]

        for day_str in days_to_remove:
            del history["days"][day_str]
            _LOGGER.debug("Removed old forecast entry: %s", day_str)

    async def async_collect_morning_forecasts(self) -> bool:
        """Collect forecast values for TODAY at 08:00. @zara

        This runs in the morning to capture forecast values BEFORE they
        change during the day (for rolling forecasts). The actual production
        will be added later in the evening by async_collect_evening_actual().
        """
        config = self._get_config()
        today_str = date.today().isoformat()

        _LOGGER.info("Collecting morning forecast values for %s", today_str)

        # Get entities from config
        external_1_entity = config.get(CONF_FORECAST_ENTITY_1)
        external_1_name = config.get(CONF_FORECAST_ENTITY_1_NAME, DEFAULT_FORECAST_ENTITY_1_NAME)
        external_2_entity = config.get(CONF_FORECAST_ENTITY_2)
        external_2_name = config.get(CONF_FORECAST_ENTITY_2_NAME, DEFAULT_FORECAST_ENTITY_2_NAME)

        # Get current forecast values (before they change during the day)
        external_1_kwh = self._get_sensor_value(external_1_entity) if external_1_entity else None
        external_2_kwh = self._get_sensor_value(external_2_entity) if external_2_entity else None

        # Get SFML forecast for today
        sfml_forecast_kwh = await self._get_sfml_forecast(today_str)

        # Load history
        history = await self._load_history()

        # Create or update entry for today (only forecasts, no actual yet)
        daily_entry: dict[str, Any] = history.get("days", {}).get(today_str, {})

        # Only update if we don't have forecast values yet
        if "sfml_forecast_kwh" not in daily_entry or daily_entry.get("sfml_forecast_kwh") is None:
            daily_entry["sfml_forecast_kwh"] = sfml_forecast_kwh

        daily_entry["timestamp"] = datetime.now().isoformat()

        # Add external forecasts if configured (capture morning values)
        if external_1_entity:
            if "external_1" not in daily_entry:
                daily_entry["external_1"] = {
                    "entity_id": external_1_entity,
                    "name": external_1_name,
                }
            # Only set forecast if not already captured
            if daily_entry["external_1"].get("forecast_kwh") is None:
                daily_entry["external_1"]["forecast_kwh"] = external_1_kwh

        if external_2_entity:
            if "external_2" not in daily_entry:
                daily_entry["external_2"] = {
                    "entity_id": external_2_entity,
                    "name": external_2_name,
                }
            # Only set forecast if not already captured
            if daily_entry["external_2"].get("forecast_kwh") is None:
                daily_entry["external_2"]["forecast_kwh"] = external_2_kwh

        # actual_kwh stays None until evening collection
        if "actual_kwh" not in daily_entry:
            daily_entry["actual_kwh"] = None

        # Save to history
        history["days"][today_str] = daily_entry
        history["metadata"]["last_updated"] = datetime.now().isoformat()

        success = await self._save_history(history)

        if success:
            _LOGGER.info(
                "Morning forecasts saved for %s: SFML=%.2f kWh, Ext1=%s, Ext2=%s",
                today_str,
                sfml_forecast_kwh or 0,
                f"{external_1_kwh:.2f} kWh" if external_1_kwh else "N/A",
                f"{external_2_kwh:.2f} kWh" if external_2_kwh else "N/A",
            )

        return success

    async def async_collect_evening_actual(self) -> bool:
        """Collect actual production for TODAY at 23:50. @zara

        This runs in the evening to capture the final actual production
        for the day and calculate accuracy metrics against the morning forecasts.
        """
        config = self._get_config()
        today_str = date.today().isoformat()

        _LOGGER.info("Collecting evening actual production for %s", today_str)

        # Get actual production
        actual_entity = config.get(CONF_SENSOR_SOLAR_YIELD_DAILY)
        actual_kwh = self._get_sensor_value(actual_entity)

        # Load history
        history = await self._load_history()

        # Get or create entry for today
        daily_entry = history.get("days", {}).get(today_str, {})

        if not daily_entry:
            _LOGGER.warning("No morning forecast data for %s, creating new entry", today_str)
            daily_entry = {"timestamp": datetime.now().isoformat()}

        # Update actual production
        daily_entry["actual_kwh"] = actual_kwh
        daily_entry["timestamp"] = datetime.now().isoformat()

        # Try to fill in missing SFML forecast if not captured in the morning
        if daily_entry.get("sfml_forecast_kwh") is None:
            sfml_forecast = await self._get_sfml_forecast(today_str)
            if sfml_forecast is not None:
                daily_entry["sfml_forecast_kwh"] = sfml_forecast
                _LOGGER.info("Filled missing SFML forecast for %s: %.2f kWh", today_str, sfml_forecast)

        # Calculate accuracy metrics if we have actual and forecast data
        if actual_kwh is not None and actual_kwh > 0:
            sfml_forecast_kwh = daily_entry.get("sfml_forecast_kwh")
            if sfml_forecast_kwh is not None:
                error = abs(sfml_forecast_kwh - actual_kwh)
                daily_entry["sfml_error_kwh"] = round(error, 2)
                daily_entry["sfml_accuracy_percent"] = round(
                    max(0, 100 - (error / actual_kwh * 100)), 1
                )

            if "external_1" in daily_entry:
                ext1_forecast = daily_entry["external_1"].get("forecast_kwh")
                if ext1_forecast is not None:
                    error = abs(ext1_forecast - actual_kwh)
                    daily_entry["external_1"]["error_kwh"] = round(error, 2)
                    daily_entry["external_1"]["accuracy_percent"] = round(
                        max(0, 100 - (error / actual_kwh * 100)), 1
                    )

            if "external_2" in daily_entry:
                ext2_forecast = daily_entry["external_2"].get("forecast_kwh")
                if ext2_forecast is not None:
                    error = abs(ext2_forecast - actual_kwh)
                    daily_entry["external_2"]["error_kwh"] = round(error, 2)
                    daily_entry["external_2"]["accuracy_percent"] = round(
                        max(0, 100 - (error / actual_kwh * 100)), 1
                    )

        # Save to history
        history["days"][today_str] = daily_entry
        history["metadata"]["last_updated"] = datetime.now().isoformat()

        # Cleanup old entries
        await self._cleanup_old_entries(history)

        success = await self._save_history(history)

        if success:
            sfml_forecast = daily_entry.get("sfml_forecast_kwh")
            ext1_forecast = daily_entry.get("external_1", {}).get("forecast_kwh")
            ext2_forecast = daily_entry.get("external_2", {}).get("forecast_kwh")
            _LOGGER.info(
                "Evening actual saved for %s: Actual=%.2f kWh, SFML=%.2f kWh, Ext1=%s, Ext2=%s",
                today_str,
                actual_kwh or 0,
                sfml_forecast or 0,
                f"{ext1_forecast:.2f} kWh" if ext1_forecast else "N/A",
                f"{ext2_forecast:.2f} kWh" if ext2_forecast else "N/A",
            )

        return success

    async def async_get_comparison_data(self, days: int = 7) -> list[dict[str, Any]]:
        """Get forecast comparison data for the last N days. @zara"""
        history = await self._load_history()

        cutoff_date = date.today() - timedelta(days=days - 1)
        result = []

        current = cutoff_date
        while current <= date.today():
            day_str = current.isoformat()
            if day_str in history.get("days", {}):
                day_data = history["days"][day_str]
                result.append({
                    "date": day_str,
                    **day_data
                })
            current = current + timedelta(days=1)

        return result

    async def _get_all_sfml_summaries(self) -> dict[str, dict[str, Any]]:
        """Load all SFML summaries from daily_summaries.json. @zara"""
        if not self._sfml_summaries_file.exists():
            _LOGGER.debug("SFML summaries file not found: %s", self._sfml_summaries_file)
            return {}

        try:
            async with aiofiles.open(self._sfml_summaries_file, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)

            result = {}
            summaries = data.get("summaries", [])
            for summary in summaries:
                day_str = summary.get("date")
                if day_str:
                    overall = summary.get("overall", {})
                    result[day_str] = {
                        "predicted_kwh": overall.get("predicted_total_kwh"),
                        "actual_kwh": overall.get("actual_total_kwh"),
                        "accuracy_percent": overall.get("accuracy_percent"),
                    }

            _LOGGER.debug("Loaded %d SFML summaries", len(result))
            return result

        except Exception as err:
            _LOGGER.warning("Error reading SFML summaries: %s", err)
            return {}

    async def _get_sensor_history_from_recorder(
        self, entity_id: str, days: int
    ) -> dict[str, float | None]:
        """Get historical values from HA Recorder. @zara"""
        result: dict[str, float | None] = {}

        if not entity_id:
            return result

        try:
            from homeassistant.components.recorder import get_instance
            from homeassistant.components.recorder.history import state_changes_during_period

            start_time = datetime.now() - timedelta(days=days)

            # Get state changes from recorder
            states = await get_instance(self._hass).async_add_executor_job(
                state_changes_during_period,
                self._hass,
                start_time,
                None,  # end_time
                entity_id,
                False,  # include_start_time_state
                True,   # significant_changes_only
                1000,   # limit
            )

            if entity_id not in states:
                _LOGGER.debug("No recorder data found for %s", entity_id)
                return result

            # Group by day, take the maximum value per day (end-of-day value for daily sensors)
            daily_values: dict[str, list[float]] = {}

            for state in states[entity_id]:
                if state.state in ("unknown", "unavailable"):
                    continue

                try:
                    value = float(state.state)
                    day_str = state.last_changed.date().isoformat()

                    if day_str not in daily_values:
                        daily_values[day_str] = []
                    daily_values[day_str].append(value)
                except (ValueError, TypeError):
                    continue

            # Take the maximum value per day (for daily kWh sensors)
            for day_str, values in daily_values.items():
                if values:
                    result[day_str] = max(values)

            _LOGGER.debug(
                "Loaded %d days of history for %s", len(result), entity_id
            )

        except ImportError:
            _LOGGER.warning("Recorder not available for historical data")
        except Exception as err:
            _LOGGER.warning("Error reading recorder history for %s: %s", entity_id, err)

        return result

    async def async_collect_historical(self, days: int = 7) -> bool:
        """Collect historical data for the last N days. @zara

        This method is called on initial setup when no data exists.
        It loads:
        - SFML forecasts from daily_summaries.json
        - Actual production from HA Recorder
        - External forecast values from HA Recorder
        """
        config = self._get_config()

        _LOGGER.info("Collecting historical forecast comparison data for last %d days", days)

        # Load all SFML summaries
        sfml_data = await self._get_all_sfml_summaries()

        # Get configured entities
        actual_entity = config.get(CONF_SENSOR_SOLAR_YIELD_DAILY)
        external_1_entity = config.get(CONF_FORECAST_ENTITY_1)
        external_1_name = config.get(CONF_FORECAST_ENTITY_1_NAME, DEFAULT_FORECAST_ENTITY_1_NAME)
        external_2_entity = config.get(CONF_FORECAST_ENTITY_2)
        external_2_name = config.get(CONF_FORECAST_ENTITY_2_NAME, DEFAULT_FORECAST_ENTITY_2_NAME)

        # Get historical data from recorder
        actual_history = await self._get_sensor_history_from_recorder(actual_entity, days)
        external_1_history = await self._get_sensor_history_from_recorder(external_1_entity, days)
        external_2_history = await self._get_sensor_history_from_recorder(external_2_entity, days)

        # Load existing history
        history = await self._load_history()

        # Build entries for each day
        end_date = date.today()
        start_date = end_date - timedelta(days=days - 1)

        current = start_date
        days_added = 0

        while current <= end_date:
            day_str = current.isoformat()

            # Skip if we already have data for this day
            if day_str in history.get("days", {}):
                current = current + timedelta(days=1)
                continue

            # Get values for this day
            sfml_info = sfml_data.get(day_str, {})
            sfml_forecast = sfml_info.get("predicted_kwh")
            actual_kwh = actual_history.get(day_str) or sfml_info.get("actual_kwh")
            external_1_kwh = external_1_history.get(day_str)
            external_2_kwh = external_2_history.get(day_str)

            # Only create entry if we have some data
            if sfml_forecast is not None or actual_kwh is not None or external_1_kwh is not None or external_2_kwh is not None:
                daily_entry: dict[str, Any] = {
                    "actual_kwh": actual_kwh,
                    "sfml_forecast_kwh": sfml_forecast,
                    "timestamp": datetime.now().isoformat(),
                }

                # Add external forecasts if configured
                if external_1_entity:
                    daily_entry["external_1"] = {
                        "entity_id": external_1_entity,
                        "name": external_1_name,
                        "forecast_kwh": external_1_kwh,
                    }

                if external_2_entity:
                    daily_entry["external_2"] = {
                        "entity_id": external_2_entity,
                        "name": external_2_name,
                        "forecast_kwh": external_2_kwh,
                    }

                # Calculate accuracy metrics if we have actual and forecast data
                if actual_kwh is not None and actual_kwh > 0:
                    if sfml_forecast is not None:
                        error = abs(sfml_forecast - actual_kwh)
                        daily_entry["sfml_error_kwh"] = round(error, 2)
                        daily_entry["sfml_accuracy_percent"] = round(
                            max(0, 100 - (error / actual_kwh * 100)), 1
                        )

                    if external_1_kwh is not None and "external_1" in daily_entry:
                        error = abs(external_1_kwh - actual_kwh)
                        daily_entry["external_1"]["error_kwh"] = round(error, 2)
                        daily_entry["external_1"]["accuracy_percent"] = round(
                            max(0, 100 - (error / actual_kwh * 100)), 1
                        )

                    if external_2_kwh is not None and "external_2" in daily_entry:
                        error = abs(external_2_kwh - actual_kwh)
                        daily_entry["external_2"]["error_kwh"] = round(error, 2)
                        daily_entry["external_2"]["accuracy_percent"] = round(
                            max(0, 100 - (error / actual_kwh * 100)), 1
                        )

                history["days"][day_str] = daily_entry
                days_added += 1

            current = current + timedelta(days=1)

        # Update metadata
        history["metadata"]["last_updated"] = datetime.now().isoformat()

        # Save history
        success = await self._save_history(history)

        if success:
            _LOGGER.info(
                "Historical forecast comparison data collected: %d days added",
                days_added
            )

        return success

    async def async_repair_missing_sfml_forecasts(self, days: int = 7) -> int:
        """Repair missing SFML forecasts from daily_summaries.json. @zara

        This method fills in any missing sfml_forecast_kwh values by looking
        them up in the SFML daily_summaries.json file.

        Returns the number of entries repaired.
        """
        _LOGGER.info("Repairing missing SFML forecasts for last %d days", days)

        # Load all SFML summaries
        sfml_data = await self._get_all_sfml_summaries()

        # Load history
        history = await self._load_history()

        repaired_count = 0
        end_date = date.today()
        start_date = end_date - timedelta(days=days - 1)

        current = start_date
        while current <= end_date:
            day_str = current.isoformat()

            if day_str in history.get("days", {}):
                daily_entry = history["days"][day_str]

                # Check if SFML forecast is missing
                if daily_entry.get("sfml_forecast_kwh") is None:
                    # Try to get from SFML summaries
                    sfml_info = sfml_data.get(day_str, {})
                    sfml_forecast = sfml_info.get("predicted_kwh")

                    if sfml_forecast is not None:
                        daily_entry["sfml_forecast_kwh"] = sfml_forecast
                        _LOGGER.info("Repaired SFML forecast for %s: %.2f kWh", day_str, sfml_forecast)

                        # Recalculate accuracy if we have actual data
                        actual_kwh = daily_entry.get("actual_kwh")
                        if actual_kwh is not None and actual_kwh > 0:
                            error = abs(sfml_forecast - actual_kwh)
                            daily_entry["sfml_error_kwh"] = round(error, 2)
                            daily_entry["sfml_accuracy_percent"] = round(
                                max(0, 100 - (error / actual_kwh * 100)), 1
                            )

                        history["days"][day_str] = daily_entry
                        repaired_count += 1

            current = current + timedelta(days=1)

        if repaired_count > 0:
            history["metadata"]["last_updated"] = datetime.now().isoformat()
            await self._save_history(history)
            _LOGGER.info("Repaired %d missing SFML forecasts", repaired_count)

        return repaired_count
