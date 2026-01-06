# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Hourly billing aggregator for SFML Stats."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiofiles

from homeassistant.core import HomeAssistant

from ..const import (
    DOMAIN,
    SFML_STATS_DATA,
    HOURLY_BILLING_HISTORY,
    CONF_SENSOR_SMARTMETER_IMPORT,
    CONF_SENSOR_SMARTMETER_EXPORT,
    CONF_SENSOR_SMARTMETER_IMPORT_KWH,
    CONF_SENSOR_SMARTMETER_EXPORT_KWH,
    CONF_SENSOR_PRICE_TOTAL,
    CONF_SENSOR_GRID_TO_HOUSE,
    CONF_SENSOR_GRID_TO_BATTERY,
    CONF_SENSOR_SOLAR_POWER,
    CONF_SENSOR_SOLAR_TO_HOUSE,
    CONF_SENSOR_SOLAR_TO_BATTERY,
    CONF_SENSOR_BATTERY_TO_HOUSE,
    CONF_SENSOR_HOME_CONSUMPTION,
    CONF_BILLING_PRICE_MODE,
    CONF_BILLING_FIXED_PRICE,
    CONF_BILLING_START_DAY,
    CONF_BILLING_START_MONTH,
    PRICE_MODE_FIXED,
    DEFAULT_BILLING_FIXED_PRICE,
)

_LOGGER = logging.getLogger(__name__)


class HourlyBillingAggregator:
    """Aggregates hourly energy values and calculates costs."""

    def __init__(self, hass: HomeAssistant, config_path: Path) -> None:
        """Initialize the aggregator."""
        self._hass = hass
        self._config_path = config_path
        self._data_path = config_path / SFML_STATS_DATA
        self._history_file = self._data_path / HOURLY_BILLING_HISTORY

    def _get_config(self) -> dict[str, Any]:
        """Get the current configuration."""
        entries = self._hass.data.get(DOMAIN, {})
        for entry_id, entry_data in entries.items():
            if isinstance(entry_data, dict) and "config" in entry_data:
                return entry_data["config"]

        config_entries = self._hass.config_entries.async_entries(DOMAIN)
        if config_entries:
            return dict(config_entries[0].data)
        return {}

    def _get_sensor_value(self, entity_id: str | None) -> float | None:
        """Read the current value of a sensor."""
        if not entity_id:
            return None

        state = self._hass.states.get(entity_id)
        if state is None or state.state in ("unknown", "unavailable"):
            return None

        try:
            return float(state.state)
        except (ValueError, TypeError):
            return None

    async def _calculate_kwh_from_recorder(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> float:
        """Calculate kWh from watt values in recorder."""
        if not entity_id:
            return 0.0

        hours = (end_time - start_time).total_seconds() / 3600

        try:
            from homeassistant.components.recorder import get_instance
            from homeassistant.components.recorder.history import state_changes_during_period

            instance = get_instance(self._hass)

            states = await instance.async_add_executor_job(
                state_changes_during_period,
                self._hass,
                start_time,
                end_time,
                [entity_id],
                True,
                False,
                None,
            )

            entity_states = states.get(entity_id, [])

            _LOGGER.debug(
                "Recorder for %s: %d states from %s to %s",
                entity_id, len(entity_states), start_time, end_time
            )

            if len(entity_states) >= 1:
                total_wh = 0.0

                if len(entity_states) == 1:
                    try:
                        watts = float(entity_states[0].state)
                        if watts > 0:
                            kwh = (watts * hours) / 1000
                            _LOGGER.debug(
                                "%s: 1 state with %.1f W -> %.4f kWh",
                                entity_id, watts, kwh
                            )
                            return kwh
                    except (ValueError, TypeError):
                        pass
                else:
                    for i in range(1, len(entity_states)):
                        prev_state = entity_states[i - 1]
                        curr_state = entity_states[i]

                        delta_seconds = (
                            curr_state.last_changed - prev_state.last_changed
                        ).total_seconds()
                        delta_hours = delta_seconds / 3600

                        try:
                            prev_watts = float(prev_state.state)
                            curr_watts = float(curr_state.state)

                            prev_watts = max(0, prev_watts)
                            curr_watts = max(0, curr_watts)

                            avg_watts = (prev_watts + curr_watts) / 2
                            total_wh += avg_watts * delta_hours

                        except (ValueError, TypeError):
                            continue

                    if total_wh > 0:
                        kwh = total_wh / 1000
                        _LOGGER.debug(
                            "%s: Calculated %.4f kWh from %d states",
                            entity_id, kwh, len(entity_states)
                        )
                        return kwh

        except ImportError as err:
            _LOGGER.debug("Recorder not available: %s", err)
        except Exception as err:
            _LOGGER.debug(
                "Recorder query for %s failed: %s",
                entity_id, err
            )

        current = self._get_sensor_value(entity_id)
        if current is not None and current > 0:
            kwh = (current * hours) / 1000
            _LOGGER.info(
                "%s: Fallback to current value %.1f W x %.2fh = %.4f kWh",
                entity_id, current, hours, kwh
            )
            return kwh

        _LOGGER.debug("%s: No value available, returning 0", entity_id)
        return 0.0

    async def _calculate_kwh_diff_from_total(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> float | None:
        """Calculate kWh difference from a total_increasing kWh sensor."""
        if not entity_id:
            return None

        try:
            from homeassistant.components.recorder import get_instance
            from homeassistant.components.recorder.history import state_changes_during_period

            instance = get_instance(self._hass)

            states = await instance.async_add_executor_job(
                state_changes_during_period,
                self._hass,
                start_time,
                end_time,
                [entity_id],
                True,
                False,
                None,
            )

            entity_states = states.get(entity_id, [])

            if len(entity_states) >= 2:
                try:
                    start_kwh = float(entity_states[0].state)
                    end_kwh = float(entity_states[-1].state)

                    diff = end_kwh - start_kwh

                    if diff >= -0.001:
                        diff = max(0, diff)
                        _LOGGER.debug(
                            "%s: kWh difference %.4f (%.4f -> %.4f)",
                            entity_id, diff, start_kwh, end_kwh
                        )
                        return diff

                except (ValueError, TypeError):
                    pass

            elif len(entity_states) == 1:
                _LOGGER.debug(
                    "%s: Only 1 state in period, cannot calculate difference",
                    entity_id
                )

        except ImportError:
            _LOGGER.debug("Recorder not available for kWh diff")
        except Exception as err:
            _LOGGER.debug("Error in kWh diff for %s: %s", entity_id, err)

        return None

    async def _load_history(self) -> dict[str, Any]:
        """Load the existing history file."""
        if not self._history_file.exists():
            return {
                "billing_period": {},
                "hours": {},
                "totals": {
                    "grid_import_kwh": 0,
                    "grid_import_cost_eur": 0,
                    "grid_export_kwh": 0,
                    "solar_yield_kwh": 0,
                    "self_consumption_kwh": 0,
                },
                "last_updated": None,
            }

        try:
            async with aiofiles.open(self._history_file, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as err:
            _LOGGER.error("Error loading billing history: %s", err)
            return {
                "billing_period": {},
                "hours": {},
                "totals": {},
                "last_updated": None,
            }

    async def _save_history(self, history: dict[str, Any]) -> bool:
        """Save the history file."""
        self._data_path.mkdir(parents=True, exist_ok=True)

        try:
            async with aiofiles.open(self._history_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(history, indent=2, ensure_ascii=False))
            return True
        except Exception as err:
            _LOGGER.error("Error saving billing history: %s", err)
            return False

    async def async_aggregate_hourly(self) -> bool:
        """Aggregate the last hour and calculate costs."""
        config = self._get_config()

        now = datetime.now()
        end_time = now.replace(minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(hours=1)
        hour_key = start_time.strftime("%Y-%m-%dT%H:00")

        _LOGGER.info(
            "Starting hourly billing aggregation for %s",
            hour_key
        )

        smartmeter_import_w = config.get(CONF_SENSOR_SMARTMETER_IMPORT)
        smartmeter_export_w = config.get(CONF_SENSOR_SMARTMETER_EXPORT)

        smartmeter_import_kwh = config.get(CONF_SENSOR_SMARTMETER_IMPORT_KWH)
        smartmeter_export_kwh = config.get(CONF_SENSOR_SMARTMETER_EXPORT_KWH)

        _LOGGER.debug(
            "Config: import_w=%s, export_w=%s, import_kwh=%s, export_kwh=%s",
            smartmeter_import_w, smartmeter_export_w,
            smartmeter_import_kwh, smartmeter_export_kwh
        )

        if not smartmeter_import_w and not smartmeter_import_kwh:
            _LOGGER.warning(
                "No smartmeter import sensor configured - billing aggregation skipped"
            )
            return False

        grid_import_kwh = 0.0
        data_source = "none"

        if smartmeter_import_kwh:
            diff = await self._calculate_kwh_diff_from_total(
                smartmeter_import_kwh, start_time, end_time
            )
            if diff is not None:
                grid_import_kwh = diff
                data_source = "kwh_sensor"
                _LOGGER.info(
                    "Grid import from kWh sensor: %.4f kWh",
                    grid_import_kwh
                )

        if grid_import_kwh == 0.0 and smartmeter_import_w:
            grid_import_kwh = await self._calculate_kwh_from_recorder(
                smartmeter_import_w, start_time, end_time
            )
            if grid_import_kwh > 0:
                data_source = "watt_recorder"
                _LOGGER.info(
                    "Grid import from watt recorder: %.4f kWh",
                    grid_import_kwh
                )
            else:
                current_w = self._get_sensor_value(smartmeter_import_w)
                _LOGGER.debug(
                    "Smartmeter import '%s' current: %s W",
                    smartmeter_import_w, current_w
                )

        grid_export_kwh = 0.0

        if smartmeter_export_kwh:
            diff = await self._calculate_kwh_diff_from_total(
                smartmeter_export_kwh, start_time, end_time
            )
            if diff is not None:
                grid_export_kwh = diff
        elif smartmeter_export_w:
            grid_export_kwh = await self._calculate_kwh_from_recorder(
                smartmeter_export_w, start_time, end_time
            )

        grid_to_house_kwh = await self._calculate_kwh_from_recorder(
            config.get(CONF_SENSOR_GRID_TO_HOUSE), start_time, end_time
        )
        grid_to_battery_kwh = await self._calculate_kwh_from_recorder(
            config.get(CONF_SENSOR_GRID_TO_BATTERY), start_time, end_time
        )
        solar_power_kwh = await self._calculate_kwh_from_recorder(
            config.get(CONF_SENSOR_SOLAR_POWER), start_time, end_time
        )
        solar_to_house_kwh = await self._calculate_kwh_from_recorder(
            config.get(CONF_SENSOR_SOLAR_TO_HOUSE), start_time, end_time
        )
        solar_to_battery_kwh = await self._calculate_kwh_from_recorder(
            config.get(CONF_SENSOR_SOLAR_TO_BATTERY), start_time, end_time
        )
        battery_to_house_kwh = await self._calculate_kwh_from_recorder(
            config.get(CONF_SENSOR_BATTERY_TO_HOUSE), start_time, end_time
        )
        home_consumption_kwh = await self._calculate_kwh_from_recorder(
            config.get(CONF_SENSOR_HOME_CONSUMPTION), start_time, end_time
        )

        price_mode = config.get(CONF_BILLING_PRICE_MODE, "dynamic")
        if price_mode == PRICE_MODE_FIXED:
            current_price = config.get(
                CONF_BILLING_FIXED_PRICE, DEFAULT_BILLING_FIXED_PRICE
            )
        else:
            current_price = self._get_sensor_value(
                config.get(CONF_SENSOR_PRICE_TOTAL)
            )
            if current_price is None:
                current_price = DEFAULT_BILLING_FIXED_PRICE
                _LOGGER.warning(
                    "No electricity price available, using default: %.2f ct/kWh",
                    current_price
                )

        grid_import_cost_eur = (grid_import_kwh * current_price) / 100

        hourly_data = {
            "grid_import_kwh": round(grid_import_kwh, 4),
            "grid_import_cost_ct": round(grid_import_kwh * current_price, 2),
            "grid_export_kwh": round(grid_export_kwh, 4),
            "price_ct_kwh": round(current_price, 2),
            "grid_to_house_kwh": round(grid_to_house_kwh, 4),
            "grid_to_battery_kwh": round(grid_to_battery_kwh, 4),
            "solar_yield_kwh": round(solar_power_kwh, 4),
            "solar_to_house_kwh": round(solar_to_house_kwh, 4),
            "solar_to_battery_kwh": round(solar_to_battery_kwh, 4),
            "battery_to_house_kwh": round(battery_to_house_kwh, 4),
            "home_consumption_kwh": round(home_consumption_kwh, 4),
            "timestamp": now.isoformat(),
            "data_source": data_source,
        }

        history = await self._load_history()

        billing_start_day = config.get(CONF_BILLING_START_DAY, 1)
        billing_start_month = config.get(CONF_BILLING_START_MONTH, 1)
        history["billing_period"] = {
            "start_day": billing_start_day,
            "start_month": billing_start_month,
        }

        history["hours"][hour_key] = hourly_data
        history["last_updated"] = now.isoformat()

        await self._recalculate_totals(history, config)

        success = await self._save_history(history)

        if success:
            _LOGGER.info(
                "Billing aggregation saved: Import=%.3f kWh (%.2f ct), "
                "Export=%.3f kWh, Price=%.2f ct/kWh",
                grid_import_kwh,
                grid_import_kwh * current_price,
                grid_export_kwh,
                current_price,
            )

        return success

    async def _recalculate_totals(
        self,
        history: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Recalculate totals for the billing period."""
        from datetime import date

        billing_start_day = config.get(CONF_BILLING_START_DAY, 1)
        billing_start_month = config.get(CONF_BILLING_START_MONTH, 1)

        if isinstance(billing_start_day, str):
            billing_start_day = int(billing_start_day)
        if isinstance(billing_start_month, str):
            billing_start_month = int(billing_start_month)

        today = date.today()
        current_year = today.year
        billing_start = date(current_year, billing_start_month, billing_start_day)

        if billing_start > today:
            billing_start = date(
                current_year - 1, billing_start_month, billing_start_day
            )

        billing_start_str = billing_start.isoformat()

        totals = {
            "grid_import_kwh": 0,
            "grid_import_cost_eur": 0,
            "grid_export_kwh": 0,
            "solar_yield_kwh": 0,
            "solar_to_house_kwh": 0,
            "solar_to_battery_kwh": 0,
            "battery_to_house_kwh": 0,
            "grid_to_house_kwh": 0,
            "grid_to_battery_kwh": 0,
            "home_consumption_kwh": 0,
            "hours_count": 0,
            "avg_price_ct": 0,
        }

        price_sum = 0

        for hour_key, hour_data in history.get("hours", {}).items():
            hour_date = hour_key[:10]
            if hour_date >= billing_start_str:
                totals["grid_import_kwh"] += hour_data.get("grid_import_kwh", 0)
                totals["grid_import_cost_eur"] += (
                    hour_data.get("grid_import_cost_ct", 0) / 100
                )
                totals["grid_export_kwh"] += hour_data.get("grid_export_kwh", 0)
                totals["solar_yield_kwh"] += hour_data.get("solar_yield_kwh", 0)
                totals["solar_to_house_kwh"] += hour_data.get("solar_to_house_kwh", 0)
                totals["solar_to_battery_kwh"] += hour_data.get(
                    "solar_to_battery_kwh", 0
                )
                totals["battery_to_house_kwh"] += hour_data.get(
                    "battery_to_house_kwh", 0
                )
                totals["grid_to_house_kwh"] += hour_data.get("grid_to_house_kwh", 0)
                totals["grid_to_battery_kwh"] += hour_data.get(
                    "grid_to_battery_kwh", 0
                )
                totals["home_consumption_kwh"] += hour_data.get(
                    "home_consumption_kwh", 0
                )
                totals["hours_count"] += 1
                price_sum += hour_data.get("price_ct_kwh", 0)

        if totals["hours_count"] > 0:
            totals["avg_price_ct"] = round(price_sum / totals["hours_count"], 2)

        for key in totals:
            if isinstance(totals[key], float):
                totals[key] = round(totals[key], 4)

        totals["self_consumption_kwh"] = round(
            totals["solar_yield_kwh"] - totals["grid_export_kwh"], 4
        )

        total_consumption = (
            totals["grid_import_kwh"] + totals["self_consumption_kwh"]
        )
        if total_consumption > 0:
            totals["autarkie_percent"] = round(
                (totals["self_consumption_kwh"] / total_consumption) * 100, 1
            )
        else:
            totals["autarkie_percent"] = 0

        if totals["avg_price_ct"] > 0:
            totals["savings_eur"] = round(
                (totals["self_consumption_kwh"] * totals["avg_price_ct"]) / 100, 2
            )
        else:
            totals["savings_eur"] = 0

        history["totals"] = totals
        history["billing_period"]["start"] = billing_start_str

    async def async_cleanup_old_data(self, keep_days: int = 400) -> int:
        """Remove old hourly data older than keep_days."""
        history = await self._load_history()
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat()[:10]

        deleted = 0
        hours_to_keep = {}

        for hour_key, hour_data in history.get("hours", {}).items():
            hour_date = hour_key[:10]
            if hour_date >= cutoff:
                hours_to_keep[hour_key] = hour_data
            else:
                deleted += 1

        if deleted > 0:
            history["hours"] = hours_to_keep
            await self._save_history(history)
            _LOGGER.info("Billing history: %d old entries deleted", deleted)

        return deleted
