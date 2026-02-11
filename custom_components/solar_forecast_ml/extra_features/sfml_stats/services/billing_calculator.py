# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Billing calculator using SFML database. @zara"""
from __future__ import annotations

import calendar
import csv
import io
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

import aiofiles
import aiosqlite

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

from ..const import (
    DOMAIN,
    CONF_SENSOR_PRICE_TOTAL,
    CONF_BILLING_PRICE_MODE,
    CONF_BILLING_FIXED_PRICE,
    CONF_BILLING_START_DAY,
    CONF_BILLING_START_MONTH,
    PRICE_MODE_FIXED,
    PRICE_MODE_NONE,
    DEFAULT_BILLING_FIXED_PRICE,
    BILLING_CACHE_TTL_SECONDS,
)

_LOGGER = logging.getLogger(__name__)


class BillingCalculator:
    """Calculate energy balance from SFML database. @zara"""

    def __init__(
        self,
        hass: HomeAssistant,
        config_path: Path,
        entry_data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the calculator. @zara"""
        self._hass = hass
        self._config_path = config_path
        self._entry_data = entry_data
        self._db_path = config_path / "solar_forecast_ml" / "solar_forecast.db"
        self._billing_cache: dict[str, Any] | None = None
        self._cache_timestamp: datetime | None = None
        self._cache_ttl_seconds = BILLING_CACHE_TTL_SECONDS

    @asynccontextmanager
    async def _get_db(self) -> AsyncIterator[aiosqlite.Connection]:
        """Get DB connection via manager with direct fallback. @zara"""
        from ..storage.db_connection_manager import get_manager
        manager = get_manager()
        if manager is not None and manager.is_connected:
            yield await manager.get_connection()
            return
        async with aiosqlite.connect(str(self._db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            yield conn

    @property
    def is_db_available(self) -> bool:
        """Check if SFML database is available. @zara"""
        return self._db_path.exists()

    def update_config(self, new_config: dict[str, Any]) -> None:
        """Update cached configuration and invalidate billing cache. @zara"""
        self._entry_data = new_config
        self._billing_cache = None
        self._cache_timestamp = None
        _LOGGER.debug("BillingCalculator config updated, cache invalidated")

    def _get_config(self) -> dict[str, Any]:
        """Get current configuration. @zara"""
        if self._entry_data:
            return self._entry_data

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

    def _get_billing_start_date(self, config: dict[str, Any]) -> date:
        """Calculate start date of current billing period. @zara"""
        billing_start_day = config.get(CONF_BILLING_START_DAY, 1)
        billing_start_month = config.get(CONF_BILLING_START_MONTH, 1)

        if isinstance(billing_start_day, str):
            billing_start_day = int(billing_start_day)
        if isinstance(billing_start_month, str):
            billing_start_month = int(billing_start_month)

        billing_start_day = max(1, billing_start_day or 1)
        billing_start_month = max(1, min(12, billing_start_month or 1))

        today = date.today()
        current_year = today.year

        max_day = calendar.monthrange(current_year, billing_start_month)[1]
        billing_start_day = min(billing_start_day, max_day)

        billing_start = date(current_year, billing_start_month, billing_start_day)

        if billing_start > today:
            max_day_prev = calendar.monthrange(current_year - 1, billing_start_month)[1]
            billing_start_day = min(billing_start_day, max_day_prev)
            billing_start = date(current_year - 1, billing_start_month, billing_start_day)

        return billing_start

    async def _get_billing_data_from_db(self, billing_start: date) -> dict[str, float]:
        """Get summed energy data from stats_daily_energy since billing start. @zara"""
        if not self.is_db_available:
            _LOGGER.warning("SFML database not available at %s", self._db_path)
            return {}

        billing_start_str = billing_start.isoformat()

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT
                        COALESCE(SUM(solar_yield_kwh), 0) as solar_total_kwh,
                        COALESCE(SUM(solar_to_house_kwh), 0) as solar_to_house_kwh,
                        COALESCE(SUM(solar_to_battery_kwh), 0) as solar_to_battery_kwh,
                        COALESCE(SUM(battery_to_house_kwh), 0) as battery_to_house_kwh,
                        COALESCE(SUM(grid_to_house_kwh), 0) as grid_to_house_kwh,
                        COALESCE(SUM(home_consumption_kwh), 0) as home_consumption_kwh,
                        COUNT(*) as days_with_data
                    FROM stats_daily_energy
                    WHERE date >= ?
                """, (billing_start_str,)) as cursor:
                    row = await cursor.fetchone()

                if row:
                    return {
                        "solar_total_kwh": row["solar_total_kwh"] or 0.0,
                        "solar_to_house_kwh": row["solar_to_house_kwh"] or 0.0,
                        "solar_to_battery_kwh": row["solar_to_battery_kwh"] or 0.0,
                        "battery_to_house_kwh": row["battery_to_house_kwh"] or 0.0,
                        "grid_to_house_kwh": row["grid_to_house_kwh"] or 0.0,
                        "home_consumption_kwh": row["home_consumption_kwh"] or 0.0,
                        "days_with_data": row["days_with_data"] or 0,
                    }

                return {}

        except Exception as err:
            _LOGGER.error("Error reading billing data from DB: %s", err)
            return {}

    async def _get_smartmeter_data_from_db(self, billing_start: date) -> dict[str, float]:
        """Get smartmeter data from stats_daily_energy if available. @zara"""
        if not self.is_db_available:
            return {}

        billing_start_str = billing_start.isoformat()

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT
                        COALESCE(SUM(smartmeter_import_kwh), 0) as smartmeter_import_kwh,
                        COALESCE(SUM(smartmeter_export_kwh), 0) as smartmeter_export_kwh,
                        COALESCE(SUM(grid_to_battery_kwh), 0) as grid_to_battery_kwh
                    FROM stats_daily_energy
                    WHERE date >= ?
                """, (billing_start_str,)) as cursor:
                    row = await cursor.fetchone()

                if row:
                    return {
                        "smartmeter_import_kwh": row["smartmeter_import_kwh"] or 0.0,
                        "smartmeter_export_kwh": row["smartmeter_export_kwh"] or 0.0,
                        "grid_to_battery_kwh": row["grid_to_battery_kwh"] or 0.0,
                    }

                return {}

        except Exception:
            return {}

    async def _get_consumer_data_from_db(self, billing_start: date) -> dict[str, float]:
        """Get consumer energy data from stats_daily_energy. @zara"""
        if not self.is_db_available:
            return {}

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT
                        COALESCE(SUM(consumer_heatpump_kwh), 0) as consumer_heatpump_kwh,
                        COALESCE(SUM(consumer_heatingrod_kwh), 0) as consumer_heatingrod_kwh,
                        COALESCE(SUM(consumer_wallbox_kwh), 0) as consumer_wallbox_kwh
                    FROM stats_daily_energy
                    WHERE date >= ?
                """, (billing_start.isoformat(),)) as cursor:
                    row = await cursor.fetchone()

                if row:
                    return {
                        "consumer_heatpump_kwh": row["consumer_heatpump_kwh"] or 0.0,
                        "consumer_heatingrod_kwh": row["consumer_heatingrod_kwh"] or 0.0,
                        "consumer_wallbox_kwh": row["consumer_wallbox_kwh"] or 0.0,
                    }

                return {}

        except Exception:
            return {}

    async def async_ensure_baselines(self) -> dict[str, Any]:
        """Compatibility stub - no longer needed. @zara"""
        return {}

    async def async_calculate_billing(self) -> dict[str, Any]:
        """Calculate current energy balance from database. @zara"""
        if self._billing_cache is not None and self._cache_timestamp is not None:
            cache_age = (datetime.now() - self._cache_timestamp).total_seconds()
            if cache_age < self._cache_ttl_seconds:
                _LOGGER.debug("Returning cached billing result (age: %.1fs)", cache_age)
                return self._billing_cache

        _LOGGER.debug("Calculating billing data from database")

        config = self._get_config()
        billing_start = self._get_billing_start_date(config)
        today = date.today()

        db_data = await self._get_billing_data_from_db(billing_start)
        smartmeter_data = await self._get_smartmeter_data_from_db(billing_start)
        consumer_data = await self._get_consumer_data_from_db(billing_start)

        if not db_data:
            _LOGGER.warning("No billing data available in database")
            return {
                "success": False,
                "error": "No data available in database",
                "data_available": False,
            }

        solar_total = db_data.get("solar_total_kwh", 0.0)
        solar_to_house = db_data.get("solar_to_house_kwh", 0.0)
        solar_to_battery = db_data.get("solar_to_battery_kwh", 0.0)
        battery_to_house = db_data.get("battery_to_house_kwh", 0.0)
        grid_to_house = db_data.get("grid_to_house_kwh", 0.0)
        home_consumption = db_data.get("home_consumption_kwh", 0.0)
        days_with_data = db_data.get("days_with_data", 0)

        smartmeter_import = smartmeter_data.get("smartmeter_import_kwh", 0.0)
        smartmeter_export = smartmeter_data.get("smartmeter_export_kwh", 0.0)
        grid_to_battery = smartmeter_data.get("grid_to_battery_kwh", 0.0)

        wr_to_house = solar_to_house + battery_to_house

        battery_total_charge = solar_to_battery + grid_to_battery

        if smartmeter_export == 0 and solar_total > 0:
            smartmeter_export = max(0, solar_total - solar_to_house - solar_to_battery)

        grid_total = grid_to_house + grid_to_battery
        if grid_total > smartmeter_import:
            smartmeter_import = grid_total

        if home_consumption > 0:
            autarkie = min(100, (wr_to_house / home_consumption) * 100)
        else:
            autarkie = 0.0

        price_mode = config.get(CONF_BILLING_PRICE_MODE, "dynamic")
        if price_mode == PRICE_MODE_NONE:
            avg_price = 0.0
            grid_cost_eur = 0.0
            savings_eur = 0.0
        elif price_mode == PRICE_MODE_FIXED:
            avg_price = config.get(CONF_BILLING_FIXED_PRICE, DEFAULT_BILLING_FIXED_PRICE)
            grid_cost_eur = (smartmeter_import * avg_price) / 100
            savings_eur = (wr_to_house * avg_price) / 100
        else:
            dynamic = await self._get_dynamic_costs_from_db(billing_start)
            grid_cost_eur = dynamic["grid_cost_eur"]
            savings_eur = dynamic["savings_eur"]
            avg_price = dynamic["weighted_avg_price_ct"]

        days_elapsed = (today - billing_start).days + 1

        next_year = billing_start.year + 1
        target_month = billing_start.month
        target_day = billing_start.day

        max_day_next_year = calendar.monthrange(next_year, target_month)[1]
        target_day = min(target_day, max_day_next_year)

        billing_end_theoretical = date(next_year, target_month, target_day) - timedelta(days=1)
        days_total = (billing_end_theoretical - billing_start).days + 1

        result = {
            "success": True,
            "data_available": days_with_data > 0,
            "data_source": "stats_daily_energy",
            "period": {
                "start": billing_start.isoformat(),
                "end": today.isoformat(),
                "end_theoretical": billing_end_theoretical.isoformat(),
                "days_elapsed": days_elapsed,
                "days_total": days_total,
                "days_with_data": days_with_data,
                "progress_percent": round((days_elapsed / days_total) * 100, 1),
            },
            "household": {
                "total_kwh": round(home_consumption, 2),
                "from_solar_kwh": round(solar_to_house, 2),
                "from_battery_kwh": round(battery_to_house, 2),
                "from_grid_kwh": round(grid_to_house, 2),
            },
            "battery": {
                "total_charge_kwh": round(battery_total_charge, 2),
                "from_solar_kwh": round(solar_to_battery, 2),
                "from_grid_kwh": round(grid_to_battery, 2),
            },
            "solar": {
                "total_kwh": round(solar_total, 2),
                "to_house_kwh": round(solar_to_house, 2),
                "to_battery_kwh": round(solar_to_battery, 2),
            },
            "grid": {
                "total_import_kwh": round(smartmeter_import, 2),
                "to_house_kwh": round(grid_to_house, 2),
                "to_battery_kwh": round(grid_to_battery, 2),
                "export_kwh": round(smartmeter_export, 2),
            },
            "finance": {
                "grid_cost_eur": round(grid_cost_eur, 2),
                "savings_eur": round(savings_eur, 2),
                "avg_price_ct": round(avg_price, 2),
            },
            "autarkie_percent": round(autarkie, 1),
            "config": {
                "billing_start_day": config.get(CONF_BILLING_START_DAY),
                "billing_start_month": config.get(CONF_BILLING_START_MONTH),
                "price_mode": price_mode,
            },
        }

        consumer_hp_kwh = consumer_data.get("consumer_heatpump_kwh", 0.0)
        consumer_hr_kwh = consumer_data.get("consumer_heatingrod_kwh", 0.0)
        consumer_wb_kwh = consumer_data.get("consumer_wallbox_kwh", 0.0)

        if price_mode == PRICE_MODE_NONE:
            consumer_price = 0.0
        else:
            consumer_price = avg_price

        result["consumers"] = {
            "heatpump": {
                "total_kwh": round(consumer_hp_kwh, 2),
                "cost_eur": round((consumer_hp_kwh * consumer_price) / 100, 2),
            },
            "heatingrod": {
                "total_kwh": round(consumer_hr_kwh, 2),
                "cost_eur": round((consumer_hr_kwh * consumer_price) / 100, 2),
            },
            "wallbox": {
                "total_kwh": round(consumer_wb_kwh, 2),
                "cost_eur": round((consumer_wb_kwh * consumer_price) / 100, 2),
            },
            "total_kwh": round(consumer_hp_kwh + consumer_hr_kwh + consumer_wb_kwh, 2),
            "total_cost_eur": round(
                ((consumer_hp_kwh + consumer_hr_kwh + consumer_wb_kwh) * consumer_price) / 100,
                2
            ),
            "price_ct_kwh": round(consumer_price, 2),
        }

        _LOGGER.debug(
            "Billing calculated: consumption=%.2f kWh, autarkie=%.1f%%, cost=%.2f EUR",
            home_consumption, autarkie, grid_cost_eur
        )

        self._billing_cache = result
        self._cache_timestamp = datetime.now()

        return result

    async def _get_dynamic_costs_from_db(self, billing_start: date) -> dict[str, float]:
        """Calculate costs and savings from hourly billing and daily energy data. @zara"""
        from collections import defaultdict

        if not self.is_db_available:
            return {"grid_cost_eur": 0, "savings_eur": 0, "weighted_avg_price_ct": 0}

        billing_start_str = billing_start.isoformat()
        total_grid_cost_ct = 0.0
        total_savings_ct = 0.0
        total_import_kwh = 0.0
        weighted_price_sum = 0.0
        hourly_dates_covered: set[str] = set()

        hourly_battery_by_date: dict[str, float] = defaultdict(float)
        hourly_self_by_date: dict[str, float] = defaultdict(float)

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT
                        hb.date,
                        hb.hour,
                        COALESCE(hb.grid_import_kwh, 0) as grid_import_kwh,
                        COALESCE(hb.grid_to_house_kwh, 0) as grid_to_house_kwh,
                        COALESCE(hb.grid_to_battery_kwh, 0) as grid_to_battery_kwh,
                        COALESCE(hb.solar_to_house_kwh, 0) as solar_to_house_kwh,
                        COALESCE(hb.battery_to_house_kwh, 0) as battery_to_house_kwh,
                        COALESCE(gpm.total_price, hb.price_ct_kwh, 0) as price_ct
                    FROM stats_hourly_billing hb
                    LEFT JOIN GPM_price_history gpm
                        ON gpm.hour = hb.hour
                        AND date(gpm.timestamp, '+1 hour') = hb.date
                    WHERE hb.date >= ?
                    ORDER BY hb.date, hb.hour
                """, (billing_start_str,)) as cursor:
                    async for row in cursor:
                        price = row["price_ct"] or 0

                        flow_total = (
                            (row["grid_to_house_kwh"] or 0)
                            + (row["grid_to_battery_kwh"] or 0)
                        )
                        import_kwh = max(row["grid_import_kwh"] or 0, flow_total)

                        self_consumption = (
                            (row["solar_to_house_kwh"] or 0)
                            + (row["battery_to_house_kwh"] or 0)
                        )

                        total_grid_cost_ct += import_kwh * price
                        total_savings_ct += self_consumption * price
                        total_import_kwh += import_kwh
                        weighted_price_sum += import_kwh * price
                        hourly_dates_covered.add(row["date"])

                        hourly_battery_by_date[row["date"]] += (
                            row["grid_to_battery_kwh"] or 0
                        )
                        hourly_self_by_date[row["date"]] += self_consumption

                async with db.execute("""
                    SELECT
                        e.date,
                        COALESCE(e.grid_to_house_kwh, 0) as grid_to_house_kwh,
                        COALESCE(e.grid_to_battery_kwh, 0) as grid_to_battery_kwh,
                        COALESCE(e.solar_to_house_kwh, 0) as solar_to_house_kwh,
                        COALESCE(e.battery_to_house_kwh, 0) as battery_to_house_kwh,
                        COALESCE(p.average_total, 0) as daily_avg_price_ct
                    FROM stats_daily_energy e
                    LEFT JOIN GPM_daily_averages p ON e.date = p.date
                    WHERE e.date >= ?
                    ORDER BY e.date
                """, (billing_start_str,)) as cursor:
                    async for row in cursor:
                        price = row["daily_avg_price_ct"] or 0

                        if row["date"] in hourly_dates_covered:
                            daily_battery = row["grid_to_battery_kwh"] or 0
                            hourly_battery = hourly_battery_by_date.get(
                                row["date"], 0
                            )
                            extra_battery = max(0, daily_battery - hourly_battery)

                            if extra_battery > 0:
                                total_grid_cost_ct += extra_battery * price
                                total_import_kwh += extra_battery
                                weighted_price_sum += extra_battery * price

                            daily_self = (
                                (row["solar_to_house_kwh"] or 0)
                                + (row["battery_to_house_kwh"] or 0)
                            )
                            hourly_self = hourly_self_by_date.get(
                                row["date"], 0
                            )
                            extra_self = max(0, daily_self - hourly_self)

                            if extra_self > 0:
                                total_savings_ct += extra_self * price
                        else:
                            import_kwh = (
                                (row["grid_to_house_kwh"] or 0)
                                + (row["grid_to_battery_kwh"] or 0)
                            )
                            self_consumption = (
                                (row["solar_to_house_kwh"] or 0)
                                + (row["battery_to_house_kwh"] or 0)
                            )

                            total_grid_cost_ct += import_kwh * price
                            total_savings_ct += self_consumption * price
                            total_import_kwh += import_kwh
                            weighted_price_sum += import_kwh * price

        except Exception as err:
            _LOGGER.error("Error calculating dynamic costs from DB: %s", err)
            return {"grid_cost_eur": 0, "savings_eur": 0, "weighted_avg_price_ct": 0}

        weighted_avg = (
            round(weighted_price_sum / total_import_kwh, 2)
            if total_import_kwh > 0 else 0
        )

        _LOGGER.debug(
            "Dynamic billing: cost=%.2f EUR, savings=%.2f EUR, "
            "avg_price=%.2f ct, hourly_days=%d",
            total_grid_cost_ct / 100, total_savings_ct / 100,
            weighted_avg, len(hourly_dates_covered),
        )

        return {
            "grid_cost_eur": round(total_grid_cost_ct / 100, 2),
            "savings_eur": round(total_savings_ct / 100, 2),
            "weighted_avg_price_ct": weighted_avg,
        }

    async def async_import_sensor_csvs(
        self,
        netzbezug_paths: list[str],
        einspeisung_paths: list[str],
        start_date: str = "2025-12-01",
    ) -> dict[str, Any]:
        """Import raw HA sensor CSVs into stats_hourly_billing. @zara"""
        from collections import defaultdict
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Europe/Berlin")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        if not self.is_db_available:
            return {"success": False, "error": "Database not available"}

        all_netzbezug: list[tuple[datetime, float]] = []
        for path in netzbezug_paths:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                content = await f.read()
            all_netzbezug.extend(self._parse_sensor_csv(content, tz))
        all_netzbezug.sort(key=lambda x: x[0])

        netzbezug_hourly: dict[str, float] = defaultdict(float)
        for i in range(1, len(all_netzbezug)):
            dt1, v1 = all_netzbezug[i - 1]
            dt2, v2 = all_netzbezug[i]

            naive1 = dt1.replace(tzinfo=None)
            if naive1 < start_dt:
                continue

            gap_s = (dt2 - dt1).total_seconds()
            if gap_s > 3600 or gap_s <= 0:
                continue

            delta = v2 - v1
            if delta < 0 or delta > 5:
                continue

            hk = dt1.strftime("%Y-%m-%d|%H")
            netzbezug_hourly[hk] += delta

        all_einspeisung: list[tuple[datetime, float]] = []
        for path in einspeisung_paths:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                content = await f.read()
            all_einspeisung.extend(self._parse_sensor_csv(content, tz))

        einspeisung_groups: dict[str, list[float]] = defaultdict(list)
        for dt, watts in all_einspeisung:
            if dt.replace(tzinfo=None) < start_dt:
                continue
            hk = dt.strftime("%Y-%m-%d|%H")
            einspeisung_groups[hk].append(watts)

        einspeisung_hourly: dict[str, float] = {}
        for hk, values in einspeisung_groups.items():
            avg_w = sum(values) / len(values)
            einspeisung_hourly[hk] = max(0, avg_w / 1000)

        all_hours = sorted(
            set(netzbezug_hourly.keys()) | set(einspeisung_hourly.keys())
        )

        imported = 0
        no_price = 0

        try:
            async with self._get_db() as db:
                for hk in all_hours:
                    date_str, hour_str = hk.split("|")
                    hour = int(hour_str)
                    hour_key = f"{date_str}T{hour:02d}:00"

                    grid_import = round(netzbezug_hourly.get(hk, 0), 4)
                    self_consumption = round(einspeisung_hourly.get(hk, 0), 4)

                    price = 0.0
                    async with db.execute(
                        """
                        SELECT total_price FROM GPM_price_history
                        WHERE hour = ? AND date(timestamp, '+1 hour') = ?
                        ORDER BY timestamp DESC LIMIT 1
                        """,
                        (hour, date_str),
                    ) as cursor:
                        row = await cursor.fetchone()
                        if row and row["total_price"] is not None:
                            price = float(row["total_price"])
                        else:
                            no_price += 1

                    cost_ct = round(grid_import * price, 4)

                    await db.execute(
                        """
                        INSERT OR REPLACE INTO stats_hourly_billing (
                            hour_key, date, hour,
                            grid_import_kwh, grid_import_cost_ct,
                            grid_export_kwh, price_ct_kwh,
                            grid_to_house_kwh, grid_to_battery_kwh,
                            solar_yield_kwh, solar_to_house_kwh, solar_to_battery_kwh,
                            battery_to_house_kwh, home_consumption_kwh,
                            data_source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            hour_key, date_str, hour,
                            grid_import, cost_ct,
                            0, round(price, 4),
                            0, 0,
                            0, self_consumption, 0,
                            0, 0,
                            "csv_sensor_import",
                        ),
                    )
                    imported += 1

                await db.commit()

        except Exception as err:
            _LOGGER.error("Sensor CSV import error: %s", err)
            return {"success": False, "error": str(err)}

        _LOGGER.info(
            "Sensor CSV import: %d hours imported, %d without GPM price",
            imported, no_price,
        )
        return {
            "success": True,
            "imported": imported,
            "hours_without_price": no_price,
            "date_range": (
                f"{all_hours[0]} to {all_hours[-1]}" if all_hours else "none"
            ),
        }

    @staticmethod
    def _parse_sensor_csv(
        content: str, tz: Any
    ) -> list[tuple[datetime, float]]:
        """Parse HA sensor CSV to local datetime value pairs. @zara"""
        readings: list[tuple[datetime, float]] = []
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            state = row.get("state", "").strip()
            if state in ("unavailable", "unknown", ""):
                continue
            try:
                value = float(state)
            except (ValueError, TypeError):
                continue

            ts = row.get("last_changed", "").strip()
            try:
                utc_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                local_dt = utc_dt.astimezone(tz)
                readings.append((local_dt, value))
            except (ValueError, TypeError):
                continue

        return readings

    async def async_reset_baselines(self) -> bool:
        """Compatibility stub - no longer needed. @zara"""
        return True
