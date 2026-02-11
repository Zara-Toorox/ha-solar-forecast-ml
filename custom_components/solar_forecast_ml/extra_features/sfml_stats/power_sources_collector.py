# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Power Sources Data Collector for SFML Stats. @zara"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

import aiosqlite

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

from .const import (
    CONF_SENSOR_SOLAR_TO_HOUSE,
    CONF_SENSOR_SOLAR_TO_BATTERY,
    CONF_SENSOR_BATTERY_TO_HOUSE,
    CONF_SENSOR_GRID_TO_HOUSE,
    CONF_SENSOR_GRID_TO_BATTERY,
    CONF_SENSOR_HOME_CONSUMPTION,
    CONF_SENSOR_BATTERY_SOC,
    CONF_SENSOR_SMARTMETER_IMPORT,
    CONF_SENSOR_SMARTMETER_EXPORT,
    SENSOR_W_TO_DAILY_KWH_MAP,
    CONF_SENSOR_HEATPUMP_DAILY,
    CONF_SENSOR_HEATINGROD_DAILY,
    CONF_SENSOR_WALLBOX_DAILY,
)

_W_KEY_TO_DB_COLUMN: dict[str, str] = {
    CONF_SENSOR_SOLAR_TO_BATTERY: "solar_to_battery_kwh",
    CONF_SENSOR_BATTERY_TO_HOUSE: "battery_to_house_kwh",
    CONF_SENSOR_GRID_TO_BATTERY: "grid_to_battery_kwh",
    CONF_SENSOR_HOME_CONSUMPTION: "home_consumption_kwh",
    CONF_SENSOR_SMARTMETER_IMPORT: "smartmeter_import_kwh",
    CONF_SENSOR_SMARTMETER_EXPORT: "smartmeter_export_kwh",
}
from .sfml_data_reader import SFMLDataReader

_LOGGER = logging.getLogger(__name__)

COLLECTION_INTERVAL = 300

MAX_DATA_AGE_DAYS = 7


class PowerSourcesCollector:
    """Collects power sources data periodically and stores in SFML DB. @zara"""

    def __init__(self, hass: HomeAssistant, config: dict[str, Any], data_path: Path) -> None:
        """Initialize the collector. @zara"""
        self.hass = hass
        self.config = config
        self.data_path = data_path
        self._db_path = Path(hass.config.path()) / "solar_forecast_ml" / "solar_forecast.db"
        self._task: asyncio.Task | None = None
        self._running = False
        self._sfml_reader = SFMLDataReader(hass)

    @asynccontextmanager
    async def _get_db(self) -> AsyncIterator[aiosqlite.Connection]:
        """Get DB connection via manager with direct fallback. @zara"""
        from .storage.db_connection_manager import get_manager
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

    async def start(self) -> None:
        """Start the data collection task. @zara"""
        if self._running:
            return

        if not self.is_db_available:
            _LOGGER.warning("SFML database not found at %s - Power Sources Collector disabled", self._db_path)
            return

        self._running = True
        self._task = asyncio.create_task(self._collection_loop())
        _LOGGER.info("Power Sources Collector started (using SFML database)")

    async def stop(self) -> None:
        """Stop the data collection task. @zara"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        _LOGGER.info("Power Sources Collector stopped")

    async def _collection_loop(self) -> None:
        """Main collection loop. @zara"""
        while self._running:
            try:
                await self._collect_data()
            except Exception as e:
                _LOGGER.error("Error collecting power sources data: %s", e)

            await asyncio.sleep(COLLECTION_INTERVAL)

    async def _collect_data(self) -> None:
        """Collect current power values and store in SFML database. @zara"""
        now = datetime.now(timezone.utc)
        local_now = now.astimezone()

        solar_power = self._sfml_reader.get_live_power()

        solar_to_house = self._get_sensor_value(CONF_SENSOR_SOLAR_TO_HOUSE)
        solar_to_battery = self._get_sensor_value(CONF_SENSOR_SOLAR_TO_BATTERY)
        battery_to_house = self._get_sensor_value(CONF_SENSOR_BATTERY_TO_HOUSE)
        grid_to_house = self._get_sensor_value(CONF_SENSOR_GRID_TO_HOUSE)
        grid_to_battery = self._get_sensor_value(CONF_SENSOR_GRID_TO_BATTERY)
        home_consumption = self._get_sensor_value(CONF_SENSOR_HOME_CONSUMPTION)
        battery_soc = self._get_sensor_value(CONF_SENSOR_BATTERY_SOC)
        smartmeter_import = self._get_sensor_value(CONF_SENSOR_SMARTMETER_IMPORT)
        smartmeter_export = self._get_sensor_value(CONF_SENSOR_SMARTMETER_EXPORT)

        if solar_power is not None and solar_power < 0:
            solar_power = 0.0
        if solar_to_house is not None and solar_to_house < 0:
            solar_to_house = 0.0
        if solar_to_battery is not None and solar_to_battery < 0:
            solar_to_battery = 0.0

        if solar_to_house is None and solar_power is not None:
            solar_to_battery_val = solar_to_battery or 0
            solar_to_house = max(0, solar_power - solar_to_battery_val)
            _LOGGER.debug("Calculated solar_to_house: %.1f W (solar %.1f - battery %.1f)",
                         solar_to_house, solar_power, solar_to_battery_val)

        if grid_to_house is None and home_consumption is not None:
            solar_to_house_val = solar_to_house or 0
            battery_to_house_val = battery_to_house or 0
            grid_to_house = max(0, home_consumption - solar_to_house_val - battery_to_house_val)
            _LOGGER.debug("Calculated grid_to_house: %.1f W (consumption %.1f - solar %.1f - battery %.1f)",
                         grid_to_house, home_consumption, solar_to_house_val, battery_to_house_val)

        if smartmeter_import is None and grid_to_house is not None:
            grid_to_battery_val = grid_to_battery or 0
            smartmeter_import = grid_to_house + grid_to_battery_val
            _LOGGER.debug("Calculated smartmeter_import: %.1f W", smartmeter_import)

        if smartmeter_export is None and solar_power is not None:
            solar_to_house_val = solar_to_house or 0
            solar_to_battery_val = solar_to_battery or 0
            smartmeter_export = max(0, solar_power - solar_to_house_val - solar_to_battery_val)
            _LOGGER.debug("Calculated smartmeter_export: %.1f W", smartmeter_export)

        try:
            async with self._get_db() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO stats_power_sources (
                        timestamp, date, hour,
                        solar_power_w, house_consumption_w,
                        solar_to_house_w, solar_to_battery_w,
                        battery_to_house_w, grid_to_house_w
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    now.isoformat(),
                    local_now.strftime("%Y-%m-%d"),
                    local_now.hour,
                    solar_power or 0,
                    home_consumption or 0,
                    solar_to_house or 0,
                    solar_to_battery or 0,
                    battery_to_house or 0,
                    grid_to_house or 0,
                ))
                await db.commit()

                cutoff = (now - timedelta(days=MAX_DATA_AGE_DAYS)).isoformat()
                await db.execute(
                    "DELETE FROM stats_power_sources WHERE timestamp < ?",
                    (cutoff,)
                )
                await db.commit()
        except Exception as e:
            _LOGGER.error("Error inserting power sources data into DB: %s", e)

        _LOGGER.debug(
            "Collected power data: solar_power=%.1f, solar_to_house=%.1f, solar_to_battery=%.1f, battery=%.1f, grid=%.1f, consumption=%.1f, soc=%s",
            solar_power or 0, solar_to_house or 0, solar_to_battery or 0, battery_to_house or 0, grid_to_house or 0,
            home_consumption or 0, battery_soc
        )

        data_point = {
            "solar_power": solar_power,
            "solar_to_house": solar_to_house,
            "solar_to_battery": solar_to_battery,
            "battery_to_house": battery_to_house,
            "grid_to_house": grid_to_house,
            "grid_to_battery": grid_to_battery,
            "home_consumption": home_consumption,
            "battery_soc": battery_soc,
            "smartmeter_import": smartmeter_import,
            "smartmeter_export": smartmeter_export,
        }
        await self._update_daily_stats(now, data_point)

    def _get_sensor_value(self, config_key: str) -> float | None:
        """Get sensor value from Home Assistant. @zara"""
        entity_id = self.config.get(config_key)
        if not entity_id:
            return None

        state = self.hass.states.get(entity_id)
        if not state or state.state in ('unknown', 'unavailable'):
            return None

        try:
            return float(state.state)
        except (ValueError, TypeError):
            return None

    async def get_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get historical data for the specified number of hours from SFML database. @zara"""
        if not self.is_db_available:
            return []

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT timestamp, date, hour,
                           solar_power_w, house_consumption_w,
                           solar_to_house_w, solar_to_battery_w,
                           battery_to_house_w, grid_to_house_w
                    FROM stats_power_sources
                    WHERE timestamp > ?
                    ORDER BY timestamp ASC
                """, (cutoff,)) as cursor:
                    rows = await cursor.fetchall()

            result = []
            for row in rows:
                result.append({
                    "timestamp": row["timestamp"],
                    "solar_power": row["solar_power_w"],
                    "solar_to_house": row["solar_to_house_w"],
                    "solar_to_battery": row["solar_to_battery_w"],
                    "battery_to_house": row["battery_to_house_w"],
                    "grid_to_house": row["grid_to_house_w"],
                    "home_consumption": row["house_consumption_w"],
                    "battery_soc": None,
                })
            return result
        except Exception as e:
            _LOGGER.error("Error reading power sources history from DB: %s", e)
            return []

    async def _ensure_db_columns(self, db: aiosqlite.Connection) -> None:
        """Ensure all required columns exist in stats_daily_energy table. @zara"""
        async with db.execute("PRAGMA table_info(stats_daily_energy)") as cursor:
            rows = await cursor.fetchall()
            existing_columns = {row[1] for row in rows}

        new_columns = [
            ("grid_to_battery_kwh", "REAL DEFAULT 0"),
            ("smartmeter_import_kwh", "REAL DEFAULT 0"),
            ("smartmeter_export_kwh", "REAL DEFAULT 0"),
            ("consumer_heatpump_kwh", "REAL DEFAULT 0"),
            ("consumer_heatingrod_kwh", "REAL DEFAULT 0"),
            ("consumer_wallbox_kwh", "REAL DEFAULT 0"),
        ]

        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                try:
                    await db.execute(f"ALTER TABLE stats_daily_energy ADD COLUMN {col_name} {col_type}")
                    _LOGGER.info("Added column %s to stats_daily_energy", col_name)
                except Exception as e:
                    _LOGGER.debug("Column %s might already exist: %s", col_name, e)

        await db.commit()

    async def _update_daily_stats(self, now: datetime, data_point: dict[str, Any]) -> None:
        """Update daily statistics in stats_daily_energy table. @zara"""
        if not self.is_db_available:
            return

        local_now = now.astimezone()
        today_str = local_now.strftime("%Y-%m-%d")

        interval_hours = COLLECTION_INTERVAL / 3600

        solar_to_house_kwh = ((data_point.get("solar_to_house") or 0) * interval_hours) / 1000
        solar_to_battery_kwh = ((data_point.get("solar_to_battery") or 0) * interval_hours) / 1000
        battery_to_house_kwh = ((data_point.get("battery_to_house") or 0) * interval_hours) / 1000
        grid_to_house_kwh = ((data_point.get("grid_to_house") or 0) * interval_hours) / 1000
        grid_to_battery_kwh = ((data_point.get("grid_to_battery") or 0) * interval_hours) / 1000
        home_consumption_kwh = ((data_point.get("home_consumption") or 0) * interval_hours) / 1000
        smartmeter_import_kwh = ((data_point.get("smartmeter_import") or 0) * interval_hours) / 1000
        smartmeter_export_kwh = ((data_point.get("smartmeter_export") or 0) * interval_hours) / 1000
        solar_yield_kwh = solar_to_house_kwh + solar_to_battery_kwh

        solar_power = data_point.get("solar_power") or 0

        try:
            async with self._get_db() as db:
                await self._ensure_db_columns(db)

                async with db.execute(
                    "SELECT * FROM stats_daily_energy WHERE date = ?",
                    (today_str,)
                ) as cursor:
                    existing = await cursor.fetchone()

                if existing:
                    await db.execute("""
                        UPDATE stats_daily_energy SET
                            solar_yield_kwh = solar_yield_kwh + ?,
                            solar_to_house_kwh = solar_to_house_kwh + ?,
                            solar_to_battery_kwh = solar_to_battery_kwh + ?,
                            battery_to_house_kwh = battery_to_house_kwh + ?,
                            grid_to_house_kwh = grid_to_house_kwh + ?,
                            grid_to_battery_kwh = COALESCE(grid_to_battery_kwh, 0) + ?,
                            home_consumption_kwh = home_consumption_kwh + ?,
                            smartmeter_import_kwh = COALESCE(smartmeter_import_kwh, 0) + ?,
                            smartmeter_export_kwh = COALESCE(smartmeter_export_kwh, 0) + ?,
                            peak_solar_w = MAX(peak_solar_w, ?),
                            peak_solar_time = CASE WHEN ? > peak_solar_w THEN ? ELSE peak_solar_time END
                        WHERE date = ?
                    """, (
                        solar_yield_kwh,
                        solar_to_house_kwh,
                        solar_to_battery_kwh,
                        battery_to_house_kwh,
                        grid_to_house_kwh,
                        grid_to_battery_kwh,
                        home_consumption_kwh,
                        smartmeter_import_kwh,
                        smartmeter_export_kwh,
                        solar_power,
                        solar_power,
                        local_now.strftime("%H:%M"),
                        today_str,
                    ))
                else:
                    await db.execute("""
                        INSERT INTO stats_daily_energy (
                            date, solar_yield_kwh, solar_to_house_kwh, solar_to_battery_kwh,
                            battery_to_house_kwh, grid_to_house_kwh, grid_to_battery_kwh,
                            home_consumption_kwh, smartmeter_import_kwh, smartmeter_export_kwh,
                            peak_solar_w, peak_solar_time
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        today_str,
                        solar_yield_kwh,
                        solar_to_house_kwh,
                        solar_to_battery_kwh,
                        battery_to_house_kwh,
                        grid_to_house_kwh,
                        grid_to_battery_kwh,
                        home_consumption_kwh,
                        smartmeter_import_kwh,
                        smartmeter_export_kwh,
                        solar_power,
                        local_now.strftime("%H:%M"),
                    ))

                for w_key, db_col in _W_KEY_TO_DB_COLUMN.items():
                    daily_key = SENSOR_W_TO_DAILY_KWH_MAP.get(w_key)
                    if not daily_key:
                        continue
                    daily_entity = self.config.get(daily_key)
                    if not daily_entity:
                        continue
                    value = self._get_sensor_value(daily_key)
                    if value is not None:
                        await db.execute(
                            f"UPDATE stats_daily_energy SET {db_col} = ? WHERE date = ?",
                            (round(max(0, value), 4), today_str)
                        )

                _consumer_daily_map = {
                    CONF_SENSOR_HEATPUMP_DAILY: "consumer_heatpump_kwh",
                    CONF_SENSOR_HEATINGROD_DAILY: "consumer_heatingrod_kwh",
                    CONF_SENSOR_WALLBOX_DAILY: "consumer_wallbox_kwh",
                }
                for sensor_key, db_col in _consumer_daily_map.items():
                    entity_id = self.config.get(sensor_key)
                    if not entity_id:
                        continue
                    value = self._get_sensor_value(sensor_key)
                    if value is not None:
                        await db.execute(
                            f"UPDATE stats_daily_energy SET {db_col} = ? WHERE date = ?",
                            (round(max(0, value), 4), today_str)
                        )

                await db.execute("""
                    UPDATE stats_daily_energy SET
                        autarkie_percent = CASE
                            WHEN home_consumption_kwh > 0
                            THEN MIN(100, MAX(0, (solar_to_house_kwh + battery_to_house_kwh) / home_consumption_kwh * 100))
                            ELSE 0
                        END,
                        self_consumption_kwh = solar_to_house_kwh + solar_to_battery_kwh
                    WHERE date = ?
                """, (today_str,))

                await db.commit()
        except Exception as e:
            _LOGGER.error("Error updating daily stats in DB: %s", e)

    async def get_daily_stats(self, days: int = 7) -> dict[str, Any]:
        """Get daily statistics for the specified number of days from SFML database. @zara"""
        if not self.is_db_available:
            return {"version": 1, "last_updated": None, "days": {}}

        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            async with self._get_db() as db:
                async with db.execute("""
                    SELECT date, solar_yield_kwh, solar_to_house_kwh, solar_to_battery_kwh,
                           battery_to_house_kwh, battery_charge_solar_kwh, battery_charge_grid_kwh,
                           grid_to_house_kwh, grid_to_battery_kwh,
                           grid_export_kwh, smartmeter_import_kwh, smartmeter_export_kwh,
                           home_consumption_kwh,
                           autarkie_percent, self_consumption_kwh, peak_solar_w, peak_solar_time
                    FROM stats_daily_energy
                    WHERE date >= ?
                    ORDER BY date DESC
                """, (cutoff,)) as cursor:
                    rows = await cursor.fetchall()

            filtered_days = {}
            for row in rows:
                solar_to_house = row["solar_to_house_kwh"] or 0
                solar_to_battery = row["solar_to_battery_kwh"] or 0
                solar_yield = row["solar_yield_kwh"] or 0
                self_cons_pct = (
                    ((solar_to_house + solar_to_battery) / solar_yield * 100)
                    if solar_yield > 0 else 0
                )
                grid_to_house = row["grid_to_house_kwh"] or 0
                # Use accumulator value, fall back to sensor value
                grid_to_battery = row["grid_to_battery_kwh"] or row["battery_charge_grid_kwh"] or 0
                grid_export = row["grid_export_kwh"] or row["smartmeter_export_kwh"] or 0
                battery_to_house = row["battery_to_house_kwh"] or 0
                # Charge from solar: accumulator or sensor
                charge_solar = solar_to_battery or row["battery_charge_solar_kwh"] or 0
                total_charged = charge_solar + grid_to_battery
                filtered_days[row["date"]] = {
                    "date": row["date"],
                    "solar_total_kwh": solar_yield,
                    "solar_yield_kwh": solar_yield,
                    "solar_to_house_kwh": solar_to_house,
                    "solar_to_battery_kwh": solar_to_battery,
                    "battery_to_house_kwh": battery_to_house,
                    "battery_charge_solar_kwh": charge_solar,
                    "battery_charge_grid_kwh": grid_to_battery,
                    "battery_total_charged_kwh": total_charged,
                    "grid_to_house_kwh": grid_to_house,
                    "grid_to_battery_kwh": grid_to_battery,
                    "grid_import_kwh": grid_to_house + grid_to_battery,
                    "grid_export_kwh": grid_export,
                    "home_consumption_kwh": row["home_consumption_kwh"] or 0,
                    "autarky_percent": row["autarkie_percent"] or 0,
                    "self_consumption_percent": min(100, self_cons_pct),
                    "peak_solar_w": row["peak_solar_w"] or 0,
                    "peak_solar_time": row["peak_solar_time"],
                }

            return {
                "version": 1,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "days": filtered_days
            }
        except Exception as e:
            _LOGGER.error("Error reading daily stats from DB: %s", e)
            return {"version": 1, "last_updated": None, "days": {}}
