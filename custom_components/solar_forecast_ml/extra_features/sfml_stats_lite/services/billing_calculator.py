"""Billing calculator for energy balance using Recorder data.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import aiofiles

from homeassistant.core import HomeAssistant

from ..const import (
    DOMAIN,
    CONF_SENSOR_PRICE_TOTAL,
    CONF_BILLING_PRICE_MODE,
    CONF_BILLING_FIXED_PRICE,
    CONF_BILLING_START_DAY,
    CONF_BILLING_START_MONTH,
    PRICE_MODE_FIXED,
    DEFAULT_BILLING_FIXED_PRICE,
    CONF_SENSOR_SOLAR_POWER,
    CONF_SENSOR_SOLAR_TO_HOUSE,
    CONF_SENSOR_SOLAR_TO_BATTERY,
    CONF_SENSOR_BATTERY_TO_HOUSE,
    CONF_SENSOR_GRID_TO_BATTERY,
    CONF_SENSOR_HOME_CONSUMPTION,
    CONF_SENSOR_SMARTMETER_IMPORT,
    CONF_SENSOR_SMARTMETER_EXPORT,
)

_LOGGER = logging.getLogger(__name__)

_LOG_FILE: Path | None = None
_LOG_BUFFER: list[str] = []


def _log(msg: str, *args, level: str = "info") -> None:
    """Log to HA logger and buffer for file."""
    formatted = msg % args if args else msg
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"{timestamp} | {level.upper():<8} | {formatted}"

    if level == "error":
        _LOGGER.error(msg, *args)
    else:
        _LOGGER.debug(msg, *args)

    _LOG_BUFFER.append(log_line)


class BillingCalculator:
    """Calculate energy balance using Riemann sum from watt sensors."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_path: Path,
        entry_data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the calculator."""
        self._hass = hass
        self._config_path = config_path
        self._entry_data = entry_data
        self._log_file = config_path / "sfml_stats_lite" / "logs" / "billing_calculator.log"
        self._billing_cache: dict[str, Any] | None = None
        self._cache_timestamp: datetime | None = None
        self._cache_ttl_seconds = 60

    def update_config(self, new_config: dict[str, Any]) -> None:
        """Update cached configuration and invalidate billing cache."""
        self._entry_data = new_config
        self._billing_cache = None
        self._cache_timestamp = None
        _log("BillingCalculator config updated, cache invalidated")

    def _get_config(self) -> dict[str, Any]:
        """Get current configuration."""
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
        """Read current value from a sensor."""
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
        """Calculate start date of current billing period."""
        import calendar

        billing_start_day = config.get(CONF_BILLING_START_DAY, 1)
        billing_start_month = config.get(CONF_BILLING_START_MONTH, 1)

        if isinstance(billing_start_day, str):
            billing_start_day = int(billing_start_day)
        if isinstance(billing_start_month, str):
            billing_start_month = int(billing_start_month)

        today = date.today()
        current_year = today.year

        # Korrigiere ungültige Tage (z.B. 31. Februar -> 28. Februar)
        max_day = calendar.monthrange(current_year, billing_start_month)[1]
        billing_start_day = min(billing_start_day, max_day)

        billing_start = date(current_year, billing_start_month, billing_start_day)

        if billing_start > today:
            # Prüfe auch das vorherige Jahr auf gültige Tage
            max_day_prev = calendar.monthrange(current_year - 1, billing_start_month)[1]
            billing_start_day = min(billing_start_day, max_day_prev)
            billing_start = date(current_year - 1, billing_start_month, billing_start_day)

        return billing_start

    async def _calculate_riemann_kwh(
        self,
        entity_id: str,
        start_date: date,
    ) -> tuple[float, int]:
        """Calculate kWh from watt sensor using left Riemann sum."""
        if not entity_id:
            return 0.0, 0

        try:
            from homeassistant.components.recorder import get_instance
            from homeassistant.components.recorder.history import state_changes_during_period
            from homeassistant.util import dt as dt_util

            instance = get_instance(self._hass)

            start_time = dt_util.start_of_local_day(
                datetime.combine(start_date, datetime.min.time())
            )
            end_time = dt_util.now()

            states = await instance.async_add_executor_job(
                state_changes_during_period,
                self._hass,
                start_time,
                end_time,
                entity_id,
                True,
                False,
                None,
            )

            if isinstance(states, dict):
                entity_states = states.get(entity_id, [])
            else:
                entity_states = states if states else []

            if not entity_states:
                return 0.0, 0

            total_kwh = 0.0
            sample_count = 0
            prev_time = None
            prev_value = None

            max_gap_hours = 4.0

            for state in entity_states:
                try:
                    current_value = float(state.state)
                    current_time = state.last_changed

                    if prev_time is not None and prev_value is not None:
                        delta_hours = (current_time - prev_time).total_seconds() / 3600.0

                        if delta_hours <= max_gap_hours:
                            kwh = (prev_value / 1000.0) * delta_hours
                            total_kwh += max(0, kwh)
                            sample_count += 1
                        else:
                            avg_value = (prev_value + current_value) / 2.0
                            kwh = (avg_value / 1000.0) * delta_hours
                            total_kwh += max(0, kwh)
                            sample_count += 1
                            _log("Large gap (%.1fh) for %s, using average value",
                                 delta_hours, entity_id)

                    prev_time = current_time
                    prev_value = current_value

                except (ValueError, TypeError):
                    continue

            if prev_time is not None and prev_value is not None:
                delta_hours = (end_time - prev_time).total_seconds() / 3600.0
                if delta_hours > 0 and delta_hours <= max_gap_hours:
                    kwh = (prev_value / 1000.0) * delta_hours
                    total_kwh += max(0, kwh)
                    sample_count += 1
                elif delta_hours > max_gap_hours:
                    kwh = (prev_value / 1000.0) * delta_hours
                    total_kwh += max(0, kwh)
                    sample_count += 1
                    _log("Final interval large (%.1fh) for %s - sensor may be stale",
                         delta_hours, entity_id)

            return total_kwh, sample_count

        except Exception as err:
            _log("Error in Riemann calculation for %s: %s", entity_id, err, level="error")
            return 0.0, 0

    async def _flush_logs(self) -> None:
        """Write buffered logs to file."""
        global _LOG_BUFFER
        if _LOG_BUFFER:
            try:
                self._log_file.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(self._log_file, "a", encoding="utf-8") as f:
                    for line in _LOG_BUFFER:
                        await f.write(line + "\n")
                _LOG_BUFFER.clear()
            except Exception:
                pass

    async def async_ensure_baselines(self) -> dict[str, Any]:
        """Compatibility stub - no longer needed."""
        return {}

    async def async_calculate_billing(self) -> dict[str, Any]:
        """Calculate current energy balance with caching."""
        if self._billing_cache is not None and self._cache_timestamp is not None:
            cache_age = (datetime.now() - self._cache_timestamp).total_seconds()
            if cache_age < self._cache_ttl_seconds:
                _log("Returning cached billing result (age: %.1fs)", cache_age)
                return self._billing_cache

        _log("=" * 60)
        _log("async_calculate_billing() - cache miss, recalculating")

        config = self._get_config()
        billing_start = self._get_billing_start_date(config)
        today = date.today()

        _log("Billing start: %s", billing_start)
        _log("Today: %s", today)

        sensors = {
            "smartmeter_import": config.get(CONF_SENSOR_SMARTMETER_IMPORT),
            "smartmeter_export": config.get(CONF_SENSOR_SMARTMETER_EXPORT),
            "solar_power": config.get(CONF_SENSOR_SOLAR_POWER),
            "solar_to_house": config.get(CONF_SENSOR_SOLAR_TO_HOUSE),
            "solar_to_battery": config.get(CONF_SENSOR_SOLAR_TO_BATTERY),
            "battery_to_house": config.get(CONF_SENSOR_BATTERY_TO_HOUSE),
            "grid_to_battery": config.get(CONF_SENSOR_GRID_TO_BATTERY),
            "home_consumption": config.get(CONF_SENSOR_HOME_CONSUMPTION),
        }

        _log("Configured sensors:")
        for name, entity in sensors.items():
            _log("  %s: %s", name, entity or "NOT CONFIGURED")

        flows = {}
        total_samples = 0

        for flow_name, entity_id in sensors.items():
            if entity_id:
                kwh, samples = await self._calculate_riemann_kwh(entity_id, billing_start)
                flows[flow_name] = kwh
                total_samples += samples
                _log("  %s: %.4f kWh (%d samples)", flow_name, kwh, samples)
            else:
                flows[flow_name] = 0.0

        smartmeter_import = flows.get("smartmeter_import", 0.0)
        smartmeter_export = flows.get("smartmeter_export", 0.0)
        solar_power = flows.get("solar_power", 0.0)
        solar_to_battery = flows.get("solar_to_battery", 0.0)
        grid_to_battery = flows.get("grid_to_battery", 0.0)

        _log("=" * 60)
        _log("CALCULATIONS:")

        # Prüfe ob Batterie-Sensoren konfiguriert sind
        has_battery = bool(
            config.get(CONF_SENSOR_BATTERY_TO_HOUSE) or
            config.get(CONF_SENSOR_SOLAR_TO_BATTERY) or
            config.get(CONF_SENSOR_GRID_TO_BATTERY)
        )
        _log("  Battery configured: %s", has_battery)

        # Solar direkt zum Haus: Sensor bevorzugen, sonst berechnen
        if config.get(CONF_SENSOR_SOLAR_TO_HOUSE):
            solar_direct = flows.get("solar_to_house", 0.0)
            _log("  Solar->House = %.2f kWh (from sensor)", solar_direct)
        else:
            solar_direct = max(0, solar_power - solar_to_battery)
            _log("  Solar->House = Solar(%.2f) - Solar->Battery(%.2f) = %.2f kWh (calculated)",
                 solar_power, solar_to_battery, solar_direct)

        # Batterie zum Haus: Sensor bevorzugen, sonst 0 wenn keine Batterie
        if config.get(CONF_SENSOR_BATTERY_TO_HOUSE):
            battery_to_house = flows.get("battery_to_house", 0.0)
            _log("  Battery->House = %.2f kWh (from sensor)", battery_to_house)
        elif has_battery:
            battery_to_house = 0.0
            _log("  Battery->House = 0 kWh (no direct sensor, cannot calculate safely)")
        else:
            battery_to_house = 0.0
            _log("  Battery->House = 0 kWh (no battery configured)")

        # Eigenverbrauch (WR -> Haus) = Solar direkt + Batterie-Entladung
        wr_to_house = solar_direct + battery_to_house
        _log("  WR->House = Solar->House(%.2f) + Battery->House(%.2f) = %.2f kWh",
             solar_direct, battery_to_house, wr_to_house)

        # Netz zum Haus
        grid_to_house = max(0, smartmeter_import - grid_to_battery)
        _log("  Grid->House = Smartmeter(%.2f) - Grid->Battery(%.2f) = %.2f kWh",
             smartmeter_import, grid_to_battery, grid_to_house)

        # Hausverbrauch: Sensor bevorzugen, sonst berechnen
        if config.get(CONF_SENSOR_HOME_CONSUMPTION):
            home_consumption_sensor = flows.get("home_consumption", 0.0)
            home_consumption_calculated = wr_to_house + grid_to_house
            if home_consumption_sensor >= wr_to_house * 0.9:  # 10% Toleranz
                home_consumption = home_consumption_sensor
                _log("  Home consumption = %.2f kWh (from sensor)", home_consumption)
            else:
                home_consumption = home_consumption_calculated
                _log("  Home consumption = %.2f kWh (calculated, sensor value %.2f implausible)",
                     home_consumption, home_consumption_sensor)
        else:
            home_consumption = wr_to_house + grid_to_house
            _log("  Home consumption = WR->House(%.2f) + Grid->House(%.2f) = %.2f kWh (calculated)",
                 wr_to_house, grid_to_house, home_consumption)

        # Batterie-Ladung
        battery_total_charge = solar_to_battery + grid_to_battery
        _log("  Battery charge = Solar->Battery(%.2f) + Grid->Battery(%.2f) = %.2f kWh",
             solar_to_battery, grid_to_battery, battery_total_charge)

        price_mode = config.get(CONF_BILLING_PRICE_MODE, "dynamic")
        if price_mode == PRICE_MODE_FIXED:
            avg_price = config.get(CONF_BILLING_FIXED_PRICE, DEFAULT_BILLING_FIXED_PRICE)
        else:
            current_price = self._get_sensor_value(config.get(CONF_SENSOR_PRICE_TOTAL))
            avg_price = current_price if current_price else DEFAULT_BILLING_FIXED_PRICE

        grid_cost_eur = (smartmeter_import * avg_price) / 100
        _log("  Grid cost = %.2f kWh * %.2f ct = %.2f EUR",
             smartmeter_import, avg_price, grid_cost_eur)

        savings_eur = (wr_to_house * avg_price) / 100
        _log("  Savings = %.2f kWh * %.2f ct = %.2f EUR",
             wr_to_house, avg_price, savings_eur)

        if home_consumption > 0:
            autarkie = (wr_to_house / home_consumption) * 100
        else:
            autarkie = 0.0

        _log("  Autarky = WR->House(%.2f) / Home consumption(%.2f) = %.1f%%",
             wr_to_house, home_consumption, autarkie)

        days_elapsed = (today - billing_start).days + 1

        # Berechne theoretisches Abrechnungsende (1 Jahr nach Start - 1 Tag)
        # Behandle Schaltjahr-Fehler (z.B. Start am 29. Februar)
        try:
            billing_end_theoretical = date(
                billing_start.year + 1, billing_start.month, billing_start.day
            ) - timedelta(days=1)
        except ValueError:
            # Fallback für ungültige Daten (z.B. 29. Feb -> 28. Feb)
            billing_end_theoretical = date(
                billing_start.year + 1, billing_start.month, 28
            ) - timedelta(days=1)

        days_total = (billing_end_theoretical - billing_start).days + 1

        result = {
            "success": True,
            "data_available": total_samples > 0,
            "period": {
                "start": billing_start.isoformat(),
                "end": today.isoformat(),
                "end_theoretical": billing_end_theoretical.isoformat(),
                "days_elapsed": days_elapsed,
                "days_total": days_total,
                "progress_percent": round((days_elapsed / days_total) * 100, 1),
            },
            "household": {
                "total_kwh": round(home_consumption, 2),
                "from_solar_kwh": round(solar_direct, 2),
                "from_battery_kwh": round(battery_to_house, 2),
                "from_grid_kwh": round(grid_to_house, 2),
            },
            "battery": {
                "total_charge_kwh": round(battery_total_charge, 2),
                "from_solar_kwh": round(solar_to_battery, 2),
                "from_grid_kwh": round(grid_to_battery, 2),
            },
            "solar": {
                "total_kwh": round(solar_power, 2),
                "to_house_kwh": round(solar_direct, 2),
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
            "recorder": {
                "sample_count": total_samples,
                "data_available": total_samples > 0,
            },
            "config": {
                "billing_start_day": config.get(CONF_BILLING_START_DAY),
                "billing_start_month": config.get(CONF_BILLING_START_MONTH),
                "price_mode": price_mode,
            },
        }

        _log("=" * 60)
        _log("RESULT:")
        _log("  Smartmeter (import): %.2f kWh", smartmeter_import)
        _log("  Home consumption total: %.2f kWh", result["household"]["total_kwh"])
        _log("    from solar direct: %.2f kWh", result["household"]["from_solar_kwh"])
        _log("    from battery: %.2f kWh", result["household"]["from_battery_kwh"])
        _log("    from grid: %.2f kWh", result["household"]["from_grid_kwh"])
        _log("  WR->House (self-consumption): %.2f kWh", wr_to_house)
        _log("  Battery charged: %.2f kWh", result["battery"]["total_charge_kwh"])
        _log("  Autarky: %.1f%%", result["autarkie_percent"])
        _log("  Grid cost: %.2f EUR", result["finance"]["grid_cost_eur"])
        _log("  Savings: %.2f EUR", result["finance"]["savings_eur"])
        _log("  Samples: %d", total_samples)
        _log("=" * 60)

        await self._flush_logs()

        self._billing_cache = result
        self._cache_timestamp = datetime.now()

        return result

    async def async_reset_baselines(self) -> bool:
        """Compatibility stub - no longer needed."""
        return True
