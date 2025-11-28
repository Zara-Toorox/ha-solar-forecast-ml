"""Battery Charge Persistence - JSON-based History Storage V10.0.0 @zara

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

import asyncio
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles

_LOGGER = logging.getLogger(__name__)

EVENTS_RETENTION_DAYS = 730
DEFAULT_BATTERY_CAPACITY = 10.0

class BatteryChargePersistence:
    """Handles JSON persistence for battery charge tracking Structure: - Daily: Detailed events + summaries - Monthly: Aggregated summaries - Yearly: Aggregated summaries"""

    def __init__(self, file_path: str, battery_capacity: float = DEFAULT_BATTERY_CAPACITY):
        """Initialize persistence handler Args: file_path: Path to JSON file battery_capacity: Battery capacity in kWh @zara"""
        self.file_path = Path(file_path)
        self.battery_capacity = battery_capacity
        self.data: Dict[str, Any] = {}
        self._save_lock = asyncio.Lock()
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure directory exists @zara"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    async def load(self) -> bool:
        """Load data from JSON file Returns: True if loaded successfully, False otherwise @zara"""
        try:
            if not self.file_path.exists():
                _LOGGER.info(f"Creating new battery charge history at {self.file_path}")
                self.data = self._create_empty_structure()
                await self.save()
                return True

            async with aiofiles.open(self.file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                self.data = json.loads(content)

            _LOGGER.info(f"Loaded battery charge history from {self.file_path}")
            return True

        except Exception as e:
            _LOGGER.error(f"Error loading battery charge history: {e}")
            self.data = self._create_empty_structure()
            return False

    async def save(self):
        """Save data to JSON file atomically with race condition protection @zara"""
        async with self._save_lock:
            try:

                self.data["last_update"] = datetime.now().isoformat()

                temp_file = self.file_path.with_suffix(".tmp")

                async with aiofiles.open(temp_file, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(self.data, indent=2, ensure_ascii=False))

                temp_file.replace(self.file_path)

                _LOGGER.debug(f"Saved battery charge history to {self.file_path}")

            except Exception as e:
                _LOGGER.error(f"Error saving battery charge history: {e}")

    def _create_empty_structure(self) -> Dict[str, Any]:
        """Create empty data structure @zara"""
        return {
            "version": "1.0",
            "battery_capacity": self.battery_capacity,
            "last_update": datetime.now().isoformat(),
            "daily": {},
            "monthly": {},
            "yearly": {},
        }

    def _ensure_day_exists(self, date_str: str):
        """Ensure daily entry exists @zara"""
        if date_str not in self.data["daily"]:
            self.data["daily"][date_str] = {
                "date": date_str,

                "grid_charge_kwh": 0.0,
                "solar_charge_kwh": 0.0,
                "grid_discharge_kwh": 0.0,
                "solar_discharge_kwh": 0.0,
                "total_discharge_kwh": 0.0,
                "grid_charge_cost_eur": 0.0,
                "solar_savings_eur": 0.0,
                "grid_arbitrage_profit_eur": 0.0,
                "total_profit_eur": 0.0,

                "solar_to_house_kwh": 0.0,
                "solar_to_battery_kwh": 0.0,
                "solar_to_grid_kwh": 0.0,
                "grid_to_house_kwh": 0.0,
                "grid_to_battery_kwh": 0.0,
                "battery_to_house_kwh": 0.0,
                "grid_import_today_kwh": 0.0,
                "grid_export_today_kwh": 0.0,

                "charge_events": [],
                "discharge_events": [],
                "summary": {
                    "total_charge_events": 0,
                    "total_discharge_events": 0,
                    "avg_grid_charge_price": 0.0,
                    "avg_discharge_price": 0.0,
                    "grid_charge_ratio": 0.0,
                    "solar_charge_ratio": 0.0,
                },
            }

    async def add_charge_event(
        self,
        timestamp: datetime,
        source: str,
        power_w: float,
        duration_min: float,
        kwh: float,
        price_cent_kwh: float,
    ):
        """Add a charging event Args: timestamp: Event timestamp (local time) source: 'grid' or 'solar' power_w: Power in Watts duration_min: Duration in minutes kwh: Energy in kWh price_cent_kwh: Price in Cent/kWh (0 for solar)"""
        date_str = timestamp.date().isoformat()
        self._ensure_day_exists(date_str)

        day_data = self.data["daily"][date_str]

        event = {
            "timestamp": timestamp.isoformat(),
            "hour": timestamp.hour,
            "source": source,
            "power_w": round(power_w, 1),
            "duration_min": round(duration_min, 2),
            "kwh": round(kwh, 4),
            "price_cent_kwh": round(price_cent_kwh, 2),
        }
        day_data["charge_events"].append(event)

        if source == "grid":
            day_data["grid_charge_kwh"] += kwh
            cost_eur = kwh * (price_cent_kwh / 100)
            day_data["grid_charge_cost_eur"] += cost_eur
        else:
            day_data["solar_charge_kwh"] += kwh

        self._update_daily_summary(date_str)

        if len(day_data["charge_events"]) % 10 == 0:
            await self.save()

    async def add_discharge_event(
        self,
        timestamp: datetime,
        power_w: float,
        duration_min: float,
        kwh: float,
        price_cent_kwh: float,
        solar_ratio: float,
    ):
        """Add a discharging event Args: timestamp: Event timestamp (local time) power_w: Power in Watts (positive value) duration_min: Duration in minutes kwh: Total energy discharged in kWh price_cent_kwh: Current electricity price in Cent/kWh solar_ratio: Ratio of solar energy in battery (0.0-1.0)"""
        date_str = timestamp.date().isoformat()
        self._ensure_day_exists(date_str)

        day_data = self.data["daily"][date_str]

        solar_kwh = kwh * solar_ratio
        grid_kwh = kwh * (1 - solar_ratio)

        event = {
            "timestamp": timestamp.isoformat(),
            "hour": timestamp.hour,
            "power_w": round(power_w, 1),
            "duration_min": round(duration_min, 2),
            "kwh": round(kwh, 4),
            "price_cent_kwh": round(price_cent_kwh, 2),
            "source_breakdown": {
                "solar_kwh": round(solar_kwh, 4),
                "grid_kwh": round(grid_kwh, 4),
                "solar_ratio": round(solar_ratio, 3),
            },
        }
        day_data["discharge_events"].append(event)

        day_data["solar_discharge_kwh"] += solar_kwh
        day_data["grid_discharge_kwh"] += grid_kwh
        day_data["total_discharge_kwh"] += kwh

        solar_savings = solar_kwh * (price_cent_kwh / 100)
        day_data["solar_savings_eur"] += solar_savings

        if grid_kwh > 0 and day_data["grid_charge_kwh"] > 0:

            avg_charge_price = (
                day_data["grid_charge_cost_eur"] / day_data["grid_charge_kwh"]
            ) * 100
            arbitrage_profit = grid_kwh * ((price_cent_kwh - avg_charge_price) / 100)
            day_data["grid_arbitrage_profit_eur"] += arbitrage_profit

        self._update_daily_summary(date_str)

        if len(day_data["discharge_events"]) % 10 == 0:
            await self.save()

    def _update_daily_summary(self, date_str: str):
        """Update daily summary calculations @zara"""
        day_data = self.data["daily"][date_str]
        summary = day_data["summary"]

        summary["total_charge_events"] = len(day_data["charge_events"])
        summary["total_discharge_events"] = len(day_data["discharge_events"])

        grid_charges = [e for e in day_data["charge_events"] if e["source"] == "grid"]
        if grid_charges:
            total_kwh = sum(e["kwh"] for e in grid_charges)
            if total_kwh > 0:
                weighted_price = (
                    sum(e["kwh"] * e["price_cent_kwh"] for e in grid_charges) / total_kwh
                )
                summary["avg_grid_charge_price"] = round(weighted_price, 2)

        if day_data["discharge_events"]:
            total_kwh = sum(e["kwh"] for e in day_data["discharge_events"])
            if total_kwh > 0:
                weighted_price = (
                    sum(e["kwh"] * e["price_cent_kwh"] for e in day_data["discharge_events"])
                    / total_kwh
                )
                summary["avg_discharge_price"] = round(weighted_price, 2)

        total_charge = day_data["grid_charge_kwh"] + day_data["solar_charge_kwh"]
        if total_charge > 0:
            summary["grid_charge_ratio"] = round(day_data["grid_charge_kwh"] / total_charge, 3)
            summary["solar_charge_ratio"] = round(day_data["solar_charge_kwh"] / total_charge, 3)

        day_data["total_profit_eur"] = (
            day_data["solar_savings_eur"]
            + day_data["grid_arbitrage_profit_eur"]
            - day_data["grid_charge_cost_eur"]
        )

    async def update_energy_flows_v10(self, timestamp: datetime, energy_flows: Dict[str, float]):
        """Update v10.0.0 energy flow data @zara"""
        date_str = timestamp.date().isoformat()
        self._ensure_day_exists(date_str)

        day_data = self.data["daily"][date_str]

        day_data["solar_to_house_kwh"] += energy_flows.get("solar_to_house_kwh", 0.0)
        day_data["solar_to_battery_kwh"] += energy_flows.get("solar_to_battery_kwh", 0.0)
        day_data["solar_to_grid_kwh"] += energy_flows.get("solar_to_grid_kwh", 0.0)
        day_data["grid_to_house_kwh"] += energy_flows.get("grid_to_house_kwh", 0.0)
        day_data["grid_to_battery_kwh"] += energy_flows.get("grid_to_battery_kwh", 0.0)
        day_data["battery_to_house_kwh"] += energy_flows.get("battery_to_house_kwh", 0.0)
        day_data["grid_import_today_kwh"] += energy_flows.get("grid_import_kwh", 0.0)
        day_data["grid_export_today_kwh"] += energy_flows.get("grid_export_kwh", 0.0)

        _LOGGER.debug(
            f"Updated v10.0.0 energy flows for {date_str}: "
            f"Solar→House={day_data['solar_to_house_kwh']:.3f}, "
            f"Solar→Battery={day_data['solar_to_battery_kwh']:.3f}, "
            f"Grid→House={day_data['grid_to_house_kwh']:.3f}"
        )

    async def update_energy_flows_v9(self, timestamp: datetime, energy_flows: Dict[str, float]):
        await self.update_energy_flows_v10(timestamp, energy_flows)

    def get_today_summary(self) -> Dict[str, Any]:
        """Get summary for today @zara"""
        today = datetime.now().date().isoformat()
        return self.data["daily"].get(today, {})

    def get_day_summary(self, date: datetime) -> Dict[str, Any]:
        """Get summary for specific day @zara"""
        date_str = date.date().isoformat()
        return self.data["daily"].get(date_str, {})

    def get_month_summary(self, year: int, month: int) -> Dict[str, Any]:
        """Get summary for specific month @zara"""
        key = f"{year}-{month:02d}"
        return self.data["monthly"].get(key, {})

    def get_year_summary(self, year: int) -> Dict[str, Any]:
        """Get summary for specific year @zara"""
        key = str(year)
        return self.data["yearly"].get(key, {})

    async def rollup_to_monthly(self, date: date):
        """Aggregate daily data to monthly summary Args: date: Date to rollup (typically yesterday) - date object, not datetime @zara"""
        try:
            year_month = f"{date.year}-{date.month:02d}"
            date_str = date.isoformat()

            if date_str not in self.data["daily"]:
                return

            if year_month not in self.data["monthly"]:
                self.data["monthly"][year_month] = {
                    "year_month": year_month,

                    "grid_charge_kwh": 0.0,
                    "solar_charge_kwh": 0.0,
                    "grid_charge_cost_eur": 0.0,
                    "solar_savings_eur": 0.0,
                    "grid_arbitrage_profit_eur": 0.0,
                    "total_profit_eur": 0.0,

                    "solar_to_house_kwh": 0.0,
                    "solar_to_battery_kwh": 0.0,
                    "solar_to_grid_kwh": 0.0,
                    "grid_to_house_kwh": 0.0,
                    "grid_to_battery_kwh": 0.0,
                    "battery_to_house_kwh": 0.0,
                    "grid_import_kwh": 0.0,
                    "grid_export_kwh": 0.0,
                    "days_tracked": 0,
                }

            day_data = self.data["daily"][date_str]
            month_data = self.data["monthly"][year_month]

            month_data["grid_charge_kwh"] += day_data["grid_charge_kwh"]
            month_data["solar_charge_kwh"] += day_data["solar_charge_kwh"]
            month_data["grid_charge_cost_eur"] += day_data["grid_charge_cost_eur"]
            month_data["solar_savings_eur"] += day_data["solar_savings_eur"]
            month_data["grid_arbitrage_profit_eur"] += day_data["grid_arbitrage_profit_eur"]
            month_data["total_profit_eur"] += day_data["total_profit_eur"]

            month_data["solar_to_house_kwh"] += day_data.get("solar_to_house_kwh", 0.0)
            month_data["solar_to_battery_kwh"] += day_data.get("solar_to_battery_kwh", 0.0)
            month_data["solar_to_grid_kwh"] += day_data.get("solar_to_grid_kwh", 0.0)
            month_data["grid_to_house_kwh"] += day_data.get("grid_to_house_kwh", 0.0)
            month_data["grid_to_battery_kwh"] += day_data.get("grid_to_battery_kwh", 0.0)
            month_data["battery_to_house_kwh"] += day_data.get("battery_to_house_kwh", 0.0)
            month_data["grid_import_kwh"] += day_data.get("grid_import_today_kwh", 0.0)
            month_data["grid_export_kwh"] += day_data.get("grid_export_today_kwh", 0.0)
            month_data["days_tracked"] += 1

            _LOGGER.info(f"Rolled up {date_str} to monthly summary {year_month}")
            await self.save()

        except Exception as e:
            _LOGGER.error(f"Error rolling up to monthly: {e}")

    async def rollup_to_yearly(self, year: int, month: int):
        """Aggregate monthly data to yearly summary Args: year: Year month: Month @zara"""
        try:
            year_str = str(year)
            year_month = f"{year}-{month:02d}"

            if year_month not in self.data["monthly"]:
                return

            if year_str not in self.data["yearly"]:
                self.data["yearly"][year_str] = {
                    "year": year,

                    "grid_charge_kwh": 0.0,
                    "solar_charge_kwh": 0.0,
                    "grid_charge_cost_eur": 0.0,
                    "solar_savings_eur": 0.0,
                    "grid_arbitrage_profit_eur": 0.0,
                    "total_profit_eur": 0.0,

                    "solar_to_house_kwh": 0.0,
                    "solar_to_battery_kwh": 0.0,
                    "solar_to_grid_kwh": 0.0,
                    "grid_to_house_kwh": 0.0,
                    "grid_to_battery_kwh": 0.0,
                    "battery_to_house_kwh": 0.0,
                    "grid_import_kwh": 0.0,
                    "grid_export_kwh": 0.0,
                    "months_tracked": 0,
                }

            month_data = self.data["monthly"][year_month]
            year_data = self.data["yearly"][year_str]

            year_data["grid_charge_kwh"] += month_data["grid_charge_kwh"]
            year_data["solar_charge_kwh"] += month_data["solar_charge_kwh"]
            year_data["grid_charge_cost_eur"] += month_data["grid_charge_cost_eur"]
            year_data["solar_savings_eur"] += month_data["solar_savings_eur"]
            year_data["grid_arbitrage_profit_eur"] += month_data["grid_arbitrage_profit_eur"]
            year_data["total_profit_eur"] += month_data["total_profit_eur"]

            year_data["solar_to_house_kwh"] += month_data.get("solar_to_house_kwh", 0.0)
            year_data["solar_to_battery_kwh"] += month_data.get("solar_to_battery_kwh", 0.0)
            year_data["solar_to_grid_kwh"] += month_data.get("solar_to_grid_kwh", 0.0)
            year_data["grid_to_house_kwh"] += month_data.get("grid_to_house_kwh", 0.0)
            year_data["grid_to_battery_kwh"] += month_data.get("grid_to_battery_kwh", 0.0)
            year_data["battery_to_house_kwh"] += month_data.get("battery_to_house_kwh", 0.0)
            year_data["grid_import_kwh"] += month_data.get("grid_import_kwh", 0.0)
            year_data["grid_export_kwh"] += month_data.get("grid_export_kwh", 0.0)
            year_data["months_tracked"] += 1

            _LOGGER.info(f"Rolled up {year_month} to yearly summary {year}")
            await self.save()

        except Exception as e:
            _LOGGER.error(f"Error rolling up to yearly: {e}")

    async def cleanup_old_events(self, keep_days: int = EVENTS_RETENTION_DAYS):
        """Remove detailed events older than keep_days, but keep summaries Args: keep_days: Number of days to keep detailed events @zara"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=keep_days)).date()
            removed_count = 0

            for date_str in list(self.data["daily"].keys()):
                try:
                    date = datetime.fromisoformat(date_str).date()
                    if date < cutoff_date:

                        day_data = self.data["daily"][date_str]
                        day_data["charge_events"] = []
                        day_data["discharge_events"] = []
                        removed_count += 1
                except ValueError:
                    continue

            if removed_count > 0:
                _LOGGER.info(f"Cleaned up events from {removed_count} old days (keeping summaries)")
                await self.save()

        except Exception as e:
            _LOGGER.error(f"Error cleaning up old events: {e}")
