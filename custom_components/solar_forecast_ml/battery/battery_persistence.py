"""
Battery Charge Persistence - JSON-based History Storage

Stores battery charge/discharge events and summaries in JSON
Completely independent from Solar/ML components

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import logging
import aiofiles
import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

# Retention settings
EVENTS_RETENTION_DAYS = 730  # Keep 2 years of detailed events
DEFAULT_BATTERY_CAPACITY = 10.0


class BatteryChargePersistence:
    """
    Handles JSON persistence for battery charge tracking

    Structure:
    - Daily: Detailed events + summaries
    - Monthly: Aggregated summaries
    - Yearly: Aggregated summaries
    """

    def __init__(self, file_path: str, battery_capacity: float = DEFAULT_BATTERY_CAPACITY):
        """
        Initialize persistence handler

        Args:
            file_path: Path to JSON file
            battery_capacity: Battery capacity in kWh
        """
        self.file_path = Path(file_path)
        self.battery_capacity = battery_capacity
        self.data: Dict[str, Any] = {}
        self._save_lock = asyncio.Lock()  # Prevent race conditions during save
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure directory exists"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    async def load(self) -> bool:
        """
        Load data from JSON file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not self.file_path.exists():
                _LOGGER.info(f"Creating new battery charge history at {self.file_path}")
                self.data = self._create_empty_structure()
                await self.save()
                return True

            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                self.data = json.loads(content)

            _LOGGER.info(f"Loaded battery charge history from {self.file_path}")
            return True

        except Exception as e:
            _LOGGER.error(f"Error loading battery charge history: {e}")
            self.data = self._create_empty_structure()
            return False

    async def save(self):
        """Save data to JSON file atomically with race condition protection"""
        async with self._save_lock:  # Prevent concurrent saves
            try:
                # Update last_update timestamp
                self.data['last_update'] = datetime.now().isoformat()

                # Write to temporary file first
                temp_file = self.file_path.with_suffix('.tmp')

                async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(self.data, indent=2, ensure_ascii=False))

                # Atomic rename
                temp_file.replace(self.file_path)

                _LOGGER.debug(f"Saved battery charge history to {self.file_path}")

            except Exception as e:
                _LOGGER.error(f"Error saving battery charge history: {e}")

    def _create_empty_structure(self) -> Dict[str, Any]:
        """Create empty data structure"""
        return {
            'version': '1.0',
            'battery_capacity': self.battery_capacity,
            'last_update': datetime.now().isoformat(),
            'daily': {},
            'monthly': {},
            'yearly': {},
        }

    def _ensure_day_exists(self, date_str: str):
        """Ensure daily entry exists"""
        if date_str not in self.data['daily']:
            self.data['daily'][date_str] = {
                'date': date_str,
                'grid_charge_kwh': 0.0,
                'solar_charge_kwh': 0.0,
                'grid_discharge_kwh': 0.0,
                'solar_discharge_kwh': 0.0,
                'total_discharge_kwh': 0.0,
                'grid_charge_cost_eur': 0.0,
                'solar_savings_eur': 0.0,
                'grid_arbitrage_profit_eur': 0.0,
                'total_profit_eur': 0.0,
                'charge_events': [],
                'discharge_events': [],
                'summary': {
                    'total_charge_events': 0,
                    'total_discharge_events': 0,
                    'avg_grid_charge_price': 0.0,
                    'avg_discharge_price': 0.0,
                    'grid_charge_ratio': 0.0,
                    'solar_charge_ratio': 0.0,
                }
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
        """
        Add a charging event

        Args:
            timestamp: Event timestamp (local time)
            source: 'grid' or 'solar'
            power_w: Power in Watts
            duration_min: Duration in minutes
            kwh: Energy in kWh
            price_cent_kwh: Price in Cent/kWh (0 for solar)
        """
        date_str = timestamp.date().isoformat()
        self._ensure_day_exists(date_str)

        day_data = self.data['daily'][date_str]

        # Add event
        event = {
            'timestamp': timestamp.isoformat(),
            'hour': timestamp.hour,
            'source': source,
            'power_w': round(power_w, 1),
            'duration_min': round(duration_min, 2),
            'kwh': round(kwh, 4),
            'price_cent_kwh': round(price_cent_kwh, 2),
        }
        day_data['charge_events'].append(event)

        # Update totals
        if source == 'grid':
            day_data['grid_charge_kwh'] += kwh
            cost_eur = kwh * (price_cent_kwh / 100)
            day_data['grid_charge_cost_eur'] += cost_eur
        else:  # solar
            day_data['solar_charge_kwh'] += kwh

        # Update summary
        self._update_daily_summary(date_str)

        # Auto-save every 10 events or every hour
        if len(day_data['charge_events']) % 10 == 0:
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
        """
        Add a discharging event

        Args:
            timestamp: Event timestamp (local time)
            power_w: Power in Watts (positive value)
            duration_min: Duration in minutes
            kwh: Total energy discharged in kWh
            price_cent_kwh: Current electricity price in Cent/kWh
            solar_ratio: Ratio of solar energy in battery (0.0-1.0)
        """
        date_str = timestamp.date().isoformat()
        self._ensure_day_exists(date_str)

        day_data = self.data['daily'][date_str]

        # Calculate breakdown
        solar_kwh = kwh * solar_ratio
        grid_kwh = kwh * (1 - solar_ratio)

        # Add event
        event = {
            'timestamp': timestamp.isoformat(),
            'hour': timestamp.hour,
            'power_w': round(power_w, 1),
            'duration_min': round(duration_min, 2),
            'kwh': round(kwh, 4),
            'price_cent_kwh': round(price_cent_kwh, 2),
            'source_breakdown': {
                'solar_kwh': round(solar_kwh, 4),
                'grid_kwh': round(grid_kwh, 4),
                'solar_ratio': round(solar_ratio, 3),
            }
        }
        day_data['discharge_events'].append(event)

        # Update totals
        day_data['solar_discharge_kwh'] += solar_kwh
        day_data['grid_discharge_kwh'] += grid_kwh
        day_data['total_discharge_kwh'] += kwh

        # Calculate savings
        solar_savings = solar_kwh * (price_cent_kwh / 100)
        day_data['solar_savings_eur'] += solar_savings

        # Calculate grid arbitrage profit
        if grid_kwh > 0 and day_data['grid_charge_kwh'] > 0:
            # Use average grid charge price for arbitrage calculation
            avg_charge_price = (day_data['grid_charge_cost_eur'] / day_data['grid_charge_kwh']) * 100
            arbitrage_profit = grid_kwh * ((price_cent_kwh - avg_charge_price) / 100)
            day_data['grid_arbitrage_profit_eur'] += arbitrage_profit

        # Update summary
        self._update_daily_summary(date_str)

        # Auto-save every 10 events
        if len(day_data['discharge_events']) % 10 == 0:
            await self.save()

    def _update_daily_summary(self, date_str: str):
        """Update daily summary calculations"""
        day_data = self.data['daily'][date_str]
        summary = day_data['summary']

        # Event counts
        summary['total_charge_events'] = len(day_data['charge_events'])
        summary['total_discharge_events'] = len(day_data['discharge_events'])

        # Average prices
        grid_charges = [e for e in day_data['charge_events'] if e['source'] == 'grid']
        if grid_charges:
            total_kwh = sum(e['kwh'] for e in grid_charges)
            if total_kwh > 0:
                weighted_price = sum(e['kwh'] * e['price_cent_kwh'] for e in grid_charges) / total_kwh
                summary['avg_grid_charge_price'] = round(weighted_price, 2)

        if day_data['discharge_events']:
            total_kwh = sum(e['kwh'] for e in day_data['discharge_events'])
            if total_kwh > 0:
                weighted_price = sum(e['kwh'] * e['price_cent_kwh'] for e in day_data['discharge_events']) / total_kwh
                summary['avg_discharge_price'] = round(weighted_price, 2)

        # Charge ratios
        total_charge = day_data['grid_charge_kwh'] + day_data['solar_charge_kwh']
        if total_charge > 0:
            summary['grid_charge_ratio'] = round(day_data['grid_charge_kwh'] / total_charge, 3)
            summary['solar_charge_ratio'] = round(day_data['solar_charge_kwh'] / total_charge, 3)

        # Total profit
        day_data['total_profit_eur'] = (
            day_data['solar_savings_eur'] +
            day_data['grid_arbitrage_profit_eur'] -
            day_data['grid_charge_cost_eur']
        )

    def get_today_summary(self) -> Dict[str, Any]:
        """Get summary for today"""
        today = datetime.now().date().isoformat()
        return self.data['daily'].get(today, {})

    def get_day_summary(self, date: datetime) -> Dict[str, Any]:
        """Get summary for specific day"""
        date_str = date.date().isoformat()
        return self.data['daily'].get(date_str, {})

    def get_month_summary(self, year: int, month: int) -> Dict[str, Any]:
        """Get summary for specific month"""
        key = f"{year}-{month:02d}"
        return self.data['monthly'].get(key, {})

    def get_year_summary(self, year: int) -> Dict[str, Any]:
        """Get summary for specific year"""
        key = str(year)
        return self.data['yearly'].get(key, {})

    async def rollup_to_monthly(self, date: datetime):
        """
        Aggregate daily data to monthly summary

        Args:
            date: Date to rollup (typically yesterday)
        """
        try:
            year_month = f"{date.year}-{date.month:02d}"
            date_str = date.date().isoformat()

            if date_str not in self.data['daily']:
                return

            # Ensure monthly entry exists
            if year_month not in self.data['monthly']:
                self.data['monthly'][year_month] = {
                    'year_month': year_month,
                    'grid_charge_kwh': 0.0,
                    'solar_charge_kwh': 0.0,
                    'grid_charge_cost_eur': 0.0,
                    'solar_savings_eur': 0.0,
                    'grid_arbitrage_profit_eur': 0.0,
                    'total_profit_eur': 0.0,
                    'days_tracked': 0,
                }

            # Add daily values to monthly
            day_data = self.data['daily'][date_str]
            month_data = self.data['monthly'][year_month]

            month_data['grid_charge_kwh'] += day_data['grid_charge_kwh']
            month_data['solar_charge_kwh'] += day_data['solar_charge_kwh']
            month_data['grid_charge_cost_eur'] += day_data['grid_charge_cost_eur']
            month_data['solar_savings_eur'] += day_data['solar_savings_eur']
            month_data['grid_arbitrage_profit_eur'] += day_data['grid_arbitrage_profit_eur']
            month_data['total_profit_eur'] += day_data['total_profit_eur']
            month_data['days_tracked'] += 1

            _LOGGER.info(f"Rolled up {date_str} to monthly summary {year_month}")
            await self.save()

        except Exception as e:
            _LOGGER.error(f"Error rolling up to monthly: {e}")

    async def rollup_to_yearly(self, year: int, month: int):
        """
        Aggregate monthly data to yearly summary

        Args:
            year: Year
            month: Month
        """
        try:
            year_str = str(year)
            year_month = f"{year}-{month:02d}"

            if year_month not in self.data['monthly']:
                return

            # Ensure yearly entry exists
            if year_str not in self.data['yearly']:
                self.data['yearly'][year_str] = {
                    'year': year,
                    'grid_charge_kwh': 0.0,
                    'solar_charge_kwh': 0.0,
                    'grid_charge_cost_eur': 0.0,
                    'solar_savings_eur': 0.0,
                    'grid_arbitrage_profit_eur': 0.0,
                    'total_profit_eur': 0.0,
                    'months_tracked': 0,
                }

            # Add monthly values to yearly
            month_data = self.data['monthly'][year_month]
            year_data = self.data['yearly'][year_str]

            year_data['grid_charge_kwh'] += month_data['grid_charge_kwh']
            year_data['solar_charge_kwh'] += month_data['solar_charge_kwh']
            year_data['grid_charge_cost_eur'] += month_data['grid_charge_cost_eur']
            year_data['solar_savings_eur'] += month_data['solar_savings_eur']
            year_data['grid_arbitrage_profit_eur'] += month_data['grid_arbitrage_profit_eur']
            year_data['total_profit_eur'] += month_data['total_profit_eur']
            year_data['months_tracked'] += 1

            _LOGGER.info(f"Rolled up {year_month} to yearly summary {year}")
            await self.save()

        except Exception as e:
            _LOGGER.error(f"Error rolling up to yearly: {e}")

    async def cleanup_old_events(self, keep_days: int = EVENTS_RETENTION_DAYS):
        """
        Remove detailed events older than keep_days, but keep summaries

        Args:
            keep_days: Number of days to keep detailed events
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=keep_days)).date()
            removed_count = 0

            for date_str in list(self.data['daily'].keys()):
                try:
                    date = datetime.fromisoformat(date_str).date()
                    if date < cutoff_date:
                        # Remove detailed events, keep summary
                        day_data = self.data['daily'][date_str]
                        day_data['charge_events'] = []
                        day_data['discharge_events'] = []
                        removed_count += 1
                except ValueError:
                    continue

            if removed_count > 0:
                _LOGGER.info(f"Cleaned up events from {removed_count} old days (keeping summaries)")
                await self.save()

        except Exception as e:
            _LOGGER.error(f"Error cleaning up old events: {e}")
