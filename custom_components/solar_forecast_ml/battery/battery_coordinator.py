"""
Battery Coordinator - Separate from Solar Forecast

Coordinates battery charge tracking, persistence, and price data
Completely independent from Solar/ML coordinator

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from datetime import timedelta, datetime
from typing import Optional

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from ..const import (
    DOMAIN,
    BASE_DATA_DIR,
    DATA_DIR,
    CONF_BATTERY_SOC_ENTITY,
    CONF_BATTERY_POWER_ENTITY,
    CONF_BATTERY_GRID_CHARGE_POWER_ENTITY,
    CONF_BATTERY_CAPACITY,
    DEFAULT_BATTERY_CAPACITY,
    CONF_ELECTRICITY_COUNTRY,
    DEFAULT_ELECTRICITY_COUNTRY,
)
from .battery_persistence import BatteryChargePersistence
from .electricity_price_service import ElectricityPriceService

_LOGGER = logging.getLogger(__name__)

# Update interval for battery tracking (2 minutes for accurate tracking)
UPDATE_INTERVAL = timedelta(minutes=2)


class BatteryCoordinator(DataUpdateCoordinator):
    """Battery Coordinator - Handles battery charge tracking Completely separate from Solar Forecast ML Coordinator Updates every 2 minutes to track charging/discharging"""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """Initialize battery coordinator Args: hass: Home Assistant instance entry: Config entry"""
        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}_battery",
            update_interval=UPDATE_INTERVAL,
        )

        self.entry = entry
        self.hass = hass

        # Get configuration
        self.battery_capacity = entry.options.get(
            CONF_BATTERY_CAPACITY,
            entry.data.get(CONF_BATTERY_CAPACITY, DEFAULT_BATTERY_CAPACITY)
        )
        self.soc_entity = entry.options.get(
            CONF_BATTERY_SOC_ENTITY,
            entry.data.get(CONF_BATTERY_SOC_ENTITY)
        )
        self.power_entity = entry.options.get(
            CONF_BATTERY_POWER_ENTITY,
            entry.data.get(CONF_BATTERY_POWER_ENTITY)
        )
        self.grid_charge_power_entity = entry.options.get(
            CONF_BATTERY_GRID_CHARGE_POWER_ENTITY,
            entry.data.get(CONF_BATTERY_GRID_CHARGE_POWER_ENTITY)
        )
        self.electricity_country = entry.options.get(
            CONF_ELECTRICITY_COUNTRY,
            entry.data.get(CONF_ELECTRICITY_COUNTRY, DEFAULT_ELECTRICITY_COUNTRY)
        )

        # Initialize components
        self.persistence: Optional[BatteryChargePersistence] = None
        self.electricity_service: Optional[ElectricityPriceService] = None

        _LOGGER.info(
            f"BatteryCoordinator initialized - "
            f"Capacity: {self.battery_capacity} kWh, "
            f"Update interval: {UPDATE_INTERVAL}"
        )

    async def async_setup(self):
        """Setup coordinator components"""
        try:
            # Initialize persistence
            persistence_file = f"{BASE_DATA_DIR}/{DATA_DIR}/battery_charge_history.json"
            self.persistence = BatteryChargePersistence(
                file_path=persistence_file,
                battery_capacity=self.battery_capacity,
            )
            await self.persistence.load()

            # Initialize electricity price service
            self.electricity_service = ElectricityPriceService(
                country=self.electricity_country
            )

            # Fetch initial prices
            await self.electricity_service.fetch_day_ahead_prices()

            # Log comprehensive battery setup summary
            _LOGGER.info(
                f"Battery Coordinator Setup Complete ✓\n"
                f"  → Battery Capacity: {self.battery_capacity} kWh\n"
                f"  → Update Interval: {UPDATE_INTERVAL}\n"
                f"  → Electricity Country: {self.electricity_country}\n"
                f"  → Price Tracking: Active (aWATTar)\n"
                f"  → Charge/Discharge Tracking: Active\n"
                f"  → Cost Optimization: Ready"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Error setting up BatteryCoordinator: {e}")
            return False

    async def _async_update_data(self):
        """Fetch battery data and track charging/discharging Returns: Dictionary with current battery data"""
        try:
            # Get current battery power
            battery_power = await self._get_battery_power()
            if battery_power is None:
                _LOGGER.debug("Battery power sensor not available, skipping update")
                return {}

            # Get grid charge power (Anker specific)
            grid_charge_power = await self._get_grid_charge_power()
            if grid_charge_power is None:
                grid_charge_power = 0.0

            # Calculate solar charge power
            # When battery is charging: solar_power = total_battery_power - grid_charge_power
            # This correctly attributes the charging source when both grid and solar contribute
            if battery_power > 50:  # Battery is charging (threshold to avoid noise)
                solar_power = max(0, battery_power - grid_charge_power)
            else:
                solar_power = 0.0

            # Get current electricity price
            current_price = self.electricity_service.get_current_price() if self.electricity_service else None
            if current_price is None:
                current_price = 0.0  # Fallback

            # Track charging/discharging events
            if battery_power > 50:  # Charging (with tolerance)
                if grid_charge_power > 50:  # Grid charging
                    await self.persistence.add_charge_event(
                        timestamp=datetime.now(),
                        source='grid',
                        power_w=grid_charge_power,
                        duration_min=UPDATE_INTERVAL.total_seconds() / 60,
                        kwh=(grid_charge_power / 1000) * (UPDATE_INTERVAL.total_seconds() / 3600),
                        price_cent_kwh=current_price,
                    )

                # Solar charging (if any)
                if solar_power > 50:
                    await self.persistence.add_charge_event(
                        timestamp=datetime.now(),
                        source='solar',
                        power_w=solar_power,
                        duration_min=UPDATE_INTERVAL.total_seconds() / 60,
                        kwh=(solar_power / 1000) * (UPDATE_INTERVAL.total_seconds() / 3600),
                        price_cent_kwh=0.0,
                    )

            elif battery_power < -50:  # Discharging (with tolerance)
                discharge_power = abs(battery_power)
                discharge_kwh = (discharge_power / 1000) * (UPDATE_INTERVAL.total_seconds() / 3600)

                # Calculate solar ratio
                today_summary = self.persistence.get_today_summary()
                total_charged = (
                    today_summary.get('grid_charge_kwh', 0) +
                    today_summary.get('solar_charge_kwh', 0)
                )
                solar_ratio = (
                    today_summary.get('solar_charge_kwh', 0) / total_charged
                    if total_charged > 0 else 0.0
                )

                await self.persistence.add_discharge_event(
                    timestamp=datetime.now(),
                    power_w=discharge_power,
                    duration_min=UPDATE_INTERVAL.total_seconds() / 60,
                    kwh=discharge_kwh,
                    price_cent_kwh=current_price,
                    solar_ratio=solar_ratio,
                )

            # Return current state
            summary = self.persistence.get_today_summary() if self.persistence else {}
            return {
                'battery_power': battery_power,
                'grid_charge_power': grid_charge_power,
                'solar_power': solar_power,
                'current_price': current_price,
                'summary': summary,
                'last_update': datetime.now().isoformat(),
            }

        except Exception as e:
            _LOGGER.error(f"Error updating battery data: {e}")
            raise UpdateFailed(f"Error updating battery data: {e}")

    async def _get_battery_power(self) -> Optional[float]:
        """Get current battery power from sensor Returns: Power in Watts (+ charging, - discharging) or None"""
        if not self.power_entity:
            return None

        state = self.hass.states.get(self.power_entity)
        if state is None or state.state in ('unknown', 'unavailable'):
            return None

        try:
            return float(state.state)
        except (ValueError, TypeError):
            _LOGGER.warning(f"Invalid battery power state: {state.state}")
            return None

    async def _get_grid_charge_power(self) -> Optional[float]:
        """Get current grid charge power from configured entity Returns: Power in Watts or None"""
        if not self.grid_charge_power_entity:
            _LOGGER.debug("No grid charge power entity configured")
            return 0.0

        state = self.hass.states.get(self.grid_charge_power_entity)
        if state is None or state.state in ('unknown', 'unavailable'):
            return 0.0

        try:
            return float(state.state)
        except (ValueError, TypeError):
            _LOGGER.warning(f"Invalid grid charge power state: {state.state}")
            return 0.0

    async def async_refresh_prices(self):
        """Refresh electricity prices"""
        if self.electricity_service:
            try:
                await self.electricity_service.fetch_day_ahead_prices()
                _LOGGER.info("Electricity prices refreshed")
            except Exception as e:
                _LOGGER.error(f"Error refreshing electricity prices: {e}")

    async def async_daily_rollup(self):
        """Perform daily rollup tasks Called at midnight to aggregate daily data to monthly"""
        try:
            if not self.persistence:
                return

            # Rollup yesterday to monthly
            yesterday = datetime.now().date() - timedelta(days=1)
            await self.persistence.rollup_to_monthly(yesterday)

            # Cleanup old events
            await self.persistence.cleanup_old_events()

            # Save
            await self.persistence.save()

            _LOGGER.info("Daily rollup completed")

        except Exception as e:
            _LOGGER.error(f"Error during daily rollup: {e}")

    async def async_monthly_rollup(self):
        """Perform monthly rollup tasks Called at end of month to aggregate monthly data to yearly"""
        try:
            if not self.persistence:
                return

            now = datetime.now()
            last_month = now.month - 1 if now.month > 1 else 12
            year = now.year if now.month > 1 else now.year - 1

            await self.persistence.rollup_to_yearly(year, last_month)
            await self.persistence.save()

            _LOGGER.info(f"Monthly rollup completed for {year}-{last_month:02d}")

        except Exception as e:
            _LOGGER.error(f"Error during monthly rollup: {e}")
