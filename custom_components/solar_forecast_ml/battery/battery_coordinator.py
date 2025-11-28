"""Battery Coordinator - Separate from Solar Forecast V10.0.0 @zara

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

import logging
from datetime import datetime, timedelta
from typing import Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from ..const import (
    BASE_DATA_DIR,
    CONF_BATTERY_CAPACITY,
    CONF_BATTERY_GRID_CHARGE_POWER_ENTITY,
    CONF_BATTERY_POWER_ENTITY,
    CONF_BATTERY_POWER_SENSOR,
    CONF_BATTERY_SOC_ENTITY,
    CONF_BATTERY_SOC_SENSOR,
    CONF_BATTERY_TEMPERATURE_SENSOR,
    CONF_ELECTRICITY_COUNTRY,
    CONF_GRID_CHARGE_POWER_SENSOR,
    CONF_GRID_EXPORT_SENSOR,
    CONF_GRID_IMPORT_SENSOR,
    CONF_HOUSE_CONSUMPTION_SENSOR,
    CONF_INVERTER_OUTPUT_SENSOR,
    CONF_SOLAR_PRODUCTION_SENSOR,
    DATA_DIR,
    DEFAULT_BATTERY_CAPACITY,
    DEFAULT_ELECTRICITY_COUNTRY,
    DOMAIN,
)
from .battery_charge_tracker import BatteryChargeTracker
from .battery_data_collector import BatteryDataCollector
from .battery_persistence import BatteryChargePersistence
from .electricity_price_service import ElectricityPriceService

_LOGGER = logging.getLogger(__name__)

UPDATE_INTERVAL = timedelta(minutes=2)

class BatteryCoordinator(DataUpdateCoordinator):
    """Battery Coordinator - Handles battery charge tracking Completely separate from Solar Forecast ML Coordinator Updates every 2 minutes to track charging/discharging"""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """Initialize battery coordinator Args: hass: Home Assistant instance entry: Config entry @zara"""
        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}_battery",
            update_interval=UPDATE_INTERVAL,
        )

        self.entry = entry
        self.hass = hass

        self.battery_capacity = entry.options.get(
            CONF_BATTERY_CAPACITY, entry.data.get(CONF_BATTERY_CAPACITY, DEFAULT_BATTERY_CAPACITY)
        )
        self.electricity_country = entry.options.get(
            CONF_ELECTRICITY_COUNTRY,
            entry.data.get(CONF_ELECTRICITY_COUNTRY, DEFAULT_ELECTRICITY_COUNTRY),
        )

        self.data_collector = BatteryDataCollector(hass, entry)

        using_v10 = self.data_collector.using_new_config

        self.charge_tracker = BatteryChargeTracker(
            hass, self.battery_capacity, using_new_config=using_v10
        )
        self.persistence: Optional[BatteryChargePersistence] = None
        self.electricity_service: Optional[ElectricityPriceService] = None

        _LOGGER.info(
            f"BatteryCoordinator initialized ({'v10 watt-based' if using_v10 else 'LEGACY v8.x'}) - "
            f"Capacity: {self.battery_capacity} kWh, "
            f"Update interval: {UPDATE_INTERVAL}"
        )

    async def async_setup(self):
        """Setup coordinator components @zara"""
        try:

            persistence_file = f"{BASE_DATA_DIR}/{DATA_DIR}/battery_charge_history.json"
            self.persistence = BatteryChargePersistence(
                file_path=persistence_file,
                battery_capacity=self.battery_capacity,
            )
            await self.persistence.load()

            self.electricity_service = ElectricityPriceService(country=self.electricity_country)

            await self.electricity_service.fetch_day_ahead_prices()

            using_v10 = self.data_collector.using_new_config

            if using_v10:
                power_sensors = self.data_collector.get_all_power_sensors()
                _LOGGER.info(
                    f"Battery Coordinator Setup Complete ✓ (v10.0.0 watt-based)\n"
                    f"  → Battery Capacity: {self.battery_capacity} kWh\n"
                    f"  → Update Interval: {UPDATE_INTERVAL}\n"
                    f"  → Battery Power: {self.data_collector.battery_power_sensor}\n"
                    f"  → Solar Production: {self.data_collector.solar_production_sensor}\n"
                    f"  → Inverter Output: {self.data_collector.inverter_output_sensor}\n"
                    f"  → Grid Import: {self.data_collector.grid_import_sensor}\n"
                    f"  → Grid Export: {self.data_collector.grid_export_sensor}\n"
                    f"  → House Consumption: {self.data_collector.house_consumption_sensor}\n"
                    f"  → Energy Flow Tracking: Active ✓\n"
                    f"  → Electricity Country: {self.electricity_country}\n"
                    f"  → Price Tracking: Active (aWATTar)"
                )
            else:
                _LOGGER.warning(
                    f"Battery Coordinator Setup (LEGACY v8.x mode)\n"
                    f"  → Battery Capacity: {self.battery_capacity} kWh\n"
                    f"  → Update Interval: {UPDATE_INTERVAL}\n"
                    f"  → ⚠️  Using LEGACY v8.x configuration!\n"
                    f"  → ⚠️  Migrate to v10.0.0 for full energy flow tracking!"
                )

            return True

        except Exception as e:
            _LOGGER.error(f"Error setting up BatteryCoordinator: {e}")
            return False

    async def _async_update_data(self):
        """Fetch battery data and track charging/discharging Returns: Dictionary with current battery data @zara"""
        try:

            using_v10 = self.data_collector.using_new_config

            if using_v10:
                return await self._update_v10()
            else:
                return await self._update_legacy()

        except Exception as e:
            _LOGGER.error(f"Error updating battery data: {e}")
            raise UpdateFailed(f"Error updating battery data: {e}")

    async def _update_v10(self):
        """Update using v10.0.0 watt-based system @zara"""

        power_sensors = self.data_collector.get_all_power_sensors()

        battery_power_w = power_sensors["battery_power_w"]
        solar_production_w = power_sensors["solar_production_w"]
        inverter_output_w = power_sensors["inverter_output_w"]
        grid_import_w = power_sensors["grid_import_w"]
        grid_export_w = power_sensors["grid_export_w"]
        house_consumption_w = power_sensors["house_consumption_w"]
        grid_charge_power_w = power_sensors["grid_charge_power_w"]

        current_price = (
            self.electricity_service.get_current_price() if self.electricity_service else 0.0
        )

        tracker_data = self.charge_tracker.update(
            battery_power=battery_power_w,
            solar_power=solar_production_w,
            inverter_output=inverter_output_w,
            house_consumption=house_consumption_w,
            grid_import=grid_import_w,
            grid_export=grid_export_w,
            grid_charge_power=grid_charge_power_w,
            using_new_config=True,
        )

        energy_flows = self.charge_tracker.get_energy_flows_v10()

        if self.persistence:
            await self.persistence.update_energy_flows_v10(
                timestamp=datetime.now(), energy_flows=energy_flows
            )

        summary = self.persistence.get_today_summary() if self.persistence else {}
        return {
            "battery_power_w": battery_power_w,
            "solar_production_w": solar_production_w,
            "inverter_output_w": inverter_output_w,
            "grid_import_w": grid_import_w,
            "grid_export_w": grid_export_w,
            "house_consumption_w": house_consumption_w,
            "grid_charge_power_w": grid_charge_power_w,
            "current_price": current_price,
            "energy_flows": energy_flows,
            "tracker_data": tracker_data,
            "summary": summary,
            "last_update": datetime.now().isoformat(),
            "version": "v10.0.0",
        }

    async def _update_legacy(self):
        """Update using LEGACY v8.x system @zara"""

        battery_power = self.data_collector.get_battery_power()
        if battery_power is None:
            _LOGGER.debug("Battery power sensor not available, skipping update")
            return {}

        grid_charge_power = self.data_collector.get_grid_charge_power() or 0.0

        if battery_power > 50:
            solar_power = max(0, battery_power - grid_charge_power)
        else:
            solar_power = 0.0

        current_price = (
            self.electricity_service.get_current_price() if self.electricity_service else 0.0
        )

        if battery_power > 50:
            if grid_charge_power > 50:
                await self.persistence.add_charge_event(
                    timestamp=datetime.now(),
                    source="grid",
                    power_w=grid_charge_power,
                    duration_min=UPDATE_INTERVAL.total_seconds() / 60,
                    kwh=(grid_charge_power / 1000) * (UPDATE_INTERVAL.total_seconds() / 3600),
                    price_cent_kwh=current_price,
                )

            if solar_power > 50:
                await self.persistence.add_charge_event(
                    timestamp=datetime.now(),
                    source="solar",
                    power_w=solar_power,
                    duration_min=UPDATE_INTERVAL.total_seconds() / 60,
                    kwh=(solar_power / 1000) * (UPDATE_INTERVAL.total_seconds() / 3600),
                    price_cent_kwh=0.0,
                )

        elif battery_power < -50:
            discharge_power = abs(battery_power)
            discharge_kwh = (discharge_power / 1000) * (UPDATE_INTERVAL.total_seconds() / 3600)

            today_summary = self.persistence.get_today_summary()
            total_charged = today_summary.get("grid_charge_kwh", 0) + today_summary.get(
                "solar_charge_kwh", 0
            )
            solar_ratio = (
                today_summary.get("solar_charge_kwh", 0) / total_charged
                if total_charged > 0
                else 0.0
            )

            await self.persistence.add_discharge_event(
                timestamp=datetime.now(),
                power_w=discharge_power,
                duration_min=UPDATE_INTERVAL.total_seconds() / 60,
                kwh=discharge_kwh,
                price_cent_kwh=current_price,
                solar_ratio=solar_ratio,
            )

        summary = self.persistence.get_today_summary() if self.persistence else {}
        return {
            "battery_power": battery_power,
            "grid_charge_power": grid_charge_power,
            "solar_power": solar_power,
            "current_price": current_price,
            "summary": summary,
            "last_update": datetime.now().isoformat(),
            "version": "LEGACY v8.x",
        }

    async def async_refresh_prices(self):
        """Refresh electricity prices @zara"""
        if self.electricity_service:
            try:
                await self.electricity_service.fetch_day_ahead_prices()
                _LOGGER.info("Electricity prices refreshed")
            except Exception as e:
                _LOGGER.error(f"Error refreshing electricity prices: {e}")

    async def async_daily_rollup(self):
        """Perform daily rollup tasks Called at midnight to aggregate daily data to monthly @zara"""
        try:
            if not self.persistence:
                return

            yesterday = datetime.now().date() - timedelta(days=1)
            await self.persistence.rollup_to_monthly(yesterday)

            await self.persistence.cleanup_old_events()

            await self.persistence.save()

            _LOGGER.info("Daily rollup completed")

        except Exception as e:
            _LOGGER.error(f"Error during daily rollup: {e}")

    async def async_monthly_rollup(self):
        """Perform monthly rollup tasks Called at end of month to aggregate monthly data to yearly @zara"""
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
