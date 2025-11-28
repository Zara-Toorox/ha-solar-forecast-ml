"""Coordinator Initialization Helpers - Extract __init__ logic V10.0.0 @zara

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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from ..const import (
    CONF_BATTERY_ENABLED,
    CONF_ELECTRICITY_COUNTRY,
    CONF_ELECTRICITY_ENABLED,
    CONF_HOURLY,
    CONF_LEARNING_ENABLED,
    CONF_POWER_ENTITY,
    CONF_SOLAR_CAPACITY,
    CONF_SOLAR_YIELD_TODAY,
    CONF_TOTAL_CONSUMPTION_TODAY,
    CONF_UPDATE_INTERVAL,
    CONF_WEATHER_ENTITY,
    DEFAULT_ELECTRICITY_COUNTRY,
    DEFAULT_SOLAR_CAPACITY,
    DOMAIN,
    UPDATE_INTERVAL,
)

_LOGGER = logging.getLogger(__name__)

@dataclass
class CoordinatorConfiguration:
    """Configuration data extracted from ConfigEntry"""

    solar_capacity: float
    learning_enabled: bool
    enable_hourly: bool

    power_entity: Optional[str]
    solar_yield_today: Optional[str]
    primary_weather_entity: Optional[str]
    total_consumption_today: Optional[str]

    battery_enabled: bool
    electricity_enabled: bool
    electricity_country: str

class CoordinatorInitHelpers:
    """Helper methods for coordinator initialization"""

    @staticmethod
    def extract_configuration(entry: ConfigEntry) -> CoordinatorConfiguration:
        """Extract configuration from entry @zara"""

        solar_capacity_value = entry.data.get(CONF_SOLAR_CAPACITY)
        if solar_capacity_value is None or solar_capacity_value == 0:

            solar_capacity_value = entry.data.get("plant_kwp", DEFAULT_SOLAR_CAPACITY)
            if solar_capacity_value != DEFAULT_SOLAR_CAPACITY:
                _LOGGER.warning(
                    f"Using legacy 'plant_kwp' value: {solar_capacity_value} kW. "
                    f"Please reconfigure to update."
                )

        return CoordinatorConfiguration(
            solar_capacity=float(solar_capacity_value),
            learning_enabled=entry.options.get(CONF_LEARNING_ENABLED, True),
            enable_hourly=entry.options.get(CONF_HOURLY, False),
            power_entity=entry.data.get(CONF_POWER_ENTITY),
            solar_yield_today=entry.data.get(CONF_SOLAR_YIELD_TODAY),
            primary_weather_entity=entry.data.get(CONF_WEATHER_ENTITY),
            total_consumption_today=entry.data.get(CONF_TOTAL_CONSUMPTION_TODAY),
            battery_enabled=entry.options.get(CONF_BATTERY_ENABLED, False),
            electricity_enabled=entry.options.get(CONF_ELECTRICITY_ENABLED, False),
            electricity_country=entry.options.get(CONF_ELECTRICITY_COUNTRY)
            or entry.data.get(CONF_ELECTRICITY_COUNTRY, DEFAULT_ELECTRICITY_COUNTRY),
        )

    @staticmethod
    def initialize_battery_collector(hass: HomeAssistant, entry: ConfigEntry, enabled: bool):
        """Initialize battery data collector if enabled @zara"""
        if not enabled:
            return None

        try:
            from ..battery.battery_data_collector import BatteryDataCollector

            collector = BatteryDataCollector(hass, entry)
            _LOGGER.info("BatteryDataCollector initialized successfully")
            return collector
        except Exception as e:
            _LOGGER.error(f"Failed to initialize BatteryDataCollector: {e}")
            return None

    @staticmethod
    def initialize_electricity_service(enabled: bool, country: str):
        """Initialize electricity price service if enabled @zara"""
        if not enabled:
            return None

        try:
            from ..battery.electricity_price_service import ElectricityPriceService

            service = ElectricityPriceService(country=country)
            _LOGGER.info(
                f"ElectricityPriceService initialized for {country} "
                f"using aWATTar API (free, no registration)"
            )
            return service
        except Exception as e:
            _LOGGER.error(f"Failed to initialize ElectricityPriceService: {e}")
            return None

    @staticmethod
    def setup_data_directory(hass: HomeAssistant) -> Path:
        """Setup and return data directory path @zara"""
        config_dir = hass.config.path()
        data_dir_path = Path(config_dir) / DOMAIN
        return data_dir_path
