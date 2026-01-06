# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from ..const import (
    CONF_HOURLY,
    CONF_INVERTER_MAX_POWER,
    CONF_LEARNING_ENABLED,
    CONF_PANEL_GROUPS,
    CONF_POWER_ENTITY,
    CONF_SOLAR_CAPACITY,
    CONF_SOLAR_YIELD_TODAY,
    CONF_TOTAL_CONSUMPTION_TODAY,
    CONF_WEATHER_ENTITY,
    DEFAULT_INVERTER_MAX_POWER,
    DEFAULT_SOLAR_CAPACITY,
    DOMAIN,
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

    # Panel groups configuration (optional)
    panel_groups: list[dict[str, Any]] = field(default_factory=list)

    # Inverter clipping configuration (optional)
    inverter_max_power: float = 0.0  # 0 = disabled

    @property
    def has_panel_groups(self) -> bool:
        """Check if panel groups are configured. @zara"""
        return len(self.panel_groups) > 0

    @property
    def has_inverter_clipping(self) -> bool:
        """Check if inverter clipping is enabled. @zara"""
        return self.inverter_max_power > 0.0

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

        # Extract panel groups if configured
        panel_groups = entry.data.get(CONF_PANEL_GROUPS, [])
        if panel_groups:
            _LOGGER.info(
                "Panel groups configured: %d groups, total %.2f kWp",
                len(panel_groups),
                float(solar_capacity_value),
            )

        # Extract inverter max power for clipping
        inverter_max_power = entry.data.get(CONF_INVERTER_MAX_POWER, DEFAULT_INVERTER_MAX_POWER)
        if inverter_max_power and inverter_max_power > 0:
            _LOGGER.info(
                "Inverter clipping enabled: max %.2f kW (forecasts capped, clipped hours excluded from training)",
                float(inverter_max_power),
            )

        return CoordinatorConfiguration(
            solar_capacity=float(solar_capacity_value),
            learning_enabled=entry.options.get(CONF_LEARNING_ENABLED, True),
            enable_hourly=entry.options.get(CONF_HOURLY, False),
            power_entity=entry.data.get(CONF_POWER_ENTITY),
            solar_yield_today=entry.data.get(CONF_SOLAR_YIELD_TODAY),
            primary_weather_entity=entry.data.get(CONF_WEATHER_ENTITY),
            total_consumption_today=entry.data.get(CONF_TOTAL_CONSUMPTION_TODAY),
            panel_groups=panel_groups,
            inverter_max_power=float(inverter_max_power) if inverter_max_power else 0.0,
        )

    @staticmethod
    def setup_data_directory(hass: HomeAssistant) -> Path:
        """Setup and return data directory path @zara"""
        config_dir = hass.config.path()
        data_dir_path = Path(config_dir) / DOMAIN
        return data_dir_path
