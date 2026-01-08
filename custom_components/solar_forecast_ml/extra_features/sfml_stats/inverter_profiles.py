# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""Inverter profiles and auto-discovery for SFML Stats. @zara"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from .const import (
    CONF_SENSOR_SOLAR_POWER,
    CONF_SENSOR_SOLAR_TO_HOUSE,
    CONF_SENSOR_SOLAR_TO_BATTERY,
    CONF_SENSOR_BATTERY_TO_HOUSE,
    CONF_SENSOR_BATTERY_TO_GRID,
    CONF_SENSOR_GRID_TO_HOUSE,
    CONF_SENSOR_GRID_TO_BATTERY,
    CONF_SENSOR_HOUSE_TO_GRID,
    CONF_SENSOR_BATTERY_SOC,
    CONF_SENSOR_BATTERY_POWER,
    CONF_SENSOR_HOME_CONSUMPTION,
    CONF_SENSOR_SOLAR_YIELD_DAILY,
    CONF_SENSOR_GRID_IMPORT_DAILY,
    CONF_SENSOR_SMARTMETER_IMPORT,
    CONF_SENSOR_SMARTMETER_EXPORT,
    CONF_WEATHER_ENTITY,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class InverterProfile:
    """Definition of an inverter/energy system profile. @zara"""

    id: str
    name: str
    manufacturer: str
    description: str
    # Integration domains to detect
    detection_domains: list[str] = field(default_factory=list)
    # Entity ID patterns to search for (regex)
    sensor_patterns: dict[str, list[str]] = field(default_factory=dict)
    # Whether this system typically has a battery
    has_battery: bool = False
    # Priority (higher = prefer this profile if multiple match)
    priority: int = 0


# =============================================================================
# Inverter Profile Definitions
# =============================================================================

INVERTER_PROFILES: dict[str, InverterProfile] = {
    "fronius": InverterProfile(
        id="fronius",
        name="Fronius",
        manufacturer="Fronius International",
        description="Fronius Symo, Primo, Gen24 Inverters",
        detection_domains=["fronius"],
        sensor_patterns={
            CONF_SENSOR_SOLAR_POWER: [
                r"sensor\.fronius_.*_power_photovoltaics",
                r"sensor\.fronius_.*_pv_power",
                r"sensor\.fronius_.*power_flow_.*pv",
            ],
            CONF_SENSOR_HOME_CONSUMPTION: [
                r"sensor\.fronius_.*_power_load",
                r"sensor\.fronius_.*_house_load",
                r"sensor\.fronius_.*power_flow_.*load",
            ],
            CONF_SENSOR_HOUSE_TO_GRID: [
                r"sensor\.fronius_.*_power_grid_export",
                r"sensor\.fronius_.*power_flow_.*grid.*export",
            ],
            CONF_SENSOR_GRID_TO_HOUSE: [
                r"sensor\.fronius_.*_power_grid_import",
                r"sensor\.fronius_.*power_flow_.*grid.*import",
            ],
            CONF_SENSOR_BATTERY_SOC: [
                r"sensor\.fronius_.*_battery_.*soc",
                r"sensor\.fronius_.*_state_of_charge",
            ],
            CONF_SENSOR_BATTERY_POWER: [
                r"sensor\.fronius_.*_power_battery",
                r"sensor\.fronius_.*power_flow_.*battery",
            ],
            CONF_SENSOR_SOLAR_YIELD_DAILY: [
                r"sensor\.fronius_.*_energy_day",
                r"sensor\.fronius_.*_daily_yield",
            ],
        },
        has_battery=True,
        priority=10,
    ),
    "sma": InverterProfile(
        id="sma",
        name="SMA",
        manufacturer="SMA Solar Technology",
        description="SMA Sunny Boy, Sunny Tripower, Home Manager",
        detection_domains=["sma", "sunny_boy"],
        sensor_patterns={
            CONF_SENSOR_SOLAR_POWER: [
                r"sensor\.sma_.*_pv_power",
                r"sensor\.sma_.*_total_power",
                r"sensor\.sunny_.*_pv_power",
            ],
            CONF_SENSOR_HOME_CONSUMPTION: [
                r"sensor\.sma_.*_consumption",
                r"sensor\.sma_.*_grid_consumption",
            ],
            CONF_SENSOR_HOUSE_TO_GRID: [
                r"sensor\.sma_.*_grid_feed_in",
                r"sensor\.sma_.*_export",
            ],
            CONF_SENSOR_GRID_TO_HOUSE: [
                r"sensor\.sma_.*_grid_import",
                r"sensor\.sma_.*_purchased",
            ],
            CONF_SENSOR_BATTERY_SOC: [
                r"sensor\.sma_.*_battery_soc",
                r"sensor\.sma_.*_state_of_charge",
            ],
            CONF_SENSOR_BATTERY_POWER: [
                r"sensor\.sma_.*_battery_power",
                r"sensor\.sma_.*_battery_discharge",
            ],
            CONF_SENSOR_SOLAR_YIELD_DAILY: [
                r"sensor\.sma_.*_daily_yield",
                r"sensor\.sma_.*_energy_today",
            ],
        },
        has_battery=True,
        priority=10,
    ),
    "huawei": InverterProfile(
        id="huawei",
        name="Huawei FusionSolar",
        manufacturer="Huawei",
        description="Huawei SUN2000 Inverters, LUNA2000 Battery",
        detection_domains=["huawei_solar", "fusion_solar"],
        sensor_patterns={
            CONF_SENSOR_SOLAR_POWER: [
                r"sensor\..*_input_power",
                r"sensor\.huawei_.*_pv_power",
                r"sensor\.inverter_.*_input_power",
            ],
            CONF_SENSOR_HOME_CONSUMPTION: [
                r"sensor\..*_house_consumption",
                r"sensor\.huawei_.*_load_power",
            ],
            CONF_SENSOR_HOUSE_TO_GRID: [
                r"sensor\..*_grid_export",
                r"sensor\.huawei_.*_feed_in",
                r"sensor\..*_active_power",  # negative = export
            ],
            CONF_SENSOR_GRID_TO_HOUSE: [
                r"sensor\..*_grid_import",
                r"sensor\.huawei_.*_consumption_from_grid",
            ],
            CONF_SENSOR_BATTERY_SOC: [
                r"sensor\..*_battery_state_of_capacity",
                r"sensor\..*_battery_soc",
                r"sensor\.luna.*_soc",
            ],
            CONF_SENSOR_BATTERY_POWER: [
                r"sensor\..*_battery_charge_discharge_power",
                r"sensor\..*_battery_power",
            ],
            CONF_SENSOR_SOLAR_YIELD_DAILY: [
                r"sensor\..*_daily_yield",
                r"sensor\.huawei_.*_yield_today",
            ],
        },
        has_battery=True,
        priority=10,
    ),
    "kostal": InverterProfile(
        id="kostal",
        name="Kostal",
        manufacturer="Kostal Solar Electric",
        description="Kostal Piko, Plenticore Inverters",
        detection_domains=["kostal_plenticore", "kostal_piko"],
        sensor_patterns={
            CONF_SENSOR_SOLAR_POWER: [
                r"sensor\.kostal_.*_pv_power",
                r"sensor\.plenticore_.*_pv_power",
                r"sensor\.piko_.*_dc_power",
            ],
            CONF_SENSOR_HOME_CONSUMPTION: [
                r"sensor\.kostal_.*_home_consumption",
                r"sensor\.plenticore_.*_consumption",
            ],
            CONF_SENSOR_HOUSE_TO_GRID: [
                r"sensor\.kostal_.*_grid_export",
                r"sensor\.plenticore_.*_grid_power",  # negative = export
            ],
            CONF_SENSOR_GRID_TO_HOUSE: [
                r"sensor\.kostal_.*_grid_import",
            ],
            CONF_SENSOR_BATTERY_SOC: [
                r"sensor\.kostal_.*_battery_soc",
                r"sensor\.plenticore_.*_soc",
            ],
            CONF_SENSOR_BATTERY_POWER: [
                r"sensor\.kostal_.*_battery_power",
            ],
            CONF_SENSOR_SOLAR_YIELD_DAILY: [
                r"sensor\.kostal_.*_yield_day",
                r"sensor\.plenticore_.*_energy_day",
            ],
        },
        has_battery=True,
        priority=10,
    ),
    "growatt": InverterProfile(
        id="growatt",
        name="Growatt",
        manufacturer="Growatt New Energy",
        description="Growatt Inverters via Growatt Server",
        detection_domains=["growatt_server"],
        sensor_patterns={
            CONF_SENSOR_SOLAR_POWER: [
                r"sensor\.growatt_.*_power",
                r"sensor\.growatt_.*_ppv",
            ],
            CONF_SENSOR_HOME_CONSUMPTION: [
                r"sensor\.growatt_.*_local_load",
                r"sensor\.growatt_.*_consumption",
            ],
            CONF_SENSOR_HOUSE_TO_GRID: [
                r"sensor\.growatt_.*_grid_export",
                r"sensor\.growatt_.*_export_to_grid",
            ],
            CONF_SENSOR_GRID_TO_HOUSE: [
                r"sensor\.growatt_.*_import_from_grid",
            ],
            CONF_SENSOR_BATTERY_SOC: [
                r"sensor\.growatt_.*_soc",
                r"sensor\.growatt_.*_battery_soc",
            ],
            CONF_SENSOR_BATTERY_POWER: [
                r"sensor\.growatt_.*_battery_power",
            ],
            CONF_SENSOR_SOLAR_YIELD_DAILY: [
                r"sensor\.growatt_.*_today_energy",
                r"sensor\.growatt_.*_energy_today",
            ],
        },
        has_battery=True,
        priority=10,
    ),
    "sungrow": InverterProfile(
        id="sungrow",
        name="Sungrow",
        manufacturer="Sungrow Power Supply",
        description="Sungrow Inverters via Modbus",
        detection_domains=["sungrow"],
        sensor_patterns={
            CONF_SENSOR_SOLAR_POWER: [
                r"sensor\.sungrow_.*_pv_power",
                r"sensor\.sungrow_.*_total_dc_power",
            ],
            CONF_SENSOR_HOME_CONSUMPTION: [
                r"sensor\.sungrow_.*_load_power",
                r"sensor\.sungrow_.*_house_load",
            ],
            CONF_SENSOR_HOUSE_TO_GRID: [
                r"sensor\.sungrow_.*_export_power",
                r"sensor\.sungrow_.*_feed_in",
            ],
            CONF_SENSOR_GRID_TO_HOUSE: [
                r"sensor\.sungrow_.*_import_power",
                r"sensor\.sungrow_.*_grid_import",
            ],
            CONF_SENSOR_BATTERY_SOC: [
                r"sensor\.sungrow_.*_battery_soc",
                r"sensor\.sungrow_.*_soc",
            ],
            CONF_SENSOR_BATTERY_POWER: [
                r"sensor\.sungrow_.*_battery_power",
            ],
            CONF_SENSOR_SOLAR_YIELD_DAILY: [
                r"sensor\.sungrow_.*_daily_.*generation",
                r"sensor\.sungrow_.*_today_energy",
            ],
        },
        has_battery=True,
        priority=10,
    ),
    "enphase": InverterProfile(
        id="enphase",
        name="Enphase",
        manufacturer="Enphase Energy",
        description="Enphase Microinverters, Envoy Gateway",
        detection_domains=["enphase_envoy"],
        sensor_patterns={
            CONF_SENSOR_SOLAR_POWER: [
                r"sensor\.envoy_.*_current_power_production",
                r"sensor\.enphase_.*_production",
            ],
            CONF_SENSOR_HOME_CONSUMPTION: [
                r"sensor\.envoy_.*_current_power_consumption",
                r"sensor\.enphase_.*_consumption",
            ],
            CONF_SENSOR_HOUSE_TO_GRID: [
                r"sensor\.envoy_.*_net_power",  # negative = export
            ],
            CONF_SENSOR_SOLAR_YIELD_DAILY: [
                r"sensor\.envoy_.*_today_s_energy_production",
                r"sensor\.enphase_.*_energy_today",
            ],
        },
        has_battery=False,
        priority=10,
    ),
    "solaredge": InverterProfile(
        id="solaredge",
        name="SolarEdge",
        manufacturer="SolarEdge Technologies",
        description="SolarEdge Inverters via Modbus or Cloud",
        detection_domains=["solaredge", "solaredge_modbus"],
        sensor_patterns={
            CONF_SENSOR_SOLAR_POWER: [
                r"sensor\.solaredge_.*_ac_power",
                r"sensor\.solaredge_.*_current_power",
            ],
            CONF_SENSOR_HOME_CONSUMPTION: [
                r"sensor\.solaredge_.*_consumption",
            ],
            CONF_SENSOR_HOUSE_TO_GRID: [
                r"sensor\.solaredge_.*_exported",
                r"sensor\.solaredge_.*_grid_power",  # negative = export
            ],
            CONF_SENSOR_GRID_TO_HOUSE: [
                r"sensor\.solaredge_.*_imported",
            ],
            CONF_SENSOR_BATTERY_SOC: [
                r"sensor\.solaredge_.*_battery_.*soc",
                r"sensor\.solaredge_.*_state_of_energy",
            ],
            CONF_SENSOR_BATTERY_POWER: [
                r"sensor\.solaredge_.*_battery_power",
            ],
            CONF_SENSOR_SOLAR_YIELD_DAILY: [
                r"sensor\.solaredge_.*_energy_today",
                r"sensor\.solaredge_.*_day_energy",
            ],
        },
        has_battery=True,
        priority=10,
    ),
    "goodwe": InverterProfile(
        id="goodwe",
        name="GoodWe",
        manufacturer="GoodWe",
        description="GoodWe Inverters via SEMS Portal or local",
        detection_domains=["goodwe", "sems"],
        sensor_patterns={
            CONF_SENSOR_SOLAR_POWER: [
                r"sensor\.goodwe_.*_pv_power",
                r"sensor\.sems_.*_current_power",
            ],
            CONF_SENSOR_HOME_CONSUMPTION: [
                r"sensor\.goodwe_.*_load",
                r"sensor\.goodwe_.*_consumption",
            ],
            CONF_SENSOR_HOUSE_TO_GRID: [
                r"sensor\.goodwe_.*_grid_export",
                r"sensor\.goodwe_.*_on_grid.*export",
            ],
            CONF_SENSOR_GRID_TO_HOUSE: [
                r"sensor\.goodwe_.*_grid_import",
            ],
            CONF_SENSOR_BATTERY_SOC: [
                r"sensor\.goodwe_.*_battery_soc",
            ],
            CONF_SENSOR_BATTERY_POWER: [
                r"sensor\.goodwe_.*_battery_power",
            ],
            CONF_SENSOR_SOLAR_YIELD_DAILY: [
                r"sensor\.goodwe_.*_e_day",
                r"sensor\.sems_.*_today_energy",
            ],
        },
        has_battery=True,
        priority=10,
    ),
    "shelly_em": InverterProfile(
        id="shelly_em",
        name="Shelly EM/3EM",
        manufacturer="Shelly (Allterco)",
        description="Shelly Energy Meter (for grid monitoring only)",
        detection_domains=["shelly"],
        sensor_patterns={
            CONF_SENSOR_SMARTMETER_IMPORT: [
                r"sensor\.shelly_.*_channel_.*_power",
                r"sensor\.shelly_.*_em.*_power",
            ],
            CONF_SENSOR_SMARTMETER_EXPORT: [
                r"sensor\.shelly_.*_channel_.*_power",  # negative = export
            ],
        },
        has_battery=False,
        priority=5,  # Lower priority, often combined with inverter
    ),
    "anker_solix": InverterProfile(
        id="anker_solix",
        name="Anker Solix",
        manufacturer="Anker",
        description="Anker Solix Solarbank, Balkonkraftwerk (E1600, 2 Pro, X1)",
        detection_domains=["anker_solix"],
        sensor_patterns={
            # Patterns for Anker Solix sensors - format: sensor.{serial}_{key}
            CONF_SENSOR_SOLAR_POWER: [
                r".*_solar_power_[1-4]$",           # Individual string power
                r".*_input_power$",                 # Total input power
                r".*_dc_input_power$",              # DC input
                r".*_dc_input_power_total$",        # Total DC input
            ],
            CONF_SENSOR_HOME_CONSUMPTION: [
                r".*_ac_to_home_load$",             # AC output to home
                r".*_dc_output_power$",             # DC output power
                r".*_output_power_total$",          # Total output
                r".*_solarbank_output_power$",      # Solarbank output
            ],
            CONF_SENSOR_BATTERY_SOC: [
                r".*_state_of_charge$",             # General SOC
                r".*_main_battery_soc$",            # Main battery SOC
            ],
            CONF_SENSOR_BATTERY_POWER: [
                r".*_battery_power_signed$",        # Signed battery power
                r".*_bat_charge_power$",            # Charging power
            ],
            CONF_SENSOR_BATTERY_TO_HOUSE: [
                r".*_bat_discharge_power$",         # Discharge power
            ],
            CONF_SENSOR_SOLAR_YIELD_DAILY: [
                r".*_daily_.*energy$",              # Daily energy
                r".*_energy_today$",                # Energy today
            ],
        },
        has_battery=True,
        priority=10,
    ),
    "manual": InverterProfile(
        id="manual",
        name="Manual Configuration",
        manufacturer="",
        description="Manually select all sensors",
        detection_domains=[],
        sensor_patterns={},
        has_battery=True,
        priority=0,
    ),
}


class InverterDiscovery:
    """Discovers installed inverter integrations and maps sensors. @zara"""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the discovery. @zara"""
        self._hass = hass
        self._detected_profiles: list[InverterProfile] = []
        self._all_sensors: list[str] = []
        self._entity_registry: er.EntityRegistry | None = None
        self._domain_entities: dict[str, list[str]] = {}

    async def async_discover(self) -> list[InverterProfile]:
        """Discover installed inverter integrations. @zara"""
        self._detected_profiles = []
        self._all_sensors = []
        self._domain_entities = {}

        # Get entity registry for platform/domain detection
        self._entity_registry = er.async_get(self._hass)

        # Build mapping of integration domains to their entities
        for entity_entry in self._entity_registry.entities.values():
            platform = entity_entry.platform
            if platform not in self._domain_entities:
                self._domain_entities[platform] = []
            self._domain_entities[platform].append(entity_entry.entity_id)

        _LOGGER.debug("Found domains: %s", list(self._domain_entities.keys()))

        # Get all sensor entity IDs
        states = self._hass.states.async_all("sensor")
        self._all_sensors = [state.entity_id for state in states]

        # Also get weather entities
        weather_states = self._hass.states.async_all("weather")
        weather_entities = [state.entity_id for state in weather_states]

        _LOGGER.debug("Found %d sensors for discovery", len(self._all_sensors))

        # Check each profile
        for profile_id, profile in INVERTER_PROFILES.items():
            if profile_id == "manual":
                continue  # Skip manual profile

            if self._check_profile(profile):
                self._detected_profiles.append(profile)
                _LOGGER.info("Detected inverter profile: %s", profile.name)

        # Sort by priority (highest first)
        self._detected_profiles.sort(key=lambda p: p.priority, reverse=True)

        return self._detected_profiles

    def _check_profile(self, profile: InverterProfile) -> bool:
        """Check if a profile matches installed entities. @zara"""
        # Method 1: Check via entity registry platform/domain
        for domain in profile.detection_domains:
            if domain in self._domain_entities:
                _LOGGER.debug(
                    "Profile %s matched by domain %s in entity registry",
                    profile.id, domain
                )
                return True

        # Method 2: Check if domain appears in entity names (fallback)
        for domain in profile.detection_domains:
            for entity_id in self._all_sensors:
                if domain in entity_id.lower():
                    _LOGGER.debug(
                        "Profile %s matched by domain %s in entity_id %s",
                        profile.id, domain, entity_id
                    )
                    return True

        # Method 3: Check if any sensor pattern matches
        for config_key, patterns in profile.sensor_patterns.items():
            for pattern in patterns:
                try:
                    regex = re.compile(pattern, re.IGNORECASE)
                    for entity_id in self._all_sensors:
                        if regex.match(entity_id):
                            _LOGGER.debug(
                                "Profile %s matched by pattern %s on %s",
                                profile.id, pattern, entity_id
                            )
                            return True
                except re.error:
                    _LOGGER.warning("Invalid regex pattern: %s", pattern)

        return False

    def get_sensor_mapping(self, profile: InverterProfile) -> dict[str, str | None]:
        """Get sensor mapping for a profile. @zara"""
        mapping: dict[str, str | None] = {}

        # Get entities from profile's integration domain
        domain_entities: list[str] = []
        for domain in profile.detection_domains:
            domain_entities.extend(self._domain_entities.get(domain, []))

        _LOGGER.debug(
            "Profile %s has %d entities from domain",
            profile.id, len(domain_entities)
        )

        for config_key, patterns in profile.sensor_patterns.items():
            mapping[config_key] = None

            # Method 1: Search in domain entities first (most reliable)
            for pattern in patterns:
                if mapping[config_key]:
                    break
                try:
                    regex = re.compile(pattern, re.IGNORECASE)
                    for entity_id in domain_entities:
                        if regex.match(entity_id) or regex.search(entity_id):
                            state = self._hass.states.get(entity_id)
                            if state and state.state not in ("unknown", "unavailable"):
                                mapping[config_key] = entity_id
                                _LOGGER.debug(
                                    "Mapped %s -> %s (domain)", config_key, entity_id
                                )
                                break
                except re.error:
                    continue

            # Method 2: Fallback - search all sensors by pattern
            if not mapping[config_key]:
                for pattern in patterns:
                    if mapping[config_key]:
                        break
                    try:
                        regex = re.compile(pattern, re.IGNORECASE)
                        for entity_id in self._all_sensors:
                            if regex.match(entity_id):
                                state = self._hass.states.get(entity_id)
                                if state and state.state not in ("unknown", "unavailable"):
                                    mapping[config_key] = entity_id
                                    _LOGGER.debug(
                                        "Mapped %s -> %s (pattern)", config_key, entity_id
                                    )
                                    break
                    except re.error:
                        continue

        # Try to find weather entity
        weather_states = self._hass.states.async_all("weather")
        if weather_states:
            # Prefer "weather.home" or first available
            for state in weather_states:
                if state.entity_id == "weather.home":
                    mapping[CONF_WEATHER_ENTITY] = state.entity_id
                    break
            if not mapping.get(CONF_WEATHER_ENTITY) and weather_states:
                mapping[CONF_WEATHER_ENTITY] = weather_states[0].entity_id

        return mapping

    def get_detected_profiles(self) -> list[InverterProfile]:
        """Return detected profiles. @zara"""
        return self._detected_profiles

    def get_profile_by_id(self, profile_id: str) -> InverterProfile | None:
        """Get a profile by ID. @zara"""
        return INVERTER_PROFILES.get(profile_id)


def get_profile_choices() -> dict[str, str]:
    """Get choices for profile selection dropdown. @zara"""
    choices = {}
    for profile_id, profile in INVERTER_PROFILES.items():
        if profile_id == "manual":
            choices[profile_id] = "🔧 Manual Configuration"
        else:
            choices[profile_id] = f"{profile.name} ({profile.manufacturer})"
    return choices
