"""Battery Charge Tracker - Real-time Grid vs Solar Charging Detection V10.0.0 @zara

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
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

POWER_TOLERANCE = 50
MIN_TRACKING_POWER = 100

class ChargeEvent:
    """Represents a single charging event"""

    def __init__(
        self,
        timestamp: datetime,
        source: str,
        power: float,
        duration_minutes: float = 1.0,
    ):
        self.timestamp = timestamp
        self.source = source
        self.power = power
        self.duration_minutes = duration_minutes

    @property
    def energy_kwh(self) -> float:
        """Calculate energy in kWh for this event @zara"""

        return (self.power / 1000) * (self.duration_minutes / 60)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage @zara"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "hour": self.timestamp.hour,
            "source": self.source,
            "power": self.power,
            "duration_minutes": self.duration_minutes,
            "energy_kwh": round(self.energy_kwh, 4),
        }

class BatteryChargeTracker:
    """Tracks battery charging sources using real-time power flow analysis

    V10.0.0: Complete energy flow tracking
    - Solar → House/Battery/Grid
    - Grid → House/Battery
    - Battery → House
    - All values in Watt (W), integrated over time to kWh
    """

    def __init__(
        self, hass: HomeAssistant, battery_capacity: float = 10.0, using_new_config: bool = False
    ):
        """Initialize charge tracker

        Args:
            hass: Home Assistant instance
            battery_capacity: Battery capacity in kWh
            using_new_config: Force v10 watt-based flow even bei 0W-Verbrauch
        """
        self.hass = hass
        self.battery_capacity = battery_capacity
        self._using_new_config = using_new_config

        self._grid_charge_today_kwh = 0.0
        self._solar_charge_today_kwh = 0.0
        self._discharge_today_kwh = 0.0

        self._solar_to_house_kwh = 0.0
        self._solar_to_battery_kwh = 0.0
        self._solar_to_grid_kwh = 0.0
        self._grid_to_house_kwh = 0.0
        self._grid_to_battery_kwh = 0.0
        self._battery_to_house_kwh = 0.0
        self._grid_import_today_kwh = 0.0
        self._grid_export_today_kwh = 0.0

        self._charge_events: List[ChargeEvent] = []

        self._last_update: Optional[datetime] = None
        self._last_status: str = "idle"

        self._last_reset_date: Optional[str] = None

        _LOGGER.info(
            f"BatteryChargeTracker initialized (v10.0.0) - Capacity: {battery_capacity} kWh, "
            f"mode={'watt-based' if using_new_config else 'auto'}"
        )

    def update(
        self,
        battery_power: float,
        solar_power: float = 0.0,
        inverter_output: float = 0.0,
        house_consumption: float = 0.0,
        grid_import: float = 0.0,
        grid_export: float = 0.0,
        grid_charge_power: float = 0.0,
        output_power: float = 0.0,
        using_new_config: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update tracker with current power values (v10.0.0)

        Args:
            battery_power: Battery power (W, +charge/-discharge)
            solar_power: Solar production (W, ≥0)
            inverter_output: Inverter AC output (W, ≥0)
            house_consumption: House consumption (W, ≥0)
            grid_import: Grid import (W, ≥0)
            grid_export: Grid export (W, ≥0)
            grid_charge_power: Grid charge power (W, ≥0) - optional
            output_power: LEGACY - for backwards compatibility

        Returns:
            Dictionary with current status and accumulated values
        """
        now = datetime.now()

        self._check_daily_reset(now)

        duration_minutes = 1.0
        if self._last_update:
            delta = (now - self._last_update).total_seconds() / 60
            duration_minutes = max(0.1, min(delta, 60))

        use_new_flow = self._using_new_config if using_new_config is None else using_new_config
        if use_new_flow:
            energy_flows = self._calculate_energy_flow_v10(
                battery_power=battery_power,
                solar_power=solar_power,
                inverter_output=inverter_output,
                house_consumption=house_consumption,
                grid_import=grid_import,
                grid_export=grid_export,
                grid_charge_power=grid_charge_power,
                duration_minutes=duration_minutes,
            )
            self._update_energy_flows_v10(energy_flows)
            current_status = energy_flows["status"]
        else:

            status_info = self._analyze_power_flow(
                battery_power=battery_power,
                solar_power=solar_power,
                output_power=output_power,
            )
            current_status = status_info["status"]

        if current_status == "charging_solar":
            energy_kwh = (battery_power / 1000) * (duration_minutes / 60)
            self._solar_charge_today_kwh += energy_kwh

            event = ChargeEvent(
                timestamp=now,
                source="solar",
                power=battery_power,
                duration_minutes=duration_minutes,
            )
            self._charge_events.append(event)

            if energy_kwh > 0.01:
                _LOGGER.debug(
                    f"Solar charging: {battery_power}W for {duration_minutes:.1f}min = "
                    f"{energy_kwh:.3f} kWh"
                )

        elif current_status == "charging_grid":
            energy_kwh = (battery_power / 1000) * (duration_minutes / 60)
            self._grid_charge_today_kwh += energy_kwh

            event = ChargeEvent(
                timestamp=now,
                source="grid",
                power=battery_power,
                duration_minutes=duration_minutes,
            )
            self._charge_events.append(event)

            if energy_kwh > 0.01:
                _LOGGER.debug(
                    f"Grid charging: {battery_power}W for {duration_minutes:.1f}min = "
                    f"{energy_kwh:.3f} kWh"
                )

        elif current_status == "discharging":
            energy_kwh = (abs(battery_power) / 1000) * (duration_minutes / 60)
            self._discharge_today_kwh += energy_kwh

            if energy_kwh > 0.01:
                _LOGGER.debug(
                    f"Discharging: {abs(battery_power)}W for {duration_minutes:.1f}min = "
                    f"{energy_kwh:.3f} kWh"
                )

        self._last_update = now
        self._last_status = current_status

        return self.get_summary()

    def _analyze_power_flow(
        self,
        battery_power: float,
        solar_power: float,
        output_power: float,
    ) -> Dict[str, Any]:
        """Analyze power flow to determine charging source Args: battery_power: Battery power in W (+charge/-discharge) solar_power: Solar production in W output_power: Output power in W Returns: Dictionary with status and power breakdown"""

        if battery_power < -POWER_TOLERANCE:
            return {
                "status": "discharging",
                "discharge_power": abs(battery_power),
                "grid_charge_power": 0,
                "solar_charge_power": 0,
            }

        elif battery_power > POWER_TOLERANCE:

            total_demand = output_power + battery_power

            tolerance = max(MIN_TRACKING_POWER, total_demand * 0.1)

            if solar_power >= (total_demand - tolerance):
                return {
                    "status": "charging_solar",
                    "solar_charge_power": battery_power,
                    "grid_charge_power": 0,
                    "discharge_power": 0,
                    "solar_available": solar_power,
                    "demand": total_demand,
                }
            else:
                return {
                    "status": "charging_grid",
                    "grid_charge_power": battery_power,
                    "solar_charge_power": 0,
                    "discharge_power": 0,
                    "solar_available": solar_power,
                    "demand": total_demand,
                }

        else:
            return {
                "status": "idle",
                "discharge_power": 0,
                "grid_charge_power": 0,
                "solar_charge_power": 0,
            }

    def _check_daily_reset(self, now: datetime):
        """Reset daily counters at midnight (local time) @zara"""
        today = now.date().isoformat()

        if self._last_reset_date != today:
            if self._last_reset_date is not None:
                _LOGGER.info(
                    f"Daily reset (local time) - "
                    f"Grid Charge: {self._grid_charge_today_kwh:.2f} kWh, "
                    f"Solar Charge: {self._solar_charge_today_kwh:.2f} kWh, "
                    f"Discharge: {self._discharge_today_kwh:.2f} kWh, "
                    f"Grid Import: {self._grid_import_today_kwh:.2f} kWh, "
                    f"Grid Export: {self._grid_export_today_kwh:.2f} kWh"
                )

            self._grid_charge_today_kwh = 0.0
            self._solar_charge_today_kwh = 0.0
            self._discharge_today_kwh = 0.0

            self._solar_to_house_kwh = 0.0
            self._solar_to_battery_kwh = 0.0
            self._solar_to_grid_kwh = 0.0
            self._grid_to_house_kwh = 0.0
            self._grid_to_battery_kwh = 0.0
            self._battery_to_house_kwh = 0.0
            self._grid_import_today_kwh = 0.0
            self._grid_export_today_kwh = 0.0

            self._charge_events = []
            self._last_reset_date = today

    def get_grid_charge_today(self) -> float:
        """Get total grid charge today in kWh @zara"""
        return round(self._grid_charge_today_kwh, 3)

    def get_solar_charge_today(self) -> float:
        """Get total solar charge today in kWh @zara"""
        return round(self._solar_charge_today_kwh, 3)

    def get_discharge_today(self) -> float:
        """Get total discharge today in kWh @zara"""
        return round(self._discharge_today_kwh, 3)

    def get_grid_charge_events(self) -> List[Dict[str, Any]]:
        """Get all grid charging events today Returns: List of charge event dictionaries with hour and energy @zara"""
        return [event.to_dict() for event in self._charge_events if event.source == "grid"]

    def get_solar_charge_events(self) -> List[Dict[str, Any]]:
        """Get all solar charging events today Returns: List of charge event dictionaries with hour and energy @zara"""
        return [event.to_dict() for event in self._charge_events if event.source == "solar"]

    def get_summary(self) -> Dict[str, Any]:
        """Get complete tracking summary Returns: Dictionary with all current values @zara"""
        return {
            "grid_charge_today_kwh": self.get_grid_charge_today(),
            "solar_charge_today_kwh": self.get_solar_charge_today(),
            "discharge_today_kwh": self.get_discharge_today(),
            "total_charge_today_kwh": round(
                self._grid_charge_today_kwh + self._solar_charge_today_kwh, 3
            ),
            "grid_charge_events_count": len([e for e in self._charge_events if e.source == "grid"]),
            "solar_charge_events_count": len(
                [e for e in self._charge_events if e.source == "solar"]
            ),
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "last_status": self._last_status,
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete state for persistence Returns: Dictionary with all state data @zara"""
        return {
            "grid_charge_today_kwh": self._grid_charge_today_kwh,
            "solar_charge_today_kwh": self._solar_charge_today_kwh,
            "discharge_today_kwh": self._discharge_today_kwh,
            "charge_events": [event.to_dict() for event in self._charge_events],
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "last_status": self._last_status,
            "last_reset_date": self._last_reset_date,
        }

    def restore_state(self, state: Dict[str, Any]):
        """Restore state from persistence Args: state: State dictionary from get_state_dict() @zara"""
        try:
            self._grid_charge_today_kwh = state.get("grid_charge_today_kwh", 0.0)
            self._solar_charge_today_kwh = state.get("solar_charge_today_kwh", 0.0)
            self._discharge_today_kwh = state.get("discharge_today_kwh", 0.0)
            self._last_status = state.get("last_status", "idle")
            self._last_reset_date = state.get("last_reset_date")

            if state.get("last_update"):
                self._last_update = datetime.fromisoformat(state["last_update"])

            self._charge_events = []
            for event_data in state.get("charge_events", []):
                event = ChargeEvent(
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    source=event_data["source"],
                    power=event_data["power"],
                    duration_minutes=event_data["duration_minutes"],
                )
                self._charge_events.append(event)

            _LOGGER.info(
                f"State restored - Grid: {self._grid_charge_today_kwh:.2f} kWh, "
                f"Solar: {self._solar_charge_today_kwh:.2f} kWh"
            )

        except Exception as e:
            _LOGGER.error(f"Error restoring charge tracker state: {e}")

    def _calculate_energy_flow_v10(
        self,
        battery_power: float,
        solar_power: float,
        inverter_output: float,
        house_consumption: float,
        grid_import: float,
        grid_export: float,
        grid_charge_power: float,
        duration_minutes: float,
    ) -> Dict[str, Any]:
        """Calculate complete energy flow (v10.0.0 - NEW Architecture)

        All values in Watts, converted to kWh by integration over time.

        NEW Architecture with separate sensors:
        - solar_power: DC production from solar panels (≥0)
        - inverter_output: AC output to house (Solar + Battery combined, ≥0)
        - house_consumption: Total house consumption (≥0)
        - grid_import: Power from grid (≥0)
        - grid_export: Power to grid (≥0)
        - battery_power: Battery charge/discharge (+charge/-discharge)

        Energy flow logic:
        1. Battery → House (if discharging)
        2. Solar → Battery (if charging)
        3. Solar → Grid (export)
        4. Solar → House (direct, remainder)
        5. Grid → House (if importing)
        6. Grid → Battery (grid charge)
        """

        duration_hours = duration_minutes / 60.0

        flows = {
            "solar_to_house_w": 0.0,
            "solar_to_battery_w": 0.0,
            "solar_to_grid_w": 0.0,
            "grid_to_house_w": 0.0,
            "grid_to_battery_w": 0.0,
            "battery_to_house_w": 0.0,
            "grid_import_w": grid_import,
            "grid_export_w": grid_export,
            "inverter_output_w": inverter_output,
            "status": "idle",
        }

        battery_charge_w = max(0.0, battery_power)
        battery_discharge_w = abs(min(0.0, battery_power))
        grid_to_battery = min(grid_charge_power, battery_charge_w)
        solar_to_battery = max(0.0, battery_charge_w - grid_to_battery)

        if battery_discharge_w > POWER_TOLERANCE:

            flows["battery_to_house_w"] = battery_discharge_w
            flows["status"] = "discharging"
        elif battery_charge_w > POWER_TOLERANCE:
            flows["grid_to_battery_w"] = grid_to_battery
            flows["solar_to_battery_w"] = solar_to_battery
            flows["status"] = (
                "charging_grid" if grid_to_battery > solar_to_battery else "charging_solar"
            )

        battery_to_house = min(flows["battery_to_house_w"], house_consumption)
        remaining_house = max(0.0, house_consumption - battery_to_house)
        flows["battery_to_house_w"] = battery_to_house

        remaining_solar = max(0.0, solar_power - flows["solar_to_battery_w"])
        flows["solar_to_grid_w"] = min(remaining_solar, grid_export)
        remaining_solar -= flows["solar_to_grid_w"]
        flows["solar_to_house_w"] = min(remaining_house, remaining_solar)
        remaining_house -= flows["solar_to_house_w"]

        flows["grid_to_house_w"] = max(0.0, remaining_house)
        grid_used = flows["grid_to_house_w"] + flows["grid_to_battery_w"]
        if grid_import > POWER_TOLERANCE:
            mismatch = grid_import - grid_used

            if mismatch < -POWER_TOLERANCE:
                reduction = min(flows["grid_to_house_w"], abs(mismatch))
                flows["grid_to_house_w"] -= reduction
                grid_used = flows["grid_to_house_w"] + flows["grid_to_battery_w"]
            elif mismatch > POWER_TOLERANCE:
                flows["grid_to_house_w"] += mismatch
                grid_used = flows["grid_to_house_w"] + flows["grid_to_battery_w"]

        expected_inverter_out = (
            flows["solar_to_house_w"] + flows["solar_to_grid_w"] + flows["battery_to_house_w"]
        )
        inverter_delta = inverter_output - expected_inverter_out
        flows["inverter_delta"] = inverter_delta
        if abs(inverter_delta) > POWER_TOLERANCE * 4:
            _LOGGER.debug(
                "Inverter output mismatch (v10 flow): expected %.1fW vs measured %.1fW (Δ=%.1fW)",
                expected_inverter_out,
                inverter_output,
                inverter_delta,
            )

        energy_flows = {}
        for key, power_w in flows.items():
            if key.endswith("_w"):
                energy_key = key.replace("_w", "_kwh")
                energy_flows[energy_key] = (power_w / 1000.0) * duration_hours
            else:
                energy_flows[key] = power_w

        return energy_flows

    def _update_energy_flows_v10(self, energy_flows: Dict[str, Any]):
        """Update accumulated energy flows (v10.0.0) @zara"""
        self._solar_to_house_kwh += energy_flows.get("solar_to_house_kwh", 0.0)
        self._solar_to_battery_kwh += energy_flows.get("solar_to_battery_kwh", 0.0)
        self._solar_to_grid_kwh += energy_flows.get("solar_to_grid_kwh", 0.0)
        self._grid_to_house_kwh += energy_flows.get("grid_to_house_kwh", 0.0)
        self._grid_to_battery_kwh += energy_flows.get("grid_to_battery_kwh", 0.0)
        self._battery_to_house_kwh += energy_flows.get("battery_to_house_kwh", 0.0)
        self._grid_import_today_kwh += energy_flows.get("grid_import_kwh", 0.0)
        self._grid_export_today_kwh += energy_flows.get("grid_export_kwh", 0.0)

        self._solar_charge_today_kwh += energy_flows.get("solar_to_battery_kwh", 0.0)
        self._grid_charge_today_kwh += energy_flows.get("grid_to_battery_kwh", 0.0)
        self._discharge_today_kwh += energy_flows.get("battery_to_house_kwh", 0.0)

    def get_energy_flows_v10(self) -> Dict[str, float]:
        """Get all energy flows (v10.0.0) @zara"""
        return {
            "solar_to_house_kwh": round(self._solar_to_house_kwh, 3),
            "solar_to_battery_kwh": round(self._solar_to_battery_kwh, 3),
            "solar_to_grid_kwh": round(self._solar_to_grid_kwh, 3),
            "grid_to_house_kwh": round(self._grid_to_house_kwh, 3),
            "grid_to_battery_kwh": round(self._grid_to_battery_kwh, 3),
            "battery_to_house_kwh": round(self._battery_to_house_kwh, 3),
            "grid_import_today_kwh": round(self._grid_import_today_kwh, 3),
            "grid_export_today_kwh": round(self._grid_export_today_kwh, 3),
        }

    def _update_energy_flows_v9(self, energy_flows: Dict[str, Any]):
        return self._update_energy_flows_v10(energy_flows)

    def get_energy_flows_v9(self) -> Dict[str, float]:
        return self.get_energy_flows_v10()
