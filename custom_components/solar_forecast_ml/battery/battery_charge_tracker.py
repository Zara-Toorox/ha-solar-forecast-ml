"""
Battery Charge Tracker - Real-time Grid vs Solar Charging Detection

Tracks battery charging sources in real-time using power flow analysis
Completely independent from Solar/ML components

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# NOTE: This tracker is simplified - persistence is handled by BatteryPersistence
# This class only provides real-time tracking and summary for current day

# Power thresholds (Watts)
POWER_TOLERANCE = 50  # Ignore changes below 50W to avoid noise
MIN_TRACKING_POWER = 100  # Minimum power to start tracking (avoid phantom loads)


class ChargeEvent:
    """Represents a single charging event"""

    def __init__(
        self,
        timestamp: datetime,
        source: str,  # 'grid' or 'solar'
        power: float,  # Watts
        duration_minutes: float = 1.0,
    ):
        self.timestamp = timestamp  # Local time
        self.source = source
        self.power = power
        self.duration_minutes = duration_minutes

    @property
    def energy_kwh(self) -> float:
        """Calculate energy in kWh for this event"""
        # Energy (kWh) = Power (kW) × Time (hours)
        return (self.power / 1000) * (self.duration_minutes / 60)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'hour': self.timestamp.hour,
            'source': self.source,
            'power': self.power,
            'duration_minutes': self.duration_minutes,
            'energy_kwh': round(self.energy_kwh, 4),
        }


class BatteryChargeTracker:
    """Tracks battery charging sources using real-time power flow analysis

    V9.0.0: Complete energy flow tracking
    - Solar → House/Battery/Grid
    - Grid → House/Battery
    - Battery → House
    - All values in Watt (W), integrated over time to kWh
    """

    def __init__(self, hass: HomeAssistant, battery_capacity: float = 10.0):
        """Initialize charge tracker

        Args:
            hass: Home Assistant instance
            battery_capacity: Battery capacity in kWh
        """
        self.hass = hass
        self.battery_capacity = battery_capacity

        # Daily tracking - Battery flows
        self._grid_charge_today_kwh = 0.0
        self._solar_charge_today_kwh = 0.0
        self._discharge_today_kwh = 0.0

        # NEW v9.0.0: Energy flow tracking
        self._solar_to_house_kwh = 0.0
        self._solar_to_battery_kwh = 0.0
        self._solar_to_grid_kwh = 0.0
        self._grid_to_house_kwh = 0.0
        self._grid_to_battery_kwh = 0.0
        self._battery_to_house_kwh = 0.0
        self._grid_import_today_kwh = 0.0
        self._grid_export_today_kwh = 0.0

        # Event history (for hourly cost calculation)
        self._charge_events: List[ChargeEvent] = []

        # State tracking
        self._last_update: Optional[datetime] = None
        self._last_status: str = 'idle'

        # Daily reset tracking
        self._last_reset_date: Optional[str] = None

        _LOGGER.info(
            f"BatteryChargeTracker initialized (v9.0.0) - Capacity: {battery_capacity} kWh"
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
        output_power: float = 0.0,  # LEGACY parameter for backwards compatibility
    ) -> Dict[str, Any]:
        """Update tracker with current power values (v9.0.0)

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
        now = datetime.now()  # Local time

        # Reset daily values at midnight
        self._check_daily_reset(now)

        # Calculate time delta since last update
        duration_minutes = 1.0  # Default 1 minute
        if self._last_update:
            delta = (now - self._last_update).total_seconds() / 60
            duration_minutes = max(0.1, min(delta, 60))  # Clamp between 0.1 and 60 min

        # NEW v9.0.0: Calculate complete energy flow
        if house_consumption > 0:  # New config detected
            energy_flows = self._calculate_energy_flow_v9(
                battery_power=battery_power,
                solar_power=solar_power,
                inverter_output=inverter_output,
                house_consumption=house_consumption,
                grid_import=grid_import,
                grid_export=grid_export,
                grid_charge_power=grid_charge_power,
                duration_minutes=duration_minutes,
            )
            self._update_energy_flows_v9(energy_flows)
            current_status = energy_flows['status']
        else:
            # LEGACY: Old v8.x flow analysis
            status_info = self._analyze_power_flow(
                battery_power=battery_power,
                solar_power=solar_power,
                output_power=output_power,
            )
            current_status = status_info['status']

        # Track charging/discharging energy
        if current_status == 'charging_solar':
            energy_kwh = (battery_power / 1000) * (duration_minutes / 60)
            self._solar_charge_today_kwh += energy_kwh

            # Record event
            event = ChargeEvent(
                timestamp=now,
                source='solar',
                power=battery_power,
                duration_minutes=duration_minutes,
            )
            self._charge_events.append(event)

            # Only log significant energy transfers (> 10 Wh)
            if energy_kwh > 0.01:
                _LOGGER.debug(
                    f"Solar charging: {battery_power}W for {duration_minutes:.1f}min = "
                    f"{energy_kwh:.3f} kWh"
                )

        elif current_status == 'charging_grid':
            energy_kwh = (battery_power / 1000) * (duration_minutes / 60)
            self._grid_charge_today_kwh += energy_kwh

            # Record event
            event = ChargeEvent(
                timestamp=now,
                source='grid',
                power=battery_power,
                duration_minutes=duration_minutes,
            )
            self._charge_events.append(event)

            # Only log significant energy transfers (> 10 Wh)
            if energy_kwh > 0.01:
                _LOGGER.debug(
                    f"Grid charging: {battery_power}W for {duration_minutes:.1f}min = "
                    f"{energy_kwh:.3f} kWh"
                )

        elif current_status == 'discharging':
            energy_kwh = (abs(battery_power) / 1000) * (duration_minutes / 60)
            self._discharge_today_kwh += energy_kwh

            # Only log significant energy transfers (> 10 Wh)
            if energy_kwh > 0.01:
                _LOGGER.debug(
                    f"Discharging: {abs(battery_power)}W for {duration_minutes:.1f}min = "
                    f"{energy_kwh:.3f} kWh"
                )

        # Update state
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
        # Discharging
        if battery_power < -POWER_TOLERANCE:
            return {
                'status': 'discharging',
                'discharge_power': abs(battery_power),
                'grid_charge_power': 0,
                'solar_charge_power': 0,
            }

        # Charging
        elif battery_power > POWER_TOLERANCE:
            # Calculate total power demand (output + battery charging)
            total_demand = output_power + battery_power

            # If solar covers total demand → Solar charging
            # Allow some tolerance for inverter losses and measurement inaccuracy
            tolerance = max(MIN_TRACKING_POWER, total_demand * 0.1)  # 10% tolerance

            if solar_power >= (total_demand - tolerance):
                return {
                    'status': 'charging_solar',
                    'solar_charge_power': battery_power,
                    'grid_charge_power': 0,
                    'discharge_power': 0,
                    'solar_available': solar_power,
                    'demand': total_demand,
                }
            else:
                return {
                    'status': 'charging_grid',
                    'grid_charge_power': battery_power,
                    'solar_charge_power': 0,
                    'discharge_power': 0,
                    'solar_available': solar_power,
                    'demand': total_demand,
                }

        # Idle (between -POWER_TOLERANCE and +POWER_TOLERANCE)
        else:
            return {
                'status': 'idle',
                'discharge_power': 0,
                'grid_charge_power': 0,
                'solar_charge_power': 0,
            }

    def _check_daily_reset(self, now: datetime):
        """Reset daily counters at midnight (local time)"""
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

            # Reset legacy counters
            self._grid_charge_today_kwh = 0.0
            self._solar_charge_today_kwh = 0.0
            self._discharge_today_kwh = 0.0

            # Reset v9.0.0 energy flows
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
        """Get total grid charge today in kWh"""
        return round(self._grid_charge_today_kwh, 3)

    def get_solar_charge_today(self) -> float:
        """Get total solar charge today in kWh"""
        return round(self._solar_charge_today_kwh, 3)

    def get_discharge_today(self) -> float:
        """Get total discharge today in kWh"""
        return round(self._discharge_today_kwh, 3)

    def get_grid_charge_events(self) -> List[Dict[str, Any]]:
        """Get all grid charging events today Returns: List of charge event dictionaries with hour and energy"""
        return [
            event.to_dict()
            for event in self._charge_events
            if event.source == 'grid'
        ]

    def get_solar_charge_events(self) -> List[Dict[str, Any]]:
        """Get all solar charging events today Returns: List of charge event dictionaries with hour and energy"""
        return [
            event.to_dict()
            for event in self._charge_events
            if event.source == 'solar'
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get complete tracking summary Returns: Dictionary with all current values"""
        return {
            'grid_charge_today_kwh': self.get_grid_charge_today(),
            'solar_charge_today_kwh': self.get_solar_charge_today(),
            'discharge_today_kwh': self.get_discharge_today(),
            'total_charge_today_kwh': round(
                self._grid_charge_today_kwh + self._solar_charge_today_kwh, 3
            ),
            'grid_charge_events_count': len([e for e in self._charge_events if e.source == 'grid']),
            'solar_charge_events_count': len([e for e in self._charge_events if e.source == 'solar']),
            'last_update': self._last_update.isoformat() if self._last_update else None,
            'last_status': self._last_status,
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete state for persistence Returns: Dictionary with all state data"""
        return {
            'grid_charge_today_kwh': self._grid_charge_today_kwh,
            'solar_charge_today_kwh': self._solar_charge_today_kwh,
            'discharge_today_kwh': self._discharge_today_kwh,
            'charge_events': [event.to_dict() for event in self._charge_events],
            'last_update': self._last_update.isoformat() if self._last_update else None,
            'last_status': self._last_status,
            'last_reset_date': self._last_reset_date,
        }

    def restore_state(self, state: Dict[str, Any]):
        """Restore state from persistence Args: state: State dictionary from get_state_dict()"""
        try:
            self._grid_charge_today_kwh = state.get('grid_charge_today_kwh', 0.0)
            self._solar_charge_today_kwh = state.get('solar_charge_today_kwh', 0.0)
            self._discharge_today_kwh = state.get('discharge_today_kwh', 0.0)
            self._last_status = state.get('last_status', 'idle')
            self._last_reset_date = state.get('last_reset_date')

            if state.get('last_update'):
                self._last_update = datetime.fromisoformat(state['last_update'])

            # Restore events
            self._charge_events = []
            for event_data in state.get('charge_events', []):
                event = ChargeEvent(
                    timestamp=datetime.fromisoformat(event_data['timestamp']),
                    source=event_data['source'],
                    power=event_data['power'],
                    duration_minutes=event_data['duration_minutes'],
                )
                self._charge_events.append(event)

            _LOGGER.info(
                f"State restored - Grid: {self._grid_charge_today_kwh:.2f} kWh, "
                f"Solar: {self._solar_charge_today_kwh:.2f} kWh"
            )

        except Exception as e:
            _LOGGER.error(f"Error restoring charge tracker state: {e}")

    # ========================================================================
    # NEW v9.0.0: Complete Energy Flow Calculation
    # ========================================================================

    def _calculate_energy_flow_v9(
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
        """Calculate complete energy flow (v9.0.0 - NEW Architecture)

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
        # Convert duration to hours
        duration_hours = duration_minutes / 60.0

        # Initialize flows
        flows = {
            'solar_to_house_w': 0.0,
            'solar_to_battery_w': 0.0,
            'solar_to_grid_w': 0.0,
            'grid_to_house_w': 0.0,
            'grid_to_battery_w': 0.0,
            'battery_to_house_w': 0.0,
            'grid_import_w': grid_import,
            'grid_export_w': grid_export,
            'status': 'idle',
        }

        # 1. Battery flows
        battery_charge_w = max(0.0, battery_power)
        battery_discharge_w = abs(min(0.0, battery_power))

        if battery_discharge_w > POWER_TOLERANCE:
            # Battery is discharging → goes to house
            flows['battery_to_house_w'] = battery_discharge_w
            flows['status'] = 'discharging'

        elif battery_charge_w > POWER_TOLERANCE:
            # Battery is charging → determine source
            if grid_charge_power > POWER_TOLERANCE:
                # Grid charging (known from sensor)
                flows['grid_to_battery_w'] = grid_charge_power
                flows['solar_to_battery_w'] = max(0.0, battery_charge_w - grid_charge_power)
                flows['status'] = 'charging_grid' if grid_charge_power > flows['solar_to_battery_w'] else 'charging_solar'
            else:
                # Assume solar charging
                flows['solar_to_battery_w'] = battery_charge_w
                flows['status'] = 'charging_solar'

        # 2. Solar flows
        if solar_power > POWER_TOLERANCE:
            # Solar → Grid (export)
            flows['solar_to_grid_w'] = grid_export

            # Solar → Battery (already calculated above)
            solar_to_battery = flows['solar_to_battery_w']

            # Solar → House (remainder)
            flows['solar_to_house_w'] = max(0.0, solar_power - solar_to_battery - flows['solar_to_grid_w'])

        # 3. Grid flows
        if grid_import > POWER_TOLERANCE:
            # Importing from grid → House and/or Battery
            grid_to_battery = flows['grid_to_battery_w']
            flows['grid_to_house_w'] = max(0.0, grid_import - grid_to_battery)

        # Convert flows to energy (kWh)
        energy_flows = {}
        for key, power_w in flows.items():
            if key.endswith('_w'):
                energy_key = key.replace('_w', '_kwh')
                energy_flows[energy_key] = (power_w / 1000.0) * duration_hours
            else:
                energy_flows[key] = power_w

        return energy_flows

    def _update_energy_flows_v9(self, energy_flows: Dict[str, Any]):
        """Update accumulated energy flows (v9.0.0)"""
        self._solar_to_house_kwh += energy_flows.get('solar_to_house_kwh', 0.0)
        self._solar_to_battery_kwh += energy_flows.get('solar_to_battery_kwh', 0.0)
        self._solar_to_grid_kwh += energy_flows.get('solar_to_grid_kwh', 0.0)
        self._grid_to_house_kwh += energy_flows.get('grid_to_house_kwh', 0.0)
        self._grid_to_battery_kwh += energy_flows.get('grid_to_battery_kwh', 0.0)
        self._battery_to_house_kwh += energy_flows.get('battery_to_house_kwh', 0.0)
        self._grid_import_today_kwh += energy_flows.get('grid_import_kwh', 0.0)
        self._grid_export_today_kwh += energy_flows.get('grid_export_kwh', 0.0)

        # Also update legacy trackers
        self._solar_charge_today_kwh += energy_flows.get('solar_to_battery_kwh', 0.0)
        self._grid_charge_today_kwh += energy_flows.get('grid_to_battery_kwh', 0.0)
        self._discharge_today_kwh += energy_flows.get('battery_to_house_kwh', 0.0)

    def get_energy_flows_v9(self) -> Dict[str, float]:
        """Get all energy flows (v9.0.0)

        Returns:
            Dictionary with all energy flows in kWh
        """
        return {
            'solar_to_house_kwh': round(self._solar_to_house_kwh, 3),
            'solar_to_battery_kwh': round(self._solar_to_battery_kwh, 3),
            'solar_to_grid_kwh': round(self._solar_to_grid_kwh, 3),
            'grid_to_house_kwh': round(self._grid_to_house_kwh, 3),
            'grid_to_battery_kwh': round(self._grid_to_battery_kwh, 3),
            'battery_to_house_kwh': round(self._battery_to_house_kwh, 3),
            'grid_import_today_kwh': round(self._grid_import_today_kwh, 3),
            'grid_export_today_kwh': round(self._grid_export_today_kwh, 3),
        }
