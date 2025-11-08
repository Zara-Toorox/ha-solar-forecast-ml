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
    """Tracks battery charging sources using real-time power flow analysis Logic: - Positive battery_power = Charging - Negative battery_power = Discharging Charging source detection: - If solar_power >= (output_power + battery_power) → Solar charging - Otherwise → Grid charging"""

    def __init__(self, hass: HomeAssistant, battery_capacity: float = 10.0):
        """Initialize charge tracker Args: hass: Home Assistant instance battery_capacity: Battery capacity in kWh"""
        self.hass = hass
        self.battery_capacity = battery_capacity

        # Daily tracking
        self._grid_charge_today_kwh = 0.0
        self._solar_charge_today_kwh = 0.0
        self._discharge_today_kwh = 0.0

        # Event history (for hourly cost calculation)
        self._charge_events: List[ChargeEvent] = []

        # State tracking
        self._last_update: Optional[datetime] = None
        self._last_status: str = 'idle'

        # Daily reset tracking
        self._last_reset_date: Optional[str] = None

        _LOGGER.info(
            f"BatteryChargeTracker initialized - Capacity: {battery_capacity} kWh"
        )

    def update(
        self,
        battery_power: float,
        solar_power: float,
        output_power: float,
    ) -> Dict[str, Any]:
        """Update tracker with current power values Args: battery_power: Battery charge/discharge power in W (+charge/-discharge) solar_power: Solar production in W output_power: Output to house/grid in W Returns: Dictionary with current status and accumulated values"""
        now = datetime.now()  # Local time

        # Reset daily values at midnight
        self._check_daily_reset(now)

        # Calculate time delta since last update
        duration_minutes = 1.0  # Default 1 minute
        if self._last_update:
            delta = (now - self._last_update).total_seconds() / 60
            duration_minutes = max(0.1, min(delta, 60))  # Clamp between 0.1 and 60 min

        # Analyze current power flow
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

            _LOGGER.debug(
                f"Grid charging: {battery_power}W for {duration_minutes:.1f}min = "
                f"{energy_kwh:.3f} kWh"
            )

        elif current_status == 'discharging':
            energy_kwh = (abs(battery_power) / 1000) * (duration_minutes / 60)
            self._discharge_today_kwh += energy_kwh

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
                    f"Daily reset (local time) - Grid: {self._grid_charge_today_kwh:.2f} kWh, "
                    f"Solar: {self._solar_charge_today_kwh:.2f} kWh, "
                    f"Discharge: {self._discharge_today_kwh:.2f} kWh"
                )

            self._grid_charge_today_kwh = 0.0
            self._solar_charge_today_kwh = 0.0
            self._discharge_today_kwh = 0.0
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
