# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from .data_io import DataManagerIO

_LOGGER = logging.getLogger(__name__)


class PanelGroupSensorReader(DataManagerIO):
    """Reads energy sensors for panel groups to enable per-group learning. @zara"""

    def __init__(
        self,
        hass: HomeAssistant,
        data_dir: Path,
        panel_groups: List[Dict[str, Any]],
    ):
        """Initialize the panel group sensor reader.

        Args:
            hass: Home Assistant instance
            data_dir: Data directory for state persistence
            panel_groups: List of panel group configurations with optional energy_sensor

        @zara
        """
        super().__init__(hass, data_dir)
        self.panel_groups = panel_groups
        self._last_values: Dict[str, float] = {}  # group_name -> last kWh value
        self._state_file = data_dir / "stats" / "panel_group_sensor_state.json"

    async def initialize(self) -> None:
        """Load last known sensor values from state file. @zara"""
        try:
            state = await self._read_json_file(self._state_file, {})
            self._last_values = state.get("last_values", {})
            _LOGGER.debug(
                "Loaded panel group sensor state: %d groups",
                len(self._last_values)
            )
        except Exception as e:
            _LOGGER.warning("Failed to load panel group sensor state: %s", e)
            self._last_values = {}

    async def _save_state(self) -> None:
        """Save current sensor values to state file. @zara"""
        try:
            state = {
                "last_updated": datetime.now().isoformat(),
                "last_values": self._last_values,
            }
            await self._atomic_write_json(self._state_file, state)
        except Exception as e:
            _LOGGER.warning("Failed to save panel group sensor state: %s", e)

    def get_groups_with_sensors(self) -> List[Dict[str, Any]]:
        """Get list of panel groups that have energy sensors configured. @zara"""
        return [
            g for g in self.panel_groups
            if g.get("energy_sensor") and len(g.get("energy_sensor", "")) > 0
        ]

    def has_any_sensor(self) -> bool:
        """Check if any panel group has an energy sensor configured. @zara"""
        return len(self.get_groups_with_sensors()) > 0

    async def read_current_energy(self, group_name: str) -> Optional[float]:
        """Read current kWh value for a specific group.

        Args:
            group_name: Name of the panel group

        Returns:
            Current kWh value or None if not available

        @zara
        """
        group = next(
            (g for g in self.panel_groups if g.get("name") == group_name),
            None
        )

        if not group:
            _LOGGER.debug("Panel group '%s' not found", group_name)
            return None

        entity_id = group.get("energy_sensor")
        if not entity_id:
            return None

        try:
            state = self.hass.states.get(entity_id)

            if state is None:
                _LOGGER.warning("Energy sensor '%s' not found for group '%s'", entity_id, group_name)
                return None

            if state.state in ["unavailable", "unknown", None]:
                _LOGGER.debug("Energy sensor '%s' is %s", entity_id, state.state)
                return None

            value = float(state.state)

            # Convert Wh to kWh if needed
            unit = state.attributes.get("unit_of_measurement", "")
            if unit.lower() == "wh":
                value = value / 1000.0

            return round(value, 4)

        except (ValueError, TypeError) as e:
            _LOGGER.warning(
                "Failed to read energy sensor '%s' for group '%s': %s",
                entity_id, group_name, e
            )
            return None

    async def get_hourly_production(
        self,
        group_name: str,
    ) -> Optional[float]:
        """Calculate production since last read (hourly delta).

        This calculates the difference between current sensor value and
        the last stored value to get hourly production.

        Args:
            group_name: Name of the panel group

        Returns:
            Production in kWh since last read, or None if not available

        @zara
        """
        current_value = await self.read_current_energy(group_name)

        if current_value is None:
            return None

        last_value = self._last_values.get(group_name)

        if last_value is None:
            # First reading - store and return None
            self._last_values[group_name] = current_value
            await self._save_state()
            _LOGGER.debug(
                "First reading for group '%s': %.4f kWh (no delta yet)",
                group_name, current_value
            )
            return None

        # Handle counter reset (e.g., midnight reset for daily sensors)
        if current_value < last_value:
            _LOGGER.info(
                "Energy counter reset detected for group '%s': %.4f -> %.4f kWh",
                group_name, last_value, current_value
            )
            # Assume current_value is the production since reset
            delta = current_value
        else:
            delta = current_value - last_value

        # Update stored value
        self._last_values[group_name] = current_value
        await self._save_state()

        # Sanity check: delta should be reasonable (< 10 kWh per hour is plausible)
        if delta > 10.0:
            _LOGGER.warning(
                "Unusually high hourly production for group '%s': %.4f kWh",
                group_name, delta
            )

        return round(delta, 4)

    async def read_all_groups(self) -> Dict[str, float]:
        """Read current energy values for all groups with sensors.

        Returns:
            Dict mapping group_name to current kWh value

        @zara
        """
        results: Dict[str, float] = {}

        for group in self.get_groups_with_sensors():
            group_name = group.get("name", "Unknown")
            value = await self.read_current_energy(group_name)

            if value is not None:
                results[group_name] = value

        return results

    async def get_all_hourly_productions(self) -> Dict[str, float]:
        """Get hourly production for all groups with sensors.

        Returns:
            Dict mapping group_name to hourly production in kWh

        @zara
        """
        results: Dict[str, float] = {}

        for group in self.get_groups_with_sensors():
            group_name = group.get("name", "Unknown")
            production = await self.get_hourly_production(group_name)

            if production is not None:
                results[group_name] = production

        return results

    async def validate_sensors(self) -> Dict[str, Dict[str, Any]]:
        """Validate all configured energy sensors.

        Returns:
            Dict with validation results per group:
            {
                "Gruppe 1": {"valid": True, "entity_id": "sensor.x", "unit": "kWh"},
                "Gruppe 2": {"valid": False, "error": "Entity not found"}
            }

        @zara
        """
        results: Dict[str, Dict[str, Any]] = {}

        for group in self.get_groups_with_sensors():
            group_name = group.get("name", "Unknown")
            entity_id = group.get("energy_sensor", "")

            validation = await self._validate_energy_sensor(entity_id)
            validation["entity_id"] = entity_id
            results[group_name] = validation

        return results

    async def _validate_energy_sensor(self, entity_id: str) -> Dict[str, Any]:
        """Validate a single energy sensor.

        Checks:
        1. Entity exists
        2. Entity is numeric
        3. Unit is kWh or Wh

        @zara
        """
        if not entity_id:
            return {"valid": False, "error": "No entity_id configured"}

        state = self.hass.states.get(entity_id)

        if state is None:
            # Try to find similar entities to help with debugging
            suggestions = self._find_similar_entities(entity_id)
            error_msg = f"Entity {entity_id} not found"
            if suggestions:
                error_msg += f". Did you mean: {', '.join(suggestions[:3])}"
            return {"valid": False, "error": error_msg}

        if state.state in ["unavailable", "unknown"]:
            return {"valid": False, "error": f"Entity {entity_id} is {state.state}"}

        try:
            float(state.state)
        except (ValueError, TypeError):
            return {"valid": False, "error": f"Entity {entity_id} is not numeric: {state.state}"}

        unit = state.attributes.get("unit_of_measurement", "")
        if unit.lower() not in ["kwh", "wh"]:
            return {
                "valid": False,
                "error": f"Entity {entity_id} has wrong unit: {unit} (expected kWh or Wh)"
            }

        return {
            "valid": True,
            "unit": unit,
            "current_value": float(state.state),
            "state_class": state.attributes.get("state_class", "unknown"),
        }

    def _find_similar_entities(self, entity_id: str) -> List[str]:
        """Find similar entity IDs to help with debugging.

        Searches for entities containing parts of the expected entity_id name.

        @zara
        """
        try:
            # Extract the sensor name without domain
            if "." in entity_id:
                _, name_part = entity_id.split(".", 1)
            else:
                name_part = entity_id

            # Extract keywords from entity name (split by underscore)
            keywords = [kw.lower() for kw in name_part.split("_") if len(kw) >= 2]

            # Get all sensor entities
            all_states = self.hass.states.async_all("sensor")

            candidates = []
            for state in all_states:
                eid = state.entity_id.lower()
                # Check if any keyword matches
                matches = sum(1 for kw in keywords if kw in eid)
                if matches >= 2:  # At least 2 keywords match
                    # Prefer energy sensors
                    unit = state.attributes.get("unit_of_measurement", "")
                    if unit and unit.lower() in ["kwh", "wh"]:
                        candidates.append((state.entity_id, matches + 1))
                    else:
                        candidates.append((state.entity_id, matches))

            # Sort by match count (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [c[0] for c in candidates[:5]]

        except Exception:
            return []

    async def check_consistency(
        self,
        total_actual: float,
        group_actuals: Dict[str, float],
        tolerance_percent: float = 15.0
    ) -> Dict[str, Any]:
        """Check if sum of group actuals matches total actual.

        Warns if the deviation exceeds the tolerance threshold.

        Args:
            total_actual: Total actual production (kWh)
            group_actuals: Dict of group_name -> actual_kwh
            tolerance_percent: Acceptable deviation percentage

        Returns:
            Dict with consistency check results

        @zara
        """
        if not group_actuals or total_actual <= 0:
            return {
                "consistent": True,
                "reason": "No data to compare"
            }

        group_sum = sum(group_actuals.values())
        deviation = abs(group_sum - total_actual) / total_actual * 100

        consistent = deviation <= tolerance_percent

        if not consistent:
            _LOGGER.warning(
                "Panel group sum (%.3f kWh) deviates %.1f%% from total (%.3f kWh)",
                group_sum, deviation, total_actual
            )

        return {
            "consistent": consistent,
            "group_sum": round(group_sum, 4),
            "total_actual": round(total_actual, 4),
            "deviation_percent": round(deviation, 1),
            "tolerance_percent": tolerance_percent,
        }

    def reset_last_values(self) -> None:
        """Reset all stored last values (e.g., at midnight). @zara"""
        self._last_values = {}
        _LOGGER.info("Panel group sensor last values reset")
