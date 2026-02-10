# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Panel Group Sensor Reader for Solar Forecast ML V16.2.0.
Reads energy sensors for panel groups to enable per-group learning.
Uses database operations via panel_group_sensor_state table.

@zara
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from .db_manager import DatabaseManager
from .data_io import DataManagerIO

_LOGGER = logging.getLogger(__name__)


class PanelGroupSensorReader(DataManagerIO):
    """Reads energy sensors for panel groups to enable per-group learning. @zara

    V16.0.0: All state persistence via panel_group_sensor_state database table.
    Replaces JSON file operations with database queries.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        db_manager: DatabaseManager,
        panel_groups: List[Dict[str, Any]],
    ):
        """Initialize the panel group sensor reader. @zara

        Args:
            hass: Home Assistant instance
            db_manager: DatabaseManager instance for DB operations
            panel_groups: List of panel group configurations with optional energy_sensor
        """
        super().__init__(hass, db_manager)
        self.panel_groups = panel_groups
        self._last_values: Dict[str, float] = {}  # group_name -> last kWh value

        _LOGGER.debug(
            "PanelGroupSensorReader initialized with %d groups",
            len(panel_groups),
        )

    async def initialize(self) -> None:
        """Load last known sensor values from database. @zara"""
        try:
            await self.ensure_initialized()

            # Load from panel_group_sensor_state table
            rows = await self.fetch_all(
                "SELECT group_name, last_value FROM panel_group_sensor_state"
            )

            self._last_values = {row[0]: row[1] for row in rows if row[1] is not None}

            _LOGGER.debug(
                "Loaded panel group sensor state: %d groups",
                len(self._last_values),
            )
        except Exception as e:
            _LOGGER.warning("Failed to load panel group sensor state: %s", e)
            self._last_values = {}

    async def _save_state(self, group_name: str, value: float) -> None:
        """Save sensor value for a group to database. @zara

        Args:
            group_name: Name of the panel group
            value: Current kWh value
        """
        try:
            await self.execute_query(
                """INSERT INTO panel_group_sensor_state (group_name, last_value, last_updated)
                   VALUES (?, ?, ?)
                   ON CONFLICT(group_name) DO UPDATE SET
                       last_value = excluded.last_value,
                       last_updated = excluded.last_updated""",
                (group_name, value, datetime.now()),
            )
        except Exception as e:
            _LOGGER.warning("Failed to save panel group sensor state: %s", e)

    async def _save_all_states(self) -> None:
        """Save all current sensor values to database. @zara"""
        try:
            state_data = {
                "last_updated": datetime.now().isoformat(),
                "last_values": self._last_values,
            }
            await self.db.save_panel_group_sensor_state(state_data)
        except Exception as e:
            _LOGGER.warning("Failed to save panel group sensor states: %s", e)

    def get_groups_with_sensors(self) -> List[Dict[str, Any]]:
        """Get list of panel groups that have energy sensors configured. @zara

        Returns:
            List of panel group configurations with energy_sensor defined
        """
        return [
            g
            for g in self.panel_groups
            if g.get("energy_sensor") and len(g.get("energy_sensor", "")) > 0
        ]

    def has_any_sensor(self) -> bool:
        """Check if any panel group has an energy sensor configured. @zara

        Returns:
            True if at least one group has a sensor
        """
        return len(self.get_groups_with_sensors()) > 0

    async def read_current_energy(self, group_name: str) -> Optional[float]:
        """Read current kWh value for a specific group. @zara

        Args:
            group_name: Name of the panel group

        Returns:
            Current kWh value or None if not available
        """
        group = next(
            (g for g in self.panel_groups if g.get("name") == group_name),
            None,
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
                _LOGGER.warning(
                    "Energy sensor '%s' not found for group '%s'",
                    entity_id,
                    group_name,
                )
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
                entity_id,
                group_name,
                e,
            )
            return None

    async def get_hourly_production(self, group_name: str) -> Optional[float]:
        """Calculate production since last read (hourly delta). @zara

        This calculates the difference between current sensor value and
        the last stored value to get hourly production.

        Args:
            group_name: Name of the panel group

        Returns:
            Production in kWh since last read, or None if not available
        """
        current_value = await self.read_current_energy(group_name)

        if current_value is None:
            return None

        last_value = self._last_values.get(group_name)

        if last_value is None:
            # First reading - store and return None
            self._last_values[group_name] = current_value
            await self._save_state(group_name, current_value)
            _LOGGER.debug(
                "First reading for group '%s': %.4f kWh (no delta yet)",
                group_name,
                current_value,
            )
            return None

        # Handle counter reset (e.g., midnight reset for daily sensors)
        if current_value < last_value:
            _LOGGER.info(
                "Energy counter reset detected for group '%s': %.4f -> %.4f kWh",
                group_name,
                last_value,
                current_value,
            )
            # Assume current_value is the production since reset
            delta = current_value
        else:
            delta = current_value - last_value

        # Update stored value
        self._last_values[group_name] = current_value
        await self._save_state(group_name, current_value)

        # Sanity check: delta should be reasonable (< 10 kWh per hour is plausible)
        if delta > 10.0:
            _LOGGER.warning(
                "Unusually high hourly production for group '%s': %.4f kWh",
                group_name,
                delta,
            )

        return round(delta, 4)

    async def read_all_groups(self) -> Dict[str, float]:
        """Read current energy values for all groups with sensors. @zara

        Returns:
            Dict mapping group_name to current kWh value
        """
        results: Dict[str, float] = {}

        for group in self.get_groups_with_sensors():
            group_name = group.get("name", "Unknown")
            value = await self.read_current_energy(group_name)

            if value is not None:
                results[group_name] = value

        return results

    async def get_all_hourly_productions(self) -> Dict[str, float]:
        """Get hourly production for all groups with sensors. @zara

        Returns:
            Dict mapping group_name to hourly production in kWh
        """
        results: Dict[str, float] = {}

        for group in self.get_groups_with_sensors():
            group_name = group.get("name", "Unknown")
            production = await self.get_hourly_production(group_name)

            if production is not None:
                results[group_name] = production

        return results

    async def backfill_missing_actuals_from_recorder(
        self,
        days: int = 30,
    ) -> int:
        """Backfill missing per-group actuals from HA Recorder history. @zara"""
        if not self.has_any_sensor():
            return 0

        try:
            from homeassistant.components.recorder import get_instance
            from homeassistant.components.recorder.history import (
                state_changes_during_period,
            )
        except ImportError:
            _LOGGER.debug("Recorder not available for backfill")
            return 0

        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()
        filled = 0

        for group in self.get_groups_with_sensors():
            group_name = group.get("name", "")
            entity_id = group.get("energy_sensor", "")
            if not entity_id:
                continue

            try:
                missing = await self.fetch_all(
                    """SELECT hp.prediction_id, hp.target_date, hp.target_hour
                       FROM hourly_predictions hp
                       JOIN prediction_panel_groups ppg
                         ON ppg.prediction_id = hp.prediction_id
                       WHERE ppg.group_name = ?
                         AND ppg.actual_kwh IS NULL
                         AND hp.target_date >= ?
                         AND hp.target_date < date('now')
                       ORDER BY hp.target_date, hp.target_hour""",
                    (group_name, cutoff),
                )

                if not missing:
                    continue

                first_date = missing[0][1]
                start_time = datetime.fromisoformat(f"{first_date}T00:00:00")
                end_time = datetime.now()

                instance = get_instance(self.hass)
                states = await instance.async_add_executor_job(
                    state_changes_during_period,
                    self.hass,
                    start_time,
                    end_time,
                    entity_id,
                    False,
                    True,
                    None,
                )

                entity_states = states.get(entity_id, [])
                if len(entity_states) < 2:
                    _LOGGER.debug(
                        "Backfill %s: insufficient recorder data (%d states)",
                        group_name, len(entity_states),
                    )
                    continue

                readings = []
                unit_wh = False
                for s in entity_states:
                    if s.state in ("unavailable", "unknown", ""):
                        continue
                    try:
                        val = float(s.state)
                        if not unit_wh and hasattr(s, "attributes"):
                            unit = s.attributes.get("unit_of_measurement", "")
                            if unit.lower() == "wh":
                                unit_wh = True
                        if unit_wh:
                            val = val / 1000.0
                        readings.append((s.last_changed, val))
                    except (ValueError, TypeError):
                        continue

                if len(readings) < 2:
                    continue

                readings.sort(key=lambda x: x[0])

                group_filled = 0
                for prediction_id, target_date, target_hour in missing:
                    hour_start = datetime.fromisoformat(
                        f"{target_date}T{target_hour:02d}:00:00"
                    )
                    hour_end = datetime.fromisoformat(
                        f"{target_date}T{target_hour:02d}:59:59"
                    )

                    val_before = None
                    val_at_end = None

                    for ts, val in readings:
                        ts_naive = ts.replace(tzinfo=None) if ts.tzinfo else ts
                        if ts_naive <= hour_start:
                            val_before = val
                        if ts_naive <= hour_end:
                            val_at_end = val

                    if val_before is None or val_at_end is None:
                        continue

                    delta = val_at_end - val_before

                    if delta < -0.001:
                        continue

                    delta = max(0.0, delta)

                    if delta > 10.0:
                        continue

                    await self.execute_query(
                        """UPDATE prediction_panel_groups
                           SET actual_kwh = ?
                           WHERE prediction_id = ? AND group_name = ?
                             AND actual_kwh IS NULL""",
                        (round(delta, 4), prediction_id, group_name),
                    )
                    group_filled += 1

                if group_filled > 0:
                    _LOGGER.info(
                        "Backfill %s: %d hours recovered from recorder",
                        group_name, group_filled,
                    )
                    filled += group_filled

            except Exception as e:
                _LOGGER.warning("Backfill failed for group %s: %s", group_name, e)

        if filled > 0:
            try:
                await self.execute_query(
                    """UPDATE hourly_predictions SET actual_kwh = (
                           SELECT ROUND(SUM(ppg.actual_kwh), 4)
                           FROM prediction_panel_groups ppg
                           WHERE ppg.prediction_id = hourly_predictions.prediction_id
                             AND ppg.actual_kwh IS NOT NULL
                       )
                       WHERE actual_kwh IS NULL
                         AND target_date >= ?
                         AND target_date < date('now')
                         AND EXISTS (
                             SELECT 1 FROM prediction_panel_groups ppg
                             WHERE ppg.prediction_id = hourly_predictions.prediction_id
                               AND ppg.actual_kwh IS NOT NULL
                         )""",
                    (cutoff,),
                )
                await self.db.commit()
            except Exception as e:
                _LOGGER.warning("Backfill hourly_predictions update failed: %s", e)

            _LOGGER.info("Backfill: %d per-group actuals recovered from recorder", filled)

        return filled

    async def validate_sensors(self) -> Dict[str, Dict[str, Any]]:
        """Validate all configured energy sensors. @zara

        Returns:
            Dict with validation results per group
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
        """Validate a single energy sensor. @zara

        Checks:
        1. Entity exists
        2. Entity is numeric
        3. Unit is kWh or Wh
        """
        if not entity_id:
            return {"valid": False, "error": "No entity_id configured"}

        state = self.hass.states.get(entity_id)

        if state is None:
            # Try to find similar entities
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
            return {
                "valid": False,
                "error": f"Entity {entity_id} is not numeric: {state.state}",
            }

        unit = state.attributes.get("unit_of_measurement", "")
        if unit.lower() not in ["kwh", "wh"]:
            return {
                "valid": False,
                "error": f"Entity {entity_id} has wrong unit: {unit} (expected kWh or Wh)",
            }

        return {
            "valid": True,
            "unit": unit,
            "current_value": float(state.state),
            "state_class": state.attributes.get("state_class", "unknown"),
        }

    def _find_similar_entities(self, entity_id: str) -> List[str]:
        """Find similar entity IDs to help with debugging. @zara

        Args:
            entity_id: The entity ID that was not found

        Returns:
            List of similar entity IDs
        """
        try:
            # Extract the sensor name without domain
            if "." in entity_id:
                _, name_part = entity_id.split(".", 1)
            else:
                name_part = entity_id

            # Extract keywords from entity name
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
        tolerance_percent: float = 15.0,
    ) -> Dict[str, Any]:
        """Check if sum of group actuals matches total actual. @zara

        Warns if the deviation exceeds the tolerance threshold.

        Args:
            total_actual: Total actual production (kWh)
            group_actuals: Dict of group_name -> actual_kwh
            tolerance_percent: Acceptable deviation percentage

        Returns:
            Dict with consistency check results
        """
        if not group_actuals or total_actual <= 0:
            return {
                "consistent": True,
                "reason": "No data to compare",
            }

        group_sum = sum(group_actuals.values())
        deviation = abs(group_sum - total_actual) / total_actual * 100

        consistent = deviation <= tolerance_percent

        if not consistent:
            _LOGGER.warning(
                "Panel group sum (%.3f kWh) deviates %.1f%% from total (%.3f kWh)",
                group_sum,
                deviation,
                total_actual,
            )

        return {
            "consistent": consistent,
            "group_sum": round(group_sum, 4),
            "total_actual": round(total_actual, 4),
            "deviation_percent": round(deviation, 1),
            "tolerance_percent": tolerance_percent,
        }

    async def reset_last_values(self) -> None:
        """Reset all stored last values (e.g., at midnight). @zara"""
        self._last_values = {}

        # Clear database entries
        try:
            await self.execute_query(
                "UPDATE panel_group_sensor_state SET last_value = NULL"
            )
            _LOGGER.info("Panel group sensor last values reset")
        except Exception as e:
            _LOGGER.warning("Failed to reset panel group sensor values: %s", e)

    async def get_group_state_from_db(
        self, group_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get panel group sensor state from database. @zara

        Args:
            group_name: Name of the panel group

        Returns:
            Dict with state data or None
        """
        try:
            row = await self.fetch_one(
                """SELECT group_name, last_value, last_updated
                   FROM panel_group_sensor_state
                   WHERE group_name = ?""",
                (group_name,),
            )

            if not row:
                return None

            return {
                "group_name": row[0],
                "last_value": row[1],
                "last_updated": row[2],
            }

        except Exception as e:
            _LOGGER.error("Failed to get group state from DB: %s", e)
            return None

    async def get_all_states_from_db(self) -> Dict[str, Dict[str, Any]]:
        """Get all panel group sensor states from database. @zara

        Returns:
            Dict mapping group_name to state data
        """
        try:
            rows = await self.fetch_all(
                "SELECT group_name, last_value, last_updated FROM panel_group_sensor_state"
            )

            return {
                row[0]: {
                    "group_name": row[0],
                    "last_value": row[1],
                    "last_updated": row[2],
                }
                for row in rows
            }

        except Exception as e:
            _LOGGER.error("Failed to get all states from DB: %s", e)
            return {}

    async def get_sensor_summary(self) -> Dict[str, Any]:
        """Get summary of panel group sensor configuration and status. @zara

        Returns:
            Dict with sensor summary information
        """
        groups_with_sensors = self.get_groups_with_sensors()
        validation_results = await self.validate_sensors()

        valid_sensors = sum(
            1 for v in validation_results.values() if v.get("valid", False)
        )

        return {
            "total_groups": len(self.panel_groups),
            "groups_with_sensors": len(groups_with_sensors),
            "valid_sensors": valid_sensors,
            "invalid_sensors": len(groups_with_sensors) - valid_sensors,
            "groups": [
                {
                    "name": g.get("name", "Unknown"),
                    "has_sensor": bool(g.get("energy_sensor")),
                    "entity_id": g.get("energy_sensor"),
                    "validation": validation_results.get(g.get("name", "Unknown"), {}),
                }
                for g in self.panel_groups
            ],
        }
