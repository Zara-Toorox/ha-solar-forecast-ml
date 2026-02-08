# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
State Handler for Solar Forecast ML V16.2.0.
Manages coordinator state and production time state via database.
All operations use DatabaseManager for SQLite access.

@zara
"""

import logging
from datetime import datetime
from typing import Any, Optional

from homeassistant.core import HomeAssistant

from ..const import DATA_VERSION
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from .db_manager import DatabaseManager
from .data_io import DataManagerIO

_LOGGER = logging.getLogger(__name__)


class DataStateHandler(DataManagerIO):
    """Handles coordinator state and production time state via database. @zara

    Provides methods to:
    - Save/load expected daily production
    - Track last collected hour
    - Manage production time state
    - Handle yield cache
    """

    def __init__(self, hass: HomeAssistant, db_manager: DatabaseManager):
        """Initialize the state handler. @zara

        Args:
            hass: Home Assistant instance
            db_manager: DatabaseManager for database operations
        """
        super().__init__(hass, db_manager)
        _LOGGER.debug("DataStateHandler initialized")

    async def save_expected_daily_production(self, value: float) -> bool:
        """Save expected daily production value. @zara

        Args:
            value: Expected daily production in kWh

        Returns:
            True if saved successfully
        """
        try:
            now_local = dt_util.now()
            state = {
                "expected_daily_production": value,
                "last_set_date": now_local.date().isoformat(),
                "last_updated": now_local.isoformat(),
            }

            await self.db.save_coordinator_state(state)
            _LOGGER.debug("Expected daily production saved: %.2f kWh", value)
            return True

        except Exception as e:
            _LOGGER.error("Failed to save expected daily production: %s", e)
            return False

    async def load_expected_daily_production(self) -> Optional[float]:
        """Load expected daily production from database. @zara

        Returns:
            Expected daily production in kWh or None if not set
        """
        try:
            state = await self.db.get_coordinator_state()
            if state is None:
                return None

            value = state.get("expected_daily_production")
            if value is not None:
                _LOGGER.debug("Loaded expected daily production: %.2f kWh", float(value))
                return float(value)

            return None

        except Exception as e:
            _LOGGER.error("Failed to load expected daily production: %s", e)
            return None

    async def clear_expected_daily_production(self) -> bool:
        """Clear expected daily production from database. @zara

        Returns:
            True if cleared successfully
        """
        try:
            now_local = dt_util.now()
            state = {
                "expected_daily_production": None,
                "last_set_date": now_local.date().isoformat(),
                "last_updated": now_local.isoformat(),
            }

            await self.db.save_coordinator_state(state)
            _LOGGER.debug("Expected daily production cleared")
            return True

        except Exception as e:
            _LOGGER.error("Failed to clear expected daily production: %s", e)
            return False

    async def get_last_collected_hour(self) -> Optional[datetime]:
        """Get the timestamp of the last collected hourly sample. @zara

        Returns:
            datetime of last collected hour or None
        """
        try:
            # Query the most recent hourly prediction @zara
            row = await self.db.fetchone(
                """SELECT MAX(prediction_created_at) as last_collected
                   FROM hourly_predictions
                   WHERE actual_kwh IS NOT NULL"""
            )

            if row and row[0]:
                return dt_util.parse_datetime(str(row[0]))

            return None

        except Exception as e:
            _LOGGER.error("Failed to get last collected hour: %s", e)
            return None

    async def set_last_collected_hour(self, timestamp: datetime) -> bool:
        """Record that a specific hour has been collected. @zara

        This is tracked implicitly by the hourly_predictions table.

        Args:
            timestamp: The hour that was collected

        Returns:
            True (always, as this is informational)
        """
        _LOGGER.debug("Last collected hour: %s", timestamp.isoformat())
        return True

    async def save_production_time_state(self, state: dict[str, Any]) -> bool:
        """Save production time state to database. @zara

        Args:
            state: Production time state dictionary

        Returns:
            True if saved successfully
        """
        try:
            await self.db.save_production_time_state(state)
            _LOGGER.debug("Production time state saved")
            return True

        except Exception as e:
            _LOGGER.error("Failed to save production time state: %s", e)
            return False

    async def load_production_time_state(self) -> Optional[dict[str, Any]]:
        """Load production time state from database. @zara

        Returns:
            Production time state dictionary or None
        """
        try:
            row = await self.db.fetchone(
                """SELECT date, accumulated_hours, is_active, start_time,
                          production_time_today, last_updated
                   FROM production_time_state WHERE id = 1"""
            )

            if row is None:
                return None

            return {
                "date": str(row[0]) if row[0] else None,
                "accumulated_hours": float(row[1]) if row[1] else 0.0,
                "is_active": bool(row[2]) if row[2] is not None else False,
                "start_time": str(row[3]) if row[3] else None,
                "production_time_today": str(row[4]) if row[4] else "00:00:00",
                "last_updated": str(row[5]) if row[5] else None,
            }

        except Exception as e:
            _LOGGER.error("Failed to load production time state: %s", e)
            return None

    async def save_yield_cache(self, value: float, timestamp: datetime) -> bool:
        """Save yield cache to database. @zara

        Args:
            value: Current yield value in kWh
            timestamp: Timestamp of the value

        Returns:
            True if saved successfully
        """
        try:
            cache = {
                "value": value,
                "time": timestamp.isoformat(),
                "date": timestamp.date().isoformat(),
            }
            await self.db.save_yield_cache(cache)
            return True

        except Exception as e:
            _LOGGER.error("Failed to save yield cache: %s", e)
            return False

    async def load_yield_cache(self) -> Optional[dict[str, Any]]:
        """Load yield cache from database. @zara

        Returns:
            Yield cache dictionary or None
        """
        try:
            row = await self.db.fetchone(
                "SELECT value, time, date FROM yield_cache WHERE id = 1"
            )

            if row is None:
                return None

            return {
                "value": float(row[0]) if row[0] is not None else None,
                "time": str(row[1]) if row[1] else None,
                "date": str(row[2]) if row[2] else None,
            }

        except Exception as e:
            _LOGGER.error("Failed to load yield cache: %s", e)
            return None

    async def save_panel_group_sensor_state(
        self,
        group_name: str,
        last_value: float
    ) -> bool:
        """Save panel group sensor state. @zara

        Args:
            group_name: Name of the panel group
            last_value: Last sensor value

        Returns:
            True if saved successfully
        """
        try:
            state = {
                "last_values": {group_name: last_value},
                "last_updated": dt_util.now().isoformat(),
            }
            await self.db.save_panel_group_sensor_state(state)
            return True

        except Exception as e:
            _LOGGER.error("Failed to save panel group sensor state: %s", e)
            return False

    async def load_panel_group_sensor_state(self) -> dict[str, float]:
        """Load all panel group sensor states. @zara

        Returns:
            Dictionary mapping group names to last values
        """
        try:
            rows = await self.db.fetchall(
                "SELECT group_name, last_value FROM panel_group_sensor_state"
            )

            return {row[0]: float(row[1]) for row in rows if row[1] is not None}

        except Exception as e:
            _LOGGER.error("Failed to load panel group sensor states: %s", e)
            return {}

    async def get_coordinator_state(self) -> Optional[dict[str, Any]]:
        """Get complete coordinator state. @zara

        Returns:
            Coordinator state dictionary or None
        """
        try:
            return await self.db.get_coordinator_state()
        except Exception as e:
            _LOGGER.error("Failed to get coordinator state: %s", e)
            return None
