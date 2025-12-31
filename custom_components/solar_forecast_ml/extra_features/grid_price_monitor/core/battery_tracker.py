# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.storage import Store

from ..const import DOMAIN

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
STORAGE_KEY = f"{DOMAIN}.battery_stats"


class BatteryTracker:
    """Tracks battery charging statistics using Left Riemann Sum @zara"""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        storage_path: Path | None = None
    ) -> None:
        """Initialize the battery tracker @zara

        Args:
            hass: Home Assistant instance
            entry_id: Config entry ID
            storage_path: Optional custom storage path
        """
        self.hass = hass
        self._entry_id = entry_id
        self._store = Store(hass, STORAGE_VERSION, f"{STORAGE_KEY}_{entry_id}")
        self._storage_path = storage_path

        # Current state
        self._power_sensor_id: str | None = None
        self._last_power_w: float = 0.0
        self._last_update: datetime | None = None

        # Statistics
        self._energy_today_wh: float = 0.0
        self._energy_week_wh: float = 0.0
        self._energy_month_wh: float = 0.0

        # Date tracking for reset
        self._current_day: int | None = None
        self._current_week: int | None = None
        self._current_month: int | None = None

        # Unsubscribe callback
        self._unsub_state_change = None

        # Debounce save - prevent race condition from rapid updates
        self._save_pending = False
        self._save_interval_seconds = 60  # Save at most every 60 seconds

    @property
    def current_power(self) -> float:
        """Get current battery charging power in W @zara"""
        return self._last_power_w

    @property
    def energy_today_kwh(self) -> float:
        """Get energy charged today in kWh @zara"""
        return round(self._energy_today_wh / 1000, 3)

    @property
    def energy_week_kwh(self) -> float:
        """Get energy charged this week in kWh @zara"""
        return round(self._energy_week_wh / 1000, 3)

    @property
    def energy_month_kwh(self) -> float:
        """Get energy charged this month in kWh @zara"""
        return round(self._energy_month_wh / 1000, 3)

    async def async_setup(self, power_sensor_id: str) -> None:
        """Set up the battery tracker with a power sensor @zara"""
        self._power_sensor_id = power_sensor_id

        # Load stored data
        await self._async_load_data()

        # Subscribe to state changes
        if power_sensor_id:
            # Read initial sensor value
            state = self.hass.states.get(power_sensor_id)
            if state and state.state not in ("unknown", "unavailable"):
                try:
                    self._last_power_w = max(0.0, float(state.state))
                    self._last_update = datetime.now(timezone.utc)
                    _LOGGER.debug(
                        "Battery tracker initial power: %.1f W from %s",
                        self._last_power_w,
                        power_sensor_id
                    )
                except (ValueError, TypeError):
                    _LOGGER.warning("Invalid initial power value: %s", state.state)

            self._unsub_state_change = async_track_state_change_event(
                self.hass,
                [power_sensor_id],
                self._async_state_changed,
            )
            _LOGGER.debug("Battery tracker set up for sensor: %s", power_sensor_id)

    async def async_unload(self) -> None:
        """Unload the battery tracker @zara"""
        if self._unsub_state_change:
            self._unsub_state_change()
            self._unsub_state_change = None

        # Save data before unloading
        await self._async_save_data()

    @callback
    def _async_state_changed(self, event) -> None:
        """Handle state changes of the power sensor @zara"""
        new_state = event.data.get("new_state")
        if new_state is None or new_state.state in ("unknown", "unavailable"):
            return

        try:
            power_w = float(new_state.state)
        except (ValueError, TypeError):
            _LOGGER.warning("Invalid power value: %s", new_state.state)
            return

        # Only track positive values (charging from grid)
        power_w = max(0.0, power_w)

        now = datetime.now(timezone.utc)

        # Check for date resets
        self._check_date_resets(now)

        # Left Riemann Sum: use the PREVIOUS power value for the interval
        if self._last_update is not None:
            # Calculate time delta in hours
            delta_hours = (now - self._last_update).total_seconds() / 3600

            # Energy = Power × Time (Wh)
            # Use previous power value (left Riemann sum)
            energy_wh = self._last_power_w * delta_hours

            # Add to statistics
            self._energy_today_wh += energy_wh
            self._energy_week_wh += energy_wh
            self._energy_month_wh += energy_wh

            _LOGGER.debug(
                "Battery energy update: +%.3f Wh (%.1f W × %.4f h)",
                energy_wh,
                self._last_power_w,
                delta_hours,
            )

        # Update state for next calculation
        self._last_power_w = power_w
        self._last_update = now

        # Schedule debounced save to prevent race conditions from rapid updates
        self._schedule_save()

    def _check_date_resets(self, now: datetime) -> None:
        """Check if daily/weekly/monthly reset is needed @zara"""
        local_now = now.astimezone()
        current_day = local_now.day
        current_week = local_now.isocalendar()[1]
        current_month = local_now.month

        # Daily reset
        if self._current_day is not None and current_day != self._current_day:
            _LOGGER.info(
                "Daily reset: %.3f kWh charged yesterday",
                self._energy_today_wh / 1000,
            )
            self._energy_today_wh = 0.0

        # Weekly reset (on Monday)
        if self._current_week is not None and current_week != self._current_week:
            _LOGGER.info(
                "Weekly reset: %.3f kWh charged last week",
                self._energy_week_wh / 1000,
            )
            self._energy_week_wh = 0.0

        # Monthly reset
        if self._current_month is not None and current_month != self._current_month:
            _LOGGER.info(
                "Monthly reset: %.3f kWh charged last month",
                self._energy_month_wh / 1000,
            )
            self._energy_month_wh = 0.0

        # Update current date values
        self._current_day = current_day
        self._current_week = current_week
        self._current_month = current_month

    @callback
    def _schedule_save(self) -> None:
        """Schedule a debounced save operation @zara"""
        if self._save_pending:
            return  # Already scheduled

        self._save_pending = True

        async def _delayed_save():
            """Perform the delayed save @zara"""
            await self._async_save_data()
            self._save_pending = False

        # Schedule save after interval
        self.hass.loop.call_later(
            self._save_interval_seconds,
            lambda: self.hass.async_create_task(_delayed_save())
        )

    async def _async_load_data(self) -> None:
        """Load stored statistics from disk @zara"""
        data = await self._store.async_load()
        if data is None:
            _LOGGER.debug("No stored battery statistics found")
            return

        now = datetime.now(timezone.utc).astimezone()

        # Load values and check if they're still valid
        stored_day = data.get("current_day")
        stored_week = data.get("current_week")
        stored_month = data.get("current_month")

        current_day = now.day
        current_week = now.isocalendar()[1]
        current_month = now.month

        # Only restore if same day
        if stored_day == current_day:
            self._energy_today_wh = data.get("energy_today_wh", 0.0)
        else:
            self._energy_today_wh = 0.0

        # Only restore if same week
        if stored_week == current_week:
            self._energy_week_wh = data.get("energy_week_wh", 0.0)
        else:
            self._energy_week_wh = 0.0

        # Only restore if same month
        if stored_month == current_month:
            self._energy_month_wh = data.get("energy_month_wh", 0.0)
        else:
            self._energy_month_wh = 0.0

        self._current_day = current_day
        self._current_week = current_week
        self._current_month = current_month

        _LOGGER.debug(
            "Loaded battery statistics: today=%.3f kWh, week=%.3f kWh, month=%.3f kWh",
            self.energy_today_kwh,
            self.energy_week_kwh,
            self.energy_month_kwh,
        )

    async def _async_save_data(self) -> None:
        """Save statistics to disk @zara"""
        data = {
            "energy_today_wh": self._energy_today_wh,
            "energy_week_wh": self._energy_week_wh,
            "energy_month_wh": self._energy_month_wh,
            "current_day": self._current_day,
            "current_week": self._current_week,
            "current_month": self._current_month,
        }
        await self._store.async_save(data)

    def get_statistics(self) -> dict[str, Any]:
        """Get all battery statistics @zara"""
        return {
            "battery_power": self._last_power_w,
            "battery_charged_today": self.energy_today_kwh,
            "battery_charged_week": self.energy_week_kwh,
            "battery_charged_month": self.energy_month_kwh,
        }
