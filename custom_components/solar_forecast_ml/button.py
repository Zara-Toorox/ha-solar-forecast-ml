"""
Button Platform for Solar Forecast ML Integration

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

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

# Use constants for domain and potentially device info
from .const import DOMAIN, INTEGRATION_MODEL, ML_VERSION, SOFTWARE_VERSION

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """
    Set up the button entities from a config entry

    v8.6.0: ALL BUTTONS REMOVED
    - No user-facing buttons to prevent confusion
    - Training controlled via service calls (solar_forecast_ml.force_retrain)
    - Users should rely on Training Readiness sensor for status
    """
    # === [v8.6.0] NO BUTTONS ===
    buttons = []

    async_add_entities(buttons)
    _LOGGER.info("Solar Forecast ML: No user-facing buttons (training via services only).")


# =============================================================================
# ALL BUTTONS REMOVED in v8.6.0
# =============================================================================
# 1. Manual Forecast Button (REMOVED - Redundant with automatic workflows)
#    - Removed on 2025-11-09: Automatic workflow at 6 AM sets the forecast
#
# 2. Manual Learning Button (REMOVED - Confusing UX with sample counts)
#    - Removed on 2025-11-11: Users confused by "samples" count vs training readiness
#    - Replaced by Training Readiness sensor showing actual quality status
#    - Advanced users can use service: solar_forecast_ml.force_retrain
#
# 3. Backfill Training Button (REMOVED - Not needed)
#    - Removed earlier: Backfill functionality not required
#
# =============================================================================
