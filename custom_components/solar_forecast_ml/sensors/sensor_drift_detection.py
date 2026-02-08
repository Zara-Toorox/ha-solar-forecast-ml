# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Drift detection diagnostic sensor for Solar Forecast ML V17.0.0.
Exposes drift monitoring status as a Home Assistant sensor entity.

@zara V17.0.0
"""

import logging
from typing import Any, Dict, Optional

from homeassistant.components.sensor import SensorEntity
from homeassistant.core import callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from ..const import (
    DOMAIN,
    INTEGRATION_MODEL,
    SOFTWARE_VERSION,
    AI_VERSION,
)

_LOGGER = logging.getLogger(__name__)


class DriftStatusSensor(CoordinatorEntity, SensorEntity):
    """Diagnostic sensor for AI model drift detection. @zara V17.0.0

    Shows overall drift status: 'stable', 'warning', 'critical', 'recovering'.
    Attributes expose per-group status, active events, physics boost, and metrics.
    """

    def __init__(self, coordinator, entry):
        """Initialize drift status sensor."""
        super().__init__(coordinator)
        self._entry = entry
        self._attr_has_entity_name = True
        self._attr_translation_key = "drift_status"
        self._attr_unique_id = f"{entry.entry_id}_drift_status"
        self._attr_device_class = None
        self._attr_state_class = None
        self._attr_icon = "mdi:chart-timeline-variant-shimmer"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Solar Forecast ML",
            manufacturer="Zara-Toorox",
            model=INTEGRATION_MODEL,
            sw_version=f"SW {SOFTWARE_VERSION} | AI {AI_VERSION}",
            configuration_url="https://github.com/Zara-Toorox/ha-solar-forecast-ml",
        )

        self._drift_status: Dict[str, Any] = {}

    @property
    def native_value(self) -> str:
        """Return overall drift state."""
        return self._drift_status.get("state", "unknown")

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return detailed drift detection attributes."""
        status = self._drift_status
        return {
            "global_mae_7d": status.get("global_mae_7d"),
            "global_mae_30d": status.get("global_mae_30d"),
            "global_bias_30d": status.get("global_bias_30d"),
            "global_coverage_20_30d": status.get("global_coverage_20_30d"),
            "active_drift_events": status.get("active_drift_events", 0),
            "physics_boost_active": status.get("physics_boost_active", False),
            "per_group_status": status.get("per_group_status", {}),
            "last_check": status.get("last_check"),
        }

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator data update — refresh drift status from cache."""
        try:
            drift_monitor = getattr(self.coordinator, "drift_monitor", None)
            if drift_monitor:
                # The drift status is cached on the coordinator by EOD step 14
                cached = getattr(self.coordinator, "_drift_status_cache", None)
                if cached:
                    self._drift_status = cached
        except Exception as e:
            _LOGGER.debug("Drift sensor update failed: %s", e)

        self.async_write_ha_state()


# Sensor class list for registration in sensor.py
DRIFT_DETECTION_SENSORS = [
    DriftStatusSensor,
]
