"""
Button platform for the Solar Forecast ML integration.

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
License: GNU Affero General Public License v3
"""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .helpers import SafeDateTimeUtil as dt_util
from .coordinator import SolarForecastMLCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the button entities from a config entry."""
    coordinator: SolarForecastMLCoordinator = hass.data[DOMAIN][entry.entry_id]

    buttons = [
        ManualForecastButton(coordinator, entry),
        ManualLearningButton(coordinator, entry),
        MLBackfillButton(coordinator, entry),
    ]

    async_add_entities(buttons)
    _LOGGER.info("Solar Forecast ML Buttons erfolgreich eingerichtet: Forecast, Learning, Backfill.")


# =============================================================================
# 1. Manual Forecast Button
# =============================================================================
class ManualForecastButton(ButtonEntity):
    """Button to trigger a manual forecast refresh."""

    _attr_has_entity_name = True
    _attr_translation_key = "manual_forecast"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry) -> None:
        """Initialize the button."""
        super().__init__()  # WICHTIG: Registriert async_press!

        self.coordinator = coordinator
        self._entry_id = entry.entry_id

        self._attr_unique_id = f"{self._entry_id}_manual_forecast"
        self._attr_name = "Manual Forecast"
        self._attr_icon = "mdi:refresh-circle"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self._entry_id)},
            name="Solar Forecast ML",
            manufacturer="Zara-Toorox",
            model="v5.0.2",
            entry_type="service",
        )

    async def async_press(self) -> None:
        """Handle the button press."""
        _LOGGER.info("Manual Forecast Button gedrückt – Starte Refresh.")
        await self.coordinator.async_request_refresh()


# =============================================================================
# 2. Manual Learning Button
# =============================================================================
class ManualLearningButton(ButtonEntity):
    """Button to trigger manual ML model training."""

    _attr_has_entity_name = True
    _attr_translation_key = "manual_learning"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry) -> None:
        """Initialize the button."""
        super().__init__()

        self.coordinator = coordinator
        self._entry_id = entry.entry_id

        self._attr_unique_id = f"{self._entry_id}_manual_learning"
        self._attr_name = "Manual Learning Process"
        self._attr_icon = "mdi:brain"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self._entry_id)},
        )

    async def async_press(self) -> None:
        """Handle the button press."""
        _LOGGER.info("Manual Learning Button gedrückt – Starte Training.")

        ml_predictor = self.coordinator.ml_predictor
        if not ml_predictor:
            _LOGGER.error("ML Predictor nicht verfügbar – Training abgebrochen.")
            return

        try:
            result = await ml_predictor.train_model()
            timestamp = dt_util.utcnow()

            if result.success:
                _LOGGER.info(f"ML-Training erfolgreich. Genauigkeit: {result.accuracy:.2f}")
                if hasattr(self.coordinator, "on_ml_training_complete"):
                    self.coordinator.on_ml_training_complete(
                        timestamp=timestamp,
                        accuracy=result.accuracy
                    )
            else:
                _LOGGER.error(f"ML-Training fehlgeschlagen: {result.error_message}")

        except Exception as e:
            _LOGGER.error(f"Unerwarteter Fehler beim Training: {e}", exc_info=True)


# =============================================================================
# 3. Backfill Training Button (DER WICHTIGE!)
# =============================================================================
class MLBackfillButton(ButtonEntity):
    """Button to trigger the backfill training process."""

    _attr_has_entity_name = True
    _attr_translation_key = "start_backfill_training"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry) -> None:
        """Initialize the backfill button."""
        super().__init__()  # KRITISCH: Ohne das wird async_press NIE aufgerufen!

        self.coordinator = coordinator
        self._entry_id = entry.entry_id

        self._attr_unique_id = f"{self._entry_id}_backfill_trigger"
        self._attr_name = "Start Backfill Training"
        self._attr_icon = "mdi:database-refresh"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self._entry_id)},
        )

    async def async_press(self) -> None:
        """Handle the button press – start backfill."""
        _LOGGER.info("BACKFILL BUTTON GEDRÜCKT – Starte Backfill-Prozess...")

        if not hasattr(self.coordinator, "trigger_backfill"):
            _LOGGER.error("Coordinator hat keine Methode 'trigger_backfill()'!")
            return

        try:
            success = await self.coordinator.trigger_backfill()
            if success:
                _LOGGER.info("Backfill-Prozess erfolgreich gestartet.")
            else:
                _LOGGER.warning("Backfill konnte nicht gestartet werden (siehe Coordinator-Logs).")
        except Exception as e:
            _LOGGER.error(f"Fehler beim Auslösen von Backfill: {e}", exc_info=True)

        # Trigger UI update
        await self.coordinator.async_request_refresh()