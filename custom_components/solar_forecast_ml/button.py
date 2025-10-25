"""
Button-Plattform für die Solar Forecast ML Integration.

Diese Datei erstellt die Entitäten für Buttons, die es dem Benutzer ermöglichen,
Aktionen wie eine manuelle Prognose oder einen Lernprozess auszulösen.

Copyright (C) 2025 Zara-Toorox

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
"""
from __future__ import annotations
import logging

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .coordinator import SolarForecastMLCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Richtet die Buttons ein."""
    coordinator: SolarForecastMLCoordinator = hass.data[DOMAIN][entry.entry_id]
    
    async_add_entities([
        ManualForecastButton(coordinator, entry),
        ManualLearningButton(coordinator, entry),
    ])
    _LOGGER.info("Buttons für manuelle Prognose und Lernen erfolgreich eingerichtet.")


class ManualForecastButton(ButtonEntity):
    """Ein Button, um die Prognose manuell auszulösen."""
    _attr_has_entity_name = True

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialisiere den Prognose-Button."""
        self.coordinator = coordinator
        self._attr_unique_id = f"{entry.entry_id}_manual_forecast"
        self._attr_name = "Manuelle Prognose"
        self._attr_icon = "mdi:refresh-circle"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Solar Forecast ML",
            manufacturer="Zara-Toorox",
            model="v5.0.2",
        )

    async def async_press(self) -> None:
        """Behandelt den Button-Druck - Nutzt existierende Coordinator-Methode - von Zara"""
        _LOGGER.info("ðŸ”„ Manuelle Prognose ausgelöst - von Zara")
        await self.coordinator.async_request_refresh()


class ManualLearningButton(ButtonEntity):
    """Ein Button, um den Lernprozess manuell auszulösen."""
    _attr_has_entity_name = True

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialisiere den Lern-Button."""
        self.coordinator = coordinator
        self._attr_unique_id = f"{entry.entry_id}_manual_learning"
        self._attr_name = "Manueller Lernprozess"
        self._attr_icon = "mdi:brain"
        # Die Geräte-Info verknüpft diesen Button mit dem selben Gerät wie die Sensoren - von Zara
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    async def async_press(self) -> None:
        """Behandelt den Button-Druck - Nutzt ML Predictor force_retrain - von Zara"""
        _LOGGER.info("ðŸ§  Manuelles ML-Training ausgelöst - von Zara")
        
        # Prüfe ob ML Predictor verfügbar - von Zara
        ml_predictor = self.coordinator.ml_predictor
        
        if not ml_predictor:
            _LOGGER.error("âŒ ML Predictor nicht verfügbar - Training nicht möglich - von Zara")
            return
        
        try:
            # Nutze existierende force_retrain Methode - von Zara
            result = await ml_predictor.force_retrain()
            _LOGGER.info(f"✓ ML-Training abgeschlossen - Accuracy: {result.accuracy} - von Zara")
            
            # Update Coordinator mit Training-Ergebnis - von Zara
            if hasattr(self.coordinator, 'on_ml_training_complete'):
                self.coordinator.on_ml_training_complete(
                    timestamp=result.timestamp,
                    accuracy=result.accuracy
                )
                
        except Exception as e:
            _LOGGER.error(f"âŒ ML-Training fehlgeschlagen: {e} - von Zara")
