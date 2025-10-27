"""
Button-Plattform fÃ¼r die Solar Forecast ML Integration.

Diese Datei erstellt die EntitÃ¤ten fÃ¼r Buttons, die es dem Benutzer ermÃ¶glichen,
Aktionen wie eine manuelle Prognose oder einen Lernprozess auszulÃ¶sen.

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
from .helpers import SafeDateTimeUtil as dt_util
from .coordinator import SolarForecastMLCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: SolarForecastMLCoordinator = hass.data[DOMAIN][entry.entry_id]
    
    async_add_entities([
        ManualForecastButton(coordinator, entry),
        ManualLearningButton(coordinator, entry),
    ])
    _LOGGER.info("Buttons fÃ¼r manuelle Prognose und Lernen erfolgreich eingerichtet.")


class ManualForecastButton(ButtonEntity):
    _attr_has_entity_name = True

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
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
        _LOGGER.info("ðŸ”„ Manuelle Prognose ausgelÃ¶st - von Zara")
        await self.coordinator.async_request_refresh()


class ManualLearningButton(ButtonEntity):
    _attr_has_entity_name = True

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        self.coordinator = coordinator
        self._attr_unique_id = f"{entry.entry_id}_manual_learning"
        self._attr_name = "Manueller Lernprozess"
        self._attr_icon = "mdi:brain"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    async def async_press(self) -> None:
        _LOGGER.info("ðŸ§  Manuelles ML-Training ausgelÃ¶st - von Zara")
        
        ml_predictor = self.coordinator.ml_predictor
        
        if not ml_predictor:
            _LOGGER.error("âŒ ML Predictor nicht verfÃ¼gbar - Training nicht mÃ¶glich - von Zara")
            return
        
        try:
            result = await ml_predictor.train_model()
            
            if result.success:
                timestamp = dt_util.utcnow()
                _LOGGER.info(f"âœ“ ML-Training abgeschlossen - Accuracy: {result.accuracy:.2f} - von Zara")
                
                if hasattr(self.coordinator, 'on_ml_training_complete'):
                    self.coordinator.on_ml_training_complete(
                        timestamp=timestamp,
                        accuracy=result.accuracy
                    )
            else:
                _LOGGER.error(f"âŒ ML-Training fehlgeschlagen: {result.error_message} - von Zara")
                
        except Exception as e:
            _LOGGER.error(f"âŒ ML-Training fehlgeschlagen: {e} - von Zara")
