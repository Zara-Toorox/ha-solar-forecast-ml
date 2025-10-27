"""
Scheduled Tasks Manager Module
Verwaltet tÃ¤gliche Updates und Verifikationen

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""
import logging
from datetime import datetime
from typing import Optional

from homeassistant.core import HomeAssistant, callback
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)


class ScheduledTasksManager:
    """
    Verwaltet geplante Tasks: Morgen-Update und Abend-Verifikation
    """
    
    def __init__(
        self,
        hass: HomeAssistant,
        coordinator,
        solar_yield_today: Optional[str],
        data_manager
    ):
        self.hass = hass
        self.coordinator = coordinator
        self.solar_yield_today = solar_yield_today
        self.data_manager = data_manager
    
    @callback
    async def scheduled_morning_update(self, now: datetime) -> None:
        _LOGGER.info("ðŸŒ… === TÃ„GLICHER MORGEN-UPDATE GESTARTET ===")
        _LOGGER.info(f"ï¿½Â Zeitpunkt: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            await self.coordinator.async_request_refresh()
            _LOGGER.info("Ã¢Å“â€œ Tagesprognose erfolgreich erstellt")
            
        except Exception as e:
            _LOGGER.error(f"Ã¢ÂÅ’ Morgen-Update fehlgeschlagen: {e}", exc_info=True)
    
    @callback
    async def scheduled_evening_verification(self, now: datetime) -> None:
        _LOGGER.info("ðŸŒ† === TÃ„GLICHE PROGNOSE-VERIFIKATION GESTARTET ===")
        _LOGGER.info(f"ï¿½Â Zeitpunkt: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not self.solar_yield_today:
            _LOGGER.warning("Ã¢Å¡Â Ã¯Â¸Â Kein solar_yield_today Sensor konfiguriert - Verifikation Ã¼bersprungen")
            return
        
        try:
            state = self.hass.states.get(self.solar_yield_today)
            
            if not state or state.state in ['unavailable', 'unknown', 'none', None]:
                _LOGGER.warning(f"Ã¢Å¡Â Ã¯Â¸Â Sensor {self.solar_yield_today} nicht verfÃ¼gbar")
                return
            
            try:
                actual_kwh = float(state.state)
            except (ValueError, TypeError):
                _LOGGER.warning(f"Ã¢Å¡Â Ã¯Â¸Â UngÃ¼ltiger Wert von {self.solar_yield_today}: {state.state}")
                return
            
            if not self.coordinator.data:
                _LOGGER.warning("Ã¢Å¡Â Ã¯Â¸Â Keine Coordinator-Daten verfÃ¼gbar")
                return
                
            predicted_kwh = self.coordinator.data.get("forecast_today", 0.0)
            
            if predicted_kwh > 0.1:
                error = abs(predicted_kwh - actual_kwh)
                relative_error = error / predicted_kwh
                accuracy = max(0.0, 1.0 - relative_error)
            else:
                accuracy = 0.0
            
            try:
                await self.data_manager.update_today_predictions_actual(actual_kwh, accuracy)
                _LOGGER.debug("Ã¢Å“â€œ Alle heutigen Prediction Records mit actual_value aktualisiert")
            except Exception as e:
                _LOGGER.warning(f"Ã¢Å¡Â Ã¯Â¸Â Konnte heutige Prediction Records nicht aktualisieren: {e}")
            
            _LOGGER.info(
                f"Ã°Å¸â€œÅ  TAGESPROGNOSE-CHECK:\n"
                f"   Predicted: {predicted_kwh:.2f} kWh\n"
                f"   Actual:    {actual_kwh:.2f} kWh\n"
                f"   Error:     {abs(predicted_kwh - actual_kwh):.2f} kWh\n"
                f"   Accuracy:  {accuracy*100:.1f}%"
            )
            
            self.coordinator.last_day_error_kwh = abs(predicted_kwh - actual_kwh)
            
        except Exception as e:
            _LOGGER.error(f"Ã¢ÂÅ’ Abend-Verifikation fehlgeschlagen: {e}", exc_info=True)
