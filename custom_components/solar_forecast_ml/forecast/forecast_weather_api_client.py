"""
Weather API Client for Solar Forecast ML Integration

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

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from homeassistant.core import HomeAssistant, State

_LOGGER = logging.getLogger(__name__)


class WeatherAPIClient:
    """Handles communication with weather entities and services by @Zara"""
    
    def __init__(self, hass: HomeAssistant):
        """Initialize weather API client by @Zara"""
        self.hass = hass
        self._weather_entity: Optional[str] = None
    
    def set_weather_entity(self, entity_id: str) -> None:
        """Set the weather entity to use by @Zara"""
        self._weather_entity = entity_id
        _LOGGER.debug(f"Weather entity set to: {entity_id}")
    
    async def get_current_weather(self) -> Optional[Dict[str, Any]]:
        """Get current weather data from configured entity by @Zara"""
        if not self._weather_entity:
            _LOGGER.error("Weather entity not configured")
            return None
        
        state = self.hass.states.get(self._weather_entity)
        if not state:
            _LOGGER.error(f"Weather entity not found: {self._weather_entity}")
            return None
        
        return self._extract_weather_data(state)
    
    async def get_hourly_forecast(self) -> Optional[List[Dict[str, Any]]]:
        """Get hourly weather forecast from configured entity by @Zara"""
        if not self._weather_entity:
            _LOGGER.error("Weather entity not configured")
            return None
        
        try:
            # Call weather.get_forecasts service
            response = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                {
                    "entity_id": self._weather_entity,
                    "type": "hourly"
                },
                blocking=True,
                return_response=True
            )
            
            if not response or self._weather_entity not in response:
                _LOGGER.error("No forecast data received")
                return None
            
            forecast_data = response[self._weather_entity].get("forecast", [])
            _LOGGER.debug(f"Received {len(forecast_data)} hourly forecasts")
            
            return forecast_data
            
        except Exception as e:
            _LOGGER.error(f"Failed to get hourly forecast: {e}", exc_info=True)
            return None
    
    async def get_daily_forecast(self) -> Optional[List[Dict[str, Any]]]:
        """Get daily weather forecast from configured entity by @Zara"""
        if not self._weather_entity:
            _LOGGER.error("Weather entity not configured")
            return None
        
        try:
            # Call weather.get_forecasts service
            response = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                {
                    "entity_id": self._weather_entity,
                    "type": "daily"
                },
                blocking=True,
                return_response=True
            )
            
            if not response or self._weather_entity not in response:
                _LOGGER.error("No forecast data received")
                return None
            
            forecast_data = response[self._weather_entity].get("forecast", [])
            _LOGGER.debug(f"Received {len(forecast_data)} daily forecasts")
            
            return forecast_data
            
        except Exception as e:
            _LOGGER.error(f"Failed to get daily forecast: {e}", exc_info=True)
            return None
    
    def _extract_weather_data(self, state: State) -> Dict[str, Any]:
        """Extract weather data from state object by @Zara"""
        attributes = state.attributes
        
        return {
            "condition": state.state,
            "temperature": attributes.get("temperature"),
            "humidity": attributes.get("humidity"),
            "pressure": attributes.get("pressure"),
            "wind_speed": attributes.get("wind_speed"),
            "wind_bearing": attributes.get("wind_bearing"),
            "cloud_coverage": attributes.get("cloud_coverage"),
            "visibility": attributes.get("visibility"),
            "ozone": attributes.get("ozone"),
            "uv_index": attributes.get("uv_index"),
        }
    
    def is_configured(self) -> bool:
        """Check if weather entity is configured by @Zara"""
        return self._weather_entity is not None
