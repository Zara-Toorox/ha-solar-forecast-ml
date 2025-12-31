# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant, State

_LOGGER = logging.getLogger(__name__)

class WeatherAPIClient:
    """Handles communication with weather entities and services"""

    def __init__(self, hass: HomeAssistant):
        """Initialize weather API client @zara"""
        self.hass = hass
        self._weather_entity: Optional[str] = None

    def set_weather_entity(self, entity_id: str) -> None:
        """Set the weather entity to use @zara"""
        self._weather_entity = entity_id
        _LOGGER.debug(f"Weather entity set to: {entity_id}")

    async def get_current_weather(self) -> Optional[Dict[str, Any]]:
        """Get current weather data from configured entity @zara"""
        if not self._weather_entity:
            _LOGGER.error("Weather entity not configured")
            return None

        state = self.hass.states.get(self._weather_entity)
        if not state:
            _LOGGER.error(f"Weather entity not found: {self._weather_entity}")
            return None

        return self._extract_weather_data(state)

    async def get_hourly_forecast(self) -> Optional[List[Dict[str, Any]]]:
        """Get hourly weather forecast from configured entity @zara"""
        if not self._weather_entity:
            _LOGGER.error("Weather entity not configured")
            return None

        try:

            response = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": self._weather_entity, "type": "hourly"},
                blocking=True,
                return_response=True,
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
        """Get daily weather forecast from configured entity @zara"""
        if not self._weather_entity:
            _LOGGER.error("Weather entity not configured")
            return None

        try:

            response = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": self._weather_entity, "type": "daily"},
                blocking=True,
                return_response=True,
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
        """Extract weather data from state object @zara"""
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
        """Check if weather entity is configured @zara"""
        return self._weather_entity is not None
