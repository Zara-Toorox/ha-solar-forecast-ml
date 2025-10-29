"""
Weather service for Solar Forecast ML integration.

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
# Version 4.2 - API Signature adapted (hass, weather_entity, error_handler)
# Bugfix: Prioritize cloud_cover attribute over condition mapping
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from homeassistant.core import HomeAssistant

from .exceptions import ConfigurationException, WeatherAPIException

_LOGGER = logging.getLogger(__name__)


# Default weather data as fallback
DEFAULT_WEATHER_DATA = {
    "temperature": 15.0,
    "humidity": 60.0,
    "cloud_cover": 50.0,
    "wind_speed": 3.0,
    "precipitation": 0.0,
    "pressure": 1013.25,
}


class WeatherService:
    """
    Weather Service for Home Assistant Weather Entities
    ✓ Cleaned: Only HA Weather Entities, no OpenWeatherMap
    # by Zara
    """
    
    def __init__(self, hass: HomeAssistant, weather_entity: str, error_handler=None):
        """Initialize weather service with explicit parameters # by Zara"""
        self.hass = hass
        self.weather_entity = weather_entity
        self.error_handler = error_handler
        
        # Validate on init
        self._validate_config()
    
    def _validate_config(self):
        """
        Validate Weather Service configuration
        
        Checks only the Home Assistant Weather Entity
        # by Zara
        """
        if not self.weather_entity:
            raise ConfigurationException("Weather Entity not configured")
        
        # Check if entity exists (can start later, so only Warning)
        state = self.hass.states.get(self.weather_entity)
        if state is None:
            _LOGGER.warning(
                f"Weather Entity {self.weather_entity} currently not available. "
                f"Will be loaded on startup."
            )
        
        _LOGGER.info(f"✓ Weather Service configured: {self.weather_entity}")
    
    
    async def initialize(self) -> bool:
        """
        Async initialization of the Weather Service
        
        Called by the ServiceManager
        # by Zara
        """
        try:
            # Check if entity is available
            state = self.hass.states.get(self.weather_entity)
            if state is None:
                _LOGGER.warning(f"Weather Entity {self.weather_entity} not yet available")
                return False
            
            if state.state in ["unavailable", "unknown"]:
                _LOGGER.warning(f"Weather Entity {self.weather_entity} is {state.state}")
                return False
            
            _LOGGER.info(f"✓ Weather Service initialized: {self.weather_entity}")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Weather Service initialization failed: {e}")
            return False

    async def get_current_weather(self) -> dict[str, Any]:
        """
        Get current weather data from Home Assistant Weather Entity
        
        Returns:
            Dict with weather data (temperature, humidity, cloud_cover, etc.)
        
        Raises:
            ConfigurationException: On Config errors
            WeatherAPIException: On API errors
        # by Zara
        """
        try:
            # Get data from HA Weather Entity
            return await self._get_ha_weather()
            
        except ConfigurationException as err:
            _LOGGER.error("Current Weather API failed (Config): %s", err)
            raise
            
        except WeatherAPIException as err:
            _LOGGER.error("Current Weather API failed (API): %s", err)
            raise
            
        except Exception as err:
            _LOGGER.error("Unexpected error in get_current_weather: %s", err)
            raise WeatherAPIException(f"Weather API error: {err}")
    
    async def _get_ha_weather(self) -> dict[str, Any]:
        """
        Get weather data from Home Assistant Weather Entity
        
        Extracts all relevant weather data from the entity
        # by Zara
        """
        if not self.weather_entity:
            raise ConfigurationException("Weather Entity not configured")
        
        state = self.hass.states.get(self.weather_entity)
        
        if state is None:
            raise WeatherAPIException(f"Weather Entity {self.weather_entity} not available")
        
        if state.state in ["unavailable", "unknown"]:
            raise WeatherAPIException(
                f"Weather Entity {self.weather_entity} is {state.state}"
            )
        
        # Extract weather data from State Attributes
        try:
            attributes = state.attributes
            
            weather_data = {
                "temperature": float(attributes.get("temperature", DEFAULT_WEATHER_DATA["temperature"])),
                "humidity": float(attributes.get("humidity", DEFAULT_WEATHER_DATA["humidity"])),
                "cloud_cover": self._extract_cloud_cover(attributes, state.state),
                "wind_speed": float(attributes.get("wind_speed", DEFAULT_WEATHER_DATA["wind_speed"])),
                "precipitation": self._extract_precipitation(attributes),
                "pressure": float(attributes.get("pressure", DEFAULT_WEATHER_DATA["pressure"])),
                "condition": state.state,
                "forecast": attributes.get("forecast", [])
            }
            
            _LOGGER.debug(
                f"✓ HA Weather data retrieved from {self.weather_entity}: "
                f"Temp={weather_data['temperature']}°C, "
                f"Clouds={weather_data['cloud_cover']}%, "
                f"Condition={weather_data['condition']}"
            )
            
            return weather_data
            
        except (ValueError, TypeError) as err:
            _LOGGER.error("Error parsing HA weather data: %s", err)
            raise WeatherAPIException(f"Failed to parse HA weather data: {err}")
    
    def _extract_cloud_cover(self, attributes: Dict[str, Any], condition: str) -> float:
        """
        Extracts cloud cover, prioritizing attributes over condition mapping.
        """
        # 1. Try 'cloud_cover' attribute
        cloud_cover = attributes.get("cloud_cover")
        if cloud_cover is not None:
            try:
                return float(cloud_cover)
            except (ValueError, TypeError):
                pass
        
        # 2. Try 'cloudiness' attribute (common alternative)
        cloudiness = attributes.get("cloudiness")
        if cloudiness is not None:
            try:
                return float(cloudiness)
            except (ValueError, TypeError):
                pass
                
        # 3. Fallback to mapping the condition string
        _LOGGER.debug(f"No 'cloud_cover' attribute found, falling back to mapping condition '{condition}'")
        return self._map_condition_to_cloud_cover(condition)

    def _map_condition_to_cloud_cover(self, condition: str) -> float:
        """
        Map HA Weather Condition to Cloud Cover Percent
        
        Args:
            condition: Weather condition (e.g. "sunny", "cloudy")
        
        Returns:
            Cloud cover in percent (0-100)
        # by Zara
        """
        condition_map = {
            "clear-night": 0.0,
            "sunny": 0.0,
            "partlycloudy": 40.0,
            "cloudy": 80.0,
            "fog": 100.0,
            "rainy": 90.0,
            "snowy": 90.0,
            "pouring": 100.0,
            "hail": 95.0,
            "lightning": 100.0,
            "lightning-rainy": 100.0,
            "windy": 30.0,
            "windy-variant": 30.0,
        }
        
        return condition_map.get(condition.lower(), 50.0)  # Default 50%
    
    def _extract_precipitation(self, attributes: dict[str, Any]) -> float:
        """
        Extract precipitation data from Attributes
        
        Home Assistant often doesn't provide direct precipitation data,
        so we try multiple attribute names.
        # by Zara
        """
        # Try various attributes
        precipitation_keys = [
            "precipitation",
            "precipitation_amount",
            "rain",
            "rainfall"
        ]
        
        for key in precipitation_keys:
            value = attributes.get(key)
            if value is not None:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    pass
        
        # Fallback: 0.0 (no precipitation)
        return 0.0
    
    async def get_forecast(self, hours: int = 24) -> list[dict[str, Any]]:
        """
        Get Forecast data from Home Assistant Weather Entity
        
        Args:
            hours: Number of hours for forecast
        
        Returns:
            List with forecast data per hour
        # by Zara
        """
        weather_entity = self.weather_entity
        
        if not weather_entity:
            raise ConfigurationException("Weather Entity not configured")
        
        state = self.hass.states.get(weather_entity)
        
        if state is None:
            raise WeatherAPIException(f"Weather Entity {weather_entity} not available")
        
        # Extract Forecast from Attributes
        attributes = state.attributes
        forecast_data = attributes.get("forecast", [])
        
        if not forecast_data:
            _LOGGER.warning(f"No forecast data available in {weather_entity}")
            return []
        
        # Limit to desired hours
        return forecast_data[:hours]
    
    def get_health_status(self) -> dict[str, Any]:
        """
        Check Health Status of the Weather Service
        
        Returns:
            Dict with health_status and details
        # by Zara
        """
        weather_entity = self.weather_entity
        
        if not weather_entity:
            return {
                "healthy": False,
                "status": "error",
                "message": "Weather Entity not configured"
            }
        
        state = self.hass.states.get(weather_entity)
        
        if state is None:
            return {
                "healthy": False,
                "status": "unavailable",
                "message": f"Weather Entity {weather_entity} not found"
            }
        
        if state.state in ["unavailable", "unknown"]:
            return {
                "healthy": False,
                "status": state.state,
                "message": f"Weather Entity {weather_entity} is {state.state}"
            }
        
        # Entity is healthy
        return {
            "healthy": True,
            "status": "ok",
            "message": f"Weather Entity {weather_entity} is available",
            "condition": state.state,
            "last_updated": state.last_updated.isoformat() if state.last_updated else None
        }
    
    def update_weather_entity(self, new_entity: str) -> None:
        """
        ✓ Update Weather Entity (e.g., on fallback)
        
        Args:
            new_entity: New Weather Entity ID
        # by Zara
        """
        old_entity = self.weather_entity
        self.weather_entity = new_entity
        
        _LOGGER.info(f"Weather Entity updated: {old_entity} -> {new_entity}")
        
        # Re-validate Config
        try:
            self._validate_config()
        except ConfigurationException as err:
            _LOGGER.error(f"Validation failed after entity update: {err}")