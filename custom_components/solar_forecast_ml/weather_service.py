"""
Weather service for Solar Forecast ML integration.
Fetches and processes data from Home Assistant weather entities.

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
from datetime import datetime, timedelta
from typing import Any, Dict

from homeassistant.core import HomeAssistant

from .exceptions import ConfigurationException, WeatherAPIException

_LOGGER = logging.getLogger(__name__)


# Default weather data as fallback ONLY if parsing fails, not if entity is loading
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
    Weather Service using Home Assistant Weather Entities.
    """

    def __init__(self, hass: HomeAssistant, weather_entity: str, error_handler=None):
        """Initialize weather service."""
        self.hass = hass
        self.weather_entity = weather_entity
        self.error_handler = error_handler

        # Validate on init
        self._validate_config()

    def _validate_config(self):
        """Validate Weather Service configuration."""
        if not self.weather_entity:
            raise ConfigurationException("Weather Entity not configured")

        # Check if entity exists (can start later, so only Warning)
        state = self.hass.states.get(self.weather_entity)
        if state is None:
            _LOGGER.warning(
                f"Weather Entity {self.weather_entity} currently not available. "
                f"Will try again during initialization."
            )

        _LOGGER.info(f"Weather Service configured with entity: {self.weather_entity}")


    async def initialize(self) -> bool:
        """Async initialization of the Weather Service."""
        try:
            # Check if entity is available now
            state = self.hass.states.get(self.weather_entity)
            if state is None:
                _LOGGER.warning(f"Weather Entity {self.weather_entity} still not available during init.")
                return False # Cannot initialize without the entity

            if state.state in ["unavailable", "unknown"]:
                _LOGGER.warning(f"Weather Entity {self.weather_entity} is {state.state} during init.")
                # Allow initialization but service might fail later
                return True # Technically initialized, even if entity state is bad

            _LOGGER.info(f"Weather Service initialized: {self.weather_entity} is available.")
            return True

        except Exception as e:
            _LOGGER.error(f"Weather Service initialization failed: {e}")
            return False

    async def get_current_weather(self) -> dict[str, Any]:
        """
        Get current weather data from Home Assistant Weather Entity.
        [FIX] Raises error if essential attributes are missing.

        Returns:
            Dict with weather data (temperature, humidity, cloud_cover, etc.)

        Raises:
            ConfigurationException: On Config errors
            WeatherAPIException: On API errors or missing attributes
        """
        try:
            # Get data from HA Weather Entity
            return await self._get_ha_weather()

        except ConfigurationException as err:
            _LOGGER.error("Current Weather retrieval failed (Config): %s", err)
            raise

        except WeatherAPIException as err:
            _LOGGER.error("Current Weather retrieval failed (API/Data): %s", err)
            raise # Re-raise to trigger UpdateFailed in coordinator

        except Exception as err:
            _LOGGER.error("Unexpected error in get_current_weather: %s", err, exc_info=True)
            # Wrap unexpected errors as WeatherAPIException
            raise WeatherAPIException(f"Unexpected weather retrieval error: {err}")

    async def _get_ha_weather(self) -> dict[str, Any]:
        """
        Get weather data from Home Assistant Weather Entity.
        [FIX] Validates essential attributes before returning data.
        """
        if not self.weather_entity:
            raise ConfigurationException("Weather Entity not configured")

        state = self.hass.states.get(self.weather_entity)

        if state is None:
            # This should ideally be caught during init, but check again
            raise WeatherAPIException(f"Weather Entity {self.weather_entity} not found")

        if state.state in ["unavailable", "unknown"]:
            raise WeatherAPIException(
                f"Weather Entity {self.weather_entity} state is {state.state}"
            )

        # Extract weather data from State Attributes
        try:
            attributes = state.attributes
            if not attributes:
                 raise WeatherAPIException(f"Weather Entity {self.weather_entity} has no attributes")


            # === START VALIDATION PATCH ===
            # Check if the 'temperature' attribute is present and not None.
            # If missing, assume the entity is still loading its data.
            temp_value = attributes.get("temperature")
            if temp_value is None:
                _LOGGER.warning(
                    f"Weather Entity {self.weather_entity} is available, "
                    f"but its 'temperature' attribute is missing (still loading?). "
                    f"Deferring update."
                )
                # Raise exception to stop _async_update_data in coordinator
                # This causes UpdateFailed, triggering a retry after a delay.
                raise WeatherAPIException(
                    f"Weather Entity attributes not populated (missing 'temperature')"
                )
            # === END VALIDATION PATCH ===

            # Proceed only if temperature is present
            weather_data = {
                "temperature": float(temp_value), # Use the validated value
                "humidity": float(attributes.get("humidity", DEFAULT_WEATHER_DATA["humidity"])),
                "cloud_cover": self._extract_cloud_cover(attributes, state.state),
                "wind_speed": float(attributes.get("wind_speed", DEFAULT_WEATHER_DATA["wind_speed"])),
                "precipitation": self._extract_precipitation(attributes),
                "pressure": float(attributes.get("pressure", DEFAULT_WEATHER_DATA["pressure"])),
                "condition": state.state, # Keep the original condition string
                "forecast": attributes.get("forecast", []) # Include forecast if available
            }

            _LOGGER.debug(
                f"HA Weather data retrieved from {self.weather_entity}: "
                f"Temp={weather_data['temperature']}°C, "
                f"Clouds={weather_data['cloud_cover']}%, "
                f"Condition={weather_data['condition']}"
            )

            return weather_data

        except (ValueError, TypeError) as err:
            _LOGGER.error(f"Error parsing HA weather data from {self.weather_entity}: {err}")
            # Raise specific error for parsing issues
            raise WeatherAPIException(f"Failed to parse HA weather data attributes: {err}")
        # Let WeatherAPIException from validation patch pass through

    def _extract_cloud_cover(self, attributes: Dict[str, Any], condition: str) -> float:
        """
        Extracts cloud cover, prioritizing attributes over condition mapping.
        Returns a value between 0.0 and 100.0.
        """
        # 1. Try 'cloud_coverage' attribute (official HA attribute)
        cloud_coverage = attributes.get("cloud_coverage")
        if cloud_coverage is not None:
            try:
                # Ensure value is clamped between 0 and 100
                return max(0.0, min(100.0, float(cloud_coverage)))
            except (ValueError, TypeError):
                _LOGGER.debug(f"Could not parse 'cloud_coverage': {cloud_coverage}")
                pass # Try next attribute

        # 2. Try 'cloudiness' attribute (common alternative)
        cloudiness = attributes.get("cloudiness")
        if cloudiness is not None:
            try:
                return max(0.0, min(100.0, float(cloudiness)))
            except (ValueError, TypeError):
                _LOGGER.debug(f"Could not parse 'cloudiness': {cloudiness}")
                pass # Fallback to mapping

        # 3. Fallback to mapping the condition string
        _LOGGER.debug(f"No valid cloud cover attribute found in {self.weather_entity}, "
                      f"falling back to mapping condition '{condition}'")
        mapped_value = self._map_condition_to_cloud_cover(condition)
        return max(0.0, min(100.0, mapped_value)) # Ensure mapped value is also clamped

    def _map_condition_to_cloud_cover(self, condition: str | None) -> float:
        """Map HA Weather Condition to approximate Cloud Cover Percent."""
        if not condition:
            return DEFAULT_WEATHER_DATA["cloud_cover"] # Default if condition is None or empty

        condition_lower = condition.lower()

        # More granular mapping
        condition_map = {
            "clear-night": 0.0,
            "sunny": 5.0, # Slightly > 0 for sunny
            "partlycloudy": 40.0,
            "cloudy": 80.0,
            "overcast": 100.0, # Add overcast
            "fog": 95.0, # Fog usually implies high cloud cover
            "hail": 90.0,
            "lightning": 70.0, # Often brief intense clouds
            "lightning-rainy": 95.0,
            "pouring": 100.0,
            "rainy": 90.0,
            "snowy": 95.0,
            "snowy-rainy": 95.0,
            "windy": 30.0, # Wind itself doesn't mean clouds
            "windy-variant": 30.0,
            "exceptional": 50.0 # Unknown, guess average
        }

        # Use get with a default
        return condition_map.get(condition_lower, DEFAULT_WEATHER_DATA["cloud_cover"])

    def _extract_precipitation(self, attributes: dict[str, Any]) -> float:
        """Extract precipitation data from Attributes (e.g., mm or inches)."""
        # Try various common attributes
        precipitation_keys = [
            "precipitation",           # Generic
            "precipitation_intensity", # Some integrations use this
            "precipitation_amount",
            "rain",
            "rainfall",
            "snow",
            "snowfall"
        ]

        for key in precipitation_keys:
            value = attributes.get(key)
            if value is not None:
                try:
                    # Return the first valid non-negative float found
                    precip_float = float(value)
                    return max(0.0, precip_float)
                except (ValueError, TypeError):
                    _LOGGER.debug(f"Could not parse precipitation key '{key}': {value}")
                    pass # Try the next key

        # Fallback: 0.0 if no valid precipitation attribute found
        return 0.0

    async def get_forecast(self, hours: int = 24) -> list[dict[str, Any]]:
        """
        Get Forecast data from Home Assistant Weather Entity's attributes.
        Limited to the forecast data provided by the underlying weather integration.
        """
        if not self.weather_entity:
            raise ConfigurationException("Weather Entity not configured")

        state = self.hass.states.get(self.weather_entity)

        if state is None or state.state in ["unavailable", "unknown"]:
            _LOGGER.warning(f"Cannot get forecast, Weather Entity {self.weather_entity} is unavailable")
            return [] # Return empty list if entity is unavailable

        # Extract Forecast list from Attributes
        attributes = state.attributes
        if not attributes:
            _LOGGER.warning(f"No attributes found for {self.weather_entity}, cannot get forecast.")
            return []

        forecast_data = attributes.get("forecast", [])

        if not isinstance(forecast_data, list):
             _LOGGER.warning(f"Forecast data in {self.weather_entity} is not a list: {type(forecast_data)}")
             return []

        if not forecast_data:
            _LOGGER.warning(f"No forecast data found in attributes of {self.weather_entity}")
            return []

        # Limit to the requested number of hours/entries if necessary
        limited_forecast = forecast_data[:hours]
        _LOGGER.debug(f"Retrieved {len(limited_forecast)} forecast entries from {self.weather_entity}")
        return limited_forecast

    def get_health_status(self) -> dict[str, Any]:
        """Check Health Status of the Weather Service."""
        if not self.weather_entity:
            return {
                "healthy": False,
                "status": "error",
                "message": "Weather Entity not configured"
            }

        state = self.hass.states.get(self.weather_entity)

        if state is None:
            return {
                "healthy": False,
                "status": "error",
                "message": f"Weather Entity {self.weather_entity} not found"
            }

        if state.state in ["unavailable", "unknown"]:
            return {
                "healthy": False,
                "status": state.state,
                "message": f"Weather Entity {self.weather_entity} is {state.state}"
            }

        # Check for essential attributes
        attributes = state.attributes
        if not attributes or attributes.get("temperature") is None:
             return {
                "healthy": False,
                "status": "degraded",
                "message": f"Weather Entity {self.weather_entity} is missing essential attributes (e.g., temperature)"
            }


        # If we reach here, the entity seems healthy
        return {
            "healthy": True,
            "status": "ok",
            "message": f"Weather Entity {self.weather_entity} is available and has attributes",
            "condition": state.state,
            "last_updated": state.last_updated.isoformat() if state.last_updated else None
        }

    def update_weather_entity(self, new_entity: str) -> None:
        """Update the Weather Entity ID used by the service."""
        old_entity = self.weather_entity
        self.weather_entity = new_entity

        _LOGGER.info(f"Weather Entity updated: {old_entity} -> {new_entity}")

        # Re-validate Config after update
        try:
            self._validate_config()
        except ConfigurationException as err:
            _LOGGER.error(f"Validation failed after entity update to {new_entity}: {err}")
            # Consider reverting or handling the error state appropriately