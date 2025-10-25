"""Weather service for Solar Forecast ML integration."""
# Version 4.1 - API-Signatur angepasst (hass, weather_entity, error_handler) # von Zara
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from homeassistant.core import HomeAssistant

from .exceptions import ConfigurationException, WeatherAPIException

_LOGGER = logging.getLogger(__name__)


# Default Wetterdaten als Fallback # von Zara
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
    Weather Service für Home Assistant Weather Entities
    ✓ Bereinigt: Nur HA Weather Entities, kein OpenWeatherMap
    # von Zara
    """
    
    def __init__(self, hass: HomeAssistant, weather_entity: str, error_handler=None):
        """Initialize weather service mit expliziten Parametern # von Zara"""
        self.hass = hass
        self.weather_entity = weather_entity
        self.error_handler = error_handler
        
        # Validiere beim Init # von Zara
        self._validate_config()
    
    def _validate_config(self):
        """
        Validiere Weather Service Konfiguration
        
        Prüft nur noch Home Assistant Weather Entity
        # von Zara
        """
        if not self.weather_entity:
            raise ConfigurationException("Weather Entity nicht konfiguriert")
        
        # Prüfe ob Entity existiert (kann später starten, daher nur Warning) # von Zara
        state = self.hass.states.get(self.weather_entity)
        if state is None:
            _LOGGER.warning(
                f"Weather Entity {self.weather_entity} aktuell nicht verfügbar. "
                f"Wird beim Start geladen."
            )
        
        _LOGGER.info(f"✓ Weather Service konfiguriert: {self.weather_entity}")
    
    
    async def initialize(self) -> bool:
        """
        Async Initialisierung des Weather Service
        
        Wird vom ServiceManager aufgerufen
        # von Zara
        """
        try:
            # Prüfe ob Entity verfügbar ist # von Zara
            state = self.hass.states.get(self.weather_entity)
            if state is None:
                _LOGGER.warning(f"Weather Entity {self.weather_entity} noch nicht verfügbar")
                return False
            
            if state.state in ["unavailable", "unknown"]:
                _LOGGER.warning(f"Weather Entity {self.weather_entity} ist {state.state}")
                return False
            
            _LOGGER.info(f"✓ Weather Service initialisiert: {self.weather_entity}")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Weather Service Initialisierung fehlgeschlagen: {e}")
            return False

    async def get_current_weather(self) -> dict[str, Any]:
        """
        Hole aktuelle Wetterdaten von Home Assistant Weather Entity
        
        Returns:
            Dict mit Wetterdaten (temperature, humidity, cloud_cover, etc.)
        
        Raises:
            ConfigurationException: Bei Config-Fehlern
            WeatherAPIException: Bei API-Fehlern
        # von Zara
        """
        try:
            # Hole Daten von HA Weather Entity # von Zara
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
        Hole Wetterdaten von Home Assistant Weather Entity
        
        Extrahiert alle relevanten Wetterdaten aus der Entity
        # von Zara
        """
        weather_entity = self.weather_entity
        
        if not self.weather_entity:
            raise ConfigurationException("Weather Entity nicht konfiguriert")
        
        state = self.hass.states.get(self.weather_entity)
        
        if state is None:
            raise WeatherAPIException(f"Weather Entity {self.weather_entity} nicht verfügbar")
        
        if state.state in ["unavailable", "unknown"]:
            raise WeatherAPIException(
                f"Weather Entity {self.weather_entity} ist {state.state}"
            )
        
        # Extrahiere Wetterdaten aus State Attributes # von Zara
        try:
            attributes = state.attributes
            
            weather_data = {
                "temperature": float(attributes.get("temperature", DEFAULT_WEATHER_DATA["temperature"])),
                "humidity": float(attributes.get("humidity", DEFAULT_WEATHER_DATA["humidity"])),
                "cloud_cover": self._map_condition_to_cloud_cover(state.state),
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
    
    def _map_condition_to_cloud_cover(self, condition: str) -> float:
        """
        Mappe HA Weather Condition zu Cloud Cover Prozent
        
        Args:
            condition: Weather condition (z.B. "sunny", "cloudy")
        
        Returns:
            Cloud cover in Prozent (0-100)
        # von Zara
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
        
        return condition_map.get(condition.lower(), 50.0)  # Default 50% # von Zara
    
    def _extract_precipitation(self, attributes: dict[str, Any]) -> float:
        """
        Extrahiere Niederschlagsdaten aus Attributes
        
        Home Assistant bietet oft keine direkten Niederschlagsdaten,
        daher versuchen wir mehrere Attribute-Namen
        # von Zara
        """
        # Versuche verschiedene Attribute # von Zara
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
        
        # Fallback: 0.0 (kein Niederschlag) # von Zara
        return 0.0
    
    async def get_forecast(self, hours: int = 24) -> list[dict[str, Any]]:
        """
        Hole Forecast-Daten von Home Assistant Weather Entity
        
        Args:
            hours: Anzahl Stunden für Forecast
        
        Returns:
            Liste mit Forecast-Daten pro Stunde
        # von Zara
        """
        weather_entity = self.weather_entity
        
        if not weather_entity:
            raise ConfigurationException("Weather Entity nicht konfiguriert")
        
        state = self.hass.states.get(weather_entity)
        
        if state is None:
            raise WeatherAPIException(f"Weather Entity {weather_entity} nicht verfügbar")
        
        # Extrahiere Forecast aus Attributes # von Zara
        attributes = state.attributes
        forecast_data = attributes.get("forecast", [])
        
        if not forecast_data:
            _LOGGER.warning(f"Keine Forecast-Daten in {weather_entity} verfügbar")
            return []
        
        # Limitiere auf gewünschte Stunden # von Zara
        return forecast_data[:hours]
    
    def get_health_status(self) -> dict[str, Any]:
        """
        Prüfe Health Status des Weather Service
        
        Returns:
            Dict mit health_status und details
        # von Zara
        """
        weather_entity = self.weather_entity
        
        if not weather_entity:
            return {
                "healthy": False,
                "status": "error",
                "message": "Weather Entity nicht konfiguriert"
            }
        
        state = self.hass.states.get(weather_entity)
        
        if state is None:
            return {
                "healthy": False,
                "status": "unavailable",
                "message": f"Weather Entity {weather_entity} nicht gefunden"
            }
        
        if state.state in ["unavailable", "unknown"]:
            return {
                "healthy": False,
                "status": state.state,
                "message": f"Weather Entity {weather_entity} ist {state.state}"
            }
        
        # Entity ist healthy # von Zara
        return {
            "healthy": True,
            "status": "ok",
            "message": f"Weather Entity {weather_entity} ist verfügbar",
            "condition": state.state,
            "last_updated": state.last_updated.isoformat() if state.last_updated else None
        }
    
    def update_weather_entity(self, new_entity: str) -> None:
        """
        ✓ Update Weather Entity (z.B. bei Fallback)
        
        Args:
            new_entity: Neue Weather Entity ID
        # von Zara
        """
        old_entity = self.weather_entity
        self.weather_entity = new_entity
        
        _LOGGER.info(f"Weather Entity aktualisiert: {old_entity} → {new_entity}")
        
        # Re-validiere Config # von Zara
        try:
            self._validate_config()
        except ConfigurationException as err:
            _LOGGER.error(f"Validation failed nach Entity-Update: {err}")
