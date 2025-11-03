"""
Weather service for Solar Forecast ML integration.
Fetches and processes data from Home Assistant weather entities.
NON-BLOCKING DESIGN: Uses cached forecast from file if live data unavailable.

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

import asyncio
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from ..exceptions import ConfigurationException, WeatherAPIException
from ..core.helpers import SafeDateTimeUtil as dt_util

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
    Weather Service with file-based caching.
    NEVER blocks startup - uses cached data if live unavailable.
    """

    def __init__(
        self, 
        hass: HomeAssistant, 
        weather_entity: str, 
        data_manager=None,
        error_handler=None
    ):
        """Initialize weather service."""
        self.hass = hass
        self.weather_entity = weather_entity
        self.data_manager = data_manager
        self.error_handler = error_handler

        self._cached_forecast: List[Dict[str, Any]] = []
        self._background_update_task: asyncio.Task | None = None
        self._weather_entity_listener = None  # Event listener for weather entity state changes
        
        # Validate on init
        self._validate_config()

    def _validate_config(self):
        """Validate Weather Service configuration."""
        if not self.weather_entity:
            raise ConfigurationException("Weather Entity not configured")

        state = self.hass.states.get(self.weather_entity)
        if state is None:
            _LOGGER.warning(
                f"Weather Entity {self.weather_entity} currently not available. "
                f"Will use cached forecast if available."
            )
        else:
            _LOGGER.info(f"Weather Service configured with entity: {self.weather_entity}")

    async def initialize(self) -> bool:
        """
        Async initialization - loads cached forecast immediately.
        Uses dummy forecast if no cache available.
        Sets up event listener for weather entity state changes.
        Does NOT wait for live data.
        """
        try:
            # Load cached forecast from file (instant)
            cache_loaded = False
            if self.data_manager:
                cached = await self.data_manager.load_weather_forecast_cache()
                if cached:
                    self._cached_forecast = cached.get("forecast_hours", [])
                    quality = cached.get("data_quality", {})
                    _LOGGER.info(
                        f"Loaded {len(self._cached_forecast)} hours from weather cache "
                        f"(today: {quality.get('today_hours', 0)}h, "
                        f"tomorrow: {quality.get('tomorrow_hours', 0)}h)"
                    )
                    cache_loaded = True
                else:
                    _LOGGER.warning("No cached forecast available")
            
            # Generate dummy forecast if no cache available
            if not cache_loaded or not self._cached_forecast:
                _LOGGER.warning("Generating dummy forecast (48h) to ensure integration starts")
                self._cached_forecast = self._generate_dummy_forecast()
                
                # Save dummy forecast to cache immediately with metadata wrapper
                if self.data_manager:
                    cache_wrapper = {
                        "forecast_hours": self._cached_forecast,
                        "cached_at": dt_util.now().isoformat(),  # LOCAL time
                        "data_quality": {
                            "today_hours": 24,
                            "tomorrow_hours": 24,
                            "total_hours": len(self._cached_forecast)
                        },
                        "is_dummy": True
                    }
                    await self.data_manager.save_weather_forecast_cache(cache_wrapper)
                    _LOGGER.info(f"Saved dummy forecast to cache: {len(self._cached_forecast)} hours")
            
            # Setup event listener for weather entity state changes
            self._setup_weather_entity_listener()
            
            # Start background update (non-blocking)
            self._background_update_task = asyncio.create_task(
                self._background_forecast_update()
            )
            
            _LOGGER.info("Weather Service initialized (non-blocking)")
            return True

        except Exception as e:
            _LOGGER.error(f"Weather Service initialization failed: {e}", exc_info=True)
            return False

    async def try_get_forecast(self, timeout: int = 5) -> List[Dict[str, Any]]:
        """
        Try to fetch live forecast with timeout.
        Returns empty list on failure - caller handles fallback.
        
        Args:
            timeout: Max seconds to wait for live data
            
        Returns:
            List of processed forecast hours or empty list
        """
        try:
            result = await asyncio.wait_for(
                self._fetch_and_process_forecast(),
                timeout=timeout
            )
            
            if result:
                # Save to cache immediately with metadata wrapper
                if self.data_manager:
                    # Calculate data quality metrics
                    now_local = dt_util.now()
                    today_hours = sum(1 for h in result if dt_util.parse_datetime(h["datetime"]).date() == now_local.date())
                    tomorrow_hours = sum(1 for h in result if dt_util.parse_datetime(h["datetime"]).date() == (now_local + timedelta(days=1)).date())
                    
                    cache_wrapper = {
                        "forecast_hours": result,
                        "cached_at": now_local.isoformat(),  # LOCAL time
                        "data_quality": {
                            "today_hours": today_hours,
                            "tomorrow_hours": tomorrow_hours,
                            "total_hours": len(result)
                        }
                    }
                    await self.data_manager.save_weather_forecast_cache(cache_wrapper)
                    _LOGGER.debug(f"Saved {len(result)} forecast hours to cache (today: {today_hours}h, tomorrow: {tomorrow_hours}h)")
                
                self._cached_forecast = result
                return result
            else:
                _LOGGER.debug("Forecast fetch returned empty result")
                return []
                
        except asyncio.TimeoutError:
            _LOGGER.warning(f"Forecast fetch timeout after {timeout}s - using cached data")
            return []
        except Exception as e:
            _LOGGER.warning(f"Forecast fetch failed: {e} - using cached data")
            return []

    async def get_processed_hourly_forecast(self) -> List[Dict[str, Any]]:
        """
        Get forecast - returns cached immediately, updates in background.
        NEVER blocks.
        """
        # Return cached data immediately
        if self._cached_forecast:
            _LOGGER.debug(f"Returning cached forecast ({len(self._cached_forecast)} hours)")
            return self._cached_forecast
        
        # Try quick fetch if no cache
        fresh = await self.try_get_forecast(timeout=3)
        if fresh:
            return fresh
        
        # Load from file as last resort
        if self.data_manager:
            cached = await self.data_manager.load_weather_forecast_cache()
            if cached:
                forecast = cached.get("forecast_hours", [])
                self._cached_forecast = forecast
                _LOGGER.info(f"Loaded {len(forecast)} hours from file cache")
                return forecast
        
        _LOGGER.warning("No forecast available - returning empty list")
        return []

    async def _fetch_and_process_forecast(self) -> List[Dict[str, Any]]:
        """Fetch raw forecast and process it."""
        raw_forecast = await self.get_forecast(hours=48)
        if not raw_forecast:
            return []

        processed: List[Dict[str, Any]] = []
        
        for entry in raw_forecast:
            if not isinstance(entry, dict):
                continue

            timestamp_str = entry.get("datetime")
            if not timestamp_str:
                continue

            timestamp = dt_util.parse_datetime(timestamp_str)
            if not timestamp:
                continue

            local_dt = dt_util.as_local(timestamp)
            temp_value = entry.get("temperature")
            if temp_value is None:
                continue

            try:
                processed_hour = {
                    "datetime": timestamp.isoformat(),
                    "local_hour": local_dt.hour,
                    "local_datetime": local_dt.isoformat(),
                    "temperature": float(temp_value),
                    "humidity": float(entry.get("humidity", DEFAULT_WEATHER_DATA["humidity"])),
                    "cloud_cover": self._extract_cloud_cover(entry, entry.get("condition")),
                    "wind_speed": float(entry.get("wind_speed", DEFAULT_WEATHER_DATA["wind_speed"])),
                    "precipitation": self._extract_precipitation(entry),
                    "pressure": float(entry.get("pressure", DEFAULT_WEATHER_DATA["pressure"])),
                    "condition": entry.get("condition")
                }
                processed.append(processed_hour)
            except Exception as e:
                _LOGGER.debug(f"Failed to process forecast entry: {e}")
                continue

        _LOGGER.info(f"Processed {len(processed)} forecast hours")
        return processed

    async def _background_forecast_update(self):
        """
        Background task: Periodically update forecast cache.
        Runs silently, never blocks anything.
        """
        update_interval = 1800  # 30 minutes
        
        while True:
            try:
                await asyncio.sleep(update_interval)
                
                _LOGGER.debug("Background forecast update starting...")
                forecast = await self.try_get_forecast(timeout=30)
                
                if forecast:
                    _LOGGER.info(f"Background update: Cached {len(forecast)} forecast hours")
                else:
                    _LOGGER.debug("Background update: No new forecast data")
                    
            except asyncio.CancelledError:
                _LOGGER.info("Background forecast update task cancelled")
                break
            except Exception as e:
                _LOGGER.error(f"Background forecast update error: {e}", exc_info=True)
                # Continue running despite errors

    async def get_current_weather(self) -> dict[str, Any]:
        """Get current weather data from Home Assistant Weather Entity."""
        try:
            return await self._get_ha_weather()
        except ConfigurationException as err:
            _LOGGER.error("Current Weather retrieval failed (Config): %s", err)
            raise
        except WeatherAPIException as err:
            _LOGGER.error("Current Weather retrieval failed (API/Data): %s", err)
            raise
        except Exception as err:
            _LOGGER.error("Unexpected error in get_current_weather: %s", err, exc_info=True)
            raise WeatherAPIException(f"Unexpected weather retrieval error: {err}")

    async def _get_ha_weather(self) -> dict[str, Any]:
        """Get weather data from Home Assistant Weather Entity."""
        if not self.weather_entity:
            raise ConfigurationException("Weather Entity not configured")

        state = self.hass.states.get(self.weather_entity)
        if state is None:
            raise WeatherAPIException(f"Weather Entity {self.weather_entity} not found")

        if state.state in ["unavailable", "unknown"]:
            raise WeatherAPIException(f"Weather Entity {self.weather_entity} state is {state.state}")

        attributes = state.attributes
        if not attributes:
            raise WeatherAPIException(f"Weather Entity {self.weather_entity} has no attributes")

        temp_value = attributes.get("temperature")
        if temp_value is None:
            _LOGGER.warning(f"Weather Entity {self.weather_entity} missing 'temperature' (loading?)")
            raise WeatherAPIException("Missing temperature in weather entity")

        weather_data = {
            "temperature": float(temp_value),
            "humidity": float(attributes.get("humidity", DEFAULT_WEATHER_DATA["humidity"])),
            "cloud_cover": self._extract_cloud_cover(attributes, state.state),
            "wind_speed": float(attributes.get("wind_speed", DEFAULT_WEATHER_DATA["wind_speed"])),
            "precipitation": self._extract_precipitation(attributes),
            "pressure": float(attributes.get("pressure", DEFAULT_WEATHER_DATA["pressure"])),
            "condition": state.state,
        }

        _LOGGER.debug(
            f"Current weather: {weather_data['temperature']}°C, "
            f"{weather_data['cloud_cover']}% clouds, {weather_data['condition']}"
        )
        return weather_data

    def _extract_cloud_cover(self, attributes: Dict[str, Any], condition: str) -> float:
        """Extract cloud cover from attributes or map from condition."""
        for key in ["cloud_coverage", "cloudiness"]:
            value = attributes.get(key)
            if value is not None:
                try:
                    return max(0.0, min(100.0, float(value)))
                except (ValueError, TypeError):
                    pass

        _LOGGER.debug(f"No cloud attribute, mapping condition '{condition}'")
        return max(0.0, min(100.0, self._map_condition_to_cloud_cover(condition)))

    def _map_condition_to_cloud_cover(self, condition: str | None) -> float:
        """Map weather condition to cloud cover percentage."""
        if not condition:
            return DEFAULT_WEATHER_DATA["cloud_cover"]

        condition_lower = condition.lower()
        condition_map = {
            "clear-night": 0.0,
            "sunny": 5.0,
            "partlycloudy": 40.0,
            "cloudy": 80.0,
            "overcast": 100.0,
            "fog": 95.0,
            "hail": 90.0,
            "lightning": 70.0,
            "lightning-rainy": 95.0,
            "pouring": 100.0,
            "rainy": 90.0,
            "snowy": 95.0,
            "snowy-rainy": 95.0,
            "windy": 30.0,
            "windy-variant": 30.0,
            "exceptional": 50.0
        }
        return condition_map.get(condition_lower, DEFAULT_WEATHER_DATA["cloud_cover"])

    def _extract_precipitation(self, attributes: dict[str, Any]) -> float:
        """Extract precipitation from various keys."""
        keys = [
            "precipitation", "precipitation_intensity", "precipitation_amount",
            "rain", "rainfall", "snow", "snowfall"
        ]
        for key in keys:
            value = attributes.get(key)
            if value is not None:
                try:
                    precip_val = float(value)
                    return max(0.0, precip_val)
                except (ValueError, TypeError):
                    pass
        return DEFAULT_WEATHER_DATA["precipitation"]

    async def get_forecast(self, hours: int = 24) -> list[dict[str, Any]]:
        """
        Get forecast using weather.get_forecasts service (HA 2024.3+).
        Falls back to attribute access for older versions.
        """
        if not self.weather_entity:
            raise ConfigurationException("Weather Entity not configured")

        state = self.hass.states.get(self.weather_entity)
        if not state or state.state in ["unavailable", "unknown"]:
            _LOGGER.debug(f"Forecast skipped: {self.weather_entity} is {state.state if state else 'missing'}")
            return []

        # Method 1: Service Call (HA 2024.3+, preferred)
        try:
            response = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": self.weather_entity, "type": "hourly"},
                blocking=True,
                return_response=True
            )
            
            if response and isinstance(response, dict):
                forecast_data = response.get(self.weather_entity, {}).get("forecast", [])
                if forecast_data and isinstance(forecast_data, list):
                    result = forecast_data[:hours]
                    _LOGGER.debug(f"Loaded {len(result)} forecast entries via service call")
                    return result
                    
        except Exception as e:
            _LOGGER.debug(f"Service call failed (HA < 2024.3?): {e}, trying attribute access")

        # Method 2: Attribute access (fallback for older HA versions)
        attributes = state.attributes
        if not attributes:
            _LOGGER.debug(f"No attributes for {self.weather_entity}")
            return []

        forecast_data = None
        source = None
        for key in ["hourly_forecast", "forecast", "forecast_hourly"]:
            data = attributes.get(key)
            if data and isinstance(data, list) and data:
                forecast_data = data
                source = key
                break

        if not forecast_data:
            _LOGGER.warning(
                f"No forecast data in {self.weather_entity}. "
                f"Available keys: {list(attributes.keys())}"
            )
            return []

        result = forecast_data[:hours]
        _LOGGER.debug(f"Loaded {len(result)} forecast entries from attribute '{source}'")
        return result

    def get_health_status(self) -> dict[str, Any]:
        """Check health status of the weather service."""
        if not self.weather_entity:
            return {"healthy": False, "status": "error", "message": "Weather Entity not configured"}

        state = self.hass.states.get(self.weather_entity)
        if state is None:
            # Still healthy if we have cached data
            has_cache = bool(self._cached_forecast)
            return {
                "healthy": has_cache,
                "status": "cached" if has_cache else "error",
                "message": f"Entity not found, using cache" if has_cache else "Entity not found"
            }

        if state.state in ["unavailable", "unknown"]:
            has_cache = bool(self._cached_forecast)
            return {
                "healthy": has_cache,
                "status": "cached" if has_cache else state.state,
                "message": f"Entity {state.state}, using cache" if has_cache else f"Entity is {state.state}"
            }

        attributes = state.attributes
        if not attributes or attributes.get("temperature") is None:
            return {
                "healthy": False,
                "status": "degraded",
                "message": "Missing temperature or attributes"
            }

        return {
            "healthy": True,
            "status": "ok",
            "message": "Weather entity is available",
            "condition": state.state,
            "cached_hours": len(self._cached_forecast),
            "last_updated": state.last_updated.isoformat() if state.last_updated else None
        }

    def update_weather_entity(self, new_entity: str) -> None:
        """Update the weather entity ID."""
        old_entity = self.weather_entity
        self.weather_entity = new_entity
        _LOGGER.info(f"Weather Entity updated: {old_entity} -> {new_entity}")
        try:
            self._validate_config()
        except ConfigurationException as err:
            _LOGGER.error(f"Validation failed after update: {err}")

    def _generate_dummy_forecast(self) -> List[Dict[str, Any]]:
        """
        Generate dummy forecast data for 48 hours (today + tomorrow).
        Uses LOCAL time for all datetime fields.
        Ensures integration can start even without weather data.
        
        Returns:
            List of 48 hourly forecast entries with standard weather values
        """
        now_local = dt_util.now()  # Current time in LOCAL timezone
        dummy_forecast = []
        
        for hour_offset in range(48):
            forecast_time_local = now_local + timedelta(hours=hour_offset)
            forecast_time_utc = dt_util.as_utc(forecast_time_local)
            
            # Create dummy entry with both UTC and local datetime
            dummy_entry = {
                "datetime": forecast_time_utc.isoformat(),  # UTC for deduplication
                "local_hour": forecast_time_local.hour,
                "local_datetime": forecast_time_local.isoformat(),  # LOCAL datetime as ISO string
                "temperature": DEFAULT_WEATHER_DATA["temperature"],
                "humidity": DEFAULT_WEATHER_DATA["humidity"],
                "cloud_cover": DEFAULT_WEATHER_DATA["cloud_cover"],
                "wind_speed": DEFAULT_WEATHER_DATA["wind_speed"],
                "precipitation": DEFAULT_WEATHER_DATA["precipitation"],
                "pressure": DEFAULT_WEATHER_DATA["pressure"],
                "condition": "partly-cloudy"
            }
            dummy_forecast.append(dummy_entry)
        
        _LOGGER.debug(f"Generated dummy forecast: {len(dummy_forecast)} hours")
        return dummy_forecast

    def _setup_weather_entity_listener(self) -> None:
        """
        Setup event listener for weather entity state changes.
        Triggers immediate forecast update when weather becomes available.
        """
        from homeassistant.core import callback
        from homeassistant.helpers.event import async_track_state_change_event
        
        @callback
        def weather_state_changed(event):
            """Handle weather entity state change."""
            new_state = event.data.get("new_state")
            old_state = event.data.get("old_state")
            
            if new_state is None:
                return
            
            # Check if weather became available
            if old_state and old_state.state in ["unavailable", "unknown"]:
                if new_state.state not in ["unavailable", "unknown"]:
                    _LOGGER.info(
                        f"Weather entity {self.weather_entity} became available, "
                        f"triggering immediate forecast update"
                    )
                    # Schedule immediate update (non-blocking)
                    asyncio.create_task(self._immediate_forecast_update())
            
            # Also update if attributes change significantly (new forecast data)
            elif old_state and new_state.attributes != old_state.attributes:
                forecast_keys = ["forecast", "hourly_forecast", "forecast_hourly"]
                for key in forecast_keys:
                    if key in new_state.attributes and key in old_state.attributes:
                        if new_state.attributes[key] != old_state.attributes[key]:
                            _LOGGER.debug(
                                f"Weather entity {self.weather_entity} forecast updated, "
                                f"refreshing cache"
                            )
                            asyncio.create_task(self._immediate_forecast_update())
                            break
        
        # Register listener
        self._weather_entity_listener = async_track_state_change_event(
            self.hass,
            [self.weather_entity],
            weather_state_changed
        )
        
        _LOGGER.info(f"Weather entity listener registered for {self.weather_entity}")

    async def force_update(self) -> bool:
        """
        PUBLIC: Force immediate forecast update (blocking).
        Used by manual refresh buttons and force-update operations.
        
        Returns:
            bool: True if fresh forecast retrieved, False otherwise
        """
        try:
            _LOGGER.info("Force update requested - fetching fresh forecast...")
            
            # Try to fetch fresh forecast with extended timeout
            fresh_forecast = await self.try_get_forecast(timeout=15)
            
            if fresh_forecast:
                _LOGGER.info(
                    f"Force update successful: {len(fresh_forecast)} hours retrieved"
                )
                return True
            else:
                _LOGGER.warning("Force update returned no data - check weather entity")
                return False
                
        except Exception as e:
            _LOGGER.error(f"Force update failed: {e}", exc_info=True)
            return False

    async def _immediate_forecast_update(self) -> None:
        """
        Perform immediate forecast update when weather becomes available.
        Non-blocking, logs errors but doesn't raise.
        """
        try:
            _LOGGER.debug("Starting immediate forecast update...")
            
            # Try to fetch fresh forecast with short timeout
            fresh_forecast = await self.try_get_forecast(timeout=10)
            
            if fresh_forecast:
                _LOGGER.info(
                    f"Immediate forecast update successful: {len(fresh_forecast)} hours retrieved"
                )
                # Cache is automatically updated in try_get_forecast
            else:
                _LOGGER.debug("Immediate forecast update returned no data")
                
        except Exception as e:
            _LOGGER.warning(f"Immediate forecast update failed: {e}", exc_info=True)

    async def cleanup(self):
        """Cleanup resources."""
        # Remove event listener
        if self._weather_entity_listener:
            self._weather_entity_listener()
            self._weather_entity_listener = None
            _LOGGER.debug("Weather entity listener removed")
        
        # Cancel background task
        if self._background_update_task and not self._background_update_task.done():
            self._background_update_task.cancel()
            try:
                await self._background_update_task
            except asyncio.CancelledError:
                pass
        
        _LOGGER.debug("Weather Service cleanup complete")