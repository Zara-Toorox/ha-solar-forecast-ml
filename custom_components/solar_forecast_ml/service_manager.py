"""
Service Manager for Solar Forecast ML.
Manages the lifecycle of all services (ML, Weather, Notification, Error Handler).
FIX: set_entities is called after ML-Init
Version 4.9.3 - solar_yield_today Fix

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
along with this program.  If not, see <https://www.gnu.org/licenses/>

Copyright (C) 2025 Zara-Toorox
"""
import asyncio
import logging
from typing import Optional, Any
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


class ServiceManager:
    """
    Manages the lifecycle and initialization of all services.
    Encapsulates service management logic from the Coordinator.
    
    PATCH: Improved service initialization and validation
    """
    
    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        data_manager,
        weather_entity: str,
        dependencies_ok: bool = False,
        power_entity: Optional[str] = None,
        solar_yield_today: Optional[str] = None,
        solar_capacity: float = 5.0,
        temp_sensor: Optional[str] = None,
        wind_sensor: Optional[str] = None,
        rain_sensor: Optional[str] = None,
        uv_sensor: Optional[str] = None,
        lux_sensor: Optional[str] = None,
    ):
        """
        Initialize Service Manager.
        
        Args:
            hass: HomeAssistant instance
            entry: ConfigEntry
            data_manager: DataManager instance
            weather_entity: Weather Entity ID
            dependencies_ok: True if dependencies are present
            power_entity: Power Entity ID
            solar_yield_today: Solar Yield Today Entity ID
            solar_capacity: Solar Capacity in kWp
        """
        self.hass = hass
        self.entry = entry
        self.data_manager = data_manager
        self.weather_entity = weather_entity
        
        self.power_entity = power_entity
        self.solar_yield_today = solar_yield_today
        self.solar_capacity = solar_capacity
        
        self.temp_sensor = temp_sensor
        self.wind_sensor = wind_sensor
        self.rain_sensor = rain_sensor
        self.uv_sensor = uv_sensor
        self.lux_sensor = lux_sensor
        
        # Service References
        self.error_handler: Optional[Any] = None
        self.weather_service: Optional[Any] = None
        self.ml_predictor: Optional[Any] = None
        self.notification_service: Optional[Any] = None
        self.sun_guard: Optional[Any] = None
        
        # Status Flags
        self._services_initialized = False
        self._ml_ready = False
        self._initialization_lock = asyncio.Lock()
        
        self.dependencies_installed = dependencies_ok
    
    async def initialize_all_services(self) -> bool:
        """
        Initializes all services in the correct order.
        
        Thread-safe with a lock for parallel calls.
        
        PATCH: Improved error handling and validation
        
        Returns:
            True if successfully initialized
        """
        async with self._initialization_lock:
            if self._services_initialized:
                _LOGGER.debug("Services already initialized")
                return True
            
            _LOGGER.info("Initializing services...")
            
            try:
                error_handler_ok = await self._initialize_error_handler()
                if not error_handler_ok:
                    _LOGGER.warning("Error Handler initialization failed - proceeding")
                
                sun_guard_ok = await self._initialize_sun_guard()
                if not sun_guard_ok:
                    _LOGGER.warning("Sun Guard initialization failed - proceeding")
                
                if not self.notification_service:
                    self.notification_service = self.hass.data.get(DOMAIN, {}).get("notification_service")
                    if self.notification_service:
                        _LOGGER.info("NotificationService taken from hass.data")
                    else:
                        notif_ok = await self._initialize_notification_service()
                        if not notif_ok:
                            _LOGGER.warning("Notification Service not available")
                else:
                    _LOGGER.debug("Notification Service already set")
                
                weather_ok = await self._initialize_weather_service()
                if not weather_ok:
                    _LOGGER.warning("Weather Service initialization failed - proceeding")
                
                ml_ok = await self._initialize_ml_predictor()
                if not ml_ok:
                    _LOGGER.warning("ML Predictor not available - fallback active")
                    self._ml_ready = False

                
                self._services_initialized = True
                _LOGGER.info(f"Services initialized - ML Ready: {self._ml_ready}")
                
                self._log_service_status()
                return True
                
            except Exception as e:
                _LOGGER.error(f"Service initialization failed: {e}", exc_info=True)
                return False
    
    def _log_service_status(self):
        """
        NEW: Log status of all services for debugging.
        """
        _LOGGER.info("Service Status:")
        _LOGGER.info(f"  - Error Handler: {'OK' if self.error_handler else 'MISSING'}")
        _LOGGER.info(f"  - Sun Guard: {'OK' if self.sun_guard else 'MISSING'}")
        _LOGGER.info(f"  - Notification Service: {'OK' if self.notification_service else 'MISSING'}")
        _LOGGER.info(f"  - Weather Service: {'OK' if self.weather_service else 'MISSING'}")
        _LOGGER.info(f"  - ML Predictor: {'OK' if self.ml_predictor else 'MISSING'}")
        _LOGGER.info(f"  - ML Ready: {'YES' if self._ml_ready else 'NO'}")
    
    async def _initialize_error_handler(self) -> bool:
        """
        Initializes Error Handler Service.
        
        PATCH: Returns success status
        """
        try:
            from .error_handling_service import ErrorHandlingService
            
            self.error_handler = ErrorHandlingService()
            _LOGGER.debug("Error Handler initialized")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Error Handler initialization failed: {e}", exc_info=True)
            return False
    
    async def _initialize_sun_guard(self) -> bool:
        """Initializes the Sun Guard service."""
        try:
            from .sun_guard import SunGuard
            
            self.sun_guard = SunGuard(
                hass=self.hass,
                buffer_hours=1.0
            )
            
            self.sun_guard.log_production_window()
            
            if self.sun_guard.is_production_time():
                _LOGGER.info("DATA COLLECTION STARTED")
            else:
                _LOGGER.info("DATA COLLECTION PAUSED")
            
            _LOGGER.debug("Sun Guard initialized")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Sun Guard initialization failed: {e}", exc_info=True)
            return False
    
    async def _initialize_notification_service(self) -> bool:
        """
        Initializes Notification Service.
        
        PATCH: Returns success status
        """
        try:
            from .notification_service import NotificationService
            
            self.notification_service = NotificationService(
                self.hass,
                self.entry.entry_id
            )
            
            _LOGGER.debug("Notification Service initialized")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Notification Service initialization failed: {e}", exc_info=True)
            return False
    
    async def _initialize_weather_service(self) -> bool:
        """
        Initializes Weather Service.
        
        PATCH: Returns success status
        """
        try:
            from .weather_service import WeatherService
            
            weather_state = self.hass.states.get(self.weather_entity)
            if not weather_state:
                _LOGGER.error(f"Weather Entity not found: {self.weather_entity}")
                return False
            
            self.weather_service = WeatherService(
                self.hass,
                self.weather_entity,
                self.error_handler
            )
            
            weather_init_success = await self.weather_service.initialize()
            
            if weather_init_success:
                _LOGGER.debug("Weather Service initialized")
                return True
            else:
                _LOGGER.error("Weather Service initialization failed")
                return False
                
        except Exception as e:
            _LOGGER.warning(f"Weather Service initialization failed: {e}")
            return False
    
    async def _initialize_ml_predictor(self) -> bool:
        """
        Initializes ML Predictor with dependencies.
        
        PATCH: Returns success status and better validation
        FIX: set_entities is called after initialize
        """
        try:
            if not self.dependencies_installed:
                _LOGGER.warning(
                    "ML Dependencies missing (numpy, aiofiles) - "
                    "ML Predictor will not be initialized"
                )
                self._ml_ready = False
                return False
            
            from .ml_predictor import MLPredictor
            
            _LOGGER.info("Initializing ML Predictor with dependencies...")
            
            if not self.data_manager:
                _LOGGER.error("DataManager not available - ML cannot be initialized")
                return False
            
            self.ml_predictor = MLPredictor(
                self.hass,
                self.data_manager,
                self.error_handler
            )
            
            ml_init_success = await self.ml_predictor.initialize()
            
            if ml_init_success:
                _LOGGER.info("ML Predictor successfully initialized")

                self.ml_predictor.set_entities(
                    power_entity=self.power_entity,
                    solar_yield_today=self.solar_yield_today,
                    weather_entity=self.weather_entity,
                    solar_capacity=self.solar_capacity,
                    forecast_cache={},
                    temp_sensor=self.temp_sensor,
                    wind_sensor=self.wind_sensor,
                    rain_sensor=self.rain_sensor,
                    uv_sensor=self.uv_sensor,
                    lux_sensor=self.lux_sensor,
                    sun_guard=self.sun_guard
                )
                _LOGGER.info(
                    f"ML Entities set: power={self.power_entity}, "
                    f"yield={self.solar_yield_today}, external_sensors: "
                    f"temp={self.temp_sensor}, wind={self.wind_sensor}, "
                    f"rain={self.rain_sensor}, uv={self.uv_sensor}, "
                    f"lux={self.lux_sensor}"
                )

                # *** BACKFILL-TRIGGER WURDE ENTFERNT ***
                # Ursprünglicher Code:
                # _LOGGER.debug("Scheduling initial backfill trigger...")
                # asyncio.create_task(self.ml_predictor.async_run_initial_backfill())
                
                self._ml_ready = True
                return True
            else:
                _LOGGER.warning("ML Predictor initialization failed - Fallback active")
                self._ml_ready = False
                return False
                
        except ImportError as e:
            _LOGGER.error(f"ML Predictor import failed (Dependencies missing?): {e}")
            self._ml_ready = False
            return False
        except Exception as e:
            _LOGGER.error(f"ML Predictor initialization failed: {e}", exc_info=True)
            self._ml_ready = False
            return False
    
    def is_ml_ready(self) -> bool:
        """
        Checks if ML is ready.
        
        Returns:
            True if ML Predictor is available and healthy
        """
        if not self._ml_ready or not self.ml_predictor:
            return False
        
        try:
            if hasattr(self.ml_predictor, 'is_healthy'):
                return self.ml_predictor.is_healthy()
            else:
                return True
        except Exception as e:
            _LOGGER.debug(f"ML Health check failed: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """
        Checks if services are initialized.
        
        Returns:
            True if initialized
        """
        return self._services_initialized
    
    def get_service_status(self) -> dict[str, Any]:
        """
        NEW: Returns status of all services for debugging/UI.
        
        Returns:
            Dict with service status
        """
        return {
            "initialized": self._services_initialized,
            "error_handler_available": self.error_handler is not None,
            "notification_service_available": self.notification_service is not None,
            "weather_service_available": self.weather_service is not None,
            "ml_predictor_available": self.ml_predictor is not None,
            "ml_ready": self._ml_ready,
            "dependencies_installed": self.dependencies_installed,
        }