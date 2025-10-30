"""
Service Manager for Solar Forecast ML Integration.
Manages the lifecycle and initialization of core services like
ML Predictor, Weather Service, Notification Service, and Error Handler.

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
from typing import Optional, Any, TYPE_CHECKING, Dict

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry

# Import constants used for configuration lookup
from .const import (
    DOMAIN, CONF_SOLAR_CAPACITY, DEFAULT_SOLAR_CAPACITY,
    SERVICE_RETRAIN_MODEL, SERVICE_RESET_LEARNING_DATA
)

# Use TYPE_CHECKING for components only needed for type hints
if TYPE_CHECKING:
    from .data_manager import DataManager
    from .error_handling_service import ErrorHandlingService
    from .weather_service import WeatherService
    from .ml_predictor import MLPredictor
    from .notification_service import NotificationService
    from .sun_guard import SunGuard
    from .coordinator import SolarForecastMLCoordinator

_LOGGER = logging.getLogger(__name__)


class ServiceManager:
    """
    Manages the lifecycle (initialization, access) of various services
    used by the Solar Forecast ML integration. Encapsulates service
    management logic, separating it from the Coordinator.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        data_manager: 'DataManager',
        weather_entity: Optional[str],
        dependencies_ok: bool = False,
        power_entity: Optional[str] = None,
        solar_yield_today: Optional[str] = None,
        solar_capacity: Optional[float] = None,
        temp_sensor: Optional[str] = None,
        wind_sensor: Optional[str] = None,
        rain_sensor: Optional[str] = None,
        uv_sensor: Optional[str] = None,
        lux_sensor: Optional[str] = None,
        humidity_sensor: Optional[str] = None,
    ):
        self.hass = hass
        self.entry = entry
        self.data_manager = data_manager
        self._initial_weather_entity = weather_entity
        self._power_entity = power_entity
        self._solar_yield_today = solar_yield_today
        self._solar_capacity = solar_capacity if solar_capacity is not None \
            else float(entry.data.get(CONF_SOLAR_CAPACITY, DEFAULT_SOLAR_CAPACITY))
        self._temp_sensor = temp_sensor
        self._wind_sensor = wind_sensor
        self._rain_sensor = rain_sensor
        self._uv_sensor = uv_sensor
        self._lux_sensor = lux_sensor
        self._humidity_sensor = humidity_sensor

        # Service instance references
        self.error_handler: Optional['ErrorHandlingService'] = None
        self.weather_service: Optional['WeatherService'] = None
        self.ml_predictor: Optional['MLPredictor'] = None
        self.notification_service: Optional['NotificationService'] = None
        self.sun_guard: Optional['SunGuard'] = None
        self.coordinator: Optional['SolarForecastMLCoordinator'] = None

        # Status Flags
        self._services_initialized: bool = False
        self._ml_ready: bool = False
        self._initialization_lock = asyncio.Lock()
        self.dependencies_installed: bool = dependencies_ok

        _LOGGER.debug("ServiceManager initialized.")

    async def initialize_all_services(self) -> bool:
        async with self._initialization_lock:
            if self._services_initialized:
                _LOGGER.debug("Services already initialized, skipping.")
                return True

            _LOGGER.info("Initializing managed services...")
            overall_success = True

            try:
                # 1. Error Handler
                error_handler_ok = await self._initialize_error_handler()
                if not error_handler_ok:
                    _LOGGER.critical("Failed to initialize Error Handling Service. Aborting.")
                    return False

                # 2. Sun Guard
                sun_guard_ok = await self._initialize_sun_guard()
                if not sun_guard_ok:
                    _LOGGER.warning("Sun Guard initialization failed, continuing without it.")

                # 3. Notification Service
                self.notification_service = self.hass.data.get(DOMAIN, {}).get("notification_service")
                if self.notification_service:
                    _LOGGER.info("NotificationService instance retrieved from hass.data.")
                else:
                    _LOGGER.warning("NotificationService not found in hass.data, attempting initialization.")
                    notif_ok = await self._initialize_notification_service()
                    if not notif_ok:
                        _LOGGER.warning("Notification Service initialization failed or unavailable.")

                # 4. Weather Service
                weather_ok = await self._initialize_weather_service()
                if not weather_ok:
                    _LOGGER.error("Weather Service initialization failed. Forecasts may be unavailable.")
                    overall_success = False

                # 5. ML Predictor
                ml_ok = await self._initialize_ml_predictor()
                if not ml_ok:
                    _LOGGER.warning("ML Predictor initialization failed or skipped (dependencies?).")
                else:
                    _LOGGER.info("ML Predictor initialized successfully.")

                self._services_initialized = True
                _LOGGER.info(f"Service initialization sequence completed. Overall Success: {overall_success}, ML Ready: {self._ml_ready}")
                self._log_service_status()
                return overall_success

            except Exception as e:
                _LOGGER.critical(f"Unexpected critical error during service initialization: {e}", exc_info=True)
                self._services_initialized = False
                return False

    def _log_service_status(self):
        _LOGGER.debug("--- Service Status Report ---")
        _LOGGER.debug(f"  - Error Handler: {'Initialized' if self.error_handler else 'FAILED'}")
        _LOGGER.debug(f"  - Sun Guard: {'Initialized' if self.sun_guard else 'Failed/Skipped'}")
        _LOGGER.debug(f"  - Notification Service: {'Available' if self.notification_service else 'Unavailable'}")
        _LOGGER.debug(f"  - Weather Service: {'Initialized' if self.weather_service else 'FAILED/Skipped'}")
        _LOGGER.debug(f"  - ML Predictor: {'Initialized' if self.ml_predictor else 'FAILED/Skipped'}")
        _LOGGER.debug(f"  - Dependencies Installed: {self.dependencies_installed}")
        _LOGGER.debug(f"  - ML Ready (Healthy): {self._ml_ready}")
        _LOGGER.debug("-----------------------------")

    async def _initialize_error_handler(self) -> bool:
        if self.error_handler:
            return True
        _LOGGER.debug("Initializing ErrorHandlingService...")
        try:
            from .error_handling_service import ErrorHandlingService
            self.error_handler = ErrorHandlingService()
            _LOGGER.info("ErrorHandlingService initialized.")
            return True
        except Exception as e:
            _LOGGER.exception(f"Failed to initialize ErrorHandlingService: {e}")
            return False

    async def _initialize_sun_guard(self) -> bool:
        """Initializes the Sun Guard service for production time window calculations."""
        if self.sun_guard:
            return True
        _LOGGER.debug("Initializing SunGuard...")
        try:
            from .sun_guard import SunGuard
            from .const import SUN_BUFFER_HOURS

            self.sun_guard = SunGuard(
                hass=self.hass,
                buffer_hours=SUN_BUFFER_HOURS
            )

            # --- KORREKTUR: await hinzufügen! ---
            # Option 1: Warte auf Logging (sicher)
            await self.sun_guard.log_production_window()

            # Option 2: Hole is_production_time() mit await
            is_prod = await self.sun_guard.is_production_time()
            _LOGGER.info(f"SunGuard initialized (Production Time: {is_prod}).")

            # BONUS: Alternativ – Hintergrund-Logging (nicht blockierend)
            # self.hass.async_create_task(self.sun_guard.log_production_window())

            return True
        except Exception as e:
            _LOGGER.exception(f"Failed to initialize SunGuard: {e}")
            return False

    async def _initialize_notification_service(self) -> bool:
        if self.notification_service:
            return True
        _LOGGER.debug("Initializing NotificationService...")
        try:
            from .notification_service import create_notification_service
            self.notification_service = await create_notification_service(self.hass)
            _LOGGER.info("NotificationService initialized.")
            return True
        except Exception as e:
            _LOGGER.exception(f"Failed to initialize NotificationService: {e}")
            return False

    async def _initialize_weather_service(self) -> bool:
        if self.weather_service:
            return True
        if not self._initial_weather_entity:
            _LOGGER.error("Cannot initialize WeatherService: No weather entity configured.")
            return False
        _LOGGER.debug(f"Initializing WeatherService for entity: {self._initial_weather_entity}...")
        try:
            from .weather_service import WeatherService
            self.weather_service = WeatherService(
                self.hass,
                self._initial_weather_entity,
                self.error_handler
            )
            init_ok = await self.weather_service.initialize()
            if init_ok:
                _LOGGER.info("WeatherService initialized.")
            else:
                _LOGGER.warning("WeatherService.initialize() returned False.")
            return init_ok
        except Exception as e:
            _LOGGER.exception(f"Failed to initialize WeatherService: {e}")
            return False

    async def _initialize_ml_predictor(self) -> bool:
        if self.ml_predictor:
            return True

        if not self.dependencies_installed:
            _LOGGER.warning("ML Predictor initialization skipped: Required Python dependencies missing.")
            self._ml_ready = False
            return False

        _LOGGER.debug("Initializing MLPredictor...")
        try:
            if not self.data_manager:
                _LOGGER.error("Cannot initialize ML Predictor: DataManager is not available.")
                return False
            if not self.error_handler:
                _LOGGER.error("Cannot initialize ML Predictor: ErrorHandler is not available.")
                return False

            from .ml_predictor import MLPredictor

            self.ml_predictor = MLPredictor(
                self.hass,
                self.data_manager,
                self.error_handler
            )

            ml_init_success = await self.ml_predictor.initialize()

            if ml_init_success:
                _LOGGER.info("MLPredictor.initialize() successful.")
                self.ml_predictor.set_entities(
                    power_entity=self._power_entity,
                    weather_entity=self._initial_weather_entity,
                    solar_capacity=self._solar_capacity,
                    temp_sensor=self._temp_sensor,
                    wind_sensor=self._wind_sensor,
                    rain_sensor=self._rain_sensor,
                    uv_sensor=self._uv_sensor,
                    lux_sensor=self._lux_sensor,
                    humidity_sensor=self._humidity_sensor,
                    sun_guard=self.sun_guard
                )
                _LOGGER.debug("Entities configured in MLPredictor and SampleCollector.")

                self._ml_ready = self.is_ml_ready()
                _LOGGER.info(f"ML Predictor initialization complete. ML Ready state: {self._ml_ready}")
                return True
            else:
                _LOGGER.warning("MLPredictor.initialize() returned False. ML features disabled.")
                self._ml_ready = False
                self.ml_predictor = None
                return False

        except ImportError as import_err:
            _LOGGER.critical(f"ML Predictor import failed: {import_err}.")
            self._ml_ready = False
            self.ml_predictor = None
            self.dependencies_installed = False
            return False
        except Exception as e:
            _LOGGER.exception(f"Unexpected error during ML Predictor initialization: {e}")
            self._ml_ready = False
            self.ml_predictor = None
            return False

    def is_ml_ready(self) -> bool:
        if not self._services_initialized:
            return False
        if not self.ml_predictor:
            return False
        if not self.dependencies_installed:
            return False

        try:
            is_healthy = self.ml_predictor.is_healthy()
            self._ml_ready = is_healthy
            return is_healthy
        except Exception as e:
            _LOGGER.warning(f"ML health check failed with an error: {e}")
            self._ml_ready = False
            return False

    def is_initialized(self) -> bool:
        return self._services_initialized

    def get_service_status(self) -> Dict[str, Any]:
        return {
            "initialization_complete": self._services_initialized,
            "dependencies_installed": self.dependencies_installed,
            "error_handler_available": self.error_handler is not None,
            "notification_service_available": self.notification_service is not None,
            "sun_guard_available": self.sun_guard is not None,
            "weather_service_available": self.weather_service is not None,
            "ml_predictor_available": self.ml_predictor is not None,
            "ml_ready_flag": self._ml_ready,
            "ml_healthy_check": self.is_ml_ready() if self._services_initialized else None,
        }

    async def cleanup(self) -> None:
        _LOGGER.info("Cleaning up ServiceManager and managed services...")
        if self.ml_predictor:
            await self.ml_predictor.async_will_remove_from_hass()
        self._services_initialized = False
        _LOGGER.info("ServiceManager cleanup complete.")

    # --- Service Handler ---
    async def _handle_service_retrain(self, service_call: Any) -> None:
        _LOGGER.info("Service 'force_retrain' called. Triggering ML model training...")
        if not self.ml_predictor:
            _LOGGER.error("Cannot retrain model: ML Predictor service is not available.")
            if self.notification_service:
                await self.notification_service.show_installation_error(
                    "ML Training Failed: The ML Predictor service is not running. Check logs.",
                    notification_id="ml_retrain_fail_no_service"
                )
            return

        try:
            result = await self.ml_predictor.train_model()
            if result and result.success:
                accuracy_str = f"{result.accuracy * 100:.1f}%" if result.accuracy is not None else "N/A"
                _LOGGER.info(f"Service-triggered ML training completed successfully. Accuracy: {accuracy_str}")
                if self.notification_service:
                    await self.notification_service.show_installation_success(
                        "ML Training Successful",
                        f"Manual ML training completed. New accuracy: {accuracy_str}",
                        notification_id="ml_retrain_success"
                    )
            elif result:
                _LOGGER.error(f"Service-triggered ML training failed: {result.error_message}")
                if self.notification_service:
                    await self.notification_service.show_installation_error(
                        f"Manual ML training failed: {result.error_message}",
                        notification_id="ml_retrain_fail"
                    )
            else:
                _LOGGER.error("Service-triggered ML training failed with an unexpected result structure.")

        except Exception as e:
            _LOGGER.error(f"An unexpected error occurred during service-triggered ML training: {e}", exc_info=True)
            if self.notification_service:
                await self.notification_service.show_installation_error(
                    f"An unexpected error occurred during training: {e}",
                    notification_id="ml_retrain_fail_exception"
                )

    async def _handle_service_reset(self, service_call: Any) -> None:
        _LOGGER.warning("Service 'reset_model' called. This will delete all learned data.")
        if not self.data_manager:
            _LOGGER.error("Cannot reset model: DataManager is not available.")
            return
        if not self.ml_predictor:
            _LOGGER.error("Cannot reset model: ML Predictor is not available.")
            return
        if not self.coordinator:
            _LOGGER.error("Cannot reset model: Coordinator reference is missing.")
            return

        try:
            await self.data_manager.reset_ml_data()
            _LOGGER.info("ML data files (weights, profile, samples) have been reset to default.")

            from .ml_scaler import StandardScaler
            self.ml_predictor.scaler = StandardScaler()
            self.ml_predictor.current_weights = None
            self.ml_predictor.current_profile = None
            self.ml_predictor.model_loaded = False
            self.ml_predictor.current_accuracy = None
            self.ml_predictor.training_samples = 0
            self.ml_predictor.last_training_time = None

            await self.ml_predictor.initialize()
            await self.coordinator.async_request_refresh()

            _LOGGER.info("ML model has been reset and re-initialized.")
            if self.notification_service:
                await self.notification_service.show_installation_success(
                    "ML Model Reset",
                    "All ML data has been deleted. The model will start learning from scratch.",
                    notification_id="ml_reset_success"
                )

        except Exception as e:
            _LOGGER.error(f"An unexpected error occurred during model reset: {e}", exc_info=True)
            if self.notification_service:
                await self.notification_service.show_installation_error(
                    f"An unexpected error occurred during reset: {e}",
                    notification_id="ml_reset_fail_exception"
                )