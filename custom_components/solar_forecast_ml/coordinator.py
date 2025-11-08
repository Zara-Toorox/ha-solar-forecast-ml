"""
Data Update Coordinator for Solar Forecast ML Integration

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

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.helpers.event import async_track_time_change
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN

from .const import (
    DOMAIN, CONF_SOLAR_CAPACITY, CONF_UPDATE_INTERVAL, CONF_WEATHER_ENTITY,
    CONF_LEARNING_ENABLED,
    CONF_HOURLY, CONF_POWER_ENTITY, CONF_SOLAR_YIELD_TODAY,
    CONF_TOTAL_CONSUMPTION_TODAY,
    CONF_HUMIDITY_SENSOR,
    CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX, DATA_DIR,
    ML_MODEL_VERSION, DAILY_UPDATE_HOUR, DAILY_VERIFICATION_HOUR,
    CONF_TEMP_SENSOR, CONF_WIND_SENSOR, CONF_RAIN_SENSOR,
    CONF_UV_SENSOR, CONF_LUX_SENSOR,
    DEFAULT_SOLAR_CAPACITY,
    UPDATE_INTERVAL,
    CONF_FALLBACK_ENTITY,
    # Battery Management constants
    CONF_BATTERY_ENABLED,
    CONF_ELECTRICITY_ENABLED,
    CONF_ELECTRICITY_COUNTRY,
    DEFAULT_ELECTRICITY_COUNTRY,
)
from .data.data_manager import DataManager
from .core.core_exceptions import SolarForecastMLException, WeatherAPIException, MLModelException
from .core.core_helpers import SafeDateTimeUtil as dt_util

# Import modular components
from .forecast.forecast_weather_calculator import WeatherCalculator
from .production.production_history import ProductionCalculator as HistoricalProductionCalculator
from .production.production_tracker import ProductionTimeCalculator
from .sensors.sensor_data_collector import SensorDataCollector
from .forecast.forecast_orchestrator import ForecastOrchestrator
from .production.production_scheduled_tasks import ScheduledTasksManager
from .ml.ml_predictor import ModelState, MLPredictor
from .services.service_error_handler import ErrorHandlingService
from .forecast.forecast_weather import WeatherService

# Battery Management components (completely separate from Solar/ML)
from .battery.battery_data_collector import BatteryDataCollector
from .battery.electricity_price_service import ElectricityPriceService

_LOGGER = logging.getLogger(__name__)


class SolarForecastMLCoordinator(DataUpdateCoordinator):
    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        dependencies_ok: bool = False,
    ):
        update_interval_seconds = entry.options.get(CONF_UPDATE_INTERVAL, UPDATE_INTERVAL.total_seconds())
        update_interval_timedelta = timedelta(seconds=update_interval_seconds)

        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=update_interval_timedelta
        )

        self.entry = entry
        self.dependencies_ok = dependencies_ok

        config_dir = hass.config.path()
        data_dir_path = Path(config_dir) / DOMAIN
        self.data_manager = DataManager(hass, entry.entry_id, data_dir_path)
        self.sensor_collector = SensorDataCollector(hass, entry)

        # Migration: Support old plant_kwp field for backward compatibility
        solar_capacity_value = entry.data.get(CONF_SOLAR_CAPACITY)
        if solar_capacity_value is None or solar_capacity_value == 0:
            # Try old field name for backward compatibility
            solar_capacity_value = entry.data.get("plant_kwp", DEFAULT_SOLAR_CAPACITY)
            if solar_capacity_value != DEFAULT_SOLAR_CAPACITY:
                _LOGGER.warning(f"Using legacy 'plant_kwp' value: {solar_capacity_value} kW. Please reconfigure to update.")
        
        self.solar_capacity = float(solar_capacity_value)
        self.learning_enabled = entry.options.get(CONF_LEARNING_ENABLED, True)
        self.enable_hourly = entry.options.get(CONF_HOURLY, False)

        self.power_entity = self.sensor_collector.strip_entity_id(entry.data.get(CONF_POWER_ENTITY))
        self.solar_yield_today = self.sensor_collector.strip_entity_id(entry.data.get(CONF_SOLAR_YIELD_TODAY))
        self.primary_weather_entity = self.sensor_collector.strip_entity_id(entry.data.get(CONF_WEATHER_ENTITY))
        self.current_weather_entity: Optional[str] = self.primary_weather_entity
        self.total_consumption_today = self.sensor_collector.strip_entity_id(entry.data.get(CONF_TOTAL_CONSUMPTION_TODAY))

        self.weather_calculator = WeatherCalculator()
        self.historical_calculator = HistoricalProductionCalculator(hass, self.data_manager)
        self.production_time_calculator = ProductionTimeCalculator(hass=hass, power_entity=self.power_entity, data_manager=self.data_manager, coordinator=self)
        self.forecast_orchestrator = ForecastOrchestrator(
            hass=hass,
            data_manager=self.data_manager,
            solar_capacity=self.solar_capacity,
            weather_calculator=self.weather_calculator
        )

        self.scheduled_tasks = ScheduledTasksManager(
            hass=hass,
            coordinator=self,
            solar_yield_today_entity_id=self.solar_yield_today,
            data_manager=self.data_manager
        )

        # Initialize services directly (no ServiceManager)
        self.error_handler = ErrorHandlingService()
        self.weather_service: Optional[WeatherService] = None
        self.ml_predictor: Optional[MLPredictor] = None
        self._services_initialized = False
        self._ml_ready = False

        self.weather_fallback_active = False
        self._last_weather_update: Optional[datetime] = None
        self._forecast_cache: Dict[str, Any] = {}
        self._startup_time: datetime = dt_util.now()
        self._last_update_success_time: Optional[datetime] = None
        self._startup_sensors_ready: bool = False

        self.next_hour_pred: float = 0.0
        self.peak_production_time_today: str = "Calculating..."
        self.production_time_today: str = "Initializing..."
        self.last_day_error_kwh: Optional[float] = None
        self.yesterday_accuracy: Optional[float] = None
        self.autarky_today: Optional[float] = None
        self.avg_month_yield: float = 0.0
        self.last_successful_learning: Optional[datetime] = None
        self.model_accuracy: Optional[float] = None
        self.learned_correction_factor: float = 1.0
        self.expected_daily_production: Optional[float] = None
        self._last_statistics_calculation: Optional[datetime] = None

        # Recovery lock to prevent concurrent recovery attempts
        self._recovery_lock = asyncio.Lock()
        self._recovery_in_progress = False

        # ========================================================================
        # BATTERY MANAGEMENT (v8.3.0 Extension) - Completely separate from Solar/ML
        # ========================================================================
        self.battery_enabled = entry.options.get(CONF_BATTERY_ENABLED, False)
        self.electricity_enabled = entry.options.get(CONF_ELECTRICITY_ENABLED, False)

        # Initialize battery collector if enabled
        self.battery_collector: Optional[BatteryDataCollector] = None
        if self.battery_enabled:
            try:
                self.battery_collector = BatteryDataCollector(hass, entry)
                _LOGGER.info("BatteryDataCollector initialized successfully")
            except Exception as e:
                _LOGGER.error(f"Failed to initialize BatteryDataCollector: {e}")

        # Initialize electricity price service if enabled (aWATTar API - Free)
        self.electricity_service: Optional[ElectricityPriceService] = None
        if self.electricity_enabled:
            try:
                # Get country from options with fallback to data for backwards compatibility
                country = entry.options.get(CONF_ELECTRICITY_COUNTRY) or entry.data.get(CONF_ELECTRICITY_COUNTRY, DEFAULT_ELECTRICITY_COUNTRY)
                self.electricity_service = ElectricityPriceService(country=country)
                _LOGGER.info(f"ElectricityPriceService initialized for {country} using aWATTar API (free, no registration)")
            except Exception as e:
                _LOGGER.error(f"Failed to initialize ElectricityPriceService: {e}")

        _LOGGER.info(f"SolarForecastMLCoordinator initialized - Using Weather Entity: {self.primary_weather_entity or 'None'}")

    async def _load_persistent_state(self) -> None:
        """Load persistent coordinator state eg expected_daily_production by Zara"""
        try:
            # Load expected_daily_production if it was set today
            loaded_value = await self.data_manager.load_expected_daily_production()
            if loaded_value is not None:
                self.expected_daily_production = loaded_value
                _LOGGER.info(
                    f"Restored expected_daily_production from persistent storage: "
                    f"{loaded_value:.2f} kWh"
                )
        except Exception as e:
            _LOGGER.warning(f"Failed to load persistent coordinator state: {e}")

    async def _initialize_services(self) -> bool:
        """Initialize all services weather ML error handler by Zara"""
        try:
            # Error handler is already initialized
            
            # Initialize WeatherService
            if self.current_weather_entity:
                self.weather_service = WeatherService(
                    hass=self.hass,
                    weather_entity=self.current_weather_entity,
                    data_manager=self.data_manager
                )
                await self.weather_service.initialize()  # Initialize to load cache
                _LOGGER.info(f"WeatherService initialized with entity: {self.current_weather_entity}")
            else:
                _LOGGER.warning("No weather entity configured - WeatherService not initialized")
                
            # Initialize MLPredictor
            if self.learning_enabled and self.dependencies_ok:
                try:
                    # Get notification service from hass.data (optional)
                    notification_service = self.hass.data.get(DOMAIN, {}).get("notification_service")
                    
                    self.ml_predictor = MLPredictor(
                        hass=self.hass,
                        data_manager=self.data_manager,
                        error_handler=self.error_handler,
                        notification_service=notification_service
                    )
                    _LOGGER.info("MLPredictor instance created successfully")
                    
                    # Set solar capacity and all sensor entities for ML predictor
                    self.ml_predictor.set_entities(
                        solar_capacity=self.solar_capacity,
                        power_entity=self.power_entity,
                        weather_entity=self.current_weather_entity,
                        temp_sensor=self.entry.data.get(CONF_TEMP_SENSOR),
                        wind_sensor=self.entry.data.get(CONF_WIND_SENSOR),
                        rain_sensor=self.entry.data.get(CONF_RAIN_SENSOR),
                        uv_sensor=self.entry.data.get(CONF_UV_SENSOR),
                        lux_sensor=self.entry.data.get(CONF_LUX_SENSOR),
                        humidity_sensor=self.entry.data.get(CONF_HUMIDITY_SENSOR)
                    )
                    _LOGGER.info(
                        f"MLPredictor entities configured: "
                        f"solar_capacity={self.solar_capacity}kW, "
                        f"power={self.power_entity}, "
                        f"weather={self.current_weather_entity}, "
                        f"temp={self.entry.data.get(CONF_TEMP_SENSOR)}, "
                        f"wind={self.entry.data.get(CONF_WIND_SENSOR)}, "
                        f"rain={self.entry.data.get(CONF_RAIN_SENSOR)}, "
                        f"uv={self.entry.data.get(CONF_UV_SENSOR)}, "
                        f"lux={self.entry.data.get(CONF_LUX_SENSOR)}, "
                        f"humidity={self.entry.data.get(CONF_HUMIDITY_SENSOR)}"
                    )
                    
                    # Initialize ML Predictor (loads model, starts background tasks)
                    init_success = await self.ml_predictor.initialize()
                    if init_success:
                        self._ml_ready = True
                        _LOGGER.info("MLPredictor initialized and ready")
                    else:
                        _LOGGER.error("MLPredictor initialization failed")
                        self.ml_predictor = None
                except Exception as e:
                    _LOGGER.error(f"Failed to initialize MLPredictor: {e}", exc_info=True)
                    self.ml_predictor = None
            else:
                reason = "disabled" if not self.learning_enabled else "dependencies missing"
                _LOGGER.info(f"ML learning {reason} - MLPredictor not initialized")

            self._services_initialized = True
            _LOGGER.info("All services initialized successfully")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Failed to initialize services: {e}", exc_info=True)
            return False

    async def _initialize_forecast_orchestrator(self) -> None:
        """Initialize the forecast orchestrator strategies by Zara"""
        if not self.ml_predictor:
            _LOGGER.info("Initializing ForecastOrchestrator without ML predictor")
            self.forecast_orchestrator.initialize_strategies(
                ml_predictor=None,
                error_handler=self.error_handler
            )
        else:
            _LOGGER.info("Initializing ForecastOrchestrator with ML predictor and error handler")
            self.forecast_orchestrator.initialize_strategies(
                ml_predictor=self.ml_predictor,
                error_handler=self.error_handler
            )
        _LOGGER.info("ForecastOrchestrator strategies initialized successfully")

    async def async_setup(self) -> bool:
        """Setup coordinator and start tracking by Zara"""
        try:
            _LOGGER.info("Starting coordinator setup...")

            # Initialize DataManager to ensure directories and files exist
            init_ok = await self.data_manager.initialize()
            if not init_ok:
                _LOGGER.error("Failed to initialize data manager")
                return False

            # Initialize services first
            services_ok = await self._initialize_services()
            if not services_ok:
                _LOGGER.error("Failed to initialize services")
                return False

            # Load persistent state
            await self._load_persistent_state()

            # Setup production time tracking with error handling
            try:
                await self.production_time_calculator.start_tracking()
                _LOGGER.info("Production time tracking started successfully")
            except Exception as track_err:
                _LOGGER.error(
                    f"Failed to start production time tracking: {track_err}. "
                    f"Production time sensors will be unavailable.",
                    exc_info=True
                )
                # Continue setup - tracking failure is not critical for core functionality

            # Setup power peak tracking
            await self._setup_power_peak_tracking()

            # Setup scheduled tasks
            self.scheduled_tasks.setup_listeners()

            # Setup next_hour forecast updates (zur vollen Stunde)
            @callback
            def _scheduled_next_hour_update(now: datetime) -> None:
                """Callback for next hour forecast update - zur vollen Stunde by Zara"""
                asyncio.create_task(self._update_next_hour_forecast())
            
            async_track_time_change(
                self.hass,
                _scheduled_next_hour_update,
                minute=0,  # Only at the top of the hour
                second=5   # 5 seconds after the top of the hour
            )
            _LOGGER.info("Scheduled: next_hour updates at every full hour")

            # One-time update after HA restart (only if during production time)
            async def startup_next_hour_init():
                """Initial next hour calculation after startup by Zara"""
                await asyncio.sleep(60)  # Wait 60 seconds for sensors to stabilize

                now_local = dt_util.now()
                next_hour = (now_local + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

                # Only if next hour is during production time
                if self.forecast_orchestrator.is_production_hour(next_hour):
                    await self._update_next_hour_forecast()
                    _LOGGER.info(f"Startup: Initial next hour forecast calculated for {next_hour.hour}:00")
                else:
                    _LOGGER.debug(f"Startup: Next hour {next_hour.hour}:00 outside production - skipped")
            
            asyncio.create_task(startup_next_hour_init())

            # Setup midnight forecast rotation (00:00:30)
            @callback
            def _scheduled_midnight_rotation(now: datetime) -> None:
                """Callback for midnight forecast rotation - thread-safe by Zara"""
                asyncio.create_task(self._rotate_forecasts_midnight())
            
            async_track_time_change(
                self.hass,
                _scheduled_midnight_rotation,
                hour=0,
                minute=0,
                second=30
            )
            _LOGGER.info("Scheduled: midnight forecast rotation (00:00:30)")

            # Calculate yesterday's deviation
            await self.scheduled_tasks.calculate_yesterday_deviation_on_startup()

            # Log comprehensive setup summary
            ml_status = "Active & Ready" if self._ml_ready else "Disabled/Fallback"
            weather_status = "Configured" if self.primary_weather_entity else "Not configured"

            _LOGGER.info(
                f"Solar Forecast Coordinator Setup Complete ✓\n"
                f"  → ML Engine: {ml_status}\n"
                f"  → Weather Integration: {weather_status}\n"
                f"  → Solar Capacity: {self.solar_capacity} kWp\n"
                f"  → Production Tracking: Active\n"
                f"  → Scheduled Tasks: All registered"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to setup coordinator: {e}", exc_info=True)
            return False

    async def async_shutdown(self) -> None:
        """Cleanup coordinator resources by Zara"""
        _LOGGER.info("Shutting down coordinator...")

        try:
            await self.production_time_calculator.stop_tracking()
            self.scheduled_tasks.cancel_listeners()
            _LOGGER.info("Coordinator shutdown completed")
        except Exception as e:
            _LOGGER.error(f"Error during coordinator shutdown: {e}", exc_info=True)

    async def _setup_power_peak_tracking(self) -> None:
        """Setup event listener for power peak tracking by Zara"""
        if not self.power_entity:
            _LOGGER.debug("No power entity configured - power peak tracking disabled")
            return
        
        current_peak_today = 0.0
        last_write_time = None
        
        async def power_state_changed(event):
            nonlocal current_peak_today, last_write_time
            
            new_state = event.data.get("new_state")
            if not new_state or new_state.state in [None, "unavailable", "unknown"]:
                return
            
            try:
                power_w = float(new_state.state)
                
                # Only process if higher than current peak
                if power_w > current_peak_today:
                    current_peak_today = power_w
                    
                    # Debounce: Max 1x per minute
                    now = dt_util.now()
                    if last_write_time is None or (now - last_write_time).total_seconds() > 60:
                        # Check if this is an all-time peak
                        all_time_peak = await self.data_manager.get_all_time_peak()
                        is_all_time = all_time_peak is None or power_w > all_time_peak
                        
                        # Save to daily_forecasts.json
                        await self.data_manager.save_power_peak(
                            power_w=power_w,
                            timestamp=now,
                            is_all_time=is_all_time
                        )
                        
                        last_write_time = now
                        
                        if is_all_time:
                            _LOGGER.info(f"NEW ALL-TIME POWER PEAK: {power_w:.2f}W")
                        else:
                            _LOGGER.debug(f"Daily power peak updated: {power_w:.2f}W")
                        
            except (ValueError, TypeError) as e:
                _LOGGER.debug(f"Invalid power state: {new_state.state} - {e}")
        
        # Register event listener
        from homeassistant.helpers.event import async_track_state_change_event
        async_track_state_change_event(
            self.hass,
            [self.power_entity],
            power_state_changed
        )
        
        _LOGGER.info(f"Power peak tracking enabled for {self.power_entity}")

    async def _async_update_data(self):
        """Fetch data from API endpoint by Zara"""
        try:
            # Initialize services on first update
            if not self._services_initialized:
                services_ok = await self._initialize_services()
                if not services_ok:
                    raise UpdateFailed("Failed to initialize services")
                    
            # Initialize forecast orchestrator
            await self._initialize_forecast_orchestrator()
            
            # Restart initialization check - verify forecast exists
            today_forecast = await self.data_manager.get_current_day_forecast()
            now_local = dt_util.now()

            if not today_forecast or not today_forecast.get("forecast_day", {}).get("locked"):
                if now_local.hour < 12:
                    _LOGGER.warning(
                        "System started without locked forecast (before 12:00) - "
                        "initiating recovery"
                    )
                    # Immediately start fallback process
                    await self._recovery_forecast_process(source="startup_recovery")
                else:
                    _LOGGER.warning(
                        "System started late without forecast (after 12:00) - "
                        "using current forecast (NOT morning baseline!)"
                    )
                    # Late startup - use current forecast
                    if self.data and "forecast_today" in self.data:
                        forecast_value = self.data.get("forecast_today")
                        await self.data_manager.save_daily_forecast(
                            prediction_kwh=forecast_value,
                            source=f"late_startup_{now_local.hour:02d}:{now_local.minute:02d}"
                        )
                        _LOGGER.warning(
                            f"Set forecast to current value: {forecast_value:.2f} kWh "
                            f"(not representative of morning prediction)"
                        )

            # Check weather service health
            if self.weather_service and not self.weather_service.get_health_status().get('healthy'):
                _LOGGER.warning("Weather service unhealthy, attempting recovery...")
                try:
                    await self.weather_service.force_update()
                except Exception as e:
                    _LOGGER.error(f"Weather service recovery failed: {e}")

            # Get current weather data
            current_weather = None
            hourly_forecast = None
            
            if self.weather_service:
                try:
                    current_weather = await self.weather_service.get_current_weather()
                    hourly_forecast = await self.weather_service.get_processed_hourly_forecast()
                    self._last_weather_update = dt_util.now()
                    
                    if not hourly_forecast or len(hourly_forecast) == 0:
                        _LOGGER.warning("Weather service returned no hourly forecast data")
                        
                except Exception as e:
                    _LOGGER.error(f"Error fetching weather data: {e}", exc_info=True)

            # Get external sensor data
            external_sensors = self.sensor_collector.collect_all_sensor_data_dict()

            # Generate forecast
            forecast = await self.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=self.learned_correction_factor
            )

            if not forecast:
                raise UpdateFailed("Forecast generation failed")

            _LOGGER.debug(f"Forecast data from orchestrator: {forecast}")

            # Update coordinator data
            result = {
                "forecast_today": forecast.get("today"),
                "forecast_tomorrow": forecast.get("tomorrow"),
                "forecast_day_after_tomorrow": forecast.get("day_after_tomorrow"),
                "hourly_forecast": forecast.get("hourly", []) if self.enable_hourly else [],
                "current_weather": current_weather,
                "external_sensors": external_sensors,
                # Production time data from tracker (using safe properties)
                "production_time": {
                    "active": self.production_time_calculator.is_active,
                    "duration_seconds": self.production_time_calculator.total_seconds,
                    "start_time": self.production_time_calculator.start_time,
                    "end_time": self.production_time_calculator.end_time
                },
                "peak_today": {
                    "power_w": getattr(self, '_peak_power_today', 0.0),
                    "at": getattr(self, '_peak_time_today', None)
                },
                "yield_today": {
                    "kwh": external_sensors.get("solar_yield_today"),
                    "sensor": self.solar_yield_today
                }
            }

            # Update sensor properties
            await self._update_sensor_properties(result)

            # Save forecasts to daily_forecasts.json
            await self._save_forecasts_to_storage(
                forecast_data={
                    "today": forecast.get("today"),
                    "tomorrow": forecast.get("tomorrow"),
                    "day_after_tomorrow": forecast.get("day_after_tomorrow"),
                    "best_hour": forecast.get("best_hour"),
                    "best_hour_kwh": forecast.get("best_hour_kwh"),
                },
                hourly_forecast=result.get("hourly_forecast", []),
            )

            # Update success tracking
            self._last_update_success_time = dt_util.now()

            # Mark sensors as ready after first successful update
            if not self._startup_sensors_ready:
                self._startup_sensors_ready = True
                _LOGGER.info("Sensor initialization complete")

            # ========================================================================
            # BATTERY MANAGEMENT: Update electricity prices (v8.3.0)
            # ========================================================================
            if self.electricity_service:
                try:
                    # Update prices once per day or if not cached
                    should_update = False
                    last_update = self.electricity_service.get_last_update()

                    if last_update is None:
                        should_update = True
                    else:
                        # Update if last update was more than 12 hours ago
                        hours_since_update = (dt_util.now() - last_update).total_seconds() / 3600
                        should_update = hours_since_update > 12

                    if should_update:
                        _LOGGER.debug("Fetching electricity prices from ENTSO-E...")
                        prices = await self.electricity_service.fetch_day_ahead_prices()
                        if prices:
                            _LOGGER.info(f"Electricity prices updated successfully: {len(prices.get('prices', []))} price points")
                        else:
                            _LOGGER.warning("Failed to fetch electricity prices")

                except Exception as e:
                    _LOGGER.error(f"Error updating electricity prices: {e}", exc_info=True)

            return result

        except UpdateFailed:
            raise
        except Exception as err:
            _LOGGER.error(f"Unexpected error updating data: {err}", exc_info=True)
            raise UpdateFailed(f"Error communicating with API: {err}")

    async def _update_sensor_properties(self, data: Dict[str, Any]) -> None:
        """Update coordinator properties used by sensors by Zara"""
        try:
            if data.get("hourly_forecast"):
                hourly = data["hourly_forecast"]
                next_hour = hourly[0] if len(hourly) > 0 else {}
                self.next_hour_pred = next_hour.get("production_kwh", 0.0)
            else:
                self.next_hour_pred = 0.0

            historical_calc = self.historical_calculator
            peak_time = await historical_calc.async_get_peak_production_time()
            self.peak_production_time_today = peak_time if peak_time else "Calculating..."

            prod_calc = self.production_time_calculator
            self.production_time_today = prod_calc.get_production_time()

            external = data.get("external_sensors", {})
            solar_yield_kwh = external.get("solar_yield_today")
            total_consumption_kwh = external.get("total_consumption_today")

            if solar_yield_kwh is not None and total_consumption_kwh is not None:
                try:
                    if total_consumption_kwh > 0:
                        self.autarky_today = (solar_yield_kwh / total_consumption_kwh) * 100
                    else:
                        self.autarky_today = 0.0
                except (ValueError, TypeError, ZeroDivisionError):
                    self.autarky_today = None
            else:
                self.autarky_today = None
        except (ValueError, TypeError, AttributeError) as e:
            _LOGGER.debug(f"Could not calculate autarky: {e}")
            self.autarky_today = None

        ml_predictor = self.ml_predictor
        if ml_predictor:
            self.last_successful_learning = getattr(ml_predictor, 'last_training_time', None)
            if self.model_accuracy is None:
                self.model_accuracy = getattr(ml_predictor, 'current_accuracy', None)
        else:
            self.last_successful_learning = None

        _LOGGER.debug("Coordinator sensor properties updated.")


    async def _save_forecasts_to_storage(
        self, forecast_data: dict, hourly_forecast: list
    ) -> None:
        """Save forecasts to daily_forecastsjson based on current time by Zara"""
        try:
            now_local = dt_util.now()
            hour = now_local.hour

            today_kwh = forecast_data.get("today")
            tomorrow_kwh = forecast_data.get("tomorrow")
            day_after_kwh = forecast_data.get("day_after_tomorrow")

            source = (
                "ML"
                if self.forecast_orchestrator.ml_strategy
                and self.forecast_orchestrator.ml_strategy.is_available()
                else "Weather"
            )

            # Always save tomorrow and day_after_tomorrow (unlocked by default)
            if tomorrow_kwh is not None:
                tomorrow_date = now_local + timedelta(days=1)
                await self.data_manager.save_forecast_tomorrow(
                    date=tomorrow_date,
                    prediction_kwh=tomorrow_kwh,
                    source=source,
                    lock=False, # Always save unlocked first
                )
                _LOGGER.debug(f"Updated tomorrow forecast: {tomorrow_kwh:.2f} kWh")

            if day_after_kwh is not None:
                day_after_date = now_local + timedelta(days=2)
                await self.data_manager.save_forecast_day_after(
                    date=day_after_date,
                    prediction_kwh=day_after_kwh,
                    source=source,
                    lock=False, # Always save unlocked first
                )
                _LOGGER.debug(f"Updated day after tomorrow forecast: {day_after_kwh:.2f} kWh")


            # Time-based saving and locking
            if 6 <= hour < 7:
                if today_kwh is not None:
                    await self.data_manager.save_forecast_today(
                        prediction_kwh=today_kwh, source=source
                    )
                    _LOGGER.info(f"Saved today forecast: {today_kwh:.2f} kWh")

                # --- BEST HOUR ---
                # Get best hour from forecast result (calculated by strategy)
                best_hour = forecast_data.get("best_hour")
                best_hour_kwh = forecast_data.get("best_hour_kwh")

                if best_hour is not None and best_hour_kwh is not None:
                    try:
                        await self.data_manager.save_forecast_best_hour(
                            hour=best_hour,
                            prediction_kwh=best_hour_kwh,
                            source=source,
                        )
                        _LOGGER.info(f"Saved best hour: {best_hour}:00 with {best_hour_kwh:.3f} kWh")
                    except Exception as e:
                        _LOGGER.warning(f"Could not save best hour: {e}")
                else:
                    _LOGGER.debug("Best hour not calculated in forecast - skipping")
            
            elif 12 <= hour < 13:
                # Lock tomorrow's forecast
                if tomorrow_kwh is not None:
                    tomorrow_date = now_local + timedelta(days=1)
                    await self.data_manager.save_forecast_tomorrow(
                        date=tomorrow_date,
                        prediction_kwh=tomorrow_kwh,
                        source=source,
                        lock=True,
                    )
                    _LOGGER.info(f"Locked tomorrow forecast: {tomorrow_kwh:.2f} kWh")
            
            elif 18 <= hour < 19:
                # Lock day after tomorrow's forecast
                if day_after_kwh is not None:
                    day_after_date = now_local + timedelta(days=2)
                    await self.data_manager.save_forecast_day_after(
                        date=day_after_date,
                        prediction_kwh=day_after_kwh,
                        source=source,
                        lock=True,
                    )
                    _LOGGER.info(f"Locked day after tomorrow forecast: {day_after_kwh:.2f} kWh")
            
        except Exception as e:
            _LOGGER.error(f"Failed to save forecasts to storage: {e}", exc_info=True)

    async def _update_next_hour_forecast(self) -> None:
        """Update next hour forecast - only during production time by Zara"""
        try:
            now_local = dt_util.now()
            # Calculate for the next full hour
            next_hour_start = (now_local + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

            # Check if next hour is within production time
            if not self.forecast_orchestrator.is_production_hour(next_hour_start):
                await self.data_manager.deactivate_next_hour_forecast()
                _LOGGER.debug(f"Next hour {next_hour_start.hour}:00 outside production - deactivated")
                return
            
            # Get today's forecast
            forecasts = await self.data_manager.load_daily_forecasts()
            forecast_today = forecasts.get("today", {}).get("forecast_day", {}).get("prediction_kwh")
            
            if not forecast_today or forecast_today <= 0:
                _LOGGER.debug("No valid today forecast for next hour calculation")
                await self.data_manager.deactivate_next_hour_forecast()
                return
            
            # Berechne Vorhersage
            current_weather = await self.weather_service.get_current_weather() if self.weather_service else None
            sensor_data = self.sensor_collector.collect_all_sensor_data_dict()
            
            next_hour_kwh = self.forecast_orchestrator.calculate_next_hour_prediction(
                forecast_today_kwh=forecast_today,
                weather_data=current_weather,
                sensor_data=sensor_data
            )
            
            next_hour_end = next_hour_start + timedelta(hours=1)
            await self.data_manager.save_forecast_next_hour(
                hour_start=next_hour_start,
                hour_end=next_hour_end,
                prediction_kwh=next_hour_kwh,
                source="ML_Hourly",
            )
            _LOGGER.info(f"Next hour forecast updated: {next_hour_start.hour}:00 = {next_hour_kwh:.3f} kWh")
            
        except Exception as e:
            _LOGGER.error(f"Next hour update failed: {e}", exc_info=True)

    async def _rotate_forecasts_midnight(self) -> None:
        """Rotate forecasts at midnight by Zara"""
        try:
            _LOGGER.info("Starting midnight forecast rotation...")
            
            success = await self.data_manager.rotate_forecasts_at_midnight()
            
            if success:
                _LOGGER.info("Midnight forecast rotation completed successfully")
            else:
                _LOGGER.error("Midnight forecast rotation failed")
            
        except Exception as e:
            _LOGGER.error(f"Failed to rotate forecasts at midnight: {e}", exc_info=True)

    @property
    def last_update_success_time(self) -> Optional[datetime]:
        return self._last_update_success_time

    @property
    def weather_source(self) -> str:
        return self.current_weather_entity or "Not configured"

    @property
    def diagnostic_status(self) -> str:
        if not self._startup_sensors_ready:
            return "Initializing (Waiting for sensors)"
        if not self.last_update_success and self._last_update_success_time is None:
            return "Error Initializing"
        elif not self.last_update_success:
            return "Update Failed"

        weather_healthy = False
        if self.weather_service:
            try:
                weather_healthy = self.weather_service.get_health_status().get('healthy', False)
            except Exception as e:
                _LOGGER.debug(f"Could not get weather service health status: {e}")

        update_age_ok = True
        if self._last_update_success_time:
            age = (dt_util.now() - self._last_update_success_time).total_seconds()
            if age > (self.update_interval.total_seconds() * 2):
                update_age_ok = False
        else:
            update_age_ok = False

        ml_active = self._ml_ready
        if ml_active and weather_healthy and update_age_ok:
            return "Optimal (ML Active)"
        elif weather_healthy and update_age_ok:
            reason = "ML Disabled/Unavailable" if not self.ml_predictor else "ML Not Ready"
            return f"Degraded ({reason})"
        elif not weather_healthy:
            return "Error (Weather Unavailable)"
        elif not update_age_ok:
            return "Stale (No Recent Update)"
        else:
            return "Initializing"

    def on_ml_training_complete(self, timestamp: datetime, accuracy: Optional[float] = None) -> None:
        _LOGGER.info(f"Coordinator notified of ML Training completion at {timestamp}. Accuracy: {accuracy}")
        self.last_successful_learning = timestamp
        if accuracy is not None:
            self.model_accuracy = accuracy
        self.async_update_listeners()

    async def set_expected_daily_production(self) -> None:
        """Set expected daily production at 6 AM and save persistently by Zara"""
        try:
            _LOGGER.info("=== Setting expected daily production (6 AM task) ===")
            
            # Option A: Use existing coordinator data if available
            if self.data and "forecast_today" in self.data and self.data.get("forecast_today") is not None:
                self.expected_daily_production = self.data.get("forecast_today")
                _LOGGER.info(
                    f"Expected daily production set to: {self.expected_daily_production:.2f} kWh "
                    f"(from existing coordinator data)"
                )
            else:
                # Option B: Force a fresh update to get current forecast
                _LOGGER.info("No forecast data available, forcing coordinator refresh...")
                
                await self.async_request_refresh()
                
                # Wait up to 10 seconds for forecast data to become available
                for i in range(10):
                    if self.data and "forecast_today" in self.data and self.data.get("forecast_today") is not None:
                        _LOGGER.debug(f"Forecast data available after {i+1} seconds")
                        break
                    await asyncio.sleep(1.0)
                else:
                    _LOGGER.warning("Forecast data not available after 10 seconds wait")
                
                # Check if refresh was successful
                if self.data and "forecast_today" in self.data and self.data.get("forecast_today") is not None:
                    self.expected_daily_production = self.data.get("forecast_today")
                    _LOGGER.info(
                        f"Expected daily production set to: {self.expected_daily_production:.2f} kWh "
                        f"(from forced refresh)"
                    )
                else:
                    _LOGGER.error("Failed to get forecast data even after forced refresh!")
                    self.expected_daily_production = None
            
            # Save to persistent storage (BOTH old and new system)
            # The 6 AM task has PRIORITY and can overwrite even locked forecasts
            if self.expected_daily_production is not None:
                _LOGGER.info(f"Saving {self.expected_daily_production:.2f} kWh to storage...")

                # Check if forecast is already locked (for logging purposes only)
                existing_forecast = await self.data_manager.get_current_day_forecast()
                if existing_forecast and existing_forecast.get("forecast_day", {}).get("locked"):
                    old_value = existing_forecast.get("forecast_day", {}).get("prediction_kwh")
                    old_source = existing_forecast.get("forecast_day", {}).get("source")
                    _LOGGER.warning(
                        f"6 AM TASK: Overwriting existing locked forecast "
                        f"(old: {old_value} kWh from {old_source}) with "
                        f"(new: {self.expected_daily_production:.2f} kWh from auto_6am)"
                    )

                # OLD system (coordinator_state.json) - for backward compatibility
                old_save_ok = await self.data_manager.save_expected_daily_production(
                    self.expected_daily_production
                )
                _LOGGER.info(f"OLD system save: {' OK' if old_save_ok else ' FAILED'}")

                # Save to daily forecasts - lock today's forecast with FORCE_OVERWRITE
                new_save_ok = await self.data_manager.save_daily_forecast(
                    prediction_kwh=self.expected_daily_production,
                    source="auto_6am",
                    force_overwrite=True  # 6 AM task can overwrite locked forecasts
                )

                _LOGGER.info(f"NEW system save: {' OK' if new_save_ok else ' FAILED'}")
                
                if not new_save_ok:
                    _LOGGER.error("CRITICAL: daily_forecasts.json NOT saved!")
                
                self.async_update_listeners()
                _LOGGER.info(
                    f" Expected daily production saved: {self.expected_daily_production:.2f} kWh"
                )
            else:
                _LOGGER.error(" Cannot save: expected_daily_production is None!")
            
        except Exception as err:
            _LOGGER.error(f" Failed to set expected daily production: {err}", exc_info=True)
            self.expected_daily_production = None

    async def reset_expected_daily_production(self) -> None:
        """Reset expected daily production at midnight and clear persistent storage by Zara"""
        self.expected_daily_production = None
        await self.data_manager.clear_expected_daily_production()
        _LOGGER.info("Expected daily production reset to None and cleared from persistent storage")
        self.async_update_listeners()

    async def _recovery_forecast_process(self, source: str) -> bool:
        """Fallback process for missing forecasts by Zara"""
        # Prevent concurrent recovery attempts with lock
        async with self._recovery_lock:
            if self._recovery_in_progress:
                _LOGGER.debug(f"Recovery already in progress, skipping duplicate attempt from {source}")
                return False

            self._recovery_in_progress = True
            try:
                return await self._execute_recovery(source)
            finally:
                self._recovery_in_progress = False

    async def _execute_recovery(self, source: str) -> bool:
        """Internal method to execute the recovery process by Zara"""
        _LOGGER.info(f"=== Starting recovery forecast process (source: {source}) ===")

        # Early validation: Check if weather service is available
        if not self.weather_service:
            _LOGGER.error(
                "Weather service not initialized - cannot execute any recovery fallback. "
                "Please check weather entity configuration."
            )
            return False

        # Double-check if forecast was already set by another task
        existing_forecast = await self.data_manager.get_current_day_forecast()
        if existing_forecast and existing_forecast.get("forecast_day", {}).get("locked"):
            _LOGGER.info(
                f"Forecast already set (locked) - cancelling recovery. "
                f"Value: {existing_forecast.get('forecast_day', {}).get('prediction_kwh')} kWh"
            )
            return True  # Not an error - forecast exists

        # FALLBACK 1: Weather cache reconstruction
        try:
            cache = await self.data_manager.load_weather_cache()
            if cache and cache.get("forecast_hours"):
                _LOGGER.info("Weather cache available - attempting forecast reconstruction")

                # Hole externe Sensoren
                external_sensors = self.sensor_collector.collect_all_sensor_data_dict()

                forecast = await self.forecast_orchestrator.orchestrate_forecast(
                    current_weather=None,  # Will be taken from cache
                    hourly_forecast=cache.get("forecast_hours"),
                    external_sensors=external_sensors,
                    correction_factor=self.learned_correction_factor
                )

                if forecast and forecast.get("today") is not None:
                    success = await self.data_manager.save_daily_forecast(
                        prediction_kwh=forecast["today"],
                        source=f"fallback_weather_cache_{source}"
                    )
                    if success:
                        _LOGGER.info(
                            f"✓ Forecast set using weather cache: {forecast['today']:.2f} kWh"
                        )
                        return True
        except Exception as e:
            _LOGGER.warning(f"Weather cache fallback failed: {e}")

        # FALLBACK 2: Rule-Based with fresh weather data
        try:
            _LOGGER.info("Weather cache unavailable - using rule-based fallback with fresh data")

            current_weather = await self.weather_service.get_current_weather()
            hourly_forecast = await self.weather_service.get_processed_hourly_forecast()
            external_sensors = self.sensor_collector.collect_all_sensor_data_dict()
            
            forecast = await self.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,  # Kein ML
                ml_prediction_tomorrow=None,
                correction_factor=self.learned_correction_factor
            )
            
            if forecast and forecast.get("today") is not None:
                success = await self.data_manager.save_daily_forecast(
                    prediction_kwh=forecast["today"],
                    source=f"fallback_rule_based_{source}"
                )
                if success:
                    _LOGGER.info(
                        f" Forecast set using rule-based: {forecast['today']:.2f} kWh"
                    )
                    return True
        except Exception as e:
            _LOGGER.error(f"Rule-based fallback failed: {e}", exc_info=True)
        
        _LOGGER.error("All fallback methods failed - unable to set forecast")
        return False

    async def force_refresh_with_weather_update(self) -> None:
        """Force refresh with immediate weather update by Zara"""
        _LOGGER.info("Force refresh requested - updating weather first...")

        # Force weather service to fetch fresh data
        if self.weather_service:
            success = await self.weather_service.force_update()
            if success:
                _LOGGER.info("Weather data successfully refreshed")
            else:
                _LOGGER.warning("Weather refresh failed - continuing with cached data")

        # Now trigger normal coordinator refresh
        await self.async_request_refresh()

        _LOGGER.info("Force refresh completed")

    async def forecast_day_after_tomorrow(self) -> None:
        """Triggers and saves the forecast for the day after tomorrow by Zara"""
        try:
            _LOGGER.info("Service call: Manually triggering forecast for day after tomorrow.")

            # 1. Get weather and sensor data
            current_weather = await self.weather_service.get_current_weather() if self.weather_service else None
            hourly_forecast = await self.weather_service.get_processed_hourly_forecast() if self.weather_service else None
            external_sensors = self.sensor_collector.collect_all_sensor_data_dict()

            # 2. Generate forecast
            forecast = await self.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=self.learned_correction_factor
            )

            if not forecast or forecast.get("day_after_tomorrow") is None:
                _LOGGER.error("Failed to generate forecast for the day after tomorrow.")
                return

            day_after_kwh = forecast.get("day_after_tomorrow")
            now_local = dt_util.now()
            day_after_date = now_local + timedelta(days=2)
            source = "manual_service"

            # 3. Save the forecast
            await self.data_manager.save_forecast_day_after(
                date=day_after_date,
                prediction_kwh=day_after_kwh,
                source=source,
                lock=True,  # Or False, depending on desired behavior
            )

            _LOGGER.info(f"Successfully saved forecast for day after tomorrow: {day_after_kwh:.2f} kWh")

            # 4. Trigger a coordinator update to reflect changes in sensors
            await self.async_request_refresh()

        except Exception as e:
            _LOGGER.error(f"Error in forecast_day_after_tomorrow service: {e}", exc_info=True)
        
        