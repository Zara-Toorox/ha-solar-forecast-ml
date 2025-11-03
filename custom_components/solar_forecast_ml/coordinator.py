"""
Data Update Coordinator for Solar Forecast ML Integration.
Orchestrates data fetching, processing, and state updates using a modular architecture.

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
    CONF_FALLBACK_ENTITY
)
from .data.data_manager import DataManager
from .exceptions import SolarForecastMLException, WeatherAPIException, MLModelException
from .core.helpers import SafeDateTimeUtil as dt_util

# Import modular components
from .forecast.weather_calculator import WeatherCalculator
from .production.history import ProductionCalculator as HistoricalProductionCalculator
from .production.tracker import ProductionTimeCalculator
from .sensors.data_collector import SensorDataCollector
from .forecast.orchestrator import ForecastOrchestrator
from .production.scheduled_tasks import ScheduledTasksManager
from .ml.ml_predictor import ModelState, MLPredictor
from .services.service_error_handler import ErrorHandlingService
from .forecast.forecast_weather import WeatherService

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
        self.historical_calculator = HistoricalProductionCalculator(hass)
        self.production_time_calculator = ProductionTimeCalculator(hass=hass, power_entity=self.power_entity)
        self.forecast_orchestrator = ForecastOrchestrator(
            hass=hass,
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
        self._last_prediction_time: Optional[datetime] = None
        self._last_prediction_value: Optional[float] = None
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

        _LOGGER.info(f"SolarForecastMLCoordinator initialized - Using Weather Entity: {self.primary_weather_entity or 'None'}")

    async def _load_persistent_state(self) -> None:
        """Load persistent coordinator state (e.g., expected_daily_production)."""
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
        """Initialize all services (weather, ML, error handler)."""
        try:
            # Error handler is already initialized
            
            # Initialize WeatherService
            if self.current_weather_entity:
                self.weather_service = WeatherService(
                    hass=self.hass,
                    weather_entity=self.current_weather_entity,
                    data_manager=self.data_manager,
                    error_handler=self.error_handler
                )
                await self.weather_service.initialize()
                _LOGGER.info("WeatherService initialized")
            
            # Initialize MLPredictor if learning enabled
            if self.learning_enabled and self.dependencies_ok:
                self.ml_predictor = MLPredictor(
                    hass=self.hass,
                    data_manager=self.data_manager,
                    error_handler=self.error_handler,
                    notification_service=None
                )
                await self.ml_predictor.initialize()
                
                # Set peak power (FIX 1: update_strategies() wird NICHT hier aufgerufen)
                if hasattr(self.ml_predictor, 'peak_power_kw'):
                    self.ml_predictor.peak_power_kw = self.solar_capacity
                
                # CRITICAL FIX: Configure entities for sample collection
                self.ml_predictor.set_entities(
                    solar_capacity=self.solar_capacity,
                    power_entity=self.power_entity,
                    weather_entity=self.primary_weather_entity,
                    temp_sensor=self.sensor_collector.get_sensor_entity_id('temperature'),
                    wind_sensor=self.sensor_collector.get_sensor_entity_id('wind_speed'),
                    rain_sensor=self.sensor_collector.get_sensor_entity_id('rain'),
                    uv_sensor=self.sensor_collector.get_sensor_entity_id('uv_index'),
                    lux_sensor=self.sensor_collector.get_sensor_entity_id('lux'),
                    humidity_sensor=self.sensor_collector.get_sensor_entity_id('humidity')
                )
                _LOGGER.info("MLPredictor entities configured for sample collection")
                
                self._ml_ready = True
                _LOGGER.info("MLPredictor initialized")
            
            # Start production time tracking
            if self.production_time_calculator:
                try:
                    self.production_time_calculator.start_tracking()
                    _LOGGER.info("Production time tracking started")
                except Exception as track_err:
                    _LOGGER.warning(f"Failed to start production time tracking (non-critical): {track_err}")
            
            self._services_initialized = True
            return True
            
        except Exception as e:
            _LOGGER.error(f"Service initialization failed: {e}")
            return False

    async def _initialize_forecast_orchestrator(self) -> None:
        """Initialize forecast strategies after services are ready."""
        try:
            self.forecast_orchestrator.initialize_strategies(
                ml_predictor=self.ml_predictor if self._ml_ready else None,
                error_handler=self.error_handler
            )
            _LOGGER.info("Forecast strategies initialized successfully")
        except Exception as e:
            _LOGGER.error(f"Forecast strategy initialization failed: {e}", exc_info=True)

    async def _get_yesterday_accuracy(self) -> tuple[Optional[float], Optional[float]]:
        """Calculate yesterday accuracy from prediction history."""
        try:
            history = await self.data_manager.get_prediction_history()
            yesterday = (dt_util.now() - timedelta(days=1)).date()  # now() already returns local time
            
            for pred in reversed(history.get('predictions', [])):
                ts = dt_util.parse_datetime(pred.get('timestamp'))
                if ts and dt_util.as_local(ts).date() == yesterday:
                    if pred.get('actual_value'):
                        error = abs(pred['predicted_value'] - pred['actual_value'])
                        accuracy = max(0.0, 100.0 - (error / pred['actual_value'] * 100)) if pred['actual_value'] > 0 else None
                        return error, accuracy
        except Exception as e:
            _LOGGER.warning(f"Yesterday accuracy failed: {e}")
        return None, None

    async def _check_entity_available(self, entity_id: str) -> bool:
        if not entity_id:
            return False
        state = self.hass.states.get(entity_id)
        if state is None:
            _LOGGER.debug(f"Wait-Check: Entity {entity_id} not found (None).")
            return False
        if state.state in [STATE_UNAVAILABLE, STATE_UNKNOWN, "None", None, ""]:
            _LOGGER.debug(f"Wait-Check: Entity {entity_id} is {state.state}.")
            return False
        if state.domain == "weather" and not state.attributes:
            _LOGGER.debug(f"Wait-Check: Weather entity {entity_id} has no attributes yet.")
            return False
        return True

    async def _wait_for_critical_entities(self, timeout: int = 120) -> bool:
        """Wait for critical entities to become available at startup."""
        required_entities = []
        
        if self.power_entity:
            required_entities.append(("Power Entity", self.power_entity))
        if self.solar_yield_today:
            required_entities.append(("Solar Yield Entity", self.solar_yield_today))
        if self.primary_weather_entity:
            required_entities.append(("Weather Entity", self.primary_weather_entity))
        
        if not required_entities:
            _LOGGER.warning("No critical entities configured - continuing without entity checks.")
            return True

        _LOGGER.info(f"Waiting for {len(required_entities)} critical entities (max {timeout}s)...")
        start_time = dt_util.now()
        
        while (dt_util.now() - start_time).total_seconds() < timeout:
            all_available = True
            for name, entity_id in required_entities:
                if not await self._check_entity_available(entity_id):
                    all_available = False
                    break
            
            if all_available:
                wait_duration = (dt_util.now() - start_time).total_seconds()
                _LOGGER.info(f"All critical entities available after {wait_duration:.1f}s.")
                return True
            
            await asyncio.sleep(2)
        
        _LOGGER.warning(f"Timeout waiting for entities after {timeout}s. Continuing anyway.")
        return False

    async def _async_update_data(self) -> Dict[str, Any]:
        """Fetch and process all data sources."""
        update_start_time = dt_util.now()
        _LOGGER.debug("Starting data update cycle...")

        try:
            # 1. Startup entity wait (only once)
            if not self._startup_sensors_ready:
                _LOGGER.info("First update - waiting for entities...")
                entities_ready = await self._wait_for_critical_entities(timeout=120)
                self._startup_sensors_ready = True
                
                if not entities_ready:
                    _LOGGER.warning("Not all entities ready, but continuing with initialization.")

            # 2. Service initialization (only once)
            if not self._services_initialized:
                _LOGGER.info("Initializing managed services...")
                services_ok = await self._initialize_services()
                if not services_ok:
                    _LOGGER.error("Service initialization incomplete - some features may be unavailable.")
                
                # Load persistent coordinator state (expected_daily_production)
                await self._load_persistent_state()
                
                # CRITICAL FIX 1: Ensure solar_capacity is properly set in ML predictor
                if self._ml_ready and self.ml_predictor:
                    if hasattr(self.ml_predictor, 'peak_power_kw'):
                        if self.ml_predictor.peak_power_kw == 0.0 and self.solar_capacity > 0:
                            _LOGGER.warning(
                                f"ML Predictor peak_power_kw not set! "
                                f"Setting from config: {self.solar_capacity} kW"
                            )
                            self.ml_predictor.peak_power_kw = float(self.solar_capacity)
                            
                            # FIX 1: SOFORT update_strategies() aufrufen
                            if hasattr(self.ml_predictor, 'prediction_orchestrator'):
                                self.ml_predictor.prediction_orchestrator.update_strategies(
                                    weights=self.ml_predictor.current_weights,
                                    profile=self.ml_predictor.current_profile,
                                    accuracy=self.ml_predictor.current_accuracy or 0.0,
                                    peak_power_kw=self.ml_predictor.peak_power_kw
                                )
                                _LOGGER.info(f"Prediction strategies updated with corrected peak_power_kw: {self.ml_predictor.peak_power_kw} kW")
                        else:
                            _LOGGER.info(
                                f"ML Predictor peak_power_kw verified: {self.ml_predictor.peak_power_kw} kW"
                            )
                
                # Initialize forecast strategies after services
                await self._initialize_forecast_orchestrator()

            if not self.weather_service:
                raise UpdateFailed("Weather service unavailable.")

            error_yesterday, accuracy_yesterday = await self._get_yesterday_accuracy()
            if error_yesterday is not None:
                self.last_day_error_kwh = error_yesterday
            if accuracy_yesterday is not None:
                self.yesterday_accuracy = accuracy_yesterday

            current_weather_data = await self.weather_service.get_current_weather()
            if not current_weather_data or not isinstance(current_weather_data, dict):
                raise UpdateFailed("Weather data unavailable or invalid format.")

            hourly_forecast_data = await self.weather_service.get_processed_hourly_forecast()

            historical_avg = None
            try:
                historical_avg = await self.historical_calculator.get_historical_average()
            except Exception as hist_err:
                _LOGGER.warning(f"Could not fetch historical average: {hist_err}")

            ml_pred_today_value = None
            ml_pred_tomorrow_value = None
            if self._ml_ready and self.ml_predictor:
                try:
                    ml_pred_today_value = await self.ml_predictor.get_today_prediction()
                    ml_pred_tomorrow_value = await self.ml_predictor.get_tomorrow_prediction()
                    _LOGGER.debug(f"ML predictions: Today={ml_pred_today_value}, Tomorrow={ml_pred_tomorrow_value}")
                except Exception as ml_err:
                    _LOGGER.warning(f"ML predictions unavailable: {ml_err}")

            external_sensor_readings = self.sensor_collector.collect_all_sensor_data_dict()

            forecast_result = await self.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather_data,
                hourly_forecast=hourly_forecast_data,
                external_sensors=external_sensor_readings,
                historical_avg=historical_avg,
                ml_prediction_today=ml_pred_today_value,
                ml_prediction_tomorrow=ml_pred_tomorrow_value,
                correction_factor=self.learned_correction_factor
            )

            today_forecast_value = forecast_result.get("today", 0.0)
            tomorrow_forecast_value = forecast_result.get("tomorrow", 0.0)
            
            # Calculate remaining forecast for today (forecast_today minus already produced)
            forecast_today_remaining = today_forecast_value
            if self.solar_yield_today:
                try:
                    yield_state = self.hass.states.get(self.solar_yield_today)
                    if yield_state and yield_state.state not in (None, "unknown", "unavailable"):
                        already_produced = float(yield_state.state)
                        forecast_today_remaining = max(0.0, today_forecast_value - already_produced)
                        _LOGGER.debug(
                            f"Remaining forecast: {forecast_today_remaining:.2f} kWh "
                            f"(Total: {today_forecast_value:.2f} - Produced: {already_produced:.2f})"
                        )
                except (ValueError, TypeError) as e:
                    _LOGGER.warning(f"Could not read solar yield for remaining calculation: {e}")
            
            # Prepare return data IMMEDIATELY after forecast calculation
            # This ensures we always return data even if post-processing fails
            final_data = {
                "forecast_today": forecast_today_remaining,  # NOW shows remaining, not total
                "forecast_tomorrow": tomorrow_forecast_value,
                "last_update": dt_util.now().isoformat(),
                "_forecast_method": forecast_result.get("method", "unknown"),
                "_model_accuracy": forecast_result.get("model_accuracy"),
                "_confidence_today": forecast_result.get("confidence"),
                "_weather_entity_used": self.current_weather_entity,
                "_update_duration_sec": 0.0,
                "_total_forecast_today": today_forecast_value,  # Store original total for reference
                "_already_produced_today": today_forecast_value - forecast_today_remaining  # Already produced
            }
            
            # Try to calculate next hour prediction (non-critical)
            try:
                current_weather_for_next_hour = current_weather_data
                sensor_data_for_next_hour = external_sensor_readings
                
                _LOGGER.debug(
                    f"Calculating next hour prediction - forecast_today={today_forecast_value:.2f} kWh, "
                    f"weather_available={current_weather_for_next_hour is not None}, "
                    f"sensors_available={sensor_data_for_next_hour is not None}"
                )
                
                next_hour_prediction_kwh = self.forecast_orchestrator.calculate_next_hour_prediction(
                    forecast_today_kwh=today_forecast_value,
                    weather_data=current_weather_for_next_hour,
                    sensor_data=sensor_data_for_next_hour
                )
                self.next_hour_pred = next_hour_prediction_kwh
                
                _LOGGER.debug(f"Next hour prediction result: {next_hour_prediction_kwh:.3f} kWh")
                
            except Exception as nhe:
                _LOGGER.warning(f"Next hour prediction failed (non-critical): {nhe}", exc_info=True)
                self.next_hour_pred = 0.0

            # Try to update sensor properties (non-critical)
            try:
                await self._update_sensor_properties(forecast_result)
            except Exception as use:
                _LOGGER.warning(f"Sensor properties update failed (non-critical): {use}")

            # Try to save prediction record (non-critical)
            try:
                if not await self._should_skip_prediction_storage(today_forecast_value):
                    local_now = dt_util.now()  # Already returns local time
                    prediction_record = {
                        'timestamp': local_now.isoformat(),
                        'predicted_value': today_forecast_value,
                        'actual_value': None,
                        'weather_data': current_weather_data or {},
                        'sensor_data': external_sensor_readings or {},
                        'accuracy': max(0.0, min(1.0, float(forecast_result.get('model_accuracy') or 0.0))),
                        'model_version': forecast_result.get('model_version', 'unknown'),
                        'weather_entity': self.current_weather_entity,
                        'method': forecast_result.get('method', 'unknown')
                    }
                    await self.data_manager.add_prediction_record(prediction_record)
                    _LOGGER.debug(f"Prediction record saved: {today_forecast_value:.2f} kWh")
            except Exception as store_err:
                _LOGGER.warning(f"Failed to save prediction record (non-critical): {store_err}")

            # Update final timing
            self._last_update_success_time = dt_util.now()
            update_duration = (self._last_update_success_time - update_start_time).total_seconds()
            final_data["last_update"] = self._last_update_success_time.isoformat()
            final_data["_update_duration_sec"] = round(update_duration, 2)
            
            # Auto-set expected_daily_production ONLY on first set (not after restart)
            # Persistent value takes priority - only set if truly None AND not yet saved today
            if self.expected_daily_production is None and today_forecast_value is not None:
                # Check if we already have a saved value from today (double-check)
                loaded_check = await self.data_manager.load_expected_daily_production()
                if loaded_check is None:
                    # Truly first forecast of the day - save it
                    self.expected_daily_production = today_forecast_value
                    await self.data_manager.save_expected_daily_production(today_forecast_value)
                    _LOGGER.info(
                        f"Auto-setting expected_daily_production to {today_forecast_value:.2f} kWh "
                        f"(first forecast of the day)"
                    )
                else:
                    # We have a saved value but it wasn't loaded - reload it
                    self.expected_daily_production = loaded_check
                    _LOGGER.info(
                        f"Restored expected_daily_production from storage: {loaded_check:.2f} kWh "
                        f"(after restart, NOT overwriting with current forecast)"
                    )
            
            _LOGGER.info(
                f"Data update completed in {update_duration:.2f}s: "
                f"Today={today_forecast_value:.2f} kWh, Tomorrow={tomorrow_forecast_value:.2f} kWh"
            )
            
            return final_data

        except UpdateFailed as uf_err:
            _LOGGER.warning(f"Update failed: {uf_err}")
            raise
        except Exception as e:
            _LOGGER.error(f"Unexpected error during data update: {e}", exc_info=True)
            raise UpdateFailed(f"Unexpected update error: {e}") from e

    async def _update_sensor_properties(self, forecast_result: Dict[str, Any]) -> None:
        """Update coordinator properties that sensors read directly."""
        self.peak_production_time_today = forecast_result.get("peak_time", "12:00")
        if forecast_result.get("model_accuracy") is not None:
            self.model_accuracy = forecast_result["model_accuracy"]
        if self.production_time_calculator:
            self.production_time_today = self.production_time_calculator.get_production_time()
        else:
            self.production_time_today = "Not available"

        try:
            if self.data_manager:
                avg_yield = await self.data_manager.get_average_monthly_yield()
                self.avg_month_yield = avg_yield if avg_yield is not None else 0.0
            else:
                self.avg_month_yield = 0.0
        except Exception as e:
            _LOGGER.warning(f"Error calculating average monthly yield: {e}")
            self.avg_month_yield = 0.0

        ml_predictor = self.ml_predictor
        if ml_predictor:
            self.last_successful_learning = getattr(ml_predictor, 'last_training_time', None)
            if self.model_accuracy is None:
                self.model_accuracy = getattr(ml_predictor, 'current_accuracy', None)
        else:
            self.last_successful_learning = None

        _LOGGER.debug("Coordinator sensor properties updated.")

    async def _should_skip_prediction_storage(self, prediction_value: Optional[float]) -> bool:
        """Check if we should skip storing this prediction (duplicate prevention)."""
        if prediction_value is None:
            return True
            
        now = dt_util.now()
        
        if (self._last_prediction_time and 
            self._last_prediction_value is not None and
            abs(prediction_value - self._last_prediction_value) < 0.01):
            
            time_since_last = (now - self._last_prediction_time).total_seconds()
            if time_since_last < 300:  # 5 minutes
                return True
        
        self._last_prediction_time = now
        self._last_prediction_value = prediction_value
        return False

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
            except:
                pass

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
        """Set expected daily production at 6 AM and save persistently."""
        try:
            _LOGGER.info("Setting expected daily production (6 AM task)...")
            
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
                
                # Wait a moment for refresh to complete
                await asyncio.sleep(0.5)
                
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
            
            # Save to persistent storage
            if self.expected_daily_production is not None:
                await self.data_manager.save_expected_daily_production(self.expected_daily_production)
                self.async_update_listeners()
                _LOGGER.info(
                    f"Expected daily production saved to persistent storage: "
                    f"{self.expected_daily_production:.2f} kWh (survives restarts)"
                )
            
        except Exception as err:
            _LOGGER.error(f"Failed to set expected daily production: {err}", exc_info=True)
            self.expected_daily_production = None

    async def reset_expected_daily_production(self) -> None:
        """Reset expected daily production at midnight and clear persistent storage."""
        self.expected_daily_production = None
        await self.data_manager.clear_expected_daily_production()
        _LOGGER.info("Expected daily production reset to None and cleared from persistent storage")
        self.async_update_listeners()

    async def force_refresh_with_weather_update(self) -> None:
        """
        Force refresh with immediate weather update.
        Called by manual forecast button to ensure fresh weather data.
        """
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

