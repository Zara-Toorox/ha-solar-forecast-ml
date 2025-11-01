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
from .data.manager import DataManager
from .exceptions import SolarForecastMLException, WeatherAPIException, MLModelException
from .core.helpers import SafeDateTimeUtil as dt_util

# Import modular components
from .services.manager import ServiceManager
from .forecast.weather_calculator import WeatherCalculator
from .production.history import ProductionCalculator as HistoricalProductionCalculator
from .production.tracker import ProductionTimeCalculator
from .sensors.data_collector import SensorDataCollector
from .forecast.orchestrator import ForecastOrchestrator
from .production.scheduled_tasks import ScheduledTasksManager
from .ml.predictor import ModelState

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

        self.solar_capacity = float(entry.data.get(CONF_SOLAR_CAPACITY, DEFAULT_SOLAR_CAPACITY))
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

        self.service_manager = ServiceManager(
            hass=hass,
            entry=entry,
            data_manager=self.data_manager,
            weather_entity=self.current_weather_entity,
            dependencies_ok=dependencies_ok,
            power_entity=self.power_entity,
            solar_yield_today=self.solar_yield_today,
            solar_capacity=self.solar_capacity,
            temp_sensor=self.sensor_collector.strip_entity_id(entry.data.get(CONF_TEMP_SENSOR)),
            wind_sensor=self.sensor_collector.strip_entity_id(entry.data.get(CONF_WIND_SENSOR)),
            rain_sensor=self.sensor_collector.strip_entity_id(entry.data.get(CONF_RAIN_SENSOR)),
            uv_sensor=self.sensor_collector.strip_entity_id(entry.data.get(CONF_UV_SENSOR)),
            lux_sensor=self.sensor_collector.strip_entity_id(entry.data.get(CONF_LUX_SENSOR)),
            humidity_sensor=self.sensor_collector.strip_entity_id(entry.data.get(CONF_HUMIDITY_SENSOR)),
        )

        self.scheduled_tasks = ScheduledTasksManager(
            hass=hass,
            coordinator=self,
            solar_yield_today_entity_id=self.solar_yield_today,
            data_manager=self.data_manager
        )

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

    async def _initialize_forecast_orchestrator(self) -> None:
        """Initialize forecast strategies after services are ready."""
        try:
            self.forecast_orchestrator.initialize_strategies(
                ml_predictor=self.service_manager.ml_predictor if self.service_manager.is_ml_ready() else None,
                error_handler=self.service_manager.error_handler
            )
            _LOGGER.info("Forecast strategies initialized successfully")
        except Exception as e:
            _LOGGER.error(f"Forecast strategy initialization failed: {e}", exc_info=True)

    async def _get_yesterday_accuracy(self) -> tuple[Optional[float], Optional[float]]:
        """Calculate yesterday accuracy from prediction history."""
        try:
            history = await self.data_manager.get_prediction_history()
            yesterday = (dt_util.as_local(dt_util.now()) - timedelta(days=1)).date()
            
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
            if not self.service_manager.is_initialized():
                _LOGGER.info("Initializing managed services...")
                services_ok = await self.service_manager.initialize_all_services()
                if not services_ok:
                    _LOGGER.error("Service initialization incomplete - some features may be unavailable.")
                
                # CRITICAL: Ensure solar_capacity is properly set in ML predictor
                # Safety check in case ServiceManager doesn't set it during initialization
                if self.service_manager.is_ml_ready():
                    ml_predictor = self.service_manager.ml_predictor
                    if hasattr(ml_predictor, 'peak_power_kw'):
                        if ml_predictor.peak_power_kw == 0.0 and self.solar_capacity > 0:
                            _LOGGER.warning(
                                f"ML Predictor peak_power_kw not set! "
                                f"Setting from config: {self.solar_capacity} kW"
                            )
                            ml_predictor.peak_power_kw = float(self.solar_capacity)
                        else:
                            _LOGGER.info(
                                f"ML Predictor peak_power_kw verified: {ml_predictor.peak_power_kw} kW"
                            )
                
                # Initialize forecast strategies after services
                await self._initialize_forecast_orchestrator()

            # 3. External sensor data
            try:
                external_sensor_data = self.sensor_collector.collect_all_sensor_data_dict()
                _LOGGER.debug(f"External sensor data collected: {len(external_sensor_data)} sensors")
            except AttributeError as e:
                _LOGGER.error(f"Sensor collector method mismatch: {e} - using empty dict")
                external_sensor_data = {}
            except Exception as e:
                _LOGGER.error(f"Failed to collect external sensor data: {e} - using empty dict")
                external_sensor_data = {}

            # 4. Weather data
            current_weather_data = None
            hourly_forecast_list = []
            
            if self.service_manager.weather_service:
                try:
                    current_weather_data = await self.service_manager.weather_service.get_current_weather()
                    if current_weather_data:
                        _LOGGER.debug("Current weather data obtained.")
                    else:
                        _LOGGER.warning("Current weather unavailable - using defaults.")
                        
                    hourly_forecast_list = await self.service_manager.weather_service.try_get_forecast(
                        timeout=10
                    )
                    if hourly_forecast_list:
                        _LOGGER.debug(f"Hourly forecast obtained: {len(hourly_forecast_list)} hours")
                    else:
                        _LOGGER.warning("Hourly forecast unavailable.")
                        
                except Exception as weather_err:
                    _LOGGER.error(f"Weather data fetch failed: {weather_err}", exc_info=True)
            else:
                _LOGGER.warning("Weather Service not initialized - using default weather data.")

            # 5. Historical production (7-day average)
            historical_production_avg = None
            try:
                if self.solar_yield_today and self.historical_calculator:
                    # FIX: ProductionCalculator hat keine calculate_historical_production() Methode
                    # Verwende stattdessen get_last_7_days_average_yield()
                    historical_production_avg = await self.historical_calculator.get_last_7_days_average_yield(
                        yield_entity=self.solar_yield_today
                    )
                    if historical_production_avg is not None and historical_production_avg > 0:
                        _LOGGER.debug(f"7-day avg production: {historical_production_avg:.2f} kWh")
                    else:
                        _LOGGER.debug("Historical production: No data available (Recorder-free mode)")
            except Exception as hist_err:
                _LOGGER.warning(f"Historical calculation failed: {hist_err}")

            # 6. ML prediction (if available) - FIXED VERSION
            ml_prediction_today = None
            ml_prediction_tomorrow = None
            
            if self.service_manager.is_ml_ready():
                try:
                    # FIX: MLPredictor hat keine predict_today() Methode
                    # Verwende predict() mit Peak-Stunde (12:00) als Indikator
                    
                    now_local = dt_util.as_local(dt_util.now())
                    today_noon = now_local.replace(hour=12, minute=0, second=0, microsecond=0)
                    
                    # Prediction fÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼r heute: Nur Peak-Stunde (12:00) * 10 als grobe TagesschÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¤tzung
                    noon_result = await self.service_manager.ml_predictor.predict(
                        weather_data=current_weather_data if current_weather_data else {},
                        prediction_hour=12,
                        prediction_date=today_noon,
                        sensor_data=external_sensor_data
                    )
                    
                    # Hochrechnen: Peak-Stunde * 10 (grobe SchÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¤tzung fÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼r Tagesertrag)
                    if noon_result and noon_result.prediction > 0:
                        ml_prediction_today = noon_result.prediction * 10
                        _LOGGER.debug(f"ML prediction today (from noon peak): {ml_prediction_today:.2f} kWh")
                    
                    # Prediction fÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼r morgen
                    if hourly_forecast_list and len(hourly_forecast_list) >= 36:
                        tomorrow_noon = today_noon + timedelta(days=1)
                        tomorrow_forecast = hourly_forecast_list[36] if len(hourly_forecast_list) > 36 else {}
                        
                        tomorrow_result = await self.service_manager.ml_predictor.predict(
                            weather_data=tomorrow_forecast,
                            prediction_hour=12,
                            prediction_date=tomorrow_noon,
                            sensor_data=external_sensor_data
                        )
                        
                        if tomorrow_result and tomorrow_result.prediction > 0:
                            ml_prediction_tomorrow = tomorrow_result.prediction * 10
                            _LOGGER.debug(f"ML prediction tomorrow (from noon peak): {ml_prediction_tomorrow:.2f} kWh")
                    else:
                        _LOGGER.debug("Insufficient forecast data for tomorrow's ML prediction")
                        
                except Exception as ml_err:
                    _LOGGER.warning(f"ML prediction failed: {ml_err}")
            else:
                _LOGGER.debug("ML predictor not ready - using fallback strategy.")

            # 7. Orchestrate forecast
            forecast_result = await self.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather_data,
                hourly_forecast=hourly_forecast_list,
                external_sensors=external_sensor_data,
                historical_avg=historical_production_avg,
                ml_prediction_today=ml_prediction_today,
                ml_prediction_tomorrow=ml_prediction_tomorrow
            )
            
            _LOGGER.info(
                f"Forecast: Today={forecast_result.get('today', 0.0):.2f} kWh, "
                f"Tomorrow={forecast_result.get('tomorrow', 0.0):.2f} kWh, "
                f"Method={forecast_result.get('method', 'unknown')}"
            )

            # 8. Update sensor properties
            await self._update_sensor_properties(forecast_result)

            # 9. Autarky calculation
            if self.total_consumption_today and self.solar_yield_today:
                try:
                    consumption_state = self.hass.states.get(self.total_consumption_today)
                    yield_state = self.hass.states.get(self.solar_yield_today)
                    
                    if (consumption_state and yield_state and 
                        consumption_state.state not in [STATE_UNAVAILABLE, STATE_UNKNOWN] and
                        yield_state.state not in [STATE_UNAVAILABLE, STATE_UNKNOWN]):
                        
                        consumption_kwh = float(consumption_state.state)
                        yield_kwh = float(yield_state.state)
                        
                        if consumption_kwh > 0:
                            self.autarky_today = min(100.0, (yield_kwh / consumption_kwh) * 100.0)
                            _LOGGER.debug(f"Autarky today: {self.autarky_today:.1f}%")
                        else:
                            self.autarky_today = None
                    else:
                        self.autarky_today = None
                except Exception as autarky_err:
                    _LOGGER.warning(f"Autarky calculation failed: {autarky_err}")
                    self.autarky_today = None

            # 10. Yesterday accuracy check
            self.last_day_error_kwh, self.yesterday_accuracy = await self._get_yesterday_accuracy()
            if self.yesterday_accuracy is not None:
                _LOGGER.debug(f"Yesterday accuracy: {self.yesterday_accuracy:.1f}%")

            # 11. Correction factor
            try:
                if self.service_manager.ml_predictor:
                    learned_factor = getattr(self.service_manager.ml_predictor, 'learned_correction_factor', 1.0)
                    if learned_factor and CORRECTION_FACTOR_MIN <= learned_factor <= CORRECTION_FACTOR_MAX:
                        self.learned_correction_factor = learned_factor
                        _LOGGER.debug(f"Learned correction factor: {self.learned_correction_factor:.3f}")
            except Exception as factor_err:
                _LOGGER.warning(f"Error getting correction factor: {factor_err}")

            # 12. Next hour prediction
            if self.enable_hourly and hourly_forecast_list:
                if current_weather_data:
                    try:
                        self.next_hour_pred = self.forecast_orchestrator.calculate_next_hour_prediction(
                            forecast_result.get("today", 0.0),
                            weather_data=current_weather_data,
                            sensor_data=external_sensor_data
                        )
                        _LOGGER.debug(f"Next Hour Prediction OK: {self.next_hour_pred:.3f} kWh")
                    except Exception as next_hour_err:
                        _LOGGER.error(f"Next hour prediction failed: {next_hour_err}", exc_info=True)
                        self.next_hour_pred = 0.0
                else:
                    _LOGGER.warning("Cannot calculate next hour: current weather missing.")
                    self.next_hour_pred = 0.0
            else:
                self.next_hour_pred = 0.0

            # 13. Store prediction record (ALWAYS - no production window check)
            try:
                today_forecast_value = forecast_result.get("today")
                
                if not await self._should_skip_prediction_storage(today_forecast_value):
                    prediction_record = {
                        "timestamp": dt_util.now().isoformat(),
                        "predicted_value": today_forecast_value,
                        "actual_value": None,
                        "weather_data": hourly_forecast_list[0] if hourly_forecast_list else {},
                        "sensor_data": external_sensor_data,
                        "accuracy": forecast_result.get("confidence", 75.0) / 100.0,
                        "model_version": ML_MODEL_VERSION
                    }
                    await self.data_manager.add_prediction_record(prediction_record)
                    _LOGGER.debug(f"Prediction record saved: {today_forecast_value:.2f} kWh")
                else:
                    _LOGGER.debug("Prediction storage skipped (duplicate prevention).")
            except Exception as store_err:
                _LOGGER.warning(f"Failed to save prediction record: {store_err}", exc_info=True)

            # 14. Complete
            self._last_update_success_time = dt_util.now()
            update_duration = (self._last_update_success_time - update_start_time).total_seconds()
            _LOGGER.info(f"Data update cycle completed in {update_duration:.2f}s.")
            
            return {
                "forecast_today": forecast_result.get("today"),
                "forecast_tomorrow": forecast_result.get("tomorrow"),
                "last_update": self._last_update_success_time.isoformat(),
                "_forecast_method": forecast_result.get("method", "unknown"),
                "_model_accuracy": forecast_result.get("model_accuracy"),
                "_confidence_today": forecast_result.get("confidence"),
                "_weather_entity_used": self.current_weather_entity,
                "_update_duration_sec": round(update_duration, 2)
            }

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

        ml_predictor = getattr(self.service_manager, 'ml_predictor', None)
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
        if self.service_manager and self.service_manager.weather_service:
            try:
                weather_healthy = self.service_manager.weather_service.get_health_status().get('healthy', False)
            except:
                pass

        update_age_ok = True
        if self._last_update_success_time:
            age = (dt_util.now() - self._last_update_success_time).total_seconds()
            if age > (self.update_interval.total_seconds() * 2):
                update_age_ok = False
        else:
            update_age_ok = False

        ml_active = self.service_manager.is_ml_ready() if self.service_manager else False
        if ml_active and weather_healthy and update_age_ok:
            return "Optimal (ML Active)"
        elif weather_healthy and update_age_ok:
            reason = "ML Disabled/Unavailable" if not self.service_manager or not self.service_manager.ml_predictor else "ML Not Ready"
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
        """Set expected daily production at 6 AM."""
        try:
            forecast = await self.forecast_orchestrator.get_forecast()
            today_forecast = forecast.get("today", {})
            self.expected_daily_production = today_forecast.get("energy_sum")
            _LOGGER.info(f"Expected daily production set to: {self.expected_daily_production} kWh")
            self.async_update_listeners()
        except Exception as err:
            _LOGGER.error(f"Failed to set expected daily production: {err}", exc_info=True)

    async def reset_expected_daily_production(self) -> None:
        """Reset expected daily production at midnight."""
        self.expected_daily_production = None
        _LOGGER.debug("Expected daily production reset to None")
        self.async_update_listeners()
