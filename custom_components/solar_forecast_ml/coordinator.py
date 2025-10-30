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
from .data_manager import DataManager
from .exceptions import SolarForecastMLException, WeatherAPIException, MLModelException
from .helpers import SafeDateTimeUtil as dt_util

# Import modular components
from .service_manager import ServiceManager
from .weather_calculator import WeatherCalculator
from .production_history import ProductionCalculator as HistoricalProductionCalculator
from .production_tracker import ProductionTimeCalculator
from .sensor_data_collector import SensorDataCollector
from .forecast_orchestrator import ForecastOrchestrator
from .scheduled_tasks_manager import ScheduledTasksManager
from .ml_predictor import ModelState

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
        self._startup_time: datetime = dt_util.utcnow()
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
        self.sun_guard_status: str = "Unknown"
        self.sun_guard_window: str = "N/A"
        self.learned_correction_factor: float = 1.0

        _LOGGER.info(f"SolarForecastMLCoordinator initialized - Using Weather Entity: {self.primary_weather_entity or 'None'}")

    async def _check_entity_available(self, entity_id: str) -> bool:
        if not entity_id:
            return False
        state = self.hass.states.get(entity_id)
        if state is None:
            _LOGGER.debug(f"Warte-Check: EntitÃƒÆ’Ã‚Â¤t {entity_id} nicht gefunden (None).")
            return False
        if state.state in [STATE_UNAVAILABLE, STATE_UNKNOWN, "None", None, ""]:
            _LOGGER.debug(f"Warte-Check: EntitÃƒÆ’Ã‚Â¤t {entity_id} ist {state.state}.")
            return False
        if state.domain == "weather" and not state.attributes:
            _LOGGER.debug(f"Warte-Check: Wetter-EntitÃƒÆ’Ã‚Â¤t {entity_id} hat noch keine Attribute.")
            return False
        return True

    async def _wait_for_entity(self, entity_id: str, name: str, timeout: int = 45) -> bool:
        if not entity_id:
            _LOGGER.warning(f"Kann nicht auf '{name}' Sensor warten: Keine EntitÃƒÆ’Ã‚Â¤t konfiguriert.")
            return True

        _LOGGER.info(f"Warte auf VerfÃƒÆ’Ã‚Â¼gbarkeit von '{name}' Sensor: {entity_id} (max {timeout}s)...")
        max_wait_time = timeout
        wait_interval = 2
        total_waited = 0

        while total_waited < max_wait_time:
            if await self._check_entity_available(entity_id):
                _LOGGER.info(f"'{name}' Sensor '{entity_id}' ist verfÃƒÆ’Ã‚Â¼gbar (nach {total_waited}s).")
                return True
            await asyncio.sleep(wait_interval)
            total_waited += wait_interval

        _LOGGER.error(f"Timeout: '{name}' Sensor '{entity_id}' wurde nach {max_wait_time}s nicht verfÃƒÆ’Ã‚Â¼gbar.")
        return False

    async def async_config_entry_first_refresh(self) -> None:
        _LOGGER.info("Starting First Refresh sequence...")
        self.current_weather_entity = self.primary_weather_entity
        _LOGGER.info(f"Using primary weather entity: {self.current_weather_entity or 'None'}")

        weather_ok = await self._wait_for_entity(self.current_weather_entity, "Weather")
        power_ok = await self._wait_for_entity(self.power_entity, "Power")
        yield_ok = await self._wait_for_entity(self.solar_yield_today, "Solar Yield")

        if not (weather_ok and power_ok and yield_ok):
            _LOGGER.error("Nicht alle kritischen Sensoren sind verfÃƒÆ’Ã‚Â¼gbar. Integration wird fortgesetzt, kann aber fehlschlagen.")
        else:
            _LOGGER.info("Alle kritischen Sensoren (Wetter, Leistung, Ertrag) sind verfÃƒÆ’Ã‚Â¼gbar.")
            self._startup_sensors_ready = True

        _LOGGER.info("Initializing managed services (Weather, ML, SunGuard)...")
        try:
            services_ok = await self.service_manager.initialize_all_services()
            if services_ok:
                _LOGGER.info("Services initialized successfully.")
            else:
                _LOGGER.warning("One or more critical services failed to initialize.")
        except Exception as err:
            _LOGGER.critical(f"Critical error during service initialization: {err}", exc_info=True)
            return

        # === AUTOMATIC DUPLICATE CLEANUP (Option 1 + 2) ===
        _LOGGER.info("Checking for duplicate hourly samples...")
        try:
            cleanup_result = await self.data_manager.cleanup_duplicate_samples()
            if cleanup_result['removed'] > 0:
                _LOGGER.warning(
                    f"Duplicate Cleanup: {cleanup_result['removed']} Duplikate entfernt, "
                    f"{cleanup_result['remaining']} Samples verbleiben."
                )
            else:
                _LOGGER.info("Keine Duplikate gefunden, hourly_samples.json ist sauber.")
        except Exception as cleanup_err:
            # Cleanup-Fehler sollten Startup nicht blockieren
            _LOGGER.error(
                f"Duplicate Cleanup fehlgeschlagen (nicht-kritisch): {cleanup_err}",
                exc_info=True
            )
        # ===================================================

        _LOGGER.info("Initializing forecast strategies...")
        try:
            ml_predictor = self.service_manager.ml_predictor
            error_handler = self.service_manager.error_handler
            self.forecast_orchestrator.initialize_strategies(ml_predictor, error_handler)
            _LOGGER.info("Forecast strategies initialized.")
        except Exception as e:
            _LOGGER.error(f"Failed to initialize forecast strategies: {e}", exc_info=True)

        try:
            if self.production_time_calculator:
                self.production_time_calculator.start_tracking()
                _LOGGER.info(f"Live production time tracking started for {self.power_entity or 'None'}.")
            else:
                _LOGGER.warning("ProductionTimeCalculator not available.")
        except Exception as e:
            _LOGGER.warning(f"Could not start live production time tracking: {e}")

        asyncio.create_task(self.sensor_collector.wait_for_external_sensors(max_wait=15))

        _LOGGER.info("Loading learned fallback correction factor...")
        try:
            ml_predictor = self.service_manager.ml_predictor
            if ml_predictor and ml_predictor.current_weights:
                loaded_factor = getattr(ml_predictor.current_weights, 'correction_factor', 1.0)
                self.learned_correction_factor = max(CORRECTION_FACTOR_MIN, min(CORRECTION_FACTOR_MAX, loaded_factor))
                _LOGGER.info(f"Loaded correction_factor from ML weights: {self.learned_correction_factor:.3f}")
            else:
                weights_from_file = await self.data_manager.get_learned_weights()
                if weights_from_file:
                    loaded_factor = getattr(weights_from_file, 'correction_factor', 1.0)
                    self.learned_correction_factor = max(CORRECTION_FACTOR_MIN, min(CORRECTION_FACTOR_MAX, loaded_factor))
                    _LOGGER.info(f"Manually loaded correction_factor from file: {self.learned_correction_factor:.3f}")
                else:
                    _LOGGER.info("No learned weights found, using default correction_factor: 1.0")
                    self.learned_correction_factor = 1.0
        except Exception as e:
            _LOGGER.warning(f"Failed to load learned correction_factor: {e}. Using default 1.0")
            self.learned_correction_factor = 1.0

        _LOGGER.info("Performing first data update...")
        await super().async_config_entry_first_refresh()

        _LOGGER.info("Calculating yesterday's deviation at startup...")
        try:
            await self.scheduled_tasks.calculate_yesterday_deviation_on_startup()
        except Exception as e:
            _LOGGER.error(f"Error during startup deviation calculation: {e}")

        _LOGGER.info("Registering daily scheduled tasks...")
        try:
            if self.scheduled_tasks:
                self.scheduled_tasks.setup_listeners()
                _LOGGER.info(f"Daily tasks registered: Morning Update ~{DAILY_UPDATE_HOUR:02d}:00, Evening Verify ~{DAILY_VERIFICATION_HOUR:02d}:00.")
            else:
                _LOGGER.error("ScheduledTasksManager not available.")
        except Exception as e:
            _LOGGER.error(f"Failed to register daily scheduled tasks: {e}")

        _LOGGER.info("First Refresh sequence completed.")

    async def _get_weather_data(self) -> dict[str, Any]:
        if not self.service_manager or not self.service_manager.weather_service:
            _LOGGER.error("WeatherService is not available.")
            raise WeatherAPIException("WeatherService is not initialized")

        try:
            weather_data = await self.service_manager.weather_service.get_current_weather()
            self._last_weather_update = dt_util.utcnow()
            current_svc_entity = self.service_manager.weather_service.weather_entity
            if self.weather_fallback_active and current_svc_entity == self.primary_weather_entity:
                self.weather_fallback_active = False
                _LOGGER.info(f"Recovered and now using primary weather entity: {self.primary_weather_entity}")
            return weather_data

        except WeatherAPIException as primary_error:
            _LOGGER.warning(f"Failed to get weather from current entity '{self.current_weather_entity}': {primary_error}")
            fallback_entity_cfg = self.entry.data.get(CONF_FALLBACK_ENTITY)
            if fallback_entity_cfg and fallback_entity_cfg != self.current_weather_entity:
                if not self.weather_fallback_active:
                    _LOGGER.info(f"Attempting fallback to weather entity: {fallback_entity_cfg}")
                    try:
                        self.service_manager.weather_service.update_weather_entity(fallback_entity_cfg)
                        self.current_weather_entity = fallback_entity_cfg
                        self.weather_fallback_active = True
                        weather_data = await self.service_manager.weather_service.get_current_weather()
                        self._last_weather_update = dt_util.utcnow()
                        _LOGGER.info(f"Successfully retrieved weather using fallback entity: {fallback_entity_cfg}")
                        return weather_data
                    except WeatherAPIException as fallback_error:
                        _LOGGER.error(f"Fallback weather entity '{fallback_entity_cfg}' also failed: {fallback_error}")
                        raise UpdateFailed(f"Fallback weather entity failed: {fallback_error}") from fallback_error
                    except Exception as fallback_init_err:
                        _LOGGER.error(f"Error during fallback attempt: {fallback_init_err}")
                        raise UpdateFailed(f"Error during fallback switch: {fallback_init_err}") from fallback_init_err
                else:
                    _LOGGER.error(f"Fallback weather entity '{self.current_weather_entity}' failed again: {primary_error}")
                    raise UpdateFailed(f"Fallback weather entity failed again: {primary_error}") from primary_error
            else:
                reason = "no fallback configured" if not fallback_entity_cfg else "fallback is same as primary/current"
                _LOGGER.error(f"Primary weather entity failed, and {reason}.")
                raise UpdateFailed(f"Primary weather failed, no usable fallback: {primary_error}") from primary_error
        except Exception as e:
            _LOGGER.error(f"Unexpected error getting weather data: {e}", exc_info=True)
            raise UpdateFailed(f"Unexpected weather error: {e}") from e

    async def _should_skip_prediction_storage(self, forecast_value: Optional[float]) -> bool:
        now_utc = dt_util.utcnow()
        if forecast_value is None or forecast_value < 0:
            return True
        if self._last_prediction_time and self._last_prediction_value is not None:
            time_diff_sec = (now_utc - self._last_prediction_time).total_seconds()
            value_diff = abs(forecast_value - self._last_prediction_value)
            if time_diff_sec < 5.0 and value_diff < 0.001:
                return True
        seconds_since_startup = (now_utc - self._startup_time).total_seconds()
        if seconds_since_startup < 45 and forecast_value < 0.01:
            return True
        self._last_prediction_time = now_utc
        self._last_prediction_value = forecast_value
        return False

    async def _get_sensor_value_safe(self, entity_id: Optional[str]) -> Optional[float]:
        if not entity_id:
            return None
        state = self.hass.states.get(entity_id)
        if not state or state.state in [STATE_UNAVAILABLE, STATE_UNKNOWN, "None", None, ""]:
            _LOGGER.warning(f"Sensor '{entity_id}' ist nicht verfÃƒÆ’Ã‚Â¼gbar (Status: {state.state if state else 'None'}).")
            return None
        try:
            cleaned_state = str(state.state).split(" ")[0].replace(",", ".")
            value = float(cleaned_state)
            return value if value >= 0 else 0.0
        except (ValueError, TypeError):
            _LOGGER.error(f"Konnte Sensorwert fÃƒÆ’Ã‚Â¼r '{entity_id}' nicht umwandeln: '{state.state}'")
            return None

    async def _calculate_autarky(self) -> Optional[float]:
        if not self.total_consumption_today or not self.solar_yield_today:
            _LOGGER.debug("Autarkie-Berechnung ÃƒÆ’Ã‚Â¼bersprungen: Sensoren nicht konfiguriert.")
            return None

        yield_val = await self._get_sensor_value_safe(self.solar_yield_today)
        consumption_val = await self._get_sensor_value_safe(self.total_consumption_today)

        if yield_val is None or consumption_val is None:
            _LOGGER.debug("Autarkie-Berechnung fehlgeschlagen: Sensor nicht verfÃƒÆ’Ã‚Â¼gbar.")
            return None

        try:
            if consumption_val <= 0.01:
                return 100.0 if yield_val >= 0 else 0.0
            if yield_val <= 0.0:
                return 0.0
            autarky = (yield_val / consumption_val) * 100.0
            autarky_capped = max(0.0, min(100.0, autarky))
            _LOGGER.debug(f"Autarkie berechnet: Ertrag={yield_val:.2f}, Verbrauch={consumption_val:.2f} -> {autarky_capped:.1f}%")
            return autarky_capped
        except Exception as e:
            _LOGGER.error(f"Unerwarteter Fehler bei Autarkie-Berechnung: {e}", exc_info=True)
            return None

    async def _async_update_data(self) -> Dict[str, Any]:
        _LOGGER.debug("Starting data update cycle...")
        update_start_time = dt_util.utcnow()
        try:
            if not self._startup_sensors_ready:
                _LOGGER.warning("Update-Zyklus pausiert, warte auf Initialisierung der kritischen Sensoren...")
                weather_ok = await self._wait_for_entity(self.current_weather_entity, "Weather", timeout=10)
                power_ok = await self._wait_for_entity(self.power_entity, "Power", timeout=10)
                yield_ok = await self._wait_for_entity(self.solar_yield_today, "Solar Yield", timeout=10)
                if not (weather_ok and power_ok and yield_ok):
                    raise UpdateFailed("Kritische Sensoren sind weiterhin nicht verfÃƒÆ’Ã‚Â¼gbar.")
                else:
                    _LOGGER.info("Kritische Sensoren sind jetzt verfÃƒÆ’Ã‚Â¼gbar. Setze Update-Zyklus fort.")
                    self._startup_sensors_ready = True

            weather_data = await self._get_weather_data()
            _LOGGER.debug(f"Weather Data OK from {self.current_weather_entity}")
            external_sensor_data = self.sensor_collector.collect_all_sensor_data_dict()
            _LOGGER.debug(f"External Sensor Data OK: {external_sensor_data}")
            
            # Hole aktuellen Ertrag fÃƒÂ¼r Mindest-Prognose-Check
            current_yield = await self._get_sensor_value_safe(self.solar_yield_today)
            
            forecast_input_sensor_data = {
                'solar_capacity': self.solar_capacity,
                'current_yield': current_yield,  # FÃƒÂ¼r Mindest-Prognose in Rule-Based
                **external_sensor_data
            }
            forecast_result = await self.forecast_orchestrator.create_forecast(
                weather_data=weather_data, sensor_data=forecast_input_sensor_data, correction_factor=self.learned_correction_factor
            )
            _LOGGER.debug(f"Forecast Result OK: Today={forecast_result.get('today'):.2f}, Method={forecast_result.get('method')}")

            try:
                if self.power_entity and self.historical_calculator:
                    historical_peak = await self.historical_calculator.calculate_peak_production_time(power_entity=self.power_entity)
                    current_peak = forecast_result.get("peak_time")
                    if historical_peak and (not current_peak or current_peak == "12:00"):
                        forecast_result["peak_time"] = historical_peak
                        _LOGGER.debug(f"Applied historical peak time: {historical_peak}")
                else:
                    forecast_result.setdefault("peak_time", "12:00")
            except Exception as peak_err:
                _LOGGER.warning(f"Historical peak time calculation failed: {peak_err}. Using default.")
                forecast_result.setdefault("peak_time", "12:00")

            self.autarky_today = await self._calculate_autarky()

            await self._update_sensor_properties(forecast_result)

            if self.enable_hourly:
                try:
                    self.next_hour_pred = self.forecast_orchestrator.calculate_next_hour_prediction(
                        forecast_result.get("today", 0.0), weather_data=weather_data, sensor_data=external_sensor_data
                    )
                    _LOGGER.debug(f"Next Hour Prediction OK: {self.next_hour_pred:.3f} kWh")
                except Exception as next_hour_err:
                    _LOGGER.error(f"Next hour prediction calculation failed: {next_hour_err}", exc_info=True)
                    self.next_hour_pred = 0.0
            else:
                self.next_hour_pred = 0.0

            try:
                today_forecast_value = forecast_result.get("today")
                should_skip = await self._should_skip_prediction_storage(today_forecast_value)
                if not should_skip and self.data_manager:
                    prediction_record = {
                        "timestamp": dt_util.now().isoformat(),
                        "predicted_value": today_forecast_value,
                        "actual_value": None,
                        "weather_data": weather_data,
                        "sensor_data": external_sensor_data,
                        "accuracy": forecast_result.get("confidence", 75.0) / 100.0,
                        "model_version": ML_MODEL_VERSION
                    }
                    await self.data_manager.add_prediction_record(prediction_record)
                    _LOGGER.debug(f"Prediction record saved: {today_forecast_value:.2f} kWh")
                elif should_skip:
                    _LOGGER.debug("Prediction storage skipped.")
            except Exception as store_err:
                _LOGGER.warning(f"Failed to save prediction record: {store_err}", exc_info=True)

            self._last_update_success_time = dt_util.now()
            update_duration = (self._last_update_success_time - update_start_time).total_seconds()
            _LOGGER.info(f"Data update cycle completed successfully in {update_duration:.2f}s.")
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

        sun_guard = getattr(self.service_manager, 'sun_guard', None)
        if sun_guard:
            try:
                # --- FIX: await hinzufÃƒÆ’Ã‚Â¼gen! ---
                is_prod_time = await sun_guard.is_production_time()
                self.sun_guard_status = "Active Window" if is_prod_time else "Outside Window"

                start_utc, end_utc = await sun_guard.get_production_window_utc()
                start_local = dt_util.as_local(start_utc)
                end_local = dt_util.as_local(end_utc)
                self.sun_guard_window = f"{start_local.strftime('%H:%M')} - {end_local.strftime('%H:%M')} (Local)"
            except Exception as sg_err:
                _LOGGER.warning(f"Error getting SunGuard status/window: {sg_err}")
                self.sun_guard_status = "Error"
                self.sun_guard_window = "N/A"
        else:
            self.sun_guard_status = "Unavailable"
            self.sun_guard_window = "N/A"

        _LOGGER.debug("Coordinator sensor properties updated.")

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
            age = dt_util.utcnow() - self._last_update_success_time
            if age > (self.update_interval * 2):
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