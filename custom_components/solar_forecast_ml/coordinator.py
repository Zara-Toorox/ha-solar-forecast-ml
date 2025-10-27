"""
Data Update Coordinator für Solar Forecast ML Integration.
REFACTORED VERSION: Modulare Struktur mit separaten Manager-Klassen
Version 5.1.0 - Modulare Architektur

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
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN, CONF_SOLAR_CAPACITY, CONF_WEATHER_ENTITY,
    CONF_LEARNING_ENABLED, CONF_PLANT_KWP,
    CONF_HOURLY, CONF_POWER_ENTITY, CONF_SOLAR_YIELD_TODAY,
    CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX, DATA_DIR,
    ML_MODEL_VERSION, DAILY_UPDATE_HOUR, DAILY_VERIFICATION_HOUR,
    CONF_TEMP_SENSOR, CONF_WIND_SENSOR, CONF_RAIN_SENSOR,
    CONF_UV_SENSOR, CONF_LUX_SENSOR
)
from .data_manager import DataManager
from .exceptions import SolarForecastMLException, WeatherException, ModelException

from .service_manager import ServiceManager
from .weather_calculator import WeatherCalculator
from .production_calculator import ProductionCalculator, ProductionTimeCalculator

from .sensor_data_collector import SensorDataCollector
from .forecast_orchestrator import ForecastOrchestrator
from .scheduled_tasks_manager import ScheduledTasksManager

_LOGGER = logging.getLogger(__name__)


class SolarForecastMLCoordinator(DataUpdateCoordinator):
    """
    Coordinator Solar Forecast ML Integration.
    REFACTORED: Schlanker Orchestrator mit modularer Architektur
    """

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        dependencies_ok: bool = False,
    ):
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=None,
        )
        
        self.entry = entry
        
        self.data_manager = DataManager(hass, entry.entry_id, Path(DATA_DIR))
        
        self.solar_capacity = entry.data.get(CONF_SOLAR_CAPACITY, 5.0)
        self.learning_enabled = entry.data.get(CONF_LEARNING_ENABLED, True)
        
        self.sensor_collector = SensorDataCollector(hass, entry)
        
        self.power_entity = self.sensor_collector.strip_entity_id(
            entry.data.get(CONF_POWER_ENTITY)
        )
        self.solar_yield_today = self.sensor_collector.strip_entity_id(
            entry.data.get(CONF_SOLAR_YIELD_TODAY)
        )
        
        self.primary_weather_entity = self.sensor_collector.strip_entity_id(
            entry.data.get(CONF_WEATHER_ENTITY)
        )
        self.current_weather_entity: Optional[str] = self.primary_weather_entity
        
        self.weather_fallback_active = False
        self.enable_hourly = entry.data.get(CONF_HOURLY, entry.options.get(CONF_HOURLY, False))
        
        self._last_weather_update = None
        self._forecast_cache = {}
        self._last_prediction_time: Optional[datetime] = None
        self._last_prediction_value: Optional[float] = None
        self._startup_time = dt_util.utcnow()
        
        self._last_update_success_time: Optional[datetime] = None
        self.last_update_time: Optional[datetime] = None
        
        self.next_hour_pred = 0.0
        self.peak_production_time_today = "12:00"
        self.production_time_today = "Initialisierung..."
        self.last_day_error_kwh = None
        self.autarky_today = None
        self.avg_month_yield = 0.0
        self.last_successful_learning = None
        self.model_accuracy = None
        self.sun_guard_status = "Unbekannt"
        self.sun_guard_window = "N/A"
        
        self.weather_calculator = WeatherCalculator()
        self.production_calculator = ProductionCalculator(hass)
        
        self.production_time_calculator = ProductionTimeCalculator(
            hass=hass,
            power_entity=self.power_entity
        )
        
        self.forecast_orchestrator = ForecastOrchestrator(
            hass=hass,
            solar_capacity=self.solar_capacity,
            weather_calculator=self.weather_calculator
        )
        
        temp_sensor = self.sensor_collector.get_sensor_entity_id(CONF_TEMP_SENSOR)
        wind_sensor = self.sensor_collector.get_sensor_entity_id(CONF_WIND_SENSOR)
        rain_sensor = self.sensor_collector.get_sensor_entity_id(CONF_RAIN_SENSOR)
        uv_sensor = self.sensor_collector.get_sensor_entity_id(CONF_UV_SENSOR)
        lux_sensor = self.sensor_collector.get_sensor_entity_id(CONF_LUX_SENSOR)
        
        self.service_manager = ServiceManager(
            hass=hass,
            entry=entry,
            data_manager=self.data_manager,
            weather_entity=self.current_weather_entity,
            dependencies_ok=dependencies_ok,
            power_entity=self.power_entity,
            solar_yield_today=self.solar_yield_today,
            solar_capacity=self.solar_capacity,
            temp_sensor=temp_sensor,
            wind_sensor=wind_sensor,
            rain_sensor=rain_sensor,
            uv_sensor=uv_sensor,
            lux_sensor=lux_sensor,
        )
        
        self.scheduled_tasks = ScheduledTasksManager(
            hass=hass,
            coordinator=self,
            solar_yield_today=self.solar_yield_today,
            data_manager=self.data_manager
        )
        
        self.dependencies_ok = dependencies_ok
        
        _LOGGER.info(
            f"SolarForecastMLCoordinator (MODULAR + SUN GUARD) initialisiert - "
            f"Weather Entity: {self.primary_weather_entity}"
        )
    
    async def async_config_entry_first_refresh(self) -> None:
        _LOGGER.info("=== Starting First Refresh v5.1.0 (Modular) ===")
        
        self.current_weather_entity = self.primary_weather_entity
        _LOGGER.info(f"⚙️ Weather Entity konfiguriert: {self.current_weather_entity}")
        
        if self.current_weather_entity:
            _LOGGER.info("Warte auf Weather Entity Verfügbarkeit...")
            max_wait_time = 30
            wait_interval = 2
            total_waited = 0
            
            while total_waited < max_wait_time:
                if await self._check_weather_entity_available(self.current_weather_entity):
                    _LOGGER.info(
                        f"Weather Entity '{self.current_weather_entity}' ist bereit "
                        f"(nach {total_waited}s)"
                    )
                    break
                
                await asyncio.sleep(wait_interval)
                total_waited += wait_interval
            
            if total_waited >= max_wait_time:
                _LOGGER.warning(
                    f"Weather Entity '{self.current_weather_entity}' nicht verfügbar "
                    f"nach {max_wait_time}s - starte trotzdem"
                )
        
        try:
            ml_predictor = self.service_manager.ml_predictor
            error_handler = self.service_manager.error_handler
            self.forecast_orchestrator.initialize_strategies(ml_predictor, error_handler)
                
        except Exception as e:
            _LOGGER.error(f"Strategy Initialisierung fehlgeschlagen: {e}")
        
        try:
            if self.power_entity:
                self.production_time_calculator.start_tracking()
                _LOGGER.info(f"Produktionszeit-Tracking gestartet für {self.power_entity}")
            else:
                _LOGGER.info("Kein Power-Sensor konfiguriert - Produktionszeit-Tracking deaktiviert")
        except Exception as e:
            _LOGGER.warning(f"Produktionszeit-Tracking konnte nicht gestartet werden: {e}")
        
        available_sensors = await self.sensor_collector.wait_for_external_sensors(max_wait=25)
        if available_sensors > 0:
            _LOGGER.info("✅ %d externe Sensoren ready", available_sensors)
        else:
            _LOGGER.warning("⚠ Keine externen Sensoren verfügbar - Predictions ohne Sensor-Daten")

        await super().async_config_entry_first_refresh()
        
        _LOGGER.info("=== Registriere tägliche Time-Trigger ===")
        
        async_track_time_change(
            self.hass,
            self.scheduled_tasks.scheduled_morning_update,
            hour=DAILY_UPDATE_HOUR,
            minute=0,
            second=0
        )
        _LOGGER.info(f"Morgen-Update registriert: {DAILY_UPDATE_HOUR}:00 Uhr")
        
        async_track_time_change(
            self.hass,
            self.scheduled_tasks.scheduled_evening_verification,
            hour=DAILY_VERIFICATION_HOUR,
            minute=0,
            second=0
        )
        _LOGGER.info(f"Abend-Verifikation registriert: {DAILY_VERIFICATION_HOUR}:00 Uhr")
    
    async def _check_weather_entity_available(self, entity_id: str) -> bool:
        if not entity_id:
            return False
        
        state = self.hass.states.get(entity_id)
        if state is None:
            return False
        
        if state.state in ["unavailable", "unknown"]:
            return False
        
        return True
    
    async def _get_weather_data(self) -> dict[str, Any]:
        if not self.current_weather_entity:
            raise WeatherException("Keine Weather Entity konfiguriert")
        
        state = self.hass.states.get(self.current_weather_entity)
        
        if not state:
            raise WeatherException(
                f"Weather Entity '{self.current_weather_entity}' nicht gefunden"
            )
        
        if state.state in ["unavailable", "unknown"]:
            raise WeatherException(
                f"Weather Entity '{self.current_weather_entity}' nicht verfügbar"
            )
        
        try:
            attributes = state.attributes
            condition = state.state
            
            cloud_cover = attributes.get("cloud_coverage", 50.0)
            if isinstance(cloud_cover, str):
                cloud_cover = float(cloud_cover.replace("%", ""))
            
            weather_data = {
                "temperature": float(attributes.get("temperature", 15.0)),
                "humidity": float(attributes.get("humidity", 60.0)),
                "cloud_cover": float(cloud_cover),
                "wind_speed": float(attributes.get("wind_speed", 3.0)),
                "precipitation": float(attributes.get("precipitation", 0.0)),
                "pressure": float(attributes.get("pressure", 1013.25)),
                "condition": condition
            }
            
            _LOGGER.debug(
                f"Weather Data: Temp={weather_data['temperature']}, "
                f"Clouds={weather_data['cloud_cover']}%, "
                f"Condition={weather_data['condition']}"
            )
            
            self._last_weather_update = dt_util.utcnow()
            
            return weather_data
            
        except (ValueError, TypeError, KeyError) as e:
            raise WeatherException(f"Fehler beim Parsen der Wetterdaten: {e}")
    
    async def _should_skip_prediction_storage(
        self,
        sensor_data: Dict[str, Any],
        forecast_value: float
    ) -> bool:
        now = dt_util.utcnow()
        
        if self._last_prediction_time and self._last_prediction_value is not None:
            time_diff = (now - self._last_prediction_time).total_seconds()
            value_diff = abs(forecast_value - self._last_prediction_value)
            
            if time_diff < 2.0 and value_diff < 0.01:
                _LOGGER.debug(
                    "⏭ Skip duplicate prediction (%.1fs ago, Δ=%.4f kWh)",
                    time_diff, value_diff
                )
                return True
        
        seconds_since_startup = (now - self._startup_time).total_seconds()
        
        if seconds_since_startup < 30:
            all_sensors_null = all(
                v is None for v in sensor_data.values()
            )
            
            if all_sensors_null:
                _LOGGER.info(
                    "⏭ Skip prediction (alle Sensoren null, %.1fs seit Start)",
                    seconds_since_startup
                )
                return True
        
        self._last_prediction_time = now
        self._last_prediction_value = forecast_value
        
        return False

    async def _async_update_data(self) -> dict[str, Any]:
        try:
            _LOGGER.debug("=== Starting Data Update (MODULAR) ===")
            
            weather_data = await self._get_weather_data()
            _LOGGER.debug(f"Weather Data: {weather_data}")
            
            sensor_data = self.sensor_collector.collect_all_sensor_data(
                self.solar_capacity,
                self.power_entity
            )
            
            sensor_data_dict = self.sensor_collector.collect_sensor_data_dict()
            
            forecast = await self.forecast_orchestrator.create_forecast(
                weather_data=weather_data,
                sensor_data=sensor_data,
                correction_factor=1.0
            )
            
            try:
                if self.power_entity:
                    historical_peak = await self.production_calculator.calculate_peak_production_time(
                        power_entity=self.power_entity
                    )
                    if forecast.get("peak_time") == "12:00" or not forecast.get("peak_time"):
                        forecast["peak_time"] = historical_peak
                        _LOGGER.debug(f"Historische Peak-Zeit verwendet: {historical_peak}")
                    else:
                        _LOGGER.debug(f"ML Peak-Zeit beibehalten: {forecast.get('peak_time')}")
                else:
                    _LOGGER.debug("Kein Power-Sensor - verwende Standard Peak-Zeit")
            except Exception as e:
                _LOGGER.warning(f"Peak-Zeit Berechnung fehlgeschlagen: {e}")
            
            await self._update_sensor_properties(forecast)
            
            self._last_update_success_time = dt_util.utcnow()
            self.last_update_time = dt_util.utcnow()
            
            self.next_hour_pred = self.forecast_orchestrator.calculate_next_hour_prediction(
                forecast.get("today", 0.0),
                weather_data=weather_data,
                sensor_data=sensor_data_dict
            )
            
            try:
                ml_predictor = self.ml_predictor
                if ml_predictor and hasattr(ml_predictor, 'last_training_time'):
                    if ml_predictor.last_training_time:
                        self.last_successful_learning = ml_predictor.last_training_time
            except Exception as e:
                _LOGGER.debug(f"ML Status Sync fehlgeschlagen: {e}")
            
            try:
                if hasattr(self, 'data_manager') and self.data_manager:
                    sensor_data_dict = self.sensor_collector.collect_sensor_data_dict()
                    
                    should_skip = await self._should_skip_prediction_storage(
                        sensor_data_dict,
                        forecast["today"]
                    )

                    if not should_skip:
                        prediction_record = {
                            "timestamp": dt_util.utcnow().isoformat(),
                            "predicted_value": forecast["today"],
                            "actual_value": None,
                            "weather_data": weather_data,
                            "sensor_data": sensor_data_dict,
                            "accuracy": forecast.get("confidence", 75.0) / 100.0,
                            "model_version": ML_MODEL_VERSION
                        }
                        
                        await self.data_manager.add_prediction_record(prediction_record)
                        _LOGGER.debug(f"✅ Forecast gespeichert: {forecast['today']:.2f} kWh (confidence={forecast.get('confidence', 75.0)}%)")
                    else:
                        _LOGGER.debug("⏭ Forecast NICHT gespeichert (Skip-Regel)")
                    
            except Exception as e:
                _LOGGER.warning(f"⚠️ Forecast-Speicherung fehlgeschlagen: {e}")
            
            try:
                if hasattr(self, 'data_manager') and self.data_manager:
                    await self.data_manager.save_all_async()
            except Exception as e:
                _LOGGER.debug(f"Auto-Save fehlgeschlagen: {e}")
            
            result = {
                "forecast_today": forecast["today"],
                "forecast_tomorrow": forecast["tomorrow"],
                "peak_time": forecast.get("peak_time", "12:00"),
                "confidence": forecast.get("confidence", 75.0),
                "method": forecast.get("method", "unknown"),
                "last_update": dt_util.utcnow().isoformat()
            }
            
            _LOGGER.debug(f"✅ Update complete: {result}")
            
            return result
            
        except WeatherException as e:
            _LOGGER.error(f"🌤️ Weather Error: {e}")
            raise UpdateFailed(f"Weather Fehler: {e}")
        except Exception as e:
            _LOGGER.error(f"❌ Update Failed: {e}", exc_info=True)
            raise UpdateFailed(f"Update Fehler: {e}")
    
    async def _update_sensor_properties(self, forecast: dict[str, Any]) -> None:
        self.peak_production_time_today = forecast.get("peak_time", "12:00")
        
        self.production_time_today = self.production_time_calculator.get_production_time()
        
        if forecast.get("model_accuracy") is not None:
            self.model_accuracy = forecast["model_accuracy"]
        
        try:
            avg_yield = await self.hass.async_add_executor_job(
                self.data_manager.get_average_monthly_yield
            )
            self.avg_month_yield = avg_yield if avg_yield else 0.0
        except Exception as e:
            _LOGGER.warning(f"Fehler bei avg_month_yield Berechnung: {e}")
            self.avg_month_yield = 0.0
        
        if self.sun_guard:
            is_production = self.sun_guard.is_production_time()
            self.sun_guard_status = "🟢 GESTARTET" if is_production else "🔴 PAUSIERT"
            
            sunrise, sunset = self.sun_guard.get_production_window()
            self.sun_guard_window = f"{sunrise.strftime('%H:%M')} - {sunset.strftime('%H:%M')}"
        else:
            self.sun_guard_status = "Nicht verfügbar"
            self.sun_guard_window = "N/A"
    
    @property
    def last_update_success_time(self) -> Optional[datetime]:
        return self._last_update_success_time
    
    @property
    def ml_predictor(self):
        if hasattr(self.service_manager, 'ml_predictor'):
            return self.service_manager.ml_predictor
        return None
    
    @property
    def sun_guard(self):
        if hasattr(self.service_manager, 'sun_guard'):
            return self.service_manager.sun_guard
        return None
    
    @property
    def weather_source(self) -> str:
        return self.current_weather_entity or "Nicht verfügbar"
    
    @property
    def retry_attempts(self) -> int:
        return 0
    
    @property
    def diagnostic_status(self) -> str:
        if not self.last_update_success:
            return "Fehler"
        
        weather_available = False
        if self.current_weather_entity:
            state = self.hass.states.get(self.current_weather_entity)
            weather_available = (
                state is not None and 
                state.state not in ['unavailable', 'unknown', 'none', None]
            )
        
        update_age_ok = True
        if self.last_update_time:
            age = dt_util.utcnow() - self.last_update_time
            if age > timedelta(hours=2):
                update_age_ok = False
        
        ml_available = False
        if self.ml_predictor:
            try:
                if hasattr(self.ml_predictor, 'get_model_health'):
                    health = self.ml_predictor.get_model_health()
                    ml_available = (
                        health.model_loaded and 
                        hasattr(health.state, 'value') and 
                        health.state.value == "ready"
                    )
            except Exception:
                pass
        
        if ml_available and weather_available and update_age_ok:
            return "Optimal"
        elif weather_available and update_age_ok:
            return "Normal"
        elif not weather_available or not update_age_ok:
            return "Eingeschränkt"
        else:
            return "Normal"
    
    def on_ml_training_complete(self, timestamp: datetime, accuracy: float = None) -> None:
        _LOGGER.info(f"✅ ML-Training abgeschlossen - Accuracy: {accuracy}")
        self.last_successful_learning = timestamp
        if accuracy is not None:
            self.model_accuracy = accuracy