"""
Data Update Coordinator for Solar Forecast ML Integration.
REFACTORED VERSION: Modular structure with separate manager classes
Version 5.1.0 - Modular Architecture

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
from .helpers import SafeDateTimeUtil as dt_util

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
    REFACTORED: Lean orchestrator with modular architecture
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
        self.production_time_today = "Initializing..."
        self.last_day_error_kwh = None
        self.autarky_today = None
        self.avg_month_yield = 0.0
        self.last_successful_learning = None
        self.model_accuracy = None
        self.sun_guard_status = "Unknown"
        self.sun_guard_window = "N/A"
        
        # === START PATCH 3: Attribut initialisieren ===
        self.learned_correction_factor: float = 1.0
        # === ENDE PATCH 3 ===
        
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
            f"SolarForecastMLCoordinator (MODULAR + SUN GUARD) initialized - "
            f"Weather Entity: {self.primary_weather_entity}"
        )
    
    async def async_config_entry_first_refresh(self) -> None:
        _LOGGER.info("=== Starting First Refresh v5.1.0 (Modular) ===")
        
        self.current_weather_entity = self.primary_weather_entity
        _LOGGER.info(f"⚡ Weather Entity configured: {self.current_weather_entity}")
        
        if self.current_weather_entity:
            _LOGGER.info("Waiting for Weather Entity availability...")
            max_wait_time = 30
            wait_interval = 2
            total_waited = 0
            
            while total_waited < max_wait_time:
                if await self._check_weather_entity_available(self.current_weather_entity):
                    _LOGGER.info(
                        f"Weather Entity '{self.current_weather_entity}' is ready "
                        f"(after {total_waited}s)"
                    )
                    break
                
                await asyncio.sleep(wait_interval)
                total_waited += wait_interval
            
            if total_waited >= max_wait_time:
                _LOGGER.warning(
                    f"Weather Entity '{self.current_weather_entity}' not available "
                    f"after {max_wait_time}s - starting anyway"
                )
        
        try:
            services_ok = await self.service_manager.initialize_all_services()
            if services_ok:
                _LOGGER.info("Services (incl. ML Predictor) initialized successfully")
            else:
                _LOGGER.warning("Some services failed to initialize")
        except Exception as err:
            _LOGGER.warning(f"Service initialization issue: {err}")
        
        
        try:
            ml_predictor = self.service_manager.ml_predictor
            error_handler = self.service_manager.error_handler
            self.forecast_orchestrator.initialize_strategies(ml_predictor, error_handler)
                
        except Exception as e:
            _LOGGER.error(f"Strategy initialization failed: {e}")
        
        try:
            if self.power_entity:
                self.production_time_calculator.start_tracking()
                _LOGGER.info(f"Production time tracking started for {self.power_entity}")
            else:
                _LOGGER.info("No power sensor configured - production time tracking disabled")
        except Exception as e:
            _LOGGER.warning(f"Could not start production time tracking: {e}")
        
        available_sensors = await self.sensor_collector.wait_for_external_sensors(max_wait=35)
        if available_sensors > 0:
            _LOGGER.info("✔ %d external sensors ready", available_sensors)
        else:
            _LOGGER.warning("⚡ No external sensors available - predictions without sensor data")

        # === START PATCH 3: Lade Korrekturfaktor ===
        try:
            # Versuche, den Faktor aus dem bereits geladenen ml_predictor zu beziehen
            if self.ml_predictor and self.ml_predictor.current_weights:
                self.learned_correction_factor = self.ml_predictor.current_weights.correction_factor
                _LOGGER.info(f"Learned fallback correction_factor loaded from ML Predictor: {self.learned_correction_factor:.3f}")
            else:
                # Fallback: Lade die Gewichte manuell (falls ML-Modell deaktiviert/fehlgeschlagen)
                _LOGGER.info("ML Predictor weights not found, loading correction_factor manually...")
                weights = await self.data_manager.get_learned_weights()
                if weights and hasattr(weights, 'correction_factor'):
                     self.learned_correction_factor = weights.correction_factor
                     _LOGGER.info(f"Manually loaded fallback correction_factor: {self.learned_correction_factor:.3f}")
                else:
                    _LOGGER.info("No learned_weights.json found, using default correction_factor: 1.0")
                    self.learned_correction_factor = 1.0

        except Exception as e:
            _LOGGER.warning(f"Failed to load learned correction_factor: {e}")
        # === ENDE PATCH 3 ===

        await super().async_config_entry_first_refresh()
        
        _LOGGER.info("=== Calculating yesterday's deviation at startup ===")
        try:
            await self.scheduled_tasks.calculate_yesterday_deviation_on_startup()
        except Exception as e:
            _LOGGER.error(f"Error during startup deviation calculation: {e}")
        
        _LOGGER.info("=== Registering daily time triggers ===")
        
        async_track_time_change(
            self.hass,
            self.scheduled_tasks.scheduled_morning_update,
            hour=DAILY_UPDATE_HOUR,
            minute=0,
            second=0
        )
        _LOGGER.info(f"Morning update registered: {DAILY_UPDATE_HOUR}:00")
        
        async_track_time_change(
            self.hass,
            self.scheduled_tasks.scheduled_evening_verification,
            hour=DAILY_VERIFICATION_HOUR,
            minute=0,
            second=0
        )
        _LOGGER.info(f"Evening verification registered: {DAILY_VERIFICATION_HOUR}:00")
    
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
            raise WeatherException("No Weather Entity configured")
        
        state = self.hass.states.get(self.current_weather_entity)
        
        if not state:
            raise WeatherException(
                f"Weather Entity '{self.current_weather_entity}' not found"
            )
        
        if state.state in ["unavailable", "unknown"]:
            raise WeatherException(
                f"Weather Entity '{self.current_weather_entity}' not available"
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
                "cloudiness": float(cloud_cover),
                "wind_speed": float(attributes.get("wind_speed", 3.0)),
                "precipitation": float(attributes.get("precipitation", 0.0)),
                "pressure": float(attributes.get("pressure", 1013.25)),
                "condition": condition
            }
            
            _LOGGER.debug(
                f"Weather Data: Temp={weather_data['temperature']}, "
                f"Clouds={weather_data['cloudiness']}%, "
                f"Condition={weather_data['condition']}"
            )
            
            self._last_weather_update = dt_util.utcnow()
            
            return weather_data
            
        except (ValueError, TypeError, KeyError) as e:
            raise WeatherException(f"Error parsing weather data: {e}")
    
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
                    "Skipping duplicate prediction (%.1fs ago, Δ=%.4f kWh)",
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
                    "Skipping prediction (all sensors null, %.1fs since start)",
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
            
            # === START PATCH 3: ANWENDEN des Korrekturfaktors ===
            forecast = await self.forecast_orchestrator.create_forecast(
                weather_data=weather_data,
                sensor_data=sensor_data,
                correction_factor=self.learned_correction_factor
            )
            # === ENDE PATCH 3 ===
            
            try:
                if self.power_entity:
                    historical_peak = await self.production_calculator.calculate_peak_production_time(
                        power_entity=self.power_entity
                    )
                    if forecast.get("peak_time") == "12:00" or not forecast.get("peak_time"):
                        forecast["peak_time"] = historical_peak
                        _LOGGER.debug(f"Used historical peak time: {historical_peak}")
                    else:
                        _LOGGER.debug(f"Kept ML peak time: {forecast.get('peak_time')}")
                else:
                    _LOGGER.debug("No power sensor - using default peak time")
            except Exception as e:
                _LOGGER.warning(f"Peak time calculation failed: {e}")
            
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
                _LOGGER.debug(f"ML status sync failed: {e}")
            
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
                        _LOGGER.debug(f"✔ Forecast saved: {forecast['today']:.2f} kWh (confidence={forecast.get('confidence', 75.0)}%)")
                    else:
                        _LOGGER.debug("Forecast NOT saved (skip rule)")
                    
            except Exception as e:
                _LOGGER.warning(f"⚠️ Forecast saving failed: {e}")
            
            try:
                # Da DataManager.save_all_async nicht existiert, entfernen wir den Aufruf
                # um Fehler zu vermeiden, da atomic_write_json bereits in add_prediction_record 
                # und anderen Methoden verwendet wird. 
                pass
            except Exception as e:
                _LOGGER.debug(f"Auto-save failed: {e}")
            
            result = {
                "forecast_today": forecast["today"],
                "forecast_tomorrow": forecast["tomorrow"],
                "peak_time": forecast.get("peak_time", "12:00"),
                "confidence": forecast.get("confidence", 75.0),
                "method": forecast.get("method", "unknown"),
                "last_update": dt_util.utcnow().isoformat()
            }
            
            _LOGGER.debug(f"✔ Update complete: {result}")
            
            return result
            
        except WeatherException as e:
            _LOGGER.error(f"☀️ Weather Error: {e}")
            raise UpdateFailed(f"Weather error: {e}")
        except Exception as e:
            _LOGGER.error(f"❌ Update Failed: {e}", exc_info=True)
            raise UpdateFailed(f"Update error: {e}")
    
    async def _update_sensor_properties(self, forecast: dict[str, Any]) -> None:
        self.peak_production_time_today = forecast.get("peak_time", "12:00")
        
        self.production_time_today = self.production_time_calculator.get_production_time()
        
        if forecast.get("model_accuracy") is not None:
            self.model_accuracy = forecast["model_accuracy"]
        
        try:
            # KORREKTUR: Direkter asynchroner Aufruf mit 'await'
            avg_yield = await self.data_manager.get_average_monthly_yield() 
            self.avg_month_yield = avg_yield if avg_yield else 0.0
        except Exception as e:
            _LOGGER.warning(f"Error during avg_month_yield calculation: {e}")
            self.avg_month_yield = 0.0
        
        if self.sun_guard:
            is_production = self.sun_guard.is_production_time()
            self.sun_guard_status = "🟡 STARTED" if is_production else "🔴 PAUSED"
            
            sunrise, sunset = self.sun_guard.get_production_window()
            self.sun_guard_window = f"{sunrise.strftime('%H:%M')} - {sunset.strftime('%H:%M')}"
        else:
            self.sun_guard_status = "Not available"
            self.sun_guard_window = "N/A"
            
    # --- START OF NEW BACKFILL METHODS ---
    
    async def trigger_backfill(self) -> bool:
        """Manuell ausgelöst – prüft Sensoren & startet Backfill als separaten Task."""
        _LOGGER.info("Backfill-Button gedrückt – prüfe Voraussetzungen...")

        # 1. Prüfe ML Predictor
        if not self.ml_predictor:
            _LOGGER.error("Backfill fehlgeschlagen: ML Predictor nicht initialisiert. Warte auf ersten Refresh.")
            return False

        if not hasattr(self.ml_predictor, 'async_run_backfill_process'):
            _LOGGER.error("Backfill fehlgeschlagen: ML Predictor hat keine Backfill-Methode.")
            return False

        # 2. Baue Liste der benötigten Sensoren aus der Konfiguration
        required_sensors = [
            self.power_entity,
            self.solar_yield_today,
            self.primary_weather_entity,
        ]

        # Füge optionale externe Sensoren hinzu
        external_sensors = [
            self.sensor_collector.get_sensor_entity_id(CONF_TEMP_SENSOR),
            self.sensor_collector.get_sensor_entity_id(CONF_WIND_SENSOR),
            self.sensor_collector.get_sensor_entity_id(CONF_RAIN_SENSOR),
            self.sensor_collector.get_sensor_entity_id(CONF_UV_SENSOR),
            self.sensor_collector.get_sensor_entity_id(CONF_LUX_SENSOR),
        ]
        required_sensors.extend([s for s in external_sensors if s])

        # Entferne Duplikate und None
        required_sensors = list(set([s for s in required_sensors if s]))

        if required_sensors:
            _LOGGER.info(f"Prüfe {len(required_sensors)} Sensoren: {required_sensors}")
            if not await self._check_sensors_ready(required_sensors):
                _LOGGER.warning("Backfill verzögert: Nicht alle Sensoren bereit. Warte...")
                return False
        else:
            _LOGGER.warning("Keine Sensoren konfiguriert – Backfill ohne Sensorprüfung")

        # 3. Starte Task
        _LOGGER.info("Alle Voraussetzungen erfüllt – starte Backfill im Hintergrund.")
        self.hass.async_create_task(self._run_backfill_task())
        return True

    async def _run_backfill_task(self):
        """Task, der den eigentlichen Backfill-Prozess im Predictor ausführt."""
        try:
            success = await self.ml_predictor.async_run_backfill_process()

            if success:
                _LOGGER.info("Backfill-Task erfolgreich abgeschlossen. Aktualisiere Sensoren.")
                await self.async_request_refresh() 
            else:
                _LOGGER.warning("Backfill-Task ist fehlgeschlagen (siehe Predictor Logs).")

        except Exception as e:
            _LOGGER.error(f"Backfill-Task Fehler: {e}", exc_info=True)

    async def _check_sensors_ready(self, entities):
        """Check if all sensors have valid data – Weather-Entities nur auf Verfügbarkeit prüfen."""
        missing = []
        for entity_id in entities:
            state = self.hass.states.get(entity_id)
            
            # Check 1: Entität nicht gefunden oder in einem kritischen Zustand
            if state is None:
                missing.append(f"{entity_id} (nicht gefunden)")
                continue
            if state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN, "none", "unavailable", "unknown"):
                missing.append(f"{entity_id} ({state.state})")
                continue
                
            # --- SPEZIALREGEL FÜR WEATHER-ENTITIES ---
            if entity_id.startswith("weather."):
                # Weather-Entity: Nur Verfügbarkeit prüfen, nicht den Zustand (z.B. "partlycloudy")
                continue  # → Keine float-Prüfung!
                
            # --- NORMALE SENSOR-PRÜFUNG: Muss numerisch sein ---
            try:
                float(state.state)
            except (ValueError, TypeError):
                # Optional: Prüfe unit_of_measurement für Sicherheit
                unit = state.attributes.get("unit_of_measurement", None)
                if unit is None:
                    missing.append(f"{entity_id} (nicht numerisch: {state.state})")
            except Exception as e:
                missing.append(f"{entity_id} (Fehler bei float-Konvertierung: {e})")
            
        if missing:
            _LOGGER.warning(f"Sensoren nicht bereit: {', '.join(missing)}")
            return False
            
        _LOGGER.info(f"Alle {len(entities)} Sensoren bereit (Weather-Entity nur auf Verfügbarkeit geprüft).")
        return True

    # --- END OF NEW BACKFILL METHODS ---
    
    @property
    def last_update_success_time(self) -> Optional[datetime]:
        return self._last_update_success_time
    
    @property
    def ml_predictor(self):
        # Stellt sicher, dass die ml_predictor-Instanz korrekt abgerufen wird
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
        return self.current_weather_entity or "Not available"
    
    @property
    def retry_attempts(self) -> int:
        return 0
    
    @property
    def diagnostic_status(self) -> str:
        if not self.last_update_success:
            return "Error"
        
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
            return "Degraded"
        else:
            return "Normal"
    
    def on_ml_training_complete(self, timestamp: datetime, accuracy: float = None) -> None:
        _LOGGER.info(f"✔ ML-Training complete - Accuracy: {accuracy}")
        self.last_successful_learning = timestamp
        if accuracy is not None:
            self.model_accuracy = accuracy