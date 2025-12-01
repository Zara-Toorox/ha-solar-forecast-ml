"""Data Update Coordinator for Solar Forecast ML Integration V10.0.0 @zara

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
from typing import Any, Dict, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    CONF_FALLBACK_ENTITY,
    CONF_HUMIDITY_SENSOR,
    CONF_LUX_SENSOR,
    CONF_PRESSURE_SENSOR,
    CONF_RAIN_SENSOR,
    CONF_SOLAR_RADIATION_SENSOR,
    CONF_TEMP_SENSOR,
    CONF_UPDATE_INTERVAL,
    CONF_WIND_SENSOR,
    CORRECTION_FACTOR_MAX,
    CORRECTION_FACTOR_MIN,
    DAILY_UPDATE_HOUR,
    DAILY_VERIFICATION_HOUR,
    DATA_DIR,
    DOMAIN,
    ML_MODEL_VERSION,
    UPDATE_INTERVAL,
)
from .core.core_exceptions import MLModelException, SolarForecastMLException, WeatherAPIException
from .core.core_helpers import SafeDateTimeUtil as dt_util
from .data.data_manager import DataManager
from .forecast.forecast_orchestrator import ForecastOrchestrator
from .forecast.forecast_weather import WeatherService

from .forecast.forecast_weather_calculator import WeatherCalculator
from .ml.ml_best_hour_calculator import BestHourCalculator
from .ml.ml_predictor import MLPredictor, ModelState
from .production.production_history import ProductionCalculator as HistoricalProductionCalculator
from .production.production_scheduled_tasks import ScheduledTasksManager
from .production.production_tracker import ProductionTimeCalculator
from .sensors.sensor_data_collector import SensorDataCollector
from .services.service_error_handler import ErrorHandlingService

_LOGGER = logging.getLogger(__name__)

class SolarForecastMLCoordinator(DataUpdateCoordinator):
    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        dependencies_ok: bool = False,
    ):
        """Initialize coordinator - refactored for clarity"""
        update_interval_seconds = entry.options.get(
            CONF_UPDATE_INTERVAL, UPDATE_INTERVAL.total_seconds()
        )
        update_interval_timedelta = timedelta(seconds=update_interval_seconds)

        super().__init__(hass, _LOGGER, name=DOMAIN, update_interval=update_interval_timedelta)

        self.entry = entry
        self.dependencies_ok = dependencies_ok

        from .core.core_coordinator_init_helpers import CoordinatorInitHelpers

        config = CoordinatorInitHelpers.extract_configuration(entry)

        data_dir_path = CoordinatorInitHelpers.setup_data_directory(hass)
        self.data_manager = DataManager(hass, entry.entry_id, data_dir_path)
        self.sensor_collector = SensorDataCollector(hass, entry)

        self.solar_capacity = config.solar_capacity
        self.learning_enabled = config.learning_enabled
        self.enable_hourly = config.enable_hourly

        self.power_entity = self.sensor_collector.strip_entity_id(config.power_entity)
        self.solar_yield_today = self.sensor_collector.strip_entity_id(config.solar_yield_today)
        self.primary_weather_entity = self.sensor_collector.strip_entity_id(
            config.primary_weather_entity
        )
        self.current_weather_entity: Optional[str] = self.primary_weather_entity
        self.total_consumption_today = self.sensor_collector.strip_entity_id(
            config.total_consumption_today
        )

        self.weather_calculator = WeatherCalculator()
        self.historical_calculator = HistoricalProductionCalculator(hass, self.data_manager)
        self.production_time_calculator = ProductionTimeCalculator(
            hass=hass,
            power_entity=self.power_entity,
            data_manager=self.data_manager,
            coordinator=self,
        )
        self.best_hour_calculator = BestHourCalculator(
            data_manager=self.data_manager
        )
        self.forecast_orchestrator = ForecastOrchestrator(
            hass=hass,
            data_manager=self.data_manager,
            solar_capacity=self.solar_capacity,
            weather_calculator=self.weather_calculator,
        )
        self.scheduled_tasks = ScheduledTasksManager(
            hass=hass,
            coordinator=self,
            solar_yield_today_entity_id=self.solar_yield_today,
            data_manager=self.data_manager,
        )

        self.error_handler = ErrorHandlingService()
        self.weather_service: Optional[WeatherService] = None
        self.ml_predictor: Optional[MLPredictor] = None
        self._services_initialized = False
        self._ml_ready = False

        from .data.data_weather_pipeline_manager import WeatherDataPipelineManager

        self.weather_pipeline_manager: Optional[WeatherDataPipelineManager] = None

        self.weather_fallback_active = False
        self._last_weather_update: Optional[datetime] = None
        self._forecast_cache: Dict[str, Any] = {}
        self._startup_time: datetime = dt_util.now()
        self._last_update_success_time: Optional[datetime] = None
        self._startup_sensors_ready: bool = False
        self._hourly_predictions_cache: Optional[Dict[str, Any]] = None

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

        self.cloudiness_trend_1h: float = 0.0
        self.cloudiness_trend_3h: float = 0.0
        self.cloudiness_volatility: float = 0.0

        self._training_ready_count: int = 0

        self._recovery_lock = asyncio.Lock()
        self._recovery_in_progress = False

        self.battery_enabled = config.battery_enabled
        self.electricity_enabled = config.electricity_enabled
        self.battery_collector = CoordinatorInitHelpers.initialize_battery_collector(
            hass, entry, config.battery_enabled
        )
        self.electricity_service = CoordinatorInitHelpers.initialize_electricity_service(
            config.electricity_enabled, config.electricity_country
        )

        self.system_status_sensor = None

        _LOGGER.debug("SolarForecastMLCoordinator initialized")

    async def _load_persistent_state(self) -> None:
        """Load persistent coordinator state eg expected_daily_production @zara"""
        try:
            loaded_value = await self.data_manager.load_expected_daily_production()
            if loaded_value is not None:
                self.expected_daily_production = loaded_value
        except Exception as e:
            _LOGGER.warning(f"Failed to load persistent coordinator state: {e}")

    async def _initialize_services(self) -> bool:
        """Initialize all services weather ML error handler @zara"""
        try:

            if self.learning_enabled and self.dependencies_ok:
                try:

                    notification_service = self.hass.data.get(DOMAIN, {}).get(
                        "notification_service"
                    )

                    self.ml_predictor = MLPredictor(
                        hass=self.hass,
                        data_manager=self.data_manager,
                        error_handler=self.error_handler,
                        notification_service=notification_service,
                        config_entry=self.entry,
                    )

                    self.ml_predictor.set_entities(
                        solar_capacity=self.solar_capacity,
                        power_entity=self.power_entity,
                        weather_entity=self.current_weather_entity,
                        temp_sensor=self.entry.data.get(CONF_TEMP_SENSOR),
                        wind_sensor=self.entry.data.get(CONF_WIND_SENSOR),
                        rain_sensor=self.entry.data.get(CONF_RAIN_SENSOR),
                        lux_sensor=self.entry.data.get(CONF_LUX_SENSOR),
                        humidity_sensor=self.entry.data.get(CONF_HUMIDITY_SENSOR),
                        pressure_sensor=self.entry.data.get(CONF_PRESSURE_SENSOR),
                        solar_radiation_sensor=self.entry.data.get(CONF_SOLAR_RADIATION_SENSOR),
                    )

                    init_success = await self.ml_predictor.initialize()
                    if init_success:
                        self._ml_ready = True
                        self.best_hour_calculator.ml_predictor = self.ml_predictor
                    else:
                        _LOGGER.error("MLPredictor initialization failed")
                        self.ml_predictor = None
                except Exception as e:
                    _LOGGER.error(f"Failed to initialize MLPredictor: {e}")
                    self.ml_predictor = None

            self._services_initialized = True
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to initialize services: {e}")
            return False

    async def _initialize_forecast_orchestrator(self) -> None:
        """Initialize the forecast orchestrator strategies @zara"""
        if not self.ml_predictor:
            self.forecast_orchestrator.initialize_strategies(
                ml_predictor=None, error_handler=self.error_handler
            )
        else:
            self.forecast_orchestrator.initialize_strategies(
                ml_predictor=self.ml_predictor, error_handler=self.error_handler
            )

    async def async_setup(self) -> bool:
        """Setup coordinator and start tracking @zara"""
        try:
            init_ok = await self.data_manager.initialize()
            if not init_ok:
                _LOGGER.error("Failed to initialize data manager")
                return False

            await self.data_manager.async_initialize()

            services_ok = await self._initialize_services()
            if not services_ok:
                _LOGGER.error("Failed to initialize services")
                return False

            try:
                from .astronomy.astronomy_cache import AstronomyCache
                from .data.data_weather_pipeline_manager import WeatherDataPipelineManager

                astronomy_cache = AstronomyCache(
                    data_dir=self.data_manager.data_dir,
                    data_manager=self.data_manager,
                )

                latitude = self.hass.config.latitude
                longitude = self.hass.config.longitude
                timezone_str = str(self.hass.config.time_zone)
                elevation_m = self.hass.config.elevation or 0
                astronomy_cache.initialize_location(latitude, longitude, timezone_str, elevation_m)

                self.weather_pipeline_manager = WeatherDataPipelineManager(
                    hass=self.hass,
                    data_dir=self.data_manager.data_dir,
                    stats_dir=self.data_manager.data_dir / "stats",
                    data_manager=self.data_manager,
                    astronomy_cache=astronomy_cache,
                    config_entry=self.entry,
                    coordinator=self,
                )

                from .astronomy.astronomy_cache_manager import get_cache_manager
                cache_manager = get_cache_manager()
                cache_file = self.data_manager.data_dir / "stats" / "astronomy_cache.json"

                if not cache_file.exists():
                    try:
                        solar_capacity = self.entry.data.get("solar_capacity", 5.0)
                        await astronomy_cache.rebuild_cache(system_capacity_kwp=solar_capacity)
                    except Exception as rebuild_err:
                        _LOGGER.error(f"Failed to rebuild astronomy cache: {rebuild_err}")

                def _init_cache_sync():
                    return cache_manager.initialize(cache_file)

                import asyncio
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, _init_cache_sync)

                pipeline_setup_ok = await self.weather_pipeline_manager.async_setup()
                if not pipeline_setup_ok:
                    _LOGGER.error("Failed to setup Weather Data Pipeline Manager")
                    return False

                pipeline_start_ok = await self.weather_pipeline_manager.start_pipeline()
                if not pipeline_start_ok:
                    _LOGGER.error("Failed to start Weather Data Pipeline")
                    return False

                self.weather_service = self.weather_pipeline_manager.weather_service

            except Exception as e:
                _LOGGER.error(f"Failed to initialize Weather Data Pipeline Manager: {e}")
                return False

            await self._load_persistent_state()

            try:
                await self.production_time_calculator.start_tracking()
            except Exception as track_err:
                _LOGGER.error(f"Failed to start production time tracking: {track_err}")

            await self._setup_power_peak_tracking()

            self.scheduled_tasks.setup_listeners()

            try:
                hourly_data = await self.data_manager.hourly_predictions._read_json_async()
                self._hourly_predictions_cache = hourly_data
            except Exception:
                pass

            if self.ml_predictor:
                @callback
                def _scheduled_weekly_retraining(now: datetime) -> None:
                    """Callback for weekly model retraining - Sundays only. @zara"""
                    if now.weekday() == 6:
                        asyncio.create_task(self.ml_predictor.train_model())

                async_track_time_change(
                    self.hass, _scheduled_weekly_retraining, hour=3, minute=0, second=0
                )

            await self.scheduled_tasks.calculate_yesterday_deviation_on_startup()

            ml_status = "ML-Ready" if self._ml_ready else "Rule-Based"
            _LOGGER.info(f"Solar Forecast Coordinator ready ({ml_status}, {self.solar_capacity} kWp)")

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to setup coordinator: {e}")
            return False

    async def async_shutdown(self) -> None:
        """Cleanup coordinator resources @zara"""

        try:
            if self.weather_pipeline_manager:
                await self.weather_pipeline_manager.stop_pipeline()

            await self.production_time_calculator.stop_tracking()
            self.scheduled_tasks.cancel_listeners()
        except Exception as e:
            _LOGGER.error(f"Error during coordinator shutdown: {e}")

    async def _setup_power_peak_tracking(self) -> None:
        """Setup event listener for power peak tracking @zara"""
        if not self.power_entity:
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

                if power_w > current_peak_today:
                    current_peak_today = power_w

                    now = dt_util.now()
                    if last_write_time is None or (now - last_write_time).total_seconds() > 60:

                        all_time_peak = await self.data_manager.get_all_time_peak()
                        is_all_time = all_time_peak is None or power_w > all_time_peak

                        await self.data_manager.save_power_peak(
                            power_w=power_w, timestamp=now, is_all_time=is_all_time
                        )

                        last_write_time = now

            except (ValueError, TypeError):
                pass

        from homeassistant.helpers.event import async_track_state_change_event

        async_track_state_change_event(self.hass, [self.power_entity], power_state_changed)

    async def _async_update_data(self):
        """Fetch data from API endpoint - refactored for clarity @zara"""
        try:

            if not self._services_initialized:
                services_ok = await self._initialize_services()
                if not services_ok:
                    raise UpdateFailed("Failed to initialize services")

            await self._initialize_forecast_orchestrator()

            from .core.core_coordinator_update_helpers import CoordinatorUpdateHelpers

            helpers = CoordinatorUpdateHelpers(self)

            await helpers.handle_startup_recovery()

            current_weather, hourly_forecast = await helpers.fetch_weather_data()

            external_sensors = self.sensor_collector.collect_all_sensor_data_dict()

            forecast = await helpers.generate_forecast(
                current_weather, hourly_forecast, external_sensors
            )

            result = await helpers.build_coordinator_result(forecast, current_weather, external_sensors)

            await self._update_sensor_properties(result)

            await helpers.save_forecasts(forecast, result.get("hourly_forecast", []))

            self._last_update_success_time = dt_util.now()

            if not self._startup_sensors_ready:
                self._startup_sensors_ready = True

            await helpers.update_electricity_prices()

            return result

        except UpdateFailed:
            raise
        except Exception as err:
            _LOGGER.error(f"Unexpected error updating data: {err}")
            raise UpdateFailed(f"Error communicating with API: {err}")

    async def _update_sensor_properties(self, data: Dict[str, Any]) -> None:
        """Update coordinator properties used by sensors @zara"""
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
            self.last_successful_learning = getattr(ml_predictor, "last_training_time", None)
            if self.model_accuracy is None:
                self.model_accuracy = getattr(ml_predictor, "current_accuracy", None)

            try:
                await ml_predictor._load_recent_weather_samples(hours_back=6)
                trends = ml_predictor._calculate_cloudiness_trends()
                self.cloudiness_trend_1h = trends.get("cloudiness_trend_1h", 0.0)
                self.cloudiness_trend_3h = trends.get("cloudiness_trend_3h", 0.0)
                self.cloudiness_volatility = trends.get("cloudiness_volatility", 0.0)
            except Exception:
                self.cloudiness_trend_1h = 0.0
                self.cloudiness_trend_3h = 0.0
                self.cloudiness_volatility = 0.0

            try:
                training_count = await self._get_training_ready_count()
                self._training_ready_count = training_count
            except Exception:
                self._training_ready_count = 0
        else:
            self.last_successful_learning = None
            self.cloudiness_trend_1h = 0.0
            self.cloudiness_trend_3h = 0.0
            self.cloudiness_volatility = 0.0
            self._training_ready_count = 0

    async def _save_forecasts_to_storage(self, forecast_data: dict, hourly_forecast: list) -> None:
        """Save forecasts to daily_forecastsjson based on current time @zara"""
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

            if tomorrow_kwh is not None:
                tomorrow_date = now_local + timedelta(days=1)
                await self.data_manager.save_forecast_tomorrow(
                    date=tomorrow_date,
                    prediction_kwh=tomorrow_kwh,
                    source=source,
                    lock=False,
                )

            if day_after_kwh is not None:
                day_after_date = now_local + timedelta(days=2)
                await self.data_manager.save_forecast_day_after(
                    date=day_after_date,
                    prediction_kwh=day_after_kwh,
                    source=source,
                    lock=False,
                )

            if hour == 6 and now_local.minute < 15:
                if today_kwh is not None:
                    today_raw = forecast_data.get("today_raw")
                    safeguard_applied = forecast_data.get("safeguard_applied", False)

                    await self.data_manager.save_forecast_day(
                        prediction_kwh=today_kwh,
                        source=source,
                        lock=True,
                        force_overwrite=True,
                        prediction_kwh_raw=today_raw,
                        safeguard_applied=safeguard_applied,
                    )

                try:
                    best_hour, best_hour_kwh = (
                        await self.best_hour_calculator.calculate_best_hour_today()
                    )

                    if best_hour is not None:
                        if best_hour_kwh is not None and best_hour_kwh > 0:
                            source = "ML-Hourly" if self.ml_predictor else "Profile"
                        else:
                            source = "Solar-Noon"

                        await self.data_manager.save_forecast_best_hour(
                            hour=best_hour,
                            prediction_kwh=best_hour_kwh if best_hour_kwh else 0.0,
                            source=source,
                        )
                except Exception:
                    pass

            elif 12 <= hour < 13:
                if tomorrow_kwh is not None:
                    tomorrow_date = now_local + timedelta(days=1)
                    await self.data_manager.save_forecast_tomorrow(
                        date=tomorrow_date,
                        prediction_kwh=tomorrow_kwh,
                        source=source,
                        lock=True,
                    )

            elif 18 <= hour < 19:
                if day_after_kwh is not None:
                    day_after_date = now_local + timedelta(days=2)
                    await self.data_manager.save_forecast_day_after(
                        date=day_after_date,
                        prediction_kwh=day_after_kwh,
                        source=source,
                        lock=True,
                    )

        except Exception as e:
            _LOGGER.error(f"Failed to save forecasts to storage: {e}")

    @property
    def last_update_success_time(self) -> Optional[datetime]:
        return self._last_update_success_time

    @property
    def weather_source(self) -> str:
        return self.current_weather_entity or "Open-Meteo (direct radiation)"

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
                weather_healthy = self.weather_service.get_health_status().get("healthy", False)
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

    def on_ml_training_complete(
        self, timestamp: datetime, accuracy: Optional[float] = None
    ) -> None:
        _LOGGER.info(
            f"Coordinator notified of ML Training completion at {timestamp}. Accuracy: {accuracy}"
        )
        self.last_successful_learning = timestamp
        if accuracy is not None:
            self.model_accuracy = accuracy
        self.async_update_listeners()

        if accuracy is not None:
            samples = self.ml_predictor.training_samples if self.ml_predictor else 0
            self.update_system_status(
                event_type="ml_training",
                event_status="success",
                event_summary=f"ML Training erfolgreich - Genauigkeit: {accuracy*100:.1f}%",
                event_details={
                    "accuracy_percent": round(accuracy * 100, 1),
                    "samples_used": samples,
                    "training_time": timestamp.isoformat(),
                },
            )
        else:
            self.update_system_status(
                event_type="ml_training",
                event_status="failed",
                event_summary="ML Training fehlgeschlagen",
                event_details={},
            )

    async def set_expected_daily_production(self) -> None:
        """Set expected daily production at 6 AM and save persistently @zara"""
        try:
            if (
                self.data
                and "forecast_today" in self.data
                and self.data.get("forecast_today") is not None
            ):
                self.expected_daily_production = self.data.get("forecast_today")
            else:
                await self.async_request_refresh()

                for i in range(10):
                    if (
                        self.data
                        and "forecast_today" in self.data
                        and self.data.get("forecast_today") is not None
                    ):
                        break
                    await asyncio.sleep(1.0)

                if (
                    self.data
                    and "forecast_today" in self.data
                    and self.data.get("forecast_today") is not None
                ):
                    self.expected_daily_production = self.data.get("forecast_today")
                else:
                    self.expected_daily_production = None

            if self.expected_daily_production is not None:
                await self.data_manager.save_expected_daily_production(
                    self.expected_daily_production
                )

                new_save_ok = await self.data_manager.save_daily_forecast(
                    prediction_kwh=self.expected_daily_production,
                    source="auto_6am",
                    force_overwrite=False,
                )

                if not new_save_ok:
                    _LOGGER.error("CRITICAL: daily_forecasts.json NOT saved!")

                self.async_update_listeners()

        except Exception as err:
            _LOGGER.error(f"Failed to set expected daily production: {err}")
            self.expected_daily_production = None

    async def reset_expected_daily_production(self) -> None:
        """Reset expected daily production at midnight and clear persistent storage @zara"""
        self.expected_daily_production = None
        await self.data_manager.clear_expected_daily_production()
        self.async_update_listeners()

    async def _recovery_forecast_process(self, source: str) -> bool:
        """Fallback process for missing forecasts @zara"""
        async with self._recovery_lock:
            if self._recovery_in_progress:
                return False

            self._recovery_in_progress = True
            try:
                return await self._execute_recovery(source)
            finally:
                self._recovery_in_progress = False

    async def _execute_recovery(self, source: str) -> bool:
        """Internal method to execute the recovery process @zara"""
        if not self.weather_service:
            _LOGGER.error("Weather service not initialized for recovery")
            return False

        existing_forecast = await self.data_manager.get_current_day_forecast()
        if existing_forecast and existing_forecast.get("forecast_day", {}).get("locked"):
            return True

        try:
            hourly_forecast = await self.weather_service.get_hourly_forecast()
            if hourly_forecast:
                external_sensors = self.sensor_collector.collect_all_sensor_data_dict()

                forecast = await self.forecast_orchestrator.orchestrate_forecast(
                    current_weather=None,
                    hourly_forecast=hourly_forecast,
                    external_sensors=external_sensors,
                    correction_factor=self.learned_correction_factor,
                )

                if forecast and forecast.get("today") is not None:
                    success = await self.data_manager.save_daily_forecast(
                        prediction_kwh=forecast["today"],
                        source=f"fallback_open_meteo_{source}",
                        force_overwrite=True,
                    )
                    if success:
                        return True
        except Exception:
            pass

        try:
            current_weather = await self.weather_service.get_current_weather()
            hourly_forecast = await self.weather_service.get_corrected_hourly_forecast()
            external_sensors = self.sensor_collector.collect_all_sensor_data_dict()

            forecast = await self.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=self.learned_correction_factor,
            )

            if forecast and forecast.get("today") is not None:
                success = await self.data_manager.save_daily_forecast(
                    prediction_kwh=forecast["today"],
                    source=f"fallback_rule_based_{source}",
                    force_overwrite=True,
                )
                if success:
                    return True
        except Exception as e:
            _LOGGER.error(f"Rule-based fallback failed: {e}")

        _LOGGER.error("All fallback methods failed - unable to set forecast")
        return False

    async def force_refresh_with_weather_update(self) -> None:
        """Force refresh with immediate weather update @zara"""
        if self.weather_service:
            await self.weather_service.force_update()

        await self.async_request_refresh()

    async def forecast_day_after_tomorrow(self) -> None:
        """Triggers and saves the forecast for the day after tomorrow @zara"""
        try:
            current_weather = (
                await self.weather_service.get_current_weather() if self.weather_service else None
            )

            hourly_forecast = (
                await self.weather_service.get_corrected_hourly_forecast()
                if self.weather_service
                else None
            )
            external_sensors = self.sensor_collector.collect_all_sensor_data_dict()

            forecast = await self.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=self.learned_correction_factor,
            )

            if not forecast or forecast.get("day_after_tomorrow") is None:
                return

            day_after_kwh = forecast.get("day_after_tomorrow")
            now_local = dt_util.now()
            day_after_date = now_local + timedelta(days=2)
            source = "manual_service"

            await self.data_manager.save_forecast_day_after(
                date=day_after_date,
                prediction_kwh=day_after_kwh,
                source=source,
                lock=True,
            )

            await self.async_request_refresh()

        except Exception as e:
            _LOGGER.error(f"Error in forecast_day_after_tomorrow service: {e}")

    def update_system_status(
        self,
        event_type: str,
        event_status: str,
        event_summary: str,
        event_details: Optional[dict] = None,
        warnings: Optional[list] = None,
    ) -> None:
        """Update system status sensor with event information"""
        if self.system_status_sensor is None:
            return

        try:
            self.system_status_sensor.update_status(
                event_type=event_type,
                event_status=event_status,
                event_summary=event_summary,
                event_details=event_details,
                warnings=warnings,
            )
        except Exception as e:
            _LOGGER.error(f"Failed to update system status: {e}")

    async def _get_training_ready_count(self) -> int:
        """Count training-ready samples from hourly_predictions.json @zara"""

        def _count_sync():
            """Synchronous file reading (executed in thread pool) @zara"""
            try:
                import json
                from pathlib import Path

                predictions_file = (
                    Path(self.data_manager.data_dir) / "stats" / "hourly_predictions.json"
                )

                if not predictions_file.exists():
                    return 0

                with open(predictions_file, "r") as f:
                    data = json.load(f)

                predictions = data.get("predictions", [])
                return sum(1 for p in predictions if p.get("actual_kwh") is not None)

            except Exception:
                return 0

        return await self.hass.async_add_executor_job(_count_sync)
