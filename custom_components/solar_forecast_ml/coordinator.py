# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Solar Forecast ML V16.2.0 - Main Coordinator.

SolarForecastMLCoordinator class - central data update coordinator.
Uses DataManager for all database operations (no JSON).

@zara
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_change, async_track_state_change_event
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    CONF_FALLBACK_ENTITY,
    CONF_UPDATE_INTERVAL,
    CORRECTION_FACTOR_MAX,
    CORRECTION_FACTOR_MIN,
    DAILY_UPDATE_HOUR,
    DAILY_VERIFICATION_HOUR,
    DOMAIN,
    ML_MODEL_VERSION,
    UPDATE_INTERVAL,
    VERSION,
    # Coordinator Data Keys
    DATA_KEY_FORECAST_TODAY,
    DATA_KEY_EXTERNAL_SENSORS,
    EXT_SENSOR_SOLAR_YIELD_TODAY,
)
from .core.core_exceptions import MLModelException, SolarForecastMLException, WeatherAPIException
from .core.core_helpers import SafeDateTimeUtil as dt_util
from .core.core_startup_data_resolver import StartupDataResolver, StartupData
from .data.data_manager import DataManager
from .forecast.forecast_orchestrator import ForecastOrchestrator
from .forecast.forecast_weather import WeatherService
from .forecast.forecast_weather_calculator import WeatherCalculator
from .ai import AIPredictor, ModelState, BestHourCalculator
from .physics.physics_calibrator import PhysicsCalibrator
from .production.production_history import ProductionCalculator as HistoricalProductionCalculator
from .production.production_scheduled_tasks import ScheduledTasksManager
from .production.production_tracker import ProductionTimeCalculator
from .sensors.sensor_data_collector import SensorDataCollector
from .services.service_error_handler import ErrorHandlingService

_LOGGER = logging.getLogger(__name__)


class SolarForecastMLCoordinator(DataUpdateCoordinator):
    """Main coordinator for Solar Forecast ML. @zara

    Manages all data updates and coordinates between components.
    All data operations use DatabaseManager via DataManager.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        dependencies_ok: bool = False,
    ):
        """Initialize coordinator. @zara"""
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

        startup_config = {
            "latitude": hass.config.latitude,
            "longitude": hass.config.longitude,
            "solar_capacity": config.solar_capacity,
            "timezone": str(hass.config.time_zone),
        }

        self.data_manager = DataManager(hass, entry.entry_id, data_dir_path, startup_config)
        self.sensor_collector = SensorDataCollector(hass, entry)

        # Store configuration @zara
        self.solar_capacity = config.solar_capacity
        self.learning_enabled = config.learning_enabled
        self.enable_hourly = config.enable_hourly
        self.panel_groups = config.panel_groups
        self.inverter_max_power = config.inverter_max_power

        # Entity IDs @zara
        self.power_entity = self.sensor_collector.strip_entity_id(config.power_entity)
        self.solar_yield_today = self.sensor_collector.strip_entity_id(config.solar_yield_today)
        self.primary_weather_entity = self.sensor_collector.strip_entity_id(
            config.primary_weather_entity
        )
        self.current_weather_entity: Optional[str] = self.primary_weather_entity
        self.total_consumption_today = self.sensor_collector.strip_entity_id(
            config.total_consumption_today
        )

        # Initialize calculators @zara
        self.weather_calculator = WeatherCalculator()
        self.historical_calculator = HistoricalProductionCalculator(hass, self.data_manager._db_manager)
        self.production_time_calculator = ProductionTimeCalculator(
            hass=hass,
            power_entity=self.power_entity,
            db_manager=self.data_manager._db_manager,
            coordinator=self,
        )
        self.best_hour_calculator = BestHourCalculator(db_manager=self.data_manager._db_manager)
        self.forecast_orchestrator = ForecastOrchestrator(
            hass=hass,
            data_manager=self.data_manager,
            solar_capacity=self.solar_capacity,
            weather_calculator=self.weather_calculator,
            db_manager=self.data_manager._db_manager,
            panel_groups=self.panel_groups,
        )
        self.scheduled_tasks = ScheduledTasksManager(
            hass=hass,
            coordinator=self,
            solar_yield_today_entity_id=self.solar_yield_today,
            db_manager=self.data_manager._db_manager,
        )

        # Service components @zara
        self.error_handler = ErrorHandlingService()
        self.weather_service: Optional[WeatherService] = None
        self.ai_predictor: Optional[AIPredictor] = None
        self._services_initialized = False
        self._ml_ready = False

        from .data.data_weather_pipeline_manager import WeatherDataPipelineManager

        self.weather_pipeline_manager: Optional[WeatherDataPipelineManager] = None

        # State tracking @zara
        self.weather_fallback_active = False
        self._last_weather_update: Optional[datetime] = None
        self._forecast_cache: Dict[str, Any] = {}
        self._startup_time: datetime = dt_util.now()
        self._last_update_success_time: Optional[datetime] = None
        self._startup_sensors_ready: bool = False
        self._hourly_predictions_cache: Optional[Dict[str, Any]] = None

        # Sensor values @zara
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

        # Cloudiness trends @zara
        self.cloudiness_trend_1h: float = 0.0
        self.cloudiness_trend_3h: float = 0.0
        self.cloudiness_volatility: float = 0.0

        self._training_ready_count: int = 0

        # Recovery state @zara
        self._recovery_lock = asyncio.Lock()
        self._recovery_in_progress = False

        self.system_status_sensor = None
        self.panel_group_sensor_reader = None

        # Physics calibrator @zara
        self.physics_calibrator: Optional[PhysicsCalibrator] = None

        # V17.0.0: Drift monitor @zara
        self.drift_monitor = None
        self._drift_status_cache: Dict[str, Any] = {}

        # Unsubscribe callbacks @zara
        self._unsub_power_peak_listener: Optional[callable] = None
        self._unsub_weekly_retraining_listener: Optional[callable] = None

        # Startup data resolver @zara
        self._startup_data_resolver: Optional[StartupDataResolver] = None
        self._startup_data: Optional[StartupData] = None

        _LOGGER.debug(f"SolarForecastMLCoordinator V{VERSION} initialized")

    async def _load_persistent_state(self) -> None:
        """Load persistent coordinator state from database. @zara"""
        try:
            loaded_value = await self.data_manager.load_expected_daily_production()
            if loaded_value is not None:
                self.expected_daily_production = loaded_value
        except Exception as e:
            _LOGGER.warning(f"Failed to load persistent coordinator state: {e}")

    async def _initialize_services(self) -> bool:
        """Initialize all services (weather, ML, error handler). @zara"""
        try:
            if self.learning_enabled and self.dependencies_ok:
                try:
                    notification_service = self.hass.data.get(DOMAIN, {}).get(
                        "notification_service"
                    )

                    self.ai_predictor = AIPredictor(
                        hass=self.hass,
                        db_manager=self.data_manager._db_manager,
                        error_handler=self.error_handler,
                        notification_service=notification_service,
                        config_entry=self.entry,
                        panel_groups=self.panel_groups,
                        solar_capacity=self.solar_capacity,
                    )

                    self.ai_predictor.set_entities(
                        solar_capacity=self.solar_capacity,
                        power_entity=self.power_entity,
                        weather_entity=self.current_weather_entity,
                    )

                    init_success = await self.ai_predictor.initialize()
                    if init_success:
                        self._ml_ready = True
                        self.best_hour_calculator.ai_predictor = self.ai_predictor
                    else:
                        _LOGGER.error("AIPredictor initialization failed")
                        self.ai_predictor = None
                except Exception as e:
                    _LOGGER.error(f"Failed to initialize AIPredictor: {e}")
                    self.ai_predictor = None

            # Initialize Physics Calibrator @zara
            try:
                self.physics_calibrator = PhysicsCalibrator(
                    db_manager=self.data_manager._db_manager
                )
                await self.physics_calibrator.async_init()
                _LOGGER.info("PhysicsCalibrator initialized for self-learning corrections")
            except Exception as e:
                _LOGGER.warning(f"Failed to initialize PhysicsCalibrator: {e}")
                self.physics_calibrator = None

            self._services_initialized = True
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to initialize services: {e}")
            return False

    async def _initialize_forecast_orchestrator(self) -> None:
        """Initialize the forecast orchestrator strategies. @zara"""
        self.forecast_orchestrator.initialize_strategies(
            ai_predictor=self.ai_predictor, error_handler=self.error_handler
        )

        # Connect PhysicsCalibrator to RuleBasedStrategy @zara
        if self.physics_calibrator:
            try:
                strategy = self.forecast_orchestrator.rule_based_strategy
                if strategy:
                    strategy.set_physics_calibrator(self.physics_calibrator)
                    _LOGGER.info(
                        "PhysicsCalibrator passed to RuleBasedStrategy - "
                        "self-learning physics corrections enabled"
                    )
            except Exception as e:
                _LOGGER.warning(f"Could not connect PhysicsCalibrator: {e}")

        # Connect WeatherActualTracker @zara
        if self.weather_pipeline_manager:
            try:
                strategy = self.forecast_orchestrator.rule_based_strategy
                weather_actual_tracker = getattr(
                    self.weather_pipeline_manager, 'weather_actual_tracker', None
                )
                if strategy and weather_actual_tracker:
                    strategy.set_weather_actual_tracker(weather_actual_tracker)
                    # V17.0.0: Set panel groups for per-group snow tracking @zara
                    if self.panel_groups:
                        weather_actual_tracker.set_panel_groups(self.panel_groups)
                    _LOGGER.info(
                        "WeatherActualTracker passed to RuleBasedStrategy - "
                        "SNOWY bucket detection enabled"
                    )
            except Exception as e:
                _LOGGER.warning(f"Could not connect WeatherActualTracker: {e}")

        # V17.0.0: Initialize DriftMonitor @zara
        try:
            from .ai.ai_drift_monitor import DriftMonitor
            db_mgr = self.data_manager._db_manager
            self.drift_monitor = DriftMonitor(db_mgr, self.panel_groups or [])
            _LOGGER.info("DriftMonitor initialized")
        except Exception as e:
            _LOGGER.warning(f"Could not initialize DriftMonitor: {e}")

    async def async_setup(self) -> bool:
        """Setup coordinator and start tracking. @zara"""
        try:
            init_ok = await self.data_manager.initialize()
            if not init_ok:
                _LOGGER.error("Failed to initialize data manager")
                return False

            self.historical_calculator.db = self.data_manager._db_manager
            self.production_time_calculator.db = self.data_manager._db_manager
            self.best_hour_calculator.db_manager = self.data_manager._db_manager
            self.forecast_orchestrator._db = self.data_manager._db_manager
            self.scheduled_tasks.db = self.data_manager._db_manager
            self.scheduled_tasks.morning_routine_handler.db = self.data_manager._db_manager
            self.scheduled_tasks.adaptive_forecast_engine.db = self.data_manager._db_manager
            self.scheduled_tasks.daily_summaries_handler.db = self.data_manager._db_manager

            await self._load_persistent_state()

            self._startup_data_resolver = StartupDataResolver(
                db_manager=self.data_manager._db_manager,
                sensor_collector=self.sensor_collector,
                on_sensors_available=self._on_sensors_available_callback,
            )

            _LOGGER.info("Resolving startup data from DB (non-blocking)...")
            self._startup_data = await self._startup_data_resolver.resolve_startup_data()

            if self._startup_data.is_usable:
                _LOGGER.info(
                    "Startup data available: source=%s, weather=%s, astro=%s, sensors=%s",
                    self._startup_data.source,
                    self._startup_data.weather_available,
                    self._startup_data.astronomy_available,
                    self._startup_data.sensors_available,
                )
            else:
                for warning in self._startup_data.warnings:
                    _LOGGER.warning("Startup: %s", warning)

            _LOGGER.info("Solar Forecast Coordinator basic setup complete - starting background initialization")

            self.hass.async_create_task(
                self._background_initialization(),
                name="solar_forecast_ml_background_init"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to setup coordinator: {e}")
            return False

    def _on_sensors_available_callback(self, sensor_data: Dict[str, Any]) -> None:
        """Callback when external sensors become available. @zara"""
        _LOGGER.info("External sensors now available - triggering data refresh")
        if self._startup_data:
            self._startup_data.sensors_available = True
            self._startup_data.sensor_data = sensor_data
        self.hass.async_create_task(
            self.async_request_refresh(),
            name="solar_forecast_ml_sensor_refresh"
        )

    async def _background_initialization(self) -> None:
        """Initialize heavy components in background without blocking HA startup. @zara"""
        try:
            _LOGGER.info("Background initialization started")

            try:
                services_ok = await self._initialize_services()
                if not services_ok:
                    _LOGGER.warning("Services initialization failed - will use rule-based forecasting")
            except Exception as e:
                _LOGGER.warning(f"Failed to initialize services: {e}")

            # Initialize weather pipeline @zara
            try:
                from .astronomy.astronomy_cache import AstronomyCache
                from .data.data_weather_pipeline_manager import WeatherDataPipelineManager

                astronomy_cache = AstronomyCache(
                    db_manager=self.data_manager._db_manager,
                )

                if self.panel_groups:
                    astronomy_cache.set_panel_groups(self.panel_groups)
                    _LOGGER.info(
                        f"AstronomyCache: Panel groups set ({len(self.panel_groups)} groups)"
                    )

                latitude = self.hass.config.latitude
                longitude = self.hass.config.longitude
                timezone_str = str(self.hass.config.time_zone)
                elevation_m = self.hass.config.elevation or 0
                astronomy_cache.initialize_location(latitude, longitude, timezone_str, elevation_m)

                self.weather_pipeline_manager = WeatherDataPipelineManager(
                    hass=self.hass,
                    db_manager=self.data_manager._db_manager,
                    config_entry=self.entry,
                    coordinator=self,
                )

                from .astronomy.astronomy_cache_manager import get_cache_manager
                cache_manager = get_cache_manager(db_manager=self.data_manager._db_manager)

                cache_initialized = await cache_manager.initialize()
                if not cache_initialized:
                    # Run cache rebuild in separate background task to not block other init
                    solar_capacity = self.entry.data.get("solar_capacity", 5.0)

                    async def _rebuild_cache_background():
                        try:
                            _LOGGER.info("Starting astronomy cache rebuild in background...")
                            await astronomy_cache.rebuild_cache(system_capacity_kwp=solar_capacity)
                            await cache_manager.initialize()
                            _LOGGER.info("Astronomy cache rebuild completed successfully")
                        except Exception as rebuild_err:
                            _LOGGER.error(f"Failed to rebuild astronomy cache: {rebuild_err}")

                    self.hass.async_create_task(
                        _rebuild_cache_background(),
                        name="solar_forecast_ml_cache_rebuild"
                    )
                    _LOGGER.info("Astronomy cache rebuild scheduled in background")

                pipeline_setup_ok = await self.weather_pipeline_manager.async_setup()
                if not pipeline_setup_ok:
                    _LOGGER.warning("Weather Data Pipeline setup failed - will retry on next update")
                else:
                    self.weather_service = WeatherService(
                        hass=self.hass,
                        latitude=self.hass.config.latitude,
                        longitude=self.hass.config.longitude,
                        data_dir=self.data_manager.data_dir,
                        db_manager=self.data_manager._db_manager,
                        data_manager=self.data_manager,
                        error_handler=self.error_handler,
                    )
                    await self.weather_service.initialize()

                    async def _start_pipeline_background():
                        try:
                            pipeline_start_ok = await self.weather_pipeline_manager.start_pipeline()
                            if not pipeline_start_ok:
                                _LOGGER.warning(
                                    "Weather Data Pipeline failed to start - will retry on next update"
                                )
                            else:
                                _LOGGER.info("Weather Data Pipeline started successfully")
                        except Exception as e:
                            _LOGGER.warning(
                                f"Weather Data Pipeline start failed: {e} - will retry on next update"
                            )

                    self.hass.async_create_task(
                        _start_pipeline_background(),
                        name="solar_forecast_ml_pipeline_start"
                    )

            except Exception as e:
                _LOGGER.warning(f"Weather pipeline initialization failed: {e}")

            # Schedule delayed sensor init @zara
            try:
                await self._schedule_delayed_sensor_init()
            except Exception as e:
                _LOGGER.warning(f"Delayed sensor init failed: {e}")

            # Setup power peak tracking @zara
            try:
                await self._setup_power_peak_tracking()
            except Exception as e:
                _LOGGER.warning(f"Power peak tracking setup failed: {e}")

            # Schedule all tasks @zara
            try:
                await self.scheduled_tasks.schedule_all_tasks()
            except Exception as e:
                _LOGGER.warning(f"Task scheduling failed: {e}")

            # Load hourly predictions and best hour from database @zara
            try:
                today_str = dt_util.now().date().isoformat()
                predictions = await self.data_manager.get_hourly_predictions(today_str)

                # Load best hour from database @zara
                best_hour_data = await self.data_manager.get_forecast_best_hour()

                self._hourly_predictions_cache = {
                    "predictions": predictions if predictions else [],
                    "best_hour_today": {
                        "hour": best_hour_data.get("best_hour") if best_hour_data else None,
                        "kwh": best_hour_data.get("best_hour_kwh") if best_hour_data else None,
                        "method": best_hour_data.get("method") if best_hour_data else None,
                    } if best_hour_data else {},
                }
            except Exception as e:
                _LOGGER.debug(f"Could not load hourly predictions cache: {e}")
                self._hourly_predictions_cache = {"predictions": [], "best_hour_today": {}}

            # Schedule weekly retraining @zara
            if self.ai_predictor:
                @callback
                def _scheduled_weekly_retraining(now: datetime) -> None:
                    """Weekly model retraining - Sundays only. @zara"""
                    if now.weekday() == 6:
                        self.hass.async_create_background_task(
                            self.ai_predictor.train_model(),
                            name="solar_forecast_ml_weekly_retraining",
                        )

                self._unsub_weekly_retraining_listener = async_track_time_change(
                    self.hass, _scheduled_weekly_retraining, hour=3, minute=0, second=0
                )

            ml_status = "AI-Ready" if self._ml_ready else "Rule-Based"
            _LOGGER.info(
                f"Solar Forecast Coordinator fully initialized ({ml_status}, {self.solar_capacity} kWp)"
            )

        except Exception as e:
            _LOGGER.error(f"Background initialization failed: {e}", exc_info=True)

    async def async_shutdown(self) -> None:
        """Cleanup coordinator resources. @zara

        Removes all event listeners to prevent accumulation on reload.
        """
        try:
            if self._startup_data_resolver:
                await self._startup_data_resolver.shutdown()

            if hasattr(self, 'weather_pipeline_manager') and self.weather_pipeline_manager:
                await self.weather_pipeline_manager.stop_pipeline()

            if hasattr(self, 'production_time_calculator'):
                await self.production_time_calculator.stop_tracking()
            if hasattr(self, 'scheduled_tasks'):
                await self.scheduled_tasks.cancel_all_tasks()

            if self._unsub_power_peak_listener is not None:
                try:
                    self._unsub_power_peak_listener()
                    self._unsub_power_peak_listener = None
                    _LOGGER.debug("Power peak tracking listener removed")
                except Exception as e:
                    _LOGGER.warning(f"Error removing power peak listener: {e}")

            if self._unsub_weekly_retraining_listener is not None:
                try:
                    self._unsub_weekly_retraining_listener()
                    self._unsub_weekly_retraining_listener = None
                    _LOGGER.debug("Weekly retraining listener removed")
                except Exception as e:
                    _LOGGER.warning(f"Error removing weekly retraining listener: {e}")

            if self.data_manager:
                await self.data_manager.cleanup()

        except Exception as e:
            _LOGGER.error(f"Error during coordinator shutdown: {e}")

    async def _schedule_delayed_sensor_init(self) -> None:
        """Schedule sensor initialization after Home Assistant is fully started. @zara"""

        async def _delayed_init(event=None):
            """Run sensor initialization after HA is fully started."""
            _LOGGER.info(
                "[SENSOR_INIT] Home Assistant fully started - beginning sensor initialization"
            )

            await asyncio.sleep(2)

            await self._initialize_panel_group_sensor_reader()
            await self._start_production_tracking_safe()

            # Refresh coordinator data so sensors reflect restored state immediately @zara
            await self.async_request_refresh()

            _LOGGER.info("[SENSOR_INIT] Delayed sensor initialization completed")

        if self.hass.is_running:
            _LOGGER.info(
                "[SENSOR_INIT] HA already running - starting sensor initialization immediately"
            )
            self.hass.async_create_task(
                _delayed_init(),
                name="solar_forecast_ml_delayed_sensor_init"
            )
        else:
            _LOGGER.info(
                "[SENSOR_INIT] HA still starting - scheduling sensor initialization for after startup"
            )
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _delayed_init)

    async def _initialize_panel_group_sensor_reader(self) -> None:
        """Initialize the panel group sensor reader. @zara"""
        try:
            if not self.panel_groups:
                return

            has_energy_sensor = any(g.get("energy_sensor") for g in self.panel_groups)

            if not has_energy_sensor:
                _LOGGER.debug(
                    "No panel group energy sensors configured - skipping sensor reader init"
                )
                return

            from .data.data_panel_group_sensor_reader import PanelGroupSensorReader

            self.panel_group_sensor_reader = PanelGroupSensorReader(
                hass=self.hass,
                db_manager=self.data_manager._db_manager,
                panel_groups=self.panel_groups,
            )

            await self.panel_group_sensor_reader.initialize()

            _LOGGER.info(f"Panel groups configuration: {len(self.panel_groups)} groups")
            for idx, pg in enumerate(self.panel_groups):
                _LOGGER.debug(
                    f"  Group {idx}: name={pg.get('name')}, "
                    f"energy_sensor={pg.get('energy_sensor')}"
                )

            self.hass.async_create_task(
                self._validate_panel_group_sensors_with_retry(),
                name="solar_forecast_ml_sensor_validation"
            )
            _LOGGER.info("Panel group sensor validation scheduled (background task)")

        except Exception as e:
            _LOGGER.error(f"Failed to initialize panel group sensor reader: {e}")
            self.panel_group_sensor_reader = None

    async def _validate_panel_group_sensors_with_retry(self) -> None:
        """Validate panel group sensors with exponential backoff retry. @zara"""
        if not self.panel_group_sensor_reader:
            _LOGGER.debug("Panel group sensor reader not initialized - skipping validation")
            return

        max_retries = 4
        base_delay = 5

        for attempt in range(1, max_retries + 1):
            retry_delay = base_delay * (2 ** (attempt - 1))

            try:
                validation_results = await self.panel_group_sensor_reader.validate_sensors()
            except Exception as e:
                _LOGGER.warning(f"Sensor validation attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                continue

            valid_count = sum(1 for r in validation_results.values() if r.get("valid"))
            total_count = len(validation_results)

            if valid_count == total_count:
                _LOGGER.info(
                    f"Panel group sensor reader initialized: {total_count} energy sensors validated"
                )
                return

            invalid_sensors = [
                (name, result.get("error", "Unknown"))
                for name, result in validation_results.items()
                if not result.get("valid")
            ]

            if attempt < max_retries:
                _LOGGER.debug(
                    f"Panel group sensor validation: {valid_count}/{total_count} valid "
                    f"(attempt {attempt}/{max_retries}). Retrying in {retry_delay}s..."
                )
                for name, error in invalid_sensors:
                    _LOGGER.debug(f"  - {name}: {error}")

                await asyncio.sleep(retry_delay)
            else:
                _LOGGER.warning(
                    f"Panel group sensor reader: {valid_count}/{total_count} sensors valid "
                    f"after {max_retries} attempts"
                )
                for name, error in invalid_sensors:
                    _LOGGER.warning(f"  - {name}: {error}")
                _LOGGER.warning(
                    "Panel groups with invalid sensors will use fallback (proportional distribution)."
                )

    async def _start_production_tracking_safe(self) -> None:
        """Start production tracking with error handling. @zara"""
        try:
            await self.production_time_calculator.start_tracking()
        except Exception as track_err:
            _LOGGER.error(f"Failed to start production time tracking: {track_err}")

    async def _setup_power_peak_tracking(self) -> None:
        """Setup event listener for power peak tracking. @zara"""
        if not self.power_entity:
            return

        # Initialize peak tracking attributes @zara
        self._peak_power_today = 0.0
        self._peak_time_today = None

        # Load today's existing peak from production_time_state @zara
        try:
            today = dt_util.now().date().isoformat()
            row = await self.data_manager._db_manager.fetchone(
                """SELECT peak_power_w, peak_power_time
                   FROM production_time_state
                   WHERE id = 1 AND date = ?""",
                (today,)
            )
            if row and row[0]:
                self._peak_power_today = float(row[0])
                self._peak_time_today = str(row[1]) if row[1] else None
        except Exception as e:
            _LOGGER.debug(f"Could not load today's peak: {e}")

        last_write_time = None

        async def power_state_changed(event):
            nonlocal last_write_time

            new_state = event.data.get("new_state")
            if not new_state or new_state.state in [None, "unavailable", "unknown"]:
                return

            try:
                power_w = float(new_state.state)

                if power_w > self._peak_power_today:
                    self._peak_power_today = power_w
                    now = dt_util.now()
                    self._peak_time_today = now.strftime("%H:%M")

                    if last_write_time is None or (now - last_write_time).total_seconds() > 60:
                        today = now.date().isoformat()
                        time_str = now.strftime("%H:%M")

                        # Save today's peak to production_time_state @zara
                        await self.data_manager._db_manager.execute(
                            """UPDATE production_time_state
                               SET peak_power_w = ?, peak_power_time = ?,
                                   date = ?, last_updated = ?
                               WHERE id = 1""",
                            (power_w, time_str, today, now.isoformat())
                        )

                        # Check and update all-time record @zara
                        all_time_peak = await self.data_manager.get_all_time_peak()
                        if all_time_peak is None or power_w > all_time_peak:
                            await self.data_manager._db_manager.execute(
                                """UPDATE production_time_state
                                   SET peak_record_w = ?, peak_record_date = ?, peak_record_time = ?
                                   WHERE id = 1""",
                                (power_w, today, time_str)
                            )
                            _LOGGER.info("New all-time peak: %.1f W at %s", power_w, time_str)

                        last_write_time = now

            except (ValueError, TypeError):
                pass

        self._unsub_power_peak_listener = async_track_state_change_event(
            self.hass, [self.power_entity], power_state_changed
        )

    async def _async_update_data(self):
        """Fetch data from API endpoint. @zara"""
        try:
            if not self._services_initialized:
                services_ok = await self._initialize_services()
                if not services_ok:
                    raise UpdateFailed("Failed to initialize services")

            await self._initialize_forecast_orchestrator()

            from .core.core_coordinator_update_helpers import CoordinatorUpdateHelpers

            helpers = CoordinatorUpdateHelpers(self, self.data_manager._db_manager)

            await helpers.handle_startup_recovery()

            current_weather, hourly_forecast = await helpers.fetch_weather_data()

            external_sensors = self.sensor_collector.collect_all_sensor_data_dict()

            forecast = await helpers.generate_forecast(
                current_weather, hourly_forecast, external_sensors
            )

            result = await helpers.build_coordinator_result(
                forecast, current_weather, external_sensors
            )

            await self._update_sensor_properties(result)

            await helpers.save_forecasts(forecast, forecast.get("hourly", []))

            # Refresh hourly predictions cache after saving to DB @zara
            await self._refresh_hourly_predictions_cache()

            self._last_update_success_time = dt_util.now()

            if not self._startup_sensors_ready:
                self._startup_sensors_ready = True

            return result

        except UpdateFailed:
            raise
        except Exception as err:
            _LOGGER.error(f"Unexpected error updating data: {err}")
            raise UpdateFailed(f"Error communicating with API: {err}")

    async def _update_sensor_properties(self, data: Dict[str, Any]) -> None:
        """Update coordinator properties used by sensors. @zara"""
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

            external = data.get(DATA_KEY_EXTERNAL_SENSORS, {})
            solar_yield_kwh = external.get(EXT_SENSOR_SOLAR_YIELD_TODAY)
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

        ai_predictor = self.ai_predictor
        if ai_predictor:
            self.last_successful_learning = getattr(ai_predictor, "last_training_time", None)
            if self.model_accuracy is None:
                self.model_accuracy = getattr(ai_predictor, "current_accuracy", None)

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

        # Load average daily yield for current month from DB @zara
        try:
            month_start = dt_util.now().date().replace(day=1).isoformat()
            row = await self.data_manager._db_manager.fetchone(
                """SELECT AVG(actual_total_kwh) FROM daily_summaries
                   WHERE date >= ? AND actual_total_kwh IS NOT NULL""",
                (month_start,)
            )
            if row and row[0] is not None:
                self.avg_month_yield = round(float(row[0]), 2)
        except Exception as e:
            _LOGGER.debug(f"Could not load avg_month_yield: {e}")

    async def _refresh_hourly_predictions_cache(self) -> None:
        """Refresh hourly predictions cache from database. @zara"""
        try:
            today_str = dt_util.now().date().isoformat()
            predictions = await self.data_manager.get_hourly_predictions(today_str)
            best_hour_data = await self.data_manager.get_forecast_best_hour()

            self._hourly_predictions_cache = {
                "predictions": predictions if predictions else [],
                "best_hour_today": {
                    "hour": best_hour_data.get("best_hour") if best_hour_data else None,
                    "kwh": best_hour_data.get("best_hour_kwh") if best_hour_data else None,
                    "method": best_hour_data.get("method") if best_hour_data else None,
                } if best_hour_data else {},
            }
            _LOGGER.debug("Hourly predictions cache refreshed: %d predictions", len(predictions) if predictions else 0)
        except Exception as e:
            _LOGGER.warning(f"Could not refresh hourly predictions cache: {e}")

    @property
    def last_update_success_time(self) -> Optional[datetime]:
        """Return last successful update time. @zara"""
        return self._last_update_success_time

    @property
    def weather_source(self) -> str:
        """Return weather source. @zara"""
        return self.current_weather_entity or "Open-Meteo (direct radiation)"

    @property
    def diagnostic_status(self) -> str:
        """Return diagnostic status string. @zara"""
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
            reason = "ML Disabled/Unavailable" if not self.ai_predictor else "ML Not Ready"
            return f"Degraded ({reason})"
        elif not weather_healthy:
            return "Error (Weather Unavailable)"
        elif not update_age_ok:
            return "Stale (No Recent Update)"
        else:
            return "Initializing"

    def on_ai_training_complete(
        self, timestamp: datetime, accuracy: Optional[float] = None
    ) -> None:
        """Callback when AI training completes. @zara"""
        _LOGGER.info(
            f"Coordinator notified of AI Training completion at {timestamp}. Accuracy: {accuracy}"
        )
        self.last_successful_learning = timestamp
        if accuracy is not None:
            self.model_accuracy = accuracy
        self.async_update_listeners()

        if accuracy is not None:
            samples = self.ai_predictor.training_samples if self.ai_predictor else 0
            self.update_system_status(
                event_type="ai_training",
                event_status="success",
                event_summary=f"AI Training successful - Accuracy: {accuracy*100:.1f}%",
                event_details={
                    "accuracy_percent": round(accuracy * 100, 1),
                    "samples_used": samples,
                    "training_time": timestamp.isoformat(),
                },
            )
        else:
            self.update_system_status(
                event_type="ai_training",
                event_status="failed",
                event_summary="AI Training failed",
                event_details={},
            )

    async def set_expected_daily_production(self) -> None:
        """Set expected daily production at 6 AM and save to database. @zara"""
        try:
            if (
                self.data
                and DATA_KEY_FORECAST_TODAY in self.data
                and self.data.get(DATA_KEY_FORECAST_TODAY) is not None
            ):
                self.expected_daily_production = self.data.get(DATA_KEY_FORECAST_TODAY)
            else:
                await self.async_request_refresh()

                for i in range(10):
                    if (
                        self.data
                        and DATA_KEY_FORECAST_TODAY in self.data
                        and self.data.get(DATA_KEY_FORECAST_TODAY) is not None
                    ):
                        break
                    await asyncio.sleep(1.0)

                if (
                    self.data
                    and DATA_KEY_FORECAST_TODAY in self.data
                    and self.data.get(DATA_KEY_FORECAST_TODAY) is not None
                ):
                    self.expected_daily_production = self.data.get(DATA_KEY_FORECAST_TODAY)
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
                    _LOGGER.error("CRITICAL: daily_forecast NOT saved to database!")

                self.async_update_listeners()

        except Exception as err:
            _LOGGER.error(f"Failed to set expected daily production: {err}")
            self.expected_daily_production = None

    async def reset_expected_daily_production(self) -> None:
        """Reset expected daily production at midnight. @zara"""
        self.expected_daily_production = None
        await self.data_manager.clear_expected_daily_production()
        self.async_update_listeners()

    async def _recovery_forecast_process(self, source: str) -> bool:
        """Fallback process for missing forecasts. @zara"""
        async with self._recovery_lock:
            if self._recovery_in_progress:
                return False

            self._recovery_in_progress = True
            try:
                return await self._execute_recovery(source)
            finally:
                self._recovery_in_progress = False

    async def _execute_recovery(self, source: str) -> bool:
        """Internal method to execute the recovery process. @zara"""
        if not self.weather_service:
            _LOGGER.error("Weather service not initialized for recovery")
            return False

        existing_forecast = await self.data_manager.get_current_day_forecast()
        if existing_forecast and existing_forecast.get("locked"):
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
        """Force refresh with immediate weather update. @zara"""
        if self.weather_service:
            await self.weather_service.force_update()

        await self.async_request_refresh()

    async def forecast_day_after_tomorrow(self) -> None:
        """Triggers and saves the forecast for the day after tomorrow. @zara"""
        try:
            current_weather = (
                await self.weather_service.get_current_weather()
                if self.weather_service
                else None
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
                target_date=day_after_date,
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
        """Update system status sensor with event information. @zara"""
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
        """Count training-ready samples from database. @zara"""
        try:
            today_str = dt_util.now().date().isoformat()
            predictions = await self.data_manager.get_hourly_predictions(
                today_str, with_actual=True
            )
            return len(predictions)
        except Exception:
            return 0
