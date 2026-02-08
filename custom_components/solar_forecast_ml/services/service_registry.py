# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

# *****************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# Refactored: JSON replaced with DatabaseManager @zara
# *****************************************************************************

"""
Service registry for Solar Forecast ML.
Central registration and handling of all integration services.
Uses DatabaseManager for all data operations.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Awaitable, Callable, List, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall

from ..const import (
    DOMAIN,
    SERVICE_ANALYZE_FEATURE_IMPORTANCE,
    SERVICE_BACKFILL_SHADOW_DETECTION,
    SERVICE_BUILD_ASTRONOMY_CACHE,
    SERVICE_RUN_WEATHER_CORRECTION,
    SERVICE_REFRESH_OPEN_METEO_CACHE,
    SERVICE_REFRESH_MULTI_WEATHER,
    SERVICE_REFRESH_CACHE_TODAY,
    SERVICE_RESET_AI_MODEL,
    SERVICE_RETRAIN_AI_MODEL,
    SERVICE_RUN_ALL_DAY_END_TASKS,
    SERVICE_RUN_GRID_SEARCH,
    SERVICE_SEND_DAILY_BRIEFING,
    SERVICE_TEST_MORNING_ROUTINE,
    SERVICE_TEST_RETROSPECTIVE_FORECAST,
)
from ..const import CONF_WINTER_MODE, DEFAULT_WINTER_MODE
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..data.db_manager import DatabaseManager

_LOGGER = logging.getLogger(__name__)


@dataclass
class ServiceDefinition:
    """Service definition for registration. @zara"""

    name: str
    handler: Callable[[ServiceCall], Awaitable[None]]
    description: str = ""


class ServiceRegistry:
    """Central service registry for Solar Forecast ML. @zara"""

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, coordinator: "SolarForecastMLCoordinator"
    ):
        """Initialize service registry. @zara"""
        self.hass = hass
        self.entry = entry
        self.coordinator = coordinator
        self._registered_services: List[str] = []

        self._astronomy_handler = None
        self._daily_briefing_handler = None

    @property
    def db_manager(self) -> Optional[DatabaseManager]:
        """Get database manager from coordinator. @zara V16.1 fix"""
        data_manager = getattr(self.coordinator, "data_manager", None)
        if data_manager:
            return getattr(data_manager, "_db_manager", None)
        return None

    async def async_register_all_services(self) -> None:
        """Register all services. @zara"""
        from ..services.service_astronomy import AstronomyServiceHandler

        self._astronomy_handler = AstronomyServiceHandler(self.hass, self.entry, self.coordinator)
        await self._astronomy_handler.initialize()

        from ..services.service_daily_briefing import DailyBriefingService

        self._daily_briefing_handler = DailyBriefingService(self.hass, self.coordinator)

        services = self._build_service_definitions()

        for service in services:
            self.hass.services.async_register(DOMAIN, service.name, service.handler)
            self._registered_services.append(service.name)

        _LOGGER.debug(f"Registered {len(services)} services")

    def unregister_all_services(self) -> None:
        """Unregister all services. @zara"""
        for service_name in self._registered_services:
            if self.hass.services.has_service(DOMAIN, service_name):
                self.hass.services.async_remove(DOMAIN, service_name)

        self._registered_services.clear()

    def _build_service_definitions(self) -> List[ServiceDefinition]:
        """Build all service definitions. @zara"""
        return [
            # AI Services
            ServiceDefinition(
                name=SERVICE_RETRAIN_AI_MODEL,
                handler=self._handle_retrain_ai_model,
                description="Retrain TinyLSTM AI model with current data",
            ),
            ServiceDefinition(
                name=SERVICE_RESET_AI_MODEL,
                handler=self._handle_reset_ai_model,
                description="Reset TinyLSTM AI model to untrained state",
            ),
            ServiceDefinition(
                name=SERVICE_RUN_GRID_SEARCH,
                handler=self._handle_run_grid_search,
                description="Run Grid-Search hyperparameter optimization",
            ),
            ServiceDefinition(
                name=SERVICE_ANALYZE_FEATURE_IMPORTANCE,
                handler=self._handle_analyze_feature_importance,
                description="Analyze feature importance using Permutation Importance",
            ),
            # Emergency Services
            ServiceDefinition(
                name=SERVICE_RUN_ALL_DAY_END_TASKS,
                handler=self._handle_run_all_day_end_tasks,
                description="Emergency: Run all day-end tasks",
            ),
            # Testing Services
            ServiceDefinition(
                name=SERVICE_TEST_MORNING_ROUTINE,
                handler=self._handle_test_morning_routine,
                description="Test: Complete 6 AM morning routine",
            ),
            ServiceDefinition(
                name=SERVICE_TEST_RETROSPECTIVE_FORECAST,
                handler=self._handle_test_retrospective_forecast,
                description="Test: Create retrospective forecast for today",
            ),
            # Weather Services
            ServiceDefinition(
                name=SERVICE_RUN_WEATHER_CORRECTION,
                handler=self._handle_run_weather_correction,
                description="Manually trigger corrected forecast generation",
            ),
            ServiceDefinition(
                name=SERVICE_REFRESH_OPEN_METEO_CACHE,
                handler=self._handle_refresh_open_meteo_cache,
                description="Refresh Open-Meteo cloud cover cache",
            ),
            ServiceDefinition(
                name=SERVICE_REFRESH_MULTI_WEATHER,
                handler=self._handle_refresh_multi_weather,
                description="Refresh Multi-Weather cache (5-source blending)",
            ),
            # Astronomy Services
            ServiceDefinition(
                name=SERVICE_BUILD_ASTRONOMY_CACHE,
                handler=self._handle_build_astronomy_cache,
                description="Build astronomy cache for date range",
            ),
            ServiceDefinition(
                name=SERVICE_REFRESH_CACHE_TODAY,
                handler=self._handle_refresh_cache_today,
                description="Refresh Astro-Cache",
            ),
            # Notification Services
            ServiceDefinition(
                name=SERVICE_SEND_DAILY_BRIEFING,
                handler=self._handle_send_daily_briefing,
                description="Send daily solar weather briefing notification",
            ),
            # Shadow Detection Services @zara V16.2
            ServiceDefinition(
                name=SERVICE_BACKFILL_SHADOW_DETECTION,
                handler=self._handle_backfill_shadow_detection,
                description="Backfill shadow detection for missing hours",
            ),
        ]

    # =========================================================================
    # AI Services
    # =========================================================================

    async def _handle_retrain_ai_model(self, call: ServiceCall) -> None:
        """Handle retrain_ai_model service. @zara"""
        try:
            if self.coordinator.ai_predictor:
                _LOGGER.info("Service: retrain_ai_model - Starting AI training")
                result = await self.coordinator.ai_predictor.train_model()
                if result.success:
                    _LOGGER.info(
                        f"AI model training complete: R2={result.accuracy:.3f}, "
                        f"samples={result.samples_used}"
                    )
                else:
                    _LOGGER.error(f"AI model training failed: {result.error_message}")
            else:
                _LOGGER.warning("AI predictor not available")
        except Exception as e:
            _LOGGER.error(f"Error in retrain_ai_model: {e}")

    async def _handle_reset_ai_model(self, call: ServiceCall) -> None:
        """Handle reset_ai_model service. @zara"""
        try:
            if self.coordinator.ai_predictor:
                _LOGGER.info("Service: reset_ai_model - Resetting AI model")
                success = await self.coordinator.ai_predictor.initialize()
                if success:
                    _LOGGER.info("AI model reset to untrained state")
                else:
                    _LOGGER.error("AI model reset failed")
            else:
                _LOGGER.warning("AI predictor not available")
        except Exception as e:
            _LOGGER.error(f"Error in reset_ai_model: {e}")

    async def _handle_run_grid_search(self, call: ServiceCall) -> None:
        """Handle run_grid_search service - Run hyperparameter optimization. @zara"""
        from ..ai import GridSearchOptimizer, TinyLSTM
        from ..ai.ai_grid_search import detect_hardware

        _LOGGER.info("SERVICE: run_grid_search - Starting in background")

        # Run actual grid search in background to not block
        async def _run_grid_search_background():
            try:
                # Run hardware detection in executor to avoid blocking I/O warnings
                hw_info = await self.hass.async_add_executor_job(detect_hardware)
                _LOGGER.info(f"Hardware: {hw_info.architecture}, {hw_info.cpu_count} CPUs")

                if not hw_info.grid_search_allowed:
                    _LOGGER.warning(f"Grid-Search not available: {hw_info.reason}")
                    return

                if not self.coordinator.ai_predictor:
                    _LOGGER.error("AI predictor not available")
                    return

                predictor = self.coordinator.ai_predictor

                _LOGGER.info("Loading training data...")
                X_sequences, y_targets, _ = await predictor._prepare_training_data()

                if len(X_sequences) < 50:
                    _LOGGER.error(f"Not enough training data: {len(X_sequences)} samples (need 50+)")
                    return

                _LOGGER.info(f"Loaded {len(X_sequences)} training samples")

                optimizer = GridSearchOptimizer(
                    db_manager=predictor.db_manager,
                    hardware_info=hw_info  # Pass cached hardware info to avoid re-detection
                )

                async def progress_callback(current, total, params, accuracy):
                    _LOGGER.info(
                        f"Grid-Search progress: {current}/{total} - "
                        f"hidden={params.get('hidden_size')}, R2={accuracy:.4f}"
                    )

                from ..ai.ai_predictor import calculate_feature_count

                feature_count = calculate_feature_count(predictor.num_groups)
                num_outputs = predictor.num_groups if predictor.num_groups > 0 else 1

                result = await optimizer.run_grid_search(
                    lstm_class=TinyLSTM,
                    X_sequences=X_sequences,
                    y_targets=y_targets,
                    input_size=feature_count,
                    sequence_length=24,
                    num_outputs=num_outputs,
                    progress_callback=progress_callback,
                )

                if not result.success:
                    _LOGGER.error(f"Grid-Search failed: {result.error_message}")
                    return

                _LOGGER.info(f"GRID-SEARCH COMPLETE: Best R2={result.best_accuracy:.4f}")

                retrain_after = call.data.get("retrain_after", True)

                if retrain_after and result.best_params:
                    _LOGGER.info("Retraining model with optimal parameters...")
                    predictor.lstm = TinyLSTM(
                        input_size=feature_count,
                        hidden_size=result.best_params.get("hidden_size", 32),
                        sequence_length=24,
                        num_outputs=num_outputs,
                        learning_rate=result.best_params.get("learning_rate", 0.005),
                    )
                    train_result = await predictor.train_model()
                    if train_result.success:
                        _LOGGER.info(f"Retrained model: R2={train_result.accuracy:.4f}")

            except Exception as e:
                _LOGGER.error(f"Error in run_grid_search: {e}", exc_info=True)

        # Start grid search in background
        self.hass.async_create_task(
            _run_grid_search_background(),
            name="solar_forecast_ml_grid_search"
        )
        _LOGGER.info("Grid-Search started in background")

    async def _handle_analyze_feature_importance(self, call: ServiceCall) -> None:
        """Handle analyze_feature_importance service. @zara"""
        _LOGGER.info("SERVICE: analyze_feature_importance")

        try:
            if not self.coordinator.ai_predictor:
                _LOGGER.error("AI predictor not available")
                return

            predictor = self.coordinator.ai_predictor
            num_permutations = call.data.get("num_permutations", 5)

            async def progress_callback(current, total, feature_name):
                _LOGGER.debug(f"Feature Importance: {current}/{total} - {feature_name}")

            result = await predictor.analyze_feature_importance(
                num_permutations=num_permutations,
                progress_callback=progress_callback,
            )

            if result is None or not result.success:
                error_msg = result.error_message if result else "Unknown error"
                _LOGGER.error(f"Feature Importance analysis failed: {error_msg}")
                return

            _LOGGER.info(f"FEATURE IMPORTANCE COMPLETE: Baseline RMSE={result.baseline_rmse:.4f}")

        except Exception as e:
            _LOGGER.error(f"Error in analyze_feature_importance: {e}", exc_info=True)

    # =========================================================================
    # Emergency Services
    # =========================================================================

    async def _handle_run_all_day_end_tasks(self, call: ServiceCall) -> None:
        """Handle run_all_day_end_tasks service. @zara"""
        try:
            if hasattr(self.coordinator, "scheduled_tasks"):
                await self.coordinator.scheduled_tasks.end_of_day_workflow(None)
        except Exception as e:
            _LOGGER.error(f"Error in run_all_day_end_tasks: {e}")

    # =========================================================================
    # Testing Services
    # =========================================================================

    async def _handle_test_morning_routine(self, call: ServiceCall) -> None:
        """Handle test_morning_routine service. @zara"""
        try:
            if hasattr(self.coordinator, "scheduled_tasks"):
                await self.coordinator.scheduled_tasks._execute_morning_routine()
        except Exception as e:
            _LOGGER.error(f"Error in test_morning_routine: {e}")

    async def _handle_test_retrospective_forecast(self, call: ServiceCall) -> None:
        """Handle test_retrospective_forecast service - Create retrospective forecast. @zara"""
        _LOGGER.info("SERVICE: test_retrospective_forecast")

        try:
            sunrise = await self._get_sunrise_for_today()
            if not sunrise:
                _LOGGER.error("Could not determine sunrise time - using fallback 08:00")
                now = dt_util.now()
                sunrise = datetime.combine(now.date(), datetime.min.time().replace(hour=8))
                if now.tzinfo:
                    sunrise = sunrise.replace(tzinfo=now.tzinfo)

            simulation_time = sunrise - timedelta(hours=1)
            today_str = dt_util.now().date().isoformat()

            _LOGGER.info(f"Sunrise today: {sunrise.strftime('%H:%M')}")
            _LOGGER.info(f"Simulation time: {simulation_time.strftime('%H:%M')}")

            weather_service = self.coordinator.weather_service
            if not weather_service:
                _LOGGER.error("Weather service not available")
                return

            hourly_weather_forecast = await weather_service.get_corrected_hourly_forecast()
            if not hourly_weather_forecast:
                _LOGGER.error("No corrected weather forecast available")
                return

            current_weather = await weather_service.get_current_weather()
            external_sensors = self.coordinator.sensor_collector.collect_all_sensor_data_dict()

            forecast = await self.coordinator.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_weather_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=self.coordinator.learned_correction_factor,
            )

            if not forecast or not forecast.get("hourly"):
                _LOGGER.error("Forecast generation failed")
                return

            all_hourly = forecast.get("hourly", [])
            today_hourly = [h for h in all_hourly if h.get("date") == today_str]

            _LOGGER.info(f"Generated {len(today_hourly)} hourly predictions for today")

            # Save to DB instead of JSON
            db = self.db_manager
            if db:
                for hour_data in today_hourly:
                    await db.execute(
                        """INSERT OR REPLACE INTO retrospective_forecasts
                           (date, hour, predicted_kwh, simulation_time, created_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            today_str,
                            hour_data.get("hour"),
                            hour_data.get("production_kwh", 0.0),
                            simulation_time.isoformat(),
                            dt_util.now().isoformat(),
                        ),
                    )

            retrospective_today_kwh = sum(
                h.get("production_kwh", 0.0) or 0.0 for h in today_hourly
            )

            _LOGGER.info(f"RETROSPECTIVE FORECAST COMPLETE: {retrospective_today_kwh:.2f} kWh")

        except Exception as e:
            _LOGGER.error(f"Error in test_retrospective_forecast: {e}", exc_info=True)

    async def _get_sunrise_for_today(self) -> Optional[datetime]:
        """Get sunrise time for today from astronomy cache. @zara"""
        try:
            from ..astronomy.astronomy_cache_manager import get_cache_manager

            cache_manager = get_cache_manager()
            today = dt_util.now().date().isoformat()

            day_data = cache_manager.get_day_data(today)
            if not day_data:
                return None

            sunrise_str = day_data.get("sunrise_local")
            if not sunrise_str:
                return None

            sunrise = datetime.fromisoformat(sunrise_str)

            if sunrise.tzinfo is None:
                local_tz = dt_util.now().tzinfo
                if local_tz:
                    sunrise = sunrise.replace(tzinfo=local_tz)

            return sunrise

        except Exception as e:
            _LOGGER.error(f"Error getting sunrise: {e}")
            return None

    # =========================================================================
    # Weather Services
    # =========================================================================

    async def _handle_run_weather_correction(self, call: ServiceCall) -> None:
        """Handle run_weather_correction service. @zara"""
        try:
            from ..data.data_weather_corrector import WeatherForecastCorrector

            winter_mode = DEFAULT_WINTER_MODE
            if self.coordinator.config_entry:
                winter_mode = self.coordinator.config_entry.options.get(
                    CONF_WINTER_MODE, DEFAULT_WINTER_MODE
                )

            corrector = WeatherForecastCorrector(
                self.hass,
                self.coordinator.data_manager._db_manager,
                winter_mode=winter_mode,
            )
            success = await corrector.create_corrected_forecast()

            if not success:
                _LOGGER.warning("Corrected forecast generation failed")

        except Exception as e:
            _LOGGER.error(f"Error in run_weather_correction: {e}")

    async def _handle_refresh_open_meteo_cache(self, call: ServiceCall) -> None:
        """Handle refresh_open_meteo_cache service. @zara"""
        try:
            if not hasattr(self.coordinator, "weather_pipeline_manager"):
                _LOGGER.warning("Weather pipeline manager not available")
                return

            pipeline = self.coordinator.weather_pipeline_manager
            success = await pipeline.update_weather_cache()

            if success:
                _LOGGER.info("Weather cache refreshed via 5-source ExpertBlender")
            else:
                _LOGGER.warning("Weather cache refresh failed")

        except Exception as e:
            _LOGGER.error(f"Error in refresh_open_meteo_cache: {e}", exc_info=True)

    async def _handle_refresh_multi_weather(self, call: ServiceCall) -> None:
        """Handle refresh_multi_weather service. @zara"""
        try:
            if not hasattr(self.coordinator, "weather_pipeline_manager"):
                _LOGGER.warning("Weather pipeline manager not available")
                return

            pipeline = self.coordinator.weather_pipeline_manager
            force_update = call.data.get("force_update", False) or call.data.get(
                "force_wttr_refresh", False
            )

            _LOGGER.info(f"Service: refresh_multi_weather (force={force_update})")

            success = await pipeline.update_weather_cache(force=force_update)

            if success:
                stats = {}
                if pipeline.weather_expert_blender:
                    stats = pipeline.weather_expert_blender.get_blend_stats()
                _LOGGER.info(
                    f"5-source weather refresh complete: "
                    f"{stats.get('active_sources', 0)} sources active"
                )
            else:
                _LOGGER.warning("Weather refresh failed")

        except Exception as e:
            _LOGGER.error(f"Error in refresh_multi_weather: {e}", exc_info=True)

    # =========================================================================
    # Astronomy Services
    # =========================================================================

    async def _handle_build_astronomy_cache(self, call: ServiceCall) -> None:
        """Handle build_astronomy_cache service. @zara"""
        if self._astronomy_handler:
            await self._astronomy_handler.handle_build_astronomy_cache(call)

    async def _handle_refresh_cache_today(self, call: ServiceCall) -> None:
        """Handle refresh_cache_today service. @zara"""
        if self._astronomy_handler:
            await self._astronomy_handler.handle_refresh_cache_today(call)

    # =========================================================================
    # Notification Services
    # =========================================================================

    async def _handle_send_daily_briefing(self, call: ServiceCall) -> None:
        """Handle send_daily_briefing service. @zara"""
        _LOGGER.info("Service: send_daily_briefing")
        try:
            if not self._daily_briefing_handler:
                _LOGGER.error("Daily briefing handler not initialized")
                return

            notify_service = call.data.get("notify_service", "persistent_notification")
            language = call.data.get("language", "de")

            result = await self._daily_briefing_handler.send_daily_briefing(
                notify_service=notify_service,
                language=language,
            )

            if result.get("success"):
                _LOGGER.info(f"Daily briefing sent successfully: {result.get('title')}")
            else:
                _LOGGER.error(f"Failed to send daily briefing: {result.get('error')}")

        except Exception as err:
            _LOGGER.error(f"Error in send_daily_briefing service: {err}", exc_info=True)

    # =========================================================================
    # Shadow Detection Services @zara V16.2
    # =========================================================================

    async def _handle_backfill_shadow_detection(self, call: ServiceCall) -> None:
        """Handle backfill_shadow_detection service. @zara V16.2

        Runs shadow detection for hours that have actual production data
        but no shadow detection results.
        """
        _LOGGER.info("Service: backfill_shadow_detection")
        try:
            target_date = call.data.get("target_date")
            if not target_date:
                target_date = dt_util.now().date().isoformat()

            _LOGGER.info(f"Backfilling shadow detection for {target_date}")

            # Get morning routine handler for shadow detection
            if not hasattr(self.coordinator, 'scheduled_tasks') or \
               not self.coordinator.scheduled_tasks or \
               not self.coordinator.scheduled_tasks.morning_routine_handler:
                _LOGGER.error("Morning routine handler not available")
                return

            morning_handler = self.coordinator.scheduled_tasks.morning_routine_handler

            # Find hours with actual_kwh but no shadow detection
            missing_hours = await self.db_manager.fetchall(
                """SELECT hp.target_hour, hp.prediction_id, hp.actual_kwh
                   FROM hourly_predictions hp
                   LEFT JOIN hourly_shadow_detection hsd ON hp.prediction_id = hsd.prediction_id
                   WHERE hp.target_date = ?
                     AND hp.actual_kwh IS NOT NULL
                     AND hp.actual_kwh > 0
                     AND hsd.prediction_id IS NULL
                     AND hp.target_hour >= 6
                     AND hp.target_hour <= 20
                   ORDER BY hp.target_hour""",
                (target_date,)
            )

            if not missing_hours:
                _LOGGER.info(f"No missing shadow detection for {target_date}")
                return

            _LOGGER.info(f"Found {len(missing_hours)} hours missing shadow detection")

            # Get shadow detector
            shadow_detector = morning_handler._shadow_detector
            if not shadow_detector:
                _LOGGER.error("Shadow detector not available")
                return

            processed = 0
            errors = 0

            for row in missing_hours:
                hour = row[0]
                prediction_id = row[1]
                actual_kwh = row[2]

                try:
                    # Get astronomy data for this hour
                    astronomy_data = await morning_handler._get_astronomy_for_hour(target_date, hour)

                    if not astronomy_data:
                        _LOGGER.debug(f"No astronomy data for hour {hour}")
                        continue

                    # Get weather actual data including snow/frost flags @zara V16.2
                    weather_row = await self.db_manager.fetchone(
                        """SELECT temperature_c, humidity_percent, cloud_cover_percent,
                                  precipitation_mm, wind_speed_ms, snow_covered_panels,
                                  frost_detected
                           FROM hourly_weather_actual
                           WHERE date = ? AND hour = ?""",
                        (target_date, hour)
                    )

                    weather_actual = {}
                    if weather_row:
                        weather_actual = {
                            "temperature_c": weather_row[0],
                            "humidity_percent": weather_row[1],
                            "cloud_cover_percent": weather_row[2],
                            "precipitation_mm": weather_row[3],
                            "wind_speed_ms": weather_row[4],
                            "snow_covered_panels": weather_row[5],
                            "frost_detected": weather_row[6],
                        }

                    # V16.2: Also check snow_tracking table for historical snow periods @zara
                    # This catches cases where hourly_weather_actual doesn't have the flag
                    if not weather_actual.get("snow_covered_panels"):
                        from datetime import datetime as dt
                        target_datetime = f"{target_date}T{hour:02d}:00:00"
                        snow_track_row = await self.db_manager.fetchone(
                            """SELECT panels_covered_since, cleared_at
                               FROM snow_tracking
                               WHERE id = 1""",
                            ()
                        )
                        if snow_track_row and snow_track_row[0]:
                            covered_since = snow_track_row[0]
                            cleared_at = snow_track_row[1]
                            # Check if target_datetime falls within snow coverage period
                            if covered_since <= target_datetime:
                                if cleared_at is None or cleared_at > target_datetime:
                                    weather_actual["snow_covered_panels"] = True
                                    _LOGGER.debug(
                                        "Backfill: Snow detected from snow_tracking for %s hour %d",
                                        target_date, hour
                                    )

                    # Build prediction dict for shadow detector
                    prediction_for_shadow = {
                        "actual_kwh": actual_kwh,
                        "astronomy": astronomy_data,
                        "target_hour": hour,
                        "weather_actual": weather_actual,
                    }

                    # Run ensemble shadow detection
                    shadow_result = await shadow_detector.detect_shadow_ensemble(
                        prediction_for_shadow
                    )

                    if shadow_result:
                        # Save shadow detection results to database
                        await self.db_manager.save_hourly_shadow_detection(prediction_id, shadow_result)
                        processed += 1
                        _LOGGER.debug(
                            "Shadow detection for %s hour %d: %s (%.1f%%)",
                            target_date, hour,
                            shadow_result.get("shadow_type", "unknown"),
                            shadow_result.get("shadow_percent", 0)
                        )

                except Exception as e:
                    errors += 1
                    _LOGGER.warning(f"Failed shadow detection for hour {hour}: {e}")

            # Refresh coordinator cache
            if hasattr(self.coordinator, '_refresh_hourly_predictions_cache'):
                await self.coordinator._refresh_hourly_predictions_cache()
                self.coordinator.async_update_listeners()

            _LOGGER.info(
                f"Shadow detection backfill complete for {target_date}: "
                f"{processed} processed, {errors} errors"
            )

        except Exception as err:
            _LOGGER.error(f"Error in backfill_shadow_detection service: {err}", exc_info=True)
