# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Awaitable, Callable, List

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall

from ..const import (
    DOMAIN,
    SERVICE_ANALYZE_FEATURE_IMPORTANCE,
    SERVICE_BORG_ASSIMILATION_REVERSE,
    SERVICE_BUILD_ASTRONOMY_CACHE,
    SERVICE_INSTALL_EXTRA_FEATURES,
    SERVICE_REPAIR_CALIBRATION,
    SERVICE_RESCUE_CALIBRATION,
    SERVICE_WINTER_BUCKET_CORRECTION,
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
from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


@dataclass
class ServiceDefinition:
    """Service definition for registration"""

    name: str
    handler: Callable[[ServiceCall], Awaitable[None]]
    description: str = ""


class ServiceRegistry:
    """Central service registry for Solar Forecast ML"""

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, coordinator: "SolarForecastMLCoordinator"
    ):
        """Initialize service registry"""
        self.hass = hass
        self.entry = entry
        self.coordinator = coordinator
        self._registered_services: List[str] = []

        self._astronomy_handler = None
        self._daily_briefing_handler = None

    async def async_register_all_services(self) -> None:
        """Register all services @zara"""

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
        """Unregister all services @zara"""
        for service_name in self._registered_services:
            if self.hass.services.has_service(DOMAIN, service_name):
                self.hass.services.async_remove(DOMAIN, service_name)

        self._registered_services.clear()

    def _build_service_definitions(self) -> List[ServiceDefinition]:
        """Build all service definitions @zara"""
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
                description="Run Grid-Search hyperparameter optimization (only on capable hardware)",
            ),
            ServiceDefinition(
                name=SERVICE_ANALYZE_FEATURE_IMPORTANCE,
                handler=self._handle_analyze_feature_importance,
                description="Analyze feature importance using Permutation Importance (developer only)",
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
                description="Test: Create retrospective forecast for today with current code",
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
                description="Refresh Multi-Weather cache (Open-Meteo + wttr.in)",
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
                description="Refresh cache for today + next 7 days",
            ),

            # Notification Services
            ServiceDefinition(
                name=SERVICE_SEND_DAILY_BRIEFING,
                handler=self._handle_send_daily_briefing,
                description="Send daily solar weather briefing notification",
            ),

            # Installation Services
            ServiceDefinition(
                name=SERVICE_INSTALL_EXTRA_FEATURES,
                handler=self._handle_install_extra_features,
                description="Install extra feature components (grid_price_monitor, sfml_stats)",
            ),

            # Data Management Services
            ServiceDefinition(
                name=SERVICE_BORG_ASSIMILATION_REVERSE,
                handler=self._handle_borg_assimilation_reverse,
                description="Reset learned data with automatic backup (ai_weights, hourly_history, all)",
            ),

            # Physics Calibration Services
            ServiceDefinition(
                name=SERVICE_RESCUE_CALIBRATION,
                handler=self._handle_rescue_calibration,
                description="Emergency rescue calibration for severely miscalibrated systems (DEVELOPER ONLY)",
            ),
            ServiceDefinition(
                name=SERVICE_REPAIR_CALIBRATION,
                handler=self._handle_repair_calibration,
                description="Analyze and repair extreme calibration factors (DEVELOPER ONLY)",
            ),
            ServiceDefinition(
                name=SERVICE_WINTER_BUCKET_CORRECTION,
                handler=self._handle_winter_bucket_correction,
                description="V13.1: Winter Bucket Correction - DEVELOPER ONLY - Kann KI nachhaltig schädigen!",
            ),
        ]

    # =========================================================================
    # AI Services
    # =========================================================================

    async def _handle_retrain_ai_model(self, call: ServiceCall) -> None:
        """Handle retrain_ai_model service @zara"""
        try:
            if self.coordinator.ai_predictor:
                _LOGGER.info("Service: retrain_ai_model - Starting AI training")
                result = await self.coordinator.ai_predictor.train_model()
                if result.success:
                    _LOGGER.info(
                        f"AI model training complete: R²={result.accuracy:.3f}, "
                        f"samples={result.samples_used}"
                    )
                else:
                    _LOGGER.error(f"AI model training failed: {result.error_message}")
            else:
                _LOGGER.warning("AI predictor not available")
        except Exception as e:
            _LOGGER.error(f"Error in retrain_ai_model: {e}")

    async def _handle_reset_ai_model(self, call: ServiceCall) -> None:
        """Handle reset_ai_model service @zara"""
        try:
            if self.coordinator.ai_predictor:
                _LOGGER.info("Service: reset_ai_model - Resetting AI model")
                # Reset by reinitializing (clears weights if file deleted)
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
        """Handle run_grid_search service - Run hyperparameter optimization @zara

        Grid-Search tests different model settings systematically to find the best
        combination. This service is only available on capable hardware (NUC, Mini-PC,
        Server). Raspberry Pi and Virtual Machines are excluded due to performance
        limitations.

        The service will:
        1. Check hardware capability
        2. Load training data
        3. Test multiple parameter combinations
        4. Save the best parameters for future training
        5. Optionally retrain the model with optimal parameters

        Parameters:
            retrain_after: bool - Retrain model with best params after search (default: True)
        """
        from ..ai import GridSearchOptimizer, TinyLSTM, detect_hardware

        _LOGGER.info("=" * 70)
        _LOGGER.info("SERVICE: run_grid_search")
        _LOGGER.info("Hyperparameter optimization for TinyLSTM")
        _LOGGER.info("=" * 70)

        try:
            # Step 1: Check hardware
            hw_info = detect_hardware()
            _LOGGER.info(f"Hardware: {hw_info.architecture}, {hw_info.cpu_count} CPUs")
            _LOGGER.info(f"Raspberry Pi: {hw_info.is_raspberry_pi}")
            _LOGGER.info(f"Virtual Machine: {hw_info.is_virtual_machine}")

            if not hw_info.grid_search_allowed:
                _LOGGER.warning(f"Grid-Search not available: {hw_info.reason}")

                # Send notification about hardware limitation
                try:
                    await self.hass.services.async_call(
                        "persistent_notification",
                        "create",
                        {
                            "title": "⚠️ Grid-Search nicht verfügbar",
                            "message": (
                                f"**Grund:** {hw_info.reason}\n\n"
                                f"Grid-Search ist nur auf leistungsstarker Hardware verfügbar:\n"
                                f"- ✅ NUC / Mini-PC\n"
                                f"- ✅ Desktop / Server\n"
                                f"- ❌ Raspberry Pi\n"
                                f"- ❌ Virtuelle Maschinen\n\n"
                                f"Die normale AI-Vorhersage funktioniert weiterhin."
                            ),
                            "notification_id": "solar_forecast_ml_grid_search",
                        },
                    )
                except Exception:
                    pass
                return

            # Step 2: Check AI predictor
            if not self.coordinator.ai_predictor:
                _LOGGER.error("AI predictor not available")
                return

            predictor = self.coordinator.ai_predictor

            # Step 3: Load training data
            _LOGGER.info("Loading training data...")
            X_sequences, y_targets, _ = await predictor._prepare_training_data()

            if len(X_sequences) < 50:
                _LOGGER.error(f"Not enough training data: {len(X_sequences)} samples (need 50+)")
                return

            _LOGGER.info(f"Loaded {len(X_sequences)} training samples")

            # Step 4: Initialize Grid-Search optimizer
            data_dir = predictor._ai_dir
            optimizer = GridSearchOptimizer(data_dir=data_dir)

            # Progress callback for logging
            async def progress_callback(current, total, params, accuracy):
                _LOGGER.info(
                    f"Grid-Search progress: {current}/{total} - "
                    f"hidden={params.get('hidden_size')}, "
                    f"batch={params.get('batch_size')}, "
                    f"R²={accuracy:.4f}"
                )

            # Step 5: Run Grid-Search
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

            _LOGGER.info("=" * 70)
            _LOGGER.info("GRID-SEARCH COMPLETE")
            _LOGGER.info(f"  Best R²: {result.best_accuracy:.4f}")
            _LOGGER.info(f"  Best params: {result.best_params}")
            _LOGGER.info(f"  Duration: {result.duration_seconds:.1f}s")
            _LOGGER.info("=" * 70)

            # Step 6: Retrain with best params if requested
            retrain_after = call.data.get("retrain_after", True)

            if retrain_after and result.best_params:
                _LOGGER.info("Retraining model with optimal parameters...")

                # Reinitialize LSTM with best params
                predictor.lstm = TinyLSTM(
                    input_size=feature_count,
                    hidden_size=result.best_params.get("hidden_size", 32),
                    sequence_length=24,
                    num_outputs=num_outputs,
                    learning_rate=result.best_params.get("learning_rate", 0.005),
                )

                # Full training with best params
                train_result = await predictor.train_model()

                if train_result.success:
                    _LOGGER.info(
                        f"Retrained model: R²={train_result.accuracy:.4f} "
                        f"(samples={train_result.samples_used})"
                    )
                else:
                    _LOGGER.error(f"Retraining failed: {train_result.error_message}")

            # Step 7: Send notification
            try:
                results_table = "\n".join(
                    f"- hidden={r['params'].get('hidden_size')}, "
                    f"batch={r['params'].get('batch_size')}: "
                    f"R²={r.get('accuracy', 0):.4f}"
                    for r in result.all_results
                    if 'accuracy' in r
                )

                await self.hass.services.async_call(
                    "persistent_notification",
                    "create",
                    {
                        "title": "✅ Grid-Search abgeschlossen",
                        "message": (
                            f"**Beste Parameter gefunden:**\n"
                            f"- Hidden Size: {result.best_params.get('hidden_size')}\n"
                            f"- Batch Size: {result.best_params.get('batch_size')}\n"
                            f"- Learning Rate: {result.best_params.get('learning_rate')}\n\n"
                            f"**Beste Genauigkeit:** R²={result.best_accuracy:.4f}\n"
                            f"**Dauer:** {result.duration_seconds:.1f}s\n\n"
                            f"**Alle Ergebnisse:**\n{results_table}"
                        ),
                        "notification_id": "solar_forecast_ml_grid_search",
                    },
                )
            except Exception:
                pass

        except Exception as e:
            _LOGGER.error(f"Error in run_grid_search: {e}", exc_info=True)

    async def _handle_analyze_feature_importance(self, call: ServiceCall) -> None:
        """Handle analyze_feature_importance service - Permutation Importance analysis @zara

        This service analyzes which features help or hurt the AI model predictions.
        Results are saved to ai/feature_importance.json for developer analysis.

        Parameters:
            num_permutations: int - Number of permutations per feature (default: 5)
        """
        _LOGGER.info("=" * 70)
        _LOGGER.info("SERVICE: analyze_feature_importance")
        _LOGGER.info("Permutation Importance analysis for TinyLSTM features")
        _LOGGER.info("=" * 70)

        try:
            # Check AI predictor
            if not self.coordinator.ai_predictor:
                _LOGGER.error("AI predictor not available")
                return

            predictor = self.coordinator.ai_predictor

            # Get num_permutations from service call
            num_permutations = call.data.get("num_permutations", 5)
            _LOGGER.info(f"Using {num_permutations} permutations per feature")

            # Progress callback for logging
            async def progress_callback(current, total, feature_name):
                _LOGGER.debug(
                    f"Feature Importance progress: {current}/{total} - {feature_name}"
                )

            # Run analysis
            result = await predictor.analyze_feature_importance(
                num_permutations=num_permutations,
                progress_callback=progress_callback,
            )

            if result is None or not result.success:
                error_msg = result.error_message if result else "Unknown error"
                _LOGGER.error(f"Feature Importance analysis failed: {error_msg}")

                # Send notification about failure
                try:
                    await self.hass.services.async_call(
                        "persistent_notification",
                        "create",
                        {
                            "title": "❌ Feature Importance fehlgeschlagen",
                            "message": (
                                f"**Fehler:** {error_msg}\n\n"
                                f"Mögliche Ursachen:\n"
                                f"- Nicht genug Trainingsdaten (min. 10 Samples)\n"
                                f"- AI-Modell nicht trainiert\n"
                                f"- Systemfehler"
                            ),
                            "notification_id": "solar_forecast_ml_feature_importance",
                        },
                    )
                except Exception:
                    pass
                return

            # Log results
            _LOGGER.info("=" * 70)
            _LOGGER.info("FEATURE IMPORTANCE COMPLETE")
            _LOGGER.info(f"  Baseline RMSE: {result.baseline_rmse:.4f} kWh")
            _LOGGER.info(f"  Helpful features: {len(result.helpful_features)}")
            _LOGGER.info(f"  Harmful features: {len(result.harmful_features)}")
            _LOGGER.info(f"  Neutral features: {len(result.neutral_features)}")
            _LOGGER.info(f"  Duration: {result.analysis_time_seconds:.1f}s")
            _LOGGER.info("=" * 70)

            # Build report for notification
            helpful_list = "\n".join(
                f"  - {f}: +{result.feature_importance[f]:.4f}"
                for f in result.helpful_features[:5]
            ) or "  (keine)"

            harmful_list = "\n".join(
                f"  - {f}: {result.feature_importance[f]:.4f}"
                for f in result.harmful_features[:5]
            ) or "  (keine)"

            # Send notification
            try:
                await self.hass.services.async_call(
                    "persistent_notification",
                    "create",
                    {
                        "title": "📊 Feature Importance Analyse abgeschlossen",
                        "message": (
                            f"**Baseline RMSE:** {result.baseline_rmse:.4f} kWh\n"
                            f"**Analysierte Samples:** {result.num_samples}\n"
                            f"**Dauer:** {result.analysis_time_seconds:.1f}s\n\n"
                            f"**🟢 Hilfreiche Features ({len(result.helpful_features)}):**\n{helpful_list}\n\n"
                            f"**🔴 Schädliche Features ({len(result.harmful_features)}):**\n{harmful_list}\n\n"
                            f"**Details:** ai/feature_importance.json"
                        ),
                        "notification_id": "solar_forecast_ml_feature_importance",
                    },
                )
            except Exception:
                pass

        except Exception as e:
            _LOGGER.error(f"Error in analyze_feature_importance: {e}", exc_info=True)

    # =========================================================================
    # Emergency Services
    # =========================================================================

    async def _handle_run_all_day_end_tasks(self, call: ServiceCall) -> None:
        """Handle run_all_day_end_tasks service @zara"""
        try:
            if hasattr(self.coordinator, "scheduled_tasks"):
                await self.coordinator.scheduled_tasks.end_of_day_workflow(None)
        except Exception as e:
            _LOGGER.error(f"Error in run_all_day_end_tasks: {e}")

    # =========================================================================
    # Testing Services
    # =========================================================================

    async def _handle_test_morning_routine(self, call: ServiceCall) -> None:
        """Handle test_morning_routine service - 100% IDENTICAL to scheduled routine @zara"""
        try:
            if hasattr(self.coordinator, "scheduled_tasks"):
                await self.coordinator.scheduled_tasks.morning_routine_complete(None)
        except Exception as e:
            _LOGGER.error(f"Error in test_morning_routine: {e}")

    async def _handle_test_retrospective_forecast(self, call: ServiceCall) -> None:
        """Handle test_retrospective_forecast service - Create retrospective forecast for today @zara

        This service simulates running the morning forecast at "sunrise - 1 hour"
        but uses the CURRENT code (with any recent changes like new correction factors).

        Purpose:
        - Test code changes that were made AFTER the actual morning forecast
        - See what the morning forecast WOULD have been with the new code
        - Results are written to stats/retrospective_forecast.json (not affecting actual predictions)

        Example use case:
        - User changes a correction factor at 14:00
        - Runs this service to see how the 07:00 forecast would look with new code
        - Compares retrospective_forecast.json with actual hourly_predictions.json
        """
        from datetime import datetime, time
        from pathlib import Path

        _LOGGER.info("=" * 70)
        _LOGGER.info("SERVICE: test_retrospective_forecast")
        _LOGGER.info("Creating retrospective forecast for today with CURRENT code")
        _LOGGER.info("=" * 70)

        try:
            # Step 1: Get sunrise time for today from astronomy cache
            sunrise = await self._get_sunrise_for_today()
            if not sunrise:
                _LOGGER.error("Could not determine sunrise time - using fallback 08:00")
                now = dt_util.now()
                sunrise = datetime.combine(now.date(), time(8, 0))
                if now.tzinfo:
                    sunrise = sunrise.replace(tzinfo=now.tzinfo)

            # Simulation time: 1 hour before sunrise (typical morning forecast time)
            simulation_time = sunrise - timedelta(hours=1)
            today_str = dt_util.now().date().isoformat()

            _LOGGER.info(f"Sunrise today: {sunrise.strftime('%H:%M')}")
            _LOGGER.info(f"Simulation time (sunrise - 1h): {simulation_time.strftime('%H:%M')}")
            _LOGGER.info(f"Target date: {today_str}")

            # Step 2: Get weather forecast (current corrected forecast)
            weather_service = self.coordinator.weather_service
            if not weather_service:
                _LOGGER.error("Weather service not available")
                return

            hourly_weather_forecast = await weather_service.get_corrected_hourly_forecast()
            if not hourly_weather_forecast:
                _LOGGER.error("No corrected weather forecast available")
                return

            current_weather = await weather_service.get_current_weather()

            _LOGGER.info(f"Loaded {len(hourly_weather_forecast)} hours of weather data")

            # Step 3: Get astronomy data for today
            astronomy_data = await self._get_astronomy_data_for_date(today_str)
            if not astronomy_data:
                _LOGGER.error("No astronomy data available for today")
                return

            _LOGGER.info(f"Loaded astronomy data (daylight: {astronomy_data.get('daylight_hours', 'N/A')}h)")

            # Step 4: Collect sensor configuration
            from ..const import (
                CONF_HUMIDITY_SENSOR,
                CONF_LUX_SENSOR,
                CONF_PRESSURE_SENSOR,
                CONF_RAIN_SENSOR,
                CONF_SOLAR_RADIATION_SENSOR,
                CONF_TEMP_SENSOR,
                CONF_WIND_SENSOR,
            )

            sensor_config = {
                "temperature": self.coordinator.entry.data.get(CONF_TEMP_SENSOR) is not None,
                "humidity": self.coordinator.entry.data.get(CONF_HUMIDITY_SENSOR) is not None,
                "lux": self.coordinator.entry.data.get(CONF_LUX_SENSOR) is not None,
                "rain": self.coordinator.entry.data.get(CONF_RAIN_SENSOR) is not None,
                "wind_speed": self.coordinator.entry.data.get(CONF_WIND_SENSOR) is not None,
                "pressure": self.coordinator.entry.data.get(CONF_PRESSURE_SENSOR) is not None,
                "solar_radiation": self.coordinator.entry.data.get(CONF_SOLAR_RADIATION_SENSOR) is not None,
            }

            # Step 5: Get current external sensor data (for reference)
            external_sensors = self.coordinator.sensor_collector.collect_all_sensor_data_dict()

            # Step 6: Run forecast orchestrator with CURRENT code
            _LOGGER.info("Running forecast orchestrator with CURRENT code...")

            forecast = await self.coordinator.forecast_orchestrator.orchestrate_forecast(
                current_weather=current_weather,
                hourly_forecast=hourly_weather_forecast,
                external_sensors=external_sensors,
                ml_prediction_today=None,
                ml_prediction_tomorrow=None,
                correction_factor=self.coordinator.learned_correction_factor,
            )

            if not forecast or not forecast.get("hourly"):
                _LOGGER.error("Forecast generation failed - no hourly data")
                return

            # Step 7: Extract today's hourly predictions
            all_hourly = forecast.get("hourly", [])
            today_hourly = [h for h in all_hourly if h.get("date") == today_str]

            _LOGGER.info(f"Generated {len(today_hourly)} hourly predictions for today")

            # Step 8: Build retrospective forecast result
            result = {
                "version": "1.0",
                "generated_at": dt_util.now().isoformat(),
                "simulation_context": {
                    "simulated_forecast_time": simulation_time.isoformat(),
                    "sunrise_today": sunrise.isoformat(),
                    "target_date": today_str,
                    "code_version": "CURRENT",
                    "purpose": "Retrospective forecast to test code changes made after morning forecast",
                },
                "forecast_summary": {
                    "today_kwh": forecast.get("today"),
                    "today_kwh_raw": forecast.get("today_raw"),
                    "safeguard_applied": forecast.get("safeguard_applied", False),
                    "tomorrow_kwh": forecast.get("tomorrow"),
                    "day_after_tomorrow_kwh": forecast.get("day_after_tomorrow"),
                    "method": forecast.get("method"),
                    "confidence": forecast.get("confidence"),
                    "model_accuracy": forecast.get("model_accuracy"),
                    "best_hour": forecast.get("best_hour"),
                    "best_hour_kwh": forecast.get("best_hour_kwh"),
                },
                "hourly_predictions": [],
                "weather_data_used": {
                    "source": "weather_forecast_corrected.json",
                    "hours_available": len(hourly_weather_forecast),
                },
                "astronomy_data_used": {
                    "sunrise": astronomy_data.get("sunrise"),
                    "sunset": astronomy_data.get("sunset"),
                    "solar_noon": astronomy_data.get("solar_noon"),
                    "daylight_hours": astronomy_data.get("daylight_hours"),
                },
                "correction_factor_used": self.coordinator.learned_correction_factor,
            }

            # Build detailed hourly predictions
            for hour_data in today_hourly:
                hour = hour_data.get("hour")
                hour_datetime = hour_data.get("datetime")

                # Get weather for this hour (filter by today's date to get correct day's data)
                weather = self._find_weather_for_hour(hourly_weather_forecast, hour, today_str)

                # Get astronomy for this hour
                hourly_astro = astronomy_data.get("hourly", {}).get(str(hour), {})

                prediction_entry = {
                    "hour": hour,
                    "datetime": hour_datetime,
                    "prediction_kwh": hour_data.get("production_kwh", 0.0),
                    "panel_group_predictions": hour_data.get("panel_group_predictions"),
                    "weather": {
                        "temperature_c": weather.get("temperature") if weather else None,
                        "cloud_cover_percent": weather.get("cloud_cover") if weather else None,
                        "humidity_percent": weather.get("humidity") if weather else None,
                        "wind_speed_ms": weather.get("wind_speed") if weather else None,
                        "precipitation_mm": weather.get("precipitation", 0) if weather else 0,
                        "direct_radiation": weather.get("direct_radiation") if weather else None,
                        "diffuse_radiation": weather.get("diffuse_radiation") if weather else None,
                        # V12.8.5: FOG detection parameters
                        "visibility_m": weather.get("visibility_m") if weather else None,
                        "fog_detected": weather.get("fog_detected", False) if weather else False,
                        "fog_type": weather.get("fog_type") if weather else None,
                    },
                    "astronomy": {
                        "sun_elevation_deg": hourly_astro.get("elevation_deg"),
                        "sun_azimuth_deg": hourly_astro.get("azimuth_deg"),
                        "theoretical_max_kwh": hourly_astro.get("theoretical_max_pv_kwh"),
                        "clear_sky_radiation_wm2": hourly_astro.get("clear_sky_solar_radiation_wm2"),
                    },
                }

                result["hourly_predictions"].append(prediction_entry)

            # V12.8.5: Calculate today_kwh from hourly predictions (full day sum)
            # The orchestrator's forecast.get("today") only counts hours >= current hour,
            # but for retrospective we want the FULL day as if calculated at sunrise
            retrospective_today_kwh = sum(
                h.get("prediction_kwh", 0.0) or 0.0
                for h in result["hourly_predictions"]
            )
            result["forecast_summary"]["today_kwh"] = round(retrospective_today_kwh, 2)
            result["forecast_summary"]["today_kwh_note"] = "Recalculated from hourly sum (full day)"

            # Also find best hour from full day
            best_hour_entry = max(
                result["hourly_predictions"],
                key=lambda h: h.get("prediction_kwh", 0.0) or 0.0,
                default=None
            )
            if best_hour_entry:
                result["forecast_summary"]["best_hour"] = best_hour_entry.get("hour")
                result["forecast_summary"]["best_hour_kwh"] = best_hour_entry.get("prediction_kwh", 0.0)

            # Step 9: Write result to retrospective_forecast.json
            output_file = self.coordinator.data_manager.data_dir / "stats" / "retrospective_forecast.json"

            def _write_result():
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)

            await self.hass.async_add_executor_job(_write_result)

            _LOGGER.info("=" * 70)
            _LOGGER.info("RETROSPECTIVE FORECAST COMPLETE")
            _LOGGER.info(f"  Total today: {retrospective_today_kwh:.2f} kWh (full day sum)")
            _LOGGER.info(f"  Method: {forecast.get('method')}")
            _LOGGER.info(f"  Best hour: {result['forecast_summary']['best_hour']} ({result['forecast_summary']['best_hour_kwh']:.3f} kWh)")
            _LOGGER.info(f"  Hours predicted: {len(today_hourly)}")
            _LOGGER.info(f"  Output file: {output_file}")
            _LOGGER.info("=" * 70)

            # Send notification
            try:
                await self.hass.services.async_call(
                    "persistent_notification",
                    "create",
                    {
                        "title": "Retrospective Forecast erstellt",
                        "message": (
                            f"**Simulierter Zeitpunkt:** {simulation_time.strftime('%H:%M')} (1h vor Sonnenaufgang)\n\n"
                            f"**Tagesprognose:** {retrospective_today_kwh:.2f} kWh (Summe aller Stunden)\n"
                            f"**Methode:** {forecast.get('method')}\n"
                            f"**Beste Stunde:** {result['forecast_summary']['best_hour']} ({result['forecast_summary']['best_hour_kwh']:.3f} kWh)\n"
                            f"**Stunden:** {len(today_hourly)}\n\n"
                            f"Ergebnis gespeichert in:\n`{output_file}`\n\n"
                            f"Vergleichen Sie mit `hourly_predictions.json` um Unterschiede zu sehen."
                        ),
                        "notification_id": "solar_forecast_ml_retrospective",
                    },
                )
            except Exception:
                pass

        except Exception as e:
            _LOGGER.error(f"Error in test_retrospective_forecast: {e}", exc_info=True)

    async def _get_sunrise_for_today(self) -> "datetime | None":
        """Get sunrise time for today from astronomy cache @zara"""
        try:
            today = dt_util.now().date()
            astronomy_cache_file = self.coordinator.data_manager.data_dir / "stats" / "astronomy_cache.json"

            if not astronomy_cache_file.exists():
                _LOGGER.warning("Astronomy cache not found")
                return None

            def _read_sync():
                with open(astronomy_cache_file, "r") as f:
                    return json.load(f)

            cache = await self.hass.async_add_executor_job(_read_sync)
            day_data = cache.get("days", {}).get(today.isoformat())

            if not day_data:
                _LOGGER.warning(f"No astronomy data for {today}")
                return None

            sunrise_str = day_data.get("sunrise_local")
            if not sunrise_str:
                return None

            from datetime import datetime
            sunrise = datetime.fromisoformat(sunrise_str)

            # Ensure timezone-aware
            if sunrise.tzinfo is None:
                local_tz = dt_util.now().tzinfo
                if local_tz:
                    sunrise = sunrise.replace(tzinfo=local_tz)

            return sunrise

        except Exception as e:
            _LOGGER.error(f"Error getting sunrise: {e}")
            return None

    async def _get_astronomy_data_for_date(self, date_str: str) -> "dict | None":
        """Get astronomy data for a specific date from cache @zara"""
        try:
            astronomy_cache_file = self.coordinator.data_manager.data_dir / "stats" / "astronomy_cache.json"

            if not astronomy_cache_file.exists():
                return None

            def _read_sync():
                with open(astronomy_cache_file, "r") as f:
                    return json.load(f)

            cache = await self.hass.async_add_executor_job(_read_sync)
            day_data = cache.get("days", {}).get(date_str)

            if not day_data:
                return None

            return {
                "sunrise": day_data.get("sunrise_local"),
                "sunset": day_data.get("sunset_local"),
                "solar_noon": day_data.get("solar_noon_local"),
                "daylight_hours": day_data.get("daylight_hours"),
                "hourly": day_data.get("hourly", {}),
            }

        except Exception as e:
            _LOGGER.error(f"Error getting astronomy data: {e}")
            return None

    def _find_weather_for_hour(self, weather_forecast: "list", hour: int, date: str = None) -> "dict | None":
        """Find weather data for a specific hour and date @zara

        Args:
            weather_forecast: List of weather data entries
            hour: The hour to find (0-23)
            date: Optional date string (YYYY-MM-DD). If provided, filters by both date and hour.
                  If not provided, returns first matching hour (legacy behavior).
        """
        for w in weather_forecast:
            if w.get("local_hour") == hour:
                # If date filter provided, also check the date
                if date is not None:
                    if w.get("date") == date:
                        return w
                else:
                    return w
        return None

    # =========================================================================
    # Weather Services
    # =========================================================================

    async def _handle_run_weather_correction(self, call: ServiceCall) -> None:
        """Handle run_weather_correction service - Manually trigger corrected forecast generation @zara"""
        try:
            if not hasattr(self.coordinator, 'weather_pipeline_manager'):
                return

            pipeline = self.coordinator.weather_pipeline_manager
            success = await pipeline.create_corrected_forecast()

            if not success:
                _LOGGER.warning("Corrected forecast generation failed")

        except Exception as e:
            _LOGGER.error(f"Error in run_weather_correction: {e}")

    async def _handle_refresh_open_meteo_cache(self, call: ServiceCall) -> None:
        """Handle refresh_open_meteo_cache service - Refresh weather with 5-source blending @zara

        V12.3: Uses WeatherExpertBlender with 5 sources and creates corrected forecast.
        """
        try:
            if not hasattr(self.coordinator, 'weather_pipeline_manager'):
                _LOGGER.warning("Weather pipeline manager not available")
                return

            pipeline = self.coordinator.weather_pipeline_manager

            # V12.3: Use unified weather refresh
            success = await pipeline.update_weather_cache()

            if success:
                _LOGGER.info("Weather cache refreshed via 5-source ExpertBlender")
            else:
                _LOGGER.warning("Weather cache refresh failed")

        except Exception as e:
            _LOGGER.error(f"Error in refresh_open_meteo_cache: {e}", exc_info=True)

    async def _handle_refresh_multi_weather(self, call: ServiceCall) -> None:
        """Handle refresh_multi_weather service - Refresh 5-source weather blending @zara

        V12.3: Uses WeatherExpertBlender (Open-Meteo, wttr.in, ECMWF, Bright Sky, Pirate Weather).
        This is now an alias for refresh_open_meteo_cache.

        V12.8: Added force_update parameter to bypass memory cache and force API fetch.
        """
        try:
            if not hasattr(self.coordinator, 'weather_pipeline_manager'):
                _LOGGER.warning("Weather pipeline manager not available")
                return

            pipeline = self.coordinator.weather_pipeline_manager

            # V12.8: Read force_update from service call data
            # Note: services.yaml uses force_wttr_refresh for backwards compatibility
            force_update = call.data.get("force_update", False) or call.data.get("force_wttr_refresh", False)

            _LOGGER.info(f"Service: refresh_multi_weather (5-source ExpertBlender, force={force_update})")

            # V12.3: Use unified weather refresh, V12.8: Pass force parameter
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
        """Handle build_astronomy_cache service @zara"""
        if self._astronomy_handler:
            await self._astronomy_handler.handle_build_astronomy_cache(call)

    async def _handle_refresh_cache_today(self, call: ServiceCall) -> None:
        """Handle refresh_cache_today service @zara"""
        if self._astronomy_handler:
            await self._astronomy_handler.handle_refresh_cache_today(call)

    # =========================================================================
    # Notification Services
    # =========================================================================

    async def _handle_send_daily_briefing(self, call: ServiceCall) -> None:
        """Handle send_daily_briefing service @zara"""
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
    # Installation Services
    # =========================================================================

    async def _handle_install_extra_features(self, call: ServiceCall) -> None:
        """Handle install_extra_features service - Install extra components @zara

        This service copies the extra feature components from the extra_features
        folder to the custom_components folder. After installation, Home Assistant
        needs to be restarted for the new integrations to be available.

        Available features:
        - grid_price_monitor: Dynamic grid price monitoring
        - sfml_stats: Solar Forecast ML statistics dashboard
        """
        from ..services.service_extra_features import ExtraFeaturesInstaller

        _LOGGER.info("=" * 60)
        _LOGGER.info("SERVICE: install_extra_features")
        _LOGGER.info("=" * 60)

        try:
            installer = ExtraFeaturesInstaller(self.hass)

            # Show current status
            status = installer.get_installation_status()
            _LOGGER.info("Current installation status:")
            for feature, state in status.items():
                _LOGGER.info(f"  - {feature}: {state}")

            # Install all features
            installed, failed = await installer.install_all()

            if installed:
                _LOGGER.info(f"✓ Successfully installed: {', '.join(installed)}")

            if failed:
                _LOGGER.error(f"✗ Failed to install: {', '.join(failed)}")

            if installed and not failed:
                _LOGGER.info("=" * 60)
                _LOGGER.info("IMPORTANT: Restart Home Assistant to activate the new integrations!")
                _LOGGER.info("After restart, configure them via Settings → Devices & Services → Add Integration")
                _LOGGER.info("=" * 60)

                # Send notification
                try:
                    installed_list = "\n".join(f"- {feat}" for feat in installed)
                    await self.hass.services.async_call(
                        "persistent_notification",
                        "create",
                        {
                            "title": "✅ Extra Features erfolgreich installiert",
                            "message": (
                                f"Die folgenden Integrationen wurden erfolgreich kopiert:\n"
                                f"{installed_list}\n\n"
                                f"**⚠️ Bitte Home Assistant neu starten!**\n\n"
                                f"Nach dem Neustart unter Einstellungen → Geräte & Dienste → Integration hinzufügen konfigurieren.\n\n"
                                f"---\n"
                                f"**🔴 WICHTIG für ARM64-Systeme (Raspberry Pi, etc.):**\n"
                                f"Auf ARM64-Architektur darf **NUR sfml_stats_lite** verwendet werden!\n"
                                f"Die vollständige sfml_stats-Integration ist nicht mit ARM64 kompatibel."
                            ),
                            "notification_id": "solar_forecast_ml_extra_features",
                        },
                    )
                except Exception as notify_err:
                    _LOGGER.debug(f"Could not send notification: {notify_err}")

        except Exception as err:
            _LOGGER.error(f"Error in install_extra_features service: {err}", exc_info=True)

    # =========================================================================
    # Data Management Services
    # =========================================================================

    async def _handle_borg_assimilation_reverse(self, call: ServiceCall) -> None:
        """Handle borg_assimilation_reverse service - Reset learned data with backup @zara

        "We are the Borg. Your data distinctiveness will be... restored."

        This service allows users to selectively reset learned data:
        - ai_weights: Reset AI learned weights
        - hourly_history: Clean hourly_predictions.json
        - all: Reset everything above

        ALWAYS creates a backup before any reset operation.

        Parameters:
            target: str - What to reset (panel_groups, ml_weights, hourly_history, all)
            before_date: str (optional) - Only reset data before this date (YYYY-MM-DD)
            keep_backup: bool - Keep backup after reset (default: True)
        """
        from datetime import datetime
        from pathlib import Path
        import shutil

        target = call.data.get("target", "all")
        before_date_str = call.data.get("before_date")
        keep_backup = call.data.get("keep_backup", True)

        _LOGGER.info("=" * 80)
        _LOGGER.info("🖖 SERVICE: borg_assimilation_reverse")
        _LOGGER.info("   'We are the Borg. Your data distinctiveness will be... restored.'")
        _LOGGER.info("=" * 80)
        _LOGGER.info(f"Target: {target}")
        _LOGGER.info(f"Before date: {before_date_str or 'ALL DATA'}")
        _LOGGER.info(f"Keep backup: {keep_backup}")

        try:
            data_dir = self.coordinator.data_manager.data_dir
            ai_dir = data_dir / "ai"
            stats_dir = data_dir / "stats"
            backups_dir = data_dir / "backups" / "borg_assimilation"

            def _ensure_backup_dir():
                backups_dir.mkdir(parents=True, exist_ok=True)
            await self.hass.async_add_executor_job(_ensure_backup_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = backups_dir / f"backup_{timestamp}"

            target_files = {
                "ai_weights": [
                    ai_dir / "learned_weights.json",
                    ai_dir / "seasonal.json",
                    ai_dir / "dni_tracker.json",
                ],
                "hourly_history": [
                    stats_dir / "hourly_predictions.json",
                    stats_dir / "hourly_weather_actual.json",
                ],
            }

            files_to_reset = []
            if target == "all":
                for file_list in target_files.values():
                    files_to_reset.extend(file_list)
            elif target in target_files:
                files_to_reset = target_files[target]
            else:
                _LOGGER.error(f"Unknown target: {target}")
                _LOGGER.error("Valid targets: ai_weights, hourly_history, all")
                return

            # Filter to only existing files
            def _filter_existing():
                return [f for f in files_to_reset if f.exists()]
            existing_files = await self.hass.async_add_executor_job(_filter_existing)

            if not existing_files:
                _LOGGER.warning("No files found to reset!")
                return

            _LOGGER.info(f"Files to reset: {len(existing_files)}")

            # ================================================================
            # STEP 1: Create backup (ALWAYS!)
            # ================================================================
            _LOGGER.info("STEP 1: Creating backup...")

            def _create_backup():
                backup_subdir.mkdir(parents=True, exist_ok=True)
                backed_up = []
                for file_path in existing_files:
                    if file_path.exists():
                        # Preserve directory structure in backup
                        relative_path = file_path.relative_to(data_dir)
                        backup_path = backup_subdir / relative_path
                        backup_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, backup_path)
                        backed_up.append(str(relative_path))
                return backed_up

            backed_up_files = await self.hass.async_add_executor_job(_create_backup)
            _LOGGER.info(f"  ✓ Backed up {len(backed_up_files)} files to {backup_subdir}")
            for f in backed_up_files:
                _LOGGER.info(f"    - {f}")

            # ================================================================
            # STEP 2: Reset files
            # ================================================================
            _LOGGER.info("STEP 2: Resetting data...")

            reset_count = 0
            partial_reset_count = 0

            for file_path in existing_files:
                try:
                    if before_date_str:
                        # Partial reset: Filter data by date
                        reset_result = await self._reset_file_before_date(
                            file_path, before_date_str
                        )
                        if reset_result:
                            partial_reset_count += 1
                            _LOGGER.info(f"  ✓ Partial reset: {file_path.name}")
                    else:
                        # Full reset: Create empty/default structure
                        await self._reset_file_complete(file_path)
                        reset_count += 1
                        _LOGGER.info(f"  ✓ Full reset: {file_path.name}")
                except Exception as e:
                    _LOGGER.error(f"  ✗ Failed to reset {file_path.name}: {e}")

            # ================================================================
            # STEP 3: Summary & Notification
            # ================================================================
            _LOGGER.info("=" * 80)
            _LOGGER.info("🖖 BORG ASSIMILATION REVERSE COMPLETE!")
            _LOGGER.info(f"  - Target: {target}")
            _LOGGER.info(f"  - Full resets: {reset_count}")
            _LOGGER.info(f"  - Partial resets: {partial_reset_count}")
            _LOGGER.info(f"  - Backup location: {backup_subdir}")
            _LOGGER.info("=" * 80)

            # Send notification
            try:
                await self.hass.services.async_call(
                    "persistent_notification",
                    "create",
                    {
                        "title": "🖖 Borg Assimilation Reverse - Complete",
                        "message": (
                            f"**Target:** {target}\n"
                            f"**Resets:** {reset_count + partial_reset_count} files\n"
                            f"**Backup:** `{backup_subdir}`\n\n"
                            f"'Resistance is futile... but your backup is secure.'"
                        ),
                        "notification_id": "solar_forecast_ml_borg_reverse",
                    },
                )
            except Exception:
                pass

        except Exception as err:
            _LOGGER.error(f"Error in borg_assimilation_reverse: {err}", exc_info=True)

    async def _reset_file_complete(self, file_path: "Path") -> None:
        """Reset a file to its default empty state @zara"""
        import json

        # Default structures for each file type
        default_structures = {
            "learned_panel_group_efficiency.json": {
                "version": "1.0",
                "last_updated": None,
                "groups": {},
                "metadata": {
                    "reset_by": "borg_assimilation_reverse",
                    "reset_at": self._get_timestamp(),
                },
            },
            "learned_weights.json": {
                "version": "1.0",
                "weights": {},
                "metadata": {
                    "reset_by": "borg_assimilation_reverse",
                    "reset_at": self._get_timestamp(),
                },
            },
            "hourly_profile.json": {
                "version": "1.0",
                "profiles": {},
                "metadata": {
                    "reset_by": "borg_assimilation_reverse",
                    "reset_at": self._get_timestamp(),
                },
            },
            "model_state.json": {
                "version": "1.0",
                "state": {},
                "metadata": {
                    "reset_by": "borg_assimilation_reverse",
                    "reset_at": self._get_timestamp(),
                },
            },
            "hourly_predictions.json": {
                "version": "1.0",
                "predictions": {},
                "metadata": {
                    "reset_by": "borg_assimilation_reverse",
                    "reset_at": self._get_timestamp(),
                },
            },
            "hourly_weather_actual.json": {
                "version": "1.0",
                "hourly_data": {},
                "metadata": {
                    "reset_by": "borg_assimilation_reverse",
                    "reset_at": self._get_timestamp(),
                },
            },
        }

        filename = file_path.name
        default_content = default_structures.get(filename, {
            "version": "1.0",
            "data": {},
            "metadata": {
                "reset_by": "borg_assimilation_reverse",
                "reset_at": self._get_timestamp(),
            },
        })

        def _write_default():
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(default_content, f, indent=2, ensure_ascii=False)

        await self.hass.async_add_executor_job(_write_default)

    async def _reset_file_before_date(self, file_path: "Path", before_date_str: str) -> bool:
        """Reset data in a file before a specific date @zara

        Returns True if any data was removed.
        """
        import json
        from datetime import datetime

        try:
            before_date = datetime.strptime(before_date_str, "%Y-%m-%d").date()
        except ValueError:
            _LOGGER.error(f"Invalid date format: {before_date_str}. Use YYYY-MM-DD")
            return False

        def _load_file():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        def _save_file(data):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        try:
            data = await self.hass.async_add_executor_job(_load_file)
        except Exception:
            return False

        modified = False
        filename = file_path.name

        # Handle different file structures
        if filename == "learned_panel_group_efficiency.json":
            # Filter groups by date
            if "groups" in data:
                for group_name, group_data in data.get("groups", {}).items():
                    if "history" in group_data:
                        original_len = len(group_data["history"])
                        group_data["history"] = [
                            h for h in group_data["history"]
                            if self._date_str_after(h.get("date"), before_date)
                        ]
                        if len(group_data["history"]) < original_len:
                            modified = True

        elif filename in ("hourly_predictions.json", "hourly_weather_actual.json"):
            # Filter hourly_data by date keys
            if "hourly_data" in data:
                original_keys = set(data["hourly_data"].keys())
                data["hourly_data"] = {
                    k: v for k, v in data["hourly_data"].items()
                    if self._date_str_after(k, before_date)
                }
                if set(data["hourly_data"].keys()) != original_keys:
                    modified = True

        elif filename == "learned_weights.json":
            # Filter weights by date
            if "weights" in data:
                for weight_type, weight_data in data.get("weights", {}).items():
                    if isinstance(weight_data, dict) and "history" in weight_data:
                        original_len = len(weight_data["history"])
                        weight_data["history"] = [
                            h for h in weight_data["history"]
                            if self._date_str_after(h.get("date"), before_date)
                        ]
                        if len(weight_data["history"]) < original_len:
                            modified = True

        if modified:
            # Update metadata
            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"]["partial_reset_by"] = "borg_assimilation_reverse"
            data["metadata"]["partial_reset_at"] = self._get_timestamp()
            data["metadata"]["reset_before_date"] = before_date_str

            await self.hass.async_add_executor_job(_save_file, data)

        return modified

    def _date_str_after(self, date_str: str, before_date) -> bool:
        """Check if a date string is on or after the before_date @zara"""
        from datetime import datetime

        if not date_str:
            return True  # Keep entries without dates

        try:
            entry_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            return entry_date >= before_date
        except (ValueError, TypeError):
            return True  # Keep entries with invalid dates

    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string @zara"""
        return dt_util.now().isoformat()

    # =========================================================================
    # Physics Calibration Services
    # =========================================================================

    async def _handle_rescue_calibration(self, call: ServiceCall) -> None:
        """Handle rescue_calibration service - Emergency calibration for miscalibrated systems @zara

        ╔══════════════════════════════════════════════════════════════════════════╗
        ║ 🚨 NUR AUF ANWEISUNG DES ENTWICKLERS AUSFÜHREN!                         ║
        ║ DO NOT RUN WITHOUT EXPLICIT DEVELOPER INSTRUCTION!                       ║
        ╚══════════════════════════════════════════════════════════════════════════╝

        This service is designed for systems that are severely miscalibrated
        (actual production > 10x physics prediction). Normal calibration skips
        such "outliers", creating a chicken-egg problem where the system can
        never correct itself.

        This service:
        1. Analyzes history for 2+ days with ratio > 10
        2. Performs one-time calibration with higher MAX_CORRECTION_FACTOR (50)
        3. Sets correction factors to enable normal learning afterwards
        """
        _LOGGER.warning("=" * 80)
        _LOGGER.warning("🚨 SERVICE: rescue_calibration")
        _LOGGER.warning("⚠️  NUR AUF ANWEISUNG DES ENTWICKLERS AUSFÜHREN!")
        _LOGGER.warning("⚠️  DO NOT RUN WITHOUT EXPLICIT DEVELOPER INSTRUCTION!")
        _LOGGER.warning("=" * 80)

        try:
            # Check if physics_calibrator is available
            if not hasattr(self.coordinator, 'physics_calibrator') or not self.coordinator.physics_calibrator:
                _LOGGER.error("Physics calibrator not available")
                await self._send_rescue_notification(
                    success=False,
                    message="Physics Calibrator ist nicht verfügbar.",
                )
                return

            calibrator = self.coordinator.physics_calibrator

            # Run rescue calibration
            result = await calibrator.rescue_calibration()

            # Send notification
            await self._send_rescue_notification(
                success=result.success,
                message=result.message,
                groups_calibrated=result.groups_calibrated,
                total_samples=result.total_samples,
                avg_correction_factor=result.avg_correction_factor,
            )

        except Exception as e:
            _LOGGER.error(f"Error in rescue_calibration: {e}", exc_info=True)
            await self._send_rescue_notification(
                success=False,
                message=f"Fehler: {e}",
            )

    async def _send_rescue_notification(
        self,
        success: bool,
        message: str,
        groups_calibrated: int = 0,
        total_samples: int = 0,
        avg_correction_factor: float = 1.0,
    ) -> None:
        """Send notification about rescue calibration result @zara"""
        try:
            if success:
                title = "✅ Rescue Calibration erfolgreich"
                body = (
                    f"**Ergebnis:** {message}\n\n"
                    f"**Gruppen kalibriert:** {groups_calibrated}\n"
                    f"**Samples verarbeitet:** {total_samples}\n"
                    f"**Durchschn. Korrektur-Faktor:** {avg_correction_factor:.2f}x\n\n"
                    f"Das System wird ab jetzt normal weiterlernen.\n"
                    f"Die nächste reguläre Kalibrierung erfolgt heute Nacht (23:30)."
                )
            else:
                title = "⚠️ Rescue Calibration"
                body = (
                    f"**Status:** {message}\n\n"
                    f"Falls das Problem weiterhin besteht, kontaktieren Sie bitte den Support."
                )

            await self.hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": title,
                    "message": body,
                    "notification_id": "solar_forecast_ml_rescue_calibration",
                },
            )
        except Exception as notify_err:
            _LOGGER.debug(f"Could not send notification: {notify_err}")

    async def _handle_repair_calibration(self, call: ServiceCall) -> None:
        """Handle repair_calibration service - Analyze and repair extreme bucket factors @zara

        ╔══════════════════════════════════════════════════════════════════════════╗
        ║ 🚨 NUR AUF ANWEISUNG DES ENTWICKLERS AUSFÜHREN!                         ║
        ║ DO NOT RUN WITHOUT EXPLICIT DEVELOPER INSTRUCTION!                       ║
        ╚══════════════════════════════════════════════════════════════════════════╝

        This service analyzes learning_config.json for extreme bucket factors that
        may have been learned from contaminated data (e.g., Snow/Frost samples).

        Parameters:
            max_allowed_factor: float - Factors above this are flagged (default: 4.0)
            max_global_ratio: float - Bucket factors > global × ratio are flagged (default: 2.0)
            target_factor: float - Value to set for corrected factors (default: 3.0)
            dry_run: bool - If True, only analyze (default: True)
        """
        import shutil
        from datetime import datetime
        from pathlib import Path

        max_allowed_factor = call.data.get("max_allowed_factor", 4.0)
        max_global_ratio = call.data.get("max_global_ratio", 2.0)
        target_factor = call.data.get("target_factor", 3.0)
        dry_run = call.data.get("dry_run", True)

        _LOGGER.warning("=" * 80)
        _LOGGER.warning("🔧 SERVICE: repair_calibration")
        _LOGGER.warning("⚠️  NUR AUF ANWEISUNG DES ENTWICKLERS AUSFÜHREN!")
        _LOGGER.warning("⚠️  DO NOT RUN WITHOUT EXPLICIT DEVELOPER INSTRUCTION!")
        _LOGGER.warning("=" * 80)
        _LOGGER.info(f"Parameters: max_factor={max_allowed_factor}, max_ratio={max_global_ratio}, "
                     f"target={target_factor}, dry_run={dry_run}")

        try:
            data_dir = self.coordinator.data_manager.data_dir
            physics_dir = data_dir / "physics"
            learning_config_file = physics_dir / "learning_config.json"

            # Check if file exists
            if not learning_config_file.exists():
                _LOGGER.error(f"learning_config.json not found at {learning_config_file}")
                await self._send_repair_notification(
                    success=False,
                    dry_run=dry_run,
                    message="learning_config.json nicht gefunden.",
                    outliers=[],
                )
                return

            # Load learning_config.json
            def _load_config():
                with open(learning_config_file, "r", encoding="utf-8") as f:
                    return json.load(f)

            config = await self.hass.async_add_executor_job(_load_config)

            # Analyze calibration data
            calibration_data = config.get("calibration", {})
            outliers = []

            _LOGGER.info("=" * 60)
            _LOGGER.info("ANALYZING CALIBRATION FACTORS")
            _LOGGER.info("=" * 60)

            for group_name, group_data in calibration_data.items():
                global_factor = group_data.get("global_factor", 1.0)
                bucket_factors = group_data.get("bucket_factors", {})

                _LOGGER.info(f"\n📊 Group: {group_name} (global_factor: {global_factor:.2f})")

                # Check global factor
                if global_factor > max_allowed_factor:
                    outlier = {
                        "group": group_name,
                        "bucket": "GLOBAL",
                        "factor": global_factor,
                        "reason": f"global_factor > {max_allowed_factor}",
                        "type": "global",
                    }
                    outliers.append(outlier)
                    _LOGGER.warning(f"  ⚠️  GLOBAL factor {global_factor:.2f} > {max_allowed_factor}")

                # Check each bucket
                for bucket_name, bucket_data in bucket_factors.items():
                    b_global = bucket_data.get("global_factor", 1.0)
                    b_hourly = bucket_data.get("hourly_factors", {})
                    sample_count = bucket_data.get("sample_count", 0)

                    # Check bucket global factor
                    is_outlier = False
                    reason = ""

                    if b_global > max_allowed_factor:
                        is_outlier = True
                        reason = f"factor {b_global:.2f} > max {max_allowed_factor}"
                    elif global_factor > 0.5 and b_global > (global_factor * max_global_ratio):
                        is_outlier = True
                        reason = f"factor {b_global:.2f} > {max_global_ratio}× global ({global_factor:.2f})"

                    if is_outlier:
                        outlier = {
                            "group": group_name,
                            "bucket": bucket_name,
                            "factor": b_global,
                            "reason": reason,
                            "type": "bucket_global",
                            "samples": sample_count,
                        }
                        outliers.append(outlier)
                        _LOGGER.warning(f"  ⚠️  {bucket_name}: {b_global:.2f} - {reason} (samples: {sample_count})")
                    else:
                        _LOGGER.debug(f"  ✓ {bucket_name}: {b_global:.2f} OK (samples: {sample_count})")

                    # Check hourly factors within bucket
                    for hour_str, h_factor in b_hourly.items():
                        h_is_outlier = False
                        h_reason = ""

                        if h_factor > max_allowed_factor:
                            h_is_outlier = True
                            h_reason = f"hourly factor {h_factor:.2f} > max {max_allowed_factor}"
                        elif b_global > 0.5 and h_factor > (b_global * max_global_ratio):
                            h_is_outlier = True
                            h_reason = f"hourly {h_factor:.2f} > {max_global_ratio}× bucket ({b_global:.2f})"

                        if h_is_outlier:
                            outlier = {
                                "group": group_name,
                                "bucket": bucket_name,
                                "hour": int(hour_str),
                                "factor": h_factor,
                                "reason": h_reason,
                                "type": "hourly",
                            }
                            outliers.append(outlier)
                            _LOGGER.warning(f"    ⚠️  Hour {hour_str}: {h_factor:.2f} - {h_reason}")

            # Summary
            _LOGGER.info("\n" + "=" * 60)
            _LOGGER.info(f"ANALYSIS COMPLETE: {len(outliers)} outliers found")
            _LOGGER.info("=" * 60)

            if not outliers:
                _LOGGER.info("✅ No outliers found - calibration looks healthy!")
                await self._send_repair_notification(
                    success=True,
                    dry_run=dry_run,
                    message="Keine Ausreißer gefunden - Kalibrierung ist gesund!",
                    outliers=[],
                )
                return

            # If dry_run, just report
            if dry_run:
                _LOGGER.info("\n🔍 DRY RUN - No changes made")
                _LOGGER.info("To apply fixes, run with dry_run=false")
                await self._send_repair_notification(
                    success=True,
                    dry_run=True,
                    message=f"{len(outliers)} Ausreißer gefunden (dry_run - keine Änderungen)",
                    outliers=outliers,
                    target_factor=target_factor,
                )
                return

            # ================================================================
            # APPLY CORRECTIONS
            # ================================================================
            _LOGGER.info("\n🔧 APPLYING CORRECTIONS...")

            # Step 1: Create backup
            backups_dir = data_dir / "backups" / "repair_calibration"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backups_dir / f"learning_config_backup_{timestamp}.json"

            def _create_backup():
                backups_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(learning_config_file, backup_file)
                return backup_file

            backup_path = await self.hass.async_add_executor_job(_create_backup)
            _LOGGER.info(f"✓ Backup created: {backup_path}")

            # Step 2: Apply corrections
            corrections_made = 0

            for outlier in outliers:
                group = outlier["group"]
                bucket = outlier.get("bucket")
                hour = outlier.get("hour")
                old_factor = outlier["factor"]

                if outlier["type"] == "global":
                    # Correct group global factor
                    if group in calibration_data:
                        calibration_data[group]["global_factor"] = target_factor
                        corrections_made += 1
                        _LOGGER.info(f"  ✓ {group} global: {old_factor:.2f} → {target_factor:.2f}")

                elif outlier["type"] == "bucket_global":
                    # Correct bucket global factor
                    if (group in calibration_data and
                        bucket in calibration_data[group].get("bucket_factors", {})):
                        calibration_data[group]["bucket_factors"][bucket]["global_factor"] = target_factor
                        corrections_made += 1
                        _LOGGER.info(f"  ✓ {group}/{bucket}: {old_factor:.2f} → {target_factor:.2f}")

                elif outlier["type"] == "hourly":
                    # Correct hourly factor
                    if (group in calibration_data and
                        bucket in calibration_data[group].get("bucket_factors", {}) and
                        str(hour) in calibration_data[group]["bucket_factors"][bucket].get("hourly_factors", {})):
                        calibration_data[group]["bucket_factors"][bucket]["hourly_factors"][str(hour)] = target_factor
                        corrections_made += 1
                        _LOGGER.info(f"  ✓ {group}/{bucket}/h{hour}: {old_factor:.2f} → {target_factor:.2f}")

            # Step 3: Save corrected config
            config["calibration"] = calibration_data

            # Add repair metadata
            if "metadata" not in config:
                config["metadata"] = {}
            config["metadata"]["last_repair"] = {
                "timestamp": datetime.now().isoformat(),
                "corrections_made": corrections_made,
                "target_factor": target_factor,
                "backup_file": str(backup_file),
            }

            def _save_config():
                with open(learning_config_file, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

            await self.hass.async_add_executor_job(_save_config)

            _LOGGER.info("\n" + "=" * 60)
            _LOGGER.info(f"✅ REPAIR COMPLETE: {corrections_made} corrections applied")
            _LOGGER.info(f"Backup: {backup_path}")
            _LOGGER.info("⚠️  WICHTIG: Home Assistant neustarten!")
            _LOGGER.info("=" * 60)

            await self._send_repair_notification(
                success=True,
                dry_run=False,
                message=f"{corrections_made} Korrekturen angewendet",
                outliers=outliers,
                target_factor=target_factor,
                backup_path=str(backup_path),
            )

        except Exception as e:
            _LOGGER.error(f"Error in repair_calibration: {e}", exc_info=True)
            await self._send_repair_notification(
                success=False,
                dry_run=dry_run,
                message=f"Fehler: {e}",
                outliers=[],
            )

    async def _send_repair_notification(
        self,
        success: bool,
        dry_run: bool,
        message: str,
        outliers: list,
        target_factor: float = 3.0,
        backup_path: str = None,
    ) -> None:
        """Send notification about repair_calibration result @zara"""
        try:
            if success and not outliers:
                title = "✅ Calibration Repair - Keine Probleme"
                body = f"**Ergebnis:** {message}\n\nAlle Kalibrierungsfaktoren sind im normalen Bereich."

            elif success and dry_run:
                title = "🔍 Calibration Repair - Analyse (dry_run)"
                outlier_list = "\n".join(
                    f"- {o['group']}/{o.get('bucket', 'GLOBAL')}"
                    f"{'/h' + str(o['hour']) if o.get('hour') else ''}: "
                    f"{o['factor']:.2f} ({o['reason']})"
                    for o in outliers[:10]  # Limit to 10
                )
                if len(outliers) > 10:
                    outlier_list += f"\n... und {len(outliers) - 10} weitere"

                body = (
                    f"**{len(outliers)} Ausreißer gefunden:**\n\n"
                    f"{outlier_list}\n\n"
                    f"**Vorgeschlagene Korrektur:** Alle auf {target_factor:.1f} setzen\n\n"
                    f"⚠️ Um Korrekturen anzuwenden, Service mit `dry_run: false` ausführen."
                )

            elif success:
                title = "✅ Calibration Repair - Korrekturen angewendet"
                body = (
                    f"**Ergebnis:** {message}\n\n"
                    f"**Ziel-Faktor:** {target_factor:.1f}\n"
                    f"**Backup:** `{backup_path}`\n\n"
                    f"⚠️ **WICHTIG:** Home Assistant neustarten um Änderungen zu aktivieren!"
                )

            else:
                title = "⚠️ Calibration Repair - Fehler"
                body = f"**Status:** {message}"

            await self.hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": title,
                    "message": body,
                    "notification_id": "solar_forecast_ml_repair_calibration",
                },
            )
        except Exception as notify_err:
            _LOGGER.debug(f"Could not send notification: {notify_err}")

    async def _handle_winter_bucket_correction(self, call: ServiceCall) -> None:
        """Handle winter_bucket_correction_v13 service - V13.1 Winter-Fix @zara

        ╔══════════════════════════════════════════════════════════════════════════╗
        ║ ⚠️ WARNUNG: NUR AUF ANWEISUNG DES ENTWICKLERS AUSFÜHREN!               ║
        ║ Falsche Anwendung kann die KI-Kalibrierung nachhaltig schädigen!        ║
        ╚══════════════════════════════════════════════════════════════════════════╝

        This service reduces the confidence of all bucket factors to allow faster
        learning with the new winter-corrected cloud factor formula.

        The FACTOR VALUES are preserved - only the confidence is reduced.
        This allows the system to quickly adapt while keeping the general
        direction of the calibration.

        Parameters:
            target_confidence: float - Target confidence value (default: 0.5)
            dry_run: bool - If True, only analyze (default: True)
        """
        import shutil
        from datetime import datetime

        target_confidence = call.data.get("target_confidence", 0.5)
        dry_run = call.data.get("dry_run", True)

        _LOGGER.info("=" * 80)
        _LOGGER.info("🌨️ SERVICE: winter_bucket_correction_v13 (DEVELOPER ONLY)")
        _LOGGER.info("⚠️ WARNUNG: Dieser Service kann die KI nachhaltig schädigen!")
        _LOGGER.info("=" * 80)
        _LOGGER.info(f"Target confidence: {target_confidence}")
        _LOGGER.info(f"Dry run: {dry_run}")

        try:
            data_dir = self.coordinator.data_manager.data_dir
            physics_dir = data_dir / "physics"
            learning_config_file = physics_dir / "learning_config.json"

            # Check if file exists
            if not learning_config_file.exists():
                _LOGGER.error(f"learning_config.json not found at {learning_config_file}")
                await self._send_winter_bucket_correction_notification(
                    success=False,
                    dry_run=dry_run,
                    message="learning_config.json nicht gefunden.",
                    buckets_affected=0,
                )
                return

            # Load learning_config.json
            def _load_config():
                with open(learning_config_file, "r", encoding="utf-8") as f:
                    return json.load(f)

            config = await self.hass.async_add_executor_job(_load_config)

            # Analyze and collect statistics
            group_calibration = config.get("group_calibration", {})
            buckets_affected = 0
            confidence_changes = []

            _LOGGER.info("=" * 60)
            _LOGGER.info("ANALYZING BUCKET CONFIDENCE VALUES")
            _LOGGER.info("=" * 60)

            for group_name, group_data in group_calibration.items():
                bucket_factors = group_data.get("bucket_factors", {})

                _LOGGER.info(f"\n📊 Group: {group_name}")

                for bucket_name, bucket_data in bucket_factors.items():
                    current_confidence = bucket_data.get("confidence", 1.0)
                    current_factor = bucket_data.get("factor", 1.0)
                    sample_count = bucket_data.get("sample_count", 0)

                    if current_confidence > target_confidence:
                        buckets_affected += 1
                        confidence_changes.append({
                            "group": group_name,
                            "bucket": bucket_name,
                            "old_confidence": current_confidence,
                            "new_confidence": target_confidence,
                            "factor": current_factor,
                            "samples": sample_count,
                        })
                        _LOGGER.info(
                            f"  ↓ {bucket_name}: conf {current_confidence:.2f} → {target_confidence:.2f} "
                            f"(factor={current_factor:.2f}, samples={sample_count})"
                        )
                    else:
                        _LOGGER.debug(
                            f"  ✓ {bucket_name}: conf {current_confidence:.2f} ≤ {target_confidence:.2f} (OK)"
                        )

            # Summary
            _LOGGER.info("\n" + "=" * 60)
            _LOGGER.info(f"ANALYSIS COMPLETE: {buckets_affected} buckets would be affected")
            _LOGGER.info("=" * 60)

            if buckets_affected == 0:
                _LOGGER.info("✅ All bucket confidences are already at or below target!")
                await self._send_winter_bucket_correction_notification(
                    success=True,
                    dry_run=dry_run,
                    message="Alle Bucket-Konfidenzen sind bereits ≤ Zielwert.",
                    buckets_affected=0,
                    target_confidence=target_confidence,
                )
                return

            # If dry_run, just report
            if dry_run:
                _LOGGER.info("\n🔍 DRY RUN - No changes made")
                _LOGGER.info("To apply changes, run with dry_run=false")
                await self._send_winter_bucket_correction_notification(
                    success=True,
                    dry_run=True,
                    message=f"{buckets_affected} Buckets würden zurückgesetzt (dry_run)",
                    buckets_affected=buckets_affected,
                    target_confidence=target_confidence,
                    changes=confidence_changes[:10],
                )
                return

            # ================================================================
            # APPLY CONFIDENCE RESET
            # ================================================================
            _LOGGER.info("\n🔧 APPLYING CONFIDENCE RESET...")

            # Step 1: Create backup
            backups_dir = data_dir / "backups" / "confidence_reset"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backups_dir / f"learning_config_backup_{timestamp}.json"

            def _create_backup():
                backups_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(learning_config_file, backup_file)
                return backup_file

            backup_path = await self.hass.async_add_executor_job(_create_backup)
            _LOGGER.info(f"✓ Backup created: {backup_path}")

            # Step 2: Apply confidence reset
            reset_count = 0

            for group_name, group_data in group_calibration.items():
                bucket_factors = group_data.get("bucket_factors", {})

                for bucket_name, bucket_data in bucket_factors.items():
                    current_confidence = bucket_data.get("confidence", 1.0)

                    if current_confidence > target_confidence:
                        bucket_data["confidence"] = target_confidence
                        reset_count += 1

            # Step 3: Update metadata
            config["group_calibration"] = group_calibration

            if "metadata" not in config:
                config["metadata"] = {}
            config["metadata"]["confidence_reset"] = {
                "timestamp": datetime.now().isoformat(),
                "target_confidence": target_confidence,
                "buckets_reset": reset_count,
                "reason": "V13.1 Winter-Fix: Enable faster learning with new cloud factor formula",
                "backup_file": str(backup_file),
            }

            # Step 4: Save config
            def _save_config():
                with open(learning_config_file, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

            await self.hass.async_add_executor_job(_save_config)

            _LOGGER.info("\n" + "=" * 60)
            _LOGGER.info(f"✅ CONFIDENCE RESET COMPLETE: {reset_count} buckets updated")
            _LOGGER.info(f"Backup: {backup_path}")
            _LOGGER.info("Das System wird jetzt schneller mit der neuen Formel lernen.")
            _LOGGER.info("=" * 60)

            await self._send_winter_bucket_correction_notification(
                success=True,
                dry_run=False,
                message=f"{reset_count} Bucket-Konfidenzen zurückgesetzt",
                buckets_affected=reset_count,
                target_confidence=target_confidence,
                backup_path=str(backup_path),
            )

        except Exception as e:
            _LOGGER.error(f"Error in winter_bucket_correction_v13: {e}", exc_info=True)
            await self._send_winter_bucket_correction_notification(
                success=False,
                dry_run=dry_run,
                message=f"Fehler: {e}",
                buckets_affected=0,
            )

    async def _send_winter_bucket_correction_notification(
        self,
        success: bool,
        dry_run: bool,
        message: str,
        buckets_affected: int,
        target_confidence: float = 0.5,
        changes: list = None,
        backup_path: str = None,
    ) -> None:
        """Send notification about Winter Bucket Correction result @zara"""
        try:
            if success and buckets_affected == 0:
                title = "✅ Winter Bucket Correction V13 - Nicht nötig"
                body = f"**Ergebnis:** {message}\n\nAlle Bucket-Konfidenzen sind bereits niedrig genug."

            elif success and dry_run:
                title = "🔍 Winter Bucket Correction V13 - Analyse (dry_run)"
                changes_list = ""
                if changes:
                    changes_list = "\n".join(
                        f"- {c['group']}/{c['bucket']}: {c['old_confidence']:.2f} → {c['new_confidence']:.2f}"
                        for c in changes[:5]
                    )
                    if len(changes) > 5:
                        changes_list += f"\n... und {len(changes) - 5} weitere"

                body = (
                    f"**{buckets_affected} Buckets würden korrigiert:**\n\n"
                    f"{changes_list}\n\n"
                    f"**Ziel-Konfidenz:** {target_confidence:.1f}\n\n"
                    f"⚠️ Um Änderungen anzuwenden, Service mit `dry_run: false` ausführen.\n"
                    f"⚠️ NUR AUF ANWEISUNG DES ENTWICKLERS AUSFÜHREN!"
                )

            elif success:
                title = "✅ Winter Bucket Correction V13 - Abgeschlossen"
                body = (
                    f"**Ergebnis:** {message}\n\n"
                    f"**Ziel-Konfidenz:** {target_confidence:.1f}\n"
                    f"**Backup:** `{backup_path}`\n\n"
                    f"Das System wird jetzt schneller mit der neuen Winter-Formel lernen.\n"
                    f"Die Faktor-WERTE wurden beibehalten - nur die Konfidenz wurde reduziert."
                )

            else:
                title = "⚠️ Winter Bucket Correction V13 - Fehler"
                body = f"**Status:** {message}"

            await self.hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": title,
                    "message": body,
                    "notification_id": "solar_forecast_ml_winter_bucket_correction",
                },
            )
        except Exception as notify_err:
            _LOGGER.debug(f"Could not send notification: {notify_err}")

