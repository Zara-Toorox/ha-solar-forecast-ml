"""Service Registry for Solar Forecast ML Integration V10.0.0 @zara

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

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Awaitable, Callable, List

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall

from ..const import (
    DOMAIN,
    SERVICE_BOOTSTRAP_FROM_HISTORY,
    SERVICE_BOOTSTRAP_PHYSICS_FROM_HISTORY,
    SERVICE_BUILD_ASTRONOMY_CACHE,
    SERVICE_RUN_WEATHER_CORRECTION,
    SERVICE_REFRESH_OPEN_METEO_CACHE,
    SERVICE_REFRESH_CACHE_TODAY,
    SERVICE_RESET_LEARNING_DATA,
    SERVICE_RETRAIN_MODEL,
    SERVICE_RUN_ALL_DAY_END_TASKS,
    SERVICE_SEND_DAILY_BRIEFING,
    SERVICE_TEST_MORNING_ROUTINE,
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

        from ..services.service_battery_discovery import async_setup_services as setup_battery_discovery

        await setup_battery_discovery(self.hass)

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
            # ML Services
            ServiceDefinition(
                name=SERVICE_RETRAIN_MODEL,
                handler=self._handle_retrain_model,
                description="Force ML model retraining",
            ),
            ServiceDefinition(
                name=SERVICE_RESET_LEARNING_DATA,
                handler=self._handle_reset_model,
                description="Reset ML model and learning data",
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
                name=SERVICE_BOOTSTRAP_FROM_HISTORY,
                handler=self._handle_bootstrap_from_history,
                description="Bootstrap pattern learning from Home Assistant history",
            ),

            # Physics Services
            ServiceDefinition(
                name=SERVICE_BOOTSTRAP_PHYSICS_FROM_HISTORY,
                handler=self._handle_bootstrap_physics_from_history,
                description="Bootstrap Physics-First models (GeometryLearner, ResidualTrainer) from history",
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
        ]

    # =========================================================================
    # ML Services
    # =========================================================================

    async def _handle_retrain_model(self, call: ServiceCall) -> None:
        """Handle force_retrain service @zara"""
        try:
            if self.coordinator.ml_predictor:
                success = await self.coordinator.ml_predictor.force_training()
                if not success:
                    _LOGGER.error("ML model retraining failed")
        except Exception as e:
            _LOGGER.error(f"Error in force_retrain: {e}")

    async def _handle_reset_model(self, call: ServiceCall) -> None:
        """Handle reset_model service @zara"""
        try:
            if self.coordinator.ml_predictor:
                success = await self.coordinator.ml_predictor.reset_model()
                if not success:
                    _LOGGER.error("ML model reset failed")
        except Exception as e:
            _LOGGER.error(f"Error in reset_model: {e}")

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
        """Handle refresh_open_meteo_cache service - Refresh Open-Meteo direct radiation cache @zara"""
        try:
            if not hasattr(self.coordinator, 'weather_pipeline_manager'):
                return

            pipeline = self.coordinator.weather_pipeline_manager

            if not pipeline.weather_corrector:
                return

            corrector = pipeline.weather_corrector

            if not corrector._open_meteo_client:
                return

            forecast = await corrector._open_meteo_client.get_hourly_forecast(hours=72)

            if not forecast:
                _LOGGER.warning("Open-Meteo fetch failed")
                return

            corrector._open_meteo_cache.clear()
            for entry in forecast:
                date = entry.get("date")
                hour = entry.get("hour")

                if date and hour is not None:
                    if date not in corrector._open_meteo_cache:
                        corrector._open_meteo_cache[date] = {}
                    corrector._open_meteo_cache[date][hour] = {
                        "direct_radiation": entry.get("direct_radiation", 0),
                        "diffuse_radiation": entry.get("diffuse_radiation", 0),
                        "ghi": entry.get("ghi", 0),
                        "cloud_cover": entry.get("cloud_cover", 0),
                        "temperature": entry.get("temperature"),
                        "humidity": entry.get("humidity"),
                        "precipitation": entry.get("precipitation", 0),
                        "wind_speed": entry.get("wind_speed"),
                    }

        except Exception as e:
            _LOGGER.error(f"Error in refresh_open_meteo_cache: {e}")

    async def _handle_bootstrap_from_history(self, call: ServiceCall) -> None:
        """Handle bootstrap_from_history service - Bootstrap pattern learning from HA history @zara"""
        from datetime import datetime, timezone
        from homeassistant.components.recorder import get_instance
        from homeassistant.components.recorder.history import state_changes_during_period
        from collections import defaultdict

        days = call.data.get('days', 30)
        cumulative_yield_sensor = call.data.get('cumulative_yield_sensor')

        tz = dt_util.get_default_time_zone()

        try:
            required_sensors = {
                'power': 'power_entity',
                'yield': 'solar_yield_today',
            }

            optional_sensors = {
                'solar_radiation_wm2': 'solar_radiation_sensor',
                'temperature': 'temp_sensor',
                'humidity': 'humidity_sensor',
                'wind_speed': 'wind_sensor',
                'rain': 'rain_sensor',
                'lux': 'lux_sensor',
                'pressure': 'pressure_sensor'
            }

            config_data = self.coordinator.config_entry.data
            config_options = self.coordinator.config_entry.options

            missing_required = []
            configured_sensors = {}

            for sensor_name, config_key in required_sensors.items():
                entity_id = config_data.get(config_key) or config_options.get(config_key)
                if not entity_id:
                    missing_required.append(sensor_name)
                else:
                    configured_sensors[sensor_name] = entity_id

            for sensor_name, config_key in optional_sensors.items():
                entity_id = config_data.get(config_key) or config_options.get(config_key)
                if entity_id:
                    configured_sensors[sensor_name] = entity_id

            if cumulative_yield_sensor:
                configured_sensors['cumulative_yield'] = cumulative_yield_sensor

            if missing_required:
                _LOGGER.error(f"Bootstrap failed - Missing: {', '.join(missing_required)}")
                return

            now = datetime.now(timezone.utc)
            start_time = now - timedelta(days=days)

            history_data = {}
            for sensor_name, entity_id in configured_sensors.items():
                sensor_history = await get_instance(self.hass).async_add_executor_job(
                    state_changes_during_period,
                    self.hass,
                    start_time,
                    now,
                    entity_id,
                    True,
                    False,
                    0.0
                )

                if sensor_history and entity_id in sensor_history:
                    history_data[entity_id] = sensor_history[entity_id]

            if not history_data:
                _LOGGER.warning("No history data found")
                return

            entity_to_sensor = {v: k for k, v in configured_sensors.items()}
            hourly_data = defaultdict(lambda: defaultdict(list))

            for entity_id, states in history_data.items():
                sensor_type = entity_to_sensor.get(entity_id)
                if not sensor_type:
                    continue

                for state in states:
                    if state.state in ("unavailable", "unknown", "none", None):
                        continue

                    try:
                        value = float(state.state)
                        state_time = state.last_changed.astimezone(tz)
                        date_key = state_time.strftime("%Y-%m-%d")
                        hour_key = state_time.hour

                        hourly_data[date_key][hour_key].append({
                            "sensor": sensor_type,
                            "value": value,
                            "time": state_time
                        })
                    except (ValueError, TypeError):
                        continue

            aggregated = {}
            for date_str, hours in hourly_data.items():
                for hour, readings in hours.items():
                    key = f"{date_str}_{hour:02d}"
                    sensor_values = defaultdict(list)
                    for reading in readings:
                        sensor_values[reading["sensor"]].append(reading["value"])

                    avg_data = {}
                    for sensor, values in sensor_values.items():
                        if sensor == "yield":
                            avg_data[sensor] = max(values)
                        else:
                            avg_data[sensor] = sum(values) / len(values)

                    if avg_data:
                        aggregated[key] = {"date": date_str, "hour": hour, **avg_data}

            if cumulative_yield_sensor and 'cumulative_yield' in configured_sensors:
                cumulative_entity = configured_sensors['cumulative_yield']
                if cumulative_entity in history_data:
                    cumulative_states = history_data[cumulative_entity]
                    sorted_cumulative = sorted(cumulative_states, key=lambda s: s.last_changed)

                    hourly_cumulative = {}

                    for state in sorted_cumulative:
                        if state.state in ("unavailable", "unknown", "none", None):
                            continue
                        try:
                            value = float(state.state)
                            if value < 0:
                                continue
                            state_time = state.last_changed.astimezone(tz)
                            date_str = state_time.strftime("%Y-%m-%d")
                            hour = state_time.hour
                            key = (date_str, hour)

                            if key not in hourly_cumulative or value > hourly_cumulative[key]:
                                hourly_cumulative[key] = value
                        except (ValueError, TypeError):
                            continue

                    sorted_keys = sorted(hourly_cumulative.keys())

                    prev_value = None
                    cumulative_hours_added = 0

                    for key in sorted_keys:
                        date_str, hour = key
                        current_value = hourly_cumulative[key]

                        if prev_value is not None:
                            delta = current_value - prev_value
                            if 0.001 < delta < 5.0:
                                agg_key = f"{date_str}_{hour:02d}"
                                if agg_key not in aggregated:
                                    aggregated[agg_key] = {"date": date_str, "hour": hour}
                                existing = aggregated[agg_key].get("yield", 0)
                                if delta > existing:
                                    aggregated[agg_key]["yield"] = delta
                                    cumulative_hours_added += 1

                        prev_value = current_value

            dates_in_data = set(hour_data["date"] for hour_data in aggregated.values())
            astronomy_cache = {}

            if dates_in_data:
                from datetime import datetime as dt
                min_date_str = min(dates_in_data)
                max_date_str = max(dates_in_data)
                min_date = dt.strptime(min_date_str, "%Y-%m-%d").date()
                max_date = dt.strptime(max_date_str, "%Y-%m-%d").date()
                days_span = (max_date - min_date).days

                from ..astronomy.astronomy_cache import AstronomyCache
                astro_cache = AstronomyCache(
                    data_dir=self.coordinator.data_manager.data_dir,
                    data_manager=self.coordinator.data_manager
                )

                lat = self.hass.config.latitude
                lon = self.hass.config.longitude
                tz_str = str(self.hass.config.time_zone)
                elev = self.hass.config.elevation

                astro_cache.initialize_location(lat, lon, tz_str, elev)
                solar_capacity = self.coordinator.solar_capacity or 5.0

                await astro_cache.rebuild_cache(
                    system_capacity_kwp=solar_capacity,
                    start_date=max_date,
                    days_back=days_span + 1,
                    days_ahead=7
                )

                astronomy_path = self.coordinator.data_manager.data_dir / "stats" / "astronomy_cache.json"
                if astronomy_path.exists():
                    try:
                        def _load_astronomy():
                            with open(astronomy_path, 'r', encoding='utf-8') as f:
                                return json.load(f)
                        astronomy_data = await self.hass.async_add_executor_job(_load_astronomy)
                        astronomy_cache = astronomy_data.get("days", {})
                    except Exception as e:
                        _LOGGER.warning(f"Could not load astronomy cache: {e}")

            # Step 5: Save hourly_weather_actual.json
            _LOGGER.info("STEP 5/8: Saving to hourly_weather_actual.json...")

            hourly_actual_path = self.coordinator.data_manager.data_dir / "stats" / "hourly_weather_actual.json"
            existing_data = {
                "version": "1.0",
                "metadata": {
                    "created_at": dt_util.now().isoformat(),
                    "last_updated": dt_util.now().isoformat(),
                },
                "hourly_data": {}
            }

            if hourly_actual_path.exists():
                try:
                    def _load_hourly_actual():
                        with open(hourly_actual_path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    existing_data = await self.hass.async_add_executor_job(_load_hourly_actual)
                    if "metadata" not in existing_data:
                        existing_data["metadata"] = {"created_at": dt_util.now().isoformat()}
                    if "hourly_data" not in existing_data:
                        existing_data["hourly_data"] = {}
                except Exception:
                    pass

            for key, data in sorted(aggregated.items()):
                date_str = data["date"]
                hour_num = data["hour"]

                if date_str not in existing_data["hourly_data"]:
                    existing_data["hourly_data"][date_str] = {}

                hour_entry = {
                    "temperature_c": data.get("temperature"),
                    "solar_radiation_wm2": data.get("solar_radiation_wm2"),
                    "humidity_percent": data.get("humidity"),
                    "wind_speed_ms": data.get("wind_speed"),
                    "precipitation_mm": data.get("rain"),
                    "pressure_hpa": data.get("pressure"),
                    "lux": data.get("lux"),
                    "timestamp": f"{date_str}T{hour_num:02d}:00:00",
                    "source": "bootstrap_from_history"
                }

                existing_data["hourly_data"][date_str][str(hour_num)] = hour_entry

            existing_data["metadata"]["last_updated"] = dt_util.now().isoformat()
            merged_data = existing_data

            def _save_hourly_actual():
                with open(hourly_actual_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_data, f, indent=2, ensure_ascii=False)

            await self.hass.async_add_executor_job(_save_hourly_actual)

            total_hours = sum(len(hours) for hours in existing_data["hourly_data"].values())
            _LOGGER.info(f"Saved {total_hours} hourly records")

            # Step 6: Learn geometry patterns
            _LOGGER.info("STEP 6/8: Learning geometry patterns...")

            learned_patterns_path = self.coordinator.data_manager.data_dir / "ml" / "learned_patterns.json"
            patterns = {
                "version": "1.0.0",
                "last_updated": dt_util.now().isoformat(),
                "geometry_factors": {
                    "sun_elevation_ranges": {
                        "0_5": {"factor": 2.5, "samples": 0, "confidence": 0.3},
                        "5_10": {"factor": 2.2, "samples": 0, "confidence": 0.3},
                        "10_15": {"factor": 1.9, "samples": 0, "confidence": 0.3},
                        "15_20": {"factor": 1.6, "samples": 0, "confidence": 0.3},
                        "20_25": {"factor": 1.4, "samples": 0, "confidence": 0.3},
                        "25_30": {"factor": 1.2, "samples": 0, "confidence": 0.3},
                        "30_plus": {"factor": 1.1, "samples": 0, "confidence": 0.3},
                    }
                },
                "geometry_corrections": {"monthly": {}},
                "cloud_impacts": {"hour_patterns": {}},
                "seasonal_adjustments": {"months": {}},
                "efficiency_curves": {"radiation_buckets": {}},
                "metadata": {
                    "created": dt_util.now().strftime("%Y-%m-%d"),
                    "total_learning_days": 0,
                    "clear_sky_days_detected": 0,
                    "cloudy_days_detected": 0,
                    "last_pattern_update": dt_util.now().isoformat()
                }
            }

            if learned_patterns_path.exists():
                try:
                    def _load_patterns():
                        with open(learned_patterns_path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    existing_patterns = await self.hass.async_add_executor_job(_load_patterns)
                    patterns.update(existing_patterns)
                    if "geometry_corrections" not in patterns:
                        patterns["geometry_corrections"] = {"monthly": {}}
                except Exception:
                    pass

            geometry_samples = defaultdict(lambda: {"actual": [], "theoretical": []})

            for date_str, day_hours in merged_data["hourly_data"].items():
                if date_str not in astronomy_cache:
                    continue

                day_astro = astronomy_cache[date_str]
                sorted_hours = sorted(day_hours.keys(), key=lambda x: int(x))

                for i, hour_str in enumerate(sorted_hours):
                    hour_num = int(hour_str)
                    hour_astro = day_astro.get("hourly", {}).get(hour_str)

                    if not hour_astro:
                        continue

                    sun_elevation = hour_astro.get("elevation_deg", 0)
                    theoretical_kwh = hour_astro.get("theoretical_max_pv_kwh", 0)
                    cumulative_yield = aggregated.get(f"{date_str}_{hour_num:02d}", {}).get("yield", 0)

                    if i == 0:
                        hourly_yield = cumulative_yield
                    else:
                        prev_hour_str = sorted_hours[i - 1]
                        prev_hour_num = int(prev_hour_str)
                        prev_cumulative = aggregated.get(f"{date_str}_{prev_hour_num:02d}", {}).get("yield", 0)
                        hourly_yield = max(0, cumulative_yield - prev_cumulative)

                    if sun_elevation <= 0 or theoretical_kwh <= 0 or hourly_yield <= 0:
                        continue

                    if hourly_yield > theoretical_kwh * 3:
                        continue

                    if sun_elevation < 5:
                        bucket = "0_5"
                    elif sun_elevation < 10:
                        bucket = "5_10"
                    elif sun_elevation < 15:
                        bucket = "10_15"
                    elif sun_elevation < 20:
                        bucket = "15_20"
                    elif sun_elevation < 25:
                        bucket = "20_25"
                    elif sun_elevation < 30:
                        bucket = "25_30"
                    else:
                        bucket = "30_plus"

                    geometry_samples[bucket]["actual"].append(hourly_yield)
                    geometry_samples[bucket]["theoretical"].append(theoretical_kwh)

            for bucket, samples in geometry_samples.items():
                if len(samples["actual"]) >= 5:
                    total_actual = sum(samples["actual"])
                    total_theoretical = sum(samples["theoretical"])

                    if total_theoretical > 0:
                        factor = total_actual / total_theoretical
                        patterns["geometry_factors"]["sun_elevation_ranges"][bucket]["factor"] = round(factor, 3)
                        patterns["geometry_factors"]["sun_elevation_ranges"][bucket]["samples"] = len(samples["actual"])
                        patterns["geometry_factors"]["sun_elevation_ranges"][bucket]["confidence"] = min(0.9, 0.3 + len(samples["actual"]) * 0.02)

            def _save_patterns():
                with open(learned_patterns_path, 'w', encoding='utf-8') as f:
                    json.dump(patterns, f, indent=2, ensure_ascii=False)

            await self.hass.async_add_executor_job(_save_patterns)
            _LOGGER.info(f"Updated geometry factors for {len(geometry_samples)} elevation buckets")

            # Step 7: Calculate statistics
            _LOGGER.info("STEP 7/8: Calculating statistics...")

            daily_yields = defaultdict(float)
            peak_power = {"power_w": 0.0, "date": None, "at": None}

            for key, hour_data in aggregated.items():
                date_str = hour_data["date"]
                hour_num = hour_data["hour"]
                yield_kwh = hour_data.get("yield", 0) or 0
                power_w = hour_data.get("power", 0) or 0

                if yield_kwh > daily_yields[date_str]:
                    daily_yields[date_str] = yield_kwh

                if power_w > peak_power["power_w"]:
                    peak_power["power_w"] = round(power_w, 1)
                    peak_power["date"] = date_str
                    peak_power["at"] = f"{date_str}T{hour_num:02d}:30:00+01:00"

            sorted_days = sorted(daily_yields.items())
            total_yield = sum(daily_yields.values())

            last_7_days = sorted_days[-7:] if len(sorted_days) >= 7 else sorted_days
            last_30_days = sorted_days[-30:] if len(sorted_days) >= 30 else sorted_days

            avg_7 = sum(y for _, y in last_7_days) / len(last_7_days) if last_7_days else 0
            avg_30 = sum(y for _, y in last_30_days) / len(last_30_days) if last_30_days else 0

            _LOGGER.info(f"7-day average: {avg_7:.2f} kWh/day")
            _LOGGER.info(f"30-day average: {avg_30:.2f} kWh/day")

            # Step 8: Update daily_forecasts.json
            _LOGGER.info("STEP 8/8: Updating daily_forecasts.json...")

            daily_forecasts_path = self.coordinator.data_manager.data_dir / "stats" / "daily_forecasts.json"
            forecasts = {
                "version": "3.0.0",
                "today": {},
                "statistics": {
                    "all_time_peak": {"power_w": 0.0, "date": None, "at": None},
                    "last_7_days": {"avg_yield_kwh": 0.0, "total_yield_kwh": 0.0},
                    "last_30_days": {"avg_yield_kwh": 0.0, "total_yield_kwh": 0.0},
                    "last_365_days": {"total_yield_kwh": 0.0}
                },
                "history": [],
                "metadata": {"retention_days": 730, "history_entries": 0}
            }

            if daily_forecasts_path.exists():
                try:
                    def _load_forecasts():
                        with open(daily_forecasts_path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    forecasts = await self.hass.async_add_executor_job(_load_forecasts)
                except Exception:
                    pass

            forecasts["statistics"]["all_time_peak"] = peak_power
            forecasts["statistics"]["last_7_days"] = {
                "avg_yield_kwh": round(avg_7, 2),
                "total_yield_kwh": round(sum(y for _, y in last_7_days), 2),
                "calculated_at": dt_util.now().isoformat()
            }
            forecasts["statistics"]["last_30_days"] = {
                "avg_yield_kwh": round(avg_30, 2),
                "total_yield_kwh": round(sum(y for _, y in last_30_days), 2),
                "calculated_at": dt_util.now().isoformat()
            }
            forecasts["statistics"]["last_365_days"] = {
                "total_yield_kwh": round(total_yield, 2),
                "calculated_at": dt_util.now().isoformat()
            }

            existing_history_dates = {h.get("date") for h in forecasts.get("history", [])}
            for date_str, yield_kwh in sorted_days:
                if date_str not in existing_history_dates:
                    forecasts["history"].append({
                        "date": date_str,
                        "yield_kwh": round(yield_kwh, 2),
                        "source": "bootstrap_from_history"
                    })

            forecasts["metadata"]["history_entries"] = len(forecasts["history"])
            forecasts["metadata"]["last_update"] = dt_util.now().isoformat()

            def _save_forecasts():
                with open(daily_forecasts_path, 'w', encoding='utf-8') as f:
                    json.dump(forecasts, f, indent=2, ensure_ascii=False)

            await self.hass.async_add_executor_job(_save_forecasts)

            _LOGGER.info("=" * 80)
            _LOGGER.info("BOOTSTRAP COMPLETE!")
            _LOGGER.info(f"Peak power: {peak_power['power_w']} W")
            _LOGGER.info(f"Historical entries: {len(forecasts['history'])} days")
            _LOGGER.info("=" * 80)

        except Exception as e:
            _LOGGER.error(f"ERROR in bootstrap_from_history: {e}", exc_info=True)

    # =========================================================================
    # Physics Services
    # =========================================================================

    async def _handle_bootstrap_physics_from_history(self, call: ServiceCall) -> None:
        """Handle bootstrap_physics_from_history service @zara

        Bootstrap Physics-First models (GeometryLearner, ResidualTrainer) from HA history.

        This service:
        1. Fetches historical production data from HA Recorder
        2. Fetches historical weather from Open-Meteo Archive API
        3. Calculates astronomy (sun position) retroactively for each hour
        4. Trains GeometryLearner with clear-sky samples
        5. Trains ResidualTrainer with all samples

        Requires: power_entity configured
        Optional: W/m² sensor, Lux sensor (improves training quality)
        """
        from datetime import datetime, timezone, timedelta
        from homeassistant.components.recorder import get_instance
        from homeassistant.components.recorder.history import state_changes_during_period
        from collections import defaultdict

        _LOGGER.info("=" * 80)
        _LOGGER.info("SERVICE: bootstrap_physics_from_history - Physics-First Bootstrap")
        _LOGGER.info("=" * 80)

        # Get parameters
        days = call.data.get('days', 180)  # Default 6 months
        cumulative_yield_sensor = call.data.get('cumulative_yield_sensor')  # Optional cumulative sensor
        _LOGGER.info(f"Requested days: {days}")
        if cumulative_yield_sensor:
            _LOGGER.info(f"Using cumulative yield sensor: {cumulative_yield_sensor}")

        tz = dt_util.get_default_time_zone()

        try:
            # =================================================================
            # STEP 1: Validate required sensors
            # =================================================================
            _LOGGER.info("STEP 1/8: Validating sensor configuration...")

            config_data = self.coordinator.config_entry.data
            config_options = self.coordinator.config_entry.options

            # Required: power_entity and yield
            power_entity = config_data.get('power_entity') or config_options.get('power_entity')
            yield_entity = config_data.get('solar_yield_today') or config_options.get('solar_yield_today')

            if not power_entity or not yield_entity:
                _LOGGER.error("BOOTSTRAP FAILED - Missing required sensors!")
                _LOGGER.error("Required: power_entity, solar_yield_today")
                return

            _LOGGER.info(f"  ✓ power_entity: {power_entity}")
            _LOGGER.info(f"  ✓ yield_entity: {yield_entity}")
            if cumulative_yield_sensor:
                _LOGGER.info(f"  ✓ cumulative_yield_sensor: {cumulative_yield_sensor} (service parameter)")

            # Optional sensors
            solar_radiation_sensor = config_data.get('solar_radiation_sensor') or config_options.get('solar_radiation_sensor')
            lux_sensor = config_data.get('lux_sensor') or config_options.get('lux_sensor')
            temp_sensor = config_data.get('temp_sensor') or config_options.get('temp_sensor')

            if solar_radiation_sensor:
                _LOGGER.info(f"  ✓ solar_radiation_sensor: {solar_radiation_sensor} (optional)")
            if lux_sensor:
                _LOGGER.info(f"  ✓ lux_sensor: {lux_sensor} (optional)")
            if temp_sensor:
                _LOGGER.info(f"  ✓ temp_sensor: {temp_sensor} (optional)")

            # =================================================================
            # STEP 2: Fetch production history from HA Recorder
            # =================================================================
            _LOGGER.info(f"STEP 2/8: Fetching production history ({days} days)...")

            now = datetime.now(timezone.utc)
            start_time = now - timedelta(days=days)

            # Fetch yield history (cumulative daily kWh)
            yield_history = await get_instance(self.hass).async_add_executor_job(
                state_changes_during_period,
                self.hass,
                start_time,
                now,
                yield_entity,
                True,
                False,
                0.0
            )

            yield_states = []
            if yield_history and yield_entity in yield_history:
                yield_states = yield_history[yield_entity]
                _LOGGER.info(f"Found {len(yield_states)} daily yield history states")

            # Also fetch cumulative sensor if provided
            cumulative_states = []
            if cumulative_yield_sensor:
                cumulative_history = await get_instance(self.hass).async_add_executor_job(
                    state_changes_during_period,
                    self.hass,
                    start_time,
                    now,
                    cumulative_yield_sensor,
                    True,
                    False,
                    0.0
                )
                if cumulative_history and cumulative_yield_sensor in cumulative_history:
                    cumulative_states = cumulative_history[cumulative_yield_sensor]
                    _LOGGER.info(f"Found {len(cumulative_states)} cumulative yield history states")

            if not yield_states and not cumulative_states:
                _LOGGER.error("No yield history found from either sensor!")
                return

            # Fetch optional sensor history
            sensor_history = {}
            for sensor_name, entity_id in [
                ('solar_radiation', solar_radiation_sensor),
                ('lux', lux_sensor),
                ('temperature', temp_sensor),
            ]:
                if entity_id:
                    history = await get_instance(self.hass).async_add_executor_job(
                        state_changes_during_period,
                        self.hass,
                        start_time,
                        now,
                        entity_id,
                        True,
                        False,
                        0.0
                    )
                    if history and entity_id in history:
                        sensor_history[sensor_name] = history[entity_id]
                        _LOGGER.info(f"  ✓ {sensor_name}: {len(history[entity_id])} states")

            # =================================================================
            # STEP 3: Aggregate to hourly production
            # =================================================================
            _LOGGER.info("STEP 3/8: Aggregating hourly production...")

            # Group yield by date and track hourly changes
            hourly_production = defaultdict(lambda: defaultdict(float))

            # ---------------------------------------------------------------
            # METHOD 1: Process daily yield sensor (resets daily)
            # ---------------------------------------------------------------
            if yield_states:
                _LOGGER.info("  Processing daily yield sensor (solar_yield_today)...")
                sorted_yields = sorted(yield_states, key=lambda s: s.last_changed)

                prev_yield = 0.0
                prev_date = None
                prev_hour = None

                for state in sorted_yields:
                    if state.state in ("unavailable", "unknown", "none", None):
                        continue

                    try:
                        current_yield = float(state.state)
                        state_time = state.last_changed.astimezone(tz)
                        date_str = state_time.strftime("%Y-%m-%d")
                        hour = state_time.hour

                        # Detect day change (yield reset)
                        if prev_date and date_str != prev_date:
                            prev_yield = 0.0

                        # Calculate hourly delta
                        if prev_date == date_str and prev_hour == hour:
                            delta = max(0, current_yield - prev_yield)
                            if delta > 0 and delta < 5.0:  # Sanity check
                                hourly_production[date_str][hour] = delta

                        prev_yield = current_yield
                        prev_date = date_str
                        prev_hour = hour

                    except (ValueError, TypeError):
                        continue

                # Also calculate from cumulative differences within day
                daily_data = defaultdict(dict)
                for state in sorted_yields:
                    if state.state in ("unavailable", "unknown", "none", None):
                        continue
                    try:
                        state_time = state.last_changed.astimezone(tz)
                        date_str = state_time.strftime("%Y-%m-%d")
                        hour = state_time.hour
                        value = float(state.state)

                        key = f"{hour:02d}"
                        if key not in daily_data[date_str] or value > daily_data[date_str][key]:
                            daily_data[date_str][key] = value
                    except (ValueError, TypeError):
                        continue

                # Calculate hourly from cumulative within day
                for date_str, hours in daily_data.items():
                    sorted_hours = sorted(hours.keys())
                    for i, hour_str in enumerate(sorted_hours):
                        hour = int(hour_str)
                        current = hours[hour_str]
                        prev = hours[sorted_hours[i-1]] if i > 0 else 0
                        delta = max(0, current - prev)
                        if 0 < delta < 5.0:  # Sanity check
                            if delta > hourly_production[date_str][hour]:
                                hourly_production[date_str][hour] = delta

                daily_yield_hours = sum(len(h) for h in hourly_production.values())
                _LOGGER.info(f"  → Daily yield: {daily_yield_hours} hourly records from {len(hourly_production)} days")

            # ---------------------------------------------------------------
            # METHOD 2: Process CUMULATIVE yield sensor (never resets)
            # ---------------------------------------------------------------
            if cumulative_states:
                _LOGGER.info("  Processing cumulative yield sensor (total kWh)...")
                sorted_cumulative = sorted(cumulative_states, key=lambda s: s.last_changed)

                # Build hourly max values across ALL time
                hourly_cumulative = {}  # key = (date_str, hour) -> max cumulative value at that hour

                for state in sorted_cumulative:
                    if state.state in ("unavailable", "unknown", "none", None):
                        continue
                    try:
                        value = float(state.state)
                        if value < 0:
                            continue
                        state_time = state.last_changed.astimezone(tz)
                        date_str = state_time.strftime("%Y-%m-%d")
                        hour = state_time.hour
                        key = (date_str, hour)

                        # Keep max value for each hour
                        if key not in hourly_cumulative or value > hourly_cumulative[key]:
                            hourly_cumulative[key] = value
                    except (ValueError, TypeError):
                        continue

                # Sort by time and calculate deltas
                sorted_keys = sorted(hourly_cumulative.keys())
                _LOGGER.info(f"  → Found {len(sorted_keys)} hourly cumulative readings")

                prev_value = None
                cumulative_hours_added = 0

                for i, key in enumerate(sorted_keys):
                    date_str, hour = key
                    current_value = hourly_cumulative[key]

                    if prev_value is not None:
                        delta = current_value - prev_value
                        # Valid production: positive, not too large (< 5 kWh/hour for typical home systems)
                        if 0.001 < delta < 5.0:
                            # Only add if we don't already have a value from daily sensor
                            # OR if cumulative gives a larger (likely more accurate) value
                            existing = hourly_production[date_str].get(hour, 0)
                            if delta > existing:
                                hourly_production[date_str][hour] = delta
                                cumulative_hours_added += 1

                    prev_value = current_value

                _LOGGER.info(f"  → Cumulative sensor added/updated {cumulative_hours_added} hourly records")

            total_hours = sum(len(h) for h in hourly_production.values())
            _LOGGER.info(f"  TOTAL: {total_hours} hourly production records across {len(hourly_production)} days")

            # =================================================================
            # STEP 4: Fetch historical weather from Open-Meteo Archive
            # =================================================================
            _LOGGER.info("STEP 4/8: Fetching historical weather from Open-Meteo Archive API...")

            from ..data.data_open_meteo_client import OpenMeteoArchiveClient

            lat = self.hass.config.latitude
            lon = self.hass.config.longitude

            archive_client = OpenMeteoArchiveClient(lat, lon)

            start_date_str = (now - timedelta(days=days)).strftime("%Y-%m-%d")
            end_date_str = (now - timedelta(days=1)).strftime("%Y-%m-%d")  # Yesterday (archive has delay)

            historical_weather = await archive_client.get_historical_weather(
                start_date=start_date_str,
                end_date=end_date_str,
            )

            if not historical_weather:
                _LOGGER.warning("Could not fetch historical weather - continuing with local sensors only")
                historical_weather = []

            # Index weather by date+hour
            weather_index = {}
            for entry in historical_weather:
                date_str = entry.get("date")
                hour = entry.get("hour")
                if date_str and hour is not None:
                    weather_index[f"{date_str}_{hour:02d}"] = entry

            _LOGGER.info(f"Indexed {len(weather_index)} hours of historical weather")

            # =================================================================
            # STEP 5: Build astronomy cache for historical dates
            # =================================================================
            _LOGGER.info("STEP 5/8: Building astronomy cache for historical dates...")

            from ..astronomy.astronomy_cache import AstronomyCache

            astro_cache = AstronomyCache(
                data_dir=self.coordinator.data_manager.data_dir,
                data_manager=self.coordinator.data_manager
            )

            tz_str = str(self.hass.config.time_zone)
            elev = self.hass.config.elevation
            astro_cache.initialize_location(lat, lon, tz_str, elev)

            solar_capacity = self.coordinator.solar_capacity or 5.0

            # Rebuild cache including historical dates
            await astro_cache.rebuild_cache(
                system_capacity_kwp=solar_capacity,
                start_date=datetime.now(tz).date(),
                days_back=days,
                days_ahead=7
            )

            # Load astronomy data
            astronomy_path = self.coordinator.data_manager.data_dir / "stats" / "astronomy_cache.json"
            astronomy_data = {}
            if astronomy_path.exists():
                def _load_astro():
                    with open(astronomy_path, 'r') as f:
                        return json.load(f)
                data = await self.hass.async_add_executor_job(_load_astro)
                astronomy_data = data.get("days", {})

            _LOGGER.info(f"Astronomy cache built: {len(astronomy_data)} days")

            # =================================================================
            # STEP 6: Build training samples for GeometryLearner
            # =================================================================
            _LOGGER.info("STEP 6/8: Building training samples for GeometryLearner...")

            geometry_samples = []
            residual_samples = []

            for date_str, hours in hourly_production.items():
                if date_str not in astronomy_data:
                    continue

                day_astro = astronomy_data[date_str]

                for hour, production_kwh in hours.items():
                    if production_kwh <= 0.01:  # Skip negligible production
                        continue

                    hour_str = str(hour)
                    hour_astro = day_astro.get("hourly", {}).get(hour_str)
                    if not hour_astro:
                        continue

                    # Get weather for this hour
                    weather_key = f"{date_str}_{hour:02d}"
                    weather = weather_index.get(weather_key, {})

                    # Extract data
                    elevation = hour_astro.get("elevation_deg", 0)
                    azimuth = hour_astro.get("azimuth_deg", 180)
                    clear_sky_wm2 = hour_astro.get("clear_sky_solar_radiation_wm2", 0)

                    ghi = weather.get("ghi", 0) or 0
                    dni = weather.get("direct_radiation", 0) or 0
                    dhi = weather.get("diffuse_radiation", 0) or 0
                    cloud_cover = weather.get("cloud_cover", 50) or 50
                    temperature = weather.get("temperature", 15) or 15

                    # Build sample
                    sample = {
                        "timestamp": f"{date_str}T{hour:02d}:30:00",
                        "sun_elevation_deg": elevation,
                        "sun_azimuth_deg": azimuth,
                        "actual_power_kwh": production_kwh,
                        "ghi_wm2": ghi,
                        "dni_wm2": dni,
                        "dhi_wm2": dhi,
                        "ambient_temp_c": temperature,
                        "cloud_cover_percent": cloud_cover,
                        "theoretical_max_wm2": clear_sky_wm2,
                    }

                    geometry_samples.append(sample)

                    # Also build residual sample (needs different format)
                    residual_sample = {
                        "timestamp": f"{date_str}T{hour:02d}:30:00",
                        "actual_kwh": production_kwh,
                        "corrected_weather": {
                            "ghi": ghi,
                            "solar_radiation_wm2": ghi,
                            "direct_radiation": dni,
                            "diffuse_radiation": dhi,
                            "temperature": temperature,
                            "clouds": cloud_cover,
                            "humidity": weather.get("humidity", 50),
                        },
                        "astronomy": {
                            "elevation_deg": elevation,
                            "azimuth_deg": azimuth,
                            "clear_sky_solar_radiation_wm2": clear_sky_wm2,
                        },
                    }
                    residual_samples.append(residual_sample)

            _LOGGER.info(f"Built {len(geometry_samples)} training samples")

            # =================================================================
            # STEP 7: Train GeometryLearner
            # =================================================================
            _LOGGER.info("STEP 7/8: Training GeometryLearner...")

            from ..physics import GeometryLearner

            geometry_learner = GeometryLearner(
                data_path=self.coordinator.data_manager.data_dir,
                system_capacity_kwp=solar_capacity,
                skip_load=True,  # Avoid blocking event loop
            )
            await geometry_learner.async_load_state()  # Load state asynchronously

            geometry_result = await geometry_learner.bulk_add_historical_data(geometry_samples)

            _LOGGER.info(f"GeometryLearner result:")
            _LOGGER.info(f"  - Samples processed: {geometry_result['samples_processed']}")
            _LOGGER.info(f"  - Accepted (clear-sky): {geometry_result['accepted']}")
            _LOGGER.info(f"  - Rejected: {geometry_result['rejected']}")
            if geometry_result['optimization_ran']:
                estimate = geometry_result['current_estimate']
                _LOGGER.info(f"  - Learned tilt: {estimate['tilt_deg']:.1f}°")
                _LOGGER.info(f"  - Learned azimuth: {estimate['azimuth_deg']:.1f}°")
                _LOGGER.info(f"  - Confidence: {estimate['confidence']:.2%}")

            # =================================================================
            # STEP 8: Train ResidualTrainer
            # =================================================================
            _LOGGER.info("STEP 8/8: Training ResidualTrainer...")

            from ..ml.ml_residual_trainer import ResidualTrainer

            residual_trainer = ResidualTrainer(
                data_dir=self.coordinator.data_manager.data_dir,
                system_capacity_kwp=solar_capacity,
                skip_load=True,  # Avoid blocking event loop
            )
            await residual_trainer.async_load_state()  # Load state asynchronously

            if len(residual_samples) >= 10:
                success, accuracy, algo = await residual_trainer.train_residual_model(
                    training_records=residual_samples,
                    algorithm="auto",
                )
                _LOGGER.info(f"ResidualTrainer result:")
                _LOGGER.info(f"  - Success: {success}")
                _LOGGER.info(f"  - Algorithm: {algo}")
                _LOGGER.info(f"  - Accuracy: {accuracy:.3f}")
            else:
                _LOGGER.warning(f"Not enough samples for ResidualTrainer: {len(residual_samples)} < 10")

            # =================================================================
            # COMPLETE
            # =================================================================
            _LOGGER.info("=" * 80)
            _LOGGER.info("PHYSICS BOOTSTRAP COMPLETE!")
            _LOGGER.info(f"  - Days processed: {days}")
            _LOGGER.info(f"  - Total hourly samples: {len(geometry_samples)}")
            _LOGGER.info(f"  - Historical weather hours: {len(weather_index)}")
            _LOGGER.info(f"  - GeometryLearner samples: {geometry_result['total_data_points']}")
            if geometry_result.get('optimization_ran'):
                _LOGGER.info(f"  - Learned geometry: tilt={geometry_result['current_estimate']['tilt_deg']:.1f}°, "
                           f"azimuth={geometry_result['current_estimate']['azimuth_deg']:.1f}°")
            _LOGGER.info("=" * 80)

        except Exception as e:
            _LOGGER.error(f"ERROR in bootstrap_physics_from_history: {e}", exc_info=True)

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

            notify_service = call.data.get("notify_service", "notify")
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
