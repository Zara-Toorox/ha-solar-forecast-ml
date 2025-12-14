"""Weather Data Pipeline Manager V12.0.0 @zara

Open-Meteo is the SINGLE DATA SOURCE for all weather data.
No HA Weather Entity required.

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
from typing import Optional, Dict, Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_change

from .data_weather_actual_tracker import WeatherActualTracker
from .data_weather_precision import WeatherPrecisionTracker
from .data_weather_corrector import WeatherForecastCorrector
from .data_multi_weather_client import MultiWeatherBlender, WeatherSourceLearner
from ..forecast.forecast_weather import WeatherService
from ..astronomy.astronomy_cache import AstronomyCache

_LOGGER = logging.getLogger(__name__)

class WeatherDataPipelineManager:
    """
    Central manager for weather data pipeline.

    Manages all weather data operations from raw forecasts to corrected forecasts.
    Runs background tasks for updates and ensures Single Source of Truth.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        data_dir: Path,
        stats_dir: Path,
        data_manager,
        astronomy_cache,
        config_entry,
        coordinator=None,
    ):
        """
        Initialize Weather Data Pipeline Manager.

        Args:
            hass: Home Assistant instance
            data_dir: Base data directory
            stats_dir: Stats directory
            data_manager: Data manager instance
            astronomy_cache: Astronomy cache instance
            config_entry: Config entry instance
            coordinator: Coordinator instance for accessing solar_capacity
        """
        self.hass = hass
        self.data_dir = data_dir
        self.stats_dir = stats_dir
        self.data_manager = data_manager
        self.astronomy_cache = astronomy_cache
        self.config_entry = config_entry
        self.coordinator = coordinator

        self._background_tasks = []
        self._listeners = []
        self._running = False

        self.weather_service: Optional[WeatherService] = None
        self.weather_actual_tracker: Optional[WeatherActualTracker] = None
        self.weather_precision: Optional[WeatherPrecisionTracker] = None
        self.weather_corrector: Optional[WeatherForecastCorrector] = None
        self.multi_weather_blender: Optional[MultiWeatherBlender] = None
        self.weather_source_learner: Optional[WeatherSourceLearner] = None

        _LOGGER.info(
            "Weather Data Pipeline Manager initialized - "
            "Multi-Source Weather (Open-Meteo + wttr.in with learned weights)"
        )

    async def async_setup(self) -> bool:
        """Setup weather data pipeline components @zara"""
        try:
            _LOGGER.info("Setting up Weather Data Pipeline Manager (Open-Meteo ONLY)...")

            latitude = self.hass.config.latitude
            longitude = self.hass.config.longitude

            self.weather_service = WeatherService(
                hass=self.hass,
                latitude=latitude,
                longitude=longitude,
                data_dir=self.data_dir,
                data_manager=self.data_manager,
            )
            _LOGGER.info(
                f"Weather Service initialized - Open-Meteo ONLY "
                f"(lat={latitude:.4f}, lon={longitude:.4f})"
            )

            self.weather_actual_tracker = WeatherActualTracker(
                hass=self.hass,
                data_dir=self.data_dir,
                config_entry=self.config_entry,
            )

            self.weather_precision = WeatherPrecisionTracker(
                hass=self.hass,
                config_entry=self.config_entry,
                data_dir=self.data_dir,
            )

            self.weather_corrector = WeatherForecastCorrector(
                hass=self.hass,
                data_manager=self.data_manager,
            )

            await self.weather_corrector.async_init()

            # Initialize Multi-Weather Blender (Open-Meteo + wttr.in)
            if self.weather_service and hasattr(self.weather_service, '_open_meteo'):
                self.multi_weather_blender = MultiWeatherBlender(
                    hass=self.hass,
                    latitude=latitude,
                    longitude=longitude,
                    data_dir=self.data_dir,
                    open_meteo_client=self.weather_service._open_meteo,
                )
                await self.multi_weather_blender.async_init()

                # CRITICAL: Set blender reference in WeatherService
                # This ensures all cache updates go through the blender
                # to preserve blend_info for weight learning
                self.weather_service.set_multi_weather_blender(self.multi_weather_blender)

                _LOGGER.info(
                    "Multi-Weather Blender initialized and linked to WeatherService - "
                    f"Weights: {self.multi_weather_blender.get_current_weights()}"
                )
            else:
                _LOGGER.warning(
                    "Multi-Weather Blender not initialized - "
                    "OpenMeteoClient not available"
                )

            # Initialize Weather Source Learner
            self.weather_source_learner = WeatherSourceLearner(
                hass=self.hass,
                data_dir=self.data_dir,
            )
            _LOGGER.info("Weather Source Learner initialized")

            _LOGGER.info("Weather Data Pipeline Manager setup complete")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to setup Weather Data Pipeline Manager: {e}", exc_info=True)
            return False

    async def start_pipeline(self) -> bool:
        """Start all background tasks for weather data pipeline @zara"""
        try:
            if self._running:
                _LOGGER.warning("Pipeline already running")
                return True

            _LOGGER.info("Starting Weather Data Pipeline...")

            self.hass.async_create_task(self._startup_open_meteo_refresh())

            self.hass.async_create_task(self._bootstrap_corrected_forecast())

            for hour in [6, 12, 18]:
                open_meteo_listener = async_track_time_change(
                    self.hass,
                    self._scheduled_open_meteo_update,
                    hour=hour,
                    minute=0,
                    second=0,
                )
                self._listeners.append(open_meteo_listener)
            _LOGGER.info("Scheduled: Open-Meteo update (3x daily: 06:00, 12:00, 18:00)")
            _LOGGER.info("Scheduled: Startup Open-Meteo refresh (1min delay after HA boot)")

            actual_listener = async_track_time_change(
                self.hass,
                self._scheduled_actual_tracking,
                minute=0,
                second=0,
            )
            self._listeners.append(actual_listener)
            _LOGGER.info("Scheduled: Hourly weather tracking (every hour at :00)")

            astronomy_listener = async_track_time_change(
                self.hass,
                self._scheduled_astronomy_update,
                hour=6,
                minute=0,
                second=0,
            )
            self._listeners.append(astronomy_listener)
            _LOGGER.info("Scheduled: Daily astronomy cache update (06:00 LOCAL)")

            # Pre-midnight refresh: Ensure fresh data before 00:15 corrected forecast
            pre_midnight_listener = async_track_time_change(
                self.hass,
                self._scheduled_pre_midnight_refresh,
                hour=0,
                minute=10,
                second=0,
            )
            self._listeners.append(pre_midnight_listener)
            _LOGGER.info("Scheduled: Pre-midnight data refresh (00:10 LOCAL - fallback before 00:15)")

            forecast_listener = async_track_time_change(
                self.hass,
                self._scheduled_corrected_forecast,
                hour=0,
                minute=15,
                second=0,
            )
            self._listeners.append(forecast_listener)
            _LOGGER.info("Scheduled: Daily corrected forecast (00:15 LOCAL - before morning routine at 00:25)")

            midday_forecast_listener = async_track_time_change(
                self.hass,
                self._scheduled_corrected_forecast,
                hour=12,
                minute=15,
                second=0,
            )
            self._listeners.append(midday_forecast_listener)
            _LOGGER.info("Scheduled: Mid-day corrected forecast refresh (12:15 LOCAL)")

            # NOTE: Daily precision calculation (23:30) moved to End-of-Day Workflow
            # in production_scheduled_tasks.py for better task coordination

            # Weather source weight learning (23:30) - learns from today's data
            weight_learning_listener = async_track_time_change(
                self.hass,
                self._scheduled_weight_learning,
                hour=23,
                minute=30,
                second=0,
            )
            self._listeners.append(weight_learning_listener)
            _LOGGER.info("Scheduled: Weather source weight learning (23:30 LOCAL)")

            self._running = True
            _LOGGER.info(f"Weather Data Pipeline started successfully ({len(self._listeners)} scheduled listeners)")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to start pipeline: {e}", exc_info=True)
            return False

    async def stop_pipeline(self):
        """Stop all background tasks and listeners. @zara"""
        try:
            _LOGGER.info("Stopping Weather Data Pipeline...")

            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            self._background_tasks.clear()

            for listener in self._listeners:
                listener()

            self._listeners.clear()

            self._running = False
            _LOGGER.info("Weather Data Pipeline stopped")

        except Exception as e:
            _LOGGER.error(f"Error stopping pipeline: {e}", exc_info=True)

    async def _startup_open_meteo_refresh(self):
        """One-time Open-Meteo refresh after HA restart @zara"""
        import asyncio

        try:
            _LOGGER.info("Pipeline: Startup Open-Meteo refresh scheduled (1min delay)")
            await asyncio.sleep(60)

            if self.weather_service:
                _LOGGER.info("Pipeline: Fetching weather data from Open-Meteo...")
                success = await self.weather_service.force_update()
                if success:
                    _LOGGER.info("Pipeline: Open-Meteo startup refresh SUCCESS")
                else:
                    _LOGGER.warning("Pipeline: Open-Meteo startup refresh FAILED - will retry in 2min")
                    await asyncio.sleep(120)
                    success = await self.weather_service.force_update()
                    if success:
                        _LOGGER.info("Pipeline: Open-Meteo retry SUCCESS")
                    else:
                        _LOGGER.warning("Pipeline: Open-Meteo unavailable - using cached data")

            await self._ensure_open_meteo_cache()

        except Exception as e:
            _LOGGER.error(f"Pipeline: Startup Open-Meteo refresh error: {e}", exc_info=True)

    async def _ensure_open_meteo_cache(self):
        """Ensure Open-Meteo cache is populated with fresh data @zara"""
        try:
            if not self.weather_corrector:
                _LOGGER.debug("Pipeline: Weather corrector not initialized - skipping Open-Meteo check")
                return

            open_meteo_file = self.data_manager.data_dir / "data" / "open_meteo_cache.json"
            needs_fetch = False
            reason = ""

            if not open_meteo_file.exists():
                needs_fetch = True
                reason = "file missing"
            else:
                try:
                    import json

                    def _read_cache():
                        with open(open_meteo_file, 'r') as f:
                            return json.load(f)

                    cache_data = await self.hass.async_add_executor_job(_read_cache)

                    forecast_data = cache_data.get("forecast", {})
                    metadata = cache_data.get("metadata", {})
                    hours_cached = metadata.get("hours_cached", 0)

                    if not forecast_data or hours_cached == 0:
                        needs_fetch = True
                        reason = "cache empty"
                    else:

                        fetched_at = metadata.get("fetched_at")
                        if fetched_at:
                            from datetime import datetime
                            try:
                                fetch_time = datetime.fromisoformat(fetched_at.replace('Z', '+00:00'))
                                age_hours = (datetime.now(fetch_time.tzinfo) - fetch_time).total_seconds() / 3600
                                if age_hours > 12:
                                    needs_fetch = True
                                    reason = f"cache stale ({age_hours:.1f}h old)"
                            except (ValueError, TypeError):
                                needs_fetch = True
                                reason = "invalid timestamp"
                        else:
                            needs_fetch = True
                            reason = "no timestamp"

                except (json.JSONDecodeError, IOError) as e:
                    needs_fetch = True
                    reason = f"read error: {e}"

            if needs_fetch:
                _LOGGER.info(f"Pipeline: Open-Meteo cache needs refresh ({reason}) - fetching now...")

                # Use MultiWeatherBlender if available (preserves blend_info)
                if self.multi_weather_blender:
                    success = await self.multi_weather_blender.update_and_save_cache()
                    if success:
                        _LOGGER.info(
                            "Pipeline: Open-Meteo cache refresh via Blender SUCCESS - "
                            "blend_info preserved for weight learning"
                        )
                    else:
                        _LOGGER.warning(
                            "Pipeline: Blender cache refresh FAILED - "
                            "falling back to direct Open-Meteo fetch"
                        )
                        # Fallback to direct fetch
                        success = await self.weather_corrector._fetch_open_meteo_forecast()
                else:
                    # No blender available, use direct fetch
                    success = await self.weather_corrector._fetch_open_meteo_forecast()

                if success:
                    _LOGGER.info(
                        "Pipeline: Open-Meteo cache refresh SUCCESS - "
                        "PRIMARY data source ready (direct GHI and cloud_cover values)"
                    )
                else:
                    _LOGGER.warning(
                        "Pipeline: Open-Meteo cache refresh FAILED - "
                        "System will use fallback data until next scheduled update (00:15 or 12:15)"
                    )
            else:
                _LOGGER.debug(f"Pipeline: Open-Meteo cache is current ({metadata.get('hours_cached', 0)} hours cached)")

        except Exception as e:
            _LOGGER.error(f"Pipeline: Open-Meteo cache check error: {e}", exc_info=True)

    async def _bootstrap_corrected_forecast(self):
        """One-time bootstrap for weather_forecast_corrected.json after fresh install @zara"""
        from homeassistant.helpers.event import async_call_later

        _LOGGER.info("Pipeline: Bootstrap corrected forecast scheduled (5min delay)")

        async_call_later(self.hass, 300, self._execute_bootstrap_corrected_forecast)

    async def _execute_bootstrap_corrected_forecast(self, _now=None):
        """Execute the bootstrap task (called after 5min delay) @zara"""
        try:
            _LOGGER.info("Pipeline: Bootstrap corrected forecast - executing now...")

            corrected_file = self.data_manager.weather_corrected_file
            should_bootstrap = False

            if not corrected_file.exists():
                _LOGGER.info("Pipeline: weather_forecast_corrected.json missing - BOOTSTRAP REQUIRED")
                should_bootstrap = True
            else:

                is_stale, stale_reason = await self._check_forecast_staleness(corrected_file)

                if is_stale:
                    _LOGGER.info(f"Pipeline: weather_forecast_corrected.json stale - {stale_reason} - BOOTSTRAP REQUIRED")
                    should_bootstrap = True
                else:
                    _LOGGER.info(
                        f"Pipeline: weather_forecast_corrected.json exists and is current - SKIP BOOTSTRAP"
                    )

            if should_bootstrap:
                _LOGGER.info("Pipeline: Bootstrap - Creating initial corrected forecast from Open-Meteo...")

                open_meteo_ok = await self._ensure_open_meteo_available()

                if not open_meteo_ok:
                    _LOGGER.warning(
                        "Pipeline: Bootstrap failed - Open-Meteo unavailable after retry. "
                        "System will retry during next morning routine (00:15)."
                    )
                    return

                if self.weather_corrector:

                    success = await self.weather_corrector.create_corrected_forecast(min_confidence=0.0)

                    if success:
                        _LOGGER.info(
                            "Pipeline: Bootstrap SUCCESS - weather_forecast_corrected.json created! "
                            "ML/RB strategies can now make predictions."
                        )
                    else:
                        _LOGGER.warning(
                            "Pipeline: Bootstrap could not create corrected forecast (this is normal for fresh installations). "
                            "System will automatically create it during the next morning routine (00:15). "
                            "Rule-Based forecasts are available in the meantime."
                        )
                else:
                    _LOGGER.error("Pipeline: Weather corrector not initialized - cannot bootstrap")

        except Exception as e:
            _LOGGER.error(f"Pipeline: Bootstrap corrected forecast error: {e}", exc_info=True)

    async def _check_forecast_staleness(self, corrected_file: Path) -> tuple[bool, str]:
        """Check if corrected forecast is stale based on forecast content @zara"""
        import json
        from datetime import datetime

        try:
            def _read_file():
                with open(corrected_file, "r") as f:
                    return json.load(f)

            data = await self.hass.async_add_executor_job(_read_file)

            forecast = data.get("forecast", {})
            if not forecast:
                return True, "no forecast data in file"

            now = datetime.now()
            latest_forecast_dt = None

            for date_str, hours in forecast.items():
                for hour_str, hour_data in hours.items():
                    try:

                        forecast_dt = datetime.strptime(f"{date_str} {hour_str}:00", "%Y-%m-%d %H:%M")
                        if latest_forecast_dt is None or forecast_dt > latest_forecast_dt:
                            latest_forecast_dt = forecast_dt
                    except ValueError:
                        continue

            if latest_forecast_dt is None:
                return True, "no valid forecast hours found"

            hours_ahead = (latest_forecast_dt - now).total_seconds() / 3600

            if hours_ahead < 12:
                return True, f"latest forecast only {hours_ahead:.1f}h ahead (need 12h+)"

            current_date = now.strftime("%Y-%m-%d")
            current_hour = str(now.hour)

            if current_date not in forecast:
                return True, f"no forecast for today ({current_date})"

            if current_hour not in forecast.get(current_date, {}):

                next_hour = str((now.hour + 1) % 24)
                if next_hour not in forecast.get(current_date, {}):
                    return True, f"no forecast for current hour ({current_hour}:00)"

            _LOGGER.debug(
                f"Pipeline: Forecast staleness check OK - latest forecast {hours_ahead:.1f}h ahead"
            )
            return False, ""

        except (json.JSONDecodeError, FileNotFoundError) as e:
            return True, f"file read error: {e}"
        except Exception as e:
            _LOGGER.warning(f"Pipeline: Staleness check error: {e}")

            import os
            file_mtime = os.path.getmtime(corrected_file)
            file_age_hours = (datetime.now() - datetime.fromtimestamp(file_mtime)).total_seconds() / 3600
            if file_age_hours > 24:
                return True, f"file is {file_age_hours:.1f}h old (fallback check)"
            return False, ""

    async def _ensure_open_meteo_available(self) -> bool:
        """Ensure Open-Meteo data is available for bootstrap @zara"""
        import json

        open_meteo_file = self.data_manager.data_dir / "data" / "open_meteo_cache.json"

        if not open_meteo_file.exists():
            _LOGGER.warning("Pipeline: open_meteo_cache.json missing - fetching from API...")
            return await self._trigger_open_meteo_refresh()

        try:
            def _read_cache():
                with open(open_meteo_file, "r") as f:
                    return json.load(f)

            cache_data = await self.hass.async_add_executor_job(_read_cache)

            forecast = cache_data.get("forecast", {})
            hours_cached = cache_data.get("metadata", {}).get("hours_cached", 0)

            if not forecast or hours_cached < 24:
                _LOGGER.warning(
                    f"Pipeline: open_meteo_cache.json has only {hours_cached} hours - "
                    f"fetching fresh data..."
                )
                return await self._trigger_open_meteo_refresh()

            _LOGGER.debug(f"Pipeline: open_meteo_cache.json OK ({hours_cached} hours available)")
            return True

        except (json.JSONDecodeError, Exception) as e:
            _LOGGER.warning(f"Pipeline: open_meteo_cache.json invalid ({e}) - fetching fresh data...")
            return await self._trigger_open_meteo_refresh()

    async def _trigger_open_meteo_refresh(self) -> bool:
        """Actively fetch Open-Meteo data for bootstrap @zara"""
        if not self.weather_service:
            _LOGGER.error("Pipeline: Weather service not initialized - cannot fetch Open-Meteo")
            return False

        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            _LOGGER.info(f"Pipeline: Bootstrap Open-Meteo fetch attempt {attempt}/{max_attempts}...")

            try:
                success = await self.weather_service.force_update()

                if success:
                    _LOGGER.info("Pipeline: Bootstrap Open-Meteo fetch SUCCESS")
                    return True
                else:
                    _LOGGER.warning("Pipeline: Bootstrap Open-Meteo fetch returned no data")

            except Exception as e:
                _LOGGER.warning(f"Pipeline: Bootstrap Open-Meteo fetch attempt {attempt} failed: {e}")

            if attempt < max_attempts:
                _LOGGER.info("Pipeline: Waiting 30s before retry...")
                await asyncio.sleep(30)

        _LOGGER.error("Pipeline: Bootstrap Open-Meteo fetch FAILED after all attempts")
        return False

    async def _scheduled_open_meteo_update(self, now):
        """Scheduled task: Update weather forecast with multi-source blending (3x daily) @zara

        Uses MultiWeatherBlender to:
        1. Fetch Open-Meteo data
        2. If cloud_cover > 50%, also fetch wttr.in (WWO)
        3. Blend cloud_cover using learned weights
        4. Save to cache for downstream components
        """
        try:
            _LOGGER.info("Pipeline: Multi-weather scheduled update triggered")

            # Try multi-weather blending first
            if self.multi_weather_blender:
                # Use update_and_save_cache to persist blended data
                success = await self.multi_weather_blender.update_and_save_cache()

                if success:
                    stats = self.multi_weather_blender.get_blend_stats()
                    blended_count = stats.get("blended_fetches", 0)

                    if blended_count > 0:
                        _LOGGER.info(
                            f"Pipeline: Multi-weather update SUCCESS - "
                            f"blended and saved to cache "
                            f"(weights: {self.multi_weather_blender.get_current_weights()})"
                        )
                    else:
                        _LOGGER.info(
                            f"Pipeline: Multi-weather update SUCCESS - "
                            f"Open-Meteo only (all cloud_cover <= 50%)"
                        )
                    return
                else:
                    _LOGGER.warning("Pipeline: Multi-weather blending failed - falling back to Open-Meteo only")

            # Fallback to original Open-Meteo only
            if self.weather_service:
                success = await self.weather_service.force_update()
                if success:
                    _LOGGER.info("Pipeline: Open-Meteo update SUCCESS (fallback)")
                else:
                    _LOGGER.warning("Pipeline: Open-Meteo update failed - using cached data")
            else:
                _LOGGER.warning("Pipeline: Weather service not initialized")

        except Exception as e:
            _LOGGER.error(f"Pipeline: Multi-weather update error: {e}", exc_info=True)

    async def _scheduled_actual_tracking(self, now):
        """Scheduled task: Track actual weather from local sensors every hour at :00 @zara"""
        try:
            _LOGGER.debug("Pipeline: Hourly weather tracking triggered")

            if self.weather_actual_tracker:
                await self.weather_actual_tracker.track_current_weather()
            else:
                _LOGGER.debug("Pipeline: Weather actual tracker not initialized")

        except Exception as e:
            _LOGGER.error(f"Pipeline: Hourly weather tracking error: {e}", exc_info=True)

    async def _scheduled_astronomy_update(self, now):
        """Scheduled task: Update astronomy cache daily at 06:00 LOCAL @zara"""
        try:
            _LOGGER.info("Pipeline: Daily astronomy cache update (06:00 LOCAL)")

            if self.astronomy_cache:
                system_capacity_kwp = self.coordinator.solar_capacity if self.coordinator else None
                if not system_capacity_kwp or system_capacity_kwp <= 0:
                    _LOGGER.error("Pipeline: Solar capacity not configured - cannot update astronomy cache")
                    return

                await self.astronomy_cache.rebuild_cache(system_capacity_kwp=system_capacity_kwp, days_ahead=7)
                _LOGGER.info("Pipeline: Astronomy cache updated (today + 7 days)")
            else:
                _LOGGER.error("Pipeline: Astronomy cache not initialized")

        except Exception as e:
            _LOGGER.error(f"Pipeline: Astronomy cache update error: {e}", exc_info=True)

    async def _scheduled_pre_midnight_refresh(self, now):
        """Scheduled task: Refresh Open-Meteo and Astronomy cache at 00:10 LOCAL @zara

        This is a fallback to ensure fresh data is available before the 00:15
        corrected forecast generation. Runs after midnight to get the new day's
        forecast data.
        """
        try:
            _LOGGER.info("Pipeline: Pre-midnight data refresh (00:10 LOCAL)")

            # Update Open-Meteo forecast
            if self.weather_service:
                success = await self.weather_service.force_update()
                if success:
                    _LOGGER.info("Pipeline: Open-Meteo refreshed (00:10 fallback)")
                else:
                    _LOGGER.warning("Pipeline: Open-Meteo refresh failed (00:10)")
            else:
                _LOGGER.warning("Pipeline: Weather service not initialized for 00:10 refresh")

            # Update Astronomy cache for new day
            if self.astronomy_cache:
                system_capacity_kwp = self.coordinator.solar_capacity if self.coordinator else None
                if system_capacity_kwp and system_capacity_kwp > 0:
                    await self.astronomy_cache.rebuild_cache(system_capacity_kwp=system_capacity_kwp, days_ahead=7)
                    _LOGGER.info("Pipeline: Astronomy cache refreshed (00:10 fallback)")
                else:
                    _LOGGER.warning("Pipeline: Solar capacity not configured for 00:10 astronomy refresh")
            else:
                _LOGGER.warning("Pipeline: Astronomy cache not initialized for 00:10 refresh")

        except Exception as e:
            _LOGGER.error(f"Pipeline: Pre-midnight refresh error: {e}", exc_info=True)

    async def _scheduled_corrected_forecast(self, now):
        """Scheduled task: Create corrected forecast daily at 00:15 LOCAL @zara"""
        try:
            _LOGGER.info("Pipeline: Daily corrected forecast generation (00:15 LOCAL)")

            if self.weather_corrector:
                success = await self.weather_corrector.create_corrected_forecast()
                if success:
                    _LOGGER.info("Pipeline: Corrected forecast created successfully")
                else:
                    _LOGGER.error("Pipeline: Corrected forecast creation failed")
            else:
                _LOGGER.error("Pipeline: Weather corrector not initialized")

        except Exception as e:
            _LOGGER.error(f"Pipeline: Corrected forecast generation error: {e}", exc_info=True)

    async def _scheduled_weight_learning(self, now):
        """Scheduled task: Learn weather source weights from today's data (23:30) @zara

        Compares forecasted cloud_cover with actual sensor readings to
        determine which weather source (Open-Meteo vs wttr.in) was more accurate.
        Updates blending weights accordingly.
        """
        try:
            _LOGGER.info("Pipeline: Weather source weight learning (23:30 LOCAL)")

            if not self.weather_source_learner:
                _LOGGER.debug("Pipeline: Weather source learner not initialized - skipping")
                return

            # Learn from today's data
            today = datetime.now().strftime("%Y-%m-%d")

            result = await self.weather_source_learner.learn_from_day(date_str=today)

            if result.get("success"):
                _LOGGER.info(
                    f"Pipeline: Weight learning SUCCESS for {today} - "
                    f"MAE: Open-Meteo={result.get('mae_open_meteo', 0):.1f}%, "
                    f"WWO={result.get('mae_wwo', 0):.1f}% - "
                    f"New weights: {result.get('new_weights', {})}"
                )

                # Reload weights in blender if available
                if self.multi_weather_blender:
                    await self.multi_weather_blender._load_weights()
                    _LOGGER.info(
                        f"Pipeline: Blender weights reloaded: "
                        f"{self.multi_weather_blender.get_current_weights()}"
                    )
            else:
                _LOGGER.debug(
                    f"Pipeline: Weight learning skipped for {today}: "
                    f"{result.get('reason', 'unknown')}"
                )

        except Exception as e:
            _LOGGER.error(f"Pipeline: Weight learning error: {e}", exc_info=True)

    async def _scheduled_precision_calculation(self, now):
        """DEPRECATED: Precision calculation moved to End-of-Day Workflow.

        This method is kept for backwards compatibility and manual service calls.
        The scheduled 23:30 task now runs in production_scheduled_tasks.py
        as part of the unified End-of-Day Workflow (Step 6).
        @zara
        """
        try:
            _LOGGER.info("Pipeline: Daily precision calculation (23:30 LOCAL)")

            if self.weather_precision and self.weather_actual_tracker and self.weather_corrector:

                yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

                success = await self.weather_precision.calculate_correction_factors_v2(
                    weather_actual_tracker=self.weather_actual_tracker,
                    weather_corrector=self.weather_corrector,
                    date_str=yesterday,
                )

                if success:
                    _LOGGER.info(f"Pipeline: Precision calculated for {yesterday}")
                else:
                    _LOGGER.info(
                        f"ℹ️ Pipeline: Precision calculation skipped for {yesterday}. "
                        "This is normal if no weather data was collected that day (new installation or restart)."
                    )
            else:
                _LOGGER.info(
                    "ℹ️ Pipeline: Precision components not yet initialized. "
                    "This will resolve automatically after the first complete day of data collection."
                )

        except Exception as e:
            _LOGGER.error(f"Pipeline: Precision calculation error: {e}", exc_info=True)

    async def update_weather_cache(self) -> bool:
        """Manually trigger multi-weather update (Open-Meteo + wttr.in blending) @zara

        Note: This method goes through the WeatherService.force_update() which
        automatically uses the MultiWeatherBlender when available.
        """
        try:
            _LOGGER.info("Pipeline: Manual multi-weather update triggered")

            if not self.weather_service:
                _LOGGER.error("Pipeline: Weather service not initialized")
                return False

            # force_update() automatically uses Blender when available
            success = await self.weather_service.force_update()

            if success:
                if self.multi_weather_blender:
                    _LOGGER.info(
                        f"Pipeline: Multi-weather update SUCCESS "
                        f"(weights: {self.multi_weather_blender.get_current_weights()})"
                    )
                else:
                    _LOGGER.info("Pipeline: Open-Meteo update SUCCESS")
                return True
            else:
                _LOGGER.warning("Pipeline: Weather update failed")
                return False

        except Exception as e:
            _LOGGER.error(f"Pipeline: Multi-weather update error: {e}", exc_info=True)
            return False

    async def update_astronomy_cache(self, days_ahead: int = 7) -> bool:
        """Manually trigger astronomy cache update @zara"""
        try:
            _LOGGER.info(f"Pipeline: Manual astronomy cache update triggered (today + {days_ahead} days)")

            if not self.astronomy_cache:
                _LOGGER.error("Pipeline: Astronomy cache not initialized")
                return False

            system_capacity_kwp = self.coordinator.solar_capacity if self.coordinator else None
            if not system_capacity_kwp or system_capacity_kwp <= 0:
                _LOGGER.error("Pipeline: Solar capacity not configured - cannot update astronomy cache")
                return False

            await self.astronomy_cache.rebuild_cache(system_capacity_kwp=system_capacity_kwp, days_ahead=days_ahead)
            _LOGGER.info("Pipeline: Astronomy cache updated successfully")
            return True

        except Exception as e:
            _LOGGER.error(f"Pipeline: Astronomy cache update error: {e}", exc_info=True)
            return False

    async def track_actual_weather(self) -> Dict[str, Any]:
        """Manually trigger actual weather tracking @zara"""
        try:
            _LOGGER.info("Pipeline: Manual actual weather tracking triggered")

            if not self.weather_actual_tracker:
                _LOGGER.warning("Pipeline: Weather actual tracker not initialized")
                return {}

            weather_data = await self.weather_actual_tracker.track_current_weather()
            return weather_data

        except Exception as e:
            _LOGGER.error(f"Pipeline: Actual weather tracking error: {e}", exc_info=True)
            return {}

    async def calculate_precision(self, date_str: str) -> bool:
        """Manually trigger precision calculation for specific date @zara"""
        try:
            _LOGGER.info(f"Pipeline: Manual precision calculation triggered for {date_str}")

            if not self.weather_precision or not self.weather_actual_tracker or not self.weather_corrector:
                _LOGGER.error("Pipeline: Precision components not initialized")
                return False

            success = await self.weather_precision.calculate_correction_factors_v2(
                weather_actual_tracker=self.weather_actual_tracker,
                weather_corrector=self.weather_corrector,
                date_str=date_str,
            )

            if success:
                _LOGGER.info(f"Pipeline: Precision calculated for {date_str}")
            else:
                _LOGGER.info(
                    f"ℹ️ Pipeline: Precision calculation skipped for {date_str}. "
                    "This is normal if no weather data was collected that day."
                )

            return success

        except Exception as e:
            _LOGGER.error(f"Pipeline: Precision calculation error: {e}", exc_info=True)
            return False

    async def create_corrected_forecast(self) -> bool:
        """Manually trigger corrected forecast generation @zara"""
        try:
            _LOGGER.info("Pipeline: Manual corrected forecast generation triggered")

            if not self.weather_corrector:
                _LOGGER.error("Pipeline: Weather corrector not initialized")
                return False

            success = await self.weather_corrector.create_corrected_forecast()

            if success:
                _LOGGER.info("Pipeline: Corrected forecast created successfully")
            else:
                _LOGGER.error("Pipeline: Corrected forecast creation failed")

            return success

        except Exception as e:
            _LOGGER.error(f"Pipeline: Corrected forecast generation error: {e}", exc_info=True)
            return False

    def is_running(self) -> bool:
        """Check if pipeline is running. @zara"""
        return self._running

    async def get_status(self) -> Dict[str, Any]:
        """Get pipeline status @zara"""
        status = {
            "running": self._running,
            "background_tasks": len([t for t in self._background_tasks if not t.done()]),
            "scheduled_listeners": len(self._listeners),
            "components": {
                "weather_service": self.weather_service is not None,
                "weather_actual_tracker": self.weather_actual_tracker is not None,
                "weather_precision": self.weather_precision is not None,
                "weather_corrector": self.weather_corrector is not None,
                "astronomy_cache": self.astronomy_cache is not None,
                "multi_weather_blender": self.multi_weather_blender is not None,
                "weather_source_learner": self.weather_source_learner is not None,
            },
        }

        # Add blender stats if available
        if self.multi_weather_blender:
            status["multi_weather"] = self.multi_weather_blender.get_blend_stats()

        return status
