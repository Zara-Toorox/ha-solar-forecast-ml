# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

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
# V12.3: MultiWeatherBlender removed - using only WeatherExpertBlender
# All blending now goes through WeatherExpertBlender with 5 sources
from .data_weather_expert_blender import (
    WeatherExpertBlender,
    WeatherExpertLearner,
    CloudType,
)
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
        self.weather_expert_blender: Optional[WeatherExpertBlender] = None
        self.weather_expert_learner: Optional[WeatherExpertLearner] = None

        _LOGGER.info(
            "Weather Data Pipeline Manager initialized - "
            "5-Source Expert Blending (V12.3)"
        )

    async def async_setup(self) -> bool:
        """Setup weather data pipeline components - V12.3 with unified ExpertBlender."""
        try:
            _LOGGER.info("Setting up Weather Data Pipeline Manager (5-Source Expert Blending)...")

            latitude = self.hass.config.latitude
            longitude = self.hass.config.longitude

            # WeatherService for Open-Meteo raw data fetching
            self.weather_service = WeatherService(
                hass=self.hass,
                latitude=latitude,
                longitude=longitude,
                data_dir=self.data_dir,
                data_manager=self.data_manager,
            )
            _LOGGER.info(
                f"Weather Service initialized "
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

            # Get optional Pirate Weather API key from config
            pirate_weather_key = None
            if self.config_entry and self.config_entry.options:
                pirate_weather_key = self.config_entry.options.get("pirate_weather_api_key")

            # V12.3: Single unified ExpertBlender for ALL 5 weather sources
            # No more MultiWeatherBlender - all blending through ExpertBlender
            self.weather_expert_blender = WeatherExpertBlender(
                hass=self.hass,
                data_dir=self.data_dir,
                open_meteo_cache={},  # Will load from file cache
                wttr_cache={},  # Will load from file cache
                latitude=latitude,
                longitude=longitude,
                pirate_weather_api_key=pirate_weather_key,
            )
            await self.weather_expert_blender.async_init()

            self.weather_expert_learner = WeatherExpertLearner(
                hass=self.hass,
                data_dir=self.data_dir,
                blender=self.weather_expert_blender,
            )

            active_count = len([
                e for e in self.weather_expert_blender._experts
                if self.weather_expert_blender._is_expert_enabled(e)
            ])
            _LOGGER.info(
                f"Weather Expert Blender initialized with {active_count} active experts "
                f"(Open-Meteo, wttr.in, ECMWF Layers, Bright Sky/DWD, "
                f"Pirate Weather: {'enabled' if pirate_weather_key else 'disabled'})"
            )

            # WeatherForecastCorrector creates the SINGLE SOURCE OF TRUTH
            self.weather_corrector = WeatherForecastCorrector(
                hass=self.hass,
                data_manager=self.data_manager,
                expert_blender=self.weather_expert_blender,
            )
            await self.weather_corrector.async_init()
            _LOGGER.info(
                "Weather Corrector initialized with Expert Blender - "
                "weather_forecast_corrected.json is SINGLE SOURCE OF TRUTH"
            )

            _LOGGER.info("Weather Data Pipeline Manager setup complete (V12.3)")
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

            # NOTE: Mid-day refresh (12:15) REMOVED - it was useless since the forecast
            # is already locked at 00:25. Dynamic tasks in production_scheduled_tasks.py
            # now handle fresh weather data before the final morning forecast:
            # sunrise-45min: Weather Blend, sunrise-40min: Corrected Forecast, sunrise-30min: Final Forecast

            # NOTE: Daily precision calculation (23:30) moved to End-of-Day Workflow
            # in production_scheduled_tasks.py for better task coordination

            # NOTE: Weather source weight learning (23:30) REMOVED - now handled by
            # End-of-Day Workflow Step 12 in production_scheduled_tasks.py
            # This avoids duplicate learn_from_day() calls at 23:30

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

            # Also fetch Expert weather sources (Bright Sky, Pirate Weather)
            await self._fetch_expert_weather_sources()

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
                _LOGGER.info(f"Pipeline: Weather data needs refresh ({reason}) - fetching now...")

                # V12.3: Fetch all expert sources and create corrected forecast
                success = await self._refresh_all_weather_sources()

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
        """Scheduled task: Update weather forecast with 5-source blending (3x daily) @zara

        V12.3: Uses WeatherExpertBlender with 5 sources:
        - Open-Meteo (primary)
        - wttr.in (World Weather Online)
        - ECMWF Layers
        - Bright Sky (DWD ICON) - free API
        - Pirate Weather (NOAA GFS/HRRR) - if API key configured
        """
        try:
            _LOGGER.info("Pipeline: Scheduled weather update triggered (V12.3 - 5 sources)")

            # V12.3: Unified update - fetch all sources and create corrected forecast
            success = await self._refresh_all_weather_sources()

            if success:
                stats = self.weather_expert_blender.get_blend_stats() if self.weather_expert_blender else {}
                _LOGGER.info(
                    f"Pipeline: Weather update SUCCESS - "
                    f"5-source blending complete, corrected forecast updated"
                )
            else:
                _LOGGER.warning("Pipeline: Weather update failed - using cached data")

        except Exception as e:
            _LOGGER.error(f"Pipeline: Weather update error: {e}", exc_info=True)

    async def _fetch_expert_weather_sources(self):
        """Fetch data from all Expert Blender weather sources (Bright Sky, Pirate Weather) @zara

        Called during scheduled updates (3x daily) to keep file caches populated.
        Each source has its own file cache with 2-year retention.
        """
        if not self.weather_expert_blender:
            return

        fetch_results = []

        # Fetch Bright Sky (DWD ICON) - free, no API key needed
        try:
            bright_sky = self.weather_expert_blender._bright_sky_expert
            success = await bright_sky._fetch_forecast()
            if success:
                fetch_results.append("Bright Sky: OK")
                _LOGGER.debug("Pipeline: Bright Sky cache updated")
            else:
                error = bright_sky.get_last_error() or "unknown"
                fetch_results.append(f"Bright Sky: FAILED ({error})")
                _LOGGER.debug(f"Pipeline: Bright Sky fetch failed: {error}")
        except Exception as e:
            fetch_results.append(f"Bright Sky: ERROR ({e})")
            _LOGGER.debug(f"Pipeline: Bright Sky fetch error: {e}")

        # Fetch Pirate Weather (NOAA GFS/HRRR) - requires API key
        try:
            pirate = self.weather_expert_blender._pirate_weather_expert
            if pirate._enabled:
                success = await pirate._fetch_forecast()
                if success:
                    fetch_results.append("Pirate Weather: OK")
                    _LOGGER.debug("Pipeline: Pirate Weather cache updated")
                else:
                    error = pirate.get_last_error() or "unknown"
                    fetch_results.append(f"Pirate Weather: FAILED ({error})")
                    _LOGGER.debug(f"Pipeline: Pirate Weather fetch failed: {error}")
            else:
                fetch_results.append("Pirate Weather: disabled (no API key)")
        except Exception as e:
            fetch_results.append(f"Pirate Weather: ERROR ({e})")
            _LOGGER.debug(f"Pipeline: Pirate Weather fetch error: {e}")

        if fetch_results:
            _LOGGER.info(f"Pipeline: Expert weather sources: {', '.join(fetch_results)}")

    async def _refresh_all_weather_sources(self) -> bool:
        """Refresh all 5 weather sources and create corrected forecast. V12.3

        This is the unified weather update method that:
        1. Fetches Open-Meteo data via WeatherService
        2. Fetches all Expert sources (Bright Sky, Pirate Weather)
        3. Creates the corrected forecast (blending happens inside via get_blended_cloud_cover)

        The blending is performed per-hour inside create_corrected_forecast() which calls
        _get_weather_for_hour_async() -> expert_blender.get_blended_cloud_cover().
        This design ensures all 5 sources are blended correctly for each hour.

        Returns:
            True if weather data was successfully refreshed and corrected forecast created
        """
        try:
            _LOGGER.info("Pipeline: Refreshing all 5 weather sources...")

            # Step 1: Fetch Open-Meteo via WeatherService
            if self.weather_service:
                om_success = await self.weather_service.force_update()
                if om_success:
                    _LOGGER.debug("Pipeline: Open-Meteo fetch SUCCESS")
                else:
                    _LOGGER.warning("Pipeline: Open-Meteo fetch FAILED - using cached data")
            else:
                _LOGGER.warning("Pipeline: Weather service not initialized")

            # Step 2: Fetch all Expert sources (Bright Sky, Pirate Weather)
            await self._fetch_expert_weather_sources()

            # Step 3: Create corrected forecast (SINGLE SOURCE OF TRUTH)
            # NOTE: Blending happens INSIDE create_corrected_forecast() via
            # _get_weather_for_hour_async() which calls expert_blender.get_blended_cloud_cover()
            # This is the correct design - no separate blend_forecast() call needed
            if self.weather_corrector:
                success = await self.weather_corrector.create_corrected_forecast()
                if success:
                    _LOGGER.info(
                        "Pipeline: Corrected forecast updated - "
                        "weather_forecast_corrected.json is SINGLE SOURCE OF TRUTH"
                    )
                    return True
                else:
                    _LOGGER.warning("Pipeline: Corrected forecast creation failed")
                    return False
            else:
                _LOGGER.warning("Pipeline: Weather corrector not initialized")
                return False

        except Exception as e:
            _LOGGER.error(f"Pipeline: Weather refresh error: {e}", exc_info=True)
            return False

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

    # NOTE: _scheduled_weight_learning() REMOVED
    # Weather expert learning is now handled exclusively by End-of-Day Workflow
    # Step 12 in production_scheduled_tasks.py._run_weather_expert_learning()
    # This avoids duplicate learn_from_day() calls at 23:30

    async def update_weather_cache(self) -> bool:
        """Manually trigger 5-source weather update @zara

        V12.3: Uses WeatherExpertBlender with 5 sources (Open-Meteo, wttr.in,
        ECMWF Layers, Bright Sky/DWD, Pirate Weather) for blending.
        """
        try:
            _LOGGER.info("Pipeline: Manual 5-source weather update triggered")

            success = await self._refresh_all_weather_sources()

            if success:
                stats = self.weather_expert_blender.get_blend_stats() if self.weather_expert_blender else {}
                _LOGGER.info(
                    f"Pipeline: 5-source weather update SUCCESS - "
                    f"{stats.get('active_sources', 0)} sources active"
                )
                return True
            else:
                _LOGGER.warning("Pipeline: Weather update failed")
                return False

        except Exception as e:
            _LOGGER.error(f"Pipeline: Weather update error: {e}", exc_info=True)
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
        """Get pipeline status @zara - V12.3 with 5-source ExpertBlender"""
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
                "weather_expert_blender": self.weather_expert_blender is not None,
                "weather_expert_learner": self.weather_expert_learner is not None,
            },
        }

        # V12.3: Add ExpertBlender stats
        if self.weather_expert_blender:
            status["expert_blender"] = self.weather_expert_blender.get_blend_stats()

        return status
