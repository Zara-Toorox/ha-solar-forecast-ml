# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall

from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)

class AstronomyServiceHandler:
    """Handle astronomy cache services"""

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, coordinator: "SolarForecastMLCoordinator"
    ):
        """Initialize astronomy service handler"""
        self.hass = hass
        self.entry = entry
        self.coordinator = coordinator

        self.astronomy_cache = None
        self.max_peak_tracker = None

    async def initialize(self):
        """Initialize astronomy cache and max peak tracker @zara"""
        from ..astronomy import AstronomyCache, MaxPeakTracker
        from ..astronomy.astronomy_cache_manager import get_cache_manager

        self.astronomy_cache = AstronomyCache(
            self.coordinator.data_manager.data_dir, self.coordinator.data_manager
        )

        latitude = self.hass.config.latitude
        longitude = self.hass.config.longitude
        timezone_str = str(self.hass.config.time_zone)
        elevation_m = self.hass.config.elevation or 0

        self.astronomy_cache.initialize_location(latitude, longitude, timezone_str, elevation_m)

        # Set panel groups for per-group theoretical max calculations @zara
        panel_groups = getattr(self.coordinator, 'panel_groups', [])
        if panel_groups:
            self.astronomy_cache.set_panel_groups(panel_groups)
            _LOGGER.info(
                f"AstronomyCache: Panel groups configured ({len(panel_groups)} groups)"
            )

        self.max_peak_tracker = MaxPeakTracker(self.astronomy_cache)

        _LOGGER.info(
            f"Astronomy services initialized: lat={latitude}, lon={longitude}, "
            f"tz={timezone_str}, elev={elevation_m}m"
        )

        await self._auto_build_cache_if_needed()

        cache_manager = get_cache_manager()
        cache_file = self.astronomy_cache.cache_file
        if cache_file.exists():

            def _load_sync():
                return cache_manager.initialize(cache_file)

            import asyncio

            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(None, _load_sync)
            if success:
                _LOGGER.info("✅ Astronomy cache loaded into memory for fast access")
            else:
                _LOGGER.warning("Failed to load astronomy cache into memory")

    async def handle_build_astronomy_cache(self, call: ServiceCall) -> None:
        """Service: Build astronomy cache for date range @zara"""
        if not self.astronomy_cache:
            await self.initialize()

        days_back = call.data.get("days_back", 30)
        days_ahead = call.data.get("days_ahead", 7)

        system_capacity_kwp = self.coordinator.solar_capacity
        if not system_capacity_kwp:
            _LOGGER.error("Solar capacity not configured in config flow!")
            return

        _LOGGER.info(
            f"Building astronomy cache: {days_back} days back, "
            f"{days_ahead} days ahead, capacity={system_capacity_kwp} kWp"
        )

        try:
            result = await self.astronomy_cache.rebuild_cache(
                system_capacity_kwp=system_capacity_kwp,
                start_date=None,
                days_back=days_back,
                days_ahead=days_ahead,
            )

            _LOGGER.info(
                f"Astronomy cache built successfully: "
                f"{result['success_count']} days, {result['error_count']} errors"
            )

            from ..astronomy.astronomy_cache_manager import get_cache_manager

            cache_manager = get_cache_manager()
            cache_file = self.coordinator.data_manager.data_dir / "stats" / "astronomy_cache.json"

            def _reinit_sync():
                return cache_manager.initialize(cache_file)

            import asyncio

            loop = asyncio.get_running_loop()
            reinit_success = await loop.run_in_executor(None, _reinit_sync)
            if reinit_success:
                _LOGGER.info("✅ Astronomy cache re-initialized from updated file")

            if (
                hasattr(self.coordinator, "notification_service")
                and self.coordinator.notification_service
            ):
                await self.coordinator.notification_service.create_notification(
                    title="Astronomy Cache Built",
                    message=(
                        f"Successfully built astronomy cache for {result['success_count']} days "
                        f"({days_back} days back + {days_ahead} days ahead). "
                        f"Cache file: {result['cache_file']}"
                    ),
                    notification_id="astronomy_cache_built",
                )

        except Exception as e:
            _LOGGER.error(f"Error building astronomy cache: {e}", exc_info=True)
            if (
                hasattr(self.coordinator, "notification_service")
                and self.coordinator.notification_service
            ):
                await self.coordinator.notification_service.create_notification(
                    title="Astronomy Cache Error",
                    message=f"Failed to build astronomy cache: {str(e)}",
                    notification_id="astronomy_cache_error",
                )

    async def handle_extract_max_peaks(self, call: ServiceCall) -> None:
        """Service: Extract max peak records from hourly_predictions.json history @zara"""
        if not self.max_peak_tracker:
            await self.initialize()

        _LOGGER.info("Extracting max peak records from history...")

        try:
            hourly_predictions_file = (
                self.coordinator.data_manager.data_dir / "stats" / "hourly_predictions.json"
            )

            result = await self.max_peak_tracker.extract_max_peaks_from_history(
                hourly_predictions_file
            )

            if "error" in result:
                _LOGGER.error(f"Failed to extract max peaks: {result['error']}")
                return

            _LOGGER.info(
                f"Max peaks extracted: {result['processed_samples']} samples, "
                f"{result['updated_hours']} hours updated, "
                f"global max: {result['global_max']['kwh']} kWh at hour {result['global_max']['hour']}"
            )

            if (
                hasattr(self.coordinator, "notification_service")
                and self.coordinator.notification_service
            ):
                await self.coordinator.notification_service.create_notification(
                    title="Max Peak Records Extracted",
                    message=(
                        f"Processed {result['processed_samples']} samples from history. "
                        f"Updated records for {result['updated_hours']} hours. "
                        f"Global max: {result['global_max']['kwh']} kWh at hour {result['global_max']['hour']}."
                    ),
                    notification_id="max_peaks_extracted",
                )

        except Exception as e:
            _LOGGER.error(f"Error extracting max peaks: {e}", exc_info=True)
            if (
                hasattr(self.coordinator, "notification_service")
                and self.coordinator.notification_service
            ):
                await self.coordinator.notification_service.create_notification(
                    title="Max Peak Extraction Error",
                    message=f"Failed to extract max peaks: {str(e)}",
                    notification_id="max_peaks_error",
                )

    async def handle_refresh_cache_today(self, call: ServiceCall) -> None:
        """Service: Refresh astronomy cache for today + next 7 days @zara"""
        if not self.astronomy_cache:
            await self.initialize()

        _LOGGER.info("Refreshing astronomy cache for today + next 7 days...")

        try:

            system_capacity_kwp = self.coordinator.solar_capacity
            if not system_capacity_kwp:
                _LOGGER.error("Solar capacity not configured in config flow!")
                return

            result = await self.astronomy_cache.rebuild_cache(
                system_capacity_kwp=system_capacity_kwp,
                start_date=dt_util.now().date(),
                days_back=0,
                days_ahead=7,
            )

            _LOGGER.info(f"Astronomy cache refreshed: {result['success_count']} days updated")

            from ..astronomy.astronomy_cache_manager import get_cache_manager

            cache_manager = get_cache_manager()
            cache_file = self.coordinator.data_manager.data_dir / "stats" / "astronomy_cache.json"

            def _reinit_sync():
                return cache_manager.initialize(cache_file)

            import asyncio

            loop = asyncio.get_running_loop()
            reinit_success = await loop.run_in_executor(None, _reinit_sync)
            if reinit_success:
                _LOGGER.debug("Astronomy cache re-initialized from updated file after refresh")

        except Exception as e:
            _LOGGER.error(f"Error refreshing astronomy cache: {e}", exc_info=True)

    async def _auto_build_cache_if_needed(self) -> None:
        """Auto-build cache on first startup if it doesn't exist @zara"""
        cache_file = self.astronomy_cache.cache_file

        if cache_file.exists():
            _LOGGER.info("Astronomy cache already exists, skipping auto-build")

            await self._auto_extract_max_peaks_if_needed()
            return

        _LOGGER.info("Astronomy cache not found - auto-building for 30 days...")

        try:
            system_capacity_kwp = self.coordinator.solar_capacity
            if not system_capacity_kwp or system_capacity_kwp <= 0:
                _LOGGER.warning(
                    "⚠️  Solar capacity not configured - using DEFAULT 5.0 kWp for auto-build! "
                    "Configure 'solar_capacity' in integration settings, then rebuild astronomy cache "
                    "via Developer Tools → Services → 'solar_forecast_ml.rebuild_astronomy_cache' "
                    "for accurate predictions based on your system size."
                )
                system_capacity_kwp = 5.0
            else:
                _LOGGER.info(f"✅ Using configured solar capacity: {system_capacity_kwp} kWp")

            result = await self.astronomy_cache.rebuild_cache(
                system_capacity_kwp=system_capacity_kwp, start_date=None, days_back=30, days_ahead=7
            )

            _LOGGER.info(
                f"✅ Astronomy cache auto-built: {result['success_count']} days, "
                f"{result['error_count']} errors"
            )

            await self._auto_extract_max_peaks_if_needed()

            if (
                hasattr(self.coordinator, "notification_service")
                and self.coordinator.notification_service
            ):
                await self.coordinator.notification_service.create_notification(
                    title="Astronomy Cache Ready",
                    message=(
                        f"Astronomy cache automatically built for {result['success_count']} days. "
                        f"The system is now fully operational and independent of sun.sun entity."
                    ),
                    notification_id="astronomy_cache_auto_built",
                )

        except Exception as e:
            _LOGGER.error(f"Failed to auto-build astronomy cache: {e}", exc_info=True)

    async def _auto_extract_max_peaks_if_needed(self) -> None:
        """Auto-extract max peaks if not in cache @zara"""
        try:

            def _check_sync():
                try:
                    if not self.astronomy_cache.cache_file.exists():
                        return False

                    import json

                    with open(self.astronomy_cache.cache_file, "r") as f:
                        cache = json.load(f)

                    hourly_max_peaks = cache.get("pv_system", {}).get("hourly_max_peaks", {})
                    if not hourly_max_peaks:
                        return False

                    for hour_data in hourly_max_peaks.values():
                        if hour_data.get("kwh", 0) > 0:
                            return True

                    return False

                except Exception:
                    return False

            import asyncio

            loop = asyncio.get_running_loop()
            has_max_peaks = await loop.run_in_executor(None, _check_sync)

            if has_max_peaks:
                _LOGGER.info("Max peaks already in cache, skipping auto-extraction")
                return

            _LOGGER.info("Max peaks not found - auto-extracting from history...")

            hourly_predictions_file = (
                self.coordinator.data_manager.data_dir / "stats" / "hourly_predictions.json"
            )

            if not hourly_predictions_file.exists():
                _LOGGER.info(
                    "No hourly_predictions.json found - max peaks will be populated as data is collected"
                )
                return

            result = await self.max_peak_tracker.extract_max_peaks_from_history(
                hourly_predictions_file
            )

            if "error" not in result:
                _LOGGER.info(
                    f"✅ Max peaks auto-extracted: {result['processed_samples']} samples, "
                    f"global max: {result['global_max']['kwh']} kWh"
                )

            # Note: all_time_peak import to astronomy_cache.json was removed because:
            # 1. The data was never read from astronomy_cache.json (dead code)
            # 2. all_time_peak lives in daily_forecasts.json (Single Source of Truth)
            # 3. astronomy_cache.json uses hourly_max_peaks for Clear-Sky calculations

        except Exception as e:
            _LOGGER.error(f"Failed to auto-extract max peaks: {e}", exc_info=True)
