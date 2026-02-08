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
Astronomy service handler for Solar Forecast ML.
Manages astronomy cache building and max peak extraction.
Uses DatabaseManager for all persistent operations.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall

from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..data.db_manager import DatabaseManager

_LOGGER = logging.getLogger(__name__)


class AstronomyServiceHandler:
    """Handle astronomy cache services using database. @zara"""

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, coordinator: "SolarForecastMLCoordinator"
    ):
        """Initialize astronomy service handler. @zara"""
        self.hass = hass
        self.entry = entry
        self.coordinator = coordinator

        self.astronomy_cache = None
        self.max_peak_tracker = None
        self._db: Optional[DatabaseManager] = None

    @property
    def db_manager(self) -> Optional[DatabaseManager]:
        """Get database manager from coordinator. @zara V16.1 fix"""
        data_manager = getattr(self.coordinator, "data_manager", None)
        if data_manager:
            return getattr(data_manager, "_db_manager", None)
        return None

    async def initialize(self):
        """Initialize astronomy cache and max peak tracker. @zara"""
        from ..astronomy import AstronomyCache, MaxPeakTracker
        from ..astronomy.astronomy_cache_manager import get_cache_manager

        self.astronomy_cache = AstronomyCache(
            db_manager=self.coordinator.data_manager._db_manager
        )

        latitude = self.hass.config.latitude
        longitude = self.hass.config.longitude
        timezone_str = str(self.hass.config.time_zone)
        elevation_m = self.hass.config.elevation or 0

        self.astronomy_cache.initialize_location(latitude, longitude, timezone_str, elevation_m)

        # Set panel groups for per-group theoretical max calculations
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

        cache_manager = get_cache_manager(db_manager=self.coordinator.data_manager._db_manager)
        success = await cache_manager.initialize()
        if success:
            _LOGGER.info("Astronomy cache loaded into memory for fast access")
        else:
            _LOGGER.debug("Astronomy cache manager initialized (no cached data yet)")

    async def handle_build_astronomy_cache(self, call: ServiceCall) -> None:
        """Service: Build astronomy cache for date range. @zara"""
        if not self.astronomy_cache:
            await self.initialize()

        # Ensure panel groups are set (might have been empty at startup) @zara V16.1
        panel_groups = getattr(self.coordinator, 'panel_groups', [])
        if panel_groups and self.astronomy_cache:
            self.astronomy_cache.set_panel_groups(panel_groups)
            _LOGGER.info(f"Panel groups set for cache rebuild: {len(panel_groups)} groups")

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

            # Reinitialize cache manager from database
            from ..astronomy.astronomy_cache_manager import get_cache_manager

            cache_manager = get_cache_manager(db_manager=self.coordinator.data_manager._db_manager)
            reinit_success = await cache_manager.initialize()
            if reinit_success:
                _LOGGER.info("Astronomy cache manager re-initialized from database")

            if (
                hasattr(self.coordinator, "notification_service")
                and self.coordinator.notification_service
            ):
                await self.coordinator.notification_service.create_notification(
                    title="Astronomy Cache Built",
                    message=(
                        f"Successfully built astronomy cache for {result['success_count']} days "
                        f"({days_back} days back + {days_ahead} days ahead)."
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
        """Service: Extract max peak records from hourly predictions. @zara"""
        if not self.max_peak_tracker:
            await self.initialize()

        _LOGGER.info("Extracting max peak records from DB history...")

        try:
            db = self.db_manager
            if not db:
                _LOGGER.error("Database manager not available for max peak extraction")
                return

            # Get hourly predictions from DB
            rows = await db.fetchall(
                """SELECT target_hour, actual_kwh, target_date
                   FROM hourly_predictions
                   WHERE actual_kwh IS NOT NULL AND actual_kwh > 0
                   ORDER BY target_hour, actual_kwh DESC"""
            )

            if not rows:
                _LOGGER.info("No hourly predictions with actual data in database")
                return

            # Process to find max per hour
            max_per_hour = {}
            for row in rows:
                hour = row[0]
                kwh = row[1]
                date_str = str(row[2])

                if hour not in max_per_hour or kwh > max_per_hour[hour]["kwh"]:
                    max_per_hour[hour] = {"kwh": kwh, "date": date_str}

            # Update astronomy hourly peaks in DB
            for hour, data in max_per_hour.items():
                await db.execute(
                    """INSERT INTO astronomy_hourly_peaks (hour, kwh, date)
                       VALUES (?, ?, ?)
                       ON CONFLICT(hour) DO UPDATE SET
                           kwh = excluded.kwh,
                           date = excluded.date
                       WHERE excluded.kwh > astronomy_hourly_peaks.kwh""",
                    (hour, data["kwh"], data["date"])
                )

            # Find global max
            global_max = max(max_per_hour.items(), key=lambda x: x[1]["kwh"])

            _LOGGER.info(
                f"Max peaks extracted: {len(rows)} samples processed, "
                f"{len(max_per_hour)} hours updated, "
                f"global max: {global_max[1]['kwh']:.2f} kWh at hour {global_max[0]}"
            )

            if (
                hasattr(self.coordinator, "notification_service")
                and self.coordinator.notification_service
            ):
                await self.coordinator.notification_service.create_notification(
                    title="Max Peak Records Extracted",
                    message=(
                        f"Processed {len(rows)} samples from history. "
                        f"Updated records for {len(max_per_hour)} hours. "
                        f"Global max: {global_max[1]['kwh']:.2f} kWh at hour {global_max[0]}."
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
        """Service: Refresh astronomy cache for today + next 7 days. @zara"""
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

            cache_manager = get_cache_manager(db_manager=self.coordinator.data_manager._db_manager)
            reinit_success = await cache_manager.initialize()
            if reinit_success:
                _LOGGER.debug("Astronomy cache manager re-initialized after refresh")

        except Exception as e:
            _LOGGER.error(f"Error refreshing astronomy cache: {e}", exc_info=True)

    async def _auto_build_cache_if_needed(self) -> None:
        """Auto-build cache on first startup if it doesn't exist. @zara"""
        from datetime import date

        today_data = await self.astronomy_cache.get_day_data(date.today())

        if today_data is not None:
            _LOGGER.info("Astronomy cache already exists in database, skipping auto-build")
            await self._auto_extract_max_peaks_if_needed()
            return

        _LOGGER.info("Astronomy cache not found in database - auto-building for 30 days...")

        try:
            system_capacity_kwp = self.coordinator.solar_capacity
            if not system_capacity_kwp or system_capacity_kwp <= 0:
                _LOGGER.warning(
                    "Solar capacity not configured - using DEFAULT 5.0 kWp for auto-build! "
                    "Configure 'solar_capacity' in integration settings, then rebuild astronomy cache "
                    "via Developer Tools -> Services -> 'solar_forecast_ml.rebuild_astronomy_cache' "
                    "for accurate predictions based on your system size."
                )
                system_capacity_kwp = 5.0
            else:
                _LOGGER.info(f"Using configured solar capacity: {system_capacity_kwp} kWp")

            result = await self.astronomy_cache.rebuild_cache(
                system_capacity_kwp=system_capacity_kwp, start_date=None, days_back=30, days_ahead=7
            )

            _LOGGER.info(
                f"Astronomy cache auto-built: {result['success_count']} days, "
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
        """Auto-extract max peaks if not in DB. @zara"""
        try:
            db = self.db_manager
            if not db:
                return

            # Check if we have max peaks in DB
            row = await db.fetchone(
                "SELECT COUNT(*) FROM astronomy_hourly_peaks WHERE kwh > 0"
            )

            if row and row[0] > 0:
                _LOGGER.info("Max peaks already in database, skipping auto-extraction")
                return

            _LOGGER.info("Max peaks not found - auto-extracting from history...")

            # Check if we have predictions to extract from
            count_row = await db.fetchone(
                "SELECT COUNT(*) FROM hourly_predictions WHERE actual_kwh IS NOT NULL AND actual_kwh > 0"
            )

            if not count_row or count_row[0] == 0:
                _LOGGER.info(
                    "No hourly predictions with actual data - max peaks will be populated as data is collected"
                )
                return

            # Create a fake service call for the extract handler
            class FakeServiceCall:
                data = {}

            await self.handle_extract_max_peaks(FakeServiceCall())

        except Exception as e:
            _LOGGER.error(f"Failed to auto-extract max peaks: {e}", exc_info=True)

    async def get_astronomy_data_for_date(self, target_date: date) -> Optional[dict]:
        """Get astronomy data for a specific date from cache. @zara"""
        try:
            if self.astronomy_cache:
                return await self.astronomy_cache.get_day_data(target_date)
            return None

        except Exception as e:
            _LOGGER.error(f"Error getting astronomy data for {target_date}: {e}")
            return None

    async def get_hourly_max_peaks(self) -> dict:
        """Get hourly max peaks from DB. @zara"""
        try:
            db = self.db_manager
            if not db:
                return {}

            rows = await db.fetchall(
                "SELECT hour, kwh, date FROM astronomy_hourly_peaks WHERE kwh > 0"
            )

            return {
                row[0]: {"kwh": row[1], "date": str(row[2])}
                for row in rows
            }

        except Exception as e:
            _LOGGER.error(f"Error getting hourly max peaks: {e}")
            return {}
