# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from typing import Any, Dict, Optional, Tuple

from homeassistant.helpers.update_coordinator import UpdateFailed

from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)

class CoordinatorUpdateHelpers:
    """Helper methods for coordinator data updates"""

    def __init__(self, coordinator: "SolarForecastMLCoordinator"):
        """Initialize helpers @zara"""
        self.coordinator = coordinator

    async def fetch_weather_data(self) -> Tuple[Optional[Dict], Optional[list]]:
        """Fetch current weather and hourly forecast @zara"""
        current_weather = None
        hourly_forecast = None

        if self.coordinator.weather_service:
            try:
                current_weather = await self.coordinator.weather_service.get_current_weather()

                hourly_forecast = (
                    await self.coordinator.weather_service.get_corrected_hourly_forecast()
                )
                self.coordinator._last_weather_update = dt_util.now()

                if not hourly_forecast or len(hourly_forecast) == 0:
                    _LOGGER.warning("Weather service returned no hourly forecast data")

            except Exception as e:
                _LOGGER.error(f"Error fetching weather data: {e}", exc_info=True)

        return current_weather, hourly_forecast

    async def generate_forecast(
        self,
        current_weather: Optional[Dict],
        hourly_forecast: Optional[list],
        external_sensors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate forecast using orchestrator"""
        forecast = await self.coordinator.forecast_orchestrator.orchestrate_forecast(
            current_weather=current_weather,
            hourly_forecast=hourly_forecast,
            external_sensors=external_sensors,
            ml_prediction_today=None,
            ml_prediction_tomorrow=None,
            correction_factor=self.coordinator.learned_correction_factor,
        )

        if not forecast:
            raise UpdateFailed("Forecast generation failed")

        hourly_count = len(forecast.get("hourly", []))
        _LOGGER.debug(
            f"Forecast data from orchestrator: "
            f"today={forecast.get('today', 'N/A')} kWh, "
            f"tomorrow={forecast.get('tomorrow', 'N/A')} kWh, "
            f"day_after={forecast.get('day_after_tomorrow', 'N/A')} kWh, "
            f"method={forecast.get('method', 'unknown')}, "
            f"hourly_entries={hourly_count}"
        )
        return forecast

    async def build_coordinator_result(
        self,
        forecast: Dict[str, Any],
        current_weather: Optional[Dict],
        external_sensors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build coordinator result dictionary (async to load diagnostic data safely)"""
        import asyncio

        # Load diagnostic data in executor to avoid blocking event loop
        loop = asyncio.get_running_loop()
        cached_patterns = await loop.run_in_executor(None, self._load_cached_patterns)
        cached_physics = await loop.run_in_executor(None, self._load_cached_physics)

        result = {
            "forecast_today": forecast.get("today"),
            "forecast_tomorrow": forecast.get("tomorrow"),
            "forecast_day_after_tomorrow": forecast.get("day_after_tomorrow"),
            "hourly_forecast": forecast.get("hourly", []) if self.coordinator.enable_hourly else [],
            "current_weather": current_weather,
            "external_sensors": external_sensors,

            "production_time": {
                "active": self.coordinator.production_time_calculator.is_active,
                "duration_seconds": self.coordinator.production_time_calculator.total_seconds,
                "start_time": self.coordinator.production_time_calculator.start_time,
                "end_time": self.coordinator.production_time_calculator.end_time,
            },
            "peak_today": {
                "power_w": getattr(self.coordinator, "_peak_power_today", 0.0),
                "at": getattr(self.coordinator, "_peak_time_today", None),
            },
            "yield_today": {
                "kwh": external_sensors.get("solar_yield_today"),
                "sensor": self.coordinator.solar_yield_today,
            },
            # Cached diagnostic data (loaded async via executor to avoid blocking)
            "_cached_patterns": cached_patterns,
            "_cached_physics": cached_physics,
        }
        return result

    def _load_cached_patterns(self) -> Dict[str, Any]:
        """Load AI seasonal data for diagnostic sensors (called from async context) @zara"""
        try:
            data_manager = getattr(self.coordinator, "data_manager", None)
            if not data_manager:
                return {}

            # Load seasonal.json from ai/ directory (replaces learned_patterns.json)
            seasonal_file = data_manager.data_dir / "ai" / "seasonal.json"
            if not seasonal_file.exists():
                return {}

            import json
            with open(seasonal_file, "r") as f:
                return json.load(f)
        except Exception as e:
            _LOGGER.debug(f"Error loading cached patterns: {e}")
            return {}

    def _load_cached_physics(self) -> Dict[str, Any]:
        """Load AI weights for diagnostic sensors (called from async context) @zara"""
        try:
            data_manager = getattr(self.coordinator, "data_manager", None)
            if not data_manager:
                return {}

            import json

            # Load learned_weights.json from ai/ directory (TinyLSTM weights)
            weights_file = data_manager.data_dir / "ai" / "learned_weights.json"
            if weights_file.exists():
                with open(weights_file, "r") as f:
                    data = json.load(f)
                    data["_source"] = "ai_weights"
                    return data

            return {}
        except Exception as e:
            _LOGGER.debug(f"Error loading cached physics: {e}")
            return {}

    async def save_forecasts(self, forecast_data: Dict[str, Any], hourly_forecast: list) -> None:
        """Save forecasts to storage @zara"""
        await self.coordinator._save_forecasts_to_storage(
            forecast_data={
                "today": forecast_data.get("today"),
                "tomorrow": forecast_data.get("tomorrow"),
                "day_after_tomorrow": forecast_data.get("day_after_tomorrow"),
                "best_hour": forecast_data.get("best_hour"),
                "best_hour_kwh": forecast_data.get("best_hour_kwh"),
            },
            hourly_forecast=hourly_forecast,
        )

    async def handle_startup_recovery(self) -> None:
        """Handle startup recovery for missing forecasts @zara

        V13.2: Added learning data freshness check and auto-restore from /share/
        """
        # V13.2: Check data freshness and restore from /share/ if needed
        await self._check_and_restore_learning_data()

        today_forecast = await self.coordinator.data_manager.get_current_day_forecast()
        now_local = dt_util.now()

        if not today_forecast or not today_forecast.get("forecast_day", {}).get("locked"):
            if now_local.hour < 12:
                _LOGGER.info(
                    "System started without locked forecast (before 12:00) - initiating recovery"
                )
                await self.coordinator._recovery_forecast_process(source="startup_recovery")
            else:
                _LOGGER.info(
                    "System started late without forecast (after 12:00) - "
                    "using current forecast (NOT morning baseline!)"
                )
                if self.coordinator.data and "forecast_today" in self.coordinator.data:
                    forecast_value = self.coordinator.data.get("forecast_today")
                    await self.coordinator.data_manager.save_daily_forecast(
                        prediction_kwh=forecast_value,
                        source=f"late_startup_{now_local.hour:02d}:{now_local.minute:02d}",
                    )
                    _LOGGER.info(
                        f"Set forecast to current value: {forecast_value:.2f} kWh "
                        f"(not representative of morning prediction)"
                    )

    async def _check_and_restore_learning_data(self) -> None:
        """V13.2: Check learning data freshness and restore from /share/ if stale.

        This protects against learning data loss when user restores an old HA backup.
        The /share/ directory is not overwritten during standard HA backup restores.

        @zara
        """
        # Check if feature is enabled and manager exists
        if not getattr(self.coordinator, "share_backup_manager", None):
            return

        if not getattr(self.coordinator, "_learning_backup_enabled", False):
            return

        try:
            share_backup_manager = self.coordinator.share_backup_manager

            # Check if local data is fresh
            is_fresh, age_hours = await share_backup_manager.check_data_freshness()

            if is_fresh:
                _LOGGER.debug("Learning data is fresh - no restore needed")
                return

            # Data is stale - attempt restore from /share/
            _LOGGER.warning(
                f"Learning data appears stale ({age_hours}h old) - "
                "possible backup restore detected. Attempting recovery from /share/..."
            )

            restored = await share_backup_manager.restore_from_share()

            if restored:
                _LOGGER.info(
                    "Learning data successfully restored from /share/ backup"
                )
                await self._notify_learning_data_restored(age_hours)
            else:
                _LOGGER.warning(
                    "Could not restore learning data from /share/ - "
                    "no valid backup found or restore failed"
                )

        except Exception as e:
            _LOGGER.error(f"Error during learning data freshness check: {e}")

    async def _notify_learning_data_restored(self, age_hours: int) -> None:
        """Send notification to user about successful learning data restore.

        @zara
        """
        try:
            await self.coordinator.hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": "SFML: Lerndaten wiederhergestellt",
                    "message": (
                        f"Nach einem Backup-Restore wurden veraltete Lerndaten "
                        f"({age_hours} Stunden alt) erkannt.\n\n"
                        "Die Lerndaten wurden automatisch aus dem geschuetzten "
                        "Speicher (/share/) wiederhergestellt.\n\n"
                        "Ihr Lernfortschritt ist erhalten geblieben!"
                    ),
                    "notification_id": "solar_forecast_ml_learning_data_restored",
                },
            )
        except Exception as e:
            _LOGGER.warning(f"Could not send restore notification: {e}")
