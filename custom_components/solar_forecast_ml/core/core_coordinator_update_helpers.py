"""
Coordinator Update Helpers - Extract methods from _async_update_data

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

import logging
from typing import Dict, Any, Optional, Tuple

from homeassistant.helpers.update_coordinator import UpdateFailed

from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class CoordinatorUpdateHelpers:
    """Helper methods for coordinator data updates"""

    def __init__(self, coordinator: "SolarForecastMLCoordinator"):
        """Initialize helpers"""
        self.coordinator = coordinator

    async def fetch_weather_data(self) -> Tuple[Optional[Dict], Optional[list]]:
        """Fetch current weather and hourly forecast"""
        current_weather = None
        hourly_forecast = None

        if self.coordinator.weather_service:
            try:
                current_weather = await self.coordinator.weather_service.get_current_weather()
                hourly_forecast = await self.coordinator.weather_service.get_processed_hourly_forecast()
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
        external_sensors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate forecast using orchestrator"""
        forecast = await self.coordinator.forecast_orchestrator.orchestrate_forecast(
            current_weather=current_weather,
            hourly_forecast=hourly_forecast,
            external_sensors=external_sensors,
            ml_prediction_today=None,
            ml_prediction_tomorrow=None,
            correction_factor=self.coordinator.learned_correction_factor
        )

        if not forecast:
            raise UpdateFailed("Forecast generation failed")

        # Log summary instead of full data (hourly array can be huge)
        hourly_count = len(forecast.get('hourly', []))
        _LOGGER.debug(
            f"Forecast data from orchestrator: "
            f"today={forecast.get('today', 'N/A')} kWh, "
            f"tomorrow={forecast.get('tomorrow', 'N/A')} kWh, "
            f"day_after={forecast.get('day_after_tomorrow', 'N/A')} kWh, "
            f"method={forecast.get('method', 'unknown')}, "
            f"hourly_entries={hourly_count}"
        )
        return forecast

    def build_coordinator_result(
        self,
        forecast: Dict[str, Any],
        current_weather: Optional[Dict],
        external_sensors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build coordinator result dictionary"""
        result = {
            "forecast_today": forecast.get("today"),
            "forecast_tomorrow": forecast.get("tomorrow"),
            "forecast_day_after_tomorrow": forecast.get("day_after_tomorrow"),
            "hourly_forecast": forecast.get("hourly", []) if self.coordinator.enable_hourly else [],
            "current_weather": current_weather,
            "external_sensors": external_sensors,
            # Production time data from tracker
            "production_time": {
                "active": self.coordinator.production_time_calculator.is_active,
                "duration_seconds": self.coordinator.production_time_calculator.total_seconds,
                "start_time": self.coordinator.production_time_calculator.start_time,
                "end_time": self.coordinator.production_time_calculator.end_time
            },
            "peak_today": {
                "power_w": getattr(self.coordinator, '_peak_power_today', 0.0),
                "at": getattr(self.coordinator, '_peak_time_today', None)
            },
            "yield_today": {
                "kwh": external_sensors.get("solar_yield_today"),
                "sensor": self.coordinator.solar_yield_today
            }
        }
        return result

    async def save_forecasts(
        self,
        forecast_data: Dict[str, Any],
        hourly_forecast: list
    ) -> None:
        """Save forecasts to storage"""
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

    async def update_electricity_prices(self) -> None:
        """Update electricity prices if battery management enabled"""
        if not self.coordinator.electricity_service:
            return

        try:
            should_update = False
            last_update = self.coordinator.electricity_service.get_last_update()

            if last_update is None:
                should_update = True
            else:
                # Update if last update was more than 12 hours ago
                hours_since_update = (dt_util.now() - last_update).total_seconds() / 3600
                should_update = hours_since_update > 12

            if should_update:
                _LOGGER.debug("Fetching electricity prices from ENTSO-E...")
                prices = await self.coordinator.electricity_service.fetch_day_ahead_prices()
                if prices:
                    _LOGGER.info(f"Electricity prices updated successfully: {len(prices.get('prices', []))} price points")
                else:
                    _LOGGER.warning("Failed to fetch electricity prices")

        except Exception as e:
            _LOGGER.error(f"Error updating electricity prices: {e}", exc_info=True)

    async def handle_startup_recovery(self) -> None:
        """Handle startup recovery for missing forecasts"""
        today_forecast = await self.coordinator.data_manager.get_current_day_forecast()
        now_local = dt_util.now()

        if not today_forecast or not today_forecast.get("forecast_day", {}).get("locked"):
            if now_local.hour < 12:
                _LOGGER.warning(
                    "System started without locked forecast (before 12:00) - "
                    "initiating recovery"
                )
                await self.coordinator._recovery_forecast_process(source="startup_recovery")
            else:
                _LOGGER.warning(
                    "System started late without forecast (after 12:00) - "
                    "using current forecast (NOT morning baseline!)"
                )
                if self.coordinator.data and "forecast_today" in self.coordinator.data:
                    forecast_value = self.coordinator.data.get("forecast_today")
                    await self.coordinator.data_manager.save_daily_forecast(
                        prediction_kwh=forecast_value,
                        source=f"late_startup_{now_local.hour:02d}:{now_local.minute:02d}"
                    )
                    _LOGGER.warning(
                        f"Set forecast to current value: {forecast_value:.2f} kWh "
                        f"(not representative of morning prediction)"
                    )

    async def check_weather_service_health(self) -> None:
        """Check and recover weather service if unhealthy"""
        if self.coordinator.weather_service and not self.coordinator.weather_service.get_health_status().get('healthy'):
            _LOGGER.warning("Weather service unhealthy, attempting recovery...")
            try:
                await self.coordinator.weather_service.force_update()
            except Exception as e:
                _LOGGER.error(f"Weather service recovery failed: {e}")
