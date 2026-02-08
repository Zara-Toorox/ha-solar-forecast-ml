# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML - ML Weather
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************
"""Weather entity for ML Weather integration."""

import logging
from typing import Any

from homeassistant.components.weather import (
    WeatherEntity,
    Forecast,
    WeatherEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    UnitOfLength,
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    ATTRIBUTION,
    DOMAIN,
    ML_WEATHER_COORDINATOR,
    CONF_NAME,
    ATTR_FORECAST_CLOUD_COVERAGE,
    ATTR_FORECAST_PRESSURE,
    ATTR_FORECAST_HUMIDITY,
    ATTR_FORECAST_SOLAR_RADIATION,
    ATTR_FORECAST_DIRECT_RADIATION,
    ATTR_FORECAST_DIFFUSE_RADIATION,
    WIND_SPEED_CONVERSION,
)
from .coordinator import MLWeatherCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up ML Weather weather entity from a config entry."""
    coordinator: MLWeatherCoordinator = hass.data[DOMAIN][entry.entry_id][ML_WEATHER_COORDINATOR]

    async_add_entities([MLWeatherEntity(coordinator, entry)], False)


class MLWeatherEntity(CoordinatorEntity[MLWeatherCoordinator], WeatherEntity):
    """Implementation of ML Weather weather entity."""

    _attr_has_entity_name = True
    _attr_attribution = ATTRIBUTION

    def __init__(
        self,
        coordinator: MLWeatherCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the weather entity."""
        super().__init__(coordinator)
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_weather"
        self._attr_name = entry.data.get(CONF_NAME, "ML Weather")

    @property
    def device_info(self) -> dict[str, Any]:
        """Return device information."""
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": self._attr_name,
            "manufacturer": "Solar Forecast ML",
            "model": "Multi-Expert Blended Weather",
            "sw_version": self.coordinator.data.get("version", "unknown") if self.coordinator.data else "unknown",
        }

    @property
    def supported_features(self) -> WeatherEntityFeature:
        """Return supported features."""
        return (
            WeatherEntityFeature.FORECAST_HOURLY |
            WeatherEntityFeature.FORECAST_DAILY
        )

    @property
    def condition(self) -> str | None:
        """Return the current condition."""
        current = self.coordinator.get_current_weather()
        return current.get("condition")

    @property
    def native_temperature(self) -> float | None:
        """Return the temperature."""
        current = self.coordinator.get_current_weather()
        return current.get("temperature")

    @property
    def native_temperature_unit(self) -> str:
        """Return the temperature unit."""
        return UnitOfTemperature.CELSIUS

    @property
    def native_pressure(self) -> float | None:
        """Return the pressure."""
        current = self.coordinator.get_current_weather()
        return current.get("pressure")

    @property
    def native_pressure_unit(self) -> str:
        """Return the pressure unit."""
        return UnitOfPressure.HPA

    @property
    def humidity(self) -> float | None:
        """Return the humidity."""
        current = self.coordinator.get_current_weather()
        return current.get("humidity")

    @property
    def native_wind_speed(self) -> float | None:
        """Return the wind speed."""
        current = self.coordinator.get_current_weather()
        wind = current.get("wind_speed")
        if wind is not None:
            # Convert m/s to km/h using constant
            return round(wind * WIND_SPEED_CONVERSION, 1)
        return None

    @property
    def native_wind_speed_unit(self) -> str:
        """Return the wind speed unit."""
        return UnitOfSpeed.KILOMETERS_PER_HOUR

    @property
    def native_precipitation_unit(self) -> str:
        """Return the precipitation unit."""
        return UnitOfLength.MILLIMETERS

    @property
    def cloud_coverage(self) -> float | None:
        """Return the cloud coverage."""
        current = self.coordinator.get_current_weather()
        return current.get("cloud_coverage")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional state attributes."""
        current = self.coordinator.get_current_weather()
        attrs = {}

        if current.get("cloud_coverage") is not None:
            attrs["cloud_coverage"] = current["cloud_coverage"]
        if current.get("cloud_cover_low") is not None:
            attrs["cloud_cover_low"] = current["cloud_cover_low"]
        if current.get("cloud_cover_mid") is not None:
            attrs["cloud_cover_mid"] = current["cloud_cover_mid"]
        if current.get("cloud_cover_high") is not None:
            attrs["cloud_cover_high"] = current["cloud_cover_high"]
        if current.get("solar_radiation") is not None:
            attrs["solar_radiation_wm2"] = current["solar_radiation"]
        if current.get("direct_radiation") is not None:
            attrs["direct_radiation_wm2"] = current["direct_radiation"]
        if current.get("diffuse_radiation") is not None:
            attrs["diffuse_radiation_wm2"] = current["diffuse_radiation"]

        # Add data version
        if self.coordinator.data:
            attrs["data_version"] = self.coordinator.data.get("version", "unknown")

        return attrs

    async def async_forecast_hourly(self) -> list[Forecast] | None:
        """Return the hourly forecast."""
        hourly = self.coordinator.get_hourly_forecast()

        forecasts = []
        for item in hourly:
            forecast: Forecast = {
                "datetime": item["datetime"],
                "native_temperature": item.get("temperature"),
                "condition": item.get("condition"),
                "native_precipitation": item.get("precipitation", 0),
                "native_wind_speed": round(item["wind_speed"] * WIND_SPEED_CONVERSION, 1) if item.get("wind_speed") else None,
                "humidity": item.get("humidity"),
            }

            # Add extra attributes
            if item.get("cloud_coverage") is not None:
                forecast[ATTR_FORECAST_CLOUD_COVERAGE] = item["cloud_coverage"]
            if item.get("pressure") is not None:
                forecast[ATTR_FORECAST_PRESSURE] = item["pressure"]
            if item.get("solar_radiation") is not None:
                forecast[ATTR_FORECAST_SOLAR_RADIATION] = item["solar_radiation"]
            if item.get("direct_radiation") is not None:
                forecast[ATTR_FORECAST_DIRECT_RADIATION] = item["direct_radiation"]
            if item.get("diffuse_radiation") is not None:
                forecast[ATTR_FORECAST_DIFFUSE_RADIATION] = item["diffuse_radiation"]

            forecasts.append(forecast)

        return forecasts

    async def async_forecast_daily(self) -> list[Forecast] | None:
        """Return the daily forecast."""
        daily = self.coordinator.get_daily_forecast()

        forecasts = []
        for item in daily:
            forecast: Forecast = {
                "datetime": item["datetime"],
                "native_temperature": item.get("temphigh"),
                "native_templow": item.get("templow"),
                "condition": item.get("condition"),
                "native_precipitation": item.get("precipitation", 0),
                "native_wind_speed": round(item["wind_speed"] * WIND_SPEED_CONVERSION, 1) if item.get("wind_speed") else None,
                "humidity": item.get("humidity"),
            }

            # Add extra attributes
            if item.get("cloud_coverage") is not None:
                forecast[ATTR_FORECAST_CLOUD_COVERAGE] = item["cloud_coverage"]
            if item.get("pressure") is not None:
                forecast[ATTR_FORECAST_PRESSURE] = item["pressure"]

            forecasts.append(forecast)

        return forecasts

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.async_write_ha_state()
