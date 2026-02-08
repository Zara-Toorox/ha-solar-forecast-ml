# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML - ML Weather
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************
"""Sensor platform for ML Weather integration."""

import logging
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    PERCENTAGE,
    UnitOfEnergy,
    UnitOfIrradiance,
    UnitOfLength,
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfTemperature,
    UnitOfPrecipitationDepth,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    ML_WEATHER_COORDINATOR,
    CONF_NAME,
    SENSOR_TYPE_TEMPERATURE,
    SENSOR_TYPE_HUMIDITY,
    SENSOR_TYPE_PRESSURE,
    SENSOR_TYPE_WIND_SPEED,
    SENSOR_TYPE_CLOUD_COVERAGE,
    SENSOR_TYPE_PRECIPITATION,
    SENSOR_TYPE_SOLAR_RADIATION,
    SENSOR_TYPE_DIRECT_RADIATION,
    SENSOR_TYPE_DIFFUSE_RADIATION,
    SENSOR_TYPE_CLOUD_COVER_LOW,
    SENSOR_TYPE_CLOUD_COVER_MID,
    SENSOR_TYPE_CLOUD_COVER_HIGH,
    SENSOR_TYPE_VISIBILITY,
    SENSOR_TYPE_FOG_DETECTED,
    SENSOR_TYPE_FOG_TYPE,
    SENSOR_TYPE_PV_FORECAST_TODAY,
    SENSOR_TYPE_PV_FORECAST_TOMORROW,
    SENSOR_TYPE_PV_FORECAST_DAY_AFTER,
)
from .coordinator import MLWeatherCoordinator

_LOGGER = logging.getLogger(__name__)

# Key mapping for weather sensors - centralized to avoid duplication
WEATHER_SENSOR_KEY_MAP: dict[str, str] = {
    SENSOR_TYPE_TEMPERATURE: "temperature",
    SENSOR_TYPE_HUMIDITY: "humidity",
    SENSOR_TYPE_PRESSURE: "pressure",
    SENSOR_TYPE_WIND_SPEED: "wind_speed",
    SENSOR_TYPE_CLOUD_COVERAGE: "cloud_coverage",
    SENSOR_TYPE_PRECIPITATION: "precipitation",
    SENSOR_TYPE_SOLAR_RADIATION: "solar_radiation",
    SENSOR_TYPE_DIRECT_RADIATION: "direct_radiation",
    SENSOR_TYPE_DIFFUSE_RADIATION: "diffuse_radiation",
    SENSOR_TYPE_CLOUD_COVER_LOW: "cloud_cover_low",
    SENSOR_TYPE_CLOUD_COVER_MID: "cloud_cover_mid",
    SENSOR_TYPE_CLOUD_COVER_HIGH: "cloud_cover_high",
    SENSOR_TYPE_VISIBILITY: "visibility",
    SENSOR_TYPE_FOG_DETECTED: "fog_detected",
    SENSOR_TYPE_FOG_TYPE: "fog_type",
}

# Key mapping for PV forecast sensors
PV_FORECAST_KEY_MAP: dict[str, str] = {
    SENSOR_TYPE_PV_FORECAST_TODAY: "today",
    SENSOR_TYPE_PV_FORECAST_TOMORROW: "tomorrow",
    SENSOR_TYPE_PV_FORECAST_DAY_AFTER: "day_after_tomorrow",
}

# Sensors that may not always have data (optional fields)
OPTIONAL_SENSORS = frozenset([
    SENSOR_TYPE_CLOUD_COVER_LOW,
    SENSOR_TYPE_CLOUD_COVER_MID,
    SENSOR_TYPE_CLOUD_COVER_HIGH,
    SENSOR_TYPE_VISIBILITY,
    SENSOR_TYPE_FOG_DETECTED,
    SENSOR_TYPE_FOG_TYPE,
])


SENSOR_DESCRIPTIONS: tuple[SensorEntityDescription, ...] = (
    SensorEntityDescription(
        key=SENSOR_TYPE_TEMPERATURE,
        name="Temperature",
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:thermometer",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_HUMIDITY,
        name="Humidity",
        native_unit_of_measurement=PERCENTAGE,
        device_class=SensorDeviceClass.HUMIDITY,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:water-percent",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_PRESSURE,
        name="Pressure",
        native_unit_of_measurement=UnitOfPressure.HPA,
        device_class=SensorDeviceClass.ATMOSPHERIC_PRESSURE,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:gauge",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_WIND_SPEED,
        name="Wind Speed",
        native_unit_of_measurement=UnitOfSpeed.METERS_PER_SECOND,
        device_class=SensorDeviceClass.WIND_SPEED,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:weather-windy",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_CLOUD_COVERAGE,
        name="Cloud Coverage",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:cloud",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_PRECIPITATION,
        name="Precipitation",
        native_unit_of_measurement=UnitOfPrecipitationDepth.MILLIMETERS,
        device_class=SensorDeviceClass.PRECIPITATION,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:weather-rainy",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_SOLAR_RADIATION,
        name="Solar Radiation (GHI)",
        native_unit_of_measurement=UnitOfIrradiance.WATTS_PER_SQUARE_METER,
        device_class=SensorDeviceClass.IRRADIANCE,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:white-balance-sunny",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_DIRECT_RADIATION,
        name="Direct Radiation (DNI)",
        native_unit_of_measurement=UnitOfIrradiance.WATTS_PER_SQUARE_METER,
        device_class=SensorDeviceClass.IRRADIANCE,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:weather-sunny",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_DIFFUSE_RADIATION,
        name="Diffuse Radiation (DHI)",
        native_unit_of_measurement=UnitOfIrradiance.WATTS_PER_SQUARE_METER,
        device_class=SensorDeviceClass.IRRADIANCE,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:weather-partly-cloudy",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_CLOUD_COVER_LOW,
        name="Cloud Cover Low",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:cloud-outline",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_CLOUD_COVER_MID,
        name="Cloud Cover Mid",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:cloud",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_CLOUD_COVER_HIGH,
        name="Cloud Cover High",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:clouds",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_VISIBILITY,
        name="Visibility",
        native_unit_of_measurement=UnitOfLength.METERS,
        device_class=SensorDeviceClass.DISTANCE,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:eye",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_FOG_DETECTED,
        name="Fog Detected",
        icon="mdi:weather-fog",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_FOG_TYPE,
        name="Fog Type",
        icon="mdi:weather-fog",
    ),
)

# PV Forecast sensor descriptions
PV_FORECAST_DESCRIPTIONS: tuple[SensorEntityDescription, ...] = (
    SensorEntityDescription(
        key=SENSOR_TYPE_PV_FORECAST_TODAY,
        name="PV Forecast Today",
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL,
        icon="mdi:solar-power",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_PV_FORECAST_TOMORROW,
        name="PV Forecast Tomorrow",
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL,
        icon="mdi:solar-power",
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_PV_FORECAST_DAY_AFTER,
        name="PV Forecast Day After Tomorrow",
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL,
        icon="mdi:solar-power",
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up ML Weather sensors from a config entry."""
    coordinator: MLWeatherCoordinator = hass.data[DOMAIN][entry.entry_id][ML_WEATHER_COORDINATOR]

    entities = [
        MLWeatherSensor(coordinator, entry, description)
        for description in SENSOR_DESCRIPTIONS
    ]

    # Add PV forecast sensors
    entities.extend([
        MLPVForecastSensor(coordinator, entry, description)
        for description in PV_FORECAST_DESCRIPTIONS
    ])

    async_add_entities(entities, False)


class MLWeatherSensor(CoordinatorEntity[MLWeatherCoordinator], SensorEntity):
    """Representation of an ML Weather sensor."""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: MLWeatherCoordinator,
        entry: ConfigEntry,
        description: SensorEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._entry = entry
        self.entity_description = description
        # Use 'weather_' prefix to avoid collision with PV sensors
        self._attr_unique_id = f"{entry.entry_id}_weather_{description.key}"
        self._attr_name = description.name

    @property
    def device_info(self) -> dict[str, Any]:
        """Return device information."""
        name = self._entry.data.get(CONF_NAME, "ML Weather")
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": name,
            "manufacturer": "Solar Forecast ML",
            "model": "Multi-Expert Blended Weather",
            "sw_version": self.coordinator.data.get("version", "unknown") if self.coordinator.data else "unknown",
        }

    @property
    def native_value(self) -> float | str | bool | None:
        """Return the sensor value."""
        current = self.coordinator.get_current_weather()
        data_key = WEATHER_SENSOR_KEY_MAP.get(self.entity_description.key)
        if data_key:
            value = current.get(data_key)
            if value is not None:
                # Handle different value types
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value
                # Numeric values get rounded
                return round(value, 1)
        return None

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        if not self.coordinator.last_update_success:
            return False

        # Optional sensors may not always have data
        if self.entity_description.key in OPTIONAL_SENSORS:
            return self.coordinator.last_update_success

        # Check if we have current weather data
        current = self.coordinator.get_current_weather()
        data_key = WEATHER_SENSOR_KEY_MAP.get(self.entity_description.key)
        return current.get(data_key) is not None if data_key else False

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.async_write_ha_state()


class MLPVForecastSensor(CoordinatorEntity[MLWeatherCoordinator], SensorEntity):
    """Representation of a PV Forecast sensor."""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: MLWeatherCoordinator,
        entry: ConfigEntry,
        description: SensorEntityDescription,
    ) -> None:
        """Initialize the PV forecast sensor."""
        super().__init__(coordinator)
        self._entry = entry
        self.entity_description = description
        # Use 'pv_' prefix to avoid collision with weather sensors
        self._attr_unique_id = f"{entry.entry_id}_pv_{description.key}"
        self._attr_name = description.name

    @property
    def device_info(self) -> dict[str, Any]:
        """Return device information."""
        name = self._entry.data.get(CONF_NAME, "ML Weather")
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": name,
            "manufacturer": "Solar Forecast ML",
            "model": "Multi-Expert Blended Weather",
            "sw_version": self.coordinator.data.get("version", "unknown") if self.coordinator.data else "unknown",
        }

    @property
    def native_value(self) -> float | None:
        """Return the PV forecast value."""
        pv_forecast = self.coordinator.get_pv_forecast()
        data_key = PV_FORECAST_KEY_MAP.get(self.entity_description.key)
        if data_key:
            value = pv_forecast.get(data_key)
            if value is not None:
                return round(value, 2)
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional state attributes."""
        pv_forecast = self.coordinator.get_pv_forecast()
        attrs = {}

        if self.entity_description.key == SENSOR_TYPE_PV_FORECAST_TODAY:
            if pv_forecast.get("today_source"):
                attrs["source"] = pv_forecast["today_source"]
            if pv_forecast.get("today_locked") is not None:
                attrs["locked"] = pv_forecast["today_locked"]
        elif self.entity_description.key == SENSOR_TYPE_PV_FORECAST_TOMORROW:
            if pv_forecast.get("tomorrow_date"):
                attrs["date"] = pv_forecast["tomorrow_date"]
            if pv_forecast.get("tomorrow_source"):
                attrs["source"] = pv_forecast["tomorrow_source"]
        elif self.entity_description.key == SENSOR_TYPE_PV_FORECAST_DAY_AFTER:
            if pv_forecast.get("day_after_tomorrow_date"):
                attrs["date"] = pv_forecast["day_after_tomorrow_date"]
            if pv_forecast.get("day_after_tomorrow_source"):
                attrs["source"] = pv_forecast["day_after_tomorrow_source"]

        return attrs

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        if not self.coordinator.last_update_success:
            return False

        pv_forecast = self.coordinator.get_pv_forecast()
        data_key = PV_FORECAST_KEY_MAP.get(self.entity_description.key)
        return pv_forecast.get(data_key) is not None if data_key else False

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.async_write_ha_state()
