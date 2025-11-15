"""
Electricity Price Sensors

Provides electricity price monitoring and charging recommendations
Separate from Solar/ML sensors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CURRENCY_EURO
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from ..const import (
    DOMAIN,
    ELECTRICITY_CHARGING_RECOMMENDATION_SENSOR,
    ELECTRICITY_CHEAPEST_HOUR_TODAY_SENSOR,
    ELECTRICITY_MOST_EXPENSIVE_HOUR_TODAY_SENSOR,
    ELECTRICITY_PRICE_AVG_TODAY_SENSOR,
    ELECTRICITY_PRICE_AVG_WEEK_SENSOR,
    ELECTRICITY_PRICE_CURRENT_SENSOR,
    ELECTRICITY_PRICE_MAX_TODAY_SENSOR,
    ELECTRICITY_PRICE_MIN_TODAY_SENSOR,
    ELECTRICITY_PRICE_NEXT_HOUR_SENSOR,
    ELECTRICITY_SAVINGS_TODAY_SENSOR,
    ICON_CHARGING_RECOMMENDATION,
    ICON_ELECTRICITY_PRICE,
    INTEGRATION_MODEL,
    SOFTWARE_VERSION,
    UNIT_CENT_PER_KWH,
    UNIT_EURO_PER_KWH,
)
from ..coordinator import SolarForecastMLCoordinator

_LOGGER = logging.getLogger(__name__)


class BaseElectricityPriceSensor(CoordinatorEntity, SensorEntity):
    """Base class for electricity price sensors"""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: SolarForecastMLCoordinator,
        entry: ConfigEntry,
        sensor_key: str,
        name: str,
    ):
        """Initialize the base electricity price sensor"""
        super().__init__(coordinator)
        self.entry = entry
        self._sensor_key = sensor_key

        self._attr_unique_id = f"{entry.entry_id}_electricity_{sensor_key}"
        self._attr_translation_key = f"electricity_{sensor_key}"

        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, f"{entry.entry_id}_battery")},
            name="Battery Management",
            manufacturer="Zara-Toorox",
            model=f"{INTEGRATION_MODEL} - Battery",
            sw_version=SOFTWARE_VERSION,
            via_device=(DOMAIN, entry.entry_id),
        )

    @property
    def available(self) -> bool:
        """Return if entity is available"""
        return (
            self.coordinator.last_update_success
            and self.coordinator.electricity_service is not None
        )


class ElectricityPriceCurrentSensor(BaseElectricityPriceSensor):
    """Current electricity price sensor (Cent/kWh)"""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL  # MONETARY only supports TOTAL or None
    _attr_native_unit_of_measurement = UNIT_CENT_PER_KWH
    _attr_icon = ICON_ELECTRICITY_PRICE

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize current price sensor"""
        super().__init__(coordinator, entry, ELECTRICITY_PRICE_CURRENT_SENSOR, "Current Price")

    @property
    def native_value(self) -> Optional[float]:
        """Return current electricity price"""
        if not self.coordinator.electricity_service:
            return None
        return self.coordinator.electricity_service.get_current_price()

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes"""
        if not self.coordinator.electricity_service:
            return {}

        avg_today = self.coordinator.electricity_service.get_average_price_today()
        current = self.native_value

        attrs = {
            "currency": "EUR",
            "country": self.coordinator.electricity_service.country,
        }

        if current and avg_today:
            diff = current - avg_today
            attrs["difference_from_average"] = round(diff, 2)
            attrs["is_below_average"] = current < avg_today

        return attrs


class ElectricityPriceNextHourSensor(BaseElectricityPriceSensor):
    """Next hour electricity price sensor (Cent/kWh)"""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL  # MONETARY only supports TOTAL or None
    _attr_native_unit_of_measurement = UNIT_CENT_PER_KWH
    _attr_icon = ICON_ELECTRICITY_PRICE

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize next hour price sensor"""
        super().__init__(coordinator, entry, ELECTRICITY_PRICE_NEXT_HOUR_SENSOR, "Next Hour Price")

    @property
    def native_value(self) -> Optional[float]:
        """Return next hour electricity price"""
        if not self.coordinator.electricity_service:
            return None

        next_hour = (datetime.now().hour + 1) % 24
        return self.coordinator.electricity_service.get_price_at_hour(next_hour)


class ElectricityPriceAvgTodaySensor(BaseElectricityPriceSensor):
    """Average electricity price today sensor (Cent/kWh)"""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL  # MONETARY only supports TOTAL or None
    _attr_native_unit_of_measurement = UNIT_CENT_PER_KWH
    _attr_icon = ICON_ELECTRICITY_PRICE

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize average today sensor"""
        super().__init__(
            coordinator, entry, ELECTRICITY_PRICE_AVG_TODAY_SENSOR, "Average Price Today"
        )

    @property
    def native_value(self) -> Optional[float]:
        """Return average price today"""
        if not self.coordinator.electricity_service:
            return None
        return self.coordinator.electricity_service.get_average_price_today()


class ElectricityPriceAvgWeekSensor(BaseElectricityPriceSensor):
    """Average electricity price this week sensor (Cent/kWh)"""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL  # MONETARY only supports TOTAL or None
    _attr_native_unit_of_measurement = UNIT_CENT_PER_KWH
    _attr_icon = ICON_ELECTRICITY_PRICE

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize average week sensor"""
        super().__init__(
            coordinator, entry, ELECTRICITY_PRICE_AVG_WEEK_SENSOR, "Average Price Week"
        )

    @property
    def native_value(self) -> Optional[float]:
        """Return average price this week"""
        if not self.coordinator.electricity_service:
            return None
        return self.coordinator.electricity_service.get_average_price_week()


class ElectricityPriceMinTodaySensor(BaseElectricityPriceSensor):
    """Minimum electricity price today sensor (Cent/kWh)"""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL  # MONETARY only supports TOTAL or None
    _attr_native_unit_of_measurement = UNIT_CENT_PER_KWH
    _attr_icon = "mdi:arrow-down-bold"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize min price today sensor"""
        super().__init__(coordinator, entry, ELECTRICITY_PRICE_MIN_TODAY_SENSOR, "Min Price Today")

    @property
    def native_value(self) -> Optional[float]:
        """Return minimum price today"""
        if not self.coordinator.electricity_service:
            return None

        cheapest = self.coordinator.electricity_service.get_cheapest_hours(1)
        if cheapest:
            return cheapest[0][1]  # Return price
        return None


class ElectricityPriceMaxTodaySensor(BaseElectricityPriceSensor):
    """Maximum electricity price today sensor (Cent/kWh)"""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL  # MONETARY only supports TOTAL or None
    _attr_native_unit_of_measurement = UNIT_CENT_PER_KWH
    _attr_icon = "mdi:arrow-up-bold"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize max price today sensor"""
        super().__init__(coordinator, entry, ELECTRICITY_PRICE_MAX_TODAY_SENSOR, "Max Price Today")

    @property
    def native_value(self) -> Optional[float]:
        """Return maximum price today"""
        if not self.coordinator.electricity_service:
            return None

        expensive = self.coordinator.electricity_service.get_most_expensive_hours(1)
        if expensive:
            return expensive[0][1]  # Return price
        return None


class ElectricityCheapestHourSensor(BaseElectricityPriceSensor):
    """Cheapest hour today sensor"""

    _attr_icon = "mdi:clock-check"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize cheapest hour sensor"""
        super().__init__(
            coordinator, entry, ELECTRICITY_CHEAPEST_HOUR_TODAY_SENSOR, "Cheapest Hour"
        )

    @property
    def native_value(self) -> Optional[str]:
        """Return cheapest hour today"""
        if not self.coordinator.electricity_service:
            return None

        cheapest = self.coordinator.electricity_service.get_cheapest_hours(1)
        if cheapest:
            hour, price = cheapest[0]
            return f"{hour:02d}:00"
        return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes"""
        if not self.coordinator.electricity_service:
            return {}

        cheapest = self.coordinator.electricity_service.get_cheapest_hours(1)
        if cheapest:
            hour, price = cheapest[0]
            return {
                "hour": hour,
                "price": price,
                "price_unit": UNIT_CENT_PER_KWH,
            }
        return {}


class ElectricityMostExpensiveHourSensor(BaseElectricityPriceSensor):
    """Most expensive hour today sensor"""

    _attr_icon = "mdi:clock-alert"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize most expensive hour sensor"""
        super().__init__(
            coordinator, entry, ELECTRICITY_MOST_EXPENSIVE_HOUR_TODAY_SENSOR, "Most Expensive Hour"
        )

    @property
    def native_value(self) -> Optional[str]:
        """Return most expensive hour today"""
        if not self.coordinator.electricity_service:
            return None

        expensive = self.coordinator.electricity_service.get_most_expensive_hours(1)
        if expensive:
            hour, price = expensive[0]
            return f"{hour:02d}:00"
        return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes"""
        if not self.coordinator.electricity_service:
            return {}

        expensive = self.coordinator.electricity_service.get_most_expensive_hours(1)
        if expensive:
            hour, price = expensive[0]
            return {
                "hour": hour,
                "price": price,
                "price_unit": UNIT_CENT_PER_KWH,
            }
        return {}


class ElectricityChargingRecommendationSensor(BaseElectricityPriceSensor):
    """Charging recommendation sensor"""

    _attr_icon = ICON_CHARGING_RECOMMENDATION

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize charging recommendation sensor"""
        super().__init__(
            coordinator,
            entry,
            ELECTRICITY_CHARGING_RECOMMENDATION_SENSOR,
            "Charging Recommendation",
        )

    @property
    def native_value(self) -> Optional[str]:
        """Return charging recommendation"""
        if not self.coordinator.electricity_service:
            return "No data"

        should_charge = self.coordinator.electricity_service.should_charge_now()
        current_price = self.coordinator.electricity_service.get_current_price()
        avg_price = self.coordinator.electricity_service.get_average_price_today()

        if current_price is None or avg_price is None:
            return "No data"

        if should_charge:
            return "Charge now"
        else:
            cheapest = self.coordinator.electricity_service.get_cheapest_hours(1)
            if cheapest:
                hour, _ = cheapest[0]
                return f"Wait until {hour:02d}:00"
            return "Wait"

    @property
    def icon(self) -> str:
        """Return dynamic icon"""
        if self.native_value == "Charge now":
            return "mdi:battery-charging"
        elif self.native_value and "Wait" in self.native_value:
            return "mdi:timer-sand"
        return ICON_CHARGING_RECOMMENDATION

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes"""
        if not self.coordinator.electricity_service:
            return {}

        current_price = self.coordinator.electricity_service.get_current_price()
        avg_price = self.coordinator.electricity_service.get_average_price_today()
        cheapest = self.coordinator.electricity_service.get_cheapest_hours(3)

        attrs = {
            "current_price": current_price,
            "average_price": avg_price,
            "should_charge": self.coordinator.electricity_service.should_charge_now(),
        }

        if cheapest:
            attrs["cheapest_hours"] = [{"hour": f"{h:02d}:00", "price": p} for h, p in cheapest]

        return attrs


class ElectricitySavingsTodaySensor(BaseElectricityPriceSensor):
    """Electricity savings today sensor (EUR)"""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = CURRENCY_EURO
    _attr_icon = "mdi:piggy-bank"

    def __init__(self, coordinator: SolarForecastMLCoordinator, entry: ConfigEntry):
        """Initialize savings sensor"""
        super().__init__(coordinator, entry, ELECTRICITY_SAVINGS_TODAY_SENSOR, "Savings Today")

    @property
    def native_value(self) -> Optional[float]:
        """Calculate savings today"""
        if not self.coordinator.battery_collector or not self.coordinator.electricity_service:
            return None

        # Get battery charge from grid today
        charge_from_grid = self.coordinator.battery_collector.get_charge_today()

        # Get average price today
        avg_price = self.coordinator.electricity_service.get_average_price_today()

        if not charge_from_grid or not avg_price:
            return 0.0

        # Calculate savings (simplified)
        # Savings = Energy charged * (Avg price - Actual paid price)
        # For now, assume we charged at cheapest hours
        cheapest = self.coordinator.electricity_service.get_cheapest_hours(1)
        if cheapest:
            cheapest_price = cheapest[0][1]
            savings = charge_from_grid * (
                (avg_price - cheapest_price) / 100
            )  # Convert cents to EUR
            return round(max(savings, 0.0), 2)

        return 0.0

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional attributes"""
        if not self.coordinator.battery_collector:
            return {}

        return {
            "charge_from_grid_today": self.coordinator.battery_collector.get_charge_today(),
            "avg_price_today": (
                self.coordinator.electricity_service.get_average_price_today()
                if self.coordinator.electricity_service
                else None
            ),
        }


# Export all electricity price sensors
ELECTRICITY_PRICE_SENSORS = [
    ElectricityPriceCurrentSensor,
    ElectricityPriceNextHourSensor,
    ElectricityPriceAvgTodaySensor,
    ElectricityPriceAvgWeekSensor,
    ElectricityPriceMinTodaySensor,
    ElectricityPriceMaxTodaySensor,
    ElectricityCheapestHourSensor,
    ElectricityMostExpensiveHourSensor,
    ElectricityChargingRecommendationSensor,
    ElectricitySavingsTodaySensor,
]
