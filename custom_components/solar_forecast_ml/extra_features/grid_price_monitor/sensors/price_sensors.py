# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.config_entries import ConfigEntry

from ..const import (
    ATTR_CHEAP_HOURS_TODAY,
    ATTR_CHEAP_HOURS_TOMORROW,
    ATTR_DATA_SOURCE,
    ATTR_FORECAST_TODAY,
    ATTR_FORECAST_TOMORROW,
    ATTR_LAST_UPDATE,
    ATTR_NEXT_CHEAP_HOUR,
    ATTR_PRICE_TREND,
    ICON_CALENDAR_TODAY,
    ICON_CALENDAR_TOMORROW,
    ICON_PRICE,
    ICON_SPOT,
    SENSOR_PRICES_TODAY,
    SENSOR_PRICES_TOMORROW,
    SENSOR_SPOT_PRICE,
    SENSOR_SPOT_PRICE_NEXT_HOUR,
    SENSOR_TOTAL_PRICE,
    SENSOR_TOTAL_PRICE_NEXT_HOUR,
    UNIT_CT_KWH,
)
from .base import GridPriceBaseSensor

if TYPE_CHECKING:
    from ..coordinator import GridPriceMonitorCoordinator


class GridPriceSpotSensor(GridPriceBaseSensor):
    """Sensor for current spot price @zara"""

    _attr_native_unit_of_measurement = UNIT_CT_KWH
    _attr_device_class = SensorDeviceClass.MONETARY
    # Note: monetary device_class does not support state_class=measurement

    def __init__(
        self, coordinator: "GridPriceMonitorCoordinator", entry: ConfigEntry
    ) -> None:
        """Initialize spot price sensor @zara"""
        super().__init__(
            coordinator,
            entry,
            SENSOR_SPOT_PRICE,
            "Spot Price",
            ICON_SPOT,
        )

    @property
    def native_value(self) -> float | None:
        """Return the current spot price @zara"""
        if self.coordinator.data:
            return self.coordinator.data.get("spot_price")
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes @zara"""
        if not self.coordinator.data:
            return {}

        return {
            "spot_price_net": self.coordinator.data.get("spot_price_net"),
            "vat_rate": self.coordinator.data.get("vat_rate"),
            ATTR_FORECAST_TODAY: self.coordinator.data.get("forecast_today", []),
            ATTR_FORECAST_TOMORROW: self.coordinator.data.get("forecast_tomorrow", []),
            ATTR_PRICE_TREND: self.coordinator.data.get("price_trend"),
            ATTR_DATA_SOURCE: self.coordinator.data.get("data_source"),
            ATTR_LAST_UPDATE: self.coordinator.data.get("last_update"),
        }


class GridPriceTotalSensor(GridPriceBaseSensor):
    """Sensor for total price (spot + markup) @zara"""

    _attr_native_unit_of_measurement = UNIT_CT_KWH
    _attr_device_class = SensorDeviceClass.MONETARY
    # Note: monetary device_class does not support state_class=measurement

    def __init__(
        self, coordinator: "GridPriceMonitorCoordinator", entry: ConfigEntry
    ) -> None:
        """Initialize total price sensor @zara"""
        super().__init__(
            coordinator,
            entry,
            SENSOR_TOTAL_PRICE,
            "Total Price",
            ICON_PRICE,
        )

    @property
    def native_value(self) -> float | None:
        """Return the current total price @zara"""
        if self.coordinator.data:
            return self.coordinator.data.get("total_price")
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes @zara"""
        if not self.coordinator.data:
            return {}

        return {
            "spot_price_net": self.coordinator.data.get("spot_price_net"),
            "spot_price_gross": self.coordinator.data.get("spot_price"),
            "markup_total": self.coordinator.data.get("markup_total"),
            "vat_rate": self.coordinator.data.get("vat_rate"),
            "max_price_threshold": self.coordinator.data.get("max_price_threshold"),
            ATTR_CHEAP_HOURS_TODAY: self.coordinator.data.get("cheap_hours_today", []),
            ATTR_CHEAP_HOURS_TOMORROW: self.coordinator.data.get("cheap_hours_tomorrow", []),
            ATTR_NEXT_CHEAP_HOUR: self.coordinator.data.get("next_cheap_hour"),
            ATTR_PRICE_TREND: self.coordinator.data.get("price_trend"),
        }


class GridPriceSpotNextHourSensor(GridPriceBaseSensor):
    """Sensor for next hour spot price @zara"""

    _attr_native_unit_of_measurement = UNIT_CT_KWH
    _attr_device_class = SensorDeviceClass.MONETARY
    # Note: monetary device_class does not support state_class=measurement

    def __init__(
        self, coordinator: "GridPriceMonitorCoordinator", entry: ConfigEntry
    ) -> None:
        """Initialize next hour spot price sensor @zara"""
        super().__init__(
            coordinator,
            entry,
            SENSOR_SPOT_PRICE_NEXT_HOUR,
            "Spot Price Next Hour",
            ICON_SPOT,
        )

    @property
    def native_value(self) -> float | None:
        """Return the next hour spot price @zara"""
        if self.coordinator.data:
            return self.coordinator.data.get("spot_price_next_hour")
        return None


class GridPriceTotalNextHourSensor(GridPriceBaseSensor):
    """Sensor for next hour total price @zara"""

    _attr_native_unit_of_measurement = UNIT_CT_KWH
    _attr_device_class = SensorDeviceClass.MONETARY
    # Note: monetary device_class does not support state_class=measurement

    def __init__(
        self, coordinator: "GridPriceMonitorCoordinator", entry: ConfigEntry
    ) -> None:
        """Initialize next hour total price sensor @zara"""
        super().__init__(
            coordinator,
            entry,
            SENSOR_TOTAL_PRICE_NEXT_HOUR,
            "Total Price Next Hour",
            ICON_PRICE,
        )

    @property
    def native_value(self) -> float | None:
        """Return the next hour total price @zara"""
        if self.coordinator.data:
            return self.coordinator.data.get("total_price_next_hour")
        return None


class GridPricesTodaySensor(GridPriceBaseSensor):
    """Sensor showing all hourly prices for today @zara

    The state shows the number of hours available.
    The prices attribute contains the full hourly breakdown.
    """

    def __init__(
        self, coordinator: "GridPriceMonitorCoordinator", entry: ConfigEntry
    ) -> None:
        """Initialize prices today sensor @zara"""
        super().__init__(
            coordinator,
            entry,
            SENSOR_PRICES_TODAY,
            "Prices Today",
            ICON_CALENDAR_TODAY,
        )

    @property
    def native_value(self) -> int | None:
        """Return the number of hours with price data @zara"""
        if self.coordinator.data:
            forecast = self.coordinator.data.get("forecast_today", [])
            return len(forecast)
        return None

    @property
    def native_unit_of_measurement(self) -> str:
        """Return unit @zara"""
        return "hours"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return hourly price data @zara

        Format optimized for ApexCharts and other visualizations.
        Each entry contains: hour, spot_price_net, spot_price, total_price, is_cheap
        """
        if not self.coordinator.data:
            return {}

        forecast = self.coordinator.data.get("forecast_today", [])
        cheap_hours = self.coordinator.data.get("cheap_hours_today", [])

        # Build a simple hour->price mapping for easy access
        prices_by_hour = {}
        for entry in forecast:
            hour = entry.get("hour")
            if hour is not None:
                prices_by_hour[f"{hour:02d}:00"] = entry.get("total_price")

        return {
            "prices": forecast,
            "prices_by_hour": prices_by_hour,
            "cheap_hours": cheap_hours,
            "average_price": self.coordinator.data.get("average_price_today"),
            "cheapest_hour": self.coordinator.data.get("cheapest_hour_today"),
            "cheapest_price": self.coordinator.data.get("cheapest_price_today"),
            "most_expensive_hour": self.coordinator.data.get("most_expensive_hour_today"),
            "most_expensive_price": self.coordinator.data.get("most_expensive_price_today"),
            ATTR_LAST_UPDATE: self.coordinator.data.get("last_update"),
        }


class GridPricesTomorrowSensor(GridPriceBaseSensor):
    """Sensor showing all hourly prices for tomorrow @zara

    The state shows the number of hours available (0 if not yet published).
    The prices attribute contains the full hourly breakdown when available.
    """

    def __init__(
        self, coordinator: "GridPriceMonitorCoordinator", entry: ConfigEntry
    ) -> None:
        """Initialize prices tomorrow sensor @zara"""
        super().__init__(
            coordinator,
            entry,
            SENSOR_PRICES_TOMORROW,
            "Prices Tomorrow",
            ICON_CALENDAR_TOMORROW,
        )

    @property
    def native_value(self) -> int | None:
        """Return the number of hours with price data @zara"""
        if self.coordinator.data:
            forecast = self.coordinator.data.get("forecast_tomorrow", [])
            return len(forecast)
        return None

    @property
    def native_unit_of_measurement(self) -> str:
        """Return unit @zara"""
        return "hours"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return hourly price data for tomorrow @zara

        Format optimized for ApexCharts and other visualizations.
        Note: Tomorrow's prices are typically available after 13:00 CET.
        """
        if not self.coordinator.data:
            return {}

        forecast = self.coordinator.data.get("forecast_tomorrow", [])
        cheap_hours = self.coordinator.data.get("cheap_hours_tomorrow", [])

        # Build a simple hour->price mapping for easy access
        prices_by_hour = {}
        for entry in forecast:
            hour = entry.get("hour")
            if hour is not None:
                prices_by_hour[f"{hour:02d}:00"] = entry.get("total_price")

        # Calculate tomorrow's statistics if data is available
        tomorrow_stats = {}
        if forecast:
            total_prices = [e.get("total_price", 0) for e in forecast if e.get("total_price") is not None]
            if total_prices:
                tomorrow_stats = {
                    "average_price": round(sum(total_prices) / len(total_prices), 2),
                    "min_price": min(total_prices),
                    "max_price": max(total_prices),
                }
                # Find cheapest and most expensive hours
                cheapest_entry = min(forecast, key=lambda x: x.get("total_price", float("inf")))
                expensive_entry = max(forecast, key=lambda x: x.get("total_price", float("-inf")))
                tomorrow_stats["cheapest_hour"] = cheapest_entry.get("hour")
                tomorrow_stats["most_expensive_hour"] = expensive_entry.get("hour")

        return {
            "prices": forecast,
            "prices_by_hour": prices_by_hour,
            "cheap_hours": cheap_hours,
            "data_available": len(forecast) > 0,
            **tomorrow_stats,
            ATTR_LAST_UPDATE: self.coordinator.data.get("last_update"),
        }
