# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from homeassistant.core import callback

_LOGGER = logging.getLogger(__name__)

class FileBasedSensorMixin(ABC):
    """Mixin for sensors that read from daily_forecasts.json"""

    def __init__(self, *args, **kwargs):
        """Initialize mixin @zara"""
        super().__init__(*args, **kwargs)
        self._cached_value: Optional[Any] = None

    @abstractmethod
    def extract_value_from_file(self, forecast_data: dict) -> Optional[Any]:
        """Extract value from loaded forecast data - must be implemented @zara"""
        pass

    async def async_added_to_hass(self) -> None:
        """Setup sensor with file loading @zara"""
        await super().async_added_to_hass()
        await self._load_from_file()
        self.async_on_remove(self._coordinator.async_add_listener(self._handle_coordinator_update))
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates @zara"""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value and update state @zara"""
        await self._load_from_file()
        self.async_write_ha_state()

    async def _load_from_file(self) -> None:
        """Load value from daily_forecasts.json @zara"""
        try:
            forecast_data = await self._coordinator.data_manager.load_daily_forecasts()
            if forecast_data and isinstance(forecast_data, dict):
                self._cached_value = self.extract_value_from_file(forecast_data)
            else:
                self._cached_value = None
        except Exception as e:
            _LOGGER.warning(f"Failed to load {self.__class__.__name__} from file: {e}")
            self._cached_value = None

    @property
    def available(self) -> bool:
        """Sensor availability @zara"""
        return self._cached_value is not None

    @property
    def native_value(self) -> Any:
        """Return cached value @zara"""
        return self._cached_value

class CoordinatorPropertySensorMixin(ABC):
    """Mixin for sensors reading from coordinator properties"""

    @abstractmethod
    def get_coordinator_value(self) -> Optional[Any]:
        """Get value from coordinator - must be implemented @zara"""
        pass

    @property
    def available(self) -> bool:
        """Sensor availability @zara"""
        return (
            self.coordinator.last_update_success
            and self.coordinator.data is not None
            and self.native_value is not None
        )

    @property
    def native_value(self) -> Any:
        """Return value from coordinator @zara"""
        return self.get_coordinator_value()

class LiveSensorMixin(ABC):
    """Mixin for sensors with live entity tracking"""

    @abstractmethod
    def get_tracked_entities(self) -> list[str]:
        """Return list of entity IDs to track - must be implemented @zara"""
        pass

    @abstractmethod
    def calculate_live_value(self) -> Optional[Any]:
        """Calculate value from tracked entities - must be implemented @zara"""
        pass

    async def async_added_to_hass(self) -> None:
        """Setup live tracking @zara"""
        await super().async_added_to_hass()

        self.async_on_remove(self._coordinator.async_add_listener(self._handle_coordinator_update))

        from homeassistant.helpers.event import async_track_state_change_event

        tracked_entities = self.get_tracked_entities()
        for entity_id in tracked_entities:
            if entity_id:
                self.async_on_remove(
                    async_track_state_change_event(self.hass, entity_id, self._handle_sensor_change)
                )

        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates @zara"""
        self.async_write_ha_state()

    @callback
    def _handle_sensor_change(self, event) -> None:
        """Handle entity state changes @zara"""
        self.async_write_ha_state()

    @property
    def native_value(self) -> Any:
        """Return calculated live value @zara"""
        return self.calculate_live_value()

class StatisticsFileBasedMixin(FileBasedSensorMixin):
    """Specialized mixin for statistics from daily_forecasts.json"""

    @property
    def available(self) -> bool:
        """Always available, shows 0.0 if no data @zara"""
        return True

    @property
    def native_value(self) -> float:
        """Return value or 0.0 if None @zara"""
        return self._cached_value if self._cached_value is not None else 0.0

class AlwaysAvailableFileBasedMixin(FileBasedSensorMixin):
    """Mixin for sensors that should always be available with fallback values"""

    @property
    def available(self) -> bool:
        """Always available - shows fallback value if no data @zara"""
        return True
