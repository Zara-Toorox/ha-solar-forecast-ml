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
Sensor mixins for Solar Forecast ML.
Provides reusable functionality for DB-based sensors.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from homeassistant.core import callback

from ..const import CACHE_HOURLY_PREDICTIONS, CACHE_PREDICTIONS, PRED_TARGET_DATE

_LOGGER = logging.getLogger(__name__)


class DBBasedSensorMixin(ABC):
    """Mixin for sensors that read from database via coordinator. @zara"""

    def __init__(self, *args, **kwargs):
        """Initialize mixin. @zara"""
        super().__init__(*args, **kwargs)
        self._cached_value: Optional[Any] = None

    @abstractmethod
    async def extract_value_from_db(self) -> Optional[Any]:
        """Extract value from database - must be implemented. @zara"""
        pass

    async def async_added_to_hass(self) -> None:
        """Setup sensor with DB loading. @zara"""
        await super().async_added_to_hass()
        await self._load_from_db()
        self.async_on_remove(self._coordinator.async_add_listener(self._handle_coordinator_update))
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates. @zara"""
        self.hass.async_create_task(self._reload_and_update())

    async def _reload_and_update(self) -> None:
        """Reload value and update state. @zara"""
        await self._load_from_db()
        self.async_write_ha_state()

    async def _load_from_db(self) -> None:
        """Load value from database. @zara"""
        try:
            self._cached_value = await self.extract_value_from_db()
        except Exception as e:
            _LOGGER.warning(f"Failed to load {self.__class__.__name__} from DB: {e}")
            self._cached_value = None

    @property
    def available(self) -> bool:
        """Sensor availability. @zara"""
        return self._cached_value is not None

    @property
    def native_value(self) -> Any:
        """Return cached value. @zara"""
        return self._cached_value


class CoordinatorPropertySensorMixin(ABC):
    """Mixin for sensors reading from coordinator properties. @zara"""

    @abstractmethod
    def get_coordinator_value(self) -> Optional[Any]:
        """Get value from coordinator - must be implemented. @zara"""
        pass

    @property
    def available(self) -> bool:
        """Sensor availability. @zara"""
        return (
            self.coordinator.last_update_success
            and self.coordinator.data is not None
            and self.native_value is not None
        )

    @property
    def native_value(self) -> Any:
        """Return value from coordinator. @zara"""
        return self.get_coordinator_value()


class LiveSensorMixin(ABC):
    """Mixin for sensors with live entity tracking. @zara"""

    @abstractmethod
    def get_tracked_entities(self) -> list[str]:
        """Return list of entity IDs to track - must be implemented. @zara"""
        pass

    @abstractmethod
    def calculate_live_value(self) -> Optional[Any]:
        """Calculate value from tracked entities - must be implemented. @zara"""
        pass

    async def async_added_to_hass(self) -> None:
        """Setup live tracking. @zara"""
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
        """Handle coordinator updates. @zara"""
        self.async_write_ha_state()

    @callback
    def _handle_sensor_change(self, event) -> None:
        """Handle entity state changes. @zara"""
        self.async_write_ha_state()

    @property
    def native_value(self) -> Any:
        """Return calculated live value. @zara"""
        return self.calculate_live_value()


class StatisticsDBSensorMixin(DBBasedSensorMixin):
    """Specialized mixin for statistics from database. @zara"""

    @property
    def available(self) -> bool:
        """Always available, shows 0.0 if no data. @zara"""
        return True

    @property
    def native_value(self) -> float:
        """Return value or 0.0 if None. @zara"""
        return self._cached_value if self._cached_value is not None else 0.0


class AlwaysAvailableDBMixin(DBBasedSensorMixin):
    """Mixin for sensors that should always be available with fallback values. @zara"""

    @property
    def available(self) -> bool:
        """Always available - shows fallback value if no data. @zara"""
        return True


class CachedDBSensorMixin:
    """Mixin that provides caching functionality for DB-based sensors. @zara"""

    def __init__(self, *args, **kwargs):
        """Initialize mixin with cache. @zara"""
        super().__init__(*args, **kwargs)
        self._cached_value: Optional[Any] = None
        self._cache_timestamp: float = 0
        self._cache_ttl_seconds: int = 60

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid. @zara"""
        import time
        return (time.monotonic() - self._cache_timestamp) < self._cache_ttl_seconds

    def _update_cache(self, value: Any) -> None:
        """Update cached value with timestamp. @zara"""
        import time
        self._cached_value = value
        self._cache_timestamp = time.monotonic()

    def _get_cached_value(self) -> Optional[Any]:
        """Get cached value if still valid. @zara"""
        if self._is_cache_valid():
            return self._cached_value
        return None


class CoordinatorCacheMixin:
    """Mixin for accessing coordinator cache data. @zara"""

    def get_from_coordinator_cache(self, *keys: str, default: Any = None) -> Any:
        """Get nested value from coordinator data cache. @zara

        Args:
            *keys: Path of keys to navigate
            default: Default value if not found

        Returns:
            Value at path or default
        """
        try:
            if not self._coordinator.data:
                return default

            data = self._coordinator.data
            for key in keys:
                if isinstance(data, dict):
                    data = data.get(key)
                else:
                    return default
                if data is None:
                    return default

            return data
        except Exception:
            return default

    def get_hourly_predictions_cache(self) -> Optional[dict]:
        """Get hourly predictions from coordinator cache. @zara"""
        return getattr(self._coordinator, CACHE_HOURLY_PREDICTIONS, None)

    def get_today_predictions(self) -> list:
        """Get today's predictions from coordinator cache. @zara"""
        try:
            from homeassistant.util import dt as dt_util

            cache = self.get_hourly_predictions_cache()
            if not cache:
                return []

            today = dt_util.now().date().isoformat()
            predictions = cache.get(CACHE_PREDICTIONS, [])

            return [p for p in predictions if p.get(PRED_TARGET_DATE) == today]
        except Exception:
            return []
