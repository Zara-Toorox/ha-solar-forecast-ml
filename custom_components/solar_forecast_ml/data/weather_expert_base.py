# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Weather Expert Base - Abstract Base Class for Weather Experts V16.0.0

Provides the foundation for weather data expert implementations.
Each expert fetches data from a specific weather API/source.

@zara
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class WeatherExpert(ABC):
    """Abstract base class for weather data experts. @zara

    All weather experts must implement:
    - get_cloud_cover(): Get cloud cover prediction for an hour
    - get_forecast_data(): Get full forecast data for an hour
    """

    def __init__(self, name: str):
        """Initialize the weather expert. @zara

        Args:
            name: Unique identifier for this expert
        """
        self.name = name
        self._last_error: Optional[str] = None
        self._consecutive_failures: int = 0

    @abstractmethod
    async def get_cloud_cover(self, date: str, hour: int) -> Optional[float]:
        """Get cloud cover prediction for a specific hour. @zara

        Args:
            date: Date string in ISO format (YYYY-MM-DD)
            hour: Hour of day (0-23)

        Returns:
            Cloud cover percentage (0-100) or None if unavailable
        """
        pass

    @abstractmethod
    async def get_forecast_data(
        self,
        date: str,
        hour: int
    ) -> Optional[dict[str, Any]]:
        """Get full forecast data for a specific hour. @zara

        Args:
            date: Date string in ISO format (YYYY-MM-DD)
            hour: Hour of day (0-23)

        Returns:
            Dictionary with forecast data or None if unavailable
        """
        pass

    def get_last_error(self) -> Optional[str]:
        """Get last error message. @zara"""
        return self._last_error

    def record_failure(self, error: str) -> None:
        """Record a failure and increment counter. @zara"""
        self._last_error = error
        self._consecutive_failures += 1

    def reset_failures(self) -> None:
        """Reset failure counter after successful operation. @zara"""
        self._consecutive_failures = 0
        self._last_error = None

    @property
    def is_healthy(self) -> bool:
        """Check if expert has low failure count. @zara"""
        return self._consecutive_failures < 5
