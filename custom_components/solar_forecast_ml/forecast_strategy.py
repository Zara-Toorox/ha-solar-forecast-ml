"""
Abstract Base Strategy for Forecast calculations.
Defines the interface for ML and Rule-based Forecasts.
Version 6.0.0

Copyright (C) 2025 Zara-Toorox
# by Zara

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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging

_LOGGER = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """
    Result of a Forecast calculation.
    Uniform format for all strategies.
    # by Zara
    """
    forecast_today: float
    forecast_tomorrow: float
    confidence_today: float
    confidence_tomorrow: float
    method: str
    calibrated: bool
    
    # Optional metadata
    base_capacity: Optional[float] = None
    correction_factor: Optional[float] = None
    features_used: Optional[int] = None
    model_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts ForecastResult to a dictionary for the Coordinator.
        # by Zara
        """
        result = {
            "forecast_today": round(self.forecast_today, 2),
            "forecast_tomorrow": round(self.forecast_tomorrow, 2),
            "confidence_today": round(self.confidence_today, 1),
            "confidence_tomorrow": round(self.confidence_tomorrow, 1),
            "_method": self.method,
            "_calibrated": self.calibrated,
        }
        
        # Add optional metadata
        if self.base_capacity is not None:
            result["_base_capacity"] = self.base_capacity
        if self.correction_factor is not None:
            result["_correction_factor"] = self.correction_factor
        if self.features_used is not None:
            result["_features_used"] = self.features_used
        if self.model_accuracy is not None:
            result["_ml_accuracy"] = self.model_accuracy
            
        return result


class ForecastStrategy(ABC):
    """
    Abstract Base Class for Forecast strategies.
    Defines the common interface for all Forecast methods.
    # by Zara
    """
    
    def __init__(self, name: str):
        """
        Initialize Forecast strategy.
        
        Args:
            name: Name of the strategy (e.g., "ml_forecast", "rule_based")
        # by Zara
        """
        self.name = name
        self._logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def calculate_forecast(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        correction_factor: float
    ) -> ForecastResult:
        """
        Calculates Forecast based on the strategy implementation.
        
        Args:
            weather_data: Weather data (temperature, clouds, humidity, etc.)
            sensor_data: Sensor data (solar_capacity, power_entity, etc.)
            correction_factor: Learned correction factor
            
        Returns:
            ForecastResult with all calculated values
            
        Raises:
            Exception on calculation errors
        # by Zara
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Checks if the strategy is available.
        
        Returns:
            True if the strategy can be used
        # by Zara
        """
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """
        Returns the priority of the strategy.
        Higher values = higher priority.
        
        Returns:
            Priority (0-100)
        # by Zara
        """
        pass
    
    def _apply_bounds(self, value: float, min_val: float, max_val: float) -> float:
        """
        Applies bounds to a value.
        
        Args:
            value: Value to be limited
            min_val: Minimum
            max_val: Maximum
            
        Returns:
            Limited value
        # by Zara
        """
        return max(min_val, min(max_val, value))
    
    def _log_calculation(self, result: ForecastResult, details: str = "") -> None:
        """
        Logs the calculation result.
        
        Args:
            result: ForecastResult
            details: Additional details (optional)
        # by Zara
        """
        self._logger.debug(
            f"✅ {self.name}: today={result.forecast_today:.2f}kWh, "  # Korrigiert
            f"tomorrow={result.forecast_tomorrow:.2f}kWh, "
            f"confidence={result.confidence_today:.1f}% {details}"
        )