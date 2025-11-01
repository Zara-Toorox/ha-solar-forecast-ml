"""
Abstract Base Strategy and Result Dataclass for Forecast calculations
in the Solar Forecast ML integration. Defines the common interface.

Copyright (C) 2025 Zara-Toorox

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
from dataclasses import dataclass, field # Import field
from typing import Any, Dict, Optional
import logging

_LOGGER = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """
    Standardized result object returned by all forecast strategies.
    Contains core forecast values and optional metadata.
    """
    # Core forecast values
    forecast_today: float        # Predicted energy for today (e.g., kWh)
    forecast_tomorrow: float     # Predicted energy for tomorrow (e.g., kWh)
    confidence_today: float    # Confidence score for today's forecast (percentage, 0-100)
    confidence_tomorrow: float # Confidence score for tomorrow's forecast (percentage, 0-100)

    # Metadata about the forecast generation
    method: str                # Identifier for the strategy used (e.g., "ml_model", "rule_based")
    calibrated: bool           # Indicates if the forecast includes learned adjustments (like correction factor)

    # Optional metadata (can be added by specific strategies)
    base_capacity: Optional[float] = None     # Base solar capacity used in calculation (kWp)
    correction_factor: Optional[float] = None # Fallback correction factor applied (if rule-based)
    features_used: Optional[int] = None       # Number of features used by the model (if ML)
    model_accuracy: Optional[float] = None    # Accuracy score of the ML model used (0.0-1.0)

    def __post_init__(self):
         """Validate values after initialization."""
         # Ensure forecasts are non-negative
         self.forecast_today = max(0.0, self.forecast_today)
         self.forecast_tomorrow = max(0.0, self.forecast_tomorrow)
         # Clamp confidence to 0-100 range
         self.confidence_today = max(0.0, min(100.0, self.confidence_today))
         self.confidence_tomorrow = max(0.0, min(100.0, self.confidence_tomorrow))

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the ForecastResult into a dictionary format suitable for
        the DataUpdateCoordinator's data payload. Rounds values for presentation.
        Internal metadata keys start with an underscore.
        """
        result = {
            "forecast_today": round(self.forecast_today, 2),
            "forecast_tomorrow": round(self.forecast_tomorrow, 2),
            "confidence_today": round(self.confidence_today, 1),
            "confidence_tomorrow": round(self.confidence_tomorrow, 1),
            # Include metadata keys for diagnostics/internal use
            "_method": self.method,
            "_calibrated": self.calibrated,
        }

        # Add optional metadata if present
        if self.base_capacity is not None:
            result["_base_capacity"] = round(self.base_capacity, 2)
        if self.correction_factor is not None:
            result["_correction_factor"] = round(self.correction_factor, 3)
        if self.features_used is not None:
            result["_features_used_count"] = self.features_used # Renamed key slightly
        if self.model_accuracy is not None:
            result["_ml_model_accuracy"] = round(self.model_accuracy, 4) # More precision

        return result


class ForecastStrategy(ABC):
    """
    Abstract Base Class for all forecast calculation strategies.
    Defines the required methods and provides common utility functions.
    """

    def __init__(self, name: str):
        """
        Initialize the forecast strategy.

        Args:
            name: A unique identifier for the strategy (e.g., "ml_forecast", "rule_based").
        """
        self.name = name
        # Create a specific logger for this strategy instance
        self._logger = logging.getLogger(f"{__name__}.{self.name}")
        self._logger.debug(f"Strategy '{self.name}' initialized.")

    @abstractmethod
    async def calculate_forecast(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        correction_factor: float
    ) -> ForecastResult:
        """
        Abstract method to calculate the solar forecast.
        Must be implemented by concrete strategy classes.

        Args:
            weather_data: Dictionary containing current weather information.
            sensor_data: Dictionary containing other relevant sensor/config data (e.g., solar_capacity).
            correction_factor: The learned fallback correction factor (may or may not be used by the strategy).

        Returns:
            A ForecastResult object containing the calculated forecast values and metadata.

        Raises:
            Exception: Concrete implementations should handle their specific errors,
                       potentially raising custom exceptions or standard ones.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Abstract method to check if the strategy is currently usable.
        For example, the ML strategy might check if a model is loaded and healthy.

        Returns:
            True if the strategy can be executed, False otherwise.
        """
        pass

    @abstractmethod
    def get_priority(self) -> int:
        """
        Abstract method to return the execution priority of the strategy.
        Higher numbers indicate higher priority (will be tried first).

        Returns:
            An integer representing the priority (e.g., 100 for ML, 50 for Rule-based).
        """
        pass

    def _apply_bounds(self, value: float, min_val: float, max_val: float) -> float:
        """
        Utility method to clamp a float value between a minimum and maximum.

        Args:
            value: The value to clamp.
            min_val: The minimum allowed value.
            max_val: The maximum allowed value.

        Returns:
            The clamped value.
        """
        if max_val < min_val:
             self._logger.warning(f"Invalid bounds provided: min_val ({min_val}) > max_val ({max_val}).")
             # Handle invalid bounds gracefully, e.g., return min_val or original value
             return max(min_val, value) # Ensure at least min_val

        return max(min_val, min(value, max_val))


    def _log_calculation(self, result: ForecastResult, details: str = "") -> None:
        """
        Helper method for consistent logging of forecast calculation results.

        Args:
            result: The ForecastResult object.
            details: Optional additional string with context-specific details.
        """
        # Log essential info at INFO level, details at DEBUG level
        self._logger.info(
            f"Forecast calculated using '{self.name}': "
            f"Today={result.forecast_today:.2f} kWh ({result.confidence_today:.1f}%), "
            f"Tomorrow={result.forecast_tomorrow:.2f} kWh ({result.confidence_tomorrow:.1f}%)"
        )
        if details:
            self._logger.debug(f"  Calculation details: {details}")