# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Forecast Strategy Base Module.
Provides abstract base class and result dataclass for all forecast strategies.
@zara
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Standardized result object returned by all forecast strategies. @zara"""

    forecast_today: float
    forecast_tomorrow: float
    forecast_day_after_tomorrow: float
    confidence_today: float
    confidence_tomorrow: float
    confidence_day_after: float
    method: str
    calibrated: bool

    # Optional detailed fields
    base_capacity: Optional[float] = None
    correction_factor: Optional[float] = None
    features_used: Optional[int] = None
    model_accuracy: Optional[float] = None

    # Best hour information
    best_hour_today: Optional[int] = None
    best_hour_production_kwh: Optional[float] = None

    # Hourly breakdown
    hourly_values: Optional[List[Dict[str, Any]]] = None

    # Raw values before safeguards
    forecast_today_raw: Optional[float] = None
    forecast_tomorrow_raw: Optional[float] = None
    forecast_day_after_raw: Optional[float] = None
    safeguard_applied_today: bool = False
    safeguard_applied_tomorrow: bool = False
    safeguard_applied_day_after: bool = False

    def __post_init__(self) -> None:
        """Validate values after initialization. @zara"""
        self.forecast_today = max(0.0, self.forecast_today)
        self.forecast_tomorrow = max(0.0, self.forecast_tomorrow)
        self.forecast_day_after_tomorrow = max(0.0, self.forecast_day_after_tomorrow)

        self.confidence_today = max(0.0, min(100.0, self.confidence_today))
        self.confidence_tomorrow = max(0.0, min(100.0, self.confidence_tomorrow))
        self.confidence_day_after = max(0.0, min(100.0, self.confidence_day_after))

    def to_dict(self) -> Dict[str, Any]:
        """Convert ForecastResult to dictionary format. @zara"""
        result = {
            "forecast_today": round(self.forecast_today, 2),
            "forecast_tomorrow": round(self.forecast_tomorrow, 2),
            "forecast_day_after_tomorrow": round(self.forecast_day_after_tomorrow, 2),
            "confidence_today": round(self.confidence_today, 1),
            "confidence_tomorrow": round(self.confidence_tomorrow, 1),
            "confidence_day_after": round(self.confidence_day_after, 1),
            "_method": self.method,
            "_calibrated": self.calibrated,
        }

        if self.base_capacity is not None:
            result["_base_capacity"] = round(self.base_capacity, 2)
        if self.correction_factor is not None:
            result["_correction_factor"] = round(self.correction_factor, 3)
        if self.features_used is not None:
            result["_features_used_count"] = self.features_used
        if self.model_accuracy is not None:
            result["_ml_model_accuracy"] = round(self.model_accuracy, 4)
        if self.hourly_values is not None:
            result["hourly"] = self.hourly_values

        return result


class ForecastStrategy(ABC):
    """Abstract Base Class for all forecast calculation strategies. @zara"""

    def __init__(self, name: str):
        """Initialize the forecast strategy. @zara

        Args:
            name: Unique name identifying this strategy
        """
        self.name = name
        self._logger = logging.getLogger(f"{__name__}.{self.name}")
        self._logger.debug(f"Strategy '{self.name}' initialized.")

    @abstractmethod
    async def calculate_forecast(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        correction_factor: float
    ) -> ForecastResult:
        """Abstract method to calculate the solar forecast. @zara

        Args:
            weather_data: Weather data including hourly forecasts
            sensor_data: Current sensor readings
            correction_factor: Correction factor to apply

        Returns:
            ForecastResult with predictions
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the strategy is currently usable. @zara

        Returns:
            True if strategy can be used
        """
        pass

    @abstractmethod
    def get_priority(self) -> int:
        """Return the execution priority of the strategy. @zara

        Higher values = higher priority.

        Returns:
            Priority value (higher = more preferred)
        """
        pass

    def _apply_bounds(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp a float value between a minimum and maximum. @zara

        Args:
            value: Value to clamp
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Clamped value
        """
        if max_val < min_val:
            self._logger.warning(
                f"Invalid bounds provided: min_val ({min_val}) > max_val ({max_val})."
            )
            return max(min_val, value)

        return max(min_val, min(value, max_val))

    def _log_calculation(self, result: ForecastResult, details: str = "") -> None:
        """Log forecast calculation results consistently. @zara

        Args:
            result: The forecast result to log
            details: Additional details to include
        """
        self._logger.info(
            f"Forecast calculated using '{self.name}': "
            f"Today={result.forecast_today:.2f} kWh ({result.confidence_today:.1f}%), "
            f"Tomorrow={result.forecast_tomorrow:.2f} kWh ({result.confidence_tomorrow:.1f}%), "
            f"Day After={result.forecast_day_after_tomorrow:.2f} kWh ({result.confidence_day_after:.1f}%)"
        )
        if details:
            self._logger.debug(f"  Calculation details: {details}")
