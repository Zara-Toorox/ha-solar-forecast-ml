# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Exception classes for Solar Forecast ML @zara"""

from enum import Enum
from typing import Any, Optional
from datetime import datetime, timezone


class ErrorSeverity(Enum):
    """Error severity levels @zara"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SolarForecastMLException(Exception):
    """Base exception for Solar Forecast ML @zara"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context or {}


class ConfigurationException(SolarForecastMLException):
    """Configuration error @zara"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None,
    ):
        super().__init__(f"Configuration Error: {message}", severity, context)


class DependencyException(SolarForecastMLException):
    """Missing or incompatible dependency @zara"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.CRITICAL,
        context: Optional[dict[str, Any]] = None,
    ):
        super().__init__(f"Dependency Error: {message}", severity, context)


class WeatherAPIException(SolarForecastMLException):
    """Weather API error @zara"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None,
    ):
        super().__init__(f"Weather API Error: {message}", severity, context)


class DataIntegrityException(SolarForecastMLException):
    """Data integrity or corruption error @zara"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None,
    ):
        super().__init__(f"Data Integrity Error: {message}", severity, context)


class DataValidationException(SolarForecastMLException):
    """Data validation failed @zara"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[dict[str, Any]] = None,
    ):
        super().__init__(f"Data Validation Error: {message}", severity, context)


class MLModelException(SolarForecastMLException):
    """ML model error @zara"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None,
    ):
        super().__init__(f"ML Model Error: {message}", severity, context)


class CircuitBreakerOpenException(SolarForecastMLException):
    """Circuit breaker is open @zara"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None,
    ):
        prefix = "" if message.lower().startswith("circuit breaker") else "Circuit Breaker Open: "
        super().__init__(f"{prefix}{message}", severity, context)


def create_context(**kwargs) -> dict[str, Any]:
    """Create context dict with timestamp @zara"""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
