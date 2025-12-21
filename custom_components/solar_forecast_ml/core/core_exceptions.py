"""Exception Classes for Solar Forecast ML Integration V12.2.0 @zara

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

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional

from .core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Defines severity levels for exceptions"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SolarForecastMLException(Exception):
    """Base exception for all custom errors in the Solar Forecast ML integration"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the base exception"""
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context or {}

    def severity_to_loglevel(self) -> int:
        """Map error severity to standard logging levels @zara"""
        if self.severity == ErrorSeverity.CRITICAL:
            return logging.CRITICAL
        elif self.severity == ErrorSeverity.HIGH:
            return logging.ERROR
        elif self.severity == ErrorSeverity.MEDIUM:
            return logging.WARNING
        else:
            return logging.INFO

class ConfigurationException(SolarForecastMLException):
    """Exception raised for errors in the integrations configuration"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(f"Configuration Error: {message}", severity, context)

class DependencyException(SolarForecastMLException):
    """Exception raised for missing or incompatible Python dependencies"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.CRITICAL,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(f"Dependency Error: {message}", severity, context)

class WeatherAPIException(SolarForecastMLException):
    """Exception raised for errors interacting with the weather data source API or e..."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(f"Weather Data Error: {message}", severity, context)

class DataIntegrityException(SolarForecastMLException):
    """Exception raised for issues with data storage files corruption IO errors"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(f"Data Integrity Error: {message}", severity, context)

class DataValidationException(SolarForecastMLException):
    """Exception raised when loaded data fails validation checks"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(f"Data Validation Error: {message}", severity, context)

class MLModelException(SolarForecastMLException):
    """Exception related to the Machine Learning model training prediction"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(f"ML Model Error: {message}", severity, context)

class CircuitBreakerOpenException(SolarForecastMLException):
    """Exception raised when an operation is blocked by an open circuit breaker"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
    ):

        prefix = (
            "Circuit Breaker Open: " if not message.lower().startswith("circuit breaker") else ""
        )
        super().__init__(f"{prefix}{message}", severity, context)

def create_context(**kwargs) -> Dict[str, Any]:
    """Creates a standardized context dictionary adding a timestamp @zara"""
    context = {
        "timestamp": dt_util.now().isoformat(),
        **kwargs,
    }
    return context
