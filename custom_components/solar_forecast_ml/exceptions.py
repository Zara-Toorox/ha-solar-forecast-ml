"""
Custom exception classes for the Solar Forecast ML integration.
Provides specific error types for better handling and diagnosis.

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
from typing import Any, Optional, Dict # Use Dict instead of dict

# Use SafeDateTimeUtil for consistent timestamps
from .helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


# --- Severity Enum ---
class ErrorSeverity(Enum):
    """Defines severity levels for exceptions."""
    LOW = "low"         # Informational or minor issue
    MEDIUM = "medium"   # Warning, potentially recoverable
    HIGH = "high"       # Error, likely requires attention
    CRITICAL = "critical" # Critical failure, may stop functionality


# --- Base Exception ---
class SolarForecastMLException(Exception):
    """Base exception for all custom errors in the Solar Forecast ML integration."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base exception.

        Args:
            message: The primary error message.
            severity: The severity level of the error.
            context: Optional dictionary containing additional context about the error.
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context or {} # Ensure context is always a dict

        # Logging is typically handled by the code catching the exception,
        # providing more specific context there. Avoid duplicate logging here.
        # Example logging where caught:
        # try:
        #     ...
        # except SolarForecastMLException as e:
        #     _LOGGER.log(e.severity_to_loglevel(), "%s - Context: %s", e, e.context)


    def severity_to_loglevel(self) -> int:
        """Map error severity to standard logging levels."""
        if self.severity == ErrorSeverity.CRITICAL:
            return logging.CRITICAL
        elif self.severity == ErrorSeverity.HIGH:
            return logging.ERROR
        elif self.severity == ErrorSeverity.MEDIUM:
            return logging.WARNING
        else: # LOW
            return logging.INFO


# --- Specific Exception Types ---

class ConfigurationException(SolarForecastMLException):
    """Exception raised for errors in the integration's configuration."""
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH, # Config errors are usually high severity
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(f"Configuration Error: {message}", severity, context)


class DependencyException(SolarForecastMLException):
    """Exception raised for missing or incompatible Python dependencies."""
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.CRITICAL, # Missing dependencies are critical
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(f"Dependency Error: {message}", severity, context)


class WeatherAPIException(SolarForecastMLException):
    """Exception raised for errors interacting with the weather data source (API or entity)."""
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH, # Failure to get weather is high severity
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(f"Weather Data Error: {message}", severity, context)


# Keep WeatherException if used for different types of weather issues?
# Or merge into WeatherAPIException? Merging seems reasonable.
# class WeatherException(SolarForecastMLException): ...


class DataIntegrityException(SolarForecastMLException):
    """Exception raised for issues with data storage files (corruption, I/O errors)."""
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH, # Data loss potential is high severity
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(f"Data Integrity Error: {message}", severity, context)


class DataValidationException(SolarForecastMLException):
    """Exception raised when loaded data fails validation checks."""
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM, # Invalid data is a warning initially
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(f"Data Validation Error: {message}", severity, context)


# Keep ValidationException if used for input validation specifically? Seems redundant with DataValidationException.
# class ValidationException(SolarForecastMLException): ...


class MLModelException(SolarForecastMLException):
    """Exception related to the Machine Learning model (training, prediction)."""
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH, # ML failures are usually high severity
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(f"ML Model Error: {message}", severity, context)


# ModelException seems redundant with MLModelException. Consider removing.
# class ModelException(SolarForecastMLException): ...


# StorageException seems redundant with DataIntegrityException. Consider removing.
# class StorageException(SolarForecastMLException): ...


# ForecastException seems redundant with MLModelException or generic SolarForecastMLException. Consider removing.
# class ForecastException(SolarForecastMLException): ...


class CircuitBreakerOpenException(SolarForecastMLException):
    """Exception raised when an operation is blocked by an open circuit breaker."""
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH, # Blocking is a high severity event
        context: Optional[Dict[str, Any]] = None
    ):
        # Message usually includes breaker name, don't prepend 'Circuit Breaker Open' if already there
        prefix = "Circuit Breaker Open: " if not message.lower().startswith("circuit breaker") else ""
        super().__init__(f"{prefix}{message}", severity, context)


# --- Helper Functions ---

def create_context(**kwargs) -> Dict[str, Any]:
    """
    Creates a standardized context dictionary, adding a timestamp.

    Args:
        **kwargs: Key-value pairs to include in the context.

    Returns:
        A dictionary containing the provided kwargs and a UTC timestamp.
    """
    context = {
        "timestamp_utc": dt_util.utcnow().isoformat(), # Use UTC timestamp
        **kwargs # Add other context details
    }
    return context


# Removed _get_current_timestamp as create_context handles it.

# Removed create_exception factory function as direct instantiation is clear enough.
# def create_exception(...) -> SolarForecastMLException: ...