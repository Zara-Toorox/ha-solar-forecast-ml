"""
Custom exceptions for Solar Forecast ML integration.

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
from typing import Any, Optional

_LOGGER = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SolarForecastMLException(Exception):
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context or {}
        
        log_message = f"[{severity.value.upper()}] {message}"
        if context:
            log_message += f" | Context: {context}"
        
        if severity == ErrorSeverity.CRITICAL:
            _LOGGER.error(log_message)
        elif severity == ErrorSeverity.HIGH:
            _LOGGER.error(log_message)
        elif severity == ErrorSeverity.MEDIUM:
            _LOGGER.warning(log_message)
        else:
            _LOGGER.info(log_message)


class ConfigurationException(SolarForecastMLException):
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None
    ):
        super().__init__(f"Configuration Error: {message}", severity, context)


class DependencyException(SolarForecastMLException):
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.CRITICAL,
        context: Optional[dict[str, Any]] = None
    ):
        super().__init__(f"Dependency Error: {message}", severity, context)


class WeatherAPIException(SolarForecastMLException):
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None
    ):
        super().__init__(f"Weather API Error: {message}", severity, context)


class WeatherException(SolarForecastMLException):
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[dict[str, Any]] = None
    ):
        super().__init__(f"Weather Error: {message}", severity, context)


class DataIntegrityException(SolarForecastMLException):
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None
    ):
        super().__init__(f"Data Integrity Error: {message}", severity, context)


class DataValidationException(SolarForecastMLException):
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[dict[str, Any]] = None
    ):
        super().__init__(f"Data Validation Error: {message}", severity, context)


class ValidationException(SolarForecastMLException):
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[dict[str, Any]] = None,
        field_name: Optional[str] = None
    ):
        if field_name:
            if context is None:
                context = {}
            context["field_name"] = field_name
        super().__init__(f"Validation Error: {message}", severity, context)


class MLModelException(SolarForecastMLException):
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None
    ):
        super().__init__(f"ML Model Error: {message}", severity, context)


class ModelException(SolarForecastMLException):
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None
    ):
        super().__init__(f"Model Error: {message}", severity, context)


class StorageException(SolarForecastMLException):
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None
    ):
        super().__init__(f"Storage Error: {message}", severity, context)


class ForecastException(SolarForecastMLException):
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[dict[str, Any]] = None
    ):
        super().__init__(f"Forecast Error: {message}", severity, context)


class CircuitBreakerOpenException(SolarForecastMLException):
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None
    ):
        super().__init__(f"Circuit Breaker Open: {message}", severity, context)


def create_context(**kwargs) -> dict[str, Any]:
    context = {
        "timestamp": _get_current_timestamp(),
        **kwargs
    }
    return context


def _get_current_timestamp() -> str:
    from datetime import datetime
    return datetime.now().isoformat()


def create_exception(
    exception_type: type[SolarForecastMLException],
    message: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    **context_kwargs
) -> SolarForecastMLException:
    context = create_context(**context_kwargs)
    return exception_type(message, severity, context)
