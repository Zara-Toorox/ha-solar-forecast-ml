"""
Error handling service with circuit breaker pattern.

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
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, Dict

from .exceptions import (
    ConfigurationException,
    WeatherAPIException,
    CircuitBreakerOpenException,
    MLModelException,
    DataIntegrityException,
)

_LOGGER = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ErrorType(Enum):
    CONFIGURATION = "configuration"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    ML_TRAINING = "ml_training"
    ML_PREDICTION = "ml_prediction"
    DATA_INTEGRITY = "data_integrity"
    JSON_OPERATION = "json_operation"
    SENSOR_ERROR = "sensor_error"
    UNKNOWN = "unknown"


class CircuitBreaker:
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        success_threshold: int = 2,
        timeout: int = 60,
        half_open_timeout: int = 30,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.half_open_timeout = half_open_timeout
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self.last_state_change: datetime = datetime.now()
        
        self.error_type_counts = defaultdict(int)
    
    def should_allow_request(self) -> bool:
        current_time = datetime.now()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            time_since_failure = (current_time - self.last_failure_time).total_seconds()
            
            if time_since_failure >= self.timeout:
                _LOGGER.info("Circuit Breaker %s: transitioning to HALF_OPEN", self.name)
                self.state = CircuitBreakerState.HALF_OPEN
                self.last_state_change = current_time
                return True
            
            return False
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self):
        current_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.success_threshold:
                _LOGGER.info("Circuit Breaker %s: transitioning to CLOSED", self.name)
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.error_type_counts.clear()
                self.last_state_change = current_time
        
        elif self.state == CircuitBreakerState.CLOSED:
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self, error_type: ErrorType = ErrorType.UNKNOWN):
        current_time = datetime.now()
        self.last_failure_time = current_time
        
        self.error_type_counts[error_type] += 1
        
        if error_type == ErrorType.CONFIGURATION:
            _LOGGER.warning(
                "Circuit Breaker %s: Configuration error detected - opening immediately",
                self.name
            )
            self.state = CircuitBreakerState.OPEN
            self.failure_count = self.failure_threshold
            self.last_state_change = current_time
            return
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            _LOGGER.warning("Circuit Breaker %s: failure in HALF_OPEN, reopening", self.name)
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
            self.last_state_change = current_time
        
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count += 1
            
            if self.failure_count >= self.failure_threshold:
                _LOGGER.warning(
                    "Circuit Breaker %s OPEN (too many failures: %d/%d)",
                    self.name,
                    self.failure_count,
                    self.failure_threshold
                )
                self.state = CircuitBreakerState.OPEN
                self.last_state_change = current_time
    
    def reset(self):
        _LOGGER.info("Circuit Breaker %s: manual reset", self.name)
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.error_type_counts.clear()
        self.last_failure_time = None
        self.last_state_change = datetime.now()
    
    def get_status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "error_types": dict(self.error_type_counts),
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change.isoformat(),
        }


class ErrorHandlingService:
    
    def __init__(self):
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.error_log: list[dict[str, Any]] = []
        self.ml_operation_log: list[dict[str, Any]] = []
        self.json_operation_log: list[dict[str, Any]] = []
        self.sensor_status_log: list[dict[str, Any]] = []
        self.max_error_log_size = 100
        self.max_ml_log_size = 200
        self.max_json_log_size = 100
        self.max_sensor_log_size = 50
    
    def register_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 3,
        success_threshold: int = 2,
        timeout: int = 60,
        half_open_timeout: int = 30,
    ) -> CircuitBreaker:
        if name in self.circuit_breakers:
            _LOGGER.warning("Circuit Breaker %s already registered", name)
            return self.circuit_breakers[name]
        
        breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            half_open_timeout=half_open_timeout,
        )
        
        self.circuit_breakers[name] = breaker
        _LOGGER.info("Circuit Breaker %s registered", name)
        return breaker
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker | None:
        return self.circuit_breakers.get(name)
    
    async def execute_with_circuit_breaker(
        self,
        breaker_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        breaker = self.circuit_breakers.get(breaker_name)
        
        if breaker is None:
            _LOGGER.error("Circuit Breaker %s not found", breaker_name)
            raise ValueError(f"Circuit Breaker {breaker_name} not registered")
        
        if not breaker.should_allow_request():
            error_msg = f"Circuit Breaker {breaker_name} is {breaker.state.value.upper()} - operation blocked"
            _LOGGER.warning(error_msg)
            self._log_error(breaker_name, "CircuitBreakerOpenException", error_msg)
            raise CircuitBreakerOpenException(error_msg)
        
        try:
            result = await operation(*args, **kwargs)
            breaker.record_success()
            return result
            
        except ConfigurationException as err:
            _LOGGER.error("[%s] ConfigurationException: %s", breaker_name, err)
            breaker.record_failure(ErrorType.CONFIGURATION)
            self._log_error(breaker_name, "ConfigurationException", str(err), ErrorType.CONFIGURATION)
            raise
            
        except WeatherAPIException as err:
            _LOGGER.error("[%s] WeatherAPIException: %s", breaker_name, err)
            breaker.record_failure(ErrorType.API_ERROR)
            self._log_error(breaker_name, "WeatherAPIException", str(err), ErrorType.API_ERROR)
            raise
            
        except Exception as err:
            _LOGGER.error("[%s] Exception: %s", breaker_name, err)
            breaker.record_failure(ErrorType.UNKNOWN)
            self._log_error(breaker_name, type(err).__name__, str(err), ErrorType.UNKNOWN)
            raise
    
    async def handle_error(
        self,
        error: Exception,
        source: str,
        context: Optional[Dict[str, Any]] = None,
        pipeline_position: Optional[str] = None
    ) -> None:
        error_type = self._classify_error(error)
        
        error_details = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "error_type": type(error).__name__,
            "error_classification": error_type.value,
            "message": str(error),
            "pipeline_position": pipeline_position,
            "context": context or {},
            "stack_trace": traceback.format_exc()
        }
        
        self.error_log.append(error_details)
        
        if len(self.error_log) > self.max_error_log_size:
            self.error_log = self.error_log[-self.max_error_log_size:]
        
        _LOGGER.error(
            "[ML ERROR] Source: %s | Type: %s | Position: %s | Context: %s | Message: %s",
            source,
            type(error).__name__,
            pipeline_position or "unknown",
            context or {},
            str(error)
        )
        
        if error_type in [ErrorType.ML_TRAINING, ErrorType.ML_PREDICTION]:
            _LOGGER.error("[ML STACK TRACE]\n%s", traceback.format_exc())
    
    def _classify_error(self, error: Exception) -> ErrorType:
        if isinstance(error, MLModelException):
            if "training" in str(error).lower():
                return ErrorType.ML_TRAINING
            elif "prediction" in str(error).lower():
                return ErrorType.ML_PREDICTION
            return ErrorType.UNKNOWN
        elif isinstance(error, DataIntegrityException):
            return ErrorType.DATA_INTEGRITY
        elif isinstance(error, ConfigurationException):
            return ErrorType.CONFIGURATION
        elif isinstance(error, WeatherAPIException):
            return ErrorType.API_ERROR
        else:
            error_str = str(error).lower()
            if "sensor" in error_str or "state" in error_str:
                return ErrorType.SENSOR_ERROR
            elif "json" in error_str or "file" in error_str:
                return ErrorType.JSON_OPERATION
            return ErrorType.UNKNOWN
    
    def log_ml_operation(
        self,
        operation: str,
        success: bool,
        metrics: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None
    ) -> None:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "success": success,
            "metrics": metrics or {},
            "context": context or {},
            "duration_seconds": duration_seconds
        }
        
        self.ml_operation_log.append(log_entry)
        
        if len(self.ml_operation_log) > self.max_ml_log_size:
            self.ml_operation_log = self.ml_operation_log[-self.max_ml_log_size:]
        
        status_icon = "âœ…" if success else "âŒ"
        _LOGGER.info(
            "[ML OPERATION] %s %s | Metrics: %s | Context: %s | Duration: %.2fs",
            status_icon,
            operation,
            metrics or {},
            context or {},
            duration_seconds or 0.0
        )
    
    def log_json_operation(
        self,
        file_name: str,
        operation: str,
        success: bool,
        file_size_bytes: Optional[int] = None,
        records_count: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> None:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "file_name": file_name,
            "operation": operation,
            "success": success,
            "file_size_bytes": file_size_bytes,
            "records_count": records_count,
            "error_message": error_message
        }
        
        self.json_operation_log.append(log_entry)
        
        if len(self.json_operation_log) > self.max_json_log_size:
            self.json_operation_log = self.json_operation_log[-self.max_json_log_size:]
        
        status_icon = "âœ…" if success else "âŒ"
        if success:
            _LOGGER.info(
                "[JSON OPERATION] %s %s | File: %s | Size: %s bytes | Records: %s",
                status_icon,
                operation,
                file_name,
                file_size_bytes or "unknown",
                records_count or "unknown"
            )
        else:
            _LOGGER.error(
                "[JSON OPERATION] %s %s | File: %s | Error: %s",
                status_icon,
                operation,
                file_name,
                error_message or "unknown"
            )
    
    def log_sensor_status(
        self,
        sensor_name: str,
        sensor_type: str,
        available: bool,
        value: Optional[Any] = None,
        error_message: Optional[str] = None
    ) -> None:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "sensor_name": sensor_name,
            "sensor_type": sensor_type,
            "available": available,
            "value": value,
            "error_message": error_message
        }
        
        self.sensor_status_log.append(log_entry)
        
        if len(self.sensor_status_log) > self.max_sensor_log_size:
            self.sensor_status_log = self.sensor_status_log[-self.max_sensor_log_size:]
        
        status_icon = "âœ…" if available else "âŒ"
        if available:
            _LOGGER.debug(
                "[SENSOR STATUS] %s %s (%s) | Value: %s",
                status_icon,
                sensor_name,
                sensor_type,
                value
            )
        else:
            _LOGGER.warning(
                "[SENSOR STATUS] %s %s (%s) | Unavailable | Error: %s",
                status_icon,
                sensor_name,
                sensor_type,
                error_message or "unknown"
            )
    
    def _log_error(
        self,
        source: str,
        error_type: str,
        message: str,
        error_classification: ErrorType = ErrorType.UNKNOWN
    ):
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "error_type": error_type,
            "message": message,
            "classification": error_classification.value,
        }
        
        self.error_log.append(error_entry)
        
        if len(self.error_log) > self.max_error_log_size:
            self.error_log = self.error_log[-self.max_error_log_size:]
    
    def get_error_log(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.error_log[-limit:]
    
    def get_ml_operation_log(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.ml_operation_log[-limit:]
    
    def get_json_operation_log(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.json_operation_log[-limit:]
    
    def get_sensor_status_log(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.sensor_status_log[-limit:]
    
    def clear_error_log(self):
        self.error_log.clear()
        _LOGGER.info("Error log cleared")
    
    def clear_ml_operation_log(self):
        self.ml_operation_log.clear()
        _LOGGER.info("ML operation log cleared")
    
    def clear_json_operation_log(self):
        self.json_operation_log.clear()
        _LOGGER.info("JSON operation log cleared")
    
    def clear_sensor_status_log(self):
        self.sensor_status_log.clear()
        _LOGGER.info("Sensor status log cleared")
    
    def get_all_status(self) -> dict[str, Any]:
        return {
            "circuit_breakers": {
                name: breaker.get_status()
                for name, breaker in self.circuit_breakers.items()
            },
            "error_log_size": len(self.error_log),
            "ml_operation_log_size": len(self.ml_operation_log),
            "json_operation_log_size": len(self.json_operation_log),
            "sensor_status_log_size": len(self.sensor_status_log),
            "recent_errors": self.get_error_log(5),
            "recent_ml_operations": self.get_ml_operation_log(5),
            "recent_json_operations": self.get_json_operation_log(5),
            "recent_sensor_status": self.get_sensor_status_log(5),
        }
    
    def reset_all_circuit_breakers(self):
        for name, breaker in self.circuit_breakers.items():
            breaker.reset()
        _LOGGER.info("All circuit breakers reset")
