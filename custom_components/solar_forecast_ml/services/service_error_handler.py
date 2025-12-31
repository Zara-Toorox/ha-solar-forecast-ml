# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from __future__ import annotations

import asyncio
import logging
import traceback
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Optional

from ..core.core_exceptions import SolarForecastMLException
from ..core.core_exceptions import (
    CircuitBreakerOpenException,
    ConfigurationException,
    DataIntegrityException,
    MLModelException,
    WeatherAPIException,
)

_LOGGER = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Possible states of the Circuit Breaker"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class ErrorType(Enum):
    """Categorization of errors for circuit breaker and logging"""

    CONFIGURATION = "configuration"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    ML_TRAINING = "ml_training"
    ML_PREDICTION = "ml_prediction"
    DATA_INTEGRITY = "data_integrity"
    JSON_OPERATION = "json_operation"
    SENSOR_ERROR = "sensor_error"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"

class CircuitBreaker:
    """Implements the Circuit Breaker pattern to prevent repeated failures"""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        success_threshold: int = 2,
        open_timeout_seconds: int = 60,

    ):
        """Initialize the Circuit Breaker"""
        if failure_threshold < 1 or success_threshold < 1 or open_timeout_seconds < 1:
            raise ValueError("Thresholds and timeout must be positive integers.")

        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.open_timeout = timedelta(seconds=open_timeout_seconds)

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.opened_at_time: Optional[datetime] = None
        self.last_state_change_time: datetime = datetime.now(timezone.utc)

        self.error_type_counts = defaultdict(int)
        _LOGGER.info(
            f"Circuit Breaker '{self.name}' initialized: "
            f"FailureThreshold={failure_threshold}, SuccessThreshold={success_threshold}, "
            f"OpenTimeout={open_timeout_seconds}s"
        )

    def _get_current_time(self) -> datetime:
        """Return the current time in UTC @zara"""

        return datetime.now(timezone.utc)

    def _reset_counts(self):
        """Reset failure and success counts @zara"""
        self.failure_count = 0
        self.success_count = 0

    def _change_state(self, new_state: CircuitBreakerState):
        """Handles state transitions and logging @zara"""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            self.last_state_change_time = self._get_current_time()
            _LOGGER.info(
                f"Circuit Breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}"
            )

            self._reset_counts()

            if new_state == CircuitBreakerState.OPEN:
                self.opened_at_time = self.last_state_change_time
            else:
                self.opened_at_time = None

            if new_state == CircuitBreakerState.CLOSED:
                self.error_type_counts.clear()

    def allow_request(self) -> bool:
        """Check if the circuit breaker should allow the operation to proceed @zara"""
        current_time = self._get_current_time()

        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:

            if self.opened_at_time and (current_time - self.opened_at_time >= self.open_timeout):

                self._change_state(CircuitBreakerState.HALF_OPEN)

                return True
            else:

                _LOGGER.debug(f"Circuit Breaker '{self.name}' is OPEN. Request blocked.")
                return False

        if self.state == CircuitBreakerState.HALF_OPEN:

            return True

        _LOGGER.error(f"Circuit Breaker '{self.name}' in unknown state: {self.state}")
        return False

    def record_success(self):
        """Record a successful operation Handles state transition from HALF_OPEN to CLOSED @zara"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            _LOGGER.debug(
                f"Circuit Breaker '{self.name}' (HALF_OPEN): Success recorded ({self.success_count}/{self.success_threshold})."
            )

            if self.success_count >= self.success_threshold:
                self._change_state(CircuitBreakerState.CLOSED)
        elif self.state == CircuitBreakerState.CLOSED:

            pass

    def record_failure(self, error_type: ErrorType = ErrorType.UNKNOWN):
        """Record a failed operation Handles state transitions to OPEN @zara"""
        current_time = self._get_current_time()
        self.last_failure_time = current_time
        self.error_type_counts[error_type] += 1
        _LOGGER.debug(
            f"Circuit Breaker '{self.name}': Failure recorded (Type: {error_type.value})."
        )

        if self.state == CircuitBreakerState.CLOSED:
            self.failure_count += 1
            _LOGGER.debug(
                f"Circuit Breaker '{self.name}' (CLOSED): Failure count incremented ({self.failure_count}/{self.failure_threshold})."
            )

            if self.failure_count >= self.failure_threshold:
                self._change_state(CircuitBreakerState.OPEN)
        elif self.state == CircuitBreakerState.HALF_OPEN:

            _LOGGER.warning(
                f"Circuit Breaker '{self.name}': Failure occurred in HALF_OPEN state. Re-opening circuit."
            )
            self._change_state(CircuitBreakerState.OPEN)

        if error_type == ErrorType.CONFIGURATION and self.state != CircuitBreakerState.OPEN:
            _LOGGER.warning(
                f"Circuit Breaker '{self.name}': Configuration error detected. Opening circuit immediately."
            )

            self._change_state(CircuitBreakerState.OPEN)

    def reset(self):
        """Manually reset the circuit breaker to the CLOSED state @zara"""
        _LOGGER.info(f"Circuit Breaker '{self.name}' manually reset to CLOSED state.")
        self._change_state(CircuitBreakerState.CLOSED)
        self.last_failure_time = None

    def get_status(self) -> Dict[str, Any]:
        """Return the current status of the circuit breaker @zara"""
        status = {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "open_timeout_seconds": self.open_timeout.total_seconds(),
            "error_types_count": dict(self.error_type_counts),
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_state_change_time": self.last_state_change_time.isoformat(),
            "opened_at_time": self.opened_at_time.isoformat() if self.opened_at_time else None,
        }

        if self.state == CircuitBreakerState.OPEN and self.opened_at_time:
            time_remaining = self.open_timeout - (self._get_current_time() - self.opened_at_time)
            status["open_time_remaining_seconds"] = max(0, round(time_remaining.total_seconds()))
        else:
            status["open_time_remaining_seconds"] = None

        return status

class ErrorHandlingService:
    """Central service for handling errors logging operational details"""

    def __init__(self):
        """Initialize the Error Handling Service @zara"""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        self.error_log: List[Dict[str, Any]] = []
        self.ml_operation_log: List[Dict[str, Any]] = []
        self.json_operation_log: List[Dict[str, Any]] = []
        self.sensor_status_log: List[Dict[str, Any]] = []

        self.max_error_log_size = 100
        self.max_ml_log_size = 200
        self.max_json_log_size = 100
        self.max_sensor_log_size = 50
        _LOGGER.info("ErrorHandlingService initialized.")

    def register_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 3,
        success_threshold: int = 2,
        open_timeout_seconds: int = 60,
    ) -> CircuitBreaker:
        """Register and configure a new circuit breaker"""
        if name in self.circuit_breakers:
            _LOGGER.warning(
                f"Circuit Breaker '{name}' is already registered. Returning existing instance."
            )
            return self.circuit_breakers[name]

        try:
            breaker = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                open_timeout_seconds=open_timeout_seconds,
            )
            self.circuit_breakers[name] = breaker
            _LOGGER.info(f"Circuit Breaker '{name}' registered successfully.")
            return breaker
        except ValueError as e:

            _LOGGER.error(f"Failed to register Circuit Breaker '{name}': {e}")
            raise

    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a registered circuit breaker by name @zara"""
        breaker = self.circuit_breakers.get(name)
        if breaker is None:
            _LOGGER.warning(f"Attempted to get non-existent Circuit Breaker '{name}'.")
        return breaker

    async def execute_with_circuit_breaker(
        self,
        breaker_name: str,
        operation: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute an asynchronous operation protected by a circuit breaker"""
        breaker = self.get_circuit_breaker(breaker_name)
        if breaker is None:
            raise ValueError(f"Circuit Breaker '{breaker_name}' is not registered.")

        if not breaker.allow_request():

            error_msg = (
                f"Circuit Breaker '{breaker_name}' is {breaker.state.value}. Operation blocked."
            )
            _LOGGER.warning(error_msg)

            self._log_error(
                breaker_name, CircuitBreakerOpenException.__name__, error_msg, ErrorType.UNKNOWN
            )
            raise CircuitBreakerOpenException(error_msg)

        try:

            result = await operation(*args, **kwargs)

            breaker.record_success()
            _LOGGER.debug(
                f"Operation '{operation.__name__}' executed successfully via Circuit Breaker '{breaker_name}'."
            )
            return result

        except Exception as e:

            error_type_enum = self._classify_error(e)
            _LOGGER.error(
                f"Operation '{operation.__name__}' failed via Circuit Breaker '{breaker_name}': {e}",
                exc_info=False,
            )
            breaker.record_failure(error_type_enum)

            await self.handle_error(
                e,
                source=f"circuit_breaker_{breaker_name}",
                context={"operation": operation.__name__},
            )
            raise

    async def handle_error(
        self,
        error: Exception,
        source: str,
        context: Optional[Dict[str, Any]] = None,
        pipeline_position: Optional[str] = None,
    ) -> None:
        """Log and store detailed information about an encountered error"""
        error_type_enum = self._classify_error(error)
        error_class_name = type(error).__name__

        error_details = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "error_type": error_class_name,
            "error_classification": error_type_enum.value,
            "message": str(error),
            "pipeline_position": pipeline_position,
            "context": context or {},

            "stack_trace": (
                traceback.format_exc()
                if isinstance(error, (MLModelException, DataIntegrityException))
                else None
            ),
        }

        self.error_log.append(error_details)
        if len(self.error_log) > self.max_error_log_size:
            self.error_log = self.error_log[-self.max_error_log_size :]

        log_level = (
            logging.ERROR
            if isinstance(error, (MLModelException, DataIntegrityException, ConfigurationException))
            else logging.WARNING
        )
        _LOGGER.log(
            log_level,
            f"[ERROR] Source: {source} | Type: {error_class_name} ({error_type_enum.value}) | "
            f"Position: {pipeline_position or 'N/A'} | Message: {error}",

            exc_info=error_details["stack_trace"] is not None,
        )
        if context:
            _LOGGER.debug(f"  Error Context: {context}")

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify an exception into an ErrorType category @zara"""
        if isinstance(error, MLModelException):

            msg = str(error).lower()
            if "training" in msg:
                return ErrorType.ML_TRAINING
            if "prediction" in msg:
                return ErrorType.ML_PREDICTION
            return ErrorType.ML_PREDICTION
        elif isinstance(error, DataIntegrityException):
            return ErrorType.DATA_INTEGRITY
        elif isinstance(error, ConfigurationException):
            return ErrorType.CONFIGURATION
        elif isinstance(error, WeatherAPIException):
            return ErrorType.API_ERROR
        elif isinstance(error, CircuitBreakerOpenException):

            return ErrorType.UNKNOWN
        elif isinstance(error, asyncio.TimeoutError):
            return ErrorType.NETWORK_ERROR
        elif isinstance(error, OSError):
            if "Network is unreachable" in str(error) or "Connection refused" in str(error):
                return ErrorType.NETWORK_ERROR
            return ErrorType.JSON_OPERATION
        elif isinstance(error, ImportError):
            return ErrorType.DEPENDENCY

        error_str = str(error).lower()
        if "network" in error_str or "connection" in error_str or "timeout" in error_str:
            return ErrorType.NETWORK_ERROR
        if "sensor" in error_str or "state" in error_str or "entity not found" in error_str:
            return ErrorType.SENSOR_ERROR
        if "json" in error_str or "decode" in error_str or "file" in error_str:

            return ErrorType.JSON_OPERATION

        return ErrorType.UNKNOWN

    def log_ml_operation(
        self,
        operation: str,
        success: bool,
        metrics: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Log the outcome of a Machine Learning operation"""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "timestamp": timestamp,
            "operation": operation,
            "success": success,
            "metrics": metrics or {},
            "context": context or {},
            "duration_seconds": duration_seconds,
        }

        self.ml_operation_log.append(log_entry)
        if len(self.ml_operation_log) > self.max_ml_log_size:
            self.ml_operation_log = self.ml_operation_log[-self.max_ml_log_size :]

        status_str = "Success" if success else "FAILED"
        duration_str = f"{duration_seconds:.2f}s" if duration_seconds is not None else "N/A"
        metrics_str = f"Metrics: {metrics}" if metrics else ""
        log_level = logging.INFO if success else logging.ERROR
        _LOGGER.log(
            log_level,
            f"[ML OP] {operation}: {status_str} | Duration: {duration_str} | {metrics_str}",
        )
        if context:
            _LOGGER.debug(f"  ML Op Context: {context}")

    def log_json_operation(
        self,
        file_name: str,
        operation: str,
        success: bool,
        file_size_bytes: Optional[int] = None,
        records_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Log the outcome of a JSON file operation"""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "timestamp": timestamp,
            "file_name": file_name,
            "operation": operation,
            "success": success,
            "file_size_bytes": file_size_bytes,
            "records_count": records_count,
            "error_message": error_message,
        }

        self.json_operation_log.append(log_entry)
        if len(self.json_operation_log) > self.max_json_log_size:
            self.json_operation_log = self.json_operation_log[-self.max_json_log_size :]

        status_str = "Success" if success else "FAILED"
        log_level = logging.INFO if success else logging.ERROR
        details = ""
        if success:
            size_str = f"{file_size_bytes} bytes" if file_size_bytes is not None else "N/A size"
            rec_str = f"{records_count} records" if records_count is not None else ""
            details = f"| Size: {size_str} {rec_str}".strip()
        else:
            details = f"| Error: {error_message or 'Unknown'}"

        _LOGGER.log(log_level, f"[JSON OP] {operation} on {file_name}: {status_str} {details}")

    def log_sensor_status(
        self,
        sensor_name: str,
        sensor_type: str,
        available: bool,
        value: Optional[Any] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Log the status and value of critical external sensors"""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "timestamp": timestamp,
            "sensor_name": sensor_name,
            "sensor_type": sensor_type,
            "available": available,
            "value": str(value) if value is not None else None,
            "error_message": error_message,
        }

        self.sensor_status_log.append(log_entry)
        if len(self.sensor_status_log) > self.max_sensor_log_size:
            self.sensor_status_log = self.sensor_status_log[-self.max_sensor_log_size :]

        status_str = "Available" if available else "UNAVAILABLE"
        if available:
            _LOGGER.debug(f"[SENSOR] {sensor_name} ({sensor_type}): {status_str} | Value: {value}")
        else:
            _LOGGER.warning(
                f"[SENSOR] {sensor_name} ({sensor_type}): {status_str} | Error: {error_message or 'Unknown'}"
            )

    def _log_error(
        self,
        source: str,
        error_type_name: str,
        message: str,
        error_classification: ErrorType = ErrorType.UNKNOWN,
    ):
        """Internal helper to add simple errors like CB open to the main error log"""
        timestamp = datetime.now(timezone.utc).isoformat()
        error_entry = {
            "timestamp": timestamp,
            "source": source,
            "error_type": error_type_name,
            "message": message,
            "classification": error_classification.value,

        }

        self.error_log.append(error_entry)
        if len(self.error_log) > self.max_error_log_size:
            self.error_log = self.error_log[-self.max_error_log_size :]

    def get_error_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent error log entries @zara"""
        return self.error_log[-limit:]

    def get_ml_operation_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent ML operation log entries @zara"""
        return self.ml_operation_log[-limit:]

    def get_json_operation_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent JSON operation log entries @zara"""
        return self.json_operation_log[-limit:]

    def get_sensor_status_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent sensor status log entries @zara"""
        return self.sensor_status_log[-limit:]

    def clear_error_log(self):
        """Clear all entries from the error log @zara"""
        self.error_log.clear()
        _LOGGER.info("Error log cleared.")

    def clear_ml_operation_log(self):
        """Clear all entries from the ML operation log @zara"""
        self.ml_operation_log.clear()
        _LOGGER.info("ML operation log cleared.")

    def clear_json_operation_log(self):
        """Clear all entries from the JSON operation log @zara"""
        self.json_operation_log.clear()
        _LOGGER.info("JSON operation log cleared.")

    def clear_sensor_status_log(self):
        """Clear all entries from the sensor status log @zara"""
        self.sensor_status_log.clear()
        _LOGGER.info("Sensor status log cleared.")

    def get_all_status(self) -> Dict[str, Any]:
        """Return a summary of the error handlers status and recent logs @zara"""

        breaker_statuses = {
            name: breaker.get_status() for name, breaker in self.circuit_breakers.items()
        }

        return {
            "circuit_breakers": breaker_statuses,
            "log_sizes": {
                "error": len(self.error_log),
                "ml_operation": len(self.ml_operation_log),
                "json_operation": len(self.json_operation_log),
                "sensor_status": len(self.sensor_status_log),
            },

            "recent_errors": self.get_error_log(5),
            "recent_ml_operations": self.get_ml_operation_log(5),
            "recent_json_operations": self.get_json_operation_log(5),
            "recent_sensor_status": self.get_sensor_status_log(5),
        }

    def reset_all_circuit_breakers(self):
        """Manually reset all registered circuit breakers to the CLOSED state @zara"""
        _LOGGER.info("Resetting all registered circuit breakers...")
        count = 0
        for name, breaker in self.circuit_breakers.items():
            breaker.reset()
            count += 1
        _LOGGER.info(f"Reset {count} circuit breaker(s).")
