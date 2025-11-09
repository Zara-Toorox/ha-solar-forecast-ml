"""
Error Handling Service for Solar Forecast ML Integration

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
from datetime import datetime, timedelta, timezone # Added timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Optional, Dict # Added Coroutine

# Import specific exception types used
from ..core.core_exceptions import (
    ConfigurationException,
    WeatherAPIException,
    CircuitBreakerOpenException,
    MLModelException,
    DataIntegrityException,
    SolarForecastMLException # Import base exception
)

_LOGGER = logging.getLogger(__name__)


# --- Enums for State and Error Types ---
class CircuitBreakerState(Enum):
    """Possible states of the Circuit Breaker"""
    CLOSED = "closed"       # Operations allowed, monitoring failures
    OPEN = "open"           # Operations blocked for a timeout period
    HALF_OPEN = "half_open" # Allows a limited number of test operations


class ErrorType(Enum):
    """Categorization of errors for circuit breaker and logging"""
    CONFIGURATION = "configuration"
    API_ERROR = "api_error"         # External API errors (e.g., weather)
    NETWORK_ERROR = "network_error"     # General network issues
    ML_TRAINING = "ml_training"     # Errors during model training
    ML_PREDICTION = "ml_prediction"   # Errors during prediction generation
    DATA_INTEGRITY = "data_integrity" # Issues with stored data files (JSON, etc.)
    JSON_OPERATION = "json_operation" # Specific errors during JSON read/write
    SENSOR_ERROR = "sensor_error"     # Errors reading HA sensor states
    DEPENDENCY = "dependency"       # Missing Python packages
    UNKNOWN = "unknown"             # Unclassified errors


# --- Circuit Breaker Implementation ---
class CircuitBreaker:
    """Implements the Circuit Breaker pattern to prevent repeated failures"""
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,  # Failures needed to open the circuit
        success_threshold: int = 2,  # Successful calls in HALF_OPEN to close
        open_timeout_seconds: int = 60, # Duration the circuit stays OPEN
        # half_open_timeout removed, not typically needed as success/failure in HALF_OPEN decides state
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
        self.success_count = 0 # Used only in HALF_OPEN state
        self.last_failure_time: Optional[datetime] = None
        self.opened_at_time: Optional[datetime] = None # When the circuit last opened
        self.last_state_change_time: datetime = datetime.now(timezone.utc)

        # Track types of errors leading to failures
        self.error_type_counts = defaultdict(int)
        _LOGGER.info(f"Circuit Breaker '{self.name}' initialized: "
                     f"FailureThreshold={failure_threshold}, SuccessThreshold={success_threshold}, "
                     f"OpenTimeout={open_timeout_seconds}s")

    def _get_current_time(self) -> datetime:
        """Return the current time in UTC"""
        # Consistently use UTC for time comparisons
        return datetime.now(timezone.utc)

    def _reset_counts(self):
        """Reset failure and success counts"""
        self.failure_count = 0
        self.success_count = 0

    def _change_state(self, new_state: CircuitBreakerState):
        """Handles state transitions and logging"""
        if self.state != new_state:
             old_state = self.state
             self.state = new_state
             self.last_state_change_time = self._get_current_time()
             _LOGGER.info(f"Circuit Breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}")
             # Reset counts on state change
             self._reset_counts()
             # Record open time if applicable
             if new_state == CircuitBreakerState.OPEN:
                  self.opened_at_time = self.last_state_change_time
             else:
                  self.opened_at_time = None
             # Clear error type counts when closing
             if new_state == CircuitBreakerState.CLOSED:
                  self.error_type_counts.clear()


    def allow_request(self) -> bool:
        """Check if the circuit breaker should allow the operation to proceed"""
        current_time = self._get_current_time()

        if self.state == CircuitBreakerState.CLOSED:
            return True # Allow requests

        if self.state == CircuitBreakerState.OPEN:
            # Check if the open timeout has expired
            if self.opened_at_time and (current_time - self.opened_at_time >= self.open_timeout):
                # Timeout expired, transition to HALF_OPEN
                self._change_state(CircuitBreakerState.HALF_OPEN)
                # Allow the *first* request in HALF_OPEN state
                return True
            else:
                # Still within timeout, block the request
                _LOGGER.debug(f"Circuit Breaker '{self.name}' is OPEN. Request blocked.")
                return False

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Allow requests in HALF_OPEN state for testing the underlying service
            # Success/failure will determine the next state transition
            return True

        # Should not be reachable
        _LOGGER.error(f"Circuit Breaker '{self.name}' in unknown state: {self.state}")
        return False


    def record_success(self):
        """Record a successful operation Handles state transition from HALF_OPEN to CLOSED"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            _LOGGER.debug(f"Circuit Breaker '{self.name}' (HALF_OPEN): Success recorded ({self.success_count}/{self.success_threshold}).")
            # Check if success threshold is met to close the circuit
            if self.success_count >= self.success_threshold:
                self._change_state(CircuitBreakerState.CLOSED)
        elif self.state == CircuitBreakerState.CLOSED:
             # Optionally reset failure count slowly on success in CLOSED state?
             # Simple approach: do nothing special on success in CLOSED state.
             pass

    def record_failure(self, error_type: ErrorType = ErrorType.UNKNOWN):
        """Record a failed operation Handles state transitions to OPEN"""
        current_time = self._get_current_time()
        self.last_failure_time = current_time
        self.error_type_counts[error_type] += 1
        _LOGGER.debug(f"Circuit Breaker '{self.name}': Failure recorded (Type: {error_type.value}).")


        if self.state == CircuitBreakerState.CLOSED:
            self.failure_count += 1
            _LOGGER.debug(f"Circuit Breaker '{self.name}' (CLOSED): Failure count incremented ({self.failure_count}/{self.failure_threshold}).")
            # Check if failure threshold is met to open the circuit
            if self.failure_count >= self.failure_threshold:
                self._change_state(CircuitBreakerState.OPEN)
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Any failure in HALF_OPEN state immediately re-opens the circuit
            _LOGGER.warning(f"Circuit Breaker '{self.name}': Failure occurred in HALF_OPEN state. Re-opening circuit.")
            self._change_state(CircuitBreakerState.OPEN)

        # Special handling for configuration errors: Open immediately
        if error_type == ErrorType.CONFIGURATION and self.state != CircuitBreakerState.OPEN:
             _LOGGER.warning(f"Circuit Breaker '{self.name}': Configuration error detected. Opening circuit immediately.")
             # Force state to OPEN regardless of current counts
             self._change_state(CircuitBreakerState.OPEN)


    def reset(self):
        """Manually reset the circuit breaker to the CLOSED state"""
        _LOGGER.info(f"Circuit Breaker '{self.name}' manually reset to CLOSED state.")
        self._change_state(CircuitBreakerState.CLOSED)
        self.last_failure_time = None # Clear last failure time on manual reset


    def get_status(self) -> Dict[str, Any]:
        """Return the current status of the circuit breaker"""
        status = {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count, # Relevant only in HALF_OPEN
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "open_timeout_seconds": self.open_timeout.total_seconds(),
            "error_types_count": dict(self.error_type_counts),
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change_time": self.last_state_change_time.isoformat(),
            "opened_at_time": self.opened_at_time.isoformat() if self.opened_at_time else None,
        }
        # Calculate time remaining if open
        if self.state == CircuitBreakerState.OPEN and self.opened_at_time:
             time_remaining = self.open_timeout - (self._get_current_time() - self.opened_at_time)
             status["open_time_remaining_seconds"] = max(0, round(time_remaining.total_seconds()))
        else:
             status["open_time_remaining_seconds"] = None

        return status


# --- Error Handling Service ---
class ErrorHandlingService:
    """Central service for handling errors logging operational details"""
    def __init__(self):
        """Initialize the Error Handling Service"""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        # Use limited-size lists (like deques) for logs if memory is a concern
        self.error_log: List[Dict[str, Any]] = []
        self.ml_operation_log: List[Dict[str, Any]] = []
        self.json_operation_log: List[Dict[str, Any]] = []
        self.sensor_status_log: List[Dict[str, Any]] = []

        # Maximum number of entries to keep in each log
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
            _LOGGER.warning(f"Circuit Breaker '{name}' is already registered. Returning existing instance.")
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
             # Catch invalid configuration errors from CircuitBreaker constructor
             _LOGGER.error(f"Failed to register Circuit Breaker '{name}': {e}")
             raise # Re-raise as this is a setup issue


    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a registered circuit breaker by name"""
        breaker = self.circuit_breakers.get(name)
        if breaker is None:
             _LOGGER.warning(f"Attempted to get non-existent Circuit Breaker '{name}'.")
        return breaker


    async def execute_with_circuit_breaker(
        self,
        breaker_name: str,
        operation: Callable[..., Coroutine[Any, Any, Any]], # Expects an async function
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Execute an asynchronous operation protected by a circuit breaker"""
        breaker = self.get_circuit_breaker(breaker_name)
        if breaker is None:
            raise ValueError(f"Circuit Breaker '{breaker_name}' is not registered.")

        # Check if the circuit allows the request
        if not breaker.allow_request():
            # Circuit is OPEN, raise specific exception
            error_msg = (f"Circuit Breaker '{breaker_name}' is {breaker.state.value}. Operation blocked.")
            _LOGGER.warning(error_msg)
            # Log this specific event
            self._log_error(breaker_name, CircuitBreakerOpenException.__name__, error_msg, ErrorType.UNKNOWN) # Or a specific type?
            raise CircuitBreakerOpenException(error_msg)

        # Attempt the operation
        try:
            # Execute the async operation
            result = await operation(*args, **kwargs)
            # Record success if operation completed without exception
            breaker.record_success()
            _LOGGER.debug(f"Operation '{operation.__name__}' executed successfully via Circuit Breaker '{breaker_name}'.")
            return result

        except Exception as e:
            # Operation failed, record failure and re-raise the exception
            error_type_enum = self._classify_error(e)
            _LOGGER.error(f"Operation '{operation.__name__}' failed via Circuit Breaker '{breaker_name}': {e}", exc_info=False) # Log less verbosely here
            breaker.record_failure(error_type_enum)
            # Use handle_error for detailed logging and storage
            await self.handle_error(e, source=f"circuit_breaker_{breaker_name}", context={"operation": operation.__name__})
            raise # Re-raise the original exception


    async def handle_error(
        self,
        error: Exception,
        source: str, # Where the error originated (e.g., 'ml_training', 'weather_api')
        context: Optional[Dict[str, Any]] = None,
        pipeline_position: Optional[str] = None # Specific step within the source
    ) -> None:
        """Log and store detailed information about an encountered error"""
        error_type_enum = self._classify_error(error)
        error_class_name = type(error).__name__

        # Prepare detailed log entry
        error_details = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "error_type": error_class_name,
            "error_classification": error_type_enum.value,
            "message": str(error),
            "pipeline_position": pipeline_position,
            "context": context or {},
            # Include stack trace only for certain error types or severities?
            "stack_trace": traceback.format_exc() if isinstance(error, (MLModelException, DataIntegrityException)) else None
        }

        # Add to error log list, maintaining max size
        self.error_log.append(error_details)
        if len(self.error_log) > self.max_error_log_size:
            self.error_log = self.error_log[-self.max_error_log_size:] # Keep latest entries

        # Log the error using standard logging
        log_level = logging.ERROR if isinstance(error, (MLModelException, DataIntegrityException, ConfigurationException)) else logging.WARNING
        _LOGGER.log(
            log_level,
            f"[ERROR] Source: {source} | Type: {error_class_name} ({error_type_enum.value}) | "
            f"Position: {pipeline_position or 'N/A'} | Message: {error}",
            # Only include exc_info for critical errors to avoid overly verbose logs
            exc_info=error_details["stack_trace"] is not None
        )
        if context:
             _LOGGER.debug(f"  Error Context: {context}")


    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify an exception into an ErrorType category"""
        if isinstance(error, MLModelException):
            # Further classify ML errors if needed
            msg = str(error).lower()
            if "training" in msg: return ErrorType.ML_TRAINING
            if "prediction" in msg: return ErrorType.ML_PREDICTION
            return ErrorType.ML_PREDICTION # Default ML error to prediction
        elif isinstance(error, DataIntegrityException):
            return ErrorType.DATA_INTEGRITY
        elif isinstance(error, ConfigurationException):
            return ErrorType.CONFIGURATION
        elif isinstance(error, WeatherAPIException):
            return ErrorType.API_ERROR
        elif isinstance(error, CircuitBreakerOpenException):
             # This is a consequence, not a root cause, classify as unknown?
             return ErrorType.UNKNOWN
        elif isinstance(error, asyncio.TimeoutError):
             return ErrorType.NETWORK_ERROR # Often network related
        elif isinstance(error, OSError): # Includes FileNotFoundError etc.
             if "Network is unreachable" in str(error) or "Connection refused" in str(error):
                  return ErrorType.NETWORK_ERROR
             return ErrorType.JSON_OPERATION # Assume file related OS errors
        elif isinstance(error, ImportError):
            return ErrorType.DEPENDENCY
        # Add more specific classifications here based on exception types

        # Fallback based on string content (less reliable)
        error_str = str(error).lower()
        if "network" in error_str or "connection" in error_str or "timeout" in error_str:
            return ErrorType.NETWORK_ERROR
        if "sensor" in error_str or "state" in error_str or "entity not found" in error_str:
            return ErrorType.SENSOR_ERROR
        if "json" in error_str or "decode" in error_str or "file" in error_str:
            # Could be DATA_INTEGRITY or JSON_OPERATION, lean towards JSON_OPERATION
            return ErrorType.JSON_OPERATION

        # Default to UNKNOWN if no specific classification matches
        return ErrorType.UNKNOWN


    def log_ml_operation(
        self,
        operation: str, # e.g., 'model_training', 'prediction'
        success: bool,
        metrics: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None
    ) -> None:
        """Log the outcome of a Machine Learning operation"""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "timestamp": timestamp,
            "operation": operation,
            "success": success,
            "metrics": metrics or {},
            "context": context or {},
            "duration_seconds": duration_seconds
        }

        # Add to ML log list, maintaining max size
        self.ml_operation_log.append(log_entry)
        if len(self.ml_operation_log) > self.max_ml_log_size:
            self.ml_operation_log = self.ml_operation_log[-self.max_ml_log_size:]

        # Log concisely using standard logging
        status_str = "Success" if success else "FAILED"
        duration_str = f"{duration_seconds:.2f}s" if duration_seconds is not None else "N/A"
        metrics_str = f"Metrics: {metrics}" if metrics else ""
        log_level = logging.INFO if success else logging.ERROR
        _LOGGER.log(
            log_level,
            f"[ML OP] {operation}: {status_str} | Duration: {duration_str} | {metrics_str}"
        )
        if context:
             _LOGGER.debug(f"  ML Op Context: {context}")


    def log_json_operation(
        self,
        file_name: str,
        operation: str, # e.g., 'read', 'write', 'migration'
        success: bool,
        file_size_bytes: Optional[int] = None,
        records_count: Optional[int] = None, # If applicable (e.g., predictions read)
        error_message: Optional[str] = None
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
            "error_message": error_message
        }

        # Add to JSON log list, maintaining max size
        self.json_operation_log.append(log_entry)
        if len(self.json_operation_log) > self.max_json_log_size:
            self.json_operation_log = self.json_operation_log[-self.max_json_log_size:]

        # Log concisely
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
        sensor_name: str, # Entity ID or descriptive name
        sensor_type: str, # e.g., 'power', 'temperature'
        available: bool,
        value: Optional[Any] = None,
        error_message: Optional[str] = None # If unavailable or parsing failed
    ) -> None:
        """Log the status and value of critical external sensors"""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "timestamp": timestamp,
            "sensor_name": sensor_name,
            "sensor_type": sensor_type,
            "available": available,
            "value": str(value) if value is not None else None, # Store value as string
            "error_message": error_message
        }

        # Add to sensor log list, maintaining max size
        self.sensor_status_log.append(log_entry)
        if len(self.sensor_status_log) > self.max_sensor_log_size:
            self.sensor_status_log = self.sensor_status_log[-self.max_sensor_log_size:]

        # Log status (Debug for available, Warning for unavailable)
        status_str = "Available" if available else "UNAVAILABLE"
        if available:
            _LOGGER.debug(f"[SENSOR] {sensor_name} ({sensor_type}): {status_str} | Value: {value}")
        else:
            _LOGGER.warning(f"[SENSOR] {sensor_name} ({sensor_type}): {status_str} | Error: {error_message or 'Unknown'}")


    def _log_error(
        self,
        source: str,
        error_type_name: str, # Class name of the exception
        message: str,
        error_classification: ErrorType = ErrorType.UNKNOWN
    ):
        """Internal helper to add simple errors like CB open to the main error log"""
        timestamp = datetime.now(timezone.utc).isoformat()
        error_entry = {
            "timestamp": timestamp,
            "source": source,
            "error_type": error_type_name,
            "message": message,
            "classification": error_classification.value,
            # No stack trace for these simpler logged errors
        }

        self.error_log.append(error_entry)
        if len(self.error_log) > self.max_error_log_size:
            self.error_log = self.error_log[-self.max_error_log_size:]
        # Logging is handled by the calling function (e.g., execute_with_circuit_breaker)


    # --- Log Access and Management ---
    def get_error_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent error log entries"""
        return self.error_log[-limit:]

    def get_ml_operation_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent ML operation log entries"""
        return self.ml_operation_log[-limit:]

    def get_json_operation_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent JSON operation log entries"""
        return self.json_operation_log[-limit:]

    def get_sensor_status_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent sensor status log entries"""
        return self.sensor_status_log[-limit:]

    def clear_error_log(self):
        """Clear all entries from the error log"""
        self.error_log.clear()
        _LOGGER.info("Error log cleared.")

    def clear_ml_operation_log(self):
        """Clear all entries from the ML operation log"""
        self.ml_operation_log.clear()
        _LOGGER.info("ML operation log cleared.")

    def clear_json_operation_log(self):
        """Clear all entries from the JSON operation log"""
        self.json_operation_log.clear()
        _LOGGER.info("JSON operation log cleared.")

    def clear_sensor_status_log(self):
        """Clear all entries from the sensor status log"""
        self.sensor_status_log.clear()
        _LOGGER.info("Sensor status log cleared.")

    def get_all_status(self) -> Dict[str, Any]:
        """Return a summary of the error handlers status and recent logs"""
        # Get status of all circuit breakers
        breaker_statuses = {
            name: breaker.get_status()
            for name, breaker in self.circuit_breakers.items()
        }

        return {
            "circuit_breakers": breaker_statuses,
            "log_sizes": {
                "error": len(self.error_log),
                "ml_operation": len(self.ml_operation_log),
                "json_operation": len(self.json_operation_log),
                "sensor_status": len(self.sensor_status_log),
            },
            # Include a few recent entries from each log for quick overview
            "recent_errors": self.get_error_log(5),
            "recent_ml_operations": self.get_ml_operation_log(5),
            "recent_json_operations": self.get_json_operation_log(5),
            "recent_sensor_status": self.get_sensor_status_log(5),
        }

    def reset_all_circuit_breakers(self):
        """Manually reset all registered circuit breakers to the CLOSED state"""
        _LOGGER.info("Resetting all registered circuit breakers...")
        count = 0
        for name, breaker in self.circuit_breakers.items():
            breaker.reset()
            count += 1
        _LOGGER.info(f"Reset {count} circuit breaker(s).")