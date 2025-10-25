"""Error handling service with circuit breaker pattern."""
# Version 4.8.0 - von Zara
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

from .exceptions import (
    ConfigurationException,
    WeatherAPIException,
    CircuitBreakerOpenException,
)

_LOGGER = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit Breaker States - von Zara"""
    CLOSED = "closed"  # Normal operation - von Zara
    OPEN = "open"      # Too many failures, blocking calls - von Zara
    HALF_OPEN = "half_open"  # Testing if service recovered - von Zara


class ErrorType(Enum):
    """Fehlertypen für differenzierte Behandlung - von Zara"""
    CONFIGURATION = "configuration"  # Config-Fehler - sofort behandeln - von Zara
    API_ERROR = "api_error"          # API-Fehler - mit Retry - von Zara
    NETWORK_ERROR = "network_error"  # Netzwerk-Fehler - mit Retry - von Zara
    UNKNOWN = "unknown"              # Unbekannte Fehler - von Zara


class CircuitBreaker:
    """Circuit Breaker Implementation mit Error Type Differenzierung - von Zara"""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        success_threshold: int = 2,
        timeout: int = 60,
        half_open_timeout: int = 30,
    ):
        """Initialize circuit breaker - von Zara"""
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
        
        # Tracke Error Types separat - von Zara
        self.error_type_counts = defaultdict(int)
    
    def should_allow_request(self) -> bool:
        """Prüfe ob Request erlaubt ist - von Zara"""
        current_time = datetime.now()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Prüfe ob Timeout abgelaufen - von Zara
            time_since_failure = (current_time - self.last_failure_time).total_seconds()
            
            if time_since_failure >= self.timeout:
                _LOGGER.info("Circuit Breaker %s: transitioning to HALF_OPEN", self.name)
                self.state = CircuitBreakerState.HALF_OPEN
                self.last_state_change = current_time
                return True
            
            return False
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Im HALF_OPEN State, erlaube limitierte Requests - von Zara
            return True
        
        return False
    
    def record_success(self):
        """Zeichne erfolgreiche Operation auf - von Zara"""
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
            # Reset failure count bei Erfolg - von Zara
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self, error_type: ErrorType = ErrorType.UNKNOWN):
        """Zeichne fehlgeschlagene Operation auf mit Error Type - von Zara"""
        current_time = datetime.now()
        self.last_failure_time = current_time
        
        # Tracke Error Type - von Zara
        self.error_type_counts[error_type] += 1
        
        # Config-Fehler öffnen Circuit Breaker sofort - von Zara
        if error_type == ErrorType.CONFIGURATION:
            _LOGGER.warning(
                "Circuit Breaker %s: Configuration error detected - opening immediately",
                self.name
            )
            self.state = CircuitBreakerState.OPEN
            self.failure_count = self.failure_threshold
            self.last_state_change = current_time
            return
        
        # Für andere Fehler normale Logik - von Zara
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
        """Reset circuit breaker manuell - von Zara"""
        _LOGGER.info("Circuit Breaker %s: manual reset", self.name)
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.error_type_counts.clear()
        self.last_failure_time = None
        self.last_state_change = datetime.now()
    
    def get_status(self) -> dict[str, Any]:
        """Hole Circuit Breaker Status - von Zara"""
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
    """Zentraler Error Handling Service mit Circuit Breakers - von Zara"""
    
    def __init__(self):
        """Initialize error handling service - von Zara"""
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.error_log: list[dict[str, Any]] = []
        self.max_error_log_size = 100
    
    def register_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 3,
        success_threshold: int = 2,
        timeout: int = 60,
        half_open_timeout: int = 30,
    ) -> CircuitBreaker:
        """Registriere einen neuen Circuit Breaker - von Zara"""
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
        """Hole Circuit Breaker by name - von Zara"""
        return self.circuit_breakers.get(name)
    
    async def execute_with_circuit_breaker(
        self,
        breaker_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Führe Operation mit Circuit Breaker Protection aus - von Zara"""
        breaker = self.circuit_breakers.get(breaker_name)
        
        if breaker is None:
            _LOGGER.error("Circuit Breaker %s not found", breaker_name)
            raise ValueError(f"Circuit Breaker {breaker_name} not registered")
        
        # Prüfe ob Request erlaubt - von Zara
        if not breaker.should_allow_request():
            error_msg = f"Circuit Breaker {breaker_name} is {breaker.state.value.upper()} - operation blocked"
            _LOGGER.warning(error_msg)
            self._log_error(breaker_name, "CircuitBreakerOpenException", error_msg)
            raise CircuitBreakerOpenException(error_msg)
        
        # Führe Operation aus - von Zara
        try:
            result = await operation(*args, **kwargs)
            breaker.record_success()
            return result
            
        except ConfigurationException as err:
            # Config-Fehler - öffne Circuit Breaker sofort - von Zara
            _LOGGER.error("[%s] ConfigurationException: %s", breaker_name, err)
            breaker.record_failure(ErrorType.CONFIGURATION)
            self._log_error(breaker_name, "ConfigurationException", str(err), ErrorType.CONFIGURATION)
            raise
            
        except WeatherAPIException as err:
            # API-Fehler - normale Behandlung - von Zara
            _LOGGER.error("[%s] WeatherAPIException: %s", breaker_name, err)
            breaker.record_failure(ErrorType.API_ERROR)
            self._log_error(breaker_name, "WeatherAPIException", str(err), ErrorType.API_ERROR)
            raise
            
        except Exception as err:
            # Unbekannter Fehler - von Zara
            _LOGGER.error("[%s] Exception: %s", breaker_name, err)
            breaker.record_failure(ErrorType.UNKNOWN)
            self._log_error(breaker_name, type(err).__name__, str(err), ErrorType.UNKNOWN)
            raise
    
    def _log_error(
        self,
        source: str,
        error_type: str,
        message: str,
        error_classification: ErrorType = ErrorType.UNKNOWN
    ):
        """Logge Fehler in Error Log - von Zara"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "error_type": error_type,
            "message": message,
            "classification": error_classification.value,
        }
        
        self.error_log.append(error_entry)
        
        # Begrenze Log-Größe - von Zara
        if len(self.error_log) > self.max_error_log_size:
            self.error_log = self.error_log[-self.max_error_log_size:]
    
    def get_error_log(self, limit: int = 20) -> list[dict[str, Any]]:
        """Hole letzte N Fehler aus Log - von Zara"""
        return self.error_log[-limit:]
    
    def clear_error_log(self):
        """Lösche Error Log - von Zara"""
        self.error_log.clear()
        _LOGGER.info("Error log cleared")
    
    def get_all_status(self) -> dict[str, Any]:
        """Hole Status aller Circuit Breakers - von Zara"""
        return {
            "circuit_breakers": {
                name: breaker.get_status()
                for name, breaker in self.circuit_breakers.items()
            },
            "error_log_size": len(self.error_log),
            "recent_errors": self.get_error_log(5),
        }
    
    def reset_all_circuit_breakers(self):
        """Reset alle Circuit Breakers - von Zara"""
        for name, breaker in self.circuit_breakers.items():
            breaker.reset()
        _LOGGER.info("All circuit breakers reset")
