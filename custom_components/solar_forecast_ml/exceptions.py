"""Custom exceptions for Solar Forecast ML integration."""
# Version 4.0 - Vollständig mit allen benötigten Exceptions # von Zara
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

_LOGGER = logging.getLogger(__name__)


# ========================================================================
# ERROR SEVERITY ENUM # von Zara
# ========================================================================

class ErrorSeverity(Enum):
    """
    Error Severity Levels für strukturiertes Error Handling
    # von Zara
    """
    LOW = "low"           # Informativ, keine Aktion erforderlich # von Zara
    MEDIUM = "medium"     # Warnung, System läuft weiter # von Zara
    HIGH = "high"         # Fehler, beeinträchtigt Funktionalität # von Zara
    CRITICAL = "critical" # Kritisch, System-Fehler # von Zara


# ========================================================================
# BASE EXCEPTION # von Zara
# ========================================================================

class SolarForecastMLException(Exception):
    """
    Base exception for Solar Forecast ML
    
    Alle Custom Exceptions erben von dieser Klasse
    # von Zara
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[dict[str, Any]] = None
    ):
        """
        Initialize exception
        
        Args:
            message: Fehlermeldung
            severity: Error Severity Level
            context: Zusätzlicher Kontext für Debugging
        # von Zara
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context or {}
        
        # Log jede Exception automatisch mit Severity # von Zara
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


# ========================================================================
# CONFIGURATION EXCEPTIONS # von Zara
# ========================================================================

class ConfigurationException(SolarForecastMLException):
    """
    Exception für Konfigurationsfehler
    
    Wird geworfen bei:
    - Ungültigen Config-Werten
    - Fehlenden Config-Parametern
    - Config-Validierungsfehlern
    # von Zara
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None
    ):
        """Initialize configuration exception # von Zara"""
        super().__init__(f"Configuration Error: {message}", severity, context)


# ========================================================================
# DEPENDENCY EXCEPTIONS # von Zara
# ========================================================================

class DependencyException(SolarForecastMLException):
    """
    Exception für fehlende oder fehlerhafte Abhängigkeiten
    
    Wird geworfen bei:
    - Fehlenden Python-Paketen (z.B. NumPy)
    - Import-Fehlern
    - Inkompatiblen Versionen
    # von Zara
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.CRITICAL,
        context: Optional[dict[str, Any]] = None
    ):
        """Initialize dependency exception # von Zara"""
        super().__init__(f"Dependency Error: {message}", severity, context)


# ========================================================================
# WEATHER EXCEPTIONS # von Zara
# ========================================================================

class WeatherAPIException(SolarForecastMLException):
    """
    Exception für Weather API Fehler
    
    Wird geworfen bei:
    - API-Timeouts
    - Ungültigen API-Responses
    - Netzwerk-Fehlern
    # von Zara
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None
    ):
        """Initialize weather API exception # von Zara"""
        super().__init__(f"Weather API Error: {message}", severity, context)


class WeatherException(SolarForecastMLException):
    """
    ✓ General Weather Exception
    
    Wird geworfen bei:
    - Allgemeinen Wetter-Problemen
    - Entity nicht verfügbar
    - Daten-Parsing-Fehler
    # von Zara
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[dict[str, Any]] = None
    ):
        """Initialize weather exception # von Zara"""
        super().__init__(f"Weather Error: {message}", severity, context)


# ========================================================================
# DATA EXCEPTIONS # von Zara
# ========================================================================

class DataIntegrityException(SolarForecastMLException):
    """
    ✓ Exception für Datenintegritätsfehler
    
    Wird geworfen bei:
    - Korrupten Daten
    - Inkonsistenten Datenstrukturen
    - Datenbank-Integritätsverletzungen
    - JSON-Parsing-Fehlern
    # von Zara
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None
    ):
        """Initialize data integrity exception # von Zara"""
        super().__init__(f"Data Integrity Error: {message}", severity, context)


class DataValidationException(SolarForecastMLException):
    """
    Exception für Datenvalidierungsfehler
    
    Wird geworfen bei:
    - Ungültigen Eingabewerten
    - Werten außerhalb gültiger Bereiche
    - Fehlenden Pflichtfeldern
    # von Zara
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[dict[str, Any]] = None
    ):
        """Initialize data validation exception # von Zara"""
        super().__init__(f"Data Validation Error: {message}", severity, context)


class ValidationException(SolarForecastMLException):
    """
    ✓ General Validation Exception
    
    Wird geworfen bei:
    - Allgemeinen Validierungsfehlern
    - Type-Checking-Fehlern
    - Schema-Validierungsfehlern
    # von Zara
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[dict[str, Any]] = None
    ):
        """Initialize validation exception # von Zara"""
        super().__init__(f"Validation Error: {message}", severity, context)


# ========================================================================
# ML MODEL EXCEPTIONS # von Zara
# ========================================================================

class MLModelException(SolarForecastMLException):
    """
    Exception für ML Model Fehler
    
    Wird geworfen bei:
    - Training-Fehlern
    - Prediction-Fehlern
    - Model-Load-Fehlern
    - Ungültigen Model-States
    # von Zara
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None
    ):
        """Initialize ML model exception # von Zara"""
        super().__init__(f"ML Model Error: {message}", severity, context)


class ModelException(SolarForecastMLException):
    """
    ✓ General Model Exception
    
    Wird geworfen bei:
    - Allgemeinen Model-Problemen
    - Model-Initialisierungsfehlern
    - Model-State-Fehlern
    # von Zara
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None
    ):
        """Initialize model exception # von Zara"""
        super().__init__(f"Model Error: {message}", severity, context)


# ========================================================================
# STORAGE EXCEPTIONS # von Zara
# ========================================================================

class StorageException(SolarForecastMLException):
    """
    Exception für Speicher-Operationen
    
    Wird geworfen bei:
    - File-I/O-Fehlern
    - Disk-Full-Problemen
    - Permissions-Fehlern
    - Backup-Fehlern
    # von Zara
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None
    ):
        """Initialize storage exception # von Zara"""
        super().__init__(f"Storage Error: {message}", severity, context)


# ========================================================================
# FORECAST EXCEPTIONS # von Zara
# ========================================================================

class ForecastException(SolarForecastMLException):
    """
    Exception für Forecast-Berechnungen
    
    Wird geworfen bei:
    - Forecast-Berechnungsfehlern
    - Ungültigen Forecast-Inputs
    - Forecast-Timeout
    # von Zara
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[dict[str, Any]] = None
    ):
        """Initialize forecast exception # von Zara"""
        super().__init__(f"Forecast Error: {message}", severity, context)


# ========================================================================
# CIRCUIT BREAKER EXCEPTION # von Zara
# ========================================================================

class CircuitBreakerOpenException(SolarForecastMLException):
    """
    Exception wenn Circuit Breaker offen ist
    
    Wird geworfen wenn:
    - Zu viele Fehler aufgetreten sind
    - Service temporär deaktiviert ist
    - Schutz-Mechanismus aktiv ist
    # von Zara
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[dict[str, Any]] = None
    ):
        """Initialize circuit breaker exception # von Zara"""
        super().__init__(f"Circuit Breaker Open: {message}", severity, context)


# ========================================================================
# HELPER FUNCTIONS # von Zara
# ========================================================================

def create_context(**kwargs) -> dict[str, Any]:
    """
    ✓ Erstellt strukturierten Error-Kontext
    
    Helper-Funktion zum Erstellen von Error-Context-Dictionaries
    mit standardisierten Feldern.
    
    Args:
        **kwargs: Beliebige Key-Value-Paare für Context
    
    Returns:
        Dictionary mit Context-Informationen
    
    Example:
        context = create_context(
            entity_id="sensor.solar_power",
            value=123.45,
            expected_range=(0, 100)
        )
    # von Zara
    """
    context = {
        "timestamp": _get_current_timestamp(),
        **kwargs
    }
    return context


def _get_current_timestamp() -> str:
    """
    Liefert aktuellen Timestamp im ISO-Format
    # von Zara
    """
    from datetime import datetime
    return datetime.now().isoformat()


# ========================================================================
# EXCEPTION FACTORY # von Zara
# ========================================================================

def create_exception(
    exception_type: type[SolarForecastMLException],
    message: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    **context_kwargs
) -> SolarForecastMLException:
    """
    Factory-Funktion zum Erstellen von Exceptions mit Context
    
    Args:
        exception_type: Exception-Klasse (z.B. DataIntegrityException)
        message: Fehlermeldung
        severity: Error Severity Level
        **context_kwargs: Context-Parameter
    
    Returns:
        Exception-Instanz mit Context
    
    Example:
        exc = create_exception(
            DataIntegrityException,
            "Invalid JSON structure",
            ErrorSeverity.HIGH,
            file_path="/path/to/file.json",
            line_number=42
        )
        raise exc
    # von Zara
    """
    context = create_context(**context_kwargs)
    return exception_type(message, severity, context)
