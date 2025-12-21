"""User-Friendly Log Messages V12.2.0 @zara

This module provides user-friendly log messages in German for Home Assistant users.
Technical messages are translated into understandable language with context
about what the system is doing and whether user action is required.

Message Categories:
- INFO_NORMAL: System is working as expected, no action needed
- INFO_LEARNING: System is in learning phase, patience required
- WARNING_DEGRADED: System works but with limitations
- WARNING_CONFIG: Configuration issue that should be addressed
- ERROR_ACTION: Error that requires user action
- ERROR_INTERNAL: Internal error (still shown user-friendly)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

from typing import Any, Dict, Optional


class UserMessages:
    """
    Centralized user-friendly messages for Home Assistant logs.

    All messages are in German and designed to be understood by
    non-technical users. Each message explains:
    1. What happened
    2. Why it happened (if relevant)
    3. What the system will do next
    4. Whether user action is required
    """

    # ==========================================================================
    # ML TRAINING MESSAGES
    # ==========================================================================

    ML_LEARNING_PHASE = (
        "Lernphase: Das ML-Modell sammelt noch Daten. "
        "Bisher {samples} Datenpunkte erfasst (Bereich: {min_val:.2f}-{max_val:.2f} kWh). "
        "Die Vorhersage wird mit jedem sonnigen Tag praeziser. "
        "System nutzt regelbasierte Vorhersage."
    )

    ML_LEARNING_PHASE_SIMPLE = (
        "Lernphase: ML-Modell sammelt Daten ({samples} Punkte). "
        "Regelbasierte Vorhersage aktiv - wird automatisch praeziser."
    )

    ML_NOT_ENOUGH_RESIDUALS = (
        "Lernphase: Noch nicht genuegend Vergleichsdaten fuer ML-Training "
        "({count} von 10 benoetigt). System nutzt Physik-Modell."
    )

    ML_TRAINING_SUCCESS = (
        "ML-Training erfolgreich abgeschlossen. "
        "Genauigkeit: {accuracy:.1%}, Datenpunkte: {samples}, Dauer: {duration:.1f}s"
    )

    ML_TRAINING_LAMBDA = (
        "ML-Modell optimiert (Lambda={lambda_val:.4f}, {samples} Datenpunkte)"
    )

    ML_TSS_ZERO = (
        "Lernphase: Zu wenig Variation in den Produktionsdaten. "
        "Dies ist normal bei neuen Installationen oder bewoelkten Perioden. "
        "System wartet auf mehr sonnige Tage."
    )

    ML_LINALG_ERROR = (
        "ML-Berechnung: Numerisches Problem bei der Optimierung. "
        "System verwendet alternative Berechnung."
    )

    # ==========================================================================
    # WEATHER DATA MESSAGES
    # ==========================================================================

    WEATHER_CACHE_UPDATING = (
        "Wetterdaten werden aktualisiert. Keine Aktion erforderlich."
    )

    WEATHER_CACHE_NOT_FOUND = (
        "Wetter-Cache nicht gefunden. Wetterdaten werden neu abgerufen."
    )

    WEATHER_NO_FORECAST_DATA = (
        "Keine Wettervorhersage fuer {date} im Cache. "
        "Wird beim naechsten Update automatisch abgerufen."
    )

    WEATHER_API_ERROR = (
        "Wetterdienst temporaer nicht erreichbar. "
        "Naechster Versuch in {retry_minutes} Minuten. "
        "Vorhersage basiert auf letzten verfuegbaren Daten."
    )

    WEATHER_PRECISION_SKIP = (
        "Wettergenauigkeits-Berechnung uebersprungen - Daten werden gesammelt."
    )

    WEATHER_FALLBACK_ACTIVE = (
        "Wetterdaten temporaer nicht verfuegbar. "
        "Vorhersage basiert auf Standardwerten."
    )

    # ==========================================================================
    # ASTRONOMY / SOLAR POSITION MESSAGES
    # ==========================================================================

    ASTRONOMY_CACHE_BUILDING = (
        "Sonnenstandsdaten werden erstmalig berechnet... "
        "Dies kann einige Sekunden dauern."
    )

    ASTRONOMY_CACHE_NOT_FOUND = (
        "Sonnenstandsdaten werden neu berechnet."
    )

    ASTRONOMY_CACHE_READY = (
        "Sonnenstandsdaten fuer {days} Tage berechnet."
    )

    ASTRONOMY_CACHE_ERROR = (
        "Sonnenstandsberechnung fehlgeschlagen fuer {date}. "
        "System verwendet Standardwerte."
    )

    # ==========================================================================
    # FILE / DATA MESSAGES
    # ==========================================================================

    FILE_NOT_FOUND_CREATING = (
        "Datei '{filename}' nicht gefunden. Wird neu erstellt."
    )

    FILE_HOURLY_PREDICTIONS_NEW = (
        "Neue Installation erkannt - Vorhersage-Datenbank wird erstellt."
    )

    FILE_SAVE_ERROR = (
        "Fehler beim Speichern von '{filename}'. "
        "Bitte Schreibrechte im Konfigurationsverzeichnis pruefen."
    )

    FILE_LOAD_ERROR = (
        "Fehler beim Laden von '{filename}'. "
        "Datei wird beim naechsten Update neu erstellt."
    )

    FILE_BACKUP_CREATED = (
        "Backup erstellt: {filename}"
    )

    FILE_BACKUP_RESTORED = (
        "Backup wiederhergestellt: {filename}"
    )

    # ==========================================================================
    # FORECAST MESSAGES
    # ==========================================================================

    FORECAST_TODAY_SAVED = (
        "Tagesprognose gespeichert: {kwh:.2f} kWh (Quelle: {source})"
    )

    FORECAST_TOMORROW_SAVED = (
        "Prognose fuer morgen gespeichert: {kwh:.2f} kWh"
    )

    FORECAST_LOCKED = (
        "Prognose fuer {date} bereits festgelegt. Keine Aktualisierung noetig."
    )

    FORECAST_ADJUSTED = (
        "Prognose angepasst: Aktuelle Produktion ({current:.2f} kWh) "
        "uebertrifft urspruengliche Prognose ({original:.2f} kWh). "
        "Neue Prognose: {adjusted:.2f} kWh"
    )

    FORECAST_FALLBACK = (
        "ML-Vorhersage nicht verfuegbar. Regelbasierte Vorhersage wird verwendet."
    )

    FORECAST_ALL_FAILED = (
        "Vorhersage konnte nicht erstellt werden. "
        "Bitte Internetverbindung und Wetterdienst pruefen."
    )

    # ==========================================================================
    # PRODUCTION TRACKING MESSAGES
    # ==========================================================================

    PRODUCTION_TRACKING_STARTED = (
        "Produktionsueberwachung gestartet fuer {entity}"
    )

    PRODUCTION_TRACKING_DISABLED = (
        "Keine Leistungs-Entity konfiguriert. "
        "Produktionszeiterfassung ist deaktiviert."
    )

    PRODUCTION_NEW_PEAK = (
        "Neuer Tagesrekord: {power_w:.0f}W um {time}"
    )

    PRODUCTION_ALL_TIME_PEAK = (
        "NEUER ALLZEIT-REKORD: {power_w:.0f}W am {date}"
    )

    # ==========================================================================
    # CONFIGURATION MESSAGES
    # ==========================================================================

    CONFIG_SOLAR_CAPACITY_ZERO = (
        "Konfigurationsproblem: PV-Anlagenleistung ist 0 oder negativ. "
        "Bitte in den Einstellungen korrigieren. "
        "System verwendet Fallback-Wert von 1.0 kWp."
    )

    CONFIG_POWER_ENTITY_MISSING = (
        "Setup unvollstaendig: Kein Leistungs-Sensor konfiguriert. "
        "Bitte in den Integrationseinstellungen auswaehlen."
    )

    CONFIG_DIRECTORY_ERROR = (
        "Keine Schreibrechte im Konfigurationsverzeichnis. "
        "Bitte Berechtigungen fuer '{path}' pruefen."
    )

    # ==========================================================================
    # INITIALIZATION MESSAGES
    # ==========================================================================

    INIT_COORDINATOR_READY = (
        "Solar Forecast bereit ({mode}, {capacity} kWp)"
    )

    INIT_ML_READY = (
        "ML-Modell initialisiert und bereit."
    )

    INIT_ML_DISABLED = (
        "ML-Funktionen deaktiviert. Regelbasierte Vorhersage aktiv."
    )

    INIT_CLEAN_SLATE = (
        "Neuinstallation erkannt - Datenstruktur wird erstellt..."
    )

    INIT_CLEAN_SLATE_COMPLETE = (
        "Neuinstallation abgeschlossen. System ist betriebsbereit."
    )

    INIT_DEPENDENCIES_MISSING = (
        "Einige optionale Abhaengigkeiten fehlen. "
        "ML-Funktionen sind eingeschraenkt."
    )

    # ==========================================================================
    # SCHEDULED TASKS MESSAGES
    # ==========================================================================

    TASK_MORNING_ROUTINE_START = (
        "Morgenroutine gestartet fuer {date}"
    )

    TASK_MORNING_ROUTINE_SUCCESS = (
        "Morgenroutine erfolgreich abgeschlossen."
    )

    TASK_MORNING_ROUTINE_RETRY = (
        "Morgenroutine fehlgeschlagen. Wiederholung in {wait}s..."
    )

    TASK_END_OF_DAY_START = (
        "Tagesabschluss-Routine gestartet"
    )

    TASK_END_OF_DAY_SUCCESS = (
        "Tagesabschluss erfolgreich. Genauigkeit: {accuracy:.1%}"
    )

    TASK_MIDNIGHT_ROTATION = (
        "Mitternachts-Rotation: Prognosen fuer neuen Tag vorbereitet."
    )

    # ==========================================================================
    # SENSOR MESSAGES
    # ==========================================================================

    SENSOR_UNAVAILABLE = (
        "Sensor '{entity}' nicht verfuegbar. "
        "Bitte pruefen ob der Sensor korrekt konfiguriert ist."
    )

    SENSOR_INVALID_VALUE = (
        "Ungueltiger Wert von Sensor '{entity}': {value}. "
        "Wird fuer Berechnung uebersprungen."
    )

    # ==========================================================================
    # SHADOW DETECTION MESSAGES
    # ==========================================================================

    SHADOW_DETECTION_INIT = (
        "Schattenanalyse initialisiert."
    )

    SHADOW_DETECTION_FALLBACK = (
        "Schattenanalyse: Alternative Methode wird verwendet."
    )

    # ==========================================================================
    # GENERAL ERROR MESSAGES
    # ==========================================================================

    ERROR_UNEXPECTED = (
        "Unerwarteter Fehler aufgetreten. "
        "System versucht fortzufahren. Details im Debug-Log."
    )

    ERROR_SERVICE_UNAVAILABLE = (
        "Dienst temporaer nicht verfuegbar. "
        "Automatischer Wiederholungsversuch."
    )

    @classmethod
    def format(cls, message_key: str, **kwargs: Any) -> str:
        """
        Format a message with the given parameters.

        Args:
            message_key: The message constant name (e.g., 'ML_LEARNING_PHASE')
            **kwargs: Parameters to format into the message

        Returns:
            Formatted message string, or the key if not found
        """
        message_template = getattr(cls, message_key, None)
        if message_template is None:
            return message_key

        try:
            return message_template.format(**kwargs)
        except KeyError:
            # Return template if formatting fails
            return message_template

    @classmethod
    def get(cls, message_key: str) -> str:
        """
        Get a message template without formatting.

        Args:
            message_key: The message constant name

        Returns:
            Message template string, or the key if not found
        """
        return getattr(cls, message_key, message_key)


# Convenience function for quick access
def user_msg(key: str, **kwargs: Any) -> str:
    """
    Shorthand function to get and format a user message.

    Usage:
        from .core.core_user_messages import user_msg
        _LOGGER.warning(user_msg('ML_LEARNING_PHASE', samples=10, min_val=0.1, max_val=5.2))

    Args:
        key: Message key from UserMessages class
        **kwargs: Format parameters

    Returns:
        Formatted user-friendly message
    """
    return UserMessages.format(key, **kwargs)
