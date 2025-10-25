"""
Abstract Base Strategy für Forecast-Berechnungen.
Definiert das Interface für ML und Rule-based Forecasts.
Version 4.8.0

Copyright (C) 2025 Zara-Toorox
# von Zara
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging

_LOGGER = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """
    Ergebnis einer Forecast-Berechnung.
    Einheitliches Format für alle Strategien.
    # von Zara
    """
    forecast_today: float
    forecast_tomorrow: float
    confidence_today: float
    confidence_tomorrow: float
    method: str
    calibrated: bool
    
    # Optionale Metadaten # von Zara
    base_capacity: Optional[float] = None
    correction_factor: Optional[float] = None
    features_used: Optional[int] = None
    model_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert ForecastResult zu Dictionary für Coordinator.
        # von Zara
        """
        result = {
            "forecast_today": round(self.forecast_today, 2),
            "forecast_tomorrow": round(self.forecast_tomorrow, 2),
            "confidence_today": round(self.confidence_today, 1),
            "confidence_tomorrow": round(self.confidence_tomorrow, 1),
            "_method": self.method,
            "_calibrated": self.calibrated,
        }
        
        # Füge optionale Metadaten hinzu # von Zara
        if self.base_capacity is not None:
            result["_base_capacity"] = self.base_capacity
        if self.correction_factor is not None:
            result["_correction_factor"] = self.correction_factor
        if self.features_used is not None:
            result["_features_used"] = self.features_used
        if self.model_accuracy is not None:
            result["_ml_accuracy"] = self.model_accuracy
            
        return result


class ForecastStrategy(ABC):
    """
    Abstract Base Class für Forecast-Strategien.
    Definiert das gemeinsame Interface für alle Forecast-Methoden.
    # von Zara
    """
    
    def __init__(self, name: str):
        """
        Initialisiere Forecast-Strategie.
        
        Args:
            name: Name der Strategie (z.B. "ml_forecast", "rule_based")
        # von Zara
        """
        self.name = name
        self._logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def calculate_forecast(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        correction_factor: float
    ) -> ForecastResult:
        """
        Berechnet Forecast basierend auf der Strategie-Implementierung.
        
        Args:
            weather_data: Wetter-Daten (temperature, clouds, humidity, etc.)
            sensor_data: Sensor-Daten (solar_capacity, power_entity, etc.)
            correction_factor: Gelernter Korrekturfaktor
            
        Returns:
            ForecastResult mit allen berechneten Werten
            
        Raises:
            Exception bei Berechnungsfehlern
        # von Zara
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Prüft ob die Strategie verfügbar ist.
        
        Returns:
            True wenn Strategie verwendet werden kann
        # von Zara
        """
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """
        Gibt Priorität der Strategie zurück.
        Höhere Werte = höhere Priorität.
        
        Returns:
            Priorität (0-100)
        # von Zara
        """
        pass
    
    def _apply_bounds(self, value: float, min_val: float, max_val: float) -> float:
        """
        Wendet Bounds auf einen Wert an.
        
        Args:
            value: Zu begrenzender Wert
            min_val: Minimum
            max_val: Maximum
            
        Returns:
            Begrenzter Wert
        # von Zara
        """
        return max(min_val, min(max_val, value))
    
    def _log_calculation(self, result: ForecastResult, details: str = "") -> None:
        """
        Loggt Berechnungsergebnis.
        
        Args:
            result: ForecastResult
            details: Zusätzliche Details (optional)
        # von Zara
        """
        self._logger.debug(
            f"✅ {self.name}: today={result.forecast_today:.2f}kWh, "
            f"tomorrow={result.forecast_tomorrow:.2f}kWh, "
            f"confidence={result.confidence_today:.1f}% {details}"
        )
