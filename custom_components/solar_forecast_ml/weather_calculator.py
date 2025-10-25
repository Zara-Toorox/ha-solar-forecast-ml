"""
Weather Calculator für Solar Forecast ML.
Berechnet Temperatur-, Cloud- und Seasonal-Faktoren.
Version 4.8.0

Copyright (C) 2025 Zara-Toorox
# von Zara
"""
import logging
from datetime import datetime
from typing import Dict, Any
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)


class WeatherCalculator:
    """
    Berechnet alle Wetter-bezogenen Faktoren für Forecast.
    Kapselt Temperatur-, Cloud- und Seasonal-Logik.
    # von Zara
    """
    
    def __init__(self):
        """
        Initialisiere Weather Calculator.
        # von Zara
        """
        # Seasonal Factors für Deutschland # von Zara
        self.SEASONAL_FACTORS = {
            "winter": 0.3,   # Deutlich niedriger im Winter # von Zara
            "spring": 0.7,   # Moderat im Frühling # von Zara
            "summer": 1.0,   # Peak im Sommer # von Zara
            "autumn": 0.6    # Niedriger im Herbst # von Zara
        }
        
        # Month to Season Mapping # von Zara
        self.SEASONAL_MONTH_MAPPING = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn"
        }
        
        # Temperatur-Konstanten # von Zara
        self.OPTIMAL_TEMPERATURE = 25.0  # Optimale Temperatur für PV # von Zara
        self.TEMP_EFFICIENCY_LOSS = 0.005  # Effizienz-Verlust pro °C über optimal # von Zara
        
        _LOGGER.debug("✅ WeatherCalculator initialisiert")
    
    def get_temperature_factor(self, temperature: float) -> float:
        """
        Berechnet Temperaturfaktor für Solar-Produktion.
        
        PV-Panels arbeiten optimal bei ~25°C.
        Bei höheren Temperaturen sinkt die Effizienz.
        
        Args:
            temperature: Temperatur in °C
            
        Returns:
            Faktor zwischen 0.7 und 1.0
        # von Zara
        """
        try:
            # Sehr kalte Temperaturen (unter 0°C) # von Zara
            if temperature < 0:
                return 0.7
            
            # Optimal bis 25°C (linearer Anstieg) # von Zara
            elif temperature <= self.OPTIMAL_TEMPERATURE:
                # Von 0.8 bei 0°C bis 1.0 bei 25°C # von Zara
                return 0.8 + (temperature / self.OPTIMAL_TEMPERATURE) * 0.2
            
            # Über optimal (Effizienz sinkt) # von Zara
            else:
                # Verlust von 0.5% pro °C über 25°C # von Zara
                factor = 1.0 - (temperature - self.OPTIMAL_TEMPERATURE) * self.TEMP_EFFICIENCY_LOSS
                return max(0.7, factor)  # Minimum 0.7 # von Zara
                
        except Exception as e:
            _LOGGER.warning(f"⚠️ Temperatur-Faktor Berechnung fehlgeschlagen: {e}")
            return 0.9  # Safe fallback # von Zara
    
    def get_cloud_factor(self, cloud_coverage: float) -> float:
        """
        Berechnet Cloud-Faktor für Solar-Produktion.
        
        Wolkenbedeckung reduziert direkte Sonneneinstrahlung erheblich.
        
        Args:
            cloud_coverage: Wolkenbedeckung in % (0-100)
            
        Returns:
            Faktor zwischen 0.2 und 1.0
        # von Zara
        """
        try:
            # Klar (< 20% Wolken) # von Zara
            if cloud_coverage < 20:
                return 1.0
            
            # Teilweise bewölkt (20-50%) # von Zara
            elif cloud_coverage < 50:
                return 0.8
            
            # Meist bewölkt (50-80%) # von Zara
            elif cloud_coverage < 80:
                return 0.4
            
            # Stark bewölkt (>= 80%) # von Zara
            else:
                return 0.2
                
        except Exception as e:
            _LOGGER.warning(f"⚠️ Cloud-Faktor Berechnung fehlgeschlagen: {e}")
            return 0.6  # Safe fallback # von Zara
    
    def get_seasonal_adjustment(self, now: datetime = None) -> float:
        """
        Berechnet saisonalen Anpassungsfaktor.
        
        Berücksichtigt Jahreszeit und Monat für realistische Anpassung.
        
        Args:
            now: Zeitpunkt (optional, default: jetzt)
            
        Returns:
            Seasonal adjustment factor (0.2 - 1.2)
        # von Zara
        """
        try:
            # Verwende aktuelle Zeit wenn nicht angegeben # von Zara
            if now is None:
                now = dt_util.utcnow()
            
            month = now.month
            
            # Hole Jahreszeit aus Monat # von Zara
            season = self.SEASONAL_MONTH_MAPPING.get(month, "autumn")
            
            # Base seasonal factor # von Zara
            factor = self.SEASONAL_FACTORS.get(season, 0.6)
            
            # Feinabstimmung für spezielle Monate # von Zara
            if month in [12, 1]:  # Tiefster Winter # von Zara
                factor *= 0.8
            elif month in [6, 7]:  # Hochsommer # von Zara
                factor *= 1.1
            
            # Safety bounds # von Zara
            return max(0.2, min(1.2, factor))
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Seasonal adjustment Berechnung fehlgeschlagen: {e}")
            return 0.6  # Safe fallback # von Zara
    
    def get_current_season(self) -> str:
        """
        Gibt aktuelle Jahreszeit zurück.
        
        Returns:
            Season name ("winter", "spring", "summer", "autumn")
        # von Zara
        """
        try:
            now = dt_util.utcnow()
            month = now.month
            return self.SEASONAL_MONTH_MAPPING.get(month, "autumn")
        except Exception:
            return "autumn"  # Safe fallback # von Zara
    
    def calculate_combined_weather_factor(
        self,
        weather_data: Dict[str, Any],
        include_seasonal: bool = True
    ) -> float:
        """
        Berechnet kombinierten Wetter-Faktor.
        
        Multipliziert Temperatur-, Cloud- und optional Seasonal-Faktor.
        
        Args:
            weather_data: Dictionary mit temperature und clouds
            include_seasonal: Ob seasonal adjustment einbezogen werden soll
            
        Returns:
            Kombinierter Faktor
        # von Zara
        """
        try:
            temp_factor = self.get_temperature_factor(
                weather_data.get("temperature", 15.0)
            )
            cloud_factor = self.get_cloud_factor(
                weather_data.get("clouds", 50.0)
            )
            
            combined = temp_factor * cloud_factor
            
            # Optional: Seasonal adjustment # von Zara
            if include_seasonal:
                seasonal = self.get_seasonal_adjustment()
                combined *= seasonal
            
            return combined
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Combined weather factor Berechnung fehlgeschlagen: {e}")
            return 0.5  # Safe fallback # von Zara
