"""
Rule-based Forecast Strategy (Fallback) for Solar Forecast ML.
Uses simple rules based on weather factors for predictions
when the ML model is unavailable or fails.

Copyright (C) 2025 Zara-Toorox

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
import logging
from typing import Any, Dict, Optional
from datetime import datetime
from homeassistant.util import dt as dt_util

# Import base strategy class and result dataclass
from .forecast_strategy import ForecastStrategy, ForecastResult
# Import weather calculator for rule factors
from .weather_calculator import WeatherCalculator
# Import exceptions if needed (No specific ones needed here currently)
# from .exceptions import ForecastException # Removed unused import

_LOGGER = logging.getLogger(__name__)


class RuleBasedForecastStrategy(ForecastStrategy):
    """
    A fallback forecast strategy that uses simple, rule-based heuristics
    derived from weather factors (temperature, clouds, condition) and a
    learned correction factor.
    """

    def __init__(self, weather_calculator: WeatherCalculator, solar_capacity: float):
        """
        Initialize the Rule-Based Forecast Strategy.

        Args:
            weather_calculator: An instance of WeatherCalculator to get rule-based factors.
            solar_capacity: The configured solar capacity (kWp) of the system.
        """
        super().__init__("rule_based") # Call base constructor
        self.weather_calculator = weather_calculator
        self.solar_capacity = solar_capacity

        # --- Heuristic Parameters ---
        self.BASE_DAILY_KWH_PER_KWP = 4.0
        self.TOMORROW_DISCOUNT_FACTOR = 0.92
        self.MAX_REALISTIC_DAILY_KWH_PER_KWP = 8.0

        _LOGGER.debug("RuleBasedForecastStrategy initialized.")


    def is_available(self) -> bool:
        """This strategy is always available as a fallback."""
        return True

    def get_priority(self) -> int:
        """
        Returns the priority of this strategy. Lower than ML strategy.
        """
        return 50 # Medium-low priority, used if ML is unavailable

    async def calculate_forecast(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any], # Contains solar_capacity if needed
        correction_factor: float # The learned fallback correction factor
    ) -> ForecastResult:
        """
        Calculates a forecast using simple weather rules combined with a learned
        correction factor.

        Args:
            weather_data: Dictionary containing current weather data.
            sensor_data: Dictionary containing other relevant data (e.g., 'solar_capacity').
            correction_factor: A learned multiplier to adjust the rule-based estimate.

        Returns:
            A ForecastResult object. Includes an emergency fallback if calculation fails.
        """
        _LOGGER.debug("Calculating forecast using Rule-based strategy...")

        try:
            # Determine solar capacity
            try:
                 base_capacity_kwp = float(sensor_data.get("solar_capacity", self.solar_capacity))
                 if base_capacity_kwp <= 0:
                      _LOGGER.warning(f"Solar capacity ({base_capacity_kwp}kWp) is zero or negative. Using fallback 1.0 kWp.")
                      base_capacity_kwp = 1.0
            except (ValueError, TypeError):
                 _LOGGER.warning(f"Invalid solar_capacity in sensor_data, using default {self.solar_capacity} kWp.")
                 base_capacity_kwp = self.solar_capacity


            # --- Calculate Combined Weather Factor ---
            combined_weather_factor = self.weather_calculator.calculate_combined_weather_factor(
                weather_data,
                include_seasonal=True
            )
            _LOGGER.debug(f"Combined weather factor: {combined_weather_factor:.3f}")

            # --- Calculate Base Production Estimate ---
            base_daily_production_kwh = base_capacity_kwp * self.BASE_DAILY_KWH_PER_KWP

            # --- Apply Adjustments ---
            adjusted_today_forecast_kwh = (
                base_daily_production_kwh
                * combined_weather_factor
                * correction_factor
            )
            adjusted_tomorrow_forecast_kwh = adjusted_today_forecast_kwh * self.TOMORROW_DISCOUNT_FACTOR

            # --- Apply Realistic Bounds ---
            min_forecast_kwh = 0.0
            max_realistic_kwh = base_capacity_kwp * self.MAX_REALISTIC_DAILY_KWH_PER_KWP
            today_forecast_kwh = max(min_forecast_kwh, min(adjusted_today_forecast_kwh, max_realistic_kwh))
            tomorrow_forecast_kwh = max(min_forecast_kwh, min(adjusted_tomorrow_forecast_kwh, max_realistic_kwh))

            # --- Mindest-Prognose Check: Prognose muss >= aktuellem Ertrag sein ---
            try:
                current_yield = sensor_data.get("current_yield")
                if current_yield is not None and current_yield > 0:
                    current_yield_float = float(current_yield)
                    
                    if current_yield_float > today_forecast_kwh:
                        # Berechne verbleibende Produktionsstunden
                        now = dt_util.now()
                        current_hour = now.hour
                        
                        # Schätze Sonnenuntergang (vereinfacht: 18:00 Uhr lokale Zeit)
                        # TODO: Könnte später durch sun_guard.get_production_hours() ersetzt werden
                        sunset_hour = 18
                        sunrise_hour = 6
                        total_production_hours = sunset_hour - sunrise_hour
                        remaining_hours = max(0, sunset_hour - current_hour)
                        
                        # Schätze zusätzliche Produktion basierend auf verbleibenden Stunden
                        if total_production_hours > 0 and remaining_hours > 0:
                            # Anteil der verbleibenden Stunden an gesamten Produktionsstunden
                            remaining_fraction = remaining_hours / total_production_hours
                            # Verwende den ursprünglichen Forecast als Basis für die Schätzung
                            additional_forecast = today_forecast_kwh * remaining_fraction
                        else:
                            # Fallback: Füge 10% des ursprünglichen Forecasts hinzu
                            additional_forecast = today_forecast_kwh * 0.1
                        
                        # Neue Prognose: Aktueller Ertrag + erwartete zusätzliche Produktion
                        adjusted_today_forecast = current_yield_float + additional_forecast
                        
                        _LOGGER.info(
                            f"Mindest-Prognose Anpassung: Aktueller Ertrag {current_yield_float:.2f} kWh > "
                            f"Ursprüngliche Prognose {today_forecast_kwh:.2f} kWh. "
                            f"Angepasst auf {adjusted_today_forecast:.2f} kWh "
                            f"(+{additional_forecast:.2f} kWh für verbleibende {remaining_hours}h)"
                        )
                        
                        # Speichere originale Werte für proportionale Anpassung
                        original_today_forecast = today_forecast_kwh
                        today_forecast_kwh = adjusted_today_forecast
                        
                        # Passe auch tomorrow_forecast proportional an
                        if original_today_forecast > 0:
                            adjustment_ratio = today_forecast_kwh / original_today_forecast
                            tomorrow_forecast_kwh = tomorrow_forecast_kwh * adjustment_ratio
                            
            except Exception as e:
                _LOGGER.debug(f"Mindest-Prognose Check konnte nicht durchgeführt werden: {e}")
                # Fehler beim Check sollte die normale Prognose nicht beeinträchtigen

            # --- Determine Confidence ---
            correction_deviation = abs(1.0 - correction_factor)
            confidence_base = max(0.0, 1.0 - correction_deviation * 0.5)
            confidence_weather_penalty = 1.0 if combined_weather_factor > 0.2 else 0.7
            confidence_today_raw = confidence_base * confidence_weather_penalty * 85.0
            confidence_today = max(30.0, min(95.0, confidence_today_raw))
            confidence_tomorrow = confidence_today * 0.9


            # --- Create Result Object ---
            result = ForecastResult(
                forecast_today=today_forecast_kwh,
                forecast_tomorrow=tomorrow_forecast_kwh,
                confidence_today=confidence_today,
                confidence_tomorrow=confidence_tomorrow,
                method="rule_based_corrected",
                calibrated=True,
                base_capacity=base_capacity_kwp,
                correction_factor=correction_factor,
            )

            _LOGGER.info(
                f"Rule-based Forecast successful: Today={result.forecast_today:.2f} kWh, "
                f"Tomorrow={result.forecast_tomorrow:.2f} kWh, "
                f"Confidence={result.confidence_today:.1f}%, "
                f"(WeatherFactor={combined_weather_factor:.2f}, CorrectionFactor={correction_factor:.2f})"
            )

            return result

        except Exception as e:
            _LOGGER.error(f"Rule-based forecast calculation failed unexpectedly: {e}", exc_info=True)

            # --- Emergency Fallback ---
            _LOGGER.warning("Using emergency fallback for Rule-based forecast.")
            fallback_capacity = self.solar_capacity if self.solar_capacity > 0 else 2.0
            emergency_yield = fallback_capacity * 1.5

            return ForecastResult(
                forecast_today=emergency_yield,
                forecast_tomorrow=emergency_yield * 0.9,
                confidence_today=20.0,
                confidence_tomorrow=15.0,
                method="emergency_fallback_rule",
                calibrated=False,
                base_capacity=fallback_capacity
            )