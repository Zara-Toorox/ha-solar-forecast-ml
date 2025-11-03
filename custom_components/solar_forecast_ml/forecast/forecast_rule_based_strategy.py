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
import math
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta

from homeassistant.util import dt as dt_util

from .strategy import ForecastStrategy, ForecastResult
from .weather_calculator import WeatherCalculator
from ..core.helpers import SafeDateTimeUtil as dt_util_safe


_LOGGER = logging.getLogger(__name__)


class RuleBasedForecastStrategy(ForecastStrategy):
    """
    A fallback forecast strategy that uses simple, rule-based heuristics
    derived from weather factors (temperature, clouds, condition) and a
    learned correction factor.
    
    Berechnet eine iterative stÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ndliche Prognose fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r alle 24 Stunden.
    """

    def __init__(
        self, 
        weather_calculator: WeatherCalculator, 
        solar_capacity: float
    ):
        """
        Initialize the Rule-Based Forecast Strategy.

        Args:
            weather_calculator: An instance of WeatherCalculator to get rule-based factors.
            solar_capacity: The configured solar capacity (kWp) of the system.
        """
        super().__init__("rule_based")
        self.weather_calculator = weather_calculator
        self.solar_capacity = solar_capacity

        self.PEAK_KW_PER_KWP = 0.95 
        self.TOMORROW_DISCOUNT_FACTOR = 0.92
        self.MAX_REALISTIC_DAILY_KWH_PER_KWP = 8.0

        _LOGGER.debug("RuleBasedForecastStrategy (Iterative) initialized.")


    def is_available(self) -> bool:
        """This strategy is always available as a fallback."""
        return True

    def get_priority(self) -> int:
        """
        Returns the priority of this strategy. Lower than ML strategy.
        """
        return 50

    async def calculate_forecast(
        self,
        hourly_weather_forecast: List[Dict[str, Any]],
        sensor_data: Dict[str, Any],
        correction_factor: float,
        **kwargs
    ) -> ForecastResult:
        """
        Calculates a forecast using simple weather rules combined with a learned
        correction factor, iteriert ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ber die stÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ndliche Wettervorhersage.

        Args:
            hourly_weather_forecast: Liste der verarbeiteten stÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ndlichen Wettervorhersagen.
            sensor_data: Dictionary containing other relevant data (e.g., 'solar_capacity').
            correction_factor: A learned multiplier to adjust the rule-based estimate.
            **kwargs: Ignoriert 'lag_features'.

        Returns:
            A ForecastResult object.
        """
        _LOGGER.debug("Calculating forecast using Rule-based (Iterative) strategy...")

        try:
            try:
                 base_capacity_kwp = float(sensor_data.get("solar_capacity", self.solar_capacity))
                 if base_capacity_kwp <= 0:
                      _LOGGER.warning(f"Solar capacity ({base_capacity_kwp}kWp) is zero or negative. Using fallback 1.0 kWp.")
                      base_capacity_kwp = 1.0
            except (ValueError, TypeError):
                 _LOGGER.warning(f"Invalid solar_capacity in sensor_data, using default {self.solar_capacity} kWp.")
                 base_capacity_kwp = self.solar_capacity

            total_today_kwh = 0.0
            total_tomorrow_kwh = 0.0
            
            now_local = dt_util_safe.as_local(dt_util_safe.utcnow())
            today_date = now_local.date()
            tomorrow_date = today_date + timedelta(days=1)

            for hour_data in hourly_weather_forecast:
                try:
                    hour_dt_local = hour_data.get("local_datetime")
                    if not hour_dt_local:
                        _LOGGER.warning("Skipping hour, missing 'local_datetime'")
                        continue
                    
                    # Parse if string (ISO format from JSON serialization)
                    if isinstance(hour_dt_local, str):
                        hour_dt_local = dt_util_safe.parse_datetime(hour_dt_local)
                        if not hour_dt_local:
                            _LOGGER.warning("Skipping hour, invalid 'local_datetime' format")
                            continue
                        
                    hour_date = hour_dt_local.date()
                    hour_local = hour_dt_local.hour
                    
                    hour_factor = self._get_hour_factor(hour_local)
                    
                    combined_weather_factor = self.weather_calculator.calculate_combined_weather_factor(
                        hour_data,
                        include_seasonal=True
                    )
                    
                    hourly_kwh = (
                        base_capacity_kwp
                        * self.PEAK_KW_PER_KWP
                        * hour_factor
                        * combined_weather_factor
                        * correction_factor
                    )
                    
                    hourly_kwh = max(0.0, hourly_kwh)

                    if hour_date == today_date:
                        total_today_kwh += hourly_kwh
                    elif hour_date == tomorrow_date:
                        total_tomorrow_kwh += hourly_kwh
                        
                except Exception as e_inner:
                    _LOGGER.warning(f"Failed to process hour {hour_data.get('local_hour')}: {e_inner}")
                    continue
            
            _LOGGER.debug(f"Rule-based iteration complete. Today (raw): {total_today_kwh:.2f} kWh, Tomorrow (raw): {total_tomorrow_kwh:.2f} kWh")

            min_forecast_kwh = 0.0
            max_realistic_kwh = base_capacity_kwp * self.MAX_REALISTIC_DAILY_KWH_PER_KWP
            
            today_forecast_kwh = max(min_forecast_kwh, min(total_today_kwh, max_realistic_kwh))
            tomorrow_forecast_kwh = max(min_forecast_kwh, min(total_tomorrow_kwh, max_realistic_kwh))

            try:
                current_yield = sensor_data.get("current_yield")
                if current_yield is not None and current_yield > 0:
                    current_yield_float = float(current_yield)
                    
                    if current_yield_float > today_forecast_kwh:
                        remaining_hours = max(0, 21 - now_local.hour)
                        total_production_hours = 15
                        
                        additional_forecast = 0.0
                        if total_production_hours > 0 and remaining_hours > 0:
                            remaining_fraction = remaining_hours / total_production_hours
                            additional_forecast = today_forecast_kwh * remaining_fraction
                        else:
                            additional_forecast = today_forecast_kwh * 0.1
                        
                        adjusted_today_forecast = current_yield_float + additional_forecast
                        
                        _LOGGER.info(
                            f"Mindest-Prognose Anpassung (Regel): Aktueller Ertrag {current_yield_float:.2f} kWh > "
                            f"UrsprÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ngliche Prognose {today_forecast_kwh:.2f} kWh. "
                            f"Angepasst auf {adjusted_today_forecast:.2f} kWh."
                        )
                        
                        original_today_forecast = today_forecast_kwh
                        today_forecast_kwh = adjusted_today_forecast
                        
                        if original_today_forecast > 0:
                            adjustment_ratio = today_forecast_kwh / original_today_forecast
                            tomorrow_forecast_kwh = tomorrow_forecast_kwh * adjustment_ratio
                            
            except Exception as e:
                _LOGGER.debug(f"Mindest-Prognose Check (Regel) konnte nicht durchgefÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼hrt werden: {e}")

            correction_deviation = abs(1.0 - correction_factor)
            confidence_base = max(0.0, 1.0 - correction_deviation * 0.5)
            confidence_today = max(30.0, min(95.0, confidence_base * 85.0))
            confidence_tomorrow = confidence_today * 0.9


            result = ForecastResult(
                forecast_today=today_forecast_kwh,
                forecast_tomorrow=tomorrow_forecast_kwh,
                confidence_today=confidence_today,
                confidence_tomorrow=confidence_tomorrow,
                method="rule_based_iterative",
                calibrated=True,
                base_capacity=base_capacity_kwp,
                correction_factor=correction_factor,
            )

            _LOGGER.info(
                f"Rule-based (Iterative) Forecast successful: Today={result.forecast_today:.2f} kWh, "
                f"Tomorrow={result.forecast_tomorrow:.2f} kWh, "
                f"Confidence={result.confidence_today:.1f}%, "
                f"(CorrectionFactor={correction_factor:.2f})"
            )

            return result

        except Exception as e:
            _LOGGER.error(f"Rule-based (Iterative) forecast calculation failed unexpectedly: {e}", exc_info=True)

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

    def _get_hour_factor(self, hour: int) -> float:
        """
        Berechnet einen Faktor (0.0 bis 1.0) basierend auf der Sonnenkurve (Sinus)
        fÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼r eine gegebene Stunde. Nachtstunden (22-5) erhalten Faktor 0.
        """
        try:
            if hour >= 22 or hour <= 5:
                return 0.0
            
            start_hour = 6
            end_hour = 21
            total_duration = end_hour - start_hour
            
            if total_duration <= 0:
                return 0.0
            
            hour_pos = (hour + 0.5 - start_hour) / total_duration
            
            if not (0.0 <= hour_pos <= 1.0):
                return 0.0
            
            factor = math.sin(hour_pos * math.pi)
            return max(0.0, factor)
            
        except Exception:
            return 0.5
