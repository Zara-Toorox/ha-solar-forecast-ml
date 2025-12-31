# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from ..astronomy.astronomy_cache_manager import get_cache_manager
from ..const import ML_MODEL_VERSION
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..ai import AIPredictor
from ..services.service_error_handler import ErrorHandlingService
from .forecast_rule_based_strategy import RuleBasedForecastStrategy
from .forecast_strategy_base import ForecastResult
from .forecast_weather_calculator import WeatherCalculator

_LOGGER = logging.getLogger(__name__)

class ForecastOrchestrator:
    """Selects and executes the most appropriate forecast strategy based on"""

    FALLBACK_PRODUCTION_START_HOUR = 6
    FALLBACK_PRODUCTION_END_HOUR = 21

    def __init__(
        self,
        hass: HomeAssistant,
        data_manager: Any,
        solar_capacity: float,
        weather_calculator: WeatherCalculator,
        panel_groups: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize the ForecastOrchestrator"""
        self.hass = hass
        self.data_manager = data_manager
        self.solar_capacity = solar_capacity
        self.weather_calculator = weather_calculator
        self.panel_groups = panel_groups or []

        self.rule_based_strategy: Optional[RuleBasedForecastStrategy] = None

        self._ai_predictor: Optional[AIPredictor] = None
        self._historical_cache: Dict = {}

        self.active_strategy_name: Optional[str] = None

        if self.panel_groups:
            _LOGGER.info(
                f"ForecastOrchestrator initialized with {len(self.panel_groups)} panel groups."
            )
        else:
            _LOGGER.debug("ForecastOrchestrator initialized.")

    def is_production_hour(self, target_dt: datetime) -> bool:
        """Checks if a given datetime is within realistic solar production hours @zara"""

        target_dt_local = dt_util.as_local(target_dt)

        try:
            cache_manager = get_cache_manager()
            if cache_manager.is_loaded():

                date_str = target_dt_local.date().isoformat()
                window = cache_manager.get_production_window(date_str)

                if window:
                    window_start_str, window_end_str = window

                    window_start = datetime.fromisoformat(window_start_str).replace(tzinfo=None)
                    window_end = datetime.fromisoformat(window_end_str).replace(tzinfo=None)

                    target_dt_naive = target_dt_local.replace(tzinfo=None)

                    if window_start <= target_dt_naive <= window_end:

                        return True
                    else:

                        return False

        except Exception as e:
            _LOGGER.debug(f"Astronomy cache access failed, using fallback: {e}")

        hour = target_dt_local.hour
        month = target_dt_local.month

        if month in [11, 12, 1, 2]:
            is_production = 6 <= hour <= 17
        elif month in [5, 6, 7, 8]:
            is_production = 4 <= hour <= 21
        else:
            is_production = 5 <= hour <= 19

        return is_production

    def initialize_strategies(
        self,
        ai_predictor: Optional[AIPredictor] = None,
        error_handler: Optional[ErrorHandlingService] = None,
    ) -> None:
        """Initializes the forecast strategy with AI @zara"""
        _LOGGER.info("Initializing forecast strategies...")

        self._ai_predictor = ai_predictor

        if self._ai_predictor:
            self._historical_cache = self._ai_predictor._historical_cache

        try:
            self.rule_based_strategy = RuleBasedForecastStrategy(
                weather_calculator=self.weather_calculator,
                solar_capacity=self.solar_capacity,
                orchestrator=self,
                panel_groups=self.panel_groups,
                ai_predictor=ai_predictor,
            )
            if ai_predictor and ai_predictor.is_ready():
                _LOGGER.info("Forecast strategy initialized with local AI")
            elif self.panel_groups:
                _LOGGER.info(
                    f"Forecast strategy initialized with {len(self.panel_groups)} panel groups (physics)"
                )
            else:
                _LOGGER.info("Forecast strategy initialized (physics fallback)")

        except Exception as e:
            _LOGGER.error(f"Failed to initialize forecast strategy: {e}", exc_info=True)
            self.rule_based_strategy = None

    async def orchestrate_forecast(
        self,
        current_weather: Optional[Dict[str, Any]] = None,
        hourly_forecast: Optional[List[Dict[str, Any]]] = None,
        external_sensors: Optional[Dict[str, Any]] = None,
        historical_avg: Optional[float] = None,
        ml_prediction_today: Optional[float] = None,
        ml_prediction_tomorrow: Optional[float] = None,
        correction_factor: float = 1.0,
    ) -> Dict[str, Any]:
        """Orchestrates forecast creation with all available data"""
        hourly_weather_forecast = hourly_forecast if hourly_forecast else []
        sensor_data = external_sensors if external_sensors else {}

        if "current_yield" not in sensor_data and current_weather:
            sensor_data["current_yield"] = 0.0

        return await self.create_forecast(
            hourly_weather_forecast=hourly_weather_forecast,
            sensor_data=sensor_data,
            correction_factor=correction_factor,
        )

    async def create_forecast(
        self,
        hourly_weather_forecast: List[Dict[str, Any]],
        sensor_data: Dict[str, Any],
        correction_factor: float = 1.0,
    ) -> Dict[str, Any]:
        """Creates daily solar forecast using AI with physics fallback @zara"""
        _LOGGER.debug("Creating forecast (AI + Physics)...")

        result: Optional[ForecastResult] = None

        # Get lag features for prediction
        lag_features = {}
        try:
            now_local = dt_util.now()
            yesterday_dt = now_local - timedelta(days=1)
            yesterday_key = yesterday_dt.date().isoformat()
            yesterday_total_kwh = self._historical_cache.get("daily_productions", {}).get(
                yesterday_key, 0.0
            )
            lag_features["production_yesterday"] = float(yesterday_total_kwh)
            _LOGGER.debug(f"Lag-Feature 'production_yesterday' = {yesterday_total_kwh:.2f} kWh")
        except Exception as e:
            _LOGGER.warning(f"Could not retrieve lag feature 'production_yesterday': {e}")
            lag_features["production_yesterday"] = 0.0

        # Use single unified strategy (TinyLSTM + Physics)
        if self.rule_based_strategy and self.rule_based_strategy.is_available():
            try:
                result = await self.rule_based_strategy.calculate_forecast(
                    hourly_weather_forecast=hourly_weather_forecast,
                    sensor_data=sensor_data,
                    correction_factor=correction_factor,
                    lag_features=lag_features,
                )

                _LOGGER.info(
                    f"Forecast complete: Today={result.forecast_today:.2f} kWh, "
                    f"Tomorrow={result.forecast_tomorrow:.2f} kWh, "
                    f"Day After={result.forecast_day_after_tomorrow:.2f} kWh, "
                    f"Method={result.method}"
                )

            except Exception as err:
                _LOGGER.error(f"Forecast calculation failed: {err}", exc_info=True)
                result = None
        else:
            _LOGGER.error("Forecast strategy not available")
            result = None

        # Handle failure case - return empty forecast
        if not result:
            _LOGGER.critical("Emergency Fallback: Forecast failed. Returning zeros.")
            return {
                "today": 0.0,
                "tomorrow": 0.0,
                "day_after_tomorrow": 0.0,
                "peak_time": "12:00",
                "confidence": 0.0,
                "method": "fallback_empty",
                "model_accuracy": None,
                "best_hour": None,
                "best_hour_kwh": None,
                "hourly": [],
                "today_raw": None,
                "safeguard_applied": False,
            }

        # Extract values from result
        best_hour = result.best_hour_today
        best_hour_kwh = result.best_hour_production_kwh
        hourly_values = result.hourly_values or []

        return {
            "today": round(result.forecast_today, 2),
            "tomorrow": round(result.forecast_tomorrow, 2),
            "day_after_tomorrow": round(result.forecast_day_after_tomorrow, 2),
            "peak_time": "12:00",
            "confidence": round(result.confidence_today, 1),
            "method": result.method,
            "model_accuracy": result.model_accuracy,
            "best_hour": best_hour,
            "best_hour_kwh": round(best_hour_kwh, 3) if best_hour_kwh is not None else None,
            "hourly": hourly_values,
            "today_raw": round(result.forecast_today_raw, 2) if result.forecast_today_raw is not None else None,
            "safeguard_applied": result.safeguard_applied_today,
        }

    def calculate_next_hour_prediction(
        self,
        forecast_today_kwh: float,
        weather_data: Optional[Dict[str, Any]] = None,
        sensor_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Estimates the solar production for the next full hour"""
        _LOGGER.debug("Calculating next hour prediction...")
        try:
            now_local = dt_util.now()
            target_dt_local = now_local + timedelta(hours=1)
            target_hour = target_dt_local.hour

            if not self.is_production_hour(target_dt_local):
                if sensor_data and sensor_data.get("lux") is not None:
                    if sensor_data["lux"] < 500:
                        _LOGGER.debug(
                            f"Target hour {target_hour} has insufficient light (lux={sensor_data['lux']}). "
                            f"Next hour prediction is 0.0 kWh."
                        )
                        return 0.0
                _LOGGER.debug(
                    f"Target hour {target_hour} (local) is outside production hours. Next hour prediction is 0.0 kWh."
                )
                return 0.0

            hourly_base_kwh: Optional[float] = self._get_hourly_profile_base(
                forecast_today_kwh, target_hour
            )

            if hourly_base_kwh is not None:
                base_kwh = hourly_base_kwh
                _LOGGER.debug(
                    f"Using profile base for hour {target_hour} (local): {base_kwh:.3f} kWh."
                )
            else:
                base_kwh = forecast_today_kwh / 10.0
                _LOGGER.warning(
                    f"Hourly profile base unavailable for hour {target_hour}. "
                    f"Using simple fallback base: {base_kwh:.3f} kWh."
                )

            adjustment_factors = self._get_realtime_adjustment_factors(weather_data, sensor_data)

            adjusted_kwh = base_kwh
            factors_log = []
            for factor_name, factor_value in adjustment_factors.items():
                adjusted_kwh *= factor_value
                factors_log.append(f"{factor_name}={factor_value:.2f}")

            max_hourly_kwh = self.solar_capacity * 1.2
            adjusted_kwh = min(adjusted_kwh, max_hourly_kwh)

            final_prediction_kwh = max(0.0, adjusted_kwh)

            _LOGGER.info(
                f"Next Hour ({target_hour:02d}:00 local) Prediction: Base={base_kwh:.3f} kWh * "
                f"Adjustments [{', '.join(factors_log)}] -> Final={final_prediction_kwh:.3f} kWh"
            )

            return round(final_prediction_kwh, 3)

        except Exception as e:
            _LOGGER.error(f"Next hour prediction calculation failed: {e}", exc_info=True)
            return 0.0

    def _get_hourly_profile_base(
        self, forecast_today_kwh: float, target_hour: int
    ) -> Optional[float]:
        """Calculates the base production for a specific hour using solar profile @zara"""
        # Use simple solar curve approximation based on hour
        # Production hours typically 6-20 with peak around 12
        if target_hour < 6 or target_hour > 20:
            return 0.0

        # Gaussian-like solar curve centered at 12:00
        peak_hour = 12
        spread = 4.0  # Standard deviation-like spread
        hour_weight = math.exp(-((target_hour - peak_hour) ** 2) / (2 * spread ** 2))

        # Normalize across production hours
        total_weight = sum(
            math.exp(-((h - peak_hour) ** 2) / (2 * spread ** 2))
            for h in range(6, 21)
        )

        if total_weight <= 0:
            return None

        hour_fraction = hour_weight / total_weight
        hourly_base_kwh = forecast_today_kwh * hour_fraction

        _LOGGER.debug(
            f"Hourly base calculation: Hour={target_hour}, "
            f"Fraction={hour_fraction:.4f} -> Base={hourly_base_kwh:.3f} kWh"
        )
        return hourly_base_kwh

    def _get_realtime_adjustment_factors(
        self,
        current_weather_data: Optional[Dict[str, Any]],
        current_sensor_data: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculates adjustment multipliers based on current weather and sensor readings"""
        factors = {
            "cloud/lux": 1.0,
            "temperature": 1.0,
            "rain": 1.0,
        }
        _LOGGER.debug("Calculating real-time adjustment factors...")

        lux_value = current_sensor_data.get("lux") if current_sensor_data else None
        cloud_value = (
            current_weather_data.get("cloud_cover", current_weather_data.get("clouds"))
            if current_weather_data
            else None
        )

        if lux_value is not None and lux_value >= 0:
            typical_bright_lux = 60000.0
            if lux_value < 1000:
                factors["cloud/lux"] = 0.1
            elif lux_value < 20000:
                factors["cloud/lux"] = 0.1 + (lux_value / 20000.0) * 0.6
            else:
                factors["cloud/lux"] = 0.7 + min((lux_value / typical_bright_lux) * 0.5, 0.5)
            _LOGGER.debug(
                f"Using Lux value ({lux_value} lx) for adjustment factor: {factors['cloud/lux']:.2f}"
            )

        elif cloud_value is not None:
            cloud_factor = self.weather_calculator.get_cloud_factor(cloud_value)
            reference_cloud_factor = 0.65
            factors["cloud/lux"] = (
                cloud_factor / reference_cloud_factor if reference_cloud_factor > 0 else 1.0
            )
            factors["cloud/lux"] = max(0.1, min(1.2, factors["cloud/lux"]))
            _LOGGER.debug(
                f"Using Cloud Cover ({cloud_value}%) for adjustment. BaseFactor={cloud_factor:.2f} -> AdjFactor={factors['cloud/lux']:.2f}"
            )
        else:
            _LOGGER.debug("No Lux or Cloud data available for real-time adjustment.")

        temp_value = current_sensor_data.get("temperature") if current_sensor_data else None
        if temp_value is None and current_weather_data:
            temp_value = current_weather_data.get("temperature")

        if temp_value is not None:
            temp_factor = self.weather_calculator.get_temperature_factor(temp_value)
            factors["temperature"] = max(0.7, min(1.1, temp_factor))
            _LOGGER.debug(
                f"Using Temperature ({temp_value}C) for adjustment factor: {factors['temperature']:.2f}"
            )
        else:
            _LOGGER.debug("No Temperature data available for real-time adjustment.")

        rain_value = current_sensor_data.get("rain") if current_sensor_data else None

        if rain_value is not None and rain_value > 0:
            if rain_value > 5.0:
                factors["rain"] = 0.2
            elif rain_value > 1.0:
                factors["rain"] = 0.5
            else:
                factors["rain"] = 0.8
            _LOGGER.debug(
                f"Using Rain value ({rain_value}) for adjustment factor: {factors['rain']:.2f}"
            )

        for key in factors:
            factors[key] = max(0.0, min(1.5, factors[key]))

        _LOGGER.debug(f"Final real-time adjustment factors: {factors}")
        return factors
