# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Data schemas and type definitions @zara"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class ForecastSummary:
    """Daily forecast summary @zara"""
    date: str
    prediction_kwh: float
    prediction_kwh_raw: Optional[float] = None
    safeguard_applied: bool = False
    safeguard_reduction_kwh: float = 0.0
    locked: bool = False
    locked_at: Optional[datetime] = None
    source: Optional[str] = None


@dataclass
class HourlyPrediction:
    """Hourly prediction data @zara"""
    target_date: str
    target_hour: int
    prediction_kwh: float
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DailySummary:
    """Daily summary statistics @zara"""
    date: str
    yield_kwh: float
    forecast_kwh: Optional[float] = None
    accuracy_percent: Optional[float] = None
    production_hours: Optional[float] = None
    peak_power_w: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeatherData:
    """Weather data point @zara"""
    datetime: datetime
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    cloud_cover: Optional[float] = None
    wind_speed: Optional[float] = None
    precipitation: Optional[float] = None
    pressure: Optional[float] = None
    ghi: Optional[float] = None
    visibility_m: Optional[float] = None


@dataclass
class AstronomyData:
    """Astronomy calculation data @zara"""
    datetime: datetime
    sun_elevation: float
    sun_azimuth: float
    dni: Optional[float] = None
    ghi: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
