# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Weather types and constants for Weather Expert System @zara"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class CloudType(Enum):
    """Cloud type classification @zara"""
    CLEAR = "clear"
    CIRRUS = "cirrus"
    FAIR = "fair"
    MIXED = "mixed"
    STRATUS = "stratus"
    OVERCAST = "overcast"
    SNOW = "snow"
    FOG = "fog"
    FOG_LIGHT = "fog_light"


# Cloud classification thresholds @zara
LAYER_THRESHOLD_DOMINANT = 50.0
LAYER_THRESHOLD_LOW = 20.0
LAYER_THRESHOLD_CLEAR = 25.0

# Visibility thresholds in meters @zara
VISIBILITY_FOG_THRESHOLD = 1000
VISIBILITY_FOG_LIGHT_THRESHOLD = 5000

# Default expert weights per cloud type @zara
DEFAULT_EXPERT_WEIGHTS: dict[str, dict[str, float]] = {
    CloudType.CLEAR.value: {
        "open_meteo": 0.15,
        "wttr_in": 0.10,
        "ecmwf_layers": 0.15,
        "bright_sky": 0.35,
        "pirate_weather": 0.25,
    },
    CloudType.CIRRUS.value: {
        "open_meteo": 0.10,
        "wttr_in": 0.10,
        "ecmwf_layers": 0.30,
        "bright_sky": 0.30,
        "pirate_weather": 0.20,
    },
    CloudType.FAIR.value: {
        "open_meteo": 0.15,
        "wttr_in": 0.10,
        "ecmwf_layers": 0.15,
        "bright_sky": 0.35,
        "pirate_weather": 0.25,
    },
    CloudType.MIXED.value: {
        "open_meteo": 0.15,
        "wttr_in": 0.10,
        "ecmwf_layers": 0.15,
        "bright_sky": 0.30,
        "pirate_weather": 0.30,
    },
    CloudType.STRATUS.value: {
        "open_meteo": 0.10,
        "wttr_in": 0.10,
        "ecmwf_layers": 0.10,
        "bright_sky": 0.40,
        "pirate_weather": 0.30,
    },
    CloudType.OVERCAST.value: {
        "open_meteo": 0.10,
        "wttr_in": 0.10,
        "ecmwf_layers": 0.10,
        "bright_sky": 0.40,
        "pirate_weather": 0.30,
    },
    CloudType.SNOW.value: {
        "open_meteo": 0.20,
        "wttr_in": 0.20,
        "ecmwf_layers": 0.20,
        "bright_sky": 0.20,
        "pirate_weather": 0.20,
    },
    CloudType.FOG.value: {
        "open_meteo": 0.10,
        "wttr_in": 0.05,
        "ecmwf_layers": 0.05,
        "bright_sky": 0.50,
        "pirate_weather": 0.30,
    },
    CloudType.FOG_LIGHT.value: {
        "open_meteo": 0.15,
        "wttr_in": 0.10,
        "ecmwf_layers": 0.10,
        "bright_sky": 0.40,
        "pirate_weather": 0.25,
    },
}

# Transmission coefficients for cloud-to-transmission conversion @zara
TRANSMISSION_TAU_BY_CLOUD_TYPE: dict[str, float] = {
    CloudType.CLEAR.value: 1.00,
    CloudType.CIRRUS.value: 0.80,
    CloudType.FAIR.value: 0.50,
    CloudType.MIXED.value: 0.35,
    CloudType.STRATUS.value: 0.20,
    CloudType.OVERCAST.value: 0.10,
    CloudType.SNOW.value: 0.02,
    CloudType.FOG.value: 0.50,
    CloudType.FOG_LIGHT.value: 0.70,
}


def cloud_to_transmission(
    cloud_percent: float,
    cloud_type: CloudType = CloudType.MIXED
) -> float:
    """Convert cloud cover to solar transmission @zara"""
    tau = TRANSMISSION_TAU_BY_CLOUD_TYPE.get(cloud_type.value, 0.35)
    transmission = 100.0 * (1.0 - (cloud_percent / 100.0) * (1.0 - tau))
    return round(max(0.0, min(100.0, transmission)), 1)


# API settings @zara
BRIGHT_SKY_BASE_URL = "https://api.brightsky.dev/weather"
BRIGHT_SKY_TIMEOUT = 15
BRIGHT_SKY_CACHE_HOURS = 3

PIRATE_WEATHER_BASE_URL = "https://api.pirateweather.net/forecast"
PIRATE_WEATHER_TIMEOUT = 15
PIRATE_WEATHER_CACHE_HOURS = 3

WTTR_IN_BASE_URL = "https://wttr.in"
WTTR_IN_TIMEOUT = 15
WTTR_IN_CACHE_HOURS = 6

# Learning parameters @zara
LEARNING_SMOOTHING_FACTOR = 0.3
LEARNING_MIN_ERROR = 5.0
LEARNING_MIN_HOURS = 4
LEARNING_ACCELERATED_THRESHOLD = 35.0
LEARNING_ACCELERATED_FACTOR = 0.5


@dataclass
class ExpertForecast:
    """Forecast from single expert @zara"""
    expert_name: str
    cloud_cover: float
    confidence: float = 1.0
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlendedHourForecast:
    """Blended hourly forecast @zara"""
    date: str
    hour: int
    cloud_cover: float
    cloud_type: CloudType
    expert_forecasts: dict[str, float]
    blend_weights: dict[str, float]
    cloud_cover_low: Optional[float] = None
    cloud_cover_mid: Optional[float] = None
    cloud_cover_high: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    precipitation: Optional[float] = None
    wind_speed: Optional[float] = None
    pressure: Optional[float] = None
    ghi: Optional[float] = None
    direct_radiation: Optional[float] = None
    diffuse_radiation: Optional[float] = None
    visibility_m: Optional[float] = None
    fog_detected: bool = False
    fog_type: Optional[str] = None
