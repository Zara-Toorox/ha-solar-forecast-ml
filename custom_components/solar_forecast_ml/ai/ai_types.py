# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class HourlyProfile:
    """Hourly production profile @zara"""
    hourly_averages: Dict[str, float] = field(default_factory=dict)
    total_samples: int = 0
    last_updated: Optional[str] = None


@dataclass
class LearnedWeights:
    """Model weights container @zara"""
    weights: Dict[str, Any] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)
    version: str = "1.0"
    last_trained: Optional[str] = None


@dataclass
class PredictionRecord:
    """Single prediction record @zara"""
    date: str = ""
    hour: int = 0
    predicted_kwh: float = 0.0
    actual_kwh: Optional[float] = None
    weather_source: str = ""
    timestamp: Optional[str] = None


def create_default_hourly_profile() -> HourlyProfile:
    """Create default hourly profile @zara"""
    return HourlyProfile(
        hourly_averages={str(h): 0.0 for h in range(24)},
        total_samples=0,
    )


def create_default_learned_weights() -> LearnedWeights:
    """Create default learned weights @zara"""
    return LearnedWeights()
