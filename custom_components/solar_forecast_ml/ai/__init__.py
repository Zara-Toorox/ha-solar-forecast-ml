# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from .ai_tiny_lstm import TinyLSTM
from .ai_feature_engineering import FeatureEngineer
from .ai_seasonal import SeasonalAdjuster
from .ai_dni_tracker import DniTracker
from .ai_predictor import (
    AIPredictor,
    ModelState,
    TrainingResult,
    GroupPrediction,
    HourlyProfile,
    LearnedWeights,
)
from .ai_best_hour import BestHourCalculator
from .ai_types import (
    PredictionRecord,
    create_default_hourly_profile,
    create_default_learned_weights,
)
from .ai_helpers import format_time_ago
from .ai_grid_search import (
    GridSearchOptimizer,
    GridSearchResult,
    HardwareInfo,
    detect_hardware,
)

__all__ = [
    "TinyLSTM",
    "FeatureEngineer",
    "SeasonalAdjuster",
    "DniTracker",
    "AIPredictor",
    "ModelState",
    "TrainingResult",
    "GroupPrediction",
    "HourlyProfile",
    "LearnedWeights",
    "create_default_hourly_profile",
    "create_default_learned_weights",
    "BestHourCalculator",
    "PredictionRecord",
    "format_time_ago",
    "GridSearchOptimizer",
    "GridSearchResult",
    "HardwareInfo",
    "detect_hardware",
]
