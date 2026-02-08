# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""AI module exports @zara"""

from .ai_best_hour import BestHourCalculator
from .ai_dni_tracker import DniTracker
from .ai_feature_engineering import FeatureEngineer
from .ai_feature_importance import FeatureImportanceAnalyzer
from .ai_grid_search import GridSearchOptimizer
from .ai_helpers import format_time_ago
from .ai_predictor import AIPredictor, ModelState
from .ai_seasonal import SeasonalAdjuster
from .ai_tiny_lstm import TinyLSTM
from .ai_tiny_ridge import TinyRidge
from .ai_types import (
    HourlyProfile,
    LearnedWeights,
    PredictionRecord,
    create_default_hourly_profile,
    create_default_learned_weights,
)

__all__ = [
    "TinyLSTM",
    "TinyRidge",
    "FeatureEngineer",
    "FeatureImportanceAnalyzer",
    "SeasonalAdjuster",
    "DniTracker",
    "AIPredictor",
    "ModelState",
    "BestHourCalculator",
    "HourlyProfile",
    "LearnedWeights",
    "PredictionRecord",
    "create_default_hourly_profile",
    "create_default_learned_weights",
    "format_time_ago",
    "GridSearchOptimizer",
]
