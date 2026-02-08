# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Charts module for SFML Stats. @zara"""
from __future__ import annotations

from .styles import ChartStyles, apply_dark_theme
from .base import BaseChart
from .weekly_report import WeeklyReportChart
from .forecast_comparison import ForecastComparisonChart

__all__ = [
    "ChartStyles",
    "apply_dark_theme",
    "BaseChart",
    "WeeklyReportChart",
    "ForecastComparisonChart",
]
