# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Readers module for SFML Stats. @zara"""
from __future__ import annotations

from .solar_reader import SolarDataReader
from .price_reader import PriceDataReader
from .forecast_comparison_reader import ForecastComparisonReader

__all__ = ["SolarDataReader", "PriceDataReader", "ForecastComparisonReader"]
