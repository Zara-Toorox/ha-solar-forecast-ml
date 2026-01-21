# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""Readers module for SFML Stats."""
from __future__ import annotations

from .solar_reader import SolarDataReader
from .price_reader import PriceDataReader
from .forecast_comparison_reader import ForecastComparisonReader

__all__ = ["SolarDataReader", "PriceDataReader", "ForecastComparisonReader"]
