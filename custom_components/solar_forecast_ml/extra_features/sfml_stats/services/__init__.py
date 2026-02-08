# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Services package for SFML Stats integration. @zara"""
from .daily_aggregator import DailyEnergyAggregator
from .billing_calculator import BillingCalculator
from .monthly_tariff_manager import MonthlyTariffManager
from .forecast_comparison_collector import ForecastComparisonCollector

__all__ = [
    "DailyEnergyAggregator",
    "BillingCalculator",
    "MonthlyTariffManager",
    "ForecastComparisonCollector",
]
