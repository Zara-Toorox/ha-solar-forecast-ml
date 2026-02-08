# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Core module exports @zara"""

from .core_exceptions import (
    CircuitBreakerOpenException,
    ConfigurationException,
    DataIntegrityException,
    DataValidationException,
    DependencyException,
    ErrorSeverity,
    MLModelException,
    SolarForecastMLException,
    WeatherAPIException,
    create_context,
)
from .core_user_messages import UserMessages, user_msg

__all__ = [
    "SolarForecastMLException",
    "ConfigurationException",
    "DependencyException",
    "WeatherAPIException",
    "DataIntegrityException",
    "DataValidationException",
    "MLModelException",
    "CircuitBreakerOpenException",
    "ErrorSeverity",
    "create_context",
    "UserMessages",
    "user_msg",
]
