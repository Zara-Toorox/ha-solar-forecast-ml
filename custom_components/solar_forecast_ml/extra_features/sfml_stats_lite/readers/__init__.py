"""Readers module for SFML Stats Lite.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""
from .forecast_comparison_reader import (
    ForecastComparisonReader,
    ForecastComparisonDay,
    ForecastComparisonStats,
    ExternalForecast,
)

__all__ = [
    "ForecastComparisonReader",
    "ForecastComparisonDay",
    "ForecastComparisonStats",
    "ExternalForecast",
]
