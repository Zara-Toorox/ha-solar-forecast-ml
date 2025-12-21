"""Data Prediction Handler for Solar Forecast ML Integration V12.2.0 @zara

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from pathlib import Path

from homeassistant.core import HomeAssistant

from .data_io import DataManagerIO

_LOGGER = logging.getLogger(__name__)

class DataPredictionHandler(DataManagerIO):
    """Legacy prediction handler - V3 uses hourly_predictions.json. @zara"""

    def __init__(self, hass: HomeAssistant, data_dir: Path):
        """Initialize handler. @zara"""
        super().__init__(hass, data_dir)
        _LOGGER.debug("DataPredictionHandler initialized (legacy stub)")
