"""Astronomy module for solar calculations and caching V12.2.0 @zara

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

from .astronomy_cache import AstronomyCache
from .astronomy_cache_manager import AstronomyCacheManager, get_cache_manager
from .max_peak_tracker import MaxPeakTracker

__all__ = ["AstronomyCache", "MaxPeakTracker", "AstronomyCacheManager", "get_cache_manager"]
