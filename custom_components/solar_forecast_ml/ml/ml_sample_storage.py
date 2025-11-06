"""
ML Sample Storage for Solar Forecast ML Integration

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
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class MLSampleStorage:
    """Handles storage and retrieval of ML training samples."""
    
    def __init__(self, data_manager):
        """Initialize sample storage."""
        self.data_manager = data_manager
        self._cache: List[Dict[str, Any]] = []
        self._cache_loaded = False
    
    async def add_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Add a single sample to storage.
        
        Args:
            sample: Sample data dictionary
            
        Returns:
            True if successful
        """
        try:
            # Load cache if not loaded
            if not self._cache_loaded:
                await self._load_cache()
            
            # Add to cache
            self._cache.append(sample)
            
            # Save to persistent storage
            samples_data = {
                "samples": self._cache,
                "count": len(self._cache),
                "last_updated": dt_util.now().isoformat()
            }
            
            success = await self.data_manager.save_hourly_samples(samples_data)
            
            if success:
                _LOGGER.debug(f"Sample added, total: {len(self._cache)}")
            
            return success
            
        except Exception as e:
            _LOGGER.error(f"Failed to add sample: {e}", exc_info=True)
            return False
    
    async def get_samples(
        self,
        limit: Optional[int] = None,
        min_timestamp: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve samples from storage.
        
        Args:
            limit: Maximum number of samples to return
            min_timestamp: Only return samples after this timestamp
            
        Returns:
            List of samples
        """
        try:
            # Load cache if not loaded
            if not self._cache_loaded:
                await self._load_cache()
            
            samples = self._cache.copy()
            
            # Filter by timestamp
            if min_timestamp:
                samples = [
                    s for s in samples
                    if datetime.fromisoformat(s.get("timestamp", "1970-01-01")) >= min_timestamp
                ]
            
            # Apply limit
            if limit and len(samples) > limit:
                samples = samples[-limit:]
            
            return samples
            
        except Exception as e:
            _LOGGER.error(f"Failed to get samples: {e}", exc_info=True)
            return []
    
    async def get_sample_count(self) -> int:
        """Get total number of stored samples."""
        if not self._cache_loaded:
            await self._load_cache()
        return len(self._cache)
    
    async def clear_samples(self) -> bool:
        """Clear all stored samples."""
        try:
            self._cache.clear()
            self._cache_loaded = True
            
            samples_data = {
                "samples": [],
                "count": 0,
                "last_updated": dt_util.now().isoformat()
            }
            
            return await self.data_manager.save_hourly_samples(samples_data)
            
        except Exception as e:
            _LOGGER.error(f"Failed to clear samples: {e}")
            return False
    
    async def _load_cache(self) -> None:
        """Load samples from persistent storage into cache."""
        try:
            data = await self.data_manager.load_hourly_samples()
            self._cache = data.get("samples", [])
            self._cache_loaded = True
            _LOGGER.debug(f"Loaded {len(self._cache)} samples from storage")
        except Exception as e:
            _LOGGER.error(f"Failed to load sample cache: {e}")
            self._cache = []
            self._cache_loaded = True
