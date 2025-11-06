"""
Coordinator Helper Functions for Solar Forecast ML Integration

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
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from .core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class CoordinatorHelpers:
    """Helper functions for the data update coordinator."""
    
    @staticmethod
    def calculate_next_update_time(
        last_update: Optional[datetime],
        interval_minutes: int = 15
    ) -> datetime:
        """
        Calculate the next scheduled update time.
        
        Args:
            last_update: Last update timestamp
            interval_minutes: Update interval in minutes
            
        Returns:
            Next update time
        """
        if last_update is None:
            return dt_util.now()
        
        next_update = last_update + timedelta(minutes=interval_minutes)
        
        # If next update is in the past, use current time
        if next_update < dt_util.now():
            return dt_util.now()
        
        return next_update
    
    @staticmethod
    def should_force_update(
        last_update: Optional[datetime],
        max_age_hours: int = 24
    ) -> bool:
        """
        Check if data should be force-updated due to age.
        
        Args:
            last_update: Last update timestamp
            max_age_hours: Maximum age in hours
            
        Returns:
            True if force update needed
        """
        if last_update is None:
            return True
        
        age = dt_util.now() - last_update
        return age.total_seconds() > max_age_hours * 3600
    
    @staticmethod
    def validate_coordinator_data(data: Dict[str, Any]) -> bool:
        """
        Validate coordinator data structure.
        
        Args:
            data: Coordinator data dictionary
            
        Returns:
            True if valid
        """
        required_keys = ["last_update", "forecasts"]
        
        for key in required_keys:
            if key not in data:
                _LOGGER.error(f"Missing required key in coordinator data: {key}")
                return False
        
        return True
    
    @staticmethod
    def merge_forecast_data(
        old_data: Dict[str, Any],
        new_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge new forecast data with existing data.
        
        Args:
            old_data: Existing data
            new_data: New data to merge
            
        Returns:
            Merged data dictionary
        """
        merged = old_data.copy()
        
        # Update with new data
        for key, value in new_data.items():
            if key == "forecasts" and key in merged:
                # Merge forecast lists
                merged[key].update(value)
            else:
                merged[key] = value
        
        # Update timestamp
        merged["last_update"] = dt_util.now().isoformat()
        
        return merged
    
    @staticmethod
    def calculate_data_staleness(last_update: Optional[datetime]) -> Dict[str, Any]:
        """
        Calculate data staleness metrics.
        
        Args:
            last_update: Last update timestamp
            
        Returns:
            Dictionary with staleness metrics
        """
        if last_update is None:
            return {
                "stale": True,
                "age_seconds": None,
                "age_human": "Never updated",
                "status": "no_data"
            }
        
        age = dt_util.now() - last_update
        age_seconds = age.total_seconds()
        
        # Determine staleness status
        if age_seconds < 900:  # 15 minutes
            status = "fresh"
            stale = False
        elif age_seconds < 3600:  # 1 hour
            status = "acceptable"
            stale = False
        elif age_seconds < 21600:  # 6 hours
            status = "stale"
            stale = True
        else:
            status = "very_stale"
            stale = True
        
        # Human-readable age
        if age_seconds < 60:
            age_human = f"{int(age_seconds)} seconds ago"
        elif age_seconds < 3600:
            age_human = f"{int(age_seconds / 60)} minutes ago"
        elif age_seconds < 86400:
            age_human = f"{int(age_seconds / 3600)} hours ago"
        else:
            age_human = f"{int(age_seconds / 86400)} days ago"
        
        return {
            "stale": stale,
            "age_seconds": age_seconds,
            "age_human": age_human,
            "status": status
        }
    
    @staticmethod
    def format_update_summary(update_results: Dict[str, bool]) -> str:
        """
        Format update results into a readable summary.
        
        Args:
            update_results: Dictionary of component -> success status
            
        Returns:
            Formatted summary string
        """
        total = len(update_results)
        successful = sum(1 for v in update_results.values() if v)
        failed = total - successful
        
        if failed == 0:
            return f"All {total} components updated successfully"
        elif successful == 0:
            return f"All {total} components failed to update"
        else:
            return f"{successful}/{total} components updated successfully"
