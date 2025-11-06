"""
ML Sample Aggregator for Solar Forecast ML Integration

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
from datetime import datetime, timedelta
from collections import defaultdict

from ..core.core_helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class MLSampleAggregator:
    """Aggregates and summarizes ML training samples."""
    
    @staticmethod
    def aggregate_by_hour(samples: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Group samples by hour of day.
        
        Args:
            samples: List of samples
            
        Returns:
            Dictionary mapping hour (0-23) to samples
        """
        hourly_samples = defaultdict(list)
        
        for sample in samples:
            try:
                timestamp = datetime.fromisoformat(sample.get("timestamp", ""))
                hour = timestamp.hour
                hourly_samples[hour].append(sample)
            except (ValueError, TypeError):
                continue
        
        return dict(hourly_samples)
    
    @staticmethod
    def aggregate_by_day(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group samples by date.
        
        Args:
            samples: List of samples
            
        Returns:
            Dictionary mapping date string to samples
        """
        daily_samples = defaultdict(list)
        
        for sample in samples:
            try:
                timestamp = datetime.fromisoformat(sample.get("timestamp", ""))
                date_str = timestamp.date().isoformat()
                daily_samples[date_str].append(sample)
            except (ValueError, TypeError):
                continue
        
        return dict(daily_samples)
    
    @staticmethod
    def calculate_hourly_statistics(samples: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        """
        Calculate statistics for each hour.
        
        Args:
            samples: List of samples
            
        Returns:
            Dictionary mapping hour to statistics
        """
        hourly_samples = MLSampleAggregator.aggregate_by_hour(samples)
        statistics = {}
        
        for hour, hour_samples in hourly_samples.items():
            if not hour_samples:
                continue
            
            targets = [s.get("target", 0.0) for s in hour_samples if s.get("target") is not None]
            
            if targets:
                statistics[hour] = {
                    "count": len(targets),
                    "mean": sum(targets) / len(targets),
                    "min": min(targets),
                    "max": max(targets),
                    "sum": sum(targets)
                }
        
        return statistics
    
    @staticmethod
    def get_recent_samples(
        samples: List[Dict[str, Any]],
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get samples from the last N days.
        
        Args:
            samples: List of samples
            days: Number of days to look back
            
        Returns:
            Filtered list of recent samples
        """
        cutoff = dt_util.now() - timedelta(days=days)
        recent_samples = []
        
        for sample in samples:
            try:
                timestamp = datetime.fromisoformat(sample.get("timestamp", ""))
                if timestamp >= cutoff:
                    recent_samples.append(sample)
            except (ValueError, TypeError):
                continue
        
        return recent_samples
    
    @staticmethod
    def calculate_feature_statistics(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each feature.
        
        Args:
            samples: List of samples
            
        Returns:
            Dictionary mapping feature name to statistics
        """
        feature_values = defaultdict(list)
        
        # Collect all feature values
        for sample in samples:
            features = sample.get("features", {})
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, (int, float)):
                    feature_values[feature_name].append(feature_value)
        
        # Calculate statistics
        statistics = {}
        for feature_name, values in feature_values.items():
            if values:
                statistics[feature_name] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": MLSampleAggregator._calculate_std(values)
                }
        
        return statistics
    
    @staticmethod
    def _calculate_std(values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    @staticmethod
    def summarize_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a comprehensive summary of samples.
        
        Args:
            samples: List of samples
            
        Returns:
            Summary dictionary
        """
        if not samples:
            return {
                "total_samples": 0,
                "date_range": None,
                "hourly_coverage": {},
                "feature_statistics": {}
            }
        
        # Get date range
        timestamps = []
        for sample in samples:
            try:
                ts = datetime.fromisoformat(sample.get("timestamp", ""))
                timestamps.append(ts)
            except (ValueError, TypeError):
                continue
        
        date_range = None
        if timestamps:
            date_range = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
                "days": (max(timestamps) - min(timestamps)).days + 1
            }
        
        # Calculate hourly coverage
        hourly_samples = MLSampleAggregator.aggregate_by_hour(samples)
        hourly_coverage = {hour: len(samples) for hour, samples in hourly_samples.items()}
        
        # Feature statistics
        feature_stats = MLSampleAggregator.calculate_feature_statistics(samples)
        
        return {
            "total_samples": len(samples),
            "date_range": date_range,
            "hourly_coverage": hourly_coverage,
            "feature_statistics": feature_stats,
            "unique_features": len(feature_stats)
        }
