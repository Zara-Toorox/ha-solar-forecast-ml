"""
ML Sample Validator for Solar Forecast ML Integration

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

_LOGGER = logging.getLogger(__name__)


class MLSampleValidator:
    """Validates ML training samples for quality and completeness."""
    
    @staticmethod
    def validate_sample(sample: Dict[str, Any]) -> bool:
        """
        Validate a single training sample.
        
        Args:
            sample: Sample dictionary to validate
            
        Returns:
            True if sample is valid
        """
        # Check required fields
        required_fields = ["timestamp", "features", "target"]
        for field in required_fields:
            if field not in sample:
                _LOGGER.warning(f"Sample missing required field: {field}")
                return False
        
        # Validate timestamp
        try:
            datetime.fromisoformat(sample["timestamp"])
        except (ValueError, TypeError):
            _LOGGER.warning(f"Invalid timestamp format: {sample.get('timestamp')}")
            return False
        
        # Validate features
        features = sample.get("features", {})
        if not isinstance(features, dict):
            _LOGGER.warning("Features must be a dictionary")
            return False
        
        if not features:
            _LOGGER.warning("Features dictionary is empty")
            return False
        
        # Validate target
        target = sample.get("target")
        if not isinstance(target, (int, float)):
            _LOGGER.warning(f"Invalid target type: {type(target)}")
            return False
        
        if target < 0:
            _LOGGER.warning(f"Negative target value: {target}")
            return False
        
        return True
    
    @staticmethod
    def validate_feature_completeness(
        sample: Dict[str, Any],
        required_features: List[str]
    ) -> Dict[str, Any]:
        """
        Check if sample has all required features.
        
        Args:
            sample: Sample to validate
            required_features: List of required feature names
            
        Returns:
            Dictionary with validation results
        """
        features = sample.get("features", {})
        
        missing_features = []
        for feature in required_features:
            if feature not in features:
                missing_features.append(feature)
        
        return {
            "complete": len(missing_features) == 0,
            "missing_features": missing_features,
            "completeness_ratio": (len(required_features) - len(missing_features)) / len(required_features) if required_features else 1.0
        }
    
    @staticmethod
    def check_sample_quality(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze quality metrics for a set of samples.
        
        Args:
            samples: List of samples to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        if not samples:
            return {
                "total_samples": 0,
                "valid_samples": 0,
                "quality_score": 0.0,
                "issues": ["No samples provided"]
            }
        
        valid_count = 0
        issues = []
        
        # Count valid samples
        for sample in samples:
            if MLSampleValidator.validate_sample(sample):
                valid_count += 1
        
        # Calculate quality score
        quality_score = valid_count / len(samples) if samples else 0.0
        
        # Identify issues
        if quality_score < 0.9:
            issues.append(f"Low validation rate: {quality_score:.1%}")
        
        # Check for feature consistency
        feature_sets = [set(s.get("features", {}).keys()) for s in samples if s.get("features")]
        if feature_sets:
            common_features = set.intersection(*feature_sets) if feature_sets else set()
            all_features = set.union(*feature_sets) if feature_sets else set()
            
            if len(common_features) < len(all_features):
                inconsistent = len(all_features) - len(common_features)
                issues.append(f"Inconsistent features: {inconsistent} features not present in all samples")
        
        return {
            "total_samples": len(samples),
            "valid_samples": valid_count,
            "invalid_samples": len(samples) - valid_count,
            "quality_score": quality_score,
            "is_sufficient": valid_count >= 100 and quality_score >= 0.8,
            "issues": issues
        }
    
    @staticmethod
    def filter_valid_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out invalid samples from a list.
        
        Args:
            samples: List of samples to filter
            
        Returns:
            List of valid samples only
        """
        valid_samples = []
        invalid_count = 0
        
        for sample in samples:
            if MLSampleValidator.validate_sample(sample):
                valid_samples.append(sample)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            _LOGGER.info(f"Filtered out {invalid_count} invalid samples, kept {len(valid_samples)}")
        
        return valid_samples
