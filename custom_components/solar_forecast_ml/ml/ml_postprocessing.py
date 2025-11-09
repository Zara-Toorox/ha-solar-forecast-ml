"""
ML Postprocessing for Solar Forecast ML Integration

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
from typing import Optional, List, Dict, Any

from ..const import CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX

_LOGGER = logging.getLogger(__name__)


class MLPostprocessor:
    """Postprocesses ML predictions with validation and correction"""
    
    def __init__(self, peak_power_kw: float = 0.0):
        """Initialize postprocessor"""
        self.peak_power_kw = peak_power_kw
    
    def clip_to_physical_limits(self, prediction: float) -> float:
        """Clip prediction to physical system limits"""
        # Ensure non-negative
        prediction = max(0.0, prediction)
        
        # Clip to peak power if configured
        if self.peak_power_kw > 0:
            prediction = min(prediction, self.peak_power_kw)
        
        return prediction
    
    def apply_correction_factor(
        self,
        prediction: float,
        correction_factor: float
    ) -> float:
        """Apply correction factor to prediction"""
        # Validate correction factor range
        correction_factor = max(CORRECTION_FACTOR_MIN, min(CORRECTION_FACTOR_MAX, correction_factor))
        
        corrected = prediction * correction_factor
        
        return self.clip_to_physical_limits(corrected)
    
    def smooth_predictions(
        self,
        predictions: List[float],
        window_size: int = 3
    ) -> List[float]:
        """Apply moving average smoothing to predictions"""
        if not predictions or window_size < 2:
            return predictions
        
        smoothed = []
        for i in range(len(predictions)):
            # Get window bounds
            start = max(0, i - window_size // 2)
            end = min(len(predictions), i + window_size // 2 + 1)
            
            # Calculate average
            window_values = predictions[start:end]
            avg = sum(window_values) / len(window_values)
            smoothed.append(avg)
        
        return smoothed
    
    def validate_prediction(
        self,
        prediction: float,
        hour: int,
        historical_avg: Optional[float] = None
    ) -> Dict[str, Any]:
        """Validate prediction and return diagnostics"""
        issues = []
        valid = True
        
        # Check for negative values
        if prediction < 0:
            issues.append("Negative prediction")
            valid = False
        
        # Check for unrealistic values
        if self.peak_power_kw > 0 and prediction > self.peak_power_kw * 1.2:
            issues.append(f"Exceeds physical limit ({self.peak_power_kw} kW)")
            valid = False
        
        # Check night hours
        if hour < 6 or hour > 20:
            if prediction > 0.1:
                issues.append(f"Unexpected production during night (hour {hour})")
        
        # Compare with historical average
        if historical_avg is not None and historical_avg > 0:
            deviation = abs(prediction - historical_avg) / historical_avg
            if deviation > 2.0:
                issues.append(f"Large deviation from historical avg ({deviation:.1%})")
        
        return {
            "valid": valid and len(issues) == 0,
            "prediction": prediction,
            "issues": issues,
            "clipped": prediction != self.clip_to_physical_limits(prediction)
        }
    
    def adjust_for_time_of_day(
        self,
        prediction: float,
        hour: int
    ) -> float:
        """Adjust prediction based on time of day"""
        # Zero out night hours
        if hour < 6 or hour > 20:
            return 0.0
        
        # Apply time-of-day scaling (simple bell curve)
        # Peak around 12:00
        time_factor = 1.0
        if 6 <= hour < 12:
            time_factor = (hour - 6) / 6.0  # Ramp up
        elif 12 <= hour < 20:
            time_factor = (20 - hour) / 8.0  # Ramp down
        
        adjusted = prediction * time_factor
        return self.clip_to_physical_limits(adjusted)
