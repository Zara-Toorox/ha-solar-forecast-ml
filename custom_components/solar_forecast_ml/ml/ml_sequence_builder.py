"""Sequence Builder for LSTM Training V12.2.0 @zara

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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

_np: Optional[Any] = None

def _ensure_numpy() -> Any:
    """Lazily imports and returns the NumPy module @zara"""
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
        except ImportError as e:
            _LOGGER.error(f"NumPy is required for SequenceBuilder: {e}")
            raise ImportError(f"NumPy library is required: {e}") from e
    return _np

class SequenceBuilder:
    """Build training sequences for LSTM from hourly predictions"""

    def __init__(
        self,
        sequence_length: int = 24,
        stride: int = 1,
        min_sequence_gap_hours: int = 0
    ):
        """
        Initialize SequenceBuilder.

        Args:
            sequence_length: Lookback window (24 = use last 24 hours to predict next hour)
            stride: Step size between sequences (1 = every hour, 24 = every day)
            min_sequence_gap_hours: Min gap between sequences (0 = overlapping OK)
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.min_sequence_gap_hours = min_sequence_gap_hours

        _LOGGER.info(
            f"SequenceBuilder initialized: length={sequence_length}, "
            f"stride={stride}, gap={min_sequence_gap_hours}"
        )

    def _encode_season(self, month: int) -> float:
        """Encode month as seasonal value (0-3) @zara"""
        if month in [12, 1, 2]:
            return 0.0
        elif month in [3, 4, 5]:
            return 1.0
        elif month in [6, 7, 8]:
            return 2.0
        else:
            return 3.0

    def _extract_features(
        self,
        hour_data: Dict[str, Any],
        feature_names: List[str]
    ) -> List[float]:
        """
        Extract features in correct order (must match FeatureEngineerV3).

        Feature order:
        1. target_hour (0-23)
        2. day_of_year (1-366)
        3. season (0-3)
        4. temperature (°C)
        5. solar_radiation_wm2 (W/m²)
        6. wind (m/s)
        7. humidity (%)
        8. rain (mm)
        9. clouds (%)
        10. sun_elevation_deg (degrees)
        11. theoretical_max_kwh (kWh)
        12. clear_sky_radiation_wm2 (W/m²)
        13. production_yesterday (kWh)
        14. production_same_hour_yesterday (kWh)
        """
        features = []

        target_hour = hour_data.get('target_hour', 12)
        features.append(float(target_hour))

        try:
            target_date = hour_data.get('target_date')
            if isinstance(target_date, str):
                dt = datetime.fromisoformat(target_date.replace('Z', '+00:00'))
                day_of_year = dt.timetuple().tm_yday
            else:
                day_of_year = 180
        except Exception:
            day_of_year = 180

        features.append(float(day_of_year))

        try:
            target_month = hour_data.get('target_month')
            if target_month is None:

                target_date = hour_data.get('target_date')
                if isinstance(target_date, str):
                    dt = datetime.fromisoformat(target_date.replace('Z', '+00:00'))
                    target_month = dt.month
                else:
                    target_month = 6
        except Exception:
            target_month = 6

        features.append(self._encode_season(target_month))

        weather = hour_data.get('weather_corrected', {})
        features.append(float(weather.get('temperature', 15.0)))
        features.append(float(weather.get('solar_radiation_wm2', 0.0)))
        features.append(float(weather.get('wind', 3.0)))
        features.append(float(weather.get('humidity', 70.0)))
        features.append(float(weather.get('rain', 0.0)))
        features.append(float(weather.get('clouds', 50.0)))

        astronomy = hour_data.get('astronomy', {})
        features.append(float(astronomy.get('sun_elevation_deg', 0.0)))
        features.append(float(astronomy.get('theoretical_max_kwh', 0.0)))
        features.append(float(astronomy.get('clear_sky_radiation_wm2', 0.0)))

        features.append(float(hour_data.get('production_yesterday', 0.0)))
        features.append(float(hour_data.get('production_same_hour_yesterday', 0.0)))

        return features

    def validate_sequence(self, sequence: Any) -> bool:
        """Validate sequence for training @zara"""
        np = _ensure_numpy()

        try:

            if np.isnan(sequence).any():
                return False

            if np.isinf(sequence).any():
                return False

            if len(sequence.shape) != 2:
                return False

            if sequence.shape[1] != 14:
                return False

            temp_col = sequence[:, 3]
            if (temp_col < -50).any() or (temp_col > 50).any():
                return False

            hour_col = sequence[:, 0]
            if (hour_col < 0).any() or (hour_col > 23).any():
                return False

            return True

        except Exception as e:
            _LOGGER.warning(f"Sequence validation failed: {e}")
            return False

    def build_sequences_from_predictions(
        self,
        hourly_predictions: List[Dict[str, Any]],
        feature_names: List[str]
    ) -> Tuple[List[Any], List[float]]:
        """
        Build sequences from hourly_predictions.json data.

        Args:
            hourly_predictions: List of hourly prediction dicts
            feature_names: List of 14 feature names (must match FeatureEngineerV3)

        Returns:
            X_sequences: List of sequences (each: [seq_len, 14 features])
            y_targets: List of target values (actual_kwh)
        """
        np = _ensure_numpy()

        _LOGGER.info(f"Building sequences from {len(hourly_predictions)} hourly predictions")

        valid_predictions = [
            p for p in hourly_predictions
            if p.get('actual_kwh') is not None and p.get('actual_kwh') > 0
        ]

        if len(valid_predictions) < self.sequence_length + 1:
            _LOGGER.warning(
                f"Not enough valid predictions: {len(valid_predictions)} "
                f"(need {self.sequence_length + 1})"
            )
            return [], []

        try:
            sorted_preds = sorted(
                valid_predictions,
                key=lambda x: (x.get('target_date', ''), x.get('target_hour', 0))
            )
        except Exception as e:
            _LOGGER.error(f"Failed to sort predictions: {e}")
            return [], []

        X_sequences = []
        y_targets = []

        for i in range(0, len(sorted_preds) - self.sequence_length, self.stride):
            sequence_data = sorted_preds[i:i + self.sequence_length]
            target_data = sorted_preds[i + self.sequence_length]

            try:
                sequence = []
                for hour_data in sequence_data:
                    features = self._extract_features(hour_data, feature_names)
                    sequence.append(features)

                sequence_arr = np.array(sequence, dtype=float)

                target = float(target_data.get('actual_kwh', 0.0))

                if self.validate_sequence(sequence_arr) and target >= 0:
                    X_sequences.append(sequence_arr)
                    y_targets.append(target)

            except Exception as e:
                _LOGGER.debug(f"Failed to build sequence {i}: {e}")
                continue

        _LOGGER.info(f"Built {len(X_sequences)} valid sequences")

        return X_sequences, y_targets

    def create_sequences_for_inference(
        self,
        recent_hours: List[Dict[str, Any]],
        feature_names: List[str]
    ) -> Optional[Any]:
        """
        Create single sequence for inference (real-time prediction).

        Args:
            recent_hours: Last 24 hours of data (from hourly_predictions.json)
            feature_names: Feature names (same order as training)

        Returns:
            Single sequence ready for model.predict() or None if invalid
        """
        np = _ensure_numpy()

        if len(recent_hours) < self.sequence_length:
            _LOGGER.warning(
                f"Not enough recent hours: {len(recent_hours)} "
                f"(need {self.sequence_length})"
            )
            return None

        try:
            sorted_hours = sorted(
                recent_hours,
                key=lambda x: (x.get('target_date', ''), x.get('target_hour', 0))
            )
            sequence_data = sorted_hours[-self.sequence_length:]

            sequence = []
            for hour_data in sequence_data:
                features = self._extract_features(hour_data, feature_names)
                sequence.append(features)

            sequence_arr = np.array(sequence, dtype=float)

            if not self.validate_sequence(sequence_arr):
                _LOGGER.warning("Inference sequence validation failed")
                return None

            return sequence_arr

        except Exception as e:
            _LOGGER.error(f"Failed to create inference sequence: {e}")
            return None

    def get_sequence_stats(
        self,
        X_sequences: List[Any],
        y_targets: List[float]
    ) -> Dict[str, Any]:
        """
        Get statistics about sequences (for debugging).

        Returns:
            stats: Dictionary with sequence statistics
        """
        np = _ensure_numpy()

        if not X_sequences:
            return {
                'count': 0,
                'sequence_length': self.sequence_length,
                'feature_count': 0
            }

        y_arr = np.array(y_targets)

        return {
            'count': len(X_sequences),
            'sequence_length': self.sequence_length,
            'feature_count': X_sequences[0].shape[1] if X_sequences else 0,
            'target_min': float(y_arr.min()),
            'target_max': float(y_arr.max()),
            'target_mean': float(y_arr.mean()),
            'target_std': float(y_arr.std())
        }
