# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Data Adapter for Solar Forecast ML V16.2.0.
Converts data between database rows and typed dataclasses.
Provides type-safe conversion for legacy structure compatibility.

@zara
"""

import logging
from datetime import datetime
from typing import Any, Optional

from ..const import (
    CORRECTION_FACTOR_MAX,
    CORRECTION_FACTOR_MIN,
    DATA_VERSION,
    ML_MODEL_VERSION,
)
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..ai import (
    HourlyProfile,
    LearnedWeights,
    PredictionRecord,
    create_default_hourly_profile,
    create_default_learned_weights,
)

_LOGGER = logging.getLogger(__name__)


class TypedDataAdapter:
    """Adapter class for converting between database rows and typed objects. @zara

    Provides static methods for bidirectional conversion between
    database dictionaries and typed dataclass instances.
    """

    @staticmethod
    def dict_to_prediction_record(data: dict[str, Any]) -> PredictionRecord:
        """Convert a dictionary to a PredictionRecord. @zara

        Args:
            data: Dictionary with prediction data

        Returns:
            PredictionRecord instance
        """
        if isinstance(data, PredictionRecord):
            return data

        try:
            return PredictionRecord(
                date=data.get("date", data.get("target_date", "")),
                hour=int(data.get("hour", data.get("target_hour", 0))),
                predicted_kwh=float(data.get("predicted_kwh", data.get("prediction_kwh", 0.0))),
                actual_kwh=float(data["actual_kwh"]) if data.get("actual_kwh") is not None else None,
                weather_source=data.get("weather_source", data.get("source", "")),
                timestamp=data.get("timestamp", data.get("prediction_created_at")),
            )
        except (ValueError, TypeError, KeyError) as e:
            _LOGGER.error("Failed to convert dict to PredictionRecord: %s", e)
            raise ValueError(f"Invalid data for PredictionRecord: {e}") from e

    @staticmethod
    def prediction_record_to_dict(record: PredictionRecord) -> dict[str, Any]:
        """Convert a PredictionRecord to a dictionary. @zara

        Args:
            record: PredictionRecord instance

        Returns:
            Dictionary representation
        """
        if not isinstance(record, PredictionRecord):
            _LOGGER.error("Invalid input: expected PredictionRecord instance")
            return {}

        return {
            "date": record.date,
            "hour": record.hour,
            "predicted_kwh": record.predicted_kwh,
            "actual_kwh": record.actual_kwh,
            "weather_source": record.weather_source,
            "timestamp": record.timestamp,
        }

    @staticmethod
    def dict_to_learned_weights(data: dict[str, Any]) -> LearnedWeights:
        """Convert a dictionary to LearnedWeights. @zara

        Handles legacy field names and provides fallbacks.

        Args:
            data: Dictionary with weight data

        Returns:
            LearnedWeights instance
        """
        if isinstance(data, LearnedWeights):
            return data

        try:
            # Handle weights field with fallback @zara
            weights = data.get("weights")
            if weights is None:
                weights = data.get("weather_weights", {})
                if weights:
                    _LOGGER.debug("Using 'weather_weights' as fallback for 'weights'")

            # Handle feature_stds @zara
            feature_stds = data.get("feature_stds", {})
            if not isinstance(feature_stds, dict):
                feature_stds = {}

            # Parse version @zara
            version = data.get("version", data.get("model_version", ML_MODEL_VERSION))

            # Parse last_trained @zara
            last_trained = data.get("last_trained")
            if last_trained is None:
                last_trained = dt_util.now().isoformat()

            return LearnedWeights(
                weights=weights if isinstance(weights, dict) else {},
                feature_stds=feature_stds,
                version=str(version),
                last_trained=last_trained,
            )

        except Exception as e:
            _LOGGER.error("Failed to convert dict to LearnedWeights: %s", e)
            return create_default_learned_weights()

    @staticmethod
    def learned_weights_to_dict(weights: LearnedWeights) -> dict[str, Any]:
        """Convert LearnedWeights to a dictionary. @zara

        Args:
            weights: LearnedWeights instance

        Returns:
            Dictionary representation
        """
        if not isinstance(weights, LearnedWeights):
            _LOGGER.error("Invalid input: expected LearnedWeights instance")
            return TypedDataAdapter.learned_weights_to_dict(create_default_learned_weights())

        return {
            "weights": weights.weights,
            "feature_stds": weights.feature_stds,
            "version": weights.version,
            "last_trained": weights.last_trained,
            "file_format_version": DATA_VERSION,
            "last_saved": dt_util.now().isoformat(),
        }

    @staticmethod
    def dict_to_hourly_profile(data: dict[str, Any]) -> HourlyProfile:
        """Convert a dictionary to HourlyProfile. @zara

        Handles legacy nested format and provides fallbacks.

        Args:
            data: Dictionary with profile data

        Returns:
            HourlyProfile instance
        """
        if isinstance(data, HourlyProfile):
            return data

        try:
            # Parse hourly_averages with legacy format support @zara
            hourly_averages_raw = data.get("hourly_averages", {})
            hourly_averages: dict[str, float] = {}

            if isinstance(hourly_averages_raw, dict):
                for key, value in hourly_averages_raw.items():
                    if isinstance(value, dict):
                        # Legacy format: {"count": 0, "total": 0.0, "average": 0.5} @zara
                        hourly_averages[str(key)] = float(value.get("average", 0.0))
                    elif isinstance(value, (int, float)):
                        hourly_averages[str(key)] = float(value)
                    else:
                        hourly_averages[str(key)] = 0.0

            # Parse other fields @zara
            total_samples = int(data.get("total_samples", data.get("samples_count", 0)))
            last_updated = data.get("last_updated")

            return HourlyProfile(
                hourly_averages=hourly_averages,
                total_samples=total_samples,
                last_updated=last_updated,
            )

        except Exception as e:
            _LOGGER.error("Failed to convert dict to HourlyProfile: %s", e)
            return create_default_hourly_profile()

    @staticmethod
    def hourly_profile_to_dict(profile: HourlyProfile) -> dict[str, Any]:
        """Convert HourlyProfile to a dictionary. @zara

        Args:
            profile: HourlyProfile instance

        Returns:
            Dictionary representation
        """
        if not isinstance(profile, HourlyProfile):
            _LOGGER.error("Invalid input: expected HourlyProfile instance")
            return TypedDataAdapter.hourly_profile_to_dict(create_default_hourly_profile())

        return {
            "hourly_averages": profile.hourly_averages,
            "total_samples": profile.total_samples,
            "last_updated": profile.last_updated,
            "file_format_version": DATA_VERSION,
            "last_saved": dt_util.now().isoformat(),
        }

    @staticmethod
    def row_to_dict(row: Any) -> dict[str, Any]:
        """Convert a database row to a dictionary. @zara

        Args:
            row: Database row (aiosqlite.Row or similar)

        Returns:
            Dictionary representation
        """
        if row is None:
            return {}

        if isinstance(row, dict):
            return row

        try:
            # aiosqlite.Row can be accessed by key @zara
            return dict(row)
        except Exception:
            pass

        try:
            # Try to access keys() method @zara
            if hasattr(row, "keys"):
                return {key: row[key] for key in row.keys()}
        except Exception:
            pass

        # Fallback: convert to dict via __iter__ @zara
        try:
            return dict(zip(range(len(row)), row))
        except Exception as e:
            _LOGGER.error("Failed to convert row to dict: %s", e)
            return {}

    @staticmethod
    def parse_datetime_safe(value: Any) -> Optional[datetime]:
        """Safely parse a datetime from various formats. @zara

        Args:
            value: datetime, string, or None

        Returns:
            datetime instance or None
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            return dt_util.parse_datetime(value)

        _LOGGER.warning("Cannot parse datetime from type: %s", type(value))
        return None

    @staticmethod
    def to_iso_string(value: Any) -> Optional[str]:
        """Convert a datetime to ISO format string. @zara

        Args:
            value: datetime, string, or None

        Returns:
            ISO format string or None
        """
        if value is None:
            return None

        if isinstance(value, str):
            return value

        if isinstance(value, datetime):
            return value.isoformat()

        _LOGGER.warning("Cannot convert to ISO string from type: %s", type(value))
        return None
