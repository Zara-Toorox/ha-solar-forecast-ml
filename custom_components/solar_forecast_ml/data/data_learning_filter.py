"""Learning Filter - Central helper for excluding flagged hours from learning V12.2.0 @zara

This module provides centralized logic for:
1. Checking if individual hours should be excluded from learning
2. Calculating the ratio of excluded hours for daily learning decisions
3. Setting weather alert flags based on sensor vs. forecast comparison

Design Decision:
- Individual hours with flags are excluded from hourly learning (ML, Panel Groups, etc.)
- Daily learning (Pattern Learner) is skipped if >25% of production hours are excluded
- This prevents learning from anomalous data (unexpected weather, sensor issues)

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
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

# Threshold: If more than 25% of production hours are excluded, skip daily learning
DAILY_LEARNING_EXCLUSION_THRESHOLD = 0.25


def should_exclude_hour_from_learning(prediction: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if an hour should be excluded from learning.

    Checks all relevant flags that indicate anomalous data:
    - exclude_from_learning (master flag)
    - has_weather_alert (unexpected weather event)
    - inverter_clipped (hardware limitation, not weather)
    - frost_detected == "heavy_frost" (in weather_actual)

    Args:
        prediction: Hourly prediction dict from hourly_predictions.json

    Returns:
        Tuple of (should_exclude: bool, reason: str)
    """
    flags = prediction.get("flags") or {}
    weather_actual = prediction.get("weather_actual") or {}

    # Master flag - explicitly marked for exclusion
    if flags.get("exclude_from_learning"):
        reason = flags.get("weather_alert_type") or "manually_excluded"
        return True, reason

    # Weather alert - unexpected weather event
    if flags.get("has_weather_alert"):
        reason = flags.get("weather_alert_type") or "weather_alert"
        return True, reason

    # Inverter clipping - hardware limitation
    if flags.get("inverter_clipped"):
        return True, "inverter_clipped"

    # Heavy frost - environmental limitation
    frost_detected = weather_actual.get("frost_detected")
    if frost_detected == "heavy_frost":
        return True, "heavy_frost"

    return False, ""


def filter_predictions_for_learning(
    predictions: List[Dict[str, Any]],
    log_exclusions: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Filter predictions list, removing hours that should be excluded from learning.

    Args:
        predictions: List of hourly predictions
        log_exclusions: Whether to log excluded hours

    Returns:
        Tuple of (filtered_predictions, exclusion_stats)
        exclusion_stats is a dict with counts per reason
    """
    filtered = []
    exclusion_stats: Dict[str, int] = {}

    for pred in predictions:
        should_exclude, reason = should_exclude_hour_from_learning(pred)

        if should_exclude:
            exclusion_stats[reason] = exclusion_stats.get(reason, 0) + 1
            if log_exclusions:
                sample_id = _get_sample_id(pred)
                _LOGGER.debug(f"Excluding {sample_id} from learning: {reason}")
        else:
            filtered.append(pred)

    if exclusion_stats and log_exclusions:
        total_excluded = sum(exclusion_stats.values())
        _LOGGER.info(
            f"Learning filter: Excluded {total_excluded} hours "
            f"({len(filtered)} remaining). Reasons: {exclusion_stats}"
        )

    return filtered, exclusion_stats


def calculate_excluded_hours_ratio(
    predictions: List[Dict[str, Any]],
    production_hours_only: bool = True
) -> Tuple[float, int, int]:
    """Calculate the ratio of excluded hours to total production hours.

    Used for daily learning decisions (Pattern Learner, etc.)

    Args:
        predictions: List of hourly predictions for the day
        production_hours_only: If True, only count hours with actual production

    Returns:
        Tuple of (ratio, excluded_count, total_count)
    """
    if not predictions:
        return 0.0, 0, 0

    # Filter to production hours if requested
    if production_hours_only:
        # Production hours: has actual_kwh > 0 OR prediction_kwh > 0 during daylight
        relevant_hours = [
            p for p in predictions
            if (p.get("actual_kwh") or 0) > 0 or (p.get("prediction_kwh") or 0) > 0.01
        ]
    else:
        relevant_hours = predictions

    if not relevant_hours:
        return 0.0, 0, 0

    excluded_count = 0
    for pred in relevant_hours:
        should_exclude, _ = should_exclude_hour_from_learning(pred)
        if should_exclude:
            excluded_count += 1

    total_count = len(relevant_hours)
    ratio = excluded_count / total_count if total_count > 0 else 0.0

    return ratio, excluded_count, total_count


def should_skip_daily_learning(
    predictions: List[Dict[str, Any]],
    threshold: float = DAILY_LEARNING_EXCLUSION_THRESHOLD
) -> Tuple[bool, float, str]:
    """Check if daily learning should be skipped due to too many excluded hours.

    Args:
        predictions: List of hourly predictions for the day
        threshold: Maximum ratio of excluded hours (default: 0.25 = 25%)

    Returns:
        Tuple of (should_skip: bool, ratio: float, reason: str)
    """
    ratio, excluded_count, total_count = calculate_excluded_hours_ratio(predictions)

    if ratio > threshold:
        reason = (
            f"Too many excluded hours: {excluded_count}/{total_count} "
            f"({ratio:.0%} > {threshold:.0%} threshold)"
        )
        _LOGGER.info(f"Daily learning skipped: {reason}")
        return True, ratio, reason

    return False, ratio, ""


def _get_sample_id(prediction: Dict[str, Any]) -> str:
    """Get a human-readable sample ID for logging."""
    target_date = prediction.get("target_date", "????-??-??")
    target_hour = prediction.get("target_hour", "??")
    return f"{target_date} {target_hour:02d}:00" if isinstance(target_hour, int) else f"{target_date} {target_hour}:00"
