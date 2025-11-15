"""
Best Hour Calculator V2 - Hybrid ML + Sun-based approach

COMPLETE REWRITE with realistic sun-aware calculations.

Strategy:
1. BEST: Use ML hourly predictions (if available)
2. FALLBACK: Use historical profile (if available)
3. EMERGENCY: Use solar noon calculation

All methods respect sunrise/sunset times - no production after dark!

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)


class BestHourCalculatorV2:
    """Calculate best production hour using hybrid ML + sun-aware approach"""

    def __init__(self, hass, data_manager, ml_predictor=None):
        """Initialize calculator

        Args:
            hass: Home Assistant instance (for fallback sun.sun access)
            data_manager: DataManager instance (for astronomy cache and profile data)
            ml_predictor: Optional ML predictor (for hourly predictions)
        """
        self.hass = hass
        self.data_manager = data_manager
        self.ml_predictor = ml_predictor

    def _load_astronomy_cache_sync(self, astronomy_cache_file) -> Optional[dict]:
        """Synchronous file read for astronomy cache (runs in executor)"""
        if astronomy_cache_file.exists():
            import json

            with open(astronomy_cache_file, "r") as f:
                return json.load(f)
        return None

    async def _get_sun_times(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get sunrise and sunset times for today from astronomy cache (with safety margins)

        Falls back to sun.sun entity if cache unavailable

        Returns:
            Tuple of (sunrise_dt, sunset_dt) in local timezone, or (None, None) if unavailable
        """
        try:
            from ..core.core_helpers import SafeDateTimeUtil as dt_util

            now_local = dt_util.now()
            today = now_local.date()

            # Try astronomy cache first
            astronomy_cache_file = self.data_manager.data_dir / "stats" / "astronomy_cache.json"

            # Load cache via executor (non-blocking)
            cache = await self.hass.async_add_executor_job(
                self._load_astronomy_cache_sync, astronomy_cache_file
            )

            if cache:

                # Get astronomy data for today
                date_str = today.isoformat()
                day_data = cache.get("days", {}).get(date_str)

                if day_data:
                    sunrise_str = day_data.get("sunrise_local")
                    sunset_str = day_data.get("sunset_local")

                    if sunrise_str and sunset_str:
                        # Parse ISO timestamps and convert to naive local time
                        sunrise_today = datetime.fromisoformat(sunrise_str).replace(tzinfo=None)
                        sunset_today = datetime.fromisoformat(sunset_str).replace(tzinfo=None)

                        # Add safety margins: -30min before sunrise, +30min after sunset
                        sunrise_with_margin = sunrise_today - timedelta(minutes=30)
                        sunset_with_margin = sunset_today + timedelta(minutes=30)

                        _LOGGER.debug(
                            f"Sun times today (cache): {sunrise_with_margin.strftime('%H:%M')} - "
                            f"{sunset_with_margin.strftime('%H:%M')} (with margins)"
                        )

                        return sunrise_with_margin, sunset_with_margin

            # Fallback to sun.sun entity if cache unavailable
            _LOGGER.debug("Astronomy cache unavailable, falling back to sun.sun entity")
            sun_state = self.hass.states.get("sun.sun")
            if not sun_state or not sun_state.attributes:
                _LOGGER.warning("sun.sun entity not available for best hour calculation")
                return None, None

            next_rising_str = sun_state.attributes.get("next_rising")
            next_setting_str = sun_state.attributes.get("next_setting")

            if not next_rising_str or not next_setting_str:
                return None, None

            next_rising = dt_util.parse_datetime(next_rising_str)
            next_setting = dt_util.parse_datetime(next_setting_str)

            if not next_rising or not next_setting:
                return None, None

            next_rising_local = dt_util.as_local(next_rising)
            next_setting_local = dt_util.as_local(next_setting)

            # Determine today's sunrise/sunset
            if next_rising_local.date() == today:
                # Sunrise is today (we're before sunrise)
                sunrise_today = next_rising_local
                sunset_today = next_setting_local
            elif next_setting_local.date() == today:
                # Sunset is today (we're after sunrise, before sunset)
                # Approximate sunrise as now - (sunset - now)
                time_until_sunset = (next_setting_local - now_local).total_seconds()
                sunrise_today = now_local - timedelta(seconds=time_until_sunset)
                sunset_today = next_setting_local
            else:
                # Sun has set today - no production possible
                _LOGGER.debug("Sun has set today - no best hour calculation")
                return None, None

            # Add safety margins: -30min before sunrise, +30min after sunset
            sunrise_with_margin = sunrise_today - timedelta(minutes=30)
            sunset_with_margin = sunset_today + timedelta(minutes=30)

            _LOGGER.debug(
                f"Sun times today (fallback): {sunrise_with_margin.strftime('%H:%M')} - "
                f"{sunset_with_margin.strftime('%H:%M')} (with margins)"
            )

            return sunrise_with_margin, sunset_with_margin

        except Exception as e:
            _LOGGER.error(f"Error getting sun times: {e}", exc_info=True)
            return None, None

    def _is_hour_in_sun_window(self, hour: int, sunrise: datetime, sunset: datetime) -> bool:
        """Check if an hour falls within the production window

        Args:
            hour: Hour to check (0-23)
            sunrise: Sunrise datetime (local)
            sunset: Sunset datetime (local)

        Returns:
            True if hour is within production window
        """
        # Check if the hour (as a full 60-minute period) overlaps with sun window
        from ..core.core_helpers import SafeDateTimeUtil as dt_util

        now_local = dt_util.now()
        hour_start = now_local.replace(hour=hour, minute=0, second=0, microsecond=0)
        hour_end = hour_start + timedelta(hours=1)

        # Production window: sunrise-30min to sunset+30min (as defined in astronomy cache)
        # NOTE: The margins are already included in the sunrise/sunset passed to this method
        # from _get_sun_times() which subtracts 30min from sunrise and adds 30min to sunset
        return hour_start >= sunrise and hour_end <= sunset

    async def calculate_best_hour_today(self) -> Tuple[Optional[int], Optional[float]]:
        """Calculate the hour with the best expected solar production TODAY

        Returns:
            Tuple of (best_hour, prediction_kwh) or (None, None) if no data available

        Strategy:
        1. BEST: Use ML hourly predictions (filtered by sun times)
        2. FALLBACK: Use historical profile (filtered by sun times)
        3. EMERGENCY: Use solar noon calculation
        """
        from ..core.core_helpers import SafeDateTimeUtil as dt_util

        # Get sun times first
        sunrise, sunset = await self._get_sun_times()

        # Strategy 1: ML hourly predictions
        if self.ml_predictor and hasattr(self.ml_predictor, "last_hourly_predictions"):
            ml_result = await self._calculate_from_ml_hourly(sunrise, sunset)
            if ml_result[0] is not None:
                _LOGGER.info(
                    f"✓ Best hour today: {ml_result[0]:02d}:00 ({ml_result[1]:.3f} kWh) "
                    f"[ML-based, sun-aware]"
                )
                return ml_result

        # Strategy 2: Historical profile
        profile_result = await self._calculate_from_profile(sunrise, sunset)
        if profile_result[0] is not None:
            _LOGGER.info(
                f"✓ Best hour today: {profile_result[0]:02d}:00 ({profile_result[1]:.3f} kWh) "
                f"[Profile-based, sun-aware]"
            )
            return profile_result

        # Strategy 3: Solar noon (if sun times available)
        if sunrise and sunset:
            solar_noon_hour = self._calculate_solar_noon(sunrise, sunset)
            _LOGGER.info(
                f"✓ Best hour today: {solar_noon_hour:02d}:00 (solar noon) "
                f"[Sun calculation, no ML/profile data]"
            )
            return solar_noon_hour, 0.0  # Return 0.0 kWh as we have no prediction

        # No data at all
        _LOGGER.warning(
            "⚠ No data available for best hour calculation (no ML, profile, or sun data)"
        )
        return None, None

    async def _calculate_from_ml_hourly(
        self, sunrise: Optional[datetime], sunset: Optional[datetime]
    ) -> Tuple[Optional[int], Optional[float]]:
        """Calculate best hour from ML hourly predictions

        Args:
            sunrise: Sunrise datetime (local) or None
            sunset: Sunset datetime (local) or None

        Returns:
            Tuple of (best_hour, prediction_kwh) or (None, None)
        """
        try:
            if not hasattr(self.ml_predictor, "last_hourly_predictions"):
                return None, None

            hourly_preds = self.ml_predictor.last_hourly_predictions
            if not hourly_preds:
                _LOGGER.debug("No ML hourly predictions available")
                return None, None

            from ..core.core_helpers import SafeDateTimeUtil as dt_util

            today = dt_util.now().date()

            # Filter predictions for today only
            today_preds = {}
            for pred in hourly_preds:
                hour_dt = pred.get("datetime")
                if not hour_dt:
                    continue

                if isinstance(hour_dt, str):
                    hour_dt = dt_util.parse_datetime(hour_dt)

                if not hour_dt:
                    continue

                # CRITICAL: Convert to local timezone before extracting hour
                hour_dt_local = dt_util.as_local(hour_dt)

                if hour_dt_local.date() != today:
                    continue

                hour = hour_dt_local.hour  # LOCAL hour!
                prediction_kwh = pred.get("prediction_kwh", 0.0)

                # Filter by sun times if available
                if sunrise and sunset:
                    if not self._is_hour_in_sun_window(hour, sunrise, sunset):
                        _LOGGER.debug(f"ML: Skipping hour {hour} (outside sun window)")
                        continue

                today_preds[hour] = prediction_kwh

            if not today_preds:
                _LOGGER.debug("No ML predictions for today within sun window")
                return None, None

            # Find hour with maximum prediction
            best_hour = max(today_preds, key=today_preds.get)
            best_kwh = today_preds[best_hour]

            _LOGGER.debug(
                f"ML best hour calculation: Hour {best_hour} with {best_kwh:.3f} kWh "
                f"(from {len(today_preds)} candidates)"
            )

            return best_hour, best_kwh

        except Exception as e:
            _LOGGER.error(f"Error calculating best hour from ML: {e}", exc_info=True)
            return None, None

    async def _calculate_from_profile(
        self, sunrise: Optional[datetime], sunset: Optional[datetime]
    ) -> Tuple[Optional[int], Optional[float]]:
        """Calculate best hour from historical profile

        Args:
            sunrise: Sunrise datetime (local) or None
            sunset: Sunset datetime (local) or None

        Returns:
            Tuple of (best_hour, avg_kwh) or (None, None)
        """
        try:
            profile_data = await self.data_manager.load_hourly_profile()

            if not profile_data or not hasattr(profile_data, "hourly_averages"):
                _LOGGER.debug("No hourly profile data available")
                return None, None

            hourly_averages = profile_data.hourly_averages
            if not hourly_averages:
                _LOGGER.debug("Hourly averages empty in profile")
                return None, None

            # Filter by sun times if available
            valid_hours = {}
            for hour_str, avg_kwh in hourly_averages.items():
                try:
                    hour = int(hour_str)

                    if sunrise and sunset:
                        if not self._is_hour_in_sun_window(hour, sunrise, sunset):
                            _LOGGER.debug(f"Profile: Skipping hour {hour} (outside sun window)")
                            continue

                    if avg_kwh > 0:
                        valid_hours[hour] = avg_kwh

                except (ValueError, TypeError):
                    continue

            if not valid_hours:
                _LOGGER.debug("No valid hours found in profile within sun window")
                return None, None

            # Find hour with maximum average
            best_hour = max(valid_hours, key=valid_hours.get)
            best_kwh = valid_hours[best_hour]

            _LOGGER.debug(
                f"Profile best hour calculation: Hour {best_hour} with {best_kwh:.3f} kWh avg "
                f"(from {len(valid_hours)} candidates)"
            )

            return best_hour, best_kwh

        except Exception as e:
            _LOGGER.error(f"Error calculating best hour from profile: {e}", exc_info=True)
            return None, None

    def _calculate_solar_noon(self, sunrise: datetime, sunset: datetime) -> int:
        """Calculate solar noon (midpoint between sunrise and sunset)

        Args:
            sunrise: Sunrise datetime (local)
            sunset: Sunset datetime (local)

        Returns:
            Hour of solar noon (0-23)
        """
        solar_noon_dt = sunrise + (sunset - sunrise) / 2
        solar_noon_hour = solar_noon_dt.hour

        _LOGGER.debug(f"Solar noon calculated at {solar_noon_hour:02d}:00")

        return solar_noon_hour

    def get_best_hour_display(self, hour: Optional[int]) -> str:
        """Format best hour for display

        Args:
            hour: Hour (0-23) or None

        Returns:
            Formatted string like "12:00" or "N/A"
        """
        if hour is None:
            return "N/A"
        return f"{hour:02d}:00"
