# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Core Helper Utilities for Solar Forecast ML V16.2.0.
Provides SafeDateTimeUtil for timezone-aware datetime operations.
Pure utility functions with no database dependencies.
"""

import logging
from datetime import datetime, timezone, tzinfo
from typing import Optional

_LOGGER = logging.getLogger(__name__)


def get_local_tz() -> Optional[tzinfo]:
    """Get local timezone, falling back to UTC if unavailable. @zara"""
    try:
        return datetime.now().astimezone().tzinfo
    except Exception:
        _LOGGER.warning("Could not determine local timezone, falling back to UTC")
        return timezone.utc


try:
    from homeassistant.util import dt as ha_dt_util
    _HAS_HA_DT = True
    _LOGGER.debug("Using Home Assistant dt util for timezone handling")
except (ImportError, AttributeError):
    _HAS_HA_DT = False
    _LOGGER.warning("Home Assistant dt util not found, using standard datetime library")


class SafeDateTimeUtil:
    """Provides timezone-aware datetime functions using HA utils or standard library. @zara"""

    @staticmethod
    def utcnow() -> datetime:
        """Return the current time in UTC. @zara"""
        if _HAS_HA_DT:
            return ha_dt_util.utcnow()
        return datetime.now(timezone.utc)

    @staticmethod
    def now() -> datetime:
        """Return the current time in the local timezone. @zara"""
        if _HAS_HA_DT:
            try:
                result = ha_dt_util.now()
                if result is not None:
                    return result
                _LOGGER.warning("HA dt_util.now() returned None, using fallback")
            except Exception as e:
                _LOGGER.warning(f"HA dt_util.now() raised exception: {e}, using fallback")

        local_tz = get_local_tz()
        return datetime.now(local_tz)

    @staticmethod
    def as_local(dt: datetime) -> datetime:
        """Convert a timezone-aware datetime to local time. @zara"""
        if _HAS_HA_DT:
            return ha_dt_util.as_local(dt)
        local_tz = get_local_tz()
        if dt.tzinfo is None:
            _LOGGER.debug("as_local received naive datetime, assuming UTC")
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(local_tz)

    @staticmethod
    def as_utc(dt: datetime) -> datetime:
        """Convert a timezone-aware datetime to UTC. @zara"""
        if dt.tzinfo is None:
            _LOGGER.warning("as_utc received naive datetime, assuming local timezone")
            dt = SafeDateTimeUtil.ensure_local(dt)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def ensure_local(dt: datetime) -> datetime:
        """Ensure datetime is timezone-aware and in local timezone. @zara"""
        if dt.tzinfo is None:
            _LOGGER.debug("ensure_local: Naive datetime, localizing to local timezone")
            if _HAS_HA_DT:
                return ha_dt_util.as_local(dt.replace(tzinfo=timezone.utc))
            local_tz = get_local_tz()
            return dt.replace(tzinfo=local_tz)
        return SafeDateTimeUtil.as_local(dt)

    @staticmethod
    def is_dst(dt: datetime) -> bool:
        """Check if the given datetime is in daylight saving time. @zara"""
        try:
            local_dt = SafeDateTimeUtil.ensure_local(dt)
            return bool(local_dt.dst())
        except (AttributeError, TypeError):
            _LOGGER.warning("Could not determine DST status for datetime")
            return False

    @staticmethod
    def parse_datetime(dt_str: str) -> Optional[datetime]:
        """Parse an ISO 8601 datetime string into a timezone-aware datetime object. @zara"""
        if not dt_str or not isinstance(dt_str, str):
            return None
        try:
            if _HAS_HA_DT:
                return ha_dt_util.parse_datetime(dt_str)
            dt_str_adj = dt_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(dt_str_adj)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError) as e:
            _LOGGER.warning(f"Failed to parse datetime string '{dt_str}': {e}")
            return None

    @staticmethod
    def start_of_day(dt: Optional[datetime] = None) -> datetime:
        """Return the start of the day 00:00 for the given datetime in local timezone. @zara"""
        if dt is None:
            dt = SafeDateTimeUtil.now()
        else:
            dt = SafeDateTimeUtil.ensure_local(dt)
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def end_of_day(dt: Optional[datetime] = None) -> datetime:
        """Return the end of the day 23:59:59 for the given datetime in local timezone. @zara"""
        if dt is None:
            dt = SafeDateTimeUtil.now()
        else:
            dt = SafeDateTimeUtil.ensure_local(dt)
        return dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    @staticmethod
    def get_default_time_zone() -> tzinfo:
        """Get the default timezone from Home Assistant or system. @zara"""
        if _HAS_HA_DT:
            try:
                return ha_dt_util.get_default_time_zone()
            except (AttributeError, Exception) as e:
                _LOGGER.warning(f"Failed to get HA default timezone: {e}, using fallback")

        local_tz = get_local_tz()
        if local_tz is None:
            _LOGGER.warning("Could not determine local timezone, using UTC")
            return timezone.utc
        return local_tz

    @staticmethod
    def is_using_ha_time() -> bool:
        """Check if Home Assistant datetime utility is being used. @zara"""
        return _HAS_HA_DT


def get_season(month: int) -> str:
    """Get season from month number. Uses WINTER_MONTHS from const.py as SSOT. @zara

    Args:
        month: Month number (1-12)

    Returns:
        Season name: 'winter', 'spring', 'summer', or 'autumn'
    """
    from ..const import WINTER_MONTHS

    if month in WINTER_MONTHS:
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:
        return "autumn"
