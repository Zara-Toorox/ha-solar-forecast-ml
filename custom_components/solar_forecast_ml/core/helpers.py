"""
Helper functions for Solar Forecast ML Integration.
Includes dependency checking and safe datetime utilities.

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
import asyncio
import importlib.util
import logging
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from homeassistant.core import HomeAssistant

# --- IMPORT HIER ENTFERNT ---
# try:
#     from importlib.metadata import version as get_version, PackageNotFoundError
# except ImportError:
#     from importlib_metadata import version as get_version, PackageNotFoundError # type: ignore
# --- ENDE ENTFERNUNG ---

_LOGGER = logging.getLogger(__name__)

# --- Safe Datetime Utility ---
try:
    from homeassistant.util import dt as ha_dt_util
    _HAS_HA_DT = True
    _LOGGER.debug("Using Home Assistant dt util for timezone handling.")
except (ImportError, AttributeError):
    _HAS_HA_DT = False
    _LOGGER.warning("Home Assistant dt util not found. Using standard datetime library.")

    def get_local_tz() -> Optional[tzinfo]:
        try:
            return datetime.now().astimezone().tzinfo
        except Exception:
            _LOGGER.warning("Could not determine local timezone, falling back to UTC.")
            return timezone.utc


class SafeDateTimeUtil:
    """Provides timezone-aware datetime functions using HA utils or standard library."""

    @staticmethod
    def utcnow() -> datetime:
        """Return the current time in UTC."""
        if _HAS_HA_DT:
            return ha_dt_util.utcnow()
        return datetime.now(timezone.utc)

    @staticmethod
    def now() -> datetime:
        """Return the current time in the local timezone."""
        if _HAS_HA_DT:
            return ha_dt_util.now()
        local_tz = get_local_tz()
        return datetime.now(local_tz)

    @staticmethod
    def as_local(dt: datetime) -> datetime:
        """Convert a timezone-aware datetime to local time."""
        if _HAS_HA_DT:
            return ha_dt_util.as_local(dt)
        local_tz = get_local_tz()
        if dt.tzinfo is None:
            _LOGGER.debug("as_local received naive datetime, assuming UTC.")
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(local_tz)

    @staticmethod
    def as_utc(dt: datetime) -> datetime:
        """Convert a timezone-aware datetime to UTC."""
        if dt.tzinfo is None:
            _LOGGER.warning("as_utc received naive datetime, assuming local timezone.")
            dt = SafeDateTimeUtil.ensure_local(dt)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def ensure_local(dt: datetime) -> datetime:
        """Ensure datetime is timezone-aware and in local timezone."""
        if dt.tzinfo is None:
            _LOGGER.debug("ensure_local: Naive datetime, localizing to local timezone.")
            if _HAS_HA_DT:
                return ha_dt_util.as_local(dt.replace(tzinfo=timezone.utc))
            local_tz = get_local_tz()
            return dt.replace(tzinfo=local_tz)
        return SafeDateTimeUtil.as_local(dt)

    @staticmethod
    def is_dst(dt: datetime) -> bool:
        """Check if the given datetime is in daylight saving time."""
        try:
            local_dt = SafeDateTimeUtil.ensure_local(dt)
            return bool(local_dt.dst())
        except (AttributeError, TypeError):
            _LOGGER.warning("Could not determine DST status for datetime.")
            return False

    @staticmethod
    def parse_datetime(dt_str: str) -> Optional[datetime]:
        """Parse an ISO 8601 datetime string into a timezone-aware datetime object."""
        if not dt_str or not isinstance(dt_str, str):
            return None
        try:
            if _HAS_HA_DT:
                return ha_dt_util.parse_datetime(dt_str)
            dt_str_adj = dt_str.replace('Z', '+00:00')
            dt = datetime.fromisoformat(dt_str_adj)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError) as e:
            _LOGGER.warning(f"Failed to parse datetime string '{dt_str}': {e}")
            return None

    @staticmethod
    def start_of_day(dt: Optional[datetime] = None) -> datetime:
        """Return the start of the day (00:00) for the given datetime in local timezone."""
        if dt is None:
            dt = SafeDateTimeUtil.now()
        else:
            dt = SafeDateTimeUtil.ensure_local(dt)
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def end_of_day(dt: Optional[datetime] = None) -> datetime:
        """Return the end of the day (23:59:59) for the given datetime in local timezone."""
        if dt is None:
            dt = SafeDateTimeUtil.now()
        else:
            dt = SafeDateTimeUtil.ensure_local(dt)
        return dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    @staticmethod
    def is_using_ha_time() -> bool:
        """Check if Home Assistant's datetime utility is being used."""
        return _HAS_HA_DT


# --- Dependency Checking ---
@dataclass
class DependencyStatus:
    """Represents the status of a checked Python dependency."""
    name: str
    required_version: str
    installed: bool
    installed_version: Optional[str] = None
    error_message: Optional[str] = None


class DependencyChecker:
    """Checks required Python dependencies without performing installation."""

    REQUIRED_PACKAGES = [
        ("numpy", "1.21.0"),
        ("aiofiles", "23.0.0")
    ]

    def __init__(self):
        self._last_check_results: Optional[Dict[str, DependencyStatus]] = None
        self._check_lock = asyncio.Lock()

    async def check_package_installed_async(
        self,
        package_name: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Asynchronously checks if a package is installed and gets its version."""
        def _sync_check() -> Tuple[bool, Optional[str], Optional[str]]:
            
            # +++ IMPORT HIER EINGEFÃƒÅ“GT +++
            # Wird erst importiert, wenn diese (ungenutzte) Funktion aufgerufen wird.
            # Verhindert den globalen Import-Crash.
            try:
                from importlib.metadata import version as get_version, PackageNotFoundError
            except ImportError:
                try:
                    from importlib_metadata import version as get_version, PackageNotFoundError # type: ignore
                except ImportError:
                    _LOGGER.error("Konnte weder 'importlib.metadata' noch 'importlib_metadata' importieren.")
                    def get_version(_): # type: ignore
                        raise PackageNotFoundError
            # +++ ENDE EINFÃƒÅ“GUNG +++
            
            try:
                spec = importlib.util.find_spec(package_name)
                if spec is None:
                    return False, None, "Package specification not found."
                try:
                    version_str = get_version(package_name)
                    return True, version_str, None
                except PackageNotFoundError:
                    return True, "unknown", "Package installed but version metadata missing."
                except Exception as meta_err:
                    return True, "unknown", f"Error getting version via metadata: {meta_err}"
            except (ImportError, ValueError, AttributeError, ModuleNotFoundError) as e:
                _LOGGER.debug(f"Package '{package_name}' check failed: {e}")
                return False, None, str(e)
            except Exception as e:
                _LOGGER.warning(f"Unexpected error during check of '{package_name}': {e}", exc_info=True)
                return False, None, f"Unexpected error: {e}"

        try:
            if sys.version_info >= (3, 9):
                return await asyncio.to_thread(_sync_check)
            else:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, _sync_check)
        except Exception as e:
            _LOGGER.error(f"Async check task for '{package_name}' failed: {e}", exc_info=True)
            return False, None, f"Async execution error: {e}"

    async def check_all_dependencies_async(self, hass: HomeAssistant) -> Dict[str, DependencyStatus]:
        """Checks all required dependencies asynchronously and caches the result."""
        async with self._check_lock:
            if self._last_check_results is not None:
                _LOGGER.debug("Returning cached dependency check results.")
                return self._last_check_results

            _LOGGER.info("Starting dependency check...")
            results: Dict[str, DependencyStatus] = {}

            check_tasks = []
            for package_name, min_version in self.REQUIRED_PACKAGES:
                task = self.check_package_installed_async(package_name)
                check_tasks.append((package_name, min_version, task))

            check_outcomes = await asyncio.gather(*(task for _, _, task in check_tasks), return_exceptions=True)

            for i, (package_name, min_version, _) in enumerate(check_tasks):
                outcome = check_outcomes[i]
                if isinstance(outcome, Exception):
                    _LOGGER.error(f"Error checking dependency '{package_name}': {outcome}", exc_info=outcome)
                    status = DependencyStatus(
                        name=package_name, required_version=min_version,
                        installed=False, error_message=f"Check failed: {outcome}"
                    )
                else:
                    installed, version, error_msg = outcome
                    status = DependencyStatus(
                        name=package_name, required_version=min_version,
                        installed=installed, installed_version=version,
                        error_message=error_msg if not installed else None
                    )

                results[package_name] = status

                if installed:
                    _LOGGER.debug(f"Dependency '{package_name}': Installed (Version: {version})")
                else:
                    _LOGGER.warning(f"Dependency '{package_name}': Missing (Required: >={min_version}, Error: {status.error_message})")

            self._last_check_results = results
            num_installed = sum(1 for s in results.values() if s.installed)
            _LOGGER.info(f"Dependency check completed: {num_installed}/{len(results)} required packages installed.")
            return results

    def get_missing_packages(self) -> List[str]:
        """Returns a list of package names that are missing based on the last check."""
        if self._last_check_results is None:
            _LOGGER.warning("Cannot get missing packages: dependency check has not been performed.")
            return []
        return [status.name for status in self._last_check_results.values() if not status.installed]

    def are_all_dependencies_installed(self) -> bool:
        """Checks if all required dependencies were found during the last check."""
        if self._last_check_results is None:
            _LOGGER.warning("Cannot determine if all dependencies are installed: check not performed.")
            return False
        return all(status.installed for status in self._last_check_results.values())


# --- Singleton Pattern for Checker ---
_global_checker_instance: Optional[DependencyChecker] = None
_global_checker_lock = asyncio.Lock()

async def get_dependency_checker_instance() -> DependencyChecker:
    """Returns the singleton instance of the DependencyChecker, creating it if necessary."""
    global _global_checker_instance
    if _global_checker_instance is None:
        async with _global_checker_lock:
            if _global_checker_instance is None:
                _global_checker_instance = DependencyChecker()
    return _global_checker_instance