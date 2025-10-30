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
from datetime import datetime, timezone, tzinfo # Import base datetime types unconditionally
# --- HIER HINZUFÜGEN ---
from homeassistant.core import HomeAssistant # Import HomeAssistant for type hinting
# ----------------------

# Use importlib.metadata for version checking (Python 3.8+)
try:
    from importlib.metadata import version as get_version, PackageNotFoundError
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import version as get_version, PackageNotFoundError # type: ignore

_LOGGER = logging.getLogger(__name__)


# --- Safe Datetime Utility ---
# Provides timezone-aware datetime objects, falling back if HA utils unavailable
try:
    # Use Home Assistant's datetime utility if available
    from homeassistant.util import dt as ha_dt_util
    _HAS_HA_DT = True
    _LOGGER.debug("Using Home Assistant dt util for timezone handling.")
except (ImportError, AttributeError):
    _HAS_HA_DT = False
    # NO 'from datetime import...' HERE ANYMORE
    _LOGGER.warning("Home Assistant dt util not found. Using standard datetime library. "
                    "Ensure system timezone is correctly configured.")

    # Basic local timezone getter if HA utils are missing
    def get_local_tz() -> Optional[tzinfo]:
        try:
            # Attempt to get local timezone from system
            # Ensure we are using the imported datetime
            return datetime.now().astimezone().tzinfo
        except Exception:
            # Fallback to UTC if local cannot be determined
            _LOGGER.warning("Could not determine local timezone, falling back to UTC.")
            return timezone.utc


class SafeDateTimeUtil:
    """Provides timezone-aware datetime functions, using HA utils or standard library."""

    @staticmethod
    def utcnow() -> datetime: # Now 'datetime' is defined
        """Return the current time in UTC."""
        if _HAS_HA_DT:
            return ha_dt_util.utcnow()
        return datetime.now(timezone.utc)

    @staticmethod
    def now() -> datetime: # Now 'datetime' is defined
        """Return the current time in the local timezone."""
        if _HAS_HA_DT:
            return ha_dt_util.now()
        # Fallback using standard library
        local_tz = get_local_tz()
        return datetime.now(local_tz)

    @staticmethod
    def as_local(dt: datetime) -> datetime: # Now 'datetime' is defined
        """Convert a timezone-aware datetime to local time."""
        if _HAS_HA_DT:
            return ha_dt_util.as_local(dt)
        # Fallback using standard library
        local_tz = get_local_tz()
        if dt.tzinfo is None:
            # Assume naive datetime is UTC if HA utils absent, otherwise it's ambiguous
            _LOGGER.debug("as_local received naive datetime, assuming UTC.")
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(local_tz)

    @staticmethod
    def parse_datetime(dt_str: str) -> Optional[datetime]: # Now 'datetime' is defined
        """Parse an ISO 8601 datetime string into a timezone-aware datetime object."""
        if not dt_str or not isinstance(dt_str, str):
            return None
        try:
            if _HAS_HA_DT:
                return ha_dt_util.parse_datetime(dt_str)
            # Fallback using standard library (supports 'Z' and +/- offsets)
            # Handle potential 'Z' for UTC
            dt_str_adj = dt_str.replace('Z', '+00:00')
            # Use the imported datetime class
            dt = datetime.fromisoformat(dt_str_adj)
            # Ensure timezone info (fromisoformat might return naive if no offset)
            if dt.tzinfo is None:
                 # Assume UTC if no timezone info parsed (though ISO usually has it)
                 return dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError) as e:
            _LOGGER.warning(f"Failed to parse datetime string '{dt_str}': {e}")
            return None

    @staticmethod
    def is_using_ha_time() -> bool:
        """Check if Home Assistant's datetime utility is being used."""
        return _HAS_HA_DT


# --- Dependency Checking ---
@dataclass
class DependencyStatus:
    """Represents the status of a checked Python dependency."""
    name: str
    required_version: str # Minimum required version string
    installed: bool
    installed_version: Optional[str] = None
    error_message: Optional[str] = None # If checking failed


class DependencyChecker:
    """
    Checks required Python dependencies without performing installation.
    Uses non-blocking checks run in an executor. Caches results.
    """

    # Define required packages and their minimum versions
    REQUIRED_PACKAGES = [
        ("numpy", "1.21.0"),
        ("aiofiles", "23.0.0")
        # Add other dependencies here if needed
    ]

    def __init__(self):
        """Initializes the Dependency Checker."""
        self._last_check_results: Optional[Dict[str, DependencyStatus]] = None
        self._check_lock = asyncio.Lock() # Ensures only one check runs at a time

    async def check_package_installed_async(
        self,
        package_name: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Asynchronously checks if a package is installed and gets its version.
        Runs blocking checks in an executor thread.

        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (installed, version, error_message)
        """
        def _sync_check() -> Tuple[bool, Optional[str], Optional[str]]:
            """Synchronous check function to run in executor."""
            try:
                # Check if module can be found
                spec = importlib.util.find_spec(package_name)
                if spec is None:
                    return False, None, "Package specification not found."

                # Try to get version using importlib.metadata
                try:
                    version_str = get_version(package_name)
                    return True, version_str, None
                except PackageNotFoundError:
                    # Package spec found, but metadata missing? Might be partially installed.
                    # Consider it installed but version unknown.
                    return True, "unknown", "Package installed but version metadata missing."
                except Exception as meta_err:
                     # Catch other metadata errors
                     return True, "unknown", f"Error getting version via metadata: {meta_err}"

            except (ImportError, ValueError, AttributeError, ModuleNotFoundError) as e:
                # Catch errors indicating the package is fundamentally missing or broken
                _LOGGER.debug(f"Package '{package_name}' check failed (likely not installed): {e}")
                return False, None, str(e)
            except Exception as e:
                # Catch unexpected errors during the check
                _LOGGER.warning(f"Unexpected error during check of '{package_name}': {e}", exc_info=True)
                return False, None, f"Unexpected error: {e}"

        try:
            # Run the synchronous check in HA's executor thread pool
            # Use asyncio.to_thread for cleaner execution in Python 3.9+
            if sys.version_info >= (3, 9):
                 return await asyncio.to_thread(_sync_check)
            else:
                 # Fallback for older Python versions
                 loop = asyncio.get_running_loop()
                 return await loop.run_in_executor(None, _sync_check)

        except Exception as e:
            _LOGGER.error(f"Async check task for '{package_name}' failed: {e}", exc_info=True)
            return False, None, f"Async execution error: {e}"


    async def check_all_dependencies_async(self, hass: HomeAssistant) -> Dict[str, DependencyStatus]: # <-- HomeAssistant type hint is now valid
        """
        Checks all required dependencies asynchronously and caches the result.

        Args:
            hass: HomeAssistant instance (needed if run_in_executor fallback is used).

        Returns:
            Dictionary mapping package name to its DependencyStatus.
        """
        async with self._check_lock: # Prevent concurrent checks
            # Return cached results if available
            if self._last_check_results is not None:
                _LOGGER.debug("Returning cached dependency check results.")
                return self._last_check_results

            _LOGGER.info("Starting dependency check...")
            results: Dict[str, DependencyStatus] = {}

            # Create tasks for parallel checks
            check_tasks = []
            for package_name, min_version in self.REQUIRED_PACKAGES:
                # Create a coroutine for each check
                task = self.check_package_installed_async(package_name)
                check_tasks.append((package_name, min_version, task))

            # Execute all checks concurrently
            check_outcomes = await asyncio.gather(*(task for _, _, task in check_tasks), return_exceptions=True)

            # Process results
            for i, (package_name, min_version, _) in enumerate(check_tasks):
                outcome = check_outcomes[i]
                if isinstance(outcome, Exception):
                    # Handle unexpected errors during the check task itself
                    _LOGGER.error(f"Error checking dependency '{package_name}': {outcome}", exc_info=outcome)
                    status = DependencyStatus(
                        name=package_name, required_version=min_version,
                        installed=False, error_message=f"Check failed: {outcome}"
                    )
                else:
                    # Unpack the result from check_package_installed_async
                    installed, version, error_msg = outcome
                    status = DependencyStatus(
                        name=package_name, required_version=min_version,
                        installed=installed, installed_version=version,
                        error_message=error_msg if not installed else None # Only store error if not installed
                    )

                results[package_name] = status

                # Log individual package status
                if installed:
                    _LOGGER.debug(f"Dependency '{package_name}': Installed (Version: {version})")
                else:
                    _LOGGER.warning(f"Dependency '{package_name}': Missing or check failed (Required: >={min_version}, Error: {status.error_message})")

            # Cache the results
            self._last_check_results = results
            num_installed = sum(1 for s in results.values() if s.installed)
            _LOGGER.info(f"Dependency check completed: {num_installed}/{len(results)} required packages installed.")
            return results


    def get_missing_packages(self) -> List[str]:
        """
        Returns a list of package names that are missing based on the last check.
        Requires check_all_dependencies_async to have been run first.

        Returns:
            List of names of missing packages. Returns empty list if check hasn't run.
        """
        if self._last_check_results is None:
            _LOGGER.warning("Cannot get missing packages: dependency check has not been performed.")
            return []

        return [
            status.name
            for status in self._last_check_results.values()
            if not status.installed
        ]

    def are_all_dependencies_installed(self) -> bool:
        """
        Checks if all required dependencies were found during the last check.
        Requires check_all_dependencies_async to have been run first.

        Returns:
            True if all dependencies are installed, False otherwise or if check hasn't run.
        """
        if self._last_check_results is None:
            _LOGGER.warning("Cannot determine if all dependencies are installed: check not performed.")
            return False # Fail safe

        return all(status.installed for status in self._last_check_results.values())


# --- Singleton Pattern for Checker ---
# Provides a single instance accessible throughout the integration
_global_checker_instance: Optional[DependencyChecker] = None
_global_checker_lock = asyncio.Lock()

async def get_dependency_checker_instance() -> DependencyChecker:
    """
    Returns the singleton instance of the DependencyChecker, creating it if necessary.
    Async-safe.
    """
    global _global_checker_instance
    if _global_checker_instance is None:
        async with _global_checker_lock:
            # Double-check inside lock
            if _global_checker_instance is None:
                _global_checker_instance = DependencyChecker()
    return _global_checker_instance