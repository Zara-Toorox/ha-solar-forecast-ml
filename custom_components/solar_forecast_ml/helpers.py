"""
Helper functions for Solar Forecast ML Integration.
✅ PRODUCTION READY: Non-blocking async operations
✅ OPTIMIZED: Comprehensive Error Handling & Logging
Version 4.9.2 - importlib.metadata Fix + UTF-8 Encoding # by Zara

Copyright (C) 2025 Zara-Toorox

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

# Fix: importlib.metadata instead of getattr()
try:
    from importlib.metadata import version as get_version
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import version as get_version

_LOGGER = logging.getLogger(__name__)


try:
    from homeassistant.util import dt as ha_dt_util
    _HAS_HA_DT = True
except (ImportError, AttributeError):
    _HAS_HA_DT = False
    from datetime import datetime, timezone
    _LOGGER.warning("Home Assistant dt_util not available - Fallback to standard datetime") # Übersetzt


class SafeDateTimeUtil:
    
    @staticmethod
    def utcnow():
        if _HAS_HA_DT:
            return ha_dt_util.utcnow()
        return datetime.now(timezone.utc)
    
    @staticmethod
    def now():
        if _HAS_HA_DT:
            return ha_dt_util.now()
        return datetime.now().astimezone()
    
    @staticmethod
    def as_local(dt):
        if _HAS_HA_DT:
            return ha_dt_util.as_local(dt)
        return dt.astimezone()
    
    @staticmethod
    def parse_datetime(dt_str):
        if _HAS_HA_DT:
            return ha_dt_util.parse_datetime(dt_str)
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    
    @staticmethod
    def is_using_ha_time():
        return _HAS_HA_DT


@dataclass
class DependencyStatus:
    """Status of a Python dependency. # by Zara""" # Übersetzt
    name: str
    required_version: str
    installed: bool
    installed_version: Optional[str] = None
    error_message: Optional[str] = None


class DependencyChecker:
    """
    Checks Python dependencies without automatic installation.
    ✅ ASYNC: All blocking operations in executor
    ✅ CACHED: Reuse of check results
    # by Zara
    """ # Übersetzt
    
    # Required packages for ML features
    REQUIRED_PACKAGES = [
        ("numpy", "1.21.0"),
        ("aiofiles", "23.0.0")
    ]
    
    def __init__(self):
        """Initializes Dependency Checker. # by Zara""" # Übersetzt
        self._last_check: Optional[Dict[str, DependencyStatus]] = None
        self._check_lock = asyncio.Lock()  # Prevents simultaneous checks
        
    async def check_package_installed_async(
        self, 
        package_name: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Checks if a Python package is installed (NON-BLOCKING).
        
        ✅ ASYNC: Uses asyncio.to_thread for blocking operations
        
        Returns:
            Tuple[bool, Optional[str]]: (installed, version)
        # by Zara
        """ # Übersetzt
        def _sync_check() -> Tuple[bool, Optional[str]]:
            """Synchronous check in Thread Pool. # by Zara""" # Übersetzt
            try:
                # Non-blocking Spec-Check
                spec = importlib.util.find_spec(package_name)
                if spec is None:
                    return False, None
                
                # Determine version with importlib.metadata
                try:
                    version = get_version(package_name)
                    return True, version
                except Exception:
                    # Fallback for packages without metadata
                    return True, "unknown"
                    
            except (ImportError, ValueError, AttributeError, ModuleNotFoundError) as e:
                _LOGGER.debug(f"Package {package_name} not found: {e}") # Übersetzt
                return False, None
            except Exception as e:
                _LOGGER.warning(
                    f"Unexpected error during check of {package_name}: {e}" # Übersetzt
                )
                return False, None
        
        try:
            # Execute sync check in Thread Pool (non-blocking)
            return await asyncio.to_thread(_sync_check)
        except Exception as e:
            _LOGGER.error(
                f"❌ Async Check for {package_name} failed: {e}", # Übersetzt & Korrigiert
                exc_info=True
            )
            return False, None
    
    def check_package_installed_sync(
        self, 
        package_name: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Synchronous version for non-async contexts.
        
        ⚠️ LEGACY: Only for compatibility, use async version if possible
        
        Returns:
            Tuple[bool, Optional[str]]: (installed, version)
        # by Zara
        """ # Übersetzt & Korrigiert
        try:
            spec = importlib.util.find_spec(package_name)
            if spec is None:
                return False, None
            
            # Version with importlib.metadata
            try:
                version = get_version(package_name)
                return True, version
            except Exception:
                return True, "unknown"
                
        except (ImportError, ValueError, AttributeError, ModuleNotFoundError):
            return False, None
    
    async def check_all_dependencies_async(self) -> Dict[str, DependencyStatus]:
        """
        Checks all required dependencies (NON-BLOCKING).
        
        ✅ ASYNC: Parallel checks for better performance
        ✅ CACHED: Uses Lock for Thread-Safety
        
        Returns:
            Dict with DependencyStatus for each package
        # by Zara
        """ # Übersetzt & Korrigiert
        async with self._check_lock:
            _LOGGER.debug("🔬 Starting Dependency Check (async)...") # Übersetzt & Korrigiert
            
            results = {}
            
            # Create tasks for parallel checks
            check_tasks = []
            for package_name, min_version in self.REQUIRED_PACKAGES:
                task = self.check_package_installed_async(package_name)
                check_tasks.append((package_name, min_version, task))
            
            # Execute all checks in parallel
            for package_name, min_version, task in check_tasks:
                try:
                    installed, version = await task
                    
                    status = DependencyStatus(
                        name=package_name,
                        required_version=min_version,
                        installed=installed,
                        installed_version=version
                    )
                    
                    results[package_name] = status
                    
                    if installed:
                        _LOGGER.debug(
                            f"✅ {package_name} installed (Version: {version})" # Übersetzt & Korrigiert
                        )
                    else:
                        _LOGGER.debug(
                            f"❌ {package_name} missing (required: >={min_version})" # Übersetzt & Korrigiert
                        )
                        
                except Exception as e:
                    _LOGGER.error(
                        f"❌ Error during check of {package_name}: {e}", # Übersetzt & Korrigiert
                        exc_info=True
                    )
                    results[package_name] = DependencyStatus(
                        name=package_name,
                        required_version=min_version,
                        installed=False,
                        error_message=str(e)
                    )
            
            self._last_check = results
            _LOGGER.info(
                f"🔬✅ Dependency Check completed: " # Übersetzt & Korrigiert
                f"{sum(1 for s in results.values() if s.installed)}/{len(results)} installed"
            )
            return results
    
    def check_all_dependencies_sync(self) -> Dict[str, DependencyStatus]:
        """
        Synchronous version of the Dependency Check.
        
        ⚠️ LEGACY: Only for non-async contexts
        
        Returns:
            Dict with DependencyStatus for each package
        # by Zara
        """ # Übersetzt & Korrigiert
        results = {}
        
        for package_name, min_version in self.REQUIRED_PACKAGES:
            installed, version = self.check_package_installed_sync(package_name)
            
            status = DependencyStatus(
                name=package_name,
                required_version=min_version,
                installed=installed,
                installed_version=version
            )
            
            results[package_name] = status
            
            if installed:
                _LOGGER.debug(f"✅ {package_name} installed (Version: {version})") # Übersetzt & Korrigiert
            else:
                _LOGGER.debug(f"❌ {package_name} missing (required: >={min_version})") # Übersetzt & Korrigiert
        
        self._last_check = results
        return results
    
    def get_missing_packages(self) -> List[str]:
        """
        Returns a list of missing packages.
        
        Returns:
            List with names of missing packages
        # by Zara
        """ # Übersetzt
        if self._last_check is None:
            # Fallback to sync check if no check has run yet
            self.check_all_dependencies_sync()
        
        return [
            status.name 
            for status in self._last_check.values() 
            if not status.installed
        ]
    
    def are_all_dependencies_installed(self) -> bool:
        """
        Checks if all dependencies are installed.
        
        ⚠️ SYNC: Uses cached results or runs sync check
        
        Returns:
            True if all are installed, else False
        # by Zara
        """ # Übersetzt & Korrigiert
        if self._last_check is None:
            status = self.check_all_dependencies_sync()
        else:
            status = self._last_check
            
        return all(dep.installed for dep in status.values())
    
    def get_installation_command(self) -> str:
        """
        Returns pip install command for missing packages.
        
        Returns:
            String with pip install command
        # by Zara
        """ # Übersetzt
        missing = self.get_missing_packages()
        if not missing:
            return ""
        
        packages = " ".join([
            f"{name}>={version}" 
            for name, version in self.REQUIRED_PACKAGES 
            if name in missing
        ])
        
        return f"pip install --user {packages}"


class DependencyInstaller:
    """
    Installs missing Python dependencies via pip.
    ✅ ASYNC: Non-blocking subprocess execution
    ✅ PROGRESS: Callback support for UI updates
    ⚠️ ATTENTION: Does not work in all HA environments (Read-Only, etc.)
    # by Zara
    """ # Übersetzt & Korrigiert
    
    def __init__(self, checker: DependencyChecker):
        """Initializes Installer with Checker. # by Zara""" # Übersetzt
        self.checker = checker
        self._installing = False
        self._install_lock = asyncio.Lock()  # Prevents parallel installations
        
    async def install_missing_dependencies(
        self, 
        progress_callback=None
    ) -> Tuple[bool, str]:
        """
        Installs all missing dependencies.
        
        ✅ ASYNC: Non-blocking with subprocess
        ✅ SAFE: Lock prevents parallel installations
        
        Args:
            progress_callback: Optional callback for Progress-Updates
            
        Returns:
            Tuple[bool, str]: (success, message)
        # by Zara
        """ # Übersetzt & Korrigiert
        async with self._install_lock:
            if self._installing:
                return False, "Installation already in progress" # Übersetzt
            
            self._installing = True
            
            try:
                # Check which packages are missing
                missing = self.checker.get_missing_packages()
                
                if not missing:
                    return True, "All dependencies already installed" # Übersetzt
                
                _LOGGER.info(f"📦 Installing {len(missing)} packages: {', '.join(missing)}") # Übersetzt & Korrigiert
                
                # Install each package individually for better feedback
                results = []
                total = len(missing)
                
                for idx, package_name in enumerate(missing, 1):
                    # Find Required Version
                    required_version = next(
                        (v for n, v in self.checker.REQUIRED_PACKAGES if n == package_name),
                        "latest"
                    )
                    
                    # Progress Callback
                    if progress_callback:
                        progress = int((idx - 1) / total * 100)
                        try:
                            await progress_callback(f"Installing {package_name}...", progress) # Übersetzt
                        except Exception as e:
                            _LOGGER.warning(f"⚠️ Progress callback failed: {e}") # Übersetzt & Korrigiert
                    
                    # Install package
                    success, message = await self._install_package(package_name, required_version)
                    results.append((package_name, success, message))
                    
                    if not success:
                        _LOGGER.error(f"❌ Installation of {package_name} failed: {message}") # Übersetzt & Korrigiert
                    else:
                        _LOGGER.info(f"✅ {package_name} successfully installed") # Übersetzt & Korrigrigiert
                
                # Final Progress
                if progress_callback:
                    try:
                        await progress_callback("Installation complete", 100) # Übersetzt
                    except Exception as e:
                        _LOGGER.warning(f"⚠️ Final progress callback failed: {e}") # Übersetzt & Korrigiert
                
                # Check overall result
                all_successful = all(success for _, success, _ in results)
                
                if all_successful:
                    return True, "All dependencies successfully installed" # Übersetzt
                else:
                    failed = [name for name, success, _ in results if not success]
                    return False, f"Installation failed for: {', '.join(failed)}" # Übersetzt
                    
            except Exception as e:
                _LOGGER.error(f"❌ Unexpected error during installation: {e}", exc_info=True) # Übersetzt & Korrigiert
                return False, f"Installation error: {str(e)}" # Übersetzt
                
            finally:
                self._installing = False
    
    async def _install_package(
        self, 
        package_name: str, 
        min_version: str
    ) -> Tuple[bool, str]:
        """
        Installs a single Python package via pip.
        
        ✅ ASYNC: subprocess with asyncio
        ✅ TIMEOUT: 3 minutes per package
        
        Args:
            package_name: Name of the package
            min_version: Minimum version
            
        Returns:
            Tuple[bool, str]: (success, message)
        # by Zara
        """ # Übersetzt & Korrigiert
        package_spec = f"{package_name}>={min_version}"
        
        try:
            _LOGGER.info(f"📦⏳ Installing {package_spec}...") # Übersetzt & Korrigiert
            
            # Determine pip executable
            pip_cmd = [
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "--user",              # User installation for better compatibility
                "--no-cache-dir",      # No cache to save space
                "--quiet",             # Less Output
                "--disable-pip-version-check",  # No pip Update-Checks
                package_spec
            ]
            
            # Execute installation asynchronously with timeout
            try:
                process = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *pip_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    ),
                    timeout=180  # 3 minutes Timeout
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=180
                )
                
                if process.returncode == 0:
                    # Successfully installed
                    _LOGGER.debug(f"pip stdout: {stdout.decode()}")
                    return True, "Installation successful" # Übersetzt
                else:
                    # Installation failed
                    error_msg = stderr.decode() if stderr else "Unknown error" # Übersetzt
                    
                    # Parse common errors
                    if "permission denied" in error_msg.lower():
                        return False, "No write permissions (Permission Denied)" # Übersetzt
                    elif "read-only" in error_msg.lower():
                        return False, "Read-Only file system" # Übersetzt
                    elif "no space" in error_msg.lower():
                        return False, "Not enough storage space" # Übersetzt
                    else:
                        return False, f"pip error: {error_msg[:200]}"
                        
            except asyncio.TimeoutError:
                return False, "Timeout (>3 minutes)" # Übersetzt
                
        except Exception as e:
            return False, f"Exception: {str(e)}"


def get_manual_install_instructions() -> str:
    """
    Returns manual installation instructions.
    
    Returns:
        Formatted string with instructions
    # by Zara
    """ # Übersetzt
    return """
MANUAL INSTALLATION:

For Home Assistant OS / Supervised (Docker):
--------------------------------------------
1. Install and open the Terminal & SSH Add-on
2. Execute the following commands:

   docker exec -it homeassistant bash
   pip install --user numpy>=1.21.0 aiofiles>=23.0.0
   exit

3. Restart Home Assistant


For Home Assistant Container (Docker):
---------------------------------------
1. Log into the container:

   docker exec -it homeassistant bash

2. Install packages:

   pip install --user numpy>=1.21.0 aiofiles>=23.0.0

3. Restart the container:

   docker restart homeassistant


For Home Assistant Core (venv):
--------------------------------
1. Activate venv:

   cd /srv/homeassistant
   source bin/activate

2. Install packages:

   pip install numpy>=1.21.0 aiofiles>=23.0.0

3. Restart Home Assistant:

   systemctl restart home-assistant@homeassistant


VERIFICATION:
--------
After installation, check:

python3 -c "import numpy, aiofiles; print('✅ OK')"

Then restart Home Assistant.
""" # Komplett übersetzt und korrigiert


# Global Checker Instance (Singleton)
_global_checker: Optional[DependencyChecker] = None
_checker_lock = asyncio.Lock()


def get_dependency_checker() -> DependencyChecker:
    """
    Returns global Dependency Checker instance (Singleton).
    
    ✅ SYNC: For compatibility with existing code
    
    Returns:
        DependencyChecker instance
    # by Zara
    """ # Übersetzt & Korrigiert
    global _global_checker
    if _global_checker is None:
        _global_checker = DependencyChecker()
    return _global_checker


async def get_dependency_checker_async() -> DependencyChecker:
    """
    Returns global Dependency Checker instance (Singleton, async-safe).
    
    ✅ ASYNC: Thread-safe with Lock
    
    Returns:
        DependencyChecker instance
    # by Zara
    """ # Übersetzt & Korrigiert
    global _global_checker
    async with _checker_lock:
        if _global_checker is None:
            _global_checker = DependencyChecker()
        return _global_checker