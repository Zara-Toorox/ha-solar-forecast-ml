"""
Helper-Funktionen fÃ¼r Solar Forecast ML Integration.
Ã¢Å“â€œ PRODUCTION READY: Non-blocking async operations
Ã¢Å“â€œ OPTIMIERT: Comprehensive Error Handling & Logging
Version 4.9.2 - importlib.metadata Fix + UTF-8 Encoding # von Zara

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

# Fix: importlib.metadata statt getattr()
try:
    from importlib.metadata import version as get_version
except ImportError:
    # Fallback fÃ¼r Python < 3.8
    from importlib_metadata import version as get_version

_LOGGER = logging.getLogger(__name__)


@dataclass
class DependencyStatus:
    """Status einer Python-AbhÃ¤ngigkeit. # von Zara"""
    name: str
    required_version: str
    installed: bool
    installed_version: Optional[str] = None
    error_message: Optional[str] = None


class DependencyChecker:
    """
    PrÃ¼ft Python-AbhÃ¤ngigkeiten ohne automatische Installation.
    Ã¢Å“â€œ ASYNC: Alle blocking operations in executor
    Ã¢Å“â€œ CACHED: Wiederverwendung von Check-Ergebnissen
    # von Zara
    """
    
    # Erforderliche Pakete fÃ¼r ML-Features
    REQUIRED_PACKAGES = [
        ("numpy", "1.21.0"),
        ("aiofiles", "23.0.0")
    ]
    
    def __init__(self):
        """Initialisiert Dependency Checker. # von Zara"""
        self._last_check: Optional[Dict[str, DependencyStatus]] = None
        self._check_lock = asyncio.Lock()  # Verhindert gleichzeitige Checks
        
    async def check_package_installed_async(
        self, 
        package_name: str
    ) -> Tuple[bool, Optional[str]]:
        """
        PrÃ¼ft ob ein Python-Paket installiert ist (NON-BLOCKING).
        
        Ã¢Å“â€œ ASYNC: Verwendet asyncio.to_thread fÃ¼r blocking operations
        
        Returns:
            Tuple[bool, Optional[str]]: (installiert, version)
        # von Zara
        """
        def _sync_check() -> Tuple[bool, Optional[str]]:
            """Synchroner Check im Thread Pool. # von Zara"""
            try:
                # Nicht-blockierender Spec-Check
                spec = importlib.util.find_spec(package_name)
                if spec is None:
                    return False, None
                
                # Version ermitteln mit importlib.metadata
                try:
                    version = get_version(package_name)
                    return True, version
                except Exception:
                    # Fallback fÃ¼r Packages ohne Metadaten
                    return True, "unknown"
                    
            except (ImportError, ValueError, AttributeError, ModuleNotFoundError) as e:
                _LOGGER.debug(f"Paket {package_name} nicht gefunden: {e}")
                return False, None
            except Exception as e:
                _LOGGER.warning(
                    f"Unerwarteter Fehler bei Check von {package_name}: {e}"
                )
                return False, None
        
        try:
            # FÃ¼hre sync Check in Thread Pool aus (non-blocking)
            return await asyncio.to_thread(_sync_check)
        except Exception as e:
            _LOGGER.error(
                f"Ã¢ÂÅ’ Async Check fÃ¼r {package_name} fehlgeschlagen: {e}",
                exc_info=True
            )
            return False, None
    
    def check_package_installed_sync(
        self, 
        package_name: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Synchrone Variante fÃ¼r nicht-async Kontexte.
        
        Ã¢Å¡Â Ã¯Â¸Â LEGACY: Nur fÃ¼r KompatibilitÃ¤t, verwende async Version wenn mÃ¶glich
        
        Returns:
            Tuple[bool, Optional[str]]: (installiert, version)
        # von Zara
        """
        try:
            spec = importlib.util.find_spec(package_name)
            if spec is None:
                return False, None
            
            # Version mit importlib.metadata
            try:
                version = get_version(package_name)
                return True, version
            except Exception:
                return True, "unknown"
                
        except (ImportError, ValueError, AttributeError, ModuleNotFoundError):
            return False, None
    
    async def check_all_dependencies_async(self) -> Dict[str, DependencyStatus]:
        """
        PrÃ¼ft alle erforderlichen AbhÃ¤ngigkeiten (NON-BLOCKING).
        
        Ã¢Å“â€œ ASYNC: Parallele Checks fÃ¼r bessere Performance
        Ã¢Å“â€œ CACHED: Verwendet Lock fÃ¼r Thread-Safety
        
        Returns:
            Dict mit DependencyStatus fÃ¼r jedes Paket
        # von Zara
        """
        async with self._check_lock:
            _LOGGER.debug("Ã°Å¸â€Â Starte Dependency Check (async)...")
            
            results = {}
            
            # Erstelle Tasks fÃ¼r parallele Checks
            check_tasks = []
            for package_name, min_version in self.REQUIRED_PACKAGES:
                task = self.check_package_installed_async(package_name)
                check_tasks.append((package_name, min_version, task))
            
            # FÃ¼hre alle Checks parallel aus
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
                            f"Ã¢Å“â€œ {package_name} installiert (Version: {version})"
                        )
                    else:
                        _LOGGER.debug(
                            f"Ã¢ÂÅ’ {package_name} fehlt (benÃ¶tigt: >={min_version})"
                        )
                        
                except Exception as e:
                    _LOGGER.error(
                        f"Ã¢ÂÅ’ Fehler bei Check von {package_name}: {e}",
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
                f"Ã°Å¸â€ÂÃ¢Å“â€œ Dependency Check abgeschlossen: "
                f"{sum(1 for s in results.values() if s.installed)}/{len(results)} installiert"
            )
            return results
    
    def check_all_dependencies_sync(self) -> Dict[str, DependencyStatus]:
        """
        Synchrone Variante des Dependency Checks.
        
        Ã¢Å¡Â Ã¯Â¸Â LEGACY: Nur fÃ¼r nicht-async Kontexte
        
        Returns:
            Dict mit DependencyStatus fÃ¼r jedes Paket
        # von Zara
        """
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
                _LOGGER.debug(f"Ã¢Å“â€œ {package_name} installiert (Version: {version})")
            else:
                _LOGGER.debug(f"Ã¢ÂÅ’ {package_name} fehlt (benÃ¶tigt: >={min_version})")
        
        self._last_check = results
        return results
    
    def get_missing_packages(self) -> List[str]:
        """
        Gibt Liste der fehlenden Pakete zurÃ¼ck.
        
        Returns:
            Liste mit Namen der fehlenden Pakete
        # von Zara
        """
        if self._last_check is None:
            # Fallback auf sync check wenn noch kein Check lief
            self.check_all_dependencies_sync()
        
        return [
            status.name 
            for status in self._last_check.values() 
            if not status.installed
        ]
    
    def are_all_dependencies_installed(self) -> bool:
        """
        PrÃ¼ft ob alle AbhÃ¤ngigkeiten installiert sind.
        
        Ã¢Å¡Â Ã¯Â¸Â SYNC: Verwendet cached results oder fÃ¼hrt sync check aus
        
        Returns:
            True wenn alle installiert, sonst False
        # von Zara
        """
        if self._last_check is None:
            status = self.check_all_dependencies_sync()
        else:
            status = self._last_check
            
        return all(dep.installed for dep in status.values())
    
    def get_installation_command(self) -> str:
        """
        Gibt pip install Command fÃ¼r fehlende Pakete zurÃ¼ck.
        
        Returns:
            String mit pip install command
        # von Zara
        """
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
    Installiert fehlende Python-AbhÃ¤ngigkeiten via pip.
    Ã¢Å“â€œ ASYNC: Non-blocking subprocess execution
    Ã¢Å“â€œ PROGRESS: Callback-Support fÃ¼r UI-Updates
    Ã¢Å¡Â Ã¯Â¸Â ACHTUNG: Funktioniert nicht in allen HA-Umgebungen (Read-Only, etc.)
    # von Zara
    """
    
    def __init__(self, checker: DependencyChecker):
        """Initialisiert Installer mit Checker. # von Zara"""
        self.checker = checker
        self._installing = False
        self._install_lock = asyncio.Lock()  # Verhindert parallele Installationen
        
    async def install_missing_dependencies(
        self, 
        progress_callback=None
    ) -> Tuple[bool, str]:
        """
        Installiert alle fehlenden AbhÃ¤ngigkeiten.
        
        Ã¢Å“â€œ ASYNC: Non-blocking mit subprocess
        Ã¢Å“â€œ SAFE: Lock verhindert parallele Installationen
        
        Args:
            progress_callback: Optional callback fÃ¼r Progress-Updates
            
        Returns:
            Tuple[bool, str]: (success, message)
        # von Zara
        """
        async with self._install_lock:
            if self._installing:
                return False, "Installation lÃ¤uft bereits"
            
            self._installing = True
            
            try:
                # PrÃ¼fe welche Pakete fehlen
                missing = self.checker.get_missing_packages()
                
                if not missing:
                    return True, "Alle AbhÃ¤ngigkeiten bereits installiert"
                
                _LOGGER.info(f"Ã°Å¸â€Â½ Installiere {len(missing)} Pakete: {', '.join(missing)}")
                
                # Installiere jedes Paket einzeln fÃ¼r besseres Feedback
                results = []
                total = len(missing)
                
                for idx, package_name in enumerate(missing, 1):
                    # Finde Required Version
                    required_version = next(
                        (v for n, v in self.checker.REQUIRED_PACKAGES if n == package_name),
                        "latest"
                    )
                    
                    # Progress Callback
                    if progress_callback:
                        progress = int((idx - 1) / total * 100)
                        try:
                            await progress_callback(f"Installiere {package_name}...", progress)
                        except Exception as e:
                            _LOGGER.warning(f"Ã¢Å¡Â Ã¯Â¸Â Progress callback failed: {e}")
                    
                    # Installiere Paket
                    success, message = await self._install_package(package_name, required_version)
                    results.append((package_name, success, message))
                    
                    if not success:
                        _LOGGER.error(f"Ã¢ÂÅ’ Installation von {package_name} fehlgeschlagen: {message}")
                    else:
                        _LOGGER.info(f"Ã¢Å“â€œ {package_name} erfolgreich installiert")
                
                # Finaler Progress
                if progress_callback:
                    try:
                        await progress_callback("Installation abgeschlossen", 100)
                    except Exception as e:
                        _LOGGER.warning(f"Ã¢Å¡Â Ã¯Â¸Â Final progress callback failed: {e}")
                
                # PrÃ¼fe Gesamt-Ergebnis
                all_successful = all(success for _, success, _ in results)
                
                if all_successful:
                    return True, "Alle AbhÃ¤ngigkeiten erfolgreich installiert"
                else:
                    failed = [name for name, success, _ in results if not success]
                    return False, f"Installation fehlgeschlagen fÃ¼r: {', '.join(failed)}"
                    
            except Exception as e:
                _LOGGER.error(f"Ã¢ÂÅ’ Unerwarteter Fehler bei Installation: {e}", exc_info=True)
                return False, f"Installationsfehler: {str(e)}"
                
            finally:
                self._installing = False
    
    async def _install_package(
        self, 
        package_name: str, 
        min_version: str
    ) -> Tuple[bool, str]:
        """
        Installiert ein einzelnes Python-Paket via pip.
        
        Ã¢Å“â€œ ASYNC: subprocess mit asyncio
        Ã¢Å“â€œ TIMEOUT: 3 Minuten pro Paket
        
        Args:
            package_name: Name des Pakets
            min_version: Minimale Version
            
        Returns:
            Tuple[bool, str]: (success, message)
        # von Zara
        """
        package_spec = f"{package_name}>={min_version}"
        
        try:
            _LOGGER.info(f"Ã°Å¸â€Â½Ã°Å¸â€œÂ¦ Installiere {package_spec}...")
            
            # Bestimme pip executable
            pip_cmd = [
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "--user",              # User-Installation fÃ¼r bessere KompatibilitÃ¤t
                "--no-cache-dir",      # Kein Cache um Platz zu sparen
                "--quiet",             # Weniger Output
                "--disable-pip-version-check",  # Keine pip Update-Checks
                package_spec
            ]
            
            # FÃ¼hre Installation asynchron aus mit Timeout
            try:
                process = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *pip_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    ),
                    timeout=180  # 3 Minuten Timeout
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=180
                )
                
                if process.returncode == 0:
                    # Erfolgreich installiert
                    _LOGGER.debug(f"pip stdout: {stdout.decode()}")
                    return True, "Installation erfolgreich"
                else:
                    # Installation fehlgeschlagen
                    error_msg = stderr.decode() if stderr else "Unbekannter Fehler"
                    
                    # Parse hÃ¤ufige Fehler
                    if "permission denied" in error_msg.lower():
                        return False, "Keine Schreibrechte (Permission Denied)"
                    elif "read-only" in error_msg.lower():
                        return False, "Read-Only Dateisystem"
                    elif "no space" in error_msg.lower():
                        return False, "Nicht genug Speicherplatz"
                    else:
                        return False, f"pip error: {error_msg[:200]}"
                        
            except asyncio.TimeoutError:
                return False, "Timeout (>3 Minuten)"
                
        except Exception as e:
            return False, f"Exception: {str(e)}"


def get_manual_install_instructions() -> str:
    """
    Gibt manuelle Installationsanweisungen zurÃ¼ck.
    
    Returns:
        Formatierter String mit Anweisungen
    # von Zara
    """
    return """
MANUELLE INSTALLATION:

FÃ¼r Home Assistant OS / Supervised (Docker):
--------------------------------------------
1. Terminal & SSH Add-on installieren und Ã¶ffnen
2. Folgende Befehle ausfÃ¼hren:

   docker exec -it homeassistant bash
   pip install --user numpy>=1.21.0 aiofiles>=23.0.0
   exit

3. Home Assistant neu starten


FÃ¼r Home Assistant Container (Docker):
---------------------------------------
1. In Container einloggen:

   docker exec -it homeassistant bash

2. Pakete installieren:

   pip install --user numpy>=1.21.0 aiofiles>=23.0.0

3. Container neu starten:

   docker restart homeassistant


FÃ¼r Home Assistant Core (venv):
--------------------------------
1. In venv aktivieren:

   cd /srv/homeassistant
   source bin/activate

2. Pakete installieren:

   pip install numpy>=1.21.0 aiofiles>=23.0.0

3. Home Assistant neu starten:

   systemctl restart home-assistant@homeassistant


PRÃœFUNG:
--------
Nach Installation prÃ¼fen:

python3 -c "import numpy, aiofiles; print('Ã¢Å“â€œ OK')"

Dann Home Assistant neu starten.
"""


# Global Checker Instance (Singleton)
_global_checker: Optional[DependencyChecker] = None
_checker_lock = asyncio.Lock()


def get_dependency_checker() -> DependencyChecker:
    """
    Gibt globale Dependency Checker Instanz zurÃ¼ck (Singleton).
    
    Ã¢Å“â€œ SYNC: FÃ¼r KompatibilitÃ¤t mit bestehendem Code
    
    Returns:
        DependencyChecker Instanz
    # von Zara
    """
    global _global_checker
    if _global_checker is None:
        _global_checker = DependencyChecker()
    return _global_checker


async def get_dependency_checker_async() -> DependencyChecker:
    """
    Gibt globale Dependency Checker Instanz zurÃ¼ck (Singleton, async-safe).
    
    Ã¢Å“â€œ ASYNC: Thread-safe mit Lock
    
    Returns:
        DependencyChecker Instanz
    # von Zara
    """
    global _global_checker
    async with _checker_lock:
        if _global_checker is None:
            _global_checker = DependencyChecker()
        return _global_checker
