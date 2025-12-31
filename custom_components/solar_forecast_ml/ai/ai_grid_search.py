# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import asyncio
import json
import logging
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

# Default parameter grid - conservative for Home Assistant
DEFAULT_PARAM_GRID = [
    {"hidden_size": 24, "batch_size": 16, "learning_rate": 0.005},
    {"hidden_size": 32, "batch_size": 16, "learning_rate": 0.005},  # Current default
    {"hidden_size": 32, "batch_size": 8, "learning_rate": 0.005},
    {"hidden_size": 48, "batch_size": 16, "learning_rate": 0.005},
]

# Minimal grid for constrained systems (if user forces it)
MINIMAL_PARAM_GRID = [
    {"hidden_size": 24, "batch_size": 16, "learning_rate": 0.005},
    {"hidden_size": 32, "batch_size": 16, "learning_rate": 0.005},
]

# Reduced epochs for grid search (faster iteration)
GRID_SEARCH_EPOCHS = 100
GRID_SEARCH_EARLY_STOPPING = 15


@dataclass
class HardwareInfo:
    """Hardware detection result @zara"""

    is_raspberry_pi: bool = False
    is_virtual_machine: bool = False
    is_container: bool = False
    cpu_count: int = 1
    architecture: str = "unknown"
    machine_type: str = "unknown"
    grid_search_allowed: bool = True
    reason: str = "Native hardware detected"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization @zara"""
        return {
            "is_raspberry_pi": self.is_raspberry_pi,
            "is_virtual_machine": self.is_virtual_machine,
            "is_container": self.is_container,
            "cpu_count": self.cpu_count,
            "architecture": self.architecture,
            "machine_type": self.machine_type,
            "grid_search_allowed": self.grid_search_allowed,
            "reason": self.reason,
        }


@dataclass
class GridSearchResult:
    """Result of grid search optimization @zara"""

    success: bool
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_accuracy: float = 0.0
    all_results: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    hardware_info: Optional[HardwareInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization @zara"""
        return {
            "success": self.success,
            "best_params": self.best_params,
            "best_accuracy": self.best_accuracy,
            "all_results": self.all_results,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "hardware_info": self.hardware_info.to_dict() if self.hardware_info else None,
            "timestamp": datetime.now().isoformat(),
        }


def detect_hardware() -> HardwareInfo:
    """Detect system hardware and determine if grid search is allowed @zara

    Checks for:
    - Raspberry Pi (ARM + /proc/cpuinfo markers)
    - Virtual Machines (DMI info, hypervisor flag)
    - Containers (Docker, Podman)

    Returns:
        HardwareInfo with detection results and grid_search_allowed flag
    """
    info = HardwareInfo(
        cpu_count=os.cpu_count() or 1,
        architecture=platform.machine(),
        machine_type=platform.system(),
    )

    # 1. Check for Raspberry Pi (ARM architecture + specific markers)
    if platform.machine().lower().startswith(("arm", "aarch")):
        try:
            cpuinfo_path = Path("/proc/cpuinfo")
            if cpuinfo_path.exists():
                cpuinfo = cpuinfo_path.read_text()
                pi_markers = ["Raspberry Pi", "BCM2", "BCM27", "BCM28"]
                if any(marker in cpuinfo for marker in pi_markers):
                    info.is_raspberry_pi = True
                    info.grid_search_allowed = False
                    info.reason = "Raspberry Pi detected - too slow for Grid-Search"
                    _LOGGER.info(f"Hardware detection: {info.reason}")
                    return info
        except Exception as e:
            _LOGGER.debug(f"Could not read /proc/cpuinfo: {e}")

    # 2. Check for Virtual Machine
    vm_detected = _detect_virtual_machine()
    if vm_detected:
        info.is_virtual_machine = True
        info.grid_search_allowed = False
        info.reason = f"Virtual Machine detected ({vm_detected}) - shared resources"
        _LOGGER.info(f"Hardware detection: {info.reason}")
        return info

    # 3. Check for Container (informational, doesn't block)
    if _detect_container():
        info.is_container = True
        # Container on powerful hardware is OK
        _LOGGER.debug("Container detected - Grid-Search still allowed if host is capable")

    # 4. Check minimum CPU count (very weak systems)
    if info.cpu_count < 2:
        info.grid_search_allowed = False
        info.reason = f"Only {info.cpu_count} CPU(s) detected - insufficient for Grid-Search"
        _LOGGER.info(f"Hardware detection: {info.reason}")
        return info

    # All checks passed
    info.grid_search_allowed = True
    info.reason = f"Capable hardware detected ({info.architecture}, {info.cpu_count} CPUs)"
    _LOGGER.info(f"Hardware detection: {info.reason}")

    return info


def _detect_virtual_machine() -> Optional[str]:
    """Detect if running in a virtual machine @zara

    Returns:
        VM type string if detected, None otherwise
    """
    # Check DMI information (Linux)
    dmi_paths = [
        ("/sys/class/dmi/id/product_name", ["VirtualBox", "VMware", "QEMU", "KVM", "Bochs"]),
        ("/sys/class/dmi/id/sys_vendor", ["QEMU", "VMware", "innotek", "Xen", "Microsoft Corporation"]),
        ("/sys/class/dmi/id/board_vendor", ["Oracle", "VMware"]),
    ]

    for path, keywords in dmi_paths:
        try:
            content = Path(path).read_text().strip()
            for keyword in keywords:
                if keyword.lower() in content.lower():
                    return keyword
        except Exception:
            continue

    # Check /proc/cpuinfo for hypervisor flag
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        if "hypervisor" in cpuinfo.lower():
            return "Hypervisor"
    except Exception:
        pass

    # Check systemd-detect-virt (if available)
    try:
        import subprocess
        result = subprocess.run(
            ["systemd-detect-virt", "--vm"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip() != "none":
            return result.stdout.strip()
    except Exception:
        pass

    return None


def _detect_container() -> bool:
    """Detect if running in a container @zara"""
    # Check for Docker
    if Path("/.dockerenv").exists():
        return True

    # Check cgroup for docker/lxc/podman
    try:
        cgroup = Path("/proc/1/cgroup").read_text()
        container_markers = ["docker", "lxc", "podman", "containerd"]
        if any(marker in cgroup.lower() for marker in container_markers):
            return True
    except Exception:
        pass

    return False


class GridSearchOptimizer:
    """Grid-Search optimizer for TinyLSTM hyperparameters @zara

    Automatically tests different parameter combinations and finds the best one.
    Only runs on capable hardware (excludes Raspberry Pi and VMs).
    """

    def __init__(
        self,
        data_dir: Path,
        param_grid: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize Grid-Search optimizer @zara

        Args:
            data_dir: Directory for saving results
            param_grid: Custom parameter grid (uses DEFAULT_PARAM_GRID if None)
        """
        self.data_dir = Path(data_dir)
        self.param_grid = param_grid or DEFAULT_PARAM_GRID
        self.hardware_info: Optional[HardwareInfo] = None
        self.last_result: Optional[GridSearchResult] = None
        self._results_file = self.data_dir / "grid_search_results.json"

    def check_hardware(self) -> HardwareInfo:
        """Check hardware and cache result @zara"""
        if self.hardware_info is None:
            self.hardware_info = detect_hardware()
        return self.hardware_info

    def is_available(self) -> bool:
        """Check if grid search is available on this hardware @zara"""
        return self.check_hardware().grid_search_allowed

    def get_status(self) -> Dict[str, Any]:
        """Get current grid search status @zara"""
        hw = self.check_hardware()

        status = {
            "available": hw.grid_search_allowed,
            "reason": hw.reason,
            "hardware": hw.to_dict(),
            "param_grid_size": len(self.param_grid),
            "last_run": None,
            "best_params": None,
        }

        # Load last result if exists
        if self._results_file.exists():
            try:
                with open(self._results_file, "r") as f:
                    last_result = json.load(f)
                    status["last_run"] = last_result.get("timestamp")
                    status["best_params"] = last_result.get("best_params")
                    status["best_accuracy"] = last_result.get("best_accuracy")
            except Exception:
                pass

        return status

    async def run_grid_search(
        self,
        lstm_class: type,
        X_sequences: List[Any],
        y_targets: List[Any],
        input_size: int,
        sequence_length: int = 24,
        num_outputs: int = 1,
        epochs: int = GRID_SEARCH_EPOCHS,
        validation_split: float = 0.2,
        progress_callback: Optional[callable] = None,
    ) -> GridSearchResult:
        """Run grid search over parameter combinations @zara

        Args:
            lstm_class: TinyLSTM class to instantiate
            X_sequences: Training sequences
            y_targets: Training targets
            input_size: Number of input features
            sequence_length: LSTM sequence length
            num_outputs: Number of output neurons
            epochs: Training epochs per combination
            validation_split: Validation data fraction
            progress_callback: Optional callback(current, total, params, accuracy)

        Returns:
            GridSearchResult with best parameters and all results
        """
        hw = self.check_hardware()

        if not hw.grid_search_allowed:
            return GridSearchResult(
                success=False,
                error_message=hw.reason,
                hardware_info=hw,
            )

        if len(X_sequences) < 50:
            return GridSearchResult(
                success=False,
                error_message=f"Need at least 50 samples, got {len(X_sequences)}",
                hardware_info=hw,
            )

        _LOGGER.info(
            f"Starting Grid-Search: {len(self.param_grid)} combinations, "
            f"{len(X_sequences)} samples, {epochs} epochs each"
        )

        start_time = datetime.now()
        all_results = []
        best_params = {}
        best_accuracy = -float('inf')

        for i, params in enumerate(self.param_grid):
            combo_start = datetime.now()

            _LOGGER.info(
                f"Grid-Search [{i+1}/{len(self.param_grid)}]: "
                f"hidden={params.get('hidden_size', 32)}, "
                f"batch={params.get('batch_size', 16)}, "
                f"lr={params.get('learning_rate', 0.005)}"
            )

            try:
                # Create LSTM with these parameters
                lstm = lstm_class(
                    input_size=input_size,
                    hidden_size=params.get("hidden_size", 32),
                    sequence_length=sequence_length,
                    num_outputs=num_outputs,
                    learning_rate=params.get("learning_rate", 0.005),
                )

                # Train
                result = await lstm.train(
                    X_sequences=X_sequences,
                    y_targets=y_targets,
                    epochs=epochs,
                    batch_size=params.get("batch_size", 16),
                    validation_split=validation_split,
                    early_stopping_patience=GRID_SEARCH_EARLY_STOPPING,
                )

                accuracy = result.get("accuracy", 0.0)
                combo_duration = (datetime.now() - combo_start).total_seconds()

                combo_result = {
                    "params": params,
                    "accuracy": accuracy,
                    "epochs_trained": result.get("epochs_trained", 0),
                    "final_val_loss": result.get("final_val_loss", 0),
                    "duration_seconds": combo_duration,
                }
                all_results.append(combo_result)

                _LOGGER.info(
                    f"Grid-Search [{i+1}/{len(self.param_grid)}]: "
                    f"R²={accuracy:.4f} in {combo_duration:.1f}s"
                )

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = params.copy()

                if progress_callback:
                    await progress_callback(i + 1, len(self.param_grid), params, accuracy)

                # Allow other tasks to run
                await asyncio.sleep(0.1)

            except Exception as e:
                _LOGGER.error(f"Grid-Search combination failed: {e}")
                all_results.append({
                    "params": params,
                    "error": str(e),
                })

        total_duration = (datetime.now() - start_time).total_seconds()

        result = GridSearchResult(
            success=True,
            best_params=best_params,
            best_accuracy=best_accuracy,
            all_results=all_results,
            duration_seconds=total_duration,
            hardware_info=hw,
        )

        self.last_result = result

        # Save results
        await self._save_results(result)

        _LOGGER.info(
            f"Grid-Search complete: best R²={best_accuracy:.4f} with "
            f"hidden={best_params.get('hidden_size')}, "
            f"batch={best_params.get('batch_size')} "
            f"in {total_duration:.1f}s total"
        )

        return result

    async def _save_results(self, result: GridSearchResult):
        """Save grid search results to file @zara"""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)

            def _write():
                with open(self._results_file, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _write)

            _LOGGER.debug(f"Grid-Search results saved to {self._results_file}")

        except Exception as e:
            _LOGGER.error(f"Failed to save Grid-Search results: {e}")

    def load_best_params(self) -> Optional[Dict[str, Any]]:
        """Load best parameters from previous grid search @zara

        Returns:
            Best parameters dict or None if no previous run
        """
        if not self._results_file.exists():
            return None

        try:
            with open(self._results_file, "r") as f:
                data = json.load(f)
                return data.get("best_params")
        except Exception as e:
            _LOGGER.debug(f"Could not load previous Grid-Search results: {e}")
            return None
