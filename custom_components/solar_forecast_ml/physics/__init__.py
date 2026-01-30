# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

# PyArmor Runtime - MUST be imported before any obfuscated modules
import sys
from pathlib import Path as _Path
_runtime_path = str(_Path(__file__).parent.parent)
if _runtime_path not in sys.path:
    sys.path.insert(0, _runtime_path)
try:
    import pyarmor_runtime_009810  # noqa: F401
except ImportError:
    pass  # Runtime not present (development mode)

from .physics_engine import (
    PhysicsEngine,
    SunPosition,
    PanelGeometry,
    IrradianceData,
    POAResult,
    PowerResult,
)
from .panel_group_calculator import (
    PanelGroup,
    PanelGroupResult,
    MultiGroupResult,
    PanelGroupCalculator,
)
from .physics_calibrator import (
    PhysicsCalibrator,
    CalibrationResult,
    GroupCalibrationFactors,
)

__all__ = [
    "PhysicsEngine",
    "SunPosition",
    "PanelGeometry",
    "IrradianceData",
    "POAResult",
    "PowerResult",
    "PanelGroup",
    "PanelGroupResult",
    "MultiGroupResult",
    "PanelGroupCalculator",
    "PhysicsCalibrator",
    "CalibrationResult",
    "GroupCalibrationFactors",
]
