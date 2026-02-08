# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Physics module exports @zara"""

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
    WeatherBucket,
    BucketFactors,
    GroupCalibrationFactors,
    CalibrationResult,
)

__all__ = [
    # Physics Engine
    "PhysicsEngine",
    "SunPosition",
    "PanelGeometry",
    "IrradianceData",
    "POAResult",
    "PowerResult",
    # Panel Group Calculator
    "PanelGroup",
    "PanelGroupResult",
    "MultiGroupResult",
    "PanelGroupCalculator",
    # Physics Calibrator
    "PhysicsCalibrator",
    "WeatherBucket",
    "BucketFactors",
    "GroupCalibrationFactors",
    "CalibrationResult",
]
