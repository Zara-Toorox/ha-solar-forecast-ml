"""Physics Module for Solar Forecast ML Integration V12.2.0 @zara

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""

from .physics_engine import (
    PhysicsEngine,
    SunPosition,
    PanelGeometry,
    IrradianceData,
    POAResult,
    PowerResult,
)
from .geometry_learner import (
    GeometryLearner,
    GeometryEstimate,
    ClearSkyDataPoint,
    PanelGroupEfficiencyLearner,
    PanelGroupEfficiency,
)
from .panel_group_calculator import (
    PanelGroup,
    PanelGroupResult,
    MultiGroupResult,
    PanelGroupCalculator,
)

__all__ = [
    "PhysicsEngine",
    "SunPosition",
    "PanelGeometry",
    "IrradianceData",
    "POAResult",
    "PowerResult",
    "GeometryLearner",
    "GeometryEstimate",
    "ClearSkyDataPoint",
    "PanelGroupEfficiencyLearner",
    "PanelGroupEfficiency",
    "PanelGroup",
    "PanelGroupResult",
    "MultiGroupResult",
    "PanelGroupCalculator",
]
