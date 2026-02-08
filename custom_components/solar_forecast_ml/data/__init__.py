# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Data module exports. @zara"""

from .db_manager import DatabaseManager
from .weather_types import CloudType, BlendedHourForecast, ExpertForecast
from .data_io import DataManagerIO
from .data_persistence import DataPersistence
from .data_adapter import TypedDataAdapter
from .data_state_handler import DataStateHandler
from .data_backup_handler import DataBackupHandler
from .data_startup_initializer import StartupInitializer
from .data_manager import DataManager

__all__ = [
    # Database @zara
    "DatabaseManager",

    # Weather Types @zara
    "CloudType",
    "BlendedHourForecast",
    "ExpertForecast",

    # Core Data Handling @zara
    "DataManagerIO",
    "DataPersistence",
    "TypedDataAdapter",
    "DataStateHandler",
    "DataBackupHandler",
    "StartupInitializer",

    # Main Data Manager @zara
    "DataManager",
]
