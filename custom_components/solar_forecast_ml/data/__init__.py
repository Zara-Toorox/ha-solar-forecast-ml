# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

# V12.4: Dual-path blending - combines cloud% (proven) with transmission (physics-based)
from .data_weather_expert_blender import (
    WeatherExpertBlender,
    WeatherExpertLearner,
    CloudType,
    classify_cloud_type,
    cloud_to_transmission,  # V12.4: Physics-based cloud→transmission conversion
    BrightSkyExpert,
    PirateWeatherExpert,
    TRANSMISSION_TAU_BY_CLOUD_TYPE,  # V12.4: Transmission coefficients
)
from .data_panel_group_sensor_reader import PanelGroupSensorReader
