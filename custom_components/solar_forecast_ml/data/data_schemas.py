# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import copy
from datetime import date, datetime
from typing import Any, Dict

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_schema(schema_name: str) -> Dict[str, Any]:
    """Get a deep copy of a schema by name.

    Always returns a copy to prevent mutation of the original schema.

    Args:
        schema_name: Name of the schema (e.g., "weather_precision_daily")

    Returns:
        Deep copy of the schema dict

    Raises:
        KeyError: If schema_name is not found
    """
    if schema_name not in SCHEMAS:
        raise KeyError(f"Unknown schema: {schema_name}. Available: {list(SCHEMAS.keys())}")
    return copy.deepcopy(SCHEMAS[schema_name])


# =============================================================================
# ROOT FILES
# =============================================================================

MIGRATIONS_COMPLETED_SCHEMA = {
    "completed_migrations": [],
    "last_run": None,
    "last_version": "1.0",
}

# =============================================================================
# AI FILES
# =============================================================================

DNI_TRACKER_SCHEMA = {
    "version": "1.0",
    "last_updated": None,
    "max_dni": {str(h): 0.0 for h in range(6, 21)},
    "history": {str(h): [] for h in range(6, 21)},
}

GRID_SEARCH_RESULTS_SCHEMA = {
    "success": False,
    "best_params": None,
    "best_accuracy": None,
    "all_results": [],
    "duration_seconds": None,
    "error_message": None,
    "hardware_info": None,
    "timestamp": None,
}

# Note: learned_weights.json structure is dynamic based on LSTM architecture.
# This schema represents the metadata fields - actual weight arrays are variable.
LEARNED_WEIGHTS_SCHEMA = {
    "version": "2.0",
    # LSTM gate weights (required) - arrays, actual sizes depend on architecture
    "Wf": [],  # Forget gate weights
    "Wi": [],  # Input gate weights
    "Wc": [],  # Cell gate weights
    "Wo": [],  # Output gate weights
    "bf": [],  # Forget gate bias
    "bi": [],  # Input gate bias
    "bc": [],  # Cell gate bias
    "bo": [],  # Output gate bias
    # Output layer weights (required)
    "Wy": [],  # Output weights
    "by": [],  # Output bias
    # Architecture info (required)
    "input_size": 17,
    "hidden_size": 32,
    "sequence_length": 24,
    "num_outputs": 1,
    # Training info (optional)
    "training_samples": 0,
    "accuracy": None,
    "rmse": None,
    "last_trained": None,
    # Attention mechanism (optional - only present when use_attention=True)
    "has_attention": False,
    # When has_attention=True, these fields are present:
    # "W_query": [],   # Query weights (hidden_size x hidden_size)
    # "W_key": [],     # Key weights (hidden_size x hidden_size)
    # "W_value": [],   # Value weights (hidden_size x hidden_size)
    # "W_attn_out": [],  # Attention output projection (hidden_size x hidden_size*2)
    # "b_attn_out": [],  # Attention output bias (hidden_size x 1)
}

SEASONAL_SCHEMA = {
    "version": "1.0",
    "factors": {
        "1": 0.85, "2": 0.9, "3": 0.95, "4": 1.0,
        "5": 1.05, "6": 1.1, "7": 1.1, "8": 1.05,
        "9": 1.0, "10": 0.95, "11": 0.9, "12": 0.85,
    },
    "sample_counts": {str(m): 0 for m in range(1, 13)},
}

# =============================================================================
# DATA FILES
# =============================================================================

BRIGHT_SKY_CACHE_SCHEMA = {
    "version": "1.0",
    "source": "bright_sky",
    "metadata": {
        "last_updated": None,
        "latitude": None,
        "longitude": None,
        "retention_days": 730,
    },
    "forecast": {},
}

COORDINATOR_STATE_SCHEMA = {
    "version": "1.0",
    "expected_daily_production": None,
    "last_set_date": None,
    "last_updated": None,
}

OPEN_METEO_CACHE_SCHEMA = {
    "version": "2.0",
    "metadata": {
        "fetched_at": None,
        "latitude": None,
        "longitude": None,
        "hours_cached": 0,
        "days_cached": 0,
        "mode": "direct_radiation",
    },
    "forecast": {},
}

PIRATE_WEATHER_CACHE_SCHEMA = {
    "version": "1.0",
    "source": "pirate_weather",
    "metadata": {
        "last_updated": None,
        "latitude": None,
        "longitude": None,
        "retention_days": 730,
    },
    "forecast": {},
}

PRODUCTION_TIME_STATE_SCHEMA = {
    "version": "1.0",
    "date": None,  # Will be set dynamically
    "accumulated_hours": 0.0,
    "is_active": False,
    "start_time": None,
    "last_updated": None,
    "production_time_today": "00:00:00",
}

WEATHER_EXPERT_WEIGHTS_SCHEMA = {
    "version": "1.1",  # V12.8: Added FOG support
    "weights": {
        "clear": {
            "open_meteo": 0.15,
            "wttr_in": 0.1,
            "ecmwf_layers": 0.15,
            "bright_sky": 0.35,
            "pirate_weather": 0.25,
        },
        "cirrus": {
            "open_meteo": 0.1,
            "wttr_in": 0.1,
            "ecmwf_layers": 0.3,
            "bright_sky": 0.3,
            "pirate_weather": 0.2,
        },
        "fair": {
            "open_meteo": 0.15,
            "wttr_in": 0.1,
            "ecmwf_layers": 0.15,
            "bright_sky": 0.35,
            "pirate_weather": 0.25,
        },
        "mixed": {
            "open_meteo": 0.15,
            "wttr_in": 0.1,
            "ecmwf_layers": 0.15,
            "bright_sky": 0.3,
            "pirate_weather": 0.3,
        },
        "stratus": {
            "open_meteo": 0.2,
            "wttr_in": 0.2,
            "ecmwf_layers": 0.35,
            "bright_sky": 0.25,
            "pirate_weather": 0.0,
        },
        "overcast": {
            "open_meteo": 0.1,
            "wttr_in": 0.1,
            "ecmwf_layers": 0.1,
            "bright_sky": 0.4,
            "pirate_weather": 0.3,
        },
        "snow": {
            "open_meteo": 0.2,
            "wttr_in": 0.2,
            "ecmwf_layers": 0.2,
            "bright_sky": 0.2,
            "pirate_weather": 0.2,
        },
        # V12.8: Fog detection - visibility-based cloud types
        "fog": {
            "open_meteo": 0.1,
            "wttr_in": 0.05,
            "ecmwf_layers": 0.05,
            "bright_sky": 0.5,  # DWD best for fog in Germany
            "pirate_weather": 0.3,
        },
        "fog_light": {
            "open_meteo": 0.15,
            "wttr_in": 0.1,
            "ecmwf_layers": 0.1,
            "bright_sky": 0.4,
            "pirate_weather": 0.25,
        },
    },
    "metadata": {
        "last_updated": None,
        "experts": [
            "open_meteo",
            "wttr_in",
            "ecmwf_layers",
            "bright_sky",
            "pirate_weather",
        ],
        "cloud_types": [
            "clear",
            "cirrus",
            "fair",
            "mixed",
            "stratus",
            "overcast",
            "snow",
            "fog",        # V12.8: Dense fog (visibility < 1km)
            "fog_light",  # V12.8: Light fog (visibility 1-5km)
        ],
    },
    "snow_prediction_stats": {
        "total_predictions": 0,
        "correct_predictions": 0,
        "accuracy": 0.0,
        "last_updated": None,
    },
}

WEATHER_SOURCE_WEIGHTS_SCHEMA = {
    "version": "1.1",
    "weights": {
        "open_meteo": 0.5,
        "wwo": 0.5,
    },
    "learning_metadata": {
        "last_updated": None,
        "last_learning_date": None,
        "last_mae": {},
        "comparison_hours": 0,
        "smoothing_factor_used": 0.3,
        "smoothing_factor_default": 0.3,
        "accelerated_learning": False,
    },
}

WTTR_IN_CACHE_SCHEMA = {
    "version": "1.0",
    "metadata": {
        "fetched_at": None,
        "source": "wttr.in",
        "latitude": None,
        "longitude": None,
        "cache_max_age_hours": 6.0,
        "retention_days": 730,
    },
    "forecast": {},
}

# =============================================================================
# PHYSICS FILES
# =============================================================================

CALIBRATION_HISTORY_SCHEMA = {
    "version": "1.0",
    "updated_at": None,
    "history": [],
}

LEARNING_CONFIG_SCHEMA = {
    "version": "2.1",  # V12.8: Added FOG buckets
    "updated_at": None,
    "physics_defaults": {
        "albedo": 0.2,
        "system_efficiency": 0.9,
        "learned_efficiency_factor": 1.0,
    },
    "group_calibration": {},
    "metadata": {
        "rolling_window_days": 21,
        "min_samples": 1,
        "total_groups": 0,
        "total_bucket_factors": 0,
        "weather_buckets": [
            "clear",
            "clear_low_sun",
            "fair",
            "fair_low_sun",
            "cloudy",
            "cloudy_low_sun",
            "overcast",
            "rainy",
            "rainy_low_sun",
            "snowy",
            "fog",          # V12.8: Dense fog (visibility < 1km)
            "fog_low_sun",  # V12.8: Fog + low sun angle
        ],
    },
}

# =============================================================================
# STATS FILES
# =============================================================================

# Helper for astronomy_cache hourly_max_peaks
def _create_hourly_max_peaks() -> Dict[str, Dict]:
    """Create hourly_max_peaks structure for all 24 hours."""
    return {
        str(h): {
            "kwh": 0.0,
            "date": None,
            "conditions": {},
        }
        for h in range(24)
    }


ASTRONOMY_CACHE_SCHEMA = {
    "version": "1.0",
    "last_updated": None,
    "location": {
        "latitude": None,
        "longitude": None,
        "elevation_m": 0,
        "timezone": None,
    },
    "pv_system": {
        "installed_capacity_kwp": None,
        "max_peak_record_kwh": 0.0,
        "max_peak_date": None,
        "max_peak_hour": None,
        "max_peak_conditions": {
            "sun_elevation_deg": None,
            "cloud_cover_percent": None,
            "temperature_c": None,
            "solar_radiation_wm2": None,
        },
        "hourly_max_peaks": _create_hourly_max_peaks(),
    },
    "cache_info": {
        "total_days": 0,
        "days_back": 0,
        "days_ahead": 0,
        "date_range_start": None,
        "date_range_end": None,
        "success_count": 0,
        "error_count": 0,
    },
    "days": {},
}

DAILY_FORECASTS_SCHEMA = {
    "version": "3.0.0",
    "today": {
        "date": None,  # Will be set dynamically
        "forecast_day": {
            "prediction_kwh": None,
            "prediction_kwh_raw": None,
            "safeguard_applied": False,
            "safeguard_reduction_kwh": 0.0,
            "locked": False,
            "locked_at": None,
            "source": None,
        },
        "forecast_tomorrow": {
            "date": None,
            "prediction_kwh": None,
            "locked": False,
            "locked_at": None,
            "source": None,
            "updates": [],
        },
        "forecast_day_after_tomorrow": {
            "date": None,
            "prediction_kwh": None,
            "locked": False,
            "next_update": None,
            "source": None,
            "updates": [],
        },
        "forecast_best_hour": {
            "hour": None,
            "prediction_kwh": None,
            "locked": False,
            "locked_at": None,
            "source": None,
        },
        "actual_best_hour": {
            "hour": None,
            "actual_kwh": None,
            "saved_at": None,
        },
        "forecast_next_hour": {
            "period": None,
            "prediction_kwh": None,
            "updated_at": None,
            "source": None,
        },
        "production_time": {
            "active": False,
            "duration_seconds": 0,
            "start_time": None,
            "end_time": None,
            "last_power_above_10w": None,
            "zero_power_since": None,
        },
        "peak_today": {
            "power_w": 0.0,
            "at": None,
        },
        "yield_today": {
            "kwh": None,
            "sensor": None,
        },
        "consumption_today": {
            "kwh": None,
            "sensor": None,
        },
        "autarky": {
            "percent": None,
            "calculated_at": None,
        },
        "finalized": {
            "yield_kwh": None,
            "consumption_kwh": None,
            "production_hours": None,
            "accuracy_percent": None,
            "at": None,
        },
    },
    "statistics": {
        "all_time_peak": {
            "power_w": 0.0,
            "date": None,
            "at": None,
        },
        "current_week": {
            "period": None,
            "date_range": None,
            "yield_kwh": 0.0,
            "consumption_kwh": 0.0,
            "days": 0,
            "updated_at": None,
        },
        "current_month": {
            "period": None,
            "yield_kwh": 0.0,
            "consumption_kwh": 0.0,
            "avg_autarky": 0.0,
            "days": 0,
            "updated_at": None,
        },
        "last_7_days": {
            "avg_yield_kwh": 0.0,
            "avg_accuracy": 0.0,
            "total_yield_kwh": 0.0,
            "calculated_at": None,
        },
        "last_30_days": {
            "avg_yield_kwh": 0.0,
            "avg_accuracy": 0.0,
            "total_yield_kwh": 0.0,
            "calculated_at": None,
        },
        "last_365_days": {
            "avg_yield_kwh": 0.0,
            "total_yield_kwh": 0.0,
            "calculated_at": None,
        },
    },
    "history": [],
    "metadata": {
        "retention_days": 730,
        "history_entries": 0,
        "last_update": None,
    },
}

DAILY_SUMMARIES_SCHEMA = {
    "version": "2.0",
    "last_updated": None,
    "summaries": [],
}

FORECAST_DRIFT_LOG_SCHEMA = {
    "entries": [],
    "version": "1.0",
}

HOURLY_PREDICTIONS_SCHEMA = {
    "version": "12.4.0",
    "last_updated": None,
    "best_hour_today": None,
    "predictions": [],
}

HOURLY_WEATHER_ACTUAL_SCHEMA = {
    "version": "1.2",
    "metadata": {
        "created_at": None,
        "last_updated": None,
        "frost_detection_version": "correlation_enhanced",
    },
    "hourly_data": {},
    "snow_tracking": {
        "last_snow_event": None,
        "estimated_depth_mm": 0.0,
        "panels_covered_since": None,
        "melt_started_at": None,
        "above_threshold_since": None,
    },
}

MULTI_DAY_HOURLY_FORECAST_SCHEMA = {
    "version": "1.0",
    "updated_at": None,
    "days": {},
}

PANEL_GROUP_SENSOR_STATE_SCHEMA = {
    "last_updated": None,
    "last_values": {},
}

PANEL_GROUP_TODAY_CACHE_SCHEMA = {
    "date": None,  # Will be set dynamically
    "last_updated": None,
    "available": False,
    "groups": {},
}

RETROSPECTIVE_FORECAST_SCHEMA = {
    "version": "1.0",
    "generated_at": None,
    "simulation_context": {
        "simulated_forecast_time": None,
        "sunrise_today": None,
        "target_date": None,
        "code_version": None,
        "purpose": None,
    },
    "forecast_summary": {
        "today_kwh": None,
        "today_kwh_raw": None,
        "safeguard_applied": False,
        "tomorrow_kwh": None,
        "day_after_tomorrow_kwh": None,
        "method": None,
        "confidence": None,
        "model_accuracy": None,
        "best_hour": None,
        "best_hour_kwh": None,
    },
    "hourly_predictions": [],
}

WEATHER_EXPERT_LEARNING_SCHEMA = {
    "version": "1.0",
    "daily_history": {},
    "metadata": {
        "last_learning_run": None,
        "total_learning_days": 0,
    },
}

WEATHER_FORECAST_CORRECTED_SCHEMA = {
    "version": "4.3",
    "forecast": {},
    "metadata": {
        "created": None,
        "source": None,
        "mode": "direct_radiation",
        "hours_forecast": 0,
        "days_forecast": 0,
    },
}

WEATHER_PRECISION_DAILY_SCHEMA = {
    "daily_tracking": {},
    "rolling_averages": {
        "sample_days": 0,
        "sample_days_clear": 0,
        "sample_days_cloudy": 0,
        "correction_factors": {
            "temperature": 0.0,
            "solar_radiation_wm2": 1.0,
            "clouds": 1.0,
            "humidity": 1.0,
            "wind": 1.0,
            "rain": 1.0,
            "pressure": 0.0,
        },
        "correction_factors_clear": {
            "temperature": 0.0,
            "solar_radiation_wm2": 1.0,
            "clouds": 1.0,
            "humidity": 1.0,
            "wind": 1.0,
            "rain": 0.0,
            "pressure": 0.0,
        },
        "correction_factors_cloudy": {
            "temperature": 0.0,
            "solar_radiation_wm2": 1.0,
            "clouds": 1.0,
            "humidity": 1.0,
            "wind": 1.0,
            "rain": 0.0,
            "pressure": 0.0,
        },
        "confidence": {
            "temperature": 0.0,
            "solar_radiation_wm2": 0.0,
            "clouds": 0.0,
            "humidity": 0.0,
            "wind": 0.0,
            "rain": 0.0,
            "pressure": 0.0,
        },
        "hourly_factors": {
            "solar_radiation_wm2": {},
            "clouds": {},
        },
        "hourly_min_samples": 3,
        "updated_at": None,
    },
    "metadata": {
        "created": None,
        "last_updated": None,
        "total_days_tracked": 0,
        "sensors_configured": [],
        "sensors_optional": True,
    },
}

WEATHER_SOURCE_LEARNING_SCHEMA = {
    "version": "1.0",
    "daily_history": {},
}

YIELD_CACHE_SCHEMA = {
    "value": None,
    "time": None,
    "date": None,
}

# =============================================================================
# SCHEMA REGISTRY
# =============================================================================

SCHEMAS: Dict[str, Dict[str, Any]] = {
    # Root
    "migrations_completed": MIGRATIONS_COMPLETED_SCHEMA,
    # AI
    "dni_tracker": DNI_TRACKER_SCHEMA,
    "grid_search_results": GRID_SEARCH_RESULTS_SCHEMA,
    "learned_weights": LEARNED_WEIGHTS_SCHEMA,
    "seasonal": SEASONAL_SCHEMA,
    # Data
    "bright_sky_cache": BRIGHT_SKY_CACHE_SCHEMA,
    "coordinator_state": COORDINATOR_STATE_SCHEMA,
    "open_meteo_cache": OPEN_METEO_CACHE_SCHEMA,
    "pirate_weather_cache": PIRATE_WEATHER_CACHE_SCHEMA,
    "production_time_state": PRODUCTION_TIME_STATE_SCHEMA,
    "weather_expert_weights": WEATHER_EXPERT_WEIGHTS_SCHEMA,
    "weather_source_weights": WEATHER_SOURCE_WEIGHTS_SCHEMA,
    "wttr_in_cache": WTTR_IN_CACHE_SCHEMA,
    # Physics
    "calibration_history": CALIBRATION_HISTORY_SCHEMA,
    "learning_config": LEARNING_CONFIG_SCHEMA,
    # Stats
    "astronomy_cache": ASTRONOMY_CACHE_SCHEMA,
    "daily_forecasts": DAILY_FORECASTS_SCHEMA,
    "daily_summaries": DAILY_SUMMARIES_SCHEMA,
    "forecast_drift_log": FORECAST_DRIFT_LOG_SCHEMA,
    "hourly_predictions": HOURLY_PREDICTIONS_SCHEMA,
    "hourly_weather_actual": HOURLY_WEATHER_ACTUAL_SCHEMA,
    "multi_day_hourly_forecast": MULTI_DAY_HOURLY_FORECAST_SCHEMA,
    "panel_group_sensor_state": PANEL_GROUP_SENSOR_STATE_SCHEMA,
    "panel_group_today_cache": PANEL_GROUP_TODAY_CACHE_SCHEMA,
    "retrospective_forecast": RETROSPECTIVE_FORECAST_SCHEMA,
    "weather_expert_learning": WEATHER_EXPERT_LEARNING_SCHEMA,
    "weather_forecast_corrected": WEATHER_FORECAST_CORRECTED_SCHEMA,
    "weather_precision_daily": WEATHER_PRECISION_DAILY_SCHEMA,
    "weather_source_learning": WEATHER_SOURCE_LEARNING_SCHEMA,
    "yield_cache": YIELD_CACHE_SCHEMA,
}

# File path mappings (relative to data_dir)
SCHEMA_FILE_PATHS: Dict[str, str] = {
    "migrations_completed": ".migrations_completed.json",
    "dni_tracker": "ai/dni_tracker.json",
    "grid_search_results": "ai/grid_search_results.json",
    "learned_weights": "ai/learned_weights.json",
    "seasonal": "ai/seasonal.json",
    "bright_sky_cache": "data/bright_sky_cache.json",
    "coordinator_state": "data/coordinator_state.json",
    "open_meteo_cache": "data/open_meteo_cache.json",
    "pirate_weather_cache": "data/pirate_weather_cache.json",
    "production_time_state": "data/production_time_state.json",
    "weather_expert_weights": "data/weather_expert_weights.json",
    "weather_source_weights": "data/weather_source_weights.json",
    "wttr_in_cache": "data/wttr_in_cache.json",
    "calibration_history": "physics/calibration_history.json",
    "learning_config": "physics/learning_config.json",
    "astronomy_cache": "stats/astronomy_cache.json",
    "daily_forecasts": "stats/daily_forecasts.json",
    "daily_summaries": "stats/daily_summaries.json",
    "forecast_drift_log": "stats/forecast_drift_log.json",
    "hourly_predictions": "stats/hourly_predictions.json",
    "hourly_weather_actual": "stats/hourly_weather_actual.json",
    "multi_day_hourly_forecast": "stats/multi_day_hourly_forecast.json",
    "panel_group_sensor_state": "stats/panel_group_sensor_state.json",
    "panel_group_today_cache": "stats/panel_group_today_cache.json",
    "retrospective_forecast": "stats/retrospective_forecast.json",
    "weather_expert_learning": "stats/weather_expert_learning.json",
    "weather_forecast_corrected": "stats/weather_forecast_corrected.json",
    "weather_precision_daily": "stats/weather_precision_daily.json",
    "weather_source_learning": "stats/weather_source_learning.json",
    "yield_cache": "stats/yield_cache.json",
}
