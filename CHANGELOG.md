


# CHANGELOG - Solar Forecast ML Integration

## v6.8.0 - Complete Architecture Overhaul

### 🏗️ STRUCTURE REFACTORING
- **Modular directory structure** implemented (7 main folders)
- All modules prefixed (ml_*, forecast_*, service_*, data_*)
- Consistent naming conventions enforced
- Redundant Manager classes renamed

### 🔧 CRITICAL TIMEZONE FIXES
**Problem:** Mixed UTC/LOCAL timestamps led to incorrect forecasts
- `io.py`: DateTimeEncoder enforces LOCAL timezone
- `tracker.py`: All internal timestamps switched to LOCAL
- `data_manager.py`: parse_datetime() enforces LOCAL for all read operations
- `forecast_weather.py`: cached_at uses LOCAL instead of UTC
- `coordinator.py`: All time calculations switched to LOCAL
- Migration script for existing UTC data created

### ⚡ SENSOR FIXES
**Error 1-2:** YieldSensorStateSensor & PowerSensorStateSensor
- `_attr_native_unit_of_measurement` correctly set (kWh/W)
- `native_value()` returns float instead of string
- Device Class correctly assigned

**Error 3:** ExpectedDailyProductionSensor
- Uses `self.data["forecast_today"]` instead of non-existent `get_forecast()`
- Explicit None value on error
- Correct 6 AM update

**Error 4:** ProductionTimeSensor
- `start_tracking()` called after initialization
- Flag `_production_tracking_started` prevents duplicate calls
- Shows "0h 0m" instead of "Initializing..."

### 🤖 ML PIPELINE IMPROVEMENTS
**Clipping & Peak Power:**
- peak_power correctly set during initialization
- `set_entities()` called at Coordinator start
- MLModelStrategy receives peak_power from Predictor
- Physical clipping to installed capacity

**Model Persistence:**
- `model_state.json` now loaded (previously only written)
- Performance metrics preserved across restarts
- Peak Power restoration from State file
- Fallback data if learned_weights is missing

**Feature Engineering:**
- New cloudiness features: `cloudiness_primary`, `cloud_impact`, `sunshine_factor`
- Non-linear Cloud Penalty for better cloud handling
- Zero-variance features skipped instead of std=1.0

**Regularization:**
- Adaptive Ridge alpha based on sample count
- Fewer samples = more regularization
- Prevents overfitting with limited data

**Next Hour Prediction:**
- Physical clipping to max_hourly_kwh
- Lux sensor check for darkness (< 500 lux)
- Better night detection

### 📊 DATA MANAGER EXTENSIONS
**New Methods:**
- `get_learned_weights()`: Loads ML weights
- `get_hourly_profile()`: Loads hourly profiles
- `add_prediction_record()`: Stores predictions
- `get_prediction_history()`: Loads history
- `get_average_monthly_yield()`: Calculates monthly average
- `add_hourly_sample()`: Stores hourly samples
- `get_hourly_samples(days)`: Loads samples
- `get_last_collected_hour()`: Finds last collection
- `get_all_training_records(days)`: Loads training data
- `cleanup_duplicate_samples()`: Removes duplicates
- `cleanup_zero_production_samples()`: Removes zero values

**Atomic Writes:**
- `_atomic_write_json()` for all write operations
- Proper error handling for file operations
- Race condition prevention via file lock

### 📝 COORDINATOR IMPROVEMENTS
**Service Manager Accessor:**
- 6 erroneous `service_manager.ml_predictor` replaced with `self.coordinator.ml_predictor`
- NotificationService correctly retrieved from `hass.data[DOMAIN]`
- Direct access without Service Manager wrapper

**Prediction Records:**
- Complete prediction_record structure with all required fields
- Includes `actual_value`, `weather_data`, `sensor_data`, `accuracy`, `model_version`
- Correct timestamps in LOCAL timezone

**Error Handling:**
- Graceful degradation for missing components
- Better logging for debugging
- Recovery mechanisms for partial failures

### 🧹 LOG OPTIMIZATION
- DEBUG spam removed from `weather_calculator.py` (Combined factors)
- Repeated state change logs reduced
- Only relevant information logged

### 🔄 FORECAST OPTIMIZATION
**Orchestrator:**
- Correct blending of ML and rule-based strategies
- Accuracy-based weighting
- Next-hour prediction with physical limits

**Weather Service:**
- Non-blocking initialization
- Dummy forecast if cache is missing
- Background update for live data
- Event listener for Weather entity changes

**Rule-Based Strategy:**
- Iterative calculation for today + tomorrow
- Correction factor integration
- Fallback on ML failure

### 🐛 BUG FIXES
**AttributeError Fixes:**
- `history.py`: `get_historical_average()` stub added
- `ml_predictor.py`: `get_today_prediction()` and `get_tomorrow_prediction()` stubs
- `diagnostic.py`: Correct accessor for ml_predictor
- `button.py`: Correct access to Coordinator components

**Syntax Errors:**
- `service_notification.py`: Unterminated f-string closed (lines 271-292)
- Installation instructions completed
- Factory function correctly implemented

**Config Flow:**
- CONF_SOLAR_CAPACITY correctly transferred to ConfigEntry
- peak_power passed to ML system works
- All user inputs persistently stored

### 📦 MIGRATION v6.4.0
- Architecture change requires one-time reset
- All ML data reset during migration
- Default files with correct structure created
- `versinfo.json` tracks migration status
- Old files cleanly deleted

### 📁 FILE STRUCTURE
```
custom_components/solar_forecast_ml/
├── core/              (4 files)
│   ├── dependency_handler.py
│   ├── exceptions.py
│   └── helpers.py
├── data/              (4 files)
│   ├── data_manager.py
│   ├── data_adapter.py
│   └── io.py
├── services/          (3 files)
│   ├── service_notification.py
│   └── service_error_handler.py
├── forecast/          (7 files)
│   ├── orchestrator.py
│   ├── forecast_strategy.py
│   ├── forecast_rule_based_strategy.py
│   ├── forecast_weather.py
│   ├── strategy.py
│   └── weather_calculator.py
├── production/        (4 files)
│   ├── history.py
│   ├── tracker.py
│   └── scheduled_tasks.py
├── sensors/           (5 files)
│   ├── base.py
│   ├── diagnostic.py
│   ├── states.py
│   └── data_collector.py
└── ml/                (9 files)
    ├── ml_predictor.py
    ├── ml_trainer.py
    ├── ml_sample_collector.py
    ├── ml_feature_engineering.py
    ├── ml_scaler.py
    ├── ml_prediction_strategies.py
    ├── ml_rule_based_strategy.py
    ├── ml_external_helpers.py
    └── ml_types.py
```

### 🔍 DIAGNOSTICS
**Recorder Check:**
- Check if entities are active in the recorder
- Warning for unrecorded entities
- ML training capability validated

**Dependency Check:**
- Asynchronous check of all Python packages
- Detailed logging for missing dependencies
- Version compatibility check

### 🌤️ WEATHER INTEGRATION
**Cloud Cover Problem:**
- Identified: All samples have cloud_cover=50.0 (fallback)
- Cause: Weather entity provides no real cloud data
- Recommendation: Use Open-Meteo integration
- Alternative: Improve condition mapping

**Forecast Cache:**
- Structure extended with `data_quality` information
- Backward compatibility for old list format
- cached_at now in LOCAL timezone
- 48h dummy forecast if cache missing

### 📅 SCHEDULED TASKS
**Morning Forecast (6 AM):**
- Expected Daily Production set
- Forecast update with fresh weather data
- Notification on activation

**Evening Verification (9 PM):**
- Comparison of forecast vs. actual value
- Accuracy calculation
- Correction factor update (fallback)
- Notification on successful verification

**Night Cleanup (2 AM):**
- Remove duplicates
- Clean zero-production samples
- Delete old backups

### 🔐 SERVICES
- `force_retrain`: Manual ML training
- `reset_model`: Model reset (WARNING: Deletes weights)

### 📊 SENSORS
**Core Sensors:**
- Expected Daily Production
- Production Time Today
- Today Forecast Remaining
- Tomorrow Forecast
- Best Hour for Consumption
- Average Monthly Yield
- Self-Sufficiency (Autarky)

**State Sensors:**
- Power Sensor State
- Yield Sensor State
- Humidity Sensor State
- Additional optional sensor states

**Diagnostic Sensors:**
- Model Accuracy
- Training Status
- Weather Service Status
- Dependency Status
- Performance Metrics

### ⚙️ CONFIGURATION
**Required Fields:**
- Weather Entity
- Power Entity (Current Power)
- Solar Yield Today Entity
- Solar Capacity (kWp)

**Optional Fields:**
- Humidity Sensor
- Temperature Sensor
- Wind Sensor
- Rain Sensor
- UV Sensor
- Lux Sensor
- Total Consumption Today

**Options:**
- Update Interval (30min default)
- Learning Enabled
- Hourly Forecast Enabled
- Notification Settings

### 🔧 PERFORMANCE
- Adaptive Ridge regularization
- Efficient caching mechanisms
- Non-blocking async operations
- Optimized database queries
- Atomic file operations

### 📖 KNOWN LIMITATIONS
- Cloud cover quality dependent on Weather entity
- ML training requires min. 30 days of data
- Accuracy increases with more training samples
- Production time tracking only with Power entity
- Timezone must be correctly configured

### 🎯 NEXT STEPS
- Weather entity with cloud cover support recommended
- Use import script for historical data
- Collect 30+ days for optimal ML training
- Enable regular backups


# 🌅 Version 6.0.0 "Risa" - Major ML Enhancement Release

This major release introduces significant improvements to the machine learning capabilities and overall system architecture of Solar Forecast ML.

## 🎯 Major Features

### Machine Learning Enhancements
- **Enhanced ML Training Pipeline**: Improved training algorithms for better forecast accuracy
- **Advanced Error Handling**: Robust error detection and recovery mechanisms throughout the ML pipeline
- **Optimized Forecasting Engine**: Refined prediction models with better performance and reliability

### System Improvements
- **Improved Data Processing**: More efficient handling of solar production and weather data
- **Better Integration Stability**: Enhanced Home Assistant integration with improved error handling
- **Performance Optimizations**: Reduced memory footprint and faster processing times

## 🔧 Technical Improvements
- Updated ML model architecture for better long-term predictions
- Enhanced data validation and preprocessing
- Improved logging and debugging capabilities
- Better handling of edge cases and missing data

## 📦 Dependencies
- aiofiles >= 23.0.0
- numpy >= 1.21.0

## 🏠 Home Assistant Compatibility
- Minimum Home Assistant version: 2024.1.0
- Fully compatible with HACS

## 📝 Breaking Changes
Please review the [CHANGELOG.md](CHANGELOG.md) for detailed information about any breaking changes.

---

**Full Changelog**: https://github.com/Zara-Toorox/ha-solar-forecast-ml/compare/v5.0.0...v6.0.0