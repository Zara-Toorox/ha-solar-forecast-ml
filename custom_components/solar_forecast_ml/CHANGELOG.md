# Changelog - Solar Forecast ML

All notable changes to this project will be documented in this file.

## [8.2.1] - 2025-01-08

### Fixed
- **Critical Fix**: Fixed `AttributeError` during daily midnight reset that prevented proper cleanup of expected daily production data from persistent storage
- **Improved**: Production tracker startup logging - first retry now shows as INFO instead of WARNING (normal during Home Assistant startup)

### Changed
- Enhanced compatibility with latest Home Assistant Core updates
- Improved error handling and logging during coordinator state management

### Technical Details
- Added missing `clear_expected_daily_production()` method to `DataStateHandler`
- Fixed daily reset process to properly clear `coordinator_state.json` at midnight
- Optimized startup sequence logging to reduce unnecessary warnings

---

## [8.2.0] - 2025-01-07

### Added
- Initial stable release with ML-based solar forecasting
- Hourly production tracking and predictions
- Weather-based forecast adjustments
- Historical data analysis and model training
- Comprehensive diagnostic sensors
- German localization support

### Features
- Real-time solar production forecasting
- Machine Learning model with automatic training
- Weather integration for enhanced accuracy
- Daily, weekly, and monthly statistics
- Autarky and self-sufficiency calculations
- Production time tracking
- Peak production hour detection

---

**Note**: This changelog follows [Keep a Changelog](https://keepachangelog.com/) format.
