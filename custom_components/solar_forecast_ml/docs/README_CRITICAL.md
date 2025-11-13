# CRITICAL COMPONENTS - DO NOT MODIFY WITHOUT PERMISSION

## Overview

This document lists all **CRITICAL** components of the Solar Forecast ML integration that required **EXTENSIVE** debugging and testing by **Zara (@Zara-Toorox)** to get working correctly.

**WARNING:** These components are protected and should **NOT** be modified without:
1. Explicit permission from Zara
2. Full understanding of the implementation
3. Comprehensive testing with production data
4. Creating a full backup before changes

---

## CRITICAL LEVEL 1 - EXTREME RISK

### Machine Learning Core
These files contain complex mathematical operations and were debugged extensively.

| File | Why Critical | Lines of Code |
|------|--------------|---------------|
| `ml/ml_trainer.py` | Ridge Regression training with adaptive alpha, bias handling, cross-validation | ~400 |
| `ml/ml_sample_collector.py` | Recorder database integration, timing-sensitive sample collection | ~500 |
| `ml/ml_inference.py` | Model predictions, feature scaling coordination | ~300 |
| `ml/ml_scaler.py` | Feature normalization (CRITICAL for model accuracy) | ~200 |

**Known Issues to Preserve:**
- Adaptive alpha formula: `max(0.01, min(1.0, 100.0 / len(y_train)))`
- Minimum 5 samples required for training
- Bias term must be added as column of ones
- Sample collection must handle recorder startup delays

### Data Persistence & Backup
**DATA LOSS RISK** - Bugs here can cause permanent data loss!

| File | Why Critical |
|------|--------------|
| `data/data_persistence.py` | ALL backup/restore operations, tar.gz handling |
| `data/data_io.py` | File I/O operations, JSON persistence |
| `ml/ml_sample_storage.py` | Training sample storage and retrieval |

**Critical Behavior:**
- Manual backups are NEVER auto-deleted
- Automatic backups follow retention policy
- Restore uses temp directory before moving files

---

## CRITICAL LEVEL 2 - HIGH RISK

### System Orchestration

| File | Why Critical |
|------|--------------|
| `coordinator.py` | Main coordinator - orchestrates ENTIRE integration |
| `core/core_coordinator_update_helpers.py` | Update logic for all components |
| `core/core_coordinator_init_helpers.py` | Initialization sequence and dependency management |

**What This Affects:**
- All sensors and their updates
- All services (train_model, generate_chart, etc.)
- ML training scheduling
- Forecast updates
- Production tracking

### Production Tracking

| File | Why Critical |
|------|--------------|
| `production/production_tracker.py` | Production start/end detection, timing calculations |
| `production/production_history.py` | Historical data aggregation and statistics |
| `production/production_scheduled_tasks.py` | Daily task scheduling (midnight, 6am, etc.) |

**Complex Logic:**
- Dynamic production start/end detection
- Peak power tracking
- Daily statistics calculation
- Autarky calculations

---

## CRITICAL LEVEL 3 - MODERATE RISK

### Forecasting Engine

| File | Why Critical |
|------|--------------|
| `forecast/forecast_orchestrator.py` | Orchestrates weather-based forecasting |
| `forecast/forecast_weather_calculator.py` | Weather condition mapping, solar calculations |
| `forecast/forecast_weather_api_client.py` | External API integration |

### Battery Management

| File | Why Critical |
|------|--------------|
| `battery/battery_coordinator.py` | Battery state coordination |
| `battery/battery_charge_tracker.py` | Charge/discharge tracking |

---

## General Rules for ALL Files

### Before Modifying ANY Code:

1. **Create a backup:**
   ```bash
   # Use the manual backup service in Home Assistant
   # Or manually backup: custom_components/solar_forecast_ml/
   ```

2. **Read the warnings:**
   - Check file header for specific warnings
   - Understand what can break
   - Know the known issues to preserve

3. **Test thoroughly:**
   - Test during HA startup
   - Test with unavailable sensors
   - Test with real production data
   - Verify no regressions in accuracy

4. **Document changes:**
   - Update this file if critical behavior changes
   - Add comments explaining why changes were needed
   - Update the "Last modified" date in file headers

---

## Protection Mechanisms

### 1. CODEOWNERS File
See `.github/CODEOWNERS` - All critical files require review by @Zara-Toorox

### 2. File Header Warnings
Each critical file has a prominent warning box at the top explaining:
- Why it's critical
- What can break
- Testing requirements
- Known issues to preserve

### 3. This Documentation
This file serves as the master list of all protected components.

---

## Contact

**If you need to modify any critical component:**
- Contact: @Zara-Toorox
- GitHub: https://github.com/Zara-Toorox/ha-solar-forecast-ml/issues

**If you break something:**
- Don't panic!
- Check backups in `solar_forecast_ml/backups/manual/`
- Restore from backup using the integration's restore service
- Report the issue with full error logs

---

## Maintenance History

| Date | Component | Change | By |
|------|-----------|--------|-----|
| 2025-11 | ml_trainer.py | Adaptive alpha calibration | Zara |
| 2025-11 | ml_sample_collector.py | Recorder startup handling | Zara |
| 2025-11 | data_persistence.py | Backup/restore implementation | Zara |
| 2025-11 | coordinator.py | Full system orchestration | Zara |

---

## Testing Checklist

When modifying critical components, ensure ALL of these pass:

- [ ] Integration loads during Home Assistant startup
- [ ] All sensors become available within 5 minutes
- [ ] Sample collection works hourly
- [ ] ML training completes without errors
- [ ] Predictions are generated successfully
- [ ] Backup creation works
- [ ] Backup restoration works
- [ ] Production tracking detects start/end correctly
- [ ] All services respond (train_model, generate_chart, etc.)
- [ ] No errors in Home Assistant logs
- [ ] Model accuracy doesn't degrade (check sensor.solar_forecast_ml_model_accuracy)

---

**Remember:** These components represent months of debugging and testing.
**Treat them with respect!**

*"With great power comes great responsibility... and a lot of coffee"* - Zara

---

**Last Updated:** November 2025
**Maintained By:** Zara (@Zara-Toorox)
**Version:** 8.4.1
