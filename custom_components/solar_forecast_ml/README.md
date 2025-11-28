# Solar Forecast ML V10 "LYRA" 1st HA full AI & ML Solar-Forecast

[![Version](https://img.shields.io/badge/version-10.0.0-blue.svg)](https://github.com/Zara-Toorox/ha-solar-forecast-ml)
[![HACS](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://hacs.xyz/)
[![License](https://img.shields.io/badge/license-AGPL--3.0-green.svg)](LICENSE)

<a href='https://ko-fi.com/Q5Q41NMZZY' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://ko-fi.com/img/githubbutton_sm.svg' border='0' alt='Buy Me a Coffee ' /></a>

**Intelligent Solar Forecasting for Home Assistant with Physics-First ML Architecture**

Imagine your smart home not just reacting, but *predicting* - using machine learning that adapts daily to your unique solar installation. That's exactly what Solar Forecast ML does. With **Version 10.0.0 "Lyra"** we introduce a revolutionary **Physics-First Architecture** that combines real solar physics with adaptive machine learning.

---

## What Makes This Integration Special?

Unlike simple weather API services, Solar Forecast ML learns from your real data **and** uses actual solar physics. This means:

- **Day 1:** Accurate forecasts using real physics (POA irradiance, temperature correction)
- **Week 1:** System learns your panel geometry (tilt/azimuth) automatically
- **Week 2+:** ML enhances physics predictions with local patterns (shadows, obstructions)
- **Month 1+:** 93-97% accuracy with full Physics+ML ensemble

The integration adapts to your specific roof orientation, local weather conditions, shading patterns, and seasonal characteristics - all without manual configuration.

---

## New in Version 10.0.0 "Lyra" AI

### Physics-First Architecture
The biggest update ever - a complete architectural redesign:

- **PhysicsEngine** - Real solar physics calculations (POA, temperature correction, cell efficiency)
- **GeometryLearner** - Automatically learns your panel tilt & azimuth using Levenberg-Marquardt optimization
- **AI Neural Network** - Pure NumPy LSTM AI that captures temporal patterns
- **Residual Learning** - ML learns the difference between physics and reality
- **Weighted Ensemble** - Confidence-based blending of physics + ML predictions

### Intelligent Detection Systems
- **Correlation-based Shadow Detection** - Distinguishes clouds from real obstructions
- **Physics-based Frost Detection** - Uses Magnus formula (dew point) for accurate frost warnings
- **Cloud Layer Physics** - Separate transmission models for low/mid/high clouds

### Bootstrap Services
- **Bootstrap from History** - Train models instantly using up to 6 months of HA history
- **No Cold Start** - Full accuracy from day 1 using historical data

---

## Core Features

### Forecasting
- **3-day yield forecasts** (today, tomorrow, day after) with Physics+ML optimization
- **Hourly predictions** for detailed planning
- **Physics-based calculations** using GHI, DNI, DHI from Open-Meteo
- **Automatic panel geometry learning** - no manual tilt/azimuth configuration needed
- **Temperature-corrected efficiency** - accounts for cell temperature effects

### Machine Learning supported by AI
- **Adaptive Algorithm Selection:**
  - < 100 samples: Ridge Regression (fast, stable)
  - 100+ samples: TinyLSTM Neural Network (+5-10% accuracy)
- **14 optimized features** per hour (time, weather, astronomy, lag)
- **24-hour sequence learning** for temporal pattern recognition
- **Zero external dependencies** - pure NumPy implementation

### Analytics & Detection (AI)
- **Shadow Detection** - Correlation-based analysis separates weather from obstructions
- **Frost Detection** - Physics-based dew point calculation with Magnus formula
- **Peak hour detection** - When do you produce the most?
- **Production time calculation** - Optimized sunrise to sunset tracking
- **Self-sufficiency calculation** - How independent are you really?

### Data Privacy
- **100% local processing** - Everything calculated on your system
- **100% local AI and ML** - No ChatGPT, CLAUDE, GEMINI, GROK,.. needed
- **No cloud services** - All data stays on your disk
- **No API keys required** - Uses free Open-Meteo API

---

## Sensors & Diagnostics

### Core Sensors
| Sensor | Description |
|--------|-------------|
| `sensor.solar_forecast_ml_today` | Today's forecast (kWh) |
| `sensor.solar_forecast_ml_tomorrow` | Tomorrow's forecast (kWh) |
| `sensor.solar_forecast_ml_day_after_tomorrow` | Day after tomorrow (kWh) |
| `sensor.solar_forecast_ml_next_hour` | Next hour prediction (kWh) |
| `sensor.solar_forecast_ml_production_time` | Production hours today |

### ML & Analytics Sensors
| Sensor | Description |
|--------|-------------|
| `sensor.solar_forecast_ml_model_state` | ML model training status |
| `sensor.solar_forecast_ml_model_accuracy` | Current prediction accuracy (%) |
| `sensor.solar_forecast_ml_training_samples` | Number of training samples |
| `sensor.solar_forecast_ml_shadow_current` | Current shadow detection |
| `sensor.solar_forecast_ml_performance_loss` | Shadow-related losses (%) |

### Weather & Trend Sensors
| Sensor | Description |
|--------|-------------|
| `sensor.solar_forecast_ml_cloudiness_trend_1h` | 1-hour cloud trend |
| `sensor.solar_forecast_ml_cloudiness_trend_3h` | 3-hour cloud trend |
| `sensor.solar_forecast_ml_weather_stability` | Weather volatility |

---

## Services

### ML Services
| Service | Description |
|---------|-------------|
| `force_retrain` | Retrain ML model with all available data |
| `reset_model` | Reset ML model to initial state |

### Bootstrap Services
| Service | Description |
|---------|-------------|
| `bootstrap_physics_from_history` | Train Physics+ML from HA history (up to 6 months) |
| `bootstrap_from_history` | Bootstrap pattern learning from history |

### Astronomy Services
| Service | Description |
|---------|-------------|
| `build_astronomy_cache` | Build comprehensive sun position cache |
| `refresh_cache_today` | Refresh astronomy data for current week |

### Testing & Maintenance
| Service | Description |
|---------|-------------|
| `test_morning_routine` | Analyze 6 AM predictions (read-only) |
| `run_all_day_end_tasks` | Manual trigger for 23:30 workflow |
| `run_weather_correction` | Rebuild corrected weather forecast |
| `send_daily_briefing` | Send formatted solar forecast notification |

---

## Installation

### HACS (Recommended)
1. Open HACS in Home Assistant
2. Go to "Integrations"
3. Click the three dots menu → "Custom repositories"
4. Add: `https://github.com/Zara-Toorox/ha-solar-forecast-ml`
5. Select category: "Integration"
6. Install "Solar Forecast ML"
7. Restart Home Assistant

### Manual Installation
1. Download the latest release
2. Copy `custom_components/solar_forecast_ml` to your `config/custom_components/`
3. Restart Home Assistant

### Configuration
1. Go to Settings → Devices & Services
2. Click "Add Integration"
3. Search for "Solar Forecast ML"
4. Configure:
   - **Power Sensor** (required): Current solar power in Watts
   - **Daily Yield Sensor** (required): Daily yield in kWh (must reset at midnight)
   - **System Capacity** (optional): Your system size in kWp
   - **Additional Sensors** (optional): Temperature, Lux, Radiation, etc.

---

## Quick Start After Installation

For immediate full accuracy, run the bootstrap service:

```yaml
service: solar_forecast_ml.bootstrap_physics_from_history
data:
  days: 180  # Uses up to 6 months of history
```

This will:
1. Fetch your production history from Home Assistant
2. Get historical weather from Open-Meteo Archive
3. Train the GeometryLearner (learns panel tilt/azimuth)
4. Train the ResidualTrainer (ML corrections)
5. Your system is fully calibrated!

---

## How It Works

### Architecture: "Physics-First, ML-Enhanced"

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA SOURCES                                          │
│  Open-Meteo (GHI, DNI, DHI) + Astronomy + Local Sensors         │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: PHYSICS ENGINE                                        │
│  POA Calculation → Temperature Correction → Power Output        │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3: LEARNING ENGINE                                       │
│  GeometryLearner + ShadowLearner + WeatherPrecision            │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 4: PREDICTION ENSEMBLE                                   │
│  Physics (Rule-Based) + TinyLSTM (ML) → Weighted Combination   │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 5: ANOMALY DETECTION                                     │
│  Frost Detection + Shadow Detection + Cloud Physics             │
└─────────────────────────────────────────────────────────────────┘
```

### Key Formulas

**Power Calculation (Physics-Based):**
```
P = (POA / 1000) × kWp × η_temp × η_system × (1 - shadow%) × seasonal_adj
```

**Temperature Correction:**
```
η_temp = 1 - 0.004 × (T_cell - 25°C)
```

**Ensemble Prediction:**
```
final = w_physics × physics_pred + w_ml × ml_residual
```

---

## System Lifecycle

| Phase | Duration | What Happens |
|-------|----------|--------------|
| **Fresh Install** | Day 0 | Physics engine active with defaults, ~70% accuracy |
| **Initial Learning** | Day 1-7 | Geometry converges, Ridge ML available |
| **Geometry Convergence** | Day 7-14 | ±3° tilt, ±8° azimuth accuracy, ~85-90% |
| **ML Activation** | Day 14-30 | TinyLSTM enabled, ensemble: 70% physics + 30% ML |
| **Production** | Day 30+ | Full calibration, 93-97% accuracy |

*With `bootstrap_physics_from_history`: Skip to "Production" phase immediately!*

---

## Requirements

- Home Assistant 2024.1.0 or newer
- Power sensor (Watts)
- Daily yield sensor (kWh, resets at midnight)
- ~50 MB disk space for data files
- ~100-150 MB RAM during ML training

### Optional but Recommended
- Lux sensor (improves shadow detection)
- Temperature sensor (improves efficiency calculation)
- Solar radiation sensor (W/m², best for precision)

---

## Battery Management (Optional - BETA)

Version 10.0.0 includes comprehensive battery support:

- **Energy Flow Tracking** - Solar to battery, battery to house, grid interactions
- **SOC Monitoring** - State of charge with historical trends
- **Charging Optimization** - When to charge from grid (with aWATTar prices)
- **Autarky Calculation** - True self-sufficiency with battery

Supported integrations: Anker Solix, Huawei Solar, Fronius

---

## Troubleshooting

### Check Logs
```bash
# Dedicated log file
/config/solar_forecast_ml/logs/solar_forecast_ml.log
```

### Common Issues

**Predictions too low?**
- Run `bootstrap_physics_from_history` to calibrate geometry
- Check if `solar_capacity` is set correctly

**ML not training?**
- Need 10+ samples for Ridge, 100+ for TinyLSTM
- Check `sensor.solar_forecast_ml_training_samples`

**Shadow detection wrong?**
- Ensure lux sensor is configured
- System needs clear-sky days to learn patterns

---

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to the `dev` branch.

---

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## Credits

- **Developer:** [Zara-Toorox](https://github.com/Zara-Toorox)
- **Architecture Design:** Developed in Python
- **Weather Data:** [Open-Meteo](https://open-meteo.com/) (free, no API key required)

---

## Support

- **Issues:** [GitHub Issues](https://github.com/Zara-Toorox/ha-solar-forecast-ml/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Zara-Toorox/ha-solar-forecast-ml/discussions)

---

*Made with solar power in mind*
