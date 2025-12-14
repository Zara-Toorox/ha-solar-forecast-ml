# Solar Forecast ML V12 "Sarpeidon" - 1st HA Full AI & ML Solar Forecast

[![Version](https://img.shields.io/badge/version-12.0.0-blue.svg)](https://github.com/Zara-Toorox/ha-solar-forecast-ml)
[![HACS](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://hacs.xyz/)
[![License](https://img.shields.io/badge/license-AGPL--3.0-green.svg)](LICENSE)

<a href='https://ko-fi.com/Q5Q41NMZZY' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://ko-fi.com/img/githubbutton_sm.svg' border='0' alt='Buy Me a Coffee ' /></a>

**Intelligent Solar Forecasting for Home Assistant with Physics-First ML Architecture**

Imagine your smart home not just reacting, but *predicting* - using machine learning that adapts daily to your unique solar installation. That's exactly what Solar Forecast ML does. With **Version 12.0.0 "Sarpeidon"** we introduce enhanced **Multi-Weather Blending**, **Per-Panel-Group Learning**, and **Hourly Correction Factors** for unprecedented accuracy.

---

## What Makes This Integration Special?

Unlike simple weather API services, Solar Forecast ML learns from your real data **and** uses actual solar physics. This means:

- **Day 1:** Accurate forecasts using real physics (POA irradiance, temperature correction)
- **Week 1:** System learns your panel geometry (tilt/azimuth) automatically
- **Week 2+:** ML enhances physics predictions with local patterns (shadows, obstructions)
- **Month 1+:** 93-97% accuracy with full Physics+ML ensemble

The integration adapts to your specific roof orientation, local weather conditions, shading patterns, and seasonal characteristics - all without manual configuration.

---

## New in Version 12.0.0 "Sarpeidon"

### Multi-Weather Source Blending
- **Open-Meteo + wttr.in** - Combines multiple weather sources for better cloud predictions
- **Adaptive Weight Learning** - System learns which weather source is more accurate for your location
- **Trigger-based Fetching** - Only queries secondary source when cloud cover > 50%

### Per-Panel-Group Learning (NEW)
- **Individual Energy Sensors** - Configure separate kWh sensors per panel group
- **Group-specific Efficiency** - Each panel group learns its own hourly efficiency factors
- **Shadow Detection per Group** - Identifies which panels are affected by shadows

### Hourly Correction Factors (NEW)
- **Hour-specific Weather Corrections** - Instead of daily averages, corrections are applied per hour
- **Morning/Afternoon Optimization** - Addresses systematic forecast biases at different times
- **7-day Rolling Learning** - Continuously improves based on recent performance

### Physics-First Architecture
The core architecture - a complete redesign from traditional solar forecasting:

- **PhysicsEngine** - Real solar physics calculations (POA, temperature correction, cell efficiency)
- **GeometryLearner** - Automatically learns your panel tilt & azimuth using Levenberg-Marquardt optimization
- **AI Neural Network** - Pure NumPy LSTM AI that captures temporal patterns
- **Residual Learning** - ML learns the difference between physics and reality
- **Weighted Ensemble** - Confidence-based blending of physics + ML predictions
- **Captain's LOG** - Monthly system health report (tilt, azimuth, seasonal tracking)

### Intelligent Detection Systems
- **Correlation-based Shadow Detection** - Distinguishes clouds from real obstructions
- **Physics-based Frost Detection** - Uses Magnus formula (dew point) for accurate frost warnings
- **Cloud Layer Physics** - Separate transmission models for low/mid/high clouds

### Unique Dashboard and statistics " SFML-STATS" BETA
- **First Home Assistant interactive Dashboard*** - Self hosted web-dashboard for local access 
- **Beautiful UX** - stunning dashboard with dark and white mode
- **Advanced statistic** - Download stunning png charts about your solar-system

### Unique Grid Price Monitoring "Grid Price Monitor" BETA
- **Germany & Austria only*** 
- **Triggers automaticaly advanced Grid Power Automations** - Automatic charging of you EV, Battery when price is low
- **Advanced statistic** - together with SFML-STATS
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
- **100% local AI and ML** - No ChatGPT, Claude, Gemini, Grok needed
- **No cloud services** - All data stays on your disk
- **No API keys required** - Uses free Open-Meteo API

---

## Companion Integrations

Solar Forecast ML works best with these companion integrations that extend its capabilities:

### SFM-Stats (Statistics Dashboard)

**What it does:** Provides comprehensive statistics, visualizations, and analytics for your solar production data. Creates beautiful Lovelace cards showing historical performance, accuracy metrics, and trend analysis.

**Features:**
- Historical production charts (daily, weekly, monthly)
- Forecast vs. actual comparison graphs
- ML model performance metrics
- Panel group efficiency visualization
- Weather correlation analysis

**Installation via Service:**
```yaml
service: solar_forecast_ml.install_sfm_stats
data: {}
```

After running this service:
1. Restart Home Assistant
2. Add the new cards to your dashboard
3. SFM-Stats entities will appear under `sensor.sfm_stats_*`

> **Platform Compatibility for SFM-Stats:**
>
> | Platform | Status | Notes |
> |----------|--------|-------|
> | x86_64 (Intel/AMD) | **Fully Supported** | Recommended |
> | Home Assistant OS (x86) | **Fully Supported** | Native installation |
> | Docker on x86 | **Fully Supported** | Standard HA container |
> | **Proxmox VE** | **Not Compatible** | NumPy/SciPy compilation issues in virtualized environment. CPU instructions not properly passed through. |
> | **Raspberry Pi (ARM)** | **Not Compatible** | Insufficient RAM and slower NumPy operations cause timeouts. |
> | **SBC / ARM Processors** | **Not Compatible** | Single-board computers (Odroid, Orange Pi, etc.) lack computational power for statistics processing. |
> | **32-bit Systems** | **Not Compatible** | Requires 64-bit architecture. |
>
> **Note:** These restrictions apply only to SFM-Stats, not to Solar Forecast ML itself. The main integration runs on all platforms including Raspberry Pi and ARM devices.

### Grid-Price Monitor

**What it does:** Integrates with dynamic electricity pricing (aWATTar, Tibber, Nordpool) to optimize your energy usage based on solar forecast and grid prices.

**Features:**
- Real-time electricity price tracking
- Optimal charging/discharging recommendations
- Price forecast integration with solar forecast
- Automation triggers for low-price periods
- Cost savings calculations

**Installation via Service:**
```yaml
service: solar_forecast_ml.install_grid_price_monitor
data:
  provider: "awattar"  # Options: awattar, tibber, nordpool
  country: "DE"        # Your country code
```

After running this service:
1. Restart Home Assistant
2. Configure your electricity provider credentials (if required)
3. Grid-Price entities will appear under `sensor.grid_price_*`

**Note:** Both companion integrations require Solar Forecast ML to be installed and configured first. They share data seamlessly without additional configuration.

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

## Services (Developer)

### ML Services (Developers only or on advice)
| Service | Description |
|---------|-------------|
| `force_retrain` | Retrain ML model with all available data |
| `reset_model` | Reset ML model to initial state |

### Bootstrap Services (Developers only or on advice)
| Service | Description |
|---------|-------------|
| `bootstrap_physics_from_history` | Train Physics+ML from HA history (up to 6 months) |
| `bootstrap_from_history` | Bootstrap pattern learning from history |

### Multi-Weather Services (Developers only or on advice)
| Service | Description |
|---------|-------------|
| `refresh_multi_weather` | Force refresh of all weather sources |
| `learn_weather_weights` | Trigger weight learning from today's data |

### Companion Integration Services
| Service | Description |
|---------|-------------|
| `install_sfm_stats` | Install the SFM-Stats companion integration |
| `install_grid_price_monitor` | Install the Grid-Price Monitor companion integration |

### Astronomy Services (Developers only or on advice)
| Service | Description |
|---------|-------------|
| `build_astronomy_cache` | Build comprehensive sun position cache |
| `refresh_cache_today` | Refresh astronomy data for current week |

### Testing & Maintenance (Developers only or on advice)
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
3. Click the three dots menu -> "Custom repositories"
4. Add: `https://github.com/Zara-Toorox/ha-solar-forecast-ml`
5. Select category: "Integration"
6. Install "Solar Forecast ML"
7. Restart Home Assistant
8. After Setup please wait 10-15 min and perform a 2nd Restart (needed to fill the caches)

### Manual Installation
1. Download the latest release
2. Copy `custom_components/solar_forecast_ml` to your `config/custom_components/`
3. Restart Home Assistant
4. After Setup please wait 10-15 min and perform a 2nd Restart (needed to fill the caches)

### Configuration
1. Go to Settings -> Devices & Services
2. Click "Add Integration"
3. Search for "Solar Forecast ML"
4. Configure:
   - **Power Sensor** (required): Current solar power in Watts
   - **Daily Yield Sensor** (required): Daily yield in kWh (must reset at midnight)
   - **System Capacity** (optional): Your system size in kWp
   - **Panel Groups** (optional): Format `Power(Wp)/Azimuth(deg)/Tilt(deg)/[EnergySensor]`
   - **Additional Sensors** (optional): Temperature, Lux, Radiation, etc.

---

## Panel Group Configuration (NEW in V12)

For systems with multiple panel orientations, you can configure individual groups:

**Format:** `Power(Wp)/Azimuth(deg)/Tilt(deg)/[EnergySensor]`

**Example:**
```
1425/180/9/sensor.pv_south_kwh_today
870/180/47/sensor.pv_roof_kwh_today
```

This creates:
- **Group 1:** 1425Wp, facing South (180deg), 9deg tilt, with its own energy sensor
- **Group 2:** 870Wp, facing South (180deg), 47deg tilt, with its own energy sensor

The system will learn individual efficiency factors for each group, improving overall accuracy.

---

## Quick Start After Installation

For immediate full accuracy, run the bootstrap service (Developers only or on advice):

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
+-------------------------------------------------------------------+
|  LAYER 1: DATA SOURCES                                             |
|  Open-Meteo + wttr.in (Blended) + Astronomy + Local Sensors        |
+-------------------------------------------------------------------+
|  LAYER 2: PHYSICS ENGINE                                           |
|  POA Calculation -> Temperature Correction -> Power Output         |
+-------------------------------------------------------------------+
|  LAYER 3: LEARNING ENGINE                                          |
|  GeometryLearner + ShadowLearner + WeatherPrecision + PanelGroups  |
+-------------------------------------------------------------------+
|  LAYER 4: PREDICTION ENSEMBLE                                      |
|  Physics (Rule-Based) + TinyLSTM (ML) -> Weighted Combination      |
+-------------------------------------------------------------------+
|  LAYER 5: ANOMALY DETECTION                                        |
|  Frost Detection + Shadow Detection + Cloud Physics                |
+-------------------------------------------------------------------+
```

---

## System Lifecycle

| Phase | Duration | What Happens |
|-------|----------|--------------|
| **Fresh Install** | Day 0 | Physics engine active with defaults, ~70% accuracy |
| **Initial Learning** | Day 1-7 | Geometry converges, Ridge ML available |
| **Geometry Convergence** | Day 7-14 | +/-3 deg tilt, +/-8 deg azimuth accuracy, ~85-90% |
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

**Note:** Solar Forecast ML runs on all platforms including Raspberry Pi and ARM devices. Only the optional SFM-Stats companion integration requires x86_64 architecture.

### Optional but Recommended
- Lux sensor (improves shadow detection)
- Temperature sensor (improves efficiency calculation)
- Solar radiation sensor (W/m2, best for precision)

---

## Troubleshooting

### Check Logs
```bash
# Dedicated log file
/config/solar_forecast_ml/logs/solar_forecast_ml.log
```

### Common Issues

**Predictions too low?**
- Check if you have used the correct parameters (kWp = the sum of all your installed panels)
- Check if `yield-sensor` is set correctly (kWh of your panels! must reset at midnight)
- Check if `power-sensor` is set correctly (W of your panels!)

**ML not training?**
- Need 10+ samples for Ridge, 100+ for TinyLSTM
- Check `sensor.solar_forecast_ml_training_samples`

**Shadow detection wrong?**
- Ensure lux sensor is configured
- System needs clear-sky days to learn patterns

**SFM-Stats not working on Proxmox/ARM?**
- The SFM-Stats companion integration is not compatible with Proxmox or ARM-based systems
- Solar Forecast ML itself works fine on these platforms
- For SFM-Stats, please use x86_64 hardware

---

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to the `dev` branch.

---

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## Credits

- **Developer:** [Zara-Toorox](https://github.com/Zara-Toorox)
- **Architecture Design:** Physics-First ML in Python , SFML-STATS in VUE, Grid Price Monitor in Python
- **Weather Data:** [Open-Meteo](https://open-meteo.com/) + [wttr.in](https://wttr.in/) (free, no API key required)

---

## Support

- **Issues:** [GitHub Issues](https://github.com/Zara-Toorox/ha-solar-forecast-ml/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Zara-Toorox/ha-solar-forecast-ml/discussions)

---

*Made with solar power in mind*
