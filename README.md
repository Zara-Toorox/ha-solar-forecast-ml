# Solar Forecast ML V12.10.0 "Sarpeidon" - 1st HA Full Hybrid-AI Solar Forecast

[![Version](https://img.shields.io/badge/version-12.10.0-blue.svg)](https://github.com/Zara-Toorox/ha-solar-forecast-ml)
[![HACS](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://hacs.xyz/)
[![License](https://img.shields.io/badge/license-Proprietary%20Non--Commercial-green.svg)](LICENSE)


Fuel my late-night ideas with a coffee? I'd really appreciate it ♡
<a href='https://ko-fi.com/Q5Q41NMZZY' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://ko-fi.com/img/githubbutton_sm.svg' border='0' alt='Buy Me a Coffee ' /></a>

Further information:
🌐 **[Website & Documentation](https://zara-toorox.github.io/index.html)**

## Solar Forecast ML – Sarpeidon Edition
*No static API calls. No rigid, off-the-shelf models.*

Experience an adaptive hybrid AI ecosystem powered by three proprietary, locally-hosted AI engines working in perfect synergy. This system continuously molds itself to the unique characteristics of your installation: from complex shading patterns and local topography to the specific thermal behavior of your inverter under peak loads.

Solar Forecast ML is not just another forecasting tool—it is a sophisticated AI Solar Companion for Home Assistant that doesn't just deliver data, but actively learns and evolves with your home.

With Version 12.8.8 "Sarpeidon," solar optimization enters a new dimension:

**Triple-AI Core Architecture:** Three specialized, locally-deployed AI models collaborate to synchronize energy generation, weather dynamics, and system behavior—all without a single external server request.

**Fog/Visibility Intelligence:** Detects fog and haze with significantly higher precision than standard weather apps, adjusting atmospheric transmittance in real-time.

**5-Weather-Expert Blending:** Five independent weather services compete for accuracy. Every hour, the AI dynamically decides which model currently provides the highest validity for your specific location.

**Precipitation Sentinel:** Provides targeted alerts for potential yield losses due to snow or heavy rain before the front even arrives.

**Transmittance-Based Learning:** The system masters the actual atmospheric transparency of your local environment—becoming more precise with every passing day.

**The Result:** Your smart home stops reacting. It starts anticipating.

**100% Local – 100% Privacy.** This system operates entirely without ChatGPT, Gemini, Claude, or other cloud-based LLMs. The entire intelligence resides on your hardware. Zero data leaves your network. No exploitative API fees, no cloud dependency—just pure physics and high-performance machine learning on your own machine.

*Welcome to the future of solar optimization. The sun doesn't just shine anymore—it is now predicted.*

<p align="center">
  <img src="docs/images/progose_graph.png" alt="Solar Forecast ML Prognose" width="600">
</p>

---

## Features in V12.10.0 "Sarpeidon"

### 5-Weather-Expert Blending
- **Multi-Source Weather Data** - Combines 5 independent weather services for optimal accuracy
- **Adaptive Weight Learning** - System learns which weather source is most accurate for your location
- **Automatic Failover** - Seamless switching between sources if one fails

### Fog & Visibility Detection
- **FOG Detection** - Visibility below 1 km triggers FOG cloud type
- **FOG_LIGHT Detection** - Visibility 1-5 km triggers light fog classification
- **Transmittance-Based Learning** - VisibilityLearner determines optimal visibility data source
- **Automatic Transmittance Reduction** - Fog conditions reduce solar transmittance calculations

### Precipitation Notifications
- **Rain Alerts** - Persistent notification when rain is expected
- **Snow Alerts** - Temperature-based snow detection (below 2°C)
- **Configurable Threshold** - Rain detection at 0.5 mm/h
- **Home Assistant Integration** - Notifications via HA persistent notification system

### Learning Filter System
- **Intelligent Data Filtering** - Excludes anomalous hours from AI training
- **Weather Alert Detection** - Automatically flags unexpected weather events
- **Inverter Clipping Detection** - Excludes hardware-limited data points
- **Daily Learning Protection** - Skips training if >25% of data is flagged

### Per-Panel-Group Learning
- **Individual Energy Sensors** - Configure separate kWh sensors per panel group
- **Group-specific Efficiency** - Each panel group learns its own hourly efficiency factors
- **Shadow Detection per Group** - Identifies which panels are affected by shadows

### Hourly Correction Factors
- **Hour-specific Weather Corrections** - Corrections applied per hour instead of daily averages
- **Morning/Afternoon Optimization** - Addresses systematic forecast biases at different times
- **7-day Rolling Learning** - Continuously improves based on recent performance

### Physics-First Hybrid-AI Architecture
- **PhysicsEngine** - Real solar physics calculations (POA, temperature correction, cell efficiency)
- **GeometryLearner** - Automatically learns panel tilt & azimuth using Levenberg-Marquardt optimization
- **AI Neural Network** - Pure NumPy AI that captures temporal patterns
- **Residual Learning** - AI learns the difference between physics and reality
- **Weighted Ensemble** - Confidence-based blending of physics + AI predictions
- **Captain's LOG** - Monthly system health report (tilt, azimuth, seasonal tracking)

### Intelligent Detection Systems
- **Correlation-based Shadow Detection** - Distinguishes clouds from real obstructions
- **Physics-based Frost Detection** - Uses Magnus formula (dew point) for accurate frost warnings
- **Cloud Layer Physics** - Separate transmission models for low/mid/high clouds

### SFML Stats Lite
- **Universal Dashboard** - Works on ALL platforms including Raspberry Pi and ARM
- **Real-time Energy Flow** - Visualize solar, battery, grid, and house consumption
- **Cost Tracking** - Fixed or dynamic pricing support
- **Multi-String Support** - Track up to 4 panel groups individually
- **Automated Reports** - Weekly and monthly chart generation

### SFML Stats (x86_64 only)
- **Interactive Web Dashboard** - Self-hosted dashboard for local access
- **Beautiful UX** - Dark and light mode support
- **Advanced Statistics** - Download charts about your solar system
- **Clothing Recommendation** - Weather-based clothing suggestions

### Grid Price Monitor (DE/AT)
- **Dynamic Spot Prices** - Real-time electricity price monitoring
- **Automation Triggers** - Automatic EV/battery charging when prices are low
- **Price Forecasts** - Today's and tomorrow's hourly prices

---

## Core Features

### Forecasting
- **3-day yield forecasts** (today, tomorrow, day after) with Physics+AI optimization
- **Hourly predictions** for detailed planning
- **Physics-based calculations** using GHI, DNI, DHI from Open-Meteo
- **Automatic panel geometry learning** - no manual tilt/azimuth configuration needed
- **Temperature-corrected efficiency** - accounts for cell temperature effects

### Hybrid-AI Engine
- **Adaptive Algorithm Selection:**
  - < 100 samples: Ridge Regression (fast, stable)
  - 100+ samples: AI Neural Network (+5-10% accuracy)
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
- **100% local Hybrid-AI** - No ChatGPT, Claude, Gemini, Grok needed
- **No cloud services** - All data stays on your disk
- **No API keys required** - Uses free Open-Meteo API

---

## Extra Modules (v6.4.4)

Solar Forecast ML includes 5 companion integrations that extend its capabilities. All modules are installed via the `install_extras` service and share data seamlessly.

| Module | Description | Platform |
|--------|-------------|----------|
| **SFML Stats** | Advanced statistics dashboard with charts | x86_64 only |
| **SFML Stats Lite** | Lightweight dashboard for all platforms | All platforms |
| **Grid Price Monitor** | Dynamic electricity price monitoring (DE/AT) | All platforms |
| **ML Weather** | Corrected weather data based on learning | All platforms |
| **ESC Easy Sensor Creation** | Create custom sensors without YAML | All platforms |



### SFML Stats (Advanced Statistics Dashboard)

<p align="center">
  <img src="docs/images/energy_flow.png" alt="SFML Stats Energy Flow" width="500">
</p>

**What it does:** Provides comprehensive statistics, visualizations, and analytics for your solar production data. A powerful dashboard that visualizes solar production, battery storage, grid consumption, and energy costs in real-time. Includes the new **Clothing Recommendation** feature.

**Features:**
- Real-time energy flow visualization (solar, battery, grid, house)
- Historical production charts (daily, weekly, monthly)
- Forecast vs. actual comparison graphs
- Cost tracking with fixed or dynamic pricing
- Multi-string support (up to 4 panel groups)
- Automated weekly and monthly report generation
- Weather overlay on charts
- Dark and light theme support
- Clothing recommendations based on weather

**Installation:**
1. Go to Settings > Devices & Services > Add Integration > SFML Stats
2. Configure your sensors (all optional - integration works with partial config)
3. Access the dashboard at `http://YOUR_HA:8123/api/sfml_stats/dashboard`

> **Platform Compatibility for SFML Stats:**
>
> | Platform | Status | Notes |
> |----------|--------|-------|
> | x86_64 (Intel/AMD) | **Fully Supported** | Recommended |
> | Home Assistant OS (x86) | **Fully Supported** | Native installation |
> | Docker on x86 | **Fully Supported** | Standard HA container |
> | **Proxmox VE** | **Try on your own if your aware what you are doing** | no support by the developer |
> | **Raspberry Pi (ARM)** | **not supported** | Use SFML Stats Lite instead |
> | **SBC / ARM Processors** | **not supported** | Use SFML Stats Lite instead |

---

### SFML Stats Lite (For Raspberry Pi & ARM)

**What it does:** A lightweight version of SFML Stats designed specifically for Raspberry Pi and ARM devices. Provides the same energy monitoring dashboard without the heavy computational requirements.

**Features:**
- Real-time energy flow visualization (solar, battery, grid, house)
- Cost tracking with fixed or dynamic pricing
- Multi-string support (up to 4 panel groups)
- Automated weekly and monthly report generation
- Weather overlay on charts
- Dark and light theme support

**Installation:**
1. Go to Settings > Devices & Services > Add Integration > SFML Stats Lite
2. Configure your sensors (all optional - integration works with partial config)
3. Access the dashboard at `http://YOUR_HA:8123/api/sfml_stats_lite/dashboard`

> **Platform Compatibility for SFML Stats Lite:**
>
> | Platform | Status |
> |----------|--------|
> | Raspberry Pi (ARM) | **Fully Supported** |
> | All ARM devices | **Fully Supported** |
> | Proxmox VE | **Fully Supported** |
> | x86_64 (Intel/AMD) | **Fully Supported** |
> | Home Assistant OS | **Fully Supported** |
> | Docker | **Fully Supported** |

### Grid-Price Monitor

**What it does:** Monitors dynamic electricity spot prices from aWATTar (Germany & Austria) and provides smart automation triggers for optimal energy usage. Perfect for charging EVs, batteries, or running high-power appliances when electricity is cheapest.

**Features:**
- **Real-time Spot Prices** - Current and next hour prices in ct/kWh
- **Price Forecasts** - Today's and tomorrow's hourly prices (available from ~14:00)
- **Cheapest/Most Expensive Hour** - Automatically identifies optimal times
- **Binary Sensor for Automations** - `binary_sensor.cheap_energy` triggers when price is below threshold
- **Configurable Price Components** - Grid fees, taxes, VAT, provider markup
- **Calibration Mode** - Match your actual electricity bill
- **Battery Tracking** - Track how much energy was charged from grid

**Sensors:**
| Sensor | Description |
|--------|-------------|
| `sensor.grid_price_monitor_spot_price` | Current spot price (ct/kWh) |
| `sensor.grid_price_monitor_total_price` | Total price incl. fees & taxes |
| `sensor.grid_price_monitor_cheapest_hour_today` | Cheapest hour today |
| `sensor.grid_price_monitor_average_price_today` | Average price today |
| `binary_sensor.grid_price_monitor_cheap_energy` | ON when price < threshold |

**Installation:**
1. Go to Settings > Devices & Services > Add Integration > Grid Price Monitor
2. Configure:
   - **Country:** Germany (DE) or Austria (AT)
   - **VAT Rate:** 19% (DE) or 20% (AT)
   - **Grid Fee:** Your grid operator fee (ct/kWh)
   - **Taxes & Fees:** Additional taxes (ct/kWh)
   - **Max Price Threshold:** Price below which `cheap_energy` is ON

**Example Automation:**
```yaml
automation:
  - alias: "Charge EV when electricity is cheap"
    trigger:
      - platform: state
        entity_id: binary_sensor.grid_price_monitor_cheap_energy
        to: "on"
    action:
      - service: switch.turn_on
        target:
          entity_id: switch.ev_charger
```

> **Availability:** Currently supports **aWATTar** API (Germany & Austria). No API key required - uses free public market data.

---

### ML Weather (Corrected Weather Data BETA 3)

**What it does:** Provides corrected weather data based on Solar Forecast ML's learning algorithms. The integration learns from your actual solar production to correct weather forecast inaccuracies for your specific location.

**Features:**
- **Learned Weather Corrections** - Uses historical data to improve weather accuracy
- **Location-specific Adjustments** - Corrections tailored to your exact location
- **Cloud Cover Optimization** - Improves cloud predictions based on actual production
- **Seamless Integration** - Works automatically with Solar Forecast ML data

**Sensors:**
| Sensor | Description |
|--------|-------------|
| `weather.ml_weather` | Corrected weather entity with improved accuracy |

**Installation:**
1. Go to Settings > Devices & Services > Add Integration > ML Weather

> **Note:** ML Weather requires Solar Forecast ML to have at least 7 days of learning data for meaningful corrections.

---

### ESC Easy Sensor Creation

**What it does:** Create custom sensors, binary sensors, and switches directly from the Home Assistant UI - no YAML required. Perfect for users who want to extend their setup without coding.

**Features:**
- **No YAML Required** - Create sensors through the UI
- **Multiple Sensor Types** - Sensors, binary sensors, and switches
- **Template Support** - Use Jinja2 templates for calculations
- **History Integration** - New sensors automatically record history
- **Easy Management** - Edit or delete sensors through the options flow

**Supported Entity Types:**
| Type | Description |
|------|-------------|
| `sensor` | Numeric or text sensors with templates |
| `binary_sensor` | On/off sensors based on conditions |
| `switch` | Controllable switches with custom logic |

**Installation:**
1. Go to Settings > Devices & Services > Add Integration > ESC Easy Sensor Creation
2. Use the options flow to create new sensors

**Example Use Cases:**
- Calculate solar self-consumption percentage
- Create a "good production day" binary sensor
- Build combined energy sensors from multiple sources

---

> **Note:** All extra modules require Solar Forecast ML to be installed and configured first. They share data seamlessly without additional configuration.

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

### AI & Analytics Sensors
| Sensor | Description |
|--------|-------------|
| `sensor.solar_forecast_ml_model_state` | AI model training status |
| `sensor.solar_forecast_ml_model_accuracy` | Current prediction accuracy (%) |
| `sensor.solar_forecast_ml_training_samples` | Number of training samples |
| `sensor.solar_forecast_ml_shadow_current` | Current shadow detection |
| `sensor.solar_forecast_ml_performance_loss` | Shadow-related losses (%) |


---

## Services (Developer)

### AI Services (Developers only or on advice)
| Service | Description |
|---------|-------------|
| `force_retrain` | Retrain AI model with all available data |
| `reset_model` | Reset AI model to initial state |

### Bootstrap Services (Developers only or on advice)
| Service | Description |
|---------|-------------|
| `bootstrap_physics_from_history` | Train Physics+AI from HA history (up to 6 months) |
| `bootstrap_from_history` | Bootstrap pattern learning from history |

### Multi-Weather Services (Developers only or on advice)
| Service | Description |
|---------|-------------|
| `refresh_multi_weather` | Force refresh of all weather sources |
| `learn_weather_weights` | Trigger weight learning from today's data |

### Companion Integration Services
| Service | Description |
|---------|-------------|
| `install_extras` | **Main installer!** Installs/updates ALL companion modules (SFML Stats Lite, SFM-Stats, Grid Price Monitor). Run via Developer Tools > Services |

### Reset & Recovery Services
| Service | Description |
|---------|-------------|
| `borg_mode` | **Complete system reset** - Deletes ALL learned data and starts fresh. Use when: predictions are very inaccurate after 3-4 days, you changed sensors, or migrated from an older installation with corrupted data |

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


> Then restart Home Assistant to apply the updates.

> This deletes ALL learned data and lets the system relearn from scratch. Recommended after major version updates or sensor changes.

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

## Panel Group Configuration

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
4. Train the ResidualTrainer (AI corrections)
5. Your system is fully calibrated!

---

## How It Works

### Architecture: "Physics-First"

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
|  Physics (Rule-Based) + AI Neural Network -> Weighted Combination  |
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
| **Initial Learning** | Day 1-7 | Geometry converges, Ridge AI available |
| **Geometry Convergence** | Day 7-14 | +/-3 deg tilt, +/-8 deg azimuth accuracy, ~85-90% |
| **AI Activation** | Day 14-30 | AI Neural Network enabled, ensemble: 70% physics + 30% AI |
| **Production** | Day 30+ | Full calibration, 93-97% accuracy |

*With `bootstrap_physics_from_history`: Skip to "Production" phase immediately!*

---

## Requirements

- Home Assistant 2024.1.0 or newer
- Power sensor (Watts)
- Daily yield sensor (kWh, resets at midnight)
- ~50 MB disk space for data files
- ~100-150 MB RAM during AI training

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

**AI not training?**
- Need more time
- Check `sensor.solar_forecast_ml_training_samples`

**Shadow detection wrong?**
- Ensure lux sensor is configured
- System needs clear-sky days to learn patterns

**SFM-Stats not working on Proxmox/ARM?**
- The SFM-Stats companion integration is not compatible with ARM-based systems
- Solar Forecast ML itself works fine on these platforms
- For SFM-Stats, please use x86_64 hardware


- This deletes ALL learned data and starts fresh
- After running, wait 3-7 days for the system to relearn your installation
- Consider running `bootstrap_physics_from_history` afterwards to speed up learning

---

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to the `dev` branch.

---

## License

This project is licensed under a **Proprietary Non-Commercial License** - see the [LICENSE](LICENSE) file for details. Free for personal, educational, and non-commercial use.

---

## Credits

- **Developer:** [Zara-Toorox](https://github.com/Zara-Toorox)
- **Architecture Design:** Physics-First Hybrid-AI in Python, SFML-STATS in VUE, Grid Price Monitor in Python


---

## Support

- **Issues:** [GitHub Issues](https://github.com/Zara-Toorox/ha-solar-forecast-ml/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Zara-Toorox/ha-solar-forecast-ml/discussions)

---

*Made with ☀️ & late-night passion in Germany*
