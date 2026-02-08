# Solar Forecast ML V16.0.0 "Sarpeidon" — Full AI & DB-Version

### The 1st Hybrid-AI Solar Forecast for Home Assistant — 100% Local, 100% Private

[![Version](https://img.shields.io/badge/version-16.0.0-blue.svg)](https://github.com/Zara-Toorox/ha-solar-forecast-ml)
[![HACS](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://hacs.xyz/)
[![License](https://img.shields.io/badge/license-Proprietary%20Non--Commercial-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-x86__64%20%7C%20ARM%20%7C%20RPi-lightgrey.svg)]()

Fuel my late-night ideas with a coffee? I'd really appreciate it!
<a href='https://ko-fi.com/Q5Q41NMZZY' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://ko-fi.com/img/githubbutton_sm.svg' border='0' alt='Buy Me a Coffee' /></a>

[Website & Documentation](https://zara-toorox.github.io/index.html)

---

## ☀️ What Is Solar Forecast ML?

I didn't want another static API wrapper. So I built an **adaptive AI companion** that learns your installation from the ground up — your roof geometry, local shading patterns, microclimate, and inverter behavior. It doesn't just calculate; it evolves with your home.

The first **100% local AI** for Home Assistant — no ChatGPT, no Grok, no Gemini, no external AI of any kind. Three proprietary AI models, a local Machine Learning engine, and a full solar physics engine work in perfect synergy to deliver **3-day hourly forecasts** with up to **97% accuracy** after calibration. Powered by a fully transactional SQL database — slim, quick, and reliable. Everything runs on your hardware. No cloud. No subscriptions. No data leaves your network.

**Your smart home stops reacting. It starts anticipating.**

---

## 🧠 Why I Built Solar Forecast ML

Most integrations rely on rigid formulas or static API calls. I wanted something better.

- **Proactive, not reactive** — I wanted my smart home to anticipate the sun, not just respond to it.
- **Precision Physics** — I combined atmospheric transmittance modeling with real-world sensor feedback.
- **100% Privacy** — I made sure every single byte of Machine Learning stays on your local machine.

---

## 🏗️ Quad-Engine Architecture

I designed the system to orchestrate four independent intelligence layers, delivering a 3-day hourly forecast with surgical precision — all 100% local, zero outbound traffic:

| Engine | Purpose | What It Does |
|--------|---------|--------------|
| **Weather AI** | Intelligence Layer | Multi-source blending & custom rolling bias correction across 5 weather services. |
| **Physics AI** | Geometric Baseline | Learns your panel tilt, azimuth, and real-world shading obstacles. Calibrates atmospheric transmittance and cell efficiency. |
| **ML Backbone** | Rapid Adaptation | High-stability model tuned for early-phase learning. Delivers predictions from as few as 10 samples. |
| **Hybrid AI** | Temporal Logic | Local Attention Model that captures complex seasonal and time-of-day patterns across 24-hour sequences. |

All four engines are blended using an **Adaptive Ensemble** — learned confidence weights that continuously evaluate which model performs best under which conditions. A disagreement detector prevents overconfident predictions when models diverge.

---

## ⚡ Key Capabilities

### 🔮 Forecasting
- **72-hour horizon** — granular hourly forecasts for today, tomorrow, and the day after
- **Dynamic sunrise scheduling** — forecast generation adapts daily to actual sunrise time
- **Adaptive re-forecast** — automatic midday correction when conditions shift significantly
- **Per-panel-group predictions** — individual forecasts for each panel orientation
- **Confidence scoring** — every prediction includes a learned confidence metric

### 🧠 AI & Machine Learning
- **Quad-Engine ensemble** — Weather AI, Physics AI, ML Backbone, and Hybrid AI with Attention — all local
- **28 engineered features** — time, weather, radiation, astronomy, historical production, panel geometry
- **Automatic training** — daily model updates at end of day
- **Auto-optimization** — the system runs hyperparameter grid searches for you
- **Feature importance analysis** — understand which inputs drive your predictions

### 🌦️ Weather Intelligence
- **5-source weather blending** — Open-Meteo, Bright Sky, Pirate Weather, wttr.in, ECMWF
- **3-stage correction** — daily rolling factors, hourly corrections, condition-specific adjustments
- **Expert weight learning** — each weather source earns trust based on historical accuracy
- **Fog & visibility detection** — identifies fog and haze conditions that standard weather apps miss
- **Cloud trend analysis** — 1-hour and 3-hour cloudiness trend sensors with volatility tracking

### 🕵️ Detection & Protection
- **Shadow mapping** — distinguishes between a passing cloud and a tree's physical shadow
- **Shadow pattern learning** — learns recurring shading from trees, buildings, terrain
- **Frost & fog warning** — physics-based dew point calculation for edge-case weather
- **MPPT clipping detection** — the AI recognizes when your inverter limits production so it doesn't pollute training data
- **Full zero-export & battery support** — detects zero-export limiting and battery-full curtailment, adjusts learning accordingly
- **Learning filter** — automatically excludes anomalous data from AI training

### 🛠️ Technical Depth
- **Transactional SQL** — slim, async SQLite engine for maximum reliability
- **Automatic backups** — daily learning data backups with 30-day retention
- **Self-healing** — automatic detection and repair of corrupted data
- **Crash recovery** — seamless restoration after system failures
- **Learning data protection** — survives Home Assistant backup restores

### 📐 Panel Group Support
- **Up to 4 panel groups** — different orientations, tilts, capacities
- **Individual efficiency learning** — each group learns its own correction factors
- **Per-group energy sensors** — optional dedicated kWh sensors per group
- **Group-specific AI** — multi-output model with per-group predictions

### ❄️ Seasonal Intelligence — "The Winter Edge"
I built a dedicated Winter Mode (Nov–Feb) into the core. It automatically adjusts for low sun angles and atmospheric clearness, ensuring your energy management stays reliable when the sun is rarest.

- **Seasonal correction factors** — monthly calibration that adapts to changing sun angles
- **DNI ratio tracking** — 7-day rolling atmospheric clearness monitoring

---

## 📊 Sensors

### Forecast
| Sensor | Description |
|--------|-------------|
| `solar_forecast_ml_today` | Today's forecast (kWh) |
| `solar_forecast_ml_tomorrow` | Tomorrow's forecast (kWh) |
| `solar_forecast_ml_day_after_tomorrow` | Day after tomorrow (kWh) |
| `solar_forecast_ml_next_hour` | Next hour prediction (kWh) |
| `solar_forecast_ml_peak_production_hour` | Best production hour today |

### Production
| Sensor | Description |
|--------|-------------|
| `solar_forecast_ml_production_time` | Production hours (start/end/duration) |
| `solar_forecast_ml_max_peak_today` | Peak power today (W) |
| `solar_forecast_ml_max_peak_all_time` | All-time peak power (W) |
| `solar_forecast_ml_expected_daily_production` | Daily production target |

### Statistics
| Sensor | Description |
|--------|-------------|
| `solar_forecast_ml_average_yield` | Cumulative average yield |
| `solar_forecast_ml_average_yield_7_days` | 7-day rolling average |
| `solar_forecast_ml_average_yield_30_days` | 30-day rolling average |
| `solar_forecast_ml_monthly_yield` | Current month total |
| `solar_forecast_ml_weekly_yield` | Current week total |

### AI & Diagnostics
| Sensor | Description |
|--------|-------------|
| `solar_forecast_ml_model_state` | Active prediction model (AI / Rule-Based) |
| `solar_forecast_ml_model_accuracy` | Current prediction accuracy (%) |
| `solar_forecast_ml_ai_rmse` | Model quality (Excellent / Very Good / Good / Fair) |
| `solar_forecast_ml_training_samples` | Available training samples |
| `solar_forecast_ml_ml_metrics` | MAE, RMSE, R² metrics |

### Shadow & Weather
| Sensor | Description |
|--------|-------------|
| `solar_forecast_ml_shadow_current` | Current shadow level (Clear / Light / Moderate / Heavy) |
| `solar_forecast_ml_performance_loss` | Shadow-related production loss (%) |
| `solar_forecast_ml_cloudiness_trend_1h` | 1-hour cloud trend |
| `solar_forecast_ml_cloudiness_trend_3h` | 3-hour cloud trend |
| `solar_forecast_ml_cloudiness_volatility` | Weather stability index |

---

## 📈 System Lifecycle

| Phase | Timeline | What Happens |
|-------|----------|--------------|
| **Fresh Install** | Day 0 | Physics engine active, ~70% accuracy |
| **Early Learning** | Day 1–7 | ML Backbone activates, geometry converges |
| **Calibration** | Day 7–14 | Tilt/azimuth learned to +/-3°, ~85–90% accuracy |
| **AI Activation** | Day 14–30 | Hybrid AI enabled, full ensemble blending begins |
| **Production** | Day 30+ | Full calibration, 93–97% accuracy |

> **Shortcut:** Use `bootstrap_physics_from_history` to skip to Production phase immediately using your existing Home Assistant history data (up to 6 months).

---

## 🚀 Installation

### HACS (Recommended)
1. Open HACS > Integrations
2. Three-dot menu > Custom repositories
3. Add `https://github.com/Zara-Toorox/ha-solar-forecast-ml` (Category: Integration)
4. Install "Solar Forecast ML"
5. Restart Home Assistant
6. Wait 10–15 minutes, then restart once more (cache initialization)

### Manual
1. Download the latest release
2. Copy `custom_components/solar_forecast_ml` to your `config/custom_components/`
3. Restart Home Assistant
4. Wait 10–15 minutes, then restart once more

### Configuration
1. Settings > Devices & Services > Add Integration > "Solar Forecast ML"
2. Configure:
   - **Power Sensor** (required) — current solar power in Watts
   - **Daily Yield Sensor** (required) — daily yield in kWh (must reset at midnight)
   - **System Capacity** (optional) — total system size in kWp
   - **Panel Groups** (optional) — `Power(Wp)/Azimuth(°)/Tilt(°)/[EnergySensor]`
   - **Additional Sensors** (optional) — temperature, lux, radiation, humidity, wind

### Panel Group Configuration

For systems with multiple panel orientations:

**Format:** `Power(Wp)/Azimuth(°)/Tilt(°)/[EnergySensor]`

**Example:**
```
1425/180/9/sensor.pv_south_kwh_today
870/180/47/sensor.pv_roof_kwh_today
```

This creates two independent groups, each with its own physics calculations, AI predictions, and efficiency learning.

---

## 🔧 Services

### Core
| Service | Description |
|---------|-------------|
| `force_forecast` | Trigger immediate forecast update |
| `send_daily_briefing` | Send formatted solar forecast notification |
| `install_extras` | Install/update all companion modules |
| `borg_mode` | Complete system reset — deletes all learned data |

### AI
| Service | Description |
|---------|-------------|
| `retrain_ai_model` | Force AI model retraining |
| `reset_ai_model` | Reset AI to initial state |
| `analyze_feature_importance` | Show which features impact predictions most |
| `run_grid_search` | Hyperparameter optimization |

### Bootstrap
| Service | Description |
|---------|-------------|
| `bootstrap_physics_from_history` | Train from HA history (up to 6 months) |
| `bootstrap_from_history` | Bootstrap pattern learning from history |

### Weather
| Service | Description |
|---------|-------------|
| `refresh_multi_weather` | Force refresh all weather sources |
| `run_weather_correction` | Rebuild corrected weather forecast |

### Maintenance
| Service | Description |
|---------|-------------|
| `run_all_day_end_tasks` | Manual trigger for end-of-day workflow |
| `test_morning_routine` | Analyze morning routine (read-only) |
| `build_astronomy_cache` | Rebuild sun position cache |

---

## 🧩 Companion Modules

Solar Forecast ML includes optional companion integrations, installed via the `install_extras` service:

### SFML Stats — The First Complete Solar & Energy Dashboard for Home Assistant
A stunning, fully self-hosted dashboard built exclusively for solar and energy tracking. Real-time energy flows, historical charts, forecast vs. actual comparisons, cost tracking, and multi-string monitoring — all in one place. **x86_64 only.**

| Module | Description | Platform |
|--------|-------------|----------|
| **SFML Stats** | Complete solar & energy tracking dashboard | x86_64 only |
| **Grid Price Monitor** | Dynamic electricity spot prices (DE/AT) | All |

---

## 📋 Requirements

- Home Assistant 2024.1.0+
- Power sensor (Watts) + Daily yield sensor (kWh, midnight reset)
- ~50 MB disk space
- ~100–200 MB RAM during AI training

**Optional but recommended:** Lux sensor, temperature sensor, solar radiation sensor (W/m²)

**Runs on all platforms** — x86_64, ARM, Raspberry Pi. Only SFML Stats requires x86_64.

---

## 🔒 Privacy

**100% Local — Zero Cloud Dependencies**

Solar Forecast ML runs entirely on your hardware. No ChatGPT, no Claude, no Gemini, no Grok — no cloud-based AI of any kind. No data leaves your network. No API keys required for core functionality (uses free Open-Meteo weather data). The entire intelligence resides on your machine.

---

## ❓ Troubleshooting

**Predictions too low?**
- Verify kWp matches total installed panel power
- Check that yield sensor is in kWh and resets at midnight
- Check that power sensor is in Watts

**AI not training?**
- Check `sensor.solar_forecast_ml_training_samples` — minimum 10 samples needed
- Allow 3–7 days for initial data collection

**Shadow detection inaccurate?**
- Configure a lux sensor for best results
- System needs clear-sky days to establish baseline patterns

**Log location:**
```
/config/solar_forecast_ml/logs/solar_forecast_ml.log
```

---

## 📄 License

Proprietary Non-Commercial License — free for personal, educational, and non-commercial use. See [LICENSE](LICENSE) for details.

---

## 👤 Credits

**Developer:** [Zara-Toorox](https://github.com/Zara-Toorox)

**Support:**
- [GitHub Issues](https://github.com/Zara-Toorox/ha-solar-forecast-ml/issues)
- [GitHub Discussions](https://github.com/Zara-Toorox/ha-solar-forecast-ml/discussions)

---

*Made with ☀️ & late-night passion in Germany*
