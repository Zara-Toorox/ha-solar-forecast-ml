# Solar Forecast ML - Version 6 "Risa" 🌟

**First machine learning solar forecasting for Home Assistant**

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub release](https://img.shields.io/github/release/Zara-Toorox/ha-solar-forecast-ml.svg)](https://github.com/Zara-Toorox/ha-solar-forecast-ml/releases)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Imagine your smart home not just reacting, but **predicting** - using machine learning that adapts daily to your unique solar installation. That's exactly what **Solar Forecast ML** does, and with Version 6 "Risa" we reach a new dimension of precision.

## What Makes This Integration Special?

Unlike simple weather API services, Solar Forecast ML **learns from your real data**. This means: The longer the integration runs, the more accurate the predictions become - adapted to your specific roof orientation, local weather conditions, shading, and even seasonal characteristics.

### Core: Self-Learning ML Model

The integration collects daily weather data and compares it with your actual solar yield. Through advanced machine learning algorithms, it recognizes patterns and continuously optimizes its predictions. The result? **Precise forecasts that are better than any weather service alone.**

## Features at a Glance

### 📊 Core Functions
- **Daily yield forecasts** for today and tomorrow with automatic ML optimization
- **Hourly predictions** for detailed planning (optional)
- **Intelligent weather integration** with automatic fallback on DWD failures
- **Self-learning system** - gets more precise every day without your intervention
- **Real-time accuracy tracking** - see the precision of your forecasts

### 🎯 Production Analytics
- **Peak hour detection** - when do you produce the most?
- **Production time calculation** - optimized from sunrise to sunset
- **Average yield** over any time period
- **Self-sufficiency calculation** - how independent are you really?
- **Deviation analysis** - yesterday vs. forecast in direct comparison

### 🔬 Diagnostics & Transparency (20+ Sensors!)
- **ML model status** - see your AI model's training state
- **Timestamp tracking** - last updates, next scheduled updates
- **Feature vector insights** - understand which factors influence predictions
- **External sensor integration** with live updates for:
  - Temperature, wind, rain
  - UV radiation, brightness (lux)
  - Any other weather data
- **Error logging** with intelligent error handling

### 🎮 Interactive Controls
Four powerful buttons for complete control:
- **Update forecast** - manual update on demand
- **Retrain model** - complete re-training with all historical data
- **Calibrate model** - fine-tuning for quick adjustments
- **Clear errors** - reset on problems

### 🔔 Smart Notifications
Stay informed with configurable notifications for:
- Integration startup
- New forecast updates
- Ongoing ML training processes
- Successfully completed training

## Installation

### HACS (Recommended)

1. Open HACS in your Home Assistant
2. Go to "Integrations"
3. Click the three dots in the top right corner
4. Select "Custom repositories"
5. Add this repository URL: `https://github.com/Zara-Toorox/ha-solar-forecast-ml`
6. Select category "Integration"
7. Click "Add"
8. Find "Solar Forecast ML" in HACS and click "Download"
9. Restart Home Assistant

### Manual Installation

1. Download the latest release from GitHub
2. Extract the `solar_forecast_ml` folder
3. Copy it to your `custom_components` directory
4. Restart Home Assistant

## Configuration

### Quick Setup in 3 Steps

1. Go to **Settings** → **Devices & Services**
2. Click **Add Integration**
3. Search for **Solar Forecast ML**

### Required Sensors
- **Weather entity** (e.g., DWD)
- **Power sensor** in watts
- **Daily yield** in kWh (must reset daily at midnight!)

### Optional Sensors
- Plant capacity (kWp) for more precise calculations
- External sensors (temperature, wind, rain, UV, lux)
- Consumption sensor for self-sufficiency calculation

The integration starts immediately with rule-based forecasts and begins ML training after 7 days. From day 10, the full AI model runs.

## Practical Automation Examples

### Example 1: Smart Washing Machine
```yaml
automation:
  - alias: "Washing machine on high solar forecast"
    trigger:
      - platform: numeric_state
        entity_id: sensor.solar_forecast_today
        above: 15
    action:
      - service: notify.mobile_app
        data:
          message: "Lots of sun today! Perfect for washing machine & dryer ({{ states('sensor.solar_forecast_today') }} kWh expected)"
```

### Example 2: Dynamic Battery Control
```yaml
automation:
  - alias: "Battery charging strategy"
    trigger:
      - platform: time
        at: "06:00:00"
    action:
      - choose:
          - conditions:
              - condition: numeric_state
                entity_id: sensor.solar_forecast_today
                below: 10
            sequence:
              - service: script.charge_battery_from_grid
          - conditions:
              - condition: numeric_state
                entity_id: sensor.solar_forecast_today
                above: 20
            sequence:
              - service: script.battery_full_solar_optimization
```

### Example 3: Intelligent Energy Management
```yaml
automation:
  - alias: "Energy-intensive devices at peak time"
    trigger:
      - platform: state
        entity_id: sensor.peak_production_hour
    condition:
      - condition: template
        value_template: "{{ now().hour == states('sensor.peak_production_hour') | int }}"
    action:
      - service: switch.turn_on
        target:
          entity_id: 
            - switch.pool_pump
            - switch.air_conditioning
```

### Example 4: Self-Sufficiency Monitoring
```yaml
automation:
  - alias: "Self-sufficiency report"
    trigger:
      - platform: time
        at: "20:00:00"
    action:
      - service: notify.telegram
        data:
          message: >
            📊 Daily energy balance:
            Yield: {{ states('sensor.solar_yield_today') }} kWh
            Self-sufficiency: {{ states('sensor.autarkie') }}%
            Forecast deviation: {{ states('sensor.yesterday_deviation') }}%
            Production time: {{ states('sensor.production_time') }}
```

## Why Version 6 "Risa"?

For all Trekkies: Risa is the legendary vacation planet from Star Trek - known for perfect weather, endless sunny days, and pure relaxation. Just as Risa stands for constant sunshine, this version delivers reliable and precise solar forecasts. Version 6 brings:

- Refined ML algorithms with better convergence
- Optimized feature weighting
- More robust error handling
- Extended diagnostic sensors
- Improved weather service integration with intelligent retry system

## Technical Highlights

- **Modular architecture** - cleanly separated manager classes
- **Async/await everywhere** - no blocking operations
- **Comprehensive error handling** - the integration runs stable, even with sensor failures
- **Data persistence** - all ML models and historical data remain after restarts
- **Performance optimized** - intelligent caching and minimal update intervals
- **Best practice compliant** - follows all Home Assistant development guidelines

## Available Sensors

### Main Forecast Sensors
- `sensor.solar_forecast_heute` - Today's forecast
- `sensor.solar_forecast_morgen` - Tomorrow's forecast
- `sensor.peak_production_hour` - Hour with highest production
- `sensor.production_time` - Active production time
- `sensor.average_yield` - Average yield
- `sensor.autarkie` - Self-sufficiency percentage

### Accuracy & Diagnostics
- `sensor.solar_accuracy` - Model accuracy
- `sensor.yesterday_deviation` - Yesterday's forecast deviation
- `sensor.diagnostic_status` - Overall status
- `sensor.ml_model_status` - ML model state
- `sensor.last_coordinator_update` - Last update timestamp
- `sensor.update_age` - Time since last update
- `sensor.last_ml_training` - Last training timestamp
- `sensor.next_scheduled_update` - Next scheduled update

### External Sensors (if configured)
- `sensor.external_temperature` - External temperature
- `sensor.external_wind` - Wind speed
- `sensor.external_rain` - Rain status
- `sensor.external_uv` - UV index
- `sensor.external_lux` - Brightness

### ML Diagnostics (when diagnostic mode enabled)
- Feature vectors, weights, model metrics, training samples, and more

## Services

### `solar_forecast_ml.update_forecast`
Manually trigger a forecast update

### `solar_forecast_ml.retrain_model`
Retrain the ML model with all historical data

### `solar_forecast_ml.calibrate_model`
Fine-tune the model for quick adjustments

### `solar_forecast_ml.clear_errors`
Clear error log

## Configuration Options

Access via **Settings** → **Devices & Services** → **Solar Forecast ML** → **Configure**

- **Update interval** - How often forecasts update (300-86400 seconds)
- **Diagnostic mode** - Enable extended diagnostic sensors
- **Hourly forecasts** - Enable hourly predictions
- **Notifications** - Configure startup, forecast, learning notifications

## Troubleshooting

### Integration not starting
- Check that all required sensors exist
- Verify daily yield sensor resets at midnight
- Check Home Assistant logs for errors

### Inaccurate predictions
- Wait at least 10 days for ML training
- Ensure weather entity provides reliable data
- Check that power and yield sensors report correct values
- Use "Retrain Model" button to rebuild ML model

### Missing sensors
- Enable diagnostic mode in configuration
- Restart Home Assistant after configuration changes
- Check entity registry for disabled sensors

## Support & Contributing

This integration is **open source** (AGPL-3.0) and lives from the community. Bug reports, feature requests, and pull requests are welcome!

- **Issues**: [GitHub Issues](https://github.com/Zara-Toorox/ha-solar-forecast-ml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Zara-Toorox/ha-solar-forecast-ml/discussions)

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Credits

Developed with ☀️ by [Zara-Toorox](https://github.com/Zara-Toorox)

---

**Ready for the future of solar forecasting?** Install Solar Forecast ML today and let AI work for you! ☀️🤖